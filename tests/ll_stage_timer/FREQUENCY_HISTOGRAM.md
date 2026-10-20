# 远端专家计数的寄存器频率直方图

分支 `kernel/ll-g4-shared-recv`，改动基线 `b89a868`。
只替换 `CountByVectorRange` 中压紧路由后的逐元素 Scalar 直方图累加。
本组 `BuildLocalTasks` 生成稳定 q 的循环、小输入 `CountByExpert`、
发送器和静态共享接收器均保持原实现，不带入动态接收分支的优化。

## 单段直接路径

现有 `GatherMask` 给出 `selectedCount`，调用侧只判断一次是否不超过 32768：

- 不超过：调用 `CountFrequency<true>`，uint16 寄存器清零后直接统计，
  最后扩宽为 uint32 写出。该模板没有分段循环、uint32 汇总寄存器清零、
  或各段结果相加；不逐路由检查溢出。
- 超过：调用 `CountFrequency<false>`，每段最多 32768 个路由元素。
  每段重新清零 uint16 直方图，扩宽后在 uint32 寄存器中累加。
- 空选择也走单段路径，正确写出零频数；不读取空输入。

32768 是整段的输入元素数上限。即使所有元素命中同一个专家，uint16
单桶值也不超过 32768。分段大小为 64 的倍数，保证后续 int32 load 起点
仍对齐。最后不足 64 项的向量使用真实元素掩码，不能把 padding 算入频数。

## 专家范围、UB 和同步

每次处理最多 128 个专家。先以 uint32 减去窗口起点，再在窄化之前比较
`relative < bins`；低于起点的输入通过无符号下溢被排除。数据和 b32 比较
掩码同步 pack 两次到 uint8/b8，然后调用 `asc_frequency_histogram_bin0`。
因此不会混淆专家 0 与 256，也支持跨 128/256 边界和更大的专家区间。

输入直接使用原 `selected` UB，其分配按 256B 对齐，最后一个完整向量 load
不会越过该分配。输出继续使用原 histogram UB；写回按实际专家数屏蔽，
只有需要高半区时才写 output+64。没有新增 UB/GM 缓冲或更改 Host 预算。
保留 GatherMask 后的 V→S 完成等待，在 VF 之后增加 V→S 后才读取结果；
原 count 记录发布和就绪协议不变。每专家写 count 记录的 Scalar 循环仍保留，
被移除的是每条压紧路由的 histogram GetValue/SetValue。

核心身份和阶段 schema7 不变。全 rank 安装同一提交的完整 wheel，
无需仅因该改动重建 SHMEM；不要更改生产架构来适配不支持该指令的设备。
缺少 C API 头文件或接口时应报告编译错误，不静默回退掩盖部署不匹配。

## 验证和 A/B

接口依据 asc-devkit `9.2.0@4d9fe40502b3423a344d570fd1a8be343cfca4a5`：
[频率直方图](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/api/SIMD-API/c_api/reg_compute/reg_histogram/asc_frequency_histogram.md)、
[pack](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/api/SIMD-API/c_api/reg_compute/reg_permute_sel/asc_pack_to_low.md)、
[unpack](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/api/SIMD-API/c_api/reg_compute/reg_permute_sel/asc_unpack.md)。
本次使用公开 `c_api/asc_simd.h`，未复制 devkit 实现或升级工程 SDK。

从工程根目录运行：

```bash
python -m pytest tests/elastic/test_ll_frequency_histogram.py \
  tests/elastic/test_ll_group_local_send.py \
  tests/ll_stage_timer/test_segment_source.py -q
```

边界模型覆盖空输入、63/64/65、32767/32768/32769、65535/65536/70001，
以及 1/63/64/65/127/128/129/257 个专家。另提供实际生产 VF 头文件的
C++ intrinsic mock，检查 predicate packing、扩宽、掩码写回及单热桶超过
65535 的汇总。mock 不验证 Bisheng 语法、寄存器分配或设备流水。
本地无 C++/CANN 编译器，原生 mock 与设备验证不能标为通过。
本次完整本地回归：448 passed、44 skipped、9 deselected。
跳过项包括缺 C++ 编译器的原生检查和一项 POSIX Bash 检查；
9 项未选为当前 Windows 环境不适用的构建脚本检查。

目标机按原构建流程全 rank 重装 wheel；先运行正确性，再与 `b89a868`
用同 seed、shape、S/R/G 和 timer OFF 对比。需覆盖少量/大量压紧路由、
热点专家、零命中和重复调用。不要只依据 B256 示例判断收益：
小分片的 VF 调用与转换成本可能高于原 Scalar 循环。诊断阶段仍记入
COUNT_COMPUTE，未增加逐向量或逐段在线计时。
