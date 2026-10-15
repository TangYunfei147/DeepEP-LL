# 单 AIV MTE2 深度诊断

此目录部署在仓库 `tests/ll_probe/mte2_depth`。实验在设备 0、1 上逐卡、顺序、单进程执行，每次 kernel 只启动一个 AIV（blockDim=1），不建立通信域。A5 仅验证实验可运行性和本机观察，不代表用户目标机性能。

本实验扫描单 AIV 独立 GM→UB 搬运请求数，对比 batch（连续发射后统一等待）与 serial（每次搬运后等待）。必须确认 UB 区间互不重叠，源地址合法，最终等待正确，并对输出逐项校验。请求数超过某值的发射耗时变化只能提示有效流水窗口；请求在发射期间可能已经完成，因此无拐点不证明不存在硬件限额。

建议覆盖 N=0、1、16、17、32，并在 12～20 加密扫描；同时改变 chunk 大小及 GM stride。N=0 观察空载计时和启动开销，不能未经论证直接从其他 case 相减。batch/serial 对照、编译器生成源码与 ASM 审查是解释结果的必要步骤；高层 DataCopy 调用数不自动等于底层事务数。MTE2 请求窗口也不能替代 MTE3、RDMA QP 或硬件 RDMA OTS。

## 原始数据与统计

在已配置 CANN 环境的仓库中执行：

```bash
bash tests/ll_probe/mte2_depth/build.sh
bash tests/ll_probe/mte2_depth/run_a5.sh \
  --devices 0,1 --requests 0,1,4,8,12,13,14,15,16,17,18,19,20,24,32 \
  --chunk 128,512,2048,4096 --stride 65536 --mode batch,serial \
  --warmup 10 --iters 100 --tail 50 \
  --output-dir results/mte2-depth-run-name --timeout 1800
```

每项列表用逗号分隔；runner 逐卡启动独立 host 进程，每个进程仅初始化一次 ACL，自动执行分析。输出目录必须是仓库内未存在的新目录，保存逐卡 CSV/log、manifest 和 analysis。`run_a5.sh` 固定四个设备可见性变量为 `0,1`；日志及缓存重定向由 `scripts/project_env.sh` 提供。计时失败或校验失败不自动重试。

每个 case 默认预热 10 次、采集 100 次，预热不写原始 CSV。原始列为：

```text
device,mode,requests,chunk_bytes,stride_bytes,iteration,issue_ticks,drain_ticks,total_ticks,event_us,verified
```

同一输入目录只放一次实验的原始 CSV，避免重复 iteration。case 由设备、模式、请求数、chunk 和 stride 唯一标识。`issue_ticks` 是发射区间，`drain_ticks` 是最终等待区间，`total_ticks` 是 kernel 内总计时区间；准确边界以 kernel 源码为准。serial 的 issue 区间包含每次请求后的等待，batch 不包含；两者的 drain 均含最终事件同步。Event 微秒是独立的完整 kernel Event 时间，包含 UB 初始化及计时后整个 UB 的 MTE3 导出，不能代替 task 时间或搬运专属时间，也不能与 ticks 混为同一口径。没有明确 GetSystemCycle 频率时不将 ticks 转为微秒。

部署后，从任意当前目录运行：

```bash
python /path/to/DeepEP-LL/tests/ll_probe/mte2_depth/analyze.py \
  --input-dir results/mte2-depth-run-name \
  --output-dir results/mte2-depth-run-name/analysis
```

相对输入输出路径通过脚本位置解析为仓库根路径，不依赖调用目录。工具按每个 case 独立取 iteration 最后 50 条，分别输出 ticks 与 Event 的 P50/P90/P99 到 `summary.csv`、`summary.json`。所有原始数据（包括前 50 条）都参与 verified 检查；任何失败返回非零退出码。覆盖检查提示缺失的推荐 case，不强制完整扫描。12～20 的局部中位数斜率只作为 diagnostic，不自动宣称精确 depth。

此实验使用原始计时，不包含 Exp/scalar all_reduce，是专项微架构诊断，**不是正式 LL 性能协议**。正式 LL 采样仍遵循 `tests/ll_probe/PROFILING.md`。原始数据、构建产物和分析输出不提交 Git；源代码、脚本及指南提交到 DeepEP-LL 分支。

## 缓存与指令审计

默认构建使用 SDK 默认 L2 hint；每次 H2D 更新内容保证校验新鲜度，但不等于 HBM 冷读。
补充对照可用 `MTE2_DEPTH_BYPASS=1 bash tests/ll_probe/mte2_depth/build.sh`，
随后换新 output-dir 运行；这表示显式设置 `CACHE_MODE_DISABLE`，不是保证所有层级缓存都被清空。
两个构建的数据分别保存，不能混合统计。build/identity.json 保存源文件、二进制、反汇编 SHA256 和实际编译器版本，runner 将其复制到 manifest。

构建保留 `build/kernel.dis` 与中间源码，自动审查 N=1/15/16/17 的无等待发射窗口。
此编译器把 N=32 的部分代码共享到其他分支，不能仅按文本相邻时间戳统计；需要结合实际控制流审查。
每个请求还有标量地址/描述符计算，因此当前探针不能保证把队列瞬间填满。

## 有界 Vector 门控对照

为了在发射期间保持请求未完成，可选择先在独立32KiB UB上提交有限次 Vector Duplicate，
随后让 MTE2 等待 V_MTE2 事件1。Vector 工作自主完成，不依赖后续 Scalar 发射或 MTE2
释放，不构成循环等待。Scalar 可在 MTE2 被门控时继续提交 DataCopy。
128KiB copy UB与32KiB gate UB互不重叠；计时前的guard初始化仍先排空。

```bash
MTE2_DEPTH_BYPASS=1 MTE2_DEPTH_GATE_REPEATS=64 \
  bash tests/ll_probe/mte2_depth/build.sh
bash tests/ll_probe/mte2_depth/run_a5.sh \
  --requests 0,1,8,12,13,14,15,16,17,18,19,20,24,32 \
  --chunk 128 --mode batch --output-dir results/mte2-gate64-new
```

GATE_REPEATS仅接受0/16/32/64/128，0关闭。必须先看N=0的drain，确认gate剩余时间
明显大于无gate的N=32发射时间，再比较不同gate长度的边界。提交Vector工作本身也可能
触发其队列背压，因此次数翻倍不保证剩余gate时间翻倍。
等待事件可能占据某一级队列项；第16次copy阻塞不能直接解释成“只能有15个MTE2请求”
或“总线OTS=16”。当前A5观察与解释边界见 [VALIDATION.md](VALIDATION.md)。
