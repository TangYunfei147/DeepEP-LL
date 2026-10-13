# LL实际AIV核数与统一分核

本分支的LL dispatch/combine不再固定启动64核。Host参考MegaMoE，调用`PlatformAscendCManager::GetInstance()->GetCoreNumAiv()`获取平台AIV数量；存在当前设备时，再通过`aclrtGetDeviceInfo(..., ACL_DEV_ATTR_VECTOR_CORE_NUM, ...)`查询运行时可用数，取两者较小值作为本次纯AIV启动数量N。

保留原来的**发送:接收=1:1**，即发送核`0..N/2-1`、接收核`N/2..N-1`。AIV核数和EP设备rank数是两回事：单机EP16仍是16张卡，每张卡各自按自己的实际核数分工。

## 查询与启动契约

- `csrc/runtime/ll_core_config.hpp`集中查询，生产绑定把N显式传给Host布局检查和dispatch/combine启动参数；设备侧通过`GetBlockNum()`取得同一个N。
- LL是纯AIV kernel，不使用MegaMoE混合AIC/AIV kernel的`CalcTschBlockDim`或`GetBlockNum()*2`作为生产LL的启动公式。
- 当前控制区容量支持**16至72之间的偶数N**。其他值明确报错，不猜测核数、不静默退回64核。下限16保证接收半区至少包含8个通知核。
- 仅做Host计划且未绑定设备时，查询结果是平台计划值，返回`device=-1`、`runtime_device_queried=false`。实际通信启动必须有当前设备并成功核验其可用核数。
- 同一个Session首次使用LL时固定N；后续检测到变化会报错，要求重新创建Session。正式LL测试在初始化控制进程组后收集各rank配置，不一致则在启动LL前拒绝。
- Host/device新增核心协议版本校验，旧设备库没有相应导出符号或版本不匹配时不能继续启动。所有rank必须重新安装完整wheel，不能只更新Python或替换部分库。
- CPU布局辅助函数保留明确的64核计划默认值供无硬件测试使用；**生产路径始终显式传入查询值**，该默认值不决定运行时启动核数。

典型配置：

| 查询并选择的N | 发送核 | 接收核 | combine通知核 |
|---:|---|---|---|
| 56 | 0..27 | 28..55 | 48..55 |
| 60 | 0..29 | 30..59 | 52..59 |
| 64 | 0..31 | 32..63 | 56..63 |
| 72 | 0..35 | 36..71 | 64..71 |

不应再用旧的`DISPATCH_SERVER_DEDUP_SEND_CORE_NUM`或`COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SEND_CORE_NUM`调整比例；分工已统一为运行时N/2。

## 计数和同步如何适配

**dispatch：** B小于现有阈值512时由全部N核统计专家计数，否则由接收半区统计。专家数、路由数不能整除参与核数时，原均分函数分配余数，确保每项恰好由一个核处理。接收计数记录和payload接收行都按N/2个接收核分配。

前缀矩阵维度、前缀副本数、跨核发布次数与等待目标均使用实际接收核数。有效参与核数为`min(E,N/2)`，未拥有记录的接收核不承担总数发布，但仍按原协议读取最终前缀。变长ready读取的UB缓冲保留向量归约所需尾部空间。

**combine：** 本地准备阶段按N核划分，DIRECT回传按N/2个发送核划分，partial与最终源token聚合按N/2个接收核划分；相应UB任务表与阶段布局也使用这个数量。

最后8个已启动核负责通知，每核通知本机16-rank组内两个peer。每个通知核只等待本卡实际N核的准备标志，不再等待未启动的核。每个接收核仍等待组内16个rank的ready，这个16是通信组大小，不是AIV数量。

**预留容量不等于参与数量：**

- magic控制前缀为72个槽，共36KiB，所有rank地址一致；实际只有N个核使用各自的槽。
- dispatch跨卡计数槽的起始偏移因此统一固定，本卡前缀矩阵再按实际接收核数布局。
- combine控制区移至窗口内1MiB，payload仍从2MiB开始，双窗口步长仍为4GiB。
- combine准备区8行，每行预留72个32B标志（2304B）；ready区起点为18432B，每行容纳16个peer的32B标志。
- 通知向预留的36个接收行写入ready，实际仅N/2个接收核等待并清理自己的行；未使用行不参与同步，也不能在Session中途更改核数后直接启用。
- Host布局及UB预算同步检查。数据包中的32B/512B、向量lane数等不是核数，不作机械替换。

## 当前诊断与审计

上一轮LL_DISPATCH_DEBUG_*与LL_DLOG核内打印已删除，当前不再输出这些阶段标签。保留Host的`--trace-ops`和启动时的`[LL cores]`配置输出。剩余固定上限、通知布局耦合及输入前提见[固定值审计](HARDCODE_AUDIT.md)。

## 更新、运行和结果记录

按[LL16指南](LL16.md)加载本机CANN、使用仓内SDK并在实际测试的Python环境重建：

```bash
source scripts/project_env.sh
export DEEPEP_NPU_ARCH=Ascend950
python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_probe/build_profile_bridge.sh
```

独立四层可执行程序另用`bash tests/ll_probe/four_layers/build.sh`重建。metadata提取探针只启动实际发送半区数量的AIV，并按同一数量校验分片/checksum；它仍仅在单张设备上运行，不代表建立了EP16通信域。metadata的event_diagnostics.csv中保留sender_cores/total_ll_cores说明行。

可在目标机先查询已安装wheel（示例只选择本机设备0）：

```bash
python3 -I - <<'PY'
import torch, torch_npu
from deep_ep._native import require_native
torch.npu.set_device(0)
native = require_native('LL core query')
print(native.low_latency_core_protocol)
print(dict(native.low_latency_core_config()))
PY
```

原LL测试会在每rank启动时输出`[LL cores] rank=...`，并把各rank配置保存在报告参数中。每个性能case记录`aiv_cores/send_cores/recv_cores`，已有开销分析按配置分组，因此不同核数不会混在同一拟合组中。10次预热、100次采集、Exp+scalar all_reduce及最后50条task统计口径保持不变。

先运行目标机正确性检查，观察每rank报告的N和kernel入口是否一致，再启用性能采样。A5仍只允许设备0/1，不能在A5启动完整EP16或伪造rank。

## 动态核数改造时的历史验证范围

以下记录包含当时尚未删除的调试打印烟测；当前打印已移除，最新审计见HARDCODE_AUDIT.md。

- A5的平台接口报告28 AIC、56 AIV；当前设备0运行时查询报告56 AIV，已安装native配置为56/28/28。
- 设备0独立启动烟测按查询值启动56个block，全部写回一次、预留槽guard保持不变；调试输出为核0和28。这不是完整LL的跨核等待或跨卡正确性验收。
- CPU使用实际生产分片方法和combine发布/通知/清理方法，覆盖16/32/48/56/60/64/72核，包含28/30等非32核接收组、重复轮次与未使用ready行；另覆盖平台/设备查询取较小值、查询失败、非法核数、布局与UB预算。
- 471项完整CPU回归通过，1项因源树测试未加载编译wheel而跳过；最终核心/启动/路径回归43项通过。完整生产wheel、全部调试宏开启的dispatch、profile探针和独立四层探针构建通过。
- A5设备0的28发送核metadata，以及设备0/1的双rank通信正确性和小Exp profiler验证通过：`results/reference-profile-fipv3c1z`；32逻辑分区metadata通过：`results/ll-core-config/metadata32`。
- 原始日志、生成快照、wheel及烟测文件保留在工程内，不提交Git。目标机原生EP16/EP32、实际互联与性能仍由用户验收；不能据上述结果宣称此前目标机卡死已完全解决。
