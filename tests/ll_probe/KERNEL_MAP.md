# 当前 MTE LL reference 路径

本分支 `kernel/ll-g4-systemcycle-profile` 保留恢复基线的MTE业务算法，
打点和解析参考 `kernel/ll-ndr-metadata-cores@c3d8640`，只做LL dispatch。
当前唯一记录格式为MTE共享接收适配schema6；旧记录器和兼容代码已删除。

| 用途 | 源码 |
|---|---|
| Host可选reference tensor、形状与ABI检查 | `csrc/ops/low_latency.cpp` |
| 7模式Host选择 | `csrc/kernels/dispatch/low_latency.cpp` |
| 独立设备入口 | `csrc/kernels/dispatch/low_latency/variants/*.cpp`、`dispatch_variant_kernel.h` |
| 主调度/窗口轮转/计数/发送/接收 | `low_latency/dispatch_server_dedup_kernel.h`、`low_latency/device/*.h` |
| reference字典及记录器 | `low_latency/common/ll_reference_profile_layout.h`、`ll_reference_profile_device.h` |
| S/R配置及UB预算 | `low_latency/common/ll_core_config.h`、`ll_overlap_layout.h`、`cmake/CoreSplit.cmake` |
| Python捕获与整数解析 | `deep_ep/utils/ll_reference_profile.py`、`ll_reference_profile_parse.py` |
| 采样 | `tests/elastic/test_ep_low_latency.py` |
| 构建/运行/分析/Excel | `tests/ll_reference_profile/one_click.py`、`analyze.py`、`export_xlsx.py`、`report_tables.py` |
| 业务模型与新格式回归 | `tests/ll_kernel`、`tests/ll_reference_profile` |

没有需要手改的生成snapshot。不能将参考NDR的M核、RDMA或新SDK带入当前业务。
当前固定56核，S/R默认28/28、目标入口显式30/26；所有核在去重路径可参与接收。
R0生成共享前缀，S发外组源包、R发本组源包，receiver保持原ready位图/双批探测/relay优先。
q表、原一包前瞻、packet layout、计数发布/清理和lane生命周期保持。

参考主阶段映射在 `tests/ll_reference_profile/README.md` 中说明；
主点写GM，接收日志只用剩余UB并在w21后导出，不能加业务屏障凑参考阶段名称。
