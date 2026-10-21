# LL 编译维护约束

修改 LL kernel、模板、打点或构建配置前，先阅读：

- `tests/ll_probe/KERNEL_MAP.md`
- `docs/BUILD_DISPATCH_SPLIT.md`（两次同类编译劣化的处理经验与验收要求）
- 编译环境见 `docs/A3_COMPILE_VALIDATION.md`；远端操作仍遵循上级 AGENTS.md 的授权范围。

## 必须保持的结构

- `csrc/kernels/dispatch/low_latency.cpp` 只做 Host 选择；不得重新包含完整 kernel 实现或在一个 global 中展开全部模式。
- 每种输入/量化模式使用独立 `variants/*.cpp`，公共设备入口保持在 `dispatch_variant_kernel.h`。禁止 unity/jumbo 合并这些编译单元。
- 新增模式或模板维度时，先评估实例化数量和最大函数规模；同步修改入口清单、Host/Device 选择校验、CMake、OFF 编译覆盖和 profiler 精确匹配。
- 保持全部支持模式、生产 O2、目标架构、打点语义及同步顺序。不得用裁剪功能、关闭打点、降低优化级别来冒充编译问题的修复。
- Host/Device 启动接口变化必须拒绝混装；CPU 编译不能替代 Bisheng 的 Host/AICore 限定检查。设备回调须显式设备限定，避免普通 lambda 隐式成为 Host 函数。

## 验证与记录

- 运行 `python -m pytest tests/ll_stage_timer/test_dispatch_variants.py tests/ll_stage_timer/test_segment_source.py tests/elastic/test_ll_st_contract.py -q`。
- 影响模板展开、打点规模或构建图的变更，须在获授权编译环境测量 ON/OFF 的独立空对象目录，并验证完整 ON wheel 的链接。只改文档或 CPU 检查不必重复设备编译。
- 对照必须固定工具链、SDK、模式集合、优化级别、核数和并行度；分开记录冷编译、暖构建、SDK准备和打包时间。失败、跳过、审批未执行均须如实记录。
- 修改缓存逻辑时另查无改动重建与头文件修改后的依赖重编；不得只凭源码摘要跳过真实构建系统的依赖检查。
- 交付前更新 `docs/BUILD_DISPATCH_SPLIT.md` 的实测证据。向其他分支移植时检查其入口是否已拆分，不得因 NDR 分支已有修复而假定其他分支也已修复。

## 当前 A/B 分支
四版策略、统一schema12口径和切换方式见docs/OVERLAP_AB.md。
仅通过ll_overlap_stage.h的提交常量选择阶段；不得让四个版本各自修改打点范围。
保留Host/device stage校验、各rank一致性及ST对安装wheel/checkout的校验。

Stage5是额外单提交实验：R前置计数/前缀发布，随后有界ready检查与异步prefix搬入，
诊断schema13与历史四版schema12不同，见docs/PREFIX_PROGRESS.md。
事件5只在接收开始前供prefix使用，禁止和在途输入6/7或接收lane重叠复用。

schema14仅改诊断，见docs/PACKET_TIMING.md。dispatch observe的ID12/13、16..21、60..63
只测前两次，coverage=first_two_only，不能当全量累计或用于阶段占比。
包级/主阶段和包数仍全量；旧schema13保持离线兼容。不得为测单包DMA完成增加同步。
S3独立基线kernel/ll-g4-s3-baseline精确ac876ba，不带当前S5/打点变更。
