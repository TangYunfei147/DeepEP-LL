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

## 当前R0前缀候选

kernel/ll-g4-group-compact现基于S3/ac876ba，撤销两轮compact改动。
身份g4-r0-prefix、协议17000+S。SEND_COUNTS仅R核，前缀仅R0生产，所有核读同一表。
固定epoch槽为R0 magic槽+32，先完成表写入再发布；消费者不清共享flag。
dispatch UB预算230KiB、combine独立预算不变。接收表向量化须处理32B对齐及V_S依赖。
当前指南tests/ll_stage_timer/R0_PREFIX.md。

非去重M<=32专用路径：S发所有route后退出，R仅计数/前缀/接收，不生成本组描述符。
去重路径不变，协议18000+S；Host/Device按M选择实际接收核数，不能仅按B判断。

## 本核q表候选

当前kernel/ll-g4-local-q-table从f7c7848派生；协议18000+S、收发调度不变。
qWork持久保存本片q，qSub播种私有专家直方图，内部量化只能复用qDst。
去重按token分片，非去重按route分片，R本组q不变。指南tests/ll_stage_timer/LOCAL_Q_TABLE.md。

q前缀播种现复用CountFrequency VF，V_S后只做本片Scalar更新；空片段跳过。
保持18000+S及收发协议，候选ST检查low_latency_q_seed_vf。A/B基线af46122。
