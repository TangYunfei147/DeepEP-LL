# LL 编译维护约束

## SHMEM 可选字段与清单校验（2026-10-26）

- 用户要求适配`hns_1825_pi_on_chip`所有值，覆盖旧仅int0/1并要求HNS标签的规则。此字段作为opaque JSON元数据，缺省及任意JSON值均原样接受，不强转bool、不推导RDMA能力；独立`rdma_backend`校验及require_rdma的HNS要求保留。
- MTE复用只忽略该可选元数据及已认可的backend能力差异，保留共同构建输入、revision和source_patches约束；wheel配置和native cache记录完整SDK build对象，不能丢字段。SDK文件覆盖及哈希校验继续严格执行。
- include/src/lib清单覆盖错误不等于字段不兼容，当前报错给出缺失总数、目录分布和示例；不得删除manifest后重新hash旧库冒充完整构建。清单不完整仍显式从锁定源码重建，不覆盖正在被其他任务使用的SDK。恢复步骤见docs/SHMEM_MASTER_UPDATE.md。

## 配对接收与小场景启动诊断（2026-10-25，最新实验授权）

- 用户要求继续分离count记录归属与payload消费归属，并允许临时删除B>32场景来定位移植性能差距。独立候选`cb59-pair`（macro3/23028/cb59-pair-recv）在EP16小direct中由相邻两个R核交换精确dense区间后均分payload；count生产、专家计数写者、source发送均保持stream/cb59。其他shape使用cb59协议，未改默认选择。
- pair只在配对内部平衡，不假定热点退化已证明由负载不均引起。单消费者mailbox用旧cumsumFlag矩阵，不占用仍在途的sum矩阵；publicprefix先于header发布，整32B清理。packed handoff只用一个TBuf，协议8+handoff1+tasks2+probe4+lane1=16，190KiB总预算同步Host；本片0payload可借入partner，按最终pair.Count判断空接收，q不得从consumer边界重新编号。
- 临时`cb59-small`（macro4/24028/cb59-small-direct）仅接受1<=B<=M<=32，原合法EP范围/七dtype/O2保留，超范围明确报错无fallback。编译期关闭dedup/relay并固定全核count；原cb59/current大场景保留。本次用户授权覆盖下方“不裁剪功能”条款，仅限独立临时small选择，不作为生产功能或性能结论。
- 新采集统一schema16：原14个点后追加TILING_DONE/PIPE_READY/OBJECT_READY/MAGIC_DONE，18点、512B每核、导出256B，内部链接sparse16拒混装。dispatch四点必需，combine不补不存在的细分点；schema15按原14点离线解析。无新增打点屏障，不能用OFF总时间减ON内部跨度估算launch/prologue。
- cb59生成文件只经generate_cb59_replay.py更新MAGIC_DONE；small生成目录只能经generate_cb59_small.py修改，逆归一化逐token/同步校验。四实现cb59/cb59-small/cb59-stream/cb59-pair都须用同一提交重编完整OFF/ON，不能复用schema15轮次wheel。SDK复用已验证版本。步骤见CB59_PAIR.md与CB59_SMALL.md；仍只本地检查，不SSH/27，不宣称设备性能已验证。

## cb59 分片接收候选（2026-10-25）

- 用户要求先做接收调度与路由协议重设计。独立选择 `DEEPEP_LL_DISPATCH_IMPL=cb59-stream`，原 `cb59` 六生成算法文件保持不变；新算法在 `low_latency/cb59_stream/` 手工维护。固定 G16/S28R28/P0，协议22028，身份cb59-stream-recv。七模式仍各自单TU、O2，无新增设备启动。
- 仅EP16/M<=32/B<=32启用分片接收，其他shape用原cb59协议，实际路径写effective_dispatch_impl。stream预算不足必须Host拒绝，不能静默回退或按单核路由切换。编译manifest及native/全rank继续严格区分实现。
- 原source counts/Sender保留；R按连续count-record拥有全部q，只等本片及前序片段sum确定dense偏移。sum仅发布上三角，消费者唯一清自己列；专家末record owner结合跨片tail输出专家计数。保留固定packet槽、公开prefix及dense行序/combine元数据。
- stream协议和接收器共享持久UB池，不Reset、不在首次接收前新增PIPE_ALL。协议8+任务2+探测4+lane1共15个TBuf，异步源独立保留到原counts.Finish。Host/device预算共用ll_stream_receive.h，190KiB不变，不能沿用总行数/R的任务上界。
- stream的PREFIX_READY表示本片输出偏移就绪，不表示全rank前缀完成；OFF完整task是性能依据。当前仅CPU/原生mock验证，无CANN/NPU结论；热点可能因按记录归属而负载偏斜。先全部构建四wheel，再正确性/normal/hotspot串行对照，步骤见tests/ll_stage_timer/CB59_STREAM.md。

## 当前 cb59 对照移植（2026-10-25）

- 用户要求把快基线cb59bff99ecd39d27ef57788c5e6fd0009e4e636移植当前分支并用打点定位差距。编译选择DEEPEP_LL_DISPATCH_IMPL=current|cb59，默认current；每个模式TU只编入一个算法，仍保留全部七种模式。
- cb59六份算法由scripts/generate_cb59_replay.py从固定Git对象生成，目录csrc/kernels/dispatch/low_latency/cb59下的算法文件不得直接编辑；修改提取脚本再生成。参考fixture保留完整业务token和同步调用序列，不以更新hash冒充等价。Host私有布局/UB helper独立维护。
- cb59移植保留全核count、R分片前缀、R份副本/R×R控制、按角色初始化、190KiB UB及原同步/缓存行为；不搬当前P1/VF/overlap。移植固定56核28/28、G16、P0（实际28生产者），协议21028/identity cb59-replay。current保留可配S/R、P及G默认4/协议19000+S。源码同一份但构建选择不同，必须核验plan/build_info/native/allrank的dispatch_impl。
- 两实现都用当前SHMEM8e3fc61、MTE Session、普通页、相同FP8与10/100/尾50采样。它是受控环境下的旧算法，不是旧SDK4dd80e2/1GiB页/动态1:1/单入口原二进制复现；不预言恢复17–22us。
- 共用schema15稀疏载体，不新增同步。cb59发送核role=send，current保留原能力role；解析拒绝同轮混角色。旧COUNTS_RECEIVED在提取/Sum后、前缀与clear前，current在clear下发后，禁止跨实现逐核阶段加速比；主要比较OFF完整task及ENTRY到prefix/probe/workdone。构建全部四wheel后再串行执行，首次仅两份SEND_FIRST。指南tests/ll_stage_timer/CB59_REPLAY.md。
- 本次仅本地CPU/原生mock验证，无27/CANN/NPU运行。完整证据见docs/BUILD_DISPATCH_SPLIT.md。

## 当前交付方式（2026-10-25，优先于历史远端验收要求）

- 用户要求后续不再使用27（177.22.27.1）验证编译；27隔离编译授权已结束，不再为此连接、构建SDK/wheel或执行Host导入，也不擅自换其他远端主机。
- 完成必要的本地检查后直接提交、推送，提供对应提交的目标机编译和测试命令；CANN编译、设备正确性与性能由用户执行。
- 远端编译不再作为推送前置条件，不重复询问27授权；未执行的验证须如实注明。下方已有27验证记录仅是历史证据。
- 用户可粘贴的构建/执行命令不得退出当前交互shell；使用函数返回和外层条件判断，失败保留终端与日志。

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

R0前缀生成改为device/dispatch_prefix_scan.h的uint32 VF扫描：64项/段、6级gather/add、寄存器carry。
输入缓冲按256B补齐，Host与Device共用预算。配置标识19000+S，wire/epoch和schema12不变。
无逐项Scalar累加，不使用float转换；A3不支持该A5 VF，不能作为运行性能验收。

当前q分支叠加9e24a79 R0前缀VF，构建标识19000+S，q表VF播种保持。
PREFIX_COMPLETE本地专家计数提取仍Scalar；没有改变wire/epoch或发送顺序。

当前schema13通信诊断：108metrics/192采样事件/每核5120B。SEND_SETUP与首包前区间分开。
First区间从KERNEL起点到第一次DataCopy/Probe之前，空任务不补零；保留无新排空的observe语义。
新报告tests/ll_stage_timer/communication_report.py，参考COMMUNICATION_DIAGNOSTICS.md。

## 当前稀疏诊断（2026-10-23，覆盖旧schema13/14采集说明）

schema15/profile sparse6，每核512B槽、14个首次时间点，结束时仅发布192B；旧逐包/逐轮询累计和trace数组已删除。
不恢复动态ProfileIndex或旧诊断SyncAll；OFF所有探针为空，业务同步和通信19000+S保持。
SEND_DONE须在已有MTE3_S或最终PIPE_ALL之后，不能把MTE3_V指令发出当作Scalar确认完成。
R0的PREFIX_LOCAL_DONE仅为原前缀阶段下发边界；小B并行路径有原V_S，不为统一测量而增加屏障。
一键diagnose默认OFF+sparse串行跑多个B，同份数据分析多项疑点；新旧schema只允许分别离线分析，不拼接阶段或时间线。
首轮只发SEND_FIRST，摘要及必要定点补充累计最多10次，重复内容不重发。包数/轮询比例未采集，不补零。
schema15须全rank完整重编；内部sparse15链接符号与Host/device布局检查拒绝旧库混装，SDK不因此重建。
26/27本次SSH均超时；本地CPU/Zig通过不代表Bisheng编译、设备正确性或目标性能已验证。详见PHASED_DIAGNOSIS.md及BUILD_DISPATCH_SPLIT.md。

诊断入口须先完成全部OFF/sparse编译及Host身份校验，再开始任何设备采样。--build-only独立编译无需设备/拓扑，保留已有可见性，成功后发布相对路径wheel-map.json；运行以新目录配合--skip-build复用并重新核验。多机先各自编译完成再统一运行，不假定脚本有跨机构建屏障。
active_capture_profile的kernel_us/status/reason属于逐rank/轮次观察值，不作共同身份；measurement口径及shape/source/SR/schema/profile保护保留。

## 2026-10-25 机内EP16小B与官方SHMEM主线

本任务仅机内EP16/M<=32/B<=32；每NPU的3个组内fullmesh、4个机内Clos、1个跨机口不能直接当作软件可指定的8个QP。LL保留MTE，物理16连续rank/G4不变。先独立测试前缀生产核P，默认DEEPEP_LL_SMALL_B_PREFIX_CORES=0表示R，显式1..R仅影响EP16小direct；其他EP用R、dedup保留原R0。S/R、receiver owner、wire、send顺序及combine不变；Host/device UB同步且230KiB不变，不降低既有lanes来换预算。所有R代次继续前进，仅前P核发布ready依赖。raw/effective P写入wheel/全rank/结果；不同P不能混装或算逐核阶段比。P-only与早接收/预取重构分开。

用户授权SHMEM升级官方最新主线：锁定master8e3fc610f91ae9e5e8cf13748cfb74826b272c2a，不改vendor源码。runtime_config替代旧sharedjetty setter，旧peerMask和可配置UDMA超时已移除，显式UDMA dense须用户配置。LL ST显式MTE；旧HT普通peer输出缺注册证明，Host/Notify前明确拒绝。新SDK与完整wheel都需重建，不能以旧SDK复用。SDK完整include/src/lib身份和源码revision均严格检查。

P26/6/1实验固定S30/R26、新SDK、采样参数及均匀随机top-k种子配对。先完成全部六个OFF/sparse wheel及Host检查，再串行采样；首次只发三个SEND_FIRST，整个实验含补充累计<=10次。实际设备与CANN验收状态以docs/BUILD_DISPATCH_SPLIT.md为准，不以CPU或历史schema编译代替。

同日用户明确授权27在`/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/topology-mainline-20261025`编译新SDK及完整OFF/ON wheel，不运行NPU，编译报错及时告知。已完成dda407e源码验证：官方SDK8e3fc61无补丁；P6/S30R26七mode+selector冷编译OFF37.682s/ON39.696s；P26/6/1六份完整wheel全部编译、链接、Host身份和实际新SDK路径/哈希核验通过。Linux Host/C++回归162pass；无编译/链接错误，仅上游宏重定义警告。原部署、SDK和共享Python未改；四可见性变量0,1。无NPU正确性/性能结论。日志及wheel-map在该隔离根目录，细节见docs/BUILD_DISPATCH_SPLIT.md。

## P1 count 单VF独立候选（2026-10-25）

用户要求固定P1，把ready检查和计数提取合并成单个VF，单独提交以便A/B，并继续27隔离目录完整OFF/ON编译、不运行NPU。基线578d32b；仅EP16小direct且有效P1启用P1CountReadyAndExtract，其他P/域/dedup保持原路径，P全局默认不改。保留MTE2_V及单次V_S；ready精确float1位模式，count保留uint32，失败轮私有scratch不得用于清理/发布。复用已有UB，不改清理顺序、前缀发布、wire、SDK、lane或combine。A/B两边固定S30R26/P1，使用各自源码匹配的wheel，先全部构建后串行采样；指南tests/ll_stage_timer/P1_COUNT_VF.md。

该候选已通过27独立VF预编译、七mode+selector冷编译OFF38.185s/ON40.251s、完整OFF/ON wheel及Host/实际SDK路径哈希核验。Linux相关142pass；初轮CPU测试漏<iterator>已修正并重跑，生产kernel无编译错误。SDK8e3fc61校验复用未重建，原部署及隔离目录其他任务未跟踪文件保留。最终源码指纹3cf59518dc900d23883db9eea6b768fbaa6fb22ac6aefdad18ec7a1c29489e0a；没有NPU正确性/性能结论。证据见docs/BUILD_DISPATCH_SPLIT.md。

用户目标SDK同官方8e3fc61但来自HNS分支，合法build字段hns_1825_pi_on_chip=1被旧MTE白名单误拒。当前SDK兼容修复接纳严格int0/1且须HNS_1825，MTE复用仍精确核对共同构建输入和完整哈希；不删除字段或重写旧manifest绕过。用户另确认旧manifest仅include34/src0/lib1，需目标机显式重建生成完整清单。新版A/B基线e19aa4d（578d32b+同SDK修复）与当前候选共用该修复，kernel差异仍只P1单VF。SDK/完整性修复仅本地CPU验证，不再27编译。命令见P1_COUNT_VF.md。
