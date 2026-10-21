# EP16/B256 FP8：dc_opt与当前LL的差异及优化候选

日期：2026-10-22。本文为源码/架构审查和CPU路由计数，不是性能归因报告。

## 1. 结论与比较边界

用户观察：ops-transformer在EP16、B256、H3584、E384、K8、FP8下比当前分支好约20%。
尚未提供双方原始profiler任务、运行参数与ON轨迹，不能将20%归因于单一阶段。
“时延低20%”和“吞吐高20%”也不同：前者Tref=0.8*Tll，后者Tref=Tll/1.2。
本文不虚构绝对微秒数，以下排序是值得验证的优先级。

固定源码：

- 当前：kernel/ll-g4-group-compact，a5c64a3a4ed513bdd1676aab0a929e2553d16b7f。
  该版本已撤销两轮compact任务重分配，保留R0共享前缀、R核计数、230KiB dispatch预算、
  pending/tokenPos向量化；最新非去重专用S/R修改不作用于M256的去重路径。
- 参考：[dc_opt@35741e3cd6f5ca8e4799a57f90a0c1975153a7ad](https://gitcode.com/zhong-zixin/ops-transformer/tree/dc_opt)。
  已只读核验远端分支。重点类为MoeDistributeDispatchV2A5FullMesh。
- 本地默认S28/R28、G4、56AIV；实际安装配置仍须核对。参考实际AIV数量不能由本地默认推断。
- B=M256，H按3584解释；FP8 ST是预量化E4M3+E8M0 scale输入，不在kernel内量化。
  参考必须核对ORIG_DTYPE_X/EXPAND_X、QuantMode、scale格式、输入/输出权重和mask。
- 本文针对dispatch。若20%是dispatch+combine总和，必须先拆分两算子；不能将总差距套入dispatch。
- G4是逻辑组，不等于物理机界；EP16不能自动推出16卡是同一物理通信域。

优先建议：先做本核q表，随后做接收端有界连续段批量化/元数据合并，再做受控的
“相同shape、不同去重策略”消融。暂不重新引入跨核任务调度。
参考最可借鉴的是降低每token的搬运指令和调度开销；不是照搬全量组索引重分配。

## 2. 两边到底跑了什么

参考源码入口：
[FullMesh kernel](https://gitcode.com/zhong-zixin/ops-transformer/blob/35741e3cd6f5ca8e4799a57f90a0c1975153a7ad/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h)
（以下行号对应该提交）。

| 项目 | 参考dc_opt | 当前LL | 可能影响 |
|---|---|---|---|
| 路由协议 | 每expert路由直接投递，无FIRST_HOP/relay | 远端同G4多命中合并首跳，目标组relay | 减跨组初发，但增加串行依赖 |
| 发送资源 | count核最多8，组发送开启时其余用于发送 | S28远端，R28计数及本组发送 | 不能直接比较“发送核数”；R也发送payload |
| 发送次序 | G4配对轮次，本组拆片穿插 | 首跳优先，后DIRECT；rank/core/token错峰 | 目的地集中度、注入时间线不同 |
| 计数/前缀 | 分片计数/前缀，写共享前缀相应片段 | R计数，R0全表prefix，消费者读同一份 | R0关键路径/全表轮询 vs 分片同步 |
| 接收分配 | 分源专家记录，再按UB可容纳token批量搬运 | 56核按输出行均分，pending逐包探测 | 行数均衡 vs 每条指令搬运的工作量 |
| 接收流水 | 两份批量ping/pong buffer | 最多6路重叠lane、每包独立metadata/relay | 两批可能胜过六个单包，不能只看lane数 |
| 就绪策略 | 对记录当前一批检查；未齐换记录 | 16个pending快照，取其中就绪包，首跳优先 | 参考摊薄开销；当前更能容忍乱序/relay |
| 输入复用 | TokenToExpert用lastTokenId控制相邻复用 | IssueInput每个初发包重新取x/scale | 参考能否实际命中由发送顺序决定 |
| wire基本块 | 480B数据+32B标记 | 同样512B块，另有包类型/relay尾部 | 不能把512B打包独认为LL劣势 |
| q计算 | 每条发送route扫描原始前缀并归约 | 首跳/DIRECT与尾部成员仍需前缀归约 | 共同问题；可优化，但不是差距的充分解释 |
| 自rank路由 | 可能仍写窗口后复制输出 | 包括本地/同rank relay槽投递 | 二者都不是零成本本地直写 |

参考关键函数位置：SetTilingDataAndCal约404；BuildSendGroupIndices约875；
SendToMoeExpertLoop约1009；CalTokenSendExpertCnt约1291；
CalRecvAndSetFlag约1738；CopyInAndOut约1997；WaitAndFormatOutput约2109。
源文件路径优先于易漂移的行号。

### 2.1 不能误判参考开关

enableGroupSend仅在isMc2Context为真时从context读取，注释标记仅V3。
Context还要通过UB_MEM和拓扑检查（连续G4、组内mesh、组外clos）。
普通V2调用即使经过同一FullMesh头也不保证开启G4。实际benchmark需保存context开关。
若未开启，参考走另一套token分片/SendIndexRange，输入复用和核均衡行为不同。
EP16/B256不满足小batch串行模式的EP<16且B<32条件；不要用该路径解释本案例。

参考接收并非无代价：要求当前copyCnt批次都到齐才复制，未齐轮询其他记录。
如果只缺一个包会拖延整批，不能将其改成无条件等待一个记录，特别是LL存在relay时。

## 3. 当前算法的定量画像（纯CPU）

使用当前ST make_routes，normal，seed2（默认3轮正确性后timed路由），S28/R28/G4。
实际用户seed、route或correctness-repeats不同需重算；不是实测网络流量。

| 全16rank、一次dispatch | 数量 |
|---|---:|
| 原始expert路由/最终输出行 | 32768 |
| 初发packet | 19158 |
| FIRST_HOP | 7830 |
| DIRECT初发 | 11328 |
| relay投递 | 13610 |
| 初发+relay | 32768 |
| 原始跨G4路由 | 24645 |
| 去重后跨G4初发 | 11035 |
| relay目的与首跳rank相同 | 3413 |
| 初发目的为源rank | 2027 |

该shape的当前wire为8×512=4096B；scale112B，payload主体3584B。
初发+relay写窗口逻辑量128MiB（包含同rank写，不能称为128MiB网络TX）。
跨组初发逻辑量从约96.27MiB降到43.11MiB，约减少55.2%，
但省下的路由成为13610次组内relay，其中3413次仍是同rank槽位写。
所以“跨组字节减半却慢20%”在逻辑上完全可能：
发送少了，但每包poll/pack/metadata及relay等待和输出复制没有消失。

接收平均每rank2048行，56核约36.6行/核；记录总数E=384，平均每记录5.33个token。
这使参考的“同源同专家多token成批”具有潜在摊销空间，但低命中/尾部记录不能盲目凑大批。
当前每次16包Probe仍逐包下发flag搬运；全就绪也要至少约32768次packet flag搬入
（每次含8个标记块），未就绪轮询会重复。不能把16包归约误当作一次GM搬运。

当前源码存在每包x/scale加载：初发19158次，相对全域4096个原token约4.68次/原token。
这是代码调用计数；相邻数据可能命中L2，不能据此说HBM重复读取4.68倍。

远端路由q前缀长度累计25260690项/全域，即约1578793项/rank；
不是Scalar指令数，原路径使用向量运算。S3式本核q表播种+本片共30080项/rank，
可减少归约发射与同步，但不能拿项数比当时延加速比。

## 4. A5架构依据与能推出的结论

使用[华为《昇腾950 NPU架构白皮书》官方PDF](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)。
本地文件SHA256：ece3405e6a17fabdd462338fb94266558649a6407a2f28008403211387b3a927，
40页，元数据创建时间2026-06-04。以下使用PDF页序，而非正文页号。

| 依据 | 页码 | 本次设计含义（推论） |
|---|---|---|
| RegFile、双发SIMD、SIMD/SIMT混合VF | 20–22 | 把规则metadata计算留在寄存器/VF，减少碎片化调用 |
| NDDMA支持多维搬运及地址生成 | 22–23 | 连续记录批量读取值得试；不能保证任意pending散点可直接合并 |
| BufferID同步 | 23–24 | 可探索简化缓冲所有权，不能直接删除lane完成依赖 |
| L2为512B line、128B sector，支持hint | 15、26 | 小记录控制流量不等于有效字节；只读输入与并发flag分别处理 |
| UB Memory与URMA、CCU是不同能力 | 30–33 | 当前MTE路径不等于CCU/URMA；白皮书不提供现协议完成性证明 |

白皮书没有给出可用于本kernel的A5/A3 Scalar时延换算系数。
用户已明确A5 Scalar强很多；不能用A3标量耗时预测A5收益。
文档中的“双发/乱序”讨论Vector SIMD，不据此宣称Scalar也具备同样执行特性。
2016GB/s是整芯片双向规格，不是本作业单向有效带宽分母；不推定未公开机型/拓扑。

## 5. 独立审查当前收发逻辑

### 5.1 发送端

[发送实现](../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h)：
CalTokenSendExpertCnt、FillTail、IssueInput、ConsumePacket。

- q逐route重复归约：DIRECT预缓存q，FIRST_HOP尾部成员在ConsumePacket中再计算。
  这把metadata成本插进payload流水；其V_S等待也可能牵连下一步发包。
- Classify按本核token处理，未重复扫描全rank所有组；此前失败compact优化不应重新引入。
- 两份input缓冲与一包lookahead已存在；仅加一个“prefetch”不是新优化。
- 两次Copy将480B数据插入512B wire，后续V_MTE3和lane MTE3_V是必要依赖。
  优化应减少生成/写入次数，不能直接删wait。
- 原R核生成本组描述符时已经顺带算q，不能再增加一份同样的本组q预处理。
- payload重用需区分原始输入、量化结果、被FillTail改写的包尾；不能共享仍被DMA读取的输出lane。
  当前FP8预量化路径无需量化，只需保持输入及scale驻留。

### 5.2 接收端

[接收实现](../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h)：
Probe/TryBatch/IssueLoad/ConsumeLoaded/ClearFlags。
[任务表](../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h)。

- tokenPos和pending初始化已向量化，不能将其再次列为未做优化；对齐短尾仍Scalar。
- 核内pending交换删除破坏连续顺序，后续快照即使最初按记录排列也逐渐散乱。
- 每个就绪包涉及flags、payload、type/destination等小块元数据、多组事件依赖；
  56核按行数均分没有按FIRST_HOP成员数/relay字节量均分。
- FIRST_HOP在本地输出前relay，避免被输出写阻塞，但relay与output同用MTE3；
  一包多relay可能占住lane并延迟其他已到包。
- 当前FIRST_HOP优先只对本次16项快照有效，不保证全队列首跳优先。
- 已验证标记的普通/terminal包走紧凑加载，首跳保留wire以便转发；
  不可再次做一遍通用去flag拷贝，否则增加UB流量。
- 检查每块flag是当前完成协议的一部分。仅检查最后一块/只检查header需要额外发布顺序证明，
  白皮书或“同一个DataCopy”本身不足以代替证明。

### 5.3 计数/前缀

[R核计数与R0前缀](../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h)。

R0先等384条计数，再生成完整prefix；56消费者均依赖该发布。优点是消除重复MTE3广播，
代价是集中全表扫描和共享读。R0自己的本组初发也在前缀发布之后，可能成为尾部。
其他R不计算prefix，可提前本组初发，但仍要等prefix才能接收。
应同时测最后count到达→prefix发布、prefix发布→首接收，而不只看CAL_CUMSUM。
现在230KiB预算并不等于用满230KiB；6lane上限来自事件规划，扩大预算不自动增大lane数。

## 6. 优化候选与优先级

| 优先级/方案 | 核心修改 | 预期减少 | 代价与必须验证 |
|---|---|---|---|
| P0 本核q表 | 原token分片播种直方图，保存本片q，Send/FillTail查表 | 重复前缀归约/V_S | 依赖随机hist更新；不能与量化scratch覆盖；首token语义 |
| P0 连续记录批量接收 | DIRECT/terminal先按记录段有界批量poll/load/output，首跳维持逐包 | MTE2/3下发、pending维护、事件数 | 不能为凑批等待未就绪terminal，保留bounded扫描/单包回退 |
| P0 小元数据合并写 | 输出type/destination按连续输出row收集成小批 | 每包4B/12B小DMA | 独立buffer生命周期，最后不足一批也写回；不延迟relay |
| P1 输入复用 | 同token连续任务共用不可变x/scale，尾部在独立lane填充 | 重复MTE2输入/预取发射 | 分组重排影响命中；保留预取收益与DMA退休 |
| P1 融合pack/metadata VF | 用A5 RegBase减少碎片化向量调用和UB中间值 | 发射/同步/UB流量 | 先小范围，保持480/512wire及标记时序；需A5编译与设备 |
| P1 受控去重消融 | 同B/M/shape可显式选direct作为实验变量 | relay/尾部/关键路径 | direct增加跨组流量；不能仅改M32，因为B256>M32非法 |
| P1 R0减关键路径 | prefix向量scan、分批拉count；或R0本组发包更早 | prefix全局门槛 | 不能未收齐就发错误offset；控制buffer事件独立 |
| P1 调S/R与lane | S/R=20/36、28/28、36/20，lane有界比较 | 某角色饱和/等待 | 和56，输出/relay负载另算；不是越多核/越多lane越快 |
| P2 同rank relay直输出 | 首跳本rank成员由当前核写对应最终行 | 本地window往返和再次poll | 原接收owner必须交接/认领，清flag/combine元数据更新，协议风险高 |
| P2 基于描述符轮询 | pending从逐行改为记录游标，必要时细化 | O(输出行)元数据与随机访存 | 稀疏乱序/relay HOL风险；需兼顾FIRST_HOP |
| P2 L2分类hint | 输入/route只读缓存；共享flag/peer写入另审 | cache污染/重复流量 | 缓存一致性不能推测；不全局关闭L2 |
| P3 CCU/URMA重设计 | 独立通信后端方案 | 核端轮询/发射 | 不是局部patch，不能从白皮书推出可直接替换 |

P0 q表建议单独提交：当前M256去重packet仍归原token核，因此局部下标成立。
每核最大10×8×4=320B q表，E384直方图1536B。播种必须覆盖[0,start*K)原始路由，
首跳mask中的所有成员也必须有q。非去重分片是flattened route，需要单独的begin；
R本组q已生成。q表要活到lookahead最后一次FillTail结束，预算/Host检查一起更新。

批量接收建议逐步实施：
A. 仅在同record且连续q、相同packet类型、已就绪的段上合并；
B. 先合并输出小metadata，再合并payload加载/输出；
C. 不足一批立即处理，不将所有包等待在最慢terminal后；
D. 批量阶段仍逐输出行维护combine handle，FIRST_HOP尾部不丢弃。

“纯收益/无风险”不适合用于上述任何新DMA/同步优化；对齐、尾部、事件复用和完成性必须测试。

## 7. 如何判断20%差距的来源

先核对：双方commit/wheel指纹、AIV、dtype/quant/scale、G4开关、输入专家表、
activeMask/共享专家、输出metadata、是否含量化、cache/Exp/all_reduce、warmup和采样尾段。
只比较各自完整kernel task时间；Event/API不能替代task，ON不能替代OFF基线。
参考的“少metadata”若来自不同API输出契约，应单列，不称同语义算法优势。

最小实验顺序：

1. 同一真实top-k与权重固定seed，保存FP8原始输入/scale格式描述。
2. 当前基线OFF和参考OFF串行采样，各10/100/尾50；同时保留原始task名字，禁止混kernel。
3. 当前ON独立采样，记录每rank每core原始interval/metrics：
   CAL_TOKEN、TOKEN_METADATA、TOKEN_BUFFER_WAIT、RECV_POLL、RECV_INPUT、
   RELAY_ISSUE、PREFIX_COMPLETE、BUILD_EXPERT_TASKS、RECV_COMPLETE_WAIT、KERNEL。
   嵌套阶段不相加；对同rank/轮/核分析，不能相减独立p50或跨rank时钟对齐。
4. q表、批量接收、小metadata合并各自单独A/B。预先设定正确性覆盖：
   随机/热点、空核、K1/8/32、非对齐分片、同rank relay、多轮窗口复用、完整combine。
5. 最后才组合有收益的改动，观察收益是否重叠。以整体OFF耗时下降且正确性通过为接受条件。

可能的诊断：
- CAL_TOKEN大而输入/通信等待小：先q表。
- RECV_POLL调用多、每批ready率低：就绪探测/记录调度；
  ready率高但RECV_INPUT/OUTPUT开销大：批量搬运。
- 首跳核MTE3等待与尾部terminal完成相关：relay依赖或负载不均；
  仅RELAY_ISSUE大不能证明传输拥塞。
- prefix晚且所有核等待：count到达或R0串行；别只扩大UB。
- 两边通信量明显不同但小包指令相近：链路不是唯一瓶颈，应区分packet提交次数和payload字节。

## 8. A3验证范围与限制

用户允许A3隔离验证。本次未启动A3设备测试：当前优先问题可以从源码和CPU路由计数确定，
尚无能把A3时间换算为A5收益的模型。没有把普通aarch64 CPU计时冒充AIV Scalar计时。

后续A3微基准只验证以下局部，单卡即可，四个可见性变量固定0,1，优先只用device0；
经scripts/remote.ps1和显式A3配置运行，输出/缓存只写/home/tyf/Q4_remote_test下独立目录。
先查现有任务/Git状态，不安装系统依赖，不覆盖用户任务、不改目标生产架构。
A3专用microkernel应另建验证目标，不能把生产Ascend950目标改成A3。

| 微基准 | 对照 | 可以得出 | 不能得出 |
|---|---|---|---|
| q | 原逐route归约 / 本核seed+表 / 纯Scalar hist | 等价性、操作量、A3相对局部成本 | A5微秒/端到端加速比 |
| 分组 | 原Classify / 重复全表 / 单次本片 | 算法复杂度、Scalar占比趋势 | A5 VF调度或跨核通信成本 |
| 接收表 | Scalar逐行 / 对齐段向量 | 索引正确性、A3同步代价 | A5 CreateVecIndex实际收益 |
| prefix | Scalar scan / 局部可用向量scan | 数值等价、单核A3局部耗时 | A5 count到达/共享GM竞争 |

计时需GetSystemCycle或设备event明确频率，空kernel/空循环基线，预分配、无计时内CPU复制，
checksum防止优化删除。测试输入normal/hotspot、B8/48/256、固定K8/E384，分别报告cycles、
发射方式与loop次数。A3与A5分别列，不用固定倍数折算；A5 Scalar更强也可能使A3的
“向量更快”结论翻转。A5专有VF/NDDMA/BufferID不能在A3缺失时伪造性能比较。

## 9. 当前交付与可复现性

本次仅新增分析文档和标准库CPU计数脚本，不改kernel。
运行tests/elastic/audit_ep16_dispatch_cost.py可重现第3节的normal/seed2计数。
脚本只报告逻辑操作量，不导出或模拟硬件时延，不以此证明20%来源。
下一步优先建议独立q表和接收批量化两个小实验，先各自A/B，再讨论合并。

### 源码导航（当前commit）

- 主调度：csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h，Process。
- 初发：device/dispatch_server_dedup_send.h，Process、SendWithDedup、FillTail、CalTokenSendExpertCnt。
- 接收：device/dispatch_server_dedup_recv.h，Probe、TryBatch、IssueLoad、ConsumeLoaded、ClearFlags。
- wire/标记：device/dispatch_server_dedup_receive_packet.h及csrc/kernels/common/ep_packet_receive_layout.h。
- 单核prefix：device/dispatch_server_dedup_counts.h，TryRecvCnt、CalRecvAndSetFlag、CompleteCumsum。
- 内存/事件规划：common/ll_overlap_layout.h、ep_memory_server_dedup_data_layout.h。
- 参考分支同时查op_host/arch35及context代码，核实是否真正启用FullMesh/G4/CCU；
  不从文件名或用户调用的“dispatchv2”字符串直接推断入口。
