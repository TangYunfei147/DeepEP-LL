# LL dispatch 卡死：分级编译宏定位

本次只修改生产dispatch kernel及本篇文档，不修改Host绑定、Python入口、构建脚本、测试算法或通信协议。所有打印默认关闭。正式性能测量必须关闭全部调试宏。

## 1. 宏在哪里改，怎样重新构建

修改唯一配置头文件：[dispatch_ll_debug.h](../../csrc/kernels/dispatch/low_latency/common/dispatch_ll_debug.h)。直接编辑对应`#define`的默认值，再在所有参与机器从同一份源码重建安装生产wheel。它们是**编译宏，不是运行环境变量**，只`export LL_DISPATCH_DEBUG_STAGE=1`没有作用。

第一轮只改：

```cpp
#define LL_DISPATCH_DEBUG_STAGE 1
#define LL_DISPATCH_DEBUG_COUNTS 0
#define LL_DISPATCH_DEBUG_PREFIX 0
#define LL_DISPATCH_DEBUG_SEND 0
#define LL_DISPATCH_DEBUG_RECV 0
#define LL_DISPATCH_DEBUG_RANK -1
#define LL_DISPATCH_DEBUG_CORE -2
```

已按本分支SDK流程准备好仓内依赖时，在仓库根、正确Python环境中执行：

```bash
source scripts/project_env.sh
export DEEPEP_NPU_ARCH=Ascend950
python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

正常增量构建会跟踪该头文件变化。检查日志确实重编译生产dispatch设备文件；若自定义构建缓存未更新，应按自己的构建配置重建对应设备目标，不能只换Python文件。安装到实际运行测试所用的解释器，重新启动所有rank。**只重建`build_profile_bridge.sh`不会替换生产LL wheel。** 开启这些宏时不要运行metadata提取快照代替完整LL。

实际运行仍使用原LL测试入口和原通信环境，第一轮建议只测一个配置，附加`--skip-perf-test --test-first-only --trace-ops`，并收集每台机器的完整stdout/stderr。`--trace-ops`是已有的Host边界标记，不会单独触发kernel打印。保留原来的外部超时；这些宏不会让等待自动退出。

## 2. 筛选及输出上限

| 宏 | 默认 | 含义 |
|---|---:|---|
| `LL_DISPATCH_DEBUG_STAGE` | 0 | 粗粒度阶段BEGIN/DONE，先开此宏 |
| `LL_DISPATCH_DEBUG_COUNTS` | 0 | 计数生成、首次跨卡发送、接收计数ready轮询 |
| `LL_DISPATCH_DEBUG_PREFIX` | 0 | 本卡接收核之间的片段总数和前缀发布同步 |
| `LL_DISPATCH_DEBUG_SEND` | 0 | 路由准备、packet输入准备、lane复用和payload发送 |
| `LL_DISPATCH_DEBUG_RECV` | 0 | 接收任务、包标志探测、预取/消费/relay及清理 |
| `LL_DISPATCH_DEBUG_RANK` | -1 | -1为所有rank，非负值只打印对应全局rank |
| `LL_DISPATCH_DEBUG_CORE` | -2 | -2为核0和第一个接收核；-1为所有核；非负值为指定AIV核 |
| `LL_DISPATCH_DEBUG_POLL_EVERY` | 65536 | 首次轮询采样，此后每N次轮询采样一次；不是时间间隔 |
| `LL_DISPATCH_DEBUG_MAX_POLLS` | 8 | 每个等待循环最多采样次数；接收packet探测为整个Receiver::Process共用预算 |
| `LL_DISPATCH_DEBUG_ITEMS` | 4 | 每次采样最多打印的缺失记录/packet槽/relay成员数 |
| `LL_DISPATCH_DEBUG_PACKETS` | 2 | 每核前N个packet的详细收发记录；不是整次kernel只允许发送N包 |

细节宏可独立启用，建议同时保留STAGE=1。默认核划分为发送核0..31、接收核32..63，因此CORE=-2通常打印核0和32。**CORE=0看不到接收等待；CORE=32看不到发送核的payload阶段。** 先用默认筛选，若被选核都完成而作业仍卡住，再设CORE=-1找未完成核；之后锁定该rank/core细查。

所有限额都是每次kernel调用的局部预算，不跨调用累计。达到打印上限后仍继续原来的轮询，日志停止不等于等待结束。若目标包不在前N包，增大PACKETS并缩小rank/core范围。COUNT_TX日志按本核有限的专家区间输出，不受PACKETS限制。

## 3. 通用日志格式与源码位置

```text
[LLD] r=0 c=32 COUNT_POLL_AFTER line=... a=1 b=1088421888 c=8 d=0
```

`r`为全局rank，前面的`c`为AIV核；`line`是打印点在编译源码中的行号。末尾`a/b/c/d`是四个无符号整数数据字段，含义见下文，末尾`c`不是核号。地址、位模式也按十进制输出。

CANN可能在日志前添加AIV前缀。用下面命令定位**当前实际编译版本**的标签：

```bash
grep -R -n 'COUNT_POLL_AFTER' csrc/kernels/dispatch/low_latency
grep -n 'DISPATCH_ENTRY' csrc/kernels/dispatch/low_latency.cpp
```

不同rank/core日志可以交错，不能按终端相邻两行推导跨核先后。按`r`和核号分组看BEGIN/DONE；同一rank/core会多次调用dispatch。`MAGIC_DONE.a`只在1/0之间轮转，是窗口选择，不是唯一调用编号。首次排查尽量缩小测试调用次数。

## 4. 第一轮：阶段宏告诉你下一步开哪个宏

阶段主调用链在[dispatch_server_dedup_kernel.h](../../csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h)，设备入口在[low_latency.cpp](../../csrc/kernels/dispatch/low_latency.cpp)。

| 最后看到的边界（对应DONE缺失） | 代码范围 | 下一步 |
|---|---|---|
| `DISPATCH_ENTRY`，无`FFTS_DONE` | `util_set_ffts_config` | 核对commArgs/FFTS配置；此时尚未计数通信 |
| `FFTS_DONE`，无`MAGIC_BEGIN` | tiling读取、TPipe构造、Kernel::Init前段 | 检查tiling/参数和进入Init的位置 |
| `MAGIC_BEGIN`，无`MAGIC_DONE` | `RotateDispatchServerDedupMagic` | 检查本卡peerMems、本地magic槽的MTE2/MTE3硬事件；不是首次跨卡通信 |
| `MAGIC_DONE`，无`COUNTS_INIT_DONE` | `counts_.Init` | 检查commArgs、world、专家数和CTRL设置 |
| `COUNTS_INIT_DONE`，无`INIT_DONE` | Sender/Receiver::Init | 检查该核角色对应的布局与参数读取 |
| `COUNT_SEND_BEGIN`，无DONE | `counts_.SendCnt` | 开COUNTS |
| `COUNT_RECV_BEGIN`，无DONE | `RecvCnt → WaitDispatch` | 开COUNTS，选接收核 |
| `PREFIX_SUM_BEGIN`，无DONE | `CalcCumsum → CalRecvAndSetFlag → GetCumSum` | 开PREFIX |
| `PREFIX_READY_BEGIN`，无DONE | `CompleteCumsum → WaitCumSumFlag`及前缀加载/输出 | 开PREFIX |
| `PREFIX_DRAIN_BEGIN`，无DONE | 接收核原有`PipeBarrier<PIPE_ALL>` | 此前写入或DMA未完成；先看COUNTS/PREFIX |
| `PAYLOAD_SEND_BEGIN`，无DONE | Sender::Process | 开SEND，选发送核 |
| `PAYLOAD_RECV_BEGIN`，无DONE | Receiver::Process | 开RECV，选接收核 |
| `FINISH_BEGIN`，无DONE | Counts::Finish中的原有PIPE_ALL | 本核在途DMA/事件未收尾；回查该核收发日志 |

`MAGIC_BEGIN`字段a/b/c/d为B/H/K/E；`MAGIC_DONE`为magic/world/0/0；`INIT_DONE.b`表示是否开启去重。发送核和接收核并行，发送核并不等待接收核完成计数前缀。只看到某个核的FINISH_DONE不能证明整张卡或整个作业完成。

## 5. COUNTS：首次跨卡写入与缺失来源rank

源码：[dispatch_server_dedup_counts.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h)。保持STAGE=1，开启COUNTS=1；先看挂住的接收核，必要时切换到日志指向的来源rank。

| 标签 | a/b/c/d | 解释 |
|---|---|---|
| `COUNT_RANGE` | startExpert/endExpert/stateOffset/world | 本核专家半开区间；空区间会直接返回 |
| `COUNT_IDS_DONE` | B*K/0/0/0 | 路由GM→UB及原MTE2等待完成 |
| `COUNT_CALC_DONE` | start/end/0/0 | 专家计数生成完成 |
| `COUNT_PUBLISH_BEGIN` | start/end/0/0 | 原S→MTE3事件完成，即将发计数 |
| `COUNT_TX_BEFORE` | dstRank/firstExpert/recordCount/relativeOffset | `SendExpertCntRange`的DataCopy前 |
| `COUNT_TX_ENQUEUED` | dstRank/firstExpert/recordCount/peerBase | DataCopy指令已下发；不代表对端已收到 |
| `COUNT_WAIT_RANGE` | firstRecord/recordCount/stateOffset/world | 本接收核负责的计数槽范围 |
| `COUNT_POLL_BEFORE` | sample/firstRecord/recordCount/stateOffset | 即将从本卡读取ready槽 |
| `COUNT_LOAD_DONE` | sample/0/0/0 | 该次DMA和MTE2→V事件通过 |
| `COUNT_POLL_AFTER` | sample/sum的FP32原始位模式/目标条数/0 | 该次归约和V→S事件通过 |
| `COUNT_MISSING` | record/sourceRank/localExpert/ready原始位模式 | 记录ready不等于浮点1.0；零计数也必须发布ready |
| `COUNT_FLAGS_DONE` | recordCount/0/0/0 | ready到齐，退出WaitDispatch轮询 |
| `COUNT_LOCAL_SUM_DONE` | 0/0/0/0 | 本片前缀和局部总数发布已下发；随后清ready |

`COUNT_POLL_BEFORE`没有LOAD_DONE：收窄到DataCopy或MTE2→V等待；有LOAD_DONE没有AFTER：收窄到ReduceSum或V→S等待。反复出现AFTER且sum未达标：读取/归约在推进，但对端ready未到齐。FLAGS_DONE出现后才挂：回查局部总数发布或清ready，而非继续找缺失来源rank。

每条ready正常位模式为`1065353216`（0x3f800000，浮点1.0）。上面示例b=1088421888解码为7.0，目标c=8，表示还没到齐。原始位模式解码示例：

```bash
python3 -c 'import struct; print(struct.unpack("<f", struct.pack("<I", 1088421888))[0])'
```

接收记录`i`的来源rank为`i % world`，本地专家为`i / world`。发送地址相对日志中的peerBase为：

```text
stateOffset + (localExpert * world + sourceRank) * 512
```

stateOffset包含本轮4GiB窗口偏移和计数区前缀。对照两端的magic、world和**相对槽位**，不要要求不同进程映射出来的虚拟地址数值相同。

对于默认EP16/B32/E256，每卡16专家、64核参与计数，每核4专家：rank0/core0首次写自身，rank0/core4写rank1。接收核32..63每核等待8条记录。看到`COUNT_MISSING sourceRank=7`时，应进一步检查rank7中负责该目标全局专家的计数核；其核编号在此配置下为`(接收rank*16+localExpert)/4`。这条核编号公式仅适用于此配置与默认编译核划分。

## 6. PREFIX：本卡跨核同步

仍在counts头文件。开启PREFIX=1，CORE选挂住的接收核。

- `PREFIX_TOTAL_BEFORE/AFTER`对应`GetCumSum`：AFTER的a=sample、b=已经汇总的ready数、c=activeCumSumCores。`PREFIX_TOTAL_MISSING`的a=未发布的本卡AIV核、b=ready原始位模式。正常是浮点1.0。
- `PREFIX_FLAGS_BEFORE/AFTER`对应`WaitCumSumFlag`：AFTER的a=sample、b=整数标志归约结果、c=目标值`activeCumSumCores*8`。`PREFIX_FLAGS_MISSING`的a=生产核、b=该32B记录内的word编号、c=异常word位模式；这里正常值是**整数1**，不是浮点1.0。
- 反复AFTER但未到目标：先查看MISSING指出的**本卡核**是否还停在COUNT_RECV、PREFIX_SUM等更早阶段。它不是远端rank编号。
- 有BEFORE没有AFTER：收窄到该函数的DMA/硬事件/归约区间。有对应AFTER达到目标但没有粗粒度DONE：继续查循环之后的前缀读写、清标志或MTE3复用等待。

## 7. SEND：payload输入、lane与跨卡写

源码：[dispatch_server_dedup_send.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h)。开启SEND=1，CORE选发送核。

| 标签 | a/b/c/d | 停在此处时检查 |
|---|---|---|
| `SEND_PREP_BEGIN` | tokenStart/tokenCount/enableDedup/reservedBytes | UB池、输入队列、路由加载；去重且count=0允许立即返回 |
| `SEND_ROUTES_DONE` | B/K/lanes/0 | 若一直没有PACKET_BEGIN，检查Classify和CalTokenSendExpertCnt、以及是否根本没有发送任务 |
| `SEND_PACKET_BEGIN` | packetSequence/token/expert/q | 输入AllocTensor、GM→UB、队列事件、量化/输入完成等待 |
| `SEND_INPUT_DONE` | sequence/token/dstRank/routeMask | FillTail和S→V依赖 |
| `SEND_LANE_WAIT/READY` | sequence/lane/dstRank/0 | WAIT无READY表示等待旧MTE3读完该lane |
| `SEND_DMA_BEFORE` | sequence/dstRank/bytes/地址 | 打包及原V→MTE3事件已通过，即将DataCopy |
| `SEND_DMA_ENQUEUED` | sequence/dstRank/token/q | 已下发payload，不是接收完成确认 |

有DMA_ENQUEUED但无PAYLOAD_SEND_DONE时，还可能卡在下一包、未被打印的包、最终lane drain或source metadata写回。先增大PACKETS限定到所需范围，再与接收日志核对，不可把最后一条已下发记录当作故障包。

## 8. RECV：等待哪个packet、预取、消费或转发

源码：[dispatch_server_dedup_recv.h](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h)。开启RECV=1，CORE选接收核。

- `RECV_RESET_BEGIN/DONE`：原TPipe::Reset前后。
- `RECV_TASKS`：a=输出行起点、b=本核任务数、c=tokenStride、d=窗口偏移。b=0时返回是正常行为。
- `RECV_PROBE_BEFORE`：a=sample、b=扫描起点、c=本批数量、d=剩余任务数。`RECV_PROBE_AFTER`：a=sample、b=readyMask、c=keepFlagsMask、d=剩余任务数。多次b=0表明本次批量探测未发现完整packet；一次b=0不能证明永久缺包。
- `RECV_SLOT`：a=本核稳定任务编号j、b=物理position、c=槽地址、d=packet块标志归约的FP32原始位模式。需按`common/ep_packet_receive_layout.h`中的`PacketReceiveLayout::Decode`解释，不能当作普通计数ready的0/1。每批只打印前ITEMS个槽，它们不一定全是缺失包；结合readyMask相应bit判断。
- `RECV_LOAD_BEGIN/ENQUEUED`：a=issued序号；BEGIN的b=j、c=lane。缺ENQUEUED时检查`IssueLoad`的旧lane MTE3→MTE2等待与加载下发。
- `RECV_CONSUME_BEGIN/ENQUEUED`：a=consumed序号；BEGIN的b=j、c=lane。内部`RECV_PAYLOAD_WAIT/READY`的a=j、b=lane、c=keepFlags，包围MTE2→S等待。READY之后仍卡住，检查输出MTE2→MTE3依赖、metadata/输出写入及relay。
- `RECV_RELAY_BEFORE`：a=j、b=原sourceRank、c=转发dstRank、d=目标地址；`RECV_RELAY_ENQUEUED`：a=j、b=dstRank、c=expertId、d=q。只在FIRST_HOP分支执行；单机16卡同组DIRECT没有relay日志是正常情况。
- `RECV_DRAIN_BEGIN/DONE`包围最终lane等待，字段a=consumed、b=lanes。`RECV_CLEAR_BEGIN/ENQUEUED`包围ClearFlags；其后粗粒度PAYLOAD_RECV_DONE之前还有原PIPE_ALL。

若需要从position反推来源，使用本配置的B和localExperts=E/world：`region=position/B`，`localExpert=region%localExperts`，`sourceRank=(region/localExperts+world-本rank)%world`，`q=position%B`。先用它找到来源rank和专家，再与SEND日志比对。

## 9. 打印本身的边界

使用CANN原生`AscendC::printf`；开启宏但设置`ASCENDC_DUMP=0`会被配置头文件拒绝编译，避免静默无日志。不需要修改Host建立新的通信通道。

打印会扰动时序、使用SDK诊断资源，不能作为性能测量。CANN的日志缓冲、异步收集及进程超时可能让日志延迟、丢失；**没有日志不能单独证明kernel没执行，最后可见的日志也不保证就是实际最后执行点。** 若完全无LLD，先确认新wheel已加载、筛选匹配、打印构建生效及stdout/stderr已收集，再结合已有Host trace/profiler确认是否进入该设备task。

本补丁没有新增跨核屏障、强制DMA完成等待或等待超时，不自动跳过缺失数据。BEFORE/DONE之间沿用原同步逻辑；ENQUEUED始终仅代表指令下发。若必须得到设备硬挂时仍能实时读取的进度缓冲，需要额外Host采集机制，不属于本次“仅kernel+文档”范围。


## 10. 本次验证范围

- CANN 9.3.0 / dav-3510：默认关闭、全部开启、STAGE/COUNTS/PREFIX/SEND/RECV分别独立开启，共七种生产dispatch编译配置通过。
- 默认关闭配置的完整生产wheel构建安装通过；关闭目标文件不含`[LLD]`格式，开启目标文件包含打印格式。现有LL协议及测试入口CPU回归29项通过。
- A5仅设备0的独立打印烟测通过，实际输出`[LLD] r=0 c=0 PRINT_SMOKE`及`[LLD] r=0 c=32 PRINT_SMOKE`，同步返回0；这验证了CANN打印通道及默认筛选，没有运行EP16 dispatch，也不证明卡死时日志一定实时刷新。
- 编译日志在工程内`results/dispatch-debug-build`，打印烟测日志在`results/dispatch-debug-smoke`；这些生成文件不提交。目标机实际卡死阶段需由用户用STAGE第一轮结果确定。
