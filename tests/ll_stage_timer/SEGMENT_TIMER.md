> 当前relay扩展为schema10，独立完成诊断与observe区别见 [RELAY_TIMING.md](RELAY_TIMING.md)；下文为早期参考格式记录。

> 本文下方记录schema7参考格式。当前实现已追加两处现有同步的计时，升级为schema9，详见 [INPUT_WAIT.md](INPUT_WAIT.md)。旧64个ID和采样语义保留，新格式用JSON/CSV/Chrome trace。

# 参考分段打点（schema7）

当前分支：`kernel/ll-g4-shared-recv`。参考版本：[`graph_dc_debug_seg@e58ece1f0b8cb8179607ac0300a22a3bdc0150b9`](https://gitcode.com/zhong-zixin/ops-transformer/tree/e58ece1f0b8cb8179607ac0300a22a3bdc0150b9)。参考公共记录器为`mc2/common/op_kernel/dc_seg_profiler.h`，插桩语义来自该版本Dispatch FullMesh与Combine MTE头文件。

本次替换诊断方式，保留当前LL的G4、错峰、去重/relay、发送后全核接收和combine协议。没有移植参考分支的通信算法或分核。

## 与参考实现一致的计时规则

1. Begin/End只读GetSystemCycle，不在阶段边界添加PIPE_ALL。只有业务原有等待进入对应区间；没有drain模式。
2. 核私有数组累计每阶段所有调用的cycles与calls；前两次调用保留真实begin/end。最多192条时间线记录（每区间两条）+64条累计记录，预留KERNEL的一对记录，满容量不循环覆盖。
3. 省略的逐次区间数明确记录。后续调用继续累计，不用首尾包络伪造完整流水图。
4. INIT单列；KERNEL从Process开始，到最后原有流水排空后结束。同一设备所有AIV结束KERNEL测量后执行SyncAll<true>()，然后才把诊断记录写出到GM。
5. 轮询次数和就绪token/slot数量用Count记录，cycles=0；它们不是耗时阶段。64个阶段编号与参考版本完全一致，未执行或当前算法没有的阶段保持缺失。
6. 嵌套父子阶段不可相加，计时包含原有等待和指令提交开销，不等同MTE/V引擎独占执行时间。跨rank时钟未校准。

写出及末尾诊断同步仍增加Host/profiler看到的算子总时间，因此正式性能必须使用OFF wheel；ON数据用于定位。代码不在每次探针处写诊断tensor；编译器对核私有数组的实际放置仍以目标编译结果为准。

## 当前LL插入位置

| 参考阶段 | 本分支对应代码/含义 |
|---|---|
| INIT / KERNEL / FINAL_DRAIN | dispatch/combine设备入口初始化；Process主体；现有最终排空 |
| SEND_TOKENS / PREPARE_INPUT | sender准备UB与路由、FIRST_HOP与DIRECT发送、源元数据写回 |
| CAL_TOKEN | 原始路由前缀计算q；不是专家总计数 |
| TOKEN_INPUT / TOKEN_QUANT / TOKEN_METADATA | token搬入及DeQue；实际MXFP8计算；包尾元数据 |
| TOKEN_BUFFER_WAIT / TOKEN_PACK / TOKEN_SEND_ISSUE / SEND_DRAIN | 发送lane复用等待；480B+32B打包及原V_MTE3依赖；DataCopy与完成事件提交；发送尾部lane归还 |
| SEND_COUNTS / COUNT_COMPUTE / COUNT_SEND_ISSUE | count路径及其计算、发布子阶段 |
| CAL_CUMSUM / WAIT_RECV_COUNTS | 原R核前缀路径；接收count等待和提取 |
| POLL_RECV_COUNTS / POLL_CORE_SUMS / WAIT_CUMSUM | 等count、核间总数、当前物理核的前缀ready，分别保留ITERATIONS |
| REDUCE_CORE_PREFIX / CLEAR_CORE_SUMS / BROADCAST_PREFIX / PUBLISH_PREFIX_FLAGS | 前缀偏移归约、总数槽清理、当前实际56份前缀写出、ready发布 |
| WRITE_COUNTS_OUTPUT | 原首个接收核写公开专家计数 |
| LOCAL_WINDOW_COPY / FORMAT_OUTPUT / BUILD_EXPERT_TASKS | 全56核接收路径、初始化/输出处理、静态任务表构建 |
| RECV_POLL / RECV_COPY / RECV_READY_TOKENS | 逐批探测；逐包消费；实际消费的包数（含首跳、转发到达，不是唯一输入token数） |
| RECV_BUFFER_WAIT / RECV_INPUT / RECV_OUTPUT / CLEAR_LOCAL_FLAGS | lane复用、预取/消费等待、输出和relay、清理本核槽位 |
| COMBINE_SEND / COMBINE_RECV | direct/partial回传；源token聚合 |
| COMBINE_INDEX_INPUT | 路由索引实际搬入与原依赖等待；过滤/分组调度留在PREPARE_INPUT或父阶段 |
| COMBINE_INPUT / COMBINE_BUFFER_WAIT | 输入搬入、消费等待和buffer复用 |
| COMBINE_PACK_QUANT / COMBINE_COPY_PACK / COMBINE_SEND_ISSUE | 打包父区间、实际打包、回传指令；当前BF16 combine无原生量化，COMBINE_QUANT留空 |
| COMBINE_RECV_POLL / COMBINE_READY_SLOTS / COMBINE_DEQUANT_REDUCE | 回传探测、有效贡献槽数、转换/加权归约 |
| COMBINE_PROCESS_TOKEN / COMBINE_CLEAR_FLAGS / COMBINE_SEND_FINISH | 逐任务处理、清理和尾部完成等待 |

由于当前LL接收采用预取/消费分离，RECV_INPUT及部分COMBINE_INPUT分别记录指令提交与后续消费等待，calls不等于token数。RECV_OUTPUT包含本地输出和FIRST_HOP relay；参考字典没有独立relay阶段，不把旧逐专家GM探针混入其中。参考的RMSNorm/共享专家等不存在的路径不生成记录。

BROADCAST_PREFIX在本分支确实写56份副本；参考分支最新算法写共享前缀。名字一致不意味着两份算法工作量一致。

## 载体、ABI与旧数据

沿用LL显式诊断tensor参数，Host校验dtype/shape/设备/512B对齐，不借用performance_info隐藏扩容。每核 `[72,8] int64`、4608B；56核每次252KiB。记录部分使用参考相同的16B event/payload/cycle结构，额外空间用于LL独立头和按核对齐。

核心标识为`10000+S`，阶段schema7。全rank重装完整匹配wheel；本改动不要求强制重建已校验SDK。`ll_stage_timer_legacy.py`只用于读取旧schema1/2/5/6；不以旧包生成新诊断。

`--ll-timer-mode drain`和`--ll-expert-timer`在新模式明确报错，Native也拒绝旧专家tensor。旧专家数据仍可离线读取。使用OFF wheel比较新旧提交的正式性能；新旧打点边界不同，不计算跨schema6/7的诊断耗时比。

## 构建、采集、解析

所有命令从本分支工程根目录运行。首次准备工程遵循[LL_KERNEL.md](../../LL_KERNEL.md)的子仓/SDK流程。已有匹配CANN/Python和工程内SDK时：

```bash
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=10
export DEEPEP_LL_RECV_CORES=46
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

保持之前目标机的设备、节点、shape配置，只需使用当前分支匹配wheel和以下诊断参数：

```text
--ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe
```

运行入口仍为`tests/ll_stage_timer/run.py`；默认10预热/100稳定调用后采5轮，每轮Exp+scalar all_reduce，延迟CPU复制与校验。原有正确性校验保留，不沿用参考脚本的精度SKIP。使用新run-name，不覆盖已有结果；完整目标运行示例见[README.md](README.md)。

```bash
python tests/ll_stage_timer/analyze.py results/ll-timer/segment-001 --trace-iteration 4
```

新增输出：

- `metrics.csv`：每rank/轮/核/阶段全部调用累计cycles、calls、均值；counter标记time_measured=false。
- `intervals.csv`：前两次实际区间，不包含未采样区间。
- `cores.csv`：记录数、角色、省略区间数。
- `stages.csv`/`summary.json`：跨轮统计；counter耗时列留空，次数保留。
- `trace.json`：所选轮的实际采样区间；每rank独立原点，INIT与KERNEL分开。
- 每个原始JSON旁的`dispatch_prof_rankN_sample_*.bin`或combine同名格式：参考dc_seg format5的body，原参考`dc_seg_profile.py`可直接读取。没有参考启用cookie，因为这是离线文件。

参考解析器一次只接受每op/rank一个文件；解析不同诊断轮时分开选择文件，不能把同rank多轮混为一轮。默认cycles/1000换算us来自已确认1GHz，不进行自动频率标定。`--show-repeated`仅影响旧格式的包络显示；新格式始终只显示实际采到的区间。

## 验证边界

剥离探针及诊断参数后，9个业务文件的有序token摘要与`6ea61a6`一致（忽略注释、空白及新增作用域花括号）；另检查Begin/End配对、阶段ID、测量边界无新增GM写或屏障、KERNEL结束后才SyncAll/Flush。这是静态证据，不代替编译或设备验证。

新格式测试覆盖前两次/全量累计、192+64容量、KERNEL保留、计数与时长区分、损坏/截断拒绝、延迟CPU复制、旧格式兼容和导出。满容量dispatch/combine模拟输出已由原参考解析器读取，各5376条采样区间、3584条metric、56核；该检查不代表真实NPU采样通过。

本地缺CANN和C++编译器，未执行设备编译或上板。`test_segment_native.py`在有g++时编译实际记录器的CPU mock，覆盖OFF无写入、ON延后写出与哨兵；真实设备仍需匹配wheel全rank验收。

schema7首次交付回归：315 passed、30项缺C++编译器跳过、9项Bash/POSIX检查未选；包含30项新增格式/插桩检查，另2项原生记录器mock因无g++跳过。原参考解析器交叉读取通过。未执行CANN或NPU。

## a192113本组初发后的打点适配

适配仅添加探针和更新分析标签，不改当前源端分工、接收器或combine。ProcessGroupLocal内补PREPARE_INPUT（输入/权重/量化暂存及lane准备）、SEND_DRAIN（lane归还、MTE3_S、scratch Reset和CTRL恢复）；无本组任务时仍不生成发送区间。LocalSendScratchPool别名池建立另记PREPARE_INPUT，发生在SEND_TOKENS调用前，不能把所有PREPARE_INPUT累计都当成其内部子区间。

非去重S核的本组过滤/远端索引压紧也单列PREPARE_INPUT。R核的q来自COUNT_COMPUTE融合扫描，不增加CAL_TOKEN或新的Scalar扫描。逐包TOKEN_*继续复用同一SendPacket探针。

原始格式头的cumsum是初始worker类，保持参考二进制兼容。LL解析器仅在该核实际出现SEND_TOKENS时显示send_and_recv，并在metrics/intervals/cores保留initial_role=cumsum；旧数据或空任务核不被误标。核心标识10000+S、schema7均不变，全rank重装最新ON wheel。

当前插桩业务核对基线为a192113；保留已更新的源端指纹及原接收/combine指纹，不再用6ea61a6声称本组初发算法未变。完整新命令见[README.md](README.md)。

跨轮汇总按initial_role保持同核同阶段唯一统计项；某轮有本组发送、其他轮为空时，不拆分KERNEL统计。汇总角色表示该采样集合中实际参与过发送，cores.csv仍逐轮保留角色。

本次a192113适配回归：357 passed、43项缺编译器/POSIX环境跳过、8项未选；含跨轮角色聚合和含空格目录迁移。B8/M256/H3584/K8/E384、BF16+FP8的EP16 CPU --plan通过。未执行CANN编译或NPU采样。新增准备/退休区间必须重装ON wheel并重采，旧数据不能补出缺失区间。
