# Dispatch包级诊断（schema14）

算法基于Stage5 `544140d`，本次只调整诊断，不改发送/接收调度、事件或同步。
独立S3基线为`kernel/ll-g4-s3-baseline`，精确指向
`ac876bab5c1f18155e9b47fc6446cc1d9bec7bc1`，未带入S4/S5或新诊断。
其他agent可在该分支上建立自己的实验分支；原基线保持不动。

## 一个包和一个token

一次发送任务提交一个通信包，包含一个源token的payload及路由元数据。
同一个源token可以有多个初发包：源端先按目标组去重，再发各组FIRST_HOP
或DIRECT；首跳收到后再转发。R核还初发本组DIRECT。因此初发包数不等于B，
接收包数也不是唯一源token数。组内同卡目标也产生工作量。

## 计时口径

|需求|阶段|边界与限制|
|---|---|---|
|发送前开销|INIT、STATIC_BEFORE_SEND|INIT单列；后者从Process开始到本核sender准备入口，包含count/prefix及其等待；未进入sender准备的核不记录该跨度|
|发送准备|PREPARE_INPUT|分配/初始化/路由输入准备，保留现有范围；counts的发送scratch准备也使用此ID，和发送前跨度可能重叠|
|每包发送|SEND_PACKET|消费已经预取的输入至提交本包，包含量化、打包、lane等待；不包含更早的IssueInput和路由任务生成|
|接收准备|WAIT_RECV_COUNTS、BUILD_RECV_CUMSUM、WAIT_CUMSUM、PREFIX_COMPLETE/LOAD_WAIT、RECV_PREPARE|分别反映等待count、构建前缀、等待ready、前缀处理/搬入等待、接收任务/缓冲准备；不能相加当总开销|
|每包接收处理|RECV_COPY、RECV_INPUT_WAIT|前者从ConsumeLoaded进入至输出提交，含relay和output子阶段；后者只包原有MTE2_S等待|
|结果拷贝|OUTPUT_ISSUE|payload、scale、输出元数据提交；排除relay循环；异步写完成由原lane退休/尾部等待保障|

网络远端写到达发生在接收核之外。RECV_POLL是检查就绪的成本，不能把它当单包网络
传输延迟。RECV_INPUT是本卡GM→UB提交；RECV_INPUT_WAIT是其消费依赖等待，含排队。
OUTPUT_ISSUE、TOKEN_SEND_ISSUE、RELAY_ISSUE是发起耗时，可能受发射背压影响，
不是远端确认或DMA完整耗时。无新增同步去测单包完成。

`PREFIX_LOAD`/`PREFIX_READY_SPAN`仍是可能包含有用交错工作的跨度。
`STATIC_BEFORE_SEND`也为跨度，不是独立可削减纯scalar成本。

## 降低探针成本

dispatch observe只对ID12/13、16..21和60..63测前两次；后续先判断已有样本数，
不再读时钟或累加这些细分时长。包括TOKEN_INPUT/QUANT/METADATA/BUFFER_WAIT/PACK/
SEND_ISSUE和RECV_INPUT/OUTPUT、CAL_TOKEN、LOCAL_WINDOW_COPY/FORMAT_OUTPUT等。
解析输出`coverage=first_two_only`，不对这些阶段计算全量阶段占比。
前两次偏向初始流水，不能外推后续稳定阶段。TOKEN_INPUT_WAIT/QUANT_WAIT及
RECV_BUFFER_WAIT/RECV_INPUT_WAIT保留全量，避免漏掉后续背压；input_wait报告
对schema14以全量SEND_PACKET次数核验等待次数，仍仅为启发式候选。

主要阶段、SEND_PACKET/RECV_COPY/OUTPUT_ISSUE及包数仍全量。
初发包数复用已有packets_，结束时一次登记，不每包新增独立计数更新。
收到转发包在改写tail前读取TERMINAL类型；仅timer启用时读取。
不加原子操作、UB packet缓冲或跨核/跨卡同步。
导出仍在原KERNEL结束后的SyncAll之后；每64B缓存行清理一次，代替每16B记录重复清理。
每核缓冲仍4608B，91个metric、192条trace容量；每阶段仍只显示前两次trace区间。
被跳过的细分调用不计入omitted；omitted仅指已计时但没有trace区间的调用。
combine和显式relay-complete保留全量细分语义；后者仍会人为排空，不用于性能基线。

ON仍有计时、数组及代码布局开销，不能宣称无扰动。诊断KERNEL不包含其后的SyncAll/GM导出，
profiler task包含这些成本。验证需同算法、shape、路由、S/R依次对照：

1. timer-OFF正式性能和正确性。
2. timer-ON不capture，观察编译布局/驻留成本。
3. timer-ON observe独立capture，观察探针+导出影响。

不要用旧算法OFF和新算法ON作“探针开销”对比，不把KERNEL与profiler task混作一个指标。

## 使用

全rank安装匹配完整ON wheel并重采；不需因诊断变化强制重建已校验SDK。
保留生产Ascend950/O2及7个独立模式TU；schema13与14不混采或算诊断阶段加速比。
沿用已有ST参数和`--ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe`。
在各rank环境加载原CANN/Python并保持原S/R编译配置后执行：

```bash
git fetch origin
git switch experiment/ll-g4-overlap-ab
git pull --ff-only origin experiment/ll-g4-overlap-ab
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

先用原ST做正确性/采样；OFF对照将DEEPEP_LL_STAGE_TIMER改为OFF，重装同提交，
运行原正式性能入口且去掉capture参数。不要混用同一结果目录，不安装本文A3编译产物到目标机。
解析入口不变：

```bash
python tests/ll_stage_timer/analyze.py results/CASE/RUN \
  --operation dispatch --output-dir results/CASE/RUN_analyze
```

新增`dispatch_packets.md`：每rank每核包数、阶段耗时、每包加权均值、慢核优化线索。
`dispatch_packet_samples.csv`保存逐轮整数包数；`dispatch_packet_cores.csv`保存跨轮汇总。
接收总包数=Direct+FirstHop+Terminal，接收转发包=Terminal；转发是实际relay写次数。
缺少新阶段的旧数据不补零，空任务没有每包时长；缺失时间显示未测/无任务。
没有额外源token位图去统计唯一token数，也未把所有包都当成跨卡链路流量。

优化线索按每rank/轮最慢核的全量阶段耗时排序，不使用细分的前两次样本推断占比。
不能据此算物理带宽利用率：必须进一步区分本卡/跨卡字节、链路、完成窗口与并发。
本次没有新增这些流量口径，报告不伪造利用率或确定的性能收益。

## 验证

最终代码快照`f7075ad1abb1d0566d6cd2cdddd5a76622fd62e6`（交付提交仅追加文档）：

- 本地CPU：309通过、13项缺C++/POSIX环境跳过；A3补齐后322通过。
- A3/CANN9.3.0/Ascend950/O2/S28R28/jobs4，独立空目录dispatch冷编译：
  ON81.602秒、OFF50.332秒，7种模式和launch bridge全部通过。
- 完整ON wheel编译/链接通过；没有安装wheel或运行NPU。
- 与544140d对照，dispatch发送/接收/入口的通信及同步调用序列保持不变。

证据在获授权远端checkout的`.logs/packet-timing-002/`：cpu.log、wheel.log、
on/timing.json、off/timing.json、revision.txt；wheel在`.artifacts/packet-timing-002/`。
远端已恢复原kernel/ll-g4-overlap-recv分支。
第一候选fe4e9fa的验证保留在packet-timing-001（321通过、ON81.497/OFF50.547秒、wheel通过）；
最终候选恢复了关键等待的全量计时，再次完整验证，未覆盖旧记录。

上述为编译耗时，不能代表算子速度或探针扰动降低幅度。
设备正确性、探针扰动和性能改善仍需目标机验证。
