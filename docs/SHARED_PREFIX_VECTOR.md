# R核计数、共享前缀及接收任务向量化

## 提交1：R核承担全部源端计数，UB预算230KiB

S核不再扫描/统计路由或发布count，仅保留发送scratch初始化。
R核分别均分本组与远端专家：本组分工与原来一致，继续生成稳定q和本组发送描述符；
远端专家分配给同一批R核统计。每专家唯一生产者，包括零计数，协议不遗漏ready。
每核最多一个本组区间和两个远端区间，count记录连续，且本组区间在最前。
R核的count记录UB峰值、局部描述符保留与发送/接收布局预算使用同一分工公式。

`kEpServerDedupCountUbBytes=230*1024`，Host检查和Device布局共用。
不修改含义不同的`UB_SINGLE_DMA_SIZE_MAX`，不增加事件上限（overlap仍最多6个lane）。
此预算是用户指定目标配置；编译通过不代替目标设备UB容量及最大合法shape验证。
S+R仍56，combine分工不变。协议15000+S拒绝旧预算/计数分工wheel混装。
timer schema14保持；S核不再记录SEND_COUNTS，R核记录真实统计/提交范围。
S核提前提交的数据仍等待接收准备后消费，count数量/原始q不依赖payload先后顺序。

前置提交保留原56份前缀发布，以便独立对照计数迁移和UB预算的影响。
可能的负收益是R核计数更重、前缀发布更晚；需同时观察SEND_COUNTS、WAIT_RECV_COUNTS、
首次接收和OFF整体耗时，不能仅凭S核开始更早判定收益。

## 提交2：单份共享GM前缀，可选parallel/r0

编译环境变量`DEEPEP_LL_PREFIX_MODE=parallel`（默认）或`r0`。
这是构建选项，不是运行时开关。Python构建指纹、Host CMake、Device CMake、
直接编译检查同步传入；全rank必须重装同模式完整wheel。
parallel协议16000+S，r0协议17000+S，Host/device及ST全rank检查拒绝混装。
ST结果记录prefix_mode；可跨模式比较，但保留各自模式，不自动调整。

parallel：R核按8条记录/32B整块均分前缀，避免不同生产核写同一GM块。
每核计算局部前缀、汇总片段总数、加偏移后只写共享表一次。
active生产核可能少于R（小E场景），非生产R核仍参与源端计数、本组发送和接收。
r0：原接收组编号0（物理核S）独自收集全部count、生成整表；
其他R核不收集目标count或生成前缀，只读共享表；所有R核仍做源端SEND_COUNTS。
跳过跨R核片段总数发布/轮询。R0最多1024条count，超过255时使用压紧ready后长计数Sum，
不向基础ReduceSum的8位repeat参数传入溢出长度。

两种模式均只保留一份GM前缀，全部56核各自加载到私有UB。
ready仍为每消费核独占的一行，消费者只清自己的ready；保持原发布顺序和源缓冲复用依赖。
没有新增全核barrier。串行fallback也改读同一张表，不能保留旧副本偏移。
共享表在本次kernel消费者完成前不复写；仍遵守原双窗口magic和同stream调用序列。
并行模式减少MTE3发布工作，R0同时减少片段汇总协调，但可能集中形成慢核；
性能必须由目标机两模式OFF对照确认。

打点schema14/ID不变，BROADCAST_PREFIX现在表示共享片段发布；r0不产生GET_CUMSUM等
跨生产核汇总阶段。未出现的阶段不得按旧算法解释为采集丢失。

```bash
export DEEPEP_LL_PREFIX_MODE=parallel  # 或 r0
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

保持原S/R配置、SDK及ST参数；每模式用不同run-name，OFF正式性能和ON诊断分开。

## 提交3：接收任务初始化及相似循环

- pendingIdx：已有32B对齐缓冲用int32视图CreateVecIndex生成0..count-1。
  Host/Device UB预算约束使count远小于INT32_MAX；不新增任务数组或临时UB。
- tokenPos：每记录段保持原region、q和分核边界。段长不足32用标量，
  长段先标量处理对齐头，再CreateVecIndex写32B整块主体，标量处理尾部。
  相邻记录的scalar边界和vector主体不共享32B块；超int32值域保留uint32标量路径。
- Build结束只做一次必要V_S依赖，覆盖两张表；后续GetValue和RemovePending才能读取/改写。
  空任务不分配、不发射、不增加该依赖。它是本核数据依赖，不是全核/跨卡屏障。
- 同类审查新增前缀统一offset修正：offset=0直接跳过，非零且片段>=32用Adds，
  短片段仍标量。保留S_V及V_MTE3依赖，避免scalar生成前缀与vector改写、MTE3读取竞争。

二分upper_bound完全保留，严格prefix>begin，仍正确跨越零记录和刚好相等的边界。
32元素是保守的向量发射粒度策略，不按B256/E384特化，不宣称短段向量化必然加速。
新增V_S成本与减少的scalar写需要在OFF目标测量中一起评估。

其他已检查的循环：

|位置|处理结论|
|---|---|
|GatherSumRecvCnt局部inclusive prefix|存在前项依赖，不用简单CreateVecIndex/Adds替代；专门scan算法另行评估|
|BuildLocalTasks稳定q直方图更新|同专家重复更新有依赖，不能直接并行scatter|
|histogram→32B count记录散写|目标count字段偏移4B且需保留ready；不直接套用对齐Copy|
|FinishPrefixLoad专家数量输出|跨rank步长抽取、int64转换和差分，小表不新增gather临时区|
|RemovePending交换删除|是有状态的待处理队列更新，不是初始化序列，保留scalar|
|源端权重复制、路由索引构造|已有向量Copy/CreateVecIndex，不重复改造|

打点仍schema14。BUILD_EXPERT_TASKS全量覆盖新初始化及必要依赖；
BUILD_RECV_CUMSUM覆盖offset修正。没有增加逐包探针，没有改变任务/包/relay顺序。

## 目标机验收与回退

先`git fetch origin`。按以下顺序切换，每次在全部rank重装匹配wheel，保持相同S/R、
shape、seed和路由。切换模式需新进程/新通信域，不混用仍运行的旧wheel。

|版本|Git引用|prefix编译变量|
|---|---|---|
|原始诊断基线|e61d213|旧实现，不读取该变量|
|R核计数+230KiB|b2b2ed3|旧实现，不读取该变量|
|共享前缀|0888f34|parallel、r0分别构建|
|共享前缀+向量化|本指南所在的向量化提交|parallel、r0分别构建|

例如先验证最终版本：

```bash
git switch experiment/ll-g4-overlap-ab
git pull --ff-only origin experiment/ll-g4-overlap-ab
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_PREFIX_MODE=parallel  # 下一轮改为r0并重新安装
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .

python tests/elastic/test_ep_low_latency.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 256 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
  --dispatch-only --skip-perf-test \
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe \
  --output-dir "$PWD/results/shared_prefix_vector" --run-name "$RUN_NAME"
```

RUN_NAME按提交/模式设置唯一名称。解析仍用tests/ll_stage_timer/analyze.py。
正式性能比较将timer编译为OFF、重装同提交，去掉上面的capture和skip-perf-test参数，
保持原正式预热/采样/Exp协议。compile-only验证机不代跑这些设备命令。

正确性边界至少覆盖：B=1/8/256，E=16/384/1024，K=1/8，BF16与FP8，
原测试支持的随机及热点路由；避免按示例shape特化。E1024尤其覆盖R0长ready归约。
接收任务表CPU测试另外覆盖全零、连续零计数、单记录热点、所有8种字偏移、短段/长段、
uint32高位值的标量回退、每核切片和pending交换删除。

对比关注：SEND_COUNTS、WAIT_RECV_COUNTS、BUILD_RECV_CUMSUM、BROADCAST_PREFIX、
PREFIX_READY_SPAN、BUILD_EXPERT_TASKS、RECV_PREPARE及首次接收。
记录实际SEND_LANES/RECV_LANES和SERIAL_FALLBACK：预算变化可能改变lane数，
R0模式的count-stage预算按最大生产片段保守估算，不把模式间差异都归为scalar计算变化。
诊断阶段是线索；OFF整体算子时间和正确性才是验收结果。

## 编译与CPU验证记录（2026-10-21）

授权A3_docker，CANN9.3.0/py311_tyf，保持Ascend950、O2、S28/R28、jobs4。
下表时间为独立空目录的dispatch（7个模式+launch bridge）编译墙钟秒数，不是算子耗时。
每一行还完成了完整ON wheel链接/打包；未安装wheel、未运行NPU、未重建SDK。

|代码快照|模式|A3 CPU/C++|ON秒|OFF秒|完整ON wheel|
|---|---|---:|---:|---:|---|
|b2b2ed3|前置/原前缀|371通过|82.162|50.528|通过|
|0888f34|parallel|379通过（含两模式分片模型）|82.833|51.262|通过|
|0888f34|r0|同上|79.471|49.522|通过|
|5af523e|parallel|381通过（含实际任务表/延迟向量写模型）|84.058|51.806|通过|
|5af523e|r0|同上|79.895|50.462|通过|

最终交付的向量化提交相对5af523e仅追加文档/维护记录，C++、Python和测试源码相同。
本地最终349通过、32项缺C++/POSIX环境跳过，已在A3补齐；目标命令CPU --plan通过。

远端checkout `/home/tyf/Q4_remote_test/DeepEP-LL-g4-overlap-recv` 中的证据：

- `.logs/recv-count-230-002/`：前置提交；对应`.artifacts/recv-count-230-002/`。
- `.logs/shared-prefix-modes-001/{parallel,r0}/`及上级cpu.log：共享前缀提交。
- `.logs/receive-vector-001/{parallel,r0}/`及上级cpu.log：向量化提交。
- 后两项wheel分别在同名`.artifacts`目录的parallel/r0子目录。

各模式含on/timing.json、off/timing.json、wheel.log和revision.txt。远端已恢复原分支。
初次前置检查的旧断言（14028协议、190KiB）失败保留在recv-count-230-001，
修正断言后完整重跑。未覆盖旧日志。编译结果不构成设备正确性或性能收益结论。
