> 历史指南：旧打点及入口适用于原提交（最后完整旧实现e4d114c）。当前已删除旧插桩/解析；构建与采样以[reference schema7指南](../ll_reference_profile/README.md)为准。下文算法与历史结果保留。

## 2026-10-22：R0整数向量前缀

当前配置标识19000+S（取代下文历史17000/18000），Stage3/schema12、wire与epoch协议不变。
384项分成6个64项寄存器块，每块6级整数gather/add扫描，段间carry在寄存器中传递。
不使用FP32转换，不再384次GetValue/SetValue。输入缓冲按256B补齐，最后一段掩码存储，
不读越界、不写尾部。E384原缓冲已满足对齐，UB总量和lane不变；其他长度最多多224B。
继续保留V_MTE3及前缀写入完成后发布ready的顺序，不能删除等待冒充优化。

全rank完整重装wheel，无需仅因此重建SDK。对比BUILD_RECV_CUMSUM及整体OFF kernel；
WAIT_RECV_COUNTS是另一个成本，不会被这项修改消除。
本地51项相关检查通过；后续标识检查另验。A3_163_docker连接失败，未进入远端编译。
尚无CANN/设备验证或性能收益结论。

实现依据：asc-devkit 9.2.0@4d9fe40502b3423a344d570fd1a8be343cfca4a5的公开
asc_gather/asc_arange/asc_select/整数add接口。高层CumSum只支持half/float，
因此没有直接用于整数计数。A5寄存器接口不支持A3运行。

当前增量：非去重M<=32恢复S发送全部路由、R专职接收，协议18000+S。
去重M>32保持4ec82a5的原始路径。下文所有核共享接收仅适用于去重。
非去重R核仍负责SEND_COUNTS和R0前缀，不建本组发送描述符；S不等待前缀。
发送顺序恢复主线main@07af8ea非去重实现（保留其8-rank分组排序阈值），核数采用当前编译S。
Host与Device接收预算按R核数核算。230KiB、R0前缀、现有输入预取和lane上限不回退。
这次不是恢复主线全部实现，不包含q预计算或combine修改。
ST能力记录额外nondedup_dispatch_recv_cores=R，dispatch_recv_cores保留去重容量56；离线inspect_dispatch_routes.py仍是旧S3
所有权模型，不能用于当前非去重分核/时序结论。原始timer头的核容量也不等于接收参与核数。
需全rank重装完整wheel，SDK无需仅因此重建。

# R0共享前缀与接收任务表向量化

分支kernel/ll-g4-group-compact。本次在一个提交中撤销613ae81、a5ce3f7的
紧凑任务重分配/共享任务表实验，恢复S3/ac876ba发送算法，再叠加以下三项。
不改写历史，不修改kernel/ll-g4-s3-baseline工作区。

当前identity=g4-r0-prefix，协议18000+S，Stage3/schema12，默认S28/R28、56核、G4。
所有rank必须完整重装相同wheel。

## 三项修改

1. SEND_COUNTS只由原R核执行。每个R核保留原本组专家分片，同时承担远端专家
   分片计数，最多形成两个远端区间加一个本组区间。原S核只初始化发送scratch，
   不扫描计数、不发送count记录。本组payload初发归属不变。
   R0（物理核S）接收所有专家×来源rank计数，计算全表前缀，只写一份共享表。
   其他核等待R0发布后读取同一表；删除R×R片段和交换、逐核前缀副本广播及消费清flag。
   保留现有控制区预留布局，combine与原始payload wire不变。
   dispatch UB规划预算统一为230*1024B；combine各阶段原有独立预算未调整。
2. pendingIdx使用CreateVecIndex一次向量初始化，随后V_S完成后才允许标量读取。
3. tokenPos按记录段生成等差序列：32B对齐的中段使用CreateVecIndex，
   未对齐头尾各至多7项保留SetValue。避免对任意UB偏移使用向量指令。
   空分片不分配/不发向量；pending的后续swap-remove仍为标量。

## 协议及生命周期

R0的发布位位于其固定512B magic槽+32，原word0仍由RotateDispatchServerDedupMagic独占。
初值为0，各dispatch交替发布1/0，不随shape或bank改变位置。
先完成前缀与输入计数flag清理，再发布epoch；消费者只读、不清理共享epoch。
共享前缀和epoch视图禁用L2。无新的SyncAll；R0仍有必要的MTE3完成依赖。
同Session沿用同流串行调用，不支持并发复用控制区。
回退路径在Reset前排空，再从同一GM前缀读取；不能重新使用旧receiver副本偏移。
R0会集中承担前缀轮询/扫描，其他R核多承担远端count工作，不能保证所有shape收益。

## UB和计时

Host最低预算、R计数记录/本组描述符峰值、R0全表轮询缓冲及overlap预算一起更新。
230KiB是软件预算，不是硬件可用容量的设备验证结论。
原发送q计算、发送顺序、G4首跳/relay和combine算法没有引入额外优化。
schema12保持；POLL_CORE_SUMS等删除路径的阶段不再有样本，不补零。
CAL_CUMSUM只在R0有实际工作，SEND_COUNTS只在R核出现。
计时阶段位置变化，不能把同编号核的阶段耗时比当作完整kernel加速比。

## 获取和验证

在已有且干净的候选checkout中依次执行：

    git fetch origin kernel/ll-g4-group-compact
    git merge --ff-only FETCH_HEAD
    source scripts/ll_timer_env.sh
    source .venv-ll-timer/bin/activate
    export DEEPEP_NPU_ARCH=Ascend950
    export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
    export DEEPEP_LL_STAGE_TIMER=OFF
    unset SHMEM_ROOT CATLASS_ROOT
    python scripts/build_shmem.py
    python -m pip install -r requirements-build.txt
    python -m pip install -v --force-reinstall --no-build-isolation --no-deps .

已校验SDK可以复用，不因本次修改强制重建。首次部署遵循DEPLOYMENT.md依赖流程。

沿用原EP16、B=M256/H3584/K8/E384/FP8命令，换新run-name，先检查完整dispatch+combine
多seed正确性，再测OFF profiler（10预热/100采集/尾50）。
ON分析观察SEND_COUNTS、CAL_CUMSUM、PREFIX_COMPLETE、BUILD_EXPERT_TASKS、KERNEL，
不要只比较发送阶段。目标性能测试由用户执行。

CPU检查：

    python -m pytest tests/ll_stage_timer tests/elastic -q
    python -m pytest tests/ll_stage_timer/test_r0_prefix.py tests/elastic/test_ll_group_local_send.py tests/ll_stage_timer/test_dispatch_variants.py tests/ll_stage_timer/test_overlap_receive.py::test_native_layout_matrix -q

原生CPU用真实receive_tasks头配合延迟vector模拟，校验对齐、空段、跨记录段及V_S依赖；
R计数计划检查每专家恰好一个R写者，保留本组初发归属。均不能代替AscendC/NPU验收。
本版未做CANN编译、完整wheel链接或设备验证。冷编译ON/OFF方法见docs/BUILD_DISPATCH_SPLIT.md。
