当前VF播种版本叠加在Scalar q表提交af46122上，作为一个独立提交。
A/B请以af46122为A、当前VF提交为B；原始无q表基线仍为f7c7848。
用户报告Scalar q表比原始基线劣化约10%，此次旨在隔离前缀播种的影响，尚无性能结论。

## VF播种更新

前缀[0,begin)调用已有CountFrequency<true/false>，384专家按128-bin窗口处理。
VF初始化/累加直方图；返回后一次V_S确保结果可被Scalar访问，再调用LlFinishLocalQ
顺序更新本片。不再逐条Scalar扫描前缀。空分片直接跳过，零长度前缀仍由VF初始化零表。
输入routeIds已有256B padding，末向量由有效掩码排除本片及padding数据。
32768项以上选择既有分段累计路径，避免uint16频数溢出。
完整q表、histogram容量、Host和overlap预算均不变，收发顺序/通信协议18000+S不变。
新增low_latency_q_seed_vf=True用于候选ST核对，保持全rank一致。

PREPARE_INPUT包含VF调用、S_V/V_S及本片Scalar更新；CAL_TOKEN仍只查表。
E384的现有helper会按三个bin窗口扫描前缀，不能当作只读一遍输入或保证性能提升。
A5支持此API，A3不支持，CPU mock仅验证运算和掩码。

# 本核q表独立A/B候选

分支kernel/ll-g4-local-q-table，基线kernel/ll-g4-group-compact@f7c784868522742f052c74e0bd8a6f5a5ac961b8。
唯一算法变化为发送q预计算。通信协议18000+S、wire、Stage3/schema12、收发顺序、R0前缀、
R核本组初发、relay、combine均沿用基线。以下为首轮Scalar q表说明；当前已新增独立VF播种提交，以上方更新为准。

## 实现

每个S核先播种原始路由[0,begin)的专家直方图，再按原始顺序生成本片q。
去重begin=startToken*K、count=tokenCount*K；非去重用真实flattened-route分片，
不能拿token分片索引非去重q。保留index<K时q=0的原语义。
原SendWithDedup/SendWithoutDedup/FillTail调用点不变，CalTokenSendExpertCnt改为
qTable[index-begin]读取。FIRST_HOP所有成员均在本核token分片内。
R本组初发已使用计数阶段生成的q，不重复构建表。

qWork原缓冲成为持久q表，qSub用于播种直方图；内部量化仅用qDst，不覆盖q表。
q表活到EndInputPrefetch完成，并保持原有UB退休时机。无新增GM读写或跨核同步；
读取已有routeIds，沿用原输入DMA就绪依赖。没有修改packet或计数发布顺序。

保留原qDst/qWork容量以减小A/B变量；qSub容量为max(原scratchBytes,align(E*4,32))。
Host最低UB和overlap规划同步增加小B/大E时可能需要的空间。
B256/K8/E384时原qSub已有8192B，直方图仅需1536B；qTable最大80项/320B，
规划总UB不增加，EP16/H3584预量化FP8/S28R28的S核105792B、6/6lane，无回退。
小B可能需要扩容qSub，不能把“所有shape零UB增量”作为结论。

每rank B256/K8/S28播种28032项、本片合计2048项，共30080项，
另有各活跃S核的E项直方图清零。操作量不是时延；Scalar随机直方图更新的代价需A5实测。
没有继续增加跨核共享计划、组间重排或q的GM交换。

## 身份与计时

通信协议保持18000+S，implementation保持g4-r0-prefix。
新增只读能力low_latency_q_table_enabled=True，候选ST要求此标记，
并把local_q_table加入rank能力一致性比较，防止误用旧基线wheel。
请完整安装，不能手工混用Host/device库。基线运行用基线checkout及wheel。
安装后可打印native.low_latency_q_table_enabled核对。

预计算计入PREPARE_INPUT，CAL_TOKEN现在仅查表，schema12保持。
CAL_TOKEN大幅缩短不等于整体加速，必须同时看PREPARE_INPUT与完整OFF kernel。
不要用不同核/轮的p50相减。无需为纯q算法变化强制重建已校验SDK。

## 测试与A/B

CPU生产helper对照独立全路由直方图，覆盖B1/3/8/32/48/256、K1/3/8/32、
E32/384/1024、S1/10/28/55、去重/非去重及重复专家的首token兼容语义。
检查乱序读取q、空片段、原route连续覆盖；原生布局检查三种输入模式及小B大E。

源代码核对：SendWithDedup、SendWithoutDedup、FillTail、SendPacket、IssueInput、
ConsumePacket、ProcessGroupLocal七个方法与基线逐字相同；主调度、counts、receiver、
core protocol与基线一致。CPU不验证真实AscendC调用域、DMA排序和设备性能。

获取候选：

    git clone --single-branch --branch kernel/ll-g4-local-q-table https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-g4-local-q-table
    cd DeepEP-LL-g4-local-q-table

按现有环境脚本/工程venv和锁定SDK流程安装，生产Ascend950和原S/R保持；
完整OFF wheel用于性能，全rank同提交；单独ON wheel用于归因。
A版本固定f7c7848，B版本取该候选提交。相同路由seed、dtype/scale、M、S/R、采样。
优先EP16 B=M256 H3584 K8 E384 FP8 do_cpu_sync=0，再覆盖BF16、内部量化及M<=32。
先dispatch+combine多seed正确性，后10预热/100采集/尾50正式OFF profiler。
使用不同run-name，保留原始task/日志，不能将ON数据混入OFF主表。

本次未执行CANN编译、ON/OFF cold build、完整wheel链接或NPU测试。
编译验证流程仍见docs/BUILD_DISPATCH_SPLIT.md；CPU通过不代表A5正确性或性能通过。
