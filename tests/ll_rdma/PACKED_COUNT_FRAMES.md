# 跨机计数帧合并（第二项提交，协议7）

在第一项小batch G16去重提交之上，合并计数通信；不删零计数、不修改SHMEM。
适用于NDR rdma模式且world>16。单机EP16和显式MTE对照沿用原计数协议。

## 发送

保留原按专家区间分配count核的分工、直方图/逐专家计数和UB预算。
每个count核负责的专家区间与目标rank区间相交形成一帧。
同一帧连续存储32B计数记录：一个连续body put，quiet，单独32B commit put，quiet。
零计数记录随帧发送，接收端仍能判断所有来源完成。
每个源rank在EP32/E384/56核时，跨机专家记录为192个，旧版384次put，
新帧为39个、78次put。核心数不同交界片段数不同，不能把78写成固定值。
若每帧只有一个专家记录，put数量不下降；此方案不宣称所有shape都更快。

## 布局与接收

总计数区域仍是E×512B，不新增对称内存或修改SDK。
物理布局为source-major分帧：连续body存放于帧首，commit位于该帧最后32B。
不同source/发送core拥有互不重叠的帧。body中的旧ready位不作为完成依据，
只有独立commit发布后才能读count。机内peer在此模式也使用同一布局并保持先body后commit。

接收核的逻辑分工仍是“本地专家×来源rank”，映射后读取帧commit，再读取对应记录。
sendCounts、prefix、expertTokenNums和payload接收器的逻辑顺序不变。
共享commit可能被多个接收核引用，不能由最早完成者直接清理。
本次先让全部接收核完成快照并发布原有cumsum总数，再通过既有GetCumSum确认，
每帧首逻辑记录的唯一owner清commit；每个body记录由其owner清零，防止后续shape变化
把旧body ready当成新的commit。清理源保持到原Finish的PIPE_ALL完成。

代价：接收端增加source-major映射和分散MTE2读取，换取更少的跨机WQE/quiet。
需要用目标机测量总dispatch延时，不能按提交次数比例宣称加速。

## 部署与验证

核心协议7拒绝旧布局Host/Device混装，所有rank更新完整wheel；SDK不改、不重建。
第一项独立提交为协议6，可单独checkout验证B<32去重；本项同时包含第一项行为。
stage schema3/expert schema1不变，专家诊断支持匹配新版协议。

CPU C++测试覆盖不同world/E/countCores下帧分区、逆映射、不重叠、边界、
实际传输body完成后commit、零计数、失败不发布commit以及多接收核共享commit。
本机未做CANN编译或NPU测试。目标机先B1/31/32、M32/M256、BF16/FP8、
normal/hotspot正确性和重复调用，再固定采样协议对比两次提交；保留全部rank日志。
