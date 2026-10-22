# NDR 跨机 count_recv 分段步长读取

基线 `bfdd043`，优化范围为packed count frame接收端，SHMEM及计数wire协议不变。
原WaitDispatch每轮为每条逻辑计数重新计算CountFrame并提交一次32B DMA。

固定目的rank/本地专家时，相邻来源rank的body和commit地址间隔均为
`localExperts * 512B`。将每个接收核的连续expert-major逻辑分片按专家行拆段，
在进入轮询前只生成一次计划；每段用strided DataCopy读入连续UB记录。
DataCopyParams源gap是`localExperts*16-1`个32B块，目的gap为0。
commit轮询和commit完成后的body快照共用计划，UB逻辑顺序保持不变。

E<=1024、world>=32、接收核数>=8，单核最多128条计数，最多跨5个专家行段。
计划占固定Scalar状态，不新增TPipe UB缓冲。非法范围明确trap。
EP32/E384/56核时，28个接收核每轮共38次DMA（原384），单核1～2次；body读取同样减少。
DMA字节数不变，计数发送仍是现有帧的78次跨机put/源rank，不在此次调整。

仍使用原MTE2_V、ReduceSum和V_S依赖；所有commit确认后才读body。
不跳过零计数，不提前清理共享commit；清理仍在GetCumSum确认全部接收快照之后。
单机EP16和MTE对照路径不变。通信协议7、stage schema3均不变。

## 验证和测量

实际读取helper在CPU C++替身中对照原逐记录地址，覆盖229680个分片、
EP32/64/128/256、E至1024、实际AIV16..72、不同count核数、目的rank、空分片和边界。
commit/body逐字节一致，UB尾部哨兵保持，5段最大值和越界拒绝验证通过。
另回归计数body/commit发布及dispatch批处理。CPU不能替代CANN编译/设备验收。

所有rank更新完整wheel，无需修改/重建SHMEM。使用同shape/路由/seed/AIV/Exp，
分别对比timer OFF的dispatch kernel和timer ON的count_receive阶段。
count_receive包含等待远端CountSend/RDMA quiet、快照及局部prefix准备，
不是纯MTE2开销；提交数减少不能直接等同阶段加速。如仍耗时高，应结合各rank
count_send/首发准备和实际计数到达，判断是否在等最慢来源。
