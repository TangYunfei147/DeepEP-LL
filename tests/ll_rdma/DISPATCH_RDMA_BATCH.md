# Dispatch 跨机 packet 批处理

2026-10-27：当前协议10已由[SO整包发布](ROCE_SO_PACKET.md)替代首gate两阶段发布。
下文保留旧批处理实现的历史设计与对照，不作为当前gate/quiet数量依据。

范围：NDR dispatch sender的FIRST_HOP/DIRECT。计数Records、接收relay、combine、
SHMEM源码/SDK、QP数量、包格式、路由及q槽位不变。基线提交为74e7160。

## 行为

原Packet每包完成body quiet、flag quiet后才返回。新sender用
`ll_dispatch_rdma_batch.h`，在每实际AIV已有128KiB对称暂存中放置独立GM packet槽：
`capacity = min(8, floor(128KiB / (tokenStride+32)))`。不新增SDK内存、不改变最终wire格式。

每个远端packet先UB→独立GM槽，另保存原首个32B flag，等待MTE3_S后把GM body首flag清零，
完成DCCI/DDR后一次put整个tokenStride。接收侧完整flag归约在首块为0时不会就绪。
此时返回打包下一条，NIC只读GM，不再持有UB lane；SDK可借用已暂存packet的前128B，
下一次sender使用lane会重写全部body，因此无需恢复前128B。
满批时先对每个不同peer quiet一次，保证所有body完成；再每packet只put保存的首flag，
最后每个不同peer quiet一次，完成后才清空描述符和复用暂存槽。
这保留Compact=1及FIRST_HOP KeepFlags=1.5，不依赖整包原子可见性。
仅128KiB极限单包无附加flag空间时使用原同步Packet，不缩小原helper输入域。
所有调用均使用本核原QP和事件7，不假设跨QP/peer排序。

尾批必须在sender返回前完成。尾批借UB lane作为scratch之前先PIPE_ALL，
确保此前机内MTE发送不再读取该lane；随后完整flush。普通机内MTE仍走原Packet。
无通信试验宏不进入批处理，沿用原试验路径；其他任务的NO_COMM本地改动不纳入本次提交。

内存生命周期关键点：满批flush在当前packet已暂存之后执行，不能在暂存前执行，
否则scratch会破坏刚打包的payload；flag源同样保留到其quiet结束。
SDK CQ失败不回退、不发布后续flag、不回收未完成槽位。

## 预期收益与限制

一批N个packet、P个不同peer，显式quiet从2N降为2P；例如8包同peer由16次降为2次。
即使peer不同，先提交多个body也允许多个peer传输重叠。2026-10-25进一步将每包
`2*blockCount`次put减为2次，网络字节仅增加32B gate；最终packet布局及目标不变。
SDK仍可因SQ回压内部等待，不宣称完全无阻塞。

首批flag会等到批满或sender结束，可能增加首包可见延迟；收益必须看完整dispatch，
不能仅按quiet数量推算。B1/K8且28发送核时，非去重最多8个核各处理1条路由，
几乎没有跨包合批空间；不要预期此改动解决小B计数通信开销。
同peer批量、多路由/较大B更可能受益。保持现有核数和任务分片，不强行减核。

timer沿用原schema：stage/body_post仍逐包记录，body_wait/flag_post/flag_wait按批记录；
尾批计入SendPayload/SendIssue，sender不再产生RdmaRestore。累计calls定义变化，
不能把新旧逐次阶段均值直接当整体加速。正式性能必须timer OFF。

## 部署与A/B

全rank更新完整wheel即可，无需修改或重建原SHMEM SDK。
上句仅针对本发送协议；若同时升级2026-10-25官方SHMEM主线，必须按
[SHMEM_MASTER_UPDATE.md](SHMEM_MASTER_UPDATE.md)重建SDK，不能复用旧版本。
同样的B/M/H/K/E、seed、dtype、AIV、Exp和采样规则下比较74e7160与本提交。
先BF16/FP8正确性，覆盖M32非去重和M256去重；再性能并保留所有rank原始task时间。
覆盖B1/8/16/32、稀疏与hotspot路由、多次调用、尾批和不同peer；不得将CPU模型当设备验收。

测试使用实际批处理helper和异步CPU传输替身：NIC延迟读取GM源、逆序完成、立即覆盖UB、
满批/尾批、capacity1/2/8、多peer、跨QP隔离和CQ失败。SHMEM未修改。
大body模型额外覆盖分片先可见、首gate逐word到达、两种marker、原UB不变、
body/gate CQ失败保留pending，以及128KiB边界。当前优化与27cf73c做同SDK对照。
