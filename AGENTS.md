# 目标RoCE通信契约

2026-10-27用户明确提供：跨机RoCE每个WQE的尾片具有SO特性，保证该SO之前的数据
先到达，且片内保序；单片默认2KB。后续本项目通信设计以此目标硬件契约为依据。

优先设计同一WQE发送一个完整消息：数据/元数据在前，末尾追加一个32B FLAG，接收端
观察尾FLAG确认消息就绪。不把现有body/flag两阶段put+quiet视为唯一正确方案。
2KB是网络分片粒度，不是消息大小上限；不自行扩展到跨QP/独立WQE、MTE或其他平台。

SO不替代发送源缓冲生命周期、接收消费完成、旧FLAG清理或epoch/双bank复用证明。
修改前核对实际SDK WQE拆分/SO覆盖和尾FLAG布局，并同步接收端与协议版本；SDK保持
既有实现，不能为该假设擅改SDK。

协议10已实现跨机payload整包单WQE：复用现有packet最后32B flag，接收端保留完整flag
检查（包含尾flag）；内部wire/机内MTE不改。dispatch删除gate副本/清gate/独立gate put，
每批每peer一次quiet；combine Packet每包一次put+quiet。独立路由/计数commit不因单WQE
SO被删。全rank更新完整wheel；具体实现及验证边界见tests/ll_rdma/ROCE_SO_PACKET.md。

详细记录及现有实现区别见[ROCE_SO_CONTRACT.md](tests/ll_rdma/ROCE_SO_CONTRACT.md)。
