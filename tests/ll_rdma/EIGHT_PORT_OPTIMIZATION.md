# EP32 的 3 fullmesh + 4 Clos + 1 跨机网口优化

2026-10-25 用户确认每张 NPU 有八个网口：3 个连接同一 fullmesh 小组，
4 个用于机内 Clos，1 个用于跨机；fullmesh 小组为连续四 rank，
四组构成连续十六 rank 的物理机。本次 EP32 使用两台这样的机器。

机内仍通过 MTE peer 映射访问，跨机仍使用 HNS_1825 RDMA put。
本仓 kernel 选择目标 peer，实际物理端口由驱动/provider 路由；
不把 QP 数当作网口数，不把 3+4 个机内口计入跨机 RDMA 带宽。
SDK 官方 master 更新见 [SHMEM_MASTER_UPDATE.md](SHMEM_MASTER_UPDATE.md)。

## 三种不同的大小

1. LL 每个存储块为 512B：480B body + 32B flags。一个 token 包有多个块。
2. RDMA put 的消息长度是提交给 SDK 的字节数，可以超过 512B；
   HNS single-SGE WRITE 的 WQE 描述符为 64B，它描述的消息并不限于 64B。
3. 网络 MTU 是 SQ/provider 的配置，NIC 可将一个 RDMA 消息分成多个网络包。
   不能从 LL 的 512B 格式推断 MTU，也不能从 WQE 长度字段推断设备最大消息。

当前 wire 公式为 `tokenStride = ceil(payloadBytes / 480) * 512`。
`payloadBytes` 包括数据、scale、对齐和路由尾部，不能只按 hidden 元素数估计。

| H3584/K8 配置 | payloadBytes | 512B块数 | tokenStride | 旧 sender put/包 | 新 sender put/包 |
|---|---:|---:|---:|---:|---:|
| 预量化 FP8 | 3808 | 8 | 4096 | 16 | 2 |
| BF16 | 7264 | 16 | 8192 | 32 | 2 |

## 连续 body 与首 flag 发布

最终接收内存格式完全不变。每个 staging slot 额外保留原始首个 32B flag：

1. UB packet 和首 flag 分别落到注册 GM；源 UB 的 flags 不修改。
2. 退休 MTE 后把 GM body 的首 flag 块清零，完成 DCCI/DDR 可见性处理。
3. 整个 tokenStride 一次 put，接收者此时不能判定就绪。
4. 同批各 peer 的所有 body quiet 后，只 put 保存的首个 flag 块。
5. gate quiet 后才复用 slot；满批和尾批都遵守这个顺序。

接收端汇总全部 flag。首块为零时，即使其余块以任意顺序提前可见，
也不满足 Compact/KeepFlags 的就绪值。首块恢复前 body 已完成；
不依赖整包写原子性，也不改变 FIRST_HOP 的 1.5 标记。
仍依赖原协议的唯一写入者、槽清理和跨轮复用约束。

每包网络字节增加 32B，staging slot 为 `tokenStride+32`，
批容量 `min(8, floor(128KiB / slotBytes))`。常见 4/8KiB 包仍为 8 个槽。
极限单包 128KiB 没有附加 flag 空间时，保留原同步分块发送。
独立接收 relay、combine 的 Packet 路径不改。每核 QP 数不变。

timer 的 body_post/flag_post 仍记录实际调用区间；data counters 记录逻辑
packet handoff，额外 32B gate 属于控制重复流量，不应把这些计数当作 NIC 线速字节。
put 减少不等于网络 RTT 按相同比例减少：旧实现也会批量提交后统一 quiet。

## 路由表已按源卡整批发送

EP32 RDMA 且 `B*K<=8192` 时，每卡完整路由表是一次 `4*B*K` 字节 body put，
最大 32768B，另有 commit 4B。它位于独立控制区，不使用 token packet 格式。
不同源卡继续使用各自的跨机网口；不增加发送端机内汇聚到单一卡的阶段。

本次将 PostBody 与 PublishCommit 分开：提交后先做本源计数和机内计数发布，
再完成 body 并提交 commit。2026-10-26协议9删除ACK及本地done汇合，
每 rank 每轮为 2 次 put、2 次 quiet；最后quiet仅退休出站commit。
同 bank 复用依赖已有双窗口epoch及调用顺序；空核直接返回，空 core0 仍负责完整网络进度。
无新增 Host collective 或全核 SyncAll。详见 [PAIRED_ROUTE_EXCHANGE.md](PAIRED_ROUTE_EXCHANGE.md)。

## 首跳选择利用 fullmesh

仅 EP32 RDMA、G16、已开启去重的 FIRST_HOP 改为：在该 token 实际命中的
远端专家中，选目标专家条数最多的四 rank fullmesh 小组内最早的 top-k 项。
并列时保留原 top-k 优先级。每条专家路由都携带相同大小 token，
这使按该拓扑估计的跨 Clos relay 数为 `远端专家条数 - 最大组内条数`。

例如目标卡依次为 16、20、21、22：旧首跳16对应3次跨组转发；新首跳20
对应1次跨组转发，以及组内到21、22的转发。首跳本身仍是实际消费者，
跨机副本仍只有一份，其他专家仍仅一层机内转发。未采用固定非目标 gateway 搬运 payload。

分组统计融合进已有 Classify 扫描，仅使用两个 uint32 scalar 字；无路由二次搬入、
额外 UB 数组、共享队列或核间同步。单目标 DIRECT 不执行四组选择。
首跳可能位于更后的 top-k，FillTail 因此枚举 mask 中除首跳外的所有成员，
不能沿用 `firstK+1` 起点。原 source/token/k、每专家 q 和权重身份保持。
combine 源码不改；归约首项变化可能改变浮点加法顺序，设备正确性需覆盖。

这只是减少估算的 Clos 流量，不保证每条物理链路占用率或整体时延改善。
额外 Scalar、目标卡热点和多源并发仍可能影响性能；EP16、EP64+、MTE 对照、
非去重路径保留原首跳策略。

## 验证与 A/B

CPU 协议模型覆盖路由异步进展、变 shape、真实调用顺序的bank复用、延迟commit/消费者、
CQ 错误、不同 scratch 复用；packet 模型覆盖乱序/部分 body 可见性、逐 word gate、
Compact/KeepFlags、多 peer/QP、尾批、slot 复用及 128KiB 边界。
fullmesh 原生 helper 对 19200 组路由核对最少跨组 relay、所有成员以及 q/权重身份保留。
这些不能代替 CANN 编译、NIC 可见性和 NPU 正确性/性能验收。

先全 rank 重建匹配 SDK 和完整 wheel。性能比较先固定新 SDK，
分别比较“仅 SDK 更新”“route/packet 合并”“再加 fullmesh 首跳”，避免把 SDK 与 kernel
收益混在一起。固定 H/K/E/dtype/seed/M、Exp及采样参数，timer OFF 测完整 dispatch，
ON 只用于定位。至少覆盖 B1/2/4/8/16/31/32、M32/M256、BF16/FP8，
无远端/单远端/同小组/跨小组/热点路由及反复调用；B32/M32 不去重，不能混同 B32/M256。
记录全 rank 原始 kernel task、最慢 rank、API 时间以及硬件端口计数（若目标提供），
不把提交次数、CPU 路由代价或阶段累计时间直接当作带宽/加速倍数。
