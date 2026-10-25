# 每卡独立网口的 EP32 配对路由交换（协议8）

基线67b1b72。目标两台机器、每台16卡，每卡独立跨机网口。
新路径仅针对NDR rdma模式EP32且B*K<=8192，覆盖既有B<=256/K<=32；
其他域和更大路由表保留原计数帧路径。SHMEM、provider及NIC配置不修改。

## 数据路径

每张卡的逻辑配对peer是`rank ^ 16`，即0↔16、1↔17、…、15↔31。
SDK/provider继续负责该NPU的实际NIC绑定；逻辑配对不推断具体物理端口编号。
每rank唯一core0向配对卡发送完整本源top-k ID表：

1. 原始B*K int32路由落本卡注册对称TX槽；PostBody完成MTE后提交一次RDMA put。
2. 网络可以进展的同时，计数核计算本源路由并通过机内MTE发布本源计数。
3. core0的PublishCommit执行body quiet，再提交单调epoch commit；此处不再立即quiet。
   各gateway计数核随后根据配对RX重建本物理机专家计数，并通过MTE分发到本机16卡。
   两轮计数分别标记原source rank和配对source rank。
4. 全部计数核读完RX、退休本地计数DMA后发布各自done。core0等待全部done后返回ACK；
   最后同peer/同QP的quiet同时退休commit与ACK，共3次put、2次quiet。
5. 下次复用同一bank前，源core0必须等到该bank上一次发送epoch的ACK。

计数核保留原全局专家分片，但夹取本机专家范围；无本机专家任务的核仍参加done。
非core0空核不加载/扫描路由；core0即使没有计数任务也负责网络提交和ACK。
gateway以原source维度生成完整计数（含零），不合并成整机总计数；原expert-major
prefix、q、token/top-k顺序和后续dispatch/combine元数据语义保持不变。

路由控制的跨机提交数是每rank每轮3次put（body、commit、ACK），每方向16卡共48次，
相比基线EP32/E384/56核的每rank78次计数put减少。payload仍走原G16首跳和relay，
不计入这3次；不宣称总网络提交只有3次。
B1/K8路由body32B，B32/K8 body1024B，另有commit和ACK各4B。
没有远端专家的source也发布路由及epoch，gateway据此生成明确零计数。

## 存储、事件与进度

每bank在固定1.25MiB之后保留TX/RX路由槽、commit/ACK/lastSent和每核done；
每核epoch计数器固定使用bank0，shape变化和中间插入combine不会重置。
区域高于计数控制区和combine控制区，低于2MiB payload；未借用会被payload批处理
复用的per-core RDMA Stage。旧combine结果清理不会覆盖这些状态。

core0独占本rank的QP0执行路由控制，完成所有quiet后才进入自己的payload发送。
其他核沿用自己的QP；无跨QP排序假设，无RDMA get。双方先完成出站发布再等入站，
ACK只依赖本gateway的路由消费，不依赖后续payload接收。共享commit永不清零。
PostBody使用稳定GM源；SDK put返回后UB scratch可用于本源计数。
PublishCommit复用scratch前，两条计数计算路径均已通过原V_S依赖退休；
机内计数MTE只引用独立statusBuf，不占用网络scratch。
epoch溢出trap，等待沿用NDR超时保护；CQ错误不回退为成功。

## 性能与打点

该改动减少跨机计数扇出，但引入gateway接收、两源路由重建、机内计数发布和ACK等待。
CountSend现在包含这些阶段，CountReceive仍包含本机计数帧等待和前缀快照。
不能只看count_recv变短；应同时比较完整dispatch、count_send、count_recv及后续prefix。
stage schema3保持，当前没有细分每个新路由子阶段。没有增加Host逐次HCCL或设备全核SyncAll。

## 更新与验证

核心协议8拒绝旧行为混装，所有rank重装匹配完整wheel。
仅路由协议优化不要求SDK重建；若同步升级2026-10-25官方主线，必须按
[SHMEM_MASTER_UPDATE.md](SHMEM_MASTER_UPDATE.md)重建SDK。
先两机B1/2/4/8/16/31/32，M32与M256、BF16/FP8、normal/hotspot正确性和重复调用，
再使用相同Exp/采样参数与67b1b72做timer OFF性能对照。
本机CPU C++协议测试覆盖12epoch、同/交替bank、延迟和空核、变路由长度、body CQ故障；
Python验证各gateway重建结果与原始逐source计数/前缀相同，包括单边全零远端路由。
2026-10-25模型增加独立网络线程，验证无quiet也能进展、延迟commit/ACK、
同bank旧ACK保护和三个独立scratch复用；优化对照基线为27cf73c。
CPU通过不等于CANN编译、NIC实际可达性或设备精度/性能通过。
