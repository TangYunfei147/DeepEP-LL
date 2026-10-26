# 每卡独立网口的 EP32 配对路由交换（协议9）

初始配对路由的基线为67b1b72；本次删除ACK的直接对照基线为1a1b6fc。
目标两台机器、每台16卡，每卡独立跨机网口。
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
4. core0在FinishSend中等待自己的commit完成，保护注册GM源及CQ复用；
   不等待其他计数核，也不返回ACK。共2次put、2次quiet。
5. 后续调用按已有双窗口及通信依赖复用bank，不再维护ACK、lastSent或每核done。

计数核保留原全局专家分片，但夹取本机专家范围；无本机专家任务的非core0核直接返回。
非core0空核不加载/扫描路由；core0即使没有计数任务也负责body、commit及完成等待。
gateway以原source维度生成完整计数（含零），不合并成整机总计数；原expert-major
prefix、q、token/top-k顺序和后续dispatch/combine元数据语义保持不变。

路由控制的跨机提交数是每rank每轮2次put（body、commit），每方向16卡共32次，
相比基线EP32/E384/56核的每rank78次计数put减少。payload仍走原G16首跳和relay，
不计入这2次；不宣称总网络提交只有2次。
B1/K8路由body32B，B32/K8 body1024B，另有commit 4B。
没有远端专家的source也发布路由及epoch，gateway据此生成明确零计数。

## 存储、事件与进度

每bank在固定1.25MiB之后保留TX/RX路由槽、sequence和commit；
每核epoch计数器固定使用bank0，shape变化和中间插入combine不会重置。
区域高于计数控制区和combine控制区，低于2MiB payload；未借用会被payload批处理
复用的per-core RDMA Stage。旧combine结果清理不会覆盖这些状态。

core0独占本rank的QP0执行路由控制，完成所有quiet后才进入自己的payload发送。
其他核沿用自己的QP；无跨QP排序假设，无RDMA get。双方先完成出站发布再等入站。
共享commit永不清零；每个实际路由消费者等待本轮epoch，不以非零计数或payload为条件。
PostBody使用稳定GM源；SDK put返回后UB scratch可用于本源计数。
PublishCommit复用scratch前，两条计数计算路径均已通过原V_S依赖退休；
机内计数MTE只引用独立statusBuf，不占用网络scratch。
epoch溢出检查保留。LL生产轮询不读取GetSystemCycle、不增加超时失败处理；
显式timer ON诊断计时及锁定SHMEM SDK自身的完成处理保持各自实现。

### 删除ACK后的复用依据

沿用同一通信器调用顺序一致、各rank单stream串行的契约：

- 连续dispatch交替使用两bank。源A再次写入第n轮bank前，必须完成第n+1轮；
  第n+1轮消费配对B的commit，而B只能在第n轮所有本地核完成后启动第n+1轮。
  因此B的第n轮RX读者已退休。即使所有远端专家计数为零，路由读取仍执行。
- 插入combine时，它也轮转bank，但现有`combine_reuse`在launch前先等本设备完成，
  再做全rank HCCL协调，上一轮所有RX读者已退休。此次不增加或改变该协调。
- 插入非配对计数路径时，原路径仍等待每个source的计数发布（包括零计数），
  该source只有上一轮完成后才能启动并发布新计数。

这依赖具体epoch、窗口和调用顺序，不能仅凭后续有payload通信推导；
任意连续强制同bank、跳过入站commit、去掉现有combine复用协调都不在该契约内。

## 性能与打点

该改动减少跨机计数扇出，保留gateway接收、两源路由重建和机内计数发布。
2026-10-26删除路由ACK及其本地汇合/复用轮询，跨机put由3次降为2次；
body和commit各自的完成quiet保留，quiet次数仍为2次。
CountSend现在包含这些阶段，CountReceive仍包含本机计数帧等待和前缀快照。
不能只看count_recv变短；应同时比较完整dispatch、count_send、count_recv及后续prefix。
stage schema3保持，当前没有细分每个新路由子阶段。没有增加Host逐次HCCL或设备全核SyncAll。

## 更新与验证

核心协议9拒绝旧ACK行为混装，所有rank重装匹配完整wheel。
仅路由协议优化不要求SDK重建；若同步升级2026-10-25官方主线，必须按
[SHMEM_MASTER_UPDATE.md](SHMEM_MASTER_UPDATE.md)重建SDK。
先两机B1/2/4/8/16/31/32，M32与M256、BF16/FP8、normal/hotspot正确性和重复调用，
再使用相同SDK、Exp和采样参数与1a1b6fc做删除ACK的timer OFF性能对照。
本机CPU C++协议测试覆盖多epoch、双bank、延迟和空核、变路由长度、body CQ故障；
Python验证各gateway重建结果与原始逐source计数/前缀相同，包括单边全零远端路由。
模型保留独立网络线程，验证无quiet也能进展、延迟commit/消费者、
真实调用顺序下的窗口复用及独立scratch复用；删除ACK的对照基线为1a1b6fc。
CPU通过不等于CANN编译、NIC实际可达性或设备精度/性能通过。

两机先更新到同一协议9提交，再用[ONE_CLICK.md](ONE_CLICK.md)入口重建wheel测试。
已匹配当前SHMEM的SDK可校验复用，无需为删除ACK重建SDK，也不要加`--skip-build`。
例如节点0/1分别运行（替换节点0真实IP，各自工程根执行）：

```bash
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-no-route-ack-001 --tokens 1 2 4 8 16 31 32
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-no-route-ack-001 --tokens 1 2 4 8 16 31 32
```

两机同加`--mode perf`并换run名可做OFF完整kernel性能对照；保留同SDK和全部采样参数。
