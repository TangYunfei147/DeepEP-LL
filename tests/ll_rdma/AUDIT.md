# LL HNS_1825 数据面审计与协议

维护分支 `kernel/ll-configurable-groups`。NDR 在这里指 NPU Direct RDMA，使用 SHMEM AIV 直驱 RoCE；不指 NVIDIA NDR InfiniBand。审计基线为 `cb59bff99ecd39d27ef57788c5e6fd0009e4e636`，SHMEM 基线仍锁定 `4dd80e2129da42bd773982c2bb3758af1ed0bba8`。

## 通信路径

以下路径相对于 `csrc/kernels`。P 为通信 rank 数，E 为专家数；逻辑组 G 和实际 AIV 数 N 沿用现有查询/配置，不能替换为固定 8/16/64。

| 路径与源码 | 源 → 目的、尺寸 | 发布/消费与复用 | RDMA 实现 |
|---|---|---|---|
| dispatch `dispatch/low_latency/device/dispatch_server_dedup_counts.h::SendExpertCntRange` | 本核 UB 计数记录 → 专家所属 rank，32B/专家；GM 记录距 512B，专家跨度 P×512B | ready 为 FP32 1 的位模式；接收核等全部 ready，提取整数计数，清除 ready | 对称暂存；先提交每条记录偏移 4 的 28B，quiet，再提交偏移 0 的 4B，quiet |
| dispatch `dispatch_server_dedup_send.h::SendPacket` | 已打包 UB → 目标 dispatch 槽，tokenStride，512B 块中 480B 数据＋32B 标志 | 保留首跳/直接包、marker、q、去重布局；全部块标志就绪后消费 | 所有数据块作为一批普通 NBI；quiet 后发布所有标志块；第二次 quiet 后复用暂存 |
| dispatch `dispatch_server_dedup_recv.h::ConsumeLoaded` | 首跳接收 UB → 其余专家所属 rank，tokenStride | 修改原有 tail/marker 后转发；一个 UB 包可发多个目的 | 同一 Packet helper；SDK 借用的前 128B UB 在返回前恢复，保证后续转发看到原包 |
| dispatch `Probe/IssueLoad/ClearFlags` | 本地对称 GM → UB → 本地公开输出；最终本地清 flag | 等全部块标志；保留 compact/keep-flags 两种读取与尾部处理 | 本地 MTE；RDMA 模式对 NIC 可写数据/标志使用 L2 bypass |
| dispatch `RotateDispatchServerDedupMagic`、本地 prefix/cumsum | 本地 GM ↔ UB | 每核独占 magic；所有实际核按原规则轮转窗口 | 保留本地 MTE，不产生 RDMA |
| combine `combine/low_latency/combine_server_dedup_kernel.h::ReturnDirectTokens` | UB 结果包 → 原 source rank 的 `(token,topk)` 槽，resultStride | 最终聚合等待结果 flags | Packet helper，两阶段发布 |
| combine `CopyLocalPendingTokens` | 本地 Torch expert GM → UB → 本地对称 stage | 每核 stage 完成后写本地 prepare-done；通知核等全部 N 个核 | 保留本地 MTE；RDMA stage 使用 L2 bypass，供远端 NIC get |
| combine `NotifyServerCard/WaitServerReady` | 本地 prepare-done → 通知核；UB ready → 同逻辑组各 rank 的接收核 ready 行 | 等本地 stage 完成，再发 ready；每个接收核等 G 个 ready 并清理自己的行 | 32B 控制记录经对称暂存和 ROCE put；独占通知核 QP；无数据/标志混写 |
| combine `combine_server_dedup_partial.h::Load` | member 0 读本地 Torch input；其他 member 读其对称 stage，H×2B | ready 到齐才 get；get quiet 后才搬到 UB 参与归约 | member 0 保留 MTE；远端 qp_get_nbi → 本核对称暂存 → quiet → 本地 GM→UB；本 rank member 本地读取 |
| combine `ServerDedupPartial::Send` | UB partial → source rank 结果槽，packetBytes | 沿用原始权重、归约与槽位映射 | Packet helper |
| combine `combine_server_dedup_source_aggregate.h` | 本地结果 flags/包 → UB → 本地 yOut；清 flags | 沿用 M4→M5 槽交接及最终聚合 | 本地 MTE，NIC 可写结果及清理使用 L2 bypass |

原远端 `DataCopy` 全部收口至 `dispatch/low_latency/device/ll_transport.h`。生产 LL 不再在 RDMA 模式建立远端 `peerMems`；远端表项为 null。本地 `peerMems[rank]` 仅用于本地接收、控制、magic、stage。剩余 `Control(rank,...)` 调用均为本 rank。没有新增远端标量读写。Torch 输入、公开输出、路由表、权重与本地 metadata 搬运不替换为 RDMA。

## 实际 SDK 接口和地址规则

检查的公共文件：`third-party/shmem/include/device/gm2gm/engine/shmem_device_rdma.h`。实现位于 SDK `src/device/gm2gm/engine/shmem_device_rdma.hpp` 和 `rdma_backends/rdma_device_backend_hns_1825.hpp`。

- 使用 `aclshmemx_roce_qp_put_nbi<T>(dst,src,ub,element_count,peer,qp,sync_id)`、对应的 `qp_get_nbi`，以及 `aclshmemx_roce_qp_quiet(peer,qp,ub,sync_id)`。实现以 `uint8_t` 计数，所以 count 是字节数。
- put/get 的两个完整范围必须在已注册对称内存中。put 的 dst、get 的 src 都传**本地对称地址**；SDK 自己转换到远端。锁定版本的显式 QP 实现调用 device `aclshmem_ptr`，它与 `aclshmem_roce_ptr` 都使用 device state 的 heap base 差值与 peer heap base；这不允许把 Host MTE 映射地址提前传进去。
- LL 对称地址为 `workspace + destination_rank*1024 + original_offset`，保留原 rank bias 和 8GiB 内部布局。额外暂存位于同一分配末尾、原 2MiB guard 之后；没有把普通 Torch 内存交给 NIC。
- 无 QP 参数的 put/get 默认 QP 0，不能用于多核并发。本实现不使用它们，也不使用共享 UDMA jetty 映射。
- HNS_1825 支持显式 QP 普通 NBI/quiet；`defer/submit` aggregate overload 在当前版本仅支持 XSCALE，不能在 1825 上使用。这里的“批”是连续提交普通 NBI，不是 XSCALE aggregate API。
- quiet 读取指定 `(peer,qp)` 的 SQ head 并轮询 CQ 至该 head；不替其他 QP 完成。1825 CQ 超时和错误路径触发 device trap，不作为成功返回继续发布标志。普通 post_send 在 SQ 临满时也可能轮询 CQ。
- 当前 SDK 有 signal/atomic 接口及 1825 atomic 实现，但高层接口并不提供本协议所需的独占核 QP 发布组合。这里不调用默认 QP 的 signal/atomic，也不从接口名推断跨 QP fence；使用数据 quiet 后再提交标志。

## 完成、可见性和复用

1. Packet 先把完整 UB 包以 L2 bypass 写入本核对称 GM，显式 `MTE3_S` Set/Wait 等待 UB→GM 完成后 Scalar 才提交 NIC WQE；不能只把请求入队视为 GM 已就绪。
2. 同一核独占每个 peer 的 QP `qp_idx=GetBlockIdx()`。计数阶段结束前已 quiet；该核后续 sender/relay/combine 阶段复用自己的 QP，不与其他核共享 SQ/CQ。
3. 所有 body 写入完成 CQ 后，才提交 flag 写入。接收方必须看到完整就绪模式才消费数据，L2 bypass 避免读到之前轮次的缓存。因 body 批已完成，flag 内部写入次序不影响 payload 先行关系。
4. 第二次 quiet 后暂存才可覆盖；get quiet 后才读取本地 GM，并在下一次 get 复用暂存前完成 GM→UB。每核 128KiB，512B 对齐，总分配按 2MiB 向上取整；核数来自实际查询，不固定分配 72 个核的启动任务。
5. Packet 的 SDK scratch 借用已暂存包的前 128B，等待已有流水后才使用，返回前从暂存恢复。计数 scratch 使用已退休的路由 workBuf；ready 使用已退休的 prepareInput；get 使用即将被覆盖的输入 lane。没有额外侵占已有 UB arena，也没有异步复用仍被 NIC 读取的 GM。
   SDK CQ 轮询使用 `MTE2_S(sync_id)`。sender/count/combine 使用未占用的 7；dispatch relay 使用刚在 ConsumeLoaded 入口消费的当前 lane ID，保留其他预取 lane 的事件。不能固定抢占接收 lane 7。恢复 UB 和 get 的 GM→UB 使用显式 MTE2_S 完成；get 借用输入 lane 前另有 V_S，避免 Scalar 提前覆盖仍被 Cast 读取的 UB。
6. combine ready 由通知核观察全部本地 stage 完成后发布；对端先读 ready 再发 get。这是显式因果依赖，不要求跨 QP 的隐式顺序。计数和 payload 可以使用不同 QP，因为接收 payload 仍独立等待包 flags。
7. RDMA 接收轮询增加与锁定 SDK Ascend950 CQ 相同的 `60e9` 原始 cycle 超时界限；不把这个常量作为性能计时频率接口。保留原 MTE 轮询行为及 SDK 错误传播。

这个论证依赖目标 HNS_1825/Hcomm 的可靠连接完成语义：成功 write CQE 表示远端 write 完成，且 NPU HBM 的可见性符合注册内存协议。代码不能证明驱动、PCIe 与 provider 的硬件实现正确；`smoke_device.cpp` 在对端发布期间轮询并验证 payload/count，在不同轮次换数据，专门验收这项前提。不能以编译、CPU 模型或 MTE 运行替代它。

首版每个包有两个 quiet，get 后有一个 quiet；虽然块内提交成批，包间尚未流水化。SDK 每次普通 NBI 也有 WQE/doorbell 同步成本。这个选择用较简单的暂存生命周期换取可审计性，可能明显增加时延。没有匹配目标机基线前不宣称加速；后续 ring/pipelining 必须保留每槽 peer/QP/完成状态、UB 生命周期和发布因果关系。

## Host、依赖和风险

- `DEEPEP_LL_BACKEND=mte|rdma` 在 Session 创建时固定；默认 mte。RDMA 使用 `ACLSHMEM_DATA_OP_ROCE`，没有 MTE/UDMA 数据域回退。公开 dispatch/combine 参数和返回值不变；ElasticBuffer 的 RDMA Session 要显式 `use_1g_huge_pages=False`，不混用 HT。
- `DEEPEP_LL_RDMA=ON` 才编译设备 RDMA；SDK 构建显式 `--rdma-backend HNS_1825`，映射为 `ACLSHMEM_RDMA_SUPPORT=ON`、`ACLSHMEM_RDMA_BACKEND=HNS_1825`。默认构建不启用 RDMA。
- 锁定 SDK 将 ROCE 与 UDMA 共用 56-QP policy limit。ROCE v2 的 queue/channel/exception 数组使用动态 `rank_count*qp_count`。`scripts/shmem_rdma_patch.py` 只将 ROCE init、v2 manager、RDMA exception 三处检查改为 72；不改 UDMA、公共头、源码 Git revision。脚本检查完整受影响文件并拒绝覆盖用户修改，构建记录实际 source patch hash。分支提交的是可复现补丁脚本，不是假造子仓提交或复制另一套 SDK。
- 这项 SDK policy 扩展不保证 provider 支持所有 `(P-1)*N` 通道；通道建立失败必须返回失败。资源容量、NPU HBM 注册及 PCIe/NIC 亲和性由目标平台验收。
- SDK 使用 Hcomm ROCE endpoint、`CommMemType::DEVICE` 的 `HcommMemReg`、通道创建/状态查询；Host endpoint 绑定 NPU physical ID。所需共享库和 provider 由本机 CANN 环境及可选 `DEEPEP_RDMA_PROVIDER_PATH` 提供，不安装或修改驱动。
- SDK 从平台 rootinfo 获取每个 physical NPU 的 ROCE IP；HNS_1825 的 fallback 使用平台 topology daemon 的 virtual topology/亲和性数据。bootstrap 的 Host IP/port 不是 NIC 绑定配置。不要为通过测试伪造拓扑或修改系统文件。
- CommArgs 大小仍为 529944B，v3 使用原保留尾部：两个地址为 unbiased workspace/staging，两个配置为 backend/QP 数。Host/device 核心协议升至 **3**；新增 RDMA 编译能力互检。必须全 rank 重建匹配 wheel，不能混搭 v2 或不同 RDMA 开关的设备库。
- 逻辑 G、实际 N、去重阈值、窗口、包布局、数值运算与 LL EP 准入保持原规则。两卡 primitive 不是 EP16 的替身；目标域集合 2/4/8/16/32 中，原生 LL 仍只有 16/32 可执行。
