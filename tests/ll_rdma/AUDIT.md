# LL 混合 MTE / NDR 协议审计（ABI 5）

分支 `kernel/ll-ndr-hns1825`。NDR 指 HNS_1825 NPU Direct RDMA。此版本把原全 peer RDMA 改为：**机内 MTE，跨机 RDMA put，禁止跨机 get**。

## 物理机与逻辑组分离

- 生产部署默认且当前固定每机 16 个连续 global rank：`host=rank/16`。启动时必须使 rank 顺序符合物理部署，不能以逻辑分组代替机界。
- Host tiling 按通信域选 key：EP≤16 → `tilingKey=4`、G=4；EP>16 → `tilingKey=16`、G=16。设备入口校验 key 与通信域一致，再选择 `<4>`/`<16>` 模板实例；两种实例在同一 wheel 中。
- 2/4/8 的 key 选择结果也是 4，但原生 LL 准入仍不支持这些域。CPU 选择 key 不表示启动或放宽了原生协议。
- `M>32` 的去重启用条件保持不变。G 选择与去重开关是两件事。EP16 开启去重时，现在有四个逻辑组；组间流量仍全部在同一物理机内，以 MTE 传输。
- G=4 和 G=16 都是 16 的因子。因此同一去重组的 relay、ready 和 partial 读取必然位于同一物理机；不需要新增跨机读协议或远端读回退。
- 旧 `DEEPEP_LL_RANKS_PER_GROUP` 全局编译宏在本分支被拒绝。原 `kernel/ll-configurable-groups` 分支保留原来的泛化接口，不受此次修改影响。

## 完整数据路径

| 路径 | 机内 | 跨机 |
|---|---|---|
| dispatch 专家计数与 ready | MTE 32B 记录写 | 对称 GM 暂存；put 28B count/body → quiet → put 4B ready → quiet |
| dispatch DIRECT / FIRST_HOP payload | 原 MTE 包写 | 每包 480B body 块批量 put → quiet → 32B flag 块批量 put → quiet |
| dispatch relay | 同逻辑组内 MTE | 按 G/机界约束不会产生跨机 relay |
| dispatch 接收、轮询、清理、magic、prefix | 本地 MTE | 没有远端读取 |
| combine stage 准备 | 本地输入 → UB → 本地对称 GM | 没有跨机 stage 准备 |
| combine ready | 同逻辑组内 MTE；保留所有实际核 prepare-done 依赖 | 不跨机 |
| combine partial 成员读取 | `LoadLocalPeer` 使用机内 MTE | 显式 trap；不调用 RDMA get |
| combine DIRECT / partial 结果返回 | MTE 包写 | 对称暂存、body/flag 两阶段 RDMA put |
| combine 最终聚合与 flag 清理 | 本地 MTE | 没有远端读取 |

生产传输收口位于 `csrc/kernels/dispatch/low_latency/device/ll_transport.h`。`RemoteWrite` 只在显式 RDMA 模式且 peer 不同物理机时为真。`LoadLocalPeer` 同时检查机界、范围和本地 MTE 映射；partial 路径额外检查成员属于当前 G 组。生产代码不再调用 `aclshmemx_roce_qp_get_nbi`。

## Host / tiling / ABI

- `DEEPEP_LL_BACKEND=rdma` 现在表示混合策略；`mte` 仍是显式对照模式。RDMA 模式不把失败的跨机写静默改为 MTE。
- EP≤16 的混合 Session 仅启用 `ACLSHMEM_DATA_OP_MTE`，不初始化 ROCE QP、不分配 RDMA 暂存，也不要求 RDMA NIC 前检。
- EP>16 的混合 Session 启用 `MTE | ROCE`，实际 AIV 核数决定 QP 数和对称暂存。Host 只为本机 peers 填写 MTE `peerMems`，跨机表项保持 null。
- RDMA 写仍传本地对称地址，由 SDK 转换远端地址；不会把 MTE 映射传给 RDMA。原 rank bias=1024B、8GiB 公开数据布局、2MiB guard 保留；跨机作业增加每实际核 128KiB 暂存，总量向 2MiB 对齐。
- CommArgs 的 `reserved0` 表示物理每机 rank 数（生产为 16）；`localRank/localRankSize` 表示选定逻辑组。原保留尾部继续保存 RDMA 对称基址、暂存、backend/QP。
- dispatch 32B tiling 的 byte 21、combine 16B tiling 的 byte 15 保存 key。Host UB 预算也按选定 G 计算。ready 控制容量仍按最大 G=16 保留，不等同于当前逻辑组。
- 核心协议升为 **5**（增加可选专家诊断缓冲；传输策略保持不变）。Host/device 检查两个 key 的选择一致性；测试入口在所有 rank 比较实际 AIV、选定 G、ABI 和构建开关。必须全 rank 重建匹配 wheel。
- kernel 名字保持不变，profiling task 匹配与 stage timer 入口保留；打点 metadata 根据通信域记录实际 G。

## 写入顺序、复用和事件

跨机 Packet：UB→注册对称 GM 后显式 MTE3_S 完成，再提交全部 body WQE；指定 peer/本核 QP quiet 后发布全部 flags，再 quiet 才复用暂存。每核独占 QP，不假设跨 QP 排序。SDK scratch 借用已暂存包前 128B，返回前恢复；dispatch relay 使用当前已消费 lane 的事件号，不能抢占其他预取 lane 的 MTE2_S。

NIC 可写接收/控制/暂存视图保持 L2 bypass。机内读取不经 NIC，不借用输入 lane 构造 RDMA WQE，原 MTE2/Vector lane 依赖负责完成与复用。接收超时、SDK CQ 错误和 trap 不转为成功。

成功 CQE 的远端完成及 HBM 可见性仍依赖目标 HNS_1825/provider；原有两阶段写的硬件验收不能由 CPU 模型替代。首版跨机写每包两个 quiet，可能有明显开销，未宣称加速。

## SDK 与验收边界

SHMEM 仍锁定 `4dd80e2129da42bd773982c2bb3758af1ed0bba8`。统一 `build_shmem.py --rdma-backend HNS_1825` 保留 ROCE-only 三处 72-QP policy 补丁与指纹，UDMA 56 上限不改；不另建 SDK 树。普通 qp_put/qp_quiet 支持 HNS_1825，XSCALE aggregate 接口不使用。SDK 可能为机内 peer 也建立 ROCE 通道，但 kernel 机内数据不使用它们；资源容量仍需目标机验收。

两 rank primitive 可显式描述同机两卡（每机 2 rank）或跨机各一卡（每机 1 rank），这是测试真实物理部署，不是伪造原生 LL rank 数。生产机界仍固定 16。A5 只允许前者、设备 0/1；跨机 RDMA 写、完整 LL EP16/32 和性能由用户在目标环境验收。
