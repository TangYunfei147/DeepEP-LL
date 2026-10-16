# 当前 LL 固定值边界

- **实际 AIV**：平台与当前设备可用数取小，16..72 偶数、1:1 收发；72 是容量，不是固定启动数。ROCE QP 每 peer 按实际 AIV 数配置，一个物理核独占一个 QP。
- **逻辑分组**：由 tiling key 选择 EP≤16 的 G=4、EP>16 的 G=16；不等同于 SHMEM UDMA shared-jetty 的 8 路映射。目标生产部署每机 16 卡，EP32 必须两机；不强迫逻辑 G 等于物理卡数。
- **LL rank 准入**：在 2/4/8/16/32 目标集合中仅 16/32 可运行。两卡 RDMA smoke 是独立 primitive，不能降低 LL 准入条件。
- **包协议**：480B body＋32B flags、512B 记录槽、原有 rank bias 1024B、8GiB 用户 workspace 与双窗口保留；这些不是网卡 MTU 或链路峰值。
- **RDMA 暂存**：128KiB/实际核，512B 对齐，放在同一对称分配的原 guard 后；SDK UB scratch 至少 128B。额外暂存不能误计入公开 8GiB 数据布局。
- **SDK QP policy**：遵循锁定 SHMEM 原生每 peer 56 QP 上限，不再扩容。启用跨机 RDMA 时每实际 AIV 独占 QP，超限必须在初始化前报错，不截断、不共享、不减少启动核数；纯 MTE 的 LL 72 核控制容量保持不变。动态数组审计不等于目标 provider 资源容量通过。
- **timeout**：RDMA 接收与 SDK Ascend950 CQ 使用 `60e9` 原始 cycle 界限；设备 timer 的微秒转换仍须显式计数器频率，不能把固定界限当作实测频率。
- **ABI**：当前核心协议 4，CommArgs 保留尾部承载 RDMA workspace/staging/backend/QP；结构总大小未改，但旧 v2 wheel 不兼容。
- **未验收的硬件前提**：HNS_1825 驱动/provider、NPU HBM 注册、PCIe/NIC 亲和性、RC 完成可见性、全域通道数量与多机连接。见 [协议审计](../ll_rdma/AUDIT.md)。

已删除的 `LL_DISPATCH_DEBUG_*`、`LL_DLOG` 不作为当前调试入口。通过原始日志、超时/错误状态、独立 primitive 和默认关闭的 stage timer 定位。

- **物理机界**：生产固定16个连续rank；机内MTE，跨机仅RDMA put。partial成员必须同G组且同物理机，跨机读取拒绝。EP≤16不初始化ROCE；两rank primitive按真实同机/跨机部署单独设置物理分区，不改变原生LL准入。
