# LL NDR 分支的分组与传输

当前分支 `kernel/ll-ndr-hns1825` 使用 Host tiling key 选择去重组：EP≤16 为 key/G=4，EP>16 为 key/G=16。两种设备模板都编译进 wheel，不再编辑 `DEEPEP_LL_RANKS_PER_GROUP`；该宏在本分支会报错。

物理机默认固定 16 个连续 rank，与逻辑 G 分开。显式 RDMA 模式机内 MTE、跨机 RDMA put；partial 组内读取由 MTE 完成，跨机读直接拒绝。EP16 因此可以进行四组去重，但不产生跨机流量。

原生 LL EP 准入与 M>32 的去重开关不变；EP2/4/8 不因 key=4 而获准启动。实际 AIV 查询和 1:1 收发分工仍支持 16..72 偶数。

核心协议为 4，切换版本须全 rank 重建匹配 wheel/SDK。原泛化+打点分支 `kernel/ll-configurable-groups` 保持原代码、ABI 2 和原配置方式。

完整步骤：[混合通信指南](tests/ll_rdma/README.md)；[协议审计](tests/ll_rdma/AUDIT.md)；[阶段打点](tests/ll_stage_timer/README.md)。
