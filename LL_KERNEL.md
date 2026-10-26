# LL NDR 分支的分组与传输

当前分支 `kernel/ll-ndr-hns1825` 使用 Host tiling key 选择去重组：EP≤16 为 key/G=4，EP>16 为 key/G=16。两种设备模板都编译进 wheel，不再编辑 `DEEPEP_LL_RANKS_PER_GROUP`；该宏在本分支会报错。

物理机默认固定 16 个连续 rank，与逻辑 G 分开。默认混合后端机内 MTE、跨机 RDMA put；partial 组内读取由 MTE 完成，跨机读直接拒绝。EP16 因此可以进行四组去重，但不产生跨机流量。

当前去重条件为`M>32 || (EP>16 && B<32)`；在公开`B≤M`约束下，跨机非去重对应B=M32。EP2/4/8 不因 key=4 而获准启动。实际 AIV 查询和 1:1 收发分工仍支持 16..72 偶数，跨机RDMA另外受SDK的QP数量限制。

当前NDR核心协议为9，切换版本须全 rank 安装匹配 wheel；SDK 配置和身份未改变时自动校验复用，无需仅因切分支重建。其他独立分支的ABI、G与调试协议以各自源码为准。

非去重dispatch执行流程与具体例子见[NON_DEDUP_DISPATCH_FLOW](tests/ll_rdma/NON_DEDUP_DISPATCH_FLOW.md)。
去重dispatch的FIRST_HOP、relay和权重流程见[DEDUP_DISPATCH_FLOW](tests/ll_rdma/DEDUP_DISPATCH_FLOW.md)。

NDR 固定编入 HNS_1825 能力，`python scripts/build_shmem.py` 默认构建对应 SDK。普通构建/运行无需 `DEEPEP_LL_RDMA` 或 `DEEPEP_LL_BACKEND`。共同入口 `tests/ll_stage_timer/run.py` 按分支选择默认后端，只有纯 MTE 对照才加 `--backend mte`。

完整步骤：[混合通信指南](tests/ll_rdma/README.md)；[协议审计](tests/ll_rdma/AUDIT.md)；[阶段打点](tests/ll_stage_timer/README.md)。

公共Runtime/Session保留MTE默认；由已确认LL用途的测试入口选择混合后端，不按缓冲容量猜用途。维护原则见[配置注意事项](tests/ll_stage_timer/CONFIGURATION_NOTES.md)。

NDR wheel/kernel增量构建、timer对象复用与目标机计时步骤见[增量构建说明](docs/BUILD_INCREMENTAL.md)。

NDR dispatch已拆为14个独立特化入口以缩短首次编译，保持全部类型/G和单次启动；内部launch符号v2需整套wheel更新。profiler名称与冷构建实测见[首次编译说明](docs/BUILD_COLD.md)。
