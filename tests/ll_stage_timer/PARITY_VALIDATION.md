> 历史记录：以下验证时默认G16；2026-10-19默认改为G4，旧测试结果不代表此次设备验收。

# 跨kernel分支同步验证（2026-10-16）

NDR基线f3369d2；泛化基线8f5eaf2。仅本地修改，远端环境已回收，没有SSH、CANN编译或NPU运行。

同步内容：阶段schema2/专家schema1及ABI5接口、细分打点、专家记录、稳定尾段采样、
Chrome trace、统一启动/分析/对比工具。共享文件清单由verify_shared.py做字节一致性核验。
泛化分支Session、计数通信、combine算法及分组宏原文件保持不变；仍为MTE、默认G16。
NDR保留自己的tiling-key与RDMA策略。

用户明确要求G作为比较变量，因此不提供expected-group强制检查，也不自动调整泛化G。
比较输出保留baseline_group/candidate_group，G16与G4能够直接对比。
同一作业内各rank的配置一致性仍由原有协议保护。

本地CPU检查覆盖旧/新schema、专家归属、损坏数据、采样协议、隐藏包络不丢统计、
目录含空格/不同cwd的启动计划和分析、属性/函数两种G查询、不同G比较、其他采样条件不匹配拒绝、
缺失RDMA阶段不补零。Windows无法执行的Linux shell环境测试未选，新的纯Python目录迁移已执行。

需要目标机重新编译两个分支的wheel，分别验证数值、单机和实际可用的跨机通信，再做性能对比。
历史A5结果不代表此次同步的设备验收。建议使用独立clone/虚拟环境/工程内SDK。

最终本地结果：两分支分别89 passed、1项Linux-shell测试未选；26个共享文件逐字节一致。两边日志为各自.logs/parity-cpu.log。G16/G4比较测试通过，分组默认值未更改。
