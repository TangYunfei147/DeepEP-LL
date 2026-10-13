# 四层实验验收要求

用户要求补齐四层，旧版双rank搬运包不能视为四层已完成。

1. 空载：真实GM tiling读取，无UB初始化/路由/数据搬运；同时保留完全空启动参照。阻止编译器删除tiling读取，保持通信基线入参布局一致。
2. 元数据：复用当前LL路由准备实现（路由加载、专家前缀位置q、去重分类/发送任务、sourceMask/sourceWeights）；记录准确的提取边界。跨rank计数交换另列为控制通信，不算纯本地元数据。不得用不同算法的玩具路由kernel代替LL元数据耗时。
3. 最简通信：统一入参、属性和GM tiling；scenario key=1一打一/2多打一/3一打多/4多打多。每个源rank独立接收槽，所有rank参与初始化和验证。多rank集合、单边发送字节、每rank总字节、全作业总字节都显式记录。A5仅两个实际rank，多方场景在此退化成一条边，真实多方竞争由用户目标机执行。
4. 完整算子：保留生产dispatch/combine profiler kernel任务，同时用独立非profiler循环测API调用至event等待和device同步完成。准备输入/专家计算/测试barrier在区间外，API内部工作在区间内；保存每rank与逐轮rank MAX原始样本。dispatch、combine、roundtrip分别定义，不把两个独立样本相加当实测roundtrip。

补充维度：发送完成与接收可消费；首次调用与稳态复用；同机/跨机、方向和双向竞争；不同路由及热点；消息大小/分包大小/核数/lane并发；逻辑payload与wire字节；mean/p50/p95/p99及rankMAX；负载正确性、重复调用、尾块和超时退出。

约束：设备0/1，所有远端产物在/home/tyf/Q4_remote_test；实际目标操作由用户执行，不披露机型，不在A5性能采样形成目标结论。
