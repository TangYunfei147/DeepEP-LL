# 跨机小batch去重

跨机world>16且实际B<32时，即使M<=32也启用去重，逻辑组G=16对应同物理机。
远端同组多个专家合为FIRST_HOP并在目标机MTE转发；本组保持DIRECT。
dispatch和combine共用同一个判断，源mask、权重、partial归约路径随之匹配。
B=32严格不纳入新增条件；原M>32条件保留。单机EP16策略不变。
核心协议6拒绝旧Host/Device混装；所有rank完整重装wheel，SHMEM不改、不重建。
CPU策略边界验证不能代替两机BF16/FP8正确性、重复调用和性能验证。
