# 协议15：固定配对卡中转穿刺

基于 `24e90d6` 新增一个独立提交。仅 EP32 RoCE 的所有合法 BS 启用：源 rank 的全部跨机 payload 发往固定配对卡 `rank ^ 16`，再由配对卡执行机内组内分发。EP16及EP64+保持原发送路径。8QP不是8份复制；按实际跨机包顺序在QP0..7轮转，沿用该peer的8个独占发送核。小BS不足8包时只有对应部分QP实际发包。

## 行为与边界

- 保留原去重/首路由选择及q，不重新按配对卡专家重排。单个远端成员的DIRECT也经过配对卡。
- 原FIRST_HOP包先到配对卡；配对卡写首专家输出槽，再把Terminal成员包写到各专家槽。首专家保留FirstHop元数据及完整relayReadIndex用于原combine，但不再次转发。原权重、sourceMask、返回K、combine归约顺序不变。
- 配对卡可以不是任何目标专家；其本地输出为零的R核仍先参与中转。接收阶段先完成本核中转，再进入原专家输出任务；不新增跨rank屏障或reader ACK。
- 在dispatch槽和BF16 combine stage上界之后安排独立中转槽，按原(token,firstK)索引隔离中转槽；源GM暂存和每peer quiet不变。该中转区避开combine实际使用的区域（FP8 dispatch槽可能比BF16 combine stage小，不能只按dispatch末尾计算）；Host检查中转长度和独立阶段UB预算。
- core0使用计数work UB分块清本轮全部中转FLAG，MTE3完成后才发布原路由帧；所有S核（含零专家计数核）取得对端本轮完整路由FLAG后才发送payload。没有新增控制WQE或全核汇合；增加的清理和等待计入本穿刺，保护BS/H变化及插入combine后的旧内容。
- 中转完整检查所有32B块FLAG（包含SO尾），再加载整包。组内继续用原MTE wire及全部flag接收，不把RoCE SO推广到MTE。
- 单包中转采用一个UB wire和Scalar逐word flag检查，转发读完成后清中转flag。明确加入本阶段的流水依赖以保护UB与清flag；此实现是保守穿刺，包含中转扫描、等待及分发成本，不宣称已优化成功。
- 独立阶段重置复用计数UB，结束后原receiver重新分配自己的lane，不增加与其同时活跃的lane。源端8QP拥有者规划以固定peer为键，QP0仍由core0在路由quiet后复用。
- route frame容量扩大至16384项以覆盖Host原UB允许的路由范围；不按BS阈值选择直达。sender/receiver/Host预算、协议身份同步更新。SDK不改。
- 同一stream同序、双bank、epoch与原combine_reuse仍是复用前提。下一轮peer路由epoch证明上一轮该peer的所有gateway读者已退出；两轮后的同bank才能重用。不能用发送quiet替代接收消费完成。
- reference的w18→w30现在包含中转阶段，不能当作旧版本纯前缀加载；未添加新时钟。正式收益看完整OFF task。

## 目标机测试

两机同一提交，全rank完整wheel，不能混装协议14/15；不要加--skip-build。节点0/1分别设置NODE，MASTER为节点0真实IP。沿用已有SDK，不用重新编SDK。

```bash
git pull --ff-only
# 正确性和完整dispatch+BF16 combine性能；FP8配置的combine仍为BF16
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" paired-gateway-probe-001 \
  --mode perf --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

另用新run名加 `--route hotspot` 测热点。对照为父提交 `24e90d6`，两机切到该提交后同参数完整重编，使用独立结果目录。原始数据保留，仅回传入口生成的单个≤60000字节报告。需要回退本穿刺时revert本提交即可，路由SO的独立提交仍保留。

本地93项检查通过，包括生产packet owner规划、生产gateway函数的延迟尾FLAG/独立槽/零远端/单成员/多成员/重复使用模型及部署身份检查。尚未运行CANN编译或NPU正确性与性能测试。
