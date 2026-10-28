# 四版独立性能比较

| 分支 | 协议 | 路由SO | 固定配对卡中转 | q算法 |
|---|---|---|---|---|
| kernel/ll-ndr-ab-base | 18 | 开 | 关 | 一次预计算 |
| kernel/ll-ndr-ab-no-route-so | 19 | 关，2put+2quiet | 关 | 一次预计算 |
| kernel/ll-ndr-ab-paired-gateway | 20 | 开 | 开 | 一次预计算 |
| kernel/ll-ndr-ab-no-q-cache | 21 | 开 | 关 | 原逐次Vector归约 |

后三版分别与base比较，不是累积叠加。全部保留8QP、SDK及采样协议；共用独立qPlanBuf和归约scratch及相同发送UB预算，避免q回退破坏QP归属。基线以24e90d6为基础加入共同缓冲准备，因此不把历史24e90d6的成绩直接当作本次基线。中转版另有其必要GM/UB阶段和清理成本，计入穿刺结果。

第3项沿用用户确认的方案：所有合法BS的跨机包发往固定rank^16，再由配对卡组内分发；仍按实际packet发多个WQE、8QP轮转，不是把所有token合成一个WQE。

两机全rank相同分支/SHA，每次切换完整重编wheel，不加--skip-build；沿用同SDK、同shape/seed/route/采样参数及独立run名。测试EP32，BF16与FP8 dispatch+BF16 combine。NODE分别为0/1，MASTER为节点0真实IP。

```bash
git fetch origin
# 将BRANCH替换为上表分支；只在干净checkout切换
BRANCH=kernel/ll-ndr-ab-base
git switch "$BRANCH" && git pull --ff-only
# RUN分别使用base-001/no-route-so-001/gateway-001/no-q-cache-001等新名字
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" "$RUN" --mode perf \
  --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

先分别验证正确性再看完整OFF task；每项建议与base做ABBA，切换前等两机任务退出，保留全部日志，不自动性能重试。单个回传报告≤60000字节。reference诊断不能代替OFF性能。此材料未经过CANN/NPU验证。
