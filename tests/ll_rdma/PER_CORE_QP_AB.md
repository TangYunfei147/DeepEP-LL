# 不限定8QP的独立基线

分支kernel/ll-ndr-ab-per-core-qp，协议22，直接基于kernel/ll-ndr-ab-base的f52ab46；不累加其他消融。保留路由SO、payload SO、q预计算、共同qPlan缓冲布局和原首跳专家选择，没有固定rank^16中转。

Session按实际active AIV数量设置每peer RoCE QP，物理核c只用QPc。以56 AIV为例：配置56QP；dispatch跨机包按每个实际目标peer在全部28个S核/QP0..27轮转；combine DIRECT由全部28个S核发送，partial由全部28个R核/QP28..55发送。计数恢复原实际核分片，每核提交和退休自己的QP。没有并发共享QP，也没有QP锁。

当前SDK原生上限56QP仍由初始化前检查保护；不改SDK、不扩大SDK上限。超过56 active AIV的跨机启动会被拒绝，不能称为无限QP。72仍只是LL控制容量。

每核128KiB源暂存、现有8包批次和每peer/QP quiet保留；8包批次不等于8QP限制。source生命周期、UB/lane/event、旧FLAG与双bank依赖保留。q中的QP字段扩为6bit，覆盖高编号发送核。对照两版的SDK、核数、shape和采样必须相同。

对比：f52ab46（8QP）与本分支（每物理核QP）。这项同时解除dispatch及combine的8QP分工限制，分别比较完整dispatch和combine task，不将两者变化混成单一发送收益。逐次Vector q回退、固定配对卡中转均不启用。

两机所有rank完整重编同分支wheel，不加--skip-build；使用新的run名，保留原始结果。NODE为0/1，MASTER为节点0真实IP：

```bash
git fetch origin
git switch kernel/ll-ndr-ab-per-core-qp && git pull --ff-only
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" per-core-qp-001 --mode perf \
  --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

FP8配置的combine仍为BF16。推荐与base交替ABBA，不能复用旧run名覆盖结果。CPU模型不代表CANN/NPU正确性或性能通过。
