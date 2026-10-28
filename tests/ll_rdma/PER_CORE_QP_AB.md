# 原始分片与每核QP基线（协议24）

当前分支kernel/ll-ndr-ab-per-core-qp已按用户要求撤销所有8QP派生实现。原四条独立实验分支不改。

- 恢复按原token/路由分片分配S核；每核按原FIRST_HOP、DIRECT顺序执行。没有按peer的packet轮转，也没有独立“先跨机、后机内”阶段。
- 删除packet owner/QP规划、跨分片GM权重读取、qPlanBuf及其Host预算；恢复原UB布局。q使用逐次Vector前缀归约，不启用预计算。
- combine恢复8QP改动前的选择/分工，删除为4+4发送者引入的分页与显式QP映射。计数恢复原核分片；Host预算恢复原公式。
- QP按实际AIV数配置，每核默认使用自身QP；SDK原生56QP上限保护不变。原8包批次是源暂存批次大小，保留，不是8QP限制。
- 路由SO整帧1put+1quiet及payload SO保留；源生命周期、原UB/lane/event与双bank复用保持。SDK未改。
- 保留reference-only打点及用户tmp.py式normal路由，hotspot原方式不变。reference w43恢复原本核路由任务数，不能与此前“源任务+remote任务”口径混用。

比较旧提交b50d599时，仍有路由SO、q回退、旧打点源码删除及测试负载差异；不能当作单变量性能比较。

两机相同最新提交，全rank重编完整wheel，不加--skip-build。使用新的run名：

```bash
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" original-partition-001 --mode perf \
  --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

NODE=0/1，MASTER为节点0真实IP。原始结果保留，只回传入口生成的紧凑报告。CPU检查不代表CANN/NPU验证。
