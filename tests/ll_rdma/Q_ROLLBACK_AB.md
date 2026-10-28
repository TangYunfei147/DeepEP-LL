# q算法回退A/B

A固定 `3f80050`，协议16为对照准备：q缓存与8QP归属迁入独立qPlanBuf，保留原qDst/qSub/qWork三块归约临时区。Host同步增加一份align32(B*K*4)预算，两版UB和lane容量相同。它可能与60f8bec的lane数不同，所以不要直接把后续回退版与60f8bec当作纯q A/B。

B为本指南所在协议17提交，独立回退逐次Vector前缀归约；对照协议16保留LlBuildRouteQ一次Scalar建表。两版保留协议15固定配对卡中转、8QP、原q语义及combine。稀有显式非去重大域路径改用同序server扫描以释放qSub链表；这项在两版一致。

同SDK、同shape/seed/route、同采样参数，全rank完整重编OFF wheel；独立run名。建议EP32 B1/2/4/8/16/32/64/128/256、BF16与FP8 dispatch+BF16 combine。B1旧版和新版都直接返回q=0。

```bash
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" q-ab-A-001 --mode perf --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

B使用q-ab-B-001等独立run名，建议ABBA。不用--skip-build，保留原始结果，仅回传入口生成的≤60000字节报告。本地CPU检查不代表CANN/NPU或收益通过。

B不再调用LlBuildRouteQ。8QP分配自身的peer计数表仍保留，不能把它当作q hash。旧Vector函数逐次执行原Duplicate/Sub/Abs/Mins/ReduceSum/V_S；独立qPlan防止归约临时区覆盖QP归属。

对照切换：两机先记录B完整SHA，再 `git switch --detach 3f80050` 编译测试A；随后切回记录的B SHA编译测试B。不要使用git reset丢弃本地改动。两版各自使用独立run名，全部rank版本一致。

`tests/ll_route_q_bench`中的prod参考仍调用旧缓存算法，协议17下不能将其标为当前生产q；本次以完整OFF的dispatch+combine对比为准。两版均未执行CANN/NPU验证。
