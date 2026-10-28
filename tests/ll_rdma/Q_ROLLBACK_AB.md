# q算法回退A/B

协议16为对照准备：q缓存与8QP归属迁入独立qPlanBuf，保留原qDst/qSub/qWork三块归约临时区。Host同步增加一份align32(B*K*4)预算，两版UB和lane容量相同。它可能与60f8bec的lane数不同，所以不要直接把后续回退版与60f8bec当作纯q A/B。

随后协议17独立回退逐次Vector前缀归约；对照协议16保留LlBuildRouteQ一次Scalar建表。两版保留协议15固定配对卡中转、8QP、原q语义及combine。稀有显式非去重大域路径改用同序server扫描以释放qSub链表；这项在两版一致。

同SDK、同shape/seed/route、同采样参数，全rank完整重编OFF wheel；独立run名。建议EP32 B1/2/4/8/16/32/64/128/256、BF16与FP8 dispatch+BF16 combine。B1旧版和新版都直接返回q=0。

```bash
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" q-ab-A-001 --mode perf --tokens 1 2 4 8 16 32 64 128 256 --dtype bf16 fp8
```

B使用q-ab-B-001等独立run名，建议ABBA。不用--skip-build，保留原始结果，仅回传入口生成的≤60000字节报告。本地CPU检查不代表CANN/NPU或收益通过。
