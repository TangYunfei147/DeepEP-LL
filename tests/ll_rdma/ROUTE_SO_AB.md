# 路由SO独立回退与A/B

协议13为对照：8QP与全部S核池分布保持，配对路由2 put + 2 quiet。
协议14独立提交应用路由SO：1 put + 1 quiet，保留源GM退休、完整尾FLAG检查、epoch、双bank、combine_reuse。
已推送历史不重写；先恢复对照，再以末提交单独应用优化。

仅EP32配对路由命中此差异。每个版本所有rank更新同一源码并完整重编wheel，不用--skip-build；SDK、shape、route、seed、OFF采样参数保持一致。独立结果目录保留原始数据。完整dispatch+BF16 combine的OFF task用于收益结论。
本地CPU检查不代表CANN或NPU验证。

## 目标机操作

A固定提交 `1343b3d`，B固定为 `24e90d6`（协议14），新增协议15穿刺不属于此A/B。A/B的差异仅为配对路由SO及必要layout、协议身份、测试和说明；8QP、packet归属、combine与SDK不变。

先在干净checkout拉取并使用固定B，两台机器使用相同完整SHA：

```bash
git pull --ff-only
ROUTE_SO_B=24e90d6bc62fdb7258dfdf5e6f6dd9816e751e0d
printf "%s\n" "$ROUTE_SO_B"
# 对照A：两台都切换到此提交并重编
# 节点0/1分别将NODE设为0/1，MASTER为节点0真实IP
git switch --detach 1343b3d && bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" route-so-A-001 \
  --mode perf --tokens 1 8 32 256 --dtype bf16 fp8
# 优化B：切到上面记录的完整SHA，两台都重编；使用不同run名
git switch --detach "$ROUTE_SO_B" && bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" route-so-B-001 \
  --mode perf --tokens 1 8 32 256 --dtype bf16 fp8
```

B记录后可按ABBA顺序测试，后两轮改用新run名；不要覆盖结果或自动重试性能。节点数2、每节点16rank得到EP32。默认参数和采样口径须完全一致。
若需要在分支上撤销优化，执行 `git revert <B的完整SHA>` 即可恢复协议13对照；所有rank仍须更新完整wheel。不要只替换kernel库。

本地验证：A与B分别执行路由CPU网络模型、计数路径、dispatch变体身份、one_click部署检查。未运行CANN/NPU性能；收益须由上述目标机OFF结果判定。
