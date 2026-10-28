# 路由SO独立回退与A/B

协议13为对照：8QP与全部S核池分布保持，配对路由2 put + 2 quiet。
下一独立提交应用路由SO：1 put + 1 quiet，保留源GM退休、完整尾FLAG检查、epoch、双bank、combine_reuse。
已推送历史不重写；先恢复对照，再以末提交单独应用优化。

仅EP32配对路由命中此差异。每个版本所有rank更新同一源码并完整重编wheel，不用--skip-build；SDK、shape、route、seed、OFF采样参数保持一致。独立结果目录保留原始数据。完整dispatch+BF16 combine的OFF task用于收益结论。
本地CPU检查不代表CANN或NPU验证。
