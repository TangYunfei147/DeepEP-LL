# 历史LL实验与当前reference入口

当前打点已整套替换为[reference schema7](../ll_reference_profile/README.md)，旧stage/expert/sparse插桩、兼容解析和诊断入口已删除。当前只为dispatch记录29阶段及包明细；combine无打点。

本目录保留业务模型、核数/算法选择检查和固定Git历史实验：

- [NO_TIMER_AB.md](NO_TIMER_AB.md)：固定439d459的三wheel A/B，支持显式续跑；原采样口径不变。
- [CB59_ORIGINAL.md](CB59_ORIGINAL.md)：原始cb59固定源码重建/复测。
- [REFERENCE_INSPECTION.md](REFERENCE_INSPECTION.md)：只读检查用户已有参考部署。

其他旧指南保留历史算法说明和既有结果；其中旧打点/编译命令仅适用于其原提交，不能用于当前源码。当前构建、采样和导出请使用tests/ll_reference_profile/one_click.sh。旧原始timer数据只能从Git历史对应版本解析。
