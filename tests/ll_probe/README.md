# LL四层测试：Git交付入口

源码定位见[脚本与kernel对应表](KERNEL_MAP.md)，包含Host入口、设备函数名、生产LL内部阶段及快照修改规则。

**性能采样已统一参考协议，正式测试请使用[统一profiler入口](PROFILING.md)。默认预热10次、采集100次、按kernel/task统计最后50条；下文旧Event命令用于诊断和正确性检查。**

目标机器只能通过clone/Git获取材料。所有目标机所需测试代码、脚本、配置和操作指南必须纳入本分支并提交推送；后续更新通过Git，不依赖ZIP/tar或未跟踪的project_control源码。生成的二进制、快照、缓存和原始结果不提交。

## 首次获取和构建

```bash
mkdir -p /home/tyf/Q4_remote_test
cd /home/tyf/Q4_remote_test
git clone --branch experiments/ll-four-layer-probes https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL
cd DeepEP-LL
git submodule update --init third-party/shmem
mkdir -p /home/tyf/Q4_remote_test/.tmp /home/tyf/Q4_remote_test/.cache /home/tyf/Q4_remote_test/project_control/results
cd tests/ll_probe
bash build_shmem_probe.sh
cd four_layers
python3 prepare_snapshot.py
bash build.sh
python3 verify_adapter.py
python3 plan_matrix.py > /home/tyf/Q4_remote_test/project_control/results/domain-matrix.json
```

前置环境：授权根目录内env_1009/cann-9.3.0，以及匹配仓库的框架/原生LL依赖。SHMEM的JSON依赖由构建脚本通过Git获取。具体设备启动见[四层运行指南](four_layers/USER_RUN.md)，早期双卡实验及依赖说明见[原指南](USER_RUN.md)。已有clone使用git fetch和切换本分支，勿覆盖现有目录。

## 注意事项与验证边界

- 所有写入限/home/tyf/Q4_remote_test；必须显式设置SHMEM、Ascend日志和缓存路径。A5_241只使用物理卡0、1，仅验证正确性和可运行性；目标机实际实验由用户执行。
- 通信微基准域2/4/8/16/32；完整原生LL在此集合只支持16/32。双卡退化场景不证明N方竞争、跨主机可达性或目标带宽。
- 本地元数据实验仅提取dispatch sender路由准备，不包含跨rank计数、packet tail和receiver元数据。四层耗时不能直接相加或相减推导纯通信成本。
- 迁入前空载、tiling、元数据以及四种通信场景的双卡子集通过。完整LL仅CPU合约/适配器检查，目标设备运行待用户执行。
- 每轮记录git rev-parse HEAD、实际rank布局及全部rank原始CSV/log。后续目标机相关修改必须提交推送，再通过git pull --ff-only获取；有本地修改时先检查状态，勿强制覆盖。