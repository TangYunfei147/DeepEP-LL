# LL四层测试：Git交付入口

新增：[LL建域与网络层诊断](network_probe/README.md)，支持YAML多机配置、UDMA已选层查询和MTE双向正确性检查。

源码定位见[脚本与kernel对应表](KERNEL_MAP.md)，包含Host入口、设备函数名、生产LL内部阶段及快照修改规则。

**性能采样已统一参考协议，正式测试请使用[统一profiler入口](PROFILING.md)。默认预热10次、采集100次、按kernel/task统计最后50条；下文旧Event命令用于诊断和正确性检查。**

目标机器只能通过clone/Git获取材料。所有目标机所需测试代码、脚本、配置和操作指南必须纳入本分支并提交推送；后续更新通过Git，不依赖ZIP/tar或未跟踪的project_control源码。生成的二进制、快照、缓存和原始结果不提交。

## 首次获取和构建

```bash
# 在本机允许写入的目录中执行；工程目录名可以更改
git clone --branch experiments/ll-four-layer-probes https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL
cd DeepEP-LL
git submodule update --init --recursive --depth 1
# 先加载本机现有CANN；也可设置CANN_ENV为其set_env.sh的工程相对路径
source scripts/project_env.sh
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_probe/build_probe.sh
bash tests/ll_probe/build_shmem_driver.sh
bash tests/ll_probe/four_layers/build.sh
bash tests/ll_probe/build_profile_bridge.sh
python3 tests/ll_probe/four_layers/verify_adapter.py
```

路径规则见[PATHS.md](PATHS.md)。CANN使用当前机器已加载的环境；SHMEM/CATLASS等子仓只留在本工程third-party内，SHMEM统一由scripts/build_shmem.py构建。已有合法SDK可用--if-needed复用，不自动覆盖。具体设备启动见[四层运行指南](four_layers/USER_RUN.md)，早期双卡实验见[原指南](USER_RUN.md)。已有clone使用Git更新，勿覆盖现有目录。

## 注意事项与验证边界

- 默认写入工程目录；A5验证仍遵守其授权写入边界；必须显式设置SHMEM、Ascend日志和缓存路径。A5_241只使用物理卡0、1，仅验证正确性和可运行性；目标机实际实验由用户执行。
- 通信微基准域2/4/8/16/32；完整原生LL在此集合只支持16/32。双卡退化场景不证明N方竞争、跨主机可达性或目标带宽。
- 本地元数据实验仅提取dispatch sender路由准备，不包含跨rank计数、packet tail和receiver元数据。四层耗时不能直接相加或相减推导纯通信成本。
- 迁入前空载、tiling、元数据以及四种通信场景的双卡子集通过。完整LL仅CPU合约/适配器检查，目标设备运行待用户执行。
- 每轮记录git rev-parse HEAD、实际rank布局及全部rank原始CSV/log。后续目标机相关修改必须提交推送，再通过git pull --ff-only获取；有本地修改时先检查状态，勿强制覆盖。
