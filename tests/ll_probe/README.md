> 2/4/8/16/32实际通信域及多机32的统一YAML入口见 [DEPLOYMENT.md](DEPLOYMENT.md)。各节点独立启动，共同run-id；旧双卡验证入口仍仅用于A5。

# LL 实验目录

代码按测试目的组织，C++ 设备代码放 `kernel/`，主机代码放 `host/`，验证/扫描脚本放 `tests/`。

| 目录 | 测试目的 | 构建与验证入口（相对本目录） |
|---|---|---|
| `communication/` | 第一层空载/tiling、第三层最简通信；共享同一通信核 | `communication/build.sh`、`communication/tests/validate.py` |
| `metadata/` | 第二层本地路由元数据提取与校验 | `metadata/build.sh`、`metadata/prepare_snapshot.py`、`metadata/tests/validate.py` |
| `bandwidth/` | 本卡、同机 IPC、两 rank SHMEM 搬运及扫描 | `bandwidth/build.sh`、`bandwidth/build_shmem_driver.sh`、`bandwidth/tests/` |
| `native_ll/` | 第四层完整 LL 的启动、API 测量适配和开销分析 | `native_ll/run_ll16.sh`、`native_ll/tests/`；生产核仍在仓库 `csrc/` |
| `network_probe/` | 多机 Session 链路诊断 | `network_probe/build.sh`、`network_probe/run.py`、`network_probe/tests/` |
| `profiling/` | 所有目的共用的正式采样、桥接与汇总 | `profiling/build.sh`、`profiling/profile_probe.py`、`profiling/tests/` |
| `four_layers/` | 跨层计划、构建与验证编排 | `four_layers/plan_matrix.py`、`four_layers/build.sh`、`four_layers/validate.py` |
| `contracts/` | 工程迁移及路径回归 | `contracts/test_paths.py` |

正式性能流程见 [PROFILING.md](PROFILING.md)，源码入口见 [KERNEL_MAP.md](KERNEL_MAP.md)。
路径迁移表为 [REORGANIZATION.json](REORGANIZATION.json)。旧源码入口已移动，请使用新路径。
新二进制按目的生成在各自 `build/`；元数据快照在 `metadata/snapshot/`，完整 LL 适配脚本在
`native_ll/snapshot/`。原有生成产物与原始数据保留，但新入口不读取旧构建目录。
所有命令仍按脚本位置定位工程根，不依赖调用时目录；含空格工程路径纳入回归。

CPU 回归（先加载本机 CANN 环境）：

```bash
python -m pytest tests/ll_probe
```

A5 验证只能使用 0/1，目标机器的完整 LL 与性能由用户执行。IPC 搬运不替代跨主机实验。

---

# LL四层测试：Git交付入口

生产LL每机16卡分组、同步协议和目标机验收见[LL16.md](LL16.md)，实际AIV核数与收发分工见[CORE_CONFIG.md](CORE_CONFIG.md)。

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
bash tests/ll_probe/bandwidth/build.sh
bash tests/ll_probe/bandwidth/build_shmem_driver.sh
bash tests/ll_probe/four_layers/build.sh
bash tests/ll_probe/profiling/build.sh
python3 tests/ll_probe/native_ll/tests/verify_adapter.py
```

路径规则见[PATHS.md](PATHS.md)。CANN使用当前机器已加载的环境；SHMEM/CATLASS等子仓只留在本工程third-party内，SHMEM统一由scripts/build_shmem.py构建。已有合法SDK可用--if-needed复用，不自动覆盖。具体设备启动见[四层运行指南](four_layers/USER_RUN.md)，早期双卡实验见[原指南](USER_RUN.md)。已有clone使用Git更新，勿覆盖现有目录。

## 注意事项与验证边界

- 默认写入工程目录；A5验证仍遵守其授权写入边界；必须显式设置SHMEM、Ascend日志和缓存路径。A5_241只使用物理卡0、1，仅验证正确性和可运行性；目标机实际实验由用户执行。
- 通信微基准域2/4/8/16/32；完整原生LL在此集合只支持16/32。双卡退化场景不证明N方竞争、跨主机可达性或目标带宽。
- 本地元数据实验仅提取dispatch sender路由准备，不包含跨rank计数、packet tail和receiver元数据。四层耗时不能直接相加或相减推导纯通信成本。
- 迁入前空载、tiling、元数据以及四种通信场景的双卡子集通过。完整LL仅CPU合约/适配器检查，目标设备运行待用户执行。
- 每轮记录git rev-parse HEAD、实际rank布局及全部rank原始CSV/log。后续目标机相关修改必须提交推送，再通过git pull --ff-only获取；有本地修改时先检查状态，勿强制覆盖。
