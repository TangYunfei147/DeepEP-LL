# 路径约定与跨机器部署

所有脚本中的工程文件、默认输入、输出、日志、缓存和临时目录均以工程根目录为基准，
通过脚本自身的位置定位工程，不依赖调用时的工作目录、工程目录名称或开发机目录结构。
路径参数如 `--config tests/ll_probe/network_probe/config.yaml`、`--output results/case1`
均相对工程根目录解析。内部传给编译器/子进程的路径可以是运行时解析出的绝对路径；
不得把开发机器的绝对路径写死在脚本、生成脚本或使用命令中。

公共实现：`scripts/project_env.sh` 和 `scripts/project_paths.py`。
默认生成目录是工程内的 `results/`、`.tmp/`、`.cache/`、`.logs/`，全部忽略提交。
`PROBE_ROOT` 可显式指定另一个允许写入的根目录，相对值也从工程根解析；输出必须在此根目录内。
设备限制不因路径修改而放宽；A5验证始终仅0、1卡。

## 外部环境

脚本不能猜测别的机器上 SDK 的安装位置。运行前二选一：

1. 在当前终端加载该机器现有 CANN `set_env.sh`，使 `ASCEND_HOME_PATH` 和动态库路径有效。
2. 设置 `CANN_ENV` 指向现有 `set_env.sh`；支持相对工程根路径，也接受用户显式提供的绝对路径。

例如 SDK 与工程并列时（名称仅示例，请换成本机的真实相对路径）：

```bash
export CANN_ENV=../cann-9.3.0/set_env.sh
source scripts/project_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

所有子仓依赖必须留在仓内 `third-party/`。探针只链接
`third-party/shmem/install/shmem/lib`，不另建SDK、不使用外部SHMEM/CATLASS覆盖。
已有合法SDK通过 `python scripts/build_shmem.py --if-needed` 验证复用；需要重建时先停止使用者，
检查现有文件，再显式传 `--rebuild`，不自动删除旧SDK。
旧 `build_shmem_probe.sh` 仅转调同一个官方构建入口。依赖准备好后编译探针：

```bash
bash tests/ll_probe/build_probe.sh
bash tests/ll_probe/build_shmem_driver.sh
bash tests/ll_probe/build_profile_bridge.sh
```

不安装系统依赖、不修改驱动。
环境信息采集中的驱动版本位置通过 `ASCEND_DRIVER_PATH` 显式提供；未提供则报告未知。

## 从任意工作目录运行

可用相对路径找到脚本，脚本内仍以它所在工程为基准。例如当前目录位于工程旁边：

```bash
bash DeepEP-LL/tests/ll_probe/network_probe/build.sh
python3 DeepEP-LL/tests/ll_probe/network_probe/run.py \
  --config tests/ll_probe/network_probe/config.yaml --peer 19 --plan
```

`four_layers/run.sh` 进入工程根目录后执行传入命令，因此命令中的脚本/可执行文件参数也要
从工程根填写，例如 `tests/ll_probe/four_layers/build/comm_probe`。

## 结果搬移与快照

生成的四层LL适配器从自身位置反查工程，随工程整体移动后无需保留原绝对路径。
二进制及CMake缓存应在新目录重新构建，不复制旧构建缓存冒充可移植产物。
多机报告可以记录每台机器实际的绝对路径作为溯源信息；相同配置的跨rank比较不应比较
机器各自的输出目录名称。历史结果内的旧路径属于记录，不改写原始数据。
