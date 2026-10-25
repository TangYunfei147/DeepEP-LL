# 纯 LL MTE 初始化修复与本轮测试

baseline主瓶颈采样与单文件回传入口见 [BASELINE_DIAGNOSIS.md](BASELINE_DIAGNOSIS.md)。

本轮使用 `commits-topology-mte.json`，只测新SDK下的 **baseline、buffered、immediate**
三项；准确SHA和比较reference见清单。原四项 `commits-topology.json` 保留为历史，
本轮不再加入旧SDK对照。

原问题发生在SHMEM初始化：ElasticBuffer无条件请求UDMA会话，虽然实际LL kernel走MTE，
仍触发新SDK缺少的HT sparse-QP检查。修复增加显式LL-only MTE会话，并让套件构建
`DEEPEP_BUILD_HT=OFF`。不编译HT实现，也不需要HT稀疏QP能力。
**不要通过设置 `DEEPEP_SHMEM_DENSE_QPS=1` 绕过问题。**

新路径在初始化前选择MTE；不按shape猜测后端，LL不支持的调用明确拒绝HT回退。
ST检查实际native会话为LL-only，记录`runtime_session=ll-mte`。
旧wheel缺少新factory时须重建匹配版本，不能用旧SHA覆盖式重打包伪装相同源码。

## 更新与SDK校验

从工程根目录执行，保留原始数据、wheel和旧SDK缓存：

```bash
git status --short
git fetch origin
git switch kernel/ll-g4-control-opt
git pull --ff-only
git submodule update --init --recursive --depth 1 -- third-party/shmem

# 使用已验证的项目Python/CANN环境。
source /home/tyf/0env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
```

有效且匹配的`8e3fc610` SDK只做校验复用，**无需重建SDK**。
如果校验报版本不符或文件损坏，先检查原因；套件不自动`--rebuild`。
已有依赖、日志、缓存路径要求沿用[拓扑指南](TOPOLOGY_EXPERIMENTS.md)。
不要修改旧suite cache或清理旧结果。Python须能在候选venv中加载匹配的Torch/torch_npu；
`--system-site-packages`不自动继承另一个外层venv内安装的包。

## 一次完成18次ST

沿用实际单机部署地址与16卡配置；下例的localhost只用于单机：

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
export SHMEM_PORT=29601
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology-mte.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-mte-001 --resume --plan
```

计划应为3个wheel、18次ST：3候选×3seed×2轮。确认后去掉`--plan`：

```bash
python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology-mte.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-mte-001 --resume
```

先完整构建全部wheel，再安装、检查空闲设备并依次运行；忙时停止，不杀其他任务。
仅编译可用同一清单加`--build-only --resume --send-cores 30`，之后同参数加`--run-only`启动。

主性能口径不变：EP16、B=M=256、H3584、K8、E384、S30/R26；10次预热、100次采集、
每rank尾50；每轮Exp+scalar all_reduce。FP8 dispatch与BF16 combine分别统计
`summary.csv`的`mean_kernel_us`。
请求seed0/17/101经3个正确性路由后，实际采样路由seed为2/19/103。
第1轮正序、第2轮反序，只在同轮同seed比较reference。

当前ST默认`--use-1g-huge-pages=0`。显式1GiB页仍可由MTE会话支持；本套件不额外改变
页参数。比较历史结果时同时检查页面设置和`runtime_session`，不能只看kernel是否相同。

## 续跑与结果来源

**使用新的run prefix `b256-topology-mte-001`，不复用旧混合MTE/UDMA初始化结果。**
旧manifest、wheel、sidecar、summary均保留；新提交、新构建能力和新会话身份独立记录。
`--resume`仅复用匹配wheel，不能把旧实验结果带入新清单。

本轮自身中断后，同清单、同seed列表、同S/R、同前缀加入`--resume-results`：

```bash
python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology-mte.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-mte-001 --resume-results
```

只复用全部rank/原始样本/正确性校验完整的项；未完成项在新attempt整项重跑。
恢复汇总写`results/ll-control-opt/b256-topology-mte-001/sessions/<id>/`，原始结果不覆盖。
初次完整运行的`comparison.csv/json`位于同run prefix根目录。
检查`reference_delta`、SDK hashes、会话/页参数和跨session标记后再解释小幅性能差异。

## 发回一个紧凑报告

回传限制为总数不超过10个文件，每个文件按UTF-8实际字节数不超过65535控制。
本轮完成后，只需导出并发送一个文件：

```bash
python tests/ll_control_opt/export_report.py \
  results/ll-control-opt/b256-topology-mte-001
```

发送命令打印的 `analysis-report.txt` 即可。它保留全部比较行、seed/轮次、
提交、算子、平均耗时、reference差值和会话/SDK身份提示；不修改原始结果。
用于先判断整体收益及多seed/多轮稳定性，不包含逐rank原始样本或物理端口利用率。
脚本在写入前核对实际字节数，超限直接拒绝，不截断数据；现有输出也不覆盖。
续跑后，将参数改为套件打印的最新 `comparison.csv` 所在 `sessions/<id>` 目录。
无需同时发送comparison.json、全部summary或profiler目录。

## 验证状态

当前本地Python/构建相关CPU检查为146项通过、14项因缺Torch跳过；
native Session生命周期检查42项通过。它们验证初始化路径、拒绝回退和资源生命周期，
不等于设备收发正确性或性能通过。
2026-10-25已在27授权隔离目录完成新SDK构建、111项CPU检查和下表全部完整wheel构建。
每个wheel均提取后单独Host导入，确认`high_throughput_enabled=False`、新LL MTE factory、
timer OFF/ON、协议19030和实际加载SDK的SHA256一致。没有创建真实Runtime或运行NPU。

| 版本 | 提交 | OFF完整wheel | ON完整wheel |
|---|---|---:|---:|
| baseline | `ed1c553` | 73.229s，通过 | 101.138s，通过 |
| buffered-c2 | `4d37eb6` | 49.362s，通过 | 129.027s，通过 |
| immediate-c2 | `c7d70a4` | 50.386s，通过 | 142.922s，通过 |

这些是顺序构建与打包的墙钟时间，包含缓存复用，不能作为冷编译或算子性能对照。
新SHMEM SDK独立构建35.955s；源提交仍为`8e3fc610`，未修改SDK源码。
LL算法与各自原候选一致；本次修复改变构建组成和会话初始化。

远端验证根目录：
`/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/topology-mainline-20261025/ll-only-fix-001`。
`.logs/validation.json`保存每阶段命令、退出码、时间与日志路径；`.logs/<label>-off/on.log`
为构建日志；`.artifacts/<label>-off/on/`保留完整wheel及导入检查用解包文件。
原部署与SDK未被本任务替换。目标机NPU正确性与性能仍需上述ST验收。
