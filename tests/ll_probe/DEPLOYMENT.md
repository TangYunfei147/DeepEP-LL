# 统一 YAML 通信域测试

统一入口为 `tests/ll_probe/run.py`。默认仅输出 CPU 计划，不启动设备；真正执行需要
`--target-run`（用户目标机）或 `--validation`（A5仅0/1）。每台主机分别启动一次，
没有远程 SSH、文件分发或伪造 rank 数。各节点通过 Git 获取同一提交并在本地构建。

## 配置与实际通信域

字段沿用参考测试工程的 `ips`、`rank_num_per_server`、`port`，增加独立 `shmem_port`：

```yaml
ips:
  - 192.0.2.10  # 改为节点0的可达IPv4
  - 192.0.2.11  # 改为节点1的可达IPv4
rank_num_per_server: 16
port: 29500        # torchrun/HCCL bootstrap
shmem_port: 29501  # SHMEM bootstrap，与上面端口不同
timeout: 600
output_root: results
```

实际 world = 主机数量 × 每机参与设备数。全局 rank = node-index × 每机设备数 + local-rank。
模板 `configs/ep2.yaml`、`ep4.yaml`、`ep8.yaml`、`ep16.yaml` 是单机；`ep32.yaml` 是两机各16卡。
32域必须多机，单机32配置拒绝。也兼容原network_probe的IPv4列表YAML，此时卡数可用
`--devices-per-host`明确设置。配置中的地址必须替换为当前机器的可达地址。
各机工程目录可不同，配置和CLI相对路径均从各自工程根解析；共享拓扑不含绝对输出路径。

## 所有测试的能力边界

| 统一 --test | 实际域2/4/8/16/32 | 测量范围 |
|---|---|---|
| empty | 全部支持，32多机 | 每rank真正空kernel；框架Exp/all_reduce使用实际world |
| tiling | 全部支持，32多机 | 每rank读取tiling，不传payload |
| comm | 全部支持，32多机 | SHMEM映射上的MTE通信，支持1→1、N→1、1→N、N→N |
| metadata | 全部支持，32多机 | 每个实际rank在本卡做相同logical source-rank=0的路由元数据计算 |
| native | 仅16/32；2/4/8明确unsupported | 生产完整dispatch/combine，保留生产协议约束 |
| network | 全部支持，32多机 | Session及指定两global rank的链路诊断，不是完整LL或带宽性能 |

metadata 的 `--partitions` 是逻辑专家分区，默认16，与实际进程world独立，且仍须满足
生产分组G和Host布局/UB限制。多rank模式仅使每卡执行本地提取，并用真实world做采样前的
Exp/all_reduce；**不包含跨rank计数交换、packet tail和receiver准备**，也不产生通信带宽指标。

旧 `bandwidth/tests` 的IPC入口只测同机两卡，旧SHMEM入口只支持两个rank；这是保留的诊断工具，
不是全域测试入口。2/4/8/16/32的正式带宽测试统一走 `--test comm`。`--bytes`按每条有向边计，
每边不超过64MiB，world×bytes不超过256MiB，因此EP32每边上限8MiB。
跨机必须支持所需SHMEM MTE peer映射；不可达时失败，不自动改成另一种通信协议。

## 构建

先按 [PATHS.md](PATHS.md) 完成仓内SDK/生产wheel流程，加载本机CANN并激活torch/torch_npu环境。
YAML解析使用requirements-build.txt中的PyYAML。所有新入口从脚本位置定位工程：

```bash
source scripts/project_env.sh
bash tests/ll_probe/profiling/build.sh
# 只有network测试需要另外构建：
bash tests/ll_probe/network_probe/build.sh
```

原生LL需要当前匹配的生产wheel，不能用探针重编译替代生产wheel构建。

## 单机与多机运行

先检查计划（不会导入Torch或启动rank）：

```bash
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml   --node-index 1 --test comm --plan
```

目标机0与目标机1分别执行同一命令，只改变node-index，run-id必须一致且本次全新：

```bash
# 节点0；节点1将 --node-index 0 改为1
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml   --node-index 0 --test comm --run-id ep32-comm-001 --target-run --   --scenario 4 --bytes 15392 --cores 8 --lanes 4
```

改用单机ep2/4/8/16模板时只有node-index=0。默认正式采样保持10次预热、100次采集、
每次Exp+scalar all_reduce、按kernel/task独立取最后50条，原始Task/Event/API时间分开。
不提供链路峰值时只报告有效带宽，不能将双rank结果冒充EP16/32性能。

其他目的复用相同部署配置：

```bash
# 空核；两机分别执行，node-index不同
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0   --test empty --run-id ep32-empty-001 --target-run
# 本地metadata，每rank相同logical source-rank0，逻辑分区显式32
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0   --test metadata --run-id ep32-meta-001 --target-run -- --partitions 32
# 完整LL正确性；同样需要两节点启动
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0   --test native --run-id ep32-native-check-001 --target-run -- --skip-perf-test
# 跨机链路诊断：global peer16位于节点1，避免默认peer1仅测同机
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0   --test network --run-id ep32-network-001 --target-run -- --peer 16
```

可转发参数必须放在 `--` 后。拓扑、rank、设备、endpoint、输出目录由统一入口管理，
不能通过转发参数覆盖。每节点结果位于 `results/<run-id>/node-N/`，含plan、launcher日志及本地状态。
launcher成功仅代表本节点；全局结论必须核对所有节点/所有rank。

## 汇总与复现

将各节点原始结果保留原目录结构汇集到同一工程results下（运行材料仍通过Git获取）。
对empty/tiling/comm/metadata：

```bash
python3 tests/ll_probe/profiling/analyze_profile.py   results/ep32-comm-001/node-*/data/rank_*/report.json   --output results/ep32-comm-001/summary.json
```

汇总拒绝缺rank、重复rank、不同run-id/拓扑/源码/采样配置，以及将同一物理主机伪装为多个节点。
允许不同节点原工程绝对路径不同。network使用network_probe/merge.py合并各node/data/report.json；
原生LL在全局rank0目录输出聚合summary，逐rank原始trace留在各节点本地目录，另核对全部launcher-report。

A5只做验证：

```bash
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/a5_validation.yaml   --test metadata --run-id a5-meta-001 --validation
```

该模式固定四个可见设备变量为0,1并使用小Exp与短采样；禁止用于目标性能结论。`--plan`
可以检查32域计划，但A5实际执行始终最多两个设备rank。旧tests/validate.py仍是A5专用入口。

## 本轮验证记录（2026-10-14）

CPU部署/报告/路径/LL启动合约141项通过，含全部目标域计划和EP32两节点命令。
A5只用真实0/1两rank、小Exp：metadata、empty、tiling及Session-only network执行和汇总通过。
comm在探针加载前的HCCL `RunAicpuCommInit`/`HcclAllreduce`初始化出现AICPU异常，
未执行被测通信kernel，不记为通过，也未重试采样。其余4/8/16/32实际运行、完整LL及目标性能
均待用户目标机验收。默认正式采样协议没有改变。
