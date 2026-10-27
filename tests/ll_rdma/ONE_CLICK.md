# 两机 NDR 一键构建、测试和本机解析

入口：`tests/ll_rdma/one_click.sh`。参考 `kernel/ll-g4-local-q-table@af094ce` 的
环境、完整ON wheel、稳定尾段采样与分析流程，使用本分支的NDR/SDK/计时schema。
两台目标机各执行一条命令；不通过SSH连接其他机器、不终止其他作业。

默认 `--mode reference` 是仅LL dispatch的参考式打点，使用独立reference wheel并关闭旧stage/expert计时；
默认1轮capture，自动生成逐核Excel。阶段定义、BS1命令和回传报告见
[LL dispatch参考式逐核打点](../ll_reference_profile/README.md)。

默认 EP32，每机设备0..15，B8/M256/H3584/K8/E384，BF16和预量化FP8两组，
默认仅dispatch，正确性检查保留；`--mode perf`执行dispatch+combine，FP8配置的combine仍是BF16。
此NDR性能分支仅编译LL，跳过与当前SHMEM内部UDMA接口不匹配的HT路径；
不会初始化CATLASS子模块。HT调用明确拒绝。

## 使用

两机的工程绝对路径必须相同，使用相同分支/提交、run名及参数。
先在各自工程根执行 `git pull --ff-only`，并加载现有CANN环境；
也可设置 `CANN_ENV` 指向实际 `set_env.sh`，脚本会加载它。
本脚本不自动切分支、合并或覆盖用户源码。

首次升级到当前SHMEM主线版本，先停止使用该工程SDK的作业，然后两机分别执行：

```bash
# 节点0：把 192.0.2.10 替换为节点0的真实可达IPv4地址
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-b8-001 --rebuild-shmem

# 节点1：使用同一个节点0地址和run名
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-b8-001 --rebuild-shmem
```

普通重测换run名并去掉 `--rebuild-shmem`，脚本会校验复用匹配SDK并构建wheel。
已安装与当前源码、SDK、reference模式完全匹配的wheel时，可加 `--skip-build`。
不要同时使用这两个选项。

默认使用三个不同端口：29500（HCCL启动）、29501（SHMEM）、29502（两机准备确认）。
可用 `--master-port`、`--shmem-port`、`--ready-port` 修改，两机配置必须一致。
准备确认只交换版本和参数JSON；不使用Gloo/TCPStore，不参与算子计数、数据通信或采样。
各机完成构建/检查后最多等待另一机1800秒，可用 `--ready-timeout` 调整。
如果一机准备失败，另一机的等待有超时；测试失败不自动重试。

## 流程与结果

脚本定位自己的工程根，执行：

1. 创建/复用工程 `.venv-ll-rdma`，继承目标已有torch/torch_npu。
2. 获取锁定子模块、校验/显式重建SDK、安装构建依赖和完整wheel。
3. 检查当前源码fingerprint、SDK版本/实际PI模式、安装包NDR协议10、G16及reference模式。
4. 用既有ST生成CPU计划，执行RDMA环境前检。
5. 两机核对工程路径、Git版本、源码、SDK与完整测试参数，一致后进入原HCCL入口。
6. 运行测试；reference模式成功后自动分析各自本机16rank，不要求复制另一机样本。

2026-10-26：允许两机 `hns_1825_pi_on_chip` 不同，只从跨机比较副本中移除该字段。
各机identity.json仍保留实际PI值，且每台机器的SDK与wheel必须匹配同一个有效0/1值。
不用为了跨机同值重建SDK，也不能手改manifest；实际模式仍须匹配本机固件，
SDK探测fallback不能当作硬件验收。
其他源码/协议/参数差异仍拒绝，错误会列出具体字段及两机值。
`source_sha256`涵盖构建脚本、测试与源码；相同Git提交但已跟踪文件有本地修改仍会不一致。
Git checkout中只纳入已跟踪测试文件，其他分支遗留的未跟踪实验目录保留但不进入指纹；
实际构建目录中的源码/头文件仍完整计入。根目录的 `core` 文件从未参与该指纹。

默认reference模式：reference ON wheel，10次预热、100次稳定调用、1轮独立capture。
每个批次只在开始做一次all_reduce；预热后固定100次Exp，再连续稳定调用和capture，算子之间无采样all_reduce/Exp。默认Exp缓冲约13.53GiB/卡，此模式不运行正式profiler。

结果目录：`results/ll-ndr/<run名>-reference/`。
每机的报告在其下`table-analysis/`，统一打开`stage-times.xlsx`；多个case按sheet前缀区分，
详细索引为`TABLE_INDEX.json`。`stages.csv`改为所选末轮每核一行、阶段为列，
多轮均值另存`stages_mean.csv`，原长表保留`stages_long.csv`。
已有样本只重导出：`bash tests/ll_rdma/one_click.sh --analyze-only RUN_DIR --output-dir NEW_DIR`，
无需CANN或重编/重采。完整说明见[TABLE_EXPORT.md](TABLE_EXPORT.md)。
命令及各步日志在 `.logs/ndr-one-click/<run名>-reference/node-0|1/execute/`。
原始样本、错误日志保留，已有run或日志目录拒绝覆盖，重跑使用新run名。

## OFF性能与参数调整

两机同样增加 `--mode perf`，脚本安装OFF wheel，执行10预热/100采集/尾50统计。
rank0输出原有完整性能汇总；各节点保存本地profiler原始数据。
OFF模式不调用阶段解析器，也不接受误装的ON wheel。

```bash
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-perf-001 --mode perf
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-perf-001 --mode perf
```

常用覆盖，两机必须相同：

```bash
# 多B、只测FP8；其余保持默认
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-bs-001 --tokens 1 2 4 8 16 32 --dtype fp8
# 先准备环境，不加入通信；随后相同run名加--skip-build执行
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-ready-001 --prepare-only
# CPU计划，不构建、不下载、不建venv、不启动NPU
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-plan --plan
```

还有 `--capacity`、`--hidden`、`--topk`、`--experts`、`--devices`、`--route hotspot`、
`--reference-iters`；`--help`列出全部选项。每机必须提供16个实际设备，物理机界不改变。
所有日志/缓存/临时/结果保持工程内，相对CANN/provider路径也按工程根解析。

本地验证仅覆盖CPU计划、调用顺序、失败停止、身份拒绝及TCP准备握手。
不代表目标CANN编译、RDMA可达性、NPU正确性或性能已通过。
