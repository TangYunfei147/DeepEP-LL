> P50/P90/P99输出位置、汇总口径及其他入口缺口见 [PERCENTILES.md](PERCENTILES.md)。

> `run.py`所有参数、默认值、适用测试和不生效兼容参数见 [RUN_ARGS.md](RUN_ARGS.md)。

> 按bs/hidden_size控制发送量、P50/P90/P99、理论耗时及利用率配置见 [BANDWIDTH.md](BANDWIDTH.md)。

> 2/4/8/16/32实际通信域及多机32的统一YAML入口见 [DEPLOYMENT.md](DEPLOYMENT.md)。各节点独立启动，共同run-id；旧双卡验证入口仍仅用于A5。

# 统一性能测试入口

源码定位见[脚本与kernel对应表](KERNEL_MAP.md)，包含Host入口、设备函数名、生产LL内部阶段及快照修改规则。

参考：[ep_test_mega_moe_T profiling.py](https://gitcode.com/candy_cloud_fly/ep_test_mega_moe_T/blob/main/core/operations/mega_moe/performance/profiling.py)，核对版本 `844f17aa428d631b2ba7d767c8063ad289e86409`，以及同仓库 `core/profiling/kernels.py`。依赖已在本仓库实现，目标机不需要复制 example.py 或参考仓库。

## 采样口径

- 默认10次预热，在一个WARMUP step中完成；同步后profiler.step进入一个ACTIVE step。
- 采集100次，每次依次执行一次原地Exp、标量fill(1)、HCCL all_reduce(SUM)、目标操作。全部结束后barrier、NPU同步、profiler.step，一次导出。
- Exp缓冲默认FP32 `[118260,30720]`，约13.53 GiB/卡，独立于目标输入。它用于扰动缓存和推迟执行，不代表校准后的延时；重复Exp变为inf不影响目标输入。
- 按明确kernel名称和task类型分别保留100条原始记录、按时间排序、取最后50条，报告mean/min/max/p50/p95/p99及task wait。优先op_summary；若其中没有目标kernel才读取kernel_details，并记录实际来源，避免重复统计。
- 不混算不同kernel/task类型，不把task时间相加当端到端时间。逐轮rank最大task区间不是全局时间戳对齐的作业完成时间。
- C++探针有独立ACL stream，因此Exp/all_reduce后额外执行NPU同步，再启动目标kernel；单进程metadata/IPC没有跨rank all_reduce，记录为identity。这些差异写入报告。
- 旧版Event/Host CSV仍可用于诊断。通过profiler入口生成的Event值受profiler影响，不作为独立性能基线。完整LL快照仍另外采集无profiler的API耗时，并单独标注。

## 构建新探针入口

先按[路径及SDK流程](PATHS.md)通过Git获取分支、初始化仓内依赖，并加载本机CANN环境。所有命令从工程根执行：

```bash
# 从工程根目录执行
mkdir -p results
bash tests/ll_probe/bandwidth/build_sdk.sh
bash tests/ll_probe/profiling/build.sh
```

本入口需要现有torch/torch_npu；分布式采样需要HCCL可用。无需安装额外报告包。单机/跨主机数据面仍是SHMEM MTE可寻址映射；不可达时报告失败，不自动改传输协议。

## 第一、第三层：空载、tiling、通信

以下为同机双卡示例；正式采样默认10/100/50和参考Exp尺寸：

```bash
# 从工程根目录执行
export PROBE_VISIBLE_DEVICES=0,1
bash tests/ll_probe/four_layers/run.sh python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ll_probe/profiling/profile_probe.py \
  --kind comm --scenario 4 --layer 3 --bytes 15392 --cores 8 --lanes 4 \
  --endpoint tcp://127.0.0.1:19761 \
  --output-dir results/profile-comm-001
```

使用新输出目录。`--layer 0/1/3`分别为空kernel、tiling读取、传输。`--scenario 1/2/3/4`分别为1对1、N对1、1对N、N对N，`--bytes`为每条通信边字节数。场景3可加`--fanout N`，仅rank0向rank1..N发送，默认world-1；例如EP32下`--scenario 3 --fanout 3`仅有3条发送边。用profile_probe.py的--plan逐项检查2/4/8/16/32域、场景和发送量。layer使用0/1/3内部编号，不是四层的序号。

只有用户目标机可以扩展PROBE_VISIBLE_DEVICES。同机域4/8等用对应数量可见卡和`--nproc_per_node`；跨主机每台运行相同命令，将`--standalone`替换为：

```text
--nnodes=节点数 --node_rank=本节点序号 --nproc_per_node=本节点卡数
--master_addr=首节点可达地址 --master_port=独立Torch端口
```

SHMEM的`--endpoint tcp://首节点可达地址:另一端口`必须所有rank一致，不能与Torch端口相同。总rank数=节点数×每节点卡数，必须实际形成所请求的2/4/8/16/32域。各节点显式设置本地连续可见卡列表；torchrun的LOCAL_RANK选本地设备。多主机先确认HCCL和SHMEM在目标互联均可用。

## 第二层：dispatch本地路由准备

```bash
bash tests/ll_probe/four_layers/run.sh python3 tests/ll_probe/profiling/profile_probe.py --kind metadata \
  --batch 32 --hidden 7168 --top-k 6 --experts 256 --partitions 16 \
  --output-dir results/profile-metadata-001
```

仍只有device0执行，不创建16个设备rank。正确性版本先验证q/掩码/权重；正式计时仅提取`four_metadata`，不混入`four_metadata_check`。阶段范围及checksum观测开销见four_layers/USER_RUN.md。

## 旧版搬运实验

同一入口支持`--kind peer`（单进程、物理卡0→1，IPC）和`--kind shmem`（torchrun双rank，仅rank0发送）。使用`--mode 0/1/2/3`、`--chunk`、`--bytes`等旧版参数。SHMEM模式需要`--endpoint`；接收rank报告校验结果，不伪造发送kernel样本。两者的默认采样协议与上述一致。

## 第四层：完整dispatch/combine

LL使用查询的实际AIV核数和1:1分核，见[CORE_CONFIG.md](CORE_CONFIG.md)。每rank启动输出核心配置，case记录aiv_cores/send_cores/recv_cores，比较时必须保持这些配置一致。

本分支统一为每机16卡，单机EP16默认启动，跨机EP32使用两台各16卡。先按[LL16指南](LL16.md)重建生产wheel并运行带超时的正确性检查，再执行下述性能协议。旧每机8卡启动配置会被拒绝。 单机也必须显式配置MASTER_ADDR、MASTER_PORT和DEEPEP_SHMEM_ENDPOINT（或对应CLI参数）；完整环境表与旧入口迁移说明见[LL16指南](LL16.md#运行前环境每个新终端都要确认)。

`tests/elastic/test_ep_low_latency.py`已采用公共协议；`metadata/prepare_snapshot.py`生成的完整API快照继承同一协议。继续使用原LL启动参数，在本任务通信域集合中仅运行实际EP16/32。默认`--warmup 10 --num-iters 100 --perf-tail 50`；`--delay-rows/--delay-cols`显式控制Exp尺寸。

旧参数`--delay-exp-before/after`仅接受1/0；旧的首调用后1000次Exp方案会报错，不再静默执行。Gloo控制组可用于诊断，会标记非参考HCCL口径；性能比较应使用相同后端和扰动设置。完整LL仍需要匹配的已安装wheel及原生依赖，A5双卡无法完成其设备验收。

## 汇总与回传

```bash
# 从工程根目录执行
python3 tests/ll_probe/profiling/analyze_profile.py \
  results/profile-comm-001/rank_*/report.json \
  --output results/profile-comm-001/summary.json
```

必须收齐全部rank的报告及相邻native.log，不混配置。脚本重新验证原始条数、tail、rank集合、源码/二进制身份和校验记录，通信单task报告计算全作业字节/逐轮rank最大task区间。可选`--peak-gbps`和`--peak-description`输入与拓扑和方向一致的参考峰值，否则只报告有效带宽。不同task类型不强行算一个带宽。

保留全部rank目录：identity.json、report.json、native.log、trace原始CSV及event_diagnostics.csv。目标机测试结果不提交到Git；源代码、配置和指南均通过Git获取。完整LL沿用summary和每rank samples.json，并保留新增kernel_statistics。

## A5验证范围

公共`project_env.sh`不再默认设置设备0/1；A5专用shell入口使用`scripts/a5_validation_env.sh`，专用Python验证器显式固定四个可见性变量为0/1。目标机入口不得source该A5脚本。通用profile入口没有隐式双卡回退，目标机可显式设置PROBE_VISIBLE_DEVICES或继承现有ASCEND可见性。

`python3 profiling/validate_profile.py`强制卡0/1，小Exp `[64,64]`、warmup2/capture3/tail2，只验正确性、profiler导出和解析；不是目标带宽或默认负载性能证据。`--plan`不初始化NPU。大Exp默认尺寸、实际N方竞争、跨主机互联和原生EP16/32需要用户目标机验收。

场景1/3可用`--targets R [R ...]`指定全局目标rank，场景2可用`--sources R [R ...]`指定来源rank；固定根rank为0。显式targets与fanout互斥，详见[RUN_ARGS.md](RUN_ARGS.md)。
