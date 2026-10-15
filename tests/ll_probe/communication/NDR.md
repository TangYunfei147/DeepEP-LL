# 通信探针：UB Memory / NDR

正式入口 `tests/ll_probe/run.py --test comm` 转发 `--transport ubmem|ndr`，默认 `ubmem`。
两种模式使用相同的域2/4/8/16/32、四种通信图、targets/sources、bytes和按字节的核间分片。
32域必须多机。这里只切换最简通信探针，不改变生产LL dispatch/combine。

## 数据路径与计时边界

| 模式 | 数据路径 | 设备task | 完成条件 |
|---|---|---|---|
| ubmem | GM→UB→SHMEM peer GM，原MTE循环 | four_comm_1..4 | 本核MTE发送流水收尾 |
| ndr | SHMEM ROCE QP put，NPU提交网卡传输 | four_ndr_1..4 | 每核、每peer的QP quiet |

模式在Host选择独立编译的库，payload循环不增加模式判断。UB原Run/启动包装保持不变。
NDR每核使用QP索引=core，对每条选中的边一次提交该核完整连续字节段，再quiet。
源和目标都在SHMEM对称内存；传入本地对称目标地址，由RDMA接口翻译到peer。
每核仅分配256B的UB描述符工作区，不把payload经UB分块搬运。`lanes/chunk`不调节NDR数据面，
仅保留公共tiling字段；`cores`同时决定分片数和每peer QP数量（当前探针1..32）。
最小32B分片可能使部分核无数据。不同peer依次提交/完成，当前不做跨peer并发队列调度。
quiet是必要的完成成本；该测量包含提交与完成，不声称是零开销网卡线速时间。
Host校验、跨rank barrier、分配和建域在目标task之外；Event/API/job_join分别保留。
empty仍是four_empty，tiling使用four_ndr_N但提前返回；二者不测NDR通信延迟。

## 每台目标机准备

先加载本机CANN 9.3.0，激活匹配的torch/torch_npu环境。以下命令在工程根执行。
本机CANN位置通过现有环境或显式CANN_ENV配置，不能复制开发机路径。
先停止使用本工程SDK的进程，再重建；所有节点使用相同Git提交与后端。

```bash
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
source scripts/project_env.sh
# 根据真实网卡/驱动选 XSCALE 或 HNS_1825；不要根据带宽名称猜测。
python3 scripts/build_shmem.py --rebuild --rdma-backend XSCALE
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python3 -m pip install -r requirements-build.txt
python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_probe/profiling/build.sh
```

首次没有SDK时可省略`--rebuild`。后续`--if-needed`会保留已记录的后端，不隐式换回RDMA关闭配置。
构建使用锁定SHMEM子仓的官方`-enable_rdma -rdma_backend`选项，不另建SDK依赖树。
当前支持的实现为XSCALE/HNS_1825；这里的NDR指NPU RDMA直通，不代表任意InfiniBand网卡兼容。
每台节点启动前，显式将`IBV_EXTEND_DRIVERS`设为实际已安装的provider库路径：
XSCALE通常为libxscale_nda.so，HNS_1825通常为libhrn5-rdmav34.so；以本机驱动为准。
本流程不安装驱动。网卡、端口、路由及SHMEM后端初始化必须在目标环境验收。

RDMA关闭时普通构建仍产出UB库并显示NDR skipped；实际选择ndr会在导入Torch前失败，
绝不回退UB。运行还会核对源码、SDK后端/版本/库hash及探针库hash，变更后必须重建。
`ndr_build.py --compile-check XSCALE|HNS_1825`仅编译链接检查，不产生可运行身份清单。

## 同机与跨机实验

先修改仓内`configs/ep2.yaml`或`ep32.yaml`的ips等部署参数。相对路径从工程根解析。
所有rank使用同一run-id、配置和参数，每次实验换新run-id。run.py根据每机卡数显式选卡。

```bash
# 同机2卡：先CPU计划，再由用户运行；将 --plan 换成 --target-run。
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep2.yaml --node-index 0 --test comm --run-id ndr-local-001 --plan -- --transport ndr --scenario 1 --bytes 15392 --cores 8

# 两机各16卡，0→16/18/31跨机；节点1运行相同命令，node-index改为1。
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0 --test comm --run-id ndr-cross-001 --target-run -- --transport ndr --scenario 3 --targets 16 18 31 --bytes 15392 --cores 8

# 汇集各节点结果后，全rank汇总
python3 tests/ll_probe/profiling/analyze_profile.py results/ndr-cross-001/node-*/data/rank_*/report.json --output results/ndr-cross-001/summary.json
```

对比UB时改为`--transport ubmem`并换run-id；按相同bytes/通信图/cores分别采样。
默认仍10预热、100采集、每次Exp+scalar all_reduce、按task取最后50条，P50/P90/P99分开报告。
配置中的UB硬件参考在ndr模式自动跳过，默认只报有效带宽；显式硬件模型必须带
`transport: ndr`，model中的容量必须来自实际NIC/拓扑，不能沿用UB芯片2016GB/s规格。
两种transport或不同SDK后端的报告不能混合汇总。

## 已验证范围

A5仅物理0/1、缩小Exp验证。XSCALE与HNS_1825设备源码及Host编译链接通过；
CPU用实际NDR函数检查分片、地址、独占QP和quiet，并检查无静默回退及报告模式隔离。
A5当前SDK的RDMA关闭，未重建SDK、未执行NDR网卡收发。编译通过不等于目标网卡可运行，
真实同机/跨机正确性与性能由用户按上述步骤执行，保存所有rank校验和原始报告。

本轮UB重建通过，双rank profiler smoke在探针调用前的HCCL初始化失败：
`RunAicpuCommInit / get kernel failed`。未获得本轮设备正确性通过结论，未重试采样；
日志保留在results/ndr-ub-validation.log及其引用的comm.log（生成日志不提交）。

## 结果 CSV 逐列说明

### 汇总 summary.csv

`profiling/analyze_profile.py --output results/<run-id>/summary.json` 默认同时生成同目录
`summary.csv`；可用 `--csv-output` 指定CSV路径。父目录须已存在，JSON/CSV输出文件须为新文件。
输入和输出建议使用同一个run-id目录，不要沿用示例中另一次运行的目录名。

以下20列按当前 `profiling/analyze_profile.py::write_table` 的实际输出顺序说明。
每个独立的 `(kernel, task_type)` 有P50/P90/P99三行，不是每rank一行。
对comm，先按每rank时间顺序取最后tail条，再对同序号的**全部rank**取task时长最大值，
最后对该序列求分位数（排序后线性插值）。即使某rank没有发送边，也参与此rank-MAX。
该值不是各rank分位数的最大值，也不是全局时间戳对齐的作业完成时间。

| 列名 | 单位/取值 | 含义 |
|---|---|---|
| `transport` | `ubmem` / `ndr` | 本次探针实际选择的数据面。 |
| `ndr_backend` | `XSCALE` / `HNS_1825`，或空 | NDR构建身份中的SDK后端；UB模式为空。 |
| `kernel` | 字符串 | profiler匹配的设备kernel名称；通信一般为`four_comm_1..4`或`four_ndr_1..4`，空核为`four_empty`。tiling与传输共用场景名称，须结合JSON中的layer判断。 |
| `task_type` | profiler原始类型 | 该kernel对应的task类型；不同类型独立统计，不相加。 |
| `samples` | 次 | 实际用于分位数统计的逐轮rank-MAX样本数，默认最后50条；不是rank数×50。 |
| `captured_samples` | 次/rank | 每个rank采集的目标task样本数，默认100；不含预热。仅有通信带宽汇总时填写。 |
| `bs` | 行/token数 | 用`--bs`指定的每条边逻辑行数；直接用`--bytes`时为空。 |
| `hidden_size` | 元素/行 | 用`--hidden-size`指定的每行元素数；直接用`--bytes`时为空。 |
| `element_bytes` | 字节/元素 | 计算发送量使用的逻辑元素大小；不是对探针pattern数据做真实模型dtype运算。直接用`--bytes`时此列也为空。 |
| `reference_kind` | `no_reference_config` / `ideal_capacity_lower_bound` | 无参考模型，或有容量模型推导的理想耗时下界。后者不是实测链路峰值。 |
| `reference_label` | 字符串或空 | 所用硬件模型的label，或显式峰值覆盖的说明；无参考时为空。 |
| `bytes_per_edge` | 字节 | 一次目标kernel调用中，每条选定有向边的payload量；形状输入时等于`bs × hidden_size × element_bytes`。不是每核发送量。 |
| `job_sent_bytes` | 字节 | 一次调用中所有选定有向边的发送量之和，即`bytes_per_edge × 边数`；不再把接收量加一次。不是100次采集的总量。 |
| `intra_host_sent_bytes` | 字节或空 | 上述发送量中同主机边的部分，根据记录的每机rank数和连续rank分配判断；缺部署拓扑时为空。 |
| `inter_host_sent_bytes` | 字节或空 | 上述发送量中跨主机边的部分；有拓扑且没有跨机边时为0，未知时为空。run-id中的cross不参与分类。 |
| `theoretical_time_us` | 微秒或空 | 对已配置且适用的各资源计算`字节量 / (容量GB/s × 1000)`，取最大值作为理想下界。不含启动、协议、串行化及拥塞成本。无参考时为空。 |
| `latency_percentile` | `P50` / `P90` / `P99` | 本行对应的时延分位数；P99表示约99%的所选时延样本不高于该值（按线性插值定义）。 |
| `time_us` | 微秒 | 本行分位数对应的逐轮rank-MAX kernel/task时长，不是Event、Python API或整作业墙钟时间。 |
| `effective_GBps` | 十进制GB/s | `job_sent_bytes / (time_us × 1000)`；1GB=10^9字节。它是在本行时延分位数处换算的有效payload带宽，**不是带宽分布自身的P50/P90/P99**。 |
| `reference_utilization_percent` | %或空 | `theoretical_time_us / time_us × 100`，相对所选理想模型的达成率；无参考时为空，不代表0%。不截断到100%，超出时需核对模型和测量口径。 |

只有comm的layer=3且只有一组kernel/task统计时，当前脚本才填充完整带宽汇总列。
empty/tiling或没有该汇总的情况只填写transport、backend、kernel、task_type、samples及分位时延，
其余列为空。NDR未提供NIC模型时仍有有效带宽，但理论耗时和利用率为空。

发送量只计目标探针payload，不含预热/正确性检查、Exp、HCCL all_reduce、NIC协议头或重传。
NDR时间包含QP提交与quiet完成等待；UB时间包含本核MTE流水收尾。
CSV的详细来源、每rank分布、原始rank-MAX样本、模型约束及配置保存在同名JSON中。

例：EP16单机，`--scenario 3 --targets 3 5 --bytes 15392`：每次调用两条边，
`job_sent_bytes=30784`，`intra_host_sent_bytes=30784`，`inter_host_sent_bytes=0`。
若某一行`time_us=10`，则`effective_GBps=3.0784`。这里10微秒仅是公式示例，不是实测结果。

### 每rank的 event_diagnostics.csv

这是Host输出的独立诊断表，位于`node-N/data/rank_R/`，不要与上面的summary.csv混为一谈。
经过profiler采集的Event/API数值受profiler影响，不作为无profiler基线。

| 列名 | 单位/含义 |
|---|---|
| `sample` | 从0开始的采集序号，不含预热。 |
| `rank` | 全局rank。 |
| `world` | 本次实际参与rank总数。 |
| `scenario` | 1：单源单目标；2：多源到0；3：0到多个目标；4：全有向无自环。具体选中边还需结合mask。 |
| `layer` | comm内部层号：0空核、1仅读tiling、3传payload；不是四层实验的顺序编号。 |
| `bytes_per_edge` | 每条边的配置payload字节数；layer0/1保留配置值但不实际发送。 |
| `rank_wire_bytes` | 本rank本轮实际配置的发送payload量；layer3为出边数×bytes，layer0/1为0。不是NIC线缆物理字节计数。 |
| `fanout` | 场景3的目标数；其他场景保留world-1的兼容值，不能直接当实际出边数。 |
| `peer_mask` | 十进制位掩码，第p位表示选中的全局rank p；场景1/3选目标，场景2选来源，场景4不使用该掩码。 |
| `event_us` | 微秒；ACL start/end事件包围目标kernel，在两次事件之间计时。 |
| `api_us` | 微秒；Host从记录start事件之前开始，到kernel提交、end事件记录及stream同步完成为止；包含这些Host调用成本。 |
| `job_join_us` | 微秒；与api_us起点相同，延伸到该rank完成后置SHMEM barrier；不是全局对齐的整作业makespan。 |
| `transport` | `ubmem`或`ndr`。 |
