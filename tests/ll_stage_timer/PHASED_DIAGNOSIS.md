# 小 B 分阶段诊断与有限长度传输

分支 `kernel/ll-g4-small-b-prefix`，实现 `g4-small-b-parallel-prefix`，通信协议及收发顺序不变。
本次只修改诊断工具、探针和计时器自身；不宣称已恢复目标机 17–23 µs 的性能。

## 一个命令跑完多个 B

在目标工程中加载已有 `/home/tyf/0env.sh` Python/CANN 环境、准备好工程内锁定的 SHMEM SDK 和构建依赖后执行：

```bash
python tests/ll_stage_timer/diagnose.py \
  --run-name small_b_phases_001 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-port "$SHMEM_PORT"
```

这里的设备列表是用户目标机完整 EP 作业用法。授权验证机仅能使用 0、1，不能在验证机执行此 16 卡命令，也不能伪造 EP16。
单机时仅在已导出 `MASTER_ADDR`、`MASTER_PORT` 和 `DEEPEP_SHMEM_ENDPOINT` 时可以省略对应参数，仍须保留显式设备列表；入口在构建前检查端点，不自动猜端口。多机每台执行同一命令、仅修改 node-rank。
默认 B=1/2/4/8/16/24/32，M=32，H=3584，K=8，E=384，FP8，CPU sync=0，dispatch-only。
需要查看命令而不构建或启动设备时追加 `--plan`。

入口先用 **OFF wheel** 做正式 profiler 10 预热/100 采集/尾50；随后分别构建并采集 overview、prepare、prefix、send、recv。
每个 ON 组先稳定100轮，再采5轮；每轮使用已有 Exp+scalar all_reduce。可用 `--bs`、`--profiles`、`--diag-iters` 缩小矩阵。
例如 `--profiles overview prefix` 先观察总览和前缀。默认不采完整 `full`，避免同时开启全部细分探针。

结果放在 `diagnostic-runs/<run-name>/`；名字必须是新的，不清空旧结果。
每台机器在该目录内建立独立 system-site-packages venv，使用当前 Python 和已有 NPU 依赖；wheel 仅安装到此 venv。
不联网安装、不重建或更换共享 SDK。缺依赖时明确报错，受限网络机器先本地准备并上传 wheel。
`--skip-build --wheel-map <JSON>` 可复用本地 wheel，但仍核验安装后的实现、ON/OFF、profile 和源码指纹。
输出目录不能放入源码指纹扫描目录，以免结果/venv 被误当源码。

多机非共享文件系统时追加 `--defer-analysis`；全部采集结束后按原相对路径汇集各 rank 的 `sample_*.json`，保留 rank0 的 summary 和 active_capture 文件，执行：

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/small_b_phases_001
```

## 发哪些文件

先发送根目录的 `SEND_FIRST.txt` 和 `PERFORMANCE_COMPARE.txt`，然后按索引发送各 case/profile 的 `transfer/00_overview.md`、`priority_*.txt`。
完整 `full_*.txt` 保留全部已解码的每轮/每rank/每核指标、前两次真实区间、配置、频率和缺失信息；需要深入定位时再发送指定部分。
每个传输文件同时限制 **60000 字符和60000 UTF-8字节**，包括总索引、manifest和分块索引；不会靠截断丢掉后面的核或轮次。
分块带总数、顺序、bundle ID、SHA256，解析端可恢复并验证全量数据。

已有 schema13 原始 JSON 无需重采即可导出传输包：

```bash
python tests/ll_stage_timer/analyze.py <原run目录> --op dispatch --output-dir <新的分析目录>
```

已有大 Markdown/CSV 不包含完整原始记录时，工具不能补出未采集的数据。新细分阶段必须重装匹配 schema14 wheel 并重采。

## 如何判断探针扰动

`PERFORMANCE_COMPARE.txt` 的两列都来自 profiler **完整设备kernel task**。OFF为正式未插桩基准；ON窗口明确包住实际 `capture()` 调用，因此包括初始化、业务、最后 SyncAll 和 Flush。
此前普通 ON wheel 的正式 perf 调用仍传入空计时指针，不能用它替代“启用打点后的完整耗时”。
若导出的 task 名称、数量或起始时间顺序不明确，完整耗时标 unavailable，不用 Event/API时间或内部 KERNEL 代替。
两列取各rank均值中的最大值，ON样本较少，只用于观察扰动；不是算法加速比。

静态 Scalar 工作增加会推迟本核开始轮询，让上游更早完成，导致测到的等待缩短。因此：

- 先看 ON完整耗时/OFF，再看 overview 和更细组是否改变首发、首Probe、实际工作核数、轮询次数和ready比例。
- 同核 `INIT→首发`、`INIT→首Probe`、`INIT→KERNEL结束` 使用真实时间戳；`TIMER_SETUP` 在 INIT 之前单列。
- 不把各组阶段相加或相减；不同组也不能拼成同一次时间线。父阶段和子阶段不能重复求和。
- 等待少不证明链路变快，MTE下发时间不等于传输完成，跨rank时钟不做相减。
- 核数按实际 TX/RX计数显示，区分初始角色与实际任务；缺失阶段不填零。

## 当前探针覆盖与静态成本

`DEEPEP_LL_TIMER_PROFILE=full|overview|prepare|prefix|send|recv` 是编译期选择，默认 full；OFF统一归一化 full，不分裂OFF缓存。
宏编译掉未选中探针，累加数组只为选中阶段分配槽。ON但不采集时不清零数组，业务组件收到空计时器指针。
`TIMER_SETUP` 从计时器构造前开始，覆盖其本地初始化。该区间本身也有测量成本，不能与OFF直接作逐阶段归因。

| 组 | 主要覆盖 |
|---|---|
| overview | INIT/KERNEL/TIMER_SETUP、首发/首Probe、末尾排空、包数与实际工作核 |
| prepare | 布局、magic、组件初始化、count池、输入清零、scratch分配、路由权重搬入、q播种/更新 |
| prefix | count快照下发/等待/检查/提取、局部VF扫描、总数发布、前序总数等待/解码、offset处理、完整前缀发布和最终ready等待 |
| send | 输入、量化、metadata、打包、lane等待、发送提交与退休 |
| recv | 接收准备、flag探测/等待/归约/解码、payload/metadata、pending、relay与清理 |

并行前缀的 TOTAL_WAIT 与 READY_WAIT 现在分开，旧 WAIT_CUMSUM 仍是父层累计，不可重复求和。
R0路径保留原 do/while 条件，未单独拆出其条件中的 GetValue，`PREFIX_READY_DECODE` 在该路径缺失。
非去重路径末尾的 `counts_.Finish()` 已补 FINAL_DRAIN；没有为 observe 添加新的同步。
显式 relay-complete 仍会改变重叠，只支持 full/recv；默认流程始终 observe。

schema14为136阶段、192条真实事件、每核5632B、512B对齐；比schema13每核增加512B的独立GM诊断缓冲。
56核单次约308KiB，5轮约1.50MiB。业务UB分配及通信布局不变；计时器本地对象压力随profile变化，不能声称零扰动。
新增探针所在的count/prefix/sender/主流程剥离探针后业务token保持一致，仍需设备正确性验收。

## 性能目标与验证边界

用户提供的OFF profiler目标分别为 B1/2/4/8/16/24/32：17.5059、17.6091、18.1671、18.18013、19.18088、20.13838、22.60927 µs；
此前当前分支为38.28851375、38.68166875、39.386495、39.81751625、40.492945、41.397745、42.4848175 µs。
原基线commit不可恢复，所以先以同机器、同协议、同S/R的新OFF结果核实，再从完整准备/收发/收尾开销定位，不能只盯前缀。
分阶段诊断的完成不等于已达到性能目标，后续优化收益最终仍以目标拓扑OFF A/B为准。

独立计时器微基准见 [TIMER_OVERHEAD_PROBE.md](TIMER_OVERHEAD_PROBE.md)。它只验证本地计时/Scalar开销，不能替代EP16通信正确性和目标拓扑性能。

## 本次验证记录（2026-10-22）

- 本地纯CPU计划确认默认7个B、OFF先行、5个ON组均启用实际capture profiler；没有构建或启动设备。
- 27独立验证目录使用`/home/tyf/0env.sh`，完整CPU回归677通过；最后计时器裁剪调整后实际C++记录器/解析器45通过。
- full/overview/prepare/prefix/send/recv六组及OFF，各7种dispatch模式与Host选择桥均通过CANN编译。
- prefix组最终完整Host/Device wheel构建、工程内隔离venv安装、native profile/layout和源码SHA核验通过。未更改共享Python。
- 计时器微基准仅运行授权设备0，10次启动前均检查无既有NPU任务；结果及解释见TIMER_OVERHEAD_PROBE.md。
- 未运行完整EP16通信，也未在目标拓扑测量一键矩阵；不能将编译/CPU/micro通过写成目标通信正确性或已恢复性能基线。

验证材料保留于27的`/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/timer-profile-validation/`：
`.logs/pytest-suite-04.log`、`.logs/final-compile-*/timing.json`、`.logs/wheel-prefix-verified.log`、
`.tmp/wheels-prefix-verified/`、`.logs/timer-overhead-02/`。这些是验证结果位置，不是脚本硬编码依赖。
