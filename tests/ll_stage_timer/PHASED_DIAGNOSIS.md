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

## 第一轮只发送一个文件

默认现在是`--transfer-mode summary`，只发送运行目录根部的 **SEND_FIRST.txt**，同时限制≤60000字符和60000UTF-8字节（含CRLF）。
这是用于初步定位的统计摘要，不是完整trace归档：统计读取全部rank/轮次，展示OFF/ON完整耗时、插桩扰动、S/R准备跨度、活动核数、包数均衡、就绪率、关键阶段和最慢核位置。
阶段详情有明确数量上限，原始JSON、CSV、trace和完整压缩包留在目标机。无需发送887个归档块，也无需重新采样。

已有完整采集可直接生成新的一页报告：

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/small_b_phases_002
```

将示例目录替换为实际run。也可直接执行`python tests/ll_stage_timer/triage_report.py <run目录>`，避免执行任何缺失case的原始解析。
根部SEND_FIRST会更新；原归档不删除，完整本地统计保存在`triage/<hash>/triage_summary.json`，无需发送此JSON。
35case/2800sample规模的合成压力检查保留每个case的核心表及阶段摘要，整个文本仍低于60000字节。若更大矩阵超出预算，会明确缩减可见详情或要求按B/profile选择，绝不自动拆成几百份。

## 后续按疑点提取原始细节

第一轮分析后，按报告中的B/profile/rank/iteration/core选择，例如：

```bash
python tests/ll_stage_timer/triage_extract.py diagnostic-runs/small_b_phases_002 \
  --profile prefix --bs 1 --rank 0 --iteration 0 --core 28 \
  --output-dir diagnostic-runs/small_b_phases_002/followup_prefix_b1
```

默认最多3块、每块≤60000字节；超过则要求缩小范围，不写超大输出、不偷偷抽样。
选择后的值和真实区间无损，保留原始core编号和active_aiv，明确标为部分视图。多条路由匹配时用`--case`指定报告的本地case路径。
可用`transfer_report.py show <提取目录> --part 1`逐块打印，接收端用`restore`恢复所选trace。

仅需要完整迁移/归档时显式选择：

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/small_b_phases_002 --transfer-mode archive
```

该模式仍可能产生很多文件；完整归档不是初次分析的前提。压缩格式与恢复步骤见[COMPRESSED_TRANSFER.md](COMPRESSED_TRANSFER.md)。

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
- 初始快照曾通过六组ON及OFF编译；随后修改RelayCompletionScope门控后只复验prefix ON，遗漏OFF。0f6f02e/72da8fe因此仍有OFF限定错误，不能将初始结果当作这两个提交的完整验证。修复与最终验收见[BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md)。
- prefix组最终完整Host/Device wheel构建、工程内隔离venv安装、native profile/layout和源码SHA核验通过。未更改共享Python。
- 计时器微基准仅运行授权设备0，10次启动前均检查无既有NPU任务；结果及解释见TIMER_OVERHEAD_PROBE.md。
- 未运行完整EP16通信，也未在目标拓扑测量一键矩阵；不能将编译/CPU/micro通过写成目标通信正确性或已恢复性能基线。

验证材料保留于27的`/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/timer-profile-validation/`：
`.logs/pytest-suite-04.log`、`.logs/final-compile-*/timing.json`、`.logs/wheel-prefix-verified.log`、
`.tmp/wheels-prefix-verified/`、`.logs/timer-overhead-02/`。这些是验证结果位置，不是脚本硬编码依赖。

## 编译报错后重跑

若遇到`ProfileEnabled: call to __host__ function from __aicore__ function`，更新本分支的限定修复后沿用原诊断命令，改用新的`--run-name small_b_phases_002`。
原失败目录保留；不需要清空构建缓存或重建SDK。构建系统会按头文件依赖重新编译受影响对象。
修复只统一打点helper的Host/AICore限定，不修改schema14、通信协议、收发顺序或压缩格式。

## 首轮摘要验证

完整相关CPU回归534通过、48项因本机缺少C++/CANN/POSIX环境跳过；最终摘要/选择/入口专项68通过。
35case/2800sample合成压力报告为47500字节（含CRLF），只有一个文件，所有case核心表均保留。
这是输出预算测试，不是对目标机这次测量的性能结论；拿到新的SEND_FIRST后再分析实际瓶颈。
