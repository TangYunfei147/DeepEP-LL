# 小 B 稀疏时间戳诊断与有限长度传输

当前分支 `kernel/ll-g4-small-b-prefix`，实现 `g4-small-b-parallel-prefix`，使用 **schema15 / sparse**。
本次仅改变诊断记录与解析，保持收发逻辑、原有业务同步顺序和通信协议不变；删除了仅用于旧计时器导出的末尾SyncAll。不宣称已恢复目标机17–23 µs性能，也不把CPU验证作为设备验收。

## 一个命令跑多个 B

所有节点先更新到同一提交并保留已有结果。使用目标机已经可用的Python/CANN环境；本会话环境为`source /home/tyf/0env.sh`。
工程内锁定的SHMEM SDK和构建依赖应已准备好。本轮用户配置是S30/R26，**显式保持该比例**：

```bash
source /home/tyf/0env.sh
export DEEPEP_LL_SEND_CORES=30
export DEEPEP_LL_RECV_CORES=26
```

在工程根目录执行下面一个诊断命令。沿用实际作业的`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`、`SHMEM_PORT`；单机EP16为`NNODES=1`、`NODE_RANK=0`，端口须为本次分配的端口。

```bash
python tests/ll_stage_timer/diagnose.py \
  --run-name small_b_sparse_001 \
  --bs 1 2 4 8 16 24 32 --m 32 \
  --hidden 3584 --topk 8 --experts 384 --dtype fp8 --cpu-sync 0 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-port "$SHMEM_PORT"
```

这些设备是目标机实际EP作业的示例，不扩大任何验证机的设备授权。多机每台执行同一命令，仅修改node-rank，并确保实际总rank数与期望EP一致。
入口继承构建环境的S/R，不会把用户配置改成默认28/28；OFF与sparse必须一致。只检查CPU计划可追加`--plan`，不会构建或启动设备。

入口顺序执行两组：

1. **OFF**：完整未插桩wheel，正式profiler 10预热/100采集/尾50。
2. **sparse**：完整schema15 ON wheel，先稳定100轮，再采5轮；实际capture调用也被profiler包住，用来量化打点扰动。

每轮沿用Exp+scalar all_reduce，dispatch-only。`--bs`和`--diag-iters`可以缩小矩阵；新采集唯一profile为`sparse`，省略`--profiles`即采用它。
旧`full/overview/prepare/prefix/send/recv`只支持已有数据离线读取，不再用于新采集。旧wheel或旧执行计划会因身份不符被拒绝。

**schema15必须全rank重编完整wheel并使用新的run-name**；不强制重建已经校验匹配的SDK，不清空旧结果或构建缓存。
默认一键入口会构建OFF和sparse wheel，并分别装入该运行目录的system-site-packages隔离venv，复用已有torch/torch_npu，不改共享Python、不联网安装依赖。
网络受限机器缺少依赖时先在本地准备并上传wheel。`--skip-build --wheel-map <JSON>`只用于已有匹配的OFF/sparse wheel，仍检查schema、profile、S/R和源码指纹。
结果位于`diagnostic-runs/<run-name>/`，输出目录不能放入源码指纹扫描目录。

多机非共享文件系统可用`--defer-analysis`。完成后按原相对路径汇集各rank的sample文件，保留rank0的summary和active_capture报告，再执行：

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/small_b_sparse_001
```

这一步纯离线，不重建或启动设备。旧schema7～14数据也可离线解析，但旧数据不会变成新稀疏测量，不能用旧ON时序代替新采集。

若ca7bd2f版本在生成trace时报`Mixed sparse capture identities`，原因可能是旧解析器把每rank/轮次不同的`active_capture_profile.kernel_us`误当成共同身份。
更新修复后对原suite执行上面的`--analyze-existing`即可复用原始采样，无需重编wheel或重跑设备。修复仅比较测量口径，保留shape、构建、S/R、schema及profile检查；真正不一致时错误会列出字段名。

## 每核只保留少数首次时间戳

每核独占一个512B、512B对齐的诊断槽；实际只发布前192B，56核每次分配28KiB、发布10.5KiB。
不再维护136阶段的累计数组、调用计数或前两次事件列表，不再在每包结束时扫描profile索引。
每点只保留首次出现，最多读取14次时钟；未执行的点不生成时间戳，解析时不补零。combine只记录ENTRY、INIT_DONE、WORK_DONE，不声称采集了其收发细分。

| 时间点 | 定义 |
|---|---|
| ENTRY | 入口开始，在计时器对象初始化前读取，因此INIT包含timer初始化成本 |
| INIT_DONE | 初始化完成，业务主流程开始 |
| COUNTS_SENT / COUNTS_RECEIVED | 现有计数发送步骤结束 / 本核计数接收步骤结束 |
| PREFIX_LOCAL_DONE / PREFIX_READY | 本核前缀阶段的原有边界 / 本核前缀准备完成，可进入后续流程 |
| SEND_PREPARED / FIRST_SEND | 发送准备完成 / 本核首次发送下发之前 |
| SEND_DONE | 本核已有发送资源退休步骤结束；共享接收时可能晚于RECV_DONE |
| RECV_PREPARED / FIRST_RECV_PROBE | 接收准备完成 / 本核首次接收Probe之前 |
| FIRST_RECV_READY / RECV_DONE | 首次观察到可消费接收任务 / 有任务接收流程的原有收尾完成 |
| WORK_DONE | 业务结束，位于本核诊断导出之前；没有额外诊断SyncAll |

`INIT=ENTRY→INIT_DONE`，`KERNEL=INIT_DONE→WORK_DONE`。其余报告使用明确的同核端点名称，如`COUNTS_TO_PREFIX_LOCAL`、`FIRST_PROBE_TO_FIRST_READY`。
这些是首次事件之间的跨度，不是逐包累计时间。`FIRST_SEND→SEND_DONE`可能包含与接收重叠的工作，**不是纯网络传输耗时**。
小B并行路径的PREFIX_LOCAL_DONE位于既有V_S之后；去重R0路径位于既有V_MTE3之后，只表示该阶段指令下发到此处，不代表Scalar已确认VF物理完成。诊断不为统一这两个含义而增加同步。
发送和接收可以交错，解析不要求按表格顺序全局递增，只验证同核合理依赖和端点范围。保留原始ticks；用户目标频率按1GHz转换，其他频率需明确传入。

## 一次采集能同时排查哪些问题

同一轮稀疏数据可以并行分析：初始化、计数准备与等待、前缀完成、首次发送前准备、发送资源退休、首次Probe/首次ready及接收收尾。
这些分析共享同一组实际时间戳，不需要为每个疑点再开一组密集探针。报告同时列出初始S/R组及实际记录FIRST_SEND、FIRST_RECV_PROBE、FIRST_RECV_READY的核心数量。
**包数、逐包均衡、轮询次数、ready/probe比例及各小段Scalar成本没有采集**，缺失不等于零，也不从首个事件推算。

源码审计、候选方案设计和同份数据的多个分析可以并行；设备OFF A/B必须各自独立、串行运行，避免作业互相干扰。
若要分别验证小VF合并、初始化裁剪或前缀策略，应各自独立提交并固定同一S/R与采样参数，不把多个改动混成一个无法归因的性能结果。

## 如何判断剩余打点扰动

摘要OFF与active ON两列均来自profiler的**完整设备kernel task**，后者包含实际capture的初始化、业务和本核Flush。
未启用capture的ON wheel调用不能替代active ON；内部KERNEL跨度也不能替代完整profiler耗时。任务身份或时间顺序不明确时标为unavailable，不以Event/API时间补齐。
两列为各rank均值中的最大值。ON只有少量诊断轮，用于观察扰动，不是算法加速比。

先看完整ON/OFF，再解释同核区间。打点仍可能推迟本核进入轮询，从而缩短观测到的等待；稀疏设计减少了操作量，**尚不能声称零扰动或已获得目标机实测收益**。
跨度可以嵌套/重叠，不能相加；跨rank时钟不相减；旧profile和新sparse不能拼接时间线。

## 只发送必要内容，累计不超过10次

第一轮只发送运行根目录的 **SEND_FIRST.txt**，同时限制≤60000字符及60000 UTF-8字节（含CRLF）。
它读取全部rank/轮次，包含OFF/ON扰动、全部B的S/R跨度、实际首次事件核心数量和必要的最慢核定位；七B合成稀疏摘要小于20KB。
本地原始JSON、详细CSV、trace、完整压缩包、ledger、manifest、索引和独立性能文件不发送；`--transfer-mode archive`禁用。

同一suite的摘要和必要补充**累计最多10个文本文件，按每文件一次发送计数**。工具按新生成的待发送内容预留额度，相同内容不重复扣额或要求重发，换输出目录不重置。
首次接入已有suite时，已有SEND_FIRST保守计作一份；不同的新摘要再占一份。实际发送由用户执行，工具无法观察是否已经粘贴，因此不要重复发送已给过的内容。
达到上限即拒绝新增待发送内容，不通过增加归档文件绕过限制。只进行离线复核时可直接使用`triage_report.py <run目录>`。

只有分析后仍存在明确疑点，才定点提取必要原始值。profile、B、rank、iteration必须明确；core可选。例如：

```bash
python tests/ll_stage_timer/triage_extract.py diagnostic-runs/small_b_sparse_001 \
  --profile sparse --bs 1 --rank 0 --iteration 0 --core 30 \
  --output-dir diagnostic-runs/small_b_sparse_001/followup_sparse_b1
```

每次默认最多3块，并受累计剩余额度限制；每块同样≤60000字符/字节。只发送该命令列出的新part，默认不需要任何补充。
所选时间戳及端点跨度无损，保留原core编号和部分视图标识，不把所选核心误作全域结果。具体还原见[COMPRESSED_TRANSFER.md](COMPRESSED_TRANSFER.md)。

## schema15当前验证边界（2026-10-23）

本地CPU覆盖固定槽解析、身份和时序拒绝、旧schema读取、压缩恢复、trace、七B摘要及10次额度；它们不是CANN或NPU验收。
本次对26、27各尝试一次SSH连接均超时，未完成schema15 CANN编译、完整wheel链接或设备测量；下方历史schema14的成功编译不能视为本次验收。
最终CPU回归数量与编译证据统一记录在[BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md)。
独立计时器微基准见[TIMER_OVERHEAD_PROBE.md](TIMER_OVERHEAD_PROBE.md)，仅验证本地计时/Scalar，不替代目标拓扑EP16正确性或性能。

---

以下为历史性能目标和旧schema验证材料，保留作追溯；旧profile命令与“无需重编”结论不适用于本次schema15升级。

## 历史性能目标与验证边界

用户提供的OFF profiler目标分别为 B1/2/4/8/16/24/32：17.5059、17.6091、18.1671、18.18013、19.18088、20.13838、22.60927 µs；
此前当前分支为38.28851375、38.68166875、39.386495、39.81751625、40.492945、41.397745、42.4848175 µs。
原基线commit不可恢复，所以先以同机器、同协议、同S/R的新OFF结果核实，再从完整准备/收发/收尾开销定位，不能只盯前缀。
分阶段诊断的完成不等于已达到性能目标，后续优化收益最终仍以目标拓扑OFF A/B为准。

独立计时器微基准见 [TIMER_OVERHEAD_PROBE.md](TIMER_OVERHEAD_PROBE.md)。它只验证本地计时/Scalar开销，不能替代EP16通信正确性和目标拓扑性能。

## 历史schema14验证记录（2026-10-22，不是schema15验收）

- 本地纯CPU计划确认默认7个B、OFF先行、5个ON组均启用实际capture profiler；没有构建或启动设备。
- 27独立验证目录使用`/home/tyf/0env.sh`，完整CPU回归677通过；最后计时器裁剪调整后实际C++记录器/解析器45通过。
- 初始快照曾通过六组ON及OFF编译；随后修改RelayCompletionScope门控后只复验prefix ON，遗漏OFF。0f6f02e/72da8fe因此仍有OFF限定错误，不能将初始结果当作这两个提交的完整验证。修复与最终验收见[BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md)。
- prefix组最终完整Host/Device wheel构建、工程内隔离venv安装、native profile/layout和源码SHA核验通过。未更改共享Python。
- 计时器微基准仅运行授权设备0，10次启动前均检查无既有NPU任务；结果及解释见TIMER_OVERHEAD_PROBE.md。
- 未运行完整EP16通信，也未在目标拓扑测量一键矩阵；不能将编译/CPU/micro通过写成目标通信正确性或已恢复性能基线。

验证材料保留于27的`/home/tyf/Q4_remote_test/DeepEP-LL-g4-small-b-prefix/.tmp/timer-profile-validation/`：
`.logs/pytest-suite-04.log`、`.logs/final-compile-*/timing.json`、`.logs/wheel-prefix-verified.log`、
`.tmp/wheels-prefix-verified/`、`.logs/timer-overhead-02/`。这些是验证结果位置，不是脚本硬编码依赖。

## 历史schema14限定修复

若遇到`ProfileEnabled: call to __host__ function from __aicore__ function`，更新本分支的限定修复后沿用原诊断命令，改用新的`--run-name small_b_phases_002`。
原失败目录保留；不需要清空构建缓存或重建SDK。构建系统会按头文件依赖重新编译受影响对象。
修复只统一打点helper的Host/AICore限定，不修改schema14、通信协议、收发顺序或压缩格式。

## 历史摘要与发送额度验证

完整相关CPU回归534通过、48项因本机缺少C++/CANN/POSIX环境跳过；最终摘要/选择/入口专项68通过。
35case/2800sample合成压力报告为47500字节（含CRLF），只有一个文件，所有case核心表均保留。
这是输出预算测试，不是对目标机这次测量的性能结论；拿到新的SEND_FIRST后再分析实际瓶颈。

2026-10-23累计发送额度更新：最终CPU回归597通过、48项因C++/CANN/POSIX环境缺失跳过。
加入10次总额说明后的35case/2800sample合成摘要为47662字节（含CRLF），仍只需一份文件。
已覆盖摘要/补充共同计费、跨输出目录重复内容、旧摘要迁移、额度耗尽及写入失败。
上述累计额度版本仅更新离线工具，无需重编wheel或重新采样；本页当前schema15升级则必须全rank重编并重新采样。
