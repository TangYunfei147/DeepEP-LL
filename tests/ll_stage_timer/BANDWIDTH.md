# EP16 已计数通信量与有效带宽（阶段 schema 3）

本功能随 `--ll-stage-timer` 自动采集，没有额外启用参数。核心通信 ABI 不因此改变；阶段 schema 从2升为3，专家 schema保持1。全 rank 更新同一提交，并重新安装匹配的 timer ON wheel后重新采样。此次计数功能不改变 SDK，不需要为它重建 SHMEM；原有 SDK 必须已通过当前身份校验。

```bash
# 在已更新的仓库根目录、已激活的目标环境执行。
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
# 沿用原 EP16 测试命令中的 --ll-stage-timer，无需额外 flag。
python tests/ll_stage_timer/analyze.py results/ll-timer --latest
```

旧 schema 1/2 文件仍可分析阶段耗时，但没有通信计数，不能补算带宽。schema 3仍是每核2048B、512B对齐，阶段首尾/累计记录不变，新增独立计数区域，不提供逐包时间线。

## 五类计数

每类记录 `bytes`、`operations`、`read_bytes`、`read_operations`。后两项是前两项中的读取子集，**不能再次相加**。operations是已插桩数据搬运请求的次数，不是NIC报文数、路由数或token数的通用替代。

| 类别 | 含义 |
|---|---|
| `local` | 本rank对本卡的已计数数据读写 |
| `intrahost_first` | 同物理机、不同rank的首发数据搬运，含partial成员读取 |
| `intrahost_forward` | 同物理机、不同rank的转发或归约结果回传 |
| `interhost_first` | 跨物理机首发数据搬运 |
| `interhost_forward` | 跨物理机转发或归约结果回传 |

物理机界仍按16个连续rank区分，与逻辑G不同。EP16应只有前三类有数据；分析器拒绝EP16出现非零跨机计数。后两类保留原始计数定义，但当前只输出EP16有效带宽。

## 具体计入哪些字节

| 操作 | 数据动作 | 类别与字节 |
|---|---|---|
| dispatch | packet首次发送，包括DIRECT/FIRST_HOP | 按源/目标rank分类，`tokenStride`，包含padding和flags |
| dispatch | 接收侧relay | forward类别，`tokenStride`，包含padding和flags |
| combine | DIRECT结果返回 | first类别，`returnStride` |
| combine | partial从memberRank读取专家贡献 | first类别中的read子集，`2H`字节；member0本卡贡献读取归local read |
| combine | partial归约后结果回传 | forward类别，`returnStride`；这里是新归约结果，不能解释成原包原样relay |

所有计数只在搬运发起点记录一次：combine gather读取归请求rank，不在对端再记一次。报表因此是每rank发起的读与写移交的有效吞吐，不是纯TX发送带宽；不要再将该读取与对端对应的数据供给重复相加。read字段始终是total子集。

本卡目标归local，不进入跨卡带宽分子。不计prepare本地staging、接收端消费同一数据的重复搬运；不单独统计count/ready等控制通信及路由表读写。packet内部的header、路由尾部和flags已包含在完整包字节中。因此这是**明确选定数据路径的搬运字节**，不是全部HBM访问量、全部链路字节或用户有效payload字节。

例如 `H=3584、top-k=8、FP8 dispatch` 时，单个dispatch packet计4096B；combine返回包计7680B；一份16位partial贡献读取计7168B。量化改变dispatch包大小，不表示combine结果也变成FP8。包大小含布局开销，不能简单用token数×H替代实际计数。

## 带宽分母与统计

每个rank、每轮、每个操作分别计算：

```text
duration_ticks = max_core(kernel.last_tick - kernel.first_tick)
duration_us = duration_ticks / 1000  # 当前已确认1GHz
intrahost_total_bytes = intrahost_first.bytes + intrahost_forward.bytes
effective_GBps = intrahost_total_bytes / (duration_us * 1000)
```

分母标为 `duration_kind=max_core_kernel`，即最长单核kernel跨度的时间代理；它不是所有核最早开始到最晚结束的包络，不是纯通信阶段时间或NIC服务时间。也不称作线速、峰值达成率。本卡字节不进入分子。两个intrahost类别各自也按相同分母报告速率。

先逐轮相除，再统计带宽均值和分位数；不使用“平均字节/平均时间”代替逐轮统计。各rank独立，不跨rank相减时间戳；各rank速率之和也不能当作整机真实链路带宽。observe保留原有重叠，drain插桩会改变分母；二者分开比较。正式性能基线仍用timer OFF。

## 输出及不可计算情况

`bandwidth.csv` 每rank/iteration/op/category一行；除五类原始类别外，另有派生的 `intrahost_total`。输出上述四类计数及 `duration_ticks`、`duration_us`、`frequency_hz`、`flags`、`status`、`effective_GBps`。

`summary.json` 的 `bandwidth` 保存逐轮数据，`per_rank_bandwidth` 按op/rank/category汇总计数、时间及速率。`stages.csv`等原有阶段输出保持原用途。

| 情况 | status / 处理 |
|---|---|
| schema1/2 | `counters_unavailable`，计数和速率为空，不伪造零 |
| `DEEPEP_LL_NO_COMM`编译试验 | `nonstandard_skip_communication`，保留诊断计数，拒绝生成速率 |
| 非EP16 | `unsupported_ep`，当前不输出有效带宽 |
| 频率缺失 | `frequency_unavailable`，保留ticks，不生成带宽或依赖频率的trace |
| local | `local_not_link_bandwidth`，保留本卡计数，无链路速率 |
| EP16跨机类别 | `not_applicable_ep16`；零计数不报告跨机速率，非零计数拒绝汇总 |

满足条件的intrahost类别若实际计数为零、时间有效，可报告0 GB/s；这与旧schema缺失计数不同。频率只能使用已确认值，不能为得到带宽随意填写。

本次实现的本地CPU检查不等于CANN编译或设备验证。目标机须全rank重建并验证正确性后采集；不要依据尚未验收的计数宣称带宽收益。

## 本地验证记录（2026-10-16）

CPU：114 passed，3 skipped（无g++/clang++，未执行C++ host模型），1 deselected（既有Bash环境迁移检查）。执行：

```bash
python -m pytest tests/ll_stage_timer -q -k 'not moved_checkout_with_spaces_and_other_cwd'
```

覆盖schema1/2兼容、schema3布局/保留位、read子集、计数缺失与真实零、禁用通信标记、非EP16/无频率、五轮逐轮除法再统计、dispatch/combine在含空格新目录及其他cwd下的CLI分析、旧wheel采集入口拒绝。C++源契约核对计数位置及无新增barrier；没有CANN编译或设备执行。三个其他任务kernel修改保留，本任务只暂存新增计数块。
