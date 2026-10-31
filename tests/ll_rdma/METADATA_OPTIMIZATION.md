# M=4 元数据优化逐提交实验

仅 kernel/ll-ndr-metadata-cores；M 固定4，不调 S/R 数量。SDK、wire、QP、
计数数学、零计数发布、前缀 expert/source 顺序及输出行顺序保持。
不把 reference ON 内部阶段缩短当作 OFF 完整 kernel 收益。

## 第一步：复用已有跨机硬件直方图

已有 `CountRankHistogram` 将跨机每专家一次完整路由扫描及 V_S 改为每目标一次
归一化/硬件直方图及 V_S；最终32B记录的Scalar填写仍保留。本机计数不变。
本提交不重复引入算法，也不更改普通构建默认值；新增实验入口显式选择方法：

```bash
python3 tests/ll_rdma/metadata_opt.py scalar "$NODE_RANK" "$MASTER_ADDR" meta-scalar-001 --mode perf --tokens 1 8 32 256 --dtype fp8
python3 tests/ll_rdma/metadata_opt.py histogram "$NODE_RANK" "$MASTER_ADDR" meta-hist-001 --mode perf --tokens 1 8 32 256 --dtype fp8
```

所有节点同序运行、使用同一个新run名。入口强制M=4，显式覆盖旧环境的histogram值，
禁止skip-build；实际构建、SDK校验、完整wheel安装、全节点源码/SDK/配置一致性检查、
正确性、重复combine、10/100/尾50和initial-reduce-exp100-batch-v3均复用one_click。
perf为timer OFF的dispatch+BF16 combine；FP8表示dispatch配置。
诊断单独用新run名和 `--mode reference --reference-iters 1`，不与perf混为一轮。
`--plan`只做CPU计划，既不拉依赖也不加载设备。仅准备SHMEM，不拉CATLASS。
首次升级到锁定SDK 4c671684时加 `--rebuild-shmem`；之后同工程同SDK正常复用。
命令以子进程返回失败，不退出当前交互shell；任何失败停止该轮，不自动重试性能。

完整Excel/raw留目标机；reference沿用one_click的return_report.md，perf多版本使用文末
metadata_report.py合为一个UTF-8不超过60000字节的文件。
硬件直方图仅CPU语义检查通过不代表目标CANN编译或NPU性能验证。

本提交入口/身份/配置/真实直方图CPU语义检查共302项通过；CPU --plan通过。
Windows沙箱曾阻止Zig临时文件重命名，改用工作区临时目录并在沙箱外运行相同
本地CPU检查后通过；未使用CANN、NPU或SSH。

## 第二步：任务表按64个索引合批

复用原256B taskBuildBuf，不新增UB、lane或event。短count记录跨记录合批，
用Scalar填入剩余槽；缓冲为空且当前记录剩余至少64条时仍用CreateVecIndex生成整块。
每核T条输出的任务表DMA从逐记录的 `sum(ceil(count_i/64))` 次变为 `ceil(T/64)` 次，
包括跨记录边界和最后不足64条的尾批；索引、顺序、GM区间和总字节数完全一致。
Scalar或Vector复用前分别用MTE3_S/MTE3_V退休同一源缓冲，提交分别用S_MTE3/V_MTE3。
最后一次任务表DMA仍在原前缀ready发布之前；不能提前发布未完成的表。

这是减少小DMA/同步的候选：短记录新增Scalar填索引成本，长记录保留Vector主体；
不预先宣称净收益。与第一步相同histogram值对照，M固定4。
真实生产count/前缀/任务表、接收器及direct-count共11项CPU检查通过；
其中M=2/4/18各运行90个count场景，覆盖零计数、热点、B=65/256、跨E复用、
输出顺序/尾部哨兵、精确DMA批次数及在途源不被覆盖。接收ON/OFF与16/72核也检查。
这不是CANN编译或设备验证。

## 第三步：整数向量加跨核offset

`CalRecvAndSetFlag` 用一次原地int32 Adds替代逐条Scalar读/加/写，offset为0时
完全跳过该循环。S_V获取Scalar产生的局部前缀，V_S保证随后的任务表构建读取完成结果；
沿用原S_MTE3发布依赖。局部前缀的Scalar累加不变，不引入浮点转换或额外scan缓冲。
本候选增加一次S_V/V_S的固定成本，极短分片可能无收益，必须同M=4完整OFF对照。

最终相关CPU回归408项通过，包含真实生产count/任务表/receiver、count帧/接收/CQ退休、
直方图、入口/身份/reference解析，以及36组原地整数offset的尾部哨兵与延迟Vector模型。
覆盖超过2^24仍精确的整数值、offset=0、非整块长度。未做CANN编译、NPU或SSH。

## 固定版本四组对照（只在新独立clone中执行）

| 组 | 固定源码 | count方法 | 目的 |
|---|---|---|---|
| base | 712f4d33ce6dd5f7c2bb4d314bf7177945165f91 | scalar | 与5b38bf2相同kernel，只有实验入口 |
| hist | 712f4d33ce6dd5f7c2bb4d314bf7177945165f91 | histogram | 只开启已有跨机直方图 |
| batch | 4aa0a108149f06fe6bcce7b2e4c161ef8a9ba229 | histogram | 在hist上增加任务表合批 |
| prefix | cdf2d877b0646dd99888dac9788583c8a3012306 | histogram | 在batch上增加offset向量加 |

每个目标节点执行以下流程；NODE_RANK分别设0、1，MASTER_ADDR均为节点0地址。
不要在现有生产或有未提交改动的工程中执行切换。新clone中复用同一份第一次新编的SDK；
各版本完整重建wheel，按顺序完成后再切下一版。只初始化third-party/shmem。

```bash
git clone --single-branch --branch kernel/ll-ndr-metadata-cores \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-meta-opt-20261031 &&
cd DeepEP-LL-meta-opt-20261031
```

确认进入这个新目录后执行；重复实验时更换META_TAG，保留所有旧结果。

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.1
export META_SEED=0
export META_TAG=meta-opt-s0-a1

meta_case() {
    local rev="$1" method="$2" name="$3"
    shift 3
    git diff --quiet && git diff --cached --quiet &&
    git switch -C kernel/ll-ndr-metadata-cores "$rev" &&
    python3 tests/ll_rdma/metadata_opt.py "$method" "$NODE_RANK" "$MASTER_ADDR" "$name" \
      --num-nodes 2 --mode perf --tokens 1 8 32 256 --capacity 256 \
      --hidden 3584 --topk 8 --experts 384 --dtype fp8 --route normal --seed "$META_SEED" "$@"
}

if meta_case 712f4d33ce6dd5f7c2bb4d314bf7177945165f91 scalar "${META_TAG}-base" --rebuild-shmem &&
   meta_case 712f4d33ce6dd5f7c2bb4d314bf7177945165f91 histogram "${META_TAG}-hist" &&
   meta_case 4aa0a108149f06fe6bcce7b2e4c161ef8a9ba229 histogram "${META_TAG}-batch" &&
   meta_case cdf2d877b0646dd99888dac9788583c8a3012306 histogram "${META_TAG}-prefix"; then
    echo "PASS: four complete OFF runs; retain raw data"
else
    echo "FAILED: stop here and retain this terminal, results and .logs"
fi
```

第一轮只用于筛选，不足以宣称稳定收益。后续相同四组反序重跑、更换run名，再更换seed；
同工程SDK已构建且无变化时不再加rebuild-shmem。不要跨失败attempt拼接或自动重试。
均值为每rank尾50样本汇池均值，最慢rank均值单独列出；不能把两者混为一个指标。

四组全部成功后，节点0恢复最新分支上的只读汇总工具，导出一个回传文件：

```bash
git switch -C kernel/ll-ndr-metadata-cores origin/kernel/ll-ndr-metadata-cores &&
mkdir -p results/meta-opt-report &&
python3 tests/ll_rdma/metadata_report.py \
  "${META_TAG}-base-perf" "${META_TAG}-hist-perf" "${META_TAG}-batch-perf" "${META_TAG}-prefix-perf" \
  --output "results/meta-opt-report/${META_TAG}-return_report.md"
```

工具只读summary与已验证的节点0安装身份，不import框架/设备；核对完成状态、M=4、
OFF、SDK、shape、路由hash、seed、采样与实际核数一致后列出各版dispatch/combine统计。
不自动筛除慢样本，不自动计算加速结论；原始样本及各rank细节保留目标机。
输出使用独占创建且按实际UTF-8字节限制60000，超限失败、不静默截断。
汇总/实验入口另外8项CPU检查通过（其中3项与前述408项重叠）。

需要解释阶段差异时，在相应固定源码上单独用新run名执行：

```bash
python3 tests/ll_rdma/metadata_opt.py histogram "$NODE_RANK" "$MASTER_ADDR" "${META_TAG}-prefix-detail" \
  --num-nodes 2 --mode reference --tokens 32 --dtype fp8 --seed "$META_SEED" --reference-iters 1
```

reference只采dispatch，完整xlsx留目标机，其内部时长不与OFF直接相减。
