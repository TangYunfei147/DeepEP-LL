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

完整Excel/raw留目标机；沿用one_click单个UTF-8不超过60000字节的return_report.md。
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
