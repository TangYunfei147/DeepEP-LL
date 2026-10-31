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
