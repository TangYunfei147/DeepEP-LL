# 纯 CPU 查看 dispatch 路由、分核数量与阶段顺序

当前group-compact分支CLI默认`--send-policy group-compact`，可用`--send-policy s3`
比较ac876ba的发送所有权。原始路由、q、FIRST_HOP成员及接收行相同，只重新分配S初发。
manifest记录策略；Python build_plan API默认仍为s3，调用时显式指定候选。
下方回放是逻辑因果模型，不是S3 DMA重叠/UB lane性能模拟。

入口 `tests/elastic/inspect_dispatch_routes.py`，依据
`kernel/ll-g4-relay-priority@550b423` 的源端分工、静态接收和 relay 优先逻辑。
只用 Python 标准库：不导入 Torch/torch_npu/deep_ep，不调用设备、建连、
启动多进程或分配 payload。Git 更新脚本后即可运行，不需要重装 wheel/SDK。

## 直接替换原 ST 入口

在工程根目录运行，原参数保持，额外传入希望分析的编译 S/R：

```bash
python tests/elastic/inspect_dispatch_routes.py \
  --num-processes 16 --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --shmem-port "$SHMEM_PORT" \
  --num-tokens 256 --num-max-tokens-per-rank 256 \
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 --do-cpu-sync 0 \
  --send-cores 28 --recv-cores 28 \
  --output-dir "$PWD/results/1020_ll-g4-shared-recv" --run-name "$RUN_NAME"
```

只需在一台 CPU 机器上运行一次，脚本统一输出全部 rank。
EP=`num-processes*num-nodes`，如NNODES=2就是EP32；node-rank仅保留身份信息，
不会把输出过滤为本机。不会读取torchrun的WORLD_SIZE/RANK来覆盖显式参数。
地址/端口、warmup/perf/打点参数仅作为元信息；ll-timer-iters=5不把计数乘5。
各项数量都是**一次 dispatch**，并非一次运行全部调用的累计量。

S/R必须为正且和56；只传一个就推导另一个，例如 `--send-cores 10` 得到10/46。
它们是模拟的源端分工，不读取已安装wheel。G默认读取当前源码宏的默认值，
显式编译覆盖时传 `--ranks-per-group`。全部56核仍参与静态接收，R不等于
dispatch实际接收参与核数。脚本沿用ST的CPU参数检查，不替代native布局/SDK准入验证。
去重开关依据M>32，不是实际B；例如B3/M256仍按去重路径统计。
当前Classify使用32位组位图，因此启用去重且逻辑组数超过32时拒绝回放，
避免把理想CPU算法当作该源码在未定义位移情况下的行为。

## 与实际 ST 的 top-k 对齐

直接复用同一文件中 `make_routes` 的Python随机生成器，不用Torch随机数。
**原ST默认correctness-repeats=3，初次检查seed、再检查seed+1和seed+2，
后续perf/打点沿用最后一次路由。** 因此新脚本默认 `--route-phase timed`，
在默认配置下输出seed+2；manifest同时保存base seed和effective_seed。

- `--route-phase initial`：查看初次正确性检查的seed路由。
- `--route-phase all`：为每个正确性换路由轮次分别输出一组报告，末轮标为timed。
- `--skip-check`时没有换路由，timed也使用原seed。

支持原ST的参数矩阵，每个case/seed独立保存，不能混合不同shape的计数。
本脚本不导出真实NPU top-k；如果原作业参数、seed、ST源码或换路由次数不同，
必须按对应配置重新生成。源码SHA和Git revision保存在manifest中。

## 输出文件

输出目录为：

```text
<output-dir>/<run-name>/cpu-dispatch-routes/
  manifest.json
  summary.md
  case_0000/
    topk_ids.json
    routes.csv
    routing_counts.csv
    expert_counts.csv
    rank_counts.csv
    core_counts.csv
    stage_counts.csv
    events.csv
```

已有同名CPU目录会拒绝覆盖；可以与原NPU结果目录共存。

| 文件 | 内容 |
|---|---|
| topk_ids.json | 完整 `[EP,B,K]` 专家ID、base/effective seed和路由阶段 |
| routes.csv | 每条路由的目标rank/组、q、接收行/核、包类型、首跳、初发核与实际线上的发送核 |
| routing_counts.csv | 每个来源rank→目标rank的逻辑路由数、初发包数和relay包数 |
| expert_counts.csv | 每专家接收数量，包括零命中专家 |
| core_counts.csv | 每卡每核初发、接收、转发packet数及各自unique token数；包含空闲核 |
| rank_counts.csv | 每卡汇总，unique按集合并集而非把各核unique直接相加 |
| stage_counts.csv | 每核每逻辑轮次的阶段数量，按首次core_sequence排列，带首尾序号 |
| events.csv | 逐事件的全局sequence、每核core_sequence、因果轮次及关联route_id |

packet数是投递次数，不是不同输入token数：同一 `(origin_rank,token)` 可发往
多个专家/组。`unique_tokens`才按这对标识去重。初发与转发分别统计，
同rank的MTE投递也计一次packet，另有same_rank/cross_rank字段。
`send_packets+forward_packets=receive_packets=EP*B*K` 是全域校验，
不是任意单核都成立。count通知、前缀广播、flag清理等控制通信不当作token。

## “按阶段顺序”的准确含义

数量和静态归属由路由唯一确定；真实到包时间无法在不运行设备的情况下知道。
`events.csv`给出的是**一条合法、确定性的因果回放**，不是设备时间线或性能预测：

1. 每逻辑轮每个初发核最多下发一包；本轮新包下轮才可见。
2. R核按本组专家区间初发，S核按原token/展平路由分片初发，保留错峰顺序。
3. 每卡R初发完成后，假定本卡前缀准备立即完成；每核完成自己的初发即可
   等前缀进入接收，不引入“所有S发送结束”的屏障。
4. 接收每轮处理一个最多16项的就绪快照，仅在该快照内FIRST_HOP优先，
   同类按原下标倒序。未就绪包不会因优先级被提前消费，也不是全表首跳优先。
5. RECV_INPUT按实际lane环的深度预取；消费FIRST_HOP时先逐包RELAY_ISSUE，
   后RECV_OUTPUT。同批pending按原下标倒序swap删除，保持下一批任务语义。

看某卡某核时，按 `core_sequence` 读取 `events.csv`；同一个route可以同时
出现在RECV_INPUT和RECV_OUTPUT中，不能把两个阶段相加当作接收token总数。
`stage_counts.csv`是聚合视图，重复阶段可能交错，其首尾序号不是连续执行区间。
本模型不模拟带宽、写队列背压、流水耗时、count实际到达、跨rank时钟或
诊断MTE3排空；不能用logical_round比较设备时延或判断硬件瓶颈。

## CPU回归

```bash
python -m pytest tests/elastic/test_inspect_dispatch_routes.py -q
```

检查覆盖计数守恒、q/接收行与ST一致、S/R比例、FIRST_HOP先转发后输出、
同卡relay、局部就绪优先、无全S发送屏障、多机域、参数矩阵、换seed阶段、
不覆盖已有结果，以及迁移到含空格路径/不同工作目录。
命令行测试在`python -S`下显式禁止导入torch/torch_npu/deep_ep/acl。
本次相关CPU回归124 passed、1 skipped；跳过项是原生产优先级函数的g++检查。
默认EP16/B256/K8/E384、seed=0、correctness-repeats=3对应effective seed=2，
模型统计为19158初发、13610转发、32768接收packet，不是NPU实测数据。
