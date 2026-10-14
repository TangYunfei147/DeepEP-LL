# 发送量、重复采样和带宽统计

## 控制每条边的数据量

`comm` 支持两种写法，形状模式计算的是每条有向通信边的payload：

```text
bytes_per_edge = bs × hidden_size × element_bytes
```

- `--bs 32 --hidden-size 7168 --element-bytes 2`：每条边458752字节；element-bytes默认2，可选1/2/4/8。
- `--bytes 458752`：直接指定字节数；不指定形状也不指定bytes时保留旧默认15392。
- 若两种方式同时提供，数值必须一致；要求32B对齐，不隐式补齐。
- kernel仍按uint8校验模式搬运字节。2字节可对应BF16/FP16大小，但不在通信核里做浮点计算。
- 这是每条边复制一份B×H数据，不按MoE top-k自动分摊。完整原生LL仍用`--num-tokens`和`--hidden`，其实际线上格式不能用本公式替代。

EP32、fanout3示例：每轮只有0→1/2/3三条边，总payload=1376256字节。两机各16连续rank时这三条都是同机边。
接收缓冲仍按world×bytes预算，EP32每边上限8MiB，其他域也受256MiB接收预算和64MiB单边上限约束。

## 重复测量

```bash
# 两台机器各执行一次，另一台node-index=1；先改ep32.yaml中的IP
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0 --test comm --run-id shape32-001 --target-run -- --scenario 3 --fanout 3 --bs 32 --hidden-size 7168 --element-bytes 2 --samples 100 --tail 50
```

默认仍是预热10、采集100、取最后50条，每次Exp+scalar all_reduce，按kernel/task独立统计。
需要更多样本时可显式用`--samples 1000 --tail 1000`，或`--tail 0`使用全部采集样本。
P99基于所选样本线性插值；50条的P99不是足够多独立尾部事件的保证。A5 validation仍仅允许2/3/2和64×64小Exp。

## 硬件参考配置

`configs/ep*.yaml`默认引用：

```yaml
hardware_config: tests/ll_probe/configs/hardware/ascend950.yaml
```

原始规格与出处现已分离到同目录`ascend950.reference.yaml`；此参考文件不会被程序加载。
`ascend950.yaml`仅保留格式/显示元信息和实际启用的model容量，目前仅tx/rx两项。
参考文件中保存用户指定[昇腾950架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)的URL、SHA256、页码和原始IO参数。
PDF第15/30/31页中UB 2016GB/s为整芯片双向，UBoE 200GB/s和PCIe 128GB/s也为双向；共享端口不能相加。
默认model用2016/2=1008GB/s作为每rank一颗完整芯片的理想单向收发上限。
这是计算假设，**不是实际单对卡链路带宽或目标机器实测值**；不同设备划分/可用端口时请调整。

所有model字段都是十进制、单向GB/s：

| 配置项 | 含义 |
|---|---|
| per_rank_tx_GBps / per_rank_rx_GBps | 每个rank总发送/总接收容量，默认1008 |
| intra_host_pair_GBps | 同机单条src→dst边容量，默认null |
| inter_host_pair_GBps | 跨机单条src→dst边容量，默认null |
| per_host_cross_tx_GBps / per_host_cross_rx_GBps | 每台主机跨机共享总出口/入口容量，默认null |
| aggregate_GBps | 整个通信作业共享fabric容量，默认null |

未配置或null表示未知而未纳入该约束。未启用字段的模板集中在`ascend950.reference.yaml`的`optional_model_fields`；要启用，须复制到运行配置的`model`并填写正数。白皮书没有给出用户实际光模块、布线和竞争条件，因此不猜测跨机值。
确认实际链路后直接改此文件，或复制一个新文件并用`--hardware-config`覆盖。报告嵌入完整配置快照与hash，
不同rank使用不同配置会被汇总拒绝。模型不使用HBM峰值推断网络峰值，不从核心数猜测PR/DT型号。

## 理论耗时与利用率口径

对实际通信图分别累计每rank发送/接收字节、同机/跨机字节、每host跨机字节。
理论时间取所有已配置资源约束的最大值：

```text
T_theory_us = max(资源负载bytes / (对应单向GB/s × 1000))
BW_effective_GBps = 每轮总payload_bytes / (实测时间us × 1000)
BW_reference_job_GBps = 每轮总payload_bytes / (T_theory_us × 1000)
参考带宽利用率% = T_theory_us / 实测时间us × 100
```

例如多个发送rank可并行时，不会把全作业数据量都除以一张芯片的峰值。
这是忽略启动、协议、串行化、缓存与拥塞的**理想容量下界**；未填写的链路限制仍未知。
出现>100%时保留数值并标记，不截断；应核对容量单位、拓扑和计时口径。

## 输出JSON、CSV与分位数

```bash
python3 tests/ll_probe/profiling/analyze_profile.py results/shape32-001/node-*/data/rank_*/report.json --output results/shape32-001/summary.json
```

默认同时生成`summary.csv`并打印表格；`--csv-output`可改CSV路径，输出文件必须尚不存在。
JSON保留每rank/task原始统计、逐轮rank-MAX，以及：

- 每条边、每rank和全作业单轮发送量；同机/跨机分类。
- 采集总次数、统计窗口次数，以及两种窗口累计payload量（不含预热、校验和Exp/all_reduce流量）。
- 延迟P50/P90/P99、均值、原始样本。
- 每次样本换算后的带宽分布及P50/P90/P99。
- 理论耗时、资源约束、参考利用率分布和各延迟分位点对应带宽/利用率。

CSV的P50/P90/P99行指**延迟分位点**。同一行的带宽由该延迟换算，不冒充带宽分布的同名分位数；
例如慢的P99延迟对应较低带宽。独立带宽分位数另存在JSON的`bandwidth_GBps`中。
正式口径是每轮各rank命名task时长的最大值，并非全局时间戳对齐的作业makespan；Event/API不混入。
若存在多个不同task类型，分别输出延迟，保留原有不混算一个带宽数值的规则。

旧报告也可重新按新模型分析，不需要重新采样：

```bash
python3 tests/ll_probe/profiling/analyze_profile.py results/old-run/node-*/data/rank_*/report.json --hardware-config tests/ll_probe/configs/hardware/ascend950.yaml --output results/old-run/reanalyzed.json
```

如果模型需要主机/边带宽约束而旧报告缺少部署拓扑，会拒绝计算；不会猜测节点分布。
原有`--peak-gbps --peak-description`仍是明确的整作业容量覆盖，采用GB/s口径，结果记录覆盖依据。

## 验证记录

2026-10-14：CPU回归134项通过，覆盖shape冲突/对齐、白皮书单位、fanout数据量、并行发送者、
跨机共享瓶颈、分位数换算、混硬件配置拒绝和CSV输出。A5仅0/1实际双rank、小Exp，
bs4×hidden256×2=2048字节的fanout1采样及全rank汇总通过。该短样本验证不代表目标机性能。
