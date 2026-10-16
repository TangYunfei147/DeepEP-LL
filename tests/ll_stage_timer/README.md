# LL阶段与专家诊断（泛化分支）

维护分支kernel/ll-configurable-groups。实验、采样、解析与NDR分支共用同一套代码，
推荐直接按[PARITY.md](PARITY.md)启动和对比。

本分支保留MTE传输、原始静态分组默认G=16；不包含RDMA data plane，也不采用NDR的G4/16 tiling策略。
G=16与G=4可以直接作为比较变量，工具记录实际G而不要求相同。

## 格式与构建

核心启动ABI5、阶段schema2、专家schema1；全rank重建同一分支的匹配wheel。
每核阶段缓冲2048B、512B对齐；专家统计有独立可选缓冲。
设置DEEPEP_LL_STAGE_TIMER=ON启用；仅阶段schema1旧数据也可离线解析。
泛化分支默认MTE构建；python scripts/build_shmem.py自动校验并复用已有SDK，包括身份匹配的HNS能力超集。
无需重复--if-needed；不匹配时保留显式--rebuild保护。详见[CONFIGURATION_NOTES.md](CONFIGURATION_NOTES.md)。

## 采样

有perf时先10次预热、100次Exp+scalar all_reduce性能采集，统计最后50条，
关闭profiler后再追加ll-timer-iters次内部打点；无perf时先同样的预热和稳定调用。
默认5轮打点，最后一轮编号4；全部完成后才复制/校验/导出。
默认Exp约13.53GiB/卡，保留多轮输出会增加显存。

observe不新增流水屏障；drain在阶段边界排空，改变重叠与等待成本。
目标计数器按cycles/1000 us换算，原始ticks保留。父子/跨核累计不可当作端到端时延相加。

## 子阶段

recv_consume细分recv_lane_wait、recv_gm_to_ub、recv_load_wait、recv_metadata、
recv_output_wait、recv_output_write、recv_forward。
send_pack细分send_input_gm_to_ub、send_input_wait、send_tail、send_lane_wait；
send_issue下的send_mte_write记录MTE写入下发。
共享schema还保留send_rdma_*名称，但本分支不执行这些阶段，缺失数据不补0。

## 专家统计

增加--ll-expert-timer（自动启用阶段capture），记录执行rank/core/专家的packet计数、
metadata、output_write及按目标专家记录的FIRST_HOP forward。
共享首跳准备、relayReadIndex和批量等待不分摊给各专家；不表示FFN/GEMM耗时。
专家缓冲每轮字节数为实际AIV×(8+全局专家数×4)×8×8。

## 分析

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001 --latest --trace-iteration 4
```

也可在测试时--run-name固定目录。输出summary.json、stages.csv、trace.json；
启用专家统计时增加experts.csv、expert_cores.csv。
默认每AIV一轨道，各rank/调用独立归零；没有跨rank时钟对齐。
六类重复发送包络默认隐藏，累计统计保留，--show-repeated可恢复。
新公共compare.py会记录不同G，不强制分组一致；步骤见PARITY.md。

本次同步只完成本地CPU回归，尚未做CANN编译或目标设备验收。
