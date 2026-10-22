# Dispatch通信邻域与首包准备打点（schema13）

当前分支kernel/ll-g4-local-q-table，算法基线fb9263d（VF q播种+R0 VF前缀）。
本次优化诊断覆盖和离线报告，用于后续降低收发开销；没有修改通信算法。

## 参考来源与采用方式

参考SimpleBrightMan/DeepEP，分支megamoe-urma-tailCover，固定提交
34d02e97168f799f9558d8999a22e258edc1b3be：

- https://github.com/SimpleBrightMan/DeepEP/blob/34d02e97168f799f9558d8999a22e258edc1b3be/experiments/megamoe/csrc/kernels/mega_moe/urma/kernel.hpp
- 同目录timer.hpp、tail_completion.hpp、abi.hpp。
- experiments/megamoe/python/deep_ep_experimental/megamoe/urma_timer.py。

参考将DispatchPutPayload/PutScale、Quiet、Publish及接收完成/尾部等待分开；
TraceMeta记录peer/cv/rows/bytes，明确提交区间不是DMA硬件持续时间，
本端出站quiet不能替代对端入站完成。统计只能在同rank内使用时钟。

当前LL是MTE协议，没有URMA Put/Quiet，不能生搬参考函数名或完成语义。
本次对应拆分既有flag搬运、lane复用、payload就绪、relay和输出的附近开销。
保留本地累计、每类前两次真实区间、kernel结束后SyncAll再Flush的既有方式；
没有照搬参考逐事件写GM/flush，以免在线GM诊断写入干扰小包路径。
没有新增drain、barrier、通信通知或等待。既有relay-complete诊断仍会新增排空，
其数据在新CSV明确标记boundary_mode，不能当observe基线。

## 新格式与同步更新

schema13，108项metrics，192条采样事件容量，每核80×8 int64=5120B，512B对齐。
56核一次约280KiB，较schema12多28KiB。计时器本地累计数组也增加，ON可能更慢。
通信协议标识仍19000+S，wire、q算法、收发顺序不变；timer ON必须全rank重装完整wheel。
ST要求schema13，Python布局、native头、analyze/compare/安装检查和测试一起更新。
本次交付只针对新采样；旧数据不用于新报表或跨schema阶段耗时比。请重新采集。

KERNEL、SEND_SUBMIT_END、BEFORE_FIRST_SEND、BEFORE_FIRST_RECV_POLL均预留区间槽。
其他高频阶段只保存前两次实际区间，累计ticks/calls覆盖所有调用；
omitted_intervals非零表示采样容量/两次限额，不表示metrics漏掉后续调用。
不能据两次区间伪造逐包时间线、最后完成时间或全rank同步时间轴。

## “发送前准备”有两个不同指标

1. SEND_SETUP：Sender::Process/ProcessGroupLocal初始化阶段，至发送循环前。
   包含UB设置、输入路由/权重加载及就绪、q表构建、lane标记初始化。
   不包含此前counts.SendCnt或R0前缀，也不包含循环内分类、首包输入和打包。
2. BEFORE_FIRST_SEND：从本核KERNEL起点到第一次实际payload DataCopy下发之前。
   覆盖上述初始化、该核此前的count/prefix依赖、首包分类/输入/metadata/pack等待。
   直接测量真实区间，不由多个阶段之和或不同核时间相减获得。
   不含KERNEL外INIT；INIT在report单列。无发送核此项为空，不补0。
   report另有entry_to_first_send，用同核INIT真实起点和首发真实终点直接计算完整区间，
   包含初始化以及INIT与KERNEL之间的间隔；两端任一缺失就不生成，不拼接平均值。

Q_SEED_SUBMIT只记录VF调用下发；Q_SEED_WAIT记录原V_S调用区间；
Q_LOCAL_UPDATE为本片Scalar更新。它们是SEND_SETUP/PREPARE_INPUT的子区间。
PREPARE_INPUT仍可在循环中累计，不可直接叫“首包前准备总时间”。
SEND_SETUP与BEFORE_FIRST_SEND有交集，不可相加。算法稳定采样中的准备时间也含ON探针扰动，
真实未插桩总耗时仍以OFF profiler为准。

BEFORE_FIRST_RECV_POLL同理，是本核KERNEL起点至首次Probe调用前，
用于看发送/前缀/建表对接收启动的延迟；它不是首次到包或首次接收完成时间。

## 新增覆盖与含义

| 路径 | 新阶段 | 解释 |
|---|---|---|
| 初始化 | SEND_SETUP | 发送器初始化整体，S/R共用含义 |
| q | Q_SEED_SUBMIT / Q_SEED_WAIT / Q_LOCAL_UPDATE | VF下发 / 原依赖等待 / 本片Scalar |
| 路由 | ROUTE_CLASSIFY / DIRECT_PLAN | 去重分类 / DIRECT描述符准备 |
| 发送 | TX_PACK_READY_WAIT / TX_METADATA_PUBLISH | 原V_MTE3 / 源端mask和weights写出调用 |
| 尾部 | SEND_RETIRE_WAIT | 原异步发送lane退休等待，包含在FINAL_DRAIN |
| prefix | PREFIX_PUBLISH_WAIT | 原前缀MTE3完成依赖 |
| 探测 | RX_FLAG_ISSUE / RX_FLAG_WAIT / RX_FLAG_REDUCE / RX_FLAG_DECODE | 标记地址和搬运下发 / 原MTE2_V / 归约及原V_S / 解码 |
| 接收lane | RX_LANE_REUSE_WAIT / RX_PAYLOAD_READY_WAIT | 原lane复用 / MTE2_S与MTE2_MTE3依赖调用 |
| 消费 | RX_METADATA / RX_PAYLOAD_OUTPUT / RX_META_OUTPUT | 包尾处理 / payload与scale写出 / type、destination和relay索引写出 |
| 队列 | RX_PENDING_UPDATE | 原pending swap-remove |
| 计数 | TX_PACKETS / RX_PROBED / RX_READY / RX_EMPTY_POLLS / RX_PACKETS | 初发包 / 探测项（含重复）/ 就绪项 / 空快照 / 消费包 |

原TOKEN_SEND_ISSUE、RELAY_ISSUE、RECV_INPUT、RECV_OUTPUT、CLEAR_LOCAL_FLAGS、
RECV_COMPLETE_WAIT等保留。原父阶段仍可能包含这些子阶段。
RX_PAYLOAD_READY_WAIT可在一个包中调用两次，因此per_call不是“每包耗时”。
TX_PACKETS不包括relay，relay仍由RELAY_PACKETS统计；不将逻辑组叫物理跨机。
本次不提供声称纯网络带宽的数值，也不把MTE依赖指令耗时认定为独占DMA时间。
默认observe没有引入额外同步；一个WaitFlag的Scalar区间并非总能独立代表整个硬件排空时延。

本次新增细分针对dispatch。combine继续使用既有COMBINE_*阶段并同步新carrier/schema，
没有把MegaMoe融合GMM/URMA的阶段强行映射进LL combine。

## analyze的新输出

在现有summary.json/stages.csv/metrics.csv/intervals.csv/trace.json之外自动生成：

- communication_bottleneck.md：口径说明、逐核准备p50、长时核上的优先排查项。
- communication_samples.csv：每rank/轮/core的原始ticks、us、首包状态和计数。
- communication_costs.csv：各子阶段累计值、调用次数、每次平均、同核kernel占比。
- communication_per_core.csv：同rank/core跨轮p50和有效样本数，缺失不补零。
- summary.json的communication字段包含相同数据。

ready_fraction=RX_READY/RX_PROBED；probes_per_consumed_packet=RX_PROBED/RX_PACKETS。
低ready比例可来自上游生产迟、轮询过密或relay依赖，不能直接认定网络拥塞。
长时核筛选为同rank/轮kernel跨度达到该rank最大值90%，只是候选定位。
成本排序可能同时列出SEND_SETUP与它的q子阶段，不求和、不以差值推造“未覆盖开销”。
无频率只给ticks；默认1GHz来自已确认的目标约定，跨rank时间戳不相减。
新报表要求有效完成采样和一致计数，拒绝缺首包区间、RX_READY与消费不一致等输入。

## 获取、构建和采集

在当前工程根、加载目标CANN和已有工程venv后：

    git fetch origin kernel/ll-g4-local-q-table
    git merge --ff-only FETCH_HEAD
    source scripts/ll_timer_env.sh
    export DEEPEP_NPU_ARCH=Ascend950
    export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
    export DEEPEP_LL_STAGE_TIMER=ON
    python -m pip install -v --force-reinstall --no-build-isolation --no-deps .

已校验SDK无需仅因打点强制重建。全部rank同提交、同S/R、同ON wheel。
公共run.py的branch.json已纠正为当前g4-r0-prefix，避免旧g4-overlap-ab身份误拒绝。

在用户已分配的目标机上使用实际设备列表，例如原EP16测试：

    python tests/ll_stage_timer/run.py --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      --num-nodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29500 --shmem-port 29501 \
      --num-tokens 256 --num-max-tokens-per-rank 256 --hidden 3584 --num-topk 8 \
      --num-experts 384 --dtype fp8 --do-cpu-sync 0 --warmup 10 --num-iters 100 \
      --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe --skip-perf-test \
      --output-dir results/communication --run-name comm-schema13-001

先加--plan核对CPU参数。--skip-perf-test仍有原稳定调用；打点与正确性在计时协议下独立采样。
多机沿用实际NNODES/NODE_RANK/MASTER_ADDR和设备配置，所有rank输出齐全后统一分析。
每次运行用新run-name；若只采dispatch用--dispatch-only，验收仍建议另做combine。

    python tests/ll_stage_timer/analyze.py results/communication --latest \
      --op dispatch --frequency-hz 1000000000 --trace-iteration 0

分析入口按case独立输出，不混不同shape/seed。已有报告目录拒绝覆盖，重分析可指定新的--output-dir。
不用旧schema12结果判断这次细分。OFF性能测试单独重装OFF wheel并沿用原10/100/尾50，
不能将ON报告中阶段下降当作OFF加速。

## 如何据此推进优化

- BEFORE_FIRST_SEND高且SEND_SETUP高：看Q_SEED_WAIT/Q_LOCAL_UPDATE、初始化输入。
- BEFORE_FIRST_SEND高而SEND_SETUP较小：看count/prefix或首包分类/输入/打包；
  结合同核第一段真实interval，不相减独立p50。
- RX_FLAG_WAIT/REDUCE高：考虑批量flag读取/归约；RX_FLAG_DECODE或PENDING_UPDATE高：
  考虑向量解码/记录段任务表示。
- TX_PACK_READY_WAIT/TOKEN_BUFFER_WAIT高：查pack流水与lane复用；不能直接说链路饱和。
- RX_PAYLOAD_OUTPUT/RX_META_OUTPUT高：查连续输出合并和小块metadata DMA。
- SEND_RETIRE_WAIT/RECV_COMPLETE_WAIT高：观察尾部在途工作，不在observe中增加排空来“证明”。
- RELAY_ISSUE与长时核相关：检查首跳成员数与接收归属，再考虑负载调整。

## 验证边界

源码归一化token检查证明send/recv/counts业务语句及同步调用序列与fb9263d相同
（忽略新增诊断宏和作用域），是补充静态证据，不是AscendC编译证明。
原生CPU计时器mock验证ON/OFF行为、延迟GM导出、容量和首包/首轮询区间预留；
合成schema13样本经过实际parse→analyze→CSV/Markdown/Chrome输出并校验单位。
未做真实CANN ON/OFF冷编译、完整ON wheel链接或NPU采样；新增指标的实际数值要重新采集。

2026-10-22元数据候选更新：上述fb9263d等价检查仅描述原始插桩提交。
当前收发算法优化已改变业务语句，communication_baseline.json记录当前候选摘要，
不得把当前摘要检查解释为与fb9263d算法逐字等价；schema13及observe同步口径保持。
