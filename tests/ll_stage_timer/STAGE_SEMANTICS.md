# schema13：每个阶段名称只表示一种工作

2026-10-22，基线9e24a79，分支kernel/ll-g4-group-compact。
本次只修改打点标签、记录布局、解析和检查，不修改业务算法，不增加流水屏障。
核心通信标识19000+S不变；timer schema从12升到13，全rank重装匹配ON wheel后重新采集。
SDK无需因此重建。OFF采样可沿用同一算法基线。

## 首包8us的解释

用户测量从SEND_TOKENS外的第一个PREPARE_INPUT开始，到第一次TOKEN_SEND_ISSUE。
这是每核首包提交前的启动跨度，包括一次性准备、路由分类、q计算和输入/打包等待。
它不是每包固定准备成本，也不等于网络传输完成时间。
预取会把下一包输入/q与上一包消费交错，不能把所有阶段第一条记录拼成一包时间线。

schema13对应：

    SEND_SETUP（循环外、每核一次）
    SEND_TOKENS
      ROUTE_CLASSIFY（每token一次，可能生成多个packet）
      CAL_TOKEN / TOKEN_INPUT / TOKEN_INPUT_WAIT
      TOKEN_METADATA（首跳尾部可嵌套CAL_TOKEN）
      TOKEN_BUFFER_WAIT / TOKEN_PACK / TOKEN_SEND_ISSUE

没有新增包ID，因此阶段之间仍不能仅凭相邻位置建立逐包关联。

## 主要名称拆分

| 旧名/位置 | schema13名称 | 唯一含义 |
|---|---|---|
| PREPARE_INPUT：S的Process开头 | SEND_SETUP | UB/队列/路由权重搬入/lane初始准备 |
| PREPARE_INPUT：Classify | ROUTE_CLASSIFY | 当前token的G4路由分类 |
| PREPARE_INPUT：LocalSendScratchPool | LOCAL_SEND_POOL_SETUP | 建立本组发送scratch视图 |
| PREPARE_INPUT：ProcessGroupLocal开头 | LOCAL_SEND_SETUP | 本组发送输入/缓冲准备 |
| SEND_DRAIN：本组收尾 | LOCAL_SEND_RETIRE | 本组发送退休及scratch重置 |
| INIT_COUNT_BUFFERS：计数发送 | SEND_COUNT_SETUP | 发送计数阶段的缓冲及初始数据 |
| INIT_COUNT_BUFFERS：R0接收 | RECV_COUNT_SETUP | R0轮询计数用缓冲 |
| COUNT_COMPUTE：向量直方图 | COUNT_HISTOGRAM | 专家频数统计 |
| COUNT_COMPUTE：逐专家前缀扫描 | COUNT_EXPERT_SCAN | 小规模专家计数 |
| COUNT_COMPUTE：本组描述符 | LOCAL_ROUTE_BUILD | 稳定q与本组初发任务生成 |
| FORMAT_OUTPUT：Prepare | RECV_SETUP | 接收器初始化整体范围 |
| FORMAT_OUTPUT：TryBatch | RECV_BATCH | 一批就绪探测/消费的范围 |
| RECV_INPUT：IssueLoad | RECV_LOAD_ISSUE | 发起payload加载 |
| RECV_INPUT：ConsumeLoaded | RECV_LOAD_WAIT | 等payload可供Scalar读取 |
| RECV_BUFFER_WAIT：lane复用 | RECV_LANE_REUSE_WAIT | 等前一包读/写退休 |
| RECV_BUFFER_WAIT：非去重复制前 | RECV_DIRECT_COPY_WAIT | 直接输出前MTE2→MTE3依赖 |
| RECV_BUFFER_WAIT：relay/metadata准备后 | RECV_RELAY_COPY_WAIT | Scalar与MTE2→MTE3依赖 |
| TRANSITION_DRAIN：主流程转接收 | RECV_TRANSITION_DRAIN | 转接收的本核排空 |
| FINAL_DRAIN：dispatch/combine | DISPATCH_FINAL_DRAIN / COMBINE_FINAL_DRAIN | 对应算子最后退休 |

combine同样拆分，避免把不同作用合成一个阶段：
- COMBINE_PENDING_SETUP / COMBINE_NOTIFY_ISSUE / COMBINE_DIRECT_SETUP /
  COMBINE_PARTIAL_ROUTE_SETUP：准备待归约行、通知、DIRECT筛选、PARTIAL路由准备。
- DIRECT/PARTIAL/SOURCE前缀区分直接返回、局部归约返回、源token最终归约。
- *_LOAD_ISSUE与*_LOAD_WAIT分开；*_INPUT_REUSE_WAIT与*_OUTPUT_WAIT分开；
  COMBINE_SHARED_LANE_RECLAIM独立记录继承输出lane的首次回收。
- DIRECT表缓存、类型表搬入、rank表搬入、单route重载分别命名。
- COMBINE_SERVER_READY_WAIT和COMBINE_SOURCE_POLL分别表示服务器级通知等待与源结果探测。
- 具体ID及全部名称见dc_seg_stages.json，具体源码位置见本页末尾清单。

允许多个位置同名的仅为真正相同的操作：
SEND_TOKENS（实际发送循环）、SEND_SUBMIT_END（最后一次发送提交标记）、
SEND_COMPLETE_WAIT（发送完成等待）。S/R角色可用于区分来源。
同一函数的多轮执行继续同名；不能按执行序号起名来掩盖含义不同。

## 记录与兼容性

- ID0..80保留原定义，用于解析旧数据；58个新ID追加到81..138。
- trace记录仍192，metric容量139；每核88×8 int64=5632B（5.5KiB），512B对齐。
  旧schema12每核4608B（4.5KiB）；新增1024B/核。不是payload UB预算的1KiB增量。
- timer内部状态也随metric数增加，ON插桩开销可能变化；不能将不同schema的ON kernel直接作性能结论。
- 仍只记录每阶段前两次实际区间，metrics累计所有calls/ticks。不同作用已不再抢占同一个名字的前两条。
- trace容量满时仍可能省略后续区间，omitted记录省略量；不能据缺少区间认定工作未执行。
  KERNEL与SEND_SUBMIT_END保留原容量预留规则。
- parser按schema选择72或88行，schema7/9/10/11/12仍可离线读取；旧PREPARE_INPUT不重命名，
  因为仅凭旧ID无法可靠判断具体是哪段代码。不能伪造拆分历史累计值。
- 新旧schema混合trace和跨schema诊断耗时比被拒绝。perf OFF可按原实验身份规则比较。
- 父子区间仍嵌套；例如ROUTE_CLASSIFY在SEND_TOKENS中，不能相加重复计时。
- RECV_SETUP和RECV_PREPARE是同一准备函数的不同层级范围，不是可相加的独立工作。

## 验证与使用

旧ON wheel不接受新采集；工程内按原流程重建完整ON wheel，全rank保持一致。
新采集沿用原shape/seed和run.py，使用新run-name；新analyze.py也能读取旧schema12。
本次7个业务文件剥离LL_SEG标签及注释后token序列与9e24a79完全一致。
此检查证明源码业务token未改，不等于CANN编译或设备等价验证。

新增检查覆盖：全部58个新阶段的区间/累计解析、schema12原名原尺寸、拒绝跨schema对比、
禁止重新使用模糊PREPARE_INPUT、ON/OFF真实记录器容量和GM延迟导出。
本地验证记录见docs/BUILD_DISPATCH_SPLIT.md。尚无CANN编译或NPU验收。

## 当前源码打点位置

| 阶段 | 源文件:行 |
|---|---|
| RECV_TRANSITION_DRAIN | csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h:118 |
| DISPATCH_FINAL_DRAIN | csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h:122 |
| CAL_CUMSUM | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:78 |
| LOCAL_SEND_POOL_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:97 |
| SEND_COUNTS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:216 |
| SEND_COUNT_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:217 |
| COUNT_HISTOGRAM | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:255 |
| COUNT_EXPERT_SCAN | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:288 |
| LOCAL_ROUTE_BUILD | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:302 |
| COUNT_SEND_ISSUE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:337 |
| RECV_COUNT_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:358 |
| PREFIX_COMPLETE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:376 |
| TRANSITION_DRAIN | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:378 |
| PREFIX_LOAD | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:390 |
| WRITE_COUNTS_OUTPUT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:398 |
| CLEAR_RECV_STATUS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:447 |
| WAIT_RECV_COUNTS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:467 |
| POLL_RECV_COUNTS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:468 |
| BUILD_RECV_CUMSUM | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:497 |
| BROADCAST_PREFIX | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:511 |
| PUBLISH_PREFIX_FLAGS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:515 |
| WAIT_CUMSUM | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h:528 |
| RECV_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:69 |
| RECV_PREPARE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:72 |
| BUILD_EXPERT_TASKS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:90 |
| LOCAL_WINDOW_COPY | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:129 |
| RECV_BATCH | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:130 |
| RECV_COMPLETE_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:186 |
| RECV_POLL | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:207 |
| RECV_LANE_REUSE_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:233 |
| RECV_LOAD_ISSUE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:236 |
| RECV_COPY | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:256 |
| RECV_LOAD_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:258 |
| RECV_OUTPUT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:261 |
| RECV_DIRECT_COPY_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:275 |
| RECV_RELAY_COPY_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:312 |
| RELAY_ISSUE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:324 |
| CLEAR_LOCAL_FLAGS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h:365 |
| SEND_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:79 |
| SEND_TOKENS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:167 |
| SEND_DRAIN | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:179 |
| SEND_SUBMIT_END | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:188 |
| SEND_COMPLETE_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:189 |
| LOCAL_SEND_SETUP | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:206 |
| SEND_TOKENS | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:251 |
| SEND_SUBMIT_END | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:260 |
| LOCAL_SEND_RETIRE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:264 |
| SEND_COMPLETE_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:267 |
| ROUTE_CLASSIFY | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:290 |
| CAL_TOKEN | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:425 |
| TOKEN_QUANT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:448 |
| TOKEN_INPUT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:522 |
| TOKEN_QUANT_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:566 |
| TOKEN_INPUT_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:570 |
| TOKEN_METADATA | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:577 |
| TOKEN_BUFFER_WAIT | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:582 |
| TOKEN_PACK | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:585 |
| TOKEN_SEND_ISSUE | csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h:606 |
| COMBINE_PENDING_SETUP | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:116 |
| COMBINE_NOTIFY_ISSUE | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:123 |
| COMBINE_PARTIAL_SEND | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:129 |
| COMBINE_SOURCE_RECV | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:132 |
| COMBINE_SHARED_OUTPUT_DRAIN | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:135 |
| COMBINE_DIRECT_RECV | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:139 |
| COMBINE_DIRECT_SEND | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:144 |
| FINAL_DRAIN | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:152 |
| COMBINE_DIRECT_TABLE_CACHE | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:269 |
| COMBINE_DIRECT_SETUP | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:278 |
| COMBINE_TYPE_LOAD | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:282 |
| COMBINE_DIRECT_RANKS_LOAD | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:305 |
| COMBINE_DIRECT_TOKEN | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:362 |
| COMBINE_DIRECT_ROUTE_LOAD | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:363 |
| COMBINE_DIRECT_INPUT | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:374 |
| COMBINE_DIRECT_PACK | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:383 |
| COMBINE_BUFFER_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:384 |
| COMBINE_COPY_PACK | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:387 |
| COMBINE_SEND_ISSUE | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:400 |
| COMBINE_DIRECT_DRAIN | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:411 |
| COMBINE_SERVER_READY_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h:442 |
| COMBINE_PARTIAL_SETUP | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:30 |
| COMBINE_PARTIAL_TOKEN | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:85 |
| COMBINE_PARTIAL_ROUTE_SETUP | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:129 |
| COMBINE_PARTIAL_ROUTE_LOAD | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:190 |
| COMBINE_PARTIAL_LOAD_ISSUE | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:199 |
| COMBINE_PARTIAL_INPUT_REUSE_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:201 |
| COMBINE_PARTIAL_REDUCE | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:225 |
| COMBINE_PARTIAL_LOAD_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:232 |
| COMBINE_PARTIAL_PACK | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:257 |
| COMBINE_PARTIAL_OUTPUT_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:258 |
| COMBINE_PARTIAL_COPY_PACK | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:261 |
| COMBINE_PARTIAL_SEND_ISSUE | csrc/kernels/combine/low_latency/combine_server_dedup_partial.h:271 |
| COMBINE_SOURCE_DRAIN | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:61 |
| COMBINE_SOURCE_SETUP | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:107 |
| COMBINE_SOURCE_POLL | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:159 |
| COMBINE_CLEAR_FLAGS | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:174 |
| COMBINE_SOURCE_LOAD_ISSUE | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:189 |
| COMBINE_SOURCE_INPUT_REUSE_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:191 |
| COMBINE_SHARED_LANE_RECLAIM | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:201 |
| COMBINE_SOURCE_TOKEN | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:213 |
| COMBINE_SOURCE_LOAD_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:221 |
| COMBINE_SOURCE_REDUCE | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:224 |
| COMBINE_SOURCE_OUTPUT_WAIT | csrc/kernels/combine/low_latency/combine_server_dedup_source_aggregate.h:245 |
