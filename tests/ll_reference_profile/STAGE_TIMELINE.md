# Small-B MTE reference 阶段与源码

参考metadata-cores@c3d8640的29阶段记录方式；当前schema7，每核640word。只dispatch，业务分核/依赖保持。

w44..49为S/R/dispatch接收核数、算法选择、overlap stage、small-B prefix配置；w62=0，w63为实际执行角色位：1发送、2接收、4调用count。兼任发送/接收时两条工作均记录，不能按静态S/R位置丢弃。

| 阶段 | 公式（1GHz，µs） | 当前真实边界 | 包含关系 |
|---|---|---|---|
| `kernel` / 核总耗时 | `(w21-w0)/1000` | 核入口至最终 PIPE_ALL/CTRL 恢复完成；包含区间内打点开销，不含末尾诊断身份/有效位发布及 cache 写回 | 主阶段 |
| `before_finalize` / 写出前总耗时 | `(w57-w0)/1000` | 核入口至诊断 Finish 开始；包含此前主点 scalar GM 写入，不含随后身份/有效位发布及 cache 写回 | 总计 |
| `entry_config` / 入口配置 | `(w1-w0)/1000` | 入口配置和 tracer 初始化 | 主阶段 |
| `init_prepare` / Init 准备 | `(w2-w1)/1000` | kernel Init 绑定至 magic 开始 | 主阶段 |
| `magic` / magic | `(w3-w2)/1000` | 窗口 magic 轮转，沿用原 DMA/事件/TPipe Reset | 主阶段 |
| `count_init` / 计数 Init | `(w4-w3)/1000` | 计数参数、控制区、CTRL 配置 | 主阶段 |
| `role_init` / 角色 Init | `(w5-w3)/1000` | sender/receiver Init 与布局准备，包含从 magic 后开始的 count Init；按源码顺序解释 | 主阶段 |
| `count_send` / 计数统计/发送提交 | `(w7-w6)/1000` | SendCnt，包含已有同步，远端写可异步继续 | 主阶段 |
| `sender` / 发送阶段返回 | `(w9-w8)/1000` | Sender.Process/ProcessGroupLocal 返回；Async 路径只提交，在原 Counts.Finish 才保证完成 | 发送主阶段 |
| `count_receive` / 计数接收/归约 | `(w11-w10)/1000` | 实际 RecvCnt 或 PrepareRecvCnt/TryRecvCnt 成功；stream 为自身片段 ReadCounts | 接收主阶段 |
| `count_prefix` / cumsum 汇合/发布提交 | `(w13-w12)/1000` | CalcCumsum；stream 为局部前缀、片段总数发布和前序片段汇合 | 接收主阶段 |
| `count_complete` / 前缀就绪/加载/输出提交 | `(w15-w14)/1000` | CompleteCumsum；stream 为 PublishOutputs，pair 延伸至配对交接完成 | 接收主阶段 |
| `count_drain` / 计数阶段 drain | `(w17-w16)/1000` | 仅记录调用方原有计数 PIPE_ALL；不存在该屏障的路径留空 | 接收主阶段 |
| `receiver` / 接收 payload 完成 | `(w19-w18)/1000` | Receiver.Process，含构建任务、轮询、接收及收尾 | 接收主阶段 |
| `final_drain` / 最终 drain/恢复 CTRL | `(w21-w20)/1000` | Counts.Finish，包含现有 PIPE_ALL | 主阶段 |
| `count_ready_poll` / 计数 ready 轮询 | `(w25-w24)/1000` | 实际 count 轮询；协作路径从 PrepareRecvCnt 结束至 TryRecvCnt 成功，包含两者间调度 | 计数接收/归约的细分 |
| `cumsum_join` / cumsum 跨核等待/清理提交 | `(w27-w26)/1000` | GetCumSum 或 stream ReadPredecessors；不存在该独立区间则留空 | cumsum 汇合/发布提交的细分 |
| `prefix_ready` / 前缀 ready 等待/清理提交 | `(w29-w28)/1000` | 原有 WaitCumSumFlag 或 parallel ready 等待；不新增同步或清理 | 前缀就绪/加载/输出提交的细分 |
| `recv_prefix_load` / 接收前缀准备/加载 | `(w30-w18)/1000` | Receiver 进入至实际前缀加载/共享视图就绪；空 stream 片段未加载则留空 | 接收 payload 完成的细分 |
| `recv_tasks` / 接收任务构建 | `(w31-w30)/1000` | 按当前算法构建稳定输出任务；包括 stream/pair 的实际任务划分 | 接收 payload 完成的细分 |
| `recv_buffers` / 接收缓冲准备 | `(w32-w31)/1000` | 任务已建立至探测/lane 缓冲准备完成；保持原 lane 数量 | 接收 payload 完成的细分 |
| `recv_packet_poll` / packet 就绪轮询累计 | `w33/1000` | 一次等待起点至成功就绪批次；空扫描不重复 clock，协作路径可包含间隙 | 接收 payload 完成的细分 |
| `recv_load_consume` / 加载/消费提交累计 | `w34/1000` | 成功就绪批次的加载/转发/输出提交和 pending 更新累计 | 接收 payload 完成的细分 |
| `recv_loop_clear` / 接收循环/清标记提交 | `(w35-w32)/1000` | 包含 packet 循环、lane 等待指令和 ClearFlags 提交 | 接收 payload 完成的细分 |
| `recv_final_drain` / 接收最终 PIPE_ALL | `(w36-w35)/1000` | 接收输出、转发及标记清理的现有完成屏障 | 接收 payload 完成的细分 |
| `send_buffers_routes` / 发送缓冲/路由加载 | `(w38-w8)/1000` | Sender UB/lane 分配及路由加载，含已有同步 | 发送 payload/元数据完成的细分 |
| `send_packets` / 路由/packet 发送提交 | `(w39-w38)/1000` | 路由、输入预取、打包及 MTE 提交；协作模式含已有回调间隔 | 发送 payload/元数据完成的细分 |
| `send_drain_metadata` / 发送 drain/元数据完成 | `(w40-w39)/1000` | 原有 lane 退休/源 metadata 提交；Async 不增加完成等待，无 RDMA quiet | 发送 payload/元数据完成的细分 |
| `recv_first_packet` / 首次 packet 就绪等待 | `(w42-w32)/1000` | 接收缓冲准备完至首个 packet batch ready | 接收 payload 完成的细分 |

## 原始word

| word | 名称 |
|---|---|
| w0 | `kernel_entry` |
| w1 | `config_ready` |
| w2 | `magic_begin` |
| w3 | `magic_end` |
| w4 | `counts_init_end` |
| w5 | `role_init_end` |
| w6 | `send_count_begin` |
| w7 | `send_count_issue_end` |
| w8 | `sender_begin` |
| w9 | `sender_end_after_existing_dependencies` |
| w10 | `recv_count_begin` |
| w11 | `recv_count_end` |
| w12 | `cumsum_begin` |
| w13 | `cumsum_issue_end` |
| w14 | `complete_cumsum_begin` |
| w15 | `complete_cumsum_issue_end` |
| w16 | `count_drain_begin` |
| w17 | `count_drain_end_existing_dependencies` |
| w18 | `receiver_begin` |
| w19 | `receiver_end` |
| w20 | `final_drain_begin` |
| w21 | `final_drain_end_PIPE_ALL` |
| w22 | `reserved` |
| w23 | `reserved` |
| w24 | `count_ready_poll_begin` |
| w25 | `count_ready_poll_end_after_V_S` |
| w26 | `cumsum_peer_join_begin` |
| w27 | `cumsum_peer_join_clear_issue_end` |
| w28 | `prefix_ready_wait_begin` |
| w29 | `prefix_ready_wait_clear_issue_end` |
| w30 | `receive_prefix_loaded_after_MTE2_S` |
| w31 | `receive_tasks_built` |
| w32 | `receive_buffers_ready` |
| w33 | `receive_packet_poll_accum_cycles` |
| w34 | `receive_load_consume_submit_accum_cycles` |
| w35 | `receive_lane_wait_clear_issue_end` |
| w36 | `receive_final_PIPE_ALL_end` |
| w37 | `receive_task_count` |
| w38 | `send_buffers_routes_ready` |
| w39 | `send_packets_issue_end` |
| w40 | `send_metadata_end_after_MTE3_S` |
| w41 | `receive_probe_batch_count` |
| w42 | `first_packet_ready` |
| w43 | `send_packet_task_count` |
| w44 | `send_cores` |
| w45 | `recv_cores` |
| w46 | `dispatch_receive_cores` |
| w47 | `dispatch_implementation_id` |
| w48 | `overlap_stage` |
| w49 | `small_b_prefix_cores` |
| w50 | `rank` |
| w51 | `AIV_core` |
| w52 | `EP` |
| w53 | `active_aiv` |
| w54 | `BS` |
| w55 | `schema` |
| w56 | `operation` |
| w57 | `before_finalize` |
| w58 | `valid_mask` |
| w59 | `magic` |
| w60 | `boundary_mode` |
| w61 | `complete` |
| w62 | `metadata_cores` |
| w63 | `executed_role_mask` |
| w64 | `packet_detail_version` |
| w65 | `send_peer_packets` |
| w66 | `send_self_packets` |
| w67 | `peer_first_begin` |
| w68 | `peer_last_mte_submit_return` |
| w69 | `self_first_begin` |
| w70 | `self_last_mte_submit_return` |
| w71 | `second_peer_begin` |
| w72 | `second_peer_prepared` |
| w73 | `second_peer_mte_begin` |
| w74 | `second_peer_mte_return` |
| w75 | `second_peer_member_q_ticks` |
| w76 | `second_peer_token` |
| w77 | `second_peer_k` |
| w78 | `second_peer_expert` |
| w79 | `second_peer_rank` |
| w80 | `reserved_mte_detail` |
| w81 | `reserved_mte_detail` |
| w82 | `reserved_mte_detail` |
| w83 | `reserved_mte_detail` |
| w84 | `peer_first_order` |
| w85 | `peer_last_order` |
| w86 | `self_first_order` |
| w87 | `self_last_order` |
| w88 | `send_packet_count` |
| w89 | `send_lanes` |
| w90 | `reserved_mte_detail` |
| w91 | `reserved_mte_detail` |
| w92 | `reserved_mte_detail` |
| w93 | `reserved_mte_detail` |
| w94 | `reserved_mte_detail` |
| w95 | `reserved_mte_detail` |
| w96 | `recv_trace_rows` |
| w97 | `recv_trace_capacity` |
| w98 | `recv_trace_total_tasks` |
| w99 | `recv_trace_ready_seen` |
| w100 | `recv_trace_processed_seen` |
| w101 | `recv_trace_lanes` |
| w102 | `recv_trace_export_begin` |
| w103 | `recv_trace_export_end` |
| w104 | `recv_trace_row_begin` |
| w105 | `recv_trace_bs` |
| w106 | `recv_trace_local_experts` |
| w107 | `recv_trace_truncated` |
| w108 | `output_ready_batches` |
| w109 | `detail_reserved` |
| w110 | `detail_reserved` |
| w111 | `detail_reserved` |
| w112 | `detail_reserved` |
| w113 | `detail_reserved` |
| w114 | `detail_reserved` |
| w115 | `detail_reserved` |
| w116 | `detail_reserved` |
| w117 | `detail_reserved` |
| w118 | `detail_reserved` |
| w119 | `detail_reserved` |
| w120 | `detail_reserved` |
| w121 | `detail_reserved` |
| w122 | `detail_reserved` |
| w123 | `detail_reserved` |
| w124 | `detail_reserved` |
| w125 | `detail_reserved` |
| w126 | `detail_reserved` |
| w127 | `detail_reserved` |

w128..639为最多128行×32B的R稳定任务日志；每行8个uint32，按ready顺序、ready批次、处理顺序、源rank、源token、header_k、role/members、slot记录。

S第二跨卡包在当前算法实际包处理入口计时：cb59为SendPacket，current为输入预取后ConsumePacket。主q在入口之前，成员q在FillTail内单独累计。w80..83、w90..95保留为0，不制造RDMA staging/quiet统计。

计数等待、前序汇合、prefix就绪只记录实际执行的区间。空记录片段或无相关等待时留空；接收零任务保留已执行准备和0累计，不补造PIPE_ALL。

current的cooperative模式中等待可跨发送回调，sender/receiver区间可以重叠。w33/w34是成功批次累计，w41是批次数。所有区间只在同核相减；不相加父子或不同核最大阶段，不解释成完整OFF task。

R日志仅占原lane末尾剩余UB，不减少lane、不增加pool项。原Counts.Finish后，诊断导出另有S→MTE3/MTE3→S依赖；w21不含该导出，w57包含导出但不含其后的header/cache发布。
