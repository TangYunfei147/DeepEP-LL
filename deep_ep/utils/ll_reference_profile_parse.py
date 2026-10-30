"""Reference-style LL dispatch parser adapted from metadata-cores@c3d8640.

Only MTE schema7 is accepted. Same-core uint64 subtraction precedes conversion;
validity distinguishes an absent interval from a measured zero. No legacy decoder.
"""
import math

FORMAT = 'll-reference-cycles-v1'
SCHEMA = 7
MAGIC = 0x4C4C4350
SLOTS, FIELDS, WORD_COUNT = 80, 8, 640
DEFAULT_FREQUENCY_HZ = 1_000_000_000
WORDS = ('kernel_entry',
 'config_ready',
 'magic_begin',
 'magic_end',
 'counts_init_end',
 'role_init_end',
 'send_count_begin',
 'send_count_issue_end',
 'sender_begin',
 'sender_end_after_existing_dependencies',
 'recv_count_begin',
 'recv_count_end',
 'cumsum_begin',
 'cumsum_issue_end',
 'complete_cumsum_begin',
 'complete_cumsum_issue_end',
 'count_drain_begin',
 'count_drain_end_existing_dependencies',
 'receiver_begin',
 'receiver_end',
 'final_drain_begin',
 'final_drain_end_PIPE_ALL',
 'reserved',
 'reserved',
 'count_ready_poll_begin',
 'count_ready_poll_end_after_V_S',
 'cumsum_peer_join_begin',
 'cumsum_peer_join_clear_issue_end',
 'prefix_ready_wait_begin',
 'prefix_ready_wait_clear_issue_end',
 'receive_prefix_loaded_after_MTE2_S',
 'receive_tasks_built',
 'receive_buffers_ready',
 'receive_packet_poll_accum_cycles',
 'receive_load_consume_submit_accum_cycles',
 'receive_lane_wait_clear_issue_end',
 'receive_final_PIPE_ALL_end',
 'receive_task_count',
 'send_buffers_routes_ready',
 'send_packets_issue_end',
 'send_metadata_end_after_MTE3_S',
 'receive_probe_batch_count',
 'first_packet_ready',
 'send_packet_task_count',
 'send_cores',
 'recv_cores',
 'dispatch_receive_cores',
 'dispatch_implementation_id',
 'overlap_stage',
 'small_b_prefix_cores',
 'rank',
 'AIV_core',
 'EP',
 'active_aiv',
 'BS',
 'schema',
 'operation',
 'before_finalize',
 'valid_mask',
 'magic',
 'boundary_mode',
 'complete',
 'metadata_cores',
 'executed_role_mask',
 'packet_detail_version',
 'send_peer_packets',
 'send_self_packets',
 'peer_first_begin',
 'peer_last_mte_submit_return',
 'self_first_begin',
 'self_last_mte_submit_return',
 'second_peer_begin',
 'second_peer_prepared',
 'second_peer_mte_begin',
 'second_peer_mte_return',
 'second_peer_member_q_ticks',
 'second_peer_token',
 'second_peer_k',
 'second_peer_expert',
 'second_peer_rank',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'peer_first_order',
 'peer_last_order',
 'self_first_order',
 'self_last_order',
 'send_packet_count',
 'send_lanes',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'reserved_mte_detail',
 'recv_trace_rows',
 'recv_trace_capacity',
 'recv_trace_total_tasks',
 'recv_trace_ready_seen',
 'recv_trace_processed_seen',
 'recv_trace_lanes',
 'recv_trace_export_begin',
 'recv_trace_export_end',
 'recv_trace_row_begin',
 'recv_trace_bs',
 'recv_trace_local_experts',
 'recv_trace_truncated',
 'output_ready_batches',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved',
 'detail_reserved') + tuple(f"packet_log_word_{i}" for i in range(512))
STAGES = ('kernel', 'before_finalize', 'entry_config', 'init_prepare', 'magic', 'count_init', 'role_init', 'count_send', 'sender', 'count_receive', 'count_prefix', 'count_complete', 'count_drain', 'receiver', 'final_drain', 'count_ready_poll', 'cumsum_join', 'prefix_ready', 'recv_prefix_load', 'recv_tasks', 'recv_buffers', 'recv_packet_poll', 'recv_load_consume', 'recv_loop_clear', 'recv_final_drain', 'send_buffers_routes', 'send_packets', 'send_drain_metadata', 'recv_first_packet')
STAGE_INFO = {'kernel': {'label': '核总耗时',
            'start': 0,
            'end': 21,
            'kind': 'difference',
            'boundary': '核入口至最终 PIPE_ALL/CTRL 恢复完成；包含区间内打点开销，不含末尾诊断身份/有效位发布及 cache 写回',
            'relationship': '主阶段',
            'reference_formula': '(w21-w0)/1000'},
 'before_finalize': {'label': '写出前总耗时',
                     'start': 0,
                     'end': 57,
                     'kind': 'difference',
                     'boundary': '核入口至诊断 Finish 开始；包含此前主点 scalar GM 写入，不含随后身份/有效位发布及 cache 写回',
                     'relationship': '总计',
                     'reference_formula': '(w57-w0)/1000'},
 'entry_config': {'label': '入口配置',
                  'start': 0,
                  'end': 1,
                  'kind': 'difference',
                  'boundary': '入口配置和 tracer 初始化',
                  'relationship': '主阶段',
                  'reference_formula': '(w1-w0)/1000'},
 'init_prepare': {'label': 'Init 准备',
                  'start': 1,
                  'end': 2,
                  'kind': 'difference',
                  'boundary': 'kernel Init 绑定至 magic 开始',
                  'relationship': '主阶段',
                  'reference_formula': '(w2-w1)/1000'},
 'magic': {'label': 'magic',
           'start': 2,
           'end': 3,
           'kind': 'difference',
           'boundary': '窗口 magic 轮转，沿用原 DMA/事件/TPipe Reset',
           'relationship': '主阶段',
           'reference_formula': '(w3-w2)/1000'},
 'count_init': {'label': '计数 Init',
                'start': 3,
                'end': 4,
                'kind': 'difference',
                'boundary': '计数参数、控制区、CTRL 配置',
                'relationship': '主阶段',
                'reference_formula': '(w4-w3)/1000'},
 'role_init': {'label': '角色 Init',
               'start': 3,
               'end': 5,
               'kind': 'difference',
               'boundary': 'sender/receiver Init 与布局准备，包含从 magic 后开始的 count Init；按源码顺序解释',
               'relationship': '主阶段',
               'reference_formula': '(w5-w3)/1000'},
 'count_send': {'label': '计数统计/发送提交',
                'start': 6,
                'end': 7,
                'kind': 'difference',
                'boundary': 'SendCnt，包含已有同步，远端写可异步继续',
                'relationship': '主阶段',
                'reference_formula': '(w7-w6)/1000'},
 'sender': {'label': '发送阶段返回',
            'start': 8,
            'end': 9,
            'kind': 'difference',
            'boundary': 'Sender.Process/ProcessGroupLocal 返回；Async 路径只提交，在原 Counts.Finish 才保证完成',
            'relationship': '发送主阶段',
            'reference_formula': '(w9-w8)/1000'},
 'count_receive': {'label': '计数接收/归约',
                   'start': 10,
                   'end': 11,
                   'kind': 'difference',
                   'boundary': '实际 RecvCnt 或 PrepareRecvCnt/TryRecvCnt 成功；stream 为自身片段 ReadCounts',
                   'relationship': '接收主阶段',
                   'reference_formula': '(w11-w10)/1000'},
 'count_prefix': {'label': 'cumsum 汇合/发布提交',
                  'start': 12,
                  'end': 13,
                  'kind': 'difference',
                  'boundary': 'CalcCumsum；stream 为局部前缀、片段总数发布和前序片段汇合',
                  'relationship': '接收主阶段',
                  'reference_formula': '(w13-w12)/1000'},
 'count_complete': {'label': '前缀就绪/加载/输出提交',
                    'start': 14,
                    'end': 15,
                    'kind': 'difference',
                    'boundary': 'CompleteCumsum；stream 为 PublishOutputs，pair 延伸至配对交接完成',
                    'relationship': '接收主阶段',
                    'reference_formula': '(w15-w14)/1000'},
 'count_drain': {'label': '计数阶段 drain',
                 'start': 16,
                 'end': 17,
                 'kind': 'difference',
                 'boundary': '仅记录调用方原有计数 PIPE_ALL；不存在该屏障的路径留空',
                 'relationship': '接收主阶段',
                 'reference_formula': '(w17-w16)/1000'},
 'receiver': {'label': '接收 payload 完成',
              'start': 18,
              'end': 19,
              'kind': 'difference',
              'boundary': 'Receiver.Process，含构建任务、轮询、接收及收尾',
              'relationship': '接收主阶段',
              'reference_formula': '(w19-w18)/1000'},
 'final_drain': {'label': '最终 drain/恢复 CTRL',
                 'start': 20,
                 'end': 21,
                 'kind': 'difference',
                 'boundary': 'Counts.Finish，包含现有 PIPE_ALL',
                 'relationship': '主阶段',
                 'reference_formula': '(w21-w20)/1000'},
 'count_ready_poll': {'label': '计数 ready 轮询',
                      'start': 24,
                      'end': 25,
                      'kind': 'difference',
                      'boundary': '实际 count 轮询；协作路径从 PrepareRecvCnt 结束至 TryRecvCnt 成功，包含两者间调度',
                      'relationship': '计数接收/归约的细分',
                      'reference_formula': '(w25-w24)/1000'},
 'cumsum_join': {'label': 'cumsum 跨核等待/清理提交',
                 'start': 26,
                 'end': 27,
                 'kind': 'difference',
                 'boundary': 'GetCumSum 或 stream ReadPredecessors；不存在该独立区间则留空',
                 'relationship': 'cumsum 汇合/发布提交的细分',
                 'reference_formula': '(w27-w26)/1000'},
 'prefix_ready': {'label': '前缀 ready 等待/清理提交',
                  'start': 28,
                  'end': 29,
                  'kind': 'difference',
                  'boundary': '原有 WaitCumSumFlag 或 parallel ready 等待；不新增同步或清理',
                  'relationship': '前缀就绪/加载/输出提交的细分',
                  'reference_formula': '(w29-w28)/1000'},
 'recv_prefix_load': {'label': '接收前缀准备/加载',
                      'start': 18,
                      'end': 30,
                      'kind': 'difference',
                      'boundary': 'Receiver 进入至实际前缀加载/共享视图就绪；空 stream 片段未加载则留空',
                      'relationship': '接收 payload 完成的细分',
                      'reference_formula': '(w30-w18)/1000'},
 'recv_tasks': {'label': '接收任务构建',
                'start': 30,
                'end': 31,
                'kind': 'difference',
                'boundary': '按当前算法构建稳定输出任务；包括 stream/pair 的实际任务划分',
                'relationship': '接收 payload 完成的细分',
                'reference_formula': '(w31-w30)/1000'},
 'recv_buffers': {'label': '接收缓冲准备',
                  'start': 31,
                  'end': 32,
                  'kind': 'difference',
                  'boundary': '任务已建立至探测/lane 缓冲准备完成；保持原 lane 数量',
                  'relationship': '接收 payload 完成的细分',
                  'reference_formula': '(w32-w31)/1000'},
 'recv_packet_poll': {'label': 'packet 就绪轮询累计',
                      'start': None,
                      'end': 33,
                      'kind': 'sum',
                      'boundary': '一次等待起点至成功就绪批次；空扫描不重复 clock，协作路径可包含间隙',
                      'relationship': '接收 payload 完成的细分',
                      'reference_formula': 'w33/1000'},
 'recv_load_consume': {'label': '加载/消费提交累计',
                       'start': None,
                       'end': 34,
                       'kind': 'sum',
                       'boundary': '成功就绪批次的加载/转发/输出提交和 pending 更新累计',
                       'relationship': '接收 payload 完成的细分',
                       'reference_formula': 'w34/1000'},
 'recv_loop_clear': {'label': '接收循环/清标记提交',
                     'start': 32,
                     'end': 35,
                     'kind': 'difference',
                     'boundary': '包含 packet 循环、lane 等待指令和 ClearFlags 提交',
                     'relationship': '接收 payload 完成的细分',
                     'reference_formula': '(w35-w32)/1000'},
 'recv_final_drain': {'label': '接收最终 PIPE_ALL',
                      'start': 35,
                      'end': 36,
                      'kind': 'difference',
                      'boundary': '接收输出、转发及标记清理的现有完成屏障',
                      'relationship': '接收 payload 完成的细分',
                      'reference_formula': '(w36-w35)/1000'},
 'send_buffers_routes': {'label': '发送缓冲/路由加载',
                         'start': 8,
                         'end': 38,
                         'kind': 'difference',
                         'boundary': 'Sender UB/lane 分配及路由加载，含已有同步',
                         'relationship': '发送 payload/元数据完成的细分',
                         'reference_formula': '(w38-w8)/1000'},
 'send_packets': {'label': '路由/packet 发送提交',
                  'start': 38,
                  'end': 39,
                  'kind': 'difference',
                  'boundary': '路由、输入预取、打包及 MTE 提交；协作模式含已有回调间隔',
                  'relationship': '发送 payload/元数据完成的细分',
                  'reference_formula': '(w39-w38)/1000'},
 'send_drain_metadata': {'label': '发送 drain/元数据完成',
                         'start': 39,
                         'end': 40,
                         'kind': 'difference',
                         'boundary': '原有 lane 退休/源 metadata 提交；Async 不增加完成等待，无 RDMA quiet',
                         'relationship': '发送 payload/元数据完成的细分',
                         'reference_formula': '(w40-w39)/1000'},
 'recv_first_packet': {'label': '首次 packet 就绪等待',
                       'start': 32,
                       'end': 42,
                       'kind': 'difference',
                       'boundary': '接收缓冲准备完至首个 packet batch ready',
                       'relationship': '接收 payload 完成的细分',
                       'reference_formula': '(w42-w32)/1000'}}

DATA_WORDS = frozenset((*range(22), *range(24,44), 57))
COUNT_WORDS = frozenset((33,34,37,41,43))
VALID_BITS = sum(1 << index for index in DATA_WORDS)
RESERVED_WORDS = frozenset((22,23))
COMMON_REQUIRED = frozenset((0,1,2,3,4,5,6,7,20,21,57))
SENDER_PACKET_METRICS = {
    'second_prepare_without_member_q': '第二跨卡MTE包准备（扣成员q区间）',
    'second_member_q': '第二跨卡MTE包成员q区间',
    'second_ready_to_submit': '第二包准备完成至MTE提交前',
    'second_mte_call': '第二跨卡包MTE提交调用',
    'second_prepared_to_submit_return': '第二包准备完成至MTE调用返回',
    'peer_submit_span': '跨卡包首开始至末MTE调用返回',
    'self_submit_span': '本卡包首开始至末MTE调用返回',
}


def parse_packet_detail(words, role, frequency_hz):
    if words[64] != 2 or any(words[80:84]) or any(words[90:96]) or any(words[109:128]):
        raise ValueError('Invalid MTE packet metadata or fabricated RDMA observations')
    def difference(a,b):
        if not words[0] <= words[a] <= words[b] <= words[57]:
            raise ValueError('Invalid packet interval')
        return words[b]-words[a]
    sender = receiver = None
    if words[63] & 1:
        peer, local = words[65],words[66]
        if peer+local != words[88] or words[88] != words[43] or not 0 <= words[89] <= 8:
            raise ValueError('Invalid sender packet counters')
        if words[88] and words[89] == 0:
            raise ValueError('Nonempty sender has no lanes')
        ticks={}
        for count,a,b,first,last,key in ((peer,67,68,84,85,'peer_submit_span'),(local,69,70,86,87,'self_submit_span')):
            if count:
                ticks[key]=difference(a,b)
                if not 1 <= words[first] <= words[last] <= words[88]:
                    raise ValueError('Invalid packet class order')
            elif any(words[i] for i in (a,b,first,last)):
                raise ValueError('Empty packet class contains timestamps')
        second=None
        if peer >= 2:
            preparation=difference(71,72)
            if words[75]>preparation:
                raise ValueError('Member-q time exceeds packet preparation')
            ticks.update(second_prepare_without_member_q=preparation-words[75],second_member_q=words[75],
                         second_ready_to_submit=difference(72,73),second_mte_call=difference(73,74),
                         second_prepared_to_submit_return=difference(72,74))
            if words[76]>=words[54] or words[77]>=32 or words[79]>=words[52] or words[79]==words[50]:
                raise ValueError('Invalid second peer packet identity')
            second=dict(token=words[76],header_k=words[77],expert=words[78],peer=words[79])
        elif any(words[71:80]):
            raise ValueError('Absent second peer packet has detail')
        sender=dict(kind='sender',peer_packets=peer,self_packets=local,packet_count=words[88],lanes=words[89],second=second,
                    ticks=ticks,durations={k:v*1e6/frequency_hz if frequency_hz is not None else v for k,v in ticks.items()},
                    packet_order_interleaves=bool(peer and local and max(words[84],words[86])<=min(words[85],words[87])),
                    intervals_interleave=bool(peer and local and max(words[67],words[69])<=min(words[68],words[70])))
    elif any(words[65:96]):
        raise ValueError('A core without sending contains sender packet metadata')
    if words[63] & 2:
        rows,capacity,total,ready,processed,lanes=words[96:102]
        if not 0 <= rows == capacity <= min(total,128) or total!=words[37] or ready!=total or processed!=total:
            raise ValueError('Invalid receive trace coverage')
        if not 0 <= lanes <= 8 or words[107]!=int(rows<total) or (total and lanes==0):
            raise ValueError('Invalid receive trace lanes/truncation')
        if words[108]!=words[41] or (total and not 1<=words[108]<=total) or (not total and words[108]):
            raise ValueError('Invalid receive-ready batch count')
        export_ticks=difference(102,103)
        if words[102]<words[21] or words[105]!=words[54] or words[106]<1:
            raise ValueError('Invalid post-business receiver export or shape')
        entries=[]
        for j in range(rows):
            packed=words[128+j*4:132+j*4]
            found,batch,order,source,token,k,role_members,slot=[x for value in packed for x in (value&0xffffffff,value>>32)]
            packet_role=role_members&255;members=role_members>>8
            if (not 1<=found<=total or not 1<=order<=total or not 1<=batch<=words[108] or
                packet_role not in (1,2,3) or source>=words[52] or token>=words[54] or
                k>=32 or members>=32 or slot>=words[52]*words[105]*words[106] or (packet_role!=2 and members)):
                raise ValueError('Invalid receive packet identity/order')
            local_expert=(slot//words[105])%words[106]
            if ((slot//words[105])//words[106]-words[50])%words[52]!=source:
                raise ValueError('Receive slot/source mismatch')
            entries.append(dict(task=j,output_row=words[104]+j,ready_order=found,ready_batch=batch,process_order=order,
                source_rank=source,source_token=token,header_k=k,packet_role={1:'DIRECT',2:'FIRST_HOP',3:'TERMINAL'}[packet_role],
                forward_members=members,destination_expert=words[50]*words[106]+local_expert,q=slot%words[105],slot=slot))
        for field in ('ready_order','process_order'):
            if len({e[field] for e in entries})!=rows:raise ValueError('Duplicate receive trace order')
        if any(words[128+rows*4:]):raise ValueError('Nonzero unused receive trace storage')
        receiver=dict(kind='receiver',entries=entries,total_tasks=total,recorded_tasks=rows,ready_batches=words[108],
                      truncated=bool(words[107]),lanes=lanes,export_ticks=export_ticks)
    elif any(words[96:]):
        raise ValueError('A core without receiving contains receiver packet metadata')
    return dict(kind='combined' if sender and receiver else 'sender' if sender else 'receiver' if receiver else 'count',
                sender=sender,receiver=receiver,raw_metadata=words[64:128])


def parse_records(raw,operation='dispatch',frequency_hz=DEFAULT_FREQUENCY_HZ):
    """Validate schema7 and compute the reference's 29 same-core intervals."""
    if operation!='dispatch':raise ValueError('Reference profile supports dispatch only')
    if frequency_hz is not None and (type(frequency_hz) not in (int,float) or not math.isfinite(frequency_hz) or frequency_hz<=0):
        raise ValueError('Counter frequency must be finite and positive')
    if not isinstance(raw,(list,tuple)) or len(raw)!=56:
        raise ValueError('This MTE branch requires the actual fixed56 AIV launch')
    records,cores,identity=[],[],None
    for core,slots in enumerate(raw):
        if len(slots)!=SLOTS or any(len(row)!=FIELDS for row in slots):raise ValueError('Expected [56,80,8] reference record')
        words=[value for row in slots for value in row]
        if any(type(v) is not int or not 0<=v<2**64 for v in words):raise ValueError('Invalid uint64 raw word')
        if any(words[i] for i in RESERVED_WORDS):raise ValueError('Nonzero reserved word')
        if words[51]!=core or words[53]!=len(raw):raise ValueError('Core identity or actual AIV mismatch')
        if words[55:57]!=[SCHEMA,1] or words[59:62]!=[MAGIC,1,1]:raise ValueError('Incomplete or mismatched reference header')
        rank,world,tokens=words[50],words[52],words[54]
        if world not in (16,32,64,128,256) or rank>=world or not 1<=tokens<=65536:raise ValueError('Invalid rank/world/BS')
        send,recv,dispatch_recv,implementation,overlap,prefix=words[44:50]
        if send<1 or recv<1 or send+recv!=len(raw) or words[62]!=0 or implementation not in (0,1,2,3,4):
            raise ValueError('Invalid MTE split or implementation identity')
        if implementation:
            if (send,recv,dispatch_recv,overlap,prefix)!=(28,28,28,0,0):raise ValueError('Invalid cb59 role configuration')
        elif dispatch_recv!=56 or overlap not in (1,2,3,4) or not 0<=prefix<=recv:
            raise ValueError('Invalid current role configuration')
        current=(rank,world,tokens,tuple(words[44:50]))
        if identity is not None and current!=identity:raise ValueError('Mixed rank/world/BS/build identity across cores')
        identity=current;mask=words[58]
        if mask & ~VALID_BITS:raise ValueError('Validity mask marks metadata/reserved words')
        valid={i for i in DATA_WORDS if mask&(1<<i)}
        if not COMMON_REQUIRED.issubset(valid):raise ValueError('Missing common reference points')
        if any(words[i] for i in DATA_WORDS-valid):raise ValueError('Unmarked data must remain zero')
        if not words[0]<=words[21]<=words[57]:raise ValueError('Invalid kernel timestamp order')
        if any(not words[0]<=words[i]<=words[57] for i in valid-COUNT_WORDS):raise ValueError('Timestamp outside own core interval')
        actual_role=int(8 in valid)|int(18 in valid)<<1|int(6 in valid)<<2
        if words[63]!=actual_role:raise ValueError('Executed role mask disagrees with recorded stages')
        role='send+receive' if actual_role&3==3 else 'send' if actual_role&1 else 'receive' if actual_role&2 else 'count'
        for first,last in ((6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(24,25),(26,27),(28,29),(30,31)):
            if (first in valid)!=(last in valid):raise ValueError('Incomplete reference interval')
            if first in valid and words[last]<words[first]:raise ValueError('Reversed reference interval')
        if actual_role&1:
            if 43 not in valid:raise ValueError('Missing sender task count')
            if words[43] and not {38,39,40}.issubset(valid):raise ValueError('Nonempty sender lacks preparation/submission points')
        elif valid.intersection({38,39,40,43}):raise ValueError('Non-sender has sending points')
        if actual_role&2:
            if not {32,33,34,37,41}.issubset(valid):raise ValueError('Missing receiver totals')
            if words[37] and (not {30,31,35,36,42}.issubset(valid) or not words[41]):raise ValueError('Nonempty receiver lacks observations')
        elif valid.intersection({*range(30,38),41,42}):raise ValueError('Non-receiver has receiving points')
        for stage in STAGES:
            spec=STAGE_INFO[stage];start,end=spec['start'],spec['end']
            if end not in valid or (start is not None and start not in valid):continue
            first=words[start] if start is not None else None
            total=words[end]-first if start is not None else words[end]
            if total<0 or (start is None and total>words[21]-words[0]):raise ValueError('Invalid stage duration: '+stage)
            record=dict(core=core,role=role,stage=stage,first_tick=first,last_tick=words[end] if start is not None else None,
                        total_ticks=total,calls=words[41] if start is None else 1)
            if frequency_hz is not None:record['total_us']=total*1e6/frequency_hz
            records.append(record)
        cores.append(dict(core=core,role=role,role_mask=actual_role,
            task_count=words[37] if actual_role&2 else words[43] if actual_role&1 else 0,
            send_task_count=words[43] if actual_role&1 else None,recv_task_count=words[37] if actual_role&2 else None,
            batch_count=words[41] if actual_role&2 else None,valid_mask=mask,
            packet_detail=parse_packet_detail(words,role,frequency_hz)))
    return dict(format=FORMAT,schema=SCHEMA,operation='dispatch',active_aiv=len(raw),metadata_cores=0,
        send_cores=identity[3][0],recv_cores=identity[3][1],dispatch_receive_cores=identity[3][2],
        dispatch_implementation_id=identity[3][3],overlap_stage=identity[3][4],small_b_prefix_cores=identity[3][5],
        boundary_mode='reference',frequency_hz=frequency_hz,rank=identity[0],world_size=identity[1],num_tokens=identity[2],
        records=records,cores=cores,clock='AscendC::GetSystemCycle; independent core clocks',
        interpretation='Same-core integer differences; nested or overlapping stages cannot be added; '
                       'MTE submit is not completion; w33/w34 are ready-batch accumulators, not envelopes.')
