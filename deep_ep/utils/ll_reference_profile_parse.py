"""Current-only MTE shared-receive adaptation of reference-cycle profiling.

Raw words are decoded as uint64; same-core subtraction uses Python integers.
No NDR role partition or RDMA completion is inferred for this MTE branch.
"""
import math

FORMAT='ll-reference-cycles-mte-shared-v1'
SCHEMA=6
MAGIC=0x4C4C4350
SLOTS,FIELDS,WORD_COUNT=80,8,640
DEFAULT_FREQUENCY_HZ=1_000_000_000
IMPLEMENTATION='mte-shared-receive'
WORDS=(
    'kernel_entry',
    'config_ready',
    'magic_begin',
    'magic_end',
    'counts_init_end',
    'role_init_end',
    'send_count_begin',
    'send_count_issue_end',
    'sender_begin',
    'sender_end_submit',
    'recv_count_begin',
    'recv_count_end',
    'cumsum_begin',
    'cumsum_issue_end',
    'complete_cumsum_begin',
    'complete_cumsum_issue_end',
    'count_drain_begin',
    'count_drain_end_PIPE_ALL',
    'receiver_begin',
    'receiver_end',
    'final_drain_begin',
    'final_drain_end_PIPE_ALL',
    'send_cores',
    'recv_cores',
    'count_ready_poll_begin',
    'count_ready_poll_end_after_V_S',
    'cumsum_peer_join_begin',
    'cumsum_peer_join_clear_issue_end',
    'prefix_ready_wait_begin',
    'prefix_ready_wait_end',
    'receive_prefix_view_ready',
    'receive_tasks_built',
    'receive_buffers_ready',
    'receive_packet_poll_accum_cycles',
    'receive_load_consume_submit_accum_cycles',
    'receive_lane_wait_clear_issue_end',
    'receive_final_PIPE_ALL_end',
    'receive_task_count',
    'send_buffers_routes_ready',
    'send_packets_issue_end',
    'send_metadata_end_submit',
    'receive_probe_batch_count',
    'first_packet_ready',
    'send_route_count',
    'reserved',
    'reserved',
    'reserved',
    'reserved',
    'reserved',
    'reserved',
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
    'initial_core_role',
    'packet_detail_version',
    'send_remote_packets',
    'send_group_packets',
    'remote_first_begin',
    'remote_last_put_return',
    'group_first_begin',
    'group_last_submit_return',
    'second_remote_begin',
    'second_remote_enqueue',
    'second_remote_put_begin',
    'second_remote_put_return',
    'second_remote_member_q_ticks',
    'second_remote_token',
    'second_remote_k',
    'second_remote_expert',
    'second_remote_peer',
    'batch_quiet_ticks',
    'batch_flush_count',
    'last_quiet_end',
    'second_remote_triggered_flush',
    'remote_first_order',
    'remote_last_order',
    'group_first_order',
    'group_last_order',
    'send_packet_count',
    'send_lanes',
    'second_stage_queued',
    'second_stage_wait_begin',
    'second_stage_wait_end',
    'deferred_remote_packets',
    'second_previous_submit_ticks',
    'second_post_deferred',
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
    'detail_reserved',
)+tuple('packet_log_word_'+str(i) for i in range(512))
STAGES=('kernel', 'before_finalize', 'entry_config', 'init_prepare', 'magic', 'count_init', 'role_init', 'count_send', 'sender', 'count_receive', 'count_prefix', 'count_complete', 'count_drain', 'receiver', 'final_drain', 'count_ready_poll', 'cumsum_join', 'prefix_ready', 'recv_prefix_load', 'recv_tasks', 'recv_buffers', 'recv_packet_poll', 'recv_load_consume', 'recv_loop_clear', 'recv_final_drain', 'send_buffers_routes', 'send_packets', 'send_drain_metadata', 'recv_first_packet')
STAGE_INFO={
    'kernel': {'label': '核总耗时', 'start': 0, 'end': 21, 'kind': 'difference', 'boundary': '核入口至最终 PIPE_ALL/CTRL 恢复完成；包含区间内打点开销，不含末尾诊断身份/有效位发布及 cache 写回', 'relationship': '主阶段', 'reference_formula': '(w21-w0)/1000'},
    'before_finalize': {'label': '写出前总耗时', 'start': 0, 'end': 57, 'kind': 'difference', 'boundary': '核入口至诊断 Finish 开始；包含此前主点 scalar GM 写入，不含随后身份/有效位发布及 cache 写回', 'relationship': '总计', 'reference_formula': '(w57-w0)/1000'},
    'entry_config': {'label': '入口配置', 'start': 0, 'end': 1, 'kind': 'difference', 'boundary': '入口配置和 tracer 初始化', 'relationship': '主阶段', 'reference_formula': '(w1-w0)/1000'},
    'init_prepare': {'label': 'Init 准备', 'start': 1, 'end': 2, 'kind': 'difference', 'boundary': 'kernel Init 绑定至 magic 开始', 'relationship': '主阶段', 'reference_formula': '(w2-w1)/1000'},
    'magic': {'label': 'magic', 'start': 2, 'end': 3, 'kind': 'difference', 'boundary': '窗口 magic 轮转，沿用原 DMA/事件/TPipe Reset', 'relationship': '主阶段', 'reference_formula': '(w3-w2)/1000'},
    'count_init': {'label': '计数 Init', 'start': 3, 'end': 4, 'kind': 'difference', 'boundary': '计数参数、控制区、CTRL 配置', 'relationship': '主阶段', 'reference_formula': '(w4-w3)/1000'},
    'role_init': {'label': '角色 Init', 'start': 4, 'end': 5, 'kind': 'difference', 'boundary': '当前LL Sender/Receiver对象Init和布局配置，不重复计数Init', 'relationship': '主阶段', 'reference_formula': '(w5-w4)/1000'},
    'count_send': {'label': '计数统计/发送提交', 'start': 6, 'end': 7, 'kind': 'difference', 'boundary': 'SendCnt，包含已有同步，远端写可异步继续', 'relationship': '主阶段', 'reference_formula': '(w7-w6)/1000'},
    'sender': {'label': '发送 payload/元数据提交', 'start': 8, 'end': 9, 'kind': 'difference', 'boundary': '当前MTE Sender.Process或R本组发送函数返回；Async返回不表示MTE3完成', 'relationship': '源发送主阶段', 'reference_formula': '(w9-w8)/1000'},
    'count_receive': {'label': '计数接收/归约', 'start': 10, 'end': 11, 'kind': 'difference', 'boundary': '当前R0 RecvCnt，包含ready轮询、归约和原清理依赖；其他核未执行则留空', 'relationship': '接收主阶段', 'reference_formula': '(w11-w10)/1000'},
    'count_prefix': {'label': 'cumsum 汇合/发布提交', 'start': 12, 'end': 13, 'kind': 'difference', 'boundary': '当前R0 CalcCumsum，包含VF前缀及共享epoch提交；末点不表示最终DMA完成', 'relationship': '接收主阶段', 'reference_formula': '(w13-w12)/1000'},
    'count_complete': {'label': '前缀就绪/加载/输出提交', 'start': 14, 'end': 15, 'kind': 'difference', 'boundary': '当前所有接收参与核CompleteCumsum，含已有prefix ready等待和前缀加载', 'relationship': '接收主阶段', 'reference_formula': '(w15-w14)/1000'},
    'count_drain': {'label': '计数阶段 drain', 'start': 16, 'end': 17, 'kind': 'difference', 'boundary': '仅实际存在的计数阶段PIPE_ALL；当前stage3没有此独立drain，留空', 'relationship': '接收主阶段', 'reference_formula': '(w17-w16)/1000'},
    'receiver': {'label': '接收 payload 完成', 'start': 18, 'end': 19, 'kind': 'difference', 'boundary': 'Receiver.Process，含构建任务、轮询、接收及收尾', 'relationship': '接收主阶段', 'reference_formula': '(w19-w18)/1000'},
    'final_drain': {'label': '最终 drain/恢复 CTRL', 'start': 20, 'end': 21, 'kind': 'difference', 'boundary': 'Counts.Finish，包含现有 PIPE_ALL', 'relationship': '主阶段', 'reference_formula': '(w21-w20)/1000'},
    'count_ready_poll': {'label': '计数 ready 轮询', 'start': 24, 'end': 25, 'kind': 'difference', 'boundary': 'WaitDispatch 轮询至最后一次 V_S 完成', 'relationship': '计数接收/归约的细分', 'reference_formula': '(w25-w24)/1000'},
    'cumsum_join': {'label': 'cumsum 跨核等待/清理提交', 'start': 26, 'end': 27, 'kind': 'difference', 'boundary': '仅存在跨核cumsum join时记录；当前R0单生产者路径无此区间，留空', 'relationship': 'cumsum 汇合/发布提交的细分', 'reference_formula': '(w27-w26)/1000'},
    'prefix_ready': {'label': '前缀 ready 等待', 'start': 28, 'end': 29, 'kind': 'difference', 'boundary': '当前WaitCumSumFlag原等待；共享epoch不由各消费者清理', 'relationship': '前缀就绪/加载/输出提交的细分', 'reference_formula': '(w29-w28)/1000'},
    'recv_prefix_load': {'label': '接收前缀视图准备', 'start': 18, 'end': 30, 'kind': 'difference', 'boundary': '当前receiver入口至前缀视图就绪；overlap复用前序加载，fallback包含本函数的原加载，均不新增DMA', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w30-w18)/1000'},
    'recv_tasks': {'label': '接收任务构建', 'start': 30, 'end': 31, 'kind': 'difference', 'boundary': '按当前共享接收分片构建本核任务与pending；不引入metadata核任务表', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w31-w30)/1000'},
    'recv_buffers': {'label': '接收缓冲准备', 'start': 31, 'end': 32, 'kind': 'difference', 'boundary': '任务构建后准备原探测、lane、标记缓冲和事件；无任务则留空', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w32-w31)/1000'},
    'recv_packet_poll': {'label': 'packet 就绪轮询累计', 'start': None, 'end': 33, 'kind': 'sum', 'boundary': '等待开始至成功ready批次，空扫描不重复读clock；累计不代表连续独占流水', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w33/1000'},
    'recv_load_consume': {'label': '加载/消费提交累计', 'start': None, 'end': 34, 'kind': 'sum', 'boundary': '成功ready批次的原加载、转发及本地输出提交累计；包含原依赖', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w34/1000'},
    'recv_loop_clear': {'label': '接收循环/清标记提交', 'start': 32, 'end': 35, 'kind': 'difference', 'boundary': '包含 packet 循环、lane 等待指令和 ClearFlags 提交', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w35-w32)/1000'},
    'recv_final_drain': {'label': '接收最终 PIPE_ALL', 'start': 35, 'end': 36, 'kind': 'difference', 'boundary': '接收输出、转发及标记清理的现有完成屏障', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w36-w35)/1000'},
    'send_buffers_routes': {'label': '发送缓冲/路由加载', 'start': 8, 'end': 38, 'kind': 'difference', 'boundary': 'Sender UB/lane 分配及路由加载，含已有同步', 'relationship': '发送 payload/元数据提交的细分', 'reference_formula': '(w38-w8)/1000'},
    'send_packets': {'label': '路由/packet 发送提交', 'start': 38, 'end': 39, 'kind': 'difference', 'boundary': 'SendWithDedup / SendWithoutDedup，包含已有 lane 依赖', 'relationship': '发送 payload/元数据提交的细分', 'reference_formula': '(w39-w38)/1000'},
    'send_drain_metadata': {'label': '发送收尾/元数据提交', 'start': 39, 'end': 40, 'kind': 'difference', 'boundary': '原输入预取尾包处理、source metadata提交和函数收尾；Async不新增MTE3_S', 'relationship': '发送 payload/元数据提交的细分', 'reference_formula': '(w40-w39)/1000'},
    'recv_first_packet': {'label': '首次 packet 就绪等待', 'start': 32, 'end': 42, 'kind': 'difference', 'boundary': '接收缓冲准备完至首个 packet batch ready', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w42-w32)/1000'},
}

DATA_WORDS=frozenset((*range(22),*range(24,44),57))
COUNT_WORDS=frozenset((33,34,37,41,43))
VALID_BITS=sum(1<<index for index in DATA_WORDS)
RESERVED_WORDS=frozenset(range(44,50))
COMMON_REQUIRED=frozenset((0,1,2,3,4,5,20,21,57))
SOURCE_REQUIRED=frozenset((8,9,43))
RECEIVE_REQUIRED=frozenset((18,19,30,31,33,34,37,41))
SENDER_PACKET_METRICS={'group_submit_span':'本组包首开始至末提交返回'}


def parse_packet_detail(words,valid,frequency_hz):
    if words[64]!=3 or any(words[109:128]):
        raise ValueError('Invalid MTE packet detail version/reserved metadata')
    # These slots describe an NDR cross-host put/quiet pipeline which this
    # implementation does not execute. Missing measurements are never zero time.
    if any(words[i] for i in (65,67,68,*range(71,86),*range(90,96))):
        raise ValueError('RDMA-only packet detail is invalid for MTE')
    def difference(a,b):
        if not words[0]<=words[a]<=words[b]<=words[57]:
            raise ValueError('Invalid same-core packet interval')
        return words[b]-words[a]
    total,group,lanes=words[88],words[66],words[89]
    if group>total or lanes>8 or (total and not lanes):
        raise ValueError('Invalid MTE source packet counters')
    ticks={}
    if group:
        ticks['group_submit_span']=difference(69,70)
        if not 1<=words[86]<=words[87]<=total:
            raise ValueError('Invalid group packet order')
    elif any(words[i] for i in (69,70,86,87)):
        raise ValueError('Empty group send has observations')
    sender=dict(kind='sender',packet_count=total,group_packets=group,lanes=lanes,
                ticks=ticks,durations={k:v*1e6/frequency_hz if frequency_hz is not None else v
                                      for k,v in ticks.items()},raw_metadata=words[64:96])
    if 37 not in valid:
        if any(words[96:]):raise ValueError('Receiver detail without receiver execution')
        receiver=None
    else:
        rows,capacity,task_count,ready,processed,receive_lanes=words[96:102]
        if not 0<=rows==capacity<=min(task_count,128) or task_count!=words[37] or ready!=task_count or processed!=task_count:
            raise ValueError('Invalid receive trace coverage')
        if receive_lanes>8 or words[107]!=int(rows<task_count):
            raise ValueError('Invalid receive trace lanes/truncation')
        if task_count and not receive_lanes:
            raise ValueError('Nonempty receiver has no lanes')
        if not task_count and (receive_lanes or words[108]):
            raise ValueError('Empty receiver contains lane/batch observations')
        if words[108]>task_count:
            raise ValueError('Invalid receive output batch count')
        export_ticks=difference(102,103)
        if words[102]<words[21]:raise ValueError('Packet trace exported before business completion')
        if words[105]!=words[54] or words[106]<1:
            raise ValueError('Invalid receive trace shape')
        entries=[]
        for j in range(rows):
            packed=words[128+j*4:132+j*4]
            fields=[v for value in packed for v in (value&0xffffffff,value>>32)]
            found,batch,order,source,token,k,role_members,slot=fields
            packet_role=role_members&255;members=role_members>>8
            if (not 1<=found<=task_count or not 1<=order<=task_count or not 1<=batch<=words[41]
                    or packet_role not in (1,2,3) or source>=words[52] or token>=words[54]
                    or k>=32 or members>=32 or (packet_role!=2 and members)):
                raise ValueError('Invalid receive packet identity/order')
            local_expert=(slot//words[105])%words[106]
            if ((slot//words[105])//words[106]-words[50])%words[52]!=source:
                raise ValueError('Receive slot/source mismatch')
            entries.append(dict(task=j,output_row=words[104]+j,ready_order=found,ready_batch=batch,
                process_order=order,source_rank=source,source_token=token,header_k=k,
                packet_role={1:'DIRECT',2:'FIRST_HOP',3:'TERMINAL'}[packet_role],forward_members=members,
                destination_expert=words[50]*words[106]+local_expert,q=slot%words[105],slot=slot))
        for field in ('ready_order','process_order'):
            if len({e[field] for e in entries})!=rows:raise ValueError('Duplicate receive trace order')
        if any(words[128+rows*4:]):raise ValueError('Nonzero unused receive trace storage')
        receiver=dict(kind='receiver',entries=entries,total_tasks=task_count,recorded_tasks=rows,
                      ready_batches=words[41],truncated=bool(words[107]),lanes=receive_lanes,
                      export_ticks=export_ticks,raw_metadata=words[96:128])
    return dict(kind='shared',sender=sender,receiver=receiver,raw_metadata=words[64:128])


def parse_records(raw,operation='dispatch',frequency_hz=DEFAULT_FREQUENCY_HZ):
    if operation!='dispatch':raise ValueError('Reference profile supports dispatch only')
    if frequency_hz is not None and (isinstance(frequency_hz,bool) or not isinstance(frequency_hz,(int,float))
            or not math.isfinite(frequency_hz) or frequency_hz<=0):
        raise ValueError('Counter frequency must be finite and positive')
    if not isinstance(raw,(list,tuple)) or len(raw)!=56:
        raise ValueError('Expected the current fixed 56 AIV configuration')
    records=[];cores=[];identity=None
    for core,slots in enumerate(raw):
        if len(slots)!=SLOTS or any(len(row)!=FIELDS for row in slots):
            raise ValueError('Reference timer shape must be [56,80,8]')
        words=[value for row in slots for value in row]
        if any(type(value) is not int or not -(1<<63)<=value<(1<<64) for value in words):
            raise ValueError('Invalid uint64/int64 raw word')
        words=[value&((1<<64)-1) for value in words]
        if any(words[index] for index in RESERVED_WORDS):raise ValueError('Nonzero reserved word')
        if words[51]!=core or words[53]!=56:raise ValueError('Core identity or AIV count mismatch')
        if words[55:57]!=[SCHEMA,1] or words[59:62]!=[MAGIC,1,1]:
            raise ValueError('Incomplete or mismatched current reference header')
        rank,world,tokens=words[50],words[52],words[54]
        if world not in (16,32,64,128,256) or rank>=world or not 1<=tokens<=65536:
            raise ValueError('Invalid rank/world identity')
        send,receive=words[22:24]
        if send<1 or receive<1 or send+receive!=56 or words[62]!=0:
            raise ValueError('Invalid current S/R core split or metadata core identity')
        if words[63]!=(0 if core<send else 1):raise ValueError('Initial core role mismatch')
        current=(rank,world,tokens,send,receive)
        if identity is not None and current!=identity:raise ValueError('Mixed rank/world/BS/S/R identity across cores')
        identity=current;mask=words[58]
        if mask&~VALID_BITS:raise ValueError('Validity mask marks metadata or reserved words')
        valid={index for index in DATA_WORDS if mask&(1<<index)}
        if not COMMON_REQUIRED<=valid:raise ValueError('Missing mandatory common reference points')
        if any(words[index] for index in DATA_WORDS-valid):raise ValueError('Unmarked data word must remain zero')
        if not words[0]<=words[21]<=words[57]:raise ValueError('Invalid completed kernel timestamp order')
        for index in valid-COUNT_WORDS:
            if not words[0]<=words[index]<=words[57]:raise ValueError('Timestamp outside own core interval')
        for chain in ((0,1,2,3,4,5),(8,38,39,40,9),(18,30,31,32,42,35,36,19),(20,21,57)):
            ticks=[words[index] for index in chain if index in valid]
            if ticks!=sorted(ticks):raise ValueError('Invalid same-core boundary order')
        role='send' if core<send else 'receive'
        for first,last in ((6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(24,25),(26,27),(28,29)):
            if (first in valid)!=(last in valid):raise ValueError('Incomplete optional reference interval')
        if valid&SOURCE_REQUIRED:
            if not SOURCE_REQUIRED<=valid:raise ValueError('Missing mandatory source reference points')
            task_points={38,39,40}
            if words[43] and not task_points<=valid:raise ValueError('Nonempty source lacks task timestamps')
            if not words[43] and valid&task_points:raise ValueError('Empty source has task-only timestamps')
        elif valid&{38,39,40}:raise ValueError('Source details without source execution')
        if valid&RECEIVE_REQUIRED:
            if not RECEIVE_REQUIRED<=valid:raise ValueError('Missing mandatory shared-receiver points')
            if words[37] and not {32,35,36,42}<=valid:raise ValueError('Nonempty receiver lacks packet observations')
            if words[37] and not words[41]:raise ValueError('Nonempty receiver lacks ready batches')
            if not words[37] and (valid&{32,35,36,42} or any(words[i] for i in (33,34,41))):
                raise ValueError('Empty receiver has task-only observations')
        for stage in STAGES:
            info=STAGE_INFO[stage];start,end=info['start'],info['end']
            if end not in valid or (start is not None and start not in valid):continue
            first=words[start] if start is not None else None
            last=words[end] if start is not None else None
            total=words[end]-first if start is not None else words[end]
            if total<0:raise ValueError('Invalid timestamp order for '+stage)
            if start is None and total>words[21]-words[0]:raise ValueError('Accumulated interval exceeds own kernel')
            item=dict(core=core,role=role,stage=stage,first_tick=first,last_tick=last,total_ticks=total,
                      calls=words[41] if start is None else 1)
            if frequency_hz is not None:item['total_us']=total*1e6/frequency_hz
            records.append(item)
        cores.append(dict(core=core,role=role,task_count=words[37] if 37 in valid else None,
                          source_task_count=words[43] if 43 in valid else None,
                          batch_count=words[41] if 41 in valid else None,valid_mask=mask,
                          packet_detail=parse_packet_detail(words,valid,frequency_hz)))
    return dict(format=FORMAT,schema=SCHEMA,implementation=IMPLEMENTATION,operation='dispatch',active_aiv=56,
                boundary_mode='reference',frequency_hz=frequency_hz,rank=identity[0],world_size=identity[1],
                num_tokens=identity[2],send_cores=identity[3],recv_cores=identity[4],metadata_cores=0,
                records=records,cores=cores,clock='AscendC::GetSystemCycle; independent core clocks',
                interpretation='Same-core integer differences; nested stages cannot be added; '
                               'w33/w34 are accumulated cycles, not first-to-last envelopes. '
                               'Both initial S/R partitions may send and receive; no NDR metadata cores or RDMA completion is inferred.')
