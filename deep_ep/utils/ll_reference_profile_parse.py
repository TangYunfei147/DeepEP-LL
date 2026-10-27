"""CPU-only parser for the independent LL dispatch reference-cycle format.

Timestamp subtraction is performed on Python integers before unit conversion.
Presence comes from the validity mask: an executed zero is distinct from absence.
"""
import math

FORMAT = 'll-reference-cycles-v1'
SCHEMA = 1
MAGIC = 0x4C4C4350
SLOTS, FIELDS, WORD_COUNT = 8, 8, 64
DEFAULT_FREQUENCY_HZ = 1_000_000_000
WORDS = ('kernel_entry', 'config_ready', 'magic_begin', 'magic_end', 'counts_init_end', 'role_init_end', 'send_count_begin', 'send_count_issue_end', 'sender_begin', 'sender_end_after_MTE3_S', 'recv_count_begin', 'recv_count_end', 'cumsum_begin', 'cumsum_issue_end', 'complete_cumsum_begin', 'complete_cumsum_issue_end', 'count_drain_begin', 'count_drain_end_PIPE_ALL', 'receiver_begin', 'receiver_end', 'final_drain_begin', 'final_drain_end_PIPE_ALL', 'reserved', 'reserved', 'count_ready_poll_begin', 'count_ready_poll_end_after_V_S', 'cumsum_peer_join_begin', 'cumsum_peer_join_clear_issue_end', 'prefix_ready_wait_begin', 'prefix_ready_wait_clear_issue_end', 'receive_prefix_loaded_after_MTE2_S', 'receive_tasks_built', 'receive_buffers_ready', 'receive_packet_poll_accum_cycles', 'receive_load_consume_submit_accum_cycles', 'receive_lane_wait_clear_issue_end', 'receive_final_PIPE_ALL_end', 'receive_task_count', 'send_buffers_routes_ready', 'send_packets_issue_end', 'send_metadata_end_after_MTE3_S', 'receive_probe_batch_count', 'first_packet_ready', 'send_route_count', 'reserved', 'reserved', 'reserved', 'reserved', 'reserved', 'reserved', 'rank', 'AIV_core', 'EP', 'active_aiv', 'BS', 'schema', 'operation', 'before_finalize', 'valid_mask', 'magic', 'boundary_mode', 'complete', 'reserved', 'reserved')
WORD_NAMES = WORDS
STAGES = ('kernel', 'before_finalize', 'entry_config', 'init_prepare', 'magic', 'count_init', 'role_init', 'count_send', 'sender', 'count_receive', 'count_prefix', 'count_complete', 'count_drain', 'receiver', 'final_drain', 'count_ready_poll', 'cumsum_join', 'prefix_ready', 'recv_prefix_load', 'recv_tasks', 'recv_buffers', 'recv_packet_poll', 'recv_load_consume', 'recv_loop_clear', 'recv_final_drain', 'send_buffers_routes', 'send_packets', 'send_drain_metadata', 'recv_first_packet')
STAGE_INFO = {
    'kernel': {'label': '核总耗时', 'start': 0, 'end': 21, 'kind': 'difference', 'boundary': '核入口至最终 PIPE_ALL/CTRL 恢复完成；包含区间内打点开销，不含末尾诊断身份/有效位发布及 cache 写回', 'relationship': '主阶段', 'reference_formula': '(w21-w0)/1000'},
    'before_finalize': {'label': '写出前总耗时', 'start': 0, 'end': 57, 'kind': 'difference', 'boundary': '核入口至诊断 Finish 开始；包含此前主点 scalar GM 写入，不含随后身份/有效位发布及 cache 写回', 'relationship': '总计', 'reference_formula': '(w57-w0)/1000'},
    'entry_config': {'label': '入口配置', 'start': 0, 'end': 1, 'kind': 'difference', 'boundary': '入口配置和 tracer 初始化', 'relationship': '主阶段', 'reference_formula': '(w1-w0)/1000'},
    'init_prepare': {'label': 'Init 准备', 'start': 1, 'end': 2, 'kind': 'difference', 'boundary': 'kernel Init 绑定至 magic 开始', 'relationship': '主阶段', 'reference_formula': '(w2-w1)/1000'},
    'magic': {'label': 'magic', 'start': 2, 'end': 3, 'kind': 'difference', 'boundary': '窗口 magic 轮转，沿用原 DMA/事件/TPipe Reset', 'relationship': '主阶段', 'reference_formula': '(w3-w2)/1000'},
    'count_init': {'label': '计数 Init', 'start': 3, 'end': 4, 'kind': 'difference', 'boundary': '计数参数、控制区、CTRL 配置', 'relationship': '主阶段', 'reference_formula': '(w4-w3)/1000'},
    'role_init': {'label': '收发 Init', 'start': 4, 'end': 5, 'kind': 'difference', 'boundary': 'sender/receiver Init 和布局计算', 'relationship': '主阶段', 'reference_formula': '(w5-w4)/1000'},
    'count_send': {'label': '计数统计/发送提交', 'start': 6, 'end': 7, 'kind': 'difference', 'boundary': 'SendCnt，包含已有同步，远端写可异步继续', 'relationship': '主阶段', 'reference_formula': '(w7-w6)/1000'},
    'sender': {'label': '发送 payload/元数据完成', 'start': 8, 'end': 9, 'kind': 'difference', 'boundary': 'Sender.Process，结束已有 MTE3_S 等待', 'relationship': '发送主阶段', 'reference_formula': '(w9-w8)/1000'},
    'count_receive': {'label': '计数接收/归约', 'start': 10, 'end': 11, 'kind': 'difference', 'boundary': 'RecvCnt，包含 ready 轮询及归约；普通路径在此清计数标记，NDR packed-count 路径随后在 CalcCumsum 中清理', 'relationship': '接收主阶段', 'reference_formula': '(w11-w10)/1000'},
    'count_prefix': {'label': 'cumsum 汇合/发布提交', 'start': 12, 'end': 13, 'kind': 'difference', 'boundary': 'CalcCumsum，包含 GetCumSum 和前缀发布；NDR packed-count 清计数标记位于 GetCumSum 返回后，不计入其子段', 'relationship': '接收主阶段', 'reference_formula': '(w13-w12)/1000'},
    'count_complete': {'label': '前缀就绪/加载/输出提交', 'start': 14, 'end': 15, 'kind': 'difference', 'boundary': 'CompleteCumsum，输出计数仍可异步继续', 'relationship': '接收主阶段', 'reference_formula': '(w15-w14)/1000'},
    'count_drain': {'label': '计数阶段 drain', 'start': 16, 'end': 17, 'kind': 'difference', 'boundary': '现有 PIPE_ALL，回收计数 UB 前完成写入', 'relationship': '接收主阶段', 'reference_formula': '(w17-w16)/1000'},
    'receiver': {'label': '接收 payload 完成', 'start': 18, 'end': 19, 'kind': 'difference', 'boundary': 'Receiver.Process，含构建任务、轮询、接收及收尾', 'relationship': '接收主阶段', 'reference_formula': '(w19-w18)/1000'},
    'final_drain': {'label': '最终 drain/恢复 CTRL', 'start': 20, 'end': 21, 'kind': 'difference', 'boundary': 'Counts.Finish，包含现有 PIPE_ALL', 'relationship': '主阶段', 'reference_formula': '(w21-w20)/1000'},
    'count_ready_poll': {'label': '计数 ready 轮询', 'start': 24, 'end': 25, 'kind': 'difference', 'boundary': 'WaitDispatch 轮询至最后一次 V_S 完成', 'relationship': '计数接收/归约的细分', 'reference_formula': '(w25-w24)/1000'},
    'cumsum_join': {'label': 'cumsum 跨核等待/清理提交', 'start': 26, 'end': 27, 'kind': 'difference', 'boundary': 'GetCumSum，包含等待片段总数和清标记提交', 'relationship': 'cumsum 汇合/发布提交的细分', 'reference_formula': '(w27-w26)/1000'},
    'prefix_ready': {'label': '前缀 ready 等待/清理提交', 'start': 28, 'end': 29, 'kind': 'difference', 'boundary': 'WaitCumSumFlag 等待及清理提交', 'relationship': '前缀就绪/加载/输出提交的细分', 'reference_formula': '(w29-w28)/1000'},
    'recv_prefix_load': {'label': '接收前缀加载', 'start': 18, 'end': 30, 'kind': 'difference', 'boundary': 'Receiver reset / 前缀 DMA，结束 MTE2_S', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w30-w18)/1000'},
    'recv_tasks': {'label': '接收任务构建', 'start': 30, 'end': 31, 'kind': 'difference', 'boundary': 'ReceiveTasks.Build', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w31-w30)/1000'},
    'recv_buffers': {'label': '接收缓冲准备', 'start': 31, 'end': 32, 'kind': 'difference', 'boundary': '探测、lane、标记缓冲和事件准备', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w32-w31)/1000'},
    'recv_packet_poll': {'label': 'packet 就绪轮询累计', 'start': None, 'end': 33, 'kind': 'sum', 'boundary': '每批完整轮询区间累积，未对每次空扫描插点', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w33/1000'},
    'recv_load_consume': {'label': '加载/消费提交累计', 'start': None, 'end': 34, 'kind': 'sum', 'boundary': 'IssueLoad / ConsumeLoaded / RemovePending 区间累积', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w34/1000'},
    'recv_loop_clear': {'label': '接收循环/清标记提交', 'start': 32, 'end': 35, 'kind': 'difference', 'boundary': '包含 packet 循环、lane 等待指令和 ClearFlags 提交', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w35-w32)/1000'},
    'recv_final_drain': {'label': '接收最终 PIPE_ALL', 'start': 35, 'end': 36, 'kind': 'difference', 'boundary': '接收输出、转发及标记清理的现有完成屏障', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w36-w35)/1000'},
    'send_buffers_routes': {'label': '发送缓冲/路由加载', 'start': 8, 'end': 38, 'kind': 'difference', 'boundary': 'Sender UB/lane 分配及路由加载，含已有同步', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w38-w8)/1000'},
    'send_packets': {'label': '路由/packet 发送提交', 'start': 38, 'end': 39, 'kind': 'difference', 'boundary': 'SendWithDedup / SendWithoutDedup，包含已有 lane 依赖', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w39-w38)/1000'},
    'send_drain_metadata': {'label': '发送 drain/元数据完成', 'start': 39, 'end': 40, 'kind': 'difference', 'boundary': '尾 RDMA batch Flush（消息完成quiet）、lane 回收、source metadata 写回及已有 MTE3_S；无尾批次时不执行 Flush', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w40-w39)/1000'},
    'recv_first_packet': {'label': '首次 packet 就绪等待', 'start': 32, 'end': 42, 'kind': 'difference', 'boundary': '接收缓冲准备完至首个 packet batch ready', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w42-w32)/1000'},
}

DATA_WORDS = frozenset((*range(22), *range(24, 44), 57))
COUNT_WORDS = frozenset((33, 34, 37, 41, 43))
VALID_BITS = sum(1 << index for index in DATA_WORDS)
RESERVED_WORDS = frozenset((22, 23, *range(44, 50), 62, 63))
COMMON_REQUIRED = frozenset((*range(8), 20, 21, 57))
SEND_REQUIRED = frozenset((8, 9, 43))
RECEIVE_REQUIRED = frozenset((*range(10, 20), 30, 31, 33, 34, 37, 41))


def parse_records(raw, operation='dispatch', frequency_hz=DEFAULT_FREQUENCY_HZ):
    """Validate a complete capture and return 29 reference-compatible durations."""
    if operation != 'dispatch':
        raise ValueError('Reference profile supports dispatch only')
    if frequency_hz is not None and (isinstance(frequency_hz, bool) or
            not isinstance(frequency_hz, (int, float)) or
            not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    if not isinstance(raw, (list, tuple)) or not 16 <= len(raw) <= 72 or len(raw) % 2:
        raise ValueError('Expected actual even AIV count in [16,72]')
    records, cores, identity = [], [], None
    for core, slots in enumerate(raw):
        if len(slots) != SLOTS or any(len(row) != FIELDS for row in slots):
            raise ValueError('Reference timer shape must be [active_aiv,8,8]')
        words = [value for row in slots for value in row]
        if any(type(value) is not int or not 0 <= value < 2**64 for value in words):
            raise ValueError('Invalid uint64 raw word')
        if any(words[index] for index in RESERVED_WORDS):
            raise ValueError('Nonzero reserved word')
        if words[51] != core or words[53] != len(raw):
            raise ValueError('Core identity or active AIV count mismatch')
        if words[55:57] != [SCHEMA, 1] or words[59:62] != [MAGIC, 1, 1]:
            raise ValueError('Incomplete or mismatched reference header')
        rank, world, tokens = words[50], words[52], words[54]
        if world not in (16, 32, 64, 128, 256) or rank >= world or not 1 <= tokens <= 65536:
            raise ValueError('Invalid rank/world identity')
        current = (rank, world, tokens)
        if identity is not None and current != identity:
            raise ValueError('Mixed rank/world/BS identity across cores')
        identity = current
        mask = words[58]
        if mask & ~VALID_BITS:
            raise ValueError('Validity mask marks metadata or reserved words')
        valid = {index for index in DATA_WORDS if mask & (1 << index)}
        if not COMMON_REQUIRED.issubset(valid):
            raise ValueError('Missing mandatory common reference points')
        if any(words[index] for index in DATA_WORDS - valid):
            raise ValueError('Unmarked data word must remain zero')
        if not words[0] <= words[21] <= words[57]:
            raise ValueError('Invalid completed kernel timestamp order')
        for index in valid - COUNT_WORDS:
            if not words[0] <= words[index] <= words[57]:
                raise ValueError('Timestamp outside own core interval')
        role = 'send' if core < len(raw)//2 else 'receive'
        required = SEND_REQUIRED if role == 'send' else RECEIVE_REQUIRED
        if not required.issubset(valid):
            raise ValueError('Missing mandatory ' + role + ' reference points')
        task_word = 43 if role == 'send' else 37
        if role == 'send' and valid.intersection({*range(10, 20), *range(24, 38), 41, 42}):
            raise ValueError('Send core contains receive-only points')
        if role == 'receive' and valid.intersection({8, 9, 38, 39, 40, 43}):
            raise ValueError('Receive core contains send-only points')
        task_points = {38, 39, 40} if role == 'send' else {32, 35, 36, 42}
        if words[task_word] == 0:
            if valid.intersection(task_points):
                raise ValueError('Empty ' + role + ' core contains task-only timestamps')
            if role == 'receive' and (words[33] or words[34] or words[41]):
                raise ValueError('Empty receive core has nonzero accumulation or packet batches')
        else:
            if not task_points.issubset(valid):
                raise ValueError('Nonempty ' + role + ' core lacks task timestamps')
            if role == 'receive' and not 1 <= words[41] <= words[37]:
                raise ValueError('Receive packet batches must be in [1, task_count]')
        # Count ownership and payload task ownership are different partitions.
        # Count detail pairs may be absent even on a nonempty receiver; never
        # infer their execution from w37. A single endpoint is always corrupt.
        for first, last in ((24, 25), (26, 27), (28, 29)):
            if (first in valid) != (last in valid):
                raise ValueError('Incomplete optional count interval')
        for stage in STAGES:
            info = STAGE_INFO[stage]
            start, end = info['start'], info['end']
            if end not in valid or (start is not None and start not in valid):
                continue
            first = words[start] if start is not None else None
            last = words[end] if start is not None else None
            total = words[end] - first if start is not None else words[end]
            if total < 0:
                raise ValueError('Invalid timestamp order for ' + stage)
            if start is None and total > words[21] - words[0]:
                raise ValueError('Accumulated interval exceeds own kernel')
            calls = words[41] if start is None else 1
            record = dict(core=core, role=role, stage=stage, first_tick=first,
                          last_tick=last, total_ticks=total, calls=calls)
            if frequency_hz is not None:
                record['total_us'] = total * 1e6 / frequency_hz
            records.append(record)
        cores.append(dict(core=core, role=role, task_count=words[task_word],
                          batch_count=words[41] if role == 'receive' else None,
                          valid_mask=mask))
    return dict(format=FORMAT, schema=SCHEMA, operation='dispatch', active_aiv=len(raw),
                boundary_mode='reference', frequency_hz=frequency_hz, rank=identity[0],
                world_size=identity[1], num_tokens=identity[2], records=records, cores=cores,
                clock='AscendC::GetSystemCycle; independent core clocks',
                interpretation='Same-core integer differences; nested stages cannot be added; '
                               'w33/w34 are accumulated cycles, not first-to-last envelopes.')
