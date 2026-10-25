"""Schema16: same-core milestones and deliberately scoped Scalar wait probes.

The two MTE3 modes change scheduling. Their wait/control differences describe
the intervened execution, never uninstrumented network or pipeline latency.
"""
import math

MAGIC = 0x4C4C5431
POINTS = ('ENTRY', 'INIT_DONE', 'COUNTS_SENT', 'COUNTS_RECEIVED',
          'PREFIX_LOCAL_DONE', 'PREFIX_READY', 'SEND_PREPARED', 'FIRST_SEND',
          'SEND_DONE', 'RECV_PREPARED', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY',
          'RECV_DONE', 'WORK_DONE', 'SOURCE_SUBMIT_DONE', 'PREFIX_WAIT_ENTER',
          'PREFIX_WAIT_EXIT', 'RECV_PENDING_EMPTY')
METRICS = ('TX_INPUT_READY', 'RX_PREPARE_VS', 'RX_PAYLOAD_MTE2S', 'RX_FINAL_ALL',
           'TX_REUSE_MTE3', 'TX_END_MTE3', 'RX_REUSE_MTE3', 'RX_END_MTE3')
MODES = {1: 'observe', 4: 'mte3-tx', 5: 'mte3-rx'}
ROLES = {1: 'send_token', 2: 'cumsum', 3: 'send_and_recv', 4: 'recv'}
PAIRS = (
    ('INIT', 'ENTRY', 'INIT_DONE'), ('KERNEL', 'INIT_DONE', 'WORK_DONE'),
    ('INIT_TO_SOURCE_SUBMIT_DONE', 'INIT_DONE', 'SOURCE_SUBMIT_DONE'),
    ('PREFIX_WAIT', 'PREFIX_WAIT_ENTER', 'PREFIX_WAIT_EXIT'),
    ('BEFORE_FIRST_RECV_POLL', 'INIT_DONE', 'FIRST_RECV_PROBE'),
    ('FIRST_PROBE_TO_FIRST_READY', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY'),
    ('FIRST_READY_TO_PENDING_EMPTY', 'FIRST_RECV_READY', 'RECV_PENDING_EMPTY'),
    ('PENDING_EMPTY_TO_RECV_DONE', 'RECV_PENDING_EMPTY', 'RECV_DONE'),
    ('RECV_DONE_TO_WORK_DONE', 'RECV_DONE', 'WORK_DONE'),
)


def parse_records(raw, operation, frequency_hz=None):
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (isinstance(frequency_hz, bool) or
            not isinstance(frequency_hz, (int, float)) or
            not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    if len(raw) != 56:
        raise ValueError('MTE3 capture requires 56 AIV')
    op = 1 if operation == 'dispatch' else 2
    records, intervals, milestones, cores, waits = [], [], [], [], []
    splits, modes = set(), set()
    for core, rows in enumerate(raw):
        if len(rows) != 8 or any(len(row) != 8 for row in rows):
            raise ValueError('MTE3 timer shape mismatch')
        words = [value for row in rows for value in row]
        if any(type(value) is not int or not 0 <= value < (1 << 64) for value in words):
            raise ValueError('Invalid MTE3 timer word')
        h = words[:8]
        mode, send, role, profile = (h[5] >> shift & 255 for shift in (0, 8, 16, 24))
        if h[:5] != [MAGIC, 16, op, 56, core] or h[5] >> 32 or mode not in MODES or profile != 7 or not 0 < send < 56:
            raise ValueError('Incomplete or mismatched MTE3 header')
        expected_role = (3 if core < send else 2) if op == 1 else (1 if core < send else 4)
        if role != expected_role:
            raise ValueError('Unexpected MTE3 core role')
        send_lanes, recv_lanes = h[7] & 255, (h[7] >> 8) & 255
        overlap_enabled = bool(h[7] & (1 << 16))
        lane_limit = 6 if overlap_enabled else 8
        if h[7] >> 17 or send_lanes > lane_limit or recv_lanes > lane_limit:
            raise ValueError('Invalid MTE3 lane metadata')
        if h[6] >> len(POINTS) or any(words[26:32]):
            raise ValueError('Invalid MTE3 bitmap or reserved words')
        points = {}
        for index, name in enumerate(POINTS):
            if h[6] & (1 << index):
                points[name] = words[8 + index]
            elif words[8 + index]:
                raise ValueError('Timestamp without valid milestone bit')
        if not {'ENTRY', 'INIT_DONE', 'WORK_DONE'} <= points.keys():
            raise ValueError('Missing required MTE3 milestones')
        if not points['ENTRY'] <= points['INIT_DONE'] <= points['WORK_DONE']:
            raise ValueError('Invalid MTE3 timestamp order')
        if any(not points['INIT_DONE'] <= tick <= points['WORK_DONE']
               for name, tick in points.items() if name != 'ENTRY'):
            raise ValueError('Milestone outside same-core work interval')
        # R local sends can interleave prefix/reception. Only causal chains
        # within one activity are ordered; enum order is not temporal order.
        for chain in (('COUNTS_RECEIVED', 'PREFIX_LOCAL_DONE', 'PREFIX_READY'),
                      ('SEND_PREPARED', 'FIRST_SEND', 'SOURCE_SUBMIT_DONE', 'SEND_DONE'),
                      ('PREFIX_WAIT_ENTER', 'PREFIX_WAIT_EXIT'),
                      ('RECV_PREPARED', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY',
                       'RECV_PENDING_EMPTY', 'RECV_DONE')):
            ticks = [points[name] for name in chain if name in points]
            if ticks != sorted(ticks):
                raise ValueError('Invalid MTE3 dependency timestamp order')
        for child, parent in (('PREFIX_WAIT_EXIT', 'PREFIX_WAIT_ENTER'),
                              ('FIRST_SEND', 'SEND_PREPARED'),
                              ('FIRST_RECV_READY', 'FIRST_RECV_PROBE')):
            if child in points and parent not in points:
                raise ValueError('Missing MTE3 milestone dependency: ' + parent)
        if op == 2 and (set(points) != {'ENTRY', 'INIT_DONE', 'WORK_DONE'} or h[7] or any(words[32:])):
            raise ValueError('Combine only records three MTE3 overview milestones')
        if op == 2 and mode != 1:
            raise ValueError('MTE3 intervention requires dispatch')
        common = dict(core=core, role=ROLES[role], initial_role=ROLES[role],
                      timer_profile='mte3', timer_profile_id=7)
        milestones.extend(dict(**common, point=name, tick=tick) for name, tick in points.items())
        for stage, first, last in PAIRS:
            if first not in points or last not in points:
                continue
            begin, end = points[first], points[last]
            if end < begin:
                raise ValueError('Invalid same-core MTE3 interval endpoint order')
            total = end - begin
            row = dict(**common, stage=stage, total_ticks=total, calls=1,
                       average_ticks=total, time_measured=True,
                       interval_kind='same-core milestone span', start_point=first, end_point=last)
            if frequency_hz is not None:
                row.update(total_us=total*1e6/frequency_hz, average_us=total*1e6/frequency_hz)
            records.append(row)
            intervals.append(dict(**common, stage=stage, sample=0, first_tick=begin,
                                  last_tick=end, total_ticks=total, start_point=first,
                                  end_point=last, interval_kind='same-core milestone span'))
        for index, name in enumerate(METRICS):
            total, maximum, calls, control = words[32 + 4*index:36 + 4*index]
            if (not calls and (total or maximum or control)) or total < maximum or total > maximum*calls:
                raise ValueError('Invalid MTE3 wait metric sum/max/calls: ' + name)
            intervention = index >= 4
            permitted = (op == 1 and core < send and
                         ((mode == 4 and index in (4, 5)) or (mode == 5 and index in (6, 7))))
            if intervention and not permitted and (calls or total or maximum or control):
                raise ValueError('MTE3 wait metric outside selected S-core intervention: ' + name)
            if not intervention and control:
                raise ValueError('Original Scalar wait cannot contain MTE3 empty control')
            if op == 2:
                continue
            row = dict(**common, metric=name, sum_ticks=total, max_ticks=maximum,
                       calls=calls, empty_control_ticks=control, time_measured=bool(calls),
                       intervention=intervention,
                       signed_excess_ticks=total-control if intervention else None)
            if frequency_hz is not None:
                row.update(sum_us=total*1e6/frequency_hz, max_us=maximum*1e6/frequency_hz,
                           empty_control_us=control*1e6/frequency_hz,
                           signed_excess_us=(total-control)*1e6/frequency_hz if intervention else None)
            waits.append(row)
        # These eight Scalar scopes do not nest. Each intervention's empty
        # control follows its raw wait, and all scopes lie inside same-core
        # INIT_DONE..WORK_DONE. Milestone spans are intentionally excluded.
        measured_wait_ticks = sum(words[offset] + words[offset+3] for offset in range(32, 64, 4))
        if measured_wait_ticks > points['WORK_DONE'] - points['INIT_DONE']:
            raise ValueError('Disjoint MTE3 wait/control totals exceed same-core work interval')
        cores.append(dict(**common, valid_bitmap=h[6], milestone_count=len(points),
                          send_lanes=send_lanes, recv_lanes=recv_lanes, overlap_enabled=overlap_enabled,
                          measured_wait_ticks=measured_wait_ticks))
        splits.add(send); modes.add(mode)
    if len(splits) != 1 or len(modes) != 1:
        raise ValueError('Mixed MTE3 core splits or modes')
    return dict(schema=16, operation=operation, active_aiv=56, send_cores=send,
                recv_cores=56-send, boundary_mode=MODES[mode], timer_profile='mte3',
                timer_profile_id=7, frequency_hz=frequency_hz, recording='mte3-backpressure',
                points=list(POINTS), records=records, intervals=intervals,
                milestones=milestones, cores=cores, wait_metrics=waits,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed',
                interpretation='Original waits observe actual Scalar blocking. MTE3_S intervention waits include queued work and producer dependencies; immediately repeated empty controls contain no new transfer. Signed differences are preserved, never clipped. Intervention changes overlap and cannot quantify baseline network latency. Same-core milestone spans may overlap and must not be added. Export follows WORK_DONE.')
