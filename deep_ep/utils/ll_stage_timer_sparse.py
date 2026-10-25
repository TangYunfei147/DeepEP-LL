"""Schema15/16 fixed first milestones; schema16 adds dispatch initialization."""
import math

MAGIC = 0x4C4C5431
POINTS_V15 = ('ENTRY', 'INIT_DONE', 'COUNTS_SENT', 'COUNTS_RECEIVED',
          'PREFIX_LOCAL_DONE', 'PREFIX_READY', 'SEND_PREPARED', 'FIRST_SEND',
          'SEND_DONE', 'RECV_PREPARED', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY',
          'RECV_DONE', 'WORK_DONE')
INIT_POINTS = ('TILING_DONE', 'PIPE_READY', 'OBJECT_READY', 'MAGIC_DONE')
POINTS = POINTS_V15 + INIT_POINTS
PAIRS_V15 = (
    ('INIT', 'ENTRY', 'INIT_DONE'), ('KERNEL', 'INIT_DONE', 'WORK_DONE'),
    ('INIT_TO_COUNTS_SENT', 'INIT_DONE', 'COUNTS_SENT'),
    ('INIT_TO_COUNTS_RECEIVED', 'INIT_DONE', 'COUNTS_RECEIVED'),
    ('COUNTS_TO_PREFIX_LOCAL', 'COUNTS_RECEIVED', 'PREFIX_LOCAL_DONE'),
    ('PREFIX_LOCAL_TO_READY', 'PREFIX_LOCAL_DONE', 'PREFIX_READY'),
    ('INIT_TO_SEND_PREPARED', 'INIT_DONE', 'SEND_PREPARED'),
    ('BEFORE_FIRST_SEND', 'INIT_DONE', 'FIRST_SEND'),
    ('FIRST_SEND_TO_SEND_DONE', 'FIRST_SEND', 'SEND_DONE'),
    ('PREFIX_READY_TO_RECV_PREPARED', 'PREFIX_READY', 'RECV_PREPARED'),
    ('BEFORE_FIRST_RECV_POLL', 'INIT_DONE', 'FIRST_RECV_PROBE'),
    ('FIRST_PROBE_TO_FIRST_READY', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY'),
    ('FIRST_READY_TO_RECV_DONE', 'FIRST_RECV_READY', 'RECV_DONE'),
)
INIT_PAIRS = (
    ('ENTRY_TO_TILING', 'ENTRY', 'TILING_DONE'),
    ('TILING_TO_PIPE', 'TILING_DONE', 'PIPE_READY'),
    ('PIPE_TO_OBJECT', 'PIPE_READY', 'OBJECT_READY'),
    ('OBJECT_TO_MAGIC', 'OBJECT_READY', 'MAGIC_DONE'),
    ('MAGIC_TO_INIT', 'MAGIC_DONE', 'INIT_DONE'),
)
PAIRS = PAIRS_V15 + INIT_PAIRS
ROLES = {1: 'send_token', 2: 'cumsum', 3: 'send_and_recv', 4: 'recv'}


def points_for_schema(schema):
    if schema not in (15, 16):
        raise ValueError('Unknown sparse timer schema')
    return POINTS_V15 if schema == 15 else POINTS


def pairs_for_schema(schema):
    points_for_schema(schema)
    return PAIRS_V15 if schema == 15 else PAIRS


def parse_records(raw, operation, frequency_hz=None):
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (isinstance(frequency_hz, bool) or not isinstance(frequency_hz, (int,float)) or not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    if len(raw) != 56:
        raise ValueError('Sparse capture requires 56 AIV')
    schema = raw[0][0][1] if raw[0] and len(raw[0][0]) > 1 else None
    point_names = points_for_schema(schema)
    pairs = pairs_for_schema(schema)
    export_words = (8 + len(point_names) + 7) // 8 * 8
    op = 1 if operation == 'dispatch' else 2
    records, intervals, milestones, cores, splits = [], [], [], [], set()
    dispatch_sender_roles = set()
    for core, rows in enumerate(raw):
        if len(rows) != 8 or any(len(row) != 8 for row in rows):
            raise ValueError('Sparse timer shape mismatch')
        words = [value for row in rows for value in row]
        if any(type(v) is not int or not 0 <= v < (1 << 64) for v in words):
            raise ValueError('Invalid sparse timer word')
        h = words[:8]
        mode, send, role, profile = (h[5] >> shift & 255 for shift in (0, 8, 16, 24))
        if h[:5] != [MAGIC, schema, op, 56, core] or h[5] >> 32 or mode != 1 or profile != 6 or not 0 < send < 56:
            raise ValueError('Incomplete or mismatched sparse header')
        allowed_roles = ({1,3} if core < send else {2}) if op == 1 else ({1} if core < send else {4})
        if role not in allowed_roles:
            raise ValueError('Unexpected sparse core role')
        if op == 1 and core < send:
            dispatch_sender_roles.add(role)
            if len(dispatch_sender_roles) != 1:
                raise ValueError('Mixed sparse dispatch sender roles')
        if h[7] or h[6] >> len(point_names) or any(words[8+len(point_names):export_words]):
            raise ValueError('Invalid sparse bitmap or reserved words')
        points = {}
        for index, name in enumerate(point_names):
            if h[6] & (1 << index):
                points[name] = words[8+index]
            elif words[8+index]:
                raise ValueError('Timestamp without valid milestone bit')
        if not {'ENTRY', 'INIT_DONE', 'WORK_DONE'} <= points.keys():
            raise ValueError('Missing required sparse milestones')
        if schema == 16 and operation == 'dispatch' and not set(INIT_POINTS) <= points.keys():
            raise ValueError('Missing required sparse initialization milestones')
        if op == 1 and role == 1 and any(name in points for name in
                ('COUNTS_RECEIVED','PREFIX_LOCAL_DONE','PREFIX_READY','RECV_PREPARED',
                 'FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE')):
            raise ValueError('Dedicated sparse sender cannot report reception milestones')
        if not points['ENTRY'] <= points['INIT_DONE'] <= points['WORK_DONE']:
            raise ValueError('Invalid sparse timestamp order')
        if any(not points['INIT_DONE'] <= tick <= points['WORK_DONE'] for name, tick in points.items()
               if name not in ('ENTRY', *INIT_POINTS)):
            raise ValueError('Milestone outside same-core work interval')
        initialization = [points[name] for name in ('ENTRY', *INIT_POINTS, 'INIT_DONE') if name in points]
        if initialization != sorted(initialization):
            raise ValueError('Invalid sparse initialization timestamp order')
        # Send and receive may interleave. Validate dependencies within each
        # chain, never require global point-index or cross-core monotonicity.
        for chain in (('COUNTS_RECEIVED', 'PREFIX_LOCAL_DONE', 'PREFIX_READY'),
                      ('SEND_PREPARED', 'FIRST_SEND', 'SEND_DONE'),
                      ('RECV_PREPARED', 'FIRST_RECV_PROBE', 'FIRST_RECV_READY', 'RECV_DONE')):
            ticks = [points[name] for name in chain if name in points]
            if ticks != sorted(ticks):
                raise ValueError('Invalid sparse dependency timestamp order')
        common = dict(core=core, role=ROLES[role], initial_role=ROLES[role],
                      timer_profile='sparse', timer_profile_id=6)
        for name, tick in points.items():
            milestones.append(dict(**common, point=name, tick=tick))
        for stage, first, last in pairs:
            if first not in points or last not in points:
                continue
            begin, end = points[first], points[last]
            if end < begin:
                raise ValueError('Invalid sparse interval endpoint order')
            total = end-begin
            row = dict(**common, stage=stage, total_ticks=total, calls=1,
                       average_ticks=total, time_measured=True,
                       interval_kind='same-core milestone span', start_point=first, end_point=last)
            if frequency_hz is not None:
                row.update(total_us=total*1e6/frequency_hz, average_us=total*1e6/frequency_hz)
            records.append(row)
            intervals.append(dict(**common, stage=stage, sample=0, first_tick=begin,
                                  last_tick=end, total_ticks=total, start_point=first, end_point=last,
                                  interval_kind='same-core milestone span'))
        cores.append(dict(**common, valid_bitmap=h[6], milestone_count=len(points)))
        splits.add(send)
    if len(splits) != 1:
        raise ValueError('Mixed core splits')
    return dict(schema=schema, operation=operation, active_aiv=56, send_cores=send,
                recv_cores=56-send, boundary_mode='observe',
                timer_profile='sparse', timer_profile_id=6, frequency_hz=frequency_hz,
                recording='sparse-milestones', points=list(point_names), records=records,
                intervals=intervals, milestones=milestones, cores=cores,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed',
                interpretation='First occurrence of fixed milestones only. Same-core endpoint spans may overlap; never add them. SEND_DONE marks existing sender retirement and may include shared reception; it is not pure transfer latency. Packet counts and repeated waits are not collected. Export follows WORK_DONE.')
