"""Optional, separate LL device-stage diagnostics. No Torch import for parsing.

Raw values are GetSystemCycle ticks. Supply that counter's frequency explicitly
to convert to microseconds; do not substitute the AIV compute clock.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

MAGIC = 0x4C4C5431
SCHEMA = 6
DEFAULT_FREQUENCY_HZ = 1_000_000_000  # Target-confirmed: time_us = ticks / 1000.
SLOTS, FIELDS = 32, 8
STAGES = ('header', 'kernel', 'magic', 'worker_init', 'count_init', 'count_send', 'sender',
          'send_prepare', 'send_payload', 'send_metadata', 'count_receive', 'count_prefix',
          'count_publish', 'receiver', 'recv_prepare', 'recv_wait', 'recv_consume', 'recv_cleanup',
          'count_finish', 'combine_prepare', 'combine_notify', 'combine_wait_ready',
          'combine_direct', 'combine_partial', 'combine_aggregate', 'combine_drain', 'combine_final_drain',
          'send_classify', 'send_pack', 'send_issue')
LEGACY_STAGES = STAGES
STAGES += ('recv_lane_wait', 'recv_gm_to_ub', 'recv_load_wait', 'recv_metadata',
           'recv_output_wait', 'recv_output_write', 'recv_forward',
           'send_input_gm_to_ub', 'send_input_wait', 'send_tail', 'send_lane_wait',
           'send_mte_write', 'send_rdma_stage', 'send_rdma_body_post', 'send_rdma_body_wait',
           'send_rdma_flag_post', 'send_rdma_flag_wait', 'send_rdma_restore')
ACTIVE = 1 << 62
NOISY_SEND_ENVELOPES = frozenset(('send_input_gm_to_ub', 'send_input_wait', 'send_tail',
                                 'send_lane_wait', 'send_issue', 'send_mte_write'))
_state = threading.local()


def parse_records(raw, operation, frequency_hz=None):
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    cores = len(raw)
    if not 16 <= cores <= 72 or cores % 2:
        raise ValueError('Expected actual even AIV count in [16,72]')
    op = 1 if operation == 'dispatch' else 2
    records, modes, schemas, splits = [], set(), set(), set()
    for core, slots in enumerate(raw):
        if len(slots) != SLOTS or any(len(row) != FIELDS for row in slots):
            raise ValueError('Timer shape mismatch')
        header = slots[0]
        schema = header[1]
        if schema not in (1, 2, 5, 6):
            raise ValueError('Unsupported timer schema')
        names = LEGACY_STAGES if schema == 1 else STAGES
        schemas.add(schema)
        if any(type(value) is not int or value < 0 for value in header):
            raise ValueError("Invalid timer header word")
        if header[:5] != [MAGIC, schema, op, cores, core] or header[6:] != [1, len(names)]:
            raise ValueError(f'Incomplete or mismatched timer header on core {core}')
        mode = header[5] & 255 if schema in (5, 6) else header[5]
        send_cores = header[5] >> 8 if schema in (5, 6) else cores//2
        if schema in (5, 6) and (cores != 56 or not 0 < send_cores < 56):
            raise ValueError('Invalid configured 56-core split')
        splits.add(send_cores)
        if mode not in (1, 2):
            raise ValueError('Unknown boundary mode')
        modes.add(mode)
        if schema in (2, 5, 6):
            words = [value for row in slots for value in row]
            if any(type(value) is not int or not 0 <= value < (1 << 63) for value in words):
                raise ValueError('Invalid raw timer word')
            normalized = [header]
            for offset in range(8, SLOTS * FIELDS, 4):
                first, last, total, count = words[offset:offset+4]
                normalized.append([first, last, total, count & ~ACTIVE, first, int(bool(count & ACTIVE)), 0, 0])
            slots = normalized
        if slots[1][3] != 1:
            raise ValueError('Exactly one completed kernel interval is required')
        for stage, row in enumerate(slots[1:], 1):
            if any(type(value) is not int or value < 0 for value in row):
                raise ValueError('Invalid raw timer word')
            if stage >= len(names):
                if any(row):
                    raise ValueError('Unknown stage data')
                continue
            first, last, total, count, active_start, active, reserved0, reserved1 = row
            if active or reserved0 or reserved1:
                raise ValueError(f'Unclosed or invalid stage {STAGES[stage]} on core {core}')
            if not count:
                if any(row):
                    raise ValueError('Unfinished stage')
                continue
            if last < first or active_start < first or active_start > last or total > last-first:
                raise ValueError('Invalid timestamp order or accumulated interval')
            role = 'send' if core < send_cores else 'receive'
            if schema == 6 and operation == 'dispatch' and core < send_cores:
                role = ('receive' if names[stage].startswith('recv_') or names[stage] in ('receiver','count_publish')
                        else 'send' if names[stage].startswith('send_') or names[stage]=='sender' else 'send_receive')
            record = dict(core=core, role=role,
                          stage=names[stage], first_tick=first, last_tick=last,
                          total_ticks=total, calls=count, average_ticks=total/count)
            if frequency_hz is not None:
                record['total_us'] = total*1e6/frequency_hz
                record['average_us'] = total*1e6/frequency_hz/count
            records.append(record)
    if len(modes) != 1:
        raise ValueError('Mixed boundary modes')
    if len(splits) != 1:
        raise ValueError('Mixed core splits')
    if len(schemas) != 1:
        raise ValueError('Mixed timer schemas')
    return dict(schema=schema, operation=operation, active_aiv=cores, send_cores=send_cores, recv_cores=cores-send_cores,
                dispatch_recv_cores=cores if schema == 6 else cores-send_cores,
                boundary_mode='observe' if modes == {1} else 'drain', frequency_hz=frequency_hz,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed', records=records,
                interpretation='Nested and overlapping stages must not be summed; repeated-stage totals exclude gaps.')


def chrome_trace(saved_records, frequency_hz=None, lane_layout='core', show_repeated=False):
    """Export rank/sample-local timelines; schema 1 cannot reconstruct repeated calls."""
    if lane_layout not in ('core', 'stage'):
        raise ValueError('Expected core or stage lane layout')
    events, identities = [], set()
    for pid, saved in enumerate(saved_records, 1):
        frequency = frequency_hz if frequency_hz is not None else saved.get('frequency_hz')
        if frequency is None:
            raise ValueError('Chrome trace requires explicit GetSystemCycle frequency_hz')
        report = parse_records(saved['raw'], saved['operation'], frequency)
        meta = saved.get('metadata', {})
        rank, iteration = meta.get('rank', 'unknown'), meta.get('iteration', pid - 1)
        identity = (meta.get('run_id'), meta.get('case_id'), report['operation'], rank, iteration)
        if identity in identities:
            raise ValueError('Duplicate rank/operation/sample trace')
        identities.add(identity)
        origin = min(row['first_tick'] for row in report['records'])
        name = f"{report['operation']} rank {rank} sample {iteration} ({report['boundary_mode']}; independent origin)"
        events.append(dict(ph='M', name='process_name', pid=pid, tid=0, args={'name': name}))
        lanes = set()
        # Parents first at equal starts, so viewers can nest complete events.
        records = sorted(report['records'], key=lambda row: (row['core'], row['first_tick'], -row['last_tick']))
        for row in records:
            if not show_repeated and row['calls'] > 1 and row['stage'] in NOISY_SEND_ENVELOPES:
                continue
            tid = row['core'] if lane_layout == 'core' else row['core'] * 64 + STAGES.index(row['stage'])
            lane = f"AIV {row['core']:02d} {row['role']}"
            if lane_layout == 'stage':
                lane += f" / {row['stage']}"
            if tid not in lanes:
                events.append(dict(ph='M', name='thread_name', pid=pid, tid=tid, args={'name': lane}))
                events.append(dict(ph='M', name='thread_sort_index', pid=pid, tid=tid, args={'sort_index': tid}))
                lanes.add(tid)
            repeated = row['calls'] > 1
            span = row['last_tick'] - row['first_tick']
            args = dict(row, rank=rank, iteration=iteration, operation=report['operation'],
                        sampling=meta.get('sampling'),
                        boundary_mode=report['boundary_mode'], frequency_hz=frequency,
                        origin_tick=origin, span_ticks=span, gap_ticks=span-row['total_ticks'],
                        interval_kind='repeated envelope' if repeated else 'single interval',
                        clock_alignment='Independent origin per rank/operation/sample; not profiler aligned')
            events.append(dict(ph='X', cat='LL.stage.envelope' if repeated else 'LL.stage',
                               name=row['stage'] + (f" [envelope x{row['calls']}]" if repeated else ''),
                               pid=pid, tid=tid, ts=(row['first_tick']-origin)*1e6/frequency,
                               dur=span*1e6/frequency, args=args))
    if not identities:
        raise ValueError('No LL stage records')
    # Chrome Trace Event ts/dur are microseconds. displayTimeUnit is a viewer
    # preference with a different enum; Chrome tracing rejects 'us'. Omit it.
    return dict(traceEvents=events,
                metadata={'format': 'LL stage Chrome trace (schemas 1/2)', 'lane_layout': lane_layout,
                          'hidden_repeated_send_envelopes': not show_repeated,
                          'interpretation': 'Repeated events show envelopes including gaps; args.total_us is accumulated active time. Independent rank/sample origins; no cross-rank or profiler clock alignment.'})

