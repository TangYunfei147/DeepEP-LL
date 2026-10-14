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
SCHEMA = 1
SLOTS, FIELDS = 32, 8
STAGES = ('header', 'kernel', 'magic', 'worker_init', 'count_init', 'count_send', 'sender',
          'send_prepare', 'send_payload', 'send_metadata', 'count_receive', 'count_prefix',
          'count_publish', 'receiver', 'recv_prepare', 'recv_wait', 'recv_consume', 'recv_cleanup',
          'count_finish', 'combine_prepare', 'combine_notify', 'combine_wait_ready',
          'combine_direct', 'combine_partial', 'combine_aggregate', 'combine_drain', 'combine_final_drain',
          'send_classify', 'send_pack', 'send_issue')
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
    records, modes = [], set()
    for core, slots in enumerate(raw):
        if len(slots) != SLOTS or any(len(row) != FIELDS for row in slots):
            raise ValueError('Timer shape mismatch')
        header = slots[0]
        if any(type(value) is not int or value < 0 for value in header):
            raise ValueError("Invalid timer header word")
        if header[:5] != [MAGIC, SCHEMA, op, cores, core] or header[6:] != [1, len(STAGES)]:
            raise ValueError(f'Incomplete or mismatched timer header on core {core}')
        if header[5] not in (1, 2):
            raise ValueError('Unknown boundary mode')
        modes.add(header[5])
        if slots[1][3] != 1:
            raise ValueError('Exactly one completed kernel interval is required')
        for stage, row in enumerate(slots[1:], 1):
            if any(type(value) is not int or value < 0 for value in row):
                raise ValueError('Invalid raw timer word')
            if stage >= len(STAGES):
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
            record = dict(core=core, role='send' if core < cores//2 else 'receive',
                          stage=STAGES[stage], first_tick=first, last_tick=last,
                          total_ticks=total, calls=count, average_ticks=total/count)
            if frequency_hz is not None:
                record['total_us'] = total*1e6/frequency_hz
                record['average_us'] = total*1e6/frequency_hz/count
            records.append(record)
    if len(modes) != 1:
        raise ValueError('Mixed boundary modes')
    return dict(schema=SCHEMA, operation=operation, active_aiv=cores,
                boundary_mode='observe' if modes == {1} else 'drain', frequency_hz=frequency_hz,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed', records=records,
                interpretation='Nested and overlapping stages must not be summed; repeated-stage totals exclude gaps.')


class StageTrace:
    def __init__(self, buffer, operation, mode='observe', frequency_hz=None):
        if operation not in ('dispatch', 'combine') or mode not in ('observe', 'drain'):
            raise ValueError('Invalid LL stage diagnostic configuration')
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError('Counter frequency must be finite and positive')
        from .._native import require_native
        import torch
        native = require_native('LL stage timer')
        if not getattr(native, 'low_latency_stage_timer_enabled', False):
            raise RuntimeError('Rebuild the full wheel with DEEPEP_LL_STAGE_TIMER=ON')
        schema = native.low_latency_stage_timer_layout()
        if schema != dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, stages=list(STAGES)):
            raise RuntimeError('LL timer Python/native schema mismatch')
        cfg = native.low_latency_core_config()
        if not cfg['runtime_device_queried']:
            raise RuntimeError('Bind the actual NPU before allocating a timer')
        words = cfg['active_aiv'] * SLOTS * FIELDS
        storage = torch.zeros(words + 63, dtype=torch.int64, device=f"npu:{cfg['device']}")
        # Align the view to 512B even if an allocator returns only 8B alignment.
        offset = (-storage.data_ptr()) % 512 // 8
        self.tensor = storage.narrow(0, offset, words).view(cfg['active_aiv'], SLOTS, FIELDS)
        self.buffer, self.operation, self.mode = buffer, operation, mode
        self.frequency_hz, self.launches, self.raw, self.report = frequency_hz, 0, None, None

    def complete(self):
        if self.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        # Public LL calls establish a current-stream dependency on completion.
        self.raw = self.tensor.cpu().tolist()
        self.report = parse_records(self.raw, self.operation, self.frequency_hz)

    def save(self, path, metadata=None):
        if self.report is None:
            raise RuntimeError('Trace is not complete')
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('Resolve trace output from your project root and pass an absolute path')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(**self.report, metadata=metadata or {}, raw=self.raw), stream, indent=2)


@contextmanager
def capture(buffer, operation, mode='observe', frequency_hz=None):
    """Capture exactly one payload LL call; use independently of profiler loops."""
    active = getattr(_state, 'active', {})
    if id(buffer) in active:
        raise RuntimeError('Nested timer contexts for the same buffer are unsupported')
    trace = StageTrace(buffer, operation, mode, frequency_hz)
    _state.active = active
    active[id(buffer)] = trace
    try:
        yield trace
        trace.complete()
    finally:
        active.pop(id(buffer), None)


def attach(buffer, tensors, operation):
    """Internal LL-only hook; tensor is retained/record_stream'ed by its caller."""
    trace = getattr(_state, 'active', {}).get(id(buffer))
    if trace is None:
        return
    if trace.operation != operation or trace.launches:
        raise RuntimeError('Wrong operation or multiple launches in timer context')
    if trace.tensor.device != tensors['input'].device:
        raise ValueError('Timer and LL input must be on the same NPU')
    trace.launches += 1
    tensors['ll_stage_timer'] = trace.tensor
    # The native binding derives mode from this CPU scalar, not an NPU read.
    tensors['ll_stage_timer_mode'] = 1 if trace.mode == 'observe' else 2
