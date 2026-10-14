"""Named task profiling, following ep_test_mega_moe_T 844f17aa428d.

Reference: core/operations/mega_moe/performance/profiling.py and
core/profiling/kernels.py. No imports from the reference checkout are needed.
"""
import csv
import math
from pathlib import Path
import re
import statistics

WARMUP = 10
CAPTURE_STEPS = 100
PROFILE_STEPS = 50
DELAY_SHAPE = (118260, 30720)


def describe(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Expected finite nonnegative task durations')
    ordered = sorted(values)
    result = dict(sample_count=len(values), mean_us=statistics.mean(values),
                  min_us=min(values), max_us=max(values))
    for p in (50, 95, 99):
        pos = (len(values)-1)*p/100
        lo, hi = math.floor(pos), math.ceil(pos)
        result[f'p{p}_us'] = ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)
    return result


def read_named_tasks(root, kernel, expected, tail=PROFILE_STEPS):
    """Prefer op_summary, fall back to kernel_details with recorded provenance.

    Keep each exact name/task-type independently. Never sum concurrent tasks
    or mix op_summary and kernel_details copies of the same capture.
    """
    if expected < 1 or tail < 0:
        raise ValueError('Invalid sample window')
    matcher = re.compile(r'(?<![A-Za-z0-9_])'+re.escape(kernel)+r'(?![A-Za-z0-9_])')
    for glob in ('op_summary*.csv', 'kernel_details.csv'):
        groups = {}
        for path in sorted(Path(root).rglob(glob)):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    r = {k.replace(' ', '').lower(): v for k, v in row.items() if k}
                    names = [r.get(k, '') for k in ('kernelname', 'opname', 'optype', 'name')]
                    matches = [name for name in names if matcher.search(name or '')]
                    if not matches:
                        continue
                    name, kind = matches[0], r.get('tasktype', r.get('type', ''))
                    value = float(r.get('taskduration(us)', r.get('duration(us)', 'nan')))
                    wait = float(r.get('taskwaittime(us)') or 0)
                    start = r.get('taskstarttime(us)', r.get('starttime(us)'))
                    stamp = float(start) if start else None
                    if any(not math.isfinite(x) or x < 0 for x in (value, wait)) or (stamp is not None and not math.isfinite(stamp)):
                        raise ValueError(f'Invalid task timing in {path}')
                    groups.setdefault((name, kind), []).append((stamp, value, wait, str(path)))
        if not groups:
            continue
        reports = []
        for (name, kind), rows in sorted(groups.items()):
            if len(rows) != expected:
                raise ValueError(f'{name}/{kind}: expected {expected} profiler records, found {len(rows)}')
            if all(row[0] is not None for row in rows):
                rows.sort(key=lambda row: row[0])
            selected = rows[-min(tail, expected):] if tail else rows
            reports.append(dict(kernel_name=name, task_type=kind,
                kernel_us=[r[1] for r in rows], wait_us=[r[2] for r in rows],
                selected_kernel_us=[r[1] for r in selected],
                duration=describe([r[1] for r in selected]),
                wait=describe([r[2] for r in selected]),
                traces=sorted({r[3] for r in rows}), source_format=glob))
        return reports
    raise ValueError(f'No named kernel {kernel} samples under {root}')


class Capture:
    """One warmup profiler step followed by one active step.

    before_sample runs on Torch's current stream. A native standalone ACL
    stream must synchronize this preparation explicitly (bridge=True).
    """
    def __init__(self, output_dir, rows=DELAY_SHAPE[0], cols=DELAY_SHAPE[1], bridge=False):
        import torch
        import torch.distributed as dist
        import torch_npu
        self.torch, self.dist, self.npu = torch, dist, torch_npu
        self.bridge = bridge
        self.active = False
        self.closed = False
        self.delay = torch.zeros(rows, cols, dtype=torch.float32, device='npu')
        self.distributed = dist.is_initialized()
        self.backend = str(dist.get_backend()) if self.distributed else 'single_rank_identity'
        control_device = 'cpu' if self.backend == 'gloo' else 'npu'
        self.temp = torch.ones(1, dtype=torch.float32, device=control_device)
        self.align()
        self.prof = torch_npu.profiler.profile(
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
            profile_memory=True, with_stack=False, with_flops=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False, data_simplification=False))
        self.prof.__enter__()

    def align(self):
        if self.backend == 'gloo':
            self.torch.npu.synchronize()
        if self.distributed:
            self.dist.all_reduce(self.temp, op=self.dist.ReduceOp.SUM, async_op=False)

    def start_capture(self):
        if self.active:
            raise RuntimeError('Capture already started')
        self.torch.npu.synchronize()
        self.prof.step()
        self.active = True

    def before_sample(self):
        if not self.active:
            raise RuntimeError('Capture not active')
        self.torch.exp(self.delay, out=self.delay)
        self.temp.fill_(1)
        self.align()
        if self.bridge:
            # Native probes own a separate ACL stream, unlike Torch operators.
            self.torch.npu.synchronize()

    def close(self, success=True):
        if self.closed:
            return
        self.closed = True
        if success:
            if self.distributed:
                self.dist.barrier()
            self.torch.npu.synchronize()
            self.prof.step()
        self.prof.__exit__(None, None, None)


def run_profiled(invoke, output_dir, warmup=WARMUP, samples=CAPTURE_STEPS,
                 rows=DELAY_SHAPE[0], cols=DELAY_SHAPE[1]):
    capture = Capture(output_dir, rows, cols)
    try:
        output = None
        for _ in range(warmup):
            output = invoke()
        capture.start_capture()
        for _ in range(samples):
            capture.before_sample()
            output = invoke()
        capture.close()
        return output
    except BaseException:
        capture.close(success=False)
        raise
