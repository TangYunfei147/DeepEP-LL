"""Bounded per-occurrence LL timing. Durations are derived offline from events only."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

MAGIC = 0x4C4C5431
SCHEMA, FIELDS = 4, 8
DEFAULT_FREQUENCY_HZ = 1_000_000_000
STAGES = ('header', 'kernel', 'magic', 'count_init', 'count_send', 'send_prepare', 'send_metadata', 'count_receive', 'count_prefix', 'count_publish', 'recv_prepare', 'recv_wait', 'recv_cleanup', 'count_finish', 'combine_prepare', 'combine_notify', 'combine_wait_ready', 'combine_drain', 'combine_final_drain', 'send_classify', 'send_route', 'recv_lane_wait', 'recv_gm_to_ub', 'recv_load_wait', 'recv_metadata', 'recv_output_wait', 'recv_output_write', 'recv_forward', 'send_input_gm_to_ub', 'send_input_wait', 'send_tail', 'send_lane_wait', 'send_mte_write', 'send_rdma_stage', 'send_rdma_body_post', 'send_rdma_body_wait', 'send_rdma_flag_post', 'send_rdma_flag_wait', 'send_rdma_restore', 'combine_return_task', 'combine_reduce_task', 'combine_output_task', 'combine_data_wait', 'combine_wait_result')
TRAFFIC_CLASSES = ('local', 'intrahost_first', 'intrahost_forward', 'interhost_first', 'interhost_forward')
MODES = {1:'observe', 2:'drain', 3:'mega'}
MAX_DYNAMIC_EVENTS = 4096
CAPACITIES = tuple(4 if name == "recv_prepare" else 1 if name in ('kernel', 'magic', 'count_init', 'count_send', 'send_prepare', 'send_metadata', 'count_receive', 'count_prefix', 'count_publish', 'recv_cleanup', 'count_finish', 'combine_prepare', 'combine_notify', 'combine_wait_ready', 'combine_drain', 'combine_final_drain') else MAX_DYNAMIC_EVENTS for name in STAGES[1:])
EVENT_OFFSETS = []
_event_end = 128
for _capacity in CAPACITIES:
    EVENT_OFFSETS.append(_event_end)
    _event_end += _capacity * 3
EVENT_WORDS = (_event_end + 63)//64*64
SLOTS = EVENT_WORDS // FIELDS
_state = threading.local()


def event_layout():
    return dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, stages=list(STAGES), capacities=list(CAPACITIES),event_fields=3)


def compact_device_tensor(cpu_tensor):
    """Extract written slots, rather than serializing the entire zero-filled arena."""
    values=cpu_tensor.view(cpu_tensor.shape[0], EVENT_WORDS)
    cores=[]
    for core in range(cpu_tensor.shape[0]):
        base=values[core,:128].tolist()
        if base[85] or any(base[8+len(CAPACITIES):64]) or any(base[86:]):
            raise ValueError(f'Event overflow/boundary error or reserved storage modified on core {core}: flags={base[85]}')
        counts=base[8:8+len(CAPACITIES)]
        events={}
        for stage,(count,capacity) in enumerate(zip(counts,CAPACITIES),1):
            if count>capacity:
                raise ValueError(f'Event unfinished/capacity exceeded: core={core}, stage={STAGES[stage]}, count={count}, capacity={capacity}')
            if count:
                offset=EVENT_OFFSETS[stage-1]
                events[STAGES[stage]]=values[core,offset:offset+count*3].view(count,3).tolist()
        cores.append(dict(header=base[:8],counts=counts,
                          traffic=[base[n:n+4] for n in range(64,84,4)],flags=base[84],errors=base[85],events=events))
    return dict(schema=SCHEMA,cores=cores)


def parse_records(raw, operation, frequency_hz=None):
    if operation not in ('dispatch','combine'):
        raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz<=0):
        raise ValueError('Counter frequency must be finite and positive')
    if not isinstance(raw,dict) or raw.get('schema')!=SCHEMA or not isinstance(raw.get('cores'),list):
        raise ValueError('This branch requires per-occurrence event schema 4; recollect with its matching wheel')
    cores=len(raw['cores'])
    if not 16<=cores<=72 or cores%2:
        raise ValueError('Expected actual even AIV count in [16,72]')
    records,events,traffic,modes,flags=[],[],[],set(),set()
    def word(value):
        return type(value) is int and 0<=value<1<<63
    for core,entry in enumerate(raw['cores']):
        header=entry.get('header');counts=entry.get('counts');ledger=entry.get('traffic');pairs=entry.get('events')
        if (not isinstance(header,list) or len(header)!=8 or not all(word(value) for value in header)
                or header[:5]!=[MAGIC,SCHEMA,1 if operation=='dispatch' else 2,cores,core]
                or header[5] not in MODES or header[6:]!=[1,len(STAGES)]):
            raise ValueError(f'Incomplete/mismatched event header: core={core}')
        modes.add(header[5])
        if not word(entry.get('errors')) or entry['errors']:
            raise ValueError(f'Event overflow or unmatched boundaries on core {core}; no truncated trace is accepted')
        if not word(entry.get('flags')) or entry['flags'] & ~1:
            raise ValueError('Unknown diagnostic traffic flags')
        flags.add(entry['flags'])
        if not isinstance(counts,list) or len(counts)!=len(CAPACITIES) or not all(word(value) for value in counts):
            raise ValueError('Invalid event counts')
        if not isinstance(pairs,dict) or set(pairs)!={STAGES[n+1] for n,count in enumerate(counts) if count}:
            raise ValueError('Event types differ from actual counts')
        if not isinstance(ledger,list) or len(ledger)!=5 or any(not isinstance(row,list) or len(row)!=4 for row in ledger):
            raise ValueError('Invalid byte counter shape')
        for category,row in zip(TRAFFIC_CLASSES,ledger):
            if not all(word(value) for value in row) or row[2]>row[0] or row[3]>row[1]:
                raise ValueError('Read counters must be a subset of total counters')
            traffic.append(dict(core=core,category=category,bytes=row[0],operations=row[1],
                                read_bytes=row[2],read_operations=row[3],flags=entry['flags']))
        if counts[0]!=1:
            raise ValueError('Exactly one kernel event is required per core')
        for stage,(count,capacity) in enumerate(zip(counts,CAPACITIES),1):
            if count>capacity:
                raise ValueError(f'Event capacity exceeded or unclosed: core={core}, stage={STAGES[stage]}')
            if not count:
                continue
            values=pairs[STAGES[stage]]
            if not isinstance(values,list) or len(values)!=count:
                raise ValueError('Event count mismatch')
            total=0;previous=None
            for index,pair in enumerate(values):
                if (not isinstance(pair,list) or len(pair)!=3 or not all(word(value) for value in pair)
                        or pair[1]<pair[0] or (previous is not None and pair[0]<previous)):
                    raise ValueError('Invalid event timestamps/order')
                first,last,task=pair;previous=last;total+=last-first
                if (stage==1 and task!=0) or (stage!=1 and task==0):
                    raise ValueError("Invalid task identity")
                event=dict(core=core,role='send' if core<cores//2 else 'receive',stage=STAGES[stage],
                           index=index,task_id=task,first_tick=first,last_tick=last,total_ticks=last-first,calls=1)
                if frequency_hz is not None:
                    event['total_us']=(last-first)*1e6/frequency_hz
                events.append(event)
            record=dict(core=core,role='send' if core<cores//2 else 'receive',stage=STAGES[stage],
                        first_tick=values[0][0],last_tick=values[-1][1],total_ticks=total,calls=len({pair[2] for pair in values}),segments=count,average_ticks=total/len({pair[2] for pair in values}))
            if frequency_hz is not None:
                record.update(total_us=total*1e6/frequency_hz,average_us=total*1e6/frequency_hz/len({pair[2] for pair in values}))
            records.append(record)
        work=sorted((row for row in events if row['core']==core and row['stage']!='kernel' and row['total_ticks']>0),key=lambda row:row['first_tick'])
        if any(a['last_tick']>b['first_tick'] for a,b in zip(work,work[1:])):
            raise ValueError('Overlapping execution/wait tasks; nested parent timing is unsupported')
        kernel=pairs['kernel'][0]
        if any(first<kernel[0] or last>kernel[1] for values in pairs.values() for first,last,task in values):
            raise ValueError('Stage event outside its core kernel event')
    if len(modes)!=1 or len(flags)!=1:
        raise ValueError('Mixed event modes or diagnostic flags')
    return dict(schema=SCHEMA,operation=operation,active_aiv=cores,boundary_mode=MODES[next(iter(modes))],
                frequency_hz=frequency_hz,records=records,events=events,traffic=traffic,
                interpretation='All timing statistics are derived from actual per-occurrence events. Execution/wait fragments are exclusive; kernel is a separate duration overview.')


def chrome_trace(saved_records, frequency_hz=None, lane_layout='core'):
    if lane_layout not in ('core','stage'):
        raise ValueError('Expected core or stage lane layout')
    output=[];identities=set()
    for pid,saved in enumerate(saved_records,1):
        frequency=frequency_hz if frequency_hz is not None else saved.get('frequency_hz')
        if frequency is None:
            raise ValueError('Chrome trace requires explicit GetSystemCycle frequency_hz')
        report=parse_records(saved['raw'],saved['operation'],frequency)
        meta=saved.get('metadata',{})
        rank,iteration=meta.get('rank','unknown'),meta.get('iteration',pid-1)
        identity=(meta.get('run_id'),meta.get('case_id'),report['operation'],rank,iteration)
        if identity in identities:
            raise ValueError('Duplicate rank/operation/sample trace')
        identities.add(identity)
        origin=min(row['first_tick'] for row in report['events'])
        output.append(dict(ph='M',name='process_name',pid=pid,tid=0,
                           args={'name':f"{report['operation']} rank {rank} sample {iteration} ({report['boundary_mode']})"}))
        lanes=set()
        for row in sorted(report['events'],key=lambda row:(row['core'],row['first_tick'],-row['last_tick'])):
            tid=10000+row['core'] if row['stage']=='kernel' else (row['core'] if lane_layout=='core' else row['core']*64+STAGES.index(row['stage']))
            if tid not in lanes:
                lanes.add(tid)
                label=(f"Kernel overview AIV {row['core']:02d}" if row['stage']=='kernel' else f"AIV {row['core']:02d} {row['role']}") + (f" / {row['stage']}" if lane_layout=='stage' else '')
                output.append(dict(ph='M',name='thread_name',pid=pid,tid=tid,args={'name':label}))
                output.append(dict(ph='M',name='thread_sort_index',pid=pid,tid=tid,args={'sort_index':tid}))
            output.append(dict(ph='X',cat='LL.event',name=row['stage'],pid=pid,tid=tid,
                               ts=(row['first_tick']-origin)*1e6/frequency,dur=row['total_us'],
                               args=dict(row,rank=rank,iteration=iteration,interval_kind='actual occurrence',
                                         origin_tick=origin,frequency_hz=frequency,boundary_mode=report['boundary_mode'],
                                         clock_alignment='Independent origin per rank/operation/sample')))
    if not identities:
        raise ValueError('No LL stage events')
    return dict(traceEvents=output,metadata={'format':'LL per-occurrence events schema 4','lane_layout':lane_layout,
                                           'interpretation':'Exclusive task execution/wait intervals; per-core kernel duration uses a separate overview lane. No cross-rank clock alignment.'})



class StageTrace:
    def __init__(self, buffer, operation, mode='mega', frequency_hz=DEFAULT_FREQUENCY_HZ,
                 expert_config=None):
        if operation not in ('dispatch', 'combine') or mode not in ('observe', 'drain', 'mega'):
            raise ValueError('Invalid LL stage diagnostic configuration')
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError('Counter frequency must be finite and positive')
        from .._native import require_native
        import torch
        native = require_native('LL stage timer')
        if not getattr(native, 'low_latency_stage_timer_enabled', False):
            raise RuntimeError('Rebuild the full wheel with DEEPEP_LL_STAGE_TIMER=ON')
        schema = native.low_latency_stage_timer_layout()
        if schema != event_layout():
            raise RuntimeError('LL timer Python/native schema mismatch')
        cfg = native.low_latency_core_config()
        if not cfg['runtime_device_queried']:
            raise RuntimeError('Bind the actual NPU before allocating a timer')
        words = cfg['active_aiv'] * EVENT_WORDS
        storage = torch.zeros(words + 63, dtype=torch.int64, device=f"npu:{cfg['device']}")
        # Align the view to 512B even if an allocator returns only 8B alignment.
        offset = (-storage.data_ptr()) % 512 // 8
        self.tensor = storage.narrow(0, offset, words).view(cfg['active_aiv'], SLOTS, FIELDS)
        self.expert_tensor, self.expert_raw, self.expert_config = None, None, expert_config
        if expert_config is not None:
            from . import ll_expert_timer as expert
            rank, world, count = (expert_config[key] for key in ('rank','world','experts'))
            if (operation != 'dispatch' or any(type(v) is not int for v in (rank,world,count)) or
                    not (0 <= rank < world <= count <= 1024 and world >= 16 and world%16==0 and count%world==0)):
                raise ValueError('Expert capture requires a valid dispatch expert configuration')
            if native.low_latency_core_protocol != 5 or not hasattr(native, 'low_latency_expert_timer_layout'):
                raise RuntimeError('Expert diagnostics require the matching ABI 5 wheel on every rank')
            if native.low_latency_expert_timer_layout() != dict(schema=expert.SCHEMA, header_rows=expert.HEADER_ROWS,
                                                               fields=expert.FIELDS, stages=list(expert.STAGES)):
                raise RuntimeError('Expert timer Python/native schema mismatch')
            rows=expert.HEADER_ROWS+count*len(expert.STAGES)
            expert_words=cfg['active_aiv']*rows*expert.FIELDS
            backing=torch.zeros(expert_words+63,dtype=torch.int64,device=f"npu:{cfg['device']}")
            offset=(-backing.data_ptr())%512//8
            self.expert_tensor=backing.narrow(0,offset,expert_words).view(cfg['active_aiv'],rows,expert.FIELDS)
        self.buffer, self.operation, self.mode = buffer, operation, mode
        self.frequency_hz, self.launches, self.raw, self.report = frequency_hz, 0, None, None

    def complete(self):
        if self.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        # Public LL calls establish a current-stream dependency on completion.
        self.raw = compact_device_tensor(self.tensor.cpu())
        self.report = parse_records(self.raw, self.operation, self.frequency_hz)
        self.report.pop("events", None)
        if self.expert_tensor is not None:
            from .ll_expert_timer import parse_records as parse_experts
            self.expert_raw=self.expert_tensor.cpu().tolist()
            report=parse_experts(self.expert_raw, **self.expert_config, frequency_hz=self.frequency_hz)
            if report['boundary_mode'] != self.mode:
                raise RuntimeError('Expert/stage timer mode mismatch')

    def save(self, path, metadata=None):
        if self.report is None:
            raise RuntimeError('Trace is not complete')
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('Resolve trace output from your project root and pass an absolute path')
        path.parent.mkdir(parents=True, exist_ok=True)
        saved=dict(schema=SCHEMA,operation=self.operation,frequency_hz=self.frequency_hz,metadata=metadata or {},raw=self.raw)
        if getattr(self, 'expert_raw', None) is not None:
            saved['expert']=dict(raw=self.expert_raw, **self.expert_config)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(saved, stream, indent=2)
        if self.frequency_hz is not None:
            # Prefix avoids the analyzer's sample_*.json raw-record glob.
            trace_path = path.with_name('trace_' + path.name)
            saved = dict(raw=self.raw, operation=self.operation,
                         frequency_hz=self.frequency_hz, metadata=metadata or {})
            with trace_path.open('x', encoding='utf-8') as stream:
                json.dump(chrome_trace([saved]), stream)


@contextmanager
def capture(buffer, operation, mode='mega', frequency_hz=DEFAULT_FREQUENCY_HZ,
            *, prepared=None, defer_complete=False):
    """Capture exactly one payload LL call; use independently of profiler loops."""
    active = getattr(_state, 'active', {})
    if id(buffer) in active:
        raise RuntimeError('Nested timer contexts for the same buffer are unsupported')
    trace = prepared if prepared is not None else StageTrace(buffer, operation, mode, frequency_hz)
    if prepared is not None and (trace.buffer is not buffer or trace.operation != operation or
                                 trace.mode != mode or trace.frequency_hz != frequency_hz or trace.launches):
        raise ValueError('Prepared trace must be unused and match this capture')
    _state.active = active
    active[id(buffer)] = trace
    try:
        yield trace
        if trace.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        if not defer_complete:
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
    if getattr(trace, 'expert_tensor', None) is not None:
        tensors['ll_expert_timer'] = trace.expert_tensor
    # The native binding derives mode from this CPU scalar, not an NPU read.
    tensors['ll_stage_timer_mode'] = {'observe':1,'drain':2,'mega':3}[trace.mode]
