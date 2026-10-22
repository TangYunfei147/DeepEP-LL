"""Reference-style sampled segments and full metrics; legacy formats are read-only.

Semantics follow graph_dc_debug_seg@e58ece1. Native LL uses a separately validated
512B-aligned per-core carrier; only original schema7 exports dc_seg format-5.
"""
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import struct
import threading

MAGIC = 0x4C4C5431
SCHEMA, SLOTS, FIELDS = 13, 80, 8
TRACE_RECORDS, METRICS, RECORDS = 192, 108, 300
DEFAULT_FREQUENCY_HZ = 1_000_000_000
STAGES = ('KERNEL', 'SEND_TOKENS', 'CAL_CUMSUM', 'PREPARE_INPUT', 'SEND_COUNTS', 'INIT_COUNT_BUFFERS', 'WAIT_RECV_COUNTS', 'CLEAR_RECV_STATUS', 'GATHER_RECV_COUNTS', 'GET_CUMSUM', 'BUILD_RECV_CUMSUM', 'WAIT_CUMSUM', 'LOCAL_WINDOW_COPY', 'FORMAT_OUTPUT', 'BUILD_EXPERT_TASKS', 'CLEAR_LOCAL_FLAGS', 'TOKEN_INPUT', 'TOKEN_QUANT', 'TOKEN_METADATA', 'TOKEN_BUFFER_WAIT', 'TOKEN_PACK', 'TOKEN_SEND_ISSUE', 'COUNT_COMPUTE', 'COUNT_SEND_ISSUE', 'POLL_CORE_SUMS', 'REDUCE_CORE_PREFIX', 'CLEAR_CORE_SUMS', 'BROADCAST_PREFIX', 'PUBLISH_PREFIX_FLAGS', 'RECV_POLL', 'RECV_COPY', 'SEND_DRAIN', 'COMBINE_SEND', 'COMBINE_INIT_RECV', 'COMBINE_RECV', 'COMBINE_INPUT', 'COMBINE_PACK_QUANT', 'COMBINE_SEND_ISSUE', 'COMBINE_SEND_FINISH', 'COMBINE_RECV_POLL', 'COMBINE_DEQUANT_REDUCE', 'COMBINE_PROCESS_TOKEN', 'COMBINE_TOKEN_INIT', 'COMBINE_CLEAR_FLAGS', 'COMBINE_SHARED_EXPERT', 'COMBINE_SPECIAL_EXPERT', 'COMBINE_RMSNORM', 'FINAL_DRAIN', 'POLL_RECV_COUNTS', 'WRITE_COUNTS_OUTPUT', 'INIT', 'POLL_CORE_SUMS_ITERATIONS', 'POLL_RECV_COUNTS_ITERATIONS', 'WAIT_CUMSUM_ITERATIONS', 'RECV_READY_TOKENS', 'COMBINE_READY_SLOTS', 'COMBINE_QUANT', 'COMBINE_BUFFER_WAIT', 'COMBINE_COPY_PACK', 'RECV_BUFFER_WAIT', 'RECV_INPUT', 'RECV_OUTPUT', 'CAL_TOKEN', 'COMBINE_INDEX_INPUT', 'TOKEN_INPUT_WAIT', 'TOKEN_QUANT_WAIT', 'RELAY_ISSUE', 'RELAY_PRE_DRAIN', 'RELAY_COMPLETE', 'RELAY_PACKETS','SEND_COMPLETE_WAIT','TRANSITION_DRAIN','RECV_PREPARE','SEND_SUBMIT_END','OVERLAP_PATH','SERIAL_FALLBACK','SEND_LANES','RECV_LANES','PREFIX_COMPLETE','PREFIX_LOAD','RECV_COMPLETE_WAIT','SEND_SETUP','Q_SEED_SUBMIT','Q_SEED_WAIT','Q_LOCAL_UPDATE','ROUTE_CLASSIFY','DIRECT_PLAN','SEND_RETIRE_WAIT','RX_FLAG_ISSUE','RX_FLAG_WAIT','RX_FLAG_REDUCE','RX_FLAG_DECODE','RX_LANE_REUSE_WAIT','RX_PAYLOAD_READY_WAIT','RX_METADATA','RX_PAYLOAD_OUTPUT','RX_META_OUTPUT','TX_PACK_READY_WAIT','TX_METADATA_PUBLISH','BEFORE_FIRST_SEND','BEFORE_FIRST_RECV_POLL','PREFIX_PUBLISH_WAIT','RX_PENDING_UPDATE','TX_PACKETS','RX_PROBED','RX_READY','RX_EMPTY_POLLS','RX_PACKETS')
COUNTERS = frozenset((*range(51,56),69,*range(74,78),*range(103,108)))
ROLES = {1:'send_token',2:'cumsum',3:'send_and_recv',4:'recv'}
_state=threading.local()
_legacy_spec=importlib.util.spec_from_file_location('_ll_stage_legacy',Path(__file__).with_name('ll_stage_timer_legacy.py'))
legacy=importlib.util.module_from_spec(_legacy_spec);_legacy_spec.loader.exec_module(legacy)
LEGACY_STAGES=legacy.LEGACY_STAGES
ACTIVE=legacy.ACTIVE
NOISY_SEND_ENVELOPES=legacy.NOISY_SEND_ENVELOPES


def parse_records(raw, operation, frequency_hz=None):
    if not raw or not raw[0] or len(raw[0][0])!=8:
        raise ValueError('Missing timer header')
    schema=raw[0][0][1]
    metrics_limit={7:64,9:66,10:70,11:78,12:81}.get(schema,METRICS)
    if schema not in (7,9,10,11,12,SCHEMA):
        return legacy.parse_records(raw,operation,frequency_hz)
    if operation not in ('dispatch','combine'): raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz<=0):
        raise ValueError('Counter frequency must be finite and positive')
    cores=len(raw);op=1 if operation=='dispatch' else 2
    if cores!=56: raise ValueError('Segment capture requires 56 AIV')
    records,intervals,core_info,splits=[],[],[],set()
    modes=set()
    for core,rows in enumerate(raw):
        if len(rows)!=SLOTS or any(len(row)!=FIELDS for row in rows): raise ValueError('Timer shape mismatch')
        words=[value for row in rows for value in row]
        if any(type(v) is not int or not -(1<<63)<=v<(1<<64) for v in words): raise ValueError('Invalid timer word')
        words=[v & ((1<<64)-1) for v in words]
        h=words[:8];mode=h[5]&255;send=(h[5]>>8)&255;role=h[5]>>16
        if h[:5]!=[MAGIC,schema,op,cores,core] or not (mode==1 or (schema in (10,11,12,13) and mode==3 and op==1)) or not 0<send<56:
            raise ValueError('Incomplete or mismatched segment header')
        expected=(3 if core<send else 2) if op==1 else (1 if core<send else 4)
        if role!=expected: raise ValueError('Unexpected segment core role')
        splits.add(send);modes.add(mode);count,omitted=h[6:]
        if not 0<count<=TRACE_RECORDS+metrics_limit or any(words[8+count*2:]): raise ValueError('Truncated or invalid segment records')
        metrics={};pairs={};trace_count=0;in_metrics=False
        for i in range(count):
            packed,value=words[8+i*2:10+i*2];event=packed&0xffffffff;payload=packed>>32
            if event>=0x8000:
                in_metrics=True;stage=event-0x8000
                if stage>=metrics_limit or stage in metrics or not payload: raise ValueError('Duplicate or invalid metric')
                if stage in COUNTERS and value: raise ValueError('Counter contains a duration')
                metrics[stage]=(payload,value)
            else:
                stage=event//2
                if in_metrics or payload or stage>=metrics_limit or stage in COUNTERS: raise ValueError('Invalid interval event')
                pair=pairs.setdefault(stage,[[],[]]);pair[event&1].append(value);trace_count+=1
        if schema in (10,11,12,13):
            relay={stage for stage in metrics if 66<=stage<=69}
            if relay and op!=1: raise ValueError('Relay diagnostics require dispatch')
            if mode==1 and relay & {67,68}: raise ValueError('Completion metrics in observe capture')
            if 66 in relay:
                groups=metrics[66][0]
                if 69 not in relay or metrics[69][0]<groups:
                    raise ValueError('Missing/inconsistent relay write count')
                if mode==3 and (not {67,68}<=relay or metrics[67][0]!=groups or
                               metrics[68][0]!=groups or metrics[68][1]<metrics[66][1]):
                    raise ValueError('Missing/inconsistent relay completion metrics')
            elif relay: raise ValueError('Relay metrics without issue intervals')
        if trace_count>TRACE_RECORDS: raise ValueError('Trace capacity exceeded')
        if metrics.get(0,(0,))[0]!=1 or 0 not in pairs: raise ValueError('Missing KERNEL interval')
        initial_role=ROLES[role]
        # Schema7's header records the initial worker class. Only actual SEND_TOKENS
        # evidence promotes an old cumsum worker to send_and_recv; old captures
        # without local sends and empty local partitions retain their old label.
        display_role='send_and_recv' if op==1 and role==2 and 1 in metrics else initial_role
        recorded=0
        for stage,(begins,ends) in pairs.items():
            if len(begins)!=len(ends) or not 0<len(begins)<=2 or stage not in metrics:
                raise ValueError('Unmatched or excessive sampled intervals')
            calls,total=metrics[stage]
            if len(begins)>calls or sum(e-b for b,e in zip(begins,ends))>total:
                raise ValueError('Sampled intervals exceed accumulated metric')
            for sample,(begin,end) in enumerate(zip(begins,ends)):
                if end<begin: raise ValueError('Invalid timestamp order')
                intervals.append(dict(core=core,role=display_role,initial_role=initial_role,stage=STAGES[stage],sample=sample,
                                      first_tick=begin,last_tick=end,total_ticks=end-begin))
            recorded+=len(begins)
        if len(pairs[0][0])!=1 or metrics[0][1]!=pairs[0][1][0]-pairs[0][0][0]:
            raise ValueError('Invalid KERNEL metric')
        if omitted!=sum(v[0] for stage,v in metrics.items() if stage not in COUNTERS)-recorded:
            raise ValueError('Incorrect omitted interval count')
        for stage,(calls,total) in sorted(metrics.items()):
            item=dict(core=core,role=display_role,initial_role=initial_role,stage=STAGES[stage],total_ticks=total,calls=calls,
                      average_ticks=total/calls,time_measured=stage not in COUNTERS)
            if frequency_hz is not None: item.update(total_us=total*1e6/frequency_hz,average_us=total*1e6/frequency_hz/calls)
            records.append(item)
        core_info.append(dict(core=core,role=display_role,initial_role=initial_role,record_count=count,omitted_intervals=omitted))
    if len(splits)!=1: raise ValueError('Mixed core splits')
    if len(modes)!=1: raise ValueError('Mixed relay diagnostic modes')
    return dict(schema=schema,operation=operation,active_aiv=cores,send_cores=send,recv_cores=cores-send,
                dispatch_recv_cores=cores,boundary_mode='relay-complete' if mode==3 else 'observe',frequency_hz=frequency_hz,
                records=records,intervals=intervals,cores=core_info,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed',
                interpretation='First two calls sampled per stage; all calls accumulated. Nested stages cannot be summed. GM export follows KERNEL and all-core synchronization.')


def chrome_trace(saved_records, frequency_hz=None, lane_layout='core', show_repeated=False):
    saved_records=list(saved_records)
    if not saved_records: raise ValueError('No LL stage records')
    if all(s['raw'][0][0][1] not in (7,9,10,11,12,SCHEMA) for s in saved_records):
        return legacy.chrome_trace(saved_records,frequency_hz,lane_layout,show_repeated)
    if any(s['raw'][0][0][1] not in (7,9,10,11,12,SCHEMA) for s in saved_records): raise ValueError('Mixed segment and legacy traces')
    if len({s['raw'][0][0][1] for s in saved_records})!=1: raise ValueError('Mixed segment schemas')
    if len({s['raw'][0][0][5]&255 for s in saved_records})!=1: raise ValueError('Mixed relay diagnostic modes')
    if lane_layout not in ('core','stage'): raise ValueError('Expected core or stage lane layout')
    events=[];identities=set();info=[]
    for pid,saved in enumerate(saved_records,1):
        frequency=frequency_hz if frequency_hz is not None else saved.get('frequency_hz')
        if frequency is None: raise ValueError('Chrome trace requires explicit frequency_hz')
        report=parse_records(saved['raw'],saved['operation'],frequency);meta=saved.get('metadata',{})
        rank,iteration=meta.get('rank','unknown'),meta.get('iteration',pid-1)
        identity=(meta.get('run_id'),meta.get('case_id'),report['operation'],rank,iteration)
        if identity in identities: raise ValueError('Duplicate rank/operation/sample trace')
        identities.add(identity)
        origin=min(row['first_tick'] for row in report['intervals'])
        lanes=set()
        for row in sorted(report['intervals'],key=lambda row:(row['core'],row['first_tick'],-row['last_tick'])):
            tid=row['core'] if lane_layout=='core' else row['core']*METRICS+STAGES.index(row['stage'])
            if tid not in lanes:
                events.append(dict(ph='M',name='thread_name',pid=pid,tid=tid,args=dict(name=f"AIV {row['core']:02d} {row['role']}"+(f" / {row['stage']}" if lane_layout=='stage' else ''))))
                lanes.add(tid)
            events.append(dict(ph='X',name=row['stage'],cat=report['operation'],pid=pid,tid=tid,
                               ts=(row['first_tick']-origin)*1e6/frequency,dur=row['total_ticks']*1e6/frequency,
                               args=dict(rank=rank,iteration=iteration,sample=row['sample'],raw_begin_cycle=row['first_tick'],
                                         raw_end_cycle=row['last_tick'],interval_kind='sampled actual interval')))
        info.append(dict(operation=report['operation'],rank=rank,iteration=iteration,cores=report['cores']))
    return dict(traceEvents=events,metadata=dict(format='dc_seg sampled intervals',sampled_calls_per_stage=2,
                all_calls_in_metrics=True,cores=info,clock='Rank-local origins; no cross-rank alignment'))


def reference_binary(raw,operation):
    """Export the reference format-5 body (no opt-in cookie); original parser can read it."""
    report=parse_records(raw,operation)
    if report['schema']!=7: raise ValueError('Reference format-5 export requires original schema7; extended schemas use JSON/CSV/Chrome trace')
    offset=64+72*64;data=bytearray(offset+72*256*16)
    struct.pack_into('<8I',data,0,0x50524f46,0x10005,len(raw),16,256,64,offset,1 if operation=='dispatch' else 2)
    for core,rows in enumerate(raw):
        words=[v for row in rows for v in row];count=words[6]
        struct.pack_into('<3I',data,64+core*64,count,words[7],words[5]>>16)
        for i in range(count): struct.pack_into('<QQ',data,offset+core*4096+i*16,*(v & ((1<<64)-1) for v in words[8+i*2:10+i*2]))
    return bytes(data)


class StageTrace:
    def __init__(self, buffer, operation, mode='observe', frequency_hz=DEFAULT_FREQUENCY_HZ,
                 expert_config=None):
        if operation not in ('dispatch', 'combine') or mode not in ('observe','relay-complete') or (mode=='relay-complete' and operation!='dispatch'):
            raise ValueError('Invalid LL stage diagnostic configuration')
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError('Counter frequency must be finite and positive')
        if expert_config is not None:
            raise ValueError('Reference segment capture does not support legacy per-expert GM instrumentation')
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
        if expert_config is not None:
            raise ValueError('Reference segment capture does not support legacy per-expert GM instrumentation')
        self.expert_tensor, self.expert_raw, self.expert_config = None, None, None
        self.buffer, self.operation, self.mode = buffer, operation, mode
        self.frequency_hz, self.launches, self.raw, self.report = frequency_hz, 0, None, None

    def complete(self):
        if self.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        # Public LL calls establish a current-stream dependency on completion.
        self.raw = self.tensor.cpu().tolist()
        self.report = parse_records(self.raw, self.operation, self.frequency_hz)
        if self.report['boundary_mode']!=self.mode:
            raise RuntimeError('Capture returned a different diagnostic mode')
        if self.report['schema'] != SCHEMA:
            raise RuntimeError('Capture returned a stale timer format; rebuild the complete wheel')
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
        saved=dict(**self.report, metadata=metadata or {}, raw=self.raw)
        if getattr(self, 'expert_raw', None) is not None:
            saved['expert']=dict(raw=self.expert_raw, **self.expert_config)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(saved, stream, indent=2)
        if self.report['schema'] == 7:
            rank=(metadata or {}).get('rank',0)
            binary=path.with_name(f'{self.operation}_prof_rank{rank}_{path.stem}.bin')
            with binary.open('xb') as stream: stream.write(reference_binary(self.raw,self.operation))
        if self.frequency_hz is not None:
            # Prefix avoids the analyzer's sample_*.json raw-record glob.
            trace_path = path.with_name('trace_' + path.name)
            saved = dict(raw=self.raw, operation=self.operation,
                         frequency_hz=self.frequency_hz, metadata=metadata or {})
            with trace_path.open('x', encoding='utf-8') as stream:
                json.dump(chrome_trace([saved]), stream)


@contextmanager
def capture(buffer, operation, mode='observe', frequency_hz=DEFAULT_FREQUENCY_HZ,
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
    tensors['ll_stage_timer_mode'] = {'observe':1,'relay-complete':3}[trace.mode]
