"""Execute production Python lifecycle with HCCL/device fault doubles, no NPU."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    tree = ast.parse((ROOT / 'deep_ep/_runtime.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
    topology = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_ndr_topology')
    log = []
    api = SimpleNamespace(current_device=lambda: 0, synchronize=lambda device: log.append(('device_sync', device)))
    dist = ModuleType('torch.distributed')
    def gather(out, packet, group):
        log.append(('hccl', packet[0]))
        out[:] = [packet] * len(out)
    dist.all_gather_object = gather
    torch = ModuleType('torch'); torch.distributed = dist
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    import os
    env = dict(npu_api=lambda: api, nullcontext=nullcontext, _LOCK=threading.Lock(), _OWNER=None, os=os)
    module = ast.Module(body=[ast.ClassDef(name='Runtime', bases=[], keywords=[], body=functions, decorator_list=[]), topology], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<production lifecycle>', 'exec'), env)
    obj = object.__new__(env['Runtime'])
    obj.closed = False; obj.coordination_failed = False; obj.host_coordinated = True
    obj.device = 0; obj.rank = 0; obj.world_size = 32; obj.group = object()
    obj.notify_lock = threading.RLock()
    obj.stream = SimpleNamespace(synchronize=lambda: log.append(('stream_sync',)))
    obj.native = SimpleNamespace(
        barrier=lambda **kw: log.append(('native_barrier', kw)),
        close=lambda **kw: log.append(('native_close', kw)))
    env['require_native'] = lambda _: SimpleNamespace(clear_low_latency_combine_windows=
        lambda native, *shape: log.append(('clear_windows', shape)))
    env['_OWNER'] = obj
    return obj, api, dist, log, env


def test_all_streams_and_all_ranks_before_native(runtime):
    obj, _, _, log, env = runtime
    obj.barrier(); obj.close()
    assert log == [('device_sync', 0), ('hccl', 'barrier'), ('native_barrier', {'host_coordinated': True}),
                   ('device_sync', 0), ('hccl', 'close'), ('native_close', {'host_coordinated': True})]
    assert obj.closed and env['_OWNER'] is None
    obj.close()  # idempotent successful close
    assert len(log) == 6


def test_combine_reuse_has_distinct_phase_and_never_closes(runtime):
    obj, _, _, log, env = runtime
    for _ in range(5):
        obj._host_collective('combine_reuse')
    assert log == [('device_sync', 0), ('hccl', 'combine_reuse'),
                   ('native_barrier', {'host_coordinated': True})] * 5
    assert not obj.closed and env['_OWNER'] is obj


@pytest.mark.parametrize('fault', ['device', 'peer', 'order', 'hccl'])
def test_combine_reuse_failure_blocks_next_call(runtime, fault):
    obj, api, dist, log, env = runtime
    def fail(*a, **kw): raise RuntimeError('injected')
    if fault == 'device': api.synchronize = fail
    if fault == 'hccl': dist.all_gather_object = fail
    if fault in ('peer', 'order'):
        def gather(out, packet, group):
            out[:] = [packet] * len(out)
            out[3] = ('close', None) if fault == 'order' else ('combine_reuse', 'failed')
        dist.all_gather_object = gather
    with pytest.raises(RuntimeError): obj._host_collective('combine_reuse')
    assert obj.coordination_failed and env['_OWNER'] is obj
    with pytest.raises(RuntimeError, match='restart'): obj._host_collective('combine_reuse')
    assert not any(x[0].startswith('native_') for x in log)


@pytest.mark.parametrize('coordinated,fail', [(True, False), (False, False), (True, True)])
def test_real_combine_entry_coordinates_before_producer(runtime, monkeypatch, coordinated, fail):
    obj, api, dist, log, _ = runtime
    obj.host_coordinated = coordinated
    obj.notify_failed = False
    torch = sys.modules['torch']
    class Tensor:
        dtype = 'bf16'
        shape = (2, 3584)
        def is_contiguous(self): return True
    torch.Tensor = Tensor; torch.bfloat16 = 'bf16'; torch.float16 = 'fp16'
    class ReachedProducer(Exception): pass
    def producer():
        log.append(('producer',))
        raise ReachedProducer()
    event = ModuleType('deep_ep.utils.event')
    event.EventHandle = producer; event.EventOverlap = object
    monkeypatch.setitem(sys.modules, 'deep_ep.utils.event', event)
    tree = ast.parse((ROOT / 'deep_ep/buffers/elastic.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ElasticBuffer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_combine_low_latency')
    env = dict(__name__='deep_ep.buffers.elastic', __package__='deep_ep.buffers',
               runtime_of=lambda _: obj, npu_api=lambda: api,
               require_native=lambda _: SimpleNamespace(low_latency_combine_tiling=lambda *a: b''))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<production combine>', 'exec'), env)
    token = object()
    buffer = SimpleNamespace(_handle_token=token)
    handle = SimpleNamespace(_buffer_token=token, metadata_only=False, num_expanded_tokens=2,
                             hidden=3584, num_tokens=1, num_topk=8, num_experts=384, num_max_tokens_per_rank=32)
    if fail:
        def broken(*a, **kw): raise RuntimeError('collective failed')
        dist.all_gather_object = broken
    with pytest.raises(RuntimeError if fail else ReachedProducer):
        env['_combine_low_latency'](buffer, Tensor(), handle, None, None, 0, 0, None, None, False, False)
    if fail:
        assert ('producer',) not in log and obj.coordination_failed
    elif coordinated:
        assert log == [('device_sync', 0), ('clear_windows', (1, 3584, 8, 384)), ('hccl', 'combine_reuse'),
                       ('native_barrier', {'host_coordinated': True}), ('producer',)]
    else:
        assert log == [('producer',)]


def test_window_prepare_before_collective_and_cached_until_dispatch(runtime):
    obj, _, _, log, _ = runtime
    for _ in range(2):
        obj._host_collective('combine_reuse', lambda: obj._prepare_combine_windows(2,3584,8,384))
    assert [x[0] for x in log] == ['device_sync','clear_windows','hccl','native_barrier',
                                  'device_sync','hccl','native_barrier']
    obj.combine_window_layout = None  # invalidation at the next dispatch
    obj._host_collective('combine_reuse', lambda: obj._prepare_combine_windows(2,3584,8,384))
    assert len([x for x in log if x[0]=='clear_windows']) == 2


def test_prepare_failure_is_exchanged_and_blocks_launch(runtime):
    obj, _, _, log, env = runtime
    def fail(): raise RuntimeError('clear failed')
    with pytest.raises(RuntimeError, match='clear failed'):
        obj._host_collective('combine_reuse', fail)
    assert obj.coordination_failed and env['_OWNER'] is obj
    assert [x[0] for x in log] == ['device_sync','hccl']


@pytest.mark.parametrize('operation', ['barrier', 'close'])
@pytest.mark.parametrize('fault', ['device', 'peer', 'order', 'hccl', 'native'])
def test_failure_blocks_reuse_and_retains_owner(runtime, operation, fault):
    obj, api, dist, log, env = runtime
    def fail(*a, **kw): raise RuntimeError('injected failure')
    if fault == 'device': api.synchronize = fail
    if fault == 'native': setattr(obj.native, operation, fail)
    if fault == 'hccl': dist.all_gather_object = fail
    if fault in ('peer', 'order'):
        def gather(out, packet, group):
            out[:] = [packet] * len(out)
            out[17] = (operation, 'peer device failed') if fault == 'peer' else ('different operation', None)
        dist.all_gather_object = gather
    with pytest.raises(RuntimeError): getattr(obj, operation)()
    assert obj.coordination_failed and not obj.closed and env['_OWNER'] is obj
    before = list(log)
    for method in (obj.ensure_alive, obj.barrier, obj.close):
        with pytest.raises(RuntimeError, match='restart'): method()
    assert log == before
    assert not any(x[0].startswith('native_') for x in log)


def topology(size=32):
    return [(f'host-{r//16}', ('0,1,...,15', '', '', ''), r % 16) for r in range(size)]


@pytest.mark.parametrize('size', [16, 32, 64, 128])
def test_valid_physical_layout(runtime, size):
    runtime[-1]['validate_ndr_topology'](topology(size))


@pytest.mark.parametrize('fault', ['missing_host', 'duplicate_device', 'mapping', 'interleaved', 'eight_per_host', 'single_host32'])
def test_invalid_physical_layout(runtime, fault):
    values = topology()
    if fault == 'missing_host': values[0] = ('', values[0][1], 0)
    if fault == 'duplicate_device': values[1] = values[0]
    if fault == 'mapping': values[1] = (values[1][0], ('different',), 1)
    if fault == 'interleaved': values[1], values[17] = values[17], values[1]
    if fault == 'eight_per_host': values = [(f'h{r//8}', (), r % 8) for r in range(32)]
    if fault == 'single_host32': values = [('one', (), r) for r in range(32)]
    with pytest.raises(ValueError): runtime[-1]['validate_ndr_topology'](values)
