"""Public LL-only construction reaches MTE without UDMA or HT fallback.

These tests run the real ElasticBuffer and Python Runtime against small native
factory doubles. No Torch installation, device allocation or NPU is required.
The separate Session lifecycle tests cover the actual C++ initialization path.
"""
import importlib
import inspect
import os
import sys
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext

import pytest

import deep_ep
from deep_ep import _runtime


@pytest.fixture
def backend(monkeypatch):
    elastic = importlib.import_module(deep_ep.ElasticBuffer.__module__)
    calls = []
    behavior = SimpleNamespace(missing_sparse=False)

    def resource(kind):
        return SimpleNamespace(
            closed=False, udma_enabled=kind == 'ht', low_latency_backend='mte' if kind == 'll' else None,
            barrier=lambda: calls.append(('barrier', kind)), close=lambda: calls.append(('close', kind)))

    def ht(*args):
        calls.append(('ht', args))
        if behavior.missing_sparse and len(args) > 5 and args[5] and os.environ.get('DEEPEP_SHMEM_DENSE_QPS', '0') != '1':
            raise RuntimeError('SHMEM lacks legacy HT sparse-QP policy')
        return resource('ht')

    def ll(*args):
        calls.append(('ll', args))
        return resource('ll')

    native = SimpleNamespace(Runtime=ht, create_low_latency_runtime=ll, runtime_busy=lambda: False)
    stream = SimpleNamespace(synchronize=lambda: calls.append(('synchronize',)))
    npu = SimpleNamespace(current_device=lambda: 0, Stream=lambda **kw: stream,
                          current_stream=lambda *args: stream)
    monkeypatch.setattr(_runtime, '_OWNER', None)
    monkeypatch.setattr(_runtime, 'require_native', lambda operation: native)
    monkeypatch.setattr(_runtime, 'npu_api', lambda: npu)
    monkeypatch.setattr(elastic, 'require_native', lambda operation: native)
    monkeypatch.setattr(elastic, 'npu_api', lambda: npu)
    monkeypatch.setenv('DEEPEP_SHMEM_ENDPOINT', 'tcp://127.0.0.1:19091')
    monkeypatch.delenv('DEEPEP_SHMEM_DENSE_QPS', raising=False)
    monkeypatch.delenv('DEEPEP_LL_BACKEND', raising=False)

    class Tensor:
        ndim = 2
        shape = (256, 3584)

    torch = ModuleType('torch')
    torch.Tensor = Tensor
    monkeypatch.setitem(sys.modules, 'torch', torch)
    yield SimpleNamespace(calls=calls, native=native, behavior=behavior, Tensor=Tensor, elastic=elastic)
    owner = _runtime._OWNER
    if owner is not None and not owner.closed:
        owner.close()


def construct(**options):
    values = dict(rank=0, world_size=16, num_bytes=8 << 30, num_max_tokens_per_rank=256,
                  explicitly_destroy=True)
    values.update(options)
    return deep_ep.ElasticBuffer(**values)


@pytest.mark.parametrize('huge', [True, False])
def test_public_ll_mte_reaches_dedicated_factory_and_preserves_pages(backend, huge):
    backend.behavior.missing_sparse = True
    buffer = construct(low_latency_backend='mte', use_1g_huge_pages=huge)
    assert [kind for kind, *_ in backend.calls if kind in ('ll', 'ht')] == ['ll']
    call = next(args for kind, args in backend.calls if kind == 'll')
    assert call == (0, 16, 8 << 30, 'tcp://127.0.0.1:19091', 300, huge)
    assert buffer._runtime.use_1g_huge_pages is huge
    assert buffer._runtime.low_latency_backend == 'mte'
    assert ('barrier', 'll') in backend.calls
    assert os.environ.get('DEEPEP_SHMEM_DENSE_QPS') is None
    buffer.destroy()
    assert ('close', 'll') in backend.calls and _runtime._OWNER is None


def test_ll_mte_defaults_keep_existing_huge_page_policy(backend):
    buffer = construct(low_latency_backend='mte')
    call = next(args for kind, args in backend.calls if kind == 'll')
    assert call[-1] is True
    buffer.destroy()


def test_default_elastic_keeps_ht_and_ignores_ll_environment(backend, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_BACKEND', 'mte')
    buffer = construct()
    assert not any(call[0] == 'll' for call in backend.calls)
    ht = next(call[1] for call in backend.calls if call[0] == 'ht')
    assert ht[-2:] == (True, True)
    buffer.destroy()


def test_ll_only_build_selects_its_fixed_runtime_without_shape_inference(backend, monkeypatch):
    backend.native.high_throughput_enabled = False
    backend.behavior.missing_sparse = True
    monkeypatch.setenv('DEEPEP_LL_BACKEND', 'udma')
    buffer = construct()
    assert [call[0] for call in backend.calls if call[0] in ('ll', 'ht')] == ['ll']
    assert buffer._runtime.low_latency_backend == 'mte'
    assert buffer._runtime.use_1g_huge_pages is True
    assert os.environ.get('DEEPEP_SHMEM_DENSE_QPS') is None
    buffer.destroy()


def test_missing_sparse_still_fails_for_default_ht_without_dense_fallback(backend, monkeypatch):
    backend.behavior.missing_sparse = True
    monkeypatch.setenv('DEEPEP_LL_BACKEND', 'mte')
    with pytest.raises(RuntimeError, match='sparse'):
        construct()
    assert any(call[0] == 'ht' for call in backend.calls)
    assert not any(call[0] == 'll' for call in backend.calls)
    assert _runtime._OWNER is None
    assert os.environ.get('DEEPEP_SHMEM_DENSE_QPS') is None


def test_old_wheel_missing_ll_factory_fails_before_generic_runtime(backend):
    del backend.native.create_low_latency_runtime
    with pytest.raises((RuntimeError, NotImplementedError), match='(?i)rebuild|install|low.latency|wheel'):
        construct(low_latency_backend='mte')
    assert backend.calls == [] and _runtime._OWNER is None


@pytest.mark.parametrize('mode', ['rdma', 'udma', 'auto', '', True, 1])
def test_invalid_ll_backend_is_not_guessed(backend, mode):
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        construct(low_latency_backend=mode)
    assert backend.calls == []


@pytest.mark.parametrize('option', [dict(combine_only=True), dict(bootstrap='mpi'), dict(num_allocated_qps=6)])
def test_ll_only_rejects_ht_runtime_options_before_acquisition(backend, option):
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        construct(low_latency_backend='mte', **option)
    assert backend.calls == []


@pytest.mark.parametrize('huge', [None, 0, 1, 'true'])
def test_ll_mte_page_mode_requires_bool(backend, huge):
    with pytest.raises((ValueError, TypeError)):
        construct(low_latency_backend='mte', use_1g_huge_pages=huge)
    assert backend.calls == []


def test_supported_ll_dispatch_never_calls_ht(backend, monkeypatch):
    buffer = construct(low_latency_backend='mte')
    sentinel, captured = object(), []
    monkeypatch.setattr(buffer, '_low_latency_plan', lambda *args: {'valid': True})
    monkeypatch.setattr(buffer, '_dispatch_low_latency', lambda *args: captured.append(args) or sentinel)
    monkeypatch.setattr(buffer, '_notify', lambda *args, **kwargs: pytest.fail('HT notify must not run'))
    assert buffer.dispatch(backend.Tensor(), topk_idx=object(), num_experts=384) is sentinel
    assert len(captured) == 1
    assert not any(call[0] == 'ht' for call in backend.calls)
    buffer.destroy()


@pytest.mark.parametrize('reason', ['shape', 'options', 'cached_ht', 'cached_ll'])
def test_ll_only_dispatch_rejects_fallback_before_ht_work(backend, monkeypatch, reason):
    buffer = construct(low_latency_backend='mte')
    monkeypatch.setattr(buffer, '_low_latency_plan', lambda *args: None if reason == 'shape' else {'valid': True})
    monkeypatch.setattr(buffer, '_dispatch_low_latency', lambda *args: pytest.fail('Unsupported request must not launch LL'))
    monkeypatch.setattr(buffer, '_notify', lambda *args, **kwargs: pytest.fail('HT notify must not run'))
    options = dict(topk_idx=object(), num_experts=384)
    if reason == 'options':
        options['num_qps'] = 2
    elif reason.startswith('cached_'):
        handle = object.__new__(backend.elastic.EPHandle)
        handle.is_low_latency = reason == 'cached_ll'
        options['handle'] = handle
    with pytest.raises((ValueError, NotImplementedError), match='(?i)LL|low.latency|cached|MTE'):
        buffer.dispatch(backend.Tensor(), **options)
    assert not any(call[0] == 'ht' for call in backend.calls)
    buffer.destroy()


def test_ll_only_combine_rejects_ht_handle(backend, monkeypatch):
    buffer = construct(low_latency_backend='mte')
    handle = object.__new__(backend.elastic.EPHandle)
    handle.is_low_latency = False
    monkeypatch.setattr(buffer, '_combine_low_latency', lambda *args: pytest.fail('HT handle must not launch LL'))
    with pytest.raises((ValueError, NotImplementedError), match='(?i)LL|low.latency|MTE'):
        buffer.combine(backend.Tensor(), handle)
    assert not any(call[0] == 'ht' for call in backend.calls)
    buffer.destroy()


def test_ll_handle_combine_preserves_existing_low_latency_path(backend, monkeypatch):
    buffer = construct(low_latency_backend='mte')
    handle = object.__new__(backend.elastic.EPHandle)
    handle.is_low_latency = True
    sentinel = object()
    monkeypatch.setattr(buffer, '_combine_low_latency', lambda *args: sentinel)
    assert buffer.combine(backend.Tensor(), handle) is sentinel
    buffer.destroy()


def test_ll_only_private_notify_rejects_ht_without_allocating(backend):
    buffer = construct(low_latency_backend='mte')
    arguments = {name: None for name, parameter in inspect.signature(buffer._notify).parameters.items()
                 if parameter.default is inspect.Parameter.empty}
    with pytest.raises(ValueError, match='(?i)LL.only|HT notify'):
        buffer._notify(**arguments)
    assert not any(call[0] == 'ht' for call in backend.calls)
    buffer.destroy()


def test_runtime_cannot_mix_explicit_ll_with_udma(backend):
    with pytest.raises(ValueError, match='udma=False'):
        _runtime.Runtime(None, 8 << 30, udma=True, low_latency_backend='mte', rank=0, world_size=16)
    assert backend.calls == []


def test_runtime_rejects_rank_disagreement_on_explicit_ll_mode(backend, monkeypatch):
    class ProcessGroup:
        pass

    class Array:
        def __init__(self, values):
            self.values = list(values)

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    def gather(peers, local, group):
        assert local.values[-1] == 32 + 8  # explicit LL and1GiB pages; no UDMA bit.
        for peer in peers:
            peer.values = list(local.values)
        peers[1].values[-1] &= ~32

    torch = sys.modules['torch']
    torch.int64 = 'int64'
    torch.inference_mode = lambda *args: nullcontext()
    torch.no_grad = lambda: nullcontext()
    torch.tensor = lambda values, **kwargs: Array(values)
    torch.empty_like = lambda value: Array([0] * len(value.values))
    dist = ModuleType('torch.distributed')
    dist.ProcessGroup = ProcessGroup
    dist.get_rank = lambda group: 0
    dist.get_world_size = lambda group: 16
    dist.get_backend = lambda group: 'gloo'
    dist.all_gather = gather
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    torch.distributed = dist
    with pytest.raises(ValueError, match='Ranks disagree'):
        deep_ep.ElasticBuffer(group=ProcessGroup(), num_bytes=8 << 30, explicitly_destroy=True,
                              low_latency_backend='mte')
    assert backend.calls == [] and _runtime._OWNER is None
