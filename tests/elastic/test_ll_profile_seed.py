"""Behavioral tests for declared-route restoration before OFF/ON sampling."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.elastic import test_ep_low_latency as st


@pytest.mark.parametrize('repeats', [1, 2, 3, 6])
@pytest.mark.parametrize('mode', ['normal', 'hotspot'])
def test_reuse_callbacks_restore_live_closure_and_matching_handle(repeats, mode):
    world, b, k, experts, rank, seed = 16, 8, 8, 384, 1, 17
    state, events = {}, []

    def select(next_seed):
        routes = st.make_routes(world, b, k, experts, next_seed, mode)
        state.update(seed=next_seed, routes=routes, indices=routes[rank],
                     entries=st.expert_entries(routes, experts, world, rank))
        state['counts'] = [len(rows) for rows in state['entries']]
        events.append(('select', next_seed))

    def dispatch():
        events.append(('dispatch', state['seed']))
        return copy.deepcopy(state)

    def check(output):
        assert output == state
        assert output['indices'] == output['routes'][rank]
        assert output['counts'] == [len(rows) for rows in output['entries']]
        events.append(('check', output['seed']))

    select(seed)
    first = dispatch()
    events.clear()
    result = st.check_route_reuse(seed, repeats, first, select_seed=select, invoke=dispatch,
        wait=lambda output: events.append(('wait', output['seed'])), check=check,
        boundary=lambda: events.append(('boundary', None)))
    selected = [seed + repeat for repeat in range(1, repeats)] + ([seed] if repeats > 1 else [])
    expected = [event for value in selected for event in
                [('boundary', None), ('select', value), ('dispatch', value), ('wait', value), ('check', value)]]
    assert events == expected
    assert result == first and state == first
    if repeats == 1:
        assert result is first
    # Both formal and active-capture invocations use the same restored closure.
    for _ in range(3):
        assert dispatch() == first


def test_reuse_failure_does_not_enter_a_sampling_phase():
    calls = []
    def fail(output):
        calls.append('failed-check')
        raise AssertionError('route reuse failed')
    with pytest.raises(AssertionError, match='route reuse failed'):
        st.check_route_reuse(0, 3, None, select_seed=lambda seed: calls.append(seed),
            invoke=lambda: object(), wait=lambda _: None, check=fail, boundary=lambda: None)
    assert calls == [1, 'failed-check']


def test_route_fingerprint_is_canonical_and_sensitive_to_original_k_order():
    routes = st.make_routes(16, 2, 8, 384, 19)
    identity = st.profile_route_identity(routes, 19)
    expected = hashlib.sha256(json.dumps(routes, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()
    assert identity == dict(profile_seed=19, profile_route_sha256=expected, profile_route_fingerprint_schema=1)
    assert st.profile_route_identity(copy.deepcopy(routes), 19) == identity
    reordered = copy.deepcopy(routes)
    reordered[0][0][0], reordered[0][0][1] = reordered[0][0][1], reordered[0][0][0]
    assert st.profile_route_identity(reordered, 19)['profile_route_sha256'] != expected
    args = st.parse_args(['--num-processes', '16', '--num-tokens', '2', '--num-experts', '384', '--seed', '19'])
    args.runtime_identity = {}
    args.profile_route_identity = identity
    metadata = st.comparison_metadata(args, 16, 32)
    assert metadata['seed'] == metadata['profile_seed'] == 19
    assert metadata['profile_route_sha256'] == expected
    del args.profile_route_identity
    assert 'profile_seed' not in st.comparison_metadata(args, 16, 32), 'do not invent identities for old/custom callers'


class Tensor:
    def __init__(self, value=None, shape=None):
        self.value = copy.deepcopy(value)
        self.shape = shape
    def npu(self): return self
    def to(self, *args, **kwargs): return self
    def repeat(self, *args): return self
    def contiguous(self): return self
    def cpu(self): return self
    def __truediv__(self, number): return self


@pytest.mark.parametrize('repeats', [1, 3])
@pytest.mark.parametrize('dispatch_only', [False, True])
@pytest.mark.parametrize('dtype', ['bf16', 'fp8'])
def test_run_case_profiles_restored_indices_and_refreshes_combine_handle(tmp_path, monkeypatch, repeats, dispatch_only, dtype):
    world, rank, seed, batch, experts, k, hidden = 16, 1, 7, 4, 384, 8, 3584
    flags = ['--num-processes', '16', '--num-tokens', str(batch), '--num-experts', str(experts),
             '--hidden', str(hidden), '--num-topk', str(k), '--num-max-tokens-per-rank', '32',
             '--seed', str(seed), '--correctness-repeats', str(repeats), '--dtype', dtype, '--do-cpu-sync', '0']
    if dispatch_only:
        flags.append('--dispatch-only')
    args = st.parse_args(flags)
    args.run_directory, args.case_id, args.runtime_identity = str(tmp_path), 0, {}
    expected_routes = st.make_routes(world, batch, k, experts, seed)
    known = {json.dumps(st.make_routes(world, batch, k, experts, value)[rank]): value
             for value in range(seed, seed + repeats)}
    calls, samples, combine_handles = [], [], []

    def assert_equal(actual, expected, **kwargs):
        assert actual.value == expected.value

    torch = SimpleNamespace(float16='float16', bfloat16='bfloat16', float8_e4m3fn='fp8',
                            int64='int64', float32='float32',
                            tensor=lambda values, **kwargs: Tensor(values),
                            arange=lambda *args, **kwargs: Tensor(),
                            npu=SimpleNamespace(synchronize=lambda: None),
                            testing=SimpleNamespace(assert_close=assert_equal))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(st, 'payload', lambda *args, **kwargs: Tensor())
    monkeypatch.setattr(st, 'fp8_scales', lambda *args, **kwargs: Tensor())
    monkeypatch.setattr(st, 'record_case_progress', lambda *args, **kwargs: None)
    monkeypatch.setattr(st, 'completion_boundary', lambda *args, **kwargs: None)
    monkeypatch.setattr(st, 'trace_call', lambda options, label, function, *args, **kwargs: function(*args, **kwargs))
    monkeypatch.setattr(st, 'agree_check', lambda function, *args: function())
    package_dir = tmp_path / 'deep_ep'
    package_dir.mkdir()
    native, kernel = package_dir / '_C.test.so', tmp_path / 'libdeepep_kernels.so'
    native.write_bytes(b'project native')
    kernel.write_bytes(b'project kernel')
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if path.as_posix() == '/proc/self/maps':
            return f'1 2 3 4 5 {native.as_posix()}\n1 2 3 4 5 {kernel.as_posix()}\n'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(st.os, 'getsid', lambda pid: 0, raising=False)

    class Buffer:
        def dispatch(self, x, **kwargs):
            actual_seed = known[json.dumps(kwargs['topk_idx'].value)]
            routes = st.make_routes(world, batch, k, experts, actual_seed)
            calls.append(actual_seed)
            event = SimpleNamespace(current_stream_wait=lambda: None)
            handle = SimpleNamespace(is_low_latency=True, source_weights=True, destination_index=True,
                                     route_seed=actual_seed, routes=routes)
            return Tensor(actual_seed), None, None, handle, event
        def combine(self, expert_output, handle, **kwargs):
            assert expert_output.value == handle.route_seed == seed
            assert handle.routes == expected_routes
            combine_handles.append(handle.route_seed)
            return Tensor(seed, (batch, hidden)), None, SimpleNamespace(current_stream_wait=lambda: None)

    def verify(recv, handle, entries, *unused):
        assert recv.value == handle.route_seed
        assert entries == st.expert_entries(handle.routes, experts, world, rank)
    monkeypatch.setattr(st, 'verify_dispatch', verify)
    def prepare(recv, counts, *unused):
        assert recv.value == seed
        assert counts == [len(rows) for rows in st.expert_entries(expected_routes, experts, world, rank)]
        return Tensor(recv.value)
    monkeypatch.setattr(st, 'prepare_experts', prepare)
    def expected(routes, *args, **kwargs):
        assert routes == expected_routes
        return Tensor(seed, (batch, hidden))
    monkeypatch.setattr(st, 'expected_combine', expected)
    def sample(buffer, invoke, check, options, operation, *unused):
        for capture in ('off', 'active-on'):
            output = invoke()
            check(output)
            metadata = st.comparison_metadata(options, world, 32)
            samples.append((capture, operation, metadata['profile_seed'], metadata['profile_route_sha256']))
        return {'mock': True}
    monkeypatch.setattr(st, 'sample_operation', sample)
    result = st.run_case(Buffer(), args, rank, world, 32, 'normal')
    correctness_calls = [seed] + list(range(seed + 1, seed + repeats)) + ([seed] if repeats > 1 else [])
    assert calls[:len(correctness_calls)] == correctness_calls
    assert calls[len(correctness_calls):] and all(value == seed for value in calls[len(correctness_calls):])
    identity = st.profile_route_identity(expected_routes, seed)
    assert result['profile_route_identity'] == args.profile_route_identity == identity
    assert all(value[2:] == (seed, identity['profile_route_sha256']) for value in samples)
    assert {value[0] for value in samples} == {'off', 'active-on'}
    assert bool(combine_handles) is (not dispatch_only)
