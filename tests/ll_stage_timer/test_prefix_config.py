"""P wheel identities, pure planning, full-task comparison and compiler flags."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from tests.elastic import test_ep_low_latency as st

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load('prefix_benchmark', 'tests/ll_stage_timer/benchmark_compile.py')










def test_installed_prefix_identity_and_all_rank_agreement(tmp_path):
    package = SimpleNamespace(__file__=str(tmp_path / '__init__.py'))
    info = dict(ll_small_b_prefix_cores=6, shmem=dict(revision='a' * 40))
    path = tmp_path / '_build_info.json'
    path.write_text(json.dumps(info))
    native = SimpleNamespace(low_latency_small_b_prefix_cores=6)
    identity = st.installed_prefix_identity(package, native, 26, {})
    assert identity == dict(small_b_prefix_cores=6)
    st.validate_matching_capabilities([identity, dict(identity)])
    for changed in (dict(identity, small_b_prefix_cores=1),):
        with pytest.raises(RuntimeError, match='differs across ranks'):
            st.validate_matching_capabilities([identity, changed])
    for requested in ('0', '1', '06', '-1', ''):
        with pytest.raises(RuntimeError, match='P differs'):
            st.installed_prefix_identity(package, native, 26, {'DEEPEP_LL_SMALL_B_PREFIX_CORES': requested})
    assert st.installed_prefix_identity(package, native, 26, {'DEEPEP_LL_SMALL_B_PREFIX_CORES': '6'}) == identity
    for malformed in ({}, dict(info, ll_small_b_prefix_cores=1), dict(info, ll_small_b_prefix_cores=True)):
        path.write_text(json.dumps(malformed))
        with pytest.raises(RuntimeError, match='P identity'):
            st.installed_prefix_identity(package, native, 26, {})
    for revision in (None, 'unknown', 'a' * 39, {}, 99):
        path.write_text(json.dumps(dict(info, shmem=dict(revision=revision))))
        assert st.installed_prefix_identity(package, native, 26, {}) == identity


@pytest.mark.parametrize('world,m,b,prefix,effective', [
    (16, 32, 1, 0, 26), (16, 32, 32, 6, 6), (16, 32, 1, 1, 1),
    (32, 32, 8, 6, 26), (64, 32, 8, 1, 26),
    (16, 64, 1, 6, None), (16, 64, 48, 6, None)])
def test_comparison_records_raw_and_only_applicable_effective_prefix(world, m, b, prefix, effective):
    args = st.parse_args(['--num-processes', str(world), '--num-tokens', str(b), '--num-max-tokens-per-rank', str(m)])
    args.runtime_identity = dict(small_b_prefix_cores=prefix, send_cores=30, recv_cores=26, shmem_revision='a' * 40,source_sha256='a'*64)
    metadata = st.comparison_metadata(args, world, m)
    assert metadata['small_b_prefix_cores'] == prefix
    assert metadata['effective_small_b_prefix_cores'] == effective
    assert metadata['session_transport'] == 'mte' and metadata['shmem_revision'] == 'a' * 40




def test_compile_benchmark_passes_prefix_to_every_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '30')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '26')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', '6')
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path / 'cann'))
    out = tmp_path / 'Compile With Spaces'
    monkeypatch.setattr(sys, 'argv', ['benchmark_compile.py', '--output', str(out), '--reference-profile', 'OFF'])
    calls = []
    def compile_mock(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(benchmark.subprocess, 'run', compile_mock)
    monkeypatch.setattr(benchmark.subprocess, 'check_output', lambda *a, **k: 'commit')
    benchmark.main()
    assert len(calls) > 1
    assert all('-DDEEPEP_LL_SMALL_B_PREFIX_CORES=6' in command for command in calls)
    report = json.loads((out / 'timing.json').read_text())
    assert '-DDEEPEP_LL_SMALL_B_PREFIX_CORES=6' in report['flags']


def test_compile_benchmark_rejects_prefix_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', '56')
    out = tmp_path / 'should-not-exist'
    monkeypatch.setattr(sys, 'argv', ['benchmark_compile.py', '--output', str(out)])
    with pytest.raises(SystemExit):
        benchmark.main()
    assert not out.exists()
