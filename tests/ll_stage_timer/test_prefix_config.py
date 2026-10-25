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


diagnose = load('prefix_diagnose', 'tests/ll_stage_timer/diagnose.py')
benchmark = load('prefix_benchmark', 'tests/ll_stage_timer/benchmark_compile.py')


def plan(root):
    args, forwarded = diagnose.parser().parse_known_args(['--build-only', '--run-name', 'prefix-build'])
    return diagnose.make_plan(args, forwarded, root=root)


@pytest.mark.parametrize('prefix', [0, 1, 6, 26])
def test_plan_freezes_prefix_for_both_wheels(tmp_path, monkeypatch, prefix):
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '30')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '26')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', str(prefix))
    result = plan(tmp_path / 'Project With Spaces')
    assert result['small_b_prefix_cores'] == prefix
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', '8')
    for stage in result['stages']:
        assert stage['run'] is None and stage['analyze'] is None
        assert diagnose._environment(result, stage)['DEEPEP_LL_SMALL_B_PREFIX_CORES'] == str(prefix)
        code = diagnose._verify_code(stage, 'candidate', result['core_split'], result['small_b_prefix_cores'])
        assert f"native.low_latency_small_b_prefix_cores == {prefix}" in code
        assert 'low_latency_core_config(' not in code


def test_plan_default_prefix_is_zero(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', raising=False)
    assert plan(tmp_path)['small_b_prefix_cores'] == 0


@pytest.mark.parametrize('value', ['', '-1', '27', '56', '01', '+1', '1.0', '１'])
def test_plan_rejects_invalid_prefix_before_any_mutation(tmp_path, monkeypatch, value):
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '30')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '26')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', value)
    with pytest.raises(ValueError, match='DEEPEP_LL_SMALL_B_PREFIX_CORES'):
        plan(tmp_path)
    assert not list(tmp_path.iterdir())


def test_installed_prefix_identity_and_all_rank_agreement(tmp_path):
    package = SimpleNamespace(__file__=str(tmp_path / '__init__.py'))
    info = dict(ll_small_b_prefix_cores=6, shmem=dict(revision='a' * 40))
    path = tmp_path / '_build_info.json'
    path.write_text(json.dumps(info))
    native = SimpleNamespace(low_latency_small_b_prefix_cores=6)
    identity = st.installed_prefix_identity(package, native, 26, {})
    assert identity == dict(small_b_prefix_cores=6, shmem_revision='a' * 40)
    st.validate_matching_capabilities([identity, dict(identity)])
    for changed in (dict(identity, small_b_prefix_cores=1), dict(identity, shmem_revision='b' * 40)):
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
    for revision in (None, 'unknown', 'a' * 39):
        path.write_text(json.dumps(dict(info, shmem=dict(revision=revision))))
        with pytest.raises(RuntimeError, match='SHMEM revision'):
            st.installed_prefix_identity(package, native, 26, {})


@pytest.mark.parametrize('world,m,b,prefix,effective', [
    (16, 32, 1, 0, 26), (16, 32, 32, 6, 6), (16, 32, 1, 1, 1),
    (32, 32, 8, 6, 26), (64, 32, 8, 1, 26),
    (16, 64, 1, 6, None), (16, 64, 48, 6, None)])
def test_comparison_records_raw_and_only_applicable_effective_prefix(world, m, b, prefix, effective):
    args = st.parse_args(['--num-processes', str(world), '--num-tokens', str(b), '--num-max-tokens-per-rank', str(m)])
    args.runtime_identity = dict(small_b_prefix_cores=prefix, send_cores=30, recv_cores=26, shmem_revision='a' * 40)
    metadata = st.comparison_metadata(args, world, m)
    assert metadata['small_b_prefix_cores'] == prefix
    assert metadata['effective_small_b_prefix_cores'] == effective
    assert metadata['session_transport'] == 'mte' and metadata['shmem_revision'] == 'a' * 40


@pytest.mark.parametrize('field,left_value,right_value', [
    ('small_b_prefix_cores', 6, 1), ('effective_small_b_prefix_cores', 6, 1),
    ('session_transport', 'mte', 'rdma'), ('shmem_revision', 'a' * 40, 'b' * 40)])
def test_optional_performance_identity_must_match_if_present(tmp_path, field, left_value, right_value):
    suite = tmp_path / 'suite'
    (suite / 'runs/off').mkdir(parents=True)
    directory = suite / 'runs/sparse/case_00000/normal'
    directory.mkdir(parents=True)
    config = dict(B=1, M=32)
    base = dict(implementation='candidate', transport='mte', logical_group=4, active_aiv=56,
                send_cores=30, recv_cores=26, core_protocol=19030, source_revision='commit', source_sha256='a' * 64)
    def write(left, right):
        (suite / 'runs/off/summary.json').write_text(json.dumps(dict(results=[dict(
            config=config, route='normal', comparison=left, dispatch=dict(max_rank_mean_kernel_us=10))])))
        (directory / 'active_capture_dispatch.json').write_text(json.dumps(dict(
            config=config, route='normal', comparison=right, status='available', diagnostic_iters=1,
            summary=dict(max_rank_mean_kernel_us=12))))
        return diagnose._performance_rows(suite)[0]
    assert write(base, base)['ratio'] == 1.2  # Existing offline pairs remain readable.
    left = dict(base, **{field: left_value})
    assert write(left, left)['ratio'] == 1.2
    for a, b in ((left, base), (base, left), (left, dict(base, **{field: right_value}))):
        row = write(a, b)
        assert row['status'].startswith('unavailable') and 'ratio' not in row


def test_compile_benchmark_passes_prefix_to_every_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '30')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '26')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', '6')
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path / 'cann'))
    out = tmp_path / 'Compile With Spaces'
    monkeypatch.setattr(sys, 'argv', ['benchmark_compile.py', '--output', str(out), '--timer', 'OFF'])
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
