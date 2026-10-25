"""Compile selection, mixed-wheel rejection and comparable OFF evidence; no NPU."""
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.ll_dispatch_implementation import dispatch_configuration, dispatch_implementation, dispatch_group_valid
from tests.elastic import test_ep_low_latency as st

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnose = load('impl_diagnose', 'tests/ll_stage_timer/diagnose.py')
comparison = load('impl_compare', 'tests/ll_stage_timer/compare.py')
benchmark = load('impl_benchmark', 'tests/ll_stage_timer/benchmark_compile.py')


@pytest.mark.parametrize('value', ['', 'CB59', 'cb59 ', 'baseline', '0'])
def test_invalid_selection_fails_before_planning(tmp_path, monkeypatch, value):
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL', value)
    args, forwarded = diagnose.parser().parse_known_args(['--build-only', '--run-name', 'trial'])
    with pytest.raises(ValueError, match='DISPATCH_IMPL'):
        diagnose.make_plan(args, forwarded, root=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('send,recv,prefix', [(30,26,0), (28,28,1), (28,28,28)])
def test_cb59_rejects_stale_split_and_prefix(send, recv, prefix):
    with pytest.raises(ValueError, match='cb59 requires'):
        dispatch_configuration('cb59', send, recv, prefix)


@pytest.mark.parametrize('choice,group,protocol', [('current',4,19028), ('cb59',16,21028)])
def test_plan_freezes_selection_for_every_build_and_run(tmp_path, monkeypatch, choice, group, protocol):
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL', choice)
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES', '28')
    monkeypatch.setenv('DEEPEP_LL_RECV_CORES', '28')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES', '0')
    args, forwarded = diagnose.parser().parse_known_args(['--build-only', '--run-name', 'build'])
    plan = diagnose.make_plan(args, forwarded, root=tmp_path / 'checkout with spaces')
    assert (plan['dispatch_impl'], plan['default_logical_group'], plan['core_protocol']) == (choice,group,protocol)
    assert plan['required_logical_group'] == (16 if choice=='cb59' else None)
    assert [stage['name'] for stage in plan['stages']] == ['off','sparse']
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL', 'cb59' if choice=='current' else 'current')
    for stage in plan['stages']:
        assert diagnose._environment(plan, stage)['DEEPEP_LL_DISPATCH_IMPL'] == choice
        assert stage['run'] is None


def test_default_and_native_cache_are_independent_of_source_hash():
    build = load('impl_build', 'build_support.py')
    assert dispatch_implementation({}) == 'current'
    config = dict(source_sha256='same-source',ll_stage_timer=False,ll_send_cores=28,ll_recv_cores=28)
    assert len({build.build_cache_key(dict(config,ll_dispatch_impl=choice,ll_stage_timer=timer))
                for choice in ('current','cb59') for timer in (False,True)}) == 4


@pytest.mark.parametrize('choice', ['current','cb59'])
def test_cmake_and_build_info_receive_selection(tmp_path, monkeypatch, choice):
    build = load('impl_package_build', 'build_support.py')
    from setuptools import Distribution, Extension
    monkeypatch.setitem(sys.modules,'pybind11',SimpleNamespace(get_cmake_dir=lambda:'pybind11-cmake'))
    config = dict(ll_dispatch_impl=choice,ll_stage_timer=False,ll_timer_profile='sparse',build_type='Release',
                  ll_send_cores=28,ll_recv_cores=28,ll_small_b_prefix_cores=0,cmake_path='cmake',
                  compiler_path='c++',ASCEND_HOME_PATH='cann',SHMEM_ROOT='shmem')
    monkeypatch.setattr(build,'build_configuration',lambda:config)
    ext=Extension('deep_ep._C',sources=[])
    command=build.CMakeBuild(Distribution({'ext_modules':[ext]}))
    command.ensure_finalized()
    command.build_temp=str(tmp_path/'native cache')
    command.build_lib=str(tmp_path/'package')
    destination=Path(command.get_ext_fullpath(ext.name)).parent
    destination.mkdir(parents=True)
    calls=[]
    def execute(arguments):
        calls.append(list(map(str,arguments)))
        if '-B' in arguments:
            target=Path(arguments[arguments.index('-B')+1])
            (target/'compiler-info.json').write_text('{"id":"mock","version":"1"}')
            (target/'capabilities.json').write_text('{}')
    monkeypatch.setattr(build,'checked_run',execute)
    command.build_extension(ext)
    assert f'-DDEEPEP_LL_DISPATCH_IMPL={choice}' in calls[0]
    assert json.loads((destination/'_build_info.json').read_text())['ll_dispatch_impl']==choice


@pytest.mark.parametrize('choice', ['current','cb59'])
def test_installed_selection_is_required_even_for_same_source(tmp_path, choice):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'))
    native=SimpleNamespace(low_latency_dispatch_impl=choice)
    path=tmp_path/'_build_info.json'
    env={'DEEPEP_LL_DISPATCH_IMPL':choice}
    for info in ({'source_sha256':'a'*64}, {'source_sha256':'a'*64,'ll_dispatch_impl':'wrong'}):
        path.write_text(json.dumps(info))
        with pytest.raises(RuntimeError,match='dispatch identity'):
            st.installed_dispatch_identity(package,native,env)
    path.write_text(json.dumps(dict(source_sha256='a'*64,ll_dispatch_impl=choice)))
    identity=st.installed_dispatch_identity(package,native,env)
    assert identity==dict(dispatch_impl=choice)
    with pytest.raises(RuntimeError,match='differs from DEEPEP_LL_DISPATCH_IMPL'):
        st.installed_dispatch_identity(package,native,{'DEEPEP_LL_DISPATCH_IMPL':'cb59' if choice=='current' else 'current'})
    with pytest.raises(RuntimeError,match='differs across ranks'):
        st.validate_matching_capabilities([identity,dict(dispatch_impl='cb59' if choice=='current' else 'current')])


@pytest.mark.parametrize('choice', ['current','cb59'])
def test_host_verification_requires_plan_native_and_info_agreement(tmp_path, monkeypatch, choice):
    identity=dispatch_configuration(choice)
    package=tmp_path/'deep_ep';package.mkdir()
    info=dict(source_sha256='same-source',ll_stage_timer=False,ll_timer_profile='sparse',
              ll_dispatch_impl=choice,ll_send_cores=28,ll_recv_cores=28,ll_small_b_prefix_cores=0)
    info_path=package/'_build_info.json';info_path.write_text(json.dumps(info))
    layout=dict(schema=15,slots=8,fields=8,recording='sparse-milestones',timer_profile='sparse',timer_profile_id=6,
                points=list(runpy.run_path(str(ROOT/'deep_ep/utils/ll_stage_timer.py'))['SPARSE_POINTS']))
    native=SimpleNamespace(low_latency_dispatch_impl=choice,low_latency_implementation=identity['implementation'],
        low_latency_ranks_per_group=identity['logical_group'],low_latency_overlap_stage=identity['overlap_stage'],
        low_latency_q_table_enabled=identity['local_q_table'],low_latency_q_seed_vf=identity['local_q_seed_vf'],
        low_latency_stage_timer_enabled=False,low_latency_stage_timer_profile='sparse',low_latency_stage_timer_profile_id=6,
        low_latency_stage_timer_layout=lambda:dict(layout),low_latency_core_protocol=identity['core_protocol'],
        low_latency_small_b_prefix_cores=0)
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda _:native))
    original=runpy.run_path
    def run(path):
        if path=='tests/elastic/test_dispatch.py':
            return dict(_import_installed_deep_ep=lambda:SimpleNamespace(__file__=str(package/'__init__.py')))
        if path=='tests/elastic/test_ep_low_latency.py':return dict(validate_sparse_timer_layout=st.validate_sparse_timer_layout)
        if path=='build_support.py':return dict(source_fingerprint=lambda:'same-source')
        return original(path)
    monkeypatch.setattr(runpy,'run_path',run)
    code=diagnose._verify_code(dict(profile='sparse',timer=False),identity['implementation'],dict(send=28,recv=28),0,choice)
    exec(code,{})
    if choice=='current':
        native.low_latency_ranks_per_group=16
        exec(code,{})  # Existing compile-time G16 override remains a legal current wheel.
    else:
        native.low_latency_ranks_per_group=4
        with pytest.raises(RuntimeError,match='native logical group'):exec(code,{})
        native.low_latency_ranks_per_group=16
    info_path.write_text(json.dumps(dict(info,ll_dispatch_impl='cb59' if choice=='current' else 'current')))
    with pytest.raises(RuntimeError,match='wheel dispatch selection'):exec(code,{})
    info_path.write_text(json.dumps(info))
    native.low_latency_dispatch_impl='cb59' if choice=='current' else 'current'
    with pytest.raises(RuntimeError,match='native dispatch selection'):exec(code,{})


def recorded_identity(choice):
    expected=dispatch_configuration(choice)
    args=st.parse_args(['--num-processes','16','--num-tokens','8','--num-max-tokens-per-rank','32'])
    args.runtime_identity=dict(implementation=expected['implementation'],dispatch_impl=choice,
        transport='mte',active_aiv=56,send_cores=28,recv_cores=28,dispatch_recv_cores=28 if choice=='cb59' else 56,
        logical_group=expected['logical_group'],core_protocol=expected['core_protocol'],stage_timer=False,
        stage_timer_schema=15,rdma_compiled=False,small_b_prefix_cores=0,shmem_revision='a'*40,
        local_q_table=expected['local_q_table'],local_q_seed_vf=expected['local_q_seed_vf'],
        source_sha256='b'*64,overlap_stage=expected['overlap_stage'])
    return st.comparison_metadata(args,16,32)


def test_same_carrier_does_not_turn_different_algorithms_into_stage_speedup():
    left,right=recorded_identity('current'),recorded_identity('cb59')
    assert right['effective_small_b_prefix_cores']==28
    def perf(identity, mean):
        return dict(status='completed',results=[dict(case_id=0,correctness='passed',comparison=identity,
                    dispatch=dict(ranks=[dict(rank=0,mean_us=mean)]))])
    assert comparison.compare(perf(left,30),perf(right,18),'perf')[0]['candidate_over_baseline']==.6
    def stage(identity, mean):
        return dict(schema=15,timer_profile='sparse',comparison=identity,per_rank_core_stage=[
            dict(operation='dispatch',rank=0,core=28,stage='ENTRY_TO_PREFIX_READY',unit='us',mean=mean)])
    row=comparison.compare(stage(left,30),stage(right,18),'stage')[0]
    assert row['status']=='dispatch_algorithm_changed' and row['candidate_over_baseline'] is None
    for field,value in [('shmem_revision','c'*40),('session_transport','udma'),('use_1g_huge_pages',True),('num_iters',50)]:
        with pytest.raises(ValueError,match='Incomparable'):
            comparison.check_identity(left,dict(right,**{field:value}))


def test_legacy_reports_remain_offline_readable_without_new_identity():
    old=recorded_identity('current');old.pop('dispatch_impl')
    comparison.check_identity(old,dict(old))
    broken=recorded_identity('cb59');broken.pop('dispatch_impl')
    with pytest.raises(ValueError,match='Missing cb59'):
        comparison.check_identity(broken,broken)


@pytest.mark.parametrize('group', [1,2,4,8,16])
def test_current_explicit_group_override_is_validated_and_recorded(group):
    identity=recorded_identity('current')
    identity['logical_group']=group
    assert dispatch_group_valid('current',group,16)
    st.validate_dispatch_algorithm_capability(identity,16)
    comparison.check_identity(identity,dict(identity))
    # Runtime agreement stays strict even though each build may choose G.
    if group!=4:
        with pytest.raises(RuntimeError,match='differs across ranks'):
            st.validate_matching_capabilities([identity,dict(identity,logical_group=4)])


@pytest.mark.parametrize('choice,group,world', [('current',3,16),('current',17,32),('current',True,16),('cb59',4,16)])
def test_invalid_group_and_cb59_override_remain_rejected(choice,group,world):
    identity=recorded_identity(choice);identity['logical_group']=group;identity['world_size']=world
    assert not dispatch_group_valid(choice,group,world)
    with pytest.raises(RuntimeError,match='algorithm/group/q'):
        st.validate_dispatch_algorithm_capability(identity,world)
    with pytest.raises(ValueError):comparison.check_identity(identity,dict(identity))


def test_compile_benchmark_selects_one_algorithm_in_every_existing_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL','cb59')
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES','28');monkeypatch.setenv('DEEPEP_LL_RECV_CORES','28')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES','0')
    monkeypatch.setenv('ASCEND_HOME_PATH',str(tmp_path/'cann'))
    monkeypatch.setattr(sys,'argv',['benchmark_compile.py','--output',str(tmp_path/'objects'),'--timer','OFF'])
    calls=[]
    monkeypatch.setattr(benchmark.subprocess,'run',lambda command,**kw:(calls.append(command) or SimpleNamespace(returncode=0)))
    monkeypatch.setattr(benchmark.subprocess,'check_output',lambda *a,**k:'revision')
    benchmark.main()
    assert len(calls)==8  # Host selector plus the existing seven mode TUs.
    assert all('-DDEEPEP_LL_DISPATCH_CB59=1' in command for command in calls)


@pytest.mark.parametrize('choice,send,recv,prefix,valid', [
    ('current',30,26,1,True),('cb59',28,28,0,True),('cb59',30,26,0,False),
    ('cb59',28,28,1,False),('invalid',28,28,0,False)])
def test_real_cmake_selection_and_constraints_match_python(tmp_path, choice, send, recv, prefix, valid):
    cmake=shutil.which('cmake')
    if cmake is None:pytest.skip('CMake unavailable')
    output=tmp_path/'definitions.txt'
    script=tmp_path/'selection.cmake'
    # Script mode has no build target; capture the definitions passed to the
    # compiler boundary while executing the actual CMake validation logic.
    script.write_text('macro(add_compile_definitions)\n'
        f'  file(WRITE "{output.as_posix()}" "${{ARGV}}")\nendmacro()\n'
        f'include("{(ROOT/"cmake/CoreSplit.cmake").as_posix()}")\n')
    result=subprocess.run([cmake,f'-DDEEPEP_LL_DISPATCH_IMPL={choice}',f'-DDEEPEP_LL_SEND_CORES={send}',
        f'-DDEEPEP_LL_RECV_CORES={recv}',f'-DDEEPEP_LL_SMALL_B_PREFIX_CORES={prefix}','-P',str(script)],
        capture_output=True,text=True,timeout=30)
    if valid:
        assert result.returncode==0,result.stderr
        expected=dispatch_configuration(choice,send,recv,prefix)
        assert f'DEEPEP_LL_DISPATCH_CB59={expected["cb59"]}' in output.read_text().split(';')
    else:
        assert result.returncode!=0
        with pytest.raises(ValueError):dispatch_configuration(choice,send,recv,prefix)
