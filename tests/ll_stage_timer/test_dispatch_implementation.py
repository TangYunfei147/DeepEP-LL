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

from scripts.ll_dispatch_implementation import (dispatch_configuration, dispatch_implementation,
                                                dispatch_group_valid, effective_dispatch_implementation, dispatch_capability)
from tests.elastic import test_ep_low_latency as st

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load('impl_benchmark', 'tests/ll_stage_timer/benchmark_compile.py')


@pytest.mark.parametrize('value', ['', 'CB59', 'cb59 ', 'baseline', '0'])
def test_invalid_selection_fails_before_planning(tmp_path, monkeypatch, value):
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL', value)
    with pytest.raises(ValueError, match='DISPATCH_IMPL'):
        dispatch_implementation()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('send,recv,prefix', [(30,26,0), (28,28,1), (28,28,28)])
@pytest.mark.parametrize('choice', ['cb59', 'cb59-stream', 'cb59-pair', 'cb59-small'])
def test_cb59_rejects_stale_split_and_prefix(send, recv, prefix, choice):
    with pytest.raises(ValueError, match=choice+' requires'):
        dispatch_configuration(choice, send, recv, prefix)




def test_default_and_native_cache_are_independent_of_source_hash():
    build = load('impl_build', 'build_support.py')
    assert dispatch_implementation({}) == 'current'
    config = dict(source_sha256='same-source',ll_reference_profile=False,ll_send_cores=28,ll_recv_cores=28)
    assert len({build.build_cache_key(dict(config,ll_dispatch_impl=choice,ll_reference_profile=timer))
                for choice in ('current','cb59','cb59-stream','cb59-pair','cb59-small') for timer in (False,True)}) == 10


@pytest.mark.parametrize('choice', ['current','cb59','cb59-stream','cb59-pair','cb59-small'])
def test_cmake_and_build_info_receive_selection(tmp_path, monkeypatch, choice):
    build = load('impl_package_build', 'build_support.py')
    from setuptools import Distribution, Extension
    monkeypatch.setitem(sys.modules,'pybind11',SimpleNamespace(get_cmake_dir=lambda:'pybind11-cmake'))
    config = dict(ll_dispatch_impl=choice,ll_dispatch_capability=dispatch_capability(choice),
                  ll_reference_profile=False,build_type='Release',
                  ll_send_cores=28,ll_recv_cores=28,ll_small_b_prefix_cores=0,cmake_path='cmake',
                  compiler_path='c++',ASCEND_HOME_PATH='cann',SHMEM_ROOT='shmem')
    monkeypatch.setattr(build,'build_configuration',lambda:config)
    monkeypatch.setattr(build,'ROOT',tmp_path)
    ext=Extension('deep_ep._C',sources=[])
    command=build.CMakeBuild(Distribution({'ext_modules':[ext]}))
    command.ensure_finalized()
    command.build_temp=str(tmp_path/'build/native cache')
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
    assert json.loads((destination/'_build_info.json').read_text())['ll_dispatch_capability']==dispatch_capability(choice)


@pytest.mark.parametrize('choice', ['current','cb59','cb59-stream','cb59-pair','cb59-small'])
def test_installed_selection_is_required_even_for_same_source(tmp_path, choice):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'))
    native=SimpleNamespace(low_latency_dispatch_impl=choice)
    path=tmp_path/'_build_info.json'
    env={'DEEPEP_LL_DISPATCH_IMPL':choice}
    for info in ({'source_sha256':'a'*64}, {'source_sha256':'a'*64,'ll_dispatch_impl':'wrong'}):
        path.write_text(json.dumps(info))
        with pytest.raises(RuntimeError,match='dispatch identity'):
            st.installed_dispatch_identity(package,native,env)
    path.write_text(json.dumps(dict(source_sha256='a'*64,ll_dispatch_impl=choice,ll_dispatch_capability=dispatch_capability(choice))))
    identity=st.installed_dispatch_identity(package,native,env)
    assert identity==dict(dispatch_impl=choice,dispatch_capability=dispatch_capability(choice))
    with pytest.raises(RuntimeError,match='differs from DEEPEP_LL_DISPATCH_IMPL'):
        st.installed_dispatch_identity(package,native,{'DEEPEP_LL_DISPATCH_IMPL':'cb59' if choice=='current' else 'current'})
    with pytest.raises(RuntimeError,match='differs across ranks'):
        st.validate_matching_capabilities([identity,dict(dispatch_impl='cb59' if choice=='current' else 'current')])












@pytest.mark.parametrize('group', [1,2,4,8,16])
def test_current_explicit_group_override_is_validated_and_recorded(group):
    identity=recorded_identity('current')
    identity['logical_group']=group
    assert dispatch_group_valid('current',group,16)
    st.validate_dispatch_algorithm_capability(identity,16)
    # Runtime agreement stays strict even though each build may choose G.
    if group!=4:
        with pytest.raises(RuntimeError,match='differs across ranks'):
            st.validate_matching_capabilities([identity,dict(identity,logical_group=4)])


@pytest.mark.parametrize('choice,group,world', [('current',3,16),('current',17,32),('current',True,16),('cb59',4,16),
                                              ('cb59-stream',4,16),('cb59-pair',4,16),('cb59-small',4,16)])
def test_invalid_group_and_cb59_override_remain_rejected(choice,group,world):
    identity=recorded_identity(choice);identity['logical_group']=group;identity['world_size']=world
    assert not dispatch_group_valid(choice,group,world)
    with pytest.raises(RuntimeError,match='algorithm/group/q'):
        st.validate_dispatch_algorithm_capability(identity,world)


@pytest.mark.parametrize('choice,macro', [('cb59',1),('cb59-stream',2),('cb59-pair',3),('cb59-small',4)])
def test_compile_benchmark_selects_one_algorithm_in_every_existing_unit(tmp_path, monkeypatch, choice, macro):
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL',choice)
    monkeypatch.setenv('DEEPEP_LL_SEND_CORES','28');monkeypatch.setenv('DEEPEP_LL_RECV_CORES','28')
    monkeypatch.setenv('DEEPEP_LL_SMALL_B_PREFIX_CORES','0')
    monkeypatch.setenv('ASCEND_HOME_PATH',str(tmp_path/'cann'))
    monkeypatch.setattr(sys,'argv',['benchmark_compile.py','--output',str(tmp_path/'objects'),'--reference-profile','OFF'])
    calls=[]
    monkeypatch.setattr(benchmark.subprocess,'run',lambda command,**kw:(calls.append(command) or SimpleNamespace(returncode=0)))
    monkeypatch.setattr(benchmark.subprocess,'check_output',lambda *a,**k:'revision')
    benchmark.main()
    assert len(calls)==8  # Host selector plus the existing seven mode TUs.
    assert all(f'-DDEEPEP_LL_DISPATCH_CB59={macro}' in command for command in calls)


@pytest.mark.parametrize('choice,send,recv,prefix,valid', [
    ('current',30,26,1,True),('cb59',28,28,0,True),('cb59',30,26,0,False),
    ('cb59',28,28,1,False),('cb59-stream',28,28,0,True),('cb59-stream',30,26,0,False),
    ('cb59-stream',28,28,1,False),('cb59-pair',28,28,0,True),('cb59-pair',30,26,0,False),
    ('cb59-pair',28,28,1,False),('cb59-small',28,28,0,True),('cb59-small',30,26,0,False),
    ('cb59-small',28,28,1,False),('invalid',28,28,0,False)])
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


@pytest.mark.parametrize('recorded,native_choice,requested', [
    ('cb59','cb59-stream','cb59-stream'),('cb59-stream','cb59','cb59'),
    ('cb59','cb59','cb59-stream'),('cb59-stream','cb59-stream','cb59')])
def test_replay_and_stream_wheels_cannot_mix_even_with_identical_source(tmp_path,recorded,native_choice,requested):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'))
    (tmp_path/'_build_info.json').write_text(json.dumps(dict(source_sha256='a'*64,ll_dispatch_impl=recorded)))
    native=SimpleNamespace(low_latency_dispatch_impl=native_choice)
    with pytest.raises(RuntimeError,match='dispatch identity'):
        st.installed_dispatch_identity(package,native,{'DEEPEP_LL_DISPATCH_IMPL':requested})


def test_stream_live_capability_requires_its_protocol_and_replay_family_configuration():
    identity=recorded_identity('cb59-stream')
    st.validate_core_split_capability(identity)
    st.validate_dispatch_algorithm_capability(identity,16)
    st.validate_overlap_stage_capability(identity)
    for value in (19028,21028,22030):
        with pytest.raises(RuntimeError,match='56-core split'):
            st.validate_core_split_capability(dict(identity,core_protocol=value))
    for field,value in [('logical_group',4),('local_q_table',True),('local_q_seed_vf',True)]:
        with pytest.raises(RuntimeError,match='algorithm/group/q'):
            st.validate_dispatch_algorithm_capability(dict(identity,**{field:value}),16)
    assert identity['small_b_prefix_cores']==0 and identity['effective_small_b_prefix_cores']==28


@pytest.mark.parametrize('world,b,m,path', [
    (16,1,1,'cb59-stream'),(16,1,32,'cb59-stream'),(16,32,32,'cb59-stream'),
    (16,1,64,'cb59'),(16,48,64,'cb59'),(32,1,32,'cb59'),(64,32,32,'cb59')])
def test_stream_records_the_actual_shape_gated_path(world,b,m,path):
    assert effective_dispatch_implementation('cb59-stream',world,b,m)==path
    args=st.parse_args(['--num-tokens',str(b),'--num-max-tokens-per-rank',str(m)])
    args.runtime_identity=dict(dispatch_impl='cb59-stream',recv_cores=28,small_b_prefix_cores=0,source_sha256='a'*64)
    result=st.comparison_metadata(args,world,m)
    assert result['dispatch_impl']=='cb59-stream' and result['effective_dispatch_impl']==path
    assert effective_dispatch_implementation('cb59',world,b,m)=='cb59'
    assert effective_dispatch_implementation('current',world,b,m)=='current'


def test_stream_effective_path_cannot_hide_an_inapplicable_or_missing_path():
    assert effective_dispatch_implementation('cb59-stream',16,1,32)=='cb59-stream'
    assert effective_dispatch_implementation('cb59-stream',32,1,32)=='cb59'



def test_stream_capability_covers_every_mode_but_does_not_claim_large_or_multihost_streaming():
    scope=dispatch_capability('cb59-stream')
    assert scope==dict(mode_count=7,scope='ep16-small-direct',world_size=16,max_B=32,max_M=32,fallback_impl='cb59')
    assert dispatch_capability('current')==dispatch_capability('cb59')==dict(mode_count=7,scope='all-supported',fallback_impl=None)


@pytest.mark.parametrize('world,b,m', [(16,1,32),(32,8,32),(64,16,32),(128,32,32),(256,1,1)])
def test_small_scope_keeps_original_domains_and_pair_falls_back(world,b,m,monkeypatch):
    assert effective_dispatch_implementation('cb59-small',world,b,m)=='cb59-small'
    assert effective_dispatch_implementation('cb59-pair',world,b,m)==('cb59-pair' if world==16 else 'cb59')
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL','cb59-small')
    args=st.parse_args(['--num-tokens',str(b),'--num-max-tokens-per-rank',str(m),'--num-experts','256'])
    assert st.validate_args(args,world)==m


@pytest.mark.parametrize('b,m', [(1,64),(32,64),(48,48),(64,128)])
def test_small_scope_rejects_large_capacity_before_execution_or_fallback(b,m,monkeypatch):
    with pytest.raises(ValueError,match='no fallback'):
        effective_dispatch_implementation('cb59-small',16,b,m)
    monkeypatch.setenv('DEEPEP_LL_DISPATCH_IMPL','cb59-small')
    args=st.parse_args(['--num-tokens',str(b),'--num-max-tokens-per-rank',str(m)])
    with pytest.raises(ValueError,match='no fallback'):
        st.validate_args(args,16)
    assert effective_dispatch_implementation('cb59-pair',16,b,m)=='cb59'


def test_pair_and_small_capabilities_do_not_mislabel_scope_or_fallback():
    assert dispatch_capability('cb59-pair')==dict(mode_count=7,scope='ep16-small-direct-paired',world_size=16,
        max_B=32,max_M=32,receiver_pair_size=2,fallback_impl='cb59')
    assert dispatch_capability('cb59-small')==dict(mode_count=7,scope='small-direct-only',max_B=32,max_M=32,
        temporary=True,fallback_impl=None)


def recorded_identity(choice):
    expected=dispatch_configuration(choice)
    args=st.parse_args(['--num-processes','16','--num-tokens','8','--num-max-tokens-per-rank','32'])
    args.runtime_identity=dict(implementation=expected['implementation'],dispatch_impl=choice,
        dispatch_capability=dispatch_capability(choice),
        transport='mte',active_aiv=56,send_cores=28,recv_cores=28,dispatch_recv_cores=56 if choice=='current' else 28,
        logical_group=expected['logical_group'],core_protocol=expected['core_protocol'],stage_timer=False,
        stage_timer_schema=16,rdma_compiled=False,small_b_prefix_cores=0,shmem_revision='a'*40,
        local_q_table=expected['local_q_table'],local_q_seed_vf=expected['local_q_seed_vf'],
        source_sha256='b'*64,overlap_stage=expected['overlap_stage'])
    return st.comparison_metadata(args,16,32)
