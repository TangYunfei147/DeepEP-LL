"""Historical and new wire/prefix ABI identities must agree before launch."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite=load('protocol_identity_suite','tests/ll_control_opt/suite.py')
st=load('protocol_identity_st','tests/elastic/test_ep_low_latency.py')
build=suite.build

VALID=[(0,0,1,19000),(1,0,1,23000),(2,0,1,24000),
       (0,1,1,25000),(1,1,1,26000),(2,1,1,27000),
       (0,1,2,25000),(1,1,2,26000),(2,1,2,28000)]


def capability(mode,prefix,wire,base):
    return dict(send_cores=30,recv_cores=26,active_aiv=56,core_protocol=base+30,
                shared_pull_probe_mode=mode,prefix_ready_revision=prefix,
                shared_pull_wire_revision=wire)


@pytest.mark.parametrize('mode,prefix,wire,base',VALID)
def test_st_and_build_agree_on_explicit_protocol_and_reject_other_abi(mode,prefix,wire,base):
    assert build.protocol_base(mode,prefix,wire)==base
    cap=capability(mode,prefix,wire,base)
    st.validate_core_split_capability(cap)
    cap['core_protocol']=99999
    with pytest.raises(RuntimeError,match='matching configured'):
        st.validate_core_split_capability(cap)


@pytest.mark.parametrize('field,value',[
    ('shared_pull_probe_mode',True),('shared_pull_probe_mode',3),
    ('prefix_ready_revision',True),('prefix_ready_revision',2),
    ('shared_pull_wire_revision',True),('shared_pull_wire_revision',0),
    ('shared_pull_wire_revision','1')])
def test_strict_integer_capabilities_reject_malformed_native_and_manifest(field,value):
    cap=capability(2,1,1,27000)
    cap[field]=value
    with pytest.raises(RuntimeError):
        st.validate_core_split_capability(cap)
    manifest=dict(cap,native_sha256='a',kernel_sha256='b',sdk_library_sha256='c')
    with pytest.raises(ValueError):
        suite.preflight_code(manifest,30)


def test_impossible_old_prefix_new_wire_combination_is_rejected():
    with pytest.raises(ValueError):
        build.protocol_base(2,0,2)
    with pytest.raises(RuntimeError):
        st.validate_core_split_capability(capability(2,0,2,24000))


def test_old_sources_default_to_historical_protocol_and_new_source_macros_are_strict(tmp_path):
    root=tmp_path/'relocated source with spaces'
    headers=root/'csrc/kernels/dispatch/low_latency/common'
    headers.mkdir(parents=True)
    expected=dict(shared_pull_probe_mode=0,prefix_ready_revision=0,shared_pull_wire_revision=1)
    assert build.source_protocol_identity(root)==expected
    probe=headers/'ll_shared_pull_probe_config.h'
    core=headers/'ll_core_config.h'
    probe.write_text('#define DEEPEP_LL_SHARED_PULL_PROBE_MODE 2\n',encoding='utf-8')
    core.write_text('// Historical core protocol.\n',encoding='utf-8')
    assert build.source_protocol_identity(root)==dict(expected,shared_pull_probe_mode=2)
    core.write_text('#define DEEPEP_LL_PREFIX_READY_REVISION 1\n',encoding='utf-8')
    probe.write_text('#define DEEPEP_LL_SHARED_PULL_PROBE_MODE 2\n'
                     '#define DEEPEP_LL_SHARED_PULL_WIRE_REVISION 2\n',encoding='utf-8')
    assert build.source_protocol_identity(root)==dict(shared_pull_probe_mode=2,
        prefix_ready_revision=1,shared_pull_wire_revision=2)
    for bad in ('3','true','1oops'):
        core.write_text(f'#define DEEPEP_LL_PREFIX_READY_REVISION {bad}\n',encoding='utf-8')
        with pytest.raises(ValueError):
            build.source_protocol_identity(root)


@pytest.mark.parametrize('mode,prefix,wire,base',VALID)
def test_preflight_checks_native_protocol_fields_and_manifest_before_device_start(
        tmp_path,monkeypatch,mode,prefix,wire,base):
    files=[tmp_path/name for name in ('_C.test.so','libdeepep_kernels.so','libshmem.so')]
    for path in files:
        path.write_bytes(path.name.encode())
    extension,kernel,sdk=files
    native=types.SimpleNamespace(__file__=str(extension),low_latency_stage_timer_enabled=False,
        low_latency_core_protocol=base+30,low_latency_implementation='g4-r0-prefix',
        low_latency_ranks_per_group=4,shmem_library_path=lambda:str(sdk))
    attributes={'shared_pull_probe_mode':mode,'prefix_ready_revision':prefix,'shared_pull_wire_revision':wire}
    for name,value in attributes.items():
        setattr(native,'low_latency_'+name,value)
    module=types.ModuleType('deep_ep._native')
    module.require_native=lambda purpose:native
    monkeypatch.setitem(sys.modules,'deep_ep._native',module)
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        if path.as_posix()=='/proc/self/maps':
            return f'7f-8f r--p 00000000 08:01 99 {kernel.as_posix()}\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    manifest=dict(native_sha256=build.sha(extension),kernel_sha256=build.sha(kernel),
                  sdk_library_sha256=build.sha(sdk),**attributes)
    exec(suite.preflight_code(manifest,30),{})
    for field in attributes:
        changed=dict(manifest)
        if field=='shared_pull_probe_mode':
            changed[field]=(mode+1)%3
        elif field=='prefix_ready_revision':
            changed[field]=1-prefix
        else:
            changed[field]=3-wire
        try:
            code=suite.preflight_code(changed,30)
        except ValueError:
            continue  # An impossible combination is rejected before loading native.
        with pytest.raises(AssertionError):
            exec(code,{})
    if prefix==0:
        delattr(native,'low_latency_prefix_ready_revision')
        delattr(native,'low_latency_shared_pull_wire_revision')
        old_manifest={key:value for key,value in manifest.items()
                      if key not in ('prefix_ready_revision','shared_pull_wire_revision')}
        exec(suite.preflight_code(old_manifest,30),{})


@pytest.mark.parametrize('field',('prefix_ready_revision','shared_pull_wire_revision'))
def test_all_rank_capability_gather_rejects_revision_mismatch(field):
    cap=capability(2,1,1,27000)
    other=dict(cap)
    other[field]+=1
    with pytest.raises(RuntimeError):
        st.validate_matching_capabilities([cap,other])
