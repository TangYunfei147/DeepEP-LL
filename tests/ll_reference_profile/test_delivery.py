"""Target delivery guards; Host verification is exercised without a driver."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_frozen_fingerprint_matches_the_build_function():
    verify=load('reference_verify_fingerprint',Path(__file__).with_name('verify_installed.py'))
    build=load('reference_build_fingerprint',ROOT/'build_support.py')
    assert verify.source_fingerprint(ROOT)==build.source_fingerprint()


@pytest.mark.parametrize('fault',[None,'source','mode','schema','implementation','requested_split'])
def test_host_preflight_checks_project_identity_without_npu_or_sdk_gate(tmp_path,monkeypatch,fault):
    verify=load('reference_verify_test',Path(__file__).with_name('verify_installed.py'))
    prefix=tmp_path/'venv';package=prefix/'lib/site-packages/deep_ep';package.mkdir(parents=True)
    info=dict(ll_reference_profile=True,ll_reference_profile_schema=7,ll_instrumentation_source='reference-mte7',
        dispatch_launch_bytes=144,combine_launch_bytes=88,source_sha256='a'*64,ll_dispatch_impl='current',
        ll_send_cores=28,ll_recv_cores=28,ll_small_b_prefix_cores=0,shmem='opaque invalid provider metadata')
    native=SimpleNamespace(low_latency_reference_profile_enabled=True,low_latency_reference_profile_layout=lambda:dict(schema=7),
                           low_latency_dispatch_impl='current')
    native.low_latency_core_config=lambda:pytest.fail('Host preflight must not query any NPU')
    native.shmem_library_path=lambda:pytest.fail('Do not inspect or hash SDK libraries')
    for key in ('DEEPEP_LL_DISPATCH_IMPL','DEEPEP_LL_SEND_CORES','DEEPEP_LL_RECV_CORES','DEEPEP_LL_SMALL_B_PREFIX_CORES'):
        monkeypatch.delenv(key,raising=False)
    if fault=='source':info['source_sha256']='b'*64
    elif fault=='mode':native.low_latency_reference_profile_enabled=False
    elif fault=='schema':info['ll_reference_profile_schema']=6
    elif fault=='implementation':native.low_latency_dispatch_impl='cb59'
    elif fault=='requested_split':monkeypatch.setenv('DEEPEP_LL_SEND_CORES','30')
    (package/'_build_info.json').write_text(json.dumps(info))
    monkeypatch.setitem(sys.modules,'deep_ep',SimpleNamespace(__file__=str(package/'__init__.py')))
    monkeypatch.setitem(sys.modules,'deep_ep._native',SimpleNamespace(require_native=lambda *a:native))
    monkeypatch.setattr(sys,'prefix',str(prefix));monkeypatch.setattr(verify,'source_fingerprint',lambda root:'a'*64)
    if fault:
        with pytest.raises(RuntimeError):verify.verify()
    else:
        result=verify.verify();assert result['host_only'] and not result['device_queried'] and result['schema']==7


def test_one_click_orders_clean_build_and_host_checks_before_capture():
    script=Path(__file__).with_name('one_click.sh').read_text(encoding='utf-8')
    commands=['submodule update --init --recursive --depth 1 -- third-party/shmem',
              'scripts/build_shmem.py','scripts/clean_build_artifacts.py --wheel','-m build --wheel --no-isolation',
              '-m pip install --no-deps --force-reinstall','verify_installed.py >"$group/build-ready.json"',
              'tests/ll_stage_timer/run.py','tests/ll_reference_profile/analyze.py']
    positions=[script.index(command) for command in commands]
    assert positions==sorted(positions)
    assert 'clone --no-hardlinks --dissociate --no-checkout' in script
    assert '--no-recurse-submodules' in script and '-c submodule.recurse=false checkout' in script
    assert 'source scripts/ll_reference_env.sh' in script
    assert 'if [[ $action != --skip-build ]]' in script and 'if [[ $action == --build-only ]]' in script
    assert '--dispatch-only --ll-reference-profile --reference-iters 1' in script
    assert '--ll-stage-timer' not in script and '--recurse-submodules' not in script
    assert 'false\nfi' in script and '\nexit ' not in script


def test_report_paths_are_project_based_and_no_legacy_decoder_is_loaded():
    for name in ('analyze.py','export_xlsx.py','report_tables.py','verify_installed.py'):
        text=Path(__file__).with_name(name).read_text(encoding='utf-8')
        assert 'C:/Users/' not in text and '/home/t30064812/' not in text
        assert 'utils/ll_stage_timer' not in text and 'utils.ll_stage_timer' not in text
