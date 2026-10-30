"""Ensure replacement changes observation only, across every active algorithm."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reference_business_checks',Path(__file__).with_name('business_contract.py'))
business=importlib.util.module_from_spec(spec);spec.loader.exec_module(business)


@pytest.mark.parametrize('relative',business.projection.ALGORITHMS)
def test_every_business_token_and_dependency_is_preserved(relative):
    business.assert_unchanged(relative)


def test_sdk_lock_gitlink_and_wire_are_unchanged():
    for relative in ('dependencies.lock.json','csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h',
                     'csrc/kernels/dispatch/low_latency/common/ll_core_config.h'):
        prior=subprocess.check_output(['git','-C',str(ROOT),'show','e4d114c:'+relative])
        assert (ROOT/relative).read_bytes().replace(b'\r\n',b'\n')==prior.replace(b'\r\n',b'\n')
    assert subprocess.check_output(['git','-C',str(ROOT),'ls-tree','HEAD','third-party/shmem'])==subprocess.check_output(
        ['git','-C',str(ROOT),'ls-tree','e4d114c','third-party/shmem'])


def test_diagnostic_insertions_cannot_hide_business_waits_or_payload_access():
    for relative in business.projection.ALGORITHMS:
        text=(ROOT/relative).read_text(encoding='utf-8')
        additions=re.findall(re.escape(business.reference.BEGIN)+r'(.*?)'+re.escape(business.reference.END),text,re.S)
        for block in additions:
            assert not re.search(r'\b(?:DataCopy|DataCopyPad|PipeBarrier|SetFlag|WaitFlag|GetValue|SetValue|SyncFunc|Sync)\s*(?:<|\()',block),relative


def test_old_active_device_and_python_implementations_are_removed():
    migration=json.loads((ROOT/'tests/ll_reference_profile/migration.json').read_text())
    assert all(not (ROOT/path).exists() for path in migration['removed_legacy_paths'])
    for folder in ('csrc/kernels/dispatch/low_latency','csrc/kernels/combine/low_latency','deep_ep/utils'):
        for path in (ROOT/folder).rglob('*'):
            if path.suffix not in ('.h','.hpp','.cpp','.py'):continue
            text=path.read_text(encoding='utf-8')
            assert not re.search(r'LlStageTimer|LlExpertTimer|LL_SEG_|LL_SPARSE_|ll_stage_timer|ll_expert_timer',text),path
    for path in (ROOT/'csrc/kernels/combine/low_latency').rglob('*.h'):
        assert 'GetSystemCycle' not in path.read_text(encoding='utf-8')


def test_original_sampling_and_route_algorithms_have_separate_identities():
    st=(ROOT/'tests/elastic/test_ep_low_latency.py').read_text(encoding='utf-8')
    old=subprocess.check_output(['git','-C',str(ROOT),'show','e4d114c:tests/elastic/test_ep_low_latency.py']).decode()
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(st),functions(old)
    for name in ('make_routes','verify_dispatch','expected_combine','prepare_experts','kernel_durations','sample_stats','summarize_kernel','profile_route_identity','check_route_reuse'):
        assert a[name]==b[name],name
    assert 'initial-reduce-exp100-batch-v3' in st
    historical=(ROOT/'tests/ll_stage_timer/no_timer_ab.py').read_text(encoding='utf-8')
    assert '439d4597a52b657858217bac7fc1e965f22507f7' in historical
    assert 'exp-reduce-stable-tail-v2' in historical
