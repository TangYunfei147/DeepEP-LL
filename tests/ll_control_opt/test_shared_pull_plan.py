"""Frozen quick-probe selection, protocol guard and bounded result handoff."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


suite=load('shared_pull_suite','tests/ll_control_opt/suite.py')
report=load('shared_pull_export','tests/ll_control_opt/export_report.py')
st=load('shared_pull_st','tests/elastic/test_ep_low_latency.py')


def test_quick_probe_is_three_verified_modes_same_sdk_six_complete_calls(tmp_path):
    args=suite.parser().parse_args(['--commits','tests/ll_control_opt/commits-shared-pull.json',
        '--devices',','.join(map(str,range(16))),'--num-nodes','1','--node-rank','0',
        '--master-addr','127.0.0.1','--master-port','29600','--shmem-port','29601',
        '--resume','--send-cores','30','--seeds','0','--rounds','2',
        '--order','alternating','--run-prefix','shared-pull-001'])
    plan=suite.make_plan(args)
    assert len(plan['phases']['build'])==3 and len(plan['phases']['run'])==6
    assert len({d['dependency_id'] for d in plan['dependencies'].values()})==1
    entries=plan['commits'];assert [e['label'] for e in entries]==['baseline','fence-control','shared-pull']
    assert entries[2]['comparison_base']==entries[1]['revision']
    for entry,mode in zip(entries,(0,1,2)):
        if mode:
            header=subprocess.check_output(['git','-C',str(ROOT),'show',entry['revision']+
                ':csrc/kernels/dispatch/low_latency/common/ll_shared_pull_probe_config.h'],text=True)
            assert f'#define DEEPEP_LL_SHARED_PULL_PROBE_MODE {mode}\n' in header
        code=suite.preflight_code(dict(native_sha256='a',kernel_sha256='b',sdk_library_sha256='c',shared_pull_probe_mode=mode),30)
        assert f'assert probe=={mode}' in code and '19030,23030,24030' in code
    diff=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',entries[1]['revision'],entries[2]['revision']],text=True)
    assert diff.strip()=='csrc/kernels/dispatch/low_latency/common/ll_shared_pull_probe_config.h'
    runs=plan['phases']['run']
    assert [r['label'] for r in runs[:3]]==[r['label'] for r in reversed(runs[3:])]
    for run in runs:
        assert run['measured_route_seed']==2
        assert not {'--ll-stage-timer','--skip-check','--dispatch-only'} & set(run['command'])
    measurements={(r,0):{e['label']:dict(dispatch=85+i,combine=138+i*.01) for i,e in enumerate(entries)} for r in (1,2)}
    rows=suite.comparisons(entries,measurements,plan['order']);assert len(rows)==12
    source=tmp_path/'comparison.csv'
    with source.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    out=tmp_path/'analysis-report.txt';size=report.export(source,out)
    assert size==len(out.read_bytes())<=65535


@pytest.mark.parametrize('mode,base',[(0,19000),(1,23000),(2,24000)])
def test_st_rejects_mixed_descriptor_protocol(mode,base):
    cap=dict(send_cores=30,recv_cores=26,active_aiv=56,shared_pull_probe_mode=mode,core_protocol=base+30)
    st.validate_core_split_capability(cap)
    cap['core_protocol']+=1000
    with pytest.raises(RuntimeError):st.validate_core_split_capability(cap)


def test_build_mode_is_read_from_source_and_malformed_header_is_rejected(tmp_path):
    assert suite.build.shared_pull_probe_mode(tmp_path)==0
    header=tmp_path/'csrc/kernels/dispatch/low_latency/common/ll_shared_pull_probe_config.h';header.parent.mkdir(parents=True)
    for mode in (0,1,2):
        header.write_text(f'#define DEEPEP_LL_SHARED_PULL_PROBE_MODE {mode}\n')
        assert suite.build.shared_pull_probe_mode(tmp_path)==mode
    header.write_text('#define DEEPEP_LL_SHARED_PULL_PROBE_MODE unexpected\n')
    with pytest.raises(ValueError):suite.build.shared_pull_probe_mode(tmp_path)
