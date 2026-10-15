"""Check actual Host/device routing predicates and rank-32 fanout accounting."""
from pathlib import Path
import subprocess
import json
import importlib.util
import pytest
from tests.ll_probe.profiling import profile_probe
from tests.ll_probe.profiling.analyze_profile import analyze
from tests.ll_probe.profiling.tests.test_deployment_reports import reports

ROOT=Path(__file__).resolve().parents[4]
PROBE=ROOT/'tests/ll_probe'


def test_host_and_device_graphs(tmp_path):
    kernel=(PROBE/'communication/kernel/comm_kernel.cpp').read_text()
    device=kernel[kernel.index('template<uint32_t Scenario>'):kernel.index('template<uint32_t Scenario>',kernel.index('template<uint32_t Scenario>')+1)]
    host=(PROBE/'communication/host/comm_host.cpp').read_text()
    host=host[host.index('static bool edge('):host.index('static uint8_t pattern')]
    cpp=tmp_path/'fanout.cpp'
    cpp.write_text('#include <cassert>\n#include <cstdint>\n#include <initializer_list>\n#include "tests/ll_probe/communication/tiling.h"\n#define __aicore__\n'+device+host+ r"""
int main() {
    static_assert(sizeof(FourLayerTiling)==32);
    for(unsigned world: {2U,4U,8U,16U,32U}) for(unsigned n=1;n<world;++n) {
        unsigned hits[4]={};
        for(unsigned s=0;s<world;++s) for(unsigned d=0;d<world;++d) {
            bool actual[]={HasEdge<1>(s,d,world,n),HasEdge<2>(s,d,world,n),
                           HasEdge<3>(s,d,world,n),HasEdge<4>(s,d,world,n)};
            for(unsigned k=1;k<=4;++k) { assert(actual[k-1]==edge(k,s,d,n));hits[k-1]+=actual[k-1]; }
            assert(actual[2]==(s==0 && d>=1 && d<=n));
        }
        assert(hits[0]==1 && hits[1]==world-1 && hits[2]==n && hits[3]==world*(world-1));
    }
}
""")
    exe=tmp_path/'fanout'
    subprocess.run(['g++','-std=c++17','-I',str(ROOT),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize('world',[2,4,8,16,32])
def test_plan_default_and_selected_receivers(world):
    base=['--kind','comm','--scenario','3','--endpoint','tcp://127.0.0.1:29501','--output-dir','results/plan']
    env=dict(WORLD_SIZE=str(world),RANK=str(world-1),LOCAL_RANK='0')
    full=profile_probe.plan(profile_probe.parser().parse_args(base),env)
    assert full['argv'][-3]==str(world-1)
    n=min(3,world-1)
    selected=profile_probe.plan(profile_probe.parser().parse_args(base+['--fanout',str(n)]),env)
    assert selected['measurement_scope']['receiver_ranks']==list(range(1,n+1))
    assert selected['world']==world


@pytest.mark.parametrize('scenario,n',[(3,0),(3,-1),(3,32),(1,3),(2,3),(4,3)])
def test_invalid_selection(scenario,n):
    args=profile_probe.parser().parse_args(['--kind','comm','--scenario',str(scenario),'--fanout',str(n),
             '--endpoint','tcp://127.0.0.1:29501','--output-dir','results/plan'])
    with pytest.raises(ValueError):profile_probe.plan(args,dict(WORLD_SIZE='32',RANK='0',LOCAL_RANK='0'))


def test_32rank_profile_uses_three_edges_and_rejects_mixed_selection(tmp_path):
    paths=reports(tmp_path,32)
    for path in paths:
        obj=json.loads(path.read_text());obj['configuration'].update(scenario=3,fanout=3)
        path.write_text(json.dumps(obj))
    assert analyze(paths)['job_wire_bytes']==3*32
    obj=json.loads(paths[-1].read_text());obj['configuration']['fanout']=4
    paths[-1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):analyze(paths)


def test_event_bytes_include_only_selected_receivers(tmp_path):
    spec=importlib.util.spec_from_file_location('event_fanout',PROBE/'communication/tests/analyze_comm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[]
    for rank in range(32):
        path=tmp_path/f'rank{rank}.csv'
        path.write_text('sample,rank,world,scenario,layer,bytes_per_edge,rank_wire_bytes,fanout,event_us,api_us,job_join_us\n'+
                        f'0,{rank},32,3,3,32,{96 if rank==0 else 0},3,1,2,3\n')
        path.with_suffix('.log').write_text('PASS receiver checks and normal cleanup\n');paths.append(path)
    assert module.analyze(paths)['job_wire_bytes']==96


def test_matrix_plan_matches_selected_edge_count():
    from tests.ll_probe.four_layers.plan_matrix import make_case
    case=make_case(32,3,15392,fanout=3)
    assert case['edge_count']==3
    assert case['job_wire_bytes']==case['max_sender_wire_bytes']==15392*3
