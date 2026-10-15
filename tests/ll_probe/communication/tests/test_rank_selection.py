"""Explicit global-rank routing, cross-host volume and validation contracts."""
import json
import subprocess
from pathlib import Path
import pytest
from tests.ll_probe.profiling.profile_probe import parser,plan
from tests.ll_probe.profiling.bandwidth_model import comm_edges,volume
from tests.ll_probe.profiling.analyze_profile import analyze
from tests.ll_probe.profiling.tests.test_deployment_reports import reports

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'tests/ll_probe'


def args(extra):
    return parser().parse_args(['--kind','comm','--endpoint','tcp://127.0.0.1:29501','--output-dir','results/plan',*extra])


@pytest.mark.parametrize('scenario,extra,expected',[
 (1,['--targets','16'],[(0,16)]),
 (3,['--targets','31','16','18'],[(0,16),(0,18),(0,31)]),
 (2,['--sources','16','31'],[(16,0),(31,0)])])
def test_cli_and_cross_host_accounting(scenario,extra,expected):
    a=args(['--scenario',str(scenario),*extra]);job=plan(a,dict(WORLD_SIZE='32',RANK='0',LOCAL_RANK='0'))
    assert comm_edges(32,scenario,a.fanout,a.targets,a.sources)==expected
    assert job['world']==32
    assert int(job['argv'][-1])==sum(1<<v for v in sorted({s if scenario==2 else d for s,d in expected}))
    vol,_=volume(32,vars(a),16)
    assert vol['inter_host_sent_bytes']==len(expected)*a.bytes
    assert vol['intra_host_sent_bytes']==0


@pytest.mark.parametrize('extra',[
 ['--scenario','1','--targets','1','2'],['--scenario','3','--targets','0'],
 ['--scenario','3','--targets','16','16'],['--scenario','3','--targets','32'],
 ['--scenario','3','--targets','-1'],['--scenario','3','--targets','16','--fanout','1'],
 ['--scenario','2','--targets','1'],['--scenario','1','--sources','1'],
 ['--scenario','4','--sources','1']])
def test_invalid_selection_rejected(extra):
    with pytest.raises(ValueError):plan(args(extra),dict(WORLD_SIZE='32',RANK='0',LOCAL_RANK='0'))


def test_real_cpp_predicates_with_rank31_bit(tmp_path):
    s=(P/'communication/kernel/comm_kernel.cpp').read_text();a=s.index('template<uint32_t Scenario>');b=s.index('template<uint32_t Scenario>',a+1)
    device=s[a:b]
    s=(P/'communication/host/comm_host.cpp').read_text();host=s[s.index('static bool edge('):s.index('static uint8_t pattern')]
    cpp=tmp_path/'selection.cpp';cpp.write_text('#include <cassert>\n#include <cstdint>\n#define __aicore__\n'+device+host+r"""
int main(){
 unsigned mask=(1U<<16)|(1U<<18)|(1U<<31);
 for(unsigned s=0;s<32;++s)for(unsigned d=0;d<32;++d){
  assert(HasEdge<1>(s,d,32,31,1U<<31)==(s==0&&d==31));
  assert(HasEdge<3>(s,d,32,3,mask)==(s==0&&(d==16||d==18||d==31)));
  assert(HasEdge<2>(s,d,32,31,mask)==(d==0&&(s==16||s==18||s==31)));
  assert(edge(1,s,d,31,1U<<31)==HasEdge<1>(s,d,32,31,1U<<31));
  assert(edge(2,s,d,31,mask)==HasEdge<2>(s,d,32,31,mask));
  assert(edge(3,s,d,3,mask)==HasEdge<3>(s,d,32,3,mask));
 }
}
""");exe=tmp_path/'selection'
    subprocess.run(['g++','-std=c++17',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_profile_volume_and_mixed_rank_lists(tmp_path):
    paths=reports(tmp_path,32)
    for path in paths:
        obj=json.loads(path.read_text());obj['configuration'].update(scenario=2,sources=[16,31])
        path.write_text(json.dumps(obj))
    result=analyze(paths)
    assert result['job_wire_bytes']==64
    assert result['bandwidth_summary']['volume']['inter_host_sent_bytes']==64
    obj=json.loads(paths[-1].read_text());obj['configuration']['sources']=[17,31]
    paths[-1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):analyze(paths)


@pytest.mark.parametrize('scenario,selected',[(1,[31]),(3,[1,16,31]),(2,[16,31])])
def test_event_csv_selection(tmp_path,scenario,selected):
    import importlib.util
    spec=importlib.util.spec_from_file_location('selected_event',P/'communication/tests/analyze_comm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mask=sum(1<<v for v in selected)
    fanout=len(selected) if scenario==3 else 31
    paths=[]
    for rank in range(32):
        count=(len(selected) if rank==0 else 0) if scenario in (1,3) else int(rank in selected)
        path=tmp_path/f'rank{rank}.csv'
        path.write_text('sample,rank,world,scenario,layer,bytes_per_edge,rank_wire_bytes,fanout,peer_mask,event_us,api_us,job_join_us\n'+
                        f'0,{rank},32,{scenario},3,32,{count*32},{fanout},{mask},1,2,3\n')
        path.with_suffix('.log').write_text('PASS receiver checks and normal cleanup\n');paths.append(path)
    assert module.analyze(paths)['job_wire_bytes']==32*len(selected)
