"""Bandwidth measurements use exact bytes, complete coverage and local clocks."""
import copy
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


timer=load('bandwidth_raw_parser','deep_ep/utils/ll_stage_timer_bandwidth.py')
report=load('bandwidth_fixture_math','tests/ll_control_opt/report_bandwidth.py')


def raw(mode=1,rank=0,iteration=0):
    # Fixture control ownership uses the independent host formula; all data
    # packets sum to EP*B*K, with both peer/self and multi-interval relay cases.
    useful,wire=report.data_layout(report.profile.suite.SHAPE)
    result=[]
    for core in range(56):
        origin=rank*10_000_000+core*100_000+iteration*1_000_000_000
        words=[0]*128
        words[:8]=[timer.MAGIC,17,1,56,core,mode|(30<<8)|((3 if core<30 else 2)<<16)|(8<<24),0,6|(6<<8)|(1<<16)]
        for name,tick in dict(ENTRY=0,INIT_DONE=1000,WORK_DONE=100000).items():
            index=timer.POINTS.index(name);words[6]|=1<<index;words[8+index]=origin+tick
        for index,stage in enumerate(timer.STAGES):
            values=dict.fromkeys(timer.FIELDS,0)
            if stage=='COUNT_CONTROL':values.update(report.expected_control(rank,core))
            elif stage=='PREFIX_LOCAL' and core==30:values.update(self_bytes=3104,self_operations=3)
            else:
                peer,self_ops=(24,0) if stage=='REMOTE_FIRST_HOP' and core<30 else (
                    (12,0) if stage=='SOURCE_DIRECT' and core<30 else (
                    (14,6) if stage=='GROUP_LOCAL' and core>=30 else ((6,2) if stage=='RELAY' else (0,0))))
                for target,count in (('peer',peer),('self',self_ops)):
                    values.update({target+'_operations':count,target+'_bytes':count*wire,target+'_payload_bytes':count*useful})
            ops=values['peer_operations']+values['self_operations']
            if ops:
                first=origin+2000+index*10000
                values.update(first_issue_tick=first,last_issue_tick=first+(100 if ops>1 else 0))
                if mode==6:
                    values.update(last_completion_tick=first+1000,completed_elapsed_ticks=500 if stage=='RELAY' else 1000,
                        completed_intervals=2 if stage=='RELAY' else 1,pre_drain_ticks=10,end_drain_ticks=20,
                        empty_control_ticks=5,completed_operations=ops)
            words[32+16*index:48+16*index]=[values[field] for field in timer.FIELDS]
        result.append([words[i:i+8] for i in range(0,128,8)])
    return result


def setword(data,core,word,value):data[core][word//8][word%8]=value


def setmetric(data,core,stage,field,value):
    setword(data,core,32+timer.STAGES.index(stage)*16+timer.FIELDS.index(field),value)


def test_completion_rates_and_repeated_intervals_are_not_fake_trace_events():
    parsed=timer.parse_records(raw(6),'dispatch',1e9)
    row=next(row for row in parsed['bandwidth_metrics'] if row['core']==0 and row['stage']=='RELAY')
    assert row['peer_bytes']==6*4096 and row['peer_payload_bytes']==6*3696
    assert row['completed_elapsed_us']==.5 and row['peer_gbps']==49.152
    assert row['completed_intervals']==2 and row['completed_operations']==8
    assert row['completed_elapsed_ticks'] < row['last_completion_tick']-row['first_issue_tick']
    assert {row['stage'] for row in parsed['intervals']}=={'INIT','KERNEL'}


def test_observe_has_counts_but_never_completion_bandwidth():
    parsed=timer.parse_records(raw(),'dispatch',1e9)
    assert any(row['peer_bytes'] for row in parsed['bandwidth_metrics'])
    assert all(row['peer_gbps'] is None and row['completed_elapsed_us'] is None for row in parsed['bandwidth_metrics'])
    no_frequency=timer.parse_records(raw(6),'dispatch')
    assert all('peer_gbps' not in row for row in no_frequency['bandwidth_metrics'])


@pytest.mark.parametrize('fault',['missing_op','wrong_mode','bytes','useful','coverage','intervals','end','outside','reserved','observe_completion','phase_total'])
def test_reject_inconsistent_measurements(fault):
    data=raw(1 if fault=='observe_completion' else 6)
    stage='REMOTE_FIRST_HOP'
    if fault=='missing_op':setmetric(data,0,stage,'peer_operations',0)
    elif fault=='wrong_mode':setword(data,0,5,4|(30<<8)|(3<<16)|(8<<24))
    elif fault=='bytes':setmetric(data,0,stage,'peer_bytes',0)
    elif fault=='useful':setmetric(data,0,stage,'peer_payload_bytes',10**9)
    elif fault=='coverage':setmetric(data,0,stage,'completed_operations',23)
    elif fault=='intervals':setmetric(data,0,stage,'completed_intervals',2)
    elif fault=='end':setmetric(data,0,stage,'end_drain_ticks',1001)
    elif fault=='outside':setmetric(data,0,stage,'last_completion_tick',200000)
    elif fault=='reserved':setmetric(data,0,stage,'reserved',1)
    elif fault=='observe_completion':setmetric(data,0,stage,'pre_drain_ticks',1)
    elif fault=='phase_total':
        setmetric(data,0,stage,'pre_drain_ticks',98000)
    with pytest.raises(ValueError):timer.parse_records(data,'dispatch',1e9)


def test_empty_phase_predrain_retained_without_bandwidth():
    data=raw(6)
    for field in timer.FIELDS:setmetric(data,0,'SOURCE_DIRECT',field,0)
    setmetric(data,0,'SOURCE_DIRECT','pre_drain_ticks',10)
    row=next(row for row in timer.parse_records(data,'dispatch',1e9)['bandwidth_metrics']
             if row['core']==0 and row['stage']=='SOURCE_DIRECT')
    assert row['pre_drain_us']==.01 and row['peer_gbps'] is None and not row['time_measured']


@pytest.mark.parametrize('frequency',[0,-1,True,float('nan'),float('inf')])
def test_frequency_must_be_explicit_finite_positive(frequency):
    with pytest.raises(ValueError):timer.parse_records(raw(),'dispatch',frequency)


def test_report_byte_formula_matches_actual_native_layout_and_count_plan(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('CPU C++ compiler unavailable')
    source=tmp_path/'wire.cpp'
    source.write_text(r'''
#include <iostream>
#include "csrc/kernels/dispatch/low_latency/common/ll_group_local_send.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h"
int main() {
 using namespace DispatchDedup;
 for(unsigned h:{128U,256U,3584U,4096U}) for(unsigned k:{4U,8U,16U}) {
   auto d=DispatchServerDedupDataLayout(256,h,k,384,EpServerDedupInputMode::PrequantizedFp8Packs);
   std::cout<<"D "<<h<<' '<<k<<' '<<h+d.scaleBytes<<' '<<d.tokenStride<<'\n';
 }
 for(unsigned rank=0;rank<16;++rank) for(unsigned core=0;core<56;++core) {
   auto p=LlMakeGroupCountPlan(384,16,rank,core);
   unsigned peerBytes=0,selfBytes=0,peerOps=0,selfOps=0;
   for(unsigned part=0;part<p.size;++part) {
     auto r=p.ranges[part];
     for(unsigned expert=r.begin;expert<r.end;) {
       unsigned dst=expert/24,count=std::min(r.end-expert,24-expert%24);
       if(dst==rank){selfBytes+=count*32;++selfOps;}
       else {peerBytes+=count*32;++peerOps;}
       expert+=count;
     }
   }
   std::cout<<"C "<<rank<<' '<<core<<' '<<peerBytes<<' '<<selfBytes<<' '<<peerOps<<' '<<selfOps<<'\n';
 }
}
''',encoding='utf-8')
    executable=tmp_path/'wire'
    subprocess.run([compiler,'-std=c++17','-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26',
                    '-I',str(ROOT),str(source),'-o',str(executable)],check=True)
    output=subprocess.check_output([str(executable)],text=True,timeout=30)
    for line in output.splitlines():
        kind,*values=line.split();values=list(map(int,values))
        if kind=='D':
            h,k,useful,wire=values
            assert report.data_layout(dict(H=h,num_topk=k))==(useful,wire)
        else:
            rank,core,*expected=values;actual=report.expected_control(rank,core)
            assert [actual[key] for key in ('peer_bytes','self_bytes','peer_operations','self_operations')]==expected


def test_native_and_python_field_dictionaries_agree():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text()
    for name,expected in [('kPointNames',timer.POINTS),('kBandwidthStageNames',timer.STAGES),('kBandwidthFieldNames',timer.FIELDS)]:
        body=re.search(name+r'\[\]\s*=\s*\{(.*?)\};',source,re.S).group(1)
        assert tuple(re.findall(r'"([A-Za-z_]+)"',body))==expected
