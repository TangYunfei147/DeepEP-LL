"""Relay issue/completion mode integrity and distinct reporting."""
import importlib.util
import copy
import json
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

fixtures=load('relay_segments','tests/ll_stage_timer/test_segments.py')
analysis=load('relay_analysis','tests/ll_stage_timer/analyze.py')
timer=analysis.timer


def raw(mode=1):
    data=fixtures.raw(schema=10,mode=mode)
    words=[v for row in data[0] for v in row]
    old=[tuple(words[8+i*2:10+i*2]) for i in range(words[6])]
    traces=[p for p in old if (p[0]&0xffffffff)<0x8000]
    metrics=[p for p in old if (p[0]&0xffffffff)>=0x8000]
    phases={66:[(225,235),(325,345)]}
    if mode==3:phases.update({67:[(210,220),(310,320)],68:[(220,260),(320,390)]})
    for stage,intervals in phases.items():
        for begin,end in intervals:traces.extend([(stage*2,begin),(stage*2+1,end)])
        metrics.append(((len(intervals)<<32)|(0x8000+stage),sum(e-b for b,e in intervals)))
    metrics.append(((3<<32)|(0x8000+69),0))
    entries=traces+metrics
    words[6]=len(entries)
    words[8:]=[0]*(len(words)-8)
    for i,pair in enumerate(entries):words[8+i*2:10+i*2]=pair
    data[0]=[words[i:i+8] for i in range(0,len(words),8)]
    return data


@pytest.mark.parametrize('mode',[1,3])
def test_reports_issue_and_completion_separately(mode):
    decoded=timer.parse_records(raw(mode),'dispatch',1e9)
    assert decoded['boundary_mode']==('observe' if mode==1 else 'relay-complete')
    result=analysis.relay_timing.report([(dict(rank=0,iteration=0),decoded)])
    row=result['per_core'][0]
    assert row['groups']==2 and row['relay_writes']==3
    assert row['issue_ticks_total']==30 and row['issue_ticks_per_group']==15
    assert row['complete_ticks_total']==(110 if mode==3 else None)
    assert row['pre_drain_ticks_total']==(20 if mode==3 else None)
    assert result['per_core'][1]['status']=='no_relay'


def test_old_schema_is_unavailable_not_zero():
    decoded=timer.parse_records(fixtures.raw(schema=9),'dispatch')
    row=analysis.relay_timing.report([(dict(rank=0,iteration=0),decoded)])['per_core'][0]
    assert row['status']=='unavailable'
    assert row['issue_ticks_total'] is None and row['groups'] is None
    assert row['complete_kernel_ratio'] is None and row['relay_writes_per_group'] is None


def test_cumulative_ratio_and_fanout_use_same_core_rounds():
    first=timer.parse_records(raw(3),'dispatch',1e9)
    second=copy.deepcopy(first)
    for record in second['records']:
        if record['core']!=0: continue
        if record['stage']=='KERNEL':record['total_ticks']=1100
        if record['stage']=='RELAY_COMPLETE':record.update(total_ticks=330,calls=4)
        if record['stage']=='RELAY_ISSUE':record['calls']=4
        if record['stage']=='RELAY_PACKETS':record['calls']=9
    result=analysis.relay_timing.report([(dict(rank=0,iteration=0),first),
                                         (dict(rank=0,iteration=1),second)])
    row=result['per_core'][0]
    assert row['samples']==2 and row['complete_ticks_total']==440
    assert row['kernel_ticks_total']==11000
    assert row['complete_us_total']==pytest.approx(.44)
    assert row['kernel_us_total']==pytest.approx(11)
    assert row['complete_kernel_ratio']==pytest.approx(.04)
    assert row['relay_writes_per_group']==2
    text=analysis.relay_timing.markdown(result)
    assert '4.00%' in text and '累计KERNEL us' in text
    assert '不能直接判定正常模式' in text


def test_zero_relay_round_still_contributes_matching_kernel():
    first=timer.parse_records(raw(3),'dispatch')
    empty=timer.parse_records(fixtures.raw(schema=10,mode=3),'dispatch')
    result=analysis.relay_timing.report([(dict(rank=0,iteration=0),first),
                                         (dict(rank=0,iteration=1),empty)])
    row=result['per_core'][0]
    assert row['complete_ticks_total']==110 and row['kernel_ticks_total']==19800
    assert row['complete_kernel_ratio']==pytest.approx(110/19800)
    assert row['relay_writes_per_group']==1.5
    assert row['complete_us_total'] is None
    assert '累计完成 ticks' in analysis.relay_timing.markdown(result)
    no_relay=result['per_core'][1]
    assert no_relay['complete_kernel_ratio']==0
    assert no_relay['relay_writes_per_group'] is None


@pytest.mark.parametrize('kernel',[None,0])
def test_missing_or_zero_kernel_does_not_produce_ratio(kernel):
    decoded=timer.parse_records(raw(3),'dispatch')
    for record in decoded['records']:
        if record['core']==0 and record['stage']=='KERNEL':record['total_ticks']=kernel
    row=analysis.relay_timing.report([(dict(rank=0,iteration=0),decoded)])['per_core'][0]
    assert row['complete_kernel_ratio'] is None


def test_observe_completion_stays_unmeasured_and_duplicate_round_rejected():
    decoded=timer.parse_records(raw(1),'dispatch',1e9)
    sample=(dict(rank=0,iteration=0),decoded)
    row=analysis.relay_timing.report([sample])['per_core'][0]
    assert row['kernel_ticks_total']==9900 and row['complete_us_total'] is None
    assert row['complete_kernel_ratio'] is None and row['relay_writes_per_group']==1.5
    with pytest.raises(ValueError,match='Duplicate'):analysis.relay_timing.report([sample,sample])


def test_mixed_modes_and_missing_completion_rejected():
    data=raw(3)
    data[1][0][5]=(data[1][0][5]&~255)|1
    with pytest.raises(ValueError,match='Mixed relay'):timer.parse_records(data,'dispatch')
    data=raw(1)
    for core in data:core[0][5]=(core[0][5]&~255)|3
    with pytest.raises(ValueError,match='completion'):timer.parse_records(data,'dispatch')
    with pytest.raises(ValueError,match='Mixed relay'):
        timer.chrome_trace([dict(raw=raw(1),operation='dispatch',frequency_hz=1e9),
                            dict(raw=raw(3),operation='dispatch',frequency_hz=1e9)])


def test_new_st_capture_only_accepts_observe_while_old_relay_is_readable():
    with pytest.raises(ValueError):timer.StageTrace(None,'combine','relay-complete')
    st=load('relay_st','tests/elastic/test_ep_low_latency.py')
    for mode in ('relay-complete', 'drain'):
        with pytest.raises(SystemExit):
            st.make_parser().parse_args(['--ll-timer-mode',mode])
        args=st.make_parser().parse_args(['--ll-stage-timer','--dispatch-only','--skip-perf-test'])
        args.ll_timer_mode=mode
        with pytest.raises(ValueError,match='only observe'):
            st.validate_args(args,16)
    assert timer.parse_records(raw(3),'dispatch')['boundary_mode']=='relay-complete'


def test_analyzer_exports_distinct_relay_files(tmp_path):
    source=tmp_path/'source';source.mkdir()
    saved=dict(raw=raw(3),operation='dispatch',frequency_hz=1e9,
        metadata=dict(world_size=1,rank=0,iteration=0,config={},correctness='passed'))
    (source/'sample_0.json').write_text(json.dumps(saved))
    output=tmp_path/'out'
    analysis.process_directory(source,output,1e9,0)
    assert (output/'relay_samples.csv').is_file()
    assert (output/'relay_timing.csv').is_file()
    assert '不能相加' in (output/'relay_timing.md').read_text(encoding='utf-8')
    result=json.loads((output/'summary.json').read_text())
    assert result['relay_timing']['mode']=='relay-complete'
    row=result['relay_timing']['per_core'][0]
    assert row['kernel_ticks_total']==9900
    assert row['complete_kernel_ratio']==pytest.approx(110/9900)
    assert 'relay_writes_per_group' in (output/'relay_timing.csv').read_text(encoding='utf-8-sig')


def test_observe_and_completion_are_not_compared_as_speedup():
    parity=load('relay_parity','tests/ll_stage_timer/test_parity.py')
    left=parity.identity()
    left.update(core_protocol=10010,send_cores=10,recv_cores=46,dispatch_recv_cores=56,
                stage_timer_schema=10,timer_mode='observe')
    right=dict(left,timer_mode='relay-complete')
    with pytest.raises(ValueError,match='timer_mode'):parity.comparison.check_identity(left,right)


@pytest.mark.parametrize('enabled',[False,True])
def test_sparse_native_rejects_serializing_modes_and_adds_no_barrier(tmp_path,enabled):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('C++ compiler unavailable; native sparse mode guard requires target mock')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint64_t clock=1;inline unsigned sets=0,waits=0;
inline uint64_t GetSystemCycle(){return clock++;}
inline uint32_t GetBlockIdx(){return 0;}inline uint32_t GetBlockNum(){return 56;}
inline void Trap(){throw std::runtime_error("trap");}
enum class HardEvent{MTE3_S};struct TPipe{int FetchEventID(HardEvent){return 0;}};
template<HardEvent E>void SetFlag(int){++sets;}
template<HardEvent E>void WaitFlag(int){++waits;clock+=10;}
enum class CacheLine{SINGLE_CACHE_LINE};enum class DcciDst{CACHELINE_OUT};
template<class T>struct GlobalTensor{void SetGlobalBuffer(T*,unsigned){}};
template<class T,CacheLine C,DcciDst D>void DataCacheCleanAndInvalid(GlobalTensor<T>&){}
}
''',encoding='utf-8')
    source=tmp_path/'guard.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
int main(){
 for(unsigned mode:{1U,2U,3U}) {
  using namespace LlStageTimer;
  std::vector<uint64_t> storage(kWordsPerCore,0);
  DeviceTimer timer;AscendC::sets=AscendC::waits=0;
  bool rejected=false;
  try{timer.Init(reinterpret_cast<uint8_t*>(storage.data()),1,mode);}
  catch(const std::runtime_error&){rejected=true;}
#ifdef DEEPEP_LL_STAGE_TIMER
  assert(rejected==(mode!=1));
  if(rejected)continue;
#else
  assert(!rejected);
#endif
  unsigned effects=0;
  LL_SEG_RELAY_COMPLETION(++effects);
  LL_SEG_COUNT(RELAY_PACKETS,++effects);
  assert(effects==0);
  timer.Mark<ENTRY>();timer.Mark<INIT_DONE>();
  timer.Mark<FIRST_SEND>();timer.Mark<SEND_DONE>();timer.Mark<WORK_DONE>();
  timer.Flush();
  assert(AscendC::sets==0 && AscendC::waits==0);
#ifdef DEEPEP_LL_STAGE_TIMER
  assert(storage[1]==15 && (storage[5]&255)==1);
#endif
 }
}
''',encoding='utf-8')
    exe=tmp_path/'guard'
    command=[compiler,'-std=c++17','-O2','-I',str(tmp_path),'-I',str(ROOT),str(source),'-o',str(exe)]
    if enabled:command.insert(1,'-DDEEPEP_LL_STAGE_TIMER')
    subprocess.run(command,check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
