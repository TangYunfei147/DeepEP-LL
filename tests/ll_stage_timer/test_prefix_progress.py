"""Prefix progress ownership and bounded retry contracts; CPU only."""
from pathlib import Path
import shutil,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'csrc/kernels/dispatch/low_latency'

def test_ready_check_and_callback_are_bounded_and_publish_before_local_tx():
    counts=(BASE/'device/dispatch_server_dedup_counts.h').read_text()
    kernel=(BASE/'dispatch_server_dedup_kernel.h').read_text()
    step=counts.split('inline bool ServerDedupCounts::TryCompletePrefix()',1)[1].split('inline void ServerDedupCounts::IssuePrefixLoad()',1)[0]
    ready=counts.split('inline bool ServerDedupCounts::TryCumSumFlag()',1)[1]
    assert 'while (' not in step and 'while (' not in ready
    assert ready.index('if (cumSumFlag != targetFlag) return false;')<ready.index('Duplicate<float>(statusCleanFp32Tensor_')
    path=kernel.split('if (overlap_.Cooperative() && core_>=sendCores_)',1)[1].split('} else {',1)[0]
    assert path.index('counts_.CalcCumsum()')<path.index('counts_.TryCompletePrefix()')<path.index('sender_.ProcessGroupLocal(')
    finish=counts.split('inline void ServerDedupCounts::FinishPrefixLoad()',1)[1].split('inline void ServerDedupCounts::CompleteCumsum()',1)[0]
    assert finish.index('WaitFlag<HardEvent::MTE2_S>(kLlPrefixLoadEvent)')<finish.index('prefix.GetValue(')
    sender=(BASE/'device/dispatch_server_dedup_send.h').read_text()
    assert 'event==HardEvent::MTE2_S' in sender and 'TEventID(7) : pipe_->FetchEventID(event)' in sender
    assert 'kLlPrefixLoadEvent = 5U' in (BASE/'common/ll_overlap_stage.h').read_text()


def test_actual_progress_body_issues_and_consumes_once(tmp_path):
    cc=shutil.which('g++') or shutil.which('clang++')
    if not cc:pytest.skip('C++ compiler required for actual prefix controller body')
    counts=(BASE/'device/dispatch_server_dedup_counts.h').read_text()
    body=counts[counts.index('__aicore__ inline bool ServerDedupCounts::TryCompletePrefix()'):counts.index('__aicore__ inline void ServerDedupCounts::IssuePrefixLoad()')]
    code=r'''#include <cassert>
#include <stdexcept>
#define __aicore__
#define LL_SEG_SCOPE(stage)
namespace AscendC { void Trap(){throw std::logic_error("unprepared");} }
struct ServerDedupCounts {
 enum class PrefixState {Unprepared,Polling,Loading,Ready};
 PrefixState prefixState_=PrefixState::Polling;
 bool available=false,pending=false;unsigned polls=0,clears=0,issues=0,waits=0;
 bool TryCumSumFlag(){++polls;if(!available)return false;++clears;available=false;return true;}
 void IssuePrefixLoad(){assert(!pending);pending=true;++issues;}
 void FinishPrefixLoad(){assert(pending);pending=false;++waits;prefixState_=PrefixState::Ready;}
 bool TryCompletePrefix();
};
'''+body+r'''
int main(){
 for(unsigned delay=0;delay<100;++delay){
  ServerDedupCounts c;
  for(unsigned i=0;i<delay;++i){assert(!c.TryCompletePrefix());assert(!c.issues && !c.clears && !c.waits);}
  c.available=true;assert(!c.TryCompletePrefix());assert(c.pending && c.issues==1 && c.clears==1 && !c.waits);
  // Arbitrary local-send preparation/batch work may run here without another poll.
  auto polls=c.polls;assert(c.TryCompletePrefix());assert(!c.pending && c.waits==1);
  for(unsigned i=0;i<10;++i)assert(c.TryCompletePrefix());
  assert(c.polls==polls && c.clears==1 && c.issues==1 && c.waits==1);
 }
 ServerDedupCounts bad;bad.prefixState_=ServerDedupCounts::PrefixState::Unprepared;
 bool trapped=false;try{bad.TryCompletePrefix();}catch(const std::logic_error&){trapped=true;}assert(trapped);
}'''
    source=tmp_path/'progress.cpp';source.write_text(code)
    exe=tmp_path/'progress'
    subprocess.run([cc,'-std=c++17',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize('schema',[12,13])
def test_async_spans_have_separate_default_tracks(schema):
    import importlib.util
    spec=importlib.util.spec_from_file_location('prefix_fixture',ROOT/'tests/ll_stage_timer/test_segments.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    raw=fixture.raw(schema=schema)
    words=[x for row in raw[0] for x in row]
    entries=[tuple(words[8+i*2:10+i*2]) for i in range(words[6])]
    trace=[x for x in entries if (x[0]&0xffffffff)<0x8000]
    metrics=[x for x in entries if (x[0]&0xffffffff)>=0x8000]
    stages=[79,83] if schema==13 else [79]
    for stage in stages:
        trace.extend([(stage*2,150),(stage*2+1,600)])
        metrics.append(((1<<32)|(0x8000+stage),450))
    entries=trace+metrics;words[6]=len(entries);words[8:]=[0]*(len(words)-8)
    for i,pair in enumerate(entries):words[8+i*2:10+i*2]=pair
    raw[0]=[words[i:i+8] for i in range(0,len(words),8)]
    events=fixture.timer.chrome_trace([dict(raw=raw,operation='dispatch',frequency_hz=1e9,metadata=dict(rank=0,iteration=0))])['traceEvents']
    spans=[e for e in events if e.get('ph')=='X' and e['name'] in ('PREFIX_LOAD','PREFIX_READY_SPAN')]
    assert len(spans)==len(stages)
    for span in spans:
        assert (span['tid']!=0)==(schema==13)
        assert span['dur']==pytest.approx(.45)
    if schema==13:assert len({e['tid'] for e in spans})==2
