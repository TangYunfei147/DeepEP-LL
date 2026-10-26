"""Control-opt sparse recorder/selector mocks and ed1 business-token guards; CPU only."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
COMMON=ROOT/'csrc/kernels/dispatch/low_latency/common'
MODES=('prequant_e4m3','fp16','fp16_e4m3','fp16_e5m2','bf16','bf16_e4m3','bf16_e5m2')

def business_tokens(source):
    """Strip only probes/entry recorder plumbing; this is not a codegen proof."""
    source=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
    source=re.sub(r'LL_(?:SEG_(?:BEGIN|END|COUNT|SCOPE|FIRST|RELAY_COMPLETION)|SPARSE_(?:MARK|AFTER|WHEN))\([^;]*?\);','',source)
    source=re.sub(r'LL_MTE3_[A-Z_]+\([^;]*?\);','',source)
    source=re.sub(r'LL_BW_[A-Z_]+\([^;]*?\);','',source)
    source=source.replace('#include "../common/ll_stage_timer_device.h"','') if 'class ServerDedupReceiveTasks' in source else source
    source=source.replace(',\n                                 LlStageTimer::DeviceTimer* timer_ = nullptr','')
    source=source.replace('(void)timer_;','')
    source=source.replace('recvCoreNum_, timer_)','recvCoreNum_)')
    source=re.sub(r'#ifdef DEEPEP_LL_STAGE_TIMER\s*(?:const uint64_t sparseEntry =[^;]+;|timer\.MarkAt<LlStageTimer::ENTRY>\(sparseEntry\);)\s*#endif','',source)
    source=re.sub(r'if \(timer.Enabled\(\)\) \{ AscendC::SyncAll<true>\(\); timer.Flush\(\); \}','',source)
    source=re.sub(r'const uint64_t (?:initBegin|kernelBegin) = timer.Begin\([^;]*\);','',source)
    source=re.sub(r'timer\.(?:End\([^;]*\)|Mark<LlStageTimer::\w+>\(\)|Flush\(\));','',source)
    source=source.replace('timer.Enabled() ? &timer : nullptr','&timer')
    source=source.replace('dispatch_server_dedup_launch_bw17_','dispatch_server_dedup_launch_')
    source=source.replace('dispatch_server_dedup_kernel_do_bw17','dispatch_server_dedup_kernel_do_v2')
    source=source.replace('combine_server_dedup_kernel_do_bw17','combine_server_dedup_kernel_do')
    source=re.sub(r'extern "C" uint32_t deepep_ll_stage_timer_(?:schema|profile_id)\(\);','',source)
    return re.findall(r'[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+|[^\s]',source)

def test_business_sources_match_ed1_after_probe_removal():
    baseline=json.loads(Path(__file__).with_name('sparse_control_business_baseline.json').read_text(encoding='utf-8'))
    assert baseline['baseline'].startswith('ed1c553')
    for relative,expected in baseline['files'].items():
        # Intentional descriptor/lifetime and per-consumer prefix-ready changes.
        # Keep the immutable old hashes for every other business file; native
        # mocks and delayed schedules exercise the changed paths separately.
        if relative in {
            'csrc/kernels/dispatch/low_latency/common/ll_core_config.h',
            'csrc/kernels/dispatch/low_latency/common/ll_receive_probe_pipeline.h',
            'csrc/kernels/dispatch/low_latency/device/dispatch_flag_probe.h',
            'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h',
            'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h',
            'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h'}:
            continue
        current=business_tokens((ROOT/relative).read_text(encoding='utf-8'))
        assert hashlib.sha256(' '.join(current).encode()).hexdigest()==expected,relative

def test_bandwidth_recorder_isolates_only_explicit_completion_intervals():
    source=(COMMON/'ll_stage_timer_device.h').read_text(encoding='utf-8')
    for removed in ('ProfileIndex','ProfileEnabled','sums_[','calls_[','events_[','cycles_[','SyncAll','PipeBarrier'):
        assert removed not in source
    assert 'ticks_[kPointCount]' in source
    assert 'operation_==1U && mode_==6U' in source
    assert 'kLlOverlapExperimentStage != 3U' in source
    assert source.count('WaitFlag<AscendC::HardEvent::MTE3_S>')==3
    assert 'offset<kExportWords;offset+=8U' in source
    for macro in ('SCOPE','BEGIN','END','COUNT','RELAY_COMPLETION','FIRST'):
        assert re.search(r'#define LL_SEG_'+macro+r'\([^\n]*\) do \{\} while\(false\)',source)
    for macro in ('WAIT_BEGIN','WAIT_END','DRAIN','DRAIN_WHEN'):
        assert re.search(r'#define LL_MTE3_'+macro+r'\([^\n]*\) do \{\} while\(false\)',source)
    split=next(line for line in source.splitlines() if line.startswith('#define LL_BW_SPLIT_INPUT') and 'CompletionEnabled' in line)
    ordered=('CompletionEnabled()', 'EndInputPrefetch(input)', 'EndStage(', 'BeginStage(', 'BeginInputPrefetch()')
    assert [split.index(x) for x in ordered]==sorted(split.index(x) for x in ordered)

def test_sparse_layout_host_identity_and_entry_lifetimes():
    layout=(COMMON/'ll_stage_timer_layout.h').read_text(encoding='utf-8')
    assert 'kSchema = 17U' in layout and 'kSlots = 16U' in layout
    assert 'kPointCount < 64U' in layout
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'deepep_ll_stage_timer_schema()!=LlStageTimer::kSchema' in host
    assert 'deepep_ll_stage_timer_profile_id()!=LlStageTimer::kProfileId' in host
    assert 'combine supports observe only' in host
    assert 'result["recording"] = "stage-bandwidth"' in host and 'result["points"] = points' in host
    assert 'module.attr("low_latency_implementation") = "g4-r0-prefix"' in host
    assert 'small_b_prefix' not in host
    for path in ('csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h','csrc/kernels/combine/low_latency.cpp'):
        source=(ROOT/path).read_text(encoding='utf-8')
        assert 'SyncAll' not in source
        assert source.index('sparseEntry =') < source.index('DeviceTimer timer') < source.index('timer.Init(')
        assert 'timer.MarkAt<LlStageTimer::ENTRY>(sparseEntry)' in source
        ordered=('kernel.Init(','timer.Mark<LlStageTimer::INIT_DONE>()','kernel.Process();','timer.Mark<LlStageTimer::WORK_DONE>()','timer.Flush()')
        assert [source.index(value) for value in ordered]==sorted(source.index(value) for value in ordered)
        assert 'timer.Enabled() ? &timer : nullptr' in source
    entry=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h').read_text(encoding='utf-8')
    assert 'LL_VARIANT_JOIN(dispatch_server_dedup_kernel_,DEEPEP_LL_VARIANT_NAME)' in entry
    assert 'LL_VARIANT_JOIN(dispatch_server_dedup_launch_bw17_,DEEPEP_LL_VARIANT_NAME)' in entry
    assert {p.stem for p in (COMMON.parent/'variants').glob('*.cpp')}==set(MODES)

def test_c1_pair_and_serial_receive_record_only_observed_ready():
    source=(COMMON.parent/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    start=source[source.index('__aicore__ inline void StartProbe(uint32_t start,uint32_t count,uint32_t offset) {',source.index('// Only this path owns event 6')):]
    start=start[:start.index('__aicore__ inline LlReceiveProbeResult FinishProbe')]
    finish=source[source.index('__aicore__ inline LlReceiveProbeResult FinishProbe(uint32_t count) {',source.index('// Only this path owns event 6')):]
    finish=finish[:finish.index('__aicore__ inline __gm__')]
    serial=source[source.index('__aicore__ inline ProbeResult Probe('):source.index('__aicore__ inline void IssueLoad(')]
    assert 'LL_SPARSE_MARK(FIRST_RECV_PROBE);' in start
    assert 'LL_SPARSE_MARK(FIRST_RECV_PROBE);' in serial
    for block in (finish,serial):
        assert 'LL_SPARSE_WHEN(result.readyMask!=0U,FIRST_RECV_READY);' in block
        assert block.index('LL_SPARSE_WHEN(')>block.index('result.readyMask=')
        assert block.index('LL_SPARSE_WHEN(')<block.index('return result;')
    assert 'LlRunReceiveProbePair(ops,tasks_.PendingCount(),scan_)' in source
    assert 'if (!probePrefetch_) return TrySerialBatch();' in source
    assert source.count('LL_SPARSE_MARK(RECV_PREPARED);')==2
    run=source[source.index('__aicore__ inline void Run()'):source.index('private:')]
    assert run.index('PipeBarrier<PIPE_ALL>();')<run.index('LL_SPARSE_MARK(RECV_DONE);')

def test_send_done_is_after_existing_scalar_visible_completion():
    device=COMMON.parent/'device'
    source=(device/'dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    retire=source[source.index('inline void Retire()'):source.index('inline void ConfigureOverlap')]
    assert 'WaitFlag<HardEvent::MTE3_V>' in retire and 'SEND_DONE' not in retire
    counts=(device/'dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    finish=counts[counts.index('inline void Finish()'):counts.index('private:')]
    assert finish.index('PipeBarrier<PIPE_ALL>();')<finish.index('LL_SPARSE_AFTER(FIRST_SEND,SEND_DONE);')
    assert re.search(r'Sync<HardEvent::MTE3_S>\(\);\s*LL_SPARSE_AFTER\(FIRST_SEND,SEND_DONE\);',source)
    without_comments=re.sub(r'//[^\n]*','',counts)
    assert re.search(r'SyncFunc<HardEvent::V_MTE3>\(\);\s*LL_SPARSE_MARK\(PREFIX_LOCAL_DONE\);',without_comments)

def test_mte3_sites_correspond_to_scalar_waits_and_bounded_s_interventions():
    device=COMMON.parent/'device'
    send=(device/'dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    recv=(device/'dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    tasks=(device/'dispatch_server_dedup_receive_tasks.h').read_text(encoding='utf-8')
    counts=(device/'dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    # Prequantized FP8's input wait is MTE2_S; only internal quantization waits V_S.
    assert re.search(r'LL_MTE3_WAIT_BEGIN\(txQuantWait\);\s*Sync<HardEvent::V_S>\(\);\s*LL_MTE3_WAIT_END\(txQuantWait,TX_INPUT_READY\);',send)
    assert re.search(r'LL_MTE3_WAIT_BEGIN\(txInputWait\);\s*WaitFlag<HardEvent::MTE2_S>\(inputReady_\[pendingReady_\]\);\s*LL_MTE3_WAIT_END\(txInputWait,TX_INPUT_READY\);',send)
    assert re.search(r'LL_MTE3_DRAIN_WHEN\(packets_>=lanes_,TX_REUSE_MTE3,4U,pipe_\);\s*WaitFlag<HardEvent::MTE3_V>\(lane\);',send)
    tail=send[send.index('LL_SEG_BEGIN(metaPublish'):send.index('// Original R cores')]
    assert tail.index('DataCopyPad(weightOut_')<tail.index('LL_SPARSE_MARK(SOURCE_SUBMIT_DONE)')<tail.index('LL_MTE3_DRAIN(TX_END_MTE3,4U,pipe_)')
    assert re.search(r'LL_MTE3_WAIT_BEGIN\(rxPrepareWait\);\s*SetFlag<HardEvent::V_S>\(ready\);\s*WaitFlag<HardEvent::V_S>\(ready\);\s*LL_MTE3_WAIT_END\(rxPrepareWait,RX_PREPARE_VS\);',tasks)
    assert 'recvCoreNum_, timer_)' in recv
    assert 'rxFlagWait' not in recv
    assert re.search(r'if \(reuse\).*?LL_MTE3_DRAIN\(RX_REUSE_MTE3,5U,pipe_\);\s*WaitFlag<HardEvent::MTE3_MTE2>\(lane\);',recv,re.S)
    run=recv[recv.index('inline void Run()'):recv.index('private:')]
    ordered=('while (tasks_.PendingCount())','LL_SPARSE_MARK(RECV_PENDING_EMPTY)',
             'if (!tasks_.Count()) return','LL_MTE3_DRAIN(RX_END_MTE3,5U,pipe_)',
             'WaitFlag<HardEvent::MTE3_MTE2>','ClearFlags()','LL_MTE3_WAIT_BEGIN(rxFinalWait)',
             'PipeBarrier<PIPE_ALL>()','LL_MTE3_WAIT_END(rxFinalWait,RX_FINAL_ALL)',
             'LL_SPARSE_MARK(RECV_DONE)')
    assert [run.index(x) for x in ordered]==sorted(run.index(x) for x in ordered)
    assert re.search(r'LL_SPARSE_MARK\(PREFIX_WAIT_ENTER\);\s*WaitCumSumFlag\(\);\s*LL_SPARSE_MARK\(PREFIX_WAIT_EXIT\);',counts)


def run_sparse_recorder_mock(tmp_path,enabled):
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('C++ compiler unavailable; target native mock remains required')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cstdint>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR=uint8_t*;
namespace AscendC {
inline uint64_t clock=1; inline uint32_t core=0; inline unsigned flushes=0;
inline unsigned sets=0,waits=0,fetches=0; inline bool outstanding=false; inline uint64_t queued=0;
inline uint64_t GetSystemCycle(){return clock++;}
inline uint32_t GetBlockIdx(){return core;}
inline uint32_t GetBlockNum(){return 56;}
inline void Trap(){throw std::runtime_error("trap");}
enum class HardEvent { MTE3_S };
struct TPipe { int FetchEventID(HardEvent){++fetches;return 3;} };
template<HardEvent E> void SetFlag(int id){if(id!=3 || outstanding) throw std::runtime_error("event collision");outstanding=true;++sets;}
template<HardEvent E> void WaitFlag(int id){if(id!=3 || !outstanding) throw std::runtime_error("event ownership");outstanding=false;++waits;clock+=2+queued;queued=0;}
enum class CacheLine{SINGLE_CACHE_LINE};enum class DcciDst{CACHELINE_OUT};
template<class T>struct GlobalTensor{void SetGlobalBuffer(T*,unsigned){}};
template<class T,CacheLine C,DcciDst D>void DataCacheCleanAndInvalid(GlobalTensor<T>&){++flushes;}
}
''')
    source=tmp_path/'check.cpp';source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h"
using namespace LlStageTimer;
struct SplitProbe {
 DeviceTimer* timer_; unsigned flush=0,begin=0;
 void EndInputPrefetch(int){++flush;timer_->Transfer(REMOTE_FIRST_HOP,true,4096,3696);}
 void BeginInputPrefetch(){++begin;}
 void Split(AscendC::TPipe* p){(void)p;LL_BW_SPLIT_INPUT(0,p);}
};
int main(){
 static_assert(kSchema==17 && kWordsPerCore*8==1024 && kPointCount==18 && kExportWords==128);
 static_assert(kBandwidthStageCount==6 && kBandwidthRecordWords==16 && kBandwidthWordOffset==32);
 static_assert(DispatchDedup::kLlSendCoreNum==30 && DispatchDedup::kLlRecvCoreNum==26);
 static_assert(DispatchDedup::kLlCoreProtocolVersion==28030 && DispatchDedup::kLlDispatchUbBytes==230*1024);
 std::vector<uint64_t> words(56*kWordsPerCore+64,0);
 std::fill(words.end()-64,words.end(),0x12345678ULL);
 AscendC::core=55;
 DeviceTimer timer;timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 auto* timer_=&timer;AscendC::TPipe pipe;
 unsigned sideEffects=0; auto clock=AscendC::clock;
 LL_SEG_SCOPE(INIT);LL_SEG_BEGIN(old,KERNEL);LL_SEG_END(old,KERNEL);
 LL_SEG_COUNT(TX_PACKETS,++sideEffects);LL_SEG_RELAY_COMPLETION(++sideEffects);LL_SEG_FIRST(BEFORE_FIRST_SEND);
 LL_MTE3_WAIT_BEGIN(gone);LL_MTE3_WAIT_END(gone,TX_INPUT_READY);
 LL_MTE3_DRAIN_WHEN(++sideEffects,RX_REUSE_MTE3,5,++sideEffects);
 assert(sideEffects==0 && AscendC::clock==clock);
 timer.MarkAt<ENTRY>(clock);timer.Mark<INIT_DONE>();LL_SPARSE_WHEN(false,FIRST_RECV_READY);
 timer.MarkIfSeen<FIRST_SEND,SEND_DONE>();LL_SPARSE_MARK(FIRST_SEND);
 auto afterFirst=AscendC::clock;
 for(unsigned i=0;i<10000;++i) LL_SPARSE_MARK(FIRST_SEND);
 assert(AscendC::clock==afterFirst);
 LL_SPARSE_AFTER(FIRST_SEND,SEND_DONE);timer.Mark<WORK_DONE>();
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
 assert(AscendC::flushes==0);timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::clock==clock+4 && AscendC::flushes==16);
 auto* row=words.data()+55*kWordsPerCore;
 assert(((row[5]>>24)&255)==8 && ((row[5]>>8)&255)==30);
 assert(row[0]==kMagic && row[1]==17 && row[7]==0);
 assert(row[6]==((1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<FIRST_SEND)|(1ULL<<SEND_DONE)|(1ULL<<WORK_DONE)));
 assert(row[8+ENTRY]<=row[8+INIT_DONE] && row[8+INIT_DONE]<row[8+FIRST_SEND]);
 assert(row[8+FIRST_SEND]<row[8+WORK_DONE] && row[8+RECV_DONE]==0);
 assert(std::all_of(words.begin(),words.begin()+55*kWordsPerCore,[](auto value){return value==0;}));
 assert(std::all_of(row+24,row+kWordsPerCore,[](auto value){return value==0;}));
#else
 (void)timer_;assert(AscendC::flushes==0 && AscendC::clock==clock);
#endif
 // Exact peer/self counters, observe without events, complete with one empty
 // control per nonempty interval, and no fake completed interval for empty work.
 for(unsigned mode: {1U,6U}) for(unsigned core: {0U,29U,30U,55U}) {
   AscendC::core=core;AscendC::queued=9;
   timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,mode);
   timer.SendLanes(6,true);timer.RecvLanes(4,true);timer.SendLanes(6,true);
   auto before=AscendC::clock;auto events=AscendC::waits;
   timer.BeginStage(RELAY,&pipe);
   timer.Transfer(RELAY,true,4096,3696);AscendC::clock+=5;
   timer.Transfer(RELAY,false,8192,7392);AscendC::queued=20;
   timer.EndStage(RELAY,&pipe);
   timer.BeginStage(RELAY,&pipe);timer.Transfer(RELAY,true,4096,3696);timer.EndStage(RELAY,&pipe);
   timer.BeginStage(SOURCE_DIRECT,&pipe);timer.EndStage(SOURCE_DIRECT,&pipe);
   timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
   auto* row=words.data()+core*kWordsPerCore;auto* stat=row+32+RELAY*16;
   assert(row[7]==(6U|(4U<<8)|(1U<<16)) && (row[5]&255)==mode);
   assert(stat[PEER_BYTES]==8192 && stat[SELF_BYTES]==8192);
   assert(stat[PEER_PAYLOAD_BYTES]==7392 && stat[SELF_PAYLOAD_BYTES]==7392);
   assert(stat[PEER_OPERATIONS]==2 && stat[SELF_OPERATIONS]==1);
   assert(stat[FIRST_ISSUE_TICK]<=stat[LAST_ISSUE_TICK] && stat[BANDWIDTH_RESERVED]==0);
   if(mode==6) {
     assert(AscendC::waits-events==7 && stat[COMPLETED_INTERVALS]==2 && stat[COMPLETED_OPERATIONS]==3);
     assert(stat[PRE_DRAIN_TICKS]==15 && stat[END_DRAIN_TICKS]==26 && stat[EMPTY_CONTROL_TICKS]==6);
     assert(stat[LAST_COMPLETION_TICK]>=stat[LAST_ISSUE_TICK]);
     assert(stat[COMPLETED_ELAPSED_TICKS]>=stat[END_DRAIN_TICKS]);
     auto* empty=row+32+SOURCE_DIRECT*16;
     assert(empty[PRE_DRAIN_TICKS]>0);
     for(unsigned f=0;f<16;++f) if(f!=PRE_DRAIN_TICKS) assert(empty[f]==0);
   } else {
     assert(AscendC::waits==events);
     for(unsigned f=LAST_COMPLETION_TICK;f<16;++f) assert(stat[f]==0);
   }
#else
   assert(AscendC::waits==events && AscendC::clock==before+5);
#endif
   // Runtime-disabled and OFF paths must not evaluate any new macro argument.
   timer.Init(nullptr,1,mode);before=AscendC::clock;events=AscendC::waits;
   LL_BW_BEGIN(BandwidthStage(++sideEffects),&pipe);
   LL_BW_TRANSFER(BandwidthStage(++sideEffects),true,32,0);
   LL_BW_TRANSFER(RELAY,++sideEffects,32,0);
   LL_BW_TRANSFER(RELAY,true,++sideEffects,0);
   LL_BW_TRANSFER(RELAY,true,32,++sideEffects);
   LL_BW_END(BandwidthStage(++sideEffects),&pipe);
   SplitProbe noCapture{&timer};noCapture.Split(&pipe);
   assert(!noCapture.flush && !noCapture.begin && sideEffects==0);
   timer.Mark<ENTRY>();timer.Flush();assert(AscendC::clock==before && AscendC::waits==events);
 }
 // The phase split flushes its pending FIRST_HOP before opening SOURCE_DIRECT.
 for(unsigned mode: {1U,6U}) {
   timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,mode);
   timer.BeginStage(REMOTE_FIRST_HOP,&pipe);SplitProbe split{&timer};split.Split(&pipe);
#ifdef DEEPEP_LL_STAGE_TIMER
   assert(split.flush==(mode==6) && split.begin==(mode==6));
#else
   assert(!split.flush && !split.begin);
#endif
   timer.EndStage(SOURCE_DIRECT,&pipe);timer.Flush();
 }
#ifdef DEEPEP_LL_STAGE_TIMER
 AscendC::core=0;
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);timer.RecvLanes(3,true);timer.SendLanes(5,true);timer.Flush();
 assert(words[7]==(5U|(3U<<8)|(1U<<16)));
 bool bad=false;try{timer.SendLanes(5,false);}catch(const std::runtime_error&){bad=true;}assert(bad);
 for(unsigned mode: {0U,2U,3U,4U,5U,7U}) {
   bool bad=false;try{timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,mode);}catch(const std::runtime_error&){bad=true;}assert(bad);
 }
 bool badCombine=false;try{timer.Init(reinterpret_cast<uint8_t*>(words.data()),2,6);}catch(const std::runtime_error&){badCombine=true;}assert(badCombine);
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),2,1);timer.Transfer(RELAY,true,4096,3696);timer.Flush();
 assert(std::all_of(words.begin()+32,words.begin()+128,[](auto v){return v==0;}));
 // Corrupt accounting lifecycle must fail rather than silently mix phases.
 for(unsigned fault: {0U,1U,2U,3U}) {
   timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,6);timer.BeginStage(RELAY,&pipe);
   bool bad=false;try{
     if(fault==0) timer.BeginStage(SOURCE_DIRECT,&pipe);
     if(fault==1) timer.Transfer(SOURCE_DIRECT,true,32,0);
     if(fault==2) timer.EndStage(SOURCE_DIRECT,&pipe);
     if(fault==3) timer.Flush();
   }catch(const std::runtime_error&){bad=true;}assert(bad);
 }
#endif
 assert(std::all_of(words.end()-64,words.end(),[](auto v){return v==0x12345678ULL;}));
}
''')
    executable=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),'-DDEEPEP_LL_TIMER_PROFILE_ID=8',
                    '-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26',
                    '-I',str(tmp_path),'-I',str(ROOT),str(source),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)

def run_sparse_host_selector_mock(tmp_path,enabled):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('C++ compiler unavailable; run this CPU mock on a prepared host')
    source=tmp_path/'selector.cpp'
    source.write_text(r'''
#include <cassert>
#include <stdexcept>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
using namespace DispatchDedup;
static unsigned calls=0,selected=99;
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_bw17_##name(void*,const deepep::EpServerDedupKernelArgs*) {++calls;selected=n;}
STUB(prequant_e4m3,0) STUB(fp16,1) STUB(fp16_e4m3,2) STUB(fp16_e5m2,3)
STUB(bf16,4) STUB(bf16_e4m3,5) STUB(bf16_e5m2,6)
int main() {
 const unsigned table[7][4]={{15,4,15,2},{3,0,3,0},{3,4,15,1},{3,4,16,1},{11,0,11,0},{11,4,15,1},{11,4,16,1}};
 for(unsigned dtype=0;dtype<256;++dtype) for(unsigned quant:{0U,1U,4U,255U})
 for(unsigned output:{3U,11U,15U,16U,255U}) for(unsigned input=0;input<4;++input) {
   unsigned expected=0xffffffffU;
   for(unsigned m=0;m<7;++m) if(dtype==table[m][0] && quant==table[m][1] && input==table[m][3] &&
       (output==table[m][2] || (output==255 && (m==1 || m==4)))) expected=m;
   assert(LlDispatchVariantForTiling(dtype,quant,output,input)==expected);
 }
 deepep::EpServerDedupKernelArgs args{};
 static_assert(sizeof(args)==152);
 for(unsigned m=0;m<7;++m) {calls=0;dispatch_server_dedup_kernel_do_bw17(nullptr,&args,m);assert(calls==1 && selected==m);}
 for(unsigned m:{7U,255U,0xffffffffU}) {
   calls=0;bool threw=false;try {dispatch_server_dedup_kernel_do_bw17(nullptr,&args,m);}catch(const std::invalid_argument&){threw=true;}
   assert(threw && !calls);
 }
 bool threw=false;try {dispatch_server_dedup_kernel_do_bw17(nullptr,nullptr,0);}catch(const std::invalid_argument&){threw=true;}assert(threw);
 assert(deepep_ll_core_protocol_version()==28030 && deepep_ll_ranks_per_group()==4);
 assert(deepep_ll_stage_timer_schema()==17 && deepep_ll_stage_timer_profile_id()==8);
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(deepep_ll_stage_timer_enabled()==1);
#else
 assert(deepep_ll_stage_timer_enabled()==0);
#endif
}
''',encoding='utf-8')
    executable=tmp_path/'selector'
    subprocess.run([compiler,'-std=c++17','-DCATLASS_ARCH=3510',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),
                    '-DDEEPEP_LL_SEND_CORES=30','-DDEEPEP_LL_RECV_CORES=26',
                    '-I',str(ROOT),str(source),str(ROOT/'csrc/kernels/dispatch/low_latency.cpp'),
                    '-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True,timeout=30)

def test_non_sparse_profile_cannot_compile(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if compiler is None:pytest.skip('C++ compiler unavailable; run this CPU check on a prepared host')
    source=tmp_path/'profile.cpp'
    source.write_text('#include "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_profile.h"\n',encoding='utf-8')
    result=subprocess.run([compiler,'-std=c++17','-DDEEPEP_LL_TIMER_PROFILE_ID=0','-I',str(ROOT),
                           '-c',str(source),'-o',str(tmp_path/'profile.o')],capture_output=True,text=True)
    assert result.returncode!=0 and 'supports stage-bandwidth diagnostics only' in result.stderr

@pytest.mark.parametrize('corruption',[None,'missing_point','unused_tail'])
def test_device_smoke_offline_contract(tmp_path,corruption):
    points=('ENTRY','INIT_DONE','COUNTS_SENT','COUNTS_RECEIVED','PREFIX_LOCAL_DONE',
            'PREFIX_READY','SEND_PREPARED','FIRST_SEND','SEND_DONE','RECV_PREPARED',
            'FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE')
    visited=('ENTRY','INIT_DONE','FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE')
    output=[]
    for core in range(56):
        words=[0]*128
        role=3 if core<30 else 2
        words[:8]=[0x4c4c5431,17,1,56,core,1|(30<<8)|(role<<16)|(8<<24),0,0]
        for sequence,name in enumerate(visited):
            index=points.index(name)
            words[6]|=1<<index;words[8+index]=100+sequence
        if core==0 and corruption=='missing_point':
            index=points.index('RECV_DONE');words[6]&=~(1<<index);words[8+index]=0
        if core==0 and corruption=='unused_tail':words[26]=1
        output.append([words[i:i+8] for i in range(0,128,8)])
    (tmp_path/'raw-1-1.json').write_text(json.dumps(output),encoding='utf-8')
    result=subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/check_device.py'),str(tmp_path)],
                          capture_output=True,text=True)
    assert (result.returncode==0)==(corruption is None),result.stderr
    if corruption is None:assert (tmp_path/'parsed-1.json').is_file()
    smoke=(ROOT/'tests/ll_stage_timer/timer_smoke.cpp').read_text(encoding='utf-8')
    assert not any(old in smoke for old in ('timer.Begin(','timer.End(','timer.Count(','SyncAll','PipeBarrier'))
    assert smoke.index('sparseEntry=')<smoke.index('DeviceTimer timer')
