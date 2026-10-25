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
    source=source.replace('#include "../common/ll_stage_timer_device.h"','') if 'class ServerDedupReceiveTasks' in source else source
    source=source.replace(',\n                                 LlStageTimer::DeviceTimer* timer_ = nullptr','')
    source=source.replace('(void)timer_;','')
    source=source.replace('recvCoreNum_, timer_)','recvCoreNum_)')
    source=re.sub(r'#ifdef DEEPEP_LL_STAGE_TIMER\s*(?:const uint64_t sparseEntry =[^;]+;|timer\.MarkAt<LlStageTimer::ENTRY>\(sparseEntry\);)\s*#endif','',source)
    source=re.sub(r'if \(timer.Enabled\(\)\) \{ AscendC::SyncAll<true>\(\); timer.Flush\(\); \}','',source)
    source=re.sub(r'const uint64_t (?:initBegin|kernelBegin) = timer.Begin\([^;]*\);','',source)
    source=re.sub(r'timer\.(?:End\([^;]*\)|Mark<LlStageTimer::\w+>\(\)|Flush\(\));','',source)
    source=source.replace('timer.Enabled() ? &timer : nullptr','&timer')
    source=source.replace('dispatch_server_dedup_launch_mte3_16_','dispatch_server_dedup_launch_')
    source=source.replace('dispatch_server_dedup_kernel_do_mte3_16','dispatch_server_dedup_kernel_do_v2')
    source=source.replace('combine_server_dedup_kernel_do_mte3_16','combine_server_dedup_kernel_do')
    source=re.sub(r'extern "C" uint32_t deepep_ll_stage_timer_(?:schema|profile_id)\(\);','',source)
    return re.findall(r'[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+|[^\s]',source)

def test_unchanged_business_sources_match_ed1_after_probe_removal():
    baseline=json.loads(Path(__file__).with_name('sparse_control_business_baseline.json').read_text(encoding='utf-8'))
    assert baseline['baseline'].startswith('ed1c553')
    for relative,expected in baseline['files'].items():
        # The explicit prequantized S-sender candidate intentionally changes
        # input/tail dependencies. Its unchanged algorithms are checked below,
        # and its new tail/schedule is exercised by the vector-tail CPU tests.
        if relative=='csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h':
            continue
        current=business_tokens((ROOT/relative).read_text(encoding='utf-8'))
        assert hashlib.sha256(' '.join(current).encode()).hexdigest()==expected,relative


def test_vector_tail_candidate_preserves_sender_route_and_fallback_algorithms():
    relative='csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h'
    baseline=subprocess.check_output(['git','-C',str(ROOT),'show',
        '9027275b31a28608296b1b71c93fa6e2bb456db5:'+relative],text=True,encoding='utf-8')
    candidate=(ROOT/relative).read_text(encoding='utf-8')
    def body(source,name):
        match=re.search(r'__aicore__ inline (?:void|uint32_t) '+name+r'\(',source)
        assert match is not None,name
        start=match.start()
        begin=source.index('{',start);depth=1;end=begin+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}');end+=1
        return business_tokens(source[begin:end])
    # Packet ordering, q, route selection, online quantization, local sender,
    # scalar fallback tail and source metadata publication are not candidates.
    for name in ('Process','ProcessGroupLocal','SendWithDedup','SendWithoutDedup',
                 'Classify','CalTokenSendExpertCnt','QuantMxfp8','FillTail','Retire'):
        assert body(candidate,name)==body(baseline,name),name

def test_mte3_recorder_limits_dependency_operations_to_explicit_intervention():
    source=(COMMON/'ll_stage_timer_device.h').read_text(encoding='utf-8')
    for removed in ('ProfileIndex','ProfileEnabled','sums_[','calls_[','events_[','cycles_[','SyncAll','PipeBarrier'):
        assert removed not in source
    assert 'ticks_[kPointCount]' in source
    drain=source[source.index('void DrainMte3('):source.index('void Flush()')]
    assert drain.count('WaitFlag<AscendC::HardEvent::MTE3_S>')==2
    assert 'mode_!=mode' in drain and 'operation_!=1U' in drain
    assert 'GetBlockIdx()>=DispatchDedup::kLlSendCoreNum' in drain
    assert source.count('WaitFlag<')==2
    assert 'offset<kExportWords;offset+=8U' in source
    for macro in ('SCOPE','BEGIN','END','COUNT','RELAY_COMPLETION','FIRST'):
        assert re.search(r'#define LL_SEG_'+macro+r'\([^\n]*\) do \{\} while\(false\)',source)

def test_sparse_layout_host_identity_and_entry_lifetimes():
    layout=(COMMON/'ll_stage_timer_layout.h').read_text(encoding='utf-8')
    assert 'kSchema = 16U' in layout and 'kSlots = 8U' in layout
    assert 'kPointCount < 64U' in layout
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'deepep_ll_stage_timer_schema()!=LlStageTimer::kSchema' in host
    assert 'deepep_ll_stage_timer_profile_id()!=LlStageTimer::kProfileId' in host
    assert 'combine supports observe only' in host
    assert 'result["recording"] = "mte3-backpressure"' in host and 'result["points"] = points' in host
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
    assert 'LL_VARIANT_JOIN(dispatch_server_dedup_launch_mte3_16_,DEEPEP_LL_VARIANT_NAME)' in entry
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
int main(){
 using namespace LlStageTimer;
 static_assert(kSchema==16 && kWordsPerCore*8==512 && kPointCount==18 && kExportWords==64);
 static_assert(DispatchDedup::kLlSendCoreNum==30 && DispatchDedup::kLlRecvCoreNum==26);
 static_assert(DispatchDedup::kLlCoreProtocolVersion==19030 && DispatchDedup::kLlDispatchUbBytes==230*1024);
 std::vector<uint64_t> words(56*kWordsPerCore+64,0);
 std::fill(words.end()-64,words.end(),0x12345678ULL);
 AscendC::core=55;
 DeviceTimer timer;timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 auto* timer_=&timer;
 // Removed probes must not evaluate arguments or read the clock at all.
 unsigned sideEffects=0; auto clock=AscendC::clock;
 LL_SEG_SCOPE(INIT); LL_SEG_BEGIN(old,KERNEL); LL_SEG_END(old,KERNEL);
 LL_SEG_COUNT(TX_PACKETS,++sideEffects); LL_SEG_RELAY_COMPLETION(++sideEffects);
 LL_SEG_FIRST(BEFORE_FIRST_SEND);
 assert(sideEffects==0 && AscendC::clock==clock);
 timer.MarkAt<ENTRY>(clock);
 timer.Mark<INIT_DONE>();
 LL_SPARSE_WHEN(false,FIRST_RECV_READY);
 timer.MarkIfSeen<FIRST_SEND,SEND_DONE>();
 LL_SPARSE_MARK(FIRST_SEND);
 auto afterFirst=AscendC::clock;
 for(unsigned i=0;i<10000;++i){LL_SPARSE_MARK(FIRST_SEND);}
 assert(AscendC::clock==afterFirst);
 LL_SPARSE_AFTER(FIRST_SEND,SEND_DONE);
 timer.Mark<WORK_DONE>();
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
 assert(AscendC::flushes==0);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::clock==clock+4);
 assert(AscendC::flushes==8);
 auto* data=words.data()+55*kWordsPerCore;
 assert(((data[5]>>24)&255)==7);
 assert(((data[5]>>8)&255)==30);
 assert(data[0]==kMagic && data[1]==16 && data[7]==0);
 assert(data[6]==((1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<FIRST_SEND)|(1ULL<<SEND_DONE)|(1ULL<<WORK_DONE)));
 assert(data[8+ENTRY]<=data[8+INIT_DONE] && data[8+INIT_DONE]<data[8+FIRST_SEND]);
 assert(data[8+FIRST_SEND]<data[8+WORK_DONE]);
 assert(data[8+RECV_DONE]==0 && data[22]==0 && data[23]==0);
 assert(std::all_of(words.begin(),words.begin()+55*kWordsPerCore,[](auto value){return value==0;}));
 assert(std::all_of(data+24,data+kWordsPerCore,[](auto value){return value==0;}));
#else
 (void)timer_;
 assert(AscendC::flushes==0 && AscendC::clock==clock);
 assert(std::all_of(words.begin(),words.end()-64,[](auto value){return value==0;}));
#endif
 assert(std::all_of(words.end()-64,words.end(),[](auto value){return value==0x12345678ULL;}));
 // Reuse enabled -> null -> enabled without stale timestamps or a clear loop.
 auto oldClock=AscendC::clock;
 timer.Init(nullptr,1,1);timer.Mark<ENTRY>();timer.Mark<WORK_DONE>();timer.Flush();
 assert(AscendC::clock==oldClock);
 timer_=nullptr;
 LL_SPARSE_WHEN(++sideEffects,FIRST_RECV_READY);
 assert(sideEffects==0 && AscendC::clock==oldClock);
 std::fill(words.begin(),words.end()-64,0);
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),2,1);
 timer.Mark<ENTRY>();timer.Mark<INIT_DONE>();timer.Mark<WORK_DONE>();timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 auto* again=words.data()+55*kWordsPerCore;
 assert(again[2]==2 && again[6]==((1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<WORK_DONE)));
 assert(again[8+FIRST_SEND]==0);
 bool rejected=false;
 try{timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,3);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
#endif
 // Both c1 FinishProbe and serial Probe use the same readiness-gated point.
 std::fill(words.begin(),words.end()-64,0);
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);timer_=&timer;
 oldClock=AscendC::clock;
 LL_SPARSE_WHEN(false,FIRST_RECV_READY);
 assert(AscendC::clock==oldClock);
 LL_SPARSE_WHEN(true,FIRST_RECV_READY);
 for(unsigned i=0;i<10000;++i) LL_SPARSE_WHEN(true,FIRST_RECV_READY);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(AscendC::clock==oldClock+1);
 assert(words[55*kWordsPerCore+6]==(1ULL<<FIRST_RECV_READY));
#else
 LL_SPARSE_WHEN(++sideEffects,FIRST_RECV_READY);
 assert(sideEffects==0 && AscendC::clock==oldClock);
#endif
 // Actual recorder contract: OFF/null/observe/R paths add no events/clocks;
 // forced S paths consume two immediate pairs and keep raw/control separate.
 // This models event latency, not CANN timing or network behavior.
 for(unsigned overlap: {0U,1U}) for(unsigned mode: {1U,4U,5U}) for(unsigned core: {0U,29U,30U,55U}) {
   AscendC::core=core;AscendC::queued=9;
   std::fill(words.begin(),words.end()-64,0);
   timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,mode);
   timer.SendLanes(6,bool(overlap));timer.RecvLanes(4,bool(overlap));
   timer.SendLanes(6,bool(overlap)); // Preserve receive lanes and overlap bit.
   AscendC::TPipe pipe;
   auto before=AscendC::clock;auto events=AscendC::waits;
   timer.DrainMte3<TX_REUSE_MTE3,4>(&pipe);
   timer.DrainMte3<RX_REUSE_MTE3,5>(&pipe);
   timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
   const bool active=mode!=1 && core<30;
   auto* row=words.data()+core*kWordsPerCore;
   assert(row[7]==(6U|(4U<<8)|(overlap<<16)) && (row[5]&255)==mode);
   assert(AscendC::waits-events==(active?2U:0U));
   if(active) {
     auto id=mode==4?TX_REUSE_MTE3:RX_REUSE_MTE3;
     auto* stat=row+kWaitMetricWordOffset+id*kWaitMetricFields;
     assert(stat[0]==12 && stat[1]==12 && stat[2]==1 && stat[3]==3);
     assert(AscendC::clock-before==16);
   } else assert(AscendC::clock==before);
#else
   assert(AscendC::waits==events && AscendC::clock==before);
#endif
   before=AscendC::clock;events=AscendC::waits;
   timer.Init(nullptr,1,mode);
   timer.DrainMte3<TX_END_MTE3,4>(&pipe);
   timer.DrainMte3<RX_END_MTE3,5>(&pipe);
   const auto begin=timer.WaitBegin();timer.WaitEnd<TX_INPUT_READY>(begin);
   timer.Flush();
   assert(AscendC::clock==before && AscendC::waits==events);
 }
 AscendC::core=0;
 timer.Init(reinterpret_cast<uint8_t*>(words.data()),1,1);
 auto first=timer.WaitBegin();AscendC::clock+=6;timer.WaitEnd<TX_INPUT_READY>(first);
 auto second=timer.WaitBegin();AscendC::clock+=3;timer.WaitEnd<TX_INPUT_READY>(second);
 timer.Flush();
#ifdef DEEPEP_LL_STAGE_TIMER
 auto* metric=words.data()+kWaitMetricWordOffset;
 assert(metric[0]==11 && metric[1]==7 && metric[2]==2 && metric[3]==0);
 assert(words[7]==0); // Init resets lanes and all metrics.
 timer.RecvLanes(3,true);timer.SendLanes(5,true);timer.Flush();
 assert(words[7]==(5U|(3U<<8)|(1U<<16)));
 bool badOverlap=false;
 try {timer.SendLanes(5,false);} catch(const std::runtime_error&) {badOverlap=true;}
 assert(badOverlap);
 for(unsigned mode: {4U,5U}) {
   bool rejected=false;
   try {timer.Init(reinterpret_cast<uint8_t*>(words.data()),2,mode);}
   catch(const std::runtime_error&) {rejected=true;}
   assert(rejected);
 }
#endif
}
''')
    executable=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                    *(['-DDEEPEP_LL_STAGE_TIMER=1'] if enabled else []),'-DDEEPEP_LL_TIMER_PROFILE_ID=7',
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
#define STUB(name,n) extern "C" void dispatch_server_dedup_launch_mte3_16_##name(void*,const deepep::EpServerDedupKernelArgs*) {++calls;selected=n;}
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
 for(unsigned m=0;m<7;++m) {calls=0;dispatch_server_dedup_kernel_do_mte3_16(nullptr,&args,m);assert(calls==1 && selected==m);}
 for(unsigned m:{7U,255U,0xffffffffU}) {
   calls=0;bool threw=false;try {dispatch_server_dedup_kernel_do_mte3_16(nullptr,&args,m);}catch(const std::invalid_argument&){threw=true;}
   assert(threw && !calls);
 }
 bool threw=false;try {dispatch_server_dedup_kernel_do_mte3_16(nullptr,nullptr,0);}catch(const std::invalid_argument&){threw=true;}assert(threw);
 assert(deepep_ll_core_protocol_version()==19030 && deepep_ll_ranks_per_group()==4);
 assert(deepep_ll_stage_timer_schema()==16 && deepep_ll_stage_timer_profile_id()==7);
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
    assert result.returncode!=0 and 'supports MTE3 diagnostics only' in result.stderr

@pytest.mark.parametrize('corruption',[None,'missing_point','unused_tail'])
def test_device_smoke_offline_contract(tmp_path,corruption):
    points=('ENTRY','INIT_DONE','COUNTS_SENT','COUNTS_RECEIVED','PREFIX_LOCAL_DONE',
            'PREFIX_READY','SEND_PREPARED','FIRST_SEND','SEND_DONE','RECV_PREPARED',
            'FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE')
    visited=('ENTRY','INIT_DONE','FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE')
    output=[]
    for core in range(56):
        words=[0]*64
        role=3 if core<30 else 2
        words[:8]=[0x4c4c5431,16,1,56,core,1|(30<<8)|(role<<16)|(7<<24),0,0]
        for sequence,name in enumerate(visited):
            index=points.index(name)
            words[6]|=1<<index;words[8+index]=100+sequence
        if core==0 and corruption=='missing_point':
            index=points.index('RECV_DONE');words[6]&=~(1<<index);words[8+index]=0
        if core==0 and corruption=='unused_tail':words[26]=1
        output.append([words[i:i+8] for i in range(0,64,8)])
    (tmp_path/'raw-1-1.json').write_text(json.dumps(output),encoding='utf-8')
    result=subprocess.run([sys.executable,str(ROOT/'tests/ll_stage_timer/check_device.py'),str(tmp_path)],
                          capture_output=True,text=True)
    assert (result.returncode==0)==(corruption is None),result.stderr
    if corruption is None:assert (tmp_path/'parsed-1.json').is_file()
    smoke=(ROOT/'tests/ll_stage_timer/timer_smoke.cpp').read_text(encoding='utf-8')
    assert not any(old in smoke for old in ('timer.Begin(','timer.End(','timer.Count(','SyncAll','PipeBarrier'))
    assert smoke.index('sparseEntry=')<smoke.index('DeviceTimer timer')
