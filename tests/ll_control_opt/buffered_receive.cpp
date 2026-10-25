#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <set>
#include <random>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_early_receive.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_input_prefetch.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_receive_probe_pipeline.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h"
using namespace DispatchDedup;
using namespace AscendC;

struct InputOps {
    unsigned submitted=0,direct=0,owned=0,peak=0;
    std::vector<unsigned> order;
    void Issue(const LlSendPacketTask&) {++owned;peak=std::max(peak,owned);assert(owned<=2);}
    void Consume(const LlSendPacketTask& t) {assert(owned);--owned;++submitted;if(!t.mask)++direct;order.push_back(t.token);}
    void Take() {}
};
static void Cadence() {
    for(unsigned fh=0;fh<20;++fh) for(unsigned direct=0;direct<32;++direct) {
        LlOnePacketInput input;InputOps ops;input.Reset();
        for(unsigned i=0;i<fh;++i) input.Push(ops,LlSendPacketTask{i,0,0,0,3});
        LlSubmittedDirectCadence cadence;cadence.Reset(ops.submitted,input.HasPending());
        std::vector<unsigned> visits;
        for(unsigned i=0;i<direct;++i) {
            input.Push(ops,LlSendPacketTask{fh+i,0,0,0,0});
            if(cadence.Visit(ops.submitted)) visits.push_back(ops.direct);
        }
        input.Flush(ops); // Final DIRECT never creates a useless late visit.
        assert(!input.HasPending() && !ops.owned && ops.submitted==fh+direct);
        std::vector<unsigned> expected;
        for(unsigned n=2;n<direct;n+=2) expected.push_back(n);
        assert(visits==expected);
        for(unsigned i=0;i<ops.order.size();++i) assert(ops.order[i]==i);
    }
}

struct Receiver {
    TPipe pipe;TBuf<> prefixBuffer;ServerDedupReceiveTasks tasks;
    std::vector<bool> ready,first;
    std::vector<unsigned> staged;
    std::set<unsigned> consumed;
    unsigned ids[32]{},cursor=0,probes=0,issued=0,consumedCount=0;
    bool immediate=false,outstanding=false;
    LlReceiveProbeResult observation;
    struct Active {LlReceivePriority order;unsigned offset;};
    explicit Receiver(unsigned size,bool now):ready(size),first(size),immediate(now) {
        pipe.InitBuffer(prefixBuffer,64);auto prefix=prefixBuffer.Get<int32_t>();
        for(unsigned i=0;i<16;++i)prefix.SetValue(i,int32_t(size));
        tasks.Build(prefix,0,16,16,size?size:1,0,&pipe,1);
    }
    LlReceiveProbeResult Probe(unsigned start,unsigned count) {
        StartProbe(start,count,0);return FinishProbe(count);
    }
    void StartProbe(unsigned start,unsigned count,unsigned offset) {
        assert(!outstanding && count && count<=16);outstanding=true;++probes;observation={};
        for(unsigned i=0;i<count;++i) {
            unsigned j=tasks.PendingIndex(start+i);ids[offset+i]=j;
            assert(!consumed.count(j) && std::find(staged.begin(),staged.end(),j)==staged.end());
            if(ready[j]) observation.readyMask|=1U<<i;
            if(ready[j] && first[j]) observation.keepFlagsMask|=1U<<i;
        }
    }
    LlReceiveProbeResult FinishProbe(unsigned) {assert(outstanding);outstanding=false;return observation;}
    void Consume(unsigned j) {
        assert(ready[j] && consumed.insert(j).second);++consumedCount;
        if(first[j] && (j^1U)<ready.size())ready[j^1U]=true;
    }
    void Accept(unsigned count,unsigned mask) {
        auto order=LlMakeReceivePriority(count,mask,mask);assert(order.count<=2);
        for(unsigned i=0;i<order.count;++i) {
            const unsigned j=ids[order.At(i)];assert(ready[j] && first[j]);++issued;
            if(immediate)Consume(j);
            else {assert(staged.size()<6);staged.push_back(j);}
        }
    }
    void Remove(unsigned start,unsigned count,unsigned mask) {RemoveProbe(start,count,mask);}
    void RemoveProbe(unsigned start,unsigned count,unsigned mask) {
        if(!mask || tasks.DropReadySuffix(start,count,mask))return;
        for(unsigned n=count;n>0;--n)if(mask&(1U<<(n-1)))tasks.RemovePending(start+n-1);
    }
    Active PrimeReady(unsigned offset,unsigned count,LlReceiveProbeResult probe) {
        return {LlMakeReceivePriority(count,probe.readyMask,probe.keepFlagsMask),offset};
    }
    void ConsumeReady(Active& a) {
        for(unsigned i=0;i<a.order.count;++i){++issued;Consume(ids[a.offset+a.order.At(i)]);}
    }
    void Invariant() {
        std::set<unsigned> all=consumed;
        for(unsigned j:staged)assert(all.insert(j).second);
        for(unsigned i=0;i<tasks.PendingCount();++i)assert(all.insert(tasks.PendingIndex(i)).second);
        assert(all.size()==tasks.Count() && issued-consumedCount==staged.size());
    }
    void Step() {
        const unsigned before=probes;
        const unsigned limit=immediate?2U:std::min(2U,6U-unsigned(staged.size()));
        LlTryEarlyFirstHopWindow(*this,tasks.PendingCount(),cursor,limit);
        assert(probes-before<=1 && !outstanding);Invariant();
    }
    void Finish() {
        // Payload readiness is joined by the device's existing per-lane events;
        // this arithmetic model intentionally makes no DMA-time assertion.
        for(unsigned j:staged)Consume(j);
        staged.clear();Invariant();unsigned scan=0,rounds=0;
        while(tasks.PendingCount() && rounds++<2000) {
            LlRunReceiveProbePair(*this,tasks.PendingCount(),scan);Invariant();
        }
        assert(!tasks.PendingCount() && consumed.size()==tasks.Count() && !outstanding);
    }
};

static void States() {
    for(bool immediate:{false,true}) for(unsigned size=0;size<=129;++size) for(unsigned seed=0;seed<8;++seed) {
        Receiver rx(size,immediate);
        for(unsigned j=0;j<size;++j)rx.first[j]=(j%2)==0;
        // Delayed ready FIRST_HOPs; compact terminals become ready only after
        // their FIRST_HOP is consumed. A callback never waits for that change.
        for(unsigned visit=0;visit<20;++visit) {
            for(unsigned j=0;j<size;j+=2)if(visit>=(j+seed)%7U)rx.ready[j]=true;
            rx.Step();
            if(!immediate)assert(rx.consumed.empty());
        }
        if(!immediate && rx.staged.size()==6) {unsigned p=rx.probes;rx.Step();assert(rx.probes==p);}
        for(unsigned j=0;j<size;j+=2)rx.ready[j]=true;
        rx.Finish();
    }
    Receiver compact(64,false);
    std::fill(compact.ready.begin(),compact.ready.end(),true);
    for(unsigned n=0;n<4;++n){compact.Step();assert(compact.staged.empty());}
    assert(compact.probes==4 && compact.cursor==0);compact.Finish();
}

static void Layout() {
    assert(LlEarlyReceiveIdBytes(6)==32 && !LlEarlyReceiveFits(100,131,6));
    assert(LlEarlyReceiveFits(100,132,6) && !LlEarlyReceiveFits(200,100,6));
    for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,EpServerDedupInputMode::PrequantizedFp8Packs})
    for(unsigned ranks:{16U,32U,64U,128U})for(unsigned bs:{1U,32U,256U,1024U})
    for(unsigned h:{128U,3584U,7168U})for(bool dedup:{false,true})for(unsigned core=0;core<56;++core) {
        const unsigned experts=ranks*6;
        auto data=DispatchServerDedupDataLayout(bs,h,8,experts,mode);
        auto p=LlMakeOverlapLayout(data,bs,8,experts,ranks,0,core,mode,dedup);
        if(!p.enabled){assert(!p.earlyReceive);continue;}
        assert(p.totalBytes<=230U*1024U && p.sendLanes<=6 && p.recvLanes<=6);
        if(p.earlyReceive)assert(dedup && core<kLlSendCoreNum && p.Async());
    }
    if(kLlSendCoreNum==28 || kLlSendCoreNum==30) {
        auto d=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
        for(unsigned core=0;core<56;++core) {
            auto p=LlMakeOverlapLayout(d,256,8,384,16,0,core,EpServerDedupInputMode::PrequantizedFp8Packs,true);
            assert(p.enabled && p.sendLanes==6 && p.recvLanes==6);
            assert(p.earlyReceive==(core<kLlSendCoreNum));
        }
    }
}
int main(){Cadence();States();Layout();}
