#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_receive_probe_pipeline.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_receive_priority.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h"
using namespace DispatchDedup;
using namespace AscendC;

struct Model {
    TPipe pipe;
    TBuf<> prefixBuffer;
    ServerDedupReceiveTasks tasks;
    std::vector<bool> ready, firstHop;
    std::set<unsigned> consumed;
    unsigned ids[32]{}, probeCount=0, probeOffset=0, probeStarts=0, primes=0, issued=0;
    bool outstanding=false, cascade=false;
    LlReceiveProbeResult captured;
    struct Active { LlReceivePriority order; unsigned offset=0, primed=0; };

    explicit Model(unsigned size) : ready(size), firstHop(size) {
        pipe.InitBuffer(prefixBuffer,64);
        auto prefix=prefixBuffer.Get<int32_t>();
        for (unsigned i=0;i<16;++i) prefix.SetValue(i,int32_t(size));
        tasks.Build(prefix,0,16,16,size?size:1,0,&pipe,1);
        assert(tasks.Count()==size);
    }
    std::vector<unsigned> Pending() {
        std::vector<unsigned> result;
        for(unsigned i=0;i<tasks.PendingCount();++i) result.push_back(tasks.PendingIndex(i));
        return result;
    }
    void StartProbe(unsigned start,unsigned count,unsigned offset) {
        assert(!outstanding && count>0 && count<=16 && (offset==0 || offset==16));
        assert(start+count<=tasks.PendingCount());
        // A is primed before any B flag submission. The real receiver queues
        // up to six payload lanes at this point, even though none is consumed.
        if(offset) assert(primes>0);
        captured={};
        for(unsigned i=0;i<count;++i) {
            const unsigned j=tasks.PendingIndex(start+i);
            assert(!consumed.count(j));
            ids[offset+i]=j;
            if(ready[j]) captured.readyMask|=1U<<i;
            if(ready[j] && firstHop[j]) captured.keepFlagsMask|=1U<<i;
        }
        // Latch the older observation. A relay during ConsumeReady may make
        // B ready, but it cannot change this snapshot into a false true result.
        probeCount=count;probeOffset=offset;outstanding=true;++probeStarts;
    }
    LlReceiveProbeResult FinishProbe(unsigned count) {
        assert(outstanding && probeCount==count);
        outstanding=false;
        return captured;
    }
    Active PrimeReady(unsigned offset,unsigned count,LlReceiveProbeResult probe) {
        assert(!outstanding && offset==probeOffset);
        ++primes;
        Active active{LlMakeReceivePriority(count,probe.readyMask,probe.keepFlagsMask),offset,0};
        active.primed=std::min(6U,active.order.count);
        issued+=active.primed;
        return active;
    }
    void ConsumeReady(Active& active) {
        for(unsigned n=0;n<active.order.count;++n) {
            const unsigned j=ids[active.offset+active.order.At(n)];
            assert(ready[j] && consumed.insert(j).second);
            if(cascade && firstHop[j] && (j^1U)<ready.size()) ready[j^1U]=true;
        }
    }
    void RemoveProbe(unsigned start,unsigned count,unsigned mask) {
        assert(!outstanding);
        for(unsigned n=count;n>0;--n) if(mask&(1U<<(n-1)))
            assert(consumed.count(tasks.PendingIndex(start+n-1)));
        if(!mask || tasks.DropReadySuffix(start,count,mask)) return;
        for(unsigned n=count;n>0;--n)
            if(mask&(1U<<(n-1))) tasks.RemovePending(start+n-1);
    }
};

static void CheckOnce(unsigned size,unsigned scan,unsigned pattern,std::mt19937& rng) {
    Model model(size);
    for(unsigned j=0;j<size;++j) {
        model.ready[j]=pattern==0?false:pattern==1?true:pattern==2?(j%2==0):bool(rng()&1U);
        model.firstHop[j]=bool(rng()&1U);
    }
    const unsigned oldScan=scan, starts=model.probeStarts;
    const bool result=LlRunReceiveProbePair(model,model.tasks.PendingCount(),scan);
    assert(!model.outstanding && model.probeStarts-starts<=2U);
    assert(result==!model.consumed.empty());
    auto remaining=model.Pending();
    assert(std::set<unsigned>(remaining.begin(),remaining.end()).size()==remaining.size());
    for(unsigned j=0;j<size;++j)
        assert(model.consumed.count(j)+std::count(remaining.begin(),remaining.end(),j)==1);
    if(size && !result) {
        const unsigned normalized=oldScan>=size?0:oldScan;
        assert(scan==normalized+std::min(32U,size-normalized));
    }
}

int main() {
    assert(kLlReceiveProbeExtraBytes==64U);
    for(unsigned remaining:{0U,1U,63U,64U,65U})
        assert(LlReceiveProbePrefetchFits(100U,100U+remaining)==(remaining>=64U));
    assert(!LlReceiveProbePrefetchFits(101,100));
    std::mt19937 rng(74102);
    for(unsigned size=0;size<=80;++size) for(unsigned scan=0;scan<=size+1;++scan)
        for(unsigned pattern=0;pattern<5;++pattern) CheckOnce(size,scan,pattern,rng);

    // All terminals begin unready. Source FIRST_HOP arrivals are delayed, and
    // a terminal can become ready only while A/B is being consumed elsewhere.
    for(unsigned seed=0;seed<80;++seed) {
        Model model(33U+2U*(seed%49));model.cascade=true;
        for(unsigned j=0;j<model.ready.size();++j) model.firstHop[j]=(j%2)==0;
        unsigned scan=seed%model.ready.size(),round=0;
        while(model.tasks.PendingCount() && round<2000) {
            for(unsigned j=0;j<model.ready.size();j+=2)
                if(round>=((j*7U+seed)%13U)) model.ready[j]=true;
            LlRunReceiveProbePair(model,model.tasks.PendingCount(),scan);
            assert(!model.outstanding);++round;
        }
        assert(!model.tasks.PendingCount() && model.consumed.size()==model.ready.size());
    }

    // The measured shape keeps all six payload lanes on every physical core,
    // including R0's larger prefix workspace, for all three input layouts.
    for(auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,
                   EpServerDedupInputMode::PrequantizedFp8Packs}) {
        auto data=DispatchServerDedupDataLayout(256,3584,8,384,mode);
        for(unsigned core=0;core<56;++core) {
            auto plan=LlMakeOverlapLayout(data,256,8,384,16,0,core,mode,true);
            assert(plan.enabled && plan.recvProbePrefetch);
            assert(plan.sendLanes==6 && plan.recvLanes==6);
            assert(plan.totalBytes<=kLlDispatchUbBytes);
        }
    }
}
