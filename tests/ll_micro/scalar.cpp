#include "kernel_operator.h"
#include "../../csrc/kernels/dispatch/low_latency/common/ll_local_q.h"
using namespace AscendC;
// Isolated operation costs, not a production dispatch timing.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void ll_scalar_micro(
    GM_ADDR ids, GM_ADDR result, uint32_t mode, uint32_t repeats, uint32_t count)
{
    TPipe pipe; TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 16384);
    auto src=storage.Get<uint32_t>(); auto dst=src[2048]; auto hist=src[3072];
    for(uint32_t i=0;i<2048;++i)src.SetValue(i,i%384);
    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
    Duplicate(dst,0U,1024); Duplicate(hist,0U,384);
    PipeBarrier<PIPE_ALL>();
    uint64_t begin=GetSystemCycle();
    for(uint32_t r=0;r<repeats;++r) {
        if(mode==1) { // Original receiver snapshot: 3 scalar loads + stores.
            for(uint32_t j=0;j<count;++j)
                for(uint32_t k=0;k<3;++k) dst.SetValue(j*4+k,src.GetValue(j*4+k));
        } else if(mode==2) { // Vector-copy candidate, including readiness cost.
            SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
            Copy(dst,src,count*4);
            SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
        } else if(mode==7) {
            GlobalTensor<uint32_t> routes;routes.SetGlobalBuffer((__gm__ uint32_t*)ids);
            SetFlag<HardEvent::S_MTE2>(0);WaitFlag<HardEvent::S_MTE2>(0);
            DataCopy(src,routes,2048);
            SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        } else if(mode==3 || mode==4) {
            // Same 72-route shard and same seed reset for both representations.
            SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
            Duplicate(hist,1U,384);
            SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
            if(mode==3) DispatchDedup::LlFinishLocalQ(src,hist,dst,8,{384,72});
            else for(uint32_t j=0;j<72;++j) {
                uint32_t e=src.GetValue(384+j),q=hist.GetValue(e);
                dst.SetValue(j,e|(q<<16));hist.SetValue(e,q+1);
            }
        } else if(mode==5) {
            for(uint32_t j=0;j<count;++j) dst.SetValue(j,j);
        } else if(mode==6) {
            SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
            CreateVecIndex(dst.ReinterpretCast<int32_t>(),0,count);
            SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
        }
    }
    PipeBarrier<PIPE_ALL>();
    uint64_t end=GetSystemCycle();
    // Consume outputs outside measured interval; prevents dead-result elimination.
    uint64_t checksum=0;
    if(mode==1 || mode==2) {
        for(uint32_t j=0;j<count;++j)for(uint32_t k=0;k<3;++k)checksum+=dst.GetValue(j*4+k);
    } else if(mode==3 || mode==4) {
        for(uint32_t j=0;j<72;++j)checksum+=dst.GetValue(j);
    } else if(mode==5 || mode==6) {
        for(uint32_t j=0;j<count;++j)checksum+=dst.GetValue(j);
    }
    if(mode==7 || mode==8)for(uint32_t j=0;j<2048;++j)checksum+=src.GetValue(j);
    if(mode==9)for(uint32_t j=0;j<count;++j)for(uint32_t k=0;k<3;++k)checksum+=src.GetValue(j*4+k);
    GlobalTensor<uint64_t> out;out.SetGlobalBuffer((__gm__ uint64_t*)result+GetBlockIdx()*4);
    auto record=src.ReinterpretCast<uint64_t>();
    record.SetValue(0,end-begin);record.SetValue(1,checksum);record.SetValue(2,mode);record.SetValue(3,count);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopy(out,record,4);PipeBarrier<PIPE_ALL>();
}
extern "C" void scalar_micro_do(uint32_t cores,void* stream,GM_ADDR ids,GM_ADDR result,uint32_t mode,uint32_t repeats,uint32_t count) {
    ll_scalar_micro<<<cores,nullptr,stream>>>(ids,result,mode,repeats,count);
}
