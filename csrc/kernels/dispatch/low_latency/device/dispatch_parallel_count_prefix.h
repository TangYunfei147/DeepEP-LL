#pragma once
#include "dispatch_count_ready_extract.h"
// Included after ServerDedupCounts is defined. Small non-dedup only.
namespace DispatchDedup {
__aicore__ inline void ServerDedupCounts::ParallelControl(GlobalTensor<uint32_t>& view,uint64_t offset)
{
    view.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args_->peerMems[epRankId_]+offset));
    view.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}
__aicore__ inline void ServerDedupCounts::PrepareParallelRecvCnt()
{
    recStatusNumPerCore_=0U;
    if(aivId_<kLlSendCoreNum)return;


    const uint32_t producers=LlSmallBPrefixProducerCount(epWorldSize_);
    const auto slice=LlParallelPrefixSlice(rscvStatusNum_,producers,aivId_-kLlSendCoreNum);
    startStatusIndex_=slice.begin;endStatusIndex_=slice.begin+slice.count;recStatusNumPerCore_=slice.count;
    RecvPool().InitBuffer(prefixPublishBuf_,32U);
    const uint32_t workBytes=Ceil(slice.count*4U,256U)*256U;
    const uint32_t receiver=aivId_-kLlSendCoreNum;
    const uint32_t snapshotBytes=slice.count ? Ceil(receiver*32U,256U)*256U : 0U;
    const uint32_t sharedBytes=workBytes>snapshotBytes?workBytes:snapshotBytes;
    if(sharedBytes)RecvPool().InitBuffer(sumContinueBuf_,sharedBytes);
    auto publication=prefixPublishBuf_.Get<uint32_t>();
    GlobalTensor<uint32_t> own;

    ParallelControl(own,LlPrefixTotalControlByte(aivId_));
    DataCopy(publication,own,8U);
    SyncFunc<HardEvent::MTE2_S>();
    prefixGeneration_=LlNextPrefixGeneration(publication.GetValue(1U));

    // Empty slices also participate in the generation/total protocol.
    if(!slice.count)return;
    RecvPool().InitBuffer(scalarBuf_,96U);
    RecvPool().InitBuffer(waitStatusBuf_,Ceil(slice.count*32U+256U,256U)*256U);
    RecvPool().InitBuffer(gatherMaskOutBuf_,Ceil(slice.count*4U,256U)*256U);
}
__aicore__ inline bool ServerDedupCounts::TryParallelRecvCnt()
{
    if(!recStatusNumPerCore_)
/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<25>();
if (profile_) profile_->Mark<11>();
/* LL_REFERENCE_END */
return true;
/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */





    auto status=waitStatusBuf_.Get<float>();
    DataCopy(status,windowInstatusFp32Tensor_[uint64_t(startStatusIndex_)*CNT_GM_STRIDE_BLOCKS*8U],
        DataCopyParams{uint16_t(recStatusNumPerCore_),1U,uint16_t(CNT_GM_STRIDE_BLOCKS-1U),0U});


    SyncFunc<HardEvent::MTE2_V>();

    if(epWorldSize_==16U && LlSmallBPrefixProducerCount(epWorldSize_)==1U){

        auto complete=scalarBuf_.GetWithOffset<uint32_t>(8U,32U);
        asc_vf_call<P1CountReadyAndExtract>(
            reinterpret_cast<__ubuf__ uint32_t*>(status.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(gatherMaskOutBuf_.Get<uint32_t>().GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(complete.GetPhyAddr()),recStatusNumPerCore_);
        // Retire every VF read/write before a retry overwrites the snapshot.
        // Extracted counts remain private scratch until the ready result passes.
        SyncFunc<HardEvent::V_S>();
        const bool ready=complete.GetValue(0U)!=0U;

        if(!ready)return false;
    }else{

    auto gather=gatherMaskOutBuf_.Get<float>();
    auto pattern=scalarBuf_.GetWithOffset<uint32_t>(8U,0U);
    auto sum=scalarBuf_.GetWithOffset<float>(8U,32U);
    pattern.SetValue(0U,1U); SyncFunc<HardEvent::S_V>();
    uint64_t n=0;
    GatherMask(gather,status,pattern,true,2U,{1U,uint16_t(recStatusNumPerCore_),1U,0U},n);
    PipeBarrier<PIPE_V>();
    ReduceSum(sum,gather,sumContinueBuf_.Get<float>(),recStatusNumPerCore_);
    SyncFunc<HardEvent::V_S>();
    if(sum.GetValue(0U)!=float(recStatusNumPerCore_)){

        return false;
    }


    pattern.SetValue(0U,2U); SyncFunc<HardEvent::S_V>();
    GatherMask(gather,status,pattern,true,2U,{1U,uint16_t(recStatusNumPerCore_),1U,0U},n);
    SyncFunc<HardEvent::V_S>();

    }
    WaitDispatchClearStatus(); // Only this producer's disjoint count slice.


/* LL_REFERENCE_BEGIN */
{
/* LL_REFERENCE_END */

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<25>();
if (profile_) profile_->Mark<11>();
/* LL_REFERENCE_END */
return true;
/* LL_REFERENCE_BEGIN */
}
/* LL_REFERENCE_END */

}
__aicore__ inline void ServerDedupCounts::CalcParallelCumsum()
{

    const uint32_t receiver=aivId_-kLlSendCoreNum;
    const uint32_t producers=LlSmallBPrefixProducerCount(epWorldSize_);
    LocalTensor<uint32_t> prefix;
    uint32_t total=0;
    if(recStatusNumPerCore_){

        prefix=gatherMaskOutBuf_.Get<uint32_t>();
        SyncFunc<HardEvent::S_V>();
        asc_vf_call<R0PrefixScan>(reinterpret_cast<__ubuf__ uint32_t*>(prefix.GetPhyAddr()),recStatusNumPerCore_);
        SyncFunc<HardEvent::V_S>(); total=prefix.GetValue(recStatusNumPerCore_-1U);

    }
    if(receiver<producers){}

    auto publication=prefixPublishBuf_.Get<uint32_t>();
    publication.SetValue(0U,total);publication.SetValue(1U,prefixGeneration_);
    GlobalTensor<uint32_t> sums;
    ParallelControl(sums,LlPrefixTotalControlByte(kLlSendCoreNum));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(sums[uint64_t(receiver)*128U],publication,8U);

    // Every R advances its private generation, including consumers with no
    // prefix slice. Its zero-total record is not a producer dependency and the
    // immutable source remains live until Finish (or the existing fallback drain).
    if(receiver>=producers)return;
    // Published total stays in its fixed slot until the next kernel.
    // A slice only needs preceding totals. R0 and empty slices need none.
    uint32_t offset=0;
    if(receiver && recStatusNumPerCore_){
        // Reuse retired count-reduction work instead of another UB allocation.
        auto snapshot=sumContinueBuf_.Get<uint32_t>();
        const DataCopyParams rows{uint16_t(receiver),1U,15U,0U};
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<26>();
/* LL_REFERENCE_END */

        {

          while(true){


            DataCopy(snapshot,sums,rows); SyncFunc<HardEvent::MTE2_S>();


            bool ready=true;
            for(uint32_t p=0;p<receiver;++p)
                if(snapshot.GetValue(p*8U+1U)!=prefixGeneration_){ready=false;break;}

            if(ready)break;
          }
        }

        for(uint32_t p=0;p<receiver;++p)offset+=snapshot.GetValue(p*8U);
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<27>();
/* LL_REFERENCE_END */


    }
    if(recStatusNumPerCore_){

        if(offset)for(uint32_t i=0;i<recStatusNumPerCore_;++i)prefix.SetValue(i,prefix.GetValue(i)+offset);


        SyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<uint32_t> workspace,output;
        workspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(recvCntWorkspaceGM_));
        workspace.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sendCountsOutGM_));
        const DataCopyExtParams part{1U,recStatusNumPerCore_*4U,0U,0U,0U};
        DataCopyPad(output[startStatusIndex_],prefix,part);
        DataCopyPad(workspace[startStatusIndex_],prefix,part);

    }
    // Finish prefix/count cleanup and old total source before reusing publication.
    {  SyncFunc<HardEvent::MTE3_S>(); }
    GlobalTensor<uint32_t> ready;
    ParallelControl(ready,LlPrefixReadyControlByte(aivId_));
    publication.SetValue(0U,prefixGeneration_); SyncFunc<HardEvent::S_MTE3>();

    DataCopy(ready,publication,8U);
    // UB source remains immutable through Finish or fallback's PIPE_ALL.
}
__aicore__ inline void ServerDedupCounts::CompleteParallelCumsum()
{
    if(!overlap_.Async()){PipeBarrier<PIPE_ALL>();}
    if(!overlap_.enabled)tpipe_->Reset();

    SetCtrlSpr<60,60>(0);
    const uint32_t producers=LlSmallBPrefixProducerCount(epWorldSize_);
    tpipe_->InitBuffer(readyBuf_,producers*32U);
    tpipe_->InitBuffer(prefixBuf_,Ceil(rscvStatusNum_*4U,32U)*32U);
    const uint32_t receiver=aivId_-kLlSendCoreNum;
    GlobalTensor<uint32_t> flags;
    ParallelControl(flags,LlPrefixReadyControlByte(kLlSendCoreNum));
    auto ready=readyBuf_.Get<uint32_t>();
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<28>();
/* LL_REFERENCE_END */


    {

      while(true){


        DataCopy(ready,flags,DataCopyParams{uint16_t(producers),1U,15U,0U});
        SyncFunc<HardEvent::MTE2_S>();


        bool complete=true;
        for(uint32_t p=0;p<producers;++p)
            if(ready.GetValue(p*8U)!=prefixGeneration_){complete=false;break;}

        if(complete)break;
      }
    }

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<29>();
/* LL_REFERENCE_END */
    auto prefix=prefixBuf_.Get<int32_t>();
    GlobalTensor<int32_t> workspace;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    workspace.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    {
      DataCopyPad(prefix,workspace,{1U,rscvStatusNum_*4U,0U,0U,0U},{false,0U,0U,0});
      SyncFunc<HardEvent::MTE2_S>();
    }
    if(receiver==0U){

        tpipe_->InitBuffer(tokenNumBuf_,Ceil(moeExpertNumPerRank_*8U,32U)*32U);
        auto counts=tokenNumBuf_.Get<int64_t>(); int64_t previous=0;
        for(uint32_t e=0;e<moeExpertNumPerRank_;++e){
            const int64_t cumulative=prefix.GetValue((e+1U)*epWorldSize_-1U);
            counts.SetValue(e,expertTokenNumsType_==0U?cumulative:cumulative-previous);previous=cumulative;
        }
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsOutGMTensor_,counts,{1U,moeExpertNumPerRank_*8U,0U,0U,0U});
        // Keep legacy alternating epoch current across small/dedup switches.
        ready.SetValue(0U,prefixEpoch_); SyncFunc<HardEvent::S_MTE3>();
        DataCopy(prefixReadyGm_,ready,8U);
    }
    // Fallback receiver resets TPipe; retire metadata and epoch sources first.
    if(!overlap_.enabled){PipeBarrier<PIPE_ALL>();}

}
}
