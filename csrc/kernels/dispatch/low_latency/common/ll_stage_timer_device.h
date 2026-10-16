#pragma once
#include "kernel_operator.h"
#include "ll_stage_timer_layout.h"

namespace LlStageTimer {
// One executing task per core. Dependency waits split and resume its real intervals.
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t operation, uint32_t mode)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        address_ = address; mode_ = mode;
        activeType_ = waitType_ = suspendedType_ = 0U;
        nextTask_ = 1U; kernelActive_=false;
        for (uint32_t stage=0; stage<StageCount; ++stage) counts_[stage]=0U;
        if (!address_) return;
        const uint32_t core = AscendC::GetBlockIdx();
        data_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address) + core * kWordsPerCore);
        data_(0)=kMagic; data_(1)=kSchema; data_(2)=operation;
        data_(3)=AscendC::GetBlockNum(); data_(4)=core; data_(5)=mode;
        data_(6)=0U; data_(7)=StageCount; data_(kEventErrorOffset)=0U;
        for (uint32_t word=kDataCounterOffset; word<kDataFlagsOffset; ++word) data_(word)=0U;
#ifdef DEEPEP_LL_NO_COMM
        data_(kDataFlagsOffset)=kDataNoComm;
#else
        data_(kDataFlagsOffset)=0U;
#endif
#endif
    }
    __aicore__ inline void Start(Stage stage)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        Boundary(stage);
        const uint64_t cycle=AscendC::GetSystemCycle();
        if (stage==Kernel) {
            if (kernelActive_) { Error(2U); return; }
            kernelIndex_=BeginRaw(stage,cycle,0U); kernelActive_=true; return;
        }
        if (DependencyWait(stage)) {
            if (waitType_) { Error(2U); return; }
            if (activeType_) {
                EndRaw(activeType_,activeIndex_,cycle);
                suspendedType_=activeType_; suspendedTask_=activeTask_; activeType_=0U;
            }
            waitIndex_=BeginRaw(stage,cycle,suspendedType_ ? suspendedTask_ : nextTask_++);
            waitType_=stage;
        } else {
            if (activeType_ || waitType_) { Error(2U); return; }
            activeType_=stage; activeTask_=nextTask_++;
            activeIndex_=BeginRaw(stage,cycle,activeTask_);
        }
#endif
    }
    __aicore__ inline void End(Stage stage)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        Boundary(stage);
        const uint64_t cycle=AscendC::GetSystemCycle();
        if (stage==Kernel) {
            if (!kernelActive_ || activeType_ || waitType_) { Error(4U); return; }
            EndRaw(stage,kernelIndex_,cycle); kernelActive_=false; return;
        }
        if (DependencyWait(stage)) {
            if (waitType_!=stage) { Error(4U); return; }
            EndRaw(stage,waitIndex_,cycle); waitType_=0U;
            if (suspendedType_) {
                activeType_=suspendedType_; activeTask_=suspendedTask_; suspendedType_=0U;
                activeIndex_=BeginRaw(activeType_,cycle,activeTask_);
            }
        } else {
            if (activeType_!=stage || waitType_) { Error(4U); return; }
            EndRaw(stage,activeIndex_,cycle); activeType_=0U;
        }
#endif
    }
    // Count once at the protocol DMA initiator, not again at local consumption.
    // Read counters are a subset of total counters; no synchronization is added.
    __aicore__ inline void CountData(uint32_t rank, uint32_t peer, uint64_t bytes,
                                   bool forwarding, bool read = false)
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        const uint32_t row = kDataCounterOffset + DataBucket(rank, peer, forwarding) * kDataCounterFields;
        data_(row) = data_(row) + bytes;
        data_(row + 1U) = data_(row + 1U) + 1U;
        if (read) {
            data_(row + 2U) = data_(row + 2U) + bytes;
            data_(row + 3U) = data_(row + 3U) + 1U;
        }
#endif
    }
    __aicore__ inline void Finish()
    {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address_) {
            if (activeType_ || waitType_ || suspendedType_ || kernelActive_) Error(16U);
            for (uint32_t stage=1; stage<StageCount; ++stage) data_(ControlOffset(stage))=counts_[stage];
            data_(6)=1U;
        }
#endif
    }
private:
#ifdef DEEPEP_LL_STAGE_TIMER
    __aicore__ inline void Error(uint64_t code) { data_(kEventErrorOffset) |= code; }
    __aicore__ inline void Boundary(uint32_t stage) {
        if (mode_==2U || (mode_==3U && !DependencyWait(stage))) AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline uint32_t BeginRaw(uint32_t stage, uint64_t cycle, uint64_t task) {
        const uint32_t index=counts_[stage]++;
        if (index<EventCapacity(stage)) {
            const uint32_t offset=EventOffset(stage,index);
            data_(offset)=cycle; data_(offset+2U)=task;
        } else Error(1U);
        return index;
    }
    __aicore__ inline void EndRaw(uint32_t stage, uint32_t index, uint64_t cycle) {
        if (index<EventCapacity(stage)) data_(EventOffset(stage,index)+1U)=cycle;
    }
    GM_ADDR address_=nullptr;
    uint32_t mode_=3U,activeType_=0U,waitType_=0U,suspendedType_=0U;
    bool kernelActive_=false;
    uint32_t counts_[StageCount];
    uint32_t kernelIndex_=0U,activeIndex_=0U,waitIndex_=0U;
    uint64_t nextTask_=1U,activeTask_=0U,suspendedTask_=0U;
    AscendC::GlobalTensor<uint64_t> data_;
#endif
};
}  // namespace LlStageTimer
