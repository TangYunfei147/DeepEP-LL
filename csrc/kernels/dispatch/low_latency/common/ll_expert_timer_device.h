#pragma once
#include "kernel_operator.h"
#include "ll_expert_timer_layout.h"
namespace LlExpertTimer {
class DeviceTimer {
public:
    __aicore__ inline void Init(GM_ADDR address, uint32_t experts, uint32_t rank, uint32_t mode) {
#ifdef DEEPEP_LL_STAGE_TIMER
        address_ = address; experts_ = experts; mode_ = mode;
        if (!address_) return;
        const uint32_t words = (kHeaderRows + experts * StageCount) * kFields;
        data_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address) + AscendC::GetBlockIdx() * words);
        data_(0)=kMagic; data_(1)=kSchema; data_(2)=rank; data_(3)=AscendC::GetBlockIdx();
        data_(4)=experts; data_(5)=mode; data_(6)=0; data_(7)=StageCount;
#endif
    }
    __aicore__ inline bool Enabled() const {
#ifdef DEEPEP_LL_STAGE_TIMER
        return address_ != nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline void CountPacket(uint32_t expert) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if (expert >= experts_) { AscendC::Trap(); return; }
        const uint32_t row=(kHeaderRows + expert*StageCount)*kFields;
        const uint64_t now=AscendC::GetSystemCycle();
        if (!data_(row+3)) data_(row)=now;
        data_(row+1)=now; data_(row+3)=data_(row+3)+1;
#endif
    }
    __aicore__ inline void Start(uint32_t expert, Stage stage) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if (expert >= experts_ || stage == Packets || stage >= StageCount) { AscendC::Trap(); return; }
        if (mode_ == 2U) AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t now=AscendC::GetSystemCycle();
        const uint32_t row=(kHeaderRows + expert*StageCount + uint32_t(stage))*kFields;
        if (!data_(row+3)) data_(row)=now;
        data_(row+4)=now; data_(row+5)=1;
#endif
    }
    __aicore__ inline void End(uint32_t expert, Stage stage) {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (!address_) return;
        if (expert >= experts_ || stage == Packets || stage >= StageCount) { AscendC::Trap(); return; }
        if (mode_ == 2U) AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t now=AscendC::GetSystemCycle();
        const uint32_t row=(kHeaderRows + expert*StageCount + uint32_t(stage))*kFields;
        data_(row+1)=now; data_(row+2)=data_(row+2)+now-data_(row+4);
        data_(row+3)=data_(row+3)+1; data_(row+5)=0;
#endif
    }
    __aicore__ inline void Finish() {
#ifdef DEEPEP_LL_STAGE_TIMER
        if (address_) data_(6)=1;
#endif
    }
private:
#ifdef DEEPEP_LL_STAGE_TIMER
    GM_ADDR address_=nullptr;
    uint32_t experts_=0, mode_=1;
    AscendC::GlobalTensor<uint64_t> data_;
#endif
};
}
