#pragma once
#include "ll_reference_profile_device.h"

namespace LlReferenceProfile {
// Reference-style stable-output-task log. Storage is appended inside the
// existing lane allocation using only its remaining UB, never another lane,
// buffer-pool entry or event. No per-packet clock or GM write is introduced.
class ReceiverLog {
public:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline uint32_t Bytes(DeviceProfile* profile, uint32_t spare, uint32_t tasks) {
        enabled_ = profile && profile->Enabled();
        if (!enabled_) return 0U;
        capacity_ = spare / 32U;
        if (capacity_ > kPacketTraceCapacity) capacity_ = kPacketTraceCapacity;
        if (capacity_ > tasks) capacity_ = tasks;
        return capacity_ * 32U;
    }
    __aicore__ inline void Init(AscendC::LocalTensor<uint32_t> storage) {
        if (capacity_) rows_ = storage;
    }
    __aicore__ inline void Ready(AscendC::LocalTensor<uint32_t> snapshot, uint32_t mask, uint32_t count) {
        if (!enabled_ || !mask) return;
        ++batches_;
        for (uint32_t i=0; i<count; ++i) {
            if (!(mask & (1U << i))) continue;
            const uint32_t j=snapshot.GetValue(i);
            ++ready_;
            if (j<capacity_) {
                rows_.SetValue(j*8U, ready_); rows_.SetValue(j*8U+1U,batches_);
            }
        }
    }
    __aicore__ inline void Processed(uint32_t j, AscendC::LocalTensor<uint8_t> wire,
                                    uint32_t headerOffset, uint32_t indexOffset, uint32_t slot) {
        if (!enabled_) return;
        ++processed_;
        if (j>=capacity_) return;
        auto header=wire[headerOffset].template ReinterpretCast<uint32_t>();
        const uint32_t role=header.GetValue(3U);
        const uint32_t members=role==2U ? wire[indexOffset].template ReinterpretCast<uint32_t>().GetValue(0U) : 0U;
        rows_.SetValue(j*8U+2U,processed_); rows_.SetValue(j*8U+3U,header.GetValue(0U));
        rows_.SetValue(j*8U+4U,header.GetValue(1U)); rows_.SetValue(j*8U+5U,header.GetValue(2U));
        rows_.SetValue(j*8U+6U,role|(members<<8U)); rows_.SetValue(j*8U+7U,slot);
    }
    __aicore__ inline void Export(DeviceProfile* profile, uint32_t tasks, uint32_t begin,
                                 uint32_t bs, uint32_t experts, uint32_t lanes) {
        profile->Detail(96U,capacity_); profile->Detail(97U,capacity_);
        profile->Detail(98U,tasks); profile->Detail(99U,ready_); profile->Detail(100U,processed_);
        profile->Detail(101U,lanes); profile->Detail(104U,begin); profile->Detail(105U,bs);
        profile->Detail(106U,experts); profile->Detail(107U,tasks>capacity_?1U:0U);
        profile->Detail(108U,batches_); profile->Detail(102U,profile->Clock());
        if (capacity_) {
            // Business has reached w21 before this diagnostic-only export.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::GlobalTensor<uint8_t> output;
            output.SetGlobalBuffer(profile->PacketLogAddress());
            output.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            AscendC::DataCopy(output, rows_.template ReinterpretCast<uint8_t>(), capacity_*32U);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
        profile->Detail(103U,profile->Clock());
    }
private:
    AscendC::LocalTensor<uint32_t> rows_;
    uint32_t capacity_=0, ready_=0, processed_=0, batches_=0;
    bool enabled_=false;
#endif
};
} // namespace LlReferenceProfile
