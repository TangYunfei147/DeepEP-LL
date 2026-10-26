#pragma once
#include "kernel_operator.h"
#include "ll_reference_profile_layout.h"
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include "shmem.h"
#endif
#if defined(DEEPEP_LL_REFERENCE_PROFILE) && defined(DEEPEP_LL_STAGE_TIMER)
#error LL reference profiling and the legacy stage timer are mutually exclusive
#endif

namespace LlReferenceProfile {
// Reference dispatch style: one clock and one scalar GM store per fixed point.
// The only repeated intervals accumulate in receiver Scalar locals. No UB,
// lane, event, pipeline drain or per-packet bookkeeping is added here.
class DeviceProfile {
public:
    __aicore__ static inline uint64_t EntryClock(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return address ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Init(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        record_ = address ? reinterpret_cast<__gm__ uint64_t*>(address) +
                            AscendC::GetBlockIdx() * kWordsPerCore : nullptr;
        valid_ = 0U;
#endif
    }
    __aicore__ inline bool Enabled() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ != nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Clock() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Value(uint64_t value)
    {
        static_assert(Word <= 21U || (Word >= 24U && Word <= 43U) || Word == 57U,
                      "reference point/counter index");
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        record_[Word] = value;
        valid_ |= uint64_t(1) << Word;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Mark()
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) Value<Word>(AscendC::GetSystemCycle());
#endif
    }
    __aicore__ inline void Finish(uint32_t rank, uint32_t world, uint32_t bs)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        Mark<57>();
        record_[50] = rank;
        record_[51] = AscendC::GetBlockIdx();
        record_[52] = world;
        record_[53] = AscendC::GetBlockNum();
        record_[54] = bs;
        record_[55] = kSchema;
        record_[56] = 1U;  // dispatch only
        record_[58] = valid_;
        record_[59] = kMagic;
        record_[60] = 1U;  // observe; existing business dependencies only
        record_[61] = 1U;
        // Python supplies a fresh zeroed tensor; reserved/absent words remain
        // zero. Publish only our 512B diagnostic record after business finish.
        // This matches the reference dispatch cache publication, not PIPE_ALL.
        // Pinned SHMEM's helper includes the cacheline containing addr+length.
        // Use the last byte so an aligned record never flushes its neighbor.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_), kWordsPerCore * sizeof(uint64_t) - 1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    }
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __gm__ uint64_t* record_ = nullptr;
    uint64_t valid_ = 0U;
#endif
};
}  // namespace LlReferenceProfile
