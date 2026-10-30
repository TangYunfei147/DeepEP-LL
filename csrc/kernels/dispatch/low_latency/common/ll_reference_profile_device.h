#pragma once
#include "kernel_operator.h"
#include "ll_reference_profile_layout.h"
#include "ll_core_config.h"
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include "shmem.h"
#endif

namespace LlReferenceProfile {
// Fixed points use the reference's one clock/one Scalar GM store. Packet
// identities have no per-packet clock. MTE has no RDMA-stage/put/quiet detail.
class DeviceProfile {
public:
    __aicore__ static inline uint64_t EntryClock(GM_ADDR address) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return address ? AscendC::GetSystemCycle() : 0U;
#else
        (void)address; return 0U;
#endif
    }
    __aicore__ inline void Init(GM_ADDR address) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        record_=address ? reinterpret_cast<__gm__ uint64_t*>(address)+AscendC::GetBlockIdx()*kWordsPerCore : nullptr;
        valid_=0U;
        for(uint32_t i=0;i<32U;++i) send_[i]=0U;
        group_=false;
#else
        (void)address;
#endif
    }
    __aicore__ inline bool Enabled() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_!=nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Clock() const {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template<uint32_t Word> __aicore__ inline void Value(uint64_t value) {
        static_assert(Word<=21U || (Word>=24U && Word<=43U) || Word==57U,"reference point/counter index");
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(record_) { record_[Word]=value; valid_|=uint64_t(1)<<Word; }
#else
        (void)value;
#endif
    }
    template<uint32_t Word> __aicore__ inline void Mark() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(record_) Value<Word>(AscendC::GetSystemCycle());
#endif
    }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void SendBegin(bool group) {
        if(!record_) return;
        group_=group;
        const uint64_t order=++send_[24];
        if(group) {
            if(++send_[2]==1U) { send_[5]=Clock(); send_[22]=order; }
            send_[23]=order;
        }
    }
    __aicore__ inline void GroupSubmitted() { if(record_ && group_) send_[6]=Clock(); }
    __aicore__ inline void SendLanes(uint32_t lanes) { if(record_) send_[25]=lanes; }
    __aicore__ inline void Detail(uint32_t word,uint64_t value) { if(record_) record_[word]=value; }
    __aicore__ inline GM_ADDR PacketLogAddress() const {
        return record_ ? reinterpret_cast<GM_ADDR>(record_+kPacketLogWord) : nullptr;
    }
#endif
    __aicore__ inline void Finish(uint32_t rank,uint32_t world,uint32_t bs) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(!record_) return;
        Mark<57>(); // Includes diagnostic packet-log export, not header/cache publication.
        send_[0]=3U; // MTE shared-receive detail: no RDMA-specific fields.
        for(uint32_t i=0;i<32U;++i) record_[64U+i]=send_[i];
        record_[22]=DispatchDedup::kLlSendCoreNum; record_[23]=DispatchDedup::kLlRecvCoreNum;
        record_[50]=rank; record_[51]=AscendC::GetBlockIdx(); record_[52]=world;
        record_[53]=AscendC::GetBlockNum(); record_[54]=bs; record_[55]=kSchema;
        record_[56]=1U; record_[58]=valid_; record_[59]=kMagic;
        record_[60]=1U; record_[61]=1U; record_[62]=0U;
        record_[63]=AscendC::GetBlockIdx()<DispatchDedup::kLlSendCoreNum ? 0U:1U;
        // SDK helper includes addr+length. Last byte avoids touching another row.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_),kPacketLogWord*sizeof(uint64_t)-1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#else
        (void)rank; (void)world; (void)bs;
#endif
    }
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __gm__ uint64_t* record_=nullptr;
    uint64_t valid_=0U,send_[32]{};
    bool group_=false;
#endif
};
} // namespace LlReferenceProfile
