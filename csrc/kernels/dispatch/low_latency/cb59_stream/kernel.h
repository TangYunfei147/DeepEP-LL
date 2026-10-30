#include "../common/ll_reference_profile_device.h"
// Copyright (c) 2026, Lu Lu
#pragma once

#include "../cb59/device/dispatch_server_dedup_counts.h"
#include "../cb59/device/dispatch_server_dedup_send.h"
#include "../device/dispatch_server_dedup_magic.h"
#include "receiver.h"
#include "stream_counts.h"

namespace DispatchDedup::Cb59Stream {
// One implementation per mode TU. The source-side algorithm is the immutable
// cb59 replay; only eligible receive calls use record-owned streaming control.
template<class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class DispatchMemoryServerDedupKernel {
public:
/* LL_REFERENCE_BEGIN */
__aicore__ inline void SetProfile(LlReferenceProfile::DeviceProfile* profile) { profile_ = profile; }
__aicore__ inline void ExportProfile() { receiver_.ExportProfile(); }
/* LL_REFERENCE_END */

    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
                                GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut,
                                uint32_t bs, uint32_t h, uint32_t topK, uint32_t experts,
                                uint32_t expertTokenNumsType, TPipe* pipe, GM_ADDR inputScales = nullptr,
                                bool enableDedup = true)
    {
/* LL_REFERENCE_BEGIN */
counts_.SetProfile(profile_); sender_.SetProfile(profile_); receiver_.SetProfile(profile_); streamCounts_.SetProfile(profile_);
/* LL_REFERENCE_END */



        pipe_ = pipe;
        const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = AscendC::GetBlockIdx();
        sendCores_ = AscendC::GetBlockNum() / 2U;
        stream_ = !enableDedup && LlUseStreamReceive(uint32_t(args->rankSize), bs, 32U);

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<2>();
/* LL_REFERENCE_END */
const uint32_t magic = RotateDispatchServerDedupMagic(args, core_, pipe);
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<3>();
/* LL_REFERENCE_END */


        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts,
                     expertTokenNumsType, magic, pipe);
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<4>();
/* LL_REFERENCE_END */


        if (core_ < sendCores_) {
            sender_.Init(args, x, expertIds, expertScales, sourceMaskOut, expertScalesOut,
                         bs, h, topK, experts, pipe, magic, inputScales, enableDedup);
        } else {
            receiver_.Init(args, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut,
                           relayReadIndexOut, bs, h, topK, experts, pipe, magic, enableDedup);
            if (stream_)
                streamCounts_.Init(args, expertTokenNumsOut, sendCountsOut, bs, topK, experts,
                                   expertTokenNumsType, magic, pipe);
        }

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<5>();
/* LL_REFERENCE_END */
}

    __aicore__ inline void Process()
    {
        counts_.SendCnt();
        if (core_ < sendCores_) {
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes());
        } else if (stream_) {
            // Same aliasing pattern as cb59's countPool: the vector count work
            // has retired, while outgoing count records remain outside scratch.
            pipe_->InitBufPool(streamPool_, Cb59Replay::kEpServerDedupCountUbBytes - counts_.ReservedBytes(),
                               counts_.ScratchPool());
            streamCounts_.Prepare(streamPool_);
            // Means this slice's exact dense output offset is known. Other
            // receivers may still be waiting for their own counts/predecessors.

            receiver_.Process( &streamCounts_, &streamPool_, counts_.ReservedBytes());
        } else {
            // Preserve the full cb59 protocol outside EP16 small-direct calls.
            counts_.RecvCnt();
            counts_.CalcCumsum();
            counts_.CompleteCumsum();

/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<16>();
/* LL_REFERENCE_END */
AscendC::PipeBarrier<PIPE_ALL>();
/* LL_REFERENCE_BEGIN */
if (profile_) profile_->Mark<17>();
/* LL_REFERENCE_END */


            receiver_.Process();
        }
        counts_.Finish();

    }


/* LL_REFERENCE_BEGIN */
private:
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;

public:
/* LL_REFERENCE_END */
private:

    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0U, sendCores_ = 0U;
    bool stream_ = false;
    Cb59Replay::ServerDedupCounts counts_;
    Cb59Replay::ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    StreamCounts streamCounts_;
    TBufPool<TPosition::VECCALC, 16> streamPool_;
};
}  // namespace DispatchDedup::Cb59Stream
