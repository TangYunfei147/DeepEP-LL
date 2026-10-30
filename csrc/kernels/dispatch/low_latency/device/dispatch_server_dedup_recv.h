// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_RECV_H
#define DISPATCH_SERVER_DEDUP_RECV_H

#include "../common/ll_overlap_layout.h"

#include "kernel_operator.h"
#include "../common/ll_reference_profile_device.h"
#include "adv_api/reduce/sum.h"
#include "../common/ep_memory_server_dedup_protocol.h"
#include "../../../common/ep_packet_receive_layout.h"
#include "dispatch_server_dedup_receive_tasks.h"
#include "../common/ll_receive_priority.h"
#include "dispatch_server_dedup_receive_packet.h"
#include "dispatch_flag_probe.h"

namespace DispatchDedup {
using namespace AscendC;

// 中文审计：完整dispatch的接收、包消费和输出元数据路径；第2层提取核不调用本类。
// 审计接收任务/包格式时继续查看dispatch_server_dedup_receive_tasks.h和receive_packet.h。
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class ServerDedupReceiver {
    static_assert(!Prequantized || WireFp8, "prequantized input requires FP8 wire");
    static constexpr EpServerDedupInputMode InputMode =
        Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs
                     : (WireFp8 ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);

public:
    // 摘要：绑定本 rank 接收输出，确定接收核编号和本轮双窗口偏移。
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, GM_ADDR expandX, GM_ADDR dynamicScales,
                                GM_ADDR tokenType, GM_ADDR destinationIndex, GM_ADDR relayReadIndex, uint32_t bs,
                                uint32_t h, uint32_t k, uint32_t experts, TPipe* pipe, int64_t magic = 0,
                                bool enableDedup = true)
    {
        args_ = args;
        coreNum_ = GetBlockNum();
        recvCoreNum_ = enableDedup ? kLlDispatchRecvCoreNum : kLlRecvCoreNum;
        pipe_ = pipe;
        enableDedup_ = enableDedup;
        bs_ = bs;
        k_ = k;
        experts_ = experts;
        localExperts_ = experts / uint32_t(args_->rankSize);
        receiver_ = enableDedup ? GetBlockIdx() : (GetBlockIdx()>=kLlSendCoreNum ? GetBlockIdx()-kLlSendCoreNum : 0U);
        hiddenBytes_ = WireFp8 ? h : h * uint32_t(sizeof(XType));
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, InputMode);
        windowOffset_ = DispatchServerDedupWindowOffset(magic);
        data_.dispatchOffset += windowOffset_;
        expandX_.SetGlobalBuffer(expandX);
        dynamicScales_.SetGlobalBuffer(dynamicScales);
        tokenType_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tokenType));
        destinationIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destinationIndex));
        relayReadIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(relayReadIndex));
    }

    __aicore__ inline void ConfigureOverlap(LlOverlapLayout plan) { overlap_=plan; }
    __aicore__ inline void SetPrefix(LocalTensor<int32_t> prefix) { sharedPrefix_=prefix; }
    // 摘要：执行接收主流程：加载前缀、构建任务、批量探测、预取并消费就绪包，最后清理本核槽位标记。
    __aicore__ inline void Process(LlReferenceProfile::DeviceProfile* profile = nullptr) {
        Prepare(profile);
        Run();
    }
    __aicore__ inline void Prepare(LlReferenceProfile::DeviceProfile* profile = nullptr)
    {
        profile_ = profile;
        if(profile_) profile_->Mark<18>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        traceEnabled_=profile_ && profile_->Enabled();
#endif

        // Overlap retains all send/count DMA sources and shares the already
        // acquired prefix. Only the drained fallback may Reset/reload.

        if (!overlap_.enabled) pipe_->Reset();
        const uint32_t prefixBytes = uint32_t(DispatchDedupAlign(uint64_t(experts_) * 4U, 32U));
        LocalTensor<int32_t> prefix=sharedPrefix_;
        if (!overlap_.enabled) {
        pipe_->InitBuffer(prefixBuf_, prefixBytes);
        GlobalTensor<int32_t> prefixGlobal;
        const auto counts = DispatchServerDedupCountLayout(experts_, coreNum_);
        prefixGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_->peerMems[args_->rank] + windowOffset_ +
                                                                       counts.cntOffset + counts.prefixOffset));
        prefixGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        prefix = prefixBuf_.Get<int32_t>();
        DataCopyPad(prefix, prefixGlobal, {1U, prefixBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        }

        if(profile_) profile_->Mark<30>();
        // 2. 按前缀定义的接收输出行均分任务，建立稳定槽位表和可压缩的 pending 表。
        // j 始终标识固定输出行/槽位；pending 下标只表示尚未消费的任务位置。
        {
        tasks_.Build(prefix, uint32_t(args_->rank), uint32_t(args_->rankSize), experts_, bs_, receiver_, pipe_, recvCoreNum_);
        }

        if(profile_) { profile_->Mark<31>(); profile_->Value<37>(tasks_.Count()); }
        // 空任务核没有拥有的接收槽，完成打点后直接返回，无需参与 packet 轮询。
        if (tasks_.Count() == 0U) {

            return;
        }

        // 3. 为每批最多 16 个 packet 的块标记分配探测/归约空间。
        // flags 和 work 额外留尾部空间，覆盖归约实现的对齐读取；先清零 flags 的无效区域。
        const uint32_t blocks = uint32_t(data_.blockCount);
        const uint32_t flagsBytes = kProbeBatch * blocks * 32U + 256U;
        // CANN 9.1/A5 Sum stores ceil(B*8/64) partials per row at a
        // 32-element stride. Reserve all rows plus a full-vector tail read.
        const uint32_t workBytes = kProbeBatch * uint32_t(DispatchDedupAlign((blocks + 7U) / 8U, 32U)) * 4U + 256U;
        pipe_->InitBuffer(flagsBuf_, flagsBytes);
        pipe_->InitBuffer(sumBuf_, kProbeBatch * 32U);
        pipe_->InitBuffer(workBuf_, workBytes);
        // LoadAlign may read past the masked tail, including inactive rows.
        Duplicate(flagsBuf_.Get<float>(), 0.0f, flagsBytes / 4U);
        ServerDedupReceiveSync<HardEvent::V_MTE2>();
        const uint32_t tablesBytes = 2U * uint32_t(DispatchDedupAlign(uint64_t(tasks_.Count()) * 4U, 32U));
        const uint32_t fixed = prefixBytes + tablesBytes + flagsBytes + workBytes + kProbeBatch * 36U;
        // 剩余 UB 作为最多 8 路预取 lane；每路包括完整 packet 和 64B 输出元数据。
        // MTE3_MTE2 事件保护 lane：旧包的输出和 relay 读完后，下一次加载才可覆盖。
        laneStride_ = uint32_t(data_.tokenStride) + 64U;
        const uint32_t available = (kEpServerDedupCountUbBytes - fixed) / laneStride_;
        lanes_ = overlap_.enabled ? overlap_.recvLanes : (available < 8U ? available : 8U);  // Safe inputs guarantee at least one lane.

        probePrefetch_ = overlap_.enabled ? overlap_.recvProbePrefetch :
            LlReceiveProbePrefetchFits(uint64_t(fixed) + uint64_t(lanes_) * laneStride_,
                                      kEpServerDedupCountUbBytes);
        pipe_->InitBuffer(probeIdxBuf_, kProbeBatch * 4U + (probePrefetch_ ? kLlReceiveProbeExtraBytes : 0U));
        pipe_->InitBuffer(lanesBuf_, lanes_ * laneStride_);
        if (!overlap_.enabled) for (uint32_t lane = 0; lane < lanes_; ++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(traceEnabled_) {
            // The overlap budget includes retained sender/count sources and the
            // optional second probe snapshot. Never borrow an active lane.
            const uint64_t used=overlap_.enabled ? overlap_.totalBytes :
                uint64_t(fixed)+uint64_t(lanes_)*laneStride_+(probePrefetch_?kLlReceiveProbeExtraBytes:0U);
            const uint32_t spare=used<kEpServerDedupCountUbBytes ? uint32_t(kEpServerDedupCountUbBytes-used):0U;
            traceCapacity_=spare/32U;
            if(traceCapacity_>LlReferenceProfile::kPacketTraceCapacity)traceCapacity_=LlReferenceProfile::kPacketTraceCapacity;
            if(traceCapacity_>tasks_.Count())traceCapacity_=tasks_.Count();
            if(traceCapacity_)pipe_->InitBuffer(packetTraceBuf_,traceCapacity_*32U);
        }
#endif
        if(profile_) profile_->Mark<32>();

    }

    // At most two adjacent 16-slot snapshots; every queued probe is joined
    // before returning, including when neither snapshot has a ready packet.
    __aicore__ inline bool TryBatch() {
        if (!probePrefetch_) return TrySerialBatch();

        ProbePipelineOps ops{this};
        return LlRunReceiveProbePair(ops,tasks_.PendingCount(),scan_);
    }
    // Exact baseline path when the optional second ID snapshot does not fit.
    __aicore__ inline bool TrySerialBatch() {
        if (!tasks_.PendingCount()) return false;

        auto snapshot=probeIdxBuf_.Get<uint32_t>();
        const uint32_t depth=lanes_;
            // One bounded scan; empty batches return to the cooperative caller.
            // Probe checks every block marker before exposing a packet.

            uint32_t count;
            ProbeResult probe;
            if (scan_>=tasks_.PendingCount()) scan_=0U;
            const uint32_t remaining=tasks_.PendingCount()-scan_;
            count=remaining<kProbeBatch ? remaining:kProbeBatch;
            probe=Probe(scan_,count);
            if (probe.readyMask==0U) {  scan_+=count; return false; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            RecordReady(snapshot,probe.readyMask,count);
#endif

            // Current senders mark only FIRST_HOP with KeepFlags; forwarded
            // terminals are Compact. Probe already validated every block flag.
            const auto order=LlMakeReceivePriority(count,probe.readyMask,
                                                   enableDedup_ ? probe.keepFlagsMask : 0U);
            uint32_t nextLoad=0U;
            for (uint32_t current=0U;current<order.count;++current) {
                // Load and consume in exactly the same priority order, so the
                // global issued_/consumed_ lane ring keeps its original ownership.
                while (issued_-consumed_<depth && nextLoad<order.count) {
                    const uint32_t index=order.At(nextLoad++);
                    IssueLoad(snapshot.GetValue(index),issued_%lanes_,
                              (probe.keepFlagsMask&(uint32_t(1)<<index))!=0U,
                              !overlap_.enabled || issued_>=lanes_);
                    ++issued_;
                }
                const uint32_t index=order.At(current);
                ConsumeLoaded(snapshot.GetValue(index),consumed_%lanes_,
                              (probe.keepFlagsMask&(uint32_t(1)<<index))!=0U);
                ++consumed_;
            }
            // Do NOT remove entries in priority order: swap-with-last would
            // invalidate the remaining original pending positions. After all
            // snapshot work is issued, remove by descending original index.
            {
            if (!tasks_.DropReadySuffix(scan_,count,probe.readyMask)) {
            for (uint32_t n=count;n>0U;--n) {
                const uint32_t index=n-1U;
                if (probe.readyMask&(uint32_t(1)<<index)) tasks_.RemovePending(scan_+index);
            }
            }
            }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        FinishBatch(lastReady_);
#endif
        return true;
    }
    __aicore__ inline void Run() {
        while (tasks_.PendingCount()) TryBatch();

        if (!tasks_.Count()) { FinishReference(); return; }

        // 5. pending 清空后，等待各 lane 在途写入，随后统一清除本核所有槽位的块标记。
        // 清理使用稳定任务表，不能使用已压缩为空的 pending 表；payload 本身不清零。

        for (uint32_t lane = 0; lane < (!overlap_.enabled || issued_>=lanes_ ? lanes_ : issued_); ++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
        // This is the end of wait instruction submission, not a new Scalar
        // completion barrier. The existing final PIPE_ALL establishes completion.

        ClearFlags();
        if(profile_) profile_->Mark<35>();

        // 最终流水屏障确保输出、relay 和标记清理完成。
        {

        PipeBarrier<PIPE_ALL>();
        if(profile_) profile_->Mark<36>();
        }
        FinishReference();
    }

#ifdef DEEPEP_LL_REFERENCE_PROFILE
    // Caller invokes this only after Counts.Finish's existing final PIPE_ALL.
    __aicore__ inline void ExportPacketTrace(LlReferenceProfile::DeviceProfile* profile) {
        if(!traceEnabled_ || !profile || !profile->Enabled()) return;
        profile->Detail(96U,traceCapacity_); profile->Detail(97U,traceCapacity_);
        profile->Detail(98U,tasks_.Count()); profile->Detail(99U,traceReady_); profile->Detail(100U,traceProcessed_);
        profile->Detail(101U,lanes_); profile->Detail(104U,tasks_.Begin());
        profile->Detail(105U,bs_); profile->Detail(106U,localExperts_);
        profile->Detail(107U,tasks_.Count()>traceCapacity_?1U:0U); profile->Detail(108U,batches_);
        profile->Detail(102U,profile->Clock());
        if(traceCapacity_) {
            GlobalTensor<uint8_t> output;
            output.SetGlobalBuffer(profile->PacketLogAddress());
            output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            ServerDedupReceiveSync<HardEvent::S_MTE3>();
            DataCopy(output,packetTraceBuf_.Get<uint8_t>(),traceCapacity_*32U);
            ServerDedupReceiveSync<HardEvent::MTE3_S>();
        }
        profile->Detail(103U,profile->Clock());
    }
#endif

private:
    __aicore__ inline void FinishReference() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(profile_) {
            profile_->Value<33>(pollTicks_); profile_->Value<34>(consumeTicks_);
            profile_->Value<41>(batches_); profile_->Mark<19>();
        }
#endif
    }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void StartPoll() {
        if(profile_ && !pollStarted_) { pollStart_=profile_->Clock(); pollStarted_=true; }
    }
    __aicore__ inline void ReadyPoll() {
        if(!profile_) return;
        lastReady_=profile_->Clock(); pollTicks_+=lastReady_-pollStart_; pollStarted_=false;
        if(!batches_) profile_->Value<42>(lastReady_);
        ++batches_;
    }
    __aicore__ inline void FinishBatch(uint64_t ready) {
        if(profile_) consumeTicks_+=profile_->Clock()-ready;
    }
    __aicore__ inline void RecordReady(LocalTensor<uint32_t> snapshot,uint32_t ready,uint32_t count) {
        if(!traceEnabled_) return;
        for(uint32_t i=0;i<count;++i) if(ready&(1U<<i)) {
            const uint32_t j=snapshot.GetValue(i); ++traceReady_;
            if(j<traceCapacity_) {
                auto row=packetTraceBuf_.Get<uint32_t>()[j*8U];
                row.SetValue(0U,traceReady_); row.SetValue(1U,batches_);
            }
        }
    }
    __aicore__ inline void RecordProcessed(uint32_t j,LocalTensor<uint8_t> wire,uint32_t headerOffset,bool keepFlags) {
        if(!traceEnabled_) return;
        ++traceProcessed_;
        if(j<traceCapacity_) {
            auto row=packetTraceBuf_.Get<uint32_t>()[j*8U];
            auto header=wire[headerOffset].template ReinterpretCast<uint32_t>();
            const uint32_t role=header.GetValue(3U);
            const uint32_t members=role==uint32_t(EpServerDedupTokenType::FirstHop) ?
                wire[uint32_t(keepFlags?data_.relayReadIndexWireOffset:data_.readIndexPayloadOffset)].template ReinterpretCast<uint32_t>().GetValue(0U):0U;
            row.SetValue(2U,traceProcessed_); row.SetValue(3U,header.GetValue(0U));
            row.SetValue(4U,header.GetValue(1U)); row.SetValue(5U,header.GetValue(2U));
            row.SetValue(6U,role|(members<<8U)); row.SetValue(7U,tasks_.Position(j));
        }
    }
    TBuf<> packetTraceBuf_;
    bool traceEnabled_=false,pollStarted_=false;
    uint32_t traceCapacity_=0,traceReady_=0,traceProcessed_=0,batches_=0;
    uint64_t pollStart_=0,lastReady_=0,pollTicks_=0,consumeTicks_=0;
#endif
    static constexpr uint32_t kProbeBatch = kLlReceiveProbeBatch;
    static_assert(kProbeBatch<=16U,"priority order uses four-bit snapshot positions");
    struct ProbeResult {
        uint32_t readyMask = 0U;
        uint32_t keepFlagsMask = 0U;
    };
    struct ReadyBatch {
        LlReceivePriority order;
        uint32_t snapshotOffset=0U, keepFlagsMask=0U, nextLoad=0U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint64_t profileBegin=0U;
#endif
    };
    struct ProbePipelineOps {
        ServerDedupReceiver* receiver;
        __aicore__ inline void StartProbe(uint32_t start,uint32_t count,uint32_t offset) {
            receiver->StartProbe(start,count,offset);
        }
        __aicore__ inline LlReceiveProbeResult FinishProbe(uint32_t count) {
            return receiver->FinishProbe(count);
        }
        __aicore__ inline ReadyBatch PrimeReady(uint32_t offset,uint32_t count,LlReceiveProbeResult probe) {
            return receiver->PrimeReady(offset,count,probe);
        }
        __aicore__ inline void ConsumeReady(ReadyBatch& batch) { receiver->ConsumeReady(batch); }
        __aicore__ inline void RemoveProbe(uint32_t start,uint32_t count,uint32_t ready) {
            receiver->RemoveProbe(start,count,ready);
        }
    };

    __aicore__ inline void FillReadyLanes(ReadyBatch& batch) {
        auto snapshot=probeIdxBuf_.Get<uint32_t>();
        while (issued_-consumed_<lanes_ && batch.nextLoad<batch.order.count) {
            const uint32_t index=batch.order.At(batch.nextLoad++);
            IssueLoad(snapshot.GetValue(batch.snapshotOffset+index),issued_%lanes_,
                      (batch.keepFlagsMask&(uint32_t(1)<<index))!=0U,
                      !overlap_.enabled || issued_>=lanes_);
            ++issued_;
        }
    }
    __aicore__ inline ReadyBatch PrimeReady(uint32_t offset,uint32_t count,LlReceiveProbeResult probe) {
        ReadyBatch batch;
        batch.order=LlMakeReceivePriority(count,probe.readyMask,enableDedup_ ? probe.keepFlagsMask:0U);
        batch.snapshotOffset=offset;
        batch.keepFlagsMask=probe.keepFlagsMask;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(probe.readyMask) {
            batch.profileBegin=lastReady_;
            RecordReady(probeIdxBuf_.Get<uint32_t>()[offset],probe.readyMask,count);
        }
#endif
        FillReadyLanes(batch);
        return batch;
    }
    __aicore__ inline void ConsumeReady(ReadyBatch& batch) {
        auto snapshot=probeIdxBuf_.Get<uint32_t>();
        for (uint32_t current=0U;current<batch.order.count;++current) {
            FillReadyLanes(batch);
            const uint32_t index=batch.order.At(current);
            ConsumeLoaded(snapshot.GetValue(batch.snapshotOffset+index),consumed_%lanes_,
                          (batch.keepFlagsMask&(uint32_t(1)<<index))!=0U);
            ++consumed_;
        }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(batch.order.count) FinishBatch(batch.profileBegin);
#endif
    }
    __aicore__ inline void RemoveProbe(uint32_t start,uint32_t count,uint32_t ready) {
        if (!ready) return;

        if (tasks_.DropReadySuffix(start,count,ready)) return;
        for (uint32_t n=count;n>0U;--n)
            if (ready&(uint32_t(1)<<(n-1U))) tasks_.RemovePending(start+n-1U);
    }

    // Only this path owns event 6 for MTE2_V and V_S. Immediate receive
    // dependencies keep event 7; payload events are different HardEvent domains.
    // A is decoded before reusing flags/sums/work for B. No in-flight probe
    // survives TryBatch, so cooperative sender/count callbacks cannot collide.
    __aicore__ inline void StartProbe(uint32_t start,uint32_t count,uint32_t offset) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        StartPoll();
#endif

        auto indices=probeIdxBuf_.Get<uint32_t>();
        auto flags=flagsBuf_.Get<float>();
        auto sums=sumBuf_.Get<float>();
        const uint32_t blocks=uint32_t(data_.blockCount);
        {
          for (uint32_t i=0U;i<count;++i) {
              const uint32_t j=tasks_.PendingIndex(start+i);
              indices.SetValue(offset+i,j);
              ReadServerDedupFlags(Slot(j),blocks,flags[i*blocks*8U]);
          }
        }
        {
          SetFlag<HardEvent::MTE2_V>(6U);
          WaitFlag<HardEvent::MTE2_V>(6U);
        }
        {
          if (blocks==8U)
              BuildReceiveFlagMasks(reinterpret_cast<__ubuf__ float*>(flags.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(sums.GetPhyAddr()),count,blocks);
          else
              Sum<float>(sums,flags,workBuf_.Get<uint8_t>(),SumParams{count,blocks*8U,blocks*8U});
          SetFlag<HardEvent::V_S>(6U);
        }
    }
    __aicore__ inline LlReceiveProbeResult FinishProbe(uint32_t count) {

        // This is the residual scalar join, not an interval spanning A's
        // payload work. REDUCE/POLL calls now count submit/join fragments.
        {  WaitFlag<HardEvent::V_S>(6U); }

        LlReceiveProbeResult result;
        if (data_.blockCount==8U) {
            auto masks=sumBuf_.Get<uint32_t>();
            result.readyMask=masks.GetValue(0U);
            result.keepFlagsMask=masks.GetValue(1U);
        } else {
            auto sums=sumBuf_.Get<float>();
            for (uint32_t i=0U;i<count;++i) {
                const auto layout=PacketReceiveLayout::Decode(sums.GetValue(i),uint32_t(data_.blockCount));
                if (layout!=PacketReceiveLayout::State::Pending) result.readyMask|=uint32_t(1)<<i;
                if (layout==PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask|=uint32_t(1)<<i;
            }
        }

#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(result.readyMask) ReadyPoll();
#endif
        return result;
    }
    // 摘要：将稳定任务编号 j 映射为本轮通信窗口中的实际 packet 槽地址。
    __aicore__ inline __gm__ uint8_t* Slot(uint32_t j)
    {
        return args_->peerMems[args_->rank] + data_.dispatchOffset + uint64_t(tasks_.Position(j)) * data_.tokenStride;
    }

    // 摘要：保存最多 16 项 pending 任务快照，检查每个包全部块的标记，返回就绪位图。
    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        StartPoll();
#endif

        auto indices = probeIdxBuf_.Get<uint32_t>();
        auto flags = flagsBuf_.Get<float>();
        auto sums = sumBuf_.Get<float>();
        auto work = workBuf_.Get<uint8_t>();
        const uint32_t blocks = uint32_t(data_.blockCount);

        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t j = tasks_.PendingIndex(start + i);
            indices.SetValue(i, j);
            ReadServerDedupFlags(Slot(j), blocks, flags[i * blocks * 8U]);
        }

        {  ServerDedupReceiveSync<HardEvent::MTE2_V>(); }
        const bool fusedMasks=blocks==8U;
        {
          if (fusedMasks) {
              BuildReceiveFlagMasks(reinterpret_cast<__ubuf__ float*>(flags.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(sums.GetPhyAddr()),count,blocks);
          } else {
              Sum<float>(sums, flags, work, SumParams{count, blocks * 8U, blocks * 8U});
          }
          ServerDedupReceiveSync<HardEvent::V_S>(); }

        ProbeResult result;
        if (fusedMasks) {
            auto masks=sums.template ReinterpretCast<uint32_t>();
            result.readyMask=masks.GetValue(0U);
            result.keepFlagsMask=masks.GetValue(1U);

        } else {
        for (uint32_t i = 0; i < count; ++i) {
            const auto layout = PacketReceiveLayout::Decode(sums.GetValue(i), blocks);
            if (layout != PacketReceiveLayout::State::Pending) { result.readyMask |= uint32_t(1) << i;  }
            if (layout == PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask |= uint32_t(1) << i;
        }
        }

#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(result.readyMask) ReadyPoll();
#endif
        return result;
    }

    // 摘要：等待 lane 的旧输出完成后，将 packet 从 GM 预取到 UB，分别发布给 Scalar 和 MTE3 的依赖事件。
    __aicore__ inline void IssueLoad(uint32_t j, uint32_t lane, bool keepFlags, bool reuse)
    {

        if (reuse) {

            WaitFlag<HardEvent::MTE3_MTE2>(lane); }

        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(Slot(j));
        if (keepFlags) {
            DataCopy(wire, source, uint32_t(data_.tokenStride));
        } else {
            // Compact every 480B payload block, skipping the 32B GM Flag.
            // GM source stride is bytes; UB destination stride is 32B blocks.
            DataCopyPad(wire, source, {uint16_t(data_.blockCount), 480U, 32U, 0U, 0U}, {false, 0U, 0U, 0});
        }
        SetFlag<HardEvent::MTE2_MTE3>(lane);
        // Publish immediately after this load, before queuing later loads.
        SetFlag<HardEvent::MTE2_S>(lane);

    }

    // 摘要：消费已预取包，写出 token/scale 和路由元数据；FIRST_HOP 还向其余专家转发，再归还 lane。
    __aicore__ inline void ConsumeLoaded(uint32_t j, uint32_t lane, bool keepFlags)
    {

        {

          WaitFlag<HardEvent::MTE2_S>(lane);
           }

        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        const uint32_t row = tasks_.Begin() + j;
        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        RecordProcessed(j,wire,headerOffset,keepFlags);
#endif
        if (!enableDedup_ && !keepFlags) {
            // Compact offsets are logical payload offsets. Retain this lane
            // until all payload, scale and metadata output reads complete.

            {  WaitFlag<HardEvent::MTE2_MTE3>(lane); }

            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});

            auto header = wire[headerOffset].template ReinterpretCast<uint32_t>();

            DataCopyPad(destinationIndex_[uint64_t(row) * 3U], header, {1U, 12U, 0U, 0U, 0U});

            SetFlag<HardEvent::MTE3_MTE2>(lane);

            return;
        }

        auto type = wire[uint32_t(data_.tokenStride)].template ReinterpretCast<uint32_t>();
        auto destination = wire[uint32_t(data_.tokenStride) + 32U].template ReinterpretCast<uint32_t>();
        ServerDedupPacketTail tail(wire, headerOffset,
                                   uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
        auto header = tail.Header();
        const uint32_t role = enableDedup_ ? header.GetValue(3U) : 0U;
        const bool firstHop = enableDedup_ && role == uint32_t(EpServerDedupTokenType::FirstHop);
        if (enableDedup_)
            BuildServerDedupMetadata(type, destination, role, tasks_.Position(j));
        // Forwarding changes only header[3]; header[0..2] remains immutable
        // until the existing MTE3_MTE2 lane retirement.
        if (!enableDedup_ || role != uint32_t(EpServerDedupTokenType::Terminal))
            destination = header;

        // FIRST_HOP preparation is shared work: retain only the core stage profile.
        if (firstHop) {
            PrepareServerDedupForward(tail);
            // Forwarded packets only need compact payload. The hint remains
            // independent of the application-level token type in the header.
            wire.template ReinterpretCast<float>().SetValue(PacketReceiveLayout::kMarkerOffset / 4U,
                                                            PacketReceiveLayout::kCompact);
        }

        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        {  WaitFlag<HardEvent::MTE2_MTE3>(lane); }

        // Initiate downstream work before local output writes on the same
        // MTE3 stream. Metadata was snapshotted before the terminal rewrite;
        // wire stays immutable until every relay/output read retires.
        if (firstHop) {
            const uint32_t sourceRank = tail.Header().GetValue(0U);
            const uint32_t ranks = uint32_t(args_->rankSize);
            const uint32_t members = tail.ReadIndex().GetValue(0U);

            for (uint32_t i = 0U; i < members; ++i) {
                const uint32_t index = tail.Entry(i).GetValue(0U);
                const uint32_t expertId = index & 0xffffU;
                const uint32_t q = index >> 16U;
                const uint32_t destinationRank = expertId / localExperts_;

                GlobalTensor<uint8_t> output;
                output.SetGlobalBuffer(args_->peerMems[destinationRank] +
                                       DispatchServerDedupDispatchOffset(data_, sourceRank, destinationRank,
                                                                         expertId % localExperts_, q, ranks,
                                                                         localExperts_, bs_));

                DataCopy(output, wire, uint32_t(data_.tokenStride));

            }

        }

        if (keepFlags) {
            WriteServerDedupPayloadRange(expandX_[uint64_t(row) * hiddenBytes_], wire, 0U, hiddenBytes_);
            if constexpr (WireFp8)
                WriteServerDedupPayloadRange(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire,
                                             uint32_t(data_.quantCount), uint32_t(data_.scaleBytes));
        } else {
            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});
        }

        if (enableDedup_) DataCopyPad(tokenType_[row], type, {1U, 4U, 0U, 0U, 0U});
        DataCopyPad(destinationIndex_[uint64_t(row) * 3U], destination, {1U, 12U, 0U, 0U, 0U});

        if (firstHop) {
            DataCopyPad(relayReadIndex_[uint64_t(row) * k_ * 2U], tail.ReadIndex(), {1U, k_ * 8U, 0U, 0U, 0U});
        }

        SetFlag<HardEvent::MTE3_MTE2>(lane);
    }

    // 摘要：仅清零本核拥有的所有接收槽中的 32B 块标记，保留 payload；完成等待由调用方负责。
    __aicore__ inline void ClearFlags()
    {

        auto zeros = flagsBuf_.Get<float>();
        Duplicate(zeros, 0.0f, uint32_t(data_.blockCount) * 8U);
        ServerDedupReceiveSync<HardEvent::V_MTE3>();
        for (uint32_t j = 0; j < tasks_.Count(); ++j) {
            GlobalTensor<float> output;
            output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Slot(j) + 480U));
            DataCopyPad(output, zeros, {uint16_t(data_.blockCount), 32U, 0U, 480U, 0U});
        }
    }

    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    LlReferenceProfile::DeviceProfile* profile_ = nullptr;
    TPipe* pipe_ = nullptr;
    LlOverlapLayout overlap_{};
    LocalTensor<int32_t> sharedPrefix_;
    bool enableDedup_ = false;
    bool probePrefetch_ = false;
    uint64_t windowOffset_ = 0;
    uint32_t recvCoreNum_ = 0, coreNum_ = 0;
    uint32_t bs_ = 0U, k_ = 0U, experts_ = 0U, localExperts_ = 0U, receiver_ = 0U;
    uint32_t scan_=0U, issued_=0U, consumed_=0U;
    uint32_t hiddenBytes_ = 0U, laneStride_ = 0U, lanes_ = 0U;
    EpServerDedupDataLayout data_{};
    ServerDedupReceiveTasks tasks_;
    TBuf<> prefixBuf_, flagsBuf_, sumBuf_, workBuf_, probeIdxBuf_, lanesBuf_;
    GlobalTensor<uint8_t> expandX_, dynamicScales_;
    GlobalTensor<uint32_t> tokenType_, destinationIndex_, relayReadIndex_;
};
}  // namespace DispatchDedup
#endif
