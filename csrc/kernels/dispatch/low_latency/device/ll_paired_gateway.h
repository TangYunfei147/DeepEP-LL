#pragma once
#include "ll_route_exchange.h"
#include "../../../common/ep_packet_receive_layout.h"
#include "dispatch_server_dedup_receive_packet.h"
#include "../common/ll_dispatch_packet_plan.h"

namespace DispatchDedup {
// Protocol15 probe: an isolated phase uses separate GM inbox slots and
// count-stage UB. No additional live receiver lane/event survives this phase.
__aicore__ inline void ForwardPairedGateway(const __gm__ CommArgs* args,
    EpServerDedupDataLayout data, uint64_t bank, uint32_t bs, uint32_t k,
    uint32_t localExperts, uint32_t core, uint32_t cores, bool dedup,
    PairedRouteExchange* exchange, AscendC::TPipe* pipe)
{
    using namespace AscendC;
    pipe->Reset();
    TBuf<> routesBuf, wireBuf, flagsBuf;
    const uint32_t routes = bs * k;
    const uint32_t blocks = uint32_t(data.blockCount);
    pipe->InitBuffer(routesBuf, uint32_t(DispatchDedupAlign(uint64_t(routes) * 4U, 32U)));
    pipe->InitBuffer(wireBuf, uint32_t(data.tokenStride));
    pipe->InitBuffer(flagsBuf, blocks * 32U);
    auto ids = routesBuf.Get<int32_t>();
    auto wire = wireBuf.Get<uint8_t>();
    auto flags = flagsBuf.Get<uint32_t>();
    exchange->ReadPeerAsync(ids);
    ServerDedupReceiveSync<HardEvent::MTE2_S>();
    const uint32_t source = uint32_t(args->rank) ^ 16U;
    for (uint32_t index = core; index < routes; index += cores) {
        const uint32_t target = uint32_t(ids.GetValue(index)) / localExperts;
        if (LlSameHost(source, target)) continue;
        const auto packet = LlPacketForRoute(ids, index, k, localExperts, source, 16U, dedup, true);
        if (!packet.present) continue;
        auto slot = args->peerMems[args->rank] + bank + data.gatewayOffset + uint64_t(index) * data.tokenStride;
        // Every 32B flag, including the SO tail, must be fully ready. The
        // gateway never assumes SO for its subsequent local MTE writes.
        bool ready;
        do {
            ServerDedupReceiveSync<HardEvent::S_MTE2>();
            ReadServerDedupFlags(slot, blocks, flags.ReinterpretCast<float>(), true);
            ServerDedupReceiveSync<HardEvent::MTE2_S>();
            ready = true;
            for (uint32_t word = 0; word < blocks * 8U; ++word) {
                const uint32_t expected = word == 0U && packet.mask ? 0x3fc00000U : 0x3f800000U;
                if (flags.GetValue(word) != expected) { ready = false; break; }
            }
        } while (!ready);
        ServerDedupReceiveSync<HardEvent::S_MTE2>();
        DataCopy(wire, LlTransport::Tensor(slot, true), uint32_t(data.tokenStride));
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        ServerDedupPacketTail tail(wire, uint32_t(data.headerWireOffset), uint32_t(data.relayReadIndexWireOffset));
        const uint32_t expert = tail.Header().GetValue(4U);
        const uint32_t q = tail.Header().GetValue(5U);
        tail.Header().SetValue(6U, 1U); // retain FirstHop metadata, suppress second fanout
        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        LlTransport::Packet(args, target,
            DispatchServerDedupDispatchOffset(data, source, target, expert % localExperts, q,
                                               uint32_t(args->rankSize), localExperts, bs),
            wire, uint32_t(data.tokenStride));
        // A local consumer may start now; retire this UB read before changing
        // the header into Terminal for member outputs.
        ServerDedupReceiveSync<HardEvent::MTE3_S>();
        if (packet.mask) {
            PrepareServerDedupForward(tail);
            wire.ReinterpretCast<float>().SetValue(PacketReceiveLayout::kMarkerOffset / 4U,
                                                   PacketReceiveLayout::kCompact);
            ServerDedupReceiveSync<HardEvent::S_MTE3>();
            const uint32_t members = tail.ReadIndex().GetValue(0U);
            for (uint32_t member = 0; member < members; ++member) {
                const uint32_t packed = tail.Entry(member).GetValue(0U);
                const uint32_t memberExpert = packed & 0xffffU;
                const uint32_t memberRank = memberExpert / localExperts;
                LlTransport::Packet(args, memberRank,
                    DispatchServerDedupDispatchOffset(data, source, memberRank, memberExpert % localExperts,
                                                       packed >> 16U, uint32_t(args->rankSize), localExperts, bs),
                    wire, uint32_t(data.tokenStride));
            }
            ServerDedupReceiveSync<HardEvent::MTE3_S>();
        }
        // The unique owner clears its inbox after all fanout reads retire.
        for (uint32_t word = 0; word < blocks * 8U; ++word) flags.SetValue(word, 0U);
        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        DataCopyPad(LlTransport::Tensor(slot + 480U, true), flags.ReinterpretCast<uint8_t>(),
                    {uint16_t(blocks), 32U, 0U, 480U, 0U});
        ServerDedupReceiveSync<HardEvent::MTE3_S>();
    }
    PipeBarrier<PIPE_ALL>(); // retire inbox cleanup before phase UB reset
}
}
