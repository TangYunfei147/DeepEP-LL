#pragma once
#include <cstdint>
#include "ll_fullmesh_first_hop.h"
#include "ll_group_policy.h"
#include "ll_rdma_layout.h"

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PACKET_PLAN_INLINE __aicore__ inline
#else
#define LL_PACKET_PLAN_INLINE inline
#endif

struct LlDispatchPacket { bool present; uint32_t mask; };
// The canonical packet order is ascending original (token, firstK), after
// dedup and first-hop selection. q, weights and destination slots stay original.
template<class Ids>
LL_PACKET_PLAN_INLINE LlDispatchPacket LlPacketForRoute(Ids ids, uint32_t index, uint32_t topK,
    uint32_t localExperts, uint32_t rank, uint32_t group, bool dedup, bool fullMesh)
{
    const uint32_t first = index / topK * topK, k = index % topK;
    const uint32_t server = uint32_t(ids.GetValue(index)) / localExperts / group;
    if (!dedup || server == rank / group) return {true, 0U};
    uint32_t mask = 0U, firstK = topK;
    LlFullMeshFirstHop mesh;
    for (uint32_t other = 0; other < topK; ++other) {
        const uint32_t destination = uint32_t(ids.GetValue(first + other)) / localExperts;
        if (destination / group != server) continue;
        mask |= 1U << other;
        if (firstK == topK) firstK = other;
        if (fullMesh) mesh.Add(destination, other);
    }
    if ((mask & (mask - 1U)) == 0U) return {true, 0U};
    if (fullMesh) firstK = mesh.FirstK();
    return {k == firstK, mask};
}

constexpr uint32_t kLlRemotePacket = 0x80000000U;
constexpr uint32_t kLlRouteQMask = 0xffffU;
LL_PACKET_PLAN_INLINE bool LlOwnsRemotePacket(uint32_t taggedQ, uint32_t core) {
    return (taggedQ & kLlRemotePacket) && ((taggedQ >> 16U) & 7U) == core;
}

// Reuse the retired qSub scratch as a peer/count hash table. At most routes
// distinct peers occur, so routes slots suffice even for B1/EP256. q's high
// bits store ownership; its original low 16 bits remain the wire slot index.
template<class Ids, class Words>
LL_PACKET_PLAN_INLINE uint32_t LlBuildRemotePacketOwners(Ids ids, Words q, Words scratch,
    uint32_t routes, uint32_t topK, uint32_t localExperts, uint32_t rank, uint32_t group,
    bool dedup, bool fullMesh, uint32_t core)
{
    for (uint32_t slot = 0; slot < routes; ++slot) scratch.SetValue(slot, 0U);
    uint32_t owned = 0U;
    for (uint32_t index = 0; index < routes; ++index) {
        const uint32_t peer = uint32_t(ids.GetValue(index)) / localExperts;
        if (LlSameHost(rank, peer)) continue;
        const auto packet = LlPacketForRoute(ids, index, topK, localExperts, rank, group, dedup, fullMesh);
        if (!packet.present) continue;
        const uint32_t key = peer + 1U;
        uint32_t slot = (peer * 0x9e3779b9U) % routes;
        uint32_t entry = scratch.GetValue(slot);
        while (entry && (entry & 0xffffU) != key) {
            if (++slot == routes) slot = 0U;
            entry = scratch.GetValue(slot);
        }
        const uint32_t ordinal = entry >> 16U;
        const uint32_t owner = ordinal % LlRdma::kQpCount;
        scratch.SetValue(slot, key | ((ordinal + 1U) << 16U));
        q.SetValue(index, (q.GetValue(index) & kLlRouteQMask) | kLlRemotePacket | (owner << 16U));
        if (owner == core) ++owned;
    }
    return owned;
}
#undef LL_PACKET_PLAN_INLINE
}
