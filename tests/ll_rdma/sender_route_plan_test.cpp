#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_sender_route_plan.h"
#include "kernels/dispatch/low_latency/common/ll_send_order.h"

using namespace DispatchDedup;

struct Words {
    uint32_t* data;
    uint32_t size;
    uint32_t* reads = nullptr;
    uint32_t GetValue(uint32_t index) const {
        assert(index < size);
        if (reads) ++*reads;
        return data[index];
    }
    void SetValue(uint32_t index, uint32_t value) {
        assert(index < size);
        data[index] = value;
    }
};

static void CheckQ(std::vector<uint32_t> ids, uint32_t topK, uint32_t experts) {
    const uint32_t n = uint32_t(ids.size());
    const uint32_t half = (n + 7U) / 8U * 8U;
    constexpr uint32_t guard = 0xfeedcafeU;
    std::vector<uint32_t> first(half + 2U, guard), second(half + 2U, guard), q(half + 2U, guard);
    uint32_t routeReads = 0;
    LlBuildRouteQ(Words{ids.data(), n, &routeReads}, n, topK, experts,
                  Words{first.data() + 1, half}, Words{second.data() + 1, half},
                  Words{q.data() + 1, half}, half);
    assert(routeReads == n); // one forward route pass, including collisions
    std::vector<uint32_t> reference(experts);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t expected = i < topK ? 0U : reference[ids[i]];
        assert(q[i + 1U] == expected);
        // Cross-check the original flat-prefix definition on the smaller cases.
        if (n <= 1024U && i >= topK)
            assert(q[i + 1U] == uint32_t(std::count(ids.begin(), ids.begin() + i, ids[i])));
        ++reference[ids[i]];
    }
    for (const auto* buffer : {&first, &second, &q}) {
        assert(buffer->front() == guard && buffer->back() == guard);
    }
}

static void CheckGroups(const std::vector<uint32_t>& ids, uint32_t groups, uint32_t perGroup,
                        uint32_t ownGroup, uint32_t rankInGroup, uint32_t core) {
    const uint32_t count = uint32_t(ids.size());
    std::vector<uint32_t> tasks(count * 4U), heads(groups), links(count, UINT32_MAX);
    for (uint32_t i = 0; i < count; ++i) {
        tasks[i * 4U] = i / 8U;
        tasks[i * 4U + 1U] = i % 8U;
        tasks[i * 4U + 2U] = ids[i];
    }
    LlBuildStableTaskGroups(Words{tasks.data(), uint32_t(tasks.size())}, count, perGroup, groups,
                            Words{heads.data(), groups});
    std::vector<uint32_t> expected, actual;
    for (uint32_t step = 0; step < groups; ++step) {
        const uint32_t group = LlDirectGroupOrder(groups, ownGroup, rankInGroup, core, step);
        for (uint32_t i = 0; i < count; ++i) if (ids[i] / perGroup == group) expected.push_back(i);
        for (uint32_t task = heads[group]; task != UINT32_MAX; task = tasks[task * 4U + 3U]) {
            assert(task < count && actual.size() < count);
            assert(tasks[task * 4U] == task / 8U && tasks[task * 4U + 1U] == task % 8U);
            assert(tasks[task * 4U + 2U] == ids[task]);
            actual.push_back(task);
        }
    }
    assert(actual == expected);
    // Flat-route ownership can split a token, and can belong to a core with no
    // token-weight slice. Check every partition without changing original q IDs.
    for (uint32_t partition = 0; partition < 7U; ++partition) {
        const uint32_t begin = uint32_t(uint64_t(partition) * count / 7U);
        const uint32_t end = uint32_t(uint64_t(partition + 1U) * count / 7U);
        LlBuildStableRouteGroups(Words{const_cast<uint32_t*>(ids.data()), count}, begin, end - begin,
                                 perGroup, groups, Words{heads.data(), groups}, Words{links.data(), count});
        expected.clear(); actual.clear();
        for (uint32_t step = 0; step < groups; ++step) {
            const uint32_t group = EpServerOrder::ServerDedupDispatchDirectOrder(groups, ownGroup, step);
            for (uint32_t i = begin; i < end; ++i) if (ids[i] / perGroup == group) expected.push_back(i);
            for (uint32_t route = heads[group]; route != UINT32_MAX; route = links[route]) {
                assert(route >= begin && route < end && actual.size() < end - begin);
                actual.push_back(route);
            }
        }
        assert(actual == expected);
    }
}

int main() {
    CheckQ({7, 7, 7, 7, 7, 7, 7, 7}, 4, 384); // first-token semantics retained
    CheckQ({1023}, 1, 1024);
    CheckQ(std::vector<uint32_t>(16352, 1023U), 32, 1024); // packed-count boundary
    std::mt19937 rng(261026);
    unsigned cases = 0;
    for (uint32_t experts : {16U, 32U, 64U, 128U, 384U, 512U, 1024U}) {
        for (uint32_t topK : {1U, 2U, 8U, 16U, 32U}) {
            for (uint32_t tokens : {1U, 2U, 8U, 17U, 32U, 64U, 255U, 256U}) {
                std::vector<uint32_t> ids(tokens * topK);
                for (auto& id : ids) id = rng() % experts;
                CheckQ(ids, topK, experts);
                std::fill(ids.begin(), ids.end(), experts - 1U);
                CheckQ(ids, topK, experts);
                ++cases;
            }
        }
    }
    // Deliberately collide the initial hash positions at <=50% occupancy.
    std::vector<uint32_t> colliding;
    for (uint32_t expert = 0; expert < 1024 && colliding.size() < 8U; ++expert) {
        uint32_t hash = expert * 0x9e3779b9U;
        hash ^= hash >> 16U;
        if (hash % 32U == 31U) colliding.push_back(expert);
    }
    assert(colliding.size() == 8U);
    auto repeated = colliding;
    repeated.insert(repeated.end(), colliding.begin(), colliding.end());
    CheckQ(repeated, 8, 1024);
    for (uint32_t groups : {1U, 2U, 4U, 8U, 16U, 32U}) {
        for (uint32_t count : {0U, 1U, 8U, 31U, 256U, 1024U}) {
            std::vector<uint32_t> ids(count);
            for (auto& id : ids) id = rng() % (groups * 16U);
            for (uint32_t own = 0; own < groups; ++own)
                CheckGroups(ids, groups, 16, own, rng() % 16U, rng() % 36U);
        }
    }
    std::cout << "PASS sender route plan: " << cases
              << " shape pairs, q prefix/first-token/UB bounds, stable task/route groups\n";
}
