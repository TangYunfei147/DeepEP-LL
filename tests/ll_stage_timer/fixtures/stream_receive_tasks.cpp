#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/cb59_stream/receive_tasks.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_stream_receive.h"

struct Pool {
    AscendC::TPipe pipe;
    uint64_t used = 0;
    template<int N> void InitBuffer(AscendC::TBuf<N>& buffer, uint64_t bytes) {
        assert(bytes && bytes % 32U == 0U);
        used += bytes;
        pipe.InitBuffer(buffer, bytes);
    }
};

int main() {
    constexpr uint32_t ranks = 16, receivers = 28, rank = 7;
    std::mt19937 random(0x59ab);
    for (uint32_t experts : {16U, 32U, 128U, 384U, 1024U})
    for (uint32_t b : {1U, 8U, 32U})
    for (uint32_t k : {1U, 8U, 32U}) {
        if (k > experts) continue;
        const uint32_t localExperts = experts / ranks;
        for (uint32_t pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint32_t> counts(experts, 0U), route(experts);
            std::iota(route.begin(), route.end(), 0U);
            if (pattern != 0U) {
                for (uint32_t source = 0; source < ranks; ++source)
                for (uint32_t token = 0; token < b; ++token) {
                    std::shuffle(route.begin(), route.end(), random);
                    if (pattern == 2U) {
                        // Every source token includes the same hot expert.
                        auto hot = std::find(route.begin(), route.end(), rank * localExperts);
                        std::iter_swap(route.begin(), hot);
                    }
                    for (uint32_t j = 0; j < k; ++j) {
                        const uint32_t expert = route[j];
                        if (expert / localExperts == rank)
                            ++counts[(expert % localExperts) * ranks + source];
                    }
                }
            }
            std::vector<uint32_t> expectedSlots;
            for (uint32_t record = 0; record < experts; ++record) {
                const uint32_t region = ((record % ranks + rank) % ranks) * localExperts + record / ranks;
                for (uint32_t q = 0; q < counts[record]; ++q) expectedSlots.push_back(region * b + q);
            }
            std::vector<uint32_t> owners(expectedSlots.size(), 0U);
            uint32_t outputBegin = 0;
            for (uint32_t receiver = 0; receiver < receivers; ++receiver) {
                const auto slice = DispatchDedup::LlStreamRecordSlice(experts, receiver, receivers);
                AscendC::TPipe prefixPipe;
                AscendC::TBuf<> prefixBuffer;
                AscendC::LocalTensor<uint32_t> prefix;
                uint32_t localRows = 0;
                if (slice.Count()) {
                    prefixPipe.InitBuffer(prefixBuffer, DispatchDedup::LlStreamAlign(uint64_t(slice.Count()) * 4U, 32U));
                    prefix = prefixBuffer.Get<uint32_t>();
                    for (uint32_t i = 0; i < slice.Count(); ++i) {
                        localRows += counts[slice.begin + i];
                        prefix.SetValue(i, localRows);
                    }
                }
                assert(localRows <= DispatchDedup::LlStreamOwnedRowsUpperBound(experts, ranks, b, k, receivers));
                Pool pool;
                DispatchDedup::Cb59Stream::ServerDedupReceiveTasks tasks;
                tasks.BuildStream(prefix, rank, ranks, experts, b, slice.begin, slice.Count(), outputBegin, pool);
                assert(tasks.Begin() == outputBegin && tasks.Count() == localRows && tasks.PendingCount() == localRows);
                assert(pool.used == 2U * DispatchDedup::LlStreamAlign(uint64_t(localRows) * 4U, 32U));
                for (uint32_t j = 0; j < localRows; ++j) {
                    assert(tasks.Position(j) == expectedSlots[outputBegin + j]);
                    assert(tasks.PendingIndex(j) == j);
                    ++owners[outputBegin + j];
                }
                std::vector<uint32_t> consumed(localRows, 0);
                while (tasks.PendingCount()) {
                    const uint32_t position = random() % tasks.PendingCount();
                    const uint32_t j = tasks.PendingIndex(position);
                    ++consumed[j];
                    tasks.RemovePending(position);
                    // Clearing ownership continues to use the immutable row table.
                    assert(tasks.Position(j) == expectedSlots[outputBegin + j]);
                }
                for (auto n : consumed) assert(n == 1U);
                outputBegin += localRows;
            }
            assert(outputBegin == expectedSlots.size());
            for (auto n : owners) assert(n == 1U);
        }
    }
}
