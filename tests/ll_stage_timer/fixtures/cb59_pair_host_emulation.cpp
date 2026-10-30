// Reuse the same bounded memory/API emulation and original-route packet type.
#define main cb59_stream_fixture_main_not_called
#include "cb59_stream_host_emulation.cpp"
#undef main
#include "csrc/kernels/dispatch/low_latency/cb59_pair/pair_handoff.h"
#include "csrc/kernels/dispatch/low_latency/cb59_pair/receive_tasks.h"

static std::atomic<uint64_t> emptyOwnersBorrowing{0}, splitRecords{0}, biasedPrefixReads{0}, skippedPrefixReads{0};

static void check_planner() {
    using namespace DispatchDedup;
    for(uint32_t base : {0U, 7U, 65536U}) for(uint32_t a = 0; a < 33; ++a) for(uint32_t b = 0; b < 33; ++b) {
        std::vector<uint32_t> original(a + b);
        std::iota(original.begin(), original.end(), base);
        const auto even = LlMakePairReceivePlan(0U, base, a, base + a, b);
        const auto odd = LlMakePairReceivePlan(1U, base + a, b, base, a);
        assert(even.valid && odd.valid);
        assert(even.begin == base && even.end == odd.begin && odd.end == base + a + b);
        assert(even.Count() == original.size() / 2U && odd.Count() == original.size() - even.Count());
        bool evenBorrows = false, oddBorrows = false;
        for(uint32_t row : original) {
            if(row < even.end && row >= base + a) evenBorrows = true;
            if(row >= odd.begin && row < base + a) oddBorrows = true;
        }
        assert(even.needsPartnerPrefix == evenBorrows && odd.needsPartnerPrefix == oddBorrows);
        assert(!(evenBorrows && oddBorrows));
    }
    assert(!LlMakePairReceivePlan(0, 0, 4, 5, 3).valid);
    assert(!LlMakePairReceivePlan(0, UINT32_MAX, 1, 0, 0).valid);
    assert(!LlMakePairReceivePlan(28, 0, 0, 0, 0).valid);
    assert(!LlMakePairReceivePlan(0, 0, 0, 0, 0, 27).valid);
    for(uint32_t experts : {16U, 32U, 384U, 1024U}) for(uint32_t b : {1U, 8U, 32U}) {
        uint64_t maximumPair = 0;
        for(uint32_t receiver = 0; receiver < 28; receiver += 2) {
            const auto a = LlStreamRecordSlice(experts, receiver), z = LlStreamRecordSlice(experts, receiver + 1);
            maximumPair = std::max(maximumPair, uint64_t(a.Count() + z.Count()));
        }
        const uint64_t capacity = 16U * uint64_t(b) * std::min(8U, experts / 16U);
        const uint64_t expected = (std::min(capacity, maximumPair * b * 8U) + 1U) / 2U;
        assert(LlPairOwnedRowsUpperBound(experts, 16U, b, 8U) == expected);
    }
}

static void run_pair_case(Arena& arena, uint32_t experts, uint32_t batch, uint32_t type,
                          uint32_t scenario, uint32_t epoch, bool prove_early = false) {
    using namespace DispatchDedup;
    constexpr uint32_t ranks = 16U, workers = 28U, topk = 8U;
    const uint32_t destination = scenario == 1U || scenario == 3U ? 0U : scenario == 2U ? 15U : epoch % ranks;
    const uint32_t localExperts = experts / ranks;
    std::mt19937 random(104729U * epoch + experts);
    std::vector<std::vector<Packet>> records(experts);
    for(uint32_t source = 0; source < ranks; ++source) for(uint32_t token = 0; token < batch; ++token) {
        std::vector<uint32_t> routes(experts);
        std::iota(routes.begin(), routes.end(), 0U);
        if(scenario == 0) std::shuffle(routes.begin(), routes.end(), random);
        if(scenario == 3U) for(uint32_t k = 0; k < topk; ++k) routes[k] = source == 0U && k == 0U ? 0U : experts - topk + k;
        for(uint32_t k = 0; k < topk; ++k) if(routes[k] / localExperts == destination) {
            auto& record = records[(routes[k] % localExperts) * ranks + source];
            record.push_back(Packet{source, token, k, routes[k], uint32_t(record.size())});
        }
    }
    std::vector<uint32_t> prefix(experts);
    std::vector<Packet> dense;
    std::unordered_map<uint32_t, Packet> inbox;
    for(uint32_t record = 0; record < experts; ++record) {
        for(const auto& packet : records[record]) {
            dense.push_back(packet);
            const uint32_t slot = (((packet.source + destination) % ranks) * localExperts + packet.expert % localExperts) * batch + packet.q;
            assert(inbox.emplace(slot, packet).second);
        }
        prefix[record] = uint32_t(dense.size());
    }
    const auto layout = Cb59Replay::DispatchServerDedupCountLayout(experts, 56U);
    auto* state = reinterpret_cast<uint32_t*>(arena.base + DispatchServerDedupWindowOffset(epoch % 2U) + layout.cntOffset);
    auto* sums = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(state) + layout.cumsumOffset);
    auto* ready = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(state) + layout.cumsumFlagOffset);
    for(uint32_t word = 0; word < workers * workers * 8U; ++word) { assert(sums[word] == 0U); assert(ready[word] == 0U); }
    for(uint32_t record = 0; record < experts; ++record) assert(state[record * 128U] == 0U);
    AscendC::MockPool hostAligned;
    AscendC::TBuf<> outputBuffer, expertOutputBuffer;
    hostAligned.InitBuffer(outputBuffer, experts * 4U);
    hostAligned.InitBuffer(expertOutputBuffer, uint32_t(LlStreamAlign(localExperts * 8U, 32U)));
    auto* output = outputBuffer.Get<uint32_t>().pointer;
    auto* expertOutput = expertOutputBuffer.Get<int64_t>().pointer;
    std::fill(output, output + experts, UINT32_MAX);
    std::fill(expertOutput, expertOutput + localExperts, int64_t(-1));
    auto args = std::make_unique<CommArgs>();
    args->rank = destination; args->rankSize = ranks; args->peerMems[destination] = arena.base;
    std::array<std::atomic<bool>, workers> finished{};
    std::atomic<bool> start{false};
    std::atomic<uint64_t> expectedPrefixReads{0};
    AscendC::pairPrefixLoads.store(0);
    std::vector<uint32_t> rowOwner(dense.size(), UINT32_MAX);
    std::vector<std::thread> threads;
    for(uint32_t reverse = workers; reverse > 0; --reverse) {
        const uint32_t worker = reverse - 1U;
        threads.emplace_back([&, worker] {
            AscendC::emulatedCore = workers + worker;
            while(!start.load()) std::this_thread::yield();
            AscendC::TPipe pipe;
            AscendC::MockPool pool;
            Cb59Stream::StreamCounts counts;
            counts.Init(args.get(), reinterpret_cast<uint8_t*>(expertOutput), reinterpret_cast<uint8_t*>(output),
                        batch, topk, experts, type, epoch % 2U, &pipe);
            counts.Prepare(pool);
            Cb59Pair::PairHandoff pair;
            pair.Init(args.get(), reinterpret_cast<uint8_t*>(output), experts, epoch % 2U, &pipe);
            pair.Prepare(pool, counts);
            const uint32_t low = worker & ~1U, high = worker | 1U;
            const uint32_t quotient = experts / workers, remainder = experts % workers;
            const uint32_t pairRecordBegin = low * quotient + std::min(low, remainder);
            const uint32_t pairRecordEnd = (high + 1U) * quotient + std::min(high + 1U, remainder);
            if(pairRecordBegin == pairRecordEnd) {
                assert(pair.Count() == 0U && pair.UbBytes() == 0U && !pair.NeedsPartnerPrefix());
                finished[worker].store(true); return;
            }
            const uint32_t pairBegin = pairRecordBegin ? prefix[pairRecordBegin - 1U] : 0U;
            const uint32_t pairTotal = prefix[pairRecordEnd - 1U] - pairBegin;
            const uint32_t desiredBegin = pairBegin + (worker % 2U ? pairTotal / 2U : 0U);
            const uint32_t desiredEnd = pairBegin + (worker % 2U ? pairTotal : pairTotal / 2U);
            assert(pair.Begin() == desiredBegin && pair.Count() == desiredEnd - desiredBegin);
            assert(pair.Count() <= LlPairOwnedRowsUpperBound(experts, ranks, batch, topk));
            assert(pair.UbBytes() == LlPairExtraBufferBytes(experts, worker));
            assert(pool.bytes == counts.UbBytes() + pair.UbBytes());
            const uint32_t partner = worker ^ 1U;
            const uint32_t partnerRecord = partner * quotient + std::min(partner, remainder);
            const uint32_t partnerRecords = quotient + uint32_t(partner < remainder);
            const uint32_t partnerBegin = partnerRecord ? prefix[partnerRecord - 1U] : 0U;
            const uint32_t partnerEnd = prefix[partnerRecord + partnerRecords - 1U];
            const bool needs = std::max(desiredBegin, partnerBegin) < std::min(desiredEnd, partnerEnd);
            assert(pair.NeedsPartnerPrefix() == needs);
            assert(pair.PartnerRecordBegin() == partnerRecord && pair.PartnerRecordCount() == partnerRecords);
            assert(pair.PartnerOutputBegin() == partnerBegin && pair.PartnerTotalRows() == partnerEnd - partnerBegin);
            if(needs) {
                ++expectedPrefixReads;
                if(counts.Count() == 0U) ++emptyOwnersBorrowing;
                if(pair.PrefixBias()) ++biasedPrefixReads;
                auto raw = pair.LocalPrefixAbsolute();
                assert(pair.PrefixBias() == partnerRecord % 8U);
                for(uint32_t i = 0; i < partnerRecords; ++i)
                    assert(raw.GetValue(pair.PrefixBias() + i) == prefix[partnerRecord + i]);
                // Leading aligned words can belong to a different producer;
                // their value is deliberately not required to be ready.
            } else ++skippedPrefixReads;
            if(worker % 2U && desiredBegin > pairBegin && desiredBegin < pairBegin + pairTotal &&
                std::find(prefix.begin(), prefix.end(), desiredBegin) == prefix.end()) ++splitRecords;
            Cb59Pair::ServerDedupReceiveTasks tasks;
            tasks.BuildPair(counts.LocalPrefix(), destination, ranks, experts, batch, counts.RecordBegin(),
                            counts.RecordCount(), counts.OutputBegin(), pair, pool);
            assert(tasks.Begin() == desiredBegin && tasks.Count() == desiredEnd - desiredBegin);
            assert(pool.buffers <= 11U); // eight count + one packed pair + two task tables
            std::vector<uint32_t> positions;
            for(uint32_t j = 0; j < tasks.Count(); ++j) {
                const uint32_t row = tasks.Begin() + j, slot = tasks.Position(j);
                positions.push_back(slot);
                assert(inbox.at(slot) == dense[row]);
                std::lock_guard<std::mutex> lock(AscendC::gmMutex);
                assert(rowOwner[row] == UINT32_MAX);
                rowOwner[row] = worker;
            }
            std::mt19937 removal(epoch * 37U + worker);
            std::vector<bool> consumed(tasks.Count(), false);
            while(tasks.PendingCount()) {
                const uint32_t p = removal() % tasks.PendingCount(), stable = tasks.PendingIndex(p);
                assert(!consumed[stable]); consumed[stable] = true; tasks.RemovePending(p);
                for(uint32_t j = 0; j < tasks.Count(); ++j) assert(tasks.Position(j) == positions[j]);
            }
            finished[worker].store(true);
        });
    }
    std::thread publisher([&] {
        while(!start.load()) std::this_thread::yield();
        std::vector<uint32_t> order(experts);
        std::iota(order.begin(), order.end(), 0U);
        if(!prove_early) std::shuffle(order.begin(), order.end(), random);
        const uint32_t firstPair = 2U * (experts / workers) + std::min(2U, experts % workers);
        for(uint32_t index = 0; index < experts; ++index) {
            if(prove_early && index == firstPair) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while((!finished[0].load() || !finished[1].load()) && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                assert(finished[0].load() && finished[1].load() && "first pair waited for later count owners");
            }
            const uint32_t record = order[index];
            {
                std::lock_guard<std::mutex> lock(AscendC::gmMutex);
                state[record * 128U + 1U] = uint32_t(records[record].size());
                state[record * 128U] = 0x3f800000U;
            }
            if(index % 11U == 0U) std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    });
    start.store(true);
    publisher.join(); for(auto& thread : threads) thread.join();
    assert(std::equal(prefix.begin(), prefix.end(), output));
    for(uint32_t expert = 0; expert < localExperts; ++expert) {
        const uint32_t end = prefix[(expert + 1U) * ranks - 1U];
        const uint32_t previous = expert ? prefix[expert * ranks - 1U] : 0U;
        assert(expertOutput[expert] == (type == 0U ? end : end - previous));
    }
    for(auto owner : rowOwner) assert(owner != UINT32_MAX);
    assert(AscendC::pairPrefixLoads.load() == expectedPrefixReads.load());
    for(uint32_t record = 0; record < experts; ++record) {
        assert(state[record * 128U] == 0U);
        assert(state[record * 128U + 1U] == records[record].size());
    }
    for(uint32_t word = 0; word < workers * workers * 8U; ++word) { assert(sums[word] == 0U); assert(ready[word] == 0U); }
}

int main() {
    check_planner();
    const uint32_t batches[] = {1, 2, 4, 8, 16, 32};
    for(uint32_t experts : {16U, 32U, 384U, 1024U}) {
        Arena arena;
        for(uint32_t type : {0U, 1U}) for(uint32_t round = 0; round < 6; ++round) {
            const uint32_t epoch = 6U * type + round + 1U;
            run_pair_case(arena, experts, batches[round], type, round % 3U, epoch);
            // Real stream Prepare on the same cleared bank between pair calls.
            if(round % 2U) run_case(arena, experts, batches[round], type, round % 3U, epoch);
        }
    }
    { Arena arena; run_pair_case(arena, 384U, 32U, 1U, 3U, 1U); }
    { Arena arena; run_pair_case(arena, 384U, 1U, 1U, 1U, 1U, true); }
    assert(emptyOwnersBorrowing.load() > 0 && splitRecords.load() > 0);
    assert(biasedPrefixReads.load() > 0 && skippedPrefixReads.load() > 0);
    std::cout << "real pair headers passed: 50 pair threaded cases plus 24 real-stream interleavings; zero-owner borrowing="
              << emptyOwnersBorrowing << ", q-splits=" << splitRecords << ", biased loads=" << biasedPrefixReads << "\n";
}
