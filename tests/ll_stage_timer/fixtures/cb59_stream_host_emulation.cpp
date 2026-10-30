// Executes the REAL StreamCounts::Prepare and ReceiveTasks::BuildStream.
// The mock provides memory operations only, never replacement routing logic.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_map>
#include "kernel_operator.h"
#include "csrc/kernels/dispatch/low_latency/cb59_stream/stream_counts.h"
#include "csrc/kernels/dispatch/low_latency/cb59_stream/receive_tasks.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

struct Arena {
    uint8_t* base = nullptr;
    static constexpr uint64_t bytes = 8ULL << 30U;
    static constexpr uint64_t commit = 2ULL << 20U;
    Arena() {
#ifdef _WIN32
        base = static_cast<uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS));
        assert(base);
        assert(VirtualAlloc(base, commit, MEM_COMMIT, PAGE_READWRITE));
        assert(VirtualAlloc(base + (4ULL << 30U), commit, MEM_COMMIT, PAGE_READWRITE));
#else
        void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(memory != MAP_FAILED);
        base = static_cast<uint8_t*>(memory);
#endif
    }
    ~Arena() {
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, bytes);
#endif
    }
};

struct Packet {
    uint32_t source, token, k, expert, q;
    bool operator==(const Packet& other) const {
        return std::tie(source, token, k, expert, q) == std::tie(other.source, other.token, other.k, other.expert, other.q);
    }
};

static void run_case(Arena& arena, uint32_t experts, uint32_t batch, uint32_t type,
                     uint32_t scenario, uint32_t epoch, bool prove_early = false) {
    using namespace DispatchDedup;
    constexpr uint32_t ranks = 16, workers = 28, topk = 8;
    const uint32_t destination = scenario == 1 ? 0U : scenario == 2 ? 15U : epoch % ranks;
    const uint32_t localExperts = experts / ranks;
    std::mt19937 random(7919U * epoch + experts);
    std::vector<std::vector<Packet>> records(experts);
    for(uint32_t source = 0; source < ranks; ++source) for(uint32_t token = 0; token < batch; ++token) {
        std::vector<uint32_t> routes(experts);
        std::iota(routes.begin(), routes.end(), 0U);
        if(scenario == 0) std::shuffle(routes.begin(), routes.end(), random);
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
            const uint32_t position = (((packet.source + destination) % ranks) * localExperts + packet.expert % localExperts) * batch + packet.q;
            assert(inbox.emplace(position, packet).second);
        }
        prefix[record] = uint32_t(dense.size());
    }
    const auto layout = Cb59Replay::DispatchServerDedupCountLayout(experts, 56U);
    auto* state = reinterpret_cast<uint32_t*>(arena.base + DispatchServerDedupWindowOffset(epoch % 2U) + layout.cntOffset);
    auto* sums = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(state) + layout.cumsumOffset);
    auto* ready = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(state) + layout.cumsumFlagOffset);
    for(uint32_t word = 0; word < workers * workers * 8U; ++word) { assert(sums[word] == 0U); assert(ready[word] == 0U); }
    for(uint32_t record = 0; record < experts; ++record) assert(state[record * 128U] == 0U);
    std::vector<uint32_t> output(experts, UINT32_MAX);
    std::vector<int64_t> expertOutput(localExperts, -1);
    auto args = std::make_unique<CommArgs>();
    args->rank = destination;
    args->rankSize = ranks;
    args->peerMems[destination] = arena.base;
    std::array<std::atomic<bool>, workers> finished{};
    std::atomic<bool> start{false};
    std::vector<uint32_t> rowOwner(dense.size(), UINT32_MAX);
    std::vector<std::thread> threads;
    for(uint32_t reverse = workers; reverse > 0; --reverse) {
        const uint32_t worker = reverse - 1U;
        threads.emplace_back([&, worker] {
            AscendC::emulatedCore = workers + worker;
            while(!start.load()) std::this_thread::yield();
            AscendC::TPipe pipe;
            AscendC::MockPool pool;
            Cb59Stream::StreamCounts actual;
            actual.Init(args.get(), reinterpret_cast<uint8_t*>(expertOutput.data()),
                reinterpret_cast<uint8_t*>(output.data()), batch, topk, experts, type, epoch % 2U, &pipe);
            actual.Prepare(pool);
            // Independent balanced partition and route-derived expected prefix.
            const uint32_t quotient = experts / workers, remainder = experts % workers;
            const uint32_t begin = worker * quotient + std::min(worker, remainder);
            const uint32_t count = quotient + uint32_t(worker < remainder);
            assert(actual.RecordBegin() == begin && actual.RecordCount() == count);
            const uint32_t offset = count && begin ? prefix[begin - 1U] : 0U;
            const uint32_t rows = count ? prefix[begin + count - 1U] - offset : 0U;
            assert(actual.Count() == rows);
            if(count) assert(actual.OutputBegin() == offset);
            auto localPrefix = actual.LocalPrefix();
            for(uint32_t i = 0; i < count; ++i) assert(localPrefix.GetValue(i) == prefix[begin + i] - offset);
            assert(actual.UbBytes() == pool.bytes);
            assert(actual.UbBytes() == LlStreamCountBufferBytes(experts, ranks, worker));
            Cb59Stream::ServerDedupReceiveTasks tasks;
            tasks.BuildStream(localPrefix, destination, ranks, experts, batch, begin, count, actual.OutputBegin(), pool);
            assert(tasks.Count() == rows);
            std::vector<uint32_t> positions;
            for(uint32_t j = 0; j < rows; ++j) {
                const auto position = tasks.Position(j);
                positions.push_back(position);
                const uint32_t row = tasks.Begin() + j;
                assert(row < dense.size());
                assert(inbox.at(position) == dense[row]);
                // Equal packet identity proves the original source/token/k
                // combine triple still accompanies the correct dense row.
                std::lock_guard<std::mutex> lock(AscendC::gmMutex);
                assert(rowOwner[row] == UINT32_MAX);
                rowOwner[row] = worker;
            }
            std::mt19937 removal(13U * epoch + worker);
            std::vector<bool> consumed(rows, false);
            while(tasks.PendingCount()) {
                const uint32_t p = removal() % tasks.PendingCount();
                const uint32_t stable = tasks.PendingIndex(p);
                assert(stable < rows && !consumed[stable]);
                consumed[stable] = true;
                tasks.RemovePending(p);
                for(uint32_t j = 0; j < rows; ++j) assert(tasks.Position(j) == positions[j]);
            }
            finished[worker].store(true);
        });
    }
    std::thread publisher([&] {
        while(!start.load()) std::this_thread::yield();
        std::vector<uint32_t> order(experts);
        std::iota(order.begin(), order.end(), 0U);
        if(!prove_early) std::shuffle(order.begin(), order.end(), random);
        const uint32_t firstSlice = experts / workers + uint32_t(experts % workers != 0U);
        for(uint32_t index = 0; index < experts; ++index) {
            if(prove_early && index == firstSlice) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while(!finished[0].load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                assert(finished[0].load() && "R0 waited for a successor's counts");
            }
            const uint32_t record = order[index];
            {
                std::lock_guard<std::mutex> lock(AscendC::gmMutex);
                state[record * 128U + 1U] = uint32_t(records[record].size());
                state[record * 128U] = 0x3f800000U;
            }
            if(index % 13U == 0U) std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    });
    start.store(true);
    publisher.join();
    for(auto& thread : threads) thread.join();
    assert(output == prefix);
    for(uint32_t expert = 0; expert < localExperts; ++expert) {
        const uint32_t end = prefix[(expert + 1U) * ranks - 1U];
        const uint32_t previous = expert ? prefix[expert * ranks - 1U] : 0U;
        assert(expertOutput[expert] == (type == 0U ? end : end - previous));
    }
    for(auto owner : rowOwner) assert(owner != UINT32_MAX);
    for(uint32_t record = 0; record < experts; ++record) {
        assert(state[record * 128U] == 0U);
        assert(state[record * 128U + 1U] == records[record].size());
    }
    for(uint32_t word = 0; word < workers * workers * 8U; ++word) { assert(sums[word] == 0U); assert(ready[word] == 0U); }
}

int main() {
    const uint32_t batches[] = {1, 2, 4, 8, 16, 32};
    for(uint32_t experts : {16U, 32U, 384U, 1024U}) {
        Arena arena;
        for(uint32_t type : {0U, 1U}) for(uint32_t round = 0; round < 6; ++round)
            run_case(arena, experts, batches[round], type, round % 3U, 6U * type + round + 1U);
    }
    { Arena arena; run_case(arena, 384U, 1U, 1U, 1U, 1U, true); }
    std::cout << "real StreamCounts/BuildStream: 49 threaded cases, both banks, count types, zero/hot routes, exact ownership and early R0 passed\n";
}
