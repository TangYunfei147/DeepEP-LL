// CPU model: compile the real tail emitter and one-packet scheduling helper.
// Stream completion times deliberately differ from scalar issue times. This is
// an ordering/bytes test, not a model of NPU latency or compiler code generation.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ll_prequant_vector_tail.h"
#include "csrc/kernels/dispatch/low_latency/common/ll_input_prefetch.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h"

using namespace DispatchDedup;
constexpr uint32_t kPoison = 0xdeadbeefU;
struct Metadata {
    uint32_t topK;
    uint32_t Expert(uint32_t route) { return (route * 37U + 29U) % 1024U; }
    uint32_t Q(uint32_t, uint32_t route) { return (route * 101U + 65530U) % 65536U; }
    uint32_t WeightBits(uint32_t local) {
        const uint32_t bits[] = {0U, 0x80000000U, 0x3f800000U, 0xbf800000U,
            0x7f800000U, 0xff800000U, 0x7fc12345U, 0x7fa12345U,
            0xffcabcdeU, 1U, 0x80000001U, 0x00800000U};
        return bits[local % (sizeof(bits) / sizeof(bits[0]))];
    }
};

// Decode the public tail ABI from the requested route set, independently of
// the emitter's walk/order. Unused fields and padding must retain their bytes.
std::map<uint32_t, uint32_t> Expected(Metadata& metadata, const EpServerDedupDataLayout& d,
    const LlSendPacketTask& task, uint32_t start, uint32_t rank)
{
    std::map<uint32_t, uint32_t> words;
    const uint32_t h = uint32_t(d.headerPayloadOffset), r = uint32_t(d.readIndexPayloadOffset);
    words[h] = rank; words[h + 4U] = task.token; words[h + 8U] = task.firstK;
    words[h + 12U] = task.mask ? 2U : 1U;
    std::vector<uint32_t> selected;
    if (task.mask) {
        for (uint32_t k = 0; k < metadata.topK; ++k)
            if (task.mask & (uint32_t(1) << k)) selected.push_back(k);
        assert(!selected.empty() && selected.front() == task.firstK);
    }
    words[r] = selected.empty() ? 0U : uint32_t(selected.size() - 1U);
    words[r + 4U] = metadata.WeightBits((task.token - start) * metadata.topK + task.firstK);
    for (size_t j = 1; j < selected.size(); ++j) {
        const uint32_t route = task.token * metadata.topK + selected[j];
        words[r + uint32_t(j) * 8U] = metadata.Expert(route) | (metadata.Q(0, route) << 16U);
        words[r + uint32_t(j) * 8U + 4U] =
            metadata.WeightBits((task.token - start) * metadata.topK + selected[j]);
    }
    return words;
}

void ApplyWord(std::vector<uint32_t>& dst, uint32_t byteOffset, uint32_t value)
{
    assert(byteOffset % 4U == 0U);
    const auto store = LlPlanTailWordStore(byteOffset);
    assert(store.blockByteOffset % 32U == 0U);
    assert(store.mask && (store.mask & (store.mask - 1U)) == 0U && store.mask < 256U);
    uint32_t changed = 0;
    for (uint32_t i = 0; i < 8U; ++i) {
        if (!(store.mask & (uint64_t(1) << i))) continue;
        const uint32_t offset = store.blockByteOffset + i * 4U;
        assert(offset == byteOffset && offset / 4U < dst.size());
        dst[offset / 4U] = value;
        ++changed;
    }
    assert(changed == 1U);
}
struct ImmediateWriter {
    std::vector<uint32_t>& dst;
    void Word(uint32_t offset, uint32_t value) { ApplyWord(dst, offset, value); }
};

void BytesAndBounds()
{
    uint64_t cases = 0, crossed = 0;
    const uint32_t sizes[] = {1, 31, 32, 127, 128, 129, 255, 256, 257, 384, 448,
        479, 480, 481, 512, 896, 960, 1024, 1920, 3584, 4096, 7168};
    for (uint32_t h : sizes) for (uint32_t k = 1; k <= 32; ++k) {
        const auto d = DispatchServerDedupDataLayout(64, h, k, 1024,
            EpServerDedupInputMode::PrequantizedFp8Packs);
        Metadata metadata{k};
        assert(d.headerPayloadOffset % 32 == 0 && d.readIndexPayloadOffset % 32 == 0);
        assert(DispatchDedupAlign(h, 32) <= d.quantCount);
        assert(d.quantCount + DispatchDedupAlign(d.scaleBytes, 32) <= d.headerPayloadOffset);
        if (d.readIndexPayloadOffset != d.headerPayloadOffset + 32) ++crossed;
        for (uint32_t first = 0; first < k; ++first) for (uint32_t kind = 0; kind < 3; ++kind) {
            uint32_t mask = 0;
            if (kind) for (uint32_t member = first; member < k; ++member)
                if (kind == 1 || member == first || member % 2) mask |= uint32_t(1) << member;
            for (uint32_t localToken = 0; localToken < 3; ++localToken) {
                LlSendPacketTask task{7U + localToken, first, 0U, 0U, mask};
                std::vector<uint32_t> dst(size_t(d.packedPayloadBytes / 4), kPoison);
                ImmediateWriter writer{dst};
                LlEmitPrequantVectorTail(writer, metadata, uint32_t(d.headerPayloadOffset),
                    uint32_t(d.readIndexPayloadOffset), 13, task.token, first, mask, 7, k);
                const auto expected = Expected(metadata, d, task, 7, 13);
                std::vector<uint32_t> oracle(dst.size(), kPoison);
                for (auto word : expected) oracle[word.first / 4] = word.second;
                for (uint32_t word = 0; word < dst.size(); ++word) {
                    assert(dst[word] == oracle[word]);
                }
                ++cases;
            }
        }
    }
    assert(crossed > 0);
    std::cout << "tail cases=" << cases << " crossed-480-layouts=" << crossed << '\n';
}

struct Event { uint64_t time, order; std::function<void()> action; };
struct Streams {
    Metadata metadata{8};
    EpServerDedupDataLayout data = DispatchServerDedupDataLayout(64, 3584, 8, 384,
        EpServerDedupInputMode::PrequantizedFp8Packs);
    std::vector<uint32_t> input[2];
    std::vector<std::vector<uint32_t>> out;
    std::vector<Event> events;
    uint64_t scalar = 0, mte2 = 0, vector = 0, mte3 = 0;
    uint64_t inputFree[2]{}, ready[2]{};
    std::vector<uint64_t> laneFree;
    int queued = -1, current = -1;
    uint32_t issued = 0, consumed = 0, sent = 0, failures = 0, reuse = 0;
    uint32_t generation[2]{};
    bool owned[2]{};
    bool vectorTail;
    uint64_t transferDelay;
    Streams(bool useVector, uint32_t lanes, uint64_t delay)
        : out(lanes), laneFree(lanes), vectorTail(useVector), transferDelay(delay) {
        for (auto& buf : input) buf.assign(size_t(data.packedPayloadBytes / 4), kPoison);
        for (auto& buf : out) buf.assign(size_t(data.packedPayloadBytes / 4), kPoison);
    }
    template<class F> void At(uint64_t time, F action) {
        events.push_back({time, events.size(), action});
    }
    void Issue(const LlSendPacketTask&) {
        const uint32_t slot = issued % 2, id = issued++;
        assert(!owned[slot] && queued == -1);
        if (id >= 2) ++reuse;
        owned[slot] = true; generation[slot] = id; queued = int(slot);
        ++scalar;
        // AllocTensor's V_MTE2 prevents DMA from overwriting a prior V reader.
        mte2 = std::max({mte2, scalar, inputFree[slot]}) + 19;
        ready[slot] = mte2;
        At(mte2, [this, slot, id]() { input[slot][0] = id; });
    }
    void Take() {
        assert(current == -1 && queued != -1);
        current = queued; queued = -1;
        // EnQue/DeQue MTE2_V blocks V only; it is not a scalar fence.
        vector = std::max(vector, ready[current]);
    }
    struct QueuedWriter {
        Streams& streams;
        uint32_t slot;
        void Word(uint32_t offset, uint32_t value) {
            ++streams.scalar;
            uint64_t when = streams.scalar;
            if (streams.vectorTail) {
                streams.vector = std::max(streams.vector, streams.scalar) + 1;
                when = streams.vector;
            }
            streams.At(when, [thisStreams = &streams, slot = slot, offset, value]() {
                ApplyWord(thisStreams->input[slot], offset, value);
            });
        }
    };
    void Consume(const LlSendPacketTask& task) {
        assert(current != -1 && owned[current]);
        const uint32_t slot = uint32_t(current), id = generation[slot];
        QueuedWriter writer{*this, slot};
        LlEmitPrequantVectorTail(writer, metadata, uint32_t(data.headerPayloadOffset),
            uint32_t(data.readIndexPayloadOffset), 13, task.token, task.firstK, task.mask, 7, 8);
        // Explicit PIPE_V and the unchanged output lane MTE3_V precede Copy.
        vector = std::max(vector, scalar) + 1;
        const uint32_t lane = consumed++ % uint32_t(out.size());
        vector = std::max(vector, laneFree[lane]) + 11;
        At(vector, [this, lane, slot]() { out[lane] = input[slot]; });
        inputFree[slot] = vector; // Original post-Copy FreeTensor V_MTE2.
        owned[slot] = false; current = -1;
        // Original V_MTE3 publication + MTE3_V lane retirement.
        mte3 = std::max(mte3, vector) + transferDelay;
        laneFree[lane] = mte3;
        const auto expected = Expected(metadata, data, task, 7, 13);
        At(mte3, [this, lane, id, expected]() {
            if (out[lane][0] != id) ++failures;
            for (auto pair : expected)
                if (out[lane][pair.first / 4] != pair.second) ++failures;
            ++sent;
        });
    }
    void Run() {
        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            return a.time != b.time ? a.time < b.time : a.order < b.order;
        });
        for (const auto& event : events) event.action();
    }
};

uint32_t ExerciseStreams(bool vectorTail, uint32_t count, uint32_t lanes, uint64_t delay)
{
    Streams streams(vectorTail, lanes, delay);
    LlOnePacketInput prefetch;
    prefetch.Reset();
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t first = i % 8;
        const uint32_t mask = i % 3 ? (0xffU & (0xffU << first)) : 0U;
        LlSendPacketTask task{7U + i % 17, first, 0U, i, mask};
        prefetch.Push(streams, task);
        task.token = 999; // Pending task must be an owned value copy.
    }
    prefetch.Flush(streams);
    prefetch.Flush(streams);
    assert(streams.current == -1 && streams.queued == -1);
    assert(!streams.owned[0] && !streams.owned[1]);
    assert(streams.reuse == (count > 2 ? count - 2 : 0));
    streams.Run();
    assert(streams.consumed == count && streams.sent == count);
    return streams.failures;
}

int main()
{
    for (uint32_t prequant = 0; prequant <= 1; ++prequant)
        for (uint32_t dedup = 0; dedup <= 1; ++dedup)
            for (uint32_t async = 0; async <= 1; ++async)
                for (uint32_t send : {1U, 10U, 28U, 30U, 55U})
                    for (uint32_t core = 0; core < 56; ++core)
                        assert(LlUsePrequantVectorTail(prequant, dedup, async, core, send) ==
                            (prequant && dedup && async && core < send));
    BytesAndBounds();
    for (uint32_t count : {0U, 1U, 2U, 3U, 7U, 8U, 9U, 57U, 256U})
        for (uint32_t lanes : {1U, 2U, 6U})
            for (uint64_t delay : {1U, 31U, 1000U})
                assert(ExerciseStreams(true, count, lanes, delay) == 0);
    // Negative control: deleting MTE2_S while retaining scalar tail stores
    // really corrupts old input readers under this adversarial schedule.
    assert(ExerciseStreams(false, 57, 6, 1000) > 0);
    std::cout << "stream cases=243 negative-control=detected\n";
}
