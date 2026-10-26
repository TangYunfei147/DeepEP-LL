#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_sender_input_reuse.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h"

using namespace DispatchDedup;
using Payload = std::array<uint32_t, 40>;

// Delayed V reads and MTE writes exercise the production cache's retirement
// callbacks. Network submission may destroy the output lane, never the cached
// input/quantBuf. Quantized data is opaque here; production quant math is unchanged.
struct Operations {
    struct Job { Payload* source; uint32_t lane; Payload expected; bool vectorDone=false, outputDone=false; };
    std::array<Payload, 2> input{}, lanes{};
    Payload quant{};
    Payload* payload = nullptr;
    std::vector<Job> jobs;
    bool quantizing, held = false;
    uint32_t nextInput = 0, loads = 0, quantizations = 0, cacheWaits = 0, mode;

    explicit Operations(uint32_t selected) : quantizing(selected == 2U || selected == 3U), mode(selected) {}
    uint32_t Body(uint32_t token, uint32_t word) const {
        return (token * 0x9e3779b9U + word * 0x10001U) ^ (mode * 0x7fc01234U);
    }
    void CompleteVector(uint32_t end) {
        for (uint32_t i=0; i<end; ++i) if (!jobs[i].vectorDone) {
            // A later Scalar tail overwrite or input reload is observable here.
            assert(*jobs[i].source == jobs[i].expected);
            lanes[jobs[i].lane] = *jobs[i].source;
            jobs[i].vectorDone = true;
        }
    }
    void CompleteOutput(uint32_t index) {
        auto& job = jobs[index];
        if (job.outputDone) return;
        CompleteVector(index + 1U);
        assert(lanes[job.lane] == job.expected);
        job.outputDone = true;
    }
    void WaitPackedRead() {
        ++cacheWaits;
        CompleteVector(uint32_t(jobs.size()));
    }
    void ReleaseInput() {
        if (!quantizing) { assert(held); held = false; }
    }
    void LoadInput(uint32_t token) {
        ++loads;
        if (quantizing) {
            // V quantization is ordered after preceding V packet copies.
            CompleteVector(uint32_t(jobs.size()));
            payload = &quant;
            ++quantizations;
        } else {
            assert(!held);
            payload = &input[nextInput++ % 2U];
            // Queue reuse waits for old V readers of the selected input slot.
            for (uint32_t i=0; i<jobs.size(); ++i)
                if (jobs[i].source == payload && !jobs[i].vectorDone) CompleteVector(i + 1U);
            held = true;
        }
        for (uint32_t word=0; word<32U; ++word) (*payload)[word] = Body(token, word);
        for (uint32_t word=32U; word<40U; ++word) (*payload)[word] = 0U;
    }
    void Emit(uint32_t token, uint32_t ordinal, bool remote) {
        // Scalar tail mutation is legal only after every older V reader retires.
        for (const auto& job : jobs) assert(job.vectorDone || job.source != payload);
        const uint32_t lane = ordinal % 2U;
        for (uint32_t i=0; i<jobs.size(); ++i) if (jobs[i].lane == lane) CompleteOutput(i);
        Payload expected{};
        for (uint32_t word=0; word<32U; ++word) expected[word] = Body(token, word);
        for (uint32_t word=32U; word<40U; ++word)
            (*payload)[word] = expected[word] = (ordinal << 8U) | word;
        jobs.push_back({payload, lane, expected});
        if (remote) {
            // MTE3 stages the complete packet before ordinary SDK Put borrows
            // this retired output lane. The registered source is a separate copy.
            const uint32_t index = uint32_t(jobs.size()) - 1U;
            CompleteVector(index + 1U);
            const Payload registeredSource = lanes[lane];
            std::fill(lanes[lane].begin(), lanes[lane].begin()+32, 0xdeadbeefU);
            assert(registeredSource == expected);
            jobs[index].outputDone = true;
        }
    }
    void Drain() {
        for (uint32_t i=0; i<jobs.size(); ++i) CompleteOutput(i);
        assert(!held);
    }
};

static void Check(const std::vector<uint32_t>& tokens, uint32_t mode, uint32_t remotePattern) {
    LlSenderInputReuse cache;
    Operations operations(mode);
    uint32_t expectedLoads=0;
    for (uint32_t i=0; i<tokens.size(); ++i) {
        if (!i || tokens[i] != tokens[i-1U]) ++expectedLoads;
        cache.Prepare(tokens[i], operations);
        operations.Emit(tokens[i], i, remotePattern==2U || (remotePattern==1U && i%3U==0U));
    }
    cache.Finish(operations);
    cache.Finish(operations); // finishing an empty cache never frees twice
    operations.Drain();
    assert(operations.loads == expectedLoads);
    assert(operations.cacheWaits == tokens.size() - expectedLoads);
    assert(operations.quantizations == (operations.quantizing ? expectedLoads : 0U));
}

static void CheckPadding(uint32_t hidden, uint32_t topK) {
    const auto layout = DispatchServerDedupDataLayout(8U, hidden, topK, 384U,
        EpServerDedupInputMode::PrequantizedFp8Packs);
    const auto plan = LlPlanPrequantPadding(hidden, uint32_t(layout.quantCount),
        uint32_t(layout.scaleBytes), uint32_t(layout.packedPayloadBytes));
    const uint32_t bytes = uint32_t(layout.packedPayloadBytes);
    std::vector<uint8_t> actual(bytes, 0xa5U), expected(bytes, 0U);
    uint32_t zeroed=0;
    auto clear = [&](uint32_t offset, uint32_t count) {
        assert(offset%4U==0U && offset+count<=bytes);
        std::fill(actual.begin()+offset, actual.begin()+offset+count, 0U);
        zeroed += count;
    };
    assert(plan.inputOffset%32U==0U && plan.tailBlockOffset%32U==0U && plan.tailOffset%32U==0U);
    clear(plan.inputOffset, plan.inputBytes);
    for (uint32_t word=0; word<8U; ++word)
        if (plan.tailBlockMask & (uint64_t(1)<<word)) clear(plan.tailBlockOffset+word*4U, 4U);
    clear(plan.tailOffset, plan.tailBytes);
    assert(zeroed == bytes-hidden-layout.scaleBytes + hidden%32U);
    // Only the last <=31 valid input bytes may share a 32B-aligned padding
    // write. The original MTE2 input copy overwrites those bytes below.
    for (uint32_t i=0; i<hidden; ++i) actual[i]=expected[i]=uint8_t(i*17U+3U);
    for (uint32_t i=0; i<layout.scaleBytes; ++i)
        actual[layout.quantCount+i]=expected[layout.quantCount+i]=uint8_t(i*29U+13U);
    assert(actual==expected); // identical to the old whole-buffer zero plus two loads
    if (hidden==3584U && topK==8U) assert(zeroed==144U && bytes==3840U);
}

struct TailWords {
    uint32_t* data;
    uint32_t size;
    void SetValue(uint32_t i,uint32_t value) { assert(i<size); data[i]=value; }
};

static void CheckPrequantTail(uint32_t topK) {
    std::vector<uint32_t> reused(topK*2U,0U);
    uint32_t previous=0U, packet=0U;
    for (uint32_t entries : {topK-1U, topK/2U, 0U, topK-1U, 0U, 0U}) {
        std::vector<uint32_t> reference(topK*2U,0U); // baseline resets every packet
        reference[0]=reused[0]=entries;
        reference[1]=reused[1]=0x3f800000U+packet; // opaque self-weight bits
        for (uint32_t entry=1; entry<=entries; ++entry) {
            reference[entry*2U]=reused[entry*2U]=(packet<<16U)|entry;
            reference[entry*2U+1U]=reused[entry*2U+1U]=0x7fc01234U+packet+entry;
        }
        LlClearStalePrequantTail(TailWords{reused.data(),uint32_t(reused.size())}, previous, entries);
        // Full K-entry output equality, including unused q/weight words, through
        // large FIRST_HOP -> small FIRST_HOP -> DIRECT and subsequent growth.
        assert(reused==reference);
        previous=entries;
        ++packet;
    }
}

int main() {
    for (uint32_t mode=0; mode<5U; ++mode) { // BF16, FP16, two quant formats, prequantized
        for (uint32_t remote=0; remote<3U; ++remote) {
            Check({}, mode, remote);
            Check(std::vector<uint32_t>(8, 7U), mode, remote); // default one token/core
            Check({0,0,1,1,2,0,0,1,1,2,2}, mode, remote); // first hops then stable DIRECT groups
            Check({0,1,2,3,0,1,2,3,0,1,2,3}, mode, remote); // no cache hits, both slots reused
            Check({31,31,31,9,9,31,31,9,9,9,9,9}, mode, remote);
        }
    }
    for (uint32_t hidden : {1U,3U,7U,31U,32U,33U,63U,65U,127U,128U,129U,256U,257U,
                            384U,512U,3584U,3591U,4096U,7168U,8192U})
        for (uint32_t topK=1; topK<=32U; ++topK) CheckPadding(hidden,topK);
    for (uint32_t topK=1; topK<=32U; ++topK) CheckPrequantTail(topK);
    std::cout << "PASS sender input reuse: opaque payload/tail conservation, delayed V/MTE, SDK scratch isolation,"
                 " exact padding and full-K prequant tail bytes\n";
}
