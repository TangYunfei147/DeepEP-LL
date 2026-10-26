"""Contracts for count UB lifetimes, publication and prefix reuse."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEVICE = ROOT / "csrc/kernels/dispatch/low_latency/device"


def _source(name):
    return (DEVICE / name).read_text(encoding="utf-8")


def _block(source, marker):
    """Return the actual braced production block after a unique marker."""
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f"Unclosed block: {marker}")


def test_count_prefix_is_loaded_only_by_its_output_owner():
    counts = _source("dispatch_server_dedup_counts.h")
    complete = _block(counts, "void ServerDedupCounts::CompleteCumsum()")
    owner = _block(complete, "if (receiver == 0U)")
    before_owner = complete[:complete.index("if (receiver == 0U)")]
    assert "WaitCumSumFlag();" in before_owner  # Every receiver still retires its ready row.
    assert "prefixBuf_" not in before_owner
    assert "DataCopyPad(prefix," not in before_owner
    assert "InitBuffer(prefixBuf_" in owner
    assert owner.index("DataCopyPad(prefix,") < owner.index("SyncFunc<HardEvent::MTE2_S>()")
    assert owner.index("SyncFunc<HardEvent::MTE2_S>()") < owner.index("prefix.GetValue(")
    assert "DataCopyPad(expertTokenNumsOutGMTensor_" in owner
    assert "LocalTensor<int32_t> Prefix()" not in counts
    receiver = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert receiver.index("pipe_->Reset();") < receiver.index("DataCopyPad(prefix, prefixGlobal")
    assert receiver.index("ServerDedupReceiveSync<HardEvent::MTE2_S>()") < receiver.index("tasks_.Build(prefix")


def test_paired_count_records_live_until_the_existing_reuse_barrier():
    counts = _source("dispatch_server_dedup_counts.h")
    send = _block(counts, "void ServerDedupCounts::SendCnt()")
    first = send.index("if (count) SendExpertCntRange(start, end)")
    overwrite = send.index("routeExchange_.ReadPeer")
    assert "SyncFunc<HardEvent::MTE3_S>();" in send[first:overwrite]
    last = send.index("SendExpertCntRange(start,end)")
    assert "SyncFunc<HardEvent::MTE3_S>();" not in send[last:]
    assert "Reset(" not in send
    assert "tpipe_->InitBuffer(statusBuf_, ReservedBytes())" in send
    assert "kEpServerDedupCountUbBytes - ReservedBytes()" in send
    assert "return stagePool_;" in _block(counts, "ScratchPool()")
    assert "PipeBarrier<PIPE_ALL>();" in _block(counts, "void Finish()")
    kernel = (DEVICE.parent / "dispatch_server_dedup_kernel.h").read_text(encoding="utf-8")
    assert "sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), timer_)" in kernel
    complete = kernel.index("counts_.CompleteCumsum();")
    consume = kernel.index("receiver_.Process(timer_);")
    assert "PipeBarrier<PIPE_ALL>();" in kernel[complete:consume]


def test_only_the_route_owner_waits_for_scalar_body_preparation():
    send = _block(_source("dispatch_server_dedup_counts.h"), "void ServerDedupCounts::SendCnt()")
    owner = _block(send, "if (pairedRoutes_ && aivId_ == 0U)")
    assert "SyncFunc<HardEvent::MTE2_S>();" in owner
    assert owner.index("SyncFunc<HardEvent::MTE2_S>();") < owner.index("routeExchange_.PostBody")
    assert send.count("SyncFunc<HardEvent::MTE2_S>();") == 1
    # All count workers still acquire input for Vector before their first use.
    loaded = send.index("DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_")
    counted = send.index("CountByVectorRange(start, end, maskBytes)")
    assert "SyncFunc<HardEvent::MTE2_V>();" in send[loaded:counted]


def test_prefix_subtotal_executes_production_scalar_scan(tmp_path):
    """Compile the production body with a delayed GatherMask dependency model."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for C++ CPU validation")
    body = _block(_source("dispatch_server_dedup_counts.h"), "void ServerDedupCounts::GatherSumRecvCnt(")
    assert "\n    Sum(" not in body
    assert body.index("GatherMask(") < body.index("SyncFunc<AscendC::HardEvent::V_S>()")
    assert body.index("SyncFunc<AscendC::HardEvent::V_S>()") < body.index("prefix.GetValue(")
    source = tmp_path / "count-prefix.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
namespace AscendC { enum class HardEvent { S_V, V_S }; }
using AscendC::HardEvent;
static float* pendingDst=nullptr;
static std::vector<float> pending;
template<class T> struct LocalTensor {
    T* ptr;
    T GetValue(uint32_t i) const {
        assert(!pendingDst);T result;std::memcpy(&result,ptr+i,sizeof(T));return result;
    }
    void SetValue(uint32_t i,T value) const {
        assert(!pendingDst);std::memcpy(ptr+i,&value,sizeof(T));
    }
    template<class U> LocalTensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(ptr)};}
};
struct GatherParams { uint16_t repeats,records,srcStride,dstStride; };
static void GatherMask(LocalTensor<float> out,LocalTensor<float> input,LocalTensor<uint32_t> pattern,
                       bool,uint32_t mask,GatherParams shape,uint64_t& elements) {
    assert(!pendingDst && mask==2U && pattern.GetValue(0)==2U);
    assert(shape.repeats==1U && shape.srcStride==1U && shape.dstStride==0U);
    pending.resize(shape.records);
    for(unsigned i=0;i<shape.records;++i) pending[i]=input.GetValue(i*8U+1U);
    pendingDst=out.ptr;elements=shape.records;
}
struct Fixture {
    LocalTensor<float> statusFp32Tensor_;
    uint32_t recStatusNumPerCore_=0,published=0;
    template<HardEvent event> void SyncFunc() {
        if constexpr(event==HardEvent::V_S) {
            assert(pendingDst);std::memcpy(pendingDst,pending.data(),pending.size()*sizeof(float));
            pendingDst=nullptr;pending.clear();
        } else assert(!pendingDst);
    }
    void PublishRecvCntSum(float value) {std::memcpy(&published,&value,sizeof(value));}
    void Run(LocalTensor<float>& gatherMaskOutTensor,LocalTensor<uint32_t>& gatherTmpTensor) {
''' + body + r'''
    }
};
static uint32_t Bits(float value) {uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
int main() {
    unsigned cases=0;
    for(unsigned count: {1U,7U,8U,13U,16U,31U,64U,128U}) for(unsigned mode=0;mode<4;++mode) {
        std::vector<float> records(count*8U),prefix(count+8U);
        std::vector<uint32_t> expected(count),pattern(8U);
        uint32_t sum=0;
        for(unsigned i=0;i<count;++i) {
            uint32_t value=mode==0U?0U:mode==1U?16384U:mode==2U?(i%5U==0U?i+1U:0U):(i*127U+3U)%16385U;
            std::memcpy(records.data()+i*8U+1U,&value,4);sum+=value;expected[i]=sum;
            const uint32_t ready=0x3f800000U;std::memcpy(records.data()+i*8U,&ready,4);
        }
        const uint32_t guard=0x5a5a5a5aU;
        for(unsigned i=count;i<prefix.size();++i) std::memcpy(prefix.data()+i,&guard,4);
        Fixture fixture{{records.data()},count};
        LocalTensor<float> output{prefix.data()};LocalTensor<uint32_t> temporary{pattern.data()};
        fixture.Run(output,temporary);
        assert(fixture.published==sum && !pendingDst);
        for(unsigned i=0;i<count;++i) assert(Bits(prefix[i])==expected[i]);
        for(unsigned i=count;i<prefix.size();++i) assert(Bits(prefix[i])==guard);
        ++cases;
    }
    std::cout<<"PASS production count prefix: "<<cases<<" partitions, exact subtotal bits, delayed Vector completion, UB guards\n";
}
''', encoding="utf-8")
    executable = tmp_path / "count-prefix.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", str(source), "-o", str(executable)],
                   check=True, timeout=180)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS production count prefix: 32 partitions" in result.stdout
