#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kUbBytes = 131072;
// 仅测本卡 GM 写的有效发射窗口。门控边界不是 RDMA/互联 OTS 规格。
#ifndef MTE3_DEPTH_BYPASS
#define MTE3_DEPTH_BYPASS 0
#endif
#ifndef MTE3_DEPTH_GATE_REPEATS
#define MTE3_DEPTH_GATE_REPEATS 0
#endif

// 编译期展开：每条请求使用独立 UB 源区域和独立 GM 目标地址。
template<unsigned I, unsigned N, bool Serial>
__aicore__ inline void Issue(LocalTensor<uint8_t> ub, GlobalTensor<uint8_t> target,
                             uint32_t chunk, uint32_t stride) {
    if constexpr (I < N) {
        DataCopy(target[I * stride], ub[I * chunk], chunk);
        if constexpr (Serial) {
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }
        Issue<I + 1, N, Serial>(ub, target, chunk, stride);
    }
}
template<unsigned N, bool Serial>
__aicore__ inline void Measure(LocalTensor<uint8_t> ub, GlobalTensor<uint8_t> target,
    LocalTensor<float> gate, GlobalTensor<uint64_t> result, uint32_t chunk, uint32_t stride) {
#if MTE3_DEPTH_GATE_REPEATS
    // 有界、自主完成的 V 工作；不依赖后续 Scalar 发射，避免环形等待。
    // MTE3 的事件等待排在 copies 前；Scalar 仍可继续提交请求。
    for (uint32_t i = 0; i < MTE3_DEPTH_GATE_REPEATS; ++i)
        Duplicate<float>(gate, float(i), 8192);
    SetFlag<HardEvent::V_MTE3>(1); WaitFlag<HardEvent::V_MTE3>(1);
#endif
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t start = GetSystemCycle();
    Issue<0, N, Serial>(ub, target, chunk, stride);
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t issued = GetSystemCycle();
    // N=0 也保留同一空 drain，供测量开销对照。
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t done = GetSystemCycle();
    result(0) = issued - start;
    result(1) = done - issued;
    result(2) = done - start;
}
template<unsigned N = 0>
__aicore__ inline void Select(uint32_t n, bool serial, LocalTensor<uint8_t> ub,
    GlobalTensor<uint8_t> target, LocalTensor<float> gate, GlobalTensor<uint64_t> result,
    uint32_t chunk, uint32_t stride) {
    if (n == N) {
        if (serial) Measure<N, true>(ub, target, gate, result, chunk, stride);
        else Measure<N, false>(ub, target, gate, result, chunk, stride);
    } else if constexpr (N < 32) Select<N + 1>(n, serial, ub, target, gate, result, chunk, stride);
}
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mte3_depth(GM_ADDR input, GM_ADDR output, GM_ADDR ticks, uint32_t n,
                uint32_t chunk, uint32_t stride, uint32_t serial) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, kUbBytes);
    auto ub = storage.Get<uint8_t>();
    TBuf<TPosition::VECCALC> gateStorage;
    pipe.InitBuffer(gateStorage, 32768);
    auto gate = gateStorage.Get<float>();
    GlobalTensor<uint8_t> src, dst;
    GlobalTensor<uint64_t> result;
    src.SetGlobalBuffer(input); dst.SetGlobalBuffer(output);
#if MTE3_DEPTH_BYPASS
    dst.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ticks));
    // 计时前加载固定128KiB输入；Scalar等待确保preload不进入测量窗口。
    DataCopy(ub, src, kUbBytes);
    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    SetFlag<HardEvent::MTE2_MTE3>(0); WaitFlag<HardEvent::MTE2_MTE3>(0);
    Select(n, serial != 0, ub, dst, gate, result, chunk, stride);
    PipeBarrier<PIPE_ALL>();
}
extern "C" void mte3_depth_do(void* stream, uint8_t* src, uint8_t* dst,
    uint8_t* ticks, uint32_t n, uint32_t chunk, uint32_t stride, uint32_t serial) {
    mte3_depth<<<1, nullptr, stream>>>(src, dst, ticks, n, chunk, stride, serial);
}
