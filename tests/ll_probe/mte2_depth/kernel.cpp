#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kUbBytes = 131072;
#ifndef MTE2_DEPTH_BYPASS
#define MTE2_DEPTH_BYPASS 0
#endif
#ifndef MTE2_DEPTH_GATE_REPEATS
#define MTE2_DEPTH_GATE_REPEATS 0
#endif

// 编译期展开：每条请求使用独立源地址和独立 UB 区域。
template<unsigned I, unsigned N, bool Serial>
__aicore__ inline void Issue(LocalTensor<uint8_t> ub, GlobalTensor<uint8_t> src,
                             uint32_t chunk, uint32_t stride) {
    if constexpr (I < N) {
        DataCopy(ub[I * chunk], src[I * stride], chunk);
        if constexpr (Serial) {
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
        }
        Issue<I + 1, N, Serial>(ub, src, chunk, stride);
    }
}
template<unsigned N, bool Serial>
__aicore__ inline void Measure(LocalTensor<uint8_t> ub, GlobalTensor<uint8_t> src,
    LocalTensor<float> gate, GlobalTensor<uint64_t> result, uint32_t chunk, uint32_t stride) {
#if MTE2_DEPTH_GATE_REPEATS
    // 有界、自主完成的 V 工作；不依赖后续 Scalar 发射，避免环形等待。
    // MTE2 的事件等待排在 copies 前；Scalar 仍可继续提交请求。
    for (uint32_t i = 0; i < MTE2_DEPTH_GATE_REPEATS; ++i)
        Duplicate<float>(gate, float(i), 8192);
    SetFlag<HardEvent::V_MTE2>(1); WaitFlag<HardEvent::V_MTE2>(1);
#endif
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t start = GetSystemCycle();
    Issue<0, N, Serial>(ub, src, chunk, stride);
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t issued = GetSystemCycle();
    // N=0 也保留同一空 drain，供测量开销对照。
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    asm volatile("" ::: "memory"); // 仅编译器边界，不排空 MTE。
    const uint64_t done = GetSystemCycle();
    result(0) = issued - start;
    result(1) = done - issued;
    result(2) = done - start;
}
template<unsigned N = 0>
__aicore__ inline void Select(uint32_t n, bool serial, LocalTensor<uint8_t> ub,
    GlobalTensor<uint8_t> src, LocalTensor<float> gate, GlobalTensor<uint64_t> result,
    uint32_t chunk, uint32_t stride) {
    if (n == N) {
        if (serial) Measure<N, true>(ub, src, gate, result, chunk, stride);
        else Measure<N, false>(ub, src, gate, result, chunk, stride);
    } else if constexpr (N < 32) Select<N + 1>(n, serial, ub, src, gate, result, chunk, stride);
}
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mte2_depth(GM_ADDR input, GM_ADDR output, GM_ADDR ticks, uint32_t n,
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
#if MTE2_DEPTH_BYPASS
    src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ticks));
    Duplicate<uint8_t>(ub, uint8_t(0xa7), kUbBytes);
    SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
    // 等待 guard 初始化真正完成，避免其执行时间混入被测 issue/drain。
    SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    Select(n, serial != 0, ub, src, gate, result, chunk, stride);
    // 计时后导出整个 UB，包括尚未覆盖的 guard；MTE3 不属于测量范围。
    SetFlag<HardEvent::MTE2_MTE3>(0); WaitFlag<HardEvent::MTE2_MTE3>(0);
    SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(dst, ub, kUbBytes);
    PipeBarrier<PIPE_ALL>();
}
extern "C" void mte2_depth_do(void* stream, uint8_t* src, uint8_t* dst,
    uint8_t* ticks, uint32_t n, uint32_t chunk, uint32_t stride, uint32_t serial) {
    mte2_depth<<<1, nullptr, stream>>>(src, dst, ticks, n, chunk, stride, serial);
}
