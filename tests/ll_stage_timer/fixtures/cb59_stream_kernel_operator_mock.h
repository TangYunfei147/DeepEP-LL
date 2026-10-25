#pragma once
// CPU-only API emulation for the real StreamCounts/ReceiveTasks headers.
// Copies complete synchronously under one GM mutex. This checks indices,
// protocol dependencies, bounds and ownership, NOT NPU cache/DMA ordering.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline thread_local uint32_t emulatedCore = 0;
inline std::mutex gmMutex;
inline uint32_t GetBlockIdx() { return emulatedCore; }
inline uint32_t GetBlockNum() { return 56U; }
inline void Trap() { std::abort(); }
enum class HardEvent { MTE2_S, S_MTE2, V_MTE3, V_S, S_MTE3 };
template<HardEvent> inline void SetFlag(unsigned) {}
template<HardEvent> inline void WaitFlag(unsigned) { std::this_thread::yield(); }
struct DataCopyParams { uint32_t blockCount, blockLen, srcStride, dstStride; };
struct DataCopyExtParams { uint32_t blockCount, blockLen, srcStride, dstStride, reserved; };
template<class T> class LocalTensor {
public:
    T* pointer = nullptr;
    uint64_t elements = 0;
    T GetValue(uint64_t i) const { assert(i < elements); return pointer[i]; }
    void SetValue(uint64_t i, T value) { assert(i < elements); pointer[i] = value; }
    LocalTensor operator[](uint64_t i) const { assert(i <= elements); return {pointer + i, elements - i}; }
};
template<class T> class GlobalTensor {
public:
    T* pointer = nullptr;
    void SetGlobalBuffer(T* value, uint64_t = 0) { pointer = value; }
    GlobalTensor operator[](uint64_t i) const { return {pointer + i}; }
};
struct Storage {
    std::vector<uint8_t> raw;
    uint8_t* data = nullptr;
    uint32_t bytes = 0;
    explicit Storage(uint32_t size) : raw(size + 31U, 0xcdU), bytes(size) {
        data = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(raw.data()) + 31U) & ~uintptr_t(31U));
    }
};
template<int = 0> class TBuf {
public:
    std::shared_ptr<Storage> storage;
    template<class T> LocalTensor<T> Get() {
        return storage ? LocalTensor<T>{reinterpret_cast<T*>(storage->data), storage->bytes / sizeof(T)} : LocalTensor<T>{};
    }
};
class MockPool {
public:
    uint32_t bytes = 0, buffers = 0;
    void InitBuffer(TBuf<>& buffer, uint32_t size) {
        assert(size && size % 32U == 0U && buffers < 32U);
        bytes += size;
        ++buffers;
        assert(bytes <= 190U * 1024U);
        buffer.storage = std::make_shared<Storage>(size);
    }
};
class TPipe {
public:
    unsigned FetchEventID(HardEvent) { return 0; }
    void InitBuffer(TBuf<>& buffer, uint32_t size) { fallback.InitBuffer(buffer, size); }
private:
    MockPool fallback;
};
template<class T> inline void Duplicate(LocalTensor<T> target, T value, uint32_t count) {
    assert(count <= target.elements);
    for(uint32_t i = 0; i < count; ++i) target.SetValue(i, value);
}
template<class T> inline void Duplicate(LocalTensor<T> target, T value, const uint64_t mask[2],
                                      uint8_t repeats, uint32_t blockStride, uint32_t repeatStride) {
    static_assert(sizeof(T) == 4);
    for(uint32_t repeat = 0; repeat < repeats; ++repeat)
        for(uint32_t lane = 0; lane < 64U; ++lane)
            if(mask[0] & (uint64_t(1) << lane))
                target.SetValue(repeat * repeatStride * 8U + lane / 8U * blockStride * 8U + lane % 8U, value);
}
template<class T> inline void DataCopy(LocalTensor<T> target, GlobalTensor<T> source, DataCopyParams p) {
    std::lock_guard<std::mutex> lock(gmMutex);
    for(uint32_t i = 0; i < p.blockCount; ++i) {
        const uint64_t dst = uint64_t(i) * (p.blockLen + p.dstStride) * 32U;
        const uint64_t src = uint64_t(i) * (p.blockLen + p.srcStride) * 32U;
        assert(dst + p.blockLen * 32U <= target.elements * sizeof(T));
        std::memcpy(reinterpret_cast<uint8_t*>(target.pointer) + dst,
                    reinterpret_cast<const uint8_t*>(source.pointer) + src, p.blockLen * 32U);
    }
}
template<class T> inline void DataCopy(GlobalTensor<T> target, LocalTensor<T> source, DataCopyParams p) {
    std::lock_guard<std::mutex> lock(gmMutex);
    for(uint32_t i = 0; i < p.blockCount; ++i) {
        const uint64_t dst = uint64_t(i) * (p.blockLen + p.dstStride) * 32U;
        const uint64_t src = uint64_t(i) * (p.blockLen + p.srcStride) * 32U;
        assert(src + p.blockLen * 32U <= source.elements * sizeof(T));
        std::memcpy(reinterpret_cast<uint8_t*>(target.pointer) + dst,
                    reinterpret_cast<const uint8_t*>(source.pointer) + src, p.blockLen * 32U);
    }
}
template<class T> inline void DataCopyPad(GlobalTensor<T> target, LocalTensor<T> source, DataCopyExtParams p) {
    std::lock_guard<std::mutex> lock(gmMutex);
    for(uint32_t i = 0; i < p.blockCount; ++i) {
        const uint64_t dst = uint64_t(i) * (p.blockLen + p.dstStride);
        const uint64_t src = uint64_t(i) * (p.blockLen + p.srcStride * 32U);
        assert(src + p.blockLen <= source.elements * sizeof(T));
        std::memcpy(reinterpret_cast<uint8_t*>(target.pointer) + dst,
                    reinterpret_cast<const uint8_t*>(source.pointer) + src, p.blockLen);
    }
}
} // namespace AscendC
