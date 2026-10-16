#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#define __aicore__
#define __gm__
#define __ubuf__
constexpr int PIPE_ALL=0;
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S, MTE2_S, V_S };
inline bool events[4][8]{};
template<HardEvent E> void SetFlag(uint32_t id) {
    if(id>=8 || events[int(E)][id]) throw std::runtime_error("event reused while live");
    events[int(E)][id]=true;
}
template<HardEvent E> void WaitFlag(uint32_t id) {
    if(id>=8 || !events[int(E)][id]) throw std::runtime_error("wait without event");
    events[int(E)][id]=false;
}
using std::printf;
inline uint64_t cycles=0;
inline uint64_t GetSystemCycle() { return cycles; }
enum class CacheMode { CACHE_MODE_DISABLE };
inline uint32_t core=0, cores=16;
inline uint32_t GetBlockIdx() { return core; }
inline uint32_t GetBlockNum() { return cores; }
inline void Trap() { throw std::runtime_error("device trap"); }
template<int> inline void PipeBarrier() {}
template<class T> struct LocalTensor {
    T* p;
    T* GetPhyAddr() { return p; }
    template<class U> LocalTensor<U> ReinterpretCast() { return {reinterpret_cast<U*>(p)}; }
};
template<class T> struct GlobalTensor {
    T* p=nullptr;
    T& operator()(uint32_t index) { return p[index]; }
    void SetGlobalBuffer(T* q) { p=q; }
    void SetL2CacheHint(CacheMode) {}
};
struct DataCopyExtParams { uint16_t count; uint32_t bytes, srcStride, dstStride, reserved; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template<class T> void DataCopy(GlobalTensor<T> d, LocalTensor<T> s, uint32_t n) { std::memcpy(d.p,s.p,n*sizeof(T)); }
template<class T> void DataCopy(LocalTensor<T> d, GlobalTensor<T> s, uint32_t n) { std::memcpy(d.p,s.p,n*sizeof(T)); }
template<class T> void DataCopyPad(GlobalTensor<T> d, LocalTensor<T> s, DataCopyExtParams c) {
    for (uint32_t i=0;i<c.count;++i) std::memcpy(reinterpret_cast<uint8_t*>(d.p)+i*(c.bytes+c.dstStride),
        reinterpret_cast<uint8_t*>(s.p)+i*((c.bytes+31U)/32U*32U+c.srcStride*32U),c.bytes);
}
template<class T> void DataCopyPad(LocalTensor<T> d, GlobalTensor<T> s, DataCopyExtParams c, DataCopyPadExtParams<T>) {
    for (uint32_t i=0;i<c.count;++i) std::memcpy(reinterpret_cast<uint8_t*>(d.p)+i*((c.bytes+31U)/32U*32U+c.dstStride*32U),
        reinterpret_cast<uint8_t*>(s.p)+i*(c.bytes+c.srcStride),c.bytes);
}
}
