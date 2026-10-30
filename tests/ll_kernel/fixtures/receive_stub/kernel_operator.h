#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#define __aicore__
using GM_ADDR=uint8_t*;
namespace AscendC {
enum class HardEvent { V_S };
inline unsigned scalarStores=0,vectorCalls=0,waits=0;
inline std::vector<std::function<void()>> queued;
struct Storage { std::vector<uint8_t> bytes; };
template<class T> struct LocalTensor {
 std::shared_ptr<Storage> storage;size_t offset=0;
 T GetValue(size_t i) const {T value;assert(offset+(i+1)*sizeof(T)<=storage->bytes.size());std::memcpy(&value,storage->bytes.data()+offset+i*sizeof(T),sizeof(T));return value;}
 void SetValue(size_t i,T value) {++scalarStores;assert(offset+(i+1)*sizeof(T)<=storage->bytes.size());std::memcpy(storage->bytes.data()+offset+i*sizeof(T),&value,sizeof(T));}
 template<class U> LocalTensor<U> ReinterpretCast() const {return {storage,offset};}
 LocalTensor operator[](size_t i) const {return {storage,offset+i*sizeof(T)};}
};
struct TBufBase {
 std::shared_ptr<Storage> storage;
 template<class T> LocalTensor<T> Get(){return {storage,0};}
};
template<int Dummy=0> struct TBuf:TBufBase {};
struct TPipe {
 template<int N> void InitBuffer(TBuf<N>& b,size_t bytes) {b.storage=std::make_shared<Storage>();b.storage->bytes.assign(bytes,0xa5);}
 int FetchEventID(HardEvent){return 0;}
};
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {++waits;for(auto& f:queued)f();queued.clear();}
template<class T> void CreateVecIndex(LocalTensor<T> dst,T first,uint32_t n) {
 assert(dst.offset%32==0);assert(dst.offset+uint64_t(n)*sizeof(T)<=dst.storage->bytes.size());++vectorCalls;
 queued.push_back([=]() {for(uint32_t i=0;i<n;++i){T v=first+T(i);std::memcpy(dst.storage->bytes.data()+dst.offset+i*sizeof(T),&v,sizeof(T));}});
}
}
