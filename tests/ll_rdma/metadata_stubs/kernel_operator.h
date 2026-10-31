#pragma once
#include "../stubs/kernel_operator.h"
#include <memory>
#include <algorithm>
#include <cmath>
constexpr int PIPE_V=1, PIPE_MTE3=2;
using TEventID=uint32_t;
namespace AscendC {
inline uint32_t Ceil(uint64_t n,uint64_t d){return uint32_t((n+d-1)/d);}
enum class TPosition { VECCALC, VECIN, VECOUT };
template<class=void> struct TBuf {
    std::shared_ptr<std::vector<uint64_t>> storage;
    template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(storage->data())};}
    template<class T> LocalTensor<T> GetWithOffset(uint32_t,uint32_t offset){return {reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(storage->data())+offset)};}
};
template<TPosition,int> struct TBufPool {
    template<class B> void InitBuffer(B& b,uint32_t n){b.storage=std::make_shared<std::vector<uint64_t>>((n+7)/8);}
};
struct TPipe {
    uint32_t bytes=0;
    template<class B> void InitBuffer(B& b,uint32_t n){bytes+=n;if(bytes>190*1024)throw std::runtime_error("UB overflow");b.storage=std::make_shared<std::vector<uint64_t>>((n+7)/8);}
    template<class P> void InitBufPool(P&,uint32_t){}
    template<class P,class Parent> void InitBufPool(P&,uint32_t,Parent&){}
    uint32_t FetchEventID(HardEvent){return 0;}
    void Reset(){bytes=0;}
};
template<class T> void Duplicate(LocalTensor<T> d,T v,uint32_t n){std::fill_n(d.p,n,v);}
template<class T> void Duplicate(LocalTensor<T> d,T v,uint32_t n,uint32_t repeats,uint32_t,uint32_t stride){
    for(uint32_t r=0;r<repeats;++r)std::fill_n(d.p+r*stride*32/sizeof(T),n,v);
}
template<class T> void Duplicate(LocalTensor<T> d,T v,uint64_t* mask,uint32_t repeats,uint32_t,uint32_t stride){
    for(uint32_t r=0;r<repeats;++r)for(uint32_t i=0;i<64;++i)if(mask[0]&(uint64_t(1)<<i))d.p[r*stride*32/sizeof(T)+i]=v;
}
template<class T> void CreateVecIndex(LocalTensor<T> d,T first,uint32_t n){for(uint32_t i=0;i<n;++i)d.p[i]=first+T(i);}
struct SumParams {uint32_t rows,stride,count;};
template<class T> void Sum(LocalTensor<T> d,LocalTensor<T> s,LocalTensor<uint8_t>,SumParams p){
    for(uint32_t r=0;r<p.rows;++r){T v=0;for(uint32_t i=0;i<p.count;++i)v+=s.p[r*p.stride+i];d.p[r]=v;}
}
template<class T> void Sum(LocalTensor<T> d,LocalTensor<T> s,SumParams p){Sum(d,s,LocalTensor<uint8_t>{nullptr},p);}
template<class T> void ReduceSum(LocalTensor<T> d,LocalTensor<T> s,LocalTensor<T> work,uint32_t n){
    T sum=0;for(uint32_t i=0;i<n;++i)sum+=s.p[i];
    std::fill_n(work.p,64,T(-123));d.p[0]=sum; // Scratch is explicitly destructive.
}
template<class T> void Sub(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,uint32_t n){for(uint32_t i=0;i<n;++i)d.p[i]=a.p[i]-b.p[i];}
template<class T> void Adds(LocalTensor<T> d,LocalTensor<T> a,T b,uint32_t n){for(uint32_t i=0;i<n;++i)d.p[i]=a.p[i]+b;}
template<class T> void And(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<T> b,uint32_t n){for(uint32_t i=0;i<n;++i)d.p[i]=a.p[i]&b.p[i];}
template<class T> void Mins(LocalTensor<T> d,LocalTensor<T> a,T b,uint32_t n){for(uint32_t i=0;i<n;++i)d.p[i]=std::min(a.p[i],b);}
inline void Abs(LocalTensor<float> d,LocalTensor<float> a,uint32_t n){for(uint32_t i=0;i<n;++i){uint32_t v;std::memcpy(&v,a.p+i,4);v&=0x7fffffff;std::memcpy(d.p+i,&v,4);}}
enum class CMPMODE {EQ,GE,LT};
template<class T> void Compares(LocalTensor<uint8_t> d,LocalTensor<T> a,T b,CMPMODE m,uint32_t n){
    std::memset(d.p,0,(n+7)/8);for(uint32_t i=0;i<n;++i){bool yes=m==CMPMODE::EQ?a.p[i]==b:m==CMPMODE::GE?a.p[i]>=b:a.p[i]<b;if(yes)d.p[i/8]|=1U<<(i%8);}
}
struct GatherParams {uint16_t srcBlockStride,repeatTimes,srcRepeatStride,src0BlockStride;};
template<class T> void GatherMask(LocalTensor<T> d,LocalTensor<T> a,LocalTensor<uint32_t> mask,bool,uint32_t count,GatherParams p,uint64_t& n){
    n=0;
    if(p.srcRepeatStride==1){for(uint32_t r=0;r<p.repeatTimes;++r)for(uint32_t i=0;i<count;++i)if(mask.p[0]&(1U<<i))d.p[n++]=a.p[r*8+i];}
    else {auto bits=reinterpret_cast<uint8_t*>(mask.p);for(uint32_t i=0;i<count;++i)if(bits[i/8]&(1U<<(i%8)))d.p[n++]=a.p[i];}
}
template<class T> void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyParams c){
    std::lock_guard<std::mutex> lock(memoryMutex);
    for(uint32_t i=0;i<c.blockCount;++i)std::memcpy(reinterpret_cast<uint8_t*>(dst.p)+i*(c.blockLen+c.dstStride)*32,
        reinterpret_cast<uint8_t*>(src.p)+i*(c.blockLen+c.srcStride)*32,c.blockLen*32);
}
}
