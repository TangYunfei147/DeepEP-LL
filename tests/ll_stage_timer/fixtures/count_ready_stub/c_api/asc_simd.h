#pragma once
#include <cassert>
#include <cstdint>
#include <algorithm>
#define __simd_vf__
#define __ubuf__
struct Vec {uint32_t v[64];};
using vector_uint32_t=Vec;
using vector_int32_t=Vec;
struct vector_bool {bool v[64];};
constexpr int PAT_ALL=1;
namespace CountMock {
inline uint32_t* inputBegin=nullptr;
inline uint32_t* inputEnd=nullptr;
inline uint32_t* outputBegin=nullptr;
inline uint32_t* outputEnd=nullptr;
inline uint32_t* result=nullptr;
inline unsigned reads=0,outputWrites=0,resultWrites=0;
inline void Bind(uint32_t* input,uint32_t* output,uint32_t* status,uint32_t count){
 inputBegin=input;inputEnd=input+count*8U;outputBegin=output;outputEnd=output+count;
 result=status;reads=outputWrites=resultWrites=0;
}
}
inline vector_bool asc_create_mask_b32(int){vector_bool m{};for(auto& x:m.v)x=true;return m;}
inline vector_bool asc_update_mask_b32(uint32_t& n){vector_bool m{};for(unsigned i=0;i<64;++i)m.v[i]=i<n;n=n>64?n-64:0;return m;}
inline void asc_arange(Vec& d,int32_t begin){for(unsigned i=0;i<64;++i)d.v[i]=uint32_t(begin)+i;}
inline void asc_duplicate_scalar(Vec& d,uint32_t x){for(auto& v:d.v)v=x;}
inline void asc_add(Vec& d,Vec a,Vec b,vector_bool m){for(unsigned i=0;i<64;++i)d.v[i]=m.v[i]?a.v[i]+b.v[i]:0;}
inline void asc_gather(Vec& d,uint32_t* src,Vec index,vector_bool m){
 assert((uintptr_t(src)&31U)==0);
 for(unsigned i=0;i<64;++i){
  if(!m.v[i]){d.v[i]=0;continue;}
  // The public VF API indexes elements, not bytes. Invalid lanes never load.
  assert(src+index.v[i]>=CountMock::inputBegin && src+index.v[i]<CountMock::inputEnd);
  d.v[i]=src[index.v[i]];++CountMock::reads;
 }
}
inline vector_bool asc_eq_scalar(Vec src,uint32_t value,vector_bool m){for(unsigned i=0;i<64;++i)m.v[i]=m.v[i]&&src.v[i]==value;return m;}
inline void asc_select(Vec& d,Vec a,Vec b,vector_bool m){for(unsigned i=0;i<64;++i)d.v[i]=m.v[i]?a.v[i]:b.v[i];}
inline void asc_reduce_min(Vec& d,Vec src,vector_bool m){
 uint32_t minimum=UINT32_MAX;for(unsigned i=0;i<64;++i)if(m.v[i])minimum=std::min(minimum,src.v[i]);
 for(auto& x:d.v)x=0xeeeeeeeeU;d.v[0]=minimum;
}
inline void asc_storealign(uint32_t* dst,Vec src,vector_bool m){
 assert((uintptr_t(dst)&31U)==0);
 for(unsigned i=0;i<64;++i)if(m.v[i]){
  if(dst==CountMock::result){assert(i==0);++CountMock::resultWrites;}
  else {assert(dst+i>=CountMock::outputBegin && dst+i<CountMock::outputEnd);++CountMock::outputWrites;}
  dst[i]=src.v[i];
 }
}
