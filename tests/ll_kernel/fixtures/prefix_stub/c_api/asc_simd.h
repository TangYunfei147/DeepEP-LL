#pragma once
#include <stdint.h>
#include <assert.h>
#define __simd_callee__
#define __simd_vf__
#define __ubuf__
struct Vec { uint32_t v[64]; };
using vector_uint32_t=Vec;
using vector_int32_t=Vec;
struct vector_bool { bool v[64]; };
constexpr int PAT_ALL=1;
inline unsigned loads=0,stores=0;
inline vector_bool asc_create_mask_b32(int) {vector_bool p{};for(auto& x:p.v)x=true;return p;}
inline vector_bool asc_update_mask_b32(uint32_t& left) {vector_bool p{};for(unsigned i=0;i<64;++i)p.v[i]=i<left;left=left>64?left-64:0;return p;}
inline void asc_arange(Vec& d,int32_t x) {for(unsigned i=0;i<64;++i)d.v[i]=uint32_t(x)+i;}
inline void asc_duplicate_scalar(Vec& d,uint32_t x) {for(auto& v:d.v)v=x;}
inline void asc_loadalign(Vec& d,uint32_t* p) {assert((uintptr_t(p)&255U)==0);++loads;for(unsigned i=0;i<64;++i)d.v[i]=p[i];}
inline void asc_storealign(uint32_t* p,Vec s,vector_bool m) {assert((uintptr_t(p)&255U)==0);++stores;for(unsigned i=0;i<64;++i)if(m.v[i])p[i]=s.v[i];}
inline void asc_sub(Vec& d,Vec a,Vec b,vector_bool m) {for(unsigned i=0;i<64;++i)d.v[i]=m.v[i]?a.v[i]-b.v[i]:0;}
inline void asc_add(Vec& d,Vec a,Vec b,vector_bool m) {for(unsigned i=0;i<64;++i)d.v[i]=m.v[i]?a.v[i]+b.v[i]:0;}
inline vector_bool asc_ge_scalar(Vec a,uint32_t b,vector_bool m) {for(unsigned i=0;i<64;++i)m.v[i]=m.v[i]&&a.v[i]>=b;return m;}
inline void asc_select(Vec& d,Vec a,Vec b,vector_bool m) {for(unsigned i=0;i<64;++i)d.v[i]=m.v[i]?a.v[i]:b.v[i];}
inline void asc_gather(Vec& d,Vec s,Vec idx) {for(unsigned i=0;i<64;++i){assert(idx.v[i]<64);d.v[i]=s.v[idx.v[i]];}}
