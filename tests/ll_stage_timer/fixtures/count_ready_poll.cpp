#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_count_ready_extract.h"
#define __aicore__
constexpr unsigned PIPE_V=0;
constexpr unsigned CNT_GM_STRIDE_BLOCKS=4;
inline unsigned configuredP=1;
inline unsigned vfCalls=0,referenceMarks=0;
inline bool vfInFlight=false,snapshotVisible=false;
struct ReferencePoints {
 template<unsigned Word> void Mark(){static_assert(Word==25||Word==11);assert(!vfInFlight);++referenceMarks;}
};
inline ReferencePoints referencePoints;
enum class HardEvent {MTE2_V,V_S,S_V};
template<HardEvent E> inline void SyncFunc(){
 if constexpr(E==HardEvent::MTE2_V)snapshotVisible=true;
 if constexpr(E==HardEvent::V_S)vfInFlight=false;
}
template<unsigned> inline void PipeBarrier(){}
template<typename T>struct LocalTensor {
 uint32_t* p;
 T GetValue(unsigned i)const {assert(!vfInFlight);T v;std::memcpy(&v,p+i,4);return v;}
 void SetValue(unsigned i,T v){assert(!vfInFlight);std::memcpy(p+i,&v,4);}
 uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(p);}
};
struct Buffer {
 alignas(256) uint32_t words[1032*8]{};
 template<typename T> LocalTensor<T> Get(){return {words};}
 template<typename T> LocalTensor<T> GetWithOffset(unsigned count,unsigned byteOffset){
  assert(byteOffset+count*4<=96U);return {words+byteOffset/4};
 }
};
struct GlobalTensor {
 uint32_t* p;
 float* operator[](uint64_t word)const{return reinterpret_cast<float*>(p+word);}
};
struct DataCopyParams {uint16_t count,blockLen,srcStride,dstStride;};
struct GatherParams {uint16_t a,count,c,d;};
inline void DataCopy(LocalTensor<float> dst,float* src,DataCopyParams params){
 assert(!vfInFlight);snapshotVisible=false;
 assert(params.blockLen==1 && params.dstStride==0);
 for(unsigned i=0;i<params.count;++i)std::memcpy(dst.p+i*8,src+i*(params.srcStride+1U)*8,32);
}
inline void GatherMask(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<uint32_t> pattern,
 bool,unsigned,GatherParams params,uint64_t& written){
 assert(snapshotVisible && !vfInFlight);
 const unsigned field=pattern.GetValue(0)==1?0:1;
 for(unsigned i=0;i<params.count;++i)dst.p[i]=src.p[i*8+field];
 written=params.count;
}
inline void ReduceSum(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float>,unsigned count){
 float sum=0;for(unsigned i=0;i<count;++i)sum+=src.GetValue(i);dst.SetValue(0,sum);
}
template<auto Fn>inline void asc_vf_call(uint32_t* records,uint32_t* counts,uint32_t* ready,unsigned count){
 assert(snapshotVisible && !vfInFlight);++vfCalls;
 CountMock::Bind(records,counts,ready,count);
 Fn(records,counts,ready,count);vfInFlight=true;
}
namespace DispatchDedup {
inline unsigned LlSmallBPrefixProducerCount(unsigned world){return world==16?configuredP:26U;}
struct ServerDedupCounts {
 ReferencePoints* profile_=&referencePoints;
 unsigned recStatusNumPerCore_=0,epWorldSize_=16,startStatusIndex_=0,clears=0;
 Buffer waitStatusBuf_,gatherMaskOutBuf_,scalarBuf_,sumContinueBuf_;
 GlobalTensor windowInstatusFp32Tensor_;
 bool TryParallelRecvCnt();
 void WaitDispatchClearStatus(){
  assert(!vfInFlight);
  // A clear cannot be issued for an incomplete snapshot. The production clear
  // still uses the unchanged disjoint strided record range after this call.
  for(unsigned i=0;i<recStatusNumPerCore_;++i){
   auto& flag=windowInstatusFp32Tensor_.p[(startStatusIndex_+i)*CNT_GM_STRIDE_BLOCKS*8U];
   assert(flag==0x3f800000U);flag=0;
  }
  ++clears;
 }
};
// Verbatim method body extracted from the production header by the test driver.
#include "actual_count_poll.h"
}
int main(){
 alignas(256) uint32_t gm[1040*CNT_GM_STRIDE_BLOCKS*8],saved[1040*CNT_GM_STRIDE_BLOCKS*8];
 DispatchDedup::ServerDedupCounts state;
 unsigned cases=0;
 for(unsigned world:{16U,32U})for(unsigned p:{1U,6U,26U})
 for(unsigned n:{0U,1U,7U,8U,63U,64U,65U,384U,1024U})for(unsigned start:{0U,3U}){
  configuredP=p;vfCalls=referenceMarks=0;vfInFlight=snapshotVisible=false;
  state.recStatusNumPerCore_=n;state.epWorldSize_=world;state.startStatusIndex_=start;state.clears=0;
  state.windowInstatusFp32Tensor_={gm};
  std::fill(std::begin(gm),std::end(gm),0xcab0cab0U);
  for(unsigned i=0;i<n;++i){gm[(start+i)*32]=0x3f800000U;gm[(start+i)*32+1]=0x80000001U+i;}
  if(!n){assert(state.TryParallelRecvCnt());assert(!vfCalls && !state.clears && referenceMarks==2);++cases;continue;}
  gm[(start+n-1)*32]=0;
  std::copy(std::begin(gm),std::end(gm),saved);
  assert(!state.TryParallelRecvCnt());
  assert(!vfInFlight && !state.clears && !referenceMarks);
  assert(std::equal(std::begin(gm),std::end(gm),saved));
  const bool fused=world==16 && p==1;
  assert(vfCalls==unsigned(fused));
  if(fused)for(unsigned i=0;i<n;++i)assert(state.gatherMaskOutBuf_.words[i]==0x80000001U+i);
  // A retry takes a new snapshot, including fresh count values. The failed
  // snapshot must be retired before this second MTE2 copy replaces its bytes.
  for(unsigned i=0;i<n;++i){gm[(start+i)*32]=0x3f800000U;gm[(start+i)*32+1]=0xf1000001U+i;}
  assert(state.TryParallelRecvCnt());
  assert(!vfInFlight && state.clears==1 && referenceMarks==2 && vfCalls==2U*unsigned(fused));
  for(unsigned i=0;i<n;++i){
   assert(gm[(start+i)*32]==0);
   assert(gm[(start+i)*32+1]==0xf1000001U+i);
   assert(state.gatherMaskOutBuf_.words[i]==0xf1000001U+i);
  }
  for(unsigned word=0;word<1040*32U;++word){
   const unsigned record=word/32;
   if(record<start || record>=start+n)assert(gm[word]==0xcab0cab0U);
  }
  ++cases;
 }
 std::printf("actual count polling body: %u cases passed\n",cases);
}
