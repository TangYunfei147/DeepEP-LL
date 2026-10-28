"""Execute the real M4 paging loop/selector and DIRECT stripe on CPU."""
from pathlib import Path
from tests.ll_rdma.test_combine_overhead import block, run_cpp

ROOT = Path(__file__).resolve().parents[2]


def test_combine_qps_are_disjoint_and_paging_preserves_every_row(tmp_path):
    partial = (ROOT/'csrc/kernels/combine/low_latency/combine_server_dedup_partial.h').read_text(encoding='utf-8')
    direct = (ROOT/'csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h').read_text(encoding='utf-8')
    process = block(partial, '__aicore__ inline uint32_t Process()')
    select = block(partial, '__aicore__ inline uint32_t SelectServer(')
    direct_loop = block(direct, '__aicore__ inline void ReturnDirectTokens()')
    start = direct_loop.index('        for (uint32_t step = 1U;')
    end = direct_loop.index('                const uint32_t row =', start)
    direct_loop = direct_loop[start:end] + 'visits.push_back(uint32_t(indexUb_.GetValue(task)));\n}}'
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
struct half {}; struct bfloat16_t {};
constexpr int PIPE_V=1;
namespace AscendC {
inline uint32_t hardwareCore=0;
inline uint32_t GetBlockIdx(){return hardwareCore;}
enum class HardEvent { V_MTE2, MTE2_V, V_S };
inline bool events[3][8]{};
template<HardEvent E> void SetFlag(uint32_t i){assert(!events[int(E)][i]);events[int(E)][i]=true;}
template<HardEvent E> void WaitFlag(uint32_t i){assert(events[int(E)][i]);events[int(E)][i]=false;}
template<int> void PipeBarrier(){}
inline void Trap(){assert(false);}
template<class T> struct Tensor {
 T* p;size_t n;
 T GetValue(uint32_t i) const {assert(i<n);return p[i];}
 void SetValue(uint32_t i,T v){assert(i<n);p[i]=v;}
 Tensor operator[](size_t i){assert(i<=n);return {p+i,n-i};}
 template<class U> Tensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p),n*sizeof(T)/sizeof(U)};}
};
template<class T> using LocalTensor=Tensor<T>;
struct Copy {uint16_t count;uint32_t bytes,src,dst,reserved;};
template<class T> struct Pad {bool pad;uint32_t left,right;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,Copy c,Pad<T>){
 assert(c.count==1 && c.bytes<=dst.n*sizeof(T) && c.bytes<=src.n*sizeof(T));std::memcpy(dst.p,src.p,c.bytes);
}
enum class CMPMODE {EQ,GE,LT};
template<class T> void Compares(Tensor<uint8_t> out,Tensor<T> in,T v,CMPMODE mode,uint32_t n){
 std::fill(out.p,out.p+(n+7)/8,0);
 for(unsigned i=0;i<n;++i){auto x=in.GetValue(i);bool yes=mode==CMPMODE::EQ?x==v:(mode==CMPMODE::GE?x>=v:x<v);
 if(yes)out.p[i/8]|=uint8_t(1U<<(i%8));}
}
template<class T> void CreateVecIndex(Tensor<T> x,T begin,uint32_t n){for(unsigned i=0;i<n;++i)x.SetValue(i,begin+T(i));}
template<class T> void Muls(Tensor<T> out,Tensor<T> in,T mul,uint32_t n){for(unsigned i=0;i<n;++i)out.SetValue(i,in.GetValue(i)*mul);}
inline void Gather(Tensor<int32_t> out,Tensor<int32_t> in,Tensor<uint32_t> offsets,uint32_t,uint32_t n){
 for(unsigned i=0;i<n;++i)out.SetValue(i,in.GetValue(offsets.GetValue(i)/4));
}
inline void And(Tensor<uint16_t> out,Tensor<uint16_t> a,Tensor<uint16_t> b,int32_t n){
 for(int i=0;i<n;++i)out.SetValue(i,a.GetValue(i)&b.GetValue(i));
}
struct GatherArgs {uint32_t a,b,c,d;};
inline void GatherMask(Tensor<int32_t> out,Tensor<int32_t> in,Tensor<uint32_t> mask,bool,uint32_t n,GatherArgs,uint64_t& count){
 auto bits=mask.ReinterpretCast<uint8_t>();count=0;
 for(unsigned i=0;i<n;++i)if(bits.GetValue(i/8)&(1U<<(i%8)))out.SetValue(count++,in.GetValue(i));
}
template<class T> void Duplicate(Tensor<T>,T,uint64_t*,uint8_t,uint32_t,uint32_t){}
}
''', encoding='utf-8')
    run_cpp(tmp_path, r'''
#include "kernels/combine/low_latency/combine_server_dedup_select.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#include "kernels/dispatch/low_latency/common/ll_rdma_layout.h"
using namespace AscendC;
using namespace CombineDedup;
struct Args {uint32_t rank,rankSize;};
namespace DispatchDedup { constexpr uint32_t DISPATCH_DEDUP_DATA_TYPE_FP16=3; }
namespace LlTransport { bool RemoteWrite(Args* a,uint32_t peer){return a->rank/16!=peer/16;} }
struct Fixture {
 static constexpr uint32_t Group=16;
 Args a; Args* args_=&a; uint32_t recvCores_,tokens_,scanRows_=64,inputLanes_=1,outputLanes_=1;
 uint32_t blocks_=1,packetBytes_=512,k_=8,dtype_=0,row_=0;
 std::vector<uint64_t> memory;
 std::vector<int32_t> rows,types,triples,t2,t3;
 std::vector<uint8_t> output;
 Tensor<uint8_t> scan_,outputs_;
 Tensor<int32_t> rows_,types_,table2_,table2Ub_,table3Ub_;
 std::vector<unsigned> visits;
 Fixture(uint32_t r,uint32_t w,uint32_t cores,std::vector<int32_t> ty,std::vector<int32_t> table):
 a{r,w},recvCores_(cores/2),tokens_(ty.size()),memory(4096),rows(EpCombineM4TaskBytes(tokens_,recvCores_)/4),
 types(std::move(ty)),triples(std::move(table)),t2(3),t3(1),output(512){
 scan_={reinterpret_cast<uint8_t*>(memory.data()),memory.size()*8};outputs_={output.data(),output.size()};
 rows_={rows.data(),rows.size()};types_={types.data(),types.size()};table2_={triples.data(),triples.size()};
 table2Ub_={t2.data(),t2.size()};table3Ub_={t3.data(),t3.size()};
 }
 template<HardEvent E> void Sync(){SetFlag<E>(0);WaitFlag<E>(0);}
 void LoadTask(uint32_t row){row_=row;for(unsigned j=0;j<3;++j)t2[j]=triples[row*3+j];t3[0]=1;}
 template<class T> void Aggregate(uint32_t,uint32_t,uint32_t){}
 void Send(uint32_t,uint32_t source,uint32_t,uint32_t){
 if(LlTransport::RemoteWrite(args_,source))assert(hardwareCore-recvCores_<recvCores_);
 visits.push_back(row_);
 }
''' + process + '\n' + select + r'''
};
struct Direct {
 static constexpr uint32_t Group=16;
 Args a;Args* args_=&a;uint32_t core_,sendCores_;
 std::vector<int32_t> ranks,selected;
 Tensor<int32_t> indexUb_;
 std::vector<unsigned> visits;
 Direct(uint32_t r,uint32_t w,uint32_t core,uint32_t cores,std::vector<int32_t> src):a{r,w},core_(core),sendCores_(cores/2),ranks(std::move(src)){}
 uint32_t SelectDirectServer(uint32_t server){selected.clear();for(unsigned i=0;i<ranks.size();++i)if(uint32_t(ranks[i])/16==server)selected.push_back(i);
 indexUb_={selected.data(),selected.size()};return selected.size();}
 void Process(){const uint32_t servers=a.rankSize/16,ownServer=a.rank/16;uint32_t owner=0;
''' + direct_loop + r'''
 }
};
int main(){
 for(unsigned cores:{16U,56U,72U})for(unsigned world:{32U,64U})for(unsigned n:{0U,1U,33U,257U,10000U})for(unsigned hot:{0U,1U}){
  std::vector<int32_t> types(n),triples(n*3),ranks(n);
  for(unsigned i=0;i<n;++i){types[i]=(hot||i%3)?2:1;ranks[i]=hot?17:int((i*13)%world);triples[i*3]=ranks[i];triples[i*3+1]=i;triples[i*3+2]=0;}
  std::vector<unsigned> partialSeen(n),directSeen(n);
  for(unsigned c=0;c<cores/2;++c){
   hardwareCore=cores/2+c;Fixture f(0,world,cores,types,triples);assert(f.Process()==f.visits.size());
   for(auto row:f.visits){assert(types[row]==2);++partialSeen[row];if(ranks[row]>=16)assert(c<cores/2);}
   Direct d(0,world,c,cores,ranks);d.Process();for(auto row:d.visits){++directSeen[row];if(ranks[row]>=16)assert(c<cores/2);}
   for(auto& eventsForType:events)for(bool live:eventsForType)assert(!live);
  }
  for(unsigned i=0;i<n;++i){assert(partialSeen[i]==unsigned(types[i]==2));assert(directSeen[i]==1);}
 }
}
''', 'combine-per-core-qp')
