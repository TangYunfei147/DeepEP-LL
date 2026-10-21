"""Execute actual task builder with deferred vector writes and scalar reads."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_actual_task_tables_and_pending_compaction(tmp_path):
    cc=shutil.which('g++')
    if cc is None:pytest.skip('C++ compiler required on A3')
    (tmp_path/'kernel_operator.h').write_text(r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#define __aicore__
namespace AscendC {
struct Storage { std::vector<uint32_t> words;unsigned size;Storage(unsigned n):words(n+8,0xcafe1234U),size(n){} };
inline std::vector<std::function<void()>> pending;
inline unsigned vectors=0,fences=0;
template<class T> struct LocalTensor {
 std::shared_ptr<Storage> storage;unsigned offset=0;
 T GetValue(unsigned i) const {assert(offset+i<storage->size);return T(storage->words[offset+i]);}
 void SetValue(unsigned i,T value) {assert(offset+i<storage->size);storage->words[offset+i]=uint32_t(value);}
 LocalTensor operator[](unsigned i)const{return {storage,offset+i};}
 template<class U>LocalTensor<U> ReinterpretCast() const {static_assert(sizeof(U)==4);return {storage,offset};}
};
template<int P=0>struct TBuf {std::shared_ptr<Storage> storage;template<class T>LocalTensor<T> Get(){return {storage,0};}};
enum class HardEvent{V_S};
struct TPipe {
 std::vector<std::shared_ptr<Storage>> buffers;
 void InitBuffer(TBuf<>& b,unsigned bytes){assert(bytes%32==0);b.storage=std::make_shared<Storage>(bytes/4);buffers.push_back(b.storage);}
 int FetchEventID(HardEvent){return 0;}
 void Check(){for(auto b:buffers)for(unsigned j=b->size;j<b->words.size();++j)assert(b->words[j]==0xcafe1234U);}
};
template<HardEvent E>void SetFlag(int){}
template<HardEvent E>void WaitFlag(int){++fences;for(auto& f:pending)f();pending.clear();}
template<class T>void CreateVecIndex(LocalTensor<T> dst,const T& first,unsigned count){
 assert(dst.offset%8==0);assert(uint64_t(dst.offset)+count<=dst.storage->size);++vectors;
 pending.push_back([=]()mutable{for(unsigned i=0;i<count;++i)dst.SetValue(i,T(first+i));});
}
}
''')
    source=tmp_path/'tables.cpp'
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h"
using namespace DispatchDedup;using namespace AscendC;
int main(){
 for(unsigned ranks:{16U,32U})for(unsigned e:{32U,384U,1024U})for(unsigned kind=0;kind<4;++kind){
  const unsigned bs=513,rank=ranks-1;
  TPipe input;TBuf<> prefixBuf;input.InitBuffer(prefixBuf,(e+7)/8*32);
  auto prefix=prefixBuf.Get<int32_t>();
  std::vector<unsigned> reference;unsigned total=0;
  for(unsigned record=0;record<e;++record){
   unsigned n=kind==0?0:kind==1?(record%5==0?0:record%17):kind==2?(record==e/2?bs:0):(record%9==0?bs:0);
   unsigned region=((record%ranks+rank)%ranks)*(e/ranks)+record/ranks;
   for(unsigned q=0;q<n;++q)reference.push_back(region*bs+q);
   total+=n;prefix.SetValue(record,int32_t(total));
  }
  for(unsigned core=0;core<56;++core){
   assert(pending.empty());TPipe pipe;ServerDedupReceiveTasks tasks;
   unsigned before=fences;
   tasks.Build(prefix,rank,ranks,e,bs,core,&pipe,56);
   unsigned begin=uint64_t(core)*total/56,end=uint64_t(core+1)*total/56;
   assert(tasks.Begin()==begin && tasks.Count()==end-begin);
   assert(fences-before==(end>begin?1U:0U));assert(pending.empty());
   std::vector<unsigned> remaining;
   for(unsigned j=0;j<tasks.Count();++j){assert(tasks.PendingIndex(j)==j);assert(tasks.Position(j)==reference[begin+j]);remaining.push_back(j);}
   while(!remaining.empty()){
    unsigned p=remaining.size()/2;assert(tasks.PendingIndex(p)==remaining[p]);
    tasks.RemovePending(p);remaining[p]=remaining.back();remaining.pop_back();
    assert(tasks.PendingCount()==remaining.size());
    for(unsigned j=0;j<remaining.size();++j)assert(tasks.PendingIndex(j)==remaining[j]);
   }
   pipe.Check();
  }
 }
 // Record boundaries at every possible word alignment, plus unsigned fallback.
 for(unsigned offset=0;offset<8;++offset)for(unsigned n:{1U,7U,8U,31U,32U,33U,65U,257U})
 for(unsigned first:{0U,123U,0x80000000U}){
  TPipe pipe;TBuf<> buf;pipe.InitBuffer(buf,((offset+n+7)/8)*32);auto dst=buf.Get<uint32_t>();
  LlFillTaskRange(dst,offset,first,n);WaitFlag<HardEvent::V_S>(0);
  for(unsigned i=0;i<n;++i)assert(dst.GetValue(offset+i)==first+i);
  for(unsigned i=0;i<offset;++i)assert(dst.GetValue(i)==0xcafe1234U);
  for(unsigned i=offset+n;i<buf.storage->size;++i)assert(dst.GetValue(i)==0xcafe1234U);
  pipe.Check();
 }
 assert(vectors>0);
}
''')
    exe=tmp_path/'tables'
    subprocess.run([cc,'-std=c++17','-O2',f'-I{tmp_path}',f'-I{ROOT}',str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=60)

def test_binary_search_remains_and_dependency_precedes_consumption():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h').read_text()
    assert 'if (uint32_t(prefix.GetValue(middle)) <= begin_)' in source
    assert 'record = middle + 1U' in source
    assert source.index('CreateVecIndex(')<source.index('WaitFlag<HardEvent::V_S>')<source.index('uint32_t PendingIndex(')
