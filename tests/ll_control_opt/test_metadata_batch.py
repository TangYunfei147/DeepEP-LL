"""Execute the production staging helper with bounded UB and deferred DMA.

The mock checks layout, original-role snapshots, output ownership and source
lifetime. It cannot establish AscendC instruction support or device speed.
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "csrc/kernels/dispatch/low_latency"

STUB = r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#define __aicore__
namespace AscendC {
enum class HardEvent { S_MTE3 };
inline uint64_t stores = 0, published = 0;
inline uint32_t copies = 0, syncs = 0;
inline std::vector<std::function<void()>> queued;
struct Storage { std::vector<uint8_t> bytes; uint64_t version = 0; };
template<class T> struct LocalTensor {
 std::shared_ptr<Storage> storage; size_t offset = 0;
 T GetValue(size_t i) const {
  T value; assert(offset+(i+1)*sizeof(T)<=storage->bytes.size());
  std::memcpy(&value,storage->bytes.data()+offset+i*sizeof(T),sizeof(T));return value;
 }
 void SetValue(size_t i,T value) {
  assert(offset+(i+1)*sizeof(T)<=storage->bytes.size());storage->version=++stores;
  std::memcpy(storage->bytes.data()+offset+i*sizeof(T),&value,sizeof(T));
 }
 template<class U> LocalTensor<U> ReinterpretCast() const {return {storage,offset};}
 LocalTensor operator[](size_t i) const {return {storage,offset+i*sizeof(T)};}
};
template<int N=0> struct TBuf {
 std::shared_ptr<Storage> storage;
 template<class T> LocalTensor<T> Get(){return {storage,0};}
};
struct TPipe {
 size_t allocated=0;
 template<int N> void InitBuffer(TBuf<N>& b,size_t bytes) {
  assert(bytes && bytes%32==0);allocated+=bytes;
  assert(allocated<=230*1024);b.storage=std::make_shared<Storage>();
  b.storage->bytes.assign(bytes,0xa5);
 }
};
template<class T> struct GlobalTensor {
 T* base;size_t capacity;
 GlobalTensor operator[](size_t i) const {assert(i<=capacity);return {base+i,capacity-i};}
};
template<HardEvent> void SetFlag(uint32_t event) {assert(event==7);}
template<HardEvent> void WaitFlag(uint32_t event) {assert(event==7);published=stores;++syncs;}
struct CopyParams {uint16_t count;uint32_t bytes;uint32_t srcStride,dstStride,reserved;};
template<class T> void DataCopyPad(GlobalTensor<T> out,LocalTensor<T> in,CopyParams p) {
 assert(p.count==1 && p.srcStride==0 && p.dstStride==0 && p.reserved==0);
 assert(in.offset%32==0 && in.offset+p.bytes<=in.storage->bytes.size());
 assert(p.bytes<=out.capacity*sizeof(T) && in.storage->version<=published);
 ++copies;const auto version=in.storage->version;
 queued.push_back([=]() {assert(in.storage->version==version);
  std::memcpy(out.base,in.storage->bytes.data()+in.offset,p.bytes);});
}
inline void Drain() {for(auto& f:queued) f();queued.clear();}
}
'''

SOURCE = r'''
#include "csrc/kernels/dispatch/low_latency/device/dispatch_recv_metadata_batch.h"
#include <algorithm>
#include <array>
#include <random>
#include <string>
using namespace DispatchDedup;
using namespace AscendC;
void CheckLayout() {
 LlOverlapLayout p{0,0,6,6,100000,true};
 for(uint32_t n=1;n<18000;++n) {
  auto b=LlMakeRecvMetadataBatchLayout(n,256,true,p);
  const uint64_t expected=((uint64_t(n)*4+31)/32)*32+((uint64_t(n)*12+31)/32)*32;
  assert(b.enabled==(expected+p.totalBytes<=kLlDispatchUbBytes));
  if(b.enabled) {assert(b.rows==n && b.bytes==expected);
   assert(b.destinationOffset%32==0 && b.destinationOffset>=uint64_t(n)*4);
   assert(b.bytes-b.destinationOffset>=uint64_t(n)*12);}
  assert(p.recvLanes==6 && p.sendLanes==6 && p.totalBytes==100000);
 }
 for(uint32_t n:{1U,35U,36U,37U,38U,39U,256U}) {
  const uint32_t bytes=LlMakeRecvMetadataBatchLayout(n,256,true,p).bytes;
  auto limit=p;limit.totalBytes=kLlDispatchUbBytes-bytes;
  assert(LlMakeRecvMetadataBatchLayout(n,256,true,limit).enabled);
  ++limit.totalBytes;assert(!LlMakeRecvMetadataBatchLayout(n,256,true,limit).enabled);
 }
 assert(LlMakeRecvMetadataBatchLayout(35,256,true,p).bytes==608);
 assert(LlMakeRecvMetadataBatchLayout(39,256,true,p).bytes==640);
 assert(!LlMakeRecvMetadataBatchLayout(UINT32_MAX,256,true,p).enabled);
 for(uint32_t m:{1U,2U,4U,8U,16U,32U})
  assert(!LlMakeRecvMetadataBatchLayout(37,m,true,p).enabled);
 assert(!LlMakeRecvMetadataBatchLayout(37,256,false,p).enabled);
 assert(!LlMakeRecvMetadataBatchLayout(0,256,true,p).enabled);
 p.enabled=false;assert(!LlMakeRecvMetadataBatchLayout(37,256,true,p).enabled);
 p.enabled=true;p.totalBytes=uint64_t(kLlDispatchUbBytes)+1;
 assert(!LlMakeRecvMetadataBatchLayout(37,256,true,p).enabled);
 // All core roles use unchanged production overlap lanes and retained pools.
 auto d=DispatchServerDedupDataLayout(256,3584,8,384,EpServerDedupInputMode::PrequantizedFp8Packs);
 for(uint32_t c=0;c<56;++c) {
  auto native=LlMakeOverlapLayout(d,256,8,384,16,0,c,EpServerDedupInputMode::PrequantizedFp8Packs,true);
  auto b=LlMakeRecvMetadataBatchLayout(39,256,true,native);
  assert(native.enabled && b.enabled && native.totalBytes+b.bytes<=kLlDispatchUbBytes);
  assert(native.recvLanes==6 && native.sendLanes==6);
 }
}
void CheckSnapshots() {
 std::mt19937 rng(502);
 for(uint32_t n:{1U,2U,7U,8U,9U,35U,36U,37U,38U,39U,127U,256U})
 for(uint32_t begin:{0U,1U,7U,8U,101U}) {
  const auto initialCopies=copies, initialSyncs=syncs;
  const auto p=LlOverlapLayout{0,0,6,6,100000,true};
  auto layout=LlMakeRecvMetadataBatchLayout(n,256,true,p);
  TPipe pipe;pipe.allocated=p.totalBytes;ServerDedupRecvMetadataBatch batch;
  batch.Prepare(layout,&pipe);assert(batch.Enabled());
  assert(pipe.allocated==p.totalBytes+layout.bytes);
  TPipe headerPipe;TBuf<> headerBuf;headerPipe.InitBuffer(headerBuf,32);
  auto header=headerBuf.Get<uint32_t>();
  std::vector<uint32_t> order(n);for(uint32_t j=0;j<n;++j)order[j]=j;
  std::shuffle(order.begin(),order.end(),rng);
  const uint32_t guard=0x9e3779b9;
  std::vector<uint32_t> types(begin+n+11,guard),destinations((begin+n+11)*3,guard);
  std::vector<uint32_t> wantTypes=types,wantDestinations=destinations;
  for(uint32_t j:order) {
   const uint32_t role=1+j%3,slot=98432+j;
   header.SetValue(0,j+7);header.SetValue(1,j+97);header.SetValue(2,j%8);header.SetValue(3,role);
   batch.Snapshot(j,role,slot,header);
   wantTypes[begin+j]=role;
   wantDestinations[(begin+j)*3]=role==3?slot:j+7;
   wantDestinations[(begin+j)*3+1]=role==3?UINT32_MAX:j+97;
   wantDestinations[(begin+j)*3+2]=role==3?UINT32_MAX:j%8;
   // Model forwarding and arbitrary subsequent reuse of the packet lane.
   header.SetValue(3,3);header.SetValue(0,UINT32_MAX);header.SetValue(1,0);header.SetValue(2,0);
  }
  batch.Flush({types.data(),types.size()},{destinations.data(),destinations.size()},begin);
  assert(copies-initialCopies==2 && syncs-initialSyncs==1);
  assert(std::all_of(types.begin(),types.end(),[=](uint32_t x){return x==guard;}));
  Drain();assert(types==wantTypes && destinations==wantDestinations);
 }
}
void CheckDisabled() {
 ServerDedupRecvMetadataBatch batch;TPipe pipe;
 batch.Prepare({},&pipe);assert(!batch.Enabled() && pipe.allocated==0);
 std::array<uint32_t,6> output{};
 batch.Flush({output.data(),output.size()},{output.data(),output.size()},0);
 assert(copies==0 && syncs==0 && queued.empty());
}
int main(int argc,char** argv) {
 assert(argc==2);const std::string mode=argv[1];
 if(mode=="layout") CheckLayout();
 else if(mode=="snapshots") CheckSnapshots();
 else if(mode=="disabled") CheckDisabled();else return 1;
}
'''


@pytest.fixture(scope="module")
def metadata_native(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", ""))
    if not command:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            pytest.skip("C++ compiler needed for production metadata batch mock")
        command = [compiler]
    path = tmp_path_factory.mktemp("metadata-batch")
    (path / "kernel_operator.h").write_text(STUB, encoding="utf-8")
    (path / "metadata.cpp").write_text(SOURCE, encoding="utf-8")
    exe = path / ("metadata.exe" if os.name == "nt" else "metadata")
    subprocess.run([*command, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(path),
                    "-I", str(ROOT), str(path / "metadata.cpp"), "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("case", ["layout", "snapshots", "disabled"])
def test_production_metadata_batch(metadata_native, case):
    subprocess.run([str(metadata_native), case], check=True)


def test_metadata_batch_does_not_change_peer_or_global_synchronization():
    source = (KERNEL / "device/dispatch_recv_metadata_batch.h").read_text(encoding="utf-8")
    # The candidate submits only its two local result arrays. The existing Run
    # barrier owns completion; it must not add a per-core/global publication.
    assert source.count("DataCopyPad(") == 2
    for forbidden in ("SyncAll(", "PipeBarrier<", "->Reset(", "peerMems", "relayReadIndex_["):
        assert forbidden not in source
    receive = (KERNEL / "device/dispatch_server_dedup_recv.h").read_text(encoding="utf-8")
    for output in ("tokenType_", "destinationIndex_"):
        assert f"{output}.GetValue" not in receive
    combine = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    assert "types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenType))" in combine
    assert "table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(assistInfoForCombine))" in combine
