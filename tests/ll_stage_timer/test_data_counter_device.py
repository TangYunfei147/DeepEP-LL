"""Schema 3 submitted-DMA counters: host model and source wiring contract."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "csrc/kernels/dispatch/low_latency/common"

def test_counter_source_contract():
    layout = (COMMON / "ll_stage_timer_layout.h").read_text()
    device = (COMMON / "ll_stage_timer_device.h").read_text()
    assert "kSchema = 4U" in layout
    assert "kDataCounterOffset = 64U" in layout
    assert "kDataFlagsOffset = 84U" in layout
    assert "#if defined(__CCE__) && defined(__CCE_IS_AICORE__)\n__aicore__ inline" in layout
    body = device.split("inline void CountData(", 1)[1].split("inline void Finish", 1)[0]
    assert "#ifdef DEEPEP_LL_STAGE_TIMER" in body and "if (!address_) return" in body
    assert "PipeBarrier" not in body and "Sync<" not in body
    kernel = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text()
    partial = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text()
    assert "CountData(uint32_t(args_->rank), source, resultStride_, false)" in kernel
    assert "workspace_, timer_)" in kernel
    assert "CountData(uint32_t(args_->rank), memberRank, h_ * 2U, false, true)" in partial
    assert "CountData(uint32_t(args_->rank), uint32_t(args_->rank), h_ * 2U, false, true)" in partial
    assert "CountData(uint32_t(args_->rank), source, packetBytes_, true)" in partial
    assert "CountData" not in kernel.split("inline void CopyLocalPendingTokens()", 1)[1].split("inline void", 1)[0]

def test_dispatch_counts_first_and_relay_once_at_issue():
    sender = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h").read_text()
    receiver = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h").read_text()
    assert sender.count("timer_->CountData(") == 1
    assert receiver.count("timer_->CountData(") == 1
    assert sender.index("timer_->CountData(") < sender.index("LlTransport::Packet(")
    assert receiver.index("timer_->CountData(") < receiver.index("LlTransport::Packet(")
    assert "CountData(uint32_t(args_->rank), destination," in sender
    assert "uint64_t(data_.blockCount) * 32U" in sender
    assert "CountData(uint32_t(args_->rank), destinationRank," in receiver
    assert "uint64_t(data_.tokenStride), true)" in receiver


@pytest.mark.parametrize("enabled,no_comm", [(False, False), (True, False), (True, True)])
def test_counter_host_model(tmp_path, enabled, no_comm):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable; source contract only")
    (tmp_path / "kernel_operator.h").write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline uint32_t GetBlockIdx() { return 0; }
inline uint32_t GetBlockNum() { return 16; }
inline uint64_t GetSystemCycle() { return 42; }
enum Pipe { PIPE_ALL };
inline unsigned barriers=0;
template<Pipe> void PipeBarrier() { ++barriers; }
template<class T> struct GlobalTensor {
 T* ptr=nullptr;
 void SetGlobalBuffer(T* p) { ptr=p; }
 T& operator()(unsigned i) { return ptr[i]; }
};
}
using AscendC::PIPE_ALL;
''')
    source = tmp_path / "counter.cpp"
    source.write_text(r'''
#include "ll_stage_timer_device.h"
#include <cassert>
using namespace LlStageTimer;
int main() {
 static_assert(kWordsPerCore*8%512==0);
 static_assert(DataBucket(0,0,false)==0 && DataBucket(0,0,true)==0);
 static_assert(DataBucket(0,15,false)==1 && DataBucket(0,15,true)==2);
 static_assert(DataBucket(0,16,false)==3 && DataBucket(0,16,true)==4);
 static_assert(DataBucket(15,16,false)==3 && DataBucket(16,31,false)==1);
 static uint64_t data[kWordsPerCore]={};
 DeviceTimer timer;
 timer.Init(reinterpret_cast<uint8_t*>(data),2,2);
 timer.CountData(0,0,64,false,true);
 timer.CountData(0,1,512,false);
 timer.CountData(0,1,128,false,true);
 timer.CountData(0,2,1024,true);
 timer.CountData(0,16,1536,false);
 timer.CountData(0,17,2048,true,true);
 assert(AscendC::barriers==0);
#ifdef DEEPEP_LL_STAGE_TIMER
 assert(data[64]==64 && data[65]==1 && data[66]==64 && data[67]==1);
 assert(data[68]==640 && data[69]==2 && data[70]==128 && data[71]==1);
 assert(data[72]==1024 && data[73]==1 && data[74]==0 && data[75]==0);
 assert(data[76]==1536 && data[77]==1);
 assert(data[80]==2048 && data[81]==1 && data[82]==2048 && data[83]==1);
 #ifdef DEEPEP_LL_NO_COMM
 assert(data[84]==1);
 #else
 assert(data[84]==0);
 #endif
#else
 for (auto v:data) assert(v==0);
#endif
 timer.Init(nullptr,2,2);
 timer.CountData(0,1,999,false);
}
''')
    exe = tmp_path / "counter.exe"
    command = [compiler,"-std=c++17","-I"+str(tmp_path),"-I"+str(COMMON),str(source),"-o",str(exe)]
    if enabled: command.append("-DDEEPEP_LL_STAGE_TIMER=1")
    if no_comm: command.append("-DDEEPEP_LL_NO_COMM=1")
    subprocess.run(command, check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
