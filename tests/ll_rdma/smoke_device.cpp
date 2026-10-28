#include "kernel_operator.h"
#include "shmem.h"
#include "kernels/dispatch/low_latency/device/ll_transport.h"
#include "kernels/common/ep_packet_receive_layout.h"
using namespace AscendC;
namespace {
constexpr uint32_t kSlot = 16384U, kSource = 2U << 20, kControl = 4U << 20;
template<HardEvent E> __aicore__ inline void Sync() { SetFlag<E>(6U); WaitFlag<E>(6U); }
__aicore__ inline uint8_t Value(uint32_t rank, uint32_t core, uint32_t epoch, uint32_t i) {
    return uint8_t((rank * 19U + core * 7U + epoch * 13U + i) % 251U + 1U);
}
}
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void ll_rdma_smoke_kernel(
    GM_ADDR argsAddress, GM_ADDR resultAddress, uint32_t mode, uint32_t epoch, uint32_t bytes) {
    auto a = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(argsAddress);
    util_set_ffts_config(a->fftsVal);
    TPipe pipe;
    TBuf<> packetBuf, readBuf, controlBuf, scratchBuf;
    pipe.InitBuffer(packetBuf,kSlot); pipe.InitBuffer(readBuf,kSlot);
    pipe.InitBuffer(controlBuf,128U); pipe.InitBuffer(scratchBuf,128U);
    auto packet=packetBuf.Get<uint8_t>(), received=readBuf.Get<uint8_t>();
    auto record=controlBuf.Get<uint32_t>(); auto scratch=scratchBuf.Get<uint8_t>();
    const uint32_t core=GetBlockIdx(), rank=uint32_t(a->rank), peer=1U-rank;
    const uint32_t writer=epoch%2U;
    const uint64_t slot=uint64_t(core)*kSlot;
    uint32_t error=0;
    if (mode==1U) {
        // Local peer source is initialized before the barrier; this mode never runs across hosts.
        LlTransport::LoadLocalPeer(a,peer,kSource+slot,received.ReinterpretCast<uint16_t>(),bytes);
        Sync<HardEvent::MTE2_S>();
        for(uint32_t i=0;i<bytes;++i) if(received.GetValue(i)!=Value(peer,core,epoch,i)) error=1U;
    } else if (rank==writer) {
        if (mode==0U) {
            for(uint32_t i=0;i<bytes;++i) packet.SetValue(i,Value(rank,core,epoch,i));
            for(uint32_t b=0;b<bytes;b+=512U)
                for(uint32_t j=0;j<8U;++j) packet.ReinterpretCast<uint32_t>().SetValue((b+480U)/4U+j,0x3f800000U);
            if(epoch%2U) packet.ReinterpretCast<uint32_t>().SetValue(480U/4U,0x3fc00000U);
            Sync<HardEvent::S_MTE3>();
            LlTransport::Packet(a,peer,slot,packet,bytes);
        } else {
            for(uint32_t j=0;j<24U;++j) record.SetValue(j,0U);
            for(uint32_t j=0;j<3U;++j) {
                record.SetValue(j*8U,0x3f800000U); record.SetValue(j*8U+1U,epoch*1000U+core*3U+j);
            }
            Sync<HardEvent::S_MTE3>();
            LlTransport::Records(a,peer,kControl+uint64_t(core)*2048U,record.ReinterpretCast<uint8_t>(),
                3U,512U,mode==2U,scratch);
        }
    } else {
        // Poll while the peer is publishing; checking only after a Host barrier
        // would hide early flag visibility. 60e9 raw cycles is a diagnostic bound.
        const uint64_t started=GetSystemCycle();
        bool ready=false;
        do {
            if (mode==0U) {
                DataCopy(received,LlTransport::Tensor(a->peerMems[rank]+slot,true),bytes);
                Sync<HardEvent::MTE2_S>();
                float sum=0.0f;
                for(uint32_t b=0;b<bytes;b+=512U) for(uint32_t j=0;j<8U;++j)
                    sum+=received.ReinterpretCast<float>().GetValue((b+480U)/4U+j);
                const auto state=DispatchDedup::PacketReceiveLayout::Decode(sum,bytes/512U);
                ready=state!=DispatchDedup::PacketReceiveLayout::State::Pending;
                if(ready && (state==DispatchDedup::PacketReceiveLayout::State::KeepFlags)!=(epoch%2U!=0U)) error=4U;
            } else {
                DataCopyPad(received,LlTransport::Tensor(a->peerMems[rank]+kControl+uint64_t(core)*2048U,true),
                    {3U,32U,480U,0U,0U},{false,0U,0U,uint8_t(0)});
                Sync<HardEvent::MTE2_S>();
                ready=true;
                for(uint32_t j=0;j<3U;++j)
                    if(received.ReinterpretCast<uint32_t>().GetValue(j*8U)!=0x3f800000U) ready=false;
            }
            if(GetSystemCycle()-started>60000000000ULL) { Trap(); return; }
        } while(!ready);
        if(mode==0U) {
            for(uint32_t i=0;i<bytes;++i)
                if(i%512U<480U && received.GetValue(i)!=Value(writer,core,epoch,i)) error=2U;
        } else if(mode==2U) {
            for(uint32_t j=0;j<3U;++j)
                if(received.ReinterpretCast<uint32_t>().GetValue(j*8U+1U)!=epoch*1000U+core*3U+j) error=3U;
        }
    }
    PipeBarrier<PIPE_ALL>();
    record.SetValue(0,error); Sync<HardEvent::S_MTE3>();
    GlobalTensor<uint32_t> result; result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(resultAddress));
    DataCopy(result[core*8U],record,8U); PipeBarrier<PIPE_ALL>();
}
extern "C" void ll_rdma_smoke_launch(uint32_t cores, void* stream, void* args, void* result,
                                      uint32_t mode,uint32_t epoch,uint32_t bytes) {
    ll_rdma_smoke_kernel<<<cores, nullptr, stream>>>((GM_ADDR)args,(GM_ADDR)result,mode,epoch,bytes);
}
