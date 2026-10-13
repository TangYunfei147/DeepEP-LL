#include "kernel_operator.h"
#include "tiling.h"
using namespace AscendC;
template<uint32_t Scenario>
__aicore__ inline bool HasEdge(uint32_t src,uint32_t dst,uint32_t world) {
    if (src==dst) return false;
    if constexpr(Scenario==1) return src==0 && dst==1;
    if constexpr(Scenario==2) return src!=0 && dst==0;
    if constexpr(Scenario==3) return src==0 && dst!=0;
    return src<world && dst<world;
}
template<uint32_t Scenario>
__aicore__ inline void Run(GM_ADDR src,GM_ADDR peers,GM_ADDR tiling) {
    const auto* td=reinterpret_cast<const volatile __gm__ FourLayerTiling*>(tiling);
    const uint32_t layer=td->layer;
    if(layer==0) return;
    // volatile GM loads are observable even when the tiling-only path returns.
    const uint64_t bytes=td->bytesPerEdge;
    const uint32_t chunk=td->chunkBytes,lanes=td->lanes,rank=td->rank,world=td->world;
    const uint32_t scene=td->scenario;
    if(layer==1) return;
    if(scene!=Scenario) { Trap(); return; }
    TPipe pipe; TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage,chunk*lanes);
    auto buf=storage.Get<uint8_t>();
    GlobalTensor<uint8_t> input; input.SetGlobalBuffer(src);
    const uint32_t core=GetBlockIdx(),cores=GetBlockNum();
    const uint64_t blocks=bytes/32;
    const uint64_t owned=blocks/cores+(core<blocks%cores);
    const uint64_t first=core*(blocks/cores)+(core<blocks%cores?core:blocks%cores);
    for(uint32_t lane=0;lane<lanes;++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
    const auto* addresses=reinterpret_cast<const __gm__ uint64_t*>(peers);
    uint64_t packet=0;
    for(uint32_t peer=0;peer<world;++peer) {
        if(!HasEdge<Scenario>(rank,peer,world)) continue;
        GlobalTensor<uint8_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(addresses[peer]));
        for(uint64_t offset=0;offset<owned*32;offset+=chunk) {
            const uint32_t n=uint32_t(owned*32-offset<chunk?owned*32-offset:chunk);
            const uint32_t lane=uint32_t(packet++%lanes); auto tile=buf[lane*chunk];
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
            DataCopy(tile,input[first*32+offset],n);
            SetFlag<HardEvent::MTE2_MTE3>(lane); WaitFlag<HardEvent::MTE2_MTE3>(lane);
            // A distinct slot for each sender avoids many-to-one write races.
            DataCopy(output[uint64_t(rank)*bytes+first*32+offset],tile,n);
            SetFlag<HardEvent::MTE3_MTE2>(lane);
        }
    }
    for(uint32_t lane=0;lane<lanes;++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
    PipeBarrier<PIPE_ALL>();
}
#define ENTRY(K) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void four_comm_##K(GM_ADDR src,GM_ADDR peers,GM_ADDR tiling) { Run<K>(src,peers,tiling); }
ENTRY(1) ENTRY(2) ENTRY(3) ENTRY(4)
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void four_empty(GM_ADDR src,GM_ADDR peers,GM_ADDR tiling) {}
extern "C" void four_comm_do(uint32_t cores,void* stream,GM_ADDR src,GM_ADDR peers,GM_ADDR tiling,uint32_t key) {
    switch(key) {
    case 0: four_empty<<<cores,nullptr,stream>>>(src,peers,tiling); break;
    case 1: four_comm_1<<<cores,nullptr,stream>>>(src,peers,tiling); break;
    case 2: four_comm_2<<<cores,nullptr,stream>>>(src,peers,tiling); break;
    case 3: four_comm_3<<<cores,nullptr,stream>>>(src,peers,tiling); break;
    case 4: four_comm_4<<<cores,nullptr,stream>>>(src,peers,tiling); break;
    }
}
