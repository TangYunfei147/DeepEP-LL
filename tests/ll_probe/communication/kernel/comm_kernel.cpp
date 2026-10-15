// 四层实验的第1层（空载/tiling）与第3层（最简通信）共用本文件。
// 脚本 profile_probe.py --kind comm → comm_host.cpp → four_comm_do。
// scenario/key 选择模板专门化入口；这是显式分发，不依赖自动生成的算子 TilingKey。
#include "kernel_operator.h"
#include "../tiling.h"
using namespace AscendC;
// 通信图不含自环：1/3由peerMask选择目标，2选择来源，4所有rank互发。
// world=2 时，场景2/3退化为单条边；双卡通过不能证明真实 N 方竞争已验证。
template<uint32_t Scenario>
__aicore__ inline bool HasEdge(uint32_t src,uint32_t dst,uint32_t world,uint32_t fanout,uint32_t peerMask=0) {
    if (src==dst) return false;
    if constexpr(Scenario==1) return src==0 && (peerMask?((peerMask>>dst)&1U):dst==1);
    if constexpr(Scenario==2) return dst==0 && (peerMask?((peerMask>>src)&1U):src!=0);
    if constexpr(Scenario==3) return src==0 && (peerMask?((peerMask>>dst)&1U):(dst>0 && dst<=fanout));
    return src<world && dst<world;
}
template<uint32_t Scenario>
__aicore__ inline void Run(GM_ADDR src,GM_ADDR peers,GM_ADDR tiling) {
    // layer 是实验内部模式号：0=空返回、1=仅读 tiling、3=搬运。
    // Host 的 layer=0 实际选择下面的 four_empty，连 GM tiling 都不读取。
    // volatile 保留 layer=1 的字段读取，避免编译器将 tiling 基线优化成空核。
    const auto* td=reinterpret_cast<const volatile __gm__ FourLayerTiling*>(tiling);
    const uint32_t layer=td->layer;
    if(layer==0) return;
    // volatile GM loads are observable even when the tiling-only path returns.
    const uint64_t bytes=td->bytesPerEdge;
    const uint32_t chunk=td->chunkBytes,lanes=td->lanes,rank=td->rank,world=td->world;
    const uint32_t scene=td->scenario,fanout=td->fanout,peerMask=td->peerMask,abi=td->abiVersion;
    if(abi!=2) { Trap(); return; }
    if(layer==1) return;
    // Host 传入的 tiling 场景与启动 key 必须一致，防止误测另一张通信图。
    if(scene!=Scenario || fanout<1 || fanout>=world) { Trap(); return; }
    TPipe pipe; TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage,chunk*lanes);
    auto buf=storage.Get<uint8_t>();
    GlobalTensor<uint8_t> input; input.SetGlobalBuffer(src);
    const uint32_t core=GetBlockIdx(),cores=GetBlockNum();
    // bytes 是每条边发送量，不是全作业总量；每个 peer 均读取同一段本地输入。
    // 以32B为单位给AIV均分，保证不同核只写各自的字节范围。
    const uint64_t blocks=bytes/32;
    const uint64_t owned=blocks/cores+(core<blocks%cores);
    const uint64_t first=core*(blocks/cores)+(core<blocks%cores?core:blocks%cores);
    for(uint32_t lane=0;lane<lanes;++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
    // peers 是 Host 经 SHMEM 获得的 MTE 可寻址映射表，不能换成未经映射的远端裸指针。
    // 每个接收rank分配 world*bytes 的数据区，发送rank r 独占 [r*bytes,(r+1)*bytes)。
    const auto* addresses=reinterpret_cast<const __gm__ uint64_t*>(peers);
    uint64_t packet=0;
    for(uint32_t peer=0;peer<world;++peer) {
        if(!HasEdge<Scenario>(rank,peer,world,fanout,peerMask)) continue;
        GlobalTensor<uint8_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(addresses[peer]));
        for(uint64_t offset=0;offset<owned*32;offset+=chunk) {
            const uint32_t n=uint32_t(owned*32-offset<chunk?owned*32-offset:chunk);
            const uint32_t lane=uint32_t(packet++%lanes); auto tile=buf[lane*chunk];
            // lane复用依赖：先等前次发送完成，再GM→UB；随后MTE2_MTE3保证数据已就绪。
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
            DataCopy(tile,input[first*32+offset],n);
            SetFlag<HardEvent::MTE2_MTE3>(lane); WaitFlag<HardEvent::MTE2_MTE3>(lane);
            // A distinct slot for each sender avoids many-to-one write races.
            DataCopy(output[uint64_t(rank)*bytes+first*32+offset],tile,n);
            SetFlag<HardEvent::MTE3_MTE2>(lane);
        }
    }
    // 只收尾本地发送流水；接收端guard/内容校验和测试barrier位于Host，不能算作LL协议。
    for(uint32_t lane=0;lane<lanes;++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
    PipeBarrier<PIPE_ALL>();
}
// profiler名称为four_comm_1..4；layer=1和layer=3共享名称，须用报告中的layer区分。
#define ENTRY(K) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void four_comm_##K(GM_ADDR src,GM_ADDR peers,GM_ADDR tiling) { Run<K>(src,peers,tiling); }
ENTRY(1) ENTRY(2) ENTRY(3) ENTRY(4)
// 真正的空载入口：保留相同三个GM入参，不读数据、不分配UB、不做同步。
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
