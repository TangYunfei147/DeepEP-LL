// Standalone AscendC transport probe. dst must be a valid local or imported
// peer-GM mapping owned by the harness; this kernel never resolves peer ranks.
// 中文审计索引：tests/ll_probe/KERNEL_MAP.md。
// 本核由 local_smoke.cpp、peer_probe.cpp、shmem_probe.cpp 共用；地址映射由 Host 完成。
// mode=0 完全空返回；mode=1 包含 UB 分配、填充值和事件收尾；mode=2/3 执行搬运。
// mode=2 的 0x5a 初始化也在本次 kernel 内，不能把整段耗时解释为纯链路传输。
// mode=3 多一次本卡 GM→UB；两者均不包含 LL 路由、ready 轮询、去重或 combine 归约。
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

// mode: 0=empty launch, 1=UB allocation/control only,
//       2=preloaded UB -> destination GM, 3=source GM -> UB -> destination GM.
// Host contract: totalBytes%32==0; chunkBytes%32==0; 0<chunkBytes<=32768;
// 1<=lanes<=4; mode<=3; iterations>=1; allocation length>=totalBytes.
// Each core owns disjoint 32-byte blocks. Every iteration writes the same
// region; remote readback must finish before the next host launch.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_transfer_probe(
    GM_ADDR src, GM_ADDR dst, uint64_t totalBytes, uint32_t chunkBytes,
    uint32_t lanes, uint32_t iterations, uint32_t mode)
{
    if (mode == 0U) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, chunkBytes * lanes);
    auto buffer = storage.Get<uint8_t>();
    // Rank-independent deterministic pattern, checked by receiving harness.
    Duplicate<uint8_t>(buffer, uint8_t(0x5a), chunkBytes * lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        SetFlag<HardEvent::MTE3_V>(lane);
    if (mode == 1U) {
        for (uint32_t lane = 0; lane < lanes; ++lane)
            WaitFlag<HardEvent::MTE3_V>(lane);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    GlobalTensor<uint8_t> input, output;
    input.SetGlobalBuffer(src);
    output.SetGlobalBuffer(dst);
    // 按 32B 块均分给 AIV，余数分给前几个核；各核输出互不重叠。
    // 修改分块或尾块策略时，要同时检查 Host 的对齐约束和接收端 guard 校验。
    const uint64_t blocks = totalBytes / 32U;
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint64_t owned = blocks / cores + (core < blocks % cores ? 1U : 0U);
    const uint64_t first = core * (blocks / cores) +
                          (core < blocks % cores ? core : blocks % cores);
    uint64_t packet = 0;
    for (uint32_t repeat = 0; repeat < iterations; ++repeat) {
        for (uint64_t offset = 0; offset < owned * 32U; offset += chunkBytes) {
            const uint32_t bytes = uint32_t(owned * 32U - offset < chunkBytes
                                              ? owned * 32U - offset : chunkBytes);
            const uint32_t lane = uint32_t(packet++ % lanes);
            auto tile = buffer[lane * chunkBytes];
            // 复用环形 lane 前等待上一包 MTE3 完成，防止覆盖仍被发送引擎读取的 UB。
            // Prior send must complete before a lane can be overwritten.
            WaitFlag<HardEvent::MTE3_V>(lane);
            if (mode == 3U) {
                SetFlag<HardEvent::V_MTE2>(lane);
                WaitFlag<HardEvent::V_MTE2>(lane);
                DataCopy(tile, input[first * 32U + offset], bytes);
                SetFlag<HardEvent::MTE2_MTE3>(lane);
                WaitFlag<HardEvent::MTE2_MTE3>(lane);
            } else {
                SetFlag<HardEvent::V_MTE3>(lane);
                WaitFlag<HardEvent::V_MTE3>(lane);
            }
            // Same transfer primitive as LL SendPacket; dst can be peer GM.
            DataCopy(output[first * 32U + offset], tile, bytes);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
    }
    // 等待所有 lane 的本地 DMA 完成；远端数据是否正确由 Host 在同步后回读验证。
    // 此处没有生产 LL 的接收消费协议，不能将其等同于远端可消费时间。
    for (uint32_t lane = 0; lane < lanes; ++lane)
        WaitFlag<HardEvent::MTE3_V>(lane);
    PipeBarrier<PIPE_ALL>();
}

// Host 启动包装：cores 控制 AIV 数，stream 由调用方提供；profiler 匹配的是
// ll_transfer_probe 设备任务名，而不是这个带 _do 后缀的 Host 函数名。
extern "C" void ll_transfer_probe_do(uint32_t cores, void* stream, GM_ADDR src,
    GM_ADDR dst, uint64_t totalBytes, uint32_t chunkBytes, uint32_t lanes,
    uint32_t iterations, uint32_t mode)
{
    ll_transfer_probe<<<cores, nullptr, stream>>>(src,
        dst, totalBytes, chunkBytes, lanes, iterations, mode);
}
