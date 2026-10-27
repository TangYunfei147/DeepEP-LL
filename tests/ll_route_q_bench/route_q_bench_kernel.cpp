// 单卡路线表基准：把 EP32 dispatch 规模（bs*topk 条路由、experts 个专家）的
// “按专家压缩的路由表 + 每专家计数”按专家区间分摊到 N 个 AIV 上，比较
//   mode 0：scalar 逐路由比较
//   mode 1：VF（复用 megamoe 的 RelayCompactRoutesVF，寄存器级 Compares/GatherMask）
// 每个核只写自己那段专家，所以核间无重叠写入，也不需要前缀扫描。
// 计时用 GetSystemCycle（Ascend 950 为系统计数器，1000 ticks/us），只覆盖计算本身。
#include "kernel_operator.h"
#include "mega_moe_route_compact.hpp"

using namespace AscendC;

namespace {
// 与仓库内其他 kernel 一致：用一个固定 event id 做显式流水同步。
template <HardEvent E>
__aicore__ inline void SyncEvent(uint32_t id)
{
    SetFlag<E>(id);
    WaitFlag<E>(id);
}
}  // namespace

// routeAddress：experts * stride 个 int32，每专家一段升序路由下标。
// countAddress：experts 个 int32，每专家实际条数。
// cycleAddress：每核 4 个 uint64，word0 为本次计算的周期数。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_route_q_bench_kernel(
    GM_ADDR idsAddress, GM_ADDR routeAddress, GM_ADDR countAddress, GM_ADDR cycleAddress, uint32_t routes,
    uint32_t experts, uint32_t stride, uint32_t mode, uint32_t reps)
{
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    // 专家区间均分，余数给前面的核；与生产侧计数阶段同一分核方式。
    const uint32_t base = experts / cores, remainder = experts % cores;
    const uint32_t owned = base + (core < remainder ? 1U : 0U);
    const uint32_t begin = core * base + (core < remainder ? core : remainder);

    const uint32_t vec = uint32_t(GetVecLen() / sizeof(int32_t));
    // VF 的 LoadAlign 每次读满一个向量，尾部必须补齐，-1 不会命中任何专家。
    const uint32_t idsItems = (routes + vec - 1U) / vec * vec + vec;

    TPipe pipe;
    TBuf<> idsBuf, tableBuf, countBuf;
    pipe.InitBuffer(idsBuf, idsItems * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(tableBuf, (owned * stride + vec) * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(countBuf, ((owned + 7U) / 8U) * 32U);
    auto ids = idsBuf.Get<int32_t>();
    auto table = tableBuf.Get<int32_t>();
    auto counts = countBuf.Get<int32_t>();

    Duplicate<int32_t>(ids, -1, idsItems);
    SyncEvent<HardEvent::V_MTE2>(0U);
    GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idsAddress));
    DataCopyPad(ids, idsGm, {1U, routes * uint32_t(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0});
    // scalar 分支由 Scalar 读 UB，VF 分支由向量读 UB，两者等待的事件不同。
    if (mode == 0U)
        SyncEvent<HardEvent::MTE2_S>(0U);
    else
        SyncEvent<HardEvent::MTE2_V>(0U);

    const uint64_t start = GetSystemCycle();
    for (uint32_t rep = 0; rep < reps; ++rep) {
        if (mode == 0U) {
            for (uint32_t expert = 0; expert < owned; ++expert) {
                const int32_t target = int32_t(begin + expert);
                const uint32_t slot = expert * stride;
                uint32_t found = 0U;
                for (uint32_t i = 0; i < routes; ++i)
                    if (ids.GetValue(i) == target) table.SetValue(slot + found++, int32_t(i));
                counts.SetValue(expert, int32_t(found));
            }
        } else {
            MegaMoeImpl::RelayCompactRoutesVF(
                reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(table.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()), routes, 8U, int32_t(begin), owned, stride);
        }
    }
    const uint64_t end = GetSystemCycle();

    if (mode == 0U)
        SyncEvent<HardEvent::S_MTE3>(1U);
    else
        SyncEvent<HardEvent::V_MTE3>(1U);
    GlobalTensor<int32_t> routeGm, countGm;
    routeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(routeAddress));
    countGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countAddress));
    if (owned != 0U) {
        DataCopyPad(routeGm[uint64_t(begin) * stride], table,
                    {1U, owned * stride * uint32_t(sizeof(int32_t)), 0U, 0U, 0U});
        DataCopyPad(countGm[begin], counts, {1U, owned * uint32_t(sizeof(int32_t)), 0U, 0U, 0U});
    }
    SyncEvent<HardEvent::MTE3_S>(2U);
    // 计时结果用标量直写，避免再走一次 MTE3。
    reinterpret_cast<__gm__ uint64_t*>(cycleAddress)[uint64_t(core) * 4U] = end - start;
    PipeBarrier<PIPE_ALL>();
}

// Host 侧启动入口；cores 为本次使用的 AIV 数。
extern "C" void ll_route_q_bench_launch(uint32_t cores, void* stream, void* ids, void* routes, void* counts,
                                        void* cycles, uint32_t routeItems, uint32_t experts, uint32_t stride,
                                        uint32_t mode, uint32_t reps)
{
    // Host 侧不要用 (GM_ADDR) 转换：那会把 __gm__ 属性带到 host 类型上并告警。
    ll_route_q_bench_kernel<<<cores, nullptr, stream>>>(static_cast<uint8_t*>(ids), static_cast<uint8_t*>(routes),
                                                        static_cast<uint8_t*>(counts), static_cast<uint8_t*>(cycles),
                                                        routeItems, experts, stride, mode, reps);
}
