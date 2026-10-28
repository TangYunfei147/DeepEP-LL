// 单卡路线表基准：生产同款 q 构建（LlBuildRouteQ 语义）按专家区间分核。
//   q[i] = 同专家在 routes[0..i) 中出现的次数；首个 token（i < topK）一律为 0。
//   每个核只处理自己专家区间的路由，因此各核写出的 q 互不重叠，1 核时就是生产侧
//   “每核全表一趟直方图”的那段计算。
// 计时：每核每轮用 GetSystemCycle 采样；路由表每轮以 CACHE_MODE_DISABLE 从 GM 重读，
//   所以单轮时间包含真实取数和同步，不含 Host 侧 launch。
#include "kernel_operator.h"
#include "kernels/dispatch/low_latency/common/ll_sender_route_plan.h"

using namespace AscendC;

namespace {
// 与仓库内其他 kernel 一致：用固定 event id 做显式流水同步。
template <HardEvent E>
__aicore__ inline void SyncEvent(uint32_t id)
{
    SetFlag<E>(id);
    WaitFlag<E>(id);
}

// UB buffer 按 32B 对齐。
__aicore__ inline uint32_t Align32(uint64_t bytes)
{
    return uint32_t((bytes + 31U) / 32U * 32U);
}
}  // namespace

// cycleAddress：每核 reps+1 个 uint64；前 reps 个是每轮周期数，最后一个是防优化校验和。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_route_q_bench_kernel(
    GM_ADDR idsAddress, GM_ADDR qAddress, GM_ADDR cycleAddress, uint32_t routes, uint32_t experts, uint32_t topK,
    uint32_t reps)
{
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    // 专家区间均分，余数给前面的核。
    const uint32_t base = experts / cores, remainder = experts % cores;
    const uint32_t owned = base + (core < remainder ? 1U : 0U);
    const uint32_t expertBegin = core * base + (core < remainder ? core : remainder);
    const uint32_t expertEnd = expertBegin + owned;

    const uint32_t vec = uint32_t(GetVecLen() / sizeof(int32_t));
    // 读取按向量对齐，尾部 padding 由 Host 填 -1；区间判断使其永远不会命中。
    const uint32_t idsItems = (routes + vec - 1U) / vec * vec + vec;

    TPipe pipe;
    TBuf<> idsBuf, qBuf, histBuf;
    pipe.InitBuffer(idsBuf, idsItems * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(qBuf, Align32(uint64_t(routes) * 4U));
    pipe.InitBuffer(histBuf, Align32(uint64_t(owned) * 4U));
    auto ids = idsBuf.Get<int32_t>();
    auto qTable = qBuf.Get<int32_t>();
    auto hist = histBuf.Get<int32_t>();
    __ubuf__ int32_t* idsRaw = reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr());
    __ubuf__ int32_t* qRaw = reinterpret_cast<__ubuf__ int32_t*>(qTable.GetPhyAddr());
    __ubuf__ int32_t* histRaw = reinterpret_cast<__ubuf__ int32_t*>(hist.GetPhyAddr());

    GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idsAddress));
    // 禁用 L2 命中：每一轮都按 HBM 时延重新取路由表。
    idsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    __gm__ uint64_t* cycleOut = reinterpret_cast<__gm__ uint64_t*>(cycleAddress);
    const uint32_t cycleStride = reps + 1U;
    uint32_t guard = 0U;

    for (uint32_t rep = 0; rep < reps; ++rep) {
        // 上一轮的标量读者退出后，MTE2 才能覆盖同一块 ids。
        if (rep != 0U) SyncEvent<HardEvent::S_MTE2>(0U);
        const uint64_t repStart = GetSystemCycle();
        DataCopyPad(ids, idsGm, {1U, idsItems * uint32_t(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0});
        SyncEvent<HardEvent::MTE2_S>(0U);
        for (uint32_t local = 0; local < owned; ++local) histRaw[local] = 0;
        for (uint32_t i = 0; i < routes; ++i) {
            const uint32_t expert = uint32_t(idsRaw[i]);
            if (expert < expertBegin || expert >= expertEnd) continue;
            const uint32_t local = expert - expertBegin;
            const int32_t previous = histRaw[local];
            qRaw[i] = i < topK ? 0 : previous;
            histRaw[local] = previous + 1;
        }
        cycleOut[uint64_t(core) * cycleStride + rep] = GetSystemCycle() - repStart;
        // 循环携带依赖，阻止编译器把计算提到循环外。
        if (owned != 0U) guard += uint32_t(histRaw[0]);
    }
    cycleOut[uint64_t(core) * cycleStride + reps] = guard;

    // 结果按专家归属标量写回，每核写自己的私有区（core * routes），避免多核标量写
    // 同一条 cache line 造成丢写；Host 侧再按“非 -1”合并。不计时。
    __gm__ int32_t* qOut = reinterpret_cast<__gm__ int32_t*>(qAddress) + uint64_t(core) * routes;
    for (uint32_t i = 0; i < routes; ++i) {
        const uint32_t expert = uint32_t(idsRaw[i]);
        if (expert < expertBegin || expert >= expertEnd) continue;
        qOut[i] = qRaw[i];
    }
    // 标量 GM 写必须排空到 DDR，否则 Host 侧 aclrtMemcpy 可能读到旧的 0xff。
    SyncEvent<HardEvent::S_MTE3>(2U);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    PipeBarrier<PIPE_ALL>();
}

// 生产参考：每核都跑一遍 LlBuildRouteQ（与 sender Init 相同，不按专家切分），
// 只由 0 核写回 q。用于 1 核时和算子内打点对齐。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_route_q_prod_kernel(
    GM_ADDR idsAddress, GM_ADDR qAddress, GM_ADDR cycleAddress, uint32_t routes, uint32_t experts, uint32_t topK,
    uint32_t reps)
{
    const uint32_t core = GetBlockIdx();
    const uint32_t vec = uint32_t(GetVecLen() / sizeof(int32_t));
    const uint32_t idsItems = (routes + vec - 1U) / vec * vec + vec;
    const uint32_t halfSlots = Align32(uint64_t(routes) * 4U) / sizeof(uint32_t);

    TPipe pipe;
    TBuf<> idsBuf, qBuf, tableBuf;
    pipe.InitBuffer(idsBuf, idsItems * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(qBuf, Align32(uint64_t(routes) * 4U));
    pipe.InitBuffer(tableBuf, uint32_t(halfSlots) * 2U * sizeof(uint32_t));
    auto ids = idsBuf.Get<int32_t>();
    auto qTable = qBuf.Get<uint32_t>();
    auto scratch = tableBuf.Get<uint32_t>();
    auto first = scratch;
    auto second = scratch[halfSlots];

    GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idsAddress));
    idsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    __gm__ uint64_t* cycleOut = reinterpret_cast<__gm__ uint64_t*>(cycleAddress);
    const uint32_t cycleStride = reps + 1U;
    uint32_t guard = 0U;

    for (uint32_t rep = 0; rep < reps; ++rep) {
        if (rep != 0U) SyncEvent<HardEvent::S_MTE2>(0U);
        const uint64_t repStart = GetSystemCycle();
        DataCopyPad(ids, idsGm, {1U, idsItems * uint32_t(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0});
        SyncEvent<HardEvent::MTE2_S>(0U);
        DispatchDedup::LlBuildRouteQ(ids, routes, topK, experts, first, second, qTable, halfSlots);
        cycleOut[uint64_t(core) * cycleStride + rep] = GetSystemCycle() - repStart;
        guard += qTable.GetValue(routes - 1U);
    }
    cycleOut[uint64_t(core) * cycleStride + reps] = guard;

    SyncEvent<HardEvent::S_MTE3>(2U);
    if (core == 0U) {
        GlobalTensor<uint32_t> qGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(qAddress));
        DataCopyPad(qGm, qTable, {1U, routes * uint32_t(sizeof(uint32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();
}

// Host 侧启动入口；cores 为本次使用的 AIV 数。
extern "C" void ll_route_q_bench_launch(uint32_t cores, void* stream, void* ids, void* q, void* cycles,
                                        uint32_t routes, uint32_t experts, uint32_t topK, uint32_t reps)
{
    // Host 侧不要用 (GM_ADDR) 转换：那会把 __gm__ 属性带到 host 类型上并告警。
    ll_route_q_bench_kernel<<<cores, nullptr, stream>>>(static_cast<uint8_t*>(ids), static_cast<uint8_t*>(q),
                                                        static_cast<uint8_t*>(cycles), routes, experts, topK, reps);
}

extern "C" void ll_route_q_prod_launch(uint32_t cores, void* stream, void* ids, void* q, void* cycles,
                                       uint32_t routes, uint32_t experts, uint32_t topK, uint32_t reps)
{
    ll_route_q_prod_kernel<<<cores, nullptr, stream>>>(static_cast<uint8_t*>(ids), static_cast<uint8_t*>(q),
                                                       static_cast<uint8_t*>(cycles), routes, experts, topK, reps);
}
