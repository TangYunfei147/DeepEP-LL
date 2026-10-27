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

// UB buffer 按 32B 对齐，保持与本仓其他 kernel 的分配习惯一致。
__aicore__ inline uint32_t Align32(uint64_t bytes)
{
    return uint32_t((bytes + 31U) / 32U * 32U);
}
}  // namespace

// routeAddress：experts * stride 个 int32，每专家一段升序路由下标。
// countAddress：experts 个 int32，每专家实际条数。
// qAddress：routes 个 int32，mode 2 的生产口径 q 表（由 0 核写出，供校验）。
// cycleAddress：每核 reps+1 个 uint64；前 reps 个是每轮周期数，最后一个是防优化校验和。
// 每一轮都把路由表从 GM 重新读入（L2 提示关闭），因此单轮时间包含真实取数。
// mode 0：逐专家扫全表（scalar）；mode 1：逐专家 VF 压缩；mode 2：生产同款单趟直方图 q。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_route_q_bench_kernel(
    GM_ADDR idsAddress, GM_ADDR routeAddress, GM_ADDR countAddress, GM_ADDR cycleAddress, GM_ADDR qAddress,
    uint32_t routes, uint32_t experts, uint32_t topK, uint32_t stride, uint32_t mode, uint32_t reps)
{
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    // 专家区间均分，余数给前面的核；与生产侧计数阶段同一分核方式。
    const uint32_t base = experts / cores, remainder = experts % cores;
    const uint32_t owned = base + (core < remainder ? 1U : 0U);
    const uint32_t begin = core * base + (core < remainder ? core : remainder);

    const uint32_t vec = uint32_t(GetVecLen() / sizeof(int32_t));
    // VF 的 LoadAlign 每次读满一个向量；尾部 padding 由 Host 用 -1 填好，不会命中任何专家。
    const uint32_t idsItems = (routes + vec - 1U) / vec * vec + vec;

    TPipe pipe;
    TBuf<> idsBuf, tableBuf, countBuf, qBuf, histBuf;
    pipe.InitBuffer(idsBuf, idsItems * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(tableBuf, (owned * stride + vec) * uint32_t(sizeof(int32_t)));
    pipe.InitBuffer(countBuf, ((owned + 7U) / 8U) * 32U);
    pipe.InitBuffer(qBuf, Align32(uint64_t(routes) * 4U));
    pipe.InitBuffer(histBuf, Align32(uint64_t(experts) * 4U));
    auto ids = idsBuf.Get<int32_t>();
    auto table = tableBuf.Get<int32_t>();
    auto counts = countBuf.Get<int32_t>();
    auto qTable = qBuf.Get<int32_t>();
    auto hist = histBuf.Get<int32_t>();
    // 标量分支直接走 UB 指针，避免 LocalTensor 访问器每轮的开销污染算法比较。
    __ubuf__ int32_t* idsRaw = reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr());
    __ubuf__ int32_t* tableRaw = reinterpret_cast<__ubuf__ int32_t*>(table.GetPhyAddr());
    __ubuf__ int32_t* countsRaw = reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr());
    __ubuf__ int32_t* qRaw = reinterpret_cast<__ubuf__ int32_t*>(qTable.GetPhyAddr());
    __ubuf__ int32_t* histRaw = reinterpret_cast<__ubuf__ int32_t*>(hist.GetPhyAddr());

    GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idsAddress));
    // 禁用 L2 命中：每一轮都按 HBM 时延重新取 8KB 路由表。
    idsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    // 计时结果按标量直写 GM（与 device profile 的做法一致，避免 64 位 GlobalTensor 下标写入）。
    __gm__ uint64_t* cycleOut = reinterpret_cast<__gm__ uint64_t*>(cycleAddress);
    const uint32_t cycleStride = reps + 1U;
    uint32_t guard = 0U;

    for (uint32_t rep = 0; rep < reps; ++rep) {
        // 上一轮的 UB 读者必须先退出，才允许 MTE2 覆盖同一块 ids。
        if (rep != 0U) {
            if (mode == 1U)
                SyncEvent<HardEvent::V_MTE2>(0U);
            else
                SyncEvent<HardEvent::S_MTE2>(0U);
        }
        const uint64_t repStart = GetSystemCycle();
        DataCopyPad(ids, idsGm, {1U, idsItems * uint32_t(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0});
        // mode 1 由向量读 UB，其余分支由 Scalar 读 UB。
        if (mode == 1U)
            SyncEvent<HardEvent::MTE2_V>(0U);
        else
            SyncEvent<HardEvent::MTE2_S>(0U);
        if (mode == 1U) {
            MegaMoeImpl::RelayCompactRoutesVF(
                reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(table.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()), routes, 8U, int32_t(begin), owned, stride);
            // __simd_vf__ 是异步发射：必须等向量真正完成，否则测到的是发射开销。
            SyncEvent<HardEvent::V_S>(1U);
        } else if (mode == 0U) {
            for (uint32_t expert = 0; expert < owned; ++expert) {
                const int32_t target = int32_t(begin + expert);
                __ubuf__ int32_t* slot = tableRaw + expert * stride;
                uint32_t found = 0U;
                for (uint32_t i = 0; i < routes; ++i)
                    if (idsRaw[i] == target) slot[found++] = int32_t(i);
                countsRaw[expert] = int32_t(found);
            }
        } else {
            // 生产 LlBuildRouteQ 口径：单趟直方图，全表（每个核都算，与 sender Init 一致）。
            for (uint32_t expert = 0; expert < experts; ++expert) histRaw[expert] = 0;
            for (uint32_t i = 0; i < routes; ++i) {
                const uint32_t expert = uint32_t(idsRaw[i]);
                const int32_t previous = histRaw[expert];
                qRaw[i] = i < topK ? 0 : previous;
                histRaw[expert] = previous + 1;
            }
        }
        cycleOut[uint64_t(core) * cycleStride + rep] = GetSystemCycle() - repStart;
        // 循环携带依赖，阻止编译器把整段计算提到循环外。
        if (mode == 2U)
            guard += uint32_t(qRaw[routes - 1U]);
        else if (owned != 0U)
            guard += uint32_t(countsRaw[0]);
    }
    cycleOut[uint64_t(core) * cycleStride + reps] = guard;

    if (mode == 1U)
        SyncEvent<HardEvent::V_MTE3>(1U);
    else
        SyncEvent<HardEvent::S_MTE3>(1U);
    GlobalTensor<int32_t> routeGm, countGm, qGm;
    routeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(routeAddress));
    countGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(countAddress));
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qAddress));
    if (owned != 0U) {
        DataCopyPad(routeGm[uint64_t(begin) * stride], table,
                    {1U, owned * stride * uint32_t(sizeof(int32_t)), 0U, 0U, 0U});
        DataCopyPad(countGm[begin], counts, {1U, owned * uint32_t(sizeof(int32_t)), 0U, 0U, 0U});
    }
    // q 只由 0 核写出，供 Host 校验；不影响计时。
    if (mode == 2U && core == 0U)
        DataCopyPad(qGm, qTable, {1U, routes * uint32_t(sizeof(int32_t)), 0U, 0U, 0U});
    PipeBarrier<PIPE_ALL>();
}

// Host 侧启动入口；cores 为本次使用的 AIV 数。
extern "C" void ll_route_q_bench_launch(uint32_t cores, void* stream, void* ids, void* routes, void* counts,
                                        void* cycles, void* q, uint32_t routeItems, uint32_t experts, uint32_t topK,
                                        uint32_t stride, uint32_t mode, uint32_t reps)
{
    // Host 侧不要用 (GM_ADDR) 转换：那会把 __gm__ 属性带到 host 类型上并告警。
    ll_route_q_bench_kernel<<<cores, nullptr, stream>>>(
        static_cast<uint8_t*>(ids), static_cast<uint8_t*>(routes), static_cast<uint8_t*>(counts),
        static_cast<uint8_t*>(cycles), static_cast<uint8_t*>(q), routeItems, experts, topK, stride, mode, reps);
}
