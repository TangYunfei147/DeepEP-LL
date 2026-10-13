// 第2层仅提取dispatch Sender的本地路由准备，并非完整LL元数据阶段。
// snapshot头文件由prepare_snapshot.py从生产源码复制并改写；不要直接编辑生成目录。
// 改实验切分边界请改prepare_snapshot.py；改生产路由算法请改csrc中的Sender原文件。
#include "kernel_operator.h"
// 两个编译变体采用不同类名，避免链接器将行为不同的模板实例合并。
// 正确性核four_metadata_check输出逐路由q；计时核four_metadata只输出每核checksum。
#if PROBE_METADATA_DIAGNOSTIC
#define ServerDedupSender FourMetadataDiagnosticSender
#else
#define ServerDedupSender FourMetadataTimingSender
#endif
#include "snapshot/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h"
#include "snapshot/kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
using namespace AscendC;
#if PROBE_METADATA_DIAGNOSTIC
#define META_KERNEL four_metadata_check
#define META_LAUNCH four_metadata_check_do
#else
#define META_KERNEL four_metadata
#define META_LAUNCH four_metadata_do
#endif
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void META_KERNEL(
    GM_ADDR args,GM_ADDR ids,GM_ADDR weights,GM_ADDR qDiagnostic,GM_ADDR masks,GM_ADDR sourceWeights,GM_ADDR tiling)
{
    const auto* td=reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tiling);
    // Production Counts::Init/Finish brackets the sender's integer-bit FP32
    // reductions. Preserve that environment when extracting the sender.
    // q前缀计算使用整数位模式的FP32归约，必须保留生产Counts::Init/Finish对CTRL[60]的约定。
    // 这不是普通FP32求和；遗漏保存/恢复会使元数据提取核与生产语义不同。
    const auto overflow=GetCtrlSpr<60,60>();
    SetCtrlSpr<60,60>(0);
    TPipe pipe;TBufPool<TPosition::VECCALC,16> scratch;
    pipe.InitBufPool(scratch,DispatchDedup::kEpServerDedupCountUbBytes);
    DispatchDedup::ServerDedupSender<bfloat16_t,false,false,false> sender;
    // qDiagnostic借用原x参数地址供观测写回；快照的SendPacket在payload打包/量化/发送前返回。
    // 保留路由加载、q计算、去重分类、DIRECT任务和sourceMask/sourceWeights；不含跨rank计数、
    // packet tail或receiver元数据。checksum的写回/同步也属于本提取核的观测开销。
    sender.Init(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(args),qDiagnostic,ids,weights,masks,sourceWeights,
        td->bs,td->h,td->topK,td->moeExpertNum,&pipe,0,nullptr,true);
    sender.Process(scratch,0);
    PipeBarrier<PIPE_ALL>();
    SetCtrlSpr<60,60>(overflow);
}
extern "C" void META_LAUNCH(void* stream,GM_ADDR args,GM_ADDR ids,GM_ADDR weights,GM_ADDR qDiagnostic,
    GM_ADDR masks,GM_ADDR sourceWeights,GM_ADDR tiling) {
    // 固定32个AIV模拟生产Sender分片；逻辑专家分区数不是实际设备rank数。
    META_KERNEL<<<32,nullptr,stream>>>(args,ids,weights,qDiagnostic,masks,sourceWeights,tiling);
}
