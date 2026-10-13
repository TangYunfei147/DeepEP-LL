#include "kernel_operator.h"
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
    const auto overflow=GetCtrlSpr<60,60>();
    SetCtrlSpr<60,60>(0);
    TPipe pipe;TBufPool<TPosition::VECCALC,16> scratch;
    pipe.InitBufPool(scratch,DispatchDedup::kEpServerDedupCountUbBytes);
    DispatchDedup::ServerDedupSender<bfloat16_t,false,false,false> sender;
    sender.Init(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(args),qDiagnostic,ids,weights,masks,sourceWeights,
        td->bs,td->h,td->topK,td->moeExpertNum,&pipe,0,nullptr,true);
    sender.Process(scratch,0);
    PipeBarrier<PIPE_ALL>();
    SetCtrlSpr<60,60>(overflow);
}
extern "C" void META_LAUNCH(void* stream,GM_ADDR args,GM_ADDR ids,GM_ADDR weights,GM_ADDR qDiagnostic,
    GM_ADDR masks,GM_ADDR sourceWeights,GM_ADDR tiling) {
    META_KERNEL<<<32,nullptr,stream>>>(args,ids,weights,qDiagnostic,masks,sourceWeights,tiling);
}
