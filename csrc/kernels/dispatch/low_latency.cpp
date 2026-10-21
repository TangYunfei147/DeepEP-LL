// Host dispatch selection. Each mode is compiled in a separate device TU.
#include <stdexcept>
#include "../launch.hpp"
#include "low_latency/launch.hpp"
#include "low_latency/common/ep_memory_server_dedup_protocol.h"
#if CATLASS_ARCH == 3510
extern "C" void dispatch_server_dedup_launch_prequant_e4m3(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e4m3(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e5m2(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e4m3(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e5m2(void*,const deepep::EpServerDedupKernelArgs*);
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_stage_timer_enabled()
{
#ifdef DEEPEP_LL_STAGE_TIMER
    return 1U;
#else
    return 0U;
#endif
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group()
{
    return DispatchDedup::kLlRanksPerGroup;
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_core_protocol_version()
{
    return DispatchDedup::kLlCoreProtocolVersion;
}

extern "C" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do_v2(
    void* stream,const deepep::EpServerDedupKernelArgs* args,uint32_t variant) {
    if (!args) throw std::invalid_argument("Null LL dispatch arguments");
    switch (variant) {
    case 0: dispatch_server_dedup_launch_prequant_e4m3(stream,args); return;
    case 1: dispatch_server_dedup_launch_fp16(stream,args); return;
    case 2: dispatch_server_dedup_launch_fp16_e4m3(stream,args); return;
    case 3: dispatch_server_dedup_launch_fp16_e5m2(stream,args); return;
    case 4: dispatch_server_dedup_launch_bf16(stream,args); return;
    case 5: dispatch_server_dedup_launch_bf16_e4m3(stream,args); return;
    case 6: dispatch_server_dedup_launch_bf16_e5m2(stream,args); return;
    default: throw std::invalid_argument("Invalid LL dispatch variant");
    }
}
#endif
