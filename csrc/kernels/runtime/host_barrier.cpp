#include "kernel_operator.h"
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "../launch.hpp"
#include "host_barrier.h"

ACLSHMEM_GLOBAL void deepep_ndr_host_barrier_kernel(uint64_t ffts_config)
{
    AscendC::SetSyncBaseAddr(ffts_config);
    deepep_host_barrier(ACLSHMEM_TEAM_WORLD);
}

extern "C" DEEPEP_EXPORT void deepep_ndr_host_barrier_launch(void* stream)
{
    deepep_ndr_host_barrier_kernel<<<1, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(util_get_ffts_config());
}
