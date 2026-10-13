#pragma once
// The shared-library runner borrows Python/Torch's ACL runtime and device.
// The executable path retains its original ownership and validation behavior.
#ifdef LL_PROFILE_BRIDGE
using ProfileCallback = int (*)(int);
extern ProfileCallback ll_profile_callback;
inline void ProfilePhase(int phase) {
    if (!ll_profile_callback || ll_profile_callback(phase))
        throw std::runtime_error("Python profiling callback failed");
}
#define main ll_probe_program
#define PROBE_ACL_INIT() ACL_SUCCESS
#define PROBE_ACL_RESET(device) ACL_SUCCESS
#define PROBE_ACL_FINALIZE() ACL_SUCCESS
#else
inline void ProfilePhase(int) {}
#define PROBE_ACL_INIT() aclInit(nullptr)
#define PROBE_ACL_RESET(device) aclrtResetDevice(device)
#define PROBE_ACL_FINALIZE() aclFinalize()
#endif
