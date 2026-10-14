#include <stdexcept>
#include "profile_bridge.h"
ProfileCallback ll_profile_callback = nullptr;
int ll_probe_program(int, char**);
extern "C" int ll_probe_run(int argc, char** argv, ProfileCallback callback) {
    if (ll_profile_callback) return 2; // one driver per process
    ll_profile_callback = callback;
    int result = ll_probe_program(argc, argv);
    ll_profile_callback = nullptr;
    return result;
}
