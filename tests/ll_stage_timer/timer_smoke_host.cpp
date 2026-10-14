#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h"
extern "C" void ll_timer_smoke_do(void*, uint8_t*, uint32_t);
static void check(int code)
{
    if (code) throw std::runtime_error("ACL error " + std::to_string(code));
}
int main(int argc, char** argv)
{
    try {
        if (argc != 4) throw std::runtime_error("output-json mode enabled");
        const auto mode = std::stoul(argv[2]);
        const bool enabled = std::stoul(argv[3]) != 0;
        if (mode != 1 && mode != 2) throw std::runtime_error("invalid mode");
        const char* visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (!visible || std::string(visible) != "0,1") throw std::runtime_error("A5 validation requires visible 0,1");
        check(aclInit(nullptr));
        check(aclrtSetDevice(0));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        const size_t words = 16 * LlStageTimer::kWordsPerCore, guard = 64;
        std::vector<uint64_t> data(words + guard, 0x12345678ULL);
        void* output;
        check(aclrtMalloc(&output, data.size() * 8, ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMemcpy(output, data.size() * 8, data.data(), data.size() * 8, ACL_MEMCPY_HOST_TO_DEVICE));
        check(aclrtMemset(output, words * 8, 0, words * 8));
        ll_timer_smoke_do(stream, static_cast<uint8_t*>(output), mode);
        check(aclrtSynchronizeStream(stream));
        check(aclrtMemcpy(data.data(), data.size() * 8, output, data.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST));
        for (size_t i = words; i < data.size(); ++i)
            if (data[i] != 0x12345678ULL) throw std::runtime_error("guard modified");
        if (!enabled)
            for (size_t i = 0; i < words; ++i)
                if (data[i]) throw std::runtime_error("disabled timer wrote output");
        if (enabled)
            for (size_t c = 0; c < 16; ++c) {
                const auto* row = data.data() + c * LlStageTimer::kWordsPerCore;
                if (row[0] != LlStageTimer::kMagic || row[4] != c || row[6] != 1 || row[5] != mode)
                    throw std::runtime_error("header mismatch");
                if (row[LlStageTimer::RecvWait * 8 + 3] != 3) throw std::runtime_error("repeat count mismatch");
            }
        std::ofstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot create output");
        file << "[";
        for (size_t c = 0; c < 16; ++c) {
            if (c) file << ",";
            file << "[";
            for (size_t s = 0; s < LlStageTimer::kSlots; ++s) {
                if (s) file << ",";
                file << "[";
                for (size_t f = 0; f < 8; ++f) {
                    if (f) file << ",";
                    file << data[c * LlStageTimer::kWordsPerCore + s * 8 + f];
                }
                file << "]";
            }
            file << "]";
        }
        file << "]";
        check(aclrtFree(output));
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(0));
        check(aclFinalize());
        std::puts("PASS timer device0: per-core records, repeated intervals and guard");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
