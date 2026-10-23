#include "../../csrc/kernels/dispatch/low_latency/common/ll_core_config.h"
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
        if (mode != 1) throw std::runtime_error("invalid mode");
        const char* visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (!visible || std::string(visible) != "0,1") throw std::runtime_error("A5 validation requires visible 0,1");
        check(aclInit(nullptr));
        check(aclrtSetDevice(0));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        const size_t words = DispatchDedup::kLlActiveCoreNum * LlStageTimer::kWordsPerCore, guard = 64;
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
            for (size_t c = 0; c < DispatchDedup::kLlActiveCoreNum; ++c) {
                const auto* row = data.data() + c * LlStageTimer::kWordsPerCore;
                if (row[0] != LlStageTimer::kMagic || row[1] != LlStageTimer::kSchema || row[4] != c || row[6] == 0)
                    throw std::runtime_error("header mismatch");
                using namespace LlStageTimer;
                const uint64_t expected=(1ULL<<ENTRY)|(1ULL<<INIT_DONE)|(1ULL<<FIRST_RECV_PROBE)|
                    (1ULL<<FIRST_RECV_READY)|(1ULL<<RECV_DONE)|(1ULL<<WORK_DONE);
                if (row[6]!=expected || row[7]!=0U || ((row[5]>>24)&255)!=6U)
                    throw std::runtime_error("sparse point mask/profile mismatch");
                uint64_t previous=0;
                for(uint32_t point=0;point<kPointCount;++point) {
                    const uint64_t tick=row[kRecordWordOffset+point];
                    if(expected&(1ULL<<point)) {
                        if(tick<previous)throw std::runtime_error("non-monotonic sparse point");
                        previous=tick;
                    } else if(tick) throw std::runtime_error("unvisited point must be zero");
                }
            }
        std::ofstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot create output");
        file << "[";
        for (size_t c = 0; c < DispatchDedup::kLlActiveCoreNum; ++c) {
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
        std::puts("PASS timer device0: per-core sparse points, first-hit mask and guard");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
