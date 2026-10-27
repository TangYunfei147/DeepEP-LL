// 单卡启动 route_q_bench_kernel，扫核数，输出 CSV。
// 只依赖 ACL runtime；不需要 shmem、catlass。
#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void ll_route_q_bench_launch(uint32_t cores, void* stream, void* ids, void* routes, void* counts,
                                        void* cycles, uint32_t routeItems, uint32_t experts, uint32_t stride,
                                        uint32_t mode, uint32_t reps);

namespace {
void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(what) + " status=" + std::to_string(status));
}

struct Options {
    int device = 0;
    uint32_t tokens = 256;   // --num-max-tokens-per-rank
    uint32_t topK = 8;       // --num-topk
    uint32_t experts = 384;  // --num-experts
    uint32_t stride = 32;    // 每专家表容量（个 int32）
    uint32_t reps = 100;
    std::string cores;       // 逗号分隔；空表示默认扫描
    std::string out = "route_q_bench.csv";
};

uint32_t ParseUint(const char* text)
{
    const unsigned long value = std::strtoul(text, nullptr, 0);
    if (value == 0UL || value > 0x7fffffffUL) throw std::invalid_argument(std::string("bad value: ") + text);
    return uint32_t(value);
}

Options Parse(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
            return argv[++i];
        };
        if (key == "--device") opt.device = int(ParseUint(next()));
        else if (key == "--tokens") opt.tokens = ParseUint(next());
        else if (key == "--topk") opt.topK = ParseUint(next());
        else if (key == "--experts") opt.experts = ParseUint(next());
        else if (key == "--stride") opt.stride = ParseUint(next());
        else if (key == "--reps") opt.reps = ParseUint(next());
        else if (key == "--cores") opt.cores = next();
        else if (key == "--out") opt.out = next();
        else throw std::invalid_argument("unknown argument: " + key);
    }
    return opt;
}

std::vector<uint32_t> ParseCores(const std::string& text)
{
    std::vector<uint32_t> values;
    size_t begin = 0U;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string item = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!item.empty()) values.push_back(ParseUint(item.c_str()));
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    return values;
}

// 每个 token 的 topK 个专家互不相同（与真实 MoE 一致），整体近似均匀。
std::vector<int32_t> BuildRoutes(uint32_t tokens, uint32_t topK, uint32_t experts)
{
    std::mt19937 rng(20261027U);
    std::vector<int32_t> ids;
    ids.reserve(uint64_t(tokens) * topK);
    std::vector<uint32_t> used;
    used.reserve(topK);
    for (uint32_t token = 0; token < tokens; ++token) {
        used.clear();
        for (uint32_t k = 0; k < topK; ++k) {
            uint32_t expert = rng() % experts;
            while (std::find(used.begin(), used.end(), expert) != used.end()) expert = (expert + 1U) % experts;
            used.push_back(expert);
            ids.push_back(int32_t(expert));
        }
    }
    return ids;
}

struct Reference {
    std::vector<uint32_t> counts;
    std::vector<std::vector<uint32_t>> lists;
};

Reference BuildReference(const std::vector<int32_t>& ids, uint32_t experts)
{
    Reference ref;
    ref.counts.assign(experts, 0U);
    ref.lists.assign(experts, {});
    for (uint32_t i = 0; i < uint32_t(ids.size()); ++i) {
        ref.lists[uint32_t(ids[i])].push_back(i);
        ++ref.counts[uint32_t(ids[i])];
    }
    return ref;
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = Parse(argc, argv);
        const uint32_t tokens = opt.tokens, topK = opt.topK, experts = opt.experts, stride = opt.stride;
        const uint32_t routeItems = tokens * topK;

        auto ids = BuildRoutes(tokens, topK, experts);
        const Reference ref = BuildReference(ids, experts);
        for (uint32_t expert = 0; expert < experts; ++expert)
            if (ref.counts[expert] > stride)
                throw std::runtime_error("stride too small: expert " + std::to_string(expert) + " needs " +
                                         std::to_string(ref.counts[expert]));

        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(opt.device), "aclrtSetDevice");
        int64_t available = 0;
        Check(aclrtGetDeviceInfo(uint32_t(opt.device), ACL_DEV_ATTR_VECTOR_CORE_NUM, &available), "AIV query");
        if (available <= 0) throw std::runtime_error("no available AIV");
        if (available > 72) available = 72;

        std::vector<uint32_t> cores = opt.cores.empty() ? std::vector<uint32_t>{} : ParseCores(opt.cores);
        if (cores.empty())
            for (uint32_t value = 1U; value <= uint32_t(available); value <<= 1) cores.push_back(value);
        if (cores.empty() || cores.back() != uint32_t(available)) cores.push_back(uint32_t(available));
        for (uint32_t& value : cores) value = std::min<uint32_t>(value, uint32_t(available));
        std::sort(cores.begin(), cores.end());
        cores.erase(std::unique(cores.begin(), cores.end()), cores.end());

        const uint32_t maxCores = cores.back();
        void* idsDev = nullptr;
        void* routeDev = nullptr;
        void* countDev = nullptr;
        void* cycleDev = nullptr;
        Check(aclrtMalloc(&idsDev, uint64_t(routeItems) * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "ids alloc");
        Check(aclrtMalloc(&routeDev, uint64_t(experts) * stride * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST),
              "route alloc");
        Check(aclrtMalloc(&countDev, uint64_t(experts) * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "count alloc");
        Check(aclrtMalloc(&cycleDev, uint64_t(maxCores) * 4U * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
              "cycle alloc");
        Check(aclrtMemcpy(idsDev, uint64_t(routeItems) * sizeof(int32_t), ids.data(),
                          uint64_t(routeItems) * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE),
              "ids upload");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "stream");

        std::vector<int32_t> routeHost(uint64_t(experts) * stride);
        std::vector<int32_t> countHost(experts);
        std::vector<uint64_t> cycleHost(uint64_t(maxCores) * 4U);

        std::ofstream csv(opt.out);
        if (!csv) throw std::runtime_error("cannot write " + opt.out);
        csv << "mode,cores,reps,routes,experts,us_per_rep,us_total\n";

        std::cout << "shape: routes=" << routeItems << " experts=" << experts << " stride=" << stride
                  << " reps=" << opt.reps << " available_aiv=" << available << "\n";
        // 预热一次，避免首启开销落进任一配置。
        ll_route_q_bench_launch(1U, stream, idsDev, routeDev, countDev, cycleDev, routeItems, experts, stride, 1U, 1U);
        Check(aclrtSynchronizeStream(stream), "warmup");

        for (uint32_t mode = 0U; mode <= 1U; ++mode) {
            for (uint32_t useCores : cores) {
                Check(aclrtMemset(routeDev, uint64_t(experts) * stride * sizeof(int32_t), 0xff,
                                  uint64_t(experts) * stride * sizeof(int32_t)), "clear route");
                Check(aclrtMemset(countDev, uint64_t(experts) * sizeof(int32_t), 0xff,
                                  uint64_t(experts) * sizeof(int32_t)), "clear count");
                ll_route_q_bench_launch(useCores, stream, idsDev, routeDev, countDev, cycleDev, routeItems, experts,
                                        stride, mode, opt.reps);
                Check(aclrtSynchronizeStream(stream), "bench kernel");
                Check(aclrtMemcpy(routeHost.data(), routeHost.size() * sizeof(int32_t), routeDev,
                                  routeHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
                      "route download");
                Check(aclrtMemcpy(countHost.data(), countHost.size() * sizeof(int32_t), countDev,
                                  countHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
                      "count download");
                Check(aclrtMemcpy(cycleHost.data(), uint64_t(useCores) * 4U * sizeof(uint64_t), cycleDev,
                                  uint64_t(useCores) * 4U * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                      "cycle download");

                for (uint32_t expert = 0; expert < experts; ++expert) {
                    if (uint32_t(countHost[expert]) != ref.counts[expert])
                        throw std::runtime_error("count mismatch: mode=" + std::to_string(mode) + " cores=" +
                                                 std::to_string(useCores) + " expert=" + std::to_string(expert));
                    for (uint32_t item = 0; item < ref.counts[expert]; ++item)
                        if (uint32_t(routeHost[uint64_t(expert) * stride + item]) != ref.lists[expert][item])
                            throw std::runtime_error("route mismatch: mode=" + std::to_string(mode) + " cores=" +
                                                     std::to_string(useCores) + " expert=" + std::to_string(expert));
                }

                // 每核只写 word0，因此按 4 个 uint64 的步长取本核周期数再取最大。
                uint64_t slowest = 0U;
                for (uint32_t core = 0U; core < useCores; ++core)
                    slowest = std::max(slowest, cycleHost[uint64_t(core) * 4U]);
                const double totalUs = double(slowest) / 1000.0;  // 系统计数器：1000 ticks/us
                const double perRepUs = totalUs / double(opt.reps);
                csv << mode << ',' << useCores << ',' << opt.reps << ',' << routeItems << ',' << experts << ','
                    << perRepUs << ',' << totalUs << '\n';
                csv.flush();
                std::cout << (mode == 0U ? "scalar" : "vf    ") << " cores=" << useCores
                          << " us/rep=" << perRepUs << "\n";
            }
        }

        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtFree(cycleDev), "free cycles");
        Check(aclrtFree(countDev), "free counts");
        Check(aclrtFree(routeDev), "free routes");
        Check(aclrtFree(idsDev), "free ids");
        Check(aclrtResetDevice(opt.device), "reset device");
        Check(aclFinalize(), "aclFinalize");
        std::cout << "CSV: " << opt.out << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
}
