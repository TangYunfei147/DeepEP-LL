// Execute the actual DeepEP helper against CPU memory/transport doubles.
// This checks algorithm/address ownership, not CANN codegen or RDMA hardware.
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#define ACLSHMEM_DEVICE inline
#define __gm__
#define __ubuf__
#define ASCEND_IS_AIV (true)
using aclshmem_team_t = int;
constexpr int ACLSHMEM_TEAM_WORLD = 0, ACLSHMEM_TRANSPORT_MTE = 1, ACLSHMEM_TRANSPORT_ROCE = 2;
constexpr int ACLSHMEM_SYNCBIT_SIZE = 64, SYNC_ARRAY_SIZE = 64 * 16;
using Bit = int32_t[16];
struct Team { int size, mype; int pe_mapping[256]; };
struct State {
    Team* team_pools[1];
    unsigned topo_list[256];
    struct { uint64_t aclshmem_ub = 0; uint32_t sync_id = 0; } rdma_config;
};
struct Memory { Bit slots[16]{}; Bit counter{}; };
struct Pending { int32_t* dst; int32_t* src; };
std::array<Memory, 256> memory;
std::array<Team, 256> teams;
std::array<State, 256> states;
std::mutex lock;
thread_local int rank_id;
thread_local std::vector<Pending> pending;
std::atomic<int> puts{0}, stores{0}, fallback{0};

int owner(const void* p) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    for (int r = 0; r < 256; ++r) {
        const auto begin = reinterpret_cast<uintptr_t>(&memory[r]);
        if (address >= begin && address < begin + sizeof(Memory)) return r;
    }
    throw std::runtime_error("unknown symmetric address");
}
void* translated(void* p, int peer) {
    assert(owner(p) == rank_id);
    const auto offset = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(&memory[rank_id]);
    return reinterpret_cast<char*>(&memory[peer]) + offset;
}
void* aclshmem_ptr(void* p, int peer) {
    // Direct Scalar access to a ROCE peer is illegal in this fixture.
    assert(states[rank_id].topo_list[peer] & ACLSHMEM_TRANSPORT_MTE);
    return translated(p, peer);
}
namespace AscendC {
enum class CacheMode { CACHE_MODE_DISABLE };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
inline int GetBlockIdx() { return 0; }
inline void Trap() { throw std::runtime_error("device trap"); }
template<class T> struct GlobalTensor {
    T* p = nullptr;
    bool bypass = false;
    void SetGlobalBuffer(T* value) { p = value; }
    void SetL2CacheHint(CacheMode) { bypass = true; }
    void SetValue(int i, T value) {
        assert(bypass && (states[rank_id].topo_list[owner(p)] & ACLSHMEM_TRANSPORT_MTE));
        std::lock_guard<std::mutex> guard(lock); p[i] = value; ++stores;
    }
    T GetValue(int i) {
        assert(bypass && owner(p) == rank_id);  // all polling must be local
        std::this_thread::yield();
        std::lock_guard<std::mutex> guard(lock); return p[i];
    }
};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>&) {}
}
inline State* aclshmemi_get_state() { return &states[rank_id]; }
inline Bit* aclshmemi_get_team_sync_pool(int) { return memory[rank_id].slots; }
inline Bit* aclshmemi_get_team_sync_counter(int) { return &memory[rank_id].counter; }
inline void dcci_cacheline(uint8_t*) {}
inline int32_t aclshmemi_load(int32_t* p) { assert(owner(p) == rank_id); return *p; }
template<bool> void aclshmemi_sync_core() {}
template<bool> void aclshmemi_sync(int) { ++fallback; }
struct Qps { unsigned qp_num = 3; } qps;
inline Qps* aclshmemi_qp_info_fetch() { return &qps; }
inline void aclshmemx_roce_qp_put_nbi(uint8_t* dst, uint8_t* src, uint8_t*, unsigned n, int peer, unsigned qp, unsigned) {
    assert(n == 4 && qp == 0 && owner(src) == rank_id);
    assert(!(states[rank_id].topo_list[peer] & ACLSHMEM_TRANSPORT_MTE));
    pending.push_back({static_cast<int32_t*>(translated(dst, peer)), reinterpret_cast<int32_t*>(src)});
    ++puts;
}
inline void aclshmemx_roce_qp_quiet(int, unsigned, uint8_t*, unsigned) {
    std::this_thread::yield();
    std::lock_guard<std::mutex> guard(lock);
    for (auto item : pending) *item.dst = *item.src;  // read source only at completion
    pending.clear();
}

#include "csrc/kernels/runtime/host_barrier.h"

int main() {
    for (int size : {2, 17, 32, 64}) {
        memory = {};
        const int per_host = size == 2 ? 1 : 16;
        std::array<std::atomic<int>, 256> entered{};
        for (int r = 0; r < size; ++r) {
            teams[r].size = size; teams[r].mype = r;
            states[r].team_pools[0] = &teams[r];
            for (int p = 0; p < size; ++p) {
                teams[r].pe_mapping[p] = size - 1 - p;
                states[r].topo_list[p] = r / per_host == p / per_host ? ACLSHMEM_TRANSPORT_MTE : ACLSHMEM_TRANSPORT_ROCE;
            }
        }
        std::vector<std::thread> workers;
        for (int r = 0; r < size; ++r) workers.emplace_back([&, r] {
            rank_id = r;
            for (int generation = 1; generation <= 8; ++generation) {
                entered[r] = generation;
                deepep_host_barrier(0);
                for (int p = 0; p < size; ++p) assert(entered[p] >= generation);
                assert(pending.empty());
            }
        });
        for (auto& worker : workers) worker.join();
    }
    assert(puts > 0 && stores > 0 && fallback == 0);
    rank_id = 0;
    teams[0].size = 16;
    for (int p = 0; p < 16; ++p) { teams[0].pe_mapping[p] = p; states[0].topo_list[p] = ACLSHMEM_TRANSPORT_MTE; }
    deepep_host_barrier(0);
    assert(fallback == 1);
}
