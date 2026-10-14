// Single-device correctness smoke. This does not measure peer bandwidth.
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <vector>
extern "C" void ll_transfer_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t);
#define CHECK(call) do { auto rc = (call); if (rc != ACL_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %d\n", #call, int(rc)); return 1; } } while (0)
int main()
{
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    constexpr uint64_t capacity = 1U << 20;
    void *src = nullptr, *dst = nullptr;
    CHECK(aclrtMalloc(&src, capacity, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dst, capacity, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint8_t> input(capacity), output(capacity);
    for (uint64_t i = 0; i < capacity; ++i) input[i] = uint8_t((i * 13U + i / 257U) % 251U);
    CHECK(aclrtMemcpy(src, capacity, input.data(), capacity, ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t mode = 0; mode <= 3; ++mode)
      for (uint32_t cores : {1U, 2U, 8U, 32U})
        for (uint32_t lanes : {1U, 2U, 4U})
          for (uint64_t bytes : {32ULL, 480ULL, 512ULL, 15392ULL, 1048576ULL}) {
            CHECK(aclrtMemset(dst, capacity, 0x33, capacity));
            ll_transfer_probe_do(cores, stream, static_cast<uint8_t*>(src),
                static_cast<uint8_t*>(dst), bytes, 4096U, lanes, 2U, mode);
            CHECK(aclrtSynchronizeStream(stream));
            CHECK(aclrtMemcpy(output.data(), capacity, dst, capacity, ACL_MEMCPY_DEVICE_TO_HOST));
            for (uint64_t i = 0; i < capacity; ++i) {
                const uint8_t expected = i >= bytes || mode < 2 ? uint8_t(0x33)
                    : (mode == 2 ? uint8_t(0x5a) : input[i]);
                if (output[i] != expected) {
                    std::fprintf(stderr, "Mismatch mode=%u cores=%u lanes=%u bytes=%llu offset=%llu actual=%u expected=%u\n",
                        mode, cores, lanes, (unsigned long long)bytes, (unsigned long long)i,
                        unsigned(output[i]), unsigned(expected));
                    return 2;
                }
            }
          }
    CHECK(aclrtFree(dst));
    CHECK(aclrtFree(src));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    std::puts("PASS: 240 local correctness cases; device 0 only; no peer-bandwidth claim.");
}
