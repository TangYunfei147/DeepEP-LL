// Two-rank probe using the production SHMEM symmetric-address mapping.
// Same executable can run on two local devices or one device on each host.
#include <acl/acl.h>
#include "shmem.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void ll_transfer_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t);
static void check(int rc, const char* label) {
    if (rc) throw std::runtime_error(std::string(label)+": "+std::to_string(rc));
}
#define CHECK(call) check(call,#call)
static uint64_t number(const char* s) {
    if (!*s || *s=='-') throw std::runtime_error("Expected unsigned integer");
    size_t pos=0; auto n=std::stoull(s,&pos);
    if (s[pos]) throw std::runtime_error("Invalid integer");
    return n;
}
#include "../../profiling/host/profile_bridge.h"
int main(int argc,char** argv) {
    int rank=-1;
    try {
        if (argc!=11) throw std::runtime_error("Usage: shmem_probe rank device endpoint bytes cores lanes chunk mode warmup samples");
        auto r=number(argv[1]),dev=number(argv[2]); rank=int(r);
        auto bytes=number(argv[4]),cores=number(argv[5]),lanes=number(argv[6]),
            chunk=number(argv[7]),mode=number(argv[8]),warmup=number(argv[9]),samples=number(argv[10]);
        if (r>1 || dev>1 || !bytes || bytes%32 || bytes>(64ULL<<20) ||
            !cores || cores>32 || !lanes || lanes>4 || !chunk || chunk%32 || chunk>32768 ||
            mode>3 || warmup>10000 || samples>10000) throw std::runtime_error("Invalid probe configuration");
        std::string endpoint=argv[3];
        if (endpoint.empty() || endpoint.size()>=ACLSHMEM_MAX_IP_PORT_LEN) throw std::runtime_error("Invalid endpoint");
        const char* visible=std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (!visible || std::string(visible)!="0,1") throw std::runtime_error("Require ASCEND_RT_VISIBLE_DEVICES=0,1");
        CHECK(PROBE_ACL_INIT()); CHECK(aclrtSetDevice(int(dev)));
        aclrtStream stream; CHECK(aclrtCreateStream(&stream));
        aclshmemx_init_attr_t attr{}; aclshmemx_uniqueid_t uid{};
        attr.my_pe=rank; attr.n_pes=2; attr.local_mem_size=128ULL<<20;
        std::memcpy(attr.ip_port,endpoint.c_str(),endpoint.size()+1);
        attr.comm_args=&uid; attr.option_attr.data_op_engine_type=ACLSHMEM_DATA_OP_MTE;
        attr.option_attr.shm_init_timeout=30; attr.option_attr.shm_create_timeout=30;
        attr.option_attr.control_operation_timeout=30;
        CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT,&attr));
        const uint64_t capacity=bytes+32;
        auto* target=static_cast<uint8_t*>(aclshmem_align(512,capacity));
        if (!target) throw std::runtime_error("Symmetric allocation failed");
        auto* peer=static_cast<uint8_t*>(aclshmem_ptr(target,1));
        if (!peer) throw std::runtime_error("No MTE-addressable rank1 symmetric pointer; no fallback");
        void* src=nullptr; CHECK(aclrtMalloc(&src,capacity,ACL_MEM_MALLOC_HUGE_FIRST));
        std::vector<uint8_t> input(capacity),output(capacity);
        for (uint64_t i=0;i<capacity;++i) input[i]=uint8_t((i*13+i/257)%251);
        CHECK(aclrtMemcpy(src,capacity,input.data(),capacity,ACL_MEMCPY_HOST_TO_DEVICE));
        auto launch=[&] {
            if (rank==0) {
                ll_transfer_probe_do(uint32_t(cores),stream,static_cast<uint8_t*>(src),
                    peer,bytes,uint32_t(chunk),uint32_t(lanes),1,uint32_t(mode));
                CHECK(aclrtSynchronizeStream(stream));
            }
        };
        auto validate=[&] {
            CHECK(aclrtMemset(target,capacity,0x33,capacity));
            aclshmem_barrier_all(); launch(); aclshmem_barrier_all();
            CHECK(aclrtMemcpy(output.data(),capacity,target,capacity,ACL_MEMCPY_DEVICE_TO_HOST));
            for (uint64_t i=0;i<capacity;++i) {
                uint8_t expected=rank==0 || i>=bytes || mode<2 ? 0x33 : (mode==2 ? 0x5a : input[i]);
                if (output[i]!=expected) throw std::runtime_error("Readback mismatch at byte "+std::to_string(i));
            }
            aclshmem_barrier_all();
        };
        validate();
        ProfilePhase(0);
        for (uint64_t i=0;i<warmup;++i) { aclshmem_barrier_all(); launch(); aclshmem_barrier_all(); }
        if (rank==0) std::puts("sample,bytes,cores,lanes,chunk,mode,event_us,host_us");
        ProfilePhase(1);
        for (uint64_t i=0;i<samples;++i) {
            ProfilePhase(2);
            aclrtEvent begin=nullptr,end=nullptr;
            if (rank==0) {
                CHECK(aclrtCreateEventWithFlag(&begin,ACL_EVENT_TIME_LINE));
                CHECK(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE));
            }
            aclshmem_barrier_all();
            if (rank==0) {
                CHECK(aclrtRecordEvent(begin,stream));
                const auto hostBegin=std::chrono::steady_clock::now();
                ll_transfer_probe_do(uint32_t(cores),stream,static_cast<uint8_t*>(src),peer,
                    bytes,uint32_t(chunk),uint32_t(lanes),1,uint32_t(mode));
                CHECK(aclrtRecordEvent(end,stream)); CHECK(aclrtSynchronizeStream(stream));
                const auto hostEnd=std::chrono::steady_clock::now();
                float ms=0; CHECK(aclrtEventElapsedTime(&ms,begin,end));
                if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("Invalid event sample");
                double hostUs=std::chrono::duration<double,std::micro>(hostEnd-hostBegin).count();
                std::printf("%llu,%llu,%llu,%llu,%llu,%llu,%.9g,%.9g\n",
                    (unsigned long long)i,(unsigned long long)bytes,(unsigned long long)cores,
                    (unsigned long long)lanes,(unsigned long long)chunk,(unsigned long long)mode,ms*1000.,hostUs);
                CHECK(aclrtDestroyEvent(end)); CHECK(aclrtDestroyEvent(begin));
            }
            aclshmem_barrier_all();
        }
        ProfilePhase(3);
        validate();
        CHECK(aclrtFree(src)); aclshmem_barrier_all(); aclshmem_free(target);
        CHECK(aclshmem_finalize()); CHECK(aclrtDestroyStream(stream));
        CHECK(PROBE_ACL_RESET(int(dev))); CHECK(PROBE_ACL_FINALIZE());
        std::fprintf(stderr,"PASS: rank=%d pre/post readback, %llu samples, normal cleanup\n",rank,(unsigned long long)samples);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: rank=%d %s\n",rank,e.what()); return 1;
    }
}
