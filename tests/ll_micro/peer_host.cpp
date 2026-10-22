// One process, two ACL contexts, explicit peer access, receiver-side readback.
// CLI: bytes cores lanes chunk mode warmup samples. samples=0 is validation only.
#include <acl/acl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
extern "C" void ll_transfer_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t);
static void check(aclError rc, const char* label) {
    if (rc != ACL_SUCCESS) throw std::runtime_error(std::string(label)+": "+std::to_string(rc));
}
#define ACL(call) check(call, #call)
static uint64_t number(const char* value) {
    if (!*value || *value=='-') throw std::runtime_error("Expected unsigned integer");
    size_t end=0; auto n=std::stoull(value,&end);
    if (value[end]) throw std::runtime_error("Invalid integer");
    return n;
}
int main(int argc, char** argv) {
    try {
        if (argc!=9) throw std::runtime_error("Usage: peer_probe bytes cores lanes chunk mode warmup samples direction(tx|rx|relay)");
        uint64_t bytes=number(argv[1]), cores=number(argv[2]), lanes=number(argv[3]),
            chunk=number(argv[4]), mode=number(argv[5]), warmup=number(argv[6]), samples=number(argv[7]);
        if (!bytes || bytes%32 || bytes>(64ULL<<20) || !cores || cores>56 ||
            !lanes || lanes>4 || !chunk || chunk%32 || chunk>32768 || mode>4 ||
            warmup>10000 || samples>10000) throw std::runtime_error("Out-of-range probe argument");
        const std::string direction=argv[8];
        if(direction!="tx" && direction!="rx" && direction!="relay")throw std::runtime_error("invalid direction");
        const char* visible=std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (!visible || std::string(visible)!="0,1") throw std::runtime_error("Require ASCEND_RT_VISIBLE_DEVICES=0,1");
        ACL(aclInit(nullptr));
        aclrtContext contexts[2]; aclrtStream stream;
        void *src=nullptr,*dst=nullptr,*mappedDst=nullptr;
        char ipcKey[128] = {};
        ACL(aclrtSetDevice(1)); ACL(aclrtGetCurrentContext(&contexts[1]));
        const uint64_t capacity=2*bytes+32;
        // Ordinary HBM allocation is not a peer-addressable allocation.
        ACL(aclrtMalloc(&dst,capacity,ACL_MEM_MALLOC_HUGE_FIRST_P2P));
        ACL(aclrtIpcMemGetExportKey(dst,capacity,ipcKey,sizeof(ipcKey),ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT));
        int32_t importPid=int32_t(getpid());
        ACL(aclrtIpcMemSetImportPid(ipcKey,&importPid,1));
        ACL(aclrtSetDevice(0)); ACL(aclrtGetCurrentContext(&contexts[0]));
        int32_t supported=0; ACL(aclrtDeviceCanAccessPeer(&supported,0,1));
        if (!supported) throw std::runtime_error("Peer 0->1 access unsupported; no local-copy fallback");
        ACL(aclrtDeviceEnablePeerAccess(1,0));
        ACL(aclrtIpcMemImportByKey(&mappedDst,ipcKey,ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS));
        ACL(aclrtMalloc(&src,capacity,ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtCreateStream(&stream));
        std::vector<uint8_t> input(capacity),output(capacity);
        for (uint64_t i=0;i<capacity;++i) input[i]=uint8_t((i*13+i/257)%251);
        ACL(aclrtMemcpy(src,capacity,input.data(),capacity,ACL_MEMCPY_HOST_TO_DEVICE));
        auto inputPtr = direction=="tx" ? static_cast<uint8_t*>(src) : static_cast<uint8_t*>(mappedDst);
        auto outputPtr = direction=="rx" ? static_cast<uint8_t*>(src) : static_cast<uint8_t*>(mappedDst)+(direction=="relay" ? bytes : 0);
        auto launch=[&] { ll_transfer_probe_do(uint32_t(cores),stream,
            inputPtr,outputPtr,bytes,
            uint32_t(chunk),uint32_t(lanes),1,uint32_t(mode)); };
        auto validate=[&] {
            ACL(aclrtSetCurrentContext(contexts[1]));
            ACL(aclrtMemset(dst,capacity,0x33,capacity));
            if(direction!="tx") ACL(aclrtMemcpy(dst,capacity,input.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
            ACL(aclrtSetCurrentContext(contexts[0]));
            if(direction=="rx") ACL(aclrtMemset(src,capacity,0x33,capacity));
            launch(); ACL(aclrtSynchronizeStream(stream));
            if(direction!="rx") ACL(aclrtSetCurrentContext(contexts[1]));
            void* checked=direction=="rx" ? src : dst;
            ACL(aclrtMemcpy(output.data(),capacity,checked,capacity,ACL_MEMCPY_DEVICE_TO_HOST));
            uint64_t start=direction=="relay" ? bytes : 0;
            for(uint64_t i=0;i<capacity;++i) {
                uint8_t expected=0x33;
                if(direction=="relay" && i<bytes)expected=input[i];
                if(i>=start && i<start+bytes && (mode==2||mode==3))expected=mode==2 ? 0x5a : input[i-start];
                if(output[i]!=expected)throw std::runtime_error("receiver/guard mismatch byte "+std::to_string(i));
            }
            ACL(aclrtSetCurrentContext(contexts[0]));
        };
        validate();
        for (uint64_t i=0;i<warmup;++i) { launch(); ACL(aclrtSynchronizeStream(stream)); }
        std::puts("sample,bytes,cores,lanes,chunk,mode,event_us,host_us,direction");
        for (uint64_t i=0;i<samples;++i) {
            aclrtEvent begin,end;
            ACL(aclrtCreateEventWithFlag(&begin,ACL_EVENT_TIME_LINE));
            ACL(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE));
            ACL(aclrtRecordEvent(begin,stream));
            const auto hostBegin=std::chrono::steady_clock::now();
            launch();
            ACL(aclrtRecordEvent(end,stream)); ACL(aclrtSynchronizeStream(stream));
            const auto hostEnd=std::chrono::steady_clock::now();
            float ms=0; ACL(aclrtEventElapsedTime(&ms,begin,end));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("Invalid event duration");
            double hostUs=std::chrono::duration<double,std::micro>(hostEnd-hostBegin).count();
            std::printf("%llu,%llu,%llu,%llu,%llu,%llu,%.9g,%.9g,%s\n",
                (unsigned long long)i,(unsigned long long)bytes,(unsigned long long)cores,
                (unsigned long long)lanes,(unsigned long long)chunk,(unsigned long long)mode,ms*1000.,hostUs,direction.c_str());
            ACL(aclrtDestroyEvent(end)); ACL(aclrtDestroyEvent(begin));
        }
        validate();
        ACL(aclrtDestroyStream(stream)); ACL(aclrtFree(src));
        ACL(aclrtIpcMemClose(ipcKey));
        ACL(aclrtDeviceDisablePeerAccess(1));
        ACL(aclrtSetCurrentContext(contexts[1])); ACL(aclrtFree(dst));
        ACL(aclrtResetDevice(1)); ACL(aclrtResetDevice(0)); ACL(aclFinalize());
        std::fprintf(stderr,"PASS: selected peer direction receiver checks before/after; %llu samples; normal cleanup\n",
            (unsigned long long)samples);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
