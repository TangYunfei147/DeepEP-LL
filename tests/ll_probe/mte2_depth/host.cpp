#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void mte2_depth_do(void*, uint8_t*, uint8_t*, uint8_t*, uint32_t,
                               uint32_t, uint32_t, uint32_t);
#define ACL(call) do { auto rc = (call); if (rc != ACL_SUCCESS) { \
  std::fprintf(stderr, "%s failed: %d\n", #call, int(rc)); return 1; } } while (0)
static std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out; std::stringstream stream(text); std::string item;
    while (std::getline(stream, item, ',')) { if(item.empty()) throw std::runtime_error("empty list item"); out.push_back(item); }
    if(out.empty()) throw std::runtime_error("empty list"); return out;
}
static uint32_t number(const std::string& s) {
    size_t end=0; auto x=std::stoull(s,&end);
    if(end!=s.size() || x>UINT32_MAX || s[0]=='-') throw std::runtime_error("invalid integer");
    return uint32_t(x);
}
static std::vector<uint32_t> numbers(const std::string& s) {
    std::vector<uint32_t> out; for(auto& x:split(s)) out.push_back(number(x)); return out;
}
int main(int argc, char** argv) try {
    uint32_t device=0, warmup=10, iters=100;
    auto requests=numbers("0,1,8,12,14,15,16,17,18,20,24,32");
    auto chunks=numbers("128,4096"), strides=numbers("65536");
    auto modes=split("batch,serial"); std::string path;
    for(int a=1;a<argc;a+=2) {
        if(a+1==argc) throw std::runtime_error("each option requires a value");
        std::string key=argv[a], value=argv[a+1];
        if(key=="--device") device=number(value);
        else if(key=="--requests") requests=numbers(value);
        else if(key=="--chunk") chunks=numbers(value);
        else if(key=="--stride") strides=numbers(value);
        else if(key=="--mode") modes=split(value);
        else if(key=="--iters") iters=number(value);
        else if(key=="--warmup") warmup=number(value);
        else if(key=="--output") path=value;
        else throw std::runtime_error("unknown option: "+key);
    }
    if(device>1 || iters==0 || iters>100000 || warmup>100000 || path.empty())
        throw std::runtime_error("require device 0/1, iters 1..100000, warmup <=100000, --output path");
    size_t capacity=32;
    for(auto n:requests) for(auto chunk:chunks) for(auto stride:strides) {
        if(n>32 || chunk<32 || chunk>4096 || chunk%32 || stride<chunk || stride%32 || stride>1048576)
            throw std::runtime_error("require N 0..32; chunk 32..4096 aligned32; stride chunk..1MiB aligned32");
        capacity=std::max(capacity, size_t(31)*stride+chunk);
    }
    for(auto& mode:modes) if(mode!="batch" && mode!="serial") throw std::runtime_error("mode must be batch or serial");
    if(capacity>32*1024*1024) throw std::runtime_error("source capacity exceeds 32MiB");
    std::ofstream csv(path); if(!csv) throw std::runtime_error("cannot create output CSV");
    csv<<"device,mode,requests,chunk_bytes,stride_bytes,iteration,issue_ticks,drain_ticks,total_ticks,event_us,verified\n";
    ACL(aclInit(nullptr)); ACL(aclrtSetDevice(device));
    aclrtStream stream; aclrtEvent begin,end;
    ACL(aclrtCreateStream(&stream)); ACL(aclrtCreateEvent(&begin)); ACL(aclrtCreateEvent(&end));
    void *src=nullptr,*dst=nullptr,*stats=nullptr;
    constexpr size_t ubBytes=131072, guard=64;
    ACL(aclrtMalloc(&src,capacity,ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL(aclrtMalloc(&dst,ubBytes+guard*2,ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL(aclrtMalloc(&stats,32,ACL_MEM_MALLOC_NORMAL_ONLY));
    std::vector<uint8_t> input(capacity), output(ubBytes+guard*2);
    uint64_t caseId=0;
    for(auto& mode:modes) for(auto n:requests) for(auto chunk:chunks) for(auto stride:strides) {
        ++caseId;
        for(uint32_t iter=0;iter<warmup+iters;++iter) {
            // 每次改变源 pattern；D2H 校验、H2D 准备均不进入 Event 区间。
            for(size_t i=0;i<capacity;++i) input[i]=uint8_t((i*13+i/251+iter*17+caseId*29)%251);
            ACL(aclrtMemcpy(src,capacity,input.data(),capacity,ACL_MEMCPY_HOST_TO_DEVICE));
            ACL(aclrtMemset(dst,output.size(),0x5c,output.size()));
            ACL(aclrtMemset(stats,32,0,32));
            ACL(aclrtRecordEvent(begin,stream));
            mte2_depth_do(stream,static_cast<uint8_t*>(src),static_cast<uint8_t*>(dst)+guard,
                static_cast<uint8_t*>(stats),n,chunk,stride,mode=="serial");
            ACL(aclrtRecordEvent(end,stream)); ACL(aclrtSynchronizeStream(stream));
            float ms=0; ACL(aclrtEventElapsedTime(&ms,begin,end));
            ACL(aclrtMemcpy(output.data(),output.size(),dst,output.size(),ACL_MEMCPY_DEVICE_TO_HOST));
            for(size_t i=0;i<output.size();++i) {
                uint8_t expected=0x5c;
                if(i>=guard && i<guard+ubBytes) {
                    size_t u=i-guard;
                    expected=u<size_t(n)*chunk ? input[(u/chunk)*stride+u%chunk] : uint8_t(0xa7);
                }
                if(output[i]!=expected) {
                    std::fprintf(stderr,"FAIL device=%u mode=%s N=%u chunk=%u stride=%u iter=%u offset=%zu got=%u expected=%u\n",
                        device,mode.c_str(),n,chunk,stride,iter,i,unsigned(output[i]),unsigned(expected)); return 2;
                }
            }
            uint64_t ticks[4]={};
            ACL(aclrtMemcpy(ticks,sizeof(ticks),stats,sizeof(ticks),ACL_MEMCPY_DEVICE_TO_HOST));
            if(ticks[0]+ticks[1]!=ticks[2]) throw std::runtime_error("invalid tick arithmetic");
            if(iter>=warmup) csv<<device<<','<<mode<<','<<n<<','<<chunk<<','<<stride<<','<<(iter-warmup)<<','
                <<ticks[0]<<','<<ticks[1]<<','<<ticks[2]<<','<<double(ms)*1000.0<<",1\n";
        }
        csv.flush(); if(!csv) throw std::runtime_error("CSV write failed");
        std::printf("PASS device=%u mode=%s N=%u chunk=%u stride=%u samples=%u\n",device,mode.c_str(),n,chunk,stride,iters);
    }
    ACL(aclrtFree(stats)); ACL(aclrtFree(dst)); ACL(aclrtFree(src));
    ACL(aclrtDestroyEvent(end)); ACL(aclrtDestroyEvent(begin)); ACL(aclrtDestroyStream(stream));
    ACL(aclrtResetDevice(device)); ACL(aclFinalize());
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"error: %s\n",e.what()); return 1; }
