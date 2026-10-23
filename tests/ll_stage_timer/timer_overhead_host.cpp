#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#define DECLARE(name) extern "C" void name##_launch(void*,uint8_t*,uint8_t*,uint32_t,uint32_t); \
                      extern "C" void name##_info(uint32_t*)
DECLARE(baseline_off);DECLARE(baseline_on);DECLARE(current_off);DECLARE(current_on);
using Launch=void(*)(void*,uint8_t*,uint8_t*,uint32_t,uint32_t);
using Info=void(*)(uint32_t*);
struct Variant {const char* name;Launch launch;Info info;};
const Variant variants[]={
    {"baseline_off",baseline_off_launch,baseline_off_info},
    {"baseline_on",baseline_on_launch,baseline_on_info},
    {"current_off",current_off_launch,current_off_info},
    {"current_on",current_on_launch,current_on_info}};
#define CK(x) do { auto rc=(x); if(rc)throw std::runtime_error(std::string(#x)+":"+std::to_string(rc)); } while(0)
static uint64_t expected(uint32_t core,uint32_t sendCores,uint32_t pattern,uint32_t repeats) {
    const uint32_t send[]={81,82,84,16,64,18,20,21,87};
    const uint32_t recv[]={22,23,6,10,78,88,90,91,94,95,96};
    uint64_t value=uint64_t(core)+1;
    if(pattern)for(uint32_t i=0;i<repeats;++i) {
        if(core<sendCores)for(auto stage:send)value=value*1664525ULL+stage+1013904223ULL;
        else for(auto stage:recv)value=value*1664525ULL+stage+1013904223ULL;
    }
    return value;
}
int main(int argc,char** argv) {try {
    if(argc!=5)throw std::runtime_error("timer_overhead_probe baseline_off|baseline_on|current_off|current_on null|capture empty|boundaries repeats");
    const Variant* selected=nullptr;
    for(const auto& variant:variants)if(variant.name==std::string(argv[1]))selected=&variant;
    if(!selected)throw std::runtime_error("unknown variant");
    std::string pointer=argv[2],patternName=argv[3];
    if(pointer!="null"&&pointer!="capture")throw std::runtime_error("invalid pointer mode");
    if(patternName!="empty"&&patternName!="boundaries")throw std::runtime_error("invalid pattern");
    size_t used=0;unsigned long parsed=std::stoul(argv[4],&used);
    if(argv[4][used]||parsed<1||parsed>4096)throw std::runtime_error("repeats must be 1..4096");
    uint32_t repeats=uint32_t(parsed),pattern=patternName=="boundaries",info[5]={};
    selected->info(info);
    if(pointer=="capture"&&!info[4])throw std::runtime_error("OFF capture rejected; choose null");
    for(auto key:{"CUDA_VISIBLE_DEVICES","ASCEND_RT_VISIBLE_DEVICES","ASCEND_VISIBLE_DEVICES","PROBE_VISIBLE_DEVICES"}) {
        const char* value=std::getenv(key);
        if(!value||(std::string(value)!="0,1"&&std::string(value)!="0"))
            throw std::runtime_error(std::string("Require ")+key+"=0,1 or 0");
    }
    CK(aclInit(nullptr));CK(aclrtSetDevice(0));aclrtStream stream;CK(aclrtCreateStream(&stream));
    void *timer=nullptr,*result=nullptr;
    const size_t timerBytes=56ULL*info[1]*sizeof(uint64_t),resultBytes=56ULL*128;
    CK(aclrtMalloc(&timer,timerBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&result,resultBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint64_t> records(resultBytes/8),timerWords(timerBytes/8);
    aclrtEvent begin,end;
    CK(aclrtCreateEventWithFlag(&begin,ACL_EVENT_TIME_LINE));CK(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE));
    std::fprintf(stderr,"MICRO_ONLY schema=%u metrics=%u words_per_core=%u send_cores=%u timer_compiled=%u warmup=10 samples=100 no_Exp_no_all_reduce; event_us includes full kernel incl result export and capture SyncAll/Flush; ticks retain raw frequency\n",info[0],info[2],info[1],info[3],info[4]);
    std::puts("variant,pointer,pattern,repeats,sample,core,constructor_init_ticks,init_span_ticks,kernel_span_ticks,export_sync_ticks,total_ticks,checksum,event_us");
    for(int sample=-10;sample<100;++sample) {
        CK(aclrtMemset(result,resultBytes,0,resultBytes));
        if(pointer=="capture")CK(aclrtMemset(timer,timerBytes,0,timerBytes));
        CK(aclrtRecordEvent(begin,stream));
        // The pointer crosses the Host launch ABI at runtime, including null.
        selected->launch(stream,pointer=="capture"?static_cast<uint8_t*>(timer):nullptr,
                         static_cast<uint8_t*>(result),pattern,repeats);
        CK(aclrtRecordEvent(end,stream));CK(aclrtSynchronizeStream(stream));
        float ms=0;CK(aclrtEventElapsedTime(&ms,begin,end));
        CK(aclrtMemcpy(records.data(),resultBytes,result,resultBytes,ACL_MEMCPY_DEVICE_TO_HOST));
        if(pointer=="capture")CK(aclrtMemcpy(timerWords.data(),timerBytes,timer,timerBytes,ACL_MEMCPY_DEVICE_TO_HOST));
        for(uint32_t core=0;core<56;++core) {
            auto* r=records.data()+core*16;
            if(r[5]!=expected(core,info[3],pattern,repeats)||r[6]!=pattern||r[7]!=repeats)
                throw std::runtime_error("result checksum/identity mismatch at core "+std::to_string(core));
            for(uint32_t j=0;j<4;++j)if(r[j+1]<r[j])throw std::runtime_error("non-monotonic local ticks");
            if(pointer=="capture") {
                auto* h=timerWords.data()+core*info[1];
                if(h[0]!=0x4c4c5431ULL||h[1]!=info[0]||h[2]!=1||h[3]!=56||h[4]!=core||
                   (h[5]&255)!=1||((h[5]>>8)&255)!=info[3])
                    throw std::runtime_error("timer header mismatch at core "+std::to_string(core));
                if(info[0]==15U) {
                    const uint64_t required=(1ULL<<0)|(1ULL<<1)|(1ULL<<13);
                    if(info[1]!=64U || ((h[5]>>24)&255)!=6U || h[7] || (h[6]&~((1ULL<<14)-1)) ||
                       (h[6]&required)!=required || h[8]>h[9] || h[9]>h[21])
                        throw std::runtime_error("sparse timer identity/point mismatch");
                    const uint64_t active=pattern ? (core<info[3] ? 0x21c7ULL : 0x3e3fULL) : required;
                    if(h[6]!=active) throw std::runtime_error("sparse timer point mask mismatch");
                    for(uint32_t point=0;point<14U;++point) {
                        const uint64_t tick=h[8+point];
                        if(h[6]&(1ULL<<point)) {
                            if(tick<h[8] || tick>h[21]) throw std::runtime_error("sparse point outside work span");
                        } else if(tick) throw std::runtime_error("invalid point must be zero");
                    }
                } else {
                if(h[6]<6 || 8+2*h[6]>info[1])throw std::runtime_error("legacy timer record bounds");
                bool kernelFound=false,initFound=false;
                for(uint64_t j=0;j<h[6];++j) {
                    const uint64_t tag=h[8+j*2],id=tag&0xffffffffULL;
                    if(id==0x8000ULL){if((tag>>32)!=1)throw std::runtime_error("KERNEL call count mismatch");kernelFound=true;}
                    if(id==0x8032ULL){if((tag>>32)!=1)throw std::runtime_error("INIT call count mismatch");initFound=true;}
                }
                if(!kernelFound||!initFound)throw std::runtime_error("required timer metric absent");
                }
            }
            if(sample>=0)std::printf("%s,%s,%s,%u,%d,%u,%llu,%llu,%llu,%llu,%llu,%llu,%.6f\n",
                selected->name,pointer.c_str(),patternName.c_str(),repeats,sample,core,
                (unsigned long long)(r[1]-r[0]),(unsigned long long)(r[2]-r[1]),
                (unsigned long long)(r[3]-r[2]),(unsigned long long)(r[4]-r[3]),
                (unsigned long long)(r[4]-r[0]),(unsigned long long)r[5],ms*1000.0);
        }
    }
    CK(aclrtDestroyEvent(begin));CK(aclrtDestroyEvent(end));CK(aclrtFree(timer));CK(aclrtFree(result));
    CK(aclrtDestroyStream(stream));CK(aclrtResetDevice(0));CK(aclFinalize());return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}}
