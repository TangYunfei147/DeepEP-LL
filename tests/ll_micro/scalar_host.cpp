#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void scalar_micro_do(uint32_t,void*,uint8_t*,uint8_t*,uint32_t,uint32_t,uint32_t);
#define CK(x) do{auto rc=(x);if(rc)throw std::runtime_error(std::string(#x)+":"+std::to_string(rc));}while(0)
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("scalar_probe cores repeats count");
    uint32_t cores=std::stoul(argv[1]),repeats=std::stoul(argv[2]),count=std::stoul(argv[3]);
    if(!cores||cores>56||!repeats||repeats>4096||!count||count>256)throw std::runtime_error("range");
    CK(aclInit(nullptr));CK(aclrtSetDevice(0));aclrtStream stream;CK(aclrtCreateStream(&stream));
    void* ids;CK(aclrtMalloc(&ids,8192,ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint32_t> routes(2048);for(uint32_t i=0;i<2048;++i)routes[i]=i%384;
    CK(aclrtMemcpy(ids,8192,routes.data(),8192,ACL_MEMCPY_HOST_TO_DEVICE));
    void* result;CK(aclrtMalloc(&result,cores*32,ACL_MEM_MALLOC_HUGE_FIRST));std::vector<uint64_t> values(cores*4);
    std::puts("sample,core,mode,repeats,count,ticks,checksum,event_us");
    for(int sample=-10;sample<100;++sample) for(uint32_t order=0;order<10;++order){
        uint32_t mode=(order+uint32_t(sample+10))%10; // Rotate order to reduce drift bias.
        aclrtEvent a,b;CK(aclrtCreateEventWithFlag(&a,ACL_EVENT_TIME_LINE));CK(aclrtCreateEventWithFlag(&b,ACL_EVENT_TIME_LINE));
        CK(aclrtRecordEvent(a,stream));scalar_micro_do(cores,stream,(uint8_t*)ids,(uint8_t*)result,mode,repeats,count);
        CK(aclrtRecordEvent(b,stream));CK(aclrtSynchronizeStream(stream));float ms;CK(aclrtEventElapsedTime(&ms,a,b));
        CK(aclrtMemcpy(values.data(),cores*32,result,cores*32,ACL_MEMCPY_DEVICE_TO_HOST));
        uint64_t expected=0;
        if(mode==1||mode==2||mode==9)for(uint32_t j=0;j<count;++j)for(uint32_t k=0;k<3;++k)expected+=(j*4+k)%384;
        if(mode==7||mode==8)for(auto v:routes)expected+=v;
        if(mode==3)expected=72;
        if(mode==4)expected=72ULL*65536+71*72/2;
        if(mode==5||mode==6)expected=uint64_t(count)*(count-1)/2;
        for(uint32_t c=0;c<cores;++c){
            if(values[c*4+1]!=expected||values[c*4+2]!=mode||values[c*4+3]!=count)throw std::runtime_error("checksum mismatch");
            if(sample>=0)std::printf("%d,%u,%u,%u,%u,%llu,%llu,%.6f\n",sample,c,mode,repeats,count,
                (unsigned long long)values[c*4],(unsigned long long)values[c*4+1],ms*1000.);
        }
        CK(aclrtDestroyEvent(a));CK(aclrtDestroyEvent(b));
    }
    CK(aclrtFree(ids));CK(aclrtFree(result));CK(aclrtDestroyStream(stream));CK(aclrtResetDevice(0));CK(aclFinalize());
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
