#include <acl/acl.h>
#include "shmem.h"
#include "tiling.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void four_comm_do(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
static void check(int rc,const char* what) { if(rc) throw std::runtime_error(std::string(what)+":"+std::to_string(rc)); }
#define CK(call) check(call,#call)
static uint64_t num(const char* s) { if(!*s||*s=='-') throw std::runtime_error("invalid integer"); size_t end; auto n=std::stoull(s,&end); if(s[end]) throw std::runtime_error("invalid integer"); return n; }
static bool edge(uint32_t key,uint32_t s,uint32_t d) {
    if(s==d)return false;
    if(key==1)return s==0&&d==1;
    if(key==2)return s!=0&&d==0;
    if(key==3)return s==0&&d!=0;
    return true;
}
static uint8_t pattern(uint32_t rank,uint64_t i) { return uint8_t((rank*31+i*13+i/257)%251); }
#include "../profile_bridge.h"
int main(int argc,char** argv) {
    try {
        if(argc!=12)throw std::runtime_error("rank world device endpoint bytes cores lanes scenario layer warmup samples");
        auto rank=num(argv[1]),world=num(argv[2]),device=num(argv[3]),bytes=num(argv[5]),cores=num(argv[6]),
            lanes=num(argv[7]),scenario=num(argv[8]),layer=num(argv[9]),warmup=num(argv[10]),samples=num(argv[11]);
        if((world!=2&&world!=4&&world!=8&&world!=16&&world!=32)||rank>=world||device>31||!bytes||bytes%32||bytes>(64ULL<<20)||world*bytes>(256ULL<<20)||
           !cores||cores>32||!lanes||lanes>4||scenario<1||scenario>4||(layer!=0&&layer!=1&&layer!=3)||warmup>10000||samples>10000)
            throw std::runtime_error("out-of-range configuration");
        const char* visible=std::getenv("ASCEND_RT_VISIBLE_DEVICES");
        if(!visible)throw std::runtime_error("visible devices must be explicit");
        uint32_t visibleCount=1;for(const char* p=visible;*p;++p)visibleCount+=*p==',';
        if(device>=visibleCount)throw std::runtime_error("device outside visible ordinal range");
        CK(PROBE_ACL_INIT()); CK(aclrtSetDevice(int(device))); aclrtStream stream; CK(aclrtCreateStream(&stream));
        aclshmemx_init_attr_t attr{}; aclshmemx_uniqueid_t uid{};
        attr.my_pe=int(rank);attr.n_pes=int(world);
        attr.local_mem_size=((world*bytes+(16ULL<<20)+(2ULL<<20)-1)/(2ULL<<20))*(2ULL<<20);
        if(std::strlen(argv[4])>=sizeof(attr.ip_port))throw std::runtime_error("endpoint too long");
        std::strcpy(attr.ip_port,argv[4]);attr.comm_args=&uid;attr.option_attr.data_op_engine_type=ACLSHMEM_DATA_OP_MTE;
        attr.option_attr.shm_init_timeout=60;attr.option_attr.shm_create_timeout=60;attr.option_attr.control_operation_timeout=60;
        std::fprintf(stderr,"READY_SHMEM rank=%llu device context created\n",(unsigned long long)rank);std::fflush(stderr);
        if(rank==0){const char* ready=std::getenv("FOUR_READY_FILE");if(ready){
            const char* project=std::getenv("DEEPEP_PROJECT_ROOT");
            const char* workspace=std::getenv("PROBE_ROOT");
            if(!project)throw std::runtime_error("use run.sh to configure the project root");
            namespace fs=std::filesystem;
            const auto base=fs::weakly_canonical(fs::path(project));
            const auto root=fs::weakly_canonical(workspace?fs::path(workspace):base);
            const auto requested=fs::path(ready);
            const auto path=fs::weakly_canonical(requested.is_absolute()?requested:base/requested);
            const auto relative=path.lexically_relative(root);
            if(relative.empty()||relative=="."||*relative.begin()=="..")throw std::runtime_error("ready file outside workspace");
            FILE* file=std::fopen(path.c_str(),"w");if(!file)throw std::runtime_error("ready file failed");std::fputs("ready\n",file);std::fclose(file);
        }}
        CK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT,&attr));
        const uint64_t capacity=world*bytes+32;
        auto* dst=static_cast<uint8_t*>(aclshmem_align(512,capacity));if(!dst)throw std::runtime_error("allocation failed");
        std::vector<uint64_t> peers(world);for(uint32_t p=0;p<world;++p) {
            peers[p]=reinterpret_cast<uint64_t>(aclshmem_ptr(dst,int(p)));
            if(!peers[p])throw std::runtime_error("peer MTE mapping unavailable");
        }
        void *src=nullptr,*peerTable=nullptr,*tiling=nullptr;
        CK(aclrtMalloc(&src,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&peerTable,world*8,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&tiling,sizeof(FourLayerTiling),ACL_MEM_MALLOC_HUGE_FIRST));
        std::vector<uint8_t> input(bytes),output(capacity);
        for(uint64_t i=0;i<bytes;++i)input[i]=pattern(uint32_t(rank),i);
        FourLayerTiling td{bytes,4096,uint32_t(lanes),uint32_t(rank),uint32_t(world),uint32_t(layer),uint32_t(scenario)};
        CK(aclrtMemcpy(src,bytes,input.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(peerTable,world*8,peers.data(),world*8,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(tiling,sizeof(td),&td,sizeof(td),ACL_MEMCPY_HOST_TO_DEVICE));
        auto launch=[&] {four_comm_do(uint32_t(cores),stream,static_cast<uint8_t*>(src),static_cast<uint8_t*>(peerTable),static_cast<uint8_t*>(tiling),layer==0?0:uint32_t(scenario));};
        auto validate=[&] {
            CK(aclrtMemset(dst,capacity,0xff,capacity));aclshmem_barrier_all();
            launch();CK(aclrtSynchronizeStream(stream));aclshmem_barrier_all();
            CK(aclrtMemcpy(output.data(),capacity,dst,capacity,ACL_MEMCPY_DEVICE_TO_HOST));
            for(uint64_t i=0;i<capacity;++i) {
                auto s=i/bytes;auto pos=i%bytes;
                uint8_t expected=layer==3&&s<world&&edge(uint32_t(scenario),uint32_t(s),uint32_t(rank))?pattern(uint32_t(s),pos):0xff;
                if(output[i]!=expected)throw std::runtime_error("receiver mismatch at "+std::to_string(i));
            }
            aclshmem_barrier_all();
        };
        validate();ProfilePhase(0);for(uint64_t i=0;i<warmup;++i){aclshmem_barrier_all();launch();CK(aclrtSynchronizeStream(stream));aclshmem_barrier_all();}
        uint32_t outgoing=0;for(uint32_t p=0;p<world;++p)outgoing+=edge(uint32_t(scenario),uint32_t(rank),p);
        std::puts("sample,rank,world,scenario,layer,bytes_per_edge,rank_wire_bytes,event_us,api_us,job_join_us");
        ProfilePhase(1);
        for(uint64_t i=0;i<samples;++i) {
            ProfilePhase(2);
            aclrtEvent start,end;CK(aclrtCreateEventWithFlag(&start,ACL_EVENT_TIME_LINE));CK(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE));
            aclshmem_barrier_all();CK(aclrtRecordEvent(start,stream));auto begin=std::chrono::steady_clock::now();
            launch();CK(aclrtRecordEvent(end,stream));CK(aclrtSynchronizeStream(stream));auto apiEnd=std::chrono::steady_clock::now();
            aclshmem_barrier_all();auto joinEnd=std::chrono::steady_clock::now();
            float ms;CK(aclrtEventElapsedTime(&ms,start,end));if(!(ms>0)||!std::isfinite(ms))throw std::runtime_error("invalid event duration");
            std::printf("%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.9g,%.9g,%.9g\n",(unsigned long long)i,(unsigned long long)rank,
              (unsigned long long)world,(unsigned long long)scenario,(unsigned long long)layer,(unsigned long long)bytes,
              (unsigned long long)(layer==3?outgoing*bytes:0),ms*1000.,
              std::chrono::duration<double,std::micro>(apiEnd-begin).count(),std::chrono::duration<double,std::micro>(joinEnd-begin).count());
            CK(aclrtDestroyEvent(end));CK(aclrtDestroyEvent(start));
        }
        ProfilePhase(3);
        validate();CK(aclrtFree(tiling));CK(aclrtFree(peerTable));CK(aclrtFree(src));aclshmem_barrier_all();aclshmem_free(dst);
        CK(aclshmem_finalize());CK(aclrtDestroyStream(stream));CK(PROBE_ACL_RESET(int(device)));CK(PROBE_ACL_FINALIZE());
        std::fprintf(stderr,"PASS rank=%llu scenario=%llu layer=%llu receiver checks and normal cleanup\n",(unsigned long long)rank,(unsigned long long)scenario,(unsigned long long)layer);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
