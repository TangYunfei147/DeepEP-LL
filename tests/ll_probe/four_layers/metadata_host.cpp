#include <acl/acl.h>
#include "snapshot/kernels/dispatch/low_latency/common/comm_args.h"
#include "snapshot/kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "snapshot/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
extern "C" void four_metadata_do(void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
extern "C" void four_metadata_check_do(void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
static void check(int rc,const char* label){if(rc)throw std::runtime_error(std::string(label)+":"+std::to_string(rc));}
#define CK(call) check(call,#call)
int main(int argc,char** argv){try{
    // Optional: B H K E logical_partitions warmup samples. No communication.
    if(argc!=1&&argc!=8)throw std::runtime_error("metadata_probe [B H K E partitions warmup samples]");
    auto value=[&](int i){size_t end;auto n=std::stoul(argv[i],&end);if(argv[i][end]||n>65536)throw std::runtime_error("invalid argument");return uint32_t(n);};
    uint32_t H=argc==8?value(2):4096,K=argc==8?value(3):6,E=argc==8?value(4):256,
       partitions=argc==8?value(5):16,warmup=argc==8?value(6):0,samples=argc==8?value(7):2;
    std::vector<uint32_t> batches=argc==8?std::vector<uint32_t>{value(1)}:std::vector<uint32_t>{4,32,96};
    if(!H||!K||K>32||K>E||E>1024||partitions<16||partitions>256||partitions%8||!E||E%partitions||warmup>10000||samples>10000)
        throw std::runtime_error("unsupported metadata shape");
    CK(aclInit(nullptr));CK(aclrtSetDevice(0));aclrtStream stream;CK(aclrtCreateStream(&stream));
    std::puts("B,H,K,E,logical_partitions,sample,event_us");
    for(uint32_t B:batches){
        if(!B||B>256)throw std::runtime_error("B must be 1..256");
        const auto layout=DispatchDedup::DispatchServerDedupDataLayout(B,H,K,E,false);
        auto align=[](uint32_t n,uint32_t a){return (n+a-1)/a*a;};
        uint32_t count=(B+31)/32,groups=partitions/8-1;groups=groups<K/2?groups:K/2;
        uint64_t used=2*layout.packedPayloadBytes+align(B*K*4,256)+3*align(B*K*4,32)+
          2*align(count*K*4,32)+align(count*4,32)+align(count*K*16,32)+
          (align(partitions/8*4,32)>64?align(partitions/8*4,32):64)+(groups?align(groups*4,32):0);
        if(used+layout.tokenStride>190*1024||layout.blockCount>255)throw std::runtime_error("sender UB budget exceeded");
        // Logical expert partition for local metadata only; no SHMEM instance
        // or 16-rank communication is created. Exactly device0 executes.
        DispatchDedup::CommArgs args{};args.rank=0;args.rankSize=partitions;
        DispatchDedup::EpServerDedupTilingData td{};td.bs=B;td.h=H;td.topK=K;td.moeExpertNum=E;
        std::vector<int32_t> ids(B*K);std::vector<float> weights(B*K),outWeights(B*K);
        std::vector<uint32_t> q(B*K),masks(B);
        for(uint32_t t=0;t<B;++t)for(uint32_t k=0;k<K;++k){ids[t*K+k]=(t*3+k)%E;weights[t*K+k]=0.5f+k*0.125f;}
        void *a,*i,*w,*d,*m,*s,*tiling;
        CK(aclrtMalloc(&a,sizeof(args),ACL_MEM_MALLOC_HUGE_FIRST));CK(aclrtMalloc(&i,B*K*4,ACL_MEM_MALLOC_HUGE_FIRST));
        const uint32_t diagnosticBytes=B*K*4>1024?B*K*4:1024;
        CK(aclrtMalloc(&w,B*K*4,ACL_MEM_MALLOC_HUGE_FIRST));CK(aclrtMalloc(&d,diagnosticBytes,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&m,B*4,ACL_MEM_MALLOC_HUGE_FIRST));CK(aclrtMalloc(&s,B*K*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&tiling,sizeof(td),ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(a,sizeof(args),&args,sizeof(args),ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(i,B*K*4,ids.data(),B*K*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(w,B*K*4,weights.data(),B*K*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(tiling,sizeof(td),&td,sizeof(td),ACL_MEMCPY_HOST_TO_DEVICE));
        for(uint32_t repeat=0;repeat<2;++repeat){
            CK(aclrtMemset(d,B*K*4,0xff,B*K*4));CK(aclrtMemset(s,B*K*4,0,B*K*4));
            four_metadata_check_do(stream,(uint8_t*)a,(uint8_t*)i,(uint8_t*)w,(uint8_t*)d,(uint8_t*)m,(uint8_t*)s,(uint8_t*)tiling);
            CK(aclrtSynchronizeStream(stream));
            CK(aclrtMemcpy(q.data(),B*K*4,d,B*K*4,ACL_MEMCPY_DEVICE_TO_HOST));
            CK(aclrtMemcpy(masks.data(),B*4,m,B*4,ACL_MEMCPY_DEVICE_TO_HOST));
            CK(aclrtMemcpy(outWeights.data(),B*K*4,s,B*K*4,ACL_MEMCPY_DEVICE_TO_HOST));
            for(uint32_t t=0;t<B;++t){
                uint32_t mask=0;std::vector<std::vector<uint32_t>> remote(partitions/8);
                for(uint32_t k=0;k<K;++k){auto server=ids[t*K+k]/(E/partitions*8);if(server)remote[server].push_back(k);else mask|=1U<<k;}
                for(auto& group:remote)if(!group.empty())mask|=1U<<group[0];
                if(masks[t]!=mask)throw std::runtime_error("source mask mismatch");
                for(uint32_t k=0;k<K;++k){
                    uint32_t expected=0xffffffffU;
                    if(mask&(1U<<k)){expected=0;for(uint32_t p=0;p<t*K+k;++p)expected+=ids[p]==ids[t*K+k];
                        auto server=ids[t*K+k]/(E/partitions*8);
                        float weight=server&&remote[server].size()>=2&&k==remote[server][0]?1.f:weights[t*K+k];
                        if(outWeights[t*K+k]!=weight)throw std::runtime_error("source weight mismatch");}
                    if(q[t*K+k]!=expected)throw std::runtime_error("expert q mismatch B="+std::to_string(B)+" t="+std::to_string(t)+" k="+std::to_string(k)+" actual="+std::to_string(q[t*K+k])+" expected="+std::to_string(expected));
                }
            }
        }
        for(uint32_t repeat=0;repeat<warmup+samples;++repeat){
            aclrtEvent begin,end;CK(aclrtCreateEventWithFlag(&begin,ACL_EVENT_TIME_LINE));CK(aclrtCreateEventWithFlag(&end,ACL_EVENT_TIME_LINE));
            CK(aclrtRecordEvent(begin,stream));
            four_metadata_do(stream,(uint8_t*)a,(uint8_t*)i,(uint8_t*)w,(uint8_t*)d,(uint8_t*)m,(uint8_t*)s,(uint8_t*)tiling);
            CK(aclrtRecordEvent(end,stream));CK(aclrtSynchronizeStream(stream));float ms;CK(aclrtEventElapsedTime(&ms,begin,end));
            if(repeat>=warmup)std::printf("%u,%u,%u,%u,%u,%u,%.9g\n",B,H,K,E,partitions,repeat-warmup,ms*1000.);
            std::vector<uint32_t> checksums(256);
            CK(aclrtMemcpy(checksums.data(),1024,d,1024,ACL_MEMCPY_DEVICE_TO_HOST));
            for(uint32_t core=0;core<32;++core){
                uint32_t count=B/32+(core<B%32),start=core*(B/32)+(core<B%32?core:B%32),expected=0;
                for(uint32_t t=start;t<start+count;++t)for(uint32_t k=0;k<K;++k)if(masks[t]&(1U<<k))expected+=q[t*K+k];
                if(count&&checksums[core*8]!=expected)throw std::runtime_error("timing checksum mismatch");
            }
            CK(aclrtDestroyEvent(end));CK(aclrtDestroyEvent(begin));
        }
        CK(aclrtFree(tiling));CK(aclrtFree(s));CK(aclrtFree(m));CK(aclrtFree(d));CK(aclrtFree(w));CK(aclrtFree(i));CK(aclrtFree(a));
    }
    CK(aclrtDestroyStream(stream));CK(aclrtResetDevice(0));CK(aclFinalize());
    std::fprintf(stderr,"PASS metadata: repeated route prefix q, dedup masks and weights; device0 only, no distributed ranks\n");
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
