#include <acl/acl.h>
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_cc.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "kernels/dispatch/low_latency/common/ll_rdma_layout.h"
#include "runtime/ll_rdma_qp_config.hpp"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void ll_rdma_smoke_launch(uint32_t,void*,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" void deepep_ndr_host_barrier_launch_v2(void*, uint64_t, uint32_t);
static void Check(int status,const char* op) {
    if(status) throw std::runtime_error(std::string(op)+" status="+std::to_string(status));
}
int main(int argc,char** argv) {
    try {
        if(argc!=8) throw std::invalid_argument("rank world device tcp://host:port rounds cores(0=available) physical-ranks-per-host(1|2)");
        const int rank=std::stoi(argv[1]), world=std::stoi(argv[2]), device=std::stoi(argv[3]);
        const std::string endpoint=argv[4];
        const int ranksPerHost=std::stoi(argv[7]);
        if(ranksPerHost!=1 && ranksPerHost!=2) throw std::invalid_argument("primitive ranks-per-host must be 1 or 2");
        const int rounds=std::stoi(argv[5]); uint32_t cores=std::stoul(argv[6]);
        if(world!=2 || rank<0 || rank>=2 || device<0 || rounds<1 || rounds>10000 || endpoint.size()>=ACLSHMEM_MAX_IP_PORT_LEN)
            throw std::invalid_argument("Primitive smoke requires exactly two real ranks and valid arguments");
        Check(aclInit(nullptr),"aclInit"); Check(aclrtSetDevice(device),"aclrtSetDevice");
        int64_t available=0; Check(aclrtGetDeviceInfo(device,ACL_DEV_ATTR_VECTOR_CORE_NUM,&available),"AIV query");
        if(cores==0) cores=uint32_t(available);
        if(cores<16 || cores>72 || cores%2 || cores>uint32_t(available)) throw std::invalid_argument("cores must be even, 16..72, <=available");
        deepep::validate_ll_rdma_qps(LlRdma::kQpCount, ranksPerHost==1);
        if(ranksPerHost==1) Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE,LlRdma::kQpCount),"ROCE QP configuration");
        constexpr uint64_t logical=8ULL<<20, guard=4096;
        const uint64_t needed=logical+guard+uint64_t(cores)*LlRdma::kCoreBytes;
        const uint64_t bytes=(needed+(2U<<20)-1U)/(2U<<20)*(2U<<20);
        aclshmemx_init_attr_t attr{}; aclshmemx_uniqueid_t uid{};
        attr.my_pe=rank; attr.n_pes=world; attr.local_mem_size=bytes;
        std::memcpy(attr.ip_port,endpoint.c_str(),endpoint.size()+1); attr.comm_args=&uid;
        attr.option_attr.data_op_engine_type=ranksPerHost==1
            ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE) : ACLSHMEM_DATA_OP_MTE;
        attr.option_attr.shm_init_timeout=120; attr.option_attr.shm_create_timeout=120;
        attr.option_attr.control_operation_timeout=120;
        Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT,&attr),"SHMEM hybrid init (no fallback)");
        auto memory=static_cast<uint8_t*>(aclshmem_align(512,bytes));
        if(!memory) throw std::runtime_error("symmetric allocation failed");
        Check(aclrtMemset(memory,bytes,0,bytes),"clear symmetric heap");
        DispatchDedup::CommArgs args;
        args.rank=rank; args.rankSize=world; args.peerMemBytes=logical; args.reserved0=ranksPerHost;
        for(int peer=0;peer<world;++peer) if(rank/ranksPerHost==peer/ranksPerHost) {
            args.peerMems[peer]=static_cast<uint8_t*>(aclshmem_ptr(memory,peer));
            if(!args.peerMems[peer]) throw std::runtime_error("Missing local MTE mapping");
            args.peerMems[peer]+=peer*1024U;
        } args.reservedAddresses[0]=memory;
        args.reservedAddresses[1]=memory+logical+guard; args.reservedConfig[0]=LlRdma::kBackend;
        args.reservedConfig[1]=LlRdma::kQpCount;
        args.reservedAddresses[2]=args.reservedAddresses[1]+uint64_t(cores)*LlRdma::kCoreBytes; args.fftsVal=shmemx_get_ffts_config();
        void *deviceArgs=nullptr,*result=nullptr; aclrtStream stream=nullptr;
        Check(aclrtMalloc(&deviceArgs,sizeof(args),ACL_MEM_MALLOC_HUGE_FIRST),"args allocation");
        Check(aclrtMalloc(&result,cores*32U,ACL_MEM_MALLOC_HUGE_FIRST),"result allocation");
        Check(aclrtMemcpy(deviceArgs,sizeof(args),&args,sizeof(args),ACL_MEMCPY_HOST_TO_DEVICE),"args upload");
        Check(aclrtCreateStream(&stream),"stream creation");
        std::vector<uint8_t> source(cores*16384U); std::vector<uint32_t> status(cores*8U);
        for(int epoch=0;epoch<rounds;++epoch) {
            for(uint32_t core=0;core<cores;++core) for(uint32_t i=0;i<16384U;++i)
                source[core*16384U+i]=uint8_t((rank*19U+core*7U+epoch*13U+i)%251U+1U);
            Check(aclrtMemcpy(memory+rank*1024U+(2U<<20),source.size(),source.data(),source.size(),ACL_MEMCPY_HOST_TO_DEVICE),"source upload");
            for(uint32_t size : {512U,1536U,15872U}) for(uint32_t mode=0;mode<4;++mode) {
                if(mode==1U && ranksPerHost==1) continue; // Cross-host reads are forbidden.
                // Clearing receive windows and collective completion allow safe repeated reuse.
                Check(aclrtMemset(memory+rank*1024U,2U<<20,0,2U<<20),"clear packet slots");
                Check(aclrtMemset(memory+rank*1024U+(4U<<20),cores*2048U,0,cores*2048U),"clear control slots");
                Check(aclrtMemset(result,cores*32U,0xff,cores*32U),"result poison");
                deepep_ndr_host_barrier_launch_v2(stream, 120000000000ULL,
                    std::getenv("DEEPEP_LL_INIT_TRACE") && std::string(std::getenv("DEEPEP_LL_INIT_TRACE")) == "1");
                Check(aclrtSynchronizeStream(stream),"DeepEP initialization barrier completion");
                ll_rdma_smoke_launch(cores,stream,deviceArgs,result,mode,epoch,size);
                Check(aclrtSynchronizeStream(stream),"RDMA kernel completion");
                Check(aclrtMemcpy(status.data(),status.size()*4U,result,status.size()*4U,ACL_MEMCPY_DEVICE_TO_HOST),"result download");
                for(uint32_t core=0;core<cores;++core) if(status[core*8U])
                    throw std::runtime_error("publication/data mismatch: rank="+std::to_string(rank)+" core="+std::to_string(core)+" mode="+std::to_string(mode));
                deepep_ndr_host_barrier_launch_v2(stream, 120000000000ULL,
                    std::getenv("DEEPEP_LL_INIT_TRACE") && std::string(std::getenv("DEEPEP_LL_INIT_TRACE")) == "1");
                Check(aclrtSynchronizeStream(stream),"DeepEP completion barrier");
            }
        }
        Check(aclrtDestroyStream(stream),"destroy stream"); Check(aclrtFree(result),"free result"); Check(aclrtFree(deviceArgs),"free args");
        aclshmem_free(memory); Check(aclshmem_finalize(),"SHMEM finalize");
        Check(aclrtResetDevice(device),"reset device"); Check(aclFinalize(),"ACL finalize");
        std::cout<<"PASS hybrid packet/count/ready rank="<<rank<<" cores="<<cores<<" local_mte_read="<<(ranksPerHost==2)<<" rounds="<<rounds<<"; primitive only, not LL or performance\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<std::endl; return 1; }
}
