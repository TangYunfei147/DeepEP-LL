#include "runtime/session.hpp"
#include "runtime/udma_peer_masks.hpp"
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
extern "C" void network_mte_copy_do(void*, void*, void*);
static void check(int rc, const char* label) {
    if (rc) throw std::runtime_error(std::string(label) + ": " + std::to_string(rc));
}
#define CHECK(x) check((x), #x)
int main(int argc, char** argv) {
    try {
        if (argc != 9) throw std::runtime_error("rank world device peer endpoint bytes timeout huge_pages");
        int rank=std::stoi(argv[1]), world=std::stoi(argv[2]), device=std::stoi(argv[3]), peer=std::stoi(argv[4]);
        auto bytes=std::stoull(argv[6]); unsigned timeout=std::stoul(argv[7]);
        if (world<2 || rank<0 || rank>=world || peer<=0 || peer>=world || device<0)
            throw std::runtime_error("invalid ranks/device");
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(device));
        std::cerr<<"rank "<<rank<<" device "<<device<<" initializing production Session"<<std::endl;
        // Compile the production Session, rather than reproducing its initialization.
        deepep::Session session(rank, world, bytes, argv[5], timeout, true, false, false, false, std::stoi(argv[8])!=0);
        bool all_mapped=true;
        for (int p=0; p<world; ++p) {
            auto ptr=aclshmem_ptr(session.workspace(), p);
            all_mapped &= ptr && reinterpret_cast<uintptr_t>(ptr)%32==0;
        }
        std::cout << "RESULT {\"rank\":"<<rank<<",\"kind\":\"mapping\",\"all_peers_mapped\":"
                  <<(all_mapped?"true":"false")<<",\"mte_actual_net_layer\":null}"<<std::endl;
        if (!all_mapped) throw std::runtime_error("LL requires aligned MTE mappings for every peer");
        if (rank==0 || rank==peer) {
            int dst=rank==0?peer:0;
            uint32_t qps=0; CHECK(aclshmemx_get_qp_num(ACLSHMEM_DATA_OP_UDMA, &qps));
            uint64_t mask=session.sparse_qp_enabled()?deepep::udma_admission::PeerMask(rank,dst,world):~uint64_t(0);
            for (uint32_t q=0;q<qps;++q) {
                if (!(mask & (uint64_t(1)<<q))) continue;
                aclshmemx_net_layer_t layer=ACLSHMEM_NET_LAYER_DEFAULT;
                int rc=aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA,dst,q,&layer);
                std::cout<<"RESULT {\"kind\":\"udma_route\",\"rank\":"<<rank<<",\"peer\":"<<dst
                         <<",\"qp\":"<<q<<",\"query_rc\":"<<rc<<",\"selected_net_layer\":";
                if(rc) std::cout<<"null"; else std::cout<<uint32_t(layer);
                std::cout<<",\"payload_sent_on_this_qp\":false}"<<std::endl;
                CHECK(rc);
            }
        }
        auto mapped=[&](int p) {
            auto* ptr=static_cast<uint8_t*>(aclshmem_ptr(session.workspace(),p));
            // Same peer-base bias as csrc/ops/low_latency.cpp.
            return ptr+(bytes==(8ULL<<30)?uint64_t(p)*1024:0);
        };
        void* src=nullptr; aclrtStream stream=nullptr;
        CHECK(aclrtMalloc(&src,32,ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK(aclrtCreateStream(&stream));
        for (int direction=0;direction<2;++direction) {
            int sender=direction==0?0:peer, receiver=direction==0?peer:0;
            unsigned char pattern=direction==0?0x5a:0xa5;
            CHECK(aclrtMemset(mapped(rank),32,0,32));
            CHECK(aclrtMemset(src,32,pattern,32));
            session.barrier();
            if (rank==sender) {
                network_mte_copy_do(stream,src,mapped(receiver));
                CHECK(aclrtSynchronizeStream(stream));
            }
            session.barrier();
            if (rank==receiver) {
                unsigned char actual[32];
                CHECK(aclrtMemcpy(actual,32,mapped(rank),32,ACL_MEMCPY_DEVICE_TO_HOST));
                for (auto v:actual) if(v!=pattern) throw std::runtime_error("MTE readback mismatch");
                std::cout<<"RESULT {\"kind\":\"mte_readback\",\"rank\":"<<rank<<",\"sender\":"<<sender
                         <<",\"bytes\":32,\"passed\":true,\"actual_net_layer\":null}"<<std::endl;
            }
            session.barrier();
        }
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtFree(src)); session.close();
        CHECK(aclrtResetDevice(device)); CHECK(aclFinalize());
        std::cout<<"RESULT {\"kind\":\"complete\",\"rank\":"<<rank<<"}"<<std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"network_probe: "<<e.what()<<std::endl;
        // Do not enter collective cleanup on a failed rank. Launcher bounds the job.
        return 1;
    }
}
