#include <cassert>
#include <iostream>
#include <vector>
#define DEEPEP_LL_RDMA 1
#include "kernels/dispatch/low_latency/device/ll_transport.h"
#include "roce_so_model.h"
using namespace AscendC;
int main() {
    std::vector<uint8_t> local(256*1024),remote(local.size()),stages(72U*LlRdma::kCoreBytes),packet(128*1024);
    Model::local=local.data(); Model::remote=remote.data();
    DispatchDedup::CommArgs a; a.rank=0; a.rankSize=32; a.reserved0=16; a.peerMemBytes=128*1024;
    a.reservedAddresses[0]=local.data(); a.reservedAddresses[1]=stages.data(); a.reservedConfig[0]=LlRdma::kBackend;
    a.peerMems[0]=local.data(); a.peerMems[1]=remote.data()+1024; // All remote MTE mappings deliberately absent.
    unsigned cases=0;
    for (cores=16;cores<=72;cores+=2) {
        a.reservedConfig[1]=8; a.reservedAddresses[2]=stages.data()+uint64_t(cores)*LlRdma::kCoreBytes;
        for (core=0;core<cores;++core) {
            for (uint32_t bytes : {512U,1536U,2048U,2560U,4096U,7680U,15872U}) {
                for (unsigned epoch=1;epoch<=3;++epoch) {
                    // Reused receive slots retain old payload; only flags clear.
                    std::fill(remote.begin(),remote.end(),0x99);
                    for(uint32_t b=0;b<bytes;b+=512U) std::memset(remote.data()+16384+b+480U,0,32U);
                    for (uint32_t i=0;i<bytes;++i) packet[i]=uint8_t((i+epoch*17+core)%251+1);
                    SoModel::Flags(packet.data(),bytes,epoch%2!=0);
                    auto saved=packet;
                    uint8_t* target=remote.data()+16384;
                    Model::before=[&](const Model::Request& r) {
                        assert(r.qp==core%8 && r.peer==16);
                        assert(r.bytes==bytes);
                        SoModel::Deliver(r);
                    };
                    unsigned q=Model::quiets;
                    LlTransport::Packet(&a,16,0,{packet.data()},bytes,7U,true,core%8);
                    assert(Model::quiets-q==1 && Model::pending.empty());
                    assert(std::memcmp(target,saved.data(),bytes)==0 && packet==saved);
                    ++cases;
                }
            }
        }
    }
    Model::before={}; core=0; cores=72; a.reservedConfig[1]=8; a.reservedAddresses[2]=stages.data()+72U*LlRdma::kCoreBytes;
    // Single-use combine packets retain wire contents and completion while
    // allowing the SDK to consume their UB prefix. Each new call repacks it.
    for (uint32_t turn=0;turn<3;++turn) {
        for (uint32_t i=0;i<1536;++i) packet[i]=uint8_t(i+turn+1U);
        const auto expected=packet;
        const auto oldQuiets=Model::quiets;
        LlTransport::Packet(&a,16,0,{packet.data()},1536,7U,false);
        assert(std::memcmp(remote.data()+16384,expected.data(),1536)==0);
        assert(std::memcmp(packet.data(),expected.data(),128)!=0);
        assert(Model::quiets-oldQuiets==1U && Model::pending.empty());
    }
    std::vector<uint8_t> records(96),scratch(128),received(16000);
    for (uint32_t i=0;i<records.size();++i) records[i]=uint8_t(i+1);
    std::fill(remote.begin(),remote.end(),0);
    Model::before=[&](const Model::Request& r) {
        if(r.bytes==4) for(unsigned i=0;i<3;++i)
            assert(std::memcmp(remote.data()+16384+512*i+4,records.data()+32*i+4,28)==0);
    };
    LlTransport::Records(&a,16,0,{records.data()},3,512,true,{scratch.data()});
    for(unsigned i=0;i<3;++i) assert(std::memcmp(remote.data()+16384+512*i,records.data()+32*i,32)==0);
    Model::before={};
    LlTransport::Records(&a,16,2048,{records.data()},3,512,false,{scratch.data()});
    auto beforeLocal=Model::puts;
    LlTransport::Packet(&a,1,0,{packet.data()},512);
    LlTransport::Records(&a,1,2048,{records.data()},3,512,true,{scratch.data()});
    LlTransport::LoadLocalPeer(&a,1,0,{reinterpret_cast<uint16_t*>(received.data())},512);
    assert(std::memcmp(received.data(),packet.data(),512)==0 && Model::puts==beforeLocal && Model::gets==0);
    bool rejectedRead=false;
    try { LlTransport::LoadLocalPeer(&a,16,0,{reinterpret_cast<uint16_t*>(received.data())},512); }
    catch(const std::runtime_error&) { rejectedRead=true; }
    assert(rejectedRead && Model::gets==0 && Model::pending.empty());
    auto p=Model::puts;
    LlTransport::Packet(&a,0,0,{packet.data()},512);
    assert(p==Model::puts && std::memcmp(local.data(),packet.data(),512)==0);
    bool trapped=false;
    try { LlTransport::Packet(&a,16,a.peerMemBytes-256,{packet.data()},512); } catch(const std::runtime_error&) { trapped=true; }
    assert(trapped);
    trapped=false; core=72;
    try { LlTransport::Packet(&a,16,0,{packet.data()},512); } catch(const std::runtime_error&) { trapped=true; }
    assert(trapped);
    core=0;
    Model::pending.push_back({remote.data()+200000,local.data()+200000,1U,0U,1U});
    LlTransport::Packet(&a,16,0,{packet.data()},512);
    assert(Model::pending.size()==1 && Model::pending[0].qp==1U); // no cross-QP quiet
    Model::pending.clear();
    // Eight receiver prefetches have live MTE2_S events. Forwarding may only
    // borrow the consumed current lane, never blindly reserve event 7.
    for(unsigned lane=0;lane<8;++lane) SetFlag<HardEvent::MTE2_S>(lane);
    for(unsigned lane=0;lane<8;++lane) {
        WaitFlag<HardEvent::MTE2_S>(lane);
        LlTransport::Packet(&a,16,0,{packet.data()},512,lane);
        for(unsigned next=lane+1;next<8;++next) assert(events[int(HardEvent::MTE2_S)][next]);
    }
    std::fill(remote.begin(),remote.end(),0);
    Model::before=[](const Model::Request&) { throw std::runtime_error("injected CQ failure"); };
    p=Model::puts; trapped=false;
    try { LlTransport::Packet(&a,16,0,{packet.data()},1536); } catch(const std::runtime_error&) { trapped=true; }
    assert(trapped && Model::puts-p==1U); // one message; source cannot be released after failed CQ
    for(unsigned b=0;b<1536;b+=512) assert(remote[16384+b+480]==0);
    Model::before={}; Model::pending.clear();
    std::cout << "PASS " << cases << " packet publications across AIV 16..72, records/local-read/cross-read-rejection/self/reuse/range/QP; CPU model only\n";
}
