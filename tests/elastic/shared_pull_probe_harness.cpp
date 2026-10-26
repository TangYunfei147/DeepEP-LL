// Compile the actual receiver descriptor publisher and payload-loader methods.
// DMA is immediate here: separate ordering checks and the lifetime model test
// cover dependencies. This checks bytes, bounds, scratch restoration and copies.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error CPU packet/lifetime checks require enabled assertions
#endif
#include "csrc/kernels/dispatch/low_latency/common/ll_shared_pull_probe_config.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"
using namespace DispatchDedup;
#define __aicore__
#define __gm__
#define LL_SEG_BEGIN(a,b)
#define LL_SEG_END(a,b)
#define LL_SEG_SCOPE(a)
#define LL_MTE3_DRAIN(a,b,c)
enum class HardEvent {MTE2_S,MTE2_MTE3,MTE3_MTE2,V_S,S_MTE3,MTE3_S};
enum class CacheMode {CACHE_MODE_DISABLE};
template<HardEvent E>void SetFlag(unsigned){}
template<HardEvent E>void WaitFlag(unsigned){}
template<HardEvent E>void ServerDedupReceiveSync(){}
void Trap(){throw std::runtime_error("bad descriptor");}
template<class T>struct LocalTensor {
    uint8_t* p;
    LocalTensor operator[](uint32_t i)const{return {p+i*sizeof(T)};}
    template<class U>LocalTensor<U> ReinterpretCast()const{return {p};}
    void SetValue(uint32_t i,T v){std::memcpy(p+i*sizeof(T),&v,sizeof v);}
    T GetValue(uint32_t i)const{T v;std::memcpy(&v,p+i*sizeof(T),sizeof v);return v;}
};
template<class T>struct GlobalTensor {
    uint8_t* p{};
    void SetGlobalBuffer(void* x){p=static_cast<uint8_t*>(x);}
    void SetL2CacheHint(CacheMode){}
    GlobalTensor operator[](uint32_t i)const{return {p+i*sizeof(T)};}
};
struct Params {uint16_t count;uint32_t len,srcStride,dstStride,reserved;};
template<class T>struct Pad {bool pad;uint8_t left,right;T value;};
uint64_t writes=0,reads=0;
template<class T>void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,Params p){
    for(unsigned i=0;i<p.count;++i){std::memcpy(dst.p+i*(p.len+p.dstStride),src.p+i*(p.len+p.srcStride),p.len);writes+=p.len;}
}
template<class T>void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,Params p,Pad<T>){
    for(unsigned i=0;i<p.count;++i){std::memcpy(dst.p+i*(p.len+p.dstStride),src.p+i*(p.len+p.srcStride),p.len);reads+=p.len;}
}
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t n){std::memcpy(dst.p,src.p,n*sizeof(T));reads+=n*sizeof(T);}
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,uint32_t n){std::memcpy(dst.p,src.p,n*sizeof(T));writes+=n*sizeof(T);}
template<class T>void Duplicate(LocalTensor<T> dst,T value,uint32_t n){for(unsigned i=0;i<n;++i)dst.SetValue(i,value);}
struct Buffer {uint8_t* p;template<class T>LocalTensor<T> Get(){return {p};}};
struct Args {uint8_t* peerMems[16];unsigned rank=0,rankSize=16;};
using CommArgs=Args;
// ACTUAL_METADATA_METHOD
struct Data {uint64_t headerPayloadOffset=3712,headerWireOffset=3936,blockCount=8,tokenStride=4096;
    uint64_t dispatchOffset=4096,dispatchBytes=8192;};
struct Receiver {
    Args* args_;Data data_;Buffer lanesBuf_;uint32_t laneStride_=4160;void* pipe_{};
    bool sharedPull_=true,pulledTerminal_[8]{};
    static constexpr uint32_t kLlRanksPerGroup=4;
    uint8_t* Slot(unsigned j){return args_->peerMems[args_->rank]+data_.dispatchOffset+j*data_.tokenStride;}
    // ACTUAL_RECEIVER_METHODS
};
int main(){
    assert(LlSharedPullScope(true,true,16)==(kLlSharedPullProbeMode!=0));
    assert(!LlSharedPullScope(false,true,16)&&!LlSharedPullScope(true,false,16)&&!LlSharedPullScope(true,true,32));
    Args owner,reader;std::vector<std::vector<uint8_t>> peers(16,std::vector<uint8_t>(65536,0));
    for(unsigned r=0;r<16;++r)owner.peerMems[r]=reader.peerMems[r]=peers[r].data();
    owner.rank=1;reader.rank=2;
    std::vector<uint8_t> lane(4160,0);Receiver pub{&owner,{}, {lane.data()}};
    for(unsigned i=0;i<4096;++i)peers[1][4096+i]=uint8_t(i*13+7);
    auto gmHeader=LocalTensor<uint8_t>{peers[1].data()+4096+3936}.ReinterpretCast<uint32_t>();
    gmHeader.SetValue(0,9);gmHeader.SetValue(1,17);gmHeader.SetValue(2,3);gmHeader.SetValue(3,2);
    auto original=peers[1];
    LocalTensor<uint8_t> wire{lane.data()};
    auto header=wire[3936].ReinterpretCast<uint32_t>();
    header.SetValue(0,9);header.SetValue(1,17);header.SetValue(2,3);header.SetValue(3,3);
    auto type=wire[4096].ReinterpretCast<uint32_t>();type.SetValue(0,2);
    GlobalTensor<uint8_t> dst{peers[2].data()+4096};
    pub.PublishSharedReference(dst,wire,type,header,0);
    assert(writes==32+8*32 && type.GetValue(0)==2);
    assert(header.GetValue(0)==9&&header.GetValue(1)==17&&header.GetValue(2)==3&&header.GetValue(3)==3);
    assert(peers[1]==original); // Header rewrite only touched the private UB.
    for(unsigned b=0;b<8;++b)for(unsigned i=0;i<8;++i){float v;std::memcpy(&v,peers[2].data()+4096+b*512+480+i*4,4);assert(v==1.0f);}
    std::vector<uint8_t> loaded(4160);Receiver get{&reader,{}, {loaded.data()}};
    get.IssueLoad(0,0,false,false);
    assert(get.pulledTerminal_[0] && reads==32+8*480);
    for(unsigned b=0;b<8;++b)assert(!std::memcmp(loaded.data()+b*480,original.data()+4096+b*512,480));
    auto pulledHeader=LocalTensor<uint8_t>{loaded.data()+3712}.ReinterpretCast<uint32_t>();
    pulledHeader.SetValue(3,uint32_t(EpServerDedupTokenType::Terminal));
    auto outType=LocalTensor<uint8_t>{loaded.data()+4096}.ReinterpretCast<uint32_t>();
    auto outDestination=LocalTensor<uint8_t>{loaded.data()+4128}.ReinterpretCast<uint32_t>();
    BuildServerDedupMetadata(outType,outDestination,pulledHeader.GetValue(3),137);
    assert(outType.GetValue(0)==3 && outDestination.GetValue(0)==137);
    assert(outDestination.GetValue(1)==UINT32_MAX && outDestination.GetValue(2)==UINT32_MAX);
    // A reference remains readable after source flag cleanup, not payload erase.
    for(unsigned b=0;b<8;++b)std::memset(peers[1].data()+4096+b*512+480,0,32);
    reads=0;get.IssueLoad(0,0,false,false);assert(get.pulledTerminal_[0] && reads==3872);
    auto desc=LocalTensor<uint8_t>{peers[2].data()+4096+3936}.ReinterpretCast<uint32_t>();
    desc.SetValue(4,8);bool rejected=false;try{get.IssueLoad(0,0,false,false);}catch(...){rejected=true;}assert(rejected);
    for(unsigned group:{1U,4U,16U})for(unsigned rank=0;rank<16;++rank)for(unsigned r=0;r<16;++r){
        assert(LlSharedPullAddressValid(rank,r,16,group,4096,4096,8192,4096)==(rank/group==r/group));
    }
    for(uint64_t off:{0ULL,4095ULL,4097ULL,12288ULL,~0ULL})assert(!LlSharedPullAddressValid(1,2,16,4,off,4096,8192,4096));
    assert(LlSharedPullAddressValid(1,2,16,4,8192,4096,8192,4096));
}
