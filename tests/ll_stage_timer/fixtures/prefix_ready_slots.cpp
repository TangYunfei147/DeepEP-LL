// Actual production publication/read/clear methods with delayed DMA semantics.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#ifdef NDEBUG
#error assertions must stay enabled
#endif
#define __aicore__
#define __gm__
#define __ubuf__
#define LL_SEG_SCOPE(...)
#define LL_SEG_COUNT(...)
#define LL_SPARSE_MARK(...)
#define LL_BW_BEGIN(...)
#define LL_BW_END(...)
#define LL_BW_TRANSFER(...)
using GM_ADDR = uint8_t*;
constexpr uint32_t kLlSendCoreNum=30, kEpServerDedupMagicStride=512;
constexpr uint32_t UB_ALIGN=32;
enum class HardEvent { MTE2_S, MTE3_S, V_MTE3, V_S, S_V, S_MTE3 };
enum class CacheMode { CACHE_MODE_DISABLE };
constexpr int PIPE_ALL=1;
struct CommArgs { std::array<GM_ADDR,16> peerMems{}; };
struct Stream {
    std::deque<std::function<void()>> read,write;
    void Reads(){while(!read.empty()){auto f=read.front();read.pop_front();f();}}
    void Writes(){while(!write.empty()){auto f=write.front();write.pop_front();f();}}
};
Stream* current=nullptr;
template<int> void PipeBarrier(){current->Reads();current->Writes();}
template<int,int> void SetCtrlSpr(unsigned){}
uint32_t Ceil(uint32_t n,uint32_t d){return (n+d-1)/d;}
template<class T> struct LocalTensor {
    T* data=nullptr;
    T GetValue(unsigned i)const{return data[i];}
    void SetValue(unsigned i,T x){data[i]=x;}
    uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(data);}
};
template<class T> struct GlobalTensor {
    T* data=nullptr;
    void SetGlobalBuffer(T* p){data=p;}
    void SetL2CacheHint(CacheMode){}
};
template<class...> struct TBuf {
    std::shared_ptr<std::vector<uint64_t>> storage;
    template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(storage->data())};}
};
struct TPipe {
    void InitBuffer(TBuf<>& b,unsigned bytes){
        b.storage=std::make_shared<std::vector<uint64_t>>((bytes+7)/8,0);
    }
    void Reset(){assert(current->read.empty()&&current->write.empty());}
};
template<class T> void Duplicate(LocalTensor<T> x,T value,unsigned count){std::fill(x.data,x.data+count,value);}
struct DataCopyExtParams { unsigned blocks,bytes,src,dst,pad; };
template<class T> struct DataCopyPadExtParams { bool pad;unsigned left,right;T value; };
template<class T> void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,unsigned count){
    // Read the UB source on completion, not at issue, to catch premature reuse.
    current->write.emplace_back([=]{std::memcpy(dst.data,src.data,count*sizeof(T));});
}
template<class T> void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,unsigned count){
    current->read.emplace_back([=]{std::memcpy(dst.data,src.data,count*sizeof(T));});
}
template<class T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams p){
    assert(p.blocks==1);current->write.emplace_back([=]{std::memcpy(dst.data,src.data,p.bytes);});
}
template<class T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
    assert(p.blocks==1);current->read.emplace_back([=]{std::memcpy(dst.data,src.data,p.bytes);});
}
struct R0PrefixScan {};
template<class> void asc_vf_call(uint32_t* p,unsigned n){std::partial_sum(p,p+n,p);}
struct Overlap { bool enabled=OVERLAP;bool Async()const{return enabled;} };
struct ServerDedupCounts {
    Overlap overlap_;
    TPipe* tpipe_;
    CommArgs* args_;
    unsigned aivId_=0,epRankId_=0,coreNum_=56,rscvStatusNum_=16;
    unsigned moeExpertNumPerRank_=1,epWorldSize_=16,expertTokenNumsType_=1;
    unsigned aivUsedAllToAll_=30;
    bool buildLocalTasks_=DEDUP;
    GM_ADDR recvCntWorkspaceGM_=nullptr,sendCountsOutGM_=nullptr;
    GlobalTensor<uint32_t> prefixReadyGm_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    TBuf<> readyBuf_,prefixBuf_,prefixPublishBuf_,gatherMaskOutBuf_,tokenNumBuf_;
    template<HardEvent E> void SyncFunc(){
        if constexpr(E==HardEvent::MTE2_S)current->Reads();
        if constexpr(E==HardEvent::MTE3_S)current->Writes();
    }
    // ACTUAL_COUNTS_METHODS
};

int main(){
    std::vector<uint32_t> control(72*512/4,0);
    CommArgs args;args.peerMems[0]=reinterpret_cast<uint8_t*>(control.data());
    std::array<Stream,56> streams;
    std::array<TPipe,56> pipes;
    std::array<ServerDedupCounts,56> counts;
    std::array<std::vector<int32_t>,2> bank{{std::vector<int32_t>(384,-1),std::vector<int32_t>(384,-1)}};
    std::vector<int32_t> publicCounts(384,0);
    std::array<int64_t,24> expertCounts{};
    unsigned magic=0;
    for(unsigned generation=1;generation<=30;++generation){
        // D-D and D-C-D, changing route tables and empty routes.
        if(generation%3)magic^=1;
        magic^=1;
        const unsigned experts=generation%2 ? 16:384;
        std::vector<int32_t> values(experts);
        for(unsigned i=0;i<experts;++i)values[i]=generation%5 ? (generation*7+i*11)%13 : 0;
        auto expected=values;std::partial_sum(expected.begin(),expected.end(),expected.begin());
        for(unsigned core=0;core<56;++core){
            assert(control[(core*512+32)/4]==0);
            counts[core]=ServerDedupCounts{};
            auto& c=counts[core];c.aivId_=core;c.tpipe_=&pipes[core];c.args_=&args;
            c.rscvStatusNum_=experts;c.moeExpertNumPerRank_=experts/16;
            c.recvCntWorkspaceGM_=reinterpret_cast<uint8_t*>(bank[magic].data());
            c.sendCountsOutGM_=reinterpret_cast<uint8_t*>(publicCounts.data());
            c.prefixReadyGm_.SetGlobalBuffer(&control[(core*512+32)/4]);
            c.expertTokenNumsOutGMTensor_.SetGlobalBuffer(expertCounts.data());
        }
        auto& r0=counts[30];current=&streams[30];
        pipes[30].InitBuffer(r0.gatherMaskOutBuf_,experts*4);
        pipes[30].InitBuffer(r0.prefixPublishBuf_,32);
        std::copy(values.begin(),values.end(),r0.gatherMaskOutBuf_.Get<int32_t>().data);
        r0.CalRecvAndSetFlag();
        assert(std::equal(expected.begin(),expected.end(),bank[magic].begin()));
        assert(std::equal(expected.begin(),expected.end(),publicCounts.begin()));
        const unsigned first=DEDUP ? 0:30;
        assert(current->write.size()==56-first);
        for(unsigned core=0;core<56;++core)assert(control[(core*512+32)/4]==0);
        current->Writes(); // ready DMA arrives after the table.
        for(unsigned core=0;core<56;++core)assert(control[(core*512+32)/4]==unsigned(core>=first));
        std::vector<unsigned> order{30};
        for(unsigned core=56;core-->first;)if(core!=30)order.push_back(core);
        for(unsigned core:order){
            current=&streams[core];counts[core].CompleteCumsum();
            assert(std::equal(expected.begin(),expected.end(),counts[core].prefixBuf_.Get<int32_t>().data));
            assert(control[(core*512+32)/4]==unsigned(OVERLAP));
            if(!OVERLAP)pipes[core].Reset(); // actual fallback releases UB next.
            if(core==30) {
                current->Writes();
                assert(control[(55*512+32)/4]==1); // fast R0 cannot hide slow core ready.
            }
        }
        for(unsigned core:order){current=&streams[core];PipeBarrier<PIPE_ALL>();}
        for(unsigned core=0;core<56;++core) {
            assert(control[(core*512+32)/4]==0);
            assert(control[(core*512)/4]==0); // never overwrote magic.
        }
    }
}
