#pragma once
#include <algorithm>
#include <functional>
#include <vector>
namespace Model {
struct Request { uint8_t* dst; uint8_t* src; uint32_t bytes,peer,qp; };
inline uint8_t *local=nullptr, *remote=nullptr;
inline std::vector<Request> pending;
inline std::function<void(const Request&)> before;
inline unsigned quiets=0, puts=0, gets=0;
inline uint8_t* translate(uint8_t* p) { return remote + (p-local); }
}
inline void aclshmemx_roce_qp_put_nbi(uint8_t* d,uint8_t* s,uint8_t* scratch,uint32_t n,int pe,uint32_t qp,uint32_t) {
    std::memset(scratch,0xa5,128); // WQE/CQE workspace really destroys its UB contents.
    Model::pending.push_back({Model::translate(d),s,n,uint32_t(pe),qp}); ++Model::puts;
}
inline void aclshmemx_roce_qp_get_nbi(uint8_t* d,uint8_t* s,uint8_t* scratch,uint32_t n,int pe,uint32_t qp,uint32_t) {
    std::memset(scratch,0xa5,128);
    Model::pending.push_back({d,Model::translate(s),n,uint32_t(pe),qp}); ++Model::gets;
}
inline void aclshmemx_roce_qp_quiet(uint32_t pe,uint32_t qp,uint8_t* scratch,uint32_t syncId) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(syncId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(syncId);
    std::memset(scratch,0x5a,128); ++Model::quiets;
    // Adversarial completion order within a phase; no cross-QP completion.
    for (size_t i=Model::pending.size(); i--;) {
        auto r=Model::pending[i];
        if (r.peer!=pe || r.qp!=qp) continue;
        if (Model::before) Model::before(r);
        std::memcpy(r.dst,r.src,r.bytes);
        Model::pending.erase(Model::pending.begin()+i);
    }
}
