#pragma once
#include <cassert>
#include <cstring>
#include "kernels/common/ep_packet_receive_layout.h"

namespace SoModel {
namespace Receive = DispatchDedup::PacketReceiveLayout;
inline Receive::State Probe(const uint8_t* p, uint32_t bytes) {
    float sum = 0;
    for (uint32_t b=0;b<bytes;b+=512U)
        for (uint32_t w=0;w<8U;++w) {
            float value; std::memcpy(&value,p+b+480U+w*4U,4U); sum+=value;
        }
    return Receive::Decode(sum,bytes/512U);
}
inline void Flags(uint8_t* p,uint32_t bytes,bool keep) {
    for (uint32_t b=0;b<bytes;b+=512U)
        for (uint32_t w=0;w<8U;++w) {
            const float value=keep && b==0 && w==0 ? 1.5f : 1.0f;
            std::memcpy(p+b+480U+w*4U,&value,4U);
        }
}
// Default 2KiB fragments. Reorder ALL non-final fragments adversarially;
// the SO final fragment cannot arrive until those predecessors retire.
// Within a fragment expose one word at a time, including a partial tail flag.
inline void Deliver(const Model::Request& r) {
    assert(r.bytes && r.bytes%512U==0);
    const uint32_t last=(r.bytes-1U)/2048U*2048U;
    auto fragment=[&](uint32_t begin,uint32_t end) {
        for (uint32_t b=begin;b<end;b+=4U) {
            std::memcpy(r.dst+b,r.src+b,4U);
            if (b%512U>=480U && b+4U<r.bytes) assert(Probe(r.dst,r.bytes)==Receive::State::Pending);
        }
    };
    for (uint32_t end=last;end;end-=2048U) fragment(end-2048U,end);
    fragment(last,r.bytes);
    assert(Probe(r.dst,r.bytes)==Probe(r.src,r.bytes));
    assert(Probe(r.dst,r.bytes)!=Receive::State::Pending);
    assert(std::memcmp(r.dst,r.src,r.bytes)==0);
}
}
