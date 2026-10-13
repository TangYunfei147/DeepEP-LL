#pragma once
#include <cstdint>
// Host/Device共用32B参数布局；修改字段类型/顺序必须同时重编Host和kernel。
// bytesPerEdge=单边字节，chunkBytes=每次搬运块，lanes=每核UB流水槽数；rank/world为实际通信域。
// layer编号0/1/3是内部模式，不等于文档四层的序号；第2层使用独立metadata入口。
struct FourLayerTiling {
    uint64_t bytesPerEdge;
    uint32_t chunkBytes;
    uint32_t lanes;
    uint32_t rank;
    uint32_t world;
    uint32_t layer;  // 0=empty launch, 1=tiling-only, 3=transfer
    uint32_t scenario; // 1=one-to-one,2=many-to-one,3=one-to-many,4=many-to-many
};
static_assert(sizeof(FourLayerTiling)==32,"one GM tiling block");
