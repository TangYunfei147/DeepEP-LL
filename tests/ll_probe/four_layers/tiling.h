#pragma once
#include <cstdint>
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
