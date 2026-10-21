#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "csrc/kernels/dispatch/low_latency/common/ll_group_compact.h"
struct Access {
    const uint32_t* indices;
    uint32_t Index(uint32_t i) { return indices[i]; }
};
struct Emit {
    uint32_t core;
    void operator()(uint32_t index,uint32_t mask) {
        printf("%u %u %u\n",core,index,mask);
    }
};
int main() {
    uint32_t world,experts,group,send,rank,dedup,b,k;
    while (scanf("%u%u%u%u%u%u%u%u",&world,&experts,&group,&send,&rank,&dedup,&b,&k)==8) {
        auto* ids=static_cast<uint32_t*>(calloc(b*k,sizeof(uint32_t)));
        auto* indices=static_cast<uint32_t*>(calloc(b*k,sizeof(uint32_t)));
        if (!ids || !indices) return 2;
        for (uint32_t i=0;i<b*k;++i) if (scanf("%u",&ids[i])!=1) return 3;
        const uint32_t groups=world/group,own=rank/group,phase=rank%group;
        auto* order=static_cast<uint32_t*>(calloc(groups,sizeof(uint32_t)));
        if (!order) return 2;
        for (uint32_t i=0;i<groups;++i) if (scanf("%u",&order[i])!=1) return 3;
        for (uint32_t core=0;core<send;++core) {
            uint32_t owner=phase%send;
            for (uint32_t pass=0;pass<(dedup?2U:1U);++pass) {
                for (uint32_t step=0;step<groups;++step) {
                    const uint32_t g=order[step];
                    if (g==own) continue;
                    uint32_t size=0;
                    for(uint32_t i=0;i<b*k;++i)
                        if (ids[i]/(experts/world*group)==g) indices[size++]=i;
                    DispatchDedup::LlVisitCompactGroup(Access{indices},size,k,
                        dedup!=0,dedup && pass==0,send,core,owner,Emit{core});
                }
            }
        }
        free(ids);free(indices);free(order);
        printf("END\n");
    }
}
