// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included after shmem.h, inside the caller's device implementation namespace.
// SHMEM master accepts LOCAL symmetric pointers and converts them to a peer MR
// address. A peer MTE mapping and a remote native VA are different address spaces.
namespace DeepepUdmaAddress {
__aicore__ inline bool Rebase(uint64_t address, uint64_t source_base, uint64_t target_base,
                            uint64_t heap_size, uint64_t bytes, uint64_t& result)
{
    if (!bytes || address < source_base) return false;
    const uint64_t offset=address-source_base;
    if (offset>=heap_size || bytes>heap_size-offset || target_base>UINT64_MAX-offset) return false;
    const uint64_t target=target_base+offset;
    if (bytes-1U>UINT64_MAX-target) return false;
    result=target;
    return true;
}

__aicore__ inline bool Reject()
{
    aclshmemi_kernel_abort("UDMA address is outside the registered symmetric MR; ordinary peer GM is unsupported\n");
    trap();
    return false;
}

template<typename T, aclshmemi_udma_opcode_t Operation,
         const aclshmemx_udma_op_config_t& Config>
__aicore__ inline bool FillLocalSymmetric(__ubuf__ aclshmemi_sqe_ctx_t* header,
                                        __gm__ uint8_t* local_address,
                                        __gm__ aclshmemi_ubmem_info_t* memory, uint32_t head)
{
    auto* state=aclshmemi_get_state();
    if (!state || !memory || !state->heap_base || !state->heap_size || !memory->addr) return Reject();
    const uint64_t local_base=reinterpret_cast<uint64_t>(state->heap_base);
    uint64_t remote=0U;
    if (!Rebase(reinterpret_cast<uint64_t>(local_address),local_base,memory->addr,
                state->heap_size,sizeof(T),remote)) return Reject();
    const aclshmemi_udma_addr_ctx_t context{local_base,memory->addr};
    aclshmemi_udma_fill_sqe_base_ctx<T,Operation,__ubuf__ aclshmemi_sqe_ctx_t*,Config>(
        header,local_address,memory,head,context);
    return true;
}

// A raw peer address is admitted only when it belongs to the peer's registered
// MR. This is not a compatibility escape for an unregistered framework output.
template<typename T, aclshmemi_udma_opcode_t Operation,
         const aclshmemx_udma_op_config_t& Config>
__aicore__ inline bool FillRegisteredRemote(__ubuf__ aclshmemi_sqe_ctx_t* header,
                                          __gm__ uint8_t* remote_address,
                                          __gm__ aclshmemi_ubmem_info_t* memory, uint32_t head)
{
    auto* state=aclshmemi_get_state();
    if (!state || !memory || !state->heap_base || !state->heap_size || !memory->addr) return Reject();
    uint64_t local=0U;
    if (!Rebase(reinterpret_cast<uint64_t>(remote_address),memory->addr,
                reinterpret_cast<uint64_t>(state->heap_base),state->heap_size,sizeof(T),local)) return Reject();
    return FillLocalSymmetric<T,Operation,Config>(header,reinterpret_cast<__gm__ uint8_t*>(local),memory,head);
}

template<typename T, aclshmemi_udma_opcode_t Operation,
         const aclshmemx_udma_op_config_t& Config>
__aicore__ inline bool FillMappedPeer(__ubuf__ aclshmemi_sqe_ctx_t* header,
                                    __gm__ uint8_t* mapped_address, uint64_t mapped_peer_base,
                                    __gm__ aclshmemi_ubmem_info_t* memory, uint32_t head)
{
    auto* state=aclshmemi_get_state();
    if (!state || !memory || !state->heap_base || !state->heap_size || !memory->addr || !mapped_peer_base) return Reject();
    uint64_t local=0U;
    if (!Rebase(reinterpret_cast<uint64_t>(mapped_address),mapped_peer_base,
                reinterpret_cast<uint64_t>(state->heap_base),state->heap_size,sizeof(T),local)) return Reject();
    return FillLocalSymmetric<T,Operation,Config>(header,reinterpret_cast<__gm__ uint8_t*>(local),memory,head);
}
}
