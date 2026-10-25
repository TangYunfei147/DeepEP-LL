// Adapted from cann/shmem@8e3fc610, src/device/gm2gm/engine/shmem_device_udma.hpp.
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0;
// see docs/licenses/CANN-Open-Software-License-2.0.txt.
#pragma once

// Only use this for a pointer in THIS PE's symmetric heap. MTE's mapped peer
// address is a different address space from the peer's registered UDMA MR.
__aicore__ inline __gm__ uint8_t* DeepepSymmetricRemoteMrAddr(
    __gm__ const void* local_symmetric_addr, __gm__ aclshmemi_ubmem_info_t* remote_mem_info)
{
    return aclshmemi_remote_mr_addr(local_symmetric_addr, aclshmemi_get_state()->heap_base, remote_mem_info->addr);
}

// These HT builders already hold a final remote VA: either an exchanged live
// framework output VA or an explicitly translated SHMEM remote MR address. The upstream
// fill_sqe_base_ctx now accepts a LOCAL symmetric pointer and translates it.
// Keep raw-VA building separate; never invent an address context or translate
// these addresses twice. Queue admission, publication and CQE handling stay in
// the existing caller/SDK path. Public SHMEM put/get APIs retain their contract.
template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR,
          const aclshmemx_udma_op_config_t& CONFIG>
__aicore__ inline void DeepepFillSqeBaseRawRemoteAddr(
    SQE_PTR sqe_ctx, __gm__ uint8_t* raw_remote_addr, __gm__ aclshmemi_ubmem_info_t* remote_mem_info,
    uint32_t cur_head)
{
    static_assert(OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE, "Raw HT builders support WRITE only");
    static_assert(sizeof(aclshmemi_sqe_ctx_t) == 48U && sizeof(aclshmemi_sge_ctx_t) == 16U,
                  "Review raw WQE encoding after an SDK layout change");
    static_assert(__builtin_offsetof(aclshmemi_sqe_ctx_t, rmt_eid_l) == 16U &&
                  __builtin_offsetof(aclshmemi_sqe_ctx_t, rmt_token_value) == 32U &&
                  __builtin_offsetof(aclshmemi_sqe_ctx_t, rmt_addr_l_or_token_id) == 40U &&
                  __builtin_offsetof(aclshmemi_sge_ctx_t, va) == 8U, "Unexpected SDK SQE/SGE field offsets");
    ACLSHMEM_DEBUG_FUNC(assert_remote_jetty_type_valid, remote_mem_info);
    using SqeQwordPtr = typename AscendC::Std::conditional<
        AscendC::IsSameType<SQE_PTR, __ubuf__ aclshmemi_sqe_ctx_t*>::value, __ubuf__ uint64_t*, __gm__ uint64_t*>::type;
    SqeQwordPtr sqe64 = (SqeQwordPtr)sqe_ctx;
    constexpr uint32_t flag = aclshmemi_udma_build_flag<OP_CODE, CONFIG>();
    const uint32_t dw0 = (flag << 16) | (static_cast<uint32_t>(remote_mem_info->token_value_valid) << 28) |
                         (ACLSHMEM_UDMA_REMOTE_JETTY_TYPE << 29);
    const uint32_t dw1 = (remote_mem_info->target_hint & 0xFFU) | (static_cast<uint32_t>(OP_CODE) << 8);
    const uint32_t dw2 = (remote_mem_info->tpn & 0xFFFFFFU) | (1U << 24);
    const uint32_t dw3 = remote_mem_info->tid & 0xFFFFFU;
    const uint32_t dw4 = remote_mem_info->rmt_token_value;
    const uint32_t dw5 = aclshmemi_udma_get_reduce_attr<T, OP_CODE>();
    auto* rmt_eid = reinterpret_cast<__gm__ uint64_t*>(remote_mem_info->eid_addr);
    sqe64[0] = static_cast<uint64_t>(dw0) | (static_cast<uint64_t>(dw1) << 32);
    sqe64[1] = static_cast<uint64_t>(dw2) | (static_cast<uint64_t>(dw3) << 32);
    sqe64[2] = rmt_eid[0];
    sqe64[3] = rmt_eid[1];
    sqe64[4] = static_cast<uint64_t>(dw4) | (static_cast<uint64_t>(dw5) << 32);
    sqe64[5] = reinterpret_cast<uint64_t>(raw_remote_addr);
    (void)cur_head;
}
