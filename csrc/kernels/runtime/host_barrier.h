// DeepEP-owned NDR Host barrier. Compiled into libdeepep_kernels, never SHMEM.
// Uses the locked SDK's existing transport APIs without modifying the SDK.
// Host callers must serialize this collective with communication kernels.
ACLSHMEM_DEVICE void deepep_barrier_store(__gm__ int32_t* address, int32_t value)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    signal.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(signal);
}

ACLSHMEM_DEVICE void deepep_barrier_wait(__gm__ int32_t* address, int32_t generation)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    while (true) {
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(signal);
        const int32_t value = signal.GetValue(0);
        if (value == generation || value == generation + 1) return;
    }
}

ACLSHMEM_DEVICE void deepep_host_barrier(aclshmem_team_t tid)
{
    if (tid == -1) return;
    auto state = aclshmemi_get_state();
    auto team = state->team_pools[tid];
    bool remote = false;
    for (int i = 0; i < team->size; ++i) {
        const int peer = team->pe_mapping[i];
        if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) remote = true;
    }
    if (!remote) {
        aclshmemi_sync<false>(tid);
        return;
    }

    // All participating cores rendezvous, but only AIV0 owns the generation,
    // QPs and SDK UB scratch. AIC and the second AIV must not issue RDMA.
    aclshmemi_sync_core<false>();
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            auto pool = aclshmemi_get_team_sync_pool(tid);
            auto counter = aclshmemi_get_team_sync_counter(tid);
            auto scratch = reinterpret_cast<__ubuf__ uint8_t*>(state->rdma_config.aclshmem_ub);
            const uint32_t event = state->rdma_config.sync_id;
            const int me = state->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
            int index = -1;
            for (int i = 0; i < team->size; ++i) {
                const int peer = team->pe_mapping[i];
                if (peer == me) index = i;
                if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) {
                    if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_ROCE)) {
                        AscendC::Trap();
                        return;
                    }
                    // Complete prior writes on every per-peer QP before arrival.
                    const uint32_t qps = aclshmemi_qp_info_fetch()->qp_num;
                    for (uint32_t qp = 0; qp < qps; ++qp)
                        aclshmemx_roce_qp_quiet(peer, qp, scratch, event);
                }
            }
            if (index < 0) { AscendC::Trap(); return; }
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(counter));
            const int32_t previous = aclshmemi_load(reinterpret_cast<__gm__ int32_t*>(counter));
            // The SDK polling convention also accepts generation+1.
            if (previous >= INT32_MAX - 1) { AscendC::Trap(); return; }
            const int32_t generation = previous + 1;
            deepep_barrier_store(reinterpret_cast<__gm__ int32_t*>(counter), generation);

            // Binary dissemination: one cacheline per round, always local polling.
            // Mapping, rather than start/stride arithmetic, supports explicit teams.
            int slot = 0;
            for (int shift = 1; shift < team->size; shift <<= 1, ++slot) {
                if (slot >= SYNC_ARRAY_SIZE / ACLSHMEM_SYNCBIT_SIZE) { AscendC::Trap(); return; }
                const int peer = team->pe_mapping[(index + shift) % team->size];
                if (state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) {
                    deepep_barrier_store(reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(pool + slot, peer)), generation);
                } else {
                    // Pass LOCAL symmetric addresses to the SDK for translation.
                    // No RDMA get, remote Scalar dereference or remote atomic.
                    aclshmemx_roce_qp_put_nbi(
                        reinterpret_cast<__gm__ uint8_t*>(pool + slot),
                        reinterpret_cast<__gm__ uint8_t*>(counter), scratch, sizeof(int32_t), peer, 0, event);
                    aclshmemx_roce_qp_quiet(peer, 0, scratch, event);
                }
                deepep_barrier_wait(reinterpret_cast<__gm__ int32_t*>(pool + slot), generation);
            }
        }
    }
    aclshmemi_sync_core<false>();
}
