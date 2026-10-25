// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstddef>
#include <cstdint>
constexpr int ACLSHMEM_MAX_IP_PORT_LEN = 64;
constexpr int ACLSHMEM_STATUS_NOT_INITIALIZED = 0;
constexpr int ACLSHMEM_DATA_OP_MTE = 1;
constexpr int ACLSHMEM_DATA_OP_UDMA = 8;
using data_op_engine_type_t = int;
constexpr int ACLSHMEMX_INIT_WITH_DEFAULT = 1;
constexpr int ACLSHMEMX_INIT_WITH_MPI = 2;
struct aclshmemx_uniqueid_t {
    char bytes[128]{};
};
enum aclshmem_config_state_t : uint32_t {
    ACLSHMEM_CONFIG_DEFAULT = 0,
    ACLSHMEM_CONFIG_DISABLED = 1,
    ACLSHMEM_CONFIG_ENABLED = 2,
};
struct aclshmem_runtime_config_t {
    aclshmem_config_state_t udma_shared_jetty = ACLSHMEM_CONFIG_DEFAULT;
    uint32_t reserved[15]{};
};
struct aclshmem_init_optional_attr_t {
    int version;
    data_op_engine_type_t data_op_engine_type;
    uint32_t shm_init_timeout, shm_create_timeout, control_operation_timeout;
    int32_t sockFd;
};
struct aclshmemx_init_attr_t {
    int my_pe = 0;
    int n_pes = 0;
    char ip_port[64]{};
    uint64_t local_mem_size = 0;
    aclshmem_init_optional_attr_t option_attr = {
        (2 << 16) + sizeof(aclshmemx_init_attr_t), ACLSHMEM_DATA_OP_MTE, 120, 120, 120, 0};
    void* comm_args = nullptr;
    uint64_t instance_id = 0;
    aclshmem_runtime_config_t runtime_config{};
};
constexpr int32_t ACLSHMEM_INIT_ATTR_VERSION = (2 << 16) + sizeof(aclshmemx_init_attr_t);
int aclshmemx_init_status();
int aclshmemx_init_attr(int, aclshmemx_init_attr_t*);
struct aclshmemx_buffer_optional_attr_t {
    void* mem_handle = nullptr;
    const void* fabric_handle = nullptr;
    uint64_t reserved[6]{};
};
struct aclshmemx_buffer_desc_t {
    void* addr = nullptr;
    uint64_t size = 0;
    const aclshmemx_buffer_optional_attr_t* optional_attr = nullptr;
    uint64_t reserved[5]{};
};
int aclshmemx_init_attr_with_buffers(int, aclshmemx_init_attr_t*, const aclshmemx_buffer_desc_t*, size_t);
void* aclshmemx_get_buffer_ptr(const void*);
int aclshmem_finalize();
int aclshmemx_set_qp_num(data_op_engine_type_t, uint32_t);
