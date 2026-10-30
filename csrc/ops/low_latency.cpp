// Copyright (c) 2026, Lu Lu

#include "ops/low_latency.hpp"
#include <pybind11/stl.h>
#include <climits>
#include "../kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "../kernels/dispatch/low_latency/common/ll_reference_profile_layout.h"
#include "../kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "../kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "../kernels/combine/low_latency/combine_layout.h"
#include "../kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../kernels/dispatch/low_latency/launch.hpp"
#include "../kernels/combine/low_latency/launch.hpp"

namespace py = pybind11;
namespace deepep {
namespace {
using Shape = std::vector<int64_t>;
constexpr uint64_t kRankBaseBiasBytes = 1024U;
uint32_t pin_session_cores(Session& session)
{
    session.workspace();
    const auto config = query_ll_core_config();
    if (config.device < 0) throw std::runtime_error("LL launch requires a current device for core verification");
    return session.pin_low_latency_cores(config.active);
}

uint8_t* peer_buffer_base(Session& session, int peer)
{
    if (peer < 0 || peer >= session.world()) throw std::invalid_argument("Invalid LL peer");
    auto* original = static_cast<uint8_t*>(aclshmem_ptr(session.workspace(), peer));
    if (original == nullptr) throw std::invalid_argument("Invalid LL peer mapping");
    if (session.capacity() != (8ULL << 30)) return original;
    // Translate the original symmetric pointer before applying the destination-rank
    // bias. The shared peer table keeps local receive and remote send views equal.
    return original + uint64_t(peer) * kRankBaseBiasBytes;
}
uint8_t* tensor(py::handle value, const Shape& shape, const char* dtype)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(value).cast<bool>()) {
        throw std::invalid_argument("LL requires tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (!obj.attr("is_contiguous")().cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}


uint8_t* reference_profile(py::dict tensors, uint32_t cores) {
    if (!tensors.contains("ll_reference_profile") || tensors["ll_reference_profile"].is_none()) return nullptr;
    if (!deepep_ll_reference_profile_enabled())
        throw std::invalid_argument("Rebuild the complete wheel with DEEPEP_LL_REFERENCE_PROFILE=ON");
    auto value = py::reinterpret_borrow<py::object>(tensors["ll_reference_profile"]);
    auto device = value.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("Reference profile must be on the input NPU");
    auto* pointer = tensor(value, {cores, LlReferenceProfile::kSlots, LlReferenceProfile::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer) % 512U)
        throw std::invalid_argument("Reference profile requires 512-byte alignment");
    return pointer;
}
void require_small_dispatch_scope(int b, int capacity)
{
    if (DispatchDedup::kLlCb59Small && !(b > 0 && b <= capacity && capacity <= 32))
        throw std::invalid_argument("cb59-small requires 1<=B<=M<=32; no large-B or dedup fallback is compiled");
}
DispatchDedup::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    require_small_dispatch_scope(b, b);
    const uint32_t coreNum = query_ll_core_config().active;
    DispatchDedup::EpMemoryServerDedupLayout result{};
    auto mode = fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                    : DispatchDedup::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536 ||
        DispatchDedup::BuildDispatchServerDedupLayout(p, b, h, k, e, &result, mode, coreNum) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(result, b, k, e, p, coreNum) || result.capacityRows > INT32_MAX) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
uint8_t dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}
DispatchDedup::EpServerDedupTilingData dispatch_format_tiling(const std::string& dtype) {
    DispatchDedup::EpServerDedupTilingData data{};
    const bool fp8=dtype=="float8_e4m3fn";
    data.dtype=dtype_code(dtype);
    data.quantMode=fp8?4:0;
    data.inputMode=static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs :
                                             DispatchDedup::EpServerDedupInputMode::Plain16);
    data.expandXOutDtype=data.dtype;
    return data;
}
}  // namespace

void bind_low_latency(py::module_& module)
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    constexpr uint32_t reference_enabled = 1U;
#else
    constexpr uint32_t reference_enabled = 0U;
#endif
    if (deepep_ll_reference_profile_enabled() != reference_enabled ||
        deepep_ll_reference_profile_schema() != LlReferenceProfile::kSchema)
        throw std::runtime_error("Reference Host/device mode or schema mismatch; rebuild the complete wheel");
    module.attr("low_latency_reference_profile_enabled") = bool(reference_enabled);
    module.def("low_latency_reference_profile_layout", []() {
        py::dict result;
        result["schema"] = LlReferenceProfile::kSchema;
        result["slots"] = LlReferenceProfile::kSlots;
        result["fields"] = LlReferenceProfile::kFields;
        std::vector<std::string> words;
        for (const auto* name : LlReferenceProfile::kNames) words.emplace_back(name);
        result["words"] = words;
        return result;
    });

    if (deepep_ll_instrumentation_source_id() != 7U ||
        deepep_ll_dispatch_launch_bytes() != sizeof(EpServerDedupKernelArgs) ||
        deepep_ll_combine_launch_bytes() != sizeof(EpServerDedupCombineKernelArgs))
        throw std::runtime_error("LL Host/device source artifact or launch ABI mismatch");
    module.attr("low_latency_instrumentation_source") = "reference-mte7";
    module.attr("dispatch_launch_bytes") = 144;
    module.attr("combine_launch_bytes") = 88;
    if (deepep_ll_core_protocol_version() != DispatchDedup::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild and reinstall the complete wheel");
    if (deepep_ll_small_b_prefix_cores()!=DispatchDedup::kLlSmallBPrefixCores)
        throw std::runtime_error("LL Host/device small-B prefix producer count mismatch; rebuild the complete wheel");
    if (deepep_ll_dispatch_implementation()!=DispatchDedup::kLlDispatchImplementationId)
        throw std::runtime_error("LL Host/device dispatch implementation mismatch; rebuild the complete wheel");
    if (deepep_ll_overlap_stage()!=(DispatchDedup::kLlCb59Replay ? 0U : DispatchDedup::kLlOverlapExperimentStage))
        throw std::runtime_error("LL A/B stage differs between Host and device library");
    module.attr("low_latency_overlap_stage")=DispatchDedup::kLlCb59Replay ? 0U : DispatchDedup::kLlOverlapExperimentStage;
    module.attr("low_latency_dispatch_impl")=DispatchDedup::kLlDispatchImplementation;
    module.attr("low_latency_core_protocol") = DispatchDedup::kLlCoreProtocolVersion;
    module.attr("low_latency_small_b_prefix_cores") = DispatchDedup::kLlSmallBPrefixCores;
    module.attr("low_latency_q_table_enabled") = !DispatchDedup::kLlCb59Replay;
    module.attr("low_latency_q_seed_vf") = !DispatchDedup::kLlCb59Replay;
    module.attr("low_latency_implementation") = DispatchDedup::kLlImplementationName;
    module.attr("low_latency_rdma_enabled") = false;


    module.def("low_latency_overlap_layout", [](int64_t ranks,int64_t bs,int64_t h,int64_t k,int64_t experts,bool fp8) {
        if (DispatchDedup::kLlCb59Replay)
            throw std::invalid_argument("cb59 replay does not use the current overlap UB planner");
        DispatchDedup::EpMemoryServerDedupLayout layout{};
        const auto mode=fp8 ? DispatchDedup::EpServerDedupInputMode::Quantize16 : DispatchDedup::EpServerDedupInputMode::Plain16;
        if (DispatchDedup::BuildDispatchServerDedupLayout(ranks,bs,h,k,experts,&layout,mode)!=0 ||
            !DispatchDedup::DispatchDedupPacketUbFits(layout,bs,k,experts,ranks,DispatchDedup::kLlDefaultPlanCoreNum))
            throw std::invalid_argument("Invalid LL shape");
        py::list rows;
        for (uint32_t rank=0;rank<uint32_t(ranks);++rank) for (uint32_t core=0;core<56;++core) {
            const auto p=DispatchDedup::LlMakeOverlapLayout(layout.data,bs,k,experts,ranks,rank,core,mode);
            py::dict row;
            row["rank"]=rank;row["core"]=core;row["overlap"]=p.enabled;
            row["send_lanes"]=p.sendLanes;row["recv_lanes"]=p.recvLanes;
            row["ub_bytes"]=p.totalBytes; rows.append(row);
        }
        return rows;
    });
    module.def("low_latency_core_config", []() {
        const auto cfg = query_ll_core_config();
        py::dict result;
        result["platform_aiv"] = cfg.platform; result["available_aiv"] = cfg.available;
        result["active_aiv"] = cfg.active; result["send_cores"] = DispatchDedup::kLlSendCoreNum;
        result["dispatch_recv_cores"] = DispatchDedup::kLlDispatchRecvCoreNum;
        result["recv_cores"] = DispatchDedup::kLlRecvCoreNum; result["device"] = cfg.device;
        result["runtime_device_queried"] = cfg.device >= 0;
        result["control_capacity_aiv"] = DispatchDedup::kLlMaxCoreNum;
        return result;
    });
    if (deepep_ll_ranks_per_group() != DispatchDedup::kLlRanksPerGroup)
        throw std::runtime_error("LL Host/device group size mismatch; rebuild the complete wheel");
    module.attr("low_latency_ranks_per_group") = DispatchDedup::kLlRanksPerGroup;
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
        // A scoped diagnostic build must reject explicitly before the generic
        // capability-to-None path could select a different transport.
        require_small_dispatch_scope(b, b);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            result["scale_packs"] = (h + 127) / 128;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_comm_args", [](Session& session) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            session.world() > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
            session.world() < 16 || session.world() % DispatchDedup::kLlRanksPerGroup != 0) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        pin_session_cores(session);
        DispatchDedup::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % DispatchDedup::kLlRanksPerGroup;
        args.localRankSize = DispatchDedup::kLlRanksPerGroup;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = peer_buffer_base(session, peer);
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_dispatch_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype,
                                                 int num_max_tokens_per_rank) {
        require_small_dispatch_scope(b, num_max_tokens_per_rank);
        const bool fp8 = dtype == "float8_e4m3fn";
        auto checked=layout(p, b, h, k, e, fp8);
        if (!DispatchDedup::EpServerDedupEnabled(num_max_tokens_per_rank) &&
            !DispatchDedup::DispatchDedupPacketUbFits(checked,b,k,e,p,DispatchDedup::kLlDefaultPlanCoreNum,false))
            throw std::invalid_argument("Non-dedup R-only receive exceeds UB budget");
        auto data=dispatch_format_tiling(dtype);
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = 1;
        return bytes(data);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank) {
                   CombineDedup::Layout plan{};
                   if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan, query_ll_core_config().active)) {
                       throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
                   }
                   CombineDedup::EpServerDedupCombineTilingData data{};
                   data.bs = b;
                   data.h = h;
                   data.numMaxTokensPerRank = num_max_tokens_per_rank;
                   data.topK = k;
                   data.moeExpertNum = e;
                   data.dtype = dtype_code(dtype);
                   return bytes(data);
               });
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
        args.coreNum = pin_session_cores(session);
        args.referenceProfile = reference_profile(tensors, args.coreNum);
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        args.x = get("input", {b, h}, dtype.c_str());
        args.expertIds = get("indices", {b, k}, "int32");
        args.expertScales = get("weights", {b, k}, "float32");
        args.expandXOut = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamicScalesOut = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expertTokenNumsOut = get("expert_recv_counts", {e / session.world()}, "int64");
        args.sendCountsOut = get("send_counts", {e}, "int32");
        args.tokenTypeOut = get("token_type", {c}, "int32");
        args.destinationIndexOut = get("destination_index", {c, 3}, "int32");
        args.relayReadIndexOut = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMaskOut = get("source_mask", {b}, "int32");
        args.expertScalesOut = get("source_weights", {b, k}, "float32");
        args.tilingData = get("tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
        args.inputScales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        session.workspace();  // Validate the owning context before launch.
        const auto format=dispatch_format_tiling(dtype);
        const auto variant=DispatchDedup::LlDispatchVariantForTiling(format.dtype,format.quantMode,
                                                                    format.expandXOutDtype,format.inputMode);
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        dispatch_server_dedup_kernel_do_reference7(reinterpret_cast<void*>(stream_address), &args, variant);
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan, query_ll_core_config().active)) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        const int64_t c = plan.capacityRows;
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.coreNum = pin_session_cores(session);
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {b, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        combine_server_dedup_kernel_do_reference7(reinterpret_cast<void*>(stream_address), &args);
    });
}
}  // namespace deepep
