#!/usr/bin/env python3
"""8 machines x 8 NPUs: BF16 -> FP8 dispatch preparation, then BF16 combine profiling.

On each machine (node-rank 0..7):
  python test_moe_combine_v2_perf_2.py -c hosts.yaml --node-rank 0
Requires Ascend 950, matching CANN/ops-transformer and cann_ops_transformer.
Kimi K3 routed-expert communication: H=3584, top-k=16, 896 experts,
14 routed experts per rank at EP64, no shared experts; 8 tokens per rank.
Profiler: original 10 warmup / 100 capture calls, with Exp + scalar AllReduce
before every capture call; per-rank trace under --out-dir.
Shape source: https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
Interface checked against ops-transformer 3c86d19c6c9e34bdfb790195ae43c7c9d041bdd4:
  mc2/common/torch_extension/moe_distribute_buffer.py
  mc2/common/docs/torchapi_low_latency_dispatch.md
  mc2/common/docs/torchapi_low_latency_combine.md
"""
import argparse
import os
from pathlib import Path

WORLD_SIZE = 64
LOCAL_WORLD_SIZE = 8
HIDDEN_DIM = 3584  # Kimi K3 routed_expert_hidden_size (latent MoE)
TOPK = 16
NUM_EXPERTS = 896
TOKENS = 8
CAPTURE_STEPS = 100
COMM_ALG = "fullmesh_v1"
# CombineV5/V3 on this SoC accepts the default empty string or "ccu".
# Keep the reference's default MTE combine; dispatch still uses fullmesh_v1.
COMBINE_COMM_ALG = ""
# at::ScalarType::Float8_e4m3fn; accepted by the extension's GetAclDataType.
FP8_E4M3FN = 24


def profile_operator(operator, args, rank, label):
    """预热 10 次，采集 100 次 Exp/AllReduce/算子；导出原始 trace。"""
    import torch
    import torch.distributed as dist
    import torch_npu

    trace_dir = args.out_dir / "profile" / f"rank_{rank:03d}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    delay = torch.zeros(118260, 30720, dtype=torch.float32, device="npu")
    tmp = torch.ones(1, dtype=torch.float32, device="npu")
    dist.all_reduce(tmp, op=dist.ReduceOp.SUM, async_op=False)
    with torch_npu.profiler.profile(
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
        profile_memory=True,
        with_stack=False, with_flops=False, with_modules=False,
        # 对齐 ep_test get_profiling_config
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as profiler:
        for _ in range(10):
            output = operator()
        torch_npu.npu.synchronize()
        profiler.step()
        for _ in range(CAPTURE_STEPS):
            torch.exp(delay, out=delay)
            tmp.fill_(1)
            with torch.profiler.record_function("moe_combine_v2_operator"):
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM, async_op=False)
                output = operator()
        dist.barrier()
        torch_npu.npu.synchronize()
        profiler.step()
    print(f"rank={rank} profile -> {trace_dir}", flush=True)
    return output



def worker(local_rank, args):
    import torch
    import torch.distributed as dist
    import torch_npu
    from cann_ops_transformer.ops import MoeDistributeBuffer

    rank = args.node_rank * LOCAL_WORLD_SIZE + local_rank
    torch_npu.npu.set_device(local_rank)
    buffer_size_mib = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
        world_size=WORLD_SIZE,
        num_max_dispatch_tokens_per_rank=TOKENS,
        hidden=HIDDEN_DIM, num_moe_expert=NUM_EXPERTS, topk=TOPK,
        num_shared_expert=0, num_shared_expert_ranks=0, comm_alg=COMM_ALG,
    )
    # KFC obtains the real HCCL window; customCclBufferSize does not resize it.
    # Configure before any communicator is initialized, including the EP group.
    hccl_size_mib = max(200, buffer_size_mib, int(os.environ.get("HCCL_BUFFSIZE", "200")))
    os.environ["HCCL_BUFFSIZE"] = str(hccl_size_mib)
    dist.init_process_group(
        backend="hccl", rank=rank, world_size=WORLD_SIZE,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    # Dedicated EP group: dispatch/combine must use the same group.
    ep_group = dist.new_group(list(range(WORLD_SIZE)), backend="hccl")
    # The extension queries with init_comm=False. Materialize the HCCL
    # communicator before constructing its context (new_group is lazy).
    if hasattr(ep_group, "get_hccl_comm_name"):
        ep_group.get_hccl_comm_name(rank, init_comm=True)
    else:
        ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(
            rank, init_comm=True,
        )
    # The helper returns HCCL_BUFFSIZE in MiB (half the complete window).
    # Ascend950 channel customCclBufferSize and operator ccl_buffer_size are bytes.
    window_bytes = 2 * buffer_size_mib * 1024 * 1024
    buffer = MoeDistributeBuffer(ep_group, ccl_buffer_size=window_bytes)
    print(f"rank={rank} device={torch_npu.npu.get_device_name(local_rank)} "
          f"BS={TOKENS} HCCL_BUFFSIZE={hccl_size_mib}MiB "
          f"context_window={buffer.ccl_buffer_size}B", flush=True)
    torch.manual_seed(rank)
    device = torch.device(f"npu:{local_rank}")
    x = (torch.randn(TOKENS, HIDDEN_DIM) * 0.05).to(
        device=device, dtype=torch.bfloat16,
    )
    # Same balanced global routing as the reference, constructing this rank only.
    route = torch.arange(rank * TOKENS * TOPK, (rank + 1) * TOKENS * TOPK)
    ids = ((route % NUM_EXPERTS + route // NUM_EXPERTS) % NUM_EXPERTS)
    ids = ids.reshape(TOKENS, TOPK).to(device=device, dtype=torch.int32)

    from moe_target_signature import make_dispatch, check_dispatch, make_combine
    active = torch.ones(TOKENS, dtype=torch.bool, device=device)
    weights = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=device)
    dispatch = make_dispatch(torch, buffer, x, ids, weights, active, rank, args.quant_mode)
    outputs = dispatch()
    torch_npu.npu.synchronize()
    check_dispatch(torch, buffer.context, outputs)
    # Synthetic expert output: communication benchmark, not a numerical roundtrip.
    # E8M0 bytes cannot use the old per-token float-scale dequantization.
    expert_x = torch.randn(7168, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    torch_npu.npu.synchronize()
    combine = make_combine(torch, buffer, expert_x, ids, weights, active, outputs, rank)
    probe = combine()
    torch_npu.npu.synchronize()
    if tuple(probe.shape) != (8, 3584) or probe.dtype != torch.bfloat16:
        raise RuntimeError(f'Combine signature mismatch: {tuple(probe.shape)} {probe.dtype}')

    output = profile_operator(combine, args, rank, "moe_combine_v2_operator")
    torch_npu.npu.synchronize()
    print(f"rank={rank} combine: {output.dtype}, shape={tuple(output.shape)}",
          flush=True)

    dist.barrier()  # default group, separate from EP
    del combine
    del buffer
    dist.destroy_process_group(ep_group)
    dist.destroy_process_group()


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="YAML ips: list of the eight machine addresses")
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument('--quant-mode', type=int, choices=(4, 5), default=4,
                        help='4=MX, 5=MX clip; both match target shapes/dtypes')
    parser.add_argument("--master-port", type=int, default=29600)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("moe_combine_fp8_profile_out"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    config_path = args.config if args.config.is_absolute() else root / args.config
    with config_path.open(encoding="utf-8") as stream:
        args.master_addr = yaml.safe_load(stream)["ips"][0]
    # Keep runtime logs/JIT cache under the project, independent of launch cwd.
    runtime = root / ".moe_fp8_runtime" / f"node_{args.node_rank}"
    for name, leaf in (("ASCEND_WORK_PATH", "ascend"),
                       ("ASCEND_CACHE_PATH", "cache"),
                       ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                       ("TMPDIR", "tmp")):
        path = runtime / leaf
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=LOCAL_WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
