#!/usr/bin/env python3
"""8 machines x 8 NPUs: BF16 -> FP8 E4M3FN dispatch profiling.

On each machine (node-rank 0..7):
  python test_moe_dispatch_v2_perf_2.py -c hosts.yaml --node-rank 0
Requires Ascend 950 and matching torch_npu/CANN with MXFP8 v2 operators.
EP64, eight hosts x eight NPUs, BS8/H3584/topk16/E896.
Uses vllm-ascend TokenDispatcherWithMC2 group-name interfaces directly.
Source: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py
Warmup10/capture100, Exp + scalar AllReduce per capture call.

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


def profile_operator(operator, args, rank, label):
    """对齐 ep_test run_performance 的采集模式。"""
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
            with torch.profiler.record_function("moe_dispatch_v2_operator"):
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

    rank = args.node_rank * LOCAL_WORLD_SIZE + local_rank
    torch_npu.npu.set_device(local_rank)
    # HCCL owns the operator window; no custom buffer/context extension.
    os.environ["HCCL_BUFFSIZE"] = str(args.hccl_buffsize)
    dist.init_process_group(
        backend="hccl", rank=rank, world_size=WORLD_SIZE,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    # Dedicated EP group: dispatch/combine must use the same group.
    ep_group = dist.new_group(list(range(WORLD_SIZE)), backend="hccl")
    group_name = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print(f"rank={rank} BS={TOKENS} HCCL_BUFFSIZE={args.hccl_buffsize}MiB "
          f"comm_alg={args.comm_alg} dispatch_quant=4", flush=True)
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
    dispatch = make_dispatch(torch, torch_npu, group_name, x, ids, weights, active, rank, args.comm_alg)
    outputs = dispatch()
    torch_npu.npu.synchronize()
    check_dispatch(torch, outputs)

    outputs = profile_operator(dispatch, args, rank, 'moe_dispatch_v2_operator')
    torch_npu.npu.synchronize()


    dist.barrier()  # default group, separate from EP
    del dispatch
    dist.destroy_process_group(ep_group)
    dist.destroy_process_group()


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="YAML ips: list of the eight machine addresses")
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument('--comm-alg', choices=('', 'fullmesh', 'hierarchy', 'fullmesh_v2'),
                        default='hierarchy', help='vLLM hierarchy path passes expert weights')
    parser.add_argument('--combine-quant-mode', type=int, choices=(0, 2, 3, 4), default=0,
                        help='0 for BF16 communication; W8A8MXFP vLLM path uses 4')
    parser.add_argument('--hccl-buffsize', type=int, default=200,
                        help='HCCL communication window setting in MiB')
    parser.add_argument("--master-port", type=int, default=29600)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("moe_dispatch_fp8_profile_out"))
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
