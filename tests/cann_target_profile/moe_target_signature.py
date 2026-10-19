"""Native seven-output dispatch contract for the supplied EP64 trace."""

def make_dispatch(torch, torch_npu, group_name, x, ids, weights, active, rank, comm_alg):
    def dispatch():
        return torch_npu.npu_moe_distribute_dispatch_v2(
            x=x, expert_ids=ids, group_ep=group_name, group_tp=group_name,
            ep_world_size=64, ep_rank_id=rank, moe_expert_num=896,
            scales=None, x_active_mask=active, expert_scales=weights,
            tp_world_size=1, tp_rank_id=0,
            expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
            quant_mode=4, y_dtype=torch.float8_e4m3fn, expert_token_nums_type=0,
            global_bs=0, comm_alg=comm_alg,
        )
    return dispatch


def check_dispatch(torch, outputs):
    shapes = ((7168, 3584), (7168, 112), (917504,), (14,), (131968,), (1,), (7168,))
    # E8M0 may be exposed as byte storage by torch_npu; do not reinterpret it as FP32.
    dtypes = (torch.float8_e4m3fn, torch.uint8, torch.int32, torch.int64,
              torch.int32, torch.int32, torch.float32)
    errors = []
    if len(outputs) != 7:
        errors.append(f'expected 7 outputs, got {len(outputs)}')
    for i, (value, shape, dtype) in enumerate(zip(outputs, shapes, dtypes)):
        valid_dtype = value.dtype == dtype
        if i == 1:
            valid_dtype = value.dtype in (torch.uint8, getattr(torch, 'float8_e8m0fnu', torch.uint8))
        if tuple(value.shape) != shape or not valid_dtype:
            errors.append(f'output[{i}]: {tuple(value.shape)} {value.dtype}; expected {shape} {dtype}')
    if errors:
        raise RuntimeError('Target signature mismatch before profiling: ' + '; '.join(errors))


def make_combine(torch, torch_npu, group_name, x, ids, weights, active, outputs, rank, comm_alg, quant_mode):
    _, _, assist, _, ep_counts, tp_counts, expand_scales = outputs
    def combine():
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=x, expert_ids=ids, group_ep=group_name, group_tp=group_name,
            assist_info_for_combine=assist, ep_send_counts=ep_counts,
            expert_scales=weights, ep_world_size=64, ep_rank_id=rank,
            moe_expert_num=896,
            tp_send_counts=tp_counts, x_active_mask=active,
            expand_scales=expand_scales,
            tp_world_size=1, tp_rank_id=0,
            expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
            global_bs=0, comm_alg=comm_alg, comm_quant_mode=quant_mode,
        )
    return combine
