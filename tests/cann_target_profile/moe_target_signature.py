"""Native seven-output dispatch contract for the supplied EP64 trace."""

def make_dispatch(torch, buffer, x, ids, weights, active, rank, quant_mode):
    def dispatch():
        return torch.ops.cann_ops_transformer.npu_moe_distribute_dispatch(
            context=buffer.context, x=x, expert_ids=ids,
            ep_world_size=64, ep_rank_id=rank, moe_expert_num=896,
            ccl_buffer_size=buffer.ccl_buffer_size,
            scales=None, x_active_mask=active, expert_scales=weights,
            elastic_info=None, performance_info=None,
            tp_world_size=1, tp_rank_id=0,
            shared_expert_num=0, shared_expert_rank_num=0,
            quant_mode=quant_mode, y_dtype=24, expert_token_nums_type=1,
            global_bs=512, comm_alg="fullmesh_v1",
        )
    return dispatch


def check_dispatch(torch, context, outputs):
    shapes = ((7168, 3584), (7168, 112), (917504,), (14,), (131968,), (1,), (7168,))
    # The extension wraps uint8 storage as ACL_FLOAT8_E8M0, not ordinary UINT8.
    dtypes = (torch.float8_e4m3fn, torch.uint8, torch.int32, torch.int64,
              torch.int32, torch.int32, torch.float32)
    errors = []
    if tuple(context.shape) != (4100,) or context.dtype != torch.int32:
        errors.append(f'context: {tuple(context.shape)} {context.dtype}')
    if len(outputs) != 7:
        errors.append(f'expected 7 outputs, got {len(outputs)}')
    for i, (value, shape, dtype) in enumerate(zip(outputs, shapes, dtypes)):
        if tuple(value.shape) != shape or value.dtype != dtype:
            errors.append(f'output[{i}]: {tuple(value.shape)} {value.dtype}; expected {shape} {dtype}')
    if errors:
        raise RuntimeError('Target signature mismatch before profiling: ' + '; '.join(errors)
                           + '. A matching CANN/cann_ops_transformer build is required. '
                           'Inspected extension 3c86d19 allocates only 896 ep_recv_counts '
                           'on Ascend950. Do not pad this tensor after the operator.')


def make_combine(torch, buffer, x, ids, weights, active, outputs, rank):
    _, _, assist, _, ep_counts, tp_counts, expand_scales = outputs
    def combine():
        return torch.ops.cann_ops_transformer.npu_moe_distribute_combine(
            context=buffer.context, expand_x=x, expert_ids=ids,
            assist_info_for_combine=assist, ep_send_counts=ep_counts,
            expert_scales=weights, ep_world_size=64, ep_rank_id=rank,
            moe_expert_num=896, ccl_buffer_size=buffer.ccl_buffer_size,
            tp_send_counts=tp_counts, x_active_mask=active,
            expand_scales=expand_scales, shared_expert_x=None, elastic_info=None,
            ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None,
            const_expert_v=None, performance_info=None,
            tp_world_size=1, tp_rank_id=0,
            shared_expert_num=0, shared_expert_rank_num=0,
            global_bs=512, comm_alg="", comm_quant_mode=0,
        )
    return combine
