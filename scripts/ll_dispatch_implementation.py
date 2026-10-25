"""CPU-only identities for the two compile-time LL dispatch implementations.

The selector is a build input, not a runtime dispatch or a template dimension.
Both OFF and sparse wheels in one diagnosis must have the same selection.
"""
import os


IMPLEMENTATIONS = {
    "current": dict(implementation="g4-small-b-parallel-prefix", protocol_base=19000,
                    logical_group=4, fixed_logical_group=None, overlap_stage=3, local_q_table=True,
                    local_q_seed_vf=True, cb59=0),
    "cb59": dict(implementation="cb59-replay", protocol_base=21000,
                 logical_group=16, fixed_logical_group=16, overlap_stage=0, local_q_table=False,
                 local_q_seed_vf=False, cb59=1),
}


def dispatch_implementation(environment=None):
    env = os.environ if environment is None else environment
    value = env.get("DEEPEP_LL_DISPATCH_IMPL", "current")
    if value not in IMPLEMENTATIONS:
        raise ValueError("DEEPEP_LL_DISPATCH_IMPL must be current or cb59")
    return value


def dispatch_configuration(choice, send=28, recv=28, prefix=0):
    if choice not in IMPLEMENTATIONS:
        raise ValueError("DEEPEP_LL_DISPATCH_IMPL must be current or cb59")
    if any(type(value) is not int for value in (send, recv, prefix)) or min(send, recv) <= 0 or send + recv != 56:
        raise ValueError("DEEPEP_LL_SEND_CORES and DEEPEP_LL_RECV_CORES must be positive and sum to 56")
    if not 0 <= prefix <= recv:
        raise ValueError("DEEPEP_LL_SMALL_B_PREFIX_CORES must be 0 or an integer in [1, RECV_CORES]")
    if choice == "cb59" and (send, recv, prefix) != (28, 28, 0):
        raise ValueError("cb59 requires DEEPEP_LL_SEND_CORES=28, DEEPEP_LL_RECV_CORES=28 and DEEPEP_LL_SMALL_B_PREFIX_CORES=0")
    result = dict(IMPLEMENTATIONS[choice], dispatch_impl=choice)
    result["core_protocol"] = result["protocol_base"] + send
    return result


def dispatch_group_valid(choice, group, world=None):
    """Current retains its explicit compile-time G override; cb59 fixes G16."""
    if choice not in IMPLEMENTATIONS or type(group) is not int or not 1 <= group <= 16:
        return False
    fixed = IMPLEMENTATIONS[choice]['fixed_logical_group']
    if fixed is not None and group != fixed:
        return False
    return world is None or (type(world) is int and world > 0 and world % group == 0)
