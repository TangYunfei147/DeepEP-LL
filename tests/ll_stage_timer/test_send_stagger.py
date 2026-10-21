"""Scheduling invariants, not device execution or a prediction of speedup."""
from collections import Counter
from pathlib import Path
import random
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def original_group_order(groups, own, step):
    if groups < 16:
        return (own + step) % groups
    split = groups // 2
    base = 0 if own < split else split
    length = split if own < split else groups - split
    relative = own - base
    ordinal = step // 2 + (0 if step % 2 == 0 else length)
    if ordinal < length:
        return base + (relative + ordinal + 1) % length
    other_base = split if own < split else 0
    other_length = groups - length
    return other_base + (relative + ordinal - length + 1) % other_length


def rotated(count, phase):
    return [(step + phase) % count for step in range(count)]


def schedule(routes, rank, world, group, experts, send_cores, stagger):
    """Independent route model: q is assigned in original route order, never send order."""
    local = experts // world
    seen, q_slots = Counter(), {}
    for token, ids in enumerate(routes):
        for k, expert in enumerate(ids):
            q_slots[token, k] = seen[expert]
            seen[expert] += 1
    result = []
    for core in range(send_cores):
        base, extra = divmod(len(routes), send_cores)
        begin = core * base + min(core, extra)
        count = base + (core < extra)
        first, direct = [], []
        masks, weights = {}, {}
        for token in range(begin, begin + count):
            ids = routes[token]
            by_group = {}
            for k, expert in enumerate(ids):
                by_group.setdefault(expert // local // group, []).append(k)
            hops = []
            for target, members in by_group.items():
                if target != rank // group and len(members) > 1:
                    first_k = members[0]
                    mask = sum(1 << k for k in members)
                    hops.append((token, first_k, ids[first_k], q_slots[token, first_k], mask))
                    masks[token] = masks.get(token, 0) | 1 << first_k
                    weights[token, first_k] = 1.0
                else:
                    for k in members:
                        direct.append((token, k, ids[k], q_slots[token, k], 0))
                        masks[token] = masks.get(token, 0) | 1 << k
                        weights[token, k] = (k + 1) / 32.0
            # Production Classify emits groups when their second member is seen.
            hops.sort(key=lambda packet: [k for k in range(len(ids)) if packet[4] >> k & 1][1])
            order = rotated(len(hops), rank % group + core + token) if stagger else range(len(hops))
            first.extend(hops[index] for index in order)
        direct.sort(key=lambda packet: packet[:2])
        groups = world // group
        steps = rotated(groups, rank % group + core) if stagger else range(groups)
        final_direct = [packet for step in steps for packet in direct
                        if packet[2] // local // group == original_group_order(groups, rank // group, step)]
        result.append((first, final_direct, masks, weights))
    return result


@pytest.mark.parametrize('world,group,b,k,experts,cores', [
    (16, 4, 256, 8, 384, 28), (16, 4, 256, 8, 384, 29), (16, 4, 256, 8, 384, 20), (16, 4, 256, 8, 384, 40), (16, 4, 1, 8, 384, 36),
    (16, 4, 65, 16, 384, 8), (32, 16, 64, 8, 384, 28),
    (64, 16, 33, 1, 512, 36), (128, 16, 9, 16, 512, 8),
    (256, 16, 17, 8, 1024, 28),
    (32, 4, 64, 8, 384, 28), (64, 4, 64, 8, 512, 28),
    (128, 4, 17, 8, 512, 28), (256, 4, 17, 8, 1024, 28),
])
def test_routes_slots_masks_weights_and_byte_totals_unchanged(world, group, b, k, experts, cores):
    rng = random.Random(world + b + k)
    # Uniform, skewed and repeated-across-token inputs; experts within a token remain distinct.
    for mode in ('uniform', 'hot_group', 'same_routes'):
        pool = range(experts if mode == 'uniform' else max(k, experts // world * group))
        fixed = rng.sample(list(pool), k)
        routes = [fixed[:] if mode == 'same_routes' else rng.sample(list(pool), k) for _ in range(b)]
        for rank in sorted({0, 1, group - 1, world - 1}):
            before = schedule(routes, rank, world, group, experts, cores, False)
            after = schedule(routes, rank, world, group, experts, cores, True)
            for old, new in zip(before, after):
                assert Counter(old[0]) == Counter(new[0])  # original FIRST_HOP target, mask, q
                assert Counter(old[1]) == Counter(new[1])  # every DIRECT exactly once
                assert old[2:] == new[2:]                 # source mask and weights
                assert len(old[0] + old[1]) == len(new[0] + new[1])
                for target in range(world // group):
                    select = lambda packets: [p for p in packets if p[2] // (experts // world) // group == target]
                    assert select(old[1]) == select(new[1])  # stable within a DIRECT group


def test_ep16_direct_start_groups_spread_across_cards_and_cores():
    for own in range(4):
        for core in range(36):
            assert {original_group_order(4, own, (card + core) % 4) for card in range(4)} == set(range(4))
        for card in range(4):
            starts = Counter(original_group_order(4, own, (card + core) % 4) for core in range(28))
            assert set(starts.values()) == {7}
    for count in range(1, 9):
        assert sorted(rotated(count, 17)) == list(range(count))
        assert {rotated(count, token)[0] for token in range(count)} == set(range(count))


def test_compact_production_wiring_preserves_first_hop_and_lane_lifetime():
    source = (ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text()
    dedup = source.split('inline void SendWithDedup(', 1)[1].split('inline void SendWithoutDedup(', 1)[0]
    assert 'firstHopMasks_.GetValue(group)' in dedup
    assert 'sourceWeights_.SetValue(token*k_+firstK,1.0f)' in dedup
    assert 'SendCompactGroups(input,true)' in dedup
    assert 'WaitFlag' not in dedup and 'PipeBarrier' not in dedup
    nondedup = source.split('inline void SendWithoutDedup(', 1)[1].split('private:', 1)[0]
    assert 'SendCompactGroups(input,false)' in nondedup
    assert 'LlDirectGroupOrder(groups,own,phase,0U,step)' in nondedup
    assert 'LlVisitCompactGroup(' in nondedup
    assert nondedup.index('uint32_t owner=')<nondedup.index('for (uint32_t pass=')
    assert 'SyncAll' not in nondedup


def test_production_order_helpers_cpp(tmp_path):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable; Python invariants and source wiring only')
    source = tmp_path / 'orders.cpp'
    source.write_text(r'''
#include "kernels/dispatch/low_latency/common/ll_send_order.h"
#include <cassert>
#include <vector>
int main() {
    using namespace DispatchDedup;
    assert(LlFirstHopOrder(0, 0, 0, 0, 0) == 0);
    for (unsigned groups : {1U, 2U, 4U, 8U, 16U, 32U, 64U})
      for (unsigned own=0; own<groups; ++own)
        for (unsigned card=0; card<16; ++card)
          for (unsigned core=0; core<36; ++core) {
            std::vector<unsigned> seen(groups,0);
            for (unsigned step=0; step<groups; ++step) {
              unsigned target=LlDirectGroupOrder(groups,own,card,core,step);
              assert(target<groups); ++seen[target];
              assert(target == EpServerOrder::ServerDedupDispatchDirectOrder(groups,own,(step+card+core)%groups));
            }
            for (auto count:seen) assert(count==1);
          }
    for (unsigned count=1; count<=8; ++count)
      for (unsigned token=0; token<256; ++token) {
        std::vector<unsigned> seen(count,0);
        for (unsigned step=0; step<count; ++step) ++seen[LlFirstHopOrder(count,3,27,token,step)];
        for (auto n:seen) assert(n==1);
      }
}
''')
    binary = tmp_path / 'orders.exe'
    subprocess.run([compiler, '-std=c++17', '-I' + str(ROOT / 'csrc'), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
