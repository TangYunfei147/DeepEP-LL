"""CPU-only user-load reproducibility and quota invariants."""
from collections import Counter
import hashlib
import json
import random

import pytest
from tests.elastic import test_ep_low_latency as st


@pytest.mark.parametrize('ep,b,digest', [
    (64, 1, 'fcaf779f67128285c98883708a15dff91651e67ee40f32aa436375620984fa77'),
    (64, 8, '01c9f8828b8b8e5eb24b0b95266a42c91642d56310aee76a904dbc831a941d37'),
    (32, 1, '54a2fc1379a8fe1867678d8773d31e893482ed12afa80f7fe8009585454383fd'),
    (32, 8, 'd0c6f08941eac2fb2116144ce4229ad6e6c77b0b1f43dcf4ada44103b5e51323'),
])
def test_exact_supplied_generator_default_routes(ep, b, digest):
    # Golden hashes from the user's standalone generator, stride explicitly E/EP.
    nested = st.make_routes(ep, b, 8, 384, 0)
    flat = [row for source in nested for row in source]
    assert hashlib.sha256(json.dumps(flat, separators=(',', ':')).encode()).hexdigest() == digest


@pytest.mark.parametrize('ep,b,k,local', [(16, 1, 1, 1), (16, 8, 16, 24),
    (32, 256, 8, 12), (64, 1, 8, 6), (128, 3, 8, 3), (256, 2, 16, 4)])
@pytest.mark.parametrize('seed', [0, 2])
def test_exact_rank_quotas_distinct_destinations_and_roundrobin(ep, b, k, local, seed):
    routes = st.make_routes(ep, b, k, ep * local, seed)
    assert len(routes) == ep and all(len(rows) == b for rows in routes)
    counts, expert_counts = Counter(), Counter()
    for rows in routes:
        for row in rows:
            assert len(row) == len(set(expert // local for expert in row)) == k
            assert all(0 <= expert < ep * local for expert in row)
            for expert in row:
                counts[expert // local] += 1
                expert_counts[expert] += 1
    assert [counts[rank] for rank in range(ep)] == [b * k] * ep
    for rank in range(ep):
        values = [expert_counts[rank * local + expert] for expert in range(local)]
        assert max(values) - min(values) <= 1


def test_seed_offset_reproducibility_and_original_hotspot():
    assert st.make_routes(32, 8, 8, 384, 0) != st.make_routes(32, 8, 8, 384, 1)
    assert st.make_routes(32, 8, 8, 384, 1) == st.make_routes(32, 8, 8, 384, 1)
    rng = random.Random(7)
    expected = [[rng.sample(range(24), 8) for _ in range(2)] for _ in range(32)]
    assert st.make_routes(32, 2, 8, 384, 7, 'hotspot') == expected
    assert st.routing_metadata(32, 8, 384, 0, 'normal') == dict(
        generator='balanced-rank-distinct-roundrobin-v1', seed_offset=0,
        base_seed=20261014, rng_seed=20269016, physical_stride=12, hot_multiplier_x2=2)


@pytest.mark.parametrize('args', [(0,1,1,384), (16,0,8,384), (16,1,17,384), (32,1,8,383)])
def test_invalid_shapes_rejected(args):
    with pytest.raises(ValueError):
        st.make_routes(*args, 0)
