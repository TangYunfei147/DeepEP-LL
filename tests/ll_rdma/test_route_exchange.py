"""Guard against reintroducing the retired two-host paired-route path."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def test_dispatch_no_longer_uses_paired_route_exchange():
    device = ROOT / 'csrc/kernels/dispatch/low_latency/device'
    code = (device / 'dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert not (device / 'll_route_exchange.h').exists()
    assert not (device.parent / 'common/ll_route_exchange_layout.h').exists()
    assert 'll_route_exchange.h' not in code and 'routeExchange_' not in code
    assert 'directRanks_ = LlTransport::Rdma(args_) && epWorldSize_ > kLlPhysicalRanksPerHost' in code
    assert 'epRankId_, localExpert, directRanks_' in code
    assert 'epRankId_, source, expert, directRanks_' in code
    assert 'readPlan, directRanks_' in code
