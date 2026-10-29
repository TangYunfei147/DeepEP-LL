from pathlib import Path
import os
import random
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

# The paired-route primitive below remains a historical regression only;
# protocol27 no longer includes or calls it from dispatch counts.
def test_dispatch_no_longer_uses_paired_route_exchange():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'll_route_exchange.h' not in code and 'routeExchange_' not in code
    assert 'directRanks_ = LlTransport::Rdma(args_) && epWorldSize_ == 32U' in code
    assert 'epRankId_, localExpert, directRanks_' in code
    assert 'epRankId_, source, expert, directRanks_' in code
    assert 'readPlan, directRanks_' in code


def test_production_exchange_with_autonomous_network_and_delays(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU C++ validation')
    exe=tmp_path/'route-exchange.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG','-pthread',
        '-I'+str(ROOT/'tests/ll_rdma/stubs'),'-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('route_exchange_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=60)
    assert 'PASS paired route exchange' in result.stdout
    assert '1 put / 1 quiet' in result.stdout
    assert 'alternating dispatch banks without rank barrier' in result.stdout
    assert 'existing combine completion boundary' in result.stdout
    assert 'PASS FinishSend retires outbound SO frame without peer or local-reader join' in result.stdout
    assert 'PASS CQ failure retains source even after SO readiness' in result.stdout
    print(result.stdout)

@pytest.mark.parametrize('b,k,e,cores',[(1,8,384,56),(2,8,384,56),(32,8,384,56),
    (256,32,1024,72),(1,1,32,72),(512,2,384,56),(1024,8,1024,72)])
@pytest.mark.parametrize('route',['random','one_host'])
def test_gateway_counts_preserve_source_prefixes(b,k,e,cores,route):
    rng=random.Random(13)
    routes=[[rng.randrange(e if route=='random' else e//2) for _ in range(b*k)] for _ in range(32)]
    local=e//32
    count_cores=cores if b<512 else cores//2
    counts={}
    for gateway in range(32):
        host_first=gateway//16*(e//2)
        host_end=host_first+e//2
        for source in (gateway,gateway^16):
            for core in range(count_cores):
                base,extra=divmod(e,count_cores)
                start=core*base+min(core,extra)
                end=start+base+(core<extra)
                for expert in range(max(start,host_first),min(end,host_end)):
                    key=(source,expert)
                    assert key not in counts
                    counts[key]=routes[source].count(expert)
    assert len(counts)==32*e  # includes explicit zeros from every source
    for destination in range(32):
        prefix=0
        for expert in range(destination*local,(destination+1)*local):
            for source in range(32):
                assert counts[source,expert]==routes[source].count(expert)
                prefix+=counts[source,expert]
        assert prefix==sum(sum(destination*local<=v<(destination+1)*local for v in row) for row in routes)
