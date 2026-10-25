"""Small direct routing equivalence and dedup boundary checks."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('fixture',['small_b.cpp','small_b_count.cpp'])
def test_native_tiny_q_and_empty_owner_equivalence(tmp_path,fixture):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('C++ compiler required')
        cc=[found]
    api=tmp_path/'c_api';api.mkdir()
    mock=runpy.run_path(str(ROOT/'tests/elastic/test_ll_frequency_histogram.py'))['MOCK']
    (api/'asc_simd.h').write_text(mock,encoding='utf-8')
    exe=tmp_path/('small.exe' if os.name=='nt' else 'small')
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-I',str(tmp_path),'-iquote',str(ROOT),
        str(Path(__file__).with_name('fixtures')/fixture),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_unchanged_dedup_and_protocol_regions():
    manifest=json.loads(Path(__file__).with_name('small_b_dedup_baseline.json').read_text())
    normalize=runpy.run_path(str(Path(__file__).with_name('test_segment_source.py')))['normalize']
    for entry in manifest['regions']:
        text=(ROOT/entry['path']).read_text(encoding='utf-8')
        if entry.get('begin'):
            start=text.index(entry['begin'])
            text=text[start:text.index(entry['end'],start)]
        if entry['path'].endswith('/ll_core_config.h'):
            # Restore only the explicit current/cb59 selector to its current
            # branch. Keep the original default-protocol digest unchanged;
            # test_cb59_native_identity executes both compiled alternatives.
            switches={
                '#include "ll_dispatch_implementation.h"\n':'',
                'static_assert(!kLlCb59Replay || (DEEPEP_LL_SEND_CORES == 28 && DEEPEP_LL_RECV_CORES == 28),\n'
                '              "cb59 replay requires the original 1:1 split on the fixed 56-core comparison launch");\n':'',
                'kLlCb59Replay ? kLlRecvCoreNum : kLlActiveCoreNum':'kLlActiveCoreNum',
                '(kLlCb59Stream ? 22000U : (kLlCb59Replay ? 21000U : 19000U)) + kLlSendCoreNum':'19000U + kLlSendCoreNum',
            }
            for selected,current in switches.items():
                assert text.count(selected)==1
                text=text.replace(selected,current)
        if entry['path'].endswith('/ll_overlap_layout.h'):
            # P-only deliberately changes the small-direct budget branch. The
            # historical digest/restoration is kept; all dedup code outside
            # that already-exempt branch must still match the old baseline.
            begin=text.index('    const bool parallelPrefix=')
            end=text.index('\n    const uint64_t maxRows=',begin)
            branch=text[begin:end]
            assert 'prefixProducers=parallelPrefix ? LlSmallBPrefixProducerCount(ranks):kLlRecvCoreNum' in branch
            prior=next(c['new'] for c in entry['restore'] if 'const bool parallelPrefix=' in c['new'])
            text=text[:begin]+prior+text[end:]
        for change in entry.get('restore',[]):
            assert text.count(change['new'])==1
            text=text.replace(change['new'],change['old'])
        digest=hashlib.sha256(' '.join(normalize(text)).encode()).hexdigest()
        assert digest==entry['sha256'],entry

def test_sender_small_gate_and_lane_cap():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert source.index('smallShard.count==0U')<source.index('LL_SEG_BEGIN(setupBegin')
    assert 'if (smallDirect && lanes_>smallShard.count)' in source
    assert 'else if (smallDirect && bs_<=4U)' in source
    assert 'const bool zeroQ=LlUseSmallDirect(bs_,enableDedup_) && bs_==1U;' in source
    # Main dedup/VF branch still follows the tiny direct-only branch.
    assert source.index('LlBuildTinyDirectQ(')<source.index('else if (qShard_.count)')

def test_count_gate_keeps_original_dedup_choice():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'if (LlUseSmallDirect(axisBS_,buildLocalTasks_)) CountSmallDirectRange' in source
    assert 'else if (buildLocalTasks_ && range.local) BuildLocalTasks' in source
    start=source.index('inline void ServerDedupCounts::CountSmallDirectRange(')
    end=source.index('inline void ServerDedupCounts::CountByVectorRange(',start)
    body=source[start:end]
    assert 'GatherMask(' not in body and 'Compares(' not in body
    assert body.index('SyncFunc<HardEvent::S_V>')<body.index('asc_vf_call')<body.index('SyncFunc<HardEvent::V_S>')
