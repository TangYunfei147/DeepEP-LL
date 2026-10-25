"""Execute production q builder against independent full-prefix reference."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest
import runpy
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize("fixture",["local_q.cpp","local_q_layout.cpp"])
def test_native_q_equivalence(tmp_path,fixture):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('C++ compiler needed for production q helper')
        cc=[found]
    api=tmp_path/'c_api';api.mkdir()
    api.joinpath('asc_simd.h').write_text(runpy.run_path(str(ROOT/'tests/elastic/test_ll_frequency_histogram.py'))['MOCK'],encoding='utf-8')
    exe=tmp_path/('q.exe' if os.name=='nt' else 'q')
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-I',str(tmp_path),'-iquote',str(ROOT),
                    str(Path(__file__).parent/'fixtures'/fixture),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_q_table_lifetime_and_protocol():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert sender.index('LlFinishLocalQ(')<sender.index('BeginInputPrefetch();')
    method=sender[sender.index('inline uint32_t CalTokenSendExpertCnt('):sender.index('template <class Fp8Type>')]
    assert 'qTable_.GetValue(index-qShard_.begin)' in method
    assert 'ReduceSum' not in method and 'Sync<' not in method
    quant=sender[sender.index('inline void QuantMxfp8('):sender.index('inline void FillTail(')]
    assert 'qWorkBuf_' not in quant and 'qTable_' not in quant and 'qSubBuf_' not in quant
    assert 'qTable_=qWorkBuf_.Get<uint32_t>()' in sender
    assert 'LlLocalQShard(bs_,k_,sendCores_,GetBlockIdx(),enableDedup_)' in sender
    core=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_core_config.h').read_text()
    assert '(kLlCb59Stream ? 22000U : (kLlCb59Replay ? 21000U : 19000U)) + kLlSendCoreNum' in core

def test_vf_seed_is_complete_before_scalar_updates():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    seed=sender[sender.index('if (qShard_.count)'):sender.index('// 每个 512B')]
    assert 'CountFrequency<true>' in seed and 'CountFrequency<false>' in seed
    assert seed.index('Sync<HardEvent::S_V>()')<seed.index('asc_vf_call')
    assert seed.rindex('asc_vf_call')<seed.index('Sync<HardEvent::V_S>()')<seed.index('LlFinishLocalQ')
    helper=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_local_q.h').read_text(encoding='utf-8')
    assert 'i<shard.begin' not in helper
