"""Execute production q builder against independent full-prefix reference."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize("fixture",["local_q.cpp","local_q_layout.cpp"])
def test_native_q_equivalence(tmp_path,fixture):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('C++ compiler needed for production q helper')
        cc=[found]
    exe=tmp_path/('q.exe' if os.name=='nt' else 'q')
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-iquote',str(ROOT),
                    str(Path(__file__).parent/'fixtures'/fixture),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_q_table_lifetime_and_protocol():
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert sender.index('LlBuildLocalQ(')<sender.index('BeginInputPrefetch();')
    method=sender[sender.index('inline uint32_t CalTokenSendExpertCnt('):sender.index('template <class Fp8Type>')]
    assert 'qTable_.GetValue(index-qShard_.begin)' in method
    assert 'ReduceSum' not in method and 'Sync<' not in method
    quant=sender[sender.index('inline void QuantMxfp8('):sender.index('inline void FillTail(')]
    assert 'qWorkBuf_' not in quant and 'qTable_' not in quant and 'qSubBuf_' not in quant
    assert 'qTable_=qWorkBuf_.Get<uint32_t>()' in sender
    assert 'LlLocalQShard(bs_,k_,sendCores_,GetBlockIdx(),enableDedup_)' in sender
    core=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_core_config.h').read_text()
    assert '18000U + kLlSendCoreNum' in core
