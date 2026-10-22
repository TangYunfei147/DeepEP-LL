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

def test_native_tiny_q_and_empty_owner_equivalence(tmp_path):
    cc=shlex.split(os.environ.get('CXX',''))
    if not cc:
        found=shutil.which('g++') or shutil.which('clang++')
        if not found:pytest.skip('C++ compiler required')
        cc=[found]
    exe=tmp_path/('small.exe' if os.name=='nt' else 'small')
    subprocess.run([*cc,'-std=c++17','-O2','-UNDEBUG','-iquote',str(ROOT),
        str(Path(__file__).with_name('fixtures')/'small_b.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_unchanged_dedup_and_protocol_regions():
    manifest=json.loads(Path(__file__).with_name('small_b_dedup_baseline.json').read_text())
    normalize=runpy.run_path(str(Path(__file__).with_name('test_segment_source.py')))['normalize']
    for entry in manifest['regions']:
        text=(ROOT/entry['path']).read_text(encoding='utf-8')
        if entry.get('begin'):text=text[text.index(entry['begin']):text.index(entry['end'])]
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
