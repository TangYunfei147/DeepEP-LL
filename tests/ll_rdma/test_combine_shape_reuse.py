"""Reproduce B1 partial stage aliasing B2 token-1 result flags."""
import struct
from pathlib import Path

def test_b1_stage_pollutes_b2_inactive_slot():
    h,k=3584,8
    blocks=(h*2+479)//480
    packet=blocks*512
    result=bytearray(2*k*packet)
    old_stage=k*packet
    # Legal BF16 payload pairs in the previous B1 partial stage.
    result[old_stage:old_stage+2*h]=b'\x80\x3f' * h
    # In the next B2 invocation token 1 expects only slot 1; slot 0 is inactive.
    for block in range(blocks):
        struct.pack_into('<f',result,(k+1)*packet+block*512+480,1.0)
    def probe():
        return sum(struct.unpack_from('<f',result,(k+slot)*packet+block*512+480)[0]
                   for slot in range(k) for block in range(blocks))
    assert probe()!=float(blocks)  # old code polls forever
    result[:]=bytes(len(result))  # pre-publication clear on every rank
    for block in range(blocks):
        struct.pack_into('<f',result,(k+1)*packet+block*512+480,1.0)
    assert probe()==float(blocks)

def test_dispatch_invalidates_prepare_cache():
    root=Path(__file__).resolve().parents[2]
    s=(root/'deep_ep/buffers/elastic.py').read_text(encoding='utf-8')
    dispatch=s.split('def _dispatch_low_latency',1)[1].split('def _combine_low_latency',1)[0]
    assert 'runtime.combine_window_layout = None' in dispatch
    native=(root/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'bank < 2' in native and 'aclrtMemset(result, bytes, 0, bytes)' in native
