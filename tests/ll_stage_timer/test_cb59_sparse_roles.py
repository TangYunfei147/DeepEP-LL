"""Old dedicated senders remain distinct from current shared receive roles."""
import runpy
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
fixtures=runpy.run_path(str(ROOT/'tests/ll_stage_timer/test_sparse_timer.py'))
timer=fixtures['timer']

def replay_raw():
    data=fixtures['raw'](send=28)
    for core in range(28):
        flags=data[core][0][5]
        data[core][0][5]=(flags & ~(255<<16)) | (1<<16)
    return data

def test_cb59_dedicated_sender_roles_decode_and_trace():
    data=replay_raw()
    decoded=timer.parse_records(data,'dispatch',1e9)
    assert {row['role'] for row in decoded['cores'][:28]}=={'send_token'}
    assert {row['role'] for row in decoded['cores'][28:]}=={'cumsum'}
    trace=timer.chrome_trace([dict(raw=data,operation='dispatch',frequency_hz=1e9,
                                  metadata=fixtures['metadata']())])
    assert trace['traceEvents']

def test_mixed_sender_roles_rejected():
    data=replay_raw()
    data[1][0][5]=(data[1][0][5] & ~(255<<16)) | (3<<16)
    with pytest.raises(ValueError,match='Mixed sparse dispatch sender roles'):
        timer.parse_records(data,'dispatch')

def test_dedicated_sender_may_not_claim_receive_work():
    data=replay_raw()
    index=timer.POINTS.index('FIRST_RECV_PROBE')
    data[0][0][6] |= 1<<index
    fixtures['setword'](data,0,8+index,1000)
    with pytest.raises(ValueError,match='Dedicated sparse sender'):
        timer.parse_records(data,'dispatch')
