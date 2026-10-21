"""Non-dedup partition and branch isolation contracts."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'csrc/kernels/dispatch/low_latency'
def method(text,name):
    a=text.index('__aicore__ inline void '+name+'(')
    o=text.index('{',a);i=o+1;depth=1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[a:i]
def test_sender_is_main_loop_and_dedup_unchanged():
    sender=(BASE/'device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    old=subprocess.check_output(['git','-C',str(ROOT),'show','4ec82a5:csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h'],encoding='utf-8')
    assert method(sender,'SendWithDedup')==method(old,'SendWithDedup')
    assert method(sender,'ProcessGroupLocal')==method(old,'ProcessGroupLocal')
    body=method(sender,'SendWithoutDedup')
    assert 'GatherMask' not in body and 'ownBegin' not in body
    assert 'const uint32_t serverNum = args_->rankSize / 8U;' in body
    assert 'for (uint32_t i = begin; i < begin + count; ++i)' in body
def test_mode_gates_and_receive_budget():
    kernel=(BASE/'dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    branch=kernel.split('if (!enableDedup_) {',1)[1].split('if (overlap_.Cooperative()',1)[0]
    send=branch.split('}')[0]
    assert 'counts_.CompleteCumsum' not in send and 'receiver_.' not in send
    assert 'sender_.Retire()' in send and 'return;' in send
    assert 'ProcessGroupLocal' not in branch
    counts=(BASE/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'if (buildLocalTasks_ && range.local)' in counts
    assert 'if (buildLocalTasks_ && countPlan_.local && countPlan_.count)' in counts
    for file in ('common/ll_overlap_layout.h','common/ep_memory_server_dedup_layout.h'):
        assert 'receiveCores=enableDedup?kLlDispatchRecvCoreNum:kLlRecvCoreNum' in (BASE/file).read_text(encoding='utf-8')
@pytest.mark.parametrize('send',[1,10,28,55])
def test_all_routes_once_and_only_r_receives(send):
    for b in (1,2,8,16,32):
      for k in (1,3,8,32):
        routes=b*k
        sent=[i for c in range(send) for i in range(c*(routes//send)+min(c,routes%send),
            c*(routes//send)+min(c,routes%send)+routes//send+(c<routes%send))]
        assert sent==list(range(routes))
        r=56-send
        for total in (0,1,7,routes*16):
            received=[row for c in range(r) for row in range(c*total//r,(c+1)*total//r)]
            assert received==list(range(total))
