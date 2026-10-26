"""Keep business statements and existing DMA/event ordering while replacing probes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pytest
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=json.loads(Path(__file__).with_name('segment_baseline.json').read_text())
def normalize(s):
 s=re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
 s=re.sub(r'#include [^\n]*ll_(?:stage|expert)_timer[^\n]*','',s)
 s=re.sub(r'if \(timer.Enabled\(\)\) \{ AscendC::SyncAll<true>\(\); timer.Flush\(\); \}','',s)
 s=re.sub(r'\bLL_SEG_(?:BEGIN|END|COUNT|SCOPE|FIRST|RELAY_COMPLETION)\([^;]*?\);','',s)
 s=re.sub(r'\bLL_SPARSE_(?:MARK|AFTER|WHEN)\([^;]*?\);','',s)
 s=re.sub(r'\bLL_(?:BW|MTE3)_[A-Z_]+\([^;]*?\);','',s)
 s=re.sub(r'#ifdef DEEPEP_LL_STAGE_TIMER\s*(?:const uint64_t sparseEntry =[^;]+;|timer\.MarkAt<LlStageTimer::ENTRY>\(sparseEntry\);)\s*#endif','',s)
 s=re.sub(r'timer\.(?:Mark<LlStageTimer::\w+>\(\)|Flush\(\));','',s)
 s=s.replace('timer.Enabled() ? &timer : nullptr','&timer')
 s=s.replace('combine_server_dedup_kernel_do_bw17','combine_server_dedup_kernel_do')
 s=re.sub(r'if \(timer_?\) timer_?->(?:Start|End)\([^;]*\);','',s)
 s=re.sub(r'(?:const uint64_t \w+ = )?timer\.(?:Start|End|Init|Finish|Begin)\([^;\n]*\);?','',s)
 s=re.sub(r'expert\.(?:Init|Finish)\([^;]*\);','',s)
 s=re.sub(r'Ll(?:Stage|Expert)Timer::DeviceTimer (?:timer|expert);','',s)
 s=re.sub(r'__aicore__ inline void SetTimer\([^)]*\) \{ timer_ = timer; \}','',s)
 s=re.sub(r'\w+\.SetTimer\(timer_\);','',s)
 s=re.sub(r'LlStageTimer::DeviceTimer\* timer_ = nullptr;|timer_ = timer;|\(void\)expertTimer;','',s)
 s=s.replace(', LlStageTimer::DeviceTimer* timer = nullptr','')
 s=s.replace('magic, pipe, timer)','magic, pipe)')
 s=s.replace('&timer, &expert','&timer, nullptr')
 s=s.replace('\\\n','')
 s=re.sub(r'kernel\.Process\(\);?', 'kernel.Process()', s)
 return re.findall(r'[A-Za-z_]\w*|\d+|[^\s{}]',s)

# Dispatch algorithms intentionally evolve on this branch; retain the immutable
# combine/entry baseline guard, not a misleading updated dispatch hash.
@pytest.mark.parametrize('relative',[p for p in MANIFEST['files'] if '/dispatch/low_latency/' not in p and not p.endswith('/dispatch/low_latency.cpp')])
def test_non_diagnostic_token_stream_matches_shared_receive_baseline(relative):
    tokens=normalize((ROOT/relative).read_text(encoding='utf-8'))
    assert hashlib.sha256(' '.join(tokens).encode()).hexdigest()==MANIFEST['files'][relative]


def test_explicit_boundaries_are_paired_and_use_reference_ids():
    names=set(json.loads(Path(__file__).with_name('dc_seg_stages.json').read_text()))
    for relative in MANIFEST['files']:
        source=(ROOT/relative).read_text(encoding='utf-8')
        begins=Counter(re.findall(r'LL_SEG_BEGIN\((\w+),\s*(\w+)\)',source))
        ends=Counter(re.findall(r'LL_SEG_END\((\w+),\s*(\w+)\)',source))
        assert begins==ends,relative
        assert {stage for _,stage in begins}<=names
        assert set(re.findall(r'LL_SEG_SCOPE\((\w+)\)',source))<=names


def test_no_gm_writes_or_new_barriers_at_measured_boundaries():
    device=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h').read_text()
    measured=device[device.index('void MarkAt('):device.index('void Flush()')]
    assert not any(word in measured for word in ('GlobalTensor','DataCopy','PipeBarrier','SyncAll','FlushLine','__gm__'))
    for relative in ('csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h','csrc/kernels/combine/low_latency.cpp'):
        source=(ROOT/relative).read_text()
        assert source.index('timer.Mark<LlStageTimer::WORK_DONE>()')<source.index('timer.Flush()')
        assert 'SyncAll' not in source


def test_legacy_source_labels_remain_but_compile_to_noops():
    counts=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    for name in ('POLL_RECV_COUNTS_ITERATIONS','WAIT_CUMSUM_ITERATIONS'):
        assert f'LL_SEG_COUNT({name},1U)' in counts.replace(', ',',')
    sender=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    for name in ('TOKEN_INPUT','TOKEN_QUANT','TOKEN_METADATA','TOKEN_BUFFER_WAIT','TOKEN_PACK','TOKEN_SEND_ISSUE','CAL_TOKEN','SEND_DRAIN'):
        assert name in sender
    recorder=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h').read_text(encoding='utf-8')
    for macro in ('SCOPE','BEGIN','END','COUNT','RELAY_COMPLETION','FIRST'):
        assert re.search(r'#define LL_SEG_'+macro+r'\([^\n]*\) do \{\} while\(false\)',recorder)
