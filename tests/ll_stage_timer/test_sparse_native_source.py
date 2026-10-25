"""Source invariants for sparse recorder isolation and unchanged business tokens."""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
COMMON=ROOT/'csrc/kernels/dispatch/low_latency/common'

def test_sparse_recorder_has_no_dynamic_stage_lookup_or_accumulator():
    source=(COMMON/'ll_stage_timer_device.h').read_text(encoding='utf-8')
    for removed in ('ProfileIndex','ProfileEnabled','sums_[','calls_[','events_[','cycles_[','SyncAll','WaitFlag','PipeBarrier'):
        assert removed not in source
    assert 'template<Point point>' in source
    assert 'ticks_[kPointCount]' in source
    assert 'offset<kExportWords;offset+=8U' in source
    for macro in ('SCOPE','BEGIN','END','COUNT','RELAY_COMPLETION','FIRST'):
        assert re.search(r'#define LL_SEG_'+macro+r'\([^\n]*\) do \{\} while\(false\)',source)

def test_only_sparse_buffers_and_compatible_linkage_are_exposed():
    layout=(COMMON/'ll_stage_timer_layout.h').read_text(encoding='utf-8')
    assert 'kSchema = 15U' in layout and 'kSlots = 8U' in layout
    assert 'kPointCount < 64U' in layout
    host=(ROOT/'csrc/ops/low_latency.cpp').read_text(encoding='utf-8')
    assert 'deepep_ll_stage_timer_schema()!=LlStageTimer::kSchema' in host
    assert 'deepep_ll_stage_timer_profile_id()!=LlStageTimer::kProfileId' in host
    assert 'Sparse LL timer supports observe only' in host
    assert 'result["recording"] = "sparse-milestones"' in host
    assert 'result["points"] = points' in host
    for path in ['csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h','csrc/kernels/combine/low_latency.cpp']:
        text=(ROOT/path).read_text(encoding='utf-8')
        assert 'SyncAll' not in text
        assert text.index('sparseEntry =') < text.index('DeviceTimer timer') < text.index('timer.Init(')
        assert 'timer.MarkAt<LlStageTimer::ENTRY>(sparseEntry)' in text
        assert 'timer.Mark<LlStageTimer::INIT_DONE>()' in text
        assert text.index('timer.Mark<LlStageTimer::WORK_DONE>()') < text.index('timer.Flush()')

def test_receive_send_count_logic_tokens_match_pre_sparse_commit():
    baseline=json.loads((Path(__file__).with_name('sparse_business_baseline.json')).read_text(encoding='utf-8'))
    for rel,expected in baseline['files'].items():
        if rel.endswith('/dispatch_parallel_count_prefix.h'):
            # The P-only candidate deliberately changes this implementation.
            # Keep the historical hash untouched; actual helper/VF/generation
            # tests in test_parallel_prefix guard default P=R equivalence.
            continue
        source=(ROOT/rel).read_text(encoding='utf-8')
        source=re.sub(r'//[^\n]*|/\*.*?\*/','',source,flags=re.S)
        source=re.sub(r'LL_SPARSE_(?:MARK|AFTER)\([^)]*\);','',source)
        tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|0[xX][0-9a-fA-F]+|\d+|[^\s]',source)
        assert hashlib.sha256(' '.join(tokens).encode()).hexdigest()==expected,rel


def test_send_done_requires_existing_scalar_visible_completion():
    device=ROOT/'csrc/kernels/dispatch/low_latency/device'
    send=(device/'dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    retire=send[send.index('inline void Retire()'):send.index('inline void ConfigureOverlap')]
    assert 'WaitFlag<HardEvent::MTE3_V>' in retire
    assert 'SEND_DONE' not in retire
    counts=(device/'dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    finish=counts[counts.index('inline void Finish()'):counts.index('private:')]
    assert finish.index('PipeBarrier<PIPE_ALL>();') < finish.index('LL_SPARSE_AFTER(FIRST_SEND,SEND_DONE);')
    # Non-async sender records may keep their earlier existing MTE3_S completion.
    process=send[send.index('inline void Process('):send.index('struct NoCooperate')]
    assert re.search(r'if \(!overlap_\.Async\(\)\) \{ LL_SEG_SCOPE\(SEND_COMPLETE_WAIT\);\s*'
                     r'Sync<HardEvent::MTE3_S>\(\);\s*LL_SPARSE_AFTER\(FIRST_SEND,SEND_DONE\);',process)
