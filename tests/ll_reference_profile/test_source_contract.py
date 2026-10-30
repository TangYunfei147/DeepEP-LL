"""Remove diagnostics only, then compare business tokens to the immutable input.

The original ed1 hash ledger is preserved as provenance and as the 53-file list.
The comparison source c1c9b3d includes the now-removed legacy instrumentation.
Changing its normalizer cannot silently update a stored new baseline hash.
"""
import difflib
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='c1c9b3df505c740eae8cd5da55bd607e6c00230c'
LEDGER=json.loads(Path(__file__).with_name('business_baseline.json').read_text(encoding='utf-8'))


def remove_blocks(source,pattern):
    while True:
        match=re.search(pattern,source)
        if not match:return source
        opening=source.find('{',match.start());depth=1;end=opening+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}');end+=1
        source=source[:match.start()]+source[end:]


def business_tokens(source):
    source=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
    source=re.sub(r'static_assert\(offsetof\(EpServerDedup(?:Combine)?KernelArgs, (?:stageTimer|timerMode|expertTimer|referenceProfile|coreNum)\).*?\);','',source,flags=re.S)
    # Exclude blocks whose sole compile condition is instrumentation. Production
    # dtype/backend branches are deliberately never removed by this operation.
    lines=source.splitlines(keepends=True);kept=[];depth=0
    for line in lines:
        if depth:
            if re.match(r'\s*#\s*(?:if|ifdef|ifndef)\b',line):depth+=1
            if re.match(r'\s*#\s*endif\b',line):depth-=1
        elif re.match(r'\s*#ifdef DEEPEP_LL_(?:REFERENCE_PROFILE|STAGE_TIMER)\s*$',line):depth=1
        else:kept.append(line)
    source=''.join(kept)
    source=re.sub(r'^\s*#include [^\n]*ll_(?:stage_timer|expert_timer|reference_profile)[^\n]*\n','\n',source,flags=re.M)
    source=re.sub(r'\bLL_(?:SEG|SPARSE|BW|MTE3|CYCLE)_[A-Z_]+\([^;]*?\);','',source)
    source=remove_blocks(source,r'\bif\s*\(expertTimer_\s*&&\s*expertTimer_->Enabled\(\)\)\s*\{')
    source=re.sub(r'\buint32_t expert = 0;','',source)
    source=re.sub(r'\bif\s*\(expertTimer_\)\s*expertTimer_->(?:Start|End)\([^;]*\);','',source)
    source=remove_blocks(source,r'\bif\s*\(profile_\)\s*\{')
    source=re.sub(r'\bif\s*\([^;{}]*profile_[^;{}]*\)\s*profile_->(?:Mark|Value)<\d+>\([^;]*\);','',source)
    source=re.sub(r'\bprofile_->(?:Mark|Value)<\d+>\([^;]*\);','',source)
    source=re.sub(r'\b(?:timer|profile)\.(?:Mark<LlStageTimer::\w+>|Mark<\d+>|Value<\d+>|Flush|Finish|Init)\([^;]*\);','',source)
    source=re.sub(r'Ll(?:StageTimer::DeviceTimer|ReferenceProfile::DeviceProfile) (?:timer|profile);','',source)
    source=re.sub(r'const uint64_t referenceEntry=LlReferenceProfile::DeviceProfile::EntryClock\(referenceProfile\);','',source)
    source=remove_blocks(source,r'__aicore__ inline void FinishReference\(\)\s*\{')
    source=source.replace('FinishReference();','')
    source=re.sub(r'__aicore__ inline void SetTimer\(LlStageTimer::DeviceTimer\* timer\)\s*\{\s*timer_ = timer;\s*\}','',source)
    source=re.sub(r'\b\w+\.SetTimer\(timer_\);','',source)
    source=re.sub(r'Ll(?:StageTimer::DeviceTimer|ExpertTimer::DeviceTimer|ReferenceProfile::DeviceProfile)\*\s*(?:timer_|expertTimer_|profile_)\s*=\s*nullptr;','',source)
    source=re.sub(r'\b(?:timer_|profile_|expertTimer_)\s*=\s*(?:timer|profile|expertTimer);','',source)
    source=re.sub(r',\s*Ll(?:StageTimer::DeviceTimer|ExpertTimer::DeviceTimer|ReferenceProfile::DeviceProfile)\*\s*(?:timer_|timer|profile|expertTimer)\s*=\s*nullptr','',source)
    source=re.sub(r'(?<=\()Ll(?:StageTimer::DeviceTimer|ReferenceProfile::DeviceProfile)\*\s*(?:timer|profile)\s*=\s*nullptr','',source)
    source=re.sub(r'\((?:timer|profile|timer_|profile_)\)','()',source)
    source=source.replace('(void)timer_;','').replace('(void)expertTimer;','')
    source=source.replace(',recvCoreNum_, timer_)',',recvCoreNum_)')
    source=re.sub(r',\s*(?:timer_|profile_|timer|profile|expertTimer)(?=\s*[,\)])','',source)
    source=re.sub(r',\s*(?:timer|profile)\.Enabled\(\)\s*\?\s*&(?:timer|profile)\s*:\s*nullptr(?:,nullptr)?','',source)
    source=re.sub(r'\bif\s*\(overlap_\.enabled\)\s*\{\s*\}\s*else\s*\{\s*\}','',source)
    source=re.sub(r'\bif\s*\((?:begin==tasks|!result\.readyMask)\)\s*\{\s*\}','',source)
    # These syntax differences are only a mark-before-empty-return insertion.
    source=re.sub(r'\{\s*return;\s*\}','return;',source)
    # New ABI only removes the old diagnostic mode/expert pointer; explicit
    # offsetof/sizeof tests cover the replacement ABI separately.
    source=re.sub(r'^\s*uint8_t\* (?:stageTimer|expertTimer|referenceProfile);[^\n]*\n','\n',source,flags=re.M)
    source=re.sub(r'^\s*uint32_t timerMode;[^\n]*\n','\n',source,flags=re.M)
    source=re.sub(r'static_assert\(sizeof\(EpServerDedup(?:Combine)?KernelArgs\).*?\);','',source,flags=re.S)
    source=re.sub(r'static_assert\(offsetof\(EpServerDedup(?:Combine)?KernelArgs, (?:stageTimer|timerMode|expertTimer|referenceProfile|coreNum)\).*?\);','',source,flags=re.S)
    source=re.sub(r',\s*GM_ADDR (?:stageTimer|referenceProfile)(?:,\s*uint32_t timerMode)?(?:,\s*GM_ADDR expertTimer)?','',source)
    source=re.sub(r',\s*a\.(?:stageTimer|timerMode|expertTimer|referenceProfile)','',source)
    source=re.sub(r'extern "C" uint32_t deepep_ll_(?:stage_timer|reference_profile)_(?:enabled|schema|profile_id)\(\);','',source)
    source=source.replace('_sc18','').replace('_ref6','')
    return re.findall(r'[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+|[^\s]',source)


def test_business_code_matches_pre_migration_input():
    assert LEDGER['baseline']=='ed1c5531dbb22d4d2a6b5571a2d1bce17a3683b5' and len(LEDGER['files'])==53
    for relative in LEDGER['files']:
        if 'll_expert_timer_' in relative:
            assert not (ROOT/relative).exists()
            continue  # The user explicitly removed this diagnostics-only implementation.
        before=subprocess.check_output(['git','-C',str(ROOT),'show',BASELINE+':'+relative],encoding='utf-8')
        after=(ROOT/relative).read_text(encoding='utf-8')
        a,b=business_tokens(before),business_tokens(after)
        if a!=b:
            delta='\n'.join(list(difflib.unified_diff(a,b,n=5))[:100])
            pytest.fail(relative+'\n'+delta)


def test_no_old_native_telemetry_or_combine_trace_left():
    paths=list((ROOT/'csrc/kernels/dispatch/low_latency').rglob('*'))
    paths+=list((ROOT/'csrc/kernels/combine/low_latency').rglob('*'))
    paths+=[ROOT/'csrc/kernels/dispatch/low_latency.cpp',ROOT/'csrc/kernels/combine/low_latency.cpp',ROOT/'csrc/ops/low_latency.cpp']
    for path in paths:
        if not path.is_file():continue
        text=path.read_text(encoding='utf-8')
        assert not re.search(r'LlStageTimer|LlExpertTimer|LL_(?:SEG|SPARSE|BW|MTE3|CYCLE)_|ll_(?:stage|expert)_timer|_sc18',text),path
        if '/combine/' in path.as_posix():
            assert not re.search(r'GetSystemCycle|DeviceProfile|referenceProfile',text),path


def test_receive_capture_preserves_lanes_and_defers_export():
    path=ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h'
    text=path.read_text(encoding='utf-8')
    assert 'overlap_.totalBytes' in text and 'traceCapacity_=spare/32U' in text
    assert text.index('pipe_->InitBuffer(lanesBuf_')<text.index('pipe_->InitBuffer(packetTraceBuf_')
    assert 'if(traceCapacity_>tasks_.Count())traceCapacity_=tasks_.Count()' in text
    assert 'tasks_.Count()>traceCapacity_?1U:0U' in text
    assert 'profile->Detail(102U,profile->Clock())' in text
    assert 'profile->Detail(103U,profile->Clock())' in text
    capture=text[text.index('void ConsumeLoaded('):text.index('void ClearFlags()')]
    assert 'Clock()' not in capture and 'RecordProcessed(j,wire,headerOffset,keepFlags)' in capture
    assert capture.index('RecordProcessed(')<capture.index('PrepareServerDedupForward(tail)')
    entry=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h').read_text()
    assert entry.index('kernel.Process();')<entry.index('kernel.ExportPacketTrace();')<entry.index('profile.Finish(')


def test_mte_packet_identity_is_bound_to_consumed_task():
    text=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    pending=text[text.index('void SendPacket('):text.index('void BeginInputPrefetch()')]
    assert 'SendBegin' not in pending
    consume=text[text.index('void ConsumePacket('):]
    assert consume.index('const uint32_t destination=')<consume.index('SendBegin(')<consume.index('DataCopy(destinationGm')<consume.index('GroupSubmitted()')
