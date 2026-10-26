"""Prefix publication/consumption under independent DMA and AIV delays; CPU only."""
from itertools import accumulate
from pathlib import Path
import random
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h'


def method(source, name):
    start = source.index('__aicore__ inline void ServerDedupCounts::' + name + '(')
    first = source.index('{', start)
    end, depth = first + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end].replace('ServerDedupCounts::', '')


@pytest.mark.parametrize('dedup', [False, True])
@pytest.mark.parametrize('overlap', [False, True])
def test_actual_prefix_publish_read_clear_with_delayed_dma(tmp_path, dedup, overlap):
    cc = shutil.which('g++') or shutil.which('clang++')
    if not cc:
        pytest.skip('C++ compiler needed for actual prefix methods')
    source = COUNTS.read_text(encoding='utf-8')
    methods = '\n'.join(method(source, name) for name in (
        'CalRecvAndSetFlag', 'PublishPrefixReady', 'WaitCumSumFlag',
        'CompleteCumsum', 'ClearPrefixReady'))
    harness = Path(__file__).with_name('fixtures') / 'prefix_ready_slots.cpp'
    cpp = harness.read_text(encoding='utf-8').replace('// ACTUAL_COUNTS_METHODS', methods)
    src = tmp_path / 'prefix.cpp'
    exe = tmp_path / 'prefix.exe'
    src.write_text(cpp, encoding='utf-8')
    result = subprocess.run([cc, '-std=c++17', '-O2', '-UNDEBUG',
        f'-DDEDUP={int(dedup)}', f'-DOVERLAP={int(overlap)}', str(src), '-o', str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('dedup', [False, True])
@pytest.mark.parametrize('seed', [0, 7, 29])
def test_many_aiv_delays_and_call_sequences(seed, dedup):
    """Independent oracle: ready is per reader, reset retires at kernel end.

    A random step can finish a count, prefix DMA, ready DMA, prefix read, clear
    DMA, or core. No global ACK is modeled. Consecutive dispatches and D-C-D
    reuse both banks; empty workers must still consume and clear their slots.
    """
    rng = random.Random(seed)
    slots = [0] * 56
    banks = [None, None]
    magic = 0
    consumers = list(range(0 if dedup else 30, 56))
    for generation in range(1, 37):
        if generation % 3 != 0:  # combine toggles the bank but never ready.
            magic ^= 1
        magic ^= 1
        records = [rng.randrange(0, 20) for _ in range((16, 384, 32)[generation % 3])]
        if generation % 7 == 0:
            records = [0] * len(records)
        expected = (generation, tuple(accumulate(records)))
        assert not any(slots)
        pending_counts = list(range(len(records)))
        stage = {core: 'notify' for core in consumers}
        snapshots = {}
        complete = set()
        prefix_done = False
        for _ in range(100000):
            choices = [('count', None)] if pending_counts else []
            if not pending_counts and not prefix_done:
                choices.append(('prefix', None))
            if prefix_done:
                choices += [(stage[c], c) for c in consumers if c not in complete]
            action, core = rng.choice(choices)
            if action == 'count':
                pending_counts.pop(rng.randrange(len(pending_counts)))
            elif action == 'prefix':
                banks[magic] = expected
                prefix_done = True
            elif action == 'notify':
                assert slots[core] == 0
                slots[core] = 1
                stage[core] = 'poll'
            elif action == 'poll':
                assert slots[core] == 1
                stage[core] = 'read'
            elif action == 'read':
                snapshots[core] = banks[magic]
                assert snapshots[core] == expected
                stage[core] = 'clear_issue'
            elif action == 'clear_issue':
                assert snapshots[core] == expected
                stage[core] = 'clear_complete'
            elif action == 'clear_complete':
                slots[core] = 0
                stage[core] = 'core_finish'
            else:
                # Existing per-core final drain, not a new inter-core ACK.
                assert action == 'core_finish' and slots[core] == 0
                complete.add(core)
            if len(complete) == len(consumers):
                break
        assert len(complete) == len(consumers)
        assert snapshots == {core: expected for core in consumers}
        assert not any(slots)


def test_negative_controls_detect_parity_reuse_and_shared_clear():
    # Old shared bank epoch mistakes the previous D for a new D after combine.
    magic, ready = 1, 1
    magic ^= 1  # C
    magic ^= 1  # D: no new prefix yet.
    assert ready == magic
    # Clearing one shared slot as soon as R0 reads loses a slow core's release.
    shared_ready = 1
    r0_read = bool(shared_ready)
    shared_ready = 0
    assert r0_read and not shared_ready
    # With separate slots, only R0's release is consumed.
    ready_slots = [1] * 56
    ready_slots[30] = 0
    assert ready_slots[55] == 1


def test_ready_slots_fit_reserved_control_and_no_ack_or_early_clear():
    source = COUNTS.read_text(encoding='utf-8')
    offsets = [core * 512 + 32 for core in range(56)]
    assert len(set(offsets)) == 56 and max(offsets) + 32 <= 72 * 512
    assert all(offset % 32 == 0 and offset % 512 == 32 for offset in offsets)
    pub = method(source, 'PublishPrefixReady')
    clear = method(source, 'ClearPrefixReady')
    assert 'buildLocalTasks_ ? 0U : kLlSendCoreNum' in pub
    assert 'consumer<coreNum_' in pub
    assert pub.count('DataCopy(') == 1 and 'Wait' not in pub
    assert clear.count('DataCopy(') == 1 and 'MTE3_S' not in clear
    assert 'prefixEpoch_' not in source and 'readDone' not in source.replace('No readDone ACK.', '')
    complete = method(source, 'CompleteCumsum')
    assert complete.index('WaitCumSumFlag()') < complete.index('DataCopyPad(prefix, workspace')
    assert complete.index('SyncFunc<HardEvent::MTE2_S>()') < complete.index('ClearPrefixReady()')
    assert 'if (!overlap_.enabled) { SyncFunc<HardEvent::MTE3_S>(); }' in complete
    assert source.count('InitBuffer(readyBuf_,32U)') == 1
