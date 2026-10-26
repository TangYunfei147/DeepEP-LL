"""Delayed protocol model for removing the *extra* shared-pull finish fence.

This models delivery/consumer completion as independent schedulable events,
not device instruction timing. Fixed EP16/G4/B256/H3584/K8/E384 layouts,
one ordered stream per communicator and matching dispatch handles are inputs.
No extra all-rank barrier or reader ACK appears in the graph. Counterexamples
remove individual existing dependencies; these tests do not prove CANN memory
ordering, arbitrary shapes, concurrent streams, or mismatched combine handles.
Payload traffic is projected onto two selected slots; unrelated positive
routes are omitted, while every rank's count publication is retained.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "csrc/kernels/dispatch/low_latency"
COMBINE = ROOT / "csrc/kernels/combine/low_latency"
WORLD, GROUP, CORES = 16, 4, 56
BASE = 2 * 1024**2
RETURN_STRIDE = ((3584 * 2 + 479) // 480) * 512
DATA_BASE = BASE + 256 * 8 * RETURN_STRIDE
PACKET, STAGE = 4096, 7168


@dataclass(frozen=True)
class Flow:
    source: int
    owner: int
    reader: int
    first_slot: int
    terminal_slot: int
    core: int


# Both writers are outside their readers' G4. The second terminal's legal
# combine-stage slot overlaps the first first-hop packet at the same bank.
# q=182 is legal for B=256; uninteresting preceding routes are abstracted away.
FLOWS = (Flow(8, 0, 1, 49152, 49152, 55),
         Flow(4, 2, 0, 36864, 28086, 30))


class Protocol:
    def __init__(self, calls, *, zero_counts=True, group_ready=True, source_mask=True):
        self.calls = calls
        self.zero_counts = zero_counts
        self.group_ready = group_ready
        self.source_mask = source_mask
        self.events = {}
        self.memory = {}
        self.trace = []
        for index, kind in enumerate(calls):
            for rank in range(WORLD):
                previous = [(index - 1, "done", rank)] if index else []
                self.add((index, "start", rank), previous)
            if kind == "D":
                # The middle D has zero routes in the tracked projection. Its
                # mandatory zero count publications protect reuse even though
                # no modeled payload creates an incidental dependency.
                self.dispatch(index, () if index == 1 else FLOWS)
            else:
                self.combine(index)

    def add(self, key, dependencies=(), action=None):
        assert key not in self.events
        self.events[key] = (set(dependencies), action)
        return key

    @staticmethod
    def bank(index):
        return (index + 1) % 2

    def write(self, bank, rank, offset, length, value):
        # Track bytes relevant to outstanding remote pulls. A partial overwrite
        # is enough to invalidate the expected packet, just as a real DMA would.
        for flow in FLOWS:
            start = DATA_BASE + flow.first_slot * PACKET
            if rank == flow.owner and offset < start + PACKET and start < offset + length:
                self.memory[bank, flow] = value

    def read(self, bank, flow, expected):
        actual = self.memory.get((bank, flow))
        assert actual == expected, ("payload overwritten", flow, expected, actual, self.trace[-12:])

    def dispatch(self, index, flows):
        outgoing = {rank: [] for rank in range(WORLD)}
        incoming = {rank: [] for rank in range(WORLD)}
        for source in range(WORLD):
            for target in range(WORLD):
                nonzero = any(f.source == source and target in (f.owner, f.reader) for f in flows)
                if not self.zero_counts and not nonzero:
                    continue
                delivered = self.add((index, "count", source, target), [(index, "start", source)])
                outgoing[source].append(delivered)
                incoming[target].append(delivered)
        receives = {}
        for flow in flows:
            value = (index, "dispatch", flow.source)
            write = self.add((index, "packet", flow.source), [(index, "start", flow.source)],
                             lambda f=flow, v=value: self.write(self.bank(index), f.owner,
                                 DATA_BASE + f.first_slot * PACKET, PACKET, v))
            outgoing[flow.source].append(write)
            owner_read = self.add((index, "owner_read", flow.source),
                                 [write, (index, "start", flow.owner), *incoming[flow.owner]],
                                 lambda f=flow, v=value: self.read(self.bank(index), f, v))
            reference = self.add((index, "reference", flow.source), [owner_read])
            outgoing[flow.owner].append(reference)
            terminal_read = self.add((index, "pull_complete", flow.reader),
                                    [reference, (index, "start", flow.reader), *incoming[flow.reader]],
                                    lambda f=flow, v=value: self.read(self.bank(index), f, v))
            receives[flow.owner, 0] = owner_read
            receives[flow.reader, flow.core] = terminal_read
        for rank in range(WORLD):
            cores = []
            for core in range(CORES):
                deps = [(index, "start", rank), *incoming[rank]]
                if (rank, core) in receives:
                    deps.append(receives[rank, core])
                cores.append(self.add((index, "core_done", rank, core), deps))
            # A rank kernel return retires its own cores and outgoing DMA only.
            # It deliberately does NOT wait for peers to consume references.
            self.add((index, "done", rank), [*cores, *outgoing[rank]])

    def combine(self, index):
        stages = {}
        outgoing = {rank: [] for rank in range(WORLD)}
        for rank in range(WORLD):
            def stage_write(rank=rank):
                for flow in FLOWS:
                    if flow.reader == rank:
                        self.write(self.bank(index), rank, DATA_BASE + flow.terminal_slot * STAGE,
                                   STAGE, (index, "combine_stage", rank))
            stages[rank] = self.add((index, "local_stage", rank), [(index, "start", rank)], stage_write)
        ready = {}
        for source in range(WORLD):
            for target in range(source // GROUP * GROUP, source // GROUP * GROUP + GROUP):
                delivery = self.add((index, "ready", source, target), [stages[source]])
                outgoing[source].append(delivery)
                ready.setdefault(target, []).append(delivery)
        for rank in range(WORLD):
            self.add((index, "group_ready", rank),
                     ready[rank] if self.group_ready else [stages[rank]])
        result = {}
        for flow in FLOWS:
            # FIRST_HOP partial reads group stage only after the G4 ready row.
            # The remote result write is below DATA_BASE, never into dispatch
            # slots; only local stage writes may alias old first-hop packets.
            partial = self.add((index, "partial", flow.source), [(index, "group_ready", flow.owner)],
                               lambda f=flow: self.write(self.bank(index), f.source,
                                   BASE, RETURN_STRIDE, (index, "return", f.owner)))
            outgoing[flow.owner].append(partial)
            result.setdefault(flow.source, []).append(partial)
        for rank in range(WORLD):
            aggregate = self.add((index, "aggregate", rank),
                                 [(index, "group_ready", rank),
                                  *(result.get(rank, []) if self.source_mask else [])])
            self.add((index, "done", rank), [aggregate, *outgoing[rank]])

    def run(self, seed, *, delay_reader=False):
        rng = random.Random(seed)
        remaining = {key: set(deps) for key, (deps, _) in self.events.items()}
        followers = {key: [] for key in self.events}
        for key, deps in remaining.items():
            for dep in deps:
                assert dep in self.events, dep
                followers[dep].append(key)
        ready = [key for key, deps in remaining.items() if not deps]
        while ready:
            candidates = ready
            if delay_reader:
                # Hold rank1/core55's actual DMA completion until every other
                # independently enabled event has run. Network deliveries and
                # all other ranks may race two calls ahead if dependencies allow.
                unheld = [key for key in ready if key != (0, "pull_complete", 1)]
                candidates = unheld or ready
            key = rng.choice(candidates)
            ready.remove(key)
            self.trace.append(key)
            action = self.events[key][1]
            if action:
                action()
            del remaining[key]
            for follower in followers[key]:
                remaining[follower].remove(key)
                if not remaining[follower]:
                    ready.append(follower)
        assert not remaining, ("deadlock", remaining)
        assert len(self.trace) == len(self.events)
        return self.trace


@pytest.mark.parametrize("calls", ["DDD", "DCD", "DCC"])
@pytest.mark.parametrize("delay_reader", [False, True])
def test_existing_dependencies_protect_reused_bank(calls, delay_reader):
    for seed in range(16):
        protocol = Protocol(calls)
        trace = protocol.run(seed, delay_reader=delay_reader)
        if calls != "DCC":
            assert trace.index((0, "pull_complete", 1)) < trace.index((2, "start", 8))
        assert any(key[1:2] == ("core_done",) and key[-1] == 54 for key in trace)
    # The checked schedule is not a globally serial invocation model.
    if delay_reader:
        assert trace.index((1, "start", 8)) < trace.index((0, "done", 1))


@pytest.mark.parametrize("calls,removed", [
    ("DDD", "zero_counts"), ("DCD", "group_ready"),
    ("DCC", "group_ready"), ("DCD", "source_mask"),
])
def test_negative_controls_expose_early_reuse(calls, removed):
    with pytest.raises(AssertionError, match="payload overwritten"):
        Protocol(calls, **{removed: False}).run(3, delay_reader=True)


def _text(path):
    return re.sub(r"\s+", "", path.read_text(encoding="utf-8"))


def test_model_dependencies_and_address_separation_exist_in_source():
    counts = _text(DISPATCH / "device/dispatch_server_dedup_counts.h")
    sender = _text(DISPATCH / "device/dispatch_server_dedup_send.h")
    kernel = _text(COMBINE / "combine_server_dedup_kernel.h")
    aggregate = _text(COMBINE / "combine_server_dedup_source_aggregate.h")
    layout = _text(DISPATCH / "common/ep_memory_server_dedup_data_layout.h")
    bank = _text(DISPATCH / "common/ep_memory_server_dedup_count_layout.h")
    assert "Duplicate<int32_t>(statusTensor_,0x3f800000,flagMask" in counts
    assert "if(sum.GetValue(0U)!=float(rscvStatusNum_))returnfalse;" in counts
    assert "SendExpertCntRange(range.begin,range.end,range.record);" in counts
    assert "sourceMask|=uint32_t(1)<<firstK;" in sender
    assert "sourceMasks_.SetValue(token,sourceMask);" in sender
    assert "if(!Probe(token,mask))" in aggregate
    assert "GetBitCount<1>(uint64_t(mask))" in aggregate
    assert kernel.index("WaitServerReady();") < kernel.index("ReturnPartialTokens();") < kernel.index("AggregateSourceTokens();")
    assert "WaitFlags(done,prepareInput_,prepareWork_,prepareSum_,coreNum_);" in kernel
    assert "WaitFlags(ready,readyInput_,readyWork_,readySum_,DispatchDedup::kLlRanksPerGroup);" in kernel
    assert "args_->peerMems[args_->rank]+stageBase_" in kernel
    assert "uint64_t(bs)*topK*resultStride_" in kernel
    assert "layout.sourceBytes=uint64_t(bs)*topK*layout.returnStride;" in layout
    assert "layout.dispatchOffset=DispatchDedupAlign(layout.sourceOffset+layout.sourceBytes,512U);" in layout
    assert "(uint64_t(magic)%2U)*kEpServerDedupWindowBytes" in bank
    assert BASE + 256 * 8 * RETURN_STRIDE == DATA_BASE
    a, b = FLOWS
    first = DATA_BASE + a.first_slot * PACKET
    stage = DATA_BASE + b.terminal_slot * STAGE
    assert stage < first + PACKET and first < stage + STAGE
    assert a.source // GROUP != a.owner // GROUP == a.reader // GROUP
