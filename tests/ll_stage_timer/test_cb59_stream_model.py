"""Independent route/reference and asynchronous protocol model for cb59-stream.

This tests conservation, dense ordering, unique ownership, completion and bank
reuse. Synthetic event ticks are deliberately not performance predictions. It
does not model NPU memory ordering, UB allocation, caches or DMA retirement;
those still require source checks and target validation of the real helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random
from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest


@dataclass(frozen=True)
class Packet:
    source: int
    token: int
    topk: int
    expert: int

    @property
    def return_slot(self) -> tuple[int, int, int]:
        return self.source, self.token, self.topk


@dataclass(frozen=True)
class RoutingReference:
    world: int
    experts: int
    destination: int
    records: tuple[tuple[Packet, ...], ...]
    rows: tuple[Packet, ...]
    prefix: tuple[int, ...]
    expert_counts: tuple[int, ...]


def reference_from_routes(routes, experts: int, destination: int) -> RoutingReference:
    """Ground truth from original routes, never from stream prefix/sum code."""
    world = len(routes)
    assert experts % world == 0
    local = experts // world
    records: list[list[Packet]] = [[] for _ in range(experts)]
    for source, tokens in enumerate(routes):
        for token, topk in enumerate(tokens):
            assert len(topk) == len(set(topk)), "one token cannot repeat an expert"
            for k, expert in enumerate(topk):
                assert 0 <= expert < experts
                if expert // local == destination:
                    records[(expert % local) * world + source].append(Packet(source, token, k, expert))
    # Group by expert and source, retaining original token order within each
    # group. This independently defines both q and the public dense row order.
    rows = tuple(packet for record in records for packet in record)
    prefix = []
    cumulative = 0
    for record in records:
        cumulative += len(record)
        prefix.append(cumulative)
    counts = tuple(sum(len(record) for record in records[e * world:(e + 1) * world]) for e in range(local))
    return RoutingReference(world, experts, destination, tuple(tuple(record) for record in records),
                            rows, tuple(prefix), counts)


def record_slices(records: int, workers: int) -> tuple[range, ...]:
    """Balanced contiguous ownership, with earlier workers receiving remainders."""
    quotient, remainder = divmod(records, workers)
    result = []
    cursor = 0
    for worker in range(workers):
        size = quotient + (worker < remainder)
        result.append(range(cursor, cursor + size))
        cursor += size
    assert cursor == records
    return tuple(result)


@dataclass(frozen=True)
class SumMessage:
    epoch: int
    total: int
    tail_expert: int
    tail_count: int


@dataclass
class Bank:
    sums: dict[tuple[int, int], SumMessage] = field(default_factory=dict)
    prefix_ready: dict[tuple[int, int], int] = field(default_factory=dict)
    counts: dict[int, tuple[int, int]] = field(default_factory=dict)
    packets: dict[tuple[int, int], tuple[int, Packet]] = field(default_factory=dict)

    def assert_clean(self) -> None:
        assert not self.sums, ("unconsumed sum cells", self.sums)
        assert not self.prefix_ready, ("unconsumed prefix flags", self.prefix_ready)
        assert not self.counts, ("unconsumed count records", self.counts)
        assert not self.packets, ("uncleared payload flags", self.packets)


@dataclass
class Worker:
    accepted: bool = False
    offset: int | None = None
    local_counts: tuple[int, ...] = ()
    tasks: tuple[tuple[int, tuple[int, int]], ...] | None = None
    consumed: set[tuple[int, int]] = field(default_factory=set)
    cleanup_queued: bool = False


class ProtocolSimulation:
    def __init__(self, reference: RoutingReference, *, mode: str, workers: int = 28,
                 seed: int = 0, bank: Bank | None = None, epoch: int = 1,
                 delayed_after_record: int | None = None):
        assert mode in {"stream", "legacy"}
        self.ref, self.mode, self.workers, self.epoch = reference, mode, workers, epoch
        self.bank = bank if bank is not None else Bank()
        self.bank.assert_clean()
        self.rng = random.Random(seed)
        self.slices = record_slices(reference.experts, workers)
        self.active = sum(bool(part) for part in self.slices)
        self.state = [Worker() for _ in range(workers)]
        self.public_prefix = [None] * reference.experts
        self.expert_cumulative = [None] * (reference.experts // reference.world)
        self.expert_counts = [None] * len(self.expert_cumulative)
        self.rows: dict[int, Packet] = {}
        self.row_owner: dict[int, int] = {}
        self.probe_time: dict[int, int] = {}
        self.prefix_time: dict[int, int] = {}
        self.count_time: dict[int, int] = {}
        self.dependencies: dict[int, tuple[int, ...]] = {}
        self.events: list[tuple[int, int, str, object]] = []
        self.serial = self.now = 0
        self.messages_written: list[tuple[int, int]] = []
        for record, packets in enumerate(reference.records):
            tick = self.rng.randrange(1, 80)
            if delayed_after_record is not None:
                tick = 1 if record < delayed_after_record else 1000 + tick
            self.queue(tick, "count", (record, len(packets)))
            for q, packet in enumerate(packets):
                self.queue(self.rng.randrange(2, 120), "packet", ((record, q), packet))

    def queue(self, tick: int, kind: str, data) -> None:
        self.serial += 1
        heapq.heappush(self.events, (tick, self.serial, kind, data))

    def later(self, kind: str, data) -> None:
        self.queue(self.now + self.rng.randrange(1, 17), kind, data)

    def dispatch_event(self, kind: str, data) -> None:
        if kind == "count":
            record, count = data
            assert record not in self.bank.counts
            self.bank.counts[record] = self.epoch, count
        elif kind == "packet":
            slot, packet = data
            assert slot not in self.bank.packets
            self.bank.packets[slot] = self.epoch, packet
        elif kind == "sum":
            key, message = data
            assert key not in self.bank.sums, ("overwrite before clear", key)
            self.bank.sums[key] = message
        elif kind == "ready":
            key = data
            assert key not in self.bank.prefix_ready
            self.bank.prefix_ready[key] = self.epoch
        elif kind.startswith("clear_"):
            table = {"clear_sum": self.bank.sums, "clear_ready": self.bank.prefix_ready,
                     "clear_count": self.bank.counts, "clear_packet": self.bank.packets}[kind]
            key = data
            value = table.pop(key)
            observed_epoch = value.epoch if isinstance(value, SumMessage) else value if isinstance(value, int) else value[0]
            assert observed_epoch == self.epoch, "late clear touched another epoch"
        else:
            raise AssertionError(kind)

    def accept_counts(self, worker: int) -> bool:
        state, part = self.state[worker], self.slices[worker]
        if state.accepted or not part or not all(record in self.bank.counts for record in part):
            return False
        state.accepted = True
        state.local_counts = tuple(self.bank.counts[record][1] for record in part)
        self.count_time[worker] = self.now
        tail_expert = (part.stop - 1) // self.ref.world
        tail_begin = max(part.start, tail_expert * self.ref.world) - part.start
        message = SumMessage(self.epoch, sum(state.local_counts), tail_expert, sum(state.local_counts[tail_begin:]))
        consumers = range(worker + 1, self.active) if self.mode == "stream" else range(self.active)
        # Publish LOCAL totals independently of offsets. Waiting for predecessors
        # before this publication would introduce a serial propagation chain.
        for consumer in consumers:
            self.messages_written.append((worker, consumer))
            self.later("sum", ((worker, consumer), message))
        for record in part:
            self.later("clear_count", record)
        return True

    def prepare_prefix(self, worker: int) -> bool:
        state, part = self.state[worker], self.slices[worker]
        if not part or not state.accepted or state.offset is not None:
            return False
        required = tuple(range(worker)) if self.mode == "stream" else tuple(range(self.active))
        if not all((producer, worker) in self.bank.sums for producer in required):
            return False
        incoming = {producer: self.bank.sums[(producer, worker)] for producer in required}
        assert all(message.epoch == self.epoch for message in incoming.values())
        self.dependencies[worker] = required
        state.offset = sum(message.total for producer, message in incoming.items() if producer < worker)
        cumulative = state.offset
        for index, record in enumerate(part):
            cumulative += state.local_counts[index]
            assert self.public_prefix[record] is None
            self.public_prefix[record] = cumulative
            if self.mode == "stream" and (record + 1) % self.ref.world == 0:
                expert = record // self.ref.world
                own_begin = max(part.start, expert * self.ref.world) - part.start
                own_count = sum(state.local_counts[own_begin:index + 1])
                previous_tail = sum(message.tail_count for message in incoming.values() if message.tail_expert == expert)
                assert self.expert_counts[expert] is None, "expert endpoint has two owners"
                self.expert_counts[expert] = own_count + previous_tail
                self.expert_cumulative[expert] = cumulative
        for producer in required:
            self.later("clear_sum", (producer, worker))
        self.prefix_time[worker] = self.now
        if self.mode == "legacy":
            # The legacy full-prefix notification is sent to ALL receivers,
            # even those with no count slice when E < R.
            for consumer in range(self.workers):
                self.later("ready", (worker, consumer))
        else:
            tasks = []
            row = state.offset
            for record, count in zip(part, state.local_counts):
                for q in range(count):
                    tasks.append((row, (record, q)))
                    row += 1
            state.tasks = tuple(tasks)
        return True

    def prepare_legacy_receiver(self, worker: int) -> bool:
        state = self.state[worker]
        if self.mode != "legacy" or state.tasks is not None:
            return False
        if not all((producer, worker) in self.bank.prefix_ready for producer in range(self.active)):
            return False
        assert all(value is not None for value in self.public_prefix)
        total = self.public_prefix[-1]
        begin, end = worker * total // self.workers, (worker + 1) * total // self.workers
        tasks = []
        previous = 0
        for record, cumulative in enumerate(self.public_prefix):
            tasks.extend((row, (record, row - previous)) for row in range(max(begin, previous), min(end, cumulative)))
            previous = cumulative
        state.tasks = tuple(tasks)
        for producer in range(self.active):
            self.later("clear_ready", (producer, worker))
        if worker == 0:
            ends = self.public_prefix[self.ref.world - 1::self.ref.world]
            self.expert_cumulative[:] = ends
            self.expert_counts[:] = [end - (ends[i - 1] if i else 0) for i, end in enumerate(ends)]
        return True

    def receive(self, worker: int) -> bool:
        state = self.state[worker]
        if state.tasks is None:
            if self.mode == "stream" and not self.slices[worker]:
                state.tasks = ()
            else:
                return False
        changed = False
        if state.tasks and worker not in self.probe_time:
            self.probe_time[worker] = self.now
        for row, slot in state.tasks:
            if slot in state.consumed or slot not in self.bank.packets:
                continue
            epoch, packet = self.bank.packets[slot]
            assert epoch == self.epoch
            assert row not in self.rows, ("duplicate dense-row owner", row)
            state.consumed.add(slot)
            self.rows[row], self.row_owner[row] = packet, worker
            changed = True
        if len(state.consumed) == len(state.tasks) and not state.cleanup_queued:
            state.cleanup_queued = True
            for _, slot in state.tasks:
                self.later("clear_packet", slot)
            changed = True
        return changed

    def advance(self) -> None:
        while True:
            changed = False
            for worker in range(self.workers):
                changed |= self.accept_counts(worker)
                changed |= self.prepare_prefix(worker)
                changed |= self.prepare_legacy_receiver(worker)
                changed |= self.receive(worker)
            if not changed:
                return

    def run(self) -> "ProtocolSimulation":
        while self.events:
            self.now, _, kind, data = heapq.heappop(self.events)
            self.dispatch_event(kind, data)
            self.advance()
        assert all(state.cleanup_queued for state in self.state), "finite delivery left a deadlocked worker"
        self.bank.assert_clean()
        assert self.public_prefix == list(self.ref.prefix)
        assert self.expert_counts == list(self.ref.expert_counts)
        assert self.expert_cumulative == list(self.ref.prefix[self.ref.world - 1::self.ref.world])
        assert tuple(self.rows[row] for row in range(len(self.ref.rows))) == self.ref.rows
        assert len(self.rows) == len(self.ref.rows)
        return self


def routes_for(*, experts: int, batch: int, topk: int = 8, world: int = 16, seed: int = 1, hot: bool = False):
    rng = random.Random(seed)
    return [[list(range(topk)) if hot else rng.sample(range(experts), topk)
             for _ in range(batch)] for _ in range(world)]


@pytest.mark.parametrize("experts,batch", [(16, 1), (32, 4), (64, 8), (384, 1), (384, 8), (384, 32), (1024, 4)])
@pytest.mark.parametrize("seed", [2, 19, 97])
def test_out_of_order_delivery_matches_original_routes(experts: int, batch: int, seed: int) -> None:
    reference = reference_from_routes(routes_for(experts=experts, batch=batch, seed=seed), experts, seed % 16)
    model = ProtocolSimulation(reference, mode="stream", seed=seed).run()
    assert all(producer < consumer < model.active for producer, consumer in model.messages_written)
    assert all(all(producer < consumer for producer in predecessors) for consumer, predecessors in model.dependencies.items())


def test_r0_receives_before_any_later_count_slice_is_ready() -> None:
    reference = reference_from_routes(routes_for(experts=384, batch=1, hot=True), 384, 0)
    stop = record_slices(384, 28)[0].stop
    stream = ProtocolSimulation(reference, mode="stream", delayed_after_record=stop).run()
    legacy = ProtocolSimulation(reference, mode="legacy", delayed_after_record=stop).run()
    assert stream.dependencies[0] == ()
    assert stream.probe_time[0] < min(tick for worker, tick in stream.count_time.items() if worker > 0)
    assert stream.probe_time[0] < legacy.probe_time[0]


def test_stream_and_legacy_alternate_on_both_banks_without_stale_ready() -> None:
    banks = [Bank(), Bank()]
    modes = ("stream", "legacy", "legacy", "stream", "stream", "legacy", "legacy", "stream")
    for iteration, mode in enumerate(modes):
        experts = (16, 384, 64, 1024)[iteration % 4]
        reference = reference_from_routes(routes_for(experts=experts, batch=1 + iteration, seed=iteration), experts, iteration % 16)
        ProtocolSimulation(reference, mode=mode, seed=101 + iteration, bank=banks[iteration % 2], epoch=iteration + 1).run()
        for bank in banks:
            bank.assert_clean()


def test_expert_spanning_many_fragments_and_empty_count_workers() -> None:
    reference = reference_from_routes(routes_for(experts=16, batch=4, topk=1, hot=True), 16, 0)
    model = ProtocolSimulation(reference, mode="stream", seed=123).run()
    assert model.active == 16 and sum(not part for part in model.slices) == 12
    assert model.expert_counts == [64]
    assert len(model.dependencies[15]) == 15
    assert all(not model.state[worker].tasks for worker in range(16, 28))
    assert all(consumer < 16 for _, consumer in model.messages_written)


def test_zero_payload_still_consumes_every_count_and_sum_flag() -> None:
    reference = reference_from_routes(routes_for(experts=384, batch=4, hot=True), 384, 15)
    assert not reference.rows
    model = ProtocolSimulation(reference, mode="stream", seed=9).run()
    assert len(model.count_time) == model.active
    assert not model.probe_time
    assert model.expert_counts == [0] * 24


def test_hotspot_ownership_and_combine_triples_are_exactly_once() -> None:
    routes = routes_for(experts=384, batch=32, hot=True)
    reference = reference_from_routes(routes, 384, 0)
    model = ProtocolSimulation(reference, mode="stream", seed=302).run()
    returns = {}
    for packet in model.rows.values():
        assert packet.return_slot not in returns
        # Stand in for the expert output and its original source-side weight.
        returns[packet.return_slot] = (packet.expert + 1) * (packet.token + 3)
    expected = {(source, token, k): (expert + 1) * (token + 3)
                for source, tokens in enumerate(routes) for token, experts in enumerate(tokens)
                for k, expert in enumerate(experts)}
    assert returns == expected
    for source, tokens in enumerate(routes):
        for token, experts in enumerate(tokens):
            assert sum((k + 1) * returns[source, token, k] for k in range(len(experts))) == sum(
                (k + 1) * (expert + 1) * (token + 3) for k, expert in enumerate(experts))
    loads = [len(state.tasks) for state in model.state]
    assert max(loads) == 14 * 32  # Record ownership requires a larger UB bound.
    assert max(loads) > (len(reference.rows) + 27) // 28


def test_missing_source_zero_count_is_not_silently_treated_as_zero() -> None:
    reference = reference_from_routes(routes_for(experts=384, batch=1, hot=True), 384, 15)
    model = ProtocolSimulation(reference, mode="stream", seed=1)
    model.events = [event for event in model.events if not (event[2] == "count" and event[3][0] == 0)]
    heapq.heapify(model.events)
    with pytest.raises(AssertionError, match="deadlocked worker"):
        model.run()


def test_actual_stream_counts_and_receive_task_headers_with_threads(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    fixtures = Path(__file__).with_name("fixtures")
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler:
        found = shutil.which("g++") or shutil.which("clang++")
        if not found:
            pytest.skip("Host C++ compiler required for real-header execution")
        compiler = [found]
    shutil.copyfile(fixtures / "cb59_stream_kernel_operator_mock.h", tmp_path / "kernel_operator.h")
    executable = tmp_path / ("stream_headers.exe" if os.name == "nt" else "stream_headers")
    command = [*compiler, "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_LL_DISPATCH_CB59=2",
               "-iquote", str(tmp_path), "-iquote", str(root),
               str(fixtures / "cb59_stream_host_emulation.cpp"), "-o", str(executable)]
    if os.name != "nt":
        command.append("-pthread")
    built = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=180)
    (tmp_path / "compile.log").write_text(built.stdout + built.stderr, encoding="utf-8")
    assert built.returncode == 0, built.stderr[-8000:]
    executed = subprocess.run([str(executable)], capture_output=True, text=True, errors="replace", timeout=90)
    (tmp_path / "execute.log").write_text(executed.stdout + executed.stderr, encoding="utf-8")
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "49 threaded cases" in executed.stdout
