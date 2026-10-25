"""Route-based asynchronous mailbox model and real-header CPU tests for pair RX."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("cb59_pair_reference_stream_model", HERE / "test_cb59_stream_model.py")
assert SPEC and SPEC.loader
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


class PairSimulation(MODEL.ProtocolSimulation):
    def __init__(self, reference, **kwargs):
        super().__init__(reference, mode="stream", **kwargs)
        self.headers_published = {}
        self.headers_consumed = {}
        self.borrowed_zero_owner = set()
        self.partner_prefix_reads = set()

    def dispatch_event(self, kind, data):
        if kind == "pair_header":
            key, header = data
            assert key not in self.bank.prefix_ready
            self.bank.prefix_ready[key] = header
        elif kind == "clear_pair_header":
            assert self.bank.prefix_ready.pop(data)[0] == self.epoch
        else:
            super().dispatch_event(kind, data)

    def prepare_prefix(self, worker):
        changed = super().prepare_prefix(worker)
        if changed:
            state, part = self.state[worker], self.slices[worker]
            state.tasks = None
            partner = worker ^ 1
            assert self.slices[partner], "eligible EP16 empty slices occur in pairs"
            header = (self.epoch, state.offset, sum(state.local_counts), part.start, len(part))
            self.headers_published[worker] = self.now
            # Public prefix is already submitted before this notification.
            self.later("pair_header", ((worker, partner), header))
        return changed

    def prepare_legacy_receiver(self, worker):
        state, own = self.state[worker], self.slices[worker]
        if not own or state.offset is None or state.tasks is not None:
            return False
        partner = worker ^ 1
        key = partner, worker
        if key not in self.bank.prefix_ready:
            return False
        epoch, partner_begin, partner_rows, record_begin, record_count = self.bank.prefix_ready[key]
        assert epoch == self.epoch
        assert (record_begin, record_count) == (self.slices[partner].start, len(self.slices[partner]))
        assert worker in self.headers_published, "publish must precede the partner wait"
        low, high = worker & ~1, worker | 1
        low_begin = self.state[low].offset
        low_rows, high_rows = sum(self.state[low].local_counts), sum(self.state[high].local_counts)
        assert self.state[high].offset == low_begin + low_rows
        # The reference is a list of original dense row IDs, split in half;
        # payload identities below are checked against original input routes.
        pair_rows = list(range(low_begin, low_begin + low_rows + high_rows))
        cut = len(pair_rows) // 2
        assigned = pair_rows[:cut] if worker == low else pair_rows[cut:]
        partner_range = set(range(partner_begin, partner_begin + partner_rows))
        if partner_range.intersection(assigned):
            self.partner_prefix_reads.add(worker)
            if not sum(state.local_counts):
                self.borrowed_zero_owner.add(worker)
        tasks = []
        previous = low_begin
        for record in range(self.slices[low].start, self.slices[high].stop):
            for row in assigned:
                if previous <= row < self.public_prefix[record]:
                    tasks.append((row, (record, row - previous)))
            previous = self.public_prefix[record]
        assert len(tasks) == len(assigned)
        state.tasks = tuple(tasks)
        self.headers_consumed[worker] = self.now
        self.later("clear_pair_header", key)
        return True


@pytest.mark.parametrize("experts,batch", [(16, 1), (32, 2), (64, 4), (384, 1), (384, 8), (384, 32), (1024, 8)])
@pytest.mark.parametrize("seed", [3, 29, 211])
def test_pair_mailbox_delivery_preserves_original_dense_rows(experts, batch, seed):
    reference = MODEL.reference_from_routes(MODEL.routes_for(experts=experts, batch=batch, seed=seed), experts, seed % 16)
    model = PairSimulation(reference, seed=seed).run()
    for first in range(0, model.workers, 2):
        assert abs(len(model.state[first].tasks) - len(model.state[first + 1].tasks)) <= 1
    assert all(model.headers_published[worker] <= tick for worker, tick in model.headers_consumed.items())


def test_pair_zero_payload_owner_can_borrow_and_probe():
    # Each sender's only route is expert0, but only source0 sends to destination0.
    # R0 owns that count record; paired R1 owns no payload, yet receives half.
    routes = [[[0] if source == 0 else [383] for _ in range(8)] for source in range(16)]
    reference = MODEL.reference_from_routes(routes, 384, 0)
    model = PairSimulation(reference, seed=7).run()
    assert model.borrowed_zero_owner == {1}
    assert len(model.state[0].tasks) == len(model.state[1].tasks) == 4
    assert [slot[1] for _, slot in model.state[1].tasks] == [4, 5, 6, 7]
    assert 1 in model.probe_time


def test_pair_zero_headers_are_published_and_completely_cleared():
    reference = MODEL.reference_from_routes(MODEL.routes_for(experts=384, batch=4, hot=True), 384, 15)
    assert not reference.rows
    model = PairSimulation(reference, seed=13).run()
    assert len(model.headers_published) == len(model.headers_consumed) == 28
    assert not model.partner_prefix_reads and not model.probe_time
    model.bank.assert_clean()


def test_first_pair_does_not_wait_for_later_record_owners():
    reference = MODEL.reference_from_routes(MODEL.routes_for(experts=384, batch=1, hot=True), 384, 0)
    pair_end = MODEL.record_slices(384, 28)[1].stop
    model = PairSimulation(reference, seed=5, delayed_after_record=pair_end).run()
    assert max(model.probe_time[0], model.probe_time[1]) < min(
        tick for owner, tick in model.count_time.items() if owner >= 2)


def test_pair_stream_legacy_alternate_for_twelve_calls_on_both_banks():
    banks = [MODEL.Bank(), MODEL.Bank()]
    modes = ("pair", "stream", "legacy", "pair", "stream", "legacy",
             "pair", "legacy", "stream", "pair", "legacy", "stream")
    for iteration, mode in enumerate(modes):
        experts = (16, 32, 384, 1024)[iteration % 4]
        reference = MODEL.reference_from_routes(
            MODEL.routes_for(experts=experts, batch=1 + iteration, seed=iteration, hot=iteration % 3 == 1), experts, iteration % 16)
        kwargs = dict(bank=banks[iteration % 2], epoch=iteration + 1, seed=31 + iteration)
        model = PairSimulation(reference, **kwargs) if mode == "pair" else MODEL.ProtocolSimulation(reference, mode=mode, **kwargs)
        model.run()
        for bank in banks:
            bank.assert_clean()


def test_actual_pair_handoff_planner_and_task_headers(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler:
        found = shutil.which("g++") or shutil.which("clang++")
        if not found:
            pytest.skip("Host C++ compiler required for real pair-header execution")
        compiler = [found]
    fixtures = HERE / "fixtures"
    base = (fixtures / "cb59_stream_kernel_operator_mock.h").read_text(encoding="utf-8")
    extension = (fixtures / "cb59_pair_kernel_operator_mock.h").read_text(encoding="utf-8")
    (tmp_path / "kernel_operator.h").write_text(base + "\n" + extension, encoding="utf-8")
    executable = tmp_path / ("pair_headers.exe" if os.name == "nt" else "pair_headers")
    command = [*compiler, "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_LL_DISPATCH_CB59=3",
               "-iquote", str(tmp_path), "-iquote", str(ROOT),
               str(fixtures / "cb59_pair_host_emulation.cpp"), "-o", str(executable)]
    if os.name != "nt":
        command.append("-pthread")
    built = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=180)
    (tmp_path / "compile.log").write_text(built.stdout + built.stderr, encoding="utf-8")
    assert built.returncode == 0, built.stderr[-8000:]
    executed = subprocess.run([str(executable)], capture_output=True, text=True, errors="replace", timeout=120)
    (tmp_path / "execute.log").write_text(executed.stdout + executed.stderr, encoding="utf-8")
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "real pair headers passed" in executed.stdout


def test_pair_device_context_propagates_without_clobbering_outer_context():
    helper = (ROOT / "csrc/kernels/dispatch/low_latency/common/ll_pair_receive.h").read_text(encoding="utf-8")
    handoff = (ROOT / "csrc/kernels/dispatch/low_latency/cb59_pair/pair_handoff.h").read_text(encoding="utf-8")
    assert "defined(DEEPEP_LL_PAIR_DEVICE_CONTEXT) && !defined(DEEPEP_LL_STREAM_DEVICE_CONTEXT)" in helper
    assert "#ifdef DEEPEP_LL_PAIR_DEFINED_STREAM_CONTEXT" in helper
    assert "#if defined(DEEPEP_LL_PAIR_DEVICE_CONTEXT) ||" in helper
    assert handoff.index('"../cb59_stream/stream_counts.h"') < handoff.index('"../common/ll_pair_receive.h"')
    assert "#ifndef DEEPEP_LL_PAIR_DEVICE_CONTEXT" in handoff
