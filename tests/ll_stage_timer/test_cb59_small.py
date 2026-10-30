"""Verified compile-time specialization; the original cb59 files stay intact."""
from pathlib import Path
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]

import runpy
_reference_contract = runpy.run_path(str(ROOT/'tests/ll_reference_profile/business_contract.py'))
spec = importlib.util.spec_from_file_location("cb59_small_generator", ROOT / "scripts/generate_cb59_small.py")
small = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(small)


def compiler():
    configured = shlex.split(os.environ.get("CXX", ""))
    if configured:
        return configured
    found = shutil.which("g++") or shutil.which("clang++")
    if not found:
        pytest.skip("native C++ compiler required for constant-branch proof")
    return [found]


def compile_run(tmp_path, source):
    cpp = tmp_path / "small.cpp"
    cpp.write_text(source, encoding="utf-8")
    exe = tmp_path / ("small.exe" if os.name == "nt" else "small")
    result = subprocess.run([*compiler(), "-std=c++17", "-O2", "-UNDEBUG", "-iquote", str(ROOT),
                             "-DDEEPEP_LL_DISPATCH_CB59=1", str(cpp), "-o", str(exe)],
                            capture_output=True, text=True, errors="replace", timeout=180)
    (tmp_path / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr[-6000:]
    subprocess.run([str(exe)], check=True, timeout=30)


def test_generation_and_full_business_sync_equivalence():
    before = {name: (small.SOURCE / name).read_bytes() for name in small.FILES}
    manifest = small.generate(check=True)
    assert before == {name: (small.SOURCE / name).read_bytes() for name in small.FILES}
    for entry in manifest["files"].values():
        assert entry["replay_business_sha256"] == entry["normalized_business_sha256"]
        assert entry["replay_synchronization_sha256"] == entry["normalized_synchronization_sha256"]


@pytest.mark.parametrize("relative,old,new", [
    (small.FILES[1], "static constexpr bool useAllCoreCount_ = true;", "static constexpr bool useAllCoreCount_ = false;"),
    (small.FILES[2], "static constexpr bool enableDedup_ = false;", "bool enableDedup_ = false;"),
    (small.FILES[2], "if constexpr (enableDedup_)", "if (enableDedup_)"),
    (small.FILES[2], "expert % localExperts_", "expert / localExperts_"),
    (small.FILES[1], "SyncFunc<AscendC::HardEvent::V_S>();", "SyncFunc<AscendC::HardEvent::S_V>();"),
    (small.FILES[3], "constexpr bool firstHop = enableDedup_ &&", "const bool firstHop = enableDedup_ &&"),
])
def test_mutations_cannot_pass_small_equivalence(relative, old, new):
    source = (small.DEST / relative).read_text(encoding="utf-8")
    assert old in source
    entry = json.loads(small.REPLAY_MANIFEST.read_text(encoding="utf-8"))["files"][relative]
    with pytest.raises(ValueError):
        small.validate_file(relative, source.replace(old, new, 1), entry)


def test_actual_constexpr_selection_discards_dedup_for_all_seven_modes(tmp_path):
    sender = (small.DEST / small.FILES[2]).read_text(encoding="utf-8")
    receiver = (small.DEST / small.FILES[3]).read_text(encoding="utf-8")
    constant = re.search(r"static constexpr bool enableDedup_ = false;", sender).group()
    dispatch = re.search(r"if constexpr \(enableDedup_\)\s+SendWithDedup\(input\);\s+else\s+SendWithoutDedup\(input\);", sender).group()
    first_hop = re.search(r"constexpr bool firstHop = enableDedup_ &&[^;]+;", receiver).group()
    # An attempted instantiation of either dedup member makes compilation fail.
    # These are the actual production constants and branch expressions, not a
    # runtime-false wrapper around the generic kernel.
    compile_run(tmp_path, r'''
#include <cassert>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"
using DispatchDedup::EpServerDedupTokenType;
struct HeaderData { unsigned GetValue(unsigned) { assert(false); return 0; } };
struct Tail { HeaderData view; HeaderData& Header() { assert(false); return view; } };
template<int Mode> struct Selection {
''' + constant + r'''
    unsigned calls = 0;
    void SendWithDedup(int&) { static_assert(Mode < 0, "dedup body was instantiated"); }
    void SendWithoutDedup(int&) { ++calls; }
    void Relay() { static_assert(Mode < 0, "relay body was instantiated"); }
    void Run() {
        int input = 0;
''' + dispatch + r'''
        Tail tail;
''' + first_hop + r'''
        if constexpr (firstHop) Relay();
    }
};
template<int Mode> void Check() { Selection<Mode> s; s.Run(); assert(s.calls == 1); }
int main() { Check<0>(); Check<1>(); Check<2>(); Check<3>(); Check<4>(); Check<5>(); Check<6>(); }
''')


def test_actual_all_core_count_selection_and_unchanged_ub_layout(tmp_path):
    counts = (small.DEST / small.FILES[1]).read_text(encoding="utf-8")
    constant = re.search(r"static constexpr bool useAllCoreCount_ = true;", counts).group()
    participant = re.search(r"const uint32_t cores = useAllCoreCount_ \? coreNum_ : aivUsedCumSum_;", counts).group()
    owner = re.search(r"const uint32_t core = useAllCoreCount_ \? aivId_ : aivId_ - aivUsedAllToAll_;", counts).group()
    compile_run(tmp_path, r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/cb59_small/count_layout.h"
struct Selection {
''' + constant + r'''
  uint32_t coreNum_ = 56, aivUsedCumSum_ = 28, aivUsedAllToAll_ = 28, aivId_ = 0;
  void Check() {
''' + participant + owner + r'''
    assert(cores == 56 && core == aivId_);
  }
};
int main() {
  for (unsigned core=0; core<56; ++core) { Selection s; s.aivId_=core; s.Check(); }
  using namespace DispatchDedup;
  static_assert(Cb59Small::kEpServerDedupCountUbBytes == 194560U);
  for (unsigned experts : {16U,32U,128U,384U,1024U}) {
    const auto a=Cb59Small::DispatchServerDedupCountLayout(experts,56);
    const auto b=Cb59Replay::DispatchServerDedupCountLayout(experts,56);
    assert(a.stateBytes==b.stateBytes && a.prefixStride==b.prefixStride && a.cumsumFlagOffset==b.cumsumFlagOffset);
  }
}
''')


def test_small_keeps_runtime_packet_layout_and_route_ownership():
    sender = (small.DEST / small.FILES[2]).read_text(encoding="utf-8")
    receiver = (small.DEST / small.FILES[3]).read_text(encoding="utf-8")
    assert 'if(enableDedup_&&count_==0U)return;' in _reference_contract['compact'](sender)
    assert "if constexpr (enableDedup_ && count_ == 0U)" not in sender
    assert "if (!enableDedup_ && !keepFlags)" in receiver
    assert "if constexpr (!enableDedup_ && !keepFlags)" not in receiver
    # Count and token ownership differ at small B; a zero token partition must
    # not silently drop the flattened routes owned by that sender.
    assert "const uint32_t total = bs_ * k_, core = GetBlockIdx();" in sender
    assert "ServerDedupDispatchDirectOrder(serverNum, ownServer, step)" in sender
