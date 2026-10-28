"""Native SDK capacity admission; CPU only, no CANN or device qualification."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_limit_precedes_initialization():
    session = (ROOT / "csrc/runtime/session.cpp").read_text()
    guard = session.index("validate_ll_rdma_qps(ll_rdma_qps_, size > int(DispatchDedup::kLlPhysicalRanksPerHost))")
    assert guard < session.index("occupied = true")
    assert guard < session.index("aclshmemx_set_qp_num(")
    assert guard < session.index("aclshmemx_init_attr(")
    assert "ll_rdma_qps_ = query_ll_core_config().active;" in session
    assert "low_latency_cores_ = ll_rdma_qps_;" in session
    assert "uint64_t(low_latency_cores_) * LlRdma::kCoreBytes" in session
    smoke = (ROOT / "tests/ll_rdma/smoke_host.cpp").read_text()
    assert smoke.index("validate_ll_rdma_qps(cores, ranksPerHost==1)") < smoke.index("aclshmemx_set_qp_num(")


def test_native_limit_boundary(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("C++ compiler required for executable capacity check")
    source = tmp_path / "qp_limit.cpp"
    source.write_text(r'''
#include "runtime/ll_rdma_qp_config.hpp"
#include <cassert>
int main() {
    static_assert(deepep::kLlNativeRoceQpLimit == 56);
    for (unsigned cores = 16; cores <= 72; cores += 2) {
        deepep::validate_ll_rdma_qps(cores, false); // local MTE retains LL capacity
        bool rejected = false;
        try { deepep::validate_ll_rdma_qps(cores, true); }
        catch (const std::runtime_error& e) {
            rejected = true;
            assert(std::string(e.what()).find(std::to_string(cores)) != std::string::npos);
            assert(std::string(e.what()).find("56") != std::string::npos);
        }
        assert(rejected == (cores > 56));
    }
}
''')
    binary = tmp_path / "qp_limit.exe"
    command = [zig, "c++"] if zig else [compiler]
    subprocess.run([*command, "-std=c++17", "-UNDEBUG", "-I" + str(ROOT / "csrc"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
