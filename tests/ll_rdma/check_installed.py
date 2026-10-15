"""Installed Host/device capability and rejected-selector checks; no device launch."""
import argparse
import os
from pathlib import Path
import runpy
from environment import environment

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rdma", type=int, choices=(0, 1), required=True)
parser.add_argument("--timer", type=int, choices=(0, 1), default=0)
args = parser.parse_args()
os.environ.update(environment())
helpers = runpy.run_path(str(ROOT / "tests/elastic/test_dispatch.py"))
package = helpers["_import_installed_deep_ep"]()
from deep_ep._native import require_native
native = require_native("RDMA build capability check")
assert native.low_latency_core_protocol == 3
assert bool(native.low_latency_rdma_enabled) == bool(args.rdma)
assert bool(native.low_latency_stage_timer_enabled) == bool(args.timer)
os.environ["DEEPEP_LL_BACKEND"] = "invalid-selector"
try:
    native.Runtime(0, 2, 2 << 20, "tcp://127.0.0.1:1", 1)
except ValueError as exc:
    assert "must be mte or rdma" in str(exc)
else:
    raise AssertionError("Invalid selector was accepted")
if not args.rdma:
    os.environ["DEEPEP_LL_BACKEND"] = "rdma"
    try:
        native.Runtime(0, 2, 2 << 20, "tcp://127.0.0.1:1", 1)
    except RuntimeError as exc:
        assert "no MTE fallback" in str(exc)
    else:
        raise AssertionError("Disabled RDMA was accepted")
print(f"PASS installed {package.__file__}: ABI=3 RDMA={args.rdma} timer={args.timer}; Host negative checks only")
