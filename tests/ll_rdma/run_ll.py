"""Explicit target LL transport selection; forward sampling to the established LL test."""
import argparse
import os
from pathlib import Path
import subprocess
import runpy
import sys
from run_smoke import devices, VISIBILITY
from environment import environment

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--backend", choices=("mte", "rdma"), required=True)
    p.add_argument("--devices", type=devices, required=True)
    p.add_argument("--validation", action="store_true", help="A5 allows CPU --plan only for native LL")
    args, remaining = p.parse_known_args()
    if args.validation and (args.devices != "0,1" or "--plan" not in remaining):
        p.error("A5 native LL validation is CPU --plan only, with devices 0,1; never fabricate 16 ranks")
    env = dict(environment(), DEEPEP_LL_BACKEND=args.backend, **{key: args.devices for key in VISIBILITY})
    test = ROOT / "tests/elastic/test_ep_low_latency.py"
    st = runpy.run_path(str(test))
    options = st["parse_args"](remaining)
    world, _, _ = st["launch_config"](options, env)
    st["validate_args"](options, world)
    local_world = int(env.get("LOCAL_WORLD_SIZE", options.num_processes))
    if args.backend == "rdma" and local_world != 16:
        p.error("Hybrid LL requires 16 contiguous ranks per physical host; logical G is selected separately")
    if "--plan" not in remaining and args.backend == "rdma" and world > 16:
        subprocess.run([sys.executable, str(Path(__file__).with_name("preflight.py"))], env=env, cwd=ROOT, check=True)
    return subprocess.call([sys.executable, str(test), *remaining], env=env, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
