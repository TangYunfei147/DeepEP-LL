"""Run the standalone two-rank RDMA primitive, locally or one rank per host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from environment import environment

ROOT = Path(__file__).resolve().parents[2]
VISIBILITY = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES")


def devices(value):
    if not value or any(not part.isdecimal() for part in value.split(",")):
        raise argparse.ArgumentTypeError("devices must be comma-separated physical device numbers")
    if len(set(value.split(","))) != len(value.split(",")):
        raise argparse.ArgumentTypeError("devices must be distinct")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", type=devices)
    p.add_argument("--validation", action="store_true", help="A5 only: force physical 0,1; local two-rank primitive")
    p.add_argument("--rank", type=int, choices=(0, 1), help="One rank per host; omit for local two-rank launch")
    p.add_argument("--endpoint", required=True, help="SHMEM tcp://rank0-address:port")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--cores", type=int, default=0)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    if args.validation:
        if args.rank is not None or args.devices not in (None, "0,1"):
            p.error("A5 validation permits only local ranks on devices 0,1")
        args.devices = "0,1"
    if not args.devices:
        p.error("Target execution requires an explicit --devices list")
    if len(args.devices.split(",")) != (2 if args.rank is None else 1):
        p.error("Local smoke needs two visible devices; one-rank-per-host smoke needs one")
    if not 1 <= args.rounds <= 10000 or args.timeout < 1 or (args.cores and (args.cores < 16 or args.cores > 72 or args.cores % 2)):
        p.error("Invalid rounds, timeout or even 16..72 core count")
    env = dict(environment(), **{name: args.devices for name in VISIBILITY})
    log_root = ROOT / ".logs"
    log_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rdma-smoke-", dir=log_root))
    check = [sys.executable, str(Path(__file__).with_name("preflight.py")), "--output", str(directory / "preflight.json")]
    if args.validation:
        check.append("--validation")
    if args.rank is not None:  # Cross-host only: local MTE does not require an RDMA NIC.
        subprocess.run(check, env=env, check=True)
    binary = ROOT / "build/ll_rdma/rdma_smoke"
    if not binary.is_file():
        p.error("Build tests/ll_rdma/build_smoke.sh first")
    children, logs = [], []
    try:
        for rank in ((0, 1) if args.rank is None else (args.rank,)):
            log = (directory / f"rank-{rank}.log").open("w")
            logs.append(log)
            command = [str(binary), str(rank), "2", str(rank if args.rank is None else 0), args.endpoint,
                       str(args.rounds), str(args.cores), "2" if args.rank is None else "1"]
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        while any(child.poll() is None for child in children):
            if any(child.returncode not in (None, 0) for child in children):
                raise RuntimeError("RDMA smoke failed; inspect per-rank logs")
            if time.monotonic() > deadline:
                raise TimeoutError("RDMA smoke timeout; no correctness/performance retries")
            time.sleep(0.2)
        if any(child.returncode for child in children):
            raise RuntimeError("RDMA smoke failed; inspect per-rank logs")
        (directory / "result.json").write_text(json.dumps({"status": "passed", "scope": "two-rank primitive only",
            "transport": "mte" if args.rank is None else "rdma-put",
            "rank": args.rank, "rounds": args.rounds, "cores": args.cores, "devices": args.devices}) + "\n")
        print(f"PASS: {directory}; full LL and performance remain separate")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for log in logs:
            log.close()
        print(f"Logs: {directory}")


if __name__ == "__main__":
    main()
