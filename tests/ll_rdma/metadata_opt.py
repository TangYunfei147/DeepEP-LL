"""M=4 metadata experiment; delegate build, identity, sampling and reports to one_click."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def experiment(argv, environment):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("count_method", choices=("scalar", "histogram"))
    parser.add_argument("one_click_args", nargs=argparse.REMAINDER,
                        help="NODE MASTER RUN_NAME followed by one_click options")
    args = parser.parse_args(argv)
    if not args.one_click_args:
        parser.error("Supply NODE MASTER RUN_NAME and one_click options")
    if any(arg == "--skip-build" or arg.startswith("--skip-build=") for arg in args.one_click_args):
        parser.error("Each experiment requires a complete wheel rebuild; --skip-build is disabled")
    env = dict(environment)
    env["DEEPEP_LL_METADATA_CORES"] = "4"
    env["DEEPEP_LL_COUNT_HISTOGRAM"] = "ON" if args.count_method == "histogram" else "OFF"
    # The plan path needs neither Bash, the vendor environment, nor a device.
    entry = ([sys.executable, "tests/ll_rdma/one_click.py"] if "--plan" in args.one_click_args
             else ["bash", "tests/ll_rdma/one_click.sh"])
    return [*entry, *args.one_click_args], env


def main(argv=None):
    command, env = experiment(sys.argv[1:] if argv is None else argv, os.environ)
    print("Metadata experiment: M=4, histogram=" + env["DEEPEP_LL_COUNT_HISTOGRAM"], flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
