#!/usr/bin/env python3
"""Build three fixed wheels, then serially compare timer removal and stream receive.

Planning and analysis use only the standard library. Build/run commands are for
the user's Linux target; this helper never automatically retries a failed measurement. SDK
admission, manifests, capabilities and library hashes are not checked.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import statistics
import subprocess
import types
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = "439d4597a52b657858217bac7fc1e965f22507f7"
LIMIT = 60000
BATCHES = (1, 2, 4, 8, 16, 24, 32)
VISIBILITY = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES")
VARIANTS = {
    "replay-off": dict(source="source-normal", mode="normal-off", dispatch_impl="cb59", implementation="cb59-replay", protocol=21028, dispatch_bytes=152, combine_bytes=96),
    "replay-clean": dict(source="source-clean", mode="clean-source", dispatch_impl="cb59", implementation="cb59-replay", protocol=21028, dispatch_bytes=136, combine_bytes=88),
    "stream-clean": dict(source="source-clean", mode="clean-source", dispatch_impl="cb59-stream", implementation="cb59-stream-recv", protocol=22028, dispatch_bytes=136, combine_bytes=88),
}
PAIRS = {"timer": ("replay-off", "replay-clean"), "optimization": ("replay-clean", "stream-clean")}
BUILD_FORMAT = "ll-no-timer-ab-build-v1"
RUN_FORMAT = "ll-no-timer-ab-runs-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = value.add_subparsers(dest="action", required=True)
    for name in ("build", "run", "analyze"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--run-name", required=True)
        command.add_argument("--output-root", type=Path, default=Path("diagnostic-runs"))
        if name != "analyze":
            command.add_argument("--plan", action="store_true")
        if name == "run":
            command.add_argument("--resume", action="store_true", help="Explicitly resume the same saved plan; retain complete ABBA blocks and repeat incomplete blocks in new directories")
            command.add_argument("--stage", choices=("timer", "optimization", "all"), default="all")
            command.add_argument("--devices", required=True)
            command.add_argument("--master-addr", required=True)
            command.add_argument("--master-port", required=True, type=int)
            command.add_argument("--shmem-port", required=True, type=int)
            command.add_argument("--seeds", nargs="+", type=int, default=[0], help="Default seed0; up to four explicit seeds, alternating ABBA/BAAB")
    for name in ("_preimage", "_dependencies", "_install", "_verify-host"):
        command = commands.add_parser(name, help=argparse.SUPPRESS, allow_abbrev=False)
        command.add_argument("--group", type=Path, required=name in {"_install", "_verify-host"})
        command.add_argument("--source", type=Path, required=name == "_preimage")
        command.add_argument("--variant", choices=tuple(VARIANTS), required=name in {"_install", "_verify-host"})
        command.add_argument("--controller", type=Path, default=ROOT)
    return value


def _local(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    _require(".." not in root.parts and ".." not in path.parts, "Paths may not contain '..'")
    _require(path.is_relative_to(root), "Path must remain in the selected project/group")
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                 "Paths may not traverse symlinks or junctions")
    _require(path.resolve() == path and root.resolve() == root, "Path resolves outside the intended location")
    return path


def group_path(args, root=ROOT):
    root = Path(root).absolute()
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name), "Use a fresh ASCII --run-name")
    parent = args.output_root if args.output_root.is_absolute() else root / args.output_root
    group = _local(root, parent / args.run_name)
    _require(group != root and group.relative_to(root).parts[0] not in {
        ".git", "build", "third-party", "csrc", "deep_ep", "scripts", "tests", "cmake", "benchmarks", "experiments"},
        "Use a diagnostic directory outside source/dependency/build trees")
    return group


def _layout(group, variant):
    group = Path(group)
    return dict(source=group / VARIANTS[variant]["source"], venv=group / "venvs" / variant,
                python=group / "venvs" / variant / "bin/python", wheels=group / "wheels" / variant,
                host=group / "host" / f"{variant}.json")


def build_plan(group, root=ROOT, python=sys.executable):
    """Pure command planning; no imports, subprocesses or filesystem writes."""
    group, root = Path(group).absolute(), Path(root).absolute()
    helper = root / "tests/ll_stage_timer/no_timer_ab.py"
    projection = root / "scripts/prepare_ll_no_timer.py"
    steps = []
    def add(name, argv, cwd, variant="replay-off"):
        steps.append(dict(name=name, argv=[str(x) for x in argv], cwd=str(cwd), variant=variant))
    for mode, dirname, variant in (("normal-off", "source-normal", "replay-off"), ("clean-source", "source-clean", "replay-clean")):
        source = group / dirname
        add(dirname + "-clone", ["git", "-c", "core.autocrlf=false", "clone", "--no-hardlinks", "--dissociate", "--no-checkout", root, source], group, variant)
        add(dirname + "-checkout", ["git", "-c", "core.autocrlf=false", "checkout", "--detach", BASE], source, variant)
        add(dirname + "-preimage", [python, "-I", helper, "_preimage", "--source", source], source, variant)
        add(dirname + "-submodules", ["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--", "third-party/shmem"], source, variant)
        add(dirname + "-prepare", [python, "-I", projection, "_prepare", "--source", source, "--mode", mode, "--controller", root], source, variant)
    for variant in VARIANTS:
        layout = _layout(group, variant)
        add(variant + "-venv", [python, "-m", "venv", "--system-site-packages", layout["venv"]], layout["source"], variant)
        add(variant + "-dependencies", [layout["python"], "-I", helper, "_dependencies"], layout["source"], variant)
    normal = _layout(group, "replay-off")
    add("shared-sdk-once", [normal["python"], normal["source"] / "scripts/build_shmem.py"], normal["source"])
    for variant, selection in VARIANTS.items():
        layout = _layout(group, variant)
        add(variant + "-source-check", [python, "-I", projection, "_verify", "--source", layout["source"], "--mode", selection["mode"], "--controller", root], layout["source"], variant)
        add(variant + "-clean-wheel", [layout["python"], layout["source"] / "scripts/clean_build_artifacts.py", "--wheel"], layout["source"], variant)
        add(variant + "-wheel", [layout["python"], "-m", "build", "--wheel", "--no-isolation", "--outdir", layout["wheels"], layout["source"]], layout["source"], variant)
    # All native builds finish before installation/Host verification, and all
    # verifications finish before the relative wheel map becomes consumable.
    for variant in VARIANTS:
        layout = _layout(group, variant)
        for action in ("_install", "_verify-host"):
            add(variant + action, [layout["python"], "-I", helper, action, "--group", group,
                                  "--variant", variant, "--controller", root], layout["source"], variant)
    return dict(format=BUILD_FORMAT, reference=BASE, root=str(root), group=str(group), python=str(python),
                steps=steps, shared_sdk="source-normal/third-party/shmem/install/shmem",
                variants=VARIANTS, receipt="build-receipt.json", wheel_map="wheel-map.json",
                note="One clean SDK build, three clean full OFF wheels, three isolated venvs; no device work")


def _environment(group, variant, *, run=None, devices=None, inherited=None):
    group = Path(group)
    env = dict(os.environ if inherited is None else inherited)
    for name in tuple(env):
        if name.startswith("DEEPEP_") or name in {"PYTHONPATH", "PYTHONHOME", "SHMEM_ROOT", "SHMEM_SOURCE_PATH", "CATLASS_ROOT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"}:
            env.pop(name)
    selected = VARIANTS[variant]
    source = group / selected["source"]
    local = source / ".cache/no-timer-ab" / (run or ("build-" + variant))
    for key, suffix in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache", "PIP_CACHE_DIR": "cache/pip",
                        "TORCH_EXTENSIONS_DIR": "cache/torch-extensions", "ASCEND_CACHE_PATH": "cache/ascend",
                        "ASCEND_WORK_PATH": "logs/ascend", "ASCEND_PROCESS_LOG_PATH": "logs/ascend", "SHMEM_LOG_PATH": "logs/shmem"}.items():
        env[key] = str(local / suffix)
    sdk = group / "source-normal/third-party/shmem/install/shmem"
    env.update(SHMEM_ROOT=str(sdk), DEEPEP_NPU_ARCH="Ascend950", DEEPEP_BUILD_TYPE="Release", DEEPEP_BUILD_MEGAMOE="OFF",
               DEEPEP_MEGAMOE_TIMER="OFF", DEEPEP_BUILD_EP_GMM_FUSED="OFF", DEEPEP_LL_STAGE_TIMER="OFF",
               DEEPEP_LL_DISPATCH_IMPL=selected["dispatch_impl"], DEEPEP_LL_SEND_CORES="28", DEEPEP_LL_RECV_CORES="28",
               DEEPEP_LL_SMALL_B_PREFIX_CORES="0", DEEPEP_LL_RANKS_PER_GROUP="16", DEEPEP_LL_TIMER_PROFILE="sparse",
               DEEPEP_LL_BACKEND="mte", CCACHE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if devices is not None:
        env.update({name: ",".join(map(str, devices)) for name in VISIBILITY})
    return env


def _provision_environment(group, env):
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
        _local(group, Path(env[key])).mkdir(parents=True, exist_ok=True)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write("\n")


def _git(source, *args):
    result = subprocess.run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(source), *args],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90,
                            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0"))
    return result.stdout.strip()


def verify_preimage(source):
    source = Path(source).absolute()
    _require(Path(_git(source, "rev-parse", "--show-toplevel")).resolve() == source.resolve(), "Source must be its own repository")
    _require(_git(source, "rev-parse", "HEAD") == BASE, "Source preimage differs from the fixed base")
    _require(not _git(source, "status", "--porcelain", "--untracked-files=all"), "Source preimage is dirty; files were preserved")
    _require(not (source / ".git/objects/info/alternates").exists(), "Clone must own its Git objects")


def _projection(controller=ROOT):
    path = Path(controller) / "scripts/prepare_ll_no_timer.py"
    module = types.ModuleType("no_timer_projection")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _proof(group, variant, controller=ROOT):
    selection = VARIANTS[variant]
    proof = _projection(controller).verify(_layout(group, variant)["source"], selection["mode"], controller=Path(controller))
    _require(proof.get("reference") == BASE and proof.get("mode") == selection["mode"]
             and re.fullmatch(r"[0-9a-f]{64}", proof.get("source_sha256", "")), "Projection proof has the wrong source identity")
    _require(proof.get("dispatch_launch_bytes") == selection["dispatch_bytes"] and proof.get("combine_launch_bytes") == selection["combine_bytes"], "Projection launch sizes differ")
    return proof


def _one_wheel(group, variant):
    folder = _local(group, _layout(group, variant)["wheels"])
    wheels = sorted(folder.glob("*.whl"))
    _require(len(wheels) == 1, f"Expected exactly one new full wheel for {variant}")
    return _local(group, wheels[0])


def verify_dependencies():
    missing = [name for name in ("torch", "torch_npu", "build", "setuptools", "wheel", "pybind11")
               if importlib.util.find_spec(name) is None]
    _require(not missing, "Select an existing environment with build/runtime dependencies; no downloads are performed: " + ", ".join(missing))


def install(group, variant):
    layout = _layout(group, variant)
    subprocess.run([str(layout["python"]), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(_one_wheel(group, variant))], cwd=layout["source"], check=True)


def verify_host(group, variant, controller=ROOT):
    """Load only the installed native module; do not query device capabilities."""
    proof = _proof(group, variant, controller)
    layout, expected = _layout(group, variant), VARIANTS[variant]
    import deep_ep
    from deep_ep._native import require_native
    package = Path(deep_ep.__file__).resolve().parent
    _require(package.is_relative_to(layout["venv"].resolve()), "Imported package is outside this variant's venv")
    native = require_native("no-timer matched build Host verification")
    attrs = dict(runtime_protocol=1, low_latency_dispatch_impl=expected["dispatch_impl"],
                 low_latency_implementation=expected["implementation"], low_latency_core_protocol=expected["protocol"],
                 low_latency_ranks_per_group=16, low_latency_small_b_prefix_cores=0,
                 low_latency_instrumentation_source=expected["mode"], dispatch_launch_bytes=expected["dispatch_bytes"],
                 combine_launch_bytes=expected["combine_bytes"])
    for name, value in attrs.items():
        _require(getattr(native, name, None) == value, "Installed native identity differs: " + name)
    _require(not getattr(native, "low_latency_stage_timer_enabled", False), "An ON wheel is forbidden")
    info = _json(package / "_build_info.json")
    for name, value in dict(source_sha256=proof["source_sha256"], ll_dispatch_impl=expected["dispatch_impl"],
                            ll_send_cores=28, ll_recv_cores=28, ll_small_b_prefix_cores=0, arch="Ascend950", build_type="Release",
                            ll_instrumentation_source=expected["mode"], dispatch_launch_bytes=expected["dispatch_bytes"],
                            combine_launch_bytes=expected["combine_bytes"]).items():
        _require(info.get(name) == value, "Installed build identity differs: " + name)
    _require(not info.get("ll_stage_timer", False), "An ON build identity is forbidden")
    if expected["mode"] == "normal-off":
        _require(info.get("ll_stage_timer") is False and getattr(native, "low_latency_stage_timer_enabled", None) is False,
                 "The normal arm must be a complete timer-OFF build")
    result = dict(variant=variant, host_only=True, device_queried=False, source_sha256=proof["source_sha256"], attributes=attrs,
                  build_metadata={key: info.get(key) for key in ("cann", "compiler", "compiler_path", "compiler_sha256", "device_compiler_sha256", "python", "arch", "build_type")})
    layout["host"].parent.mkdir(parents=True, exist_ok=True)
    _write(layout["host"], result)
    return result


def _command(argv, cwd, env, log):
    print("[no-timer-ab] " + Path(log).name, flush=True)
    with Path(log).open("xb") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def execute_build(plan):
    _require(platform.system() == "Linux", "Build execution requires the Linux target; --plan is CPU-only")
    _require(plan == build_plan(plan["group"], plan["root"], plan["python"]), "Build plan differs from the fixed three-variant plan")
    group, root = _local(plan["root"], Path(plan["group"])), Path(plan["root"])
    _require(not group.exists(), "Build group already exists; no files were changed")
    _git(root, "cat-file", "-e", BASE + "^{commit}")
    controller_revision = _git(root, "rev-parse", "HEAD")
    _require(not _git(root, "status", "--porcelain", "--untracked-files=no"), "Controller checkout has tracked changes; commit/stash intentionally before building")
    group.mkdir(parents=True, exist_ok=False)
    (group / "logs/build").mkdir(parents=True)
    _write(group / "build-plan.json", plan, exclusive=True)
    for index, step in enumerate(plan["steps"]):
        env = _environment(group, step["variant"])
        # Do not create anything inside a source before git clone/preimage.
        if step["name"].endswith(("-clone", "-checkout", "-preimage")):
            scratch = group / "bootstrap"
            for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
                env[key] = str(scratch / key.lower())
        _provision_environment(group, env)
        _command(step["argv"], Path(step["cwd"]), env, group / "logs/build" / f"{index:02d}-{step['name']}.log")
    variants = {}
    for variant, selection in VARIANTS.items():
        proof = _proof(group, variant, root)
        host = _json(_layout(group, variant)["host"])
        _require(host.get("source_sha256") == proof["source_sha256"] and host.get("host_only") is True
                 and host.get("device_queried") is False, "All wheels must pass matching Host verification before publication")
        wheel = _one_wheel(group, variant)
        layout = _layout(group, variant)
        variants[variant] = dict(**selection, source_sha256=proof["source_sha256"],
            wheel=wheel.relative_to(group).as_posix(), python=layout["python"].relative_to(group).as_posix(),
            host=layout["host"].relative_to(group).as_posix(), proof=proof,
            wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    receipt = dict(format=BUILD_FORMAT, reference=BASE, controller_revision=controller_revision,
                   shared_sdk=plan["shared_sdk"], sdk_builds=1, variants=variants,
                   note="Same SDK by construction: one clean normal-source build and explicit shared SHMEM_ROOT; no SDK attestation")
    _write(group / "build-receipt.json", receipt, exclusive=True)
    _write(group / "wheel-map.json", dict(format=BUILD_FORMAT, reference=BASE, variants={name: item["wheel"] for name, item in variants.items()}), exclusive=True)
    return group / "wheel-map.json"


def _build_receipt(group):
    group = Path(group)
    receipt, mapping = _json(_local(group, group / "build-receipt.json")), _json(_local(group, group / "wheel-map.json"))
    _require(receipt.get("format") == mapping.get("format") == BUILD_FORMAT and receipt.get("reference") == mapping.get("reference") == BASE,
             "Missing or incompatible completed build receipt/map")
    _require(set(receipt.get("variants", {})) == set(mapping.get("variants", {})) == set(VARIANTS), "All three variants must be built first")
    _require(receipt.get("sdk_builds") == 1 and receipt.get("shared_sdk") == "source-normal/third-party/shmem/install/shmem", "Build did not use the shared single-build plan")
    for name, expected in VARIANTS.items():
        record = receipt["variants"][name]
        _require(all(record.get(key) == value for key, value in expected.items()), "Build variant differs from the fixed matrix")
        _require(mapping["variants"][name] == record.get("wheel"), "Wheel map differs from completed build receipt")
        for key in ("wheel", "python", "host"):
            _require(isinstance(record.get(key), str) and not Path(record[key]).is_absolute(), "Build paths must remain relative")
            path = group / record[key]
            if key == "python":
                _require(path == _layout(group, name)["python"], "Venv interpreter path differs from the variant")
                # POSIX venv creation normally links this final executable to
                # the selected existing Python. Its parent must remain local;
                # imported deep_ep ownership is checked by every Host preflight.
                _local(group, path.parent)
            else:
                _local(group, path)
            _require(path.is_file(), "Build artifact is missing: " + key)
        _require(hashlib.sha256((group / record["wheel"]).read_bytes()).hexdigest() == record.get("wheel_sha256"), "Project wheel bytes changed after Host verification")
    return receipt


def make_run_plan(args, root=ROOT):
    group = group_path(args, root)
    pieces = args.devices.split(",")
    _require(len(pieces) == 16 and all(value.isdecimal() for value in pieces) and len(set(map(int, pieces))) == 16,
             "Supply exactly 16 distinct target devices")
    _require(args.master_addr and not any(value.isspace() for value in args.master_addr), "Supply the allocated master address")
    _require(1 <= args.master_port <= 65535 and 1 <= args.shmem_port <= 65535 and args.master_port != args.shmem_port, "Supply distinct valid job ports")
    _require(1 <= len(args.seeds) <= 4 and len(set(args.seeds)) == len(args.seeds)
             and all(type(value) is int and 0 <= value <= 2147483647 for value in args.seeds), "Choose one to four distinct nonnegative seeds")
    runs = []
    stages = list(PAIRS) if args.stage == "all" else [args.stage]
    for stage in stages:
        for route in (("normal",) if stage == "timer" else ("normal", "hotspot")):
            for seed_index, seed in enumerate(args.seeds):
                sides = "ABBA" if seed_index % 2 == 0 else "BAAB"
                repeats = {"A": 0, "B": 0}
                for order, side in enumerate(sides, 1):
                    repeats[side] += 1
                    variant = PAIRS[stage][side == "B"]
                    layout = _layout(group, variant)
                    name = f"{stage}-{route}-seed{seed}-order{order:02d}-{variant}"
                    directory = group / "runs" / name
                    argv = [str(layout["python"]), str(layout["source"] / "tests/ll_stage_timer/run.py"),
                            "--devices", args.devices, "--backend", "mte", "--num-processes", "16", "--num-nodes", "1", "--node-rank", "0",
                            "--master-addr", args.master_addr, "--master-port", str(args.master_port), "--shmem-port", str(args.shmem_port),
                            "--num-tokens", *map(str, BATCHES), "--num-max-tokens-per-rank", "32", "--hidden", "3584", "--num-topk", "8", "--num-experts", "384",
                            "--dtype", "fp8", "--fp8-scale-dtype", "int32", "--do-cpu-sync", "0", "--route", route, "--seed", str(seed),
                            "--correctness-repeats", "3", "--repeat-combine", "2", "--warmup", "10", "--num-iters", "100", "--perf-tail", "50",
                            "--delay-rows", "118260", "--delay-cols", "30720", "--delay-exp-before", "1", "--delay-exp-after", "0",
                            "--output-dir", str(group / "runs"), "--run-name", name, "--output", str(directory / "full-report.json")]
                    runs.append(dict(name=name, stage=stage, route=route, seed=seed, order=order, side=side, repeat=repeats[side],
                                     variant=variant, directory="runs/" + name, report="runs/" + name + "/full-report.json", command=argv))
    name = args.stage + "-seeds-" + "-".join(map(str, args.seeds))
    return dict(format=RUN_FORMAT, reference=BASE, root=str(Path(root).absolute()), group=str(group), stage=args.stage,
                devices=list(map(int, pieces)), seeds=list(args.seeds), batches=list(BATCHES), runs=runs,
                plan_file=f"run-plans/{name}.json", status_file=f"run-status/{name}.json",
                note="Full dispatch plus BF16 combine; 10/100/tail50, per-sample Exp+scalar all-reduce; no ON capture")


def _percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def _close(observed, expected, name):
    _require(type(observed) in (int, float) and math.isfinite(observed) and math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9), "Reported statistic differs from raw samples: " + name)


def validate_report(report, run, receipt):
    selected = receipt["variants"][run["variant"]]
    _require(report.get("status") == "completed" and report.get("world_size") == 16 and report.get("case_count") == report.get("result_count") == len(BATCHES), "Incomplete report or wrong actual EP/case coverage")
    config = report.get("config", {})
    expected = dict(num_processes=16, num_nodes=1, node_rank=0, num_max_tokens_per_rank=32, hidden=3584, num_topk=8, num_experts=384,
                    dtype="fp8", fp8_scale_dtype="int32", do_cpu_sync=0, warmup=10, num_iters=100, perf_tail=50, seed=run["seed"], route=run["route"],
                    delay_rows=118260, delay_cols=30720, delay_exp_before=1, delay_exp_after=0, group_backend="hccl", dispatch_only=False,
                    correctness_repeats=3, repeat_combine=2, skip_check=False, skip_perf_test=False)
    _require(all(config.get(key) == value for key, value in expected.items()), "Run configuration differs from the fixed dispatch+combine protocol")
    _require(config.get("num_tokens") == list(BATCHES), "B matrix differs")
    _require(not any(config.get(key) for key in ("ll_stage_timer", "ll_expert_timer", "ll_profile_capture", "use_1g_huge_pages")), "Timer capture or huge pages changed the comparison")
    cap_key = "ll_stage_timer_capabilities" if selected["mode"] == "normal-off" else "runtime_capabilities"
    caps = config.get(cap_key)
    _require(isinstance(caps, list) and len(caps) == 16 and all(value == caps[0] for value in caps), "Missing or inconsistent all-rank native identity")
    identity = dict(active_aiv=56, send_cores=28, recv_cores=28, logical_group=16, small_b_prefix_cores=0,
                    core_protocol=selected["protocol"], dispatch_impl=selected["dispatch_impl"], implementation=selected["implementation"],
                    instrumentation_source=selected["mode"], dispatch_launch_bytes=selected["dispatch_bytes"], combine_launch_bytes=selected["combine_bytes"],
                    source_sha256=selected["source_sha256"], transport="mte")
    _require(all(caps[0].get(key) == value for key, value in identity.items()) and not caps[0].get("stage_timer", False), "Native identity differs from the built variant")
    rows = []
    results = report.get("results", [])
    _require(len(results) == len(BATCHES), "Missing case results")
    for case_id, (batch, case) in enumerate(zip(BATCHES, results)):
        shape = dict(B=batch, M=32, H=3584, num_topk=8, num_experts=384, dtype="fp8", do_cpu_sync=False, fp8_scale_dtype="int32")
        _require(case.get("case_id") == case_id and case.get("route") == run["route"] and case.get("backend") == "ll"
                 and case.get("correctness") == "passed" and case.get("config") == shape, "Case shape/route/correctness differs")
        comparison = case.get("comparison", {})
        _require(all(comparison.get(key) == value for key, value in identity.items()), "Case native identity differs")
        _require(comparison.get("sampling_protocol") == "exp-reduce-stable-tail-v2" and comparison.get("dispatch_only") is False
                 and comparison.get("profile_seed") == run["seed"] and comparison.get("profile_route_fingerprint_schema") == 1,
                 "Sampling protocol or actual route seed is missing/mismatched")
        route_hash = comparison.get("profile_route_sha256", "")
        _require(isinstance(route_hash, str) and re.fullmatch(r"[0-9a-f]{64}", route_hash), "Actual route fingerprint is missing")
        for operation in ("dispatch", "combine"):
            perf = case.get(operation, {})
            _require(perf.get("kernel") == operation + "_server_dedup_kernel" and perf.get("measurement") == "kernel_duration_us"
                     and perf.get("perf_tail") == 50, "Both complete dispatch/combine profiler tasks are required")
            ranks = perf.get("ranks", [])
            _require(len(ranks) == 16 and {item.get("rank") for item in ranks} == set(range(16)), "Missing/duplicate rank samples")
            means, pooled = [], []
            for rank in ranks:
                raw = rank.get("kernel_us", [])
                _require(len(raw) == 100 and rank.get("samples") == 50 and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in raw), "Expected 100 finite samples/rank and tail50")
                mean = statistics.mean(raw[-50:])
                _close(rank.get("mean_us"), mean, "rank tail50 mean")
                means.append(mean)
                pooled.extend(raw[-50:])
            maximum, mean, p99 = max(means), statistics.mean(pooled), _percentile(pooled, .99)
            _close(perf.get("max_rank_mean_kernel_us"), maximum, "max rank tail50 mean")
            for key, value in (("mean_kernel_us", mean), ("p99_kernel_us", p99)):
                if key in perf:
                    _close(perf[key], value, key)
            rows.append(dict(stage=run["stage"], route=run["route"], seed=run["seed"], B=batch, operation=operation,
                             side=run["side"], repeat=run["repeat"], order=run["order"], variant=run["variant"], name=run["name"],
                             max_rank_mean_us=maximum, pooled_mean_us=mean, pooled_p99_us=p99, route_sha256=route_hash))
    return rows


def _validate_plan(plan):
    _require(plan.get("format") == RUN_FORMAT and plan.get("reference") == BASE, "Unknown saved run plan")
    # Recreate the public plan so edits cannot silently add a timer, shape,
    # source, retry or dispatch-only command to the approved matrix.
    first = plan["runs"][0]["command"]
    def value(flag):
        return first[first.index(flag) + 1]
    args = argparse.Namespace(run_name=Path(plan["group"]).name, output_root=Path(plan["group"]).parent, stage=plan["stage"],
        devices=",".join(map(str, plan["devices"])), master_addr=value("--master-addr"), master_port=int(value("--master-port")),
        shmem_port=int(value("--shmem-port")), seeds=plan["seeds"])
    _require(plan == make_run_plan(args, plan["root"]), "Run plan differs from the fixed ABBA matrix")


def _attempt_run(run, attempt):
    """Derive output locations; a saved state cannot supply arbitrary commands."""
    _require(type(attempt) is int and 0 <= attempt <= 999, "Invalid resume attempt")
    result = dict(run, command=list(run["command"]))
    if attempt:
        folder = f"resume{attempt:03d}"
        result["directory"] = "runs/" + folder + "/" + run["name"]
        result["report"] = result["directory"] + "/full-report.json"
        argv = result["command"]
        directory_index, report_index = argv.index("--output-dir") + 1, argv.index("--output") + 1
        output_root = Path(argv[directory_index]) / folder
        argv[directory_index] = str(output_root)
        argv[report_index] = str(output_root / run["name"] / "full-report.json")
    return result


def _run_log(group, run, attempt):
    directory = Path(group) / "logs/runs"
    if attempt:
        directory /= f"resume{attempt:03d}"
    return directory / (run["name"] + ".log")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save_state(group, plan, state):
    path = _local(group, group / plan["status_file"])
    temporary = _local(group, path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp"))
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(state, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


@contextmanager
def _run_lock(group):
    path = _local(group, group / ".run.lock")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("Run group is locked; another controller may be active. Inspect an interrupted process before handling its stale .run.lock") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(dict(pid=os.getpid(), started_utc=_utc_now()), output)
        yield
    finally:
        path.unlink()


def _read_state(group, plan):
    state = _json(_local(group, group / plan["status_file"]))
    _require(state.get("format") == RUN_FORMAT and state.get("status") in {"preflight", "running", "failed", "interrupted", "completed"}, "Invalid saved run state")
    records = state.get("runs")
    _require(isinstance(records, list) and len(records) <= len(plan["runs"]), "Invalid saved run records")
    for index, record in enumerate(records):
        _require(record.get("name") == plan["runs"][index]["name"] and record.get("status") in {"running", "failed", "completed"}, "Saved run records differ from the ordered plan")
        _attempt_run(plan["runs"][index], record.get("attempt", 0))
        _require(index == len(records) - 1 or record["status"] == "completed", "Nonterminal failed/running record in saved state")
    if state["status"] == "completed":
        _require(len(records) == len(plan["runs"]) and all(record["status"] == "completed" for record in records), "Run group is incomplete; no completed report issued")
    return state


def _completed_blocks(group, plan, state, receipt):
    """A block is reusable only after raw-sample and cross-arm route checks."""
    count = 0
    for start in range(0, len(plan["runs"]), 4):
        records = state["runs"][start:start + 4]
        if len(records) != 4 or any(record["status"] != "completed" for record in records):
            break
        _require(len({record.get("attempt", 0) for record in records}) == 1, "An ABBA block mixes resume attempts")
        rows = []
        for run, record in zip(plan["runs"][start:start + 4], records):
            selected = _attempt_run(run, record.get("attempt", 0))
            rows.extend(validate_report(_json(_local(group, group / selected["report"])), selected, receipt))
        summarize(rows)
        count += 4
    return count


def execute_run(plan, *, resume=False):
    _require(platform.system() == "Linux", "Run execution requires the user's Linux target; --plan is CPU-only")
    _validate_plan(plan)
    group = _local(plan["root"], Path(plan["group"]))
    receipt = _build_receipt(group)
    with _run_lock(group):
        return _execute_run_locked(plan, receipt, resume=resume)


def _execute_run_locked(plan, receipt, *, resume):
    group, root = Path(plan["group"]), Path(plan["root"])
    attempt, offset = 0, 0
    stem = Path(plan["plan_file"]).stem
    if resume:
        saved = _json(_local(group, group / plan["plan_file"]))
        _validate_plan(saved)
        _require(saved == plan, "Resume arguments differ from the original saved plan; reuse the same stage, seeds, devices, address and ports")
        previous = _read_state(group, plan)
        _require(previous["status"] in {"failed", "interrupted", "completed"}, "Saved run is still active; explicit resume requires failed/interrupted/completed state")
        offset = _completed_blocks(group, plan, previous, receipt)
        if previous["status"] == "completed":
            return analyze(group)
        history_dir = _local(group, group / "run-history" / stem)
        attempt = 1
        while _local(group, history_dir / f"resume{attempt:03d}.json").exists():
            attempt += 1
        _require(attempt <= 999, "Resume attempt limit reached")
        pending = [_attempt_run(run, attempt) for run in plan["runs"][offset:]]
        for run in pending:
            for path in (group / run["directory"], _run_log(group, run, attempt)):
                _require(not _local(group, path).exists(), "Resume output already exists; no files were changed")
        history_dir.mkdir(parents=True, exist_ok=True)
        history = (history_dir / f"resume{attempt:03d}.json").relative_to(group).as_posix()
        _write(group / history, dict(resumed_utc=_utc_now(), attempt=attempt, retained_blocks=offset // 4,
               rerun_names=[run["name"] for run in pending], previous_state=previous), exclusive=True)
        state = dict(format=RUN_FORMAT, status="preflight", runs=previous["runs"][:offset],
                     resume_history=[*previous.get("resume_history", []), history])
        print(f"[no-timer-ab] resume{attempt:03d}: retained {offset // 4} complete ABBA blocks; {len(pending)} sampling processes remain", flush=True)
    else:
        paths = [group / plan["plan_file"], group / plan["status_file"], *(group / run["directory"] for run in plan["runs"]),
                 *(_run_log(group, run, 0) for run in plan["runs"])]
        _require(all(not _local(group, path).exists() for path in paths), "A selected stage/seed/order already exists; use explicit --resume for the same saved plan")
        for path in sorted((group / "run-plans").glob("*.json")):
            other = _json(_local(group, path))
            _validate_plan(other)
            _require(not ({run["name"] for run in plan["runs"]} & {run["name"] for run in other["runs"]}), "A selected stage/seed/order already exists in another plan")
        state = dict(format=RUN_FORMAT, status="preflight", runs=[])
        pending = plan["runs"]
    session = stem + (f"-resume{attempt:03d}" if attempt else "")
    for directory in ("run-plans", "run-status", "logs/run-preflight", "logs/runs", "runs"):
        _local(group, group / directory).mkdir(parents=True, exist_ok=True)
    if not resume:
        _write(group / plan["plan_file"], plan, exclusive=True)
    _save_state(group, plan, state)
    try:
        # Recheck all source/installed identities before the first device run.
        for variant in VARIANTS:
            proof = _proof(group, variant, root)
            _require(proof["source_sha256"] == receipt["variants"][variant]["source_sha256"], "Frozen source changed after building")
            layout = _layout(group, variant)
            env = _environment(group, variant, run=session)
            _provision_environment(group, env)
            log = _local(group, group / "logs/run-preflight" / (session + "-" + variant + ".log"))
            _command([str(layout["python"]), "-I", str(root / "tests/ll_stage_timer/no_timer_ab.py"), "_verify-host", "--group", str(group), "--variant", variant, "--controller", str(root)], layout["source"], env, log)
        state["status"] = "running"
        for run in pending:
            record = dict(name=run["name"], status="running", attempt=attempt, started_utc=_utc_now())
            state["runs"].append(record)
            _save_state(group, plan, state)
            _local(group, (group / run["directory"]).parent).mkdir(parents=True, exist_ok=True)
            log = _local(group, _run_log(group, run, attempt))
            log.parent.mkdir(parents=True, exist_ok=True)
            env = _environment(group, run["variant"], run=run["name"] + (f"-resume{attempt:03d}" if attempt else ""), devices=plan["devices"])
            _provision_environment(group, env)
            _command(run["command"], _layout(group, run["variant"])["source"], env, log)
            validate_report(_json(_local(group, group / run["report"])), run, receipt)
            record["status"] = "completed"
            record["finished_utc"] = _utc_now()
            _save_state(group, plan, state)
        state["status"] = "completed"
        _save_state(group, plan, state)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=type(error).__name__, message=str(error))
        if state["runs"] and state["runs"][-1]["status"] == "running":
            state["runs"][-1]["status"] = "failed"
        _save_state(group, plan, state)
        raise
    return analyze(group)


def _load_budget():
    path = Path(__file__).with_name("transfer_budget.py")
    module = types.ModuleType("no_timer_transfer_budget")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("stage", "route", "seed", "B", "operation"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, records in sorted(groups.items()):
        slots = {(item["side"], item["repeat"]): item for item in records}
        _require(len(records) == 4 and set(slots) == {("A", 1), ("A", 2), ("B", 1), ("B", 2)}, "Incomplete or duplicate ABBA block")
        _require(len({row["route_sha256"] for row in records}) == 1, "Actual A/B route fingerprints differ")
        means = {side: statistics.mean(slots[(side, repeat)]["max_rank_mean_us"] for repeat in (1, 2)) for side in "AB"}
        summary = dict(zip(("stage", "route", "seed", "B", "operation"), key),
                       **{side + str(repeat): slots[(side, repeat)]["max_rank_mean_us"] for side in "AB" for repeat in (1, 2)},
                       A=means["A"], B_mean=means["B"], delta_us=means["B"] - means["A"], delta_percent=(means["B"] / means["A"] - 1) * 100,
                       route_sha256=records[0]["route_sha256"], regression_guard=key[0] == "optimization" and key[1] == "hotspot" and key[3] >= 16)
        for metric in ("pooled_mean_us", "pooled_p99_us"):
            for side in "AB":
                summary[side + "_" + metric] = statistics.mean(slots[(side, repeat)][metric] for repeat in (1, 2))
        summaries.append(summary)
    return summaries


def render_summary(receipt, plans, rows):
    summary = summarize(rows)
    lines = ["NO-TIMER / STREAM MATCHED A/B -- SEND THIS ONE FILE ONLY", "Status: completed; requested blocks and both operations validated",
             "Base=" + BASE, "Shared SDK: one clean build in source-normal; both source projects used the same explicit SHMEM_ROOT. No SDK identity/hash admission.",
             "Variants: replay-off=cb59 normal-off (152/96-byte dispatch/combine launch); replay-clean=cb59 clean-source (136/88); stream-clean=cb59-stream clean-source (136/88).",
             "Timer stage isolates source instrumentation removal including timer launch arguments. Optimization stage compares clean replay with clean stream receive.",
             "Shape: EP16 G16 S28/R28 P0; B1/2/4/8/16/24/32 M32 H3584 K8 E384; prequantized FP8 INT32 dispatch + BF16 combine; cpu_sync0; ordinary pages; MTE.",
             "Sampling: 10 warmup, 100 tasks/rank/operation, each sample preceded by Exp + scalar all-reduce, each rank tail50. Correctness repeats3; repeat-combine2.",
             "Main value: first compute each rank tail50 mean, take its run's maximum across16 ranks, then average the two same-variant repetitions. Raw tasks are never pooled before this maximum.",
             "Pooled mean/p99 columns: each run pools only its16 rank tail50 samples; table averages the two run statistics, never pools across repeats.",
             "Delta=B-A; negative favors B. Hotspot B>=16 is a regression guard, not a previously established positive-benefit case.",
             "Only cb59-stream is the measured-benefit candidate. Pair/small and different current P/VF/core splits are excluded. No legacy sampling protocol is mixed in.",
             "Raw task samples, build logs, source proofs and per-rank files remain on target; this single file omits those details.", "", "BUILD IDENTITIES"]
    for name, record in receipt["variants"].items():
        lines.append(f"{name} | source={record['source_sha256']} | wheel={record['wheel_sha256']} | {record['wheel']}")
    lines += ["", "SERIAL RUN ORDER (all builds/Host checks completed first)", "stage | route | seed | order | side/repeat | variant | source report"]
    for plan in plans:
        for run in plan["runs"]:
            lines.append(f"{run['stage']} | {run['route']} | {run['seed']} | {run['order']} | {run['side']}{run['repeat']} | {run['variant']} | {run['report']}")
    resumed = [plan for plan in plans if plan.get("resume_history")]
    if resumed:
        lines += ["", "EXPLICIT RESUME: only complete validated ABBA blocks were retained; incomplete blocks were repeated in full.",
                  "Original and superseded reports/logs remain on target. Paths above identify selected attempts; no samples are pooled across attempts.",
                  "An interruption separates retained blocks from later blocks; each ABBA comparison remains within its own attempt."]
        for plan in resumed:
            lines.append(f"{Path(plan['plan_file']).stem}: {len(plan['resume_history'])} resumes; state snapshots under run-history/{Path(plan['plan_file']).stem}/")
    lines += ["", "COMPLETE TASK RESULTS (us; per-seed ABBA/BAAB, never averaging seeds)",
              "stage route seed B op | A1 A2 B1 B2 | Amean Bmean | delta_us delta_% | pooled_mean_A/B | pooled_p99_A/B | guard"]
    for row in summary:
        values = " ".join(f"{row[key]:.6f}" for key in ("A1", "A2", "B1", "B2"))
        lines.append(f"{row['stage']} {row['route']} {row['seed']} {row['B']} {row['operation']} | {values} | {row['A']:.6f} {row['B_mean']:.6f} | "
                     f"{row['delta_us']:.6f} {row['delta_percent']:.4f} | {row['A_pooled_mean_us']:.6f}/{row['B_pooled_mean_us']:.6f} | "
                     f"{row['A_pooled_p99_us']:.6f}/{row['B_pooled_p99_us']:.6f} | {'regression' if row['regression_guard'] else '-'}")
    lines += ["", "ACTUAL PROFILE ROUTES (same across all four members of each block)", "route | seed | B | SHA256"]
    hashes = {}
    for row in summary:
        key = (row["route"], row["seed"], row["B"])
        _require(key not in hashes or hashes[key] == row["route_sha256"], "Same route/seed/B changed across stages or operations")
        hashes[key] = row["route_sha256"]
    lines += [f"{route} | {seed} | {batch} | {digest}" for (route, seed, batch), digest in sorted(hashes.items())]
    content = "\n".join(lines) + "\n"
    _require(len(content.encode("utf-8")) + content.count("\n") <= LIMIT, "Report exceeds60000 bytes including CRLF; no truncated report was published")
    return content, summary


def analyze(group):
    group = Path(group).absolute()
    receipt = _build_receipt(group)
    paths = sorted((group / "run-plans").glob("*.json"))
    _require(paths, "No run plans exist")
    plans, rows, names = [], [], set()
    for path in paths:
        plan = _json(_local(group, path))
        _validate_plan(plan)
        _require(Path(plan["group"]) == group, "Saved run plan belongs to another group")
        state = _read_state(group, plan)
        _require(state.get("status") == "completed" and [item.get("name") for item in state.get("runs", [])] == [run["name"] for run in plan["runs"]]
                 and all(item.get("status") == "completed" for item in state["runs"]), "Run group is incomplete; no completed report issued")
        _require(_completed_blocks(group, plan, state, receipt) == len(plan["runs"]), "Run group is incomplete")
        selected_runs = [_attempt_run(run, record.get("attempt", 0)) for run, record in zip(plan["runs"], state["runs"])]
        plans.append(dict(plan, runs=selected_runs, resume_history=state.get("resume_history", [])))
        for run in selected_runs:
            _require(run["name"] not in names, "Duplicate run across plans")
            names.add(run["name"])
            rows.extend(validate_report(_json(_local(group, group / run["report"])), run, receipt))
    content, summary = render_summary(receipt, plans, rows)
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    for relative in ("triage", "triage/send_budget.json", "triage/send_budget.lock", "SEND_FIRST.txt"):
        _local(group, group / relative)
    with _load_budget().transaction(group, "summary", [digest]):
        archive = _local(group, group / "triage" / ("no-timer-" + digest[:20]))
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "SEND_FIRST.txt").write_bytes(raw)
        _write(archive / "summary.json", dict(reference=BASE, rows=rows, summary=summary))
        (group / "SEND_FIRST.txt").write_bytes(raw)
    return group / "SEND_FIRST.txt"


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "_preimage":
            verify_preimage(args.source)
            return 0
        if args.action == "_dependencies":
            verify_dependencies()
            return 0
        if args.action == "_install":
            install(args.group, args.variant)
            return 0
        if args.action == "_verify-host":
            verify_host(args.group, args.variant, args.controller)
            return 0
        group = group_path(args)
        if args.action == "build":
            plan = build_plan(group)
            if args.plan:
                print(json.dumps(plan, indent=2))
                return 0
            result = execute_build(plan)
        elif args.action == "run":
            plan = make_run_plan(args)
            if args.plan:
                print(json.dumps(plan, indent=2))
                return 0
            result = execute_run(plan, resume=args.resume)
        else:
            with _run_lock(group):
                result = analyze(group)
        print(result)
        print("Build only: no NPU work." if args.action == "build" else "Return only SEND_FIRST.txt; raw results and logs stay on target.")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"No-timer A/B stopped: {error}. Existing sources, logs and results are preserved.", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
