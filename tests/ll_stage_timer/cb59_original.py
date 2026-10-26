"""Build and rerun the historical cb59 OFF kernel in an isolated project group.

Plans and analysis use only the standard library. Actual target runs call the
historical ST with its own installed wheel, routing, correctness and profiling.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "cb59bff99ecd39d27ef57788c5e6fd0009e4e636"
LIMIT = 60000
DEFAULT_BS = [1, 2, 4, 8, 16, 24, 32]
VISIBILITY = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES")
HISTORICAL_US = {1: 17.515, 2: 17.5547, 4: 18.0762, 8: 18.110666666666667,
                 16: 19.160966666666667, 24: 19.9418, 32: 21.901666666666667}


def parser():
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = value.add_subparsers(dest="action", required=True)
    for name in ("build", "run", "analyze"):
        command = actions.add_parser(name, allow_abbrev=False)
        command.add_argument("--run-name", required=True)
        command.add_argument("--output-root", default="diagnostic-runs")
        if name in ("build", "run"):
            command.add_argument("--plan", action="store_true", help="Print a read-only plan; no Torch import or execution")
        if name == "run":
            command.add_argument("--devices", required=True, help="Exactly 16 distinct device IDs, comma separated")
            command.add_argument("--master-addr", required=True)
            command.add_argument("--master-port", type=int, required=True)
            command.add_argument("--shmem-port", type=int, required=True)
            command.add_argument("--bs", type=int, nargs="+", default=DEFAULT_BS)
            command.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
            command.add_argument("--rounds", type=int, default=2)
    return value


def group_path(args, root=ROOT):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
        raise ValueError("--run-name must be an ASCII directory name of 1..96 characters")
    parent = Path(args.output_root)
    group = (parent if parent.is_absolute() else root / parent) / args.run_name
    group = group.resolve()
    if not group.is_relative_to(root) or group == root:
        raise ValueError("Output group must remain inside this project")
    if group.relative_to(root).parts[0] in {".git", "build", "csrc", "deep_ep", "scripts", "tests", "cmake", "third-party", "benchmarks", "experiments"}:
        raise ValueError("Use a separate project output directory, outside cleaned build/source/dependency paths")
    return group


def _load_helper(name):
    path = Path(__file__).with_name(name + ".py")
    module = types.ModuleType("original_helper_" + name)
    module.__file__ = str(path)
    # Avoid import cache writes during --plan.
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _load_builder():
    return _load_helper("cb59_original_build")


def _relative_path(group, value, *, executable=False):
    if not isinstance(value, str) or not value:
        raise ValueError("Build receipt paths must be nonempty relative strings")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Build receipt paths must remain relative to their group")
    path = group / relative
    # A venv Python executable may itself link to system Python. Its containing
    # directory must still be inside the group; data/source paths may not escape.
    checked = path.parent.resolve() if executable else path.resolve()
    if not checked.is_relative_to(group.resolve()):
        raise ValueError("Build receipt path escapes its group")
    return path


def _receipt(group):
    receipt = json.loads((group / "build-receipt.json").read_text(encoding="utf-8"))
    if receipt.get("reference") != REFERENCE:
        raise ValueError("This runner requires the historical cb59 build receipt")
    source = _relative_path(group, receipt.get("source"))
    python = _relative_path(group, receipt.get("python"), executable=True)
    wheel = _relative_path(group, receipt.get("wheel"))
    if not source.is_dir() or not python.is_file() or not wheel.is_file():
        raise ValueError("Historical source, isolated Python or built wheel is missing")
    return receipt, source, python


def make_run_plan(args, *, root=ROOT):
    group = group_path(args, root)
    receipt, source, python = _receipt(group)
    raw_devices = args.devices.split(",")
    if len(raw_devices) != 16 or any(not re.fullmatch(r"[0-9]+", item.strip()) for item in raw_devices):
        raise ValueError("--devices must contain exactly 16 numeric device IDs")
    devices = [int(item) for item in raw_devices]
    if len(set(devices)) != 16:
        raise ValueError("--devices must contain 16 distinct device IDs")
    if not args.master_addr or any(ch.isspace() for ch in args.master_addr):
        raise ValueError("An explicit communication address is required")
    if not 1 <= args.master_port <= 65535 or not 1 <= args.shmem_port <= 65535:
        raise ValueError("Communication ports must be in 1..65535")
    if args.master_port == args.shmem_port:
        raise ValueError("MASTER_PORT and SHMEM_PORT must be different")
    if not args.bs or len(set(args.bs)) != len(args.bs) or any(not 1 <= b <= 32 for b in args.bs):
        raise ValueError("B values must be distinct and in 1..32")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) > 4 or any(s < 0 for s in args.seeds):
        raise ValueError("Choose one to four distinct nonnegative profile seeds")
    if not 1 <= args.rounds <= 4:
        raise ValueError("--rounds must be in 1..4")
    if len(args.bs) * len(args.seeds) * args.rounds > 128:
        raise ValueError("The one-file report supports at most 128 measured B/seed/round rows")
    script = source / "tests/elastic/test_ep_low_latency.py"
    if not script.is_file():
        raise ValueError("The historical ST is missing from the isolated source")
    runs = []
    for round_index in range(args.rounds):
        seeds = args.seeds if round_index % 2 == 0 else list(reversed(args.seeds))
        for seed in seeds:
            relative = f"runs/round_{round_index + 1:02d}_seed_{seed}"
            directory = group / relative
            command = [str(python), str(script), "--num-processes", "16", "--num-nodes", "1", "--node-rank", "0",
                       "--master-addr", args.master_addr, "--master-port", str(args.master_port),
                       "--shmem-port", str(args.shmem_port), "--num-tokens", *map(str, args.bs),
                       "--num-max-tokens-per-rank", "32", "--hidden", "3584", "--num-topk", "8",
                       "--num-experts", "384", "--dtype", "fp8", "--fp8-scale-dtype", "int32",
                       "--do-cpu-sync", "0", "--route", "normal", "--seed", str(seed), "--dispatch-only",
                       "--warmup", "10", "--num-iters", "100", "--perf-tail", "50",
                       "--delay-rows", "118260", "--delay-cols", "30720",
                       "--delay-exp-before", "1", "--delay-exp-after", "0",
                       "--output-dir", str(directory), "--output", str(directory / "full-report.json")]
            runs.append(dict(name=f"round_{round_index + 1:02d}_seed_{seed}", round=round_index + 1, seed=seed,
                             directory=relative, report=f"{relative}/full-report.json", command=command))
    return dict(format="cb59-original-runs-v1", group=str(group), reference=REFERENCE,
                source=receipt["source"], python=receipt["python"], devices=devices, bs=list(args.bs),
                seeds=list(args.seeds), rounds=args.rounds, runs=runs,
                conditions=dict(world_size=16, M=32, H=3584, K=8, E=384, dtype="fp8", fp8_scale_dtype="int32",
                                cpu_sync=0, timer="OFF", warmup=10, samples=100, tail=50,
                                delay_rows=118260, delay_cols=30720),
                note="Original Session/backend/page defaults are preserved. No --backend/page/core override is added.")


def _run_environment(group, source, run, devices):
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("DEEPEP_") or key in {"PYTHONPATH", "PYTHONHOME", "SHMEM_SOURCE_PATH", "SHMEM_ROOT"}:
            environment.pop(key)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        environment.pop(key, None)
    for key in VISIBILITY:
        environment[key] = ",".join(map(str, devices))
    # The old SDK's runtime path must precede any unrelated inherited SDK.
    sdk = source / "third-party/shmem/install/shmem"
    environment["LD_LIBRARY_PATH"] = str(sdk / "lib") + (os.pathsep + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    environment["SHMEM_ROOT"] = str(sdk)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    directory = group / run["directory"]
    paths = {"TMPDIR": directory / "tmp", "TMP": directory / "tmp", "TEMP": directory / "tmp",
             "XDG_CACHE_HOME": directory / "cache", "PIP_CACHE_DIR": directory / "cache/pip",
             "TORCH_EXTENSIONS_DIR": directory / "cache/torch-extensions",
             "ASCEND_CACHE_PATH": directory / "cache/ascend", "ASCEND_WORK_PATH": directory / "logs/ascend",
             "ASCEND_PROCESS_LOG_PATH": directory / "logs/ascend", "SHMEM_LOG_PATH": directory / "logs/shmem"}
    environment.update({name: str(path) for name, path in paths.items()})
    return environment, set(paths.values())


def _write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def execute_run(plan):
    group = Path(plan["group"])
    _, source, _ = _receipt(group)
    if (group / "runs").exists() or (group / "run-plan.json").exists() or (group / "SEND_FIRST.txt").exists():
        raise ValueError("This build group already has runs; results are never overwritten or retried automatically")
    (group / "runs").mkdir()
    _write_json(group / "run-plan.json", plan, exclusive=True)
    state = dict(format="cb59-original-run-status-v1", status="running", runs=[])
    _write_json(group / "run-status.json", state, exclusive=True)
    for run in plan["runs"]:
        directory = group / run["directory"]
        directory.mkdir(exist_ok=False)
        environment, directories = _run_environment(group, source, run, plan["devices"])
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)
        record = dict(name=run["name"], status="running")
        state["runs"].append(record)
        _write_json(group / "run-status.json", state)
        try:
            with (directory / "logs/st.log").open("x", encoding="utf-8") as log:
                subprocess.run(run["command"], cwd=source, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            validate_report(json.loads((group / run["report"]).read_text(encoding="utf-8")), run, plan["bs"])
        except Exception as error:
            record.update(status="failed", error=type(error).__name__, message=str(error))
            state["status"] = "failed"
            _write_json(group / "run-status.json", state)
            raise
        record["status"] = "completed"
        _write_json(group / "run-status.json", state)
    state["status"] = "completed"
    _write_json(group / "run-status.json", state)
    return analyze(group)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, run, batches):
    _require(report.get("status") == "completed" and report.get("world_size") == 16, "Incomplete original report or wrong actual EP")
    _require(report.get("case_count") == len(batches) and report.get("result_count") == len(batches), "Original case coverage is incomplete")
    config = report.get("config", {})
    expected = dict(num_processes=16, num_nodes=1, node_rank=0, num_max_tokens_per_rank=32,
                    hidden=3584, num_topk=8, num_experts=384, dtype="fp8", fp8_scale_dtype="int32",
                    do_cpu_sync=0, warmup=10, num_iters=100, perf_tail=50, seed=run["seed"],
                    delay_rows=118260, delay_cols=30720, delay_exp_before=1, delay_exp_after=0,
                    group_backend="hccl", route="normal", dispatch_only=True)
    for key, value in expected.items():
        _require(config.get(key) == value, f"Original run {run['name']} has mismatched {key}")
    _require(not config.get("ll_stage_timer") and not config.get("skip_check") and not config.get("skip_perf_test"),
             "Original reference must be OFF with correctness and performance enabled")
    configured_bs = config.get("num_tokens")
    _require((configured_bs if isinstance(configured_bs, list) else [configured_bs]) == list(batches), "Original B matrix differs from the plan")
    caps = config.get("ll_stage_timer_capabilities", [])
    _require(isinstance(caps, list) and len(caps) == 16 and all(item == caps[0] for item in caps), "Missing or inconsistent original native capabilities")
    capability = caps[0]
    active = capability.get("active_aiv")
    _require(type(active) is int and 16 <= active <= 72 and active % 2 == 0, "Invalid actual AIV count")
    _require(capability.get("stage_timer") is False, "Original native wheel is not timer OFF")
    _require(capability.get("core_protocol") == 2, "Original native core protocol is not cb59 protocol2")
    _require(type(capability.get("logical_group")) is int and capability["logical_group"] > 0, "Missing actual logical group")
    results = report.get("results", [])
    _require(len(results) == len(batches), "Missing original cases")
    rows = []
    for case_id, (batch, result) in enumerate(zip(batches, results)):
        shape = dict(B=batch, M=32, H=3584, num_topk=8, num_experts=384, dtype="fp8", do_cpu_sync=False, fp8_scale_dtype="int32")
        _require(result.get("case_id") == case_id and result.get("route") == "normal" and result.get("backend") == "ll"
                 and result.get("correctness") == "passed" and result.get("tested_operations") == ["dispatch"], "Original case identity/correctness differs")
        _require(result.get("config") == shape, "Original case shape differs")
        perf = result.get("dispatch", {})
        _require(perf.get("kernel") == "dispatch_server_dedup_kernel" and perf.get("measurement") == "kernel_duration_us"
                 and perf.get("perf_tail") == 50, "Original measurement must be the complete dispatch kernel task, tail50")
        ranks = perf.get("ranks", [])
        _require(len(ranks) == 16 and {r.get("rank") for r in ranks} == set(range(16)), "Original per-rank coverage is incomplete")
        means = []
        for rank in ranks:
            raw = rank.get("kernel_us", [])
            _require(len(raw) == 100 and rank.get("samples") == 50 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in raw),
                     "Original profile must contain 100 finite samples per rank and select tail50")
            mean = statistics.mean(raw[-50:])
            _require(type(rank.get("mean_us")) in (float, int) and math.isclose(mean, rank["mean_us"], rel_tol=1e-9, abs_tol=1e-9),
                     "Original per-rank tail mean disagrees with raw samples")
            means.append(mean)
        measured = max(means)
        _require(type(perf.get("max_rank_mean_kernel_us")) in (float, int) and
                 math.isclose(measured, perf["max_rank_mean_kernel_us"], rel_tol=1e-9, abs_tol=1e-9),
                 "Original maximum rank mean disagrees with tail50 samples")
        rows.append(dict(round=run["round"], seed=run["seed"], B=batch, us=measured,
                         active_aiv=active, logical_group=capability["logical_group"],
                         core_group_matches_small=active == 56 and capability["logical_group"] == 16,
                         source_report=run["report"]))
    return rows, capability


def original_route_hash(source, seed, batch):
    """Execute only the historical pure-CPU routing function, not its module."""
    path = Path(source) / "tests/elastic/test_ep_low_latency.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_routes"]
    _require(len(matches) == 1 and not matches[0].decorator_list, "Historical make_routes is missing or ambiguous")
    function = ast.Module(body=matches, type_ignores=[])
    namespace = {"random": random}
    exec(compile(ast.fix_missing_locations(function), str(path), "exec"), namespace)
    routes = namespace["make_routes"](16, batch, 8, 384, seed)
    encoded = json.dumps(routes, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def analyze(group):
    group = Path(group).resolve()
    receipt, source, _ = _receipt(group)
    plan = json.loads((group / "run-plan.json").read_text(encoding="utf-8"))
    _require(plan.get("format") == "cb59-original-runs-v1" and plan.get("reference") == REFERENCE, "Original run plan is invalid")
    state = json.loads((group / "run-status.json").read_text(encoding="utf-8"))
    _require(state.get("status") == "completed" and [r.get("name") for r in state.get("runs", [])] == [r["name"] for r in plan["runs"]]
             and all(r.get("status") == "completed" for r in state["runs"]), "Original run group is incomplete; no completed summary generated")
    rows, runs = [], []
    for run in plan["runs"]:
        report_path = _relative_path(group, run["report"])
        result, capability = validate_report(json.loads(report_path.read_text(encoding="utf-8")), run, plan["bs"])
        rows.extend(result)
        runs.append((run, capability))
    hashes = {(seed, batch): original_route_hash(source, seed, batch) for seed in plan["seeds"] for batch in plan["bs"]}
    lines = ["CB59 ORIGINAL OFF REPRODUCTION -- SEND THIS ONE FILE ONLY", f"Group: {group.name}",
             f"Reference: {REFERENCE}", "Status: completed; all requested rounds/seeds/B values validated",
             "Measurement: full original dispatch profiler task; 100 samples/rank, each rank's tail50 mean, maximum across16 ranks.",
             "Shape: EP16 M32 H3584 K8 E384 FP8 INT32 scales cpu_sync0; correctness passed; timer OFF.",
             "Actual native AIV/G are reported below, never replaced with assumed56/G16.",
             "Core/group match does not establish matched SDK, pages, compiler, environment or sampling date.",
             "Original Session/backend/page defaults preserved; no current MTE/page override was added.",
             "This summary omits raw samples/profiles/logs, which remain on the target.",
             "No ON-minus-OFF or cross-rank-clock inference is made.", "",
             "BUILD SOURCE / ALLOWED ADAPTATIONS (metadata only; no SDK hash/identity admission)",
             json.dumps(receipt, sort_keys=True, ensure_ascii=True), "", "RUN SOURCES",
             "run | seed | actual AIV | S/R | G | protocol | core/group match to fixed56/G16 small ONLY | report"]
    for run, cap in runs:
        active = cap["active_aiv"]
        match = "yes: core/group only" if active == 56 and cap["logical_group"] == 16 else "NO: actual core/group differs"
        lines.append(f"{run['name']} | {run['seed']} | {active} | {active//2}/{active//2} | {cap['logical_group']} | {cap['core_protocol']} | {match} | {run['report']}")
    lines += ["", "FULL OFF RESULTS", "round | profile_seed | B | max rank tail50 mean us"]
    for row in rows:
        lines.append(f"{row['round']} | {row['seed']} | {row['B']} | {row['us']:.6f}")
    lines += ["", "EXACT PROFILE ROUTE HASHES (original make_routes, seed is the actual ST --seed)",
              "seed | B | SHA256(json.dumps(routes,separators=(',',':'),ensure_ascii=True))"]
    for (seed, batch), digest in sorted(hashes.items()):
        lines.append(f"{seed} | {batch} | {digest}")
    lines += ["", "HISTORICAL USER REFERENCE -- separate earlier run, NOT a same-run/control result", "B | historical us"]
    lines += [f"{batch} | {HISTORICAL_US[batch]:.6f}" for batch in plan["bs"] if batch in HISTORICAL_US]
    lines += ["", "Compare only matching seed/route hashes and recorded conditions. The earlier reference is not proof of a reproduced regression.",
              "One UTF-8 report; raw samples and SDK outputs are not requested for transfer."]
    content = "\n".join(lines) + "\n"
    raw = content.encode("utf-8")
    _require(len(raw) + content.count("\n") <= LIMIT, "Original summary exceeds60000 bytes including CRLF conversion; no truncated completed report was written")
    destination = group / "SEND_FIRST.txt"
    digest = hashlib.sha256(raw).hexdigest()
    with _load_helper("transfer_budget").transaction(group, "summary", [digest]):
        archive = group / "triage" / ("original_" + digest[:20])
        archive.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != raw:
            previous = destination.read_bytes()
            prior = group / "triage" / ("original_" + hashlib.sha256(previous).hexdigest()[:20])
            prior.mkdir(parents=True, exist_ok=True)
            (prior / "SEND_FIRST.txt").write_bytes(previous)
        (archive / "SEND_FIRST.txt").write_bytes(raw)
        _write_json(archive / "summary.json", dict(reference=REFERENCE, runs=plan["runs"], rows=rows,
            profile_routes=[dict(seed=seed, B=batch, sha256=value) for (seed, batch), value in sorted(hashes.items())]))
        destination.write_bytes(raw)
    return destination


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        group = group_path(args)
        if args.action == "build":
            builder = _load_builder()
            plan = builder.make_plan(group, root=ROOT, python=sys.executable)
            if args.plan:
                print(json.dumps(plan, indent=2))
                return 0
            result = builder.execute(plan)
        elif args.action == "run":
            plan = make_run_plan(args)
            if args.plan:
                print(json.dumps(plan, indent=2))
                return 0
            result = execute_run(plan)
        else:
            result = analyze(group)
        print(result)
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"cb59 original: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
