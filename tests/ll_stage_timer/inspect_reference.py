"""Inspect an existing reference checkout and profiler results without running it.

Only standard-library file reads and explicitly read-only Git commands inspect
the reference. No framework, SDK, compiler, loader, device query or inspected
Python source is executed. Evidence is written only to a new current-project
diagnostic group; return its single bounded SEND_FIRST.txt.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import statistics
import subprocess
import types

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 60000
MAX_TEXT = 16 * 1024 * 1024
REFS = {"cbd04eb": "cbd04eb2c1cbc120c035a59ddf7cd55feb79e200",
        "cb59bff": "cb59bff99ecd39d27ef57788c5e6fd0009e4e636"}
ST = "tests/elastic/test_ep_low_latency.py"
KEY_FILES = (
    "csrc/kernels/dispatch/low_latency.cpp", "csrc/kernels/dispatch/low_latency/launch.hpp",
    "csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h",
    "csrc/kernels/dispatch/low_latency/common/ll_stage_timer_device.h",
    "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h",
    "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h",
    "csrc/runtime/session.cpp", "csrc/ops/low_latency.cpp", "deep_ep/buffers/elastic.py",
    "deep_ep/elastic.py", ST, "build_support.py", "CMakeLists.txt", "cmake/Dependencies.cmake",
    "cmake/Device.cmake", "cmake/device/CMakeLists.txt", "dependencies.lock.json",
)
SOURCE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inl", ".cce", ".cu"}
FUNCTIONS = ("make_parser", "measure", "resolve_delay_counts", "summarize_kernel", "sample_stats", "kernel_durations", "make_routes")
CONFIG_KEYS = ("warmup", "num_iters", "perf_tail", "dispatch_only", "do_cpu_sync", "cpu_sync", "dtype",
               "fp8_scale_dtype", "num_tokens", "num_max_tokens_per_rank", "hidden", "num_topk", "num_experts",
               "delay_rows", "delay_cols", "delay_exp_before", "delay_exp_after", "seed", "profile_seed",
               "num_processes", "num_nodes", "node_rank", "group_backend", "without_buffer_group",
               "do_handle_copy", "ll_stage_timer", "skip_check", "skip_perf_test", "use_1g_huge_pages",
               "devices", "backend", "fresh_buffer_per_case", "ll_stage_timer_capabilities")
INFO_KEYS = ("source_version", "source_sha256", "source_revision", "source_dirty", "arch", "build_type", "python",
             "python_executable", "machine", "compiler", "compiler_path", "compiler_sha256", "compiler_version_sha256",
             "device_compiler_sha256", "device_compiler_path", "cmake_path", "cmake_version", "cann", "ll_stage_timer", "ll_timer_profile",
             "ll_dispatch_impl", "ll_send_cores", "ll_recv_cores", "low_latency_core_protocol", "ASCEND_HOME_PATH", "shmem")
SECRET = re.compile(r"(?i)(password|passwd|token|secret|credential|private.key|access.key)")


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path, help="Existing result directory or JSON file; relative to reference root")
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", type=Path, default=Path("diagnostic-runs"))
    return p


def _link(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def make_plan(args, root=ROOT):
    root = Path(root).resolve()
    supplied_reference = args.reference_root.expanduser()
    reference = (supplied_reference if supplied_reference.is_absolute() else root / supplied_reference).resolve()
    results = (args.results if args.results.is_absolute() else reference / args.results).resolve()
    if not reference.is_dir():
        raise ValueError("--reference-root must be an existing source directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
        raise ValueError("--run-name must be a fresh ASCII directory name")
    parent = args.output_root if args.output_root.is_absolute() else root / args.output_root
    candidate = parent / args.run_name
    for ancestor in (*reversed(candidate.absolute().parents), candidate.absolute()):
        if _link(ancestor):
            raise ValueError("Output may not traverse a symlink or junction")
    group = candidate.resolve()
    if not group.is_relative_to(root) or group == root:
        raise ValueError("Output must remain inside the current script's project")
    if group.relative_to(root).parts[0] in {".git", "csrc", "deep_ep", "tests", "scripts", "cmake", "third-party", "build", "benchmarks", "experiments"}:
        raise ValueError("Output must be in a diagnostic directory, not a source/dependency directory")
    for supplied in (reference, results):
        if group.is_relative_to(supplied) or supplied.is_relative_to(group):
            raise ValueError("Output must not overlap the inspected source or results; run this helper from a separate current checkout")
    if group.exists():
        raise ValueError("Output group already exists; choose a fresh --run-name")
    return dict(root=str(root), reference=str(reference), results=str(results), group=str(group))


def git_read(repo, *arguments):
    """No optional index writes, fsmonitor hooks, external diffs or textconv."""
    if not arguments or arguments[0] not in {"rev-parse", "symbolic-ref", "status", "submodule", "diff", "show"}:
        raise ValueError("Only the inspection's read-only Git operations are allowed")
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *arguments]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return None, type(error).__name__
    if result.returncode:
        return None, f"Git returned {result.returncode}"
    if len(result.stdout) > MAX_TEXT:
        return None, "Git output exceeded the bounded inspection limit"
    return result.stdout, None


def file_hash(path):
    try:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return dict(path=str(path), size=size, sha256=digest.hexdigest(), status="read")
    except OSError as error:
        return dict(path=str(path), status="unavailable", reason=type(error).__name__)


def _text(path):
    try:
        if not path.is_file():
            return None, "missing"
        if path.stat().st_size > MAX_TEXT:
            return None, "over read limit"
        return path.read_text(encoding="utf-8-sig", errors="replace"), None
    except OSError as error:
        return None, type(error).__name__


def _json(path):
    text, error = _text(path)
    if error:
        return None, error
    try:
        return json.loads(text), None
    except ValueError:
        return None, "invalid JSON"


def _walk(base, names=None, limit=8000):
    """Only the explicitly selected subtree, never a symlinked directory."""
    if _link(base):
        return [], [f"symlink/junction subtree skipped: {base}"]
    if not base.is_dir():
        return [], []
    paths, skipped = [], []
    seen = 0
    for parent, directories, files in os.walk(base, followlinks=False):
        parent = Path(parent)
        kept = []
        for name in sorted(directories):
            path = parent / name
            if name in {".git", "__pycache__", ".pytest_cache"}:
                continue
            if _link(path):
                skipped.append(str(path))
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            seen += 1
            if seen > limit:
                return paths, skipped + [f"file inventory limit {limit} reached under {base}"]
            path = parent / name
            if _link(path):
                skipped.append(str(path))
                continue
            if names is None or name in names or any(name.startswith(prefix) for prefix in names if prefix.endswith("_")):
                paths.append(path)
    return paths, skipped


def _source_text(reference, relative):
    path = reference / relative
    current = reference
    for part in Path(relative).parts:
        current /= part
        if _link(current):
            return None, "symlink/junction source path skipped"
    return _text(path)


def ast_protocol(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return dict(error="source cannot be parsed as Python", functions={}, parser_defaults={})
    functions, defaults = {}, {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in FUNCTIONS:
            continue
        representation = ast.dump(node, annotate_fields=True, include_attributes=False)
        functions[node.name] = hashlib.sha256(representation.encode()).hexdigest()
        if node.name == "make_parser":
            for call in ast.walk(node):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "add_argument" or not call.args:
                    continue
                if not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
                    continue
                name = call.args[0].value.lstrip("-").replace("-", "_")
                if name not in CONFIG_KEYS:
                    continue
                for option in call.keywords:
                    if option.arg == "default":
                        try:
                            defaults[name] = ast.literal_eval(option.value)
                        except (ValueError, TypeError):
                            defaults[name] = "nonliteral default: not evaluated"
    return dict(functions=functions, parser_defaults=defaults)


def collect_source(reference, controller):
    owner, owner_error = git_read(reference, "rev-parse", "--show-toplevel")
    owner_path = Path(owner.decode("utf-8", errors="replace").strip()).resolve() if owner else None
    is_root = owner_path == reference.resolve()
    metadata = dict(is_repository_root=is_root, repository_owner=str(owner_path) if owner_path else None)
    for name, command in (("head", ("rev-parse", "HEAD")), ("branch", ("symbolic-ref", "--short", "-q", "HEAD")),
                          ("tracked_status", ("status", "--porcelain=v1", "--untracked-files=no", "--", "csrc", "deep_ep", "tests", "scripts", "cmake", "build_support.py", "CMakeLists.txt", "dependencies.lock.json")),
                          ("submodules", ("submodule", "status", "--recursive"))):
        value, error = git_read(reference, *command) if is_root else (None, owner_error or "given reference-root is not its repository root; parent repository metadata was not used")
        metadata[name] = value.decode("utf-8", errors="replace").strip() if value is not None else None
        if error:
            metadata[name + "_unavailable"] = error
    sdk_source = reference / "third-party/shmem"
    sdk_owner, _ = git_read(sdk_source, "rev-parse", "--show-toplevel") if (sdk_source / ".git").exists() else (None, None)
    sdk_is_root = sdk_owner is not None and Path(sdk_owner.decode("utf-8", errors="replace").strip()).resolve() == sdk_source.resolve()
    sdk_revision, sdk_error = git_read(sdk_source, "rev-parse", "HEAD") if sdk_is_root else (None, "SDK source is not its own initialized submodule/checkout")
    metadata["sdk_source_revision_informational"] = sdk_revision.decode().strip() if sdk_revision else None
    if sdk_error:
        metadata["sdk_source_revision_unavailable"] = sdk_error
    raw_diff, error = git_read(reference, "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--",
                               "csrc", "deep_ep", ST, "build_support.py", "CMakeLists.txt", "cmake", "dependencies.lock.json") if is_root else (None, "source directory is not its repository root")
    diff = raw_diff.decode("utf-8", errors="replace") if raw_diff else ""
    if error:
        metadata["tracked_diff_unavailable"] = error
    files, skipped = [], []
    for directory in ("csrc", "deep_ep"):
        if not (reference / directory).is_dir():
            skipped.append(f"missing core source directory: {reference / directory}")
        paths, exclusions = _walk(reference / directory)
        skipped.extend(exclusions)
        for path in paths:
            if path.suffix.lower() in SOURCE_SUFFIXES:
                info = file_hash(path)
                info["path"] = path.relative_to(reference).as_posix()
                files.append(info)
    files.sort(key=lambda item: item["path"])
    aggregate = hashlib.sha256()
    for info in files:
        aggregate.update((info["path"] + "\0" + str(info.get("size", "missing")) + "\0" + info.get("sha256", "missing") + "\n").encode())
    source = dict(git=metadata, core_files=files,
        core_byte_aggregate=dict(sha256=aggregate.hexdigest(), files=len(files), complete=not skipped and all(x["status"] == "read" for x in files),
            algorithm="SHA256(sorted relative-path NUL byte-size NUL file-SHA256 LF); csrc/deep_ep source extensions only"),
        skipped=skipped, key_files=[], reference_protocols={})
    differences = {}
    for relative in KEY_FILES:
        path = reference / relative
        local, local_error = _source_text(reference, relative)
        raw_local = path.read_bytes() if local is not None else None
        entry = dict(path=relative, current_sha256=hashlib.sha256(raw_local).hexdigest() if raw_local is not None else None,
                     status="read" if raw_local is not None else local_error, comparisons={})
        for label, revision in REFS.items():
            blob, error = git_read(controller, "show", f"{revision}:{relative}")
            if blob is None:
                entry["comparisons"][label] = dict(state="reference-missing", reason=error)
                continue
            same = raw_local == blob
            entry["comparisons"][label] = dict(state="missing" if raw_local is None else ("equal" if same else "changed"),
                sha256=hashlib.sha256(blob).hexdigest(), git_blob_sha1=hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest())
            if relative == ST:
                source["reference_protocols"][label] = ast_protocol(blob.decode("utf-8", errors="replace"))
            if raw_local is not None and not same:
                differences[f"{label}:{relative}"] = "".join(difflib.unified_diff(blob.decode("utf-8", errors="replace").splitlines(keepends=True),
                    local.splitlines(keepends=True), fromfile=f"{revision}/{relative}", tofile=f"working/{relative}"))
        source["key_files"].append(entry)
    current_st, error = _source_text(reference, ST)
    source["working_protocol"] = ast_protocol(current_st) if current_st is not None else dict(error=error, functions={}, parser_defaults={})
    measure = source["working_protocol"]["functions"].get("measure")
    source["known_measure_protocol"] = next((label for label, protocol in source["reference_protocols"].items()
        if measure is not None and protocol["functions"].get("measure") == measure), "unknown; inspected source is not executed")
    source["tracked_diff"] = diff
    source["reference_diffs"] = differences
    return source


def _numeric_samples(values):
    return isinstance(values, list) and bool(values) and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values)


def window_statistics(ranks, window):
    required = None if window == "all" else int(window)
    means = []
    for rank in ranks:
        samples = rank.get("kernel_us")
        if not _numeric_samples(samples):
            return dict(status="unavailable", reason="missing/nonfinite/nonpositive raw samples")
        if required is not None and len(samples) < required:
            return dict(status="unavailable", reason=f"rank{rank.get('rank')} has{len(samples)} samples, fewer than{required}")
        selected = samples if required is None else samples[-required:]
        means.append((rank.get("rank"), statistics.mean(selected)))
    if not means:
        return dict(status="unavailable", reason="no raw ranks")
    slow = max(means, key=lambda row: row[1])
    return dict(status="available", ranks=len(means), min_rank_mean_us=min(x[1] for x in means),
        median_rank_mean_us=statistics.median(x[1] for x in means), max_rank_mean_us=slow[1], slowest_rank=slow[0],
        rank0_mean_us=next((value for rank, value in means if rank == 0), None))


def analyze_report(report, path, sample_files=None):
    config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
    evidence = dict(path=str(path), status=report.get("status"), world_size=report.get("world_size"),
        config={key: config[key] for key in CONFIG_KEYS if key in config}, package=report.get("package"),
        measurement=report.get("measurement"), cases=[], issues=[])
    results = report.get("results")
    if not isinstance(results, list):
        results = [report] if isinstance(report.get("config"), dict) and any(name in report for name in ("dispatch", "combine")) else []
    for result in results:
        if not isinstance(result, dict):
            continue
        for operation in ("dispatch", "combine"):
            perf = result.get(operation)
            if not isinstance(perf, dict):
                continue
            shape = result.get("config", {}) if isinstance(result.get("config"), dict) else {}
            raw_ranks = perf.get("ranks", [])
            ranks = []
            for item in raw_ranks if isinstance(raw_ranks, list) else []:
                if isinstance(item, dict):
                    ranks.append(dict(rank=item.get("rank"), kernel_us=item.get("kernel_us"),
                                      reported_mean_us=item.get("mean_us"), reported_selected_samples=item.get("samples")))
            if not ranks or any(not _numeric_samples(item.get("kernel_us")) for item in ranks):
                recovered = []
                case_name = f"case_{int(result.get('case_id', 0)):05d}" if type(result.get("case_id", 0)) is int else None
                for sample_path, data in (sample_files or []):
                    parts = sample_path.parts
                    if case_name in parts and operation in parts and result.get("route", "normal") in parts:
                        rank_parts = [part for part in parts if re.fullmatch(r"rank_[0-9]+", part)]
                        if rank_parts and _numeric_samples(data.get("kernel_us")):
                            recovered.append(dict(rank=int(rank_parts[-1].split("_")[-1]), kernel_us=data["kernel_us"], source=str(sample_path)))
                if recovered:
                    ranks = sorted(recovered, key=lambda item: item["rank"])
            tails = {str(window): window_statistics(ranks, window) for window in ("all", 10, 30, 50, 100)}
            recorded_tail = perf.get("perf_tail", config.get("perf_tail"))
            recorded = window_statistics(ranks, "all" if recorded_tail == 0 else recorded_tail) if type(recorded_tail) is int and recorded_tail >= 0 else dict(status="unavailable", reason="recorded tail unknown")
            blocks = []
            for item in ranks:
                samples = item.get("kernel_us")
                if not _numeric_samples(samples):
                    continue
                blocks.append(dict(rank=item["rank"], blocks=[dict(first=i + 1, last=min(i + 10, len(samples)),
                    mean_us=statistics.mean(samples[i:i + 10])) for i in range(0, len(samples), 10)]))
            case = dict(case_id=result.get("case_id"), operation=operation, route=result.get("route"), shape=shape,
                correctness=result.get("correctness"), backend=result.get("backend"), kernel=perf.get("kernel"),
                measurement=perf.get("measurement"), recorded_tail=recorded_tail, recorded_recomputed=recorded,
                reported_statistics={key: perf[key] for key in ("max_rank_mean_kernel_us", "median_rank_mean_kernel_us", "min_rank_mean_kernel_us", "slowest_rank") if key in perf},
                actual_sample_counts=[dict(rank=r["rank"], samples=len(r["kernel_us"]) if isinstance(r.get("kernel_us"), list) else None) for r in ranks],
                windows=tails, block_means=blocks)
            if type(config.get("num_iters")) is int:
                case["raw_count_matches_recorded_num_iters"] = all(isinstance(r.get("kernel_us"), list) and len(r["kernel_us"]) == config["num_iters"] for r in ranks) if ranks else None
            if len({r["rank"] for r in ranks}) != len(ranks):
                case["issue"] = "duplicate raw rank identities; recomputed aggregate is not trustworthy"
            if recorded.get("status") == "available" and type(perf.get("max_rank_mean_kernel_us")) in (int, float):
                case["reported_minus_recomputed_us"] = perf["max_rank_mean_kernel_us"] - recorded["max_rank_mean_us"]
            evidence["cases"].append(case)
    if not evidence["cases"]:
        evidence["issues"].append("no dispatch/combine result rows recognized")
    return evidence


def collect_results(results):
    issues, reports, auxiliaries, loaded, sample_files = [], [], [], [], []
    if results.is_file():
        paths = [results]
        base = results.parent
    else:
        base = results
        paths, skipped = _walk(base, {"summary.json", "full-report.json", "result.json", "samples.json", "cases.json", "plan.json", "config.json", "run-plan.json", "loaded_rank_"})
        issues.extend(skipped)
    if not paths:
        issues.append("no matching result evidence found")
    def priority(path):
        name = path.name
        if name in {"summary.json", "full-report.json"}:
            return (0, len(path.parts), name != "full-report.json", str(path))
        if name in {"cases.json", "plan.json", "config.json", "run-plan.json"} or name.startswith("loaded_rank_"):
            return (1, len(path.parts), False, str(path))
        return (3 if name == "samples.json" else 2, len(path.parts), False, str(path))
    paths.sort(key=priority)
    parsed = []
    for path in paths[:512]:
        if path.name.startswith("loaded_rank_") and not re.fullmatch(r"loaded_rank_[0-9]+\.json", path.name):
            continue
        data, error = _json(path)
        if error:
            issues.append(f"{path}: {error}")
            continue
        if not isinstance(data, dict):
            continue
        if path.name == "samples.json":
            sample_files.append((path, data))
        elif re.fullmatch(r"loaded_rank_[0-9]+\.json", path.name):
            libraries = {}
            declared = data.get("libraries", {})
            for name, digest in declared.items() if isinstance(declared, dict) else []:
                if not isinstance(name, str) or not isinstance(digest, str):
                    continue
                base_name = Path(name).name
                if base_name == "libdeepep_kernels.so" or re.fullmatch(r"_C[^/]*\.so", base_name) or base_name in {"libshmem.so", "aclshmem_bootstrap_config_store.so"}:
                    libraries[name] = digest
            loaded.append(dict(path=str(path), rank=data.get("rank"), pid=data.get("pid"), session=data.get("session"), libraries=libraries,
                               evidence="recorded at that run; SDK entries are recorded metadata, not revalidated"))
        elif isinstance(data.get("results"), list):
            parsed.append((path, data))
        elif path.name in {"cases.json", "plan.json", "config.json", "run-plan.json"}:
            auxiliary = data.get("config", data)
            auxiliaries.append(dict(path=str(path), configuration={k: auxiliary[k] for k in CONFIG_KEYS if k in auxiliary},
                                    conditions={k: v for k, v in data.get("conditions", {}).items() if k in {*CONFIG_KEYS, "B", "M", "H", "K", "E", "world_size", "samples", "tail", "timer"}} if isinstance(data.get("conditions"), dict) else {},
                                    world_size=data.get("world_size"), case_count=data.get("case_count")))
        elif "dispatch" in data or "combine" in data:
            parsed.append((path, data))
    # Prefer the complete root/full report over its per-case duplicates. A full
    # report and its summary copy are deduplicated by their result/config bytes.
    if any(isinstance(data.get("results"), list) for _, data in parsed):
        parsed = [(path, data) for path, data in parsed if isinstance(data.get("results"), list)]
    seen = set()
    for path, data in sorted(parsed, key=lambda pair: (pair[0].name != "full-report.json", str(pair[0]))):
        fingerprint = hashlib.sha256(json.dumps({k: data.get(k) for k in ("config", "results", "dispatch", "combine")}, sort_keys=True, allow_nan=True).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        scoped = [(p, d) for p, d in sample_files if p.is_relative_to(path.parent)]
        reports.append(analyze_report(data, path, scoped))
    if results.is_file():
        siblings = sorted(base.glob("loaded_rank_*.json"))[:256]
        for path in siblings:
            data, error = _json(path)
            if not error and isinstance(data, dict):
                declared = data.get("libraries", {})
                loaded.append(dict(path=str(path), rank=data.get("rank"), libraries={k: v for k, v in (declared.items() if isinstance(declared, dict) else [])
                    if isinstance(k, str) and isinstance(v, str) and (Path(k).name in {"libdeepep_kernels.so", "libshmem.so", "aclshmem_bootstrap_config_store.so"} or re.fullmatch(r"_C[^/]*\.so", Path(k).name))},
                    evidence="recorded at that run; not revalidated"))
    if len(paths) > 512:
        issues.append("result evidence capped at512 selected files; raw input was preserved")
    if not reports:
        issues.append("no complete or per-case dispatch/combine report recognized; recorded run conditions may be unavailable")
    return dict(reports=reports, auxiliary=auxiliaries, loaded_libraries=loaded, issues=issues)


def collect_task_intervals(results):
    base = results.parent if results.is_file() else results
    if base == Path(base.anchor):
        return dict(files=[], issues=["CSV scan refused at filesystem root"])
    paths, skipped = _walk(base, {"kernel_details.csv"})
    paths.sort(key=lambda path: ("rank_000" not in path.parts, str(path)))
    files = []
    for path in paths[:16]:
        text, error = _text(path)
        if error:
            files.append(dict(path=str(path), unavailable=error))
            continue
        reader = csv.DictReader(io.StringIO(text))
        groups = {}
        for row in reader:
            normalized = {re.sub(r"[ _]", "", str(key).lower()): value for key, value in row.items() if key is not None}
            name = normalized.get("name", normalized.get("opname", normalized.get("taskname", ""))) or ""
            match = re.search(r"\b((?:dispatch|combine)_server_dedup_kernel(?:_[A-Za-z0-9]+)*)\b", name)
            if not match:
                continue
            def number(*keys):
                value = next((normalized[key] for key in keys if key in normalized), None)
                try:
                    parsed = float(value)
                    return parsed if math.isfinite(parsed) else None
                except (ValueError, TypeError):
                    return None
            groups.setdefault(name, dict(kernel=match.group(1), values=[]))["values"].append((
                number("starttime(us)", "taskstarttime(us)"), number("duration(us)", "taskduration(us)")))
        summary = dict(path=str(path), header=reader.fieldnames, tasks=[])
        for name, group in groups.items():
            values = group["values"]
            starts_complete = all(start is not None for start, _ in values)
            durations_complete = all(duration is not None for _, duration in values)
            task = dict(full_task_name=name, kernel=group["kernel"], N=len(values),
                        starts_complete=starts_complete, durations_complete=durations_complete)
            if starts_complete and len(values) > 1:
                ordered = sorted(values, key=lambda pair: pair[0])
                intervals = [ordered[i][0] - ordered[i - 1][0] for i in range(1, len(ordered))]
                task["first_to_second_start_us"] = intervals[0]
                task["later_adjacent_start_gaps_us"] = dict(N=len(intervals[1:]), min=min(intervals[1:]),
                    p50=statistics.median(intervals[1:]), max=max(intervals[1:])) if len(intervals) > 1 else None
                if durations_complete:
                    idle = [ordered[i][0] - ordered[i - 1][0] - ordered[i - 1][1] for i in range(1, len(ordered))]
                    task["first_end_to_second_start_us"] = idle[0]
                task["order"] = "sorted recorded task start times"
            else:
                task["intervals_unavailable"] = "missing start time column/value or fewer than two matching tasks"
            summary["tasks"].append(task)
        files.append(summary)
    return dict(files=files, total_matching_csv=len(paths), scanned=min(len(paths), 16), issues=skipped,
                note="At most16 existing CSVs, rank_000 first. Start-to-start intervals are task/launch gaps, not kernel execution durations or measured CPU launch cost.")


def _safe_flag_text(value):
    text = str(value)
    if SECRET.search(text):
        return "[redacted-sensitive-option/value]"
    text = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", text)
    return " ".join(text.split())[:12000]


def _filter_build_info(info):
    if not isinstance(info, dict):
        return {}
    result = {key: info[key] for key in INFO_KEYS if key in info}
    # Never copy build_environment or arbitrary environment/config dictionaries.
    for key, value in list(result.items()):
        if key == "shmem":
            if isinstance(value, dict):
                notes = {"revision": _safe_flag_text(value["revision"]) if isinstance(value.get("revision"), str) else value.get("revision")}
                build = value.get("build")
                if isinstance(build, dict):
                    keys = {"cann_version", "cann_timestamp", "target", "rdma_backend", "hns_1825_pi_on_chip", "source_patches"}
                    notes["build"] = {}
                    for field in keys:
                        if field not in build:
                            continue
                        item = build[field]
                        if field == "source_patches" and isinstance(item, list):
                            item = [{k: _safe_flag_text(v) if isinstance(v, str) else v for k, v in patch.items() if k in {"kind", "sha256"}}
                                    if isinstance(patch, dict) else _safe_flag_text(patch) for patch in item]
                        elif isinstance(item, str):
                            item = _safe_flag_text(item)
                        elif isinstance(item, (dict, list)):
                            item = "non-scalar metadata: retained only as type " + type(item).__name__
                        notes["build"][field] = item
                notes["informational_only"] = True
                result[key] = notes
            else:
                result[key] = "unrecognized/missing SDK build metadata; not validated"
            continue
        if isinstance(value, str):
            result[key] = _safe_flag_text(value)
        elif isinstance(value, dict):
            result[key] = {k: _safe_flag_text(v) if isinstance(v, str) else v for k, v in value.items()
                           if k in {"id", "version", "timestamp", "path", "sha256", "source"}}
    return result


def collect_installed(result_evidence):
    packages, seen = [], set()
    for report in result_evidence["reports"]:
        supplied = report.get("package")
        if not isinstance(supplied, str):
            continue
        path = Path(supplied)
        if not path.is_absolute() or path.name != "__init__.py" or path.parent.name != "deep_ep":
            packages.append(dict(reported_path=supplied, status="not inspected: package must name absolute deep_ep/__init__.py"))
            continue
        directory = path.parent
        if str(directory) in seen:
            continue
        seen.add(str(directory))
        info, error = _json(directory / "_build_info.json")
        entry = dict(reported_path=str(path), status="current files only; may differ from files used during the recorded run",
                     build_info=_filter_build_info(info), build_info_unavailable=error, libraries=[])
        if directory.is_dir():
            libraries = sorted(directory.glob("_C*.so")) + [directory / "libdeepep_kernels.so"]
            for library in libraries:
                if not library.is_file():
                    continue
                record = file_hash(library)
                recorded = [item["libraries"].get(str(library)) for item in result_evidence["loaded_libraries"]
                            if str(library) in item.get("libraries", {})]
                record["recorded_run_hashes"] = sorted(set(recorded))
                record["same_bytes_as_recorded_path"] = all(value == record.get("sha256") for value in recorded) if recorded else None
                entry["libraries"].append(record)
        packages.append(entry)
    return packages


def _extract_flags(command):
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    selected, compiler = [], None
    for index, token in enumerate(tokens):
        name = Path(token).name
        if compiler is None and re.fullmatch(r"(?:[\w.-]*-)?(?:g\+\+|gcc|clang\+\+|clang|c\+\+|cc|bisheng)(?:-[0-9.]+)?", name):
            compiler = token
        if token.startswith(("-D", "-O", "-std=", "--npu-arch", "-x", "-f", "-W", "-I", "-L", "-l")):
            value = token
            if token in {"-I", "-L", "-D", "-x"} and index + 1 < len(tokens):
                value += " " + tokens[index + 1]
            selected.append(_safe_flag_text(value))
    return dict(compiler=compiler, flags=list(dict.fromkeys(selected)))


def collect_build(reference):
    records, issues = [], []
    for base in (reference / "build", reference / "third-party/shmem/build"):
        paths, skipped = _walk(base, {"CMakeCache.txt", "flags.make", "link.txt", "build.ninja", "compile_commands.json"})
        issues.extend(skipped)
        for path in paths[:128]:
            text, error = _text(path)
            if error:
                issues.append(f"{path}: {error}")
                continue
            record = dict(path=str(path), evidence="current build file, not proof of historical build", values={}, commands=[])
            if path.name == "CMakeCache.txt":
                for line in text.splitlines():
                    match = re.match(r"([A-Za-z0-9_]+):[^=]*=(.*)", line)
                    if not match:
                        continue
                    key, value = match.groups()
                    if key.startswith(("CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_C_FLAGS", "CMAKE_SHARED_LINKER_FLAGS", "DEEPEP_")) or key in {"CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "ASCEND_HOME_PATH", "SHMEM_ROOT", "SOC_TYPE", "CMAKE_MAKE_PROGRAM", "Python_EXECUTABLE", "PYTHON_EXECUTABLE", "ACLSHMEM_RDMA_BACKEND"}:
                        if not SECRET.search(key):
                            record["values"][key] = _safe_flag_text(value)
            elif path.name == "compile_commands.json":
                try:
                    commands = json.loads(text)
                except ValueError:
                    commands = []
                signatures = set()
                for row in commands if isinstance(commands, list) else []:
                    if not isinstance(row, dict):
                        continue
                    arguments = row.get("arguments")
                    command = row.get("command") if isinstance(row.get("command"), str) else (
                        shlex.join(arguments) if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments) else "")
                    flags = _extract_flags(command)
                    signature = json.dumps(flags, sort_keys=True)
                    if signature not in signatures:
                        signatures.add(signature)
                        record["commands"].append(flags)
                    if len(record["commands"]) >= 64:
                        break
            else:
                for line in text.splitlines():
                    if path.name == "link.txt" or re.match(r"\s*(?:CXX|C|CUDA)_(?:DEFINES|INCLUDES|FLAGS)\s*=", line) or re.match(r"\s*(?:FLAGS|DEFINES|INCLUDES|LINK_FLAGS|LINK_LIBRARIES|command)\s*=", line):
                        parsed = _extract_flags(line)
                        if parsed["compiler"] or parsed["flags"]:
                            if parsed not in record["commands"]:
                                record["commands"].append(parsed)
                    if len(record["commands"]) >= 64:
                        break
            records.append(record)
        if len(paths) > 128:
            issues.append(f"build metadata capped at128 files under {base}")
    return dict(files=records, issues=issues)


def collect_cann(installed, build, environment=None):
    environment = os.environ if environment is None else environment
    homes = set()
    if environment.get("ASCEND_HOME_PATH"):
        homes.add(environment["ASCEND_HOME_PATH"])
    compiler_paths = set()
    for package in installed:
        info = package.get("build_info", {})
        if isinstance(info.get("ASCEND_HOME_PATH"), str):
            homes.add(info["ASCEND_HOME_PATH"])
        for name in ("compiler_path", "device_compiler_path"):
            if isinstance(info.get(name), str):
                compiler_paths.add(info[name])
    for record in build["files"]:
        if record["values"].get("ASCEND_HOME_PATH"):
            homes.add(record["values"]["ASCEND_HOME_PATH"])
        for name in ("CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER"):
            if record["values"].get(name):
                compiler_paths.add(record["values"][name])
    versions = []
    for home in sorted(homes):
        base = Path(home)
        if not base.is_absolute():
            continue
        for relative in ("compiler/version.info", "version.info", "opp/version.info"):
            path = base / relative
            text, error = _text(path)
            if not error:
                values = {}
                for line in text.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip().lower() in {"version", "timestamp", "build_time", "build_date", "platform"}:
                            values[key.strip()] = value.strip()
                versions.append(dict(path=str(path), values=values))
        compiler_paths.add(str(base / "tools/bisheng_compiler/bin/bisheng"))
    binaries = [file_hash(Path(path)) for path in sorted(compiler_paths) if Path(path).is_absolute() and Path(path).is_file()]
    return dict(version_files=versions, compiler_files=binaries,
                note="Current referenced files only; no compiler/SDK command executed and no SHMEM admission gate")


def _compact(value, maximum=1200):
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= maximum else text[:maximum] + "... [full value in local evidence.json]"


def _fmt(value):
    if type(value) in (int, float) and math.isfinite(value):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return "unknown" if value is None else str(value)


def _stats_line(stats):
    if stats.get("status") != "available":
        return "unavailable: " + str(stats.get("reason", "unknown"))
    return "; ".join(f"{label}={_fmt(stats.get(key))}" for label, key in (
        ("min", "min_rank_mean_us"), ("median", "median_rank_mean_us"), ("max", "max_rank_mean_us"),
        ("rank0", "rank0_mean_us"), ("slow_rank", "slowest_rank"), ("observed_ranks", "ranks")))


def _build_display(records):
    selected, seen = [], set()
    def choose(record):
        identity = json.dumps(dict(values=record["values"], commands=record["commands"]), sort_keys=True)
        if identity not in seen:
            selected.append(record)
            seen.add(identity)
    # Keep one Host, Device and SDK cache first, irrespective of how many
    # fingerprint directories remain in this historical checkout.
    for kind in ("host", "device", "sdk"):
        for record in records:
            path = Path(record["path"])
            text = path.as_posix().replace("\\", "/")
            category = "sdk" if "/third-party/shmem/" in text else ("device" if "device" in path.parts else "host")
            if path.name == "CMakeCache.txt" and category == kind:
                choose(record)
                break
    for record in sorted(records, key=lambda row: (
            "deepep_kernels" not in row["path"], Path(row["path"]).name != "CMakeCache.txt", row["path"])):
        choose(record)
    return selected[:12], len(selected)


def render(evidence):
    source, results = evidence["source"], evidence["results"]
    lines = ["REFERENCE SOURCE / PROFILER INSPECTION -- SEND THIS ONE FILE ONLY", "No NPU work, imports of the inspected package, SDK loading, compiler execution, or rerun was performed.",
             f"Reference source: {evidence['reference']}", f"Existing results: {evidence['result_path']}",
             "Current files/build metadata are not assumed to be those used during the historical run.",
             "Raw evidence and complete key-file diffs stay in this new diagnostic group; no source/results were modified.", "", "GIT / SOURCE",
             "HEAD=" + str(source["git"].get("head")) + "; branch=" + str(source["git"].get("branch")),
             "Given directory is repository root=" + str(source["git"].get("is_repository_root")) + "; Git owner=" + str(source["git"].get("repository_owner")),
             "Tracked status (code/build/test scope only; untracked contents are not read as diffs): " + _compact(source["git"].get("tracked_status")),
             "Submodule status: " + _compact(source["git"].get("submodules")),
             "SDK source revision, informational only: " + str(source["git"].get("sdk_source_revision_informational")),
             "Current csrc/deep_ep byte aggregate: " + _compact(source["core_byte_aggregate"]),
             "key file | working SHA256 | cbd04eb | cb59bff"]
    for item in source["key_files"]:
        lines.append(f"{item['path']} | {item['current_sha256'] or item['status']} | " + " | ".join(item["comparisons"].get(label, {}).get("state", "unknown") for label in REFS))
    lines += ["", "STATIC SAMPLING SOURCE (AST only; no source execution)", "Working measure function matches: " + source["known_measure_protocol"],
              "Working parser defaults (not actual-run proof): " + _compact(source["working_protocol"].get("parser_defaults")),
              "function | working AST SHA256 | cbd04eb AST | cb59bff AST"]
    for function in FUNCTIONS:
        current = source["working_protocol"].get("functions", {}).get(function)
        states = []
        for label in REFS:
            old = source["reference_protocols"].get(label, {}).get("functions", {}).get(function)
            states.append("missing" if current is None or old is None else ("equal" if current == old else "changed"))
        lines.append(f"{function} | {current or 'missing'} | " + " | ".join(states))
    if source["known_measure_protocol"] == "cbd04eb":
        lines.append("Known cbd04eb measure: initial128-element async all-reduce; first dispatch; Exp padding(default1000, actual recorded config controls the count); one further all-reduce; remaining dispatches run consecutively. No per-sample Exp/scalar-all-reduce protocol.")
    elif source["known_measure_protocol"] == "cb59bff":
        lines.append("Known cb59 measure: each profiled sample follows one Exp and a scalar all-reduce. This differs from cbd04eb's consecutive later dispatches.")
    else:
        lines.append("Unknown measure AST: do not infer protocol from branch name or result-directory name; inspect local source evidence.")
    lines.append("Result configuration below is the recorded run evidence and takes precedence over current parser defaults; missing fields remain unknown.")
    for report in results["reports"]:
        lines += ["", "RESULT " + report["path"], f"Recorded status={report['status']}; actual world={report['world_size']}; measurement={report['measurement']}",
                  "Recorded configuration: " + _compact(report["config"], 4500),
                  "case/op/route/B | actual N by rank | recorded tail | reported max | recomputed max/min/median/rank0 | slow rank"]
        for case in report["cases"]:
            label = f"{case['case_id']}/{case['operation']}/{case['route']}/B{case['shape'].get('B', '?')}"
            recorded = case["recorded_recomputed"]
            counts = {}
            for item in case["actual_sample_counts"]:
                counts.setdefault(str(item["samples"]), []).append(item["rank"])
            lines.append(label + " | N->ranks:" + _compact(counts, 250) + f" | {case['recorded_tail']} | " +
                         str(case["reported_statistics"].get("max_rank_mean_kernel_us")) + " | " + _stats_line(recorded))
            lines.append("  shape/correctness: " + _compact(dict(shape=case["shape"], correctness=case["correctness"], kernel=case["kernel"], issue=case.get("issue"))))
            for window, stats in case["windows"].items():
                lines.append(f"  tail={window}: " + _stats_line(stats))
            slow_rank = recorded.get("slowest_rank")
            for blocks in case["block_means"]:
                if blocks["rank"] in {0, slow_rank}:
                    lines.append(f"  rank{blocks['rank']} ten-sample block means: " + ", ".join(
                        f"{item['first']}-{item['last']}={_fmt(item['mean_us'])}" for item in blocks["blocks"])[:1000])
            if "reported_minus_recomputed_us" in case:
                lines.append(f"  reported_minus_recomputed_us={case['reported_minus_recomputed_us']:.9g}")
            if case.get("raw_count_matches_recorded_num_iters") is False:
                lines.append("  MISMATCH: raw sample count differs from recorded num_iters; no missing samples are filled in.")
            if case["actual_sample_counts"] and all(item["samples"] == 50 for item in case["actual_sample_counts"]):
                lines.append("  N=50: tail50 uses all50 including the first task; tail30 uses the last30. Re-selecting a tail does not reproduce a different sampling protocol.")
    lines += ["", "AUXILIARY RECORDED PLAN/CONFIG EVIDENCE"]
    lines.extend(_compact(value, 1600) for value in results.get("auxiliary", []))
    lines += ["", "EXISTING TASK START INTERVALS (not kernel duration or CPU launch-cost measurements)"]
    intervals = evidence.get("task_intervals", {})
    lines.append(_compact({k: intervals.get(k) for k in ("total_matching_csv", "scanned", "note")}))
    for file in intervals.get("files", []):
        lines.append(_compact(file, 900))
    lines += ["", "CURRENT INSTALLED FILES FROM RESULT package PATH (not imported; not necessarily historical files)"]
    for package in evidence["installed"][:4]:
        lines.append(_compact(package, 2000))
    if len(evidence["installed"]) > 4:
        lines.append(f"Installed-package entries shown4/{len(evidence['installed'])}; remaining entries stay in evidence.json.")
    lines += ["", "EXISTING RUN-TIME LOADED-LIBRARY RECORDS (historical evidence; SDK hashes are not revalidated)"]
    library_groups = {}
    for item in results["loaded_libraries"]:
        identity = json.dumps(item["libraries"], sort_keys=True)
        group = library_groups.setdefault(identity, dict(ranks=[], libraries=item["libraries"]))
        group["ranks"].append(item.get("rank"))
    for item in list(library_groups.values())[:6]:
        lines.append(_compact(item, 1500))
    if len(library_groups) > 6:
        lines.append(f"Loaded-library maps shown6/{len(library_groups)}; remaining maps stay in evidence.json.")
    lines.append("Identical recorded library maps are grouped above; per-rank record paths/PIDs remain in evidence.json.")
    lines += ["", "CURRENT BUILD METADATA (filtered; compiler tools were not executed)"]
    displayed, unique = _build_display(evidence["build"]["files"])
    lines.append(f"Recorded build files={len(evidence['build']['files'])}; unique flag/cache signatures={unique}; displayed={len(displayed)}; all omitted/duplicate entries stay in evidence.json.")
    for item in displayed:
        lines.append(_compact(dict(path=item["path"], values=item["values"], commands=item["commands"][:3]), 800))
    lines += ["CANN/version/compiler file evidence: " + _compact(evidence["cann"], 3500), "", "LIMITS / MISSING EVIDENCE"]
    issues = source["skipped"] + results["issues"] + evidence["build"]["issues"] + [f"{key}: {value}" for key, value in source["git"].items() if key.endswith("_unavailable")]
    lines.extend(str(value) for value in issues[:40])
    lines.append("No SHMEM source/library/capability/hash value gates this inspection. Installed/build/compiler file hashes describe current bytes only.")
    lines.append("Tail windows larger than actual N are unavailable, never fabricated. Per-rank means/blocks are recomputed from existing raw task durations only.")
    lines.append("Full local evidence includes all recognized cases/ranks, tracked/reference diffs and filtered build metadata; this report may abbreviate long metadata.")
    content = "\n".join(lines) + "\n"
    if len(content.encode("utf-8")) + content.count("\n") > LIMIT:
        # Keep readable complete lines and an explicit omission notice; detailed
        # evidence remains local. No partial JSON or silent analytical omission.
        kept = []
        reserve = 500
        for line in lines:
            candidate = "\n".join(kept + [line]) + "\n"
            if len(candidate.encode("utf-8")) + candidate.count("\n") > LIMIT - reserve:
                break
            kept.append(line)
        kept += ["", "REPORT SIZE LIMIT: later rows/metadata are omitted from this transferable view; all remain in local evidence.json.",
                 "This is a bounded read-only inspection, not a rerun, new qualification or proof of historical environment identity."]
        content = "\n".join(kept) + "\n"
    return content


def inspect(plan):
    reference, results, group = (Path(plan[key]) for key in ("reference", "results", "group"))
    # Repeat the non-overwrite check immediately before collection/publication.
    if group.exists():
        raise ValueError("Output group already exists")
    source = collect_source(reference, Path(plan["root"]))
    result_evidence = collect_results(results)
    installed = collect_installed(result_evidence)
    build = collect_build(reference)
    evidence = dict(format="reference-read-only-inspection-v1", reference=str(reference), result_path=str(results),
                    source=source, results=result_evidence, installed=installed, build=build,
                    cann=collect_cann(installed, build), task_intervals=collect_task_intervals(results))
    content = render(evidence)
    raw = content.encode("utf-8")
    if len(raw) + content.count("\n") > LIMIT:
        raise ValueError("Inspection summary exceeds the UTF-8/CRLF transfer limit")
    group.mkdir(parents=True, exist_ok=False)
    (group / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (group / "tracked_source.diff").write_text(source["tracked_diff"], encoding="utf-8")
    for label in REFS:
        difference = "\n".join(value for key, value in source["reference_diffs"].items() if key.startswith(label + ":"))
        (group / f"working_vs_{label}.diff").write_text(difference, encoding="utf-8")
    helper = Path(__file__).with_name("transfer_budget.py")
    budget = types.ModuleType("reference_inspection_transfer_budget")
    exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), budget.__dict__)
    with budget.transaction(group, "summary", [hashlib.sha256(raw).hexdigest()]):
        (group / "SEND_FIRST.txt").write_bytes(raw)
    return group / "SEND_FIRST.txt"


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        destination = inspect(make_plan(args))
        print(destination)
        print(f"Return only SEND_FIRST.txt ({destination.stat().st_size} UTF-8 bytes); detailed evidence stays local.")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Reference inspection: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
