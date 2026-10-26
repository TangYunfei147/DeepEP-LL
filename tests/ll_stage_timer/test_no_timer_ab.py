"""CPU-only orchestration, source identity and matched-statistic regressions."""
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types

import pytest

SPEC = importlib.util.spec_from_file_location("no_timer_ab_tested", Path(__file__).with_name("no_timer_ab.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def args(*extra, stage="all", name="trial"):
    return runner.parser().parse_args(["run", "--run-name", name, "--stage", stage,
        "--devices", ",".join(map(str, range(16))), "--master-addr", "10.1.2.3", "--master-port", "29501", "--shmem-port", "29502", *extra])


def proof(variant):
    value = runner.VARIANTS[variant]
    return dict(format="projection-test", reference=runner.BASE, mode=value["mode"], manifest="ll-no-timer-source.json",
                source_sha256=("a" if value["mode"] == "normal-off" else "b") * 64,
                dispatch_launch_bytes=value["dispatch_bytes"], combine_launch_bytes=value["combine_bytes"], changed_paths=[], deleted_paths=[])


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def build_fixture(root):
    group = root / "diagnostic-runs/trial"
    variants = {}
    for name, selected in runner.VARIANTS.items():
        layout = runner._layout(group, name)
        write(layout["source"] / "tests/ll_stage_timer/run.py", "fixture")
        write(layout["python"], "fixture interpreter")
        wheel = write(layout["wheels"] / (name + ".whl"), ("wheel-" + name).encode())
        host = dict(variant=name, host_only=True, device_queried=False, source_sha256=proof(name)["source_sha256"])
        write(layout["host"], json.dumps(host))
        variants[name] = dict(**selected, source_sha256=proof(name)["source_sha256"], proof=proof(name),
            python=layout["python"].relative_to(group).as_posix(), wheel=wheel.relative_to(group).as_posix(),
            host=layout["host"].relative_to(group).as_posix(), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    receipt = dict(format=runner.BUILD_FORMAT, reference=runner.BASE, shared_sdk="source-normal/third-party/shmem/install/shmem", sdk_builds=1,
                   controller_revision="f" * 40, variants=variants)
    write(group / "build-receipt.json", json.dumps(receipt))
    write(group / "wheel-map.json", json.dumps(dict(format=runner.BUILD_FORMAT, reference=runner.BASE, variants={name: item["wheel"] for name, item in variants.items()})))
    return group, receipt


def report_for(run, receipt):
    chosen = receipt["variants"][run["variant"]]
    identity = dict(active_aiv=56, send_cores=28, recv_cores=28, logical_group=16, small_b_prefix_cores=0,
        core_protocol=chosen["protocol"], dispatch_impl=chosen["dispatch_impl"], implementation=chosen["implementation"],
        instrumentation_source=chosen["mode"], dispatch_launch_bytes=chosen["dispatch_bytes"], combine_launch_bytes=chosen["combine_bytes"],
        source_sha256=chosen["source_sha256"], transport="mte")
    config = dict(num_processes=16, num_nodes=1, node_rank=0, num_max_tokens_per_rank=32, num_tokens=list(runner.BATCHES),
        hidden=3584, num_topk=8, num_experts=384, dtype="fp8", fp8_scale_dtype="int32", do_cpu_sync=0,
        warmup=10, num_iters=100, perf_tail=50, seed=run["seed"], route=run["route"], delay_rows=118260, delay_cols=30720,
        delay_exp_before=1, delay_exp_after=0, group_backend="hccl", dispatch_only=False, correctness_repeats=3,
        repeat_combine=2, skip_check=False, skip_perf_test=False, use_1g_huge_pages=0)
    cap_key = "ll_stage_timer_capabilities" if chosen["mode"] == "normal-off" else "runtime_capabilities"
    config[cap_key] = [dict(identity) for _ in range(16)]
    if chosen["mode"] == "normal-off":
        config["ll_stage_timer"] = False
        for cap in config[cap_key]:
            cap["stage_timer"] = False
    results = []
    for case_id, batch in enumerate(runner.BATCHES):
        digest = hashlib.sha256(f"{run['route']}-{run['seed']}-{batch}".encode()).hexdigest()
        case = dict(case_id=case_id, route=run["route"], backend="ll", correctness="passed",
            config=dict(B=batch, M=32, H=3584, num_topk=8, num_experts=384, dtype="fp8", do_cpu_sync=False, fp8_scale_dtype="int32"),
            comparison=dict(identity, sampling_protocol="exp-reduce-stable-tail-v2", dispatch_only=False,
                            profile_seed=run["seed"], profile_route_sha256=digest, profile_route_fingerprint_schema=1))
        for operation in ("dispatch", "combine"):
            ranks = []
            for rank in range(16):
                mean = 20 + batch / 10 + rank / 100 + run["repeat"] / 1000 + (2 if operation == "combine" else 0) - (1 if run["side"] == "B" else 0)
                ranks.append(dict(rank=rank, samples=50, mean_us=mean, kernel_us=[mean + 7] * 50 + [mean] * 50))
            tails = [value for rank in ranks for value in rank["kernel_us"][-50:]]
            case[operation] = dict(kernel=operation + "_server_dedup_kernel", measurement="kernel_duration_us", perf_tail=50,
                ranks=ranks, max_rank_mean_kernel_us=max(rank["mean_us"] for rank in ranks),
                mean_kernel_us=sum(tails) / len(tails), p99_kernel_us=runner._percentile(tails, .99))
        results.append(case)
    return dict(status="completed", world_size=16, case_count=7, result_count=7, config=config, results=results)


def test_build_plan_is_pure_fixed_and_builds_every_wheel_before_host_publication(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *values, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu", "deep_ep"}
        return original_import(name, *values, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("planning must not execute commands"))
    group = tmp_path / "root with spaces/diagnostic-runs/trial"
    plan = runner.build_plan(group, root=tmp_path / "root with spaces", python="test-python")
    assert not group.exists() and json.loads(json.dumps(plan)) == plan
    names = [step["name"] for step in plan["steps"]]
    clones = [step for step in plan["steps"] if step["name"].endswith("-clone")]
    assert len(clones) == 2 and all("--no-hardlinks" in step["argv"] and "--dissociate" in step["argv"] for step in clones)
    assert sum(name == "shared-sdk-once" for name in names) == 1
    builds = {variant + "-wheel" for variant in runner.VARIANTS}
    assert sum(name in builds for name in names) == 3
    assert max(index for index, name in enumerate(names) if name.endswith("-wheel")) < min(index for index, name in enumerate(names) if name.endswith("_install"))
    assert all(plan["steps"][index - 1]["name"].endswith("-clean-wheel") for index, name in enumerate(names) if name in builds)
    assert all("--system-site-packages" in step["argv"] for step in plan["steps"] if step["name"].endswith("-venv"))
    assert not any("test_ep_low_latency.py" in arg for step in plan["steps"] for arg in step["argv"])


def test_run_matrix_has_twelve_serial_runs_both_operations_and_no_timer(tmp_path):
    plan = runner.make_run_plan(args(), root=tmp_path)
    assert len(plan["runs"]) == 12
    assert len({run["name"] for run in plan["runs"]}) == 12
    assert [(run["stage"], run["route"], run["side"]) for run in plan["runs"]] == [
        (stage, route, side) for stage, route in (("timer", "normal"), ("optimization", "normal"), ("optimization", "hotspot")) for side in "ABBA"]
    for run in plan["runs"]:
        argv = run["command"]
        assert argv[1].endswith("/tests/ll_stage_timer/run.py") or argv[1].endswith("\\tests\\ll_stage_timer\\run.py")
        for forbidden in ("--dispatch-only", "--ll-stage-timer", "--ll-profile-capture", "--skip-check", "--skip-perf-test"):
            assert forbidden not in argv
        assert argv[argv.index("--num-iters") + 1] == "100" and argv[argv.index("--perf-tail") + 1] == "50"
        assert argv[argv.index("--correctness-repeats") + 1] == "3"
        assert argv[argv.index("--delay-exp-before") + 1] == "1"
    assert not Path(plan["group"]).exists()
    reversed_plan = runner.make_run_plan(args("--seeds", "0", "2", stage="timer"), root=tmp_path)
    assert [(run["seed"], run["side"]) for run in reversed_plan["runs"]] == [(0, side) for side in "ABBA"] + [(2, side) for side in "BAAB"]
    assert len(runner.make_run_plan(args(stage="optimization"), root=tmp_path)["runs"]) == 8


@pytest.mark.parametrize("extra", [("--devices", "0,1"), ("--seeds", "0", "0"), ("--seeds", "-1"), ("--master-port", "29502")])
def test_run_bad_configuration_is_rejected_before_creating_files(tmp_path, extra):
    with pytest.raises(ValueError):
        runner.make_run_plan(args(*extra), root=tmp_path)
    assert not (tmp_path / "diagnostic-runs").exists()


@pytest.mark.parametrize("folder", ["../outside", "build", "tests", "scripts", "third-party/shmem/install", "experiments"])
def test_output_cannot_escape_or_enter_source_build_dependency_trees(tmp_path, folder):
    options = runner.parser().parse_args(["build", "--run-name", "trial", "--output-root", folder, "--plan"])
    with pytest.raises(ValueError):
        runner.group_path(options, root=tmp_path)


def test_all_variants_share_one_explicit_sdk_and_private_log_cache_paths(tmp_path):
    inherited = dict(DEEPEP_LL_STAGE_TIMER="ON", DEEPEP_LL_SEND_CORES="40", DEEPEP_LL_DISPATCH_IMPL="cb59-pair",
        PYTHONPATH="/wrong/python", SHMEM_ROOT="/wrong/sdk", SHMEM_SOURCE_PATH="/wrong/source", CATLASS_ROOT="/wrong/catlass",
        WORLD_SIZE="64", LOCAL_RANK="4", LD_LIBRARY_PATH="/toolchain/lib", ASCEND_HOME_PATH="/toolchain", CXXFLAGS="original flags")
    for variant in runner.VARIANTS:
        env = runner._environment(tmp_path, variant, run="sample", devices=list(range(16)), inherited=inherited)
        assert env["SHMEM_ROOT"] == str(tmp_path / "source-normal/third-party/shmem/install/shmem")
        assert env["LD_LIBRARY_PATH"].startswith(env["SHMEM_ROOT"])
        assert env["DEEPEP_LL_STAGE_TIMER"] == "OFF" and env["DEEPEP_LL_SEND_CORES"] == env["DEEPEP_LL_RECV_CORES"] == "28"
        assert env["DEEPEP_LL_DISPATCH_IMPL"] == runner.VARIANTS[variant]["dispatch_impl"]
        assert env["CCACHE_DISABLE"] == "1" and env["CXXFLAGS"] == "original flags"
        assert all(name not in env for name in ("PYTHONPATH", "SHMEM_SOURCE_PATH", "CATLASS_ROOT", "WORLD_SIZE", "LOCAL_RANK"))
        assert all(env[key] == ",".join(map(str, range(16))) for key in runner.VISIBILITY)
        for key in ("TMPDIR", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "SHMEM_LOG_PATH"):
            assert Path(env[key]).is_relative_to(tmp_path / runner.VARIANTS[variant]["source"] / ".cache/no-timer-ab")


def mock_build(tmp_path, monkeypatch, fail=None):
    root = tmp_path / "controller with spaces"
    root.mkdir()
    group = root / "diagnostic-runs/trial"
    plan = runner.build_plan(group, root=root, python="test-python")
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner, "_git", lambda source, *args: "f" * 40 if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(runner, "_proof", lambda group, variant, controller: proof(variant))
    environment = runner._environment
    monkeypatch.setattr(runner, "_environment", lambda *a, **k: environment(*a, **dict(k, inherited={})))
    calls = []
    def execute(argv, cwd, env, log):
        step = plan["steps"][len(calls)]
        calls.append(step["name"])
        assert argv == step["argv"] and env["SHMEM_ROOT"] == str(group / plan["shared_sdk"])
        write(log, "mock command log")
        if fail == step["name"]:
            raise subprocess.CalledProcessError(7, argv)
        if step["name"].endswith("-clone"):
            Path(argv[-1]).mkdir()
        if step["name"].endswith("-venv"):
            write(runner._layout(group, step["variant"])["python"], "mock python")
        if step["name"] in {variant + "-wheel" for variant in runner.VARIANTS}:
            write(runner._layout(group, step["variant"])["wheels"] / "deep_ep.whl", "built full wheel")
        if step["name"].endswith("_verify-host"):
            assert sum(name in {variant + "-wheel" for variant in runner.VARIANTS} for name in calls) == 3
            write(runner._layout(group, step["variant"])["host"], json.dumps(dict(variant=step["variant"], host_only=True, device_queried=False,
                source_sha256=proof(step["variant"])["source_sha256"])))
        assert not (group / "wheel-map.json").exists()
    monkeypatch.setattr(runner, "_command", execute)
    return plan, calls


def test_build_publishes_relative_map_only_after_three_full_wheels_and_host_checks(tmp_path, monkeypatch):
    plan, calls = mock_build(tmp_path, monkeypatch)
    destination = runner.execute_build(plan)
    assert len(calls) == len(plan["steps"])
    receipt = runner._build_receipt(Path(plan["group"]))
    assert receipt["sdk_builds"] == 1 and len(receipt["variants"]) == 3
    mapping = json.loads(destination.read_text())
    assert all(not Path(value).is_absolute() for value in mapping["variants"].values())
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        runner.execute_build(plan)
    assert destination.read_bytes() == before


def test_build_failure_stops_before_install_sampling_or_map_and_keeps_logs(tmp_path, monkeypatch):
    plan, calls = mock_build(tmp_path, monkeypatch, fail="replay-clean-wheel")
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_build(plan)
    assert calls[-1] == "replay-clean-wheel"
    assert not any(name.endswith("_install") for name in calls)
    assert not Path(plan["group"], "wheel-map.json").exists()
    assert list(Path(plan["group"], "logs/build").glob("*replay-clean-wheel.log"))


def test_missing_base_or_dirty_controller_fails_before_output_directory(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    plan = runner.build_plan(root / "diagnostic-runs/new", root=root)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner, "_git", lambda *a: " M tracked.py")
    with pytest.raises(ValueError, match="tracked changes"):
        runner.execute_build(plan)
    assert not Path(plan["group"]).exists()
    def absent(*args):
        raise subprocess.CalledProcessError(1, ["git", "cat-file"])
    monkeypatch.setattr(runner, "_git", absent)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_build(plan)
    assert not Path(plan["group"]).exists()


def test_receipt_rejects_changed_project_wheel_without_reading_sdk(tmp_path):
    group, _ = build_fixture(tmp_path)
    runner._build_receipt(group)
    runner._one_wheel(group, "replay-clean").write_bytes(b"changed project implementation")
    with pytest.raises(ValueError, match="wheel bytes changed"):
        runner._build_receipt(group)


def test_receipt_allows_normal_posix_venv_executable_symlink_only(tmp_path, monkeypatch):
    group, _ = build_fixture(tmp_path)
    python = runner._layout(group, "replay-off")["python"]
    original_lstat = Path.lstat
    def lstat(path):
        if path == python:
            return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    assert runner._build_receipt(group)["reference"] == runner.BASE
    with pytest.raises(ValueError, match="symlinks"):
        runner._local(group, python)


def mock_run(root, monkeypatch, stage="all", fail_order=None, seeds=None):
    group, receipt = build_fixture(root)
    options = args(*(["--seeds", *map(str, seeds)] if seeds is not None else []), stage=stage)
    plan = runner.make_run_plan(options, root=root)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner, "_proof", lambda group, variant, controller: proof(variant))
    environment = runner._environment
    monkeypatch.setattr(runner, "_environment", lambda *a, **k: environment(*a, **dict(k, inherited={})))
    calls, measured = [], []
    def execute(argv, cwd, env, log):
        calls.append(argv)
        write(log, "mock execution")
        if "_verify-host" in argv:
            assert not measured
            return
        run = plan["runs"][len(measured)]
        measured.append(run["name"])
        assert sum("_verify-host" in command for command in calls) == 3
        assert argv == run["command"] and cwd == runner._layout(group, run["variant"])["source"]
        assert env["SHMEM_ROOT"] == str(group / "source-normal/third-party/shmem/install/shmem")
        assert all(env[name] == ",".join(map(str, range(16))) for name in runner.VISIBILITY)
        if len(measured) == fail_order:
            raise subprocess.CalledProcessError(8, argv)
        write(group / run["report"], json.dumps(report_for(run, receipt)))
    monkeypatch.setattr(runner, "_command", execute)
    return group, receipt, plan, calls, measured


def test_serial_abba_produces_one_bounded_report_and_analysis_is_idempotent(tmp_path, monkeypatch):
    group, _, plan, calls, measured = mock_run(tmp_path, monkeypatch)
    path = runner.execute_run(plan)
    text = path.read_text()
    assert len(measured) == 12 and len(calls) == 15
    assert len(path.read_bytes()) + text.count("\n") <= 60000
    assert "BF16 combine" in text and "Hotspot B>=16 is a regression guard" in text
    assert "optimization hotspot 0 32 combine" in text and "timer normal 0 1 dispatch" in text
    assert "-1.000000" in text
    assert runner._load_budget().budget_state(group)["issued"] == 1
    assert runner.analyze(group) == path and runner._load_budget().budget_state(group)["issued"] == 1
    with pytest.raises(ValueError, match="already exists"):
        runner.execute_run(plan)
    assert len(measured) == 12


def test_run_failure_is_not_retried_and_cannot_be_reported_completed(tmp_path, monkeypatch):
    group, _, plan, _, measured = mock_run(tmp_path, monkeypatch, fail_order=2)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    assert len(measured) == 2 and not (group / "SEND_FIRST.txt").exists()
    state = json.loads((group / plan["status_file"]).read_text())
    assert state["status"] == "failed" and state["runs"][-1]["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        runner.analyze(group)


def test_separate_timer_then_optimization_runs_accumulate_without_path_overlap(tmp_path, monkeypatch):
    group, receipt = build_fixture(tmp_path)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner, "_proof", lambda group, variant, controller: proof(variant))
    environment = runner._environment
    monkeypatch.setattr(runner, "_environment", lambda *a, **k: environment(*a, **dict(k, inherited={})))
    observed = []
    for index, stage in enumerate(("timer", "optimization"), 1):
        plan = runner.make_run_plan(args(stage=stage), root=tmp_path)
        current, host_checks = [], []
        def execute(argv, cwd, env, log):
            write(log, "mock sequential stage")
            if "_verify-host" in argv:
                host_checks.append(argv)
                assert not current
                return
            assert len(host_checks) == 3
            run = plan["runs"][len(current)]
            current.append(run["name"])
            observed.append(run["name"])
            write(group / run["report"], json.dumps(report_for(run, receipt)))
        monkeypatch.setattr(runner, "_command", execute)
        destination = runner.execute_run(plan)
        assert runner._load_budget().budget_state(group)["issued"] == index
    text = destination.read_text()
    assert len(observed) == len(set(observed)) == 12
    assert "timer normal 0 32 combine" in text and "optimization hotspot 0 32 combine" in text
    assert len(list((group / "run-plans").glob("*.json"))) == 2
    with pytest.raises(ValueError, match="already exists"):
        runner.execute_run(runner.make_run_plan(args(stage="all"), root=tmp_path))
    assert len(list((group / "run-plans").glob("*.json"))) == 2


def test_frozen_source_failure_stops_before_any_host_or_device_launch(tmp_path, monkeypatch):
    group, _, plan, calls, measured = mock_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_proof", lambda *a: dict(proof("replay-off"), source_sha256="c" * 64))
    with pytest.raises(ValueError, match="Frozen source changed"):
        runner.execute_run(plan)
    assert calls == measured == [] and not (group / "SEND_FIRST.txt").exists()


@pytest.mark.parametrize("mutation", ["dispatch_only", "missing_combine", "seed", "source", "abi", "rank", "count", "max", "p99", "legacy", "shape", "on"])
def test_mismatched_or_incomplete_measurements_fail_closed(tmp_path, mutation):
    _, receipt = build_fixture(tmp_path)
    run = runner.make_run_plan(args(), root=tmp_path)["runs"][0]
    report = report_for(run, receipt)
    case = report["results"][0]
    if mutation == "dispatch_only": report["config"]["dispatch_only"] = True
    elif mutation == "missing_combine": del case["combine"]
    elif mutation == "seed": case["comparison"]["profile_seed"] += 2
    elif mutation == "source": case["comparison"]["source_sha256"] = "d" * 64
    elif mutation == "abi": case["comparison"]["dispatch_launch_bytes"] = 136
    elif mutation == "rank": case["dispatch"]["ranks"].pop()
    elif mutation == "count": case["dispatch"]["ranks"][0]["kernel_us"].pop()
    elif mutation == "max": case["dispatch"]["max_rank_mean_kernel_us"] += 1
    elif mutation == "p99": case["dispatch"]["p99_kernel_us"] += 1
    elif mutation == "legacy": case["comparison"]["sampling_protocol"] = "legacy-consecutive"
    elif mutation == "shape": case["config"]["M"] = 256
    elif mutation == "on": report["config"]["ll_stage_timer"] = True
    with pytest.raises(ValueError):
        runner.validate_report(report, run, receipt)


def test_summary_averages_run_maxima_not_pooled_rank_means(tmp_path):
    _, receipt = build_fixture(tmp_path)
    plan = runner.make_run_plan(args(stage="timer"), root=tmp_path)
    rows = []
    for run in plan["runs"]:
        report = report_for(run, receipt)
        rows.extend(runner.validate_report(report, run, receipt))
    summaries = runner.summarize(rows)
    row = next(item for item in summaries if item["B"] == 1 and item["operation"] == "dispatch")
    assert row["A"] == pytest.approx((row["A1"] + row["A2"]) / 2)
    assert row["A"] > row["A_pooled_mean_us"]
    assert row["delta_us"] == pytest.approx(-1)
    rows[-1]["route_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="route fingerprints differ"):
        runner.summarize(rows)


def test_four_seed_full_matrix_stays_under_single_report_byte_limit(tmp_path):
    _, receipt = build_fixture(tmp_path)
    plan = runner.make_run_plan(args("--seeds", "0", "2", "4", "6"), root=tmp_path)
    rows = [row for run in plan["runs"] for row in runner.validate_report(report_for(run, receipt), run, receipt)]
    text, summary = runner.render_summary(receipt, [plan], rows)
    assert len(plan["runs"]) == 48 and len(summary) == 168
    assert len(text.encode()) + text.count("\n") <= 60000
    assert "optimization hotspot 6 32 combine" in text
    guards = [row for row in summary if row["regression_guard"]]
    assert len(guards) == 4 * 3 * 2


def test_budget_exhaustion_preserves_previous_summary(tmp_path, monkeypatch):
    group, receipt, plan, _, _ = mock_run(tmp_path, monkeypatch, stage="timer")
    path = runner.execute_run(plan)
    before = path.read_bytes()
    budget = runner._load_budget()
    with budget.transaction(group, "detail", [hashlib.sha256(f"detail{i}".encode()).hexdigest() for i in range(9)]):
        pass
    # A valid changed statistic creates a distinct summary without changing any
    # source/SDK identity or rewriting the existing transferable report.
    run = plan["runs"][0]
    report = json.loads((group / run["report"]).read_text())
    perf = report["results"][0]["dispatch"]
    for rank in perf["ranks"]:
        rank["kernel_us"] = [value + .1 for value in rank["kernel_us"]]
        rank["mean_us"] += .1
    for key in ("mean_kernel_us", "p99_kernel_us", "max_rank_mean_kernel_us"):
        perf[key] += .1
    (group / run["report"]).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="transmission limit"):
        runner.analyze(group)
    assert path.read_bytes() == before and budget.budget_state(group)["issued"] == 10


def test_host_identity_verifies_fixed_source_and_abi_without_device_query(tmp_path, monkeypatch):
    group, _ = build_fixture(tmp_path)
    variant = "stream-clean"
    layout = runner._layout(group, variant)
    package = write(layout["venv"] / "lib/site-packages/deep_ep/__init__.py", "fake package").parent
    value = runner.VARIANTS[variant]
    info = dict(source_sha256=proof(variant)["source_sha256"], ll_dispatch_impl=value["dispatch_impl"],
                ll_send_cores=28, ll_recv_cores=28, ll_small_b_prefix_cores=0, arch="Ascend950", build_type="Release",
                ll_instrumentation_source=value["mode"], dispatch_launch_bytes=136, combine_launch_bytes=88)
    write(package / "_build_info.json", json.dumps(info))
    native = types.SimpleNamespace(runtime_protocol=1, low_latency_dispatch_impl=value["dispatch_impl"],
        low_latency_implementation=value["implementation"], low_latency_core_protocol=value["protocol"], low_latency_ranks_per_group=16,
        low_latency_small_b_prefix_cores=0, low_latency_instrumentation_source=value["mode"], dispatch_launch_bytes=136, combine_launch_bytes=88)
    native.low_latency_core_config = lambda: pytest.fail("Host verification must not query the device")
    module = types.ModuleType("deep_ep")
    module.__file__ = str(package / "__init__.py")
    loader = types.ModuleType("deep_ep._native")
    loader.require_native = lambda message: native
    monkeypatch.setitem(sys.modules, "deep_ep", module)
    monkeypatch.setitem(sys.modules, "deep_ep._native", loader)
    monkeypatch.setattr(runner, "_proof", lambda *a: proof(variant))
    result = runner.verify_host(group, variant, controller=tmp_path)
    assert result["host_only"] and result["device_queried"] is False
    native.dispatch_launch_bytes = 152
    with pytest.raises(ValueError, match="dispatch_launch_bytes"):
        runner.verify_host(group, variant, controller=tmp_path)


def test_real_fixed_projection_keeps_provisioned_cache_out_of_source_identity(tmp_path):
    """Local Git/file integration only: no submodule download, SDK or devices."""
    # This Windows Python runtime cannot open the longest baseline source names
    # below pytest's long per-test path. Keep this disposable local fixture in
    # the already-ignored project .tmp, without touching production sources.
    group = (Path(tempfile.mkdtemp(prefix="nt-", dir=runner.ROOT / ".tmp"))
             if sys.platform == "win32" else tmp_path / "diagnostic-runs/local-check")
    source = group / "source-clean"
    group.mkdir(parents=True, exist_ok=True)
    cloned = subprocess.run(["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "clone", "--config", "core.longpaths=true", "--config", "core.autocrlf=false", "--no-hardlinks", "--dissociate", "--no-checkout", runner.ROOT.as_posix(), source.as_posix()],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert cloned.returncode == 0, cloned.stderr.decode(errors="replace")
    checked = subprocess.run(["git", "-C", str(source), "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "checkout", "--detach", runner.BASE],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert checked.returncode == 0, checked.stderr.decode(errors="replace")
    projection = runner._projection()
    before = projection.prepare(source, "clean-source", controller=runner.ROOT)
    env = runner._environment(group, "replay-clean", run="mock-runtime", inherited={})
    runner._provision_environment(group, env)
    cache = write(Path(env["ASCEND_CACHE_PATH"]) / "compiler-cache.json", '{"mock":"existing tool output"}')
    subprocess.run(["git", "-C", str(source), "check-ignore", "--quiet", str(cache)], check=True)
    after = projection.verify(source, "clean-source", controller=runner.ROOT)
    assert before == after
    assert runner._proof(group, "replay-clean")["source_sha256"] == before["source_sha256"]

    # The exact generated ST parser must accept the orchestrator's arguments.
    # Executing module definitions is CPU-only: the framework imports are inside
    # worker/measure functions, which this test never calls.
    namespace = dict(__file__=str(source / "tests/elastic/test_ep_low_latency.py"), __name__="projected_parser_fixture")
    exec(compile(Path(namespace["__file__"]).read_text(encoding="utf-8"), namespace["__file__"], "exec"), namespace)
    run = runner.make_run_plan(args(), root=tmp_path)["runs"][1]
    forwarded = run["command"][2:]
    index = forwarded.index("--devices")
    forwarded = forwarded[:index] + forwarded[index + 2:]
    options = namespace["parse_args"](forwarded)
    assert options.dispatch_only is False and options.num_iters == 100 and options.repeat_combine == 2
    assert namespace["validate_args"](options, 16) == 32
