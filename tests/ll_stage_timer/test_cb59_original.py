"""CPU-only original-ST orchestration, evidence validation and bounded export."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import types

import pytest

SPEC = importlib.util.spec_from_file_location("cb59_original_runner", Path(__file__).with_name("cb59_original.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ROUTES = '''import random
def make_routes(world, b, k, e, seed):
    rng = random.Random(seed)
    result = []
    for _ in range(world):
        rows = []
        for _ in range(b):
            chosen = rng.sample(range(e), k)
            rows.append(chosen)
        result.append(rows)
    return result
raise AssertionError("The historical module must not be executed during analysis")
'''


def group_fixture(tmp_path, name="original001"):
    group = tmp_path / "diagnostic-runs" / name
    source = group / "source"
    script = source / "tests/elastic/test_ep_low_latency.py"
    script.parent.mkdir(parents=True)
    script.write_text(ROUTES, encoding="utf-8")
    python = group / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("mock existing venv executable")
    wheel = group / "wheels/reference.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"mock wheel")
    receipt = dict(reference=runner.REFERENCE, source="source", python="venv/bin/python",
                   wheel="wheels/reference.whl", source_patch_paths=["scripts/build_shmem.py"],
                   sdk_source_revision="original-submodule-metadata-only")
    (group / "build-receipt.json").write_text(json.dumps(receipt))
    return group


def args(name="original001", *extra):
    return runner.parser().parse_args(["run", "--run-name", name,
        "--devices", ",".join(map(str, range(16))), "--master-addr", "10.0.0.1",
        "--master-port", "29501", "--shmem-port", "29502", *extra])


def report_for(run, batches, active=56, group=16):
    config = dict(num_processes=16, num_nodes=1, node_rank=0, num_max_tokens_per_rank=32,
        num_tokens=list(batches), hidden=3584, num_topk=8, num_experts=384, dtype="fp8", fp8_scale_dtype="int32",
        do_cpu_sync=0, warmup=10, num_iters=100, perf_tail=50, seed=run["seed"], delay_rows=118260,
        delay_cols=30720, delay_exp_before=1, delay_exp_after=0, group_backend="hccl", route="normal",
        dispatch_only=True, ll_stage_timer=False, skip_check=False, skip_perf_test=False,
        ll_stage_timer_capabilities=[dict(active_aiv=active, logical_group=group, core_protocol=2, stage_timer=False)
                                     for _ in range(16)])
    results = []
    for case_id, batch in enumerate(batches):
        ranks = []
        for rank in range(16):
            # Early samples are deliberately slower; using all100 must fail validation.
            value = 20.0 + batch / 10 + rank / 100 + run["seed"] / 1000 + run["round"] / 10000
            ranks.append(dict(rank=rank, kernel_us=[value + 7] * 50 + [value] * 50, samples=50, mean_us=value))
        results.append(dict(case_id=case_id, route="normal", backend="ll", correctness="passed",
            tested_operations=["dispatch"],
            config=dict(B=batch, M=32, H=3584, num_topk=8, num_experts=384, dtype="fp8",
                        do_cpu_sync=False, fp8_scale_dtype="int32"),
            dispatch=dict(kernel="dispatch_server_dedup_kernel", measurement="kernel_duration_us", perf_tail=50,
                          ranks=ranks, max_rank_mean_kernel_us=max(row["mean_us"] for row in ranks))))
    return dict(status="completed", world_size=16, case_count=len(batches), result_count=len(batches),
                config=config, results=results)


def test_pure_plan_has_original_arguments_visibility_and_reversed_seed_order(tmp_path):
    group = group_fixture(tmp_path)
    before = sorted(path.relative_to(group).as_posix() for path in group.rglob("*"))
    plan = runner.make_run_plan(args(), root=tmp_path)
    assert [(r["round"], r["seed"]) for r in plan["runs"]] == [(1, 0), (1, 2), (2, 2), (2, 0)]
    for run in plan["runs"]:
        command = run["command"]
        assert command[:2] == [str(group / "venv/bin/python"), str(group / "source/tests/elastic/test_ep_low_latency.py")]
        assert command[command.index("--num-processes") + 1] == "16"
        assert command[command.index("--do-cpu-sync") + 1] == "0"
        assert command[command.index("--num-iters") + 1] == "100"
        assert command[command.index("--perf-tail") + 1] == "50"
        assert all(option not in command for option in ("--devices", "--run-name", "--backend", "--use-1g-huge-pages", "--ll-stage-timer", "--skip-check"))
    after = sorted(path.relative_to(group).as_posix() for path in group.rglob("*"))
    assert before == after and not (group / "runs").exists()


@pytest.mark.parametrize("parent", ["build", "third-party/shmem/install", "tests", "benchmarks", "experiments"])
def test_group_stays_outside_cleaned_or_fingerprinted_trees(tmp_path, parent):
    options = runner.parser().parse_args(["build", "--run-name", "new", "--output-root", parent, "--plan"])
    with pytest.raises(ValueError, match="separate project output"):
        runner.group_path(options, root=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("extra", [
    ["--devices", "0,1,2,3,4,5,6,7"],
    ["--devices", ",".join(["0"] * 16)],
    ["--master-port", "0"], ["--shmem-port", "29501"], ["--bs", "33"],
    ["--seeds", "0", "0"], ["--rounds", "0"],
])
def test_invalid_plans_do_not_start_or_write(tmp_path, extra):
    group = group_fixture(tmp_path)
    with pytest.raises(ValueError):
        runner.make_run_plan(args("original001", *extra), root=tmp_path)
    assert not (group / "runs").exists()


def test_plan_main_never_executes_or_imports_torch(tmp_path, monkeypatch, capsys):
    group_fixture(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    original_group_path = runner.group_path
    original_run_plan = runner.make_run_plan
    monkeypatch.setattr(runner, "group_path", lambda value, root=None: original_group_path(value, root=tmp_path))
    monkeypatch.setattr(runner, "make_run_plan", lambda value: original_run_plan(value, root=tmp_path))
    def forbidden(*_args, **_kwargs):
        pytest.fail("Plan attempted execution or filesystem mutation")
    monkeypatch.setattr(runner, "execute_run", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    before_torch = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    argv = ["run", "--run-name", "original001", "--devices", ",".join(map(str, range(16))),
            "--master-addr", "10.0.0.1", "--master-port", "29501", "--shmem-port", "29502", "--plan"]
    assert runner.main(argv) == 0
    assert len(json.loads(capsys.readouterr().out)["runs"]) == 4
    assert before_torch == {name for name in sys.modules if name == "torch" or name.startswith("torch.")}


def test_build_cli_delegates_pure_plan_and_execute(tmp_path, monkeypatch, capsys):
    calls = []
    builder = types.SimpleNamespace(make_plan=lambda group, **kw: dict(group=str(group), reference=runner.REFERENCE),
        execute=lambda plan: calls.append(plan) or Path(plan["group"]) / "build-receipt.json")
    monkeypatch.setattr(runner, "_load_builder", lambda: builder)
    original_group_path = runner.group_path
    monkeypatch.setattr(runner, "group_path", lambda value: original_group_path(value, root=tmp_path))
    assert runner.main(["build", "--run-name", "new", "--plan"]) == 0
    assert json.loads(capsys.readouterr().out)["reference"] == runner.REFERENCE and calls == []
    assert runner.main(["build", "--run-name", "new"]) == 0 and len(calls) == 1


def test_sequential_success_emits_one_bounded_complete_report(tmp_path, monkeypatch):
    group = group_fixture(tmp_path)
    plan = runner.make_run_plan(args(), root=tmp_path)
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("DEEPEP_LL_BACKEND", "mte")
    monkeypatch.setenv("DEEPEP_LL_DISPATCH_IMPL", "cb59-small")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    monkeypatch.setenv("SHMEM_SOURCE_PATH", "/unrelated/sdk")
    def execute(argv, *, cwd, env, stdout, stderr, check):
        run = plan["runs"][len(calls)]
        calls.append(run["name"])
        assert argv == run["command"] and cwd == group / "source" and check is True
        assert all(key not in env for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        assert not any(key.startswith("DEEPEP_") for key in env)
        assert "PYTHONPATH" not in env and "SHMEM_SOURCE_PATH" not in env and env["PYTHONNOUSERSITE"] == "1"
        assert all(env[key] == ",".join(map(str, range(16))) for key in runner.VISIBILITY)
        for key in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "SHMEM_LOG_PATH"):
            assert Path(env[key]).is_relative_to(group) and Path(env[key]).is_dir()
        assert env["LD_LIBRARY_PATH"].startswith(str(group / "source/third-party/shmem/install/shmem/lib"))
        assert not any("--backend" in arg or "--use-1g" in arg for arg in argv)
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report_for(run, plan["bs"])))
        stdout.write("original ST completed\n")
    monkeypatch.setattr(runner.subprocess, "run", execute)
    report = runner.execute_run(plan)
    raw = report.read_bytes()
    assert len(raw) <= 60000 and report == group / "SEND_FIRST.txt"
    assert len(calls) == 4 and "Actual native AIV/G" in raw.decode()
    assert "HISTORICAL USER REFERENCE" in raw.decode()
    assert "original-submodule-metadata-only" in raw.decode()
    assert runner.analyze(group) == report  # exact repeat analysis is idempotent
    budget = runner._load_helper("transfer_budget")
    assert budget.budget_state(group)["issued"] == 1
    receipt_path = group / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["note"] = "local provenance note changed; raw results are unchanged"
    receipt_path.write_text(json.dumps(receipt))
    runner.analyze(group)
    assert budget.budget_state(group)["issued"] == 2
    assert len(list((group / "triage").glob("original_*/SEND_FIRST.txt"))) == 2
    with budget.transaction(group, "detail", [f"{number:064x}" for number in range(8)]):
        pass
    previous = report.read_bytes()
    receipt["note"] = "would require an eleventh transmission"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="transmission limit"):
        runner.analyze(group)
    assert report.read_bytes() == previous and budget.budget_state(group)["issued"] == 10
    with pytest.raises(ValueError, match="already has runs"):
        runner.execute_run(plan)


def test_failure_stops_next_runs_and_analysis_cannot_claim_completion(tmp_path, monkeypatch):
    group = group_fixture(tmp_path)
    plan = runner.make_run_plan(args(), root=tmp_path)
    calls = []
    def fail(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(9, argv)
    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    assert len(calls) == 1 and not (group / "SEND_FIRST.txt").exists()
    with pytest.raises(ValueError, match="incomplete"):
        runner.analyze(group)


@pytest.mark.parametrize("mutation", ["world", "on", "seed", "shape", "missing_rank", "samples", "mean", "case", "group_disagreement"])
def test_malformed_original_measurements_are_rejected(tmp_path, mutation):
    group_fixture(tmp_path)
    plan = runner.make_run_plan(args(), root=tmp_path)
    run = plan["runs"][0]
    report = report_for(run, plan["bs"])
    if mutation == "world": report["world_size"] = 8
    if mutation == "on": report["config"]["ll_stage_timer_capabilities"][0]["stage_timer"] = True
    if mutation == "seed": report["config"]["seed"] += 1
    if mutation == "shape": report["results"][0]["config"]["M"] = 64
    if mutation == "missing_rank": report["results"][0]["dispatch"]["ranks"].pop()
    if mutation == "samples": report["results"][0]["dispatch"]["ranks"][0]["kernel_us"].pop()
    if mutation == "mean": report["results"][0]["dispatch"]["max_rank_mean_kernel_us"] += 1
    if mutation == "case": report["results"].pop()
    if mutation == "group_disagreement": report["config"]["ll_stage_timer_capabilities"][1]["logical_group"] = 4
    with pytest.raises(ValueError):
        runner.validate_report(report, run, plan["bs"])


def test_non56_aiv_is_recorded_without_fabricating_comparability(tmp_path):
    group_fixture(tmp_path)
    plan = runner.make_run_plan(args(), root=tmp_path)
    run = plan["runs"][0]
    rows, capability = runner.validate_report(report_for(run, plan["bs"], active=64), run, plan["bs"])
    assert capability["active_aiv"] == 64
    assert all(row["active_aiv"] == 64 and row["core_group_matches_small"] is False for row in rows)


def test_route_hash_uses_only_original_pure_function_and_exact_seed(tmp_path):
    group = group_fixture(tmp_path)
    for seed in (0, 2):
        for batch in (1, 8, 32):
            rng = random.Random(seed)
            routes = [[rng.sample(range(384), 8) for _ in range(batch)] for _ in range(16)]
            expected = hashlib.sha256(json.dumps(routes, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
            assert runner.original_route_hash(group / "source", seed, batch) == expected
    assert runner.original_route_hash(group / "source", 0, 8) != runner.original_route_hash(group / "source", 2, 8)


def test_receipt_paths_cannot_escape_group(tmp_path):
    group = group_fixture(tmp_path)
    receipt_path = group / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source"] = "../../outside"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="relative"):
        runner.make_run_plan(args(), root=tmp_path)
