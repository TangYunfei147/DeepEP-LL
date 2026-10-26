"""CPU integration coverage for explicit whole-block no-timer A/B resume."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "no_timer_resume_fixtures", Path(__file__).with_name("test_no_timer_ab.py"))
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)
runner = fixtures.runner


def scenario(tmp_path, monkeypatch, *, stage="all", fail_at=(), interrupt_at=()):
    group, receipt = fixtures.build_fixture(tmp_path)
    plan = runner.make_run_plan(fixtures.args(stage=stage), root=tmp_path)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner, "_proof", lambda group, variant, controller: fixtures.proof(variant))
    environment = runner._environment
    monkeypatch.setattr(runner, "_environment", lambda *a, **k: environment(*a, **dict(k, inherited={})))
    calls, measured = [], []
    failures, interruptions = set(fail_at), set(interrupt_at)
    by_name = {run["name"]: run for run in plan["runs"]}

    def command(argv, cwd, env, log):
        log = Path(log)
        calls.append(dict(argv=list(argv), log=log))
        # Preserve the real command helper's exclusive-log semantics. A retry
        # that accidentally points at an old log must fail in this fixture too.
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as stream:
            stream.write(("mock invocation " + str(len(calls))).encode())
        if "_verify-host" in argv:
            return
        name = argv[argv.index("--run-name") + 1]
        canonical = by_name[name]
        report = Path(argv[argv.index("--output") + 1])
        output = Path(argv[argv.index("--output-dir") + 1])
        assert report == output / name / "full-report.json"
        assert Path(cwd) == runner._layout(group, canonical["variant"])["source"]
        # The retry may relocate output only: all measurement options, source,
        # interpreter, seed, and ordering must remain the original values.
        restored = list(argv)
        for option in ("--output", "--output-dir"):
            restored[restored.index(option) + 1] = canonical["command"][canonical["command"].index(option) + 1]
        assert restored == canonical["command"]
        measured.append(dict(name=name, report=report, log=log))
        if len(measured) in interruptions:
            raise KeyboardInterrupt()
        if len(measured) in failures:
            raise subprocess.CalledProcessError(8, argv)
        fixtures.write(report, json.dumps(fixtures.report_for(canonical, receipt)))

    monkeypatch.setattr(runner, "_command", command)
    return group, receipt, plan, calls, measured


def state_path(group, plan):
    return group / plan["status_file"]


def read_state(group, plan):
    return json.loads(state_path(group, plan).read_text())


def legacy_status(group, plan):
    """Model the exact pre-resume v1 status records written on the target."""
    state = read_state(group, plan)
    state["runs"] = [dict(name=item["name"], status=item["status"]) for item in state["runs"]]
    fixtures.write(state_path(group, plan), json.dumps(state))
    return state_path(group, plan).read_bytes()


def retained_files(group):
    return {path: path.read_bytes() for directory in ("logs", "runs", "run-plans")
            for path in (group / directory).rglob("*") if path.is_file()}


def assert_retained(before):
    assert all(path.is_file() and path.read_bytes() == data for path, data in before.items())


def test_legacy_first_eight_resume_only_hotspot_block_without_overwriting(tmp_path, monkeypatch):
    group, _, plan, calls, measured = scenario(tmp_path, monkeypatch, fail_at={9})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    previous_status = legacy_status(group, plan)
    before = retained_files(group)
    old_call_count = len(calls)
    output = runner.execute_run(plan, resume=True)
    added = calls[old_call_count:]
    assert len(added) == 7
    assert all("_verify-host" in call["argv"] for call in added[:3])
    assert all("-resume001-" in call["log"].name for call in added[:3])
    assert [item["name"] for item in measured[9:]] == [run["name"] for run in plan["runs"][8:]]
    assert all(item["report"].parent.parent == group / "runs/resume001" for item in measured[9:])
    assert all(item["log"].parent == group / "logs/runs/resume001" for item in measured[9:])
    assert_retained(before)
    history = group / "run-history" / Path(plan["plan_file"]).stem / "resume001.json"
    snapshot = json.loads(history.read_text())
    assert snapshot["previous_state"] == json.loads(previous_status)
    assert snapshot["attempt"] == 1 and snapshot["retained_blocks"] == 2
    assert snapshot["rerun_names"] == [run["name"] for run in plan["runs"][8:]]
    state = read_state(group, plan)
    assert state["status"] == "completed"
    assert [item.get("attempt", 0) for item in state["runs"]] == [0] * 8 + [1] * 4
    assert output == group / "SEND_FIRST.txt"
    assert "resume001" in output.read_text()
    assert len(output.read_bytes()) <= runner.LIMIT
    assert not (group / ".run.lock").exists()
    assert runner.analyze(group) == output


def test_partial_block_retries_all_four_and_second_failure_gets_fresh_attempt(tmp_path, monkeypatch):
    group, _, plan, calls, measured = scenario(tmp_path, monkeypatch, stage="timer", fail_at={3, 5})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    legacy_status(group, plan)
    before = retained_files(group)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan, resume=True)
    assert [item["name"] for item in measured[3:]] == [run["name"] for run in plan["runs"][:2]]
    assert all(item["report"].parent.parent.name == "resume001" for item in measured[3:])
    after_first_retry = retained_files(group)
    runner.execute_run(plan, resume=True)
    assert [item["name"] for item in measured[5:]] == [run["name"] for run in plan["runs"]]
    assert all(item["report"].parent.parent.name == "resume002" for item in measured[5:])
    assert_retained(before)
    assert_retained(after_first_retry)
    history = group / "run-history" / Path(plan["plan_file"]).stem
    assert {path.name for path in history.glob("*.json")} == {"resume001.json", "resume002.json"}
    assert [item.get("attempt", 0) for item in read_state(group, plan)["runs"]] == [2] * 4
    assert len([call for call in calls if "_verify-host" in call["argv"]]) == 9


@pytest.mark.parametrize("corruption", ["sample", "cross_run_route"])
def test_corrupt_completed_block_refuses_resume_before_launch(tmp_path, monkeypatch, corruption):
    group, _, plan, calls, _ = scenario(tmp_path, monkeypatch, fail_at={9})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    legacy_status(group, plan)
    report_path = group / plan["runs"][0]["report"]
    report = json.loads(report_path.read_text())
    if corruption == "sample":
        report["results"][0]["dispatch"]["ranks"][0]["kernel_us"].pop()
    else:
        report["results"][0]["comparison"]["profile_route_sha256"] = "e" * 64
    fixtures.write(report_path, json.dumps(report))
    previous_count = len(calls)
    before = retained_files(group)
    with pytest.raises(ValueError):
        runner.execute_run(plan, resume=True)
    assert len(calls) == previous_count
    assert_retained(before)
    assert not (group / "SEND_FIRST.txt").exists()
    assert not (group / ".run.lock").exists()


def test_changed_valid_plan_refuses_resume_without_writing_history_or_launching(tmp_path, monkeypatch):
    group, _, plan, calls, _ = scenario(tmp_path, monkeypatch, fail_at={9})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    changed = runner.make_run_plan(fixtures.args("--master-port", "29503"), root=tmp_path)
    before = {path: path.read_bytes() for path in group.rglob("*") if path.is_file()}
    previous_count = len(calls)
    with pytest.raises(ValueError):
        runner.execute_run(changed, resume=True)
    assert len(calls) == previous_count
    assert_retained(before)
    assert not (group / "run-history").exists()
    assert not (group / ".run.lock").exists()


def test_completed_legacy_resume_is_analysis_only(tmp_path, monkeypatch):
    group, _, plan, calls, _ = scenario(tmp_path, monkeypatch, stage="timer")
    output = runner.execute_run(plan)
    legacy_status(group, plan)
    previous_calls = len(calls)
    before = retained_files(group)
    expected = output.read_bytes()
    assert runner.execute_run(plan, resume=True) == output
    assert len(calls) == previous_calls
    assert output.read_bytes() == expected
    assert_retained(before)
    assert not (group / "run-history").exists()
    assert not (group / ".run.lock").exists()


def test_keyboard_interrupt_records_interrupted_and_can_resume(tmp_path, monkeypatch):
    group, _, plan, _, measured = scenario(tmp_path, monkeypatch, stage="timer", interrupt_at={2})
    with pytest.raises(KeyboardInterrupt):
        runner.execute_run(plan)
    state = read_state(group, plan)
    assert state["status"] == "interrupted"
    assert state["runs"][-1]["status"] == "failed"
    assert not (group / ".run.lock").exists()
    before = retained_files(group)
    runner.execute_run(plan, resume=True)
    assert [item["name"] for item in measured[2:]] == [run["name"] for run in plan["runs"]]
    assert_retained(before)
    assert read_state(group, plan)["status"] == "completed"


def test_existing_group_lock_refuses_resume_without_removing_lock(tmp_path, monkeypatch):
    group, _, plan, calls, _ = scenario(tmp_path, monkeypatch, stage="timer", fail_at={2})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    lock = fixtures.write(group / ".run.lock", "other invocation owns this lock")
    before = retained_files(group)
    previous_count = len(calls)
    with pytest.raises((ValueError, OSError)):
        runner.execute_run(plan, resume=True)
    assert len(calls) == previous_count
    assert lock.read_text() == "other invocation owns this lock"
    assert_retained(before)


@pytest.mark.parametrize("status", ["running", "preflight"])
def test_live_or_unknown_prior_invocation_is_not_resumed(tmp_path, monkeypatch, status):
    group, _, plan, calls, _ = scenario(tmp_path, monkeypatch, stage="timer", fail_at={2})
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    state = read_state(group, plan)
    state["status"] = status
    fixtures.write(state_path(group, plan), json.dumps(state))
    previous_count = len(calls)
    with pytest.raises(ValueError):
        runner.execute_run(plan, resume=True)
    assert len(calls) == previous_count
    assert read_state(group, plan) == state
    assert not (group / ".run.lock").exists()


def test_failed_legacy_preflight_reuses_build_and_starts_all_runs(tmp_path, monkeypatch):
    group, _, plan, calls, measured = scenario(tmp_path, monkeypatch, stage="timer")
    command = runner._command
    def failed_preflight(argv, cwd, env, log):
        command(argv, cwd, env, log)
        raise subprocess.CalledProcessError(7, argv)
    monkeypatch.setattr(runner, "_command", failed_preflight)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(plan)
    assert measured == []
    legacy_status(group, plan)
    before = retained_files(group)
    monkeypatch.setattr(runner, "_command", command)
    runner.execute_run(plan, resume=True)
    assert [item["name"] for item in measured] == [run["name"] for run in plan["runs"]]
    assert all(item["report"].parent.parent.name == "resume001" for item in measured)
    assert len(calls) == 8
    assert_retained(before)


def test_attempt_mapping_preserves_canonical_command_and_rejects_bad_attempt(tmp_path):
    plan = runner.make_run_plan(fixtures.args(stage="timer"), root=tmp_path)
    canonical = plan["runs"][0]
    before = copy.deepcopy(canonical)
    assert runner._attempt_run(canonical, 0) == canonical
    retried = runner._attempt_run(canonical, 2)
    assert canonical == before
    assert retried["name"] == canonical["name"]
    assert retried["directory"] == "runs/resume002/" + canonical["name"]
    assert retried["report"] == retried["directory"] + "/full-report.json"
    group = Path(plan["group"])
    assert runner._run_log(group, canonical, 0) == group / "logs/runs" / (canonical["name"] + ".log")
    assert runner._run_log(group, canonical, 2) == group / "logs/runs/resume002" / (canonical["name"] + ".log")
    for bad in (-1, True, "2", 1.5):
        with pytest.raises(ValueError):
            runner._attempt_run(canonical, bad)


def test_four_seed_resume999_summary_remains_within_single_file_limit(tmp_path):
    _, receipt = fixtures.build_fixture(tmp_path)
    plan = runner.make_run_plan(fixtures.args("--seeds", "0", "2", "4", "6"), root=tmp_path)
    selected = dict(plan, runs=[runner._attempt_run(run, 999) for run in plan["runs"]],
                    resume_history=["run-history/all-seeds-0-2-4-6/resume999.json"])
    rows = [row for run in selected["runs"]
            for row in runner.validate_report(fixtures.report_for(run, receipt), run, receipt)]
    content, summary = runner.render_summary(receipt, [selected], rows)
    assert len(selected["runs"]) == 48 and len(summary) == 168
    assert all(run["report"].startswith("runs/resume999/") for run in selected["runs"])
    assert "runs/resume999/" in content and "optimization hotspot 6 32 combine" in content
    assert len(content.encode("utf-8")) + content.count("\n") <= runner.LIMIT


@pytest.mark.parametrize("first_stage,second_stage", [("timer", "all"), ("all", "timer")])
def test_overlapping_saved_plan_is_rejected_even_before_run_directories_exist(
        tmp_path, monkeypatch, first_stage, second_stage):
    group, _, first, calls, measured = scenario(tmp_path, monkeypatch, stage=first_stage)
    command = runner._command
    def failed_preflight(argv, cwd, env, log):
        command(argv, cwd, env, log)
        raise subprocess.CalledProcessError(7, argv)
    monkeypatch.setattr(runner, "_command", failed_preflight)
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute_run(first)
    assert measured == []
    assert all(not (group / run["directory"]).exists() for run in first["runs"])
    before = retained_files(group)
    previous_count = len(calls)
    second = runner.make_run_plan(fixtures.args(stage=second_stage), root=tmp_path)
    with pytest.raises(ValueError):
        runner.execute_run(second)
    assert len(calls) == previous_count and measured == []
    assert not (group / second["plan_file"]).exists()
    assert not (group / second["status_file"]).exists()
    assert_retained(before)
    assert not (group / ".run.lock").exists()
