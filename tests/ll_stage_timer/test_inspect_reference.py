"""Read-only reference evidence, real sample math and bounded single-file export."""
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import types

import pytest

SPEC = importlib.util.spec_from_file_location("inspect_reference_tested", Path(__file__).with_name("inspect_reference.py"))
inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspection)


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def source_text(label):
    values = (5, 50, 30) if label == "cbd04eb" else (10, 100, 50)
    return f'''import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", default={values[0]})
    parser.add_argument("--num-iters", default={values[1]})
    parser.add_argument("--perf-tail", default={values[2]})
    return parser
def measure():
    return "{label}"
def summarize_kernel(samples):
    return max(samples)
raise AssertionError("INSPECTED SOURCE MUST NOT BE EXECUTED")
'''


def sample_report(package=None):
    ranks = [dict(rank=rank, kernel_us=[float(i + rank * 10) for i in range(1, 51)], samples=30,
                  mean_us=35.5 + rank * 10) for rank in range(2)]
    return dict(status="completed", world_size=2, package=str(package) if package else None,
        config=dict(warmup=5, num_iters=50, perf_tail=30, dispatch_only=True, do_cpu_sync=0,
                    dtype="fp8", fp8_scale_dtype="int32", num_tokens=8, seed=0,
                    delay_rows=118260, delay_cols=30720, delay_exp_before=0, delay_exp_after=1000),
        measurement="complete kernel task", results=[dict(case_id=0, route="normal", correctness="passed", backend="ll",
            config=dict(B=8, M=32, H=3584, num_topk=8, num_experts=384),
            dispatch=dict(kernel="dispatch_server_dedup_kernel", measurement="kernel_duration_us", perf_tail=30,
                          max_rank_mean_kernel_us=45.5, median_rank_mean_kernel_us=40.5,
                          min_rank_mean_kernel_us=35.5, slowest_rank=1, ranks=ranks))])


def mock_git(monkeypatch, blobs=None):
    calls = []
    blobs = blobs or {}
    def run(command, *, stdout, stderr, env, check, timeout):
        calls.append(command)
        assert command[:5] == ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C"]
        assert env["GIT_OPTIONAL_LOCKS"] == "0" and env["GIT_TERMINAL_PROMPT"] == "0"
        action = command[6:]
        assert action[0] in {"show", "diff", "rev-parse", "status", "symbolic-ref", "submodule"}
        if action[0] == "show":
            data = blobs.get(action[1])
            return types.SimpleNamespace(returncode=0 if data is not None else 128, stdout=data or b"", stderr=b"")
        if action[0] == "diff":
            assert "--no-ext-diff" in action and "--no-textconv" in action
            return types.SimpleNamespace(returncode=0, stdout=b"diff --git a/csrc/foo.cpp b/csrc/foo.cpp\n", stderr=b"")
        if action[0] == "status":
            assert "--untracked-files=no" in action
            output = b" M csrc/foo.cpp\n"
        elif action[0] == "rev-parse":
            output = str(command[5]).encode() + b"\n" if action[-1] == "--show-toplevel" else b"a" * 40 + b"\n"
        elif action[0] == "symbolic-ref":
            output = b"arbitrary-reference-branch\n"
        else:
            output = b" 123456 third-party/shmem\n"
        return types.SimpleNamespace(returncode=0, stdout=output, stderr=b"")
    monkeypatch.setattr(inspection.subprocess, "run", run)
    return calls


def fixture(tmp_path):
    controller = tmp_path / "current project"
    reference = tmp_path / "old reference with spaces"
    controller.mkdir()
    reference.mkdir()
    write(reference, inspection.ST, source_text("cbd04eb"))
    write(reference, "csrc/foo.cpp", "int x=1;\n")
    write(reference, "deep_ep/buffers/elastic.py", "raise RuntimeError('not importable')\n")
    write(reference, ".env", "UNTRACKED_SECRET=not-a-diff\n")
    package = write(reference, "venv/lib/python3.11/site-packages/deep_ep/__init__.py", "raise RuntimeError('do not import package')\n")
    write(package.parent, "_build_info.json", json.dumps(dict(source_revision="old", arch="Ascend950",
        build_type="Release", cann=dict(version="9.2.0", timestamp="stamp"),
        build_environment={"PRIVATE_TOKEN": "must-not-leak"})))
    library = write(package.parent, "libdeepep_kernels.so", "binary-like kernel fixture")
    write(package.parent, "_C.fake.so", "binary-like native fixture")
    results = reference / "results/cbd04eb/existing run"
    report = sample_report(package)
    write(results, "summary.json", json.dumps(report))
    write(results, "loaded_rank_000.json", json.dumps(dict(rank=0, pid=100, libraries={str(library): hashlib.sha256(library.read_bytes()).hexdigest(),
        "/unrelated/path/not_a_library": "never inspect", "/original/libshmem.so": "recorded-only-sdk-hash"})))
    write(reference, "build/temp.linux/fingerprint/device/CMakeCache.txt", "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CXX_COMPILER:FILEPATH=/path/to/bisheng\nSECRET_ENV:STRING=must-not-leak\n")
    write(reference, "build/temp.linux/fingerprint/device/flags.make", "CXX_FLAGS = -O2 --npu-arch=dav-3510 -DAPI_TOKEN='hidden phrase'\n")
    write(reference, "third-party/shmem/build/CMakeCache.txt", "CMAKE_CXX_COMPILER:FILEPATH=/sdk/compiler/bisheng\nSOC_TYPE:STRING=Ascend950\n")
    argv = ["--reference-root", str(reference), "--results", "results/cbd04eb/existing run", "--run-name", "inspect001"]
    return controller, reference, results, argv


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_actual_tail_math_and_blocks_without_assuming_n100():
    report = sample_report()
    observed = inspection.analyze_report(report, Path("summary.json"))
    case = observed["cases"][0]
    assert case["recorded_tail"] == 30
    assert case["recorded_recomputed"]["max_rank_mean_us"] == 45.5
    assert case["recorded_recomputed"]["min_rank_mean_us"] == 35.5
    assert case["recorded_recomputed"]["median_rank_mean_us"] == 40.5
    assert case["recorded_recomputed"]["rank0_mean_us"] == 35.5
    assert case["windows"]["10"]["max_rank_mean_us"] == 55.5
    assert case["windows"]["all"]["max_rank_mean_us"] == 35.5
    assert case["windows"]["50"] == case["windows"]["all"]
    assert case["windows"]["100"]["status"] == "unavailable"
    assert case["block_means"][0]["blocks"][0] == dict(first=1, last=10, mean_us=5.5)
    assert case["block_means"][0]["blocks"][-1] == dict(first=41, last=50, mean_us=45.5)
    assert case["reported_minus_recomputed_us"] == 0


def test_full_read_only_inspection_preserves_inputs_and_executes_only_git(tmp_path, monkeypatch):
    controller, reference, results, argv = fixture(tmp_path)
    blobs = {f"{revision}:{inspection.ST}": source_text(label).encode() for label, revision in inspection.REFS.items()}
    calls = mock_git(monkeypatch, blobs)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    before = snapshot(reference)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu", "deep_ep"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    plan = inspection.make_plan(inspection.parser().parse_args(argv), root=controller)
    output = inspection.inspect(plan)
    assert snapshot(reference) == before
    assert output.is_relative_to(controller) and output.name == "SEND_FIRST.txt"
    content = output.read_text(encoding="utf-8")
    assert len(content.encode()) + content.count("\n") <= 60000
    assert "Working measure function matches: cbd04eb" in content
    assert "No per-sample Exp/scalar-all-reduce protocol" in content
    assert "must-not-leak" not in content and "hidden phrase" not in content
    assert "actual world=2" in content and '"num_iters":50' in content
    evidence = json.loads((output.parent / "evidence.json").read_text())
    assert evidence["source"]["core_byte_aggregate"]["complete"] is True
    assert evidence["installed"][0]["libraries"][1]["same_bytes_as_recorded_path"] is True
    assert all(command[0] == "git" for command in calls)
    assert (output.parent / "working_vs_cbd04eb.diff").exists()
    ledger = json.loads((output.parent / "triage/send_budget.json").read_text())
    assert len(ledger["artifacts"]) == 1 and ledger["limit"] == 10
    with pytest.raises(ValueError, match="already exists"):
        inspection.inspect(plan)


def test_missing_or_unknown_evidence_is_reported_without_inventing_defaults(tmp_path, monkeypatch):
    controller = tmp_path / "current"
    reference = tmp_path / "reference"
    controller.mkdir(); reference.mkdir()
    write(reference, inspection.ST, "def measure():\n    return 'unrecognized'\n")
    mock_git(monkeypatch)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    arguments = inspection.parser().parse_args(["--reference-root", str(reference), "--results", "missing", "--run-name", "unknown"])
    output = inspection.inspect(inspection.make_plan(arguments, root=controller))
    text = output.read_text()
    assert "unknown; inspected source is not executed" in text
    assert "no matching result evidence found" in text
    evidence = json.loads((output.parent / "evidence.json").read_text())
    assert not evidence["source"]["core_byte_aggregate"]["complete"]
    assert evidence["results"]["reports"] == []


@pytest.mark.parametrize("selection", ["outside", "input", "results", "source"])
def test_output_escape_or_input_overlap_is_rejected(tmp_path, selection):
    controller, reference, results, argv = fixture(tmp_path)
    if selection == "outside": target = tmp_path / "elsewhere"
    elif selection == "input": target = reference
    elif selection == "results": target = results
    else: target = controller / "csrc"
    args = inspection.parser().parse_args([*argv, "--output-root", str(target)])
    with pytest.raises(ValueError):
        inspection.make_plan(args, root=controller)
    assert not (target / "inspect001").exists()


def test_relative_results_single_file_and_nonexistent_installed_package(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    package = root / "missing/deep_ep/__init__.py"
    path = write(root, "reports/full-report.json", json.dumps(sample_report(package)))
    reports = inspection.collect_results(path)
    packages = inspection.collect_installed(reports)
    assert len(reports["reports"]) == 1
    assert packages[0]["build_info_unavailable"] == "missing"
    assert packages[0]["libraries"] == []


def test_samples_json_can_recover_stripped_case_results(tmp_path):
    report = sample_report()
    for row in report["results"][0]["dispatch"]["ranks"]:
        del row["kernel_us"]
    path = write(tmp_path, "summary.json", json.dumps(report))
    for rank in range(2):
        write(tmp_path, f"case_00000/normal/dispatch/rank_{rank:03d}/samples.json",
              json.dumps(dict(kernel="dispatch_server_dedup_kernel", kernel_us=[3.0 + rank] * 50)))
    evidence = inspection.collect_results(tmp_path)
    stats = evidence["reports"][0]["cases"][0]["windows"]["30"]
    assert stats["max_rank_mean_us"] == 4.0


def test_ast_comparison_ignores_line_numbers_but_does_not_execute_source():
    first = inspection.ast_protocol(source_text("cbd04eb"))
    second = inspection.ast_protocol("\n\n# shifted lines\n" + source_text("cbd04eb"))
    assert first == second
    assert first["parser_defaults"] == {"warmup": 5, "num_iters": 50, "perf_tail": 30}


def test_render_is_bounded_even_with_many_existing_cases(tmp_path, monkeypatch):
    controller, reference, results, argv = fixture(tmp_path)
    mock_git(monkeypatch)
    source = inspection.collect_source(reference, controller)
    report = inspection.analyze_report(sample_report(), results / "summary.json")
    report["cases"] *= 100
    evidence = dict(reference=str(reference), result_path=str(results), source=source,
        results=dict(reports=[report], loaded_libraries=[], issues=[]), installed=[],
        build=dict(files=[], issues=[]), cann={})
    output = inspection.render(evidence)
    assert len(output.encode()) + output.count("\n") <= 60000
    assert "REPORT SIZE LIMIT" in output


def test_known_package_names_only_and_build_metadata_does_not_dump_environment(tmp_path):
    write(tmp_path, "build/variant/device/CMakeCache.txt", "CMAKE_CXX_COMPILER:FILEPATH=/bin/bisheng\nPRIVATE_TOKEN:STRING=secret-value\n")
    write(tmp_path, "build/variant/device/compile_commands.json", json.dumps([
        dict(command="/bin/bisheng -O2 -DAPI_TOKEN='secret phrase' --npu-arch=dav-3510 code.cpp", file="code.cpp")]))
    captured = inspection.collect_build(tmp_path)
    serialized = json.dumps(captured)
    assert "secret phrase" not in serialized and "secret-value" not in serialized
    assert "-O2" in serialized and "/bin/bisheng" in serialized
    rejected = inspection.collect_installed(dict(reports=[dict(package="/etc/passwd")], loaded_libraries=[]))
    assert rejected[0]["status"].startswith("not inspected")


def test_referenced_cann_files_are_hashed_without_running_tools(tmp_path, monkeypatch):
    cann = tmp_path / "CANN installation"
    compiler = write(cann, "tools/bisheng_compiler/bin/bisheng", "never execute this fake binary")
    write(cann, "compiler/version.info", "Version=9.2.0\ntimestamp=2026-test\nPRIVATE_TOKEN=not copied\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("tool must not run"))
    evidence = inspection.collect_cann([], dict(files=[]), environment={"ASCEND_HOME_PATH": str(cann)})
    assert evidence["compiler_files"][0]["sha256"] == hashlib.sha256(compiler.read_bytes()).hexdigest()
    assert evidence["version_files"][0]["values"] == {"Version": "9.2.0", "timestamp": "2026-test"}


def test_internal_source_link_is_not_followed_for_hashes_or_key_file_reads(tmp_path, monkeypatch):
    controller = tmp_path / "current"
    reference = tmp_path / "reference"
    controller.mkdir(); reference.mkdir()
    link = reference / "csrc/kernels"
    write(reference, "csrc/kernels/dispatch/low_latency.cpp", "outside-like content must not be read")
    write(reference, "deep_ep/__init__.py", "x=1")
    real_lstat = Path.lstat
    def lstat(path):
        if path == link:
            return types.SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return real_lstat(path)
    monkeypatch.setattr(Path, "lstat", lstat)
    mock_git(monkeypatch)
    source = inspection.collect_source(reference, controller)
    assert not source["core_byte_aggregate"]["complete"]
    assert all(not item["path"].startswith("csrc/kernels") for item in source["core_files"])
    entry = next(item for item in source["key_files"] if item["path"] == "csrc/kernels/dispatch/low_latency.cpp")
    assert entry["status"] == "symlink/junction source path skipped"


@pytest.mark.parametrize("directory", ["build", "benchmarks", "experiments"])
def test_output_cannot_enter_cleaned_or_fingerprinted_directories(tmp_path, directory):
    controller, reference, results, argv = fixture(tmp_path)
    args = inspection.parser().parse_args([*argv, "--output-root", directory])
    with pytest.raises(ValueError, match="source/dependency"):
        inspection.make_plan(args, root=controller)


def test_parent_repository_metadata_is_not_attributed_to_nonroot_input(tmp_path, monkeypatch):
    controller, reference, _, _ = fixture(tmp_path)
    calls = []
    def git_read(repo, *arguments):
        calls.append((repo, arguments))
        if repo == reference and arguments == ("rev-parse", "--show-toplevel"):
            return str(reference.parent).encode(), None
        if arguments[0] == "show":
            return None, "missing reference"
        pytest.fail(f"Unexpected parent-repository metadata read: {repo} {arguments}")
    monkeypatch.setattr(inspection, "git_read", git_read)
    source = inspection.collect_source(reference, controller)
    assert source["git"]["head"] is None and source["git"]["is_repository_root"] is False
    assert "parent repository metadata was not used" in source["git"]["head_unavailable"]


def test_existing_csv_task_start_intervals_are_bounded_and_not_duration_differences(tmp_path):
    valid = "Op Name,Start Time(us),Duration(us)\n"
    valid += "other_kernel,1,1\n"
    valid += "\n".join(f"dispatch_server_dedup_kernel,{start},10" for start in (0, 1000, 1020, 1041)) + "\n"
    chosen = write(tmp_path, "case_00000/normal/dispatch/rank_000/profile/kernel_details.csv", valid)
    unknown = write(tmp_path, "case_00001/normal/dispatch/rank_000/profile/kernel_details.csv",
                    "Name,Duration(us)\ndispatch_server_dedup_kernel,12\ndispatch_server_dedup_kernel,13\n")
    for rank in range(1, 18):
        write(tmp_path, f"case_00000/normal/dispatch/rank_{rank:03d}/profile/kernel_details.csv", valid)
    evidence = inspection.collect_task_intervals(tmp_path)
    assert evidence["scanned"] == 16 and len(evidence["files"]) == 16
    assert [entry["path"] for entry in evidence["files"][:2]] == [str(chosen), str(unknown)]
    task = evidence["files"][0]["tasks"][0]
    assert task["N"] == 4 and task["starts_complete"] is True
    assert task["first_to_second_start_us"] == 1000
    assert task["first_end_to_second_start_us"] == 990
    assert task["later_adjacent_start_gaps_us"] == dict(N=2, min=20, p50=20.5, max=21)
    assert evidence["files"][1]["tasks"][0]["starts_complete"] is False
    assert "intervals_unavailable" in evidence["files"][1]["tasks"][0]


def test_render_long_csv_header_keeps_each_task_interval_and_missing_values():
    header = ["Name", "Start Time(us)", "Duration(us)"] + [f"Long profiler metadata column {index}" for index in range(46)]
    record = dict(path="/existing results/rank_000/kernel_details.csv", header=header, tasks=[
        dict(full_task_name="dispatch_server_dedup_kernel()", N=50, starts_complete=True, durations_complete=True,
             first_to_second_start_us=123456.25, first_end_to_second_start_us=123446.25,
             later_adjacent_start_gaps_us=dict(N=48, min=20, p50=20.5, max=22)),
        dict(kernel="combine_server_dedup_kernel", N=1, starts_complete=False, durations_complete=True,
             intervals_unavailable="missing start column")])
    text = "\n".join(inspection._task_interval_lines(record))
    assert text.splitlines()[0] == "CSV " + record["path"]
    assert 'header_columns=49; task columns=["Name","Start Time(us)","Duration(us)"]' in text
    assert "Long profiler metadata" not in text
    for value in ("dispatch_server_dedup_kernel()", "N=50; starts_complete=True; durations_complete=True",
                  "first_to_second_start_us=123456.25", "first_end_to_second_start_us=123446.25",
                  "N=48; min=20; p50=20.5; max=22", "combine_server_dedup_kernel",
                  "first_to_second_start_us=unknown", "unavailable=missing start column"):
        assert value in text


def test_render_build_long_includes_do_not_hide_last_optimization_or_core_defines(tmp_path):
    includes = " ".join(f"-I/path/to/a/very/long/ordinary/include/directory/{index}" for index in range(80))
    path = write(tmp_path, "build/variant/device/CMakeFiles/deepep_kernels.dir/flags.make",
                 "CXX_DEFINES = -DCATLASS_ARCH=3510 -DDEEPEP_LL_SEND_CORES=40 -DDEEPEP_LL_RECV_CORES=16\n" +
                 f"CXX_INCLUDES = {includes} -I/actual/sdk/shmem/install/include\n" +
                 "C_FLAGS = -fPIC\nCXX_FLAGS = -O3 -DNDEBUG --npu-arch=dav-3510 -O2\n")
    records = inspection.collect_build(tmp_path)["files"]
    assert len(records) == 1
    text = "\n".join(inspection._build_lines(records[0]))
    assert text.splitlines()[0] == "BUILD " + str(path)
    assert "optimization compile flags[4]: -O3 -> -O2" in text
    assert "-DDEEPEP_LL_SEND_CORES=40" in text and "-DDEEPEP_LL_RECV_CORES=16" in text
    assert "SHMEM includes (1 unique)=[\"-I/actual/sdk/shmem/install/include\"]" in text
    assert "other includes (80 unique; first2)=" in text
    assert len(text) < 1600


def test_render_cache_and_link_optimization_cannot_be_mistaken_for_compile_flags():
    cache = dict(path="/build/device/CMakeCache.txt", commands=[], values={
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_CXX_COMPILER": "/toolchain/bisheng", "SHMEM_ROOT": "/sdk/shmem",
        "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG", "CMAKE_CXX_FLAGS_RELWITHDEBINFO": "-O2 -g"})
    linked = dict(path="/build/device/link.txt", values={}, commands=[dict(compiler="/toolchain/bisheng", flags=["-O3"])])
    cache_text = "\n".join(inspection._build_lines(cache))
    link_text = "\n".join(inspection._build_lines(linked))
    assert 'CMAKE_BUILD_TYPE="Release"' in cache_text and 'SHMEM_ROOT="/sdk/shmem"' in cache_text
    assert "optimization cache CMAKE_CXX_FLAGS_RELEASE: -O3" in cache_text
    assert "optimization cache CMAKE_CXX_FLAGS_RELWITHDEBINFO: -O2" in cache_text
    assert "optimization link only[1]: -O3" in link_text
    assert "compile flags" not in cache_text + link_text


def test_repeated_optimization_flags_keep_order_in_new_evidence_and_render(tmp_path):
    write(tmp_path, "build/device/compile_commands.json", json.dumps([
        dict(command="/bin/bisheng -O2 -O3 -O2 source.cpp", file="source.cpp")]))
    record = inspection.collect_build(tmp_path)["files"][0]
    command = record["commands"][0]
    assert command["flags"] == ["-O2", "-O3", "-O2"]
    assert command["order_preserved"] is True
    assert "optimization compile flags[1]: -O2 -> -O3 -> -O2; order_preserved=True" in "\n".join(inspection._build_lines(record))
    del command["order_preserved"]
    assert "order_preserved=unknown" in "\n".join(inspection._build_lines(record))


def test_reexport_reads_only_saved_evidence_and_shares_original_budget(tmp_path, monkeypatch):
    controller, reference, _, argv = fixture(tmp_path)
    mock_git(monkeypatch)
    original = inspection.inspect(inspection.make_plan(inspection.parser().parse_args(argv), root=controller))
    evidence_path = original.parent / "evidence.json"
    saved = json.loads(evidence_path.read_text())
    saved["cann"]["note"] = "Offline rendering sentinel"
    evidence_path.write_text(json.dumps(saved), encoding="utf-8")
    evidence_bytes = evidence_path.read_bytes()
    source_before = snapshot(reference)
    for name in ("collect_source", "collect_results", "collect_installed", "collect_build", "collect_cann", "collect_task_intervals", "git_read", "file_hash"):
        monkeypatch.setattr(inspection, name, lambda *a, **k: pytest.fail("reexport must not collect or read target files"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("reexport must not execute a subprocess"))
    actual_json = inspection._json
    read_json = []
    def read_saved(path):
        assert path == evidence_path
        read_json.append(path)
        return actual_json(path)
    monkeypatch.setattr(inspection, "_json", read_saved)
    for index in (1, 2):
        args = inspection.parser().parse_args(["--reexport-evidence", str(evidence_path.relative_to(controller)), "--run-name", f"offline{index}"])
        plan = inspection.make_plan(args, root=controller)
        output = inspection.reexport(plan)
        assert "Offline rendering sentinel" in output.read_text()
        assert "OFFLINE SNAPSHOT REEXPORT" in output.read_text()
        assert "Reference files were not reread and the current deployment was not checked." in output.read_text()
        assert output.parent != original.parent and not (output.parent / "evidence.json").exists()
        assert not (output.parent / "triage/send_budget.json").exists()
        origin = json.loads((output.parent / "reexport_origin.json").read_text())
        assert origin["evidence"] == str(evidence_path) and origin["budget_group"] == str(original.parent)
        assert len(output.read_bytes()) + output.read_text().count("\n") <= 60000
    assert read_json == [evidence_path, evidence_path]
    assert evidence_path.read_bytes() == evidence_bytes and snapshot(reference) == source_before
    ledger = json.loads((original.parent / "triage/send_budget.json").read_text())
    assert len(ledger["artifacts"]) == 2  # identical reexport has no second charge


def test_reexport_refuses_new_report_when_original_budget_is_full(tmp_path, monkeypatch):
    controller, _, _, argv = fixture(tmp_path)
    mock_git(monkeypatch)
    original = inspection.inspect(inspection.make_plan(inspection.parser().parse_args(argv), root=controller))
    evidence_path = original.parent / "evidence.json"
    saved = json.loads(evidence_path.read_text())
    saved["cann"]["note"] = "Changed report needs one more slot"
    evidence_path.write_text(json.dumps(saved), encoding="utf-8")
    budget = inspection._transfer_budget()
    with budget.transaction(original.parent, "detail", [hashlib.sha256(f"detail{i}".encode()).hexdigest() for i in range(9)]):
        pass
    ledger_path = original.parent / "triage/send_budget.json"
    before = ledger_path.read_bytes()
    args = inspection.parser().parse_args(["--reexport-evidence", str(evidence_path), "--run-name", "budget_full"])
    plan = inspection.make_plan(args, root=controller)
    with pytest.raises(ValueError, match="transmission limit is 10"):
        inspection.reexport(plan)
    assert not Path(plan["group"]).exists() and ledger_path.read_bytes() == before


@pytest.mark.parametrize("case", ["results", "outside", "source", "overlap", "linked_budget", "invalid_format"])
def test_reexport_rejects_unsafe_or_mixed_input_without_output(tmp_path, monkeypatch, case):
    controller, reference, _, argv = fixture(tmp_path)
    evidence_path = write(controller, "diagnostic-runs/saved/evidence.json", "{}")
    extra = []
    if case == "results":
        extra = ["--results", "anything"]
    elif case == "outside":
        evidence_path = write(reference, "evidence.json", "{}")
    elif case == "source":
        evidence_path = write(controller, "tests/evidence.json", "{}")
    elif case == "overlap":
        extra = ["--output-root", str(evidence_path.parent)]
    elif case == "linked_budget":
        real_link = inspection._link
        monkeypatch.setattr(inspection, "_link", lambda path: path == evidence_path.parent / "triage" or real_link(path))
    args = inspection.parser().parse_args(["--reexport-evidence", str(evidence_path), "--run-name", "refused", *extra])
    with pytest.raises(ValueError):
        plan = inspection.make_plan(args, root=controller)
        inspection.reexport(plan)
    assert not list(controller.rglob("SEND_FIRST.txt"))


def test_reexport_and_reference_are_mutually_exclusive_and_results_is_required():
    with pytest.raises(SystemExit):
        inspection.parser().parse_args(["--reexport-evidence", "evidence.json", "--reference-root", "reference", "--run-name", "unused"])
    args = inspection.parser().parse_args(["--reference-root", "reference", "--run-name", "unused"])
    with pytest.raises(ValueError, match="--results is required"):
        inspection.make_plan(args)


def test_root_summary_is_prioritized_before_many_sample_files(tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    samples = [tmp_path / f"case_{i:05d}/normal/dispatch/rank_000/samples.json" for i in range(600)]
    monkeypatch.setattr(inspection, "_walk", lambda *a, **k: (samples + [summary], []))
    read = []
    def read_json(path):
        read.append(path)
        return (sample_report(), None) if path == summary else (dict(kernel_us=[1.0] * 50), None)
    monkeypatch.setattr(inspection, "_json", read_json)
    evidence = inspection.collect_results(tmp_path)
    assert read[0] == summary and len(read) == 512
    assert len(evidence["reports"]) == 1
    assert evidence["reports"][0]["cases"][0]["windows"]["30"]["max_rank_mean_us"] == 45.5


def test_wheel_sdk_notes_are_recorded_without_any_sdk_admission_validation():
    value = inspection._filter_build_info(dict(shmem=dict(revision="historical-sdk", build={
        "cann_version": "9.2.0", "cann_timestamp": "stamp", "target": "Ascend950", "rdma_backend": "HNS_1825",
        "hns_1825_pi_on_chip": 17, "source_patches": [{"kind": "local-worktree", "sha256": "a" * 64, "diff": "not-copied"}],
        "password": "secret-value"}, library_hashes={"libshmem.so": "not-copied"})))
    notes = value["shmem"]
    assert notes["informational_only"] is True and notes["revision"] == "historical-sdk"
    assert notes["build"]["hns_1825_pi_on_chip"] == 17  # no field-range gate
    assert notes["build"]["source_patches"] == [{"kind": "local-worktree", "sha256": "a" * 64}]
    assert "secret-value" not in json.dumps(value) and "not-copied" not in json.dumps(value)


def test_typical_fourteen_cases_keep_all_evidence_categories_under_transfer_limit(tmp_path, monkeypatch):
    controller, reference, results, _ = fixture(tmp_path)
    mock_git(monkeypatch)
    source = inspection.collect_source(reference, controller)
    full = sample_report()
    full["world_size"] = 16
    template = full["results"][0]
    full["results"] = []
    for index, batch in enumerate((1, 2, 4, 8, 16, 24, 32)):
        row = copy.deepcopy(template)
        row["case_id"] = index
        row["config"]["B"] = batch
        row["dispatch"]["ranks"] = [dict(rank=rank, kernel_us=[20 + batch / 10 + rank / 100] * 50, samples=30)
                                      for rank in range(16)]
        row["combine"] = copy.deepcopy(row["dispatch"])
        row["combine"]["kernel"] = "combine_server_dedup_kernel"
        full["results"].append(row)
    report = inspection.analyze_report(full, results / "summary.json")
    build_files = [dict(path=str(reference / f"build/fingerprint_{i}/device/CMakeCache.txt"),
                        values={"CMAKE_CXX_FLAGS": "-O2 " + "-DVERY_LONG_DEFINITION=1 " * 60, "DEEPEP_VARIANT": i}, commands=[])
                   for i in range(40)]
    build_files += [dict(path=str(reference / "build/host/CMakeCache.txt"), values={"CMAKE_BUILD_TYPE": "Release"}, commands=[]),
                    dict(path=str(reference / "third-party/shmem/build/CMakeCache.txt"), values={"SOC_TYPE": "Ascend950"}, commands=[])]
    build_files.append(dict(path=str(reference / "build/variant/device/CMakeFiles/deepep_kernels.dir/flags.make"), values={}, commands=[
        dict(compiler=None, flags=["-DDEEPEP_LL_SEND_CORES=40", "-DDEEPEP_LL_RECV_CORES=16"]),
        dict(compiler=None, flags=[f"-I/long/ordinary/include/directory/{i}" for i in range(80)] + ["-I/actual/sdk/shmem/include"]),
        dict(compiler=None, flags=["-O3", "-DNDEBUG", "-O2"])]))
    header = ["Name", "Start Time(us)", "Duration(us)"] + [f"Long profiler metadata column {i}" for i in range(46)]
    csvs = [dict(path=str(results / f"case_{i:05d}/rank_000/kernel_details.csv"), header=header,
                 tasks=[dict(kernel="dispatch_server_dedup_kernel", N=50, starts_complete=True,
                    first_to_second_start_us=1000, later_adjacent_start_gaps_us=dict(N=48, min=22, p50=23, max=24))]) for i in range(16)]
    evidence = dict(reference=str(reference), result_path=str(results), source=source, offline_snapshot=True,
        results=dict(reports=[report], auxiliary=[], loaded_libraries=[], issues=[]),
        installed=[dict(reported_path="/old/deep_ep/__init__.py", build_info=dict(arch="Ascend950"), libraries=[])],
        build=dict(files=build_files, issues=[]), cann=dict(note="CURRENT_CANN_SENTINEL"),
        task_intervals=dict(files=csvs, total_matching_csv=112, scanned=16, note="selected rank0 intervals"))
    output = inspection.render(evidence)
    assert len(output.encode()) + output.count("\n") < 60000
    assert "REPORT SIZE LIMIT" not in output
    for required in ("B32", "CURRENT INSTALLED FILES", "CURRENT BUILD METADATA", "CURRENT_CANN_SENTINEL",
                     "LIMITS / MISSING EVIDENCE", "EXISTING TASK START INTERVALS", "displayed=12",
                     "optimization compile flags[3]: -O3 -> -O2", "-I/actual/sdk/shmem/include"):
        assert required in output
    assert output.count("header_columns=49") == 16
    assert output.count("first_to_second_start_us=1000") == 16
    assert output.count("later_adjacent_start_gaps_us: N=48; min=22; p50=23; max=24") == 16
    assert "Long profiler metadata column" not in output
    displayed, unique = inspection._build_display(build_files)
    assert len(displayed) == 12 and unique == 43
    assert any("third-party" in item["path"] for item in displayed)
    assert any("build/host" in Path(item["path"]).as_posix() for item in displayed)
