"""CPU behavior contracts for the fixed cb59 no-timer Python projection."""
from __future__ import annotations

import ast
import builtins
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASE = "439d459"
PROJECTION = runpy.run_path(str(ROOT / "scripts/ll_no_timer_python.py"))
TRANSFORM = PROJECTION["transform"]
ST_PATH = "tests/elastic/test_ep_low_latency.py"


@pytest.fixture(scope="module")
def sources():
    return {path: subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT).decode("utf-8")
            for path in PROJECTION["PATCH_PATHS"]}


def module_from(source, path, package=None):
    module = ModuleType("projected_" + Path(path).stem)
    module.__file__ = str(ROOT / path)
    if package:
        module.__package__ = package
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def st_module(sources, mode="clean-source"):
    return module_from(TRANSFORM(ST_PATH, sources[ST_PATH], mode), ST_PATH)


def artifact(mode):
    dispatch, combine = PROJECTION["MODES"][mode]
    return dict(ll_instrumentation_source=mode, dispatch_launch_bytes=dispatch, combine_launch_bytes=combine)


def native_identity(mode):
    result = artifact(mode)
    result["low_latency_instrumentation_source"] = result.pop("ll_instrumentation_source")
    return SimpleNamespace(runtime_protocol=1, **result)


@pytest.mark.parametrize("mode", ["normal-off", "clean-source"])
def test_projection_compiles_and_preserves_business_functions(sources, mode):
    for path, source in sources.items():
        ast.parse(TRANSFORM(path, source, mode), filename=path)
    before = {node.name: node for node in ast.parse(sources[ST_PATH]).body if isinstance(node, ast.FunctionDef)}
    after = {node.name: node for node in ast.parse(TRANSFORM(ST_PATH, sources[ST_PATH], mode)).body if isinstance(node, ast.FunctionDef)}
    preserved = ("make_routes", "profile_route_identity", "check_route_reuse", "expert_entries", "payload", "fp8_scales",
                 "expected_combine", "prepare_experts", "verify_dispatch", "control_device", "trace_call", "agree_check",
                 "completion_boundary", "record_case_progress", "sampling_prelude", "measure", "kernel_durations",
                 "sample_stats", "summarize_kernel", "case_result_summary")
    for name in preserved:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    if mode == "normal-off":
        assert ast.dump(before["run_case"]) == ast.dump(after["run_case"])
        assert ast.dump(before["sample_operation"]) == ast.dump(after["sample_operation"])
        assert TRANSFORM("deep_ep/buffers/elastic.py", sources["deep_ep/buffers/elastic.py"], mode) == sources["deep_ep/buffers/elastic.py"]
    else:
        # The sole run_case edit removes the capture alternative from its OFF guard.
        guard = next(node for node in ast.walk(before["run_case"])
                     if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                     and ast.unparse(node.test) == "not args.skip_perf_test or args.ll_stage_timer")
        guard.test = guard.test.values[0]
        assert ast.dump(before["run_case"]) == ast.dump(after["run_case"])
        clean = TRANSFORM(ST_PATH, sources[ST_PATH], mode)
        assert "ll_stage_timer" not in clean and "ll_expert_timer" not in clean
        assert "timer_layout" not in clean and "StageTrace" not in clean


def test_clean_package_has_no_import_of_deleted_modules(sources):
    deleted = {Path(path).stem for path in PROJECTION["DELETE_PATHS"]}
    assert len(deleted) == 4
    for path in (ROOT / "deep_ep").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative in PROJECTION["DELETE_PATHS"]:
            continue
        source = sources.get(relative, path.read_text(encoding="utf-8"))
        if relative in PROJECTION["PATCH_PATHS"]:
            source = TRANSFORM(relative, source, "clean-source")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or "").split(".")) & deleted, relative
            elif isinstance(node, ast.Import):
                assert not any(set(alias.name.split(".")) & deleted for alias in node.names), relative


@pytest.mark.parametrize("expected", ["normal-off", "clean-source"])
@pytest.mark.parametrize("recorded", ["normal-off", "clean-source"])
@pytest.mark.parametrize("loaded", ["normal-off", "clean-source"])
def test_loader_rejects_both_mixing_directions_and_caches_success(sources, tmp_path, expected, recorded, loaded):
    path = "deep_ep/_native.py"
    module = module_from(TRANSFORM(path, sources[path], expected), path)
    package = tmp_path / "deep_ep"
    package.mkdir()
    module.__file__ = str(package / "_native.py")
    extension = package / ("_C" + module.EXTENSION_SUFFIXES[0])
    extension.touch()
    (package / "_build_info.json").write_text(json.dumps(artifact(recorded)), encoding="utf-8")
    calls = []
    native = native_identity(loaded)
    module.importlib = SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: SimpleNamespace(origin=str(extension))),
                                       import_module=lambda name: calls.append(name) or native)
    if expected == recorded == loaded:
        assert module.require_native("test") is native
        assert module.require_native("test again") is native
        assert calls == ["deep_ep._C"], "identity and native import are outside repeated communication calls"
    else:
        with pytest.raises(ImportError, match="source/launch ABI"):
            module.require_native("mixed installation")
        assert module._load_native.cache_info().currsize == 0


@pytest.mark.parametrize("mode", ["normal-off", "clean-source"])
def test_st_identity_checks_and_records_installed_artifact(sources, tmp_path, mode):
    st = st_module(sources, mode)
    package = SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    info = tmp_path / "_build_info.json"
    info.write_text(json.dumps(artifact(mode)), encoding="utf-8")
    native = native_identity(mode)
    actual = st.installed_instrumentation_identity(package, native)
    assert actual == dict(instrumentation_source=mode, dispatch_launch_bytes=artifact(mode)["dispatch_launch_bytes"],
                          combine_launch_bytes=artifact(mode)["combine_launch_bytes"])
    args = st.parse_args(["--num-processes", "16"])
    args.runtime_identity = {**actual, "source_sha256": "a" * 64}
    args.profile_route_identity = st.profile_route_identity(st.make_routes(16, 1, 8, 384, 7), 7)
    metadata = st.comparison_metadata(args, 16, 32)
    assert all(metadata[key] == value for key, value in actual.items())
    assert metadata["profile_seed"] == 7 and len(metadata["profile_route_sha256"]) == 64
    mismatch = "normal-off" if mode == "clean-source" else "clean-source"
    with pytest.raises(RuntimeError, match="source/launch ABI"):
        st.installed_instrumentation_identity(package, native_identity(mismatch))
    info.write_text(json.dumps(artifact(mismatch)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source/launch ABI"):
        st.installed_instrumentation_identity(package, native)
    with pytest.raises(RuntimeError, match="differs across ranks"):
        st.validate_matching_capabilities([actual, {**actual, "combine_launch_bytes": 999}])


@pytest.mark.parametrize("option", ["--ll-stage-timer", "--ll-profile-capture", "--ll-expert-timer",
                                    "--ll-timer-iters=1", "--ll-timer-mode=observe", "--ll-timer-frequency-hz=1000000000"])
def test_clean_cli_rejects_capture_and_timer_configuration(sources, option):
    with pytest.raises(SystemExit) as error:
        st_module(sources).parse_args([option])
    assert error.value.code == 2


def test_clean_json_rejects_false_capture_instead_of_accepting_a_fake_switch(sources, tmp_path):
    path = tmp_path / "false_capture.json"
    path.write_text(json.dumps({"ll_stage_timer": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown.*ll_stage_timer"):
        st_module(sources).parse_args(["--config", str(path)])


@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("dispatch_only", [False, True])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_projected_real_run_case_restores_seed_and_combine_handle(sources, tmp_path, monkeypatch, repeats, dispatch_only, dtype):
    # Reuse the existing behavioral fixture against the actual projected module.
    # It executes run_case and real route helpers, and checks every sample closure
    # and every refreshed combine handle after changed-route correctness calls.
    from tests.elastic import test_ll_profile_seed as contract
    monkeypatch.setattr(contract, "st", st_module(sources))
    contract.test_run_case_profiles_restored_indices_and_refreshes_combine_handle(
        tmp_path, monkeypatch, repeats, dispatch_only, dtype)


def elastic_trace(source, monkeypatch, fp8, copy_indices, fail_operation=None):
    module = module_from(source, "deep_ep/buffers/elastic.py", "deep_ep.buffers")
    trace, tensors, launches, events = [], [], {}, []

    class Tensor:
        def __init__(self, shape, dtype="torch.bfloat16", device="npu", label=None):
            self.shape, self.dtype, self.device = tuple(shape), dtype, device
            self.label = label or f"alloc{len(tensors)}"
            self.records = []
            tensors.append(self)
        def record_stream(self, stream):
            self.records.append(stream)
            trace.append(("retain", self.label))
        def clone(self): return Tensor(self.shape, self.dtype, label="indices.copy")
        def to(self, dtype): return Tensor(self.shape, dtype, label="indices.int32")
        def is_contiguous(self): return True

    class Event:
        def __init__(self): trace.append(("event.create",))
        def current_stream_wait(self): trace.append(("event.wait",))

    class Overlap:
        def __init__(self, event, retained):
            self.retained = retained
            events.append(self)
            trace.append(("overlap.create", tuple(value.label for value in retained)))
        def current_stream_wait(self): trace.append(("overlap.wait",))

    torch = SimpleNamespace(Tensor=Tensor, bfloat16="torch.bfloat16", float16="torch.float16",
                            int64="torch.int64", int32="torch.int32", float32="torch.float32",
                            empty=lambda shape, **kwargs: Tensor(shape, **kwargs))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "deep_ep.utils.event", SimpleNamespace(EventHandle=Event, EventOverlap=Overlap))
    runtime = SimpleNamespace(notify_lock=nullcontext(), notify_failed=False, notify_inflight=None,
                              low_latency_comm_args=None, native=object(), world_size=16, rank=1,
                              stream=SimpleNamespace(npu_stream=123), ensure_alive=lambda: None)
    module.runtime_of = lambda self: runtime
    module.npu_api = lambda: SimpleNamespace(stream=lambda stream: nullcontext())

    def launch(operation, native_runtime, named, *args):
        assert native_runtime is runtime.native
        assert runtime.notify_inflight
        for value in named.values():
            if value is not None:
                assert value in runtime.notify_inflight and value.records
        launches[operation] = tuple((key, None if value is None else value.label) for key, value in named.items())
        trace.append(("launch", operation))
        if fail_operation == operation:
            raise RuntimeError("injected native failure")

    native = SimpleNamespace(low_latency_comm_args=lambda _: b"comm", low_latency_dispatch_tiling=lambda *args: b"dispatch",
        low_latency_combine_tiling=lambda *args: b"combine",
        dispatch_low_latency=lambda *args: launch("dispatch", *args), combine_low_latency=lambda *args: launch("combine", *args))
    module.require_native = lambda operation: native
    buffer = SimpleNamespace(_handle_token=object(), _notify_low_latency=lambda *args: None,
        _upload_low_latency=lambda value, device: Tensor((32,), "torch.int8", device, label=value.decode("ascii")))
    hidden = Tensor((4, 128), "torch.float8_e4m3fn" if fp8 else "torch.bfloat16", label="hidden")
    scales = Tensor((4, 1), "torch.float32", label="scales") if fp8 else None
    indices = Tensor((4, 8), "torch.int64", label="indices")
    weights = Tensor((4, 8), "torch.float32", label="weights")
    native_import = builtins.__import__
    clean = "from ..utils.ll_stage_timer import attach" not in source
    imports = []
    def intercept(name, *args, **kwargs):
        if "ll_stage_timer" in name or "ll_expert_timer" in name:
            imports.append(name)
            assert not clean, "clean dispatch/combine must not import a diagnostic module"
            return SimpleNamespace(attach=lambda *args: None)
        return native_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", intercept)
    try:
        result = module.ElasticBuffer._dispatch_low_latency(buffer, (hidden, scales) if fp8 else hidden,
            indices, weights, 384, 32, copy_indices, False, {"capacity_rows": 4096})
        handle = result[3]
        assert result[1:3] == (None, None) and handle.is_low_latency
        assert runtime.notify_inflight is None
        assert all(tensor.records for tensor in events[-1].retained)
        expert = Tensor((handle.num_expanded_tokens, handle.hidden), label="expert")
        combined = module.ElasticBuffer._combine_low_latency(buffer, expert, handle, None, None, 0, 0, None, None, False, False)
        assert combined[1] is None and combined[0].shape == (4, 128)
        assert runtime.notify_inflight is None
        assert all(tensor.records for tensor in events[-1].retained)
    except RuntimeError as error:
        assert fail_operation and str(error) == "injected native failure"
        assert runtime.notify_failed and runtime.notify_inflight
        trace.append(("failed-retained", tuple(value.label for value in runtime.notify_inflight)))
    else:
        assert fail_operation is None
        assert not runtime.notify_failed
    assert bool(imports) is not clean
    return trace, launches


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("copy_indices", [False, True])
@pytest.mark.parametrize("fail_operation", [None, "dispatch", "combine"])
def test_actual_elastic_methods_keep_tensor_lifetimes_events_and_failures(sources, monkeypatch, fp8, copy_indices, fail_operation):
    original = sources["deep_ep/buffers/elastic.py"]
    with monkeypatch.context() as patch:
        baseline = elastic_trace(original, patch, fp8, copy_indices, fail_operation)
    with monkeypatch.context() as patch:
        clean = elastic_trace(TRANSFORM("deep_ep/buffers/elastic.py", original, "clean-source"), patch, fp8, copy_indices, fail_operation)
    assert clean == baseline


@pytest.mark.parametrize("mode", ["normal-off", "clean-source"])
@pytest.mark.parametrize("configured", [None, "shared-sdk", "absolute"])
def test_launcher_uses_same_explicit_sdk_lib_in_both_sources(sources, tmp_path, monkeypatch, mode, configured):
    path = "tests/ll_stage_timer/run.py"
    module = module_from(TRANSFORM(path, sources[path], mode), path)
    module.ROOT = tmp_path
    monkeypatch.delenv("DEEPEP_RDMA_PROVIDER_PATH", raising=False)
    if configured is None:
        monkeypatch.delenv("SHMEM_ROOT", raising=False)
        expected = tmp_path / "third-party/shmem/install/shmem/lib"
    else:
        root = str(tmp_path / "explicit sdk") if configured == "absolute" else configured
        monkeypatch.setenv("SHMEM_ROOT", root)
        expected = (Path(root) if Path(root).is_absolute() else tmp_path / root) / "lib"
    env = module.environment("0,1")
    assert env["LD_LIBRARY_PATH"].startswith(str(expected) + ":")
    assert all(env[name] == "0,1" for name in module.VISIBILITY)


@pytest.mark.parametrize("mode", ["normal-off", "clean-source"])
@pytest.mark.parametrize("backend", ["hccl", "gloo"])
def test_actual_sample_operation_preserves_10_100_tail50_exp_reduce_order(sources, tmp_path, monkeypatch, mode, backend):
    st = st_module(sources, mode)
    args = st.parse_args(["--num-processes", "16", "--group-backend", backend])
    args.run_directory, args.case_id = str(tmp_path), 0
    trace = []
    class Tensor:
        def __init__(self, label, value=0): self.label, self.value = label, value
        def fill_(self, value): trace.append("fill"); self.value = value
        def item(self): return self.value
    class Profiler:
        def __enter__(self): trace.append("profile.enter"); return self
        def __exit__(self, *args): trace.append("profile.exit")
        def step(self): trace.append("step")
    samples = [float(value) for value in range(1, 101)]
    def gather(rows, local):
        assert local == samples
        for rank in range(16): rows[rank] = [value + rank for value in samples]
        trace.append("gather")
    dist = SimpleNamespace(ReduceOp=SimpleNamespace(SUM="SUM", MAX="MAX"),
        all_reduce=lambda tensor, **kwargs: trace.append("reduce." + kwargs["op"]),
        barrier=lambda: trace.append("barrier"), all_gather_object=gather,
        get_rank=lambda: 0, get_world_size=lambda: 16)
    torch = SimpleNamespace(float32="float32", int32="int32", distributed=dist,
        zeros=lambda *args, **kwargs: Tensor("delay"), ones=lambda *args, **kwargs: Tensor("align", 1),
        tensor=lambda values, **kwargs: Tensor("status", values[0]),
        exp=lambda *args, **kwargs: trace.append("exp"), npu=SimpleNamespace(synchronize=lambda: trace.append("sync")))
    profile = SimpleNamespace(profile=lambda **kwargs: Profiler(), schedule=lambda **kwargs: kwargs,
        tensorboard_trace_handler=lambda path: path, _ExperimentalConfig=lambda **kwargs: kwargs,
        AiCMetrics=SimpleNamespace(PipeUtilization=0), ProfilerLevel=SimpleNamespace(Level1=1),
        ExportType=SimpleNamespace(Text="text"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profile))
    monkeypatch.setattr(st, "kernel_durations", lambda *args: trace.append("parse") or samples)
    result = st.sample_operation(None, lambda: trace.append("invoke") or object(),
        lambda _: pytest.fail("formal profiling does not enter diagnostic checks"),
        args, "dispatch", 0, 32, "normal", None)
    start, end = trace.index("profile.enter") + 1, trace.index("profile.exit")
    expected = ["invoke"] * 10 + ["sync", "step"]
    one = ["exp", "fill"] + (["sync"] if backend == "gloo" else []) + ["reduce.SUM", "invoke"]
    expected += one * 100 + ["barrier", "sync", "step"]
    assert trace[start:end] == expected
    assert trace[end + 1:] == ["parse", "reduce.MAX", "gather"]
    assert result["perf_tail"] == 50 and result["kernel_samples"] == 16 * 50
    assert result["summary_statistics_scope"] == "pooled_rank_tail_kernel_samples"
    pooled = [value + rank for rank in range(16) for value in samples[-50:]]
    assert result["mean_kernel_us"] == st.sample_stats(pooled)["mean_us"]
    assert result["p99_kernel_us"] == st.sample_stats(pooled)["p99_us"]
