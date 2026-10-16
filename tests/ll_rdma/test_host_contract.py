from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backend,udma", [("rdma", False), ("mte", True)])
def test_explicit_runtime_selection(monkeypatch, backend, udma):
    from deep_ep.buffers import elastic
    calls = []
    monkeypatch.setenv("DEEPEP_LL_BACKEND", backend)
    monkeypatch.setattr(elastic, "require_native", lambda _: None)
    monkeypatch.setattr(elastic, "Runtime", lambda *a, **k: calls.append(k) or SimpleNamespace(closed=True))
    elastic.ElasticBuffer(num_bytes=8 << 30, explicitly_destroy=True, use_1g_huge_pages=False,
                          rank=0, world_size=16)
    assert calls[0]["udma"] is udma


def test_no_ht_fallback(monkeypatch):
    from deep_ep.buffers import elastic
    monkeypatch.setattr(elastic, "require_native", lambda _: None)
    monkeypatch.setattr(elastic, "npu_api", lambda: None)
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer._runtime = SimpleNamespace(ensure_alive=lambda: None, ll_backend="rdma", closed=True)
    with pytest.raises(NotImplementedError, match="no HT/MTE fallback"):
        buffer.dispatch(object())
    with pytest.raises(NotImplementedError, match="no HT/MTE fallback"):
        buffer.combine(object(), object())


def test_moved_checkout_plan_and_environment(tmp_path):
    moved = tmp_path / "new project with spaces"
    relatives = ["tests/elastic/test_ep_low_latency.py", "tests/ll_rdma/run_ll.py", "tests/ll_rdma/run_smoke.py",
                 "tests/ll_rdma/environment.py", "tests/ll_rdma/target16.json", "tests/ll_rdma/target32.json",
                 "tests/ll_stage_timer/run.py", "tests/ll_stage_timer/branch.json"]
    for relative in relatives:
        dst = moved / relative; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dst)
    elsewhere = tmp_path / "other cwd"; elsewhere.mkdir()
    for world in (16, 32):
        command = [sys.executable, str(moved / "tests/ll_rdma/run_ll.py"), "--backend", "rdma", "--validation",
                   "--devices", "0,1", "--config", f"tests/ll_rdma/target{world}.json", "--plan"]
        result = subprocess.run(command, cwd=elsewhere, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["world_size"] == world
    assert (moved / ".logs/shmem").is_dir()
    rejected = subprocess.run(command[:-1], cwd=elsewhere, capture_output=True, text=True)
    assert rejected.returncode and "CPU --plan only" in rejected.stderr
