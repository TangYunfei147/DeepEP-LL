import json
import subprocess
import sys
from pathlib import Path
from tests.ll_probe.profiling.startup_diagnostics import StartupDiagnostics


def test_stages_and_exception_cleanup(tmp_path):
    try:
        with StartupDiagnostics(tmp_path, 7) as diag:
            diag.mark('hccl_sync_begin')
            raise ValueError('test')
    except ValueError:
        pass
    records=[json.loads(x) for x in (tmp_path/'startup.jsonl').read_text().splitlines()]
    assert [x['stage'] for x in records] == ['diagnostics_ready','hccl_sync_begin','startup_exception:ValueError','startup_exit']
    assert all(x['rank']==7 for x in records)
    assert not diag.active and diag.stacks.closed


def test_watchdog_dumps_without_exit_and_stops_before_capture(tmp_path):
    root=Path(__file__).resolve().parents[4]
    code="""
import time
from pathlib import Path
from tests.ll_probe.profiling.startup_diagnostics import StartupDiagnostics
with StartupDiagnostics(Path(__import__('sys').argv[1]), 0, timeout=0.05) as diag:
    diag.mark('hccl_sync_begin')
    time.sleep(0.13)
    diag.stop('native_validation_done')
    before=Path(__import__('sys').argv[1], 'startup_stacks.log').read_bytes()
    time.sleep(0.08)
    assert before==Path(__import__('sys').argv[1], 'startup_stacks.log').read_bytes()
"""
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=root,check=True)
    assert 'Timeout' in (tmp_path/'startup_stacks.log').read_text()


def test_probe_stops_watchdog_before_capture():
    source=(Path(__file__).resolve().parents[1]/'profile_probe.py').read_text()
    assert source.index("startup.stop('native_validation_done')") < source.index('capture = Capture(')
