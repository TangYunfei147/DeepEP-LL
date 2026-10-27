"""CPU checks for schema-3 byte counters and per-sample EP16 effective rates."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bandwidth_analyzer', Path(__file__).with_name('analyze.py'))
analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
timer = analyzer.timer


def raw(schema=3, duration=1000, flags=0, operation='dispatch'):
    result = []
    for core in range(16):
        words = [0]*256
        words[:8] = [timer.MAGIC, schema, 1 if operation=='dispatch' else 2, 16, core, 1, 1, len(timer.STAGES)]
        # Different per-core clock origins ensure neither sum nor cross-core envelope is used.
        start = 10000*core
        words[8:12] = [start, start+duration+core, duration+core, 1]
        if schema==3:
            words[208:212] = [500, 1, 0, 0]
            words[212:216] = [1000, 2, 250, 1]
            words[216:220] = [100, 1, 0, 0]
            words[228] = flags
        result.append([words[i:i+8] for i in range(0,256,8)])
    return result


def sample(schema=3, duration=1000, flags=0, world=16, frequency=1e9, operation='dispatch'):
    report = timer.parse_records(raw(schema,duration,flags,operation),operation,frequency)
    meta = dict(rank=0,world_size=world,iteration=0)
    return analyzer.bandwidth_sample(report,meta)


def test_schema3_layout_counter_subset_and_trace_compatibility():
    values=raw()
    report=timer.parse_records(values,'dispatch',1e9)
    assert report['traffic_available'] and len(report['traffic'])==80
    assert len(report['records'])==16
    assert report['traffic'][1]['read_bytes']==250
    assert len(values[0])*len(values[0][0])*8==2048
    trace=timer.chrome_trace([dict(raw=values,operation='dispatch',frequency_hz=1e9)])
    assert len([event for event in trace['traceEvents'] if event['ph']=='X'])==16


@pytest.mark.parametrize('word,value', [(196,1),(207,1),(229,1),(255,1),(228,2),(212,-1),(214,1001),(215,3)])
def test_invalid_traffic_rejected(word,value):
    values=raw();values[0][word//8][word%8]=value
    with pytest.raises(ValueError):timer.parse_records(values,'dispatch')


def test_effective_rate_counts_read_once_and_uses_max_core_duration():
    rows={row['category']:row for row in sample(operation='combine')}
    total=rows['intrahost_total']
    assert total['bytes']==17600 and total['read_bytes']==4000
    assert total['operations']==48 and total['read_operations']==16
    assert total['duration_ticks']==1015 and total['duration_us']==1.015
    assert total['effective_GBps']==pytest.approx(17600/1015)
    assert rows['local']['bytes']==8000 and rows['local']['effective_GBps'] is None
    assert rows['interhost_first']['bytes']==0 and rows['interhost_first']['effective_GBps'] is None


@pytest.mark.parametrize('kwargs,status', [
    ({'schema':2},'counters_unavailable'), ({'flags':1},'nonstandard_skip_communication'),
    ({'world':32},'unsupported_ep'), ({'frequency':None},'frequency_unavailable')])
def test_unavailable_rates_are_not_zero(kwargs,status):
    row=next(row for row in sample(**kwargs) if row['category']=='intrahost_total')
    assert row['status']==status and row['effective_GBps'] is None
    if kwargs.get('schema')==2:assert row['bytes'] is None


def test_ep16_interhost_is_rejected():
    values=raw();values[0][220//8][220%8]=1
    report=timer.parse_records(values,'dispatch',1e9)
    with pytest.raises(ValueError,match='EP16'):
        analyzer.bandwidth_sample(report,dict(rank=0,world_size=16,iteration=0))


def test_five_rounds_divide_before_statistics_and_export(tmp_path):
    paths=[]
    for rank in range(16):
        for iteration in range(5):
            path=tmp_path/f'sample_{rank}_{iteration}.json'
            path.write_text(json.dumps(dict(raw=raw(duration=1000*(iteration+1)),operation='dispatch',frequency_hz=1e9,
                metadata=dict(rank=rank,world_size=16,iteration=iteration,config={},correctness='passed',
                              sampling=dict(diagnostic_iters=5)))))
            paths.append(path)
    report=analyzer.analyze(paths)
    entry=next(row for row in report['per_rank_bandwidth'] if row['rank']==0 and row['category']=='intrahost_total')
    expected=sum(17600/(1000*(iteration+1)+15) for iteration in range(5))/5
    assert entry['effective_GBps']['mean']==pytest.approx(expected)
    assert entry['effective_GBps']['samples']==5
    assert entry['effective_GBps']['mean']!=pytest.approx(17600/3015)
    out=tmp_path/'output'
    analyzer.process_directory(tmp_path,out,None,0)
    assert (out/'bandwidth.csv').is_file() and (out/'trace.json').is_file()
    assert json.loads((out/'summary.json').read_text())['per_rank_bandwidth']


def test_ticks_only_output_has_no_trace_or_rate(tmp_path):
    for rank in range(16):
        path=tmp_path/f'sample_{rank}.json'
        path.write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=None,
            metadata=dict(rank=rank,world_size=16,iteration=0,config={},correctness='passed'))))
    output=tmp_path/'output'
    analyzer.process_directory(tmp_path,output,None,0)
    assert not (output/'trace.json').exists()
    report=json.loads((output/'summary.json').read_text())
    assert all(row['effective_GBps'] is None for row in report['bandwidth'])


def test_schema3_both_operations_from_moved_checkout(tmp_path):
    import csv
    import shutil
    import subprocess
    import sys
    moved = tmp_path / "moved project with spaces"
    for relative in ("deep_ep/utils/ll_stage_timer.py", "deep_ep/utils/ll_expert_timer.py", "tests/ll_stage_timer/analyze.py",
                     "tests/ll_rdma/report_tables.py", "tests/ll_reference_profile/export_xlsx.py"):
        target = moved / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    source = moved / "data"
    source.mkdir()
    for operation in ("dispatch", "combine"):
        for rank in range(16):
            (source / f"sample_{operation}_{rank}.json").write_text(json.dumps(dict(
                raw=raw(operation=operation), operation=operation, frequency_hz=1e9,
                metadata=dict(rank=rank, world_size=16, iteration=0, config={}, correctness="passed"))))
    other = tmp_path / "unrelated cwd"
    other.mkdir()
    subprocess.run([sys.executable, str(moved / "tests/ll_stage_timer/analyze.py"),
                    "data", "--output-dir", "report"], cwd=other, check=True,
                   capture_output=True, text=True)
    with (moved / "report/bandwidth.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["operation"] for row in rows} == {"dispatch", "combine"}
    assert len(rows) == 16 * 2 * 6
    for row in rows:
        if row["category"] == "intrahost_total":
            assert row["status"] == "ok"
            assert float(row["effective_GBps"]) == pytest.approx(17600 / 1015)


def test_zero_measured_bytes_is_a_valid_zero_rate():
    values = raw()
    for core in values:
        flat = [word for slot in core for word in slot]
        flat[208:228] = [0] * 20
        core[:] = [flat[i:i+8] for i in range(0, 256, 8)]
    report = timer.parse_records(values, "dispatch", 1e9)
    rows = analyzer.bandwidth_sample(report, dict(rank=0, world_size=16, iteration=0))
    total = next(row for row in rows if row["category"] == "intrahost_total")
    assert total["status"] == "ok" and total["effective_GBps"] == 0


def test_capture_entry_accepts_new_schema_and_rejects_old_wheel():
    import runpy
    validate = runpy.run_path(str(ROOT / "tests/elastic/test_ep_low_latency.py"))["validate_stage_timer_capability"]
    validate(True, dict(stage_timer=True, stage_timer_schema=timer.SCHEMA))
    validate(False, dict(stage_timer=False, stage_timer_schema=2))
    with pytest.raises(RuntimeError, match="schema 3"):
        validate(True, dict(stage_timer=True, stage_timer_schema=2))
    with pytest.raises(RuntimeError, match="STAGE_TIMER=ON"):
        validate(True, dict(stage_timer=False, stage_timer_schema=timer.SCHEMA))
