"""Synthetic offline export tests; no framework or device imports."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from tests.ll_rdma.analyze import analyze_run
from tests.ll_reference_profile.test_parse import inputs
from tests.ll_stage_timer.test_bandwidth import raw

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_reference_two_cases_retargets_formula_and_preserves_precision(tmp_path):
    for case in range(2):
        for path in inputs(tmp_path/f'case_{case}/ll_reference_profile'):
            value = json.loads(path.read_text())
            value['metadata']['case_id'] = case
            path.write_text(json.dumps(value))
    out = analyze_run(tmp_path)
    with ZipFile(out/'stage-times.xlsx') as z:
        assert z.testzip() is None
        sheets = ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS)
        names = [s.attrib['name'] for s in sheets]
        assert len(names) == 15 and len(set(names)) == 15
        assert 'C001_逐核耗时' in names and 'C002_全部轮次明细' in names
        detail = z.read('xl/worksheets/sheet3.xml').decode()
        assert "'C001_原始Cycle'!" in detail and "'原始Cycle'!" not in detail
        assert '100000000000000000' in z.read('xl/worksheets/sheet5.xml').decode()
        for name in z.namelist():
            ET.fromstring(z.read(name))
    wide = rows(out/'case_001/stages.csv')
    assert len(wide) == 256 and {r['iteration'] for r in wide} == {'1'}
    assert len({(r['rank'], r['core']) for r in wide}) == 256
    assert (out/'return_report.md').stat().st_size < 60000
    with pytest.raises(FileExistsError):
        analyze_run(tmp_path)


@pytest.mark.parametrize('frequency', [None, 1e9])
def test_native_last_mean_operations_and_raw_words(tmp_path, frequency):
    source = tmp_path/'case/ll_stage_timer'
    source.mkdir(parents=True)
    for operation in ('dispatch', 'combine'):
        for rank in range(16):
            for iteration in range(2):
                saved = dict(raw=raw(duration=1000*(iteration+1), operation=operation),
                             operation=operation, frequency_hz=frequency,
                             metadata=dict(rank=rank, world_size=16, iteration=iteration,
                                           run_id='synthetic', case_id=0, config={}, correctness='passed'))
                (source/f'sample_{operation}_{rank}_{iteration}.json').write_text(json.dumps(saved))
    out = analyze_run(tmp_path)
    wide, mean = rows(out/'case_001/stages.csv'), rows(out/'case_001/stages_mean.csv')
    factor = 1000 if frequency else 1
    assert len(wide) == len(mean) == 512
    assert {r['operation'] for r in wide} == {'dispatch', 'combine'}
    assert {r['iteration'] for r in wide} == {'1'}
    assert float(wide[0]['kernel']) == 2000/factor
    assert float(mean[0]['kernel']) == 1500/factor
    assert wide[0]['magic'] == ''  # absent is not an executed zero
    assert {r['unit'] for r in rows(out/'case_001/stage_calls.csv')} == {'calls'}
    with ZipFile(out/'stage-times.xlsx') as z:
        names = [s.attrib['name'] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS)]
        assert {'逐核耗时_dispatch', '逐核耗时_combine', '通信计数', '接收明细', '原始Cycle'} <= set(names)


def test_no_capture_has_no_fake_stage_table(tmp_path):
    with pytest.raises(ValueError, match='No diagnostic samples'):
        analyze_run(tmp_path)
    assert not (tmp_path/'table-analysis').exists()


def test_shell_analyze_only_bypasses_cann_from_other_cwd(tmp_path):
    inputs(tmp_path/'data with spaces/ll_reference_profile')
    script = Path(__file__).with_name('one_click.sh')
    env = dict(os.environ, PYTHON=sys.executable, CANN_ENV='nonexistent-cann-file')
    result = subprocess.run(['bash', str(script), '--analyze-only', str(tmp_path/'data with spaces'),
                             '--output-dir', str(tmp_path/'output with spaces')],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'output with spaces/stage-times.xlsx').is_file()
