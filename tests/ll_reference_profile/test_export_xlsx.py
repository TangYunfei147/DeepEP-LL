"""OOXML, integer precision and single-iteration export checks with fake data."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from tests.ll_reference_profile.test_parse import analyzer, inputs, sample, timer

NS = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def cells(path, sheet):
    with ZipFile(path) as z:
        tree = ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml'))
    return {c.attrib['r']:c for c in tree.findall('.//m:sheetData/m:row/m:c', NS)}, tree


def text(cell):
    return ''.join(cell.itertext())


def export(tmp_path, **kwargs):
    paths = inputs(tmp_path/'ll_reference_profile', **kwargs)
    report = analyzer.analyze(paths)
    selected = [json.loads(p.read_text()) for p in paths if p.name == 'sample_0001.json']
    path = tmp_path/'out.xlsx'
    analyzer.exporter.export_workbook(path, report, selected, timer)
    return path, report, selected


def test_export_preserves_raw_precision_formulas_missing_zero_and_layout(tmp_path):
    path, report, selected = export(tmp_path, cores=56)
    detail, tree = cells(path,2)
    assert tree.find('m:dimension',NS).attrib['ref'] == 'A1:AH897'
    assert tree.find('m:sheetViews/m:sheetView/m:pane',NS).attrib['topLeftCell'] == 'F2'
    assert [text(detail[f'{c}1']) for c in 'ABCDE'] == ['Rank', '核号', '收发角色', '任务数', '核总耗时 (µs)']
    assert not {'实验组', '服务器', '本地卡', '采集轮次'} & {text(c) for key,c in detail.items() if key.rstrip('0123456789')+'1' == key}
    assert tree.find('m:autoFilter',NS).attrib['ref'] == 'A1:AH897'
    assert detail['E2'].find('m:f',NS).text == "'原始Cycle'!BO2/'阶段说明'!$B$2"
    assert float(detail['E2'].find('m:v',NS).text) == .36
    assert detail['AE3'].find('m:f',NS) is None  # empty sender has no packet-submit segment
    receiver_row = 30
    assert float(detail[f'Z{receiver_row}'].find('m:v',NS).text) == 0
    assert detail['D2'].find('m:v',NS).text == '1'
    raw, _ = cells(path,4)
    assert raw['C2'].attrib['t'] == 'inlineStr'
    assert text(raw['C2']) == str(10**17)
    assert raw['BO2'].find('m:v',NS).text == '360'
    assert raw['B2'].find('m:v',NS).text == '0'
    definitions, _ = cells(path,3)
    assert float(definitions['B2'].find('m:v',NS).text) == 1000
    with ZipFile(path) as z:
        assert z.testzip() is None
        for name in z.namelist(): ET.fromstring(z.read(name))
        names = [s.attrib['name'] for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',NS)]
        assert names == ['采样汇总','逐核耗时','阶段说明','原始Cycle','S包时序','R包顺序','包记录覆盖','包原始记录']
    with pytest.raises(FileExistsError): analyzer.exporter.export_workbook(path,report,selected,timer)


def test_small_raw_uses_reference_differences_and_no_frequency_keeps_ticks(tmp_path):
    path, _, _ = export(tmp_path, frequency=None, offset=0)
    detail, _ = cells(path,2)
    assert text(detail['E1']) == '核总耗时 (ticks)'
    assert detail['E2'].find('m:f',NS).text == "('原始Cycle'!X2-'原始Cycle'!C2)"
    assert detail['E2'].find('m:v',NS).text == '360'
    definition, _ = cells(path,3)
    assert definition['B2'].find('m:v',NS) is None


def test_selection_incomplete_duplicate_wrong_iteration_and_metadata_rejected(tmp_path):
    path, report, selected = export(tmp_path)
    for bad in (selected[:-1], selected+[selected[0]]):
        with pytest.raises(ValueError): analyzer.exporter.export_workbook(tmp_path/'bad.xlsx',report,bad,timer)
    selected[0]['metadata']['iteration'] = 0
    with pytest.raises(ValueError): analyzer.exporter.export_workbook(tmp_path/'bad.xlsx',report,selected,timer)
    selected[0]['metadata']['iteration'] = 1
    selected[0]['metadata']['config']['hidden'] = 1
    with pytest.raises(ValueError, match='metadata'): analyzer.exporter.export_workbook(tmp_path/'bad.xlsx',report,selected,timer)
    assert not (tmp_path/'bad.xlsx').exists()


def test_process_outputs_all_rounds_and_default_last_cli_from_other_cwd(tmp_path):
    source = tmp_path/'path with spaces/ll_reference_profile'
    inputs(source)
    script = Path(__file__).with_name('analyze.py')
    result = subprocess.run([sys.executable,str(script),str(source)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    output = source/'reference-analysis'
    report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert report['selected_iteration'] == 1 and report['sample_count'] == 32
    assert (output/'stages.csv').read_text(encoding='utf-8-sig').count('\n') == 16*16+1
    assert (output/'stages_long.csv').read_text(encoding='utf-8-sig').count('\n') > 16*16*2
    assert 0 < (output/'return_report.md').stat().st_size <= 60000
    assert (output/'core_stage_timings.xlsx').is_file()
    again = subprocess.run([sys.executable,str(script),str(source)],cwd=tmp_path,capture_output=True,text=True)
    assert again.returncode != 0


def test_text_escaping_and_formula_like_strings_stay_text(tmp_path):
    paths = inputs(tmp_path/'ll_reference_profile')
    for path in paths:
        value = json.loads(path.read_text()); value['metadata']['run_id'] = '=SUM(A1)&<tag>"'
        path.write_text(json.dumps(value),encoding='utf-8')
    report = analyzer.analyze(paths)
    selected = [json.loads(path.read_text()) for path in paths if path.name == 'sample_0001.json']
    path = tmp_path/'escaped.xlsx'; analyzer.exporter.export_workbook(path,report,selected,timer)
    detail,_ = cells(path,1)
    assert detail['B2'].attrib['t'] == 'inlineStr' and detail['B2'].find('m:f',NS) is None
    assert '=SUM(A1)&<tag>"' in text(detail['B2'])


def test_multi_case_cli_single_bounded_return_and_actual_route_identity(tmp_path):
    run = tmp_path/'run'
    for case in range(2):
        paths = inputs(run/f'case_{case}'/'normal'/'ll_reference_profile')
        for path in paths:
            saved = json.loads(path.read_text())
            saved['metadata']['case_id'] = case
            path.write_text(json.dumps(saved), encoding='utf-8')
    output = run/'export'
    command = [sys.executable, str(Path(__file__).with_name('analyze.py')), str(run), '--output-dir', str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compact = (output/'return_report.md').read_text(encoding='utf-8')
    assert '共2个case/route' in compact and '核总耗时' in compact
    assert len(compact.encode('utf-8')) <= 60000
    assert len(list(output.rglob('core_stage_timings.xlsx'))) == 2
    saved = json.loads(paths[-1].read_text())
    saved['metadata']['route_sha256'] = 'f'*64
    paths[-1].write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='Mixed'):
        analyzer.analyze(paths)
    saved['metadata']['route_sha256'] = '1'*64
    saved['metadata']['profile_seed'] += 1
    paths[-1].write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='profile seed'):
        analyzer.analyze(paths)
    report = analyzer.analyze(inputs(tmp_path/'long-run'))
    report['selected_iteration'] = 1
    bounded = analyzer.run_report([report]*100)
    assert len(bounded.encode('utf-8')) <= 60000
    assert '共100个case/route' in bounded
