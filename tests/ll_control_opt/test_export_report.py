"""The transfer report is one bounded UTF-8 file with every comparison row."""
import csv
import importlib.util
import io
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location('control_export_report_tests', Path(__file__).with_name('export_report.py'))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def comparison_rows():
    rows = []
    labels = ('baseline', 'buffered-c2', 'immediate-c2')
    for round_id in (1, 2):
        for seed in (0, 17, 101):
            for index, label in enumerate(labels):
                for operation in ('dispatch', 'combine'):
                    rows.append(dict(
                        round=str(round_id), seed=str(seed), measured_route_seed=str(seed+2),
                        label=label, revision=str(index+1)*40, operation=operation,
                        mean_kernel_us='00088.12345678901234' if operation=='dispatch' else '1.3012345678901234e+02',
                        reference_label='baseline', reference_revision='1'*40,
                        reference_delta_us='-0.000000000000001' if index else '0.0',
                        reference_delta_pct='-1.234567890123456e-08' if index else '0.0',
                        execution_position=str(index+1 if round_id==1 else 3-index),
                        result_status='new' if round_id==1 else 'reused', source_session_id='a'*32,
                        reference_crosses_sessions='False' if round_id==1 else '',
                        shmem_revision='b'*40, reference_shmem_revision='b'*40,
                        sdk_identity_source='recorded-build' if round_id==1 else 'legacy-unavailable',
                        sdk_library_matches_reference='True' if round_id==1 else '',
                        source_directory='/a/very/long/local/path/that/is/not/transmitted',
                        sdk_library_sha256='c'*64, previous_manifest_delta_us='123.5'))
    return rows


def write_csv(path, rows, fields=None, *, bom=False):
    fields = list(fields or rows[0])
    with path.open('w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_report(data):
    header, csv_text = data.decode('utf-8').split('\n\n', 1)
    return header, list(csv.DictReader(io.StringIO(csv_text, newline='')))


@pytest.mark.parametrize('bom', [False, True])
def test_all36_rows_and_original_numeric_strings_survive(tmp_path, bom):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    original = comparison_rows()
    write_csv(source, original, bom=bom)
    before = source.read_bytes()
    size = report.export(source, output)
    data = output.read_bytes()
    heading, rows = read_report(data)
    assert size == len(data) <= 65535
    assert len(rows) == len(original) == 36
    assert 'Rows: 36' in heading
    assert list(rows[0]) == list(report.FIELDS) and len(report.FIELDS) == 19
    assert rows == [{field: row[field] for field in report.FIELDS} for row in original]
    assert {row['operation'] for row in rows} == {'dispatch', 'combine'}
    assert {row['measured_route_seed'] for row in rows} == {'2', '19', '103'}
    assert 'source_directory' not in rows[0] and 'sdk_library_sha256' not in rows[0]
    assert source.read_bytes() == before
    assert set(path.name for path in tmp_path.iterdir()) == {'comparison.csv', 'analysis-report.txt'}


def test_missing_optional_identity_stays_blank_not_false_or_zero(tmp_path):
    source = tmp_path/'comparison.csv'
    rows = comparison_rows()[:1]
    write_csv(source, rows, report.REQUIRED)
    heading, output = read_report(report.render(source))
    assert 'Blank identity fields mean unavailable' in heading
    for field in ('sdk_library_matches_reference', 'reference_crosses_sessions', 'sdk_identity_source', 'source_session_id'):
        assert output[0][field] == ''
    assert output[0]['mean_kernel_us'] == rows[0]['mean_kernel_us']


@pytest.mark.parametrize('missing', report.REQUIRED)
def test_missing_required_column_rejected_without_output(tmp_path, missing):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    rows = comparison_rows()
    write_csv(source, rows, [field for field in rows[0] if field != missing])
    before = source.read_bytes()
    with pytest.raises(ValueError, match='comparison.csv'):
        report.export(source, output)
    assert not output.exists() and source.read_bytes() == before


def test_utf8_byte_limit_rejects_even_when_character_count_is_small(tmp_path):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    rows = comparison_rows()[:1]
    rows[0]['label'] = '测' * 23000
    write_csv(source, rows)
    assert len(source.read_text(encoding='utf-8')) < 65535
    before = source.read_bytes()
    with pytest.raises(ValueError, match='above 65535.*no rows truncated'):
        report.export(source, output)
    assert not output.exists() and source.read_bytes() == before


def test_exact_byte_boundary_is_accepted_and_next_byte_is_rejected(tmp_path):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    rows = comparison_rows()[:1]
    rows[0]['label'] = 'x'
    write_csv(source, rows)
    free = report.MAX_BYTES-len(report.render(source))
    rows[0]['label'] = 'x' + '测' * (free//3) + 'a' * (free%3)
    write_csv(source, rows)
    assert report.export(source, output) == 65535
    assert len(output.read_bytes()) == 65535
    rows[0]['label'] += 'x'
    write_csv(source, rows)
    rejected = tmp_path/'too-big.txt'
    with pytest.raises(ValueError, match='65536 bytes'):
        report.export(source, rejected)
    assert not rejected.exists() and len(output.read_bytes()) == 65535


def test_existing_output_and_source_are_never_overwritten(tmp_path):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    write_csv(source, comparison_rows())
    original_source = source.read_bytes()
    output.write_bytes(b'user-owned prior report\n')
    with pytest.raises(FileExistsError):
        report.export(source, output)
    assert output.read_bytes() == b'user-owned prior report\n'
    with pytest.raises(FileExistsError):
        report.export(source, source)
    assert source.read_bytes() == original_source


@pytest.mark.parametrize('contents', ['', ','.join(report.REQUIRED)+'\n', ','.join(report.REQUIRED)+'\n1,0\n'])
def test_empty_or_structurally_incomplete_input_is_rejected(tmp_path, contents):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    source.write_text(contents, encoding='utf-8')
    with pytest.raises(ValueError):
        report.export(source, output)
    assert not output.exists()


@pytest.mark.parametrize('value', ['', '   '])
def test_required_values_cannot_be_blank(tmp_path, value):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    rows = comparison_rows()[:1]
    rows[0]['mean_kernel_us'] = value
    write_csv(source, rows)
    with pytest.raises(ValueError, match='malformed'):
        report.export(source, output)
    assert not output.exists()


def test_duplicate_header_does_not_silently_discard_a_column(tmp_path):
    source, output = tmp_path/'comparison.csv', tmp_path/'analysis-report.txt'
    source.write_text(','.join((*report.REQUIRED, 'mean_kernel_us'))+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Expected'):
        report.export(source, output)
    assert not output.exists()
