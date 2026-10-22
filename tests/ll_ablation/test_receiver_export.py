"""CPU checks for bounded, lossless transfer of receiver replay evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receiver_export as export


def result_fixture(*, full=False, long_text=None):
    case = json.dumps({'B': 256, 'H': 3584, 'route': '正常'}, ensure_ascii=False)
    rows = []
    for iteration in range(9 if full else 1):
        for rank in range(16 if full else 1):
            cores = []
            for core in range(56 if full else 1):
                packets = 35 + (core + rank + iteration) % 5
                flags = (packets + 15) // 16
                payload = packets * 3696
                tick = 25000 + iteration * 1000 + rank * 100 + core
                arms = {}
                for offset, name in enumerate(export.CASES):
                    arms[name] = dict(
                        ticks=tick + offset * 17,
                        flag_batches=0 if name == 'skip-flags' else flags,
                        payload_bytes=0 if name == 'no-payload-output' else payload,
                        first_tick=1000000 + tick + offset * 70000,
                        last_tick=1000000 + 2 * tick + offset * 70017,
                    )
                cores.append(dict(core=core, order=iteration % 3, arms=arms))
            rows.append(dict(rank=rank, iteration=iteration, order=iteration % 3,
                             frequency_hz=1000000000, cores=cores,
                             actual_route_seed=2, route_digest='a' * 64,
                             max_core_us={name: max(c['arms'][name]['ticks'] for c in cores) / 1000
                                          for name in export.CASES}))
    summary = dict(limitations=['不推断生产收益'], actual_route_seed=2)
    if long_text is not None:
        summary['long_record'] = long_text
    return dict(identity={'mode': 'ready-breakdown', '说明': '测试😀'},
                interpretation='离线结果，不运行 NPU',
                summary={case: summary}, samples={case: rows})


def recover_groups(directory, manifest):
    """Independent receiver: reassemble escaped JSON fragments and verify them."""
    groups = {kind: [] for kind in ('summary', 'ranks', 'cores')}
    for entry in manifest['files']:
        part = json.loads((directory / entry['file']).read_bytes())
        groups[part['kind']].extend(part['records'])
    recovered = []
    for kind, records in groups.items():
        output = []
        pos = 0
        while pos < len(records):
            first = records[pos]
            if 'fragment_record_sha256' not in first:
                output.append(first)
                pos += 1
                continue
            count = first['fragments']
            pieces = records[pos:pos + count]
            assert len(pieces) == count
            assert [piece['fragment'] for piece in pieces] == list(range(1, count + 1))
            assert all(piece['fragments'] == count and
                       piece['fragment_record_sha256'] == first['fragment_record_sha256']
                       for piece in pieces)
            text = ''.join(piece['json_text'] for piece in pieces)
            assert hashlib.sha256(text.encode('ascii')).hexdigest() == first['fragment_record_sha256']
            output.append(json.loads(text))
            pos += count
        recovered.append((kind, output))
    assert export.digest(recovered) == manifest['dataset']
    return recovered


@pytest.fixture(scope='module')
def full_export(tmp_path_factory):
    output = tmp_path_factory.mktemp('receiver transfer with spaces') / 'full evidence'
    result = result_fixture(full=True)
    manifest = export.export_result(result, output)
    return result, output, manifest


def test_every_transfer_file_is_ascii_and_at_most_60000_characters(full_export):
    _, directory, manifest = full_export
    assert export.verify(directory) == manifest
    assert manifest['max_chars'] == 60000
    for path in directory.iterdir():
        raw = path.read_bytes()
        text = raw.decode('ascii')
        assert len(raw) == len(text) == len(text.encode('utf-16-le')) // 2
        assert len(raw) <= 60000 < 65535
        assert raw.endswith(b'\n')
        assert isinstance(json.loads(text), dict)
    kinds = [entry['kind'] for entry in manifest['files']]
    assert kinds == sorted(kinds, key=('summary', 'ranks', 'cores').index)
    assert manifest['send_first'] == [entry['file'] for entry in manifest['files']
                                      if entry['kind'] != 'cores']


def test_all_56_by_16_by_9_core_ticks_and_counters_roundtrip(full_export):
    source, directory, manifest = full_export
    recovered = recover_groups(directory, manifest)
    assert recovered == export.evidence_groups(source)
    groups = dict(recovered)
    assert len(groups['ranks']) == len(groups['cores']) == 16 * 9
    assert sum(len(sample['cores']) for sample in groups['cores']) == 56 * 16 * 9
    original = {(row['iteration'], row['rank']): row
                for rows in source['samples'].values() for row in rows}
    for sample in groups['cores']:
        row = original[(sample['iteration'], sample['rank'])]
        assert sample['order'] == row['order']
        assert sample['frequency_hz'] == row['frequency_hz']
        assert [value[0] for value in sample['cores']] == list(range(56))
        for value in sample['cores']:
            core, base_ticks, skip_ticks, noout_ticks, batches, payload = value
            actual = row['cores'][core]['arms']
            restored = {
                'baseline': (base_ticks, batches, payload),
                'skip-flags': (skip_ticks, 0, payload),
                'no-payload-output': (noout_ticks, batches, 0),
            }
            for name, values in restored.items():
                assert values == tuple(actual[name][key]
                                       for key in ('ticks', 'flag_batches', 'payload_bytes'))


def test_unicode_and_oversized_single_record_reassemble_without_loss(tmp_path):
    text = '汉字😀\\\"\n' * 11000
    result = result_fixture(long_text=text)
    output = tmp_path / 'unicode fragments'
    manifest = export.export_result(result, output)
    parts = [json.loads((output / item['file']).read_bytes()) for item in manifest['files']]
    fragments = [record for part in parts for record in part['records']
                 if 'fragment_record_sha256' in record]
    assert len(fragments) > 1
    assert max(record['fragments'] for record in fragments) > 1
    assert all((output / item['file']).stat().st_size <= 60000 for item in manifest['files'])
    assert recover_groups(output, manifest) == export.evidence_groups(result)
    assert export.verify(output) == manifest


def test_modified_part_fails_sha_verification(tmp_path):
    output = tmp_path / 'tampered'
    manifest = export.export_result(result_fixture(), output)
    file = output / manifest['files'][0]['file']
    data = file.read_bytes()
    # Preserve valid JSON and the byte count, forcing the SHA rather than size gate.
    file.write_bytes(data[:-1] + b' ')
    with pytest.raises(ValueError, match='checksum'):
        export.verify(output)


def test_missing_part_is_rejected(tmp_path):
    output = tmp_path / 'missing'
    manifest = export.export_result(result_fixture(), output)
    (output / manifest['files'][0]['file']).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        export.verify(output)


def test_existing_output_directory_is_not_overwritten(tmp_path):
    output = tmp_path / 'existing data'
    output.mkdir()
    sentinel = output / 'user data.txt'
    sentinel.write_text('keep this unchanged', encoding='utf-8')
    with pytest.raises(ValueError, match='exists'):
        export.export_result(result_fixture(), output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding='utf-8') == 'keep this unchanged'


def test_cli_verify_accepts_paths_containing_spaces(full_export, tmp_path):
    _, directory, manifest = full_export
    unrelated = tmp_path / 'unrelated working directory'
    unrelated.mkdir()
    completed = subprocess.run([sys.executable, str(HERE / 'receiver_export.py'),
                                str(directory), '--verify'], cwd=unrelated,
                               text=True, encoding='utf-8', capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert f'Verified {manifest["parts"]} parts' in completed.stdout


@pytest.mark.parametrize('limit', [-1, 0, 2047, 65536, 100000])
def test_invalid_character_limits_fail_without_output(tmp_path, limit):
    output = tmp_path / 'invalid limit'
    with pytest.raises(ValueError, match='max-chars'):
        export.export_result(result_fixture(), output, limit)
    assert not output.exists()


@pytest.mark.parametrize('limit', [2048, 65535])
def test_inclusive_character_limit_boundaries(tmp_path, limit):
    output = tmp_path / str(limit)
    manifest = export.export_result(result_fixture(), output, limit)
    assert export.verify(output) == manifest
    assert all(path.stat().st_size <= limit for path in output.iterdir())


def test_mismatched_counters_cannot_be_compacted(tmp_path):
    result = result_fixture(full=True)
    row = next(iter(result['samples'].values()))[0]
    row['cores'][0]['arms']['skip-flags']['flag_batches'] = 1
    with pytest.raises(ValueError, match='unequal operation counters'):
        export.export_result(result, tmp_path / 'bad counters')


def test_reordered_parts_are_rejected(tmp_path):
    output = tmp_path / 'reordered'
    manifest = export.export_result(result_fixture(), output, 2048)
    assert len(manifest['files']) >= 2
    altered = copy.deepcopy(manifest)
    altered['files'][0], altered['files'][1] = altered['files'][1], altered['files'][0]
    (output / '000-index.json').write_bytes(export.encoded(altered))
    with pytest.raises(ValueError, match='reordered'):
        export.verify(output)


def rewrite_with_local_checksums(directory, manifest, index, edit):
    """Simulate consistent local re-encoding without changing dataset identity."""
    entry = manifest['files'][index]
    path = directory / entry['file']
    part = json.loads(path.read_bytes())
    edit(part['records'])
    part['records_sha256'] = export.digest(part['records'])
    data = export.encoded(part)
    path.write_bytes(data)
    entry.update(bytes=len(data), characters=len(data), sha256=hashlib.sha256(data).hexdigest())
    (directory / '000-index.json').write_bytes(export.encoded(manifest))


def test_whole_dataset_sha_rejects_mutation_with_valid_local_checksums(tmp_path):
    output = tmp_path / 'local hashes still valid'
    manifest = export.export_result(result_fixture(), output)
    def alter(records):
        records[0]['data']['unexpected_edit'] = 'must change dataset hash'
    rewrite_with_local_checksums(output, manifest, 0, alter)
    with pytest.raises(ValueError, match='Dataset checksum'):
        export.verify(output)


@pytest.mark.parametrize('mutation', ['text', 'missing', 'duplicate_number'])
def test_fragment_sha_and_completeness_survive_local_rehashing(tmp_path, mutation):
    output = tmp_path / mutation
    manifest = export.export_result(result_fixture(long_text='汉字😀\\\"\n' * 11000), output)
    candidates = []
    for index, entry in enumerate(manifest['files']):
        part = json.loads((output / entry['file']).read_bytes())
        if any('fragment_record_sha256' in record for record in part['records']):
            candidates.append(index)
    assert candidates
    def alter(records):
        selected = next(i for i, item in enumerate(records) if 'fragment_record_sha256' in item)
        record = records[selected]
        if mutation == 'text':
            record['json_text'] = ('[' if record['json_text'][0] != '[' else '{') + record['json_text'][1:]
        elif mutation == 'missing':
            records.pop(selected)
        else:
            record['fragment'] = 99999
    rewrite_with_local_checksums(output, manifest, candidates[0], alter)
    with pytest.raises(ValueError, match='[Ff]ragment'):
        export.verify(output)


def test_missing_optional_intervals_are_not_invented(tmp_path):
    result = result_fixture()
    core = next(iter(result['samples'].values()))[0]['cores'][0]
    for arm in core['arms'].values():
        arm['first_tick'] = arm['last_tick'] = None
    output = tmp_path / 'unknown trace positions'
    manifest = export.export_result(result, output)
    groups = dict(recover_groups(output, manifest))
    assert 'optional_sampled_intervals' not in groups['cores'][0]
