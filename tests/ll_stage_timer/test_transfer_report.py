"""Transfer bounds, complete reconstruction and absence/identity guarantees."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ll_transfer_report', Path(__file__).with_name('transfer_report.py'))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def fixture_samples(count=2, rows=6, unicode_metadata=False):
    samples = []
    for iteration in range(count):
        records = []
        intervals = []
        for core in range(2):
            for i in range(rows):
                stage = 'KERNEL' if i == 0 else ('PREFIX_COMPLETE' if i == 1 else f'STAGE_{i}')
                records.append(dict(core=core, role='send_and_recv', initial_role='send_and_recv',
                                    stage=stage, total_ticks=1000 if i == 0 else 10+core+i,
                                    calls=3 if i else 1, time_measured=True))
                intervals.append(dict(core=core, stage=stage, sample=0,
                                      first_tick=100, last_tick=1100 if i == 0 else 110+core+i,
                                      total_ticks=1000 if i == 0 else 10+core+i))
            records.append(dict(core=core, role='send_and_recv', stage='TX_PACKETS' if core == 0 else 'RX_PACKETS',
                                total_ticks=0, calls=7, time_measured=False))
        meta = dict(rank=0, iteration=iteration, config=dict(B=1, M=32), revision='abc123',
                    topology='目标A5/通路' if unicode_metadata else 'A5')
        decoded = dict(schema=13, operation='dispatch', active_aiv=2, send_cores=1,
                       boundary_mode='observe', frequency_hz=1000000000,
                       records=records, intervals=intervals,
                       cores=[dict(core=c, omitted_intervals=rows) for c in range(2)])
        samples.append((meta, decoded))
    report = dict(world_size=1, schema=13, sampling=dict(diagnostic_iters=count),
                  configurations=dict(dispatch=dict(B=1, M=32)), communication=dict(derived='excluded'))
    return report, samples


def test_lossless_unicode_multiple_parts_and_bounds(tmp_path):
    import random
    report,samples=fixture_samples(count=4,rows=60,unicode_metadata=True)
    samples[0][0]['noise']=random.Random(17).randbytes(20000).hex()
    manifest=transfer.export_bundle(tmp_path,report,samples,limit=6000)
    directory=tmp_path/'transfer'
    assert manifest['series_counts']['compressed']>1
    assert manifest['export_complete'] and manifest['source_sample_coverage']['dispatch']['complete']
    for path in directory.iterdir():
        raw=path.read_bytes();text=raw.decode('utf-8')
        assert len(text)<=6000 and len(raw)+text.count('\n')<=6000
    context,restored=transfer.restore_samples(directory)
    assert restored==samples and context['configurations']==report['configurations']
    assert not list(directory.glob('priority_*')) and not list(directory.glob('full_*'))


def test_absent_null_and_large_metadata_preserved(tmp_path):
    report,samples=fixture_samples(count=1)
    samples[0][1]['records'][0]['optional_value']=None
    samples[0][0]['large_unicode']='长'*5000
    manifest=transfer.export_bundle(tmp_path,report,samples,limit=4096)
    _,restored=transfer.restore_samples(tmp_path/'transfer')
    assert restored==samples
    assert 'optional_value' not in restored[0][1]['records'][1]
    assert manifest['transmission_bytes_crlf']<manifest['transport']['raw_bytes']


@pytest.mark.parametrize('limit',[0,4095,65536,60000.0,True])
def test_illegal_limits(limit,tmp_path):
    report,samples=fixture_samples(count=1)
    with pytest.raises(ValueError,match='limit'):
        transfer.export_bundle(tmp_path,report,samples,limit=limit)
    assert not (tmp_path/'transfer').exists()


def test_corruption_missing_and_manifest_traversal_rejected(tmp_path):
    report,samples=fixture_samples(count=2,rows=80)
    manifest=transfer.export_bundle(tmp_path,report,samples)
    directory=tmp_path/'transfer';part=directory/manifest['parts'][0]
    original=part.read_text();index=original.index('DATA\n')+5
    part.write_text(original[:index]+('A' if original[index]!='A' else 'B')+original[index+1:])
    with pytest.raises(ValueError): transfer.restore_samples(directory)
    part.write_text(original);part.unlink()
    with pytest.raises(FileNotFoundError):transfer.restore_samples(directory)
    manifest['parts']=['../../outside.txt']
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='filename'):transfer.restore_samples(directory)


def test_completeness_is_not_inferred_from_lossless_export(tmp_path):
    report,samples=fixture_samples(count=2);report['world_size']=2
    manifest=transfer.export_bundle(tmp_path,report,samples)
    assert manifest['export_complete']
    assert manifest['source_sample_coverage']['dispatch']['complete'] is False
    assert manifest['all_observed_kernel_cores_complete']
    report,samples=fixture_samples(count=1);samples.append(copy.deepcopy(samples[0]))
    manifest=transfer.export_bundle(tmp_path/'duplicates',report,samples)
    assert manifest['source_sample_coverage']['dispatch']['complete'] is False
    assert transfer.restore_samples(tmp_path/'duplicates/transfer')[1]==samples


def test_big_ticks_profile_unknown_fields_and_compression_savings(tmp_path):
    report,samples=fixture_samples(count=32,rows=128)
    for meta,decoded in samples:
        decoded.update(schema=14,timer_profile='prefix',timer_profile_id=3,extra_parser_metadata={'unknown':'kept'})
        decoded['records'].append(dict(core=1,stage='PREFIX_READY_WAIT',calls=1,total_ticks=2**60+17,time_measured=True))
    manifest=transfer.export_bundle(tmp_path,report,samples)
    assert transfer.restore_samples(tmp_path/'transfer')[1]==samples
    assert manifest['transmission_bytes_crlf']<len(transfer.encode_samples(report,samples))*.15
    assert manifest['series_counts']['compressed']<10  # Old uncompressed fixture required >100 parts at4096.


def test_same_input_idempotent_and_other_bundle_not_overwritten(tmp_path):
    report,samples=fixture_samples(count=1)
    first=transfer.export_bundle(tmp_path,report,samples)
    assert transfer.export_bundle(tmp_path,report,samples)==first
    samples[0][0]['revision']='changed'
    with pytest.raises(ValueError,match='different bundle'):transfer.export_bundle(tmp_path,report,samples)


def test_mixed_profiles_not_aggregated(tmp_path):
    report,samples=fixture_samples(count=2)
    samples[0][1].update(schema=14,timer_profile='prefix',timer_profile_id=3)
    samples[1][1].update(schema=14,timer_profile='recv',timer_profile_id=5)
    with pytest.raises(ValueError,match='Mixed timer profiles'):transfer.export_bundle(tmp_path,report,samples)
    assert not (tmp_path/'transfer').exists()


def test_historical_uncompressed_bundle_is_readable(tmp_path):
    # v1 wire layout fixture; the current writer does not generate this format.
    report,samples=fixture_samples(count=1)
    records=list(transfer._full_records(report,samples,60000))
    bundle='old-schema13-capture'
    def save(name,value):
        text=transfer._json(value);(tmp_path/name).write_text(text)
        return dict(file=name,sha256=transfer._hash(text))
    part=save('full_001_of_001.txt',dict(bundle_id=bundle,part=1,parts=1,series='full',records=records))
    part.update(series='full',part=1)
    index=save('part_index_001_of_001.txt',dict(bundle_id=bundle,part=1,parts=1,series='index',records=[dict(kind='part_index',data=part)]))
    save('manifest.json',dict(format=transfer.LEGACY_FORMAT,bundle_id=bundle,part_index=[index],
        series_counts=dict(full=1),samples=1,full_record_count=len(records),full_payload_sha256=transfer._hash(transfer._json(records))))
    assert transfer.restore_samples(tmp_path)[1]==samples


def test_decode_rejects_missing_or_duplicate_rows():
    report,samples=fixture_samples(count=1)
    data=json.loads(transfer.encode_samples(report,samples))
    data['records'].append(data['records'][1])
    with pytest.raises(ValueError,match='Duplicate'):transfer.decode_samples(json.dumps(data).encode())
    data=json.loads(transfer.encode_samples(report,samples))
    data['records']=[r for r in data['records'] if r['kind']!='intervals']
    with pytest.raises(ValueError,match='row count'):transfer.decode_samples(json.dumps(data).encode())


def test_cli_restore_without_manifest_and_regenerate_trace(tmp_path):
    import subprocess,sys
    base=Path(__file__).resolve().parents[2]
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,base/path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    fixtures=load('restore_fixture','tests/ll_stage_timer/test_segments.py')
    timer=load('restore_timer','deep_ep/utils/ll_stage_timer.py')
    analysis=load('restore_analyzer','tests/ll_stage_timer/analyze.py')
    decoded=timer.parse_records(fixtures.raw(schema=14),'dispatch',1e9)
    # Synthetic parser fixture, not a device launch.
    meta=dict(world_size=1,rank=0,iteration=0,correctness='passed',config=dict(B=1,M=32),
              sampling=dict(diagnostic_iters=1),case_id=0,run_id='fixture')
    samples=[(meta,decoded)];report=analysis.analyze_samples(samples)
    manifest=transfer.export_bundle(tmp_path,report,samples)
    copied=tmp_path/'pasted.txt'
    copied.write_text('\n'.join((tmp_path/'transfer'/n).read_text() for n in reversed(manifest['parts'])))
    destination=tmp_path/'restored with spaces'
    subprocess.run([sys.executable,str(base/'tests/ll_stage_timer/transfer_report.py'),'restore',str(copied),
                    '--output-dir',str(destination)],cwd=tmp_path,check=True)
    restored=json.loads((destination/'decoded_samples.json').read_text())
    assert restored==[dict(metadata=meta,decoded=decoded)]
    assert json.loads((destination/'summary.json').read_text())==report
    trace=json.loads((destination/'trace.json').read_text())
    assert any(e.get('name')=='RECV_POLL' for e in trace['traceEvents'] if e.get('ph')=='X')
    assert any(e.get('name')=='process_name' for e in trace['traceEvents'])
    with pytest.raises(FileExistsError):transfer.write_restored(destination,{},samples,render=False)
