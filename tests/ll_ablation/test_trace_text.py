import copy
import json
from pathlib import Path
import random
import string
import subprocess
import sys

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import trace_text as transfer
from receiver_export import export_result
from test_receiver_export import result_fixture
from test_receiver_breakdown import report_directory


def test_legacy_files_roundtrip_and_cli_show_has_only_one_block(tmp_path):
    source=tmp_path/'source with spaces';export_result(result_fixture(),source)
    (source/'private.env').write_text('not part of the experiment')
    archive=tmp_path/'archive'
    index=transfer.export_text(source,archive)
    parts=[transfer.show_part(archive,n) for n in range(1,len(index['parts'])+1)]
    restored=tmp_path/'restored'
    status,rendered=transfer.restore_text(parts,restored,render=False)
    assert status['status']=='complete' and not rendered
    assert not (restored/'private.env').exists()
    for record in index['files']:
        assert (source/record['name']).read_bytes()==(restored/record['name']).read_bytes()
    result=subprocess.run([sys.executable,str(HERE/'trace_text.py'),'show',str(archive),'--part','1'],
                          cwd=tmp_path,capture_output=True,check=True)
    assert result.stdout.decode('ascii').replace('\r\n','\n')==parts[0]


def test_full_raw_run_roundtrip_keeps_all_captures_and_generates_trace(tmp_path):
    source=tmp_path/'raw';source.mkdir();report_directory.__wrapped__(source)
    (source/'ignore.json').write_text('{"private":"not selected"}')
    archive=tmp_path/'raw text'
    index=transfer.export_text(source,archive)
    assert index['source_kind']=='receiver-run'
    assert len(index['files'])==50  # 16ranks *3captures +2identity files
    restored=tmp_path/'raw restored'
    texts=[transfer.show_part(archive,n) for n in range(1,len(index['parts'])+1)]
    status,rendered=transfer.restore_text(texts,restored)
    assert status['status']=='complete' and rendered
    assert not (restored/'ignore.json').exists()
    for record in index['files']:
        assert (source/record['name']).read_bytes()==(restored/record['name']).read_bytes()
    traces=[json.loads(p.read_text()) for p in rendered if p.suffix=='.json']
    assert any('traceEvents' in trace and trace['traceEvents'] for trace in traces)


def test_missing_parts_and_existing_outputs_do_not_write(tmp_path):
    rng=random.Random(5)
    source=tmp_path/'source'
    export_result(result_fixture(long_text=''.join(rng.choices(string.ascii_letters,k=12000))),source)
    archive=tmp_path/'text';index=transfer.export_text(source,archive,1024)
    assert len(index['parts'])>1
    texts=[transfer.show_part(archive,n) for n in range(1,len(index['parts'])+1)]
    out=tmp_path/'missing'
    with pytest.raises(ValueError,match='Missing parts'):transfer.restore_text(texts[:-1],out,False)
    assert not out.exists()
    with pytest.raises(ValueError,match='exists'):transfer.export_text(source,archive)
    source_before={p.name:p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError,match='exists'):transfer.restore_text(texts,source,False)
    assert source_before=={p.name:p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize('name',['../escape.json','/absolute.json','C:/escape.json','x\\escape.json',
                                 'a/../escape.json','a//b.json','CON.json','a./b.json','rendered/x.json'])
def test_restore_rejects_unsafe_paths(name):
    with pytest.raises(ValueError,match='Invalid restored'):transfer.safe_name(name)


def test_file_hash_and_duplicate_path_validation(tmp_path):
    source=tmp_path/'source';export_result(result_fixture(),source)
    bundle=transfer.collect(source)
    modified=copy.deepcopy(bundle);modified['files'][0]['text']+=' '
    with pytest.raises(ValueError,match='checksum'):transfer.validate_bundle(modified)
    duplicated=copy.deepcopy(bundle);duplicated['files'].append(duplicated['files'][0])
    with pytest.raises(ValueError,match='Duplicate'):transfer.validate_bundle(duplicated)
