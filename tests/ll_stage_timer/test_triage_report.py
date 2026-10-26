"""First-pass output is one readable file even for the 35-case user matrix."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    spec=importlib.util.spec_from_file_location('test_first_'+name,Path(__file__).with_name(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


triage=module('triage_report')


def worst_matrix():
    fixture=module('test_triage_metrics')
    metric=module('triage_metrics')
    cases=[];performance=[];off=[]
    bs=[1,2,4,8,16,24,32]
    names=module('test_segments').timer.STAGES
    for b in bs:
        off.append(dict(B=b,route='normal',us=17.5+b/8,correctness='passed'))
        for profile in ('overview','prepare','prefix','send','recv'):
            result=metric.summarize_case({},[fixture.sample(i,r,profile=profile) for r in range(2) for i in range(5)])
            result['identity'].update(B=b,M=32,H=3584,K=8,E=384,dtype='fp8',world_size=16,send_cores=28,recv_cores=28,active_aiv=56)
            result['coverage'].update(samples=80,expected_rank_iterations=80,complete_rank_iterations=True,
                                     complete_kernel_cores=True,kernel_core_count=[80,56,56,56])
            # Every known stage appears; real selective captures usually have fewer.
            result['stages']=[[name,g,[2240,1.234567,100.25,998.5],[80,15.625,60.375,998.5],[2240,3,100,999]]
                              for name in names for g in ('S','R')]
            for g in ('S','R'):
                result['slowest'][g]['phase_totals']=[[name,998.5,999,998500] for name in names]
            case=dict(id=f'C{len(cases)+1:03d}',metrics=result,
                      source=f'analysis/{profile}/case_{b:05d}/normal/ll_stage_timer/transfer',
                      source_bundle_id='a'*20,provenance=dict(source_revision='a'*40,source_sha256='b'*64,
                          implementation='g4-small-b-parallel-prefix',core_protocol=19028,transport='mte',logical_group=4))
            cases.append(case)
            performance.append(dict(B=b,profile=profile,status='available',off_max_rank_mean_us=18.5,
                                    active_max_rank_mean_us=66.125,ratio=66.125/18.5,delta_us=47.625))
    return dict(format=triage.FORMAT,suite='35-cases-2800-samples',matrix=dict(B=bs,M=32,H=3584,K=8,E=384,dtype='fp8'),
                cases=cases,off=off,performance=performance,issues=[],complete=True)


def test_35_cases_2800_samples_all_have_essential_rows_in_one_file():
    data=worst_matrix()
    text=triage.render(data)
    assert len(data['cases'])==35 and sum(c['metrics']['coverage']['samples'] for c in data['cases'])==2800
    assert len(text)<=60000 and len(text.replace('\n','\r\n').encode())<=60000
    assert 'SEND THIS ONE FILE ONLY' in text
    assert 'Full case detail' not in text
    for case in data['cases']:
        assert case['id']+'/S |' in text and case['id']+'/R |' in text
        assert f"{case['id']} | {case['metrics']['identity']['B']} | {case['metrics']['identity']['profile']}" in text
    assert 'entry_to_kernel_end' in text
    assert 'top 0 ' not in text


def test_summary_preserves_missing_and_probe_overhead_warnings():
    data=worst_matrix()
    data['cases'][0]['metrics']['latencies']=[]
    data['cases'][0]['metrics']['activity']=[]
    data['performance'][0]=dict(B=1,profile='overview',status='unavailable')
    text=triage.render(data)
    assert 'C001/S | - | - | - | - | - | -' in text
    assert '1 | overview | - | - | - | - | unavailable' in text
    assert 'ALL supplied samples' in text and 'Probe overhead' in text


def test_stream_summary_reports_effective_path_and_slice_ready_scope():
    data = worst_matrix()
    for case in data['cases']:
        case['provenance'].update(implementation='cb59-stream-recv', core_protocol=22028,
                                  dispatch_impl='cb59-stream', effective_dispatch_impl='cb59-stream')
    text = triage.render(data)
    assert 'this owner slice can receive' in text
    assert 'not full-rank prefix readiness' in text
    assert 'effective_dispatch_impl' in text and 'cb59-stream' in text
    assert len(text.replace('\n', '\r\n').encode()) <= 60000


def test_pair_summary_reports_pair_ready_scope_without_claiming_stream_ownership():
    data=worst_matrix()
    for case in data['cases']:
        case['provenance'].update(implementation='cb59-pair-recv',core_protocol=23028,
                                  dispatch_impl='cb59-pair',effective_dispatch_impl='cb59-pair')
    text=triage.render(data)
    assert 'adjacent-pair range exchange' in text
    assert 'not full-rank prefix readiness' in text
    assert 'this owner slice can receive' not in text
    assert len(text.replace('\n','\r\n').encode())<=60000


def test_export_report_replaces_only_pointer_and_keeps_all_local_metrics(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    (suite/'SEND_FIRST.txt').write_text('old archive says transmit887files')
    (suite/'old-parts.txt').write_text('keep archive')
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    receipt={}
    result=triage.export_report(suite,receipt=receipt)
    assert result==suite/'SEND_FIRST.txt'
    saved=list((suite/'triage').glob('*/triage_summary.json'))
    assert len(saved)==1 and json.loads(saved[0].read_text())==data
    assert (suite/'old-parts.txt').read_text()=='keep archive'
    assert result.stat().st_size<=60000
    # The prior producer's root summary is conservatively one earlier send.
    assert receipt['issued']==2 and receipt['new_count']==1 and receipt['remaining']==8
    again={}
    assert triage.export_report(suite,receipt=again)==result
    assert again['issued']==2 and again['new_count']==0


def test_collect_reads_existing_cases_and_all_rounds_not_suite_archive(tmp_path):
    suite=tmp_path/'suite';(suite/'node_000').mkdir(parents=True)
    plan=dict(matrix=dict(B=[1],M=32),stages=[dict(name='prepare',timer=True)])
    (suite/'node_000/plan.json').write_text(json.dumps(plan))
    fixture=module('test_triage_metrics');transfer=module('transfer_report')
    samples=[fixture.sample(i,r,kernel=1000 if i<4 else 9000) for i in range(5) for r in range(2)]
    context=dict(world_size=2,sampling=dict(diagnostic_iters=5),configurations=dict(dispatch=samples[0][0]['config']),
                 comparison=dict(seed=0, correctness_repeats=3, profile_seed=0,
                                 profile_route_sha256='a'*64, profile_route_fingerprint_schema=1))
    transfer.export_bundle(suite/'analysis/prepare/case_00000',context,samples)
    # The huge previous suite archive is deliberately unreadable and ignored.
    (suite/'transfer').mkdir();(suite/'transfer/manifest.json').write_text('not used')
    data=triage.collect(suite)
    assert len(data['cases'])==1
    assert data['cases'][0]['metrics']['coverage']['samples']==10
    assert data['cases'][0]['metrics']['slowest']['R']['iteration']==4
    assert data['cases'][0]['provenance']['profile_seed'] == 0
    assert data['cases'][0]['provenance']['profile_route_sha256'] == 'a'*64
    assert data['complete'] is False and any('Missing OFF' in x for x in data['issues'])
    text=triage.render(data)
    assert 'INCOMPLETE' in text and 'C001' in text
    assert '"profile_seed":0' in text and '"profile_route_sha256":"' + 'a'*64 in text


def test_empty_focus_is_rejected_without_overwriting_existing_report(tmp_path,monkeypatch):
    import pytest
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    (suite/'SEND_FIRST.txt').write_text('keep previous')
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    with pytest.raises(ValueError,match='No parsed cases'):
        triage.export_report(suite,bs=[999])
    assert (suite/'SEND_FIRST.txt').read_text()=='keep previous'
    triage.export_report(suite,bs=[1],profiles=['prefix'])
    text=(suite/'SEND_FIRST.txt').read_text()
    assert 'Selection: {"B":[1],"profiles":["prefix"]}' in text
    assert 'cases=1;' in text


def test_summary_receipt_is_idempotent_and_changed_content_consumes_one_send(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    first={};path=triage.export_report(suite,receipt=first)
    raw=path.read_bytes()
    assert first['issued']==1 and first['new_count']==1 and first['remaining']==9
    second={};triage.export_report(suite,receipt=second)
    assert path.read_bytes()==raw
    assert second['issued']==1 and second['new_count']==0 and second['remaining']==9
    assert second['already_issued']==[hashlib.sha256(raw).hexdigest()]
    data['issues'].append('New evidence changes the necessary summary')
    third={};triage.export_report(suite,receipt=third)
    assert path.read_bytes()!=raw
    assert third['issued']==2 and third['new_count']==1 and third['remaining']==8
    ledger=json.loads((suite/'triage/send_budget.json').read_text())
    assert len(ledger['artifacts'])==2 and all(item['kind']=='summary' for item in ledger['artifacts'])


def test_summary_change_over_budget_preserves_previous_report_and_artifacts(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    target=triage.export_report(suite)
    quota=module('transfer_budget')
    with quota.transaction(suite,'detail',[hashlib.sha256(str(i).encode()).hexdigest() for i in range(9)]):
        pass
    prior={str(path.relative_to(suite)):path.read_bytes() for path in suite.rglob('*') if path.is_file()}
    data['issues'].append('This change requires an eleventh send')
    receipt={'unchanged':True}
    with pytest.raises(ValueError,match='limit is 10'):
        triage.export_report(suite,receipt=receipt)
    after={str(path.relative_to(suite)):path.read_bytes() for path in suite.rglob('*') if path.is_file()}
    assert after==prior and target.read_bytes()==prior['SEND_FIRST.txt']
    assert receipt=={'unchanged':True}
    # Reissuing the exact old summary remains possible at the hard limit.
    data['issues'].pop()
    receipt={};triage.export_report(suite,receipt=receipt)
    assert receipt['new_count']==0 and receipt['issued']==10 and receipt['remaining']==0


def test_corrupt_budget_refuses_summary_without_overwriting_prior_content(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    target=triage.export_report(suite)
    original=target.read_bytes()
    ledger=suite/'triage/send_budget.json';ledger.write_text('{broken json')
    directories=set((suite/'triage').iterdir())
    data['issues'].append('Updated data must not reset a corrupt budget')
    with pytest.raises(ValueError,match='Invalid send budget'):
        triage.export_report(suite)
    assert target.read_bytes()==original
    assert ledger.read_text()=='{broken json'
    assert set((suite/'triage').iterdir())==directories


def test_existing_identical_untracked_summary_is_adopted_without_another_send(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    content=triage.render(data).encode('utf-8')
    (suite/'SEND_FIRST.txt').write_bytes(content)
    receipt={};target=triage.export_report(suite,receipt=receipt)
    assert target.read_bytes()==content
    assert receipt['new_count']==0 and receipt['issued']==1 and receipt['remaining']==9
    assert (suite/'triage/send_budget.json').is_file()
