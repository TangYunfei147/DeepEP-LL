"""First-pass output is one readable file even for the 35-case user matrix."""
import copy
import importlib.util
import json
from pathlib import Path


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


def test_export_report_replaces_only_pointer_and_keeps_all_local_metrics(tmp_path,monkeypatch):
    data=worst_matrix();suite=tmp_path/'suite';suite.mkdir()
    (suite/'SEND_FIRST.txt').write_text('old archive says transmit887files')
    (suite/'old-parts.txt').write_text('keep archive')
    monkeypatch.setattr(triage,'collect',lambda p:copy.deepcopy(data))
    result=triage.export_report(suite)
    assert result==suite/'SEND_FIRST.txt'
    saved=list((suite/'triage').glob('*/triage_summary.json'))
    assert len(saved)==1 and json.loads(saved[0].read_text())==data
    assert (suite/'old-parts.txt').read_text()=='keep archive'
    assert result.stat().st_size<=60000
    assert triage.export_report(suite)==result


def test_collect_reads_existing_cases_and_all_rounds_not_suite_archive(tmp_path):
    suite=tmp_path/'suite';(suite/'node_000').mkdir(parents=True)
    plan=dict(matrix=dict(B=[1],M=32),stages=[dict(name='prepare',timer=True)])
    (suite/'node_000/plan.json').write_text(json.dumps(plan))
    fixture=module('test_triage_metrics');transfer=module('transfer_report')
    samples=[fixture.sample(i,r,kernel=1000 if i<4 else 9000) for i in range(5) for r in range(2)]
    context=dict(world_size=2,sampling=dict(diagnostic_iters=5),configurations=dict(dispatch=samples[0][0]['config']))
    transfer.export_bundle(suite/'analysis/prepare/case_00000',context,samples)
    # The huge previous suite archive is deliberately unreadable and ignored.
    (suite/'transfer').mkdir();(suite/'transfer/manifest.json').write_text('not used')
    data=triage.collect(suite)
    assert len(data['cases'])==1
    assert data['cases'][0]['metrics']['coverage']['samples']==10
    assert data['cases'][0]['metrics']['slowest']['R']['iteration']==4
    assert data['complete'] is False and any('Missing OFF' in x for x in data['issues'])
    text=triage.render(data)
    assert 'INCOMPLETE' in text and 'C001' in text


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
