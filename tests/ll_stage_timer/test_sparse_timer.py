"""Schema15 fixed milestones, historical isolation and offline round trip."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


timer = load('sparse_timer_tested', 'deep_ep/utils/ll_stage_timer.py')


def raw(send=30, operation='dispatch', minimal=False):
    result = []
    for core in range(56):
        points = dict(ENTRY=100+core, INIT_DONE=200+core, WORK_DONE=3000+core)
        if operation == 'dispatch' and not minimal:
            if core < send:
                points.update(COUNTS_SENT=300+core, SEND_PREPARED=400+core, FIRST_SEND=500+core,
                              SEND_DONE=2000+core)
            else:
                points.update(COUNTS_RECEIVED=350+core, PREFIX_LOCAL_DONE=450+core,
                              PREFIX_READY=550+core, RECV_PREPARED=650+core,
                              FIRST_RECV_PROBE=750+core, FIRST_RECV_READY=850+core, RECV_DONE=2500+core)
        role = (3 if core < send else 2) if operation == 'dispatch' else (1 if core < send else 4)
        words = [0]*64
        words[:8] = [timer.MAGIC,15,1 if operation == 'dispatch' else 2,56,core,
                     1|(send<<8)|(role<<16)|(6<<24),0,0]
        for point, tick in points.items():
            i = timer.POINTS.index(point); words[6] |= 1<<i; words[8+i] = tick
        result.append([words[i:i+8] for i in range(0,64,8)])
    return result


def setword(data, core, offset, value):
    data[core][offset//8][offset%8] = value


def metadata(rank=0, iteration=0):
    return dict(rank=rank, iteration=iteration, world_size=1, correctness='passed',
                config=dict(B=1,M=32,H=3584,num_topk=8,num_experts=384,dtype='fp8'),
                sampling=dict(diagnostic_iters=1),run_id='sparse',case_id='B1',seed=1)


def test_sparse_raw_and_actual_trace_have_no_packet_metrics():
    decoded = timer.parse_records(raw(), 'dispatch', 1e9)
    assert decoded['recording'] == 'sparse-milestones'
    assert decoded['timer_profile'] == 'sparse'
    assert decoded['recv_cores'] == 26
    assert 'dispatch_recv_cores' not in decoded  # Raw header cannot distinguish M32 from shared reception.
    assert len(decoded['milestones']) == 30*7+26*10
    assert all(row['calls'] == 1 for row in decoded['records'])
    assert not {'TX_PACKETS','RX_PACKETS','RX_READY','RX_PROBED'} & {r['stage'] for r in decoded['records']}
    trace = timer.chrome_trace([dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=metadata())])
    assert trace['metadata']['all_calls_in_metrics'] is False
    assert trace['metadata']['sampled_calls_per_stage'] is None
    assert sum(r['ph']=='i' for r in trace['traceEvents']) == len(decoded['milestones'])
    assert all(r['args']['interval_kind']=='same-core milestone span' for r in trace['traceEvents'] if r['ph']=='X')


@pytest.mark.parametrize('operation', ['dispatch','combine'])
def test_three_required_points_suffice_and_absent_points_stay_absent(operation):
    decoded = timer.parse_records(raw(operation=operation,minimal=True), operation)
    assert len(decoded['records']) == 112
    assert {r['stage'] for r in decoded['records']} == {'INIT','KERNEL'}
    assert all('total_us' not in row for row in decoded['records'])


@pytest.mark.parametrize('offset,value', [(1,14),(3,55),(4,1),(5,0),(6,1<<14),(7,1),
                                         (8,-1),(8,True),(8,1<<64),(9,99),(21,199),(22,1)])
def test_invalid_header_and_timestamp_words_rejected(offset,value):
    data=raw();setword(data,0,offset,value)
    with pytest.raises(ValueError): timer.parse_records(data,'dispatch')


def test_missing_bit_nonzero_and_dependency_reversal_rejected():
    data=raw();data[0][0][6] &= ~(1<<timer.POINTS.index('SEND_PREPARED'))
    with pytest.raises(ValueError,match='without valid'):timer.parse_records(data,'dispatch')
    data=raw();setword(data,0,8+timer.POINTS.index('FIRST_SEND'),350)
    with pytest.raises(ValueError,match='dependency'):timer.parse_records(data,'dispatch')
    data=raw();data[0][0][6] &= ~1;setword(data,0,8,0)
    with pytest.raises(ValueError,match='required'):timer.parse_records(data,'dispatch')


def test_interleaved_send_receive_and_unwritten_padding_are_legal():
    data=raw()
    # A sending core later joins reception: prefix is later than first send.
    for point,tick in [('PREFIX_READY',2100),('RECV_PREPARED',2200),('FIRST_RECV_PROBE',2300),
                       ('FIRST_RECV_READY',2400),('RECV_DONE',2500)]:
        idx=timer.POINTS.index(point);data[0][0][6] |= 1<<idx;setword(data,0,8+idx,tick)
    setword(data,0,63,123)  # Native only publishes first 24 words.
    decoded=timer.parse_records(data,'dispatch')
    assert any(r['core']==0 and r['point']=='PREFIX_READY' and r['tick']==2100 for r in decoded['milestones'])


def test_mixed_schema_split_and_trace_profile_rejected():
    for field,value in [(1,14),(5,1|(29<<8)|(2<<16)|(6<<24))]:
        data=raw();setword(data,55,field,value)
        with pytest.raises(ValueError):timer.parse_records(data,'dispatch')
    legacy=load('sparse_old_fixture','tests/ll_stage_timer/test_segments.py')
    saved=[dict(raw=x,operation='dispatch',frequency_hz=1e9) for x in [raw(),legacy.raw(schema=14)]]
    with pytest.raises(ValueError,match='Mixed segment schemas'):timer.chrome_trace(saved)


def test_roundtrip_trace_analysis_and_summary_preserve_sparse_meaning():
    transfer=load('sparse_transfer','tests/ll_stage_timer/transfer_report.py')
    decoded=timer.parse_records(raw(),'dispatch',1e9)
    source=[(metadata(),decoded)]
    context,samples=transfer.decode_samples(transfer.encode_samples(dict(world_size=1),source))
    assert samples==source
    trace_module=load('sparse_transfer_trace','tests/ll_stage_timer/transfer_trace.py')
    trace=trace_module.trace_from_decoded(samples)
    assert trace['metadata']['all_calls_in_metrics'] is False
    assert sum(r['ph']=='i' for r in trace['traceEvents'])==len(decoded['milestones'])
    analyzer=load('sparse_analyze','tests/ll_stage_timer/analyze.py')
    report=analyzer.analyze_samples(samples)
    assert report['communication']['status']=='not_collected_sparse_milestones'
    assert not report['communication']['samples']
    metric=load('sparse_triage_metrics','tests/ll_stage_timer/triage_metrics.py')
    summary=metric.summarize_case({},samples)
    activity={(g,q):v for g,q,v in summary['activity']}
    assert activity['S','first_send_cores']==[1,30,30,30]
    assert activity['R','first_probe_cores']==[1,26,26,26]
    assert activity['R','first_ready_cores']==[1,26,26,26]
    assert activity['S','tx_packets_observed']==[0,None,None,None]
    assert activity['R','ready_fraction']==[0,None,None,None]


def test_seven_bs_summary_is_one_small_file_and_no_old_counter_claims():
    metric=load('sparse_summary_metrics','tests/ll_stage_timer/triage_metrics.py')
    report_module=load('sparse_report','tests/ll_stage_timer/triage_report.py')
    cases=[]
    for b in [1,2,4,8,16,24,32]:
        summary=metric.summarize_case({},[(metadata(),timer.parse_records(raw(),'dispatch',1e9))])
        summary['identity']['B']=b
        cases.append(dict(id=f'C{len(cases)+1:03d}',metrics=summary,source='analysis/sparse/transfer',provenance={}))
    content=report_module.render(dict(suite='seven-bs',cases=cases,matrix={},off=[],performance=[],issues=[],complete=True))
    assert len(content.replace('\n','\r\n').encode())<20000
    assert 'Packet totals, packet balance and ready/probe ratios are NOT COLLECTED' in content
    assert 'FIRST_PROBE_TO_FIRST_READY' in content
    assert 'non-envelope' not in content


def test_analysis_directory_writes_only_sparse_local_reports(tmp_path):
    analyzer=load('sparse_analyze_directory','tests/ll_stage_timer/analyze.py')
    source=tmp_path/'source';source.mkdir();output=tmp_path/'analysis'
    decoded=timer.parse_records(raw(),'dispatch',1e9)
    (source/'sample_000.json').write_text(json.dumps(dict(**decoded,raw=raw(),metadata=metadata())))
    analyzer.process_directory(source,output,1e9,0)
    assert (output/'milestones.csv').is_file()
    assert not (output/'communication_bottleneck.md').exists()
    transfer=load('sparse_roundtrip','tests/ll_stage_timer/transfer_report.py')
    _,restored=transfer.restore_samples(output/'transfer')
    assert restored[0][1]['milestones']==decoded['milestones']


def test_sparse_precise_followup_preserves_one_core_and_send_budget(tmp_path):
    extract=load('sparse_extract','tests/ll_stage_timer/triage_extract.py')
    transfer=load('sparse_extract_transfer','tests/ll_stage_timer/transfer_report.py')
    suite=tmp_path/'suite'
    decoded=timer.parse_records(raw(),'dispatch',1e9)
    samples=[(metadata(),decoded)]
    transfer.export_bundle(suite/'analysis/sparse/case_000',dict(world_size=1),samples)
    manifest=extract.export_selection(suite,'sparse',1,tmp_path/'selection',rank=0,iteration=0,core=30)
    context,selected=transfer.restore_samples(tmp_path/'selection')
    assert manifest['send_budget']['issued']==1
    assert manifest['send_budget']['remaining']==8  # one summary slot remains reserved
    assert len(manifest['send_first'])==1
    assert all(row['core']==30 for row in selected[0][1]['milestones'])
    rendered=transfer.prepare_restored(context,selected)
    trace=json.loads(rendered['trace.json'])
    assert trace['metadata']['export_selection']['selectors']['core']==30
    assert all(row['args']['core']==30 for row in trace['traceEvents'] if row['ph']=='i')
    again=extract.export_selection(suite,'sparse',1,tmp_path/'same',rank=0,iteration=0,core=30)
    assert not again['send_first']
    assert again['send_budget']['issued']==1


def test_combine_does_not_claim_uninstrumented_send_activity():
    metric=load('sparse_combine_metrics','tests/ll_stage_timer/triage_metrics.py')
    decoded=timer.parse_records(raw(operation='combine'),'combine',1e9)
    summary=metric.summarize_case({},[(metadata(),decoded)])
    assert not any(name in ('first_send_cores','first_probe_cores','first_ready_cores') for _,name,_ in summary['activity'])


@pytest.mark.parametrize('frequency', [True,0,-1,float('nan'),float('inf'),'1000000000'])
def test_sparse_frequency_must_be_real_positive(frequency):
    with pytest.raises(ValueError,match='frequency'):
        timer.parse_records(raw(),'dispatch',frequency)


def test_sparse_raw_trace_rejects_mixed_shape_identity():
    a=metadata();b=metadata(rank=1);b['config']['B']=2
    samples=[dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=m) for m in (a,b)]
    with pytest.raises(ValueError,match='Mixed sparse capture identities'):timer.chrome_trace(samples)


def st_profile_metadata(rank=0, iteration=0, iterations=3):
    """Use the real ST configuration/comparison contract plus its sample fields."""
    st=load('sparse_profile_st','tests/elastic/test_ep_low_latency.py')
    args=st.parse_args(['--num-processes','16','--num-tokens','1',
                       '--num-max-tokens-per-rank','32','--hidden','3584',
                       '--num-topk','8','--num-experts','384','--dtype','fp8',
                       '--ll-stage-timer','--ll-profile-capture','--ll-timer-iters',str(iterations)])
    args.runtime_identity=dict(implementation='g4-small-b-parallel-prefix',transport='mte',
        active_aiv=56,send_cores=30,recv_cores=26,logical_group=4,core_protocol=19030,
        source_sha256='a'*64,stage_timer=True,stage_timer_schema=15,timer_profile='sparse',
        timer_profile_id=6,stage_timer_slots=8,stage_timer_fields=8,
        stage_timer_points=list(timer.POINTS),stage_timer_recording='sparse-milestones',
        profile_capture_requested=True,rdma_compiled=False)
    return dict(rank=rank,world_size=16,iteration=iteration,correctness='passed',
        comparison=st.comparison_metadata(args,16,32),run_id='sparse_profile',seed=args.seed,case_id=0,
        config={**st.case_config(args,32),'logical_group_size':4},
        sampling=dict(after_perf=True,warmup=args.warmup,settle_iters=args.num_iters,
                      diagnostic_iters=iterations,delay_rows=args.delay_rows,delay_cols=args.delay_cols,
                      group_backend=args.group_backend,expert_timer=False),
        active_capture_profile=dict(status='available',all_ranks_status='available',
            measurement='instrumented_full_device_kernel_task_us',
            kernel_us=38.5+rank*.01+iteration*.1,reason=None))


def test_st_multi_rank_multi_iteration_profiler_observations_are_not_trace_identity():
    base=st_profile_metadata()
    data=raw()
    samples=[]
    for rank in range(16):
        for iteration in range(3):
            meta=copy.deepcopy(base)
            meta.update(rank=rank,iteration=iteration)
            meta['active_capture_profile']['kernel_us']=38.5+rank*.01+iteration*.1
            samples.append(dict(raw=data,operation='dispatch',frequency_hz=1e9,metadata=meta))
    before=copy.deepcopy(samples)
    trace=timer.chrome_trace(samples)
    assert len(trace['metadata']['cores'])==48
    assert {(r['rank'],r['iteration']) for r in trace['metadata']['cores']}=={
        (rank,iteration) for rank in range(16) for iteration in range(3)}
    assert samples==before  # Retain every measured result in the source metadata.


def test_missing_active_profiler_result_does_not_discard_valid_sparse_milestones():
    a=st_profile_metadata();b=copy.deepcopy(a);b['rank']=1
    b['active_capture_profile'].update(status='unavailable',all_ranks_status='unavailable',
        kernel_us=None,reason='No exported kernel_details.csv for this rank')
    samples=[dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=m) for m in (a,b)]
    assert len(timer.chrome_trace(samples)['metadata']['cores'])==2
    assert b['active_capture_profile']['status']=='unavailable'


@pytest.mark.parametrize('field,value',[
    ('config.B',2),('comparison.source_sha256','b'*64),('comparison.source_revision','different'),
    ('comparison.send_cores',29),('comparison.core_protocol',19029),
    ('comparison.timer_profile','full'),('comparison.timer_profile_id',0),
    ('comparison.stage_timer_schema',14),('comparison.profile_capture',False),
    ('sampling.diagnostic_iters',4),('active_capture_profile.measurement','different_measurement'),
])
def test_st_trace_keeps_real_config_build_profile_guards_and_names_mismatch(field,value):
    a=st_profile_metadata();b=copy.deepcopy(a);b['rank']=1
    parent,key=field.split('.')
    b[parent][key]=value
    samples=[dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=m) for m in (a,b)]
    with pytest.raises(ValueError) as failure:
        timer.chrome_trace(samples)
    assert str(failure.value)==f'Mixed sparse capture identities: {field} (rank=1, iteration=0)'
    assert 'kernel_us' not in str(failure.value)


def test_st_trace_rejects_raw_split_and_frequency_mismatch():
    a=st_profile_metadata();b=copy.deepcopy(a);b['rank']=1
    samples=[dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=a),
             dict(raw=raw(send=29),operation='dispatch',frequency_hz=1e9,metadata=b)]
    with pytest.raises(ValueError,match='Mixed sparse capture identities: send_cores'):
        timer.chrome_trace(samples)
    samples[1]['raw']=raw();samples[1]['frequency_hz']=5e8
    with pytest.raises(ValueError,match='Mixed sparse capture identities: frequency_hz'):
        timer.chrome_trace(samples)
    samples[1]['frequency_hz']=1_000_000_000  # Same frequency may be serialized as an integer.
    assert len(timer.chrome_trace(samples)['metadata']['cores'])==2


def test_st_multi_rank_profiled_directory_analysis_retains_per_sample_results(tmp_path):
    analyzer=load('sparse_profile_directory','tests/ll_stage_timer/analyze.py')
    source=tmp_path/'source';source.mkdir()
    base=st_profile_metadata(iterations=2)
    for rank in range(16):
        for iteration in range(2):
            meta=copy.deepcopy(base);meta.update(rank=rank,iteration=iteration)
            meta['active_capture_profile']['kernel_us']=40+rank*.1+iteration*.01
            path=source/f'rank_{rank:03d}'/f'sample_{iteration:04d}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=1e9,metadata=meta)))
    output=tmp_path/'analysis'
    analyzer.process_directory(source,output,1e9,0)
    trace=json.loads((output/'trace.json').read_text())
    assert len(trace['metadata']['cores'])==16
    transfer=load('sparse_profile_restore','tests/ll_stage_timer/transfer_report.py')
    _,samples=transfer.restore_samples(output/'transfer')
    assert len(samples)==32
    assert {m['active_capture_profile']['kernel_us'] for m,_ in samples}=={
        40+rank*.1+iteration*.01 for rank in range(16) for iteration in range(2)}


def test_off_comparison_allows_unused_layout_change_but_diagnostics_reject_it():
    parity=load('sparse_compare_parity','tests/ll_stage_timer/test_parity.py')
    baseline=parity.identity()
    baseline.update(core_protocol=19030,send_cores=30,recv_cores=26,active_aiv=56,
                    stage_timer=False,stage_capture=False,expert_capture=False,
                    stage_timer_schema=14,timer_profile='full',timer_profile_id=0)
    candidate=dict(baseline,stage_timer_schema=15,timer_profile='sparse',timer_profile_id=6,
                   stage_timer_slots=8,stage_timer_fields=8,stage_timer_points=list(timer.POINTS),
                   stage_timer_recording='sparse-milestones',profile_capture_requested=False)
    parity.comparison.check_identity(baseline,candidate)
    def perf(identity,value):
        return dict(status='completed',results=[dict(case_id=0,correctness='passed',comparison=identity,
                    dispatch=dict(ranks=[dict(rank=0,mean_us=value)]))])
    assert parity.comparison.compare(perf(baseline,40),perf(candidate,39),'perf')[0]['candidate_over_baseline']==pytest.approx(39/40)
    for flag in ('stage_timer','stage_capture','expert_capture'):
        changed=dict(candidate,**{flag:True})
        with pytest.raises(ValueError,match='different stage schemas'):
            parity.comparison.check_identity(baseline,changed)
    # Diagnostic reports must not bypass their schema guard through OFF flags.
    rows=[dict(operation='dispatch',rank=0,core=0,stage='INIT',unit='us',mean=1)]
    a=dict(schema=14,timer_profile='sparse',comparison=baseline,per_rank_core_stage=rows)
    b=dict(schema=15,timer_profile='sparse',comparison=candidate,per_rank_core_stage=rows)
    with pytest.raises(ValueError,match='different stage schemas'):
        parity.comparison.compare(a,b,'stage')


def test_same_sparse_schema_different_profiles_remain_incomparable():
    parity=load('sparse_compare_profiles','tests/ll_stage_timer/test_parity.py')
    identity=parity.identity()
    identity.update(core_protocol=19030,send_cores=30,recv_cores=26,active_aiv=56,
                    stage_timer_schema=15,timer_profile='sparse',timer_profile_id=6)
    with pytest.raises(ValueError,match='Incomparable settings'):
        parity.comparison.check_identity(identity,dict(identity,timer_profile='full',timer_profile_id=0))
