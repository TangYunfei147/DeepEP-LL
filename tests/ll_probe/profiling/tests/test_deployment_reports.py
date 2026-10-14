"""Reject mixed runs and false cross-host reports without executing devices."""
import copy
import json
from pathlib import Path
import pytest
from tests.ll_probe.profiling.analyze_profile import analyze


def reports(tmp_path, world=32, kind='comm'):
    nodes = 2 if world == 32 else 1
    local = world // nodes
    topology = dict(ips=['192.0.2.10', '192.0.2.11'][:nodes],
                    rank_num_per_server=local, world=world,
                    master_addr='192.0.2.10', master_port=29500, shmem_port=29501)
    paths = []
    for rank in range(world):
        folder = tmp_path / f'node {rank // local}' / f'rank_{rank}'
        folder.mkdir(parents=True)
        (folder / 'native.log').write_text('PASS: normal cleanup\n')
        report = dict(rank=rank, world=world, device=rank % local, kernel='test_kernel',
                      warmup=2, capture_steps=2, sample_window=1, delay_shape=[64,64],
                      reference_defaults=False, measurement='named_kernel_task_duration_not_end_to_end',
                      git_revision='fixture', configuration=dict(kind=kind, layer=3, scenario=4, bytes=32,
                      output_dir=f'/different/root/node{rank // local}/result'), source_sha256={},
                      binary_sha256='fixture', correctness_passed=True,
                      deployment_identity=dict(run_id='same-run', topology=topology),
                      rank_provenance=dict(node_index=rank // local, boot_id=f'boot-{rank // local}'),
                      kernel_statistics=[dict(kernel_name='test_kernel', task_type='AI_VECTOR_CORE',
                                              kernel_us=[2,3], selected_kernel_us=[3])])
        path=folder/'report.json'; path.write_text(json.dumps(report)); paths.append(path)
    return paths


@pytest.mark.parametrize('world', [2,4,8,16,32])
@pytest.mark.parametrize('kind', ['comm','metadata'])
def test_actual_world_and_separate_output_roots(tmp_path, world, kind):
    result=analyze(reports(tmp_path,world,kind))
    assert result['all_rank_audit']=='passed'
    assert ('aggregate_wire_gbps' in result)==(kind=='comm')


@pytest.mark.parametrize('problem', ['run_id','topology','device','node','boot','missing','duplicate'])
def test_reject_false_global_success(tmp_path, problem):
    paths=reports(tmp_path)
    if problem=='missing':paths.pop()
    elif problem=='duplicate':paths.append(paths[0])
    elif problem=='boot':
        for path in paths[16:]:
            obj=json.loads(path.read_text()); obj['rank_provenance']['boot_id']='boot-0'
            path.write_text(json.dumps(obj))
    else:
        obj=json.loads(paths[-1].read_text())
        if problem=='run_id':obj['deployment_identity']['run_id']='another-run'
        elif problem=='topology':obj['deployment_identity']['topology']['shmem_port']=29502
        elif problem=='device':obj['device']=0
        elif problem=='node':obj['rank_provenance']['node_index']=0
        paths[-1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):analyze(paths)
