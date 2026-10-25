"""Accounting invariants against the production ST's generated uniform routes."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('topology_workload_test', Path(__file__).with_name('topology_workload.py'))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_three_mesh_four_clos_one_inter_host_classes():
    assert model.path_class(0,0) == 'self'
    assert {model.path_class(0,peer) for peer in (1,2,3)} == {'fullmesh'}
    assert {model.path_class(0,peer) for peer in range(4,16)} == {'clos'}
    assert model.path_class(0,16) == 'inter_host'
    assert model.path_class(16,19) == 'fullmesh'


def test_group_first_hop_preserves_same_rank_relay_and_original_first_expert():
    # Two experts on target card4 must retain a self relay in the current policy.
    assert list(model.packet_edges(0,[8,9,10],2)) == [
        (0,4,'initial_first_hop'),(4,4,'relay'),(4,5,'relay')]
    assert list(model.packet_edges(0,[10,8,9],2)) == [
        (0,5,'initial_first_hop'),(5,4,'relay'),(5,4,'relay')]
    assert list(model.packet_edges(0,[0,2],2)) == [(0,0,'initial_direct'),(0,1,'initial_direct')]


def test_multi_seed_traffic_is_dynamic_and_not_port_utilization():
    outputs = []
    for seed in (0,17,101):
        report, rows = model.account(seed)
        assert report['measured_route_seed'] == seed+2
        assert report['logical_expert_rows'] == 16*256*8
        assert sum(row['packets'] for row in rows) == report['logical_expert_rows']
        assert report['packet_stride_bytes'] == 4096
        assert report['paths']['inter_host']['packets'] == 0
        assert all(row['ll_packet_bytes'] == row['packets']*4096 for row in rows)
        assert all(row['path'] != 'clos' for row in rows if row['leg']=='relay')
        outputs.append(report['paths'])
    assert outputs[0] != outputs[1] != outputs[2]
