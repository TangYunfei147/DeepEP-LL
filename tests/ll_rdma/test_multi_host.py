"""Multi-host scope and deployment gate regressions; CPU and loopback only."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import runpy
import socket
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location('audit_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch, ready = load('one_click'), load('target_ready')
IDENTITY = {'source': 'same-commit', 'protocol': 28, 'parameters': {'dtype': 'fp8'}}


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize('nodes', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('mode', ['perf', 'reference'])
def test_launcher_uses_requested_world_and_node(nodes, mode):
    a = launch.parse_args([str(nodes - 1), '192.0.2.1', 'multi', '--num-nodes', str(nodes),
                           '--experts', '1024', '--mode', mode])
    args = launch.test_arguments(a)
    assert args[args.index('--num-nodes') + 1] == str(nodes)
    assert args[args.index('--node-rank') + 1] == str(nodes - 1)
    assert args[args.index('--num-processes') + 1] == '16'
    launch.validate_test_arguments(a)
    st = runpy.run_path(str(ROOT / 'tests/elastic/test_ep_low_latency.py'))
    options = st['parse_args'](args[:2] + args[4:])
    assert st['launch_config'](options, {})[0] == nodes * 16


@pytest.mark.parametrize('node,nodes', [(-1, 4), (4, 4), (0, 0), (0, -1)])
def test_node_bounds_are_relative_to_host_count(node, nodes):
    with pytest.raises(SystemExit):
        launch.parse_args([str(node), '192.0.2.1', 'bad', '--num-nodes', str(nodes)])


@pytest.mark.parametrize('nodes', [4, 8, 16])
def test_all_nodes_receive_the_same_identity_agreement(nodes):
    endpoint = port()
    with ThreadPoolExecutor(max_workers=nodes) as pool:
        futures = [pool.submit(ready.synchronize, rank, '127.0.0.1', endpoint, 'multi', IDENTITY, 4, nodes)
                   for rank in reversed(range(nodes))]
        reports = [future.result(timeout=6) for future in futures]
    assert all(report == reports[0] for report in reports)
    assert [node['node_rank'] for node in reports[0]['nodes']] == list(range(nodes))


def test_no_early_release_before_the_last_node_is_prepared():
    endpoint = port()
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = [pool.submit(ready.synchronize, rank, '127.0.0.1', endpoint, 'wait', IDENTITY, 3, 4)
                 for rank in range(3)]
        time.sleep(0.15)
        assert all(not future.done() for future in first)
        last = pool.submit(ready.synchronize, 3, '127.0.0.1', endpoint, 'wait', IDENTITY, 3, 4)
        reports = [future.result(timeout=4) for future in [*first, last]]
    assert all(report == reports[0] for report in reports)


def test_late_node_mismatch_rejects_prepared_nodes():
    endpoint = port()
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = [pool.submit(ready.synchronize, rank, '127.0.0.1', endpoint, 'fail', IDENTITY, 2, 4)
                 for rank in range(3)]
        time.sleep(0.1)
        bad = pool.submit(ready.synchronize, 3, '127.0.0.1', endpoint, 'fail', {'source': 'different'}, 2, 4)
        for future in [*first, bad]:
            with pytest.raises(ready.TargetReadyError):
                future.result(timeout=3)


def test_missing_node_cannot_release_the_partial_group():
    endpoint = port()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(ready.synchronize, rank, '127.0.0.1', endpoint, 'missing', IDENTITY, 0.3, 4)
                   for rank in range(3)]
        for future in futures:
            with pytest.raises(ready.TargetReadyError):
                future.result(timeout=2)


def test_single_host_needs_no_preparation_socket(monkeypatch):
    monkeypatch.setattr(ready.socket, 'create_server', lambda *a, **k: pytest.fail('unexpected network'))
    report = ready.synchronize(0, '127.0.0.1', 12345, 'single', IDENTITY, 1, 1)
    assert len(report['nodes']) == 1


def test_duplicate_node_does_not_count_as_another_prepared_host():
    endpoint = port()
    deadline = time.monotonic() + 3
    hello = {'protocol': ready.PROTOCOL, 'type': 'prepared', 'node_rank': 1, 'num_nodes': 4,
             'run_id': 'duplicate', 'identity': IDENTITY}
    with ThreadPoolExecutor(max_workers=1) as pool:
        server = pool.submit(ready.synchronize, 0, '127.0.0.1', endpoint, 'duplicate', IDENTITY, 3, 4)
        with ready._connect('127.0.0.1', endpoint, socket.AF_INET, deadline) as first:
            ready._send(first, hello, deadline)
            with ready._connect('127.0.0.1', endpoint, socket.AF_INET, deadline) as duplicate:
                ready._send(duplicate, hello, deadline)
                assert ready._receive(duplicate, deadline)['ok'] is False
                assert ready._receive(first, deadline)['ok'] is False
        with pytest.raises(ready.TargetReadyError, match='unique nonzero'):
            server.result(timeout=4)


def test_many_nodes_do_not_duplicate_identity_past_frame_limit():
    nodes, endpoint = 16, port()
    identity = {'build': 'x' * 6000}
    assert len(ready._json_bytes([identity] * nodes)) > ready.MAX_FRAME_BYTES
    with ThreadPoolExecutor(max_workers=nodes) as pool:
        futures = [pool.submit(ready.synchronize, rank, '127.0.0.1', endpoint, 'compact', identity, 4, nodes)
                   for rank in range(nodes)]
        reports = [future.result(timeout=6) for future in futures]
    assert all(report == reports[0] for report in reports)
    assert all(node['identity'] == identity for node in reports[0]['nodes'])
