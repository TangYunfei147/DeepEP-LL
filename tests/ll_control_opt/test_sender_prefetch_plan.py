"""Delivery contract for the pinned, uninstrumented sender optimization A/B."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import re


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite = load('sender_prefetch_suite', 'suite.py')
report = load('sender_prefetch_report', 'export_report.py')
BASELINE = '9027275b31a28608296b1b71c93fa6e2bb456db5'
MANIFEST = Path(__file__).with_name('commits-sender-prefetch.json')


def plan():
    args = suite.parser().parse_args([
        '--commits', str(MANIFEST), '--devices', ','.join(map(str, range(16))),
        '--num-nodes', '1', '--node-rank', '0', '--master-addr', '127.0.0.1',
        '--master-port', '6000', '--shmem-port', '6001', '--run-prefix', 'sender-vector-tail-001',
        '--resume', '--send-cores', '30', '--seeds', '0', '17', '101',
        '--rounds', '2', '--order', 'alternating'])
    return args, suite.make_plan(args)


def test_pinned_pair_uses_same_sdk_and_balanced_full_off_st():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert manifest['schema'] == 2 and manifest['comparison_base'] == BASELINE
    assert [item['label'] for item in manifest['commits']] == ['baseline', 'prequant-vector-tail']
    assert manifest['commits'][0]['revision'] == BASELINE
    assert all(re.fullmatch(r'[0-9a-f]{40}', item['revision']) for item in manifest['commits'])
    args, selected = plan()
    assert args.resume and not args.resume_results
    assert selected['timer'] == 'OFF'
    assert selected['send_cores'] == 30 and selected['recv_cores'] == 26
    assert selected['sampling'] == dict(warmup=10, num_iters=100, perf_tail=50)
    assert len(selected['phases']['build']) == len(selected['phases']['install']) == 2
    dependencies = list(selected['dependencies'].values())
    assert dependencies[0] == dependencies[1]
    runs = selected['phases']['run']
    assert len(runs) == 12 and len({suite._run_key(item) for item in runs}) == 12
    for round_id in (1, 2):
        for seed in (0, 17, 101):
            pair = [item for item in runs if (item['round'], item['seed']) == (round_id, seed)]
            labels = ['baseline', 'prequant-vector-tail']
            assert [item['label'] for item in pair] == (labels if round_id == 1 else labels[::-1])
            for item in pair:
                command = item['command']
                assert item['measured_route_seed'] == seed + 2
                assert command[command.index('--seed') + 1] == str(seed)
                for forbidden in ('--dispatch-only', '--skip-check', '--ll-stage-timer', '--ll-profile-capture'):
                    assert forbidden not in command
                assert item['revision'] in {entry['revision'] for entry in selected['commits']}


def test_all_twenty_four_comparison_rows_fit_one_return_file(tmp_path):
    _, selected = plan()
    measurements = {(round_id, seed): {
        'baseline': dict(dispatch=85.4, combine=137.8),
        'prequant-vector-tail': dict(dispatch=84.9, combine=137.9),
    } for round_id in (1, 2) for seed in (0, 17, 101)}
    rows = suite.comparisons(selected['commits'], measurements, selected['order'])
    source = tmp_path / 'comparison.csv'
    with source.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destination = tmp_path / 'analysis-report.txt'
    size = report.export(source, destination)
    assert size == len(destination.read_bytes()) <= 65535
    heading, body = destination.read_text(encoding='utf-8').split('\n\n', 1)
    retained = list(csv.DictReader(io.StringIO(body)))
    assert len(rows) == len(retained) == 24 and 'Rows: 24' in heading
    assert {(row['round'], row['seed'], row['label'], row['operation']) for row in retained} == {
        (str(row['round']), str(row['seed']), row['label'], row['operation']) for row in rows}
    assert {row['reference_revision'] for row in retained} == {BASELINE}
