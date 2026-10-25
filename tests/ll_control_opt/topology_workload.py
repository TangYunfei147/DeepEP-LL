"""Logical G4 packet traffic for the suite's uniform routes; never NIC telemetry.

The user-confirmed topology is 16 NPUs/host, contiguous groups of four: three
fullmesh peer ports, four intra-host Clos ports, one inter-host port per NPU.
We classify packet endpoints, not physical port assignment or elapsed time.
"""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUITE = load('topology_control_suite', 'tests/ll_control_opt/suite.py')
ROUTING = load('topology_route_layout', 'tests/elastic/inspect_dispatch_routes.py')


def path_class(source, destination, group=4, ranks_per_host=16):
    if source == destination:
        return 'self'
    if source // ranks_per_host != destination // ranks_per_host:
        return 'inter_host'
    return 'fullmesh' if source // group == destination // group else 'clos'


def packet_edges(source, experts, experts_per_rank, group=4):
    """Exact initial/relay endpoint multiset; timing and core ownership excluded."""
    destinations = {}
    for expert in experts:
        destination = expert // experts_per_rank
        destinations.setdefault(destination // group, []).append(destination)
    for target_group, peers in destinations.items():
        if target_group == source // group:
            for peer in peers:
                yield source, peer, 'initial_direct'
        else:
            first, *members = peers
            yield source, first, 'initial_first_hop' if members else 'initial_direct'
            for peer in members:
                yield first, peer, 'relay'


def account(requested_seed, *, world=16, correctness_repeats=3):
    if world < 16 or world % 16 or correctness_repeats < 1:
        raise ValueError('Require complete 16-rank hosts and positive correctness repeats')
    shape = SUITE.SHAPE
    experts, tokens, topk = shape['num_experts'], shape['B'], shape['num_topk']
    if experts % world:
        raise ValueError('Experts must divide equally across ranks')
    measured_seed = requested_seed + correctness_repeats - 1
    routes = ROUTING.ST.make_routes(world, tokens, topk, experts, measured_seed, 'normal')
    _, packet_bytes = ROUTING.packet_layout(shape['H'], topk, shape['dtype'])
    edges = Counter()
    for source, rows in enumerate(routes):
        for ids in rows:
            for sender, destination, leg in packet_edges(source, ids, experts // world):
                edges[sender, destination, path_class(sender, destination), leg] += 1
    records = [dict(requested_seed=requested_seed, measured_route_seed=measured_seed,
                    source=source, destination=destination, path=path, leg=leg,
                    packets=count, ll_packet_bytes=count*packet_bytes)
               for (source, destination, path, leg), count in sorted(edges.items())]
    by_path = {kind: dict(packets=sum(row['packets'] for row in records if row['path']==kind),
                         ll_packet_bytes=sum(row['ll_packet_bytes'] for row in records if row['path']==kind))
               for kind in ('self', 'fullmesh', 'clos', 'inter_host')}
    rows = world*tokens*topk
    if sum(row['packets'] for row in records) != rows:
        raise RuntimeError('Every expert route must have exactly one final payload delivery')
    return dict(requested_seed=requested_seed, measured_route_seed=measured_seed,
                world_size=world, logical_expert_rows=rows, packet_stride_bytes=packet_bytes,
                paths=by_path), records


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 17, 101])
    parser.add_argument('--world-size', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    summaries, records = [], []
    for seed in args.seeds:
        summary, rows = account(seed, world=args.world_size)
        summaries.append(summary)
        records.extend(rows)
    output = (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='cpu_logical_packet_accounting', shape=SUITE.SHAPE,
                  ranks_per_host=16, ranks_per_fullmesh_group=4,
                  ports_per_npu=dict(fullmesh=3, intra_host_clos=4, inter_host=1),
                  correctness_repeats=3, seeds=summaries,
                  physical_port_mapping='not_measured', physical_link_utilization='not_measured',
                  note='Packet bytes include LL block flags, not network framing. Endpoint classes are inferred '
                       'from the confirmed rank topology. This does not assign packets to the four Clos ports, '
                       'measure bandwidth, replay timing, or infer speedup. Buffered/immediate receive scheduling '
                       'changes timing but must preserve this endpoint/byte multiset.')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (output/'topology_counts.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(output/'summary.json')


if __name__ == '__main__':
    main()
