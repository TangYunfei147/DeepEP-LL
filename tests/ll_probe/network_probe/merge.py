"""Require complete reports from every node before declaring the job successful."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import project_path, results_dir, runtime_environment, workspace_root


def merge(paths):
    reports=[json.loads(project_path(p).read_text()) for p in paths]
    if not reports: raise ValueError('No reports')
    ref=reports[0]['job']
    seen=set(); boots=set(); records=[]
    keys=('hosts','devices_per_host','world','peer','endpoint','workspace_bytes','huge_pages','scope','dense_qps','identity')
    for report in reports:
        job=report['job']; node=job['node_index']
        if not report['local_success'] or node in seen:
            raise ValueError('Failed or duplicate node report')
        if any(job[k]!=ref[k] for k in keys): raise ValueError('Configuration/source identity mismatch')
        if job['boot_id'] in boots: raise ValueError('Multiple node reports came from the same machine')
        boots.add(job['boot_id'])
        if job['shmem_library']['cmake_flags']!=ref['shmem_library']['cmake_flags']:
            raise ValueError('SHMEM build flags differ across hosts')
        seen.add(node); records.extend(report['records'])
        expected=set(range(node*ref['devices_per_host'],(node+1)*ref['devices_per_host']))
        done=[r['rank'] for r in report['records'] if r['kind']=='complete']
        if set(done)!=expected or len(done)!=len(expected): raise ValueError('Missing or duplicate rank completion')
    if seen!=set(range(len(ref['hosts']))): raise ValueError('Missing node reports')
    checks={(r['sender'],r['rank']) for r in records if r['kind']=='mte_readback' and r['passed']}
    if checks!={(0,ref['peer']),(ref['peer'],0)}: raise ValueError('Missing bidirectional MTE readback')
    routes=[r for r in records if r['kind']=='udma_route']
    if not routes or any(r['query_rc'] for r in routes): raise ValueError('UDMA query failure')
    return {'success':True,'scope':ref['scope'],'mte_actual_net_layer':None,
            'note':'UDMA established routes only; no UDMA payload; MTE physical route is unobserved',
            'binary_hashes_match':len({r['job']['executable_sha256'] for r in reports})==1,
            'shmem_library_hashes_match':len({r['job']['shmem_library']['sha256'] for r in reports})==1,
            'routes':routes,'mte_readback_directions':sorted(checks)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('reports',nargs='+'); a=p.parse_args()
    print(json.dumps(merge(a.reports),indent=2))
