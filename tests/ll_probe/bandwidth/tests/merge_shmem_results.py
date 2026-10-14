"""Require both rank reports before publishing cross-host experiment success."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from paths import project_path

def merge(sender,receiver,allow_same_host=False):
    for report in [sender,receiver]:
        if report.get('status')!='local-ranks-passed': raise ValueError('Rank report not successful')
        if report['plan']['topology']!='user-cross-host': raise ValueError('Require separately launched rank reports')
        if len(report['results'])!=len(report['plan']['cases']): raise ValueError('Incomplete cases')
    left,right=sender['plan'],receiver['plan']
    for field in ['direction','transport','endpoint','cases','samples','warmup','reference_peak_gbps','binary_sha256']:
        if left[field]!=right[field]: raise ValueError(f'Mismatched {field}')
    if not allow_same_host and left['hostname']==right['hostname']:
        raise ValueError('Same hostname on both reports; use --allow-same-host only for launcher verification')
    combined=[]
    for index,(a,b) in enumerate(zip(sender['results'],receiver['results'])):
        case=left['cases'][index]
        for report,rank in [(a,0),(b,1)]:
            if report['local_ranks']!=[rank] or report['local_checks']!='passed': raise ValueError('Rank identity/checks mismatch')
            if report['samples']!=left['samples']: raise ValueError('Mismatched sample counts')
            if any(report[key]!=value for key,value in case.items()): raise ValueError('Mismatched case configuration')
        combined.append({**a,'all_rank_receiver_checks':'passed','ranks':[0,1]})
    return {'status':'passed','scope':'same-host launcher verification' if allow_same_host else 'cross-host user experiment',
            'hosts':[left['hostname'],right['hostname']], 'plan':left,'results':combined}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('rank0_summary',type=project_path)
    p.add_argument('rank1_summary',type=project_path)
    p.add_argument('--output',type=project_path,required=True)
    p.add_argument('--allow-same-host',action='store_true')
    args=p.parse_args()
    result=merge(json.loads(args.rank0_summary.read_text()),json.loads(args.rank1_summary.read_text()),args.allow_same_host)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)

if __name__=='__main__': main()
