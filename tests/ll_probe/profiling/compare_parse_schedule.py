"""Time complete test_ep_low_latency.py commands, including deferred parsing."""
import argparse
import json
import os
from pathlib import Path
import statistics
import signal
import hashlib
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT,output_path,runtime_environment
from parse_schedule_probe import require_two_devices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=output_path,required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--modes',nargs='+',choices=('immediate','deferred','deferred_all'))
    args=p.parse_args(); require_two_devices(); runtime_environment()
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[REPO_ROOT/'tests/elastic/test_ep_low_latency.py',Path(__file__),
             Path(__file__).with_name('parse_schedule_probe.py'),Path(__file__).with_name('profile_common.py')]
    (args.output/'identity.json').write_text(json.dumps({
        'revision':subprocess.check_output(['git','-C',str(REPO_ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(REPO_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'protocol':'10 warmup,100 capture,last50 per task; full runs only; Exp64x64 A5 validation'},indent=2)+'\n')
    rows=[]
    modes=args.modes or (('deferred_all',) if args.pilot else
                         ('immediate','deferred','deferred_all','deferred_all','deferred','immediate'))
    for index,mode in enumerate(modes):
        out=args.output/f'{index}-{mode}'
        cmd=[sys.executable,str(REPO_ROOT/'tests/elastic/test_ep_low_latency.py'),'--parse-schedule-probe',
             '--schedule',mode,'--output',str(out),'--port',str(19891+index)]
        if args.pilot: cmd+=['--tokens','1','--dtypes','bf16','--warmup','1','--samples','3','--tail','2']
        print('Starting '+mode,flush=True)
        with (args.output/f'{index}-{mode}.log').open('w') as log:
            start=time.perf_counter()
            process=subprocess.Popen(cmd,cwd=REPO_ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=1800)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                raise
            elapsed=time.perf_counter()-start
        if code: raise RuntimeError(f'{mode} failed; see {log.name}')
        reports=[json.loads((out/f'rank_{rank}.json').read_text()) for rank in range(2)]
        if not all(r['all_profiles_parsed'] and r['correctness_passed'] for r in reports):
            raise RuntimeError('Missing completed work')
        stats=[]
        for r in reports:
            stats.append({k:sum(x[k] for x in r['records']) for k in
                          ('capture_including_export_seconds','close_sync_export_seconds','parse_gather_seconds','offline_analysis_seconds')})
        row={'index':index,'mode':mode,'command_wall_seconds':elapsed,'command':cmd,'rank_phases':stats,
             'case_count':reports[0]['case_count'],'profiles_per_rank':len(reports[0]['records'])}
        rows.append(row)
        (args.output/'wall_times.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    times={mode:[r['command_wall_seconds'] for r in rows if r['mode']==mode] for mode in dict.fromkeys(modes)}
    means={mode:statistics.mean(v) for mode,v in times.items()}
    result={'runs':rows,'mean_seconds':means,
            'reduction_percent':{m:100*(means['immediate']-v)/means['immediate'] for m,v in means.items() if m!='immediate'} if 'immediate' in means else {},
            'scope':'A5 devices0/1, local-copy proxy, Exp64x64; command startup through all parsing and exit',
            'export_analysis_deferred_modes':['deferred_all'],'pilot':args.pilot}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__=='__main__': main()
