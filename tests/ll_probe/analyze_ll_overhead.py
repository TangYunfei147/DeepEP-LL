"""Fit existing native LL profiler reports versus B; intercept is an estimate."""
import argparse
import json
import math
from pathlib import Path
import statistics

def fit(points):
    if any(not math.isfinite(x) or not math.isfinite(y) or x<=0 or y<=0 for x,y in points):
        raise ValueError('Fit requires positive finite B and durations')
    if len({x for x, _ in points}) < 3:
        raise ValueError('At least three distinct B values are required')
    xbar = statistics.mean(x for x, _ in points)
    ybar = statistics.mean(y for _, y in points)
    sxx=sum((x-xbar)**2 for x,_ in points)
    slope = sum((x-xbar)*(y-ybar) for x,y in points)/sxx
    intercept = ybar-slope*xbar
    residuals = [y-intercept-slope*x for x,y in points]
    sse = sum(r*r for r in residuals)
    sst = sum((y-ybar)**2 for _,y in points)
    # OLS standard error assumes independent, equally variable observations.
    # It is a model diagnostic, not a hardware fixed-cost confidence guarantee.
    variance=sse/(len(points)-2)
    intercept_se=math.sqrt(variance*(1/len(points)+xbar*xbar/sxx))
    warnings=[]
    if intercept<0: warnings.append('Negative intercept; retain it, do not interpret as negative physical overhead')
    if slope<0: warnings.append('Negative slope; check noise, route and execution-regime changes')
    if sst and 1-sse/sst<0.9: warnings.append('Weak linear fit; inspect residuals and narrower B intervals')
    return {'intercept_us_estimate':intercept,'slope_us_per_token':slope,
            'intercept_standard_error_us':intercept_se,
            'residual_rmse_us':math.sqrt(sse/len(points)),
            'standard_error_assumption':'independent homoscedastic per-B observations; excludes systematic/profiler bias',
            'warnings':warnings,
            'r_squared':1-sse/sst if sst else None,
            'residual_us':residuals, 'B_range':[min(x for x,_ in points),max(x for x,_ in points)],
            'points':points, 'interpretation':'empirical t(B) intercept; not directly measured zero-payload LL cost'}

def analyze(report, bmin, bmax):
    if report.get('status') != 'completed': raise ValueError('Native LL run must be completed')
    groups = {}
    per_rank = {}
    slowest = {}
    if report.get('result_count',len(report['results']))!=len(report['results']):
        raise ValueError('Incomplete result list')
    for case in report['results']:
        if case.get('correctness') != 'passed' or case.get('backend') != 'll':
            raise ValueError('Require passed correctness and native LL backend for all cases')
        config=case['config']; b=config['B']
        if not bmin <= b <= bmax: continue
        identity={k:v for k,v in config.items() if k!='B'}
        identity.update(route=case['route'],world_size=report['world_size'])
        for op in ['dispatch','combine']:
            if op not in case: continue
            perf=case[op]
            if perf.get('measurement')!='kernel_duration_us': raise ValueError('Require profiler kernel tasks')
            ranks=perf['ranks']
            if len(ranks)!=report['world_size'] or sorted(row['rank'] for row in ranks)!=list(range(report['world_size'])):
                raise ValueError('Missing or duplicate ranks')
            means=[]
            for row in ranks:
                raw=row['kernel_us']; tail=perf['perf_tail']
                if not isinstance(tail,int) or tail<0: raise ValueError('Invalid perf_tail')
                selected=raw[-tail:] if tail else raw
                if not selected or any(not math.isfinite(t) or t<=0 for t in selected): raise ValueError('Invalid raw samples')
                means.append(statistics.mean(selected))
            key=json.dumps({**identity,'operation':op},sort_keys=True)
            groups.setdefault(key,[]).append((b,max(means)))
            slowest.setdefault(key,[]).append({'B':b,'rank':ranks[means.index(max(means))]['rank']})
            for row,mean in zip(ranks,means):
                per_rank.setdefault((key,row['rank']),[]).append((b,mean))
    results=[]
    for key,points in groups.items():
        result={'group':json.loads(key),**fit(sorted(points)),
                'slowest_rank_by_B':slowest[key],
                'per_rank_fits':[{'rank':rank,**fit(sorted(rank_points))}
                                 for (group,rank),rank_points in per_rank.items() if group==key]}
        if len({entry['rank'] for entry in slowest[key]})>1:
            result['warnings'].append('Slowest rank changes across B; envelope intercept may differ from per-rank fixed costs')
        results.append(result)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('summary',type=Path)
    p.add_argument('--b-min',type=int,default=1)
    p.add_argument('--b-max',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    reports=analyze(json.loads(args.summary.read_text()),args.b_min,args.b_max)
    if not reports: raise ValueError('No matching LL samples')
    # Keep M constant in the experiment. Different M/route/dtype/H/K/E stay separate.
    with args.output.open('x') as stream:
        json.dump({'fits':reports,'statistic':'max of per-rank means, recomputed from raw samples',
                   'exit_audit':'Summary completion is required; normal per-rank cleanup must be checked separately'},stream,indent=2)

if __name__=='__main__': main()
