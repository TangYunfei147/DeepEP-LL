"""Synthetic analysis checks; these are not hardware measurement evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_ll_overhead import analyze, fit

line=fit([(1,5),(2,8),(4,14)])
assert abs(line['intercept_us_estimate']-2)<1e-12
assert abs(line['slope_us_per_token']-3)<1e-12
assert line['r_squared']==1
assert line['intercept_standard_error_us']==0
assert abs(fit([(1,1),(2,3),(4,7)])['intercept_us_estimate']+1)<1e-12
report={'status':'completed','world_size':2,'results':[]}
for b in [1,2,4]:
    report['results'].append({'correctness':'passed','backend':'ll','route':'normal',
        'config':{'B':b,'M':256,'H':7168},'dispatch':{'measurement':'kernel_duration_us',
        'perf_tail':2,'ranks':[{'rank':0,'kernel_us':[999,1+2*b,1+2*b]},
        {'rank':1,'kernel_us':[999,2+3*b,2+3*b]}]}})
analysis=analyze(report,1,4)[0]
assert abs(analysis['intercept_us_estimate']-2)<1e-12
assert len(analysis['per_rank_fits'])==2
assert all(entry['rank']==1 for entry in analysis['slowest_rank_by_B'])
# A real max-rank envelope need not have the same intercept as each rank.
switch={'status':'completed','world_size':2,'results':[]}
for b in [1,2,4]:
    switch['results'].append({'correctness':'passed','backend':'ll','route':'normal',
        'config':{'B':b,'M':256},'dispatch':{'measurement':'kernel_duration_us','perf_tail':0,
        'ranks':[{'rank':0,'kernel_us':[10+b]},{'rank':1,'kernel_us':[2+4*b]}]}})
assert any('Slowest rank changes' in warning for warning in analyze(switch,1,4)[0]['warnings'])
for case in switch['results']: case['dispatch']['ranks'].reverse()
assert analyze(switch,1,4)[0]['slowest_rank_by_B']==[
    {'B':1,'rank':0},{'B':2,'rank':0},{'B':4,'rank':1}]
noisy=fit([(1,6),(2,7),(4,15)])
assert noisy['intercept_standard_error_us']>0 and noisy['residual_rmse_us']>0
report['results'][0]['correctness']='skipped'
try: analyze(report,1,4)
except ValueError: pass
else: raise AssertionError('Skipped correctness accepted')
report['results'][0]['correctness']='passed'
report['results'][0]['dispatch']['ranks'].pop()
try: analyze(report,1,4)
except ValueError: pass
else: raise AssertionError('Missing rank accepted')
print('PASS: known/noisy fit, uncertainty diagnostics, negative intercept, tail, rank envelope switch, skip/missing-rank rejection')
