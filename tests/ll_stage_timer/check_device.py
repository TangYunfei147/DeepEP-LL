"""CPU parse of real device smoke output; no device initialization."""
import importlib.util
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('timer',root/'deep_ep/utils/ll_stage_timer.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
out=Path(sys.argv[1])
for mode in (1,):
    data=json.loads((out/f'raw-1-{mode}.json').read_text())
    report=timer.parse_records(data,'dispatch')
    assert report['schema']==timer.SCHEMA
    assert len(report['records'])==56*4
    assert all(core['omitted_intervals']==1 for core in report['cores'])
    assert all(row['calls']==3 for row in report['records'] if row['stage']=='RECV_POLL')
    (out/f'parsed-{mode}.json').write_text(json.dumps(report,indent=2))
print('PASS real device timer parser, sampled intervals/full metrics and all 56 AIV records')
