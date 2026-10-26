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
    assert report['recording']=='systemcycle-stages' and report['timer_profile_id']==9
    expected={'ENTRY','INIT_DONE','FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE'}
    assert len(report['cores'])==56 and len(report['milestones'])==56*len(expected)
    for core in report['cores']:
        points={row['point'] for row in report['milestones'] if row['core']==core['core']}
        assert points==expected and core['milestone_count']==len(expected)
    assert all(row['calls']==1 for row in report['records'])
    assert all(value==0 for core in data for value in [v for row in core for v in row][48:])
    (out/f'parsed-{mode}.json').write_text(json.dumps(report,indent=2))
print('PASS timer smoke parser, SystemCycle first-hit points and all 56 AIV slots')
