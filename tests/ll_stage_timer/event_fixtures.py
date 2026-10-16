"""Small explicit event fixtures; never allocate the full device arena."""
import copy
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fixture_event_parser',ROOT/'deep_ep/utils/ll_stage_timer.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)


def raw(operation='dispatch',mode=1,cores=16):
    return dict(schema=4,cores=[dict(header=[timer.MAGIC,4,1 if operation=='dispatch' else 2,cores,core,mode,1,len(timer.STAGES)],
        counts=[1]+[0]*(len(timer.STAGES)-2),traffic=[[0]*4 for _ in range(5)],flags=0,errors=0,
        events={'kernel':[[100,1000,0]]}) for core in range(cores)])


def set_events(data,core,stage,pairs):
    entry=data['cores'][core]
    entry['counts'][timer.STAGES.index(stage)-1]=len(pairs)
    if pairs:
        entry['events'][stage]=[pair[:] if len(pair)==3 else pair+[0 if stage=='kernel' else timer.STAGES.index(stage)*10000+index+1] for index,pair in enumerate(pairs)]
    else:
        entry['events'].pop(stage,None)
