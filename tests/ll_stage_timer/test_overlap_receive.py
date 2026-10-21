"""CPU checks for overlap capture and layout/progress invariants."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
fixtures=load("overlap_segments","tests/ll_stage_timer/test_segments.py")
timer=load("overlap_timer","deep_ep/utils/ll_stage_timer.py")
def test_schema11_and_legacy():
    for schema in (7,9,10,11):
        assert timer.parse_records(fixtures.raw(schema=schema),"dispatch")["schema"]==schema
    assert timer.SCHEMA==11
    assert timer.STAGES[70:]==("SEND_COMPLETE_WAIT","TRANSITION_DRAIN","RECV_PREPARE","SEND_SUBMIT_END","OVERLAP_PATH","SERIAL_FALLBACK","SEND_LANES","RECV_LANES")
    assert (8+2*timer.RECORDS)*8<=timer.SLOTS*timer.FIELDS*8
