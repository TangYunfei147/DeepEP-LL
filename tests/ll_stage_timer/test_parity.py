"""Configuration guards, shared launch plan, group selection and comparison semantics."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
st=load('parity_st','tests/elastic/test_ep_low_latency.py')






def test_attribute_and_function_group_queries():
    assert st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=4),16)==4
    assert st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=lambda p:4 if p<=16 else 16),32)==16
    for bad in (0,17,True,3):
        with pytest.raises(RuntimeError):st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=bad),16)










@pytest.mark.parametrize('implementation,rdma',[('configurable-groups',False),('ndr-hns1825',True)])
def test_portable_launcher_plan_and_backend_guard(tmp_path,implementation,rdma):
    moved=tmp_path/'checkout with spaces'
    for path in ('tests/ll_stage_timer/run.py','tests/elastic/test_ep_low_latency.py',
                 'scripts/ll_dispatch_implementation.py'):
        target=moved/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    (moved/'tests/ll_stage_timer/branch.json').write_text(json.dumps(dict(implementation=implementation,rdma=rdma)))
    env=dict(os.environ)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(key,None)
    command=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--backend','mte','--validation','--plan',
             '--num-processes','16']
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['world_size']==16
    command[command.index('mte')]='rdma'
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==rdma
    if not rdma:assert 'MTE-only' in result.stderr
    # No backend/process-count options: branch default and explicit device list
    # are sufficient; A5 --plan models 16 ranks without launching devices.
    default=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--validation','--plan']
    env['DEEPEP_LL_BACKEND']='rdma' if not rdma else 'mte'  # stale opposite-branch shell state
    result=subprocess.run(default,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['world_size']==16



    config=moved/'settings.json'
    config.write_text(json.dumps({'backend':'rdma','num_processes':16}))
    configured=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--validation','--plan','--config','settings.json']
    result=subprocess.run(configured,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==rdma
    overridden=configured+['--backend','mte']
    subprocess.run(overridden,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
