"""Verify the user two-terminal launch and merge path on devices 0/1 only."""
import json
from pathlib import Path
import subprocess
from merge_shmem_results import merge

from paths import workspace_root, project_path, runtime_environment, results_dir
ROOT = workspace_root()
import tempfile
import os
os.environ.update(PROBE_VISIBLE_DEVICES="0,1", CUDA_VISIBLE_DEVICES="0,1",
                  ASCEND_RT_VISIBLE_DEVICES="0,1", ASCEND_VISIBLE_DEVICES="0,1")
runtime_environment()
base=Path(tempfile.mkdtemp(prefix='shmem-separate-launch-',dir=results_dir()))

here=Path(__file__).resolve().parent
children=[]
streams=[]
try:
    for rank in [0,1]:
        stream=(base/f'launcher-rank{rank}.log').open('w'); streams.append(stream)
        command=['bash',str(here/'run_shmem_scan.sh'),'--rank',str(rank),'--device',str(rank),
                 '--endpoint','tcp://127.0.0.1:19732','--execute','--bytes','15392',
                 '--cores','8','--lanes','4','--modes','0','1','2','3',
                 '--warmup','0','--samples','2','--output',str(base/f'rank{rank}')]
        children.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT))
    for child in children:
        if child.wait(timeout=540): raise RuntimeError('Rank launcher failed; see logs')
finally:
    for child in children:
        if child.poll() is None: child.kill()
        child.wait()
    for stream in streams: stream.close()
left=json.loads((base/'rank0/summary.json').read_text())
right=json.loads((base/'rank1/summary.json').read_text())
result=merge(left,right,allow_same_host=True)
(base/'merged.json').write_text(json.dumps(result,indent=2))
try: merge(left,right)
except ValueError: pass
else: raise AssertionError('Cross-host report incorrectly accepted same host')
right['results'][0]['local_ranks']=[0]
try: merge(left,right,allow_same_host=True)
except ValueError: pass
else: raise AssertionError('Missing receiver rank accepted')
print('PASS: four modes, separate rank launch, two event samples per case, all-rank merge and rejection guards')
