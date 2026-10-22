from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
MODES={'baseline':0,'fixed-payload':1,'preload':2,'prepack':3,'ready-replay':4,'ready-breakdown':5}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def source_checkout(root=ROOT):
    """Build committed HEAD; do not confuse dependency artifacts with source edits."""
    changed=set()
    for cached in (False,True):
        command=['git','-C',str(root),'diff','--name-only','-z','--ignore-submodules=dirty']
        if cached:command.append('--cached')
        command.append('HEAD')
        changed.update(p for p in subprocess.check_output(command,text=True).split('\0') if p)
    if changed:
        raise RuntimeError('Tracked source or pinned submodule revision differs from HEAD:\n  '+
            '\n  '.join(sorted(changed))+
            '\nBuild uses committed HEAD. Commit intended changes or use a clean checkout; '
            'no reset/clean/stash was performed.')
    status=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=normal'],text=True)
    if status:
        print('Building committed HEAD. Untracked files and dependency working-tree changes '
              'remain untouched and are NOT copied into isolated variants.',flush=True)
    return dict(source_policy='committed_HEAD',excluded_worktree_status=status.splitlines())
def idle():
    import re
    text=subprocess.check_output(['npu-smi','info'],text=True)
    devices=set(re.findall(r'^\|\s*(\d+)\s*\|\s*Ascend',text,re.M))
    empty=set(re.findall(r'No running processes found in NPU (\d+)',text))
    if not devices or devices!=empty:raise RuntimeError('NPU busy or occupancy unknown: no launch')
    return text
