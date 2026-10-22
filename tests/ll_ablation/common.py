from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
MODES={'baseline':0,'fixed-payload':1,'preload':2,'prepack':3,'ready-replay':4}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def idle():
    import re
    text=subprocess.check_output(['npu-smi','info'],text=True)
    devices=set(re.findall(r'^\|\s*(\d+)\s*\|\s*Ascend',text,re.M))
    empty=set(re.findall(r'No running processes found in NPU (\d+)',text))
    if not devices or devices!=empty:raise RuntimeError('NPU busy or occupancy unknown: no launch')
    return text
