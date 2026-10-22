"""Install one verified experiment wheel into an explicitly activated venv."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,sha

p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
manifest=a.manifest if a.manifest.is_absolute() else ROOT/a.manifest
data=json.loads(manifest.read_text());wheel=manifest.parent/data['wheel']
if sha(wheel)!=data['wheel_sha256']:raise RuntimeError('Wheel digest mismatch')
if sys.prefix==sys.base_prefix:raise RuntimeError('Activate a project venv first; system installation is refused')
subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps',str(wheel)],check=True)
