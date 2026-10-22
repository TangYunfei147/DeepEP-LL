"""Advance a verified old ON build worktree while retaining its dependency-tracked cache."""
import json
from pathlib import Path
import subprocess
import os
import uuid
from common import sha,write_json

HEADER='csrc/kernels/dispatch/low_latency/common/ll_ablation_config.h'
STATE='.tmp/receiver-cache-state.json'

def acquire(source):
    lock=source/'.tmp/receiver-build.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    token=f'{os.getpid()} {uuid.uuid4().hex}\n'
    try:
        with lock.open('x') as file:file.write(token)
    except FileExistsError:
        raise RuntimeError(f'Cache worktree is locked: {lock}; do not run concurrent builds') from None
    def release():
        if lock.exists() and lock.read_text()==token:lock.unlink()
    return release

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def common_dir(root):
    value=Path(git(root,'rev-parse','--git-common-dir'))
    return (value if value.is_absolute() else root/value).resolve()

def inspect(root,manifest,revision,send):
    root=root.resolve();manifest=manifest.resolve()
    info=json.loads(manifest.read_text())
    if info['mode']!='ready-replay' or info['timer']!='ON' or info['send_cores']!=send or info['recv_cores']!=56-send:
        raise RuntimeError('Cache donor must be ready-replay ON with the same S/R')
    if sha(manifest.parent/info['wheel'])!=info['wheel_sha256']:raise RuntimeError('Cache donor wheel checksum mismatch')
    source=root/info['source']
    if source.resolve()!=source or not source.is_relative_to(root/'.ablation/sources'):
        raise RuntimeError('Cache donor must be a real isolated worktree under this project')
    if common_dir(source)!=common_dir(root):raise RuntimeError('Cache donor does not share this Git object store')
    sdk=source/'third-party/shmem/install/shmem/deepep-sdk.json'
    if not sdk.is_file():raise RuntimeError('Cache donor SDK missing; use --fresh-build')
    state=json.loads((source/STATE).read_text()) if (source/STATE).exists() else None
    expected_commit=state['commit'] if state else info['commit']
    expected_header=state['header_sha256'] if state else info['header_sha256']
    if state and state['donor_wheel_sha256']!=info['wheel_sha256']:raise RuntimeError('Cache state donor mismatch')
    if git(source,'rev-parse','HEAD')!=expected_commit:raise RuntimeError('Cache worktree revision changed externally')
    if sha(source/HEADER)!=expected_header:raise RuntimeError('Cache config changed externally; refusing overwrite')
    staged=git(source,'diff','--cached','--name-only','--ignore-submodules=dirty','HEAD')
    changed=set(git(source,'diff','--name-only','--ignore-submodules=dirty','HEAD').splitlines())
    untracked=git(source,'ls-files','--others','--exclude-standard')
    if staged or changed-{HEADER} or untracked:
        raise RuntimeError('Cache worktree has user changes/untracked files; use --fresh-build')
    if git(root,'diff',expected_commit,revision,'--','dependencies.lock.json','.gitmodules'):
        raise RuntimeError('Dependency identity changed; cached SDK cannot be selected automatically')
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',expected_commit,revision],check=True)
    return source,info

def advance(root,manifest,revision,send,mode):
    source,info=inspect(root,manifest,revision,send)
    header=source/HEADER
    base=git(source,'show','HEAD:'+HEADER)+'\n'
    head=git(source,'rev-parse','HEAD')
    if head!=revision:
        header.write_text(base,encoding='utf-8',newline='\n')
        subprocess.run(['git','-C',str(source),'merge','--ff-only',revision],check=True)
    base=git(source,'show','HEAD:'+HEADER)+'\n'
    target=base.replace('kLlAblationMode = 0;',f'kLlAblationMode = {mode};')
    if header.read_text()!=target:header.write_text(target,encoding='utf-8',newline='\n')
    (source/STATE).parent.mkdir(parents=True,exist_ok=True)
    write_json(source/STATE,dict(commit=revision,header_sha256=sha(header),donor_wheel_sha256=info['wheel_sha256']))
    return source
