"""Export locked recursive Git dependencies locally; import without network access."""
import argparse
import json
from pathlib import Path
import subprocess
from common import sha,write_json

def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def links(root):
    result=[]
    for line in git(root,'ls-tree','-r','HEAD').splitlines():
        mode,kind,rest=line.split(' ',2)
        if mode=='160000':
            revision,path=rest.split('\t',1);result.append((path,revision))
    return result

def pack(source,output):
    source=source.resolve();output.mkdir(parents=True,exist_ok=False)
    entries=[]
    def visit(parent,prefix=''):
        for path,revision in links(parent):
            child=parent/path;relative=str(Path(prefix)/path).replace('\\','/')
            if Path(git(child,'rev-parse','--show-toplevel')).resolve()!=child.resolve():
                raise RuntimeError(f'Uninitialized dependency {relative}')
            if git(child,'rev-parse','HEAD')!=revision:raise RuntimeError(f'Wrong revision {relative}')
            if git(child,'rev-parse','--is-shallow-repository')=='true':
                raise RuntimeError(f'Unshallow {relative} on the networked machine before export')
            if git(child,'status','--porcelain'):raise RuntimeError(f'Dirty dependency {relative}')
            file=output/f'{len(entries):03}-{revision}.bundle'
            subprocess.run(['git','-C',str(child),'bundle','create',str(file),'HEAD'],check=True)
            entries.append(dict(path=relative,revision=revision,file=file.name,sha256=sha(file)))
            visit(child,relative)
    visit(source)
    write_json(output/'deps.json',dict(root_revision=git(source,'rev-parse','HEAD'),entries=entries))

def unpack(bundle,root):
    bundle=bundle.resolve();root=root.resolve()
    data=json.loads((bundle/'deps.json').read_text())
    # Every dependency revision must match the receiving parent gitlink.
    for entry in data['entries']:
        rel=Path(entry['path']);file=bundle/entry['file']
        if rel.is_absolute() or '..' in rel.parts or file.parent!=bundle:raise ValueError('Unsafe dependency path')
        target=root/rel
        if target.resolve()!=target or root not in target.parents:raise ValueError('Dependency escapes checkout')
        if sha(file)!=entry['sha256']:raise ValueError('Dependency bundle digest mismatch')
        parent=target.parent
        while parent!=root and not (parent/'.git').exists():parent=parent.parent
        pinned=dict(links(parent)).get(str(target.relative_to(parent)).replace('\\','/'))
        if pinned!=entry['revision']:raise ValueError('Bundle differs from pinned gitlink')
        if target.exists() and any(target.iterdir()):raise ValueError(f'Refuse to overwrite {target}')
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init',str(target)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(target),'fetch',str(file),'HEAD'],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',entry['revision']],check=True)
    # Reject an incomplete recursive bundle rather than fetching missing repos.
    def check(parent):
        for path,revision in links(parent):
            child=parent/path
            if not (child/'.git').exists() or git(child,'rev-parse','HEAD')!=revision:raise ValueError('Incomplete offline dependencies')
            check(child)
    check(root)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','unpack'])
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args()
    if a.action=='pack':pack(a.checkout,a.bundle.resolve())
    else:unpack(a.bundle,a.checkout)

if __name__=='__main__':main()
