"""Export locked recursive Git dependencies locally; import without network access."""
import argparse
import json
from pathlib import Path
import subprocess
import re
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
    assets=[]
    json_path='third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp'
    if any(path=='third-party/shmem' for path,revision in links(source)):
        header=source/json_path
        if not header.is_file():raise RuntimeError('Prepare nlohmann json v3.11.3 on the networked machine before export')
        text=header.read_text()
        for name,value in [('MAJOR',3),('MINOR',11),('PATCH',3)]:
            if not re.search(r'#define\s+NLOHMANN_JSON_VERSION_'+name+r'\s+'+str(value)+r'\b',text):
                raise RuntimeError('Expected nlohmann json v3.11.3')
        asset=output/'nlohmann-json-v3.11.3.hpp';asset.write_bytes(header.read_bytes())
        assets.append(dict(path=json_path,file=asset.name,sha256=sha(asset)))
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
    write_json(output/'deps.json',dict(root_revision=git(source,'rev-parse','HEAD'),entries=entries,assets=assets))

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
    for asset in data.get('assets',[]):
        if asset['path']!='third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp':raise ValueError('Unknown offline asset')
        file=bundle/asset['file'];target=root/asset['path']
        if file.parent!=bundle or sha(file)!=asset['sha256']:raise ValueError('Offline asset digest mismatch')
        if target.exists():raise ValueError('Refuse to overwrite offline asset')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
    if (root/'third-party/shmem').exists() and not (root/'third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp').is_file():
        raise ValueError('Missing uploaded nlohmann header; no network fallback')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','unpack'])
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args()
    if a.action=='pack':pack(a.checkout,a.bundle.resolve())
    else:unpack(a.bundle,a.checkout)

if __name__=='__main__':main()
