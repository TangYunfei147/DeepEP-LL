"""Lossless offline trace transport through text blocks smaller than65535 characters.

Export a completed receiver run (full raw captures), or a verified legacy receiver
JSON export. Restore original files before rendering actual recorded intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from common import ROOT
import receiver_export
from receiver_report import report
from trace_transfer_codec import encode_bundle, inspect_parts, decode_bundle

FORMAT='ll-trace-files-v1'
MAX_INPUT_BYTES=128*1024*1024
MAX_FILES=4096
NOTES=[
    'All listed files are preserved byte-for-byte. No events or cores are sampled during transport.',
    'Raw device clocks are rank-local. Captures have separate origins; no cross-rank latency is inferred.',
    'Rendering includes actual recorded intervals only. Missing timestamps and cumulative-only stages are not expanded.',
    'Receiver replay is conditioned and instrumented; use separate OFF full-dispatch measurements for production gains.',
]


def sha(data):return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')


def project_path(path):return path if path.is_absolute() else ROOT/path


def safe_name(name):
    if not isinstance(name,str):raise ValueError('File name must be text')
    path=PurePosixPath(name)
    if (path.is_absolute() or str(path)!=name or not path.parts or '\\' in name or ':' in name or
        any(ord(c)<32 for c in name) or any(p in ('','.','..') or p.endswith((' ','.')) for p in path.parts) or
        any(re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]',p.split('.')[0],re.I) for p in path.parts) or
        path.suffix!='.json' or path.parts[0] in ('rendered','transfer_manifest.json')):
        raise ValueError(f'Invalid restored JSON path: {name}')
    return path


def collect(source):
    source=source.resolve()
    if (source/'000-index.json').is_file():
        index=receiver_export.verify(source)
        names=['000-index.json']+[item['file'] for item in index['files']]
        kind='receiver-transfer'
    else:
        result=report(source)
        names=['ablation_identity.json','summary.json']
        names+=sorted({r['source_file'] for rows in result['samples'].values() for r in rows})
        names+=sorted(p.name for p in source.glob('ablation_metadata_rank*.json')
                      if re.fullmatch(r'ablation_metadata_rank\d+\.json',p.name))
        kind='receiver-run'
    if len(names)>MAX_FILES:raise ValueError('Too many source files')
    files=[];size=0
    for name in names:
        safe_name(name);path=source/name
        if not path.resolve().is_relative_to(source):raise ValueError('Source file resolves outside selected directory')
        size+=path.stat().st_size
        if size>MAX_INPUT_BYTES:raise ValueError('Input exceeds128MiB; select a smaller capture')
        data=path.read_bytes()
        # JSON is the only collected file type; no logs, environment or credentials.
        text=data.decode('utf-8');json.loads(text.lstrip('\ufeff'))
        files.append(dict(name=name,bytes=len(data),sha256=sha(data),text=text))
    return dict(format=FORMAT,source_kind=kind,label=source.name,notes=NOTES,files=files)


def validate_bundle(bundle):
    if bundle.get('format')!=FORMAT or bundle.get('source_kind') not in ('receiver-run','receiver-transfer'):
        raise ValueError('Unsupported trace file bundle')
    records=bundle.get('files')
    if not isinstance(records,list) or not 0<len(records)<=MAX_FILES:raise ValueError('Invalid file count')
    files={};size=0
    for row in records:
        name=row['name'];safe_name(name)
        if name in files:raise ValueError('Duplicate restored file name')
        data=row['text'].encode('utf-8');size+=len(data)
        if size>MAX_INPUT_BYTES:raise ValueError('Restored files exceed128MiB')
        if len(data)!=row['bytes'] or sha(data)!=row['sha256']:raise ValueError(f'File checksum mismatch: {name}')
        json.loads(row['text'].lstrip('\ufeff'));files[name]=data
    for name in files:
        if any(str(p) in files for p in PurePosixPath(name).parents):raise ValueError('Conflicting file/directory paths')
    required={'000-index.json'} if bundle['source_kind']=='receiver-transfer' else {'ablation_identity.json','summary.json'}
    if not required<=files.keys():raise ValueError('Missing source identity files')
    return files


def export_text(source,output,max_bytes=60000):
    if output.exists():raise ValueError('Output directory exists; choose a new directory')
    bundle=collect(source)
    codec,parts=encode_bundle(bundle,max_bytes=max_bytes)
    # Round-trip before publishing files, including original byte identities.
    status,compressed=inspect_parts(parts)
    restored=decode_bundle(compressed,expected_raw_bytes=status['raw_bytes'],expected_raw_sha256=status['raw_sha256'])
    if validate_bundle(restored)!=validate_bundle(bundle):raise ValueError('Internal round-trip mismatch')
    index=dict(format=FORMAT,source_kind=bundle['source_kind'],label=bundle['label'],max_bytes=max_bytes,
               codec=codec,files=[{k:v for k,v in row.items() if k!='text'} for row in bundle['files']],parts=[])
    output.mkdir(parents=True)
    for number,text in enumerate(parts,1):
        data=text.encode('ascii');name=f'part-{number:06d}.txt'
        (output/name).write_bytes(data)
        index['parts'].append(dict(file=name,sha256=sha(data),bytes=len(data),crlf_bytes=len(text.replace('\n','\r\n').encode('ascii'))))
    # A failed/interrupted export has no completed manifest and cannot be shown.
    (output/'manifest.json').write_bytes(json_bytes(index))
    return index


def show_part(directory,number):
    index=json.loads((directory/'manifest.json').read_bytes())
    if index.get('format')!=FORMAT or not 1<=number<=len(index['parts']):raise ValueError('Invalid part selection')
    data=(directory/f'part-{number:06d}.txt').read_bytes();entry=index['parts'][number-1]
    if sha(data)!=entry['sha256']:raise ValueError('Part changed since export')
    text=data.decode('ascii')
    if len(text.replace('\n','\r\n').encode('ascii'))>min(index['max_bytes'],65535):raise ValueError('Part exceeds text limit')
    inspect_parts([text])
    return text


def restore_text(texts,output,render=True):
    if output.exists():raise ValueError('Output directory exists; choose a new directory')
    status,compressed=inspect_parts(texts)
    if compressed is None:raise ValueError(f'Missing parts: {status["missing_parts"]}; no output written')
    bundle=decode_bundle(compressed,expected_raw_bytes=status['raw_bytes'],expected_raw_sha256=status['raw_sha256'])
    files=validate_bundle(bundle)
    output.mkdir(parents=True)
    for name,data in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    # Semantic validation before claiming restoration succeeded or rendering.
    if bundle['source_kind']=='receiver-transfer':receiver_export.verify(output)
    else:report(output)
    (output/'transfer_manifest.json').write_bytes(json_bytes(dict(status=status,format=FORMAT,
        source_kind=bundle['source_kind'],notes=NOTES,files=[{k:v for k,v in row.items() if k!='text'} for row in bundle['files']])))
    rendered=[]
    if render:
        from receiver_trace import render_bundle
        rendered=render_bundle(output,bundle['source_kind'])
    return status,rendered


def read_texts(paths):
    if '-' in paths:
        if paths!=['-']:raise ValueError("Use '-' alone for stdin")
        return [sys.stdin.read()]
    return [project_path(Path(p)).read_text(encoding='utf-8-sig') for p in paths]


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('source',type=Path);e.add_argument('--output-dir',type=Path,required=True)
    e.add_argument('--max-bytes',type=int,default=60000)
    s=sub.add_parser('show');s.add_argument('directory',type=Path);s.add_argument('--part',type=int,required=True)
    i=sub.add_parser('inspect');i.add_argument('texts',nargs='+')
    r=sub.add_parser('restore');r.add_argument('texts',nargs='+');r.add_argument('--output-dir',type=Path,required=True)
    r.add_argument('--no-render',action='store_true')
    a=p.parse_args()
    try:
        if a.command=='export':
            output=project_path(a.output_dir);index=export_text(project_path(a.source),output,a.max_bytes)
            print(f'Exported {len(index["files"])} original files into {len(index["parts"])} compressed text parts: {output}')
            print(f'Each whole block fits {a.max_bytes} ASCII characters/bytes including CRLF. Use show <directory> --part 1, then --part 2, etc.')
        elif a.command=='show':sys.stdout.write(show_part(project_path(a.directory),a.part))
        elif a.command=='inspect':
            status,_=inspect_parts(read_texts(a.texts));print(json.dumps(status,indent=2))
        else:
            status,rendered=restore_text(read_texts(a.texts),project_path(a.output_dir),not a.no_render)
            print(f'All {status["part_count"]} parts verified; restored to {project_path(a.output_dir)}')
            for path in rendered:print(path)
    except (ValueError,OSError,KeyError,TypeError,UnicodeError) as error:p.exit(1,f'Trace transfer error: {error}\n')


if __name__=='__main__':main()
