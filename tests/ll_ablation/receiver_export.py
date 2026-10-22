"""Offline receiver evidence in independently valid JSON files below a text limit.

ASCII JSON makes byte count, Unicode character count and UTF-16 unit count equal.
No raw captures are changed. Summary/rank parts come before optional core parts.
"""
import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT
from receiver_report import CASES, report

FORMAT = 'll-receiver-transfer-v1'
DEFAULT_LIMIT = 60000
HARD_LIMIT = 65535
CORE_COLUMNS = ['core', 'baseline_ticks', 'skip_flags_ticks', 'no_output_ticks',
                'flag_batches', 'payload_bytes']


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, separators=(',', ':'),
                       sort_keys=True, allow_nan=False) + '\n').encode('ascii')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def envelope(dataset, kind, records, part=99999999, parts=99999999):
    return dict(format=FORMAT, dataset=dataset, part=part, parts=parts,
                kind=kind, records_sha256=digest(records), records=records)


def fragments(dataset, kind, record, limit):
    """Rare oversized records remain recoverable, never truncated or dropped."""
    if len(encoded(envelope(dataset, kind, [record]))) <= limit:
        return [record]
    text=encoded(record).decode('ascii')
    record_hash=hashlib.sha256(text.encode('ascii')).hexdigest()
    width=max(1,(limit-1000)//3)
    while width:
        pieces=[text[i:i+width] for i in range(0,len(text),width)]
        result=[dict(fragment_record_sha256=record_hash, fragment=i+1,
                     fragments=len(pieces), json_text=piece) for i,piece in enumerate(pieces)]
        if all(len(encoded(envelope(dataset,kind,[r]))) <= limit for r in result):
            return result
        width//=2
    raise ValueError('Transfer limit too small for a record envelope')


def group_parts(dataset, kind, records, limit):
    parts=[];current=[]
    for record in records:
        for piece in fragments(dataset,kind,record,limit):
            proposed=current+[piece]
            if current and len(encoded(envelope(dataset,kind,proposed))) > limit:
                parts.append((kind,current));current=[]
            current.append(piece)
    if current:parts.append((kind,current))
    return parts


def evidence_groups(result):
    cases={case:f'case_{i:03d}' for i,case in enumerate(sorted(result['summary']))}
    metadata={key:value for key,value in result.items() if key not in ('summary','samples')}
    metadata.update(case_keys={value:key for key,value in cases.items()},
        core_columns=CORE_COLUMNS,
        core_decoding={
            'arm_order':list(CASES),
            'us':'ticks * 1e6 / frequency_hz',
            'flag_batches':'baseline and no-output; skip-flags is zero (validated)',
            'payload_bytes':'baseline and skip-flags; no-output is zero (validated)',
            'scope':'All per-core replay duration/counter values, not the full raw timer trace',
            'fragments':'Join json_text in fragment order, verify fragment_record_sha256, then parse JSON'},
        send_first='Send all summary parts and rank parts first. Core parts are optional follow-up evidence.')
    summaries=[dict(section='metadata',data=metadata)]
    # Split large case summaries by field, preserving all audit findings.
    for case,summary in sorted(result['summary'].items()):
        for field,value in summary.items():
            summaries.append(dict(case=cases[case],section=field,data=value))
    ranks=[];cores=[]
    for case,rows in sorted(result['samples'].items()):
        for row in sorted(rows,key=lambda r:(r['iteration'],r['rank'])):
            common=dict(case=cases[case],rank=row['rank'],iteration=row['iteration'],order=row['order'])
            ranks.append(dict(**common,**{k:v for k,v in row.items()
                              if k not in ('rank','iteration','order','cores')}))
            values=[];intervals=[]
            for core in sorted(row['cores'],key=lambda c:c['core']):
                base,skip,noout=[core['arms'][name] for name in CASES]
                if (skip['flag_batches']!=0 or noout['payload_bytes']!=0 or
                    base['flag_batches']!=noout['flag_batches'] or base['payload_bytes']!=skip['payload_bytes']):
                    raise ValueError('Cannot compact unequal operation counters')
                values.append([core['core'],base['ticks'],skip['ticks'],noout['ticks'],
                               base['flag_batches'],base['payload_bytes']])
                if any(arm.get('first_tick') is not None or arm.get('last_tick') is not None
                       for arm in (base,skip,noout)):
                    intervals.append([core['core']]+[[arm.get('first_tick'),arm.get('last_tick')]
                                                    for arm in (base,skip,noout)])
            sample=dict(**common,frequency_hz=row['frequency_hz'],cores=values)
            if intervals:sample['optional_sampled_intervals']=intervals
            cores.append(sample)
    return [('summary',summaries),('ranks',ranks),('cores',cores)]


def export_result(result, output, limit=DEFAULT_LIMIT):
    if not 2048 <= limit <= HARD_LIMIT:
        raise ValueError('max-chars must be between2048 and65535')
    if output.exists():raise ValueError('Output directory exists; use a new directory')
    groups=evidence_groups(result)
    dataset=digest(groups)
    packed=[]
    for kind,records in groups:packed.extend(group_parts(dataset,kind,records,limit))
    if len(packed)>=99999999:raise ValueError('Too many transfer parts')
    files=[];entries=[]
    for index,(kind,records) in enumerate(packed,1):
        name=f'{index:03d}-{kind}.json'
        data=encoded(envelope(dataset,kind,records,index,len(packed)))
        if len(data)>limit:raise ValueError('Internal error: oversized transfer part')
        files.append((name,data))
        entries.append(dict(file=name,kind=kind,bytes=len(data),characters=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(format=FORMAT,dataset=dataset,max_chars=limit,parts=len(packed),files=entries,
        send_first=[e['file'] for e in entries if e['kind'] in ('summary','ranks')],
        instructions='Send each listed file separately as pasted text or attachment; do not concatenate files. '
                     'Keep core files for follow-up. SHA256 is over exact ASCII file bytes including final LF.')
    manifest_data=encoded(manifest)
    if len(manifest_data)>limit:raise ValueError('Manifest exceeds limit; export fewer cases per run')
    # All sizes are validated before creating any output. Existing runs stay intact.
    output.mkdir(parents=True)
    for name,data in files:(output/name).write_bytes(data)
    (output/'000-index.json').write_bytes(manifest_data)
    return manifest


def verify(directory):
    manifest=json.loads((directory/'000-index.json').read_bytes())
    if manifest.get('format')!=FORMAT:raise ValueError('Unknown transfer format')
    limit=manifest['max_chars']
    if not 2048<=limit<=HARD_LIMIT:raise ValueError('Invalid transfer limit')
    if (directory/'000-index.json').stat().st_size>limit:raise ValueError('Oversized index')
    if len(manifest['files'])!=manifest['parts']:raise ValueError('Incomplete index')
    groups=[]
    for index,entry in enumerate(manifest['files'],1):
        name=entry['file']
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Invalid part filename')
        data=(directory/name).read_bytes()
        text=data.decode('ascii')
        if len(data)>limit or len(text)!=entry['characters'] or len(data)!=entry['bytes']:
            raise ValueError(f'{name}: size mismatch or exceeded limit')
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError(f'{name}: checksum mismatch')
        part=json.loads(text)
        if (part['format'],part['dataset'],part['part'],part['parts'],part['kind'])!=(
                FORMAT,manifest['dataset'],index,manifest['parts'],entry['kind']):
            raise ValueError(f'{name}: mixed or reordered dataset')
        if digest(part['records'])!=part['records_sha256']:raise ValueError(f'{name}: record checksum mismatch')
        if not groups or groups[-1][0]!=part['kind']:groups.append((part['kind'],[]))
        groups[-1][1].extend(part['records'])
    if [kind for kind,_ in groups]!=['summary','ranks','cores']:
        raise ValueError('Missing, duplicated or reordered evidence groups')
    recovered=[]
    for kind,records in groups:
        originals=[];index=0
        while index<len(records):
            item=records[index]
            if 'fragment_record_sha256' not in item:
                originals.append(item);index+=1;continue
            count=item['fragments']
            if type(count) is not int or count<1 or count>len(records)-index:
                raise ValueError('Incomplete record fragments')
            pieces=records[index:index+count]
            for number,piece in enumerate(pieces,1):
                if (piece.get('fragment'),piece.get('fragments'),piece.get('fragment_record_sha256'))!=(
                        number,count,item['fragment_record_sha256']):
                    raise ValueError('Missing, reordered or mixed record fragments')
            value=''.join(piece['json_text'] for piece in pieces)
            if hashlib.sha256(value.encode('ascii')).hexdigest()!=item['fragment_record_sha256']:
                raise ValueError('Fragment record checksum mismatch')
            originals.append(json.loads(value));index+=count
        recovered.append((kind,originals))
    if digest(recovered)!=manifest['dataset']:raise ValueError('Dataset checksum mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Existing completed ready-breakdown run directory')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--max-chars',type=int,default=DEFAULT_LIMIT)
    parser.add_argument('--verify',action='store_true',help='Verify an already exported directory')
    args=parser.parse_args()
    directory=args.directory if args.directory.is_absolute() else ROOT/args.directory
    if args.verify:
        manifest=verify(directory)
        print(f'Verified {manifest["parts"]} parts; maximum {manifest["max_chars"]} ASCII bytes/characters each')
        return
    if args.output_dir is None:parser.error('--output-dir required for export')
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    manifest=export_result(report(directory),output,args.max_chars)
    verify(output)
    print(f'Exported {manifest["parts"]} parts to {output}')
    print('Send first: 000-index.json, '+', '.join(manifest['send_first']))


if __name__=='__main__':main()
