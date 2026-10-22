"""Lossless compressed timer transfer; all processing is offline, with no NPU work."""
from collections import defaultdict
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

FORMAT = 'll-timer-transfer-v2'
LEGACY_FORMAT = 'll-timer-transfer-v1'
SAMPLE_FORMAT = 'll-timer-samples-v2'
DEFAULT_LIMIT = 60000
MAX_LIMIT = 65535
NOTES = (
    'Raw ticks and calls are retained. Missing stages/counters are absent, never imputed as zero.',
    'Stage totals may nest or overlap; do not sum them. Submission is not DMA completion.',
    'Only recorded real intervals are exported; typically first two calls, not a full packet timeline.',
    'No cross-rank clock alignment or subtraction. Kernel spans exclude uninstrumented final synchronization/export.',
    'Activity comes from observed TX/RX/relay counters; header roles alone do not establish activity.',
    'ON diagnostics are not an OFF performance baseline; low ready fraction alone does not prove congestion.',
)


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True,
                      allow_nan=False)


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _tables(kind, sample_id, rows, budget):
    """Keep exact key presence/order of rows, splitting tables before size limits."""
    fields = None
    table = None
    for row in rows:
        keys = sorted(row)
        if keys != fields or (table and len(table['rows']) >= 64):
            if table:
                yield table
            fields = keys
            table = dict(kind=kind, sample=sample_id, fields=fields, rows=[])
        values = [row[key] for key in fields]
        candidate = dict(table, rows=table['rows'] + [values])
        if len(_json(candidate)) > budget:
            if table['rows']:
                yield table
                table = dict(kind=kind, sample=sample_id, fields=fields, rows=[])
                candidate = dict(table, rows=[values])
            if len(_json(candidate)) > budget:
                raise ValueError(f'{kind}: one row exceeds transfer limit; no data was truncated')
        table = candidate
    if table:
        yield table


def _full_records(report, samples, budget):
    derived = {'communication', 'overlap_receive', 'input_wait_bottleneck', 'relay_timing',
               'per_rank_core_stage', 'per_rank_stage_max', 'per_expert',
               'prefix_diagnosis', 'prefix_diagnostics'}
    yield dict(kind='report_context', data={k: v for k, v in report.items() if k not in derived})
    for sample_id, (meta, decoded) in enumerate(samples):
        yield dict(kind='sample', sample=sample_id, metadata=meta,
                   decoded={k: v for k, v in decoded.items() if k not in ('records', 'intervals')},
                   expected_stage_rows=len(decoded.get('records', [])),
                   expected_interval_rows=len(decoded.get('intervals', [])),
                   records_present='records' in decoded, intervals_present='intervals' in decoded)
        yield from _tables('stages', sample_id, decoded.get('records', []), budget)
        yield from _tables('intervals', sample_id, decoded.get('intervals', []), budget)


def _load(name):
    spec=importlib.util.spec_from_file_location('ll_transfer_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


codec=_load('text_codec')


def _safe_name(name):
    if not isinstance(name,str) or not name or '/' in name or '\\' in name or ':' in name or name in ('.','..'):
        raise ValueError('Invalid transfer filename')
    return name


def _check_profiles(samples):
    profiles=defaultdict(set)
    for _,decoded in samples:
        profiles[decoded.get('operation')].add(decoded.get('timer_profile','full' if decoded.get('schema',0)<=13 else 'unknown'))
    if any(len(values)>1 for values in profiles.values()):
        raise ValueError('Mixed timer profiles cannot be aggregated; export each capture profile separately')


def encode_samples(report,samples):
    """Table-encode every decoded field; no sampling, ranking or rounding."""
    samples=list(samples);_check_profiles(samples)
    payload=dict(format=SAMPLE_FORMAT,samples=len(samples),records=list(_full_records(report,samples,codec.MAX_RAW_BYTES)))
    raw=_json(payload).encode('utf-8')
    if len(raw)>codec.MAX_RAW_BYTES: raise ValueError('Case payload exceeds512MiB; split capture iterations/ranks explicitly')
    return raw


def _decode_records(records,count):
    if type(count) is not int or count<0 or not isinstance(records,list) or count>len(records): raise ValueError('Invalid sample count/records')
    headers={};rows=defaultdict(lambda:defaultdict(list));context=None
    for record in records:
        if not isinstance(record,dict): raise ValueError('Invalid transfer record')
        kind=record.get('kind')
        if kind=='report_context':
            if context is not None: raise ValueError('Duplicate report context')
            context=record['data']
        elif kind=='sample':
            sample=record['sample']
            if type(sample) is not int or not 0<=sample<count or sample in headers: raise ValueError('Duplicate or invalid sample header')
            headers[sample]=record
        elif kind in ('stages','intervals'):
            sample=record['sample'];fields=record['fields']
            if type(sample) is not int or not 0<=sample<count: raise ValueError('Invalid table sample')
            if not isinstance(fields,list) or any(not isinstance(f,str) for f in fields) or len(set(fields))!=len(fields):
                raise ValueError('Invalid table fields')
            for values in record['rows']:
                if len(values)!=len(fields): raise ValueError('Malformed table row')
                rows[sample][kind].append(dict(zip(fields,values)))
        else: raise ValueError('Unknown transfer record kind')
    if context is None or set(headers)!=set(range(count)): raise ValueError('Missing context/sample headers')
    result=[]
    for sample in range(count):
        h=headers[sample];decoded=dict(h['decoded'])
        for kind,key,present,expected in [('stages','records','records_present','expected_stage_rows'),
                                           ('intervals','intervals','intervals_present','expected_interval_rows')]:
            if len(rows[sample][kind])!=h[expected]: raise ValueError('Sample row count mismatch')
            if h[present]: decoded[key]=rows[sample][kind]
            elif rows[sample][kind]: raise ValueError('Rows for absent field')
        result.append((h['metadata'],decoded))
    _check_profiles(result)
    return context,result


def decode_samples(payload):
    if not isinstance(payload,bytes) or len(payload)>codec.MAX_RAW_BYTES: raise ValueError('Invalid sample payload size')
    data=json.loads(payload)
    if data.get('format')!=SAMPLE_FORMAT: raise ValueError('Unsupported sample payload format')
    return _decode_records(data['records'],data['samples'])


def _coverage(report,samples):
    world=report.get('world_size');iters=(report.get('sampling') or {}).get('diagnostic_iters')
    if iters is None:
        observed=[m.get('iteration') for m,_ in samples if type(m.get('iteration')) is int]
        iters=max(observed)+1 if observed else None
    expected={(r,i) for r in range(world) for i in range(iters)} if type(world) is int and type(iters) is int else None
    actual=defaultdict(list);cores=[]
    for meta,decoded in samples:
        actual[decoded.get('operation')].append((meta.get('rank'),meta.get('iteration')))
        seen={r['core'] for r in decoded.get('records',[]) if r['stage']=='KERNEL'}
        cores.append(len(seen)==decoded.get('active_aiv'))
    coverage={op:dict(observed_samples=len(pairs),unique_samples=len(set(pairs)),
        expected_samples=len(expected) if expected is not None else None,
        complete=set(pairs)==expected and len(pairs)==len(set(pairs)) if expected is not None else None) for op,pairs in actual.items()}
    return coverage,all(cores) if cores else None


def _bounded(content,limit):
    raw=content.encode('utf-8')
    return len(content)<=limit and len(raw)+content.count('\n')<=limit


def _overview(manifest,report,samples):
    lines=['# LL compressed transfer', '', 'Send this overview and every part_*.txt; no separate priority/full series.',
           f"Bundle: {manifest['bundle_id']}; parts: {manifest['series_counts']['compressed']}",
           f"Payload: {manifest['transport']['raw_bytes']} bytes; compressed: {manifest['transport']['compressed_bytes']} bytes; text including CRLF: {manifest['transmission_bytes_crlf']} bytes.",
           f"Samples: {len(samples)}; stage rows: {manifest['stage_rows']}; real interval rows: {manifest['interval_rows']}",
           'Profiles: '+_json(sorted({d.get('timer_profile','full') for _,d in samples})),
           'Coverage: '+_json(manifest['source_sample_coverage']), '',
           'All raw ticks, counts, metadata and sampled intervals are in the compressed payload.',
           'Only this readable overview is selective; no underlying data are discarded.', '',
           '| operation | rank | iteration | core | KERNEL ticks |', '|---|---:|---:|---:|---:|']
    longest=[]
    for meta,decoded in samples:
        rows=[r for r in decoded.get('records',[]) if r['stage']=='KERNEL']
        if rows:
            row=max(rows,key=lambda r:r['total_ticks'])
            longest.append((row['total_ticks'],decoded['operation'],meta.get('rank'),meta.get('iteration'),row['core']))
    for ticks,operation,rank,iteration,core in sorted(longest,reverse=True)[:8]:
        lines.append(f'| {operation} | {rank} | {iteration} | {core} | {ticks} |')
    lines+=['',*['- '+note for note in NOTES], '', 'Restore: python tests/ll_stage_timer/transfer_report.py restore copied_part*.txt --output-dir restored_case', '']
    text='\n'.join(lines)
    if not _bounded(text,min(manifest['limit'],8000)):
        # Extremely wide coverage metadata stays in the payload, not silently lost.
        lines=[line for line in lines if not line.startswith('Coverage:')]
        text='\n'.join(lines)
    if not _bounded(text,manifest['limit']): raise ValueError('Overview exceeds limit')
    return text


def export_bundle(output_dir,report,samples,limit=DEFAULT_LIMIT):
    samples=list(samples)
    payload=encode_samples(report,samples)
    transport,parts=codec.make_parts(payload,limit)
    bundle_id=transport['raw_sha256'][:20]
    names=[f'part_{i:04d}_of_{len(parts):04d}.txt' for i in range(1,len(parts)+1)]
    coverage,cores=_coverage(report,samples)
    manifest=dict(format=FORMAT,bundle_id=bundle_id,limit=limit,transport=transport,
        send_first=['00_overview.md',*names],parts=names,series_counts=dict(compressed=len(parts)),
        samples=len(samples),stage_rows=sum(len(d.get('records',[])) for _,d in samples),
        interval_rows=sum(len(d.get('intervals',[])) for _,d in samples),
        transmission_bytes_lf=sum(len(p.encode('ascii')) for p in parts),
        transmission_bytes_crlf=sum(len(p)+p.count('\n') for p in parts),
        export_complete=True,source_sample_coverage=coverage,all_observed_kernel_cores_complete=cores,notes=list(NOTES))
    files=dict(zip(names,parts));files['00_overview.md']=_overview(manifest,report,samples)
    files['manifest.json']=_json(manifest)+'\n'
    if any(not _bounded(content,limit) for content in files.values()): raise ValueError('Transfer metadata exceeds limit')
    output=Path(output_dir)/'transfer'
    if output.exists():
        if not output.is_dir() or set(p.name for p in output.iterdir())!=set(files):
            raise ValueError('transfer contains a different bundle; choose a new analysis output directory')
        for name,content in files.items():
            if (output/name).read_bytes()!=content.encode('utf-8'):
                raise ValueError('transfer contains a different bundle; choose a new analysis output directory')
        return manifest
    output.mkdir(parents=True,exist_ok=False)
    for name,content in files.items():
        (output/name).write_bytes(content.encode('utf-8'))
    return manifest


def _restore_legacy(directory):
    """Verify all full parts and return exact (report context, decoded samples)."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    records = []
    all_parts = []
    for ordinal, entry in enumerate(manifest['part_index'], 1):
        raw = (directory / _safe_name(entry['file'])).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Part index hash mismatch')
        index = json.loads(raw)
        if (index['bundle_id'] != manifest['bundle_id'] or index['part'] != ordinal
                or index['parts'] != len(manifest['part_index']) or index['series'] != 'index'):
            raise ValueError('Part index identity mismatch')
        all_parts.extend(r['data'] for r in index['records'])
    parts = [p for p in all_parts if p['series'] == 'full']
    if len(parts) != manifest['series_counts']['full']:
        raise ValueError('Full part count mismatch')
    if [p['part'] for p in parts] != list(range(1, len(parts) + 1)):
        raise ValueError('Missing, duplicate or reordered full parts')
    for expected in parts:
        raw = (directory / _safe_name(expected['file'])).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Full part hash mismatch')
        part = json.loads(raw)
        if (part['bundle_id'] != manifest['bundle_id'] or part['part'] != expected['part']
                or part['parts'] != len(parts) or part['series'] != 'full'):
            raise ValueError('Full part identity mismatch')
        records.extend(part['records'])
    if len(records) != manifest['full_record_count'] or _hash(_json(records)) != manifest['full_payload_sha256']:
        raise ValueError('Full payload incomplete or modified')
    return _decode_records(records,manifest['samples'])


def restore_samples(directory):
    """Read compressed v2 and historical uncompressed v1 bundles."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format')==LEGACY_FORMAT: return _restore_legacy(directory)
    if manifest.get('format')!=FORMAT: raise ValueError('Unsupported transfer format')
    names=manifest['parts']
    if len(set(names))!=len(names): raise ValueError('Duplicate part filename')
    status,payload=codec.inspect_parts([(directory/_safe_name(n)).read_text(encoding='utf-8') for n in names])
    if payload is None: raise ValueError('Missing parts: '+str(status['missing_parts']))
    if status['raw_sha256']!=manifest['transport']['raw_sha256'] or status['raw_sha256'][:20]!=manifest['bundle_id']:
        raise ValueError('Payload identity mismatch')
    return decode_samples(payload)


def prepare_restored(context,samples,trace_iteration=0,render=True):
    """Validate and render offline files in memory before touching the output directory."""
    samples=list(samples);files={'context.json':_json(context)+'\n',
        'decoded_samples.json':_json([dict(metadata=m,decoded=d) for m,d in samples])+'\n'}
    if render:
        trace=_load('transfer_trace').trace_from_decoded(samples,trace_iteration)
        files['trace.json']=_json(trace)+'\n'
        # Complete exports are reanalyzed with the same aggregation code as raw captures.
        if samples and all(m.get('correctness')=='passed' and 'world_size' in m for m,_ in samples):
            analysis=_load('analyze');report=analysis.analyze_samples(samples)
            files['summary.json']=_json(report)+'\n'
            files['communication_bottleneck.md']=analysis.communication_report.markdown(report['communication'])
        else:
            files['analysis_status.json']=_json(dict(status='unavailable',reason='Source metadata lacks complete correctness/world identity; decoded data and real intervals retained.'))+'\n'
    return files


def write_restored(output,context,samples,trace_iteration=0,render=True):
    files=prepare_restored(context,samples,trace_iteration,render)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,content in files.items(): (output/name).write_bytes(content.encode('utf-8'))
    return output


def _read_inputs(paths):
    if '-' in paths:
        if paths!=['-']: raise ValueError('Use - alone for stdin')
        return [sys.stdin.read()]
    return [Path(p).read_text(encoding='utf-8-sig') for p in paths]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    show=commands.add_parser('show',help='Print exactly one checked part')
    show.add_argument('directory',type=Path);show.add_argument('--part',type=int,required=True)
    show.add_argument('--group',type=int,help='Suite group ordinal; omit only for a one-group suite')
    inspect=commands.add_parser('inspect',help='Check pasted parts and list missing pieces')
    inspect.add_argument('texts',nargs='+')
    restore=commands.add_parser('restore',help='Restore copied parts and generate trace locally')
    restore.add_argument('texts',nargs='+');restore.add_argument('--output-dir',required=True,type=Path)
    restore.add_argument('--trace-iteration',type=int,default=0);restore.add_argument('--no-render',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='show':
            manifest=json.loads((args.directory/'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('format')==FORMAT:
                if args.group is not None: raise ValueError('--group is only for suite bundles')
                if not 1<=args.part<=len(manifest['parts']): raise ValueError('Invalid part number')
                name=_safe_name(manifest['parts'][args.part-1]);expected=manifest['transport']['raw_sha256']
            elif manifest.get('format')=='ll-timer-suite-v1':
                group=args.group or (1 if len(manifest['groups'])==1 else None)
                if group is None or not 1<=group<=len(manifest['groups']): raise ValueError('Supply a valid --group ordinal')
                info=manifest['groups'][group-1];count=info['part_count']
                if not 1<=args.part<=count: raise ValueError('Invalid part number')
                name=f'suite_{group:03d}_part_{args.part:03d}_of_{count:03d}.txt';expected=info['payload_sha256']
            else: raise ValueError('show requires a compressed manifest')
            text=(args.directory/name).read_text(encoding='utf-8')
            status,_=codec.inspect_parts([text])
            if status['raw_sha256']!=expected: raise ValueError('Part identity mismatch')
            sys.stdout.write(text)
        elif args.command=='inspect':
            states=[codec.inspect_parts(group)[0] for group in codec.group_parts(_read_inputs(args.texts))]
            print(json.dumps(states,indent=2))
        else:
            groups=codec.group_parts(_read_inputs(args.texts))
            status,payload=codec.inspect_parts(groups[0])
            if payload is None: raise ValueError('Missing parts: '+str(status['missing_parts']))
            kind=json.loads(payload).get('format')
            if kind==SAMPLE_FORMAT:
                if len(groups)!=1: raise ValueError('Restore different cases separately, or use a suite bundle')
                context,samples=decode_samples(payload)
                write_restored(args.output_dir,context,samples,args.trace_iteration,not args.no_render)
            elif kind=='ll-timer-suite-v1':
                del payload
                _load('suite_transfer').restore_text_groups(groups,args.output_dir,
                    trace_iteration=args.trace_iteration,render=not args.no_render)
            else: raise ValueError('Unsupported or mixed payload formats')
            print('Verified and restored: '+str(args.output_dir))
    except (ValueError,OSError,KeyError,TypeError) as error:
        parser.exit(1,str(error)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
