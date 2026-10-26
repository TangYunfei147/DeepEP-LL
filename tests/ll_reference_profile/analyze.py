"""CPU-only analysis of independent LL dispatch reference-cycle captures."""
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timer = _load('ll_reference_profile_parser', ROOT/'deep_ep/utils/ll_reference_profile_parse.py')
exporter = _load('ll_reference_profile_exporter', Path(__file__).with_name('export_xlsx.py'))
COMMON_KEYS = ('world_size', 'config', 'run_id', 'case_id', 'seed', 'sampling', 'route',
               'comparison', 'source', 'profile_seed', 'route_sha256')


def stats(values):
    return dict(samples=len(values), mean=statistics.mean(values), p50=statistics.median(values),
                minimum=min(values), maximum=max(values))


def read_sample(path, frequency_hz=None):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    if saved.get('format') != timer.FORMAT:
        raise ValueError('Expected ll-reference-cycles-v1, not an old stage trace')
    meta = saved.get('metadata')
    if not isinstance(meta, dict) or meta.get('correctness') != 'passed':
        raise ValueError('Missing instrumented correctness evidence')
    for name in ('rank', 'world_size', 'iteration', 'seed'):
        if type(meta.get(name)) is not int or meta[name] < 0:
            raise ValueError('Invalid metadata ' + name)
    for name in ('run_id', 'case_id', 'config', 'sampling'):
        if name not in meta or meta[name] is None:
            raise ValueError('Missing metadata ' + name)
    if not isinstance(meta['config'], dict) or not isinstance(meta['sampling'], dict):
        raise ValueError('Config and sampling must be mappings')
    if type(meta.get('profile_seed')) is not int or meta['profile_seed'] != meta['seed']:
        raise ValueError('Missing or mismatched actual profile seed')
    if not isinstance(meta.get('route_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', meta['route_sha256']):
        raise ValueError('Missing actual routing fingerprint')
    comparison = meta.get('comparison')
    if (not isinstance(comparison, dict) or
            not re.fullmatch(r'[0-9a-f]{64}', str(comparison.get('installed_source_sha256', '')))):
        raise ValueError('Missing installed source fingerprint')
    stored_frequency = saved.get('frequency_hz')
    if frequency_hz is not None and stored_frequency is not None and frequency_hz != stored_frequency:
        raise ValueError('Frequency override disagrees with saved frequency')
    effective = frequency_hz if frequency_hz is not None else stored_frequency
    parsed = timer.parse_records(saved['raw'], saved.get('operation'), effective)
    if parsed['rank'] != meta['rank'] or parsed['world_size'] != meta['world_size']:
        raise ValueError('Raw/metadata rank or world mismatch')
    tokens = meta['config'].get('B', meta['config'].get('num_tokens', meta['config'].get('tokens')))
    if tokens is not None and tokens != parsed['num_tokens']:
        raise ValueError('Raw/metadata BS mismatch')
    return saved, parsed


def analyze(paths, frequency_hz=None, require_all_ranks=False):
    identity, ranks, samples, stage_groups = None, {}, [], {}
    for path in sorted(map(Path, paths)):
        saved, parsed = read_sample(path, frequency_hz)
        meta = saved['metadata']
        common = ([meta.get(k) for k in COMMON_KEYS], parsed['active_aiv'], parsed['frequency_hz'], parsed['num_tokens'])
        if identity is not None and common != identity:
            raise ValueError('Mixed configuration, source, case, route, sampling, or frequency')
        identity = common
        rank, iteration = meta['rank'], meta['iteration']
        if iteration in ranks.setdefault(rank, set()):
            raise ValueError('Duplicate rank/iteration')
        ranks[rank].add(iteration)
        unit = 'us' if parsed['frequency_hz'] is not None else 'ticks'
        kernels = [r for r in parsed['records'] if r['stage'] == 'kernel']
        slow = max(kernels, key=lambda r: (r['total_ticks'], -r['core']))
        samples.append(dict(rank=rank, iteration=iteration, core=slow['core'],
                            duration=slow['total_us'] if unit == 'us' else slow['total_ticks'],
                            duration_ticks=slow['total_ticks']))
        for row in parsed['records']:
            key = (rank, row['core'], row['role'], row['stage'])
            stage_groups.setdefault(key, []).append(row['total_us'] if unit == 'us' else row['total_ticks'])
    if identity is None:
        raise ValueError('No reference samples')
    common, active, frequency, tokens = identity
    result = dict(zip(COMMON_KEYS, common))
    world = result['world_size']
    present = set(ranks)
    hosts = {rank // 16 for rank in present}
    expected = set(range(world)) if require_all_ranks else {r for h in hosts for r in range(h*16, (h+1)*16)}
    if present != expected:
        raise ValueError('Missing ranks: expected complete physical 16-rank groups')
    iterations = ranks[min(present)]
    declared = result['sampling'].get('diagnostic_iters')
    if type(declared) is not int or declared < 1:
        raise ValueError('Missing positive sampling.diagnostic_iters')
    if iterations != set(range(declared)) or any(values != iterations for values in ranks.values()):
        raise ValueError('Missing or misaligned diagnostic iterations')
    result.update(format=timer.FORMAT, schema=timer.SCHEMA, operation='dispatch',
                  active_aiv=active, frequency_hz=frequency, num_tokens=tokens, unit=unit,
                  boundary_mode='reference', analyzed_ranks=sorted(present),
                  missing_remote_ranks=sorted(set(range(world))-present),
                  analysis_scope='all_ranks' if present == set(range(world)) else 'local_rank_subset',
                  iterations=sorted(iterations), sample_count=len(samples),
                  rank_kernel=[dict(rank=rank, **stats([s['duration'] for s in samples if s['rank'] == rank])) for rank in sorted(present)],
                  longest_core_samples=samples,
                  per_rank_core_stage=[dict(rank=rank, core=core, role=role, stage=stage, **stats(values))
                                       for (rank, core, role, stage), values in sorted(stage_groups.items())],
                  interpretation='Per-rank max single-core w21-w0 per capture, then statistics; '
                                 'not OFF profiler latency. Detail stages overlap and must not be added. '
                                 'Missing stages are absent, never imputed as zero. Independent core clocks.')
    return result


def compact_report(report, selected_iteration):
    unit = report['unit']
    lines = ['# LL dispatch 参考打点', '',
             f"run={report['run_id']}；case={report['case_id']}；route={report.get('route')}。",
             f"EP={report['world_size']}；AIV={report['active_aiv']}；频率={report['frequency_hz']} Hz；单位={unit}。",
             f"范围={report['analysis_scope']}；global ranks={report['analyzed_ranks']}；全部轮次={report['iterations']}；逐核表选轮={selected_iteration}（从0开始）。", '',
             '每rank每轮取最长单核w21-w0再统计。包含区间内打点指令，不含末尾诊断身份发布/cache写回，不能代替OFF profiler性能。细分包含于主阶段，不能相加；缺失不补零。', '',
             '| Rank | 轮数 | mean | p50 | min | max |', '|---|---|---|---|---|---|']
    for row in report['rank_kernel']:
        lines.append('| '+ ' | '.join(str(row[k]) if k in ('rank', 'samples') else f"{row[k]:.6g}"
                                     for k in ('rank','samples','mean','p50','minimum','maximum')) +' |')
    lines += ['', '## 慢rank逐核阶段', '',
              '取平均核跨度最慢的3个rank，选择该rank在所选轮的最长核；以下为该固定核各阶段在全部诊断轮的均值，不相加。', '',
              '| Rank | Core | 阶段 | 有效轮数 | 均值 |', '|---|---|---|---|---|']
    for rank in sorted(report['rank_kernel'], key=lambda row: (-row['mean'], row['rank']))[:3]:
        core = next(row['core'] for row in report['longest_core_samples']
                    if row['rank'] == rank['rank'] and row['iteration'] == selected_iteration)
        stages = {row['stage']: row for row in report['per_rank_core_stage']
                  if row['rank'] == rank['rank'] and row['core'] == core}
        for name in timer.STAGES:
            if name in stages:
                row = stages[name]
                lines.append(f"| {rank['rank']} | {core} | {timer.STAGE_INFO[name]['label']} | {row['samples']} | {row['mean']:.6g} |")
    lines += ['', f"源码指纹：{report['comparison']['installed_source_sha256']}；实际seed={report['profile_seed']}；路由SHA256={report['route_sha256']}。",
              '配置：`'+json.dumps(report['config'], ensure_ascii=False, sort_keys=True)+'`',
              '采样：`'+json.dumps(report['sampling'], ensure_ascii=False, sort_keys=True)+'`', '',
              '原始sample与完整core_stage_timings.xlsx、summary.json、stages.csv保留目标机。此文件省略逐核原始cycle及未展示核的阶段明细。']
    content = '\n'.join(lines)+'\n'
    if len(content.encode('utf-8')) > 60000:
        raise ValueError('Compact report exceeds 60000 UTF-8 bytes')
    return content


def run_report(reports):
    """One bounded, readable return file for a run; complete evidence stays local."""
    intro = '# LL dispatch 参考打点回传\n\n原始样本、完整逐核表及各case报告保留目标机。每rank统计只覆盖列明的global ranks；未合并远端不代表全域。\n'
    parts, omitted = [intro], 0
    for report in reports:
        section = compact_report(report, report['selected_iteration'])
        # Reserve space for the final omission notice; never split a UTF-8 character.
        if len(('\n'.join(parts) + section).encode('utf-8')) > 59000:
            omitted += 1
            continue
        parts.append(section)
    parts.append(f'共{len(reports)}个case/route；本文件省略{omitted}个case详细统计、全部逐核cycle。完整结果均在各case导出目录。\n')
    content = '\n'.join(parts)
    if len(content.encode('utf-8')) > 60000:
        raise ValueError('Run return report exceeds 60000 UTF-8 bytes')
    return content


def process_directory(source, output=None, frequency_hz=None, iteration='last', require_all_ranks=False):
    source = Path(source)
    paths = sorted(source.rglob('sample_*.json'))
    report = analyze(paths, frequency_hz, require_all_ranks)
    chosen = report['iterations'][-1] if iteration == 'last' else int(iteration)
    if chosen not in report['iterations']:
        raise ValueError('Selected iteration is absent')
    selected = [saved for path in paths for saved, _ in [read_sample(path, frequency_hz)]
                if saved['metadata']['iteration'] == chosen]
    exporter.validate_selected(report, selected, timer)
    text = compact_report(report, chosen)
    output = Path(output) if output is not None else source/'reference-analysis'
    output.mkdir(parents=True, exist_ok=False)
    report['selected_iteration'] = chosen
    with (output/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    fields = ('rank','iteration','core','role','stage','label','first_tick','last_tick','total_ticks','total_us','calls','task_count','batch_count')
    with (output/'stages.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            saved, parsed = read_sample(path, frequency_hz)
            core_info = {v['core']:v for v in parsed['cores']}
            for record in parsed['records']:
                core = core_info[record['core']]
                writer.writerow(dict(record, rank=parsed['rank'], iteration=saved['metadata']['iteration'],
                                     label=timer.STAGE_INFO[record['stage']]['label'],
                                     task_count=core['task_count'], batch_count=core['batch_count']))
    exporter.export_workbook(output/'core_stage_timings.xlsx', report, selected, timer)
    (output/'return_report.md').write_text(text, encoding='utf-8')
    print(f'{source} -> {output} (dispatch; all iterations summarized; selected {chosen})')
    return report


def discover(source):
    source = Path(source)
    roots = set()
    for path in source.rglob('sample_*.json'):
        ancestors = [p for p in path.parents if p.name == 'll_reference_profile' and (p == source or source in p.parents)]
        roots.add(ancestors[0] if ancestors else source)
    if not roots:
        raise ValueError('No sample_*.json under input')
    return sorted(roots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--input', '--source', dest='source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--iteration', default='last', help='Selected per-core view, zero-based N or last')
    parser.add_argument('--frequency-hz', type=float)
    parser.add_argument('--require-all-ranks', action='store_true')
    args = parser.parse_args()
    source = args.source or args.directory
    if source is None or (args.source and args.directory):
        parser.error('Specify one input directory')
    source = source if source.is_absolute() else ROOT/source
    output = args.output_dir if args.output_dir is not None else source/'reference-analysis'
    output = output if output.is_absolute() else ROOT/output
    if output.exists():
        raise FileExistsError('Report directory already exists; use a new --output-dir')
    directories = discover(source)
    reports = []
    for directory in directories:
        destination = output if len(directories) == 1 else output/directory.relative_to(source)
        reports.append(process_directory(directory, destination, args.frequency_hz, args.iteration, args.require_all_ranks))
    if len(reports) > 1:
        with (output/'return_report.md').open('x', encoding='utf-8') as stream:
            stream.write(run_report(reports))
    print(f'Return this one file: {output / "return_report.md"}')


if __name__ == '__main__':
    main()
