"""Analyze a complete local run into one multi-sheet XLSX; no Torch/CANN required."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def analyze_run(source, output=None, iteration='last', frequency_hz=None, require_all_ranks=False):
    source = Path(source)
    source = source if source.is_absolute() else ROOT/source
    output = Path(output) if output is not None else source/'table-analysis'
    output = output if output.is_absolute() else ROOT/output
    if output.exists():
        raise FileExistsError('Output exists; use a fresh --output-dir')
    groups = {}
    for sample in sorted(source.rglob('sample_*.json')):
        ancestors = [p for p in sample.parents if (p == source or source in p.parents) and
                     p.name in ('ll_reference_profile', 'll_stage_timer')]
        if ancestors:
            parent = ancestors[0]
        else:
            parent = source
        value = json.loads(sample.read_text(encoding='utf-8'))
        kind = 'reference' if value.get('format') == 'll-reference-cycles-v1' else 'timer'
        previous = groups.setdefault(parent, kind)
        if previous != kind:
            raise ValueError('Mixed timer/reference records in one case directory')
    if not groups:
        raise ValueError('No diagnostic samples; --mode perf alone has no per-core stage data')
    tables = load('run_report_tables', 'tests/ll_rdma/report_tables.py')
    entries = []
    for index, (directory, kind) in enumerate(sorted(groups.items()), 1):
        destination = output/f'case_{index:03d}'
        if kind == 'reference':
            parser = load('run_reference_analyzer', 'tests/ll_reference_profile/analyze.py')
            report = parser.process_directory(directory, destination, frequency_hz, iteration, require_all_ranks)
        else:
            parser = load('run_stage_analyzer', 'tests/ll_stage_timer/analyze.py')
            report = parser.process_directory(directory, destination, frequency_hz, iteration,
                                               require_all_ranks=require_all_ranks)
        entries.append((destination/'core_stage_timings.xlsx', report, kind))
    tables.merge_workbooks(output/'stage-times.xlsx', entries)
    manifest = [dict(directory=str(path.parent.relative_to(output)), kind=kind, run_id=report.get('run_id'),
                     case_id=report.get('case_id'), ranks=report['analyzed_ranks'],
                     selected_iteration=report['selected_iteration']) for path, report, kind in entries]
    (output/'TABLE_INDEX.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# LL逐核表格导出', '', '主文件：stage-times.xlsx；每个case/算子分sheet，不混合均值。',
             '逐核耗时/stages.csv为选中单轮，逐核均值/stages_mean.csv为全部有效轮次均值；缺失不补零。',
             '完整原始数据及Excel保留目标机；下列统计仅来自诊断capture，不代表OFF性能。', '']
    omitted = 0
    for item in manifest:
        line = json.dumps(item, ensure_ascii=False)
        if len(('\n'.join(lines)+line).encode('utf-8')) > 58000:
            omitted += 1
        else:
            lines.append(line)
    lines.append(f'共{len(manifest)}组；索引省略{omitted}组，完整索引见TABLE_INDEX.json。')
    for path, _, kind in entries:
        report_path = path.parent/('return_report.md' if kind == 'reference' else 'bottleneck_summary.md')
        lines.append(f'\n## {path.parent.name}\n')
        for line in report_path.read_text(encoding='utf-8').splitlines():
            if len(('\n'.join(lines)+line).encode('utf-8')) > 59000:
                lines.append('报告达到字节上限，剩余统计见对应case目录及Excel。')
                break
            lines.append(line)
        if len('\n'.join(lines).encode('utf-8')) > 59000:
            break
    (output/'return_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Combined workbook: {output / "stage-times.xlsx"}')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--iteration', default='last')
    p.add_argument('--frequency-hz', type=float)
    p.add_argument('--require-all-ranks', action='store_true')
    a = p.parse_args()
    analyze_run(a.directory, a.output_dir, a.iteration, a.frequency_hz, a.require_all_ranks)


if __name__ == '__main__':
    main()
