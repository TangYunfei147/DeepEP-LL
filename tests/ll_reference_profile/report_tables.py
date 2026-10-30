"""Offline wide tables and run-level XLSX assembly; Python standard library only."""
import csv
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ll_table_writer', ROOT/'tests/ll_reference_profile/export_xlsx.py')
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def csv_table(path, headers, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        output = csv.writer(stream)
        output.writerow(headers)
        output.writerows(rows)


def wide_records(selected, timer, frequency):
    """One row for each operation/rank/core, from exactly one capture iteration."""
    stages = tuple(s for s in timer.STAGES if s != 'header')
    headers = ['operation', 'rank', 'host_group', 'local_rank', 'core', 'role', 'iteration', 'unit', *stages]
    rows, calls, raw_rows = [], [], []
    for saved in sorted(selected, key=lambda v: (v['operation'], v['metadata']['rank'])):
        parsed = timer.parse_records(saved['raw'], saved['operation'], frequency)
        rank, iteration = saved['metadata']['rank'], saved['metadata']['iteration']
        by_core = {core: {} for core in range(parsed['active_aiv'])}
        for record in parsed['records']:
            by_core[record['core']][record['stage']] = record
        for core, records in by_core.items():
            identity = [saved['operation'], rank, rank//16, rank%16, core,
                        parsed['cores'][core]['role'], iteration,
                        'us' if frequency is not None else 'ticks']
            rows.append(identity + [records[s].get('total_us', records[s]['total_ticks']) if s in records else None for s in stages])
            calls.append(identity[:7] + ['calls'] + [records[s]['calls'] if s in records else None for s in stages])
            words = [v & ((1<<64)-1) for block in saved['raw'][core] for v in block]
            raw_rows.append(identity[:7] + words)
    keys = [tuple(r[i] for i in (0, 1, 4)) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate operation/rank/core in selected wide table')
    return stages, headers, rows, calls, raw_rows


def mean_rows(report, headers, stages):
    groups = {}
    for r in report['per_rank_core_stage']:
        key = (r.get('operation', 'dispatch'), r['rank'], r['core'], r['role'])
        groups.setdefault(key, {})[r['stage']] = r
    result = []
    for (operation, rank, core, role), records in sorted(groups.items()):
        result.append([operation, rank, rank//16, rank%16, core, role, 'all_mean',
                       'us' if report['frequency_hz'] is not None else 'ticks'] +
                      [records[s]['mean'] if s in records else None for s in stages])
    return result


def number(value):
    if value == '':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def display_table(rows, widths, preserve_iteration=False):
    """Compact Excel view; CSV keeps full identity for offline joins and audit."""
    drop = {'实验组', '服务器', '服务器组', '本地卡', '采集轮次',
            'host_group', 'local_rank', 'iteration'}
    pinned = {'算子', 'operation', 'Rank', 'rank', '核号', 'core',
              '收发分工', '收发角色', 'role', '任务数', 'task_count',
              'executing_rank', 'expert_rank', 'expert_id'}
    if preserve_iteration:
        drop -= {'iteration','采集轮次'}
        pinned |= {'iteration','采集轮次'}
    head = rows[0]
    keep = [i for i, h in enumerate(head) if h not in drop]
    frozen = [i for i in keep if head[i] in pinned or head[i] == 'kernel' or head[i].startswith('kernel (')]
    order = frozen + [i for i in keep if i not in frozen]
    return ([[r[i] if i < len(r) else None for i in order] for r in rows],
            [widths[min(i, len(widths)-1)] for i in order], len(frozen))


def merge_workbooks(output, entries):
    """Bundle independent cases as separate tabs; retarget formulas and keep cached values."""
    ns = {'m': writer.NS}
    names, parts, extra, index = [], [], [], [['组', 'run / case', '模式', '配置', 'global ranks', '逐核轮次']]
    styles = None
    for group, (path, report, kind) in enumerate(entries, 1):
        if kind != 'reference':
            raise ValueError('Only current reference workbooks can be merged')
        prefix = f'C{group:03d}_' if len(entries) > 1 else ''
        with ZipFile(path) as source:
            current_style = source.read('xl/styles.xml')
            if styles is not None and styles != current_style:
                raise ValueError('Workbook styles differ; cannot merge without preserving them')
            styles = current_style
            sheets = ET.fromstring(source.read('xl/workbook.xml')).findall('m:sheets/m:sheet', ns)
            mapping = {s.attrib['name']: prefix+s.attrib['name'] for s in sheets}
            for i, s in enumerate(sheets, 1):
                name = mapping[s.attrib['name']]
                data = source.read(f'xl/worksheets/sheet{i}.xml')
                for old, new in mapping.items():
                    data = data.replace(("'"+old+"'!").encode('utf-8'), ("'"+new+"'!").encode('utf-8'))
                names.append(name); parts.append(data)
        if kind == 'reference':
            for filename, title in [('stages_mean.csv', '逐核均值'), ('stage_calls.csv', '阶段调用次数'),
                                    ('stages_long.csv', '全部轮次明细')]:
                table_path = Path(path).parent/filename
                if table_path.is_file():
                    with table_path.open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.reader(stream))
                    for r in range(1, len(rows)):
                        rows[r] = [number(v) for v in rows[r]]
                    extra.append((prefix+title, rows))
        index.append([prefix or '本次', f"{report.get('run_id')} / {report.get('case_id')}", kind,
                      json.dumps(report.get('config', report.get('configurations')), ensure_ascii=False),
                      str(report.get('analyzed_ranks')), report.get('selected_iteration')])
    if len({report.get('run_id') for _, report, _ in entries}) > 1:
        raise ValueError('Do not combine different run IDs into one run workbook')
    names += [name for name, _ in extra]
    if len(entries) > 1:
        names.insert(0, '报告索引')
    with ZipFile(output, 'x', compression=ZIP_DEFLATED, compresslevel=6) as target:
        writer._package(target, names)
        first = 1
        if len(entries) > 1:
            writer._sheet(target, 1, index, len(index), 6, [12, 40, 14, 80, 40, 12], explanatory=True)
            first = 2
        for i, data in enumerate(parts, first):
            target.writestr(f'xl/worksheets/sheet{i}.xml', data)
        for i, (name, rows) in enumerate(extra, first+len(parts)):
            rows, widths, freeze = display_table(rows, [18, 12, 12, 12, 12, 12, 14, 12, 24],
                                                 preserve_iteration=name.endswith('全部轮次明细'))
            writer._sheet(target, i, rows, len(rows), len(rows[0]), widths,
                          freeze=freeze, explanatory=False)
