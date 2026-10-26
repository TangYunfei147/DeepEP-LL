"""Portable, streaming OOXML writer for validated reference-cycle reports.

Only Python's standard library is required on the target machine. Original
uint64 values with more than 15 decimal digits are written as text; subtraction
has already occurred in Python's integer domain before Excel sees a duration.
"""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
SHEET_NAMES = ('采样汇总', '逐核耗时', '阶段说明', '原始Cycle')
IDENTITY_HEADERS = ('实验组', 'Rank', '服务器', '本地卡', '核号', '收发角色', '采集轮次', '任务数')
EXACT_LIMIT = 999_999_999_999_999


def column(index):
    name = ''
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(65 + digit) + name
    return name


def _text(value):
    value = str(value)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
        raise ValueError('Unsupported XML control character in report text')
    if len(value) > 32767:
        raise ValueError('Excel cell exceeds 32767 characters')
    return escape(value)


def _cell(address, value, style=0):
    if value is None:
        return f'<c r="{address}" s="{style}"/>'
    if isinstance(value, dict):
        formula, cached = value['formula'], value['value']
        return (f'<c r="{address}" s="{style}"><f>{_text(formula)}</f>'
                f'<v>{_number(cached)}</v></c>')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int) and abs(value) > EXACT_LIMIT:
            value = str(value)
            style = 6  # Excel text format, in addition to inlineStr storage.
        else:
            return f'<c r="{address}" s="{style}"><v>{_number(value)}</v></c>'
    return (f'<c r="{address}" s="{style}" t="inlineStr"><is>'
            f'<t xml:space="preserve">{_text(value)}</t></is></c>')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Non-finite or nonnumeric spreadsheet result')
    return str(value)


def _sheet(archive, number, rows, count, columns, widths, header=1, freeze=0, numeric_from=None):
    with archive.open(f'xl/worksheets/sheet{number}.xml', 'w') as stream:
        def write(value):
            stream.write(value.encode('utf-8'))
        write(f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{NS}">')
        write(f'<dimension ref="A1:{column(columns)}{count}"/><sheetViews><sheetView showGridLines="0" workbookViewId="0">')
        if freeze:
            write(f'<pane xSplit="{freeze}" ySplit="{header}" topLeftCell="{column(freeze+1)}{header+1}" activePane="bottomRight" state="frozen"/>')
        write('</sheetView></sheetViews><sheetFormatPr defaultRowHeight="22"/><cols>')
        for c in range(1, columns + 1):
            width = widths[c-1] if c <= len(widths) else widths[-1]
            write(f'<col min="{c}" max="{c}" width="{width}" customWidth="1"/>')
        write('</cols><sheetData>')
        actual = 0
        for r, values in enumerate(rows, 1):
            actual = r
            height = 60 if r == header else (44 if number in (1, 3) else 22)
            wrapped = list(enumerate(values)) if number in (1, 3) else list(enumerate(values[:1]))
            if r != header:
                for c, value in wrapped:
                    if isinstance(value, str):
                        width = widths[c] if c < len(widths) else widths[-1]
                        chars = sum(2 if ord(ch) > 255 else 1 for ch in value)
                        height = max(height, 15 * math.ceil(chars / max(1, width - 2)))
            write(f'<row r="{r}" ht="{height}" customHeight="1">')
            for c, value in enumerate(values, 1):
                if r == header:
                    style = 1
                else:
                    stripe = number in (2, 4) and r % 2 == 0
                    numeric = numeric_from is not None and c >= numeric_from
                    style = (4 if stripe else 3) if numeric else (2 if stripe else 0)
                    if number in (1, 3):
                        style = 5  # wrapped explanatory cells, no truncation
                    elif c == 1:
                        style = 7 if stripe else 5
                write(_cell(f'{column(c)}{r}', value, style))
            write('</row>')
        if actual != count:
            raise ValueError('Spreadsheet row count mismatch')
        write('</sheetData>')
        if number in (2, 4):
            write(f'<autoFilter ref="A{header}:{column(columns)}{count}"/>')
        write('</worksheet>')


def _package(archive):
    types = ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, 5))
    archive.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+types+'</Types>')
    archive.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+REL+'/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    sheets = ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(SHEET_NAMES, 1))
    archive.writestr('xl/workbook.xml', f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="{REL}"><bookViews><workbookView/></bookViews><sheets>{sheets}</sheets><calcPr calcMode="auto" fullCalcOnLoad="1"/></workbook>')
    relations = ''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 5))
    archive.writestr('xl/_rels/workbook.xml.rels', f'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relations}<Relationship Id="rId5" Type="{REL}/styles" Target="styles.xml"/></Relationships>')
    archive.writestr('xl/styles.xml', f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS}">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>
<fonts count="2"><font><sz val="10"/><name val="Arial"/><color rgb="FF172B3A"/></font><font><b/><sz val="10"/><name val="Arial"/><color rgb="FFFFFFFF"/></font></fonts>
<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF173D62"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE6F1F8"/></patternFill></fill></fills>
<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="8"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="49" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>''')


def validate_selected(report, selected_saved, timer):
    """Do not allow a mismatched single-iteration view beside a valid summary."""
    selected, seen, iterations = [], set(), set()
    expected = set(report['analyzed_ranks'])
    for saved in selected_saved:
        meta = saved['metadata']
        parsed = timer.parse_records(saved['raw'], saved['operation'], report['frequency_hz'])
        rank = meta['rank']
        if rank in seen:
            raise ValueError('Duplicate selected rank')
        seen.add(rank); iterations.add(meta['iteration'])
        if saved.get('format') != timer.FORMAT or meta.get('correctness') != 'passed':
            raise ValueError('Selected data lacks reference format/correctness')
        for key in ('world_size', 'run_id', 'case_id', 'seed', 'config', 'sampling', 'route', 'comparison', 'source', 'profile_seed', 'route_sha256'):
            if meta.get(key) != report.get(key):
                raise ValueError('Selected metadata mismatch: ' + key)
        if (parsed['rank'] != rank or parsed['world_size'] != report['world_size'] or
                parsed['active_aiv'] != report['active_aiv'] or parsed['num_tokens'] != report['num_tokens']):
            raise ValueError('Selected raw identity mismatch')
        selected.append((saved, parsed))
    if seen != expected or len(iterations) != 1 or not iterations.issubset(set(report['iterations'])):
        raise ValueError('Missing ranks or mixed selected iterations')
    return sorted(selected, key=lambda value: value[1]['rank'])


def export_workbook(output_path, report, selected_saved, timer):
    selected = validate_selected(report, selected_saved, timer)
    frequency = report['frequency_hz']
    unit = 'µs' if frequency is not None else 'ticks'
    divisor = frequency / 1e6 if frequency is not None else 1
    group = f"{report['run_id']} / {report['case_id']} / {report.get('route') or ''}"
    iteration = selected[0][0]['metadata']['iteration']
    row_count = sum(parsed['active_aiv'] for _, parsed in selected) + 1
    stages = timer.STAGES
    summary = [
        ['LL dispatch 逐核阶段耗时'],
        ['运行 / case / 路由', group],
        ['统计范围', report['analysis_scope'], '实际global ranks', ', '.join(map(str, report['analyzed_ranks']))],
        ['计时口径', '每rank每轮最长单核w21-w0；包含打点指令，不含末尾诊断身份发布/cache写回。非OFF profiler性能。'],
        ['计数器频率 (Hz)', frequency, '单位', unit],
        ['逐核表选轮（从0开始）', iteration, '汇总全部轮次', ', '.join(map(str, report['iterations']))],
        [], ['Rank', '有效轮数', f'均值 ({unit})', f'p50 ({unit})', f'最小 ({unit})', f'最大 ({unit})'],
    ]
    for value in report['rank_kernel']:
        summary.append([value[k] for k in ('rank', 'samples', 'mean', 'p50', 'minimum', 'maximum')])
    summary += [[], ['配置项', '值']]
    for key, value in sorted(report['config'].items()):
        summary.append([key, json.dumps(value, ensure_ascii=False)])
    summary += [['采样', json.dumps(report['sampling'], ensure_ascii=False)],
                ['实际核数 / EP', report['active_aiv'], report['world_size']],
                ['拓扑字段', '无原始hostname时显示连续16rank物理组；本地rank按rank%16推导，不编造IP。'],
                ['阶段语义', '主阶段包含细分，不能相加；空白=未执行或无点，0=确有记录且耗时为0。']]

    def per_core():
        for saved, parsed in selected:
            meta = saved['metadata']
            by_core = {c['core']: {} for c in parsed['cores']}
            for record in parsed['records']:
                by_core[record['core']][record['stage']] = record
            for core in parsed['cores']:
                index = core['core']; rank = parsed['rank']
                identity = [group, rank, meta.get('hostname', f'物理组 {rank//16}'),
                            meta.get('local_rank', rank % 16), index,
                            '发送' if core['role'] == 'send' else '接收', iteration, core['task_count']]
                words = [v for row in saved['raw'][index] for v in row]
                yield identity, words, by_core[index], core

    def detail_rows():
        yield [*IDENTITY_HEADERS, *[f"{timer.STAGE_INFO[name]['label']} ({unit})" for name in stages], 'packet就绪批次数']
        for row, (identity, words, records, core) in enumerate(per_core(), 2):
            values = []
            for index, name in enumerate(stages):
                rec = records.get(name)
                if rec is None:
                    values.append(None); continue
                info = timer.STAGE_INFO[name]
                start, end = info['start'], info['end']
                if start is None:
                    formula = f"'原始Cycle'!{column(end+4)}{row}"
                elif max(words[start], words[end]) <= EXACT_LIMIT:
                    formula = f"('原始Cycle'!{column(end+4)}{row}-'原始Cycle'!{column(start+4)}{row})"
                else:
                    formula = f"'原始Cycle'!{column(68+index)}{row}"
                value = rec['total_ticks'] / divisor if frequency is not None else rec['total_ticks']
                if frequency is not None:
                    formula += "/'阶段说明'!$B$2"
                if frequency is None and rec['total_ticks'] > EXACT_LIMIT:
                    values.append(str(rec['total_ticks']))
                else:
                    values.append(dict(formula=formula, value=value))
            yield [*identity, *values, core['batch_count']]

    definitions = [
        ['阶段换算', '同核整数cycle差 / 每µs的cycle数；w33/w34直接累计'],
        ['每µs的cycle数', divisor if frequency is not None else None, '无频率时全表保留ticks' if frequency is None else 'GetSystemCycle频率'],
        ['包含关系', '提交仅描述提交；已有异步工作在后续drain中完成。细分项不与主项相加。'],
        [], ['阶段', 'word换算（ticks）', '源码边界 / NDR说明', '包含关系'],
    ]
    for name in stages:
        info = timer.STAGE_INFO[name]
        expression = f"w{info['end']}" if info['start'] is None else f"w{info['end']}-w{info['start']}"
        definitions.append([info['label'], expression, info['boundary'], info['relationship']])
    definitions += [
        ['有效性', 'w58 valid_mask', '未执行点mask未置位且word为0；执行时刻或累计为0仍保留有效位。'],
        ['任务 / 次数', '发送任务w43；接收任务w37；packet batch w41', 'w33/w34的calls对应w41批次数，不是packet总数。'],
        ['原始精度', '超过15位整数保存为文本', 'BP列起保留Python整数域作差得到的29项ticks；大时钟公式引用这些差值，避免Excel大数消减。'],
        ['核时钟', '仅同核相减', '不做跨rank绝对时钟对齐；不从嵌套阶段之和推算总时长。'],
        ['NDR边界', '参考阶段命名保留', '同名SendCnt包含当前NDR已有路由交换；payload提交可能包含RDMA post/quiet；不移植参考业务逻辑。'],
        ['接收批次', '标量累计两段', '只对完整ready批次和加载消费区间累计，不对每次空扫描插点。'],
    ]

    def raw_rows():
        yield ['实验组', 'Rank', '核号', *[f'w{i} {name}' for i, name in enumerate(timer.WORDS)],
               *[f"{timer.STAGE_INFO[name]['label']} 精确ticks（整数作差）" for name in stages]]
        for identity, words, records, _ in per_core():
            yield [identity[0], identity[1], identity[4], *words,
                   *[records[name]['total_ticks'] if name in records else None for name in stages]]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # x mode is essential: analysis must never replace an existing measurement.
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        _package(archive)
        _sheet(archive, 1, summary, len(summary), 6, [35, 68, 24, 48, 24, 24], header=8)
        _sheet(archive, 2, detail_rows(), row_count, 38, [28, 10, 22, 10, 10, 12, 12, 12, 22], freeze=6, numeric_from=9)
        _sheet(archive, 3, definitions, len(definitions), 4, [36, 48, 95, 38], header=5)
        _sheet(archive, 4, raw_rows(), row_count, 67+len(stages), [28, 10, 10, 27], freeze=3)
    return path
