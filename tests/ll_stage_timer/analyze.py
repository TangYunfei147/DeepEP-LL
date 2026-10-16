"""Aggregate validated LL stage traces per rank/core/stage; retain raw ticks."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ll_stage_timer_parser',ROOT/'deep_ep/utils/ll_stage_timer.py')
timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)


def stats(values):
    values=sorted(values)
    def q(p):
        x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(samples=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                p50=q(.5),p90=q(.9),p99=q(.99))



def bandwidth_sample(report, metadata):
    """Per-rank effective byte rate; max core interval is a duration proxy, not NIC time."""
    spans = [row['last_tick']-row['first_tick'] for row in report['records'] if row['stage']=='kernel']
    duration_ticks = max(spans)
    frequency = report['frequency_hz']
    duration_us = duration_ticks*1e6/frequency if frequency is not None else None
    available = report['traffic_available']
    totals = {name: dict(bytes=0, operations=0, read_bytes=0, read_operations=0)
              for name in timer.TRAFFIC_CLASSES}
    flags = 0
    if available:
        for row in report['traffic']:
            flags |= row['flags']
            for field in totals[row['category']]:
                totals[row['category']][field] += row[field]
        if metadata['world_size']==16 and any(totals[name][field]
                for name in ('interhost_first','interhost_forward') for field in totals[name]):
            raise ValueError('EP16 has nonzero interhost traffic')
    totals['intrahost_total'] = {field: totals['intrahost_first'][field]+totals['intrahost_forward'][field]
                               for field in totals['local']}
    rows = []
    for category, values in totals.items():
        if not available: status = 'counters_unavailable'
        elif flags & 1: status = 'nonstandard_skip_communication'
        elif metadata['world_size'] != 16: status = 'unsupported_ep'
        elif category == 'local': status = 'local_not_link_bandwidth'
        elif category.startswith('interhost'): status = 'not_applicable_ep16'
        elif frequency is None: status = 'frequency_unavailable'
        elif duration_ticks <= 0: status = 'zero_duration'
        else: status = 'ok'
        rows.append(dict(operation=report['operation'],rank=metadata['rank'],iteration=metadata['iteration'],
            category=category, **(values if available else {key:None for key in values}),
            duration_kind='max_core_kernel',duration_ticks=duration_ticks,duration_us=duration_us,
            frequency_hz=frequency,flags=flags if available else None,status=status,
            effective_GBps=values['bytes']/(duration_us*1000) if status=='ok' else None))
    return rows


def bandwidth_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['operation'],row['rank'],row['category']),[]).append(row)
    result = []
    for (operation,rank,category), samples in sorted(groups.items()):
        result.append(dict(operation=operation,rank=rank,category=category,samples=len(samples),
            statuses=sorted({sample['status'] for sample in samples}),
            **{field:stats([sample[field] for sample in samples if sample[field] is not None])
               if any(sample[field] is not None for sample in samples) else None
               for field in ('bytes','operations','read_bytes','read_operations','duration_ticks','duration_us','effective_GBps')}))
    return result


def diagnostic_sample(report, metadata):
    """Compare elapsed intervals, never absolute clocks across ranks or cores."""
    kernels = [row for row in report['records'] if row['stage'] == 'kernel']
    slow = max(kernels, key=lambda row: (row['last_tick'] - row['first_tick'], -row['core']))
    elapsed = slow['last_tick'] - slow['first_tick']
    factor = 1e6 / report['frequency_hz'] if report['frequency_hz'] is not None else 1
    stages = []
    for row in report['records']:
        if row['core'] != slow['core'] or row['stage'] == 'kernel':
            continue
        contained = slow['first_tick'] <= row['first_tick'] <= row['last_tick'] <= slow['last_tick']
        stages.append(dict(stage=row['stage'], duration=row['total_ticks'] * factor, calls=row['calls'],
                           fraction=row['total_ticks'] / elapsed if contained and elapsed > 0 else None))
    return dict(operation=report['operation'], rank=metadata['rank'], iteration=metadata['iteration'],
                core=slow['core'], role=slow['role'], duration=elapsed * factor, stages=stages)


def diagnostic_summary(samples):
    ranks, stages, rounds, cores = {}, {}, {}, {}
    for sample in samples:
        key = (sample['operation'], sample['rank'])
        ranks.setdefault(key, []).append(sample['duration'])
        rounds.setdefault((sample['operation'], sample['iteration']), []).append(sample)
        cores.setdefault((*key, sample['core']), 0)
        cores[(*key, sample['core'])] += 1
        for stage in sample['stages']:
            stages.setdefault((*key, stage['stage']), []).append(stage)
    return dict(
        rank_kernel=[dict(operation=op, rank=rank, **stats(values)) for (op, rank), values in sorted(ranks.items())],
        longest_core_samples=[{key:value for key,value in sample.items() if key != 'stages'} for sample in samples],
        longest_core_counts=[dict(operation=op, rank=rank, core=core, samples=count)
                             for (op,rank,core), count in sorted(cores.items())],
        longest_core_stages=[dict(operation=op, rank=rank, stage=stage, observed_samples=len(values),
                                 duration=stats([value['duration'] for value in values]),
                                 fraction=stats([value['fraction'] for value in values if value['fraction'] is not None])
                                 if any(value['fraction'] is not None for value in values) else None)
                             for (op,rank,stage), values in sorted(stages.items())],
        round_rank_spread=[dict(operation=op, iteration=iteration,
                               slow_rank=max(values, key=lambda row: (row['duration'], -row['rank']))['rank'],
                               max_over_median=max(row['duration'] for row in values) /
                               statistics.median(row['duration'] for row in values)
                               if statistics.median(row['duration'] for row in values) > 0 else None)
                           for (op,iteration), values in sorted(rounds.items())])


def render_bottleneck_report(report):
    """Evidence-led Markdown; stage shares are conditional, nested and non-additive."""
    def number(value):
        return '不可用' if value is None else f'{value:.4g}'
    def safe(value):
        return str(value).replace('|', '\\|').replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;')
    def table(headers, rows):
        return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + [
            '| ' + ' | '.join(safe(value) for value in row) + ' |' for row in rows]
    unit = 'us' if report['frequency_hz'] is not None else 'ticks'
    diag = report['diagnostics']
    lines = ['# LL 打点瓶颈分析总结', '',
             f"EP={report['world_size']}；schema={report['schema']}；模式={report['boundary_mode']}；频率={report['frequency_hz']} Hz。", '',
             f"run={safe(report.get('run_id'))}；case={safe(report.get('case_id'))}；seed={safe(report.get('seed'))}；原始样本文件数={report.get('sample_count')}。", '',
             '本报告分析本次诊断尾段，不能替代未插桩 profiler 基线。每个rank的耗时代理为最长单核kernel跨度；不是全局端到端时延，也不代表该核必然是因果关键路径。', '',
             '## 数据与口径', '',
             '- 输入已通过正确性标记、rank/轮次完整性及配置一致性检查；不同shape/case分别分析。',
             '- 阶段占比先在同rank/同轮/同核计算，再统计；父子阶段可能嵌套，不能相加。未执行阶段不补零。',
             '- 多个核同样长时选择编号最小的核展示；其他核仍见stages.csv。慢核位置可以随轮次变化。',
             '- 无物理拓扑、逐peer字节和逐包到达时间，不能证明多对一拥塞，也不计算峰值带宽达成率。', '']
    if report['boundary_mode'] == 'drain':
        lines += ['**drain主动串行化阶段边界：等待位置和重叠已被改变，不能据此判断observe生产路径瓶颈。**', '']
    if any(row.get('flags', 0) for row in report['bandwidth']):
        lines += ['**存在跳过通信的非标准诊断标记：不生成带宽，也不能据此推断正常通信瓶颈。**', '']
    if report['frequency_hz'] is None:
        lines += ['频率未提供：时间保留ticks，带宽及微秒数不可用。', '']
    identity = report.get('comparison') or {}
    if identity:
        lines += table(['运行身份', '值'], [(key, identity[key]) for key in
                       ('implementation','transport','logical_group','active_aiv','source_revision','source_dirty') if key in identity])
        lines += ['']
    for operation, config in sorted(report['configurations'].items(), key=lambda item: item[0] != 'dispatch'):
        rank_rows = sorted((row for row in diag['rank_kernel'] if row['operation']==operation),
                           key=lambda row: (-row['mean'], row['rank']))
        slow = rank_rows[0]
        lines += [f'## {operation}', '', '### 本次配置', '']
        lines += table(['参数', '值'], [(key, json.dumps(value, ensure_ascii=False)) for key,value in sorted(config.items())])
        lines += ['', f"最慢平均耗时为rank {slow['rank']}：{number(slow['mean'])} {unit}，共{slow['samples']}轮。以下排名仅比较独立时长，不相减跨rank时间戳。", '']
        lines += table(['rank', '轮数', f'均值 ({unit})', f'p50 ({unit})', f'最大值 ({unit})'],
                       [(row['rank'], row['samples'], number(row['mean']), number(row['p50']), number(row['maximum'])) for row in rank_rows])
        spread = [row['max_over_median'] for row in diag['round_rank_spread'] if row['operation']==operation and row['max_over_median'] is not None]
        if spread:
            lines += ['', f"逐轮rank最大/中位耗时比：均值{number(statistics.mean(spread))}，最大{number(max(spread))}。"]
            if statistics.mean(spread) >= 1.25:
                lines += ['线索：rank耗时存在偏斜（启发式阈值1.25）；需区分工作量差异、上游等待和资源竞争，不能直接归因为网络拥塞。']
        if slow['samples'] < 10:
            lines += ['采样少于10轮：分位数和波动估计仅供定位，不能据此断言稳定性能差异。']
        lines += ['', '### 慢rank的最长核及阶段线索', '',
                  '仅展示平均耗时最慢的3个rank。占比是阶段累计时间/该轮该核kernel跨度；均值仅覆盖该阶段有记录的轮次。以下各行可能互相包含，不能求和。', '']
        for rank in [row['rank'] for row in rank_rows[:3]]:
            counts = [row for row in diag['longest_core_counts'] if row['operation']==operation and row['rank']==rank]
            lines += [f"rank {rank} 最长核出现次数：" + '；'.join(f"AIV {row['core']}：{row['samples']}轮" for row in counts) + '。', '']
            candidates = [row for row in diag['longest_core_stages'] if row['operation']==operation and row['rank']==rank]
            # Omit outer wrappers; show useful internal phases without summing nested children.
            candidates = [row for row in candidates if row['stage'] not in ('sender','receiver','send_payload')]
            candidates.sort(key=lambda row: -row['duration']['mean'])
            lines += table(['阶段', '有记录轮数', f'累计时间均值 ({unit})', '占比有效轮数', '同核占比均值'],
                           [(row['stage'], row['observed_samples'], number(row['duration']['mean']),
                             row['fraction']['samples'] if row['fraction'] else 0,
                             number(row['fraction']['mean']*100)+'%' if row['fraction'] else '不可用') for row in candidates[:8]])
            hints = {'recv_wait': '接收等待较多；核对发送准备/首跳转发是否推迟数据到达，不能单凭等待证明链路拥塞。',
                     'send_pack': '封包路径较重；结合输入搬入、量化和lane等待子阶段区分数据处理与流水复用成本。',
                     'send_issue': '发送调用较重；observe下MTE下发不是完成时间，RDMA路径还包含提交和quiet等待。',
                     'recv_consume': '接收处理较重；结合metadata、输出写回与forward子阶段检查，避免将整个区间当成通信。',
                     'recv_forward': '转发路径占比较高；对照转发字节与轮次变化，尚不能推断目标端拥塞。',
                     'combine_wait_ready': 'combine准备等待较多；检查组内贡献准备与同步进度。',
                     'combine_partial': 'partial阶段混合了贡献读取、计算和回传，现有粒度不能独立判定哪一项受限。',
                     'combine_direct': 'DIRECT阶段混合了输入、打包、回传及排空，需避免等同纯传输耗时。',
                     'combine_aggregate': '最终聚合较重；既可能等待返回结果，也包含读取和归约计算。'}
            for row in candidates:
                flagged = any(item.get('flags', 0) for item in report['bandwidth'] if item['operation']==operation)
                if not flagged and row['stage'] in hints and row['fraction'] and row['fraction']['mean'] >= .30:
                    lines += ['', '待验证线索（同核占比≥30%的启发式）：' + hints[row['stage']]]
            lines += ['']
        lines += ['### 数据搬运及有效带宽', '',
                  '字节来自本次记录，不按固定hidden或包大小推算。跨卡带宽分子为机内首发+转发；read是总量子集，本卡不计入。是发起端读+写吞吐，不是纯TX或线速。', '']
        traffic_rows = []
        for rank in sorted(row['rank'] for row in rank_rows):
            entries = {row['category']:row for row in report['per_rank_bandwidth'] if row['operation']==operation and row['rank']==rank}
            metric = lambda category, field: number(entries[category][field]['mean']) if entries[category][field] else '不可用'
            per_round = [row for row in report['bandwidth'] if row['operation']==operation and row['rank']==rank]
            forwards = {row['iteration']:row['bytes'] for row in per_round if row['category']=='intrahost_forward'}
            ratios = [forwards[row['iteration']]/row['bytes'] for row in per_round
                      if row['category']=='intrahost_total' and row['bytes'] is not None and row['bytes']>0]
            traffic_rows.append((rank, metric('local','bytes'), metric('intrahost_first','bytes'),
                                 metric('intrahost_forward','bytes'), metric('intrahost_total','read_bytes'),
                                 number(statistics.mean(ratios)*100)+'%' if ratios else '不可用',
                                 metric('intrahost_total','effective_GBps'), ', '.join(entries['intrahost_total']['statuses'])))
        lines += table(['rank','本卡B/轮','机内跨卡首发B/轮','机内跨卡转发B/轮','其中读取B/轮','机内转发占比均值','有效GB/s均值','状态'], traffic_rows)
        if report['world_size'] != 16:
            lines += ['', '当前带宽计算仅支持EP16。下面保留跨机计数，不能将机内字节当成全部通信量。', '']
            remote = [row for row in report['per_rank_bandwidth']
                      if row['operation']==operation and row['category'].startswith('interhost')]
            lines += table(['rank','类别','平均B/轮'],
                           [(row['rank'],row['category'],number(row['bytes']['mean']) if row['bytes'] else '不可用') for row in remote])
        lines += ['', '低有效带宽可能来自计算、等待、包处理或通信，不能据此直接确定物理链路瓶颈。各rank速率不相加为整机真实带宽。', '']
    if report.get('per_expert'):
        packets = sorted((row for row in report['per_expert'] if row['stage']=='packets'), key=lambda row: -row['mean_calls'])[:10]
        lines += ['## 专家投递热点线索', '', '仅按已有专家记录的平均投递次数排序，不将共享转发/等待成本分摊给专家。', '']
        lines += table(['执行rank','expert','平均投递次数/有记录轮'], [(row['executing_rank'],row['expert_id'],number(row['mean_calls'])) for row in packets])
    lines += ['', '## 建议的下一步', '',
              '1. 先按上述慢rank和同核阶段定位，再查看stages.csv及trace.json中的子阶段；重复区间是包络，不是逐包时间线。',
              '2. 对相同shape/seed做单变量A/B。shape改变时分别报告，不直接将耗时或带宽变化当成优化收益。',
              '3. 若怀疑多对一或共享链路竞争，补充源→目标通信矩阵、实际拓扑及对应链路计数器；当前汇总无法确认。',
              '4. 优化收益以timer OFF的正式task性能及正确性为准；本报告未混入profiler、Event或API时间。', '',
              '机器可读依据：summary.json的diagnostics、bandwidth和per_rank_bandwidth；逐核阶段：stages.csv。', '']
    return '\n'.join(lines)


def analyze(paths, frequency_hz=None):
    configurations={};ranks={};groups={};identity=None;stage_max={};bandwidth=[];diagnostics=[]
    for path in paths:
        saved=json.loads(path.read_text())
        meta=saved['metadata']
        if meta.get('correctness')!='passed':raise ValueError('Missing instrumented correctness evidence')
        report=timer.parse_records(saved['raw'],saved['operation'],frequency_hz if frequency_hz is not None else saved.get('frequency_hz'))
        common=(meta['world_size'],meta['config'],report['active_aiv'],report['boundary_mode'],report['frequency_hz'],
                meta.get('run_id'),meta.get('seed'),meta.get('case_id'),meta.get('sampling'),report['schema'],meta.get('comparison'))
        if identity is not None and common!=identity:raise ValueError('Mixed configurations or timer modes')
        identity=common
        op=report['operation'];rank=meta['rank'];iteration=meta['iteration']
        if not 0<=rank<meta['world_size'] or type(iteration) is not int or iteration<0:raise ValueError('Invalid rank/iteration')
        key=(op,rank)
        if iteration in ranks.setdefault(key,set()):raise ValueError('Duplicate rank/iteration')
        ranks[key].add(iteration)
        configurations[op]=meta['config']
        bandwidth.extend(bandwidth_sample(report,meta))
        diagnostics.append(diagnostic_sample(report,meta))
        unit='us' if report['frequency_hz'] is not None else 'ticks'
        for row in report['records']:
            key=(op,rank,row['core'],row['role'],row['stage'])
            duration=row['total_us'] if unit=='us' else row['total_ticks']
            groups.setdefault(key,[]).append((duration,row['calls']))
            stage_key=(op,rank,row['stage'])
            per_iteration=stage_max.setdefault(stage_key,{})
            per_iteration[iteration]=max(per_iteration.get(iteration,0),duration)
    if identity is None:raise ValueError('No traces')
    for op in configurations:
        if {rank for kind,rank in ranks if kind==op}!=set(range(identity[0])):
            raise ValueError(f'Missing ranks for {op}')
        reference=ranks[(op,0)]
        if reference!=set(range(len(reference))) or any(ranks[(op,r)]!=reference for r in range(identity[0])):
            raise ValueError('Missing or misaligned diagnostic iterations')
        if identity[8] is not None:
            expected = identity[8].get('diagnostic_iters')
            if type(expected) is not int or expected < 1 or reference != set(range(expected)):
                raise ValueError('Incomplete diagnostic tail')
    rows=[]
    for (op,rank,core,role,stage), values in sorted(groups.items()):
        rows.append(dict(operation=op,rank=rank,core=core,role=role,stage=stage,unit=unit,
                         **stats([v for v,n in values]),mean_calls=statistics.mean(n for v,n in values)))
    return dict(schema=identity[9],world_size=identity[0],boundary_mode=identity[3],frequency_hz=identity[4],comparison=identity[10],
                run_id=identity[5],case_id=identity[7],seed=identity[6],sample_count=len(diagnostics),
                configurations=configurations,sampling=identity[8],per_rank_core_stage=rows,
                bandwidth=bandwidth,per_rank_bandwidth=bandwidth_summary(bandwidth),
                diagnostics=diagnostic_summary(diagnostics),
                bandwidth_interpretation='Effective counted bytes / max_core_kernel duration proxy, not link peak utilization or isolated NIC service time. Each iteration is divided before statistics. Counts represent rank-initiated read plus write data transfers, not pure rank TX or physical wire bytes. Reads are included in totals; write_bytes = bytes - read_bytes. Local bytes are not link traffic; intrahost_total includes first and forward. No cross-rank clock alignment.',
                per_rank_stage_max=[dict(operation=op,rank=rank,stage=stage,unit=unit,**stats(list(values.values())))
                                    for (op,rank,stage),values in sorted(stage_max.items())],
                note='Per-call diagnostic stage totals; nested/parallel stages cannot be added. No cross-rank timestamp subtraction.')


def discover(source, latest=False):
    if latest:
        runs = [p for p in source.iterdir() if p.is_dir() and (p/'cases.json').is_file()]
        if not runs:
            raise ValueError('No run directories containing cases.json')
        # Manifest mtime is the run creation time, unaffected by later report directories.
        source = max(runs, key=lambda p: ((p/'cases.json').stat().st_mtime_ns, p.name))
    directories = sorted(p for p in source.rglob('ll_stage_timer')
                         if p.is_dir() and not (p/'summary.json').is_file())
    if source.name == 'll_stage_timer' or not directories:
        directories = [source]
    return source, directories


def expert_reports(paths, frequency_hz):
    spec=importlib.util.spec_from_file_location('expert_timer_parser',ROOT/'deep_ep/utils/ll_expert_timer.py')
    expert=importlib.util.module_from_spec(spec); spec.loader.exec_module(expert)
    rows, groups, presence = [], {}, set()
    for path in paths:
        saved=json.loads(path.read_text())
        if saved['operation'] != 'dispatch':
            if 'expert' in saved: raise ValueError('Expert records only supported for dispatch')
            continue
        payload=saved.get('expert'); meta=saved['metadata']; presence.add(payload is not None)
        if payload is None:
            if (meta.get('sampling') or {}).get('expert_timer'):
                raise ValueError('Missing requested expert records')
            continue
        if (payload['rank'],payload['world']) != (meta['rank'],meta['world_size']):
            raise ValueError('Expert sample identity mismatch')
        if payload['experts'] != meta['config'].get('num_experts',payload['experts']):
            raise ValueError('Expert count differs from case configuration')
        report=expert.parse_records(**payload, frequency_hz=frequency_hz)
        if report['active_aiv'] != len(saved['raw']) or report['boundary_mode'] != ('observe' if saved['raw'][0][0][5]==1 else 'drain'):
            raise ValueError('Expert/stage core count or boundary mode mismatch')
        per_sample={}
        for row in report['records']:
            rows.append(dict(iteration=meta['iteration'], **row))
            key=(row['executing_rank'],row['expert_id'],row['expert_rank'],row['stage'])
            entry=per_sample.setdefault(key,dict(total=0,calls=0,first=row['first_tick'],last=row['last_tick']))
            entry['total']+=row['total_ticks']; entry['calls']+=row['calls']
            entry['first']=min(entry['first'],row['first_tick']);entry['last']=max(entry['last'],row['last_tick'])
        for key,value in per_sample.items(): groups.setdefault(key,[]).append(value)
    if len(presence)>1: raise ValueError('Mixed presence of expert diagnostics')
    summaries=[]
    factor=1e6/frequency_hz if frequency_hz is not None else 1
    for (rank,expert_id,owner,stage),values in sorted(groups.items()):
        summaries.append(dict(executing_rank=rank,expert_id=expert_id,expert_rank=owner,stage=stage,
            unit='us' if frequency_hz is not None else 'ticks',
            duration_kind='counter_only' if stage=='packets' else 'summed_core_work',
            **stats([v['total']*factor for v in values]),
            calls_total=sum(v['calls'] for v in values),mean_calls=statistics.mean(v['calls'] for v in values),
            mean_span=statistics.mean((v['last']-v['first'])*factor for v in values)))
        if stage=='packets':
            for key in ('mean','minimum','maximum','p50','p90','p99'): summaries[-1][key]=None
    return rows,summaries


def process_directory(source, output, frequency_hz, trace_iteration, lane_layout='core', show_repeated=False):
    paths = sorted(source.rglob('sample_*.json'))
    report = analyze(paths, frequency_hz)
    selected = []
    for path in paths:
        item = json.loads(path.read_text())
        if item['metadata']['iteration'] == trace_iteration:
            selected.append(item)
    if not selected:
        raise ValueError(f'No diagnostic iteration {trace_iteration} in {source}')
    trace = timer.chrome_trace(selected, frequency_hz, lane_layout, show_repeated) if report['frequency_hz'] is not None else None
    expert_rows, expert_summary = expert_reports(paths, report['frequency_hz'])
    report['per_expert']=expert_summary
    report['expert_interpretation']='Per executing rank and expert; core durations are summed work, not latency. Packet stage counts deliveries only. Shared FIRST_HOP preparation and batch waiting are not assigned. Only samples with executed records enter stage statistics.'
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    (output/'bottleneck_summary.md').write_text(render_bottleneck_report(report), encoding='utf-8')
    rows = report['per_rank_core_stage']
    with (output/'stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if trace is not None:
        (output/'trace.json').write_text(json.dumps(trace))
    with (output/'bandwidth.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['bandwidth'][0]))
        writer.writeheader();writer.writerows(report['bandwidth'])
    for filename,values in (('experts.csv',expert_summary),('expert_cores.csv',expert_rows)):
        if values:
            with (output/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]))
                writer.writeheader();writer.writerows(values)
    print(f'{source} -> {output} (all samples summarized; trace iteration {trace_iteration}; bottleneck_summary.md)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Timer directory, case, run, or run parent; every rank required per case')
    parser.add_argument('--latest',action='store_true',help='Select newest child run by cases.json creation mtime')
    parser.add_argument('--frequency-hz',type=float,default=None,
                        help='Override recorded counter frequency; missing frequency retains ticks without rates or trace')
    parser.add_argument('--trace-iteration',type=int,default=0,help='Timeline iteration; statistics always include all iterations')
    parser.add_argument('--lane-layout',choices=('core','stage'),default='core',help='One track per AIV (default), or separate stage tracks')
    parser.add_argument('--show-repeated',action='store_true',help='Restore the six repeated sender envelopes (statistics are always retained)')
    parser.add_argument('--output-dir',type=Path,help='Default: stage-analysis under selected source; existing reports rejected')
    args=parser.parse_args()
    source=args.directory if args.directory.is_absolute() else ROOT/args.directory
    source, directories = discover(source, args.latest)
    output = (args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir) if args.output_dir else source/'stage-analysis'
    if len(directories)>1 and (output/'REPORT_INDEX.md').exists():
        raise FileExistsError('Report index already exists; use a new output directory')
    documents = []
    for directory in directories:
        destination = output if len(directories)==1 else output/directory.relative_to(source)
        process_directory(directory, destination, args.frequency_hz, args.trace_iteration, args.lane_layout, args.show_repeated)
        documents.append((directory.relative_to(source), (destination/'bottleneck_summary.md').relative_to(output)))
    if len(documents) > 1:
        from urllib.parse import quote
        with (output/'REPORT_INDEX.md').open('x', encoding='utf-8') as stream:
            stream.write('# LL各case瓶颈报告\n\n不同shape/路由单独分析，不合并统计。\n\n' +
                '\n'.join(f'- [{str(case).replace(chr(93), "")}]({quote(path.as_posix())})' for case,path in documents) + '\n')


if __name__=='__main__':main()
