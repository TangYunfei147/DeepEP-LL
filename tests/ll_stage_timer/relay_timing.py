"""Keep relay issue and isolated-MTE3 completion measurements distinct."""
from collections import defaultdict


def report(samples):
    rows=[];identity=set();seen=set()
    for meta,decoded in samples:
        if decoded['operation']!='dispatch': continue
        schema=decoded['schema'];mode=decoded['boundary_mode'];hz=decoded['frequency_hz']
        identity.add((schema,mode,hz))
        cores=defaultdict(dict)
        for record in decoded['records']:cores[record['core']][record['stage']]=record
        for core,stages in sorted(cores.items()):
            key=(meta['rank'],core,meta['iteration'])
            if key in seen: raise ValueError('Duplicate relay rank/core/iteration')
            seen.add(key)
            available=schema==10
            issue=stages.get('RELAY_ISSUE')
            groups=issue['calls'] if issue else 0 if available else None
            packets=stages.get('RELAY_PACKETS',{}).get('calls',0) if available else None
            measured_completion=available and mode=='relay-complete'
            rows.append(dict(rank=meta['rank'],iteration=meta['iteration'],core=core,mode=mode,
                status='unavailable' if not available else 'measured' if groups else 'no_relay',
                groups=groups,relay_writes=packets,
                kernel_ticks=stages.get('KERNEL',{}).get('total_ticks'),
                issue_ticks=issue['total_ticks'] if issue else 0 if available else None,
                complete_ticks=stages.get('RELAY_COMPLETE',{}).get('total_ticks',0) if measured_completion else None,
                pre_drain_ticks=stages.get('RELAY_PRE_DRAIN',{}).get('total_ticks',0) if measured_completion else None))
    if len(identity)>1: raise ValueError('Mixed relay schemas, diagnostic modes or frequencies')
    mode=next(iter(identity))[1] if identity else None
    hz=next(iter(identity))[2] if identity else None
    grouped=defaultdict(list)
    for row in rows:grouped[(row['rank'],row['core'])].append(row)
    summaries=[]
    for (rank,core),items in sorted(grouped.items()):
        available=all(i['status']!='unavailable' for i in items)
        groups=sum(i['groups'] for i in items) if available else None
        writes=sum(i['relay_writes'] for i in items) if available else None
        item=dict(rank=rank,core=core,mode=mode,samples=len(items),groups=groups,relay_writes=writes,
                  status='unavailable' if not available else 'measured' if groups else 'no_relay')
        for name in ('issue','complete','pre_drain'):
            values=[i[name+'_ticks'] for i in items]
            total=sum(values) if all(v is not None for v in values) else None
            per_group=total/groups if total is not None and groups else None
            item[name+'_ticks_total']=total
            item[name+'_us_total']=total*1e6/hz if total is not None and hz else None
            item[name+'_ticks_per_group']=per_group
            item[name+'_us_per_group']=per_group*1e6/hz if per_group is not None and hz else None
        # A fanout group has several writes; this is amortized, not individual latency.
        item['issue_ticks_per_write']=item['issue_ticks_total']/writes if available and writes else None
        # Pair completion and KERNEL using exactly these rank/core/iteration
        # rows, including measured zero-relay rounds. Never sum a subset of
        # KERNEL records or divide an accumulated numerator by a mean duration.
        kernels=[i['kernel_ticks'] for i in items]
        kernel=sum(kernels) if all(v is not None for v in kernels) else None
        complete=item['complete_ticks_total']
        item['kernel_ticks_total']=kernel
        item['kernel_us_total']=kernel*1e6/hz if kernel is not None and hz else None
        item['complete_kernel_ratio']=complete/kernel if complete is not None and kernel is not None and kernel>0 else None
        item['relay_writes_per_group']=writes/groups if available and groups else None
        summaries.append(item)
    return dict(mode=mode,frequency_hz=hz,samples=rows,per_core=summaries,
        interpretation='Issue is scalar loop/submit time including possible backpressure. Completion mode first drains '
        'prior local MTE3 work, then times issue through producer MTE3 completion. Pre-drain is separate; '
        'issue is nested inside completion, never add them. Completion is not remote-consumer acknowledgement '
        'or uncontended link latency. Relay writes include self-rank and peer-rank targets. Counts are per FIRST_HOP fanout group and per relay write. '
        'Completion/KERNEL is a ratio of accumulated durations from the same rank/core/iterations, not a normal-mode bottleneck proof. '
        'Average fanout is total relay writes / total FIRST_HOP groups. '
        'Old formats are unavailable, not zero; completion is unmeasured in observe mode.')


def markdown(result):
    unit='us' if result['frequency_hz'] else 'ticks'
    lines=['# Relay下发与完成时延','',f"采样模式：{result['mode'] or '无dispatch样本'}。",
           '下发区间：relay循环内地址计算、写请求下发及可能的发射背压。',
           '完成区间：排空此前本核MTE3后，从开始下发到本组MTE3写完成；不表示对端已消费。',
           '统计包含relay循环的本rank及其他rank目标，没有将其全部视为跨卡链路流量。',
           '前置排空单列。下发包含在完成区间中，两者不能相加。',
           'relay-complete会破坏正常重叠，只用于诊断；其他核/卡仍有并发流量。','',
           '完成占比=同rank、同核、相同采样轮次的累计完成时间/累计KERNEL时间；包含有测量但零转发的轮次。',
           '该比例只描述完成诊断模式，不能直接判定正常模式的瓶颈或可获得的加速比例。','',
           f'| rank | core | 状态 | 轮次 | 首跳组数 | relay写数 | 平均转发次数/组 | 累计完成 {unit} | 累计KERNEL {unit} | 完成/KERNEL | 下发 {unit}/组 | 完成 {unit}/组 | 前置排空 {unit}/组 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in result['per_core']:
        def value(name):
            v=row[f'{name}_{unit}_per_group']
            return '未测/不适用' if v is None else f'{v:.3f}'
        def field(name,percent=False):
            v=row[name]
            return '未测/不适用' if v is None else f'{v:.2%}' if percent else f'{v:.3f}'
        lines.append(f"| {row['rank']} | {row['core']} | {row['status']} | "
                     f"{row['samples']} | "
                     f"{row['groups'] if row['groups'] is not None else '未知'} | "
                     f"{row['relay_writes'] if row['relay_writes'] is not None else '未知'} | "
                     f"{field('relay_writes_per_group')} | {field('complete_'+unit+'_total')} | "
                     f"{field('kernel_'+unit+'_total')} | {field('complete_kernel_ratio',True)} | "
                     f"{value('issue')} | {value('complete')} | {value('pre_drain')} |")
    if not result['per_core']:lines+=['','没有dispatch诊断数据。']
    lines+=['','均值按总ticks/总首跳组数加权；一次首跳组可能含多条转发写。',
            '同组内各写可能重叠，按写数摊销的值不是单条写延迟；跨核累计值也不是kernel时延。',
            '正常observe未测完成时间，旧schema未测relay，均不补零。']
    return '\n'.join(lines)+'\n'
