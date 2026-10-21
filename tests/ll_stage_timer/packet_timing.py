"""Per-core dispatch work counts and timing, without inferring DMA completion."""
from collections import defaultdict
import csv
import statistics

COUNTS = {'sent': 'INITIAL_PACKETS', 'received': 'RECV_READY_TOKENS',
          'forwarded': 'RELAY_PACKETS', 'received_first_hop': 'RECV_FIRST_HOP',
          'received_forwarded': 'RECV_TERMINAL'}
PHASES = ('INIT', 'STATIC_BEFORE_SEND', 'PREPARE_INPUT', 'BUILD_Q_TABLE', 'COUNT_COMPUTE',
          'WAIT_RECV_COUNTS', 'BUILD_RECV_CUMSUM', 'WAIT_CUMSUM',
          'PREFIX_COMPLETE', 'PREFIX_LOAD_WAIT', 'PREFIX_READY_SPAN',
          'RECV_PREPARE', 'SEND_TOKENS', 'SEND_PACKET', 'RECV_POLL',
          'RECV_COPY', 'RECV_INPUT_WAIT', 'RELAY_ISSUE', 'OUTPUT_ISSUE',
          'RECV_BUFFER_WAIT', 'TOKEN_INPUT_WAIT', 'TOKEN_QUANT_WAIT',
          'RECV_COMPLETE_WAIT', 'FINAL_DRAIN')
CLUES = ('PREPARE_INPUT', 'BUILD_Q_TABLE', 'COUNT_COMPUTE', 'WAIT_RECV_COUNTS', 'BUILD_RECV_CUMSUM',
         'WAIT_CUMSUM', 'PREFIX_LOAD_WAIT', 'RECV_PREPARE', 'SEND_PACKET',
         'RECV_POLL', 'RECV_INPUT_WAIT', 'RECV_BUFFER_WAIT', 'TOKEN_INPUT_WAIT',
         'TOKEN_QUANT_WAIT', 'RELAY_ISSUE', 'OUTPUT_ISSUE', 'FINAL_DRAIN')


def report(samples):
    rows, clues = [], []
    for meta, decoded in samples:
        if decoded['operation'] != 'dispatch':
            continue
        if decoded['schema'] < 14:
            continue  # Unavailable, never manufacture zero counters for old files.
        cores = defaultdict(dict)
        for record in decoded['records']:
            cores[record['core']][record['stage']] = record
        for core, stages in sorted(cores.items()):
            row = dict(rank=meta['rank'], iteration=meta['iteration'], core=core,
                       mode=decoded['boundary_mode'], kernel_ticks=stages['KERNEL']['total_ticks'])
            for name, stage in COUNTS.items():
                row[name] = stages.get(stage, {}).get('calls', 0)
            if row['received_first_hop'] + row['received_forwarded'] > row['received']:
                raise ValueError('Packet receive classes exceed received packet count')
            row['received_direct'] = row['received'] - row['received_first_hop'] - row['received_forwarded']
            for stage in PHASES:
                measured = stages.get(stage)
                # Missing timing is not a measured zero. In particular empty
                # senders have no STATIC_BEFORE_SEND or per-packet average.
                row[stage+'_ticks'] = measured['total_ticks'] if measured else None
                row[stage+'_calls'] = measured['calls'] if measured else 0
                row[stage+'_per_call_ticks'] = measured['average_ticks'] if measured else None
            rows.append(row)
        if cores:
            slow = max(cores, key=lambda c: cores[c]['KERNEL']['total_ticks'])
            stages = cores[slow]
            kernel = stages['KERNEL']['total_ticks']
            ranked = sorted(((s, stages[s]['total_ticks']) for s in CLUES if s in stages),
                            key=lambda pair: pair[1], reverse=True)
            clues.append(dict(rank=meta['rank'], iteration=meta['iteration'], core=slow,
                              stages=[dict(stage=s, ticks=t, kernel_ratio=t/kernel if kernel else None)
                                      for s, t in ranked[:3]]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['rank'], row['core'])].append(row)
    per_core = []
    for (rank, core), items in sorted(grouped.items()):
        value = dict(rank=rank, core=core, rounds=len(items))
        for name in (*COUNTS, 'received_direct'):
            value[name+'_total'] = sum(r[name] for r in items)
            value[name+'_per_round'] = value[name+'_total'] / len(items)
        for stage in PHASES:
            values = [r[stage+'_ticks'] for r in items if r[stage+'_ticks'] is not None]
            calls = sum(r[stage+'_calls'] for r in items)
            value[stage+'_mean_ticks'] = statistics.mean(values) if values else None
            value[stage+'_measured_rounds'] = len(values)
            value[stage+'_per_call_ticks'] = sum(values)/calls if calls else None
        per_core.append(value)
    return dict(samples=rows, per_core=per_core, slow_core_clues=clues)


def write(output, result):
    lines = ['# Dispatch 分核包数与耗时', '',
             '包数为实际工作量：一个包携带一个源 token 的数据，但一个源 token 可初发多个包。',
             '以下包数为每轮均值；CSV保留逐轮整数和跨轮总数。包括本卡目标，不能全部当作跨卡流量。',
             '接收转发包按修改前的 TERMINAL 类型计数；DIRECT、FIRST_HOP、TERMINAL互斥。',
             '转发数为输出包数，不是首跳组数；不统计唯一源 token 数。', '',
             '|rank|core|轮次|初发包/轮|接收包/轮|转发包/轮|收到首跳包/轮|收到转发包/轮|',
             '|---|---|---|---|---|---|---|---|']
    def number(value):
        return '未测/无任务' if value is None else f'{value:.2f}'
    for row in result['per_core']:
        lines.append('|'+ '|'.join(str(row[n]) for n in ('rank','core','rounds'))+'|' +
                     '|'.join(number(row[n+'_per_round']) for n in COUNTS)+'|')
    lines += ['', '## 各核时间（原始 ticks）', '',
              'INIT单列；STATIC_BEFORE_SEND从Process开始至本核第一次发送准备前，包括其间count/前缀等待，不能称为纯计算开销。',
              'BUILD_Q_TABLE是S核去重路径的一次建表（schema15），位于SEND_TOKENS之前；未出现不补零。PREPARE_INPUT是发送缓冲/输入准备。SEND_PACKET从消费预取输入到提交一个初发包，含量化/打包/既有依赖等待，不含更早的输入预取提交。',
              'RECV_PREPARE仅任务/缓冲准备；count、前缀准备分别见CSV。RECV_COPY从消费预取包到输出提交，包括转发、输出等子阶段。',
              'OUTPUT_ISSUE仅最终payload/scale/元数据写请求提交；RECV_INPUT_WAIT为原有MTE2_S等待。',
              '以上均不是网络到达延迟、远端消费确认或单包DMA完成时间；不增加同步以测这些量。',
              '每包均值为总ticks/总调用数；阶段均值只包含存在记录的轮次，覆盖轮次数在CSV。父子阶段不相加。', '',
              '|rank|core|发送前跨度/轮|发送准备/轮|q建表/轮|发送/包|接收准备/轮|接收消费/包|输出提交/包|输入等待/包|',
              '|---|---|---|---|---|---|---|---|---|---|']
    fields = ('STATIC_BEFORE_SEND_mean_ticks','PREPARE_INPUT_mean_ticks','BUILD_Q_TABLE_mean_ticks','SEND_PACKET_per_call_ticks',
              'RECV_PREPARE_mean_ticks','RECV_COPY_per_call_ticks','OUTPUT_ISSUE_per_call_ticks','RECV_INPUT_WAIT_per_call_ticks')
    for row in result['per_core']:
        lines.append(f"|{row['rank']}|{row['core']}|"+'|'.join(number(row[f]) for f in fields)+'|')
    lines += ['', '## 优化线索', '',
              '取每rank每轮KERNEL最慢核，列出已全量计时阶段中的前三项。占比只用于定位，不代表可获得的加速或链路利用率。',
              'TOKEN_INPUT/QUANT/METADATA/BUFFER_WAIT/PACK/SEND_ISSUE、RECV_INPUT/OUTPUT、CAL_TOKEN及LOCAL_WINDOW_COPY/FORMAT_OUTPUT仅测前两次（observe），不外推占比。早期样本可能偏向冷启动。关键输入/量化末尾/接收lane依赖等待保留全量。',
              'trace只含每阶段前两个区间；后续空白不代表核空闲。总工作量以metrics和本报告为准。',
              'timer-OFF才是性能基线；ON未capture可检查编译布局/驻留开销，ON capture还包含探针及KERNEL后的全核同步/导出。', '',
              '|rank|轮次|最慢core|阶段与该核KERNEL占比|', '|---|---|---|---|']
    for row in result['slow_core_clues']:
        labels = ', '.join(f"{s['stage']} {s['kernel_ratio']:.1%}" if s['kernel_ratio'] is not None else s['stage']
                           for s in row['stages'])
        lines.append(f"|{row['rank']}|{row['iteration']}|{row['core']}|{labels}|")
    if not result['samples']:
        lines += ['', '无schema14 dispatch数据；旧记录不能补出新计数。']
    (output/'dispatch_packets.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for filename, values in (('dispatch_packet_samples.csv', result['samples']),
                             ('dispatch_packet_cores.csv', result['per_core'])):
        if values:
            with (output/filename).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
