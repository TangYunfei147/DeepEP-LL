"""Schema13 dispatch preparation and communication diagnosis, same core/sample only."""
import statistics

# Candidate list contains leaf/subphase costs; NEVER add it into a total.
CANDIDATES = (
 'SEND_SETUP','Q_SEED_SUBMIT','Q_SEED_WAIT','Q_LOCAL_UPDATE','ROUTE_CLASSIFY','DIRECT_PLAN',
 'TOKEN_INPUT_WAIT','TOKEN_QUANT','TOKEN_QUANT_WAIT','TOKEN_METADATA','TOKEN_PACK',
 'TX_PACK_READY_WAIT','TOKEN_BUFFER_WAIT','TOKEN_SEND_ISSUE','TX_METADATA_PUBLISH',
 'SEND_RETIRE_WAIT','WAIT_RECV_COUNTS','BUILD_RECV_CUMSUM','PREFIX_PUBLISH_WAIT','WAIT_CUMSUM','PREFIX_LOAD',
 'RX_FLAG_ISSUE','RX_FLAG_WAIT','RX_FLAG_REDUCE','RX_FLAG_DECODE','RX_LANE_REUSE_WAIT',
 'RX_PAYLOAD_READY_WAIT','RX_METADATA','RELAY_ISSUE','RX_PAYLOAD_OUTPUT','RX_META_OUTPUT',
 'RX_PENDING_UPDATE','CLEAR_LOCAL_FLAGS','RECV_COMPLETE_WAIT')
ADVICE={
 'SEND_SETUP':'拆看q准备和初始化输入，不能将SEND_SETUP与子阶段相加',
 'Q_SEED_WAIT':'核查VF播种与此前Vector依赖；A5单独对比播种策略',
 'Q_LOCAL_UPDATE':'核查本片Scalar直方图更新成本',
 'ROUTE_CLASSIFY':'核查去重分类与专家分布',
 'DIRECT_PLAN':'核查DIRECT描述符准备，保持发包顺序做局部A/B',
 'TOKEN_INPUT_WAIT':'核查输入预取距离和同token数据复用',
 'TOKEN_BUFFER_WAIT':'核查发送lane复用和在途写压力',
 'TX_PACK_READY_WAIT':'核查pack/Vector到MTE3依赖',
 'TOKEN_SEND_ISSUE':'核查小包下发次数；不是DMA传输持续时间',
 'SEND_RETIRE_WAIT':'核查接收之后仍未退休的本核发送lane',
 'WAIT_RECV_COUNTS':'核查上游count发布和R0轮询',
 'WAIT_CUMSUM':'核查R0前缀发布是否推迟本核接收',
 'RX_FLAG_ISSUE':'核查逐包flag DMA发射及地址生成',
 'RX_FLAG_WAIT':'结合就绪率区分轮询策略和数据生产延迟',
 'RX_FLAG_REDUCE':'核查flag归约/同步成本',
 'RX_FLAG_DECODE':'核查Scalar标记解码',
 'RX_LANE_REUSE_WAIT':'核查旧输出/relay对接收lane的占用',
 'RX_PAYLOAD_READY_WAIT':'核查预取深度及原MTE2依赖',
 'RELAY_ISSUE':'核查首跳成员数和relay提交量，不直接归因为拥塞',
 'RX_PAYLOAD_OUTPUT':'核查连续记录批量输出',
 'RX_META_OUTPUT':'核查小metadata写合并',
 'RX_PENDING_UPDATE':'核查pending逐包维护',
 'RECV_COMPLETE_WAIT':'核查最终输出/relay/清flag尾部'
}
ENVELOPES=('BEFORE_FIRST_SEND','BEFORE_FIRST_RECV_POLL')
COUNTERS=('TX_PACKETS','RX_PROBED','RX_READY','RX_EMPTY_POLLS','RX_PACKETS','RELAY_PACKETS')
WAIT_STAGES={'Q_SEED_WAIT','TOKEN_INPUT_WAIT','TOKEN_QUANT_WAIT','TOKEN_BUFFER_WAIT',
 'TX_PACK_READY_WAIT','SEND_RETIRE_WAIT','WAIT_RECV_COUNTS','PREFIX_PUBLISH_WAIT','WAIT_CUMSUM',
 'PREFIX_LOAD','RX_FLAG_WAIT','RX_LANE_REUSE_WAIT','RX_PAYLOAD_READY_WAIT','RECV_COMPLETE_WAIT'}

def poll_snapshot_counts(st,probed,empty):
    """Count completed flag snapshots, not split submit/join timer fragments."""
    segments=st.get('RECV_POLL',{}).get('calls',0)
    counts=[st.get(name,{}).get('calls') for name in
            ('RX_FLAG_ISSUE','RX_FLAG_WAIT','RX_FLAG_DECODE')]
    if empty<0:
        raise ValueError('RX_EMPTY_POLLS must be nonnegative')
    if not (probed or empty or segments or any(value is not None for value in counts)):
        return segments,0,'no_receive'
    if counts[2] is None:
        # Earlier/incomplete schema13 exports cannot establish this denominator.
        # In particular, RECV_POLL can contain one OR two fragments per snapshot.
        if counts[0] is not None and counts[1] is not None and counts[0]!=counts[1]:
            raise ValueError('Inconsistent RX_FLAG_ISSUE/WAIT snapshot calls')
        return segments,None,'missing_flag_snapshot_metrics'
    if not counts[2]>0 or counts[0]!=counts[2] or counts[1]!=counts[2]:
        raise ValueError('Inconsistent RX_FLAG_ISSUE/WAIT/DECODE snapshot calls')
    if empty>counts[2]:
        raise ValueError('RX_EMPTY_POLLS exceeds completed flag snapshots')
    return segments,counts[2],'measured'

def report(samples):
    rows=[];costs=[];unsupported=0
    for meta,decoded in samples:
        if decoded['operation']!='dispatch': continue
        if decoded['schema']!=13:
            unsupported+=1;continue
        by_core={}
        for r in decoded['records']:by_core.setdefault(r['core'],{})[r['stage']]=r
        longest=max((st.get('KERNEL',{}).get('total_ticks',0) for st in by_core.values()),default=0)
        hz=decoded.get('frequency_hz')
        for core,st in sorted(by_core.items()):
            identity=dict(rank=meta['rank'],iteration=meta['iteration'],core=core,
                          boundary_mode=decoded.get('boundary_mode','observe'))
            kernel=st['KERNEL']['total_ticks']
            row=dict(**identity,kernel_ticks=kernel,kernel_us=kernel*1e6/hz if hz else None,
                     critical_core=bool(longest and kernel>=longest*.9))
            for name in (*ENVELOPES,'SEND_SETUP','PREPARE_INPUT','INIT'):
                value=st.get(name,{}).get('total_ticks')
                row[name.lower()+'_ticks']=value
                row[name.lower()+'_us']=value*1e6/hz if value is not None and hz else None
            intervals={i['stage']:i for i in decoded.get('intervals',[]) if i['core']==core and i.get('sample',0)==0}
            init=intervals.get('INIT');first=intervals.get('BEFORE_FIRST_SEND')
            entry_span=first['last_tick']-init['first_tick'] if init and first else None
            if entry_span is not None and entry_span<0:raise ValueError('Invalid initialization/first-send interval')
            row['entry_to_first_send_ticks']=entry_span
            row['entry_to_first_send_us']=entry_span*1e6/hz if entry_span is not None and hz else None
            for name in COUNTERS:row[name.lower()]=st.get(name,{}).get('calls',0)
            # Counters are explicitly instrumented only when their code runs.
            probed=row['rx_probed'];ready=row['rx_ready'];rx=row['rx_packets'];tx=row['tx_packets']
            row['ready_fraction']=ready/probed if probed else None
            row['probes_per_consumed_packet']=probed/rx if rx else None
            segments,snapshots,status=poll_snapshot_counts(st,probed,row['rx_empty_polls'])
            row['recv_poll_segments']=segments
            row['poll_snapshots']=snapshots
            row['poll_snapshot_status']=status
            # Compatibility alias: corrected to snapshots; legacy reports used
            # RECV_POLL fragments and understated the split-path empty fraction.
            row['poll_batches']=snapshots
            row['empty_poll_fraction']=row['rx_empty_polls']/snapshots if snapshots else None
            row['first_send_status']='measured' if tx and row['before_first_send_ticks'] is not None else ('missing' if tx else 'no_send')
            row['first_recv_status']='measured' if probed and row['before_first_recv_poll_ticks'] is not None else ('missing' if probed else 'no_receive')
            if ready>probed or ready!=rx:raise ValueError('Incomplete RX readiness/consumption counters')
            if tx and (row['before_first_send_ticks'] is None or not 0<=row['before_first_send_ticks']<=kernel):
                raise ValueError('Missing or invalid first-send envelope')
            if probed and (row['before_first_recv_poll_ticks'] is None or not 0<=row['before_first_recv_poll_ticks']<=kernel):
                raise ValueError('Missing or invalid first-receive envelope')
            ranked=[]
            for name in CANDIDATES:
                r=st.get(name)
                if not r:continue
                ticks=r['total_ticks'];calls=r['calls']
                cost=dict(**identity,stage=name,total_ticks=ticks,total_us=ticks*1e6/hz if hz else None,
                          calls=calls,per_call_ticks=ticks/calls,per_call_us=ticks*1e6/hz/calls if hz else None,kernel_share=ticks/kernel if kernel else None,
                          critical_core=row['critical_core'],next_check=ADVICE.get(name,'结合原始同核区间核对该阶段'),
                          interpretation='existing dependency instruction interval' if name in WAIT_STAGES else 'instrumented device instruction interval')
                costs.append(cost);ranked.append((ticks,name))
            row['largest_observed_costs']='; '.join(name for _,name in sorted(ranked,reverse=True)[:4])
            row['next_check']=ADVICE.get(sorted(ranked,reverse=True)[0][1],'结合原始同核区间核对') if ranked else 'no measured subphase'
            rows.append(row)
    aggregate=[]
    for key in sorted({(r['rank'],r['core']) for r in rows}):
        group=[r for r in rows if (r['rank'],r['core'])==key]
        item=dict(rank=key[0],core=key[1],samples=len(group),
                  critical_fraction=sum(r['critical_core'] for r in group)/len(group))
        for field in ('entry_to_first_send_us','entry_to_first_send_ticks','before_first_send_us','before_first_send_ticks','send_setup_us','send_setup_ticks',
                      'before_first_recv_poll_us','before_first_recv_poll_ticks','kernel_us','kernel_ticks',
                      'ready_fraction','probes_per_consumed_packet','empty_poll_fraction'):
            values=[r[field] for r in group if r[field] is not None]
            item[field+'_samples']=len(values)
            item[field+'_p50']=statistics.median(values) if values else None
        aggregate.append(item)
    return dict(schema=13,status='available' if rows else 'requires_schema13_dispatch',unsupported_samples=unsupported,
                samples=rows,costs=costs,per_core=aggregate,
                note='Elapsed intervals and accumulated costs are nested. Do not add them. '
                     'poll_snapshots uses RX_FLAG_DECODE calls, validated against RX_FLAG_ISSUE/WAIT; '
                     'poll_batches is its corrected compatibility alias, not the legacy RECV_POLL fragment count. '
                     'recv_poll_segments preserves RECV_POLL calls; split paths also have two RX_FLAG_REDUCE calls per snapshot. '
                     'Missing snapshot metrics leave the denominator and empty fraction unavailable. '
                     'Submission is not transfer completion. No cross-rank clock subtraction. Relay-complete mode adds drains and is not an observe baseline. '
                     'Largest observed costs are prioritization hints, not proof of a hardware bottleneck.')

def markdown(data):
    lines=['# Dispatch准备与通信诊断','',data['note'],'',
      'BEFORE_FIRST_SEND：本核KERNEL起点至第一包DataCopy下发之前（包含首包输入/打包）。',
      'SEND_SETUP：进入Sender初始化至发送循环前；Q_*为其子阶段。',
      'entry_to_first_send：同核INIT真实起点到首发真实终点；INIT在KERNEL外单列。无发包核不补零。',
      'BEFORE_FIRST_RECV_POLL：本核KERNEL起点至首次接收Probe之前；不表示首包到达。',
      'RX_FLAG_*拆分下发/等待/归约/解码，RX_PAYLOAD_READY_WAIT是原有依赖等待。',
      '空快照率=RX_EMPTY_POLLS / RX_FLAG_DECODE.calls；ISSUE/WAIT/DECODE次数须一致。',
      'poll_snapshots及兼容字段poll_batches均表示完整快照数；旧报表poll_batches曾使用RECV_POLL片段数。',
      'recv_poll_segments保留RECV_POLL片段数，分段路径每快照有两段；RX_FLAG_REDUCE.calls也按片段计。缺解码计数时快照数和空快照率留空。',
      'ready_fraction低只说明被探测项尚未全部就绪，不能单独归因为网络拥塞。',
      '计时仅记录每类前两次真实区间，metrics累计全部调用；不能重建逐包完整时间线。','',
      '## 各核准备开销','',
      '| rank | core | 样本 | INIT至首发 p50(ticks) | KERNEL至首发 p50(ticks) | 初始化 p50(ticks) | 首Probe前 p50(ticks) | ready比例 p50 |',
      '|---|---|---|---|---|---|---|---|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in data['per_core']:
        lines.append(f"| {r['rank']} | {r['core']} | {r['samples']} | {fmt(r['entry_to_first_send_ticks_p50'])} | {fmt(r['before_first_send_ticks_p50'])} | {fmt(r['send_setup_ticks_p50'])} | {fmt(r['before_first_recv_poll_ticks_p50'])} | {fmt(r['ready_fraction_p50'])} |")
    lines+=['','## 长时核上的优先排查项','',
            '按每rank/轮最长核跨度的90%筛选，仅用于定位。完整数值见CSV，先观察同核/轮再聚合。','']
    for r in data['samples']:
        if r['critical_core']:
            lines.append(f"- rank{r['rank']} / iter{r['iteration']} / core{r['core']}: {r['largest_observed_costs']}。{r['next_check']}")
    return '\n'.join(lines)+'\n'
