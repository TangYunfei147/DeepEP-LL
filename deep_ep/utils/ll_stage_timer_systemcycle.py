"""Schema18 LL stage timings: same-core milestones and selected cumulative scopes.

No added pipeline barrier. Scope sums can contain other scopes; they are not a
partition of elapsed time. Only milestone spans have recorded timeline endpoints.
"""
import math

MAGIC = 0x4C4C5431
POINTS = ('ENTRY','INIT_DONE','COUNTS_SENT','COUNTS_RECEIVED','PREFIX_LOCAL_DONE',
          'PREFIX_READY','SEND_PREPARED','FIRST_SEND','SEND_DONE','RECV_PREPARED',
          'FIRST_RECV_PROBE','FIRST_RECV_READY','RECV_DONE','WORK_DONE',
          'SOURCE_SUBMIT_DONE','PREFIX_WAIT_ENTER','PREFIX_WAIT_EXIT','RECV_PENDING_EMPTY',
          'ENTRY_SETUP_DONE','LAYOUT_DONE','MAGIC_DONE','COUNTS_INIT_DONE','COUNTS_BEGIN',
          'SOURCE_BEGIN','SOURCE_RETURN','COUNTS_WAIT_BEGIN','PREFIX_BEGIN','PREFIX_RETURN',
          'COMPLETE_PREFIX_BEGIN','COMPLETE_PREFIX_RETURN','RECV_BEGIN','RECV_TASKS_DONE',
          'RECV_BUFFERS_DONE','RETIRE_BEGIN','RETIRE_END','FINISH_BEGIN',
          'COMBINE_PREPARE_BEGIN','COMBINE_PREPARE_END','COMBINE_ROLE_BEGIN','COMBINE_ROLE_END')
METRIC_NAMES = ('COUNT_COMPUTE','Q_SEED_SUBMIT','Q_SEED_WAIT','Q_LOCAL_UPDATE','ROUTE_CLASSIFY',
                'DIRECT_PLAN','TOKEN_INPUT_WAIT','TOKEN_QUANT_WAIT','TOKEN_METADATA','TOKEN_PACK',
                'TOKEN_SEND_ISSUE','RECV_POLL','RX_FLAG_WAIT','RX_FLAG_REDUCE','RX_FLAG_DECODE',
                'RX_PAYLOAD_READY_WAIT','RX_METADATA','RELAY_ISSUE','RX_PAYLOAD_OUTPUT','RX_META_OUTPUT',
                'RX_PENDING_UPDATE','FINAL_DRAIN','COMBINE_SEND','COMBINE_RECV','COMBINE_SEND_FINISH',
                'COMBINE_INDEX_INPUT','COMBINE_INPUT','COMBINE_BUFFER_WAIT','COMBINE_COPY_PACK',
                'COMBINE_SEND_ISSUE','COMBINE_RECV_POLL','COMBINE_DEQUANT_REDUCE',
                'COMBINE_PROCESS_TOKEN','COMBINE_CLEAR_FLAGS','TX_PACKETS','RX_PACKETS',
                'RELAY_PACKETS','RX_PROBED','RX_READY','COMBINE_READY_SLOTS')
METRIC_INDICES = (22,82,83,84,85,86,64,65,18,20,21,29,89,90,91,93,94,66,95,96,102,47,
                  32,34,38,63,35,57,58,37,39,40,41,43,103,107,69,104,105,55)
COUNTER_INDICES = tuple(range(34,40))
ROLES = {1:'send_token',2:'cumsum',3:'send_and_recv',4:'recv'}


def phase(key, label, first, last, note):
    return dict(stage=key,label=label,start_point=first,end_point=last,note=note,kind='span')


COMMON_PHASES = (
    phase('ENTRY_TO_WORK_DONE','核内总耗时','ENTRY','WORK_DONE','含入口、初始化及业务处理；不含最后诊断导出，完整task另列。'),
    phase('INIT','初始化总耗时','ENTRY','INIT_DONE','父包络，包含入口/布局/magic及对象初始化细分。'),
    phase('KERNEL','处理阶段总耗时','INIT_DONE','WORK_DONE','同核初始化后到Process返回；可包含异步依赖和已有排空。'),
    phase('ENTRY_SETUP','入口准备','ENTRY','ENTRY_SETUP_DONE','包含本次诊断初始化开销，不能当作OFF固定开销。'),
)
PHASES = {
    'dispatch': COMMON_PHASES + (
        phase('LAYOUT','布局与overlap配置','ENTRY_SETUP_DONE','LAYOUT_DONE','原布局计算及ConfigureOverlap。'),
        phase('MAGIC','窗口轮转','LAYOUT_DONE','MAGIC_DONE','RotateDispatchServerDedupMagic原有工作与依赖。'),
        phase('COUNTS_INIT','计数对象初始化','MAGIC_DONE','COUNTS_INIT_DONE','Counts.Init返回，不包含后续SendCnt计算。'),
        phase('SEND_RECV_INIT','收发对象初始化','COUNTS_INIT_DONE','INIT_DONE','Sender.Init和Receiver.Init。'),
        phase('COUNT_SEND','计数准备与提交','COUNTS_BEGIN','COUNTS_SENT','含池设置、路由加载、统计及count下发；无COUNTS_SENT则留空。'),
        phase('SOURCE','发送函数包络','SOURCE_BEGIN','SOURCE_RETURN','S远端或R本组发送函数；Async返回不代表MTE3完成。'),
        phase('SOURCE_SETUP','发送缓冲与路由准备','SOURCE_BEGIN','SEND_PREPARED','输入/路由缓冲、q表和lane准备。'),
        phase('SOURCE_SUBMIT','路由与packet提交','SEND_PREPARED','SOURCE_SUBMIT_DONE','包含分类、元数据及packet/source元数据提交，不等于通信完成。'),
        phase('SOURCE_RETURN_TAIL','发送提交后至返回','SOURCE_SUBMIT_DONE','SOURCE_RETURN','原有退休/恢复及函数收尾，Async路径不新增排空。'),
        phase('COUNT_RECV','R0收计数','COUNTS_WAIT_BEGIN','COUNTS_RECEIVED','R0接收缓冲准备、等待、提取及原count-ready清理依赖。'),
        phase('PREFIX','R0前缀计算与发布','PREFIX_BEGIN','PREFIX_RETURN','含VF、前缀表和epoch提交；最终边界是提交侧。'),
        phase('PREFIX_VF_SUBMIT','前缀VF提交','PREFIX_BEGIN','PREFIX_LOCAL_DONE','到原V_MTE3下发，不是Scalar确认VF完成。'),
        phase('COMPLETE_PREFIX','前缀等待与加载','COMPLETE_PREFIX_BEGIN','COMPLETE_PREFIX_RETURN','CompleteCumsum原等待/加载；R0还包含计数输出下发。'),
        phase('PREFIX_WAIT','prefix-ready等待','PREFIX_WAIT_ENTER','PREFIX_WAIT_EXIT','原WaitCumSumFlag的Scalar包络。'),
        phase('RECEIVE','接收函数包络','RECV_BEGIN','RECV_DONE','含任务准备、收包、转发、本地输出和原最终排空。'),
        phase('RECV_TASKS','接收前缀与任务构建','RECV_BEGIN','RECV_TASKS_DONE','tasks.Build返回，包含空任务核真实准备。'),
        phase('RECV_BUFFERS','接收缓冲准备','RECV_TASKS_DONE','RECV_BUFFERS_DONE','仅非空任务核；含flags/probe/lane分配及原事件seed。'),
        phase('FIRST_READY','首次packet就绪等待','RECV_BUFFERS_DONE','FIRST_RECV_READY','包含首次探测及等待；并非单独网络延迟。'),
        phase('RECV_LOOP','收包循环包络','RECV_BUFFERS_DONE','RECV_PENDING_EMPTY','包含轮询、提交、relay及本地输出；不能再加内部累计项。'),
        phase('RECV_FINISH','接收循环后收尾','RECV_PENDING_EMPTY','RECV_DONE','含原lane退休、清flags与PIPE_ALL。'),
        phase('SENDER_RETIRE','发送lane退休提交','RETIRE_BEGIN','RETIRE_END','原MTE3_V事件循环；不是新增Scalar完成确认。'),
        phase('FINISH','最终排空与恢复','FINISH_BEGIN','WORK_DONE','Counts.Finish原PIPE_ALL到Process返回，含恢复CTRL。'),
    ),
    'combine': COMMON_PHASES + (
        phase('MAGIC','窗口轮转','ENTRY_SETUP_DONE','MAGIC_DONE','原RotateMagic及其TPipe作用域。'),
        phase('COMBINE_INIT','combine对象初始化','MAGIC_DONE','INIT_DONE','tiling读取和kernel.Init。'),
        phase('COMBINE_PREPARE','本地pending准备','COMBINE_PREPARE_BEGIN','COMBINE_PREPARE_END','仅去重路径CopyLocalPendingTokens，不存在则留空。'),
        phase('COMBINE_ROLE','角色工作包络','COMBINE_ROLE_BEGIN','COMBINE_ROLE_END','当前核实际S/R分支，含回传、等待及归约等；内部累计有包含关系。'),
        phase('COMBINE_FINISH','角色结束至Process返回','COMBINE_ROLE_END','WORK_DONE','包括原最终PIPE_ALL和收尾。'),
    ),
}
PAIRS = tuple(dict.fromkeys((row['stage'],row['start_point'],row['end_point'])
                           for rows in PHASES.values() for row in rows))
METRIC_LABELS = (
    'count计算','q播种提交','q播种等待','本片q更新','路由分类','DIRECT计划',
    '发送输入等待','发送量化等待','发送packet元数据','发送打包提交','发送packet提交',
    '接收探测包络','接收flag等待','接收flag归约提交','接收flag解码','接收payload依赖',
    '接收元数据','relay提交','本地payload输出提交','本地元数据输出提交','pending更新',
    '最终排空包络','combine发送包络','combine接收包络','combine发送收尾',
    'combine索引输入','combine输入包络','combine缓冲依赖','combine打包提交',
    'combine发送提交','combine接收等待','combine反量化归约','combine逐token包络',
    'combine清flags','发送packet数','接收packet数','relay packet数','探测槽位数',
    '就绪槽观测数','combine就绪槽数')
METRIC_OPERATIONS = {
    'dispatch':tuple(range(22))+tuple(range(34,39)),
    'combine':tuple(range(21,34))+(39,),
}


def definitions(operation):
    result=[dict(row) for row in PHASES[operation]]
    for index in METRIC_OPERATIONS[operation]:
        counter=index in COUNTER_INDICES
        note=('调用点累计计数，不是耗时。' if counter else
              '所有实际调用的同核GetSystemCycle差值累计；含原依赖及读钟开销，与父包络/其他细分不能直接相加。')
        result.append(dict(stage=METRIC_NAMES[index],label=METRIC_LABELS[index],
                           kind='counter' if counter else 'metric',metric_index=index,
                           legacy_stage_id=METRIC_INDICES[index],note=note))
    return result


def parse_records(raw, operation, frequency_hz=None):
    if operation not in PHASES:raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (isinstance(frequency_hz,bool) or not isinstance(frequency_hz,(int,float)) or
            not math.isfinite(frequency_hz) or frequency_hz<=0):
        raise ValueError('Counter frequency must be finite and positive')
    if len(raw)!=56:raise ValueError('SystemCycle capture requires 56 AIV')
    op=1 if operation=='dispatch' else 2
    records=[];intervals=[];milestones=[];cores=[];metrics=[];splits=set()
    for core,rows in enumerate(raw):
        if len(rows)!=16 or any(len(row)!=8 for row in rows):raise ValueError('SystemCycle carrier shape mismatch')
        words=[word for row in rows for word in row]
        if any(type(word) is not int or not -(1<<63)<=word<(1<<64) for word in words):
            raise ValueError('Invalid SystemCycle word')
        words=[word&((1<<64)-1) for word in words]
        h=words[:8];mode,send,role,profile=(h[5]>>shift&255 for shift in (0,8,16,24))
        if h[:5]!=[MAGIC,18,op,56,core] or mode!=1 or profile!=9 or h[5]>>32 or not 0<send<56:
            raise ValueError('Incomplete or mismatched SystemCycle header')
        expected=(3 if core<send else 2) if op==1 else (1 if core<send else 4)
        if role!=expected:raise ValueError('Unexpected SystemCycle core role')
        if h[6]>>40 or ((h[7]>>17)&127):raise ValueError('Invalid SystemCycle valid/reserved bits')
        send_lanes,recv_lanes=h[7]&255,(h[7]>>8)&255
        overlap=bool(h[7]&(1<<16));metric_valid=h[7]>>24
        if max(send_lanes,recv_lanes)>(6 if overlap else 8):raise ValueError('Invalid SystemCycle lane metadata')
        points={}
        for index,name in enumerate(POINTS):
            if h[6]&(1<<index):points[name]=words[8+index]
            elif words[8+index]:raise ValueError('Timestamp without valid point bit')
        if not {'ENTRY','INIT_DONE','WORK_DONE'}<=points.keys():raise ValueError('Missing required SystemCycle points')
        first,last=points['ENTRY'],points['WORK_DONE']
        if not first<=points['INIT_DONE']<=last or any(not first<=value<=last for value in points.values()):
            raise ValueError('SystemCycle point outside same-core work interval')
        for begin,end in (('COUNTS_BEGIN','COUNTS_SENT'),('COUNTS_WAIT_BEGIN','COUNTS_RECEIVED'),
                          ('SOURCE_BEGIN','SOURCE_RETURN'),('PREFIX_BEGIN','PREFIX_RETURN'),
                          ('COMPLETE_PREFIX_BEGIN','COMPLETE_PREFIX_RETURN'),
                          ('PREFIX_WAIT_ENTER','PREFIX_WAIT_EXIT'),('RECV_BEGIN','RECV_TASKS_DONE'),
                          ('RETIRE_BEGIN','RETIRE_END'),('COMBINE_PREPARE_BEGIN','COMBINE_PREPARE_END'),
                          ('COMBINE_ROLE_BEGIN','COMBINE_ROLE_END')):
            if end in points and begin not in points:
                raise ValueError('Missing SystemCycle boundary prerequisite: '+begin)
        if op==2 and set(points)-{'ENTRY','INIT_DONE','WORK_DONE','ENTRY_SETUP_DONE','MAGIC_DONE',
                'COMBINE_PREPARE_BEGIN','COMBINE_PREPARE_END','COMBINE_ROLE_BEGIN','COMBINE_ROLE_END'}:
            raise ValueError('Dispatch-only point in combine capture')
        common=dict(core=core,role=ROLES[role],initial_role=ROLES[role],timer_profile='systemcycle',timer_profile_id=9)
        milestones.extend(dict(**common,point=name,tick=tick) for name,tick in points.items())
        for definition in PHASES[operation]:
            begin_name,end_name=definition['start_point'],definition['end_point']
            if begin_name not in points or end_name not in points:continue
            begin,end=points[begin_name],points[end_name]
            if end<begin:raise ValueError('Invalid SystemCycle phase order: '+definition['stage'])
            total=end-begin
            row=dict(**common,**definition,total_ticks=total,calls=1,average_ticks=total,time_measured=True)
            if frequency_hz is not None:row.update(total_us=total*1e6/frequency_hz,average_us=total*1e6/frequency_hz)
            records.append(row)
            intervals.append(dict(**common,stage=definition['stage'],sample=0,first_tick=begin,last_tick=end,
                                  total_ticks=total,start_point=begin_name,end_point=end_name,
                                  interval_kind='same-core milestone span; may contain other spans'))
        for index,name in enumerate(METRIC_NAMES):
            total,calls=words[48+index*2:50+index*2]
            if not metric_valid&(1<<index):
                if total or calls:raise ValueError('Metric words without valid bit')
                continue
            counter=index in COUNTER_INDICES
            if counter and total:raise ValueError('Counter metric contains elapsed ticks')
            if not counter and (not calls or total>last-first):
                raise ValueError('Invalid SystemCycle cumulative metric time/calls')
            if index not in METRIC_OPERATIONS[operation]:
                raise ValueError('SystemCycle metric belongs to the other operation')
            row=dict(**common,stage=name,label=METRIC_LABELS[index],kind='counter' if counter else 'metric',
                     metric_index=index,legacy_stage_id=METRIC_INDICES[index],total_ticks=total,calls=calls,
                     average_ticks=total/calls if calls and not counter else None,time_measured=not counter,
                     value=calls if counter else total)
            if frequency_hz is not None and not counter:
                row.update(total_us=total*1e6/frequency_hz,average_us=total*1e6/frequency_hz/calls)
            records.append(row);metrics.append(row)
        cores.append(dict(**common,valid_bitmap=h[6],metric_valid_bitmap=metric_valid,milestone_count=len(points),
                          metric_count=metric_valid.bit_count(),send_lanes=send_lanes,recv_lanes=recv_lanes,
                          overlap_enabled=overlap))
        splits.add(send)
    if len(splits)!=1:raise ValueError('Mixed SystemCycle core splits')
    return dict(schema=18,operation=operation,active_aiv=56,send_cores=send,recv_cores=56-send,
                boundary_mode='observe',timer_profile='systemcycle',timer_profile_id=9,frequency_hz=frequency_hz,
                recording='systemcycle-stages',points=list(POINTS),records=records,intervals=intervals,
                milestones=milestones,cores=cores,cycle_metrics=metrics,
                clock='AscendC::GetSystemCycle; rank/core-local differences only',
                interpretation='Main phases are same-core milestone spans. Selected repeated scopes accumulate elapsed ticks and calls, with no added barriers. Parent/child scopes overlap and cannot be summed. Issue spans do not prove transfer completion. Missing fields stay absent, valid zero stays zero. All diagnostic GM export follows WORK_DONE; actual full task includes export.')
