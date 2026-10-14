"""Unified, per-host probe launcher. Defaults to a CPU-only plan; never SSHs."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
from deployment import load_config, make_plan, TESTS
from paths import REPO_ROOT, runtime_environment

HERE=Path(__file__).resolve().parent

def forwarded(test, tokens):
    profile={'bs','hidden-size','element-bytes','hardware-config','bytes','cores','lanes','chunk','scenario','fanout','mode','batch','hidden','top-k','experts','partitions','warmup','samples','tail','delay-rows','delay-cols'}
    native={'num-tokens','num-max-tokens-per-rank','hidden','num-topk','num-experts','dtype','fp8-scale-dtype','do-cpu-sync','seed','warmup','num-iters','perf-tail','delay-rows','delay-cols','dispatch-only','repeat-combine','skip-perf-test','test-first-only'}
    allowed={'peer','no-huge-pages'} if test=='network' else native if test=='native' else profile
    for token in tokens:
        if token.startswith('-') and not token.lstrip('-').replace('.','',1).isdigit():
            name=token.split('=',1)[0]
            if not name.startswith('--') or name[2:] not in allowed:
                raise ValueError('Forwarded option is not allowed: '+name)
    return tokens

def command_for(job, extra):
    node=Path(job['output']); test=job['test']; python=sys.executable
    torchrun=[python,'-m','torch.distributed.run','--nnodes',str(len(job['ips'])),
              '--nproc-per-node',str(job['local_world_size']),'--node-rank',str(job['node_index']),
              '--master-addr',job['master_addr'],'--master-port',str(job['master_port']), '--max-restarts','0']
    if test=='network':
        p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
        p.add_argument('--peer',type=int,default=1)
        p.add_argument('--no-huge-pages',action='store_true')
        parsed=p.parse_args(extra)
        if not 0 < parsed.peer < job['world']: raise ValueError('peer must be a valid nonzero global rank')
        return [python,str(HERE/'network_probe/run.py'),'--config',str(node/'hosts.yaml'),
                '--node-index',str(job['node_index']),'--devices-per-host',str(job['local_world_size']),
                '--port',str(job['shmem_port']),'--timeout',str(job['timeout']),
                '--validation' if job['validation'] else '--target-run','--output',str(node/'data'),*extra]
    if test=='native':
        args=[str(REPO_ROOT/'tests/elastic/test_ep_low_latency.py'),'--num-processes',str(job['local_world_size']),
              '--num-nodes',str(len(job['ips'])),'--node-rank',str(job['node_index']),
              '--master-addr',job['master_addr'],'--master-port',str(job['master_port']),
              '--shmem-port',str(job['shmem_port']),'--output-dir',str(node/'data'),*extra]
        module=runpy.run_path(args[0])
        parsed=module['parse_args'](args[1:])
        module['validate_args'](parsed,job['world'])
    else:
        args=[str(HERE/'profiling/profile_probe.py'),'--kind','metadata' if test=='metadata' else 'comm',
              '--layer',str({'empty':0,'tiling':1}.get(test,3)), '--endpoint',job['endpoint'],
              '--output-dir',str(node/'data')]
        if job['validation']:
            args += ['--warmup','2','--samples','3','--tail','2','--delay-rows','64','--delay-cols','64']
        if job.get('hardware_config'):
            args += ['--hardware-config',job['hardware_config']]
        args += extra
        module=runpy.run_path(args[0]); parsed=module['parser']().parse_args(args[1:])
        module['plan'](parsed,dict(WORLD_SIZE=str(job['world']),RANK=str(job['ranks'][0]),LOCAL_RANK='0'))
        if job['validation'] and (parsed.delay_rows>64 or parsed.delay_cols>64 or
                parsed.samples>3 or parsed.warmup>2 or parsed.tail>parsed.samples):
            raise ValueError('Validation requires Exp dimensions <=64, samples<=3 and warmup<=2')
        if test!='metadata' and (not 0 < parsed.bytes <= 64*1024*1024 or parsed.bytes%32 or
                parsed.bytes*job['world']>256*1024*1024 or not 1<=parsed.cores<=32 or not 1<=parsed.lanes<=4):
            raise ValueError('comm requires aligned bytes in 32..64MiB, world*bytes<=256MiB, cores1..32, lanes1..4')
        if test=='metadata' and (not 1<=parsed.batch<=256 or not 1<=parsed.hidden<=65536 or
                not 1<=parsed.top_k<=min(32,parsed.experts) or not 1<=parsed.experts<=1024 or
                not 16<=parsed.partitions<=256 or parsed.experts%parsed.partitions):
            raise ValueError('Invalid metadata B/H/K/E/logical partitions; compiled G and UB checks remain in Host')
    return torchrun+args

def supervise(command, env, output, timeout):
    """Terminate the entire launched process group, including descendants."""
    with (output/'launcher.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,cwd=REPO_ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        finally:
            # Also clean descendants when the leader has exited unexpectedly.
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="完整中文参数说明：tests/ll_probe/RUN_ARGS.md。测试参数放在 -- 后；未指定执行模式时仅输出计划。")
    p.add_argument('--config',required=True,help='部署YAML路径，相对工程根解析')
    p.add_argument('--node-index','--node-rank',type=int,default=0,help='本机在ips列表中的索引，从0开始')
    p.add_argument('--devices-per-host',type=int,help='覆盖YAML的每机rank数，1..16；所有节点一致')
    p.add_argument('--test',choices=TESTS,required=True,help='选择测试目的；native仅支持EP16/32')
    p.add_argument('--run-id',help='各节点共享的全新运行标识，优先于YAML；执行时必需')
    p.add_argument('--plan',action='store_true',help='只打印CPU计划，优先于执行模式，不启动设备')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--target-run',action='store_true',help='用户目标机实际执行；按配置选卡，不用于A5')
    modes.add_argument('--validation',action='store_true',help='A5单机双rank0/1验证，使用小Exp及短采样')
    p.add_argument('extra',nargs=argparse.REMAINDER,help='-- 后的测试专用参数，完整列表见RUN_ARGS.md')
    args=p.parse_args(argv)
    extra=args.extra
    if extra and extra[0]!='--': p.error('Forwarded arguments require -- separator')
    extra=forwarded(args.test,extra[1:] if extra else [])
    config=load_config(args.config)
    execute=(args.target_run or args.validation) and not args.plan
    run_id=args.run_id or config.get('run_id')
    if execute and not run_id: raise ValueError('Execution requires the same explicit --run-id on every node')
    job=make_plan(config,args.node_index,args.devices_per_host,args.test,run_id or 'plan-only',args.validation)
    topology=dict(ips=job['ips'],rank_num_per_server=job['local_world_size'],world=job['world'],
                  master_addr=job['master_addr'],master_port=job['master_port'],shmem_port=job['shmem_port'])
    job['topology']=topology
    job['command']=command_for(job,extra) if job['status']=='supported' else None
    print(json.dumps(job,indent=2),flush=True)
    if not execute: return
    if job['status']!='supported': raise ValueError(job['reason'])
    if args.validation:
        for key in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
            if os.environ.get(key,'0,1')!='0,1': raise ValueError('Validation requires '+key+'=0,1')
    # Deliberately constrain all runtime outputs to this checkout.
    os.environ['PROBE_ROOT']=str(REPO_ROOT)
    runtime_environment()
    output=Path(job['output']).resolve()
    if output==REPO_ROOT or not output.is_relative_to(REPO_ROOT):
        raise ValueError('Final output must remain inside the project')
    output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','TORCHELASTIC_RUN_ID'):
        env.pop(key,None)
    visible=','.join(map(str,range(job['local_world_size'])))
    for key in ('PROBE_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'): env[key]=visible
    env.update(DEEPEP_PROBE_RUN_ID=job['run_id'],DEEPEP_PROBE_TOPOLOGY=json.dumps(topology,sort_keys=True),DEEPEP_PROBE_NODE_INDEX=str(job['node_index']))
    lib=str(REPO_ROOT/'third-party/shmem/install/shmem/lib')
    env['LD_LIBRARY_PATH']=lib+':'+env.get('LD_LIBRARY_PATH','')
    (output/'hosts.yaml').write_text(json.dumps(dict(ips=job['ips'],rank_num_per_server=job['local_world_size'],
        port=job['master_port'],shmem_port=job['shmem_port'],timeout=job['timeout']),indent=2)+'\n')
    job.update(hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    (output/'plan.json').write_text(json.dumps(job,indent=2)+'\n')
    started=time.time(); code=1; error=None
    try:
        command=['bash',str(HERE/'four_layers/run.sh'),*job['command']]
        code=supervise(command,env,output,job['timeout'])
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc: error=type(exc).__name__
    finally:
        (output/'launcher-report.json').write_text(json.dumps(dict(job=job,returncode=code,error=error,
            elapsed_seconds=time.time()-started,local_success=code==0 and error is None),indent=2)+'\n')
    if code or error: raise SystemExit(code or 1)

if __name__=='__main__': main()
