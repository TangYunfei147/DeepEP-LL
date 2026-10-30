"""Host-only validation of the frozen reference wheel; no device query or SDK gate."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def source_fingerprint(root):
    text=(root/'build_support.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='source_fingerprint')
    namespace=dict(ROOT=root,Path=Path,hashlib=hashlib)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'build_support.py'),'exec'),namespace)
    return namespace['source_fingerprint']()


def verify():
    import deep_ep
    from deep_ep._native import require_native
    package=Path(deep_ep.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Reference wheel must belong to this run venv')
    native=require_native('reference Host-only verification')
    info=json.loads((package/'_build_info.json').read_text(encoding='utf-8'))
    if (getattr(native,'low_latency_reference_profile_enabled',None) is not True or
        native.low_latency_reference_profile_layout()['schema']!=7 or
        info.get('ll_reference_profile') is not True or info.get('ll_reference_profile_schema')!=7 or
        info.get('ll_instrumentation_source')!='reference-mte7' or
        info.get('dispatch_launch_bytes')!=144 or info.get('combine_launch_bytes')!=88 or
        info.get('source_sha256')!=source_fingerprint(ROOT)):
        raise RuntimeError('Frozen source / installed reference wheel identity differs; rebuild a new group')
    for env_name,info_name in (('DEEPEP_LL_DISPATCH_IMPL','ll_dispatch_impl'),('DEEPEP_LL_SEND_CORES','ll_send_cores'),
                               ('DEEPEP_LL_RECV_CORES','ll_recv_cores'),('DEEPEP_LL_SMALL_B_PREFIX_CORES','ll_small_b_prefix_cores')):
        if env_name in os.environ and str(info.get(info_name))!=os.environ[env_name]:
            raise RuntimeError('Requested build setting differs from installed wheel: '+env_name)
    if getattr(native,'low_latency_dispatch_impl',None)!=info.get('ll_dispatch_impl'):
        raise RuntimeError('Installed Python/native algorithm identity differs')
    return dict(host_only=True,device_queried=False,schema=7,source_sha256=info['source_sha256'],
                dispatch_impl=info['ll_dispatch_impl'],send_cores=info['ll_send_cores'],recv_cores=info['ll_recv_cores'],
                small_b_prefix_cores=info['ll_small_b_prefix_cores'])


if __name__=='__main__':
    print(json.dumps(verify(),sort_keys=True))
