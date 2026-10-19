"""Run LL ST with a CPU group for object metadata only.

Tensor collectives, HCCL WORLD, kernel calls and sampling are unchanged.
The top-level setup also runs in multiprocessing.spawn children.
"""
import argparse
from datetime import timedelta
import hashlib
from pathlib import Path
import sys

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--test-script',type=Path,required=True)
args,forwarded=parser.parse_known_args()
script=args.test_script.resolve()
expected='a4d64dd8950c085661c1fdce5b0d44b5bbda86455ab2a83ae87658b7c066be17'
if hashlib.sha256(script.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('Unexpected LL ST source; re-audit object-only adapter before use')
sys.path.insert(0,str(script.parent))

import torch.distributed as dist

original_init=dist.init_process_group
original_broadcast=dist.broadcast_object_list
original_all_gather=dist.all_gather_object
cpu_group=None


def init_group(*positional,**kwargs):
    global cpu_group
    result=original_init(*positional,**kwargs)
    cpu_group=dist.new_group(backend='gloo',timeout=kwargs.get('timeout',timedelta(seconds=300)))
    return result


def metadata_group(group):
    if group is None or group is dist.group.WORLD:
        if cpu_group is None:
            raise RuntimeError('CPU metadata group has not been initialized')
        return cpu_group
    return group


def broadcast_objects(object_list,src=0,group=None,device=None,**kwargs):
    selected=metadata_group(group)
    return original_broadcast(object_list,src=src,group=selected,
                              device=None if selected is cpu_group else device,**kwargs)


def gather_objects(object_list,obj,group=None):
    return original_all_gather(object_list,obj,group=metadata_group(group))


dist.init_process_group=init_group
dist.broadcast_object_list=broadcast_objects
dist.all_gather_object=gather_objects

import test_ep_low_latency as test

if __name__=='__main__':
    print('LL ST adapter: object metadata uses Gloo; tensor collectives and sampling retain HCCL',flush=True)
    test.main(forwarded)
