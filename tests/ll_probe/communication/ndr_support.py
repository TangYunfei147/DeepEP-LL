"""Build/runtime checks for the in-repository NDR SDK; never load device runtimes."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sdk_backend(root=ROOT):
    sdk=root/'third-party/shmem'
    cache=sdk/'build/CMakeCache.txt'
    if not cache.is_file():return None
    values={line.split('=',1)[0].split(':')[0]:line.split('=',1)[1] for line in cache.read_text().splitlines()
            if '=' in line and not line.startswith(('#','//'))}
    if values.get('ACLSHMEM_RDMA_SUPPORT') not in ('ON','1','TRUE'):return None
    backend=values.get('ACLSHMEM_RDMA_BACKEND','').upper()
    if backend not in ('XSCALE','HNS_1825'):raise ValueError('NDR requires XSCALE or HNS_1825 SDK backend')
    manifest=json.loads((sdk/'install/shmem/deepep-sdk.json').read_text())
    if manifest.get('build',{}).get('rdma_backend')!=backend:
        raise ValueError('SDK backend not recorded: rebuild with scripts/build_shmem.py --rebuild --rdma-backend ...')
    return backend

def identity(backend):
    sdk=ROOT/'third-party/shmem/install/shmem'
    manifest=json.loads((sdk/'deepep-sdk.json').read_text())
    files=[HERE/'kernel/comm_kernel.cpp',HERE/'host/comm_host.cpp',HERE/'tiling.h',HERE/'ndr_build.py',HERE/'ndr_support.py']
    return dict(backend=backend,sdk_revision=manifest['revision'],sdk_build=manifest['build'],
                sdk_library_sha256=digest(sdk/'lib/libshmem.so'),
                source_sha256={str(f.relative_to(ROOT)):digest(f) for f in files})

def verify_runtime():
    backend=sdk_backend()
    if backend is None:raise RuntimeError('NDR unavailable: installed SHMEM has RDMA disabled; build the actual NIC backend explicitly')
    manifest=HERE/'build/ndr-build.json'
    if not manifest.is_file():raise RuntimeError('NDR probe not built; run profiling/build.sh after building RDMA SDK')
    built=json.loads(manifest.read_text());current=identity(backend)
    if built['identity']!=current:raise RuntimeError('NDR source/SDK identity changed; rebuild probe before running')
    if digest(HERE/'build/libcomm_ndr_profile.so')!=built['library_sha256']:raise RuntimeError('NDR probe binary changed; rebuild')
    if not os.environ.get('IBV_EXTEND_DRIVERS'):
        raise RuntimeError('Set IBV_EXTEND_DRIVERS to the actual installed NIC provider library before NDR execution')
    return built
