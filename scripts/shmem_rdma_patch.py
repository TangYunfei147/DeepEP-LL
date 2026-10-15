"""Narrow, reproducible host-only patch for 72 independent ROCE QPs.

The locked SDK uses dynamic rank*QP arrays but shares UDMA's 56-QP policy
limit. Keep UDMA and public headers unchanged. No vendor files are committed;
the exact source delta is generated from the locked Git revision and recorded
by build_shmem.py. Refuse edits in the affected files before changing any.
"""
from pathlib import Path
import subprocess

PATCH_ID = "deepep-roce-72-v1"
REPLACEMENTS = {
    "src/host/init/shmem_init.cpp": (
        'if (!is_valid_qp_num(qp_num, ACLSHMEM_MAX_QP_NUM)) {\n'
        '                SHM_LOG_ERROR("invalid ROCE qp num: " << qp_num);',
        'if (!is_valid_qp_num(qp_num, 72U)) { // DeepEP: one ROCE QP per actual AIV\n'
        '                SHM_LOG_ERROR("invalid ROCE qp num: " << qp_num);',
    ),
    "src/host/transport/device_rdma/device_rdma_transport_manager_v2.cpp": (
        "if (qpNum_ == 0 || qpNum_ > ACLSHMEM_MAX_QP_NUM) {",
        "if (qpNum_ == 0 || qpNum_ > 72U) { // DeepEP ROCE-only capacity",
    ),
    "src/host/utils/exception/shmem_exception_rdma_dump.cpp": (
        "info.count > ACLSHMEM_MAX_QP_NUM",
        "info.count > 72U /* DeepEP ROCE-only capacity */",
    ),
}


def prepare(source, *, apply):
    source = Path(source)
    pending = []
    managed = set()
    for relative, (old, new) in REPLACEMENTS.items():
        baseline = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + relative]).decode()
        if baseline.count(old) != 1:
            raise RuntimeError(f"{PATCH_ID}: pinned source context changed: {relative}")
        expected = baseline.replace(old, new)
        path = source / relative
        current = path.read_text()
        if current not in (baseline, expected):
            raise RuntimeError(f"{PATCH_ID}: preserve user edits; affected file differs: {relative}")
        if current == expected:
            managed.add(relative)
        if apply and current != expected:
            pending.append((path, expected))
            managed.add(relative)
    for path, content in pending:
        path.write_text(content)
    changed = set(subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True).splitlines())
    untracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"]).strip()
    return changed == managed and not untracked
