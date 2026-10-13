"""Project-relative paths, independent of the checkout name and current directory.

External CANN locations may be provided explicitly through the environment.
Relative overrides and command-line paths are always relative to REPO_ROOT.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def workspace_root():
    return project_path(os.environ.get('PROBE_ROOT', '.'))


def output_path(value):
    path = project_path(value)
    root = workspace_root()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Output must be below PROBE_ROOT (default: repository root)')
    return path


def results_dir():
    path = workspace_root() / 'results'
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_environment():
    """Call before importing device runtimes; do not change visibility policy."""
    root = workspace_root()
    os.environ['PROBE_ROOT'] = str(root)
    os.environ['DEEPEP_PROJECT_ROOT'] = str(REPO_ROOT)
    os.environ.pop('SHMEM_ROOT', None)
    os.environ.pop('CATLASS_ROOT', None)
    os.environ['SHMEM_LIB_DIR'] = str(REPO_ROOT / 'third-party/shmem/install/shmem/lib')
    folders = {
        'TMPDIR': '.tmp', 'TMP': '.tmp', 'TEMP': '.tmp', 'XDG_CACHE_HOME': '.cache',
        'CUDA_CACHE_PATH': '.cache/cuda', 'TORCH_EXTENSIONS_DIR': '.cache/torch_extensions',
        'TRITON_CACHE_DIR': '.cache/triton', 'PIP_CACHE_DIR': '.cache/pip', 'CCACHE_DIR': '.cache/ccache',
        'ASCEND_CACHE_PATH': '.cache/ascend', 'SHMEM_LOG_PATH': '.logs/shmem',
        'ASCEND_WORK_PATH': '.logs/ascend', 'ASCEND_PROCESS_LOG_PATH': '.logs/ascend',
    }
    for key, relative in folders.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return root
