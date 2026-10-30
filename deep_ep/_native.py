# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Lazy native loading. Importing deep_ep never imports a framework or driver."""

import importlib
import json
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from functools import lru_cache
from pathlib import Path
from typing import NoReturn


def _unavailable(operation: str) -> NoReturn:
    raise NotImplementedError(
        f"{operation} is not implemented in DeepEP. "
        "No communication operation or simulated result has been performed. "
        "See docs/api.md for implementation status."
    )


def require_native(operation):
    directory = Path(__file__).resolve().parent
    if not any((directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES):
        _unavailable(operation)
    return _load_native()


def _validate_ll_source_identity(module):
    """Reject mixed source/package/native launch layouts before communication."""
    expected = ('reference-mte7', 144, 88)
    directory = Path(__file__).resolve().parent
    for name in ('ll_stage_timer', 'll_stage_timer_sparse', 'll_stage_timer_legacy', 'll_expert_timer'):
        if any((directory / 'utils' / (name + suffix)).exists() for suffix in ('.py', '.pyc', '.pyo')):
            raise ImportError('Retired LL timer modules remain in this package; install a clean complete wheel')
    try:
        info = json.loads((directory / '_build_info.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ImportError('Missing LL instrumentation build identity; install the matching complete reference wheel') from exc
    if (type(info.get('ll_reference_profile')) is not bool or
        type(getattr(module, 'low_latency_reference_profile_enabled', None)) is not bool or
        info.get('ll_reference_profile') != getattr(module, 'low_latency_reference_profile_enabled', None) or
        info.get('ll_reference_profile_schema') != 7 or module.low_latency_reference_profile_layout().get('schema') != 7):
        raise ImportError('Reference profiling mode/schema differs between Python and native library')
    actual = (getattr(module, 'low_latency_instrumentation_source', None),
              getattr(module, 'dispatch_launch_bytes', None),
              getattr(module, 'combine_launch_bytes', None))
    recorded = (info.get('ll_instrumentation_source'), info.get('dispatch_launch_bytes'), info.get('combine_launch_bytes'))
    if (actual != expected or recorded != expected
            or any(type(value) is not int for value in actual[1:] + recorded[1:])):
        raise ImportError('LL instrumentation source/launch ABI differs between source, package and native library')


@lru_cache(maxsize=1)
def _load_native():
    spec = importlib.util.find_spec("deep_ep._C")
    if spec is None:
        raise ImportError("Native extension file exists but cannot be resolved")
    if not spec.origin or Path(spec.origin).resolve().parent != Path(__file__).resolve().parent:
        raise ImportError("Native extension must belong to this deep_ep installation")
    try:
        module = importlib.import_module("deep_ep._C")
    except ImportError as exc:
        raise ImportError("Cannot load deep_ep._C; configure the matching CANN/SHMEM SDK libraries") from exc
    if module.runtime_protocol != 1:
        raise ImportError("Unsupported DeepEP runtime protocol")
    _validate_ll_source_identity(module)
    return module


def npu_api():
    import torch
    import torch_npu  # noqa: F401

    return torch.npu
