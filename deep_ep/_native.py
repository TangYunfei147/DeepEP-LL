# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Lazy native loading. Importing deep_ep never imports a framework or driver."""

import importlib
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
    return module


def npu_api():
    import torch
    import torch_npu  # noqa: F401

    return torch.npu
