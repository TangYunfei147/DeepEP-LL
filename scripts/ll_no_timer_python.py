"""Deterministic Python projection for the fixed-source LL timer-cost experiment.

Only the explicitly listed files are transformed. The caller validates their
Git preimages and removes DELETE_PATHS for clean-source; this module never writes
files or imports a framework. Normal and clean projections use the same loader
identity check, outside communication and profiling.
"""
from __future__ import annotations

import ast


PATCH_PATHS = TRANSFORM_PATHS = (
    "deep_ep/_native.py",
    "deep_ep/buffers/elastic.py",
    "tests/elastic/test_ep_low_latency.py",
    "tests/ll_stage_timer/run.py",
)
DELETE_PATHS = (
    "deep_ep/utils/ll_stage_timer.py",
    "deep_ep/utils/ll_stage_timer_sparse.py",
    "deep_ep/utils/ll_stage_timer_legacy.py",
    "deep_ep/utils/ll_expert_timer.py",
)
MODES = {"normal-off": (152, 96), "clean-source": (136, 88)}


def _replace(text: str, old: str, new: str, count: int = 1) -> str:
    found = text.count(old)
    if found != count:
        raise ValueError(f"Python projection preimage differs: expected {count}, found {found}: {old[:100]!r}")
    return text.replace(old, new)


def _remove_functions(text: str, names: tuple[str, ...]) -> str:
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Python projection diagnostic function preimage differs")
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    return "".join(lines)


def _remove_statements(text: str, function: str, selector, count: int) -> str:
    tree = ast.parse(text)
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    nodes = [node for node in ast.walk(owner) if isinstance(node, ast.stmt) and selector(node)]
    if len(nodes) != count:
        raise ValueError(f"Python projection {function} statement preimage differs: {len(nodes)} != {count}")
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    return "".join(lines)


def _has_diagnostic_attribute(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Attribute) and item.attr in {
        "ll_stage_timer", "ll_expert_timer", "ll_timer_mode", "ll_timer_iters",
        "ll_timer_frequency_hz", "ll_profile_capture",
    } for item in ast.walk(node))


def _identity_source(mode: str, *, loader: bool) -> str:
    dispatch, combine = MODES[mode]
    if loader:
        header = "def _validate_ll_source_identity(module):"
        package = "Path(__file__).resolve().parent"
        native = "module"
        error = "ImportError"
    else:
        header = "def installed_instrumentation_identity(package, native):"
        package = "Path(package.__file__).resolve().parent"
        native = "native"
        error = "RuntimeError"
    tail = "" if loader else "\n    return dict(instrumentation_source=actual[0], dispatch_launch_bytes=actual[1], combine_launch_bytes=actual[2])"
    return f'''{header}
    """Reject mixed source/package/native launch layouts before communication."""
    expected = ({mode!r}, {dispatch}, {combine})
    try:
        info = json.loads(({package} / '_build_info.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise {error}('Missing LL instrumentation build identity; install the matching projected wheel') from exc
    actual = (getattr({native}, 'low_latency_instrumentation_source', None),
              getattr({native}, 'dispatch_launch_bytes', None),
              getattr({native}, 'combine_launch_bytes', None))
    recorded = (info.get('ll_instrumentation_source'), info.get('dispatch_launch_bytes'), info.get('combine_launch_bytes'))
    if (actual != expected or recorded != expected
            or any(type(value) is not int for value in actual[1:] + recorded[1:])):
        raise {error}('LL instrumentation source/launch ABI differs between source, package and native library')
{tail}

'''


def _native(text: str, mode: str) -> str:
    text = _replace(text, "import importlib\n", "import importlib\nimport json\n")
    text = _replace(text, "@lru_cache(maxsize=1)\ndef _load_native():", _identity_source(mode, loader=True) + "@lru_cache(maxsize=1)\ndef _load_native():")
    return _replace(text, '        raise ImportError("Unsupported DeepEP runtime protocol")\n    return module',
                    '        raise ImportError("Unsupported DeepEP runtime protocol")\n    _validate_ll_source_identity(module)\n    return module')


def _elastic(text: str) -> str:
    for operation, indent in (("dispatch", " " * 20), ("combine", " " * 16)):
        text = _replace(text, f'{indent}from ..utils.ll_stage_timer import attach\n{indent}attach(self, tensors, "{operation}")\n', "")
    text = _replace(text,
        'retained += tuple(value for key, value in tensors.items() if value is not None and key != "ll_stage_timer_mode")',
        'retained += tuple(value for value in tensors.values() if value is not None)')
    return _replace(text,
        'retained = tuple(value for key, value in tensors.items() if key != "ll_stage_timer_mode")',
        'retained = tuple(tensors.values())')


def _st(text: str, mode: str) -> str:
    text = _replace(text, "def installed_source_fingerprint(package):", _identity_source(mode, loader=False) + "def installed_source_fingerprint(package):")
    text = _replace(text, '                      "source_sha256": installed_source_fingerprint(deep_ep),',
                    '                      "source_sha256": installed_source_fingerprint(deep_ep),\n                      **installed_instrumentation_identity(deep_ep, native),')
    # JSON comparison already carries the full runtime and actual route identity;
    # include the same identities in the convenient CSV exports as well.
    text = _replace(text, "'core_protocol','source_revision')", "'core_protocol','source_revision','source_sha256','instrumentation_source','dispatch_launch_bytes','combine_launch_bytes','profile_seed','profile_route_sha256')")
    if mode == "normal-off":
        return text

    text = _remove_functions(text, (
        "validate_stage_timer_capability", "validate_sparse_timer_layout", "new_stage_trace",
        "active_capture_profiler", "read_active_capture_profile", "summarize_active_capture_profiles", "diagnostic_calls",
    ))
    text = _replace(text, "from contextlib import nullcontext\n", "")
    text = _remove_statements(text, "make_parser", lambda node: (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_argument"
        and any(isinstance(item, ast.Constant) and isinstance(item.value, str)
                and item.value.startswith(("--ll-timer-", "--ll-stage-timer", "--ll-expert-timer", "--ll-profile-capture"))
                for item in node.value.args)), 6)
    text = _remove_statements(text, "parse_args", lambda node: isinstance(node, ast.If) and _has_diagnostic_attribute(node.test), 1)
    text = _remove_statements(text, "validate_args", lambda node: isinstance(node, ast.If) and _has_diagnostic_attribute(node.test), 7)
    text = _replace(text,
        "                timer_mode=args.ll_timer_mode if args.ll_stage_timer else None,\n"
        "                timer_iters=args.ll_timer_iters if args.ll_stage_timer else 0,\n"
        "                frequency_hz=args.ll_timer_frequency_hz, stage_capture=args.ll_stage_timer,\n"
        "                profile_capture=bool(getattr(args,'ll_profile_capture',False)),\n"
        "                expert_capture=args.ll_expert_timer, do_handle_copy=args.do_handle_copy,",
        "                do_handle_copy=args.do_handle_copy,")
    tree = ast.parse(text)
    old_sample = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample_operation")
    old_source = ast.get_source_segment(text, old_sample)
    text = _replace(text, old_source, '''def sample_operation(buffer, invoke, check, args, operation, rank, m, mode, progress):
    """Run formal OFF profiling; this source has no diagnostic entry points."""
    if args.skip_perf_test:
        return None
    profile = Path(args.run_directory) / f'case_{args.case_id:05d}' / mode / operation / f'rank_{rank:03d}'
    return measure(invoke, args, operation, profile, progress, after_profile=None)''')
    text = _replace(text, "    if not args.skip_perf_test or args.ll_stage_timer:", "    if not args.skip_perf_test:")
    text = _replace(text, "        timer_layout = native.low_latency_stage_timer_layout()\n", "")
    text = _replace(text,
        '                      "core_protocol": native.low_latency_core_protocol,\n'
        '                      "stage_timer": bool(getattr(native, "low_latency_stage_timer_enabled", False)),\n'
        '                      "profile_capture_requested": bool(getattr(args, "ll_profile_capture", False)),\n'
        '                      "stage_timer_schema": timer_layout["schema"],\n'
        '                      "stage_timer_slots": timer_layout.get("slots"),\n'
        '                      "stage_timer_fields": timer_layout.get("fields"),\n'
        '                      "stage_timer_points": timer_layout.get("points"),\n'
        '                      "stage_timer_recording": timer_layout.get("recording"),\n'
        '                      "timer_profile": getattr(native,"low_latency_stage_timer_profile",None),\n'
        '                      "timer_profile_id": getattr(native,"low_latency_stage_timer_profile_id",None),\n'
        '                      "expert_timer_requested": bool(args.ll_expert_timer)}',
        '                      "core_protocol": native.low_latency_core_protocol}')
    text = _remove_statements(text, "worker", lambda node: isinstance(node, ast.If) and _has_diagnostic_attribute(node.test), 2)
    text = _replace(text, "        args.ll_stage_timer_capabilities = capabilities", "        args.runtime_capabilities = capabilities")
    return _replace(text, "LL AIV/group/protocol/timer/profile/source build differs across ranks", "LL AIV/group/protocol/source/launch ABI build differs across ranks")


def transform(relative: str, text: str, mode: str) -> str:
    """Project one allowlisted fixed-preimage file, failing on unknown edits."""
    if mode not in MODES:
        raise ValueError(f"Unknown Python projection mode: {mode}")
    relative = relative.replace("\\", "/")
    if relative not in PATCH_PATHS:
        raise ValueError(f"File is outside Python projection: {relative}")
    newline = "\r\n" if "\r\n" in text else "\n"
    text = text.replace("\r\n", "\n")
    if relative == "deep_ep/_native.py":
        result = _native(text, mode)
    elif relative == "deep_ep/buffers/elastic.py":
        result = _elastic(text) if mode == "clean-source" else text
    elif relative == "tests/ll_stage_timer/run.py":
        result = _replace(text, "    libraries=[str(ROOT/'third-party/shmem/install/shmem/lib')]",
            "    sdk = Path(env.get('SHMEM_ROOT', ROOT/'third-party/shmem/install/shmem'))\n"
            "    if not sdk.is_absolute():\n"
            "        sdk = ROOT / sdk\n"
            "    libraries=[str(sdk/'lib')]")
    else:
        result = _st(text, mode)
    ast.parse(result, filename=relative)
    return result.replace("\n", newline)


transform_python = transform
