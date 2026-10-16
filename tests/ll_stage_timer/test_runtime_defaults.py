"""CPU-only tests for page defaults; no torch/CANN import."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ll_backend", ROOT / "deep_ep/utils/ll_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


@pytest.mark.parametrize("requested", [None, "mte"])
@pytest.mark.parametrize("combine_only", [False, True])
def test_public_defaults_preserve_udma_pages(requested, combine_only):
    assert backend.elastic_backend_options(requested, combine_only=combine_only,
        huge_pages=None) == ("mte", True)


@pytest.mark.parametrize("pages", [None, False])
def test_explicit_rdma_uses_ordinary_pages(pages):
    assert backend.elastic_backend_options("rdma", combine_only=False,
        huge_pages=pages) == ("rdma", False)


def test_explicit_page_and_backend_errors():
    assert backend.elastic_backend_options("mte", combine_only=False, huge_pages=False) == ("mte", False)
    for pages, combine in [(True, False), (False, True)]:
        with pytest.raises(ValueError):
            backend.elastic_backend_options("rdma", combine_only=combine, huge_pages=pages)
    for invalid in ("", "auto", "RDMA", "roce", 1):
        with pytest.raises(ValueError):
            backend.elastic_backend_options(invalid, combine_only=False, huge_pages=None)
    with pytest.raises(ValueError):
        backend.elastic_backend_options(None, combine_only=False, huge_pages=1)
