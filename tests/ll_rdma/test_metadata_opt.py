import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("metadata_opt", Path(__file__).with_name("metadata_opt.py"))
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


@pytest.mark.parametrize("method,value", [("scalar", "OFF"), ("histogram", "ON")])
def test_fixed_m_and_explicit_method(method, value):
    original = {"DEEPEP_LL_METADATA_CORES": "8", "DEEPEP_LL_COUNT_HISTOGRAM": "stale"}
    options = ["0", "192.0.2.1", "trial-001", "--mode", "perf", "--seed", "2"]
    command, env = opt.experiment([method, *options], original)
    assert env["DEEPEP_LL_METADATA_CORES"] == "4"
    assert env["DEEPEP_LL_COUNT_HISTOGRAM"] == value
    assert original["DEEPEP_LL_METADATA_CORES"] == "8"
    assert command == ["bash", "tests/ll_rdma/one_click.sh", *options]


def test_plan_avoids_vendor_shell_and_skip_build_is_rejected():
    options = ["0", "192.0.2.1", "trial-001"]
    command, _ = opt.experiment(["histogram", *options, "--plan"], {})
    assert command[1] == "tests/ll_rdma/one_click.py"
    with pytest.raises(SystemExit):
        opt.experiment(["histogram", *options, "--skip-build"], {})
