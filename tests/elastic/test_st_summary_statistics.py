"""Summary statistics pool per-rank stable tails, never rank percentiles or timer data."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("st_summary", Path(__file__).with_name("test_ep_low_latency.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


@pytest.mark.parametrize("tail,mean,p50,p90,p99,count", [
    (2, 4.5, 4, 6.8, 7.88, 4),
    (0, 119 / 6, 4, 54, 95.4, 6),
])
def test_summary_csv_contains_pooled_tail_statistics(tmp_path, tail, mean, p50, p90, p99, count):
    results = []
    for case_id, factor in enumerate((1, 10)):
        result = dict(case_id=case_id, config=dict(hidden=3584 * factor), route="normal")
        for operation, multiplier in (("dispatch", 1), ("combine", 2)):
            scale = factor * multiplier
            result[operation] = st.summarize_kernel(
                [[100 * scale, 2 * scale, 4 * scale], [scale, 4 * scale, 8 * scale]], operation, tail)
        results.append(result)
    st.write_reports(tmp_path, {"results": results})
    with (tmp_path / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        factor = (1, 10)[int(row["case_id"])] * (1 if row["operation"] == "dispatch" else 2)
        for key, expected in (("mean", mean), ("p50", p50), ("p90", p90), ("p99", p99)):
            assert float(row[f"{key}_kernel_us"]) == pytest.approx(expected * factor)
        assert int(row["kernel_samples"]) == count
        assert int(row["perf_tail"]) == tail
        assert row["summary_statistics_scope"] == "pooled_rank_tail_kernel_samples"
        assert "max_rank_mean_kernel_us" in row
    # Raw samples remain complete even when only the tail enters statistics.
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["results"][0]["dispatch"]["ranks"][0]["kernel_us"] == [100, 2, 4]
    assert results[0]["dispatch"]["ranks"][0]["p90_us"] >= results[0]["dispatch"]["ranks"][0]["p50_us"]


def test_single_sample_and_case_result_summary():
    perf = st.summarize_kernel([[3]], "dispatch", 50)
    for name in ("mean", "p50", "p90", "p99"):
        assert perf[f"{name}_kernel_us"] == 3
    compact = st.case_result_summary({"dispatch": perf})
    assert compact["dispatch"]["p99_kernel_us"] == 3
    assert "kernel_us" not in compact["dispatch"]["ranks"][0]
