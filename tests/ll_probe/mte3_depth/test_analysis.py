"""CPU regression checks; all temporary data stays under the project .tmp."""
import csv
import tempfile
import unittest
from pathlib import Path

import analyze


ROOT = Path(__file__).resolve().parents[3]


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="mte3-analysis-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def rows(self, count=100):
        records = []
        for iteration in range(count):
            record = dict(zip(analyze.CASE, (0, "batch", 16, 128, 65536)))
            record.update(iteration=iteration, issue_ticks=iteration,
                          drain_ticks=2, total_ticks=iteration + 2,
                          event_us=iteration / 10, verified=1)
            records.append(record)
        return records

    def write(self, records):
        with (self.directory / "raw.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=analyze.REQUIRED)
            writer.writeheader()
            writer.writerows(records)

    def test_tail_fifty_statistics_are_per_metric(self):
        # Reverse file order: iteration, not CSV row position, defines the tail.
        self.write(list(reversed(self.rows())))
        summary, diagnostics = analyze.analyze(self.directory, 50)
        row = summary[0]
        self.assertEqual(row["samples_total"], 100)
        self.assertEqual(row["samples_used"], 50)
        self.assertAlmostEqual(row["issue_ticks_p50"], 74.5)
        self.assertAlmostEqual(row["issue_ticks_p90"], 94.1)
        self.assertAlmostEqual(row["issue_ticks_p99"], 98.51)
        self.assertAlmostEqual(row["event_us_p50"], 7.45)
        self.assertAlmostEqual(row["total_ticks_p50"], 76.5)
        self.assertTrue(diagnostics["all_verified"])

    def test_failed_verification_before_tail_is_retained(self):
        records = self.rows()
        records[0]["verified"] = 0
        self.write(records)
        _, diagnostics = analyze.analyze(self.directory, 50)
        self.assertFalse(diagnostics["all_verified"])
        self.assertEqual(len(diagnostics["failed_rows"]), 1)

    def test_duplicate_iteration_is_rejected(self):
        records = self.rows()
        self.write(records + [records[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate iterations"):
            analyze.analyze(self.directory, 50)

    def test_invalid_timings_are_rejected(self):
        for metric in analyze.METRICS:
            for invalid in (-1, float("nan"), float("inf"), float("-inf")):
                with self.subTest(metric=metric, value=invalid):
                    records = self.rows(1)
                    records[0][metric] = invalid
                    self.write(records)
                    with self.assertRaisesRegex(ValueError, "invalid timing"):
                        analyze.analyze(self.directory, 50)


if __name__ == "__main__":
    unittest.main()
