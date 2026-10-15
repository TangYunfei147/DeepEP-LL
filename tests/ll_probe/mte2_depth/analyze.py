#!/usr/bin/env python3
"""Summarize independent MTE2 diagnostic cases without inferring hardware depth."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

CASE = ("device", "mode", "requests", "chunk_bytes", "stride_bytes")
METRICS = ("issue_ticks", "drain_ticks", "total_ticks", "event_us")
REQUIRED = CASE + ("iteration",) + METRICS + ("verified",)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def resolve(root, path):
    path = Path(path)
    root = root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Input/output paths must remain inside the repository')
    return resolved


def analyze(input_dir, tail):
    groups = defaultdict(list)
    failures = []
    files = sorted(input_dir.rglob("*.csv"))
    for file in files:
        if file.name == "summary.csv":
            continue
        with file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not set(REQUIRED).issubset(reader.fieldnames or ()):
                raise ValueError(f"{file}: missing required columns")
            for line, row in enumerate(reader, 2):
                key = (int(row["device"]), row["mode"], int(row["requests"]),
                       int(row["chunk_bytes"]), int(row["stride_bytes"]))
                if key[1] not in ("batch", "serial"):
                    raise ValueError(f"{file}:{line}: unknown mode {key[1]}")
                values = {metric: float(row[metric]) for metric in METRICS}
                if any(not math.isfinite(v) or v < 0 for v in values.values()):
                    raise ValueError(f"{file}:{line}: invalid timing")
                if row["verified"].strip().lower() not in ("1", "true", "yes", "pass", "verified"):
                    failures.append(f"{file}:{line}")
                groups[key].append((int(row["iteration"]), values))
    if not groups:
        raise ValueError("No raw measurements found")
    summary = []
    for key, rows in sorted(groups.items()):
        iterations = [iteration for iteration, _ in rows]
        if len(set(iterations)) != len(iterations):
            raise ValueError(f"Duplicate iterations for case {key}; separate runs into input directories")
        rows.sort(key=lambda item: item[0])
        selected = rows[-tail:]
        result = dict(zip(CASE, key))
        result.update(samples_total=len(rows), samples_used=len(selected))
        for metric in METRICS:
            values = [row[metric] for _, row in selected]
            for label, fraction in (("p50", .5), ("p90", .9), ("p99", .99)):
                result[f"{metric}_{label}"] = percentile(values, fraction)
        summary.append(result)
    coverage = []
    configurations = sorted(set((k[0], k[3], k[4]) for k in groups))
    for device, chunk, stride in configurations:
        available = {(k[1], k[2]) for k in groups if (k[0], k[3], k[4]) == (device, chunk, stride)}
        missing = [f"{mode}:N{n}" for mode in ("batch", "serial") for n in (0, 1, 16, 17, 32)
                   if (mode, n) not in available]
        coverage.append(dict(device=device, chunk_bytes=chunk, stride_bytes=stride,
                             missing_recommended_cases=missing))
    slopes = []
    for device, chunk, stride in configurations:
        for mode in ("batch", "serial"):
            rows = sorted((r for r in summary if r["device"] == device and r["mode"] == mode
                           and r["chunk_bytes"] == chunk and r["stride_bytes"] == stride
                           and 12 <= r["requests"] <= 20), key=lambda r: r["requests"])
            for a, b in zip(rows, rows[1:]):
                delta = b["requests"] - a["requests"]
                slopes.append(dict(device=device, mode=mode, chunk_bytes=chunk, stride_bytes=stride,
                                   n_from=a["requests"], n_to=b["requests"],
                                   issue_ticks_p50_per_request=(b["issue_ticks_p50"] - a["issue_ticks_p50"]) / delta,
                                   total_ticks_p50_per_request=(b["total_ticks_p50"] - a["total_ticks_p50"]) / delta))
    return summary, dict(all_verified=not failures, failed_rows=failures, coverage=coverage,
                         local_slopes_12_20=slopes,
                         interpretation="Local slopes are diagnostic only; no exact hardware depth is inferred. Ticks and Event microseconds are independent measurements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail", type=int, default=50)
    args = parser.parse_args()
    if args.tail <= 0:
        parser.error("--tail must be positive")
    root = Path(__file__).resolve().parents[3]
    summary, diagnostics = analyze(resolve(root, args.input_dir), args.tail)
    output = resolve(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps(dict(cases=summary, diagnostics=diagnostics),
                                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0 if diagnostics["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
