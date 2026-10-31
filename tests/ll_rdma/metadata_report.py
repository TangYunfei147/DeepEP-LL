"""Export matched, completed M=4 OFF runs to one <=60000-byte report; no Torch/NPU."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ("world_size", "case", "route", "seed", "routing", "route_sha256", "warmup",
          "num_iters", "perf_tail", "active_aiv", "sampling_protocol", "exp_prefix_repeats",
          "per_iteration_exp", "per_iteration_all_reduce", "group_backend", "do_handle_copy",
          "correctness_repeats", "repeat_combine", "fresh_buffer_per_case", "trace_ops",
          "num_bytes", "use_1g_huge_pages", "logical_group", "transport")


def render(project, runs):
    project = Path(project).resolve()
    lines = ["# M=4 metadata OFF comparison", "",
             "Times are us. mean/p99 pool the per-rank tail samples; max-rank-mean is a separate statistic.",
             "FP8 refers to dispatch; combine uses BF16. One run is not a repeated A/B performance conclusion.",
             "Full raw samples, per-rank statistics and logs remain on the target. No samples are filtered here.", ""]
    common = sdk = None
    for run in runs:
        if Path(run).name != run or run in (".", ".."):
            raise ValueError("Supply run directory names, not paths")
        directory = project / "results/ll-ndr" / run
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        identity = json.loads((project / ".logs/ndr-one-click" / run / "node-0/execute/identity.json").read_text(encoding="utf-8"))
        if (report["status"] != "completed" or report["result_count"] != len(report["results"])
                or not report["results"] or identity["metadata_cores"] != 4
                or identity["reference_profile"] or identity["mode"] != "perf"):
            raise ValueError(f"{run}: incomplete run or wrong M/timer mode")
        this_sdk = (identity["sdk_revision"], identity["sdk_build"])
        if sdk is not None and this_sdk != sdk:
            raise ValueError(f"{run}: SDK identity differs")
        sdk = this_sdk
        signature = []
        block = [f"## {run}", f"Source: {identity['git_revision']}; histogram={identity['count_histogram']}; protocol={identity['core_protocol']}",
                 f"Source fingerprint: {identity['source_sha256']}", f"Data: {directory}", "",
                 "| case | operation | mean | p99 | max-rank-mean | slowest rank | samples |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for result in report["results"]:
            meta = result["comparison"]
            if (meta["skip_check"] or meta["dispatch_only"] or not meta["perf_enabled"]
                    or meta["timer_mode"] is not None or meta["source_revision"] != identity["git_revision"]
                    or meta["sampling_protocol"] != "initial-reduce-exp100-batch-v3"
                    or not meta["route_sha256"]):
                raise ValueError(f"{run}: invalid OFF sampling/source identity")
            signature.append({key: meta[key] for key in COMMON})
            for op in ("dispatch", "combine"):
                value = result[op]
                if (value["summary_statistics_scope"] != "pooled_rank_tail_kernel_samples"
                        or len(value["ranks"]) != report["world_size"]):
                    raise ValueError(f"{run}: missing ranks or different statistics")
                block.append(f"| {result['case_id']} | {op} | {value['mean_kernel_us']:.3f} | {value['p99_kernel_us']:.3f} | "
                             f"{value['max_rank_mean_kernel_us']:.3f} | {value['slowest_rank']} | {value['kernel_samples']} |")
        if common is not None and signature != common:
            raise ValueError(f"{run}: shapes/routes/seeds/sampling/core configuration differ")
        if common is None:
            lines += ["SDK: " + json.dumps(sdk, ensure_ascii=False), "Matched cases: " + json.dumps(signature, ensure_ascii=False), ""]
        common = signature
        lines += block + [""]
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > 60000:
        raise ValueError("Report exceeds 60000 UTF-8 bytes; select fewer cases/runs. No output written.")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.project, args.runs)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(f"{args.output}: {len(text.encode('utf-8'))} UTF-8 bytes")


if __name__ == "__main__":
    main()
