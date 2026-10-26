# Read-only inspection of the reproducible small_bs-DeepEP deployment

The user located a reproducible independent deployment at
`/home/t30064812/small_bs-DeepEP`, with results under
`results/cbd04eb/10261635_18whtneo`, and confirmed the same CANN environment.
The assistant cannot connect to that machine. Run the following inspection on
the target from this current `kernel/ll-g4-small-b-prefix` checkout.

## Inspection command

```bash
inspect_reproducible_reference() {
  git pull --ff-only || return $?
  python tests/ll_stage_timer/inspect_reference.py \
    --reference-root /home/t30064812/small_bs-DeepEP \
    --results results/cbd04eb/10261635_18whtneo \
    --run-name cbd04eb_static_check_001
}
if inspect_reproducible_reference; then
  echo 'Return only diagnostic-runs/cbd04eb_static_check_001/SEND_FIRST.txt'
else
  echo 'Inspection stopped. Preserve the message; terminal remains open.'
fi
```

Use a fresh run name for a new inspection. The output directory belongs to the
current script checkout. The reference deployment and its performance results
are read-only inputs. Relative result paths resolve from `--reference-root`,
not from the shell's current directory.

The script uses Python's standard library and read-only Git commands. It does
not import Torch, torch_npu, deep_ep or inspected Python source; load a native
extension; query NPU devices; run a compiler or ldd; build/install anything;
change the reference checkout; initialize submodules; or start a benchmark.
Git optional locks and external filesystem-monitor/diff helpers are disabled.
No CANN environment activation is required for the script itself. Missing
optional evidence is reported as missing, without installing or rebuilding it.

Return only the bounded `SEND_FIRST.txt` (at most60,000 UTF-8 bytes). Detailed
evidence stays in the new output directory. The original performance files and
source are preserved. SDK revisions and existing records are observational
metadata, not SDK admission checks or instructions to rebuild.

## What it checks

- Real Git HEAD, branch and local changes, submodule metadata, and source bytes
  against available `cbd04eb` and `cb59bff` references. The result directory's
  name is not treated as proof of the code actually used.
- Static measurement-function identity and parser defaults, plus the actual
  result configuration: warmup, captured count, tail, Exp placement/count,
  alignment backend, dispatch-only versus combined tests, CPU sync, shape,
  seed, handles and buffer lifetime settings.
- Actual per-rank samples and reported statistics. Tail10/30/50/all statistics
  are recomputed only where samples exist; a50-sample run cannot supply100
  observed samples. Min/median/max rank means and rank0 remain distinct.
- Up to16 existing profiler CSVs, prioritizing rank0: task names/counts, recorded
  start-time completeness, first-to-second and later adjacent task intervals.
  These intervals are not kernel execution durations or measured CPU launch costs.
- Existing CMake caches and compiler command records under the reference
  build trees, original wheel metadata and static installed package files
  identified by the saved result. No compiler or package is executed.
- Existing loaded-library records from the run versus current static package
  evidence where available. A package file found today is not automatically
  claimed to be the file used by the saved historical run.

The report records missing evidence and limits explicitly. It does not collect
credentials, an unrestricted environment dump or unrelated filesystem trees.

## A concrete difference already established from Git

The newly identified `cbd04eb2c1cbc120c035a59ddf7cd55feb79e200` precedes
`cb59bff99ecd39d27ef57788c5e6fd0009e4e636`, which the previous reproduction
rebuilt. Their measurement functions differ substantially:

| Item | cbd04eb defaults / code | cb59bff defaults / code |
|---|---|---|
| Warmup / capture / tail | 5 / 50 / 30 | 10 / 100 / 50 |
| Initial alignment | 128-element async all-reduce, then wait | scalar alignment |
| Captured invocation pattern | first dispatch,1000 Exp calls,one all-reduce,remaining dispatches consecutively | one Exp and one scalar all-reduce before every dispatch |
| Tail invocation context | consecutive dispatch calls, no per-call Exp/all-reduce | Exp/all-reduce still precedes every tail call |
| Reported quantity | complete device kernel task, max of rank tail means | same quantity and aggregation |

These are source facts, not yet confirmation that the target used all defaults.
The inspection reads the target's actual config and exact `measure()` identity.
Changing the offline tail length alone cannot remove differences in the work
and synchronization preceding each kernel invocation.

The SDK gitlink and dependencies lock are unchanged between these two commits.
The OFF device build retains O2 and the monolithic mode entry. The timer commit
also expands dispatch launch arguments from136 to144 bytes, adds a timer pointer
member and changes the core protocol from1 to2. OFF timer method bodies are
compiled out; source inspection alone does not prove identical generated code
or assign a latency to the remaining signature/object changes.

The current evidence therefore does not justify attributing the historical
17–22us versus rebuilt22–27us difference to SDK upgrade, split mode TUs, or
timer code alone. First identify the exact deployment and sampling protocol.
This inspection changes neither protocol nor kernel and starts no new tests.

Local CPU checks passed103 cases, with2 native-compiler cases skipped. The
reference/result fixture bytes remained unchanged; no target or NPU validation
is claimed before the user runs this inspection.
