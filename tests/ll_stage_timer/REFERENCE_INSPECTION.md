# Read-only inspection of the reproducible small_bs-DeepEP deployment

## Returned target evidence, 2026-10-26

The returned report confirms HEAD `cbd04eb2c1cbc120c035a59ddf7cd55feb79e200`,
no tracked changes in the inspected code/build/test scope, and exact cbd04eb
key-file and measurement-function identities. Its actual recorded settings are
5 warmups,50 samples,tail30,seed0,FP8,CPU sync0,EP16/M32/H3584/K8/E384,
`dispatch_only=false`, `delay_exp_before=0`, `delay_exp_after=1000`.
All14 operation/case results contain50 samples on each of16 ranks; recomputed
reported-tail maxima agree with the saved results.

The current installed `_C` and `libdeepep_kernels.so` hashes match every rank's
saved loaded-library record. The saved SDK load paths point into this reference
project, and the source/build metadata both identify SDK4dd80e2. CANN9.2.0
timestamp20260930_161523958, GNU12.2 and device-compiler hash3b243db6... match
the prior cb59 reproduction. This is stronger evidence than a matching version
label alone; it does not prove that every historical build option or runtime
condition was identical.

| B | cbd04eb recorded tail30, us | Same cbd04eb run, all50, us | Earlier cb59 seed0 two-round mean, us |
|---|---:|---:|---:|
| 1 | 17.5557 | 21.1838 | 22.7939 |
| 2 | 17.8661 | 21.1098 | 22.9257 |
| 4 | 18.2745 | 19.0711 | 23.1733 |
| 8 | 18.0286 | 18.8395 | 22.9735 |
| 16 | 19.3471 | 21.1490 | 23.4120 |
| 24 | 20.4447 | 23.3843 | 25.0585 |
| 32 | 22.5794 | 23.4973 | 26.9269 |

The last column used10/100/tail50 and dispatch-only. It is not an isolated
kernel comparison against the first column. Merely selecting all50 old samples
also includes the old first invocation; it does not recreate the newer100-call
per-iteration Exp/all-reduce protocol. For example, B1 combine's old all50
maximum is268.5127us while tail30 is12.0315us, with a large early block on one
rank. Keep this raw evidence rather than silently filtering individual outliers.

The primary verified mismatch is now the measurement protocol plus the
dispatch+combine versus dispatch-only call context. A causal test would keep
the exact currently installed fast wheel and SDK fixed and vary only measurement
protocol, using the same seed0 and dispatch+combine context on both sides.
An intermediate condition retaining consecutive calls but using10/100/tail50
would separate sample-window/warmup choices from per-call Exp/all-reduce.
No such NPU experiment is performed or implemented by this read-only inspector.

The build tree contains multiple cache fingerprints, different SDK include
paths and S40/R16/S28/R28 definitions. They are leftover build records, not
evidence that a particular cache produced the currently loaded library. The
cbd04eb source uses dynamic1:1 cores and does not use those later S/R knobs.
Likewise, reaching the8000-file inventory cap limits whole-profile-tree coverage,
but does not invalidate the complete root-summary sample reconstruction above.

The first report renderer abbreviated whole CSV/build JSON objects and let long
headers/include lists hide task intervals and final flags. Those values remain
in its local `evidence.json`; abbreviated fields are not missing measurements.
Subsequent rendering must preserve task numbers, record paths and optimization
option order before trimming optional header/include detail.

### Optional offline export of the already collected evidence

The configuration/library evidence above is sufficient to establish the test
conditions. Repeating the inspection or a device benchmark is unnecessary to
recover the omitted CSV/flag display. After updating this current branch, use
the existing evidence JSON if those details are needed:

```bash
reexport_reference_evidence() {
  git pull --ff-only || return $?
  python tests/ll_stage_timer/inspect_reference.py \
    --reexport-evidence diagnostic-runs/cbd04eb_static_check_001/evidence.json \
    --run-name cbd04eb_static_check_002
}
if reexport_reference_evidence; then
  echo 'Offline report: diagnostic-runs/cbd04eb_static_check_002/SEND_FIRST.txt'
else
  echo 'Export stopped. The original evidence and terminal remain available.'
fi
```

This reads only the saved JSON and the current export helper; it does not read
the reference deployment again or run Git/compilers/frameworks/devices. The
original JSON remains unchanged. The new report explicitly describes a saved
snapshot and consumes the original inspection group's remaining transfer budget;
an identical re-export does not consume another slot. Use a fresh output name.

New inspections preserve compiler flag duplicates and order. Older evidence
may have deduplicated flags, so its displayed order is marked unknown rather
than claimed to be a full effective compile command. CMake cache defaults,
device compile options and link-only options are shown separately. Neither
the presence of a cache nor a final link's O3 proves the device compile usedO3.

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
