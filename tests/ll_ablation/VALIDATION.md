# Validation status, 2026-10-22

Base algorithm197052c; native experiment sourcebc664a1. Later commits change
scripts, comparisons, offline support and documentation only.

- CPU suite:55passed,2local C++-compiler discovery skips.
- Local offline Git dependency export/import roundtrip passed with checksum/path
  rejection checks. Metadata comparison rejects mismatches; latency rejects ON.
- Config B=M256/H3584/K8/E384/FP8/int32 and EP16 plan validated without devices.
- CANN9.3.0/O2/dav-3510 on177.22.27.1: modes0/2/3/4, seven dispatch entries,
  separate ON/OFF object directories:56objects passed. Mode1 uses the same native
  execution path as mode0; its input changes in the Python harness.
- Initial recursive dependency preparation exceeded its timeout. Following the
  user's network restriction, subsequent updates used uploaded Git bundles and
  compiler checks used existing pinned SHMEM headers. No further external fetch.
- Serial compilation was replaced with4-way parallel after stopping only the
  exact task script/children and confirming header restoration. Initial parallel
  upload CRLF was corrected before running; logs are preserved.
- No SDK rebuild, wheel installation, full-wheel link or NPU execution was done
  during this task. Full EP16 correctness, metadata matching and performance are
  pending on the user's target. Do not use the ablation wheel for production.

Remote workspace: /home/tyf/Q4_remote_test/DeepEP-LL-ablations.
Logs: /home/tyf/Q4_remote_test/.logs/ablation-device-001.log and003.log
(the second full name is ablation-device-003.log).
Modes2/3/4 object directory: .tmp/ablation-device-qcVRR5Og under that workspace.
The original metadata/static hash manifest describes a historical baseline and
is not an equivalence claim for the new experimental modes. Use this suite's
comparison script and native mode identity, not historical timer comparison.

Source-guard follow-up:14ablation tests passed, including real Git roundtrips.
Untracked files and dirty dependency contents at the pinned commit are preserved
and excluded; staged/unstaged tracked edits and changed dependency commits are
rejected with filenames. No CANN/kernel changes in this launcher-only fix.
