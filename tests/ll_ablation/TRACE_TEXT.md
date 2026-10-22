# Compressed text transfer and local trace rendering

The receiver suite now ends with a compressed text export. This is an offline
tool update: no wheel rebuild, SDK change or NPU run is needed to transfer an
existing result. The old `receiver_export.py` JSON format remains readable.

Reference reviewed at DJLRY/DeepEP commit
`855f848de7bc41e4cbe310cb74099fc8321b401e`:
[trace_text.py](https://github.com/DJLRY/DeepEP/blob/855f848de7bc41e4cbe310cb74099fc8321b401e/experiments/megamoe/scripts/trace_text.py)
delegates to
[text_transfer.py](https://github.com/DJLRY/DeepEP/blob/855f848de7bc41e4cbe310cb74099fc8321b401e/experiments/megamoe/python/deep_ep_experimental/megamoe/timer/text_transfer.py).
We use the same workflow idea: preserve data, compress it, transfer checksummed
text blocks, restore locally and render there. Our format and implementation are
separate and do not claim compatibility with MegaMoE's CSV layout or event schema.

## Export on the target

Prefer the original run directory: this preserves the full raw timer captures,
including all initial-dispatch records, counters and actual sampled intervals.

```bash
python3 tests/ll_ablation/trace_text.py export \
  .ablation/results/recv40-002-ready-breakdown \
  --output-dir .ablation/results/recv40-002-trace-text

python3 tests/ll_ablation/trace_text.py show \
  .ablation/results/recv40-002-trace-text --part 1
```

`export` prints the number of parts. Repeat `show` with2,3,...; copy one complete
BEGIN/END block per transfer. `show` prints only that block. The default limit is
60000 ASCII characters/bytes, counting the envelope, final newline and worst-case
CRLF endings. `--max-bytes` may be set up to65535. The manifest remains on the
target; it does not have to be copied separately. Existing directories are never
overwritten.

If only the previous JSON export remains, use it as source instead:

```bash
python3 tests/ll_ablation/trace_text.py export \
  .ablation/results/recv40-002-export \
  --output-dir .ablation/results/recv40-002-export-text
```

This verifies and preserves the complete existing index and its listed parts;
unrelated neighboring files are not collected. It retains only the evidence that
the old export actually contained, not the absent initial-dispatch raw stages.

## Inspect and restore copied text

Save copied blocks in one or more UTF-8 text files. They can arrive out of order;
identical repeated blocks are accepted. Missing parts are listed. Conflicting
duplicates, mixed bundles, bad hashes and incomplete envelopes are rejected.
Whitespace wrapping and CRLF conversion of Base64 are accepted; data edits are
never guessed or repaired.

```bash
python3 tests/ll_ablation/trace_text.py inspect copied-1.txt copied-2.txt

python3 tests/ll_ablation/trace_text.py restore copied-1.txt copied-2.txt \
  --output-dir .ablation/restored/recv40-002
```

Use `-` alone instead of file arguments to read stdin. Project-relative paths
resolve from the checkout, not from the caller's working directory. Absolute
paths are also supported. Missing/corrupt bundles are not restored. Restoration
checks compressed and raw bundle hashes, each original file's hash and semantic
receiver data validity. All original file bytes are preserved. Decompression
and total input are bounded; only plain JSON files below the selected new output
directory are restored. No external downloads or installed NPU packages are used.

Restoration renders Chrome/Perfetto-compatible JSON under `rendered/` by default.
`--no-render` restores verified source data only. `render-report.json` records
coverage, missing intervals and generated files. Open an appropriate rank's
trace JSON in the trace viewer locally; there is no need to transfer rendered
trace JSON back from the target.

## Trace semantics

- Raw runs use the existing schema parser and keep all actual sampled stages.
  Cumulative calls/cycles do not become fabricated repeated events. Original
  counters and unavailable intervals remain in the restored source captures.
- Old JSON exports can render the three replay envelopes only where timestamps
  exist. They cannot reconstruct preparation, verification, synchronization or
  the original production timeline. Missing intervals remain missing.
- Each capture uses its own origin; ranks are not aligned to a common clock.
  Never subtract raw timestamps between ranks to infer network latency.
- Transport and rendering do not fix the experiment's order/carryover effects,
  layout-equivalence gaps or missing per-arm metadata validation. See
  [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md). The kernel instrumentation is unchanged.

The complete transfer preserves precision and data coverage; compression merely
reduces the number of copy operations. For a quick human-readable summary, the
existing receiver comparison JSON and CSV files remain available.

## Validation

The complete recv40-002 evidence supplied through JSON text was verified after
the missing part11 arrived:24data parts,144rank/capture records and8064core
records. All file/record hashes and the whole dataset hash matched. Re-encoding
the25original files (1240759bytes including the index) produced155584compressed
bytes and4text blocks; the largest block including CRLF was59990bytes.
Out-of-order restore with an identical duplicate recovered all original files
and generated16rank traces containing24192recorded replay intervals, with no
missing intervals. This is the legacy replay export, not full raw dispatch data.

CPU regression:211passed,2local C++ compiler skips. Tests cover raw-run and legacy
package round-trips, exact byte preservation, block limits, wrapping/CRLF,
missing/mixed/conflicting parts, bounded decompression, path checks, absent
timestamps, independent capture clocks and existing-file preservation. No kernel
or timer schema changed; no new NPU performance measurement was performed.
