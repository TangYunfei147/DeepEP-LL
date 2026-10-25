"""Export one readable comparison report within the user's attachment limit."""
import argparse
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 65535
FIELDS = (
    'round', 'seed', 'measured_route_seed', 'label', 'revision', 'operation',
    'mean_kernel_us', 'reference_label', 'reference_revision',
    'reference_delta_us', 'reference_delta_pct', 'execution_position',
    'result_status', 'source_session_id', 'reference_crosses_sessions',
    'shmem_revision', 'reference_shmem_revision', 'sdk_identity_source',
    'sdk_library_matches_reference',
)
REQUIRED = (
    'round', 'seed', 'label', 'revision', 'operation', 'mean_kernel_us',
    'reference_label', 'reference_revision', 'reference_delta_us', 'reference_delta_pct',
)


def render(source):
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or any(name not in reader.fieldnames for name in REQUIRED)):
            raise ValueError('Expected the suite comparison.csv with explicit round/seed/reference fields')
        rows = list(reader)
    if not rows or any(None in row or any(not (row[name] or '').strip() for name in REQUIRED) for row in rows):
        raise ValueError('Empty or malformed comparison CSV')
    output = io.StringIO(newline='')
    output.write(
        'LL comparison report; one UTF-8 file, at most 65535 bytes.\n'
        'Metric: mean_kernel_us pools each rank\'s last 50 of 100 kernel samples.\n'
        'Reference deltas are current minus reference within the same round and seed; negative is faster.\n'
        'FP8 dispatch configuration uses BF16 combine. Blank identity fields mean unavailable.\n'
        'All comparison rows and original numeric text are retained. Raw samples, per-rank details,\n'
        'full hashes, paths and redundant delta columns are omitted; this is not port utilization data.\n'
        f'Rows: {len(rows)}\n\n')
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    data = output.getvalue().encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError(f'Report is {len(data)} bytes, above {MAX_BYTES}; no file written and no rows truncated')
    return data


def export(source, destination):
    data = render(source)  # Validate the full report before creating any output.
    with Path(destination).open('xb') as stream:
        stream.write(data)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing comparison.csv; use the selected session directory after resume')
    parser.add_argument('--output', type=Path, help='Fresh report path; default <directory>/analysis-report.txt')
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    destination = (ROOT / args.output).resolve() if args.output else directory / 'analysis-report.txt'
    try:
        size = export(directory / 'comparison.csv', destination)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(f'Send only this file: {destination}\nFiles: 1/10; bytes: {size}/{MAX_BYTES}')


if __name__ == '__main__':
    main()
