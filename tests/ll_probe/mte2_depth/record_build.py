"""Save reproducible diagnostic build identity, without checking in artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bypass', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gate-repeats', type=int, choices=(0,16,32,64,128), default=0)
    args = parser.parse_args()
    files = ('kernel.cpp', 'host.cpp', 'build.sh', 'build/mte2_depth', 'build/kernel.dis')
    record = dict(l2_cache='bypass_hint' if args.bypass else 'sdk_default',
                  vector_gate_repeats=args.gate_repeats,
                  files={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files},
                  compiler=(HERE / 'build/compiler-version.txt').read_text())
    (HERE / 'build/identity.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
