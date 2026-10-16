"""Verify the shared experiment implementation against another checkout."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('other',type=Path);args=p.parse_args()
    other=args.other if args.other.is_absolute() else ROOT/args.other
    paths=json.loads(Path(__file__).with_name('shared_files.json').read_text())
    bad=[]
    for name in paths:
        if not (other/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).digest()!=hashlib.sha256((other/name).read_bytes()).digest():bad.append(name)
    if bad:raise ValueError('Shared implementation drift: '+', '.join(bad))
    print(f'PASS {len(paths)} shared files are identical; branch transport/group code is intentionally separate')


if __name__=='__main__':main()
