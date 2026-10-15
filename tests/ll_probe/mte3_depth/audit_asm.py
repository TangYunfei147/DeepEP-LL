"""Inspect straight-line timestamp windows in the saved device disassembly."""
import argparse
import json
import re
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disassembly', default=str(here / 'build/kernel.dis'))
    args = parser.parse_args()
    lines = Path(args.disassembly).read_text().splitlines()
    marks = [i for i, line in enumerate(lines) if 'SYS_CNT' in line]
    blocks = []
    for a, b in zip(marks, marks[1:]):
        text = '\n'.join(lines[a + 1:b])
        copies = len(re.findall(r'\bMOV_UB_TO_OUT(?:_ALIGN_V2)?\b', text))
        blocks.append(dict(start=lines[a].strip().split(':')[0], copies=copies,
                           mte3_waits=text.count('WAIT_FLAG.MTE3.S'),
                           pipe_barriers=text.count('PIPE_BARRIER'),
                           instructions=b-a-1))
    # N32 可被编译器与其他N共享尾部；直线文本窗口不能替代CFG审查。
    expected = (1, 15, 16, 17)
    for n in expected:
        if not any(x['copies'] == n and x['mte3_waits'] == 0 and x['pipe_barriers'] == 0 for x in blocks):
            raise RuntimeError(f'No unsynchronized timestamp window for N={n}; inspect assembly')
    result = dict(timestamp_reads=len(marks), windows=blocks,
                  limitation='Straight-line windows only; review branches/control flow manually. No hardware OTS inferred.')
    (here / 'build/asm-windows.json').write_text(json.dumps(result, indent=2))
    print(json.dumps([x for x in blocks if x['copies'] in (15,16,17,32)], indent=2))


if __name__ == '__main__':
    main()
