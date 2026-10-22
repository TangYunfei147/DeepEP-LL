"""Compact display; raw per-core samples remain the source of truth."""
import json
from pathlib import Path
import statistics
import sys

data=json.loads(Path(sys.argv[1]).read_text())
for case,rows in data.items():
    print(case)
    if rows and 'ticks_p50' in rows[0]:
        for mode in dict.fromkeys(r['mode'] for r in rows):
            selected=[r for r in rows if r['mode']==mode]
            print(' ',mode,'median_core_ticks_per_repeat',
                  statistics.median(r['ticks_per_repeat_p50'] for r in selected))
    else:
        print(rows)
