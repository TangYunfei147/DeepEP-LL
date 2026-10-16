"""Compatibility alias: all launch policy lives in the shared LL runner."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'ll_stage_timer/run.py'),run_name='__main__')
