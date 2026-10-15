"""Startup-only progress and Python stack dumps; no device calls or watchdog exits."""
import faulthandler
import json
import os
import time
import sys
from datetime import datetime, timezone


class StartupDiagnostics:
    def __init__(self, directory, rank, timeout=30):
        self.rank = rank
        self.timeout = timeout
        self.active = True
        self.progress = (directory / 'startup.jsonl').open('x', buffering=1)
        self.stacks = (directory / 'startup_stacks.log').open('x', buffering=1)

    def mark(self, stage):
        if not self.active:
            return
        faulthandler.cancel_dump_traceback_later()
        record = dict(stage=stage, rank=self.rank, pid=os.getpid(),
                      utc=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic())
        self.progress.write(json.dumps(record) + '\n')
        self.stacks.write(json.dumps(record) + '\n')
        print(f"STARTUP rank={self.rank} pid={os.getpid()} stage={stage}", file=sys.stderr, flush=True)
        faulthandler.dump_traceback_later(self.timeout, repeat=True, file=self.stacks, exit=False)

    def stop(self, stage='startup_complete'):
        if self.active:
            self.mark(stage)
            faulthandler.cancel_dump_traceback_later()
            self.active = False
            self.progress.close()
            self.stacks.close()

    def __enter__(self):
        self.mark('diagnostics_ready')
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active and exc_type is not None:
            self.mark('startup_exception:' + exc_type.__name__)
            faulthandler.dump_traceback(file=self.stacks, all_threads=True)
        self.stop('startup_exit')
