"""A5 correctness/profiler smoke validation; fixed devices 0/1 in shared runner."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    probe = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(probe / "profiling/validate_profile.py"),
                    "--cases", 'metadata'], check=True)
