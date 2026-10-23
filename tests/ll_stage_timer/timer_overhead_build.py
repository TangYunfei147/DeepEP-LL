"""Build standalone real-DeviceTimer A5 probes; does not start any NPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COMMON = Path("csrc/kernels/dispatch/low_latency/common")
HEADERS = ("ll_stage_timer_device.h", "ll_stage_timer_layout.h", "ll_core_config.h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory, relative to checkout")
    parser.add_argument("--baseline", default="98c5f50")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    compiler = os.environ.get("BISHENG") or shutil.which("bisheng") or str(cann / "bin/bisheng")
    report = {"baseline": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", args.baseline], text=True).strip(),
        "current_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "headers": {}, "commands": []}
    for source in ("baseline", "current"):
        dest = output / source
        dest.mkdir()
        for name in (*HEADERS, "ll_stage_timer_profile.h"):
            relative = COMMON / name
            data = (subprocess.check_output(["git", "-C", str(ROOT), "show",
                    args.baseline + ":" + relative.as_posix()]) if source == "baseline"
                    else (ROOT / relative).read_bytes())
            (dest / name).write_bytes(data)
            report["headers"][source + "/" + name] = hashlib.sha256(data).hexdigest()
    include = ["-I" + str(cann / "include"), "-I" + str(cann / "asc/include")]
    send = int(os.environ.get("DEEPEP_LL_SEND_CORES", "28"))
    recv = int(os.environ.get("DEEPEP_LL_RECV_CORES", "28"))
    if send < 1 or recv < 1 or send + recv != 56:
        parser.error("positive S/R must sum to 56")
    report["send_cores"], report["recv_cores"] = send, recv

    def run(command, label):
        start = time.monotonic()
        with (output / (label + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report["commands"].append({"label": label, "argv": command,
            "seconds": time.monotonic() - start, "returncode": result.returncode})
        (output / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(label, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit("Build failed; inspect " + str(output / (label + ".log")))

    objects = []
    for source in ("baseline", "current"):
        for mode in ("off", "on"):
            label = source + "_" + mode
            obj = output / (label + ".o")
            command = [compiler, "-O2", "-std=c++17", "-xasc", "--npu-arch=dav-3510",
                "--cce-auto-infer-kernel-type=false", *include, "-I" + str(output / source),
                "-DLL_TIMER_PROBE_NAME=" + label, "-DDEEPEP_LL_SEND_CORES=" + str(send),
                "-DDEEPEP_LL_RECV_CORES=" + str(recv)]
            if mode == "on":
                command.append("-DDEEPEP_LL_STAGE_TIMER=1")
            if source == "current":
                command += ["-DLL_TIMER_PROBE_SPARSE=1", "-DDEEPEP_LL_TIMER_PROFILE_ID=6"]
            command += ["-c", str(HERE / "timer_overhead_probe.cpp"), "-o", str(obj)]
            run(command, label)
            objects.append(str(obj))
    run([compiler, "-O2", "-std=c++17", *include, str(HERE / "timer_overhead_host.cpp"),
         *objects, "--cce-fatobj-link", "-L" + str(cann / "lib64"),
         "-Wl,-rpath," + str(cann / "lib64"), "-lascendcl", "-lruntime", "-o",
         str(output / "timer_overhead_probe")], "link")


if __name__ == "__main__":
    main()
