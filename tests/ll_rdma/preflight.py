"""Read-only HNS_1825 environment evidence; never declares RDMA data correctness."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import sys
from environment import environment

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("HcommEndpointCreate", "HcommEndpointDestroy", "HcommEndpointGetListenPort",
           "HcommMemReg", "HcommMemUnreg", "HcommChannelCreate", "HcommChannelDestroy", "HcommChannelGetStatus")


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def pci(bdf):
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
        raise ValueError("PCI BDF must have form dddd:bb:ss.f")
    path = Path("/sys/bus/pci/devices") / bdf
    return {"bdf": bdf, "exists": path.exists(), "topology_path": str(path.resolve()),
            **{key: read(path / key) for key in ("vendor", "device", "numa_node", "local_cpulist", "current_link_speed", "current_link_width")},
            "driver": (path / "driver").resolve().name if (path / "driver").exists() else None}


def inspect(args):
    blocked = []
    report = {"schema": 1, "purpose": "environment inventory only; not an RDMA test",
              "visibility": {key: os.environ.get(key) for key in
                             ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES")}}
    if args.validation and set(report["visibility"].values()) != {"0,1"}:
        blocked.append("A5 validation requires all four visibility variables fixed to 0,1")
    manifest = ROOT / "third-party/shmem/install/shmem/deepep-sdk.json"
    try:
        report["sdk"] = json.loads(manifest.read_text())["build"]
        if report["sdk"].get("rdma_backend") != "HNS_1825":
            blocked.append("SDK was not built explicitly for HNS_1825")
    except (OSError, KeyError, ValueError) as exc:
        blocked.append(f"SDK identity unavailable: {exc}")
    try:
        hcomm = ctypes.CDLL("libhcomm.so")
        report["hcomm_missing_symbols"] = [s for s in SYMBOLS if not hasattr(hcomm, s)]
        if report["hcomm_missing_symbols"]:
            blocked.append("libhcomm.so lacks required ROCE endpoint/MR/channel symbols")
    except OSError as exc:
        report["hcomm_load_error"] = str(exc)
        blocked.append("libhcomm.so cannot load; configure the matching CANN/provider library path")
    report["rdma_devices"] = []
    for device in sorted(Path("/sys/class/infiniband").glob("*")):
        link = device / "device"
        report["rdma_devices"].append({"name": device.name, "pci": pci(link.resolve().name),
            "ports": [{"port": p.name, "state": read(p / "state"), "link_layer": read(p / "link_layer")}
                      for p in sorted((device / "ports").glob("*"))]})
    if not report["rdma_devices"]:
        blocked.append("No host RDMA devices enumerated; usable HNS_1825 hardware/provider is not established")
    report["npu_pci"] = [pci(bdf) for bdf in args.npu_bdf]
    if args.nic_bdf:
        report["selected_nic"] = pci(args.nic_bdf)
    report["affinity"] = "Compare PCI ancestry/NUMA with the platform's NPU-to-NIC mapping; common NUMA alone is not proof"
    if args.topology:
        path = project_path(args.topology)
        report["topology"] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            blocked.append("Configured topology file is absent")
    report["unverified"] = ["HNS_1825 identity and provider/driver compatibility", "NPU HBM registration",
                            "per-NPU ROCE IP from SDK rootinfo/virtual topology", "RC ordering and NPU visibility",
                            "channel resource capacity for actual AIV count", "multi-host connectivity"]
    report["blocked"] = blocked
    report["status"] = "blocked" if blocked else "inventory_complete_requires_smoke"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--npu-bdf", action="append", default=[])
    parser.add_argument("--nic-bdf")
    parser.add_argument("--topology", help="Read-only deployment topology file; relative paths use the project root")
    parser.add_argument("--output", default=".logs/ll-rdma-preflight.json")
    args = parser.parse_args()
    os.environ.update(environment())
    output = project_path(args.output)
    if not output.is_relative_to(ROOT):
        parser.error("Report output must remain inside this project")
    report = inspect(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 2 if report["blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
