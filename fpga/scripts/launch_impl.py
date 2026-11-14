"""launch_impl.py : run several fpga/scripts/build.tcl implementations in parallel (Vivado batch).

usage (repository root):
    python fpga/scripts/launch_impl.py --configs 1:250,2:250,4:250,8:250 [--strategy default]
                                       [--stagger 120] [--retries 3] [--tag first] [--suffix _v2]

Each configuration L:MHz runs in fpga/build/L<L>_<MHz>MHz (suffix _<strategy> for non-default
strategies) with that directory as the working directory, so every Vivado process has its own
.Xil scratch area. Launches are staggered because concurrent synth_design start-up on this host
intermittently fails inside Vivado's own Tcl loader ("rt-undefined", "couldn't read file
.../scripts/rt/..."); such start-up failures are detected in the log and the run is relaunched
(the failed log is kept as vivado_attempt<k>.log). Design errors are not retried.

Progress is written to fpga/build/launcher_<tag>.json. The script never programs a device.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VIVADO = r"<vivado-2022.2>\bin\vivado.bat"
BUILD_TCL = os.path.join(REPO, "fpga", "scripts", "build.tcl")
BUILD_ROOT = os.path.join(REPO, "fpga", "build")

STARTUP_SIGNATURES = [
    "rt-undefined",
    "rtSynthParallelPrep.tcl",
    "couldn't read file \"<vivado-2022.2>/scripts/rt",
    "rt/data/common.tcl",
    "Unable to obtain license",
        # transient host file-access failures seen on this machine in the middle of synth_design (for example the tool's own .Xil/<pid>/realtime/<top>_synth.xdc "Cannot open file")
    "Cannot open file:",
    ": No error",
]

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def run_dir(lanes, mhz, strategy, suffix=""):
    name = f"L{lanes}_{mhz}MHz" + ("" if strategy == "default" else f"_{strategy}") + suffix
    return name, os.path.join(BUILD_ROOT, name)

def read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def summary(path):
    kv = {}
    for line in read(path).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

def launch(job):
    d = job["dir"]
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "vivado.log")
    if os.path.exists(log):
        os.replace(log, os.path.join(d, f"vivado_attempt{job['attempt']}.log"))
    summ = os.path.join(d, "summary.txt")
    if os.path.exists(summ):
        os.replace(summ, os.path.join(d, f"summary_attempt{job['attempt']}.txt"))
    job["attempt"] += 1
    cmd = (f'"{VIVADO}" -mode batch -nojournal -log "{log}" -source "{BUILD_TCL}" '
           f'-tclargs {job["lanes"]} {job["mhz"]} "{d}" {job["strategy"]}')
    job["command"] = cmd
    job["cwd"] = d
    out = open(os.path.join(d, "stdout.txt"), "w")
    err = open(os.path.join(d, "stderr.txt"), "w")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    job["proc"] = subprocess.Popen(cmd, cwd=d, stdout=out, stderr=err, shell=True, creationflags=flags)
    job["pid"] = job["proc"].pid
    job["state"] = "RUNNING"
    job["launched"] = now()
    print(f"[{now()}] launched {job['name']} attempt {job['attempt']}: {cmd}", flush=True)

def save(state_path, jobs, meta):
    rec = dict(meta)
    rec["updated"] = now()
    rec["jobs"] = [{k: v for k, v in j.items() if k != "proc"} for j in jobs]
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, state_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma list of LANES:MHZ")
    ap.add_argument("--strategy", default="default")
    ap.add_argument("--stagger", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--suffix", default="", help="run directory suffix, e.g. _v2 (keeps older builds intact)")
    a = ap.parse_args()

    jobs = []
    for c in a.configs.split(","):
        lanes, mhz = (int(x) for x in c.split(":"))
        name, d = run_dir(lanes, mhz, a.strategy, a.suffix)
        jobs.append(dict(name=name, lanes=lanes, mhz=mhz, strategy=a.strategy, dir=d,
                         attempt=0, state="PENDING", history=[]))
    os.makedirs(BUILD_ROOT, exist_ok=True)
    state_path = os.path.join(BUILD_ROOT, f"launcher_{a.tag}.json")
    meta = dict(record="fpga_impl_launcher", started=now(), configs=a.configs,
                strategy=a.strategy, stagger_s=a.stagger, retries=a.retries,
                note="no device is programmed by this launcher")

    last_launch = 0.0
    while True:
        # launch pending jobs, one per stagger interval
        for j in jobs:
            if j["state"] in ("PENDING", "RETRY") and time.time() - last_launch >= a.stagger:
                launch(j)
                last_launch = time.time()
                break
        # check running jobs
        for j in jobs:
            if j["state"] != "RUNNING":
                continue
            rc = j["proc"].poll()
            if rc is None:
                continue
            j["returncode"] = rc
            j["finished"] = now()
            kv = summary(os.path.join(j["dir"], "summary.txt"))
            log = read(os.path.join(j["dir"], "vivado.log"))
            startup = [s for s in STARTUP_SIGNATURES if s in log]
            j["history"].append(dict(attempt=j["attempt"], returncode=rc, finished=j["finished"],
                                     build_status=kv.get("BUILD_STATUS"), startup_failure=startup))
            if kv.get("BUILD_STATUS") == "DONE":
                j["state"] = "DONE"
            elif startup and j["attempt"] <= a.retries:
                j["state"] = "RETRY"
            else:
                errs = [ln for ln in log.splitlines() if ln.startswith("ERROR")][:20]
                j["errors"] = errs
                j["state"] = "FAILED"
            print(f"[{now()}] {j['name']} exited rc={rc} -> {j['state']} {startup}", flush=True)
        save(state_path, jobs, meta)
        if all(j["state"] in ("DONE", "FAILED") for j in jobs):
            break
        time.sleep(a.poll)
    print(f"[{now()}] all done: " + ", ".join(f"{j['name']}={j['state']}" for j in jobs), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
