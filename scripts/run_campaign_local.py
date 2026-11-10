#!/usr/bin/env python
"""Local parallel driver for the staged granular wave campaign.

The cluster queue is deep, so the campaign is executed on the work node in
short, resumable bursts: every case is a chain of checkpointed stages
(see :mod:`scripts.wave_case_staged`), and each driver call launches the pending
cases in parallel until its own wall-clock budget is spent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CASES = [
    ("p010kPa", 1.0e4, 0.30),
    ("p025kPa", 2.5e4, 0.30),
    ("p050kPa", 5.0e4, 0.30),
    ("p100kPa", 1.0e5, 0.30),
    ("p200kPa", 2.0e5, 0.30),
    ("p400kPa", 4.0e5, 0.30),
    ("neg100kPa_mu0", 1.0e5, 1.0e-6),
]
STAGES = ("prep", "ringP", "ringS", "sweepP", "sweepS", "report")


def case_file(out_dir: str, label: str) -> str:
    return os.path.join(out_dir, label + ".json")


def pending(out_dir: str, label: str) -> list[str]:
    path = case_file(out_dir, label)
    if not os.path.exists(path):
        return list(STAGES)
    with open(path) as fh:
        rec = json.load(fh)
    done = set(rec.get("stages_done", []))
    if os.path.exists(os.path.join(out_dir, label + "_state.pt")):
        done.add("prep")
    return [s for s in STAGES if s not in done]


def reset_stages(out_dir: str, label: str, stages: tuple[str, ...]) -> None:
    path = case_file(out_dir, label)
    if not os.path.exists(path):
        return
    with open(path) as fh:
        rec = json.load(fh)
    rec["stages_done"] = [s for s in rec.get("stages_done", []) if s not in stages]
    for stage in stages:
        if stage in ("sweepP", "sweepS"):
            mode = "P" if stage == "sweepP" else "S"
            rec.get("waves", {}).get(mode, {}).pop("sweep", None)
        if stage == "ringP":
            rec.get("waves", {}).pop("P", None)
        if stage == "ringS":
            rec.get("waves", {}).pop("S", None)
        if stage == "prep":
            state = os.path.join(out_dir, label + "_state.pt")
            if os.path.exists(state):
                os.remove(state)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="runs/wave_campaign")
    ap.add_argument("--budget", type=float, default=90.0, help="driver wall-clock budget [s]")
    ap.add_argument("--case-budget", type=float, default=75.0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--sweep-points", type=int, default=3)
    ap.add_argument("--ringdown-cycles", type=int, default=8)
    ap.add_argument("--sweep-amplitude", type=float, default=1.0e-3)
    ap.add_argument("--kick-amplitude", type=float, default=2.0e-3)
    ap.add_argument("--hold-steps", type=int, default=120)
    ap.add_argument("--hold-window", type=int, default=25)
    ap.add_argument("--reset", type=str, default="",
                    help="comma separated label:stage pairs to redo, e.g. p100kPa:sweepP")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.reset:
        for item in args.reset.split(","):
            label, stage = item.split(":")
            reset_stages(args.out_dir, label, (stage.strip(),))
    t_start = time.time()
    while time.time() - t_start < args.budget:
        todo = [(lab, p, f) for lab, p, f in CASES if pending(args.out_dir, lab)]
        if not todo:
            break
        batch = todo[: args.parallel]
        procs = []
        for label, pressure, friction in batch:
            cmd = [sys.executable, "scripts/wave_case_staged.py",
                   "--label", label, "--pressure", repr(pressure),
                   "--friction", repr(friction), "--nx", str(args.nx),
                   "--nz", str(args.nz), "--threads", str(args.threads),
                   "--budget", str(args.case_budget), "--out-dir", args.out_dir,
                   "--sweep-points", str(args.sweep_points),
                   "--ringdown-cycles", str(args.ringdown_cycles),
                   "--sweep-amplitude", repr(args.sweep_amplitude),
                   "--kick-amplitude", repr(args.kick_amplitude),
                   "--trim-passes", "1", "--hold-tol", "0.12",
                   "--hold-steps", str(args.hold_steps),
                   "--hold-window", str(args.hold_window)]
            env = dict(os.environ, PYTHONPATH="src")
            procs.append((label, subprocess.Popen(cmd, env=env,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.STDOUT)))
        for label, proc in procs:
            proc.wait()
            print(f"[{time.time() - t_start:6.1f}s] {label} rc={proc.returncode} "
                  f"pending={pending(args.out_dir, label)}", flush=True)
    remaining = [(lab, pending(args.out_dir, lab)) for lab, _p, _f in CASES]
    print("remaining:", [r for r in remaining if r[1]], flush=True)


if __name__ == "__main__":
    main()