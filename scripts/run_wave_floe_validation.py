#!/usr/bin/env python
"""Run the wave-ice coupling verification campaign (p348w2-n07).

Usage
-----
    python scripts/run_wave_floe_validation.py --select C1,C2 --out part1.json
    python scripts/run_wave_floe_validation.py --select C3,C4,C5,C6,C7 --out part2.json
    python scripts/run_wave_floe_validation.py --merge part1.json part2.json \
        --out artifacts/wave_floe_validation/metrics.json --figures artifacts/.../figures

The full campaign costs a few minutes of CPU (the time-domain RAO sweep dominates),
so the runner supports criterion selection + merge: the shipped
``metrics.json`` was produced by exactly this code path, in two parts, and merged.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.wave_floe import validation as V  # noqa: E402

ALL = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
RUNNERS = {
    "C1": V.criterion_c1,
    "C2": V.criterion_c2,
    "C3": V.criterion_c3,
    "C4": V.criterion_c4,
    "C5": V.criterion_c5,
    "C6": V.criterion_c6,
}
NAMES = {
    "C1": "C1_single_floe_rao",
    "C2": "C2_array_transmission",
    "C3": "C3_wave_breakup",
    "C4": "C4_spectral_input",
    "C5": "C5_energy",
    "C6": "C6_refinement",
    "C7": "C7_honest_fail",
}


def run(selection):
    out = {"campaign": "wave-ice coupling verification (p348w2-n07)",
           "environment": V.run_validation.__globals__["platform"].platform(),
           "criteria": {}}
    for key in selection:
        if key == "C7":
            continue
        fn = RUNNERS[key]
        print(f"[{key}] running ...", flush=True)
        out["criteria"][NAMES[key]] = fn()
        print(f"[{key}] status = {out['criteria'][NAMES[key]]['status']}", flush=True)
    if "C7" in selection:
        out["criteria"][NAMES["C7"]] = V.criterion_c7(out["criteria"])
    return out


def merge(paths):
    merged = {"campaign": "wave-ice coupling verification (p348w2-n07)",
              "criteria": {}}
    for p in paths:
        rec = json.loads(pathlib.Path(p).read_text())
        merged["criteria"].update(rec["criteria"])
        merged.setdefault("environment", rec.get("environment"))
    # C7 must be recomputed on the *merged* set: a per-part run only sees its own
    # criteria and would under-report the honest-FAIL ledger.
    body = {k: v for k, v in merged["criteria"].items() if k != NAMES["C7"]}
    merged["criteria"][NAMES["C7"]] = V.criterion_c7(body)
    order = ["C1_single_floe_rao", "C2_array_transmission", "C3_wave_breakup",
             "C4_spectral_input", "C5_energy", "C6_refinement", "C7_honest_fail"]
    merged["criteria"] = {k: merged["criteria"][k] for k in order
                          if k in merged["criteria"]}
    merged["summary"] = {
        k: v.get("status") for k, v in merged["criteria"].items()}
    merged["conventions"] = {
        "units": "SI",
        "floe": "2-D beam-seas section, unit width",
        "rao_pitch_normalisation": "|theta|/(k A) (surface slope)",
        "hs": "4 sqrt(m0)",
    }
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=",".join(ALL))
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default=None)
    args = ap.parse_args(argv)

    if args.merge:
        result = merge(args.merge)
    else:
        selection = [s.strip().upper() for s in args.select.split(",") if s.strip()]
        result = run(selection)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {out}  ({out.stat().st_size} bytes)", flush=True)
    if args.figures:
        from runpy import run_path
        fig_script = pathlib.Path(__file__).with_name("plot_wave_floe_validation.py")
        run_path(str(fig_script), run_name="__main__",
                 init_globals={"_METRICS": str(out), "_OUT": args.figures})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
