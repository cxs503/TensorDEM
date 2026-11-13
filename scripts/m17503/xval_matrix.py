"""Strain-matched XVAL matrix (m17503): legacy periodic SLLOD vs Lees-Edwards.

The two arms are compared at *equal accumulated strain* (``gamma_end``), the same
(phi, mu, n, seed, ratio, contact model), and the same physical duration per
rate.  Everything the report quotes is written to
``artifacts/xval_periodic_sllod_m17503/results.json`` and the figure is rendered
from that file (no hand-typed digits).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from tensordem.mts_contact import shear_cell_run

OUT = pathlib.Path(__file__).resolve().parents[2] / "artifacts/xval_periodic_sllod_m17503"
# outer-step size measured on this cell (strain per outer step / shear rate)
STRAIN_PER_STEP_PER_RATE = 2.2e-5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--phi", type=float, default=0.80)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--gammas", default="10,20,40")
    ap.add_argument("--gamma-end", type=float, default=0.2)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sens", action="store_true")
    ap.add_argument("--only-invariants", action="store_true")
    ap.add_argument("--write-every", action="store_true", default=True)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cases, sens = [], []
    t0 = time.perf_counter()
    if args.only_invariants:
        prev = json.loads((OUT / "results.json").read_text())
        inv = _invariants()
        _write(OUT, prev["cases"], prev.get("sensitivity", []), inv, t0)
        print(json.dumps(inv, indent=2, default=float))
        return 0
    for g in [float(x) for x in args.gammas.split(",")]:
        steps = max(20, int(round(args.gamma_end / (STRAIN_PER_STEP_PER_RATE * g))))
        for arm in ("legacy", "lees_edwards"):
            t = time.perf_counter()
            r = shear_cell_run(phi=args.phi, mu=args.mu, n=args.n, gamma_dot=g,
                               ratio=args.ratio, steps=steps, relax_steps=2000,
                               sample_every=args.sample_every, seed=args.seed,
                               contact_model="linear",
                               lees_edwards=(arm == "lees_edwards"))
            cases.append({
                "arm": arm, "gamma_dot": g, "steps": steps,
                "tau_over_p": r["tau_over_p"], "pressure": r["pressure"],
                "shear": r["shear"], "shear_stderr": r["shear_stderr"],
                "pressure_stderr": r["pressure_stderr"], "coord": r["coord"],
                "fraction_sliding": r["fraction_sliding"],
                "strain_end": r["strain_end"], "I": r["I"] if "I" in r else
                g * 0.01 * math.sqrt(2000.0 / r["pressure"]),
                "samples": r["samples_averaged"],
                "wall_clock": time.perf_counter() - t,
            })
            print(f"[{arm:13s}] g={g:5.1f} steps={steps:5d} tau/P="
                  f"{r['tau_over_p']:+.4f} P={r['pressure']:.3e} "
                  f"coord={r['coord']:.2f} strain={r['strain_end']:.3f} "
                  f"({cases[-1]['wall_clock']:.1f}s)", flush=True)
    _write(OUT, cases, [], {}, t0)
    # time-step sensitivity of the LE arm at the middle rate
    if args.no_sens:
        print("skipping sensitivity")
        return 0
    g = float(args.gammas.split(",")[len(args.gammas.split(",")) // 2])
    steps = max(20, int(round(args.gamma_end / (STRAIN_PER_STEP_PER_RATE * g))))
    for ratio in (args.ratio, 0.5 * args.ratio):
        r = shear_cell_run(phi=args.phi, mu=args.mu, n=args.n, gamma_dot=g,
                           ratio=ratio, steps=steps * 2, relax_steps=2000,
                           sample_every=args.sample_every, seed=args.seed,
                           contact_model="linear", lees_edwards=True)
        sens.append({"kind": "substep", "ratio": ratio, "gamma_dot": g,
                     "tau_over_p": r["tau_over_p"], "pressure": r["pressure"],
                     "strain_end": r["strain_end"]})
        print(f"[sensitivity ] ratio={ratio:.3f} tau/P={r['tau_over_p']:+.4f} "
              f"P={r['pressure']:.3e} strain={r['strain_end']:.3f}", flush=True)
    inv = _invariants()
    _write(OUT, cases, sens, inv, t0)
    return 0


def _invariants():
    """Pair-level invariants (analytic, from the tests) on a jammed cell."""
    inv = {}
    for arm in ("legacy", "lees_edwards"):
        row = {}
        forces = []
        for delta in (0.0, 0.371):
            sysm, box = _jammed_cell(lees_edwards=arm == "lees_edwards")
            sysm.pos[:, 1] = np.mod(sysm.pos[:, 1] + delta * box, box)
            sysm._slow_pass()
            f, _ = sysm._forces(sysm.dt, passes=1, record=True)
            forces.append(f)
        denom = max(float(np.linalg.norm(forces[0])), 1e-30)
        row["frame_invariance_rel"] = float(np.linalg.norm(forces[0] - forces[1]) / denom)
        sysm, box = _jammed_cell(lees_edwards=arm == "lees_edwards")
        c = sysm.contacts
        pair = c["kind"] == 0
        branch = c["branch"][pair]
        wrapped_dy = sysm.pos[c["j"][pair], 1] - sysm.pos[c["i"][pair], 1]
        straddle = np.abs(wrapped_dy) > 0.5 * box
        row["n_contacts"] = int(pair.sum())
        row["n_straddle"] = int(straddle.sum())
        row["straddle_fraction"] = float(straddle.sum() / max(int(pair.sum()), 1))
        row["box"] = float(box)
        row["mean_abs_ly_straddle"] = (
            float(np.mean(np.abs(branch[straddle, 1]))) if straddle.any() else float("nan"))
        # exact shear-rate error of the un-corrected field: it uses (l_y - shift_y)
        # instead of l_y, with |shift_y| = Ly
        row["rate_error_legacy_exact"] = (
            float(box / row["mean_abs_ly_straddle"]) if straddle.any() else float("nan"))
        inv[arm] = row
    return inv


def _write(out: pathlib.Path, cases, sens, inv, t0) -> None:
    payload = {"cases": cases, "sensitivity": sens, "invariants": inv,
               "wall_clock_total": time.perf_counter() - t0}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", out / "results.json")


def _jammed_cell(n: int = 96, phi: float = 0.82, gamma_dot: float = 40.0,
                 seed: int = 11, *, lees_edwards: bool):
    """Dense bidisperse cell evaluated without dynamics (contacts by construction)."""
    from tensordem.mts_contact import MTSConfig, MTSSystem
    rng = np.random.default_rng(seed)
    r = 5.0e-3
    radii = r * (0.8 + 0.4 * rng.random(n))
    box = math.sqrt(float(np.sum(math.pi * radii ** 2)) / phi)
    pos = rng.random((n, 2)) * box
    cfg = MTSConfig(kn=1.0e5, dt=1.0e-6, n_sub=1, mu=0.5, zeta=0.4,
                    density=2000.0, thickness=0.01, contact_model="linear",
                    skin=0.5 * r)
    sysm = MTSSystem(pos, radii, cfg, periodic=(box, box), shear_rate=gamma_dot,
                     lees_edwards=lees_edwards)
    sysm.strain = 0.37
    sysm.vel = np.zeros((n, 2))
    sysm._slow_pass()
    sysm._forces(sysm.dt, passes=1, record=True)
    return sysm, box


if __name__ == "__main__":
    raise SystemExit(main())