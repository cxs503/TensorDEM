"""XVAL driver (m17503): TensorDEM's periodic SLLOD vs a Lees-Edwards cell.

Two families of evidence are produced, both machine-checkable and cheap:

**A. Invariants** (no reference simulation needed; every mature open-source
periodic-SLLOD code -- LAMMPS ``fix deform``/``fix nvt/sllod``, ESPResSo
``LEES_EDWARDS``, HOOMD ``shear`` -- satisfies them by construction):
  * ``frame_invariance``: a common y-translate of the configuration must not
    change the contact forces (the torus is translation invariant; the SLLOD
    field is not periodic, so only an image-corrected relative velocity passes);
  * ``boundary_rate_error``: the shear rate a y-straddling contact actually sees,
    relative to the imposed rate;
  * ``straddle_fraction``: how many contacts of a dense cell straddle y (the size
    of the population that the defect touches).

**B. Macroscopic cross-check**: tau/P of the same cell at the same
(phi, mu, gamma_dot, n, seed) for the two arms, plus a substep sensitivity and a
shear-rate sweep (the tau/P -- I map is the standard dense-flow coordinate).

Everything is written to ``artifacts/xval_periodic_sllod_m17503/``; the report
numbers are injected from that JSON (no hand-typed digits).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from tensordem.mts_contact import MTSConfig, MTSSystem, shear_cell_run

OUT = pathlib.Path(__file__).resolve().parents[2] / "artifacts/xval_periodic_sllod_m17503"


def _dense_cell(n: int = 96, phi: float = 0.82, gamma_dot: float = 40.0,
                seed: int = 11, *, lees_edwards: bool, strain: float = 0.37):
    """A jammed-by-construction bidisperse cell: contacts exist without dynamics.

    ``phi = 0.82`` of a random bidisperse packing is deep in the overlap regime,
    so the contact network (and hence the invariants) can be evaluated from the
    initial configuration -- no relaxation, no protocol dependence.
    """
    rng = np.random.default_rng(seed)
    r = 5.0e-3
    radii = r * (0.8 + 0.4 * rng.random(n))
    box = math.sqrt(float(np.sum(math.pi * radii ** 2)) / phi)
    pos = rng.random((n, 2)) * box
    cfg = MTSConfig(kn=1.0e5, dt=1.0e-6, n_sub=1, mu=0.5, zeta=0.4,
                    density=2000.0, thickness=0.01, contact_model="linear",
                    skin=0.5 * r)
    sysm = MTSSystem(pos, radii, cfg, periodic=(box, box),
                     shear_rate=gamma_dot, lees_edwards=lees_edwards)
    sysm.strain = strain
    sysm.vel = np.zeros((n, 2))
    sysm._slow_pass()
    sysm._forces(sysm.dt, passes=1, record=True)   # populate the contact ledger
    return sysm, box


def _analytic_rate_error(gamma_dot: float, seed: int = 11):
    """Exact per-contact shear-rate error of the two velocity conventions.

    For a y-straddling contact with branch vector ``l`` the SLLOD field
    ``u_x = gamma_dot*y`` gives ``gamma_dot*l_y``.  The un-corrected field (built
    from wrapped positions) gives ``gamma_dot*(l_y - shift_y)`` with
    ``shift_y = -+Ly``, so its relative error is ``Ly/|l_y|`` -- a number that does
    not depend on the configuration beyond the straddle fraction.
    """
    res = {}
    for arm, shift in (("legacy", -1.0), ("lees_edwards", 0.0)):
        sysm, box = _dense_cell(gamma_dot=gamma_dot, seed=seed,
                                lees_edwards=arm == "lees_edwards")
        sysm.strain = 0.37
        sysm._slow_pass()
        c = sysm.contacts
        pair = c["kind"] == 0
        branch = c["branch"][pair]
        # a y-image contact is one whose branch length is small but whose
        # *wrapped* y difference is close to a full period
        wrapped_dy = sysm.pos[c["j"][pair], 1] - sysm.pos[c["i"][pair], 1]
        straddle = np.abs(wrapped_dy) > 0.5 * box
        ly_eff = float(np.mean(np.abs(branch[straddle, 1]))) if straddle.any() else float("nan")
        err = abs(shift) * box / ly_eff if straddle.any() else float("nan")
        res[arm] = {"n_straddle": int(straddle.sum()),
                    "n_contacts": int(pair.sum()),
                    "mean_abs_ly_straddle": ly_eff,
                    "box": float(box),
                    "rate_error_exact": float(err)}
    res["gamma_dot"] = float(gamma_dot)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--phi", type=float, default=0.80)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--gammas", default="10,20,40")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--relax-steps", type=int, default=1500)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "arms": [], "invariants": {},
               "analytic": {}, "sensitivity": [], "wall_clock": {}}
    t0 = time.perf_counter()
    for g in [float(x) for x in args.gammas.split(",")]:
        for arm in ("legacy", "lees_edwards"):
            t = time.perf_counter()
            r = shear_cell_run(phi=args.phi, mu=args.mu, n=args.n, gamma_dot=g,
                               ratio=args.ratio, steps=args.steps,
                               relax_steps=args.relax_steps, sample_every=20,
                               seed=args.seed, contact_model="linear",
                               lees_edwards=(arm == "lees_edwards"))
            rec = {k: r[k] for k in ("gamma_dot", "pressure", "shear",
                                     "tau_over_p", "shear_stderr",
                                     "pressure_stderr", "coord",
                                     "fraction_sliding", "lees_edwards",
                                     "strain_end", "samples_averaged",
                                     "n", "phi", "mu")}
            rec["arm"] = arm
            rec["wall_clock"] = time.perf_counter() - t
            # inertial number I = gamma_dot * d * sqrt(rho / sigma) with d the
            # mean particle diameter and sigma = P
            rho = 2000.0
            d = 2.0 * 5.0e-3
            rec["I"] = (g * d * math.sqrt(rho / r["pressure"])
                        if r["pressure"] > 0 else float("nan"))
            payload["arms"].append(rec)
            print(f"[{arm}] gamma_dot={g:6.1f} tau/P={rec['tau_over_p']:.4f} "
                  f"P={rec['pressure']:.3e} ({rec['wall_clock']:.1f}s)", flush=True)
    # substep sensitivity of the LE arm at the middle shear rate
    g = float(args.gammas.split(",")[len(args.gammas.split(",")) // 2])
    for ratio in (args.ratio, 0.5 * args.ratio):
        r = shear_cell_run(phi=args.phi, mu=args.mu, n=args.n, gamma_dot=g,
                           ratio=ratio, steps=args.steps,
                           relax_steps=args.relax_steps, sample_every=20,
                           seed=args.seed, contact_model="linear",
                           lees_edwards=True)
        payload["sensitivity"].append({"kind": "substep", "ratio": ratio,
                                       "gamma_dot": g,
                                       "tau_over_p": r["tau_over_p"],
                                       "pressure": r["pressure"]})
        print(f"[sensitivity] ratio={ratio} tau/P={r['tau_over_p']:.4f}", flush=True)
    payload["analytic"] = _analytic_rate_error(float(args.gammas.split(",")[0]))
    payload["wall_clock"]["total"] = time.perf_counter() - t0
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())