#!/usr/bin/env python3
"""Independent re-derivation for the board cross-audit of m16065
(CASE-DEM-MINDLIN-TANGENTIAL, TensorDEM).

Nothing here imports TensorDEM: every reference number is rebuilt from
P, R, E, nu, mu with closed-form Hertz/Mindlin relations and an independent
adaptive quadrature of the 1-D reduction integral.  The only inputs taken from
the audited tree are the *delivered curves* (curves.npz) and results.json.

Outputs artifacts/audit_m16065/independent_check.json.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from scipy.integrate import quad

CASE = pathlib.Path("/root/tp-agent/work/m16065/benchmarks/case_mindlin_tangential")
OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "audit_m16065"

# audited inputs (read from results.json so the audit cannot silently swap them)
R = json.loads((CASE / "results.json").read_text())
P = R["parameters"]["normal_load_N"]
RP = R["parameters"]["radius_m"]
E1 = R["parameters"]["young_modulus_Pa"]
NU = R["parameters"]["poisson_ratio"]
MU = R["parameters"]["friction_coefficient"]


def hertz():
    e_star = E1 / (2.0 * (1.0 - NU ** 2))
    g_star = E1 / (4.0 * (1.0 + NU) * (2.0 - NU))  # 1/G* = 2(2-nu)/G, G=E/2(1+nu)
    r_star = RP / 2.0
    a = (3.0 * P * r_star / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * P / (2.0 * math.pi * a ** 2)
    k_t0 = 8.0 * g_star * a
    d_star = 3.0 * MU * P / (16.0 * g_star * a)
    return dict(e_star=e_star, g_star=g_star, r_star=r_star, a=a, p0=p0,
                k_t0=k_t0, delta_star=d_star)


def ref_virgin(dl, h):
    s = min(dl / h["delta_star"], 1.0)
    return MU * P * (1.0 - (1.0 - s) ** 1.5)


def mdr_exact(dl, h):
    """F = 4 G* int_0^a min(dl, mu p(x)/(4G*)) dx  (x2 for the two half-planes)."""
    e_star, g_star, a, r_star = h["e_star"], h["g_star"], h["a"], h["r_star"]
    p = lambda x: e_star * (a * a - x * x) / r_star
    g = lambda x: min(dl, MU * p(x) / (4.0 * g_star))
    val, err = quad(g, 0.0, a, limit=400, epsabs=1e-18, epsrel=1e-6)
    return 4.0 * g_star * 2.0 * val, err


def reversal_ref(h, n=4001):
    """Exact 1-D-reduction reversal branch after gross slip in +x.

    Straight-line construction: after gross slip every spring sits at +mu p(x);
    reversing by Delta flips the springs whose |mu p| is overcome first, i.e.
    x <= x_r with 4 G* Delta = 2 mu p(x_r).  Full reversal (all springs at
    -mu p) therefore needs Delta = 2 mu p(0)/(4 G*) = 2 delta*.
    """
    e_star, g_star, a, r_star = h["e_star"], h["g_star"], h["a"], h["r_star"]
    p = lambda x: e_star * (a * a - x * x) / r_star
    amps = np.linspace(0.0, 2.0 * h["delta_star"], n)
    f = np.empty_like(amps)
    for i, dl in enumerate(amps):
        # a spring flips when the reversal increment overcomes +mu p(x):
        # 4 G* Delta = 2 mu p(x)  ->  x_r^2 = a^2 - 2 G* Delta R* /(mu E*)
        xr2 = a * a - 2.0 * g_star * dl * r_star / (MU * e_star)
        xr = math.sqrt(max(0.0, xr2))
        stuck, e1 = quad(lambda x: MU * p(x) - 4.0 * g_star * dl, 0.0, xr, limit=400)
        slid, e2 = quad(lambda x: MU * p(x), xr, a, limit=400)
        f[i] = 2.0 * (stuck - slid) if xr > 0 else -2.0 * MU * P / 2.0 / 2.0
    return amps, f


def full_rev(h):
    """Delta at which the independent reversal branch reaches -mu P (bisection)."""
    lo, hi = 0.0, 4.0 * h["delta_star"]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        amps, f = reversal_ref(h, n=3)
        amps = np.array([mid])
        _, f = reversal_ref_single(mid, h)
        if f > -0.999 * MU * P:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reversal_ref_single(dl, h):
    e_star, g_star, a, r_star = h["e_star"], h["g_star"], h["a"], h["r_star"]
    p = lambda x: e_star * (a * a - x * x) / r_star
    xr = math.sqrt(max(0.0, a * a - 2.0 * g_star * dl * r_star / (MU * e_star)))
    stuck, _ = quad(lambda x: MU * p(x) - 4.0 * g_star * dl, 0.0, xr, limit=400)
    slid, _ = quad(lambda x: MU * p(x), xr, a, limit=400)
    return np.array([dl]), np.array([2.0 * (stuck - slid)])


def main() -> None:
    h = hertz()
    rep = {}
    rep["hertz"] = h
    # (1) delivered results.json derived block vs independent Hertz
    d = R["derived"]
    rep["derived_vs_independent"] = {
        k: {"reported": d[k], "independent": h[k],
            "rel_diff": abs(d[k] - h[k]) / max(abs(h[k]), 1e-300)}
        for k in ("e_star", "g_star", "r_star", "a", "p0", "k_t0", "delta_star")
    }

    # (2) closed form vs independent quadrature on the delivered sampling grid
    curves = np.load(CASE / "curves.npz")
    grid = curves["ref_delta"]
    cf = np.array([ref_virgin(x, h) for x in grid])
    q = np.array([mdr_exact(x, h)[0] for x in grid[::20]])
    cf_q = np.array([ref_virgin(x, h) for x in grid[::20]])
    rep["r1_vs_independent_quadrature"] = {
        "n_points": int(q.size),
        "max_abs_dev_over_muP": float(np.max(np.abs(q - cf_q)) / (MU * P)),
        "note": "independent adaptive quad vs closed form, same abscissae",
    }

    # (3) delivered G1 curve: recompute BOTH error normalisations
    dl, fl = curves["delta_law"], curves["force_law"]
    sel = (dl >= 0.02 * h["delta_star"]) & (dl <= 0.99 * h["delta_star"])
    grid40 = np.linspace(0.02, 0.99, 40) * h["delta_star"]
    f_got = np.interp(grid40, dl, fl)
    f_ref = np.array([ref_virgin(x, h) for x in grid40])
    dev_norm = np.abs(f_got - f_ref) / (MU * P)          # what the case calls "rel_err"
    true_rel = np.abs(f_got - f_ref) / f_ref             # conventional relative error
    rep["g1_error_metrics"] = {
        "micro_max_dev_over_muP": float(dev_norm.max()),
        "micro_max_true_relative_error": float(true_rel.max()),
        "true_rel_at_lowest_delta_over_delta_star": float(true_rel[0]),
        "claimed_in_results_json": R["primary"]["law"]["micro_max_rel_err"],
        "plateau_got_N": float(np.interp(1.6 * h["delta_star"], dl, fl)),
        "plateau_ref_N": MU * P,
        "plateau_rel_err": abs(np.interp(1.6 * h["delta_star"], dl, fl) - MU * P) / (MU * P),
    }

    # (4) reversal branch: delivered law vs independent reversal reference
    amps, fref = reversal_ref(h)
    drav, frev = curves["delta_rev"], curves["force_rev"]
    n_fwd = int(curves["n_forward"])
    # delivered path: 0 -> delta_max (forward) then back; amplitude from turnaround
    dmax = float(drav[n_fwd - 1])
    a_got = dmax - drav[n_fwd - 1:]
    f_got_rev = frev[n_fwd - 1:]
    # Cundall-Strack sign convention in the audited driver: force opposes the
    # increment, so the reversal branch starts at -mu P (see run.py: f_ref = -ref.force)
    f_ref_i = -np.interp(np.clip(a_got, 0.0, amps[-1]), amps, fref)
    dev = np.abs(f_got_rev - f_ref_i) / (MU * P)
    tru = np.abs(f_got_rev - f_ref_i) / np.maximum(np.abs(f_ref_i), 1e-12)
    hit = np.where(f_got_rev >= 0.999 * MU * P)[0]
    rep["reversal"] = {
        "reference_quadrature_points": int(amps.size),
        "full_reversal_amplitude_ref_m": float(full_rev(h)),
        "full_reversal_amplitude_ref_over_delta_star": float(full_rev(h) / h["delta_star"]),
        "delivered_law_full_reversal_amplitude_over_delta_star":
            float(a_got[hit[0]] / h["delta_star"]) if hit.size else None,
        "max_dev_over_muP": float(dev.max()),
        "max_true_relative_error": float(tru.max()),
        "claimed_reverse_max_rel_err": R["reversal"]["reverse_max_rel_err"],
        "claimed_law_full_slip_over_delta_star":
            R["reversal"]["reverse_full_slip_amplitude_law"] / h["delta_star"],
        "claimed_reverse_full_slip_amplitude_rel_err":
            R["reversal"]["reverse_full_slip_amplitude_rel_err"],
        "finding": "reversal branch is NOT reproduced (both normalisations >> 5 %); "
                   "the case registers this as capability boundary F1, not as a pass",
    }

    # (5) plateau / slope sanity: dF/d(delta) at delta* must vanish
    eps = 1e-3 * h["delta_star"]
    slope = (ref_virgin(h["delta_star"], h) - ref_virgin(h["delta_star"] - eps, h)) / eps
    rep["plateau_slope_over_k_t0"] = float(slope / h["k_t0"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "independent_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:4000])


if __name__ == "__main__":
    main()