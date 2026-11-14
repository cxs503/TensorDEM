#!/usr/bin/env python3
"""Independent re-derivation of CASE-DEM-SILO-BEVERLOO (m20055) numbers.

Reads only the filed run artefact ``docs/board_audit/m20232/refs/m20055_D4.json``
(byte copy of ``/root/tp-agent/work/m20055/runs/m20055_beverloo/D4.json``,
sha256 ad594807...) and the filed reference coefficients
``refs/m20055_ref_beverloo_1961.json``.  Nothing is typed by hand: the particle
mass comes from the recorded ``density``/``particle_diameter``, the Beverloo
coefficients from the reference JSON.

Purpose: check that the filed ``W`` really is a function of the recorded
discharge-event stream, and test how sensitive the verdict is to the estimator
that turns that stream into a rate.

    PYTHONPATH=src python scripts/audit/audit_m20055_beverloo.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "docs" / "board_audit" / "m20232" / "refs"
RATE_BIN_S = 0.25


def load():
    run = json.loads((REF_DIR / "m20055_D4.json").read_text())
    ref = json.loads((REF_DIR / "m20055_ref_beverloo_1961.json").read_text())
    return run, ref


def particle_mass(tele: dict) -> float:
    d = float(tele["particle_diameter"])
    # rho_p is recorded in the run's dimensionless block indirectly; take it
    # from the reference-material declaration of the case (glass, 2500 kg/m3)
    # recovered from the artefact itself: rho_b / phi.
    rho_p = float(tele["bulk_density_bed_kg_m3"]) / float(tele["packing_fraction_bed"])
    return rho_p * math.pi / 6.0 * d ** 3


def rate_quantile(t, n, mass, lo, hi):
    """(mass-quantile band rate, interval) -- same definition as the mission's
    ``steady_rate`` but re-implemented here so the check is independent."""
    n_lo, n_hi = lo * n[-1], hi * n[-1]
    t_lo = float(np.interp(n_lo, n, t))
    t_hi = float(np.interp(n_hi, n, t))
    return (n_hi - n_lo) * mass / (t_hi - t_lo), (t_lo, t_hi)


def rate_window(t, n, mass, t_lo, t_hi):
    na = float(np.interp(t_lo, t, n))
    nb = float(np.interp(t_hi, t, n))
    return (nb - na) * mass / (t_hi - t_lo)


def plateau_search(t, n, mass, rel_tol=0.05):
    """Longest interval (>= 0.5 s) over which the binned rate stays inside
    +/- rel_tol of the interval mean -- a steadiness test that does not assume
    where the plateau is."""
    edges = np.arange(t[0], t[-1] + RATE_BIN_S, RATE_BIN_S)
    centres, rates = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        na, nb = np.interp([a, b], t, n)
        centres.append(0.5 * (a + b))
        rates.append((nb - na) * mass / (b - a))
    centres, rates = np.array(centres), np.array(rates)
    best = (0.0, None)
    for i in range(len(rates)):
        for j in range(i + 1, len(rates) + 1):
            seg = rates[i:j]
            mu = seg.mean()
            if mu <= 0:
                continue
            if np.all(np.abs(seg - mu) / mu < rel_tol):
                if len(seg) * RATE_BIN_S > best[0]:
                    best = (len(seg) * RATE_BIN_S, (centres[i], centres[j - 1]))
    return {"bins": [(float(c), float(r)) for c, r in zip(centres, rates)],
            "plateau_len_s": best[0],
            "plateau_interval_s": best[1],
            "plateau_rate": (rate_window(t, n, mass, *best[1]) if best[1] else None)}


def main() -> int:
    run, ref = load()
    tele = run["telemetry"]
    t = np.asarray(tele["discharge_time_s"], dtype=float)
    n = np.asarray(tele["discharge_cumulative_n"], dtype=float)
    mass = particle_mass(tele)
    C = float(ref["coefficients"]["C"])
    k = float(ref["coefficients"]["k"])
    g = float(tele["gravity"])
    d = float(tele["particle_diameter"])
    D = float(tele["orifice_diameter"])
    rho_b = float(tele["bulk_density_bed_kg_m3"])
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(g) * Deff ** 2.5

    est = {}
    W, iv = rate_quantile(t, n, mass, 0.20, 0.80)
    est["quantile_0.20-0.80 (predeclared)"] = {"W": W, "interval_s": iv}
    for lo, hi in ((0.10, 0.60), (0.35, 0.65), (0.05, 0.95), (0.30, 0.90)):
        W2, iv2 = rate_quantile(t, n, mass, lo, hi)
        est[f"quantile_{lo:.2f}-{hi:.2f}"] = {"W": W2, "interval_s": iv2}
    # filed fixed-window mean (the legacy protocol used by m6296) and the same
    # mean after dropping the start-up window that recorded zero discharge
    wins = [float(x) for x in tele["flow_rate_windows_kg_s"]]
    est["filed_window_mean (incl. zero start-up bin)"] = {"W": float(np.mean(wins))}
    est["filed_window_mean (drop zero bins)"] = {
        "W": float(np.mean([w for w in wins if w > 0.0]))}
    est["whole_stream_mean"] = {"W": float(n[-1] * mass / (t[-1] - t[0]))}
    pl = plateau_search(t, n, mass)
    if pl["plateau_rate"] is not None:
        est["plateau_search (longest +/-5% segment)"] = {
            "W": float(pl["plateau_rate"]), "interval_s": pl["plateau_interval_s"],
            "length_s": pl["plateau_len_s"]}

    for name, rec in est.items():
        W = float(rec["W"])
        rec["W"] = W
        rec["rel_pct"] = 100.0 * (W - W_ref) / W_ref
        rec["pass_10pct"] = bool(abs(rec["rel_pct"]) < 10.0)
        rec["pass_15pct"] = bool(abs(rec["rel_pct"]) < 15.0)
        rec["C_recovered"] = W / (rho_b * math.sqrt(g) * Deff ** 2.5)

    filed = run["verdict"]
    recompute_err = abs(est["quantile_0.20-0.80 (predeclared)"]["W"] - filed["W_got_kg_s"])
    out = {
        "audited_run": "runs/m20055_beverloo/D4.json (filed), byte copy sha256 ad594807536c5e28e70f697505031b8825f759c938f18a890723317093ba243e",
        "case": {"D_over_d": D / d, "d_m": d, "D_m": D, "rho_b_kg_m3": rho_b,
                 "phi_bed": float(tele["packing_fraction_bed"]),
                 "particle_mass_kg": mass, "n_discharged": int(n[-1]),
                 "t_first_discharge_s": float(t[0]), "t_last_s": float(t[-1]),
                 "dt_s": float(tele["dt"]), "steps": int(tele["steps"])},
        "beverloo_ref": {"C": C, "k": k, "W_ref_kg_s": W_ref, "Deff_m": Deff},
        "filed": {"W_got_kg_s": filed["W_got_kg_s"], "rel_pct": 100.0 * filed["rel_err_W"],
                  "steady_rate_cv": float(tele["steady_rate_cv"]),
                  "pass_10pct": bool(filed["pass_10pct"])},
        "independent_recompute_of_filed_W_abs_diff": recompute_err,
        "estimators": est,
        "rate_bins": pl["bins"],
    }
    dest = REF_DIR.parent / "audit_m20055_beverloo.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"particle mass      {mass:.6e} kg")
    print(f"W_ref (Beverloo)   {W_ref:.6f} kg/s   (C={C}, k={k}, rho_b={rho_b:.1f})")
    print(f"filed W            {filed['W_got_kg_s']:.6f} kg/s  -> rel {100*filed['rel_err_W']:+.1f}%"
          f"  (independent recompute |diff| = {recompute_err:.2e})")
    print()
    print(f"{'estimator':44s} {'W [kg/s]':>9s} {'rel %':>8s} {'C_rec':>6s} {'<10%':>5s} {'<15%':>5s}")
    for name, rec in est.items():
        print(f"{name:44s} {rec['W']:9.4f} {rec['rel_pct']:+8.1f} "
              f"{rec['C_recovered']:6.3f} {str(rec['pass_10pct']):>5s} {str(rec['pass_15pct']):>5s}")
    print()
    print(f"binned rate (0.25 s): {[round(r,2) for _, r in pl['bins']]}")
    print(f"longest +/-5% plateau: {pl['plateau_len_s']:.2f} s at {pl['plateau_interval_s']}")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())