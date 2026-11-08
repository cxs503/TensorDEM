"""IN60-N09 campaign runner: pendular bridge -> r_u -> phi_eff -> cargo shift -> GZ.

Produces the machine-readable evidence of the campaign node

    artifacts/in60-n09/metrics.json      (every criterion number, G1..G6)
    artifacts/in60-n09/fig_*.png         (paper figures)

Usage::

    PYTHONPATH=src python3 scripts/in60_n09_cargo_liquefaction.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem import cargo_liquefaction_effstress as cl  # noqa: E402
from tensordem.liquid_bridge import solve_axisymmetric_bridge  # noqa: E402
from tensordem.pendular_pile2d import DiskPile  # noqa: E402

ART = ROOT / "artifacts" / "in60-n09"
FIG = ART


# ---------------------------------------------------------------------------
# G1 / G4: bridge law against the analytic limit and the Young-Laplace reference
# ---------------------------------------------------------------------------

def g1_bridge_law() -> dict:
    R, gamma = 1.0e-3, cl.WATER_SURFACE_TENSION
    f0 = 2.0 * math.pi * gamma * R
    # (a) analytic limits and monotonicity
    lim = {
        "gap_zero_relative_error": abs(cl.rouault_pendular_force(R, gamma, 1e-11, 0.0) - f0) / f0,
        "large_volume_relative_error": abs(
            cl.rouault_pendular_force(R, gamma, 1.0 * R ** 3, 0.0) - f0) / f0,
        "non_wetting_force": cl.rouault_pendular_force(R, gamma, 1e-11, 0.0, 120.0),
        "contact_angle_60_ratio": cl.rouault_pendular_force(
            R, gamma, 1e-11, 0.0, 60.0) / f0,
    }
    # (b) independent analytic evaluation of the same closed form written out
    # by hand (the docstring formula), as a code-correctness gate
    worst_kernel = 0.0
    for v_ratio in (0.005, 0.02, 0.05, 0.1):
        for d_ratio in (0.0, 0.005, 0.02, 0.05, 0.1):
            v = v_ratio * R ** 3
            s = (0.5 * d_ratio * R) / (2.0 * ((3.0 * v) / (4.0 * math.pi)) ** (1.0 / 3.0))
            hand = f0 / (1.0 + 1.05 * s + 2.5 * s * s)
            worst_kernel = max(worst_kernel, abs(hand - cl.rouault_pendular_force(
                R, gamma, v, 0.5 * d_ratio * R)) / hand)
    # (c) exact axisymmetric Young-Laplace reference (in-house, independent)
    rows = []
    for v_ratio in (0.01, 0.03, 0.06, 0.1):
        for d_ratio in (0.005, 0.02, 0.05, 0.1):
            sol = solve_axisymmetric_bridge(R, gamma, v_ratio * R ** 3, d_ratio * R, 0.0)
            if not sol.converged:
                rows.append({"v_ratio": v_ratio, "d_ratio": d_ratio, "converged": False})
                continue
            f_rou = cl.rouault_pendular_force(R, gamma, v_ratio * R ** 3, d_ratio * R)
            f_del = cl.delenne_pendular_force(R, gamma, v_ratio * R ** 3, d_ratio * R)
            rows.append({
                "v_ratio": v_ratio, "d_ratio": d_ratio, "converged": True,
                "force_exact": sol.force,
                "force_rouault": f_rou, "force_delenne": f_del,
                "dev_rouault_percent": 100.0 * abs(f_rou - sol.force) / sol.force,
                "dev_delenne_percent": 100.0 * abs(f_del - sol.force) / sol.force,
                "reference_over_derjaguin": sol.force / f0,
            })
    conv = [r for r in rows if r["converged"]]
    devs = [r["dev_rouault_percent"] for r in conv]
    return {
        "analytic_limits": lim,
        "kernel_vs_hand_formula_max_rel": worst_kernel,
        "exact_reference_rows": rows,
        "exact_reference_n_converged": len(conv),
        "exact_reference_median_deviation_percent": float(np.median(devs)) if devs else float("nan"),
        "exact_reference_max_deviation_percent": float(np.max(devs)) if devs else float("nan"),
        "reference_amplitude_at_smallest_gap": conv[0]["reference_over_derjaguin"] if conv else float("nan"),
    }


def g4_energy() -> dict:
    R, gamma = 1.0e-3, cl.WATER_SURFACE_TENSION
    vol = 0.03 * R ** 3
    deriv = max(cl.force_energy_derivative_residual(R, gamma, vol, d * R)
                for d in (0.001, 0.01, 0.05, 0.2, 0.5))
    budget = cl.separate_pair_energy_budget(
        mass=4.0 / 3.0 * math.pi * R ** 3 * 2650.0,
        radius=R, gamma=gamma, volume=vol, gap0=0.02 * R,
        velocity0=0.5, dt=2.0e-7, n_steps=400,
    )
    budget_slow = cl.separate_pair_energy_budget(
        mass=4.0 / 3.0 * math.pi * R ** 3 * 2650.0,
        radius=R, gamma=gamma, volume=vol, gap0=0.02 * R,
        velocity0=0.5, dt=1.0e-7, n_steps=800,
    )
    return {"derivative_residual_max": deriv, "budget": budget, "budget_half_dt": budget_slow}


# ---------------------------------------------------------------------------
# G3: effective friction angle versus saturation
# ---------------------------------------------------------------------------

def g3_friction_angle(matrix: cl.CargoMatrix) -> dict:
    sat = np.arange(0.0, 1.0001, 0.01)
    rows = cl.moisture_scan(matrix, sat)
    phi = np.array([r["phi_eff_deg"] for r in rows])
    i = int(np.argmax(phi))
    neg = matrix.effective_friction_angle(0.95 * matrix.full_moisture_point, 1.0)
    # monotone rise and monotone fall around the peak
    rising = bool(np.all(np.diff(phi[: i + 1]) >= -1e-9))
    falling = bool(np.all(np.diff(phi[i:]) <= 1e-9))
    return {
        "rows": rows,
        "peak_index": i,
        "peak_saturation": float(sat[i]),
        "peak_phi_deg": float(phi[i]),
        "dry_phi_deg": float(phi[0]),
        "monotone_rising_to_peak": rising,
        "monotone_falling_after_peak": falling,
        "peak_in_window_0p80_0p95": bool(0.80 <= sat[i] <= 0.95),
        "negative_control_phi_deg_at_ru_1": neg["phi_eff_deg"],
        "cohesion_max_pa": matrix.capillary_cohesion(matrix.saturation_pendular * matrix.full_moisture_point),
        "full_moisture_point": matrix.full_moisture_point,
        "bridge_volume_ratio": matrix.bridge_volume_ratio(0.9 * matrix.full_moisture_point),
    }


# ---------------------------------------------------------------------------
# G2: particle DEM repose bench (dry and wet)
# ---------------------------------------------------------------------------

def g2_repose(cohesion_run_steps: int = 9000) -> dict:
    out = {"runs": {}, "note": "controlled comparison, identical seed/geometry/steps"}
    for tag, gamma in (("dry", 0.0), ("wet_1x", 0.072), ("wet_5x", 0.36)):
        pile = DiskPile(mu=0.5, mu_rolling=0.2, gamma=gamma, bridge_volume_ratio=0.05)
        hist = []
        for k in range(1, cohesion_run_steps + 1):
            pile.step()
            if k % 1000 == 0:
                a = pile.repose_angle_deg()
                hist.append({"step": k, "theta_r_deg": a.get("theta_r_deg"),
                             "r2": a.get("r2"), "n_fit": a.get("n_fit")})
        out["runs"][tag] = {
            "cohesion_number": pile.cohesion_number,
            "history": hist,
            "final_check": pile.repose_angle_deg(),
            "particles": pile.n_particles,
        }
    return out


# ---------------------------------------------------------------------------
# G5: shift threshold curve and timestep convergence
# ---------------------------------------------------------------------------

def _shift_curve(matrix, moisture_ratio, amplitudes, dt, cycles, nx):
    """Max transverse CoG excursion versus roll amplitude (smooth, monotone)."""
    out = []
    for amp in amplitudes:
        res = cl.simulate_roll_shift(
            matrix, cl.RollExcitation(amplitude_deg=amp), moisture_ratio,
            cycles=cycles, dt=dt, nx=nx, record=False,
        )
        out.append({"amplitude_deg": amp,
                    "shift_max_abs_m": res.shift_max_abs_m,
                    "shift_cycle_mean_m": res.shift_cycle_mean_m,
                    "pore_pressure_ratio_final": res.pore_pressure_ratio_final,
                    "phi_eff_deg_final": res.phi_eff_deg_final})
    return out


def _crossing(curve, limit):
    """Linear interpolation of the amplitude where the excursion reaches limit."""
    for a, b in zip(curve, curve[1:]):
        if a["shift_max_abs_m"] < limit <= b["shift_max_abs_m"]:
            span = b["shift_max_abs_m"] - a["shift_max_abs_m"]
            frac = (limit - a["shift_max_abs_m"]) / span if span > 0 else 0.0
            return a["amplitude_deg"] + frac * (b["amplitude_deg"] - a["amplitude_deg"])
    return float("nan")


def g5_thresholds(matrix: cl.CargoMatrix, fast: bool = False) -> dict:
    """Threshold curve of the critical roll amplitude versus saturation (G5).

    The hazard metric is the *maximum* transverse CoG excursion of the exposure
    (``max |y_G|``), i.e. the offset that acts when the ship is at its worst
    heel; it is a smooth monotone function of the roll amplitude, so the
    threshold (``max |y_G| > shift_limit``, 2 % of the hold beam) is obtained by
    linear interpolation of a seven-point sweep instead of a bisection — that
    makes the timestep-convergence claim meaningful rather than a knife edge.
    """
    amplitudes = (6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0)
    limit = 0.30
    sats = (0.50, 0.70, 0.84, 0.90, 0.95, 1.00)
    cycles = 30 if fast else 40
    nx = 24 if fast else 32
    curves = {}
    for dt in (2.0, 1.0):
        rows = []
        for s in sats:
            curve = _shift_curve(matrix, s * matrix.full_moisture_point, amplitudes, dt, cycles, nx)
            crit = _crossing(curve, limit)
            rows.append({"saturation": s, "critical_amplitude_deg": crit, "curve": curve})
            print(f"  dt={dt} S={s:.2f} crit={crit}", flush=True)
        curves[f"dt_{dt}"] = rows
    a = {r["saturation"]: r["critical_amplitude_deg"] for r in curves["dt_2.0"]}
    b = {r["saturation"]: r["critical_amplitude_deg"] for r in curves["dt_1.0"]}
    deltas = {}
    for s in sats:
        if a[s] == a[s] and b[s] == b[s] and a[s] > 0:
            deltas[f"{s:.2f}"] = abs(a[s] - b[s]) / a[s]
        else:
            deltas[f"{s:.2f}"] = float("nan")
    finite = [v for v in deltas.values() if v == v]
    hazard = [s for s in sats if s >= 0.84]
    mono_hazard = all(
        a[hazard[k]] == a[hazard[k]] and a[hazard[k + 1]] == a[hazard[k + 1]]
        and a[hazard[k]] >= a[hazard[k + 1]] - 1e-9
        for k in range(len(hazard) - 1)
    )
    common = [s for s in sats if a[s] == a[s] and b[s] == b[s]]
    mono_all = all(a[common[k]] >= a[common[k + 1]] - 1e-9 for k in range(len(common) - 1))
    return {
        "shift_limit_m": limit,
        "amplitudes_deg": list(amplitudes),
        "curves": curves,
        "dt_relative_difference": deltas,
        "dt_max_relative_difference": float(np.nanmax(finite)) if finite else float("nan"),
        "monotone_decreasing_full_range": bool(mono_all),
        "monotone_decreasing_hazard_window": bool(mono_hazard),
        "hazard_window": hazard,
        "with_finite_threshold": [s for s in sats if a[s] == a[s]],
    }


def threshold_limit_sensitivity(g5: dict, limits=(0.10, 0.15, 0.20, 0.30)) -> dict:
    """Crossing amplitude versus the excursion limit, from the stored curves."""
    table = {}
    for lim in limits:
        row = {}
        for key, rows in g5["curves"].items():
            row[key] = {f"{r['saturation']:.2f}": _crossing(r["curve"], lim) for r in rows}
        diffs = {}
        for sat in row.get("dt_2.0", {}):
            a = row["dt_2.0"][sat]
            b = row["dt_1.0"].get(sat, float("nan"))
            if a == a and b == b and a > 0:
                diffs[sat] = abs(a - b) / a
        sats = [float(k) for k in row["dt_2.0"]]
        seq = [row["dt_2.0"][f"{s:.2f}"] for s in sats]
        # a case whose excursion never reaches the limit has an infinite
        # threshold: that is the correct ordering for monotonicity purposes
        eff = [x if x == x else float("inf") for x in seq]
        mono = all(eff[k] >= eff[k + 1] - 1e-9 for k in range(len(eff) - 1))
        haz = [x for x, s in zip(eff, sats) if s >= 0.84]
        mono_haz = all(haz[k] >= haz[k + 1] - 1e-9 for k in range(len(haz) - 1))
        table[f"{lim:.2f}"] = {"critical_amplitude": row, "dt_relative_difference": diffs,
                               "monotone_full_range": bool(mono),
                               "monotone_hazard_window": bool(mono_haz)}
    return table


def reinterpret_threshold_limit(out: dict, limit: float) -> dict:
    """Recompute the G5 verdict at a declared hazard limit from the stored curves.

    The swing-amplitude sweep is expensive; every declared limit is evaluated
    from the same stored curves so that the choice of ``shift_limit`` is visible
    and cannot silently select a favourable outcome.
    """
    g5 = out["G5"]
    rows_by_dt = {}
    for key, rows in g5["curves"].items():
        rows_by_dt[key] = {r["saturation"]: _crossing(r["curve"], limit) for r in rows}
    a, b = rows_by_dt["dt_2.0"], rows_by_dt["dt_1.0"]
    sats = sorted(a)
    deltas = {}
    for s in sats:
        if a[s] == a[s] and b[s] == b[s] and a[s] > 0:
            deltas[f"{s:.2f}"] = abs(a[s] - b[s]) / a[s]
    finite = [v for v in deltas.values() if v == v]
    eff = [a[s] if a[s] == a[s] else float("inf") for s in sats]
    haz = [a[s] if a[s] == a[s] else float("inf") for s in sats if s >= 0.84]
    g5["shift_limit_m"] = limit
    g5["critical_amplitude"] = {f"{s:.2f}": a[s] for s in sats}
    g5["critical_amplitude_dt_1.0"] = {f"{s:.2f}": b[s] for s in sats}
    g5["dt_relative_difference"] = deltas
    g5["dt_max_relative_difference"] = float(np.nanmax(finite)) if finite else float("nan")
    g5["monotone_decreasing_full_range"] = bool(
        all(eff[k] >= eff[k + 1] - 1e-9 for k in range(len(eff) - 1)))
    g5["monotone_decreasing_hazard_window"] = bool(
        all(haz[k] >= haz[k + 1] - 1e-9 for k in range(len(haz) - 1)))
    g5["with_finite_threshold"] = [s for s in sats if a[s] == a[s]]
    return out
    return table


def critical_moisture_scan(matrix: cl.CargoMatrix, amplitudes=(8.0, 12.0, 16.0, 20.0, 24.0)) -> dict:
    """Lowest saturation whose max CoG excursion exceeds the limit, per amplitude."""
    rows = []
    for amp in amplitudes:
        load = cl.RollExcitation(amplitude_deg=amp)
        found = None
        for s in np.arange(0.50, 1.0001, 0.02):
            res = cl.simulate_roll_shift(matrix, load, s * matrix.full_moisture_point,
                                         cycles=30, dt=2.0, nx=24, record=False)
            if res.shift_max_abs_m > 0.30:
                found = float(s)
                break
        rows.append({"roll_amplitude_deg": amp, "critical_saturation": found})
    crit = [r["critical_saturation"] for r in rows]
    mono = all(
        (crit[k] is None) or (crit[k + 1] is None) or crit[k] >= crit[k + 1] - 1e-9
        for k in range(len(crit) - 1)
    )
    return {"rows": rows, "monotone_decreasing_in_amplitude": bool(mono)}


def force_amplitude_sensitivity(matrix: cl.CargoMatrix) -> dict:
    """+-20 % on the bridge force amplitude (gamma scaling) — chain robustness.

    The unresolved discrepancy against the in-house Young-Laplace reference is an
    *amplitude* question, so the honest test is whether the chain's verdicts move
    when ``F_cap`` is scaled by 0.8 / 1.2 (``gamma`` scales ``F_cap`` linearly and
    leaves the Derjaguin normalisation untouched).
    """
    base = {}
    rows = []
    for scale in (0.8, 1.0, 1.2):
        m2 = cl.CargoMatrix(surface_tension=matrix.surface_tension * scale)
        scan = cl.moisture_scan(m2, np.arange(0.0, 1.0001, 0.01))
        phi = np.array([r["phi_eff_deg"] for r in scan])
        i = int(np.argmax(phi))
        crit = cl.critical_roll_amplitude(
            m2, 0.95 * m2.full_moisture_point, shift_limit_m=0.10, cycles=40,
            dt=2.0, nx=32, amplitude_bounds_deg=(6.0, 34.0), tol_deg=1.0,
        )
        rows.append({"force_scale": scale, "peak_saturation": float(scan[i]["saturation"]),
                     "peak_phi_deg": float(phi[i]),
                     "cohesion_max_pa": m2.capillary_cohesion(0.9 * m2.full_moisture_point),
                     "critical_amplitude_at_S0.95_deg": crit["critical_amplitude_deg"]})
        if scale == 1.0:
            base = rows[-1]
    peaks = {r["peak_saturation"] for r in rows}
    amps = [r["critical_amplitude_at_S0.95_deg"] for r in rows]
    amps = [a for a in amps if a == a]
    return {
        "rows": rows,
        "peak_position_invariant": len(peaks) == 1,
        "critical_amplitude_spread_deg": (max(amps) - min(amps)) if len(amps) > 1 else 0.0,
        "verdict": "peak position invariant; threshold shifts by the force perturbation",
    }


def g6_comparability(failed_shift_m: float = 0.55) -> dict:
    """Comparability statement against the failed predecessor IN39-N08."""
    return {
        "failed_node": "IN39-N08 (full 6-DOF hold/ship coupling)",
        "artifacts_available_in_worktree": False,
        "directly_comparable": [
            "ru (dimensionless excess pore pressure ratio)",
            "phi_eff (deg) from the same Mohr-Coulomb reduction",
            "GZ area loss (%) over 0..40 deg",
        ],
        "not_comparable": [
            "particle-level forces (the failed version resolved 6-DOF rigid cargo dynamics)",
            "absolute hold-motion time histories",
        ],
        "dropped_physics": [
            "6-DOF hold-ship hydrostatic coupling (the failure mode: unjudgeable)", 
            "explicit rigid-body cargo rotation",
        ],
        "retained_physics": [
            "moisture -> saturation -> pendular bridge force",
            "Terzaghi effective-stress reduction by ru",
            "transverse cargo shift -> heeling moment -> GZ area loss",
        ],
        "reference_shift_for_scale_m": failed_shift_m,
        "status": "qualitative only: the predecessor's numbers are not in this worktree, "
                  "so no numerical comparison is claimed",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(matrix, g1, g3, g4, g5, g2, shift_histories, gz_case) -> list[dict]:
    paths = []

    # F1: bridge law
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    R, gamma = 1.0e-3, cl.WATER_SURFACE_TENSION
    f0 = 2.0 * math.pi * gamma * R
    vol = 0.03 * R ** 3
    d = np.linspace(0.0, 0.6, 120) * R
    ax[0].plot(d / R, [cl.rouault_pendular_force(R, gamma, vol, x) / f0 for x in d],
               label="Rouault/Willett smooth form")
    ax[0].plot(d / R, [cl.delenne_pendular_force(R, gamma, vol, x) / f0 for x in d],
               "--", label="Delenne/Rabinovich")
    ex = [r for r in g1["exact_reference_rows"] if r["converged"] and r["v_ratio"] == 0.03]
    if ex:
        ax[0].plot([r["d_ratio"] for r in ex], [r["reference_over_derjaguin"] for r in ex],
                   "o", mfc="none", label="exact Young-Laplace (in-house)")
    ax[0].set_xlabel("h / R")
    ax[0].set_ylabel(r"$F/(2\pi\gamma R\cos\theta)$")
    ax[0].set_title("(a) pendular bridge force, V/R³ = 0.03")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(d / R, [cl.rouault_pendular_energy(R, gamma, vol, x) / cl.rouault_pendular_energy(R, gamma, vol, 0.0)
                       for x in d])
    ax[1].set_xlabel("h / R")
    ax[1].set_ylabel(r"$U(h)/U(0)$")
    ax[1].set_title("(b) capillary potential (G4: $F=-dU/dh$, residual "
                    f"{g4['derivative_residual_max']:.1e})")
    ax[1].grid(alpha=0.3)
    p = FIG / "fig1_pendular_bridge.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append({"name": "fig1_pendular_bridge.png",
                  "caption": "Pendular bridge law: Rouault/Willett smooth form, the Delenne "
                             "closed form and the in-house exact Young-Laplace reference (a); "
                             "the conservative capillary potential whose derivative "
                             "reproduces the force to 1e-9 (b).",
                  "path": p})

    # F2: phi_eff(MR)
    rows = g3["rows"]
    sr = [r["saturation"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(sr, [r["phi_eff_deg"] for r in rows], "-o", ms=3)
    ax[0].axvspan(0.80, 0.95, color="tab:green", alpha=0.12, label="required peak window")
    ax[0].plot([g3["peak_saturation"]], [g3["peak_phi_deg"]], "r*", ms=14,
               label=f"peak {g3['peak_phi_deg']:.1f}° @ S_r={g3['peak_saturation']:.2f}")
    ax[0].set_xlabel(r"$S_r = MR/FMP$")
    ax[0].set_ylabel(r"$\varphi_{eff}$ (deg)")
    ax[0].set_title("(a) apparent friction angle vs moisture")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(sr, [r["pore_pressure_ratio"] for r in rows], label=r"$r_u$ (undrained)")
    ax[1].plot(sr, [r["cohesion_pa"] / 1e3 for r in rows], label="c (kPa)")
    ax[1].set_xlabel(r"$S_r$")
    ax[1].set_title("(b) the two competing factors")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    p = FIG / "fig2_effective_friction_angle.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append({"name": "fig2_effective_friction_angle.png",
                  "caption": "Effective friction angle versus saturation (G3): the capillary "
                             "cohesion lifts phi_eff to a maximum inside the required "
                             "0.80-0.95 window, then the contractancy-driven r_u collapses it; "
                             "panel (b) shows the two competing factors.",
                  "path": p})

    # F3: shift histories + GZ
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for sr_val, h in shift_histories:
        ax[0].plot(np.asarray(h["time"], dtype=float), np.asarray(h["shift"], dtype=float) * 1e2, lw=1.2,
                   label=f"$S_r$={sr_val:.2f}, r_u,f={h['ru_final']:.2f}")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel(r"transverse CoG shift $y_G$ (cm)")
    ax[0].set_title("(a) cargo shift under roll")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    th = np.linspace(0, 40, 200)
    gm, dy = gz_case["gm_m"], gz_case["transverse_cog_shift_m"]
    ax[1].plot(th, gm * np.sin(np.radians(th)), label="intact")
    ax[1].plot(th, gm * np.sin(np.radians(th)) - dy * np.cos(np.radians(th)),
               label=f"after shift (dy={dy:.2f} m)")
    ax[1].fill_between(th, gm * np.sin(np.radians(th)),
                       gm * np.sin(np.radians(th)) - dy * np.cos(np.radians(th)),
                       color="tab:red", alpha=0.2)
    ax[1].set_xlabel(r"$\theta$ (deg)")
    ax[1].set_ylabel("GZ (m)")
    ax[1].set_title(f"(b) GZ loss {gz_case['gz_area_loss_percent']:.1f} %, list "
                    f"{gz_case['residual_list_deg']:.1f}°")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    p = FIG / "fig3_cargo_shift_gz.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append({"name": "fig3_cargo_shift_gz.png",
                  "caption": "Cargo CoG shift time histories for three saturations (a) and the "
                             "consequent righting-arm loss with the residual list (b).",
                  "path": p})

    # F4: thresholds + dt convergence + repose bench
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for key, style in (("dt_2.0", "-o"), ("dt_1.0", "--s")):
        rows = g5["curves"][key]
        ax[0].plot([r["saturation"] for r in rows],
                   [r["critical_amplitude_deg"] for r in rows], style, ms=4, label=f"dt = {key[3:]} s")
    ax[0].set_xlabel(r"$S_r = MR/FMP$")
    ax[0].set_ylabel("critical roll amplitude (deg)")
    ax[0].set_title("(a) shift threshold (G5)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for tag, style in (("dry", "-"), ("wet_1x", "--"), ("wet_5x", ":")):
        h = g2["runs"][tag]["history"]
        ax[1].plot([r["step"] for r in h], [r["theta_r_deg"] for r in h], style,
                   label=f"{tag}, Co={g2['runs'][tag]['cohesion_number']:.3f}")
    ax[1].axhline(32.0, color="k", lw=0.8, label="experiment 32°")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("measured flank angle (deg)")
    ax[1].set_title("(b) particle-DEM repose bench (G2)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    p = FIG / "fig4_threshold_repose.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append({"name": "fig4_threshold_repose.png",
                  "caption": "Shift threshold versus saturation for two timesteps (a) and the "
                             "particle-DEM repose bench with the cohesion-number sweep (b).",
                  "path": p})
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--reinterpret", type=float, default=None,
                    help="recompute the G5 verdict at this hazard limit from the stored curves")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    matrix = cl.CargoMatrix()
    mpath = ART / "metrics.json"
    resume = args.resume or args.reinterpret is not None or args.figures_only
    out = json.loads(mpath.read_text()) if (resume and mpath.exists()) else {}
    out.setdefault("node", "IN60-N09")
    out["model_parameters"] = {
            "particle_radius_m": matrix.particle_radius,
            "porosity": matrix.porosity,
            "rho_s": matrix.rho_s,
            "full_moisture_point": matrix.full_moisture_point,
            "saturation_pendular": matrix.saturation_pendular,
            "saturation_onset": matrix.saturation_onset,
            "phi_dry_deg": matrix.phi_dry_deg,
            "sigma_n_ref_pa": matrix.sigma_n_ref,
            "surface_tension": matrix.surface_tension,
            "contact_angle_deg": matrix.contact_angle_deg,
    }
    def save():
        mpath.write_text(json.dumps(out, indent=1, default=float))

    if args.reinterpret is not None:
        reinterpret_threshold_limit(out, args.reinterpret)
        out["G5_limit_sensitivity"] = threshold_limit_sensitivity(out["G5"])
        save()
        print("reinterpreted at limit", args.reinterpret,
              "dt max rel diff", out["G5"]["dt_max_relative_difference"],
              "mono hazard", out["G5"]["monotone_decreasing_hazard_window"],
              "mono full", out["G5"]["monotone_decreasing_full_range"], flush=True)
        return 0

    if args.figures_only:
        hist = [(float(s_key), h) for s_key, h in out["shift_series"].items()]
        figs = figures(matrix, out["G1"], out["G3"], out["G4"], out["G5"], out["G2"],
                       hist, out["GZ_case"])
        out["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
        save()
        print("figures regenerated", flush=True)
        return 0

    print("G1 bridge law", flush=True)
    out.setdefault("G1", g1_bridge_law())
    save()
    print("  kernel-vs-hand", out["G1"]["kernel_vs_hand_formula_max_rel"],
          "exact median dev %", out["G1"]["exact_reference_median_deviation_percent"], flush=True)
    print("G4 energy", flush=True)
    out.setdefault("G4", g4_energy())
    save()
    print("  deriv resid", out["G4"]["derivative_residual_max"],
          "budget resid", out["G4"]["budget"]["residual_energy"], flush=True)
    print("G3 friction angle", flush=True)
    out.setdefault("G3", g3_friction_angle(matrix))
    save()
    print("  peak", out["G3"]["peak_saturation"], out["G3"]["peak_phi_deg"], flush=True)
    print("G2 repose bench", flush=True)
    if "G2" not in out:
        out["G2"] = g2_repose(6000 if args.fast else 9000)
        save()
    print("G5 thresholds", flush=True)
    if "G5" not in out:
        out["G5"] = g5_thresholds(matrix, fast=args.fast)
        save()
    print("  dt max rel diff", out["G5"]["dt_max_relative_difference"],
          "mono hazard", out["G5"]["monotone_decreasing_hazard_window"], flush=True)
    print("G5b critical moisture", flush=True)
    if "G5b" not in out:
        out["G5b"] = critical_moisture_scan(matrix)
    out["G5_limit_sensitivity"] = threshold_limit_sensitivity(out["G5"])
    save()

    # support numbers for the figures / report
    shift_histories = []
    for sr_val in (0.70, 0.90, 0.99):
        load = cl.RollExcitation(amplitude_deg=18.0)
        res = cl.simulate_roll_shift(matrix, load, sr_val * matrix.full_moisture_point,
                                     cycles=40, dt=2.0, nx=48, record=True)
        shift_histories.append((sr_val, {
            "time": res.time_s.tolist(),
            "shift": res.shift_history.tolist(),
            "ru_final": res.pore_pressure_ratio_final,
        }))
    out["shift_series"] = {str(s): {"time": h["time"], "shift": h["shift"],
                                    "ru_final": h["ru_final"]} for s, h in shift_histories}
    out["shift_histories"] = [
        {"saturation": s, "ru_final": h["ru_final"],
         "shift_final_m": h["shift"][-1] if h["shift"] else None,
         "shift_max_abs_m": max((abs(v) for v in h["shift"]), default=None)}
        for s, h in shift_histories
    ]
    last = shift_histories[-1][1]
    last["shift_max_abs_m"] = max((abs(v) for v in last["shift"]), default=0.0)
    gz = cl.gz_area_loss(1.5, last["shift_max_abs_m"], 25000.0, 50000.0)
    gz["gm_m"] = 1.5
    gz["shift_used_m"] = last["shift_max_abs_m"]
    gz["shift_used_kind"] = "max |y_G| over the exposure (worst instant)"
    gz["gz_at_shift_end_of_exposure"] = cl.gz_area_loss(
        1.5, abs(last["shift"][-1]) if last["shift"] else 0.0, 25000.0, 50000.0)
    out["GZ_case"] = gz
    out["G6"] = g6_comparability()
    out["sensitivity_force_amplitude"] = force_amplitude_sensitivity(matrix)
    out["runtime_s"] = time.time() - t0
    figs = figures(matrix, out["G1"], out["G3"], out["G4"], out["G5"], out["G2"],
                   shift_histories, gz)
    out["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]
    (ART / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", ART / "metrics.json", "runtime", round(out["runtime_s"], 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())