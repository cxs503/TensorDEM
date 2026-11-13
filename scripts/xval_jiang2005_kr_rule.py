#!/usr/bin/env python
"""XVAL: Jiang-2005 quadratic rolling-stiffness rule vs analytic references.

Anchor: ``XVAL-DEM-ELASTIC-PLASTIC-ROLLING-SPRING-KR-225-MUR2-KN-R2``
(ledger slug truncated to ``...-KR-225-MU``).  The anchor is the *stiffness
rule*

    k_r = 2.25 * mu_r**2 * k_n * R**2      (Jiang, Yu & Harris 2005)

as opposed to the linear ``k_r = 2.25 * mu_r * k_n * R**2`` that
``docs/XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS.md`` attributed to the same
reference.  This harness machines-checks the rule and its consequences against
closed-form references, driving only the library contact law
(``tensordem.rolling_resistance``):

  A. rule table (k_r vs mu_r, both rules, two parameter sets, ratio column);
  B. elastic rolling-mode frequency, analytic ``sqrt(k_r/(m R^2 (1+I/mR^2)))``
     vs integrated through the library EPSD torque (both rules);
  C. plastic branch: torque ramp to the limit, yield time, plastic dissipation
     identity ``W = mu_r |F_n| r_eff (theta - theta_y)``;
  D. single-sphere incline harness (``simulate_sphere_on_incline``): steady
     downhill acceleration vs analytic, under the Jiang rule;
  E. static branch: hold at 0.8 * arctan(mu_r) must not creep.

Outputs (``--output DIR``): ``summary.json``, ``rule_table.csv``,
``figures/*.png``.

Usage:
    PYTHONPATH=src python scripts/xval_jiang2005_kr_rule.py --output runs/xval_kr_rule
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.rolling_resistance import (  # noqa: E402
    JIANG2005_ANCHOR_STATE,
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    fit_acceleration,
    incline_rolling_acceleration,
    pure_rolling_feasible,
    rolling_mode_frequency,
    rolling_resistance_torque,
    rolling_stiffness,
    rolling_stiffness_table,
    rolling_yield_angle,
    rolling_yield_time,
    simulate_sphere_on_incline,
)

MU_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
ACCEPT = 0.15   # owner precision standard: >15 % is not acceptable
TARGET = 0.03   # this harness' internal pass mark for closures it controls


# ---------------------------------------------------------------------------
# A. rule table
# ---------------------------------------------------------------------------

def study_rule_table() -> dict:
    anchor = dict(JIANG2005_ANCHOR_STATE)
    sets = {
        "anchor_kr225": dict(k_n=anchor["k_n"], radius=anchor["radius"]),
        "ice_dem_harness": dict(k_n=2.0e4, radius=0.01),
    }
    out = {"parameter_sets": sets, "mu_values": list(MU_SWEEP), "tables": {}}
    for name, ps in sets.items():
        rows = rolling_stiffness_table(MU_SWEEP, k_n=ps["k_n"], radius=ps["radius"])
        out["tables"][name] = rows
    kr_anchor = rolling_stiffness(anchor["k_n"], anchor["mu_r"], anchor["radius"])
    out["anchor_state"] = anchor
    out["k_r_at_anchor_state"] = kr_anchor
    out["anchor_identity_rel_error"] = abs(kr_anchor - 225.0) / 225.0
    # ratio == mu_r is the analytic signature of the mu_r^2 rule
    out["max_ratio_error"] = max(
        abs(r["ratio_jiang_over_linear"] - r["mu_r"]) / r["mu_r"]
        for rows in out["tables"].values() for r in rows
    )
    return out


# ---------------------------------------------------------------------------
# B. elastic rolling-mode frequency (library EPSD torque, 1-DOF rolling)
# ---------------------------------------------------------------------------

def _upward_zero_crossings(y: torch.Tensor, dt: float) -> list:
    """Times of upward zero crossings of ``y``, linearly interpolated."""
    idx = torch.nonzero((y[1:] > 0.0) & (y[:-1] <= 0.0)).flatten()
    times = []
    for i in idx.tolist():
        y0, y1 = float(y[i - 1]), float(y[i])
        frac = 0.0 if y1 == y0 else -y0 / (y1 - y0)
        times.append((i - 1 + frac) * dt)
    return times


def _roll_oscillator(mu_r: float, rule: str, *, k_n: float, radius: float,
                     normal_force: float, theta0: float, dt: float,
                     duration: float) -> dict:
    """Integrate ``I_eff theta_ddot = M_r(omega, state)`` with the library torque.

    ``I_eff = m R^2 (1 + I/(mR^2))`` -- the no-slip rolling inertia, so the
    analytic reference is exactly ``rolling_mode_frequency``.
    """
    r_eff = radius
    # sphere mass from a representative ice density; only I_eff/k_r matters
    density = 900.0
    mass = 4.0 / 3.0 * math.pi * radius ** 3 * density
    inertia_factor = 0.4
    i_eff = mass * radius ** 2 * (1.0 + inertia_factor)
    k_r = rolling_stiffness(k_n, mu_r, radius, rule=rule)
    limit = mu_r * normal_force * r_eff
    theta_y = limit / k_r
    if theta0 >= theta_y:
        raise ValueError("initial angle must stay in the elastic range")

    cfg = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
    theta = torch.tensor([theta0], dtype=torch.float64)
    omega = torch.zeros(1, 3, dtype=torch.float64)
    # Stress-free EPSD state is defined at theta = 0, so an initial angle
    # theta0 needs the matching initial elastic torque -k_r * theta0.
    state = torch.zeros(1, 3, dtype=torch.float64)
    state[0, 2] = -k_r * theta0
    f_n = torch.tensor([normal_force], dtype=torch.float64)
    r_t = torch.tensor([r_eff], dtype=torch.float64)

    steps = int(round(duration / dt))
    hist = torch.empty(steps, dtype=torch.float64)
    thetas = torch.empty(steps, dtype=torch.float64)
    for i in range(steps):
        torque, state = rolling_resistance_torque(
            omega, f_n, r_t, cfg, dt=dt, torque_state=state
        )
        m_z = torque[0, 2]
        omega = omega.clone()
        omega[:, 2] = omega[:, 2] + (m_z / i_eff) * dt
        theta = theta + omega[:, 2] * dt
        hist[i] = m_z
        thetas[i] = theta[0]

    # Frequency from upward zero crossings of theta, located by linear
    # interpolation *inside* the bracketing step.  Rounding crossing times to
    # integer steps would quantise the period to a multiple of dt and can fake
    # exact agreement with the analytic value, so interpolation is required.
    cross_t = _upward_zero_crossings(thetas, dt)
    if len(cross_t) < 3:
        raise RuntimeError("not enough zero crossings to measure the frequency")
    seg = [cross_t[i + 1] - cross_t[i] for i in range(len(cross_t) - 1)]
    period = sum(seg) / len(seg)
    measured = 2.0 * math.pi / period
    analytic = rolling_mode_frequency(k_r, mass, radius, inertia_factor=inertia_factor)
    return {
        "rule": rule,
        "mu_r": mu_r,
        "k_r": k_r,
        "limit": limit,
        "theta_y": theta_y,
        "theta0": theta0,
        "period": period,
        "freq_measured": measured,
        "freq_analytic": analytic,
        "rel_error": abs(measured - analytic) / analytic,
        "max_torque_over_limit": float(hist.abs().max()) / limit,
        "dt": dt,
        "steps_per_period": period / dt,
        "theta_max_ratio_to_yield": float(thetas.abs().max()) / theta_y,
    }


def study_mode_frequency() -> dict:
    rows = []
    for mu_r in (0.1, 0.2, 0.3, 0.5):
        for rule in ("linear", "jiang2005"):
            k_r = rolling_stiffness(1.0e6, mu_r, 0.05, rule=rule)
            mass = 4.0 / 3.0 * math.pi * 0.05 ** 3 * 900.0
            f_n = mass * 9.81
            theta_y = rolling_yield_angle(mu_r, f_n, 0.05, k_r)
            f_n_ana = rolling_mode_frequency(k_r, mass, 0.05)
            t_ana = 2.0 * math.pi / f_n_ana
            # deliberately NOT commensurate with t_ana (1973.7 != 2000); the
            # refined run is kept for one mu_r only (cost control)
            divisors = (1973.7, 3947.4) if mu_r == 0.2 else (1973.7,)
            for divisor in divisors:
                dt = t_ana / divisor
                row = _roll_oscillator(
                    mu_r, rule, k_n=1.0e6, radius=0.05, normal_force=f_n,
                    theta0=0.5 * theta_y, dt=dt, duration=5.0 * t_ana,
                )
                row["dt_divisor"] = divisor
                rows.append(row)
    # frequency ratio between the rules must be sqrt(mu_r)
    ratio_rows = []
    for mu_r in (0.1, 0.2, 0.3, 0.5):
        lin = next(r for r in rows if r["mu_r"] == mu_r and r["rule"] == "linear"
                   and r["dt_divisor"] == 1973.7)
        jia = next(r for r in rows if r["mu_r"] == mu_r and r["rule"] == "jiang2005"
                   and r["dt_divisor"] == 1973.7)
        ratio = jia["freq_measured"] / lin["freq_measured"]
        ratio_rows.append({
            "mu_r": mu_r,
            "freq_linear": lin["freq_measured"],
            "freq_jiang2005": jia["freq_measured"],
            "ratio_measured": ratio,
            "ratio_analytic_sqrt_mu": math.sqrt(mu_r),
            "rel_error": abs(ratio - math.sqrt(mu_r)) / math.sqrt(mu_r),
        })
    conv = {}
    for mu_r in (0.2,):
        for rule in ("linear", "jiang2005"):
            sel = sorted([r for r in rows if r["mu_r"] == mu_r and r["rule"] == rule],
                         key=lambda r: -r["dt"])
            if len(sel) == 2 and sel[0]["rel_error"] > 0.0:
                conv[f"{rule}_mu{mu_r}"] = {
                    "err_coarse": sel[0]["rel_error"],
                    "err_fine": sel[1]["rel_error"],
                    "ratio": sel[0]["rel_error"] / sel[1]["rel_error"],
                    "dt_refined_by": sel[0]["dt"] / sel[1]["dt"],
                }
    return {
        "rows": rows,
        "convergence": conv,
        "max_rel_error": max(r["rel_error"] for r in rows),
        "ratio_rows": ratio_rows,
        "max_ratio_rel_error": max(r["rel_error"] for r in ratio_rows),
        "max_torque_over_limit": max(r["max_torque_over_limit"] for r in rows),
    }


# ---------------------------------------------------------------------------
# C. plastic branch: yield time + dissipation identity
# ---------------------------------------------------------------------------

def study_plastic(mu_r: float = 0.2, omega_roll: float = 1.0) -> dict:
    k_n, radius = 1.0e6, 0.05
    density = 900.0
    mass = 4.0 / 3.0 * math.pi * radius ** 3 * density
    f_n = mass * 9.81
    limit = mu_r * f_n * radius
    rows = []
    for rule in ("linear", "jiang2005"):
        k_r = rolling_stiffness(k_n, mu_r, radius, rule=rule)
        cfg = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
        theta_y = rolling_yield_angle(mu_r, f_n, radius, k_r)
        t_y = rolling_yield_time(mu_r, f_n, radius, k_r, omega_roll)
        # dt tied to the *elastic period* only, deliberately not to t_y, so the
        # measured saturation step cannot agree with the analytic value by
        # construction; the event time is then interpolated inside the step.
        period = 2.0 * math.pi / rolling_mode_frequency(k_r, mass, radius)
        dt = min(period / 1997.3, t_y / 397.3)
        steps = int(round(4.0 * t_y / dt))
        omega = torch.tensor([[0.0, 0.0, -omega_roll]], dtype=torch.float64)  # rolling spin
        f_nt = torch.tensor([f_n], dtype=torch.float64)
        r_t = torch.tensor([radius], dtype=torch.float64)
        state = torch.zeros(1, 3, dtype=torch.float64)
        torque_hist = torch.empty(steps, dtype=torch.float64)
        theta = 0.0
        plastic_work = 0.0
        saturate_step = None
        t_saturate = None
        prev_mag = 0.0
        for i in range(steps):
            torque, state = rolling_resistance_torque(
                omega, f_nt, r_t, cfg, dt=dt, torque_state=state
            )
            mag = float(torque.abs().max())
            torque_hist[i] = mag
            elastic_increment = mag - prev_mag
            if saturate_step is None and mag >= limit * (1.0 - 1e-12):
                saturate_step = i + 1
                # linear-interpolate the instant |M| crosses the limit
                frac = (limit - prev_mag) / (mag - prev_mag) if mag > prev_mag else 0.0
                t_saturate = (i + frac) * dt
            else:
                plastic_work += limit * max(abs(omega_roll) * dt - elastic_increment / k_r, 0.0)
            prev_mag = mag
            theta += abs(omega_roll) * dt
        t_sat = t_saturate if t_saturate is not None else float("nan")
        rows.append({
            "rule": rule,
            "k_r": k_r,
            "limit": limit,
            "theta_y": theta_y,
            "t_y_analytic": t_y,
            "saturate_step": saturate_step,
            "t_saturate_measured": t_sat,
            "yield_time_rel_error": abs(t_sat - t_y) / t_y,
            "yield_time_in_steps": (t_sat - t_y) / dt,
            "max_torque_over_limit": float(torque_hist.max()) / limit,
            "plastic_work_measured": plastic_work,
            "plastic_work_analytic": limit * max(theta - theta_y, 0.0),
            "plastic_work_rel_error": abs(plastic_work - limit * max(theta - theta_y, 0.0))
            / max(limit * max(theta - theta_y, 0.0), 1e-30),
            "dt": dt,
        })
    return {
        "mu_r": mu_r,
        "omega_roll": omega_roll,
        "rows": rows,
        "max_yield_time_rel_error": max(r["yield_time_rel_error"] for r in rows),
        "max_plastic_work_rel_error": max(r["plastic_work_rel_error"] for r in rows),
        "max_torque_over_limit": max(r["max_torque_over_limit"] for r in rows),
    }


# ---------------------------------------------------------------------------
# D. incline harness: steady acceleration under the Jiang rule
# ---------------------------------------------------------------------------

def study_incline(mu_r: float = 0.2) -> dict:
    k_n, radius, dt = 2.0e4, 0.01, 5.0e-5
    density = 900.0
    mass = 4.0 / 3.0 * math.pi * radius ** 3 * density
    f_n_ref = mass * 9.81
    theta_c = critical_incline_angle_deg(mu_r)
    angles = sorted({1.25 * theta_c, 1.5 * theta_c, 2.0 * theta_c,
                     max(3.0 * theta_c, 8.0), 12.0, 15.0})
    angles = [a for a in angles if pure_rolling_feasible(a, mu_r, 0.9)]
    rows = []
    for rule in ("linear", "jiang2005"):
        k_r = rolling_stiffness(k_n, mu_r, radius, rule=rule)
        cfg_roll = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
        res = simulate_sphere_on_incline(
            angles, cfg_roll,
            config=SphereInclineConfig(radius=radius, k_n=k_n, dt=dt),
            duration=0.12, settle_time=0.04,
        )
        acc, _ = fit_acceleration(res)
        for j, ang in enumerate(angles):
            a_sim = float(acc[j])
            a_ref = incline_rolling_acceleration(ang, mu_r)
            rows.append({
                "rule": rule,
                "angle_deg": ang,
                "k_r": k_r,
                "a_sim": a_sim,
                "a_analytic": a_ref,
                "rel_error": abs(a_sim - a_ref) / abs(a_ref) if a_ref != 0 else abs(a_sim),
            })
    return {
        "mu_r": mu_r,
        "theta_c_deg": theta_c,
        "rows": rows,
        "max_rel_error": max(r["rel_error"] for r in rows),
        "max_rel_error_jiang": max(r["rel_error"] for r in rows if r["rule"] == "jiang2005"),
        "normal_force_ref": f_n_ref,
    }


# ---------------------------------------------------------------------------
# E. static branch: hold below the critical angle
# ---------------------------------------------------------------------------

def study_static(mu_r: float = 0.2) -> dict:
    k_n, radius, dt = 2.0e4, 0.01, 5.0e-5
    theta_c = critical_incline_angle_deg(mu_r)
    angle = 0.8 * theta_c
    rows = []
    for rule in ("linear", "jiang2005"):
        k_r = rolling_stiffness(k_n, mu_r, radius, rule=rule)
        cfg_roll = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
        res = simulate_sphere_on_incline(
            [angle], cfg_roll,
            config=SphereInclineConfig(radius=radius, k_n=k_n, dt=dt),
            duration=0.12, settle_time=0.04,
        )
        v = res["v"][-1000:, 0]
        rows.append({
            "rule": rule,
            "k_r": k_r,
            "angle_deg": angle,
            "theta_c_deg": theta_c,
            "mean_abs_v": float(v.abs().mean()),
            "max_abs_v": float(v.abs().max()),
            "displacement": float(res["x"][-1, 0]),
            "unresisted_accel": 9.81 * math.sin(math.radians(angle)),
        })
    return {"mu_r": mu_r, "theta_c_deg": theta_c, "rows": rows,
            "max_mean_abs_v": max(r["mean_abs_v"] for r in rows)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(summary: dict, fig_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # (a) k_r(mu_r): both rules + anchor state
    tab = summary["rule_table"]["tables"]["anchor_kr225"]
    mu = [r["mu_r"] for r in tab]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(mu, [r["k_r_linear"] for r in tab], "o-", label=r"linear $2.25\mu_r k_n R^2$")
    ax.loglog(mu, [r["k_r_jiang2005"] for r in tab], "s-",
              label=r"Jiang-2005 $2.25\mu_r^2 k_n R^2$")
    a = summary["rule_table"]["anchor_state"]
    ax.plot([a["mu_r"]], [summary["rule_table"]["k_r_at_anchor_state"]], "k*", ms=14,
            label=rf"anchor state $\mu_r$={a['mu_r']}, $k_n$={a['k_n']:.0e}, R={a['radius']} -> 225 N·m/rad")
    ax.set_xlabel(r"$\mu_r$"); ax.set_ylabel(r"$k_r$  [N·m/rad]")
    ax.set_title("A. Rolling stiffness rules (ratio = $\\mu_r$)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    p = fig_dir / "fig_a_stiffness_rules.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    out["fig_a_stiffness_rules"] = p

    # (b) oscillation: theta(t) for both rules at mu_r=0.3
    sel = [r for r in summary["mode_frequency"]["rows"] if r["mu_r"] == 0.3]
    if sel:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for r in summary["mode_frequency"]["rows"]:
            if r["mu_r"] != 0.3:
                continue
            t = [i * r["dt"] for i in range(int(3.0 * round(r["period"] / r["dt"])))]
            theta = [r["theta0"] * math.cos(2.0 * math.pi * t[i] / r["period"]) for i in range(len(t))]
            ax.plot(t, theta, lw=1.4,
                    label=f"{r['rule']}: f={r['freq_measured']:.2f} Hz (analytic {r['freq_analytic']:.2f})")
        ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_title(r"B. Elastic rolling mode, $\mu_r$=0.3 (reconstructed from measured period)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        p = fig_dir / "fig_b_rolling_mode.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        out["fig_b_rolling_mode"] = p

    # (c) torque ramp to the plastic limit
    plast = summary["plastic"]["rows"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for r in plast:
        n = r["saturate_step"] or 1
        t = [(i + 1) * r["dt"] * 1000.0 for i in range(n)]
        m = [min(r["k_r"] * summary["plastic"]["omega_roll"] * (i + 1) * r["dt"], r["limit"])
             for i in range(n)]
        ax.plot(t, [1e3 * v for v in m], lw=1.6, label=f"{r['rule']} ($k_r$={r['k_r']:.1f})")
    ax.axhline(1e3 * plast[0]["limit"], ls="--", c="k", lw=1,
               label=rf"plastic limit $\mu_r |F_n| r_{{eff}}$ = {plast[0]['limit']:.2f} N·m")
    ax.set_xlabel(r"t [ms]"); ax.set_ylabel(r"$|M_r|$ [mN·m]")
    ax.set_title(r"C. Torque ramp to yield, $\mu_r$=0.2, $\omega$=1 rad/s")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = fig_dir / "fig_c_torque_ramp.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    out["fig_c_torque_ramp"] = p

    # (d) incline acceleration parity
    inc = summary["incline"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for rule, marker in (("linear", "o"), ("jiang2005", "s")):
        pts = [r for r in inc["rows"] if r["rule"] == rule]
        ax.plot([r["a_analytic"] for r in pts], [r["a_sim"] for r in pts], marker + "-",
                label=f"{rule} (max err {max(r['rel_error'] for r in pts)*100:.3f}%)")
    lim = [0.0, max(r["a_analytic"] for r in inc["rows"]) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel(r"analytic $a$ [m/s$^2$]"); ax.set_ylabel(r"simulated $a$ [m/s$^2$]")
    ax.set_title(r"D. Incline parity, $\mu_r$=0.2 (steady rolling)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = fig_dir / "fig_d_incline_parity.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    out["fig_d_incline_parity"] = p
    return {k: str(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/xval_kr_rule")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rule_table = study_rule_table()
    mode = study_mode_frequency()
    plastic = study_plastic()
    incline = study_incline()
    static = study_static()

    checks = {
        "A_anchor_identity_225": rule_table["anchor_identity_rel_error"],
        "A_ratio_equals_mu_r": rule_table["max_ratio_error"],
        "B_mode_frequency": mode["max_rel_error"],
        "B_frequency_ratio_sqrt_mu": mode["max_ratio_rel_error"],
        "B_no_excess_torque": max(mode["max_torque_over_limit"] - 1.0, 0.0),
        "C_yield_time": plastic["max_yield_time_rel_error"],
        "C_plastic_dissipation": plastic["max_plastic_work_rel_error"],
        "C_no_excess_torque": max(plastic["max_torque_over_limit"] - 1.0, 0.0),
        "D_incline_acceleration_jiang": incline["max_rel_error_jiang"],
        "E_static_hold_velocity": static["max_mean_abs_v"] / 0.1,
    }
    verdicts = {k: {"value": float(v), "passes_3pct": bool(v <= TARGET),
                    "passes_15pct": bool(v <= ACCEPT)} for k, v in checks.items()}

    summary = {
        "schema": "tensordem.xval-kr-rule/1",
        "anchor": "XVAL-DEM-ELASTIC-PLASTIC-ROLLING-SPRING-KR-225-MUR2-KN-R2",
        "reference": "Jiang, M.J., Yu, H.-S., Harris, D. (2005). Comput. Geotech. 32, 340-357.",
        "rule": "k_r = 2.25 * mu_r**2 * k_n * R**2",
        "rule_table": rule_table,
        "mode_frequency": mode,
        "plastic": plastic,
        "incline": incline,
        "static": static,
        "checks": verdicts,
        "overall": "PASS" if all(v["passes_15pct"] for v in verdicts.values()) else "FAIL",
        "all_within_3pct": all(v["passes_3pct"] for v in verdicts.values()),
    }

    if not args.no_figures:
        summary["figures"] = make_figures(summary, out / "figures")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with (out / "rule_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter_set", "mu_r", "k_r_linear", "k_r_jiang2005", "ratio_jiang_over_linear"])
        for name, rows in rule_table["tables"].items():
            for r in rows:
                w.writerow([name, r["mu_r"], r["k_r_linear"], r["k_r_jiang2005"],
                            r["ratio_jiang_over_linear"]])

    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("overall", "all_within_3pct", "checks")}, indent=2))
    print(f"\nartifacts -> {out.resolve()}")
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())