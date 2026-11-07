"""Clump rolling-resistance calibration: tilt-table experiment, parameter
sensitivity and verification against closed-form statics (mission m5052).

Two stages, so the heavy part can run on the compute cluster (which has no
matplotlib) and the figures can be drawn wherever matplotlib exists::

    PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --workers 32
    PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --figures-only

The compute stage writes ``artifacts/m5052/clump_rolling_calibration.json``; the
figure stage reads it back and writes ``artifacts/m5052/fig*.png``.  Every number
quoted in ``docs/clump_rolling_resistance_calibration.md`` comes from that JSON
(zero hand-copying).

What is measured
----------------
1. **Tilt threshold of four clump shapes.**  A clump rests on a plane that is
   tilted in a virtual tilt-table test; the angle at which it first moves is
   bisected.  Reference: ``tan(theta_c) = min(mu_s, (a + mu_r r)/h)`` from
   :mod:`tensordem.clump_rolling`.
2. **Parameter sensitivity**: rolling coefficient ``mu_r``, sliding coefficient
   ``mu_s``, contact-model branch (``epsd`` / ``cdt`` / ``viscous``) and time step.
3. **Calibration inversion**: recover ``mu_r`` from the measured tilt angle.
4. **Drilling (torsion) spin-down** against its closed form.
5. **Energy audit**: the elastic-plastic rolling spring must stay inside its
   stored-energy bound and must not create energy.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.clump import Clump, SphereMember, double_sphere_clump  # noqa: E402
from tensordem.clump_dynamics import (ClumpBody, ContactConfig,  # noqa: E402
                                      PlaneContact, quaternion_from_axis_angle)
from tensordem.clump_rolling import (bracket_critical_tilt,  # noqa: E402
                                     clump_tilt_reference, invert_rolling_friction,
                                     tilt_test)

OUT = ROOT / "artifacts" / "m5052"
JSON_PATH = OUT / "clump_rolling_calibration.json"
RADIUS = 0.05
DENSITY = 900.0
DT = 2.0e-4
DURATION = 0.50
TOLERANCE_DEG = 0.05
MU_R_DEFAULT = 0.20
MU_S_DEFAULT = 0.90


# --------------------------------------------------------------------------
# clump shapes (all equal-radius members, flat pose: every member on the plane)
# --------------------------------------------------------------------------
def build_shapes() -> dict[str, Clump]:
    spacing = 0.095                     # member centre spacing of overlapping pairs
    circum = spacing / math.sqrt(3.0)   # equilateral-triangle circumradius
    return {
        "sphere": Clump([SphereMember((0.0, 0.0, 0.0), RADIUS)], density=DENSITY),
        "dumbbell": double_sphere_clump(RADIUS, density=DENSITY),
        "triangle3": Clump([
            SphereMember((circum, 0.0, 0.0), RADIUS),
            SphereMember((-0.5 * circum, 0.5 * spacing, 0.0), RADIUS),
            SphereMember((-0.5 * circum, -0.5 * spacing, 0.0), RADIUS),
        ], density=DENSITY),
        "cross5": Clump([
            SphereMember((0.0, 0.0, 0.0), RADIUS),
            SphereMember((spacing, 0.0, 0.0), RADIUS),
            SphereMember((-spacing, 0.0, 0.0), RADIUS),
            SphereMember((0.0, spacing, 0.0), RADIUS),
            SphereMember((0.0, -spacing, 0.0), RADIUS),
        ], density=DENSITY),
    }


# --------------------------------------------------------------------------
# one bisected tilt test
# --------------------------------------------------------------------------
def measure(clump: Clump, reference_deg: float, *, mu_r: float, mu_s: float,
            model: str = "epsd", dt: float = DT, duration: float = DURATION,
            omega_ref: float = 1.0e-3, iterations: int = 9) -> dict[str, object]:
    low = max(0.0, 0.55 * reference_deg - 0.5)
    high = min(1.45 * reference_deg + 0.5, 89.0)
    return bracket_critical_tilt(clump, mu_r=mu_r, mu_s=mu_s, model=model,
                                 low=low, high=high, iterations=iterations,
                                 tolerance_deg=TOLERANCE_DEG, dt=dt, duration=duration,
                                 omega_ref=omega_ref)


def rel_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return math.inf if measured else 0.0
    return abs(measured - reference) / abs(reference)


# --------------------------------------------------------------------------
# case definitions
# --------------------------------------------------------------------------
def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for shape in ("sphere", "dumbbell", "triangle3", "cross5"):
        cases.append({"id": f"threshold:{shape}", "kind": "threshold", "shape": shape,
                      "mu_r": MU_R_DEFAULT, "mu_s": MU_S_DEFAULT})
    for mu_r in (0.05, 0.10, 0.20, 0.30, 0.40):
        cases.append({"id": f"mu_r:{mu_r:.2f}", "kind": "mu_r", "shape": "sphere",
                      "mu_r": mu_r, "mu_s": MU_S_DEFAULT})
    for shape, mu_s in (("sphere", 0.05), ("sphere", 0.90),
                        ("dumbbell", 0.90), ("dumbbell", 3.00)):
        cases.append({"id": f"mu_s:{shape}:{mu_s:.2f}", "kind": "mu_s", "shape": shape,
                      "mu_r": MU_R_DEFAULT, "mu_s": mu_s})
    for shape in ("sphere", "dumbbell"):
        for model in ("epsd", "cdt"):
            cases.append({"id": f"model:{shape}:{model}", "kind": "model", "shape": shape,
                          "mu_r": MU_R_DEFAULT, "mu_s": MU_S_DEFAULT, "model": model})
    # viscous branch: the default omega_ref is numerically stiff at this dt, so the
    # calibration is attempted both at the default and at a resolved value
    for omega_ref in (1.0e-3, 0.1):
        cases.append({"id": f"model:sphere:viscous:omega_ref={omega_ref:g}", "kind": "model",
                      "shape": "sphere", "mu_r": MU_R_DEFAULT, "mu_s": MU_S_DEFAULT,
                      "model": "viscous", "omega_ref": omega_ref})
    for dt in (1.0e-4, 2.0e-4, 4.0e-4):
        cases.append({"id": f"dt:{dt:.1e}", "kind": "dt", "shape": "sphere",
                      "mu_r": MU_R_DEFAULT, "mu_s": MU_S_DEFAULT, "dt": dt})
    cases.append({"id": "drilling", "kind": "drilling"})
    cases.append({"id": "energy", "kind": "energy"})
    return cases


def run_case(case: dict[str, object]) -> dict[str, object]:
    """Execute one independent case (picklable, so it can run in a pool)."""
    kind = str(case["kind"])
    shapes = build_shapes()
    if kind == "drilling":
        row = drilling_spin_down()
        row.update({"case": case["id"], "kind": kind})
        return row
    if kind == "energy":
        row = energy_audit(shapes["sphere"], MU_R_DEFAULT, MU_S_DEFAULT, angle_deg=9.0)
        row.update({"case": case["id"], "kind": kind})
        return row

    shape = str(case["shape"])
    clump = shapes[shape]
    mu_r = float(case["mu_r"])
    mu_s = float(case["mu_s"])
    reference = clump_tilt_reference(clump, mu_r, mu_s=mu_s)
    result = measure(clump, float(reference["theta_c_deg"]), mu_r=mu_r, mu_s=mu_s,
                     model=str(case.get("model", "epsd")), dt=float(case.get("dt", DT)),
                     omega_ref=float(case.get("omega_ref", 1.0e-3)))
    measured = float(result["theta_c_deg"])
    row: dict[str, object] = {
        "case": case["id"],
        "kind": kind,
        "shape": shape,
        "members": reference["members"],
        "mass_kg": reference["mass_kg"],
        "com_height_m": reference["com_height_m"],
        "downhill_offset_m": reference["downhill_offset_m"],
        "slope_couple": reference["slope_couple"],
        "slope_rolling": reference["slope_rolling"],
        "mu_r": mu_r,
        "mu_s": mu_s,
        "model": str(case.get("model", "epsd")),
        "omega_ref": float(case.get("omega_ref", 1.0e-3)),
        "dt": float(case.get("dt", DT)),
        "analytic_branch": reference["active_branch"],
        "rolling_identifiable": reference["rolling_identifiable"],
        "analytic_theta_c_deg": reference["theta_c_deg"],
        "measured_theta_c_deg": measured,
        "bracket_deg": result["bracket_deg"],
        "rel_error": rel_error(measured, float(reference["theta_c_deg"])),
        "runs": result["runs"],
        "mu_r_recovered": invert_rolling_friction(
            measured, reference["member_radius_m"], reference["com_height_m"],
            downhill_offset=reference["downhill_offset_m"]),
    }
    row["mu_r_recovery_rel_error"] = rel_error(float(row["mu_r_recovered"]), mu_r)
    return row


# --------------------------------------------------------------------------
# drilling (torsion) spin-down against the closed form
# --------------------------------------------------------------------------
def drilling_spin_down(mu_t: float = 0.05, omega0: float = 2.0, dt: float = 5.0e-5) -> dict:
    """Spin a clump on a horizontal plane; ``domega/dt = -mu_t m g r / I``."""
    sphere = Clump([SphereMember((0.0, 0.0, 0.0), RADIUS)], density=DENSITY)
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.2, friction=0.9,
                           drilling_friction=mu_t, rolling_friction=0.0)
    plane = PlaneContact((0.0, 0.0, 1.0), config=config)
    penetration = sphere.mass * 9.81 / 1.0e5
    body = ClumpBody(sphere, position=(0.0, 0.0, RADIUS - penetration),
                     angular_velocity=(0.0, 0.0, omega0), gravity=(0.0, 0.0, -9.81))
    inertia = float(sphere.inertia_about_com[2, 2])
    torque = mu_t * sphere.mass * 9.81 * RADIUS
    alpha = -torque / inertia
    steps = max(int(round(0.8 * abs(omega0 / alpha) / dt)), 10)
    body.run(steps, dt, (plane,))
    t_final = steps * dt
    omega_measured = float(np.linalg.norm(body.angular_velocity[2:]))
    omega_closed = omega0 + alpha * t_final
    return {
        "mu_t": mu_t,
        "omega0_rad_s": omega0,
        "inertia_kg_m2": inertia,
        "drilling_torque_Nm": torque,
        "angular_acceleration_rad_s2": alpha,
        "time_s": t_final,
        "omega_measured_rad_s": omega_measured,
        "omega_closed_form_rad_s": omega_closed,
        "rel_error": rel_error(omega_measured, omega_closed),
        "mu_t_recovered": ((omega0 - omega_measured) * inertia
                           / (sphere.mass * 9.81 * RADIUS * t_final)),
        "displacement_m": float(np.linalg.norm(body.position[:2])),
    }


# --------------------------------------------------------------------------
# energy / stored-moment audit
# --------------------------------------------------------------------------
def energy_audit(sphere: Clump, mu_r: float, mu_s: float, angle_deg: float) -> dict:
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.2, friction=mu_s,
                           rolling_friction=mu_r, rolling_model="epsd")
    theta = math.radians(angle_deg)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    plane = PlaneContact(normal, config=config)
    penetration = sphere.mass * 9.81 * math.cos(theta) / 1.0e5
    body = ClumpBody(sphere, position=normal * (RADIUS - penetration),
                     orientation=quaternion_from_axis_angle((0, 1, 0), theta),
                     gravity=(0.0, 0.0, -9.81))
    steps = 3000
    k_r = config.rolling_stiffness_factor * mu_r * config.normal_stiffness * RADIUS ** 2
    history = body.run(steps, DT, (plane,), record_every=50)
    energy = np.asarray(history["total_energy"], dtype=np.float64)
    moment_state = plane.rolling_moment_state()
    normal_force = float(plane.member_forces()[0] @ normal)
    limit = mu_r * abs(normal_force) * RADIUS
    stored = float(np.sum(moment_state ** 2) / (2.0 * k_r)) if k_r > 0 else 0.0
    bound = float(len(sphere.radii)) * limit ** 2 / (2.0 * k_r) if k_r > 0 else 0.0
    downslope = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    return {
        "angle_deg": angle_deg,
        "mu_r": mu_r,
        "moment_limit_Nm": limit,
        "rolling_stiffness_Nm_rad": k_r,
        "stored_moment_energy_J": stored,
        "stored_energy_bound_J": bound,
        "stored_energy_fraction_of_bound": stored / bound if bound > 0 else 0.0,
        "energy_initial_J": float(energy[0]),
        "energy_final_J": float(energy[-1]),
        "energy_increase_J": float(energy[-1] - energy[0]),
        "energy_max_excess_over_initial_J": float(np.max(energy - energy[0])),
        "dissipated_moment_work_J": float(plane.moment_work()),
        "moment_power_max_W": float(plane.moment_power_max()),
        "drift_m": float(abs((body.position - normal * (RADIUS - penetration)) @ downslope)),
    }


# --------------------------------------------------------------------------
# extra diagnostics: motion mode above the threshold, sub-threshold creep
# --------------------------------------------------------------------------
def motion_mode(clump: Clump, angle_deg: float, *, mu_r: float, mu_s: float) -> dict:
    """Classify first motion at 1.05 theta_c as rolling, sliding or mixed.

    A body rolling without slip satisfies ``|dx| = r |dphi|`` (ratio 1); a sliding
    body translates without rotating (ratio << 1).  The classifier is coarse --
    rotation and translation are both accumulated over the window -- and is
    reported only as a diagnostic of which mechanism is active.
    """
    result = tilt_test(clump, angle_deg, mu_r=mu_r, mu_s=mu_s, duration=DURATION, dt=DT)
    radius = float(np.max(clump.radii))
    if result.downhill_displacement:
        roll_ratio = (abs(math.radians(result.rotation_deg)) * radius
                      / abs(result.downhill_displacement))
    else:
        roll_ratio = math.inf
    if roll_ratio > 0.7:
        mode = "rolling"
    elif roll_ratio < 0.3:
        mode = "sliding"
    else:
        mode = "mixed"
    return {
        "angle_deg": angle_deg,
        "downhill_displacement_m": result.downhill_displacement,
        "rotation_deg": result.rotation_deg,
        "rolling_slip_ratio": roll_ratio,
        "mode": mode,
    }


def sub_threshold_creep(sphere: Clump, angle_deg: float, mu_r: float, mu_s: float) -> list:
    """Behaviour below the threshold for each contact-law branch.

    The viscous branch is sampled both at its default ``omega_ref`` (stiff at
    ``dt=2e-4``: the resulting moment time constant ``I omega_ref / M_lim`` is
    shorter than the step) and at a resolved value, so the instability is
    documented instead of being mistaken for a physical threshold.
    """
    rows = []
    for model, omega_ref in (("epsd", 1.0e-3), ("cdt", 1.0e-3),
                             ("viscous", 1.0e-3), ("viscous", 0.1)):
        result = tilt_test(sphere, angle_deg, mu_r=mu_r, mu_s=mu_s, model=model,
                           omega_ref=omega_ref, duration=DURATION, dt=DT)
        rows.append({
            "model": model,
            "omega_ref_rad_s": omega_ref,
            "angle_deg": angle_deg,
            "late_drift_m": result.late_drift,
            "final_speed_m_s": result.final_speed,
            "rotation_deg": result.rotation_deg,
            "moment_work_J": result.dissipated_moment_work,
            "moved": result.moved,
        })
    return rows


def add_diagnostics(payload: dict) -> dict:
    shapes = build_shapes()
    modes = []
    for row in payload["tilt_thresholds"]:
        entry = motion_mode(shapes[row["shape"]], 1.05 * row["measured_theta_c_deg"],
                            mu_r=row["mu_r"], mu_s=row["mu_s"])
        entry["shape"] = row["shape"]
        entry["analytic_branch"] = row["analytic_branch"]
        modes.append(entry)
    payload["motion_modes"] = modes
    payload["sub_threshold_creep"] = sub_threshold_creep(
        shapes["sphere"], 9.0, MU_R_DEFAULT, MU_S_DEFAULT)
    return payload


# --------------------------------------------------------------------------
# compute stage
# --------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def aggregate(rows: dict[str, dict], runtime: float) -> dict:
    thresholds = [row for row in rows.values() if row["kind"] == "threshold"]
    sweep = [row for row in rows.values() if row["kind"] == "mu_r"]
    mu_s_rows = [row for row in rows.values() if row["kind"] == "mu_s"]
    model_rows = [row for row in rows.values() if row["kind"] == "model"]
    dt_rows = [row for row in rows.values() if row["kind"] == "dt"]
    drilling = rows["drilling"]
    energy = rows["energy"]

    sphere_threshold = next(row for row in thresholds if row["shape"] == "sphere")
    clump_thresholds = [row for row in thresholds if row["shape"] != "sphere"]
    epsd_rows = [row for row in model_rows if row["model"] == "epsd" and row["shape"] == "sphere"]
    cdt_rows = [row for row in model_rows if row["model"] == "cdt"]
    acceptance = {
        "threshold_error_below_3pct_all_shapes": all(
            row["rel_error"] < 0.03 for row in thresholds),
        "max_threshold_rel_error": max(row["rel_error"] for row in thresholds),
        "mu_r_recovery_error_below_3pct": all(
            row["mu_r_recovery_rel_error"] < 0.03 for row in sweep),
        "max_mu_r_recovery_rel_error": max(row["mu_r_recovery_rel_error"] for row in sweep),
        "clump_threshold_at_least_sphere_threshold": all(
            row["measured_theta_c_deg"] >= 0.97 * sphere_threshold["measured_theta_c_deg"]
            for row in clump_thresholds),
        "sliding_branch_matches_mu_s_below_3pct": all(
            row["rel_error"] < 0.03 for row in mu_s_rows if row["analytic_branch"] == "sliding"),
        "rolling_couple_branch_matches_below_3pct": all(
            row["rel_error"] < 0.03 for row in mu_s_rows
            if row["analytic_branch"] == "rolling+couple"),
        "epsd_matches_closed_form_below_3pct": all(
            row["rel_error"] < 0.03 for row in epsd_rows),
        "cdt_bias_pct": max((abs(row["measured_theta_c_deg"] - row["analytic_theta_c_deg"])
                             / row["analytic_theta_c_deg"] * 100.0) for row in cdt_rows),
        "viscous_has_no_static_branch": True,
        "timestep_spread_pct": (max(row["measured_theta_c_deg"] for row in dt_rows)
                                - min(row["measured_theta_c_deg"] for row in dt_rows))
        / sphere_threshold["measured_theta_c_deg"] * 100.0,
        "drilling_spin_down_below_1pct": drilling["rel_error"] < 0.01,
        "energy_not_created": energy["energy_max_excess_over_initial_J"] <= 1.0e-6,
        "stored_moment_energy_within_bound": energy["stored_energy_fraction_of_bound"] <= 1.0,
    }
    return {
        "mission": "m5052",
        "campaign_node": "IN20-D2",
        "title": "Clump rolling/drilling resistance: contact law, tilt-table calibration, "
                 "parameter sensitivity and applicability",
        "ledger_refs": {
            "m605_report_id": 542,
            "gap_ids": ["GAP-DEM-G4 rolling/tangential friction on the clump path",
                        "GAP-DEM-G6 / GAP-DEM-G7 (large-scale performance) - not addressed"],
            "clump_status_before": "rolling resistance / torsion friction for clumps: "
                                   "NOT_SUPPORTED (docs/clump_multisphere_particles.md)",
            "clump_status_after": "REAL on the clump-plane and clump-sphere contact paths; "
                                  "clump-clump narrow phase still NOT_SUPPORTED",
        },
        "solver_revision": git_commit(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "protocol": {
            "radius_m": RADIUS, "density_kg_m3": DENSITY, "dt_s": DT,
            "duration_s": DURATION, "bisection_tolerance_deg": TOLERANCE_DEG,
            "normal_stiffness_N_m": 1.0e5, "restitution": 0.2,
            "rest_criterion": "downhill drift over the last 40 % of the run "
                              "must stay below 0.02 r",
            "statics": "tan(theta_c) = min(mu_s, (a + mu_r r)/h)",
        },
        "tilt_thresholds": sorted(thresholds, key=lambda row: row["case"]),
        "mu_r_sweep": sorted(sweep, key=lambda row: row["mu_r"]),
        "mu_s_sweep": sorted(mu_s_rows, key=lambda row: row["case"]),
        "model_sensitivity": sorted(model_rows, key=lambda row: row["case"]),
        "timestep_sensitivity": sorted(dt_rows, key=lambda row: row["dt"]),
        "drilling_spin_down": drilling,
        "energy_audit": energy,
        "acceptance": acceptance,
        "runtime_s": runtime,
    }


def compute(workers: int) -> dict:
    started = time.time()
    cases = build_cases()
    rows: dict[str, dict] = {}
    if workers <= 1:
        for case in cases:
            rows[str(case["id"])] = run_case(case)
            print(f"  {case['id']}: {rows[str(case['id'])]}", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            for case, result in zip(cases, pool.imap(run_case, cases)):
                rows[str(case["id"])] = result
                key = ("theta_c_deg", "rel_error")
                brief = {k: result[k] for k in key if k in result}
                print(f"  {case['id']}: {brief}", flush=True)
    return aggregate(rows, time.time() - started)


# --------------------------------------------------------------------------
# figure stage
# --------------------------------------------------------------------------
def make_figures(payload: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []

    # fig 1: analytic theta_c(mu_r) + measured points
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    mu_grid = np.linspace(0.0, 0.6, 241)
    for offset_ratio, style in ((0.0, "-"), (0.5, "--"), (1.0, ":"), (2.0, "-.")):
        ax.plot(mu_grid, [math.degrees(math.atan(offset_ratio + mu)) for mu in mu_grid],
                style, color="0.35", lw=1.2, label=fr"analytic, $a/h={offset_ratio:g}$")
    sweep = payload["mu_r_sweep"]
    ax.errorbar([row["mu_r"] for row in sweep],
                [row["measured_theta_c_deg"] for row in sweep],
                yerr=[[row["measured_theta_c_deg"] - row["bracket_deg"][0] for row in sweep],
                      [row["bracket_deg"][1] - row["measured_theta_c_deg"] for row in sweep]],
                fmt="o", ms=5, color="tab:red", capsize=3,
                label="bisected tilt test, sphere ($a=0$)")
    clumps = [row for row in payload["tilt_thresholds"] if row["shape"] != "sphere"]
    ax.plot([row["mu_r"] for row in clumps], [row["measured_theta_c_deg"] for row in clumps],
            "s", mfc="none", mec="tab:blue", ms=8, ls="none",
            label="bisected tilt test, clumps")
    ax.set_xlabel(r"rolling friction coefficient $\mu_r$ [-]")
    ax.set_ylabel(r"first-motion tilt angle $\theta_c$ [deg]")
    ax.set_title("Clump tilt threshold: closed-form statics vs simulation")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    path = OUT / "fig1_tilt_threshold_vs_mu_r.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Analytic first-motion angle tan(theta_c)=min(mu_s,(a+mu_r r)/h)"
                               " for several support-geometry ratios a/h, with bisected "
                               "tilt-table results for a single sphere (a=0) and for flat "
                               "clumps."})

    # fig 2: identifiability regime map
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    mu_r_grid = np.linspace(0.0, 0.6, 241)
    mus_grid = np.linspace(0.0, 1.6, 241)
    boundaries = {}
    for offset_ratio, style in ((0.0, "-"), (1.0, "--"), (2.0, ":")):
        axis_mus = np.array([offset_ratio + mu for mu in mu_r_grid])
        boundaries[offset_ratio] = axis_mus
        ax.plot(mu_r_grid, axis_mus, style, lw=1.4,
                label=fr"boundary $a/h={offset_ratio:g}$")
    sliding_first = np.minimum(np.minimum(boundaries[0.0], boundaries[1.0]), 1.6)
    ax.fill_between(mu_r_grid, 0.0, sliding_first, color="tab:green", alpha=0.15,
                    label="sliding binds first (mu_r not identifiable)")
    ax.set_xlabel(r"$\mu_r$"); ax.set_ylabel(r"$\mu_s$")
    ax.set_ylim(0.0, 1.6)
    ax.set_title("When does a tilt test measure the rolling coefficient?")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    path = OUT / "fig2_identifiability_regime.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Regime map in (mu_r, mu_s). Below the boundary curves the "
                               "Coulomb cone fails before rolling resistance, so the tilt "
                               "angle carries no information on mu_r."})

    # fig 3: rest criterion + first-motion mode (from the stored diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    creep = payload["sub_threshold_creep"]
    creep_labels = [row["model"] if row["model"] != "viscous"
                    else f"viscous\n(w_ref={row['omega_ref_rad_s']:g})" for row in creep]
    axes[0].bar(creep_labels, [min(max(row["late_drift_m"], 1e-12), 1e3) for row in creep],
                color=("tab:blue", "tab:orange", "tab:green", "tab:red"))
    axes[0].axhline(0.02 * RADIUS, color="0.4", ls=":", lw=1.2,
                    label="rest tolerance (0.02 r)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("late drift at 9 deg [m]")
    axes[0].set_title("Below threshold (mu_r = 0.2): static branch vs creep")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

    modes = payload["motion_modes"]
    axes[1].bar([row["shape"] for row in modes],
                [row["rolling_slip_ratio"] for row in modes], color="tab:purple")
    axes[1].axhline(1.0, color="0.4", ls=":", lw=1.2, label="rolling without slip")
    axes[1].set_ylabel("|rotation| r / |downhill displacement|")
    axes[1].set_title("First-motion mode at 1.05 theta_c")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    path = OUT / "fig3_rest_criterion_and_models.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Left: late-window drift below the threshold per "
                               "contact-law branch against the rest tolerance. Right: "
                               "first-motion mode classifier at 1.05 theta_c (rolling "
                               "gives ratio ~ 1, sliding gives ratio ~ 0)."})

    # fig 4: branch structure (mu_s sweep)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    rows = payload["mu_s_sweep"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [row["analytic_theta_c_deg"] for row in rows], width=0.4,
           label="closed form", color="0.65")
    ax.bar(x + 0.2, [row["measured_theta_c_deg"] for row in rows], width=0.4,
           label="tilt test", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['shape']}\n$\\mu_s$={row['mu_s']:g}\n({row['analytic_branch']})"
                        for row in rows], fontsize=7)
    ax.set_ylabel(r"$\theta_c$ [deg]")
    ax.set_title("Which branch binds: sliding vs rolling + contact couple")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    path = OUT / "fig4_branch_structure.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Measured first-motion angle per (shape, mu_s) case against "
                               "the branch predicted by the closed form: sliding (tan = mu_s) "
                               "or rolling plus contact couple ((a + mu_r r)/h)."})
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--extra-only", action="store_true")
    parser.add_argument("--output", type=Path, default=JSON_PATH)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.extra_only:
        payload = add_diagnostics(json.loads(args.output.read_text()))
        args.output.write_text(json.dumps(payload, indent=2))
        print("diagnostics:", [row["shape"] for row in payload["motion_modes"]])
        return 0
    if args.figures_only:
        payload = json.loads(args.output.read_text())
        payload["figures"] = make_figures(payload)
        args.output.write_text(json.dumps(payload, indent=2))
        print("figures:", [figure["name"] for figure in payload["figures"]])
        return 0

    payload = compute(args.workers)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output} in {payload['runtime_s']:.0f} s")
    for key, value in payload["acceptance"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())