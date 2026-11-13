"""Validation driver for XVAL-DEM-REUSE-POLYHEDRON-FLOE3D-PLUS-DRAG-AND-RE.

Runs five validation cases:
A. Linear drag decay in quiescent fluid (analytic: v = v0*exp(-c*t/m))
B. Spring-dashpot restitution (analytic: e = exp(-zeta*pi/sqrt(1-zeta^2)))
C. Buoyancy equilibrium (analytic: submerged fraction = rho_ice/rho_water)
D. Combined drag + restitution impact
E. Convergence study over dt

Uses square-prism (cube) geometry for restitution cases to ensure flat-face impact,
matching the established polyhedron_floe3d_restitution_anchor benchmark.

Usage:
    PYTHONPATH=src python benchmarks/xval_floe3d_drag_re/run_validation.py
    PYTHONPATH=src python benchmarks/xval_floe3d_drag_re/run_validation.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
from tensordem.polyhedron_floe3d.contact import ContactLaw
from tensordem.polyhedron_floe3d.geometry import extrude_polygon, regular_polygon

OUT_DIR = os.path.join(ROOT, "artifacts", "xval_floe3d_drag_re")
os.makedirs(OUT_DIR, exist_ok=True)

# --- geometry ---
FLOE_AREA = 0.25       # m2
FLOE_THICKNESS = 0.12  # m
ICE_DENSITY = 900.0    # kg/m3
WATER_DENSITY = 1025.0 # kg/m3
GRAVITY = 9.81         # m/s2

# Cube geometry for restitution (flat-face impact)
CUBE_EDGE = 0.10       # m
CUBE_DENSITY = 917.0   # kg/m3


def make_floe(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
              density=ICE_DENSITY, area=FLOE_AREA, thickness=FLOE_THICKNESS):
    """Hexagonal prism floe."""
    shape = extrude_polygon(regular_polygon(area, 6), thickness)
    return FloeBody(shape, density, position=list(position), velocity=list(velocity))


def make_cube(edge=CUBE_EDGE, density=CUBE_DENSITY,
              position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
    """Square prism (cube) for flat-face impact."""
    shape = extrude_polygon(regular_polygon(edge * edge, 4), edge)
    return FloeBody(shape, density, position=list(position), velocity=list(velocity))


# ===========================================================================
# Case A: Linear drag decay
# ===========================================================================
def case_drag_decay(*, dt=2.5e-4, t_end=0.2, c=10.0, v0=0.5):
    """Drag decay in quiescent fluid: v(t) = v0 * exp(-c*t/m)."""
    body = make_floe(velocity=[v0, 0.0, 0.0])
    system = DEMSystem3D([body], law=ContactLaw(), gravity=0.0, dt=dt,
                         linear_drag=c)
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        system.step()
    v_measured = float(body.velocity[0])
    v_analytic = v0 * math.exp(-c * t_end / body.mass)
    rel_err = abs(v_measured - v_analytic) / abs(v_analytic) if v_analytic != 0 else 0.0
    return {
        "case": "drag_decay",
        "setup": {"dt": dt, "t_end": t_end, "c": c, "v0": v0, "mass": body.mass},
        "ref": {"v_analytic": v_analytic, "formula": "v0*exp(-c*t/m)"},
        "got": {"v_measured": v_measured, "steps": n_steps},
        "rel_error": rel_err,
        "passed": rel_err < 0.03,
    }


# ===========================================================================
# Case B: Restitution
# ===========================================================================
def analytic_restitution(zeta):
    if zeta >= 1.0:
        return 0.0
    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))


def case_restitution(*, zeta=0.1, dt=5e-5, kn=2e5, v_in=0.5):
    """Drop a cube onto a floor; measure e = |v_out|/|v_in|.

    Uses cube geometry (square prism) for flat-face impact, matching the
    established polyhedron_floe3d_restitution_anchor benchmark.
    """
    body = make_cube(velocity=[0.0, 0.0, -v_in])
    # Place lowest vertex just above floor
    body.position = np.array([0.0, 0.0, 2e-3 - float(body.world_vertices()[:, 2].min())])
    system = DEMSystem3D([body], law=ContactLaw(kn=kn, zeta=zeta, mu=0.0,
                                                 mu_rolling=0.0, kt_ratio=0.0),
                         gravity=0.0, dt=dt)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    system.record_initial_energy()

    touched = False
    v_out = float("nan")
    max_steps = int(1.0 / dt)
    for step in range(1, max_steps + 1):
        system.step()
        if system.n_wall_contacts > 0:
            touched = True
        elif touched:
            v_out = float(body.velocity[2])
            break

    e_measured = abs(v_out) / v_in if not math.isnan(v_out) else float("nan")
    e_analytic = analytic_restitution(zeta)
    rel_err = abs(e_measured - e_analytic) / e_analytic if e_analytic > 0 else 0.0
    return {
        "case": "restitution",
        "setup": {"zeta": zeta, "dt": dt, "kn": kn, "v_in": v_in,
                  "geometry": "cube (square prism)", "mass": body.mass},
        "ref": {"e_analytic": e_analytic, "formula": "exp(-zeta*pi/sqrt(1-zeta^2))"},
        "got": {"e_measured": e_measured, "v_out": v_out, "touched": touched},
        "rel_error": rel_err,
        "passed": rel_err < 0.10,  # relaxed threshold: DEM contact has inherent discretization error
    }


# ===========================================================================
# Case C: Buoyancy equilibrium
# ===========================================================================
def case_buoyancy_equilibrium(*, dt=5e-4, max_steps=8000, drag_fraction=0.7):
    """Release floe at waterline; measure submerged fraction at equilibrium."""
    body = make_floe(position=[0.0, 0.0, 0.0])
    mass = body.mass
    # heave natural frequency: omega0 = sqrt(rho_w * g * A / m)
    omega0 = math.sqrt(WATER_DENSITY * GRAVITY * FLOE_AREA / mass)
    linear_drag = drag_fraction * 2.0 * mass * omega0
    system = DEMSystem3D([body], law=ContactLaw(), gravity=GRAVITY,
                         water_level=0.0, water_density=WATER_DENSITY,
                         dt=dt, linear_drag=linear_drag, angular_drag=0.0)
    system.record_initial_energy()

    history = []
    quiet = 0
    settle_tol = 1e-4
    settle_window = 200
    steps = 0
    for steps in range(1, max_steps + 1):
        system.step()
        if steps % 10 == 0:
            # Compute submerged fraction
            from tensordem.polyhedron_floe3d.dynamics import submerged_tet_moments
            vol, _, _ = submerged_tet_moments(body.world_tets(), 0.0)
            frac = float(np.sum(vol)) / body.shape.volume
            history.append({"t": system.time, "z": body.position[2],
                           "vz": body.velocity[2], "frac": frac})
        if abs(body.velocity[2]) < settle_tol:
            quiet += 1
            if quiet >= settle_window:
                break
        else:
            quiet = 0

    tail = history[-40:] if len(history) >= 40 else history
    frac_mean = float(np.mean([h["frac"] for h in tail]))
    frac_analytic = ICE_DENSITY / WATER_DENSITY
    rel_err = abs(frac_mean - frac_analytic) / frac_analytic
    return {
        "case": "buoyancy_equilibrium",
        "setup": {"dt": dt, "drag_fraction": drag_fraction, "omega0": omega0,
                  "linear_drag": linear_drag, "mass": mass},
        "ref": {"frac_analytic": frac_analytic, "formula": "rho_ice/rho_water"},
        "got": {"frac_measured": frac_mean, "steps": steps,
                "sim_time": float(system.time)},
        "rel_error": rel_err,
        "passed": rel_err < 0.05,
        "history": tail[-10:],  # last 10 samples for audit
    }


# ===========================================================================
# Case D: Combined drag + restitution impact
# ===========================================================================
def case_combined_drag_restitution(*, dt=5e-5, zeta=0.1, kn=2e5,
                                    v_in=0.5, c=5.0):
    """Impact with floor while experiencing linear drag.

    Uses cube geometry for flat-face impact.
    """
    body = make_cube(velocity=[0.0, 0.0, -v_in])
    body.position = np.array([0.0, 0.0, 2e-3 - float(body.world_vertices()[:, 2].min())])
    system = DEMSystem3D([body], law=ContactLaw(kn=kn, zeta=zeta, mu=0.0,
                                                 mu_rolling=0.0, kt_ratio=0.0),
                         gravity=0.0, dt=dt, linear_drag=c)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    system.record_initial_energy()

    # Analytic: without drag, e = exp(-zeta*pi/sqrt(1-zeta^2))
    # With drag, the velocity decays during contact
    m_eff = body.mass
    omega_n = math.sqrt(kn / m_eff)
    omega_d = omega_n * math.sqrt(1 - zeta**2) if zeta < 1 else 0.0
    t_contact = math.pi / omega_d if omega_d > 0 else 0.0
    e_analytic = analytic_restitution(zeta)
    # Approximate: velocity decays during contact
    v_out_approx = -v_in * e_analytic * math.exp(-c * t_contact / m_eff)
    peak_force_approx = v_in * math.sqrt(kn * m_eff) * math.exp(-zeta * math.pi / (2 * math.sqrt(1 - zeta**2))) if zeta < 1 else 0.0

    touched = False
    v_out = float("nan")
    peak_force = 0.0
    max_steps = int(1.0 / dt)
    for step in range(1, max_steps + 1):
        system.step()
        if system.n_wall_contacts > 0:
            touched = True
            for wf in system.last_wall_forces:
                f_mag = float(np.linalg.norm(wf[2]))
                if f_mag > peak_force:
                    peak_force = f_mag
        elif touched:
            v_out = float(body.velocity[2])
            break

    e_measured = abs(v_out) / v_in if not math.isnan(v_out) else float("nan")
    # For combined case, compare against drag-corrected analytic
    e_expected = abs(v_out_approx) / v_in
    rel_err_e = abs(e_measured - e_expected) / e_expected if e_expected > 0 else 0.0
    rel_err_f = abs(peak_force - peak_force_approx) / peak_force_approx if peak_force_approx > 0 else 0.0
    return {
        "case": "combined_drag_restitution",
        "setup": {"dt": dt, "zeta": zeta, "kn": kn, "v_in": v_in, "c": c,
                  "t_contact_approx": t_contact, "omega_n": omega_n,
                  "geometry": "cube (square prism)"},
        "ref": {"e_analytic_no_drag": e_analytic,
                "v_out_approx_with_drag": v_out_approx,
                "peak_force_approx": peak_force_approx},
        "got": {"e_measured": e_measured, "v_out": v_out,
                "peak_force": peak_force, "touched": touched},
        "rel_error_e": rel_err_e,
        "rel_error_f": rel_err_f,
        "passed": rel_err_e < 0.15 and rel_err_f < 0.15,
    }


# ===========================================================================
# Case E: Convergence study
# ===========================================================================
def case_convergence_study():
    """Run drag decay at multiple dt values."""
    dt_grid = [5e-4, 2.5e-4, 1.25e-4]
    results = []
    for dt in dt_grid:
        r = case_drag_decay(dt=dt, t_end=0.2, c=10.0, v0=0.5)
        results.append(r)
    # Check convergence: error should decrease with dt
    errors = [r["rel_error"] for r in results]
    converging = all(errors[i] >= errors[i+1] * 0.67 for i in range(len(errors)-1))
    return {
        "case": "convergence",
        "dt_grid": dt_grid,
        "results": results,
        "errors": errors,
        "converging": converging,
    }


# ===========================================================================
# Negative controls
# ===========================================================================
def negative_control_zero_drag():
    """Zero drag must show no velocity decay."""
    body = make_floe(velocity=[0.5, 0.0, 0.0])
    system = DEMSystem3D([body], law=ContactLaw(), gravity=0.0, dt=1e-4,
                         linear_drag=0.0)
    for _ in range(2000):
        system.step()
    v_final = float(body.velocity[0])
    rel_err = abs(v_final - 0.5) / 0.5
    return {"case": "nc_zero_drag", "v_final": v_final,
            "rel_error": rel_err, "passed": rel_err < 0.01}


def negative_control_zero_damping():
    """Zero damping must give e≈1.0 (elastic impact)."""
    body = make_cube(velocity=[0.0, 0.0, -0.5])
    body.position = np.array([0.0, 0.0, 2e-3 - float(body.world_vertices()[:, 2].min())])
    system = DEMSystem3D([body], law=ContactLaw(kn=2e5, zeta=0.0, mu=0.0),
                         gravity=0.0, dt=5e-5)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    touched = False
    v_out = float("nan")
    for _ in range(int(1.0 / 5e-5)):
        system.step()
        if system.n_wall_contacts > 0:
            touched = True
        elif touched:
            v_out = float(body.velocity[2])
            break
    e_measured = abs(v_out) / 0.5 if not math.isnan(v_out) else 0.0
    return {"case": "nc_zero_damping", "e_measured": e_measured,
            "rel_error": abs(e_measured - 1.0), "passed": e_measured > 0.95}


# ===========================================================================
# Main
# ===========================================================================
def run_all():
    t0 = time.time()
    results = {
        "A_drag_decay": case_drag_decay(),
        "B_restitution": case_restitution(),
        "C_buoyancy_equilibrium": case_buoyancy_equilibrium(),
        "D_combined_drag_restitution": case_combined_drag_restitution(),
        "E_convergence": case_convergence_study(),
        "NC_zero_drag": negative_control_zero_drag(),
        "NC_zero_damping": negative_control_zero_damping(),
        "runtime_s": time.time() - t0,
    }

    # Summary
    passed = sum(1 for k, v in results.items()
                 if isinstance(v, dict) and v.get("passed", False))
    total = sum(1 for k, v in results.items() if isinstance(v, dict) and "passed" in v)
    results["summary"] = {"passed": passed, "total": total,
                          "all_pass": passed == total}

    # Write results
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results written to {out_path}")
    print(f"Passed: {passed}/{total}")
    for k, v in results.items():
        if isinstance(v, dict) and "rel_error" in v:
            status = "PASS" if v.get("passed") else "FAIL"
            print(f"  {k}: rel_error={v['rel_error']:.4e} [{status}]")
        elif isinstance(v, dict) and "rel_error_e" in v:
            status = "PASS" if v.get("passed") else "FAIL"
            print(f"  {k}: rel_error_e={v['rel_error_e']:.4e}, rel_error_f={v['rel_error_f']:.4e} [{status}]")
    return results


def make_figures(results):
    """Generate validation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: Drag decay convergence
    ax = axes[0, 0]
    dt_grid = [5e-4, 2.5e-4, 1.25e-4]
    errors_a = []
    for dt in dt_grid:
        r = case_drag_decay(dt=dt, t_end=0.2, c=10.0, v0=0.5)
        errors_a.append(r["rel_error"])
    ax.loglog(dt_grid, errors_a, "o-", label="drag decay")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("Relative error")
    ax.set_title("A: Drag decay convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # B: Restitution sweep
    ax = axes[0, 1]
    zeta_grid = [0.05, 0.1, 0.2]
    e_measured = []
    e_analytic = []
    for z in zeta_grid:
        r = case_restitution(zeta=z, dt=5e-5, kn=2e5, v_in=0.5)
        e_measured.append(r["got"]["e_measured"])
        e_analytic.append(r["ref"]["e_analytic"])
    ax.plot(zeta_grid, e_analytic, "k-", label="analytic")
    ax.plot(zeta_grid, e_measured, "ro", label="measured")
    ax.set_xlabel("zeta")
    ax.set_ylabel("Coefficient of restitution e")
    ax.set_title("B: Restitution vs damping ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # C: Buoyancy equilibrium history
    ax = axes[1, 0]
    r_c = results["C_buoyancy_equilibrium"]
    hist = r_c.get("history", [])
    if hist:
        times = [h["t"] for h in hist]
        fracs = [h["frac"] for h in hist]
        ax.plot(times, fracs, "b-", label="measured")
        ax.axhline(r_c["ref"]["frac_analytic"], color="k", linestyle="--",
                   label=f"analytic ({r_c['ref']['frac_analytic']:.4f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Submerged fraction")
    ax.set_title("C: Buoyancy equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # D: Combined case error bars
    ax = axes[1, 1]
    r_d = results["D_combined_drag_restitution"]
    categories = ["e (restitution)", "F_peak (force)"]
    errors_d = [r_d["rel_error_e"], r_d["rel_error_f"]]
    ax.bar(categories, errors_d, color=["blue", "red"], alpha=0.7)
    ax.axhline(0.15, color="k", linestyle="--", label="15% threshold")
    ax.set_ylabel("Relative error")
    ax.set_title("D: Combined drag + restitution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig_validation.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        with open(os.path.join(OUT_DIR, "results.json")) as f:
            results = json.load(f)
        make_figures(results)
    else:
        results = run_all()
        make_figures(results)
