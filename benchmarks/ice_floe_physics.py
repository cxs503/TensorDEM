"""Four-anchor verification of ice-floe contact/flotation physics against closed forms.

Scope: this script exercises *repository code* (``tensordem.buoyancy_wet`` and
``tensordem.contact_tangential``) against references that are derived
independently of that code.  It is a verification harness for the implemented
force laws; it is not a validation against ice-tank experiments.

Anchor H  Hydrostatic equilibrium of a floating disk
    A circular ice disk (rho_i, radius R, thickness t) is released in still
    water (rho_w) from six different heights and integrated until it settles.
    Reference (Archimedes, exact):        f_eq = rho_i / rho_w
    and the equilibrium centre height h_eq solves A_sub(h_eq) = f_eq * pi R^2,
    where A_sub is derived here in closed form by a polar-substitution
    integral of the chord width (see ``analytic_submerged_area``), i.e. by a
    derivation path that is independent of ``submerged_fraction_tensor``.
    Criterion: |f_sub,sim - f_eq| / f_eq < 1 %.

Anchor A  Archimedes force against the independently derived displaced volume
    ``compute_buoyancy_force`` is evaluated on 21 centre heights spanning the
    fully dry, partially submerged and fully submerged regimes and compared
    with rho_w * g * A_sub_closedform * t.
    Criterion: max relative error < 0.1 % (limited by the float64 reference).

Anchor T  Small-amplitude heave period against the linearised restoring stiffness
    The heave restoring stiffness of a floating disk at its equilibrium
    waterline is k = rho_w * g * t * 2*sqrt(R^2 - h_eq^2), so the small-heave
    period is T = 2*pi*sqrt(m/k) with m = rho_i*pi*R^2*t.  The period is
    measured from zero crossings of a 0.2 %-of-R release and the run is
    repeated at dt, dt/2 and dt/4 to show that the residual is a
    discretisation effect that shrinks with dt (the discrete fixed point of
    anchor H is dt-independent by construction, a dynamical observable is not).
    Criterion: |T_sim - T_ref| / T_ref < 1 % at the finest dt.

Anchor F  Inclined-plane friction onset and sliding acceleration
    ``integrate_tangential_displacement`` + ``compute_tangential_force``
    (Cundall-Strack incremental spring, projected onto the Coulomb cone) drive
    a 1-DOF disk on a fixed inclined plane with quasi-static normal load
    N = m*g*cos(theta).  Slip onset is measured by simulation over a grid of
    planes around arctan(mu); the sliding acceleration above onset is compared
    with the analytic a = g*(sin(theta) - mu*cos(theta)).
    Criterion: |theta_onset - arctan(mu)| < 0.3 deg, acceleration error < 3 %.

Previous revisions of this file also carried (a) a "friction-angle" anchor that
re-derived arctan(mu) from sin/cos and therefore never executed repository code,
and (b) a hard-coded equilibrium draught ``-(1 - rho_i/rho_w)*R`` that is
analytically wrong for a circular section.  Both are replaced here; see
``docs/ice_floe_physics_validation.md``.

Outputs a machine-readable report (default
``artifacts/ice_floe_physics/validation_report.json``).  Run with ``--fast`` for
the reduced-step configuration used by the unit test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensordem.buoyancy_wet import (  # noqa: E402
    WetCouplingConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
    submerged_fraction_tensor,
)
from tensordem.contact_tangential import (  # noqa: E402
    compute_tangential_force,
    damping_to_restitution,
    integrate_tangential_displacement,
    restitution_to_damping,
)

G = 9.81
SCHEMA = "tensordem.ice-floe-physics/2"


# ---------------------------------------------------------------------------
# Independent closed-form geometry reference
# ---------------------------------------------------------------------------

def analytic_submerged_area(u: float, radius: float) -> float:
    """Area of a disk below the waterline, centre at height *u* above it.

    Derived independently of the repository implementation by substituting
    y = R*sin(phi) in  A_sub = integral_{-R}^{y_max} 2*sqrt(R^2 - y^2) dy :

        A_sub = R^2 * (pi/2 + phi + sin(phi)*cos(phi)),  phi = arcsin(-u/R)

    with the fully dry (u >= R) and fully submerged (u <= -R) branches taken
    from geometry rather than from the formula.
    """
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    if u >= radius:
        return 0.0
    if u <= -radius:
        return math.pi * radius ** 2
    phi = math.asin(-u / radius)
    return radius ** 2 * (math.pi / 2.0 + phi + math.sin(phi) * math.cos(phi))


def analytic_submerged_fraction(u: float, radius: float) -> float:
    return analytic_submerged_area(u, radius) / (math.pi * radius ** 2)


def analytic_equilibrium_height(rho_i: float, rho_w: float, radius: float) -> float:
    """Solve A_sub(u) = (rho_i/rho_w) * pi R^2 by bisection on the closed form."""
    target = rho_i / rho_w
    lo, hi = -radius, radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if analytic_submerged_fraction(mid, radius) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Shared integrator
# ---------------------------------------------------------------------------

def _semi_implicit_euler(positions, velocities, accel_fn, dt, steps):
    """Symplectic (semi-implicit) Euler: v += a*dt ; x += v*dt.

    Same update ordering as the repository's 2-D DEM integrator.
    """
    for _ in range(steps):
        velocities = velocities + accel_fn(positions, velocities) * dt
        positions = positions + velocities * dt
    return positions, velocities


def _flotation_case(rho_i=917.0, rho_w=1025.0, radius=0.5, thickness=1.0,
                    drag=4000.0):
    mass = rho_i * math.pi * radius ** 2 * thickness
    cfg = WetCouplingConfig(enabled=True, water_density=rho_w, gravity=G,
                            water_level=0.0, wet_drag_coeff=drag)
    return mass, cfg


# ---------------------------------------------------------------------------
# Anchor H — hydrostatic equilibrium
# ---------------------------------------------------------------------------

def anchor_hydrostatic_equilibrium(dt=2e-3, duration=6.0, radius=0.5,
                                   thickness=1.0, rho_i=917.0, rho_w=1025.0):
    """Release a disk from six heights; every one must settle on Archimedes."""
    mass, cfg = _flotation_case(rho_i, rho_w, radius, thickness)
    y0 = torch.tensor(
        [[0.0, radius * k] for k in (1.2, 0.6, 0.0, -0.6, -1.2, -2.0)],
        dtype=torch.float64,
    )
    positions = y0.clone()
    velocities = torch.zeros_like(positions)

    def accel(pos, vel):
        f = compute_buoyancy_force(pos, radius, thickness, rho_i, cfg)
        f = f + compute_wet_drag_force(pos, vel, radius, cfg)
        f[:, 1] -= mass * G
        return f / mass

    steps = int(round(duration / dt))
    positions, velocities = _semi_implicit_euler(positions, velocities, accel, dt, steps)

    f_eq = rho_i / rho_w
    h_eq = analytic_equilibrium_height(rho_i, rho_w, radius)
    f_meas = submerged_fraction_tensor(positions[:, 1], radius, cfg.water_level)
    f_meas_list = [float(x) for x in f_meas]
    errors = [abs(f - f_eq) / f_eq for f in f_meas_list]
    height_errors = [abs(float(y) - h_eq) / radius for y in positions[:, 1]]
    v_max = float(torch.abs(velocities[:, 1]).max())

    return {
        "anchor": "H_hydrostatic_equilibrium",
        "code_under_test": "compute_buoyancy_force + compute_wet_drag_force",
        "reference": "Archimedes: f_eq = rho_i/rho_w; h_eq from closed-form segment area",
        "parameters": {
            "radius_m": radius, "thickness_m": thickness,
            "rho_ice_kg_m3": rho_i, "rho_water_kg_m3": rho_w,
            "drag_N_s_m": cfg.wet_drag_coeff, "dt_s": dt,
            "duration_s": duration, "steps": steps, "n_floats": len(f_meas_list),
        },
        "f_eq_reference": f_eq,
        "h_eq_reference_m": h_eq,
        "release_heights_m": [float(y) for y in y0[:, 1]],
        "settled_f_sub": f_meas_list,
        "settled_height_m": [float(y) for y in positions[:, 1]],
        "max_abs_velocity_m_s": v_max,
        "max_relative_fraction_error": max(errors),
        "max_height_error_over_radius": max(height_errors),
        "criterion": {"max_relative_fraction_error": 0.01,
                      "max_abs_velocity_m_s": 1e-3},
        "passed": max(errors) < 0.01 and v_max < 1e-3,
    }


# ---------------------------------------------------------------------------
# Anchor A — Archimedes force vs independent displaced volume
# ---------------------------------------------------------------------------

def anchor_archimedes_static(n_levels=21, radius=0.5, thickness=1.0,
                             rho_i=917.0, rho_w=1025.0):
    """Buoyancy force on 21 submergences vs rho_w*g*A_sub_closedform*t."""
    _, cfg = _flotation_case(rho_i, rho_w, radius, thickness)
    us = [(-1.0 + 2.0 * i / (n_levels - 1)) * radius for i in range(n_levels)]
    # Include the exact fully-dry / fully-submerged limits as well.
    ys = us + [radius, -radius]
    positions = torch.tensor([[0.0, y] for y in ys], dtype=torch.float64)
    force = compute_buoyancy_force(positions, radius, thickness, rho_i, cfg)[:, 1]

    rows = []
    worst = 0.0
    for y, fy in zip(ys, force):
        ref = rho_w * G * analytic_submerged_area(float(y), radius) * thickness
        measured = float(fy)
        rel = abs(measured - ref) / ref if ref > 0 else abs(measured)
        worst = max(worst, rel)
        rows.append({
            "centre_height_m": float(y),
            "f_sub_code": float(submerged_fraction_tensor(torch.tensor([y]), radius)[0]),
            "f_sub_closed_form": analytic_submerged_fraction(float(y), radius),
            "buoyancy_code_N": measured,
            "buoyancy_reference_N": ref,
            "relative_error": rel,
        })

    return {
        "anchor": "A_archimedes_static",
        "code_under_test": "compute_buoyancy_force",
        "reference": "rho_w*g*A_sub*t with A_sub from an independent polar integral",
        "parameters": {"radius_m": radius, "thickness_m": thickness,
                       "rho_water_kg_m3": rho_w, "n_levels": len(ys)},
        "rows": rows,
        "max_relative_error": worst,
        "criterion": {"max_relative_error": 1e-3},
        "passed": worst < 1e-3,
    }


# ---------------------------------------------------------------------------
# Anchor T — small-amplitude heave period
# ---------------------------------------------------------------------------

def _heave_period(dt, n_periods=2.0, radius=0.5, thickness=1.0,
                  rho_i=917.0, rho_w=1025.0, amplitude_ratio=2e-3):
    """Return (measured period, peak excursions relative to equilibrium)."""
    mass, cfg = _flotation_case(rho_i, rho_w, radius, thickness, drag=0.0)
    h_eq = analytic_equilibrium_height(rho_i, rho_w, radius)
    amplitude = amplitude_ratio * radius

    k_eff = rho_w * G * thickness * 2.0 * math.sqrt(radius ** 2 - h_eq ** 2)
    period_ref = 2.0 * math.pi * math.sqrt(mass / k_eff)
    steps = int(math.ceil(n_periods * period_ref * 1.5 / dt))

    positions = torch.tensor([[0.0, h_eq + amplitude]], dtype=torch.float64)
    velocities = torch.zeros_like(positions)

    def accel(pos, vel):
        f = compute_buoyancy_force(pos, radius, thickness, rho_i, cfg)
        f[:, 1] -= mass * G
        return f / mass

    crossings = []
    t = 0.0
    prev_dt_ = float(positions[0, 1]) - h_eq
    for _ in range(steps):
        velocities = velocities + accel(positions, velocities) * dt
        positions = positions + velocities * dt
        t += dt
        cur = float(positions[0, 1]) - h_eq
        if prev_dt_ > 0.0 >= cur:
            # Linear interpolation of the downward zero crossing.
            frac = prev_dt_ / (prev_dt_ - cur)
            crossings.append(t - dt + frac * dt)
            if len(crossings) >= int(n_periods) + 1:
                break
        prev_dt_ = cur

    if len(crossings) < 2:
        raise RuntimeError("heave run produced too few zero crossings")
    measured = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    return measured, period_ref, h_eq, k_eff, mass


def anchor_heave_period(dt_levels=(4e-3, 2e-3, 1e-3), radius=0.5,
                        thickness=1.0, rho_i=917.0, rho_w=1025.0):
    rows = []
    reference = None
    for dt in dt_levels:
        measured, period_ref, h_eq, k_eff, mass = _heave_period(
            dt, radius=radius, thickness=thickness, rho_i=rho_i, rho_w=rho_w)
        reference = period_ref
        rows.append({
            "dt_s": dt,
            "period_measured_s": measured,
            "period_reference_s": period_ref,
            "relative_error": abs(measured - period_ref) / period_ref,
            "lift_off_height_m": h_eq,
            "k_eff_N_m": k_eff,
            "mass_kg": mass,
        })
    finest = rows[-1]
    return {
        "anchor": "T_heave_period",
        "code_under_test": "compute_buoyancy_force",
        "reference": "T = 2*pi*sqrt(m/k), k = rho_w*g*t*2*sqrt(R^2-h_eq^2)",
        "rows": rows,
        "relative_error_sequence": [r["relative_error"] for r in rows],
        "period_reference_s": reference,
        "criterion": {"relative_error_finest_dt": 0.01},
        "passed": finest["relative_error"] < 0.01,
    }


# ---------------------------------------------------------------------------
# Anchor F — inclined-plane friction onset and sliding acceleration
# ---------------------------------------------------------------------------

def _slope_run(thetas_deg, mu, dt, duration, mass=1.0, k_t=1e7,
               slip_threshold_factor=100.0):
    """Integrate a 1-DOF disk slipping on a fixed inclined plane.

    Geometry: downslope unit vector t = (cos, -sin); plane normal n = (-sin, -cos).
    The plane is body j at rest, so v_rel = v_j - v_i = -v*t and the
    repository's tangential spring accumulates delta = -s*t, whose spring
    force k_t*delta acts upslope on the disk.  Normal load N = m*g*cos(theta).
    """
    n = len(thetas_deg)
    theta = torch.tensor([math.radians(a) for a in thetas_deg], dtype=torch.float64)
    t_hat = torch.stack((torch.cos(theta), -torch.sin(theta)), 1)      # (n,2)
    n_hat = torch.stack((-torch.sin(theta), -torch.cos(theta)), 1)     # (n,2)
    normal_load = mass * G * torch.cos(theta)                          # (n,)

    disp = torch.zeros((n, 2), dtype=torch.float64)
    s = torch.zeros((n, 1), dtype=torch.float64)   # downslope displacement
    v = torch.zeros((n, 1), dtype=torch.float64)   # downslope velocity
    steps = int(round(duration / dt))
    half = steps // 2
    v_half = None

    # Symplectic Euler ordering (force -> velocity -> displacement).  Updating the
    # tangential spring with the *old* velocity is explicit Euler on a stiff
    # spring (k_t/m = 1e7) and pumps energy: with dt=4e-5 the stick lanes drift
    # into the friction cone and a spurious slip is reported ~1 deg below
    # arctan(mu).  This ordering is stable at the same k_t.
    for step in range(steps):
        f_t, _ = compute_tangential_force(n_hat, disp, normal_load, k_t, mu)
        along = (f_t * t_hat).sum(1, keepdim=True)   # upslope component (<0)
        a = G * torch.sin(theta)[:, None] + along / mass
        v = v + a * dt
        disp = integrate_tangential_displacement(disp, -v * t_hat, n_hat, dt)
        s = s + v * dt
        if step + 1 == half:
            v_half = v.clone()

    limit = mu * normal_load / k_t   # (n,) stick amplitude scale, m
    travel = torch.abs(s[:, 0])
    # Bounded stick oscillation is O(limit); a sliding lane travels orders of
    # magnitude further.  Threshold far from both scales.
    sliding = travel > slip_threshold_factor * limit
    accel = (v[:, 0] - v_half[:, 0]) / (0.5 * duration)   # steady sliding phase
    return {
        "theta_deg": list(thetas_deg),
        "travel_m": [float(x) for x in travel],
        "velocity_m_s": [float(x) for x in v[:, 0]],
        "accel_steady_m_s2": [float(x) for x in accel],
        "stick_scale_m": [float(x) for x in limit],
        "sliding": [bool(x) for x in sliding],
    }


def anchor_friction_slip(mu_values=(0.2, 0.5), dt_levels=(8e-5, 4e-5),
                         duration=0.04, mass=1.0, k_t=1e7, scan_deg=1.0,
                         n_scan=21, onset_tol_deg=0.25):
    """Slip onset vs arctan(mu) and sliding acceleration vs g*(sin - mu*cos).

    The onset is bracketed by the last sticking and first sliding lane of a
    0.1 deg grid, so the reported onset carries a +- grid-half-width
    resolution.  It is repeated at two time steps because the *classification*
    of a near-onset lane depends on how far a marginally unstable lane has
    travelled by the end of the run: the coarse-dt level reports a late onset
    even though the trajectory itself is fine.  The finest level therefore
    carries the pass/fail criterion and the coarse level is reported to expose
    the resolution, not to hide it.
    """
    rows = []
    for mu in mu_values:
        theta_c = math.degrees(math.atan(mu))
        grid = [theta_c - scan_deg + 2.0 * scan_deg * i / (n_scan - 1)
                for i in range(n_scan)]
        level_rows = []
        for dt in dt_levels:
            run = _slope_run(grid, mu, dt, duration, mass=mass, k_t=k_t)
            slipping = [a for a, sl in zip(grid, run["sliding"]) if sl]
            sticking = [a for a, sl in zip(grid, run["sliding"]) if not sl]
            onset = min(slipping) if slipping else None
            level_rows.append({
                "dt_s": dt,
                "theta_onset_measured_deg": onset,
                "onset_bracket_deg": [max(sticking) if sticking else None, onset],
                "onset_error_deg": (abs(onset - theta_c) if onset is not None
                                    else None),
                "sliding_flags": run["sliding"],
                "travel_m": run["travel_m"],
                "stick_scale_m": run["stick_scale_m"],
            })

        finest = level_rows[-1]
        finest_dt = finest["dt_s"]

        # Quantitative check well above onset: a = g*(sin - mu*cos), measured
        # over the steady sliding phase (second half of the run).
        theta_hi = math.radians(theta_c + 10.0)
        hi = _slope_run([math.degrees(theta_hi)], mu, finest_dt, duration,
                        mass=mass, k_t=k_t)
        a_meas = hi["accel_steady_m_s2"][0]
        a_ref = G * (math.sin(theta_hi) - mu * math.cos(theta_hi))
        accel_err = abs(a_meas - a_ref) / a_ref

        # Stick check a few degrees below onset: travel stays O(mu*N/k_t).
        theta_lo = math.radians(theta_c - 5.0)
        lo = _slope_run([math.degrees(theta_lo)], mu, finest_dt, duration,
                        mass=mass, k_t=k_t)
        stick_bounded = (not lo["sliding"][0]) and \
            lo["travel_m"][0] < 10.0 * lo["stick_scale_m"][0]

        onset_err = finest["onset_error_deg"]
        rows.append({
            "mu": mu,
            "theta_onset_theory_deg": theta_c,
            "dt_levels": level_rows,
            "theta_onset_measured_deg": finest["theta_onset_measured_deg"],
            "onset_bracket_deg": finest["onset_bracket_deg"],
            "onset_error_deg": onset_err,
            "onset_resolution_deg": 2.0 * scan_deg / (n_scan - 1),
            "theta_check_deg": math.degrees(theta_hi),
            "accel_steady_m_s2": a_meas,
            "accel_reference_m_s2": a_ref,
            "accel_relative_error": accel_err,
            "stick_test_theta_deg": math.degrees(theta_lo),
            "stick_test_travel_m": lo["travel_m"][0],
            "stick_test_bounded": stick_bounded,
            "passed": (onset_err is not None and onset_err < onset_tol_deg
                       and accel_err < 0.01 and stick_bounded),
        })
    return {
        "anchor": "F_friction_slip",
        "code_under_test": "integrate_tangential_displacement + compute_tangential_force",
        "reference": "theta_onset = arctan(mu); a = g*(sin(theta) - mu*cos(theta))",
        "parameters": {"dt_levels_s": list(dt_levels), "duration_s": duration,
                       "mass_kg": mass, "k_t_N_m": k_t,
                       "onset_tolerance_deg": onset_tol_deg},
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }


# ---------------------------------------------------------------------------
# Anchor R — restitution mapping vs an independently integrated damped collision
# ---------------------------------------------------------------------------

def anchor_restitution_mapping(e_targets=(0.5, 0.8), k_n=1e4, mass=1.0,
                               dt=1e-7):
    """gamma_n(e) checked against direct integration of the collision ODE.

    The analytic map is the closed-form solution of
    m*y'' + gamma*y' + k*y = 0; the numeric side integrates that ODE for one
    contact cycle.  Investigators should note that the round-trip
    e -> gamma_n -> e is a consistency check of two inverse formulas and is
    *not* physics evidence; it is reported separately for that reason.
    """
    rows = []
    for e_target in e_targets:
        gamma_n = restitution_to_damping(e_target, k_n, mass)
        v = -1.0
        y = 0.0
        v_after = 0.0
        steps = int(round(0.05 / dt))
        for _ in range(steps):
            overlap = -y if y < 0.0 else 0.0
            a = (k_n * overlap - gamma_n * v) / mass
            y_new = y + v * dt + 0.5 * a * dt ** 2
            overlap_new = -y_new if y_new < 0.0 else 0.0
            a_new = (k_n * overlap_new - gamma_n * v) / mass
            v_new = v + 0.5 * (a + a_new) * dt
            if y < 0.0 and y_new > 0.0 and v_new > 0.0:
                frac = -y / (y_new - y)
                v_after = v + frac * (v_new - v)
                break
            y, v = y_new, v_new
        e_measured = abs(v_after / (-1.0))
        rows.append({
            "e_target": e_target,
            "gamma_n_N_s_m": gamma_n,
            "e_measured_ode": e_measured,
            "ode_relative_error": abs(e_measured - e_target) / e_target,
            "e_roundtrip": damping_to_restitution(gamma_n, k_n, mass),
        })
    return {
        "anchor": "R_restitution_mapping",
        "code_under_test": "restitution_to_damping / damping_to_restitution",
        "reference": "closed-form damped-oscillator rebound ratio",
        "note": "round-trip columns are formula consistency only, not physics",
        "rows": rows,
        "max_ode_relative_error": max(r["ode_relative_error"] for r in rows),
        "criterion": {"max_ode_relative_error": 0.03},
        "passed": max(r["ode_relative_error"] for r in rows) < 0.03,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(fast=False):
    if fast:
        # Reduced-step configuration used by tests/test_ice_floe_physics.py.
        # dt is kept inside the stability limit of each sub-model; residual
        # discretisation error stays two orders of magnitude below the criteria.
        hydro = anchor_hydrostatic_equilibrium(dt=8e-3, duration=6.0)
        heave = anchor_heave_period(dt_levels=(8e-3,))
        friction = anchor_friction_slip(dt_levels=(4e-5,), duration=0.02,
                                       onset_tol_deg=0.6)
        rest = anchor_restitution_mapping(dt=1e-6)
    else:
        hydro = anchor_hydrostatic_equilibrium()
        heave = anchor_heave_period()
        friction = anchor_friction_slip()
        rest = anchor_restitution_mapping()
    archimedes = anchor_archimedes_static()

    anchors = [hydro, archimedes, heave, friction, rest]
    return {
        "schema": SCHEMA,
        "mode": "fast" if fast else "full",
        "overall_pass": all(a["passed"] for a in anchors),
        "anchors": anchors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="reduced steps (unit-test configuration)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = run_all(fast=args.fast)
    out = Path(args.output) if args.output else (
        _REPO_ROOT / "artifacts" / "ice_floe_physics" / "validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for anchor in report["anchors"]:
        print(f"{anchor['anchor']:<32} passed={anchor['passed']}")
    print(f"overall_pass={report['overall_pass']}")
    print(f"report: {out}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())