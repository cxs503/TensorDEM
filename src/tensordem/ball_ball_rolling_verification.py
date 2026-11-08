"""Numerical verification drivers for ball-ball rolling-friction contact mechanics.

Six analytic-vs-numerical verification faces, one function per face, each
returning plain dictionaries of numbers so the test suite and the report script
consume exactly the same evidence:

1. :func:`verify_tangential_stiffness` -- Cundall-Strack ``k_t`` recovery.
2. :func:`verify_coulomb_limit` -- hard Coulomb cone assertion (10 cases).
3. :func:`verify_rolling_moment` -- ``M_r = mu_r r_eff F_n`` (9 cases).
4. :func:`verify_rotational_damping` -- ``omega(t) = omega_0 exp(-c_r t/I)``.
5. :func:`verify_rolling_sliding_transition` -- ``v_t = omega r`` (exact
   kinematics) and the ``r_eff`` convention of the campaign spec, plus the
   sustained-creep balance ``|F_t| r_j = M_r`` when rolling resistance is on.
6. :func:`verify_incline_stability` -- two-ball stack on an incline: static
   equilibrium capacity ``mu_r r_eff / r_b`` vs the plane-limit
   ``mu_r >= tan(theta_cr)``, plus a 1-DOF dynamic confirm.
7. :func:`verify_energy_ledger` -- collision energy audit with an exact
   semi-implicit work identity; residual <= 1e-4.

All quantities SI.  Report: ``docs/ball_ball_rolling_contact.md``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .ball_ball_rolling import (
    COULOMB_CASES,
    ROLLING_NORMAL_FORCES,
    ROLLING_RADIUS_PAIRS,
    RollingFrictionBallBall,
    TangentialState,
    advance_tangential,
    analytic_incline_critical_angle_deg,
    analytic_slip_decay_rate,
    analytic_stack_critical_angle_deg,
    analytic_transition_spin_ratio,
    check_coulomb_limit_case,
    check_rolling_moment_case,
    effective_radius,
    simulate_tangential_spring,
    solid_sphere_inertia,
    solid_sphere_mass,
)
from .contact_tangential import restitution_to_damping

GRAVITY = 9.81
NORMAL = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
TANGENT = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
BINORMAL = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)


# ---------------------------------------------------------------------------
# 1. Cundall-Strack tangential stiffness recovery
# ---------------------------------------------------------------------------
def verify_tangential_stiffness(
    cases: Optional[list[tuple[float, float, float]]] = None,
) -> dict:
    """Regress ``|F_t|`` on ``|delta_t|`` in the elastic branch (<=1 % error)."""
    if cases is None:
        cases = [(2.0e4, 0.5, 5.0), (5.0e4, 0.6, 20.0), (8.0e3, 0.4, 2.0)]
    rows = []
    for k_t, mu_s, f_n in cases:
        n_steps = 200
        slip_speed = 1.0e-4
        # keep the peak elastic force near 10 % of the Coulomb cone
        dt = 0.1 * mu_s * f_n / (k_t * slip_speed * n_steps)
        rows.append(
            simulate_tangential_spring(
                f_n=f_n, k_t=k_t, mu_s=mu_s, dt=dt, n_steps=n_steps,
                slip_speed=slip_speed,
            )
        )
    worst = max(r["k_t_rel_err"] for r in rows)
    return {
        "cases": rows,
        "n_cases": len(rows),
        "worst_k_t_rel_err": worst,
        "all_elastic": all(r["elastic"] for r in rows),
        "criterion": 1.0e-2,
        "passed": worst <= 1.0e-2 and all(r["elastic"] for r in rows),
    }


# ---------------------------------------------------------------------------
# 2. Coulomb limit
# ---------------------------------------------------------------------------
def verify_coulomb_limit(cases: Optional[list[tuple]] = None) -> dict:
    """Drive ten independent contacts into the friction cone."""
    cases = COULOMB_CASES if cases is None else cases
    rows = [check_coulomb_limit_case(c) for c in cases]
    worst = max(r["cap_rel_err"] for r in rows)
    worst_dir = max(abs(r["direction_cos"] - 1.0) for r in rows)
    return {
        "cases": rows,
        "n_cases": len(rows),
        "worst_cap_rel_err": worst,
        "worst_direction_err": worst_dir,
        "all_saturated": all(r["saturated"] for r in rows),
        "criterion": 1.0e-12,
        "passed": (
            len(rows) >= 10
            and worst <= 1.0e-12
            and worst_dir <= 1.0e-12
            and all(r["saturated"] for r in rows)
        ),
    }


# ---------------------------------------------------------------------------
# 3. Rolling-resistance moment
# ---------------------------------------------------------------------------
def verify_rolling_moment(mu_r: float = 0.05) -> dict:
    rows = [
        check_rolling_moment_case(r_i, r_j, f_n, mu_r)
        for r_i, r_j in ROLLING_RADIUS_PAIRS
        for f_n in ROLLING_NORMAL_FORCES
    ]
    worst = max(r["moment_rel_err"] for r in rows)
    worst_legacy = max(r["legacy_rel_err"] for r in rows)
    worst_cos = max(abs(r["cos_opposition"] - 1.0) for r in rows)
    return {
        "cases": rows,
        "n_cases": len(rows),
        "worst_moment_rel_err": worst,
        "worst_legacy_rel_err": worst_legacy,
        "worst_opposition_err": worst_cos,
        "criterion": 2.0e-2,
        "passed": len(rows) >= 9 and worst <= 2.0e-2 and worst_cos <= 1.0e-9,
    }


# ---------------------------------------------------------------------------
# 4. Rotational damping (exponential spin-down)
# ---------------------------------------------------------------------------
def verify_rotational_damping(
    *,
    radius: float = 0.05,
    density: float = 2500.0,
    omega_0: float = 20.0,
    tau: float = 0.05,
    dt: float = 2.5e-5,
    n_steps: int = 4000,
) -> dict:
    """Semi-implicit integration of ``I domega/dt = -c_r omega``."""
    mass = solid_sphere_mass(radius, density)
    inertia = solid_sphere_inertia(radius, mass)
    c_r = inertia / tau
    omega = torch.tensor([0.0, 0.0, omega_0], dtype=torch.float64)
    law = RollingFrictionBallBall(mu_r=0.0, c_r=c_r, model="viscous")
    times, measured = [], []
    for k in range(n_steps + 1):
        t = k * dt
        times.append(t)
        measured.append(float(omega.norm()))
        if k == n_steps:
            break
        torque, _ = law.moment(
            omega, 0.0, effective_radius(radius, radius), NORMAL
        )
        omega = omega + torque / inertia * dt

    t_t = torch.tensor(times, dtype=torch.float64)
    m_t = torch.tensor(measured, dtype=torch.float64)
    analytic = omega_0 * torch.exp(-t_t / tau)
    ss_res = float(((m_t - analytic) ** 2).sum())
    ss_tot = float(((analytic - analytic.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rel_max = float(((m_t - analytic).abs() / analytic).max())
    mask = m_t > 1.0e-3 * omega_0
    slope = float(
        (t_t[mask] * torch.log(m_t[mask] / omega_0)).sum()
        / (t_t[mask] ** 2).sum()
    )
    tau_fit = -1.0 / slope
    return {
        "radius": radius,
        "density": density,
        "mass": mass,
        "inertia": inertia,
        "c_r": c_r,
        "tau_input": tau,
        "tau_fitted": tau_fit,
        "tau_rel_err": abs(tau_fit - tau) / tau,
        "omega_0": omega_0,
        "omega_final": float(m_t[-1]),
        "decays": math.log10(omega_0 / float(m_t[-1])),
        "r2": r2,
        "max_rel_err": rel_max,
        "criterion_r2": 0.999,
        "dt": dt,
        "n_steps": n_steps,
        "times": times,
        "measured": measured,
        "analytic": analytic.tolist(),
        "passed": r2 >= 0.999 and abs(tau_fit - tau) / tau <= 1.0e-2,
    }


# ---------------------------------------------------------------------------
# 5. Rolling-sliding transition
# ---------------------------------------------------------------------------
def simulate_roll_slip_transition(
    *,
    r_i: float = 0.05,
    r_j: float = 0.05,
    density: float = 2500.0,
    f_n: float = 10.0,
    mu_s: float = 0.4,
    k_t: float = 2.0e5,
    omega_0: float = 40.0,
    mu_r: float = 0.0,
    c_r_roll: float = 0.0,
    dt: float = 5.0e-5,
    max_steps: int = 40_000,
) -> dict:
    """Sphere ``j`` spinning on a fixed sphere ``i``: friction spins it up.

    The normal coordinate of ``j`` is held (prescribed load ``F_n``); the
    tangential velocity and the rotation are free.  Integration stops at the
    first sign change of the contact slip (pure rolling) when ``mu_r = 0``, or
    runs to a steady state when rolling resistance is active.
    """
    mass = solid_sphere_mass(r_j, density)
    inertia = solid_sphere_inertia(r_j, mass)
    r_eff = effective_radius(r_i, r_j)
    roll_law = RollingFrictionBallBall(mu_r=mu_r, c_r=c_r_roll, model="viscous")

    v = torch.zeros(3, dtype=torch.float64)
    omega = omega_0 * BINORMAL.clone()
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))

    ft_hist: list[float] = []
    mr_hist: list[float] = []
    v_hist: list[float] = []
    w_hist: list[float] = []
    slip_hist: list[float] = []
    prev_s = None
    transition = None

    for k in range(max_steps):
        s = v - torch.cross(r_j * omega, NORMAL, dim=-1)
        s_t = s - (s @ NORMAL) * NORMAL
        f_t = advance_tangential(state, s_t, NORMAL, f_n, k_t, mu_s, dt)
        # rolling couple opposes the rolling (omega_i = 0, so omega_rel = omega)
        torque, _ = roll_law.moment(omega, f_n, r_eff, NORMAL)
        v = v + f_t / mass * dt
        torque_total = torch.cross(-r_j * NORMAL, f_t, dim=-1) + torque
        omega = omega + torque_total / inertia * dt

        s_val = float(s_t @ TANGENT)
        slip_hist.append(s_val)
        ft_hist.append(float(f_t.norm()))
        mr_hist.append(float(torque.norm()))
        v_hist.append(float(v.norm()))
        w_hist.append(float(omega.norm()))

        if mu_r == 0.0 and prev_s is not None and prev_s * s_val <= 0.0:
            frac = abs(prev_s) / (abs(prev_s) + abs(s_val))
            transition = {
                "time": (k - 1) * dt + frac * dt,
                "v": float(v.norm()),
                "w": float(omega.norm()),
            }
            break
        prev_s = s_val

    result = {
        "r_i": r_i,
        "r_j": r_j,
        "r_eff": r_eff,
        "f_n": f_n,
        "mu_s": mu_s,
        "k_t": k_t,
        "omega_0": omega_0,
        "mu_r": mu_r,
        "mass": mass,
        "inertia": inertia,
        "dt": dt,
        "n_steps": len(ft_hist),
        "analytic_spin_ratio": analytic_transition_spin_ratio(mass, inertia, r_j),
        "analytic_slip_rate": analytic_slip_decay_rate(
            mu_s, f_n, mass, inertia, r_j
        ),
        "analytic_transition_time": (
            r_j * omega_0 * mass
            / (mu_s * f_n * (1.0 + mass * r_j * r_j / inertia))
        ),
        "slip_history": slip_hist,
        "v_history": v_hist,
        "w_history": w_hist,
    }

    if transition is not None:
        ratio_exact = transition["v"] / (r_j * transition["w"])
        ratio_eff = transition["v"] / (r_eff * transition["w"])
        result.update(
            {
                "transition_time": transition["time"],
                "transition_v": transition["v"],
                "transition_w": transition["w"],
                "transition_spin_ratio": transition["w"] / omega_0,
                "spin_ratio_rel_err": abs(
                    transition["w"] / omega_0
                    - result["analytic_spin_ratio"]
                ) / result["analytic_spin_ratio"],
                "time_rel_err": abs(
                    transition["time"] - result["analytic_transition_time"]
                ) / result["analytic_transition_time"],
                "rolling_condition_exact": ratio_exact,
                "rolling_condition_exact_err": abs(ratio_exact - 1.0),
                "rolling_condition_r_eff": ratio_eff,
                "rolling_condition_r_eff_err": abs(ratio_eff - 1.0),
                "r_j_over_r_eff": r_j / r_eff,
                "ft_steady": ft_hist[-1],
                "mr_steady": mr_hist[-1],
            }
        )
    else:
        n_tail = max(50, len(ft_hist) // 10)
        ft_steady = sum(ft_hist[-n_tail:]) / n_tail
        mr_steady = sum(mr_hist[-n_tail:]) / n_tail
        v_steady = sum(v_hist[-n_tail:]) / n_tail
        w_steady = sum(w_hist[-n_tail:]) / n_tail
        required = mr_steady / r_j
        result.update(
            {
                "transition_time": None,
                "ft_steady": ft_steady,
                "mr_steady": mr_steady,
                "v_steady": v_steady,
                "w_steady": w_steady,
                "ft_balance_required": required,
                "balance_rel_err": (
                    abs(ft_steady - required) / required if required > 0 else None
                ),
                "rolling_condition_exact_err": None,
                "rolling_condition_r_eff_err": None,
                "creep_ratio": v_steady / (r_j * w_steady) if w_steady else None,
                "rolling_limit_rel_err": abs(
                    mr_steady - mu_r * r_eff * f_n
                ) / (mu_r * r_eff * f_n),
            }
        )
    return result


def simulate_driven_rolling(
    *,
    r_i: float = 0.05,
    r_j: float = 0.05,
    density: float = 2500.0,
    f_n: float = 10.0,
    mu_s: float = 0.4,
    mu_r: float = 0.10,
    k_t: float = 2.0e5,
    c_r_roll: float = 1.0e-2,
    drive_factor: float = 1.0,
    omega_0: float = 80.0,
    dt: float = 1.0e-4,
    n_steps: int = 6_000,
) -> dict:
    """Steady rolling of a ball on a fixed sphere against rolling resistance.

    A constant tangential drive force acts on the centre of ``j`` (the DEM
    analogue of gravity on an incline).  In steady rolling the drive must equal

        F_drive = M_r / r_j = mu_r r_eff F_n / r_j ,

    so with ``drive_factor = 1`` the velocity must stay constant, below one the
    ball decelerates and above one it accelerates.
    """
    mass = solid_sphere_mass(r_j, density)
    inertia = solid_sphere_inertia(r_j, mass)
    r_eff = effective_radius(r_i, r_j)
    limit = mu_r * r_eff * f_n
    f_balance = limit / r_j
    f_drive = drive_factor * f_balance
    roll_law = RollingFrictionBallBall(mu_r=mu_r, c_r=c_r_roll, model="viscous")

    # start exactly in the rolling state, motion along +x (slip-free)
    omega = -omega_0 * BINORMAL.clone()
    v = torch.cross(r_j * omega, NORMAL, dim=-1)  # = -r_j omega_0 x_hat
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))
    drive = f_drive * TANGENT  # pushes in +x (down-slope analogue)

    v_hist: list[float] = []
    ft_hist: list[float] = []
    mr_hist: list[float] = []
    slip_hist: list[float] = []
    for _ in range(n_steps):
        s = v - torch.cross(r_j * omega, NORMAL, dim=-1)
        s_t = s - (s @ NORMAL) * NORMAL
        f_t = advance_tangential(state, s_t, NORMAL, f_n, k_t, mu_s, dt)
        torque, _ = roll_law.moment(omega, f_n, r_eff, NORMAL)
        v = v + (f_t + drive) / mass * dt
        torque_total = torch.cross(-r_j * NORMAL, f_t, dim=-1) + torque
        omega = omega + torque_total / inertia * dt
        v_hist.append(float(v.norm()))
        ft_hist.append(float(f_t.norm()))
        mr_hist.append(float(torque.norm()))
        slip_hist.append(float(s_t.norm()))

    n_tail = max(50, n_steps // 5)
    v_early = sum(v_hist[n_tail:2 * n_tail]) / n_tail
    v_late = sum(v_hist[-n_tail:]) / n_tail
    ft_late = sum(ft_hist[-n_tail:]) / n_tail
    mr_late = sum(mr_hist[-n_tail:]) / n_tail
    w_late = omega.norm()
    accel = (v_late - v_early) / ((n_steps - 1.5 * n_tail) * dt)
    return {
        "r_i": r_i,
        "r_j": r_j,
        "r_eff": r_eff,
        "f_n": f_n,
        "mu_r": mu_r,
        "drive_factor": drive_factor,
        "f_drive": f_drive,
        "f_balance_analytic": f_balance,
        "v_early": v_early,
        "v_late": v_late,
        "accel": accel,
        "ft_late": ft_late,
        "mr_late": mr_late,
        "rolling_limit": limit,
        "moment_rel_err": abs(mr_late - limit) / limit,
        "force_balance_rel_err": abs(ft_late - f_balance) / f_balance,
        "rolling_condition": float(v_late) / (r_j * float(w_late)),
        "rolling_condition_err": abs(
            float(v_late) / (r_j * float(w_late)) - 1.0
        ),
        "slip_late": sum(slip_hist[-n_tail:]) / n_tail,
        "dt": dt,
        "n_steps": n_steps,
    }


def verify_driven_rolling(mu_r: float = 0.10) -> dict:
    """Force balance of rolling resistance in steady rolling."""
    on_balance = simulate_driven_rolling(mu_r=mu_r, drive_factor=1.0)
    under = simulate_driven_rolling(mu_r=mu_r, drive_factor=0.8)
    over = simulate_driven_rolling(mu_r=mu_r, drive_factor=1.2)
    return {
        "on_balance": on_balance,
        "under_drive": under,
        "over_drive": over,
        "moment_rel_err": on_balance["moment_rel_err"],
        "force_balance_rel_err": on_balance["force_balance_rel_err"],
        "rolling_condition_err": on_balance["rolling_condition_err"],
        "steady": abs(on_balance["accel"]) < 0.05,
        "decelerates": under["accel"] < 0.0,
        "accelerates": over["accel"] > 0.0,
        "criterion": 2.0e-2,
        "passed": (
            on_balance["moment_rel_err"] <= 2.0e-2
            and on_balance["force_balance_rel_err"] <= 2.0e-2
            and on_balance["rolling_condition_err"] <= 3.0e-2
            and abs(on_balance["accel"]) < 0.05
            and under["accel"] < 0.0
            and over["accel"] > 0.0
        ),
    }


def verify_rolling_sliding_transition() -> dict:
    """Transition in exact-kinematics and ``r_eff`` conventions + creep balance."""
    plane = simulate_roll_slip_transition(mu_r=0.0, r_i=1.0e9, r_j=0.05)
    equal = simulate_roll_slip_transition(mu_r=0.0, r_i=0.05, r_j=0.05)
    ratio_20 = simulate_roll_slip_transition(mu_r=0.0, r_i=1.0, r_j=0.05)
    cases = {
        "plane_limit": plane,
        "equal_spheres": equal,
        "ratio_20": ratio_20,
    }
    worst_exact = max(
        plane["rolling_condition_exact_err"],
        equal["rolling_condition_exact_err"],
        ratio_20["rolling_condition_exact_err"],
    )
    return {
        "cases": cases,
        "worst_exact_rolling_err": worst_exact,
        "plane_limit_r_eff_err": plane["rolling_condition_r_eff_err"],
        "plane_limit_exact_err": plane["rolling_condition_exact_err"],
        "spin_ratio_rel_err_plane": plane["spin_ratio_rel_err"],
        "spin_ratio_rel_err_equal": equal["spin_ratio_rel_err"],
        "time_rel_err_plane": plane["time_rel_err"],
        "criterion": 3.0e-2,
        "passed": (
            worst_exact <= 3.0e-2
            and plane["rolling_condition_r_eff_err"] <= 3.0e-2
            and plane["spin_ratio_rel_err"] <= 3.0e-2
            and max(plane["time_rel_err"], equal["time_rel_err"]) <= 3.0e-2
        ),
    }


# ---------------------------------------------------------------------------
# 6. Inclined-plane stack stability
# ---------------------------------------------------------------------------
def stack_equilibrium(
    theta_deg: float,
    *,
    mu_r: float,
    mass_a: float,
    mass_b: float,
    r_a: float,
    r_b: float,
) -> dict:
    """Solve the static equilibrium of a two-sphere stack on an incline.

    Unknowns ``x = [f_p, f_c, M_p, M_c]``: plane friction on the lower ball,
    pair friction on the upper ball, plane rolling couple on the lower ball,
    pair rolling couple on the upper ball.  The four linear equations are the
    translational equilibrium of each ball and the rotational equilibrium of
    each ball about its own centre.  Rolling resistance caps each couple at
    ``mu_r r_eff F_n``; ``theta_cr`` is where the utilisation reaches one.
    """
    th = math.radians(theta_deg)
    sin, cos = math.sin(th), math.cos(th)
    n_p = (mass_a + mass_b) * GRAVITY * cos
    n_c = mass_b * GRAVITY * cos
    r_eff = effective_radius(r_a, r_b)
    a = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [r_a, r_a, 1.0, -1.0],
            [0.0, r_b, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor(
        [-mass_a * GRAVITY * sin, -mass_b * GRAVITY * sin, 0.0, 0.0],
        dtype=torch.float64,
    )
    f_p, f_c, m_p, m_c = torch.linalg.solve(a, b).tolist()
    cap_p = mu_r * r_a * n_p
    cap_c = mu_r * r_eff * n_c
    util_p = abs(m_p) / cap_p
    util_c = abs(m_c) / cap_c
    return {
        "theta_deg": theta_deg,
        "n_plane": n_p,
        "n_pair": n_c,
        "f_plane": f_p,
        "f_pair": f_c,
        "m_plane": m_p,
        "m_pair": m_c,
        "cap_plane": cap_p,
        "cap_pair": cap_c,
        "util_plane": util_p,
        "util_pair": util_c,
        "utilisation": max(util_p, util_c),
        "radius_ratio": r_b / r_a,
        "tan_theta": math.tan(th),
    }


def critical_stack_angle_deg(
    *, mu_r: float = 0.2, mass_a: float = 1.0, mass_b: float = 1.0,
    r_a: float = 0.05, r_b: float = 0.05,
) -> float:
    """Bisect the incline angle at which the rolling capacity is exhausted."""
    lo, hi = 0.0, 45.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        eq = stack_equilibrium(
            mid, mu_r=mu_r, mass_a=mass_a, mass_b=mass_b, r_a=r_a, r_b=r_b
        )
        if eq["utilisation"] <= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_ball_on_incline(
    theta_deg: float,
    *,
    mu_r: float = 0.2,
    radius: float = 0.05,
    density: float = 2500.0,
    k_r_over_i: float = 4.0e4,
    damping_time: float = 0.02,
    dt: float = 2.0e-5,
    duration: float = 1.0,
) -> dict:
    """1-DOF dynamic confirm of the plane-limit criterion ``tan(theta_cr)=mu_r``.

    Pure rolling (``v = r omega``) with an elastic-plastic rolling spring:
    ``m_eff dv/dt = m g sin - M_r / r`` with ``m_eff = m + I/r^2`` and the
    rolling couple elastic in the rolling angle, capped at ``mu_r r N``.
    """
    th = math.radians(theta_deg)
    mass = solid_sphere_mass(radius, density)
    inertia = solid_sphere_inertia(radius, mass)
    m_eff = mass + inertia / (radius * radius)
    n = mass * GRAVITY * math.cos(th)
    cap = mu_r * radius * n
    k_r = k_r_over_i * inertia
    c_visc = m_eff / damping_time  # rolling/tangential dashpot, DEM standard
    v = 0.0
    phi = 0.0
    m_roll = 0.0
    n_steps = int(duration / dt)
    tail: list[float] = []
    for k in range(n_steps):
        m_roll = max(-cap, min(cap, k_r * phi))
        a = (mass * GRAVITY * math.sin(th) - m_roll / radius - c_visc * v) / m_eff
        v += a * dt
        phi += v / radius * dt
        if k > 0.8 * n_steps:
            tail.append(abs(v))
    v_tail = max(tail) if tail else abs(v)
    return {
        "theta_deg": theta_deg,
        "theta_cr_analytic_deg": analytic_incline_critical_angle_deg(mu_r),
        "mu_r": mu_r,
        "v_tail_max": v_tail,
        "rolling_moment_final": m_roll,
        "cap": cap,
        "c_visc": c_visc,
        "held": v_tail < 1.0e-3,
        "dt": dt,
        "duration": duration,
    }


def verify_incline_stability(mu_r: float = 0.2) -> dict:
    """Two-ball stack (static capacity) + plane-limit dynamic confirm."""
    configs = [
        ("equal_spheres", 0.05, 0.05),
        ("ratio_2", 0.10, 0.05),
        ("ratio_20", 1.00, 0.05),
        ("ratio_50", 2.50, 0.05),
    ]
    rows = []
    for name, r_a, r_b in configs:
        m_a = solid_sphere_mass(r_a, 2500.0)
        m_b = solid_sphere_mass(r_b, 2500.0)
        measured = critical_stack_angle_deg(
            mu_r=mu_r, mass_a=m_a, mass_b=m_b, r_a=r_a, r_b=r_b
        )
        analytic = analytic_stack_critical_angle_deg(mu_r, m_a, m_b, r_a, r_b)
        plane_limit = analytic_incline_critical_angle_deg(mu_r)
        rows.append(
            {
                "case": name,
                "r_a": r_a,
                "r_b": r_b,
                "radius_ratio": r_b / r_a,
                "theta_cr_measured_deg": measured,
                "theta_cr_analytic_deg": analytic,
                "rel_err_vs_analytic": abs(measured - analytic) / analytic,
                "rel_err_vs_plane_limit": abs(measured - plane_limit) / plane_limit,
                "tan_measured": math.tan(math.radians(measured)),
                "mu_r_over_tan": mu_r / math.tan(math.radians(measured)),
            }
        )
    theta_cr = analytic_incline_critical_angle_deg(mu_r)
    below = simulate_ball_on_incline(0.9 * theta_cr, mu_r=mu_r)
    above = simulate_ball_on_incline(1.1 * theta_cr, mu_r=mu_r)
    plane_row = [r for r in rows if r["case"] == "ratio_50"][0]
    return {
        "mu_r": mu_r,
        "stack_cases": rows,
        "dynamic_below": below,
        "dynamic_above": above,
        "worst_rel_err_vs_analytic": max(r["rel_err_vs_analytic"] for r in rows),
        "plane_limit_rel_err": plane_row["rel_err_vs_plane_limit"],
        "criterion": 5.0e-2,
        "passed": (
            max(r["rel_err_vs_analytic"] for r in rows) <= 5.0e-2
            and plane_row["rel_err_vs_plane_limit"] <= 5.0e-2
            and below["held"]
            and not above["held"]
        ),
    }


# ---------------------------------------------------------------------------
# 7. Energy ledger of a ball-ball collision
# ---------------------------------------------------------------------------
def simulate_collision(
    *,
    r_a: float = 0.05,
    r_b: float = 0.05,
    density: float = 2500.0,
    k_n: float = 1.0e5,
    k_t: float = 2.0e4,
    restitution: float = 1.0,
    gamma_scale: float = 1.0,
    mu_s: float = 0.0,
    mu_r: float = 0.0,
    v_approach: float = 0.5,
    v_t: float = 0.0,
    omega_0: float = 0.0,
    dt: float = 1.0e-6,
    max_time: float = 0.05,
) -> dict:
    """Two-sphere collision with a complete dissipation ledger.

    The work of every contribution is accumulated with the mid-step velocity of
    semi-implicit Euler, for which the identity

        K(t+dt) - K(t) = [F . (v + v_new)/2 + tau . (omega + omega_new)/2] dt

    is algebraically exact.  The ledger therefore audits the physical
    bookkeeping (which term takes which share) rather than the integration
    error, and the spring/dashpot split is cross-checked against the analytic
    restitution mapping ``K_n,rebound = e^2 K_n,impact``.
    """
    m_a = solid_sphere_mass(r_a, density)
    m_b = solid_sphere_mass(r_b, density)
    i_a = solid_sphere_inertia(r_a, m_a)
    i_b = solid_sphere_inertia(r_b, m_b)
    m_eff = m_a * m_b / (m_a + m_b)
    gamma_n = gamma_scale * restitution_to_damping(restitution, k_n, m_eff)
    r_eff = effective_radius(r_a, r_b)
    roll_law = RollingFrictionBallBall(mu_r=mu_r, c_r=0.0, model="viscous")

    gap0 = 5.0e-4
    x_a = torch.zeros(3, dtype=torch.float64)
    x_b = torch.tensor([0.0, r_a + r_b + gap0, 0.0], dtype=torch.float64)
    # closing pair: a moves +y, b moves -y (approach speed = v_approach)
    v_a = torch.tensor([-v_t / 2.0, v_approach / 2.0, 0.0], dtype=torch.float64)
    v_b = torch.tensor([v_t / 2.0, -v_approach / 2.0, 0.0], dtype=torch.float64)
    w_a = omega_0 * BINORMAL.clone()
    w_b = -omega_0 * BINORMAL.clone()
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))

    def kinetic(va, wa, vb, wb) -> float:
        return (
            0.5 * m_a * float(va @ va)
            + 0.5 * i_a * float(wa @ wa)
            + 0.5 * m_b * float(vb @ vb)
            + 0.5 * i_b * float(wb @ wb)
        )

    k0 = kinetic(v_a, w_a, v_b, w_b)
    w_spring = w_dash = w_tang = w_roll = 0.0
    u_n_release = 0.0
    overlap_max = 0.0
    contact_steps = 0
    separated = False
    n_steps = int(max_time / dt)

    for _ in range(n_steps):
        n_vec = x_b - x_a
        dist = float(n_vec.norm())
        overlap = r_a + r_b - dist
        if overlap <= 0.0 and contact_steps > 0:
            separated = True
            break
        if overlap <= 0.0:
            x_a = x_a + v_a * dt
            x_b = x_b + v_b * dt
            continue
        contact_steps += 1
        overlap_max = max(overlap_max, overlap)
        n_hat = n_vec / dist
        v_rel = v_b - v_a
        v_n = float(v_rel @ n_hat)
        f_spring = k_n * overlap
        f_dash = -gamma_n * v_n
        f_n = f_spring + f_dash
        if f_n <= 0.0:
            # tension cut-off: the contact is lost and the stored elastic energy
            # is released (standard non-adhesive DEM contact); it is booked
            # explicitly so the ledger stays closed.
            u_n_release = 0.5 * k_n * overlap * overlap
            separated = True
            break
        s = v_rel - torch.cross(r_a * w_a + r_b * w_b, n_hat, dim=-1)
        s_t = s - (s @ n_hat) * n_hat
        f_t = advance_tangential(state, s_t, n_hat, f_n, k_t, mu_s, dt)
        torque_roll, _ = roll_law.moment(w_b - w_a, f_n, r_eff, n_hat)

        f_a = -f_n * n_hat - f_t
        f_b = f_n * n_hat + f_t
        t_a = torch.cross(r_a * n_hat, f_a, dim=-1) - torque_roll
        t_b = torch.cross(-r_b * n_hat, f_b, dim=-1) + torque_roll

        v_a_new = v_a + f_a / m_a * dt
        v_b_new = v_b + f_b / m_b * dt
        w_a_new = w_a + t_a / i_a * dt
        w_b_new = w_b + t_b / i_b * dt

        v_rel_mid = 0.5 * (v_b + v_b_new - v_a - v_a_new)
        w_rel_mid = 0.5 * (w_b + w_b_new - w_a - w_a_new)
        s_new = v_rel_mid - torch.cross(
            r_a * 0.5 * (w_a + w_a_new) + r_b * 0.5 * (w_b + w_b_new),
            n_hat,
            dim=-1,
        )
        s_new = s_new - (s_new @ n_hat) * n_hat

        w_spring += f_spring * float(v_rel_mid @ n_hat) * dt
        w_dash += f_dash * float(v_rel_mid @ n_hat) * dt
        w_tang += float(f_t @ s_new) * dt
        w_roll += float(torque_roll @ w_rel_mid) * dt

        x_a = x_a + v_a_new * dt
        x_b = x_b + v_b_new * dt
        v_a, v_b, w_a, w_b = v_a_new, v_b_new, w_a_new, w_b_new

    n_vec = x_b - x_a
    dist = float(n_vec.norm())
    n_hat_final = n_vec / dist
    v_rel_final = v_b - v_a
    overlap_final = max(0.0, r_a + r_b - dist)
    k_final = kinetic(v_a, w_a, v_b, w_b)
    # after a tension cut-off the contact no longer exists (its elastic energy
    # was already booked as u_n_release), so only count a still-open contact
    u_n_final = 0.0 if separated and u_n_release > 0.0 else 0.5 * k_n * overlap_final ** 2
    u_t_final = 0.5 * k_t * float(state.delta_t @ state.delta_t)

    d_dash = -w_dash
    d_roll = -w_roll
    d_tang = -(w_tang + u_t_final)
    dissipated = d_dash + d_roll + d_tang
    closure = abs(
        (k0 - u_n_release - u_n_final - u_t_final) - (k_final + dissipated)
    )
    return {
        "restitution": restitution,
        "mu_s": mu_s,
        "mu_r": mu_r,
        "k_n": k_n,
        "k_t": k_t,
        "m_a": m_a,
        "m_b": m_b,
        "m_eff": m_eff,
        "gamma_n": gamma_n,
        "gamma_scale": gamma_scale,
        "r_eff": r_eff,
        "v_approach": v_approach,
        "v_t": v_t,
        "omega_0": omega_0,
        "dt": dt,
        "k_initial": k0,
        "k_final": k_final,
        "k_final_rel": k_final / k0,
        "overlap_max": overlap_max,
        "overlap_final": overlap_final,
        "u_n_final": u_n_final,
        "u_t_final": u_t_final,
        "w_spring": w_spring,
        "w_dashpot_negative": w_dash,
        "w_tangential": w_tang,
        "w_rolling": w_roll,
        "d_dashpot": d_dash,
        "d_rolling": d_roll,
        "d_tangential": d_tang,
        "dissipated_total": dissipated,
        "u_n_release": u_n_release,
        "dissipated_rel": dissipated / k0,
        "unaccounted_rel": (k0 - k_final - dissipated - u_n_release
                            - u_n_final - u_t_final) / k0,
        "closure_abs": closure,
        "closure_rel": closure / k0,
        "v_rel_normal_final": float(v_rel_final @ n_hat_final),
        "restitution_measured": float(v_rel_final @ n_hat_final) / v_approach,
        "separated": separated,
        "contact_steps": contact_steps,
        "all_dissipative_nonnegative": min(d_dash, d_roll, d_tang) >= -1.0e-12 * k0,
    }


def verify_energy_ledger() -> dict:
    """Elastic limit, fully dissipative limit and an oblique frictional case."""
    elastic = simulate_collision(
        restitution=1.0, mu_s=0.0, mu_r=0.0, dt=1.0e-6, max_time=0.03
    )
    dissipative = simulate_collision(
        restitution=0.005, gamma_scale=10.0, mu_s=0.6, mu_r=0.10, v_t=0.0,
        omega_0=0.0, dt=1.0e-6, max_time=0.05,
    )
    oblique = simulate_collision(
        restitution=0.6, mu_s=0.4, mu_r=0.05, v_t=0.3, omega_0=15.0,
        dt=1.0e-6, max_time=0.03,
    )
    elastic_conservation = abs(elastic["k_final_rel"] + elastic["u_n_final"]
                               / elastic["k_initial"] - 1.0)
    stats = {
        "elastic_limit": elastic,
        "fully_dissipative": dissipative,
        "oblique_frictional": oblique,
        "elastic_energy_err": elastic_conservation,
        "dissipative_residual": dissipative["k_final_rel"]
        + dissipative["u_n_final"] / dissipative["k_initial"],
        "dissipative_split": {
            "dissipated": dissipative["dissipated_rel"],
            "released": dissipative["u_n_release"] / dissipative["k_initial"],
            "residual_kinetic": dissipative["k_final_rel"],
        },
        "oblique_restitution_err": abs(
            oblique["restitution_measured"] - oblique["restitution"]
        ) / oblique["restitution"],
        "worst_closure_rel": max(
            elastic["closure_rel"],
            dissipative["closure_rel"],
            oblique["closure_rel"],
        ),
    }
    stats.update(
        {
            "criterion": 1.0e-4,
            "passed": (
                elastic_conservation <= 1.0e-4
                and stats["dissipative_residual"] <= 1.0e-4
                and stats["worst_closure_rel"] <= 1.0e-4
                and stats["oblique_restitution_err"] <= 1.0e-1
                and elastic["all_dissipative_nonnegative"]
                and dissipative["all_dissipative_nonnegative"]
                and oblique["all_dissipative_nonnegative"]
            ),
        }
    )
    return stats
