"""Reproducible verification of the Hertz contact contract in `tensordem.dem3d`.

Supersedes the un-referenced numbers in ``docs/HERTZ_VERIFY.md`` (that document
quoted six result numbers with no script, no artifact and no test behind any of
them, so the doc-evidence pointer audit could not see it: that audit only
resolves pointers that exist).

Three questions, each answered from the shipped code path:

Q1  static force-indentation law   F(delta) = 4/3 E* sqrt(R*) delta^(3/2)
    measured through ``IceDEM3D.forces()`` on a two-sphere pair, against the
    closed form with the reduced constants the module itself defines.
Q2  sphere-rigid constants         E* = E / (1 - nu^2), R* = R (plane limit).
Q3  restitution mapping            the shipped Hertz pair law is
    ``F = max(k_s(delta) delta - 2 zeta sqrt(k_s(delta) m_eff) v, 0)`` with
    ``zeta`` from the *linear* mapping; because ``k_s`` follows the overlap,
    this is not the linear dashpot the mapping was derived for, so the realised
    e is not the requested e.  Quantified by RK4 integration of the shipped law
    (velocity-free of solver step-size error) and cross-checked on the solver.

Run: ``PYTHONPATH=src python scripts/verify_hertz_contract.py``
Outputs ``artifacts/hertz_contract/verification.json``,
``docs/figures/hertz_contract.png``.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import dem3d as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "hertz_contract"
FIG = ROOT / "docs" / "figures"

E_MOD = 70e9
NU = 0.23
RADIUS = 0.01
DENSITY = 2500.0
GRAVITY = 0.0


def _mass(radius=RADIUS, density=DENSITY):
    return density * 4.0 / 3.0 * math.pi * radius ** 3


def _sim(radius=RADIUS, density=DENSITY, **over):
    cfg = D.DEM3DConfig(
        nx=2, ny=2, nz=2, radius=radius, density=density,
        contact_model="hertz", young_modulus=E_MOD, poisson_ratio=NU,
        contact_damping=0.0, drag=0.0,
        fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        **{"restitution_coefficient": 1.0, **over},
    )
    sim = D.IceDEM3D(cfg)
    sim.bonded[:] = False          # no bond carries the load
    sim.alive[:] = False
    sim.fixed[:] = True            # start from a fully held assembly
    return cfg, sim


def q1_static(fracs=(0.01, 0.05, 0.1, 0.2)):
    """Force-indentation law through the solver's own pair force."""
    e_star = D.hertz_reduced_modulus(E_MOD, NU)
    r_star = D.hertz_effective_radius(RADIUS)
    rows = []
    for frac in fracs:
        delta = frac * RADIUS
        _, sim = _sim()
        # particle 0 stays pinned (the "rigid" support); particle 1 is released
        # only so that its force is not masked, and is never stepped.
        sim.fixed[0] = False
        # park every other lattice particle 1 m away: a 2x2x2 default lattice
        # overlaps, and a third body would contaminate the measured pair force
        for k in range(2, len(sim.positions)):
            sim.positions[k] = torch.tensor([10.0 + float(k), 10.0, 10.0], dtype=torch.float64)
            sim.initial_positions[k] = sim.positions[k].clone()
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        sim.positions[1] = torch.tensor([2 * RADIUS - delta, 0.0, 0.0], dtype=torch.float64)
        sim.initial_positions[0] = sim.positions[0].clone()
        sim.initial_positions[1] = sim.positions[1].clone()
        f = sim.forces()
        meas = float(torch.linalg.vector_norm(f[0]))
        ref = float(D.hertz_normal_force(delta, e_star, r_star))
        rows.append({
            "delta_over_R": frac,
            "delta_m": delta,
            "force_measured_N": meas,
            "force_hertz_N": ref,
            "relative_error": (meas - ref) / ref,
        })
    return {"e_star_Pa": e_star, "r_star_m": r_star, "rows": rows}


def q2_rigid_constants():
    e_star = D.hertz_reduced_modulus(E_MOD, NU, rigid_counterpart=True)
    r_star = D.hertz_effective_radius(RADIUS, rigid_counterpart=True)
    e_closed = E_MOD / (1.0 - NU ** 2)
    return {
        "e_star_rigid_Pa": e_star,
        "e_star_closed_form_Pa": e_closed,
        "e_star_relative_error": (e_star - e_closed) / e_closed,
        "r_star_rigid_m": r_star,
        "r_star_closed_form_m": RADIUS,
        "r_star_relative_error": (r_star - RADIUS) / RADIUS,
        "e_star_sphere_sphere_ratio": D.hertz_reduced_modulus(E_MOD, NU) / e_closed,
    }


def _zeta(e_target):
    ln_e = math.log(e_target)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _hertz_law_rhs(delta, v_rel, zeta, e_star, r_star, m_eff):
    """Shipped pair law, exactly as `dem3d.forces` assembles the Hertz branch."""
    if delta <= 0.0:
        return 0.0
    ks = float(D.hertz_secant_stiffness(delta, e_star, r_star))
    gamma = 2.0 * zeta * math.sqrt(ks * m_eff)
    # `dem3d.forces` forms contact_mag = k delta - gamma * vn with
    # vn = (v_j - v_i).n = SEPARATION rate; the integrator state v_rel here is
    # the penetration rate, so vn = -v_rel.  Using vn = +v_rel instead makes the
    # dashpot feed energy in during restitution (e > 1), which is a property of
    # that wrong sign and not of the shipped law.
    f = max(ks * delta + gamma * v_rel, 0.0)    # unilateral clamp
    return f / m_eff


def q3_restitution(e_targets=(0.9, 0.7, 0.5), v0s=(0.05, 0.5), both_free=True):
    """RK4 integration of the shipped law: realised e vs requested e."""
    e_star = D.hertz_reduced_modulus(E_MOD, NU)
    r_star = D.hertz_effective_radius(RADIUS)
    m = _mass()
    m_eff = 0.5 * m if both_free else m
    k_ref = float(D.hertz_secant_stiffness(0.1 * RADIUS, e_star, r_star))
    dt = 2.0 * math.pi * math.sqrt(m_eff / k_ref) / 400.0
    rows = []
    for e_t in e_targets:
        zeta = _zeta(e_t)
        for v0 in v0s:
            x, v = -1e-9, float(v0)
            v_in = None
            e_meas = float("nan")
            steps = 0
            while steps < 200_000:
                def acc(xx, vv):
                    return -_hertz_law_rhs(xx, vv, zeta, e_star, r_star, m_eff)
                k1x, k1v = v, acc(x, v)
                k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
                k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
                k4x, k4v = v + dt * k3v, acc(x + dt * k3x, v + dt * k3v)
                x_n = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
                v_n = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
                if v_in is None and x_n > 0.0 and x <= 0.0:
                    v_in = abs(v_n)
                if v_in is not None and x_n < 0.0 and v_n < 0.0:
                    e_meas = abs(v_n) / v_in
                    break
                x, v = x_n, v_n
                steps += 1
            rows.append({
                "e_target": e_t, "zeta_linear_mapping": zeta, "v_in_m_s": v0,
                "e_measured": e_meas,
                "relative_error_percent": 100.0 * (e_meas - e_t) / e_t,
                "rk4_dt_s": dt, "rk4_steps": steps, "m_eff_kg": m_eff,
                "max_overlap_over_R": None,
            })
    return {"both_free": both_free, "rows": rows}


def q3_solver(e_target=0.9, v0=0.5, both_free=True, max_steps=20_000):
    """Same quantity measured on the real solver (explicit stepping)."""
    cfg, sim = _sim(restitution_coefficient=e_target)
    left, right = 0, 1
    sim.fixed[left] = False
    if both_free:
        sim.fixed[right] = False
    gap = 1e-6
    sim.positions[left, 0] -= gap
    sim.initial_positions[left, 0] -= gap
    sim.velocities[left, 0] = v0
    if both_free:
        sim.positions[right, 0] += gap
        sim.initial_positions[right, 0] += gap
        sim.velocities[right, 0] = -v0
    sim.dt = cfg.recommended_dt
    cd = 2 * RADIUS
    in_contact = False
    v_in = 0.0
    max_ov = 0.0
    result = {"e_target": e_target, "v_in_m_s": v0, "both_free": both_free,
              "dt_s": sim.dt, "e_measured": float("nan"),
              "max_overlap_over_R": None, "steps": 0}
    for step in range(max_steps):
        dist = float(torch.linalg.vector_norm(sim.positions[left] - sim.positions[right]))
        rel = float(sim.velocities[left, 0] - sim.velocities[right, 0])
        if not in_contact and dist < cd:
            in_contact, v_in = True, rel
        elif in_contact and dist >= cd:
            result["e_measured"] = abs(rel / v_in)
            result["max_overlap_over_R"] = max_ov / RADIUS
            result["steps"] = step
            break
        if in_contact:
            max_ov = max(max_ov, cd - dist)
        sim.step()
    return result


def figure(static, rest):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return f"matplotlib unavailable: {exc}"
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    d = [r["delta_over_R"] for r in static["rows"]]
    axes[0].loglog(d, [r["force_hertz_N"] for r in static["rows"]], "k-", label="Hertz closed form")
    axes[0].loglog(d, [r["force_measured_N"] for r in static["rows"]], "ro", mfc="none",
                   label="IceDEM3D pair force")
    axes[0].set_xlabel(r"$\delta / R$")
    axes[0].set_ylabel("normal force $F$ [N]")
    axes[0].set_title("Q1 static Hertz law (slope 3/2)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)
    et = [r["e_target"] for r in rest["rows"]]
    em = [r["e_measured"] for r in rest["rows"]]
    axes[1].plot([0.4, 1.0], [0.4, 1.0], "k--", label="1:1 (requested = realised)")
    axes[1].plot(et, em, "bs", mfc="none", label="shipped Hertz law (RK4)")
    axes[1].set_xlabel("requested $e$")
    axes[1].set_ylabel("realised $e$")
    axes[1].set_title("Q3 restitution mapping bias")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = FIG / "hertz_contract.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    static = q1_static()
    rigid = q2_rigid_constants()
    rest = q3_restitution()
    solver = q3_solver()
    fig_path = figure(static, rest)
    payload = {
        "case": {"E_Pa": E_MOD, "nu": NU, "radius_m": RADIUS, "density_kg_m3": DENSITY,
                 "mass_kg": _mass()},
        "q1_static": static,
        "q2_rigid_constants": rigid,
        "q3_restitution_reference": rest,
        "q3_restitution_solver": solver,
        "figure": fig_path,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("q1_static", "q2_rigid_constants")}, indent=2)[:2500])
    print("\nQ3 reference:")
    for r in rest["rows"]:
        print(f"  e_t={r['e_target']:.2f} v0={r['v_in_m_s']:.2f}: e_meas={r['e_measured']:.6f} "
              f"err={r['relative_error_percent']:+.2f}% steps={r['rk4_steps']}")
    print("\nQ3 solver:", json.dumps(solver))
    print("\nwrote", OUT / "verification.json", "and", fig_path)


if __name__ == "__main__":
    main()