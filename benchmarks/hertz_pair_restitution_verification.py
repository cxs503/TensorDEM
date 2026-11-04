"""Hertz pair-contact restitution: is the declared ``restitution_coefficient`` the real one?

Verification-debt closure for the ``contact_model="hertz"`` branch of
:class:`tensordem.dem3d.IceDEM3D`.

Background
----------
With ``contact_model="hertz"`` and the default ``contact_damping_model="restitution"``
the pair law integrated by the engine is

    F_n = clamp( k_h * d**(3/2) - gamma(d) * v_n , 0, inf )
    k_h     = (4/3) E* sqrt(R*)
    k_s(d)  = k_h * sqrt(d)                          secant stiffness
    gamma(d)= 2 zeta sqrt(k_s(d) * m_eff)            = 2 zeta sqrt(k_h m_eff) d**(1/4)

``zeta`` is *not* a user input: it comes from the requested restitution through
the **linear Kelvin-Voigt** map ``zeta = |ln e| / sqrt(pi**2 + ln(e)**2)``.
Because ``gamma`` carries a ``d**(1/4)`` factor, the *local* damping ratio
``gamma / (2 sqrt(k_s m_eff))`` equals ``zeta`` at every overlap - the
amplitude-independent construction used by Hunt-Crossley-type models.  Whether
the resulting impact restitution equals the requested ``e`` had no test and no
document; the merged Hertz document (docs/dem3d_hertz_contact_verification.md,
line 174) still states that Hertz damping "remains the legacy absolute c_n"
and that restitution is "not yet controllable".

What this script does
---------------------
1. **Independent reference** (no engine code in the loop): the exact SDOF ODE
   above is integrated with classical RK4 in SI units, with step refinement and
   Richardson extrapolation; a first-order (small-zeta) energy-loss quadrature
   supplies a second, algebraically independent anchor.
2. **Engine measurement**: 2x2x2 :class:`IceDEM3D`, bonds removed, two armed
   particles, head-on impact (harness convention of
   ``scripts/verify_dem3d_restitution.py``).
3. **Comparisons**: engine vs reference (implementation closure), reference vs
   the requested ``e`` (calibration bias of the linear map applied to a Hertz
   law), speed independence, time-step convergence, and the platen branch
   (absolute ``platen_damping``, restitution ignored).

Run:
    PYTHONPATH=src python3 benchmarks/hertz_pair_restitution_verification.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src", ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tensordem.dem3d import (  # noqa: E402
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)
import verify_dem3d_restitution as V  # noqa: E402

OUT = ROOT / "artifacts" / "hertz_pair_restitution"

RADIUS = V.RADIUS                      # 0.025 m
DENSITY = V.DENSITY                    # 917 kg/m3 (ice)
YOUNG = 1.0e9                          # Pa
POISSON = 0.3
IMPACT_SPEED = 0.5                     # m/s per particle (harness default)
V_REL = 2.0 * IMPACT_SPEED             # relative approach speed of the pair
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
M_EFF = MASS / 2.0                     # engine uses cfg.mass/2 for pairs
E_PAIR = hertz_reduced_modulus(YOUNG, POISSON)
R_PAIR = hertz_effective_radius(RADIUS)
K_H = (4.0 / 3.0) * E_PAIR * math.sqrt(R_PAIR)


def zeta_from_restitution(e: float) -> float:
    """The linear Kelvin-Voigt map the engine uses to seed zeta from e."""
    if e >= 1.0:
        return 0.0
    ln_e = -math.log(e)
    return ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


def restitution_from_zeta(zeta: float) -> float:
    """Inverse of :func:`zeta_from_restitution` (linear Kelvin-Voigt)."""
    if zeta <= 0.0:
        return 1.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


# ---------------------------------------------------------------------------
# 1. Independent reference: RK4 on the exact single-contact ODE
# ---------------------------------------------------------------------------

def _accel(delta: float, u: float, a: float, b: float) -> tuple[float, float]:
    """Return (delta_dot, u_dot); ``u > 0`` means the pair is approaching.

    Engine law is ``F = k_s d - gamma v_n`` with ``v_n = -u``, i.e.
    ``F = k_s d + gamma u``, clipped at zero (a damper may not pull).
    """
    if delta <= 0.0:
        return u, 0.0
    k_s = a * math.sqrt(delta)                     # a = k_h
    gamma = b * delta ** 0.25                      # b = 2 zeta sqrt(k_h m_eff)
    force = k_s * delta + gamma * u
    if force <= 0.0:                               # unilateral clamp: no tension
        return u, 0.0
    return u, -force / M_EFF


def reference_restitution(zeta: float, v_in: float, steps: int = 20000,
                          *, max_steps: int = 8_000_000) -> dict:
    """RK4 integration of the exact engine law; returns the restitution ratio."""
    a = K_H
    b = 2.0 * zeta * math.sqrt(K_H * M_EFF)
    t_scale = 2.94 * (5.0 * M_EFF / (4.0 * K_H)) ** 0.4 * v_in ** (-0.2)
    h = t_scale / steps
    delta, u = 0.0, v_in
    work = 0.0        # integral of the total contact force over overlap
    elastic = 0.0     # integral of the elastic part only
    peak = 0.0
    n = 0
    while n < max_steps:
        k1d, k1u = _accel(delta, u, a, b)
        k2d, k2u = _accel(delta + 0.5 * h * k1d, u + 0.5 * h * k1u, a, b)
        k3d, k3u = _accel(delta + 0.5 * h * k2d, u + 0.5 * h * k2u, a, b)
        k4d, k4u = _accel(delta + h * k3d, u + h * k3u, a, b)
        u_mid = u + h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        d_mid = delta + h * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0
        if d_mid > 0.0:
            k_s = a * math.sqrt(d_mid)
            force = k_s * d_mid + b * d_mid ** 0.25 * u_mid
            if force > 0.0:
                work += force * (d_mid - delta)
                elastic += k_s * d_mid * (d_mid - delta)
        delta, u = d_mid, u_mid
        peak = max(peak, delta)
        n += 1
        if u < 0.0:
            force_now = (a * math.sqrt(delta) * delta + b * delta ** 0.25 * u
                         if delta > 0.0 else 0.0)
            if delta <= 0.0 or force_now <= 0.0:
                break
    ke_in = 0.5 * M_EFF * v_in ** 2
    ke_out = 0.5 * M_EFF * u ** 2
    damper = work - elastic
    return {
        "restitution": -u / v_in,
        "steps": n,
        "dt": h,
        "damper_work_J": damper,
        "ke_deficit_J": ke_in - ke_out,
        "energy_balance_rel": (work - (ke_in - ke_out)) / ke_in,
        "max_overlap_m": peak,
    }


def reference_with_refinement(zeta: float, v_in: float, levels: int = 3) -> dict:
    """Richardson-refined reference value of the restitution."""
    runs = [reference_restitution(zeta, v_in, steps=4000 * 2 ** i) for i in range(levels)]
    r1, r2 = runs[-2]["restitution"], runs[-1]["restitution"]
    r0 = runs[0]["restitution"]
    d1, d2 = abs(r0 - r1), abs(r1 - r2)
    p = math.log2(d1 / d2) if d2 > 0.0 else 4.0
    p = p if math.isfinite(p) and p > 0.0 else 4.0
    return {
        "restitution": r2 + (r2 - r1) / (2.0 ** p - 1.0),
        "restitution_raw": r2,
        "richardson_order": p,
        "levels": [r["restitution"] for r in runs],
        "dt": runs[-1]["dt"],
        "energy_balance_rel": runs[-1]["energy_balance_rel"],
        "damper_work_J": runs[-1]["damper_work_J"],
        "ke_deficit_J": runs[-1]["ke_deficit_J"],
    }


# ---------------------------------------------------------------------------
# 2. Second independent anchor: first-order energy-loss quadrature
# ---------------------------------------------------------------------------

def _gauss_legendre(n: int = 40) -> tuple[list[float], list[float]]:
    """Nodes and weights on [-1, 1] (Newton iteration on Legendre polynomials)."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        dp = 1.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


def first_order_restitution(zeta: float, v_in: float) -> dict:
    """Small-zeta perturbation: ``1 - e**2 = D / E0`` with the Hertz energy scale.

    ``E0 = (2/5) k_h dmax**(5/2)``, ``gamma(d) = 2 zeta sqrt(k_h m) d**(1/4)``,
    ``u(d) = sqrt(2 (E0 - (2/5) k_h d**(5/2)) / m)`` and the loss is twice the
    compression-phase integral (symmetric restitution phase to first order).
    """
    nodes, weights = _gauss_legendre(40)
    integral = 0.0
    for x, w in zip(nodes, weights):
        s = 0.5 * (x + 1.0)
        integral += 0.5 * w * s ** 0.25 * math.sqrt(max(0.0, 1.0 - s ** 2.5))
    delta_max = (5.0 * M_EFF * v_in ** 2 / (4.0 * K_H)) ** 0.4
    e0 = 0.4 * K_H * delta_max ** 2.5
    loss = (4.0 * zeta * math.sqrt(K_H * M_EFF) * delta_max ** 1.25
            * math.sqrt(2.0 * e0 / M_EFF) * integral)
    one_minus_e2 = loss / e0
    return {"restitution": math.sqrt(max(0.0, 1.0 - one_minus_e2)),
            "delta_max_m": delta_max, "elastic_energy_J": e0,
            "loss_J": loss}


def calibration_inverse(e_target: float, v_in: float = IMPACT_SPEED,
                        tol: float = 1e-7) -> dict:
    """Bisect zeta so that the reference restitution reproduces ``e_target``."""
    lo, hi = 0.0, 0.95
    steps = 2000
    f_lo = reference_restitution(lo, v_in, steps)["restitution"] - e_target
    f_hi = reference_restitution(hi, v_in, steps)["restitution"] - e_target
    if f_lo * f_hi > 0.0:
        return {"zeta": float("nan"), "bracket_failed": True}
    mid = 0.5 * (lo + hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = reference_restitution(mid, v_in, steps)["restitution"] - e_target
        if abs(f_mid) < tol or (hi - lo) < 1e-14:
            return {"zeta": mid, "residual": f_mid}
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return {"zeta": mid}


# ---------------------------------------------------------------------------
# 3. Engine measurement (Hertz branch)
# ---------------------------------------------------------------------------

def _build_engine(e_target: float, *, platen: bool = False,
                  dt_scale: float = 1.0) -> IceDEM3D:
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=V.STIFFNESS, drag=0.0,
        contact_model="hertz", young_modulus=YOUNG, poisson_ratio=POISSON,
        restitution_coefficient=e_target,
        contact_damping_model="restitution",
        platen_stiffness=V.STIFFNESS, platen_damping=0.0,
        fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=platen, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    )
    if dt_scale != 1.0:
        kwargs["dt"] = DEM3DConfig(**kwargs).recommended_dt * dt_scale
    sim = IceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    return sim


def measure_pair(e_target: float, *, impact_speed: float = IMPACT_SPEED,
                 dt_scale: float = 1.0) -> dict:
    """Head-on sphere/sphere impact inside IceDEM3D with the Hertz pair law."""
    sim = _build_engine(e_target, dt_scale=dt_scale)
    V._park_idle(sim)
    with torch.no_grad():
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([impact_speed, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-impact_speed, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * RADIUS - float(
            torch.linalg.norm(sim.positions[1] - sim.positions[0]))

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    result = V._measure_impact(sim, overlap, relative_velocity)
    result["dt"] = sim.dt
    return result


def measure_platen(e_target: float, *, impact_speed: float = IMPACT_SPEED) -> dict:
    """Hertz platen branch: absolute ``platen_damping``, restitution ignored."""
    sim = _build_engine(e_target, platen=True)
    V._park_idle(sim, start=1)
    platen_z = float(sim.bottom_platen_start_z)
    with torch.no_grad():
        sim.positions[0] = torch.tensor([0.0, 0.0, RADIUS + platen_z + 0.0005],
                                        dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([0.0, 0.0, -impact_speed], dtype=sim.dtype)

    def overlap() -> float:
        return platen_z + RADIUS - float(sim.positions[0, 2])

    def relative_velocity() -> float:
        return float(sim.velocities[0, 2])

    return V._measure_impact(sim, overlap, relative_velocity)


# ---------------------------------------------------------------------------
# 4. Campaign
# ---------------------------------------------------------------------------

def build_audit() -> dict:
    targets = [0.9, 0.7, 0.5, 0.3]
    rows = []
    for e_target in targets:
        zeta = zeta_from_restitution(e_target)
        ref = reference_with_refinement(zeta, V_REL)
        fo = first_order_restitution(zeta, V_REL)
        coarse = measure_pair(e_target, dt_scale=1.0)
        fine = measure_pair(e_target, dt_scale=0.25)
        finest = measure_pair(e_target, dt_scale=0.0625)
        inv = calibration_inverse(e_target)
        rows.append({
            "e_target": e_target,
            "zeta_used": zeta,
            "e_linear_closed_form": e_target,
            "e_reference_ode": ref["restitution"],
            "e_reference_first_order": fo["restitution"],
            "richardson_order": ref["richardson_order"],
            "reference_levels": ref["levels"],
            "reference_energy_balance_rel": ref["energy_balance_rel"],
            "e_engine_dt_scale_1": coarse["restitution"],
            "e_engine_dt_scale_0.25": fine["restitution"],
            "e_engine_dt_scale_0.0625": finest["restitution"],
            "peak_overlap_m": finest["peak_overlap_m"],
            "steps_in_contact": finest["steps_in_contact"],
            "engine_dt_s": finest["dt"],
            "bias_vs_target_pct": 100.0 * (finest["restitution"] - e_target) / e_target,
            "engine_vs_reference_pct": (100.0 * (finest["restitution"] - ref["restitution"])
                                        / ref["restitution"]),
            "reference_vs_target_pct": (100.0 * (ref["restitution"] - e_target) / e_target),
            "zeta_required": inv["zeta"],
            "zeta_correction_ratio": (zeta / inv["zeta"]) if inv.get("zeta") else float("nan"),
        })

    calibration = []
    for e_desired in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        inv = calibration_inverse(e_desired)
        declared = restitution_from_zeta(inv["zeta"])
        calibration.append({
            "e_desired": e_desired,
            "zeta_required": inv["zeta"],
            "declared_restitution_coefficient": declared,
            "declared_over_desired": declared / e_desired,
        })

    speed_rows = []
    for v in (0.1, 0.25, 0.5, 1.0):
        zeta = zeta_from_restitution(0.5)
        ref = reference_with_refinement(zeta, 2.0 * v)
        engine = measure_pair(0.5, impact_speed=v, dt_scale=0.25)
        speed_rows.append({
            "impact_speed": v,
            "e_reference_ode": ref["restitution"],
            "e_engine": engine["restitution"],
            "peak_overlap_m": engine["peak_overlap_m"],
        })

    platen_rows = []
    for e_target in (0.9, 0.3):
        res = measure_platen(e_target)
        platen_rows.append({"e_target": e_target, "e_engine_platen": res["restitution"]})

    return {
        "schema": "tensordem.hertz-pair-restitution-audit/1",
        "parameters": {
            "radius_m": RADIUS, "density_kg_m3": DENSITY, "young_modulus_Pa": YOUNG,
            "poisson_ratio": POISSON, "mass_kg": MASS, "m_eff_kg": M_EFF,
            "e_star_pair_Pa": E_PAIR, "r_star_pair_m": R_PAIR, "k_h_SI": K_H,
            "impact_speed_m_s": IMPACT_SPEED,
        },
        "pair_rows": rows,
        "speed_rows": speed_rows,
        "platen_rows": platen_rows,
        "calibration": calibration,
        "limitations": [
            "Platen/tool branches keep the absolute dashpot (platen_damping / "
            "contact_damping) and ignore restitution_coefficient; measured here.",
            "Engine values use dt = recommended_dt * scale; the finest scale "
            "(1/16) is the reported engine restitution.",
            "First-order quadrature is only valid for small zeta and is reported "
            "as a cross-check, not as the primary anchor.",
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print("e_target  zeta     e_ref(ODE)  e_ref(1st)  e_eng(dt/16)  bias%    eng-ref%  zeta_req  ratio")
    for r in audit["pair_rows"]:
        print(f"{r['e_target']:8.3f}  {r['zeta_used']:.5f}  "
              f"{r['e_reference_ode']:.6f}   {r['e_reference_first_order']:.6f}    "
              f"{r['e_engine_dt_scale_0.0625']:.6f}   "
              f"{r['bias_vs_target_pct']:+7.2f}  {r['engine_vs_reference_pct']:+7.3f}  "
              f"{r['zeta_required']:.5f}  {r['zeta_correction_ratio']:.4f}")
    print("\nspeed independence at e_target=0.5:")
    for r in audit["speed_rows"]:
        print(f"  v={r['impact_speed']:.2f}  e_ref={r['e_reference_ode']:.6f}"
              f"  e_eng={r['e_engine']:.6f}")
    print("\nplaten branch (restitution_coefficient ignored):")
    for r in audit["platen_rows"]:
        print(f"  e_target={r['e_target']:.1f}  e_platen={r['e_engine_platen']:.6f}")
    print("\ncalibration table:")
    for r in audit["calibration"]:
        print(f"  e_desired={r['e_desired']:.2f}  zeta_req={r['zeta_required']:.5f}"
              f"  declare e={r['declared_restitution_coefficient']:.5f}")
    try:
        import hertz_pair_restitution_figures as figs
        print("\nfigures:", ", ".join(figs.write_figures(audit, OUT)))
    except Exception as exc:  # figures are best-effort evidence
        print("\nfigure generation failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())