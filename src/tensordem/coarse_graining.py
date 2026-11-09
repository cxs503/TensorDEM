"""DEM coarse-graining scaling laws and physical-deviation ledger.

Physical background
-------------------
Coarse-graining replaces N particles of radius R with N/k³ particles of
radius kR while preserving macroscopic behaviour.  The scaling laws are
derived from dimensional analysis and contact-mechanics invariance:

Geometric scaling
    R → k·R,  all lengths → k·lengths

Mass scaling (constant density ρ)
    m = ρ·(4π/3)·R³  →  k³·m
    I = 0.4·m·R²     →  k⁵·I

Contact-time invariance (primary constraint)
    t_c = π·√(m_eff / k_n)  must be invariant
    → m_eff / k_n = invariant
    → k_n → k³·k_n

Damping invariance (constant restitution e)
    ζ = γ / (2·√(k_n·m_eff))  must be invariant
    → γ → k³·γ

Tangential stiffness
    k_t → k³·k_t  (same ratio k_t/k_n preserved)

Rolling stiffness
    k_r = k_t·R²  →  k³·k_t · k²·R²  =  k⁵·k_r

Dimensionless invariants
    e (restitution)  →  e  (unchanged)
    μ (Coulomb friction)  →  μ  (unchanged)
    μ_r (rolling friction)  →  μ_r  (unchanged)
    g (gravity)  →  g  (unchanged)

Timestep scaling
    Since t_c is invariant, the same dt resolves the contact equally well.
    However, the macroscopic flow time scale t_flow ∝ √(L/g) → √k·t_flow,
    so the *number* of steps to reach a given macroscopic state drops by √k.
    In practice we keep dt unchanged for safety, gaining speed purely from
    fewer particles (N/k³).

Negative control
    If only mass is scaled (m → k³m) but stiffness is NOT rescaled
    (k_n unchanged), then t_c → √(k³) = k^(3/2)·t_c, i.e. the contact
    time deviates by >30 % for k≥2, breaking dynamic similarity.

All quantities are SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional


# ---------------------------------------------------------------------------
# Scaling-law engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoarseGrainResult:
    """Result of applying a k-fold coarse-grain to a material."""
    k: float
    radius: float
    mass: float
    inertia: float
    kn: float
    kt: float
    kr: float
    gamma_pair: float
    gamma_wall: float
    restitution: float
    mu: float
    mu_rolling: float
    gravity: float
    contact_time: float
    timestep: float


def scale_contact_time_invariant(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    k: float,
) -> CoarseGrainResult:
    """Apply k-fold coarse-graining with contact-time invariance.

    Scaling summary:
        R  → k·R
        m  → k³·m
        k_n → k³·k_n
        k_t → k³·k_t
        k_r → k⁵·k_r
        γ  → k³·γ
        e, μ, μ_r, g  →  unchanged
    """
    if k <= 0:
        raise ValueError("k must be positive")

    r_new = k * radius
    mass_new = k ** 3 * density * 4.0 / 3.0 * math.pi * radius ** 3
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = k ** 3 * kn
    kt_new = k ** 3 * kt_ratio * kn
    kr_new = k ** 5 * kt_ratio * kn * radius ** 2

    # Damping: γ = 2·ζ·√(k_n·m_eff), m_eff = m/2 for equal-mass pair
    ln_e = math.log(max(min(restitution, 1.0 - 1e-12), 1e-6))
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    mass_eff_orig = 0.5 * density * 4.0 / 3.0 * math.pi * radius ** 3
    gamma_orig = 2.0 * zeta * math.sqrt(kn * mass_eff_orig)
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    # Contact time: t_c = π·√(m_eff/k_n) — should be invariant
    tc_orig = math.pi * math.sqrt(mass_eff_orig / kn)
    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)

    # Timestep: 20 steps per contact period
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        k=k,
        radius=r_new,
        mass=mass_new,
        inertia=inertia_new,
        kn=kn_new,
        kt=kt_new,
        kr=kr_new,
        gamma_pair=gamma_new,
        gamma_wall=gamma_new * math.sqrt(2.0),
        restitution=restitution,
        mu=mu,
        mu_rolling=mu_rolling,
        gravity=gravity,
        contact_time=tc_new,
        timestep=dt_new,
    )


def scale_mass_only(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    k: float,
) -> CoarseGrainResult:
    """Negative control: scale mass only, keep stiffness unchanged.

    This breaks contact-time invariance: t_c → k^(3/2)·t_c.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    r_new = k * radius
    mass_new = k ** 3 * density * 4.0 / 3.0 * math.pi * radius ** 3
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = kn
    kt_new = kt_ratio * kn
    kr_new = kt_ratio * kn * r_new ** 2

    ln_e = math.log(max(min(restitution, 1.0 - 1e-12), 1e-6))
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    mass_eff_orig = 0.5 * density * 4.0 / 3.0 * math.pi * radius ** 3
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    tc_orig = math.pi * math.sqrt(mass_eff_orig / kn)
    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        k=k,
        radius=r_new,
        mass=mass_new,
        inertia=inertia_new,
        kn=kn_new,
        kt=kt_new,
        kr=kr_new,
        gamma_pair=gamma_new,
        gamma_wall=gamma_new * math.sqrt(2.0),
        restitution=restitution,
        mu=mu,
        mu_rolling=mu_rolling,
        gravity=gravity,
        contact_time=tc_new,
        timestep=dt_new,
    )


# ---------------------------------------------------------------------------
# Contact-time and restitution verification
# ---------------------------------------------------------------------------

def contact_time_analytic(kn: float, mass_eff: float) -> float:
    """Undamped contact period: t_c = π·√(m_eff/k_n)."""
    return math.pi * math.sqrt(mass_eff / kn)


def restitution_from_damping(
    gamma: float, kn: float, mass_eff: float
) -> float:
    """Compute realised restitution from damping coefficient."""
    zeta = gamma / (2.0 * math.sqrt(kn * mass_eff))
    if zeta >= 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))


def verify_contact_scaling(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    k_values: list[float],
) -> dict:
    """Verify contact-level scaling for a list of k values."""
    mass_orig = density * 4.0 / 3.0 * math.pi * radius ** 3
    mass_eff_orig = 0.5 * mass_orig
    tc_orig = contact_time_analytic(kn, mass_eff_orig)

    ln_e = math.log(max(min(restitution, 1.0 - 1e-12), 1e-6))
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    gamma_orig = 2.0 * zeta * math.sqrt(kn * mass_eff_orig)
    e_check = restitution_from_damping(gamma_orig, kn, mass_eff_orig)

    rows = []
    for k in k_values:
        cg = scale_contact_time_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, gravity, k
        )
        mass_eff_new = 0.5 * cg.mass
        tc_new = contact_time_analytic(cg.kn, mass_eff_new)
        e_new = restitution_from_damping(cg.gamma_pair, cg.kn, mass_eff_new)

        tc_dev = abs(tc_new - tc_orig) / tc_orig * 100.0
        e_dev = abs(e_new - restitution)

        rows.append({
            "k": k,
            "radius": cg.radius,
            "mass": cg.mass,
            "kn": cg.kn,
            "gamma": cg.gamma_pair,
            "contact_time": tc_new,
            "contact_time_dev_pct": tc_dev,
            "restitution": e_new,
            "restitution_dev": e_dev,
        })

    neg_rows = []
    for k in k_values:
        nc = scale_mass_only(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, gravity, k
        )
        mass_eff_new = 0.5 * nc.mass
        tc_new = contact_time_analytic(nc.kn, mass_eff_new)
        tc_dev = abs(tc_new - tc_orig) / tc_orig * 100.0
        neg_rows.append({
            "k": k,
            "contact_time": tc_new,
            "contact_time_dev_pct": tc_dev,
        })

    return {
        "base": {
            "radius": radius,
            "mass": mass_orig,
            "kn": kn,
            "contact_time": tc_orig,
            "restitution_target": restitution,
            "restitution_check": e_check,
        },
        "scaled": rows,
        "negative_control": neg_rows,
    }


# ---------------------------------------------------------------------------
# Macro-anchor simulation wrappers (small-scale for test budget)
# ---------------------------------------------------------------------------

def make_repose_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.4,
    k: float = 1.0,
) -> dict:
    """Build a ReposeCase with coarse-grained parameters."""
    cg = scale_contact_time_invariant(
        radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, k
    )
    return {
        "mu": cg.mu,
        "mu_rolling": cg.mu_rolling,
        "radius": cg.radius,
        "density": density,
        "kn": cg.kn,
        "kt_ratio": kt_ratio,
        "restitution": cg.restitution,
        "gravity": cg.gravity,
        "cylinder_radius": 0.05 * k,
        "fill_height": 0.08 * k,
        "packing_gap": 0.02,
        "jitter": 0.02,
        "hold_time": 0.05,
        "lift_time": 0.15,
        "settle_time": 0.30,
        "dt": cg.timestep,
        "seed": 4970,
    }


def make_beverloo_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.3,
    k: float = 1.0,
) -> dict:
    """Build a BeverlooCase with coarse-grained parameters.

    Orifice is set to 4 particle diameters (8 * radius) to satisfy
    the solver's minimum of 3 diameters.
    """
    cg = scale_contact_time_invariant(
        radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, k
    )
    silo_radius = 0.04 * k
    fill_height = 0.12 * k
    orifice_diameter = 8.0 * cg.radius  # 4 particle diameters
    return {
        "radius": cg.radius,
        "density": density,
        "mu": cg.mu,
        "mu_rolling": cg.mu_rolling,
        "restitution": cg.restitution,
        "kn": cg.kn,
        "kt_ratio": kt_ratio,
        "gravity": cg.gravity,
        "silo_radius": silo_radius,
        "fill_height": fill_height,
        "orifice_diameter": orifice_diameter,
        "packing_gap": 0.02,
        "jitter": 0.01,
        "dt": cg.timestep,
        "seed": 5278,
        "max_time": 2.0 * math.sqrt(k),
        "settle_time": 0.15 * math.sqrt(k),
        "window": 0.2 * math.sqrt(k),
        "n_windows": 5,
        "wall_mu": 0.0,
    }


def make_shear_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.3,
    k: float = 1.0,
) -> dict:
    """Build parameters for a direct-shear cell with coarse-grained particles."""
    cg = scale_contact_time_invariant(
        radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, k
    )
    box_width = 20.0 * cg.radius
    box_depth = 10.0 * cg.radius
    fill_height = 20.0 * cg.radius
    return {
        "radius": cg.radius,
        "density": density,
        "mu": cg.mu,
        "mu_rolling": cg.mu_rolling,
        "restitution": cg.restitution,
        "kn": cg.kn,
        "kt_ratio": kt_ratio,
        "gravity": cg.gravity,
        "box_width": box_width,
        "box_depth": box_depth,
        "fill_height": fill_height,
        "packing_gap": 0.02,
        "jitter": 0.01,
        "dt": cg.timestep,
        "seed": 6107,
    }
