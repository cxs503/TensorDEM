"""DEM coarse-graining scaling laws and physical-deviation ledger.

Physical background
-------------------
Coarse-graining replaces N particles of radius R with N/f³ particles of
radius f·R while preserving macroscopic behaviour.  Two scaling口径 are
implemented and cross-validated:

口径 A — Contact-time invariance (k_n ∝ f³)
    Derived from t_c = π·√(m_eff/k_n) = invariant.
    Since m ∝ f³, we need k_n ∝ f³.
    This preserves the contact dynamics exactly.

口径 B — Sound-speed / timestep stability (k_n ∝ f)
    Derived from the wave speed c = √(k_n/m_eff) · L_char.
    For a granular assembly the macroscopic wave speed should be invariant:
    c ∝ √(k_n/m) · R  →  √(k_n/f³) · f = √(k_n/f)  →  k_n ∝ f.
    This preserves the macroscopic wave propagation speed but changes
    the contact time: t_c ∝ √(f³/f) = f.

Geometric scaling (both口径)
    R → f·R,  all lengths → f·lengths

Mass scaling (constant density ρ)
    m = ρ·(4π/3)·R³  →  f³·m
    I = 0.4·m·R²     →  f⁵·I

口径 A: k_n ∝ f³ (contact-time invariant)
    k_n → f³·k_n,  k_t → f³·k_t,  k_r → f⁵·k_r
    γ → f³·γ
    t_c → t_c (invariant)

口径 B: k_n ∝ f (sound-speed invariant)
    k_n → f·k_n,  k_t → f·k_t,  k_r → f³·k_r
    γ → f²·γ  (to keep ζ invariant: γ ∝ √(k_n·m) ∝ √(f·f³) = f²)
    t_c → f·t_c  (contact time grows linearly)

Dimensionless invariants (both口径)
    e (restitution)  →  e  (unchanged)
    μ (Coulomb friction)  →  μ  (unchanged)
    μ_r (rolling friction)  →  μ_r  (unchanged)
    g (gravity)  →  g  (unchanged)

Negative control
    If only mass is scaled (m → f³m) but stiffness is NOT rescaled
    (k_n unchanged), then t_c → √(f³) = f^(3/2)·t_c, i.e. the contact
    time deviates by >30 % for f≥2, breaking dynamic similarity.

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
    """Result of applying an f-fold coarse-grain to a material."""
    f: float
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
    scaling_approach: str = "contact_time_invariant"  # or "sound_speed_invariant"


def _damping_from_restitution(restitution: float) -> float:
    """Compute damping ratio ζ from restitution coefficient e."""
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _base_mass(radius: float, density: float) -> float:
    return density * 4.0 / 3.0 * math.pi * radius ** 3


# ---- 口径 A: Contact-time invariance (k_n ∝ f³) ----

def scale_contact_time_invariant(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    f: float,
) -> CoarseGrainResult:
    """Apply f-fold coarse-graining with contact-time invariance (口径 A).

    Scaling summary:
        R  → f·R
        m  → f³·m
        k_n → f³·k_n
        k_t → f³·k_t
        k_r → f⁵·k_r
        γ  → f³·γ
        e, μ, μ_r, g  →  unchanged
    """
    if f <= 0:
        raise ValueError("f must be positive")

    r_new = f * radius
    mass_new = f ** 3 * _base_mass(radius, density)
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = f ** 3 * kn
    kt_new = f ** 3 * kt_ratio * kn
    kr_new = f ** 5 * kt_ratio * kn * radius ** 2

    zeta = _damping_from_restitution(restitution)
    mass_eff_orig = 0.5 * _base_mass(radius, density)
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        f=f,
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
        scaling_approach="contact_time_invariant",
    )


# ---- 口径 B: Sound-speed invariance (k_n ∝ f) ----

def scale_sound_speed_invariant(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    f: float,
) -> CoarseGrainResult:
    """Apply f-fold coarse-graining with sound-speed invariance (口径 B).

    Scaling summary:
        R  → f·R
        m  → f³·m
        k_n → f·k_n       (wave speed c ∝ √(k_n/m)·R invariant)
        k_t → f·k_t
        k_r → f³·k_r      (k_r = k_t·R² → f·k_t · f²·R²)
        γ  → f²·γ         (ζ invariant: γ ∝ √(k_n·m) ∝ √(f·f³) = f²)
        e, μ, μ_r, g  →  unchanged
    """
    if f <= 0:
        raise ValueError("f must be positive")

    r_new = f * radius
    mass_new = f ** 3 * _base_mass(radius, density)
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = f * kn
    kt_new = f * kt_ratio * kn
    kr_new = f ** 3 * kt_ratio * kn * radius ** 2

    zeta = _damping_from_restitution(restitution)
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        f=f,
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
        scaling_approach="sound_speed_invariant",
    )


# ---- Negative control: mass-only scaling ----

def scale_mass_only(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    f: float,
) -> CoarseGrainResult:
    """Negative control: scale mass only, keep stiffness unchanged.

    This breaks contact-time invariance: t_c → f^(3/2)·t_c.
    """
    if f <= 0:
        raise ValueError("f must be positive")

    r_new = f * radius
    mass_new = f ** 3 * _base_mass(radius, density)
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = kn
    kt_new = kt_ratio * kn
    kr_new = kt_ratio * kn * r_new ** 2

    zeta = _damping_from_restitution(restitution)
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        f=f,
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
        scaling_approach="mass_only",
    )


# ---- Wrong scaling: k_n ∝ f³ used where k_n ∝ f is needed ----

def scale_wrong_kn_f3(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    f: float,
) -> CoarseGrainResult:
    """Negative control: apply k_n ∝ f³ scaling but with sound-speed intent.

    This is the 'wrong口径' test: using contact-time-invariant stiffness
    scaling when the macroscopic wave speed should be preserved.  The
    macroscopic quantities will deviate because the wave speed changes
    by √(f²) = f.
    """
    if f <= 0:
        raise ValueError("f must be positive")

    r_new = f * radius
    mass_new = f ** 3 * _base_mass(radius, density)
    inertia_new = 0.4 * mass_new * r_new ** 2
    kn_new = f ** 3 * kn  # wrong: should be f·kn for sound-speed
    kt_new = f ** 3 * kt_ratio * kn
    kr_new = f ** 5 * kt_ratio * kn * radius ** 2

    zeta = _damping_from_restitution(restitution)
    mass_eff_new = 0.5 * mass_new
    gamma_new = 2.0 * zeta * math.sqrt(kn_new * mass_eff_new)

    tc_new = math.pi * math.sqrt(mass_eff_new / kn_new)
    dt_new = tc_new / 20.0

    return CoarseGrainResult(
        f=f,
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
        scaling_approach="wrong_kn_f3",
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
    """Verify contact-level scaling for a list of f values (口径 A)."""
    mass_orig = _base_mass(radius, density)
    mass_eff_orig = 0.5 * mass_orig
    tc_orig = contact_time_analytic(kn, mass_eff_orig)

    zeta = _damping_from_restitution(restitution)
    gamma_orig = 2.0 * zeta * math.sqrt(kn * mass_eff_orig)
    e_check = restitution_from_damping(gamma_orig, kn, mass_eff_orig)

    rows = []
    for f in k_values:
        cg = scale_contact_time_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, gravity, f
        )
        mass_eff_new = 0.5 * cg.mass
        tc_new = contact_time_analytic(cg.kn, mass_eff_new)
        e_new = restitution_from_damping(cg.gamma_pair, cg.kn, mass_eff_new)

        tc_dev = abs(tc_new - tc_orig) / tc_orig * 100.0
        e_dev = abs(e_new - restitution)

        rows.append({
            "f": f,
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
    for f in k_values:
        nc = scale_mass_only(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, gravity, f
        )
        mass_eff_new = 0.5 * nc.mass
        tc_new = contact_time_analytic(nc.kn, mass_eff_new)
        tc_dev = abs(tc_new - tc_orig) / tc_orig * 100.0
        neg_rows.append({
            "f": f,
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


def verify_sound_speed_scaling(
    radius: float,
    density: float,
    kn: float,
    kt_ratio: float,
    restitution: float,
    mu: float,
    mu_rolling: float,
    gravity: float,
    f_values: list[float],
) -> dict:
    """Verify contact-level scaling for 口径 B (sound-speed invariance)."""
    mass_orig = _base_mass(radius, density)
    mass_eff_orig = 0.5 * mass_orig

    # Sound speed: c ∝ √(k_n/m_eff) · R
    c_orig = math.sqrt(kn / mass_eff_orig) * radius

    rows = []
    for f in f_values:
        cg = scale_sound_speed_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, gravity, f
        )
        mass_eff_new = 0.5 * cg.mass
        c_new = math.sqrt(cg.kn / mass_eff_new) * cg.radius
        c_dev = abs(c_new - c_orig) / c_orig * 100.0

        tc_new = contact_time_analytic(cg.kn, mass_eff_new)
        tc_orig = contact_time_analytic(kn, mass_eff_orig)
        tc_dev = abs(tc_new - tc_orig) / tc_orig * 100.0

        e_new = restitution_from_damping(cg.gamma_pair, cg.kn, mass_eff_new)
        e_dev = abs(e_new - restitution)

        rows.append({
            "f": f,
            "radius": cg.radius,
            "mass": cg.mass,
            "kn": cg.kn,
            "gamma": cg.gamma_pair,
            "sound_speed": c_new,
            "sound_speed_dev_pct": c_dev,
            "contact_time": tc_new,
            "contact_time_dev_pct": tc_dev,
            "restitution": e_new,
            "restitution_dev": e_dev,
        })

    return {
        "base": {
            "radius": radius,
            "mass": mass_orig,
            "kn": kn,
            "sound_speed": c_orig,
            "contact_time": contact_time_analytic(kn, mass_eff_orig),
        },
        "scaled": rows,
    }


# ---------------------------------------------------------------------------
# Macro-anchor simulation wrappers
# ---------------------------------------------------------------------------

def make_repose_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.4,
    f: float = 1.0,
    approach: str = "contact_time_invariant",
) -> dict:
    """Build a ReposeCase with coarse-grained parameters."""
    if approach == "contact_time_invariant":
        cg = scale_contact_time_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
        )
    else:
        cg = scale_sound_speed_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
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
        "cylinder_radius": 0.05 * f,
        "fill_height": 0.08 * f,
        "packing_gap": 0.02,
        "jitter": 0.02,
        "hold_time": 0.05,
        "lift_time": 0.15,
        "settle_time": 0.30,
        "dt": cg.timestep,
        "seed": 4970,
        "approach": approach,
    }


def make_shear_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.3,
    f: float = 1.0,
    approach: str = "contact_time_invariant",
) -> dict:
    """Build parameters for a direct-shear cell with coarse-grained particles."""
    if approach == "contact_time_invariant":
        cg = scale_contact_time_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
        )
    else:
        cg = scale_sound_speed_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
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
        "approach": approach,
    }


def make_ucs_case(
    mu: float = 0.5,
    mu_rolling: float = 0.05,
    radius: float = 0.005,
    density: float = 2500.0,
    kn: float = 1.0e5,
    kt_ratio: float = 0.5,
    restitution: float = 0.3,
    f: float = 1.0,
    approach: str = "contact_time_invariant",
) -> dict:
    """Build parameters for a uniaxial compression test with coarse-grained particles."""
    if approach == "contact_time_invariant":
        cg = scale_contact_time_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
        )
    else:
        cg = scale_sound_speed_invariant(
            radius, density, kn, kt_ratio, restitution, mu, mu_rolling, 9.81, f
        )
    # Specimen dimensions scale with f
    specimen_radius = 0.02 * f
    specimen_height = 0.04 * f
    return {
        "radius": cg.radius,
        "density": density,
        "mu": cg.mu,
        "mu_rolling": cg.mu_rolling,
        "restitution": cg.restitution,
        "kn": cg.kn,
        "kt_ratio": kt_ratio,
        "gravity": 0.0,  # UCS typically no gravity
        "specimen_radius": specimen_radius,
        "specimen_height": specimen_height,
        "packing_gap": 0.02,
        "jitter": 0.01,
        "dt": cg.timestep,
        "seed": 7201,
        "strain_rate": 0.01,  # m/s platen speed
        "strain_target": 0.02,
        "approach": approach,
    }


# ---------------------------------------------------------------------------
# Scaling rule table
# ---------------------------------------------------------------------------

def scaling_rule_table() -> dict:
    """Return the complete scaling rule table for both口径."""
    return {
        "quantity": {
            "radius": {"A": "f·R", "B": "f·R"},
            "mass": {"A": "f³·m", "B": "f³·m"},
            "inertia": {"A": "f⁵·I", "B": "f⁵·I"},
            "kn": {"A": "f³·k_n", "B": "f·k_n"},
            "kt": {"A": "f³·k_t", "B": "f·k_t"},
            "kr": {"A": "f⁵·k_r", "B": "f³·k_r"},
            "gamma": {"A": "f³·γ", "B": "f²·γ"},
            "contact_time": {"A": "t_c (invariant)", "B": "f·t_c"},
            "sound_speed": {"A": "f·c", "B": "c (invariant)"},
            "restitution": {"A": "e (invariant)", "B": "e (invariant)"},
            "mu": {"A": "μ (invariant)", "B": "μ (invariant)"},
            "mu_rolling": {"A": "μ_r (invariant)", "B": "μ_r (invariant)"},
            "gravity": {"A": "g (invariant)", "B": "g (invariant)"},
        },
        "applicability": {
            "A_contact_time_invariant": {
                "best_for": "Contact dynamics, force-displacement curves, restitution",
                "valid_range": "f ≤ 4 (f=8 may show minor deviations)",
                "failure_mode": "Wave speed scales as f·c, may affect dynamic loading",
            },
            "B_sound_speed_invariant": {
                "best_for": "Wave propagation, dynamic loading, seismic response",
                "valid_range": "f ≤ 4 (f=8 contact time grows, may need larger dt)",
                "failure_mode": "Contact time grows as f·t_c, force peaks may smear",
            },
        },
    }


# ---------------------------------------------------------------------------
# Cross-validation: compare macro quantities between口径
# ---------------------------------------------------------------------------

def cross_validate_approaches(
    base_params: dict,
    f_values: list[float],
) -> dict:
    """Compare macro-level predictions between口径 A and B.

    Returns a dict with per-f deviation between the two approaches.
    """
    results = []
    for f in f_values:
        cg_a = scale_contact_time_invariant(**base_params, f=f)
        cg_b = scale_sound_speed_invariant(**base_params, f=f)

        # Compare key ratios
        kn_ratio = cg_a.kn / cg_b.kn  # should be f²
        gamma_ratio = cg_a.gamma_pair / cg_b.gamma_pair  # should be f
        tc_ratio = cg_a.contact_time / cg_b.contact_time  # should be 1/f

        # Sound speed comparison
        me_a = 0.5 * cg_a.mass
        me_b = 0.5 * cg_b.mass
        c_a = math.sqrt(cg_a.kn / me_a) * cg_a.radius
        c_b = math.sqrt(cg_b.kn / me_b) * cg_b.radius

        results.append({
            "f": f,
            "kn_ratio_A_over_B": kn_ratio,
            "kn_ratio_expected": f ** 2,
            "gamma_ratio_A_over_B": gamma_ratio,
            "gamma_ratio_expected": f,
            "tc_ratio_A_over_B": tc_ratio,
            "tc_ratio_expected": 1.0 / f,
            "sound_speed_A": c_a,
            "sound_speed_B": c_b,
            "sound_speed_ratio": c_a / c_b if c_b > 0 else float("inf"),
        })

    return {"cross_validation": results}
