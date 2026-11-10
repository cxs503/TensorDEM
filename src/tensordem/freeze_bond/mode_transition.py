"""Freeze-bond strength and the adhesive -> cohesive failure-mode transition.

Physical target
---------------
Ice frozen onto a sub-cooled structural surface forms a *freeze bond*.  Two
competing failure mechanisms exist when the bond is pulled apart:

* **adhesive failure** — the crack runs along the ice/steel interface, i.e. the
  interface bond is the weaker link.  Dominant at warm sub-cooling and short
  contact times.
* **cohesive failure** — the bond is stronger than the ice, so the crack runs
  through the ice body.  Dominant at cold sub-cooling and long contact times.

Because the two branches have very different temperature sensitivity the peak
force versus temperature curve has a **kink** at the transition temperature
``T_trans``: steep on the adhesive side, nearly flat on the cohesive side (the
ice body strength depends only weakly on ``ΔT``).  This module implements the
whole chain — interface constitutive law, mode selection, transition
temperature, closed-form traction-separation (``F-u``) curves with a fracture
energy ledger, a discrete fibre-bundle (DEM-style) pull-off that reproduces the
continuum curve from ``N`` independent bonds, and the comparison machinery for
the laboratory anchor set (Molenaar 1996, see
``benchmarks/freeze_bond/freeze_bond_anchor_set.json``).

Constitutive laws (SI, temperatures in °C, ``ΔT = T_melt - T ≥ 0``)
-------------------------------------------------------------------
interface bond strength (acceptance criterion (a))::

    σ_bond(T) = σ_0 · exp(-β · (T - T_melt)) = σ_0 · exp(β · ΔT),   β > 0

contact-time growth of the bonded area fraction::

    a(t) = 1 - exp(-t / τ),      σ_bond(T, t) = σ_bond(T) · a(t)

bulk ice tensile strength (weak temperature dependence)::

    σ_ice(T) = σ_ice_ref · (ΔT / ΔT_ref)^n,   n ≈ 0.25

failure-mode selection and peak stress::

    cohesive  if  σ_bond ≥ σ_ice / k_coh      →  σ_fail = k_coh · σ_ice
    adhesive  otherwise                       →  σ_fail = σ_bond

with ``k_coh`` a cohesion-reinforcement factor (``k_coh = 1`` is the pure
stress criterion, which gives a *continuous, kinked* ``F_p(T)``; ``k_coh > 1``
adds the energy-based peak-load jump used in earlier freeze-bond models).

traction-separation with exact fracture energy ``G_f`` per unit bonded area::

    F(u) = k u                     u ≤ u_p,   u_p = F_p / k,  k = E_ice A / h
    bilinear:      F = F_p (1 - (u-u_p)/(u_f-u_p)),  u_f-u_p = 2 G_f A / F_p
    exponential:   F = F_p exp(-(u-u_p)/u_c),        u_c    = G_f A / F_p

so that ``∫_0^∞ F du = ½ F_p u_p + G_f A`` exactly in both cases.

Evidence grading
----------------
REAL   : the constitutive chain, the mode selection, the closed-form ``F-u``
         curves, their fracture-energy ledger, and the fibre-bundle pull-off
         (energy closure to machine precision; bundle-vs-continuum agreement).
WITHHELD : external validity against Molenaar (1996).  The anchor set shipped in
         ``benchmarks/freeze_bond/freeze_bond_anchor_set.json`` is a
         *reconstructed* anchor (see its ``provenance`` block): peak forces,
         initial stiffness and post-peak separations were taken from reported
         ranges, the individual coordinates were reconstructed to be consistent
         with them and are **not** point-digitised originals.  Until a true
         digitisation is supplied every literature-comparison number is a
         comparison against a declared anchor, not against measured data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "T_MELT_C",
    "MODE_ADHESIVE",
    "MODE_COHESIVE",
    "MODE_NONE",
    "BondInterfaceParams",
    "ForceDisplacement",
    "bond_strength_temperature",
    "bond_area_fraction",
    "melt_ramp",
    "bond_strength",
    "ice_tensile_strength",
    "failure_stress",
    "failure_mode",
    "transition_temperature",
    "peak_force",
    "traction_separation",
    "pull_off_curve",
    "fracture_energy_J",
    "peak_jump_ratio",
    "peak_slope_kink",
    "failure_mode_map",
    "fiber_bundle_pull_off",
    "r_squared",
    "load_anchor_set",
    "validate_peak_tests",
    "validate_curves",
    "validate_against_anchor",
    "run_validation_report",
    "EVIDENCE",
]

T_MELT_C: float = 0.0
MODE_ADHESIVE = "adhesive"
MODE_COHESIVE = "cohesive"
MODE_NONE = "no_bond"

EVIDENCE: dict[str, str] = {
    "constitutive_law": "REAL",
    "mode_transition": "REAL",
    "traction_separation_ledger": "REAL",
    "fiber_bundle_pull_off": "REAL",
    "molenaar1996_external": "WITHHELD",
}

ANCHOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "freeze_bond"
    / "freeze_bond_anchor_set.json"
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BondInterfaceParams:
    """Physical parameters of one ice/structure freeze-bond patch.

    Defaults are freshwater-ice-on-mild-steel lab-scale values (patch
    ``A = 1 cm²``, ice layer ``h = 10 mm``).  Magnitudes are consistent with the
    reported ranges of ice adhesion (≈0.4–2 MPa at −2…−20 °C) and bulk ice
    tensile strength (≈0.7–1.5 MPa); see the module docstring for grading.
    """

    sigma_0_pa: float = 0.35e6        # bond strength extrapolated to T_melt
    beta_per_C: float = 0.09          # temperature sensitivity, 1/°C  (>0)
    t_melt_c: float = T_MELT_C
    tau_growth_s: float = 30.0        # bond-area growth time constant
    sigma_ice_ref_pa: float = 0.75e6  # ice tensile strength at ΔT_ref
    ice_ref_subcooling_C: float = 10.0
    ice_exponent: float = 0.25
    area_m2: float = 1.0e-4           # bonded patch area (1 cm²)
    young_ice_pa: float = 9.0e9
    ice_thickness_m: float = 0.01
    G_fa_J_m2: float = 3.4            # adhesive (interface) fracture energy
    G_fc_J_m2: float = 19.0           # cohesive (bulk ice) fracture energy
    k_cohesive: float = 1.0           # cohesion reinforcement factor
    t_ramp_C: float = 1.0             # melt-film existence ramp width (ΔT)
    softening: str = "bilinear"       # "bilinear" | "exponential"

    def __post_init__(self) -> None:
        for name in (
            "sigma_0_pa",
            "beta_per_C",
            "tau_growth_s",
            "sigma_ice_ref_pa",
            "ice_ref_subcooling_C",
            "ice_exponent",
            "area_m2",
            "young_ice_pa",
            "ice_thickness_m",
            "G_fa_J_m2",
            "G_fc_J_m2",
            "k_cohesive",
            "t_ramp_C",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if name == "ice_exponent":
                continue
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.softening not in ("bilinear", "exponential"):
            raise ValueError(
                f"softening must be 'bilinear' or 'exponential', got {self.softening!r}"
            )

    @property
    def stiffness_n_per_m(self) -> float:
        """Series interface stiffness ``k = E_ice A / h`` [N/m]."""
        return self.young_ice_pa * self.area_m2 / self.ice_thickness_m

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["stiffness_n_per_m"] = self.stiffness_n_per_m
        return d


# ---------------------------------------------------------------------------
# Constitutive chain
# ---------------------------------------------------------------------------
def bond_strength_temperature(
    temperature_c: float, p: BondInterfaceParams | None = None
) -> float:
    """``σ_bond(T) = σ_0 · exp(-β (T - T_melt))`` [Pa]; 0 at/above the melt point."""
    p = p or BondInterfaceParams()
    if temperature_c >= p.t_melt_c:
        return 0.0
    return p.sigma_0_pa * math.exp(-p.beta_per_C * (temperature_c - p.t_melt_c))


def bond_area_fraction(contact_time_s: float, p: BondInterfaceParams | None = None) -> float:
    """Bonded area fraction ``1 - exp(-t/τ)`` in ``[0, 1)``."""
    p = p or BondInterfaceParams()
    if contact_time_s <= 0.0:
        return 0.0
    return 1.0 - math.exp(-contact_time_s / p.tau_growth_s)


def melt_ramp(temperature_c: float, p: BondInterfaceParams | None = None) -> float:
    """Existence factor of the interfacial water film: ``min(1, ΔT/ΔT_ramp)``.

    The bonded film cannot exist at the melt point, so the *physical* interface
    strength vanishes continuously as ``T → T_melt`` (acceptance criterion (d)),
    while the asymptotic Arrhenius law :func:`bond_strength_temperature` is
    recovered for ``ΔT ≥ ΔT_ramp``.
    """
    p = p or BondInterfaceParams()
    if temperature_c >= p.t_melt_c:
        return 0.0
    subcool = p.t_melt_c - temperature_c
    return min(1.0, subcool / p.t_ramp_C)


def bond_strength(
    temperature_c: float,
    contact_time_s: float,
    p: BondInterfaceParams | None = None,
) -> float:
    """Physical interface strength ``σ_bond·melt_ramp·area_fraction`` [Pa].

    Vanishes continuously at the melt point (criterion (d)); equals the
    Arrhenius law :func:`bond_strength_temperature` for ``ΔT ≥ ΔT_ramp`` and
    saturated contact time.
    """
    p = p or BondInterfaceParams()
    return (
        bond_strength_temperature(temperature_c, p)
        * melt_ramp(temperature_c, p)
        * bond_area_fraction(contact_time_s, p)
    )


def ice_tensile_strength(
    temperature_c: float, p: BondInterfaceParams | None = None
) -> float:
    """Bulk ice tensile strength ``σ_ice = σ_ice_ref (ΔT/ΔT_ref)^n`` [Pa]."""
    p = p or BondInterfaceParams()
    if temperature_c >= p.t_melt_c:
        return 0.0
    subcool = p.t_melt_c - temperature_c
    return p.sigma_ice_ref_pa * (subcool / p.ice_ref_subcooling_C) ** p.ice_exponent


def failure_mode(
    sigma_bond_pa: float, sigma_ice_pa: float, p: BondInterfaceParams | None = None
) -> str:
    """Mode label: cohesive iff ``σ_bond ≥ σ_ice / k_coh``, else adhesive."""
    p = p or BondInterfaceParams()
    if sigma_bond_pa <= 0.0 or sigma_ice_pa <= 0.0:
        return MODE_NONE
    if sigma_bond_pa >= sigma_ice_pa / p.k_cohesive:
        return MODE_COHESIVE
    return MODE_ADHESIVE


def failure_stress(
    temperature_c: float,
    contact_time_s: float,
    p: BondInterfaceParams | None = None,
) -> tuple[float, str]:
    """Return ``(σ_fail [Pa], mode)``; ``σ_fail`` is the strength of the weaker link."""
    p = p or BondInterfaceParams()
    sb = bond_strength(temperature_c, contact_time_s, p)
    si = ice_tensile_strength(temperature_c, p)
    mode = failure_mode(sb, si, p)
    if mode == MODE_NONE:
        return 0.0, MODE_NONE
    if mode == MODE_COHESIVE:
        return p.k_cohesive * si, MODE_COHESIVE
    return sb, MODE_ADHESIVE


def _stress_gap(
    temperature_c: float, contact_time_s: float, p: BondInterfaceParams
) -> float:
    """``σ_bond - σ_ice/k_coh``; sign changes at the transition temperature."""
    return bond_strength(temperature_c, contact_time_s, p) - (
        ice_tensile_strength(temperature_c, p) / p.k_cohesive
    )


def transition_temperature(
    p: BondInterfaceParams | None = None,
    contact_time_s: float = 600.0,
    t_warm_c: float = -1.0e-3,
    t_cold_c: float = -60.0,
) -> float:
    """Solve ``σ_bond(T) = σ_ice(T)/k_coh`` on ``[t_cold_c, t_warm_c]``.

    Raises ``ValueError`` when no transition exists inside the bracket (e.g. a
    contact time so short that the bond never reaches the ice strength).
    """
    p = p or BondInterfaceParams()
    f_warm = _stress_gap(t_warm_c, contact_time_s, p)  # sub-cooled close to melt
    f_cold = _stress_gap(t_cold_c, contact_time_s, p)  # deep sub-cooling
    if f_warm >= 0.0:
        raise ValueError(
            "no transition: the bond already exceeds the ice strength at the "
            "warm bracket end (contact time too long / sigma_0 too high)"
        )
    if f_cold <= 0.0:
        raise ValueError(
            "no transition: the bond never reaches the ice strength down to "
            f"{t_cold_c} °C (contact time too short / beta too small)"
        )
    lo, hi = t_cold_c, t_warm_c  # f(lo) > 0, f(hi) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _stress_gap(mid, contact_time_s, p) > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1.0e-10:
            break
    return 0.5 * (lo + hi)


def peak_force(
    temperature_c: float,
    contact_time_s: float = 600.0,
    p: BondInterfaceParams | None = None,
) -> dict[str, Any]:
    """Peak force of a displacement-controlled pull-off [N] and its mode."""
    p = p or BondInterfaceParams()
    sigma_fail, mode = failure_stress(temperature_c, contact_time_s, p)
    fp = sigma_fail * p.area_m2
    return {
        "temperature_c": float(temperature_c),
        "contact_time_s": float(contact_time_s),
        "bond_strength_pa": bond_strength(temperature_c, contact_time_s, p),
        "ice_strength_pa": ice_tensile_strength(temperature_c, p),
        "failure_stress_pa": sigma_fail,
        "peak_force_n": fp,
        "mode": mode,
        "peak_displacement_m": fp / p.stiffness_n_per_m,
    }


def peak_jump_ratio(
    p: BondInterfaceParams | None = None, contact_time_s: float = 600.0
) -> dict[str, Any]:
    """Peak-load step across the transition, one-sided limits.

    With the pure stress criterion (``k_coh = 1``) the ratio is 1 — the two
    branches meet continuously and the transition shows as a *kink*; the ratio
    becomes ``k_coh²`` only if the cohesive branch is reinforced by ``k_coh``
    while the mode criterion is not.
    """
    p = p or BondInterfaceParams()
    t_trans = transition_temperature(p, contact_time_s=contact_time_s)
    eps = 1.0e-4
    f_cold = peak_force(t_trans - eps, contact_time_s, p)["peak_force_n"]
    f_warm = peak_force(t_trans + eps, contact_time_s, p)["peak_force_n"]
    return {
        "transition_temperature_c": t_trans,
        "peak_force_cold_n": f_cold,
        "peak_force_warm_n": f_warm,
        "jump_ratio": f_cold / f_warm if f_warm > 0 else float("inf"),
        "k_cohesive": p.k_cohesive,
    }


def peak_slope_kink(
    p: BondInterfaceParams | None = None,
    contact_time_s: float = 600.0,
    dt_c: float = 0.5,
) -> dict[str, float]:
    """One-sided ``dF_p/dT`` [N/°C] on both sides of the transition.

    The adhesive branch is Arrhenius-steep, the cohesive branch only follows the
    weak ``ΔT^n`` of the ice body, so the ratio should approach the analytic
    value ``β·ΔT_trans / n`` (≈ 2.8 for the default parameters) — a *kink*, not a
    jump.  This is the measurable signature of the mode transition.
    """
    p = p or BondInterfaceParams()
    t_trans = transition_temperature(p, contact_time_s=contact_time_s)

    def slope(t_a: float, t_b: float) -> float:
        fa = peak_force(t_a, contact_time_s, p)["peak_force_n"]
        fb = peak_force(t_b, contact_time_s, p)["peak_force_n"]
        return (fb - fa) / (t_b - t_a)

    # warmer than T_trans -> adhesive branch; colder -> cohesive branch
    s_adhesive = slope(t_trans + dt_c, t_trans + 2.0 * dt_c)
    s_cohesive = slope(t_trans - 2.0 * dt_c, t_trans - dt_c)
    ratio = abs(s_adhesive) / abs(s_cohesive) if s_cohesive != 0.0 else float("inf")
    subcool = p.t_melt_c - t_trans
    predicted = p.beta_per_C * subcool / p.ice_exponent
    return {
        "transition_temperature_c": t_trans,
        "slope_adhesive_n_per_c": s_adhesive,
        "slope_cohesive_n_per_c": s_cohesive,
        "slope_ratio": ratio,
        "slope_ratio_predicted": predicted,
        "slope_ratio_rel_error": abs(ratio - predicted) / predicted,
    }


# ---------------------------------------------------------------------------
# Traction-separation (F-u) curve
# ---------------------------------------------------------------------------
@dataclass
class ForceDisplacement:
    """A displacement-controlled pull-off curve with its energy ledger."""

    temperature_c: float
    contact_time_s: float
    mode: str
    u_m: np.ndarray
    force_n: np.ndarray
    peak_force_n: float
    peak_displacement_m: float
    stiffness_n_per_m: float
    fracture_energy_J: float
    work_in_J: float
    G_f_J_m2: float
    softening: str
    n_points: int
    elastic_energy_J: float = 0.0

    @property
    def dissipated_energy_J(self) -> float:
        """Work absorbed beyond the reversible elastic triangle [J]."""
        return self.work_in_J - self.elastic_energy_J

    @property
    def energy_error(self) -> float:
        """Relative gap between the measured dissipation and ``G_f·A``.

        The loading work splits into a reversible elastic term ``½F_p²/k`` and
        the irreversible fracture term ``G_f·A``; a cohesive element must spend
        exactly ``G_f`` per unit debonded area, which is what this residual
        checks.
        """
        if self.fracture_energy_J <= 0.0:
            return 0.0
        return abs(self.dissipated_energy_J - self.fracture_energy_J) / self.fracture_energy_J

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary (arrays are omitted, they live in the npz/CSV)."""
        return {
            "temperature_c": self.temperature_c,
            "contact_time_s": self.contact_time_s,
            "mode": self.mode,
            "peak_force_n": self.peak_force_n,
            "peak_displacement_m": self.peak_displacement_m,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "fracture_energy_J": self.fracture_energy_J,
            "work_in_J": self.work_in_J,
            "elastic_energy_J": self.elastic_energy_J,
            "dissipated_energy_J": self.dissipated_energy_J,
            "G_f_J_m2": self.G_f_J_m2,
            "energy_relative_error": self.energy_error,
            "softening": self.softening,
            "n_points": self.n_points,
        }


def fracture_energy_J(mode: str, p: BondInterfaceParams | None = None) -> float:
    """Total fracture energy ``G_f · A`` [J] for a failure mode."""
    p = p or BondInterfaceParams()
    g = p.G_fc_J_m2 if mode == MODE_COHESIVE else p.G_fa_J_m2
    return g * p.area_m2


def traction_separation(
    u_m: Sequence[float] | np.ndarray,
    temperature_c: float,
    contact_time_s: float = 600.0,
    p: BondInterfaceParams | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate ``F(u)`` [N] for a given interface state."""
    p = p or BondInterfaceParams()
    u = np.asarray(u_m, dtype=float)
    if np.any(u < 0.0):
        raise ValueError("displacement must be non-negative")
    info = peak_force(temperature_c, contact_time_s, p)
    fp = info["peak_force_n"]
    mode = info["mode"]
    if fp <= 0.0:
        return np.zeros_like(u), MODE_NONE
    k = p.stiffness_n_per_m
    u_p = fp / k
    energy = fracture_energy_J(mode, p)
    force = np.empty_like(u)
    elastic = u <= u_p
    force[elastic] = k * u[elastic]
    if p.softening == "bilinear":
        u_f = u_p + 2.0 * energy / fp
        post = ~elastic & (u < u_f)
        force[post] = fp * (1.0 - (u[post] - u_p) / (u_f - u_p))
        force[u >= u_f] = 0.0
    else:  # exponential
        u_c = energy / fp
        post = ~elastic
        force[post] = fp * np.exp(-(u[post] - u_p) / u_c)
        force[force < 1.0e-9 * fp] = 0.0
    return force, mode


def pull_off_curve(
    temperature_c: float,
    contact_time_s: float = 600.0,
    p: BondInterfaceParams | None = None,
    n_points: int = 801,
    n_post: int = 120,
) -> ForceDisplacement:
    """Closed-form pull-off curve with the exact fracture-energy ledger.

    ``n_post`` points are placed on the softening branch so that the trapezoidal
    work integral resolves the tail to better than ``1e-6`` relative in energy.
    """
    p = p or BondInterfaceParams()
    info = peak_force(temperature_c, contact_time_s, p)
    fp = info["peak_force_n"]
    mode = info["mode"]
    if fp <= 0.0:
        u = np.zeros(2)
        return ForceDisplacement(
            temperature_c=float(temperature_c),
            contact_time_s=float(contact_time_s),
            mode=MODE_NONE,
            u_m=u,
            force_n=np.zeros_like(u),
            peak_force_n=0.0,
            peak_displacement_m=0.0,
            stiffness_n_per_m=p.stiffness_n_per_m,
            fracture_energy_J=0.0,
            work_in_J=0.0,
            G_f_J_m2=0.0,
            softening=p.softening,
            n_points=2,
        )
    k = p.stiffness_n_per_m
    u_p = fp / k
    energy = fracture_energy_J(mode, p)
    if p.softening == "bilinear":
        u_end = u_p + 2.0 * energy / fp
        n_post_eff = max(n_post, 2)
    else:
        # cut the exponential tail at 1e-9 F_p and resolve it finely enough that
        # the trapezoidal work integral closes on G_f to < 1e-6 relative
        u_end = u_p + (energy / fp) * math.log(1.0 / 1.0e-9)
        n_post_eff = max(n_post, 12000)
    u = np.concatenate(
        [
            np.linspace(0.0, u_p, max(n_points, 2)),
            np.linspace(u_p, u_end, n_post_eff)[1:],
        ]
    )
    force, mode = traction_separation(u, temperature_c, contact_time_s, p)
    work = float(np.trapezoid(force, u))
    return ForceDisplacement(
        temperature_c=float(temperature_c),
        contact_time_s=float(contact_time_s),
        mode=mode,
        u_m=u,
        force_n=force,
        peak_force_n=float(np.max(force)),
        peak_displacement_m=u_p,
        stiffness_n_per_m=k,
        fracture_energy_J=energy,
        work_in_J=work,
        G_f_J_m2=energy / p.area_m2,
        softening=p.softening,
        n_points=u.size,
        elastic_energy_J=0.5 * fp * u_p,
    )


# ---------------------------------------------------------------------------
# Discrete (DEM-style) pull-off: N independent bond fibres
# ---------------------------------------------------------------------------
def fiber_bundle_pull_off(
    temperature_c: float,
    contact_time_s: float = 600.0,
    p: BondInterfaceParams | None = None,
    n_fibers: int = 1024,
    weibull_m: float = 12.0,
    seed: int = 20261110,
    n_u: int = 8001,
) -> ForceDisplacement:
    """Pull-off of a patch discretised into ``n_fibers`` independent bonds.

    Each fibre carries an area ``A/N``, its own strength drawn from a Weibull
    distribution (shape ``m``, mean-normalised) and the *same* fracture energy
    per unit area, so the ensemble energy closes on ``G_f A`` exactly.  This is
    the discrete counterpart of :func:`pull_off_curve`; the mean response is
    compared with the continuum curve in the benchmark.
    """
    p = p or BondInterfaceParams()
    if n_fibers < 2:
        raise ValueError("n_fibers must be >= 2")
    rng = np.random.default_rng(seed)
    a_f = p.area_m2 / n_fibers
    info = peak_force(temperature_c, contact_time_s, p)
    sigma_fail = info["failure_stress_pa"]
    mode = info["mode"]
    if sigma_fail <= 0.0:
        u = np.zeros(2)
        return ForceDisplacement(
            temperature_c=float(temperature_c),
            contact_time_s=float(contact_time_s),
            mode=MODE_NONE,
            u_m=u,
            force_n=np.zeros_like(u),
            peak_force_n=0.0,
            peak_displacement_m=0.0,
            stiffness_n_per_m=p.stiffness_n_per_m,
            fracture_energy_J=0.0,
            work_in_J=0.0,
            G_f_J_m2=0.0,
            softening=p.softening,
            n_points=2,
        )
    # Weibull strengths with unit mean: W = U^(1/m) * Gamma(1+1/m)
    shape = weibull_m
    scale = 1.0 / math.gamma(1.0 + 1.0 / shape)
    w = rng.weibull(shape, size=n_fibers) * scale
    f_i = sigma_fail * a_f * w                      # fibre peak force [N]
    k_i = float(p.young_ice_pa * a_f / p.ice_thickness_m)  # fibre stiffness [N/m]
    u_p = f_i / k_i
    g_i = fracture_energy_J(mode, p) / n_fibers     # fibre fracture energy [J]
    if p.softening == "bilinear":
        u_end = float(np.max(u_p + 2.0 * g_i / f_i))
    else:
        u_c = g_i / f_i
        u_end = float(np.max(u_p + u_c * math.log(1.0 / 1.0e-9)))
    u = np.linspace(0.0, u_end, n_u)
    # F_i(u) = k_i u  (u<=u_p,i);  f_i * softening((u-u_p,i))  otherwise
    uu = u[None, :]
    up = u_p[:, None]
    fi = f_i[:, None]
    elastic = uu <= up
    if p.softening == "bilinear":
        uf = up + 2.0 * (g_i / f_i)[:, None]
        soft = 1.0 - (uu - up) / (uf - up)
        soft = np.clip(soft, 0.0, 1.0)
    else:
        uc = (g_i / f_i)[:, None]
        soft = np.exp(-(uu - up) / uc)
        soft = np.where(soft < 1.0e-9, 0.0, soft)
    force = np.where(elastic, k_i * uu, fi * soft)
    total = force.sum(axis=0)
    work = float(np.trapezoid(total, u))
    return ForceDisplacement(
        temperature_c=float(temperature_c),
        contact_time_s=float(contact_time_s),
        mode=mode,
        u_m=u,
        force_n=total,
        peak_force_n=float(np.max(total)),
        peak_displacement_m=float(u[int(np.argmax(total))]),
        stiffness_n_per_m=p.stiffness_n_per_m,
        fracture_energy_J=fracture_energy_J(mode, p),
        work_in_J=work,
        G_f_J_m2=fracture_energy_J(mode, p) / p.area_m2,
        softening=p.softening,
        n_points=u.size,
        elastic_energy_J=float(0.5 * np.sum(f_i * u_p)),
    )


# ---------------------------------------------------------------------------
# Failure-mode map
# ---------------------------------------------------------------------------
def failure_mode_map(
    temperatures_c: Sequence[float] | np.ndarray,
    contact_times_s: Sequence[float] | np.ndarray,
    p: BondInterfaceParams | None = None,
) -> dict[str, Any]:
    """2-D map of the failure mode over (temperature, contact time)."""
    p = p or BondInterfaceParams()
    temps = np.asarray(temperatures_c, dtype=float)
    times = np.asarray(contact_times_s, dtype=float)
    codes = np.zeros((times.size, temps.size), dtype=int)
    for i, t in enumerate(times):
        for j, temp in enumerate(temps):
            sb = bond_strength(temp, t, p)
            si = ice_tensile_strength(temp, p)
            mode = failure_mode(sb, si, p)
            codes[i, j] = {MODE_NONE: 0, MODE_ADHESIVE: 1, MODE_COHESIVE: 2}[mode]
    return {
        "temperatures_c": temps,
        "contact_times_s": times,
        "mode_codes": codes,
        "code_map": {"0": MODE_NONE, "1": MODE_ADHESIVE, "2": MODE_COHESIVE},
    }


# ---------------------------------------------------------------------------
# Comparison machinery against the laboratory anchor
# ---------------------------------------------------------------------------
def r_squared(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """Coefficient of determination ``1 - SS_res/SS_tot``."""
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed must have the same shape")
    if obs.size < 2:
        return 0.0
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - float(np.mean(obs))) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def load_anchor_set(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the laboratory anchor set."""
    p = Path(path) if path is not None else ANCHOR_PATH
    if not p.is_file():
        raise FileNotFoundError(f"anchor set not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("dataset_id", "provenance", "test_conditions", "peak_tests", "curves", "transition"):
        if key not in data:
            raise ValueError(f"anchor set missing section {key!r}")
    if data["provenance"].get("evidence_grade") not in ("REAL", "WITHHELD"):
        raise ValueError("anchor provenance must declare its evidence grade")
    temps = [row["T_c"] for row in data["peak_tests"]]
    if len(temps) < 5:
        raise ValueError("anchor set needs at least 5 peak tests")
    if any(b <= a for a, b in zip(temps, temps[1:])):
        raise ValueError("anchor peak tests must be sorted by increasing temperature")
    for curve in data["curves"]:
        if len(curve["u_um"]) != len(curve["F_N"]):
            raise ValueError(f"curve at {curve['T_c']} °C: u/F length mismatch")
        if len(curve["u_um"]) < 8:
            raise ValueError(f"curve at {curve['T_c']} °C: too few points")
    return data


def validate_peak_tests(
    anchor: dict[str, Any],
    p: BondInterfaceParams | None = None,
    contact_time_s: float | None = None,
) -> dict[str, Any]:
    """Peak force and failure mode versus temperature against the anchor."""
    p = p or BondInterfaceParams()
    t_c = contact_time_s if contact_time_s is not None else anchor["test_conditions"]["contact_time_s"]
    rows = []
    for row in anchor["peak_tests"]:
        t = row["T_c"]
        pred = peak_force(t, t_c, p)
        exp_f = row["F_peak_N"]
        dev = abs(pred["peak_force_n"] - exp_f) / exp_f
        agree = pred["mode"] == row["mode"]
        rows.append(
            {
                "T_c": t,
                "mode_obs": row["mode"],
                "mode_pred": pred["mode"],
                "mode_agree": agree,
                "F_peak_obs_n": exp_f,
                "F_peak_pred_n": pred["peak_force_n"],
                "rel_error": dev,
                "in_transition_band": bool(row.get("in_transition_band", False)),
            }
        )
    scored = [r for r in rows if not r["in_transition_band"]]
    errors = [r["rel_error"] for r in scored]
    return {
        "rows": rows,
        "max_rel_error": float(max(errors)),
        "mean_rel_error": float(np.mean(errors)),
        "mode_agreement": float(np.mean([r["mode_agree"] for r in scored])),
        "n_scored": len(scored),
        "contact_time_s": t_c,
    }


def transition_error(
    anchor: dict[str, Any],
    p: BondInterfaceParams | None = None,
    contact_time_s: float | None = None,
) -> dict[str, Any]:
    """``|T_trans,model - T_trans,anchor|`` against the anchor's own band."""
    p = p or BondInterfaceParams()
    t_c = contact_time_s if contact_time_s is not None else anchor["test_conditions"]["contact_time_s"]
    t_model = transition_temperature(p, contact_time_s=t_c)
    tr = anchor["transition"]
    t_ref = float(tr["T_trans_c"])
    band = float(tr.get("band_c", 0.0))
    return {
        "T_trans_model_c": t_model,
        "T_trans_anchor_c": t_ref,
        "anchor_band_c": band,
        "abs_error_c": abs(t_model - t_ref),
        "within_band": bool(abs(t_model - t_ref) <= band),
        "contact_time_s": t_c,
    }


def validate_curves(
    anchor: dict[str, Any],
    p: BondInterfaceParams | None = None,
    contact_time_s: float | None = None,
) -> dict[str, Any]:
    """``F-u`` curve comparison: shape ``R²``, peak error, fracture-energy error."""
    p = p or BondInterfaceParams()
    t_c = contact_time_s if contact_time_s is not None else anchor["test_conditions"]["contact_time_s"]
    out = []
    for curve in anchor["curves"]:
        t = curve["T_c"]
        u_obs = np.asarray(curve["u_um"], dtype=float) * 1.0e-6
        f_obs = np.asarray(curve["F_N"], dtype=float)
        f_pred, mode = traction_separation(u_obs, t, t_c, p)
        # anchor fracture energy from its own trapezoidal area
        e_obs = float(np.trapezoid(f_obs, u_obs))
        e_pred = float(np.trapezoid(f_pred, u_obs))
        g_obs = float(curve.get("G_f_J_m2", e_obs / p.area_m2))
        r2 = r_squared(f_pred, f_obs)
        peak_rel = abs(max(f_obs) - f_pred.max()) / max(f_obs)
        ge_rel = abs(e_pred - e_obs) / e_obs
        out.append(
            {
                "T_c": t,
                "mode_obs": curve["mode"],
                "mode_pred": mode,
                "n_points": int(u_obs.size),
                "r2": r2,
                "peak_force_obs_n": float(max(f_obs)),
                "peak_force_pred_n": float(f_pred.max()),
                "peak_rel_error": peak_rel,
                "work_obs_J": e_obs,
                "work_pred_J": e_pred,
                "G_f_obs_J_m2": g_obs,
                "fracture_energy_rel_error": ge_rel,
            }
        )
    return {
        "curves": out,
        "min_r2": float(min(c["r2"] for c in out)),
        "max_peak_rel_error": float(max(c["peak_rel_error"] for c in out)),
        "max_fracture_energy_rel_error": float(max(c["fracture_energy_rel_error"] for c in out)),
        "contact_time_s": t_c,
    }


def validate_against_anchor(
    anchor: dict[str, Any] | None = None,
    p: BondInterfaceParams | None = None,
    contact_time_s: float | None = None,
) -> dict[str, Any]:
    """Run the three external comparisons and return a flat summary."""
    anchor = anchor or load_anchor_set()
    p = p or BondInterfaceParams()
    t_c = contact_time_s if contact_time_s is not None else anchor["test_conditions"]["contact_time_s"]
    peaks = validate_peak_tests(anchor, p, t_c)
    trans = transition_error(anchor, p, t_c)
    curves = validate_curves(anchor, p, t_c)
    return {
        "dataset_id": anchor["dataset_id"],
        "anchor_evidence_grade": anchor["provenance"].get("evidence_grade"),
        "anchor_digitization": anchor["provenance"].get("digitization"),
        "contact_time_s": t_c,
        "peak_validation": peaks,
        "transition_validation": trans,
        "curve_validation": curves,
        "summary": {
            "max_peak_rel_error": peaks["max_rel_error"],
            "mode_agreement": peaks["mode_agreement"],
            "transition_abs_error_c": trans["abs_error_c"],
            "min_curve_r2": curves["min_r2"],
            "max_fracture_energy_rel_error": curves["max_fracture_energy_rel_error"],
        },
    }


# ---------------------------------------------------------------------------
# Predeclared acceptance thresholds (fixed before running the benchmark)
# ---------------------------------------------------------------------------
ACCEPTANCE: dict[str, float] = {
    "max_peak_rel_error": 0.15,       # platform accuracy standard: <15%
    "min_curve_r2": 0.90,             # criterion (c)
    "max_fracture_energy_rel_error": 0.15,
    "max_transition_error_c": 2.0,    # criterion (b)
    "min_bundle_r2": 0.90,            # discrete-vs-continuum (REAL tier)
    "max_energy_rel_error": 1.0e-6,   # ledger closure
}


def run_validation_report(
    p: BondInterfaceParams | None = None,
    contact_time_s: float | None = None,
) -> dict[str, Any]:
    """Full machine-checkable acceptance report for this node."""
    p = p or BondInterfaceParams()
    anchor = load_anchor_set()
    t_c = contact_time_s if contact_time_s is not None else anchor["test_conditions"]["contact_time_s"]
    res = validate_against_anchor(anchor, p, t_c)

    # (a) exponential law with beta > 0: recover beta from a log-linear fit
    temps = np.linspace(-1.0, -25.0, 49)
    sig = np.array([bond_strength_temperature(t, p) for t in temps])
    slope, intercept = np.polyfit(temps, np.log(sig), 1)
    beta_fit = -float(slope)

    # (d) negative control: T -> T_melt, physical interface strength vanishes
    dts = [-0.001, -0.01, -0.1, -1.0]
    ramp_series = [
        {"dT_C": abs(dt), "sigma_bond_pa": bond_strength(p.t_melt_c + dt, t_c, p)}
        for dt in dts
    ]
    neg_control = {
        "sigma_at_minus_0p001_C": bond_strength(p.t_melt_c - 0.001, t_c, p),
        "sigma_at_minus_0p01_C": bond_strength(p.t_melt_c - 0.01, t_c, p),
        "sigma_at_minus_0p1_C": bond_strength(p.t_melt_c - 0.1, t_c, p),
        "sigma_at_T_melt": bond_strength(p.t_melt_c, t_c, p),
        "sigma_at_plus_5_C": bond_strength(p.t_melt_c + 5.0, t_c, p),
        "asymptotic_law_at_minus_10_C": bond_strength_temperature(-10.0, p),
        "physical_at_minus_10_C": bond_strength(-10.0, t_c, p),
        "ramp_series": ramp_series,
        "limit_ratio": bond_strength(p.t_melt_c - 0.001, t_c, p) / p.sigma_0_pa,
    }

    # closed-form curve ledgers + discrete bundle cross-check
    ledgers, bundles = [], []
    for temp in (-5.0, -15.0):
        cd = pull_off_curve(temp, t_c, p)
        bd = fiber_bundle_pull_off(temp, t_c, p)
        # compare on the discrete grid, normalised by their own peaks
        f_cont, _ = traction_separation(bd.u_m, temp, t_c, p)
        continue_r2 = r_squared(bd.force_n / bd.peak_force_n, f_cont / f_cont.max())
        ledgers.append(cd.as_dict())
        item = bd.as_dict()
        item["bundle_vs_continuum_r2"] = continue_r2
        # grid-convergence of the trapezoidal work integral (2nd order expected)
        coarse = fiber_bundle_pull_off(temp, t_c, p, n_u=bd.u_m.size // 2)
        item["energy_error_coarse_grid"] = coarse.energy_error
        item["energy_convergence_order"] = (
            math.log(item["energy_error_coarse_grid"] / item["energy_relative_error"])
            / math.log(2.0)
            if item["energy_relative_error"] > 0.0
            else float("inf")
        )
        bundles.append(item)

    kink = peak_slope_kink(p, t_c)
    jump = peak_jump_ratio(p, t_c)
    jump_k2 = peak_jump_ratio(
        BondInterfaceParams(**{**p.as_dict_fields(), "k_cohesive": 2.0}), t_c
    )

    checks = {
        "a_temperature_law": {
            "form": "sigma_bond = sigma_0 * exp(-beta*(T-T_melt))",
            "beta_input_per_C": p.beta_per_C,
            "beta_fitted_per_C": beta_fit,
            "beta_fit_error": abs(beta_fit - p.beta_per_C),
            "passed": bool(beta_fit > 0.0 and abs(beta_fit - p.beta_per_C) < 1.0e-6),
        },
        "b_mode_transition": {
            **res["transition_validation"],
            "passed": bool(res["transition_validation"]["abs_error_c"] <= ACCEPTANCE["max_transition_error_c"]),
        },
        "c_force_displacement": {
            **res["curve_validation"],
            "threshold_r2": ACCEPTANCE["min_curve_r2"],
            "passed": bool(
                res["curve_validation"]["min_r2"] >= ACCEPTANCE["min_curve_r2"]
                and res["curve_validation"]["max_peak_rel_error"] <= ACCEPTANCE["max_peak_rel_error"]
                and res["curve_validation"]["max_fracture_energy_rel_error"]
                <= ACCEPTANCE["max_fracture_energy_rel_error"]
            ),
        },
        "d_negative_control": {
            **neg_control,
            "monotone_to_zero": bool(
                neg_control["sigma_at_minus_0p001_C"]
                < neg_control["sigma_at_minus_0p01_C"]
                < neg_control["sigma_at_minus_0p1_C"]
                < bond_strength(-1.0, t_c, p)
            ),
            "passed": bool(
                neg_control["sigma_at_T_melt"] == 0.0
                and neg_control["sigma_at_plus_5_C"] == 0.0
                and neg_control["limit_ratio"] < 2.0e-3
            ),
        },
        "peak_force_vs_temperature": {
            "threshold": ACCEPTANCE["max_peak_rel_error"],
            "passed": bool(res["peak_validation"]["max_rel_error"] <= ACCEPTANCE["max_peak_rel_error"]),
        },
        "energy_ledger": {
            "curves": ledgers,
            "max_energy_rel_error": max(c["energy_relative_error"] for c in ledgers),
            "passed": bool(max(c["energy_relative_error"] for c in ledgers) <= ACCEPTANCE["max_energy_rel_error"]),
        },
        "discrete_bundle": {
            "cases": bundles,
            "min_r2": float(min(b["bundle_vs_continuum_r2"] for b in bundles)),
            "max_energy_rel_error": max(b["energy_relative_error"] for b in bundles),
            "passed": bool(
                min(b["bundle_vs_continuum_r2"] for b in bundles) >= ACCEPTANCE["min_bundle_r2"]
                and max(b["energy_relative_error"] for b in bundles) <= 1.0e-6
            ),
        },
        "kink_signature": {**kink, "jump_k1": jump, "jump_k2": jump_k2},
    }
    all_passed = all(v.get("passed", True) for k, v in checks.items() if isinstance(v, dict))
    return {
        "node": "P46W3-N07",
        "module": "tensordem.freeze_bond.mode_transition",
        "parameters": p.as_dict(),
        "evidence": EVIDENCE,
        "anchor": {
            "dataset_id": res["dataset_id"],
            "evidence_grade": res["anchor_evidence_grade"],
            "digitization": res["anchor_digitization"],
        },
        "acceptance_thresholds": ACCEPTANCE,
        "checks": checks,
        "peak_validation": res["peak_validation"],
        "all_checks_passed": bool(all_passed),
    }


# helper: dataclass fields without the derived stiffness key
def _as_dict_fields(self: BondInterfaceParams) -> dict[str, Any]:  # pragma: no cover
    from dataclasses import asdict

    return asdict(self)


BondInterfaceParams.as_dict_fields = _as_dict_fields  # type: ignore[attr-defined]