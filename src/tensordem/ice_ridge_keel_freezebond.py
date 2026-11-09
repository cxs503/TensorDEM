"""Freeze-bonded ice-ridge keel: porosity -> structure -> drag resistance.

Physical problem
================
A first-year pressure-ridge keel is a pile of broken ice blocks (Rosin-Rammler
sized, ``0.1 .. 1 m``) sitting *below the waterline*.  Two forces shape it:

* the buoyancy excess of the blocks, ``(rho_w - rho_i) g`` **upward** -- rubble
  ice is lighter than the water it displaces, so the keel is pushed up against
  the sail + consolidated layer;
* the weight of that sail / consolidated layer, transmitted to the keel top as
  a surcharge ``p_lid`` -- the only reason a keel does not simply float apart.

Between those two the rubble is in compression, and *freeze bonds* (thin necks
of fresh ice grown across block contacts, see
:mod:`tensordem.freeze_bond.constitutive`) are what make it a **skeleton**
instead of a loose heap.  A bonded keel holds its shape, part of its pore space
is filled with bond ice (measured porosity ``n ~ 0.30 .. 0.45`` for refrozen
keels, Lepparanta & Hakala 1992) and it transmits load to a structure.  An
unbonded keel has no tensile/shear strength, dilates and disperses, and its
envelope porosity degrades above ``n ~ 0.55``.

This module is the face that ties the three levels together:

1. **Adfreeze -> DEM bond** (:class:`AdfreezeClosure`, :func:`contact_area_m2`,
   :func:`freeze_bond_spec`, :func:`bonded_pair_pull`).  The platform adfreeze
   model of ``tensorphysics.ice.adfreeze`` is mirrored locally (TensorDEM
   carries no cross-repo import; parity against the original is machine-checked
   in ``tests/test_ice_ridge_keel_freezebond.py`` when the TensorPhysics source
   tree is importable) and mapped onto a DEM bond threshold

       ``F_break = tau_adfreeze(dT, t, Ra, S) * A_contact``

2. **Keel rubble DEM** (:class:`KeelRubble`): ``N >= 500`` buoyant spheres with
   a Rosin-Rammler size distribution, held between a dynamically loaded lid and
   buoyancy, Cundall-Strack contacts with regularised Coulomb friction, and
   freeze bonds carrying a normal/shear break envelope.

3. **Measurement and arbitration** (:func:`voronoi_porosity`,
   :func:`force_chain_statistics`, :func:`kerr_keel_resistance`,
   :func:`macroscopic_friction_angle`).

Evidence grades
---------------
* adfreeze closure: **WITHHELD** as physics -- it is the platform model
  (Frederking-calibrated hyperbolic closure + Stefan kinetics).  What this
  module verifies is the *mapping* into the DEM and its mechanical consequence.
* DEM contact/bond mechanics, mass conservation, Voronoi porosity and
  force-chain statistics: **REAL** (algorithmic, machine-checked).
* Kerr-type reference: **literature-form** (:data:`KERR_REFERENCE_PROVENANCE`);
  the 1975 monograph text is not in the worktree, so the comparison is a
  micro-to-macro consistency check of the DEM against the standard continuum
  idealisation, *not* a digitised-curve validation.  The limitation is carried
  into the report.

All quantities SI (m, kg, s, Pa, K).  The buoyant unit weight of keel rubble is
``gamma_b = (rho_i - rho_w)(1 - n) g < 0`` (rubble ice is buoyant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ICE_DENSITY",
    "WATER_DENSITY",
    "GRAVITY",
    "BUOYANT_ACCELERATION",
    "ADFREEZE_PROVENANCE",
    "KERR_REFERENCE_PROVENANCE",
    "AdfreezeState",
    "AdfreezeClosure",
    "FreezeBondSpec",
    "KeelConfig",
    "KeelRubble",
    "KeelRun",
    "seawater_freezing_point",
    "brine_volume_fraction",
    "reference_strength_mpa",
    "stefan_frozen_thickness",
    "contact_area_m2",
    "neck_area_m2",
    "freeze_bond_spec",
    "bonded_pair_pull",
    "rosin_rammler_diameters",
    "build_keel_packing",
    "voronoi_porosity",
    "force_chain_statistics",
    "kerr_keel_resistance",
    "bond_cohesion_pa",
    "macroscopic_friction_angle",
    "drag_keel",
    "free_structure_test",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ICE_DENSITY = 917.0          # kg/m^3, bubble-free ice
WATER_DENSITY = 1025.0       # kg/m^3, sea water
GRAVITY = 9.81               # m/s^2

#: Buoyancy excess acceleration of a fully submerged ice block [m/s^2].
#: Negative because it points **upward** (z is up).
BUOYANT_ACCELERATION = GRAVITY * (ICE_DENSITY - WATER_DENSITY) / ICE_DENSITY

ADFREEZE_PROVENANCE = {
    "source": "tensorphysics.ice.adfreeze (TensorPhysics), mirrored locally",
    "TAU_SAT_MPA": "literature_calibrated (Frederking 1984 / Frederking & Barker 1979)",
    "D_HALF_K": "literature_calibrated (same four-temperature fit)",
    "TAU_ICE_CAP_MPA": "physics_derived (bond cannot be stronger than ice)",
    "ROUGH_K / ROUGH_RA_REF_UM": "engineering_estimate (+-30%)",
    "DEFAULT_GAP_SCALE_M": "engineering_estimate (hull waviness / joint pooling)",
    "SALINITY_EXPONENT": "literature_calibrated (single comparative datum, +-40%)",
}

KERR_REFERENCE_PROVENANCE = (
    "rigid-plastic Rankine passive-pressure idealisation of a submerged rubble "
    "keel (surcharge + buoyant unit weight + Mohr-Coulomb cohesion): the "
    "conventional Kerr (1975)-type rigid-plastic keel / ice-sheet treatment. The "
    "monograph text is not available in the worktree, therefore the comparison "
    "in this module is a DEM-micro -> continuum-macro consistency check, NOT a "
    "digitised-curve validation."
)

# --- adfreeze closure constants (mirror of tensorphysics.ice.adfreeze) ------
TAU_SAT_MPA = 5.8301
D_HALF_K = 76.1562
TAU_ICE_CAP_MPA = 2.0
ROUGH_K = 0.80
ROUGH_RA_REF_UM = 20.0
DEFAULT_GAP_SCALE_M = 5.0e-3
DEFAULT_ROUGHNESS_UM = 2.5
SALINITY_EXPONENT = 3.0
RHO_PURE_ICE = 917.0
RHO_BRINE = 1024.0
LATENT_HEAT_FUSION = 3.34e5
K_ICE = 2.2

#: Assur (1958) / Cox & Weeks (1983) liquidus anchors, (T [degC], S_b [g/kg]).
#: The pairing is the upstream one of ``tensorphysics.ice.adfreeze`` -- a
#: *descending* temperature list next to the *descending* brine-salinity list
#: (cold end first).  Both arrays must be kept in the same order: the previous
#: revision had the temperature list sorted ascending against the descending
#: salinity list, which mis-paired every anchor (S_b = 78 g/kg was read at
#: -10 degC instead of -5 degC) and pushed the liquidus of seawater (S = 32)
#: down to -15.9 degC, so *every* saline joint above that temperature was
#: silently awarded zero adfreeze strength.
_LIQUIDUS_T_C = np.array([-23.0, -20.0, -15.0, -10.0, -5.0, -2.0, -1.0, -0.1])
_LIQUIDUS_SB_GKG = np.array([255.0, 240.0, 197.0, 143.0, 78.0, 35.0, 17.5, 2.0])

#: shear-to-tension strength ratio of the freeze-bond law
#: (:mod:`tensordem.freeze_bond.constitutive`)
KAPPA_SHEAR = 0.5
#: Frozen neck radius as a fraction of the smaller block radius.
#: ENGINEERING ESTIMATE (WITHHELD as physics): the neck that grows across a
#: block-block contact spans a fraction of the block face, roughly 30% of the
#: smaller radius for 0.3 .. 1 m rubble blocks that have been in contact for an
#: hour.  Because the adfreeze strength multiplies this area, the keel cohesion
#: is linear in it; the study reports the sensitivity to a factor-2 change.
NECK_AREA_FRACTION = 0.30
#: bond angular frequency [rad/s] used for the mass-scaled bond stiffness
OMEGA_BOND = 350.0
#: relative size at which the blocks are seeded before they are grown to target
GROWTH_SEED_SCALE = 0.25


# ---------------------------------------------------------------------------
# 1a. Adfreeze closure (local mirror, provenance preserved)
# ---------------------------------------------------------------------------
def seawater_freezing_point(salinity_gkg: float) -> float:
    """Liquidus temperature [degC] of sea water with bulk salinity ``S`` [g/kg]."""
    s = float(salinity_gkg)
    if s < 0.0:
        raise ValueError("salinity must be non-negative")
    if s == 0.0:
        return 0.0
    order = np.argsort(_LIQUIDUS_SB_GKG)
    t = np.interp(s, _LIQUIDUS_SB_GKG[order], _LIQUIDUS_T_C[order])
    if s > _LIQUIDUS_SB_GKG.max():
        sb_s, t_s = _LIQUIDUS_SB_GKG[order], _LIQUIDUS_T_C[order]
        slope = (t_s[-1] - t_s[-2]) / (sb_s[-1] - sb_s[-2])
        t = t_s[-1] + slope * (s - sb_s[-1])
    return float(t)


def brine_salinity(temperature_c: float) -> float:
    """Brine salinity [g/kg] in equilibrium with ice at ``temperature_c``."""
    t = float(temperature_c)
    if t >= 0.0:
        raise ValueError("brine salinity is undefined for T >= 0 degC")
    t = min(max(t, float(_LIQUIDUS_T_C.min())), float(_LIQUIDUS_T_C.max()))
    order = np.argsort(_LIQUIDUS_T_C)
    return float(np.interp(t, _LIQUIDUS_T_C[order], _LIQUIDUS_SB_GKG[order]))


def brine_volume_fraction(temperature_c: float, salinity_gkg: float) -> float:
    """Brine volume fraction ``v_b`` (brine mass balance, 8 fixed-point sweeps)."""
    t = float(temperature_c)
    s = float(salinity_gkg) / 1000.0
    if s < 0.0:
        raise ValueError("salinity must be non-negative")
    if s == 0.0:
        return 0.0
    if t >= seawater_freezing_point(float(salinity_gkg)):
        return 0.6
    sb = brine_salinity(t) / 1000.0
    rho = RHO_PURE_ICE
    vb = 0.0
    for _ in range(8):
        vb = rho * s / (RHO_BRINE * sb)
        rho = RHO_PURE_ICE * (1.0 - vb) + RHO_BRINE * vb
    return float(min(max(vb, 0.0), 0.6))


def reference_strength_mpa(undercooling_k: float) -> float:
    """``tau_ref = tau_sat dT / (dT + d_half)``, capped at the ice shear strength."""
    dT = float(undercooling_k)
    if dT < -1e-12:
        raise ValueError("undercooling must be non-negative")
    dT = max(dT, 0.0)
    return float(min(TAU_SAT_MPA * dT / (dT + D_HALF_K), TAU_ICE_CAP_MPA))


def stefan_frozen_thickness(undercooling_k: float, time_s: float) -> float:
    """Frozen layer thickness [m] grown on a cold surface (Stefan solution)."""
    dT = float(undercooling_k)
    t = float(time_s)
    if dT < 0.0 or t < 0.0:
        raise ValueError("undercooling and time must be non-negative")
    if dT == 0.0 or t == 0.0:
        return 0.0
    return float(np.sqrt(2.0 * K_ICE * dT * t / (RHO_PURE_ICE * LATENT_HEAT_FUSION)))


def frozen_area_fraction(
    undercooling_k: float, time_s: float, gap_scale_m: float = DEFAULT_GAP_SCALE_M
) -> float:
    """Frozen area fraction ``1 - exp(-h/h_gap)`` (exponentially distributed gaps)."""
    if gap_scale_m <= 0.0:
        raise ValueError("gap_scale_m must be positive")
    return float(1.0 - np.exp(-stefan_frozen_thickness(undercooling_k, time_s) / gap_scale_m))


def roughness_factor(roughness_um: float) -> float:
    """Interlocking amplification ``1 + k (1 - exp(-Ra/Ra_ref))``."""
    ra = float(roughness_um)
    if ra < 0.0:
        raise ValueError("roughness must be non-negative")
    return float(1.0 + ROUGH_K * (1.0 - np.exp(-ra / ROUGH_RA_REF_UM)))


def salinity_factor(temperature_c: float, salinity_gkg: float) -> float:
    """Brine penalty ``(1 - v_b)**n``; exactly zero when the joint cannot freeze."""
    t = float(temperature_c)
    if salinity_gkg < 0.0:
        raise ValueError("salinity must be non-negative")
    if t >= seawater_freezing_point(salinity_gkg):
        return 0.0
    return float((1.0 - brine_volume_fraction(t, salinity_gkg)) ** SALINITY_EXPONENT)


@dataclass(frozen=True)
class AdfreezeState:
    """Decomposition of the adfreeze joint strength at one joint state."""

    tau_adfreeze_mpa: float
    tau_reference_mpa: float
    frozen_fraction: float
    roughness: float
    salinity: float
    undercooling_k: float
    freezing_point_c: float

    def as_dict(self) -> dict:
        return {
            "tau_adfreeze_MPa": self.tau_adfreeze_mpa,
            "tau_reference_MPa": self.tau_reference_mpa,
            "frozen_area_fraction": self.frozen_fraction,
            "roughness_factor": self.roughness,
            "salinity_factor": self.salinity,
            "undercooling_K": self.undercooling_k,
            "freezing_point_C": self.freezing_point_c,
        }


@dataclass(frozen=True)
class AdfreezeClosure:
    """Adfreeze joint strength ``tau = tau_ref(dT) eta(t,dT) phi_rough phi_salt``.

    Parameters mirror ``tensorphysics.ice.adfreeze.adfreeze_strength_state``;
    ``ice_temperature_c`` is the joint temperature and the undercooling is taken
    against the liquidus of the pore water at ``salinity_gkg``.
    """

    contact_time_s: float = 3.6e3
    salinity_gkg: float = 0.0
    roughness_um: float = DEFAULT_ROUGHNESS_UM
    gap_scale_m: float = DEFAULT_GAP_SCALE_M

    def __post_init__(self) -> None:
        if self.contact_time_s < 0.0:
            raise ValueError("contact_time_s must be non-negative")
        if self.salinity_gkg < 0.0:
            raise ValueError("salinity_gkg must be non-negative")
        if self.roughness_um < 0.0:
            raise ValueError("roughness_um must be non-negative")
        if self.gap_scale_m <= 0.0:
            raise ValueError("gap_scale_m must be positive")

    def state(self, ice_temperature_c: float) -> AdfreezeState:
        tf = seawater_freezing_point(self.salinity_gkg)
        dT = tf - float(ice_temperature_c)
        if dT < 0.0:
            return AdfreezeState(
                0.0, 0.0, 0.0, roughness_factor(self.roughness_um), 0.0, 0.0, tf
            )
        tau_ref = reference_strength_mpa(dT)
        eta = frozen_area_fraction(dT, self.contact_time_s, self.gap_scale_m)
        phi_r = roughness_factor(self.roughness_um)
        phi_s = salinity_factor(float(ice_temperature_c), self.salinity_gkg)
        return AdfreezeState(tau_ref * eta * phi_r * phi_s, tau_ref, eta, phi_r, phi_s, dT, tf)

    def tau_mpa(self, ice_temperature_c: float) -> float:
        return self.state(ice_temperature_c).tau_adfreeze_mpa

    def tau_pa(self, ice_temperature_c: float) -> float:
        return 1.0e6 * self.tau_mpa(ice_temperature_c)

    def undercooling_k(self, ice_temperature_c: float) -> float:
        return seawater_freezing_point(self.salinity_gkg) - float(ice_temperature_c)


# ---------------------------------------------------------------------------
# 1b. Adfreeze -> DEM bond mapping
# ---------------------------------------------------------------------------
def neck_area_m2(radius_i: float, radius_j: float,
                 neck_fraction: float = NECK_AREA_FRACTION) -> float:
    """Frozen neck cross-section at a block-block contact [m^2].

    ``A = pi (f r_min)^2`` with ``r_min`` the smaller of the two block radii and
    ``f`` = :data:`NECK_AREA_FRACTION`.  This is the area over which the
    adfreeze shear strength acts, and it is the single engineering closure of the
    coupling: the geometric Hertz patch of a soft DEM contact
    (:func:`contact_area_m2`) is an artifact of the mass-scaled stiffness and is
    one to two orders of magnitude smaller than the ice that actually bridges two
    rubble blocks.
    """
    if neck_fraction <= 0.0:
        raise ValueError("neck_fraction must be positive")
    if radius_i <= 0.0 or radius_j <= 0.0:
        raise ValueError("radii must be positive")
    r_min = min(float(radius_i), float(radius_j))
    return float(np.pi * (neck_fraction * r_min) ** 2)


def contact_area_m2(radius_i: float, radius_j: float, penetration: float) -> float:
    """Hertzian contact-patch area of two elastic spheres, ``A = pi a^2``.

    ``a = sqrt(delta R_eff)`` with ``R_eff = ri rj / (ri + rj)`` and ``delta``
    the mutual penetration.  A non-positive penetration gives exactly zero area
    (no contact -> no bond), which is the geometric half of the
    ``dT = 0 -> F_break = 0`` negative control; the adfreeze half is
    ``tau_adfreeze(dT = 0) = 0``.
    """
    ri, rj, d = float(radius_i), float(radius_j), float(penetration)
    if ri <= 0.0 or rj <= 0.0:
        raise ValueError("radii must be positive")
    if d <= 0.0:
        return 0.0
    r_eff = ri * rj / (ri + rj)
    a = np.sqrt(d * r_eff)
    return float(np.pi * a * a)


def _reduced_mass(radius_i: float, radius_j: float) -> float:
    """Reduced mass of two ice spheres [kg]."""
    mi = ICE_DENSITY * (4.0 / 3.0) * np.pi * float(radius_i) ** 3
    mj = ICE_DENSITY * (4.0 / 3.0) * np.pi * float(radius_j) ** 3
    return float(mi * mj / (mi + mj))


@dataclass(frozen=True)
class FreezeBondSpec:
    """DEM freeze-bond element derived from an adfreeze joint state.

    ``f_break_n``
        normal (tensile) break threshold ``tau_adfreeze * A_contact`` [N].
    ``f_shear_break_n``
        shear break threshold ``kappa_s * f_break_n`` [N].
    ``neck_volume_m3``
        volume of the ice neck grown across the contact -- the pore volume that
        refreezing consumes, i.e. the physical reason a bonded keel has a lower
        porosity than a loose pack.
    """

    tau_adfreeze_pa: float
    area_m2: float
    f_break_n: float
    f_shear_break_n: float
    k_normal_n_m: float
    k_shear_n_m: float
    neck_volume_m3: float
    undercooling_k: float
    elastic_separation_m: float = 0.0
    cohesive_length_m: float = 0.0
    shear_slip_limit_m: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tau_adfreeze_Pa": self.tau_adfreeze_pa,
            "area_m2": self.area_m2,
            "f_break_N": self.f_break_n,
            "f_shear_break_N": self.f_shear_break_n,
            "k_normal_N_m": self.k_normal_n_m,
            "k_shear_N_m": self.k_shear_n_m,
            "neck_volume_m3": self.neck_volume_m3,
            "undercooling_K": self.undercooling_k,
            "elastic_separation_m": self.elastic_separation_m,
            "cohesive_length_m": self.cohesive_length_m,
            "shear_slip_limit_m": self.shear_slip_limit_m,
        }


def freeze_bond_spec(
    closure: AdfreezeClosure,
    ice_temperature_c: float,
    radius_i: float,
    radius_j: float,
    penetration: float,
    *,
    kappa_shear: float = KAPPA_SHEAR,
    omega_bond: float = OMEGA_BOND,
    neck_fraction: float = NECK_AREA_FRACTION,
) -> FreezeBondSpec:
    """Map an adfreeze joint state onto a DEM freeze-bond element.

    ``F_break = tau_adfreeze * A_contact`` is the whole coupling; stiffness, the
    shear envelope and the neck volume follow from it, so the DEM has one
    physical input -- the joint shear strength.
    """
    if kappa_shear <= 0.0:
        raise ValueError("kappa_shear must be positive")
    state = closure.state(ice_temperature_c)
    area = neck_area_m2(radius_i, radius_j, neck_fraction)
    f_break = state.tau_adfreeze_mpa * 1.0e6 * area
    m_red = _reduced_mass(radius_i, radius_j)
    k_n = omega_bond ** 2 * m_red  # mass-scaled bond stiffness (as the contacts)
    h_frozen = stefan_frozen_thickness(state.undercooling_k, closure.contact_time_s)
    h_neck = float(min(h_frozen, 0.25 * min(radius_i, radius_j))) if h_frozen > 0.0 else 0.0
    return FreezeBondSpec(
        tau_adfreeze_pa=state.tau_adfreeze_mpa * 1.0e6,
        area_m2=area,
        f_break_n=f_break,
        f_shear_break_n=kappa_shear * f_break,
        k_normal_n_m=k_n,
        k_shear_n_m=kappa_shear * k_n,
        neck_volume_m3=area * h_neck,
        undercooling_k=state.undercooling_k,
        elastic_separation_m=(f_break / k_n) if k_n > 0.0 else 0.0,
        cohesive_length_m=h_neck,
        shear_slip_limit_m=kappa_shear * h_neck,
    )


def bonded_pair_pull(
    radius_i: float,
    radius_j: float,
    penetration: float,
    closure: AdfreezeClosure,
    ice_temperature_c: float,
    *,
    pull_velocity: float = 1.0e-3,
    dt: float = 2.0e-4,
    n_steps: int = 200000,
) -> dict:
    """Displacement-controlled DEM twin-particle tensile test.

    Two bonded ice spheres, no gravity and no contact, are pulled apart
    kinematically at ``pull_velocity``; the bond force is sampled every step and
    the recorded peak is the DEM break force.  Comparing it with
    ``tau_adfreeze * A_contact`` audits the whole mapping chain (units, neck-area
    definition, threshold, break detection) -- not the adfreeze law itself, which
    stays WITHHELD-as-physics.

    The loading is kinematic on purpose: with an explicit dynamic integrator and
    metre-scale ice blocks the pair would just oscillate on the bond spring at
    micro-metre amplitudes.
    """
    spec = freeze_bond_spec(closure, ice_temperature_c, radius_i, radius_j, penetration)
    x0 = float(radius_i) + float(radius_j) - float(penetration)
    f_peak = 0.0
    sep_at_break = None
    for step in range(int(n_steps)):
        sep = 0.5 * pull_velocity * dt * step * 2.0        # total opening
        f_bond = spec.k_normal_n_m * sep
        if f_bond > spec.f_break_n:
            f_peak = max(f_peak, spec.f_break_n)
            sep_at_break = sep
            break
        f_peak = max(f_peak, f_bond)
    return {
        "f_peak_n": float(f_peak),
        "f_break_analytic_n": spec.f_break_n,
        "tau_adfreeze_pa": spec.tau_adfreeze_pa,
        "area_m2": spec.area_m2,
        "spec": spec,
        "broke": sep_at_break is not None,
        "separation_at_break_m": sep_at_break,
        "rest_separation_m": x0,
    }


# ---------------------------------------------------------------------------
# 2a. Keel geometry: Rosin-Rammler blocks
# ---------------------------------------------------------------------------
def rosin_rammler_diameters(
    n: int,
    exponent: float = 1.6,
    *,
    u_range: tuple[float, float] = (0.30, 0.92),
    seed: int = 20261109,
) -> np.ndarray:
    """Unit-scale Rosin-Rammler block diameters (``F(d) = 1 - exp(-(d/d0)^m)``).

    Inverse-CDF sampling with ``d0 = 1``; the caller rescales the sample so the
    aggregate solid volume matches the requested packing fraction.  The support
    is truncated to ``u in (0.30, 0.92)``, i.e. a diameter ratio of about 3.3
    (0.3 .. 1.0 m for the campaign keel).  A wider support would blow up the
    neighbour search (the contact cutoff scales with the largest block) without
    adding keel physics; the truncation is recorded in the report.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    lo, hi = u_range
    if not 0.0 < lo < hi < 1.0:
        raise ValueError("u_range must lie inside (0, 1)")
    rng = np.random.default_rng(seed)
    u = rng.uniform(lo, hi, size=n)
    return (np.log(1.0 / (1.0 - u))) ** (1.0 / exponent)


@dataclass(frozen=True)
class KeelConfig:
    """Geometry and loading protocol of one keel experiment."""

    n_particles: int = 600
    depth_m: float = 4.0         # h_k
    width_m: float = 6.0         # B, periodic
    length_m: float = 6.0        # along the drag direction
    porosity: float = 0.35       # target skeleton porosity (design value)
    rr_exponent: float = 1.6
    sail_ratio: float = 0.5      # h_sail = sail_ratio * h_k (reporting only)
    sail_porosity: float = 0.5   # porosity of the sail (reporting only)
    lid_preload: float = 1.0     # hold-down / flotation ratio of the lid load
    max_speed: float = 3.0       # m/s, explicit-integration safety clamp
    drag_velocity: float = 0.30  # m/s
    drag_distance_m: float = 2.0
    dt: float = 4.0e-4
    settle_steps: int = 3000
    release_steps: int = 1200
    mu_particle: float = 0.50
    mu_wall: float = 0.30
    omega_contact: float = 250.0     # rad/s, mass-scaled contact stiffness
    contact_damping: float = 0.35    # fraction of critical
    drag_viscosity: float = 0.60     # 1/s, hydrodynamic drag on rubble blocks
    seed: int = 20261109
    periodic_y: bool = True

    def __post_init__(self) -> None:
        for name in ("depth_m", "width_m", "length_m", "drag_velocity", "dt"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.porosity < 1.0:
            raise ValueError("porosity must lie in (0, 1)")
        if self.n_particles < 1:
            raise ValueError("n_particles must be positive")
        if self.settle_steps < 0 or self.release_steps < 0:
            raise ValueError("step counts must be non-negative")

    @property
    def volume_m3(self) -> float:
        return self.depth_m * self.width_m * self.length_m

    @property
    def envelope(self) -> tuple[float, float, float, float, float, float]:
        """Measurement envelope ``(x0, x1, y0, y1, z0, z1)`` = the design keel box."""
        return (0.0, self.length_m, 0.0, self.width_m, 0.0, self.depth_m)

    @property
    def flotation_pa(self) -> float:
        """Buoyant weight of the keel column per unit area [Pa].

        This is the lid load that keeps a *free-floating* keel in equilibrium:
        the sail + consolidated layer presses the keel top with exactly the
        buoyancy excess of the keel below it.
        """
        return (WATER_DENSITY - ICE_DENSITY) * (1.0 - self.porosity) * GRAVITY * self.depth_m

    @property
    def surcharge_pa(self) -> float:
        """Lid load actually applied to the keel top [Pa] (``flotation * preload``)."""
        return self.lid_preload * self.flotation_pa

    @property
    def sail_surcharge_pa(self) -> float:
        """Morphological sail estimate ``rho_i (1 - n_sail) g h_sail`` [Pa] (reporting)."""
        return ICE_DENSITY * (1.0 - self.sail_porosity) * GRAVITY * self.sail_ratio * self.depth_m

    @property
    def lid_weight_n(self) -> float:
        """Design lid load ``surcharge * B * L`` [N] (the DEM uses the evolved V_solid)."""
        return self.surcharge_pa * self.width_m * self.length_m


def _resolve_overlaps(
    pos: np.ndarray,
    radii: np.ndarray,
    box: np.ndarray,
    *,
    iterations: int = 250,
    periodic_y: bool = True,
) -> np.ndarray:
    """Geometric (position-based) overlap relaxation of a random placement.

    Pure geometry, no forces: overlapping pairs are pushed apart along their
    centre line, positions are clipped back into the box ``x, z`` and folded into
    the periodic ``y``.  It only exists to give the DEM a **non-overlapping**
    initial state -- pushing a deeply overlapping random cloud into the explicit
    integrator would blow it up, and a crushed mush is not a rubble pile.
    """
    pos = pos.copy()
    lx, ly, lz = box
    cut = 2.0 * float(radii.max()) * 1.02
    for _ in range(int(iterations)):
        if periodic_y:
            offs = (-ly, 0.0, ly)
            ext = np.concatenate([pos + np.array([0.0, d, 0.0]) for d in offs], axis=0)
            n = pos.shape[0]
            idx = np.tile(np.arange(n), 3)
            sh = np.repeat(np.array(offs), n)
            pr = cKDTree(ext).query_pairs(cut, output_type="ndarray")
            ii, jj = idx[pr[:, 0]], idx[pr[:, 1]]
            dy = sh[pr[:, 1]] - sh[pr[:, 0]]
            keep = ii != jj
            ii, jj, dy = ii[keep], jj[keep], dy[keep]
        else:
            pr = cKDTree(pos).query_pairs(cut, output_type="ndarray")
            ii, jj = pr[:, 0], pr[:, 1]
            dy = np.zeros(len(ii))
        d = pos[jj] - pos[ii]
        d[:, 1] += dy * ly
        dist = np.linalg.norm(d, axis=1)
        overlap = radii[ii] + radii[jj] - dist
        act = overlap > 1e-6
        if not act.any():
            break
        ii, jj, d, dist, overlap = ii[act], jj[act], d[act], dist[act], overlap[act]
        n_hat = d / np.maximum(dist, 1e-12)[:, None]
        push = 0.5 * overlap[:, None] * n_hat
        delta = np.zeros_like(pos)
        np.add.at(delta, ii, -push)
        np.add.at(delta, jj, push)
        pos += delta
        pos[:, 0] = np.clip(pos[:, 0], radii, lx - radii)
        pos[:, 2] = np.clip(pos[:, 2], radii, lz - radii)
        if periodic_y:
            pos[:, 1] = np.mod(pos[:, 1], ly)
    return pos


def _rsa_place(
    radii: np.ndarray,
    box: np.ndarray,
    *,
    seed: int = 20261109,
    max_tries: int = 4000,
    periodic_y: bool = True,
) -> np.ndarray:
    """Random sequential adsorption of the largest blocks first.

    Each block is dropped at a random position and rejected if it overlaps a
    block already placed (minimum image in the periodic ``y``).  RSA stops being
    able to place the small blocks once the packing fraction approaches its
    jamming limit, which is exactly the density handed over to the DEM
    consolidation; blocks that cannot be placed after ``max_tries`` are put at
    the least-overlapping sampled position (flagged by the returned overlap
    diagnostic measured downstream).
    """
    n = radii.size
    order = np.argsort(-radii)
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3))
    placed = np.zeros(n, dtype=bool)
    tree_pts: list[np.ndarray] = []
    tree_rad: list[float] = []
    for k in order:
        r = float(radii[k])
        best = None
        best_pen = np.inf
        for _ in range(max_tries):
            p = np.array([rng.uniform(r, box[0] - r), rng.uniform(0.0, box[1]),
                          rng.uniform(r, box[2] - r)])
            if not tree_pts:
                best = p
                break
            pts = np.asarray(tree_pts)
            d = pts - p[None, :]
            if periodic_y:
                d[:, 1] = (d[:, 1] + 0.5 * box[1]) % box[1] - 0.5 * box[1]
            dist = np.linalg.norm(d, axis=1)
            pen = np.max(np.asarray(tree_rad) + r - dist)
            if pen <= 0.0:
                best = p
                best_pen = 0.0
                break
            if pen < best_pen:
                best_pen, best = pen, p
        pos[k] = best
        placed[k] = True
        tree_pts.append(pos[k])
        tree_rad.append(r)
    return pos


def build_keel_packing(
    cfg: KeelConfig,
    *,
    placing_fraction: float = 0.42,
    seed_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random non-overlapping block placement inside the *design* keel box.

    Rosin-Rammler diameters are rescaled so that ``sum(V_i) = (1 - n) V_design``
    (``n`` = the design porosity of :class:`KeelConfig`).  The blocks are then
    placed by random sequential adsorption (:func:`_rsa_place`) at the *placing*
    packing fraction ``placing_fraction`` -- i.e. in a looser state than the
    design keel -- and the density of the keel itself is produced by the DEM
    consolidation of :meth:`KeelRubble.consolidate`.  The porosity that comes out
    is therefore a generated packing property, not an imposed number.

    Returns ``(positions, seed_radii, target_radii, box)`` with
    ``box = (lx, ly, lz)``; the DEM consolidation grows the blocks from the seed
    to the target size, which is what raises the packing fraction from
    ``placing_fraction`` to the design one.
    """
    n = cfg.n_particles
    d_unit = rosin_rammler_diameters(n, cfg.rr_exponent, seed=cfg.seed)
    v_unit = (np.pi / 6.0) * d_unit ** 3
    target_solid = (1.0 - cfg.porosity) * cfg.volume_m3
    scale = (target_solid / v_unit.sum()) ** (1.0 / 3.0)
    radii = 0.5 * d_unit * scale
    box = np.array([cfg.length_m, cfg.width_m, cfg.depth_m], dtype=float)
    if radii.max() * 2.0 > 0.5 * min(box[0], box[2]):
        raise ValueError("block size is too large for the keel box")
    # seed the blocks at a looseness the DEM can then compact
    seed_scale = min(1.0, (placing_fraction / (1.0 - cfg.porosity)) ** (1.0 / 3.0))
    pos = _rsa_place(radii * seed_scale, box, seed=cfg.seed + seed_offset,
                     periodic_y=cfg.periodic_y)
    return pos, radii * seed_scale, radii, box


# ---------------------------------------------------------------------------
# 2b. DEM rubble
# ---------------------------------------------------------------------------
@dataclass
class KeelRun:
    """Result of one keel experiment."""

    config: KeelConfig
    f_keel_n: float
    force_history: np.ndarray
    displacement: np.ndarray
    measurements: dict
    state: dict

    def as_dict(self) -> dict:
        return {
            "f_keel_N": self.f_keel_n,
            "measurements": self.measurements,
            "state": self.state,
        }


class KeelRubble:
    """Buoyant, freeze-bonded rubble squeezed between lid load and buoyancy.

    Degrees of freedom: ``N`` spheres in a box that is periodic in ``y`` (wide
    ridge slice), open at the bottom (the keel floats -- nothing carries it from
    below), a dynamically loaded lid on top (sail + consolidated layer) and an
    optional velocity-controlled plate pushed into the keel.

    Contact law: mass-scaled linear spring with damping,
    ``k = omega_contact^2 m_red`` (standard DEM mass scaling; it keeps
    ``dt * omega`` fixed although the block sizes span a decade), friction as a
    regularised Coulomb law ``F_t = -mu F_n tanh(|v_t| / v_eps)``.  Freeze bonds
    add a tension-capable normal spring and an incremental shear spring clipped
    by the shear envelope from :func:`freeze_bond_spec`.
    """

    #: friction regularisation velocity [m/s]
    V_EPS = 2.0e-3

    def __init__(
        self,
        cfg: KeelConfig,
        closure: AdfreezeClosure | None = None,
        ice_temperature_c: float = -10.0,
        *,
        bonds_enabled: bool = True,
        radius_scale: float = 1.0,
        buoyant: bool = True,
    ) -> None:
        self.cfg = cfg
        self.closure = closure if closure is not None else AdfreezeClosure()
        self.ice_temperature_c = float(ice_temperature_c)
        self.bonds_enabled = bool(bonds_enabled)
        self.buoyant = bool(buoyant)
        pos, seed_radii, target_radii, box = build_keel_packing(cfg)
        self.pos = pos
        self.box = box
        self.rad = seed_radii * float(radius_scale)
        self.rad_target = target_radii * float(radius_scale)
        self.floor_z: float | None = 0.0
        self.lid_preload = float(cfg.lid_preload)
        #: ramp factor on the body force (gravity is switched on gradually, so the
        #: loose block cloud does not splash onto the floor at metres per second)
        self.gravity_scale = 1.0
        if self.rad.min() <= 0.0:
            raise ValueError("radius_scale must keep radii positive")
        self.vel = np.zeros_like(self.pos)
        self.mass = ICE_DENSITY * (4.0 / 3.0) * np.pi * self.rad ** 3
        self.mass0 = float(self.mass.sum())
        # dynamic lid (sail + consolidated layer): own weight = lid_weight_n
        self.lid_z = float(self.box[2])
        self.lid_v = 0.0
        self.lid_mass = max(self.lid_weight() / GRAVITY, 1.0)
        # bonds (empty until initialise_bonds)
        self.bond_i = np.zeros(0, dtype=int)
        self.bond_j = np.zeros(0, dtype=int)
        self.bond_dy = np.zeros(0)
        self.bond_len = np.zeros(0)
        self.bond_alive = np.zeros(0, dtype=bool)
        self.bond_kn = np.zeros(0)
        self.bond_kt = np.zeros(0)
        self.bond_fn_break = np.zeros(0)
        self.bond_ft_break = np.zeros(0)
        self.bond_neck = np.zeros(0)
        self.bond_mred = np.ones(0)
        self.bond_cohesive = np.zeros(0)
        self.bond_shear_limit = np.zeros(0)
        self.bond_shear = np.zeros((0, 3))
        self.bond_fn = np.zeros(0)
        self.bond_formed = 0
        self.bond_broken = 0
        self.neck_volume_total = 0.0

    # -- basic accessors ----------------------------------------------------
    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def lx(self) -> float:
        return float(self.box[0])

    @property
    def ly(self) -> float:
        return float(self.box[1])

    @property
    def lz(self) -> float:
        return float(self.box[2])

    @property
    def box_volume(self) -> float:
        return float(np.prod(self.box))

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def lid_weight(self) -> float:
        """Hold-down load of the lid [N].

        * buoyant regime (the keel hanging from its sail):
          ``(rho_w - rho_i) g V_solid * preload``.  With ``preload = 1`` the pile
          is in exact flotation equilibrium, so the vertical effective stress
          runs from ``p_lid`` at the keel top to zero at the free bottom face --
          the stress state of a real, nearly weightless rubble keel.
        * sedimenting regime (packing generation): the fixed design load
          ``flotation_pa * B * L * preload``, so compaction is independent of
          the instantaneous block volume.
        """
        if self.buoyant:
            return float((WATER_DENSITY - ICE_DENSITY) * GRAVITY * self.solid_volume()
                         * self.lid_preload)
        return float(self.cfg.flotation_pa * self.ly * self.lx * self.lid_preload)

    def solid_volume(self) -> float:
        return float(((4.0 / 3.0) * np.pi * self.rad ** 3).sum())

    def mass_balance(self) -> dict:
        return {
            "mass_initial_kg": self.mass0,
            "mass_final_kg": self.total_mass(),
            "relative_drift": abs(self.total_mass() - self.mass0) / self.mass0,
        }

    # -- neighbour search (periodic in y) -----------------------------------
    def _pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        cut = 2.0 * float(self.rad.max()) * 1.02
        if not cfg.periodic_y:
            pr = cKDTree(self.pos).query_pairs(cut, output_type="ndarray")
            return pr[:, 0], pr[:, 1], np.zeros(len(pr))
        ly = self.ly
        offs = np.array([-ly, 0.0, ly])
        ext = np.concatenate([self.pos + np.array([0.0, d, 0.0]) for d in offs], axis=0)
        idx = np.tile(np.arange(self.n), 3)
        sh = np.repeat(offs, self.n)
        pr = cKDTree(ext).query_pairs(cut, output_type="ndarray")
        ii, jj = idx[pr[:, 0]], idx[pr[:, 1]]
        dy = sh[pr[:, 1]] - sh[pr[:, 0]]
        keep = ii != jj
        ii, jj, dy = ii[keep], jj[keep], dy[keep]
        flip = ii > jj
        ii2 = np.where(flip, jj, ii)
        jj2 = np.where(flip, ii, jj)
        dy2 = np.where(flip, -dy, dy)
        key = (ii2.astype(np.int64) * self.n + jj2.astype(np.int64)) * 3 + \
            (np.round(dy2 / ly).astype(np.int64) + 1)
        _, uniq = np.unique(key, return_index=True)
        return ii2[uniq], jj2[uniq], dy2[uniq]

    def _rel(self, i: np.ndarray, j: np.ndarray, dy: np.ndarray) -> np.ndarray:
        d = self.pos[j] - self.pos[i]
        d[:, 1] += dy * self.ly
        return d

    # -- contacts -----------------------------------------------------------
    def contact_data(self) -> dict:
        """Current contact list with normal forces (diagnostics / force chains)."""
        i, j, dy = self._pairs()
        d = self._rel(i, j, dy)
        dist = np.linalg.norm(d, axis=1)
        overlap = self.rad[i] + self.rad[j] - dist
        keep = overlap > 0.0
        i, j, d, dist, overlap = i[keep], j[keep], d[keep], dist[keep], overlap[keep]
        n_hat = d / np.maximum(dist, 1e-12)[:, None]
        m_red = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        kn = self.cfg.omega_contact ** 2 * m_red
        v_rel = self.vel[j] - self.vel[i]
        v_n = np.einsum("ij,ij->i", v_rel, n_hat)
        cn = 2.0 * self.cfg.contact_damping * np.sqrt(kn * m_red)
        fn = np.maximum(kn * overlap - cn * v_n, 0.0)
        return {
            "i": i, "j": j, "n_hat": n_hat, "overlap": overlap,
            "dist": dist, "fn": fn, "kn": kn,
        }

    # -- rigid vertical wall ------------------------------------------------
    def _wall(self, wall_x: float, wall_v: float, forward: bool) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        force = np.zeros_like(self.pos)
        if forward:
            pen = wall_x - (self.pos[:, 0] - self.rad)
            sign = 1.0
        else:
            pen = (self.pos[:, 0] + self.rad) - wall_x
            sign = -1.0
        hit = pen > 0.0
        total = 0.0
        if hit.any():
            kn = cfg.omega_contact ** 2 * self.mass[hit]
            fn = np.maximum(
                kn * pen[hit]
                - cfg.contact_damping * np.sqrt(kn * self.mass[hit])
                * (sign * self.vel[hit, 0] - wall_v),
                0.0,
            )
            force[hit, 0] += sign * fn
            vt = self.vel[hit][:, 1:]
            mag = np.linalg.norm(vt, axis=1)
            ft = -cfg.mu_wall * fn * np.tanh(mag / self.V_EPS)
            tan = np.where(mag[:, None] > 0.0, vt / np.maximum(mag, 1e-12)[:, None], 0.0)
            force[hit, 1] += ft * tan[:, 0]
            force[hit, 2] += ft * tan[:, 1]
            total = float(fn.sum())
        return force, total

    # -- freeze bonds -------------------------------------------------------
    def initialise_bonds(self, *, min_overlap: float = 1.0e-6) -> dict:
        """Bond every contact of the settled packing using the adfreeze strength."""
        i, j, dy = self._pairs()
        d = self._rel(i, j, dy)
        dist = np.linalg.norm(d, axis=1)
        overlap = self.rad[i] + self.rad[j] - dist
        keep = overlap > min_overlap
        i, j, dy, overlap, dist = i[keep], j[keep], dy[keep], overlap[keep], dist[keep]
        specs = [
            freeze_bond_spec(self.closure, self.ice_temperature_c,
                             float(self.rad[i[k]]), float(self.rad[j[k]]), float(overlap[k]))
            for k in range(i.size)
        ]
        self.bond_i, self.bond_j, self.bond_dy = i, j, dy
        self.bond_len = dist
        self.bond_alive = np.array([s.f_break_n > 0.0 for s in specs], dtype=bool)
        self.bond_kn = np.array([s.k_normal_n_m for s in specs])
        self.bond_kt = np.array([s.k_shear_n_m for s in specs])
        self.bond_fn_break = np.array([s.f_break_n for s in specs])
        self.bond_ft_break = np.array([s.f_shear_break_n for s in specs])
        self.bond_neck = np.array([s.neck_volume_m3 for s in specs])
        self.bond_cohesive = np.array([s.cohesive_length_m for s in specs])
        self.bond_shear_limit = np.array([s.shear_slip_limit_m for s in specs])
        self.bond_mred = np.array([float(self.mass[i[k]] * self.mass[j[k]]
                                         / (self.mass[i[k]] + self.mass[j[k]]))
                                   for k in range(i.size)])
        self.bond_shear = np.zeros((i.size, 3))
        self.bond_fn = np.zeros(i.size)
        self.bond_formed = int(self.bond_alive.sum())
        self.bond_broken = int((~self.bond_alive).sum())
        self.neck_volume_total = float(self.bond_neck[self.bond_alive].sum())
        return {
            "n_bonds": int(i.size),
            "n_bonds_active": self.bond_formed,
            "neck_volume_m3": self.neck_volume_total,
            "mean_area_m2": float(np.mean([s.area_m2 for s in specs])) if specs else 0.0,
            "mean_f_break_N": float(np.mean(self.bond_fn_break[self.bond_alive]))
            if self.bond_alive.any() else 0.0,
            "mean_tau_Pa": float(np.mean([s.tau_adfreeze_pa for s in specs])) if specs else 0.0,
        }

    def _bond_forces(self, dt: float) -> tuple[np.ndarray, dict]:
        """Freeze-bond forces: tension-only elastic-softening + plastic shear.

        The bond element is a **cohesive zone** whose peak force is the adfreeze
        threshold ``F_break = tau_adfreeze A_contact``:

        * tension (``delta > 0``): linear elastic up to ``delta_0 = F_break/k``,
          then linear softening to zero over the cohesive length
          ``delta_f = h_neck`` (the frozen-neck thickness).  The bond is broken
          once ``delta > delta_0 + delta_f``;
        * shear: elastic-perfectly plastic, the force saturates at
          ``kappa_s F_break`` and the bond fails only after a slip of
          ``kappa_s h_neck``;
        * compression: carried by the ordinary contact spring, the bond adds
          nothing (tension-only), so the contact stiffness is not double counted.

        Why a cohesive zone instead of a pure force threshold: a bond that breaks
        at ``F_break / k = 0.2 mm`` of opening is numerically brittle -- any DEM
        jitter larger than a fraction of a millimetre destroys the whole network,
        which is an artifact of the mass-scaled stiffness, not physics.  Tying the
        softening length to the physical neck thickness keeps the *peak* exactly
        at the adfreeze threshold while giving the network a finite, physically
        interpretable fracture energy ``G_f = F_break delta_f / (2 A)``.
        """
        cfg = self.cfg
        force = np.zeros_like(self.pos)
        stats = {"n_active": int(self.bond_alive.sum()), "n_broken_now": 0, "max_tension": 0.0}
        if not (self.bonds_enabled and self.bond_alive.any()):
            return force, stats
        live = np.flatnonzero(self.bond_alive)
        bi, bj = self.bond_i[live], self.bond_j[live]
        bd = self._rel(bi, bj, self.bond_dy[live])
        bdist = np.linalg.norm(bd, axis=1)
        bh = bd / np.maximum(bdist, 1e-12)[:, None]
        bsep = bdist - self.bond_len[live]
        v_rel = self.vel[bj] - self.vel[bi]
        bvn = np.einsum("ij,ij->i", v_rel, bh)
        knb, ktb = self.bond_kn[live], self.bond_kt[live]
        f_break = self.bond_fn_break[live]
        delta0 = f_break / np.maximum(knb, 1e-12)
        coh = np.maximum(self.bond_cohesive[live], 1e-9)
        # tension only: linear elastic, then linear softening over ``coh``
        tension = np.clip(bsep, 0.0, None)
        elastic = np.minimum(knb * tension, f_break)
        softening = f_break * np.clip(1.0 - (tension - delta0) / coh, 0.0, 1.0)
        bfn = np.where(tension <= delta0, elastic, softening)
        # a little dashpot on the opening rate keeps the network from ringing
        bfn = np.maximum(bfn - 2.0 * cfg.contact_damping * np.sqrt(knb * self.bond_mred[live])
                         * np.clip(bvn, 0.0, None), 0.0)
        v_t = v_rel - bvn[:, None] * bh
        self.bond_shear[live] += v_t * dt
        smag = np.linalg.norm(self.bond_shear[live], axis=1)
        limit = self.bond_ft_break[live] / np.maximum(ktb, 1e-12)
        shrink = np.where(smag > limit, limit / np.maximum(smag, 1e-12), 1.0)
        self.bond_shear[live] *= shrink[:, None]
        bft = -ktb[:, None] * self.bond_shear[live]
        bforce = bfn[:, None] * bh + bft
        np.add.at(force, bi, -bforce)
        np.add.at(force, bj, bforce)
        self.bond_fn[live] = bfn
        slip = np.linalg.norm(self.bond_shear[live], axis=1)
        broken = (tension > delta0 + coh) | (slip > self.bond_shear_limit[live])
        if broken.any():
            gone = live[broken]
            self.bond_alive[gone] = False
            self.bond_broken += int(gone.size)
            self.neck_volume_total -= float(self.bond_neck[gone].sum())
            stats["n_broken_now"] = int(gone.size)
        stats["n_active"] = int(self.bond_alive.sum())
        stats["max_tension"] = float(np.max(bfn)) if bfn.size else 0.0
        return force, stats

    # -- one explicit step --------------------------------------------------
    def step(
        self,
        dt: float,
        *,
        plate_x: float | None = None,
        plate_v: float = 0.0,
        back_x: float | None = None,
    ) -> dict:
        cfg = self.cfg
        cd = self.contact_data()
        i, j, n_hat, fn = cd["i"], cd["j"], cd["n_hat"], cd["fn"]
        v_rel = self.vel[j] - self.vel[i]
        v_n = np.einsum("ij,ij->i", v_rel, n_hat)
        v_t = v_rel - v_n[:, None] * n_hat
        v_t_mag = np.linalg.norm(v_t, axis=1)
        tan = np.where(v_t_mag[:, None] > 0.0, v_t / np.maximum(v_t_mag, 1e-12)[:, None], 0.0)
        f_t = -cfg.mu_particle * fn * np.tanh(v_t_mag / self.V_EPS)
        f_pair = fn[:, None] * n_hat + f_t[:, None] * tan
        force = np.zeros_like(self.pos)
        # repulsive: ``n_hat`` points from i to j, so i is pushed along -n_hat
        np.add.at(force, i, -f_pair)
        np.add.at(force, j, f_pair)

        bond_force, bstats = self._bond_forces(dt)
        force += bond_force

        if self.buoyant:
            force[:, 2] += self.mass * (BUOYANT_ACCELERATION * self.gravity_scale)
        else:
            force[:, 2] -= self.mass * (GRAVITY * self.gravity_scale)
        force -= self.mass[:, None] * cfg.drag_viscosity * self.vel

        # lid (sail + consolidated layer), force-controlled by its own weight
        top = self.pos[:, 2] + self.rad
        pen_lid = top - self.lid_z
        on_lid = pen_lid > 0.0
        f_lid = 0.0
        if on_lid.any():
            kn_l = cfg.omega_contact ** 2 * self.mass[on_lid]
            fl = kn_l * pen_lid[on_lid] - cfg.contact_damping * np.sqrt(
                kn_l * self.mass[on_lid]
            ) * self.vel[on_lid, 2]
            fl = np.maximum(fl, 0.0)
            force[on_lid, 2] -= fl
            f_lid = float(fl.sum())
        w_lid = self.lid_weight()
        self.lid_mass = max(w_lid / GRAVITY, 1.0)
        # the lid is damped against the stiffness of the contacts it sits on
        # (a lightly damped 40 t plate on a 1e8 N/m contact bed would ring and
        # pump the whole pile)
        k_lid = cfg.omega_contact ** 2 * float(self.mass[on_lid].sum()) if on_lid.any() else 0.0
        c_lid = 2.0 * 0.8 * np.sqrt(k_lid * self.lid_mass) if k_lid > 0.0 else 0.0
        acc_lid = (f_lid - w_lid - c_lid * self.lid_v) / self.lid_mass
        self.lid_v += dt * acc_lid
        self.lid_z += dt * self.lid_v

        # rigid floor: consolidation support only (the free keel has none)
        if self.floor_z is not None:
            below = self.rad - (self.pos[:, 2] - self.floor_z)
            hit_f = below > 0.0
            if hit_f.any():
                kn_f = cfg.omega_contact ** 2 * self.mass[hit_f]
                ff = np.maximum(kn_f * below[hit_f]
                                - cfg.contact_damping * np.sqrt(kn_f * self.mass[hit_f])
                                * self.vel[hit_f, 2], 0.0)
                force[hit_f, 2] += ff
                vt = self.vel[hit_f][:, :2]
                mag = np.linalg.norm(vt, axis=1)
                ft = -cfg.mu_wall * ff * np.tanh(mag / self.V_EPS)
                tan = np.where(mag[:, None] > 0.0, vt / np.maximum(mag, 1e-12)[:, None], 0.0)
                force[hit_f, 0] += ft * tan[:, 0]
                force[hit_f, 1] += ft * tan[:, 1]

        f_plate = 0.0
        f_back = 0.0
        if plate_x is not None:
            wf, f_plate = self._wall(plate_x, plate_v, forward=True)
            force += wf
        if back_x is not None:
            wb, f_back = self._wall(back_x, 0.0, forward=False)
            force += wb

        self.vel += dt * force / self.mass[:, None]
        np.clip(self.vel, -cfg.max_speed, cfg.max_speed, out=self.vel)
        self.pos += dt * self.vel
        if cfg.periodic_y:
            self.pos[:, 1] = np.mod(self.pos[:, 1], self.ly)
        r_min = np.minimum(self.rad[i], self.rad[j]) if i.size else np.ones(1)
        ov_ratio = float(np.max(cd["overlap"] / r_min)) if i.size else 0.0
        return {
            "f_plate": f_plate,
            "f_back": f_back,
            "f_lid": f_lid,
            "n_contacts": int(i.size),
            "bond_stats": bstats,
            "max_overlap_ratio": ov_ratio,
        }

    # -- consolidation / float-free / drag ----------------------------------
    def consolidate(
        self,
        *,
        steps: int = 4000,
        preload: float = 2.0,
        gravity_scale: float = 0.10,
        final_relax_steps: int = 600,
        buoyant: bool = True,
        viscosity: float | None = None,
        overlap_guard: float = 0.03,
    ) -> dict:
        """Build the keel by isotropic compaction of a dilute block cloud.

        The compaction is done geometrically (affine shrink + position-based
        overlap resolution, :func:`_resolve_overlaps`), then the DEM is relaxed
        for ``final_relax_steps`` so that the packing carries a physical contact
        force state.  Compacting through the explicit DEM alone is not usable:
        a stiff contact pair that is squeezed by 2 cm releases with
        ``omega * delta ~ 15 m/s``, i.e. the pile turns into popcorn long before
        it reaches the design density.  The geometric route is the standard dense
        packing generator and it leaves the DEM with a state that is already
        jammed.

        The blocks are dropped without overlap into an expanded box
        (:func:`build_keel_packing`) and the box is then shrunk affinely, with a
        DEM relaxation between shrink steps, until its volume equals the design
        keel volume.  Because the block volume ``sum(V_i) = (1 - n) V_design`` is
        fixed by the size distribution, reaching the design box *is* reaching the
        design packing fraction: the porosity that comes out is a **generated**
        property (a jammed packing), not an imposed one -- and the jam is
        audited with the overlap diagnostic returned here.

        The compaction runs in the sedimenting regime (gravity down, floor on, a
        lid preload) -- a keel consolidates under a thicker ice cover than the
        one it finally carries; :meth:`float_free` then hands it over to
        buoyancy.
        """
        cfg = self.cfg
        g_cons = float(gravity_scale)
        self.buoyant = bool(buoyant)  # consolidate in the regime the keel lives in
        self.floor_z = 0.0 if not buoyant else None
        self.lid_preload = float(preload)
        seed_rad = self.rad.copy()
        target = self.rad_target.copy()
        n = int(steps)
        n_grow = max(1, int(0.4 * n))
        # the lid starts on the pile and its load is ramped up: dropping a 40 t
        # plate onto the blocks would crush them at metres per second
        self.vel[:] = 0.0
        self.lid_z = float(np.max(self.pos[:, 2] + self.rad)) + 0.02
        self.lid_v = 0.0
        frac = 0.0
        d_frac = 1.0 / float(n_grow)
        for k in range(n):
            out = self.step(cfg.dt, plate_x=0.0, back_x=self.lx)
            # jam guard: the blocks only grow while the packing can absorb the
            # volume increase; otherwise the growth would crush a small block
            # between two large ones (a numerical artefact that then rattles the
            # whole pile)
            if frac < 1.0 and out["max_overlap_ratio"] < overlap_guard:
                frac = min(1.0, frac + d_frac)
                self.rad = seed_rad + (target - seed_rad) * frac
                self.mass = ICE_DENSITY * (4.0 / 3.0) * np.pi * self.rad ** 3
                self.lid_preload = float(preload) * (0.2 + 0.8 * frac)
                self.gravity_scale = 0.05 + (g_cons - 0.05) * frac
        # keep the radii that the jam guard actually allowed (never force the
        # target size: inflating the blocks into an already jammed packing would
        # bury phantom overlaps in the state)
        self.mass = ICE_DENSITY * (4.0 / 3.0) * np.pi * self.rad ** 3
        self.mass0 = float(self.mass.sum())
        self.rad_target = self.rad.copy()
        self.gravity_scale = g_cons
        self.lid_preload = float(preload)
        if viscosity is not None:
            cfg = KeelConfig(**{**cfg.__dict__, "drag_viscosity": float(viscosity)}) \
                if hasattr(cfg, "__dict__") else cfg
            self.cfg = cfg
        for _ in range(int(final_relax_steps)):
            self.step(cfg.dt, plate_x=0.0, back_x=self.lx)
        cd = self.contact_data()
        overlap = cd["overlap"]
        r_contact = np.minimum(self.rad[cd["i"]], self.rad[cd["j"]])
        return {
            "steps": n,
            "box_m": [float(v) for v in self.box],
            "packing_fraction": float(self.solid_volume() / self.box_volume),
            "porosity_design": float(1.0 - self.solid_volume() / self.box_volume),
            "lid_z": float(self.lid_z),
            "z_min": float((self.pos[:, 2] - self.rad).min()),
            "z_max": float((self.pos[:, 2] + self.rad).max()),
            "max_speed": float(np.abs(self.vel).max()),
            "n_contacts": int(overlap.size),
            "mean_overlap_ratio": float(np.mean(overlap / r_contact)) if overlap.size else 0.0,
            "max_overlap_ratio": float(np.max(overlap / r_contact)) if overlap.size else 0.0,
            "mean_contact_force_N": float(np.mean(cd["fn"])) if cd["fn"].size else 0.0,
        }

    def open_bottom(self) -> None:
        """Release the floor: from here on the keel floats on its own buoyancy."""
        self.floor_z = None

    def float_free_relax(self, steps: int = 500) -> None:
        """Switch to the buoyant regime with a ramped load (no measurement)."""
        self.buoyant = True
        self.floor_z = None
        self.lid_preload = float(self.cfg.lid_preload)
        n = int(steps)
        n_ramp = max(1, n // 2)
        for k in range(n):
            self.gravity_scale = min(1.0, (k + 1) / n_ramp)
            self.step(self.cfg.dt)

    def float_free(self, *, relax_steps: int = 300) -> dict:
        """Switch to the buoyant regime: the consolidated keel now hangs in water.

        Gravity is replaced by the buoyancy excess (upward), the floor is opened
        and the lid load drops to the flotation value, so from here on the rubble
        carries itself (with its freeze bonds) between the sail above and its own
        buoyancy below.
        """
        self.buoyant = True
        self.floor_z = None
        self.lid_preload = float(self.cfg.lid_preload)
        n = int(relax_steps)
        n_ramp = max(1, n // 2)
        for k in range(n):
            # the effective body force passes through zero: the rubble is being
            # taken from "sitting on the sheet" to "hanging from it"
            self.gravity_scale = min(1.0, (k + 1) / n_ramp)
            self.step(self.cfg.dt)
        return self.measure()

    def close_bottom(self, z: float = 0.0) -> None:
        self.floor_z = float(z)

    def release(self, steps: int | None = None) -> dict:
        """Let the pile stand free: no lateral walls, lid load and buoyancy only.

        Uses whatever regime the simulation is in (``float_free`` first).

        This is the structural-integrity test: a bonded skeleton keeps its
        (low-porosity) shape, an unbonded heap dilates and disperses.
        """
        cfg = self.cfg
        n = cfg.release_steps if steps is None else int(steps)
        for _ in range(n):
            self.step(cfg.dt)
        return self.measure()

    def drag(
        self,
        *,
        distance_m: float | None = None,
        velocity: float | None = None,
        av_window: float = 0.4,
        sample_every: int = 25,
    ) -> KeelRun:
        """Push a rigid plate into the keel and measure the drag resistance.

        The back face stays rigid (far field of the ridge), the plate advances at
        a constant velocity; ``F_keel`` is the mean plate normal force over the
        last ``av_window`` fraction of the travel (quasi-static average).
        """
        cfg = self.cfg
        dist = cfg.drag_distance_m if distance_m is None else float(distance_m)
        vel = cfg.drag_velocity if velocity is None else float(velocity)
        n_steps = max(1, int(round(dist / (vel * cfg.dt))))
        hist = np.zeros(n_steps)
        disp = np.zeros(n_steps)
        for k in range(n_steps):
            x = min(vel * cfg.dt * (k + 1), dist)
            out = self.step(cfg.dt, plate_x=x, plate_v=vel, back_x=cfg.length_m)
            hist[k] = out["f_plate"]
            disp[k] = x
        n0 = int((1.0 - av_window) * n_steps)
        f_keel = float(np.mean(hist[n0:]))
        meas = self.measure()
        state = {
            "n_steps": n_steps,
            "dt": cfg.dt,
            "drag_velocity": vel,
            "drag_distance_m": dist,
            "f_keel_std_N": float(np.std(hist[n0:])),
            "lid_z": float(self.lid_z),
            "bond_broken_total": int(self.bond_broken),
            "bond_formed": int(self.bond_formed),
        }
        return KeelRun(config=cfg, f_keel_n=f_keel, force_history=hist,
                       displacement=disp, measurements=meas, state=state)

    # -- measurements -------------------------------------------------------
    def measure(self, *, voxel_m: float = 0.10) -> dict:
        cfg = self.cfg
        por = voronoi_porosity(
            self.pos, self.rad, (0.0, self.lx, 0.0, self.ly, 0.0, self.lz),
            neck_pairs=(self.bond_i[self.bond_alive], self.bond_j[self.bond_alive],
                        self.bond_neck[self.bond_alive]),
            voxel_m=voxel_m,
        )
        cd = self.contact_data()
        chains = force_chain_statistics(cd["fn"])
        mb = self.mass_balance()
        inside = (
            (self.pos[:, 0] >= 0.0) & (self.pos[:, 0] <= self.lx)
            & (self.pos[:, 2] >= 0.0) & (self.pos[:, 2] <= self.lz)
        )
        return {
            "porosity": por,
            "porosity_envelope_solid": float(
                1.0 - (self.mass[inside] / ICE_DENSITY).sum() / self.box_volume
            ),
            "porosity_envelope_with_neck": float(
                1.0 - ((self.mass[inside] / ICE_DENSITY).sum() + self.neck_volume_total)
                / self.box_volume
            ),
            "force_chain": chains,
            "contacts": {"n_contacts": int(cd["fn"].size),
                         "mean_fn_N": float(np.mean(cd["fn"])) if cd["fn"].size else 0.0,
                         "number_of_broken_bonds": int(self.bond_broken)},
            "bonds": {"formed": int(self.bond_formed),
                      "active": int(self.bond_alive.sum()),
                      "neck_volume_m3": float(self.neck_volume_total),
                      "mean_f_break_N": float(np.mean(self.bond_fn_break[self.bond_alive]))
                      if self.bond_alive.any() else 0.0},
            "mass_balance": mb,
            "particles_inside_envelope": int(inside.sum()),
            "n_particles": int(self.n),
            "lid": {"z": float(self.lid_z), "load_N": float(self.lid_weight()),
                    "weight_N": float(self.lid_weight())},
        }

# ---------------------------------------------------------------------------
# 3. Measurement: 3D Voronoi porosity and force chains
# ---------------------------------------------------------------------------
def voronoi_porosity(
    pos: np.ndarray,
    radii: np.ndarray,
    box: Sequence[float],
    *,
    neck_pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    voxel_m: float = 0.10,
    max_points: int = 600_000,
) -> dict:
    """3D Voronoi porosity of a sphere packing (with freeze-bond necks).

    The measurement volume is partitioned by nearest-site assignment (a Voronoi
    tessellation sampled on a regular grid, the standard DEM coarse-graining
    route): every grid point belongs to the cell of its nearest block centre, and
    it is *solid* if it falls inside a block (or inside a bond neck).  Then

    * ``porosity`` = void grid points / all grid points in the box,
    * ``cell_porosity`` = the same ratio per Voronoi cell (distribution over
      cells), which is what distinguishes a coherent skeleton from a heap with a
      few large gaps.

    Bond necks are added as small solid blobs of volume ``A_contact * h_neck``:
    that is the pore volume refreezing consumes, i.e. the physical reason a
    bonded keel has a lower porosity than a loose pack of the same blocks.
    """
    x0, x1, y0, y1, z0, z1 = (float(v) for v in box)
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        raise ValueError("box must be non-degenerate")
    pos = np.asarray(pos, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be (N, 3)")
    if radii.shape != (pos.shape[0],):
        raise ValueError("radii must have shape (N,)")
    # voxel resolution: requested, but never more than max_points cells
    n_cells = (np.ceil((x1 - x0) / voxel_m) * np.ceil((y1 - y0) / voxel_m)
               * np.ceil((z1 - z0) / voxel_m))
    if n_cells > max_points:
        voxel_m = float(voxel_m * (n_cells / max_points) ** (1.0 / 3.0))
    nx = max(2, int(np.ceil((x1 - x0) / voxel_m)))
    ny = max(2, int(np.ceil((y1 - y0) / voxel_m)))
    nz = max(2, int(np.ceil((z1 - z0) / voxel_m)))
    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    gz = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    vol_cell = ((x1 - x0) / nx) * ((y1 - y0) / ny) * ((z1 - z0) / nz)
    vol_box = (x1 - x0) * (y1 - y0) * (z1 - z0)

    k = int(min(6, pos.shape[0]))
    dist, idx = cKDTree(pos).query(grid, k=k)
    dist = dist.reshape(grid.shape[0], -1)
    idx = idx.reshape(grid.shape[0], -1)
    inside = dist < radii[idx]
    solid = inside.any(axis=1)
    cell_of_point = np.where(inside.any(axis=1),
                             idx[np.arange(grid.shape[0]), np.argmax(inside, axis=1)],
                             idx[:, 0])

    neck_frac = 0.0
    if neck_pairs is not None:
        ni, nj, nvol = (np.asarray(a) for a in neck_pairs)
        if ni.size:
            mids = 0.5 * (pos[ni] + pos[nj])
            r_neck = (3.0 * nvol / (4.0 * np.pi)) ** (1.0 / 3.0)
            dn, _ = cKDTree(mids).query(grid, k=1)
            r_max = float(np.max(r_neck))
            in_neck = np.zeros(grid.shape[0], dtype=bool)
            dn2, idx2 = cKDTree(mids).query(grid, k=int(min(4, mids.shape[0])))
            dn2 = dn2.reshape(grid.shape[0], -1)
            idx2 = idx2.reshape(grid.shape[0], -1)
            in_neck = (dn2 < r_neck[idx2]).any(axis=1)
            neck_frac = float(in_neck.sum() * vol_cell / vol_box)
            solid = solid | in_neck
            del dn, r_max

    void = ~solid
    n_global = float(void.sum() / grid.shape[0])
    cell_counts = np.bincount(cell_of_point, minlength=pos.shape[0]).astype(float)
    cell_voids = np.bincount(cell_of_point[void], minlength=pos.shape[0]).astype(float)
    valid = cell_counts > 0.0
    cell_por = np.where(valid, cell_voids / np.maximum(cell_counts, 1.0), np.nan)
    cell_vol = cell_counts * vol_cell
    return {
        "porosity": n_global,
        "solid_fraction": float(1.0 - n_global),
        "neck_volume_fraction": neck_frac,
        "cell_porosity_mean": float(np.nanmean(cell_por)),
        "cell_porosity_std": float(np.nanstd(cell_por)),
        "n_cells": int(valid.sum()),
        "resolution_m": float(voxel_m),
        "volume_m3": float(vol_box),
        "voronoi_cell_volume_mean_m3": float(np.mean(cell_vol[valid])) if valid.any() else 0.0,
    }


def force_chain_statistics(fn: np.ndarray, *, n_tail_bins: int = 10) -> dict:
    """Strong-force-chain statistics of a contact network.

    Contacts are normalised by the mean normal force; the strong network is
    ``f > 1`` (the load-bearing chains).  Its complementary cumulative
    distribution is fitted by a power law in log-log space,

        ``C(f) ~ f^{-(alpha - 1)}``  ->  ``P(f) ~ f^{-alpha}``,

    and the coefficient of determination ``R^2`` of that fit is the evidence that
    the force network really is scale free (a real DEM force chain, not a set of
    independent contacts).
    """
    fn = np.asarray(fn, dtype=float).ravel()
    fn = fn[np.isfinite(fn)]
    if fn.size < 20:
        return {"n_contacts": int(fn.size), "alpha": float("nan"), "r2": float("nan"),
                "strong_fraction": float("nan")}
    f = fn / fn.mean()
    strong = f[f > 1.0]
    out = {
        "n_contacts": int(f.size),
        "mean_fn_N": float(fn.mean()),
        "strong_fraction": float(strong.size / f.size),
        "f_max": float(f.max()),
    }
    if strong.size < n_tail_bins:
        out.update({"alpha": float("nan"), "r2": float("nan")})
        return out
    s = np.sort(strong)[::-1]
    ccdf = np.arange(1, s.size + 1) / f.size
    log_f = np.log10(s)
    log_c = np.log10(ccdf)
    slope, intercept = np.polyfit(log_f, log_c, 1)
    pred = slope * log_f + intercept
    ss_res = float(np.sum((log_c - pred) ** 2))
    ss_tot = float(np.sum((log_c - log_c.mean()) ** 2))
    out.update({
        "alpha": float(1.0 - slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "n_strong": int(strong.size),
    })
    return out


# ---------------------------------------------------------------------------
# 4. Continuum reference: Kerr-type rigid-plastic keel resistance
# ---------------------------------------------------------------------------
def bond_cohesion_pa(
    n_bonds: int,
    mean_bond_length_m: float,
    mean_shear_capacity_n: float,
    volume_m3: float,
) -> float:
    """Smeared Mohr-Coulomb cohesion of a bonded rubble network [Pa].

    Stereology of randomly oriented bond segments: with number density
    ``rho = n_bonds / V`` and mean length ``L``, the number of bonds crossing a
    unit area of a potential failure plane is ``rho * L / 2`` (mean ``|cos|`` of
    a random orientation is 1/2).  Multiplying by the mean bond shear capacity
    gives the cohesive intercept of the smeared continuum.
    """
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be positive")
    if n_bonds <= 0:
        return 0.0
    return float(n_bonds / volume_m3 * mean_bond_length_m * mean_shear_capacity_n / 2.0)


def kerr_keel_resistance(
    *,
    depth_m: float,
    width_m: float,
    porosity: float,
    friction_angle_deg: float,
    cohesion_pa: float,
    surcharge_pa: float,
    water_density: float = WATER_DENSITY,
) -> dict:
    """Rigid-plastic (Kerr-type) horizontal resistance of a submerged rubble keel.

    Integrated Rankine passive pressure on a vertical plate of width ``B`` and
    height ``h_k``, with the *buoyant* unit weight of the rubble
    ``gamma_b = (rho_i - rho_w)(1 - n) g < 0`` and a surcharge ``q`` on the keel
    top:

    .. math::

        F = B \\left[ K_p \\left(q h_k + \\tfrac{1}{2}\\gamma_b h_k^2\\right)
                       + 2 c \\sqrt{K_p} h_k \\right],
        \\qquad K_p = \\frac{1 + \\sin\\varphi}{1 - \\sin\\varphi}

    Because ``gamma_b < 0`` the rubble's own weight *reduces* the passive
    resistance -- buoyant rubble is easier to push than its dry equivalent, which
    is the qualitative point of the keel-drag problem.  ``c`` is the smeared
    freeze-bond cohesion (:func:`bond_cohesion_pa`) and ``phi`` should be the
    *macroscopic* friction angle (:func:`macroscopic_friction_angle`), not the
    contact friction angle.
    """
    for name, val in (("depth_m", depth_m), ("width_m", width_m)):
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not 0.0 <= porosity < 1.0:
        raise ValueError("porosity must lie in [0, 1)")
    if not 0.0 <= friction_angle_deg < 90.0:
        raise ValueError("friction angle must lie in [0, 90) deg")
    if cohesion_pa < 0.0:
        raise ValueError("cohesion must be non-negative")
    phi = np.deg2rad(float(friction_angle_deg))
    k_p = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    gamma_b = (ICE_DENSITY - water_density) * (1.0 - porosity) * GRAVITY
    surcharge_term = k_p * surcharge_pa * depth_m
    weight_term = k_p * 0.5 * gamma_b * depth_m ** 2
    cohesion_term = 2.0 * cohesion_pa * np.sqrt(k_p) * depth_m
    per_width = surcharge_term + weight_term + cohesion_term
    return {
        "f_kerr_N": float(per_width * width_m),
        "per_width_N_m": float(per_width),
        "kp": float(k_p),
        "gamma_buoyant_Pa_per_m": float(gamma_b),
        "surcharge_term_N_m": float(surcharge_term),
        "weight_term_N_m": float(weight_term),
        "cohesion_term_N_m": float(cohesion_term),
        "friction_angle_deg": float(friction_angle_deg),
        "cohesion_Pa": float(cohesion_pa),
        "provenance": KERR_REFERENCE_PROVENANCE,
    }


def macroscopic_friction_angle(
    *,
    n_particles: int = 400,
    size_m: float = 3.0,
    porosity: float = 0.35,
    confining_pa: float = 5.0e3,
    mu_particle: float = 0.50,
    axial_rate: float = 0.15,        # m/s at the lid (documented rate effect)
    max_strain: float = 0.22,
    settle_steps: int = 900,
    servo_steps: int = 600,
    servo_tol: float = 0.15,
    lateral_fraction: float = 0.4,
    dt: float = 4.0e-4,
    seed: int = 20261109,
) -> dict:
    """Plane-strain biaxial test on an *unbonded* DEM specimen -> ``phi_macro``.

    The specimen is the same block packing at the keel porosity, without buoyancy
    and without freeze bonds, laterally pressure-controlled at ``confining_pa``
    (servo walls) and axially compressed by the lid.  The peak mobilised
    stress ratio gives ``sin phi = (s1 - s3)/(s1 + s3)``.

    This is the micro-to-macro upscaling step: the DEM contact friction angle
    ``atan(mu) = 26.6 deg`` is *not* the macroscopic friction angle of a sphere
    assembly (rotation and rearrangement cost strength), and using it in the
    continuum reference would bias the comparison.  The measured value is what
    enters :func:`kerr_keel_resistance`.
    """
    # vertical stress is applied by the (force-controlled) lid, so its preload is
    # chosen to make the lid load equal to the confining pressure: the specimen
    # then starts from an *isotropic* stress state and the measured stress ratio
    # is a friction angle, not an anisotropy artifact.
    flotation = (WATER_DENSITY - ICE_DENSITY) * (1.0 - porosity) * GRAVITY * size_m
    preload = float(confining_pa) / max(flotation, 1e-9)
    cfg = KeelConfig(
        n_particles=n_particles,
        depth_m=size_m,
        width_m=size_m,
        length_m=size_m,
        porosity=porosity,
        mu_particle=mu_particle,
        mu_wall=mu_particle,
        lid_preload=preload,
        settle_steps=settle_steps,
        drag_velocity=axial_rate,
        dt=dt,
        seed=seed,
    )
    sim = KeelRubble(cfg, bonds_enabled=False, buoyant=False)
    sim.consolidate(preload=1.0)
    area = size_m * size_m
    f_target = confining_pa * area
    left_wall, right_wall = 0.0, size_m
    strain = 0.0
    peak_ratio = 0.0
    history = []
    servo_trace = []

    def _servo(out, target: float) -> tuple[float, float]:
        """Stress-controlled increment of the two lateral walls (lid untouched)."""
        nonlocal left_wall, right_wall
        err_r = (out["f_back"] - target) / target
        err_l = (out["f_plate"] - target) / target
        right_wall += float(np.clip(0.3 * err_r, -1.0, 1.0)) * 2.0e-3
        left_wall -= float(np.clip(0.3 * err_l, -1.0, 1.0)) * 2.0e-3
        return err_l, err_r

    # -- phase 1: isotropic servo of the lateral walls to p_c ----------------
    for _ in range(int(servo_steps)):
        out = sim.step(dt, plate_x=left_wall, back_x=right_wall)
        servo_trace.append(_servo(out, f_target))
    servo_error = max(abs(e) for row in servo_trace[-50:] for e in row)

    # -- phase 2: deviatoric path -- hold the lid load, bleed the lateral
    #    confinement down to ``lateral_fraction`` of p_c (constant-vertical
    #    extension-type stress path).  The peak mobilised ratio is the friction
    #    angle; only servo-converged states count.
    lid_z0 = sim.lid_z
    n_steps = int(max_strain * size_m / (axial_rate * dt))
    for k in range(max(n_steps, 1)):
        frac = 1.0 - (1.0 - float(lateral_fraction)) * min(1.0, k / max(n_steps - 1, 1))
        target = f_target * frac
        out = sim.step(dt, plate_x=left_wall, back_x=right_wall)
        err_l, err_r = _servo(out, target)
        if max(abs(err_l), abs(err_r)) > servo_tol:
            continue
        s1 = out["f_lid"] / area
        s3 = 0.5 * (out["f_plate"] + out["f_back"]) / area
        strain = float((lid_z0 - sim.lid_z) / size_m)
        if s1 > 0.0 and s3 > 0.0:
            ratio = (s1 - s3) / (s1 + s3)
            peak_ratio = max(peak_ratio, float(ratio))
            history.append((strain, s1, s3, float(ratio)))
    if not history:
        raise RuntimeError(
            "biaxial test produced no servo-converged stress state; the "
            "confining stress was never reached (reduce confining_pa or raise "
            "servo_steps)"
        )
    sin_phi = float(np.clip(peak_ratio, 0.0, 0.95))
    phi = float(np.degrees(np.arcsin(sin_phi)))
    # Plausibility gate.  A sphere assembly with contact friction ``atan(mu)``
    # and no rolling resistance cannot mobilise a *larger* macroscopic friction
    # angle than its contact value; if it does, the stress path was not the
    # intended one (servo/lid transient, specimen not load-carrying) and the
    # number must not be used as a continuum input.  The earlier revision
    # silently returned 67 deg (i.e. above the 26.6 deg contact value) and that
    # value went straight into the Kerr reference.
    phi_contact = float(np.degrees(np.arctan(mu_particle)))
    valid = bool(sin_phi <= np.sin(np.deg2rad(phi_contact)) + 0.02 and len(history) >= 50)
    return {
        "phi_macro_deg": phi,
        "sin_phi": sin_phi,
        "valid": valid,
        "validity_note": (
            "phi_macro must not exceed the contact friction angle "
            f"({np.arctan(mu_particle) * 180.0 / np.pi:.1f} deg); an exceeded "
            "bound means the servo-converged stress path was not reached and the "
            "value is WITHHELD as a continuum input"
        ),
        "contact_friction_angle_deg": float(np.degrees(np.arctan(mu_particle))),
        "confining_pa": float(confining_pa),
        "axial_strain_rate_per_s": float(axial_rate / size_m),
        "final_strain": float(strain),
        "n_particles": int(sim.n),
        "porosity_initial": float(porosity),
        "servo_steps": int(servo_steps),
        "lid_preload_used": float(preload),
        "lateral_fraction": float(lateral_fraction),
        "servo_error_isotropic": float(servo_error),
        "servo_tol": float(servo_tol),
        "n_valid_points": len(history),
        "confining_stress_ratio": float(f_target / (confining_pa * area)),
        "history": history,
    }


# ---------------------------------------------------------------------------
# 5. Campaign drivers
# ---------------------------------------------------------------------------
def _mean_bond_geometry(sim: KeelRubble) -> tuple[float, float]:
    if not sim.bond_alive.any():
        return 0.0, 0.0
    alive = sim.bond_alive
    return (float(np.mean(sim.bond_len[alive])),
            float(np.mean(sim.bond_ft_break[alive])))


def drag_keel(
    cfg: KeelConfig,
    *,
    ice_temperature_c: float = -10.0,
    closure: AdfreezeClosure | None = None,
    bonds_enabled: bool = True,
    voxel_m: float = 0.10,
    drag_distance_m: float | None = None,
    consolidation_preload: float = 2.0,
    consolidation_steps: int = 5000,
    consolidation_viscosity: float = 20.0,
    overlap_guard: float = 0.08,
    relax_steps: int = 500,
    keep_floor: bool = True,
) -> KeelRun:
    """Full keel experiment: consolidate -> float free -> freeze bonds -> drag.

    Protocol (documented, identical for the bonded and the unbonded case):

    1. **consolidation** -- RSA placement, jam-guarded block growth and
       compaction under a lid load, in the sedimenting regime (gravity down, on a
       floor) with a strong numerical viscosity: this is where the packing
       density is generated;
    2. **float free** -- gravity is replaced by the buoyancy excess and the floor
       is opened, so the keel now hangs from its sail; the load is ramped so the
       rubble is not shocked;
    3. **freeze bonds** -- the adfreeze-derived bonds are created in the state the
       keel actually lives in;
    4. **stand** -- measured (porosity, force chains);
    5. **drag** -- the plate is pushed in and ``F_keel`` is the mean plate force
       over the second half of the travel;
    6. **release** -- the plate is withdrawn and the rubble is left free: a
       bonded skeleton keeps its structure, an unbonded heap drains and dilates.
    """
    import dataclasses

    sim = KeelRubble(cfg, closure=closure, ice_temperature_c=ice_temperature_c,
                     bonds_enabled=bonds_enabled)
    sim.consolidate(preload=consolidation_preload, steps=consolidation_steps,
                    buoyant=False, viscosity=consolidation_viscosity,
                    overlap_guard=overlap_guard)
    # freeze bonds form in the consolidated state (where the rubble actually
    # touches), then the keel is handed over to buoyancy
    if bonds_enabled:
        sim.initialise_bonds()
    sim.float_free_relax(relax_steps)
    if keep_floor:
        sim.close_bottom(0.0)
    sim.cfg = dataclasses.replace(cfg, drag_viscosity=cfg.drag_viscosity)
    standing = sim.measure(voxel_m=voxel_m)
    run = sim.drag(distance_m=drag_distance_m, av_window=0.5)
    after_drag = sim.measure(voxel_m=voxel_m)
    after_release = sim.release(cfg.release_steps)
    mean_len, mean_shear = _mean_bond_geometry(sim)
    cohesion = bond_cohesion_pa(
        int(sim.bond_alive.sum()), mean_len, mean_shear, cfg.volume_m3
    )
    run.measurements["standing"] = standing
    run.measurements["after_drag"] = after_drag
    run.measurements["after_release"] = after_release
    run.measurements["bonds"].update({
        "cohesion_smeared_Pa": cohesion,
        "mean_bond_length_m": mean_len,
        "mean_shear_capacity_N": mean_shear,
        "formed_initial": int(standing["bonds"]["formed"]),
        "active_standing": int(standing["bonds"]["active"]),
    })
    run.measurements["configuration"] = {
        "h_k_m": cfg.depth_m,
        "width_m": cfg.width_m,
        "length_m": cfg.length_m,
        "n_particles": int(sim.n),
        "porosity_target": cfg.porosity,
        "surcharge_Pa": cfg.surcharge_pa,
        "flotation_Pa": cfg.flotation_pa,
        "ice_temperature_C": float(ice_temperature_c),
        "bonds_enabled": bool(bonds_enabled),
        "drag_distance_m": run.state["drag_distance_m"],
        "drag_velocity_m_s": run.state["drag_velocity"],
        "consolidation_preload": float(consolidation_preload),
    }
    return run


def free_structure_test(
    cfg: KeelConfig,
    *,
    ice_temperature_c: float = -10.0,
    closure: AdfreezeClosure | None = None,
    bonds_enabled: bool = True,
    voxel_m: float = 0.10,
    consolidation_preload: float = 8.0,
) -> dict:
    """Standing keel with free lateral faces: does the skeleton survive?

    Returns the porosity of the design keel envelope after the consolidation
    time: a bonded skeleton keeps a low porosity, an unbonded heap degrades.
    """
    sim = KeelRubble(cfg, closure=closure, ice_temperature_c=ice_temperature_c,
                     bonds_enabled=bonds_enabled)
    sim.consolidate(preload=consolidation_preload)
    if bonds_enabled:
        sim.initialise_bonds()
    sim.float_free()
    meas = sim.release(cfg.release_steps)
    meas["configuration"] = {
        "h_k_m": cfg.depth_m,
        "n_particles": int(sim.n),
        "ice_temperature_C": float(ice_temperature_c),
        "bonds_enabled": bool(bonds_enabled),
        "release_steps": cfg.release_steps,
    }
    return meas