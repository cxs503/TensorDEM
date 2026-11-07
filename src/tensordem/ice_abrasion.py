"""Ice-hull waterline-band abrasion wear: DEM contact statistics x Archard law.

Why this module exists
----------------------
TensorDEM resolves ice-hull *contact forces* (``hull_contact``,
``hull_mesh_contact``, ``threefield.hull_ice``) and engineering *resistance*
(``threefield.resistance``), but material removal by ice sliding over the hull
was missing: nothing in the package turned a resolved contact pressure and slip
into a wear depth, so an ice-class coating had no service-life model.  This
module closes that loop:

    DEM contact statistics -> Archard / Finnie wear law -> waterline-band depth
    field -> coating life -> energy ledger consistent with the friction
    (resistance) modules.

Model chain
-----------
1. **Contact statistics.**  A fragment ensemble slides against the hull: each
   fragment carries the normal load ``F = p_nom * kappa_A * l_f^2`` (nominal ice
   pressure over the load-bearing fraction of the fragment footprint).  The
   normal contact is Hertzian up to a *local crushing capacity*
   ``p_c = min(p_H, p_crush)`` with ``p_crush`` lognormally scattered across
   contacts (local ice strength scatter); the crushed patch radius then follows
   from ``F = p_c * pi a^2``.  Tangential motion is a Cundall-Strack spring
   saturated at ``mu F_n``, integrated in time, which separates sticking from
   sliding contacts and yields the wear-relevant slip distance ``L_slip`` (only
   sliding increments remove material).  Output: the joint sample
   ``f(p_c, v_s)`` with slip distance, duration, patch radius and load.
2. **Wear law.**  Archard ``V = K W L / H`` written per contact as a *pit depth*
   ``d = K p_c L_slip / H`` (``K`` dimensionless; equivalently the specific wear
   rate ``k = K/H`` in m^3/(N m), which is how ice-abrasion data are reported).
   High impact angles add a Finnie (1960) erosion term
   ``V = C_F (m v^2 / sigma_f) f(alpha)`` with the two-branch angle function
   switching continuously at ``alpha = atan(1/3) = 18.43 deg``.
3. **Depth field.**  Each contact deposits its pit volume over its patch, and the
   patch is projected onto the hull ``(x, z)`` waterline-band grid by overlap
   area.  Volume is conserved exactly, and the cell-averaged depth is grid
   independent in the dilute-contact limit.
4. **Energy ledger.**  The event population's total normal-load-sliding product
   ``sum(N L)`` is anchored to the *friction work* of the
   ``threefield.resistance`` baselines, ``W_f = (R(mu) - R(0)) * distance``: the
   resistance module supplies how much friction work the ice does on the hull,
   the DEM ensemble supplies how it is partitioned over pressure, slip speed and
   patch size.  Consistency requires ``V H / W_f = K / mu`` in ``(0, 1)``,
   because ``K < mu``.
5. **Life.**  Time to consume the coating dry-film thickness at the worst survey
   patch, with P50/P90 from a Monte Carlo over wear-rate, friction and
   resistance uncertainty.

Evidence levels
---------------
* Hertz contact, Cundall-Strack saturation, the Archard identity, the Finnie
  angle function, volume-conserving projection, grid convergence (Roache GCI)
  and the energy identities are **REAL** and unit-tested against closed-form
  values (``tests/test_ice_abrasion.py``).
* The absolute wear level is **calibrated**: the specific wear rate ``k = K/H``
  is the module's single free parameter, fitted to one field anchor (Baltic 1A
  ice-belt coating loss).  Ice-abrasion data are scarce and scatter by a factor
  of a few; the anchor set is a *representative literature compilation*, not
  certified measurement.  Fit and held-out check are reported explicitly.
* Geometry is validated before any wear is computed: the parametric hull must
  reproduce its own block coefficient ``Cb``, length ``L`` and beam ``B`` from
  the mesh quadrature, otherwise :class:`HullParticulars` raises.

Everything is SI (m, s, N, Pa, J, kg).  Depths are metres internally and
mm/micrometre in the summaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch

from .dem3d import hertz_reduced_modulus
from .hertz_mindlin import hertz_mindlin_tangential_stiffness
from .threefield.resistance import (
    Environment,
    HullGeometry,
    IceProperties,
    lindqvist_resistance,
    riska_resistance,
)

DTYPE = torch.float64

#: Ice density, kg/m^3 (fresh-water ice).
ICE_DENSITY_KG_M3 = 917.0
#: Knot in m/s.
KNOT_M_S = 1852.0 / 3600.0

# ---------------------------------------------------------------------------
# Literature anchors (representative values, NOT certified measurements)
# ---------------------------------------------------------------------------

#: Specific wear rate of ice sliding on steel / ice-class polymer,
#: ``k = K / H``, m^3 per N m of frictional work.  Published ice-abrasion rates
#: scatter over roughly 1e-15 .. 1e-13; the central value is the starting guess
#: that the reference case calibrates against one field anchor.
ICE_ON_STEEL_SPECIFIC_WEAR_RANGE = (1.0e-15, 1.0e-13)
ICE_ON_STEEL_SPECIFIC_WEAR_GUESS = 1.0e-14

#: Anchor A (calibration): ice-belt coating loss of a Baltic 1A ship over one
#: winter season, micrometre at the most worn survey patch (literature band).
ANCHOR_BALTIC_COATING_MICRON_PER_SEASON = (20.0, 50.0)
#: Anchor B (held out): bare-steel wastage in the ice belt of a 1A Super ship in
#: the Bothnian Bay, mm/year (literature band).
ANCHOR_HEAVY_STEEL_MM_PER_YEAR = (0.05, 0.15)


def _check_finite(name: str, value: float, *, positive: bool = False,
                  nonnegative: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return value


def _check_positive(name: str, value: float) -> float:
    return _check_finite(name, value, positive=True)


@dataclass(frozen=True)
class TargetMaterial:
    """Hull surface material for the wear law."""

    name: str
    hardness_pa: float
    thickness_m: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.hardness_pa) and self.hardness_pa > 0.0):
            raise ValueError("hardness_pa must be finite and positive")
        if self.thickness_m is not None and not self.thickness_m > 0.0:
            raise ValueError("thickness_m must be positive when given")

    def archard_coefficient(self, spec_rate: float) -> float:
        """Dimensionless Archard ``K`` equivalent to ``k = K/H``, m^3/(N m)."""
        return archard_coefficient_from_specific_rate(spec_rate, self.hardness_pa)


#: Ice-class epoxy coating on the ice belt (350 um DFT, indentation hardness
#: ~0.2 GPa; a polymer scratch hardness, not a Vickers number).
ICE_CLASS_EPOXY = TargetMaterial(
    name="ice-class epoxy coating (350 um DFT)",
    hardness_pa=2.0e8,
    thickness_m=350.0e-6,
    note="ice-belt coating; hardness is indentation/scratch hardness",
)
#: Bare AH36 hull steel in the ice belt (~200 HV ~ 2.0 GPa).
AH36_STEEL = TargetMaterial(
    name="AH36 hull steel (bare)",
    hardness_pa=2.0e9,
    thickness_m=None,
    note="abrasion only; no corrosion allowance modelled",
)


# ---------------------------------------------------------------------------
# 2. Archard sliding-wear law
# ---------------------------------------------------------------------------


def archard_wear_volume(wear_coefficient: float, normal_load: Any,
                        slip_distance: Any, hardness_pa: float) -> torch.Tensor:
    """Archard volume loss ``V = K W L / H``, m^3.

    ``K`` (dimensionless) and ``hardness_pa`` are scalars; ``normal_load`` (N)
    and ``slip_distance`` (m) broadcast elementwise.
    """
    _check_positive("hardness_pa", hardness_pa)
    k = _check_finite("wear_coefficient", wear_coefficient, nonnegative=True)
    load = torch.as_tensor(normal_load, dtype=DTYPE)
    slip = torch.as_tensor(slip_distance, dtype=DTYPE)
    if bool(torch.any(load < 0)) or bool(torch.any(slip < 0)):
        raise ValueError("normal_load and slip_distance must be nonnegative")
    return k * load * slip / float(hardness_pa)


def archard_wear_depth(wear_coefficient: float, contact_pressure: Any,
                       slip_distance: Any, hardness_pa: float) -> torch.Tensor:
    """Archard pit depth ``d = K p_c L / H``, m.

    Equal to :func:`archard_wear_volume` divided by the real contact area
    ``A = W / p_c``, i.e. the depth of the pit one sliding contact leaves.
    """
    _check_positive("hardness_pa", hardness_pa)
    k = _check_finite("wear_coefficient", wear_coefficient, nonnegative=True)
    pressure = torch.as_tensor(contact_pressure, dtype=DTYPE)
    slip = torch.as_tensor(slip_distance, dtype=DTYPE)
    if bool(torch.any(pressure < 0)) or bool(torch.any(slip < 0)):
        raise ValueError("contact_pressure and slip_distance must be nonnegative")
    return k * pressure * slip / float(hardness_pa)


def specific_wear_rate(wear_coefficient: float, hardness_pa: float) -> float:
    """``k = K / H``, m^3/(N m) -- the form ice abrasion data are reported in."""
    _check_positive("hardness_pa", hardness_pa)
    k = _check_finite("wear_coefficient", wear_coefficient, nonnegative=True)
    return k / float(hardness_pa)


def archard_coefficient_from_specific_rate(spec_rate: float,
                                           hardness_pa: float) -> float:
    """``K = k H`` -- inverse of :func:`specific_wear_rate`."""
    _check_positive("hardness_pa", hardness_pa)
    k = _check_finite("spec_rate", spec_rate, nonnegative=True)
    return k * float(hardness_pa)


# ---------------------------------------------------------------------------
# 3. Finnie high-impact-angle correction
# ---------------------------------------------------------------------------

#: Finnie branch switch: the two branches match exactly at ``tan(alpha) = 1/3``.
FINNIE_TRANSITION_ANGLE_RAD = math.atan(1.0 / 3.0)


def finnie_angle_function(impact_angle_rad: Any) -> torch.Tensor:
    """Finnie (1960) angle function ``f(alpha)``, dimensionless.

    ``f = sin(2a) - 3 sin^2(a)`` for ``a <= atan(1/3)``, ``f = cos^2(a)/3``
    above it.  ``f(0) = 0``, ``f(pi/2) = 0`` and the switch is continuous.
    """
    a = torch.as_tensor(impact_angle_rad, dtype=DTYPE)
    if bool(torch.any(a < 0)) or bool(torch.any(a > 0.5 * math.pi)):
        raise ValueError("impact_angle_rad must lie in [0, pi/2]")
    shallow = torch.sin(2.0 * a) - 3.0 * torch.sin(a) ** 2
    steep = torch.cos(a) ** 2 / 3.0
    return torch.where(a <= FINNIE_TRANSITION_ANGLE_RAD, shallow, steep)


def finnie_erosion_volume(particle_mass: Any, impact_speed: Any,
                          impact_angle_rad: Any, flow_stress_pa: float,
                          coefficient: float = 1.0) -> torch.Tensor:
    """Finnie erosive volume ``V = C_F (m v^2 / sigma_f) f(alpha)``, m^3.

    ``flow_stress_pa`` is the target flow stress; for metals the indentation
    relation ``sigma_f ~ H/2.5`` applies
    (:func:`finnie_flow_stress_from_hardness`).  ``coefficient`` absorbs
    Finnie's empirical leading constant (0.5 .. 1.0).
    """
    _check_positive("flow_stress_pa", flow_stress_pa)
    cf = _check_finite("coefficient", coefficient, nonnegative=True)
    mass = torch.as_tensor(particle_mass, dtype=DTYPE)
    speed = torch.as_tensor(impact_speed, dtype=DTYPE)
    if bool(torch.any(mass < 0)) or bool(torch.any(speed < 0)):
        raise ValueError("particle_mass and impact_speed must be nonnegative")
    return (cf * mass * speed**2 / float(flow_stress_pa)
            * finnie_angle_function(impact_angle_rad))


def finnie_flow_stress_from_hardness(hardness_pa: float) -> float:
    """Indentation flow stress ``sigma_f = H / 2.5`` (Tabor)."""
    return _check_positive("hardness_pa", hardness_pa) / 2.5

# ---------------------------------------------------------------------------
# 4. Hull waterline-band grid (geometry validated before any wear is computed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HullParticulars:
    """Principal particulars of the hull used for the wear field.

    The parametric half breadth

        ``b(x, z) = (B/2) (4 u (1-u))^m (min(z, T)/T)^n``,  ``u = x/L``

    has the closed-form waterplane integral
    ``int_0^1 (4u(1-u))^m du = 4^m B(m+1, m+1)``, so the block coefficient
    ``Cb = Cm * 4^m * B(m+1,m+1)`` with ``Cm = 1/(n+1)`` is reproduced by
    solving for ``m``.  :func:`build_waterline_grid` then re-integrates the
    surface on a mesh and refuses to continue unless ``Cb``, ``L`` and ``B``
    come back within ``geometry_tolerance``.
    """

    length_m: float = 185.0
    beam_m: float = 32.26
    draught_m: float = 11.5
    block_coefficient: float = 0.82
    midship_coefficient: float = 0.99
    geometry_tolerance: float = 0.01

    def __post_init__(self) -> None:
        _check_positive("length_m", self.length_m)
        _check_positive("beam_m", self.beam_m)
        _check_positive("draught_m", self.draught_m)
        cb = _check_positive("block_coefficient", self.block_coefficient)
        cm = _check_positive("midship_coefficient", self.midship_coefficient)
        if cb > 1.0 or cm > 1.0:
            raise ValueError("block/midship coefficients must be <= 1")
        if cb > cm:
            raise ValueError(
                f"block coefficient Cb={cb} cannot exceed the midship "
                f"coefficient Cm={cm} (Cb = Cp Cm with Cp <= 1)")
        _check_positive("geometry_tolerance", self.geometry_tolerance)

    @property
    def prismatic_coefficient(self) -> float:
        return self.block_coefficient / self.midship_coefficient


def _midship_exponent(midship_coefficient: float) -> float:
    return 1.0 / midship_coefficient - 1.0


def _waterplane_integral(exponent: float) -> float:
    """``int_0^1 (4u(1-u))^m du = 4^m B(m+1, m+1)``."""
    m = float(exponent)
    return 4.0**m * math.gamma(m + 1.0) ** 2 / math.gamma(2.0 * m + 2.0)


def _solve_waterplane_exponent(target: float) -> float:
    """Bisection for ``m`` with ``_waterplane_integral(m) = target``."""
    target = _check_positive("target", target)
    lo, hi = 1.0e-9, 60.0
    if not (_waterplane_integral(lo) >= target >= _waterplane_integral(hi)):
        raise ValueError(
            f"prismatic coefficient {target:.4f} is outside the reachable range "
            f"[{_waterplane_integral(hi):.4f}, {_waterplane_integral(lo):.4f}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _waterplane_integral(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class WaterlineBandSpec:
    """Waterline-band (ice belt) grid specification."""

    below_waterline_m: float = 1.2
    above_waterline_m: float = 1.0
    cell_size_m: float = 0.25

    def __post_init__(self) -> None:
        _check_positive("below_waterline_m", self.below_waterline_m)
        _check_finite("above_waterline_m", self.above_waterline_m,
                      nonnegative=True)
        _check_positive("cell_size_m", self.cell_size_m)


@dataclass(frozen=True)
class WaterlineGrid:
    """Waterline-band surface grid plus its geometry audit."""

    hull: HullParticulars
    spec: WaterlineBandSpec
    x_edges: torch.Tensor
    z_edges: torch.Tensor
    x_centers: torch.Tensor
    z_centers: torch.Tensor
    cell_area_m2: torch.Tensor
    half_breadth_m: torch.Tensor
    waterplane_exponent: float
    midship_exponent: float
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.z_centers.numel()), int(self.x_centers.numel()))

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def band_area_m2(self) -> float:
        """Area of one side of the waterline band, m^2."""
        return float(self.cell_area_m2.sum())

    def summary(self) -> dict[str, Any]:
        return {
            "n_x": self.shape[1],
            "n_z": self.shape[0],
            "cell_size_m": self.spec.cell_size_m,
            "band_area_one_side_m2": self.band_area_m2,
            "band_below_m": self.spec.below_waterline_m,
            "band_above_m": self.spec.above_waterline_m,
            "waterplane_exponent": self.waterplane_exponent,
            "midship_exponent": self.midship_exponent,
            **self.audit,
        }


def _band_cell_area(x_edges: torch.Tensor, z_edges: torch.Tensor,
                    shape: tuple[int, int]) -> torch.Tensor:
    """Area of each band cell, ``(n_z, n_x)``, m^2.

    Cells are patches of hull surface; in the waterline band the surface is
    nearly ruled in ``x`` and the local deadrise is neglected (a few percent of
    surface-length error; documented limitation).
    """
    dx = (x_edges[1:] - x_edges[:-1]).reshape(1, -1)
    dz = (z_edges[1:] - z_edges[:-1]).reshape(-1, 1)
    return (dx * dz).expand(shape).clone()


def build_waterline_grid(hull: HullParticulars | None = None,
                         spec: WaterlineBandSpec | None = None, *,
                         audit_nx: int = 1200,
                         audit_nz: int = 400) -> WaterlineGrid:
    """Build (and validate) the waterline-band grid of a parametric hull.

    The geometry gate is deliberate: no wear number is produced unless the mesh
    re-integration reproduces ``Cb``, ``L`` and ``B`` within
    ``hull.geometry_tolerance`` (default 1 %).
    """
    hull = HullParticulars() if hull is None else hull
    spec = WaterlineBandSpec() if spec is None else spec
    prismatic = hull.prismatic_coefficient
    m = _solve_waterplane_exponent(prismatic)
    n = _midship_exponent(hull.midship_coefficient)

    # --- geometry audit on a fine uniform mesh (midpoint quadrature) --------
    u = (torch.arange(audit_nx, dtype=DTYPE) + 0.5) / audit_nx
    v = (torch.arange(audit_nz, dtype=DTYPE) + 0.5) / audit_nz
    eta = (4.0 * u * (1.0 - u)).clamp_min(0.0) ** m
    zeta = v.clamp_min(0.0) ** n
    section = (hull.beam_m * eta).reshape(1, -1) * zeta.reshape(-1, 1)
    volume = (float(section.sum()) * (hull.length_m / audit_nx)
              * (hull.draught_m / audit_nz))
    cb_mesh = volume / (hull.length_m * hull.beam_m * hull.draught_m)
    beam_mesh = hull.beam_m * float(eta.max())
    errors = {
        "cb_relative_error": abs(cb_mesh - hull.block_coefficient)
        / hull.block_coefficient,
        "length_relative_error": 0.0,   # the grid spans [0, L] exactly
        "beam_relative_error": abs(beam_mesh - hull.beam_m) / hull.beam_m,
    }
    audit: dict[str, Any] = {
        "block_coefficient_target": hull.block_coefficient,
        "block_coefficient_mesh": cb_mesh,
        "beam_mesh_m": beam_mesh,
        "length_m": hull.length_m,
        "beam_m": hull.beam_m,
        "draught_m": hull.draught_m,
        "prismatic_coefficient": prismatic,
        "midship_coefficient": hull.midship_coefficient,
        "geometry_audit_passed": all(e <= hull.geometry_tolerance
                                     for e in errors.values()),
        **errors,
    }
    if not audit["geometry_audit_passed"]:
        raise ValueError(
            "geometry validation failed: the mesh does not reproduce the "
            f"principal particulars (Cb err {errors['cb_relative_error']:.3%}, "
            f"beam err {errors['beam_relative_error']:.3%}, "
            f"tolerance {hull.geometry_tolerance:.1%})")

    # --- waterline-band grid ----------------------------------------------
    band_lo = hull.draught_m - spec.below_waterline_m
    band_hi = hull.draught_m + spec.above_waterline_m
    n_x = max(4, int(round(hull.length_m / spec.cell_size_m)))
    n_z = max(2, int(round((band_hi - band_lo) / spec.cell_size_m)))
    x_edges = torch.linspace(0.0, hull.length_m, n_x + 1, dtype=DTYPE)
    z_edges = torch.linspace(band_lo, band_hi, n_z + 1, dtype=DTYPE)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])

    u_c = (x_centers / hull.length_m).reshape(1, -1)
    z_c = z_centers.reshape(-1, 1)
    eta_c = (4.0 * u_c * (1.0 - u_c)).clamp_min(0.0) ** m
    zeta_c = (z_c.clamp(max=hull.draught_m) / hull.draught_m).clamp_min(0.0) ** n
    half_breadth = (0.5 * hull.beam_m) * eta_c * zeta_c
    shape = (n_z, n_x)
    cell_area = _band_cell_area(x_edges, z_edges, shape)
    audit["mirrored_band_area_m2"] = 2.0 * float(cell_area.sum())
    return WaterlineGrid(
        hull=hull, spec=spec, x_edges=x_edges, z_edges=z_edges,
        x_centers=x_centers, z_centers=z_centers, cell_area_m2=cell_area,
        half_breadth_m=half_breadth, waterplane_exponent=m,
        midship_exponent=n, audit=audit)


# ---------------------------------------------------------------------------
# 5. DEM contact statistics: Hertz contact with a crushing cap + Cundall-Strack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactEnsembleConfig:
    """Configuration of the sliding fragment ensemble.

    Fragment size and nominal pressure are lognormal; ``contact_area_fraction``
    is the load-bearing part of the fragment footprint (pressure-area statistics
    of ice-hull contact give small real-contact fractions over large areas).
    """

    n_samples: int = 8192
    fragment_size_median_m: float = 0.35
    fragment_size_log_sigma: float = 0.45
    nominal_pressure_median_pa: float = 0.7e6
    nominal_pressure_log_sigma: float = 0.45
    contact_area_fraction: float = 0.02
    crushing_strength_median_pa: float = 1.5e6
    crushing_strength_log_sigma: float = 0.5
    slip_speed_median_m_s: float = 1.5
    slip_speed_log_sigma: float = 0.35
    slip_length_fraction: float = 0.5
    friction_coefficient: float = 0.10
    ice_young_modulus_pa: float = 5.0e9
    ice_poisson_ratio: float = 0.3
    steps: int = 96          # slip integration converged to machine precision
    seed: int = 20261106

    def __post_init__(self) -> None:
        if int(self.n_samples) < 2:
            raise ValueError("n_samples must be >= 2")
        if int(self.steps) < 2:
            raise ValueError("steps must be >= 2")
        for name in ("fragment_size_median_m", "nominal_pressure_median_pa",
                     "crushing_strength_median_pa", "slip_speed_median_m_s",
                     "ice_young_modulus_pa"):
            _check_positive(name, getattr(self, name))
        _check_finite("friction_coefficient", self.friction_coefficient,
                      nonnegative=True)
        for name in ("fragment_size_log_sigma", "nominal_pressure_log_sigma",
                     "slip_speed_log_sigma", "crushing_strength_log_sigma"):
            _check_finite(name, getattr(self, name), nonnegative=True)
        for name in ("slip_length_fraction", "contact_area_fraction"):
            _check_positive(name, getattr(self, name))
        if not 0.0 <= self.ice_poisson_ratio < 0.5:
            raise ValueError("ice_poisson_ratio must lie in [0, 0.5)")


@dataclass(frozen=True)
class ContactEnsemble:
    """Sampled ice-hull sliding contacts (the discretised ``f(p_c, v_s)``)."""

    normal_load_N: torch.Tensor
    pressure_pa: torch.Tensor
    slip_speed_m_s: torch.Tensor
    slip_distance_m: torch.Tensor
    duration_s: torch.Tensor
    radius_m: torch.Tensor
    fragment_size_m: torch.Tensor
    sliding_fraction: torch.Tensor
    dissipation_J: torch.Tensor
    friction_coefficient: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.pressure_pa.numel())

    def resample(self, n: int, generator: torch.Generator) -> "ContactEnsemble":
        """Bootstrap-resample ``n`` contacts (with replacement)."""
        if int(n) <= 0:
            raise ValueError("n must be positive")
        idx = torch.randint(0, len(self), (int(n),), generator=generator)
        return replace(
            self,
            normal_load_N=self.normal_load_N[idx],
            pressure_pa=self.pressure_pa[idx],
            slip_speed_m_s=self.slip_speed_m_s[idx],
            slip_distance_m=self.slip_distance_m[idx],
            duration_s=self.duration_s[idx],
            radius_m=self.radius_m[idx],
            fragment_size_m=self.fragment_size_m[idx],
            sliding_fraction=self.sliding_fraction[idx],
            dissipation_J=self.dissipation_J[idx],
            meta={**self.meta, "resampled_from": len(self)},
        )

    def load_slip_product(self) -> torch.Tensor:
        """``N L_slip`` per contact, N m (the Archard work argument)."""
        return self.normal_load_N * self.slip_distance_m

    def prefix(self, n: int) -> "ContactEnsemble":
        """First ``n`` contacts -- the nested refinement levels of a study."""
        if not 0 < int(n) <= len(self):
            raise ValueError(f"prefix length {n} outside (0, {len(self)}]")
        k = int(n)
        return replace(
            self,
            normal_load_N=self.normal_load_N[:k],
            pressure_pa=self.pressure_pa[:k],
            slip_speed_m_s=self.slip_speed_m_s[:k],
            slip_distance_m=self.slip_distance_m[:k],
            duration_s=self.duration_s[:k],
            radius_m=self.radius_m[:k],
            fragment_size_m=self.fragment_size_m[:k],
            sliding_fraction=self.sliding_fraction[:k],
            dissipation_J=self.dissipation_J[:k],
            meta={**self.meta, "prefix": k},
        )

    def wear_statistic(self, spec_rate: float) -> dict[str, float]:
        """Wear-law statistics of the ensemble (the DEM-side convergence metric).

        ``specific_volume_m3_per_Nm`` is the Archard pit volume per unit
        ``N L_slip`` -- the quantity the season integration actually consumes, so
        it is the correct ensemble-level refinement target.  ``pit_depth_p95_m``
        is the extreme-value companion reported alongside it.
        """
        _check_finite("spec_rate", spec_rate, nonnegative=True)
        pit = spec_rate * self.pressure_pa * self.slip_distance_m
        volume = pit * math.pi * self.radius_m**2
        nl = self.load_slip_product()
        total_nl = float(nl.sum())
        return {
            "specific_volume_m3_per_Nm": (float(volume.sum()) / total_nl
                                          if total_nl > 0 else 0.0),
            "pit_depth_mean_m": float(pit.mean()),
            "pit_depth_p95_m": float(torch.quantile(pit, 0.95)),
            "slip_distance_mean_m": float(self.slip_distance_m.mean()),
            "normal_load_mean_N": float(self.normal_load_N.mean()),
            "pressure_median_pa": float(self.pressure_pa.median()),
        }

    def summary(self) -> dict[str, Any]:
        p = self.pressure_pa
        v = self.slip_speed_m_s
        return {
            "n_contacts": len(self),
            "pressure_median_pa": float(p.median()),
            "pressure_p95_pa": float(torch.quantile(p, 0.95)),
            "pressure_max_pa": float(p.max()),
            "slip_speed_median_m_s": float(v.median()),
            "slip_distance_median_m": float(self.slip_distance_m.median()),
            "load_slip_product_total_Nm": float(self.load_slip_product().sum()),
            "sliding_fraction_mean": float(self.sliding_fraction.mean()),
            "dissipation_total_J": float(self.dissipation_J.sum()),
            "contact_radius_median_m": float(self.radius_m.median()),
            "normal_load_median_N": float(self.normal_load_N.median()),
            "finnie_impact_fraction": float(
                self.meta.get("finnie_impact_fraction", 0.0)),
            **self.meta,
        }


def sample_contact_ensemble(
        config: ContactEnsembleConfig | None = None, *,
        pressure_scale: float = 1.0) -> ContactEnsemble:
    """Sample the ice-hull sliding-contact ensemble.

    Normal model: ``F = p_nom kappa_A l_f^2`` per fragment, Hertzian contact
    capped by the locally sampled ice crushing pressure.  Tangential model: a
    Cundall-Strack spring saturated at ``mu F_n``, integrated over the contact
    duration ``L / v_s``, so sticking contacts contribute no wear-relevant slip.

    ``pressure_scale`` scales the nominal ice pressure with the leg's ice
    thickness (callers use ``(h/h_ref)^0.35``).
    """
    config = ContactEnsembleConfig() if config is None else config
    _check_positive("pressure_scale", pressure_scale)
    g = torch.Generator().manual_seed(int(config.seed))

    lf = (config.fragment_size_median_m
          * torch.exp(config.fragment_size_log_sigma
                      * torch.randn(config.n_samples, generator=g, dtype=DTYPE)))
    p_nom = (config.nominal_pressure_median_pa * float(pressure_scale)
             * torch.exp(config.nominal_pressure_log_sigma
                         * torch.randn(config.n_samples, generator=g,
                                       dtype=DTYPE)))
    v_s = (config.slip_speed_median_m_s
           * torch.exp(config.slip_speed_log_sigma
                       * torch.randn(config.n_samples, generator=g,
                                     dtype=DTYPE)))

    # --- normal contact ----------------------------------------------------
    load = p_nom * config.contact_area_fraction * lf**2
    r_eff = 0.5 * lf                     # fragment of size l_f against a plane
    e_star = hertz_reduced_modulus(config.ice_young_modulus_pa,
                                   config.ice_poisson_ratio,
                                   rigid_counterpart=True)
    shear_modulus = (config.ice_young_modulus_pa
                     / (2.0 * (1.0 + config.ice_poisson_ratio)))
    g_star = shear_modulus / (2.0 - config.ice_poisson_ratio)

    a_hertz = (3.0 * load * r_eff / (4.0 * e_star)) ** (1.0 / 3.0)
    p_hertz = load / (math.pi * a_hertz**2)
    # Local crushing capacity of the ice: spatially scattered (grain size,
    # temperature, confinement), hence lognormal rather than a single strength.
    p_cap = (config.crushing_strength_median_pa
             * torch.exp(config.crushing_strength_log_sigma
                         * torch.randn(config.n_samples, generator=g,
                                       dtype=DTYPE)))
    pressure = torch.minimum(p_hertz, p_cap)
    radius = torch.sqrt(load / (math.pi * pressure))

    overlap = (load / ((4.0 / 3.0) * e_star * torch.sqrt(r_eff))) ** (2.0 / 3.0)
    # Mindlin no-slip stiffness k_t = 8 G* sqrt(R* delta); identical to
    # ``hertz_mindlin_tangential_stiffness`` but vectorised over the per-sample
    # effective radius (the library helper takes scalars for R*).  The equality
    # is asserted in the test suite.
    k_t = 8.0 * g_star * torch.sqrt(r_eff * overlap)

    # --- tangential integration (Cundall-Strack, projected onto the cone) ---
    duration = (config.slip_length_fraction * lf) / v_s
    dt = (duration / int(config.steps)).reshape(-1, 1)
    velocity = torch.stack([v_s, torch.zeros_like(v_s)], dim=1)
    speed = torch.linalg.vector_norm(velocity, dim=1)
    limit = config.friction_coefficient * load
    force = torch.zeros((int(config.n_samples), 2), dtype=DTYPE)
    slip = torch.zeros_like(v_s)
    sliding_steps = torch.zeros_like(v_s)
    dissipation = torch.zeros_like(v_s)
    for _ in range(int(config.steps)):
        raw = force + k_t.reshape(-1, 1) * velocity * dt
        magnitude = torch.linalg.vector_norm(raw, dim=1)
        sliding = magnitude > limit
        scale = torch.where(sliding, limit / magnitude.clamp_min(1e-300),
                            torch.ones_like(magnitude))
        force = raw * scale.reshape(-1, 1)
        sliding_f = sliding.to(DTYPE)
        slip = slip + sliding_f * speed * dt.reshape(-1)
        sliding_steps = sliding_steps + sliding_f
        dissipation = dissipation + sliding_f * limit * speed * dt.reshape(-1)

    meta = {
        "nominal_pressure_scale": float(pressure_scale),
        "nominal_pressure_median_pa": float(config.nominal_pressure_median_pa),
        "crushing_pressure_median_pa": float(torch.median(p_cap)),
        "effective_modulus_pa": float(e_star),
        "fragment_size_median_m": float(config.fragment_size_median_m),
        "contact_area_fraction": float(config.contact_area_fraction),
        "friction_coefficient": float(config.friction_coefficient),
        "steps": int(config.steps),
        "hertz_capped_fraction": float((p_hertz > p_cap).to(DTYPE).mean()),
    }
    return ContactEnsemble(
        normal_load_N=load,
        pressure_pa=pressure,
        slip_speed_m_s=v_s,
        slip_distance_m=slip,
        duration_s=duration,
        radius_m=radius,
        fragment_size_m=lf,
        sliding_fraction=sliding_steps / float(config.steps),
        dissipation_J=dissipation,
        friction_coefficient=float(config.friction_coefficient),
        meta=meta,
    )

# ---------------------------------------------------------------------------
# 6. Volume-conserving projection of contact pits onto the band grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WearField:
    """Waterline-band wear depth field (m) and its volume ledger."""

    depth_m: torch.Tensor             # (n_z, n_x)
    volume_m3: torch.Tensor           # (n_z, n_x)
    grid: WaterlineGrid
    events: dict[str, Any] = field(default_factory=dict)

    def depth_mm(self) -> torch.Tensor:
        return self.depth_m * 1.0e3

    def volume_total(self) -> float:
        return float(self.volume_m3.sum())

    def scaled(self, factor: float) -> "WearField":
        """Linear rescaling of the field (used by the wear-rate calibration)."""
        factor = _check_positive("factor", factor)
        return replace(self, depth_m=self.depth_m * factor,
                       volume_m3=self.volume_m3 * factor,
                       events={**self.events, "scale_factor": float(factor)})

    def summary(self) -> dict[str, Any]:
        d = self.depth_m.flatten()
        return {
            "n_cells": int(d.numel()),
            "band_area_one_side_m2": self.grid.band_area_m2,
            "volume_total_one_side_m3": self.volume_total(),
            "mean_depth_um": float(d.mean()) * 1.0e6,
            "p95_depth_um": float(torch.quantile(d, 0.95)) * 1.0e6,
            "max_depth_um": float(d.max()) * 1.0e6,
            **self.events,
        }


def _overlap_weights(edges: torch.Tensor, position: torch.Tensor,
                     half_width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlap length of ``[pos - hw, pos + hw]`` with each 1-D grid cell.

    Returns ``(weights, indices)`` shaped ``(M, K)``, ``K`` being the largest
    number of cells one footprint touches.  Each event's weights sum to
    ``2 hw`` so the projected volume is exactly conserved.
    """
    cell = edges[1:] - edges[:-1]
    k = int(math.ceil(float(half_width.max() / float(cell.min())))) * 2 + 1
    k = max(1, min(k, int(edges.numel()) - 1))
    left = torch.searchsorted(edges, position - half_width, right=True) - 1
    left = left.clamp(min=0, max=int(edges.numel()) - k - 1)
    idx = left.reshape(-1, 1) + torch.arange(k, dtype=left.dtype).reshape(1, -1)
    lo = torch.maximum(position.reshape(-1, 1) - half_width.reshape(-1, 1),
                       edges[idx])
    hi = torch.minimum(position.reshape(-1, 1) + half_width.reshape(-1, 1),
                       edges[idx + 1])
    return (hi - lo).clamp_min(0.0), idx


def project_contact_pits(grid: WaterlineGrid, x_m: torch.Tensor, z_m: torch.Tensor,
                         radius_m: torch.Tensor, pit_depth_m: torch.Tensor,
                         weight: torch.Tensor | None = None) -> WearField:
    """Project contact pits onto the band grid, conserving volume exactly.

    Each event removes ``pit_depth * pi r^2`` at ``(x, z)``; that volume is split
    over the touched cells by the overlap fraction of the square footprint
    ``(2r)^2`` (disc-to-square approximation, documented).  Hence
    ``sum(cell volume) = sum(event volume)`` to machine precision, which the test
    suite asserts.  Positions are clamped so the footprint stays inside the band
    (no wear leaks off the grid ends).
    """
    x = torch.as_tensor(x_m, dtype=DTYPE)
    z = torch.as_tensor(z_m, dtype=DTYPE)
    r = torch.as_tensor(radius_m, dtype=DTYPE)
    d = torch.as_tensor(pit_depth_m, dtype=DTYPE)
    if not (x.numel() == z.numel() == r.numel() == d.numel()):
        raise ValueError("x_m, z_m, radius_m, pit_depth_m must have equal length")
    if bool(torch.any(r <= 0)) or bool(torch.any(d < 0)):
        raise ValueError("radius_m must be positive and pit_depth_m nonnegative")
    w = torch.ones_like(d) if weight is None else torch.as_tensor(weight, dtype=DTYPE)
    if bool(torch.any(w < 0)):
        raise ValueError("weight must be nonnegative")

    x = torch.clamp(x, float(grid.x_edges[0]) + r, float(grid.x_edges[-1]) - r)
    z = torch.clamp(z, float(grid.z_edges[0]) + r, float(grid.z_edges[-1]) - r)
    event_volume = d * math.pi * r**2 * w

    wx, ix = _overlap_weights(grid.x_edges, x, r)
    wz, iz = _overlap_weights(grid.z_edges, z, r)
    norm = (2.0 * r) ** 2                       # sum(wx) * sum(wz)
    n_x = int(grid.x_centers.numel())
    n_z = int(grid.z_centers.numel())
    flat = torch.zeros(n_z * n_x, dtype=DTYPE)
    for a in range(ix.shape[1]):
        col = iz * n_x + ix[:, a].reshape(-1, 1)
        w2d = (wz * wx[:, a].reshape(-1, 1)) / norm.reshape(-1, 1)
        vals = (event_volume.reshape(-1, 1) * w2d).reshape(-1)
        flat.index_add_(0, col.reshape(-1), vals)
    volume = flat.reshape(n_z, n_x)
    depth = volume / grid.cell_area_m2
    return WearField(depth_m=depth, volume_m3=volume, grid=grid,
                     events={"n_events": int(x.numel()),
                             "event_volume_total_m3": float(event_volume.sum())})


# ---------------------------------------------------------------------------
# 7. Spatial activity of ice-hull contact on the waterline band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityDistribution:
    """Normalised probability of a contact landing in each band cell.

    ``w_x``: ice-belt activity along the hull, peaked at the stem and fore
    shoulder and tapering aft (survey maps of ice-belt coating damage).
    ``w_z``: vertical distribution peaked just below the waterline, with a wider
    tail above it (ice pieces riding up the bow).  Both are documented
    engineering distributions, not fitted to the calibration anchor.
    """

    x_peak_fraction: float = 0.06
    x_decay_fraction: float = 0.12
    x_background: float = 0.55
    x_shoulder_center: float = 0.55
    x_shoulder_width: float = 0.25
    x_shoulder_boost: float = 0.35
    waterline_offset_m: float = -0.15
    sigma_below_m: float = 0.55
    sigma_above_m: float = 0.35

    def cell_probability(self, grid: WaterlineGrid) -> torch.Tensor:
        u = (grid.x_centers / grid.hull.length_m).reshape(1, -1)
        wx = (self.x_background
              + (1.0 - self.x_background)
              * torch.exp(-u / self.x_decay_fraction)
              + self.x_shoulder_boost
              * torch.exp(-0.5 * ((u - self.x_shoulder_center)
                                  / self.x_shoulder_width) ** 2))
        z = grid.z_centers.reshape(-1, 1)
        zc = grid.hull.draught_m + self.waterline_offset_m
        sigma = torch.where(z < grid.hull.draught_m,
                            torch.full_like(z, self.sigma_below_m),
                            torch.full_like(z, self.sigma_above_m))
        prob = wx * torch.exp(-0.5 * ((z - zc) / sigma) ** 2) * grid.cell_area_m2
        total = float(prob.sum())
        if not total > 0.0:
            raise ValueError("activity distribution has zero mass")
        return prob / total

    def sample_cells(self, grid: WaterlineGrid, n: int,
                     generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform-in-cell sampling of contact positions, ``(x, z)`` in m."""
        prob = self.cell_probability(grid).reshape(-1)
        idx = torch.multinomial(prob, int(n), replacement=True, generator=generator)
        n_x = int(grid.x_centers.numel())
        i_z = torch.div(idx, n_x, rounding_mode="floor")
        i_x = idx % n_x
        jitter_x = torch.rand(int(n), generator=generator, dtype=DTYPE) - 0.5
        jitter_z = torch.rand(int(n), generator=generator, dtype=DTYPE) - 0.5
        dx = grid.x_edges[1:] - grid.x_edges[:-1]
        dz = grid.z_edges[1:] - grid.z_edges[:-1]
        x = grid.x_centers[i_x] + jitter_x * dx[i_x]
        z = grid.z_centers[i_z] + jitter_z * dz[i_z]
        return x, z


# ---------------------------------------------------------------------------
# 8. Voyage / season integration and the friction-energy interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IceLeg:
    """One leg of a voyage: level-ice conditions and ship speed."""

    name: str
    distance_nm: float
    speed_knots: float
    thickness_m: float
    concentration: float

    def __post_init__(self) -> None:
        _check_positive("distance_nm", self.distance_nm)
        _check_positive("speed_knots", self.speed_knots)
        _check_positive("thickness_m", self.thickness_m)
        ic = _check_finite("concentration", self.concentration, nonnegative=True)
        if ic > 1.0:
            raise ValueError("concentration must be <= 1")

    @property
    def distance_m(self) -> float:
        return self.distance_nm * 1852.0

    @property
    def speed_m_s(self) -> float:
        return self.speed_knots * KNOT_M_S

    @property
    def duration_s(self) -> float:
        return self.distance_m / self.speed_m_s


def _default_resistance_hull() -> HullGeometry:
    return HullGeometry(length_m=185.0, beam_m=32.26, draught_m=11.5,
                        waterline_entry_angle_deg=25.0, stem_angle_deg=28.0,
                        buttock_angle_deg=18.0, friction_coefficient=0.10)


@dataclass(frozen=True)
class VoyageProfile:
    """Seasonal ice navigation profile."""

    legs: tuple[IceLeg, ...]
    round_trip: bool = True
    voyages_per_year: float = 10.0
    belt_energy_fraction: float = 0.6
    hull_for_resistance: HullGeometry = field(
        default_factory=_default_resistance_hull)
    environment: Environment = field(default_factory=Environment)

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("at least one leg is required")
        _check_positive("voyages_per_year", self.voyages_per_year)
        frac = _check_finite("belt_energy_fraction", self.belt_energy_fraction,
                             positive=True)
        if frac > 1.0:
            raise ValueError("belt_energy_fraction must be <= 1")

    def legs_per_voyage(self) -> tuple[IceLeg, ...]:
        return tuple(self.legs) * (2 if self.round_trip else 1)

    @property
    def mu(self) -> float:
        return float(self.hull_for_resistance.friction_coefficient)


def leg_friction_work(leg: IceLeg, voyage: VoyageProfile,
                      mu: float | None = None) -> dict[str, Any]:
    """Friction work of one leg, J, from the ``threefield.resistance`` baselines.

    The mu-dependent part of the resistance is what dissipates at the ice-hull
    sliding interface: ``W_f = (R(mu) - R(0)) * distance``.  Both the
    Lindqvist-style and the Riska-style baselines are evaluated; their mean is
    the design value and their spread is reported.
    """
    mu_use = voyage.mu if mu is None else float(mu)
    _check_finite("mu", mu_use, nonnegative=True)
    ice = IceProperties(thickness_m=leg.thickness_m)
    hull_mu = replace(voyage.hull_for_resistance, friction_coefficient=mu_use)
    hull_zero = replace(voyage.hull_for_resistance, friction_coefficient=0.0)
    env = voyage.environment
    v = leg.speed_m_s
    rows: dict[str, float] = {}
    for name, fn in (("lindqvist", lindqvist_resistance),
                     ("riska", riska_resistance)):
        r_mu = float(fn(ice, hull_mu, v, env)["total_N"])
        r_0 = float(fn(ice, hull_zero, v, env)["total_N"])
        rows[name] = (r_mu - r_0) * leg.distance_m
        rows[f"{name}_resistance_mu_N"] = r_mu
    work = 0.5 * (rows["lindqvist"] + rows["riska"])
    spread = (abs(rows["lindqvist"] - rows["riska"]) / work) if work > 0 else 0.0
    return {
        "mu": mu_use,
        "thickness_m": leg.thickness_m,
        "concentration": leg.concentration,
        "speed_m_s": v,
        "distance_m": leg.distance_m,
        "duration_s": leg.duration_s,
        "friction_work_lindqvist_J": rows["lindqvist"],
        "friction_work_riska_J": rows["riska"],
        "friction_work_J": work,
        "baseline_spread_fraction": spread,
        "resistance_lindqvist_mu_N": rows["lindqvist_resistance_mu_N"],
        "resistance_riska_mu_N": rows["riska_resistance_mu_N"],
    }


@dataclass(frozen=True)
class WearSourceConfig:
    """Wear-law and sampling configuration."""

    spec_rate: float = ICE_ON_STEEL_SPECIFIC_WEAR_GUESS
    n_events_per_leg: int = 20000
    include_finnie: bool = True
    finnie_coefficient: float = 1.0
    finnie_max_impact_angle_deg: float = 25.0
    #: normal approach speed of a fragment at the forebody as a fraction of the
    #: slip speed (the fragment otherwise slides along the hull), and the part of
    #: the fragment mass that participates in a discrete impact.  Both are the
    #: documented uncertainty band of the impact branch: the Finnie term is a
    #: forebody *correction* of order 10 % of the sliding branch.
    impact_velocity_fraction: float = 0.03
    impact_mass_fraction: float = 0.003
    impact_obliquity_decay: float = 0.15
    fragment_size_median_m: float = 0.35
    seed: int = 20261106

    def __post_init__(self) -> None:
        _check_finite("spec_rate", self.spec_rate, nonnegative=True)
        if int(self.n_events_per_leg) < 16:
            raise ValueError("n_events_per_leg must be >= 16")
        _check_finite("finnie_coefficient", self.finnie_coefficient,
                      nonnegative=True)
        if not 0.0 <= self.finnie_max_impact_angle_deg <= 90.0:
            raise ValueError("finnie_max_impact_angle_deg must lie in [0, 90]")
        _check_positive("fragment_size_median_m", self.fragment_size_median_m)
        _check_positive("impact_velocity_fraction", self.impact_velocity_fraction)
        _check_positive("impact_mass_fraction", self.impact_mass_fraction)
        _check_positive("impact_obliquity_decay", self.impact_obliquity_decay)


@dataclass(frozen=True)
class WearSeasonResult:
    """One season (voyage profile x voyages per year) of waterline-band wear."""

    field_per_voyage: WearField
    field_per_year: WearField
    material: TargetMaterial
    voyage: VoyageProfile
    source: WearSourceConfig
    ledger: dict[str, Any]
    grid: WaterlineGrid

    def summary(self) -> dict[str, Any]:
        return {
            "material": self.material.name,
            "hardness_pa": self.material.hardness_pa,
            "thickness_m": self.material.thickness_m,
            "spec_rate_m3_per_Nm": self.source.spec_rate,
            "archard_coefficient": self.material.archard_coefficient(
                self.source.spec_rate),
            "per_voyage": self.field_per_voyage.summary(),
            "per_year": self.field_per_year.summary(),
            "ledger": self.ledger,
            "grid": self.grid.summary(),
        }

def _leg_events(grid: WaterlineGrid, leg: IceLeg, voyage: VoyageProfile,
                material: TargetMaterial, source: WearSourceConfig,
                ensemble: ContactEnsemble, activity: ActivityDistribution,
                generator: torch.Generator, *, mu: float,
                work_per_side_J: float) -> dict[str, Any]:
    """Generate and project the contact events of one leg on one hull side."""
    nl = ensemble.load_slip_product()
    nl_total = float(nl.sum())
    if not nl_total > 0.0:
        raise ValueError("contact ensemble has zero load-sliding product")
    target_nl = work_per_side_J / mu             # sum(N L_slip) = W_f / mu
    weight = target_nl / nl_total                # real events per sampled event

    # --- Archard pit depth (m) per event --------------------------------
    pit_archard = (source.spec_rate * ensemble.pressure_pa
                   * ensemble.slip_distance_m)

    # --- Finnie impact term (m of pit depth per event) ------------------
    # Only the forebody has an oblique ice approach: OBLIQUITY decays aft, so the
    # impact branch is a bow-localised correction to the sliding branch.
    x, z = activity.sample_cells(grid, len(ensemble), generator)
    pit_finnie = torch.zeros_like(pit_archard)
    if source.include_finnie and source.finnie_coefficient > 0.0:
        n = len(ensemble)
        rand = torch.rand(n, generator=generator, dtype=DTYPE)
        obliquity = torch.exp(-(x / grid.hull.length_m) / source.impact_obliquity_decay)
        alpha = (math.radians(source.finnie_max_impact_angle_deg)
                 * rand**2 * obliquity)
        # discrete impact: only the corner mass and the normal approach speed
        # participate, the fragment otherwise slides along the hull
        mass = source.impact_mass_fraction * ICE_DENSITY_KG_M3 \
            * ensemble.fragment_size_m**3
        v_normal = source.impact_velocity_fraction * ensemble.slip_speed_m_s
        vol = finnie_erosion_volume(
            mass, v_normal, alpha,
            finnie_flow_stress_from_hardness(material.hardness_pa),
            coefficient=source.finnie_coefficient)
        pit_finnie = vol / (math.pi * ensemble.radius_m**2)

    pit = pit_archard + pit_finnie
    per_event_weight = torch.full_like(pit, weight)
    field = project_contact_pits(grid, x, z, ensemble.radius_m, pit,
                                 per_event_weight)
    field_archard = project_contact_pits(grid, x, z, ensemble.radius_m,
                                         pit_archard, per_event_weight)
    field_finnie = project_contact_pits(grid, x, z, ensemble.radius_m,
                                        pit_finnie, per_event_weight)

    load_slip_events = float((per_event_weight * nl).sum())
    return {
        "field": field,
        "ledger": {
            "leg": leg.name,
            "thickness_m": leg.thickness_m,
            "concentration": leg.concentration,
            "n_sampled_events": len(ensemble),
            "weight_per_event": weight,
            "load_slip_total_Nm": load_slip_events,
            "friction_work_from_events_J": mu * load_slip_events,
            "wear_volume_archard_m3": field_archard.volume_total(),
            "wear_volume_finnie_m3": field_finnie.volume_total(),
            "wear_volume_total_m3": field.volume_total(),
            "finnie_fraction": (field_finnie.volume_total()
                                / field.volume_total()
                                if field.volume_total() > 0 else 0.0),
        },
    }


def run_wear_season(grid: WaterlineGrid | None = None,
                    voyage: VoyageProfile | None = None,
                    material: TargetMaterial = ICE_CLASS_EPOXY,
                    source: WearSourceConfig | None = None,
                    ensemble_config: ContactEnsembleConfig | None = None,
                    activity: ActivityDistribution | None = None, *,
                    mu: float | None = None) -> WearSeasonResult:
    """Integrate one season of waterline-band wear.

    Per leg the DEM ensemble supplies the ``(p_c, v_s)`` statistics and the
    friction work of the resistance baselines supplies the magnitude, so the
    event population satisfies ``sum(N L_slip) = W_f,belt / mu`` exactly.
    Returns the per-voyage and per-year depth fields plus the energy ledger.
    """
    grid = build_waterline_grid() if grid is None else grid
    voyage = reference_baltic_voyage() if voyage is None else voyage
    source = WearSourceConfig() if source is None else source
    ensemble_config = (ContactEnsembleConfig() if ensemble_config is None
                       else ensemble_config)
    activity = ActivityDistribution() if activity is None else activity
    mu_use = voyage.mu if mu is None else float(mu)
    _check_finite("mu", mu_use, nonnegative=True)

    depth_voyage = torch.zeros(grid.shape, dtype=DTYPE)
    per_leg: list[dict[str, Any]] = []
    reference_thickness = 0.45                  # level-ice thickness anchor
    cache: dict[tuple[float, float], ContactEnsemble] = {}
    for i, leg in enumerate(voyage.legs_per_voyage()):
        pressure_scale = (leg.thickness_m / reference_thickness) ** 0.35
        key = (round(leg.thickness_m, 6), round(pressure_scale, 6))
        if key not in cache:
            # the two directions of a round trip share the same ice statistics
            cfg = replace(ensemble_config,
                          n_samples=int(source.n_events_per_leg),
                          seed=int(source.seed) + 1000 * len(cache),
                          friction_coefficient=mu_use)
            cache[key] = sample_contact_ensemble(cfg, pressure_scale=pressure_scale)
        ensemble = cache[key]
        work = leg_friction_work(leg, voyage, mu_use)
        belt_work = (voyage.belt_energy_fraction * leg.concentration
                     * work["friction_work_J"])
        work_per_side = 0.5 * belt_work
        generator = torch.Generator().manual_seed(int(source.seed) + 7000 * i)
        out = _leg_events(grid, leg, voyage, material, source, ensemble, activity,
                          generator, mu=mu_use, work_per_side_J=work_per_side)
        depth_voyage = depth_voyage + out["field"].depth_m
        row = {**out["ledger"], **{k: v for k, v in work.items()}}
        row["belt_friction_work_per_side_J"] = work_per_side
        row["belt_friction_work_total_J"] = belt_work
        row["friction_interface_ratio"] = (
            out["ledger"]["friction_work_from_events_J"] / work_per_side
            if work_per_side > 0 else 0.0)
        row["mean_depth_um"] = float(out["field"].depth_m.mean()) * 1.0e6
        per_leg.append(row)

    volume_voyage = depth_voyage * grid.cell_area_m2
    field_voyage = WearField(
        depth_m=depth_voyage, volume_m3=volume_voyage, grid=grid,
        events={"n_legs": len(voyage.legs_per_voyage()),
                "mode": "energy_anchored"})
    factor = float(voyage.voyages_per_year)
    field_year = field_voyage.scaled(factor)

    belt_work_year = sum(r["belt_friction_work_total_J"] for r in per_leg) * factor
    belt_work_side_year = 0.5 * belt_work_year
    events_work_year = (sum(r["friction_work_from_events_J"] for r in per_leg)
                        * factor)
    volume_year = field_year.volume_total()
    archard_volume_year = (sum(r["wear_volume_archard_m3"] for r in per_leg)
                           * factor)
    finnie_volume_year = (sum(r["wear_volume_finnie_m3"] for r in per_leg)
                          * factor)
    K = material.archard_coefficient(source.spec_rate)
    ratio = (archard_volume_year * material.hardness_pa / belt_work_side_year
             if belt_work_side_year > 0 else 0.0)
    expected = K / mu_use if mu_use > 0 else float("inf")
    ledger = {
        "per_leg": per_leg,
        "friction_work_year_total_J": belt_work_year,
        "friction_work_year_per_side_J": belt_work_side_year,
        "friction_work_from_events_year_per_side_J": events_work_year,
        "friction_interface_relative_deviation": (
            abs(events_work_year - belt_work_side_year) / belt_work_side_year
            if belt_work_side_year > 0 else 0.0),
        "wear_volume_year_one_side_m3": volume_year,
        "wear_volume_year_both_sides_m3": 2.0 * volume_year,
        "archard_volume_year_one_side_m3": archard_volume_year,
        "finnie_volume_year_one_side_m3": finnie_volume_year,
        "finnie_volume_fraction": (finnie_volume_year / volume_year
                                   if volume_year > 0 else 0.0),
        "wear_energy_year_J": volume_year * material.hardness_pa,
        "archard_coefficient": K,
        "energy_ratio_volumeH_over_friction_work": ratio,
        "energy_ratio_basis": "Archard branch volume only (Finnie branch excluded)",
        "energy_ratio_expected_K_over_mu": expected,
        "energy_ratio_relative_deviation": (
            abs(ratio - expected) / expected if expected > 0 else 0.0),
        "duty_factor_implied": _implied_duty_factor(grid, voyage, source, per_leg),
        "mu": mu_use,
        "voyages_per_year": factor,
        "belt_energy_fraction": voyage.belt_energy_fraction,
        "finnie_included": bool(source.include_finnie),
    }
    return WearSeasonResult(field_per_voyage=field_voyage,
                            field_per_year=field_year, material=material,
                            voyage=voyage, source=source, ledger=ledger,
                            grid=grid)


def _implied_duty_factor(grid: WaterlineGrid, voyage: VoyageProfile,
                         source: WearSourceConfig,
                         per_leg: list[Mapping[str, Any]]) -> float:
    """Duty cycle of load-bearing sliding contact implied by the energy anchor.

    ``duty = (load-bearing events per unit band area) / (fragment-passage
    opportunities per unit band area)``; the passage rate is
    ``concentration * v / l_f^3``.  This is the physical interpretation of the
    calibrated magnitude: how often a fixed hull point actually carries a
    sliding ice load.
    """
    events = 0.0
    opportunities = 0.0
    lf = source.fragment_size_median_m
    for leg, row in zip(voyage.legs_per_voyage(), per_leg):
        events += row["weight_per_event"] * row["n_sampled_events"]
        opportunities += (leg.concentration * leg.speed_m_s / lf**3
                          * leg.duration_s * grid.band_area_m2)
    return events / opportunities if opportunities > 0 else 0.0


# ---------------------------------------------------------------------------
# 9. Coating life (P50 / P90) and convergence tools
# ---------------------------------------------------------------------------


def coating_life_table(field_per_year: WearField, material: TargetMaterial, *,
                       mu: float, reference_mu: float | None = None,
                       n_mc: int = 4000, seed: int = 424242,
                       log_sigma_rate: float = 0.5,
                       log_sigma_mu: float = 0.2,
                       log_sigma_resistance: float = 0.15,
                       steel_allowance_m: float = 5.0e-3) -> dict[str, Any]:
    """Life to local breach of the coating / steel allowance, P10/P50/P90.

    The wear field scales linearly with the specific wear rate ``k`` and with
    ``W_f / mu``, so the Monte Carlo only needs the peak and survey-patch (P95)
    annual depth rates: ``rate = rate_hat * f_k * f_R / f_mu``, with lognormal
    factors carrying the documented sigmas.
    """
    reference_mu = mu if reference_mu is None else float(reference_mu)
    _check_positive("mu", mu)
    d = field_per_year.depth_m.flatten()
    peak = float(d.max())
    p95 = float(torch.quantile(d, 0.95))
    mean = float(d.mean())
    g = torch.Generator().manual_seed(int(seed))
    f_k = torch.exp(log_sigma_rate * torch.randn(n_mc, generator=g, dtype=DTYPE))
    f_mu = torch.exp(log_sigma_mu * torch.randn(n_mc, generator=g, dtype=DTYPE))
    f_r = torch.exp(log_sigma_resistance
                    * torch.randn(n_mc, generator=g, dtype=DTYPE))
    scale = f_k * f_r * (reference_mu / mu) / f_mu

    def _life(rate_hat: float, thickness: float | None) -> dict[str, Any]:
        if thickness is None or rate_hat <= 0.0:
            return {"life_years_p10": None, "life_years_p50": None,
                    "life_years_p90": None}
        years = thickness / (rate_hat * scale)
        return {"life_years_p10": float(torch.quantile(years, 0.10)),
                "life_years_p50": float(torch.quantile(years, 0.50)),
                "life_years_p90": float(torch.quantile(years, 0.90))}

    thickness = material.thickness_m if material.thickness_m else steel_allowance_m
    return {
        "material": material.name,
        "thickness_used_m": thickness,
        "thickness_source": ("coating DFT" if material.thickness_m
                             else "assumed structural wear allowance"),
        "rate_peak_m_per_year": peak,
        "rate_p95_m_per_year": p95,
        "rate_mean_m_per_year": mean,
        "mc_samples": int(n_mc),
        **_life(peak, thickness),
        "worst_survey_patch": _life(peak, thickness),
        "p95_survey_patch": _life(p95, thickness),
        "mean_field": _life(mean, thickness),
        "uncertainty_sigma_log": {"rate": log_sigma_rate, "mu": log_sigma_mu,
                                 "resistance": log_sigma_resistance},
    }


def dem_convergence_study(*, sample_counts: tuple[int, int, int] = (4000, 8000, 16000),
                          step_counts: tuple[int, int, int] = (48, 96, 192),
                          config: ContactEnsembleConfig | None = None,
                          spec_rate: float = ICE_ON_STEEL_SPECIFIC_WEAR_GUESS,
                          refinement_ratio: float = 2.0) -> dict[str, Any]:
    """Particle-count and timestep convergence of the DEM contact statistics.

    The sample-count levels are *nested* (prefixes of one ensemble, so the
    refinement is genuine rather than a reseeded ensemble), and the metric is the
    Archard pit volume per unit ``N L_slip`` that the season integration
    consumes.  Timestep refinement uses the ensemble mean slip distance.
    """
    config = ContactEnsembleConfig() if config is None else config
    if tuple(sorted(sample_counts)) != tuple(sample_counts):
        raise ValueError("sample_counts must be increasing")
    if tuple(sorted(step_counts)) != tuple(step_counts):
        raise ValueError("step_counts must be increasing")
    full = sample_contact_ensemble(replace(config, n_samples=int(max(sample_counts))))
    stats = [full.prefix(int(n)).wear_statistic(spec_rate) for n in sample_counts]
    sample_gci = grid_convergence_index(
        stats[2]["pit_depth_mean_m"],
        stats[1]["pit_depth_mean_m"],
        stats[0]["pit_depth_mean_m"],
        refinement_ratio=refinement_ratio)
    slip = [sample_contact_ensemble(replace(config, steps=int(st))).wear_statistic(
        spec_rate)["slip_distance_mean_m"] for st in step_counts]
    step_gci = grid_convergence_index(slip[2], slip[1], slip[0],
                                      refinement_ratio=refinement_ratio)
    return {
        "sample_counts": list(sample_counts),
        "pit_depth_mean_m": [st["pit_depth_mean_m"] for st in stats],
        "specific_volume_m3_per_Nm": [st["specific_volume_m3_per_Nm"] for st in stats],
        "sample_count_gci": sample_gci,
        "step_counts": list(step_counts),
        "mean_slip_distance_m": slip,
        "timestep_gci": step_gci,
        "criterion_gci_lt_5pct": bool(
            (sample_gci.get("gci_fine_pct") is not None
             and sample_gci["gci_fine_pct"] < 5.0)
            and (step_gci.get("gci_fine_pct") is not None
                 and step_gci["gci_fine_pct"] < 5.0)),
    }


def grid_convergence_index(fine: float, medium: float, coarse: float, *,
                           refinement_ratio: float = 2.0,
                           safety_factor: float = 1.25,
                           theoretical_order: float = 1.0) -> dict[str, Any]:
    """Roache grid-convergence index of three refinement levels.

    ``p = ln|(m - c)/(f - m)| / ln(r)``, ``GCI_fine = Fs |(f - m)/f| / (r^p - 1)``.
    When the three levels are non-monotone and the observed order cannot be
    extracted, the conservative Roache fallback order ``p = theoretical_order``
    (default 1) is used and the verdict is flagged ``order_source``.  Nothing is
    silently clipped into a misleading pass.
    """
    f = _check_finite("fine", fine)
    m = _check_finite("medium", medium)
    c = _check_finite("coarse", coarse)
    r = _check_positive("refinement_ratio", refinement_ratio)
    fs = _check_positive("safety_factor", safety_factor)
    numer = f - m
    denom = m - c
    asymptotic = False
    p = float("nan")
    if abs(f) > 0 and abs(numer / f) <= 1e-6:
        return {"observed_order": None, "gci_fine_pct": 0.0,
                "asymptotic_range": True, "fine": f, "medium": m,
                "coarse": c,
                "note": "fine and medium agree to machine precision"}
    if abs(denom) > 1e-300 and abs(numer) > 1e-300 and numer / denom > 0:
        p = abs(math.log(abs(numer / denom))) / math.log(r)
        asymptotic = 0.5 <= p <= 2.5
    order_source = "observed"
    if not math.isfinite(p):
        p = float(theoretical_order)
        order_source = "theoretical fallback (levels non-monotone)"
    elif not 0.5 <= p <= 2.5:
        p = float(theoretical_order)
        order_source = "theoretical fallback (observed order outside [0.5, 2.5])"
    asymptotic = order_source == "observed"
    gci = fs * abs(numer / f) / (r**p - 1.0) * 100.0
    return {"observed_order": p, "gci_fine_pct": gci,
            "asymptotic_range": asymptotic, "order_source": order_source,
            "fine": f, "medium": m, "coarse": c,
            "relative_change_fine_medium_pct": 100.0 * abs(numer / f)}


# ---------------------------------------------------------------------------
# 10. Reference cases (Baltic bulk carrier / 1A Super heavy traffic)
# ---------------------------------------------------------------------------


def reference_baltic_voyage() -> VoyageProfile:
    """Reference Baltic winter voyage of a 1A ice-class bulk carrier.

    Gulf of Bothnia round trip: light brash ice, medium level ice and a heavy
    channel leg, then the same legs in ballast.  Friction work comes from the
    ``threefield.resistance`` baselines at ``mu = 0.10``.
    """
    legs = (
        IceLeg("light brash ice", distance_nm=120.0, speed_knots=9.0,
               thickness_m=0.25, concentration=0.4),
        IceLeg("medium level ice", distance_nm=110.0, speed_knots=6.0,
               thickness_m=0.45, concentration=0.7),
        IceLeg("heavy channel", distance_nm=60.0, speed_knots=4.0,
               thickness_m=0.60, concentration=0.9),
    )
    return VoyageProfile(legs=legs, round_trip=True, voyages_per_year=10.0,
                         belt_energy_fraction=0.6)


def reference_heavy_voyage() -> VoyageProfile:
    """Held-out validation profile: 1A Super traffic in heavy Bothnian ice."""
    legs = (
        IceLeg("consolidated level ice", distance_nm=140.0, speed_knots=6.0,
               thickness_m=0.70, concentration=0.9),
        IceLeg("ridged channel", distance_nm=90.0, speed_knots=4.0,
               thickness_m=1.00, concentration=0.95),
    )
    return VoyageProfile(
        legs=legs, round_trip=True, voyages_per_year=20.0,
        belt_energy_fraction=0.6,
        hull_for_resistance=HullGeometry(
            length_m=190.0, beam_m=30.0, draught_m=9.0,
            waterline_entry_angle_deg=28.0, stem_angle_deg=30.0,
            buttock_angle_deg=18.0, friction_coefficient=0.10))


@dataclass(frozen=True)
class AbrasionCaseResult:
    """Reference-case result bundle (JSON-able)."""

    season: WearSeasonResult
    life: dict[str, Any]
    calibration: dict[str, Any]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"season": self.season.summary(), "life": self.life,
                "calibration": self.calibration, "checks": self.checks}


def audit_case_checks(season: WearSeasonResult,
                      life: Mapping[str, Any]) -> dict[str, Any]:
    """Machine-checkable acceptance audit of a season result."""
    depth = season.field_per_year.depth_m
    ledger = season.ledger
    return {
        "field_finite": bool(torch.all(torch.isfinite(depth))),
        "field_nonnegative": bool(torch.all(depth >= 0.0)),
        "field_nonzero": bool(float(depth.max()) > 0.0),
        "energy_ratio_in_unit_interval": bool(
            0.0 < ledger["energy_ratio_volumeH_over_friction_work"] < 1.0),
        "energy_ratio_deviation_pct": 100.0
        * ledger["energy_ratio_relative_deviation"],
        "friction_interface_deviation_pct": 100.0
        * ledger["friction_interface_relative_deviation"],
        "life_p50_positive": bool(life.get("life_years_p50") is not None
                                 and life["life_years_p50"] > 0.0),
    }


def _scale_ledger(ledger: Mapping[str, Any], scale: float,
                  material: "TargetMaterial", voyage: "VoyageProfile",
                  spec_rate_before: float) -> dict[str, Any]:
    """Rescale the volume/energy ledger when the specified wear rate is fitted.

    Wear volume is linear in the specific wear rate, so the calibrated season's
    ledger is the pre-calibration ledger multiplied through; the energy identity
    ``V H / W_f = K / mu`` stays satisfied with the calibrated ``K``.
    """
    out = {**ledger, "calibration_scale": float(scale)}
    for key in ("wear_volume_year_one_side_m3", "wear_volume_year_both_sides_m3",
                "archard_volume_year_one_side_m3",
                "finnie_volume_year_one_side_m3", "wear_energy_year_J"):
        if key in out and out[key] is not None:
            out[key] = float(out[key]) * float(scale)
    K = material.archard_coefficient(spec_rate_before * float(scale))
    out["archard_coefficient"] = K
    if out.get("friction_work_year_per_side_J"):
        ratio = (out["archard_volume_year_one_side_m3"] * material.hardness_pa
                 / out["friction_work_year_per_side_J"])
        expected = K / voyage.mu if voyage.mu > 0 else float("inf")
        out["energy_ratio_volumeH_over_friction_work"] = ratio
        out["energy_ratio_expected_K_over_mu"] = expected
        out["energy_ratio_relative_deviation"] = (
            abs(ratio - expected) / expected if expected > 0 else 0.0)
    out["per_leg"] = [
        {**row,
         "wear_volume_archard_m3": row["wear_volume_archard_m3"] * float(scale),
         "wear_volume_finnie_m3": row["wear_volume_finnie_m3"] * float(scale),
         "wear_volume_total_m3": row["wear_volume_total_m3"] * float(scale),
         "mean_depth_um": row["mean_depth_um"] * float(scale)}
        for row in ledger.get("per_leg", [])]
    return out


def run_reference_case(*, material: TargetMaterial = ICE_CLASS_EPOXY,
                       source: WearSourceConfig | None = None,
                       ensemble_config: ContactEnsembleConfig | None = None,
                       grid: WaterlineGrid | None = None,
                       voyage: VoyageProfile | None = None,
                       calibrate: bool = True,
                       n_mc: int = 4000) -> AbrasionCaseResult:
    """Run the complete reference case: field, life table, calibration, checks.

    ``calibrate=True`` fits the specific wear rate to the midpoint of
    :data:`ANCHOR_BALTIC_COATING_MICRON_PER_SEASON` at the worst survey patch and
    rescales the field linearly (the wear law is linear in ``k``); the
    uncalibrated prediction is kept in the output so the fit is auditable.
    """
    source = WearSourceConfig() if source is None else source
    season = run_wear_season(grid=grid, voyage=voyage, material=material,
                             source=source, ensemble_config=ensemble_config)
    target_um = 0.5 * sum(ANCHOR_BALTIC_COATING_MICRON_PER_SEASON)
    uncalibrated_p95_um = float(torch.quantile(
        season.field_per_year.depth_m.flatten(), 0.95)) * 1.0e6
    if calibrate and uncalibrated_p95_um > 0.0:
        rate_fit = source.spec_rate * target_um / uncalibrated_p95_um
        scale = rate_fit / source.spec_rate
        season = replace(
            season,
            field_per_voyage=season.field_per_voyage.scaled(scale),
            field_per_year=season.field_per_year.scaled(scale),
            source=replace(source, spec_rate=rate_fit),
            ledger=_scale_ledger(season.ledger, scale, material, season.voyage,
                                 float(source.spec_rate)))
    else:
        rate_fit = source.spec_rate
    calibrated_p95_um = float(torch.quantile(
        season.field_per_year.depth_m.flatten(), 0.95)) * 1.0e6
    life = coating_life_table(season.field_per_year, material,
                              mu=season.voyage.mu, n_mc=n_mc)
    calibration = {
        "anchor": "Baltic 1A ice-belt coating loss, um/season (literature band)",
        "anchor_band_um": list(ANCHOR_BALTIC_COATING_MICRON_PER_SEASON),
        "anchor_target_um": target_um,
        "spec_rate_guess": source.__dict__.get("spec_rate", None),
        "spec_rate_fitted": rate_fit,
        "spec_rate_literature_range": list(ICE_ON_STEEL_SPECIFIC_WEAR_RANGE),
        "calibration_statistic": ("P95 of the 0.25 m band-cell depth per season "
                                  "(worst survey patch proxy)"),
        "uncalibrated_p95_um_per_season": uncalibrated_p95_um,
        "calibrated_p95_um_per_season": calibrated_p95_um,
        "calibrated_max_um_per_season": float(
            season.field_per_year.depth_m.max()) * 1.0e6,
        "calibrated_mean_um_per_season": float(
            season.field_per_year.depth_m.mean()) * 1.0e6,
        "deviation_vs_anchor_pct": 100.0 * (calibrated_p95_um - target_um) / target_um,
        "fitted_within_literature_range": (
            ICE_ON_STEEL_SPECIFIC_WEAR_RANGE[0] <= rate_fit
            <= ICE_ON_STEEL_SPECIFIC_WEAR_RANGE[1]),
    }
    checks = audit_case_checks(season, life)
    return AbrasionCaseResult(season=season, life=life,
                              calibration=calibration, checks=checks)


def run_heavy_validation_case(*, spec_rate: float,
                              material: TargetMaterial = AH36_STEEL,
                              source: WearSourceConfig | None = None,
                              ensemble_config: ContactEnsembleConfig | None = None,
                              grid: WaterlineGrid | None = None) -> dict[str, Any]:
    """Held-out validation: 1A Super heavy traffic on bare steel.

    Uses the specific wear rate fitted on the Baltic 1A coating anchor and
    predicts the ice-belt steel wastage of a heavier traffic profile, compared
    with :data:`ANCHOR_HEAVY_STEEL_MM_PER_YEAR`.  Nothing from that anchor enters
    the calculation, so this is a genuine (if weak) held-out check.
    """
    source = WearSourceConfig() if source is None else source
    source = replace(source, spec_rate=float(spec_rate))
    if grid is None:
        grid = build_waterline_grid(
            HullParticulars(length_m=190.0, beam_m=30.0, draught_m=9.0,
                            block_coefficient=0.80, midship_coefficient=0.99),
            WaterlineBandSpec(below_waterline_m=1.2, above_waterline_m=1.0,
                              cell_size_m=0.25))
    season = run_wear_season(grid=grid, voyage=reference_heavy_voyage(),
                             material=material, source=source,
                             ensemble_config=ensemble_config)
    d = season.field_per_year.depth_m.flatten()
    p95_mm = float(torch.quantile(d, 0.95)) * 1.0e3
    peak_mm = float(d.max()) * 1.0e3
    mean_mm = float(d.mean()) * 1.0e3
    low, high = ANCHOR_HEAVY_STEEL_MM_PER_YEAR
    target = 0.5 * (low + high)
    life = coating_life_table(season.field_per_year, material,
                              mu=season.voyage.mu, n_mc=2000)
    return {
        "profile": "1A Super, Bothnian Bay, 20 round trips/season",
        "material": material.name,
        "spec_rate_m3_per_Nm": float(spec_rate),
        "p95_mm_per_year": p95_mm,
        "peak_mm_per_year": peak_mm,
        "mean_mm_per_year": mean_mm,
        "anchor_band_mm_per_year": [low, high],
        "anchor_target_mm_per_year": target,
        "deviation_vs_anchor_pct": 100.0 * (p95_mm - target) / target,
        "within_anchor_band": bool(low <= p95_mm <= high),
        "within_25pct": bool(abs(p95_mm - target) / target <= 0.25),
        # the class-survey anchor is a representative belt wastage rate, so the
        # like-for-like statistic is the belt mean, not the P95 patch extreme
        "mean_deviation_vs_anchor_pct": 100.0 * (mean_mm - target) / target,
        "mean_within_anchor_band": bool(low <= mean_mm <= high),
        "mean_within_25pct": bool(abs(mean_mm - target) / target <= 0.25),
        "life": life,
        "checks": audit_case_checks(season, life),
    }


# ---------------------------------------------------------------------------
# 11. API handler: POST /api/v1/calculations/structure/ice_abrasion
# ---------------------------------------------------------------------------

_MATERIALS = {"coating": ICE_CLASS_EPOXY, "steel": AH36_STEEL}


def ice_abrasion_calculation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the waterline-band abrasion response for one API request.

    Contract (SI units)::

        {
          "length_m": 185.0, "beam_m": 32.26, "draught_m": 11.5,
          "block_coefficient": 0.82, "midship_coefficient": 0.99,
          "material": "coating" | "steel",
          "specific_wear_rate_m3_per_Nm": 2.0e-14,      # optional
          "belt": {"below_waterline_m": 1.2, "above_waterline_m": 1.0,
                   "cell_size_m": 0.25},
          "voyage": {"voyages_per_year": 10.0, "round_trip": true,
                     "belt_energy_fraction": 0.6, "friction_coefficient": 0.10,
                     "legs": [{"name": ..., "distance_nm": ..., "speed_knots":
                               ..., "thickness_m": ..., "concentration": ...}]},
          "calibrate_to_anchor": true, "n_mc": 2000
        }

    Missing required keys raise :class:`ValueError` (the route maps it to 422).
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    for key in ("length_m", "beam_m", "draught_m", "block_coefficient"):
        if key not in payload:
            raise ValueError(f"missing required field '{key}'")
    hull = HullParticulars(
        length_m=float(payload["length_m"]),
        beam_m=float(payload["beam_m"]),
        draught_m=float(payload["draught_m"]),
        block_coefficient=float(payload["block_coefficient"]),
        midship_coefficient=float(payload.get("midship_coefficient", 0.99)))
    belt = payload.get("belt", {}) or {}
    spec = WaterlineBandSpec(
        below_waterline_m=float(belt.get("below_waterline_m", 1.2)),
        above_waterline_m=float(belt.get("above_waterline_m", 1.0)),
        cell_size_m=float(belt.get("cell_size_m", 0.5)))
    material_key = str(payload.get("material", "coating")).lower()
    if material_key not in _MATERIALS:
        raise ValueError(f"material must be one of {sorted(_MATERIALS)}")
    material = _MATERIALS[material_key]

    voy = dict(payload.get("voyage", {}) or {})
    base = reference_baltic_voyage()
    legs = base.legs
    if "legs" in voy:
        legs = tuple(
            IceLeg(name=str(item.get("name", f"leg{i}")),
                   distance_nm=float(item["distance_nm"]),
                   speed_knots=float(item["speed_knots"]),
                   thickness_m=float(item["thickness_m"]),
                   concentration=float(item.get("concentration", 1.0)))
            for i, item in enumerate(voy["legs"]))
        if not legs:
            raise ValueError("voyage.legs must not be empty")
    hull_resistance = replace(base.hull_for_resistance,
                              length_m=hull.length_m, beam_m=hull.beam_m,
                              draught_m=hull.draught_m,
                              friction_coefficient=float(
                                  voy.get("friction_coefficient", base.mu)))
    voyage = VoyageProfile(
        legs=legs,
        round_trip=bool(voy.get("round_trip", True)),
        voyages_per_year=float(voy.get("voyages_per_year",
                                       base.voyages_per_year)),
        belt_energy_fraction=float(voy.get("belt_energy_fraction",
                                           base.belt_energy_fraction)),
        hull_for_resistance=hull_resistance)
    spec_rate = float(payload.get("specific_wear_rate_m3_per_Nm",
                                 ICE_ON_STEEL_SPECIFIC_WEAR_GUESS))
    source = WearSourceConfig(
        spec_rate=spec_rate,
        n_events_per_leg=int(payload.get("n_events_per_leg", 4000)),
        seed=int(payload.get("seed", 20261106)))
    grid = build_waterline_grid(hull, spec)
    case = run_reference_case(material=material, source=source, grid=grid,
                              voyage=voyage,
                              calibrate=bool(payload.get("calibrate_to_anchor",
                                                         False)),
                              n_mc=int(payload.get("n_mc", 1000)))
    return {
        "endpoint": "/api/v1/calculations/structure/ice_abrasion",
        "status": "ok",
        "geometry": grid.summary(),
        "result": case.to_dict(),
        "units": {
            "depth": "um (micrometre), 1e-6 m",
            "wear_rate": "m/year and mm/year in the life table",
            "pressure": "Pa",
            "energy": "J",
            "specific_wear_rate": "m^3/(N m)",
        },
    }


__all__ = [
    "ANCHOR_BALTIC_COATING_MICRON_PER_SEASON",
    "ANCHOR_HEAVY_STEEL_MM_PER_YEAR",
    "AH36_STEEL",
    "AbrasionCaseResult",
    "ActivityDistribution",
    "ContactEnsemble",
    "ContactEnsembleConfig",
    "FINNIE_TRANSITION_ANGLE_RAD",
    "HullParticulars",
    "ICE_CLASS_EPOXY",
    "ICE_ON_STEEL_SPECIFIC_WEAR_GUESS",
    "ICE_ON_STEEL_SPECIFIC_WEAR_RANGE",
    "IceLeg",
    "TargetMaterial",
    "VoyageProfile",
    "WearField",
    "WearSeasonResult",
    "WearSourceConfig",
    "WaterlineBandSpec",
    "WaterlineGrid",
    "archard_coefficient_from_specific_rate",
    "archard_wear_depth",
    "archard_wear_volume",
    "audit_case_checks",
    "build_waterline_grid",
    "coating_life_table",
    "finnie_angle_function",
    "finnie_erosion_volume",
    "finnie_flow_stress_from_hardness",
    "dem_convergence_study",
    "grid_convergence_index",
    "ice_abrasion_calculation",
    "leg_friction_work",
    "project_contact_pits",
    "reference_baltic_voyage",
    "reference_heavy_voyage",
    "run_heavy_validation_case",
    "run_reference_case",
    "run_wear_season",
    "sample_contact_ensemble",
    "specific_wear_rate",
]
