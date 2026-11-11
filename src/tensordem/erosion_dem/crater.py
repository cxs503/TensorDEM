"""Single-particle impact crater: from one removal event to a depth field.

The chain implemented here is

    impact kinematics -> erosion ratio E [kg/kg] -> removed volume
    -> crater depth / radius (spherical cap) -> volumetric deposit on the wall

so that a single impact (criterion "single impact crater volume") and a
multi-particle stream (:mod:`tensordem.erosion_dem.stream`) share one code path
and one bookkeeping convention.  Nothing in this module imports the breakage /
PBM modules: erosion accounting is structurally decoupled from particle
comminution (criterion C7).

Crater geometry
---------------
A particle of radius ``R`` that excavates a volume ``V`` leaves a spherical cap
of depth ``h`` and footprint radius ``a``::

    V(h) = pi (R h**2 - h**3 / 3),     a(h) = sqrt(2 R h - h**2)

``V(h)`` is monotone on ``[0, R]``, so the inverse is solved by bisection to
machine precision.  ``h > R`` would mean the crater is deeper than the particle
radius, i.e. the particle dug its own diameter into the wall -- the geometry
guard rejects it instead of silently extrapolating.

Erosion law
-----------
The ratio is Oka's plastic model in its published scaling structure,

    E(v, alpha, d) = E_anchor (v/v_ref)**k (d/d_ref)**k3 g(alpha)/g(alpha_ref),
    k = 2.3 Hv**0.038,  k3 = 0.19,  g = (sin a)**n1 (1 + Hv (1 - sin a))**n2,
    n1 = 0.71 Hv**0.14, n2 = 2.4 Hv**-0.94,

with the angle function and the exponents taken from
:mod:`tensordem.erosion.models` (single source of truth) and **one** anchor
``E_anchor`` in kg/kg.  The anchor exists for the same reason the library
refuses an absolute Oka rate: the leading constant's unit convention cannot be
re-derived offline.  It is fixed once, by least squares against the DNV-RP-O501
ductile rate over the velocity envelope, and the per-point residual is then the
*shape* error of the comparison (criterion C3).

Finnie's cutting model is carried alongside as the second, independent angle
function (:func:`tensordem.erosion.models.finnie_erosion_ratio`) for the mutual
adjudication of the angle curves (criterion C1).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tensordem.erosion.materials import Material, get_material
from tensordem.erosion.models import (
    OKA_CONSTANTS,
    finnie_erosion_ratio,
    oka_angle_function,
    oka_peak_angle_deg,
)

__all__ = [
    "SPHERE_FACTOR",
    "CraterGeometry",
    "sphere_cap_volume",
    "sphere_cap_depth_from_volume",
    "sphere_cap_radius",
    "indentation_depth_m",
    "OkaAnchoredLaw",
    "single_impact_crater",
    "WallSurface",
]

SPHERE_FACTOR = math.pi / 6.0
"""Volume of a unit-diameter sphere."""

_DEPTH_TOL_M = 1e-15


@dataclass(frozen=True)
class CraterGeometry:
    """Geometry and mass book of one impact crater."""

    removed_volume_m3: float
    depth_m: float
    radius_m: float
    footprint_area_m2: float
    removed_mass_kg: float
    particle_radius_m: float
    particle_volume_m3: float
    volume_fraction_of_particle: float
    depth_fraction_of_particle_radius: float

    @property
    def physically_admissible(self) -> bool:
        """Crater must be within the excavated-particle geometry and positive."""
        return (
            self.removed_volume_m3 >= 0.0
            and self.depth_m >= 0.0
            and self.depth_m <= self.particle_radius_m * (1.0 + 1e-12)
        )

    @property
    def excavation_ratio(self) -> float:
        """Removed solid volume per unit erodent volume (dimensionless)."""
        return self.removed_volume_m3 / self.particle_volume_m3


def sphere_cap_volume(depth_m: float, particle_radius_m: float) -> float:
    """Volume of the spherical cap cut by a sphere of radius ``R`` at depth h."""
    h = float(depth_m)
    r = float(particle_radius_m)
    if r <= 0.0:
        raise ValueError("particle_radius_m must be > 0")
    if h < 0.0:
        raise ValueError("depth_m must be >= 0")
    if h > 2.0 * r:
        raise ValueError("depth_m cannot exceed the sphere diameter")
    return math.pi * (r * h * h - h * h * h / 3.0)


def sphere_cap_depth_from_volume(
    volume_m3: float,
    particle_radius_m: float,
    *,
    n_iter: int = 200,
) -> float:
    """Invert :func:`sphere_cap_volume` by bisection on ``h in [0, R]``.

    The search interval stops at ``R`` (a crater cannot be deeper than the
    particle radius): a removal volume that needs more is rejected, because the
    single-particle crater model is no longer the right geometric picture there
    (that regime is a puncture / penetration problem, not erosion).
    """
    v = float(volume_m3)
    r = float(particle_radius_m)
    if r <= 0.0:
        raise ValueError("particle_radius_m must be > 0")
    if v < 0.0:
        raise ValueError("volume_m3 must be >= 0")
    if v == 0.0:
        return 0.0
    v_max = sphere_cap_volume(r, r)
    if v > v_max * (1.0 + 1e-12):
        raise ValueError(
            f"removed volume {v:.6e} m3 exceeds the half-sphere crater capacity "
            f"{v_max:.6e} m3 of a particle of radius {r:.6e} m (penetration "
            "regime, not erosion)"
        )
    lo, hi = 0.0, r
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if sphere_cap_volume(mid, r) < v:
            lo = mid
        else:
            hi = mid
        if hi - lo < _DEPTH_TOL_M * max(r, 1.0):
            break
    return 0.5 * (lo + hi)


def sphere_cap_radius(depth_m: float, particle_radius_m: float) -> float:
    """Footprint radius ``a = sqrt(2 R h - h**2)`` of a cap of depth ``h``."""
    h = float(depth_m)
    r = float(particle_radius_m)
    if h < 0.0 or h > 2.0 * r:
        raise ValueError("depth_m must lie within [0, 2 R]")
    return math.sqrt(max(0.0, 2.0 * r * h - h * h))


def indentation_depth_m(
    normal_impact_speed_m_s: float,
    particle_mass_kg: float,
    particle_radius_m: float,
    flow_stress_pa: float,
) -> float:
    """Plastic indentation depth of a sphere from the work balance.

    The normal kinetic energy is spent against a constant contact pressure ``p``
    (the hardness-derived flow stress) acting over the projected area of the
    growing cap::

        0.5 m v_n**2 = p V(h)   ->   V = 0.5 m v_n**2 / p  ->  h by inversion

    In the shallow limit ``V ~ pi R h**2`` this gives ``h ~ v_n``, i.e. crater
    *volume* ``~ v_n**2``: the quadratic velocity dependence that the erosion
    laws inherit.  Used as the geometric leg of the single-impact crater and as
    the physical plausibility bound on the analytic removals.
    """
    if normal_impact_speed_m_s < 0.0:
        raise ValueError("normal_impact_speed_m_s must be >= 0")
    if particle_mass_kg <= 0.0:
        raise ValueError("particle_mass_kg must be > 0")
    if flow_stress_pa <= 0.0:
        raise ValueError("flow_stress_pa must be > 0")
    work = 0.5 * particle_mass_kg * normal_impact_speed_m_s**2
    return sphere_cap_depth_from_volume(work / flow_stress_pa, particle_radius_m)


@dataclass(frozen=True)
class OkaAnchoredLaw:
    """Oka plastic erosion law with one explicit anchor in kg/kg.

    ``anchor_ratio_kg_kg`` is the erosion ratio at
    ``(v_ref, alpha_ref, d_ref)``.  Everything else is published Oka physics
    (exponents ``k = 2.3 Hv**0.038``, ``k3 = 0.19`` and the hardness-parameterised
    angle function), so the law has exactly one calibrated number -- the one the
    open literature cannot pin down offline.
    """

    material: Material
    anchor_ratio_kg_kg: float
    v_ref_m_s: float = 30.0
    alpha_ref_deg: float = 90.0
    d_ref_um: float = 150.0
    k3: float = OKA_CONSTANTS["k3"]
    anchor_provenance: str = ""

    @classmethod
    def for_material(
        cls,
        material: Material | str,
        anchor_ratio_kg_kg: float,
        **kwargs,
    ) -> "OkaAnchoredLaw":
        mat = get_material(material) if isinstance(material, str) else material
        return cls(material=mat, anchor_ratio_kg_kg=float(anchor_ratio_kg_kg), **kwargs)

    def __post_init__(self) -> None:
        if self.anchor_ratio_kg_kg < 0.0:
            raise ValueError("anchor_ratio_kg_kg must be >= 0")
        if self.v_ref_m_s <= 0.0 or not 0.0 < self.alpha_ref_deg <= 90.0:
            raise ValueError("v_ref_m_s must be > 0 and alpha_ref_deg in (0, 90]")
        if self.d_ref_um <= 0.0:
            raise ValueError("d_ref_um must be > 0")

    # -- published Oka exponents ---------------------------------------
    @property
    def velocity_exponent(self) -> float:
        """``k = 2.3 Hv**0.038`` (Oka et al. 2005)."""
        return 2.3 * self.material.hardness_hv_gpa**0.038

    @property
    def peak_angle_deg(self) -> float:
        return float(oka_peak_angle_deg(self.material.hardness_hv_gpa))

    def angle_factor(self, alpha_deg) -> np.ndarray:
        """``g(alpha)/g(alpha_ref)``: the angle function normalised at the anchor."""
        g_ref = float(oka_angle_function(self.alpha_ref_deg, self.material.hardness_hv_gpa))
        return np.asarray(
            oka_angle_function(alpha_deg, self.material.hardness_hv_gpa), dtype=float
        ) / g_ref

    def ratio(self, speed_m_s, alpha_deg, diameter_um: float | np.ndarray = None):
        """Erosion ratio ``E`` in kg removed per kg erodent."""
        v = np.asarray(speed_m_s, dtype=float)
        if np.any(v < 0.0):
            raise ValueError("speed_m_s must be >= 0")
        d_um = self.d_ref_um if diameter_um is None else np.asarray(diameter_um, dtype=float)
        if np.any(d_um <= 0.0):
            raise ValueError("diameter_um must be > 0")
        e = (
            self.anchor_ratio_kg_kg
            * (v / self.v_ref_m_s) ** self.velocity_exponent
            * (d_um / self.d_ref_um) ** self.k3
            * self.angle_factor(alpha_deg)
        )
        return float(e) if np.ndim(e) == 0 else e


def single_impact_crater(
    speed_m_s: float,
    alpha_deg: float,
    particle_diameter_m: float,
    particle_density_kg_m3: float,
    material: Material | str,
    *,
    law: OkaAnchoredLaw | None = None,
) -> CraterGeometry:
    """Crater of one particle striking the wall at ``(speed_m_s, alpha_deg)``.

    Removed mass comes from the erosion law (``m_rem = E m_p``), the geometry
    from the spherical-cap inversion.  ``law=None`` uses Finnie's closed-form
    cutting ratio, which is the model-independent low bound of the ductile
    family and needs no anchor.
    """
    mat = get_material(material) if isinstance(material, str) else material
    d = float(particle_diameter_m)
    rho_p = float(particle_density_kg_m3)
    if d <= 0.0 or rho_p <= 0.0:
        raise ValueError("particle diameter and density must be > 0")
    v = float(speed_m_s)
    if v < 0.0:
        raise ValueError("speed_m_s must be >= 0")
    r_p = 0.5 * d
    v_p = SPHERE_FACTOR * d**3
    m_p = rho_p * v_p
    if law is None:
        ratio = float(finnie_erosion_ratio(v, alpha_deg, mat))
    else:
        ratio = float(law.ratio(v, alpha_deg, d * 1e6))
    if ratio < 0.0 or not math.isfinite(ratio):
        raise ValueError("erosion law returned an invalid ratio")
    v_rem = ratio * m_p / mat.density_kg_m3
    h = sphere_cap_depth_from_volume(v_rem, r_p)
    a = sphere_cap_radius(h, r_p)
    return CraterGeometry(
        removed_volume_m3=v_rem,
        depth_m=h,
        radius_m=a,
        footprint_area_m2=math.pi * a * a,
        removed_mass_kg=v_rem * mat.density_kg_m3,
        particle_radius_m=r_p,
        particle_volume_m3=v_p,
        volume_fraction_of_particle=(v_rem / v_p) if v_p > 0.0 else 0.0,
        depth_fraction_of_particle_radius=(h / r_p) if r_p > 0.0 else 0.0,
    )


class WallSurface:
    """Volumetric bookkeeping of the eroded wall depth field.

    The wall patch is a Cartesian grid of ``nx * ny`` cells of area
    ``cell_area_m2``.  Each crater deposits its removed volume into the four
    surrounding cells with bilinear weights, so the deposit operator conserves
    volume to machine precision; the closure residual reported by the stream is
    therefore a statement about the *operator and the grid*, not a fitting
    remainder.
    """

    __slots__ = ("nx", "ny", "cell_area_m2", "depth", "deposits", "volume_deposited_m3")

    def __init__(self, nx: int, ny: int, cell_area_m2: float) -> None:
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be >= 1")
        if cell_area_m2 <= 0.0:
            raise ValueError("cell_area_m2 must be > 0")
        self.nx = int(nx)
        self.ny = int(ny)
        self.cell_area_m2 = float(cell_area_m2)
        self.depth = np.zeros((nx, ny), dtype=float)
        self.deposits = 0
        self.volume_deposited_m3 = 0.0

    @property
    def area_m2(self) -> float:
        return self.nx * self.ny * self.cell_area_m2

    @property
    def integrated_volume_m3(self) -> float:
        """``integral h dA`` by the exact cell-wise sum."""
        return float(self.depth.sum() * self.cell_area_m2)

    @property
    def closure_residual(self) -> float:
        """Relative gap between ``integral h dA`` and the deposited volume."""
        if self.volume_deposited_m3 <= 0.0:
            return 0.0
        return abs(self.integrated_volume_m3 - self.volume_deposited_m3) / (
            self.volume_deposited_m3
        )

    @property
    def max_depth_m(self) -> float:
        return float(self.depth.max()) if self.depth.size else 0.0

    @property
    def mean_depth_m(self) -> float:
        return self.integrated_volume_m3 / self.area_m2

    def deposit(self, x_m: float, y_m: float, volume_m3: float, extent_m: float) -> None:
        """Bilinearly distribute ``volume_m3`` around ``(x_m, y_m)``.

        ``extent_m`` is the physical side of the patch; coordinates outside are
        folded back (periodic) so that the operator never loses volume at the
        patch boundary.  Negative or non-finite volumes are rejected -- the
        "removed mass must stay strictly non-negative" half of criterion C4.
        """
        v = float(volume_m3)
        if v < 0.0 or not math.isfinite(v):
            raise ValueError("volume_m3 must be finite and >= 0")
        if extent_m <= 0.0:
            raise ValueError("extent_m must be > 0")
        if v == 0.0:
            return
        fx = (float(x_m) / extent_m + 0.5) * self.nx - 0.5
        fy = (float(y_m) / extent_m + 0.5) * self.ny - 0.5
        i0 = int(math.floor(fx))
        j0 = int(math.floor(fy))
        tx = fx - i0
        ty = fy - j0
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wy in ((0, 1.0 - ty), (1, ty)):
                w = wx * wy
                if w <= 0.0:
                    continue
                self.depth[(i0 + di) % self.nx, (j0 + dj) % self.ny] += (
                    v * w / self.cell_area_m2
                )
        self.deposits += 1
        self.volume_deposited_m3 += v