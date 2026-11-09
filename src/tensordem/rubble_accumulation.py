"""Rubble-pile accumulation around a fixed cylindrical structure in a drifting
broken-ice field (campaign node IN69-N06).

Physical problem
----------------
A fixed vertical cylinder of diameter ``D`` stands in a drifting broken-ice field
of piece thickness ``h_i``, ice areal concentration ``C_i`` and far-field ice
speed ``v_i``.  Fragments pile up on the upstream side of the structure.  Three
engineering questions follow:

(a) how does the pile geometry (cross-stream width ``B_p``, thickness ``H_p`` and
    upstream length ``L_p``) grow in time and what sets its steady state?
(b) does the pile *shield* the structure from ice load (``R_shield < 1``) or
    *amplify* it (``R_shield > 1``), and where is the crossover?
(c) what is the steady pile shape as a function of the Froude number
    ``Fr = v_i / sqrt(g h_i)`` and the concentration ``C_i``?

Model (plan-view, depth-averaged 2-D DEM)
----------------------------------------
The horizontal plane is resolved; the vertical structure of the pile is carried
by an *areal* thickness map, not by a third DEM axis:

* every fragment is a rigid disk of diameter ``d`` and thickness ``h_i``
  (``m = rho_i pi r^2 h_i``) with planar translation only -- rotation is not
  integrated (the same idealisation as :mod:`tensordem.ridge_dem`; the rolling of
  irregular ice fragments is absorbed into the declared friction coefficient
  instead of being modelled);
* the drive is the water drag of the drifting field, linearised as
  ``m (v_i - v) / tau_d`` with the *quadratic-drag* time constant
  ``tau_d = rho_i pi r / (rho_w C_d v_i)`` (``C_d = 1``).  A fragment therefore
  relaxes onto the drift velocity in ``tau_d`` (verified analytically in the
  tests); with ``v_i = 0`` there is no drive at all, which is one of the
  pre-registered negative controls;
* contacts are linear spring-dashpot normal + regularised Coulomb tangential
  (``f_t = mu F_n tanh(v_t / v_eps)``, rate form, no per-contact history);
* **buoyant-overburden friction**: a fragment buried in rubble is confined by the
  buoyant weight of the column above it,
  ``N_ov = (rho_w - rho_i) g H_p(local) pi r^2``, and this confining force raises
  the Coulomb cap ``mu (F_n + N_ov)``.  ``H_p(local)`` is read from the areal
  thickness map with the rubble porosity anchor ``n = 0.32 +- 0.05``
  (Timco & O'Brien 2004, see :mod:`tensordem.ice_rubble_microstructure`).  This
  is the only route by which gravity enters a plan-view model, and it is the
  route that produces a Froude-number-dependent pile strength
  (``sigma_pile ~ mu (1-n) (rho_w - rho_i) g H_p``);
* the cylinder is an exact circle SDF (no faceting error);
* the flume is periodic in the flow direction and walled laterally, so mass is
  conserved exactly and the through-flow is continuous (a "circulating flume").

Thickness recovery (the bridge between the 2-D model and the 3-D pile)
---------------------------------------------------------------------
For an areal ice fraction ``phi`` in a cell, the pile thickness that this ice
would occupy at solid fraction ``1-n`` is

.. math:: H_p = \\phi h_i / (1 - n)

which is the standard "concentration -> thickness" map used for ice fields, here
with the rubble porosity anchor instead of the open-water value.  ``H_p`` is
therefore a *derived* quantity and is graded as such.

Load definition
---------------
The cylinder load is reported as a line load per unit submerged height
``q = F_x / h_i`` and as a pressure ``p_front = q / D``.  Because the same
definition is used for the accumulation run and for the pile-clearing control,
the shielding ratio ``R_shield = p_front / p_0`` is independent of this
convention.

Evidence grading (machine readable in the study JSON)
-----------------------------------------------------
``REAL``            DEM-computed: growth series, geometry, cylinder load, mass
                    ledger, penetration guard, negative controls.
``ANALYTIC``        pre-registered momentum-balance reference for the scaling
                    exponents and the crossover Froude number.
``LITERATURE``      rubble porosity ``n = 0.32 +- 0.05`` (Timco & O'Brien 2004)
                    and the ice/steel and ice/ice friction ranges.
``WITHHELD``        a per-dataset comparison of ``B_p/H_p/L_p`` against digitised
                    model-test data: no digitised source is available in this
                    tree, so the scaling comparison is made against the analytic
                    reference only.  Declared in
                    ``prereg/IN69_N06_rubble_accumulation_prereg.json`` *before*
                    the sweep was run.

Limitations (see ``docs/IN69_N06_rubble_accumulation.md``)
----------------------------------------------------------
Planar translation only; monodisperse thickness; no water column (added mass,
vertical ride-up, in-pile percolation) and no refreezing/cohesion; the lateral
banks are walls, so lateral bypass can only occur inside the flume;
rotation-free Coulomb contacts make the residual friction angle a *measured*
rather than an assumed quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "GRAVITY",
    "RUBBLE_POROSITY",
    "ICE_DENSITY",
    "WATER_DENSITY",
    "RubbleMaterial",
    "StructureSpec",
    "IceConditions",
    "DomainSpec",
    "AccumulationConfig",
    "PieceField",
    "generate_piece_field",
    "areal_thickness_mapping",
    "RubbleAccumulation",
    "AccumulationRun",
    "exponential_growth_fit",
    "loglog_exponent",
    "momentum_balance_reference",
    "sweep_rows",
    "run_case",
]

GRAVITY = 9.80665
ICE_DENSITY = 917.0
WATER_DENSITY = 1025.0
#: Timco & O'Brien (2004) rubble porosity, mean +- std (LITERATURE anchor).
RUBBLE_POROSITY = (0.32, 0.05)
#: Ice-ice friction range used for the fragment contacts (LITERATURE anchor).
ICE_ICE_FRICTION = (0.3, 0.6)


def _wrap(values: np.ndarray, period: float) -> np.ndarray:
    return values - period * np.round(values / period)


# --------------------------------------------------------------------------- #
# specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RubbleMaterial:
    """Contact and transport parameters (SI)."""

    density_ice: float = ICE_DENSITY
    density_water: float = WATER_DENSITY
    porosity: float = RUBBLE_POROSITY[0]
    mu_ice: float = 0.5
    mu_structure: float = 0.3
    kn: float = 3.0e4
    zeta: float = 0.2
    drag_coefficient: float = 1.0
    v_eps: float = 5.0e-3
    overburden_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("density_ice", "density_water", "kn"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.porosity < 1.0:
            raise ValueError("porosity must be in [0, 1)")
        if not (math.isfinite(self.overburden_scale) and self.overburden_scale >= 0.0):
            raise ValueError("overburden_scale must be finite and nonnegative")

    @property
    def solid_fraction(self) -> float:
        return 1.0 - self.porosity

    @property
    def buoyancy_deficit(self) -> float:
        """``(rho_w - rho_i) / rho_i`` -- the buoyancy that sets the effective g."""
        return (self.density_water - self.density_ice) / self.density_ice


@dataclass(frozen=True)
class StructureSpec:
    """Fixed circular cylinder (exact SDF, no faceting error)."""

    diameter: float
    x_center: float = 0.0
    y_center: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.diameter) and self.diameter > 0.0):
            raise ValueError("diameter must be finite and positive")

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def front_x(self) -> float:
        return self.x_center - self.radius


@dataclass(frozen=True)
class IceConditions:
    """Far-field broken-ice field."""

    velocity: float
    thickness: float
    concentration: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.velocity) and self.velocity >= 0.0):
            raise ValueError("velocity must be finite and nonnegative")
        if not (math.isfinite(self.thickness) and self.thickness > 0.0):
            raise ValueError("thickness must be finite and positive")
        if not 0.0 <= self.concentration <= 1.0:
            raise ValueError("concentration must be in [0, 1]")

    @property
    def froude(self) -> float:
        return self.velocity / math.sqrt(GRAVITY * self.thickness)

    @property
    def mass_flux(self) -> float:
        """Ice mass flux per unit width, ``rho_i C_i h_i v_i`` (kg m^-1 s^-1)."""
        return ICE_DENSITY * self.concentration * self.thickness * self.velocity

    @property
    def driving_stress(self) -> float:
        """Momentum flux, ``rho_i C_i v_i^2`` (Pa)."""
        return ICE_DENSITY * self.concentration * self.velocity ** 2


@dataclass(frozen=True)
class DomainSpec:
    length: float = 26.0
    width: float = 20.0

    def __post_init__(self) -> None:
        for name in ("length", "width"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass(frozen=True)
class AccumulationConfig:
    """Everything that defines one case."""

    material: RubbleMaterial = RubbleMaterial()
    domain: DomainSpec = DomainSpec()
    structure: StructureSpec = StructureSpec(diameter=3.0, x_center=0.62 * DomainSpec().length,
                                             y_center=0.5 * DomainSpec().width)
    conditions: IceConditions = IceConditions(velocity=0.3, thickness=0.3, concentration=0.3)
    duration: float = 40.0
    sample_dt: float = 0.25
    seed: int = 0
    d_min: float = 0.15
    d_max: float = 0.55
    clearing: bool = False
    dt: float | None = None

    def __post_init__(self) -> None:
        for name in ("duration", "sample_dt"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if self.d_min <= 0.0 or self.d_max <= self.d_min:
            raise ValueError("need 0 < d_min < d_max")
        if self.d_max > 0.5 * self.domain.width:
            raise ValueError("pieces must be small compared with the flume width")


# --------------------------------------------------------------------------- #
# fragment field
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PieceField:
    """Initial fragment field (plan view)."""

    positions: np.ndarray
    radii: np.ndarray
    domain: DomainSpec

    @property
    def diameters(self) -> np.ndarray:
        return 2.0 * self.radii

    @property
    def n(self) -> int:
        return len(self.radii)

    @property
    def areal_fraction(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.sum(math.pi * self.radii ** 2) / self.domain.area)

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {"n_pieces": 0.0, "d_min": 0.0, "d_max": 0.0, "d_mean": 0.0,
                    "areal_fraction": 0.0}
        dia = self.diameters
        return {
            "n_pieces": float(self.n),
            "d_min": float(dia.min()),
            "d_max": float(dia.max()),
            "d_mean": float(dia.mean()),
            "areal_fraction": self.areal_fraction,
        }


def _sample_diameters(count: int, d_min: float, d_max: float,
                      rng: np.random.Generator, exponent: float = 2.6) -> np.ndarray:
    """Truncated power-law fragment size distribution ``p(d) ~ d^-alpha``.

    ``alpha = 2.6`` is a *declared* shape (broken-ice fields and
    :mod:`tensordem.broken_ice_micro` sit around 2.5-3), not a tuned one.
    """
    if count <= 0:
        return np.zeros(0)
    u = rng.random(count)
    lo, hi = d_min ** (1.0 - exponent), d_max ** (1.0 - exponent)
    return (u * lo + (1.0 - u) * hi) ** (1.0 / (1.0 - exponent))


def _periodic_pairs(pos: np.ndarray, length: float, r_max: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour pairs on an ``x``-periodic strip, with minimum-image vectors.

    ``cKDTree`` knows nothing about periodicity, so the point set is replicated
    at ``x +- length`` and only pairs involving a base point are kept; the pair
    separation is then taken between the *actual* (possibly wrapped) images, so
    the contact across the periodic seam is exactly the contact across the seam.
    """
    n = len(pos)
    empty_i = np.zeros(0, dtype=np.int64)
    if n < 2:
        return empty_i, empty_i, np.zeros((0, 2)), np.zeros(0)
    aug = np.concatenate([pos, pos + np.array([length, 0.0]),
                          pos - np.array([length, 0.0])])
    tree = cKDTree(aug)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    if pairs is None or len(pairs) == 0:
        return empty_i, empty_i, np.zeros((0, 2)), np.zeros(0)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    i = pairs[:, 0]
    j = pairs[:, 1]
    keep = i < n
    i, j = i[keep], j[keep]
    jb = j % n
    keep = i != jb
    i, j, jb = i[keep], j[keep], jb[keep]
    key = np.minimum(i, jb) * n + np.maximum(i, jb)
    _, first = np.unique(key, return_index=True)
    i, j, jb = i[first], j[first], jb[first]
    dvec = aug[j] - pos[i]
    dist = np.linalg.norm(dvec, axis=1)
    return i, jb, dvec, dist


def _lattice_sites(domain: DomainSpec, spacing: float) -> np.ndarray:
    """Triangular (hexagonal) lattice of candidate insertion points."""
    nx = max(1, int(math.ceil(domain.length / spacing)) + 1)
    ny = max(1, int(math.ceil(domain.width / spacing)) + 1)
    xs = np.arange(nx) * spacing
    ys = np.arange(ny) * spacing
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    gx = gx + 0.5 * spacing * (np.arange(ny)[:, None] % 2)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _relax_overlaps(positions: np.ndarray, radii: np.ndarray, domain: DomainSpec,
                    structure: StructureSpec, *, iterations: int = 120,
                    r_max: float | None = None) -> np.ndarray:
    """Position-based removal of initial overlaps and of structure penetration."""
    pos = positions.copy()
    r_max = r_max if r_max is not None else 2.0 * float(radii.max())
    for _ in range(iterations):
        i, j, dvec, dist = _periodic_pairs(pos, domain.length, r_max)
        if len(i):
            dist = np.maximum(dist, 1.0e-12)
            overlap = radii[i] + radii[j] - dist
            active = overlap > 1.0e-4
            if active.any():
                unit = dvec[active] / dist[active, None]
                shift = 0.5 * overlap[active, None] * unit
                np.add.at(pos, i[active], -shift)
                np.add.at(pos, j[active], +shift)
        dvec = pos - np.array([structure.x_center, structure.y_center])
        dvec[:, 0] -= domain.length * np.round(dvec[:, 0] / domain.length)
        dist = np.maximum(np.linalg.norm(dvec, axis=1), 1.0e-12)
        overlap = radii + structure.radius - dist
        active = overlap > 1.0e-4
        if active.any():
            unit = dvec[active] / dist[active, None]
            pos[active] += (overlap[active] + 1.0e-3)[:, None] * unit
        # keep inside the lateral banks
        pos[:, 1] = np.clip(pos[:, 1], radii, domain.width - radii)
        pos[:, 0] = np.mod(pos[:, 0], domain.length)
    return pos


def generate_piece_field(domain: DomainSpec, structure: StructureSpec,
                         *, concentration: float, d_min: float = 0.15,
                         d_max: float = 0.55, seed: int = 0,
                         clearance: float = 0.05,
                         relax_iterations: int = 120) -> PieceField:
    """Jittered-lattice fragment field with overlap relaxation.

    Random sequential insertion stalls at the random-close-packing limit
    (``phi ~ 0.55`` for disks), which blocks the high-concentration end of the
    sweep.  The field is therefore laid out on a jittered triangular lattice and
    the residual overlaps are removed by position-based relaxation, which reaches
    the requested areal fraction for ``C_i`` up to ``0.7``.  The realised areal
    fraction is reported with the field, never assumed.
    """
    rng = np.random.default_rng(seed)
    if concentration <= 0.0:
        return PieceField(np.zeros((0, 2)), np.zeros(0), domain)
    mean_radius = 0.25 * (d_min + d_max)
    count = max(4, int(round(concentration * domain.area / (math.pi * mean_radius ** 2))))
    diameters = _sample_diameters(count, d_min, d_max, rng)
    radii = 0.5 * diameters
    # scale the population to reach the requested areal fraction, respecting the
    # declared size window; iterate because clipping changes the total area
    for _ in range(40):
        total = float(np.sum(math.pi * radii ** 2))
        radii = np.clip(radii * math.sqrt(concentration * domain.area / total),
                        0.5 * d_min, 0.5 * d_max)
    spacing = math.sqrt(domain.area / count)
    for _try in range(40):
        sites = _lattice_sites(domain, spacing)
        keep = np.hypot(sites[:, 0] - structure.x_center,
                        sites[:, 1] - structure.y_center) > structure.radius + clearance
        sites = sites[keep]
        sites = sites[(sites[:, 1] > radii.max()) & (sites[:, 1] < domain.width - radii.max())]
        sites[:, 0] = np.clip(sites[:, 0], radii.max(), domain.length - radii.max())
        if len(sites) >= count:
            break
        spacing *= 0.97
    else:  # pragma: no cover
        raise RuntimeError("could not build a fragment lattice for this box")
    pick = rng.permutation(len(sites))[:count]
    positions = sites[pick] + 0.25 * spacing * (rng.random((count, 2)) - 0.5)
    positions[:, 0] = np.mod(positions[:, 0], domain.length)
    positions = _relax_overlaps(positions, radii, domain, structure,
                                iterations=relax_iterations)
    return PieceField(positions, radii, domain)


def areal_thickness_mapping(area_fraction, thickness: float, porosity: float) -> np.ndarray:
    """``H_p = phi h_i / (1 - n)`` -- concentration to pile-thickness map."""
    if not (0.0 <= porosity < 1.0):
        raise ValueError("porosity must be in [0, 1)")
    if not (math.isfinite(thickness) and thickness > 0.0):
        raise ValueError("thickness must be finite and positive")
    phi = np.asarray(area_fraction, dtype=np.float64)
    if np.any(phi < 0.0):
        raise ValueError("area fractions must be nonnegative")
    return phi * thickness / (1.0 - porosity)# --------------------------------------------------------------------------- #
# the DEM engine
# --------------------------------------------------------------------------- #
class RubbleAccumulation:
    """Deterministic plan-view DEM of a broken-ice field around a cylinder."""

    def __init__(self, field: PieceField, config: AccumulationConfig) -> None:
        self.cfg = config
        self.domain = config.domain
        self.material = config.material
        self.structure = config.structure
        self.conditions = config.conditions
        self.pos = np.array(field.positions, dtype=np.float64).reshape(-1, 2)
        self.radii = np.array(field.radii, dtype=np.float64).reshape(-1)
        self.n = len(self.radii)
        self.vel = np.zeros_like(self.pos)
        self.mass = (self.material.density_ice * math.pi * self.radii ** 2
                     * self.conditions.thickness)
        self.area = math.pi * self.radii ** 2
        self.time = 0.0
        self.steps = 0
        self.structure_force = np.zeros(2)
        self.structure_force_x_history: list[float] = []
        self.teleports = 0
        self.max_penetration_ratio = 0.0
        # flux ledger: the box spans the pile region, bounded upstream by the
        # inlet gate and downstream by the structure plane, full flume width
        self.box_length = min(0.45 * self.domain.length, 6.0 * self.cfg.d_max + 4.0)
        self.gate_inlet = self.structure.front_x - self.box_length
        self.mass_total0 = float(self.mass.sum())
        self.ledger = {"inlet_in": 0.0, "inlet_back": 0.0, "outlet": 0.0}
        self._gate_offset = np.mod(self.pos[:, 0] - self.gate_inlet, self.domain.length) \
            if self.n else np.zeros(0)
        self._outlet_offset = np.mod(self.pos[:, 0] - self.structure.front_x,
                                     self.domain.length) if self.n else np.zeros(0)
        self._contacts: dict[str, np.ndarray] = {
            "i": np.zeros(0, dtype=int), "j": np.zeros(0, dtype=int),
            "overlap": np.zeros(0), "fn": np.zeros(0), "ft": np.zeros(0)}
        self._overburden = np.zeros(self.n)
        self._thickness_grid = self._compute_thickness_grid()
        self.dt = config.dt if config.dt is not None else self.recommended_dt
        if self.dt > self.recommended_dt * (1.0 + 1e-12):
            raise ValueError(f"dt exceeds the stability limit {self.recommended_dt:.6g} s")

    # ------------------------------------------------------------------ setup
    @property
    def recommended_dt(self) -> float:
        if self.n == 0:
            return 1.0e-2
        m_min = float(self.mass.min())
        omega = math.sqrt(2.0 * self.material.kn / (0.5 * m_min))
        return 0.1 / omega

    @property
    def drag_time(self) -> float:
        """Quadratic-drag linearisation time constant ``rho_i pi r / (rho_w C_d v)``."""
        velocity = self.conditions.velocity
        if velocity <= 0.0:
            return math.inf
        r_mean = float(self.radii.mean()) if self.n else 0.15
        return (self.material.density_ice * math.pi * r_mean
                / (self.material.density_water * self.material.drag_coefficient * velocity))

    def _wrapped_offsets(self) -> np.ndarray:
        dvec = self.pos - np.array([self.structure.x_center, self.structure.y_center])
        dvec[:, 0] -= self.domain.length * np.round(dvec[:, 0] / self.domain.length)
        return dvec

    def _wrap_dx(self) -> np.ndarray:
        return _wrap(self.pos[:, 0] - self.structure.front_x, self.domain.length)

    def _compute_thickness_grid(self) -> dict[str, Any]:
        cell = 2.0 * self.cfg.d_max
        nx = max(1, int(math.ceil(self.domain.length / cell)))
        ny = max(1, int(math.ceil(self.domain.width / cell)))
        phi = np.zeros(nx * ny)
        ix = np.zeros(0, dtype=int)
        iy = np.zeros(0, dtype=int)
        if self.n:
            ix = np.clip((self.pos[:, 0] / self.domain.length * nx).astype(int), 0, nx - 1)
            iy = np.clip((self.pos[:, 1] / self.domain.width * ny).astype(int), 0, ny - 1)
            np.add.at(phi, ix * ny + iy, self.area)
        phi /= self.domain.area / (nx * ny)
        thickness = areal_thickness_mapping(phi, self.conditions.thickness,
                                            self.material.porosity)
        local = thickness[ix * ny + iy] if self.n else np.zeros(0)
        return {"nx": nx, "ny": ny, "cell": cell, "phi": phi.reshape(nx, ny),
                "thickness": thickness.reshape(nx, ny), "local": local}

    def _overburden_force(self) -> np.ndarray:
        """Buoyant confining force ``N_ov`` on each buried piece (N)."""
        if self.material.overburden_scale <= 0.0 or self.n == 0:
            return np.zeros(self.n)
        dg = (self.material.density_water - self.material.density_ice) * GRAVITY
        return (self.material.overburden_scale * dg
                * self._thickness_grid["local"] * self.area)

    # ----------------------------------------------------------------- forces
    def _pair_contacts(self):
        empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 2)),
                 np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 2)))
        if self.n < 2:
            return empty
        r_max = 2.0 * float(self.radii.max())
        i, j, dvec, dist = _periodic_pairs(self.pos, self.domain.length, r_max)
        if len(i) == 0:
            return empty
        keep = dist < self.radii[i] + self.radii[j]
        i, j, dvec, dist = i[keep], j[keep], dvec[keep], dist[keep]
        if len(i) == 0:
            return empty
        overlap = self.radii[i] + self.radii[j] - dist
        normal = dvec / dist[:, None]
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_n = 2.0 * self.material.zeta * np.sqrt(self.material.kn * m_eff)
        v_rel = self.vel[i] - self.vel[j]
        # normal points from i to j, so the penetration rate is d(delta)/dt = v_n:
        # the dashpot must *add* to the spring while the pair keeps closing
        approach = np.sum(v_rel * normal, axis=1)
        fn = np.maximum(0.0, self.material.kn * overlap + c_n * approach)
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
        v_t = np.sum(v_rel * tangent, axis=1)
        cap = self.material.mu_ice * (fn + 0.5 * (self._overburden[i] + self._overburden[j]))
        # the pair force is applied as -f_pair on i, so the tangential part must
        # carry +cap*tanh(v_t/v_eps) for the friction on i to oppose v_rel
        ft = cap * np.tanh(v_t / self.material.v_eps)
        return i, j, normal, overlap, fn, ft, tangent

    def _structure_forces(self) -> tuple[np.ndarray, np.ndarray]:
        force = np.zeros((self.n, 2))
        if self.n == 0:
            return force, np.zeros(2)
        dvec = self._wrapped_offsets()
        dist = np.maximum(np.linalg.norm(dvec, axis=1), 1.0e-12)
        overlap = self.radii + self.structure.radius - dist
        mask = overlap > 0.0
        if not mask.any():
            return force, np.zeros(2)
        normal = dvec[mask] / dist[mask, None]
        c_n = 2.0 * self.material.zeta * np.sqrt(self.material.kn * self.mass[mask])
        v_n = np.sum(self.vel[mask] * normal, axis=1)
        fn = np.maximum(0.0, self.material.kn * overlap[mask] - c_n * v_n)
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
        v_t = np.sum(self.vel[mask] * tangent, axis=1)
        cap = self.material.mu_structure * (fn + self._overburden[mask])
        ft = -cap * np.tanh(v_t / self.material.v_eps)
        f_on_piece = fn[:, None] * normal + ft[:, None] * tangent
        force[mask] = f_on_piece
        self.max_penetration_ratio = max(
            self.max_penetration_ratio,
            float(np.max(overlap[mask] / self.radii[mask])))
        return force, -f_on_piece.sum(axis=0)

    def _wall_forces(self) -> np.ndarray:
        force = np.zeros((self.n, 2))
        if self.n == 0:
            return force
        cases = ((-1.0, self.pos[:, 1] - self.radii),
                 (+1.0, self.pos[:, 1] + self.radii - self.domain.width))
        for sign, penetration in cases:
            overlap = -penetration if sign < 0 else penetration
            mask = overlap > 0.0
            if not mask.any():
                continue
            c_n = 2.0 * self.material.zeta * np.sqrt(self.material.kn * self.mass[mask])
            approach = self.vel[mask, 1] * sign  # d(overlap)/dt for this bank
            fn = np.maximum(0.0, self.material.kn * overlap[mask] + c_n * approach)
            v_t = self.vel[mask, 0]
            cap = self.material.mu_structure * (fn + self._overburden[mask])
            ft = -cap * np.tanh(v_t / self.material.v_eps)
            # the bank pushes the disk back into the flume: the outward normal of
            # the lower bank is +y (i.e. -sign), of the upper bank -y
            force[mask, 1] += fn * (-sign)
            force[mask, 0] += ft
            self.max_penetration_ratio = max(
                self.max_penetration_ratio,
                float(np.max(overlap[mask] / self.radii[mask])))
        return force

    def _drag_force(self) -> np.ndarray:
        velocity = self.conditions.velocity
        if velocity <= 0.0 or self.n == 0:
            return np.zeros((self.n, 2))
        target = np.array([velocity, 0.0])
        return self.mass[:, None] * (target[None, :] - self.vel) / self.drag_time

    def forces(self) -> np.ndarray:
        self._overburden = self._overburden_force()
        force = np.zeros((self.n, 2))
        i, j, normal, overlap, fn, ft, tangent = self._pair_contacts()
        if len(i):
            f_pair = fn[:, None] * normal + ft[:, None] * tangent
            np.add.at(force, i, -f_pair)
            np.add.at(force, j, +f_pair)
            self.max_penetration_ratio = max(
                self.max_penetration_ratio,
                float(np.max(overlap / (self.radii[i] + self.radii[j]))))
        self._contacts = {"i": np.asarray(i), "j": np.asarray(j),
                          "overlap": np.asarray(overlap), "fn": np.asarray(fn),
                          "ft": np.abs(np.asarray(ft))}
        f_struct, load = self._structure_forces()
        force += f_struct
        self.structure_force = load
        force += self._wall_forces()
        force += self._drag_force()
        return force

    # -------------------------------------------------------------- integration
    def step(self, count: int = 1) -> None:
        for _ in range(count):
            force = self.forces()
            if self.n:
                self.vel += self.dt * force / self.mass[:, None]
                self.pos += self.dt * self.vel
                self.pos[:, 0] = np.mod(self.pos[:, 0], self.domain.length)
                if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
                    raise FloatingPointError("non-finite state: DEM diverged")
            self.time += self.dt
            self.steps += 1
            self._update_ledger()
            self.structure_force_x_history.append(float(self.structure_force[0]))
            if self.cfg.clearing:
                self._clear_exclusion_zone()

    def _clear_exclusion_zone(self) -> None:
        """Pile-clearing control: flush pieces out of the exclusion disc."""
        if self.n == 0:
            return
        radius = self.structure.radius + 1.5 * self.cfg.d_max
        inside = np.linalg.norm(self._wrapped_offsets(), axis=1) < radius
        if not inside.any():
            return
        idx = np.where(inside)[0]
        self.pos[idx, 0] = np.mod(self.structure.front_x - 0.45 * self.domain.length,
                                  self.domain.length)
        self.vel[idx] = np.array([self.conditions.velocity, 0.0])
        self.teleports += int(len(idx))

    # --------------------------------------------------------------- measurement
    def _update_ledger(self) -> None:
        """Count mass crossing the inlet gate and the structure-plane outlet."""
        if self.n == 0:
            return
        pos = self.pos
        dx = np.mod(pos[:, 0] - self._last_x + 0.5 * self.domain.length,
                    self.domain.length) - 0.5 * self.domain.length \
            if hasattr(self, "_last_x") else np.zeros(self.n)
        self._last_x = pos[:, 0].copy()
        for key, gate, state_key, sign in (
                ("inlet", self.gate_inlet, "_gate_offset", +1),
                ("outlet", self.structure.front_x, "_outlet_offset", +1)):
            previous = getattr(self, state_key)
            current = previous + dx
            n_cross = np.floor(current / self.domain.length) - np.floor(
                previous / self.domain.length)
            if np.any(n_cross != 0):
                forward = n_cross > 0
                backward = n_cross < 0
                if key == "inlet":
                    self.ledger["inlet_in"] += float(self.mass[forward].sum())
                    self.ledger["inlet_back"] += float(self.mass[backward].sum())
                else:
                    self.ledger["outlet"] += float(self.mass[(forward | backward)].sum())
            setattr(self, state_key, np.mod(current, self.domain.length))

    def box_mass(self) -> tuple[float, float]:
        """Mass inside the pile box and mass inside the pile cluster."""
        if self.n == 0:
            return 0.0, 0.0
        dx = self._wrap_dx()
        inside = (dx <= 0.0) & (dx >= -self.box_length)
        pile = self.pile_mask()
        return float(self.mass[inside].sum()), float(self.mass[pile].sum())

    def mass_audit(self) -> dict[str, float]:
        """Mass bookkeeping: total drift, trapped ice and the flux closure.

        Every fragment is followed individually, so the pile mass must obey the
        integral balance over the pile box

            m_pile(t) - m_pile(0) = [inlet_in - inlet_back - outlet]
                                    - [m_free(t) - m_free(0)]

        where ``outlet`` is the mass that passes the structure plane (i.e. the
        bypass) and ``m_free`` is the ice inside the box that is *not* part of the
        contact cluster attached to the cylinder.  The residual of this identity
        is the closure error; it can only be non-zero if a crossing is missed by
        the finite time step, which is what makes it a usable numerical guard.
        """
        in_box, in_pile = self.box_mass()
        free = in_box - in_pile
        if getattr(self, "_ledger_ref", None) is None:
            self._ledger_ref = {"in_pile": in_pile, "free": free,
                                "inlet_in": self.ledger["inlet_in"],
                                "inlet_back": self.ledger["inlet_back"],
                                "outlet": self.ledger["outlet"]}
        ref = self._ledger_ref
        inlet = self.ledger["inlet_in"] - ref["inlet_in"]
        back = self.ledger["inlet_back"] - ref["inlet_back"]
        outlet = self.ledger["outlet"] - ref["outlet"]
        expected = inlet - back - outlet - (free - ref["free"])
        residual = (in_pile - ref["in_pile"]) - expected
        closure = abs(residual) / inlet if inlet > 0.0 else float("nan")
        return {"mass_total_initial": self.mass_total0,
                "mass_total_final": float(self.mass.sum()),
                "mass_total_drift": float(self.mass.sum()) - self.mass_total0,
                "mass_in_box": in_box, "mass_in_pile": in_pile,
                "ledger_inlet_in": inlet, "ledger_inlet_back": back,
                "ledger_outlet": outlet,
                "mass_closure_relative": float(closure),
                "mass_pile_growth": float(in_pile - self._ledger_ref["in_pile"])
                if getattr(self, "_ledger_ref", None) else 0.0,
                "ledger_net_inlet": float(inlet), "ledger_bypass": float(outlet),
                "mass_trapped": float(self.mass[(np.linalg.norm(self._wrapped_offsets(),
                                                                axis=1)
                                                 < self.structure.radius)].sum())}

    def pile_mask(self) -> np.ndarray:
        """Pieces connected to the cylinder by a contact path (the pile)."""
        mask = np.zeros(self.n, dtype=bool)
        if self.n == 0:
            return mask
        touching = (np.linalg.norm(self._wrapped_offsets(), axis=1)
                    < self.radii + self.structure.radius + 1.0e-9)
        if not touching.any():
            return mask
        i, j = self._contacts["i"], self._contacts["j"]
        if len(i):
            graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(self.n, self.n))
            _n_comp, labels = connected_components(graph, directed=False)
            roots = np.unique(labels[touching])
            mask = np.isin(labels, roots)
        else:
            mask = touching.copy()
        # the pile lives upstream of (or beside) the cylinder, never downstream
        mask &= self._wrap_dx() < 0.5 * self.structure.diameter
        return mask

    def geometry(self) -> dict[str, float]:
        """Pile geometry at the current state (metres)."""
        mask = self.pile_mask()
        out: dict[str, float] = {
            "pile_discs": float(mask.sum()),
            "pile_areal_fraction": float(self.area[mask].sum() / self.domain.area),
            "pile_mass": float(self.mass[mask].sum()),
            "pile_cluster_share": float(mask.sum() / self.n) if self.n else 0.0,
            "B_p": 0.0, "H_p": 0.0, "H_p_mean": 0.0, "L_p": 0.0,
        }
        if not mask.any():
            return out
        dx = self._wrap_dx()[mask]
        y = self.pos[mask, 1]
        out["L_p"] = float(max(0.0, -float(dx.min())))
        width = 0.0
        if len(dx) >= 2:
            nbins = max(2, int(len(dx) // 4) + 1)
            bins = np.linspace(float(dx.min()), float(dx.max()), nbins)
            which = np.clip(np.digitize(dx, bins) - 1, 0, nbins - 2)
            for b in np.unique(which):
                sel = which == b
                if sel.sum() >= 2:
                    width = max(width, float(y[sel].max() - y[sel].min()))
        out["B_p"] = width
        grid = self._thickness_grid
        ix = np.clip((self.pos[mask, 0] / self.domain.length * grid["nx"]).astype(int),
                     0, grid["nx"] - 1)
        iy = np.clip((self.pos[mask, 1] / self.domain.width * grid["ny"]).astype(int),
                     0, grid["ny"] - 1)
        values = grid["thickness"][ix, iy]
        out["H_p"] = float(values.max())
        out["H_p_mean"] = float(values.mean())
        return out

    def load(self, window: float = 2.0) -> float:
        """Window-averaged front pressure on the cylinder (Pa)."""
        size = int(max(1, round(window / max(self.dt, 1e-12))))
        history = self.structure_force_x_history[-size:]
        if not history:
            return 0.0
        line_load = float(np.mean(history)) / self.conditions.thickness
        return line_load / self.structure.diameter

    def snapshot(self) -> dict[str, float]:
        self._thickness_grid = self._compute_thickness_grid()
        out = {"time": float(self.time)}
        out.update(self.geometry())
        out["p_front"] = self.load()
        out["f_x"] = float(self.structure_force[0])
        out["f_y"] = float(self.structure_force[1])
        out["max_penetration_ratio"] = float(self.max_penetration_ratio)
        out["teleports"] = float(self.teleports)
        out.update(self.mass_audit())
        return out


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
@dataclass
class AccumulationRun:
    """Time series and summary of one accumulation (or control) run."""

    config: AccumulationConfig
    samples: list[dict[str, float]]
    field: dict[str, float]
    dt: float
    drag_time: float
    wall_seconds: float
    clearing: bool
    final_state: dict | None = None

    def series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        t = np.array([s["time"] for s in self.samples], dtype=float)
        y = np.array([s.get(key, 0.0) for s in self.samples], dtype=float)
        return t, y

    def plateau(self, key: str, fraction: float = 0.25) -> float:
        t, y = self.series(key)
        if len(y) == 0:
            return 0.0
        keep = t >= t.max() * (1.0 - fraction)
        return float(np.mean(y[keep]))

    def summary(self) -> dict[str, Any]:
        cond = self.config.conditions
        out = {
            "v_i": cond.velocity, "h_i": cond.thickness, "C_i": cond.concentration,
            "D": self.config.structure.diameter, "Fr": cond.froude,
            "seed": self.config.seed, "clearing": bool(self.clearing),
            "duration": self.config.duration, "dt": self.dt, "drag_time": self.drag_time,
            "wall_seconds": self.wall_seconds,
            "n_samples": len(self.samples),
        }
        out.update({f"field_{k}": v for k, v in self.field.items()})
        for key in ("B_p", "H_p", "H_p_mean", "L_p", "pile_mass", "pile_discs",
                    "pile_cluster_share", "p_front", "max_penetration_ratio"):
            out[f"{key}_plateau"] = self.plateau(key)
            out[f"{key}_peak"] = float(np.max(self.series(key)[1])) if self.samples else 0.0
        out["p0_analytic"] = cond.driving_stress
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "samples": self.samples}


def run_accumulation(config: AccumulationConfig, *,
                     engine: RubbleAccumulation | None = None,
                     progress: bool = False) -> AccumulationRun:
    """Integrate one case and record the pile diagnostics on ``sample_dt``."""
    import time as _time

    field = generate_piece_field(config.domain, config.structure,
                                concentration=config.conditions.concentration,
                                d_min=config.d_min, d_max=config.d_max,
                                seed=config.seed)
    sim = engine or RubbleAccumulation(field, config)
    dt = sim.dt
    steps_per_sample = max(1, int(round(config.sample_dt / dt)))
    n_samples = max(2, int(round(config.duration / config.sample_dt)) + 1)
    samples = [sim.snapshot()]
    started = _time.perf_counter()
    for k in range(1, n_samples):
        sim.step(steps_per_sample)
        samples.append(sim.snapshot())
        if progress and k % 20 == 0:
            print(f"  t={sim.time:7.2f}s pile={samples[-1]['pile_discs']:5.0f} "
                  f"L_p={samples[-1]['L_p']:5.2f} H_p={samples[-1]['H_p']:5.2f}", flush=True)
    wall = _time.perf_counter() - started
    final_state = {
        "positions": sim.pos.tolist(), "radii": sim.radii.tolist(),
        "thickness": np.asarray(sim._thickness_grid["thickness"]).tolist(),
        "axis": [sim.structure.x_center, sim.structure.y_center],
        "diameter": sim.structure.diameter,
        "box_length": sim.box_length,
    }
    return AccumulationRun(config=config, samples=samples, field=field.summary(),
                           dt=dt, drag_time=sim.drag_time, wall_seconds=wall,
                           clearing=config.clearing, final_state=final_state)


def run_case(row: dict[str, Any], *, duration: float = 40.0,
             sample_dt: float = 0.25, clearing: bool = False,
             material: RubbleMaterial | None = None,
             domain: DomainSpec | None = None,
             seed: int = 0, progress: bool = False) -> AccumulationRun:
    """Run one sweep row: ``row`` carries ``v_i, h_i, C_i, D``."""
    cfg = AccumulationConfig(
        material=material or RubbleMaterial(),
        domain=domain or DomainSpec(),
        structure=StructureSpec(diameter=float(row["D"]),
                                x_center=0.62 * (domain or DomainSpec()).length,
                                y_center=0.5 * (domain or DomainSpec()).width),
        conditions=IceConditions(velocity=float(row["v_i"]),
                                 thickness=float(row["h_i"]),
                                 concentration=float(row["C_i"])),
        duration=duration, sample_dt=sample_dt, seed=seed, clearing=clearing)
    return run_accumulation(cfg, progress=progress)


# --------------------------------------------------------------------------- #
# growth law and scaling analysis
# --------------------------------------------------------------------------- #
def exponential_growth_fit(time: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit ``X(t) = X_inf (1 - exp(-t/tau))`` and return the fit quality."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(t) < 4 or np.ptp(y) <= 0.0:
        return {"X_inf": float(y.max() if len(y) else 0.0), "tau": float("nan"),
                "r2": float("nan"), "converged": 0.0, "n": float(len(t))}

    def model(tt, inf, tau):
        return inf * (1.0 - np.exp(-tt / tau))

    span = float(np.max(y) - np.min(y))
    p0 = [float(y.max()), max(1.0, 0.3 * float(t.max()))]
    bounds = ([0.0, 1.0e-3], [float(y.max()) * 20.0 + span + 1e-9, 1.0e4 * max(1.0, t.max())])
    try:
        popt, _cov = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=40000)
        converged = 1.0
    except Exception:  # pragma: no cover - degenerate series
        popt = np.array(p0, dtype=float)
        converged = 0.0
    resid = y - model(t, *popt)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0.0 else float("nan")
    return {"X_inf": float(popt[0]), "tau": float(popt[1]), "r2": float(r2),
            "converged": converged, "n": float(len(t))}


def loglog_exponent(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS slope of ``ln y`` against ``ln x`` with a 1-sigma standard error."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = (xa > 0.0) & (ya > 0.0)
    xa, ya = xa[keep], ya[keep]
    if len(xa) < 2:
        return {"exponent": float("nan"), "stderr": float("nan"),
                "r2": float("nan"), "n": float(len(xa))}
    lx, ly = np.log(xa), np.log(ya)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    resid = ly - pred
    dof = max(1, len(lx) - 2)
    sst = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0.0 else float("nan")
    stderr = math.sqrt(float(np.sum(resid ** 2)) / dof
                       / max(float(np.sum((lx - lx.mean()) ** 2)), 1e-30))
    return {"exponent": float(slope), "stderr": float(stderr), "r2": float(r2),
            "n": float(len(lx))}


def momentum_balance_reference(conditions: IceConditions,
                               material: RubbleMaterial | None = None,
                               *, height_ratio: float | None = None) -> dict[str, float]:
    """Pre-registered analytic reference for the steady pile geometry.

    Steady pile height from a balance between the driving momentum flux
    ``rho_i C_i v_i^2`` and the buoyant-rubble strength
    ``mu (1-n) (rho_w - rho_i) g H_p``::

        H_inf / h_i = C_i Fr^2 / (mu (1-n) (rho_w/rho_i - 1))

    Arrest length of a fragment inside a rubble pile whose resistance is the
    buoyant overburden friction ``mu (rho_w - rho_i) g H_p pi r^2``::

        L_inf / h_i = Fr^2 / (2 mu (rho_w/rho_i - 1) (H_inf/h_i))

    Cross-stream spreading at the residual friction angle::

        B_inf / L_inf ~ 2 mu

    and the shielding/amplification crossover (``R = 1``) sits where the driving
    stress equals the pile strength, i.e. at ``Fr_t = Fr*sqrt(H_inf/h_i / C_i)``
    evaluated with the measured ``H_inf/h_i`` in the study analysis.
    """
    mat = material or RubbleMaterial()
    fr = conditions.froude
    ratio = mat.density_water / mat.density_ice - 1.0
    h_ratio = (conditions.concentration * fr ** 2
               / (mat.mu_ice * mat.solid_fraction * ratio))
    if height_ratio is not None:
        h_ratio = float(height_ratio)
    l_ratio = (fr ** 2 / (2.0 * mat.mu_ice * ratio * max(h_ratio, 1e-9)))
    return {"Fr": fr,
            "H_inf_over_h_i": float(h_ratio),
            "L_inf_over_h_i": float(l_ratio),
            "B_inf_over_L_inf": float(2.0 * mat.mu_ice),
            "H_exponent_on_Fr": 2.0,
            "H_exponent_on_C": 1.0,
            "L_exponent_on_Fr": 2.0,
            "strength_stress": float(mat.mu_ice * mat.solid_fraction
                                     * (mat.density_water - mat.density_ice)
                                     * GRAVITY * h_ratio * conditions.thickness)}


def sweep_rows(*, v_i: float = 0.3, h_i: float = 0.3, C_i: float = 0.3,
               D: float = 3.0, froude: float | None = None,
               velocity: float | None = None) -> list[dict[str, float]]:
    """One-factor-at-a-time sweep around the base case (pre-registered design)."""
    v_values = velocity if velocity is not None else (
        [row["v"] for row in froude] if froude is not None else
        [0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    rows: list[dict[str, float]] = []

    def add(**kw) -> None:
        row = {"v_i": v_i, "h_i": h_i, "C_i": C_i, "D": D}
        row.update(kw)
        key = (round(row["v_i"], 6), round(row["h_i"], 6),
               round(row["C_i"], 6), round(row["D"], 6))
        if key not in {(round(r["v_i"], 6), round(r["h_i"], 6), round(r["C_i"], 6),
                        round(r["D"], 6)) for r in rows}:
            rows.append(row)

    add()
    for v in v_values:
        add(v_i=float(v))
    for h in (0.1, 0.2, 0.3, 0.4, 0.5):
        add(h_i=float(h))
    for c in (0.1, 0.2, 0.3, 0.4, 0.5):
        add(C_i=float(c))
    for d in (1.0, 2.0, 3.0, 4.0, 5.0):
        add(D=float(d))
    return rows