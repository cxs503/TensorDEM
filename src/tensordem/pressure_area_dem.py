"""Pressure-area index emergence from a 2-D bonded-particle DEM indentation.

Physical problem
----------------
The global ice pressure transmitted by a wide structure face falls with the
nominal contact area, ``p_G = C A**(-ex)`` with ``ex ~ 0.5`` (Sanderson 1988;
Masterson & Frederking; Dempsey).  Two mechanisms are quoted in the literature:

* geometric -- the *true* contact area is a fraction of the nominal area and
  that fraction grows as the indentation proceeds;
* statistical -- ice strength is spatially variable, so failure is
  **non-simultaneous**: weak patches crush first, load is redistributed to the
  survivors, and the peak global load is set by progressive collapse rather
  than by a single simultaneous brittle event.

This module makes both mechanisms *computed* rather than *assumed*.  A rigid
punch of width ``w`` is driven quasi-statically into a brittle bonded-particle
(lattice) ice layer of thickness ``h``; the punched area ``A = w * 1 m``
(out-of-plane depth of one metre) is scanned, the global pressure
``p_G = F_peak / A`` is measured, and ``ex`` is estimated from the slope of
``log p_G`` versus ``log A`` with a bootstrap confidence interval.

Mechanisms in the model
-----------------------
1. **Contact-area growth** (geometric): every particle in contact with the
   punch carries a Hertz-consistent contact width ``b_i = sqrt(2 r delta_i)``,
   so the instantaneous contact area per unit depth ``A_c = sum_i 2 b_i``
   grows with the local penetration ``delta_i``.  The punch force per particle
   is ``F_i = 2 K_c b_i delta_i`` (Winkler foundation with geometry-consistent
   contact width), hence ``F_i ~ delta_i**1.5``: the exponent 3/2 is a
   *consequence* of ``A_c ~ sqrt(delta)``, not an input.
2. **Non-simultaneous failure** (statistical): every bond carries its own
   brittle failure strain drawn from a spatially correlated lognormal field
   with correlation length ``l`` (``COV = 0.52`` is the lognormal equivalent of
   a Weibull modulus ``m ~ 2``).  Bonds fail irreversibly, the load is
   redistributed by a genuine quasi-static equilibrium solve (not by an assumed
   load-sharing rule), and particles that lose their bond path to the clamped
   base drop out of contact: a *broken element exits the contact*.

Negative control
----------------
``correlation_length -> inf`` gives one perfectly correlated patch: every bond
shares the median strength, all near-threshold bonds fail within one increment
(simultaneous failure) and there is no weak-link statistics left.  A size
effect requires statistics, so the exponent must collapse towards zero.  The
module reports ``ex(l -> inf) ~ 0`` as the control and the family ``ex(l)``.

Evidence grade: REAL.  No coefficient of the pressure-area law is hard-coded;
only material parameters (E, strength median/COV, correlation length) enter and
the exponent is fitted to computed peak loads.

Units: SI, forces per unit out-of-plane depth (N/m), areas per unit depth (m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

__all__ = [
    "PunchConfig",
    "PunchResult",
    "AreaScan",
    "build_triangular_lattice",
    "sample_strength_field",
    "run_punch_indentation",
    "fit_pressure_area_exponent",
    "bootstrap_exponent_ci",
    "run_area_scan",
    "gaussian_field_1d",
    "lattice_modulus_factor",
]


# --------------------------------------------------------------------------
# Lattice calibration
# --------------------------------------------------------------------------
def lattice_modulus_factor() -> float:
    """``E_2D / k_bond`` for an ideal triangular central-force lattice.

    With three bond directions (0, 60, 120 deg), bond stiffness ``k`` and area
    per node ``A_0 = sqrt(3)/2 s**2``, the affine strain energy density is

        W = (k / sqrt(3)) * sum_j (n_j . eps . n_j)**2 .

    For a uniaxial *stress* state ``eps = diag(e, -nu e)`` the triangular
    lattice has ``nu = 1/3``, the bracket collapses to ``e**2`` and
    ``W = (k/sqrt(3)) e**2 = (1/2) E_2D e**2``, hence

        E_2D = 2 k / sqrt(3) = 1.1547 k .

    (A uniaxial *strain* state ``eps = diag(e, 0)`` would instead give
    ``W = (9/8)(k/sqrt(3)) e**2``, the 9/(4 sqrt(3)) factor often quoted for
    the strain-controlled test.)  ``tests/test_pressure_area_dem.py`` measures
    the realised modulus of the built lattice and requires 5 % agreement.
    """
    return 2.0 / math.sqrt(3.0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LatticeGeometry:
    """Triangular bonded-particle layer filling ``[0, width] x [0, height]``."""

    positions: np.ndarray        # (N, 2) reference coordinates [m]
    bonds: np.ndarray            # (M, 2) node index pairs
    spacing: float               # nearest-neighbour distance [m]
    radius: float                # particle radius [m]
    width: float
    height: float
    bottom: np.ndarray           # (N,) bool, clamped base row
    top: np.ndarray              # (N,) bool, topmost row


def build_triangular_lattice(width: float, height: float,
                             spacing: float) -> LatticeGeometry:
    """Build a triangular (isostatic, in-plane isotropic) bond lattice."""
    if width <= 0 or height <= 0 or spacing <= 0:
        raise ValueError("width, height and spacing must be positive")
    dy = spacing * math.sqrt(3.0) / 2.0      # exact triangular-lattice pitch
    n_rows = max(4, int(round(height / dy)) + 1)
    # Snap the layer thickness to a whole number of lattice pitches: the
    # triangular lattice only has the analytic modulus E_2D = 9k/(4 sqrt3) when
    # the pitch is exact (a stretched pitch costs ~10 % of the modulus).
    height = (n_rows - 1) * dy
    rows: list[np.ndarray] = []
    for j in range(n_rows):
        offset = 0.5 * spacing if j % 2 else 0.0
        xs = np.arange(offset, width + 1e-12, spacing)
        if xs.size == 0:
            xs = np.array([0.0])
        xs = xs[xs <= width + 1e-12]
        rows.append(np.column_stack((xs, np.full(xs.shape, j * dy))))
    positions = np.vstack(rows)
    # Drop coincide nt nodes (row offsets can land on top of each other); a
    # zero-length bond would divide by zero and poison the unit vectors.
    key = np.round(positions / (1e-6 * spacing)).astype(np.int64)
    _, unique_index = np.unique(key, axis=0, return_index=True)
    positions = positions[np.sort(unique_index)]
    pairs = cKDTree(positions).query_pairs(1.15 * spacing, output_type="ndarray")
    if pairs.size == 0:
        raise ValueError("no bonds constructed; spacing too small for the domain")
    pairs = np.asarray(pairs, dtype=np.int64)
    rest = np.linalg.norm(positions[pairs[:, 1]] - positions[pairs[:, 0]], axis=1)
    pairs = pairs[rest > 0.5 * spacing]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    bottom = positions[:, 1] <= 1e-9
    top = positions[:, 1] >= height - 1e-9
    return LatticeGeometry(positions=positions, bonds=pairs, spacing=spacing,
                           radius=0.5 * spacing, width=width, height=height,
                           bottom=bottom, top=top)


# --------------------------------------------------------------------------
# Spatially correlated strength field
# --------------------------------------------------------------------------
def sample_strength_field(positions: np.ndarray, spacing: float,
                          correlation_length: float, cov: float,
                          median: float, seed: int,
                          component: str = "bonds",
                          bonds: np.ndarray | None = None) -> np.ndarray:
    """Lognormal brittle-strength field with a Gaussian correlation length.

    ``correlation_length = inf`` returns a *perfectly correlated* field: every
    element shares the median strength (the simultaneous-failure control).
    """
    if not np.isfinite(cov) or cov < 0:
        raise ValueError("cov must be finite and nonnegative")
    if median <= 0 or not np.isfinite(median):
        raise ValueError("median must be finite and positive")
    if not (correlation_length > 0):
        raise ValueError("correlation_length must be positive (inf allowed)")
    n = len(positions)

    def _gather(node_values: np.ndarray) -> np.ndarray:
        if component == "bonds":
            if bonds is None:
                raise ValueError("bonds required for component='bonds'")
            return 0.5 * (node_values[bonds[:, 0]] + node_values[bonds[:, 1]])
        return node_values

    if np.isinf(correlation_length):
        count = len(bonds) if (component == "bonds" and bonds is not None) else n
        return np.full(count, float(median))
    xmin, ymin = positions[:, 0].min(), positions[:, 1].min()
    ix = np.rint((positions[:, 0] - xmin) / spacing).astype(np.int64)
    iy = np.rint((positions[:, 1] - ymin) / spacing).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((nx, ny))
    filled = np.zeros((nx, ny), dtype=bool)
    filled[ix, iy] = True
    white = np.where(filled, white, 0.0)
    sigma = float(correlation_length) / spacing
    if sigma > 1e4:  # numerically identical to the perfectly correlated limit
        count = len(bonds) if (component == "bonds" and bonds is not None) else n
        return np.full(count, float(median))
    smooth = ndimage.gaussian_filter(white, max(sigma, 1e-6), mode="reflect")
    sd = float(smooth[filled].std())
    if sd <= 1e-12:  # pragma: no cover - degenerate grid
        count = len(bonds) if (component == "bonds" and bonds is not None) else n
        return np.full(count, float(median))
    z = (smooth - float(smooth[filled].mean())) / sd
    sigma_ln = math.sqrt(math.log(1.0 + cov * cov))
    values = median * np.exp(sigma_ln * z)
    return _gather(values[ix, iy])


def gaussian_field_1d(x: np.ndarray, spacing: float, length_scale: float,
                      seed: int) -> np.ndarray:
    """Standardised Gaussian height profile along ``x`` (punch-face roughness)."""
    if length_scale <= 0 or not np.isfinite(length_scale):
        raise ValueError("length_scale must be finite and positive")
    order = np.argsort(x)
    xs = x[order]
    n = len(xs)
    step = float(spacing)   # rows share x, so consecutive diffs can be zero
    rng = np.random.default_rng(seed)
    n_grid = max(2, int(round((xs.max() - xs.min()) / step)) + 1)
    white = rng.standard_normal(n_grid)
    sigma = max(length_scale / step, 1e-6)
    smooth = ndimage.gaussian_filter1d(white, sigma, mode="reflect")
    idx = np.rint((xs - xs.min()) / step).astype(np.int64)
    vals = smooth[np.clip(idx, 0, n_grid - 1)]
    sd = float(vals.std())
    vals = (vals - vals.mean()) / (sd if sd > 1e-12 else 1.0)
    out = np.empty(n)
    out[order] = vals
    return out


# --------------------------------------------------------------------------
# Configuration / results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PunchConfig:
    """Rigid-punch indentation of a brittle bonded-particle ice layer."""

    punch_width: float = 0.5            # w [m]; nominal punched area A = w * 1 m
    sheet_thickness: float = 0.2        # h [m]
    spacing: float = 0.04               # particle diameter [m] = 2 r
    young_modulus: float = 9.0e9        # E [Pa]
    poisson_ratio: float = 1.0 / 3.0    # -> E* = E / (1 - nu^2) in plane strain
    # Winkler stiffness of the punch-particle contact, K_c = factor * E*.
    # O(1) is the physical asperity-layer value; the sensitivity to this
    # numerical-looking knob is reported in the run manifest.
    contact_stiffness_factor: float = 1.0
    # Confined-crushing constraint factor: in the contact zone the crushing
    # stress of ice is a multiple (>1) of the uniaxial brittle strength because
    # lateral expansion is restrained.  f_c = 3 is the standard confined value.
    crushing_constraint_factor: float = 3.0
    # Post-yield contact comminution (particles leaving the contact).  Off by
    # default: this campaign measures the *elastic limit* pressure, i.e. the
    # load at the first brittle failure, so post-yield crushing is out of scope
    # and is documented as a limitation rather than half-modelled.
    contact_crushing: bool = False
    # The run stops once the first brittle failure has been resolved; the
    # reported pressure is the elastic-limit (first-failure) pressure.
    stop_at_first_failure: bool = True
    strength_median: float = 1.0e-3     # median brittle failure strain
    strength_cov: float = 0.52          # lognormal equivalent of Weibull m ~ 2
    correlation_length: float = 0.08    # l [m]; math.inf = simultaneous control
    # Punch-face roughness: the punch is not perfectly conformal with the
    # particle layer, so only the high spots touch at first and the contact
    # area grows as the punch advances (mechanism 1 needs this non-uniformity
    # to be physical, otherwise a flat punch loads every particle equally).
    surface_roughness: float = 0.02     # std of the height offset / particle radius
    roughness_correlation: float = 0.08  # correlation length of the height profile [m]
    side_margin_factor: float = 1.5     # free margin each side, in units of h
    max_penetration_fraction: float = 0.03    # u_max / h
    n_increments: int = 50
    newton_iterations: int = 8
    newton_tol: float = 1.0e-6          # relative force residual
    peak_drop_fraction: float = 0.55    # stop after the peak load drop
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("punch_width", "sheet_thickness", "spacing", "young_modulus",
                     "contact_stiffness_factor", "strength_median",
                     "max_penetration_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if self.strength_cov < 0 or not np.isfinite(self.strength_cov):
            raise ValueError("strength_cov must be finite and nonnegative")
        if not (self.correlation_length > 0):
            raise ValueError("correlation_length must be positive (inf allowed)")
        if self.n_increments < 2 or self.newton_iterations < 1:
            raise ValueError("n_increments >= 2 and newton_iterations >= 1 required")
        if self.surface_roughness < 0 or not np.isfinite(self.surface_roughness):
            raise ValueError("surface_roughness must be finite and nonnegative")
        if not (self.roughness_correlation > 0):
            raise ValueError("roughness_correlation must be positive")
        if self.spacing * 3.0 > self.sheet_thickness:
            raise ValueError("spacing too coarse for the sheet thickness")

    @property
    def effective_modulus(self) -> float:
        """Plane-strain modulus ``E / (1 - nu**2)`` used by contacts [Pa]."""
        return self.young_modulus / (1.0 - self.poisson_ratio ** 2)

    @property
    def bond_stiffness(self) -> float:
        """Bond axial stiffness that realises ``E_2D = young_modulus`` [N/m]."""
        return self.young_modulus / lattice_modulus_factor()

    @property
    def nominal_area(self) -> float:
        """Nominal punched area ``A = w * 1 m`` [m**2]."""
        return self.punch_width * 1.0


@dataclass
class PunchResult:
    """Measured response of one punch indentation run."""

    config: PunchConfig
    area: float
    peak_force: float
    peak_pressure: float
    peak_increment: int
    broken_at_peak: int
    contacts_at_peak: int
    contact_area_at_peak: float
    contact_fraction_at_peak: float
    total_bonds: int
    failed_total: int
    non_simultaneity: float
    field_cov_realised: float
    displacement: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)
    contact_area: list[float] = field(default_factory=list)
    failure_events: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_m2": self.area,
            "punch_width_m": self.config.punch_width,
            "peak_force_N_per_m": self.peak_force,
            "peak_pressure_Pa": self.peak_pressure,
            "peak_increment": self.peak_increment,
            "broken_bonds_at_peak": self.broken_at_peak,
            "contacts_at_peak": self.contacts_at_peak,
            "contact_area_at_peak_m": self.contact_area_at_peak,
            "contact_fraction_at_peak": self.contact_fraction_at_peak,
            "total_bonds": self.total_bonds,
            "failed_bonds_total": self.failed_total,
            "non_simultaneity": self.non_simultaneity,
            "field_cov_realised": self.field_cov_realised,
            "correlation_length_m": (None if math.isinf(self.config.correlation_length)
                                     else self.config.correlation_length),
            "strength_cov": self.config.strength_cov,
            "seed": self.config.seed,
            "displacement_m": self.displacement,
            "force_N_per_m": self.force,
            "contact_area_m": self.contact_area,
            "failure_events": self.failure_events,
        }


# --------------------------------------------------------------------------
# Core solver
# --------------------------------------------------------------------------
def _dof_map(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (free node indices, dof map with -1 for inactive nodes)."""
    free = np.flatnonzero(active)
    dof = np.full((len(active), 2), -1, dtype=np.int64)
    dof[free, 0] = 2 * np.arange(len(free))
    dof[free, 1] = 2 * np.arange(len(free)) + 1
    return free, dof


def _stiffness(geom: LatticeGeometry, alive: np.ndarray, active: np.ndarray,
               cfg: PunchConfig) -> csr_matrix:
    """SPD bond stiffness matrix ``K`` over the free dofs."""
    _, dof = _dof_map(active)
    ndof = 2 * int(active.sum())
    bonds = geom.bonds
    delta = geom.positions[bonds[:, 1]] - geom.positions[bonds[:, 0]]
    unit = delta / np.linalg.norm(delta, axis=1)[:, None]
    kk = cfg.bond_stiffness * alive
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for a in (0, 1):
        for b in (0, 1):
            block = kk * unit[:, a] * unit[:, b]
            ra, ca = dof[bonds[:, 1], a], dof[bonds[:, 1], b]
            rb, cb = dof[bonds[:, 0], a], dof[bonds[:, 0], b]
            rows += [ra, rb, ra, rb]
            cols += [ca, cb, cb, ca]
            vals += [block, block, -block, -block]
    rows_cat = np.concatenate(rows)
    cols_cat = np.concatenate(cols)
    vals_cat = np.concatenate(vals)
    keep = (rows_cat >= 0) & (cols_cat >= 0)
    return coo_matrix((vals_cat[keep], (rows_cat[keep], cols_cat[keep])),
                      shape=(ndof, ndof)).tocsr()


def _punch_contact(geom: LatticeGeometry, u: np.ndarray, active: np.ndarray,
                   contactable: np.ndarray, z_punch: float, cfg: PunchConfig,
                   height_offset: np.ndarray | None = None,
                   crushable: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Punch load, contact tangent and diagnostics for the current state.

    The contact is a Winkler foundation with a geometry-consistent width:
    ``b = sqrt(2 r delta)`` and ``F = 2 K_c b delta ~ delta**1.5``.  Returns the
    *load* vector ``q`` acting on the particles (downward, i.e. negative z), the
    tangent ``dF/ddelta`` per particle, the punch reaction and the true contact
    area per unit depth (``sum 2 b``).
    """
    n = len(geom.positions)
    offset = 0.0 if height_offset is None else height_offset
    gap = geom.positions[:, 1] + u[:, 1] + geom.radius + offset - z_punch
    contact = contactable & active & (gap > 0.0)
    if crushable is not None:
        contact &= crushable
    q = np.zeros(n)
    stiff = np.zeros(n)
    if not np.any(contact):
        return q, stiff, 0.0, 0, 0.0
    d = gap[contact]
    b = np.sqrt(2.0 * geom.radius * d)
    k_mag = 2.0 * cfg.contact_stiffness_factor * cfg.effective_modulus
    force = k_mag * b * d
    q[contact] = -force
    stiff[contact] = 1.5 * k_mag * b
    return q, stiff, float(force.sum()), int(contact.sum()), float(2.0 * b.sum())


def _connected_to_base(geom: LatticeGeometry, alive: np.ndarray) -> np.ndarray:
    """Nodes still connected to the clamped base through alive bonds."""
    n = len(geom.positions)
    live = geom.bonds[alive]
    if live.size == 0:
        out = np.zeros(n, dtype=bool)
        out[geom.bottom] = True
        return out
    data = np.ones(len(live))
    graph = coo_matrix((np.concatenate((data, data)),
                        (np.concatenate((live[:, 0], live[:, 1])),
                         np.concatenate((live[:, 1], live[:, 0])))),
                       shape=(n, n)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    base_label = int(labels[np.flatnonzero(geom.bottom)[0]])
    return labels == base_label


def run_punch_indentation(cfg: PunchConfig) -> PunchResult:
    """Drive a rigid punch quasi-statically and return the measured response.

    Displacement controlled: the punch surface is lowered in ``n_increments``
    steps.  At each step the nonlinear equilibrium

        K u = q(u)      (bonded-particle lattice)  +  unilateral punch contact

    is solved by Newton iterations on the SPD system
    ``(K + diag(dF/ddelta))``.  A bond whose axial strain exceeds its correlated
    failure strain is removed irreversibly; the load is redistributed by the
    elastic solve and the run stops at the *first* failure, so the reported
    pressure is the elastic-limit (first-failure) pressure -- the quantity that
    the non-simultaneous-failure interpretation of the pressure-area law is
    about.  Particles that lose their bond path to the clamped base leave the
    contact automatically (a broken element exits the contact).
    """
    margin = cfg.side_margin_factor * cfg.sheet_thickness
    width = cfg.punch_width + 2.0 * margin
    geom = build_triangular_lattice(width, cfg.sheet_thickness, cfg.spacing)
    pos = geom.positions
    bonds = geom.bonds
    n = len(pos)
    # One spatially correlated strength field drives both the bond failure
    # strains (material) and, when enabled, the contact crushing stresses.
    node_strength = sample_strength_field(pos, cfg.spacing, cfg.correlation_length,
                                          cfg.strength_cov, cfg.strength_median,
                                          cfg.seed, component="nodes")
    strength = 0.5 * (node_strength[bonds[:, 0]] + node_strength[bonds[:, 1]])
    crush_strength = (cfg.crushing_constraint_factor * cfg.young_modulus
                      * node_strength)
    rest = np.linalg.norm(pos[bonds[:, 1]] - pos[bonds[:, 0]], axis=1)
    unit = (pos[bonds[:, 1]] - pos[bonds[:, 0]]) / rest[:, None]
    alive = np.ones(len(bonds), dtype=bool)
    crushed = np.zeros(n, dtype=bool)
    u = np.zeros((n, 2), dtype=np.float64)

    x_center = 0.5 * width
    contactable = np.abs(pos[:, 0] - x_center) <= 0.5 * cfg.punch_width + 1e-12
    height_offset = np.zeros(n)
    if cfg.surface_roughness > 0:
        profile = gaussian_field_1d(pos[:, 0], cfg.spacing,
                                    cfg.roughness_correlation, cfg.seed + 10_007)
        height_offset = np.where(contactable,
                                 cfg.surface_roughness * geom.radius * profile, 0.0)
    z_punch = float((pos[:, 1] + height_offset).max()) + geom.radius
    u_max = cfg.max_penetration_fraction * cfg.sheet_thickness
    du = u_max / cfg.n_increments

    disp_hist: list[float] = []
    force_hist: list[float] = []
    area_hist: list[float] = []
    events: list[int] = []
    failed_total = 0
    punch_force = 0.0
    n_contacts = 0
    contact_area = 0.0
    prev = (0.0, 0, 0.0)           # last intact (force, contacts, contact area)
    peak_force = 0.0
    peak_pressure = 0.0
    peak_increment = -1
    yield_increment = -1
    broken_at_yield = 0
    yield_contacts = 0
    yield_area = 0.0

    for increment in range(1, cfg.n_increments + 1):
        z_punch -= du
        first_failure_this_increment = False
        for _ in range(cfg.newton_iterations):
            connected = _connected_to_base(geom, alive)
            active = connected & (~geom.bottom)
            free, _ = _dof_map(active)
            K = _stiffness(geom, alive, active, cfg)
            q, stiff, punch_force, n_contacts, contact_area = _punch_contact(
                geom, u, active, contactable, z_punch, cfg, height_offset, ~crushed)
            ndof = 2 * len(free)
            if ndof == 0:
                break
            u_vec = np.zeros(ndof)
            u_vec[0::2] = u[free, 0]
            u_vec[1::2] = u[free, 1]
            q_vec = np.zeros(ndof)
            q_vec[1::2] = q[free]           # contact load on the vertical dofs
            residual = q_vec - K @ u_vec
            tangent = K + csr_matrix((stiff[free],
                                      (np.arange(len(free)) * 2 + 1,
                                       np.arange(len(free)) * 2 + 1)),
                                     shape=(ndof, ndof))
            sol = spsolve(tangent.tocsc(), residual)
            if not np.all(np.isfinite(sol)):
                break
            u_vec = u_vec + sol
            u[free, 0] = u_vec[0::2]
            u[free, 1] = u_vec[1::2]

            # Contact-zone crushing (opt-in): comminution of contact elements.
            if cfg.contact_crushing:
                gap = pos[:, 1] + u[:, 1] + geom.radius + height_offset - z_punch
                touching = contactable & active & (~crushed) & (gap > 0.0)
                pressure = np.zeros(n)
                pressure[touching] = (cfg.contact_stiffness_factor
                                      * cfg.effective_modulus * gap[touching])
                newly_crushed = touching & (pressure > crush_strength)
                if newly_crushed.any():
                    crushed |= newly_crushed
                    shed = crushed[bonds[:, 0]] | crushed[bonds[:, 1]]
                    alive &= ~shed
                    events.append(int(newly_crushed.sum()))

            # Brittle bond failure (irreversible, strain driven, correlated field)
            strain = np.abs(np.einsum("ij,ij->i",
                                      u[bonds[:, 1]] - u[bonds[:, 0]], unit)) / rest
            new_failed = alive & (strain > strength)
            if new_failed.any():
                alive[new_failed] = False
                events.append(int(new_failed.sum()))
                failed_total += int(new_failed.sum())
                first_failure_this_increment = True
                if cfg.stop_at_first_failure:
                    break
                continue

        _, _, punch_force, n_contacts, contact_area = _punch_contact(
            geom, u, active, contactable, z_punch, cfg, height_offset, ~crushed)
        disp_hist.append(u_max - (cfg.n_increments - increment) * du)
        force_hist.append(punch_force)
        area_hist.append(contact_area)
        if punch_force > peak_force:
            peak_force = punch_force
            peak_pressure = punch_force / cfg.nominal_area
            peak_increment = increment
        if first_failure_this_increment and cfg.stop_at_first_failure:
            # Elastic limit = the last *intact* configuration.
            yield_increment = increment
            broken_at_yield = failed_total
            yield_force, yield_contacts, yield_area = prev
            if yield_increment == 1:
                yield_force, yield_contacts, yield_area = (punch_force, n_contacts,
                                                           contact_area)
            break
        prev = (punch_force, n_contacts, contact_area)
        if (peak_force > 0 and punch_force < cfg.peak_drop_fraction * peak_force
                and increment > 3):
            break

    if yield_increment > 0:
        reported_force = yield_force
        reported_area = yield_area
        reported_contacts = yield_contacts
    else:
        reported_force, reported_area, reported_contacts = (peak_force,
                                                            area_hist[-1],
                                                            n_contacts)
    non_sim = 0.0
    if failed_total > 0:
        non_sim = 1.0 - float(max(events) if events else 0) / float(failed_total)
    return PunchResult(
        config=cfg, area=cfg.nominal_area,
        peak_force=float(reported_force),
        peak_pressure=float(reported_force / cfg.nominal_area),
        peak_increment=int(yield_increment if yield_increment > 0 else peak_increment),
        broken_at_peak=int(broken_at_yield),
        contacts_at_peak=int(reported_contacts),
        contact_area_at_peak=float(reported_area),
        contact_fraction_at_peak=float(reported_area / cfg.punch_width),
        total_bonds=int(len(bonds)), failed_total=int(failed_total),
        non_simultaneity=float(non_sim),
        field_cov_realised=float(np.std(node_strength) / np.mean(node_strength)),
        displacement=disp_hist, force=force_hist, contact_area=area_hist,
        failure_events=events,
    )


# --------------------------------------------------------------------------
# Statistics: exponent, bootstrap CI, area scan
# --------------------------------------------------------------------------
def fit_pressure_area_exponent(areas: Sequence[float],
                               pressures: Sequence[float]) -> dict[str, float]:
    """Least-squares slope of ``log p_G`` against ``log A`` (exponent ``ex``)."""
    a = np.asarray(areas, dtype=float)
    p = np.asarray(pressures, dtype=float)
    mask = (a > 0) & (p > 0) & np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if a.size < 2:
        raise ValueError("need at least two finite (area, pressure) pairs")
    x, y = np.log(a), np.log(p)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"exponent": float(-slope), "slope": float(slope),
            "intercept": float(intercept), "r2": r2,
            "n_points": int(a.size)}


def bootstrap_exponent_ci(areas: Sequence[float], pressures: Sequence[float],
                          n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap CI for the exponent (residual resampling)."""
    fit = fit_pressure_area_exponent(areas, pressures)
    a = np.asarray(areas, dtype=float)
    p = np.asarray(pressures, dtype=float)
    mask = (a > 0) & (p > 0) & np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    x, y = np.log(a), np.log(p)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        y_star = slope * x + intercept + rng.choice(resid, size=resid.size,
                                                    replace=True)
        draws[k] = -np.polyfit(x, y_star, 1)[0]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"exponent": fit["exponent"],
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_width": float(hi - lo), "r2": fit["r2"],
            "n_points": fit["n_points"], "n_boot": int(n_boot)}


@dataclass
class AreaScan:
    """An area scan (many seeds per area) with its exponent estimate."""

    label: str
    correlation_length: float
    areas: list[float]
    runs: list[dict[str, Any]]
    fit: dict[str, float]
    seed_means: dict[float, float]
    seed_stdev: dict[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "correlation_length_m": (None if math.isinf(self.correlation_length)
                                     else self.correlation_length),
            "areas_m2": self.areas,
            "n_seeds_per_area": len(self.runs) // max(len(self.areas), 1),
            "fit": self.fit,
            "seed_mean_pressure_Pa": {f"{k:g}": v for k, v in self.seed_means.items()},
            "seed_stdev_pressure_Pa": {f"{k:g}": v for k, v in self.seed_stdev.items()},
            "runs": self.runs,
        }


def _one_run(args: tuple[PunchConfig, float, int]) -> dict[str, Any]:
    base, area, seed = args
    cfg = PunchConfig(**{**base.__dict__, "punch_width": area, "seed": seed})
    return run_punch_indentation(cfg).to_dict()


def run_area_scan(areas: Sequence[float], seeds: int, base: PunchConfig,
                  label: str = "scan", n_boot: int = 2000,
                  bootstrap_seed: int = 0, workers: int = 1,
                  progress: bool = False) -> AreaScan:
    """Indent every area with ``seeds`` realisations and fit the exponent."""
    jobs = [(base, float(a), int(s)) for a in areas for s in range(seeds)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            runs = pool.map(_one_run, jobs, chunksize=1)
    else:
        runs = []
        for k, job in enumerate(jobs):
            runs.append(_one_run(job))
            if progress:
                print(f"  [{label}] {k + 1}/{len(jobs)} area={job[1]:.3g} seed={job[2]}",
                      flush=True)
    areas_out = [float(a) for a in areas]
    seed_means: dict[float, float] = {}
    seed_stdev: dict[float, float] = {}
    flat_a: list[float] = []
    flat_p: list[float] = []
    for a in areas_out:
        vals = [r["peak_pressure_Pa"] for r in runs if abs(r["area_m2"] - a) < 1e-12]
        seed_means[a] = float(np.mean(vals))
        seed_stdev[a] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        flat_a.extend([a] * len(vals))
        flat_p.extend(vals)
    fit = bootstrap_exponent_ci(flat_a, flat_p, n_boot=n_boot, seed=bootstrap_seed)
    fit["correlation_length_m"] = (None if math.isinf(base.correlation_length)
                                   else base.correlation_length)
    fit["n_seeds_per_area"] = int(seeds)
    return AreaScan(label=label, correlation_length=base.correlation_length,
                    areas=areas_out, runs=runs, fit=fit,
                    seed_means=seed_means, seed_stdev=seed_stdev)