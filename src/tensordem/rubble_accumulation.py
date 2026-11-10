"""Rubble accumulation around a fixed cylindrical structure (campaign IN69-N06).

Physical problem
----------------
A drifting broken-ice field (areal concentration ``C_i``, floe thickness
``h_i``, drift speed ``v_i``) is driven against a fixed bottom-founded cylinder
of diameter ``D``.  Blocks that cannot pass the structure pile up in front of it
and form a *rubble accumulation* (a "sail" / rubble field).  The observables:

* geometry -- upstream length ``L_p(t)``, pile width ``B_p(t)``, pile height
  ``H_p(t)`` measured above the undisturbed floating ice surface;
* the front ice pressure on the structure ``p_front(t)``, ratioed against the
  pressure ``p_0`` of the *same* ice field on the *same* bare cylinder (no
  accumulation, the pile is continuously swept away), giving the
  shielding/amplification factor ``R_shield = p_front / p_0``;
* the steady state ``(L_inf, B_inf, H_inf)`` versus Froude number
  ``Fr = v_i / sqrt(g h_i)`` and areal concentration ``C_i``.

Model (and its honest limits)
-----------------------------
* Engine: three-dimensional soft-sphere DEM, ``numpy`` + ``scipy`` only, reusing
  :func:`tensordem.granular3d_core.build_pairs` for the neighbour search.  Equal
  spheres of radius ``r = particle_radius_ratio*h_i`` stand for rubble blocks:
  the *block* scale, not the crystal scale.
* Free surface at ``z = 0``.  Buoyancy uses the exact spherical-cap volume below
  the waterline, ``V_sub = pi h^2 (3r - h)/3``; weight is ``rho_i g V``.  A
  floating block therefore finds the exact Archimedean equilibrium submerged
  fraction ``rho_i/rho_w`` (unit-tested).
* Driving: the ice field is fed through the upstream face at its drift speed
  ``v_i`` (rate-controlled feed -- the Lagrangian equivalent of the classical
  velocity-controlled platen of ice-load DEM).  Detached blocks additionally feel
  a linear water drag towards ``(v_i, 0, 0)``.  The drag is *not* the driving
  mechanism (it is too weak, as in nature); the feed is, exactly as a continuous
  ice sheet pushes.
* Structure: an infinite circular cylinder; horizontal-plane signed distance is
  closed form, ``sdf = hypot(x, y) - R``.  Contact is Kelvin-Voigt normal plus a
  Cundall-Strack tangential law capped at ``mu_structure F_n``.
* Open boundaries: blocks drifting past the structure (``x > x_out``) or around
  it (``|y| > W``) leave the domain and are booked as *bypass* mass, so the mass
  ledger is a genuine open-system balance (flux crossing the capture-zone
  boundary is measured independently of the inventory).
* Limits, reported with every result: rolling resistance is a viscous angular
  damping; blocks are spheres (no interlock geometry), so the emergent internal
  friction angle is a model property, not a measured ice property; no block
  crushing/fragmentation, no added mass, no freeze bonds.  Particle size is
  capped by a count budget, so small ``h_i`` at large ``D`` cases are
  *under-resolved* in sheet thickness and are flagged, never silently averaged.

Units: SI (m, kg, s, N, Pa) throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from .granular3d_core import build_pairs

__all__ = [
    "G",
    "GrowthFit",
    "PileGeometry",
    "RubbleConfig",
    "RubbleAccumulation",
    "cylinder_sdf",
    "extract_pile_geometry",
    "fit_exponential_growth",
    "floating_equilibrium_height",
    "momentum_baseline_pressure",
    "shielding_ratio",
    "run_rubble_case",
    "submerged_volume",
]

G = 9.81

#: Tangential/normal stiffness ratio of block contacts.
KT_RATIO = 0.5


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def submerged_volume(z, r: float):
    """Volume of a sphere (radius ``r``, centre height ``z``) below ``z = 0``.

    Exact spherical cap: with the submerged height ``h = clip(r - z, 0, 2r)``
    the cap volume is ``pi h^2 (3r - h) / 3``.  Vectorised.
    """
    z = np.asarray(z, dtype=float)
    h = np.clip(r - z, 0.0, 2.0 * r)
    return math.pi * h * h * (3.0 * r - h) / 3.0


def floating_equilibrium_height(r: float, rho_ice: float = 900.0,
                                rho_water: float = 1025.0) -> float:
    """Centre height ``z`` of a freely floating sphere (Archimedes).

    Solves ``V_sub(z)/V = rho_ice/rho_water`` for the centre height; the
    returned value is negative (the centre sits below the waterline) because ice
    is lighter than water.
    """
    frac = rho_ice / rho_water
    v_full = 4.0 / 3.0 * math.pi * r ** 3

    def dbg(z: float) -> float:
        return submerged_volume(z, r) / v_full - frac

    lo, hi = -r, r
    if dbg(lo) < 0.0:
        return -r
    if dbg(hi) > 0.0:
        return r
    # bisection (cheap, monotone, no scipy root dependency on this path)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dbg(mid) > 0.0:
            lo = mid        # still too deep: the root lies above mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cylinder_sdf(points: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance to an infinite circular cylinder about the ``z`` axis.

    ``< 0`` inside, ``> 0`` outside.  Closed form: ``hypot(x, y) - R``.
    """
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.zeros(0)
    return np.hypot(p[:, 0], p[:, 1]) - radius


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RubbleConfig:
    """Physical + numerical specification of one accumulation case (SI).

    ``diameter`` D [m], ``ice_thickness`` h_i [m], ``concentration`` C_i [-],
    ``drift_speed`` v_i [m/s] are the four physical knobs of the campaign.
    """

    diameter: float = 3.0
    ice_thickness: float = 0.25
    concentration: float = 0.30
    drift_speed: float = 0.30

    # -- discretisation / domain (all in units of D or h_i) -----------------
    layer_target: float = 2.5                # target block layers across h_i
    particle_budget: int = 7000              # count cap -> r may be raised
    upstream_diameters: float = 2.0          # feed length upstream of the axis
    downstream_diameters: float = 1.0        # open outflow length
    half_width_diameters: float = 1.25       # half width (free-slip walls)
    depth_thickness_ratio: float = 8.0       # water column depth / h_i

    # -- contact / material -------------------------------------------------
    rho_ice: float = 900.0
    rho_water: float = 1025.0
    mu_particle: float = 0.35                # ice-ice Coulomb friction
    mu_structure: float = 0.10               # ice-steel Coulomb friction
    restitution: float = 0.20
    target_static_overlap: float = 0.01      # delta/r under the block weight
    angular_damping_time: float = 0.05       # s, surrogate rolling resistance
    driving_pressure: float = 20_000.0        # Pa, ice-field driving pressure
    rotation: bool = False                   # central-force (rotationless) DEM
    drag_time: float = 2.0                   # s, horizontal water drag
    vertical_drag_time: float = 0.35         # s, vertical water damping

    # -- upstream sheet thrust (pressure boundary) --------------------------
    #: Pressure [Pa] of the (unmodelled) semi-infinite ice sheet pushing the
    #: modelled field towards the structure, applied as a per-block force
    #: ``sigma * (2r)**2`` on the blocks inside the feed band
    #: ``[x_in, x_in + thrust_band_diameters*D]``.  ``0`` disables the term, so
    #: the plain feed + water-drag drive (previous behaviour) is the default.
    thrust_pressure: float = 0.0
    thrust_band_diameters: float = 1.0

    # -- time stepping / sampling ------------------------------------------
    timestep_safety: float = 0.15            # dt = safety * contact time
    max_timestep: float = 5.0e-3
    max_steps: int = 26_000
    samples: int = 200
    time_factor: float = 3.0                # build-up time scale multiplier
    max_physical_time: float = 150.0

    # -- measurement --------------------------------------------------------
    pile_shell_diameters: float = 1.0        # capture shell beyond R for stats
    pile_speed_fraction: float = 0.5         # |v| < frac*v_i  => "resident"
    #: >0 enables the *sweeper* (cleared-cylinder) baseline; the removal shell
    #: is ``R + sweep_shell_radii * r`` (in block radii beyond the surface).
    sweep_shell_radii: float = 0.0
    #: Residence time [s] a block must spend slow inside the capture shell
    #: before the sweeper baseline removes it.  Contacts deliver their impulse
    #: during that window, so the cleared-cylinder baseline ``p_0`` stays
    #: physical (a block-level "instant sweep" deletes the load before it can be
    #: measured and gave p_0 = 0 in some cases -- see the amendment-1 record).
    sweep_dwell_time: float = 0.05
    seed: int = 0

    # -- derived ------------------------------------------------------------
    @cached_property
    def structure_radius(self) -> float:
        return 0.5 * self.diameter

    @cached_property
    def particle_radius(self) -> float:
        """Block radius: ``ratio*h_i`` unless the count budget forces coarser."""
        r_target = self.ice_thickness / (2.0 * self.layer_target)
        area = self.domain_area
        vol_per_area = self.concentration * self.ice_thickness
        if vol_per_area <= 0.0:
            return r_target
        v_particle = vol_per_area * area / self.particle_budget
        r_budget = (3.0 * v_particle / (4.0 * math.pi)) ** (1.0 / 3.0)
        return max(r_target, r_budget)

    @cached_property
    def sheet_resolution(self) -> float:
        """``h_i / (2r)``: number of block diameters across the sheet."""
        return self.ice_thickness / (2.0 * self.particle_radius)

    @cached_property
    def sheet_resolved(self) -> bool:
        """False when the block size exceeds ``1.33 h_i`` (thickness unresolved)."""
        return self.sheet_resolution >= 0.75

    @cached_property
    def x_in(self) -> float:
        return -(self.structure_radius + self.upstream_diameters * self.diameter)

    @cached_property
    def x_out(self) -> float:
        return self.structure_radius + self.downstream_diameters * self.diameter

    @cached_property
    def half_width(self) -> float:
        return self.half_width_diameters * self.diameter

    @cached_property
    def domain_area(self) -> float:
        return (self.x_out - self.x_in) * 2.0 * self.half_width

    @cached_property
    def seabed(self) -> float:
        return -self.depth_thickness_ratio * self.ice_thickness

    @cached_property
    def equilibrium_height(self) -> float:
        return floating_equilibrium_height(self.particle_radius, self.rho_ice,
                                           self.rho_water)

    @cached_property
    def z_reference(self) -> float:
        """Top of an undisturbed floating block (the ice-surface reference)."""
        r = self.particle_radius
        return floating_equilibrium_height(r, self.rho_ice, self.rho_water) + r

    @cached_property
    def thrust_force(self) -> float:
        """Per-block thrust force [N] of the sheet pressure boundary.

        A block occupies a frontal share ``(2r)**2`` of the ice-sheet face, so a
        sheet pressure ``sigma`` pushes it with ``sigma (2r)**2``.
        """
        r = self.particle_radius
        return self.thrust_pressure * 4.0 * r * r

    @cached_property
    def thrust_band_front(self) -> float:
        """Downstream edge ``x`` of the thrust band."""
        return self.x_in + self.thrust_band_diameters * self.diameter

    @cached_property
    def particle_mass(self) -> float:
        r = self.particle_radius
        return self.rho_ice * 4.0 / 3.0 * math.pi * r ** 3

    @cached_property
    def normal_stiffness(self) -> float:
        r = self.particle_radius
        return self.particle_mass * G / (self.target_static_overlap * r)

    @cached_property
    def tangential_stiffness(self) -> float:
        return KT_RATIO * self.normal_stiffness

    @cached_property
    def damping_coefficient(self) -> float:
        e = min(max(self.restitution, 1e-6), 1.0 - 1e-12)
        zeta = abs(math.log(e)) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
        k = self.normal_stiffness
        m_eff = 0.5 * self.particle_mass
        return 2.0 * zeta * math.sqrt(k * m_eff)

    @cached_property
    def contact_time(self) -> float:
        """``pi sqrt(m_eff / k_n)``: half-period of a two-body contact."""
        m_eff = 0.5 * self.particle_mass
        return math.pi * math.sqrt(m_eff / self.normal_stiffness)

    @cached_property
    def timestep(self) -> float:
        return min(self.timestep_safety * self.contact_time, self.max_timestep)

    @cached_property
    def froude(self) -> float:
        return self.drift_speed / math.sqrt(G * self.ice_thickness)

    @cached_property
    def injection_rate(self) -> float:
        """Areal feed of the drifting field: blocks per second through x_in."""
        v_particle = 4.0 / 3.0 * math.pi * self.particle_radius ** 3
        return (self.concentration * self.ice_thickness * self.drift_speed *
                2.0 * self.half_width / v_particle)

    @cached_property
    def driving_force_limit(self) -> float:
        """Driving force the ice field delivers: ``sigma_drive 2W h_i`` [N]."""
        return self.driving_pressure * 2.0 * self.half_width * self.ice_thickness

    @cached_property
    def pile_speed_threshold(self) -> float:
        return max(self.pile_speed_fraction * self.drift_speed, 1e-9)

    @cached_property
    def pile_shell(self) -> float:
        return self.pile_shell_diameters * self.diameter

    @cached_property
    def capture_zone_radius(self) -> float:
        """Radius of the capture zone used for the crossing-flux mass ledger."""
        return self.structure_radius + self.pile_shell_diameters * self.diameter

    @cached_property
    def flux_time_scale(self) -> float:
        """Flux-limited pile build-up time ``~ D / (C_i v_i)``."""
        c = max(self.concentration, 0.02)
        v = max(self.drift_speed, 1e-6)
        return 0.8 * self.diameter / (c * v)

    def recommended_duration(self) -> float:
        """Physical duration: build-up time scale plus the feed transit."""
        transit = abs(self.x_in) / max(self.drift_speed, 1e-9)
        t_end = self.time_factor * self.flux_time_scale + 0.5 * transit
        return min(t_end, self.max_physical_time)

    def with_changes(self, **kw) -> "RubbleConfig":
        return replace(self, **kw)


# --------------------------------------------------------------------------
# growth-law fitting and geometry extraction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GrowthFit:
    """Result of fitting ``y(t) = y_inf (1 - exp(-t/tau))``."""

    inf: float
    tau: float
    r2: float
    n: int
    ok: bool
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "inf": self.inf,
            "tau": self.tau,
            "r2": self.r2,
            "n": self.n,
            "ok": self.ok,
            "message": self.message,
        }


def _exp_saturation(t, y_inf, tau):
    return y_inf * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


def fit_exponential_growth(t, y, *, min_points: int = 8) -> GrowthFit:
    """Least-squares fit of ``y(t) = y_inf (1 - exp(-t/tau))``.

    Returns the fitted plateau ``y_inf``, the growth time ``tau`` and the
    coefficient of determination ``R**2``.  Failure (too few points, zero
    amplitude, non-convergence) is returned as ``ok=False`` with a message --
    never as a fabricated number.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if len(t) < min_points:
        return GrowthFit(0.0, 0.0, 0.0, len(t), False, "too few points")
    amp = float(np.max(y) - np.min(y))
    scale = max(float(np.max(np.abs(y))), 1e-30)
    if amp <= 1e-9 * max(scale, 1.0):
        return GrowthFit(float(np.mean(y)), 0.0, 0.0, len(t), False,
                         "no growth (flat signal)")
    t0 = t - t[0]
    span = t0[-1] if t0[-1] > 0 else 1.0
    p0 = [float(y[-1]) if y[-1] > 0 else float(np.max(y)), 0.3 * span]
    bounds = ([0.0, 1.0e-6 * span], [10.0 * np.max(np.abs(y)) + 1e-12, 1.0e6 * span])
    try:
        popt, _ = curve_fit(_exp_saturation, t0, y, p0=p0, bounds=bounds,
                            maxfev=20000)
    except Exception as exc:  # pragma: no cover - numerical failure path
        return GrowthFit(0.0, 0.0, 0.0, len(t), False, f"{type(exc).__name__}")
    pred = _exp_saturation(t0, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return GrowthFit(float(popt[0]), float(popt[1]), float(r2), len(t), True)


@dataclass(frozen=True)
class PileGeometry:
    """Instantaneous rubble-pile geometry from a particle cloud."""

    mass: float
    count: int
    upstream_length: float
    width: float
    height: float
    x_tip: float
    y_edge: float
    z_top: float

    def as_dict(self) -> dict:
        return {
            "mass": self.mass,
            "count": self.count,
            "upstream_length": self.upstream_length,
            "width": self.width,
            "height": self.height,
            "x_tip": self.x_tip,
            "y_edge": self.y_edge,
            "z_top": self.z_top,
        }


#: Percentile used for the pile envelope (robust to stray flyers).
PILE_PERCENTILE = 98.0


def extract_pile_geometry(pos: np.ndarray, vel: np.ndarray,
                          cfg: RubbleConfig) -> PileGeometry:
    """Rubble-pile envelope from particle positions.

    A block belongs to the pile when it sits inside the capture shell,
    ``hypot(x, y) <= R + shell``, and is *resident*, ``|v| < f v_i`` (otherwise
    it is drifting ice, not rubble).  Envelopes use the :data:`PILE_PERCENTILE`
    percentile so a single escaping flyer cannot define the pile.

    Definitions (all relative to the structure / undisturbed ice surface):

    * ``L_p = (-R) - x_2%``  upstream length in front of the cylinder surface;
    * ``B_p = 2 |y|_98%``    total pile width;
    * ``H_p = z_98% + r - z_ref`` pile top above the undisturbed ice surface.
    """
    p = np.asarray(pos, dtype=float)
    if p.size == 0:
        return PileGeometry(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v = np.asarray(vel, dtype=float)
    d = np.hypot(p[:, 0], p[:, 1])
    speed = np.linalg.norm(v, axis=1) if v.size else np.zeros(len(p))
    mask = (d <= cfg.structure_radius + cfg.pile_shell) & \
           (speed < cfg.pile_speed_threshold)
    if not mask.any():
        return PileGeometry(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pile = p[mask]
    mass = float(mask.sum()) * cfg.particle_mass
    x_tip = float(np.percentile(pile[:, 0], 100.0 - PILE_PERCENTILE))
    up = -cfg.structure_radius - x_tip
    y_edge = float(np.percentile(np.abs(pile[:, 1]), PILE_PERCENTILE))
    z_top = float(np.percentile(pile[:, 2], PILE_PERCENTILE)) + cfg.particle_radius
    return PileGeometry(
        mass=mass,
        count=int(mask.sum()),
        upstream_length=max(up, 0.0),
        width=2.0 * y_edge,
        height=max(z_top - cfg.z_reference, 0.0),
        x_tip=x_tip,
        y_edge=y_edge,
        z_top=z_top,
    )


# --------------------------------------------------------------------------
# shielding baselines
# --------------------------------------------------------------------------
#: Drag coefficient of a circular cylinder in potential flow -- the declared
#: reference value of the momentum-flux baseline (no fitting, no tuning).
CYLINDER_DRAG_COEFFICIENT = 2.0


def momentum_baseline_pressure(concentration: float, drift_speed: float, *,
                               rho_ice: float = 900.0,
                               drag_coefficient: float = CYLINDER_DRAG_COEFFICIENT) -> float:
    """Free-stream momentum-flux pressure of a drifting broken-ice field [Pa].

    The ice field carries the momentum flux ``rho_i C_i h_i v_i**2`` per unit
    width.  On a cylinder of diameter ``D`` and height ``h_i`` (projected area
    ``D h_i``) a drag coefficient ``C_d`` turns it into

    .. math:: p_0 = \\tfrac{1}{2} C_d rho_i C_i v_i^2 ,

    i.e. the pressure the ice field would exert *without any accumulation*.
    This is the literature-anchored baseline of the shielding factor.  It does
    **not** contain a model measurement; ``C_d`` is a declared reference value,
    so the baseline answers "is the front pressure above the free-stream
    momentum flux", not "is it above a measured bare-cylinder load".

    Parameters
    ----------
    concentration : float
        Areal ice concentration ``C_i`` [-].
    drift_speed : float
        Drift speed ``v_i`` [m/s].
    rho_ice : float, optional
        Ice density [kg/m^3].
    drag_coefficient : float, optional
        Cylinder drag coefficient [-], default :data:`CYLINDER_DRAG_COEFFICIENT`.
    """
    c = float(concentration)
    v = float(drift_speed)
    if c <= 0.0 or v <= 0.0:
        return 0.0
    return 0.5 * float(drag_coefficient) * float(rho_ice) * c * v * v


def shielding_ratio(p_front: float, p_baseline: float) -> float:
    """``R = p_front / p_0``; ``nan`` (never a fabricated number) if undefined."""
    try:
        p1 = float(p_front)
        p0 = float(p_baseline)
    except (TypeError, ValueError):
        return float("nan")
    if not (p0 > 0.0) or not math.isfinite(p1):
        return float("nan")
    return p1 / p0


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------
class RubbleAccumulation:
    """Drifting broken-ice field driven against a fixed circular cylinder.

    Driving boundary condition
    --------------------------
    The ice field is compressed against the structure by a **force-limited servo
    platen** at the upstream face: the platen advances at the drift speed
    ``v_i`` while the resistance it meets is below ``sigma_drive 2W h_i`` and
    stalls as the resistance approaches that value (a smooth
    ``v = v_i (1 - F/F_limit)`` yield law).  ``sigma_drive`` is the ice-field
    driving pressure (wind/current drag integrated over the field), the classic
    ice-engineering input.  This matters: an infinite-strength feed would crush
    the rubble without bound, which is exactly what we must *not* model.

    Blocks leaving the downstream or lateral faces are removed and booked as
    *bypass* mass, so the pile reaches a genuine steady state.
    """

    HISTORY_KEYS = ("time", "pile_mass", "L_p", "B_p", "H_p", "p_front",
                    "p_rear", "n_blocks", "mass_injected", "mass_bypass",
                    "zone_inventory", "zone_flux_in", "zone_flux_out",
                    "contact_force", "wall_force", "wall_advance",
                    "inflow_mass", "inventory_mass")

    def __init__(self, config: RubbleConfig | None = None) -> None:
        self.cfg = config or RubbleConfig()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.n = 0
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.omega = np.zeros((0, 3))
        self._tkeys = np.zeros(0, dtype=np.int64)
        self._tang = np.zeros((0, 3))
        self.time = 0.0
        self.steps = 0

        # -- driving platen --
        self.wall_x = c.x_in
        self._pending = 0.0
        self.feed_enabled = True
        self.feed_steps = 0
        self.feed_blocked_steps = 0
        self.injected_blocks = 0
        self.wall_force = 0.0
        self.wall_force_avg = 0.0
        self.wall_advance = 0.0

        # -- ledgers --
        self.mass_initial = 0.0
        self.mass_injected = 0.0
        self.mass_exited = 0.0
        self.mass_bypass = 0.0
        self.mass_swept = 0.0
        self.zone_flux_in = 0.0
        self.zone_flux_out = 0.0
        self.blocks_removed = 0
        self._prev_d = np.zeros(0)

        # -- force bookkeeping --
        self.front_impulse = 0.0
        self.rear_impulse = 0.0
        self.front_time = 0.0
        self.max_overlap_ratio = 0.0
        self._last_contacts = 0
        self._last_wall_force = 0.0
        self._last_front_force = 0.0
        self._last_rear_force = 0.0

        self._z_eq = floating_equilibrium_height(c.particle_radius, c.rho_ice,
                                                 c.rho_water)
        self.history: dict[str, list[float]] = {k: [] for k in self.HISTORY_KEYS}
        self._initialise_field()
        self.mass_initial = self.n * c.particle_mass
        self.mass_injected = self.mass_initial

    # -- helpers ----------------------------------------------------------
    def _initialise_field(self) -> None:
        """Place the drifting ice field: uniform block sheet, shell empty.

        Triangular in-plane lattice (nearest-neighbour 2.10 r, lattice-line
        spacing ``sqrt(3)/2`` of that, alternate lines offset by half a cell) on
        as many layers as ``C_i h_i`` needs; layers are offset so vertical
        neighbours never coincide.  The capture shell around the structure is
        left empty so the pile geometry is exactly zero at ``t = 0``.
        """
        c = self.cfg
        r = c.particle_radius
        if c.concentration <= 0.0:
            return
        v_particle = 4.0 / 3.0 * math.pi * r ** 3
        n_target = int(round(c.concentration * c.ice_thickness *
                             c.domain_area / v_particle))
        if n_target <= 0:
            return
        spacing = 2.10 * r
        line = spacing * math.sqrt(3.0) / 2.0
        z_eq = c.equilibrium_height
        margin = 1.2 * r
        zone = c.capture_zone_radius + 1.5 * r
        ny = max(1, int(2.0 * c.half_width / spacing))
        sites = []
        i = 0
        x = c.x_in + 0.5 * line
        while x <= c.x_out - margin:
            y_off = 0.5 * spacing if i % 2 else 0.0
            for j in range(ny):
                y = -c.half_width + (j + 0.5) * spacing + y_off
                if abs(y) > c.half_width - margin:
                    continue
                if math.hypot(x, y) <= zone:
                    continue
                sites.append((x, y))
            i += 1
            x += line
        if not sites:
            return
        per_layer = len(sites)
        n_layers = max(1, int(round(n_target / per_layer)))
        keep_frac = min(1.0, n_target / (per_layer * n_layers))
        pts, vels = [], []
        for k in range(n_layers):
            shift_x = 0.5 * line * (k % 2)
            shift_y = 0.5 * spacing * (k % 2)
            z = z_eq + (k - 0.5 * (n_layers - 1)) * 2.10 * r
            for (sx, sy) in sites:
                if keep_frac < 1.0 and self.rng.random() > keep_frac:
                    continue
                pts.append((sx + shift_x + self.rng.uniform(-0.02, 0.02) * spacing,
                            sy + shift_y + self.rng.uniform(-0.02, 0.02) * spacing,
                            z + self.rng.uniform(-0.02, 0.02) * r))
                vels.append((c.drift_speed, 0.0, 0.0))
        if not pts:
            return
        self.pos = np.array(pts, dtype=float)
        self.vel = np.array(vels, dtype=float)
        self.n = len(self.pos)
        self.omega = np.zeros((self.n, 3))
        self._prev_d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        #: residence timer [s] of the sweeper baseline, one entry per block
        self._dwell = np.zeros(len(self.pos), dtype=float)

    @property
    def particle_mass(self) -> float:
        return self.cfg.particle_mass

    def zone_inventory(self) -> float:
        if self.n == 0:
            return 0.0
        d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        return float(np.count_nonzero(d <= self.cfg.capture_zone_radius)) * \
            self.cfg.particle_mass

    def pile_geometry(self) -> PileGeometry:
        return extract_pile_geometry(self.pos, self.vel, self.cfg)

    def mass_in_domain(self) -> float:
        return self.n * self.cfg.particle_mass

    def inflow_mass(self) -> float:
        """Ice mass fed through the upstream face since ``t = 0`` [kg]."""
        return self.mass_injected - self.mass_initial

    # -- dynamics ---------------------------------------------------------
    def _contact_forces(self, tree=None):
        """Particle-particle, structure and platen forces (fully vectorised).

        Returns ``(force, torque, n_contacts)``; the rigid structure and the
        platen are handled by :meth:`_resolve_structure_and_platen`.
        The tangential law keeps a per-pair 3-vector of accumulated tangential
        displacement, matched between steps by a sorted pair-key lookup, so the
        kernel never loops over contacts in python.
        """
        c = self.cfg
        r = c.particle_radius
        n = self.n
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        n_contacts = 0

        if n >= 2:
            iu, ju = build_pairs(self.pos, r)
            m_pairs = len(iu)
            if m_pairs:
                dvec = self.pos[ju] - self.pos[iu]
                dist2 = np.einsum("ij,ij->i", dvec, dvec)
                active = dist2 < (2.0 * r) ** 2
                if active.any():
                    iu_a, ju_a = iu[active], ju[active]
                    dvec_a = dvec[active]
                    dist_a = np.sqrt(dist2[active])
                    overlap = 2.0 * r - dist_a
                    n_contacts = int(overlap.size)
                    self.max_overlap_ratio = max(self.max_overlap_ratio,
                                                 float(overlap.max()) / r)
                    nvec = dvec_a / np.maximum(dist_a, 1e-30)[:, None]

                    keys = iu * n + ju
                    order = np.argsort(keys, kind="stable")
                    sk = keys[order]
                    inv = np.empty(m_pairs, dtype=np.int64)
                    inv[order] = np.arange(m_pairs, dtype=np.int64)
                    if len(self._tkeys):
                        idx = np.searchsorted(self._tkeys, sk)
                        idx_c = np.minimum(idx, len(self._tkeys) - 1)
                        hit = self._tkeys[idx_c] == sk
                    else:
                        idx_c = np.zeros(0, dtype=np.int64)
                        hit = np.zeros(0, dtype=bool)
                    tang = np.zeros((m_pairs, 3))
                    if hit.any():
                        tang[hit] = self._tang[idx_c[hit]]

                    v_rel = self.vel[ju_a] - self.vel[iu_a]
                    v_n = np.einsum("ij,ij->i", v_rel, nvec)
                    f_n = np.maximum(c.normal_stiffness * overlap -
                                     c.damping_coefficient * v_n, 0.0)
                    v_t = v_rel - v_n[:, None] * nvec
                    if c.rotation:
                        surf_i = np.cross(self.omega[iu_a], nvec * r)
                        surf_j = np.cross(self.omega[ju_a], -nvec * r)
                        v_t = v_t + (surf_j - surf_i)

                    slot = inv[active]
                    tang_a = tang[slot] + v_t * c.timestep
                    t_norm = np.linalg.norm(tang_a, axis=1)
                    cap = c.mu_particle * f_n / c.tangential_stiffness
                    scale = np.where(t_norm > cap, cap / np.maximum(t_norm, 1e-30),
                                     1.0)
                    tang_a = tang_a * scale[:, None]
                    tang[slot] = tang_a
                    f_t = -c.tangential_stiffness * tang_a

                    f_total = f_n[:, None] * nvec + f_t
                    for d in range(3):
                        force[:, d] += (
                            np.bincount(iu_a, weights=-f_total[:, d], minlength=n) +
                            np.bincount(ju_a, weights=f_total[:, d], minlength=n))
                    if c.rotation:
                        t_i = np.cross(-nvec * r, f_t)
                        for d in range(3):
                            torque[:, d] += (
                                np.bincount(iu_a, weights=t_i[:, d], minlength=n) +
                                np.bincount(ju_a, weights=t_i[:, d], minlength=n))
                    self._tkeys = sk
                    self._tang = tang

        return force, torque, n_contacts

    def _feed(self, tree=None) -> None:
        """Feed the drifting field through the upstream face (open system).

        Blocks enter at ``x = x_in`` with the drift velocity, at the areal feed
        rate ``C_i h_i v_i`` per unit width, provided the structure load is
        below the driving capacity (:attr:`RubbleConfig.driving_force_limit`).
        Insertion is refused when the inlet region is crowded, so the feed
        throttles itself against a jam instead of pushing through it.
        """
        c = self.cfg
        if not getattr(self, "feed_enabled", True):
            return
        if c.injection_rate <= 0.0:
            return
        r = c.particle_radius
        self._pending += c.injection_rate * c.timestep
        if self._pending < 1.0:
            return
        if tree is None and self.n:
            tree = cKDTree(self.pos)
        n_new = 0
        while self._pending >= 1.0 and n_new < 40:
            placed = False
            for _ in range(8):
                cand = np.array([[c.x_in + self.rng.uniform(0.3, 1.5) * r,
                                  self.rng.uniform(-c.half_width + 2.0 * r,
                                                   c.half_width - 2.0 * r),
                                  self._z_eq + self.rng.uniform(-0.9, 0.9) * r]])
                ok = True
                if tree is not None and tree.n:
                    dist, _ = tree.query(cand, k=1)
                    ok = bool(dist[0] > 2.05 * r)
                if ok and n_new:
                    dd = np.linalg.norm(self.pos[-n_new:] - cand, axis=1)
                    ok = bool(dd.size == 0 or dd.min() > 2.05 * r)
                if ok:
                    placed = True
                    break
            if not placed:
                self.feed_blocked_steps += 1
                break
            self.pos = np.vstack([self.pos, cand])
            self.vel = np.vstack([self.vel,
                                  np.array([[c.drift_speed, 0.0, 0.0]])])
            self.omega = np.vstack([self.omega, np.zeros((1, 3))])
            self._prev_d = np.append(self._prev_d,
                                     math.hypot(cand[0, 0], cand[0, 1]))
            self._dwell = np.append(self._dwell, 0.0)
            self.n += 1
            n_new += 1
            self._pending -= 1.0
            self.mass_injected += c.particle_mass
            self.injected_blocks += 1

    def _resolve_structure_and_platen(self) -> float:
        """Hard-constraint resolution of the structure and the driving platen.

        Penalty contacts are unusable for a *rigid* structure: a jammed block
        can be pushed inside the cylinder and then ejected with an arbitrarily
        large spring force (measured: 68 m/s blocks, overlap saturating at the
        block diameter).  The structure and the platen are therefore treated as
        impenetrable surfaces: penetrating blocks are projected back onto the
        surface, the inward normal velocity is reversed with restitution ``e``
        and the tangential velocity loses a Coulomb-limited impulse
        ``mu_structure |dv_n|``.  The load on the structure is then the
        momentum change divided by ``dt`` -- the physical ice load, and exactly
        the quantity the shielding factor is built from.

        Returns the platen (driving) force.
        """
        c = self.cfg
        if self.n == 0:
            return 0.0
        r = c.particle_radius
        m = c.particle_mass
        dt = c.timestep
        n = self.n
        front = 0.0
        rear = 0.0

        # ---- rigid cylinder -------------------------------------------
        d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        r_eff = c.structure_radius + r
        hit = d < r_eff
        if hit.any():
            idx = np.flatnonzero(hit)
            dd = np.maximum(d[idx], 1e-12)
            rad = self.pos[idx, :2] / dd[:, None]
            self.pos[idx, 0] = rad[:, 0] * r_eff
            self.pos[idx, 1] = rad[:, 1] * r_eff
            v = self.vel[idx]
            vn = v[:, 0] * rad[:, 0] + v[:, 1] * rad[:, 1]
            dvn = np.where(vn < 0.0, -(1.0 + c.restitution) * vn, 0.0)
            v[:, 0] += dvn * rad[:, 0]
            v[:, 1] += dvn * rad[:, 1]
            vn2 = v[:, 0] * rad[:, 0] + v[:, 1] * rad[:, 1]
            vt = v.copy()
            vt[:, 0] -= vn2 * rad[:, 0]
            vt[:, 1] -= vn2 * rad[:, 1]
            vt_norm = np.linalg.norm(vt, axis=1)
            dv_t = np.minimum(c.mu_structure * np.abs(dvn), vt_norm)
            scale = np.where(vt_norm > 1e-12, dv_t / np.maximum(vt_norm, 1e-12), 0.0)
            v -= scale[:, None] * vt
            self.vel[idx] = v
            # momentum change of the blocks; the force ON the structure is the
            # negative of it, so a loaded structure reads a positive pressure
            jx = m * (dvn * rad[:, 0] - scale * vt[:, 0])
            upstream = self.pos[idx, 0] < 0.0
            front = -float(np.sum(jx[upstream])) / dt
            rear = -float(np.sum(jx[~upstream])) / dt
        self._last_front_force = front
        self._last_rear_force = rear

        # ---- driving platen (rigid plane at wall_x) -------------------
        # ---- lateral free-slip walls (flume idealisation) --------------
        wy = c.half_width - r
        over = np.abs(self.pos[:, 1]) > wy
        if over.any():
            self.pos[over, 1] = np.sign(self.pos[over, 1]) * wy
            self.vel[over, 1] = -self.vel[over, 1]

        return 0.0

    def _environment_forces(self) -> np.ndarray:
        """Buoyancy, weight and water drag (anisotropic time scales)."""
        c = self.cfg
        n = self.n
        f = np.zeros((n, 3))
        if n == 0:
            return f
        m = c.particle_mass
        v_sub = submerged_volume(self.pos[:, 2], c.particle_radius)
        f[:, 2] += c.rho_water * G * v_sub - m * G
        target = np.zeros((n, 3))
        target[:, 0] = c.drift_speed
        f += m * (target - self.vel) / np.array(
            [c.drag_time, c.drag_time, c.vertical_drag_time])
        if c.thrust_pressure > 0.0:
            # Sheet thrust: the semi-infinite ice sheet behind the feed band is
            # not modelled; its effect is this pressure boundary.  The thrust
            # acts only while the block is slower than the sheet (``v_x < v_i``):
            # an ice sheet drifts at v_i and cannot accelerate a block beyond
            # its own speed.  Without this switch the pressure would pump a
            # dilute stream to tens of m/s (measured: p_front ~ 6-16 kPa and
            # blocks launched 1.4-3 m into the air at sigma = 20 kPa), i.e. the
            # boundary condition would inject energy instead of transmitting a
            # jam pressure.
            band = (self.pos[:, 0] <= c.thrust_band_front) & \
                   (self.vel[:, 0] < c.drift_speed)
            if band.any():
                f[band, 0] += c.thrust_force
        return f

    def _sweeper_mask(self) -> np.ndarray:
        """Blocks the *sweeper* baseline removes (cleared-cylinder control).

        A block is swept once it has been **resident** -- ``|v| <
        pile_speed_fraction * v_i`` -- inside the capture shell for
        ``sweep_dwell_time``.  Blocks that are merely passing keep their
        momentum, and a block that loads the structure delivers its contact
        impulse for the whole dwell window before it is removed, so the
        baseline pressure ``p_0`` is a real (non-zero) bare-cylinder load rather
        than the value of an instantaneous deletion.
        """
        c = self.cfg
        if self.n == 0:
            return np.zeros(0, dtype=bool)
        d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        speed = np.linalg.norm(self.vel, axis=1)
        shell = c.structure_radius + c.sweep_shell_radii * c.particle_radius
        resident = (d <= shell) & \
                   (speed < c.pile_speed_fraction * max(c.drift_speed, 1e-9))
        self._dwell = np.where(resident, self._dwell + c.timestep, 0.0)
        return self._dwell >= c.sweep_dwell_time

    def _remove_mask(self, mask: np.ndarray) -> None:
        c = self.cfg
        if not mask.any() or self.n == 0:
            return
        keep = ~mask
        d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        inside_zone = mask & (d <= c.capture_zone_radius)
        self.zone_flux_out += float(np.count_nonzero(inside_zone)) * c.particle_mass
        get_past = mask & (self.pos[:, 0] > 0.5 * c.structure_radius)
        self.mass_bypass += float(np.count_nonzero(get_past)) * c.particle_mass
        self.mass_exited += float(np.count_nonzero(mask)) * c.particle_mass
        self.blocks_removed += int(np.count_nonzero(mask))
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.omega = self.omega[keep]
        self._prev_d = self._prev_d[keep]
        if len(self._dwell) == len(keep):
            self._dwell = self._dwell[keep]
        else:  # pragma: no cover - defensive resync after an inconsistent edit
            self._dwell = np.zeros(int(keep.sum()), dtype=float)
        self.n = int(keep.sum())
        # pair-key state is index based: drop it on re-indexing (a small,
        # documented tangential-history reset when a block leaves the domain)
        self._tkeys = np.zeros(0, dtype=np.int64)
        self._tang = np.zeros((0, 3))

    def step(self) -> None:
        c = self.cfg
        tree = cKDTree(self.pos) if self.n else None

        # load-limited feed: the field enters at its drift speed only while the
        # structure load is below the driving capacity sigma_drive 2W h_i; above
        # it the field stalls (an infinite-strength feed would crush the rubble)
        cap = c.driving_force_limit
        load = abs(self._last_front_force)
        self.feed_enabled = bool(cap <= 0.0 or load <= cap)
        if self.feed_enabled:
            self.feed_steps += 1
        self._feed(tree)
        if self.n:
            tree = cKDTree(self.pos)

        force, torque, n_contacts = self._contact_forces(tree)
        if self.n:
            force += self._environment_forces()

        if self.n:
            m = c.particle_mass
            self.vel += force / m * c.timestep
            if c.rotation:
                inertia = 0.4 * m * c.particle_radius ** 2
                self.omega += torque / inertia * c.timestep
                self.omega *= math.exp(-c.timestep / c.angular_damping_time)
                w = np.linalg.norm(self.omega, axis=1)
                w_max = 20.0  # rad/s cap, one bad contact cannot blow up the run
                over = w > w_max
                if over.any():
                    self.omega[over] *= (w_max / w[over])[:, None]
            self.pos += self.vel * c.timestep

        # capture-zone crossing fluxes (measured on the *displaced* positions,
        # against the distance stored at the end of the previous step, so the
        # integral of the flux is independent of the inventory)
        if self.n:
            d_new = np.hypot(self.pos[:, 0], self.pos[:, 1])
            enter = (self._prev_d > c.capture_zone_radius) & \
                    (d_new <= c.capture_zone_radius)
            leave = (self._prev_d <= c.capture_zone_radius) & \
                    (d_new > c.capture_zone_radius)
            self.zone_flux_in += float(np.count_nonzero(enter)) * c.particle_mass
            self.zone_flux_out += float(np.count_nonzero(leave)) * c.particle_mass

        wall_force = self._resolve_structure_and_platen()
        self.front_impulse += self._last_front_force * c.timestep
        self.rear_impulse += self._last_rear_force * c.timestep
        self.front_time += c.timestep
        self.wall_force = wall_force
        self.wall_force_avg = 0.9 * self.wall_force_avg + 0.1 * wall_force

        self.time += c.timestep
        self.steps += 1
        self._last_wall_force = wall_force

        if self.n:
            out = (self.pos[:, 0] > c.x_out) | (self.pos[:, 2] < c.seabed)
            if c.sweep_shell_radii > 0.0:
                swept = self._sweeper_mask()
                if swept.any():
                    self.mass_swept += float(np.count_nonzero(swept)) * c.particle_mass
                    out = out | swept
            if out.any():
                self._remove_mask(out)
            self._prev_d = np.hypot(self.pos[:, 0], self.pos[:, 1])
        self._last_contacts = n_contacts

    # -- sampling ---------------------------------------------------------
    def sample(self) -> None:
        c = self.cfg
        geo = self.pile_geometry()
        norm = max(c.diameter * c.ice_thickness, 1e-12)
        denom = self.front_time if self.front_time > 0 else 1.0
        h = self.history
        h["time"].append(self.time)
        h["pile_mass"].append(geo.mass)
        h["L_p"].append(geo.upstream_length)
        h["B_p"].append(geo.width)
        h["H_p"].append(geo.height)
        h["p_front"].append(self.front_impulse / denom / norm)
        h["p_rear"].append(-self.rear_impulse / denom / norm)
        h["n_blocks"].append(float(self.n))
        h["mass_injected"].append(self.mass_injected)
        h["mass_bypass"].append(self.mass_bypass)
        h["zone_inventory"].append(self.zone_inventory())
        h["zone_flux_in"].append(self.zone_flux_in)
        h["zone_flux_out"].append(self.zone_flux_out)
        h["contact_force"].append(float(getattr(self, "_last_contacts", 0)))
        h["wall_force"].append(self.wall_force)
        h["wall_advance"].append(float(self.feed_steps))
        h["inflow_mass"].append(self.inflow_mass())
        h["inventory_mass"].append(self.mass_in_domain())

    def run(self, t_end: float | None = None,
            sample_every: int | None = None) -> dict:
        c = self.cfg
        if t_end is None:
            t_end = c.recommended_duration()
        n_steps = int(min(c.max_steps, max(10, math.ceil(t_end / c.timestep))))
        if sample_every is None:
            sample_every = max(1, n_steps // max(c.samples, 1))
        self.sample()
        for k in range(n_steps):
            self.step()
            if (k + 1) % sample_every == 0 or k + 1 == n_steps:
                self.sample()
        self.sample()
        return self.history

    # -- post-processing --------------------------------------------------
    def steady_window(self, fraction: float = 0.25) -> dict:
        t = np.asarray(self.history["time"])
        if len(t) < 4:
            raise RuntimeError("run too short to define a steady window")
        t0 = t[-1] - fraction * (t[-1] - t[0])
        sel = t >= t0
        out = {}
        for key in ("p_front", "p_rear", "pile_mass", "L_p", "B_p", "H_p",
                    "n_blocks", "contact_force", "wall_force"):
            out[key] = float(np.mean(np.asarray(self.history[key])[sel]))
        out["t_start"] = float(t0)
        out["t_end"] = float(t[-1])
        return out

    def mass_ledger(self) -> dict:
        """Mass conservation, two independent routes.

        1. *Kinematic inflow vs inventory*: the mass the platen swept in,
           ``rho_i C_i h_i 2W dx``, compared with what is actually in the domain
           plus what left it.  The closure error therefore tests the
           concentration/bookkeeping chain, not an identity.
        2. *Crossing flux vs inventory*: the mass flux across the capture-zone
           boundary integrated over the run, compared with the zone inventory
           (an Euler-type integration check on the geometry extraction side).
        """
        c = self.cfg
        scale = max(self.mass_initial, c.particle_mass)
        pile = self.pile_geometry().mass
        inventory = self.mass_in_domain()
        swept = self.inflow_mass()
        partition_err = abs(self.mass_initial + swept - inventory - self.mass_exited)
        zone_now = self.zone_inventory()
        zone_expected = self.zone_flux_in - self.zone_flux_out
        zone_scale = max(self.zone_flux_in, c.particle_mass)
        return {
            "mass_initial": self.mass_initial,
            "mass_inflow": swept,
            "mass_in_domain": inventory,
            "mass_exited": self.mass_exited,
            "mass_bypass": self.mass_bypass,
            "mass_swept": self.mass_swept,
            "pile_mass": pile,
            "free_mass": inventory - pile,
            "closure_error": partition_err / scale,
            "zone_inventory": zone_now,
            "zone_flux_in": self.zone_flux_in,
            "zone_flux_out": self.zone_flux_out,
            "zone_balance_expected": zone_expected,
            "zone_closure_error": abs(zone_now - zone_expected) / zone_scale,
            "n_blocks": self.n,
            "blocks_removed": self.blocks_removed,
            "feed_duty": (self.feed_steps / max(self.steps, 1)),
            "feed_blocked_steps": self.feed_blocked_steps,
            "injected_blocks": self.injected_blocks,
            "driving_force_limit": c.driving_force_limit,
        }


# --------------------------------------------------------------------------
# single-case driver
# --------------------------------------------------------------------------
def run_rubble_case(config: RubbleConfig, *, t_end: float | None = None,
                    fit_fraction: float = 1.0) -> dict:
    """Run one case end-to-end and return a JSON-serialisable record.

    Contains the geometry time series, the three exponential growth fits, the
    steady-window averages, the mass ledger and the resolution flags.
    """
    sim = RubbleAccumulation(config)
    sim.run(t_end=t_end)
    h = sim.history
    t = np.asarray(h["time"])
    fits = {}
    for key, label in (("L_p", "L"), ("B_p", "B"), ("H_p", "H")):
        y = np.asarray(h[key])
        # the fit window starts when the drifting ice first reaches the
        # structure (first sample with a nonzero pile signal), so the feed
        # transit is not mistaken for a growth law
        nz = np.flatnonzero(y > 0.02 * max(np.max(y), 1e-30)) if len(y) else []
        if len(nz):
            start = max(int(nz[0]) - 1, 0)
            start = int(start + (1.0 - fit_fraction) * (len(t) - start))
            fits[label] = fit_exponential_growth(t[start:], y[start:]).as_dict()
            fits[label]["fit_window_start"] = float(t[start])
        else:
            fits[label] = fit_exponential_growth(t, y).as_dict()
            fits[label]["fit_window_start"] = float(t[0]) if len(t) else 0.0
    steady = sim.steady_window()
    cfg = config
    plateau = {}
    for label, key in (("L", "L_p"), ("B", "B_p"), ("H", "H_p")):
        y = np.asarray(h[key])
        inf = fits[label]["inf"]
        plateau[label] = float(y[-1] / inf) if inf > 0 else 0.0
    return {
        "config": {
            "diameter": cfg.diameter,
            "ice_thickness": cfg.ice_thickness,
            "concentration": cfg.concentration,
            "drift_speed": cfg.drift_speed,
            "froude": cfg.froude,
            "particle_radius": cfg.particle_radius,
            "layer_target": cfg.layer_target,
            "sheet_resolution": cfg.sheet_resolution,
            "sheet_resolved": cfg.sheet_resolved,
            "n_blocks_target": cfg.particle_budget,
            "timestep": cfg.timestep,
            "steps": sim.steps,
            "physical_time": sim.time,
            "sweep_shell_radii": cfg.sweep_shell_radii,
            "thrust_pressure": cfg.thrust_pressure,
            "thrust_force": cfg.thrust_force,
            "rotation": cfg.rotation,
            "seed": cfg.seed,
        },
        "fits": fits,
        "steady": steady,
        "plateau_fraction": plateau,
        "converged": bool(all(v >= 0.95 for v in plateau.values())),
        "capped_by_time": bool(sim.time >= 0.999 * cfg.max_physical_time),
        "ledger": sim.mass_ledger(),
        "history": {k: list(v) for k, v in h.items()},
    }