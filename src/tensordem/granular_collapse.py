"""Granular column collapse with a mean-field ``mu(I)`` rheology and an
explicit potential-energy / dissipation ledger.

Physical background
-------------------
A rectangular column of dry grains of initial length ``L0`` (along the flow) and
height ``H`` is released on a rough horizontal floor.  The final geometry is set
by the initial aspect ratio ``a = H / L0`` through power laws (Lube, Huppert,
Sparks & Freundt, JFM 508, 175-199, 2004 for axisymmetric columns; Lube et al.,
PRE 72, 041301, 2005 and Lajeunesse, Monnier & Homsy, Phys. Fluids 17, 103302,
2005 for the rectangular channel used here):

* axisymmetric, ``a > 1.7``:  ``r_inf = r_i (1 + 1.6 a^(1/2))``  - exponent 1/2
  (Lube et al. 2004, quoted verbatim in the abstract: "for 1.7 < a the
  corresponding relationship is r_inf = r_i(1 + 1.6 a^(1/2))");
* rectangular channel: ``dL / L0 ~ a`` for ``a <~ 3`` and ``~ a^(2/3)`` for
  ``a >~ 3`` (Lajeunesse et al. 2005).

The DEM side runs the collapse on
:class:`tensordem.granular2d.engine.DiskSystem` (Hertz-Mindlin contacts with
normal damping, Coulomb friction and optional rolling resistance) and closes the
contact friction coefficient with the ``mu(I)`` rheology of Jop, Forterre &
Pouliquen (Nature 441, 727, 2006)

    ``mu(I) = mu_s + (mu_2 - mu_s) / (1 + I_0 / I)``,
    ``I = gamma_dot d / sqrt(P / rho_p)``.

``I`` is evaluated *at the scale of the flowing layer* (flow-averaged shear rate
from the measured velocity profile and the measured basal pressure) and the
resulting ``mu`` is imposed on every contact for the next step.  That is a
**mean-field** closure - a fully local per-contact ``I`` needs the contact loop of
the engine to accept a spatially varying friction coefficient (declared as an
extension, not silently claimed).

Accounting
----------
:class:`EnergyLedger` measures every dissipation channel *forwards* from the
contact kinematics each step (normal dashpot ``c_n max(v_n,0)^2``, Coulomb
friction ``f_t |v_t|``), stores the recoverable elastic energies
``k_n ov^2 / 2`` and ``f_t^2 / (2 k_t)``, and compares the sum with the released
gravitational potential energy

    ``E_pot(0) = E_pot(t) + E_kin(t) + E_elastic(t) + E_dissipated(t)``.

Rolling resistance is off by default (:attr:`CollapseCase.mu_rolling`) so that
every dissipating channel is measurable from the contact records; the consequence
(somewhat longer runout than a rolling-resistant packing) is reported with the
result.

Units are SI (m, kg, s, N, J).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from .granular2d import DiskConfig, DiskSystem

__all__ = [
    "AXISYMMETRIC_EXPONENT",
    "JOP_I_0",
    "JOP_MU_2",
    "JOP_MU_S",
    "LAJEUNESSE_RECTANGULAR_EXPONENT_HIGH",
    "LAJEUNESSE_RECTANGULAR_EXPONENT_LOW",
    "CollapseCase",
    "CollapseResult",
    "EnergyLedger",
    "build_column",
    "collapse_energy_ledger",
    "column_collapse",
    "contact_keys",
    "contact_kinematics",
    "deposit_front",
    "fit_power_law",
    "granular_collapse_calculation",
    "jop_mu_I",
    "local_inertial_number",
    "potential_energy",
    "rolling_angle",
    "rolling_stiffness",
    "spring_energies",
    "runout_scan",
    "scaling_comparison",
]

#: Lube et al. (2004) axisymmetric exponent for ``a > 1.7``.
AXISYMMETRIC_EXPONENT = 0.5
#: Lajeunesse et al. (2005) rectangular channel: ``dL/L0 ~ a`` for ``a < 3``.
LAJEUNESSE_RECTANGULAR_EXPONENT_LOW = 1.0
#: Lajeunesse et al. (2005) rectangular channel: ``dL/L0 ~ a^(2/3)`` for ``a > 3``.
LAJEUNESSE_RECTANGULAR_EXPONENT_HIGH = 2.0 / 3.0
#: ``mu(I)`` parameters of Jop et al. (2006) for glass beads / sand.
JOP_MU_S = 0.36
JOP_MU_2 = 0.84
JOP_I_0 = 0.30


class CollapseError(ValueError):
    """Raised for unphysical collapse inputs."""


def jop_mu_I(inertial_number, mu_s: float = JOP_MU_S, mu_2: float = JOP_MU_2,
             i_0: float = JOP_I_0):
    """Jop et al. (2006) ``mu(I)`` law (elementwise for arrays)."""
    if not (0.0 <= mu_s <= mu_2):
        raise CollapseError("require 0 <= mu_s <= mu_2")
    if not math.isfinite(i_0) or i_0 <= 0.0:
        raise CollapseError("i_0 must be finite and positive")
    i = np.maximum(np.asarray(inertial_number, dtype=np.float64), 0.0)
    return mu_s + (mu_2 - mu_s) / (1.0 + i_0 / np.maximum(i, 1e-12))


@dataclass(frozen=True)
class CollapseCase:
    """Geometry, material and rheology of a 2D column-collapse experiment."""

    #: aspect ratio ``a = H / L0``
    aspect: float = 2.0
    #: particles across the initial column length ``L0``
    width: int = 16
    radius: float = 0.005
    density: float = 2500.0
    thickness: float = 0.03
    #: normal contact stiffness (N/m); softened w.r.t. rock elasticity for
    #: wall clock - the resulting overlap ratio is reported with every result
    kn: float = 3.0e4
    #: normal restitution target -> dashpot ratio zeta
    restitution: float = 0.5
    mu_s: float = JOP_MU_S
    mu_2: float = JOP_MU_2
    i_0: float = JOP_I_0
    mu_rolling: float = 0.0
    #: local deposit height (in diameters) that counts as deposit for the runout
    #: toe; the sensitivity to this choice is reported for 0.5/1/2/3/4 diameters
    pile_threshold_diameters: float = 2.0
    #: random jitter of the initial lattice as a fraction of ``2R``
    jitter: float = 0.02
    seed: int = 20261108
    gravity: float = 9.81
    #: stop once the flow has decayed to ``ke_tol`` of its own peak kinetic
    #: energy (scale free: a tall column has a larger peak than a short one),
    #: held for ``ke_hold`` consecutive steps, and only after the collapse has
    #: actually started (peak ``E_kin > 1 % E_pot(0)``)
    ke_tol: float = 1.0e-2
    ke_hold: int = 500
    max_steps: int = 240000
    #: set False to keep the contact friction coefficient fixed at ``mu_s``
    mu_i_coupling: bool = True
    #: fraction of the contact-period stability limit used as the time step
    dt_safety: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.aspect) or self.aspect <= 0.0:
            raise CollapseError("aspect must be finite and positive")
        if int(self.width) < 4:
            raise CollapseError("width must be >= 4 particles")
        for name in ("radius", "density", "thickness", "kn", "mu_2", "i_0",
                     "gravity", "dt_safety"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise CollapseError(f"{name} must be finite and positive")
        if float(self.mu_s) < 0.0 or float(self.mu_s) > float(self.mu_2):
            raise CollapseError("require 0 <= mu_s <= mu_2")
        if not 0.0 <= self.restitution < 1.0:
            raise CollapseError("restitution must lie in [0, 1)")
        if self.mu_rolling < 0.0:
            raise CollapseError("mu_rolling must be nonnegative")
        if not math.isfinite(self.pile_threshold_diameters) or self.pile_threshold_diameters <= 0.0:
            raise CollapseError("pile_threshold_diameters must be finite and positive")

    # --------------------------------------------------------------- derived
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def pile_threshold(self) -> float:
        """Deposit height that counts as a pile: ``pile_threshold`` diameters."""
        return float(self.pile_threshold_diameters) * self.diameter

    @property
    def length(self) -> float:
        """Initial column length along the flow ``L0`` (m)."""
        return self.width * self.diameter

    @property
    def height(self) -> float:
        """Initial column height ``H = a L0`` (m)."""
        return self.aspect * self.length

    @property
    def zeta(self) -> float:
        """Dashpot ratio reproducing the requested normal restitution."""
        e = max(min(self.restitution, 0.999), 1.0e-6)
        ln_e = math.log(e)
        return -ln_e / math.sqrt(math.pi**2 + ln_e**2)

    def disk_config(self) -> DiskConfig:
        return DiskConfig(
            kn=float(self.kn),
            kt_ratio=0.5,
            zeta=self.zeta,
            mu=self.mu_s,
            mu_rolling=self.mu_rolling,
            density=self.density,
            thickness=self.thickness,
            gravity=(0.0, -self.gravity),
        )


def build_column(case: CollapseCase) -> DiskSystem:
    """Lattice-packed column resting on a rough floor with a back wall at ``x = 0``."""
    rng = np.random.default_rng(case.seed)
    d = case.diameter
    nx = int(case.width)
    ny = int(round(case.height / d))
    if ny < 1:
        raise CollapseError("height must exceed one particle diameter")
    xs = (np.arange(nx) + 0.5) * d
    ys = (np.arange(ny) + 0.5) * d
    dx, dy = np.meshgrid(xs, ys, indexing="xy")
    pos = np.stack([dx.ravel(), dy.ravel()], axis=1)
    pos += rng.uniform(-case.jitter * d, case.jitter * d, size=pos.shape)
    pos[:, 0] = np.clip(pos[:, 0], 0.5 * d, (nx + 0.5) * d)
    pos[:, 1] = np.maximum(pos[:, 1], 0.5 * d)
    radii = np.full(len(pos), case.radius)
    cfg = case.disk_config()
    floor = np.asarray([[-0.5 * d, 0.0], [nx * d + 12.0 * case.height + 20.0 * d, 0.0]])
    back = np.asarray([[0.0, -d], [0.0, case.height + 6.0 * d]])
    walls = np.asarray([floor, back], dtype=np.float64)
    system = DiskSystem(pos, radii, cfg, walls=walls)
    system.dt = case.dt_safety * system.recommended_dt
    return system


# ------------------------------------------------------------------- rheology
def local_inertial_number(system: DiskSystem, case: CollapseCase, *,
                          n_bins: int = 8) -> float:
    """Flow-scale inertial number ``I = gamma_dot d / sqrt(P / rho_p)``.

    ``gamma_dot`` is the largest velocity gradient of the x-velocity profile
    (binned in ``y``, finite differences of the bin means) and ``P`` is the
    basal pressure ``sum(f_n on the floor) / (footprint * thickness)`` from the
    contact records.  Both are measured from the current DEM state.
    """
    contacts = getattr(system, "contacts", None)
    if not contacts or "kind" not in contacts:
        contacts = system.contact_forces()
    kind = contacts["kind"]
    fn = contacts["fn"]
    wall = kind == 1
    if np.any(wall):
        footprint = max(np.ptp(system.pos[:, 0]) + 2.0 * case.radius, 4.0 * case.radius)
        pressure = float(np.sum(fn[wall])) / (footprint * case.thickness)
    else:
        pressure = 0.0
    if pressure <= 0.0:
        pressure = case.density * case.gravity * max(float(np.max(system.pos[:, 1])),
                                                     case.diameter)
    vx = system.vel[:, 0]
    y = system.pos[:, 1]
    edges = np.linspace(float(y.min()), float(y.max()) + 1e-12, int(n_bins) + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, int(n_bins) - 1)
    prof = np.full(int(n_bins), np.nan)
    for b in range(int(n_bins)):
        sel = idx == b
        if np.count_nonzero(sel) >= 3:
            prof[b] = vx[sel].mean()
    good = np.isfinite(prof)
    if np.count_nonzero(good) < 2:
        return 0.0
    centres = 0.5 * (edges[:-1] + edges[1:])[good]
    grad = np.abs(np.diff(prof[good]) / np.diff(centres))
    shear = float(grad.max()) if len(grad) else 0.0
    return shear * case.diameter / math.sqrt(pressure / case.density)


# -------------------------------------------------------------------- ledger
@dataclass
class EnergyLedger:
    """Energy ledger of a DEM collapse (all terms in J).

    ``dissipated`` is measured *forwards* at contact level from the second-law
    statement

        ``D = max(0, -(W_contact + dE_elastic))``

    accumulated per step, with ``W_contact`` the work done by the contact forces
    on the disks through the relative velocity at the contact point and
    ``dE_elastic`` the change of the recoverable spring energies
    (``k_n ov^2 / 2`` normal, ``f_t^2 / (2 k_t)`` tangential).  Every
    irreversible channel of the Cundall-Strack contact is inside that statement:
    the normal dashpot, Coulomb sliding *and* the tangential-spring clipping that
    the engine performs when the friction cone is hit.
    """

    potential0: float = 0.0
    potential: float = 0.0
    kinetic: float = 0.0
    elastic_normal: float = 0.0
    elastic_tangential: float = 0.0
    dissipated: float = 0.0
    dissipated_damping: float = 0.0
    dissipated_friction: float = 0.0
    dissipated_signed: float = 0.0
    created_by_discretisation: float = 0.0
    steps: int = 0

    @property
    def mechanical(self) -> float:
        return (self.potential + self.kinetic + self.elastic_normal
                + self.elastic_tangential)

    @property
    def closure_error(self) -> float:
        """``|E_pot(0) - (mechanical + dissipated)| / E_pot(0)``."""
        if self.potential0 <= 0.0:
            return 0.0
        return abs(self.potential0 - (self.mechanical + self.dissipated)) / self.potential0

    def as_dict(self) -> dict[str, float]:
        return {
            "potential_initial_j": self.potential0,
            "potential_j": self.potential,
            "kinetic_j": self.kinetic,
            "elastic_normal_j": self.elastic_normal,
            "elastic_tangential_j": self.elastic_tangential,
            "dissipated_j": self.dissipated,
            "dissipated_signed_j": self.dissipated_signed,
            "created_by_discretisation_j": self.created_by_discretisation,
            "dissipated_damping_j": self.dissipated_damping,
            "dissipated_friction_j": self.dissipated_friction,
            "closure_error": self.closure_error,
            "ledger_steps": self.steps,
        }


def _mean_effective_mass(system: DiskSystem) -> float:
    """Reduced mass of an equal-disk pair (the engine's ``m_eff``)."""
    return 0.5 * float(np.min(system.mass))


def deposit_front(x_extent: np.ndarray, y_extent: np.ndarray, *, cell: float,
                  height_threshold: float, max_gap: int = 3) -> float:
    """Downstream toe of the *pile* (m).

    A 2D disc packing throws single grains far beyond the deposit, so neither the
    raw maximum nor a purely binary occupancy test is a runout.  The x axis is
    binned in cells of width ``cell`` (one particle diameter); a bin counts as
    deposit only if its local height reaches ``height_threshold`` (a few
    diameters - i.e. a pile, not a single rolling grain).  The front is the last
    deposit bin before the first gap of ``max_gap`` consecutive empty bins.
    """
    x = np.asarray(x_extent, dtype=np.float64)
    y = np.asarray(y_extent, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise CollapseError("x and y extents must be equal-length 1D arrays")
    if not math.isfinite(cell) or cell <= 0.0:
        raise CollapseError("cell must be finite and positive")
    if not math.isfinite(height_threshold) or height_threshold <= 0.0:
        raise CollapseError("height_threshold must be finite and positive")
    lo = math.floor(float(x.min()) / cell)
    hi = math.ceil(float(x.max()) / cell)
    n_bins = hi - lo + 1
    top = np.full(n_bins, -np.inf)
    idx = np.clip(np.floor(x / cell).astype(np.int64) - lo, 0, n_bins - 1)
    np.maximum.at(top, idx, y)
    occupied = top >= height_threshold
    gap = 0
    front = 0.0
    for i, occ in enumerate(occupied):
        if occ:
            gap = 0
            front = (lo + i + 1) * cell
        else:
            gap += 1
            if gap >= int(max_gap):
                break
    return float(front)


def potential_energy(system: DiskSystem, gravity: float) -> float:
    """Gravitational potential energy relative to the floor (J)."""
    return float(np.sum(system.mass * gravity * system.pos[:, 1]))


def contact_keys(system: DiskSystem, contacts: dict[str, np.ndarray]) -> np.ndarray:
    """Stable per-contact keys (``i n + j`` for disk pairs, ``n^2 + idx`` for walls)."""
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    n = system.n
    keys = np.empty(len(i), dtype=np.int64)
    pair = j >= 0
    keys[pair] = i[pair] * n + j[pair]
    if np.any(~pair):
        keys[~pair] = n * n + np.arange(np.count_nonzero(~pair), dtype=np.int64)
    return keys


def rolling_stiffness(system: DiskSystem,
                      contacts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-spring stiffness ``k_r = k_t r_eff^2 / 4`` and ``r_eff`` per contact."""
    kt = system.config.kt_ratio * system.config.kn
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    pair = j >= 0
    r_eff = np.empty(len(i), dtype=np.float64)
    if np.any(pair):
        r_i = system.radii[i[pair]]
        r_j = system.radii[j[pair]]
        r_eff[pair] = r_i * r_j / (r_i + r_j)
    if np.any(~pair):
        r_eff[~pair] = system.radii[i[~pair]]
    return 0.25 * kt * r_eff**2, r_eff


def rolling_angle(system: DiskSystem, keys: np.ndarray) -> np.ndarray:
    """Engine rolling-spring state (rad) mapped onto ``keys``.

    The engine keeps ``roll`` per contact key (``i n + j`` for pairs, ``n^2 + idx``
    for walls); unknown keys get zero, exactly like ``_load_state``.
    """
    engine_keys = np.asarray(getattr(system, "_state_keys", np.zeros(0, dtype=np.int64)))
    engine_roll = np.asarray(getattr(system, "_state_roll", np.zeros(0)))
    if engine_keys.size == 0:
        return np.zeros(len(keys))
    order = np.argsort(engine_keys)
    sorted_keys = engine_keys[order]
    pos = np.clip(np.searchsorted(sorted_keys, keys), 0, len(sorted_keys) - 1)
    found = sorted_keys[pos] == keys
    return np.where(found, engine_roll[order][pos], 0.0)


def spring_energies(system: DiskSystem,
                    contacts: dict[str, np.ndarray]) -> np.ndarray:
    """Recoverable elastic energy stored in every contact (J).

    Normal ``k_n ov^2 / 2``, tangential ``f_t^2 / (2 k_t)`` and, when rolling
    resistance is active, the rolling spring ``k_r roll^2 / 2``.
    """
    kn = system.config.kn
    kt = system.config.kt_ratio * kn
    ov = np.maximum(np.asarray(contacts["overlap"], dtype=np.float64), 0.0)
    ft = np.asarray(contacts["ft"], dtype=np.float64)
    energy = 0.5 * kn * ov**2 + 0.5 * ft**2 / kt
    if system.config.mu_rolling > 0.0:
        k_r, _ = rolling_stiffness(system, contacts)
        roll = rolling_angle(system, contact_keys(system, contacts))
        energy = energy + 0.5 * k_r * roll**2
    return energy


def contact_kinematics(system: DiskSystem, contacts: dict[str, np.ndarray]) -> tuple[
        np.ndarray, np.ndarray]:
    """Relative velocity at the contact point, plus its normal/tangential split."""
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    point = np.asarray(contacts["point"], dtype=np.float64)
    n = np.asarray(contacts["n"], dtype=np.float64)
    r_i = point - system.pos[i]
    v_i = system.vel[i] + system.omega[i][:, None] * np.stack(
        [-r_i[:, 1], r_i[:, 0]], axis=1)
    v_j = np.zeros_like(v_i)
    pair = j >= 0
    if np.any(pair):
        jj = j[pair]
        r_j = point[pair] - system.pos[jj]
        v_j[pair] = system.vel[jj] + system.omega[jj][:, None] * np.stack(
            [-r_j[:, 1], r_j[:, 0]], axis=1)
    v_rel = v_i - v_j
    v_n = np.sum(v_rel * n, axis=1)
    v_t = np.linalg.norm(v_rel - v_n[:, None] * n, axis=1)
    return v_rel, v_n, v_t


def _matched_delta(prev_keys: np.ndarray, prev_energy: np.ndarray,
                   keys: np.ndarray, energy: np.ndarray) -> float:
    """``sum(E_now - E_prev)`` over the contact set, with vanished contacts -> 0."""
    order = np.argsort(keys)
    keys_sorted = keys[order]
    energy_sorted = energy[order]
    pos = np.searchsorted(keys_sorted, prev_keys)
    pos_clipped = np.clip(pos, 0, max(len(keys_sorted) - 1, 0))
    found = (len(keys_sorted) > 0) & (keys_sorted[pos_clipped] == prev_keys)
    now = np.where(found, energy_sorted[pos_clipped], 0.0)
    return float(np.sum(now - prev_energy))


def _elastic_split(system: DiskSystem, kt: float) -> tuple[float, float]:
    """Recoverable elastic energies ``(normal, tangential)`` of the current contacts."""
    contacts = system.contacts
    kn = system.config.kn
    ov = np.maximum(np.asarray(contacts["overlap"], dtype=np.float64), 0.0)
    ft = np.asarray(contacts["ft"], dtype=np.float64)
    return 0.5 * kn * float(np.sum(ov**2)), 0.5 * float(np.sum(ft**2)) / kt


@dataclass(frozen=True)
class CollapseResult:
    """Outcome of one column collapse."""

    aspect: float
    runout: float
    runout_max: float
    initial_length: float
    initial_height: float
    final_height: float
    relative_runout: float
    runout_thresholds: dict[str, float]
    relative_height: float
    steps: int
    time: float
    free_fall_time: float
    inertial_number_peak: float
    inertial_number_mean: float
    mu_flow_mean: float
    max_overlap_ratio: float
    peak_kinetic_fraction: float
    stopped: bool
    ledger: dict[str, float]
    digest: str
    front_history: list[tuple[float, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "aspect": self.aspect,
            "runout_p99_m": self.runout,
            "runout_max_m": self.runout_max,
            "initial_length_m": self.initial_length,
            "initial_height_m": self.initial_height,
            "final_height_m": self.final_height,
            "relative_runout": self.relative_runout,
            "runout_thresholds_m": self.runout_thresholds,
            "relative_height": self.relative_height,
            "steps": self.steps,
            "time_s": self.time,
            "free_fall_time_s": self.free_fall_time,
            "inertial_number_peak": self.inertial_number_peak,
            "inertial_number_mean": self.inertial_number_mean,
            "mu_flow_mean": self.mu_flow_mean,
            "max_overlap_ratio": self.max_overlap_ratio,
            "peak_kinetic_fraction": self.peak_kinetic_fraction,
            "stopped": self.stopped,
            "energy_ledger": self.ledger,
            "state_digest": self.digest,
        }


def column_collapse(case: CollapseCase, *, record_every: int = 100,
                    max_wall_clock: float | None = None,
                    with_ledger: bool = False) -> CollapseResult:
    """Release the column and measure the final deposit.

    Two runout measures are reported: ``runout_max`` is the raw front (most
    downstream particle surface) and ``runout`` is the toe of the *connected*
    deposit (:func:`deposit_front`).  Single grains are routinely ejected far
    beyond the pile by a 2D disc packing, so the connected-deposit toe is the
    estimator used for the scaling fit.

    With ``with_ledger`` the exact contact-level energy accounting of
    :class:`EnergyLedger` is accumulated (one extra force evaluation per step,
    i.e. roughly twice the cost); without it only the mechanical terms and the
    bookkeeping remainder are reported.
    """
    import time as _time

    system = build_column(case)
    e_pot0 = potential_energy(system, case.gravity)
    dt = system.dt
    kn = system.config.kn
    kt = system.config.kt_ratio * kn
    ledger = EnergyLedger(potential0=e_pot0)
    x0 = float(np.percentile(system.pos[:, 0] + system.radii, 99.0))
    front: list[tuple[float, float]] = [(0.0, x0)]
    i_values: list[float] = []
    mus: list[float] = []
    prev = None
    work_prev = 0.0
    damping_prev = 0.0
    friction_prev = 0.0
    step = 0
    quiet = 0
    peak_ke = 0.0
    kinetic = 0.0
    gate = 0.01 * e_pot0
    t_start = _time.perf_counter()
    while step < case.max_steps:
        if with_ledger:
            contacts = system.contact_forces()          # state at t_k
            keys = contact_keys(system, contacts)
            energy = spring_energies(system, contacts)
            v_rel, v_n, v_t = contact_kinematics(system, contacts)
            force = np.asarray(contacts["f"], dtype=np.float64)
            ft_vec = np.asarray(contacts["ft"], dtype=np.float64)
            ov = np.maximum(np.asarray(contacts["overlap"], dtype=np.float64), 0.0)
            c_n = 2.0 * system.config.zeta * math.sqrt(
                kn * _mean_effective_mass(system))
            damping_now = float(np.sum(np.minimum(c_n * np.maximum(v_n, 0.0), kn * ov)
                                       * np.maximum(v_n, 0.0)))
            friction_now = float(np.sum(ft_vec * v_t))
        if case.mu_i_coupling:
            i_now = local_inertial_number(system, case)
            i_values.append(i_now)
            mu_now = float(jop_mu_I(i_now, case.mu_s, case.mu_2, case.i_0))
            mus.append(mu_now)
            # DiskConfig is frozen: swap in a copy carrying the flow-scale mu(I)
            system.config = replace(system.config, mu=mu_now)
        system.step(1)
        step += 1
        if with_ledger:
            # work of the contact forces through the displacement the semi-implicit
            # update actually used (dx = v_{k+1} dt): consistent with the spring
            # compression, so ``W + dE_spring`` carries no O(dt) bias
            v_next = contact_kinematics(system, contacts)[0]
            work = float(np.sum(np.sum(force * v_next, axis=1))) * dt
            if prev is not None:
                delta_e = _matched_delta(prev[0], prev[1], keys, energy)
                irrevers = -(work_prev + delta_e)
                ledger.dissipated += max(0.0, irrevers)
                ledger.dissipated_signed += irrevers
                if irrevers < 0.0:
                    ledger.created_by_discretisation -= irrevers
                ledger.dissipated_damping += 0.5 * (damping_prev + damping_now) * dt
                ledger.dissipated_friction += 0.5 * (friction_prev + friction_now) * dt
            order = np.argsort(keys)
            prev = (keys[order], energy[order])
            work_prev = work
            damping_prev = damping_now
            friction_prev = friction_now
        if step % max(int(record_every), 1) == 0:
            front.append((system.time, float(np.percentile(
                system.pos[:, 0] + system.radii, 99.0))))
        kinetic = system.kinetic_energy()
        peak_ke = max(peak_ke, kinetic)
        if peak_ke > gate and kinetic < case.ke_tol * peak_ke:
            quiet += 1
            if quiet >= case.ke_hold:
                break
        else:
            quiet = 0
        if max_wall_clock is not None and _time.perf_counter() - t_start > max_wall_clock:
            break
    x_extent = system.pos[:, 0] + system.radii
    y_extent = system.pos[:, 1] + system.radii
    thresholds = {f"{k}d": deposit_front(x_extent, y_extent, cell=case.diameter,
                                         height_threshold=k * case.diameter)
                  for k in (0.5, 1.0, 2.0, 3.0, 4.0)}
    runout = deposit_front(x_extent, y_extent, cell=case.diameter,
                           height_threshold=case.pile_threshold)
    runout_max = float(np.max(x_extent))
    final_height = float(np.percentile(system.pos[:, 1] + system.radii, 99.5))
    max_overlap = float(np.max(np.maximum(np.asarray(system.contacts["overlap"]), 0.0))
                        / case.diameter) if system.contacts["overlap"].size else 0.0
    e_pot = potential_energy(system, case.gravity)
    e_normal, e_tangential = _elastic_split(system, kt)
    ledger.steps = step
    ledger.potential = e_pot
    ledger.kinetic = kinetic
    ledger.elastic_normal = e_normal
    ledger.elastic_tangential = e_tangential
    if not with_ledger:
        # bookkeeping remainder: tautological, flagged as such in the report
        ledger.dissipated = e_pot0 - (e_pot + kinetic + e_normal + e_tangential)
    energy = ledger.as_dict()
    energy["measured"] = ("contact-level forward accounting" if with_ledger
                          else "remainder (tautological, not an independent measurement)")
    free_fall = math.sqrt(2.0 * case.height / case.gravity)
    return CollapseResult(
        aspect=case.aspect,
        runout=runout,
        runout_max=runout_max,
        initial_length=case.length,
        initial_height=case.height,
        final_height=final_height,
        relative_runout=(runout - case.length) / case.length,
        runout_thresholds=thresholds,
        relative_height=final_height / case.length,
        steps=step,
        time=system.time,
        free_fall_time=free_fall,
        inertial_number_peak=float(np.max(i_values)) if i_values else 0.0,
        inertial_number_mean=float(np.mean(i_values)) if i_values else 0.0,
        mu_flow_mean=float(np.mean(mus)) if mus else case.mu_s,
        max_overlap_ratio=max_overlap,
        peak_kinetic_fraction=peak_ke / e_pot0 if e_pot0 else 0.0,
        stopped=quiet >= case.ke_hold,
        ledger=energy,
        digest=system.digest(),
        front_history=front,
    )


def collapse_energy_ledger(case: CollapseCase, **kwargs) -> dict[str, object]:
    """Run the collapse with the contact-level ledger and return it."""
    result = column_collapse(case, with_ledger=True, **kwargs)
    return {"ledger": result.ledger, "result": result, "steps": result.steps,
            "dt": None, "time": result.time, "settled": None,
            "digest": result.digest}


# ------------------------------------------------------------------- scaling
def fit_power_law(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Least-squares fit ``y = c x^p``; returns ``c``, ``p`` and ``r2``."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim != 1 or len(xa) < 2:
        raise CollapseError("need two equal-length 1D arrays with >= 2 points")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise CollapseError("power-law fit needs strictly positive data")
    p, log_c = np.polyfit(np.log(xa), np.log(ya), 1)
    pred = log_c + p * np.log(xa)
    ss_res = float(np.sum((np.log(ya) - pred) ** 2))
    ss_tot = float(np.sum((np.log(ya) - np.mean(np.log(ya))) ** 2))
    return {"prefactor": float(math.exp(log_c)), "exponent": float(p),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
            "n_points": int(len(xa))}


def scaling_comparison(exponent: float, *, reference: float,
                       tolerance: float = 0.10) -> dict[str, object]:
    """Grade a fitted exponent against a literature reference."""
    if not math.isfinite(exponent):
        raise CollapseError("exponent must be finite")
    rel = abs(exponent - reference) / reference
    return {"exponent": float(exponent), "reference_exponent": float(reference),
            "relative_error": float(rel), "tolerance": float(tolerance),
            "passes": bool(rel <= tolerance)}


def runout_scan(aspects: Iterable[float], *, case: CollapseCase | None = None,
                **kwargs) -> dict[str, object]:
    """Run the aspect-ratio scan and fit the runout scaling law."""
    base = case or CollapseCase()
    results = [column_collapse(replace(base, aspect=float(a)), **kwargs)
               for a in aspects]
    low = [r for r in results if r.aspect <= 3.0]
    high = [r for r in results if r.aspect >= 3.0]
    fit_low = (fit_power_law([r.aspect for r in low],
                             [max(r.relative_runout, 1e-9) for r in low])
               if len(low) >= 2 else None)
    fit_high = (fit_power_law([r.aspect for r in high],
                              [max(r.relative_runout, 1e-9) for r in high])
                if len(high) >= 2 else None)
    fit_all = fit_power_law([r.aspect for r in results],
                            [max(r.relative_runout, 1e-9) for r in results])
    return {
        "cases": [r.as_dict() for r in results],
        "fit_all": fit_all,
        "fit_low_aspect": fit_low,
        "fit_high_aspect": fit_high,
        "front_history": {f"{r.aspect:g}": r.front_history for r in results},
    }


# ---------------------------------------------------------------------- API
def granular_collapse_calculation(request: dict) -> dict:
    """API contract for ``POST /api/v1/calculations/dem/granular_collapse``.

    Request (all keys optional)``::``

        {"aspect": 2.0, "width": 16, "radius": 0.005, "density": 2500.0,
         "restitution": 0.5, "mu_s": 0.36, "mu_2": 0.84, "i_0": 0.3,
         "aspects": [1.0, 2.0, 3.0], "max_steps": 240000}

    With ``aspects`` the handler runs the full scan and returns the fitted
    runout exponent; otherwise it runs a single collapse.
    """
    if not isinstance(request, dict):
        raise CollapseError("request must be a mapping")
    kwargs = {k: request[k] for k in
              ("width", "radius", "density", "thickness", "restitution", "mu_s",
               "mu_2", "i_0", "kn", "max_steps", "jitter", "seed", "mu_i_coupling")
              if k in request}
    aspects = request.get("aspects")
    if aspects:
        case = CollapseCase(**kwargs)
        scan = runout_scan([float(a) for a in aspects], case=case)
        low = scan["fit_low_aspect"]
        verdict = (scaling_comparison(low["exponent"],
                                      reference=LAJEUNESSE_RECTANGULAR_EXPONENT_LOW)
                   if low else None)
        return {
            "endpoint": "/api/v1/calculations/dem/granular_collapse",
            "mode": "scan",
            "case": {k: getattr(case, k) for k in
                     ("width", "radius", "density", "restitution", "mu_s", "mu_2",
                      "i_0", "mu_rolling")},
            "scan": {k: scan[k] for k in ("cases", "fit_all", "fit_low_aspect",
                                          "fit_high_aspect")},
            "verdict": verdict,
            "capability": "REAL",
        }
    case = CollapseCase(aspect=float(request.get("aspect", 2.0)), **kwargs)
    result = column_collapse(case)
    return {
        "endpoint": "/api/v1/calculations/dem/granular_collapse",
        "mode": "single",
        "case": {k: getattr(case, k) for k in
                 ("aspect", "width", "radius", "density", "restitution", "mu_s",
                  "mu_2", "i_0", "mu_rolling", "mu_i_coupling")},
        "result": result.as_dict(),
        "capability": "REAL",
    }