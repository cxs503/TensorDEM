"""Inertial (time-stepped) DEM fragmentation of a bonded ice lattice.

Why a second instrument
-----------------------
:mod:`tensordem.fsd_evolution.cascade` measures the breakage kernel on a
*quasi-static* engine: the lattice is pushed at a prescribed displacement and
the energy ledger is an elastic identity, ``U_before - U_after``.  Two things
therefore cannot be measured there, and both are needed for a physically
credible floe-size distribution:

* a **rate** -- the quasi-static engine has no clock, so the breakage frequency
  ``nu`` is per unit applied strain, not per second.  A floe field evolves in
  time (1000 s in this node), so the kernel needs ``nu`` in ``1/s``;
* the **inertial energy budget** -- breakage converts external work into
  fracture energy plus dissipation plus kinetic energy; without inertia the
  ledger can only close as a tautology.

Two loadings are implemented on the same bonded lattice:

``loading="platen"`` (primary, rate controlled tension)
    The two end columns are displaced apart at a constant strain rate
    ``strain_rate`` (1/s).  This is the dynamic twin of the verified
    quasi-static platen, so the daughter distributions are directly comparable
    while the clock and the kinetic energy are real.  The fragment census is
    taken at every bond-break batch, which yields

    * a nested ``parent -> daughters`` catalogue (the empirical
      ``beta(d | d')``), and
    * a **time-based** hazard: split events per bin divided by the floe-time
      exposure of that bin (Nelson-Aalen style, right-censored floes handled).

``loading="wall" | "stretch"`` (threshold / energy instrument)
    A uniform velocity ``impact_velocity`` is imposed (against a rigid wall, or
    as a tensile spall of the two halves) and the lattice rings down.  This
    instrument measures the *breakage energy scale*: the fragmentation
    probability as a function of impact speed, hence the threshold kinetic
    energy of a floe of a given size.

Equation of motion (explicit leapfrog / velocity Verlet)
--------------------------------------------------------
.. math::

   m_i \\ddot{x}_i = \\sum_{b \\ni i} [k (\\ell_b - a) + c\\, \\dot\\ell_b] \\hat{n}_b
                    + F^{wall}_i

Bonds are axial (central-force) springs of stiffness ``k = E t / sqrt(3)`` --
the same lattice constant as the verified quasi-static engine, so both
instruments are the same material model -- with a Weibull threshold population
``P(eps_c > eps) = exp(-(eps/eps_0)**m)``.  A bond fails when the tensile strain
exceeds its own threshold, or when the compressive strain exceeds
``compression_factor`` times it.

Energy ledger (measured, not assumed)
-------------------------------------
=========================  =====================================================
``KE``                     ``1/2 sum m v^2``
``U_elastic``              ``1/2 sum_alive k (ell - a)^2``
``U_wall``                 wall contact spring energy ``1/2 k_w delta^2``
``E_dissipated``           bond dashpot ``sum c (v_rel . n)^2 dt`` + wall dashpot
``E_fracture``             ``1/2 k (ell-a)^2`` of each bond *at the instant it
                           fails* -- the energy that leaves the elastic field
``W_external``             work of the platens / moving wall on the lattice
=========================  =====================================================

Closure is ``KE + U_elastic + U_wall + E_dissipated + E_fracture
= KE_0 + W_external`` up to the ``O(dt**2)`` drift of the explicit scheme; the
drift is *reported* and its convergence with ``dt`` is an acceptance criterion.

Fragments
---------
Because bonds are only ever removed, the connected components of the surviving
bond network are the fragments.  Every node belongs to exactly one component,
so the fragment-area census conserves area (node mass) by construction -- the
mass criterion is a check on the bookkeeping, not on the physics.

Honesty / scope
---------------
* Two-dimensional central-force lattice: no bending moments, so absolute
  strength, absolute ``G_c`` and the absolute fracture energy are model numbers.
  Exponents, ratios and the *columns* of the energy ledger are transferable.
* The platen instrument loads a *rooted* cascade: a fragment that has detached
  from both platens is no longer loaded and cannot break again (this instrument
  has no fragment-fragment contact).  The measured hazard is therefore a lower
  bound for a debris-covered field; the caveat is carried into the report.
* Single-node fragments sit on the lattice resolution floor; the mass fraction
  at the floor is reported so the resolution-limited part of the daughter
  distribution is never silently used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..broken_ice_micro.fragmentation import (
    _component_labels,
    fragment_census,
    triangular_lattice,
)

__all__ = [
    "DynamicImpactConfig",
    "DynamicImpactResult",
    "SplitEvent",
    "DynamicFragmentation",
    "run_impact",
    "impact_scan",
    "platen_scan",
]


@dataclass(frozen=True)
class DynamicImpactConfig:
    """Lattice, material, loading and integrator parameters (SI units)."""

    nx: int = 32
    ny: int = 16
    spacing: float = 0.03
    thickness: float = 0.5
    density: float = 917.0
    young_modulus: float = 5.0e9
    #: Weibull scale of the failure strain.  2e-4 * 5 GPa = 1 MPa tensile
    #: strength, inside the sea-ice band (Timco & Weeks 2010: 0.7-1.4 MPa).
    failure_strain_scale: float = 2.0e-4
    weibull_modulus: float = 4.0
    compression_factor: float = 5.0
    damping_ratio: float = 0.05
    loading: str = "platen"
    #: platen: imposed strain rate (1/s); impact: impact speed (m/s).
    strain_rate: float = 10.0
    impact_velocity: float = 1.0
    target_strain: float = 0.05
    dt: float = 2.0e-6
    t_end: float = 5.0e-3
    wall_gap: float = 0.0
    wall_stiffness_factor: float = 20.0
    wall_damping_ratio: float = 0.5
    seed: int = 0
    min_fragment_nodes: int = 1
    record_every: int = 25
    census_stride: int = 10

    def __post_init__(self) -> None:
        if int(self.nx) < 4 or int(self.ny) < 4:
            raise ValueError("nx and ny must be >= 4")
        if not (self.spacing > 0.0 and self.thickness > 0.0 and self.density > 0.0):
            raise ValueError("geometry/material constants must be positive")
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive")
        if self.dt <= 0.0 or self.t_end <= 0.0:
            raise ValueError("dt and t_end must be positive")
        if self.dt > 0.75 * self.stable_dt:
            raise ValueError(
                f"dt={self.dt:.3e} s is too close to the leapfrog stability limit "
                f"({self.stable_dt:.3e} s); use dt <= 0.75 * limit"
            )
        if self.loading not in ("platen", "wall", "stretch"):
            raise ValueError("loading must be 'platen', 'wall' or 'stretch'")
        if not (0.0 <= self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must be in [0, 1)")
        if self.min_fragment_nodes < 1:
            raise ValueError("min_fragment_nodes must be >= 1")
        if self.loading == "platen" and self.strain_rate <= 0.0:
            raise ValueError("strain_rate must be positive")
        if self.loading != "platen" and self.impact_velocity <= 0.0:
            raise ValueError("impact_velocity must be positive")

    # -- derived material constants (same lattice as the quasi-static engine) --
    @property
    def node_area(self) -> float:
        return 0.5 * math.sqrt(3.0) * self.spacing**2

    @property
    def node_mass(self) -> float:
        """Lumped nodal mass (kg): the plate is a 2-D sheet of constant ``t``."""
        return self.density * self.thickness * self.node_area

    @property
    def bond_stiffness(self) -> float:
        """Axial bond stiffness (N/m), identical to the quasi-static lattice."""
        return self.young_modulus * self.thickness / math.sqrt(3.0)

    @property
    def bond_damping(self) -> float:
        """Axial dashpot (N s/m) at a given fraction of critical damping."""
        reduced = 0.5 * self.node_mass
        return 2.0 * self.damping_ratio * math.sqrt(self.bond_stiffness * reduced)

    @property
    def wall_stiffness(self) -> float:
        return self.wall_stiffness_factor * self.bond_stiffness

    @property
    def wall_damping(self) -> float:
        return 2.0 * self.wall_damping_ratio * math.sqrt(self.wall_stiffness * self.node_mass)

    @property
    def stable_dt(self) -> float:
        """Leapfrog stability limit ``2 / omega_max`` (bonds + wall contact)."""
        omega_bond = math.sqrt(6.0 * self.bond_stiffness / self.node_mass)
        omega_wall = math.sqrt(self.wall_stiffness / self.node_mass)
        return 2.0 / max(omega_bond, omega_wall)

    @property
    def node_length(self) -> float:
        return (self.nx - 1) * self.spacing

    @property
    def node_width(self) -> float:
        return (self.ny - 1) * self.spacing * math.sin(math.pi / 3.0)

    @property
    def plate_area(self) -> float:
        return self.node_length * self.node_width

    @property
    def lattice_area(self) -> float:
        """Area owned by the lattice nodes (m^2) -- the census measure."""
        return float(self.nx * self.ny) * self.node_area

    @property
    def parent_diameter(self) -> float:
        """Diameter (m) of the area-equivalent disk of the intact plate."""
        return math.sqrt(4.0 * self.lattice_area / math.pi)

    @property
    def plate_mass(self) -> float:
        return self.node_mass * float(self.nx * self.ny)

    @property
    def wave_speed(self) -> float:
        """Axial wave speed of the lattice (m/s)."""
        return self.spacing * math.sqrt(self.bond_stiffness / self.node_mass)

    @property
    def strain_rate_ratio(self) -> float:
        """Loading time / wave transit time: >> 1 means quasi-static loading."""
        if self.loading != "platen":
            return float("nan")
        return self.strain_rate / (self.wave_speed / self.node_length)


@dataclass(frozen=True)
class SplitEvent:
    """One observed ``parent -> daughters`` split, with its own time stamp."""

    t: float
    parent_diameter: float
    daughter_diameters: np.ndarray

    @property
    def n_daughters(self) -> int:
        return int(self.daughter_diameters.size)

    @property
    def daughter_ratios(self) -> np.ndarray:
        return self.daughter_diameters / self.parent_diameter


@dataclass
class DynamicImpactResult:
    """Outcome of one run: fragments, timing, energy ledger, history."""

    config: DynamicImpactConfig
    parent_diameter: float
    parent_mass: float
    parent_area: float
    diameters: np.ndarray
    areas: np.ndarray
    node_counts: np.ndarray
    n_fragments: int
    largest_fraction: float
    t_first_break: float | None
    t_split: float | None
    n_broken_bonds: int
    energies: dict
    events: list = field(default_factory=list)
    exposure: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    # -- derived ------------------------------------------------------------
    @property
    def broke(self) -> bool:
        return self.t_split is not None and self.n_fragments >= 2

    @property
    def split_events(self) -> list:
        return [e for e in self.events if e.n_daughters >= 2]

    @property
    def daughter_ratios(self) -> np.ndarray:
        """``x_i = d_i / d_parent`` of every final fragment (``sum x^2 = 1``)."""
        return self.diameters / self.parent_diameter

    @property
    def mass_residual(self) -> float:
        """Relative area (node mass) defect of the fragment census."""
        total = float(np.sum(self.areas))
        return abs(total - self.parent_area) / self.parent_area

    @property
    def ledger_residual(self) -> float:
        return float(self.energies["residual"])

    def floor_mass_fraction(self) -> float:
        """Mass fraction of the census sitting on the lattice resolution floor."""
        area = np.asarray(self.areas, dtype=np.float64)
        if area.size == 0 or area.sum() == 0.0:
            return 0.0
        floor_area = self.config.node_area
        return float(area[area <= 1.5 * floor_area].sum() / area.sum())

    def as_dict(self) -> dict:
        return dict(
            nx=int(self.config.nx),
            ny=int(self.config.ny),
            loading=str(self.config.loading),
            strain_rate_1_s=float(self.config.strain_rate),
            impact_velocity_m_s=float(self.config.impact_velocity),
            dt_s=float(self.config.dt),
            t_end_s=float(self.config.t_end),
            seed=int(self.config.seed),
            parent_diameter_m=float(self.parent_diameter),
            parent_mass_kg=float(self.parent_mass),
            n_fragments=int(self.n_fragments),
            largest_fraction=float(self.largest_fraction),
            t_first_break_s=(None if self.t_first_break is None else float(self.t_first_break)),
            t_split_s=(None if self.t_split is None else float(self.t_split)),
            n_broken_bonds=int(self.n_broken_bonds),
            n_split_events=int(len(self.split_events)),
            mass_residual=float(self.mass_residual),
            ledger_residual=float(self.ledger_residual),
            energies={k: float(v) for k, v in self.energies.items()},
            daughter_ratios=[float(v) for v in self.daughter_ratios],
            floor_mass_fraction=float(self.floor_mass_fraction()),
        )


class DynamicFragmentation:
    """Explicit-dynamics bonded lattice, rate-controlled or impact loaded."""

    def __init__(self, config: DynamicImpactConfig | None = None) -> None:
        self.config = config or DynamicImpactConfig()
        cfg = self.config
        positions, bonds = triangular_lattice(cfg.nx, cfg.ny, cfg.spacing)
        self.positions = positions
        self.bonds = bonds
        self.n_nodes = positions.shape[0]
        self.mass = cfg.node_mass
        self.masses = np.full(self.n_nodes, self.mass)
        self.alive = np.ones(len(bonds), dtype=bool)
        rng = np.random.default_rng(cfg.seed)
        u = rng.random(len(bonds))
        self.thresholds = cfg.failure_strain_scale * (-np.log(np.maximum(u, 1e-15))) ** (
            1.0 / cfg.weibull_modulus
        )
        self.velocities = np.zeros((self.n_nodes, 2), dtype=np.float64)
        self._initialise_velocity()
        self.wall_x = float(positions[:, 0].max()) + cfg.wall_gap
        ids = np.arange(self.n_nodes).reshape(cfg.ny, cfg.nx)
        self._left = ids[:, 0]
        self._right = ids[:, -1]
        self._prescribed = np.zeros(self.n_nodes, dtype=bool)
        self._prescribed[self._left] = True
        self._prescribed[self._right] = True
        self.t = 0.0
        self.n_broken = 0
        self.t_first_break: float | None = None
        self.t_split: float | None = None
        self.labels = np.zeros(self.n_nodes, dtype=np.int64)
        self.n_fragments = 1
        self.largest_fraction = 1.0
        self._broken_energy_pending = 0.0
        self._prev_labels = np.zeros(self.n_nodes, dtype=np.int64)
        self.events: list[SplitEvent] = []
        self._exposure_bins = np.logspace(math.log10(0.02), math.log10(2.0), 21)
        self._exposure = np.zeros(self._exposure_bins.size - 1, dtype=np.float64)
        self._exposure_clock = 0.0
        self.exposure: dict = {}
        self.energies = self._zero_energies()
        self.history = {"t": [], "ke": [], "u_elastic": [], "e_fracture": [],
                        "e_dissipated": [], "w_external": [], "n_fragments": [],
                        "largest_fraction": []}

    # -- setup --------------------------------------------------------------
    def _initialise_velocity(self) -> None:
        cfg = self.config
        if cfg.loading == "platen":
            return
        v0 = cfg.impact_velocity
        x = self.positions[:, 0]
        if cfg.loading == "wall":
            self.velocities[:, 0] = v0
        else:  # tensile spall: the two halves separate at 2*v0
            x_mid = 0.5 * (x.min() + x.max())
            self.velocities[:, 0] = np.where(x < x_mid, -v0, v0)

    def _zero_energies(self) -> dict:
        cfg = self.config
        return {
            "ke_initial": 0.5 * self.mass * float(np.sum(self.velocities**2)),
            "ke": 0.0,
            "u_elastic": 0.0,
            "u_wall": 0.0,
            "e_dissipated": 0.0,
            "e_wall_dissipated": 0.0,
            "e_fracture": 0.0,
            "w_external": 0.0,
            "input": 0.0,
            "residual": 0.0,
            "closure": 1.0,
            "bond_damping": cfg.bond_damping,
            "peak_reaction_N": 0.0,
            "peak_speed_ratio": 0.0,
        }

    # -- forces -------------------------------------------------------------
    def _bond_state(self, positions: np.ndarray):
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        delta = positions[j] - positions[i]
        length = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        safe = np.where(length > 0.0, length, 1.0)
        unit = delta / safe[:, None]
        return i, j, length, unit

    def _bond_forces(self, positions: np.ndarray, velocities: np.ndarray):
        """Bond springs + dashpots: force, elastic energy, dissipation rate."""
        cfg = self.config
        k, c = cfg.bond_stiffness, cfg.bond_damping
        i, j, length, unit = self._bond_state(positions)
        stretch = length - cfg.spacing
        alive = self.alive
        v_rel = velocities[j] - velocities[i]
        v_axial = np.einsum("ij,ij->i", v_rel, unit)
        magnitude = np.where(alive, k * stretch + c * v_axial, 0.0)
        fvec = magnitude[:, None] * unit
        force = np.zeros_like(positions)
        for axis in range(2):
            force[:, axis] = (
                np.bincount(i, weights=fvec[:, axis], minlength=self.n_nodes)
                - np.bincount(j, weights=fvec[:, axis], minlength=self.n_nodes)
            )
        u_elastic = float(0.5 * np.sum(np.where(alive, k * stretch**2, 0.0)))
        diss_rate = float(np.sum(np.where(alive, c * v_axial**2, 0.0)))
        return force, u_elastic, diss_rate, stretch

    def _wall_forces(self, positions: np.ndarray, velocities: np.ndarray):
        cfg = self.config
        penetration = positions[:, 0] - self.wall_x
        contact = penetration > 0.0
        force = np.zeros_like(positions)
        if not np.any(contact):
            return force, 0.0, 0.0
        fx = np.where(
            contact,
            -(cfg.wall_stiffness * penetration + cfg.wall_damping * velocities[:, 0]),
            0.0,
        )
        force[:, 0] = fx
        u_wall = float(0.5 * cfg.wall_stiffness * np.sum(penetration[contact] ** 2))
        wall_diss = float(np.sum(cfg.wall_damping * velocities[contact, 0] ** 2))
        return force, u_wall, wall_diss

    # -- failure ------------------------------------------------------------
    def _fail_bonds(self, stretch: np.ndarray) -> int:
        cfg = self.config
        strain = stretch / cfg.spacing
        tension = strain > self.thresholds
        compression = strain < -cfg.compression_factor * self.thresholds
        newly = self.alive & (tension | compression)
        count = int(newly.sum())
        if not count:
            return 0
        self._broken_energy_pending += float(
            0.5 * cfg.bond_stiffness * np.sum(stretch[newly] ** 2)
        )
        self.alive[newly] = False
        self.n_broken += count
        if self.t_first_break is None:
            self.t_first_break = float(self.t)
        return count

    # -- census / events ----------------------------------------------------
    def _census(self, positions: np.ndarray, *, record_events: bool = True) -> None:
        labels = _component_labels(self.n_nodes, self.bonds[self.alive])
        counts = np.bincount(labels)
        counts = counts[counts > 0]
        n_frag = int(counts.size)
        largest = float(counts.max() / self.n_nodes)
        if record_events and n_frag > self.n_fragments:
            self._register_events(labels)
        self.labels = labels
        self.n_fragments = n_frag
        self.largest_fraction = largest
        if self.t_split is None and n_frag >= 2 and largest <= 0.99:
            self.t_split = float(self.t)

    def _register_events(self, labels: np.ndarray) -> None:
        """Nest the previous census into the new one: parent -> daughters."""
        prev = self._prev_labels
        stride = int(labels.max()) + 2
        pairs, counts = np.unique(prev.astype(np.int64) * stride + labels, return_counts=True)
        children: dict[int, list[int]] = {}
        for pair, count in zip(pairs, counts):
            parent = int(pair // stride)
            children.setdefault(parent, []).append(int(count))
        area = self.config.node_area
        for parent in sorted(children):
            child_nodes = children[parent]
            if len(child_nodes) < 2:
                continue
            parent_area = float(sum(child_nodes)) * area
            daughter = np.asarray(
                [math.sqrt(4.0 * n * area / math.pi) for n in child_nodes], dtype=np.float64
            )
            self.events.append(
                SplitEvent(
                    t=float(self.t),
                    parent_diameter=math.sqrt(4.0 * parent_area / math.pi),
                    daughter_diameters=np.sort(daughter)[::-1],
                )
            )

    def _accumulate_exposure(self, dt_stage: float) -> None:
        """Add ``n_floes * seconds`` to the absolute size bins."""
        if dt_stage <= 0.0 or self.labels.size == 0:
            return
        counts = np.bincount(self.labels)
        ids = np.nonzero(counts)[0]
        area = counts[ids] * self.config.node_area
        diameters = np.sqrt(4.0 * area / math.pi)
        idx = np.clip(
            np.searchsorted(self._exposure_bins, diameters) - 1,
            0,
            self._exposure.size - 1,
        )
        np.add.at(self._exposure, idx, dt_stage)

    # -- integration --------------------------------------------------------
    def run(self) -> DynamicImpactResult:
        if self.config.loading == "platen":
            return self._run_platen()
        return self._run_impact()

    def _record(self, step: int, n_steps: int, ke: float, u_el: float, e: dict) -> None:
        record = max(int(self.config.record_every), 1)
        if step % record == 0 or step == n_steps:
            self.history["t"].append(self.t)
            self.history["ke"].append(ke)
            self.history["u_elastic"].append(u_el)
            self.history["e_fracture"].append(e["e_fracture"])
            self.history["e_dissipated"].append(e["e_dissipated"] + e["e_wall_dissipated"])
            self.history["w_external"].append(e["w_external"])
            self.history["n_fragments"].append(self.n_fragments)
            self.history["largest_fraction"].append(self.largest_fraction)

    def _finalise(self, positions, velocities, e, u_el, u_wall) -> DynamicImpactResult:
        cfg = self.config
        ke = 0.5 * self.mass * float(np.sum(velocities**2))
        e["ke"] = ke
        e["u_elastic"] = float(u_el)
        e["u_wall"] = float(u_wall)
        stored = (
            ke + u_el + u_wall + e["e_dissipated"] + e["e_wall_dissipated"]
            + e["e_fracture"]
        )
        input_energy = e["ke_initial"] + e["w_external"]
        e["input"] = float(input_energy)
        e["residual"] = float((stored - input_energy) / input_energy)
        e["closure"] = float(stored / input_energy)
        self.positions = positions
        self.velocities = velocities
        self._census(positions, record_events=False)
        census = fragment_census(
            self.n_nodes,
            self.bonds[self.alive],
            cfg.node_area,
            min_fragment_nodes=int(cfg.min_fragment_nodes),
        )
        self.exposure = {
            "bin_lo": self._exposure_bins[:-1].tolist(),
            "bin_hi": self._exposure_bins[1:].tolist(),
            "time": self._exposure.tolist(),
            "total_time": float(self._exposure.sum()),
        }
        return DynamicImpactResult(
            config=cfg,
            parent_diameter=float(cfg.parent_diameter),
            parent_mass=float(cfg.plate_mass),
            parent_area=float(cfg.lattice_area),
            diameters=np.asarray(census.diameters, dtype=np.float64),
            areas=np.asarray(census.areas, dtype=np.float64),
            node_counts=np.asarray(census.node_counts, dtype=np.int64),
            n_fragments=int(census.n_fragments),
            largest_fraction=float(census.largest_fraction),
            t_first_break=self.t_first_break,
            t_split=self.t_split,
            n_broken_bonds=int(self.n_broken),
            energies=dict(e),
            events=list(self.events),
            exposure=dict(self.exposure),
            history={k: list(v) for k, v in self.history.items()},
        )

    def _run_platen(self) -> DynamicImpactResult:
        """Rate-controlled tension: the dynamic twin of the verified platen."""
        cfg = self.config
        dt = float(cfg.dt)
        n_steps = int(math.ceil(cfg.t_end / dt))
        positions = self.positions.copy()
        velocities = self.velocities.copy()  # starts at rest
        base = positions.copy()
        half_rate = 0.5 * cfg.strain_rate
        span = float(base[:, 0].max() - base[:, 0].min())
        prescribed = self._prescribed
        e = self.energies
        e["ke_initial"] = 0.0
        force, u_el, diss, stretch = self._bond_forces(positions, velocities)
        accel = force / self.masses[:, None]
        accel[prescribed, 0] = 0.0
        peak_reaction = 0.0
        census_stride = max(int(cfg.census_stride), 1)

        vel_p = np.zeros(self.n_nodes, dtype=np.float64)
        vel_p[self._left] = -half_rate * span
        vel_p[self._right] = half_rate * span
        for step in range(1, n_steps + 1):
            self.t = step * dt
            v_half = velocities + 0.5 * dt * accel
            # work of the platens over this step uses the force held constant
            # during the step (the end-of-previous-step force), which keeps the
            # accounting exact across an abrupt bond-failure batch.
            e["w_external"] += float(-np.dot(force[:, 0], vel_p) * dt)
            peak_reaction = max(peak_reaction, float(np.max(np.abs(force[prescribed, 0]))))
            positions = positions + dt * v_half
            disp = half_rate * span * self.t
            positions[self._left, 0] = base[self._left, 0] - disp
            positions[self._right, 0] = base[self._right, 0] + disp
            velocities[self._left, 0] = -half_rate * span
            velocities[self._right, 0] = half_rate * span
            force, u_el, diss, stretch = self._bond_forces(positions, v_half)
            broke = self._fail_bonds(stretch)
            if broke:
                force, u_el, diss, stretch = self._bond_forces(positions, v_half)
            accel = force / self.masses[:, None]
            accel[prescribed, 0] = 0.0
            velocities = v_half + 0.5 * dt * accel
            e["e_dissipated"] += diss * dt
            e["e_fracture"] += self._broken_energy_pending
            self._broken_energy_pending = 0.0
            if broke:
                self._accumulate_exposure(self.t - self._exposure_clock)
                self._exposure_clock = self.t
                self._prev_labels = self.labels.copy()
                self._census(positions)
                self._prev_labels = self.labels.copy()
            elif step % census_stride == 0:
                self._accumulate_exposure(self.t - self._exposure_clock)
                self._exposure_clock = self.t
            ke = 0.5 * self.mass * float(np.sum(velocities**2))
            self._record(step, n_steps, ke, u_el, e)
        e["peak_reaction_N"] = peak_reaction
        return self._finalise(positions, velocities, e, u_el, 0.0)

    def _run_impact(self) -> DynamicImpactResult:
        cfg = self.config
        dt = float(cfg.dt)
        n_steps = int(math.ceil(cfg.t_end / dt))
        positions = self.positions.copy()
        velocities = self.velocities.copy()
        e = self.energies
        force, u_el, diss, stretch = self._bond_forces(positions, velocities)
        wall_force, u_wall, wall_diss = (np.zeros_like(positions), 0.0, 0.0)
        if cfg.loading == "wall":
            wall_force, u_wall, wall_diss = self._wall_forces(positions, velocities)
        accel = (force + wall_force) / self.masses[:, None]
        peak = float(np.max(np.abs(velocities))) if velocities.size else 0.0
        for step in range(1, n_steps + 1):
            v_half = velocities + 0.5 * dt * accel
            positions = positions + dt * v_half
            force, u_el, diss, stretch = self._bond_forces(positions, v_half)
            wall_force, u_wall, wall_diss = (np.zeros_like(positions), 0.0, 0.0)
            if cfg.loading == "wall":
                wall_force, u_wall, wall_diss = self._wall_forces(positions, v_half)
            broke = self._fail_bonds(stretch)
            if broke:
                force, u_el, diss, stretch = self._bond_forces(positions, v_half)
                if cfg.loading == "wall":
                    wall_force, u_wall, wall_diss = self._wall_forces(positions, v_half)
                self._accumulate_exposure(dt)
                self._exposure_clock = 0.0
                self._prev_labels = self.labels.copy()
                self._census(positions)
                self._prev_labels = self.labels.copy()
            accel = (force + wall_force) / self.masses[:, None]
            velocities = v_half + 0.5 * dt * accel
            self.t = step * dt
            e["e_dissipated"] += diss * dt
            e["e_wall_dissipated"] += wall_diss * dt
            e["e_fracture"] += self._broken_energy_pending
            self._broken_energy_pending = 0.0
            peak = max(peak, float(np.max(np.abs(velocities))))
            ke = 0.5 * self.mass * float(np.sum(velocities**2))
            self._record(step, n_steps, ke, u_el, e)
        e["peak_speed_ratio"] = peak / max(cfg.impact_velocity, 1e-30)
        return self._finalise(positions, velocities, e, u_el, u_wall)


def run_impact(config: DynamicImpactConfig) -> DynamicImpactResult:
    """Convenience wrapper: build and run one loading."""
    return DynamicFragmentation(config).run()


def impact_scan(
    sizes=(12, 16, 24, 32, 48),
    velocities=(0.5, 1.0, 2.0, 4.0, 8.0),
    *,
    seeds=(0, 1, 2, 3),
    ny_ratio: float = 0.5,
    loading: str = "stretch",
    dt: float = 2.0e-6,
    t_end: float = 1.5e-3,
    spacing: float = 0.03,
) -> list:
    """Impact matrix over lattice size x impact velocity x disorder seed."""
    results = []
    for nx in sizes:
        ny = max(4, int(round(float(ny_ratio) * nx)))
        for v0 in velocities:
            for seed in seeds:
                cfg = DynamicImpactConfig(
                    nx=int(nx), ny=int(ny), spacing=float(spacing),
                    impact_velocity=float(v0), dt=float(dt), t_end=float(t_end),
                    loading=str(loading), seed=int(seed),
                )
                results.append(run_impact(cfg))
    return results


def platen_scan(
    sizes=(16, 24, 32, 48),
    seeds=(0, 1, 2),
    *,
    ny_ratio: float = 0.5,
    strain_rate: float = 10.0,
    target_strain: float = 0.05,
    dt: float = 2.0e-6,
    spacing: float = 0.03,
) -> list:
    """Rate-controlled platen matrix over lattice size x disorder seed."""
    results = []
    for nx in sizes:
        ny = max(4, int(round(float(ny_ratio) * nx)))
        t_end = float(target_strain) / float(strain_rate)
        for seed in seeds:
            cfg = DynamicImpactConfig(
                nx=int(nx), ny=int(ny), spacing=float(spacing),
                loading="platen", strain_rate=float(strain_rate),
                target_strain=float(target_strain), t_end=t_end,
                dt=float(dt), seed=int(seed),
            )
            results.append(run_impact(cfg))
    return results