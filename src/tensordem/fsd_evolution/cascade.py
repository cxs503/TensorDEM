"""DEM fragment-cascade extraction: parent -> daughter breakage events.

Why this module exists
----------------------
A floe-size distribution (FSD) is the *output* of a breakage kernel
``K(d, d') = nu(d') * beta(d | d')``.  The population-balance forecast of the
FSD therefore needs two measured ingredients:

* ``beta(d | d')`` -- the daughter-size distribution of a floe of size ``d'``,
* ``nu(d')``        -- the breakage frequency of a floe of size ``d'``.

This module *measures* both on the lattice-fracture instrument of
:mod:`tensordem.broken_ice_micro.fragmentation`, which solves the quasi-static
elastic equilibrium of a bonded ice plate with Weibull-disordered bond
thresholds and removes bonds past their threshold.  Because bonds are only ever
removed, the connected components of the surviving bond network form a nested
sequence of partitions: a fragment that exists at load stage ``k+1`` is a subset
of exactly one fragment of stage ``k``.  That nesting is what makes a genuine
``parent -> daughters`` census possible without any assumption about the
breakage law: we do not postulate ``beta``, we *count* it.

Energy ledger
-------------
The same nesting gives the two-scale energy bookkeeping that closes the
campaign's energy criterion:

* the platen work ``W = int F d(delta)`` (displacement control),
* the elastic energy stored in the surviving bond network,
  ``U = 1/2 sum_alive k (eps_b a)**2``,
* the energy released at a break batch,
  ``E_release = U(before) - U(after)`` at fixed prescribed displacement, of
  which ``E_fracture = 1/2 sum_newly_broken k (eps_b a)**2`` is stored in the
  bonds that actually fail,
* the **new fracture surface** created by the split,
  ``A_new = t * (P_after - P_before)`` with ``P = sum_i pi d_i`` the summed
  perimeter of the area-equivalent floe disks and ``t`` the plate thickness.

``E_fracture / A_new`` is then the effective fracture energy ``G_c`` (J/m^2) of
the lattice, and calibrating ``G_c`` on early break batches and validating it on
the later (held-out) batches is a genuine *predictive* energy-ledger test rather
than an identity.  The complement ``E_release - E_fracture`` is the energy
released by network relaxation at a break; the quasi-static engine drops it (no
inertia), so it is tracked and reported as ``E_relaxation`` -- never hidden.

Honesty / scope
---------------
* Two-dimensional, central-force lattice: no bending moments, so absolute
  strength and absolute ``G_c`` are model numbers, not measurements of sea ice;
  only ratios and exponents are carried into the population balance.
* A detached (non-load-carrying) fragment cannot break in this instrument, so
  the measured hazard ``nu(d)`` is fitted on the load-bearing exposure only; it
  is a *lower bound* for debris.  This is documented and propagated as a caveat
  on the breakage-rate exponent, not smoothed away.
* The parent -> daughter census keeps *all* components (including detached
  debris): the daughter distribution is a property of the material point that
  broke, not of the load path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..broken_ice_micro.fragmentation import (
    FragmentationConfig,
    QuasiStaticFragmentation,
    _component_labels,
)

#: Absolute size grid (m) on which the breakage hazard is exposed and counted.
#: Absolute rather than per-run so that cascades on different lattices with
#: different parent diameters pool into one hazard estimate.
EXPOSURE_BINS = np.logspace(math.log10(0.008), math.log10(2.5), 25)

__all__ = [
    "EXPOSURE_BINS",
    "CascadeConfig",
    "SplitEvent",
    "CascadeResult",
    "FragmentationCascade",
    "fragment_perimeter",
    "run_cascade",
]


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class CascadeConfig:
    """Lattice, disorder and loading parameters of one cascade run (SI)."""

    nx: int = 32
    ny: int = 16
    spacing: float = 0.03
    thickness: float = 0.5
    density: float = 917.0
    young_modulus: float = 5.0e9
    failure_strain_scale: float = 2.0e-3
    weibull_modulus: float = 4.0
    compression_factor: float = 5.0
    strain_increment: float = 2.0e-4
    target_strain: float = 0.05
    target_largest_fraction: float = 0.35
    seed: int = 0
    min_fragment_nodes: int = 1

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "strain_increment", "target_strain", "spacing"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.nx) < 4 or int(self.ny) < 4:
            raise ValueError("nx and ny must be >= 4")
        if int(self.min_fragment_nodes) < 1:
            raise ValueError("min_fragment_nodes must be >= 1")

    def as_fragmentation(self) -> FragmentationConfig:
        """The underlying quasi-static fragmentation configuration."""
        return FragmentationConfig(
            nx=int(self.nx),
            ny=int(self.ny),
            spacing=float(self.spacing),
            thickness=float(self.thickness),
            density=float(self.density),
            young_modulus=float(self.young_modulus),
            failure_strain_scale=float(self.failure_strain_scale),
            weibull_modulus=float(self.weibull_modulus),
            compression_factor=float(self.compression_factor),
            strain_increment=float(self.strain_increment),
            target_strain=float(self.target_strain),
            target_largest_fraction=float(self.target_largest_fraction),
            seed=int(self.seed),
            min_fragment_nodes=int(self.min_fragment_nodes),
            stage_stride=1,
        )

    @property
    def parent_area(self) -> float:
        return self.as_fragmentation().plate_area

    @property
    def parent_diameter(self) -> float:
        """Area-equivalent diameter of the intact floe (m)."""
        return math.sqrt(4.0 * self.parent_area / math.pi)


# --------------------------------------------------------------------------- events
@dataclass
class SplitEvent:
    """One measured ``parent -> daughters`` breakage event."""

    parent_diameter: float
    daughter_diameters: np.ndarray
    parent_area: float
    applied_strain: float
    released_energy: float

    @property
    def n_daughters(self) -> int:
        return int(self.daughter_diameters.size)

    @property
    def new_perimeter(self) -> float:
        """``sum_i pi d_i - pi d'`` -- the perimeter created by the split (m)."""
        return math.pi * (float(self.daughter_diameters.sum()) - self.parent_diameter)

    @property
    def largest_fraction(self) -> float:
        return float(self.daughter_diameters.max() / self.parent_diameter)

    def normalised_sizes(self) -> np.ndarray:
        """Daughter diameters divided by the parent diameter (mass-weighted sense)."""
        return self.daughter_diameters / self.parent_diameter


@dataclass
class CascadeResult:
    """Everything one cascade run yields."""

    events: list[SplitEvent]
    exposure: dict
    energies: dict
    config: CascadeConfig
    final_census: dict
    batches: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def parent_diameter(self) -> float:
        return self.config.parent_diameter

    def as_dict(self) -> dict:
        return dict(
            parent_diameter_m=float(self.parent_diameter),
            n_events=int(len(self.events)),
            n_split_events=int(sum(1 for e in self.events if e.n_daughters >= 2)),
            energies={k: float(v) for k, v in self.energies.items()},
            exposure={
                k: np.asarray(v).tolist() for k, v in self.exposure.items()
            },
            final_census=dict(self.final_census),
            diagnostics=dict(self.diagnostics),
        )


def fragment_perimeter(diameters) -> float:
    """Summed perimeter ``pi sum d`` of area-equivalent floe disks (m)."""
    values = np.asarray(diameters, dtype=np.float64)
    return float(math.pi * values.sum())


# --------------------------------------------------------------------------- engine
class FragmentationCascade:
    """Load a lattice plate step by step and record the breakage cascade."""

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config = config or CascadeConfig()
        self.engine = QuasiStaticFragmentation(self.config.as_fragmentation())
        cfg = self.engine.config
        self._k = cfg.bond_stiffness
        self._a = cfg.spacing
        self._t = cfg.thickness
        self._node_area = cfg.node_area
        self._bins = EXPOSURE_BINS.copy()
        self.exposure = np.zeros(self._bins.size - 1, dtype=np.float64)
        self.split_counts = np.zeros(self._bins.size - 1, dtype=np.int64)
        self.events: list[SplitEvent] = []
        self.energies = dict(
            platen_work=0.0,
            platen_work_energy_consistent=0.0,
            elastic_final=0.0,
            elastic_initial=0.0,
            fracture_energy=0.0,
            fracture_interior=0.0,
            released_energy=0.0,
            new_surface_area=0.0,
            batches=0,
            n_split_events=0,
        )
        self.batch_ledger: list[dict] = []
        #: bonds in the platen contact layer: a displacement-controlled lattice
        #: over-strains the first row (``delta / spacing`` >> the interior
        #: strain), so their failure energy is an instrument boundary artifact
        #: and is reported separately from the specimen-interior fracture work.
        touching = np.zeros(self.engine.n_nodes, dtype=bool)
        touching[self.engine._left] = True
        touching[self.engine._right] = True
        self._boundary_bond = touching[self.engine.bonds[:, 0]] | touching[self.engine.bonds[:, 1]]

    # -- mechanics helpers --------------------------------------------------
    def _elastic_energy(self) -> float:
        strain = self.engine.bond_strain()
        return 0.5 * self._k * (self._a**2) * float(np.sum(strain[self.engine.alive] ** 2))

    def _labels(self) -> np.ndarray:
        return self.engine._components()

    def _spanning_labels(self, labels: np.ndarray) -> np.ndarray:
        left = set(labels[self.engine._left].tolist())
        right = set(labels[self.engine._right].tolist())
        return np.asarray(sorted(left & right), dtype=np.int64)

    def _census_of(self, labels: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Diameter census of every component inside ``nodes`` under ``labels``.

        The floor is one node here (not ``min_fragment_nodes``) so that the
        perimeter accounting that defines the new fracture surface is
        consistent with the daughter lists of the split events.
        """
        if nodes.size == 0:
            return np.zeros(0)
        inside = labels[nodes]
        _, counts = np.unique(inside, return_counts=True)
        counts = counts[counts >= 1]
        areas = counts.astype(np.float64) * self._node_area
        return np.sort(np.sqrt(4.0 * areas / math.pi))[::-1]

    def _size_bin(self, diameter: float) -> int:
        idx = int(np.searchsorted(self._bins, diameter, side="right") - 1)
        return min(max(idx, 0), self._bins.size - 2)

    # -- main loop ----------------------------------------------------------
    def run(self) -> CascadeResult:
        engine = self.engine
        cfg = self.config
        node_length = engine.config.node_length
        self.energies["elastic_initial"] = self._elastic_energy()
        force_prev = 0.0
        # energy-consistent reaction: for a fixed bond set U is exactly quadratic
        # in the prescribed displacement, so F = dU/d(delta) = 2 U / delta.
        force_ec_prev = 0.0
        labels_prev = self._labels()
        strain_prev = engine.applied_strain

        while engine.applied_strain < cfg.target_strain - 1e-15:
            engine.applied_strain = min(
                cfg.target_strain, engine.applied_strain + cfg.strain_increment
            )
            delta = (engine.applied_strain - strain_prev) * node_length
            engine.solve()
            force_now = engine.wall_force()
            self.energies["platen_work"] += 0.5 * (force_prev + force_now) * delta
            displacement = engine.applied_strain * node_length
            force_ec = 2.0 * self._elastic_energy() / displacement if displacement > 0 else 0.0
            self.energies["platen_work_energy_consistent"] += 0.5 * (
                force_ec_prev + force_ec
            ) * delta
            force_ec_prev = force_ec
            strain_prev = engine.applied_strain
            force_prev = force_now

            # hazard exposure: only load-carrying fragments can break
            spanning = set(self._spanning_labels(labels_prev).tolist())
            for lab in spanning:
                nodes = np.flatnonzero(labels_prev == lab)
                diameter = math.sqrt(4.0 * nodes.size * self._node_area / math.pi)
                self.exposure[self._size_bin(diameter)] += cfg.strain_increment

            # break cascade at fixed prescribed displacement
            for _ in range(60):
                strain = engine.bond_strain()
                alive = engine.alive
                over = alive & (
                    (strain > engine.thresholds)
                    | (strain < -cfg.compression_factor * engine.thresholds)
                )
                if not over.any():
                    break
                energy_fracture = 0.5 * self._k * (self._a**2) * float(np.sum(strain[over] ** 2))
                interior = over & ~self._boundary_bond
                energy_interior = 0.5 * self._k * (self._a**2) * float(
                    np.sum(strain[interior] ** 2)
                )
                u_before = self._elastic_energy()
                census_before = self._census_of(labels_prev, np.arange(engine.n_nodes))
                engine.break_bonds()
                engine.solve()
                u_after = self._elastic_energy()
                labels_new = self._labels()
                self._record_batch(
                    labels_prev, labels_new, census_before, u_before - u_after,
                    energy_fracture, energy_interior,
                )
                labels_prev = labels_new

            # the post-cascade platen force is the start of the next load
            # increment: a break relaxes the network at fixed displacement, so
            # integrating the work from the pre-break force would book that
            # relaxation as lost work and the energy ledger would not close.
            force_prev = engine.wall_force()
            if engine.applied_strain > 0.0:
                force_ec_prev = 2.0 * self._elastic_energy() / (
                    engine.applied_strain * node_length
                )

            census = engine.census()
            if census.n_fragments > 1 and census.largest_fraction < cfg.target_largest_fraction:
                break

        census = engine.census()
        self.energies["elastic_final"] = self._elastic_energy()
        return CascadeResult(
            events=self.events,
            exposure=dict(
                edges=self._bins.copy(),
                exposure=self.exposure.copy(),
                splits=self.split_counts.copy(),
            ),
            energies=dict(self.energies),
            config=cfg,
            final_census=census.summary(),
            batches=list(self.batch_ledger),
            diagnostics=dict(
                applied_strain=float(engine.applied_strain),
                broken_bonds=int((~engine.alive).sum()),
                batches=int(len(self.batch_ledger)),
                solve_count=int(engine.solve_count),
            ),
        )

    # -- batch bookkeeping --------------------------------------------------
    def _record_batch(
        self,
        labels_old: np.ndarray,
        labels_new: np.ndarray,
        census_before: np.ndarray,
        released: float,
        energy_fracture: float,
        energy_interior: float = 0.0,
    ) -> None:
        """Attribute one break batch to parent -> daughter split events."""
        engine = self.engine
        if not np.any(labels_old >= 0):
            return
        census_after = self._census_of(labels_new, np.arange(engine.n_nodes))
        new_area = max(
            self._t
            * (fragment_perimeter(census_after) - fragment_perimeter(census_before)),
            0.0,
        )

        batch_events: list[SplitEvent] = []
        for lab in np.unique(labels_old):
            if lab < 0:
                continue
            nodes = np.flatnonzero(labels_old == lab)
            area = nodes.size * self._node_area
            parent_d = math.sqrt(4.0 * area / math.pi)
            inside = labels_new[nodes]
            sizes = []
            for child in np.unique(inside):
                if child < 0:
                    continue
                count = int(np.count_nonzero(inside == child))
                if count >= 1:
                    sizes.append(math.sqrt(4.0 * count * self._node_area / math.pi))
            if len(sizes) < 2:
                continue
            batch_events.append(
                SplitEvent(
                    parent_diameter=parent_d,
                    daughter_diameters=np.sort(np.asarray(sizes, dtype=np.float64))[::-1],
                    parent_area=area,
                    applied_strain=float(engine.applied_strain),
                    released_energy=0.0,
                )
            )

        # ---- energy ledger: recorded for EVERY break batch, independent of
        # whether the batch produced a countable split event, so that the
        # audit ``W = dU + sum(E_release)`` is not silently truncated.
        self.energies["batches"] += 1
        self.energies["fracture_energy"] += float(energy_fracture)
        self.energies["fracture_interior"] += float(energy_interior)
        self.energies["released_energy"] += float(released)
        self.energies["new_surface_area"] += float(new_area)
        self.energies["n_split_events"] = self.energies.get("n_split_events", 0) + len(
            batch_events
        )
        self.batch_ledger.append(
            dict(
                strain=float(engine.applied_strain),
                n_events=len(batch_events),
                released=float(released),
                fracture=float(energy_fracture),
                fracture_interior=float(energy_interior),
                new_area=float(new_area),
            )
        )

        perimeters = np.array([ev.new_perimeter for ev in batch_events], dtype=np.float64)
        total = float(perimeters.sum())
        for event, perimeter in zip(batch_events, perimeters):
            share = (perimeter / total) if total > 0.0 else (1.0 / max(len(batch_events), 1))
            event.released_energy = float(released * share)
            self.events.append(event)
            self.split_counts[self._size_bin(event.parent_diameter)] += 1


def run_cascade(config: CascadeConfig | None = None) -> CascadeResult:
    """Convenience wrapper: build the engine and run the cascade."""
    return FragmentationCascade(config).run()