"""Keel consolidation: deposition + wave-shaken DEM of the rubble skeleton.

Pipeline
--------
1. **Parcel mapping** -- the keel frustum volume that holds the block set at the
   target initial porosity is mapped onto an equivalent periodic cell
   (:func:`..keel_bed.parcel_from_keel`): the cell footprint is the keel
   cross-section at the parcel mid-depth, the cell height follows from mass
   balance.
2. **Deposition** -- blocks are scattered dilutely in the cell and settle under
   gravity.  The end of this stage is ``t = 0`` of the evolution: the loosely
   packed, freshly formed keel.
3. **Wave shaking** -- a cyclic horizontal acceleration
   ``a_x = A sin(2 pi t / T)`` (plus the secondary y component) shakes the bed at
   the wave period; the rubble rearranges and the void ratio decays.
4. **Recording** -- the void ratio is measured two independent ways
   (:func:`..rubble_dem.cell_porosity`): the oedometer/void-ratio definition and
   a Monte-Carlo point estimator.  Their agreement is the estimator's control.

The DEM is :class:`..rubble_dem.RubbleDEM`.  Everything downstream of the DEM
(maintenance of ``ConsolidationConfig``/``ConsolidationResult`` and
:func:`compute_porosity_timeseries`) keeps the historical names so existing
callers and tests continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import KeelGeometry
from .packing import PackingResult, ICE_DENSITY
from .keel_bed import (CellParcel, dilute_cell_positions, parcel_from_keel,
                       solid_volume_of)
from .rubble_dem import DEMRubbleConfig, RubbleDEM, net_effective_gravity

__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "run_consolidation",
    "compute_porosity_timeseries",
    "build_cell",
]


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration of the keel consolidation run.

    Attributes
    ----------
    dt : float
        DEM time step (s).  Must respect ``dt < 0.2 sqrt(m_min/k_n)``; the
        smallest block (d = 0.2 m) sets the bound, see
        :attr:`..rubble_dem.RubbleDEM.dt_max_stable`.
    deposit_time : float
        Duration of the deposition stage (s); excluded from the recorded
        evolution time series (it ends at t = 0).
    total_time : float
        Duration of the shaken (consolidation) stage (s).
    record_interval : float
        Sampling interval of the porosity record (s).
    wave_amplitude, wave_period : float
        Shaking amplitude (m/s2) and period (s).
    gravity : float
        Net *effective* gravity of the skeleton (m/s2, +z downward).  The
        default is the dry self-weight idealisation; use
        :func:`..rubble_dem.net_effective_gravity` for the submerged (buoyant)
        value, which is negative for ice.
    mode : str
        ``"cell"`` (periodic cell, default) or ``"keel"`` (rigid frustum walls;
        arches and jams, kept for the geometry-sensitivity study).
    """

    dt: float = 1.0e-3
    deposit_time: float = 20.0
    total_time: float = 600.0
    record_interval: float = 5.0
    gravity: float = 9.81
    kn: float = 2.0e4
    kt_ratio: float = 0.5
    friction: float = 0.70
    rolling_friction: float = 0.05
    restitution: float = 0.25
    wall_friction: float = 0.30
    damping_ratio: float = 0.10
    wave_amplitude: float = 0.5
    wave_period: float = 10.0
    wave_enabled: bool = True
    placement_porosity: float = 0.75
    porosity_points: int = 40000
    mode: str = "cell"
    rotation: bool = False
    seed: int = 42

    def to_dem(self, parcel: CellParcel) -> DEMRubbleConfig:
        return DEMRubbleConfig(
            mode=self.mode,
            box_size=parcel.box_size,
            floor_depth=parcel.floor,
            dt=self.dt,
            kn=self.kn,
            kt_ratio=self.kt_ratio,
            friction=self.friction,
            rolling_friction=self.rolling_friction,
            restitution=self.restitution,
            wall_friction=self.wall_friction,
            damping_ratio=self.damping_ratio,
            gravity=self.gravity,
            wave_amplitude=self.wave_amplitude,
            wave_period=self.wave_period,
            wave_enabled=self.wave_enabled,
            rotation=self.rotation,
            wave_start_time=self.deposit_time,
            total_time=self.deposit_time + self.total_time,
            record_interval=self.record_interval,
            porosity_points=self.porosity_points,
            seed=self.seed,
        )


@dataclass
class ConsolidationResult:
    """Result of a keel consolidation run.

    ``time`` starts at 0, the end of the deposition stage (freshly formed keel).
    """

    time: np.ndarray
    porosity: np.ndarray
    mean_height: np.ndarray
    kinetic_energy: np.ndarray
    final_positions: np.ndarray
    final_radii: np.ndarray
    config: ConsolidationConfig
    porosity_point: np.ndarray | None = None
    surface_depth: np.ndarray | None = None
    coordination: np.ndarray | None = None
    elastic_energy: np.ndarray | None = None
    energy_ledger: dict = field(default_factory=dict)
    parcel: CellParcel | None = None
    deposition: dict = field(default_factory=dict)
    seed: int = 42
    wall_penetration_max: float = 0.0

    @property
    def solid_volume(self) -> float:
        return solid_volume_of(self.final_radii)


def build_cell(radii: np.ndarray, geom: KeelGeometry,
               config: ConsolidationConfig | None = None) -> CellParcel:
    """Parcel mapping for a block set (keel volume -> periodic cell)."""
    cfg = config or ConsolidationConfig()
    return parcel_from_keel(geom, solid_volume_of(radii), porosity=0.40,
                            placement_porosity=cfg.placement_porosity)


def run_consolidation(
    packing: PackingResult | np.ndarray,
    geom: KeelGeometry,
    config: ConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Deposit the rubble, then shake it and record the void ratio.

    Parameters
    ----------
    packing : PackingResult or ndarray
        Either a :class:`..packing.PackingResult` or the block radii (N,).
    geom : KeelGeometry
        Keel container geometry (sets the parcel size and the container walls).
    config : ConsolidationConfig, optional

    Returns
    -------
    ConsolidationResult
    """
    if config is None:
        config = ConsolidationConfig()
    radii = packing.radii if hasattr(packing, "radii") else np.asarray(packing, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) == 0:
        raise ValueError("empty block set, cannot run consolidation")

    parcel = build_cell(radii, geom, config)
    rng = np.random.default_rng(config.seed)
    dilute = dilute_cell_positions(parcel, radii, rng,
                                   placement_porosity=config.placement_porosity)
    dem_cfg = config.to_dem(parcel)
    sim = RubbleDEM(dilute.positions, dilute.radii, geom, dem_cfg)
    dem = sim.run()

    t_all = dem.time
    keep = t_all >= config.deposit_time - 1e-9
    time = t_all[keep] - config.deposit_time
    # Primary record = the oedometer / void-ratio definition
    # n = 1 - V_solid/(A*h_bed), the classical measure of a bed in a container.
    # The Monte-Carlo point estimator is kept as the independent cross-check:
    # the two differ by the half-block caps above the mean free surface and by
    # the contact overlap, measured at ~0.01 (see the docs' error table).
    porosity = dem.porosity_slab[keep]
    porosity_point = dem.porosity_point[keep]
    surface = dem.surface_depth[keep]
    coordination = dem.coordination[keep]
    ke = dem.kinetic_energy[keep]
    ee = dem.elastic_energy[keep]
    # mean block depth below the free surface (bed thickness measure)
    mean_height = np.array([float(np.mean(r)) for r in [surface]]) * 0.0
    bed_thickness = float(np.max(surface) - np.min(surface)) if len(surface) else 0.0
    mean_height = np.full(len(time), bed_thickness)

    deposition = dict(
        placement_porosity=dilute.placement_porosity,
        n_blocks=int(len(radii)),
        deposit_time=config.deposit_time,
        porosity_at_t0=float(porosity[0]) if len(porosity) else float("nan"),
        porosity_point_at_t0=float(porosity_point[0]) if len(porosity_point) else float("nan"),
        attempts_used=dilute.attempts_used,
    )
    return ConsolidationResult(
        time=time, porosity=porosity, mean_height=mean_height,
        kinetic_energy=ke, final_positions=dem.positions, final_radii=dem.radii,
        config=config, porosity_point=porosity_point, surface_depth=surface,
        coordination=coordination, elastic_energy=ee,
        energy_ledger=dem.energy_ledger, parcel=parcel, deposition=deposition,
        seed=config.seed,
        wall_penetration_max=float(getattr(sim, "wall_penetration_max", 0.0)),
    )


def compute_porosity_timeseries(result: ConsolidationResult):
    """Return ``(time, porosity)`` from a consolidation result."""
    return result.time, result.porosity
