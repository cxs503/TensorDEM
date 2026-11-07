"""The ice-zone structural-health digital twin (online chain).

Per monitoring window (default 1 s at 100 Hz) the twin

1. inverts the modal coordinates from the strain gauges (least squares),
2. smooths and differentiates them and forms the modal forces
   ``F_n = m q'' + c q' + k q``,
3. projects the modal forces back to the total ice force under the contact
   patch (Galerkin) and to the spatial distribution ``f_hat g(x - x0)``,
4. splits the strain record into rigid-body (< 0.5 Hz) and elastic
   (springing/whipping, > 0.5 Hz) parts,
5. updates the Miner fatigue damage incrementally (windowed rainflow),
6. compares the inverted load with the DEM forecast when one is supplied.

All steps are numpy-sized; the per-window latency is measured and reported
(campaign criterion: < 50 ms at 100 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from ..whipping.fatigue import DNV_CURVES, SNCurve
from .dem_compare import LoadComparison, compare_loads
from .fatigue_update import IncrementalFatigue
from .load_inversion import (differentiate, invert_modal_force,
                            reconstruct_total_force)
from .modal_filter import ModalFilter
from .rom import FreeFreeBeamROM
from .separation import rigid_elastic_split

__all__ = ["TwinStep", "IceTwin"]


@dataclass
class TwinStep:
    """Result of one online update."""

    q: np.ndarray                 # (n_modes, n_win) modal coordinates
    f_hat: np.ndarray             # (n_win,) inverted total ice force (N)
    f_dem: np.ndarray | None      # (n_win,) DEM forecast (N) or None
    rigid: np.ndarray             # (n_win,) low-frequency strain part
    elastic: np.ndarray           # (n_win,) high-frequency strain part
    frac_rigid: float
    frac_elastic: float
    sigma: np.ndarray             # (n_win,) deck stress (Pa)
    damage_increment: float
    damage_total: float
    comparison: LoadComparison | None
    latency_ms: float

    def as_dict(self) -> dict:
        d = {
            "damage_increment": self.damage_increment,
            "damage_total": self.damage_total,
            "frac_rigid": self.frac_rigid,
            "frac_elastic": self.frac_elastic,
            "latency_ms": self.latency_ms,
            "f_hat_peak_N": float(np.abs(self.f_hat).max()),
        }
        if self.comparison is not None:
            d["comparison"] = self.comparison.as_dict()
        return d


@dataclass
class IceTwin:
    """Online ice-load inversion + fatigue twin on a FreeFreeBeamROM."""

    rom: FreeFreeBeamROM
    x_gauges: np.ndarray
    x_load: float
    patch_width: float
    fs: float = 100.0
    window_s: float = 1.0
    curve: SNCurve = field(default_factory=lambda: DNV_CURVES["D_air"])
    stress_weights: np.ndarray | None = None   # w_n (Pa per unit q_n)
    x_station: float | None = None
    E: float = 2.0e11
    y_deck: float = 10.0
    smooth_cutoff_hz: float | None = None      # None -> 1.5 x top mode
    snr_db: float = 30.0
    seed: int = 0

    _filter: ModalFilter = field(init=False, repr=False)
    _fatigue: IncrementalFatigue = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _latencies: list = field(default_factory=list, init=False, repr=False)
    _steps: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._filter = ModalFilter(self.rom, self.x_gauges)
        self._fatigue = IncrementalFatigue(self.curve)
        self._rng = np.random.default_rng(self.seed)
        if self.stress_weights is None:
            if self.x_station is None:
                self.x_station = 0.5 * self.rom.L
            self.stress_weights = self.rom.stress_weights(
                self.x_station, self.E, self.y_deck)
        if self.smooth_cutoff_hz is None:
            # keep the smoothing inside the Nyquist band of the monitor
            self.smooth_cutoff_hz = min(1.5 * float(self.rom.freq_hz[-1]),
                                       0.45 * self.fs)
        self.dt = 1.0 / self.fs
        self.n_win = max(4, int(round(self.window_s * self.fs)))

    # ------------------------------------------------------------------ #
    def simulate_strain(self, f, t, snr_db: float | None = None) -> np.ndarray:
        """Forward model: ice force -> gauge strains (with optional noise).

        ``snr_db=None`` uses the twin's configured SNR; pass ``float('inf')``
        for the noise-free forward model.
        """
        sol = self.rom.forward(f, t, self.x_load, self.patch_width)
        eps = self.rom.strain(sol.q, self.x_gauges)
        snr = self.snr_db if snr_db is None else snr_db
        if snr_db is None or snr_db >= 999.0:
            return eps
        sig = float(np.sqrt(np.mean(eps ** 2)))
        if sig == 0.0:
            return eps
        sigma = sig / (10.0 ** (snr_db / 20.0))
        return eps + self._rng.normal(0.0, sigma, eps.shape)

    # ------------------------------------------------------------------ #
    def update(self, eps_window, f_dem: np.ndarray | None = None) -> TwinStep:
        """Process one window of gauge strains; returns the step result."""
        t0 = time.perf_counter()
        eps = np.atleast_2d(np.asarray(eps_window, dtype=float))
        if eps.shape[0] != self._filter.H.shape[0]:
            raise ValueError("eps_window must have one row per gauge")
        n = eps.shape[1]
        t = self.dt * np.arange(n)

        q = self._filter.extract(eps)
        qdot, qddot = differentiate(q, self.dt, self.smooth_cutoff_hz)
        F_modal = invert_modal_force(self.rom, q, qdot, qddot)
        f_hat = reconstruct_total_force(self.rom, F_modal,
                                       self.x_load, self.patch_width)

        eps_mid = eps.mean(axis=0)
        split = rigid_elastic_split(eps_mid, self.dt)

        sigma = self.stress_weights @ q
        dD = self._fatigue.update(sigma)

        comparison = None
        if f_dem is not None:
            comparison = compare_loads(f_hat, np.asarray(f_dem, dtype=float),
                                      self.dt)
        latency = (time.perf_counter() - t0) * 1.0e3
        step = TwinStep(q=q, f_hat=f_hat, f_dem=f_dem,
                        rigid=split["rigid"], elastic=split["elastic"],
                        frac_rigid=split["frac_rigid"],
                        frac_elastic=split["frac_elastic"],
                        sigma=sigma, damage_increment=dD,
                        damage_total=self._fatigue.damage,
                        comparison=comparison, latency_ms=latency)
        self._latencies.append(latency)
        self._steps.append(step)
        return step

    def finish(self) -> float:
        """Close the fatigue record (residual half cycles)."""
        return self._fatigue.finish()

    @property
    def damage(self) -> float:
        return self._fatigue.damage

    @property
    def remaining_life_cycles(self) -> float:
        return self._fatigue.remaining_life_cycles

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    def as_dict(self) -> dict:
        return {
            "rom": self.rom.as_dict(),
            "x_gauges": [float(v) for v in self._filter.x_gauges],
            "x_load": self.x_load,
            "patch_width": self.patch_width,
            "fs_hz": self.fs,
            "window_s": self.window_s,
            "snr_db": self.snr_db,
            "damage": self.damage,
            "remaining_life_cycles": self.remaining_life_cycles,
            "mean_latency_ms": self.mean_latency_ms,
            "n_windows": len(self._steps),
        }
