"""Bidisperse granular segregation under shear and vibration (P155-N09).

Physical background
-------------------
A bed of two particle sizes segregates when it is sheared or vibrated.  The two
canonical mechanisms are *kinetic sieving* (small particles percolate down
through the voids opened by shear and push the large ones up) and *squeeze
expulsion*; their competition decides whether the large species ends up at the
top (Brazil nut) or at the bottom (reverse Brazil nut).

This module is the verification surface for that physics on top of the
frictional-disk engine :mod:`tensordem.granular2d.engine`:

1. **Shear cell (Couette)** -- periodic in x, frictional top and bottom walls,
   top wall moving at ``v_top = gamma_dot * H``.  The wall tangential velocity is
   a first-class engine quantity (``DiskSystem(wall_velocity=...)``), so the
   drive is a friction boundary condition, not a velocity overwrite of the
   particles.
2. **Vibration cell** -- walls on all sides, integrated in the *shaker frame*,
   where the walls are static and the effective vertical acceleration is
   ``-g + A w^2 sin(w t)``.
3. **Segregation metric** -- label based and area weighted:

   * ``Phi(z)`` is the solid area fraction of the *large* species per height bin.
     Species membership comes from the particle labels (or a global radius
     threshold ``r_thr = sqrt(r_s r_l)``), never from the particles that happen
     to sit in a bin.
   * ``S = sum_b w_b |Phi_b - Phi_0| / sum_b w_b`` with ``w_b`` the solid area in
     bin ``b``; empty bins carry no weight.
   * direction ``dPhi = <Phi>_top - <Phi>_bottom`` with a binomial standard error
     from the per-bin effective particle number; a sign is declared only when
     ``|dPhi| > 2 se``.

   ``Phi_0`` and the finite-size noise floor are measured, not assumed.
4. **Scaling law** -- ``S = A (d_l/d_s - 1)^alpha`` fitted by least squares on
   ``S`` itself, with a bootstrap CI over seeds.
5. **Kozeny-Carman permeability** of the segregated column,
   ``k(z) = d_eff^2 eps^3 / (180 (1-eps)^2)`` with
   ``d_eff^2 = Phi d_l^2 + (1-Phi) d_s^2``, feeding a 1D Terzaghi consolidation
   solve that yields the excess-pore-pressure sensitivity table.
6. **Repose angle** of the stratified bed measured with the P96-N3 tooling
   (:func:`tensordem.repose_calibration.repose_angle`) on the surface cloud of a
   column collapse.

Everything is SI.  The 2D formulation is deliberate: it keeps the campaign
affordable and makes the statistics (effective particle number per bin) explicit.

References
----------
- Thornton, Ciomocos & Nayak (2006), Granular Matter 8, 197.
- Guillard, Forterre & Pouliquen (2016), PRE 94, 022904.
- Hong, Quinn & Luding (2001), Phys. Rev. Lett. 86, 3423 (reverse Brazil nut).
- Kozeny (1927); Carman (1937) -- permeability of packed beds.
- Terzaghi (1943) -- one-dimensional consolidation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence

import numpy as np

from .granular2d.engine import DiskConfig, DiskSystem, box_walls

__all__ = [
    "BidisperseConfig",
    "Packing",
    "ShearBoxCase",
    "VibrationBoxCase",
    "generate_packing",
    "run_shear_box",
    "run_vibration_box",
    "concentration_profile_data",
    "compute_concentration_profile",
    "compute_segregation_degree",
    "noise_floor",
    "segregation_direction_test",
    "compute_segregation_direction",
    "fit_scaling_law",
    "bootstrap_scaling_law",
    "kozeny_carman_permeability",
    "stratified_permeability_profile",
    "consolidation_response",
    "permeability_sensitivity_table",
    "surface_cloud",
    "measure_rest_angle",
    "collapse_and_measure_repose",
    "run_monodisperse_negative_control",
    "run_bidisperse_parameter_sweep",
    "BidisperseSegregationReport",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BidisperseConfig:
    """Material, geometry and contact parameters of a bidisperse bed (SI)."""

    d_small: float = 0.010
    d_large: float = 0.015
    density: float = 2500.0
    thickness: float = 0.010
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    zeta: float = 0.3
    mu: float = 0.5
    mu_rolling: float = 0.05
    phi_large: float = 0.5          # initial large-species AREA fraction
    volume_fraction: float = 0.60   # target solid area fraction
    n_particles: int = 150
    gravity: float = 9.81
    seed: int = 42

    @property
    def size_ratio(self) -> float:
        return self.d_large / self.d_small

    @property
    def r_small(self) -> float:
        return 0.5 * self.d_small

    @property
    def r_large(self) -> float:
        return 0.5 * self.d_large

    @property
    def r_threshold(self) -> float:
        """Radius threshold separating the two species (geometric mean)."""
        return math.sqrt(self.r_small * self.r_large)

    @property
    def n_large(self) -> int:
        """Number of large particles giving ``phi_large`` as an AREA fraction.

        Segregation studies are quoted at equal *volume* (here area) fractions of
        the two species; equal number fractions would make the large species cover
        90 % of the area at ``d_l/d_s = 3`` and leave no room for ``Phi`` to move.
        """
        a_l, a_s = self.r_large**2, self.r_small**2
        n = int(self.n_particles)
        f = float(self.phi_large)
        return int(round(f * n * a_s / ((1.0 - f) * a_l + f * a_s)))

    @property
    def mean_particle_area(self) -> float:
        n_l = self.n_large
        n_s = int(self.n_particles) - n_l
        return math.pi * (n_l * self.r_large**2 + n_s * self.r_small**2) / self.n_particles

    @property
    def box_area(self) -> float:
        """Cell area that holds ``n_particles`` at the target solid fraction."""
        return self.n_particles * self.mean_particle_area / self.volume_fraction

    @property
    def box_size(self) -> float:
        return math.sqrt(self.box_area)

    def __post_init__(self) -> None:
        for name in ("d_small", "d_large", "density", "thickness", "kn", "gravity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.d_large < self.d_small:
            raise ValueError("d_large must be >= d_small")
        for name in ("kt_ratio", "zeta", "mu", "mu_rolling"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.phi_large < 1.0:
            raise ValueError("phi_large must be in (0, 1)")
        if not 0.0 < self.volume_fraction < 0.90:
            raise ValueError("volume_fraction must be in (0, 0.90)")
        if int(self.n_particles) < 8:
            raise ValueError("n_particles must be at least 8")

    def disk_config(self, *, damping_time: float = 0.0) -> DiskConfig:
        return DiskConfig(
            kn=self.kn, kt_ratio=self.kt_ratio, zeta=self.zeta, mu=self.mu,
            mu_rolling=self.mu_rolling, density=self.density,
            thickness=self.thickness, gravity=(0.0, -self.gravity),
            damping_time=damping_time,
        )


@dataclass(frozen=True)
class Packing:
    """Initial state of the bed: positions, radii, species labels."""

    positions: np.ndarray
    radii: np.ndarray
    is_large: np.ndarray

    @property
    def n(self) -> int:
        return len(self.radii)

    @property
    def areas(self) -> np.ndarray:
        return math.pi * self.radii**2

    @property
    def solid_fraction_large(self) -> float:
        """Large-species area fraction -- the reference ``Phi_0``."""
        return float(self.areas[self.is_large].sum() / self.areas.sum())


@dataclass(frozen=True)
class ShearBoxCase:
    """Couette shear cell driven by a moving frictional top wall."""

    config: BidisperseConfig
    shear_rate: float = 10.0        # gamma_dot (1/s)
    target_strain: float = 5.0      # total shear strain gamma_dot * T
    normal_stress: float = 1000.0   # Pa, dead weight on the load plate
    compact_steps: int = 20000      # plate equilibration before production
    n_bins: int = 12
    sample_every: int = 2000        # steps between profile samples
    max_steps: int = 600_000
    stratified_init: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.shear_rate) or self.shear_rate <= 0.0:
            raise ValueError("shear_rate must be finite and positive")
        if not math.isfinite(self.target_strain) or self.target_strain <= 0.0:
            raise ValueError("target_strain must be finite and positive")
        if int(self.n_bins) < 2:
            raise ValueError("n_bins must be >= 2")


@dataclass(frozen=True)
class VibrationBoxCase:
    """Vertically shaken cell, integrated in the shaker frame."""

    config: BidisperseConfig
    amplitude: float = 0.002        # A (m)
    frequency: float = 30.0         # f (Hz)
    n_cycles: int = 20
    n_bins: int = 12
    sample_every: int = 500
    stratified_init: bool = False
    label: str = ""

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency

    @property
    def peak_acceleration(self) -> float:
        """Shaking intensity ``Gamma = A w^2 / g``."""
        return self.amplitude * self.omega**2 / 9.81

    def __post_init__(self) -> None:
        for name in ("amplitude", "frequency"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.n_cycles) < 1:
            raise ValueError("n_cycles must be >= 1")


# ---------------------------------------------------------------------------
# Initial packing
# ---------------------------------------------------------------------------

def generate_packing(config: BidisperseConfig, *, seed: int | None = None,
                     stratified: bool = False) -> Packing:
    """Jittered-lattice bidisperse packing inside the square cell.

    The cell is sized so that ``n_particles`` gives the requested solid fraction;
    the lattice spacing is ``sqrt(W H / n)``, which can leave small overlaps
    between the largest pairs.  Those are removed by the deterministic relaxation
    of :func:`_prepare_system` (radius inflation), so nothing is injected into the
    production run.
    """
    rng = np.random.default_rng(config.seed if seed is None else int(seed))
    n = int(config.n_particles)
    n_large = config.n_large
    radii = np.concatenate([np.full(n_large, config.r_large),
                            np.full(n - n_large, config.r_small)])
    W = H = config.box_size
    spacing = math.sqrt(W * H / n)
    nx = max(1, int(math.ceil(W / spacing)))
    ny = int(math.ceil(n / nx))
    points = []
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * W / nx + rng.uniform(-0.09, 0.09) * W / nx
            y = (j + 0.5) * H / ny + rng.uniform(-0.09, 0.09) * H / ny
            points.append((x, y))
    points = np.asarray(points, dtype=np.float64)
    points = points[rng.permutation(len(points))[:n]]

    if stratified:
        order = np.argsort(points[:, 1], kind="stable")
        radii_sorted = np.empty(n)
        radii_sorted[order[:n_large]] = config.r_large
        radii_sorted[order[n_large:]] = config.r_small
        radii = radii_sorted
    else:
        radii = rng.permutation(radii)

    is_large = radii > config.r_threshold
    if config.size_ratio == 1.0:  # degenerate: labels are pure convention
        is_large = np.zeros(n, dtype=bool)
        is_large[rng.permutation(n)[:n_large]] = True

    margin = radii.max()
    points[:, 0] = np.clip(points[:, 0], 0.05 * margin, W - 0.05 * margin)
    points[:, 1] = np.clip(points[:, 1], 0.05 * margin, H - 0.05 * margin)
    return Packing(points, radii, is_large)


def _write_radii(system: DiskSystem, radii: np.ndarray) -> None:
    """Set disk radii together with the mass/inertia that follow from them."""
    system.radii = np.array(radii, dtype=np.float64)
    cfg = system.config
    system.mass = cfg.density * math.pi * system.radii**2 * cfg.thickness
    system.inertia = 0.5 * system.mass * system.radii**2


def _prepare_system(config: BidisperseConfig, packing: Packing, *,
                    walls: np.ndarray, periodic_x: float | None,
                    wall_velocity: np.ndarray | None,
                    settle_steps: int = 400) -> DiskSystem:
    """Build the system and relax overlaps by deterministic radius inflation."""
    cfg = config.disk_config(damping_time=2.0e-3)
    r_target = packing.radii.copy()
    system = DiskSystem(packing.positions, 0.8 * r_target, cfg, walls=walls,
                        periodic_x=periodic_x, wall_velocity=wall_velocity)
    system.step(settle_steps)
    for scale in (0.85, 0.90, 0.95, 1.0):
        _write_radii(system, scale * r_target)
        system.step(settle_steps)
    system.config = config.disk_config(damping_time=0.0)
    system.step(settle_steps)
    return system


# ---------------------------------------------------------------------------
# Concentration profile and segregation metrics
# ---------------------------------------------------------------------------

def concentration_profile_data(positions: np.ndarray, radii: np.ndarray,
                               height: float, n_bins: int = 12, *,
                               is_large: np.ndarray | None = None,
                               r_threshold: float | None = None) -> dict:
    """Area-weighted concentration profile of the large species.

    Returns ``phi_z`` (nan in empty bins), ``area_z`` (solid area per bin, the
    weight), ``count_z``, ``n_eff_z = (sum a)^2 / sum a^2`` (effective particle
    number for the binomial error bar) and the bin edges/centres.

    Species membership is *global*: either the supplied ``is_large`` labels or a
    fixed radius threshold.  It is never inferred from the particles that fall in
    a bin -- doing that flips the sign of the profile whenever a bin happens to
    contain only large particles.
    """
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if len(positions) != len(radii):
        raise ValueError("positions and radii must have the same length")
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("height must be finite and positive")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if is_large is None:
        thr = (math.sqrt(radii.min() * radii.max()) if r_threshold is None
               else float(r_threshold))
        is_large = radii > thr
    is_large = np.asarray(is_large, dtype=bool)
    if is_large.shape != radii.shape:
        raise ValueError("is_large must match radii")

    areas = math.pi * radii**2
    edges = np.linspace(0.0, height, n_bins + 1)
    idx = np.clip(np.digitize(positions[:, 1], edges) - 1, 0, n_bins - 1)

    a_large = float(np.mean(areas[is_large])) if is_large.any() else 0.0
    a_small = float(np.mean(areas[~is_large])) if (~is_large).any() else 0.0

    phi = np.full(n_bins, np.nan)
    var_phi = np.full(n_bins, np.nan)
    area_z = np.zeros(n_bins)
    count_z = np.zeros(n_bins, dtype=np.int64)
    n_eff = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        count_z[b] = int(sel.sum())
        if count_z[b] == 0:
            continue
        a = areas[sel]
        lab = is_large[sel]
        n_b = count_z[b]
        n_l = int(lab.sum())
        area_z[b] = a.sum()
        n_eff[b] = float(n_l if n_l else 1)      # informative large-species count
        phi[b] = a[lab].sum() / a.sum()
        # delta method on the binomial count of large particles: Phi =
        # n_l a_l / (n_l a_l + n_s a_s), so dPhi/dn_l = a_l a_s n_b / total^2
        total = a.sum()
        if a_large > 0.0 and a_small > 0.0 and total > 0.0:
            dphi = a_large * a_small * n_b / total**2
            p_l = n_l / n_b
            var_phi[b] = dphi**2 * n_b * p_l * (1.0 - p_l)
    return {
        "phi_z": phi,
        "var_phi_z": var_phi,
        "area_z": area_z,
        "count_z": count_z,
        "n_eff_z": n_eff,
        "bin_edges": edges,
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
    }


def compute_concentration_profile(positions: np.ndarray, radii: np.ndarray,
                                  height: float, n_bins: int = 12, *,
                                  is_large: np.ndarray | None = None,
                                  r_threshold: float | None = None) -> np.ndarray:
    """``Phi(z)`` only; see :func:`concentration_profile_data`."""
    return concentration_profile_data(
        positions, radii, height, n_bins, is_large=is_large,
        r_threshold=r_threshold)["phi_z"]


def compute_segregation_degree(phi_z: Sequence[float], phi_0: float, *,
                               weights: Sequence[float] | None = None) -> float:
    """``S = <|Phi - Phi_0|> / Phi_0`` (nan-safe, area weighted)."""
    phi = np.asarray(phi_z, dtype=np.float64)
    if not (0.0 < phi_0 < 1.0):
        raise ValueError("phi_0 must be in (0, 1)")
    w = (np.ones_like(phi) if weights is None
         else np.asarray(weights, dtype=np.float64))
    if w.shape != phi.shape:
        raise ValueError("weights must match phi_z")
    ok = np.isfinite(phi) & np.isfinite(w) & (w > 0.0)
    if not ok.any():
        return 0.0
    return float(np.sum(w[ok] * np.abs(phi[ok] - phi_0)) / np.sum(w[ok]) / phi_0)


def noise_floor(n_particles: int, phi_0: float = 0.5, n_bins: int = 12) -> float:
    """Expected ``S`` of a perfectly mixed bed at this resolution.

    For ``n`` independent particles spread over ``k`` bins,
    ``<|Phi_b - Phi_0|> ~ sqrt(2/(pi n_b)) Phi_0 (1 - Phi_0)`` with ``n_b = n/k``.
    Any measured ``S`` must beat this floor to count as segregation.
    """
    if n_particles < 2 or n_bins < 2:
        raise ValueError("n_particles and n_bins must be >= 2")
    n_b = n_particles / n_bins
    # E|Phi - phi_0| = sqrt(phi_0 (1-phi_0) / n_b) * sqrt(2/pi); S divides by phi_0
    return float(math.sqrt(2.0 / (math.pi * n_b)) * math.sqrt((1.0 - phi_0) / phi_0))


def segregation_direction_test(phi_z: Sequence[float], *,
                               area_z: Sequence[float] | None = None,
                               n_eff_z: Sequence[float] | None = None,
                               var_phi_z: Sequence[float] | None = None,
                               phi_0: float = 0.5,
                               top_fraction: float = 1.0 / 3.0,
                               z_threshold: float = 2.0) -> dict:
    """Direction and significance of the vertical ``Phi`` gradient.

    ``dPhi = mean(Phi, top third) - mean(Phi, bottom third)``; the standard error
    follows from the binomial variance of each bin, ``Phi(1-Phi)/n_eff``.  A sign
    is declared only when ``|dPhi| > z_threshold * se`` (2 sigma by default),
    else ``0`` ("not resolved").
    """
    phi = np.asarray(phi_z, dtype=np.float64)
    k = len(phi)
    if k < 3:
        raise ValueError("at least three bins are required")
    w = (np.ones(k) if area_z is None else np.asarray(area_z, dtype=np.float64))
    neff = (np.ones(k) if n_eff_z is None
            else np.asarray(n_eff_z, dtype=np.float64))
    ok = (np.isfinite(phi) & np.isfinite(w) & (w > 0.0)
          & np.isfinite(neff) & (neff > 0.0))
    if ok.sum() < 3:
        return {"delta_phi": 0.0, "stderr": float("nan"), "z_score": 0.0,
                "direction": 0, "slope": float("nan"),
                "n_bins_used": int(ok.sum()), "phi_mean": float("nan")}
    order = np.flatnonzero(ok)
    n_take = max(1, int(round(top_fraction * len(order))))
    bottom, top = order[:n_take], order[-n_take:]

    def _mean(sel: np.ndarray) -> float:
        ww = w[sel]
        return float(np.sum(ww * phi[sel]) / np.sum(ww))

    d_phi = _mean(top) - _mean(bottom)
    var = 0.0
    if var_phi_z is not None:
        vphi = np.asarray(var_phi_z, dtype=np.float64)
        for sel in (top, bottom):
            ww = w[sel] / np.sum(w[sel])
            var += float(np.sum(ww**2 * np.nan_to_num(vphi[sel])))
    else:
        for sel in (top, bottom):
            ww = w[sel] / np.sum(w[sel])
            var += float(np.sum(ww**2 * phi[sel] * (1.0 - phi[sel])
                               / np.maximum(neff[sel], 1.0)))
    se = math.sqrt(max(var, 1e-30))
    zs = d_phi / se
    direction = 1 if zs > z_threshold else (-1 if zs < -z_threshold else 0)

    zz = np.arange(k, dtype=np.float64)[ok]
    ww = w[ok]
    A = np.stack([zz, np.ones_like(zz)], axis=1)
    Aw = A * ww[:, None]
    coef = np.linalg.solve(A.T @ Aw, Aw.T @ phi[ok])
    return {
        "delta_phi": float(d_phi),
        "stderr": float(se),
        "z_score": float(zs),
        "direction": int(direction),
        "slope": float(coef[0]),
        "n_bins_used": int(ok.sum()),
        "phi_mean": float(np.sum(ww * phi[ok]) / np.sum(ww)),
    }


def compute_segregation_direction(phi_z: Sequence[float]) -> int:
    """Legacy helper: unweighted direction with a 0.05 dead band."""
    phi = np.asarray(phi_z, dtype=np.float64)
    if not np.isfinite(phi).any():
        return 0
    k = len(phi)
    mid = max(1, k // 2)
    bottom = np.nanmean(phi[:mid])
    top = np.nanmean(phi[mid:])
    if not (np.isfinite(top) and np.isfinite(bottom)):
        return 0
    diff = top - bottom
    return 1 if diff > 0.05 else (-1 if diff < -0.05 else 0)


# ---------------------------------------------------------------------------
# Shear cell
# ---------------------------------------------------------------------------

def _plate_force_up(system: DiskSystem, y_plate: float) -> float:
    """Upward force the particles exert on the top plate (per unit thickness)."""
    c = system.contacts
    if not len(c["i"]):
        return 0.0
    kind = np.asarray(c["kind"])
    point = np.asarray(c["point"])
    f = np.asarray(c["f_on_i"])
    sel = (kind == 1) & (point[:, 1] > 0.5 * y_plate)
    if not sel.any():
        return 0.0
    return float(-np.sum(f[sel, 1]))


def _settle_plate(config: BidisperseConfig, system: DiskSystem, W: float,
                  y_plate: float, vy_plate: float, m_plate: float,
                  *, v_top: float, max_steps: int) -> float:
    """Integrate the plate vertical DOF until the load is balanced."""
    dt = system.dt
    done = 0
    calm = 0
    while done < max_steps:
        n = min(50, max_steps - done)
        system.step(n)
        done += n
        f_up = _plate_force_up(system, y_plate)
        ay = f_up / m_plate - config.gravity - 20.0 * vy_plate
        vy_plate += ay * (n * dt)
        y_plate += vy_plate * (n * dt)
        system.walls[1, :, 1] = y_plate
        system.wall_velocity[1] = (v_top, vy_plate)
        calm = calm + 1 if abs(vy_plate) < 1.0e-4 else 0
        if calm >= 8:
            break
    return float(vy_plate)


def run_shear_box(case: ShearBoxCase, *, return_history: bool = False,
                  settle_steps: int = 400) -> dict:
    """Couette shear cell with a dead-weight load plate (constant normal stress).

    The bed is first relaxed with an open top, then a frictional plate carrying the
    dead weight ``normal_stress * W`` is placed on the free surface.  The plate has
    one vertical degree of freedom (mass-spring-damper, integrated together with
    the DEM) and moves at ``v_top = shear_rate * W`` in +x, so the normal stress on
    the bed is the imposed ``normal_stress`` and the shear is transmitted by
    friction -- the standard constant-pressure shear cell.  The plate vertical
    motion is reported, which also proves that the load is actually applied.
    """
    config = case.config
    packing = generate_packing(config, stratified=case.stratified_init)
    W = config.box_size
    open_walls = box_walls(W, W, left=False, right=False, bottom=True, top=False)
    system = _prepare_system(config, packing, walls=open_walls, periodic_x=W,
                             wall_velocity=None, settle_steps=settle_steps)

    gap = 0.25 * config.d_small
    y_plate = float(np.max(system.pos[:, 1] + system.radii)) + gap
    system.walls = box_walls(W, y_plate, left=False, right=False,
                             bottom=True, top=True)
    v_top = case.shear_rate * W
    m_plate = case.normal_stress * W / config.gravity      # kg per unit thickness
    vy_plate = 0.0
    system.wall_velocity = np.array([[0.0, 0.0], [v_top, vy_plate]])

    dt = system.dt
    gamma_dot = v_top / W
    n_steps = int(min(math.ceil(case.target_strain / (gamma_dot * dt)),
                      int(case.max_steps)))
    labels = packing.is_large
    phi_0 = packing.solid_fraction_large

    def sample() -> dict:
        return concentration_profile_data(system.pos, system.radii, W,
                                          case.n_bins, is_large=labels)

    # --- pre-compaction under the dead weight, no shear -------------------
    vy_plate = _settle_plate(config, system, W, y_plate, vy_plate, m_plate,
                             v_top=0.0, max_steps=int(case.compact_steps))
    y_plate = float(system.walls[1, 0, 1])
    system.wall_velocity = np.array([[0.0, 0.0], [v_top, vy_plate]])
    y_plate = _settle_plate(config, system, W, y_plate, vy_plate, m_plate,
                            v_top=v_top, max_steps=int(case.compact_steps // 2))
    y_plate = float(system.walls[1, 0, 1])
    system.time = 0.0
    system.steps = 0

    initial = sample()
    history = [{"time": 0.0, "strain": 0.0, "S": 0.0, "phi_z": initial["phi_z"].copy()}]
    done = 0
    plate_forces, plate_heights = [], []
    while done < n_steps:
        chunk = min(int(case.sample_every), n_steps - done)
        sub = 0
        while sub < chunk:
            n = min(25, chunk - sub)
            system.step(n)
            sub += n
            f_up = _plate_force_up(system, y_plate)
            ay = f_up / m_plate - config.gravity - 20.0 * vy_plate
            vy_plate += ay * (n * dt)
            y_plate += vy_plate * (n * dt)
            system.walls[1, :, 1] = y_plate
            system.wall_velocity[1] = (v_top, vy_plate)
        done += chunk
        y_plate = float(system.walls[1, 0, 1])
        plate_forces.append(_plate_force_up(system, y_plate))
        plate_heights.append(y_plate)
        history.append({"time": float(system.time),
                        "strain": float(gamma_dot * system.time),
                        "S": compute_segregation_degree(sample()["phi_z"], phi_0),
                        "phi_z": sample()["phi_z"].copy()})

    data = sample()
    S = compute_segregation_degree(data["phi_z"], phi_0, weights=data["area_z"])
    direction = segregation_direction_test(data["phi_z"], area_z=data["area_z"],
                                           n_eff_z=data["n_eff_z"],
                                           var_phi_z=data["var_phi_z"], phi_0=phi_0)
    idx = np.clip(np.digitize(system.pos[:, 1], data["bin_edges"]) - 1,
                  0, case.n_bins - 1)
    vx_z = np.full(case.n_bins, np.nan)
    for b in range(case.n_bins):
        sel = idx == b
        if sel.any():
            vx_z[b] = float(np.mean(system.vel[sel, 0]))
    result = {
        "kind": "shear",
        "label": case.label,
        "size_ratio": config.size_ratio,
        "volume_fraction": config.volume_fraction,
        "shear_rate": case.shear_rate,
        "normal_stress": case.normal_stress,
        "seed": config.seed,
        "n_particles": packing.n,
        "box_size": W,
        "dt": dt,
        "steps": int(done),
        "time": float(system.time),
        "strain": float(gamma_dot * system.time),
        "phi_0": phi_0,
        "phi_z": data["phi_z"],
        "area_z": data["area_z"],
        "n_eff_z": data["n_eff_z"],
        "initial_phi_z": initial["phi_z"],
        "initial_S": compute_segregation_degree(initial["phi_z"], phi_0,
                                                weights=initial["area_z"]),
        "segregation_degree": S,
        "excess_segregation": float(S - compute_segregation_degree(
            initial["phi_z"], phi_0, weights=initial["area_z"])),
        "noise_floor": noise_floor(packing.n, phi_0, case.n_bins),
        "direction": direction["direction"],
        "direction_test": direction,
        "vx_z": vx_z,
        "v_top": float(v_top),
        "shear_penetration": float(np.nanmax(np.abs(vx_z)) / v_top) if v_top else 0.0,
        "plate_height": float(y_plate),
        "solid_fraction_measured": float(
            np.sum(math.pi * system.radii**2) / (W * max(y_plate, 1e-9))),
        "plate_height_initial": float(plate_heights[0]) if plate_heights else float(y_plate),
        "plate_force_mean": (float(np.mean(plate_forces)) if plate_forces else 0.0),
        "plate_force_target": float(case.normal_stress * W),
        "plate_force_history": [float(v) for v in plate_forces],
        "history_S": [{"time": h["time"], "strain": h["strain"], "S": h["S"]}
                      for h in history],
    }
    if return_history:
        result["history"] = history
        result["positions"] = system.pos.copy()
        result["radii"] = system.radii.copy()
        result["is_large"] = labels.copy()
    return result


# ---------------------------------------------------------------------------
# Vibration cell
# ---------------------------------------------------------------------------

def run_vibration_box(case: VibrationBoxCase, *, return_history: bool = False,
                      settle_steps: int = 400) -> dict:
    """Vertically shaken cell integrated in the shaker frame.

    In the frame of the container the walls are static and the effective vertical
    body acceleration is ``-g + A w^2 sin(w t)``.  This is mechanically
    equivalent to shaking the container in the lab frame and avoids moving-wall
    bookkeeping for four walls at once.
    """
    config = case.config
    packing = generate_packing(config, stratified=case.stratified_init)
    W = config.box_size
    walls = box_walls(W, W, left=True, right=True, bottom=True, top=True)
    system = _prepare_system(config, packing, walls=walls, periodic_x=None,
                             wall_velocity=None, settle_steps=settle_steps)
    dt = system.dt
    omega = case.omega
    amp = case.amplitude
    n_steps = max(int(math.ceil(case.n_cycles / (case.frequency * dt))), 1)
    labels = packing.is_large
    phi_0 = packing.solid_fraction_large

    initial = concentration_profile_data(system.pos, system.radii, W,
                                         case.n_bins, is_large=labels)
    history = []
    done = 0
    while done < n_steps:
        chunk = min(int(case.sample_every), n_steps - done)
        t_mid = system.time + 0.5 * chunk * dt
        system.vel[:, 1] += amp * omega**2 * math.sin(omega * t_mid) * chunk * dt
        system.step(chunk)
        done += chunk
        data = concentration_profile_data(system.pos, system.radii, W,
                                          case.n_bins, is_large=labels)
        history.append({"time": float(system.time),
                        "cycle": float(system.time * case.frequency),
                        "S": compute_segregation_degree(data["phi_z"], phi_0)})

    data = concentration_profile_data(system.pos, system.radii, W, case.n_bins,
                                      is_large=labels)
    S = compute_segregation_degree(data["phi_z"], phi_0, weights=data["area_z"])
    direction = segregation_direction_test(data["phi_z"], area_z=data["area_z"],
                                           n_eff_z=data["n_eff_z"],
                                           var_phi_z=data["var_phi_z"], phi_0=phi_0)
    areas = math.pi * system.radii**2
    z_large = float(np.sum(areas[labels] * system.pos[labels, 1]) / areas[labels].sum())
    z_small = float(np.sum(areas[~labels] * system.pos[~labels, 1]) / areas[~labels].sum())
    result = {
        "kind": "vibration",
        "label": case.label,
        "size_ratio": config.size_ratio,
        "volume_fraction": config.volume_fraction,
        "amplitude": case.amplitude,
        "frequency": case.frequency,
        "gamma": case.peak_acceleration,
        "seed": config.seed,
        "n_particles": packing.n,
        "box_size": W,
        "dt": dt,
        "steps": int(done),
        "time": float(system.time),
        "cycles": float(system.time * case.frequency),
        "phi_0": phi_0,
        "phi_z": data["phi_z"],
        "area_z": data["area_z"],
        "n_eff_z": data["n_eff_z"],
        "segregation_degree": S,
        "initial_S": compute_segregation_degree(initial["phi_z"], phi_0,
                                                weights=initial["area_z"]),
        "excess_segregation": float(S - compute_segregation_degree(
            initial["phi_z"], phi_0, weights=initial["area_z"])),
        "noise_floor": noise_floor(packing.n, phi_0, case.n_bins),
        "direction": direction["direction"],
        "direction_test": direction,
        "z_cm_large": z_large,
        "z_cm_small": z_small,
        "dz_cm": z_large - z_small,
        "solid_fraction_measured": float(
            np.sum(math.pi * system.radii**2)
            / (W * max(float(np.max(system.pos[:, 1] + system.radii)), 1e-9))),
        "bed_height": float(np.max(system.pos[:, 1] + system.radii)),
        "history_S": history,
    }
    if return_history:
        result["positions"] = system.pos.copy()
        result["radii"] = system.radii.copy()
        result["is_large"] = labels.copy()
    return result


# ---------------------------------------------------------------------------
# Scaling laws
# ---------------------------------------------------------------------------

def _r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_power(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Least squares on S (not on log S) for ``S = A x^alpha``."""
    from scipy.optimize import least_squares

    def resid(p):
        return np.sqrt(w) * (p[0] * np.power(x, p[1]) - y)

    x0 = np.array([max(float(np.mean(y)), 1e-6), 0.5])
    sol = least_squares(resid, x0, bounds=([0.0, -2.0], [100.0, 5.0]),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000)
    A, alpha = float(sol.x[0]), float(sol.x[1])
    pred = A * np.power(x, alpha)
    return {"model": "power", "A": A, "alpha": alpha, "R2": _r2(y, pred, w),
            "predicted": pred, "n": int(len(x))}


def _fit_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    A = np.stack([x, np.ones_like(x)], axis=1)
    Aw = A * w[:, None]
    coef = np.linalg.solve(A.T @ Aw, Aw.T @ y)
    pred = A @ coef
    return {"model": "linear", "A": float(coef[0]), "intercept": float(coef[1]),
            "slope": float(coef[0]), "R2": _r2(y, pred, w), "predicted": pred,
            "n": int(len(x))}


def fit_scaling_law(size_ratios: Sequence[float],
                    segregation_degrees: Sequence[float], *,
                    model: str = "power",
                    weights: Sequence[float] | None = None) -> dict:
    """Fit ``S = A (d_l/d_s - 1)^alpha`` (or linear); ``R^2`` on ``S``."""
    ratio = np.asarray(size_ratios, dtype=np.float64)
    S = np.asarray(segregation_degrees, dtype=np.float64)
    if ratio.shape != S.shape:
        raise ValueError("size_ratios and segregation_degrees must match")
    x = ratio - 1.0
    keep = np.isfinite(x) & np.isfinite(S) & (x > 0.0)
    if keep.sum() < 2:
        raise ValueError("need at least two distinct size ratios > 1")
    x, S = x[keep], S[keep]
    w = (np.ones_like(S) if weights is None
         else np.asarray(weights, dtype=np.float64)[keep])
    out = _fit_power(x, S, w) if model == "power" else _fit_linear(x, S, w)
    out["x"] = x
    out["y"] = S
    return out


def bootstrap_scaling_law(size_ratios: Sequence[float],
                          samples: Sequence[Sequence[float]], *,
                          model: str = "power", n_boot: int = 2000,
                          seed: int = 0, confidence: float = 0.95) -> dict:
    """Bootstrap the scaling exponent over repeated seeds.

    ``samples[i]`` holds the per-seed ``S`` values of size ratio ``i``.
    """
    rng = np.random.default_rng(seed)
    ratio = np.asarray(size_ratios, dtype=np.float64)
    groups = [np.asarray(g, dtype=np.float64) for g in samples]
    if len(ratio) != len(groups):
        raise ValueError("one sample group per size ratio is required")
    point = fit_scaling_law(ratio, [float(np.mean(g)) for g in groups],
                            model=model)
    key = "alpha" if model == "power" else "slope"
    draws = []
    for _ in range(int(n_boot)):
        y = np.array([float(np.mean(rng.choice(g, size=len(g), replace=True)))
                      for g in groups])
        try:
            draws.append(fit_scaling_law(ratio, y, model=model)[key])
        except Exception:                  # pragma: no cover - degenerate resample
            continue
    draws = np.asarray(draws)
    if len(draws):
        lo, hi = np.percentile(draws, [100 * (1 - confidence) / 2,
                                       100 * (1 + confidence) / 2])
    else:                                  # pragma: no cover
        lo = hi = float("nan")
    point[f"{key}_ci"] = [float(lo), float(hi)]
    point["n_boot"] = int(len(draws))
    point["confidence"] = float(confidence)
    point["per_seed_means"] = [float(np.mean(g)) for g in groups]
    return point


# ---------------------------------------------------------------------------
# Permeability and consolidation (cargo-shift sensitivity)
# ---------------------------------------------------------------------------

def kozeny_carman_permeability(d_eff, solid_fraction) -> np.ndarray:
    """``k = d_eff^2 eps^3 / (180 (1-eps)^2)`` with ``eps = 1 - solid_fraction``."""
    d_eff = np.asarray(d_eff, dtype=np.float64)
    sf = np.asarray(solid_fraction, dtype=np.float64)
    eps = np.clip(1.0 - sf, 1e-6, 1.0 - 1e-12)
    return d_eff**2 * eps**3 / (180.0 * (1.0 - eps) ** 2)


def stratified_permeability_profile(phi_z: Sequence[float],
                                    solid_fraction_z: Sequence[float],
                                    d_large: float, d_small: float) -> dict:
    """Local ``k(z)`` of a stratified column, ``d^2``-weighted by composition."""
    phi = np.asarray(phi_z, dtype=np.float64)
    sf = np.asarray(solid_fraction_z, dtype=np.float64)
    ok = np.isfinite(phi) & np.isfinite(sf) & (sf > 0.0)
    phi_f = np.where(ok, phi, np.nan)
    sf_f = np.where(ok, sf, np.nan)
    d_eff_sq = phi_f * d_large**2 + (1.0 - phi_f) * d_small**2
    k = kozeny_carman_permeability(np.sqrt(d_eff_sq), sf_f)
    return {"d_eff_sq": d_eff_sq, "k_z": k, "solid_fraction_z": sf_f,
            "phi_z": phi_f}


def _tridiag_solve(lower: np.ndarray, main: np.ndarray, upper: np.ndarray,
                   rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm (deterministic, no LAPACK pivot choices)."""
    n = len(main)
    cp = np.zeros(max(n - 1, 0))
    dp = np.zeros(n)
    cp[0] = upper[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        denom = main[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / denom
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def consolidation_response(k_z: Sequence[float], height: float, *,
                           skeleton_modulus: float = 5.0e6,
                           gamma_w: float = 9810.0,
                           n_t: int = 400, t_end: float | None = None,
                           target_fraction: float = 0.5) -> dict:
    """1D Terzaghi consolidation with depth-varying permeability.

    ``du/dt = d/dz (c_v du/dz)``, ``c_v = k K_s / gamma_w`` (constant ``K_s``),
    uniform initial excess pressure ``u_0 = 1``, drained top, impermeable base.
    Backward Euler on a cell-centred finite-volume grid: a fixed closed algorithm,
    hence bitwise reproducible.

    Returns the time for the *base* overpressure to fall to ``target_fraction`` of
    its initial value (``t50`` for the default 0.5), in seconds.
    """
    k = np.asarray(k_z, dtype=np.float64)
    ok = np.isfinite(k) & (k > 0.0)
    if not ok.any():
        raise ValueError("at least one positive finite permeability is required")
    k = np.where(ok, k, float(np.mean(k[ok])))
    n = len(k)
    if n < 2:
        raise ValueError("at least two layers are required")
    dz = height / n
    cv = k * skeleton_modulus / gamma_w

    if t_end is None:
        t_end = 2.0 * height**2 / max(float(np.min(cv)), 1e-30)
    if not math.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive")

    dt = t_end / float(n_t)
    lam = dt / dz**2
    face = 2.0 / (1.0 / cv[:-1] + 1.0 / cv[1:])       # harmonic mean between cells
    main = np.ones(n)
    lower = np.zeros(n - 1)
    upper = np.zeros(n - 1)
    for i in range(n - 1):
        main[i] += lam * face[i]
        main[i + 1] += lam * face[i]
        upper[i] = -lam * face[i]
        lower[i] = -lam * face[i]
    main[-1] += lam * cv[-1]                          # drained top (u = 0)

    u = np.ones(n)
    times = (np.arange(1, n_t + 1) * dt).astype(np.float64)
    bottom = np.empty(n_t)
    for step in range(n_t):
        u = _tridiag_solve(lower, main, upper, u)
        bottom[step] = u[0]

    target = float(target_fraction)
    cross = np.flatnonzero(bottom <= target)
    if cross.size == 0:
        t_target = float("inf")
    else:
        i = int(cross[0])
        if i == 0:
            t_target = float(times[0])
        else:
            u0, u1 = float(bottom[i - 1]), float(bottom[i])
            frac = (target - u0) / (u1 - u0) if u1 != u0 else 1.0
            t_target = float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return {
        "t_target": t_target,
        "target_fraction": target,
        "t_end": float(t_end),
        "n_layers": int(n),
        "cv_min": float(np.min(cv)),
        "cv_max": float(np.max(cv)),
        "times": times,
        "u_bottom": bottom,
    }


def permeability_sensitivity_table(phi_z: Sequence[float],
                                   solid_fraction_z: Sequence[float],
                                   d_large: float, d_small: float,
                                   height: float, **kwargs) -> dict:
    """Drainage sensitivity to the stratification pattern.

    Compares a homogeneous column, the measured profile, its mirror image and the
    fully segregated extremes, all with the same mean composition.
    """
    phi = np.asarray(phi_z, dtype=np.float64)
    sf = np.asarray(solid_fraction_z, dtype=np.float64)
    ok = np.isfinite(phi) & np.isfinite(sf)
    if ok.sum() < 2:
        raise ValueError("need at least two populated bins")
    phi_f = np.where(ok, phi, float(np.mean(phi[ok])))
    sf_f = np.where(ok, sf, float(np.mean(sf[ok])))
    phi_mean = float(np.mean(phi_f))
    profiles = {
        "measured": phi_f,
        "mirrored": phi_f[::-1],
        "uniform": np.full_like(phi_f, phi_mean),
        "large_at_top": np.where(np.arange(len(phi_f)) >= len(phi_f) // 2, 1.0, 0.0),
        "large_at_bottom": np.where(np.arange(len(phi_f)) < len(phi_f) // 2, 1.0, 0.0),
    }
    ref = stratified_permeability_profile(profiles["uniform"], sf_f,
                                          d_large, d_small)
    t_ref = consolidation_response(ref["k_z"], height, **kwargs)["t_target"]
    rows = []
    for name, prof in profiles.items():
        layer = stratified_permeability_profile(prof, sf_f, d_large, d_small)
        k = layer["k_z"][np.isfinite(layer["k_z"])]
        resp = consolidation_response(layer["k_z"], height, **kwargs)
        finite = np.isfinite(layer["k_z"])
        kzk = layer["k_z"][finite]
        rows.append({
            "profile": name,
            "k_mean": float(np.mean(k)),
            "k_min": float(np.min(k)),
            "k_max": float(np.max(k)),
            "k_top_over_bottom": float(kzk[-1] / kzk[0]) if len(kzk) else float("nan"),
            "t50": float(resp["t_target"]),
            "t50_ratio_to_uniform": (float(resp["t_target"] / t_ref)
                                     if t_ref not in (0.0, float("inf"))
                                     and math.isfinite(t_ref)
                                     and math.isfinite(resp["t_target"])
                                     else float("nan")),
        })
    return {"rows": rows, "t50_uniform": float(t_ref),
            "height": float(height), "mean_phi": phi_mean}


# ---------------------------------------------------------------------------
# Repose angle (same estimator as P96-N3)
# ---------------------------------------------------------------------------

def surface_cloud(positions: np.ndarray, *, n_x_bins: int = 24) -> np.ndarray:
    """Top-most particle of each x strip -- the free surface as a point cloud."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    x = pos[:, 0]
    edges = np.linspace(x.min() - 1e-12, x.max() + 1e-12, int(n_x_bins) + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_x_bins - 1)
    pts = []
    for b in range(int(n_x_bins)):
        sel = idx == b
        if not sel.any():
            continue
        i = np.flatnonzero(sel)[np.argmax(pos[sel, 1])]
        pts.append(pos[i])
    return np.asarray(pts, dtype=np.float64)


def measure_rest_angle(positions: np.ndarray, radii: np.ndarray, *,
                       sectors: int = 1, min_count: int = 2,
                       n_x_bins: int = 24) -> dict:
    """Repose angle of a 2D bed with the P96-N3 estimator.

    The surface cloud of the bed is embedded in the axisymmetric form expected by
    :func:`tensordem.repose_calibration.repose_angle` (radial coordinate = x,
    azimuth = 0, elevation = y) and handed to *that* function, so the fitting
    procedure (20-85 % flank, least-squares line, ``atan(-slope)``) is the P96-N3
    one rather than a private reimplementation.
    """
    import torch
    from .repose_calibration import repose_angle

    cloud = surface_cloud(positions, n_x_bins=n_x_bins)
    if len(cloud) < 4:
        raise ValueError("need at least four surface points")
    r_mean = float(np.mean(np.asarray(radii, dtype=np.float64)))
    span = float(cloud[:, 0].max() - cloud[:, 0].min())
    # radial bin width: P96-N3 uses the particle radius because its heaps carry
    # thousands of surface grains; a 2D surface cloud is orders of magnitude
    # sparser, so the same estimator is run with the width that keeps >=12
    # annuli populated (never smaller than one particle radius).
    width = max(r_mean, span / 12.0)
    x = cloud[:, 0] - cloud[:, 0].mean()
    tensor = torch.tensor(
        np.stack([x, np.zeros_like(x), cloud[:, 1]], axis=1), dtype=torch.float64)
    out = repose_angle(tensor, width, sectors=int(sectors),
                       min_count=int(min_count))
    out = {k: (float(v) if isinstance(v, (int, float)) else v)
           for k, v in out.items()}
    out["estimator"] = "tensordem.repose_calibration.repose_angle (P96-N3)"
    out["n_surface_points"] = int(len(cloud))
    out["bin_width"] = float(width)
    out["surface_cloud"] = cloud.tolist()
    return out


def collapse_and_measure_repose(config: BidisperseConfig, *,
                                phi_of_z=None, aspect: float = 4.0,
                                n_particles: int | None = None,
                                settle_steps: int = 1200,
                                collapse_steps: int = 24000,
                                seed: int | None = None) -> dict:
    """Column-collapse repose angle, measured with the P96-N3 estimator.

    The P96-N3 workflow (:func:`tensordem.repose_calibration.simulate`) is a
    column collapse: a packed column is released and the free surface of the
    resulting heap is fitted.  This function reproduces that protocol in 2D for a
    *stratified* composition:

    * a column of aspect ratio ``aspect`` is packed at the nominal solid fraction;
    * the species are assigned from the depth profile ``phi_of_z(u)`` measured on
      the segregated bed (``u = 0`` bottom, ``1`` top), so the column carries the
      stratification of the segregated state;
    * the lateral wall is removed and the heap forms under gravity;
    * the surface cloud is handed to ``repose_calibration.repose_angle``.

    ``phi_of_z=None`` gives the monodisperse reference (the P96-N3 case).
    """
    rng = np.random.default_rng(config.seed if seed is None else int(seed))
    n = int(n_particles or config.n_particles)
    phi_nom = config.volume_fraction
    n_large = config.n_large if n_particles is None else max(
        1, int(round(config.phi_large * n)))
    area = n * config.mean_particle_area / phi_nom
    W = math.sqrt(area / aspect)
    H = aspect * W

    spacing = math.sqrt(W * H / n)
    nx = max(1, int(math.ceil(W / spacing)))
    ny = max(1, int(math.ceil(H / spacing)))
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append(((i + 0.5) * W / nx + rng.uniform(-0.08, 0.08) * W / nx,
                        (j + 0.5) * H / ny + rng.uniform(-0.08, 0.08) * H / ny))
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[rng.permutation(len(pts))[:n]]

    # species by depth, following the measured profile when given
    if phi_of_z is None:
        is_large = np.zeros(n, dtype=bool)
        is_large[rng.permutation(n)[:n_large]] = True
    else:
        u = pts[:, 1] / H
        probs = np.array([float(phi_of_z(x)) for x in u])
        is_large = rng.random(n) < probs
    radii = np.where(is_large, config.r_large, config.r_small)

    floor_w = 6.0 * W
    x = pts[:, 0] + 0.5 * (floor_w - W)
    walls = box_walls(floor_w, H, left=True, right=False, bottom=True, top=True)
    system = DiskSystem(np.column_stack([x, pts[:, 1]]), 0.85 * radii,
                        config.disk_config(damping_time=5.0e-3), walls=walls)
    for scale in (0.92, 1.0):
        _write_radii(system, scale * radii)
        system.step(max(1, settle_steps // 4))
    system.step(max(1, settle_steps // 2))
    height_confined = float(np.max(system.pos[:, 1] + system.radii))

    system.walls = box_walls(floor_w, H, left=False, right=False,
                             bottom=True, top=False)
    system.config = config.disk_config(damping_time=2.0e-3)
    system.step(int(collapse_steps))
    out = measure_rest_angle(system.pos, system.radii)
    pts_cloud = np.asarray(out.pop("surface_cloud"), dtype=np.float64)
    out.update({
        "column_width": float(W),
        "column_height": float(H),
        "aspect_ratio": float(aspect),
        "confined_height": height_confined,
        "heap_height": float(np.max(system.pos[:, 1] + system.radii)),
        "heap_width": float(np.ptp(system.pos[:, 0])),
        "n_particles": int(n),
        "collapse_steps": int(collapse_steps),
        "mean_solid_fraction": float(np.sum(math.pi * system.radii**2)
                                     / (floor_w * height_confined)),
        "surface_x": pts_cloud[:, 0].tolist(),
        "surface_y": pts_cloud[:, 1].tolist(),
    })
    return out


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------

def run_monodisperse_negative_control(*, n_particles: int = 40,
                                      seeds: Sequence[int] = (1, 2),
                                      n_bins: int = 10,
                                      shear_rate: float = 10.0,
                                      target_strain: float = 0.4,
                                      settle_steps: int = 100) -> dict:
    """Monodisperse negative control (criterion C1).

    Three independent checks:

    * ``frozen``: with ``max_steps = 0`` the pipeline runs no dynamics, so the
      reported ``Phi(z)`` must equal the initial profile *bitwise*
      (``max|dPhi| == 0.0``).
    * ``determinism``: two runs with the same seed must agree bitwise (identical
      digests and identical ``Phi``); this is what makes every other number
      reproducible.
    * ``sheared``: a sheared equal-size bed may only show finite-size noise.  Its
      ``S`` is compared with the analytic noise floor
      :func:`noise_floor` and its direction must not be systematically resolved.
    """
    base = dict(d_small=0.010, d_large=0.010, n_particles=n_particles,
                volume_fraction=0.60)

    def _case(seed: int, max_steps: int, strain: float) -> ShearBoxCase:
        cfg = BidisperseConfig(seed=seed, **base)
        return ShearBoxCase(config=cfg, shear_rate=shear_rate,
                            target_strain=strain, n_bins=n_bins,
                            sample_every=2000, max_steps=max_steps,
                            compact_steps=400)

    frozen = run_shear_box(_case(seeds[0], 0, target_strain),
                           settle_steps=settle_steps)
    frozen_dev = float(np.max(np.abs(
        np.nan_to_num(frozen["phi_z"], nan=0.0)
        - np.nan_to_num(frozen["initial_phi_z"], nan=0.0))))

    first = run_shear_box(_case(seeds[0], 10**9, target_strain),
                          settle_steps=settle_steps)
    second = run_shear_box(_case(seeds[0], 10**9, target_strain),
                           settle_steps=settle_steps)
    det_dev = float(np.max(np.abs(
        np.nan_to_num(first["phi_z"], nan=0.0)
        - np.nan_to_num(second["phi_z"], nan=0.0))))

    sheared = [run_shear_box(_case(s, 10**9, target_strain),
                             settle_steps=settle_steps) for s in seeds]
    S_values = [float(r["segregation_degree"]) for r in sheared]
    floor = float(noise_floor(n_particles, 0.5, n_bins))
    return {
        "frozen_max_deviation": frozen_dev,
        "frozen_pass": bool(frozen_dev == 0.0),
        "determinism_max_deviation": det_dev,
        "determinism_pass": bool(det_dev == 0.0),
        "sheared_S": S_values,
        "sheared_directions": [int(r["direction"]) for r in sheared],
        "noise_floor": floor,
        "sheared_max_S_over_floor": float(max(S_values) / floor) if floor > 0 else None,
        "sheared_pass": bool(all(S <= 2.5 * floor for S in S_values)),
        "pass": bool(frozen_dev == 0.0 and det_dev == 0.0
                     and all(S <= 2.5 * floor for S in S_values)),
        "n_particles": int(n_particles),
        "n_bins": int(n_bins),
        "target_strain": float(target_strain),
        "seeds": [int(s) for s in seeds],
    }


# ---------------------------------------------------------------------------
# Small in-process sweep (the production campaign lives in scripts/)
# ---------------------------------------------------------------------------

def run_bidisperse_parameter_sweep(size_ratios: Sequence[float],
                                   volume_fractions: Sequence[float], *,
                                   shear_rates: Sequence[float] = (10.0,),
                                   seeds: Sequence[int] = (1,),
                                   d_small: float = 0.010,
                                   n_particles: int = 40,
                                   target_strain: float = 0.5,
                                   n_bins: int = 10,
                                   settle_steps: int = 100,
                                   kind: str = "shear",
                                   amplitude: float = 0.002,
                                   frequency: float = 30.0,
                                   n_cycles: int = 4) -> list[dict]:
    """Run a small grid in-process; returns one record per case."""
    records = []
    for ratio in size_ratios:
        for phi in volume_fractions:
            for seed in seeds:
                cfg = BidisperseConfig(d_small=d_small, d_large=ratio * d_small,
                                       volume_fraction=phi,
                                       n_particles=n_particles, seed=int(seed))
                if kind == "shear":
                    for rate in shear_rates:
                        rec = run_shear_box(
                            ShearBoxCase(config=cfg, shear_rate=rate,
                                         target_strain=target_strain,
                                         n_bins=n_bins),
                            settle_steps=settle_steps)
                elif kind == "vibration":
                    rec = run_vibration_box(
                        VibrationBoxCase(config=cfg, amplitude=amplitude,
                                         frequency=frequency,
                                         n_cycles=n_cycles, n_bins=n_bins),
                        settle_steps=settle_steps)
                else:
                    raise ValueError("kind must be 'shear' or 'vibration'")
                rec["record_seed"] = int(seed)
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class BidisperseSegregationReport:
    """Machine-readable bundle of the campaign results."""

    criteria: dict
    negative_control: dict
    sweep: list
    scaling: dict
    permeability: dict
    repose: dict
    meta: dict

    def to_dict(self) -> dict:
        return {
            "criteria": self.criteria,
            "negative_control": self.negative_control,
            "sweep": self.sweep,
            "scaling": self.scaling,
            "permeability": self.permeability,
            "repose": self.repose,
            "meta": self.meta,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
