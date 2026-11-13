"""Standardised sandpile repose-angle protocol: TensorDEM macro anchor.

Why this module exists
----------------------
The DEM contact laws of this repository (``contact_tangential`` Part 4 Mindlin
history model, ``rolling_resistance`` EPSD/cdt/viscous laws) are *microscopic*.
What a hold, a hopper or an ice-rubble pile exhibits is *macroscopic*: the angle
of repose, the packing fraction of the deposit and the contact fabric.  Turning
microscopic parameters into those numbers requires a protocol that is (i) fixed,
(ii) reproducible, (iii) invariant to the knobs that are *not* physics (mass
flow rate, sample size) and (iv) measured with an operator whose bias is known.

This module is that protocol.  It is deliberately self-contained and
dependency-light (numpy + scipy cKDTree for the neighbour search, torch for the
contact kernels) so the N = 1e3 -> 1e5 scale scan can be sharded over cluster
nodes.

Protocol (v1, frozen)
---------------------
1. **Geometry.** Flat frictional base plane at ``z = 0``; no lateral walls; the
   base radius is sized from the expected heap radius so the pile never touches
   an edge (edge-wall effect removed by construction *and* by measurement).
2. **Pouring.** Grains are released from a **fixed drop height** above the
   current apex inside a nozzle disc of fixed radius, at a **constant mass flow
   rate** ``pour_rate`` [grains/s].  Released grains start at rest, so the
   impact energy per grain is fixed.  Spawn positions that would overlap an
   existing grain are rejected and the rejection is counted, not hidden.
3. **Settling.** After the last grain the pile relaxes for ``settle_time``.
4. **Measurement.**
   * free-surface height field ``h(r)``: annulus width ``2 r``, azimuthal
     sectors, sector *maximum* elevation (hostile to single-grain outliers),
     annulus mean over occupied sectors;
   * repose angle: least-squares **linear fit of h(r)** over the flank between
     25 % and 85 % of the toe radius, the outer ``edge_exclusion`` band being
     excluded (edge-wall effect) -> ``theta = atan(-slope)``;
   * packing fraction: probe-volume method (grains inside an interior probe
     cylinder), validated against analytic hcp/cubic packings in
     ``tests/test_dem_macro_anchor_closure.py``;
   * contact fabric anisotropy and coordination number from the contact list.
5. **Energy ledger.** Kinetic + potential + stored elastic energy and the
   accumulated dissipative work are accounted every step so a run can be
   audited; :func:`pair_energy_ledger` is the rigorous invariant test.

Physics (contact laws)
----------------------
* normal: linear spring-dashpot, ``gamma_n`` from the restitution coefficient
  (``contact_tangential.restitution_to_damping``);
* tangential: ``contact_tangential.mindlin_tangential_step`` -- Mindlin
  elastic-plastic backbone, elastic unloading branch, Coulomb cap
  ``mu_s |F_n|`` and per-contact memory of the tangential increment.  The
  negative control ``tangential_model="no_memory"`` replaces the history by the
  instantaneous slip direction (no stored increment, no unloading branch, no
  re-sticking);
* rolling: ``contact_tangential.rolling_resistance_si3`` -> the single
  ``rolling_resistance`` EPSD implementation, limit ``mu_r r_eff |F_n|``;
* rotational degrees of freedom are integrated (sphere inertia ``2/5 m r^2``).

Priors and the no-calibration guard
-----------------------------------
:data:`PRIORS` holds the literature values used by the campaign
(``mu_s = 0.5``, ``mu_r = 0.05``).  :class:`SandpileConfig` is frozen and
:func:`assert_priors` runs inside ``SandpileRun.__init__``, i.e. before the
first step: a baseline run whose coefficients differ from ``PRIORS`` is
refused, and every deviation must be a whitelisted negative control tagged
``kind="negative_control"``, so no control can be mistaken for a calibration.
The run record stores the values the stepper actually used, which makes the
"passed to the stepper == parametrised" claim machine-checkable.

References
----------
Mindlin, R.D., Deresiewicz, H. (1953). Elastic spheres in contact under varying
    oblique forces. J. Appl. Mech. 20, 327-344.
Cundall, P.A., Strack, O.D.L. (1979). A discrete numerical model for granular
    assemblies. Geotechnique 29, 47-65.
Ai, J., Chen, J.-F., Rotter, J.M., Ooi, J.Y. (2011). Assessment of rolling
    resistance models in discrete element simulations. Powder Technology 206,
    269-282.
Luding, S. (2008). Cohesive, frictional powders: contact models for tension.
    Granular Matter 10, 235-246.
Zhou, Y.C., Wright, B.D., Yang, R.Y., Xu, B.H., Yu, A.B. (1999). Rolling
    friction in the dynamic simulation of sandpile formation. Physica A 269,
    536-553.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree

from ..contact_tangential import (
    MindlinTangentialConfig,
    mindlin_tangential_step,
    restitution_to_damping,
    rolling_resistance_si3,
)

DTYPE = torch.float64
GRAVITY = 9.81

#: Frozen literature priors.  Fitting them to the measured repose angle is
#: disabled by construction: guarded configurations must use exactly these.
PRIORS: dict[str, float] = {"mu_s": 0.5, "mu_r": 0.05}
PRIOR_SOURCE = "Luding (2008) / Zhou et al. (1999) priors: mu_s = 0.5, mu_r = 0.05"

#: Whitelisted negative-control mutations (one direction only, never fitted).
CONTROL_MUTATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "mu_s_zero": {"mu_s": 0.0},
    "mu_r_zero": {"mu_r": 0.0},
    "no_memory": {"tangential_model": "no_memory"},
    "instant_fill": {"pour_rate": math.inf},
    "undamped": {"restitution": 1.0, "mu_s": 0.0, "mu_r": 0.0},
}
CONTROL_KEYS = tuple(CONTROL_MUTATIONS)


class SandpileProtocolError(ValueError):
    """Raised when a configuration violates the frozen protocol/prior guard."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandpileConfig:
    """Frozen sandpile protocol configuration (SI units).

    Attributes:
        n_particles:       Number of grains poured.
        radius:            Grain radius, m.
        density:           Grain density, kg/m^3.
        mu_s:              Coulomb sliding friction coefficient (prior 0.5).
        mu_r:              Rolling-resistance coefficient (prior 0.05).
        restitution:       Normal restitution coefficient.
        k_n:               Normal contact stiffness, N/m.
        kt_ratio:          ``k_t0 = kt_ratio * k_n`` (Mindlin secant ratio
                           ``2(1-nu)/(2-nu)`` = 0.82 for nu = 0.3).
        kr_ratio:          ``k_r = kr_ratio * k_t0 * r_eff^2`` rolling spring.
        gamma_t_ratio:     Tangential dashpot ``gamma_t = ratio * gamma_n``.
        dt:                Time step, s (``None`` -> 0.1 of the contact time).
        pour_rate:         Constant mass flow rate, grains/s.
        drop_height:       Release height above the current apex, m.
        nozzle_radius:     Radius of the release disc, m.
        settle_time:       Relaxation time after the last grain, s.
        tangential_model:  ``"mindlin_history"`` (default) or ``"no_memory"``.
        backbone:          Mindlin backbone shape (``"mindlin"``/``"linear"``).
        rolling_model:     ``"epsd"`` (default), ``"cdt"`` or ``"viscous"``.
        control:           Control key; ``"baseline"`` enforces the priors.
        calibration_sweep: Explicitly flagged **calibration** surface point:
                           ``mu_s``/``mu_r`` are the swept variables and are
                           exempt from the prior guard.  Such a run is tagged
                           ``kind="calibration_sweep"`` and can never be
                           reported as a ``baseline`` validation run.
                           ``control`` must stay ``"baseline"``.
        sector_count:      Azimuthal sectors for the height field.
        bin_width_factor:  Annulus width in grain radii.
        edge_exclusion:    Outer radial fraction excluded from the fit.
        max_spawn_tries:   Overlap-rejection attempts per injected grain.
        seed:              RNG seed (deterministic protocol).
    """

    n_particles: int
    radius: float = 0.005
    density: float = 2500.0
    mu_s: float = PRIORS["mu_s"]
    mu_r: float = PRIORS["mu_r"]
    restitution: float = 0.4
    k_n: float = 6000.0
    kt_ratio: float = 0.82
    kr_ratio: float = 0.5
    gamma_t_ratio: float = 0.0
    dt: float | None = None
    pour_rate: float = 1.0e5
    drop_height: float = 0.05
    nozzle_radius: float = 0.03
    settle_time: float = 0.10
    tangential_model: str = "mindlin_history"
    backbone: str = "mindlin"
    rolling_model: str = "epsd"
    control: str = "baseline"
    calibration_sweep: bool = False
    sector_count: int = 16
    bin_width_factor: float = 2.0
    edge_exclusion: float = 0.02
    max_spawn_tries: int = 24
    seed: int = 6932

    def __post_init__(self) -> None:
        if not isinstance(self.n_particles, int) or self.n_particles < 64:
            raise SandpileProtocolError("n_particles must be an integer >= 64")
        for name in ("radius", "density", "k_n", "kt_ratio", "kr_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise SandpileProtocolError(f"{name} must be finite and positive")
        if not (0.0 < self.restitution <= 1.0):
            raise SandpileProtocolError("restitution must lie in (0, 1]")
        if self.control not in CONTROL_MUTATIONS:
            raise SandpileProtocolError(
                f"control must be one of {CONTROL_KEYS}, got {self.control!r}")
        if self.calibration_sweep and self.control != "baseline":
            raise SandpileProtocolError(
                "calibration_sweep=True sweeps mu_s/mu_r on the baseline "
                f"protocol; control={self.control!r} is a frozen negative "
                "control and cannot be combined with it")
        if self.tangential_model not in ("mindlin_history", "no_memory"):
            raise SandpileProtocolError(
                "tangential_model must be 'mindlin_history' or 'no_memory'")
        if self.backbone not in ("mindlin", "linear"):
            raise SandpileProtocolError("backbone must be 'mindlin' or 'linear'")
        if self.rolling_model not in ("epsd", "cdt", "viscous"):
            raise SandpileProtocolError(
                "rolling_model must be 'epsd', 'cdt' or 'viscous'")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise SandpileProtocolError("dt must be finite and positive")
        if not (math.isfinite(self.drop_height) and self.drop_height > 0.0):
            raise SandpileProtocolError("drop_height must be finite and positive")
        if not (math.isfinite(self.nozzle_radius) and self.nozzle_radius > 0.0):
            raise SandpileProtocolError("nozzle_radius must be finite and positive")
        if self.nozzle_radius < 2.0 * self.radius:
            raise SandpileProtocolError(
                "nozzle_radius must be at least two grain radii (no room for a grain)")
        if not (math.isfinite(self.settle_time) and self.settle_time >= 0.0):
            raise SandpileProtocolError("settle_time must be finite and >= 0")
        if math.isinf(self.pour_rate):
            if self.control != "instant_fill":
                raise SandpileProtocolError(
                    "pour_rate=inf is only legal for control='instant_fill'")
        elif not (math.isfinite(self.pour_rate) and self.pour_rate > 0.0):
            raise SandpileProtocolError("pour_rate must be finite and positive")
        if not 0.0 <= self.edge_exclusion < 0.5:
            raise SandpileProtocolError("edge_exclusion must lie in [0, 0.5)")
        if not isinstance(self.max_spawn_tries, int) or self.max_spawn_tries < 1:
            raise SandpileProtocolError("max_spawn_tries must be a positive integer")

    # -- derived quantities ----------------------------------------------
    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def k_t0(self) -> float:
        return self.kt_ratio * self.k_n

    @property
    def r_eff(self) -> float:
        """Reduced radius of an equal-sphere pair, m."""
        return 0.5 * self.radius

    @property
    def k_r(self) -> float:
        return self.kr_ratio * self.k_t0 * self.r_eff ** 2

    @property
    def gamma_n(self) -> float:
        return restitution_to_damping(self.restitution, self.k_n, 0.5 * self.mass)

    @property
    def gamma_t(self) -> float:
        return self.gamma_t_ratio * self.gamma_n

    @property
    def critical_dt(self) -> float:
        """Undamped half-period of the linear pair contact, s."""
        return math.pi * math.sqrt(0.5 * self.mass / self.k_n)

    @property
    def dt_effective(self) -> float:
        return self.critical_dt * 0.1 if self.dt is None else self.dt

    @property
    def base_radius(self) -> float:
        """Base-plate radius, sized so the heap never reaches the edge."""
        volume = self.n_particles * (4.0 / 3.0 * math.pi * self.radius ** 3) / 0.6
        heap_radius = (3.0 * volume / (math.pi * math.tan(math.radians(30.0)))) ** (1.0 / 3.0)
        return 1.6 * heap_radius + 4.0 * self.radius

    @property
    def heap_height_estimate(self) -> float:
        """Expected heap height for this grain count, m (protocol geometry)."""
        volume = self.n_particles * (4.0 / 3.0 * math.pi * self.radius ** 3) / 0.6
        heap_radius = (3.0 * volume / (math.pi * math.tan(math.radians(30.0)))) ** (1.0 / 3.0)
        return math.tan(math.radians(30.0)) * heap_radius + 2.0 * self.radius

    @property
    def release_height(self) -> float:
        """Fixed release height above the base plate, m.

        Constant for a given grain count, so the pour never feeds back on
        itself (a stream of in-flight grains cannot lift the release point).
        The height is the estimated heap height plus ``drop_height``, i.e. the
        drop height above the *final* apex is ``drop_height`` and always at
        least that above the instantaneous apex.
        """
        return self.heap_height_estimate + self.drop_height

    @property
    def pour_duration(self) -> float:
        if math.isinf(self.pour_rate):
            return 10.0 * self.dt_effective
        return self.n_particles / self.pour_rate

    @property
    def kind(self) -> str:
        if self.calibration_sweep:
            return "calibration_sweep"
        return "baseline" if self.control == "baseline" else "negative_control"

    def mindlin_config(self) -> MindlinTangentialConfig:
        return MindlinTangentialConfig(
            k_t0=self.k_t0,
            mu_s=self.mu_s,
            backbone=self.backbone,
            gamma_t=self.gamma_t,
        )

    def as_record(self) -> dict[str, Any]:
        """JSON-friendly record of the configuration actually used."""
        return {
            "n_particles": self.n_particles,
            "radius_m": self.radius,
            "density_kg_m3": self.density,
            "mu_s": self.mu_s,
            "mu_r": self.mu_r,
            "restitution": self.restitution,
            "k_n_N_m": self.k_n,
            "k_t0_N_m": self.k_t0,
            "k_r_Nm_rad": self.k_r,
            "gamma_n_Ns_m": self.gamma_n,
            "gamma_t_Ns_m": self.gamma_t,
            "dt_s": self.dt_effective,
            "critical_dt_s": self.critical_dt,
            "pour_rate_grains_s": ("inf" if math.isinf(self.pour_rate) else self.pour_rate),
            "pour_duration_s": self.pour_duration,
            "drop_height_m": self.drop_height,
            "nozzle_radius_m": self.nozzle_radius,
            "settle_time_s": self.settle_time,
            "base_radius_m": self.base_radius,
            "release_height_m": self.release_height,
            "heap_height_estimate_m": self.heap_height_estimate,
            "tangential_model": self.tangential_model,
            "backbone": self.backbone,
            "rolling_model": self.rolling_model,
            "control": self.control,
            "kind": self.kind,
            "seed": self.seed,
            "priors": dict(PRIORS),
            "priors_used": {"mu_s": self.mu_s, "mu_r": self.mu_r},
            "priors_match_file": (self.mu_s == PRIORS["mu_s"]
                                  and self.mu_r == PRIORS["mu_r"]),
            "prior_source": PRIOR_SOURCE,
            "calibration_disabled": not self.calibration_sweep,
            "calibration_sweep": bool(self.calibration_sweep),
            "prior_guard": ("bypassed:calibration_sweep"
                            if self.calibration_sweep else "enforced:priors"),
        }


def assert_priors(config: SandpileConfig) -> None:
    """Fail-closed guard against calibrating the priors to the measured angle.

    A ``baseline`` run must use exactly :data:`PRIORS`; a control run must be
    the corresponding :data:`CONTROL_MUTATIONS` applied to the priors.  Anything
    else -- in particular a hand-tuned ``mu_r`` chasing a target repose angle --
    is refused before a single step is taken.  The single exemption is
    ``calibration_sweep=True`` (the explicitly tagged calibration surface of
    :mod:`tensordem.protocols.sandpile_calibration`), which frees ``mu_s`` and
    ``mu_r`` only; the remaining guarded knobs are still checked.

    Raises:
        SandpileProtocolError: on any deviation from the whitelist.
    """
    if config.control not in CONTROL_MUTATIONS:
        raise SandpileProtocolError(f"unknown control {config.control!r}")
    expected: dict[str, Any] = dict(PRIORS)
    expected.update(CONTROL_MUTATIONS[config.control])
    actual = {
        "mu_s": config.mu_s,
        "mu_r": config.mu_r,
        "tangential_model": config.tangential_model,
        "pour_rate": config.pour_rate,
        "restitution": config.restitution,
    }
    for key, value in expected.items():
        if config.calibration_sweep and key in ("mu_s", "mu_r"):
            # The calibration surface sweeps exactly these two coefficients;
            # every other guarded knob stays frozen (see module docstring).
            continue
        got = actual[key]
        if isinstance(value, float) and isinstance(got, float):
            same = (value == got) or (math.isinf(value) and math.isinf(got))
            if not same:
                raise SandpileProtocolError(
                    f"prior guard: control={config.control!r} requires {key}={value}, "
                    f"got {got}; calibrating the priors is disabled")
        elif value != got:
            raise SandpileProtocolError(
                f"prior guard: control={config.control!r} requires {key}={value}, "
                f"got {got}")# ---------------------------------------------------------------------------
# Measurement operators
# ---------------------------------------------------------------------------

def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def free_surface_profile(
    positions: Any,
    radius: float,
    *,
    sector_count: int = 16,
    bin_width_factor: float = 2.0,
    min_sector_fill: float = 0.5,
) -> dict[str, Any]:
    """Cylindrical free-surface height field ``h(r)`` of a heap.

    The heap centre is the (x, y) centroid of the grains.  Every annulus of
    width ``bin_width_factor * radius`` is split into ``sector_count`` azimuthal
    sectors; the *maximum* elevation inside a sector is the surface there and
    the annulus height is the mean over occupied sectors.  Annuli with fewer
    than ``min_sector_fill`` occupied sectors are dropped (toe and apex are not
    a free surface).

    Returns:
        dict with ``radii``, ``heights``, ``sector_std``, ``centre_xy``,
        ``apex_z``, ``toe_radius``, ``bin_width``, ``n_grains``.
    """
    pos = _as_numpy(positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise SandpileProtocolError("positions must be an (n, 3) array")
    if len(pos) < 8:
        raise SandpileProtocolError("need at least 8 grains to measure a surface")
    centre = pos[:, :2].mean(axis=0)
    rel = pos[:, :2] - centre
    r = np.hypot(rel[:, 0], rel[:, 1])
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    z = pos[:, 2]
    width = bin_width_factor * radius
    n_bins = max(3, int(math.ceil(r.max() / width)))
    bin_idx = np.minimum((r / width).astype(np.int64), n_bins - 1)
    sector_idx = np.minimum(
        ((theta + math.pi) / (2.0 * math.pi) * sector_count).astype(np.int64),
        sector_count - 1)

    sector_max = np.full((n_bins, sector_count), np.nan)
    for s in range(sector_count):
        mask_s = sector_idx == s
        if not mask_s.any():
            continue
        zs = z[mask_s]
        bs = bin_idx[mask_s]
        for b in np.unique(bs):
            sector_max[b, s] = zs[bs == b].max()
    heights = np.full(n_bins, np.nan)
    sector_std = np.full(n_bins, np.nan)
    min_fill = max(2, int(min_sector_fill * sector_count))
    for b in range(n_bins):
        row = sector_max[b]
        filled = ~np.isnan(row)
        if filled.sum() >= min_fill:
            heights[b] = float(np.mean(row[filled]))
            sector_std[b] = float(np.std(row[filled]))
    radii = (np.arange(n_bins) + 0.5) * width
    valid = ~np.isnan(heights)
    toe = float(radii[valid].max() + width) if valid.any() else float(width)
    return {
        "radii": radii.tolist(),
        "heights": heights.tolist(),
        "sector_std": sector_std.tolist(),
        "valid": valid.tolist(),
        "centre_xy": centre.tolist(),
        "apex_z": float(z.max()),
        "base_z": float(z.min() - radius),
        "toe_radius": toe,
        "bin_width": float(width),
        "n_grains": int(len(pos)),
    }


def repose_angle_from_profile(
    profile: dict[str, Any],
    *,
    shoulder_fraction: float = 0.25,
    toe_fraction: float = 0.85,
    edge_exclusion: float = 0.02,
) -> dict[str, Any]:
    """Half-angle of the flank from a **linear fit** of the height field.

    The fit window is ``[shoulder_fraction, toe_fraction]`` of the toe radius,
    further shrunk by ``edge_exclusion`` (relative) at the outer end, so the
    base-plate edge and the toe fringe never enter the slope.

    Returns:
        dict with ``theta_deg``, ``slope``, ``r2``, ``r_lo``, ``r_hi``,
        ``n_bins``, ``theta_stderr_deg``, ``apex_z``, ``toe_radius``.
    """
    radii = np.asarray(profile["radii"], dtype=np.float64)
    heights = np.asarray(profile["heights"], dtype=np.float64)
    std = np.asarray(profile["sector_std"], dtype=np.float64)
    valid = ~np.isnan(heights)
    toe = profile["toe_radius"] * (1.0 - edge_exclusion)
    r_lo = shoulder_fraction * toe
    r_hi = toe_fraction * toe
    window = valid & (radii >= r_lo) & (radii <= r_hi)
    if window.sum() < 3:
        raise SandpileProtocolError(
            "fewer than 3 annuli inside the fit window; enlarge the pile or "
            "widen the window")
    x = radii[window]
    y = heights[window]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    theta = math.degrees(math.atan(-slope)) if slope < 0.0 else float("nan")
    n = int(window.sum())
    sigma = math.sqrt(ss_res / max(n - 2, 1))
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope_err = sigma / math.sqrt(sxx) if sxx > 0.0 else 0.0
    tan_err = slope_err / (1.0 + slope * slope)
    return {
        "theta_deg": float(theta),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "r_lo": float(r_lo),
        "r_hi": float(r_hi),
        "n_bins": n,
        "fit_sigma_m": float(sigma),
        "theta_stderr_deg": float(math.degrees(tan_err)),
        "sector_height_std_mean_m": float(np.nanmean(std[window])),
        "apex_z": float(profile["apex_z"]),
        "toe_radius": float(profile["toe_radius"]),
    }


def probe_packing_fraction(
    positions: Any,
    radius: float,
    *,
    probe_radius: float | None = None,
    probe_height: float | None = None,
) -> dict[str, float]:
    """Solid fraction of a deposit measured with an interior probe volume.

    A cylinder of radius ``probe_radius`` (default ``6 r``) and height
    ``probe_height`` (default ``min(12 r, 0.55 * deposit height)``) is placed at
    the barycentre of the grains inside the middle third of the deposit height,
    away from the base, the free surface and the toe; grains whose centres lie
    well inside the probe contribute their exact sphere volume.  The operator is
    validated against analytic hcp / fcc / simple-cubic packings in
    ``tests/test_dem_macro_anchor_closure.py``.

    Returns:
        dict with ``packing_fraction``, ``n_probe``, ``probe_volume_m3``,
        ``probe_centre``, ``grain_volume_m3``.
    """
    pos = _as_numpy(positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise SandpileProtocolError("positions must be an (n, 3) array")
    grain_volume = 4.0 / 3.0 * math.pi * radius ** 3
    z = pos[:, 2]
    z_lo, z_hi = z.min(), z.max()
    thickness = max(z_hi - z_lo, 2.0 * radius)
    probe_radius = 6.0 * radius if probe_radius is None else probe_radius
    if probe_height is None:
        probe_height = min(12.0 * radius, 0.55 * thickness + 2.0 * radius)
    band = (z > z_lo + 0.33 * (z_hi - z_lo)) & (z < z_lo + 0.67 * (z_hi - z_lo))
    if band.sum() < 8:
        raise SandpileProtocolError("too few grains in the mid-height band")
    centre_xy = pos[band, :2].mean(axis=0)
    centre_z = float(z[band].mean())
    rel = pos[:, :2] - centre_xy
    r = np.hypot(rel[:, 0], rel[:, 1])
    inner_r = probe_radius - radius
    inner_h = 0.5 * probe_height - radius
    inside = (r <= inner_r) & (np.abs(z - centre_z) <= inner_h)
    n_probe = int(inside.sum())
    volume = math.pi * inner_r ** 2 * 2.0 * inner_h
    return {
        "packing_fraction": float(n_probe * grain_volume / volume),
        "n_probe": n_probe,
        "probe_volume_m3": float(volume),
        "probe_radius_m": float(probe_radius),
        "probe_height_m": float(probe_height),
        "probe_centre": [float(centre_xy[0]), float(centre_xy[1]), centre_z],
        "grain_volume_m3": float(grain_volume),
    }


def bulk_packing_fraction(
    positions: Any,
    radius: float,
    profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Solid fraction of the whole deposit from the integrated height field.

    ``V_deposit = int_0^R 2 pi r h(r) dr`` with ``h`` the measured free-surface
    height field (axisymmetric integration of the sector-averaged annulus
    heights), and ``phi = N V_grain / V_deposit``.  This is the operator that
    works for shallow deposits, where a probe cylinder cannot fit inside the
    packing.
    """
    pos = _as_numpy(positions)
    grain_volume = 4.0 / 3.0 * math.pi * radius ** 3
    if profile is None:
        profile = free_surface_profile(pos, radius)
    radii = np.asarray(profile["radii"], dtype=np.float64)
    heights = np.asarray(profile["heights"], dtype=np.float64)
    base = float(profile["base_z"]) + radius
    valid = ~np.isnan(heights)
    if valid.sum() < 2:
        raise SandpileProtocolError("height field has too few valid annuli")
    h = np.where(valid, heights - base, 0.0).clip(min=0.0)
    width = float(profile["bin_width"])
    volume = float(np.sum(2.0 * math.pi * radii * h * width))
    return {
        "packing_fraction": float(len(pos) * grain_volume / volume) if volume > 0 else float("nan"),
        "deposit_volume_m3": volume,
        "n_grains": int(len(pos)),
        "mean_height_m": float(volume / (math.pi * float(profile["toe_radius"]) ** 2)),
    }


def voronoi_packing_fraction(
    positions: Any,
    radius: float,
    *,
    centre: Any | None = None,
    sample_radius: float | None = None,
) -> dict[str, float]:
    """Unbiased bulk solid fraction from Voronoi cell volumes.

    Grains inside ``sample_radius`` (default ``20 r``) of ``centre`` (default:
    the barycentre of the mid-height band) are tessellated; only grains well
    inside the sample sphere are counted, so every boundary cell is discarded and
    the estimator has no boundary bias -- unlike a finite probe cylinder, whose
    surface layer biases ``phi`` by ``O(r/R_probe)``.  This is the operator used
    for the packing-fraction anchor; it is validated against analytic cubic and
    hcp packings in the tests.

    Returns:
        dict with ``packing_fraction``, ``n_cells``, ``mean_cell_volume_m3``,
        ``centre``.
    """
    from scipy.spatial import Voronoi

    pos = _as_numpy(positions)
    grain_volume = 4.0 / 3.0 * math.pi * radius ** 3
    if centre is None:
        z = pos[:, 2]
        band = (z > z.min() + 0.33 * (z.max() - z.min())) & (
            z < z.min() + 0.67 * (z.max() - z.min()))
        centre = pos[band, :2].mean(axis=0) if band.sum() > 8 else pos[:, :2].mean(axis=0)
    centre = np.asarray(centre, dtype=np.float64)
    sample_radius = 20.0 * radius if sample_radius is None else sample_radius
    d = np.linalg.norm(pos - np.array([centre[0], centre[1], pos[:, 2].mean()]), axis=1)
    subset = pos[d <= sample_radius]
    if len(subset) < 12:
        raise SandpileProtocolError("too few grains for a Voronoi packing estimate")
    vor = Voronoi(subset)
    inner = np.linalg.norm(subset - subset.mean(axis=0), axis=1) <= 0.6 * sample_radius
    volumes = []
    for idx in np.nonzero(inner)[0]:
        region_index = vor.point_region[idx]
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        vertices = vor.vertices[region]
        if len(vertices) < 4:
            continue
        hull = ConvexHull(vertices)
        volumes.append(float(hull.volume))
    if len(volumes) < 8:
        raise SandpileProtocolError("too few closed Voronoi cells inside the sample")
    volumes = np.asarray(volumes)
    return {
        "packing_fraction": float(len(volumes) * grain_volume / volumes.sum()),
        "n_cells": int(len(volumes)),
        "mean_cell_volume_m3": float(volumes.mean()),
        "centre": [float(centre[0]), float(centre[1])],
        "sample_radius_m": float(sample_radius),
    }


def core_packing_fraction(
    positions: Any,
    radius: float,
    *,
    core_fraction: float = 0.25,
    profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Bulk solid fraction of the *core* of a (possibly shallow) deposit.

    A central column of radius ``core_fraction * toe_radius`` is cut through the
    deposit and ``phi = N_core V_grain / (pi rc^2 H_core)`` is evaluated with
    ``H_core`` the core thickness.  For a deposit only a few grain diameters
    thick this is the only operator that stays unbiased: the height-field
    integral loses the toe and the probe cylinder does not fit inside.
    """
    pos = _as_numpy(positions)
    grain_volume = 4.0 / 3.0 * math.pi * radius ** 3
    if profile is None:
        profile = free_surface_profile(pos, radius)
    rc = core_fraction * float(profile["toe_radius"])
    rel = pos[:, :2] - np.asarray(profile["centre_xy"], dtype=np.float64)
    r = np.hypot(rel[:, 0], rel[:, 1])
    core = r <= rc
    if int(core.sum()) < 10:
        raise SandpileProtocolError("too few grains in the core column")
    z = pos[core, 2]
    base = float(profile["base_z"]) + radius
    height = float(z.max() - base)
    volume = math.pi * rc ** 2 * max(height, 2.0 * radius)
    return {
        "packing_fraction": float(int(core.sum()) * grain_volume / volume),
        "core_radius_m": float(rc),
        "core_height_m": float(height),
        "n_core": int(core.sum()),
    }


def contact_fabric(normal: Any, *, n_particles: int) -> dict[str, Any]:
    """Coordination number and fabric anisotropy of the contact network.

    ``Phi_ij = <n_i n_j>`` over the contact normals; the anisotropy magnitude is
    ``Phi_max - Phi_min`` (deviatoric strength of the normal-direction
    distribution).
    """
    if normal is None or len(normal) == 0:
        return {"coordination_number": 0.0, "fabric_anisotropy": 0.0,
                "fabric_eigenvalues": [0.0, 0.0, 0.0], "n_contacts": 0}
    n = _as_numpy(normal)
    phi = (n[:, :, None] * n[:, None, :]).mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(phi)
    return {
        "coordination_number": float(2.0 * len(n) / max(n_particles, 1)),
        "fabric_anisotropy": float(eigenvalues[-1] - eigenvalues[0]),
        "fabric_eigenvalues": [float(v) for v in eigenvalues],
        "n_contacts": int(len(n)),
    }


# ---------------------------------------------------------------------------
# Energy-ledger invariant
# ---------------------------------------------------------------------------

def pair_energy_ledger(
    config: SandpileConfig,
    *,
    steps: int = 100_000,
    closing_speed: float = 0.05,
    tangential_drive: float = 0.0,
) -> dict[str, float]:
    """Two-grain collision energy audit over ``steps`` time steps.

    The pair starts separated and closes head-on; gravity and the base plane are
    off.  The audit follows the *discrete* scheme exactly (semi-implicit Euler),
    so every flux is evaluated with the midpoint velocity of the step:

    * kinetic energy of translation + rotation;
    * stored normal-spring energy ``1/2 k_n delta^2``;
    * stored tangential elastic energy ``|F_t|^2 / (2 k_t0)`` (exact for the
      kinematic-hardening Mindlin branch: ``F = k_t0 (delta - delta_p)``);
    * stored rolling elastic energy ``|M|^2 / (2 k_r)``;
    * dissipative work: normal dashpot, tangential plastic flow, rolling plastic
      flow.

    The closure identity is ``E_end + D_total - E_start = 0`` where
    ``D_total >= 0`` is everything dissipated and ``E`` includes the three
    elastic reservoirs.

    Returns:
        dict with ``drift_relative`` (undamped energy drift),
        ``ledger_residual`` (full closure), ``dissipated_total``, ``ke0``,
        ``ke_end``, ``elastic_end``, ``steps``, ``dt``.
    """
    dt = config.dt_effective
    m = config.mass
    r = config.radius
    gap = 2.0 * r * (1.0 + 1e-3)
    pos = torch.tensor([[0.0, 0.0, 0.0], [gap, 0.0, 0.0]], dtype=DTYPE)
    vel = torch.tensor([[closing_speed, 0.0, 0.0],
                        [-closing_speed, 0.0, 0.0]], dtype=DTYPE)
    omega = torch.zeros((2, 3), dtype=DTYPE)
    force_state = torch.zeros((1, 3), dtype=DTYPE)
    disp_state = torch.zeros((1, 3), dtype=DTYPE)
    anchor_f = torch.zeros((1, 3), dtype=DTYPE)
    anchor_d = torch.zeros((1, 3), dtype=DTYPE)
    moment = torch.zeros((1, 3), dtype=DTYPE)
    mindlin = config.mindlin_config()
    gamma_n = config.gamma_n
    work_n_dash = 0.0
    work_t = 0.0
    work_r = 0.0
    du_t = 0.0
    du_r = 0.0

    def u_n() -> float:
        dist = float(torch.linalg.vector_norm(pos[1] - pos[0]))
        delta = max(2.0 * r - dist, 0.0)
        return 0.5 * config.k_n * delta ** 2

    def u_t() -> float:
        if config.tangential_model != "mindlin_history":
            return 0.0
        return 0.5 * float(torch.sum(force_state ** 2)) / config.k_t0

    def u_r() -> float:
        return 0.5 * float(torch.sum(moment ** 2)) / config.k_r

    mu = float(config.mu_r)
    ke = lambda: (0.5 * m * float(torch.sum(vel ** 2))
                  + 0.5 * config.inertia * float(torch.sum(omega ** 2)))

    e_start = ke() + u_n() + u_t() + u_r()
    ke_start = ke()
    ut0, ur0 = u_t(), u_r()

    for _ in range(steps):
        dvec = pos[1] - pos[0]
        dist = torch.linalg.vector_norm(dvec)
        normal = dvec / dist
        overlap = 2.0 * r - dist
        f = torch.zeros((2, 3), dtype=DTYPE)
        tau = torch.zeros((2, 3), dtype=DTYPE)
        f_n = 0.0
        f_t_vec = torch.zeros(3, dtype=DTYPE)
        torque_j = torch.zeros((1, 3), dtype=DTYPE)
        active = bool(overlap > 0.0)
        if active:
            v_rel = vel[1] - vel[0]
            vn = float(torch.dot(v_rel, normal))
            f_spring = config.k_n * float(overlap)
            f_n = max(f_spring - gamma_n * vn, 0.0)
            v_t = v_rel - vn * normal
            spin_i = torch.linalg.cross(omega[0], normal * r)
            spin_j = torch.linalg.cross(omega[1], -normal * r)
            v_surf = (v_t + (spin_j - spin_i)).reshape(1, 3)
            if tangential_drive:
                v_surf = v_surf + torch.tensor(
                    [[0.0, tangential_drive, 0.0]], dtype=DTYPE)
            f_t = torch.zeros((1, 3), dtype=DTYPE)
            if config.tangential_model == "mindlin_history":
                (f_t, force_state, disp_state, anchor_f, anchor_d, _slide) = \
                    mindlin_tangential_step(
                        force_state, disp_state, anchor_f, anchor_d, v_surf,
                        torch.tensor([f_n], dtype=DTYPE), mindlin, dt)
            else:
                slip = torch.linalg.vector_norm(v_surf).clamp_min(1e-30)
                f_t = -config.mu_s * f_n * v_surf / slip
                force_state = torch.zeros((1, 3), dtype=DTYPE)
            f_t_vec = f_t[0]
            total = f_n * normal + f_t_vec
            f[1] = total
            f[0] = -total
            tau[1] = tau[1] + torch.linalg.cross(-normal * r, f_t_vec)
            tau[0] = tau[0] + torch.linalg.cross(normal * r, -f_t_vec)
            torque_j, moment, _power = rolling_resistance_si3(
                (omega[1] - omega[0]).unsqueeze(0), normal.unsqueeze(0),
                torch.tensor([f_n], dtype=DTYPE),
                torch.tensor([config.r_eff], dtype=DTYPE), mu_r=config.mu_r,
                model=config.rolling_model, k_r=config.k_r, dt=dt,
                torque_state=moment)
            tau[1] = tau[1] + torque_j[0]
            tau[0] = tau[0] - torque_j[0]
        v_old = vel.clone()
        w_old = omega.clone()
        vel = vel + f * (dt / m)
        omega = omega + tau * (dt / config.inertia)
        pos = pos + vel * dt
        v_mid = 0.5 * (v_old + vel)
        w_mid = 0.5 * (w_old + omega)
        if active:
            v_rel_mid = v_mid[1] - v_mid[0]
            vn_mid = float(torch.dot(v_rel_mid, normal))
            f_dash = f_n - config.k_n * float(overlap)
            work_n_dash += f_dash * vn_mid * dt
            v_t_mid = (v_rel_mid - vn_mid * normal) + (
                torch.linalg.cross(w_mid[1], -normal * r)
                - torch.linalg.cross(w_mid[0], normal * r))
            work_t += float(torch.dot(f_t_vec, v_t_mid)) * dt
            work_r += float(torch.dot(torque_j[0], w_mid[1] - w_mid[0])) * dt
    u_n_end, u_t_end, u_r_end = u_n(), u_t(), u_r()
    e_end = ke() + u_n_end + u_t_end + u_r_end
    # Dissipation: dashpot directly; plastic flow = applied work minus the
    # change of the elastic reservoir it feeds.
    d_t = -((work_t) - (u_t_end - ut0))
    d_r = -((work_r) - (u_r_end - ur0))
    d_n = -work_n_dash
    d_total = d_n + d_t + d_r
    return {
        "steps": steps,
        "dt": dt,
        "ke0": ke_start,
        "ke_end": ke(),
        "elastic_end": u_n_end,
        "u_t_end": u_t_end,
        "u_r_end": u_r_end,
        "work_normal_dashpot": work_n_dash,
        "work_tangential": work_t,
        "work_rolling": work_r,
        "dissipated_normal": d_n,
        "dissipated_tangential": d_t,
        "dissipated_rolling": d_r,
        "dissipated_total": d_total,
        "drift_relative": (e_end - e_start) / e_start,
        "ledger_residual": (e_end + d_total - e_start) / e_start,
        "ledger_residual_vs_dissipated": (
            (e_end + d_total - e_start) / d_total if d_total > 0.0 else 0.0),
    }


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------

class SandpileRun:
    """One sandpile pour under the frozen protocol.

    Usage::

        run = SandpileRun(SandpileConfig(n_particles=4000))
        result = run.run()

    The instance owns the grain state; create a fresh one per run.
    """

    def __init__(self, config: SandpileConfig) -> None:
        self.config = config
        assert_priors(config)  # fail-closed, before the first step
        self._rng = np.random.default_rng(config.seed)
        self.n = config.n_particles
        self.positions = torch.zeros((self.n, 3), dtype=DTYPE)
        self.velocities = torch.zeros((self.n, 3), dtype=DTYPE)
        self.omega = torch.zeros((self.n, 3), dtype=DTYPE)
        self.n_poured = 0
        self.time = 0.0
        self.steps = 0
        self.raised_spawns = 0
        self.spawn_rejections = 0
        self._slot_cursor = 0
        self._slot_cache: np.ndarray | None = None
        # Per-pair history, keyed by i * n + j (sorted).
        self._keys = np.zeros(0, dtype=np.int64)
        self._force = torch.zeros((0, 3), dtype=DTYPE)
        self._disp = torch.zeros((0, 3), dtype=DTYPE)
        self._anchor_force = torch.zeros((0, 3), dtype=DTYPE)
        self._anchor_disp = torch.zeros((0, 3), dtype=DTYPE)
        self._roll_moment = torch.zeros((0, 3), dtype=DTYPE)
        # Floor-contact history, per particle.
        self._floor_force = torch.zeros((self.n, 3), dtype=DTYPE)
        self._floor_disp = torch.zeros((self.n, 3), dtype=DTYPE)
        self._floor_anchor_f = torch.zeros((self.n, 3), dtype=DTYPE)
        self._floor_anchor_d = torch.zeros((self.n, 3), dtype=DTYPE)
        self._floor_moment = torch.zeros((self.n, 3), dtype=DTYPE)
        # Diagnostics.
        self.work_dashpot = 0.0
        self.work_tangential = 0.0
        self.work_rolling = 0.0
        self.max_speed = 0.0
        self.max_overlap = 0.0
        self.n_contacts = 0
        self._contacts = torch.zeros((0, 3), dtype=DTYPE)

    # -- pouring ---------------------------------------------------------
    def settled_apex(self, speed_tolerance: float = 0.30) -> float:
        """Highest *settled* grain, m.

        In-flight grains move at the injection speed (>= 0.05 m/s) or faster, so
        a speed filter separates the deposit from the stream.  The release plane
        is referenced to this apex, otherwise the deposit grows into the nozzle
        and the feed starts spawning grains inside the pile.
        """
        if self.n_poured == 0:
            return 0.0
        speeds = torch.linalg.vector_norm(self.velocities[:self.n_poured], dim=1)
        settled = speeds < speed_tolerance
        if not bool(settled.any()):
            return 0.0
        return float(self.positions[:self.n_poured, 2][settled].max())

    def _release_plane(self) -> float:
        cfg = self.config
        return max(cfg.release_height, self.settled_apex() + cfg.drop_height)

    def injection_velocity(self) -> float:
        """Exit speed of the feed stream, m/s.

        A nozzle that emits grains *at rest* can only deliver one grain per
        ``sqrt(2 * 2r / g)`` per slot (~20 grains/s), which is far below any
        useful pour rate.  The feed is therefore a plug-flow stream: the exit
        speed that carries the requested mass flow rate at a solid fraction of
        0.55 through the nozzle area,

            ``v0 = pour_rate * V_grain / (0.55 * pi * nozzle_radius^2)``,

        clamped to [0.05, 3] m/s.  It is recorded with the run and is the same
        for every grain count at a given rate.
        """
        cfg = self.config
        if math.isinf(cfg.pour_rate):
            return 3.0
        area = math.pi * cfg.nozzle_radius ** 2
        v0 = cfg.pour_rate * (4.0 / 3.0 * math.pi * cfg.radius ** 3) / (0.55 * area)
        return float(min(max(v0, 0.05), 3.0))

    def _inject(self, target: int, tree: cKDTree | None) -> None:
        """Release grains through the nozzle disc, up to ``target``.

        Placement is *random* inside the nozzle disc but must clear every
        existing grain by one diameter (checked against the neighbour tree and
        against the grains released in this same call); otherwise the release is
        deferred, which is how a real nozzle throttles.  A fixed slot lattice is
        deliberately avoided: it feeds each slot down one vertical column and
        builds a 1-D "spaghetti" of chains (coordination number 2) instead of a
        packing, which destroys the repose-angle measurement.
        """
        cfg = self.config
        r = cfg.radius
        clearance = 2.0 * r * 1.02
        v0 = self.injection_velocity()
        plane = self._release_plane()
        released: list[np.ndarray] = []
        while self.n_poured < target and self.n_poured < self.n:
            placed = False
            for _ in range(cfg.max_spawn_tries):
                if math.isinf(cfg.pour_rate):
                    rr = cfg.nozzle_radius * math.sqrt(float(self._rng.random()))
                else:
                    # uniform over the disc: the stream fills the nozzle
                    rr = cfg.nozzle_radius * math.sqrt(float(self._rng.random()))
                th = 2.0 * math.pi * float(self._rng.random())
                cand = np.array([rr * math.cos(th), rr * math.sin(th), plane])
                free = True
                if tree is not None and tree.n > 0:
                    dist, _ = tree.query(cand, k=1)
                    free = float(dist) >= clearance
                if free and released:
                    free = min(float(np.linalg.norm(cand - q)) for q in released) >= clearance
                if not free:
                    self.spawn_rejections += 1
                    continue
                self.positions[self.n_poured] = torch.from_numpy(cand)
                self.velocities[self.n_poured] = torch.tensor(
                    [0.0, 0.0, -v0], dtype=DTYPE)
                self.omega[self.n_poured] = 0.0
                released.append(cand)
                self.n_poured += 1
                placed = True
                break
            if not placed:
                return

    # -- pair history ----------------------------------------------------
    def _lookup(self, keys: np.ndarray) -> tuple[torch.Tensor, ...]:
        m = keys.size
        zeros = torch.zeros((m, 3), dtype=DTYPE)
        if self._keys.size == 0:
            return zeros, zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
        idx = np.searchsorted(self._keys, keys)
        idx_c = np.clip(idx, 0, self._keys.size - 1)
        hit = self._keys[idx_c] == keys
        if not hit.any():
            return zeros, zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
        rows = torch.from_numpy(np.nonzero(hit)[0])
        src = torch.from_numpy(idx_c[hit])
        out = []
        for store in (self._force, self._disp, self._anchor_force,
                      self._anchor_disp, self._roll_moment):
            dst = torch.zeros((m, 3), dtype=DTYPE)
            dst[rows] = store[src]
            out.append(dst)
        return tuple(out)

    # -- one time step ---------------------------------------------------
    def _step(self, dt: float) -> None:
        cfg = self.config
        n = self.n_poured
        if n == 0:
            return
        r = cfg.radius
        pos = self.positions[:n]
        vel = self.velocities[:n]
        omega = self.omega[:n]
        force = torch.zeros((n, 3), dtype=DTYPE)
        torque = torch.zeros((n, 3), dtype=DTYPE)
        tree = cKDTree(pos.numpy())
        pair_list = tree.query_pairs(2.0 * r * (1.0 + 1e-9), output_type="ndarray")
        self._tree = tree
        self._contacts = torch.zeros((0, 3), dtype=DTYPE)
        self.n_contacts = 0

        if len(pair_list):
            iu = np.minimum(pair_list[:, 0], pair_list[:, 1]).astype(np.int64)
            ju = np.maximum(pair_list[:, 0], pair_list[:, 1]).astype(np.int64)
            keys = iu * max(n, 1) + ju
            order = np.argsort(keys)
            iu, ju, keys = iu[order], ju[order], keys[order]
            ti = torch.from_numpy(iu)
            tj = torch.from_numpy(ju)
            (f_state, d_state, af_state, ad_state, moment_state) = self._lookup(keys)

            dvec = pos[tj] - pos[ti]
            dist = torch.linalg.vector_norm(dvec, dim=1)
            normal = dvec / dist.clamp_min(torch.finfo(DTYPE).tiny)[:, None]
            overlap = 2.0 * r - dist
            active = overlap > 0.0
            keep = active.to(DTYPE)
            v_rel = vel[tj] - vel[ti]
            vn = (v_rel * normal).sum(dim=1)
            f_n = (cfg.k_n * overlap - cfg.gamma_n * vn).clamp_min(0.0) * keep
            self.work_dashpot += cfg.gamma_n * float((vn * vn * keep).sum()) * dt

            v_t = v_rel - vn[:, None] * normal
            spin_i = torch.linalg.cross(omega[ti], normal * r)
            spin_j = torch.linalg.cross(omega[tj], -normal * r)
            v_surf = v_t + (spin_j - spin_i)

            if cfg.tangential_model == "mindlin_history":
                (f_t, f_state, d_state, af_state, ad_state, _sliding) = \
                    mindlin_tangential_step(
                        f_state, d_state, af_state, ad_state, v_surf, f_n,
                        cfg.mindlin_config(), dt)
            else:
                # Memoryless control: the tangential force is re-derived from the
                # instantaneous slip direction only (no stored increment, no
                # unloading branch, no re-sticking history).
                slip_mag = torch.linalg.vector_norm(v_surf, dim=1)
                in_slip = slip_mag > 1e-14
                direction = v_surf / slip_mag.clamp_min(1e-30)[:, None]
                f_t = -cfg.mu_s * f_n[:, None] * direction * in_slip[:, None]
                zero = torch.zeros_like(f_t)
                f_state, d_state, af_state, ad_state = zero, zero, zero.clone(), \
                    zero.clone()
            self.work_tangential += float((f_t * v_surf).sum()) * dt

            pair_force = f_n[:, None] * normal + f_t
            force.index_add_(0, ti, -pair_force)
            force.index_add_(0, tj, pair_force)
            contact_torque = torch.linalg.cross(-normal * r, f_t)
            torque.index_add_(0, ti, contact_torque)
            torque.index_add_(0, tj, contact_torque)

            torque_j, moment_state, power = rolling_resistance_si3(
                omega[tj] - omega[ti], normal, f_n,
                torch.full_like(f_n, cfg.r_eff), mu_r=cfg.mu_r,
                model=cfg.rolling_model, k_r=cfg.k_r, dt=dt,
                torque_state=moment_state)
            self.work_rolling += -float(power.sum()) * dt
            torque.index_add_(0, tj, torque_j)
            torque.index_add_(0, ti, -torque_j)

            self.max_overlap = max(self.max_overlap, float(overlap.max()))
            self.n_contacts = int(active.sum())
            self._contacts = normal[active]
            self._keys = keys
            self._force, self._disp = f_state, d_state
            self._anchor_force, self._anchor_disp = af_state, ad_state
            self._roll_moment = moment_state
        else:
            self._keys = np.zeros(0, dtype=np.int64)
            self._force = torch.zeros((0, 3), dtype=DTYPE)
            self._disp = torch.zeros((0, 3), dtype=DTYPE)
            self._anchor_force = torch.zeros((0, 3), dtype=DTYPE)
            self._anchor_disp = torch.zeros((0, 3), dtype=DTYPE)
            self._roll_moment = torch.zeros((0, 3), dtype=DTYPE)

        # ---- frictional base plane at z = 0 --------------------------
        pen = (r - pos[:, 2]).clamp_min(0.0)
        touching = pen > 0.0
        if bool(touching.any()):
            idx = torch.nonzero(touching, as_tuple=True)[0]
            vz = vel[idx, 2]
            f_n = (cfg.k_n * pen[idx] - cfg.gamma_n * vz).clamp_min(0.0)
            plane_n = torch.zeros((len(idx), 3), dtype=DTYPE)
            plane_n[:, 2] = 1.0
            v_surf = vel[idx] - vz[:, None] * plane_n
            v_surf = v_surf + torch.linalg.cross(omega[idx], -plane_n * r)
            if cfg.tangential_model == "mindlin_history":
                (f_t, f_state, d_state, af_state, ad_state, _sl) = \
                    mindlin_tangential_step(
                        self._floor_force[idx], self._floor_disp[idx],
                        self._floor_anchor_f[idx], self._floor_anchor_d[idx],
                        v_surf, f_n, cfg.mindlin_config(), dt)
            else:
                slip_mag = torch.linalg.vector_norm(v_surf, dim=1)
                in_slip = slip_mag > 1e-14
                direction = v_surf / slip_mag.clamp_min(1e-30)[:, None]
                f_t = -cfg.mu_s * f_n[:, None] * direction * in_slip[:, None]
                zero = torch.zeros_like(f_t)
                f_state, d_state, af_state, ad_state = zero, zero, zero.clone(), \
                    zero.clone()
            self._floor_force[idx] = f_state
            self._floor_disp[idx] = d_state
            self._floor_anchor_f[idx] = af_state
            self._floor_anchor_d[idx] = ad_state
            self.work_tangential += float((f_t * v_surf).sum()) * dt
            self.work_dashpot += cfg.gamma_n * float((vz * vz).sum()) * dt
            force[idx] += f_n[:, None] * plane_n + f_t
            torque.index_add_(0, idx, torch.linalg.cross(-plane_n * r, f_t))
            torque_j, moment_next, power = rolling_resistance_si3(
                omega[idx], plane_n, f_n, torch.full_like(f_n, r),
                mu_r=cfg.mu_r, model=cfg.rolling_model, k_r=cfg.k_r, dt=dt,
                torque_state=self._floor_moment[idx])
            self._floor_moment[idx] = moment_next
            self.work_rolling += -float(power.sum()) * dt
            torque.index_add_(0, idx, torque_j)
        self._floor_force[:n][~touching] = 0.0
        self._floor_disp[:n][~touching] = 0.0
        self._floor_anchor_f[:n][~touching] = 0.0
        self._floor_anchor_d[:n][~touching] = 0.0
        self._floor_moment[:n][~touching] = 0.0

        # ---- gravity + semi-implicit Euler --------------------------
        force[:, 2] -= cfg.mass * GRAVITY
        vel += force * (dt / cfg.mass)
        omega += torque * (dt / cfg.inertia)
        pos += vel * dt
        self.steps += 1
        self.time += dt
        self.max_speed = max(self.max_speed,
                             float(torch.linalg.vector_norm(vel, dim=1).max()))

    # -- driver ----------------------------------------------------------
    def run(self, *, progress: bool = False) -> dict[str, Any]:
        """Pour, settle and measure.  Returns the run record + anchors."""
        cfg = self.config
        dt = cfg.dt_effective
        n_pour = max(1, int(math.ceil(cfg.pour_duration / dt)))
        n_settle = int(math.ceil(cfg.settle_time / dt))
        injected = 0.0
        total_steps = n_pour + n_settle
        step = 0
        while step < total_steps or self.n_poured < self.n:
            if step < n_pour or self.n_poured < self.n:
                if math.isinf(cfg.pour_rate):
                    target = self.n
                else:
                    injected += cfg.pour_rate * dt
                    target = int(min(self.n, injected))
                # A long deferral must not discharge as a burst: cap the backlog
                # at one nozzle layer and let the deficit show as a rate loss.
                target = min(target, self.n_poured + 64)
                if target > self.n_poured:
                    self._inject(target, getattr(self, "_tree", None))
            self._step(dt)
            step += 1
            if step > total_steps + 20 * n_settle + 5000:
                raise SandpileProtocolError(
                    "the hopper did not empty; check pour_rate/nozzle geometry")
            if progress and step % 500 == 0:
                print(f"  step {step}/{n_pour + n_settle} poured={self.n_poured}",
                      flush=True)
        return self.measure()

    # -- measurement -----------------------------------------------------
    def measure(self) -> dict[str, Any]:
        cfg = self.config
        pos = self.positions[:self.n_poured]
        profile = free_surface_profile(
            pos, cfg.radius, sector_count=cfg.sector_count,
            bin_width_factor=cfg.bin_width_factor)
        angle = repose_angle_from_profile(
            profile, edge_exclusion=cfg.edge_exclusion)
        def _safe(fn):
            """Measure, or report NaN + reason: a degenerate deposit is a result.

            A negative control that removes the rolling resistance spreads the
            grains into a flat splat with no mid-height band.  The *angle* of
            that surface is exactly the quantity under study, so the packing
            probes must not be able to abort the measurement (they did before
            m17676: ``probe_packing_fraction`` raised and the run record -- and
            with it the repose angle -- was lost).
            """
            try:
                return fn()
            except SandpileProtocolError as exc:  # degenerate/flat deposits
                return {"packing_fraction": float("nan"), "error": str(exc)}

        packing = _safe(lambda: probe_packing_fraction(pos, cfg.radius))
        packing["bulk"] = _safe(lambda: bulk_packing_fraction(pos, cfg.radius, profile))
        packing["core"] = _safe(lambda: core_packing_fraction(
            pos, cfg.radius, profile=profile))
        packing["voronoi"] = _safe(lambda: voronoi_packing_fraction(pos, cfg.radius))
        fabric = contact_fabric(self._contacts, n_particles=self.n_poured)
        ke = 0.5 * cfg.mass * float(torch.sum(
            self.velocities[:self.n_poured] ** 2)) + 0.5 * cfg.inertia * float(
            torch.sum(self.omega[:self.n_poured] ** 2))
        pe = cfg.mass * GRAVITY * float(torch.sum(pos[:, 2]))
        dissipated = self.work_dashpot + self.work_tangential + self.work_rolling
        record = cfg.as_record()
        record.update({
            "protocol": "sandpile-v1",
            "steps": self.steps,
            "simulated_time_s": self.time,
            "poured": self.n_poured,
            "spawn_rejections": self.spawn_rejections,
            "injection_velocity_m_s": self.injection_velocity(),
            "raised_spawns": self.raised_spawns,
            "max_speed_m_s": self.max_speed,
            "max_overlap_ratio": (self.max_overlap / cfg.radius),
            "repose_deg": angle["theta_deg"],
            "repose_fit": angle,
            "packing": packing,
            "fabric": fabric,
            "energy": {
                "kinetic_J": ke,
                "potential_J": pe,
                "dissipated_J": dissipated,
                "work_dashpot_J": self.work_dashpot,
                "work_tangential_J": self.work_tangential,
                "work_rolling_J": self.work_rolling,
            },
            "surface_profile": profile,
        })
        return record