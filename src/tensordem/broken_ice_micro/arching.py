"""Janssen silo mechanics and the arching / clogging criterion for broken ice.

Two questions are answered here, and they share one analytic anchor.

1. **Does a confined broken-ice column carry its own weight the way a granular
   silo does?**  For a channel of width ``W`` filled with a medium of bulk
   density ``rho_b``, wall friction ``mu_w`` and lateral-stress ratio
   ``K = sigma_xx / sigma_zz``, the vertical stress obeys

   .. math::  \\frac{d\\sigma_{zz}}{dz} + \\rho_b g - \\frac{2 K \\mu_w}{W}\\sigma_{zz} = 0 ,

   with the closed-form solution (Janssen 1895)

   .. math::  \\sigma_{zz}(z) = \\rho_b g \\lambda\\,(1 - e^{-z/\\lambda}),
              \\qquad \\lambda = \\frac{W}{2 K \\mu_w},
              \\qquad \\sigma_{zz}(\\infty) = \\rho_b g \\lambda .

   :func:`janssen_profile` is that solution and :func:`fit_janssen` recovers
   ``(K, mu_w)`` from a DEM wall-pressure profile together with the relative
   error -- the campaign's acceptance criterion (c) is ``max error < 15 %``.

2. **At which width-to-floe ratio does a broken-ice channel arch off (clog)?**
   :func:`run_silo` discharges a gravity-driven polydisperse disk bed through a
   channel of width ``W`` on the :mod:`tensordem.granular2d` engine and reports
   whether an arch arrested the flow.  :func:`clogging_curve` turns a sweep of
   ``W/h`` into ``P_clog(W/h)`` and :func:`critical_width_ratio` interpolates the
   50 % crossing ``(W/h)_crit``.

The Janssen saturation length ``lambda`` is the length scale that makes the
arching question well posed: an arch can only be load-bearing if it sits inside
the region where stress is still building up (``z ~ lambda``); the clogging
probability therefore depends on ``W/h`` rather than on the column height.  That
statement is checked, not assumed, in the study driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math

import numpy as np

from ..granular2d import DiskConfig, DiskSystem, box_walls
from .chain import contact_network_stats

__all__ = [
    "JanssenProfile",
    "JanssenFit",
    "SiloConfig",
    "SiloRun",
    "ColumnRun",
    "janssen_profile",
    "fit_janssen",
    "run_column",
    "run_silo",
    "clogging_curve",
    "critical_width_ratio",
]


# --------------------------------------------------------------------- Janssen
@dataclass(frozen=True)
class JanssenProfile:
    """Closed-form 2D Janssen solution for a channel of width ``width``."""

    width: float
    rho_bulk: float
    k_ratio: float
    wall_friction: float
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if not self.width > 0.0:
            raise ValueError("width must be positive")
        if not (self.rho_bulk > 0.0 and self.k_ratio > 0.0 and self.wall_friction > 0.0):
            raise ValueError("rho_bulk, k_ratio and wall_friction must be positive")

    @property
    def decay_length(self) -> float:
        """Janssen saturation length ``lambda = W / (2 K mu_w)`` (m)."""
        return self.width / (2.0 * self.k_ratio * self.wall_friction)

    @property
    def plateau_stress(self) -> float:
        """Saturated vertical stress ``rho_b g lambda`` (Pa)."""
        return self.rho_bulk * self.gravity * self.decay_length

    def stress_zz(self, z):
        z = np.asarray(z, dtype=np.float64)
        return self.plateau_stress * (1.0 - np.exp(-z / self.decay_length))

    def wall_stress(self, z):
        """Normal stress on the side wall, ``sigma_w = K sigma_zz`` (Pa)."""
        return self.k_ratio * self.stress_zz(z)

    @property
    def wall_unit_weight(self) -> float:
        """Weight of the wall-supported slice, ``rho_b g W`` (Pa)."""
        return self.rho_bulk * self.gravity * self.width


def janssen_profile(
    *,
    width: float,
    rho_bulk: float,
    k_ratio: float,
    wall_friction: float,
    gravity: float = 9.81,
) -> JanssenProfile:
    return JanssenProfile(
        width=float(width),
        rho_bulk=float(rho_bulk),
        k_ratio=float(k_ratio),
        wall_friction=float(wall_friction),
        gravity=float(gravity),
    )


@dataclass
class JanssenFit:
    """Fit of the Janssen profile to a measured wall-pressure profile."""

    decay_length: float
    k_ratio: float
    wall_friction: float
    plateau_stress: float
    rms_relative_error: float
    max_relative_error: float
    r2: float
    wall_unit_weight: float
    depths: np.ndarray
    measured: np.ndarray
    model: np.ndarray

    @property
    def passes(self) -> bool:
        """Campaign criterion (c): Janssen fit error < 15 %."""
        return self.max_relative_error < 0.15

    def as_dict(self) -> dict:
        return dict(
            decay_length_m=float(self.decay_length),
            k_ratio=float(self.k_ratio),
            wall_friction=float(self.wall_friction),
            plateau_stress_Pa=float(self.plateau_stress),
            rms_relative_error=float(self.rms_relative_error),
            max_relative_error=float(self.max_relative_error),
            r2=float(self.r2),
            wall_unit_weight_Pa=float(self.wall_unit_weight),
            passes=bool(self.passes),
            error_threshold=0.15,
            depths_m=[float(v) for v in self.depths],
            measured_Pa=[float(v) for v in self.measured],
            model_Pa=[float(v) for v in self.model],
        )


def _janssen_shape(z: np.ndarray, lam: float) -> np.ndarray:
    return 1.0 - np.exp(-z / lam)


def _janssen_cumulative_shape(z: np.ndarray, lam: float) -> np.ndarray:
    """Integral of ``1 - exp(-z/lam)`` over depth, i.e. ``z - lam(1-exp(-z/lam))``."""
    return z - lam * (1.0 - np.exp(-z / lam))


def fit_janssen(
    depths,
    wall_stress,
    *,
    width: float,
    rho_bulk: float,
    gravity: float = 9.81,
    min_points: int = 4,
    cumulative: bool = False,
) -> JanssenFit:
    """Least-squares fit of ``(lambda, K)`` to a measured wall-stress profile.

    The amplitude of the measured stress is allowed to be free (calibration of
    the effective wall contact area is a modelling unknown), so the model is

        ``sigma_w(z) = A (1 - exp(-z/lambda))``,  ``A = K rho_b g lambda``,

    solved on a log-spaced grid of ``lambda`` with ``A`` eliminated analytically
    by linear least squares.  The reported ``K`` and ``mu_w`` follow from
    ``lambda`` and ``A``.
    """
    z = np.asarray(depths, dtype=np.float64).reshape(-1)
    s = np.asarray(wall_stress, dtype=np.float64).reshape(-1)
    if z.shape != s.shape:
        raise ValueError("depths and wall_stress must have the same length")
    keep = np.isfinite(z) & np.isfinite(s) & (z > 0.0)
    z, s = z[keep], s[keep]
    if z.size < min_points:
        raise ValueError(f"need at least {min_points} valid samples, got {z.size}")
    if not (width > 0.0 and rho_bulk > 0.0 and gravity > 0.0):
        raise ValueError("width, rho_bulk and gravity must be positive")

    span = float(z.max())
    lambdas = np.exp(np.linspace(math.log(0.02 * span), math.log(20.0 * span), 600))
    best = (math.inf, lambdas[0], 0.0)
    for lam in lambdas:
        basis = _janssen_cumulative_shape(z, lam) if cumulative else _janssen_shape(z, lam)
        denom = float(basis @ basis)
        if denom <= 0.0:
            continue
        amp = float(basis @ s) / denom
        resid = s - amp * basis
        sse = float(resid @ resid)
        if sse < best[0]:
            best = (sse, float(lam), amp)
    sse, lam, amp = best
    model = amp * (_janssen_cumulative_shape(z, lam) if cumulative else _janssen_shape(z, lam))
    scale = max(float(np.abs(s).max()), 1e-12)
    rel = np.abs(model - s) / np.maximum(np.abs(s), 0.05 * scale)
    ss_tot = float(((s - s.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0.0 else 0.0
    k_ratio = amp / (rho_bulk * gravity * lam)
    mu_w = width / (2.0 * k_ratio * lam)
    return JanssenFit(
        decay_length=lam,
        k_ratio=float(k_ratio),
        wall_friction=float(mu_w),
        plateau_stress=float(amp),
        rms_relative_error=float(np.sqrt(np.mean(rel**2))),
        max_relative_error=float(rel.max()),
        r2=float(r2),
        wall_unit_weight=rho_bulk * gravity * width,
        depths=z,
        measured=s,
        model=model,
    )


# --------------------------------------------------------------------- DEM runs
@dataclass(frozen=True)
class SiloConfig:
    """Configuration of the channel-discharge (arching) experiment."""

    width: float
    floe_diameter: float = 0.10
    size_cv: float = 0.30
    height: float = 1.20
    fill_fraction: float = 0.50
    density: float = 917.0
    thickness: float = 0.05
    mu: float = 0.5
    mu_rolling: float = 0.05
    wall_friction: float = 0.5
    zeta: float = 0.2
    gravity: float = 9.81
    seed: int = 0
    settle_steps: int = 1500
    max_time: float = 3.0
    stop_time: float = 0.35
    block_steps: int = 100
    bin_height: float = 0.10
    max_attempts: int = 4000
    bottom_open: bool = True
    body_factor: float = 2.0

    def __post_init__(self) -> None:
        for name in ("width", "floe_diameter", "height", "density", "thickness"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.fill_fraction < 0.95:
            raise ValueError("fill_fraction must lie in (0, 0.95)")
        if not self.size_cv >= 0.0:
            raise ValueError("size_cv must be non-negative")
        if not 0.0 <= float(self.wall_friction) <= 1.0:
            raise ValueError("wall_friction must lie in [0, 1]")

    @property
    def body_width(self) -> float:
        """Width of the silo body that feeds the outlet (m)."""
        return max(float(self.body_factor), 1.0) * float(self.width)

    @property
    def width_ratio(self) -> float:
        """Channel width over floe diameter, ``W/h``."""
        return self.width / self.floe_diameter

    @property
    def disk_config(self) -> DiskConfig:
        return DiskConfig(
            mu=self.mu,
            mu_rolling=self.mu_rolling,
            zeta=self.zeta,
            density=self.density,
            thickness=self.thickness,
            gravity=(0.0, -abs(self.gravity)),
        )


def _polydisperse_bed(cfg: SiloConfig) -> tuple[np.ndarray, np.ndarray]:
    """Jittered hexagonal bed of log-normal disks filling the lower channel.

    A lattice seed (rather than random sequential adsorption) is used on
    purpose: RSA jams at an area fraction of ~0.5 for the size spread used here
    and leaves the column height uncontrolled, while the DEM contact law relaxes
    a jittered lattice into a well-defined jammed bed with the requested area
    fraction.  Radii are capped at ``0.49`` lattice spacings so the seed itself
    is overlap-free.
    """
    rng = np.random.default_rng(cfg.seed)
    mean_r = 0.5 * cfg.floe_diameter
    sigma = cfg.size_cv
    # dense hexagonal bed: the lattice spacing follows the floe size, so W/h is
    # the prescribed geometric ratio and ``fill_fraction`` sets how much of the
    # channel is filled (the realised area fraction of a dense bed is ~0.65)
    r_max = 1.2 * mean_r
    spacing = 1.02 * 2.0 * r_max
    box_width = cfg.body_width
    rows = max(2, int(cfg.fill_fraction * cfg.height / (spacing * math.sin(math.pi / 3.0))))
    cols = max(2, int(box_width / spacing) + 1)
    positions, radii = [], []
    jitter = 0.02 * spacing
    for j in range(rows):
        y = (j + 0.5) * spacing * math.sin(math.pi / 3.0)
        if y + 0.5 * spacing > cfg.height:
            break
        offset = 0.5 * spacing if j % 2 else 0.0
        for i in range(cols):
            x = (i + 0.5) * spacing + offset
            if x + 0.5 * spacing > box_width:
                continue
            r = mean_r * math.exp(sigma * float(rng.standard_normal()) - 0.5 * sigma**2)
            r = min(max(r, 0.6 * mean_r), r_max)
            px = float(np.clip(x + rng.uniform(-jitter, jitter), r, box_width - r))
            py = y + rng.uniform(-jitter, jitter)
            positions.append([px, py])
            radii.append(r)
    if len(radii) < 10:
        raise RuntimeError("bed placement produced too few disks")
    return np.asarray(positions, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def _apply_wall_friction(system: DiskSystem, *, mu_w: float, dt: float) -> None:
    """Coulomb friction impulse at the side walls (operator split).

    The :mod:`tensordem.granular2d` engine models the walls as *frictionless*
    segments (no tangential branch on segment contacts).  Janssen mechanics,
    however, is entirely about the vertical load the walls carry by friction, so
    the wall friction is applied here as an operator-split impulse after each
    step: the slip velocity at every wall contact is reduced by at most
    ``mu_w f_n dt`` per step, which is the Coulomb cone integrated over ``dt``.
    Only the translational slip is braked (no frictional torque), which
    over-estimates the slip slightly; the effect is quantified by the
    ``mu_w`` sensitivity case in the study.
    """
    if mu_w <= 0.0:
        return
    contacts = system.contact_forces()
    kind = np.asarray(contacts["kind"])
    wall = kind == 1
    if not wall.any():
        return
    disk = np.asarray(contacts["i"])[wall]
    fn = np.asarray(contacts["fn"])[wall]
    n = np.asarray(contacts["n"])[wall]
    tangent = np.stack([-n[:, 1], n[:, 0]], axis=1)
    slip = np.einsum("ij,ij->i", system.vel[disk], tangent)
    cap = mu_w * fn * dt
    impulse = np.clip(system.mass[disk] * slip, -cap, cap)
    np.add.at(system.vel, disk, -impulse[:, None] * tangent)


def _advance(system: DiskSystem, steps: int, *, mu_w: float, block: int = 50,
             friction_stride: int = 5) -> None:
    """Advance ``steps`` steps with the wall-friction operator split.

    The friction impulse is applied every ``friction_stride`` steps with the cone
    scaled by the stride (same impulse per unit time, one force evaluation per
    stride instead of per step).
    """
    dt = system.dt * int(friction_stride)
    remaining = int(steps)
    while remaining > 0:
        n = min(block, remaining)
        system.step(n)
        for _ in range(0, n, friction_stride):
            _apply_wall_friction(system, mu_w=mu_w, dt=dt)
        remaining -= n


def _wall_force_profile(system: DiskSystem, *, bin_height: float, thickness: float):
    """Depth profile of the side-wall normal stress (Pa) and its integral.

    The *differential* profile (stress per depth bin) is very noisy: a 10 cm bin
    of a floe bed holds only one or two wall contacts.  The measured quantity is
    therefore the **cumulative** wall stress ``Sigma_w(z) = int_0^z sigma_w dz'``,
    which integrates the whole contact population above ``z``; the differential
    profile is returned as a diagnostic only.  Both are normalised by the wall
    area ``dz * thickness`` of the bin.
    """
    contacts = system.contact_forces()
    kind = np.asarray(contacts["kind"])
    wall = kind == 1
    if not wall.any():
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    fn = np.asarray(contacts["fn"])[wall]
    point = np.asarray(contacts["point"])[wall]
    top = float(np.max(system.pos[:, 1] + system.radii))
    depth = top - point[:, 1]
    edges = np.arange(0.0, float(depth.max()) + bin_height, bin_height)
    if edges.size < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    idx = np.digitize(depth, edges) - 1
    centres, stress = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if not sel.any():
            continue
        stress.append(float(fn[sel].sum()) / (bin_height * thickness))
        centres.append(0.5 * (edges[b] + edges[b + 1]))
    centres = np.asarray(centres)
    stress = np.asarray(stress)
    cumulative = np.cumsum(stress * bin_height)
    return centres, stress, cumulative, stress


@dataclass
class ColumnRun:
    """Static-fill run used to verify the Janssen stress profile."""

    width: float
    bulk_density: float
    area_fraction: float
    heights: np.ndarray
    stresses: np.ndarray
    cumulative: np.ndarray
    system: DiskSystem
    settle_time: float
    n_contacts: int
    network: dict = dataclass_field(default_factory=dict)

    def fit(self, *, gravity: float = 9.81) -> JanssenFit:
        """Fit the cumulative wall stress to the integral of the Janssen law."""
        return fit_janssen(
            self.heights,
            self.cumulative,
            width=self.width,
            rho_bulk=self.bulk_density,
            gravity=gravity,
            cumulative=True,
        )

    def as_dict(self) -> dict:
        return dict(
            width_m=float(self.width),
            bulk_density_kg_m3=float(self.bulk_density),
            area_fraction=float(self.area_fraction),
            settle_time_s=float(self.settle_time),
            n_samples=int(self.heights.size),
            n_wall_contacts=int(self.n_contacts),
            network=self.network,
        )


def run_column(cfg: SiloConfig, *, settle_steps: int | None = None,
               speed_tol: float = 0.02) -> ColumnRun:
    """Fill a closed channel, settle it and measure the wall-pressure profile."""
    pos, radii = _polydisperse_bed(cfg)
    walls = box_walls(cfg.width, cfg.height, left=True, right=True, bottom=True, top=False)
    system = DiskSystem(pos, radii, cfg.disk_config, walls=walls)
    max_steps = int(settle_steps if settle_steps is not None else cfg.settle_steps)
    block = 500
    while system.steps < max_steps:
        _advance(system, block, mu_w=cfg.wall_friction)
        if system.mean_speed() < speed_tol:
            break
    system.vel[:] = 0.0
    system.omega[:] = 0.0
    heights, stresses, cumulative, _ = _wall_force_profile(
        system, bin_height=cfg.bin_height, thickness=cfg.thickness
    )
    area = float(np.sum(math.pi * system.radii**2))
    occupied = float(
        (system.pos[:, 1] + system.radii).max() - max(0.0, (system.pos[:, 1] - system.radii).min())
    )
    bulk = float(np.sum(system.mass)) / (cfg.width * max(occupied, 1e-6) * cfg.thickness)
    contacts = system.contact_forces()
    pair = np.asarray(contacts["kind"]) == 0
    network = contact_network_stats(
        {"i": np.asarray(contacts["i"])[pair], "j": np.asarray(contacts["j"])[pair],
         "normal_force": np.asarray(contacts["fn"])[pair]},
        system.n,
        area_fraction=area / (cfg.width * max(occupied, 1e-6)),
    )
    return ColumnRun(
        width=cfg.width,
        bulk_density=bulk,
        area_fraction=area / (cfg.width * max(occupied, 1e-6)),
        heights=heights,
        stresses=stresses,
        cumulative=cumulative,
        system=system,
        settle_time=system.time,
        n_contacts=int(np.sum(np.asarray(contacts["kind"]) == 1)),
        network=network,
    )


@dataclass
class SiloRun:
    """Outcome of one channel-discharge realisation."""

    width_ratio: float
    clogged: bool
    n_particles: int
    n_escaped: int
    mean_floe_diameter: float
    flow_rate: float
    residence_time: float
    outflow_times: list[float]
    final_mean_speed: float
    arch_span: float
    arch_reaches_both_walls: bool
    seed: int
    time_limit_hit: bool

    def as_dict(self) -> dict:
        return dict(
            width_ratio=float(self.width_ratio),
            clogged=bool(self.clogged),
            n_particles=int(self.n_particles),
            n_escaped=int(self.n_escaped),
            mean_floe_diameter_m=float(self.mean_floe_diameter),
            flow_rate_per_s=float(self.flow_rate),
            residence_time_s=float(self.residence_time),
            final_mean_speed_m_s=float(self.final_mean_speed),
            arch_span_m=float(self.arch_span),
            arch_span_over_floe=float(self.arch_span / self.mean_floe_diameter)
            if self.mean_floe_diameter > 0.0 else 0.0,
            arch_reaches_both_walls=bool(self.arch_reaches_both_walls),
            seed=int(self.seed),
            time_limit_hit=bool(self.time_limit_hit),
        )


def _strong_arch(system: DiskSystem, *, threshold: float = 1.0) -> tuple[float, bool]:
    """Horizontal span of the load-bearing (strong) contact network.

    Strong contacts are those whose normal force exceeds ``threshold`` times the
    mean normal force over all particle-particle contacts.  The largest spanning
    component that touches both side walls gives the arch span; if no component
    touches both walls the span of the widest component is returned with
    ``reaches_both_walls=False``.
    """
    contacts = system.contact_forces()
    kind = np.asarray(contacts["kind"])
    pair = kind == 0
    if not pair.any():
        return 0.0, False
    i = np.asarray(contacts["i"])[pair]
    j = np.asarray(contacts["j"])[pair]
    fn = np.asarray(contacts["fn"])[pair]
    mean = float(fn.mean())
    strong = fn > threshold * mean
    if not strong.any():
        return 0.0, False
    n = system.n
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i[strong], j[strong]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    labels = np.asarray([find(k) for k in range(n)])
    touching_left = system.pos[:, 0] - system.radii <= 1e-6
    touching_right = (
        system.pos[:, 0] + system.radii >= float(system.pos[:, 0].max() + 0)
    )
    # right wall position is the channel width; recover it from the wall segments
    wall_x = float(np.max(system.walls[:, :, 0]))
    touching_right = system.pos[:, 0] + system.radii >= wall_x - 1e-6
    best = (0.0, False)
    for label in np.unique(labels):
        sel = labels == label
        if not sel.any():
            continue
        span = float(
            (system.pos[sel, 0] + system.radii[sel]).max()
            - (system.pos[sel, 0] - system.radii[sel]).min()
        )
        reaches = bool((touching_left & sel).any() and (touching_right & sel).any())
        if span > best[0]:
            best = (span, reaches)
    return best


def run_silo(cfg: SiloConfig, *, settle_steps: int | None = None) -> SiloRun:
    """Discharge a broken-ice bed through an open channel and detect clogging."""
    pos, radii = _polydisperse_bed(cfg)
    mean_floe = 2.0 * float(np.mean(radii))
    box = cfg.body_width
    centre = 0.5 * box
    half = 0.5 * cfg.width
    if cfg.bottom_open:
        gate = np.asarray([
            [[0.0, 0.0], [centre - half, 0.0]],
            [[centre + half, 0.0], [box, 0.0]],
        ], dtype=np.float64)
    else:
        gate = np.asarray([[[0.0, 0.0], [box, 0.0]]], dtype=np.float64)
    walls = np.concatenate([
        box_walls(box, cfg.height, left=True, right=True, bottom=False, top=False),
        gate,
    ], axis=0)
    system = DiskSystem(pos, radii, cfg.disk_config, walls=walls)
    _advance(system, int(settle_steps if settle_steps is not None else cfg.settle_steps),
             mu_w=cfg.wall_friction)
    system.vel[:] = 0.0
    system.omega[:] = 0.0

    escaped = np.zeros(system.n, dtype=bool)
    outflow_times: list[float] = []
    last_outflow_time = system.time
    start_time = system.time
    chronicle: list[tuple[float, int]] = []
    time_limit_hit = False
    while system.time - start_time < cfg.max_time:
        _advance(system, cfg.block_steps, mu_w=cfg.wall_friction)
        new = (~escaped) & (system.pos[:, 1] < -0.30 * mean_floe)
        if new.any():
            escaped |= new
            outflow_times.extend([system.time] * int(new.sum()))
            last_outflow_time = system.time
            chronicle.append((system.time - start_time, int(new.sum())))
        remaining = ~escaped
        if remaining.any() and (system.time - last_outflow_time) >= cfg.stop_time:
            break
        if escaped.all():
            break
    else:
        time_limit_hit = True

    elapsed = max(system.time - start_time, 1e-12)
    remaining = ~escaped
    n_escaped = int(escaped.sum())
    # the flow stopped if material is still in the channel after the quiescent window
    clogged = bool(remaining.sum() > 0 and (system.time - last_outflow_time) >= cfg.stop_time)
    span, reaches = (0.0, False)
    if clogged:
        span, reaches = _strong_arch(system)
    flow_rate = n_escaped / elapsed
    mean_speed = float(np.mean(np.linalg.norm(system.vel[remaining], axis=1))) if remaining.any() else 0.0
    return SiloRun(
        width_ratio=cfg.width_ratio,
        clogged=clogged,
        n_particles=system.n,
        n_escaped=n_escaped,
        mean_floe_diameter=mean_floe,
        flow_rate=flow_rate,
        residence_time=float(np.mean(outflow_times) - start_time) if outflow_times else float("nan"),
        outflow_times=[float(t) for t in outflow_times],
        final_mean_speed=mean_speed,
        arch_span=span,
        arch_reaches_both_walls=reaches,
        seed=cfg.seed,
        time_limit_hit=time_limit_hit,
    )


# --------------------------------------------------------------------- criterion
def clogging_curve(runs) -> dict:
    """Aggregate realisations into ``P_clog(W/h)``."""
    grouped: dict[float, list[bool]] = {}
    for run in runs:
        grouped.setdefault(round(float(run.width_ratio), 6), []).append(bool(run.clogged))
    ratios = sorted(grouped)
    probability = [float(np.mean(grouped[r])) for r in ratios]
    counts = [len(grouped[r]) for r in ratios]
    spans = [float(np.mean([r.arch_span for r in runs if round(float(r.width_ratio), 6) == rr]))
             for rr in ratios]
    return dict(
        width_ratio=ratios,
        p_clog=probability,
        n_realisations=counts,
        mean_arch_span_m=spans,
        n_runs=int(len(list(runs))),
    )


def critical_width_ratio(curve: dict, *, level: float = 0.5) -> dict:
    """Width ratio at which ``P_clog`` crosses ``level`` (log-linear interpolation)."""
    x = np.asarray(curve["width_ratio"], dtype=np.float64)
    p = np.asarray(curve["p_clog"], dtype=np.float64)
    order = np.argsort(x)
    x, p = x[order], p[order]
    critical = float("nan")
    if p.size >= 2 and p[0] >= level > p[-1]:
        for k in range(1, p.size):
            if p[k] <= level < p[k - 1]:
                t = (p[k - 1] - level) / (p[k - 1] - p[k])
                critical = float(math.exp(
                    math.log(x[k - 1]) + t * (math.log(x[k]) - math.log(x[k - 1]))
                ))
                break
    elif p.size and np.all(p >= level):
        critical = float(x.min())
    elif p.size and np.all(p < level):
        critical = float(x.max())
    # bracketing interval: the largest ratio that still clogs and the smallest
    # one that does not
    lower = float(x[p >= level].max()) if np.any(p >= level) else float("nan")
    upper = float(x[p <= level].min()) if np.any(p <= level) else float("nan")
    return dict(
        critical_width_ratio=critical,
        level=float(level),
        lower_bracket=lower,
        upper_bracket=upper,
    )