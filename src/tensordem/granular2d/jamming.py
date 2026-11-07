"""Packing preparation, jamming-density protocol and the driven channel (ship) transit.

The module provides the mechanical protocol used by
``benchmarks/force_chain_jamming.py``:

1. :func:`bidisperse_radii` / :func:`random_packing_positions` build a dilute,
   non-overlapping 50:50 bidisperse disk assembly (bidispersity suppresses
   crystallisation, which would otherwise lock the jamming density to the
   hexagonal value 0.9069).
2. :func:`grow_to_fraction` implements the Lubachevsky-Stillinger style slow
   growth protocol: radii are inflated in small steps and the assembly is
   over-damped between steps.  The infinite-time (zero growth-rate) limit is the
   jamming volume fraction :math:`\\varphi_J`.
3. :func:`jamming_density` repeats the protocol at several growth rates and
   extrapolates :math:`\\varphi_J(\\dot\\varphi) \\to \\varphi_J(0)`, which is the
   external hard anchor of the study (frictionless disks 0.84, frictional
   0.80-0.84).
4. :func:`run_channel` puts the packing in a periodic-x channel with two banks
   and a fixed ship hull, drives the ice with a linear drift drag
   :math:`m(V_d - v)/\\tau_d` (the ship-frame equivalent of an ice drift
   current), and records hull resistance, bulk velocity and the force-chain
   state.  A run is *jammed* when the strong-force backbone bridges both banks
   and the mean ice speed has collapsed below ``2 %`` of ``V_d``.
5. :func:`piecewise_bic_breakpoint` locates the rheology breakpoint of
   :math:`\\tau(\\varphi)`, :math:`Z(\\varphi)` and :math:`P_{jam}(\\varphi)` with
   a piecewise-linear fit selected by the Bayesian information criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .engine import DiskConfig, DiskSystem, box_walls, ship_segments
from .force_chain import analyze_force_network, chain_components, coordination_number
from .stress import stress_invariants, virial_tensor

__all__ = [
    "BreakpointFit",
    "bidisperse_radii",
    "random_packing_positions",
    "grow_to_fraction",
    "jamming_density",
    "make_channel_system",
    "run_channel",
    "piecewise_bic_breakpoint",
]


# --------------------------------------------------------------------- packing
def bidisperse_radii(n: int, mean_radius: float, *, ratio: float = 1.4,
                     fraction_small: float = 0.5, seed: int = 0) -> np.ndarray:
    """50:50 bidisperse radii whose area-averaged radius equals ``mean_radius``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if mean_radius <= 0.0 or not math.isfinite(mean_radius):
        raise ValueError("mean_radius must be finite and positive")
    if ratio <= 1.0:
        raise ValueError("ratio must be > 1 for a bidisperse packing")
    if not (0.0 < fraction_small < 1.0):
        raise ValueError("fraction_small must lie in (0, 1)")
    r_small = mean_radius / math.sqrt(fraction_small + (1.0 - fraction_small) * ratio**2)
    radius = np.where(np.arange(n) % 2 == 0, r_small, r_small * ratio)
    rng = np.random.default_rng(seed)
    return radius[rng.permutation(n)]


def _inside_polygon(point: np.ndarray, segments: np.ndarray) -> bool:
    """Even-odd ray casting against the polygon whose vertices are ``segments[:, 0]``."""
    verts = np.asarray(segments[:, 0, :], dtype=np.float64)
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(verts)
    for k in range(n):
        x1, y1 = verts[k]
        x2, y2 = verts[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def _touches_segments(point: np.ndarray, radius: float, segments: np.ndarray) -> bool:
    """True when a disk of ``radius`` at ``point`` overlaps any segment."""
    if segments.size == 0:
        return False
    a = segments[:, 0, :]
    ab = segments[:, 1, :] - a
    ap = point[None, :] - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-30)
    t = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
    cp = a + t[:, None] * ab
    dist = np.linalg.norm(point[None, :] - cp, axis=1)
    return bool(np.any(dist < radius))


def random_packing_positions(n: int, radii: np.ndarray, width: float, height: float, *,
                             seed: int = 0, margin: float | None = None,
                             obstacles=None, max_tries: int = 400000) -> np.ndarray:
    """Random sequential adsorption of ``n`` disks inside ``[0,width]x[0,height]``.

    ``obstacles`` is an optional ``(S, 2, 2)`` segment set (e.g. a hull) that the
    disks must not touch.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) != n:
        raise ValueError("radii must have length n")
    rng = np.random.default_rng(seed)
    if margin is None:
        margin = float(radii.max())
    cell = 2.0 * float(radii.max()) + 1e-9
    nx = max(1, int(math.ceil(width / cell)))
    ny = max(1, int(math.ceil(height / cell)))
    grid: dict[tuple[int, int], list[int]] = {}
    pos = np.zeros((n, 2))
    lo = np.array([margin, margin])
    hi = np.array([width - margin, height - margin])
    for k in range(n):
        r = radii[k]
        placed = False
        for _ in range(max_tries // n + 2000):
            p = lo + rng.random(2) * (hi - lo)
            ok = True
            gx, gy = int(p[0] / cell), int(p[1] / cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for idx in grid.get((gx + dx, gy + dy), ()):
                        if np.linalg.norm(p - pos[idx]) < r + radii[idx] + 1e-12:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok and obstacles is not None:
                obs = np.asarray(obstacles, dtype=np.float64)
                ok = not (_touches_segments(p, r, obs) or _inside_polygon(p, obs))
            if ok:
                pos[k] = p
                grid.setdefault((gx, gy), []).append(k)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"could not place disk {k} of {n}: packing too dense")
    return pos


def _set_radii(system: DiskSystem, radii: np.ndarray) -> None:
    system.radii = np.array(radii, dtype=np.float64)
    cfg = system.config
    system.mass = cfg.density * math.pi * system.radii**2 * cfg.thickness
    system.inertia = 0.5 * system.mass * system.radii**2


def grow_to_fraction(system: DiskSystem, phi_target: float, *, dphi: float = 0.01,
                     relax_steps: int = 200, area: float | None = None) -> dict[str, object]:
    """Inflate the radii in steps until ``phi_target`` and record ``P(phi)``.

    The assembly is over-damped (``config.damping_time > 0``) so each growth step
    is followed by a quasi-static relaxation.  Returns the growth history; the
    contact pressure is taken from the discrete Virial stress of the last
    evaluation of each window, i.e. it contains *only* the contact contribution
    (a dilute gas with no contacts gives exactly zero).
    """
    area = system.domain_area() if area is None else float(area)
    if area is None or area <= 0.0:
        raise ValueError("a positive area (or walls) is required")
    phi0 = system.area_fraction(area)
    if phi_target <= phi0:
        raise ValueError("phi_target must exceed the current area fraction")
    base = system.radii.copy()
    n_inc = max(1, int(math.ceil((phi_target - phi0) / dphi)))
    history = {"phi": [], "pressure": [], "kinetic_energy": [], "coordination": []}
    for k in range(1, n_inc + 1):
        phi_k = phi0 + (phi_target - phi0) * k / n_inc
        _set_radii(system, base * math.sqrt(phi_k / phi0))
        system.step(relax_steps)
        system.contact_forces()
        sigma = virial_tensor(system, area=area)
        press = stress_invariants(sigma)["pressure"]
        n_contacts = int(np.sum(system.contacts["kind"] == 0)) if len(system.contacts["i"]) else 0
        history["phi"].append(phi_k)
        history["pressure"].append(press)
        history["kinetic_energy"].append(system.kinetic_energy())
        history["coordination"].append(2.0 * n_contacts / system.n)
    for key in history:
        history[key] = np.asarray(history[key])
    history["phi_target"] = phi_target
    return history


# ------------------------------------------------------------------- jamming
@dataclass(frozen=True)
class JammingResult:
    phi_j: float
    rates: tuple[float, ...]
    phi_j_by_rate: tuple[float, ...]
    slope: float
    threshold: float
    history: dict[str, np.ndarray]

    def as_dict(self) -> dict[str, object]:
        return {
            "phi_j": float(self.phi_j),
            "phi_j_by_rate": list(map(float, self.phi_j_by_rate)),
            "rates_per_step": list(map(float, self.rates)),
            "slope_P_over_phi": float(self.slope),
            "pressure_threshold": float(self.threshold),
        }


def _jamming_at_rate(seed: int, mu: float, mu_rolling: float, rate_steps: int, *,
                     n: int, mean_radius: float, phi_start: float, phi_max: float,
                     dphi: float, dt: float | None) -> tuple[float, dict[str, np.ndarray]]:
    radii = bidisperse_radii(n, mean_radius, seed=seed)
    area0 = float(np.sum(math.pi * radii**2)) / phi_start
    side = math.sqrt(area0)
    pos = random_packing_positions(n, radii, side, side, seed=seed, margin=0.0)
    cfg = DiskConfig(kn=3.0e4, mu=mu, mu_rolling=mu_rolling, zeta=0.3,
                     damping_time=2.0e-3, dt=dt, skin=0.0035)
    walls = box_walls(side, side)
    system = DiskSystem(pos, radii, cfg, walls=walls)
    history = grow_to_fraction(system, phi_max, dphi=dphi, relax_steps=rate_steps, area=area0)
    phi = history["phi"]
    p_tilde = history["pressure"] * mean_radius / cfg.kn
    threshold = 1.0e-4
    above = p_tilde > threshold
    if not above.any():
        return float("nan"), history
    first = int(np.argmax(above))
    lo = max(first - 1, 0)
    sel = slice(lo, min(first + 4, len(phi)))
    xs, ys = phi[sel], p_tilde[sel]
    if len(xs) >= 2 and np.ptp(xs) > 0.0:
        slope, intercept = np.polyfit(xs, ys, 1)
        phi_j = float(-intercept / slope) if slope > 0.0 else float(phi[first])
    else:
        phi_j = float(phi[first])
    return phi_j, history


def jamming_density(*, seed: int = 0, mu: float = 0.0, mu_rolling: float = 0.0,
                    n: int = 256, mean_radius: float = 0.012,
                    phi_start: float = 0.32, phi_max: float = 0.92,
                    dphi: float = 0.01, rates=(100, 400, 1600),
                    dt: float | None = None) -> JammingResult:
    """Jamming area fraction from the slow-growth protocol, extrapolated to zero rate.

    ``rates`` are relaxation steps per growth increment; slower growth (more
    steps) gives a denser jamming point.  The zero-rate limit is obtained by a
    linear fit of :math:`\\varphi_J` against :math:`1/\\text{steps}`.
    """
    if len(rates) < 2:
        raise ValueError("at least two growth rates are required for extrapolation")
    values = []
    last_history: dict[str, np.ndarray] = {}
    for steps in rates:
        phi_j, history = _jamming_at_rate(seed, mu, mu_rolling, int(steps), n=n,
                                          mean_radius=mean_radius, phi_start=phi_start,
                                          phi_max=phi_max, dphi=dphi, dt=dt)
        values.append(phi_j)
        last_history = history
    inverse = 1.0 / np.asarray(rates, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() < 2:
        raise RuntimeError("the growth protocol never reached the pressure threshold")
    slope, intercept = np.polyfit(inverse[good], np.asarray(values)[good], 1)
    phi_j0 = float(intercept)
    return JammingResult(phi_j=phi_j0, rates=tuple(float(r) for r in rates),
                         phi_j_by_rate=tuple(values), slope=float(slope),
                         threshold=1.0e-4, history=last_history)


# ------------------------------------------------------------------- channel
def make_channel_system(*, n_particles: int = 300, mean_radius: float = 0.014,
                        width: float = 0.40, length: float = 0.62,
                        ship_length: float = 0.22, ship_beam: float = 0.17,
                        ship_bow: float = 0.06, phi: float = 0.75,
                        seed: int = 0, mu: float = 0.5, mu_rolling: float = 0.05,
                        drift_speed: float = 0.20, drag_time: float = 0.02,
                        phi_start: float = 0.28, dphi: float = 0.02,
                        relax_steps: int = 150, dt: float | None = None,
                        damping_time: float = 2.0e-3):
    """Build a channel packing of area fraction ``phi`` around a fixed ship hull.

    ``phi`` is the ice concentration in the *free* channel area, i.e. the disk
    area divided by (channel area - hull area), which is the quantity that
    controls jamming around the obstacle.  The assembly is prepared by slow
    radius growth from a dilute random packing so the preparation is identical
    at every density.
    """
    x0 = 0.5 * (length - ship_length)
    y0 = 0.5 * (width - ship_beam)
    hull = ship_segments(ship_length, ship_beam, (x0, y0), bow=ship_bow)
    hull_area = ship_length * ship_beam
    free_area = width * length - hull_area
    radii = bidisperse_radii(n_particles, mean_radius, seed=seed)
    disk_area = float(np.sum(math.pi * radii**2))
    r0 = radii * math.sqrt(phi_start * free_area / disk_area)
    pos = random_packing_positions(n_particles, r0, length, width,
                                   seed=seed + 7, obstacles=hull)
    cfg = DiskConfig(kn=3.0e4, mu=mu, mu_rolling=mu_rolling, zeta=0.3,
                     damping_time=damping_time, dt=dt, skin=0.0035,
                     drift_velocity=(-drift_speed, 0.0), drag_time=drag_time)
    walls = box_walls(length, width, left=False, right=False, margin=0.05)
    system = DiskSystem(pos, r0, cfg, walls=walls, hull=hull, periodic_x=length)
    history = grow_to_fraction(system, phi, dphi=dphi, relax_steps=relax_steps, area=free_area)
    # the grown radii set the stability limit; use the (larger) production dt
    system.dt = min(system.recommended_dt, 3.0e-4)
    info = {
        "phi": float(system.area_fraction(free_area)),
        "phi_channel": float(system.area_fraction(width * length)),
        "n_particles": int(system.n),
        "free_area": float(free_area),
        "hull_area_fraction": float(hull_area / (width * length)),
        "n_growth_steps": int(len(history["phi"])),
        "growth_final_pressure": float(history["pressure"][-1]),
        "clearance": float(0.5 * (width - ship_beam)),
        "ship_length": ship_length, "ship_beam": ship_beam,
        "width": width, "length": length,
        "phi_history": history["phi"], "pressure_history": history["pressure"],
    }
    return system, info


def run_channel(system: DiskSystem, *, drift_speed: float, duration: float,
                area: float | None = None, warmup: float = 0.4,
                sample_every: int = 250, k_threshold: float = 1.0,
                axis: int = 1, bounds=None, seed: int = 0) -> dict[str, object]:
    """Drive the channel packing and measure resistance, mobility and force chains.

    The configuration is switched to pure dynamics (no damping) with the drift
    drag active, then integrated for ``duration`` seconds; the first ``warmup``
    fraction is discarded.  The jamming verdict requires *both* a bank-to-bank
    strong-force backbone and a collapsed mean speed (``< 2 %`` of the drift
    speed), which is the mechanical definition used by the campaign.
    """
    cfg = replace(system.config, damping_time=0.0,
                  drift_velocity=(-abs(drift_speed), 0.0))
    system.config = cfg
    system.zero_velocities()
    if bounds is None and len(system.walls):
        pts = system.walls.reshape(-1, 2)
        bounds = (float(pts[:, axis].min()), float(pts[:, axis].max()))
    n_steps = int(round(duration / system.dt))
    n_warm = int(round(n_steps * warmup))
    fx, fy, speed, z_all, aniso, load_ratio, span, n_chains, largest = [], [], [], [], [], [], [], [], []
    load_strong_span = []
    for step in range(n_steps):
        system.step(1)
        if step >= n_warm and (step - n_warm) % max(1, sample_every) == 0:
            system.contact_forces()
            net = analyze_force_network(system, k=k_threshold, axis=axis, bounds=bounds)
            fx.append(float(system.hull_force[0]))
            fy.append(float(system.hull_force[1]))
            speed.append(system.mean_speed())
            z_all.append(float(net["z_all"]))
            aniso.append(float(net["fabric_anisotropy"]))
            load_ratio.append(float(net["load_bearing_ratio"]))
            span.append(bool(net["spanning_all"]["bridging"]))
            n_chains.append(int(net["chain"]["n_chains"]))
            load_strong_span.append(bool(net["spanning"]["bridging"]))
            labels = net["labels_strong"]
            largest.append(int(np.bincount(labels).max()) if len(labels) else 0)
    fx = np.asarray(fx)
    speed = np.asarray(speed)
    mean_speed = float(speed.mean()) if len(speed) else 0.0
    jam = bool(np.mean(span) > 0.5) and bool(mean_speed < 0.02 * abs(drift_speed))
    hull_pts = system.hull.reshape(-1, 2) if len(system.hull) else np.zeros((1, 2))
    ship_length_ref = float(np.ptp(hull_pts[:, 0])) if len(system.hull) else 1.0
    resistance = float(-fx.mean()) if len(fx) else 0.0
    return {
        "phi": float(system.area_fraction(area)),
        "resistance_n": resistance,
        "resistance_std_n": float(fx.std()) if len(fx) else 0.0,
        "resistance_stress_pa": float(resistance / (cfg.thickness * max(ship_length_ref, 1e-9))),
        "lateral_force_n": float(np.mean(fy)) if len(fy) else 0.0,
        "mean_speed": mean_speed,
        "speed_ratio": float(mean_speed / abs(drift_speed)),
        "z_all": float(np.mean(z_all)) if z_all else 0.0,
        "fabric_anisotropy": float(np.mean(aniso)) if aniso else 0.0,
        "load_bearing_ratio": float(np.mean(load_ratio)) if load_ratio else 0.0,
        "spanning_fraction": float(np.mean(span)) if span else 0.0,
        "strong_spanning_fraction": float(np.mean(load_strong_span)) if load_strong_span else 0.0,
        "largest_chain": int(np.max(largest)) if largest else 0,
        "mean_n_chains": float(np.mean(n_chains)) if n_chains else 0.0,
        "jammed": jam,
        "samples": int(len(fx)),
        "seed": int(seed),
        "dt": float(system.dt),
        "fx_series": fx.tolist(),
        "speed_series": speed.tolist(),
    }


# ---------------------------------------------------------------- breakpoints
@dataclass(frozen=True)
class BreakpointFit:
    x_break: float
    slope_left: float
    slope_right: float
    bic_single: float
    bic_piecewise: float
    delta_bic: float
    index: int
    preferred: str

    def as_dict(self) -> dict[str, object]:
        return {
            "x_break": float(self.x_break),
            "slope_left": float(self.slope_left),
            "slope_right": float(self.slope_right),
            "bic_single": float(self.bic_single),
            "bic_piecewise": float(self.bic_piecewise),
            "delta_bic": float(self.delta_bic),
            "index": int(self.index),
            "preferred": self.preferred,
        }


def piecewise_bic_breakpoint(x, y, *, min_segment: int = 3) -> BreakpointFit:
    """Fit one vs two straight segments and choose by BIC.

    ``delta_bic = BIC(single) - BIC(piecewise)``; values ``> 10`` are decisive
    evidence for a breakpoint (Kass & Raftery 1995).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if len(x) < 2 * min_segment + 1:
        raise ValueError("not enough points for a piecewise fit")
    order = np.argsort(x)
    x, y = x[order], y[order]

    n = len(x)
    # round-off floor: an exactly linear data set has RSS ~ 1e-30 for both models
    # and would otherwise be decided by floating-point noise in the logarithm
    floor = max(1e-12 * float(np.sum((y - y.mean()) ** 2)), 1e-300)
    coef = np.polyfit(x, y, 1)
    rss_single = float(np.sum((y - np.polyval(coef, x)) ** 2))
    bic_single = n * math.log(max(rss_single, floor) / n) + 2 * math.log(n)

    best = None
    for idx in range(min_segment, n - min_segment + 1):
        xl, yl = x[:idx], y[:idx]
        xr, yr = x[idx:], y[idx:]
        cl = np.polyfit(xl, yl, 1)
        cr = np.polyfit(xr, yr, 1)
        r = float(np.sum((yl - np.polyval(cl, xl)) ** 2) + np.sum((yr - np.polyval(cr, xr)) ** 2))
        bic = n * math.log(max(r, floor) / n) + 4 * math.log(n)
        if best is None or bic < best[0]:
            best = (bic, idx, cl, cr)
    bic_pw, idx, cl, cr = best
    x_break = float(0.5 * (x[idx - 1] + x[idx]))
    delta = float(bic_single - bic_pw)
    return BreakpointFit(x_break=x_break, slope_left=float(cl[0]), slope_right=float(cr[0]),
                         bic_single=float(bic_single), bic_piecewise=float(bic_pw),
                         delta_bic=delta, index=int(idx),
                         preferred="piecewise" if delta > 10.0 else "single")

# --------------------------------------------------- isotropic servo compression
def scale_box(system: DiskSystem, factor: float) -> float:
    """Affinely scale the periodic box / walls about the box centre."""
    if len(system.walls):
        pts = system.walls.reshape(-1, 2)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (lo + hi)
        system.pos = centre + factor * (system.pos - centre)
        side = float((hi[0] - lo[0]) * factor)
        system.walls = box_walls(side, side, margin=0.0)
        if system.periodic_x is not None:
            system.pos[:, 0] = np.mod(system.pos[:, 0], side)
            system.periodic_x = side
        return side
    raise ValueError("scale_box needs a walled system")


def servo_pressure(system: DiskSystem, p_target: float, *, steps: int = 4000,
                   eta: float = 200.0, max_step: float = 3e-4,
                   every: int = 20) -> dict[str, float]:
    """Servo-control the box size so that the contact pressure tracks ``p_target``.

    ``d ln L / dt = (P - P_target) / eta``; ``eta`` is the servo viscosity (N s/m).
    Positive pressure mismatch expands the box, negative compresses it.
    """
    pressure = 0.0
    for k in range(steps):
        if k % every == 0:
            side = float(system.walls.reshape(-1, 2)[:, 0].max())
            system.contact_forces()
            pressure = stress_invariants(virial_tensor(system, area=side * side))["pressure"]
        delta = system.dt * (pressure - p_target) / eta
        delta = float(min(max(delta, -max_step), max_step))
        scale_box(system, 1.0 + delta)
        system.step(1)
    side = float(system.walls.reshape(-1, 2)[:, 0].max())
    system.contact_forces()
    pressure = stress_invariants(virial_tensor(system, area=side * side))["pressure"]
    return {"phi": system.area_fraction(side * side), "pressure": float(pressure),
            "side": side, "kinetic_energy": system.kinetic_energy()}


def isotropic_packing_at_pressure(*, mu: float = 0.0, mu_rolling: float = 0.0,
                                  n: int = 400, mean_radius: float = 0.010,
                                  side: float | None = None, seed: int = 0,
                                  p_compress: float = 3.0e3,
                                  compress_steps: int = 6000,
                                  eta: float = 250.0, dt: float | None = None,
                                  damping_time: float = 1.0e-3) -> tuple[DiskSystem, float]:
    """Start from a random (overlapping) configuration and compact it to ``p_compress``."""
    radii = bidisperse_radii(n, mean_radius, seed=seed)
    if side is None:
        side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.45)
    rng = np.random.default_rng(seed + 11)
    rmax = float(radii.max())
    pos = rmax + rng.random((n, 2)) * (side - 2.0 * rmax)
    cfg = DiskConfig(kn=3.0e4, mu=mu, mu_rolling=mu_rolling, zeta=0.4,
                     damping_time=damping_time, dt=dt, skin=0.003)
    system = DiskSystem(pos, radii, cfg, walls=box_walls(side, side, margin=0.0))
    res = servo_pressure(system, p_compress, steps=compress_steps, eta=eta)
    return system, side


def jamming_density_isotropic(*, mu: float = 0.0, mu_rolling: float = 0.0,
                              n: int = 400, mean_radius: float = 0.010,
                              seed: int = 0, p_compress: float = 3.0e3,
                              levels=(300.0, 100.0, 30.0, 10.0, 3.0, 1.0),
                              steps_per_level: int = 2500, eta: float = 150.0,
                              extrapolate_below: float = 30.0) -> dict[str, object]:
    """Jamming density from the dense (isotropic decompression) branch.

    The packing is compacted to ``p_compress`` and then decompressed through
    ``levels``; :math:`\\varphi_J` is the ``P -> 0`` extrapolation of the
    low-pressure part of :math:`\\varphi(P)`.  Both branches (loose growth,
    dense decompression) are protocol branches of the same physics; the dense
    branch is the one that corresponds to the classical random-close-packing
    value quoted for frictionless disks.
    """
    system, side = isotropic_packing_at_pressure(
        mu=mu, mu_rolling=mu_rolling, n=n, mean_radius=mean_radius, seed=seed,
        p_compress=p_compress)
    series = []
    for level in levels:
        res = servo_pressure(system, float(level), steps=steps_per_level, eta=eta)
        series.append((float(level), float(res["phi"]), float(res["pressure"])))
    p = np.asarray([s[0] for s in series])
    phi = np.asarray([s[1] for s in series])
    sel = p <= extrapolate_below
    slope, intercept = np.polyfit(p[sel], phi[sel], 1)
    return {
        "phi_j": float(intercept),
        "phi_at_lowest_pressure": float(phi[-1]),
        "dphi_dp": float(slope),
        "levels": [list(map(float, s)) for s in series],
        "mu": float(mu),
        "n": int(n),
        "seed": int(seed),
    }
