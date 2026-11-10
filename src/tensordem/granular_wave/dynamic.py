"""Dynamic (wave-propagation) arm of the granular wave-speed verification.

Why this module exists
----------------------
:mod:`tensordem.granular_wave.emt` gives the *closed-form* Hertz-Mindlin/Digby
prediction, which reproduces ``v ~ p^(1/6)`` **by construction** -- evaluating the
formula is an algebraic identity, not a measurement.  The verification only means
something if the same scaling is recovered from a wave that actually travels
through a DEM packing.  This module provides the device:

``prepare_state``       build a *non-overlapping* lattice, jam it by monotone
                        affine compaction, then trim the confining pressure by
                        strain bisection on the **equilibrium** platen pressure and
                        freeze the cell at constant volume;
``ringdown_velocity``   free ring-down of the fundamental axial mode of the
                        fixed-fixed column (``v = 2 L f``) plus ``Q`` from the
                        log decrement of the envelope;
``steady_resonance``    steady sinusoidal sweep through the same resonance; a
                        Lorentzian fit gives ``f0`` (hence ``v``) and
                        ``Q = f0 / df`` -- the half-power bandwidth estimator.

Two velocity routes and two attenuation routes are therefore available on the
same specimen, which is what the acceptance protocol asks for.

Defects of the previous dynamic path that this module fixes
-----------------------------------------------------------
* ``build_packing`` places the lattice at ``1.001 * 2 R_max`` while the radii are
  polydisperse and the centres are jittered by ``0.08 R``, so the "non-overlapping"
  start state already interpenetrates by up to ~0.8 mm.  Starting the real
  stiffness there injects contact forces of order kN and the packing diverges
  (measured: kinetic energy 1.4 J -> 3.4e5 J within 4% strain, 30 MPa platen
  pressure).  :func:`build_loose_lattice` guarantees a positive gap instead.
* The shared servo uses ``v_max = 0.05`` m/s, i.e. 175 grain radii per millisecond,
  so a 1% pressure fluctuation becomes a bang-bang platen excursion that unloads
  the specimen (measured: 33 kPa -> 0.6 kPa in 5.5k steps).  Pressure is therefore
  controlled by *strain*, not by a servo.
* A single pressure reading is a transient, not an equilibrium value (measured:
  1.06 kPa -> 37 Pa over one relaxation history, 100 kPa -> 47 kPa over a hold).
  Only block-averaged, converged pressures are used.
* A travel-time pick needs a source shorter than the transit time (``lambda``
  < ~10 grain diameters) where the packing is already dispersive; measured, a
  12-diameter burst in a 32-grain column rang at the 7th axial mode with 5x the
  drive amplitude, and an unrestricted cross-correlation returned 113 km/s.
  The resonant routes avoid that trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Sequence

import numpy as np
import torch

from .engine import GranularPacking, GranularWaveConfig

__all__ = [
    "DynamicConfig",
    "build_loose_lattice",
    "safe_timestep",
    "prepare_loose",
    "set_pressure_by_strain",
    "prepare_state",
    "block_pressure",
    "equilibrate",
    "densify",
    "receiver_layout",
    "frequency_from_zero_crossings",
    "spectral_peak",
    "decay_q",
    "lorentzian_fit",
    "ringdown_velocity",
    "steady_resonance",
]


# --------------------------------------------------------------------------
# Initial state: a genuinely non-overlapping lattice
# --------------------------------------------------------------------------
def build_loose_lattice(cfg: GranularWaveConfig, *, gap: float = 1.06,
                        jitter: float = 0.05) -> dict[str, Any]:
    """Lattice state whose *closest pair cannot touch* before relaxation.

    Args:
        cfg:    packing configuration (``nx, ny, nz, radius, polydispersity``).
        gap:    lattice spacing in units of ``2 * R_max`` (>1 means a gap).
        jitter: centre displacement amplitude in units of the mean radius.

    The worst-case approach of two neighbours is ``2 * jitter * R_mean``; with
    ``gap = 1.06`` and ``jitter = 0.05`` the guaranteed clearance is
    ``0.06 * 2 R_max - 0.1 R_mean > 0`` for the default polydispersity, i.e. the
    first contact appears only when the servo compacts the box.
    """
    if gap <= 1.0:
        raise ValueError("gap must exceed 1 so the initial lattice is separated")
    clearance = (gap - 1.0) * 2.0 * cfg.radius_max - 2.0 * jitter * cfg.radius
    if clearance <= 0.0:
        raise ValueError("gap too small for the requested jitter: lattice would overlap")
    gen = torch.Generator().manual_seed(int(cfg.seed))
    axes = [torch.arange(v, dtype=torch.float64) for v in (cfg.nx, cfg.ny, cfg.nz)]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    spacing = gap * 2.0 * cfg.radius_max
    n = len(grid)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    radius = cfg.radius * (1.0 + cfg.polydispersity * (2.0 * u - 1.0))
    pos = grid * spacing + (torch.rand(n, 3, generator=gen, dtype=torch.float64) - 0.5) \
        * (2.0 * jitter * cfg.radius)
    pos = pos - 0.5 * (pos.max(dim=0).values + pos.min(dim=0).values)
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    m = cfg.radius_max
    wall_pos = torch.tensor(
        [-extent[0] / 2 - m, extent[0] / 2 + m,
         -extent[1] / 2 - m, extent[1] / 2 + m,
         -extent[2] / 2 - m, extent[2] / 2 + m],
        dtype=torch.float64,
    )
    mass = cfg.density * (4.0 / 3.0) * math.pi * radius.pow(3)
    inertia = 0.4 * mass * radius.pow(2)
    return {"positions": pos, "radii": radius, "mass": mass,
            "inertia": inertia, "wall_pos": wall_pos,
            "clearance_m": float(clearance)}


# --------------------------------------------------------------------------
# Confined state preparation
# --------------------------------------------------------------------------
@dataclass
class DynamicConfig:
    """Settings of the confinement/wave protocol (SI units unless stated)."""

    pressure_Pa: float = 1.0e5
    # --- preparation ---
    gap: float = 1.06
    jitter: float = 0.05
    settle_steps: int = 300
    strain_step_free: float = 2.0e-3    # strain increment while p == 0
    strain_step_bearing: float = 2.0e-3  # strain increment at p = 0+
    strain_step_min: float = 5.0e-5     # increment floor close to the target
    relax_steps: int = 40
    max_jam_increments: int = 400
    bisect_tol: float = 0.01            # |p - target| / target
    bisect_max_trials: int = 24
    trim_passes: int = 3
    hold_tol: float = 0.05
    hold_steps: int = 120
    hold_window: int = 25
    preload_factor: float = 1.6
    max_jam_strain: float = 0.32
    # --- wave measurement ---
    slab_thickness_factor: float = 3.0    # receiver slab thickness in mean radii
    resonance_fraction: float = 0.50      # receiver at mid-height (mode antinode)
    pulse_amplitude: float = 0.01         # m/s, kick amplitude (strain O(1e-5))
    kick_cycles: float = 1.5              # kick length in periods
    ringdown_cycles: int = 10             # free-decay record length in periods
    sweep_points: int = 7                 # steady sweep points across the resonance
    sweep_span: float = 0.40              # +-fraction of f0 covered by the sweep
    sweep_amplitude: float = 0.01
    sweep_warmup_cycles: int = 2
    sweep_measure_cycles: int = 3


def _pressures(sim: GranularPacking) -> np.ndarray:
    return sim.pressures().detach().cpu().numpy()


def _mean_pressure(sim: GranularPacking) -> float:
    return float(_pressures(sim).mean())


def _relax(sim: GranularPacking, steps: int, *, rebuild_every: int = 25) -> None:
    for n in range(steps):
        sim.step()
        if (n + 1) % rebuild_every == 0:
            sim.rebuild_pairs()


def safe_timestep(sim: GranularPacking, *, max_overlap_fraction: float = 0.01,
                  factor: float = 0.10) -> dict[str, float]:
    """Largest stable step for the *realised* contact stiffness.

    ``GranularWaveConfig.recommended_dt`` sizes the step for a 5% overlap, which
    is 5-20x above what a pressure-confined packing actually reaches
    (``max_overlap_fraction`` ~ 1e-3 ... 1e-2), so it wastes most of the cost of
    every wave record.  Here the stiffest realised Hertz/Mindlin contact sets the
    step instead: ``dt = factor * T_min`` with ``T_min`` the contact period of a
    grain against ``m_min`` at ``max_overlap_fraction`` (the same worst case for
    grains and for the rigid platens, which see ``E*_wall = 2 E*``).
    """
    cfg = sim.cfg
    r_star = 0.5 * cfg.radius_max
    delta = max_overlap_fraction * cfg.radius
    sq = math.sqrt(r_star * delta)
    k_n = 2.0 * cfg.e_star_wall * sq
    k_t = 8.0 * cfg.g_star_wall * sq
    k_max = max(k_n, k_t)
    m_min = cfg.density * (4.0 / 3.0) * math.pi * cfg.radius_min ** 3
    dt = factor * 2.0 * math.pi * math.sqrt(m_min / k_max)
    return {"dt_s": float(dt), "dt_factor_of_default": float(dt / sim.cfg.recommended_dt),
            "k_max_N_m": float(k_max), "m_min_kg": float(m_min),
            "max_overlap_fraction_assumed": max_overlap_fraction}


def prepare_loose(sim: GranularPacking, cfg: DynamicConfig | None = None) -> dict[str, Any]:
    """Drain the velocity of the freshly built (separated) lattice."""
    cfg = cfg or DynamicConfig()
    sim.quench = 0.95
    sim.damping_scale = 2.0
    _relax(sim, cfg.settle_steps, rebuild_every=20)
    sim.quench = 1.0
    sim.damping_scale = 1.0
    sim.vel.zero_()
    sim.omega.zero_()
    return {"steps": cfg.settle_steps, "pressure_Pa": _mean_pressure(sim),
            "porosity": sim.porosity()}


def block_pressure(sim: GranularPacking, steps: int = 120, *,
                   avg: int = 25, quench: float = 0.95) -> float:
    """Overdamped relaxation block followed by a time-averaged pressure read.

    The confining pressure of a finite DEM packing is the *equilibrium* platen
    reaction, not its instantaneous value: a 40-step reading of a state that is
    still rearranging is a transient and can be an order of magnitude above the
    relaxed value (measured: 1.06 kPa -> 37 Pa over one relaxation history).
    """
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.quench = quench
    _relax(sim, max(1, steps - avg))
    hist = []
    for _ in range(avg):
        sim.step()
        hist.append(_pressures(sim))
    sim.quench = 1.0
    return float(np.mean(np.asarray(hist), axis=0).mean())


def equilibrate(sim: GranularPacking, cfg: DynamicConfig | None = None,
                *, max_blocks: int = 12, tol: float = 0.01) -> dict[str, Any]:
    """Relax in blocks until the block-averaged pressure stops drifting."""
    cfg = cfg or DynamicConfig()
    history: list[float] = [block_pressure(sim, cfg.hold_steps, avg=cfg.hold_window)]
    rel = float("inf")
    for _ in range(max_blocks - 1):
        p = block_pressure(sim, cfg.hold_steps, avg=cfg.hold_window)
        rel = abs(p - history[-1]) / max(abs(history[-1]), 1e-30)
        history.append(p)
        if rel <= tol:
            break
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.quench = 1.0
    sim.vel.zero_()
    sim.omega.zero_()
    sim.rebuild_pairs()
    return {"pressure_Pa": history[-1], "history_Pa": history,
            "relative_block_change": rel, "blocks": len(history),
            "converged": bool(rel <= tol)}


def densify(sim: GranularPacking, pressure_target: float,
            cfg: DynamicConfig | None = None, *,
            factor: float = 1.4, check_every: int = 5) -> dict[str, Any]:
    """Affine isotropic compression until the *equilibrium* pressure exceeds target.

    The stopping rule is the relaxed pressure, not the running one: a jittered
    lattice is only at the *jamming onset* when its running pressure first
    spikes, and a state prepared there collapses to zero pressure as soon as it
    is allowed to relax (measured: solid fraction 0.51 at a 127 kPa transient,
    equilibrium pressure 0 after relaxation).  Compaction therefore continues
    until the relaxed pressure is ``factor`` times the target, which puts the
    specimen safely inside the jammed regime before the pressure is trimmed down.
    """
    cfg = cfg or DynamicConfig()
    sim.quench = 0.95
    strain = 0.0
    increments = 0
    p_eq = _mean_pressure(sim)
    while strain < cfg.max_jam_strain:
        step = cfg.strain_step_free if p_eq <= 0.0 else cfg.strain_step_bearing
        sim.pos = sim.pos * (1.0 - step)
        sim.wall_pos = sim.wall_pos * (1.0 - step)
        strain += step
        increments += 1
        sim.rebuild_pairs()
        _relax(sim, cfg.relax_steps)
        if increments % check_every == 0:
            p_eq = block_pressure(sim, cfg.hold_steps, avg=cfg.hold_window)
            if p_eq >= factor * pressure_target:
                break
    sim.quench = 1.0
    sim.vel.zero_()
    sim.omega.zero_()
    sim.rebuild_pairs()
    return {"increments": increments, "applied_strain": strain,
            "equilibrium_pressure_Pa": p_eq,
            "porosity": sim.porosity(),
            "solid_fraction": 1.0 - sim.porosity(),
            "coordination_number": sim.contact_network()["coordination_number"]}


def _scaled_trial(sim: GranularPacking, scale: float,
                  cfg: DynamicConfig) -> dict[str, Any]:
    """Equilibrium pressure of the cell affinely scaled by ``scale``."""
    pos0 = sim.pos.clone()
    wall0 = sim.wall_pos.clone()
    sim.pos = sim.pos * scale
    sim.wall_pos = sim.wall_pos * scale
    sim.rebuild_pairs()
    sim.vel.zero_()
    sim.omega.zero_()
    p = block_pressure(sim, cfg.hold_steps, avg=cfg.hold_window)
    out = {"pressure_Pa": p, "porosity": sim.porosity(),
           "kinetic_energy_J": sim.kinetic_energy()}
    sim.pos = pos0
    sim.wall_pos = wall0
    sim.vel.zero_()
    sim.omega.zero_()
    sim.rebuild_pairs()
    return out


def set_pressure_by_strain(sim: GranularPacking, target: float,
                           cfg: DynamicConfig | None = None) -> dict[str, Any]:
    """Bring the specimen to ``target`` confining pressure by strain bisection.

    A platen servo converts a pressure fluctuation into a platen velocity, so its
    accuracy is limited by the velocity clamp (the shared servo of this engine
    used ``v_max = 0.05`` m/s and unloaded a 33 kPa specimen to 0.6 kPa).  Here
    the *strain* is the control variable: the relaxed pressure of a
    constant-fabric packing is monotone in the affine scale factor, so the scale
    reproducing the target is found by bisection, each trial relaxing from the
    same stored state.  The specimen is frozen at constant volume afterwards,
    which is exactly the condition the wave measurement needs.
    """
    cfg = cfg or DynamicConfig()
    target = float(target)
    if target <= 0:
        raise ValueError("target pressure must be positive")
    trials: list[dict[str, Any]] = []

    def f(scale: float) -> float:
        rec = _scaled_trial(sim, scale, cfg)
        rec["scale"] = scale
        trials.append(rec)
        return rec["pressure_Pa"]

    p_now = _mean_pressure(sim)
    if p_now > target:
        s_lo, p_lo = 1.0, p_now
        delta = 5.0e-4
        p_hi, s_hi = p_now, 1.0
        while len(trials) < cfg.bisect_max_trials:
            s_hi = 1.0 + delta
            p_hi = f(s_hi)
            if p_hi <= target:
                break
            s_lo, p_lo = s_hi, p_hi
            delta *= 2.0
            if s_hi > 1.40:
                raise RuntimeError("specimen cannot be unloaded to the target: "
                                   "bracket exceeded 40% strain")
    else:
        s_hi, p_hi = 1.0, p_now
        delta = 5.0e-4
        s_lo, p_lo = 1.0, p_now
        while len(trials) < cfg.bisect_max_trials:
            s_lo = 1.0 - delta
            p_lo = f(s_lo)
            if p_lo >= target:
                break
            s_hi, p_hi = s_lo, p_lo
            delta *= 2.0
            if s_lo < 0.60:
                raise RuntimeError("specimen cannot be loaded to the target: "
                                   "bracket exceeded 40% strain")

    best = min(trials, key=lambda r: abs(r["pressure_Pa"] - target) / target)
    for _ in range(cfg.bisect_max_trials):
        if abs(best["pressure_Pa"] - target) / target <= cfg.bisect_tol:
            break
        s_mid = 0.5 * (s_lo + s_hi)
        p_mid = f(s_mid)
        if p_mid > target:
            s_lo, p_lo = s_mid, p_mid
        else:
            s_hi, p_hi = s_mid, p_mid
        if abs(p_mid - target) < abs(best["pressure_Pa"] - target):
            best = trials[-1]

    sim.pos = sim.pos * best["scale"]
    sim.wall_pos = sim.wall_pos * best["scale"]
    sim.rebuild_pairs()
    sim.vel.zero_()
    sim.omega.zero_()
    p_now = _mean_pressure(sim)
    return {"target_pressure_Pa": target,
            "achieved_pressure_Pa": p_now,
            "relative_error": abs(p_now - target) / target,
            "scale": best["scale"],
            "n_trials": len(trials),
            "trials": trials}


def prepare_state(sim: GranularPacking, target_pressure: float,
                  cfg: DynamicConfig | None = None) -> dict[str, Any]:
    """Full protocol: separated lattice -> jam -> equilibrate -> trim -> hold."""
    cfg = cfg or DynamicConfig()
    target = float(target_pressure)
    log: dict[str, Any] = {"target_pressure_Pa": target,
                           "preload_pressure_Pa": cfg.preload_factor * target}
    t0 = time.perf_counter()
    log["loose"] = prepare_loose(sim, cfg)
    log["loose"]["elapsed_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    log["jam"] = densify(sim, target, cfg, factor=cfg.preload_factor)
    log["jam"]["elapsed_s"] = time.perf_counter() - t0
    passes = []
    t0 = time.perf_counter()
    eq = equilibrate(sim, cfg)
    for _ in range(cfg.trim_passes):
        trim = set_pressure_by_strain(sim, target, cfg)
        eq = equilibrate(sim, cfg)
        passes.append({"trim": {k: trim[k] for k in (
            "achieved_pressure_Pa", "relative_error", "scale", "n_trials")},
            "equilibrate": eq})
        if abs(eq["pressure_Pa"] - target) / target <= cfg.hold_tol:
            break
    log["trim"] = {"passes": passes,
                   "achieved_pressure_Pa": eq["pressure_Pa"],
                   "relative_error": abs(eq["pressure_Pa"] - target) / target,
                   "elapsed_s": time.perf_counter() - t0}
    log["state"] = sim.state()
    return log


# --------------------------------------------------------------------------
# Receivers
# --------------------------------------------------------------------------
def receiver_layout(sim: GranularPacking, fractions: Sequence[float] = (0.22, 0.72),
                    thickness_factor: float = 3.0) -> dict[str, Any]:
    """Full cross-section slab receivers between the platens.

    Averaging over the *whole* cross-section is what makes the receiver a plane
    wave sensor: individual grains rattle, the section average does not.
    """
    z_bottom = float(sim.wall_pos[4]) + float(sim.radius.mean())
    z_top = float(sim.wall_pos[5]) - float(sim.radius.mean())
    height = z_top - z_bottom
    thickness = thickness_factor * float(sim.radius.mean())
    z = sim.pos[:, 2].detach().cpu().numpy()
    slabs = []
    for f in fractions:
        c = z_bottom + f * height
        idx = np.nonzero((z >= c - 0.5 * thickness) & (z <= c + 0.5 * thickness))[0]
        if idx.size == 0:
            raise RuntimeError(f"receiver slab at fraction {f} is empty")
        slabs.append(idx)
    distance = ((fractions[1] - fractions[0]) * height) if len(fractions) > 1 else float("nan")
    return {"z_bottom": z_bottom, "z_top": z_top, "height": height,
            "thickness": thickness,
            "centres": [z_bottom + f * height for f in fractions],
            "slabs": slabs, "distance": distance}


def _sponge(sim: GranularPacking, *, start_fraction: float, v_ref: float,
            gain: float = 4.0) -> None:
    z_bottom = float(sim.wall_pos[4]) + float(sim.radius.mean())
    z_top = float(sim.wall_pos[5]) - float(sim.radius.mean())
    z0 = z_bottom + start_fraction * (z_top - z_bottom)
    length = max(z_top - z0, 1e-9)
    sim.sponge = {"z0": z0, "z_top": z_top, "gamma": gain * v_ref / length}


def _drive(sim: GranularPacking, mode: str, value: float) -> None:
    """Platen source: normal velocity for P, tangential-x for S."""
    sim.wall_vnormal[4] = value if mode == "P" else 0.0
    sim.wall_tangential_drive.zero_()
    if mode == "S":
        sim.wall_tangential_drive[4, 0] = value


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
def _analytic_envelope(sig: np.ndarray) -> np.ndarray:
    """Analytic-signal envelope (zero padded, so no circular wrap-around)."""
    x = np.asarray(sig, dtype=float)
    n = x.size
    pad = 1 << int(math.ceil(math.log2(max(2 * n, 16))))
    spec = np.fft.fft(x, pad)
    h = np.zeros(pad)
    h[0] = 1.0
    h[pad // 2] = 1.0
    h[1:pad // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))[:n]


def _bandpass(sig: np.ndarray, dt: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Zero-phase FFT band pass (keeps the phase used by the frequency estimate)."""
    x = np.asarray(sig, dtype=float)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=dt)
    spec = spec * ((f >= f_lo) & (f <= f_hi))
    return np.fft.irfft(spec, x.size)


def frequency_from_zero_crossings(sig: np.ndarray, dt: float) -> dict[str, float]:
    """Frequency from the zero crossings of a narrow-band signal.

    Counting crossings over ``N`` periods gives a period estimate whose
    fractional uncertainty is ``~1/(2 N f dt)`` -- with 8 periods of 400 samples
    that is 1.5e-4, far better than the FFT bin width of a short ring-down.
    """
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    s = np.sign(x)
    s[s == 0] = 1.0
    cross = np.nonzero(np.diff(s) != 0)[0]
    if cross.size < 3:
        return {"frequency_hz": float("nan"), "n_crossings": int(cross.size)}
    # linear interpolation of the crossing instants
    t = []
    for i in cross:
        x0, x1 = x[i], x[i + 1]
        frac = abs(x0) / (abs(x0) + abs(x1)) if (abs(x0) + abs(x1)) > 0 else 0.5
        t.append((i + frac) * dt)
    t = np.asarray(t)
    n_half = t.size - 1
    freq = n_half / (2.0 * (t[-1] - t[0]))
    return {"frequency_hz": float(freq), "n_crossings": int(cross.size),
            "span_s": float(t[-1] - t[0])}


def spectral_peak(sig: np.ndarray, dt: float, *, f_lo: float, f_hi: float) -> dict[str, float]:
    """Dominant spectral line inside ``[f_lo, f_hi]`` (zero-padded, interpolated)."""
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    n = x.size
    pad = 1 << int(math.ceil(math.log2(max(8 * n, 64))))
    amp = np.abs(np.fft.rfft(x * np.hanning(n), pad))
    f = np.fft.rfftfreq(pad, d=dt)
    band = (f >= f_lo) & (f <= f_hi)
    if not bool(band.any()):
        return {"peak_hz": float("nan"), "peak_amplitude": float("nan")}
    idx = np.nonzero(band)[0]
    k = int(idx[np.argmax(amp[idx])])
    frac = 0.0
    if 0 < k < amp.size - 1:
        y0, y1, y2 = amp[k - 1], amp[k], amp[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-300:
            frac = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return {"peak_hz": float((k + frac) * (f[1] - f[0])),
            "peak_amplitude": float(amp[k])}


def decay_q(envelope: np.ndarray, dt: float, frequency: float,
            *, floor: float = 0.05) -> dict[str, float]:
    """``Q`` from the exponential decay of the ring-down envelope.

    ``env(t) = env0 exp(-t / tau)`` with ``Q = pi f tau`` (the standard
    log-decrement relation ``Q = pi f / gamma``).  The fit window is restricted to
    the part of the ring-down still above ``floor`` of the envelope maximum, where
    the decay is not yet swamped by numerical noise.
    """
    env = np.asarray(envelope, dtype=float)
    if env.size < 16 or env.max() <= 0:
        return {"q": float("nan"), "tau_s": float("nan"), "gamma_1_s": float("nan")}
    i_max = int(np.argmax(env))
    above = np.nonzero(env[i_max:] >= floor * env.max())[0] + i_max
    if above.size < 16:
        return {"q": float("nan"), "tau_s": float("nan"), "gamma_1_s": float("nan")}
    lo, hi = int(above[0]), int(above[-1])
    t = np.arange(lo, hi + 1) * dt
    y = np.log(np.clip(env[lo:hi + 1], 1e-300, None))
    slope = float(np.polyfit(t, y, 1)[0])
    gamma = -slope
    if gamma <= 0:
        return {"q": float("nan"), "tau_s": float("nan"), "gamma_1_s": float(gamma)}
    return {"q": float(math.pi * frequency / gamma), "tau_s": float(1.0 / gamma),
            "gamma_1_s": float(gamma)}


def lorentzian_fit(frequencies: Sequence[float], amplitudes: Sequence[float],
                   *, f_bounds: tuple[float, float], q_bounds: tuple[float, float] = (1.0, 80.0)
                   ) -> dict[str, float]:
    """Fit ``A(f) = A0 / sqrt(((f0^2-f^2)^2 + (f0 f/Q)^2))`` by grid search.

    ``f0`` and ``Q`` are searched on a grid (``A0`` follows from a one-parameter
    least-squares scale), which is enough for a seven-point resonance curve and
    needs no optimiser dependency.  The half-power width follows from the fit:
    ``Q = f0 / df``, i.e. this is the *half-power bandwidth* estimator.
    """
    f = np.asarray(frequencies, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    ok = np.isfinite(a) & (a > 0)
    if ok.sum() < 3:
        return {"f0_hz": float("nan"), "q_bandwidth": float("nan"),
                "amplitude_scale": float("nan"), "residual": float("nan"),
                "half_power_width_hz": float("nan")}
    f, a = f[ok], a[ok]
    best = (float("inf"), float("nan"), float("nan"), float("nan"))
    qs = np.geomspace(q_bounds[0], q_bounds[1], 160)
    f0s = np.linspace(f_bounds[0], f_bounds[1], 400)
    for f0 in f0s:
        for q in qs:
            shape = 1.0 / np.sqrt((f0 ** 2 - f ** 2) ** 2 + (f0 * f / q) ** 2)
            denom = float((shape * shape).sum())
            if denom <= 0:
                continue
            scale = float((a * shape).sum() / denom)
            resid = float(((a - scale * shape) ** 2).sum())
            if resid < best[0]:
                best = (resid, f0, q, scale)
    resid, f0, q, scale = best
    return {"f0_hz": float(f0), "q_bandwidth": float(q),
            "amplitude_scale": float(scale), "residual": float(resid),
            "half_power_width_hz": float(f0 / q) if q > 0 else float("nan")}


# --------------------------------------------------------------------------
# Resonant-column measurement
# --------------------------------------------------------------------------
def _burst_source(mode: str, amplitude: float, frequency: float, n_cycles: float):
    duration = n_cycles / frequency

    def source(t: float) -> float:
        if t >= duration:
            return 0.0
        return (amplitude * math.sin(2.0 * math.pi * frequency * t)
                * math.sin(math.pi * t / duration) ** 2)

    source.mode = mode
    source.duration = duration
    source.frequency = frequency
    return source


def _record(sim: GranularPacking, slabs: Sequence[np.ndarray], comp: int,
            n_steps: int, source) -> tuple[np.ndarray, np.ndarray]:
    """Run ``n_steps`` with ``source(t)`` and return the two receiver traces."""
    sig = np.zeros((len(slabs), n_steps))
    times = np.zeros(n_steps)
    for n in range(n_steps):
        t = n * sim.dt
        times[n] = t
        _drive(sim, source.mode if source is not None else "P",
               source(t) if source is not None else 0.0)
        sim.step()
        v = sim.vel[:, comp].detach().cpu().numpy()
        for k, idx in enumerate(slabs):
            sig[k, n] = v[idx].mean()
    return sig, times


def ringdown_velocity(sim: GranularPacking, *, mode: str = "P", v_ref: float = 250.0,
                      cfg: DynamicConfig | None = None,
                      keep_signals: bool = False) -> dict[str, Any]:
    """Free ring-down of the fundamental axial mode (resonant-column method).

    A ``n_cycles`` burst at the estimated fundamental frequency kicks the specimen
    and the drive is then removed; both platens are rigid, so the specimen is a
    fixed-fixed column whose fundamental is ``f1 = v / (2 L)`` with ``L`` the
    platen gap.  The wavelength is then ``lambda = 2 L`` -- two orders of
    magnitude above the grain size -- which is the regime in which the
    Hertz-Mindlin effective medium is supposed to hold.

    Why not a travel-time pick?  A separable first arrival needs the source to be
    shorter than the transit time, i.e. ``lambda < 2 d_receiver`` ~ 15 grain
    diameters.  At that wavelength the discrete packing is already dispersive
    (measured: a 12-diameter burst in a 32-grain column rang at the 7th axial
    mode, with platen amplitude 5x the drive), so a travel-time estimate measures
    the group velocity of a dispersive wave rather than the long-wavelength
    speed.  The resonant estimates avoid that trade-off.
    """
    cfg = cfg or DynamicConfig()
    if mode not in ("P", "S"):
        raise ValueError("mode must be 'P' or 'S'")
    L = float(sim.box_lengths()[2])
    f_guess = v_ref / (2.0 * L)
    layout = receiver_layout(sim, (cfg.resonance_fraction,), cfg.slab_thickness_factor)
    source = _burst_source(mode, cfg.pulse_amplitude, f_guess, cfg.kick_cycles)
    n_kick = int(math.ceil(source.duration / sim.dt))
    n_rec = int(math.ceil(cfg.ringdown_cycles / f_guess / sim.dt))
    comp = 2 if mode == "P" else 0
    sim.vel.zero_()
    sim.omega.zero_()
    sim.sponge = None
    sig, times = _record(sim, layout["slabs"], comp, n_kick + n_rec, source)
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.vel.zero_()
    sim.omega.zero_()
    sim.rebuild_pairs()
    trace = sig[0]
    peak = spectral_peak(trace[n_kick:], sim.dt,
                         f_lo=0.4 * f_guess, f_hi=2.5 * f_guess)
    f_peak = peak["peak_hz"]
    band = _bandpass(trace, sim.dt, 0.75 * f_peak, 1.25 * f_peak)
    seg = band[n_kick:]
    env = _analytic_envelope(seg)
    zc = frequency_from_zero_crossings(seg, sim.dt)
    freq = zc["frequency_hz"] if math.isfinite(zc["frequency_hz"]) else f_peak
    q = decay_q(env, sim.dt, freq)
    out: dict[str, Any] = {
        "mode": mode,
        "speed_m_s": float(2.0 * L * freq),
        "frequency_hz": float(freq),
        "frequency_spectral_peak_hz": float(f_peak),
        "column_length_m": L,
        "f_guess_hz": float(f_guess),
        "wavelength_m": float(2.0 * L),
        "wavelength_grains": float(2.0 * L / (2.0 * float(sim.radius.mean()))),
        "q_decay": q.get("q", float("nan")),
        "q_inv_decay": (1.0 / q["q"]) if q.get("q") and q["q"] > 0 else float("nan"),
        "tau_s": q.get("tau_s", float("nan")),
        "amplitude_m_s": float(np.abs(seg).max()),
        "drive_amplitude_m_s": cfg.pulse_amplitude,
        "n_steps": int(n_kick + n_rec),
        "kick_cycles": cfg.kick_cycles,
        "ringdown_cycles": cfg.ringdown_cycles,
        "receiver_m": float(layout["centres"][0]),
        "v_ref_m_s": float(v_ref),
    }
    if keep_signals:
        out["signals"] = {"times_s": times.tolist(), "trace": trace.tolist(),
                          "n_kick": int(n_kick)}
    return out


def steady_resonance(sim: GranularPacking, *, mode: str = "P", v_ref: float = 250.0,
                     cfg: DynamicConfig | None = None,
                     f0_guess: float | None = None,
                     reset=None) -> dict[str, Any]:
    """Steady-state frequency sweep through the resonance (half-power bandwidth).

    The platen is driven sinusoidally at ``n_points`` frequencies spanning
    ``f0_guess (1 +- span)``; at each frequency the lock-in amplitude at the
    mid-height receiver is recorded after a warm-up.  A Lorentzian fit to the
    resonance curve gives ``f0`` (hence ``v = 2 L f0``) and ``Q = f0 / df`` --
    the half-power bandwidth estimator -- independently of the free-decay route.
    """
    cfg = cfg or DynamicConfig()
    L = float(sim.box_lengths()[2])
    f0_guess = float(f0_guess) if f0_guess else v_ref / (2.0 * L)
    layout = receiver_layout(sim, (cfg.resonance_fraction,), cfg.slab_thickness_factor)
    comp = 2 if mode == "P" else 0
    span = cfg.sweep_span
    fractions = np.linspace(1.0 - span, 1.0 + span, cfg.sweep_points)
    freqs, amps = [], []
    for frac in fractions:
        if reset is not None:
            # a fresh (checkpointed) specimen per frequency: a strongly driven
            # packing at a sharp resonance can lose grains locally, and one
            # damaged point would otherwise corrupt every later frequency
            sim = reset()
        f_target = f0_guess * float(frac)
        n_cycle_steps = max(48, int(round(1.0 / (f_target * sim.dt))))
        f = 1.0 / (n_cycle_steps * sim.dt)
        n_warm = n_cycle_steps * cfg.sweep_warmup_cycles
        n_meas = n_cycle_steps * cfg.sweep_measure_cycles
        sim.vel.zero_()
        sim.omega.zero_()
        n_total = n_warm + n_meas
        sig = np.zeros(n_total)
        for n in range(n_total):
            t = n * sim.dt
            _drive(sim, mode, cfg.sweep_amplitude * math.sin(2.0 * math.pi * f * t))
            sim.step()
            sig[n] = sim.vel[layout["slabs"][0], comp].detach().cpu().numpy().mean()
        seg = sig[n_warm:]
        tt = (np.arange(n_meas) + n_warm) * sim.dt
        c = np.sum(seg * np.exp(-2j * math.pi * f * tt)) / n_meas
        freqs.append(f)
        amps.append(float(2.0 * abs(c)))
    sim.wall_vnormal.zero_()
    sim.wall_tangential_drive.zero_()
    sim.vel.zero_()
    sim.omega.zero_()
    sim.rebuild_pairs()
    fit = lorentzian_fit(freqs, amps, f_bounds=(f0_guess * (1 - span) * 0.95,
                                               f0_guess * (1 + span) * 1.05))
    return {
        "mode": mode,
        "speed_m_s": float(2.0 * L * fit["f0_hz"]) if math.isfinite(fit["f0_hz"]) else float("nan"),
        "frequency_hz": fit["f0_hz"],
        "q_bandwidth": fit["q_bandwidth"],
        "q_inv_bandwidth": (1.0 / fit["q_bandwidth"]) if fit["q_bandwidth"] > 0 else float("nan"),
        "half_power_width_hz": fit["half_power_width_hz"],
        "frequencies_hz": [float(f) for f in freqs],
        "amplitudes_m_s": [float(a) for a in amps],
        "column_length_m": L,
        "f0_guess_hz": f0_guess,
        "sweep_points": cfg.sweep_points,
        "sweep_span": span,
        "cycles_warmup": cfg.sweep_warmup_cycles,
        "cycles_measure": cfg.sweep_measure_cycles,
        "amplitude_m_s": cfg.sweep_amplitude,
        "fit": fit,
    }