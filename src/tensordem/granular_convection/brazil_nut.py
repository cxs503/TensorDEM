"""Brazil-nut / reverse Brazil-nut experiments on a vertically shaken bimodal bed.

Observables (all defined so that the sign carries the physics)
-------------------------------------------------------------
``dz``   : mass-weighted centre-of-mass difference ``z_large - z_small`` [m].
           ``dz > 0``  -> large species on top  (Brazil-nut effect, BNE)
           ``dz < 0``  -> large species at bottom (reverse Brazil-nut, RBNE)
``S``    : area-weighted segregation degree of the large species at the free
           surface, ``S = (phi_top - phi_0) / phi_0`` (same convention as
           ``tensordem.bidisperse_segregation``), used here only as a secondary,
           profile-based reading of the same direction.
``tau``  : segregation timescale, in vibration cycles, obtained from the 63.2 %
           rise of ``|dz|`` towards its late-time asymptote.  ``tau`` is the
           quantity compared against the experimental anchor (AC3).

The competition that produces RBNE
----------------------------------
A size contrast drives the large species up (percolation into the void opened
under it during the dilation phase); a density contrast drives the dense species
down (condensation).  For a fixed size ratio the two mechanisms cancel at a
critical density ratio.  ``run_phase_scan`` maps the sign of ``dz`` over
``(Gamma, rho_large/rho_large=rho_l)`` and locates that boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bed import BedConfig, BedSystem, build_bed

__all__ = [
    "ShakeCase",
    "PhasePoint",
    "run_shake",
    "rise_history",
    "timescale_from_history",
    "segregation_timescale",
    "run_phase_scan",
    "segregation_profile",
]


@dataclass(frozen=True)
class ShakeCase:
    """One shaking run."""

    gamma: float = 2.0             # dimensionless peak acceleration A w^2 / g
    frequency: float = 20.0        # [Hz]
    n_cycles: int = 40
    sample_every: int = 2000
    stratified_init: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("gamma", "frequency", "n_cycles", "sample_every"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency

    def amplitude(self, gravity: float = 9.81) -> float:
        return self.gamma * gravity / self.omega**2

    @property
    def peak_acceleration(self) -> float:
        return self.gamma


@dataclass
class PhasePoint:
    gamma: float
    density_ratio: float
    dz: float
    dz_initial: float
    segregation: float
    direction: int
    tau_cycles: float
    dz_inf: float
    label: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "density_ratio": self.density_ratio,
            "dz": self.dz,
            "dz_initial": self.dz_initial,
            "segregation": self.segregation,
            "direction": self.direction,
            "tau_cycles": self.tau_cycles,
            "dz_inf": self.dz_inf,
            "label": self.label,
        }


def segregation_profile(system: BedSystem, is_large: np.ndarray,
                        n_bins: int = 10) -> dict:
    """Area-weighted large-species fraction per horizontal bin."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    y = system.pos[:, 1]
    area = math.pi * system.radii**2
    y0, y1 = float(y.min()), float(y.max() + system.radii.max())
    edges = np.linspace(y0, max(y1, y0 + 1e-6), n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    phi = np.full(n_bins, np.nan)
    area_z = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        a = area[sel]
        area_z[b] = float(a.sum())
        phi[b] = float(a[is_large[sel]].sum() / a.sum())
    phi_0 = float(area[is_large].sum() / area.sum())
    good = np.isfinite(phi)
    # Free-surface enrichment: area-weighted large fraction over the uppermost
    # 25 % of the *filled* bed.  (A plain area-weighted mean deviation over all
    # bins is identically zero by construction -- it carries no information and
    # would silently report S = 0 for a fully segregated bed.)
    y_fill = float(np.max(y + system.radii))
    top = edges[1:] >= y_fill - 0.25 * (y_fill - edges[0])
    sel = good & top
    if sel.sum() < 1 or phi_0 <= 0.0:
        S = 0.0
    else:
        S = float(np.sum(area_z[sel] * (phi[sel] - phi_0)) / np.sum(area_z[sel]) / phi_0)
    return {"phi": phi, "area_z": area_z, "phi_0": phi_0, "S": S,
            "edges": edges}


def rise_history(result: dict) -> dict:
    """Extract ``(cycles, dz)`` arrays from a :func:`run_shake` result."""
    hist = result["history"]
    cycles = np.array([h["cycles"] for h in hist], dtype=float)
    dz = np.array([h["dz"] for h in hist], dtype=float)
    return {"cycles": cycles, "dz": dz}


def timescale_from_history(cycles: np.ndarray, dz: np.ndarray, *,
                           tail_fraction: float = 0.25,
                           level: float = 1.0 - math.exp(-1.0)) -> dict:
    """63.2 % rise time of ``|dz|`` towards its late-time asymptote.

    The asymptote is the mean of the last ``tail_fraction`` of the record (that
    window is also required to be quasi-stationary: its drift is returned as a
    diagnostic).  The crossing is linearly interpolated, so the estimate is not
    quantised by the sampling interval.
    """
    cycles = np.asarray(cycles, dtype=float)
    dz = np.asarray(dz, dtype=float)
    if cycles.size != dz.size or cycles.size < 4:
        raise ValueError("need at least 4 matched samples")
    n_tail = max(int(round(tail_fraction * cycles.size)), 2)
    dz_inf = float(np.mean(dz[-n_tail:]))
    dz_0 = float(np.mean(dz[:max(1, cycles.size - n_tail)])) if False else float(dz[0])
    amp = abs(dz_inf - dz_0)
    out = {"dz_inf": dz_inf, "dz_0": dz_0, "amplitude": amp,
           "tails_drift": float(np.max(dz[-n_tail:]) - np.min(dz[-n_tail:]))}
    if amp <= 0.0:
        out.update({"tau_cycles": float("inf"), "crossed": False})
        return out
    target = dz_0 + level * (dz_inf - dz_0)
    reached = np.nonzero((dz - target) * math.copysign(1.0, dz_inf - dz_0) >= 0.0)[0]
    if reached.size == 0:
        out.update({"tau_cycles": float("inf"), "crossed": False})
        return out
    k = int(reached[0])
    if k == 0:
        out.update({"tau_cycles": float(cycles[0]), "crossed": True})
        return out
    y0, y1 = dz[k - 1], dz[k]
    if y1 == y0:
        frac = 0.0
    else:
        frac = (target - y0) / (y1 - y0)
    out.update({"tau_cycles": float(cycles[k - 1] + frac * (cycles[k] - cycles[k - 1])),
                "crossed": True})
    return out


def run_shake(case: ShakeCase, config: BedConfig | None = None, *,
              settle_steps: int = 3000, seed: int | None = None,
              n_bins: int = 10) -> dict:
    """Shake a bed and record the segregation history.

    The cell is integrated in the shaker frame, so the walls stay fixed and the
    vertical body acceleration oscillates as ``-g + A w^2 sin(w t)``.
    """
    config = config or BedConfig()
    system, is_large = build_bed(config, stratified=case.stratified_init,
                                 settle_steps=settle_steps, seed=seed)
    dt = system.dt
    omega = case.omega
    amp = case.amplitude(config.gravity)
    n = len(system.pos)

    def body_force(t: float) -> np.ndarray:
        acc = np.zeros((n, 2))
        acc[:, 1] = amp * omega**2 * math.sin(omega * t)
        return acc

    system.time_force = body_force
    z_large0 = system.gravity_centre(is_large)[1]
    z_small0 = system.gravity_centre(~is_large)[1]
    dz_initial = z_large0 - z_small0
    prof0 = segregation_profile(system, is_large, n_bins)

    n_steps = max(int(round(case.n_cycles / (case.frequency * dt))), 1)
    history = []
    done = 0
    while done < n_steps:
        chunk = min(case.sample_every, n_steps - done)
        system.step(chunk)
        done += chunk
        zl = system.gravity_centre(is_large)[1]
        zs = system.gravity_centre(~is_large)[1]
        prof = segregation_profile(system, is_large, n_bins)
        history.append({"time": float(system.time),
                        "cycles": float(system.time * case.frequency),
                        "dz": float(zl - zs),
                        "S": float(prof["S"]),
                        "ke": system.kinetic_energy(),
                        "contacts": int(system.last_contact_count)})
    prof = segregation_profile(system, is_large, n_bins)
    zl = system.gravity_centre(is_large)[1]
    zs = system.gravity_centre(~is_large)[1]
    fit = timescale_from_history(*[np.array([h[k] for h in history], dtype=float)
                                  for k in ("cycles", "dz")])
    return {
        "label": case.label,
        "gamma": case.gamma,
        "frequency": case.frequency,
        "amplitude": amp,
        "n_cycles": case.n_cycles,
        "n_particles": int(n),
        "dt": dt,
        "steps": int(done),
        "dz_initial": float(dz_initial),
        "dz": float(zl - zs),
        "dz_from_initial": float((zl - zs) - dz_initial),
        "S_initial": float(prof0["S"]),
        "S": float(prof["S"]),
        "direction": (0 if not (math.isfinite(zl) and math.isfinite(zs))
                      else int(np.sign(round((zl - zs) - dz_initial, 12)))),
        "tau_cycles": float(fit["tau_cycles"]),
        "dz_inf": float(fit["dz_inf"]),
        "tails_drift": float(fit["tails_drift"]),
        "crossed": bool(fit["crossed"]),
        "phi": prof["phi"].tolist(),
        "phi_0": float(prof["phi_0"]),
        "history": history,
        "final": system.snapshots(),
        "is_large": is_large.copy(),
    }


def segregation_timescale(case: ShakeCase, config: BedConfig | None = None,
                          **kwargs) -> dict:
    """Convenience wrapper: run one scenario and return the timescale block."""
    result = run_shake(case, config, **kwargs)
    return {"gamma": case.gamma, "tau_cycles": result["tau_cycles"],
            "dz": result["dz"], "dz_initial": result["dz_initial"],
            "direction": result["direction"], "crossed": result["crossed"],
            "dz_inf": result["dz_inf"]}


def run_phase_scan(gammas, density_ratios, *, config: BedConfig | None = None,
                   case: ShakeCase | None = None, **kwargs) -> list[PhasePoint]:
    """Map the segregation direction over ``(Gamma, rho_large / rho_small)``.

    The size ratio is held fixed (``config.d_large / config.d_small``), so the
    only competing contrast is the density one: the locus where ``dz`` changes
    sign is the BNE/RBNE boundary.
    """
    base = config or BedConfig()
    template = case or ShakeCase()
    points: list[PhasePoint] = []
    rho_ref = base.rho_small
    for gamma in gammas:
        for ratio in density_ratios:
            cfg = BedConfig(**{**base.__dict__, "rho_large": rho_ref * ratio})
            this = ShakeCase(gamma=float(gamma), frequency=template.frequency,
                             n_cycles=template.n_cycles,
                             sample_every=template.sample_every,
                             stratified_init=template.stratified_init,
                             label=f"g{gamma:g}_r{ratio:g}")
            res = run_shake(this, cfg, **kwargs)
            points.append(PhasePoint(
                gamma=float(gamma), density_ratio=float(ratio),
                dz=res["dz_from_initial"], dz_initial=res["dz_initial"],
                segregation=res["S"] - res["S_initial"],
                direction=res["direction"], tau_cycles=res["tau_cycles"],
                dz_inf=res["dz_inf"], label=this.label,
                extra={"dz_absolute": res["dz"]}))
    return points