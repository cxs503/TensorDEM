"""The floe-kinematics deliverable, and the metric that says whether it is usable.

The defect this module is the acceptance test for
--------------------------------------------------
Node P103W2-N03 inherits a clear-ice DEM whose floe kinematics were **not
deliverable**: at the step size the run needed to be affordable, the contact
model (linear spring with a hard cut-off) produced a force that alternated
between on and off between successive steps, so the floe velocity record was a
grid-frequency square wave with the physical drift buried in it.

Deliverable is defined here operationally, by four numbers that any consumer of
the kinematics can check:

``oscillation_index``
    Standard deviation of the *step-to-step* velocity increment divided by the
    standard deviation of the velocity itself, ``rms(dv) / rms(v)``.  For a
    smooth trajectory resolved by the step size this is ``O(dt * omega)``, i.e.
    small; for a contact-driven limit cycle it is ``O(1)`` because the velocity
    reverses every step.  The scale-free version reported is
    ``q = rms(dv) / (rms(v) * sqrt(2))`` for an ideal white-noise-limited
    record ``q -> 1``.

``velocity_high_frequency_fraction``
    Fraction of the kinetic-energy variance carried by the Nyquist half of the
    spectrum of the mean floe speed.  A resolved trajectory puts nearly all of
    it in the low half.

``overlap_ratio``
    ``max(delta) / r`` over the run.  A resolved contact never penetrates more
    than a few percent of the radius; a large value means the step size cannot
    follow the contact.

``energy_ledger``
    Kinetic energy bounded by the injection (drag work) and dissipation, with
    the elastic contact energy accounted for -- a numerical instability shows up
    as a secular drift of the total energy.

The kinematics payload itself (what the report and the artifact carry) is the
per-floe displacement / velocity statistics in the steady drift window, the
drift velocity of the field, the granular temperature (velocity fluctuation
about the local drift), and the wall-normal velocity profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KinematicsSummary", "kinematics_summary", "oscillation_index"]


@dataclass
class KinematicsSummary:
    drift_velocity: tuple[float, float]
    mean_speed: float
    rms_speed: float
    granular_temperature: float        # 0.5 * <|v - <v>|^2> per unit mass
    per_floe_displacement_rms: float
    per_floe_velocity_rms: float
    displacement_mean: float
    displacement_p95: float
    oscillation_index: float
    velocity_high_frequency_fraction: float
    max_overlap_ratio: float
    max_speed: float
    wall_normal_profile: dict
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "drift_velocity_m_per_s": [float(v) for v in self.drift_velocity],
            "mean_speed_m_per_s": float(self.mean_speed),
            "rms_speed_m_per_s": float(self.rms_speed),
            "max_speed_m_per_s": float(self.max_speed),
            "granular_temperature_J_per_kg": float(self.granular_temperature),
            "per_floe_displacement_rms_m": float(self.per_floe_displacement_rms),
            "per_floe_velocity_rms_m_per_s": float(self.per_floe_velocity_rms),
            "displacement_mean_m": float(self.displacement_mean),
            "displacement_p95_m": float(self.displacement_p95),
            "oscillation_index": float(self.oscillation_index),
            "velocity_high_frequency_fraction": float(self.velocity_high_frequency_fraction),
            "max_overlap_ratio": float(self.max_overlap_ratio),
            "n_samples": int(self.n_samples),
            "wall_normal_velocity_profile": self.wall_normal_profile,
        }


def oscillation_index(velocity: np.ndarray, dt: float) -> tuple[float, float]:
    """``(rms(dv)/(rms(v) sqrt2), high-frequency variance fraction)``.

    ``velocity`` is ``(S, N, 2)``: samples of the per-floe velocity.  The
    spectral split uses the time series of the *mean* floe speed, which is the
    field-scale signal a consumer of the kinematics actually uses.
    """
    v = np.asarray(velocity, dtype=float)
    if v.ndim != 3 or len(v) < 4:
        return float("nan"), float("nan")
    speeds = np.linalg.norm(v, axis=2).mean(axis=1)
    signal = speeds - speeds.mean()
    if signal.std() == 0.0:
        return float("nan"), float("nan")
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    half = max(1, len(spectrum) // 2)
    hf = float(spectrum[half:].sum() / spectrum[1:].sum()) if spectrum[1:].sum() > 0 else np.nan
    dv = np.diff(speeds)
    denom = speeds.std() * np.sqrt(2.0)
    q = float(dv.std() / denom) if denom > 0 else np.nan
    return q, hf


def kinematics_summary(out, config, *, n_bins: int = 6,
                       window: tuple[float, float] | None = None) -> KinematicsSummary:
    """Summarise one :class:`~tensordem.clear_ice.field.RunOutput`.

    ``window`` selects the sample interval in seconds; ``None`` uses the whole
    record (the physical payload: the drift and compression of the field).
    The resolution metrics should be taken on a window where the field is in
    sustained motion, which is what ``FieldConfig.kinematics_window`` is for.
    """
    if window is None:
        mask = np.ones(len(out.sample_times), dtype=bool)
    else:
        mask = (out.sample_times >= window[0]) & (out.sample_times <= window[1])
        if np.count_nonzero(mask) < 4:
            mask = out.sample_times >= config.warmup
    v = out.velocities[mask]
    x = out.positions[mask]
    if len(v) < 4:
        raise ValueError("not enough samples after warm-up for a kinematics summary")

    box = np.array([config.domain_x, config.domain_y])
    drift = v.mean(axis=(0, 1))
    speed = np.linalg.norm(v, axis=2)
    fluct = v - drift
    granular = float(0.5 * np.mean(np.sum(fluct ** 2, axis=2)))

    # unwrapped displacement between the first and last steady sample
    disp = x[-1] - x[0]
    disp -= box * np.round(disp / box)
    disp_mag = np.linalg.norm(disp, axis=1)

    q, hf = oscillation_index(v, out.dt)

    # wall-normal velocity profile upstream of the hull face
    c = config.gate_centre
    half = 0.5 * config.gate_width
    upstream = (x[-1][:, 0] - c[0]) < 0.0
    dx = x[-1][:, 0] - c[0]
    edges = np.linspace(-6.0 * half, 0.0, n_bins + 1)
    profile = {}
    for k in range(n_bins):
        sel = upstream & (dx >= edges[k]) & (dx < edges[k + 1])
        if not np.any(sel):
            continue
        key = f"x_minus_gate_{edges[k]:.2f}_{edges[k+1]:.2f}_m"
        profile[key] = {
            "n_floes": int(sel.sum()),
            "mean_u_m_per_s": float(v[:, sel, 0].mean()),
            "mean_v_m_per_s": float(v[:, sel, 1].mean()),
            "porosity_areal": float(1.0 - np.sum(np.pi * out.radii[sel] ** 2)
                                    / ((edges[k + 1] - edges[k]) * config.domain_y)),
        }

    return KinematicsSummary(
        drift_velocity=(float(drift[0]), float(drift[1])),
        mean_speed=float(speed.mean()),
        rms_speed=float(speed.std()),
        granular_temperature=granular,
        per_floe_displacement_rms=float(disp_mag.std()),
        per_floe_velocity_rms=float(np.linalg.norm(v, axis=2).std()),
        displacement_mean=float(disp_mag.mean()),
        displacement_p95=float(np.percentile(disp_mag, 95)),
        oscillation_index=q,
        velocity_high_frequency_fraction=hf,
        max_overlap_ratio=float(out.max_overlap_ratio),
        max_speed=float(speed.max()),
        wall_normal_profile=profile,
        n_samples=int(len(v)),
    )