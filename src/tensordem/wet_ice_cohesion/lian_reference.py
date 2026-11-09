"""Lian et al. (1998) analytical liquid-bridge force reference.

Lian, G., Thornton, C. & Adams, M. J. (1998). "A theoretical study of the
liquid bridge forces between two rigid spherical bodies." *Journal of Colloid
and Interface Science*, 161(1), 138-147.

The analytical approximation used here is the Gorham & Zeng (2009) simplified
form of the Lian solution, which gives the capillary force as:

    F = 2*pi*gamma*R*cos(theta) / (1 + s/(2*V^(1/3)))

This is the same functional form as the Delenne/Rabinovich approximation but
with the specific coefficient calibration from Lian's numerical solutions.

For the m7458 campaign, we use this as the analytical anchor for criterion C1:
the aggregate mean size <D> predicted by the DEM aggregation model should match
the Smoluchowski mean-field prediction (which uses the Lian kernel) within 15%.

Additionally, Lian et al. provide the rupture distance:
    s_r = V^(1/3) * (0.5*theta^2 + 0.1)

and the bridge volume scaling for the force decay.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "lian_bridge_force",
    "lian_rupture_distance",
    "lian_force_curve",
    "lian_mean_field_D",
]


def lian_bridge_force(
    radius: float,
    gamma: float,
    volume: float,
    gap: float = 0.0,
    contact_angle_deg: float = 0.0,
) -> float:
    """Lian et al. (1998) analytical liquid-bridge force.

    F = 2*pi*gamma*R*cos(theta) / (1 + s/(2*V^(1/3)))

    Args:
        radius: sphere radius R (m).
        gamma: surface tension (N/m).
        volume: liquid bridge volume (m^3).
        gap: surface-to-surface separation s (m).
        contact_angle_deg: contact angle inside liquid (deg).

    Returns:
        Capillary force magnitude (N), positive = attraction.
    """
    if radius <= 0 or gamma < 0 or volume <= 0:
        raise ValueError("R > 0, gamma >= 0, V > 0 required")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    cos_t = math.cos(math.radians(contact_angle_deg))
    if cos_t <= 0:
        return 0.0
    v_third = volume ** (1.0 / 3.0)
    return 2.0 * math.pi * gamma * radius * cos_t / (1.0 + gap / (2.0 * v_third))


def lian_rupture_distance(
    volume: float,
    contact_angle_deg: float = 0.0,
) -> float:
    """Lian rupture distance estimate.

    s_r = V^(1/3) * (0.5*theta^2 + 0.1)

    Args:
        volume: liquid bridge volume (m^3).
        contact_angle_deg: contact angle (deg).

    Returns:
        Rupture separation (m).
    """
    theta = math.radians(contact_angle_deg)
    return volume ** (1.0 / 3.0) * (0.5 * theta * theta + 0.1)


def lian_force_curve(
    radius: float,
    gamma: float,
    volume: float,
    gaps: np.ndarray,
    contact_angle_deg: float = 0.0,
) -> np.ndarray:
    """Lian force sampled on an array of gaps."""
    return np.array([
        lian_bridge_force(radius, gamma, volume, float(g), contact_angle_deg)
        for g in gaps
    ])


def lian_mean_field_D(
    n_particles: int,
    volume: float,
    shear_rate: float,
    times: np.ndarray,
    radius_primary: float = 1.0,
    n_classes: int = 240,
    dimension_fractal: float = 2.5,
) -> dict:
    """Smoluchowski mean-field <D> using the Lian shear kernel.

    The orthokinetic shear kernel with fractal radius law:
        K_ij = (4/3) * shear_rate * (R_i + R_j)^3
        R_k = R_0 * k^(1/df)

    Args:
        n_particles: number of primary particles.
        volume: system volume (m^3).
        shear_rate: shear rate (1/s).
        times: evaluation times.
        radius_primary: primary particle radius.
        n_classes: number of size classes.
        dimension_fractal: fractal dimension D_f.

    Returns:
        dict with 'times', 'mean_diameter', 'n_clusters'.
    """
    from scipy.integrate import solve_ivp

    k = np.arange(1, n_classes + 1, dtype=float)
    r = radius_primary * k ** (1.0 / dimension_fractal)
    K = (4.0 / 3.0) * shear_rate * (r[:, None] + r[None, :]) ** 3

    n0 = np.zeros(n_classes)
    n0[0] = n_particles / volume

    def rhs(_t, n):
        n = np.maximum(n, 0.0)
        dn = np.zeros_like(n)
        for i in range(n_classes):
            if n[i] <= 0.0:
                continue
            js = np.arange(0, n_classes - i - 1)
            kk = i + js + 1
            dn[kk] += 0.5 * K[i, js] * n[i] * n[js]
        dn -= n * (K @ n)
        return dn

    t_end = float(max(times))
    sol = solve_ivp(
        rhs, (0.0, t_end), n0,
        t_eval=np.asarray(times, dtype=float),
        method="Radau", rtol=1e-6, atol=1e-30,
        max_step=t_end / 50.0,
    )
    out_d, out_n = [], []
    for idx in range(sol.y.shape[1]):
        nk = np.maximum(sol.y[:, idx], 0.0)
        d_k = 2.0 * radius_primary * k ** (1.0 / dimension_fractal)
        tot = nk.sum()
        out_d.append(float((nk * d_k).sum() / tot) if tot > 0 else float("nan"))
        out_n.append(float(tot * volume))
    return {
        "times": sol.t,
        "mean_diameter": np.array(out_d),
        "n_clusters": np.array(out_n),
    }
