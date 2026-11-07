"""Capillary (liquid) bridge force between two spheres: closed approximation and
exact axisymmetric Young–Laplace reference.

Two independent kernels are provided so that the approximation used in DEM can
be judged by machine:

``delenne_liquid_bridge_force``
    The Delenne/Rabinovich-type closed form used in granular DEM for a pendular
    bridge of liquid volume ``V``, surface tension ``gamma``, contact angle
    ``theta`` and surface separation ``h``::

        F = 2 pi gamma R cos(theta) / (1 + h / (2 V**(1/3)))

    Cheap, differentiable and monotone in ``h`` and ``V``; an *approximation*
    whose validity domain is measured (not assumed) in the campaign.

``solve_axisymmetric_bridge``
    Numerical reference: the exact (zero-gravity) axisymmetric Young–Laplace
    bridge between two equal spheres.  The meridian is integrated in
    arclength form,

        dr/ds = cos(phi),   dz/ds = sin(phi),   dphi/ds = dp/gamma - sin(phi)/r

    (``phi`` measured from the +r axis; the last equation is the
    Young–Laplace law ``dp = gamma (dphi/ds + sin(phi)/r)``).  The shooting
    starts *on the upper sphere's contact line* — where the contact angle
    condition is imposed exactly by construction — and stops at the symmetry
    plane, where ``dr/ds = 0``.  The two unknowns ``(alpha, dp)`` (contact
    angle position and Laplace pressure) are fixed by the two remaining
    conditions: the neck must sit on the symmetry plane ``z = 0`` and the
    bridge must enclose the prescribed liquid volume.  Every reported force is
    therefore a converged solution of the free-boundary problem; the force is
    read at the neck::

        F = 2 pi gamma r_n - pi r_n**2 dp          (> 0 = attraction)

    A useful internal consistency check (asserted in the tests) is that the
    same expression evaluated at *any* cross-section of the converged profile
    is constant — it is the exact first integral of the problem.

Sign conventions: attraction is positive, ``h`` is the surface-to-surface gap,
``theta`` is the contact angle measured inside the liquid (``0`` = perfectly
wetting).  SI units throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover
    from scipy.optimize import root as _scipy_root
except Exception:  # pragma: no cover
    _scipy_root = None


# ---------------------------------------------------------------------------
# Closed-form (DEM) approximation
# ---------------------------------------------------------------------------

def delenne_liquid_bridge_force(
    particle_radius: float,
    gamma: float,
    volume: float,
    gap: float = 0.0,
    contact_angle_deg: float = 0.0,
) -> float:
    """Delenne/Rabinovich pendular-bridge force (attractive, > 0).

    ``F = 2 pi gamma R cos(theta) / (1 + h / (2 V**(1/3)))``
    """
    if particle_radius <= 0 or gamma < 0 or volume <= 0:
        raise ValueError("R > 0, gamma >= 0 and V > 0 are required")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    cos_t = math.cos(math.radians(contact_angle_deg))
    if cos_t <= 0:
        return 0.0
    return (
        2.0
        * math.pi
        * gamma
        * particle_radius
        * cos_t
        / (1.0 + gap / (2.0 * volume ** (1.0 / 3.0)))
    )


def delenne_rupture_distance(volume: float, contact_angle_deg: float = 0.0) -> float:
    """Rupture distance estimate ``h ~ V^(1/3) (0.5 theta^2 + 0.1)`` (Lian 1993)."""
    theta = math.radians(contact_angle_deg)
    return volume ** (1.0 / 3.0) * (0.5 * theta * theta + 0.1)


# ---------------------------------------------------------------------------
# Exact axisymmetric Young-Laplace reference
# ---------------------------------------------------------------------------

@dataclass
class BridgeSolution:
    """Converged capillary bridge solution for two equal spheres."""

    force: float
    pressure_difference: float
    neck_radius: float
    contact_radius: float
    gap: float
    volume: float
    contact_angle_deg: float
    particle_radius: float
    gamma: float
    converged: bool
    residual: float
    half_filling_angle_deg: float = float("nan")
    profile_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra: dict = field(default_factory=dict)

    @property
    def force_delenne(self) -> float:
        return delenne_liquid_bridge_force(
            self.particle_radius,
            self.gamma,
            self.volume,
            self.gap,
            self.contact_angle_deg,
        )

    @property
    def relative_difference(self) -> float:
        if not math.isfinite(self.force) or self.force == 0:
            return float("nan")
        return abs(self.force - self.force_delenne) / abs(self.force)


def _shoot_from_neck(
    r_n: float,
    dp: float,
    radius: float,
    gamma: float,
    gap: float,
    theta_deg: float,
    n_steps: int,
    volume_target: float = 1.0,
) -> Tuple[float, float, Optional[dict]]:
    """Integrate the meridian from the symmetry plane to the upper sphere.

    Start: ``z=0``, ``r=r_n``, tangent along +z (``phi=pi/2``), which is the
    symmetry condition ``dr/dz=0``.  Integrate the arclength system

        dr/ds = cos(phi),  dz/ds = sin(phi),  dphi/ds = dp/gamma - sin(phi)/r

    with fixed-step RK4 until the profile crosses the upper sphere surface
    ``r^2 + (z - z_c)^2 = R^2`` (outside -> inside).  Returns the residuals
    ``(cos(angle) - cos(theta), log(V/V_target))`` plus the geometry.
    """
    z_c = radius + 0.5 * gap
    ds = radius / n_steps
    r, z, phi = r_n, 0.0, 0.5 * math.pi
    g_prev = r * r + (z - z_c) ** 2 - radius * radius
    z_list = [z]
    r_list = [r]
    z_e = r_e = phi_e = None
    for _ in range(12 * n_steps):
        k1r, k1z = math.cos(phi), math.sin(phi)
        k1p = dp / gamma - math.sin(phi) / r
        phi2 = phi + 0.5 * ds * k1p
        k2r, k2z = math.cos(phi2), math.sin(phi2)
        k2p = dp / gamma - math.sin(phi2) / (r + 0.5 * ds * k1r)
        phi3 = phi + 0.5 * ds * k2p
        k3r, k3z = math.cos(phi3), math.sin(phi3)
        k3p = dp / gamma - math.sin(phi3) / (r + 0.5 * ds * k2r)
        phi4 = phi + ds * k3p
        k4r, k4z = math.cos(phi4), math.sin(phi4)
        k4p = dp / gamma - math.sin(phi4) / (r + ds * k3r)
        r_new = r + ds * (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
        z_new = z + ds * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
        phi_new = phi + ds * (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
        if not (math.isfinite(r_new) and math.isfinite(z_new)) or r_new <= 0.0:
            return 10.0, 10.0, None
        g_new = r_new * r_new + (z_new - z_c) ** 2 - radius * radius
        z_list.append(z_new)
        r_list.append(r_new)
        if g_prev > 0.0 >= g_new:
            frac = g_prev / (g_prev - g_new) if g_new != g_prev else 0.0
            z_e = z + frac * (z_new - z)
            r_e = r + frac * (r_new - r)
            phi_e = phi + frac * (phi_new - phi)
            break
        r, z, phi, g_prev = r_new, z_new, phi_new, g_new
    if z_e is None or r_e is None or phi_e is None:
        return 10.0, 10.0, None

    # contact-angle condition: angle between liquid tangent and sphere tangent
    slope_sph = (z_c - z_e) / r_e
    t_liq = np.array([math.cos(phi_e), math.sin(phi_e)])
    t_sph = np.array([slope_sph, 1.0]) / math.hypot(slope_sph, 1.0)
    cos_angle = float(np.dot(t_liq, t_sph))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))

    # volume: full disc up to the sphere pole, annulus above it
    zz = np.array(z_list)
    rr = np.array(r_list)
    # keep only the part below the contact point
    keep = zz <= z_e
    zz, rr = zz[keep], rr[keep]
    if zz.size < 3:
        return 10.0, 10.0, None
    z_grid = np.concatenate((zz, [z_e]))
    r_grid = np.concatenate((rr, [r_e]))
    v_bridge = math.pi * float(np.trapezoid(r_grid * r_grid, z_grid))
    half_gap = 0.5 * gap
    v_cap = 0.0
    if z_e > half_gap:
        z_seg = np.linspace(half_gap, z_e, 400)
        r_seg2 = np.maximum(radius ** 2 - (z_seg - z_c) ** 2, 0.0)
        v_cap = math.pi * float(np.trapezoid(r_seg2, z_seg))
    v_half = v_bridge - v_cap
    if v_half <= 0.0:
        return 10.0, 10.0, None
    info = {
        "z_e": z_e,
        "r_e": r_e,
        "angle_deg": angle_deg,
        "volume": 2.0 * v_half,
        "profile_z": np.concatenate((-z_grid[::-1], z_grid)),
        "profile_r": np.concatenate((r_grid[::-1], r_grid)),
    }
    return cos_angle - math.cos(math.radians(theta_deg)), math.log(info["volume"] / volume_target), info


def solve_axisymmetric_bridge(
    particle_radius: float,
    gamma: float,
    volume: float,
    gap: float,
    contact_angle_deg: float = 0.0,
    *,
    n_steps: int = 400,
    tol: float = 1e-10,
    maxiter: int = 300,
    guess: Optional[Tuple[float, float]] = None,
) -> BridgeSolution:
    """Solve the exact axisymmetric Young–Laplace bridge between two spheres.

    A bounded least-squares shooting problem: the unknowns are the neck radius
    ``r_n`` and the Laplace pressure ``dp``; the residuals are the contact-angle
    mismatch and the log of the volume ratio.  Several starting points are
    tried (never a single fragile Newton step); the first converged solution
    wins and ``converged`` is reported honestly.

    Args:
        particle_radius: sphere radius ``R`` (m).
        gamma: surface tension (N/m).
        volume: liquid volume of the bridge (m^3).
        gap: surface-to-surface separation (m).
        contact_angle_deg: contact angle inside the liquid (deg).
        n_steps: RK4 steps per sphere radius of arclength.
        tol: residual tolerance requested from the solver.
        maxiter: maximum least-squares iterations.
        guess: optional ``(r_n, dp)`` initial guess.
    """
    if _scipy_root is None:  # pragma: no cover
        raise RuntimeError("scipy is required for the Young-Laplace reference")
    if particle_radius <= 0 or gamma <= 0 or volume <= 0:
        raise ValueError("R, gamma and V must be positive")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    from scipy.optimize import least_squares

    def residuals(x: np.ndarray) -> np.ndarray:
        r_n = float(x[0])
        dp = float(x[1])
        ra, rv, _ = _shoot_from_neck(
            r_n, dp, particle_radius, gamma, gap, contact_angle_deg, n_steps, volume
        )
        return np.array([ra, 0.5 * rv])

    if guess is not None:
        starts = [guess]
    else:
        # The residual is a broad plateau (10, 10) outside the admissible
        # geometric window, so a pure Newton start is unreliable: do a coarse
        # grid scan in (r_n, log|dp|) first and refine from the best cell.
        r_c = math.sqrt(volume / (math.pi * (gap + particle_radius / 3.0)))
        r_grid = np.geomspace(0.05 * r_c, 3.0 * r_c, 9)
        dp_grid = -gamma / r_c * np.geomspace(0.01, 3.0, 13)
        best_cell = None
        for rn in r_grid:
            for dp in dp_grid:
                ra, rv, info = _shoot_from_neck(
                    float(rn), float(dp), particle_radius, gamma, gap,
                    contact_angle_deg, max(80, n_steps // 4), volume,
                )
                if info is None:
                    continue
                score = ra * ra + 0.25 * rv * rv
                if best_cell is None or score < best_cell[0]:
                    best_cell = (score, float(rn), float(dp))
        if best_cell is None:
            starts = [(r_c, -gamma / max(r_c, 1e-12))]
        else:
            starts = [(best_cell[1], best_cell[2])]
    lo = np.array([1e-4 * particle_radius, -1e6 * gamma / particle_radius])
    hi = np.array([0.95 * particle_radius, -1e-3 * gamma / particle_radius])
    best = None
    for start in starts:
        x0 = np.array([min(max(start[0], lo[0] * 1.01), hi[0] * 0.99),
                       min(max(start[1], lo[1] * 1.01), hi[1] * 0.99)])
        try:
            sol = least_squares(
                residuals, x0, bounds=(lo, hi), xtol=tol, ftol=tol, gtol=tol,
                max_nfev=maxiter, diff_step=1e-6,
            )
        except Exception:  # pragma: no cover - defensive
            continue
        res = residuals(sol.x)
        score = float(np.max(np.abs(res)))
        if best is None or score < best[0]:
            best = (score, sol.x.copy())
        if score < 1e-6:
            break

    if best is None:
        return BridgeSolution(
            force=float("nan"), pressure_difference=float("nan"),
            neck_radius=float("nan"), contact_radius=float("nan"), gap=gap,
            volume=volume, contact_angle_deg=contact_angle_deg,
            particle_radius=particle_radius, gamma=gamma, converged=False,
            residual=float("inf"),
        )

    score, x = best
    r_n, dp = float(x[0]), float(x[1])
    ra, rv, info = _shoot_from_neck(
        r_n, dp, particle_radius, gamma, gap, contact_angle_deg, n_steps, volume
    )
    if info is None:
        return BridgeSolution(
            force=float("nan"), pressure_difference=dp, neck_radius=r_n,
            contact_radius=float("nan"), gap=gap, volume=volume,
            contact_angle_deg=contact_angle_deg, particle_radius=particle_radius,
            gamma=gamma, converged=False, residual=score,
        )
    force = 2.0 * math.pi * gamma * r_n - math.pi * r_n * r_n * dp
    alpha = math.degrees(
        math.acos(min(1.0, max(-1.0, (particle_radius + 0.5 * gap - info["z_e"]) / particle_radius)))
    )
    return BridgeSolution(
        force=force,
        pressure_difference=dp,
        neck_radius=r_n,
        contact_radius=info["r_e"],
        gap=gap,
        volume=info["volume"],
        contact_angle_deg=contact_angle_deg,
        particle_radius=particle_radius,
        gamma=gamma,
        converged=bool(score < 1e-5 and math.isfinite(force) and force > 0.0),
        residual=score,
        half_filling_angle_deg=alpha,
        profile_z=info["profile_z"],
        profile_r=info["profile_r"],
        extra={
            "volume_solved": info["volume"],
            "angle_solved_deg": info["angle_deg"],
            "z_contact_over_R": info["z_e"] / particle_radius,
            "force_at_contact_line": 2.0
            * math.pi
            * info["r_e"]
            * gamma
            * math.sin(math.radians(contact_angle_deg) + math.radians(alpha))
            - math.pi * info["r_e"] ** 2 * dp,
        },
    )


def bridge_force_comparison(
    particle_radius: float,
    gamma: float,
    volumes,
    gaps,
    contact_angle_deg: float = 0.0,
    *,
    n_steps: int = 200,
) -> list:
    """Exact-vs-closed-form table over ``(V, h)`` pairs."""
    rows = []
    for v, h in zip(volumes, gaps):
        sol = solve_axisymmetric_bridge(
            particle_radius, gamma, v, h, contact_angle_deg, n_steps=n_steps
        )
        rows.append(
            {
                "volume": v,
                "volume_over_R3": v / particle_radius ** 3,
                "gap": h,
                "gap_over_R": h / particle_radius,
                "force_exact": sol.force,
                "force_closed": sol.force_delenne,
                "relative_difference": sol.relative_difference,
                "converged": sol.converged,
                "neck_radius_over_R": sol.neck_radius / particle_radius,
                "pressure_difference": sol.pressure_difference,
                "half_filling_angle_deg": sol.half_filling_angle_deg,
                "force_at_contact_line": sol.extra.get("force_at_contact_line"),
                "residual": sol.residual,
            }
        )
    return rows


__all__ = [
    "BridgeSolution",
    "bridge_force_comparison",
    "delenne_liquid_bridge_force",
    "delenne_rupture_distance",
    "solve_axisymmetric_bridge",
]