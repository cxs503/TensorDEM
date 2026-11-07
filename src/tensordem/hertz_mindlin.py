"""Hertz–Mindlin contact model: normal Hertz spring + tangential Mindlin spring.

The Hertz–Mindlin model is the classical two-sphere contact theory for elastic
frictional spheres (Mindlin 1949, "Compliance of Elastic Bodies in Contact").

Normal direction (Hertz 1882):
    F_n = (4/3) E* sqrt(R*) delta^(3/2)
    k_n = F_n/delta = (4/3) E* sqrt(R*) sqrt(delta)   (secant stiffness)

Tangential direction (Mindlin no-slip):
    F_t = 8 G* sqrt(R* delta) * delta_t    (incremental, for small tangential displacement)
    k_t = 8 G* sqrt(R* delta)              (tangential stiffness, overlap-dependent)

where:
    1/E* = sum_i (1 - nu_i^2) / E_i        (reduced Young's modulus)
    1/R* = sum_i 1/R_i                      (effective radius)
    1/G* = sum_i (2 - nu_i) / G_i           (reduced shear modulus, Johnson 1985 eq.3.39)

For two equal spheres of the same material (G = E/(2(1+nu))):
    E* = E / (2 * (1 - nu^2))
    R* = R / 2
    G* = E / [4 * (2 - nu) * (1 + nu)]

The stiffness ratio for equal spheres is:
    k_t / k_n = 6 G* / E* = 3 (1 - nu) / (2 - nu)

For nu = 0.3: ratio ≈ 1.235
For nu = 0.25: ratio ≈ 1.333
For nu = 0.0: ratio = 1.5

Coulomb slip criterion (same as Cundall-Strack):
    |F_t| <= mu * |F_n|

When sliding occurs, the tangential displacement state is projected onto the
friction cone (delta_t <- F_t / k_t), exactly as in the Cundall-Strack model.

Energy accounting:
    - Stick phase: tangential spring stores recoverable energy U_t = 0.5 * k_t * |delta_t|^2
    - Slip phase: friction dissipates energy at rate P = F_t . (v_i - v_j) < 0

All quantities use SI units.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def hertz_reduced_shear_modulus(
    young_modulus: float,
    poisson_ratio: float,
    *,
    rigid_counterpart: bool = False,
) -> float:
    """Reduced shear modulus ``G*`` for Hertz–Mindlin tangential stiffness.

    The Mindlin tangential stiffness is ``k_t = 8 a G*`` where ``a`` is the
    contact radius.  The effective shear modulus follows Johnson (Contact
    Mechanics, 1985, eq. 3.39):

        1/G* = (2 - nu_1)/G_1 + (2 - nu_2)/G_2

    For two equal spheres of the same material (``G = E/(2(1+nu))``):

        G* = E / [4 (2 - nu) (1 + nu)]

    Against a rigid counterpart the compliance is entirely from the sphere:

        G* = E / [2 (2 - nu) (1 + nu)]

    Args:
        young_modulus:     Young's modulus E, Pa (positive).
        poisson_ratio:     Poisson's ratio nu, in [0, 0.5).
        rigid_counterpart: True for sphere-rigid contact, False for sphere-sphere.

    Returns:
        G*: reduced shear modulus, Pa.
    """
    if not math.isfinite(young_modulus) or young_modulus <= 0.0:
        raise ValueError("young_modulus must be finite and positive")
    if not (math.isfinite(poisson_ratio) and 0.0 <= poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")

    nu = poisson_ratio
    denom = (2.0 - nu) * (1.0 + nu)
    if rigid_counterpart:
        return float(young_modulus / (2.0 * denom))
    else:
        return float(young_modulus / (4.0 * denom))


def hertz_mindlin_tangential_stiffness(
    overlap: float | torch.Tensor,
    e_star: float,
    g_star: float,
    r_eff: float,
) -> torch.Tensor:
    """Mindlin tangential stiffness ``k_t = 8 G* sqrt(R* delta)``.

    This is the incremental tangential stiffness for the no-slip Mindlin
    solution.  It is overlap-dependent, just like the Hertz normal stiffness.

    Args:
        overlap:  Normal overlap delta, m (>= 0).
        e_star:   Reduced Young's modulus E*, Pa.
        g_star:   Reduced shear modulus G*, Pa.
        r_eff:    Effective radius R*, m.

    Returns:
        k_t: tangential stiffness, N/m (same shape as overlap).
    """
    delta = _hertz_mindlin_argument(overlap).clamp_min(0.0)
    return 8.0 * float(g_star) * math.sqrt(float(r_eff)) * delta.sqrt()


def hertz_mindlin_tangential_force(
    overlap: float | torch.Tensor,
    tangential_disp: torch.Tensor,
    e_star: float,
    g_star: float,
    r_eff: float,
    mu: float,
    normal_force_mag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute tangential force with Mindlin stiffness and Coulomb saturation.

    This combines the Mindlin no-slip tangential spring with the Coulomb
    friction limit:
        F_t_raw = k_t(delta) * delta_t
        |F_t| = min(|F_t_raw|, mu * |F_n|)

    The returned state is the projected displacement F_t / k_t, ensuring
    history-consistent sliding (Cundall-Strack projection).

    Args:
        overlap:            Normal overlap delta, m (>= 0).
        tangential_disp:    (N, D) accumulated tangential displacement.
        e_star:             Reduced Young's modulus E*, Pa.
        g_star:             Reduced shear modulus G*, Pa.
        r_eff:              Effective radius R*, m.
        mu:                 Coulomb friction coefficient (>= 0).
        normal_force_mag:   (N,) magnitude of normal contact force.

    Returns:
        (tangential_force, tangential_disp_next): saturated force and projected
        state, each (N, D).
    """
    if mu < 0:
        raise ValueError(f"mu must be nonnegative, got {mu}")

    k_t = hertz_mindlin_tangential_stiffness(overlap, e_star, g_star, r_eff)  # (N,)

    # Ensure k_t has the right shape for broadcasting
    if k_t.dim() == 0:
        k_t = k_t.expand(tangential_disp.shape[0])

    # Raw tangential force
    F_t_raw = k_t[:, None] * tangential_disp  # (N, D)

    # Coulomb limit
    F_limit = mu * normal_force_mag  # (N,)

    # Saturation
    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)  # (N,)
    eps = torch.finfo(tangential_disp.dtype).eps
    scale = torch.where(
        F_t_mag > F_limit,
        F_limit / F_t_mag.clamp_min(eps),
        torch.ones_like(F_t_mag),
    )
    F_t = F_t_raw * scale[:, None]

    # Project state onto friction cone
    return F_t, F_t / k_t[:, None]


def _hertz_mindlin_argument(value: float | torch.Tensor) -> torch.Tensor:
    """Promote a scalar overlap to a float64 tensor without changing tensor input."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError("overlap must be a floating-point tensor")
        return value.to(torch.float64) if value.dtype != torch.float64 else value
    return torch.tensor(float(value), dtype=torch.float64)


def mindlin_stiffness_ratio(poisson_ratio: float) -> float:
    """Ratio k_t / k_n for equal spheres in the Hertz–Mindlin model.

    Using the secant normal stiffness ``k_n = (4/3) E* sqrt(R* delta)``
    and the Mindlin tangential stiffness ``k_t = 8 G* sqrt(R* delta)``:

        k_t / k_n = 6 G* / E* = 3 (1 - nu) / (2 - nu)

    For nu = 0.3:  ratio ≈ 1.235
    For nu = 0.25: ratio ≈ 1.333
    For nu = 0.0:  ratio = 1.5

    This is useful for sanity checks: the tangential stiffness should never
    exceed a reasonable multiple of the normal stiffness.
    """
    if not (math.isfinite(poisson_ratio) and 0.0 <= poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    nu = poisson_ratio
    return 3.0 * (1.0 - nu) / (2.0 - nu)
