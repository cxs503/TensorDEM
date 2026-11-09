"""Verification Surface 4: Dense limit Ergun inter-validation.

Sweep the solid volume fraction from dilute (phi = 1e-4) to dense packing
(phi = 0.45) and verify that the drag closure transitions correctly from
the dilute Schiller-Naumann limit to the Ergun packed-bed relation.

Physical basis
--------------
The Ergun equation for pressure drop in a packed bed::

    dp/dz = 150 mu (1-eps)^2 / (eps^3 d^2) u_s + 1.75 rho_f (1-eps) / (eps^3 d) u_s^2

In the DEM-CFD Model-A convention, the inter-phase momentum exchange per unit
volume is::

    f = beta(eps, u_rel) u_rel

where for the Ergun branch::

    beta_ergun = 150 mu (1-eps)^2 / (eps^2 d^2) + 1.75 rho_f (1-eps) |u_rel| / (eps^2 d)

Note the ``eps^2`` (not ``eps^3``) in the denominator. This is the Model-A
grouping that makes the force balance reproduce the textbook minimum
fluidisation velocity. The relation to the classic Ergun equation is::

    f = eps * (dp/dz)_classic

Test protocol
-------------
1. For each solid fraction phi in [1e-4, 0.45]:
   a. Compute the drag force per volume from the DEM side (Gidaspow blend).
   b. Compute the Ergun pressure gradient (Model-A form: multiply classic by eps).
   c. Compare: f_DEM should match f_Ergun_ModelA.

2. The transition should be smooth at the Gidaspow switch point (eps = 0.8).

3. At very low phi, the drag should reduce to the isolated-sphere Stokes drag.

4. At high phi (approaching packing), the Ergun viscous term should dominate.
"""

from __future__ import annotations

import math
import numpy as np

from ..cfd_dem_dense.forces import (
    GRAVITY,
    ForceModel,
    sphere_volume,
    sphere_projected_area,
    schiller_naumann_cd,
    per_particle_drag,
    drag_density,
)

__all__ = [
    "ergun_pressure_gradient_classic",
    "ergun_pressure_gradient_model_a",
    "dem_drag_per_volume",
    "run_ergun_sweep",
]


def ergun_pressure_gradient_classic(
    voidage: float,
    superficial_velocity: float,
    diameter: float,
    rho_fluid: float,
    mu: float,
) -> float:
    """Classic Ergun (1952) pressure gradient for a packed bed.

    dp/dz = 150 mu (1-eps)^2 / (eps^3 d^2) u_s + 1.75 rho_f (1-eps) / (eps^3 d) u_s^2
    """
    eps = voidage
    phi = 1.0 - eps
    u_s = abs(superficial_velocity)

    viscous = 150.0 * mu * phi**2 / (eps**3 * diameter**2) * u_s
    inertial = 1.75 * rho_fluid * phi / (eps**3 * diameter) * u_s**2

    return viscous + inertial


def ergun_pressure_gradient_model_a(
    voidage: float,
    superficial_velocity: float,
    diameter: float,
    rho_fluid: float,
    mu: float,
) -> float:
    """Model-A Ergun inter-phase force per unit volume.

    f = eps * (dp/dz)_classic

    This is the form used by the DEM-CFD beta function.
    """
    eps = voidage
    return eps * ergun_pressure_gradient_classic(eps, superficial_velocity, diameter, rho_fluid, mu)


def dem_drag_per_volume(
    voidage: float,
    superficial_velocity: float,
    diameter: float,
    rho_fluid: float,
    mu: float,
    model: str = "gidaspow",
) -> float:
    """DEM-side drag force per unit volume from the closure.

    f = beta(eps, u_rel) * u_rel
    """
    u_rel = abs(superficial_velocity)
    if u_rel < 1e-15:
        return 0.0

    fm = ForceModel(drag_model=model, pressure_gradient=True, added_mass=False,
                    fluid_reaction=True)
    beta_val = drag_density(u_rel, voidage, diameter=diameter,
                            rho_fluid=rho_fluid, mu=mu, model=fm)
    return beta_val


def run_ergun_sweep(
    diameter: float = 2.0e-3,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    n_phi: int = 30,
    n_velocities: int = 5,
    u_max: float = 0.5,
) -> dict:
    """Sweep solid fraction and compare DEM drag to Ergun (Model-A form)."""
    phi_values = np.concatenate([
        np.logspace(-4, np.log10(0.45), n_phi // 2),
        np.linspace(0.1, 0.45, n_phi // 2),
    ])
    phi_values = np.unique(phi_values)
    phi_values = phi_values[phi_values <= 0.45]
    n_phi_actual = len(phi_values)

    velocities = np.linspace(0.001, u_max, n_velocities)

    ergun_dp = np.zeros((n_phi_actual, n_velocities))
    dem_gidaspow = np.zeros((n_phi_actual, n_velocities))
    dem_wen_yu = np.zeros((n_phi_actual, n_velocities))

    for i, phi in enumerate(phi_values):
        eps = 1.0 - phi
        for j, u_s in enumerate(velocities):
            # Ergun (Model-A form)
            ergun_dp[i, j] = ergun_pressure_gradient_model_a(
                eps, u_s, diameter, rho_fluid, mu
            )

            # DEM Gidaspow
            dem_gidaspow[i, j] = dem_drag_per_volume(
                eps, u_s, diameter, rho_fluid, mu, model="gidaspow"
            )

            # DEM Wen-Yu (dilute branch only)
            if eps > 0.2:
                dem_wen_yu[i, j] = dem_drag_per_volume(
                    eps, u_s, diameter, rho_fluid, mu, model="wen_yu"
                )
            else:
                dem_wen_yu[i, j] = np.nan

    # Relative errors (Gidaspow vs Ergun Model-A)
    mask = ergun_dp > 1e-15
    rel_err_gidaspow = np.full_like(ergun_dp, np.nan)
    rel_err_gidaspow[mask] = np.abs(
        dem_gidaspow[mask] - ergun_dp[mask]
    ) / ergun_dp[mask]

    rel_err_wen_yu = np.full_like(ergun_dp, np.nan)
    mask_wy = mask & ~np.isnan(dem_wen_yu) & (dem_wen_yu > 1e-15)
    rel_err_wen_yu[mask_wy] = np.abs(
        dem_wen_yu[mask_wy] - ergun_dp[mask_wy]
    ) / ergun_dp[mask_wy]

    # Summary statistics
    # In the dense regime (eps < 0.8, phi > 0.2), Gidaspow = Ergun, so error should be ~0
    dense_mask = phi_values > 0.2
    dense_errors = rel_err_gidaspow[dense_mask]
    dense_errors = dense_errors[~np.isnan(dense_errors)]

    # In the dilute regime (eps > 0.8, phi < 0.2), Gidaspow switches to Wen-Yu
    dilute_mask = phi_values < 0.2
    dilute_errors_g = rel_err_gidaspow[dilute_mask]
    dilute_errors_g = dilute_errors_g[~np.isnan(dilute_errors_g)]

    return {
        "solid_fractions": phi_values.tolist(),
        "velocities": velocities.tolist(),
        "ergun_dp": ergun_dp.tolist(),
        "dem_drag_gidaspow": dem_gidaspow.tolist(),
        "dem_drag_wen_yu": dem_wen_yu.tolist(),
        "relative_errors_gidaspow": rel_err_gidaspow.tolist(),
        "relative_errors_wen_yu": rel_err_wen_yu.tolist(),
        "dense_regime_max_error": float(np.max(dense_errors)) if len(dense_errors) > 0 else None,
        "dense_regime_mean_error": float(np.mean(dense_errors)) if len(dense_errors) > 0 else None,
        "dilute_regime_max_error_gidaspow": float(np.max(dilute_errors_g)) if len(dilute_errors_g) > 0 else None,
        "gidaspow_switch_voidage": 0.8,
        "n_phi": n_phi_actual,
        "n_velocities": n_velocities,
    }
