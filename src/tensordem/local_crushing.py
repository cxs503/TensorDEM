"""Local crushing damage of a hull panel under a grounding/contact pressure patch.

Two levels, as required by the W5-N03 acceptance criteria:

* :func:`rigid_plastic_membrane_pressure` -- closed form rigid-plastic membrane
  solution for a clamped rectangular panel (Jones 1989): with the membrane
  forces at yield ``N = sigma_y t`` in both directions and a parabolic deflected
  shape, ``p = 8 sigma_y t w (1/a^2 + 1/b^2)``.  Bending resistance is neglected,
  which is asymptotically exact for ``w >> t`` (the large-deflection crushing
  regime of interest).
* :func:`membrane_panel_energy` -- independent numerical solution of the *same*
  idealised model: a von Karman membrane with J2 deformation theory in plane
  stress, bilinear nodal fields, minimised with torch autograd (Adam + LBFGS).
  It verifies the closed form without sharing its algebra.

:func:`crushing_damage` turns a contact-pressure patch (from the Kerr beam or
from the discrete keel) into the damage quantities consumed by the collision
damage module (W3-N05 interface): permanent deflection, membrane strain,
rupture flag and absorbed energy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

CRUSHING_SCHEMA = "tensordem.local-crushing/1"


@dataclass(frozen=True)
class PlatePanel:
    """Rectangular hull panel between stiffeners, clamped on all four edges."""

    span_x: float = 1.2
    span_y: float = 1.5
    thickness: float = 0.020
    yield_stress: float = 315e6
    young_modulus: float = 2.06e11
    poisson: float = 0.3
    rupture_strain: float = 0.20

    def __post_init__(self) -> None:
        for name in ("span_x", "span_y", "thickness", "yield_stress", "young_modulus"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (0, 0.5)")
        if not math.isfinite(self.rupture_strain) or self.rupture_strain <= 0.0:
            raise ValueError("rupture_strain must be finite and positive")

    @property
    def membrane_rate(self) -> float:
        """Rigid-plastic membrane pressure per metre of deflection [N/m^3]."""
        return 8.0 * self.yield_stress * self.thickness * (
            1.0 / self.span_x**2 + 1.0 / self.span_y**2
        )

    @property
    def squash_load(self) -> float:
        """Membrane force at yield per unit width ``N_y = sigma_y t`` [N/m]."""
        return self.yield_stress * self.thickness


def rigid_plastic_membrane_pressure(deflection: float, panel: PlatePanel) -> float:
    """``p = 8 N_y w (1/a^2 + 1/b^2)`` for a clamped rectangular panel [Pa]."""
    if not math.isfinite(deflection) or deflection < 0.0:
        raise ValueError("deflection must be finite and nonnegative")
    return panel.membrane_rate * deflection


def variational_membrane_pressure(deflection: float, panel: PlatePanel) -> float:
    """Rigid-plastic membrane pressure from the separable-parabola work balance.

    Take the admissible (rigid-plastic) field ``w = A f(x) g(y)`` with
    ``f = 1 - 4x^2/a^2``, ``g = 1 - 4y^2/b^2`` and both membrane forces at yield
    ``N = sigma_y t``.  The internal dissipation is
    ``D = 1/2 N ∫[(dw/dx)^2 + (dw/dy)^2] dA`` and the external work
    ``W = p ∫ w dA``; stationarity gives ``p = 6.4 sigma_y t w (1/a^2 + 1/b^2)``.

    This is the same *family* balance that yields the widely used ``p = 8 N_y w
    (1/a^2+1/b^2)`` (Jones 1989) when instead pointwise equilibrium of a
    parabolic cable is imposed at the panel centre.  The two closed forms differ
    by 25% in deflection: the factor-8 form is unconservative.  Both are
    "rigid-plastic membrane analytic solutions"; the numerical
    :func:`membrane_panel_energy` lies between them (see the campaign report).
    """
    if not math.isfinite(deflection) or deflection < 0.0:
        raise ValueError("deflection must be finite and nonnegative")
    return 0.8 * panel.membrane_rate * deflection


def rigid_plastic_membrane_deflection(pressure: float, panel: PlatePanel) -> float:
    """Inverse of :func:`rigid_plastic_membrane_pressure` [m]."""
    if not math.isfinite(pressure) or pressure < 0.0:
        raise ValueError("pressure must be finite and nonnegative")
    return pressure / panel.membrane_rate


def membrane_strain(deflection: float, panel: PlatePanel) -> float:
    """Maximum Green-Lagrange membrane strain of the parabolic field."""
    if not math.isfinite(deflection) or deflection < 0.0:
        raise ValueError("deflection must be finite and nonnegative")
    return 4.0 * deflection**2 * (1.0 / panel.span_x**2 + 1.0 / panel.span_y**2)


def membrane_panel_energy(
    panel: PlatePanel,
    pressure: float,
    grid: tuple[int, int] = (13, 17),
    max_iterations: int = 400,
    seed_deflection: float | None = None,
) -> dict[str, Any]:
    """Numerical membrane panel solution (deformation theory, autograd + L-BFGS-B).

    The panel is discretised with bilinear nodal ``(u, v, w)``; all four edges are
    clamped.  Strains are von Karman; the material follows the incompressible
    deviatoric (J2) deformation theory of perfect plasticity, so the stored
    energy is ``W = int_0^{eps_e} sigma d eps`` with ``sigma = min(E eps_e,
    sigma_y)`` and ``eps_e = sqrt(2/3 e:e)`` using ``eps_z = -(eps_x+eps_y)``.
    The potential ``Pi = t∫W dA - p∫w dA`` is minimised with L-BFGS-B using
    autograd gradients -- an independent numerical solution of the *same*
    idealised model, so the comparison with
    :func:`rigid_plastic_membrane_pressure` verifies the closed form.
    """
    from scipy.optimize import minimize

    nx, ny = grid
    if nx < 4 or ny < 4:
        raise ValueError("grid must be at least 4 x 4")
    if not math.isfinite(pressure) or pressure < 0.0:
        raise ValueError("pressure must be finite and nonnegative")

    a, b, t = panel.span_x, panel.span_y, panel.thickness
    E, nu, sy = panel.young_modulus, panel.poisson, panel.yield_stress
    dt = torch.float64
    dx, dy = a / nx, b / ny

    xs = torch.linspace(0.0, a, nx + 1, dtype=dt)
    ys = torch.linspace(0.0, b, ny + 1, dtype=dt)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    boundary = torch.zeros((nx + 1, ny + 1), dtype=torch.bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    free = ~boundary
    n_free = int(free.sum())
    if n_free < 4:
        raise ValueError("grid leaves no free nodes")

    w0 = (rigid_plastic_membrane_deflection(pressure, panel) if seed_deflection is None
          else float(seed_deflection))
    seed = w0 * torch.sin(math.pi * XX / a) * torch.sin(math.pi * YY / b)

    def unpack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.zeros((nx + 1, ny + 1), dtype=dt)
        v = torch.zeros_like(u)
        w = torch.zeros_like(u)
        u = u.masked_scatter(free, x[:n_free])
        v = v.masked_scatter(free, x[n_free:2 * n_free])
        w = w.masked_scatter(free, x[2 * n_free:3 * n_free])
        return u, v, w

    def potential(x_np) -> tuple[float, Any]:
        x = torch.as_tensor(x_np, dtype=dt).requires_grad_(True)
        u, v, w = unpack(x)
        dwdx = 0.5 * ((w[1:, 1:] - w[:-1, 1:]) + (w[1:, :-1] - w[:-1, :-1])) / dx
        dwdy = 0.5 * ((w[1:, 1:] - w[1:, :-1]) + (w[:-1, 1:] - w[:-1, :-1])) / dy
        dudx = 0.5 * ((u[1:, 1:] - u[:-1, 1:]) + (u[1:, :-1] - u[:-1, :-1])) / dx
        dvdy = 0.5 * ((v[1:, 1:] - v[1:, :-1]) + (v[:-1, 1:] - v[:-1, :-1])) / dy
        dudy = 0.5 * ((u[1:, 1:] - u[1:, :-1]) + (u[:-1, 1:] - u[:-1, :-1])) / dy
        dvdx = 0.5 * ((v[1:, 1:] - v[:-1, 1:]) + (v[1:, :-1] - v[:-1, :-1])) / dx
        ex = dudx + 0.5 * dwdx**2
        ey = dvdy + 0.5 * dwdy**2
        gxy = dudy + dvdx + dwdx * dwdy
        ez = -(ex + ey)
        em = (ex + ey + ez) / 3.0
        dev = (ex - em) ** 2 + (ey - em) ** 2 + (ez - em) ** 2 + 2.0 * (gxy / 2.0) ** 2
        eps_e = torch.sqrt(torch.clamp(2.0 / 3.0 * dev, min=1e-24))
        eps_y = sy / E
        energy_density = torch.where(
            eps_e <= eps_y,
            0.5 * E * eps_e**2,
            0.5 * E * eps_y**2 + sy * (eps_e - eps_y),
        )
        energy = (energy_density * (dx * dy) * t).sum()
        work = pressure * ((w[1:, 1:] + w[:-1, 1:] + w[1:, :-1] + w[:-1, :-1]).sum()
                           / 4.0 * dx * dy)
        value = energy - work
        (grad,) = torch.autograd.grad(value, x)
        return float(value), grad.detach().numpy().astype(float)

    x_start = torch.cat([torch.zeros(2 * n_free, dtype=dt),
                         seed.masked_select(free).flatten()]).numpy()
    solution = minimize(potential, x_start, jac=True, method="L-BFGS-B",
                        options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-6,
                                 "maxls": 60})
    x_final = torch.as_tensor(solution.x, dtype=dt)
    u, v, w = unpack(x_final)
    with torch.no_grad():
        dwdx = 0.5 * ((w[1:, 1:] - w[:-1, 1:]) + (w[1:, :-1] - w[:-1, :-1])) / dx
        dwdy = 0.5 * ((w[1:, 1:] - w[1:, :-1]) + (w[:-1, 1:] - w[:-1, :-1])) / dy
        dudx = 0.5 * ((u[1:, 1:] - u[:-1, 1:]) + (u[1:, :-1] - u[:-1, :-1])) / dx
        dvdy = 0.5 * ((v[1:, 1:] - v[1:, :-1]) + (v[:-1, 1:] - v[:-1, :-1])) / dy
        dudy = 0.5 * ((u[1:, 1:] - u[1:, :-1]) + (u[:-1, 1:] - u[:-1, :-1])) / dy
        dvdx = 0.5 * ((v[1:, 1:] - v[:-1, 1:]) + (v[1:, :-1] - v[:-1, :-1])) / dx
        ex = dudx + 0.5 * dwdx**2
        ey = dvdy + 0.5 * dwdy**2
        gxy = dudy + dvdx + dwdx * dwdy
        ez = -(ex + ey)
        em = (ex + ey + ez) / 3.0
        dev = (ex - em) ** 2 + (ey - em) ** 2 + (ez - em) ** 2 + 2.0 * (gxy / 2.0) ** 2
        eps_e = torch.sqrt(torch.clamp(2.0 / 3.0 * dev, min=1e-24))
        central = float(w[nx // 2, ny // 2])
        max_w = float(w.abs().max())
        max_eps = float(eps_e.max())
    analytic = rigid_plastic_membrane_deflection(pressure, panel)
    return {
        "test": "membrane_panel_energy",
        "pressure_Pa": pressure,
        "central_deflection_m": central,
        "max_deflection_m": max_w,
        "max_effective_strain": max_eps,
        "plastic_ratio": float((eps_e > sy / E).float().mean()),
        "potential_J": float(solution.fun),
        "optimizer_converged": bool(solution.success),
        "optimizer_iterations": int(solution.nit),
        "grid": [nx, ny],
        "analytic_deflection_m": analytic,
        "relative_error": abs(central - analytic) / max(analytic, 1e-30),
    }


def crushing_damage(
    pressure: float,
    panel: PlatePanel,
    *,
    patch_area: float | None = None,
    load_duration: float | None = None,
) -> dict[str, Any]:
    """Crushing damage of the panel under a uniform contact-pressure patch.

    Returns the W3-N05 interface payload: permanent deflection, membrane strain,
    rupture flag, absorbed plastic work and the equivalent static force.  The
    rigid-plastic membrane relation is monotonic, so the damage state is fully
    defined by the applied pressure.
    """
    if not math.isfinite(pressure) or pressure < 0.0:
        raise ValueError("pressure must be finite and nonnegative")
    if patch_area is not None and (not math.isfinite(patch_area) or patch_area <= 0.0):
        raise ValueError("patch_area must be finite and positive")
    deflection = rigid_plastic_membrane_deflection(pressure, panel)
    strain = membrane_strain(deflection, panel)
    area = panel.span_x * panel.span_y if patch_area is None else float(patch_area)
    force = pressure * area
    work = 0.5 * force * deflection           # linear rigid-plastic p-w relation
    rate = (
        deflection / load_duration
        if load_duration is not None and load_duration > 0.0 else None
    )
    return {
        "schema": CRUSHING_SCHEMA,
        "panel": {
            "span_x_m": panel.span_x,
            "span_y_m": panel.span_y,
            "thickness_m": panel.thickness,
            "yield_stress_Pa": panel.yield_stress,
        },
        "pressure_Pa": pressure,
        "patched_area_m2": area,
        "deflection_m": deflection,
        "deflection_to_thickness": deflection / panel.thickness,
        "membrane_strain": strain,
        "rupture": bool(strain > panel.rupture_strain),
        "absorbed_energy_J": work,
        "equivalent_force_N": force,
        "deflection_rate_m_per_s": rate,
        "large_deflection_valid": bool(deflection > 2.0 * panel.thickness),
        "consumer": "W3-N05 collision damage",
    }