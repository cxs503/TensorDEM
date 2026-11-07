"""Frictional heat generation, heat partition and the Jaeger moving-source field.

A sliding ice-structure contact turns tangential work into heat at the
interface.  Three ingredients make that quantitative:

1. **Source strength.**  ``q'' = mu p v`` (W/m^2) for a mean contact pressure
   ``p`` and slip speed ``v`` (:func:`friction_heat_flux`).
2. **Partition.**  Heat flows into both bodies.  For two semi-infinite bodies in
   sliding contact the Blok solution gives
   ``gamma_i = sqrt((k rho c)_i) / (sqrt((k rho c)_1) + sqrt((k rho c)_2))``
   (:func:`blok_partition_coefficient`).  It follows from matching the surface
   temperatures of the two bodies, which for the *same* flux distribution over
   the *same* contact size is a statement about ``sqrt(k rho c)`` only.
3. **Temperature field.**  The Jaeger moving band source: a band of half width
   ``a`` carrying a uniform flux ``q`` moves at speed ``v`` over the surface of
   a semi-infinite body.  Rosenthal's equation for the steady field in the frame
   of the source is

       laplacian(T) = (v / (2 alpha)) dT/dx,
       T = Q / (2 pi k R) * exp(-v (R + x) / (2 alpha))   (point source Q)

   (the factor 2 over the infinite-medium Green's function is the
   insulated-surface image).  Integrating that kernel along the band, and over
   the infinite transverse direction (which collapses to a modified Bessel
   function), gives the surface temperature

       T(xi) = (q a / (pi k)) * int_{-1}^{1} exp(-V (xi_hat - u)) K0(V |xi_hat - u|) du,
       V = v a / (2 alpha),

   evaluated by :func:`jaeger_band_surface_temperature` with a
   log-singularity-aware substitution.  :func:`moving_band_spectral_solution`
   solves the *same* boundary value problem independently in Fourier space
   (``T_s(k) = q(k) / (k_m sqrt(k^2 + i v k / (2 alpha)))``), so agreement
   between the two is a verification of the integral, not a restatement of it.

Two analytic limits pin the quadrature down without any reference table:

* ``V >> 1``: the surface temperature grows along the band exactly like the
  transient response to a constant flux applied for the transit time ``L / v``,
  ``T_max ~ (2 q / k) sqrt(alpha (L/v) / pi)`` with ``L = 2a`` the band width
  (:func:`jaeger_high_speed_limit`).
* ``V << 1``: the steady state of a band of infinite transverse extent does not
  exist -- the temperature diverges logarithmically,
  ``T(0) ~ (q a / (pi k)) [-2 ln(V/2) - 2 gamma + 2]``
  (:func:`jaeger_low_speed_limit`).  That divergence is why the *transient*
  constant-flux solution
  (:func:`tensordem.interface_thermal.conduction.semi_infinite_surface_temperature`)
  is the friction-heat reference at low sliding speed.

Known limitations: no surface cooling / convection, no radiation, uniform flux
over the nominal contact, constant properties, single phase on each side.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import k0

__all__ = [
    "friction_heat_flux",
    "friction_power",
    "blok_partition_coefficient",
    "jaeger_band_surface_temperature",
    "jaeger_band_max_temperature",
    "jaeger_high_speed_limit",
    "jaeger_low_speed_limit",
    "moving_band_fvm_solution",
]

_EULER_GAMMA = 0.5772156649015329


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


# ---------------------------------------------------------------------------
# source strength and partition
# ---------------------------------------------------------------------------
def friction_heat_flux(mu: float, pressure_pa: float, velocity_m_s: float) -> float:
    """Frictional heat flux ``q'' = mu p v`` (W/m^2) at the interface."""
    mu = _finite("mu", mu, nonnegative=True)
    pressure = _finite("pressure_pa", pressure_pa, nonnegative=True)
    velocity = _finite("velocity_m_s", velocity_m_s, nonnegative=True)
    return mu * pressure * velocity


def friction_power(mu: float, normal_force_n: float, velocity_m_s: float) -> float:
    """Frictional power ``mu F_n v`` (W)."""
    mu = _finite("mu", mu, nonnegative=True)
    force = _finite("normal_force_n", normal_force_n, nonnegative=True)
    velocity = _finite("velocity_m_s", velocity_m_s, nonnegative=True)
    return mu * force * velocity


def blok_partition_coefficient(
    k1_w_m_k: float, rho1_kg_m3: float, cp1_j_kg_k: float,
    k2_w_m_k: float, rho2_kg_m3: float, cp2_j_kg_k: float,
) -> dict[str, float]:
    """Blok heat partition for two semi-infinite bodies in sliding contact.

    ``gamma_i = K_i / (K_1 + K_2)`` with the thermal effusivity
    ``K_i = sqrt(k rho c_p)``.  ``gamma_1 + gamma_2 = 1`` exactly and equal
    effusivities give ``gamma = 0.5``; both are asserted by the tests.  The
    partition is speed independent because the same flux distribution is
    assumed to enter both bodies over the same contact.
    """
    k1 = _finite("k1_w_m_k", k1_w_m_k, positive=True)
    rho1 = _finite("rho1_kg_m3", rho1_kg_m3, positive=True)
    cp1 = _finite("cp1_j_kg_k", cp1_j_kg_k, positive=True)
    k2 = _finite("k2_w_m_k", k2_w_m_k, positive=True)
    rho2 = _finite("rho2_kg_m3", rho2_kg_m3, positive=True)
    cp2 = _finite("cp2_j_kg_k", cp2_j_kg_k, positive=True)
    e1 = math.sqrt(k1 * rho1 * cp1)
    e2 = math.sqrt(k2 * rho2 * cp2)
    return {
        "effusivity_1": e1,
        "effusivity_2": e2,
        "partition_1": e1 / (e1 + e2),
        "partition_2": e2 / (e1 + e2),
    }


# ---------------------------------------------------------------------------
# Jaeger moving band source: exact integral
# ---------------------------------------------------------------------------
def _exp_k0(x: float, *, growing: bool) -> float:
    """``exp(+-x) K0(x)`` without overflow for large ``x``.

    For ``x > 50`` the asymptotic series
    ``K0(x) ~ sqrt(pi/(2x)) (1 - 1/(8x) + 9/(128 x^2) - 225/(3072 x^3))`` is
    used, so the growing branch stays ``O(x^-1/2)`` instead of overflowing.
    """
    if x <= 0.0:
        raise ValueError("x must be positive")
    if x < 50.0:
        return math.exp(x if growing else -x) * float(k0(x))
    series = (1.0 - 1.0 / (8.0 * x) + 9.0 / (128.0 * x ** 2)
              - 225.0 / (3072.0 * x ** 3))
    factor = math.sqrt(math.pi / (2.0 * x)) * series
    return factor if growing else factor * math.exp(-2.0 * x)


def _band_kernel_integral(xi_hat: float, V: float) -> float:
    """``int_{-1}^{1} exp(-V(xi-u)) K0(V|xi-u|) du`` via ``s = exp(t)``.

    The substitution removes the logarithmic singularity at ``u = xi``: on each
    side the integrand becomes ``exp(+-V e^t) K0(V e^t) e^t``, a smooth function
    of ``t`` decaying like ``|t| e^t`` as ``t -> -inf``.
    """
    xi_hat = max(min(_finite("xi_hat", xi_hat), 1.0), -1.0)
    V = _finite("V", V, positive=True)

    def integral(span: float, growing: bool) -> float:
        if span <= 0.0:
            return 0.0
        t_min = math.log(min(1.0 / V, 1.0) * 1e-13)
        t_max = math.log(span)

        def integrand(t: float) -> float:
            return _exp_k0(V * math.exp(t), growing=growing) * math.exp(t)

        value, _ = quad(integrand, t_min, t_max, epsabs=1e-15, epsrel=1e-12,
                        limit=200)
        return value

    return integral(xi_hat + 1.0, False) + integral(1.0 - xi_hat, True)


def jaeger_band_surface_temperature(
    xi: Sequence[float] | float, half_width_m: float, velocity_m_s: float,
    alpha_m2_s: float, k_w_m_k: float, flux_w_m2: float,
) -> np.ndarray:
    """Surface temperature of a moving band source (Jaeger integral).

    ``xi`` is the position along the band measured from its centre, in the band
    frame; the body is swept in ``-xi`` so the wake (and the hottest point) sits
    at the trailing edge ``xi = -a``.  Returns the temperature **rise** in K
    above the undisturbed body.
    """
    a = _finite("half_width_m", half_width_m, positive=True)
    v = _finite("velocity_m_s", velocity_m_s, positive=True)
    alpha = _finite("alpha_m2_s", alpha_m2_s, positive=True)
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    flux = _finite("flux_w_m2", flux_w_m2)
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    V = v * a / (2.0 * alpha)
    scale = flux * a / (math.pi * k)
    return np.array([scale * _band_kernel_integral(float(x) / a, V) for x in xi])


def jaeger_band_max_temperature(
    half_width_m: float, velocity_m_s: float, alpha_m2_s: float,
    k_w_m_k: float, flux_w_m2: float, samples: int = 401,
) -> dict[str, float]:
    """Peak surface temperature rise and its position (trailing edge)."""
    a = _finite("half_width_m", half_width_m, positive=True)
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    xi = np.linspace(-a, a, samples)
    profile = jaeger_band_surface_temperature(xi, half_width_m, velocity_m_s,
                                             alpha_m2_s, k_w_m_k, flux_w_m2)
    index = int(np.argmax(profile))
    return {
        "max_temperature_k": float(profile[index]),
        "position_m": float(xi[index]),
        "profile_x": xi,
        "profile_t": profile,
    }


def jaeger_high_speed_limit(half_width_m: float, velocity_m_s: float,
                            alpha_m2_s: float, k_w_m_k: float,
                            flux_w_m2: float) -> float:
    """High-Peclet asymptote ``T_max ~ (2q/k) sqrt(alpha (2a/v) / pi)``.

    Heuristic derivation (verified numerically): at large ``V`` the field along
    the band is indistinguishable from the transient semi-infinite response to a
    constant flux that has been applied for the transit time ``t = 2a / v``;
    substituting ``t`` in ``T = 2 q sqrt(alpha t / pi) / k`` gives the
    expression.  It therefore has no free constant.
    """
    a = _finite("half_width_m", half_width_m, positive=True)
    v = _finite("velocity_m_s", velocity_m_s, positive=True)
    alpha = _finite("alpha_m2_s", alpha_m2_s, positive=True)
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    flux = _finite("flux_w_m2", flux_w_m2)
    return 2.0 * flux / k * math.sqrt(alpha * (2.0 * a / v) / math.pi)


def jaeger_low_speed_limit(half_width_m: float, alpha_m2_s: float,
                           k_w_m_k: float, flux_w_m2: float,
                           velocity_m_s: float) -> float:
    """Low-Peclet asymptote ``T(0) ~ (q a/(pi k)) [-2 ln(V/2) - 2 gamma + 2]``.

    The band of infinite transverse extent has no steady state as ``v -> 0``;
    the logarithm is the two-dimensional signature of that divergence.
    """
    a = _finite("half_width_m", half_width_m, positive=True)
    alpha = _finite("alpha_m2_s", alpha_m2_s, positive=True)
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    flux = _finite("flux_w_m2", flux_w_m2)
    v = _finite("velocity_m_s", velocity_m_s, positive=True)
    V = v * a / (2.0 * alpha)
    bracket = -2.0 * math.log(V / 2.0) - 2.0 * _EULER_GAMMA + 2.0
    return flux * a / (math.pi * k) * bracket


# ---------------------------------------------------------------------------
# independent finite-volume solution of the same boundary value problem
# ---------------------------------------------------------------------------
def moving_band_fvm_solution(
    half_width_m: float, velocity_m_s: float, alpha_m2_s: float,
    k_w_m_k: float, flux_w_m2: float, *,
    domain_factor: float = 12.0, cells_x: int = 801, cells_z: int = 90,
    grading_z: float = 1.06, depth_m: float | None = None,
) -> dict[str, object]:
    """Steady Rosenthal field of a moving band source solved by finite volumes.

    The same boundary value problem as :func:`jaeger_band_surface_temperature`
    -- ``-u dT/dxi = alpha (d^2T/dxi^2 + d^2T/dz^2)`` with the material velocity
    ``u = -v`` in the frame of the source, the flux ``q`` prescribed on
    ``|xi| < a`` and the surface insulated elsewhere -- but discretised
    directly: first-order upwind convection, centred diffusion, a geometrically
    graded wall-normal grid and a sparse direct solve.  It shares no algebra
    with the Bessel integral (no Green's function, no singularity handling), so
    agreement between the two verifies the integral instead of restating it.

    The energy balance is written for the temperature field, i.e. divided by
    ``rho c_p = k / alpha``; the surface flux therefore enters as
    ``q alpha / (k dz_0)``.  Getting that factor wrong (or omitting the division
    by the surface cell width) mis-scales the whole field -- a trap this
    function walks through explicitly so the next reader does not have to.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    a = _finite("half_width_m", half_width_m, positive=True)
    v = _finite("velocity_m_s", velocity_m_s, positive=True)
    alpha = _finite("alpha_m2_s", alpha_m2_s, positive=True)
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    flux = _finite("flux_w_m2", flux_w_m2)
    domain_factor = _finite("domain_factor", domain_factor, positive=True)
    grading = _finite("grading_z", grading_z, positive=True)
    for name, value in (("cells_x", cells_x), ("cells_z", cells_z)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")
    half_length = domain_factor * a
    depth = a if depth_m is None else _finite("depth_m", depth_m, positive=True)

    xi = np.linspace(-half_length, half_length, cells_x)
    dx = xi[1] - xi[0]
    dz0 = depth * (grading - 1.0) / (grading ** cells_z - 1.0)
    z_face = np.concatenate(([0.0], np.cumsum(dz0 * grading ** np.arange(cells_z))))
    z = 0.5 * (z_face[:-1] + z_face[1:])
    dz_cell = np.diff(z_face)
    dz_dist = np.diff(z)

    nx, nz = cells_x, cells_z
    u = -v                     # material velocity in the frame of the source
    index = lambda i, j: i * nz + j
    rows, cols, vals = [], [], []
    rhs = np.zeros(nx * nz)

    def add(row, col, value):
        rows.append(row)
        cols.append(col)
        vals.append(value)

    for i in range(nx):
        for j in range(nz):
            row = index(i, j)
            # undisturbed inflow boundary: Dirichlet T = 0 (the far field).
            # It is the *upwind* end, where the upwind stencil has no neighbour;
            # skipping this row leaves a negative diagonal and the solve blows
            # up (measured: 1e16 K at the outflow end).
            if (u < 0.0 and i == nx - 1) or (u > 0.0 and i == 0):
                add(row, row, 1.0)
                continue
            centre = 0.0
            # --- convection u dT/dxi, first-order upwind ---
            if u > 0.0:
                centre += u / dx
                if i > 0:
                    add(row, index(i - 1, j), -u / dx)
            else:
                # u(T_{i+1} - T_i)/dx: diagonal -u/dx, forward neighbour +u/dx
                centre += -u / dx
                if i < nx - 1:
                    add(row, index(i + 1, j), u / dx)
            # --- xi diffusion ---
            if i > 0:
                add(row, index(i - 1, j), -alpha / dx ** 2)
                centre += alpha / dx ** 2
            if i < nx - 1:
                add(row, index(i + 1, j), -alpha / dx ** 2)
                centre += alpha / dx ** 2
            # --- z diffusion ---
            if j > 0:
                add(row, index(i, j - 1), -alpha / (dz_dist[j - 1] * dz_cell[j]))
                centre += alpha / (dz_dist[j - 1] * dz_cell[j])
            if j < nz - 1:
                add(row, index(i, j + 1), -alpha / (dz_dist[j] * dz_cell[j]))
                centre += alpha / (dz_dist[j] * dz_cell[j])
            add(row, row, centre)
            if j == 0 and abs(xi[i]) < a:
                rhs[row] += flux * alpha / (k * dz_cell[0])
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(nx * nz, nx * nz))
    solution = spla.spsolve(matrix, rhs).reshape(nx, nz)
    surface = solution[:, 0]
    return {
        "x": xi,
        "surface_temperature": surface,
        "max_temperature_k": float(surface.max()),
        "n_unknowns": int(nx * nz),
        "dx_m": dx,
        "dz_first_m": float(dz_cell[0]),
    }
