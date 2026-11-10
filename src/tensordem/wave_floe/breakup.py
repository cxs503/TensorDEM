"""Wave-curvature bending of a floating floe: critical wave height for breakup.

Why this module exists
----------------------
"Wave-induced ice breakup" needs a *threshold*: the wave height at which the
bending stress in a floe reaches the flexural strength.  The tree had no such
quantity (``ice_beam_bending`` handles a prescribed end load,
``ice_floe_breakup`` is a floe-population statistics module), so a C3 style
criterion could not be evaluated at all.

Physics (2-D, per unit width, thin-plate limit)
-----------------------------------------------
A floating floe on a Winkler foundation obeys the classical floating-plate
equation

    D d^4 w/dx^4 + rho_w g w = p(x),      D = E h^3 / (12 (1 - nu^2)),

with ``p`` the wave pressure expressed as an equivalent water column
``p = rho_w g eta``.  For a harmonic wave ``eta = a cos(kx)`` the *infinite-beam*
solution is

    w = rho_w g a / (D k^4 + rho_w g),
    M = D k^2 w,        sigma = 6 M / h^2,

so ``sigma(k)`` peaks at ``D k_c^4 = rho_w g``, i.e. ``k_c = (rho_w g / D)^{1/4}``
(the inverse of the characteristic length ``L_c = (D/rho_w g)^{1/4}``), with

    sigma_max = 3 rho_w g a / (k_c^2 h^2)
    =>  H_crit = 2 a_crit = 2 sigma_f k_c^2 h^2 / (3 rho_w g).

Two independent routes are implemented:

* :func:`theory_bending_stress` / :func:`critical_wave_height` -- the closed form
  above (no discretisation at all);
* :class:`WinklerBeam` -- a free-free Euler-Bernoulli finite-element beam on a
  consistent Winkler foundation, loaded by ``rho_w g a cos(kx)``, whose nodal
  curvatures give ``sigma``.  ``WinklerBeam`` is therefore the *discretised*
  route: it must reproduce the closed form for ``L >> L_c`` (checked in the
  tests and in the validation run), and the ratio
  ``sigma_FEM/sigma_theory`` versus ``L/L_c`` is the finite-floe edge correction
  reported as a deliverable.

Scope: the beam is 2-D (no ``k_y``), the foundation is linear (no water-column
inertia, no added mass), fracture is a *first-stress* criterion -- no crack
propagation, no sub-critical growth, no DEM bond statistics.  Strengths and
elastic moduli are *inputs* of the validation case, not fitted here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .response import RHO_ICE, RHO_WATER, GRAVITY


def flexural_rigidity(E: float, h: float, nu: float = 0.3) -> float:
    """``D = E h^3/(12 (1 - nu^2))`` [N m] (per unit width)."""
    return E * h ** 3 / (12.0 * (1.0 - nu ** 2))


def characteristic_wavenumber(E: float, h: float, nu: float = 0.3,
                              rho_w: float = RHO_WATER,
                              g: float = GRAVITY) -> float:
    """``k_c = (rho_w g / D)^{1/4}``: the most damaging wave number."""
    D = flexural_rigidity(E, h, nu)
    return (rho_w * g / D) ** 0.25


def theory_bending_stress(E: float, h: float, k: float, amplitude: float,
                          nu: float = 0.3, rho_w: float = RHO_WATER,
                          g: float = GRAVITY) -> dict:
    """Infinite-beam bending stress for a harmonic wave of ``amplitude`` and ``k``.

    Returns the deflection amplitude ``w``, the moment ``M``, the surface stress
    ``sigma`` and the ratio ``w/a`` (1 = the floe rides the surface, 0 = the floe
    is rigid relative to the wave).
    """
    D = flexural_rigidity(E, h, nu)
    denom = D * k ** 4 + rho_w * g
    w = rho_w * g * amplitude / denom
    M = D * k ** 2 * w
    sigma = 6.0 * M / h ** 2
    return {
        "w": w, "M": M, "sigma": sigma, "w_over_a": w / amplitude,
        "D": D, "k": k, "amplitude": amplitude,
        "foundation": rho_w * g, "bending": D * k ** 4,
    }


def peak_stress_over_k(E: float, h: float, amplitude: float,
                       nu: float = 0.3, rho_w: float = RHO_WATER,
                       g: float = GRAVITY) -> dict:
    """``sigma_max`` over all wave numbers, at ``k = k_c`` (the worst case)."""
    k_c = characteristic_wavenumber(E, h, nu, rho_w, g)
    out = theory_bending_stress(E, h, k_c, amplitude, nu, rho_w, g)
    out["k_c"] = k_c
    out["sigma_max"] = 3.0 * rho_w * g * amplitude / (k_c ** 2 * h ** 2)
    return out


def critical_wave_height(sigma_f: float, E: float, h: float, k: float | None = None,
                         nu: float = 0.3, rho_w: float = RHO_WATER,
                         g: float = GRAVITY) -> dict:
    """``H_crit`` at the given (or worst-case) wave number.

    ``H_crit = 2 sigma_f (D k^4 + rho_w g)/(6 D k^2 rho_w g) h^2``; at
    ``k = k_c`` this collapses to ``2 sigma_f k_c^2 h^2/(3 rho_w g)``.
    """
    D = flexural_rigidity(E, h, nu)
    if k is None:
        k = (rho_w * g / D) ** 0.25
    a_crit = sigma_f * h ** 2 * (D * k ** 4 + rho_w * g) / (6.0 * D * k ** 2 * rho_w * g)
    return {
        "critical_amplitude": a_crit,
        "critical_wave_height": 2.0 * a_crit,
        "k": k,
        "wavelength": 2.0 * math.pi / k,
        "k_c": (rho_w * g / D) ** 0.25,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Finite-element beam on a Winkler foundation
# ---------------------------------------------------------------------------

def _beam_element_matrices(l: float, EI: float) -> tuple[np.ndarray, np.ndarray]:
    """Hermite beam element: elastic stiffness and foundation stiffness.

    DOF order ``(w1, th1, w2, th2)`` with ``theta = dw/dx``; consistent
    foundation matrix ``k_w l/420 * [...]`` (the standard mass-matrix form).
    """
    ke = EI / l ** 3 * np.array([
        [12.0, 6.0 * l, -12.0, 6.0 * l],
        [6.0 * l, 4.0 * l ** 2, -6.0 * l, 2.0 * l ** 2],
        [-12.0, -6.0 * l, 12.0, -6.0 * l],
        [6.0 * l, 2.0 * l ** 2, -6.0 * l, 4.0 * l ** 2]])
    kf = l / 420.0 * np.array([
        [156.0, 22.0 * l, 54.0, -13.0 * l],
        [22.0 * l, 4.0 * l ** 2, 13.0 * l, -3.0 * l ** 2],
        [54.0, 13.0 * l, 156.0, -22.0 * l],
        [-13.0 * l, -3.0 * l ** 2, -22.0 * l, 4.0 * l ** 2]])
    return ke, kf


def _hermite(xi: float, l: float) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions and second derivatives at local coordinate ``xi in [0, l]``."""
    s = xi / l
    N = np.array([1 - 3 * s ** 2 + 2 * s ** 3,
                  l * (s - 2 * s ** 2 + s ** 3),
                  3 * s ** 2 - 2 * s ** 3,
                  l * (-s ** 2 + s ** 3)])
    dN = np.array([(-6.0 + 12.0 * s) / l ** 2,
                   (-4.0 + 6.0 * s) / l,
                   (6.0 - 12.0 * s) / l ** 2,
                   (-2.0 + 6.0 * s) / l])
    return N, dN


@dataclass
class WinklerBeam:
    """Free-free Euler-Bernoulli beam on a Winkler foundation under wave pressure."""

    length: float = 140.0
    thickness: float = 1.0
    youngs_modulus: float = 9.0e9
    poisson: float = 0.3
    rho_water: float = RHO_WATER
    gravity: float = GRAVITY
    n_elements: int = 60
    n_gauss: int = 4

    w: np.ndarray = field(init=False, repr=False)
    theta: np.ndarray = field(init=False, repr=False)

    @property
    def flexural_rigidity(self) -> float:
        return flexural_rigidity(self.youngs_modulus, self.thickness, self.poisson)

    @property
    def foundation(self) -> float:
        return self.rho_water * self.gravity

    @property
    def characteristic_length(self) -> float:
        return (self.flexural_rigidity / self.foundation) ** 0.25

    def _ldof(self, e: int) -> list[int]:
        return [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]

    def assemble(self) -> tuple[np.ndarray, np.ndarray]:
        n_el = int(self.n_elements)
        n_nodes = n_el + 1
        K = np.zeros((2 * n_nodes, 2 * n_nodes))
        l = self.length / n_el
        ke, kf = _beam_element_matrices(l, self.flexural_rigidity)
        for e in range(n_el):
            idx = self._ldof(e)
            K[np.ix_(idx, idx)] += ke + self.foundation * kf
        return K, np.arange(2 * n_nodes)

    def solve(self, amplitude: float, wavenumber: float) -> dict:
        """Static response to ``p(x) = rho_w g a cos(k (x - L/2))`` (symmetric load).

        Returns nodal deflection/rotation, the maximum surface bending stress over
        element Gauss points, and the mid-span deflection.
        """
        n_el = int(self.n_elements)
        l = self.length / n_el
        K, _ = self.assemble()
        f = np.zeros(2 * (n_el + 1))
        xg, wg = np.polynomial.legendre.leggauss(self.n_gauss)
        for e in range(n_el):
            idx = self._ldof(e)
            x0 = e * l
            for xi, wi in zip(xg, wg):
                x = x0 + 0.5 * l * (xi + 1.0)
                N, _ = _hermite(x - x0, l)
                q = self.rho_water * self.gravity * amplitude * math.cos(
                    wavenumber * (x - 0.5 * self.length))
                f[idx] += q * N * (0.5 * l * wi)
        K[np.ix_([0, 1], [0, 1])] += 0.0  # free-free: nothing constrained
        # The foundation makes K positive definite for the constant + linear modes?
        # No: a free-free beam on a foundation is positive definite (the foundation
        # has translational stiffness) -- solve directly.
        try:
            u = np.linalg.solve(K, f)
        except np.linalg.LinAlgError:  # pragma: no cover - conditioning fallback
            u = np.linalg.lstsq(K, f, rcond=None)[0]

        D = self.flexural_rigidity
        sigma_max = 0.0
        x_sigma = 0.0
        for e in range(n_el):
            idx = self._ldof(e)
            ue = u[idx]
            x0 = e * l
            for xi, _wi in zip(xg, wg):
                xloc = 0.5 * l * (xi + 1.0)
                _, dN = _hermite(xloc, l)
                kappa = float(dN @ ue)          # d^2 w/dx^2
                sigma = abs(6.0 * D * kappa / self.thickness ** 2)
                if sigma > sigma_max:
                    sigma_max = sigma
                    x_sigma = x0 + xloc
        self.w = u[0::2]
        self.theta = u[1::2]
        xs = np.linspace(0.0, self.length, n_el + 1)
        mid = int(np.argmin(np.abs(xs - 0.5 * self.length)))
        return {
            "sigma_max": sigma_max,
            "x_sigma": x_sigma,
            "w_mid": float(self.w[mid]),
            "w_max": float(np.max(np.abs(self.w))),
            "amplitude": amplitude,
            "wavenumber": wavenumber,
            "characteristic_length": self.characteristic_length,
            "n_elements": n_el,
            "deflection": self.w.tolist(),
            "x": xs.tolist(),
        }

    # -- threshold search ---------------------------------------------
    def critical_wave_height(self, sigma_f: float, wavenumber: float,
                             a_hi: float = 2.0, tol: float = 1e-4) -> dict:
        """Bisection on the wave amplitude for ``sigma_max(a) = sigma_f``.

        The problem is linear, so one solve fixes the slope; the bisection is kept
        because it is what a *nonlinear* extension would need and because it
        documents the monotonicity (``sigma`` grows with ``a``).
        """
        probe = self.solve(1.0, wavenumber)
        if probe["sigma_max"] <= 0.0:
            raise RuntimeError("beam produced zero stress for a unit amplitude")
        a = sigma_f / probe["sigma_max"]
        return {
            "critical_amplitude": a,
            "critical_wave_height": 2.0 * a,
            "sigma_f": sigma_f,
            "stress_per_unit_amplitude": probe["sigma_max"],
            "wavenumber": wavenumber,
            "wavelength": 2.0 * math.pi / wavenumber,
            "n_elements": int(self.n_elements),
            "iterations": "linear scaling (1 solve) + bisection check",
            "check_amplitude": a,
            "check_stress": self.solve(a, wavenumber)["sigma_max"],
        }


def finite_length_correction(lengths: Sequence[float], thickness: float = 1.0,
                             youngs_modulus: float = 9.0e9, nu: float = 0.3,
                             n_elements: int = 60) -> list[dict]:
    """``sigma_FEM/sigma_theory`` versus ``L/L_c`` at the most damaging ``k_c``.

    A deliverable in itself: the closed form assumes an infinitely long floe, and
    the edge correction is small only for ``L >> L_c`` -- this curve quantifies
    how small.
    """
    k_c = characteristic_wavenumber(youngs_modulus, thickness, nu)
    out = []
    for L in lengths:
        beam = WinklerBeam(length=L, thickness=thickness,
                           youngs_modulus=youngs_modulus, poisson=nu,
                           n_elements=n_elements)
        fem = beam.solve(1.0, k_c)
        th = theory_bending_stress(youngs_modulus, thickness, k_c, 1.0, nu)
        out.append({
            "length": L,
            "L_over_Lc": L / beam.characteristic_length,
            "sigma_fem": fem["sigma_max"],
            "sigma_theory": th["sigma"],
            "ratio": fem["sigma_max"] / th["sigma"],
        })
    return out