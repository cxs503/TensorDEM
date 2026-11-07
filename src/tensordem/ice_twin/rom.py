"""Reduced-order model of the hull girder for the ice-zone digital twin.

The hull girder is a uniform free-free Euler-Bernoulli beam.  Unlike the FEM
discretisation in :mod:`tensordem.whipping.modes`, the mode shapes here are the
*closed-form* free-free shapes

    phi_n(x) = cosh(b x) + cos(b x) - s_n (sinh(b x) + sin(b x)),
    s_n      = (cosh(b L) + cos(b L)) / (sinh(b L) + sin(b L)),

with ``b L`` the roots of ``cosh(b L) cos(b L) = 1`` and

    omega_n = (b_n L)^2 sqrt(EI / (rho A L^4)).

The shapes are mass-normalised (``int rho A phi_n^2 dx = 1``), so the modal
mass matrix is the identity and the reduced equations of motion are

    q_n'' + 2 zeta_n omega_n q_n' + omega_n^2 q_n = F_n(t) / m_n,  m_n = 1.

Wet (hydroelastic) modes are obtained by the standard added-mass scaling
``omega_wet,n = omega_dry,n / sqrt(1 + a_n)`` where ``a_n`` is the modal
added-mass ratio (from a 2-D section BEM or a 3-D relief estimate); the dry
model is recovered with ``a_n = 0``.

The ice load is a *distributed* pressure ``p(x, t) = f(t) g(x - x0)`` with a
half-sine contact patch ``g`` of width ``w`` normalised to unit integral, so
``f(t)`` is the total ice force in newtons and the modal forces are

    F_n(t) = f(t) gamma_n,   gamma_n = int rho A g(x) phi_n(x) dx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..whipping.response import ModalSolution, modal_response

__all__ = [
    "FREE_FREE_BETA_L",
    "FreeFreeBeamROM",
    "half_sine_patch",
]

# roots of cosh(beta L) cos(beta L) = 1 (free-free beam), n = 1..12
FREE_FREE_BETA_L = (
    4.730040744862704, 7.853204624095838, 10.995607838001671,
    14.137165491257464, 17.278759657399481, 20.420352241590418,
    23.561944901923449, 26.703537265486766, 29.845129756946657,
    32.986722348309575, 36.128315018242584, 39.269907671789195,
)


def half_sine_patch(x, x0: float, width: float) -> np.ndarray:
    """Unit-integral half-sine contact patch centred at ``x0``.

    ``g(x) = (2/w) sin(pi (x-x0)/w)`` for ``|x-x0| <= w/2`` and 0 elsewhere,
    so that ``int g dx = 1`` and ``f(t)`` in ``p = f g`` is the total force.
    """
    x = np.asarray(x, dtype=float)
    if width <= 0.0:
        raise ValueError("patch width must be positive")
    u = (x - x0) / width
    g = np.zeros_like(x)
    m = np.abs(u) <= 0.5
    g[m] = (2.0 / width) * np.sin(np.pi * u[m])
    return g


@dataclass(frozen=True)
class FreeFreeBeamROM:
    """Mass-normalised analytic free-free beam ROM (dry or wet)."""

    L: float
    EI: float
    rhoA: float
    n_modes: int = 8
    zeta: float = 0.01
    wet_scale: tuple = ()          # per-mode added-mass ratio a_n (0 = dry)
    _norm: tuple = field(default=(), compare=False, repr=False)

    def __post_init__(self):
        if not (self.L > 0.0 and self.EI > 0.0 and self.rhoA > 0.0):
            raise ValueError("L, EI, rhoA must be positive")
        if not (1 <= self.n_modes <= len(FREE_FREE_BETA_L)):
            raise ValueError(f"n_modes must lie in 1..{len(FREE_FREE_BETA_L)}")
        if not (0.0 <= self.zeta < 1.0):
            raise ValueError("zeta must lie in [0, 1)")
        a = np.zeros(self.n_modes)
        if self.wet_scale:
            a = np.asarray(self.wet_scale, dtype=float)
            if a.size != self.n_modes or np.any(a < 0.0):
                raise ValueError("wet_scale must be nonnegative, one per mode")
        object.__setattr__(self, "_a", a)
        # mass normalisation on a fine grid (exact to quadrature error)
        x = np.linspace(0.0, self.L, 4001)
        dx = x[1] - x[0]
        norms = []
        for n in range(self.n_modes):
            phi = self._raw_shape(x, n)
            norms.append(math.sqrt(self.rhoA * np.trapezoid(phi * phi, dx=dx)))
        object.__setattr__(self, "_norm", tuple(norms))

    # ------------------------------------------------------------------ #
    # mode shapes and derivatives (closed form)
    # ------------------------------------------------------------------ #
    def _raw_shape(self, x, n: int) -> np.ndarray:
        b = FREE_FREE_BETA_L[n] / self.L
        bx = b * np.asarray(x, dtype=float)
        bL = FREE_FREE_BETA_L[n]
        s = (math.cosh(bL) + math.cos(bL)) / (math.sinh(bL) + math.sin(bL))
        return (np.cosh(bx) + np.cos(bx) - s * (np.sinh(bx) + np.sin(bx)))

    def _raw_slope(self, x, n: int) -> np.ndarray:
        b = FREE_FREE_BETA_L[n] / self.L
        bx = b * np.asarray(x, dtype=float)
        bL = FREE_FREE_BETA_L[n]
        s = (math.cosh(bL) + math.cos(bL)) / (math.sinh(bL) + math.sin(bL))
        return b * (np.sinh(bx) - np.sin(bx) - s * (np.cosh(bx) + np.cos(bx)))

    def _raw_curvature(self, x, n: int) -> np.ndarray:
        b = FREE_FREE_BETA_L[n] / self.L
        bx = b * np.asarray(x, dtype=float)
        bL = FREE_FREE_BETA_L[n]
        s = (math.cosh(bL) + math.cos(bL)) / (math.sinh(bL) + math.sin(bL))
        return b**2 * (np.cosh(bx) - np.cos(bx) - s * (np.sinh(bx) - np.sin(bx)))

    def phi(self, x) -> np.ndarray:
        """Mass-normalised mode shapes, shape (n_modes, len(x))."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array([self._raw_shape(x, n) / self._norm[n]
                         for n in range(self.n_modes)])

    def phi_prime(self, x) -> np.ndarray:
        """Strain operator: d phi_n / dx, shape (n_modes, len(x))."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array([self._raw_slope(x, n) / self._norm[n]
                         for n in range(self.n_modes)])

    def phi_double_prime(self, x) -> np.ndarray:
        """Curvature operator: d2 phi_n / dx2, shape (n_modes, len(x))."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array([self._raw_curvature(x, n) / self._norm[n]
                         for n in range(self.n_modes)])

    # ------------------------------------------------------------------ #
    # modal operators
    # ------------------------------------------------------------------ #
    @property
    def omega_dry(self) -> np.ndarray:
        bl = np.asarray(FREE_FREE_BETA_L[: self.n_modes])
        return bl**2 * math.sqrt(self.EI / (self.rhoA * self.L**4))

    @property
    def omega(self) -> np.ndarray:
        return self.omega_dry / np.sqrt(1.0 + self._a)

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    @property
    def M(self) -> np.ndarray:
        return np.eye(self.n_modes)

    @property
    def K(self) -> np.ndarray:
        return np.diag(self.omega**2)

    @property
    def C(self) -> np.ndarray:
        z = np.full(self.n_modes, self.zeta)
        return np.diag(2.0 * z * self.omega)

    def strain_matrix(self, x_gauges) -> np.ndarray:
        """H[i, n] = phi_n'(x_i): strain at gauge i per unit modal coordinate."""
        return self.phi_prime(x_gauges).T

    def stress_weights(self, x_station: float, E: float, y_deck: float) -> np.ndarray:
        """w_n = E y_deck phi_n''(x_station): deck stress per unit q_n (Pa)."""
        return E * y_deck * self.phi_double_prime([x_station]).T.ravel()

    # ------------------------------------------------------------------ #
    # ice load <-> modal force
    # ------------------------------------------------------------------ #
    def patch_participation(self, x0: float, width: float,
                            n_grid: int = 2001) -> np.ndarray:
        """gamma_n = int rho A g(x) phi_n(x) dx (N per unit total force)."""
        x = np.linspace(0.0, self.L, n_grid)
        g = half_sine_patch(x, x0, width)
        return self.rhoA * np.trapezoid(g[:, None] * self.phi(x).T, x=x, axis=0)

    def modal_force(self, f, x0: float, width: float) -> np.ndarray:
        """Modal force F_n(t) = f(t) gamma_n for a patch load of total f(t)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        return self.patch_participation(x0, width)[:, None] * f[None, :]

    def forward(self, f, t, x0: float, width: float) -> ModalSolution:
        """Exact Duhamel response to the patch load (reuses the p29 integrator)."""
        t = np.asarray(t, dtype=float)
        F = self.modal_force(f, x0, width)
        return modal_response(t, F, self.omega, np.full(self.n_modes, self.zeta))

    def strain(self, q, x_gauges) -> np.ndarray:
        """Strain history at the gauges: eps_i(t) = sum_n H[i,n] q_n(t)."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        return self.strain_matrix(x_gauges) @ q

    def as_dict(self) -> dict:
        return {
            "L_m": self.L, "EI_Nm2": self.EI, "rhoA_kg_m": self.rhoA,
            "n_modes": self.n_modes, "zeta": self.zeta,
            "wet_scale": [float(v) for v in self._a],
            "omega_dry_rad_s": [float(v) for v in self.omega_dry],
            "omega_rad_s": [float(v) for v in self.omega],
            "freq_hz": [float(v) for v in self.freq_hz],
        }
