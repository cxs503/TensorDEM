"""Structural transfer: random pressure field -> plating hot-spot stress.

The structure is the ice-belt plating panel between frames, modelled as a
**simply supported rectangular plate** (Navier solution).  That choice is
deliberate: it is the one plate model whose modes, frequencies and static
response are available in closed form, so the same object provides both the
transfer used by the fatigue chain *and* an independent analytic reference for
it (``navier_static_*`` below, plus the classical coefficient
``w_max = 0.00406 q a^4 / D`` for a square plate, which is asserted in the
tests).  Nothing here is fitted.

Modal equations (mode ``mn``, ``phi = sin(m pi x/a) sin(n pi y/b)``)::

    M_mn q'' + C_mn q' + K_mn q = Q_mn(t),   M_mn = rho t a b / 4,
    K_mn = M_mn omega_mn^2,   omega_mn = pi^2 sqrt(D/(rho t)) ((m/a)^2 + (n/b)^2)

and the surface bending stress at a hot spot is

    sigma_x = E t / (2 (1 - nu^2)) * sum_mn q_mn [(m pi/a)^2 + nu (n pi/b)^2] phi_mn
    sigma_y = E t / (2 (1 - nu^2)) * sum_mn q_mn [(n pi/b)^2 + nu (m pi/a)^2] phi_mn

Time integration uses the exact piecewise-linear ramp recurrence of
:func:`tensordem.fatigue_chain.response.sdof_response`, vectorised over modes
and over random-field realisations (the tests assert the two agree on a shared
case).  With a linear structural model the modal superposition is exact, so
"dynamic amplification" is computed, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "PlatePanel",
    "modal_forces",
    "modal_response_ramp",
    "hot_spot_stress",
    "stress_history_from_modal_force",
    "navier_static_uniform",
    "navier_static_point_load",
    "static_point_load_stress",
]


@dataclass(frozen=True)
class PlatePanel:
    """Simply supported plating panel (steel by default)."""

    length: float = 2.4          # a, m
    width: float = 0.6           # b, m  (frame spacing)
    thickness: float = 0.030     # t, m
    young: float = 2.06e11       # Pa
    poisson: float = 0.30
    density: float = 7850.0      # kg/m^3
    damping: float = 0.01
    n_modes_x: int = 6
    n_modes_y: int = 6

    def __post_init__(self) -> None:
        for name in ("length", "width", "thickness", "young", "density"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 <= self.poisson < 0.5):
            raise ValueError("poisson must be in [0, 0.5)")
        if not (0.0 <= self.damping < 1.0):
            raise ValueError("damping must be in [0, 1)")
        if self.n_modes_x < 1 or self.n_modes_y < 1:
            raise ValueError("need at least one mode per direction")

    # -- basic properties --------------------------------------------------
    @property
    def flexural_rigidity(self) -> float:
        return self.young * self.thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    @property
    def areal_density(self) -> float:
        return self.density * self.thickness

    @property
    def modes(self) -> List[Tuple[int, int]]:
        return [(m, n)
                for m in range(1, self.n_modes_x + 1)
                for n in range(1, self.n_modes_y + 1)]

    def omega(self, m: int, n: int) -> float:
        return math.pi ** 2 * math.sqrt(self.flexural_rigidity / self.areal_density) * \
            ((m / self.length) ** 2 + (n / self.width) ** 2)

    def omega_all(self) -> np.ndarray:
        return np.array([self.omega(m, n) for m, n in self.modes])

    def frequency_all(self) -> np.ndarray:
        return self.omega_all() / (2.0 * math.pi)

    def modal_mass(self) -> np.ndarray:
        return np.full(len(self.modes),
                       self.density * self.thickness * self.length * self.width / 4.0)

    def modal_stiffness(self) -> np.ndarray:
        return self.modal_mass() * self.omega_all() ** 2

    def mode_values(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Mode shapes evaluated at points, shape ``(n_modes, ...)``."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        out = np.empty((len(self.modes),) + np.broadcast(X, Y).shape)
        for k, (m, n) in enumerate(self.modes):
            out[k] = np.sin(m * math.pi * X / self.length) * \
                np.sin(n * math.pi * Y / self.width)
        return out

    def curvature_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-mode factors ``(m pi/a)^2`` and ``(n pi/b)^2``."""
        ax = np.array([(m * math.pi / self.length) ** 2 for m, _ in self.modes])
        ay = np.array([(n * math.pi / self.width) ** 2 for _, n in self.modes])
        return ax, ay


def modal_forces(
    panel: PlatePanel,
    pressure: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    """Modal force ``Q_mn = int p phi_mn dA`` for a stack of pressure fields.

    ``pressure`` has shape ``(..., ny, nx)``; returns ``(..., n_modes)``.
    """
    p = np.asarray(pressure, dtype=float)
    x = np.asarray(x_centers, dtype=float).ravel()
    y = np.asarray(y_centers, dtype=float).ravel()
    if p.shape[-2:] != (y.size, x.size):
        raise ValueError("pressure trailing shape must be (ny, nx)")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    phi = panel.mode_values(x[None, :], y[:, None]).reshape(len(panel.modes), -1)
    return (p.reshape(p.shape[:-2] + (-1,)) @ phi.T) * (dx * dy)


def modal_response_ramp(
    dt: float,
    force: np.ndarray,
    omega: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    """Exact ramp-recurrence response of many SDOF modes at once.

    ``force`` has shape ``(..., n_t)`` with the leading axes broadcast against
    ``omega``/``zeta`` (shape ``(M,)``).  Returns the response with the same
    shape; the response is normalised so that the static gain is 1 (``q`` is
    the static displacement ``Q/K``).

    This is the vectorised form of
    :func:`tensordem.fatigue_chain.response.sdof_response` (same recurrence,
    same initial condition ``q0 = F0``, ``v0 = 0``); agreement to solver
    precision is asserted in the tests.
    """
    f = np.asarray(force, dtype=float)
    if f.ndim < 2:
        raise ValueError("force must have at least 2 dimensions (..., n_t)")
    omega = np.asarray(omega, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    if omega.shape != zeta.shape:
        raise ValueError("omega and zeta must have the same shape")
    if f.shape[:-1][-1] != omega.size:
        raise ValueError("the mode axis of force must match omega")
    n_t = f.shape[-1]
    lead = f.shape[:-1]
    wd = omega * np.sqrt(1.0 - zeta ** 2)
    out = np.empty_like(f)
    out[..., 0] = f[..., 0]
    q = f[..., 0].copy()
    v = np.zeros(lead, dtype=float)
    h = float(dt)
    if h <= 0.0:
        raise ValueError("dt must be positive")
    e = np.exp(-zeta * omega * h)
    c = np.cos(wd * h)
    s = np.sin(wd * h)
    for i in range(1, n_t):
        f0 = f[..., i - 1]
        f1 = f[..., i]
        df = f1 - f0
        shift = 2.0 * zeta * df / (omega * h)
        qp0 = f0 - shift
        a = q - qp0
        b = (v - df / h + zeta * omega * a) / wd
        q = e * (a * c + b * s) + f1 - shift
        v = e * ((-zeta * omega * a + b * wd) * c
                 + (-zeta * omega * b - a * wd) * s) + df / h
        out[..., i] = q
    return out


def hot_spot_stress(
    panel: PlatePanel,
    q_history: np.ndarray,
    x_hs: float,
    y_hs: float,
) -> np.ndarray:
    """Surface bending stresses ``(sigma_x, sigma_y)`` from modal amplitudes.

    ``q_history`` shape ``(..., n_modes, n_t)`` (modal displacement, metres).
    """
    q = np.asarray(q_history, dtype=float)
    if q.shape[-2] != len(panel.modes):
        raise ValueError("second-to-last axis must be the mode axis")
    phi_hs = panel.mode_values(np.array([x_hs]), np.array([y_hs]))[:, 0]
    ax, ay = panel.curvature_basis()
    coef = panel.young * panel.thickness / (2.0 * (1.0 - panel.poisson ** 2))
    sx = coef * np.tensordot(q, phi_hs * (ax + panel.poisson * ay), axes=([-2], [0]))
    sy = coef * np.tensordot(q, phi_hs * (ay + panel.poisson * ax), axes=([-2], [0]))
    return np.stack([sx, sy], axis=-2)


def stress_history_from_modal_force(
    panel: PlatePanel,
    force_modal: np.ndarray,
    dt: float,
    x_hs: float,
    y_hs: float,
    *,
    dynamic: bool = True,
) -> np.ndarray:
    """Modal forces ``(..., n_modes, n_t)`` -> hot-spot ``sigma_x`` history."""
    f = np.asarray(force_modal, dtype=float)
    k = panel.modal_stiffness()
    f_static = f / k.reshape((1,) * (f.ndim - 2) + (k.size, 1))
    if dynamic:
        q = modal_response_ramp(dt, f_static, panel.omega_all(),
                                np.full(len(panel.modes), panel.damping))
    else:
        q = f_static
    return hot_spot_stress(panel, q, x_hs, y_hs)[..., 0, :]


# ---------------------------------------------------------------------------
# analytic references (independent of the modal machinery above)
# ---------------------------------------------------------------------------

def navier_static_uniform(panel: PlatePanel, q0: float,
                          n_terms: int = 99) -> Dict[str, float]:
    """Navier double series for a uniformly loaded, simply supported plate."""
    a, b, D = panel.length, panel.width, panel.flexural_rigidity
    m = np.arange(1, n_terms + 1, 2)
    n = np.arange(1, n_terms + 1, 2)
    M, N = np.meshgrid(m, n, indexing="ij")
    denom = M * N * ((M / a) ** 2 + (N / b) ** 2) ** 2
    coef = 16.0 * q0 / (math.pi ** 6 * D) / denom
    # deflection at the centre (all odd modes +1 there)
    sgn = np.where(((M - 1) // 2 + (N - 1) // 2) % 2 == 0, 1.0, -1.0)
    w_max = float(np.sum(coef * sgn))
    # curvature at the centre:  w,xx = -sum coef (m pi/a)^2 sin sin
    kx = np.sum(coef * (-(M * math.pi / a) ** 2) * sgn)
    ky = np.sum(coef * (-(N * math.pi / b) ** 2) * sgn)
    z = panel.thickness / 2.0
    sig_x = float(-panel.young * z / (1.0 - panel.poisson ** 2) * (kx + panel.poisson * ky))
    sig_y = float(-panel.young * z / (1.0 - panel.poisson ** 2) * (ky + panel.poisson * kx))
    return {
        "w_max": w_max,
        "sigma_x": sig_x,
        "sigma_y": sig_y,
        "alpha": w_max / (q0 * a ** 4 / D),
        "w_max_analytic": 0.00406 * q0 * a ** 4 / D if abs(a - b) < 1e-9 else float("nan"),
    }


def navier_static_point_load(panel: PlatePanel, load: float, x0: float, y0: float,
                             n_terms: int = 99) -> Dict[str, float]:
    """Navier double series for a point load ``P`` at ``(x0, y0)``."""
    a, b, D = panel.length, panel.width, panel.flexural_rigidity
    m = np.arange(1, n_terms + 1)
    n = np.arange(1, n_terms + 1)
    M, N = np.meshgrid(m, n, indexing="ij")
    sm = np.sin(M * math.pi * x0 / a)
    sn = np.sin(N * math.pi * y0 / b)
    coef = 4.0 * load / (math.pi ** 4 * D * a * b) * sm * sn / ((M / a) ** 2 + (N / b) ** 2) ** 2
    # deflection at the load point
    w = float(np.sum(coef * sm * sn))
    kx = float(np.sum(coef * (-(M * math.pi / a) ** 2) * sm * sn))
    ky = float(np.sum(coef * (-(N * math.pi / b) ** 2) * sm * sn))
    z = panel.thickness / 2.0
    return {
        "w": w,
        "kappa_x": kx,
        "kappa_y": ky,
        "sigma_x": float(-panel.young * z / (1.0 - panel.poisson ** 2)
                         * (kx + panel.poisson * ky)),
        "sigma_y": float(-panel.young * z / (1.0 - panel.poisson ** 2)
                         * (ky + panel.poisson * kx)),
    }


def static_point_load_stress(panel: PlatePanel, x0: float, y0: float,
                             x_hs: float, y_hs: float, n_terms: int = 99) -> float:
    """Static stress per unit point load at the hot spot, Pa/N (x component).

    Used by the analytic Miner anchor: the stress history of a point-load record
    is this scalar times the load history, so the analytic damage can be summed
    by hand.
    """
    a, b, D = panel.length, panel.width, panel.flexural_rigidity
    m = np.arange(1, n_terms + 1)
    n = np.arange(1, n_terms + 1)
    M, N = np.meshgrid(m, n, indexing="ij")
    sm0 = np.sin(M * math.pi * x0 / a) * np.sin(N * math.pi * y0 / b)
    smh = np.sin(M * math.pi * x_hs / a) * np.sin(N * math.pi * y_hs / b)
    coef = 4.0 / (math.pi ** 4 * D * a * b) * sm0 * smh / ((M / a) ** 2 + (N / b) ** 2) ** 2
    kx = np.sum(coef * (-(M * math.pi / a) ** 2))
    ky = np.sum(coef * (-(N * math.pi / b) ** 2))
    z = panel.thickness / 2.0
    return float(-panel.young * z / (1.0 - panel.poisson ** 2)
                 * (kx + panel.poisson * ky))