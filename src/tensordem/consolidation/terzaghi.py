"""Terzaghi one-dimensional consolidation: the analytic anchor of the chain.

Governing equation (excess pore pressure ``u``, constant total stress after a
step surcharge ``d_sigma``)::

    du/dt = C_v d2u/dz2 ,     C_v = k E_oed / gamma_w

with ``k`` the hydraulic conductivity (m/s), ``E_oed`` the oedometer
constrained modulus (Pa) and ``gamma_w`` the unit weight of water (N/m^3).
The drainage path is ``H_dr`` (``H`` for one-way drainage, ``H/2`` for
two-way drainage) and the time factor is ``T_v = C_v t / H_dr^2``.

The classical Fourier solution for a uniform initial excess pressure
``u(z,0) = d_sigma`` on ``0 <= z <= H_dr`` with a drained face at ``z = 0`` and
an impermeable face at ``z = H_dr`` is::

    u(z,T_v)/d_sigma = sum_m (2/M) sin(M z/H_dr) exp(-M^2 T_v),  M=(2m+1)pi/2
    U(T_v) = 1 - sum_m (2/M^2) exp(-M^2 T_v)

and the classical small-time branch ``U ~ sqrt(4 T_v / pi)`` is accurate to
about one per cent up to ``U = 0.53`` (``T_v ~ 0.22``), which is exactly the
"``U < 53 %`` use the square-root form" convention of the campaign node.

Nothing in this module knows about DEM: it is the independent reference every
other layer of the chain is judged against.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "TV_AT_U50",
    "TV_AT_U90",
    "TERMS",
    "U_sqrt_approx",
    "U_series",
    "U_of_Tv",
    "U_two_branch",
    "Tv_of_U",
    "u_profile",
    "r_u_profile",
    "settlement_series",
]

#: Tabulated Terzaghi time factors (Das, *Advanced Soil Mechanics*; Terzaghi
#: 1943): ``T_v = 0.197`` at ``U = 50 %`` and ``T_v = 0.848`` at ``U = 90 %``.
TV_AT_U50 = 0.197
TV_AT_U90 = 0.848

#: Time factor at which the small-time branch is switched off (U = 0.53).
TV_SWITCH = math.pi * 0.53 ** 2 / 4.0  # 0.2206

#: Default truncation of the Fourier series.
TERMS = 400


def _series_terms(Tv: np.ndarray, n_terms: int) -> np.ndarray:
    """``sum_m (2/M^2) exp(-M^2 T_v)`` for an array of time factors."""
    m = np.arange(n_terms, dtype=np.float64)
    M = 0.5 * math.pi * (2.0 * m + 1.0)
    # (n_Tv, n_terms) is small (n_terms=400) -- keep it simple and exact
    exponent = np.exp(-np.outer(np.asarray(Tv, dtype=np.float64), M * M))
    return exponent @ (2.0 / (M * M))


def U_series(Tv, n_terms: int = TERMS):
    """Degree of consolidation from the full Fourier series (scalar or array)."""
    Tv_arr = np.asarray(Tv, dtype=np.float64)
    if np.any(Tv_arr < 0.0):
        raise ValueError("time factor must be non-negative")
    U = 1.0 - _series_terms(Tv_arr.ravel(), n_terms)
    # the truncated series leaves O(1/n_terms) at T_v = 0; the exact value is 0
    U = np.where(Tv_arr.ravel() == 0.0, 0.0, np.clip(U, 0.0, 1.0))
    U = U.reshape(Tv_arr.shape)
    return float(U) if U.ndim == 0 else U


def U_sqrt_approx(Tv):
    """Small-time branch ``U = sqrt(4 T_v / pi)`` (exact to O(T_v^{3/2}))."""
    Tv_arr = np.asarray(Tv, dtype=np.float64)
    if np.any(Tv_arr < 0.0):
        raise ValueError("time factor must be non-negative")
    U = np.sqrt(4.0 * Tv_arr / math.pi)
    return float(U) if np.ndim(U) == 0 else U


def U_two_branch(Tv):
    """Small-time square root below ``U = 0.53``, Fourier series above.

    This is the two-branch rule used by the campaign node: cheap and
    monotonic, with a documented switch-over.  The switch is continuous to
    the accuracy of the square-root branch (see the tests, < 1 % there).
    """
    Tv_arr = np.asarray(Tv, dtype=np.float64)
    small = Tv_arr < TV_SWITCH
    out = np.empty_like(Tv_arr, dtype=np.float64)
    if np.any(small):
        out[small] = U_sqrt_approx(Tv_arr[small])
    if np.any(~small):
        out[~small] = U_series(Tv_arr[~small])
    return float(out) if out.ndim == 0 else out


def U_of_Tv(Tv, *, method: str = "series"):
    """Degree of consolidation -- ``"series"`` (default) or ``"two_branch"``."""
    if method == "series":
        return U_series(Tv)
    if method == "two_branch":
        return U_two_branch(Tv)
    raise ValueError(f"unknown method {method!r}")


def Tv_of_U(U, *, tol: float = 1e-12, max_iter: int = 200):
    """Invert ``U(T_v)`` (bisection on the monotone series).

    Used to tabulate the classical anchors and to convert a measured
    ``t_50``/``t_90`` into ``C_v``.  Accepts a scalar or an array.
    """
    U_arr = np.asarray(U, dtype=np.float64)
    if np.any(U_arr <= 0.0) or np.any(U_arr >= 1.0):
        raise ValueError("U must lie strictly inside (0, 1)")
    flat = U_arr.ravel()
    lo = np.zeros_like(flat)
    hi = np.full_like(flat, 10.0)  # U(10) = 1 - O(1e-11)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = U_series(mid)
        above = val < flat
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
        if np.max(hi - lo) < tol:
            break
    out = (0.5 * (lo + hi)).reshape(U_arr.shape)
    return float(out) if out.ndim == 0 else out


def u_profile(z, Tv, *, n_terms: int = TERMS):
    """``u(z, T_v)/d_sigma`` on ``0 <= z <= H_dr`` (depth from the drained face)."""
    z_arr = np.asarray(z, dtype=np.float64)
    Tv_arr = np.asarray(Tv, dtype=np.float64)
    if np.any(z_arr < 0.0) or np.any(z_arr > 1.0):
        raise ValueError("z must be normalised by the drainage path H_dr")
    m = np.arange(n_terms, dtype=np.float64)
    M = 0.5 * math.pi * (2.0 * m + 1.0)
    zz = z_arr[..., None]
    decay = np.exp(-np.outer(np.atleast_1d(Tv_arr).ravel(), M * M))
    shape = np.sin(zz * M) * (2.0 / M)
    prof = np.einsum("...m,tm->t...", shape, decay)
    if np.ndim(Tv) == 0:
        prof = prof[0]
    return np.clip(prof, 0.0, 1.0)


def r_u_profile(z_over_hdr, Tv, *, n_terms: int = TERMS):
    """Excess pore-pressure ratio ``r_u = u/d_sigma`` at dimensionless depth."""
    return u_profile(z_over_hdr, Tv, n_terms=n_terms)


def settlement_series(Tv, s_final: float):
    """``s(t) = s_final * U(T_v)`` -- the analytic settlement-time curve."""
    return float(s_final) * U_series(Tv)