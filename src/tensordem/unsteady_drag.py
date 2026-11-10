"""Unsteady hydrodynamic force closure for a sphere in a viscous fluid.

The total force is written as a **separable four-term ledger** (Maxey & Riley
1983) so that each contribution can be pinned down by its own analytic anchor::

    F_total = F_quasi-steady  (Schiller-Naumann)
            + F_added-mass     (0.5 rho_f V d(u_f-u_p)/dt)
            + F_Basset         (history convolution, 1/sqrt(t) kernel)
            + F_pressure-grad  (rho_f V du_f/dt, Froude-Krylov)

Because the four terms have different frequency signatures they can be
*isolated* by controlled scenarios:

* steady uniform slip        -> only F_quasi-steady survives (negative control)
* sudden acceleration, Re->0 -> added mass dominates, A = F_am/(rho_f V du/dt) = 0.5
* oscillatory flow          -> history force dominates the phase lag

Analytic anchors
----------------
* Stokes quasi-steady drag ``6 pi mu a u`` (Schiller-Naumann at Re->0).
* Landau & Lifshitz, *Fluid Mechanics* section 24 (exact unsteady Stokes
  solution).  For a complex slip velocity ``W e^{i omega t}`` and
  ``lam = a sqrt(i omega/nu) = (1+i) alpha``, ``alpha = a sqrt(omega/(2 nu))``,
  the force **from slip motion** is::

      F_hat / (6 pi mu a W_hat) = 1 + lam + lam^2/9

  (quasi-steady + added mass + Basset), while for a **fixed sphere** in an
  oscillatory stream (add the pressure gradient / Froude-Krylov force
  ``rho_f V du_f/dt``)::

      F_hat / (6 pi mu a U_hat_f) = 1 + lam + lam^2/3

* Stokes impulsive start ``F(t) = 6 pi mu a U (1 + a/sqrt(pi nu t))`` - pins the
  Basset kernel prefactor ``6 a^2 sqrt(pi rho mu)`` (= ``1.5 d^2 sqrt(pi rho mu)``).
* Stokes second problem ``u_f(y,t) = U0 exp(-kappa y) cos(omega t - kappa y)``,
  ``kappa = sqrt(omega/(2 nu))`` - the local flow field fed to the particle.

Basset history force: O(N) recursion
------------------------------------
The reference discretisation is the piecewise-constant quadrature of
``integral_0^{t_n} a(tau)/sqrt(t_n - tau) dtau``::

    F_B[n] = 6 a^2 sqrt(pi rho mu) * sum_{j<=n} a_j w(n-j+1)
    w(m)   = 2 sqrt(dt) (sqrt(m) - sqrt(m-1))          (O(N^2) reference)

The O(N) scheme approximates the discrete kernel by a sum of exponentials
``w(m) ~ sum_k c_k rho_k^{m-1}``, giving one scalar state per term::

    S_k[n] = rho_k S_k[n-1] + a_n ,  F_B[n] = 6 a^2 sqrt(pi rho mu) sum_k c_k S_k[n]

``(c_k, rho_k)`` come from a *relative-error weighted* least squares fit of the
discrete kernel on a log-spaced relaxation-rate grid
``s_k in [1/(2 N dt), 10/dt]`` - the van Hinsberg-Winckelmans (2010) idea of
replacing 1/sqrt(t) by a Prony/exponential series.  Their reference scheme uses
three terms; here ``n_exp`` is selectable (default 16) because the three-term
fit cannot reach the 2 % full-history L2 target (measured in
``benchmarks/unsteady_drag_n11_validation.py``).

Sign conventions
----------------
``u_slip = u_f - u_p``; every force is positive along ``+u_slip`` (the force the
fluid exerts on the particle).  Phases are reported as **lag** angles in degrees
for the ``e^{i omega t}`` convention: ``lag = atan2(Im F_hat, Re F_hat)``.

References
----------
- Schiller & Naumann (1935); Stokes (1851); Basset (1888); Boussinesq (1885).
- Landau & Lifshitz, *Fluid Mechanics* section 24 (oscillating sphere).
- Maxey & Riley (1983), *Phys. Fluids* 26, 883 (particle equation of motion).
- van Hinsberg & Winckelmans (2010), *J. Comput. Phys.* 229, 4735 (O(N) Basset).
- Daitche (2013), *J. Comput. Phys.* 249, 243 (kernel-splitting accuracy).

All quantities are SI unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "schiller_naumann_drag",
    "added_mass_force",
    "pressure_gradient_force",
    "sphere_volume",
    "basset_direct_convolution",
    "basset_kernel_fit",
    "basset_kernel_relative_l2",
    "basset_recursive",
    "basset_recursive_scalar",
    "basset_recursive_blocked",
    "basset_recursive_naive",
    "BassetRecursionState",
    "benchmark_basset_methods",
    "unsteady_drag_force",
    "UnsteadyDragResult",
    "stokes_oscillatory_parameter",
    "landau_lifshitz_oscillatory_drag",
    "landau_lifshitz_oscillatory_drag_amplitude_phase",
    "stokes_impulsive_start_force",
    "stokes_second_problem_velocity",
    "run_sudden_acceleration",
    "run_steady_uniform",
    "run_ramp_then_hold",
    "run_impulsive_start",
    "run_oscillatory_flow_simulation",
    "run_free_particle_oscillatory",
    "stokes_disturbance_velocity",
    "multi_sphere_unsteady_drag",
    "run_dilute_limit_convergence",
]


def sphere_volume(d: float) -> float:
    """Volume of a sphere of diameter ``d`` [m^3]."""
    return math.pi * d ** 3 / 6.0


# ---------------------------------------------------------------------------
# 1. Quasi-steady drag (Schiller-Naumann)
# ---------------------------------------------------------------------------

def schiller_naumann_drag(
    mu: float,
    rho_f: float,
    d: float,
    u_slip: float,
) -> float:
    """Quasi-steady drag on a sphere, Schiller-Naumann correlation.

    ``F = 0.5 rho_f u|u| A Cd`` with ``Cd = 24/Re (1 + 0.15 Re^0.687)``; the
    ``Re -> 0`` limit is Stokes drag ``3 pi mu d u``.

    Parameters
    ----------
    mu : dynamic viscosity [Pa s]
    rho_f : fluid density [kg/m^3]
    d : sphere diameter [m]
    u_slip : signed slip velocity u_f - u_p [m/s]

    Returns
    -------
    F_qs : signed drag [N] (same sign as ``u_slip``)
    """
    if d <= 0.0 or mu <= 0.0:
        raise ValueError("mu and d must be positive")
    nu = mu / rho_f
    Re = abs(u_slip) * d / nu if nu > 0 else 0.0
    if Re < 1e-12:
        return 3.0 * math.pi * mu * d * u_slip
    Cd = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
    A_proj = math.pi * d ** 2 / 4.0
    return 0.5 * rho_f * u_slip * abs(u_slip) * A_proj * Cd


# ---------------------------------------------------------------------------
# 2. Added-mass and pressure-gradient (Froude-Krylov) forces
# ---------------------------------------------------------------------------

def added_mass_force(
    rho_f: float,
    d: float,
    du_slip_dt: float,
    coefficient: float = 0.5,
) -> float:
    """Added-mass force ``C_A rho_f V d(u_f-u_p)/dt`` with ``C_A = 0.5``.

    For a sphere in an unbounded inviscid fluid the potential-flow added-mass
    coefficient is 1/2 (Lamb 1932; Landau & Lifshitz section 11).  The
    high-frequency (``alpha -> inf``) asymptote of the exact unsteady Stokes
    solution recovers the same value through a different route - see
    :func:`landau_lifshitz_oscillatory_drag_amplitude_phase`.
    """
    return coefficient * rho_f * sphere_volume(d) * du_slip_dt


def pressure_gradient_force(rho_f: float, d: float, du_f_dt: float) -> float:
    """Froude-Krylov pressure-gradient force ``rho_f V du_f/dt`` [N]."""
    return rho_f * sphere_volume(d) * du_f_dt


# ---------------------------------------------------------------------------
# 3. Basset history force: O(N^2) reference and O(N) recursion
# ---------------------------------------------------------------------------

def _basset_prefactor(mu: float, rho_f: float, d: float) -> float:
    """``6 a^2 sqrt(pi rho_f mu)`` = ``1.5 d^2 sqrt(pi rho_f mu)`` [N s^0.5]."""
    a = d / 2.0
    return 6.0 * a ** 2 * math.sqrt(math.pi * rho_f * mu)


def _basset_kernel_weights(n: int, dt: float) -> np.ndarray:
    """Discrete 1/sqrt(t) kernel weights ``w(m) = 2 sqrt(dt)(sqrt(m)-sqrt(m-1))``."""
    m = np.arange(1.0, n + 1.0)
    return 2.0 * math.sqrt(dt) * (np.sqrt(m) - np.sqrt(m - 1.0))


def basset_direct_convolution(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
) -> np.ndarray:
    """O(N^2) reference: piecewise-constant quadrature of the Basset integral.

    ``F_B[n] = 6a^2 sqrt(pi rho mu) sum_{j<=n} a_j int_{(n-j)dt}^{(n-j+1)dt} tau^-0.5 dtau``

    The self interval (``j = n``, the singularity) is integrated exactly; that
    term is what makes this quadrature a faithful discretisation of the
    continuous convolution (dropping it biases amplitude and phase).

    Returns
    -------
    F_basset : array of shape (N,) [N]
    """
    accel = np.asarray(accel_history, dtype=float)
    n = accel.size
    out = np.zeros(n)
    if n < 2:
        return out
    w = _basset_kernel_weights(n, dt)
    coeff = _basset_prefactor(mu, rho_f, d)
    for i in range(1, n):
        out[i] = coeff * float(np.dot(accel[i::-1], w[: i + 1]))
    return out


def basset_kernel_fit(
    dt: float,
    n_steps: int,
    n_exp: int = 16,
    s_max_ratio: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``w(m) ~ sum_k c_k rho_k^{m-1}`` (discrete kernel, relative LSQ).

    Parameters
    ----------
    dt : time step [s]
    n_steps : history horizon in steps (kernel fitted over lags 1..n_steps)
    n_exp : number of exponential terms
    s_max_ratio : fastest relaxation rate in units of ``1/dt``

    Returns
    -------
    (c, rho) : coefficients ``c_k`` and per-step decay factors ``rho_k = exp(-s_k dt)``
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if n_exp < 1:
        raise ValueError("n_exp must be >= 1")
    m = np.arange(1.0, n_steps + 1.0)
    target = np.sqrt(m) - np.sqrt(m - 1.0)          # = w(m)/(2 sqrt(dt))
    s = np.geomspace(1.0 / (2.0 * n_steps * dt), s_max_ratio / dt, n_exp)
    design = np.exp(-np.outer((m - 1.0) * dt, s))    # rho_k^{m-1}
    design[design < 1.0e-290] = 0.0                  # avoid denormal SVD input
    sample = np.unique(np.concatenate([
        np.arange(0, min(200, n_steps)),
        np.geomspace(1, n_steps, min(600, n_steps)).astype(int) - 1,
    ]))
    weight = 1.0 / target[sample]
    A = design[sample] * weight[:, None]
    b = target[sample] * weight
    # normal equations with a tiny ridge: the design is wide, so a plain SVD
    # least squares gets expensive and ill-conditioned once n_exp > 16
    gram = A.T @ A
    ridge = 1.0e-12 * float(np.trace(gram)) + 1.0e-300
    c = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), A.T @ b)
    return c, np.exp(-s * dt)


def basset_kernel_relative_l2(dt: float, n_steps: int, n_exp: int = 16) -> float:
    """Relative L2 error of the exponential kernel fit over the whole history."""
    c, rho = _kernel_fit_cached(dt, int(n_steps), n_exp, 10.0)
    m = np.arange(1.0, n_steps + 1.0)
    target = np.sqrt(m) - np.sqrt(m - 1.0)
    s = -np.log(rho) / dt
    approx = np.exp(-np.outer((m - 1.0) * dt, s)) @ c
    return float(np.linalg.norm(approx - target) / np.linalg.norm(target))


_KERNEL_FIT_CACHE: Dict[Tuple[float, int, int, float], Tuple[np.ndarray, np.ndarray]] = {}


def _kernel_fit_cached(dt: float, n_steps: int, n_exp: int,
                       s_max_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Memoise :func:`basset_kernel_fit` (the fit is geometry-independent)."""
    key = (float(dt), int(n_steps), int(n_exp), float(s_max_ratio))
    hit = _KERNEL_FIT_CACHE.get(key)
    if hit is None:
        if len(_KERNEL_FIT_CACHE) > 256:
            _KERNEL_FIT_CACHE.clear()
        hit = basset_kernel_fit(dt, n_steps, n_exp=n_exp, s_max_ratio=s_max_ratio)
        _KERNEL_FIT_CACHE[key] = hit
    return hit


@dataclass
class BassetRecursionState:
    """State of the O(N) Basset recursion (``n_exp`` auxiliary variables)."""

    c: np.ndarray
    rho: np.ndarray
    state: np.ndarray
    coeff: float
    dt: float

    @classmethod
    def create(
        cls,
        mu: float,
        rho_f: float,
        d: float,
        dt: float,
        n_exp: int = 16,
        n_steps: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> "BassetRecursionState":
        """Create the recursion state; ``n_steps`` sets the memory horizon."""
        if n_steps is None:
            if t_max is None:
                t_max = max(10000.0 * dt, 10.0)
            n_steps = max(int(round(t_max / dt)), 2)
        c, rho = _kernel_fit_cached(dt, int(n_steps), n_exp, 10.0)
        return cls(c=c, rho=rho, state=np.zeros_like(c),
                   coeff=_basset_prefactor(mu, rho_f, d), dt=dt)

    def step(self, accel: float, dt: Optional[float] = None) -> float:
        """Advance one step with piecewise-constant ``accel``; return F_B."""
        if dt is not None and abs(dt - self.dt) > 1e-18:
            raise ValueError("dt mismatch with recursion state")
        self.state = self.state * self.rho + accel
        return self.coeff * 2.0 * math.sqrt(self.dt) * float(self.c @ self.state)

    def reset(self) -> None:
        self.state[:] = 0.0


def basset_recursive_scalar(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    n_exp: int = 16,
) -> np.ndarray:
    """O(N) Basset recursion, one scalar state per exponential term.

    This is the online form (same arithmetic as
    :class:`BassetRecursionState`), used as the reference for the blocked
    implementation.
    """
    accel = np.asarray(accel_history, dtype=float)
    n = accel.size
    out = np.zeros(n)
    if n < 2:
        return out
    state = BassetRecursionState.create(mu, rho_f, d, dt, n_exp=n_exp, n_steps=n)
    c = state.c
    rho = state.rho
    scale = state.coeff * 2.0 * math.sqrt(dt)
    s = state.state
    for i in range(n):
        s *= rho
        s += accel[i]
        if i > 0:
            out[i] = scale * float(c.dot(s))
    return out


def basset_recursive_blocked(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    n_exp: int = 16,
    block: int = 256,
) -> np.ndarray:
    """Same recursion evaluated block-wise (BLAS matmuls instead of Python loop).

    Inside a block of ``B`` steps the history is split into

    * the *tail* of older steps, carried by the ``n_exp`` state variables
      (``sum_k c_k rho_k^{r+1} S_k`` - an ``(B, n_exp) @ (n_exp,)`` matmul),
    * the *intra-block* part, the convolution with the local kernel
      ``k(lag) = sum_k c_k rho_k^{lag}`` (``np.convolve``, ``O(B^2)`` C-loop).

    Per-step cost is ``O(B + n_exp)`` compiled operations instead of a
    Python-level state update; results are identical to
    :func:`basset_recursive_scalar` up to round-off.  ``np.convolve`` is used
    instead of a BLAS triangular matvec on purpose: a 256x256 ``@`` on a
    many-core BLAS costs ~70 ms per call through thread-pool overhead, i.e.
    three orders of magnitude more than the convolve it would replace.
    """
    accel = np.asarray(accel_history, dtype=float)
    n = accel.size
    if n < 2:
        return np.zeros(n)
    state = BassetRecursionState.create(mu, rho_f, d, dt, n_exp=n_exp, n_steps=n)
    c, rho = state.c, state.rho
    block = int(min(block, max(n, 1)))
    n_blk = -(-n // block)
    padded = np.zeros(n_blk * block)
    padded[:n] = accel

    rho_pow = rho[None, :] ** np.arange(block)[:, None]     # (B, n_exp): rho^r
    rho_pow[rho_pow < 1.0e-290] = 0.0        # flush underflow: denormals are ~100x slower
    local_kernel = rho_pow @ c                               # k(lag) = sum_k c_k rho_k^lag
    tail_matrix = rho_pow * (c * rho)                        # (B, n_exp)
    state_matrix = rho_pow[block - 1::-1].T                  # (n_exp, B)
    rho_b = rho ** block
    scale = state.coeff * 2.0 * math.sqrt(dt)

    out = np.zeros(n_blk * block)
    s = np.zeros_like(c)
    for p in range(n_blk):
        sl = slice(p * block, (p + 1) * block)
        blk = padded[sl]
        intra = np.convolve(blk, local_kernel)[:block]
        out[sl] = scale * (tail_matrix @ s + intra)
        s = rho_b * s + state_matrix @ blk
    out = out[:n]
    out[0] = 0.0
    return out


def basset_recursive_naive(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
) -> np.ndarray:
    """Textbook O(N^2) double loop (timing baseline for the speedup claim).

    Not used for physics - only to measure what the recursion buys over a naive
    implementation of the same quadrature.
    """
    accel = np.asarray(accel_history, dtype=float)
    n = accel.size
    out = np.zeros(n)
    coeff = _basset_prefactor(mu, rho_f, d)
    two_sqrt_dt = 2.0 * math.sqrt(dt)
    for i in range(1, n):
        acc = 0.0
        for j in range(i + 1):
            m = i - j + 1
            acc += accel[j] * (math.sqrt(m) - math.sqrt(m - 1))
        out[i] = coeff * two_sqrt_dt * acc
    return out


def basset_recursive(
    accel_history: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    n_exp: int = 16,
) -> np.ndarray:
    """O(N) recursive Basset force (peer of :func:`basset_direct_convolution`).

    Dispatches to the blocked BLAS evaluation for long histories and to the
    scalar recursion otherwise (both agree to round-off).
    """
    accel = np.asarray(accel_history, dtype=float)
    if accel.size >= 8192:
        return basset_recursive_blocked(accel, dt, mu, rho_f, d, n_exp=n_exp)
    return basset_recursive_scalar(accel, dt, mu, rho_f, d, n_exp=n_exp)


def benchmark_basset_methods(
    dt: float,
    n_steps: int,
    mu: float,
    rho_f: float,
    d: float,
    accel: Optional[np.ndarray] = None,
    n_exp_values: Sequence[int] = (3, 8, 16, 24),
) -> Dict[str, object]:
    """Compare O(N^2) reference vs O(N) recursion: accuracy, wall time, term sweep."""
    import time as _time

    t = np.arange(n_steps) * dt
    if accel is None:
        omega = 2.0 * math.pi / (n_steps * dt / 8.0)
        accel = np.sin(omega * t) * np.exp(-t / (0.5 * n_steps * dt))
    accel = np.asarray(accel, dtype=float)

    def _best_cpu_time(func, repeats: int = 3):
        """Minimum CPU time over ``repeats`` runs (robust on a shared machine)."""
        best = float("inf")
        value = None
        for _ in range(repeats):
            t0 = _time.process_time()
            value = func()
            best = min(best, _time.process_time() - t0)
        return best, value

    t_direct, ref = _best_cpu_time(
        lambda: basset_direct_convolution(accel, dt, mu, rho_f, d))

    rows = []
    for n_exp in n_exp_values:
        t_rec, rec = _best_cpu_time(
            lambda: basset_recursive(accel, dt, mu, rho_f, d, n_exp=n_exp))
        rows.append({
            "n_exp": int(n_exp),
            "kernel_rel_l2": basset_kernel_relative_l2(dt, n_steps, n_exp=n_exp),
            "l2_rel_error": float(np.linalg.norm(rec - ref) / np.linalg.norm(ref)),
            "max_abs_error_rel": float(np.max(np.abs(rec - ref)) / np.max(np.abs(ref))),
            "t_recursive_cpu_s": t_rec,
            "speedup_cpu_vs_direct": t_direct / t_rec if t_rec > 0 else float("inf"),
        })
    best = min(rows, key=lambda r: r["l2_rel_error"])
    return {
        "n_steps": int(n_steps),
        "dt": dt,
        "t_direct_cpu_s": t_direct,
        "terms": rows,
        "best": best,
    }


# ---------------------------------------------------------------------------
# 4. Combined unsteady drag (separable ledger)
# ---------------------------------------------------------------------------

@dataclass
class UnsteadyDragResult:
    """Container for the four-term force decomposition."""

    time: np.ndarray
    F_quasi_steady: np.ndarray
    F_added_mass: np.ndarray
    F_basset: np.ndarray
    F_pressure_gradient: np.ndarray
    F_total: np.ndarray

    @property
    def F_qs(self) -> np.ndarray:
        return self.F_quasi_steady

    @property
    def F_am(self) -> np.ndarray:
        return self.F_added_mass

    @property
    def F_pg(self) -> np.ndarray:
        return self.F_pressure_gradient


def unsteady_drag_force(
    u_f: np.ndarray,
    u_p: np.ndarray,
    dt: float,
    mu: float,
    rho_f: float,
    d: float,
    rho_p: float = 1000.0,
    method: str = "recursive",
    n_exp: int = 16,
    derivative: str = "central",
) -> UnsteadyDragResult:
    """Full unsteady force ledger along one axis.

    Parameters
    ----------
    u_f, u_p : fluid / particle velocity time series (same length) [m/s]
    dt : time step [s]
    mu, rho_f, d : fluid viscosity, fluid density, sphere diameter
    rho_p : particle density (bookkeeping only for the one-way problem)
    method : "direct" (O(N^2)) or "recursive" (O(N))
    n_exp : terms of the exponential Basset kernel (recursive method)
    derivative : "central" (2nd order, smooth histories) or "backward"
        (1st order, but it keeps a velocity step as a *single* discrete
        impulse - central differences smear a step over two samples and add a
        spurious half-impulse that inflates the history force by ~50 %)
    """
    u_f = np.asarray(u_f, dtype=float)
    u_p = np.asarray(u_p, dtype=float)
    if u_f.shape != u_p.shape:
        raise ValueError("u_f and u_p must have the same shape")
    if u_f.size < 2:
        raise ValueError("need at least two samples")
    n = u_f.size

    u_slip = u_f - u_p
    F_qs = np.array([schiller_naumann_drag(mu, rho_f, d, s) for s in u_slip])

    if derivative == "central":
        du_slip_dt = np.gradient(u_slip, dt)
        du_f_dt = np.gradient(u_f, dt)
    elif derivative == "backward":
        du_slip_dt = np.zeros(n)
        du_f_dt = np.zeros(n)
        du_slip_dt[1:] = np.diff(u_slip) / dt
        du_f_dt[1:] = np.diff(u_f) / dt
    else:
        raise ValueError(f"unknown derivative scheme: {derivative!r}")

    F_am = np.array([added_mass_force(rho_f, d, a) for a in du_slip_dt])
    F_pg = np.array([pressure_gradient_force(rho_f, d, a) for a in du_f_dt])

    if method == "direct":
        F_basset = basset_direct_convolution(du_slip_dt, dt, mu, rho_f, d)
    elif method == "recursive":
        F_basset = basset_recursive(du_slip_dt, dt, mu, rho_f, d, n_exp=n_exp)
    else:
        raise ValueError(f"unknown method: {method!r}")

    return UnsteadyDragResult(
        time=np.arange(n) * dt,
        F_quasi_steady=F_qs,
        F_added_mass=F_am,
        F_basset=F_basset,
        F_pressure_gradient=F_pg,
        F_total=F_qs + F_am + F_basset + F_pg,
    )


# ---------------------------------------------------------------------------
# 5. Analytic anchors
# ---------------------------------------------------------------------------

def stokes_oscillatory_parameter(omega: float, d: float, mu: float,
                                 rho_f: float) -> complex:
    """``lam = a sqrt(i omega/nu) = (1+i) alpha``, ``alpha = a sqrt(omega/(2 nu))``."""
    a = d / 2.0
    nu = mu / rho_f
    return a * math.sqrt(omega / nu) * (1.0 + 1.0j) / math.sqrt(2.0)


def landau_lifshitz_oscillatory_drag(
    omega: float,
    d: float,
    mu: float,
    rho_f: float,
    U0: float,
    qs_factor: float = 1.0,
) -> complex:
    """Complex force **from slip motion**: ``6 pi mu a W (qs_factor + lam + lam^2/9)``.

    This is the exact unsteady-Stokes (Landau & Lifshitz section 24) result for
    the part of the force that stems from the relative motion, i.e. the sum of
    the quasi-steady, added-mass and Basset contributions.
    """
    a = d / 2.0
    lam = stokes_oscillatory_parameter(omega, d, mu, rho_f)
    H = qs_factor + lam + lam * lam / 9.0
    return 6.0 * math.pi * mu * a * U0 * H


def landau_lifshitz_oscillatory_drag_amplitude_phase(
    omega: float,
    d: float,
    mu: float,
    rho_f: float,
    U0: float,
    include_pressure_gradient: bool = True,
    qs_factor: float = 1.0,
) -> Tuple[float, float]:
    """Amplitude and lag phase of the total force in oscillatory flow.

    A *fixed* sphere in ``u_f = U0 cos(omega t)`` feels
    ``F_hat = 6 pi mu a U0 (qs_factor + lam + lam^2/3)``; the
    ``lam^2/9 -> lam^2/3`` difference is the Froude-Krylov force
    ``rho_f V du_f/dt`` that must be added for a fixed (non-comoving) sphere.

    Returns
    -------
    (amplitude [N], lag [deg]) with ``F(t) = |F_hat| cos(omega t - lag)``
    """
    a = d / 2.0
    lam = stokes_oscillatory_parameter(omega, d, mu, rho_f)
    H = qs_factor + lam + lam * lam / 9.0
    if include_pressure_gradient:
        H = H + 2.0 * lam * lam / 9.0
    F = 6.0 * math.pi * mu * a * U0 * H
    return abs(F), math.degrees(math.atan2(F.imag, F.real))


def stokes_impulsive_start_force(
    t: np.ndarray,
    mu: float,
    rho_f: float,
    d: float,
    U: float,
) -> np.ndarray:
    """Exact unsteady-Stokes force after an impulsive start (Re -> 0).

    ``F(t) = 6 pi mu a U (1 + a/sqrt(pi nu t))`` (Basset 1888): the 1/sqrt(t)
    correction *is* the history force, so this pins the Basset prefactor
    independently of the O(N)/O(N^2) machinery.
    """
    a = d / 2.0
    nu = mu / rho_f
    t = np.asarray(t, dtype=float)
    corr = np.zeros_like(t)
    pos = t > 0.0
    corr[pos] = a / np.sqrt(math.pi * nu * t[pos])
    return 6.0 * math.pi * mu * a * U * (1.0 + corr)


def stokes_second_problem_velocity(
    y: float,
    t: float,
    U0: float,
    omega: float,
    nu: float,
) -> float:
    """Stokes' second problem: ``u_f(y,t) = U0 exp(-kappa y) cos(omega t - kappa y)``.

    Exact solution for an oscillating flat plate; used as the prescribed local
    flow field fed to the particle (one-way coupling).
    """
    kappa = math.sqrt(omega / (2.0 * nu))
    return U0 * math.exp(-kappa * y) * math.cos(omega * t - kappa * y)


# ---------------------------------------------------------------------------
# 6. Controlled scenarios (each isolates one term)
# ---------------------------------------------------------------------------

def run_sudden_acceleration(
    rho_f: float,
    d: float,
    mu: float,
    du_dt: float,
    t_end: float,
    dt: float,
    rho_p: float = 2000.0,
) -> Dict[str, object]:
    """Constant acceleration of the sphere in quiescent fluid (Re -> 0).

    Isolates the added-mass term: ``A = F_am / (rho_f V du/dt)``.  The exact
    Stokes impulsive-start force and the high-frequency (``alpha -> inf``)
    asymptote of the exact unsteady Stokes solution are returned as
    independent references.

    ``rho_p`` is accepted for interface symmetry; it does not enter a one-way
    (prescribed-flow) computation.
    """
    n = int(round(t_end / dt)) + 1
    time = np.arange(n) * dt
    u_slip = du_dt * time
    u_f = np.zeros(n)
    u_p = -u_slip                              # slip = u_f - u_p = du_dt t
    res = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p=rho_p)

    V = sphere_volume(d)
    A_series = res.F_added_mass / (rho_f * V * du_dt)
    return {
        "time": time,
        "u_slip": u_slip,
        "A_measured_mean": float(np.mean(A_series[1:])),
        "A_measured_max_dev": float(np.max(np.abs(A_series[1:] - 0.5))),
        "A_expected": 0.5,
        "F_added_mass": res.F_added_mass,
        "F_basset": res.F_basset,
        "F_quasi_steady": res.F_quasi_steady,
        "F_total": res.F_total,
        "F_stokes_impulsive": stokes_impulsive_start_force(
            time, mu, rho_f, d, du_dt * time[-1]),
    }


def run_steady_uniform(
    rho_f: float,
    d: float,
    mu: float,
    rho_p: float,
    U_slip: float,
    t_end: float,
    dt: float,
    ramp_time: float = 0.0,
) -> Dict[str, object]:
    """Steady slip: only the quasi-steady term may survive (negative control).

    ``ramp_time = 0`` keeps the slip constant from ``t = 0``, so the whole
    unsteady ledger must vanish identically.  ``ramp_time > 0`` ramps linearly
    to ``U_slip`` and holds; once ``t >> a^2/nu`` the history residual must drop
    below ``1e-3 |F_qs|``.
    """
    n = int(round(t_end / dt)) + 1
    time = np.arange(n) * dt
    if ramp_time > 0.0:
        slip = np.where(time < ramp_time, U_slip * time / ramp_time, U_slip)
    else:
        slip = np.full(n, U_slip)
    u_f = slip.copy()
    u_p = np.zeros(n)
    res = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p=rho_p)

    F_sn = schiller_naumann_drag(mu, rho_f, d, U_slip)
    late = slice(max(int(0.5 * n), 1), None)
    return {
        "time": time,
        "F_quasi_steady_mean": float(np.mean(res.F_quasi_steady[late])),
        "F_schiller_naumann": F_sn,
        "F_total_vs_sn_rel_error": float(
            np.max(np.abs(res.F_total[late] - F_sn)) / abs(F_sn)),
        "F_basset_max_late": float(np.max(np.abs(res.F_basset[late]))),
        "F_basset_vs_qs": float(np.max(np.abs(res.F_basset[late])) / abs(F_sn)),
        "F_added_mass_max_late": float(np.max(np.abs(res.F_added_mass[late]))),
        "F_pressure_gradient_max_late": float(
            np.max(np.abs(res.F_pressure_gradient[late]))),
        "F_total": res.F_total,
        "F_basset": res.F_basset,
    }


def run_ramp_then_hold(
    rho_f: float,
    d: float,
    mu: float,
    U_slip: float,
    ramp_time: float,
    t_end: float,
    dt: float,
) -> Dict[str, object]:
    """Ramp the slip, then hold: Basset memory must decay below 1e-3 |F_qs|."""
    return run_steady_uniform(rho_f, d, mu, rho_f, U_slip, t_end, dt,
                              ramp_time=ramp_time)


def run_impulsive_start(
    rho_f: float,
    d: float,
    mu: float,
    U: float,
    t_end: float,
    dt: float,
) -> Dict[str, object]:
    """Impulsive start of steady slip: model vs exact Stokes 1/sqrt(t) decay."""
    n = int(round(t_end / dt)) + 1
    time = np.arange(n) * dt
    # sphere impulsively set into motion at t = dt (the step must be resolved by
    # one time step, otherwise the discrete acceleration history is empty and
    # the history force cannot exist)
    u_p = np.zeros(n)
    u_p[1:] = U
    u_f = np.zeros(n)
    res = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, derivative="backward")
    exact = stokes_impulsive_start_force(time, mu, rho_f, d, U)
    a = d / 2.0
    sel = slice(3, None)
    rel = np.abs(np.abs(res.F_total[sel]) - exact[sel]) / np.abs(exact[sel])
    return {
        "time": time,
        "F_total": res.F_total,
        "F_qs": res.F_quasi_steady,
        "F_basset": res.F_basset,
        "F_exact": exact,
        "F_stokes_steady": 6.0 * math.pi * mu * a * U,
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
    }


def run_oscillatory_flow_simulation(
    rho_f: float,
    d: float,
    mu: float,
    rho_p: float,
    U0: float,
    omega: float,
    t_end: float,
    dt: float,
    method: str = "recursive",
    n_exp: int = 16,
    measure_fraction: float = 0.5,
    qs_factor: float = 1.0,
) -> Dict[str, object]:
    """Fixed sphere in ``u_f = U0 cos(omega t)`` vs the Landau-Lifshitz result.

    The force is fitted as ``F(t) ~ a cos(omega t) + b sin(omega t)`` over the
    last ``1 - measure_fraction`` of the record; the lag is ``-atan2(b, a)``,
    compared with ``atan2(Im F_hat, Re F_hat)`` of the analytic result (same
    convention).
    """
    n = int(round(t_end / dt)) + 1
    time = np.arange(n) * dt
    u_f = U0 * np.cos(omega * time)
    u_p = np.zeros(n)
    res = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p, method=method,
                              n_exp=n_exp)

    start = max(int(measure_fraction * n), 1)
    tt = time[start:]
    basis = np.column_stack([np.cos(omega * tt), np.sin(omega * tt)])
    coeffs, *_ = np.linalg.lstsq(basis, res.F_total[start:], rcond=None)
    a_cos, b_sin = coeffs
    amplitude = math.hypot(a_cos, b_sin)
    phase_lag = math.degrees(-math.atan2(b_sin, a_cos))

    amp_ref, phase_ref = landau_lifshitz_oscillatory_drag_amplitude_phase(
        omega, d, mu, rho_f, U0, qs_factor=qs_factor
    )
    nu = mu / rho_f
    alpha = d / 2.0 * math.sqrt(omega / (2.0 * nu))
    Re_amp = U0 * d / nu

    parts = {
        "quasi_steady": res.F_quasi_steady[start:],
        "added_mass": res.F_added_mass[start:],
        "basset": res.F_basset[start:],
        "pressure_gradient": res.F_pressure_gradient[start:],
    }
    rms = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in parts.items()}
    rms_total = float(np.sqrt(np.mean(res.F_total[start:] ** 2)))

    return {
        "alpha": alpha,
        "Re_amplitude": Re_amp,
        "amplitude_measured": amplitude,
        "amplitude_analytical": amp_ref,
        "amplitude_rel_error": abs(amplitude - amp_ref) / amp_ref,
        "phase_measured_deg": phase_lag,
        "phase_analytical_deg": phase_ref,
        "phase_error_deg": abs(phase_lag - phase_ref),
        "component_rms": rms,
        "history_share": rms["basset"] / rms_total if rms_total > 0 else 0.0,
        "time": time,
        "F_total": res.F_total,
        "F_quasi_steady": res.F_quasi_steady,
        "F_added_mass": res.F_added_mass,
        "F_basset": res.F_basset,
        "F_pressure_gradient": res.F_pressure_gradient,
        "u_f": u_f,
    }


def run_free_particle_oscillatory(
    rho_f: float,
    d: float,
    mu: float,
    rho_ratio: float,
    U0: float,
    omega: float,
    t_end: float,
    dt: float,
    n_exp: int = 16,
) -> Dict[str, object]:
    """Freely responding particle in an oscillatory stream (one-way coupling).

    The particle momentum equation ``m_p du_p/dt = F_qs + F_am + F_B + F_pg`` is
    integrated with the same four-term ledger; the flow field is prescribed.  At
    ``rho_p = rho_f`` the slip tends to zero (the particle co-moves with the
    fluid), so the *ratio* of the unsteady slip-force terms is what remains
    meaningful - that ratio is reported as ``history_share``.
    """
    rho_p = rho_ratio * rho_f
    n = int(round(t_end / dt)) + 1
    time = np.arange(n) * dt
    u_f = U0 * np.cos(omega * time)

    m_p = rho_p * sphere_volume(d)
    V = sphere_volume(d)
    u_p = np.zeros(n)
    du_p = np.zeros(n)
    basset = BassetRecursionState.create(mu, rho_f, d, dt, n_exp=n_exp, n_steps=n)

    prev_slip = 0.0
    # Added mass is treated semi-implicitly: the -C_A rho_f V du_p/dt part moves
    # to the left-hand side, so the effective inertia is (m_p + C_A rho_f V) and
    # the explicit scheme stays stable for every density ratio.
    m_eff = m_p + 0.5 * rho_f * V
    for i in range(n):
        slip = u_f[i] - u_p[i]
        dslip = (slip - prev_slip) / dt
        dsf = (u_f[i] - u_f[i - 1]) / dt if i > 0 else 0.0
        f_qs = schiller_naumann_drag(mu, rho_f, d, slip)
        f_am = added_mass_force(rho_f, d, dsf)      # slip part handled below
        f_pg = pressure_gradient_force(rho_f, d, dsf)
        f_b = basset.step(dslip)
        du_p[i] = (f_qs + f_am + f_b + f_pg) / m_eff
        if i + 1 < n:
            u_p[i + 1] = u_p[i] + du_p[i] * dt
        prev_slip = slip

    res = unsteady_drag_force(u_f, u_p, dt, mu, rho_f, d, rho_p,
                              n_exp=n_exp)
    start = max(int(0.6 * n), 1)
    comp = {
        "quasi_steady": float(np.sqrt(np.mean(res.F_quasi_steady[start:] ** 2))),
        "added_mass": float(np.sqrt(np.mean(res.F_added_mass[start:] ** 2))),
        "basset": float(np.sqrt(np.mean(res.F_basset[start:] ** 2))),
        "pressure_gradient": float(np.sqrt(np.mean(res.F_pressure_gradient[start:] ** 2))),
    }
    V_s = sphere_volume(d)
    dsdt = np.gradient(u_f - u_p, dt)
    A_series = res.F_added_mass[start:] / (rho_f * V_s * dsdt[start:])
    slip_rms = float(np.sqrt(np.mean((u_f[start:] - u_p[start:]) ** 2)))
    slip_amp = float(U0)

    slip_terms = comp["quasi_steady"] + comp["basset"]
    return {
        "rho_ratio": rho_ratio,
        "component_rms": comp,
        "history_share": comp["basset"] / slip_terms if slip_terms > 0 else 0.0,
        "history_share_of_total": comp["basset"] / (
            comp["quasi_steady"] + comp["added_mass"] + comp["basset"]
            + comp["pressure_gradient"]),
        "added_mass_coefficient_mean": float(np.mean(A_series)),
        "slip_rms": slip_rms,
        "slip_ratio": slip_rms / slip_amp,
        "time": time,
        "u_f": u_f,
        "u_p": u_p,
        "F_total": res.F_total,
    }


# ---------------------------------------------------------------------------
# 7. Multi-sphere (dilute) extension: concentration -> 0 must recover 1 sphere
# ---------------------------------------------------------------------------

def stokes_disturbance_velocity(
    x: np.ndarray,
    x_src: np.ndarray,
    a: float,
    w: np.ndarray,
) -> np.ndarray:
    """Velocity disturbance of a translating sphere in Stokes flow.

    Exact Stokes solution for a sphere of radius ``a`` translating with velocity
    ``w`` in a quiescent fluid (Stokeslet + potential dipole)::

        u = (3a/4r)(I + r_hat r_hat) w + (a^3/4r^3)(I - 3 r_hat r_hat) w
    """
    r_vec = x - x_src
    r = float(np.linalg.norm(r_vec))
    if r <= a * (1.0 + 1e-9):
        return np.zeros(3)
    r_hat = r_vec / r
    rr = np.outer(r_hat, r_hat)
    eye = np.eye(3)
    return (3.0 * a / (4.0 * r)) * ((eye + rr) @ w) + \
        (a ** 3 / (4.0 * r ** 3)) * ((eye - 3.0 * rr) @ w)


def multi_sphere_unsteady_drag(
    positions: np.ndarray,
    diameter: float,
    rho_f: float,
    mu: float,
    U_slip: np.ndarray,
    phi: float,
    n_images: int = 0,
) -> Dict[str, object]:
    """Force per sphere in a dilute cloud (steady slip, Stokes neighbour field).

    Each sphere feels the prescribed slip plus the Stokes disturbance of its
    periodic images; the quasi-steady drag uses that *effective* slip.  As
    ``phi -> 0`` (box volume ``N V_p/phi -> inf``) every neighbour distance grows
    like ``phi^{-1/3}`` and the per-sphere force must converge to the single
    sphere value.

    Parameters
    ----------
    positions : (N,3) positions inside the primary cubic box
    diameter : sphere diameter
    rho_f, mu : fluid properties
    U_slip : (N,3) or (3,) prescribed slip velocity (uniform flow, fixed spheres)
    phi : solid volume fraction (defines the periodic box size)
    n_images : image shells summed in each direction
    """
    pos = np.atleast_2d(np.asarray(positions, dtype=float))
    n_sph = pos.shape[0]
    a = diameter / 2.0
    V_p = sphere_volume(diameter)
    L = (n_sph * V_p / phi) ** (1.0 / 3.0)

    slip = np.asarray(U_slip, dtype=float)
    if slip.ndim == 1:
        slip = np.tile(slip, (n_sph, 1))

    # image lattice vectors
    shifts = []
    for ix in range(-n_images, n_images + 1):
        for iy in range(-n_images, n_images + 1):
            for iz in range(-n_images, n_images + 1):
                shifts.append(np.array([ix, iy, iz], dtype=float) * L)

    forces = np.zeros((n_sph, 3))
    eff = np.zeros((n_sph, 3))
    dist_mag = np.zeros(n_sph)
    for i in range(n_sph):
        neigh = np.zeros(3)
        dsum = 0.0
        for j in range(n_sph):
            for sh in shifts:
                if i == j and np.allclose(sh, 0.0):
                    continue
                x_src = pos[j] + sh
                r = float(np.linalg.norm(pos[i] - x_src))
                if r <= a:
                    continue
                neigh += stokes_disturbance_velocity(pos[i], x_src, a, slip[j])
                dsum += a / r
        eff[i] = slip[i] + neigh
        dist_mag[i] = dsum
        f_mag = schiller_naumann_drag(mu, rho_f, diameter,
                                      float(np.linalg.norm(eff[i])))
        forces[i] = f_mag * eff[i] / max(np.linalg.norm(eff[i]), 1e-300)

    return {
        "box_length": L,
        "phi": phi,
        "n_spheres": n_sph,
        "forces": forces,
        "force_magnitude_mean": float(np.mean(np.linalg.norm(forces, axis=1))),
        "effective_slip_mean": float(np.mean(np.linalg.norm(eff, axis=1))),
        "disturbance_sum_mean": float(np.mean(dist_mag)),
        "single_sphere_force": schiller_naumann_drag(
            mu, rho_f, diameter, float(np.linalg.norm(slip[0]))),
    }


def run_dilute_limit_convergence(
    rho_f: float,
    d: float,
    mu: float,
    U_slip: float,
    phi_values: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2),
    n_spheres: int = 8,
    n_images: int = 0,
) -> Dict[str, object]:
    """Sweep the solid fraction and track the per-sphere force vs one sphere.

    The cluster is a finite cubic array (no periodic images), so growing the
    box as ``phi^-1/3`` makes every neighbour distance diverge and the
    per-sphere force must converge to the single-sphere value.  Both the direct
    smallest-``phi`` value and a ``1 + C phi^(1/3)`` extrapolation are reported.
    """
    # cubic lattice inside the (implicitly defined) box; positions normalised,
    # the box grows with phi so the lattice spacing is phi-independent in units
    # of the box.
    m = int(round(n_spheres ** (1.0 / 3.0)))
    grid = np.array([[ix, iy, iz] for ix in range(m) for iy in range(m)
                     for iz in range(m)], dtype=float)[:n_spheres]
    grid = grid / m + 0.5 / m
    rows = []
    for phi in phi_values:
        V_p = sphere_volume(d)
        L = (n_spheres * V_p / phi) ** (1.0 / 3.0)
        pos = grid * L
        out = multi_sphere_unsteady_drag(pos, d, rho_f, mu,
                                         np.array([U_slip, 0.0, 0.0]), phi,
                                         n_images=n_images)
        rows.append({
            "phi": phi,
            "box_length": out["box_length"],
            "force_ratio": out["force_magnitude_mean"] / out["single_sphere_force"],
            "disturbance_sum_mean": out["disturbance_sum_mean"],
        })
    smallest = min(rows, key=lambda r: r["phi"])
    # dilute expansion: ratio = 1 + C phi^(1/3) + ... -> extrapolate to phi = 0
    x = np.array([r["phi"] ** (1.0 / 3.0) for r in rows])
    y = np.array([r["force_ratio"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1) if rows.__len__() >= 2 else (0.0, y[0])
    return {
        "rows": rows,
        "single_sphere_force": out["single_sphere_force"],
        "convergence_error_smallest_phi": abs(smallest["force_ratio"] - 1.0),
        "convergence_phi": smallest["phi"],
        "phi13_extrapolated_ratio": float(intercept),
        "phi13_slope": float(slope),
        "extrapolation_error": abs(float(intercept) - 1.0),
    }