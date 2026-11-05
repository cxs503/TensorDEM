"""Verification harness for the D3Q19 BGK + Smagorinsky core in :mod:`wake_lbm`.

Why this module exists
----------------------
``wake_lbm.py`` is a real D3Q19 lattice-Boltzmann core (Guo-type actuator-disk
force, Smagorinsky subgrid closure, free-slip lateral boundaries), and
``docs/THREEFIELD_ICE_HULL_PROPELLER.md`` uses it to produce a propeller race.
But nothing in the repository graded the *transport* of that core:

* the viscosity claim ``nu = (tau0 - 1/2) / 3`` was never measured against a
  decaying solution of the Navier-Stokes equations, and
* the subgrid closure ``_effective_tau`` had no reference at all, so the
  eddy viscosity it actually realises was unknown.

This module supplies independent references and the measurement machinery.  It
**does not change the physics of the solver**: it only drives the solver's own
kernels (:meth:`WakeLBM._equilibrium`, :meth:`WakeLBM._effective_tau`,
:meth:`WakeLBM._stream`) and compares the outcome with closed forms.

1. :func:`periodic_stream` -- the box built by ``LBMConfig`` is bounded, so a
   spectral/analytic test needs a periodic closure.  Padding the field with one
   wrapped halo plane and extracting the interior makes ``_stream`` an *exact*
   periodic streaming operator, and :func:`reference_stream` re-implements the
   same shift with ``torch.roll`` as an independent check (bit-exact expected).
2. :func:`sgs_fixed_point_tau` / :func:`sgs_eddy_viscosity` -- closed forms for
   the Smagorinsky closure.  With ``Pi_ab = sum_i c_ia c_ib f_i^neq``,
   ``|Pi| = sqrt(Pi_ab Pi_ab)`` (Frobenius norm of the full symmetric tensor)
   and ``|S| = sqrt(2 S_ab S_ab)``, the code's update
   ``tau_eff = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho)) / 2``
   reproduces ``nu_t = Cs^2 |S|`` (lattice spacing 1) *exactly* at its fixed
   point ``tau_eff = tau0 + 3 Cs^2 |S|``.  Any deviation is a closure defect.
3. :func:`synthetic_strain_field` -- builds populations whose non-equilibrium
   part is the Chapman-Enskog one for a prescribed velocity gradient, so the
   closure can be graded in isolation on a pure shear, pure normal-strain or
   mixed field.
4. :func:`shear_wave_decay` / :func:`shear_wave_rate` -- the diffusive check:
   ``u_x = A cos(k y)`` is an exact solution of the incompressible
   Navier-Stokes equations (its convective term vanishes identically) and decays
   as ``exp(-nu k^2 t)``; on the lattice the same mode decays with the discrete
   Laplacian ``k_d^2 = 2 (1 - cos k)``.  Measuring the decay rate therefore
   measures the solver's momentum diffusivity, not a curve fit.

Conventions: lattice units, ``dx = dt = 1``, ``cs^2 = 1/3``, ``rho ~ 1``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .wake_lbm import LBMConfig, WakeLBM

# Slot order used by WakeLBM._outer: the six independent components of a
# symmetric second-order tensor.
TENSOR_SLOTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
# Frobenius norm weights: off-diagonal components appear twice in Pi_ab Pi_ab.
FROBENIUS_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# periodic closure built from the solver's own streaming kernel
# ---------------------------------------------------------------------------
def _circular_pad(f: torch.Tensor) -> torch.Tensor:
    """Wrap one halo plane onto each axis (torch's ``pad`` has no 4-D circular mode)."""
    g = torch.cat((f[-1:], f, f[:1]), dim=0)
    g = torch.cat((g[:, -1:], g, g[:, :1]), dim=1)
    return torch.cat((g[:, :, -1:], g, g[:, :, :1]), dim=2)


def periodic_stream(inst: WakeLBM, f: torch.Tensor) -> torch.Tensor:
    """Exact periodic streaming of ``f`` using ``WakeLBM._stream``.

    ``_stream`` shifts every population by one cell and leaves the cells outside
    the shifted slab at zero, i.e. it is a *bounded-box* operator.  If the field
    is padded with one wrapped halo plane per axis, the interior of the shifted
    array receives exactly the wrapped neighbour, so ``[1:-1]`` of the padded
    result is the periodic streaming operator applied to ``f``.
    """
    padded = _circular_pad(f)
    # ``_stream`` slices with the *instance* dimensions, so point them at the
    # padded shape for the duration of the call (the shift itself is generic).
    saved = (inst.nx, inst.ny, inst.nz)
    inst.nx, inst.ny, inst.nz = padded.shape[0], padded.shape[1], padded.shape[2]
    try:
        shifted = inst._stream(padded)
    finally:
        inst.nx, inst.ny, inst.nz = saved
    return shifted[1:-1, 1:-1, 1:-1, :]


def reference_stream(inst: WakeLBM, f: torch.Tensor) -> torch.Tensor:
    """Independent periodic streaming: sum_i roll(f_i, shift=c_i)."""
    out = torch.zeros_like(f)
    for i in range(len(inst.c)):
        cx, cy, cz = inst._cx[i], inst._cy[i], inst._cz[i]
        out[..., i] = torch.roll(f[..., i], shifts=(cx, cy, cz), dims=(0, 1, 2))
    return out


def double_symmetric_contraction(pi: torch.Tensor) -> torch.Tensor:
    """sqrt(Pi_ab Pi_ab) over the full symmetric tensor.

    ``pi`` carries the six independent components; the three off-diagonal ones
    occur twice in ``Pi_ab Pi_ab``, so they are weighted by two.
    """
    weights = torch.tensor(FROBENIUS_WEIGHTS, dtype=pi.dtype, device=pi.device)
    return torch.sqrt((weights * pi * pi).sum(dim=-1).clamp_min(0.0))


def reduced_symmetric_contraction(pi: torch.Tensor) -> torch.Tensor:
    """sqrt over the six stored slots only -- the pre-m2949 behaviour.

    Kept as the *negative control* of the audit: it is not a Frobenius norm of a
    symmetric tensor and under-estimates ``|Pi|`` by up to ``sqrt(2)``.
    """
    return torch.sqrt((pi * pi).sum(dim=-1).clamp_min(0.0))


def non_equilibrium_flux(inst: WakeLBM, f: torch.Tensor) -> torch.Tensor:
    """``Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq)`` in the six-slot layout."""
    rho, u = inst._macroscopic(f)
    f_neq = f - inst._equilibrium(rho.unsqueeze(-1), u)
    return torch.einsum("ik,...i->...k", inst._outer, f_neq)


# ---------------------------------------------------------------------------
# Smagorinsky closure: closed forms
# ---------------------------------------------------------------------------
def strain_invariants(gradient: Sequence[Sequence[float]]) -> dict[str, float]:
    """Decompose a velocity gradient into ``|S|``, divergence and vorticity."""
    g = [[float(v) for v in row] for row in gradient]
    if len(g) != 3 or any(len(row) != 3 for row in g):
        raise ValueError("velocity gradient must be 3x3")
    s = [[0.5 * (g[a][b] + g[b][a]) for b in range(3)] for a in range(3)]
    s_sq = sum(s[a][b] * s[a][b] for a in range(3) for b in range(3))
    w = [[0.5 * (g[a][b] - g[b][a]) for b in range(3)] for a in range(3)]
    w_sq = sum(w[a][b] * w[a][b] for a in range(3) for b in range(3))
    return {
        "strain": math.sqrt(2.0 * s_sq),
        "divergence": g[0][0] + g[1][1] + g[2][2],
        "vorticity": math.sqrt(2.0 * w_sq),
        "normal": math.sqrt(s[0][0] ** 2 + s[1][1] ** 2 + s[2][2] ** 2),
        "shear": math.sqrt(2.0 * sum(s[a][b] ** 2 for a in range(3) for b in range(3) if a != b)),
    }


def sgs_fixed_point_tau(tau0: float, cs: float, strain: float) -> float:
    """Closed-form fixed point ``tau_eff = tau0 + 3 Cs^2 |S|``.

    Derived from the code's own update together with the Chapman-Enskog flux
    ``Pi_ab = -(2/3) rho tau S_ab`` (incompressible, lattice units).
    """
    if tau0 <= 0.5:
        raise ValueError("tau0 must exceed 0.5")
    if cs < 0.0:
        raise ValueError("cs must be nonnegative")
    if strain < 0.0:
        raise ValueError("strain must be nonnegative")
    return tau0 + 3.0 * cs * cs * strain


def sgs_eddy_viscosity(tau_eff: torch.Tensor | float, tau0: float) -> torch.Tensor | float:
    """``nu_t = (tau_eff - tau0) / 3``."""
    return (tau_eff - tau0) / 3.0


def target_eddy_viscosity(cs: float, strain: float) -> float:
    """Smagorinsky target ``nu_t = (Cs dx)^2 |S|`` with ``dx = 1``."""
    return cs * cs * strain


# ---------------------------------------------------------------------------
# synthetic Chapman-Enskog non-equilibrium field
# ---------------------------------------------------------------------------
def synthetic_strain_field(
    inst: WakeLBM,
    gradient: Sequence[Sequence[float]],
    *,
    tau: float,
    rho: float = 1.0,
) -> torch.Tensor:
    """Populations whose non-equilibrium part is exactly ``-3 w tau Q_ab d_a u_b``.

    ``Q_iab = c_ia c_ib - cs^2 delta_ab`` is the second Hermite polynomial of the
    D3Q19 set, so that ``Pi_ab = -(2/3) rho tau S_ab`` to machine accuracy.  The
    field is uniform, so it also exercises the vectorised kernel shape.
    """
    g = torch.tensor(gradient, dtype=inst.dtype)
    if g.shape != (3, 3):
        raise ValueError("velocity gradient must be 3x3")
    if g.abs().sum() == 0:
        raise ValueError("velocity gradient must be non-zero")
    shape = (inst.nx, inst.ny, inst.nz, 3)
    u = torch.zeros(shape, dtype=inst.dtype)
    rho_field = torch.full((inst.nx, inst.ny, inst.nz), float(rho), dtype=inst.dtype)
    c = inst.c  # (19, 3)
    cq = torch.einsum("ia,ib->iab", c, c)  # (19, 3, 3)
    q = cq - (1.0 / 3.0) * torch.eye(3, dtype=inst.dtype)
    f_neq = -3.0 * inst.w.view(1, 1, 1, -1) * float(tau) * float(rho) * torch.einsum(
        "iab,ab->i", q, g
    )
    feq = inst._equilibrium(rho_field.unsqueeze(-1), u)
    return feq + f_neq.view(1, 1, 1, -1)


# ---------------------------------------------------------------------------
# diffusive check: decaying shear wave
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShearWaveResult:
    """Outcome of a decaying shear-wave measurement."""

    tau0: float
    wavenumber: float
    steps: int
    amplitude0: float
    rate_measured: float
    rate_continuum: float
    rate_discrete: float
    history: tuple[float, ...] = ()
    times: tuple[int, ...] = ()

    @property
    def viscosity_measured(self) -> float:
        return self.rate_measured / (self.wavenumber**2)

    @property
    def viscosity_nominal(self) -> float:
        return (self.tau0 - 0.5) / 3.0

    @property
    def discrete_laplacian(self) -> float:
        return 2.0 * (1.0 - math.cos(self.wavenumber)) / (self.wavenumber**2)

    def relative_error(self) -> float:
        """Error of the measured rate against the discrete-Laplacian reference."""
        return self.rate_measured / self.rate_discrete - 1.0


def shear_wave_rate(tau0: float, wavenumber: float) -> tuple[float, float]:
    """Analytic decay rates ``(continuum, discrete-Laplacian)`` for the mode."""
    nu = (tau0 - 0.5) / 3.0
    k = wavenumber
    return nu * k * k, nu * 2.0 * (1.0 - math.cos(k))


def shear_wave_decay(
    *,
    tau0: float,
    size: int = 16,
    mode: int = 1,
    amplitude: float = 1e-3,
    steps: int = 400,
    fit_from: int = 20,
) -> ShearWaveResult:
    """March ``u_x = A cos(k y)`` in a periodic box and fit its decay rate.

    No forcing is applied and ``smagorinsky_cs`` is zero: this measures the
    momentum diffusivity implied by ``tau0`` alone.
    """
    if size < 8:
        raise ValueError("size must be at least 8 cells")
    if not 0.5 < tau0:
        raise ValueError("tau0 must exceed 0.5")
    if mode < 1 or 2 * mode >= size:
        raise ValueError("mode must resolve the wave (2 * mode < size)")
    config = LBMConfig(
        nx=size,
        ny=size,
        nz=size,
        tau0=tau0,
        smagorinsky_cs=0.0,
        inflow_velocity=0.05,
        jet_velocity=0.05 + amplitude,
        disk_radius=1.0,
        disk_x=size // 2,
        disk_thickness=1,
        quarter_domain=False,
    )
    inst = WakeLBM(config)
    k = 2.0 * math.pi * mode / size
    y = torch.arange(size, dtype=inst.dtype).view(1, size, 1)
    u = torch.zeros((size, size, size, 3), dtype=inst.dtype)
    u[:, :, :, 0] = amplitude * torch.cos(k * y)
    rho = torch.ones((size, size, size), dtype=inst.dtype)
    f = inst._equilibrium(rho.unsqueeze(-1), u)

    # Isolate the k-mode: the initial condition f = f^eq is not the exact
    # eigenvector, so the first few steps contain a relaxation transient that a
    # pointwise maximum would fold into the slope.  Project onto cos(k y) instead.
    cosk = torch.cos(k * torch.arange(size, dtype=inst.dtype))

    def mode_amplitude(field: torch.Tensor) -> float:
        rho_now, u_now = inst._macroscopic(field)
        profile = u_now[:, :, :, 0].mean(dim=(0, 2))
        return float((profile * cosk).sum().item() * 2.0 / size)

    history = [(0, amplitude)]
    for step in range(1, steps + 1):
        rho_f, u_f = inst._macroscopic(f)
        tau = inst._effective_tau(f, rho_f, u_f)
        feq = inst._equilibrium(rho_f.unsqueeze(-1), u_f)
        f = f - (1.0 / tau).unsqueeze(-1) * (f - feq)
        f = periodic_stream(inst, f)
        if step > fit_from:
            history.append((step, mode_amplitude(f)))

    # Closed-form least squares on ln(A) vs t.  ``torch.linalg.lstsq`` returned a
    # slope two orders of magnitude too large on this well-conditioned system
    # (identical data, verified against the endpoint ratio), so the two-point
    # normal equations are written out explicitly.
    times = torch.tensor([row[0] for row in history], dtype=torch.float64)
    a = torch.tensor([row[1] for row in history], dtype=torch.float64)
    mask = a > 0.0
    t = times[mask]
    y = torch.log(a[mask])
    t_bar = t.mean()
    y_bar = y.mean()
    slope = ((t - t_bar) * (y - y_bar)).sum() / ((t - t_bar) ** 2).sum()
    rate_cont, rate_disc = shear_wave_rate(tau0, k)
    return ShearWaveResult(
        tau0=tau0,
        wavenumber=k,
        steps=steps,
        amplitude0=amplitude,
        rate_measured=float(-slope),
        rate_continuum=rate_cont,
        rate_discrete=rate_disc,
        history=tuple(value for _, value in history),
        times=tuple(step for step, _ in history),
    )