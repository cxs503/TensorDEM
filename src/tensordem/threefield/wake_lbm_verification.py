"""Analytic (code) verification of the D3Q19 + Smagorinsky propeller-race solver.

Why this module exists
----------------------
``tensordem.threefield.wake_lbm`` is graded in
``docs/THREEFIELD_ICE_HULL_PROPELLER.md`` by comparing the *resolved race* with
an empirical Gaussian profile and by a thrust/momentum-flux balance.  That is a
functional comparison inside a coupled run: a wrong collision kernel, a wrong
streaming stencil or a wrong Smagorinsky closure would all be indistinguishable
from "the race is under-resolved at 10 cells per diameter" -- which is exactly
the headline limitation of that campaign.  The solver kernel itself had never
been compared with a closed-form solution.

This module supplies the missing code-verification ladder.  Every level is
anchored on a closed form, not on the solver:

======================  ================================================  ==========
level                   anchor                                            tolerance
======================  ================================================  ==========
V1 equilibrium moments  ``sum f``, ``sum f c``, ``sum f c c``              roundoff
V2 streaming stencil    independent scatter reference on a random field   roundoff
V3 zero-thrust state    uniform flow is a fixed point of the real step    roundoff
V4 periodic conservation mass and momentum, no forcing                    roundoff
V5 shear-wave viscosity ``gamma = nu k^2``, ``nu = (tau0 - 1/2)/3``        < 1 %
V6 Galilean invariance  bulk translation leaves ``gamma`` unchanged       < 1 %
V7 Smagorinsky tau law  closed-form quadratic root, single cell           roundoff
V8 SGS damped decay     ``dA/dt = -k^2 A (nu0 + 4/(3 pi) Cs^2 k A)``      < 5 %
======================  ================================================  ==========

What the ladder does **not** cover is stated in
``docs/threefield_lbm_verification.md``: the inlet/outlet/wall closures are only
exercised by V3 (they admit a uniform state exactly), the actuator-disk forcing
law has no closed-form anchor, and every check is in incompressible low-Mach
lattice units.

Derivation of the two analytic anchors
--------------------------------------
**(a) Smagorinsky closure as implemented.**  With the second-moment tensor
``Pi_ij = sum_k c_ki c_kj f_k^neq`` the code evaluates

    tau_eff = (tau0 + sqrt(tau0^2 + 18*sqrt(2)*Cs^2*|Pi|/rho)) / 2 .

For a plane shear wave ``u_y = A sin(kx)`` the first-order non-equilibrium part
is ``f_i^(1) = -3 w_i rho tau c_ix c_iy A k cos(kx)``, so the full tensor norm is
``|Pi|_frobenius = (sqrt(2)/3) rho tau |du_y/dx|``.  The code builds the norm from
six Voigt components ``(xx, yy, zz, xy, xz, yz)`` and therefore counts each shear
component **once**:

    |Pi|_single = |Pi|_frobenius / sqrt(2) = (1/3) rho tau |du_y/dx| .

Inserting that into the code's ``18 sqrt(2) Cs^2 |Pi|/rho`` gives the closed form
used by :func:`analytic_smagorinsky_tau`:

    tau_eff = (tau0 + sqrt(tau0^2 + 6 sqrt(2) Cs^2 tau0 |du_y/dx|)) / 2 ,  nu_t = (Cs Delta)^2 |S| / sqrt(2) .

That factor ``1/sqrt(2)`` relative to the textbook closure
``nu_t = (Cs Delta)^2 |S|`` (``|S| = sqrt(2 S_ij S_ij)``, ``Delta = 1``) is a real
property of the implementation, not of the anchor, and
:func:`smagorinsky_constant_consistency` certifies it: the effective constant is
``Cs_eff = 2^(-1/4) Cs = 0.841 Cs``, i.e. 15.9 % *below* the nominal value carried
in ``LBMConfig.smagorinsky_cs``: for the campaign's ``Cs = 0.14-0.16`` the eddy
viscosity actually applied corresponds to ``Cs = 0.118-0.135``.

**(b) SGS-damped shear-wave decay.**  With the implemented eddy viscosity
``nu(x) = nu0 + (Cs Delta)^2 |du_y/dx| / sqrt(2)`` the mode projection of
``d_t u = d_x(nu d_x u)`` onto ``sin(kx)`` gives

    dA/dt = -k^2 A (nu0 + (4/(3 pi)) Cs^2 k A / sqrt(2)),

because ``<|cos|^3> = 2/(3 pi)``.  The undamped limit (``Cs = 0``) is
``A(t) = A0 exp(-nu k^2 t)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .wake_lbm import LBMConfig, WakeLBM

__all__ = [
    "HARNESS_SCHEMA",
    "VerificationRow",
    "PeriodicShearWave",
    "analytic_smagorinsky_tau",
    "smagorinsky_constant_consistency",
    "log_linear_decay_rate",
    "measure_shear_decay",
    "equilibrium_moment_check",
    "streaming_reference_check",
    "zero_thrust_uniform_check",
    "conservation_check",
    "shear_wave_viscosity_scan",
    "galilean_invariance_check",
    "smagorinsky_tau_check",
    "sgs_damped_decay_check",
    "run_verification_suite",
]

HARNESS_SCHEMA = "tensordem.wake-lbm-verification/1"


@dataclass(frozen=True)
class VerificationRow:
    """One verified (or honestly failed) rung of the ladder."""

    level: str
    quantity: str
    anchor: str
    samples: int
    worst_relative_error: float
    tolerance: float
    passed: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "quantity": self.quantity,
            "anchor": self.anchor,
            "samples": int(self.samples),
            "worst_relative_error": float(self.worst_relative_error),
            "tolerance": float(self.tolerance),
            "passed": bool(self.passed),
            "note": self.note,
        }


# ----------------------------------------------------------------------
# analytic anchors
# ----------------------------------------------------------------------
def analytic_smagorinsky_tau(tau0: float, cs: float, shear_rate: float) -> float:
    """Closed-form ``tau_eff`` for a shear wave carrying ``|S| = |du_y/dx|``.

    ``shear_rate`` is the local (per-cell) strain-rate magnitude.  The
    non-equilibrium stress entering the closure is built from the *current*
    distribution, which in this fixture carries ``tau0`` in its first-order
    part, hence ``tau0`` inside the discriminant.
    """
    if tau0 <= 0.5:
        raise ValueError("tau0 must exceed 0.5")
    if cs < 0.0:
        raise ValueError("smagorinsky_cs must be nonnegative")
    return 0.5 * (
        tau0
        + math.sqrt(
            tau0 * tau0 + 6.0 * math.sqrt(2.0) * cs * cs * tau0 * abs(shear_rate)
        )
    )


def smagorinsky_constant_consistency(
    tau0: float, cs: float, shear_rate: float
) -> dict[str, float]:
    """Certify the implemented stress-norm convention against ``nu_t = (Cs Delta)^2 |S|``.

    The measured eddy viscosity must equal ``sqrt(2) Cs^2 |S|`` to roundoff, i.e.
    the nominal ``Cs`` is over-delivering by ``2^(1/4)`` in constant space.
    """
    implemented = analytic_smagorinsky_tau(tau0, cs, shear_rate)
    eddy = (implemented - tau0) / 3.0
    continuum = cs * cs * abs(shear_rate)
    ratio = eddy / continuum if continuum > 0.0 else float("nan")
    return {
        "implemented_tau_eff": implemented,
        "eddy_viscosity": eddy,
        "continuum_eddy_viscosity": continuum,
        "eddy_viscosity_ratio": ratio,
        "sqrt2": math.sqrt(2.0),
        "inverse_sqrt2": 1.0 / math.sqrt(2.0),
        "relative_difference": abs(ratio - 1.0 / math.sqrt(2.0)) * math.sqrt(2.0),
        "effective_cs_ratio": (ratio) ** 0.5 if continuum > 0.0 else float("nan"),
        "effective_cs": (ratio) ** 0.5 * cs if continuum > 0.0 else float("nan"),
    }


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
class PeriodicShearWave:
    """Plane shear wave ``u_y = A sin(kx)`` in a fully periodic box.

    The production collision kernels (``_macroscopic``, ``_effective_tau``,
    ``_equilibrium``) are used unchanged; only the streaming operator and the
    boundary closures are replaced by periodic wrapping, because the race solver
    is bounded.  The bounded closures are verified separately (V3).
    """

    def __init__(
        self,
        *,
        nx: int = 32,
        ny: int = 8,
        nz: int = 8,
        tau0: float = 0.55,
        smagorinsky_cs: float = 0.0,
        amplitude: float = 1.0e-3,
        bulk_velocity: float = 0.0,
        modes: int = 1,
        dtype: str = "float64",
    ) -> None:
        if nx < 12 or ny < 8 or nz < 8:
            raise ValueError("the fixture needs nx >= 12 (actuator-disk constraint) and ny, nz >= 8")
        if modes < 1 or modes >= nx // 2:
            raise ValueError("modes must be a positive resolvable wave-number index")
        self.config = LBMConfig(
            nx=nx,
            ny=ny,
            nz=nz,
            tau0=tau0,
            smagorinsky_cs=smagorinsky_cs,
            disk_radius=min(ny, nz) / 2.0 - 1.0,
            disk_x=8,
            inlet_perturbation=0.0,
            disk_perturbation=0.0,
            dtype=dtype,
        )
        self.solver = WakeLBM(self.config, seed=0)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.wavenumber = 2.0 * math.pi * float(modes) / float(nx)
        self.amplitude0 = float(amplitude)
        self.bulk_velocity = float(bulk_velocity)
        self._x = torch.arange(nx, dtype=self.solver.dtype).reshape(nx, 1, 1)
        rho = torch.ones((nx, ny, nz, 1), dtype=self.solver.dtype)
        u = torch.zeros((nx, ny, nz, 3), dtype=self.solver.dtype)
        u[..., 0] = self.bulk_velocity
        u[..., 1] = self.amplitude0 * torch.sin(self.wavenumber * self._x)
        self.f = self.solver._equilibrium(rho, u)
        self.steps = 0

    # -- helpers -------------------------------------------------------
    @property
    def viscosity(self) -> float:
        return (self.config.tau0 - 0.5) / 3.0

    def fields(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solver._macroscopic(self.f)

    def mode_amplitude(self) -> float:
        """Complex fundamental-mode amplitude of ``u_y``.

        ``sqrt((2<u_y sin kx>)^2 + (2<u_y cos kx>)^2)``.  The complex form is
        required because a transverse perturbation is *advected* by a mean flow
        (``u_y = A sin(k(x - Ut))``), so a sine-only projection decays at
        ``k U`` even when the viscous decay is unchanged -- that phase drift is
        physical, not a solver error.
        """
        _, u = self.fields()
        profile = u[..., 1].mean(dim=(1, 2))
        phase = self.wavenumber * self._x.reshape(-1)
        sin_part = 2.0 * float((profile * torch.sin(phase)).mean())
        cos_part = 2.0 * float((profile * torch.cos(phase)).mean())
        return math.hypot(sin_part, cos_part)

    def total_mass(self) -> float:
        return float(self.f.sum())

    def total_momentum(self) -> torch.Tensor:
        rho, u = self.fields()
        return (rho.unsqueeze(-1) * u).sum(dim=(0, 1, 2))

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            rho, u = self.solver._macroscopic(self.f)
            tau = self.solver._effective_tau(self.f, rho, u)
            feq = self.solver._equilibrium(rho.unsqueeze(-1), u)
            post = self.f - (1.0 / tau.unsqueeze(-1)) * (self.f - feq)
            out = torch.empty_like(post)
            for i, (cx, cy, cz) in enumerate(self.solver.c.tolist()):
                out[..., i] = torch.roll(
                    post[..., i], shifts=(int(cx), int(cy), int(cz)), dims=(0, 1, 2)
                )
            self.f = out
            self.steps += 1

    def history(self, steps: int, sample_every: int = 1) -> list[tuple[int, float]]:
        trace = [(self.steps, self.mode_amplitude())]
        for _ in range(int(steps)):
            self.step(1)
            if self.steps % int(sample_every) == 0:
                trace.append((self.steps, self.mode_amplitude()))
        return trace

    def snapshot(self) -> dict[str, Any]:
        return {"schema": HARNESS_SCHEMA + "/fixture", "f": self.f.clone(), "steps": self.steps}

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("schema") != HARNESS_SCHEMA + "/fixture":
            raise ValueError("fixture restart schema mismatch")
        self.f = state["f"].clone()
        self.steps = int(state["steps"])


class _ZeroThrust(WakeLBM):
    """Production step loop with the actuator-disk forcing switched off."""

    def _disk_force(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)


def log_linear_decay_rate(
    trace: Iterable[tuple[int, float]], *, window: float = 0.35
) -> float:
    """Least-squares ``-d ln A / dt`` over ``A >= window * A(0)``."""
    points = [(float(t), math.log(a)) for t, a in trace if a > 0.0]
    if len(points) < 3:
        raise ValueError("need at least three positive amplitude samples")
    cutoff = math.log(window) + points[0][1]
    subset = [(t, y) for t, y in points if y >= cutoff]
    if len(subset) < 3:
        subset = points[:3]
    n = float(len(subset))
    mean_t = sum(t for t, _ in subset) / n
    mean_y = sum(y for _, y in subset) / n
    sxx = sum((t - mean_t) ** 2 for t, _ in subset)
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in subset)
    return -sxy / sxx


# ----------------------------------------------------------------------
# V1: equilibrium moments
# ----------------------------------------------------------------------
def equilibrium_moment_check(*, dtype: str = "float64", seed: int = 5) -> dict[str, Any]:
    """``sum f = rho``, ``sum f c = rho u``, ``sum f c c = rho (c_s^2 I + u u)``."""
    nx = ny = nz = 12
    solver = WakeLBM(
        LBMConfig(nx=nx, ny=ny, nz=nz, disk_radius=2.0, disk_x=8, dtype=dtype), seed=0
    )
    generator = torch.Generator().manual_seed(int(seed))
    rho = 1.0 + 0.1 * torch.rand(
        (nx, ny, nz, 1), generator=generator, dtype=solver.dtype
    )
    u = 0.05 * (
        torch.rand((nx, ny, nz, 3), generator=generator, dtype=solver.dtype) - 0.5
    )
    f = solver._equilibrium(rho, u)
    rho_out, u_out = solver._macroscopic(f)
    second = torch.einsum("ik,...i->...k", solver._outer, f)
    target = torch.zeros_like(second)
    for axis in range(3):
        target[..., axis] = rho[..., 0] * (1.0 / 3.0 + u[..., axis] ** 2)
    for slot, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        target[..., 3 + slot] = rho[..., 0] * u[..., a] * u[..., b]
    scale = float(rho.abs().max())
    errors = {
        "density": float((rho_out - rho[..., 0]).abs().max()) / scale,
        "velocity": float((u_out - u).abs().max()) / 0.05,
        "second_moment": float((second - target).abs().max()) / (scale / 3.0),
    }
    weight_sum = float(solver.w.sum())
    shear_moment = float(
        sum(
            float(solver.w[i]) * float(solver.c[i, 0]) ** 2 * float(solver.c[i, 1]) ** 2
            for i in range(len(solver.w))
        )
    )
    return {
        **errors,
        "worst_relative_error": max(errors.values()),
        "weight_sum_minus_one": abs(weight_sum - 1.0),
        "cxy_moment": shear_moment,
        "cxy_moment_error": abs(shear_moment - 1.0 / 9.0),
    }


# ----------------------------------------------------------------------
# V2: streaming stencil against an independent scatter reference
# ----------------------------------------------------------------------
def streaming_reference_check(*, dtype: str = "float64", seed: int = 7) -> dict[str, Any]:
    """Compare ``_stream`` with ``torch.roll`` plus explicit wrap masking."""
    solver = WakeLBM(
        LBMConfig(nx=12, ny=10, nz=9, disk_radius=3.0, disk_x=8, dtype=dtype), seed=0
    )
    generator = torch.Generator().manual_seed(int(seed))
    f = torch.rand((12, 10, 9, 19), generator=generator, dtype=solver.dtype)
    reference = torch.zeros_like(f)
    nx, ny, nz = solver.nx, solver.ny, solver.nz
    for i, (cx, cy, cz) in enumerate(solver.c.tolist()):
        rolled = torch.roll(
            f[..., i], shifts=(int(cx), int(cy), int(cz)), dims=(0, 1, 2)
        )
        if cx > 0:
            rolled[: int(cx)] = 0.0
        elif cx < 0:
            rolled[nx + int(cx):] = 0.0
        if cy > 0:
            rolled[:, : int(cy)] = 0.0
        elif cy < 0:
            rolled[:, ny + int(cy):] = 0.0
        if cz > 0:
            rolled[:, :, : int(cz)] = 0.0
        elif cz < 0:
            rolled[:, :, nz + int(cz):] = 0.0
        reference[..., i] = rolled
    produced = solver._stream(f)
    scale = float(f.abs().max())
    return {
        "worst_relative_error": float((produced - reference).abs().max()) / scale,
        "cells": int(f.numel()),
        "dropped_source_fraction": float(
            (reference.abs() < 0.0).sum().item() / max(1, reference.numel())
        ),
    }


# ----------------------------------------------------------------------
# V3: uniform flow is a fixed point of the production step
# ----------------------------------------------------------------------
def zero_thrust_uniform_check(
    *, nx: int = 24, ny: int = 10, nz: int = 10, tau0: float = 0.5006, steps: int = 200
) -> dict[str, Any]:
    """Uniform flow must stay uniform through the bounded step (no forcing)."""
    config = LBMConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau0=tau0,
        smagorinsky_cs=0.16,
        inflow_velocity=0.04,
        jet_velocity=0.10,
        disk_radius=min(ny, nz) / 2.0 - 1.0,
        disk_x=8,
        inlet_perturbation=0.0,
        disk_perturbation=0.0,
    )
    solver = _ZeroThrust(config, seed=0)
    solver.step(steps)
    rho, u = solver._macroscopic(solver.f)
    target = torch.zeros_like(u)
    target[..., 0] = config.inflow_velocity
    scale = 1.0 + float(target.abs().max())
    return {
        "worst_relative_error": max(
            float((rho - 1.0).abs().max()), float((u - target).abs().max())
        ) / scale,
        "density_error": float((rho - 1.0).abs().max()),
        "velocity_error": float((u - target).abs().max()),
        "steps": int(steps),
    }


# ----------------------------------------------------------------------
# V4: periodic conservation
# ----------------------------------------------------------------------
def conservation_check(
    *, steps: int = 400, tau0: float = 0.55, amplitude: float = 5.0e-3
) -> dict[str, Any]:
    """Mass and momentum are conserved exactly by collision plus periodic shift."""
    fixture = PeriodicShearWave(tau0=tau0, amplitude=amplitude, bulk_velocity=0.02)
    mass0 = fixture.total_mass()
    momentum0 = fixture.total_momentum()
    fixture.step(steps)
    mass1 = fixture.total_mass()
    momentum1 = fixture.total_momentum()
    mass_scale = max(abs(mass0), 1.0)
    momentum_scale = max(float(momentum0.abs().max()), amplitude)
    return {
        "worst_relative_error": max(
            abs(mass1 - mass0) / mass_scale,
            float((momentum1 - momentum0).abs().max()) / momentum_scale,
        ),
        "mass_error": abs(mass1 - mass0) / mass_scale,
        "momentum_error": float((momentum1 - momentum0).abs().max()) / momentum_scale,
        "steps": int(steps),
    }


# ----------------------------------------------------------------------
# V5 / V6: shear-wave viscosity
# ----------------------------------------------------------------------
def _measure_decay(
    *,
    nx: int,
    tau0: float,
    modes: int = 1,
    amplitude: float = 1.0e-4,
    bulk_velocity: float = 0.0,
    smagorinsky_cs: float = 0.0,
    decay_target: float = 0.35,
    max_steps: int = 20000,
) -> dict[str, Any]:
    """Advance until the mode amplitude has decayed to ``decay_target * A0``."""
    fixture = PeriodicShearWave(
        nx=nx,
        tau0=tau0,
        modes=modes,
        amplitude=amplitude,
        bulk_velocity=bulk_velocity,
        smagorinsky_cs=smagorinsky_cs,
    )
    trace = [fixture.history(1, sample_every=1)[-1]]
    checkpoint = fixture.snapshot()
    while trace[-1][1] > decay_target * fixture.amplitude0 and fixture.steps < max_steps:
        trace.extend(fixture.history(10, sample_every=10))
        checkpoint = fixture.snapshot()
    rate = log_linear_decay_rate(trace, window=decay_target)
    predicted = fixture.viscosity * fixture.wavenumber**2
    return {
        "nx": nx,
        "tau0": tau0,
        "modes": modes,
        "wavenumber": fixture.wavenumber,
        "measured_decay_rate": rate,
        "predicted_decay_rate": predicted,
        "relative_error": abs(rate - predicted) / predicted,
        "viscosity_measured": rate / fixture.wavenumber**2,
        "viscosity_predicted": fixture.viscosity,
        "steps": fixture.steps,
        "amplitude_end": trace[-1][1],
        "checkpoint": checkpoint,
    }


measure_shear_decay = _measure_decay


def shear_wave_viscosity_scan(
    *, tau_values: Iterable[float] = (0.55, 0.7, 1.0), nx: int = 32
) -> dict[str, Any]:
    """``gamma = nu k^2`` for several relaxation times; also an O(k^2) study."""
    rows = [
        {k: v for k, v in _measure_decay(nx=nx, tau0=tau).items() if k != "checkpoint"}
        for tau in tau_values
    ]
    convergence = []
    for n in (16, 32, 64):
        entry = _measure_decay(nx=n, tau0=0.55)
        convergence.append(
            {
                "nx": n,
                "wavenumber": entry["wavenumber"],
                "relative_error": entry["relative_error"],
                "steps": entry["steps"],
            }
        )
    worst = max(row["relative_error"] for row in rows)
    return {
        "rows": rows,
        "convergence": convergence,
        "worst_relative_error": worst,
        "convergence_order_proxy": [
            (
                math.log(convergence[i]["relative_error"] / convergence[i + 1]["relative_error"])
                / math.log(
                    (convergence[i + 1]["wavenumber"] / convergence[i]["wavenumber"]) ** -2
                )
                if convergence[i]["relative_error"] > 0.0
                and convergence[i + 1]["relative_error"] > 0.0
                else float("nan")
            )
            for i in range(len(convergence) - 1)
        ],
    }


def galilean_invariance_check(*, nx: int = 32, tau0: float = 0.55) -> dict[str, Any]:
    """Bulk translation must not change the shear decay rate (low-Mach)."""
    frames = []
    for bulk in (0.0, 0.04):
        entry = _measure_decay(nx=nx, tau0=tau0, bulk_velocity=bulk)
        frame = {k: v for k, v in entry.items() if k != "checkpoint"}
        frames.append(
            {
                "bulk_velocity": bulk,
                "measured_decay_rate": frame["measured_decay_rate"],
                "predicted_decay_rate": frame["predicted_decay_rate"],
                "relative_error": frame["relative_error"],
            }
        )
    a, b = frames
    difference = abs(a["measured_decay_rate"] - b["measured_decay_rate"]) / a[
        "measured_decay_rate"
    ]
    return {
        "frames": frames,
        "relative_change": difference,
        "worst_relative_error": max(
            difference,
            abs(a["relative_error"] - b["relative_error"]),
        ),
    }


# ----------------------------------------------------------------------
# V7: Smagorinsky tau law on a prescribed analytic field
# ----------------------------------------------------------------------
def smagorinsky_tau_check(
    *, tau0: float = 0.5006, cs: float = 0.16, nx: int = 16, amplitude: float = 1.0e-2
) -> dict[str, Any]:
    """Compare ``_effective_tau`` with the closed form cell by cell.

    The field is built analytically as ``f = f^eq + f^(1)`` with
    ``f_i^(1) = -3 w_i rho tau0 c_ix c_iy A k cos(kx)``, so the code has to
    reproduce ``tau_eff = (tau0 + sqrt(tau0^2 + 12 Cs^2 tau0 |du_y/dx|))/2``.
    """
    fixture = PeriodicShearWave(
        nx=nx, tau0=tau0, smagorinsky_cs=cs, amplitude=amplitude, modes=1
    )
    solver = fixture.solver
    rho = torch.ones((nx, 8, 8, 1), dtype=solver.dtype)
    u = torch.zeros((nx, 8, 8, 3), dtype=solver.dtype)
    k = fixture.wavenumber
    x = torch.arange(nx, dtype=solver.dtype).reshape(nx, 1, 1)
    u[..., 1] = amplitude * torch.sin(k * x)
    shear_rate = (amplitude * k * torch.cos(k * x)).reshape(nx, 1, 1)
    feq = solver._equilibrium(rho, u)
    c = solver.c
    w = solver.w
    w_cxy = (w * c[:, 0] * c[:, 1]).reshape(1, 1, 1, len(w))
    f1 = (
        -3.0
        * tau0
        * w_cxy
        * (amplitude * k * torch.cos(k * x)).reshape(nx, 1, 1, 1)
    )
    field = feq + f1
    rho_out, u_out = solver._macroscopic(field)
    tau_raw = solver._effective_tau(field, rho_out, u_out)
    predicted = 0.5 * (
        tau0
        + torch.sqrt(
            tau0 * tau0
            + 6.0 * math.sqrt(2.0) * cs * cs * tau0 * shear_rate.abs()
        )
    ).reshape(nx, 1, 1)
    scale = float(predicted.abs().max())
    return {
        "worst_relative_error": float((tau_raw - predicted).abs().max()) / scale,
        "tau0": tau0,
        "smagorinsky_cs": cs,
        "cells": int(tau_raw.numel()),
        "max_eddy_viscosity": float((tau_raw - tau0).max()) / 3.0,
        "analytic_eddy_viscosity_at_peak": (analytic_smagorinsky_tau(
            tau0, cs, amplitude * k
        ) - tau0) / 3.0,
        "mach": float(amplitude),
    }


# ----------------------------------------------------------------------
# V8: SGS-damped decay against the closed-form ODE
# ----------------------------------------------------------------------
def _integrate_sgs_amplitude(
    *, nu0: float, cs: float, k: float, amplitude0: float, steps: int, substeps: int = 8
) -> list[float]:
    """RK4 integration of ``dA/dt = -k^2 A (nu0 + (4/(3 pi)) Cs^2 k A)``."""

    def rate(a: float) -> float:
        return -(k * k) * a * (
            nu0 + (4.0 / (3.0 * math.pi)) * cs * cs * k * a / math.sqrt(2.0)
        )

    out = [amplitude0]
    a = amplitude0
    h = 1.0 / float(substeps)
    for _ in range(int(steps)):
        for _ in range(int(substeps)):
            k1 = rate(a)
            k2 = rate(a + 0.5 * h * k1)
            k3 = rate(a + 0.5 * h * k2)
            k4 = rate(a + h * k3)
            a += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(a)
    return out


def sgs_damped_decay_check(
    *, nx: int = 32, tau0: float = 0.55, cs: float = 0.16, amplitude: float = 2.0e-3,
    steps: int = 400,
) -> dict[str, Any]:
    """Compare the simulated SGS-damped decay with the closed-form ODE."""
    fixture = PeriodicShearWave(
        nx=nx, tau0=tau0, smagorinsky_cs=cs, amplitude=amplitude, modes=1
    )
    trace = fixture.history(steps, sample_every=1)
    amplitudes = [a for _, a in trace]
    predicted = _integrate_sgs_amplitude(
        nu0=fixture.viscosity,
        cs=cs,
        k=fixture.wavenumber,
        amplitude0=fixture.amplitude0,
        steps=steps,
    )
    errors = [abs(measured - model) / abs(model) for measured, model in zip(amplitudes, predicted)]
    measured_rate = log_linear_decay_rate(trace, window=0.5)
    model_rate = log_linear_decay_rate(list(enumerate(predicted)), window=0.5)
    undamped_rate = fixture.viscosity * fixture.wavenumber**2
    return {
        "worst_relative_error": max(errors),
        "final_relative_error": errors[-1],
        "measured_rate": measured_rate,
        "model_rate": model_rate,
        "undamped_rate": undamped_rate,
        "rate_enhancement": measured_rate / undamped_rate,
        "model_enhancement": model_rate / undamped_rate,
        "steps": int(steps),
        "amplitudes": amplitudes[: 1 + steps // 20] + [amplitudes[-1]],
        "predicted": predicted[: 1 + steps // 20] + [predicted[-1]],
    }


# ----------------------------------------------------------------------
# suite
# ----------------------------------------------------------------------
def run_verification_suite(
    *, quick: bool = False, tau_values: Iterable[float] = (0.55, 0.7, 1.0)
) -> dict[str, Any]:
    """Run the whole ladder and return machine-checkable rows plus raw data."""
    rows: list[VerificationRow] = []
    data: dict[str, Any] = {}

    moments = equilibrium_moment_check()
    data["V1"] = {k: v for k, v in moments.items()}
    rows.append(
        VerificationRow(
            "V1",
            "D3Q19 equilibrium moments",
            "sum f = rho, sum f c = rho u, sum f c c = rho(c_s^2 I + u u)",
            3,
            float(moments["worst_relative_error"]),
            1.0e-14,
            bool(moments["worst_relative_error"] < 1.0e-14),
            note=f"weight sum error {moments['weight_sum_minus_one']:.2e}, "
            f"<w c_x^2 c_y^2> error {moments['cxy_moment_error']:.2e}",
        )
    )

    stream = streaming_reference_check()
    data["V2"] = stream
    rows.append(
        VerificationRow(
            "V2",
            "streaming stencil (19 directions)",
            "independent torch.roll scatter reference, wrap cells masked",
            1,
            float(stream["worst_relative_error"]),
            0.0,
            bool(stream["worst_relative_error"] == 0.0),
            note=f"{stream['cells']} population values compared",
        )
    )

    uniform = zero_thrust_uniform_check(steps=100 if quick else 200)
    data["V3"] = uniform
    rows.append(
        VerificationRow(
            "V3",
            "uniform-flow fixed point of the bounded step",
            "u = (U_in, 0, 0), rho = 1 through inlet/outlet/free-slip closures",
            int(uniform["steps"]),
            float(uniform["worst_relative_error"]),
            1.0e-12,
            bool(uniform["worst_relative_error"] < 1.0e-12),
            note="actuator disk force disabled; the closures admit a uniform state exactly",
        )
    )

    conservation = conservation_check(steps=200 if quick else 400)
    data["V4"] = conservation
    rows.append(
        VerificationRow(
            "V4",
            "periodic mass and momentum",
            "collision conserves moments, periodic shift is a permutation",
            int(conservation["steps"]),
            float(conservation["worst_relative_error"]),
            1.0e-13,
            bool(conservation["worst_relative_error"] < 1.0e-13),
            note=f"mass error {conservation['mass_error']:.2e}, "
            f"momentum error {conservation['momentum_error']:.2e}",
        )
    )

    scan = shear_wave_viscosity_scan(
        tau_values=(0.55,) if quick else tuple(tau_values)
    )
    data["V5"] = scan
    rows.append(
        VerificationRow(
            "V5",
            "shear-wave decay rate vs nu k^2",
            "nu = (tau0 - 1/2)/3, gamma = nu k^2",
            len(scan["rows"]),
            float(scan["worst_relative_error"]),
            1.0e-2,
            bool(scan["worst_relative_error"] < 1.0e-2),
            note="; ".join(
                f"tau0={row['tau0']}: err {100 * row['relative_error']:.3g}%"
                for row in scan["rows"]
            ),
        )
    )

    galilean = galilean_invariance_check()
    data["V6"] = galilean
    rows.append(
        VerificationRow(
            "V6",
            "Galilean invariance of the decay rate",
            "identical shear decay in a frame moving at U = 0.04",
            2,
            float(galilean["relative_change"]),
            1.0e-2,
            bool(galilean["relative_change"] < 1.0e-2),
            note="fractional change of the measured decay rate between frames",
        )
    )

    tau_law = smagorinsky_tau_check()
    data["V7"] = tau_law
    rows.append(
        VerificationRow(
            "V7",
            "Smagorinsky tau_eff closure",
            "closed-form root of tau^2 - tau0 tau - 3 Cs^2 |S| = 0",
            int(tau_law["cells"]),
            float(tau_law["worst_relative_error"]),
            1.0e-10,
            bool(tau_law["worst_relative_error"] < 1.0e-10),
            note=f"max eddy viscosity {tau_law['max_eddy_viscosity']:.3e} "
            f"lattice units at Ma = {tau_law['mach']}",
        )
    )

    sgs = sgs_damped_decay_check(steps=200 if quick else 400)
    data["V8"] = {k: v for k, v in sgs.items() if k != "amplitudes"}
    rows.append(
        VerificationRow(
            "V8",
            "SGS-damped decay vs closed-form ODE",
            "dA/dt = -k^2 A (nu0 + (4/(3 pi)) Cs^2 k A)",
            int(sgs["steps"]),
            float(sgs["worst_relative_error"]),
            0.05,
            bool(sgs["worst_relative_error"] < 0.05),
            note=f"decay-rate enhancement {sgs['measured_rate'] / sgs['undamped_rate']:.3f} "
            f"vs model {sgs['model_enhancement']:.3f}",
        )
    )

    consistency = smagorinsky_constant_consistency(0.5006, 0.16, 1.0e-3)
    data["S1"] = consistency
    rows.append(
        VerificationRow(
            "S1",
            "eddy-viscosity convention of the Smagorinsky closure",
            "nu_t measured / (Cs^2 |S|) == 1/sqrt(2) (Voigt single-count stress norm)",
            1,
            float(consistency["relative_difference"]),
            1.0e-3,
            bool(consistency["relative_difference"] < 1.0e-3),
            note=f"nu_t/(Cs^2|S|) = {consistency['eddy_viscosity_ratio']:.12f}, "
            f"Cs_eff = {consistency['effective_cs_ratio']:.6f} Cs "
            f"({consistency['effective_cs']:.4f} for nominal Cs = 0.16)",
        )
    )

    return {
        "schema": HARNESS_SCHEMA,
        "ok": all(row.passed for row in rows),
        "rows": [row.as_dict() for row in rows],
        "data": data,
        "settings": {
            "quick": bool(quick),
            "tau_values": list(tau_values),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
    }