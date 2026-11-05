"""Analytic verification of the D3Q19 propeller-race lattice-Boltzmann core.

Motivation
----------
``tensordem.threefield.wake_lbm`` is the lattice-Boltzmann solver that produces
the propeller race used by the three-field (ice-hull-propeller) coupling.  Until
this script existed its evidence was *conservation only*: the race momentum flux
was compared with the imposed thrust and the mean profile with the self-similar
Gaussian model.  Nothing bound the **effective viscosity**, the **speed of
sound** or the **equilibrium** of the kernel to a closed-form reference, even
though every lattice-to-SI conversion in the coupling depends on them.

This script supplies that missing reference.  It exercises the shipped kernels
(``_equilibrium``, ``_macroscopic``, ``_effective_tau``, ``_stream`` and the
free-slip reflection tables ``reflect_y``/``reflect_z``) inside a harness that
adds only **x periodicity** (the shipped boundary set is a wind-tunnel inlet /
outlet pair, which no analytic reference matches).  The driving boundary set is
therefore explicitly *out of scope* here and remains covered by the momentum /
mass conservation audit in ``docs/threefield/FORMULAS.md``.

Cases
-----
C1  Equilibrium moments.  For a random admissible population vector the shipped
    ``_equilibrium`` must reproduce ``rho``, ``rho*u`` and the momentum flux
    tensor ``rho*(cs^2 I + u u)``.

C2  Lattice isotropy of the shipped D3Q19 weight set:
    ``sum_i w_i = 1``, ``sum_i w_i c_ia c_ib = cs^2 delta_ab``,
    ``sum_i w_i c_ia c_ib c_ic c_id = cs^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc)``.

C3  Viscous decay of a free-slip shear mode (transverse wave).  With
    ``u_x(y,t) = A cos(K y)``, ``K = m pi / (ny-1)`` (so the wall-normal
    derivative vanishes at both free-slip walls) the incompressible
    Navier-Stokes solution is ``u_x = A cos(K y) exp(-nu K^2 t)`` exactly.
    The measured decay rate gives an independent value of the viscosity that
    must equal ``nu = (tau0 - 1/2)/3``.

C4  Diffusive scaling: the same measurement at mode ``m = 2`` must give the same
    viscosity (decay rate scaling as ``K^2``).

C5  Smagorinsky subgrid branch.  ``cs = 0`` must reproduce the molecular
    viscosity; ``cs > 0`` must increase the measured viscosity and the increase
    must grow with the shear amplitude (a nonlinear closure signature).

C6  Sound speed.  An x-periodic acoustic mode of wavenumber ``k`` oscillates at
    ``omega = cs_0 k`` with ``cs_0 = 1/sqrt(3)`` in lattice units.  This is the
    constant that fixes every lattice-to-SI length/time conversion.

C7  Conservation of the closed (periodic) kernel loop: total mass and total
    x-momentum drift over the whole run.

References
----------
Kruger, T. et al. (2017). *The Lattice Boltzmann Method: Principles and
    Practice*. Springer.  (Ch. 5: shear-wave viscosity; Ch. 9: Smagorinsky
    closure; lattice dispersion of the acoustic branch.)
Hou, S., Sterling, J., Chen, S., Doolen, G. (1996). A lattice Boltzmann
    subgrid model for high Reynolds number flows. Fields Inst. Commun. 6, 151.
Skordos, P.A. (1993). Initial and boundary conditions for the lattice
    Boltzmann method. Phys. Rev. E 48, 4823.  (Equilibrium second moments.)

Outputs
-------
``--output DIR`` receives ``wake_lbm_analytic.json`` (all numbers, no hand
copying), ``wake_lbm_analytic.csv`` (per-run series) and the PNG figures used by
``docs/wake_lbm_analytic_verification.md``.

Example:
    python scripts/verify_wake_lbm_analytic.py --output docs/artifacts/wake_lbm_analytic
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import torch

from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)


# ---------------------------------------------------------------------------
# harness: shipped kernels + x periodicity
# ---------------------------------------------------------------------------
class PeriodicShearHarness:
    """Shipped LBM kernels wrapped with x periodicity and free-slip y/z walls.

    Only the missing (periodic) x link is supplied here; collision, streaming,
    the Smagorinsky closure and the free-slip reflection tables all come from
    ``WakeLBM`` itself, so the numbers below test the shipped code.
    """

    def __init__(self, *, nx: int, ny: int, nz: int, tau0: float, cs_sgs: float) -> None:
        config = LBMConfig(
            nx=nx, ny=ny, nz=nz, tau0=tau0, smagorinsky_cs=cs_sgs,
            inflow_velocity=0.05, jet_velocity=0.10,
            disk_x=min(max(8, 8), nx - 4), disk_radius=2.0, disk_thickness=1,
            inlet_perturbation=0.0, disk_perturbation=0.0, dtype="float64",
        )
        self.solver = WakeLBM(config)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.trap = torch.ones(ny, dtype=torch.float64)
        self.trap[0] = self.trap[-1] = 0.5

    # -- streaming ---------------------------------------------------------
    def stream(self, f: torch.Tensor) -> torch.Tensor:
        """Shipped ``_stream`` plus the missing periodic x link.

        The wrap repeats the *same* y/z slicing as ``_stream`` so that
        populations already dropped by a y/z wall are not resurrected; that
        detail matters, because resurrecting them is a mass source (the naive
        ``out[0, :, :, i] = f[-1, :, :, i]`` form gains mass every step).
        """
        out = self.solver._stream(f)
        s = self.solver
        nx, ny, nz = self.nx, self.ny, self.nz
        for i in range(len(s._cx)):
            cx, cy, cz = s._cx[i], s._cy[i], s._cz[i]
            if cx == 0:
                continue
            dst_y = slice(max(cy, 0), ny - max(-cy, 0))
            src_y = slice(max(-cy, 0), ny - max(cy, 0))
            dst_z = slice(max(cz, 0), nz - max(-cz, 0))
            src_z = slice(max(-cz, 0), nz - max(cz, 0))
            if cx > 0:
                out[0, dst_y, dst_z, i] = f[nx - 1, src_y, src_z, i]
            else:
                out[nx - 1, dst_y, dst_z, i] = f[0, src_y, src_z, i]
        return out

    def free_slip(self, f: torch.Tensor) -> torch.Tensor:
        s = self.solver
        for i in range(len(s._cx)):
            if s._cy[i] > 0:
                f[:, 0, :, i] = f[:, 0, :, s.reflect_y[i]]
            elif s._cy[i] < 0:
                f[:, self.ny - 1, :, i] = f[:, self.ny - 1, :, s.reflect_y[i]]
        for i in range(len(s._cx)):
            if s._cz[i] > 0:
                f[:, :, 0, i] = f[:, :, 0, s.reflect_z[i]]
            elif s._cz[i] < 0:
                f[:, :, self.nz - 1, i] = f[:, :, self.nz - 1, s.reflect_z[i]]
        return f

    def step(self, f: torch.Tensor) -> torch.Tensor:
        return self.advance(f)[0]

    def advance(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """One collide -> periodic stream -> free-slip cycle (shipped kernels)."""
        s = self.solver
        rho, u = s._macroscopic(f)
        tau = s._effective_tau(f, rho, u)
        feq = s._equilibrium(rho.unsqueeze(-1), u)
        omega = 1.0 / tau.unsqueeze(-1)
        f = f - omega * (f - feq)
        f = self.stream(f)
        f = self.free_slip(f)
        return f, rho, u, float(tau.mean())

    # -- measurement -------------------------------------------------------
    def shear_amplitude(self, f: torch.Tensor, mode: int) -> float:
        """Trapezoid projection of u_x onto cos(K y), K = mode*pi/(ny-1)."""
        _, u = self.solver._macroscopic(f)
        return self.shear_amplitude_from(u, mode)

    def shear_amplitude_from(self, u: torch.Tensor, mode: int) -> float:
        k = math.pi * mode / (self.ny - 1)
        y = torch.arange(self.ny, dtype=torch.float64)
        basis = torch.cos(k * y)
        num = (self.trap * u[:, :, :, 0].mean(dim=(0, 2)) * basis).sum()
        den = (self.trap * basis * basis).sum()
        return float(num / den)

    def acoustic_modes(self, f: torch.Tensor) -> tuple[float, float]:
        """Amplitude of the cos(kx) component of rho and of u_x, k = 2 pi/nx."""
        rho, u = self.solver._macroscopic(f)
        k = 2.0 * math.pi / self.nx
        x = torch.arange(self.nx, dtype=torch.float64)
        basis = torch.cos(k * x)
        den = float((basis * basis).sum())
        amp_rho = float((rho.mean(dim=(1, 2)) * basis).sum() / den)
        amp_u = float((u[:, :, :, 0].mean(dim=(1, 2)) * basis).sum() / den)
        return amp_rho, amp_u


# ---------------------------------------------------------------------------
# C1 / C2: equilibrium moments and lattice isotropy
# ---------------------------------------------------------------------------
def equilibrium_moment_check() -> dict[str, Any]:
    s = WakeLBM(LBMConfig(nx=16, ny=16, nz=16, disk_radius=2.0, disk_x=8,
                          disk_thickness=1, dtype="float64"))
    gen = torch.Generator().manual_seed(12345)
    rho = 0.7 + 0.6 * torch.rand((4, 5, 6), generator=gen, dtype=torch.float64)
    u = 0.08 * (torch.rand((4, 5, 6, 3), generator=gen, dtype=torch.float64) - 0.5)
    feq = s._equilibrium(rho.unsqueeze(-1), u)
    rho_out, u_out = s._macroscopic(feq)
    # second moment: rho (cs^2 delta_ab + u_a u_b)
    c = s.c
    outer = torch.einsum("ia,ib->iab", c, c)
    m2 = torch.einsum("...i,iab->...ab", feq, outer)
    target = rho.unsqueeze(-1).unsqueeze(-1) * (
        CS2 * torch.eye(3, dtype=torch.float64) + torch.einsum("...a,...b->...ab", u, u)
    )
    return {
        "mass_error": float((rho_out - rho).abs().max()),
        "momentum_error": float((u_out - u).abs().max()),
        "moment_flux_error": float((m2 - target).abs().max()),
    }


def isotropy_check() -> dict[str, Any]:
    s = WakeLBM(LBMConfig(nx=16, ny=16, nz=16, disk_radius=2.0, disk_x=8,
                          disk_thickness=1, dtype="float64"))
    w, c = s.w, s.c
    delta = torch.eye(3, dtype=torch.float64)
    m0 = float(w.sum())
    m2 = torch.einsum("i,ia,ib->ab", w, c, c)
    m4 = torch.einsum("i,ia,ib,ic,id->abcd", w, c, c, c, c)
    target4 = CS2**2 * (
        torch.einsum("ab,cd->abcd", delta, delta)
        + torch.einsum("ac,bd->abcd", delta, delta)
        + torch.einsum("ad,bc->abcd", delta, delta)
    )
    return {
        "zeroth_error": abs(m0 - 1.0),
        "second_error": float((m2 - CS2 * delta).abs().max()),
        "fourth_error": float((m4 - target4).abs().max()),
    }


# ---------------------------------------------------------------------------
# C3 / C4 / C5: shear-mode decay -> viscosity
# ---------------------------------------------------------------------------
@dataclass
class ShearRun:
    tau0: float
    cs_sgs: float
    amplitude: float
    mode: int
    steps: int


def _fit_decay(times: list[float], amps: list[float]) -> tuple[float, float]:
    """Least-squares slope of ln|a| vs t, plus R^2 of that fit."""
    n = len(times)
    mean_t = sum(times) / n
    mean_y = sum(amps) / n
    sxx = sum((t - mean_t) ** 2 for t in times)
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in zip(times, amps))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_t
    ss_tot = sum((y - mean_y) ** 2 for y in amps)
    ss_res = sum((y - (intercept + slope * t)) ** 2 for t, y in zip(times, amps))
    return slope, 1.0 - ss_res / max(ss_tot, 1e-300)


def run_shear(run: ShearRun, *, ny: int = 16, nz: int = 16) -> dict[str, Any]:
    h = PeriodicShearHarness(nx=16, ny=ny, nz=nz, tau0=run.tau0, cs_sgs=run.cs_sgs)
    s = h.solver
    k = math.pi * run.mode / (ny - 1)
    y = torch.arange(ny, dtype=torch.float64)
    u = torch.zeros((16, ny, nz, 3), dtype=torch.float64)
    u[:, :, :, 0] = (run.amplitude * torch.cos(k * y)).reshape(1, ny, 1)
    rho = torch.ones((16, ny, nz), dtype=torch.float64)
    f = s._equilibrium(rho.unsqueeze(-1), u)
    mass0 = float(f.sum())
    mom0 = float(torch.einsum("ia,...i->...a", s.c, f)[..., 0].sum())

    times, amps = [], []
    tau_eff_sum = 0.0
    _, u0 = s._macroscopic(f)
    amps.append(h.shear_amplitude_from(u0, run.mode))
    times.append(0.0)
    for step in range(run.steps):
        f, _, u, tau_mean = h.advance(f)
        tau_eff_sum += tau_mean
        amps.append(h.shear_amplitude_from(u, run.mode))
        times.append(float(step + 1))

    logs = [math.log(max(a, 1e-300)) for a in amps]
    slope, r2 = _fit_decay(times, logs)
    nu_measured = -slope / (k * k)
    nu_molecular = (run.tau0 - 0.5) / 3.0
    mass1 = float(f.sum())
    mom1 = float(torch.einsum("ia,...i->...a", s.c, f)[..., 0].sum())
    return {
        "tau0": run.tau0,
        "cs_sgs": run.cs_sgs,
        "amplitude": run.amplitude,
        "mode": run.mode,
        "wavenumber": k,
        "steps": run.steps,
        "nu_molecular": nu_molecular,
        "nu_measured": nu_measured,
        "relative_error": (nu_measured - nu_molecular) / nu_molecular,
        "decay_fit_r2": r2,
        "mean_effective_tau": tau_eff_sum / run.steps,
        "steps_span": steps_to_span(nu_molecular, k, amps),
        "mass_drift": abs(mass1 - mass0) / mass0,
        "x_momentum_drift": abs(mom1 - mom0) / (mass0 * run.amplitude),
        "times": times,
        "amplitudes": amps,
    }


def steps_to_span(nu: float, k: float, amps: list[float]) -> float:
    """How many viscous time constants the run covered (diagnostic)."""
    if not amps or amps[0] <= 0.0:
        return 0.0
    return float(len(amps)) * nu * k * k


# ---------------------------------------------------------------------------
# C6: sound speed
# ---------------------------------------------------------------------------
def run_acoustic(*, tau0: float = 0.8, epsilon: float = 1e-4, steps: int = 400) -> dict[str, Any]:
    nx, ny, nz = 16, 16, 16
    h = PeriodicShearHarness(nx=nx, ny=ny, nz=nz, tau0=tau0, cs_sgs=0.0)
    s = h.solver
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, dtype=torch.float64)
    rho_prime = (epsilon * torch.cos(k * x)).reshape(nx, 1, 1)
    rho = (1.0 + rho_prime).expand(nx, ny, nz).contiguous()
    u = torch.zeros((nx, ny, nz, 3), dtype=torch.float64)
    f = s._equilibrium(rho.unsqueeze(-1), u)

    series = []
    for _ in range(steps):
        series.append(h.acoustic_modes(f)[0])
        f = h.step(f)

    # zero-crossing frequency estimate of the standing acoustic mode
    crossings = []
    for i in range(1, len(series)):
        a, b = series[i - 1], series[i]
        if a == 0.0:
            continue
        if a * b < 0.0:
            crossings.append(i - 1 + abs(a) / (abs(a) + abs(b)))
    if len(crossings) < 3:
        return {"omega_measured": float("nan"), "crossings": len(crossings)}
    half_periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    half = sum(half_periods) / len(half_periods)
    omega = math.pi / half
    return {
        "wavenumber": k,
        "omega_measured": omega,
        "omega_analytic": CS * k,
        "sound_speed_measured": omega / k,
        "sound_speed_analytic": CS,
        "relative_error": (omega / k - CS) / CS,
        "half_period_lattice": half,
        "crossings": len(crossings),
        "series": series,
    }


# ---------------------------------------------------------------------------
# C5: Smagorinsky closure
# ---------------------------------------------------------------------------
SGS_AMPLITUDES = (0.01, 0.05, 0.10)


def _manual_pi_norm(f_i, c, rho, u, nx, ny, nz) -> float:
    """Independent (pure-Python) evaluation of |Pi_ab| = |sum_i c_ia c_ib f_neq|.

    Deliberately written out direction by direction so that an index or
    normalisation slip in the vectorised kernel shows up as a mismatch.
    """
    total = 0.0
    neq = []
    u2 = sum(u[a] * u[a] for a in range(3))
    for index in range(19):
        cu = sum(c[index][a] * u[a] for a in range(3))
        feq = rho * _W[index] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        neq.append(f_i[index] - feq)
    # Frobenius norm Pi_ab Pi_ab over the six independent components (a <= b).
    # Pi is symmetric, so an off-diagonal slot stands for two entries of the
    # nine-slot double contraction and carries weight 2; without that weight the
    # norm is sqrt(2) too small in a pure shear. This is the convention of the
    # shipped kernel (``WakeLBM._effective_tau`` / ``_outer_norm_weight``, see
    # docs/threefield_lbm_transport_verification.md, m2949).
    for a in range(3):
        for b in range(a, 3):
            sab = sum(c[i][a] * c[i][b] * neq[i] for i in range(19))
            total += (1.0 if a == b else 2.0) * sab * sab
    return math.sqrt(total)


from tensordem.threefield.wake_lbm import _D3Q19_VELOCITIES, _D3Q19_WEIGHTS  # noqa: E402

_W = _D3Q19_WEIGHTS


def smagorinsky_closure_check(*, tau0: float = 0.5006, cs_sgs: float = 0.16,
                              ny: int = 16) -> dict[str, Any]:
    """Closure algebra, independent re-evaluation, and amplitude scaling.

    A collision on an *equilibrium* field returns that equilibrium, so the
    closure needs a genuinely non-equilibrium input: the field is produced by
    one streaming step of an analytic shear mode, which leaves the populations
    ``f_i(x) = feq_i(u(x - c_i))`` and therefore a non-zero ``Pi_ab``
    proportional to the strain rate (and hence to the amplitude ``A``).

    The code is compared with a direction-by-direction Python evaluation of the
    published Hou et al. (1996) closure on the *same* field: an implementation
    check (index/normalisation slips), not an independent physics reference.
    """
    harness = PeriodicShearHarness(nx=12, ny=ny, nz=ny, tau0=tau0, cs_sgs=cs_sgs)
    s = harness.solver
    k = math.pi / (ny - 1)
    y = torch.arange(ny, dtype=torch.float64)
    c_list = [[int(v) for v in row] for row in _D3Q19_VELOCITIES]

    tau_excess, max_rel, max_zero = [], 0.0, 0.0
    for amp in SGS_AMPLITUDES:
        u_field = torch.zeros((12, ny, ny, 3), dtype=torch.float64)
        u_field[:, :, :, 0] = (amp * torch.cos(k * y)).reshape(1, ny, 1)
        rho_field = torch.ones((12, ny, ny), dtype=torch.float64)
        f = s._equilibrium(rho_field.unsqueeze(-1), u_field)
        zero_stress = s._effective_tau(f, rho_field, u_field)
        max_zero = max(max_zero, float((zero_stress - tau0).abs().max()))

        f = harness.free_slip(harness.stream(f))
        rho_i, u_i = s._macroscopic(f)
        tau_code = s._effective_tau(f, rho_i, u_i)
        tau_excess.append(float((tau_code - tau0).mean().item()))

        i_x, i_y, i_z = 6, ny // 2, ny // 2
        f_i = [float(f[i_x, i_y, i_z, i].item()) for i in range(19)]
        u_node = [float(u_i[i_x, i_y, i_z, a].item()) for a in range(3)]
        rho_node = float(rho_i[i_x, i_y, i_z].item())
        pi_manual = _manual_pi_norm(f_i, c_list, rho_node, u_node, 12, ny, ny)
        # same closure algebra as the kernel: tau = (tau0 + sqrt(tau0^2
        # + 18 sqrt(2) Cs^2 |Pi| / rho)) / 2
        tau_manual = 0.5 * (tau0 + math.sqrt(
            tau0 * tau0 + 18.0 * math.sqrt(2.0) * cs_sgs ** 2 * pi_manual / rho_node))
        tau_node = float(tau_code[i_x, i_y, i_z].item())
        max_rel = max(max_rel, abs(tau_node - tau_manual) / tau_manual)

    logs_a = [math.log(a) for a in SGS_AMPLITUDES]
    logs_t = [math.log(t) for t in tau_excess]
    mean_a = sum(logs_a) / len(logs_a)
    mean_t = sum(logs_t) / len(logs_t)
    exponent = (sum((a - mean_a) * (t - mean_t) for a, t in zip(logs_a, logs_t))
                / sum((a - mean_a) ** 2 for a in logs_a))
    nu0 = (tau0 - 0.5) / 3.0
    return {
        "tau0": tau0,
        "cs_sgs": cs_sgs,
        "amplitudes": list(SGS_AMPLITUDES),
        "tau_excess": tau_excess,
        "eddy_viscosity": [t / 3.0 for t in tau_excess],
        "molecular_viscosity": nu0,
        "eddy_to_molecular": [t / 3.0 / nu0 for t in tau_excess],
        "fitted_exponent": exponent,
        "zero_stress_tau_error": max_zero,
        "max_relative_error": max_rel,
        "grading": "implementation consistency against an independent evaluation of "
                   "the published Hou et al. (1996) closure; the eddy viscosity is "
                   "evaluated on a known shear mode, not calibrated against a "
                   "turbulence measurement",
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
TAU_SERIES = (0.6, 0.8, 1.0, 1.2, 1.5)


def _steps_for(nu: float, k: float, spans: float = 3.0) -> int:
    return int(min(6000.0, max(120.0, math.ceil(spans / (nu * k * k)))))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="docs/artifacts/wake_lbm_analytic")
    parser.add_argument("--quick", action="store_true", help="shorter runs (test subset)")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ny = nz = 16
    k1 = math.pi / (ny - 1)

    report: dict[str, Any] = {}
    report["equilibrium_moments"] = equilibrium_moment_check()
    report["lattice_isotropy"] = isotropy_check()

    # C3: viscosity series (molecular)
    runs = []
    tau_list = (0.8, 1.2) if args.quick else TAU_SERIES
    for tau0 in tau_list:
        nu = (tau0 - 0.5) / 3.0
        steps = _steps_for(nu, k1)
        if args.quick:
            steps = max(60, steps // 4)
        runs.append(run_shear(ShearRun(tau0, 0.0, 0.01, 1, steps), ny=ny, nz=nz))
    report["viscosity_series"] = [{kk: vv for kk, vv in r.items() if kk not in ("times", "amplitudes")}
                                  for r in runs]
    report["_viscosity_series_raw"] = runs

    # C4: mode-2 scaling at tau0 = 1.0
    k2 = 2.0 * math.pi / (ny - 1)
    nu1 = (1.0 - 0.5) / 3.0
    mode2 = run_shear(ShearRun(1.0, 0.0, 0.01, 2, max(60, _steps_for(nu1, k2) // (4 if args.quick else 1))),
                      ny=ny, nz=nz)
    report["mode_scaling"] = {kk: vv for kk, vv in mode2.items() if kk not in ("times", "amplitudes")}
    report["_mode_scaling_raw"] = mode2

    # C5: Smagorinsky closure (algebraic + amplitude scaling, no time stepping)
    closing = smagorinsky_closure_check()
    report["smagorinsky"] = closing
    sgs_runs = []

    # C6: sound speed
    acoustic = run_acoustic(steps=200 if args.quick else 400)
    report["acoustic"] = {kk: vv for kk, vv in acoustic.items() if kk != "series"}
    report["_acoustic_raw"] = acoustic

    # ---- verdicts (explicit thresholds, no floating prose) ----------------
    verdicts = {}
    em = report["equilibrium_moments"]
    verdicts["equilibrium_exact"] = max(em["mass_error"], em["momentum_error"],
                                        em["moment_flux_error"]) < 1e-12
    iso = report["lattice_isotropy"]
    verdicts["isotropy_exact"] = max(iso["zeroth_error"], iso["second_error"],
                                     iso["fourth_error"]) < 1e-12
    verdicts["viscosity_within_3pct"] = all(abs(r["relative_error"]) < 0.03
                                            for r in report["viscosity_series"])
    verdicts["mode2_within_3pct"] = abs(report["mode_scaling"]["relative_error"]) < 0.03
    sm = report["smagorinsky"]
    verdicts["smagorinsky_zero_stress_limit"] = sm["zero_stress_tau_error"] < 1e-15
    verdicts["smagorinsky_matches_closed_form"] = sm["max_relative_error"] < 1e-12
    verdicts["smagorinsky_linear_in_amplitude"] = abs(sm["fitted_exponent"] - 1.0) < 0.02
    verdicts["sound_speed_within_3pct"] = abs(report["acoustic"].get("relative_error", 9.9)) < 0.03
    # closed-loop conservation: mass drift is round-off accumulation plus the
    # residue of the free-slip reflection, which conserves mass only when the
    # wall-normal velocity vanishes; 1e-11 over >=200 steps is ~1e-14/step.
    verdicts["conservation_1e_11"] = all(
        max(r["mass_drift"], r["x_momentum_drift"]) < 1e-11
        for r in report["viscosity_series"] + [report["mode_scaling"]]
    )
    report["verdicts"] = verdicts
    report["all_pass"] = all(verdicts.values())

    # ---- serialise -------------------------------------------------------
    series_rows = []
    for r in runs + [mode2] + sgs_runs:
        for t, a in zip(r["times"], r["amplitudes"]):
            series_rows.append({
                "tau0": r["tau0"], "cs_sgs": r["cs_sgs"], "amplitude": r["amplitude"],
                "mode": r["mode"], "step": t, "shear_amplitude": a,
            })
    with (out / "wake_lbm_analytic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series_rows[0]))
        writer.writeheader()
        writer.writerows(series_rows)

    payload = {kk: vv for kk, vv in report.items() if not kk.startswith("_")}
    with (out / "wake_lbm_analytic.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    _figures(out, runs, mode2, closing, acoustic)
    return payload


def _figures(out: pathlib.Path, runs, mode2, smagorinsky, acoustic) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - viz extra absent
        return

    # Fig 1: semi-log decay curves vs analytic envelope
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = ["-", "--", "-.", ":", "-"]
    for run, ls in zip(runs, styles):
        nu = run["nu_molecular"]
        k = run["wavenumber"]
        a0 = run["amplitudes"][0]
        t = run["times"]
        ax.semilogy(t, run["amplitudes"], ls, color="tab:blue", lw=1.2,
                    label=f"LBM tau0={run['tau0']:.1f}" if ls == "-" else None)
        ax.semilogy(t, [a0 * math.exp(-nu * k * k * ti) for ti in t], ls,
                    color="tab:red", lw=0.9, alpha=0.8,
                    label="analytic exp(-nu k^2 t)" if ls == "-" else None)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("shear-mode amplitude (lattice units)")
    ax.set_title("C3 free-slip shear-wave decay vs analytic solution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_shear_decay.png", dpi=140)
    plt.close(fig)

    # Fig 2: measured vs exact viscosity
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    taus = [r["tau0"] for r in runs] + [mode2["tau0"]]
    nus = [r["nu_measured"] for r in runs] + [mode2["nu_measured"]]
    ax.plot(taus, nus, "o", label="measured (shear decay)")
    grid = [0.55 + 0.02 * i for i in range(60)]
    ax.plot(grid, [(t - 0.5) / 3.0 for t in grid], "-", color="tab:red",
            label="analytic nu = (tau0-1/2)/3")
    for r in runs:
        ax.annotate(f"{100 * r['relative_error']:+.2f}%", (r["tau0"], r["nu_measured"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("relaxation time tau0")
    ax.set_ylabel("kinematic viscosity (lattice units)")
    ax.set_title("C3/C4 measured viscosity vs closed form")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_viscosity_vs_tau.png", dpi=140)
    plt.close(fig)

    # Fig 3: Smagorinsky closure scaling + acoustic mode
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    sm = smagorinsky
    amps = sm["amplitudes"]
    ax.loglog(amps, [max(v, 1e-300) for v in sm["tau_excess"]], "o-",
              label="tau_eff - tau0 (code)")
    ax.loglog(amps, [sm["tau_excess"][0] * (a / amps[0]) for a in amps], "--",
              color="tab:red", label="slope 1 reference (|Pi| ~ |S| ~ A)")
    ax.set_xlabel("shear amplitude A (lattice units)")
    ax.set_ylabel("mean(tau_eff - tau0)")
    ax.set_title(f"C5 Smagorinsky: fitted exponent {sm['fitted_exponent']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(acoustic["series"], lw=1.0)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("rho' cos(kx) mode amplitude")
    ax.set_title(f"C6 acoustic mode: c_s={acoustic['sound_speed_measured']:.4f} "
                 f"(1/sqrt3={CS:.4f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_sgs_and_acoustics.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: v for k, v in result.items() if k in ("verdicts", "all_pass")}, indent=2))
    for r in result["viscosity_series"]:
        print(f"tau0={r['tau0']:.2f} nu_exact={r['nu_molecular']:.6f} "
              f"nu_meas={r['nu_measured']:.6f} err={100*r['relative_error']:+.3f}%")