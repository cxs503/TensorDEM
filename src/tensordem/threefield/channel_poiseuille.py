"""Planar Poiseuille flow: analytic parabola, periodic reference channel and the
bounded-box inlet/outlet precision face.

Why this module exists
----------------------
``boundary_verification.py`` (``docs/threefield_wake_boundary_verification.md``)
audited the *bounded-box* boundary rules of :mod:`tensordem.threefield.wake_lbm`
-- inlet Dirichlet, zero-gradient outlet, specular lateral walls -- but every
check there was a *consistency* check (a uniform stream is a fixed point, a
specular wall carries no tangential traction, the mass books close).  None of
them had a **spatial profile** to compare against, so the question "how much
error does the bounded box add to a developed profile?" was unanswered, and the
DAG node asked for exactly that: planar Poiseuille against its analytic
parabola, and periodic channel versus bounded-box zero-gradient / pressure-anchor
outlets.

What is measured
----------------
1. :func:`poiseuille_profile` / :func:`poiseuille_flux` -- the closed form for
   plane Poiseuille flow driven by a constant body force ``G`` between walls at
   ``y = -1/2`` and ``y = H - 1/2`` (the half-way bounce-back wall positions),
   ``u(y) = G/(2 nu) (y + 1/2)(H - 1/2 - y)``.
2. :class:`ChannelFlow` -- a D3Q19 BGK channel solver that drives the *shipped*
   kernels (``WakeLBM._equilibrium``, ``._macroscopic``, ``._effective_tau``) and
   reproduces the shipped collision + Guo-forcing update of ``WakeLBM.step`` line
   for line, so a defect found here is a defect of the production kernel.  Only
   the boundary layer is mirrored: ``wall="no_slip"`` is half-way bounce-back
   (which the shipped solver does *not* have), ``wall="free_slip"`` is the
   shipped specular rule, ``axial="bounded"`` uses the shipped inlet Dirichlet and
   outlet zero-gradient rules, and ``outlet="pressure_anchor"`` adds the density
   anchor the shipped outlet lacks.
3. :func:`run_channel` -- march to steady state and book the profile error against
   the parabola, the inlet/outlet mass-flux ratio, the mass drift and the density
   excursion.  This is the precision face: the periodic channel is the
   reference (it has no inlet/outlet at all), so the difference between a bounded
   run and the periodic run *is* the boundary error of that rule pair.

Geometry convention: ``ny`` fluid rows, walls midway between the ghost row and
row ``0`` / row ``ny-1``, so the wall-to-wall height is ``H = ny`` and the channel
half-width is ``H/2``.  The flow is invariant in ``z``; a one-cell ``z`` axis with
periodic wrapping is therefore exact (populations with ``c_z = +-1`` map onto
themselves), and the channel is a genuine 2-D problem at the cost of one plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch

from .wake_lbm import LBMConfig, WakeLBM

__all__ = [
    "ChannelConfig",
    "ChannelFlow",
    "ChannelResult",
    "poiseuille_profile",
    "poiseuille_flux",
    "poiseuille_peak",
    "run_channel",
    "plug_index",
    "profile_error",
]


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------
def poiseuille_profile(force: float, nu: float, ny: int) -> list[float]:
    """``u(y) = G/(2 nu) (y + 1/2)(H - 1/2 - y)`` at the lattice rows.

    The wall positions ``-1/2`` and ``H - 1/2`` are where half-way bounce-back
    puts them, so this expression -- not the textbook ``y(H-y)`` -- is the
    reference a second-order lattice solution can actually reach.
    """
    if force <= 0.0:
        raise ValueError("force must be positive")
    if nu <= 0.0:
        raise ValueError("viscosity must be positive")
    if ny < 4:
        raise ValueError("need at least four rows")
    h = float(ny)
    return [force / (2.0 * nu) * (y + 0.5) * (h - 0.5 - y) for y in range(ny)]


def poiseuille_flux(force: float, nu: float, ny: int) -> float:
    """Volume flux per unit width, ``G H^3 / (12 nu)`` (lattice cells^2 / step)."""
    if force <= 0.0 or nu <= 0.0:
        raise ValueError("force and viscosity must be positive")
    return force * float(ny) ** 3 / (12.0 * nu)


def poiseuille_peak(force: float, nu: float, ny: int) -> float:
    """Centreline velocity ``G H^2 / (8 nu)``."""
    if force <= 0.0 or nu <= 0.0:
        raise ValueError("force and viscosity must be positive")
    return force * float(ny) ** 2 / (8.0 * nu)


def profile_error(
    simulated: Iterable[float], reference: Iterable[float], *, peak: float | None = None
) -> dict[str, float]:
    """L-infinity, L2 and bias of ``simulated`` against ``reference``.

    Normalised by ``peak`` (the analytic or measured centreline value) so the
    number is a *relative* profile error, i.e. the quantity a boundary rule can be
    graded on.
    """
    sim = [float(value) for value in simulated]
    ref = [float(value) for value in reference]
    if len(sim) != len(ref):
        raise ValueError("profiles must have the same length")
    scale = peak if peak is not None else max(abs(value) for value in ref)
    if scale <= 0.0:
        raise ValueError("reference peak must be positive")
    diff = [a - b for a, b in zip(sim, ref)]
    return {
        "linf": max(abs(value) for value in diff) / scale,
        "l2": math.sqrt(sum(value * value for value in diff) / len(diff)) / scale,
        "bias": sum(diff) / len(diff) / scale,
        "peak": scale,
    }


def plug_index(profile: Iterable[float]) -> float:
    """``1 - max|u - mean|/mean``: exactly 1 for a plug.

    The maximum deviation of a parabola from its own mean sits at the *wall*, so
    the analytic no-slip profile scores ``1 - (mean - u_wall)/mean = 0.181`` here
    (``ny = 16``).  Used to show what a free-slip wall does to a profile: it
    cannot support a wall shear stress, so it relaxes the profile to the plug.
    """
    values = [float(value) for value in profile]
    mean = sum(values) / len(values)
    if mean <= 0.0:
        raise ValueError("plug_index needs a forward flow")
    return 1.0 - max(abs(value - mean) for value in values) / mean


# ---------------------------------------------------------------------------
# channel solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelConfig:
    """Geometry and physics of one channel run (lattice units)."""

    ny: int = 16
    nx: int = 48
    tau0: float = 0.8
    force: float = 6.25e-5
    axial: str = "periodic"
    outlet: str = "zero_gradient"
    wall: str = "no_slip"
    smagorinsky_cs: float = 0.0
    forcing: str = "simple"
    inlet_profile: str = "analytic"
    inlet_speed: float = 0.02
    inlet_pressure: float = 1.0
    initial: str = "rest"

    def __post_init__(self) -> None:
        if self.ny < 8 or self.nx < 8:
            raise ValueError("need at least eight rows and eight stations")
        if self.tau0 <= 0.5:
            raise ValueError("tau0 must exceed 0.5")
        if self.axial not in {"periodic", "bounded"}:
            raise ValueError("axial must be 'periodic' or 'bounded'")
        if self.outlet not in {"zero_gradient", "pressure_anchor"}:
            raise ValueError("outlet must be 'zero_gradient' or 'pressure_anchor'")
        if self.wall not in {"no_slip", "free_slip"}:
            raise ValueError("wall must be 'no_slip' or 'free_slip'")
        if self.axial == "periodic" and self.outlet != "zero_gradient":
            raise ValueError("a periodic channel has no outlet model")
        if self.force < 0.0:
            raise ValueError("force must be nonnegative")
        if self.initial not in {"rest", "inlet"}:
            raise ValueError("initial must be 'rest' or 'inlet'")
        if self.forcing not in {"simple", "guo"}:
            raise ValueError("forcing must be 'simple' (shipped) or 'guo' (consistent)")
        if self.inlet_profile not in {"analytic", "uniform"}:
            raise ValueError("inlet_profile must be 'analytic' or 'uniform'")

    @property
    def nu(self) -> float:
        return (self.tau0 - 0.5) / 3.0

    @property
    def height(self) -> float:
        return float(self.ny)

    def inlet_velocity(self) -> list[float]:
        """Prescribed inlet profile (per row)."""
        if self.inlet_profile == "uniform":
            mean = (
                poiseuille_peak(self.force, self.nu, self.ny) * 2.0 / 3.0
                if self.force > 0.0
                else float(self.inlet_speed)
            )
            return [mean] * self.ny
        return poiseuille_profile(self.force, self.nu, self.ny)


def _host(tau0: float, cs: float) -> WakeLBM:
    """A ``WakeLBM`` instance used only for its lattice tables and kernels."""
    config = LBMConfig(
        nx=16,
        ny=16,
        nz=16,
        tau0=tau0,
        smagorinsky_cs=cs,
        inflow_velocity=0.05,
        jet_velocity=0.06,
        disk_radius=1.0,
        disk_x=8,
        disk_thickness=1,
        quarter_domain=False,
        dtype="float64",
    )
    return WakeLBM(config)


class ChannelFlow:
    """D3Q19 BGK channel with selectable walls and axial boundary rules."""

    def __init__(self, config: ChannelConfig) -> None:
        self.config = config
        self.ny = int(config.ny)
        self.nx = int(config.nx)
        self.nz = 1
        self.host = _host(config.tau0, config.smagorinsky_cs)
        self.dirs = [(int(c[0]), int(c[1]), int(c[2])) for c in self.host.c.tolist()]
        self.opposite = [int(v) for v in self.host.opposite]
        self.reflect_y = [int(v) for v in self.host.reflect_y]
        rho = torch.ones((self.nx, self.ny, self.nz), dtype=self.host.dtype)
        u = torch.zeros((self.nx, self.ny, self.nz, 3), dtype=self.host.dtype)
        if config.initial == "inlet":
            # Start in the inlet state so a run measures the boundary model, not the
            # start-up transient (``initial="rest"`` is kept as the fill-time case).
            profile = torch.tensor(config.inlet_velocity(), dtype=self.host.dtype)
            u[:, :, :, 0] = profile.view(1, self.ny, 1)
        self.f = self.host._equilibrium(rho.unsqueeze(-1), u)
        self.steps = 0
        self.boundary_mass = 0.0  # net mass injected by the inlet/outlet rules
        self.force_momentum = 0.0  # cumulative body-force impulse in x

    # -- kernels -----------------------------------------------------------
    def _collide(self) -> torch.Tensor:
        """Shipped collision + Guo forcing (mirrors ``WakeLBM.step``)."""
        config = self.config
        rho, u = self.host._macroscopic(self.f)
        force = torch.zeros_like(u)
        if config.force > 0.0:
            force[:, :, :, 0] = rho * config.force
        self.force_momentum += float(force[:, :, :, 0].sum().item())
        u_forced = u + 0.5 * force / rho.unsqueeze(-1)
        tau = self.host._effective_tau(self.f, rho, u)
        feq = self.host._equilibrium(rho.unsqueeze(-1), u_forced)
        omega = 1.0 / tau.unsqueeze(-1)
        if config.forcing == "guo":
            # Consistent Guo source: the (1 - omega/2) prefactor is what makes the
            # discrete momentum balance read div(Pi) = F instead of
            # div(Pi) = (1 + omega/2) F.  The u-dependent terms are the second
            # half of the same scheme.
            cu = torch.einsum("ia,...a->...i", self.host.c, u_forced)
            cf = torch.einsum("ia,...a->...i", self.host.c, force)
            uf = (u_forced * force).sum(dim=-1, keepdim=True)
            source = self.host.w.view(1, 1, 1, -1) * (
                3.0 * cf + 9.0 * cu * cf - 3.0 * uf
            )
            forcing_term = (1.0 - 0.5 * omega) * source
        else:
            forcing_term = 3.0 * self.host.w.view(1, 1, 1, -1) * torch.einsum(
                "ia,...a->...i", self.host.c, force
            )
        return self.f - omega * (self.f - feq) + forcing_term

    def _stream(self, g: torch.Tensor) -> torch.Tensor:
        """Stream with periodic x/z, a bounded drop-out or a reflecting wall in y."""
        periodic = self.config.axial == "periodic"
        no_slip = self.config.wall == "no_slip"
        nz = self.nz
        out = torch.zeros_like(g)
        for i, (cx, cy, cz) in enumerate(self.dirs):
            plane = g[..., i]
            if cx and periodic:
                plane = torch.roll(plane, cx, dims=0)
            elif cx and cx > 0:
                # out[x] = plane[x - 1]; the population leaving the last station is lost
                plane = torch.cat((torch.zeros_like(plane[:1]), plane[:-1]), dim=0)
            elif cx and cx < 0:
                plane = torch.cat((plane[1:], torch.zeros_like(plane[:1])), dim=0)
            if cz and nz > 1:
                plane = torch.roll(plane, cz, dims=2)
            elif cz and nz == 1:
                # z-invariant flow: a one-cell z axis is mapped onto itself
                pass
            if cy > 0:
                out[:, 1:, :, i] = plane[:, :-1]
                if no_slip:
                    out[:, self.ny - 1, :, self.opposite[i]] += g[:, self.ny - 1, :, i]
            elif cy < 0:
                out[:, :-1, :, i] = plane[:, 1:]
                if no_slip:
                    out[:, 0, :, self.opposite[i]] += g[:, 0, :, i]
            else:
                out[:, :, :, i] = plane
        if not no_slip:
            # shipped specular rule (free-slip): mirror the wall-normal component
            for i, (_cx, cy, _cz) in enumerate(self.dirs):
                if cy > 0:
                    out[:, 0, :, i] = out[:, 0, :, self.reflect_y[i]]
                elif cy < 0:
                    out[:, self.ny - 1, :, i] = out[:, self.ny - 1, :, self.reflect_y[i]]
        return out

    def _apply_axial(self, f: torch.Tensor) -> torch.Tensor:
        """Inlet Dirichlet + outlet rule, exactly as ``WakeLBM._apply_boundaries``."""
        config = self.config
        if config.axial == "periodic":
            return f
        before = float(f[0].sum().item()) + float(f[-1].sum().item())
        profile = config.inlet_velocity()
        u = torch.zeros((self.ny, self.nz, 3), dtype=self.host.dtype)
        u[:, :, 0] = torch.tensor(profile, dtype=self.host.dtype).view(self.ny, 1)
        rho_in = torch.full((self.ny, self.nz, 1), config.inlet_pressure, dtype=self.host.dtype)
        f[0] = self.host._equilibrium(rho_in, u)
        for i, (cx, _cy, _cz) in enumerate(self.dirs):
            if cx < 0:
                f[self.nx - 1, :, :, i] = f[self.nx - 2, :, :, i]
        if config.outlet == "pressure_anchor":
            # Density-anchored outlet: the incoming populations are the equilibrium
            # of the extrapolated interior velocity at the anchor density, which is
            # the pressure/density constraint the zero-gradient outlet lacks.
            rho_int, u_int = self.host._macroscopic(f[self.nx - 2].unsqueeze(0))
            state = torch.zeros((self.ny, self.nz, 3), dtype=self.host.dtype)
            state[:, :, 0] = u_int[0, :, :, 0]
            state[:, :, 1] = u_int[0, :, :, 1]
            state[:, :, 2] = u_int[0, :, :, 2]
            del rho_int
            anchor = self.host._equilibrium(
                torch.full((self.ny, self.nz, 1), config.inlet_pressure, dtype=self.host.dtype),
                state,
            )
            for i, (cx, _cy, _cz) in enumerate(self.dirs):
                if cx < 0:
                    f[self.nx - 1, :, :, i] = anchor[:, :, i]
        self.boundary_mass += float(f[0].sum().item()) + float(f[-1].sum().item()) - before
        return f

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self.f = self._apply_axial(self._stream(self._collide()))
            self.steps += 1

    # -- diagnostics -------------------------------------------------------
    def mass(self) -> float:
        return float(self.f.sum().item())

    def density(self) -> torch.Tensor:
        return self.f.sum(dim=-1)

    def velocity(self) -> torch.Tensor:
        _rho, u = self.host._macroscopic(self.f)
        return u

    def profile(self, *, start: int = 0, stop: int | None = None) -> list[float]:
        """``u_x`` averaged over the axial band ``[start, stop)``, per row."""
        u = self.velocity()
        band = u[start:stop if stop is not None else self.nx, :, :, 0]
        return [float(value) for value in band.mean(dim=(0, 2)).tolist()]

    def plane_flux(self, station: int) -> float:
        """Net axial mass flux through ``station``, ``sum_i c_ix f_i``."""
        cx = self.host.c[:, 0].view(1, 1, -1)
        return float((self.f[station] * cx).sum().item())


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelResult:
    """Booked outcome of one channel run."""

    label: str
    config: ChannelConfig
    steps: int
    profile: tuple[float, ...]
    reference: tuple[float, ...]
    error: dict[str, float]
    flux_in: float
    flux_out: float
    flux_ratio: float
    mass_drift_rate: float
    density_excursion: float
    plug: float
    force_impulse: float
    boundary_mass: float
    station_profiles: dict[str, list[float]] = field(default_factory=dict)
    mass_history: tuple[tuple[int, float], ...] = ()
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "steps": self.steps,
            "axial": self.config.axial,
            "outlet": self.config.outlet,
            "wall": self.config.wall,
            "ny": self.config.ny,
            "nx": self.config.nx,
            "tau0": self.config.tau0,
            "nu": self.config.nu,
            "force": self.config.force,
            "profile": list(self.profile),
            "reference": list(self.reference),
            "error": dict(self.error),
            "flux_in": self.flux_in,
            "flux_out": self.flux_out,
            "flux_ratio": self.flux_ratio,
            "mass_drift_rate": self.mass_drift_rate,
            "density_excursion": self.density_excursion,
            "plug_index": self.plug,
            "boundary_mass": self.boundary_mass,
            "station_profiles": {k: list(v) for k, v in self.station_profiles.items()},
            "mass_history": [[step, mass] for step, mass in self.mass_history],
            **dict(self.extra),
        }


def run_channel(
    config: ChannelConfig,
    *,
    steps: int,
    label: str = "",
    sample_every: int = 50,
    developed_fraction: float = 0.4,
) -> ChannelResult:
    """March a channel to steady state and grade the developed profile.

    The profile is read over the downstream ``developed_fraction`` of the box so
    the inlet transient is excluded; the outlet is included deliberately, because
    the outlet model is one of the things under test.  ``mass_drift_rate`` is the
    relative mass change per step over the second half of the run, i.e. after the
    diffusive start-up; for a periodic channel it is a pure rounding diagnostic,
    for a bounded channel it measures whether the inlet/outlet pair balances.
    """
    if steps < 4 * sample_every:
        raise ValueError("run must cover several samples")
    flow = ChannelFlow(config)
    mass_start = flow.mass()
    density0 = float(flow.density().mean().item())
    halfway = steps // 2
    mass_half = None
    density_max = 0.0
    history: list[tuple[int, float]] = [(0, mass_start)]
    for step in range(1, int(steps) + 1):
        flow.step(1)
        if step == halfway:
            mass_half = flow.mass()
        if step % sample_every == 0:
            density_max = max(
                density_max, float((flow.density() - 1.0).abs().max().item())
            )
            history.append((step, flow.mass()))
    if mass_half is None:
        mass_half = flow.mass()
    mass_end = flow.mass()
    start = int(round((1.0 - developed_fraction) * flow.nx))
    profile = flow.profile(start=start)
    reference = (
        poiseuille_profile(config.force, config.nu, config.ny)
        if config.force > 0.0
        else [0.0] * config.ny
    )
    peak = poiseuille_peak(config.force, config.nu, config.ny) if config.force > 0.0 else 1.0
    error = profile_error(profile, reference, peak=peak) if config.force > 0.0 else {
        "linf": 0.0,
        "l2": 0.0,
        "bias": 0.0,
        "peak": peak,
    }
    flux_in = flow.plane_flux(0)
    flux_out = flow.plane_flux(flow.nx - 1)
    span = float(len(range(halfway + 1, steps + 1)))
    drift = (mass_end - mass_half) / span / mass_half
    return ChannelResult(
        label=label or f"{config.axial}-{config.outlet}-{config.wall}",
        config=config,
        steps=int(steps),
        profile=tuple(profile),
        reference=tuple(reference),
        error=error,
        flux_in=flux_in,
        flux_out=flux_out,
        flux_ratio=flux_out / flux_in if flux_in else float("nan"),
        mass_drift_rate=drift,
        density_excursion=max(density_max, abs(float(flow.density().mean().item()) - density0)),
        plug=plug_index(profile) if max(profile) > 0.0 else float("nan"),
        force_impulse=flow.force_momentum,
        boundary_mass=flow.boundary_mass,
        station_profiles={
            f"x{station}": [float(v) for v in flow.profile(start=station, stop=station + 1)]
            for station in sorted(
                {0, max(1, flow.nx // 4), flow.nx // 2, int(round(0.9 * (flow.nx - 1)))}
            )
        },
        mass_history=tuple(history),
        extra={
            "developed_start": float(start),
            "mass_start": mass_start,
            "mass_end": mass_end,
            "peak_measured": max(profile),
            "peak_analytic": peak,
        },
    )