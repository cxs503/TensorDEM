"""D3Q19 lattice-Boltzmann propeller-race solver with a Smagorinsky subgrid model.

The solver is a single-relaxation-time D3Q19 lattice Boltzmann core with

* Guo forcing for the propeller actuator disk (thrust imposed as an axial body
  force over a disk of prescribed diameter, which makes the race momentum flux
  directly available for the momentum-conservation audit),
* a Smagorinsky eddy-viscosity subgrid closure ::

      tau_eff = 0.5 ( tau0 + sqrt( tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho ) )

  where ``Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq)`` is the non-equilibrium
  momentum flux,
* free-slip (specular) lateral boundaries, which allow both a full box and a
  quarter domain cut along the two jet symmetry planes,
* a uniform-inflow / zero-gradient-outflow pair, and
* time averaging for mean profiles and turbulence intensity.

The empirical reference for the far-field race is the self-similar Gaussian
wake/jet model with momentum conservation (:func:`gaussian_race_profile`): the
excess momentum flux of the race equals the imposed thrust, the half-width grows
linearly with ``beta`` (``0.085`` s^-1 * x, the classic turbulent round-jet
spreading rate) and the centreline excess velocity follows from momentum
conservation rather than from a curve fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import torch

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------
_D3Q19_VELOCITIES = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)
_D3Q19_WEIGHTS = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)


def _opposite_index() -> tuple[int, ...]:
    lookup = {velocity: index for index, velocity in enumerate(_D3Q19_VELOCITIES)}
    return tuple(lookup[tuple(-component for component in velocity)] for velocity in _D3Q19_VELOCITIES)


def _reflect_index(axis: int) -> tuple[int, ...]:
    lookup = {velocity: index for index, velocity in enumerate(_D3Q19_VELOCITIES)}
    out = []
    for velocity in _D3Q19_VELOCITIES:
        flipped = list(velocity)
        flipped[axis] = -flipped[axis]
        out.append(lookup[tuple(flipped)])
    return tuple(out)


def _finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class LBMConfig:
    """Lattice and actuator-disk parameters for the propeller-race solve."""

    nx: int = 96
    ny: int = 24
    nz: int = 24
    tau0: float = 0.5006
    smagorinsky_cs: float = 0.16
    inflow_velocity: float = 0.04
    jet_velocity: float = 0.10
    disk_x: int = 8
    disk_radius: float = 5.0
    disk_thickness: int = 2
    inlet_perturbation: float = 0.02
    disk_profile: str = "parabolic"
    disk_perturbation: float = 0.15
    quarter_domain: bool = True
    dtype: str = "float64"

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz", "disk_x", "disk_thickness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.nx < 8 or self.ny < 8 or self.nz < 8:
            raise ValueError("domain must be at least 8 cells in every direction")
        if not 8 <= self.disk_x <= self.nx - 4:
            raise ValueError("disk_x must lie inside the domain, at least 4 cells from the outlet")
        if self.disk_thickness < 1:
            raise ValueError("disk_thickness must be >= 1")
        _finite("tau0", self.tau0, positive=True)
        if self.tau0 <= 0.5:
            raise ValueError("tau0 must exceed 0.5 for a positive viscosity")
        _finite("smagorinsky_cs", self.smagorinsky_cs, nonnegative=True)
        _finite("inflow_velocity", self.inflow_velocity, positive=True)
        _finite("jet_velocity", self.jet_velocity, positive=True)
        if self.jet_velocity >= 0.2:
            raise ValueError("jet_velocity must stay below 0.2 to keep the solver in the low-Mach limit")
        if self.jet_velocity <= self.inflow_velocity:
            raise ValueError("jet_velocity must exceed inflow_velocity for a propulsor race")
        _finite("disk_radius", self.disk_radius, positive=True)
        if self.disk_radius > min(self.ny, self.nz) / 2.0 - 1.0:
            raise ValueError("disk_radius leaves fewer than one free cell to the lateral wall")
        _finite("inlet_perturbation", self.inlet_perturbation, nonnegative=True)
        _finite("disk_perturbation", self.disk_perturbation, nonnegative=True)
        if self.disk_profile not in {"top_hat", "parabolic", "gaussian"}:
            raise ValueError("disk_profile must be 'top_hat', 'parabolic' or 'gaussian'")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    @property
    def viscosity(self) -> float:
        return (self.tau0 - 0.5) / 3.0

    @property
    def disk_diameter(self) -> float:
        return 2.0 * self.disk_radius


class WakeLBM:
    """D3Q19 + Smagorinsky propeller-race solver (lattice units)."""

    def __init__(self, config: LBMConfig = LBMConfig(), *, seed: int = 20261031) -> None:
        self.config = config
        self.dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.nx, self.ny, self.nz = config.nx, config.ny, config.nz
        self.c = torch.tensor(_D3Q19_VELOCITIES, dtype=self.dtype)
        self.w = torch.tensor(_D3Q19_WEIGHTS, dtype=self.dtype)
        self.opposite = _opposite_index()
        self.reflect_y = _reflect_index(1)
        self.reflect_z = _reflect_index(2)
        self._cx = [int(v) for v in self.c[:, 0].tolist()]
        self._cy = [int(v) for v in self.c[:, 1].tolist()]
        self._cz = [int(v) for v in self.c[:, 2].tolist()]
        self._outer = torch.tensor(
            [
                [ci[a] * ci[b] for (a, b) in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
                for ci in _D3Q19_VELOCITIES
            ],
            dtype=self.dtype,
        )
        self.generator = torch.Generator().manual_seed(int(seed))
        axis_y = 0.0 if config.quarter_domain else (self.ny - 1) / 2.0
        axis_z = 0.0 if config.quarter_domain else (self.nz - 1) / 2.0
        self._axis_y = (
            torch.arange(self.ny, dtype=self.dtype).reshape(1, self.ny, 1) - axis_y
        )
        self._axis_z = (
            torch.arange(self.nz, dtype=self.dtype).reshape(1, 1, self.nz) - axis_z
        )
        self.radius_field = torch.sqrt(self._axis_y**2 + self._axis_z**2)
        # Radial shape of the actuator-disk race velocity. The parabolic bell
        # (zero at the hub and at the tip) avoids the single-cell shear layer of
        # a top-hat disk, which otherwise sheds a strong starting vortex ring
        # that contaminates the near-field mean profile.
        if config.disk_profile == "gaussian":
            sigma0 = 0.5 * config.disk_radius
            shape = torch.exp(-(self.radius_field**2) / (2.0 * sigma0**2))
            shape = torch.where(self.radius_field <= config.disk_radius, shape,
                                torch.zeros_like(shape))
        elif config.disk_profile == "parabolic":
            shape = (1.0 - (self.radius_field / config.disk_radius) ** 2).clamp_min(0.0)
        else:
            shape = (self.radius_field <= config.disk_radius).to(self.dtype)
        self.disk_shape = shape
        self.disk_mask = torch.zeros((self.nx, self.ny, self.nz), dtype=torch.bool)
        thickness = int(config.disk_thickness)
        self.disk_mask[config.disk_x:config.disk_x + thickness] = (
            self.radius_field <= config.disk_radius
        )
        self.disk_cells = int(self.disk_mask.sum().item())
        if self.disk_cells == 0:
            raise ValueError("actuator disk resolved no lattice cells")
        self.rho = torch.ones((self.nx, self.ny, self.nz), dtype=self.dtype)
        self.u = torch.zeros((self.nx, self.ny, self.nz, 3), dtype=self.dtype)
        self.u[:, :, :, 0] = config.inflow_velocity
        self.f = self._equilibrium(self.rho.unsqueeze(-1), self.u)
        self.step_count = 0
        self.thrust_impulse = 0.0
        self.forcing_work = 0.0
        self._mean_u = torch.zeros_like(self.u)
        self._mean_u2 = torch.zeros_like(self.u)
        self._samples = 0

    # -- lattice kernels ---------------------------------------------------
    @property
    def cs2(self) -> float:
        return 1.0 / 3.0

    def _equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cu = torch.einsum("ia,...a->...i", self.c, u)
        u2 = (u * u).sum(dim=-1, keepdim=True)
        return rho * self.w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)

    def _macroscopic(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = f.sum(dim=-1)
        u = torch.einsum("ia,...i->...a", self.c, f) / rho.unsqueeze(-1)
        return rho, u

    def _effective_tau(self, f: torch.Tensor, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cs = self.config.smagorinsky_cs
        if cs <= 0.0:
            return torch.full_like(rho, self.config.tau0)
        f_neq = f - self._equilibrium(rho.unsqueeze(-1), u)
        pi = torch.einsum("ik,...i->...k", self._outer, f_neq)
        pi_norm = torch.sqrt((pi * pi).sum(dim=-1).clamp_min(0.0))
        tau0 = self.config.tau0
        discriminant = tau0 * tau0 + 18.0 * math.sqrt(2.0) * cs * cs * pi_norm / rho
        return 0.5 * (tau0 + torch.sqrt(discriminant.clamp_min(1e-30)))

    def _stream(self, f: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(f)
        nx, ny, nz = self.nx, self.ny, self.nz
        for i in range(len(self._cx)):
            cx, cy, cz = self._cx[i], self._cy[i], self._cz[i]
            src = (
                slice(max(-cx, 0), nx - max(cx, 0)),
                slice(max(-cy, 0), ny - max(cy, 0)),
                slice(max(-cz, 0), nz - max(cz, 0)),
            )
            dst = (
                slice(max(cx, 0), nx - max(-cx, 0)),
                slice(max(cy, 0), ny - max(-cy, 0)),
                slice(max(cz, 0), nz - max(-cz, 0)),
            )
            out[dst + (i,)] = f[src + (i,)]
        return out

    def _apply_boundaries(self, f: torch.Tensor) -> torch.Tensor:
        config = self.config
        u_in = config.inflow_velocity
        f[0, :, :, :] = self._inlet_populations()
        # Outlet: zero-gradient (convective) copy from the last interior plane.
        for i in range(len(self._cx)):
            if self._cx[i] < 0:
                f[self.nx - 1, :, :, i] = f[self.nx - 2, :, :, i]
        # Free-slip lateral walls (specular reflection of the normal component).
        for i in range(len(self._cx)):
            if self._cy[i] > 0:
                f[:, 0, :, i] = f[:, 0, :, self.reflect_y[i]]
            elif self._cy[i] < 0:
                f[:, self.ny - 1, :, i] = f[:, self.ny - 1, :, self.reflect_y[i]]
        for i in range(len(self._cx)):
            if self._cz[i] > 0:
                f[:, :, 0, i] = f[:, :, 0, self.reflect_z[i]]
            elif self._cz[i] < 0:
                f[:, :, self.nz - 1, i] = f[:, :, self.nz - 1, self.reflect_z[i]]
        del u_in
        return f

    def _inlet_populations(self) -> torch.Tensor:
        config = self.config
        u_in = config.inflow_velocity
        amplitude = config.inlet_perturbation
        if amplitude > 0.0:
            noise = torch.rand(
                (self.ny, self.nz, 3), generator=self.generator, dtype=self.dtype
            ) - 0.5
            noise = noise * (2.0 * amplitude * u_in)
        else:
            noise = torch.zeros((self.ny, self.nz, 3), dtype=self.dtype)
        u = torch.zeros((self.ny, self.nz, 3), dtype=self.dtype)
        u[:, :, 0] = u_in + noise[:, :, 0]
        u[:, :, 1] = noise[:, :, 1]
        u[:, :, 2] = noise[:, :, 2]
        rho = torch.ones((self.ny, self.nz, 1), dtype=self.dtype)
        return self._equilibrium(rho, u)

    def _disk_force(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Body force that drives the actuator-disk cells to the race velocity."""
        u_in = self.config.inflow_velocity
        delta = self.config.jet_velocity - u_in
        target = u_in + delta * self.disk_shape
        amplitude = self.config.disk_perturbation * delta
        if amplitude > 0.0:
            noise = torch.rand(
                (self.ny, self.nz), generator=self.generator, dtype=self.dtype
            ) - 0.5
            target = target + (2.0 * amplitude) * noise * self.disk_shape
        force = torch.zeros_like(u)
        current = u[:, :, :, 0]
        deficit = (target - current)[self.disk_mask]
        force[:, :, :, 0][self.disk_mask] = rho[self.disk_mask] * deficit
        return force

    # -- public API --------------------------------------------------------
    def step(self, n: int = 1, *, accumulate: bool = False) -> None:
        for _ in range(int(n)):
            rho, u = self._macroscopic(self.f)
            force = self._disk_force(rho, u)
            self.thrust_impulse += float(force[:, :, :, 0].sum().item())
            self.forcing_work += float((force * u).sum().item())
            u_forced = u + 0.5 * force / rho.unsqueeze(-1)
            tau = self._effective_tau(self.f, rho, u)
            feq = self._equilibrium(rho.unsqueeze(-1), u_forced)
            omega = 1.0 / tau.unsqueeze(-1)
            forcing_term = 3.0 * self.w.view(1, 1, 1, -1) * torch.einsum(
                "ia,...a->...i", self.c, force
            )
            self.f = (
                self.f
                - omega * (self.f - feq)
                + forcing_term
            )
            self.f = self._apply_boundaries(self._stream(self.f))
            self.rho, self.u = rho, u_forced
            self.step_count += 1
            if accumulate:
                self._mean_u += self.u
                self._mean_u2 += self.u * self.u
                self._samples += 1

    def run(self, steps: int, *, average_from: int | None = None) -> dict[str, Any]:
        """Advance ``steps`` iterations, averaging the last ``steps - average_from``."""
        if steps < 1:
            raise ValueError("steps must be positive")
        start = 0 if average_from is None else max(0, int(average_from))
        if start:
            self.step(start)
        self.step(steps - start, accumulate=True)
        return {
            "steps": self.step_count,
            "averaged_samples": self._samples,
            "thrust_impulse": self.thrust_impulse,
            "forcing_work": self.forcing_work,
        }

    def mean_fields(self) -> dict[str, torch.Tensor]:
        if self._samples == 0:
            raise RuntimeError("no time-averaged samples collected; call run() first")
        mean_u = self._mean_u / self._samples
        mean_u2 = self._mean_u2 / self._samples
        variance = (mean_u2 - mean_u * mean_u).clamp_min(0.0)
        return {
            "mean_u": mean_u,
            "variance": variance,
            "rms": torch.sqrt(variance),
        }

    def axial_velocity_profile(self, station: int) -> dict[str, Any]:
        """Azimuthally averaged mean axial velocity at the given x-plane."""
        mean_u = self.mean_fields()["mean_u"]
        profile = mean_u[station, :, :, 0]
        radius = self.radius_field[0]
        fields = self.mean_fields()
        rms_field = fields["rms"][station, :, :, 0]
        width = max(0.5, self.config.disk_radius / 4.0)
        bins: dict[int, list[float]] = {}
        rms_bins: dict[int, list[float]] = {}
        for r, u, rms in zip(
            radius.reshape(-1).tolist(),
            profile.reshape(-1).tolist(),
            rms_field.reshape(-1).tolist(),
        ):
            key = int(math.floor(r / width))
            bins.setdefault(key, []).append(u)
            rms_bins.setdefault(key, []).append(rms)
        keys = sorted(bins)
        radii = [key * width + 0.5 * width for key in keys]
        values = [sum(bins[key]) / len(bins[key]) for key in keys]
        rms_values = [sum(rms_bins[key]) / len(rms_bins[key]) for key in keys]
        return {"x": int(station), "radius": radii, "mean_axial_velocity": values,
                "axial_rms_velocity": rms_values}

    def half_width(self, station: int) -> float:
        """Radius where the time-averaged excess velocity drops to half its peak."""
        profile = self.axial_velocity_profile(station)
        radii = profile["radius"]
        values = profile["mean_axial_velocity"]
        u_in = self.config.inflow_velocity
        excess = [v - u_in for v in values]
        if not excess or max(excess) <= 0.0:
            return 0.0
        peak = max(excess)
        for radius, value in zip(radii, excess):
            if value <= 0.5 * peak:
                return float(radius)
        return float(radii[-1])

    def gaussian_fit(self, station: int) -> dict[str, float]:
        """Least-squares Gaussian fit ``U + A exp(-r^2/(2 sigma^2))`` of the race.

        The fit is performed on the resolved mean profile over the region where
        the excess velocity exceeds 15 % of its peak, which is the standard
        self-similar core of a jet/wake.
        """
        profile = self.axial_velocity_profile(station)
        radii = profile["radius"]
        u_in = self.config.inflow_velocity
        excess = [value - u_in for value in profile["mean_axial_velocity"]]
        peak = max(excess)
        if peak <= 0.0:
            raise ValueError("no positive race excess velocity at this station")
        pairs = [
            (r, e) for r, e in zip(radii, excess)
            if e >= 0.15 * peak and r > 0.0
        ]
        if len(pairs) < 3:
            raise ValueError("too few points above the 15 % threshold for a Gaussian fit")
        xs = [r * r for r, _ in pairs]
        ys = [math.log(e) for _, e in pairs]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx > 0 else 0.0
        intercept = mean_y - slope * mean_x
        sigma = math.sqrt(-1.0 / (2.0 * slope)) if slope < 0 else float("inf")
        amplitude = math.exp(intercept)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        return {
            "sigma": sigma,
            "amplitude": amplitude,
            "log_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "points": float(n),
            "half_width": sigma * math.sqrt(2.0 * math.log(2.0)),
        }

    def mass_flux(self, station: int) -> float:
        rho, _ = self._macroscopic(self.f)
        ux = self.u[station, :, :, 0]
        return float((rho[station] * ux).sum().item())

    def momentum_flux_excess(self, station: int) -> float:
        """``int rho u_x (u_x - U_in) dA`` in lattice units (area weight 1)."""
        rho, _ = self._macroscopic(self.f)
        ux = self.u[station, :, :, 0]
        u_in = self.config.inflow_velocity
        return float((rho[station] * ux * (ux - u_in)).sum().item())

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Serialisable solver state, allowing long runs to be chunked."""
        return {
            "f": self.f.clone(),
            "rho": self.rho.clone(),
            "u": self.u.clone(),
            "mean_u": self._mean_u.clone(),
            "mean_u2": self._mean_u2.clone(),
            "samples": self._samples,
            "step_count": self.step_count,
            "thrust_impulse": self.thrust_impulse,
            "forcing_work": self.forcing_work,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.f = state["f"].to(self.dtype).clone()
        self.rho = state["rho"].to(self.dtype).clone()
        self.u = state["u"].to(self.dtype).clone()
        self._mean_u = state["mean_u"].to(self.dtype).clone()
        self._mean_u2 = state["mean_u2"].to(self.dtype).clone()
        self._samples = int(state["samples"])
        self.step_count = int(state["step_count"])
        self.thrust_impulse = float(state["thrust_impulse"])
        self.forcing_work = float(state["forcing_work"])
        self.generator.set_state(state["generator_state"].clone())

    def thrust_lattice(self) -> float:
        """Net axial thrust accumulated over the run (lattice momentum per step)."""
        if self.step_count == 0:
            return 0.0
        return self.thrust_impulse / self.step_count


def gaussian_race_profile(
    radius: Iterable[float],
    *,
    inflow_velocity: float,
    thrust: float,
    distance: float,
    spreading_rate: float = 0.085,
    virtual_origin: float = 0.0,
    sigma_floor: float = 0.0,
    density: float = 1.0,
) -> dict[str, Any]:
    """Self-similar Gaussian propeller-race profile with momentum conservation.

    ``sigma(x) = spreading_rate * (x - virtual_origin) + sigma_floor`` and the
    centreline excess velocity solves
    ``2 pi rho (U + du/2) du sigma^2 = T`` exactly.
    """
    distance = float(distance)
    thrust = _finite("thrust", thrust, positive=True)
    u_in = _finite("inflow_velocity", inflow_velocity, nonnegative=True)
    sigma = spreading_rate * max(distance - virtual_origin, 0.0) + sigma_floor
    if sigma <= 0.0:
        raise ValueError("spreading model gives a non-positive half-width")
    delta = -u_in + math.sqrt(u_in * u_in + thrust / (math.pi * density * sigma * sigma))
    radii = list(radius)
    values = [delta * math.exp(-(r**2) / (2.0 * sigma * sigma)) for r in radii]
    momentum = 2.0 * math.pi * density * (u_in * delta * sigma**2 + 0.5 * delta**2 * sigma**2)
    return {
        "radius": radii,
        "excess_velocity": values,
        "axial_velocity": [u_in + v for v in values],
        "centreline_excess": delta,
        "sigma": sigma,
        "momentum_flux": momentum,
    }


def profile_error(
    simulated: list[float], reference: list[float], *, baseline: float | None = None
) -> dict[str, float]:
    """Relative L2 error of a simulated profile against a reference profile."""
    if len(simulated) != len(reference) or not simulated:
        raise ValueError("profiles must be non-empty and of equal length")
    residuals = [s - r for s, r in zip(simulated, reference)]
    ss_res = sum(r * r for r in residuals)
    mean_ref = sum(reference) / len(reference)
    ss_tot = sum((r - mean_ref) ** 2 for r in reference)
    scale = baseline if baseline is not None else math.sqrt(max(ss_tot, 1e-300))
    return {
        "relative_l2_error": math.sqrt(ss_res) / max(scale, 1e-300),
        "r_squared": 1.0 - ss_res / max(ss_tot, 1e-300),
        "rms_error": math.sqrt(ss_res / len(residuals)),
        "max_abs_error": max(abs(r) for r in residuals),
    }