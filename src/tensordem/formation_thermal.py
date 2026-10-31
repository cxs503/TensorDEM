"""Refreezing thermodynamics of an icebreaker channel (multi-ship formation).

Physics
-------
The lead ship of a formation clears a channel of width ``W`` through level ice.
The water in the channel is at the freezing point; the atmosphere is colder, so a
new ice sheet grows downward from the free surface.  Growth is a classical
one-phase (Stefan) phase-change problem: heat is conducted through the new ice
layer, latent heat is released at the ice/water interface, and the water column is
isothermal at the freezing point (standard Stefan assumption: the water has a
much larger thermal inertia and an O(1) water-side Nusselt number).

Two analytical references are provided for qualification:

* quasi-steady Stefan solution ``h = sqrt(2 k dT t / (rho L))`` (classical
  engineering formula, sensible heat of the new ice neglected);
* the exact one-phase Neumann similarity solution ``h = 2 lambda sqrt(alpha t)``
  with ``lambda exp(lambda^2) erf(lambda) = Ste / sqrt(pi)`` and
  ``Ste = c dT / L``, which retains the sensible heat of the new ice sheet.

The production model (:class:`ChannelRefreezing`) is a finite-volume solver on a
Landau-transformed (interface-fitted) grid.  It does not use either analytical
solution: it integrates Fourier's law plus the interfacial latent-heat balance and
is only *qualified* against them, which keeps the comparison an independent check.

All quantities are SI: m, s, K, W/(m K), J/(kg K), J/kg.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def _nonzero_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ChannelThermalConfig:
    """Thermophysical parameters of the channel refreezing problem.

    Defaults are sea-ice values: k = 2.2 W/(m K), rho = 917 kg/m3,
    L = 3.34e5 J/kg, c = 2.1e3 J/(kg K).
    """

    ice_conductivity: float = 2.2
    ice_density: float = 917.0
    latent_heat: float = 3.34e5
    specific_heat: float = 2.1e3
    air_temperature: float = -20.0
    water_temperature: float = 0.0
    nodes: int = 60
    # Effective latent heat of brash-filled channel water is reduced because the
    # brash fragments already carry part of the latent heat. Multiplier on L;
    # 1.0 = clean open water.
    brash_latent_heat_fraction: float = 1.0
    # Lateral (channel-wall) freezing uses the same Stefan law scaled by this
    # geometric shape factor: 1.0 = planar wall, the upper bound for a channel
    # much wider than the new ice thickness.
    lateral_shape_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("ice_conductivity", "ice_density", "latent_heat", "specific_heat"):
            if _nonzero_finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("air_temperature", "water_temperature"):
            _nonzero_finite(getattr(self, name), name)
        if self.air_temperature >= self.water_temperature:
            raise ValueError("air_temperature must be below water_temperature (freezing)")
        if isinstance(self.nodes, bool) or not isinstance(self.nodes, int) or self.nodes < 16:
            raise ValueError("nodes must be an integer >= 16")
        for name in ("brash_latent_heat_fraction", "lateral_shape_factor"):
            if _nonzero_finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def freezing_range(self) -> float:
        """Surface-to-water temperature difference ``dT`` in kelvin."""
        return self.water_temperature - self.air_temperature

    @property
    def thermal_diffusivity(self) -> float:
        return self.ice_conductivity / (self.ice_density * self.specific_heat)

    @property
    def effective_latent_heat(self) -> float:
        return self.latent_heat * self.brash_latent_heat_fraction

    @property
    def stefan_number(self) -> float:
        return self.specific_heat * self.freezing_range / self.effective_latent_heat


def stefan_thickness(t: float, config: ChannelThermalConfig) -> float:
    """Quasi-steady Stefan growth ``h = sqrt(2 k dT t / (rho L))``."""
    t = _nonzero_finite(t, "t")
    if t < 0:
        raise ValueError("t must be nonnegative")
    factor = 2.0 * config.ice_conductivity * config.freezing_range / (
        config.ice_density * config.effective_latent_heat)
    return math.sqrt(factor * t)


def similarity_lambda(stefan_number: float) -> float:
    """Solve ``lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)`` by bisection."""
    ste = _nonzero_finite(stefan_number, "stefan_number")
    if ste < 0:
        raise ValueError("stefan_number must be nonnegative")
    if ste == 0.0:
        return 0.0
    target = ste / math.sqrt(math.pi)
    lo, hi = 0.0, 1.0
    while hi * math.exp(hi * hi) * math.erf(hi) < target:
        hi *= 2.0
        if hi > 50.0:
            raise ValueError("similarity equation is not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(mid * mid) * math.erf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def neumann_thickness(t: float, config: ChannelThermalConfig) -> float:
    """Exact one-phase Neumann similarity thickness (sensible heat retained)."""
    t = _nonzero_finite(t, "t")
    if t < 0:
        raise ValueError("t must be nonnegative")
    return 2.0 * similarity_lambda(config.stefan_number) * math.sqrt(
        config.thermal_diffusivity * t)


class ChannelRefreezing:
    """One-phase Stefan solver for a refreezing icebreaker channel.

    The new ice occupies ``0 <= y <= h(t)`` measured downward from the free
    surface: ``y = 0`` is held at the air temperature, ``y = h`` at the freezing
    point.  The domain is mapped to the unit interval ``s = y / h(t)`` (Landau
    transform) so the moving interface sits on the last grid node:

        rho c (dT/dt|_s - (s hdot / h) dT/ds) = (k / h^2) d2T/ds2

    discretised with second-order central differences and backward Euler, with
    the interface speed iterated inside each step.  Interfacial heat balance:
    ``rho L hdot = k dT/dy`` evaluated at ``y = h``.

    Marching starts from a seed layer (1 mm by default) that is initialised with
    the exact similarity temperature profile, so the start-up transient is below
    the discretisation error.  The step size follows ``dt = 0.2 h^2 / alpha``,
    which keeps the interface advance below 3 % of the current thickness.
    """

    SEED_THICKNESS_M = 1.0e-3
    STEP_FACTOR = 0.2

    def __init__(self, config: ChannelThermalConfig | None = None, *,
                 initial_thickness: float = 0.0) -> None:
        self.config = config or ChannelThermalConfig()
        initial_thickness = _nonzero_finite(initial_thickness, "initial_thickness")
        if initial_thickness < 0:
            raise ValueError("initial_thickness must be nonnegative")
        self.initial_thickness = initial_thickness
        self._dtype = torch.float64
        self._s = torch.linspace(0.0, 1.0, self.config.nodes + 1, dtype=self._dtype)

    # -- analytical references -------------------------------------------------
    def stefan_analytic(self, t: float) -> float:
        """Quasi-steady Stefan reference thickness (engineering formula)."""
        return stefan_thickness(t, self.config)

    def neumann_analytic(self, t: float) -> float:
        """Exact similarity-solution thickness for the same material data."""
        return neumann_thickness(t, self.config)

    # -- initial state ---------------------------------------------------------
    def _seed_state(self) -> tuple[float, float, torch.Tensor]:
        """Return (start time, thickness, similarity temperature profile)."""
        cfg = self.config
        lam = similarity_lambda(cfg.stefan_number)
        if self.initial_thickness <= 0.0:
            thickness = self.SEED_THICKNESS_M
            start_time = (thickness / (2.0 * lam)) ** 2 / cfg.thermal_diffusivity
            profile = (cfg.air_temperature + cfg.freezing_range
                       * torch.erf(lam * self._s) / math.erf(lam))
            return start_time, thickness, profile
        thickness = self.initial_thickness
        # Linear conductive profile: representative of ice that grew to this
        # thickness before the formation entered the channel.
        profile = cfg.air_temperature + cfg.freezing_range * self._s
        return 0.0, thickness, profile

    def _march(self, thickness: float, profile: torch.Tensor, dt: float,
               rate: float) -> tuple[float, torch.Tensor, float]:
        """Advance one backward-Euler step; inner iteration on the interface rate."""
        cfg = self.config
        nodes = cfg.nodes
        s = self._s
        ds = 1.0 / nodes
        rho_c = cfg.ice_density * cfg.specific_heat
        new_thickness = thickness
        new_profile = profile
        for _ in range(4):
            h = max(thickness, 1.0e-9)
            diag = torch.full((nodes + 1,), 1.0, dtype=self._dtype)
            lower = torch.zeros_like(diag)
            upper = torch.zeros_like(diag)
            rhs = torch.zeros_like(diag)
            alpha = rho_c * h * h * nodes * nodes / (cfg.ice_conductivity * dt)
            adv = rho_c * h * rate / (cfg.ice_conductivity * nodes)
            k = torch.arange(1, nodes)
            lower[k] = -(1.0 + 0.5 * adv * s[k])
            upper[k] = -(1.0 - 0.5 * adv * s[k])
            diag[k] = 2.0 + alpha
            rhs[k] = alpha * profile[k]
            rhs[1] -= lower[1] * cfg.air_temperature
            rhs[nodes - 1] -= upper[nodes - 1] * cfg.water_temperature
            rhs[0] = cfg.air_temperature
            rhs[-1] = cfg.water_temperature
            solved = _solve_tridiagonal(lower, diag, upper, rhs)
            # dT/ds at s = 1 with a second-order one-sided stencil.
            gradient = (3.0 * solved[-1] - 4.0 * solved[-2] + solved[-3]) / (2.0 * ds)
            flux = cfg.ice_conductivity * float(gradient) / h  # dT/dy at y = h
            new_rate = flux / (cfg.ice_density * cfg.effective_latent_heat)
            new_thickness = max(thickness + dt * new_rate, 1.0e-9)
            new_profile = solved
            converged = abs(new_rate - rate) <= 1.0e-10 * max(1.0, abs(new_rate))
            rate = new_rate
            if converged:
                break
        return new_thickness, new_profile, rate

    def thickness_history(self, times: Sequence[float]) -> list[float]:
        """New-ice thickness at each requested time (s after channel clearance)."""
        values = [_nonzero_finite(t, "times") for t in times]
        if any(t < 0 for t in values):
            raise ValueError("times must be nonnegative")
        result = [0.0] * len(values)
        thickness = 0.0
        profile = self._s
        now = 0.0
        start_time, thickness, profile = self._seed_state()
        now = start_time
        rate = (thickness / (2.0 * now) if now > 0.0 else
                self.config.ice_conductivity * self.config.freezing_range
                / (self.config.ice_density * self.config.effective_latent_heat
                   * max(thickness, 1.0e-9)))
        for index in sorted(range(len(values)), key=lambda j: values[j]):
            target = values[index]
            if target < start_time:
                # Before the seed layer exists the similarity solution is exact.
                result[index] = self.neumann_analytic(target)
                continue
            while now < target - 1.0e-9:
                dt_limit = self.STEP_FACTOR * thickness * thickness / self.config.thermal_diffusivity
                dt = min(dt_limit, target - now)
                thickness, profile, rate = self._march(thickness, profile, dt, rate)
                now += dt
            result[index] = thickness
        return result

    def thickness(self, t: float) -> float:
        """New-ice thickness at time ``t`` (seconds after channel clearance)."""
        return self.thickness_history([t])[0]

    # -- channel geometry ------------------------------------------------------
    def lateral_growth(self, t: float) -> float:
        """Ice growth measured inward from each channel wall (m)."""
        return self.config.lateral_shape_factor * self.thickness(t)

    def channel_width(self, t: float, cleared_width: float) -> float:
        """Remaining open-water width of the channel at time ``t``."""
        cleared_width = _nonzero_finite(cleared_width, "cleared_width")
        if cleared_width <= 0:
            raise ValueError("cleared_width must be positive")
        return max(0.0, cleared_width - 2.0 * self.lateral_growth(t))

    def closure_time(self, cleared_width: float) -> float | None:
        """Time at which the channel closes laterally (``None`` if it never does)."""
        cleared_width = _nonzero_finite(cleared_width, "cleared_width")
        if cleared_width <= 0:
            raise ValueError("cleared_width must be positive")
        half = 0.5 * cleared_width / self.config.lateral_shape_factor
        lo, hi = 0.0, max(self.SEED_THICKNESS_M / 1.0e-3, 1.0)
        while self.thickness(hi) < half:
            hi *= 2.0
            if hi > 5.0e9:
                return None
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.thickness(mid) < half:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _solve_tridiagonal(lower: torch.Tensor, diag: torch.Tensor, upper: torch.Tensor,
                       rhs: torch.Tensor) -> torch.Tensor:
    """Thomas algorithm (backward-stable for the diagonally dominant Stefan matrix)."""
    n = diag.numel()
    c = upper.tolist()
    d = rhs.tolist()
    b = diag.tolist()
    a = lower.tolist()
    for i in range(1, n):
        if abs(b[i - 1]) < 1.0e-300:
            raise ZeroDivisionError("tridiagonal system is singular")
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = [0.0] * n
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return torch.tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Encountered ice thickness seen by a following ship
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelEncounterConfig:
    """Geometry of the channel cross-section as seen by a follower.

    Inside the cleared width the ice thickness is the refrozen layer plus a
    *brash* contribution: the lead ship does not leave clean open water but a
    brash-filled channel whose mechanical resistance is commonly represented as
    an equivalent level-ice thickness proportional to the parent thickness
    (:attr:`brash_equivalent_fraction`).  Outside the cleared width the parent
    sheet is undisturbed, and a ridge zone of width ``edge_zone_fraction * beam``
    separates the two.
    """

    brash_equivalent_fraction: float = 0.5
    edge_zone_fraction: float = 0.15
    edge_thickness_amplification: float = 1.6

    def __post_init__(self) -> None:
        for name in ("brash_equivalent_fraction", "edge_zone_fraction",
                     "edge_thickness_amplification"):
            value = _nonzero_finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.edge_thickness_amplification < 1.0:
            raise ValueError("edge_thickness_amplification must be >= 1")


def channel_thickness_profile(
    transverse_positions: Sequence[float],
    *,
    parent_thickness: float,
    channel_width: float,
    refrozen_thickness: float,
    beam: float,
    config: ChannelEncounterConfig | None = None,
) -> list[float]:
    """Level-ice-equivalent thickness along a transverse cut of the channel.

    ``transverse_positions`` are measured from the channel centreline (m).
    Inside ``|y| <= channel_width / 2`` the thickness is
    ``refrozen + brash_equivalent_fraction * parent``; across the ridge zone it
    ramps linearly up to ``edge_thickness_amplification * parent``; beyond that
    the undisturbed parent sheet thickness is returned.
    """
    cfg = config or ChannelEncounterConfig()
    parent_thickness = _nonzero_finite(parent_thickness, "parent_thickness")
    channel_width = _nonzero_finite(channel_width, "channel_width")
    refrozen_thickness = _nonzero_finite(refrozen_thickness, "refrozen_thickness")
    beam = _nonzero_finite(beam, "beam")
    if parent_thickness <= 0 or beam <= 0:
        raise ValueError("parent_thickness and beam must be positive")
    if channel_width < 0 or refrozen_thickness < 0:
        raise ValueError("channel_width and refrozen_thickness must be nonnegative")
    inner = refrozen_thickness + cfg.brash_equivalent_fraction * parent_thickness
    ridge = cfg.edge_thickness_amplification * parent_thickness
    zone = cfg.edge_zone_fraction * beam
    half = 0.5 * channel_width
    profile: list[float] = []
    for position in transverse_positions:
        y = abs(_nonzero_finite(position, "transverse_position"))
        if y <= half:
            profile.append(inner)
        elif zone <= 0.0 or y >= half + zone:
            profile.append(parent_thickness)
        else:
            ramp = (y - half) / zone
            profile.append(inner + ramp * (ridge - inner))
    return profile