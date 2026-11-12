"""Propeller-ice milling transient dynamics (blade-ice contact, torque ripple, debris).

Campaign node IN27-S4.  Scope
-----------------------------
1. ``PropellerGeometry`` -- parametric helicoidal propeller discretised into
   radial x chordwise *contact panels* whose total area reproduces the stated
   expanded-area ratio (geometric gate, checked in the tests).
2. ``IceSlab`` / ``MillingCase`` -- bonded polar DEM ice cells at the propeller
   disc plane, each with mass, a Weibull crushing strength (mean from the
   temperature dependent ice strength law of :mod:`tensordem.rudder_ice_load`,
   scatter in the Timco-Jordan COV band) and a pulverisation state.
3. ``MillingSimulation`` -- explicit transient integration of
   * blade rotation (prescribed shaft speed) and ice-cell translation + spin,
   * blade-ice contact: Hertz normal branch capped by a crushing-pressure law
     ``p_crush * A_c`` with ``A_c = chord_panel * penetration``; Coulomb
     tangential friction with ``mu(v, T, p)`` from the calibrated
     :class:`tensordem.ice_steel_friction.IceSteelFrictionModel`; rolling
     resistance torque on the ice cell,
   * damage / pulverisation driven by the dissipated contact work against the
     crushing energy of the cell, mass-conserving conversion of a pulverised
     (or debonded) cell into debris fragments, and periodic re-seeding of fresh
     ice at the slab feed rate ``h_ice / V_cut``.
4. ``blade_element_open_water`` -- blade-element + momentum strip theory for the
   hydrodynamic thrust / torque the milling torque is superposed on.
5. ``RaceModel`` -- momentum-conserving self-similar propeller race (Gaussian
   profile, spreading rate 0.085 as in :mod:`tensordem.threefield.wake_lbm`).
6. ``integrate_debris`` -- quadratic-drag fragment trajectories; the *ejection
   speed* is the fragment speed where it crosses ``x = x_exit``.
7. ``torque_spectrum`` / ``evaluate_acceptance`` -- FFT peak identification
   (BPF and harmonics) and the machine-checkable acceptance table.

Honesty notes / limitations
---------------------------
* Contact law = plastic-capped Hertz: elastic Hertz branch for the equivalent
  sphere pair, cap = ice crushing pressure x loaded face area.  Crushing
  pressure equals the ice compressive strength at the cell temperature (no
  strain-rate factor) -- a documented idealisation.
* The ice slab is represented by one cell through the thickness; the slab feed
  is closed periodically at ``h_ice / V_cut`` so the process reaches a
  statistically stationary state instead of milling itself empty.  This is a
  periodic-feed closure, not a Lagrangian whole-sheet simulation.
* Hydrodynamic thrust/torque come from blade-element strip theory with a section
  polar (not a measured Wageningen-B table) and without hub/cavitation
  corrections: absolute values carry the usual +-15 % engineering-method
  uncertainty.  Absolute torque is not an acceptance criterion; the ripple
  ratio, spectral content and debris kinematics are.
* Debris hydrodynamics neglects gravity/buoyancy over the 1 m discharge length
  and treats fragments as non-interacting spheres (no fragment-fragment
  collisions) -- an O(1) assumption for dilute debris.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .hertz_mindlin import hertz_reduced_shear_modulus
from .ice_steel_friction import IceSteelFrictionModel, fit_ice_steel_friction
from .rudder_ice_load import IceStrengthTemperatureModel

RHO_ICE = 917.0
RHO_WATER = 1025.0
ICE_YOUNG_MODULUS = 5.0e9
ICE_POISSON = 0.3
STEEL_YOUNG_MODULUS = 2.10e11
STEEL_POISSON = 0.30
RACE_SPREADING_RATE = 0.085
MAX_THREADS = 4


def configure_threads(max_threads: int = MAX_THREADS) -> int:
    """Cap the torch thread pool for the element-wise milling kernels.

    The milling step launches many small (~1e4 element) element-wise kernels;
    on a many-core host (e.g. 128-core loong64) the per-kernel thread launch
    overhead dominates and a single case becomes ~20x slower.  The cap is only
    applied when the pool is larger than ``max_threads``.
    """
    current = torch.get_num_threads()
    if current > max_threads:
        torch.set_num_threads(max_threads)
    return torch.get_num_threads()


# ---------------------------------------------------------------------------
# propeller geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PropellerGeometry:
    """Parametric helicoidal propeller with a discretised blade surface.

    Planform ``c(xi) = c_max ((xi (1 - xi)) ** e) / 0.615`` with
    ``xi = (r - r_hub)/(R - r_hub)``; ``c_max`` is fixed by requiring
    ``Z * integral c dr = EAR * pi D^2 / 4`` so the panel set reproduces the
    expanded-area ratio.
    """

    diameter: float = 3.0
    n_blades: int = 4
    hub_ratio: float = 0.40
    expanded_area_ratio: float = 0.70
    pitch_ratio: float = 0.95
    rake_deg: float = 6.0
    thickness_ratio: float = 0.050
    n_radial_panels: int = 8
    n_chordwise_panels: int = 3
    planform_exponent: float = 0.35

    def __post_init__(self) -> None:
        for name in ("diameter", "hub_ratio", "expanded_area_ratio", "pitch_ratio",
                     "thickness_ratio", "planform_exponent", "rake_deg"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in ("diameter", "hub_ratio", "expanded_area_ratio", "pitch_ratio",
                     "thickness_ratio", "planform_exponent"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.hub_ratio < 1.0:
            raise ValueError("hub_ratio must be in (0, 1)")
        if self.n_blades < 2:
            raise ValueError("n_blades must be >= 2")
        if self.n_radial_panels < 1 or self.n_chordwise_panels < 1:
            raise ValueError("panel counts must be >= 1")

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def hub_radius(self) -> float:
        return 0.5 * self.hub_ratio * self.diameter

    @property
    def pitch(self) -> float:
        return self.pitch_ratio * self.diameter

    @property
    def blade_area(self) -> float:
        return self.expanded_area_ratio * math.pi * self.diameter ** 2 / 4.0 / self.n_blades

    @property
    def total_blade_area(self) -> float:
        return self.n_blades * self.blade_area

    def pitch_angle(self, r: float) -> float:
        return math.atan(self.pitch / (2.0 * math.pi * max(r, 1e-9)))

    def _planform(self, xi: float) -> float:
        xi = min(max(xi, 0.0), 1.0)
        return (xi * (1.0 - xi)) ** self.planform_exponent / 0.615

    def chord_scale(self) -> float:
        n = 2000
        r0, r1 = self.hub_radius, self.radius
        dr = (r1 - r0) / n
        acc = 0.0
        for i in range(n):
            r = r0 + (i + 0.5) * dr
            acc += self._planform((r - r0) / (r1 - r0)) * dr
        return self.blade_area / acc

    def chord(self, r: float) -> float:
        xi = (r - self.hub_radius) / (self.radius - self.hub_radius)
        return self.chord_scale() * self._planform(xi)

    def thickness(self, r: float) -> float:
        return self.thickness_ratio * self.chord(r)

    def panels(self) -> dict[str, torch.Tensor]:
        """Contact panels in the blade frame at blade angle zero (float64)."""
        n_r, n_c = self.n_radial_panels, self.n_chordwise_panels
        r0, r1 = self.hub_radius, self.radius
        edges = torch.linspace(r0, r1, n_r + 1, dtype=torch.float64)
        dr = float(edges[1] - edges[0])
        rs, thetas, half_thick, chords, areas, phis, blade_idx = [], [], [], [], [], [], []
        for k in range(self.n_blades):
            for i in range(n_r):
                r = float(edges[i]) + 0.5 * dr
                c = self.chord(r)
                phi = self.pitch_angle(r)
                for j in range(n_c):
                    xi_c = (j + 0.5) / n_c
                    dtheta = (xi_c - 0.5) * c * math.cos(phi) / r
                    rs.append(r)
                    thetas.append(k * 2.0 * math.pi / self.n_blades + dtheta)
                    half_thick.append(0.5 * self.thickness(r))
                    chords.append(c / n_c)
                    areas.append(c / n_c * dr)
                    phis.append(phi)
                    blade_idx.append(k)
        return {
            "r": torch.tensor(rs, dtype=torch.float64),
            "theta0": torch.tensor(thetas, dtype=torch.float64),
            "half_thickness": torch.tensor(half_thick, dtype=torch.float64),
            "chord": torch.tensor(chords, dtype=torch.float64),
            "area": torch.tensor(areas, dtype=torch.float64),
            "pitch_angle": torch.tensor(phis, dtype=torch.float64),
            "blade": torch.tensor(blade_idx, dtype=torch.int64),
        }


# ---------------------------------------------------------------------------
# ice slab
# ---------------------------------------------------------------------------
def weibull_shape_for_cov(cov: float) -> float:
    """Weibull shape ``k`` matching a target coefficient of variation."""
    if not 0.0 < cov < 1.0:
        raise ValueError("cov must be in (0, 1)")
    lo, hi = 0.5, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g1 = math.gamma(1.0 + 1.0 / mid)
        g2 = math.gamma(1.0 + 2.0 / mid)
        if math.sqrt(g2 / (g1 * g1) - 1.0) > cov:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class IceSlab:
    """Polar cell decomposition of the ice annulus facing the propeller."""

    n_radial_cells: int = 6
    n_azimuth_cells: int = 40
    coverage: float = 0.40
    crushed_layer: float = 0.002
    strength_cov: float = 0.30
    crush_strain: float = 0.005
    bond_stiffness: float = 5.0e6
    bond_damping: float = 2.0e4
    bond_strength_fraction: float = 0.40
    rolling_resistance: float = 0.05

    def __post_init__(self) -> None:
        if self.n_radial_cells < 1 or self.n_azimuth_cells < 3:
            raise ValueError("need at least one radial and three azimuth cells")
        if not 0.0 < self.strength_cov < 1.0:
            raise ValueError("strength_cov must be in (0, 1)")
        if not 0.0 < self.coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        if self.crushed_layer <= 0.0:
            raise ValueError("crushed_layer must be positive")
        if self.crush_strain <= 0.0 or self.bond_stiffness <= 0.0:
            raise ValueError("crush_strain and bond_stiffness must be positive")

    def build(self, geometry: PropellerGeometry, ice_thickness: float,
              temperature_c: float, *, seed: int = 0) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(int(seed))
        r0, r1 = geometry.hub_radius, geometry.radius
        edges = torch.linspace(r0, r1, self.n_radial_cells + 1, dtype=torch.float64)
        # the ice sheet covers only part of the disc (a wide floe field at the
        # propeller plane); `coverage` is the azimuthal fraction actually loaded
        dtheta = 2.0 * math.pi * self.coverage / self.n_azimuth_cells
        rs, thetas, masses, radii, faces = [], [], [], [], []
        for i in range(self.n_radial_cells):
            ra, rb = float(edges[i]), float(edges[i + 1])
            r = 0.5 * (ra + rb)
            dr = rb - ra
            for j in range(self.n_azimuth_cells):
                theta = (j + 0.5) * dtheta
                volume = dr * r * dtheta * ice_thickness
                rs.append(r)
                thetas.append(theta)
                masses.append(RHO_ICE * volume)
                radii.append((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))
                faces.append(dr * r * dtheta)
        n = len(rs)
        sigma_mean = IceStrengthTemperatureModel().sigma_pa(temperature_c)
        k = weibull_shape_for_cov(self.strength_cov)
        lam = sigma_mean / math.gamma(1.0 + 1.0 / k)
        u = torch.rand(n, dtype=torch.float64, generator=gen).clamp_min(1e-9)
        strengths = lam * (-torch.log(u)) ** (1.0 / k)
        r_t = torch.tensor(rs, dtype=torch.float64)
        th_t = torch.tensor(thetas, dtype=torch.float64)
        mass = torch.tensor(masses, dtype=torch.float64)
        return {
            "pos": torch.stack([r_t * torch.cos(th_t), r_t * torch.sin(th_t)], dim=1),
            "radius": torch.tensor(radii, dtype=torch.float64),
            "mass": mass,
            "strength": strengths,
            "r": r_t,
            "volume": mass / RHO_ICE,
            "face_area": torch.tensor(faces, dtype=torch.float64),
            "sigma_mean": torch.tensor(sigma_mean, dtype=torch.float64),
        }


# ---------------------------------------------------------------------------
# open-water strip theory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionPolar:
    """Linear-lift / quadratic-drag section polar (documented idealisation)."""

    lift_slope: float = 5.7
    zero_lift_alpha: float = 0.035
    drag_floor: float = 0.012
    drag_factor: float = 0.030

    def coefficients(self, alpha: float) -> tuple[float, float]:
        a = min(max(alpha, -0.45), 0.45)
        cl = min(max(self.lift_slope * (a - self.zero_lift_alpha), -1.30), 1.40)
        cd = self.drag_floor + self.drag_factor * cl * cl
        return cl, cd


def blade_element_open_water(geometry: PropellerGeometry, n_rps: float,
                             ship_speed: float, *, n_strips: int = 24,
                             polar: SectionPolar | None = None) -> dict[str, float]:
    """Open-water thrust / torque from blade-element + momentum strip theory.

    Per strip the axial induction ``a`` solves
    ``dT_BE(a) = 4 pi rho r V^2 a (1 + a) dr`` by bisection and the swirl
    induction ``a'`` follows from angular momentum.  Returns ``K_T``, ``K_Q``,
    open-water efficiency and the dimensional thrust / torque.
    """
    polar = polar or SectionPolar()
    rho = RHO_WATER
    if not (math.isfinite(n_rps) and n_rps > 0.0):
        raise ValueError("n_rps must be finite and positive")
    if not (math.isfinite(ship_speed) and ship_speed > 1.0e-3):
        raise ValueError("ship_speed must be positive (milling advance is non-zero)")
    omega = 2.0 * math.pi * n_rps
    r0, r1 = geometry.hub_radius, geometry.radius
    dr = (r1 - r0) / n_strips
    thrust = 0.0
    torque = 0.0
    for i in range(n_strips):
        r = r0 + (i + 0.5) * dr
        c = geometry.chord(r)
        phi_g = geometry.pitch_angle(r)

        def section(a: float) -> tuple[float, float]:
            v_a = ship_speed * (1.0 + a)
            w2 = v_a * v_a + (omega * r) ** 2
            phi = math.atan2(v_a, omega * r)
            cl, cd = polar.coefficients(phi_g - phi)
            dq = 0.5 * rho * w2 * c * (cl * math.sin(phi) + cd * math.cos(phi)) * r
            a_prime = dq / (4.0 * math.pi * rho * ship_speed * omega * r ** 3 * (1.0 + a))
            a_prime = min(max(a_prime, 0.0), 0.95)
            vt = omega * r * (1.0 - a_prime)
            w2 = v_a * v_a + vt * vt
            phi = math.atan2(v_a, vt)
            cl, cd = polar.coefficients(phi_g - phi)
            d_t = 0.5 * rho * w2 * c * (cl * math.cos(phi) - cd * math.sin(phi)) * dr
            d_q = 0.5 * rho * w2 * c * (cl * math.sin(phi) + cd * math.cos(phi)) * r * dr
            return d_t, d_q

        lo, hi = 0.0, 0.75
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            momentum = 4.0 * math.pi * rho * r * ship_speed ** 2 * mid * (1.0 + mid) * dr
            if section(mid)[0] > momentum:
                lo = mid
            else:
                hi = mid
        d_t, d_q = section(0.5 * (lo + hi))
        thrust += d_t
        torque += d_q
    k_t = thrust / (rho * n_rps ** 2 * geometry.diameter ** 4)
    k_q = torque / (rho * n_rps ** 2 * geometry.diameter ** 5)
    advance = ship_speed / (n_rps * geometry.diameter)
    eta = advance * k_t / (2.0 * math.pi * k_q) if k_q > 0.0 else 0.0
    return {
        "advance_ratio": advance,
        "K_T": k_t,
        "K_Q": k_q,
        "efficiency": eta,
        "thrust": thrust,
        "torque": torque,
    }


# ---------------------------------------------------------------------------
# propeller race
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RaceModel:
    """Momentum-conserving self-similar propeller race.

    ``u(x, r) = u_c(x) exp(-r^2 / (2 sigma^2))``, ``sigma = R/2 + beta x`` and
    ``u_c = sqrt(M / (pi sigma^2))`` with momentum flux ``M = T / rho``.  Because
    ``int_0^inf rho u^2 2 pi r dr = rho u_c^2 pi sigma^2 = rho M = T``, the profile
    conserves momentum flux exactly at every station, which is the physical gate
    applied in the tests (``G2_momentum_flux_*``).

    History: the normalisation was ``u_c = sqrt(M / (2 pi sigma^2))``, which
    delivers ``T/2`` (DEFECT-D1, repaired in this revision).  Note that the repair
    is a factor ``sqrt(2)``, not the ``sqrt(2 M / (pi sigma^2))`` factor*2 written
    in the first issue of ``docs/propeller_ice_milling_legacy_verification.md``:
    the acceptance gate ``int rho u^2 2 pi r dr = T`` is the arbiter.
    """

    thrust: float
    diameter: float
    spreading_rate: float = RACE_SPREADING_RATE

    def __post_init__(self) -> None:
        if not math.isfinite(self.thrust):
            raise ValueError("thrust must be finite")
        if not (math.isfinite(self.diameter) and self.diameter > 0.0):
            raise ValueError("diameter must be positive")
        if not (math.isfinite(self.spreading_rate) and self.spreading_rate > 0.0):
            raise ValueError("spreading_rate must be positive")

    @property
    def momentum_flux(self) -> float:
        """``M = T / rho`` -- the flux the axial-velocity field must deliver."""
        return max(self.thrust, 1.0e-6) / RHO_WATER

    @property
    def wake_half_width(self) -> float:
        """Wake width ``sigma`` at ``x = 0`` (``R/2``); the spreading origin."""
        return 0.25 * self.diameter

    def sigma(self, x: float) -> float:
        return self.wake_half_width + self.spreading_rate * max(x, 0.0)

    def sigma_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorised :meth:`sigma` (same convention, same origin)."""
        return self.wake_half_width + self.spreading_rate * x.clamp_min(0.0)

    def centreline(self, x: float) -> float:
        s = self.sigma(x)
        return math.sqrt(self.momentum_flux / (math.pi * s * s))

    def axial_velocity_tensor(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Vectorised :meth:`axial_velocity` -- the single source of truth for the race.

        ``integrate_debris`` used to carry its own ``sigma = 0.5 D`` closure, i.e.
        twice the wake width and a four-times weaker centreline (DEFECT-D2).  Every
        consumer of the race field must go through this method.
        """
        x = x.to(torch.float64)
        r = r.to(torch.float64)
        s = self.sigma_tensor(x)
        u_c = torch.sqrt(torch.as_tensor(self.momentum_flux, dtype=torch.float64)
                         / (math.pi * s * s))
        return u_c * torch.exp(-(r * r) / (2.0 * s * s))

    def axial_velocity(self, x: float, r: float) -> float:
        s = self.sigma(x)
        return self.centreline(x) * math.exp(-(r * r) / (2.0 * s * s))

    def induced_velocity(self, ship_speed: float) -> float:
        area = math.pi * (0.5 * self.diameter) ** 2
        return -0.5 * ship_speed + math.sqrt(
            0.25 * ship_speed ** 2 + self.momentum_flux / (2.0 * area))


# ---------------------------------------------------------------------------
# milling case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MillingCase:
    rpm: float = 60.0
    ice_thickness: float = 0.30
    ice_temperature_c: float = -5.0
    ship_speed: float = 1.50
    revolutions: float = 2.0
    dt: float = 2.0e-4
    x_exit: float = 1.0
    seed: int = 20261106
    geometry: PropellerGeometry = field(default_factory=PropellerGeometry)
    slab: IceSlab = field(default_factory=IceSlab)

    def __post_init__(self) -> None:
        for name in ("rpm", "ice_thickness", "revolutions", "dt", "x_exit"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (math.isfinite(self.ship_speed) and self.ship_speed >= 0.0):
            raise ValueError("ship_speed must be finite and non-negative")
        if not self.ice_temperature_c < 0.0:
            raise ValueError("ice_temperature_c must be below freezing")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def n_rps(self) -> float:
        return self.rpm / 60.0

    @property
    def blade_passing_frequency(self) -> float:
        return self.geometry.n_blades * self.n_rps

    @property
    def duration(self) -> float:
        return self.revolutions / self.n_rps


@dataclass
class MillingResult:
    """Time histories and derived quantities of one milling case."""

    case: MillingCase
    time: torch.Tensor
    torque_ice: torch.Tensor
    thrust_ice: torch.Tensor
    contact_force: torch.Tensor
    active_contacts: torch.Tensor
    debris_time: torch.Tensor
    debris_mass_each: torch.Tensor
    debris_initial: torch.Tensor
    debris_exit_speed: torch.Tensor
    debris_exit_radius: torch.Tensor
    contact_pairs: torch.Tensor
    hydro: dict[str, float]
    race: RaceModel
    diagnostics: dict[str, Any]

    @property
    def total_torque(self) -> torch.Tensor:
        """Shaft torque: hydrodynamic drive minus the ice milling resistance."""
        return self.hydro["torque"] - self.torque_ice

    def summary(self) -> dict[str, Any]:
        q_total = self.total_torque
        q_mean = float(q_total.mean())
        q_lo = float(torch.quantile(q_total, 0.05))
        q_hi = float(torch.quantile(q_total, 0.95))
        n = int(self.debris_exit_speed.numel())
        return {
            "rpm": self.case.rpm,
            "ice_thickness": self.case.ice_thickness,
            "n_rps": self.case.n_rps,
            "bpf_hz": self.case.blade_passing_frequency,
            "advance_ratio": self.hydro["advance_ratio"],
            "K_T": self.hydro["K_T"],
            "K_Q": self.hydro["K_Q"],
            "efficiency": self.hydro["efficiency"],
            "hydro_thrust": self.hydro["thrust"],
            "hydro_torque": self.hydro["torque"],
            "torque_mean": q_mean,
            "torque_ice_mean": float(self.torque_ice.mean()),
            "torque_ice_peak": float(self.torque_ice.max()),
            "ice_torque_fraction": float(self.torque_ice.mean()) / q_mean if q_mean else 0.0,
            "ripple_robust": (q_hi - q_lo) / q_mean,
            "ripple_p2p": float(q_total.max() - q_total.min()) / q_mean,
            "ripple_std": float(q_total.std(unbiased=True)) * 2.0 * math.sqrt(2.0) / q_mean,
            "thrust_ice_mean": float(self.thrust_ice.mean()),
            "net_thrust": self.hydro["thrust"] - float(self.thrust_ice.mean()),
            "contact_force_mean": float(self.contact_force.mean()),
            "contact_force_peak": float(self.contact_force.max()),
            "active_pairs_mean": float(self.contact_pairs.mean()),
            "debris_count": n,
            "debris_mass_total": float(self.debris_mass_each.sum()) if n else 0.0,
            "ejection_speed_mean": float(self.debris_exit_speed.mean()) if n else 0.0,
            "ejection_speed_p10": float(torch.quantile(self.debris_exit_speed, 0.10)) if n else 0.0,
            "ejection_speed_p90": float(torch.quantile(self.debris_exit_speed, 0.90)) if n else 0.0,
            "ejection_radius_mean": float(self.debris_exit_radius.mean()) if n else 0.0,
            "race_induced_velocity": self.race.induced_velocity(self.case.ship_speed),
            "x_exit": self.case.x_exit,
            "sigma_mean_pa": self.diagnostics["sigma_mean_pa"],
            "energy_j": self.diagnostics["energy_j"],
            "renewals": self.diagnostics["renewals"],
            "bond_broken_mass": self.diagnostics["bond_broken_mass"],
            "n_steps": self.diagnostics["n_steps"],
            "n_panels": self.diagnostics["n_panels"],
            "n_cells": self.diagnostics["n_cells"],
        }


class MillingSimulation:
    """Explicit propeller-ice milling integrator (SI units, float64).

    Contact model: each blade *panel* (a chord strip of one radial station) is a
    circumferential patch of azimuthal half-width
    ``0.5 * chord_panel * cos(phi) + half_thickness``.  It touches an ice cell
    when the swept arc between the panel and the cell centre is smaller than
    that half-width, i.e. only the *leading* part of the blade is in contact --
    the trailing chord sweeps through the kerf the leading edge has just opened.
    The loaded depth is the arc overlap, capped by the crushed-layer thickness.
    """

    def __init__(self, case: MillingCase, *, friction: IceSteelFrictionModel | None = None,
                 max_debris: int = 500) -> None:
        configure_threads()
        self.case = case
        self.geometry = case.geometry
        self.friction = friction or fit_ice_steel_friction()
        self.max_debris = int(max_debris)
        self.panels = self.geometry.panels()
        self.cells = case.slab.build(self.geometry, case.ice_thickness,
                                     case.ice_temperature_c, seed=case.seed)
        n = self.n_cells = int(self.cells["pos"].shape[0])
        self.anchor = self.cells["pos"].clone()
        self.mass = self.cells["mass"]
        self.radius = self.cells["radius"]
        self.volume = self.cells["volume"]
        self.face_area = self.cells["face_area"]
        self.state = {
            "pos": self.cells["pos"].clone(),
            "vel": torch.zeros(n, 2, dtype=torch.float64),
            "spin": torch.zeros(n, dtype=torch.float64),
            "damage": torch.zeros(n, dtype=torch.float64),
            "alive": torch.ones(n, dtype=torch.bool),
            "strength": self.cells["strength"].clone(),
            "renew_at": torch.full((n,), float("inf"), dtype=torch.float64),
        }
        self._contact_force = torch.zeros(n, 2, dtype=torch.float64)
        self._last_bond = torch.zeros(n, dtype=torch.float64)
        self.e_star = 1.0 / ((1.0 - ICE_POISSON ** 2) / ICE_YOUNG_MODULUS
                             + (1.0 - STEEL_POISSON ** 2) / STEEL_YOUNG_MODULUS)
        self.g_star = hertz_reduced_shear_modulus(ICE_YOUNG_MODULUS, ICE_POISSON,
                                                  rigid_counterpart=True)
        self._gen = torch.Generator().manual_seed(int(case.seed) + 7919)
        self._debris: list[dict[str, float]] = []
        self._debris_total = 0
        self._pulverised = 0.0
        self._bond_broken = 0.0
        self._renewals = 0
        self._energy = {"friction": 0.0, "plastic": 0.0, "rolling": 0.0}
        self._build_pair_list()

    def _build_pair_list(self) -> None:
        """Static radial-proximity candidate list (pure sparsity optimisation)."""
        case = self.case
        r_panel = self.panels["r"]
        r_cell = self.cells["r"]
        half_thick = self.panels["half_thickness"]
        d_cell = (float(r_cell.max()) - float(r_cell.min())) / max(case.slab.n_radial_cells - 1, 1)
        near = (r_panel[:, None] - r_cell[None, :]).abs() <= (0.5 * d_cell + half_thick[:, None])
        ip, ic = near.nonzero(as_tuple=True)
        pos = self.cells["pos"]
        theta_cell = torch.atan2(pos[ic, 1], pos[ic, 0])
        cos_t, sin_t = torch.cos(theta_cell), torch.sin(theta_cell)
        phi = self.panels["pitch_angle"]
        self._pair = {
            "ip": ip,
            "ic": ic,
            "theta0": self.panels["theta0"][ip],
            "theta_cell": theta_cell,
            "e_theta": torch.stack([-sin_t, cos_t], dim=1),
            "e_r": torch.stack([cos_t, sin_t], dim=1),
            "r_cell": r_cell[ic],
            "az_half": 0.5 * self.panels["chord"][ip] * torch.cos(phi[ip]) + half_thick[ip],
            "chord": self.panels["chord"][ip],
            "a_p": half_thick[ip],
            "a_c": self.radius[ic],
            "sin_phi": torch.sin(phi[ip]),
            "cos_phi": torch.cos(phi[ip]),
        }

    def _panel_state(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        omega = 2.0 * math.pi * self.case.n_rps
        theta = self.panels["theta0"] + omega * t
        r = self.panels["r"]
        pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        vel = torch.stack([-omega * pos[:, 1], omega * pos[:, 0]], dim=1)
        return pos, vel

    def step(self, t: float, dt: float) -> dict[str, float]:
        slab = self.case.slab
        case = self.case
        h_ice = case.ice_thickness
        omega = 2.0 * math.pi * case.n_rps
        pair = self._pair
        cell_vel = self.state["vel"]
        alive = self.state["alive"]
        zeros2 = torch.zeros(self.n_cells, 2, dtype=torch.float64)

        theta_panel = pair["theta0"] + omega * t
        two_pi = 2.0 * math.pi
        dtheta = torch.remainder(pair["theta_cell"] - theta_panel + math.pi, two_pi) - math.pi
        arc = dtheta * pair["r_cell"]
        overlap = pair["az_half"] - arc
        active = (overlap > 0.0) & alive[pair["ic"]]

        if not bool(active.any()):
            self._contact_force = zeros2
            self._advance(dt, zeros2, torch.zeros(self.n_cells, dtype=torch.float64))
            self._break_and_pulverise(t, dt)
            self._apply_renewal(t)
            return {"torque": 0.0, "thrust": 0.0, "force": 0.0, "pairs": 0.0}

        idx_c = pair["ic"][active]
        d = overlap[active]
        chord = pair["chord"][active]
        a_p = pair["a_p"][active]
        a_c = pair["a_c"][active]
        e_theta = pair["e_theta"][active]
        e_r = pair["e_r"][active]
        r_c = pair["r_cell"][active]
        sin_phi = pair["sin_phi"][active]
        cos_phi = pair["cos_phi"][active]

        # loaded face: chord strip x crushed-layer depth (deeper arc penetration
        # is expelled as debris and transmits no extra force)
        loaded_depth = torch.clamp(d, max=min(h_ice, slab.crushed_layer))
        area_c = chord * loaded_depth
        r_star = a_c * a_p / (a_c + a_p)
        f_hertz = (4.0 / 3.0) * self.e_star * torch.sqrt(r_star) * d ** 1.5
        f_cap = self.state["strength"][idx_c] * area_c
        f_n = torch.minimum(f_hertz, f_cap)

        # slip velocity: circumferential blade speed relative to the cell
        v_theta_cell = (cell_vel[idx_c] * e_theta).sum(dim=1)
        v_r_cell = (cell_vel[idx_c] * e_r).sum(dim=1)
        slip = (omega * r_c - v_theta_cell).abs()
        pressure_mpa = (f_n / area_c.clamp_min(1e-12)) / 1.0e6
        # the friction law is calibrated on 1e-3 ... 20 m/s; clamp the slip speed
        # to that domain (milling slip stays inside it up to ~19 m/s)
        mu = self.friction.mu_tensor(
            slip.clamp(1e-3, 19.0),
            torch.full_like(slip, case.ice_temperature_c),
            pressure_mpa.clamp(0.01, 30.0))
        f_t = mu * f_n

        # force on the ice cell: circumferential push + radial friction
        force_on_cell = (f_n[:, None] * e_theta
                         - (f_t * torch.sign(v_r_cell))[:, None] * e_r)
        raw_force = torch.zeros(self.n_cells, 2, dtype=torch.float64)
        raw_force.index_add_(0, idx_c, force_on_cell)
        capacity = self.state["strength"] * self.face_area
        magnitude = torch.sqrt((raw_force * raw_force).sum(dim=1))
        scale = torch.where(magnitude > capacity,
                            capacity / magnitude.clamp_min(1e-12),
                            torch.ones_like(magnitude))
        pair_scale = scale[idx_c]
        f_n = f_n * pair_scale
        f_t = f_t * pair_scale
        force_on_cell = force_on_cell * pair_scale[:, None]

        # resisting shaft torque and axial force: inclined-blade decomposition of
        # the contact resultant (normal + friction along the sliding direction)
        torque = (r_c * f_n * (sin_phi + mu * cos_phi)).sum()
        thrust = (f_n * (cos_phi - mu * sin_phi)).sum()
        contact_mag = f_n.sum()

        # rolling resistance of the ice cell about its own axis
        spin_rel = self.state["spin"][idx_c]
        tau_roll = -slab.rolling_resistance * a_c * f_n * torch.sign(spin_rel)
        self._energy["rolling"] += float((tau_roll * spin_rel).abs().sum()) * dt

        cell_force = torch.zeros(self.n_cells, 2, dtype=torch.float64)
        cell_force.index_add_(0, idx_c, force_on_cell)
        self._contact_force = cell_force
        cell_torque = torch.zeros(self.n_cells, dtype=torch.float64)
        cell_torque.index_add_(0, idx_c, -(f_t * a_c) * torch.sign(v_r_cell) + tau_roll)

        e_crush = self.state["strength"] * self.volume * slab.crush_strain
        power = torch.zeros(self.n_cells, dtype=torch.float64)
        power.index_add_(0, idx_c, f_n * slip + f_t * v_r_cell.abs())
        self._energy["friction"] += float((f_t * v_r_cell.abs()).sum()) * dt
        self._energy["plastic"] += float((f_n * slip).sum()) * dt
        self.state["damage"] = self.state["damage"] + power * dt / e_crush.clamp_min(1e-12)

        self._advance(dt, cell_force, cell_torque)
        self._break_and_pulverise(t, dt)
        self._apply_renewal(t)
        return {"torque": float(torque), "thrust": float(thrust),
                "force": float(contact_mag), "pairs": float(idx_c.numel())}

    def _advance(self, dt: float, cell_force: torch.Tensor, cell_torque: torch.Tensor) -> None:
        slab = self.case.slab
        inertia = 0.4 * self.mass * self.radius ** 2
        u = self.state["pos"] - self.anchor
        f_bond = -slab.bond_stiffness * u - slab.bond_damping * self.state["vel"]
        acc = (cell_force + f_bond) / self.mass[:, None]
        self.state["vel"] = self.state["vel"] + acc * dt
        self.state["pos"] = self.state["pos"] + self.state["vel"] * dt
        self.state["spin"] = self.state["spin"] + (cell_torque / inertia) * dt
        self._last_bond = torch.sqrt((f_bond * f_bond).sum(dim=1))

    def _break_and_pulverise(self, t: float, dt: float) -> None:
        """Convert load-bearing ice cells into debris, at the feed-limited rate.

        A cell leaves the bonded field either when its accumulated crushing
        damage reaches unity (pulverisation) or when the transmitted contact
        force exceeds its bond capacity ``tau_bond * A_face`` (debonding of a
        whole block).  The number of conversion events per step is limited by
        the ice volume the sheet advance brings into the cut,
        ``A_swept * V_cut * dt``; cells with the largest damage go first.  The
        remaining cells keep transmitting contact force -- they are the ice the
        blades are still crushing.
        """
        slab = self.case.slab
        contact_mag = torch.sqrt((self._contact_force * self._contact_force).sum(dim=1))
        bond_limit = slab.bond_strength_fraction * self.state["strength"] * self.face_area
        candidate = self.state["alive"] & ((self.state["damage"] >= 1.0)
                                           | (contact_mag > bond_limit))
        if not bool(candidate.any()):
            return
        budget = float(self.face_area.sum()) * self.case.ship_speed * dt
        priority = torch.where(candidate, -self.state["damage"],
                               torch.full_like(self.state["damage"], 1e9))
        order = torch.argsort(priority)
        used = 0.0
        for index in order.tolist():
            if used > budget:
                break
            if not bool(candidate[index]):
                break
            used += float(self.volume[index])
            self._spawn_debris(index, t, coarse=bool(contact_mag[index] > bond_limit[index]))
            self._schedule_renewal(index, t)

    def _schedule_renewal(self, index: int, t: float) -> None:
        self.state["alive"][index] = False
        self.state["pos"][index] = self.anchor[index]
        self.state["vel"][index] = 0.0
        self.state["spin"][index] = 0.0
        self.state["damage"][index] = 0.0
        feed = self.case.ice_thickness / max(self.case.ship_speed, 1e-6)
        self.state["renew_at"][index] = t + feed

    def _apply_renewal(self, t: float) -> None:
        due = (~self.state["alive"]) & (self.state["renew_at"] <= t)
        if not bool(due.any()):
            return
        idx = due.nonzero().flatten()
        u = torch.rand(idx.numel(), dtype=torch.float64, generator=self._gen).clamp_min(1e-9)
        k = weibull_shape_for_cov(self.case.slab.strength_cov)
        lam = float(self.cells["sigma_mean"]) / math.gamma(1.0 + 1.0 / k)
        self.state["strength"][idx] = lam * (-torch.log(u)) ** (1.0 / k)
        self.state["alive"][idx] = True
        self.state["damage"][idx] = 0.0
        self.state["renew_at"][idx] = float("inf")
        self._renewals += int(idx.numel())

    def _spawn_debris(self, index: int, t: float, *, coarse: bool) -> None:
        mass_cell = float(self.mass[index])
        volume_cell = float(self.volume[index])
        self._pulverised += mass_cell
        if coarse:
            self._bond_broken += mass_cell
            n_frag = 1
        else:
            n_frag = int(torch.randint(3, 7, (1,), generator=self._gen).item())
        gen = self._gen
        weights = torch.rand(n_frag, dtype=torch.float64, generator=gen) + 0.25
        fractions = (weights / weights.sum()).tolist()
        omega = 2.0 * math.pi * self.case.n_rps
        pos = self.state["pos"][index]
        vel = self.state["vel"][index]
        r_here = float(torch.linalg.norm(pos))
        theta_here = math.atan2(float(pos[1]), float(pos[0]))
        v_blade = torch.tensor([-omega * float(pos[1]), omega * float(pos[0])],
                               dtype=torch.float64)
        phi = self.geometry.pitch_angle(r_here)
        c, s = math.cos(theta_here), math.sin(theta_here)
        for f in fractions:
            m_f = mass_cell * f
            v_f = volume_cell * f
            d_eq = (6.0 * v_f / math.pi) ** (1.0 / 3.0)
            added = 0.5 * RHO_WATER * v_f
            kappa = m_f / (m_f + added)
            v0 = vel + kappa * (v_blade - vel)
            jitter = torch.randn(2, dtype=torch.float64, generator=gen)
            v0 = v0 + 0.15 * float(torch.linalg.norm(v0)) * jitter
            speed = float(torch.linalg.norm(v0))
            vr = float(v0[0] * c + v0[1] * s)
            vt = float(-v0[0] * s + v0[1] * c)
            v_axial = max(speed * math.tan(phi) * 0.5, 0.05)
            self._debris_total += 1
            record = {
                "t": t,
                "x": float(torch.randn((), generator=gen)) * 0.01 * self.case.ice_thickness,
                "r": r_here,
                "theta": theta_here,
                "vr": vr,
                "vt": vt,
                "vx": v_axial,
                "mass": m_f,
                "d": d_eq,
            }
            # reservoir sampling: an unbiased fixed-size subset of all fragments
            if len(self._debris) < self.max_debris:
                self._debris.append(record)
            else:
                slot = int(torch.randint(0, self._debris_total, (1,), generator=gen).item())
                if slot < self.max_debris:
                    self._debris[slot] = record

    # -- driver -------------------------------------------------------------
    def run(self) -> MillingResult:
        case = self.case
        hydro = blade_element_open_water(self.geometry, case.n_rps, case.ship_speed)
        n_steps = int(round(case.duration / case.dt))
        time, torque, thrust, force, pairs = [], [], [], [], []
        for i in range(n_steps):
            t = i * case.dt
            rec = self.step(t, case.dt)
            time.append(t)
            torque.append(rec["torque"])
            thrust.append(rec["thrust"])
            force.append(rec["force"])
            pairs.append(rec["pairs"])
        time_t = torch.tensor(time, dtype=torch.float64)
        torque_t = torch.tensor(torque, dtype=torch.float64)
        thrust_t = torch.tensor(thrust, dtype=torch.float64)
        force_t = torch.tensor(force, dtype=torch.float64)
        pairs_t = torch.tensor(pairs, dtype=torch.float64)
        thrust_net = hydro["thrust"] - float(thrust_t.mean())
        # the propeller race is driven by the hydrodynamic blade forces; the ice
        # load is a shaft/hull resistance, so it does not cancel the jet
        race = RaceModel(thrust=max(hydro["thrust"], 1.0e3), diameter=self.geometry.diameter)
        transported = integrate_debris(self._debris, race, x_exit=case.x_exit)
        diagnostics = {
            "n_steps": n_steps,
            "n_panels": int(self.panels["r"].numel()),
            "n_cells": self.n_cells,
            "renewals": self._renewals,
            "bond_broken_mass": self._bond_broken,
            "pulverised_mass": self._pulverised,
            "energy_j": dict(self._energy),
            "thrust_net": thrust_net,
            "race_induced": race.induced_velocity(case.ship_speed),
            "e_star": self.e_star,
            "g_star": self.g_star,
            "sigma_mean_pa": float(self.cells["sigma_mean"]),
            "friction_model": {"mu_ref": self.friction.mu_ref, "form": self.friction.form},
        }
        empty = torch.zeros(0, dtype=torch.float64)
        return MillingResult(
            case=case,
            time=time_t,
            torque_ice=torque_t,
            thrust_ice=thrust_t,
            contact_force=force_t,
            active_contacts=pairs_t,
            debris_time=torch.tensor([d["t"] for d in transported], dtype=torch.float64)
            if transported else empty,
            debris_mass_each=torch.tensor([d["mass"] for d in transported], dtype=torch.float64)
            if transported else empty,
            debris_initial=torch.tensor([[d["x"], d["r"], d["vx"], d["vr"], d["vt"]]
                                         for d in transported], dtype=torch.float64)
            if transported else torch.zeros(0, 5, dtype=torch.float64),
            debris_exit_speed=torch.tensor([d["exit_speed"] for d in transported],
                                           dtype=torch.float64) if transported else empty,
            debris_exit_radius=torch.tensor([d["exit_radius"] for d in transported],
                                            dtype=torch.float64) if transported else empty,
            contact_pairs=pairs_t,
            hydro=hydro,
            race=race,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# debris transport in the race
# ---------------------------------------------------------------------------
def integrate_debris(debris: list[dict[str, float]], race: RaceModel, *,
                     x_exit: float = 1.0, dt: float = 1.0e-3, c_d: float = 1.10,
                     max_time: float = 40.0, rho_water: float = RHO_WATER
                     ) -> list[dict[str, float]]:
    """Advect every fragment through the race and record its discharge speed.

    ``dv/dt = (0.5 rho_w C_d A / m) |u - v| (u - v)`` with the axial water
    velocity from :class:`RaceModel`, integrated with RK2 for all fragments at
    once; ``exit_speed`` is ``|v|`` where the fragment crosses ``x = x_exit``.
    Gravity/buoyancy are neglected (fragment Froude number >> 1 over 1 m).
    """
    if not debris:
        return []
    n = len(debris)
    mass = torch.tensor([d["mass"] for d in debris], dtype=torch.float64).clamp_min(1e-9)
    diam = torch.tensor([d["d"] for d in debris], dtype=torch.float64)
    coeff = 0.5 * rho_water * c_d * math.pi * (0.5 * diam) ** 2 / mass
    x = torch.tensor([d["x"] for d in debris], dtype=torch.float64)
    r = torch.tensor([d["r"] for d in debris], dtype=torch.float64)
    vx = torch.tensor([d["vx"] for d in debris], dtype=torch.float64)
    vr = torch.tensor([d["vr"] for d in debris], dtype=torch.float64)
    # DEFECT-D2 (repaired): the carrier is the RaceModel field itself, not a local
    # sigma = 0.5 D closure -- chips are advected by the jet the milling torque was
    # computed with.
    water_velocity = race.axial_velocity_tensor

    moving = torch.ones(n, dtype=torch.bool)
    t = 0.0
    while t < max_time and bool(moving.any()):
        u = water_velocity(x, r)
        rel = u - vx
        ax1 = coeff * rel.abs() * rel
        ar1 = -coeff * vr.abs() * vr
        u_m = water_velocity(x + 0.5 * dt * vx, r + 0.5 * dt * vr)
        rel_m = u_m - (vx + 0.5 * dt * ax1)
        vr_m = vr + 0.5 * dt * ar1
        vx = torch.where(moving, vx + dt * (coeff * rel_m.abs() * rel_m), vx)
        vr = torch.where(moving, vr + dt * (-coeff * vr_m.abs() * vr_m), vr)
        x = torch.where(moving, x + dt * vx, x)
        r = torch.where(moving, r + dt * vr, r)
        t += dt
        moving = moving & (x < x_exit) & (r <= race.diameter)
    out: list[dict[str, float]] = []
    for i, frag in enumerate(debris):
        record = dict(frag)
        record["exit_speed"] = float(math.hypot(float(vx[i]), float(vr[i])))
        record["exit_radius"] = float(r[i])
        record["travel_time"] = float(t)
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# torque spectrum
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectrumResult:
    """Single-sided torque spectrum with the BPF peak metrics."""

    frequencies: torch.Tensor
    amplitudes: torch.Tensor
    peak_frequency: float
    peak_amplitude: float
    bpf_amplitude: float
    harmonic_amplitudes: tuple[float, ...]
    noise_floor: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_frequency": self.peak_frequency,
            "peak_amplitude": self.peak_amplitude,
            "bpf_amplitude": self.bpf_amplitude,
            "harmonic_amplitudes": list(self.harmonic_amplitudes),
            "noise_floor": self.noise_floor,
        }


def torque_spectrum(time: torch.Tensor, torque: torch.Tensor, bpf: float,
                    *, n_harmonics: int = 3) -> SpectrumResult:
    """Amplitude spectrum of the torque record (Hann window, DC removed)."""
    if time.numel() < 32:
        raise ValueError("need at least 32 samples")
    dt = float(time[1] - time[0])
    n = int(torque.numel())
    x = torque.to(torch.float64)
    detrended = x - x.mean()
    window = torch.hann_window(n, periodic=False, dtype=torch.float64)
    spectrum = torch.fft.rfft(detrended * window)
    amps = spectrum.abs() * (2.0 / float(window.sum()))
    freqs = torch.fft.rfftfreq(n, d=dt)
    amps[0] = 0.0
    # drop the lowest two bins (residual trend / record-length leakage)
    valid = freqs > 2.0 / (n * dt)
    search = torch.where(valid, amps, torch.zeros_like(amps))
    peak_index = int(torch.argmax(search))
    noise = float(torch.median(amps[valid])) if bool(valid.any()) else 0.0
    df = float(freqs[1])

    def amp_at(f: float) -> float:
        centre = int(round(f / df))
        lo, hi = max(centre - 1, 0), min(centre + 2, int(freqs.numel()))
        return float(amps[lo:hi].max()) if hi > lo else 0.0

    harmonics = tuple(amp_at(k * bpf) for k in range(2, 2 + n_harmonics))
    return SpectrumResult(
        frequencies=freqs,
        amplitudes=amps,
        peak_frequency=float(freqs[peak_index]),
        peak_amplitude=float(amps[peak_index]),
        bpf_amplitude=amp_at(bpf),
        harmonic_amplitudes=harmonics,
        noise_floor=noise,
    )


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------
RIPPLE_BAND = (0.10, 0.30)
EJECTION_BAND = (0.50, 2.00)
FREQUENCY_TOLERANCE = 0.10


def evaluate_acceptance(rows: list[dict[str, Any]], *, cases_required: int = 4,
                        rpm_range: tuple[float, float] = (60.0, 120.0),
                        thickness_range: tuple[float, float] = (0.3, 0.8),
                        ) -> dict[str, Any]:
    """Machine-checkable acceptance table for the IN27-S4 criteria."""
    if not rows:
        raise ValueError("no case rows supplied")
    per_case = []
    for row in rows:
        ripple_ok = RIPPLE_BAND[0] <= row["ripple_robust"] <= RIPPLE_BAND[1]
        speed_ok = EJECTION_BAND[0] <= row["ejection_speed_mean"] <= EJECTION_BAND[1]
        bpf = row["bpf_hz"]
        freq_err = abs(row["peak_frequency"] - bpf) / bpf
        # the dominant non-DC peak must be BPF or one of its harmonics
        harmonic_index = round(row["peak_frequency"] / bpf)
        harmonic_ok = (harmonic_index >= 1 and freq_err <= FREQUENCY_TOLERANCE
                       and abs(row["peak_frequency"] / bpf - harmonic_index) <= FREQUENCY_TOLERANCE)
        harmonic_energy_ok = all(amp > 2.0 * row["noise_floor"]
                                 for amp in row["harmonic_amplitudes"])
        per_case.append({
            "rpm": row["rpm"],
            "ice_thickness": row["ice_thickness"],
            "ripple_robust": row["ripple_robust"],
            "ripple_p2p": row["ripple_p2p"],
            "ripple_ok": bool(ripple_ok),
            "peak_frequency": row["peak_frequency"],
            "bpf_hz": bpf,
            "frequency_error": freq_err,
            "harmonic_index": int(harmonic_index),
            "frequency_ok": bool(harmonic_ok),
            "harmonic_amplitudes": list(row["harmonic_amplitudes"]),
            "noise_floor": row["noise_floor"],
            "harmonics_ok": bool(harmonic_energy_ok),
            "ejection_speed_mean": row["ejection_speed_mean"],
            "ejection_speed_band": list(EJECTION_BAND),
            "ejection_ok": bool(speed_ok),
            "debris_count": row["debris_count"],
        })
    rpms = sorted({row["rpm"] for row in rows})
    thicknesses = sorted({row["ice_thickness"] for row in rows})
    coverage_ok = (len(rows) >= cases_required
                   and min(rpms) <= rpm_range[0] + 1e-9 and max(rpms) >= rpm_range[1] - 1e-9
                   and min(thicknesses) <= thickness_range[0] + 1e-9
                   and max(thicknesses) >= thickness_range[1] - 1e-9)
    criteria = {
        "a_torque_ripple": {
            "definition": "(Q_95 - Q_05) / Q_mean of the total shaft torque",
            "band": list(RIPPLE_BAND),
            "values": [row["ripple_robust"] for row in per_case],
            "pass": all(row["ripple_ok"] for row in per_case),
        },
        "b_bpf_spectrum": {
            "definition": "dominant non-DC spectral peak at BPF (or a BPF harmonic); "
                          "harmonics 2-4 x BPF above twice the median noise floor",
            "tolerance": FREQUENCY_TOLERANCE,
            "pass": all(row["frequency_ok"] and row["harmonics_ok"] for row in per_case),
        },
        "c_ejection_speed": {
            "definition": "mean debris speed at the x = x_exit discharge plane",
            "band": list(EJECTION_BAND),
            "values": [row["ejection_speed_mean"] for row in per_case],
            "pass": all(row["ejection_ok"] for row in per_case),
        },
        "d_case_matrix": {
            "cases": len(rows),
            "rpm_values": rpms,
            "thickness_values": thicknesses,
            "rpm_range": list(rpm_range),
            "thickness_range": list(thickness_range),
            "pass": bool(coverage_ok),
        },
    }
    return {
        "criteria": criteria,
        "per_case": per_case,
        "all_pass": all(item["pass"] for item in criteria.values()),
    }


def run_case(rpm: float, ice_thickness: float, **kwargs: Any) -> MillingResult:
    """Convenience wrapper: build the case and run it."""
    case_keys = set(MillingCase.__dataclass_fields__)
    case_kwargs = {k: v for k, v in kwargs.items() if k in case_keys}
    sim_kwargs = {k: v for k, v in kwargs.items() if k not in case_keys}
    case = MillingCase(rpm=rpm, ice_thickness=ice_thickness, **case_kwargs)
    return MillingSimulation(case, **sim_kwargs).run()


def analyse_case(result: MillingResult) -> dict[str, Any]:
    """Summary row + torque spectrum for one case (report-ready)."""
    row = result.summary()
    spectrum = torque_spectrum(result.time, result.total_torque,
                               result.case.blade_passing_frequency)
    row.update(spectrum.as_dict())
    row["spectrum_frequencies"] = spectrum.frequencies.tolist()
    row["spectrum_amplitudes"] = spectrum.amplitudes.tolist()
    row["torque_history"] = result.total_torque.tolist()
    row["time_history"] = result.time.tolist()
    return row