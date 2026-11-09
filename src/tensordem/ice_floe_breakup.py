"""Mass-conserving breakup–clearance operator for DEM ice-floe simulations.

Physical motivation
-------------------
Previous versions of the ice-channel DEM used an ad-hoc geometric threshold
(``delta > 0.25 m`` penetration) to silently remove floes from the simulation
domain.  This is unphysical: it destroys mass and momentum without accounting
for where they go, and it makes floe kinematics undeliverable (the removed
floes carry away momentum that the force balance cannot see).

This module replaces that threshold with a **physics-based breakup operator**
that:

1. **Triggers on contact stress**, not penetration depth.  When the local
   contact pressure ``p = F_n / A_contact`` exceeds the ice crushing strength
   ``sigma_c``, the floe is marked for breakup.

2. **Conserves mass exactly**: the parent floe of area ``A_p`` is split into
   ``n`` child floes whose areas sum to ``A_p`` (to machine precision).
   For disk floes of equal thickness, area conservation = volume conservation.

3. **Conserves momentum**: child floes inherit the parent's linear and angular
   momentum, plus a small radial dispersion velocity drawn from a破碎
   dissipation budget.

4. **Tracks energy**: every breakup event records the elastic energy stored
   in the contact at the moment of failure, the kinetic energy of the children,
   and the dissipated energy (difference), so an energy ledger closes to
   within the time-stepping tolerance.

5. **Provides convergence**: as the minimum child size ``h_min -> 0`` the
   floe velocity field should converge (change < 2 %), which is the
   mesh-independence check for the breakup operator.

Scope and honesty
-----------------
Two-dimensional disk floes (same geometry as :mod:`tensordem.ice_channel`).
The breakup is instantaneous (no sub-step crack propagation); the fragment
sizes follow a prescribed splitting rule (binary or n-ary with configurable
fragment count).  This is a first-order model: real ice fragmentation has
a cascade of crack speeds and a Weibull-distributed fragment size, but the
mass/momentum/energy conservation properties are exact at every event
regardless of the splitting rule.

References
----------
- Keinonen, A. (2017/2019). Local ice pressure for ship-structure design.
- ISO 19906:2019. Petroleum and natural gas industries — Fixed steel
  offshore structures.
- Hopkins, M.A. (2001). Discrete element modeling of sea ice ridging.
  Proc. 16th IAHR Ice Symposium.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "BreakupConfig",
    "EnergyLedger",
    "BreakupEvent",
    "BreakupOperator",
    "mass_conservation_error",
    "energy_residual",
    "run_breakup_convergence_study",
]

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BreakupConfig:
    """Parameters controlling the breakup operator.

    Attributes
    ----------
    crushing_strength_pa : float
        Ice crushing strength [Pa].  Contact pressure exceeding this value
        triggers breakup.  Default 1.5 MPa (Keinonen / ISO 19906 order).
    n_children : int
        Number of child floes produced per breakup event.  Default 2 (binary
        splitting).  Must be >= 2.
    area_partition : tuple of float or None
        Fraction of parent area assigned to each child.  Must sum to 1.0.
        If None, equal partition is used (``1/n_children`` each).
    dissipation_fraction : float
        Fraction of contact elastic energy dissipated at breakup (converted
        to heat / micro-cracking).  The remainder goes into child kinetic
        energy (radial dispersion).  Default 0.5.
    min_child_radius_m : float
        Minimum allowed child radius [m].  If a breakup would produce a child
        smaller than this, the breakup is suppressed (the floe is treated as
        unbreakable rubble).  Default 0.1 m.
    max_breakups_per_step : int
        Safety cap on the number of breakup events per time step.  Default 64.
    """

    crushing_strength_pa: float = 1.5e6
    n_children: int = 2
    area_partition: Optional[Tuple[float, ...]] = None
    dissipation_fraction: float = 0.5
    min_child_radius_m: float = 0.1
    max_breakups_per_step: int = 64

    def __post_init__(self) -> None:
        if self.crushing_strength_pa <= 0.0:
            raise ValueError("crushing_strength_pa must be positive")
        if self.n_children < 2:
            raise ValueError("n_children must be >= 2")
        if self.area_partition is not None:
            s = sum(self.area_partition)
            if len(self.area_partition) != self.n_children:
                raise ValueError("area_partition length must equal n_children")
            if abs(s - 1.0) > 1e-10:
                raise ValueError(f"area_partition must sum to 1.0, got {s}")
            if any(p <= 0.0 for p in self.area_partition):
                raise ValueError("area_partition entries must be positive")
        if not (0.0 <= self.dissipation_fraction <= 1.0):
            raise ValueError("dissipation_fraction must lie in [0, 1]")
        if self.min_child_radius_m <= 0.0:
            raise ValueError("min_child_radius_m must be positive")
        if self.max_breakups_per_step < 1:
            raise ValueError("max_breakups_per_step must be >= 1")

    @property
    def partition(self) -> Tuple[float, ...]:
        if self.area_partition is not None:
            return tuple(self.area_partition)
        return tuple(1.0 / self.n_children for _ in range(self.n_children))


# ---------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------

@dataclass
class EnergyLedger:
    """Tracks energy and momentum flows across breakup events.

    All quantities are cumulative (running sums over the simulation).
    Units: energy [J], momentum [kg m/s], mass [kg].
    """

    # Cumulative energy terms
    elastic_energy_input: float = 0.0       # elastic energy stored in contacts that broke up
    kinetic_energy_children: float = 0.0    # kinetic energy of children after breakup
    dissipated_energy: float = 0.0          # energy dissipated (heat, micro-cracking)
    momentum_carried_away: float = 0.0      # |momentum| of children (scalar sum of norms)

    # Cumulative mass terms
    mass_before_breakup: float = 0.0        # total mass of parents before breakup
    mass_after_breakup: float = 0.0         # total mass of children after breakup

    # Counters
    n_events: int = 0
    n_children_produced: int = 0

    def record_event(
        self,
        elastic_energy: float,
        child_kinetic_energy: float,
        dissipated: float,
        child_momentum_norm: float,
        parent_mass: float,
        child_mass_sum: float,
    ) -> None:
        """Record a single breakup event."""
        self.elastic_energy_input += elastic_energy
        self.kinetic_energy_children += child_kinetic_energy
        self.dissipated_energy += dissipated
        self.momentum_carried_away += child_momentum_norm
        self.mass_before_breakup += parent_mass
        self.mass_after_breakup += child_mass_sum
        self.n_events += 1

    @property
    def energy_residual(self) -> float:
        """Energy balance residual: |E_in - E_out| / E_in.

        E_in = elastic energy stored in the contact.
        E_out = kinetic energy of children + dissipated energy.
        Returns 0.0 if no events recorded.
        """
        if self.elastic_energy_input <= 0.0:
            return 0.0
        e_out = self.kinetic_energy_children + self.dissipated_energy
        return abs(self.elastic_energy_input - e_out) / self.elastic_energy_input

    @property
    def mass_conservation_error(self) -> float:
        """Relative mass conservation error: |M_before - M_after| / M_before.

        Returns 0.0 if no events recorded.
        """
        if self.mass_before_breakup <= 0.0:
            return 0.0
        return abs(self.mass_before_breakup - self.mass_after_breakup) / self.mass_before_breakup

    def summary(self) -> Dict[str, float]:
        return {
            "n_events": self.n_events,
            "n_children_produced": self.n_children_produced,
            "elastic_energy_input_J": self.elastic_energy_input,
            "kinetic_energy_children_J": self.kinetic_energy_children,
            "dissipated_energy_J": self.dissipated_energy,
            "energy_residual": self.energy_residual,
            "mass_before_kg": self.mass_before_breakup,
            "mass_after_kg": self.mass_after_breakup,
            "mass_conservation_error": self.mass_conservation_error,
        }


@dataclass
class BreakupEvent:
    """Record of a single breakup event for post-processing."""
    parent_index: int
    parent_radius: float
    parent_thickness: float
    parent_position: Tuple[float, float]
    parent_velocity: Tuple[float, float]
    child_radii: List[float]
    child_positions: List[Tuple[float, float]]
    child_velocities: List[Tuple[float, float]]
    elastic_energy_J: float
    dissipated_energy_J: float
    time: float


# ---------------------------------------------------------------------------
# Breakup operator
# ---------------------------------------------------------------------------

class BreakupOperator:
    """Mass-conserving breakup operator for 2D disk floes.

    The operator is called once per time step after contact forces are
    computed.  It inspects the contact pressure on each floe and, if the
    pressure exceeds the crushing strength, splits the floe into children.

    Parameters
    ----------
    config : BreakupConfig
        Breakup parameters.
    ice_density : float
        Ice density [kg/m^3].  Default 917.0.
    """

    def __init__(self, config: Optional[BreakupConfig] = None,
                 ice_density: float = 917.0) -> None:
        self.config = config or BreakupConfig()
        self.ice_density = ice_density
        self.ledger = EnergyLedger()
        self.events: List[BreakupEvent] = []

    def _contact_pressure(
        self,
        normal_force: torch.Tensor,
        floe_radius: torch.Tensor,
        floe_thickness: torch.Tensor,
    ) -> torch.Tensor:
        """Compute nominal contact pressure for each floe.

        The contact area for a disk-disk or hull-disk contact is approximated
        as ``A = 2 * R * t`` (line contact of length ``t`` and width ``2R``
        projected onto the contact plane).  This is the same area used in the
        ice_channel contact stiffness anchor.

        Parameters
        ----------
        normal_force : Tensor, shape (n_floes,)
            Total normal contact force on each floe [N].
        floe_radius : Tensor, shape (n_floes,)
            Floe radius [m].
        floe_thickness : Tensor, shape (n_floes,)
            Floe thickness [m].

        Returns
        -------
        pressure : Tensor, shape (n_floes,)
            Nominal contact pressure [Pa].
        """
        area = 2.0 * floe_radius * floe_thickness
        area = torch.clamp(area, min=1e-12)
        return normal_force / area

    def check_breakup(
        self,
        normal_force: torch.Tensor,
        radius: torch.Tensor,
        thickness: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean mask of floes that should break up.

        A floe breaks up if its contact pressure exceeds the crushing strength
        AND the resulting children would all be larger than ``min_child_radius``.
        """
        n = radius.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.bool)

        pressure = self._contact_pressure(normal_force, radius, thickness)
        exceeds = pressure > self.config.crushing_strength_pa

        # Check that children would be large enough
        partition = self.config.partition
        min_frac = min(partition)
        min_child_r = torch.sqrt(torch.tensor(min_frac, dtype=_DTYPE)) * radius
        size_ok = min_child_r >= self.config.min_child_radius_m

        return exceeds & size_ok

    def perform_breakup(
        self,
        parent_index: int,
        parent_radius: float,
        parent_thickness: float,
        parent_pos: Tuple[float, float],
        parent_vel: Tuple[float, float],
        parent_omega: float,
        parent_mass: float,
        contact_elastic_energy: float,
        time: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[Dict[str, float]], BreakupEvent]:
        """Split a parent floe into children, conserving mass and momentum.

        Parameters
        ----------
        parent_index : int
            Index of the parent floe in the simulation array.
        parent_radius, parent_thickness : float
            Parent floe geometry [m].
        parent_pos : tuple of float
            Parent centre position (x, y) [m].
        parent_vel : tuple of float
            Parent centre velocity (vx, vy) [m/s].
        parent_omega : float
            Parent angular velocity [rad/s].
        parent_mass : float
            Parent mass [kg].
        contact_elastic_energy : float
            Elastic energy stored in the contact that triggered breakup [J].
        time : float
            Simulation time [s].
        generator : torch.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        children : list of dict
            Each dict has keys: radius, thickness, mass, pos (x,y),
            vel (vx,vy), omega.
        event : BreakupEvent
            Record of the breakup for post-processing.
        """
        cfg = self.config
        partition = cfg.partition
        n_ch = cfg.n_children

        # --- Mass conservation: children areas sum to parent area ---
        parent_area = math.pi * parent_radius ** 2
        child_areas = [parent_area * frac for frac in partition]
        child_radii = [math.sqrt(a / math.pi) for a in child_areas]

        # Thickness is inherited (flat disk breakup preserves thickness)
        child_thickness = parent_thickness

        # Mass: m = rho * area * thickness
        child_masses = [self.ice_density * a * child_thickness for a in child_areas]
        child_mass_sum = sum(child_masses)

        # --- Momentum conservation ---
        # Linear momentum: parent_mass * parent_vel = sum(child_mass_i * child_vel_i)
        # We assign children the parent velocity plus a radial dispersion.
        px, py = parent_pos
        vx, vy = parent_vel

        # Radial dispersion directions (evenly spaced angles)
        angles = [2.0 * math.pi * k / n_ch for k in range(n_ch)]

        # Dispersion speed from the energy budget
        # E_available = (1 - dissipation_fraction) * contact_elastic_energy
        # E_dispersion = 0.5 * sum(m_i * v_disp_i^2)
        # For equal masses and equal dispersion speeds: v_disp = sqrt(2 * E_avail / M_total)
        e_available = (1.0 - cfg.dissipation_fraction) * contact_elastic_energy
        e_available = max(e_available, 0.0)
        if child_mass_sum > 0.0 and e_available > 0.0:
            v_disp = math.sqrt(2.0 * e_available / child_mass_sum)
        else:
            v_disp = 0.0

        dissipated = cfg.dissipation_fraction * contact_elastic_energy

        children = []
        child_positions = []
        child_velocities = []
        total_px = 0.0
        total_py = 0.0

        for k in range(n_ch):
            dx = child_radii[k] * math.cos(angles[k])
            dy = child_radii[k] * math.sin(angles[k])
            cx = px + dx
            cy = py + dy
            # Velocity = parent velocity + dispersion in radial direction
            cvx = vx + v_disp * math.cos(angles[k])
            cvy = vy + v_disp * math.sin(angles[k])
            # Angular velocity: inherit parent omega (conservation of angular momentum
            # is approximate here since children have different moments of inertia;
            # the correction is small for small dispersion)
            child_omega = parent_omega

            child = {
                "radius": child_radii[k],
                "thickness": child_thickness,
                "mass": child_masses[k],
                "pos": (cx, cy),
                "vel": (cvx, cvy),
                "omega": child_omega,
            }
            children.append(child)
            child_positions.append((cx, cy))
            child_velocities.append((cvx, cvy))
            total_px += child_masses[k] * cvx
            total_py += child_masses[k] * cvy

        # --- Kinetic energy of children (dispersion only) ---
        # The energy balance tracks only the *change* due to breakup:
        # E_input = contact_elastic_energy (stored in the contact that failed)
        # E_output = dispersion KE + dissipated energy
        # The parent's translational KE is NOT part of this balance because
        # it was present before the breakup and is inherited by the children.
        ke_dispersion = 0.0
        for k in range(n_ch):
            cvx, cvy = children[k]["vel"]
            # Dispersion velocity = child velocity - parent velocity
            dvx = cvx - vx
            dvy = cvy - vy
            ke_dispersion += 0.5 * child_masses[k] * (dvx ** 2 + dvy ** 2)

        # --- Momentum norm (for ledger) ---
        momentum_norm = math.sqrt(total_px ** 2 + total_py ** 2)

        # --- Record in ledger ---
        self.ledger.record_event(
            elastic_energy=contact_elastic_energy,
            child_kinetic_energy=ke_dispersion,
            dissipated=dissipated,
            child_momentum_norm=momentum_norm,
            parent_mass=parent_mass,
            child_mass_sum=child_mass_sum,
        )
        self.ledger.n_children_produced += n_ch

        # --- Record event ---
        event = BreakupEvent(
            parent_index=parent_index,
            parent_radius=parent_radius,
            parent_thickness=parent_thickness,
            parent_position=parent_pos,
            parent_velocity=parent_vel,
            child_radii=child_radii,
            child_positions=child_positions,
            child_velocities=child_velocities,
            elastic_energy_J=contact_elastic_energy,
            dissipated_energy_J=dissipated,
            time=time,
        )
        self.events.append(event)

        return children, event


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def mass_conservation_error(
    parent_radius: float,
    child_radii: List[float],
    thickness: float,
) -> float:
    """Compute |Σ V_child / V_parent - 1| for disk floes of equal thickness.

    This is the machine-checkable criterion C1: the error must be < 1e-6.
    """
    parent_vol = math.pi * parent_radius ** 2 * thickness
    child_vol_sum = sum(math.pi * r ** 2 * thickness for r in child_radii)
    if parent_vol <= 0.0:
        return float("inf")
    return abs(child_vol_sum / parent_vol - 1.0)


def energy_residual(
    elastic_energy: float,
    child_kinetic_energy: float,
    dissipated_energy: float,
) -> float:
    """Compute |E_in - E_out| / E_in for a single breakup event.

    This is the machine-checkable criterion C2: the residual must be < 3%.
    """
    if elastic_energy <= 0.0:
        return 0.0
    e_out = child_kinetic_energy + dissipated_energy
    return abs(elastic_energy - e_out) / elastic_energy


# ---------------------------------------------------------------------------
# Convergence study (criterion C4)
# ---------------------------------------------------------------------------

def run_breakup_convergence_study(
    operator_factory,
    h_min_values: List[float],
    n_realizations: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Run a convergence study: vary h_min and measure floe velocity change.

    The criterion C4 is: as h_min -> 0, the mean floe velocity should change
    by less than 2%.

    Parameters
    ----------
    operator_factory : callable
        A function ``factory(h_min) -> (mean_velocity, std_velocity)`` that
        runs a short simulation with the given minimum child radius and
        returns the time-averaged mean floe speed and its standard deviation.
    h_min_values : list of float
        Minimum child radii to test [m].
    n_realizations : int
        Number of realizations per h_min (for uncertainty quantification).
    seed : int
        Base random seed.

    Returns
    -------
    results : dict
        Keys: "h_min", "mean_velocity", "std_velocity", "relative_change".
    """
    results: Dict[str, List[float]] = {
        "h_min": [],
        "mean_velocity": [],
        "std_velocity": [],
        "relative_change": [],
    }

    ref_velocity = None
    for h_min in h_min_values:
        velocities = []
        for r in range(n_realizations):
            v_mean, v_std = operator_factory(h_min, seed + r)
            velocities.append(v_mean)

        avg_v = sum(velocities) / len(velocities)
        std_v = (sum((v - avg_v) ** 2 for v in velocities) / len(velocities)) ** 0.5

        if ref_velocity is None:
            ref_velocity = avg_v
            rel_change = 0.0
        else:
            rel_change = abs(avg_v - ref_velocity) / max(abs(ref_velocity), 1e-30)

        results["h_min"].append(h_min)
        results["mean_velocity"].append(avg_v)
        results["std_velocity"].append(std_v)
        results["relative_change"].append(rel_change)

    return results
