"""Integrated ice-channel DEM with mass-conserving breakup operator.

This module extends :class:`tensordem.ice_channel.IceChannelSimulation` with
the breakup operator from :mod:`tensordem.ice_floe_breakup`.  The integration
is non-invasive: the base simulation is unchanged, and the breakup is applied
as a post-processing step after each time step.

The key addition is the **energy ledger**: every time step records the contact
impulse, breakup dissipation, momentum carried away by children, and the
penalty elastic energy.  This closes the energy balance and makes floe
kinematics deliverable (previously, floes removed by the ad-hoc delta > 0.25 m
threshold carried away untracked momentum).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Tuple

import torch

from tensordem.ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
    _DTYPE,
)
from tensordem.ice_floe_breakup import (
    BreakupConfig,
    BreakupEvent,
    BreakupOperator,
    EnergyLedger,
)

__all__ = [
    "IntegratedIceChannelSimulation",
    "IntegratedResult",
]


@dataclass
class IntegratedResult:
    """Result of an integrated simulation run."""
    # Force statistics
    mean_hull_force_N: float
    max_hull_force_N: float
    std_hull_force_N: float
    # Floe kinematics
    mean_floe_speed_m_s: float
    max_floe_speed_m_s: float
    # Breakup statistics
    n_breakup_events: int
    n_children_produced: int
    # Energy ledger
    energy_residual: float
    mass_conservation_error: float
    # Time series
    time_history: List[float]
    hull_force_history: List[float]
    mean_speed_history: List[float]
    n_floes_history: List[int]
    # Ledger summary
    ledger_summary: Dict[str, float]


class IntegratedIceChannelSimulation:
    """Ice-channel DEM with mass-conserving breakup.

    This wraps :class:`IceChannelSimulation` and adds:
    1. Per-step contact pressure monitoring
    2. Breakup triggering when pressure > crushing strength
    3. Mass-conserving floe splitting
    4. Energy/momentum ledger tracking
    """

    def __init__(
        self,
        radius: List[float],
        thickness: List[float],
        water: Optional[WaterEnvironment] = None,
        contact: Optional[ContactParameters] = None,
        hull: Optional[HullGeometry] = None,
        breakup_config: Optional[BreakupConfig] = None,
        dt: float = 1e-2,
    ) -> None:
        self._water = water or WaterEnvironment()
        # Initialize base simulation
        self.base = IceChannelSimulation(
            radius=list(radius),
            thickness=list(thickness),
            water=self._water,
            contact=contact,
            hull=hull,
            dt=dt,
        )

        # Breakup operator
        self.breakup = BreakupOperator(
            config=breakup_config,
            ice_density=self._water.ice_density,
        )

        # Per-step ledger
        self.step_ledger: List[Dict[str, float]] = []

        # History
        self._hull_forces: List[float] = []
        self._mean_speeds: List[float] = []
        self._n_floes: List[int] = []
        self._times: List[float] = []

    @property
    def ledger(self) -> EnergyLedger:
        return self.breakup.ledger

    @property
    def n_floes(self) -> int:
        return self.base.n

    def _compute_floe_contact_forces(self, t: float) -> torch.Tensor:
        """Compute the total normal contact force on each floe."""
        n = self.base.n
        total_fn = torch.zeros(n, dtype=_DTYPE)

        # Floe-floe contacts
        if n >= 2:
            idx, f_pair, _, _, _ = self.base._floe_floe_contacts()
            if idx.numel() > 0:
                i, j = idx[:, 0], idx[:, 1]
                fn_mag = f_pair.norm(dim=1)
                total_fn.index_add_(0, i, fn_mag)
                total_fn.index_add_(0, j, fn_mag)

        # Floe-hull contacts
        if self.base.hulls:
            for hull_obj in self.base.hulls:
                f_floes, _ = self.base._hull_force(t, self.base.pos, hull=hull_obj)
                fn_mag = torch.linalg.norm(f_floes, dim=1)
                total_fn = total_fn + fn_mag

        return total_fn

    def _compute_contact_elastic_energy(self) -> torch.Tensor:
        """Compute the elastic energy stored in contacts for each floe."""
        n = self.base.n
        energy = torch.zeros(n, dtype=_DTYPE)

        if n >= 2:
            pos = self.base.pos
            delta = pos[:, None, :] - pos[None, :, :]
            dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
            r_sum = self.base.radius[:, None] + self.base.radius[None, :]
            overlap = r_sum - dist
            eye = torch.eye(n, dtype=torch.bool)
            upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
            active = (overlap > 0.0) & (~eye) & upper
            idx = active.nonzero(as_tuple=False)
            if idx.numel() > 0:
                i, j = idx[:, 0], idx[:, 1]
                pen = overlap[i, j]
                r_i = self.base.radius[i]
                r_j = self.base.radius[j]
                r_eff = (r_i * r_j) / (r_i + r_j)
                t_eff = torch.minimum(self.base.thickness[i], self.base.thickness[j])
                k_n = 2.0 * self.base.contact.crushing_strength * t_eff * torch.sqrt(
                    2.0 * r_eff / (self.base.contact.reference_penetration_fraction * 2.0 * r_eff))
                e_pair = 0.5 * k_n * pen ** 2
                energy.index_add_(0, i, e_pair)
                energy.index_add_(0, j, e_pair)

        return energy

    def _apply_breakup(self, t: float) -> int:
        """Check for and apply breakup events. Returns number of events."""
        fn = self._compute_floe_contact_forces(t)
        e_contact = self._compute_contact_elastic_energy()

        should_break = self.breakup.check_breakup(
            fn, self.base.radius, self.base.thickness
        )

        break_indices = should_break.nonzero(as_tuple=False).flatten()
        if break_indices.numel() == 0:
            return 0

        max_bu = self.breakup.config.max_breakups_per_step
        if break_indices.numel() > max_bu:
            break_indices = break_indices[:max_bu]

        n_events = 0
        new_floes: List[Dict] = []
        remove_mask = torch.zeros(self.base.n, dtype=torch.bool)

        for idx in break_indices:
            i = int(idx)
            if i >= self.base.n:
                continue

            parent_r = float(self.base.radius[i])
            parent_t = float(self.base.thickness[i])
            parent_pos = (float(self.base.pos[i, 0]), float(self.base.pos[i, 1]))
            parent_vel = (float(self.base.vel[i, 0]), float(self.base.vel[i, 1]))
            parent_omega = float(self.base.omega[i])
            parent_mass = float(self.base.mass[i])
            elastic_e = float(e_contact[i])

            children, event = self.breakup.perform_breakup(
                parent_index=i,
                parent_radius=parent_r,
                parent_thickness=parent_t,
                parent_pos=parent_pos,
                parent_vel=parent_vel,
                parent_omega=parent_omega,
                parent_mass=parent_mass,
                contact_elastic_energy=elastic_e,
                time=t,
            )

            remove_mask[i] = True
            for child in children:
                new_floes.append(child)
            n_events += 1

        if n_events == 0:
            return 0

        # Rebuild simulation arrays
        keep = ~remove_mask

        new_radius = self.base.radius[keep].clone()
        new_thickness = self.base.thickness[keep].clone()
        new_pos = self.base.pos[keep].clone()
        new_vel = self.base.vel[keep].clone()
        new_omega = self.base.omega[keep].clone()
        new_z = self.base.z[keep].clone()
        new_vz = self.base.vz[keep].clone()
        new_tilt = self.base.tilt[keep].clone()
        new_tilt_rate = self.base.tilt_rate[keep].clone()
        new_yaw = self.base.yaw[keep].clone()

        for child in new_floes:
            r = child["radius"]
            t = child["thickness"]
            cx, cy = child["pos"]
            cvx, cvy = child["vel"]

            new_radius = torch.cat([new_radius, torch.tensor([r], dtype=_DTYPE)])
            new_thickness = torch.cat([new_thickness, torch.tensor([t], dtype=_DTYPE)])
            new_pos = torch.cat([new_pos, torch.tensor([[cx, cy]], dtype=_DTYPE)])
            new_vel = torch.cat([new_vel, torch.tensor([[cvx, cvy]], dtype=_DTYPE)])
            new_omega = torch.cat([new_omega, torch.tensor([child["omega"]], dtype=_DTYPE)])

            z_eq = 0.5 * t - (self._water.ice_density / self._water.water_density) * t
            new_z = torch.cat([new_z, torch.tensor([z_eq], dtype=_DTYPE)])
            new_vz = torch.cat([new_vz, torch.tensor([0.0], dtype=_DTYPE)])
            new_tilt = torch.cat([new_tilt, torch.tensor([0.0], dtype=_DTYPE)])
            new_tilt_rate = torch.cat([new_tilt_rate, torch.tensor([0.0], dtype=_DTYPE)])
            new_yaw = torch.cat([new_yaw, torch.tensor([0.0], dtype=_DTYPE)])

        n_total = new_radius.shape[0]

        self.base.n = n_total
        self.base.radius = new_radius
        self.base.thickness = new_thickness
        self.base.pos = new_pos
        self.base.vel = new_vel
        self.base.omega = new_omega
        self.base.z = new_z
        self.base.vz = new_vz
        self.base.tilt = new_tilt
        self.base.tilt_rate = new_tilt_rate
        self.base.yaw = new_yaw

        self.base.waterplane_area = math.pi * new_radius ** 2
        self.base.mass = self._water.ice_density * self.base.waterplane_area * new_thickness
        self.base.inertia_yaw = 0.5 * self.base.mass * new_radius ** 2
        self.base.inertia_tilt = 0.25 * self.base.mass * new_radius ** 2

        return n_events

    def step(self) -> None:
        """Advance one time step with breakup."""
        self.base.step()
        n_events = self._apply_breakup(self.base.time)

        fn = self._compute_floe_contact_forces(self.base.time)
        step_record = {
            "time": self.base.time,
            "n_floes": self.base.n,
            "n_breakup_events": n_events,
            "max_contact_force_N": float(fn.max()) if fn.numel() > 0 else 0.0,
            "mean_contact_force_N": float(fn.mean()) if fn.numel() > 0 else 0.0,
        }
        self.step_ledger.append(step_record)

    def run(self, n_steps: int, record_every: int = 1) -> IntegratedResult:
        """Run the integrated simulation for n_steps."""
        for step_idx in range(n_steps):
            self.step()
            if (step_idx + 1) % record_every == 0:
                if self.base.hulls:
                    f_hull = self.base.hull_contact(self.base.time)
                    total_fn = sum(
                        torch.linalg.norm(entry["force_on_hull_N"]).item()
                        for entry in f_hull
                    )
                else:
                    total_fn = 0.0

                speed = torch.linalg.norm(self.base.vel, dim=1)
                self._hull_forces.append(total_fn)
                self._mean_speeds.append(float(speed.mean()))
                self._n_floes.append(self.base.n)
                self._times.append(self.base.time)

        if self._hull_forces:
            mean_fn = sum(self._hull_forces) / len(self._hull_forces)
            max_fn = max(self._hull_forces)
            std_fn = (sum((f - mean_fn) ** 2 for f in self._hull_forces) / len(self._hull_forces)) ** 0.5
            mean_speed = sum(self._mean_speeds) / len(self._mean_speeds)
            max_speed = max(self._mean_speeds)
        else:
            mean_fn = max_fn = std_fn = mean_speed = max_speed = 0.0

        return IntegratedResult(
            mean_hull_force_N=mean_fn,
            max_hull_force_N=max_fn,
            std_hull_force_N=std_fn,
            mean_floe_speed_m_s=mean_speed,
            max_floe_speed_m_s=max_speed,
            n_breakup_events=self.ledger.n_events,
            n_children_produced=self.ledger.n_children_produced,
            energy_residual=self.ledger.energy_residual,
            mass_conservation_error=self.ledger.mass_conservation_error,
            time_history=self._times,
            hull_force_history=self._hull_forces,
            mean_speed_history=self._mean_speeds,
            n_floes_history=self._n_floes,
            ledger_summary=self.ledger.summary(),
        )
