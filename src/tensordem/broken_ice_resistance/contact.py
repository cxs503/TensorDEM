"""Contact laws for the broken-ice direct-navigation DEM.

Two laws are implemented behind one interface so the campaign can A/B-test them
on the *same* floe field and the *same* ship outline:

``crushing``  (primary)  ``F_n = sigma_c * h * min(2 sqrt(2 R_eff delta), 2 R_eff)``
              The Hertzian chord of a disk pressed ``delta`` into a plane is
              ``w = 2 sqrt(2 R delta)``; the contact pressure is capped at the
              ice crushing strength ``sigma_c``, so the normal force is a
              *displacement-dependent area times strength* -- no free stiffness.
              A viscous term ``c_n = 2 zeta sqrt((dF/ddelta) m)`` stabilises it.
``penalty``   (baseline) ``F_n = k_n delta`` with
              ``k_n = 2 sigma_c h / sqrt(reference_penetration_fraction)``,
              i.e. ``tensordem.ice_channel``'s linear spring calibrated at a 1 %
              penetration of the floe diameter.  It is stiffness-calibrated at
              one point only and therefore over-stiffens at large penetration.

Tangential force is a Coulomb cone ``|F_t| <= mu F_n`` with velocity damping,
matching the baseline model; the hull is a :class:`ShipForm` convex outline
queried by signed distance, so it works for both the wedge-bow and block forms.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from ..ice_channel import IceChannelSimulation
from .hull import ShipForm


class BrokenIceChannel(IceChannelSimulation):
    """Plan-view floe DEM with a crushing-limited contact law and a ship outline.

    ``law`` selects ``"crushing"`` (primary) or ``"penalty"`` (baseline); every
    other mechanism (heave, tilt, yaw, water/wind drag, integrator) is inherited
    unchanged from :class:`tensordem.ice_channel.IceChannelSimulation`.
    """

    law: str = "crushing"
    chord_regularisation_m: float = 1.0e-3
    chord_cap_diameters: float = 1.0

    # -- construction -------------------------------------------------------
    @classmethod
    def from_case(cls, spec, *, dtype=torch.float64) -> "BrokenIceChannel":
        """Build the floe field and the ship of a :class:`CaseSpec`."""
        from ..ice_channel import ContactParameters, WaterEnvironment

        contact = ContactParameters(
            crushing_strength=spec.compressive_strength_pa,
            friction=spec.friction_coefficient,
        )
        water = WaterEnvironment(wind_velocity=(0.0, 0.0))
        # the floe field occupies x in [0, L_domain] and y in [0, W_domain]
        # (tensordem.ice_channel.from_field clamps floe centres into that box), so
        # the track runs down the middle of the field and starts just ahead of the
        # ice: a stem inside the box at t = 0 would pre-compress a floe and inject
        # a spurious startup impulse into the whole field.
        hull = ShipForm(length=105.0, beam=26.0, speed=spec.speed_m_s,
                        entry_angle_deg=18.0, draught=8.0,
                        y_position=0.5 * spec.domain_width_m, x_start=-1.0,
                        form=spec.hull_form)
        if spec.null_field:
            # field-free control: a single 5 cm floe parked in the far corner of
            # the domain, i.e. ice material that the transit can never reach.
            sim = cls([0.05], [spec.thickness_m], hull=hull, water=water,
                      contact=contact, dt=spec.time_step_s,
                      integrator="verlet", dtype=dtype)
            sim.pos = torch.tensor(
                [[spec.domain_length_m - 0.1, spec.domain_width_m - 0.1]],
                dtype=dtype)
            ratio = sim.water.ice_density / sim.water.water_density
            sim.z = 0.5 * sim.thickness - ratio * sim.thickness
            sim.domain = (spec.domain_length_m, spec.domain_width_m)
            sim.target_concentration = 0.0
            sim.seed = int(spec.seed)
            sim.law = spec.law
            sim.case = spec
            return sim
        lo = 0.4 * spec.mean_floe_radius_m
        hi = 1.6 * spec.mean_floe_radius_m
        sim = cls.from_field(
            spec.concentration,
            domain_length=spec.domain_length_m,
            domain_width=spec.domain_width_m,
            seed=spec.seed,
            mean_radius=spec.mean_floe_radius_m,
            radius_jitter=0.35,
            radius_range=(lo, hi),
            thickness_range=(spec.thickness_m, spec.thickness_m),
            hull=hull, water=water, contact=contact,
            dt=spec.time_step_s, integrator="verlet", dtype=dtype,
        )
        sim.law = spec.law
        sim.case = spec
        return sim

    # -- contact law --------------------------------------------------------
    def _normal_force(self, pen: torch.Tensor, r_eff: torch.Tensor,
                      t_eff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normal force magnitude and tangent stiffness for a penetration."""
        sigma = torch.as_tensor(self.contact.crushing_strength, dtype=pen.dtype)
        if self.law == "penalty":
            k_n = 2.0 * sigma * t_eff / math.sqrt(
                self.contact.reference_penetration_fraction)
            return k_n * pen, k_n
        eps = torch.as_tensor(self.chord_regularisation_m, dtype=pen.dtype)
        width = 2.0 * torch.sqrt(torch.clamp(2.0 * r_eff * (pen + eps), min=0.0))
        width = torch.clamp(width, max=self.chord_cap_diameters * 2.0 * r_eff)
        force = sigma * t_eff * width
        stiffness = sigma * t_eff * torch.sqrt(
            2.0 * r_eff / torch.clamp(pen + eps, min=eps))
        return force, stiffness

    def _tangential(self, normal, tangent, v_rel, f_mag, stiffness, mass):
        v_t = (v_rel * tangent).sum(dim=1)
        c_t = 2.0 * self.contact.tangential_damping_ratio * torch.sqrt(
            torch.clamp(stiffness * mass, min=0.0))
        f_t = torch.clamp(c_t * v_t, min=-self.contact.friction * f_mag,
                          max=self.contact.friction * f_mag)
        return tangent * (-f_t)[:, None]

    # -- hull contact -------------------------------------------------------
    def _hull_force(self, t: float, pos: torch.Tensor, hull=None):
        hull = self.hull if hull is None else hull
        signed, normal = hull.signed_distance(pos, hull.x_centre(t))
        pen = torch.clamp(self.radius - signed, min=0.0)
        f_mag, stiffness = self._normal_force(pen, self.radius, self.thickness)
        v_rel = self.vel - hull.velocity().to(self.dtype)
        v_n = (v_rel * normal).sum(dim=1)
        c_n = 2.0 * self.contact.normal_damping_ratio * torch.sqrt(
            torch.clamp(stiffness * self.mass, min=0.0))
        f_mag = torch.clamp(f_mag - c_n * v_n, min=0.0)
        f_mag = torch.where(pen > 0.0, f_mag, torch.zeros_like(f_mag))
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
        force = normal * f_mag[:, None] + self._tangential(
            normal, tangent, v_rel, f_mag, stiffness, self.mass)
        arm = -(self.radius - 0.5 * pen)
        torque = arm * (normal[:, 0] * force[:, 1] - normal[:, 1] * force[:, 0])
        return force, torque

    # -- floe-floe contact --------------------------------------------------
    def _floe_floe_contacts(self):
        pos, vel, n = self.pos, self.vel, self.n
        if n < 2:
            empty = torch.zeros(0, dtype=self.dtype)
            return (empty, empty, empty, empty, empty)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
        overlap = (self.radius[:, None] + self.radius[None, :]) - dist
        eye = torch.eye(n, dtype=torch.bool)
        overlap = torch.where(eye, torch.zeros_like(overlap), overlap)
        iu, ju = torch.triu_indices(n, n, offset=1)
        pen = torch.clamp(overlap[iu, ju], min=0.0)
        active = pen > 0.0
        if not bool(active.any()):
            empty = torch.zeros(0, dtype=self.dtype)
            return (empty, empty, empty, empty, empty)
        i, j = iu[active], ju[active]
        pen = pen[active]
        normal = delta[i, j] / dist[i, j][:, None]
        r_eff = 0.5 * (self.radius[i] + self.radius[j])
        t_eff = 0.5 * (self.thickness[i] + self.thickness[j])
        f_mag, stiffness = self._normal_force(pen, r_eff, t_eff)
        m_eff = 0.5 * (self.mass[i] + self.mass[j])
        v_rel = vel[i] - vel[j]
        v_n = (v_rel * normal).sum(dim=1)
        c_n = 2.0 * self.contact.normal_damping_ratio * torch.sqrt(
            torch.clamp(stiffness * m_eff, min=0.0))
        f_mag = torch.clamp(f_mag - c_n * v_n, min=0.0)
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
        f_pair = normal * f_mag[:, None] + self._tangential(
            normal, tangent, v_rel, f_mag, stiffness, m_eff)
        # contact lever arms about each floe centre, as vectors along the normal
        # (the parent class takes ``arm_pt`` vectors, not scalar torques)
        arm_i = -(self.radius[i] - 0.5 * pen)
        arm_j = -(self.radius[j] - 0.5 * pen)
        return (torch.stack((i, j), dim=1), f_pair,
                normal * arm_i[:, None], normal * arm_j[:, None], f_pair)


class PenaltyIceChannel(BrokenIceChannel):
    """Baseline: identical ship outline and field, linear penalty contact law.

    The active law is carried by the :class:`CaseSpec` (``law="penalty"``); the
    class attribute is only the default so that an ad-hoc instantiation keeps the
    baseline behaviour.
    """

    law = "penalty"