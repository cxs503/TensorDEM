"""Contact laws of the steady-state transit surface, with their potentials.

Two things are new relative to :mod:`tensordem.ice_channel` /
:mod:`tensordem.broken_ice_resistance`:

1. **Zero-offset crushing law.**  The chord of a disk pressed ``delta`` into a
   plane is ``w = 2 sqrt(2 R delta)``; capping the contact pressure at the
   crushing strength gives ``F_n = sigma_c h w``.  The tangent stiffness
   ``dF/ddelta = sigma_c h sqrt(2R/delta)`` diverges at ``delta -> 0``, so the
   existing surfaces evaluate the chord at ``delta + eps`` -- but without
   subtracting the offset, which leaves a finite force at *zero* penetration
   (``sigma_c h 2 sqrt(2 R eps)``, i.e. 1.1 MN for a 5 m, 1 m thick floe with
   ``eps = 5 mm``).  Here the offset is removed,

       ``F_n(delta) = sigma_c h [2 sqrt(2 R (delta+eps)) - 2 sqrt(2 R eps)]``

   capped at ``2 R`` (the floe is fully engaged once the chord reaches its
   diameter).  ``F_n(0) = 0`` exactly, the tangent stiffness is finite, and the
   regularisation length has a physical meaning:
   ``eps = sigma_c R / E`` is the penetration at which an elastic ice edge of
   curvature radius ``R`` reaches the crushing strength (``E`` = ice modulus).

2. **Closed-form potential.**  ``U(delta) = int_0^delta F_n(s) ds`` is available
   in closed form, so a contact carries an *elastic* energy that can be summed
   over the whole field.  Together with the dissipative power of the dashpots and
   of Coulomb sliding this closes the energy balance reported by
   :class:`tensordem.ice_transit.transit.SteadyTransit`.

Sign conventions used everywhere below (they are what makes the ledger exact):

* ``F_i`` is the force on the *first* index of a pair, ``v_rel = v_i - v_j``;
  the dissipation rate of the contact is ``-F_i . v_rel``.
* normal part: ``-f_n v_n = -elastic*v_n + (elastic - f_n) v_n``.  The first
  term is the elastic power (book-kept through ``U``), the second is the normal
  dashpot dissipation ``= c_n v_n^2`` when the contact is not separating.
* tangential part: the force is ``-t f_t``, so its dissipation is ``+ f_t v_t``
  (``= c_t v_t^2`` below the Coulomb cap, ``= mu f_n |v_t|`` at the cap).

The tangential law is the repository's Coulomb cone with velocity damping
(``|F_t| <= mu F_n``); it has no tangential spring, so all tangential work is
dissipative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

__all__ = [
    "CrushingLaw",
    "polygon_distance",
    "floe_floe_contacts",
    "hull_contacts",
    "wall_contacts",
]


def polygon_distance(pos: torch.Tensor, points: torch.Tensor,
                     normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact signed distance of points to a convex polygon (segments).

    ``ShipForm.signed_distance`` reports ``max_k n_k . (p - v_k)``, i.e. the
    distance to the nearest *face plane*.  For a point outside a convex polygon
    that value is a lower bound of the true distance and it is badly wrong near a
    vertex: with the 18 deg bow apex, a floe 7 m ahead of the stem is reported at
    a face-plane distance of 2.2 m, so a 5 m floe is put **in contact** (and, with
    a strength-limited law, that is a ~24 MJ spurious contact) before it touches
    the hull at all.  This routine returns the true point-to-boundary distance
    (min over segments) with the sign of the inside test, plus the outward
    normal, so the bow-apex contact is resolved correctly.
    """
    dtype = pos.dtype
    next_points = torch.roll(points, shifts=-1, dims=0)
    d = (pos[:, None, :] - points[None, :, :]) * normals[None, :, :]
    plane = d.sum(dim=-1)
    inside = plane.max(dim=1).values <= 0.0
    segments = next_points - points
    rel = pos[:, None, :] - points[None, :, :]
    denom = torch.clamp((segments ** 2).sum(dim=-1), min=1e-30)
    t = torch.clamp((rel * segments[None, :, :]).sum(dim=-1) / denom[None, :],
                    min=0.0, max=1.0)
    closest = points[None, :, :] + t[:, :, None] * segments[None, :, :]
    gap = pos[:, None, :] - closest
    distance = torch.linalg.norm(gap, dim=-1)
    index = distance.argmin(dim=1)
    nearest = distance.gather(1, index[:, None]).squeeze(1)
    nearest_point = closest.gather(
        1, index[:, None, None].expand(-1, 1, 2)).squeeze(1)
    outward = pos - nearest_point
    length = torch.linalg.norm(outward, dim=1, keepdim=True)
    fallback = normals.gather(0, index[:, None].expand(-1, 2))
    normal = torch.where(length > 1e-12, outward / torch.clamp(length, min=1e-30),
                         fallback)
    signed = torch.where(inside, -nearest, nearest)
    normal = torch.where(inside[:, None], -normal, normal)
    return signed, normal


@dataclass(frozen=True)
class CrushingLaw:
    """Zero-offset crushing contact law with its closed-form potential.

    ``sigma_c``        ice crushing strength [Pa]
    ``modulus``        ice elastic modulus [Pa], used only through ``eps``
    ``cap_diameters``  chord cap in floe diameters (1.0 = full diameter)
    """

    sigma_c: float
    modulus: float
    cap_diameters: float = 1.0

    def __post_init__(self) -> None:
        for name in ("sigma_c", "modulus", "cap_diameters"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    # -- regularisation -----------------------------------------------------
    def eps(self, r_eff: torch.Tensor) -> torch.Tensor:
        """Elastic penetration at crushing onset, ``sigma_c R / E`` [m]."""
        return self.sigma_c * r_eff / self.modulus

    def chord(self, pen: torch.Tensor, r_eff: torch.Tensor) -> torch.Tensor:
        """Engaged chord width [m]; zero at zero penetration, capped at ``2 R``."""
        eps = self.eps(r_eff)
        raw = 2.0 * torch.sqrt(2.0 * r_eff * (torch.clamp(pen, min=0.0) + eps)) \
            - 2.0 * torch.sqrt(2.0 * r_eff * eps)
        cap = self.cap_diameters * 2.0 * r_eff
        return torch.clamp(raw, min=torch.zeros_like(raw), max=cap)

    def force(self, pen: torch.Tensor, r_eff: torch.Tensor,
              h_eff: torch.Tensor) -> torch.Tensor:
        """Normal contact force [N]; exactly zero at zero penetration."""
        return self.sigma_c * h_eff * self.chord(pen, r_eff)

    def potential(self, pen: torch.Tensor, r_eff: torch.Tensor,
                  h_eff: torch.Tensor) -> torch.Tensor:
        """``int_0^pen F_n(s) ds`` [J] (elastic energy stored in the contact)."""
        pen = torch.clamp(pen, min=0.0)
        eps = self.eps(r_eff)
        root = torch.sqrt(2.0 * r_eff)
        offset = 2.0 * torch.sqrt(2.0 * r_eff * eps)
        cap = self.cap_diameters * 2.0 * r_eff
        # penetration at which the uncapped chord reaches the cap
        pen_cap = torch.clamp((cap / 2.0 + offset / 2.0) ** 2 / (2.0 * r_eff) - eps,
                              min=0.0)

        def _uncapped(x: torch.Tensor) -> torch.Tensor:
            return self.sigma_c * h_eff * (
                (4.0 / 3.0) * root * (torch.pow(x + eps, 1.5) - torch.pow(eps, 1.5))
                - offset * x)

        capped_force = self.sigma_c * h_eff * torch.clamp(cap - offset, min=0.0)
        below = _uncapped(torch.minimum(pen, pen_cap))
        beyond = _uncapped(pen_cap) + capped_force * torch.clamp(pen - pen_cap, min=0.0)
        return torch.where(pen <= pen_cap, below, beyond)

    def tangent_stiffness(self, pen: torch.Tensor, r_eff: torch.Tensor,
                          h_eff: torch.Tensor) -> torch.Tensor:
        """``dF_n/ddelta`` [N/m], finite at ``delta = 0`` (equals ``dU/ddelta``)."""
        eps = self.eps(r_eff)
        grad = self.sigma_c * h_eff * (2.0 * r_eff) / torch.sqrt(
            2.0 * r_eff * (torch.clamp(pen, min=0.0) + eps))
        cap = self.cap_diameters * 2.0 * r_eff
        engaged = self.chord(pen, r_eff) < (cap - 1e-12)
        return torch.where(engaged, grad, torch.zeros_like(grad))

    def as_dict(self) -> dict[str, float]:
        return {"sigma_c_pa": self.sigma_c, "modulus_pa": self.modulus,
                "cap_diameters": self.cap_diameters}


def _empty(pos: torch.Tensor) -> dict:
    dtype = pos.dtype
    empty_i = torch.zeros(0, dtype=torch.long)
    return {
        "force": torch.zeros_like(pos), "torque": torch.zeros(pos.shape[0], dtype=dtype),
        "count": 0, "potential": torch.zeros((), dtype=dtype),
        "p_normal_damp": torch.zeros((), dtype=dtype),
        "p_tangential": torch.zeros((), dtype=dtype),
        "i": empty_i, "j": empty_i, "pen": torch.zeros(0, dtype=dtype),
    }


def floe_floe_contacts(pos: torch.Tensor, vel: torch.Tensor, omega: torch.Tensor,
                       radius: torch.Tensor, thickness: torch.Tensor,
                       mass: torch.Tensor, law: CrushingLaw, friction: float,
                       normal_damping_ratio: float,
                       tangential_damping_ratio: float) -> dict:
    """Instrumented floe-floe contact kernel (all ``i < j`` pairs)."""
    n = pos.shape[0]
    if n < 2:
        return _empty(pos)
    iu, ju = torch.triu_indices(n, n, offset=1)
    delta = pos[iu] - pos[ju]
    dist = torch.sqrt((delta ** 2).sum(dim=1) + 1e-30)
    pen_all = (radius[iu] + radius[ju]) - dist
    active = pen_all > 0.0
    if not bool(active.any()):
        return _empty(pos)
    i, j = iu[active], ju[active]
    pen = pen_all[active]
    normal = delta[active] / dist[active][:, None]

    r_eff = 0.5 * (radius[i] + radius[j])
    h_eff = 0.5 * (thickness[i] + thickness[j])
    m_eff = (mass[i] * mass[j]) / (mass[i] + mass[j])
    elastic = law.force(pen, r_eff, h_eff)
    stiffness = law.tangent_stiffness(pen, r_eff, h_eff)
    tangent_dir = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
    arm_i = -(radius[i] - 0.5 * pen)
    arm_j = (radius[j] - 0.5 * pen)
    v_i = vel[i] + (omega[i] * arm_i)[:, None] * tangent_dir
    v_j = vel[j] + (omega[j] * arm_j)[:, None] * tangent_dir
    v_rel = v_i - v_j
    v_n = (v_rel * normal).sum(dim=1)
    c_n = 2.0 * normal_damping_ratio * torch.sqrt(torch.clamp(stiffness * m_eff, min=0.0))
    f_n = torch.clamp(elastic - c_n * v_n, min=torch.zeros_like(pen))
    v_t = (v_rel * tangent_dir).sum(dim=1)
    c_t = 2.0 * tangential_damping_ratio * torch.sqrt(torch.clamp(stiffness * m_eff, min=0.0))
    f_t = torch.clamp(c_t * v_t, min=-friction * f_n, max=friction * f_n)
    f_pair = normal * f_n[:, None] - tangent_dir * f_t[:, None]

    force = torch.zeros_like(pos)
    torque = torch.zeros(n, dtype=pos.dtype)
    force.index_add_(0, i, f_pair)
    force.index_add_(0, j, -f_pair)
    torque.index_add_(0, i, arm_i * (normal[:, 0] * f_pair[:, 1] - normal[:, 1] * f_pair[:, 0]))
    torque.index_add_(0, j, arm_j * (normal[:, 0] * (-f_pair[:, 1]) - normal[:, 1] * (-f_pair[:, 0])))

    return {
        "force": force, "torque": torque, "count": int(i.numel()),
        "potential": law.potential(pen, r_eff, h_eff).sum(),
        "p_normal_damp": ((elastic - f_n) * v_n).sum(),
        "p_tangential": (f_t * v_t).sum(),
        "i": i, "j": j, "pen": pen,
    }


def hull_contacts(pos: torch.Tensor, vel: torch.Tensor, omega: torch.Tensor,
                  radius: torch.Tensor, thickness: torch.Tensor,
                  mass: torch.Tensor, law: CrushingLaw,
                  friction: float, normal_damping_ratio: float,
                  tangential_damping_ratio: float,
                  hull_dist: torch.Tensor, hull_normal: torch.Tensor,
                  hull_velocity: torch.Tensor) -> dict:
    """Instrumented floe-hull contact kernel (signed-distance hull query).

    ``hull_dist``/``hull_normal`` come from
    :meth:`tensordem.broken_ice_resistance.hull.ShipForm.signed_distance`
    (``s > 0`` outside the outline).  A floe penetrates by ``R - s``.
    """
    dtype = pos.dtype
    pen = torch.clamp(radius - hull_dist, min=0.0)
    active = pen > 0.0
    if not bool(active.any()):
        out = _empty(pos)
        out["active"] = active
        return out
    normal = hull_normal
    arm_contact = -(radius - 0.5 * pen)
    tangent_all = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
    # rim velocity of the floe at the contact, including its yaw rate
    v_rel = vel + (omega * arm_contact)[:, None] * tangent_all - hull_velocity
    elastic = law.force(pen, radius, thickness)
    stiffness = law.tangent_stiffness(pen, radius, thickness)
    v_n = (v_rel * normal).sum(dim=1)
    c_n = 2.0 * normal_damping_ratio * torch.sqrt(torch.clamp(stiffness * mass, min=0.0))
    f_n = torch.clamp(elastic - c_n * v_n, min=torch.zeros_like(pen))
    f_n = torch.where(active, f_n, torch.zeros_like(f_n))
    tangent_dir = tangent_all
    v_t = (v_rel * tangent_dir).sum(dim=1)
    c_t = 2.0 * tangential_damping_ratio * torch.sqrt(torch.clamp(stiffness * mass, min=0.0))
    f_t = torch.clamp(c_t * v_t, min=-friction * f_n, max=friction * f_n)
    f_t = torch.where(active, f_t, torch.zeros_like(f_t))
    force = normal * f_n[:, None] - tangent_dir * f_t[:, None]
    torque = arm_contact * (normal[:, 0] * force[:, 1] - normal[:, 1] * force[:, 0])
    p_normal_damp = torch.where(active, (elastic - f_n) * v_n, torch.zeros_like(pen))
    p_tangential = torch.where(active, f_t * v_t, torch.zeros_like(pen))
    potential = law.potential(pen, radius, thickness)
    return {
        "force": force, "torque": torque, "count": int(active.sum()),
        "potential": potential.sum(),
        "p_normal_damp": p_normal_damp.sum(),
        "p_tangential": p_tangential.sum(),
        "active": active, "pen": pen,
    }


def wall_contacts(pos: torch.Tensor, vel: torch.Tensor, radius: torch.Tensor,
                  half_width: float, stiffness_per_floe: torch.Tensor,
                  damping_ratio: float, mass: torch.Tensor) -> dict:
    """Compliant pack boundary at ``y = 0`` and ``y = half_width``.

    The boundary represents the *surrounding ice pack*, not a rigid tank: its
    stiffness is anchored to the floe's own hydrostatic stiffness
    ``rho_w g A_wp`` -- the stiffness a floe feels when it is pushed into a pack
    -- so it is a declared model choice whose sensitivity to ``half_width`` is
    reported.  Force direction ``u`` points into the domain
    (``u = -e_y`` at the high wall, ``u = +e_y`` at the low wall).
    """
    dtype = pos.dtype
    y = pos[:, 1]
    pen_low = torch.clamp(radius - y, min=0.0)
    pen_high = torch.clamp(y + radius - half_width, min=0.0)
    pen = pen_low + pen_high
    if not bool((pen > 0.0).any()):
        out = _empty(pos)
        out["p_dissipation"] = torch.zeros((), dtype=dtype)
        return out
    direction = torch.where(pen_high > 0.0, -torch.ones_like(y), torch.ones_like(y))
    k = stiffness_per_floe
    v_out = -(vel[:, 1] * direction)
    c = 2.0 * damping_ratio * torch.sqrt(torch.clamp(k * mass, min=0.0))
    f_n = torch.clamp(k * pen + c * v_out, min=torch.zeros_like(pen))
    f_n = torch.where(pen > 0.0, f_n, torch.zeros_like(f_n))
    force = torch.zeros_like(pos)
    force[:, 1] = direction * f_n
    potential = 0.5 * (k * pen ** 2).sum()
    p_elastic = (k * pen * direction * vel[:, 1]).sum()
    p_total = (force[:, 1] * vel[:, 1]).sum()
    return {"force": force, "torque": torch.zeros(pos.shape[0], dtype=dtype),
            "count": int((pen > 0.0).sum()), "potential": potential,
            "p_dissipation": -(p_total - p_elastic),
            "p_normal_damp": torch.zeros((), dtype=dtype),
            "p_tangential": torch.zeros((), dtype=dtype)}