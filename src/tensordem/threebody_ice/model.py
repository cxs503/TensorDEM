"""Plan-view DEM core for hull-floe-floe (three-body) contact dynamics.

Scope and modelling choices (declared; see ``docs/P262W1_N04_THREEBODY_ICE.md``)
---------------------------------------------------------------------------------
The floes are *flat-lying rigid disks* in the horizontal plane: 2 translations
and 1 rotation per floe, hydrostatic heave relaxed (the flat-lying assumption
already used by :mod:`tensordem.ice_channel` and :mod:`tensordem.ice_ice_collision`).
The ship is a **kinematic** rigid outline (infinite mass, prescribed velocity
``V e_x``) taken from :class:`tensordem.broken_ice_resistance.hull.ShipForm`.

Contacts are the Cundall-Strack (1979) linear spring-dashpot normal law plus a
Coulomb tangential slider::

    F_n = k_n delta + c_n delta_dot,        c_n = 2 beta sqrt(k_n m_eff)
    F_t = -mu |F_n| sign(v_t)               (sliding only, no tangential spring)

The pair restitution of a linear spring-dashpot with damping ratio
``beta < 1`` is ``e = exp(-beta pi / sqrt(1 - beta^2))``, so the *target*
coefficient of restitution is an input parameter and is verified against the
measured value by the validation surface (``threebody.elastic_reference``).

The dashpot damping ``c_n`` is built from the *pair* reduced mass
(``m_eff = m_i m_j / (m_i + m_j)`` for floe-floe, ``m_eff = m_i`` for
hull-floe), which is what makes the restitution mass-independent.

Water drag on a floe is the quadratic side-wall law of
:mod:`tensordem.ice_channel` (``F = -1/2 rho_w C_d (2 R t) |v| v``), i.e. the
submerged projected area times a drag coefficient; no current and no wind are
applied.  There is no added mass and no heave forcing (declared limitations).

Energy accounting (*every* term is measured, nothing is inferred)
---------------------------------------------------------------
Per step the model accumulates

``w_hull``   work done by the hull on the ice, ``sum F_hull->floe . v_hull``
``w_drag``   work done by water drag on the floes (negative)
``e_diss_n`` dashpot dissipation, ``sum c_n delta_dot^2``
``e_diss_t`` friction dissipation, ``sum mu |F_n| |v_t|``
``e_out``    kinetic energy carried out of the domain by removed floes

so that, with ``E = E_kin + E_spring``,

    dE/dt = w_hull + w_drag - e_diss_n - e_diss_t        (removals: -e_out)

is an identity of the discrete scheme up to the first-order work-rate
quadrature.  This identity is what the resistance decomposition of
:mod:`tensordem.threebody_ice.field` uses; the closure residual is reported,
not tuned away.

Units are SI: metres, seconds, kilograms, newtons.  All state is ``float64``
torch tensors; the reduction order is fixed so a seeded run is bit-identical.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..broken_ice_resistance.hull import ShipForm

DTYPE = torch.float64

__all__ = [
    "ContactLaw",
    "FloeState",
    "Ledger",
    "MultiBodyModel",
    "restitution_from_damping_ratio",
    "damping_ratio_from_restitution",
    "PARAMETER_LEDGER",
    "parameter_ledger_sha256",
]


# ---------------------------------------------------------------------------
# Parameters and their provenance grades
# ---------------------------------------------------------------------------

#: Graded parameter ledger.  ``REAL`` = material/measured value;
#: ``CONVENTION`` = declared modelling convention with no independent anchor
#: available offline; ``DEM-SOFT`` = numerically softened so that the explicit
#: integrator is stable at the declared step (the softening is stated).
PARAMETER_LEDGER: Dict[str, Tuple[float, str, str]] = {
    "ice_density_kg_m3": (917.0, "REAL", "columnar sea ice, 900-927 kg/m3 range"),
    "water_density_kg_m3": (1025.0, "REAL", "sea water"),
    "ice_floe_thickness_m": (0.8, "CONVENTION", "level first-year floe thickness"),
    "water_drag_coefficient": (1.0, "CONVENTION",
                               "normal flat-plate drag on the submerged side wall"),
    "ice_ice_friction": (0.15, "CONVENTION", "sliding friction of ice on ice"),
    "hull_ice_friction": (0.10, "CONVENTION", "steel-ice sliding friction (wet)"),
    "normal_stiffness_N_m": (2.0e7, "DEM-SOFT",
                             "softened contact stiffness (explicit stability); the "
                             "crushing stiffness at 1.5 MPa over a 2t sqrt(2 R d) "
                             "line contact is O(1e8-1e9 N/m)"),
    "restitution_floe_floe": (0.40, "CONVENTION",
                              "crushed-layer dissipation; the analytic single-collision "
                              "fractions (1-e)/2 = 0.30 and (1-e)/(1+e) = 0.43 are then "
                              "measured, not assumed"),
    "restitution_hull_floe": (0.40, "CONVENTION", "as above for steel-ice impact"),
}


def parameter_ledger_sha256() -> str:
    """SHA-256 over the graded parameter ledger (provenance for the artifacts)."""
    payload = json.dumps({k: [v[0], v[1], v[2]] for k, v in sorted(PARAMETER_LEDGER.items())},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def restitution_from_damping_ratio(beta: float) -> float:
    """Pair restitution of a linear spring-dashpot with damping ratio ``beta``."""
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("damping ratio must lie in [0, 1)")
    if beta == 0.0:
        return 1.0
    return math.exp(-beta * math.pi / math.sqrt(1.0 - beta * beta))


def damping_ratio_from_restitution(restitution: float) -> float:
    """Invert :func:`restitution_from_damping_ratio`."""
    e = float(restitution)
    if not 0.0 < e <= 1.0:
        raise ValueError("restitution must lie in (0, 1]")
    if e == 1.0:
        return 0.0
    lam = -math.log(e) / math.pi
    return lam / math.sqrt(1.0 + lam * lam)


@dataclass(frozen=True)
class ContactLaw:
    """Contact-law parameters shared by every contact of one simulation."""

    normal_stiffness: float = 2.0e7
    floe_floe_restitution: float = 0.60
    hull_floe_restitution: float = 0.60
    floe_floe_friction: float = 0.15
    hull_floe_friction: float = 0.10
    water_drag_coefficient: float = 1.0
    water_density: float = 1025.0
    ice_density: float = 917.0
    thickness: float = 0.8
    #: Optional numerically calibrated damping ratios (see
    #: :func:`tensordem.threebody_ice.threebody.calibrate_law`).  ``None`` means
    #: "use the analytic ``exp(-beta pi / sqrt(1-beta^2))`` inverse", which is
    #: only approximate for a linear spring-dashpot.
    beta_floe_floe_override: Optional[float] = None
    beta_hull_floe_override: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("normal_stiffness", "water_density", "ice_density", "thickness"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("floe_floe_restitution", "hull_floe_restitution"):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("floe_floe_friction", "hull_floe_friction",
                     "water_drag_coefficient"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def beta_floe_floe(self) -> float:
        if self.beta_floe_floe_override is not None:
            return float(self.beta_floe_floe_override)
        return damping_ratio_from_restitution(self.floe_floe_restitution)

    @property
    def beta_hull_floe(self) -> float:
        if self.beta_hull_floe_override is not None:
            return float(self.beta_hull_floe_override)
        return damping_ratio_from_restitution(self.hull_floe_restitution)

    def contact_period(self, reduced_mass: float) -> float:
        """Linear contact period for reduced mass ``reduced_mass`` [s]."""
        return 2.0 * math.pi * math.sqrt(float(reduced_mass) / self.normal_stiffness)

    def stable_time_step(self, reduced_mass: float, *, safety: float = 0.08) -> float:
        """Explicit step ``safety * contact_period`` for reduced mass."""
        return safety * self.contact_period(reduced_mass)

    def as_dict(self) -> Dict[str, float]:
        return {
            "normal_stiffness_N_m": self.normal_stiffness,
            "floe_floe_restitution": self.floe_floe_restitution,
            "hull_floe_restitution": self.hull_floe_restitution,
            "floe_floe_friction": self.floe_floe_friction,
            "hull_floe_friction": self.hull_floe_friction,
            "water_drag_coefficient": self.water_drag_coefficient,
            "beta_floe_floe": self.beta_floe_floe,
            "beta_hull_floe": self.beta_hull_floe,
        }


# ---------------------------------------------------------------------------
# Ledger of measured work / energy
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    """Cumulative measured work and energy terms of one simulation [J]."""

    w_hull: float = 0.0
    w_drag: float = 0.0
    e_diss_normal: float = 0.0
    e_diss_friction: float = 0.0
    e_out: float = 0.0
    e_out_clearing: float = 0.0      # pushed out of the swept corridor (lateral)
    e_out_wake: float = 0.0          # left behind the stern
    e_out_downstream: float = 0.0    # left through the downstream domain boundary
    n_removed: int = 0
    hull_contacts_peak: int = 0
    max_penetration_fraction: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"w_hull_J": self.w_hull, "w_drag_J": self.w_drag,
                "e_diss_normal_J": self.e_diss_normal,
                "e_diss_friction_J": self.e_diss_friction,
                "e_out_J": self.e_out, "e_out_clearing_J": self.e_out_clearing,
                "e_out_wake_J": self.e_out_wake,
                "e_out_downstream_J": self.e_out_downstream,
                "n_removed": self.n_removed,
                "hull_contacts_peak": self.hull_contacts_peak,
                "max_penetration_fraction": self.max_penetration_fraction}

    def copy(self) -> "Ledger":
        return Ledger(**self.__dict__)

    def difference(self, other: "Ledger") -> "Ledger":
        """``self - other`` for windowed accounting."""
        out = Ledger()
        for name, value in self.__dict__.items():
            if name == "n_removed":
                out.n_removed = int(value) - int(getattr(other, name))
            elif isinstance(value, float):
                setattr(out, name, value - getattr(other, name))
            else:
                setattr(out, name, value)
        return out


# ---------------------------------------------------------------------------
# Floe state
# ---------------------------------------------------------------------------

@dataclass
class FloeState:
    """Flat-lying circular floe state (2 translations + 1 rotation)."""

    radius: torch.Tensor
    pos: torch.Tensor
    vel: torch.Tensor
    angle: torch.Tensor
    omega: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    active: torch.Tensor

    @classmethod
    def create(cls, radius: Sequence[float], pos: Sequence[Sequence[float]], *,
               law: ContactLaw) -> "FloeState":
        r = torch.as_tensor(radius, dtype=DTYPE)
        p = torch.as_tensor(pos, dtype=DTYPE)
        if p.shape != (r.numel(), 2):
            raise ValueError("pos must be (N, 2) matching radius")
        mass = law.ice_density * math.pi * r ** 2 * law.thickness
        inertia = 0.5 * mass * r ** 2
        return cls(radius=r, pos=p.clone(), vel=torch.zeros_like(p),
                   angle=torch.zeros_like(r), omega=torch.zeros_like(r),
                   mass=mass, inertia=inertia,
                   active=torch.ones_like(r, dtype=torch.bool))

    @property
    def n(self) -> int:
        return int(self.radius.numel())

    def kinetic_energy(self) -> float:
        """Total planar kinetic energy of the active floes [J]."""
        mask = self.active.to(DTYPE)
        trans = 0.5 * (self.mass * mask * (self.vel ** 2).sum(dim=1)).sum()
        rot = 0.5 * (self.inertia * mask * self.omega ** 2).sum()
        return float(trans + rot)

    def momentum(self) -> torch.Tensor:
        """Total planar momentum of the active floes [kg m/s]."""
        mask = self.active.to(DTYPE)
        return (self.mass * mask).unsqueeze(1).mul(self.vel).sum(dim=0)

    def digest(self) -> str:
        h = hashlib.sha256()
        for t in (self.radius, self.pos, self.vel, self.angle, self.omega):
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def clone(self) -> "FloeState":
        return FloeState(self.radius.clone(), self.pos.clone(), self.vel.clone(),
                         self.angle.clone(), self.omega.clone(),
                         self.mass.clone(), self.inertia.clone(),
                         self.active.clone())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiBodyModel:
    """Hull (kinematic outline) + N flat-lying floes, with an energy ledger."""

    def __init__(self, state: FloeState, law: ContactLaw, *,
                 hull: Optional[ShipForm] = None,
                 hull_speed: float = 0.0,
                 domain_half_width: Optional[float] = None,
                 domain_length: Optional[float] = None,
                 clearing_distance_m: Optional[float] = None,
                 corridor_half_width: Optional[float] = None) -> None:
        self.state = state
        self.law = law
        self.hull = hull
        self.hull_speed = float(hull_speed)
        self.domain_half_width = domain_half_width
        self.domain_length = domain_length
        #: a floe this far behind the stem counts as *cleared from the channel*
        #: and is retired with its kinetic energy banked as the clearing flux
        self.clearing_distance_m = (clearing_distance_m
                                    if clearing_distance_m is not None
                                    else (hull.length if hull is not None else None))
        #: a floe pushed further than this from the centreline can no longer be
        #: touched by the hull, so it has been *cleared out of the swept corridor*
        #: and is retired with its kinetic energy banked
        self.corridor_half_width = corridor_half_width
        self.time = 0.0
        self.ledger = Ledger()
        self.last_hull_force = torch.zeros(2, dtype=DTYPE)
        self.last_hull_contacts = 0
        self.hull_impulse = torch.zeros(2, dtype=DTYPE)
        self.drag_impulse = torch.zeros(2, dtype=DTYPE)
        self._cache_pairs()

    # -- helpers ------------------------------------------------------------
    @property
    def hull_velocity(self) -> torch.Tensor:
        return torch.tensor([self.hull_speed, 0.0], dtype=DTYPE)

    def _fluid_drag(self) -> torch.Tensor:
        s, law = self.state, self.law
        speed = torch.linalg.norm(s.vel, dim=1, keepdim=True)
        area = 2.0 * s.radius * law.thickness
        f = (-0.5 * law.water_density * law.water_drag_coefficient
             * area.unsqueeze(1) * speed * s.vel)
        return f * s.active.to(DTYPE).unsqueeze(1)

    def _floe_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._pairs

    def _cache_pairs(self) -> None:
        n = self.state.n
        if n < 2:
            empty = torch.zeros(0, dtype=torch.long)
            self._pairs = (empty, empty)
        else:
            idx = torch.triu_indices(n, n, offset=1)
            self._pairs = (idx[0], idx[1])

    # -- forces -------------------------------------------------------------
    def forces(self) -> Dict[str, torch.Tensor]:
        """Forces, torques and the instantaneous ledger rates of the current state."""
        s, law = self.state, self.law
        n = s.n
        force = torch.zeros_like(s.pos)
        torque = torch.zeros_like(s.radius)
        zero = torch.zeros((), dtype=DTYPE)
        diss_normal = zero.clone()
        diss_friction = zero.clone()
        spring = zero.clone()
        max_pen = zero.clone()
        w_hull = zero.clone()
        hull_force = torch.zeros(2, dtype=DTYPE)
        hull_force_per_floe = torch.zeros_like(s.pos)
        n_hull_contacts = 0

        active = s.active

        # --- floe-floe (Cundall-Strack) ------------------------------------
        i, j = self._floe_pairs()
        if i.numel() > 0:
            dvec = s.pos[j] - s.pos[i]
            dist = torch.linalg.norm(dvec, dim=1)
            pen = s.radius[i] + s.radius[j] - dist
            keep = (pen > 0.0) & active[i] & active[j] & (dist > 1.0e-12)
            if bool(keep.any()):
                i, j = i[keep], j[keep]
                dist, pen = dist[keep], pen[keep]
                nvec = dvec[keep] / dist.unsqueeze(1)
                tvec = torch.stack([-nvec[:, 1], nvec[:, 0]], dim=1)
                m_eff = s.mass[i] * s.mass[j] / (s.mass[i] + s.mass[j])
                c_n = 2.0 * law.beta_floe_floe * torch.sqrt(law.normal_stiffness * m_eff)
                v_ij = s.vel[j] - s.vel[i]
                delta_dot = -(v_ij * nvec).sum(dim=1)
                f_n_mag = law.normal_stiffness * pen + c_n * delta_dot
                f_n_mag = torch.clamp(f_n_mag, min=0.0)
                # n points from i to j, so the repulsion on i is along -n
                f_n = -f_n_mag.unsqueeze(1) * nvec
                # relative tangential velocity at the contact point
                v_t = ((v_ij * tvec).sum(dim=1)
                       - s.omega[j] * s.radius[j] - s.omega[i] * s.radius[i])
                f_t_mag = law.floe_floe_friction * f_n_mag
                # F_on_i = mu|Fn| sign(v_t) t : the friction on i points along the
                # relative surface motion of j (see the module docstring)
                f_t = f_t_mag * torch.sign(v_t)
                f_t_vec = f_t.unsqueeze(1) * tvec
                total = f_n + f_t_vec
                force.index_add_(0, i, total)
                force.index_add_(0, j, -total)
                torque.index_add_(0, i, s.radius[i] * f_t)
                torque.index_add_(0, j, s.radius[j] * f_t)
                spring = spring + 0.5 * law.normal_stiffness * (pen ** 2).sum()
                diss_normal = diss_normal + (c_n * delta_dot ** 2).sum()
                diss_friction = diss_friction + (f_t_mag * v_t.abs()).sum()
                max_pen = torch.maximum(max_pen,
                                        (pen / (2.0 * s.radius[i])).max())

        # --- hull-floe ------------------------------------------------------
        if self.hull is not None and self.hull_speed != 0.0:
            x_stem = self.hull.x_centre(self.time)
            signed, normal = self.hull.signed_distance(s.pos, x_stem)
            pen_h = s.radius - signed
            keep = (pen_h > 0.0) & active
            if bool(keep.any()):
                idx = torch.nonzero(keep, as_tuple=True)[0]
                pen_h = pen_h[idx]
                nvec = normal[idx]
                tvec = torch.stack([-nvec[:, 1], nvec[:, 0]], dim=1)
                c_n = 2.0 * law.beta_hull_floe * torch.sqrt(
                    law.normal_stiffness * s.mass[idx])
                v_rel = s.vel[idx] - self.hull_velocity
                delta_dot = -(v_rel * nvec).sum(dim=1)
                f_n_mag = law.normal_stiffness * pen_h + c_n * delta_dot
                f_n_mag = torch.clamp(f_n_mag, min=0.0)
                v_t = ((v_rel * tvec).sum(dim=1) - s.omega[idx] * s.radius[idx])
                f_t_mag = law.hull_floe_friction * f_n_mag
                f_t = -f_t_mag * torch.sign(v_t)
                total = f_n_mag.unsqueeze(1) * nvec + f_t.unsqueeze(1) * tvec
                force.index_add_(0, idx, total)
                torque.index_add_(0, idx, -s.radius[idx] * f_t)
                spring = spring + 0.5 * law.normal_stiffness * (pen_h ** 2).sum()
                diss_normal = diss_normal + (c_n * delta_dot ** 2).sum()
                diss_friction = diss_friction + (f_t_mag * v_t.abs()).sum()
                max_pen = torch.maximum(max_pen,
                                        (pen_h / (2.0 * s.radius[idx])).max())
                hull_force = total.sum(dim=0)
                hull_force_per_floe[idx] = total
                w_hull = (total * self.hull_velocity.unsqueeze(0)).sum()
                n_hull_contacts = int(idx.numel())

        # --- water drag -----------------------------------------------------
        drag = self._fluid_drag()
        force = force + drag
        w_drag = (drag * s.vel).sum()

        self.last_hull_force = hull_force
        self.last_hull_contacts = n_hull_contacts
        return {"force": force, "torque": torque, "spring_energy": spring,
                "diss_normal_rate": diss_normal, "diss_friction_rate": diss_friction,
                "w_hull_rate": w_hull, "w_drag_rate": w_drag,
                "hull_force": hull_force,
                "drag_force": drag,
                "hull_force_per_floe": hull_force_per_floe,
                "n_hull_contacts": n_hull_contacts,
                "max_penetration_fraction": max_pen}

    # -- integration --------------------------------------------------------
    def accelerations(self, force: torch.Tensor, torque: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.state
        mask = s.active.to(DTYPE)
        acc = force / s.mass.unsqueeze(1) * mask.unsqueeze(1)
        alpha = torque / s.inertia * mask
        return acc, alpha

    def step(self, dt: float) -> Dict[str, torch.Tensor]:
        """One velocity-Verlet step with midpoint work rates.

        The work/energy rates of the interval ``[t, t+dt]`` are evaluated at the
        *midpoint* state (post-drift position, mid-step velocity, which is where
        velocity Verlet actually samples the velocity), and the hull/drag impulse
        uses the trapezoid of the two force evaluations -- exactly what the
        velocity kicks integrate.  The energy identity of the module docstring is
        therefore second order in ``dt`` rather than first order.
        """
        s = self.state
        out1 = self.forces()
        acc1, alpha1 = self.accelerations(out1["force"], out1["torque"])

        # kick-drift-kick
        s.vel = s.vel + 0.5 * dt * acc1
        s.omega = s.omega + 0.5 * dt * alpha1
        s.pos = s.pos + dt * s.vel
        s.angle = s.angle + dt * s.omega
        self.time += dt

        out2 = self.forces()
        acc2, alpha2 = self.accelerations(out2["force"], out2["torque"])
        s.vel = s.vel + 0.5 * dt * acc2
        s.omega = s.omega + 0.5 * dt * alpha2

        half = 0.5 * dt
        # the impulse uses the trapezoid of the two force evaluations -- this is
        # *exactly* what the velocity kicks integrate, so it audits the force
        # bookkeeping (a missing external term shows up here)
        self.hull_impulse = self.hull_impulse + half * (
            out1["hull_force"] + out2["hull_force"])
        self.drag_impulse = self.drag_impulse + half * (
            out1["drag_force"].sum(dim=0) + out2["drag_force"].sum(dim=0))
        # the work rates use the *midpoint* state (post-drift position, mid-step
        # velocity): a genuine midpoint quadrature, second order in dt
        self.ledger.w_hull += float(out2["w_hull_rate"]) * dt
        self.ledger.w_drag += float(out2["w_drag_rate"]) * dt
        self.ledger.e_diss_normal += float(out2["diss_normal_rate"]) * dt
        self.ledger.e_diss_friction += float(out2["diss_friction_rate"]) * dt
        self.ledger.hull_contacts_peak = max(
            self.ledger.hull_contacts_peak,
            int(out1["n_hull_contacts"]), int(out2["n_hull_contacts"]))
        self.ledger.max_penetration_fraction = max(
            self.ledger.max_penetration_fraction,
            float(out1["max_penetration_fraction"]),
            float(out2["max_penetration_fraction"]))
        self.last_hull_force = out2["hull_force"]
        self.last_hull_contacts = int(out2["n_hull_contacts"])

        self._retire_escaped()
        return out2

    def _retire_escaped(self) -> None:
        """Remove floes that left the domain, banking their kinetic energy."""
        if self.domain_half_width is None:
            return
        s = self.state
        outside_y = s.pos[:, 1].abs() > self.domain_half_width
        outside_x = (s.pos[:, 0] < -1.0e-9) if self.domain_length is None else (
            (s.pos[:, 0] < -1.0e-9) | (s.pos[:, 0] > self.domain_length))
        behind_stern = torch.zeros_like(outside_y)
        if (self.hull is not None and self.clearing_distance_m is not None):
            x_stern = self.hull.x_centre(self.time) - float(self.clearing_distance_m)
            behind_stern = s.pos[:, 0] < x_stern
        cleared_lateral = torch.zeros_like(outside_y)
        if self.corridor_half_width is not None:
            cleared_lateral = s.pos[:, 1].abs() > float(self.corridor_half_width)
        leaving = (outside_y | outside_x | behind_stern | cleared_lateral) & s.active
        if not bool(leaving.any()):
            return
        idx = torch.nonzero(leaving, as_tuple=True)[0]
        ke = (0.5 * s.mass[idx] * (s.vel[idx] ** 2).sum(dim=1)
              + 0.5 * s.inertia[idx] * s.omega[idx] ** 2)
        downstream = (s.pos[idx, 0] > float(self.domain_length)) if self.domain_length else (
            torch.zeros(idx.numel(), dtype=torch.bool))
        lateral = ~downstream
        self.ledger.e_out += float(ke.sum())
        self.ledger.e_out_downstream += float(ke[downstream].sum())
        self.ledger.e_out_clearing += float(ke[lateral].sum())
        wake = lateral & (s.pos[idx, 0] < (self.hull.x_centre(self.time)
                                          - float(self.clearing_distance_m))) \
            if (self.hull is not None and self.clearing_distance_m is not None) \
            else torch.zeros(idx.numel(), dtype=torch.bool)
        self.ledger.e_out_wake += float(ke[wake].sum())
        self.ledger.n_removed += int(idx.numel())
        s.active[idx] = False
        s.vel[idx] = 0.0
        s.omega[idx] = 0.0

    # -- diagnostics --------------------------------------------------------
    def spring_energy(self) -> float:
        return float(self.forces()["spring_energy"])

    def mechanical_energy(self) -> float:
        return self.state.kinetic_energy() + self.spring_energy()

    def total_hull_impulse(self) -> torch.Tensor:
        """Accumulated impulse delivered by the hull to the ice [N s]."""
        return self.hull_impulse.clone()

    def external_impulse(self) -> torch.Tensor:
        """Accumulated external impulse on the ice (hull + water drag) [N s]."""
        return self.hull_impulse + self.drag_impulse

    def as_dict(self) -> Dict[str, float]:
        return {"time_s": self.time, "n_floes": self.state.n,
                "ledger": self.ledger.as_dict(),
                "mechanical_energy_J": self.mechanical_energy()}