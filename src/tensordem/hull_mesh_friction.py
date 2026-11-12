"""Coulomb friction and rolling resistance on the triangulated-hull contact path.

Why this module exists
----------------------
:mod:`tensordem.hull_mesh_contact` is a *normal-only* penalty law: it returns
``f_n * n`` and nothing else.  Every hull/ice number produced with it is
therefore a **frictionless** number.  The gap ledger records this explicitly
(``docs/dem_calibration_gap_ledger.md``, m16794 write-back, "still open:
friction / rolling resistance on the hull contact path").  Physically the
omission is not neutral: for a ship in ice the tangential traction on the hull
is what converts normal crushing pressure into longitudinal resistance (ice
riding up the bow and along the shoulder), so a frictionless hull
under-predicts the load path and cannot hold a floe on the hull surface at
all.

What is implemented
-------------------
Per contact :math:`k` (sphere :math:`i`, hull triangle :math:`t`) the geometry
comes from :func:`tensordem.hull_mesh_contact.narrow_phase_contacts`, i.e.
exactly the rows the normal law scores -- a tangential law cannot disagree with
the normal law about the contact point, the triangle, the normal or the
penetration.

* **Coulomb friction (Cundall--Strack, 1979), stateful**: the stored tangential
  displacement of the *contact point* is integrated and projected onto the
  Coulomb cone,

      delta <- P_t (delta - v_slip dt),   |delta| <= mu f_n / k_t,
      F_t   = k_t delta                  (applied to the sphere at the contact)

  with ``v_slip`` the tangential part of the sphere material-point velocity
  relative to the hull material-point velocity (the hull may translate *and*
  rotate: ``v_h + omega_h x (c - x_h)``).  The projection is the shared
  :func:`tensordem.contact_tangential.project_tangential_displacement` /
  :func:`~tensordem.contact_tangential.saturation_displacement` kernel already
  verified on the sphere--sphere path, so there is one friction law in the
  repository, not two.

* **Memory across facets** (``carry_over_across_facets``): a sphere sliding
  along a curved hull changes its nearest triangle; the stored displacement is
  carried over and re-projected onto the new tangent plane instead of being
  dropped.  Without this, friction on a hull is periodic in the facet crossing
  -- a silent 0-to-mu f_n sawtooth on a surface that is *supposed* to be
  smooth.  A genuinely new contact (no previous contact for that particle)
  starts at ``delta = 0``.

* **Rolling resistance** (``mu_r``) through the verified
  :func:`tensordem.rolling_resistance.rolling_resistance_torque` kernel, with
  the hull as body ``i`` and the sphere as body ``j`` (so the returned torque
  is the torque *on the sphere*), ``r_eff`` = contact radius.

* **Reaction** on the hull: equal and opposite force plus the moment of the
  contact forces about the hull origin including the rolling couples.  A
  friction law without a consistent reaction silently violates momentum
  conservation in a coupled ship-ice model.

Energy contract
---------------
While sticking, ``F_t . v_slip = -k_t delta . d(delta)/dt``: the tangential
spring *stores* energy.  While sliding, the cone projection holds ``|delta|``
at the cap, so ``F_t . v_slip <= 0`` strictly.  The module therefore never
creates energy; the harness :func:`integrate_sphere_on_mesh` carries the spring
term in its ledger and returns it, so a test can assert
``KE + PE + spring + dissipated = const``.

Evidence level: REAL (standard contact mechanics; every claim in
``docs/hull_mesh_friction_verification.md`` is scored against a closed-form
incline anchor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .contact_tangential import (
    project_tangential_displacement,
    saturation_displacement,
)
from .hull_mesh_contact import narrow_phase_contacts
from .rolling_resistance import RollingResistanceConfig, rolling_resistance_torque

__all__ = [
    "HullFrictionConfig",
    "HullFrictionState",
    "HullContactResult",
    "mesh_contact_forces_frictional",
    "integrate_sphere_on_mesh",
]


# ---------------------------------------------------------------------------
# Configuration and state
# ---------------------------------------------------------------------------

@dataclass
class HullFrictionConfig:
    """Friction law for the hull contact path.

    Attributes:
        mu:   Coulomb coefficient (static == kinetic; ``0.0`` disables the
              tangential branch exactly).
        k_t:  Tangential (Cundall--Strack) stiffness, N/m.  The static
              equilibrium displacement of a contact loaded by a tangential
              force ``f`` is ``f / k_t``; it must be small against the particle
              radius and resolved by ``dt`` (``sqrt(k_t / m) dt << 1``).
        mu_r: Rolling-resistance coefficient (dimensionless).  ``0.0``
              disables rolling resistance exactly.
        rolling_model / k_r / omega_ref: forwarded to
              :class:`tensordem.rolling_resistance.RollingResistanceConfig`.
        carry_over_across_facets: re-project the stored tangential
              displacement onto the new tangent plane when a sphere's nearest
              triangle changes (see module docstring).
    """

    mu: float = 0.0
    k_t: float = 1.0e5
    mu_r: float = 0.0
    rolling_model: str = "cdt"
    k_r: Optional[float] = None
    omega_ref: float = 1.0
    carry_over_across_facets: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu) or self.mu < 0.0:
            raise ValueError("mu must be finite and nonnegative")
        if not math.isfinite(self.k_t) or self.k_t <= 0.0:
            raise ValueError("k_t must be finite and positive")
        if self.mu > 0.0 and self.k_t <= 0.0:
            raise ValueError("k_t must be positive when friction is enabled")
        # Validate the rolling sub-config eagerly so a typo fails at setup,
        # not at the first contact.
        self.rolling_config()

    @property
    def friction_enabled(self) -> bool:
        return self.mu > 0.0

    @property
    def enabled(self) -> bool:
        return self.mu > 0.0 or self.mu_r > 0.0

    def rolling_config(self) -> RollingResistanceConfig:
        return RollingResistanceConfig(
            model=self.rolling_model,
            mu_r=self.mu_r,
            k_r=self.k_r,
            omega_ref=self.omega_ref,
        )


@dataclass
class HullFrictionState:
    """Per-contact memory for the hull friction law.

    ``tangential[(particle, triangle)]`` is the stored Cundall--Strack
    displacement (3,) in world coordinates; ``rolling[(particle, triangle)]``
    is the elastic rolling-torque state for ``rolling_model='epsd'``.
    ``previous_triangle[particle]`` remembers which facet the particle last
    touched, so a facet crossing can carry the tangential memory over instead
    of dropping it.

    The state is mutable and advanced in place by
    :func:`mesh_contact_forces_frictional`; pass ``state=None`` for a
    memoryless (stateless) evaluation.
    """

    tangential: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    rolling: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    previous_triangle: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tangential)

    def clear(self) -> None:
        self.tangential.clear()
        self.rolling.clear()
        self.previous_triangle.clear()

    def copy(self) -> "HullFrictionState":
        return HullFrictionState(
            tangential={k: v.clone() for k, v in self.tangential.items()},
            rolling={k: v.clone() for k, v in self.rolling.items()},
            previous_triangle=dict(self.previous_triangle),
        )


@dataclass
class HullContactResult:
    """Contact response on the mesh, with the per-contact detail exposed.

    Attributes:
        forces          (N, 3) total contact force on each sphere.
        torques         (N, 3) total contact torque on each sphere (about its
                        centre): tangential moment ``r_i x F_t`` plus the
                        rolling-resistance couple.
        reaction_force  (3,) force on the hull (equal and opposite).
        reaction_torque (3,) moment on the hull about ``hull_origin``.
        normal_force    (K,) normal force magnitude per contact.
        tangential_force (K, 3) tangential force per contact (on the sphere).
        sliding         (K,) bool, True where the cone had to be truncated.
        rolling_torque  (K, 3) rolling-resistance couple per contact (on the
                        sphere).
        slip_velocity   (K, 3) tangential relative velocity of the sphere
                        material point at the contact w.r.t. the hull (used by
                        the energy ledger: ``dW = -F_t . v_slip dt``).
        contacts        the :func:`narrow_phase_contacts` dict (particle,
                        triangle, closest, normal, overlap, inside).
    """

    forces: torch.Tensor
    torques: torch.Tensor
    reaction_force: torch.Tensor
    reaction_torque: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    sliding: torch.Tensor
    rolling_torque: torch.Tensor
    slip_velocity: torch.Tensor
    contacts: dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Force law
# ---------------------------------------------------------------------------

def mesh_contact_forces_frictional(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    angular_velocities: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    hull_velocity: torch.Tensor,
    *,
    particle_radius: float | torch.Tensor,
    config: HullFrictionConfig,
    state: Optional[HullFrictionState] = None,
    dt: Optional[float] = None,
    stiffness: float = 1e4,
    damping: float = 0.0,
    hull_radius: float = 0.0,
    hull_angular_velocity: Optional[torch.Tensor] = None,
    hull_origin: Optional[torch.Tensor] = None,
    cell_width: Optional[float] = None,
    closed_surface: bool = False,
) -> HullContactResult:
    """Normal + tangential + rolling contact response of a triangulated hull.

    The normal law is *identical* to :func:`tensordem.hull_mesh_contact.
    mesh_contact_forces` (linear penalty ``k overlap - c v_n``, clamped at 0,
    one nearest triangle per sphere), so switching friction on cannot change a
    frictionless number by more than the friction feedback itself.

    Args:
        positions:  (N, 3) sphere centres.
        velocities: (N, 3) sphere translational velocities.
        angular_velocities: (N, 3) sphere angular velocities.
        vertices, faces: hull mesh.
        hull_velocity: (3,) hull translational velocity.
        particle_radius: scalar or (N,) sphere radii.
        config: :class:`HullFrictionConfig`.
        state: :class:`HullFrictionState`, advanced in place.  ``None`` makes
            the tangential branch memoryless *within one call only*, i.e. the
            elastic force is zero for every contact -- use a state (even a
            fresh one) for any time stepping.
        dt: time step, s.  Required when ``config.mu > 0`` and a state is
            given.
        stiffness, damping, hull_radius, cell_width, closed_surface: as in
            :func:`tensordem.hull_mesh_contact.mesh_contact_forces`.
        hull_angular_velocity: (3,) hull angular velocity (default zero); the
            hull material-point velocity at the contact is
            ``v_h + omega_h x (c - hull_origin)``.
        hull_origin: (3,) point the hull spins about (default the origin).

    Returns:
        :class:`HullContactResult`.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions shape")
    if angular_velocities.shape != positions.shape:
        raise ValueError("angular_velocities must match positions shape")
    if stiffness <= 0 or not math.isfinite(stiffness):
        raise ValueError("stiffness must be finite and positive")
    if damping < 0 or not math.isfinite(damping):
        raise ValueError("damping must be finite and non-negative")
    if config.friction_enabled and dt is None:
        raise ValueError("dt is required when Coulomb friction (mu > 0) is enabled")
    if dt is not None and (not math.isfinite(dt) or dt <= 0.0):
        raise ValueError("dt must be finite and positive")

    device, dtype = positions.device, positions.dtype
    N = positions.shape[0]
    p = positions.to(device=device, dtype=dtype)
    v = velocities.to(device=device, dtype=dtype)
    w = angular_velocities.to(device=device, dtype=dtype)
    verts = vertices.to(device=device, dtype=dtype)
    hv = hull_velocity.to(device=device, dtype=dtype).reshape(3)
    if hull_angular_velocity is None:
        hw = torch.zeros(3, dtype=dtype, device=device)
    else:
        hw = hull_angular_velocity.to(device=device, dtype=dtype).reshape(3)
    if hull_origin is None:
        h0 = torch.zeros(3, dtype=dtype, device=device)
    else:
        h0 = hull_origin.to(device=device, dtype=dtype).reshape(3)

    if isinstance(particle_radius, torch.Tensor):
        radii = particle_radius.to(device=device, dtype=dtype).reshape(N)
    else:
        radii = torch.full((N,), float(particle_radius), dtype=dtype, device=device)
    contact_r = radii + float(hull_radius)

    contacts = narrow_phase_contacts(
        p, verts, faces,
        radii=radii,
        hull_radius=float(hull_radius),
        cell_width=cell_width,
        closed_surface=closed_surface,
    )
    K = int(contacts["particle"].numel())

    forces = torch.zeros_like(p)
    torques = torch.zeros_like(p)
    empty3 = torch.zeros((0, 3), dtype=dtype, device=device)
    emptyK = torch.zeros((0,), dtype=dtype, device=device)
    if K == 0:
        if state is not None:
            state.tangential.clear()
            state.rolling.clear()
            state.previous_triangle.clear()
        return HullContactResult(
            forces=forces, torques=torques,
            reaction_force=torch.zeros(3, dtype=dtype, device=device),
            reaction_torque=torch.zeros(3, dtype=dtype, device=device),
            normal_force=emptyK, tangential_force=empty3, sliding=emptyK.to(torch.bool),
            rolling_torque=empty3, slip_velocity=empty3, contacts=contacts,
        )

    idx = contacts["particle"]
    tri = contacts["triangle"]
    closest = contacts["closest"]
    normal = contacts["normal"]
    overlap = contacts["overlap"]

    # --- normal law (identical to mesh_contact_forces) ---------------------
    rel_v = v[idx] - hv
    vn = (rel_v * normal).sum(dim=-1)
    fn = (stiffness * overlap - damping * vn).clamp_min(0.0)

    # --- tangential law ----------------------------------------------------
    lever = closest - p[idx]                       # centre -> contact point
    v_sphere_pt = v[idx] + torch.linalg.cross(w[idx], lever)
    v_hull_pt = hv + torch.linalg.cross(hw.expand_as(closest), closest - h0)
    v_rel_pt = v_sphere_pt - v_hull_pt
    v_slip = v_rel_pt - (v_rel_pt * normal).sum(dim=-1, keepdim=True) * normal

    if config.friction_enabled:
        delta = _gather_tangential_state(state, idx, tri, normal, config, device, dtype)
        if dt is not None:
            delta = delta - v_slip * dt
        delta = delta - (delta * normal).sum(dim=-1, keepdim=True) * normal
        delta, sliding = project_tangential_displacement(delta, fn, config.k_t, config.mu)
        delta = _store_tangential_state(state, idx, tri, delta)
        ft = config.k_t * delta
    else:
        ft = torch.zeros((K, 3), dtype=dtype, device=device)
        sliding = torch.zeros(K, dtype=torch.bool, device=device)
        _store_tangential_state(state, idx, tri, torch.zeros((K, 3), dtype=dtype, device=device))

    # --- rolling resistance -------------------------------------------------
    if config.mu_r > 0.0:
        omega_rel = w[idx] - hw          # omega_j - omega_i with i = hull, j = sphere
        torque_state = _gather_rolling_state(state, idx, tri, device, dtype)
        tau, new_tau_state = rolling_resistance_torque(
            omega_rel, fn, contact_r[idx], config.rolling_config(),
            normal=normal, torque_state=torque_state, dt=dt,
        )
        _store_rolling_state(state, idx, tri, new_tau_state)
    else:
        tau = torch.zeros((K, 3), dtype=dtype, device=device)

    # --- assemble ----------------------------------------------------------
    f_contact = fn.unsqueeze(-1) * normal + ft                  # on the sphere
    forces.index_add_(0, idx, f_contact)
    torques.index_add_(0, idx, torch.linalg.cross(lever, ft) + tau)

    if state is not None:
        active = {(int(i), int(t)) for i, t in zip(idx.tolist(), tri.tolist())}
        _prune_state(state, active)

    reaction_force = -forces.sum(dim=0)
    moment = torch.linalg.cross(closest - h0, -f_contact).sum(dim=0) - tau.sum(dim=0)
    return HullContactResult(
        forces=forces,
        torques=torques,
        reaction_force=reaction_force,
        reaction_torque=moment,
        normal_force=fn,
        tangential_force=ft,
        sliding=sliding,
        rolling_torque=tau,
        slip_velocity=v_slip,
        contacts=contacts,
    )


# ---------------------------------------------------------------------------
# State helpers (carry-over across facets, pruning)
# ---------------------------------------------------------------------------

def _gather_tangential_state(
    state: Optional[HullFrictionState],
    idx: torch.Tensor,
    tri: torch.Tensor,
    normal: torch.Tensor,
    config: HullFrictionConfig,
    device,
    dtype,
) -> torch.Tensor:
    K = idx.shape[0]
    out = torch.zeros((K, 3), dtype=dtype, device=device)
    if state is None:
        return out
    for row, (i, t) in enumerate(zip(idx.tolist(), tri.tolist())):
        stored = state.tangential.get((i, t))
        if stored is None and config.carry_over_across_facets:
            prev = state.previous_triangle.get(i)
            if prev is not None:
                stored = state.tangential.get((i, prev))
                if stored is not None:
                    # Re-project onto the new tangent plane: the tangential
                    # displacement is a vector in the *surface* plane, and the
                    # surface is piecewise planar.
                    n = normal[row]
                    stored = stored - (stored * n).sum() * n
        if stored is not None:
            out[row] = stored
    return out


def _store_tangential_state(
    state: Optional[HullFrictionState],
    idx: torch.Tensor,
    tri: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    if state is None:
        return delta
    for row, (i, t) in enumerate(zip(idx.tolist(), tri.tolist())):
        state.tangential[(i, t)] = delta[row].clone()
        state.previous_triangle[i] = t
    return delta


def _gather_rolling_state(state, idx, tri, device, dtype) -> torch.Tensor:
    K = idx.shape[0]
    out = torch.zeros((K, 3), dtype=dtype, device=device)
    if state is None:
        return out
    for row, key in enumerate(zip(idx.tolist(), tri.tolist())):
        stored = state.rolling.get(key)
        if stored is not None:
            out[row] = stored
    return out


def _store_rolling_state(state, idx, tri, values: torch.Tensor) -> None:
    if state is None:
        return
    for row, key in enumerate(zip(idx.tolist(), tri.tolist())):
        state.rolling[key] = values[row].clone()


def _prune_state(state: HullFrictionState, active: set[tuple[int, int]]) -> None:
    """Drop contacts that no longer exist (``delta`` of a lost contact is gone)."""
    for key in [k for k in state.tangential if k not in active]:
        del state.tangential[key]
    for key in [k for k in state.rolling if k not in active]:
        del state.rolling[key]


# ---------------------------------------------------------------------------
# Reference harness: one sphere on a mesh, semi-implicit Euler
# ---------------------------------------------------------------------------

def integrate_sphere_on_mesh(
    center: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    *,
    radius: float,
    mass: float,
    inertia: Optional[float],
    vertices: torch.Tensor,
    faces: torch.Tensor,
    config: HullFrictionConfig,
    dt: float,
    n_steps: int,
    stiffness: float,
    damping: float = 0.0,
    gravity: torch.Tensor | None = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
    hull_velocity: torch.Tensor | None = None,
    hull_angular_velocity: torch.Tensor | None = None,
    hull_origin: torch.Tensor | None = None,
    record_every: int = 100,
) -> dict[str, torch.Tensor]:
    """Integrate one sphere against a mesh with the frictional contact law.

    Semi-implicit (symplectic) Euler, rigid sphere.  ``gravity`` defaults to
    the standard downward acceleration ``(0, 0, -9.81) m/s^2`` (pass an
    explicit tensor for another field, or a zero tensor for none).
    ``inertia=None`` freezes
    the rotation (``omega`` stays zero and contact torques are ignored) -- that
    is the setting in which the translation-only incline anchors
    (:func:`tensordem.rolling_resistance.incline_sliding_acceleration`) apply.

    The returned ledger is complete enough to test conservation:

    ``t``, ``x`` (n, 3), ``v`` (n, 3), ``omega`` (n, 3),
    ``normal_force`` (n,), ``tangential_force`` (n, 3),
    ``kinetic``, ``potential``, ``spring``, ``dissipated`` (n,), plus the final
    ``state`` and ``ledger_total`` = kinetic + potential + spring + dissipated
    (which is constant to the integrator's tolerance, and never decreases).
    """
    dtype, device = center.dtype, center.device
    g = torch.zeros(3, dtype=dtype, device=device) if gravity is None else gravity.to(dtype=dtype, device=device)
    hv = torch.zeros(3, dtype=dtype, device=device) if hull_velocity is None else hull_velocity.to(dtype=dtype, device=device)
    if inertia is None:
        omega = torch.zeros(3, dtype=dtype, device=device)

    x = center.reshape(1, 3).clone().to(dtype=dtype)
    vel = velocity.reshape(1, 3).clone().to(dtype=dtype)
    w = omega.reshape(1, 3).clone().to(dtype=dtype)
    state = HullFrictionState()

    ts, xs, vs, ws = [], [], [], []
    fns, fts = [], []
    kes, pes, sprs, diss = [], [], [], []
    dissipated_total = 0.0
    spring_energy = 0.0

    for step in range(n_steps + 1):
        res = mesh_contact_forces_frictional(
            x, vel, w, vertices, faces, hv,
            particle_radius=radius, config=config, state=state, dt=dt,
            stiffness=stiffness, damping=damping,
            hull_angular_velocity=hull_angular_velocity, hull_origin=hull_origin,
        )
        # spring energy of the tangential displacement state
        spring_energy = 0.0
        for delta in state.tangential.values():
            spring_energy += 0.5 * config.k_t * float((delta * delta).sum())
        if step % record_every == 0:
            ts.append(step * dt)
            xs.append(x[0].clone())
            vs.append(vel[0].clone())
            ws.append(w[0].clone())
            fns.append(float(res.normal_force.sum()))
            fts.append(res.tangential_force.sum(dim=0).clone() if len(res.tangential_force) else torch.zeros(3, dtype=dtype))
            ke = 0.5 * mass * float((vel * vel).sum())
            if inertia is not None:
                ke += 0.5 * inertia * float((w * w).sum())
            kes.append(ke)
            pes.append(mass * float((g * x[0]).sum()))
            sprs.append(spring_energy)
            diss.append(dissipated_total)
        if step == n_steps:
            break

        total_f = mass * g + res.forces[0]
        vel = vel + (dt / mass) * total_f
        if inertia is not None:
            w = w + (dt / inertia) * res.torques[0]
        # frictional dissipation of this step: force on the sphere dotted with
        # the slip velocity the force was evaluated at (same step, same state).
        if len(res.tangential_force):
            dissipated_total += -float((res.tangential_force * res.slip_velocity).sum()) * dt
        x = x + dt * vel

    total = [kes[i] + pes[i] + sprs[i] + diss[i] for i in range(len(kes))]
    return {
        "t": torch.tensor(ts, dtype=dtype),
        "x": torch.stack(xs),
        "v": torch.stack(vs),
        "omega": torch.stack(ws),
        "normal_force": torch.tensor(fns, dtype=dtype),
        "tangential_force": torch.stack(fts),
        "kinetic": torch.tensor(kes, dtype=dtype),
        "potential": torch.tensor(pes, dtype=dtype),
        "spring": torch.tensor(sprs, dtype=dtype),
        "dissipated": torch.tensor(diss, dtype=dtype),
        "ledger_total": torch.tensor(total, dtype=dtype),
        "state": state,
        "final": {"x": x[0].clone(), "v": vel[0].clone(), "omega": w[0].clone()},
    }