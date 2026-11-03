"""Kinematic contact-patch verification of the DEM3D friction / rolling model.

The Coulomb branch of ``IceDEM3D`` is a Cundall-Strack incremental spring with
``|F_t| <= mu * F_n``; the rolling branch is a resisting couple reduced to a
tangential force because the solver carries no rotational degree of freedom.
Both are stated in closed form, so they can be checked against analytic values
in a *kinematically driven* two-sphere patch: the pair geometry is restored
every step and the relative velocity is prescribed, which makes the normal
force an exact constant and isolates the tangential branch.

Reference relations used here (``analytic`` block of every scenario):

* normal force            ``F_n = k_n * overlap``
* friction limit          ``F_t,max = mu * F_n``
* sticking limit slip     ``a_stick = F_t,max / k_t``  (``k_t = k_n``)
* slip needed to reverse a saturated friction force
                          ``s_rev = a_stick`` up to one integrator quantum,
                          independent of how long the contact has been sliding
* sliding dissipation rate ``P = -mu * F_n * |v_rel|``
* rolling dissipation rate ``P_rr = -mu_r * F_n * |v_rel|``

Every scenario returns measured values next to their reference so that the
downstream report can quote a relative error without hand transcription.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "PatchConfig",
    "ContactPatch",
    "run_campaign",
]


@dataclass(frozen=True)
class PatchConfig:
    """Two-sphere contact patch with analytic normal force."""

    radius: float = 0.025
    contact_stiffness: float = 5000.0
    overlap_fraction: float = 0.1          # overlap = fraction * 2R
    dt: float = 1e-4
    density: float = 917.0

    @property
    def overlap(self) -> float:
        return self.overlap_fraction * 2.0 * self.radius

    @property
    def normal_force(self) -> float:
        return self.contact_stiffness * self.overlap


@dataclass
class ContactPatch:
    """A two-sphere contact whose geometry and velocity are driven."""

    cfg: PatchConfig
    mu: float = 0.0
    mu_r: float = 0.0
    sim: IceDEM3D = field(init=False)

    def __post_init__(self) -> None:
        c = self.cfg
        config = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=c.radius, density=c.density,
            contact_stiffness=c.contact_stiffness, contact_damping=0.0,
            drag=0.0, friction_coefficient=self.mu,
            rolling_resistance_coefficient=self.mu_r,
            restitution_coefficient=1.0, dt=c.dt,
            fix_x_edges=False, fix_bottom=False, tool_speed=1e-9,
            # Keep bond springs out of the picture and preserve the bond
            # topology while the patch is reconfigured.
            breaking_strain=1e3, shear_breaking_strain=1e3,
        )
        self.sim = IceDEM3D(config)
        self.sim.alive[:] = False          # isolate contact physics
        self.place(normal_axis=0)

    # -- geometry -----------------------------------------------------
    def place(self, *, normal_axis: int, separation: float | None = None) -> None:
        """Restore the pair with its normal along ``normal_axis``."""
        dtype = self.sim.dtype
        sep = (2.0 * self.cfg.radius - self.cfg.overlap
               if separation is None else separation)
        p = torch.zeros(3, dtype=dtype)
        p[normal_axis] = sep
        self.sim.positions[0] = -p / 2.0
        self.sim.positions[1] = +p / 2.0
        for k in range(2, len(self.sim.positions)):
            self.sim.positions[k] = torch.tensor([0.0, 0.0, 1.0 + k], dtype=dtype)
        self.sim.initial_positions.copy_(self.sim.positions)

    def set_velocity(self, *, tangent_axis: int, speed: float) -> None:
        dtype = self.sim.dtype
        v = torch.zeros(3, dtype=dtype)
        v[tangent_axis] = speed
        self.sim.velocities[0] = -0.5 * v
        self.sim.velocities[1] = +0.5 * v

    # -- probes -------------------------------------------------------
    def tangential_force(self) -> float:
        """Tangential (y) force component on particle 1 (N)."""
        force, _ = self.sim.forces()
        return float(force[1].tolist()[1])

    def stored_slip(self) -> float:
        """Magnitude of the persisted tangential displacement (m)."""
        if not self.sim._tangential_state:
            return 0.0
        vectors = torch.stack([v[0] for v in self.sim._tangential_state.values()])
        return float(torch.linalg.vector_norm(vectors, dim=1).max())

    def n_states(self) -> int:
        return len(self.sim._tangential_state)

    def drive(self, *, speed: float, steps: int, tangent_axis: int = 1,
              normal_axis: int = 0) -> list[dict[str, float]]:
        """Prescribe relative tangential velocity for ``steps`` force evaluations."""
        records: list[dict[str, float]] = []
        for _ in range(steps):
            self.set_velocity(tangent_axis=tangent_axis, speed=speed)
            force, _ = self.sim.forces()
            comp = force[1].tolist()
            records.append({
                "f_tangent": comp[tangent_axis],
                "f_normal": comp[normal_axis],
                "stored_slip": self.stored_slip(),
                "n_states": float(self.n_states()),
            })
            self.place(normal_axis=normal_axis)
        return records


def _stick_limit(cfg: PatchConfig, mu: float) -> float:
    return mu * cfg.normal_force / cfg.contact_stiffness


def _rel_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if measured == 0.0 else math.inf
    return (measured - reference) / reference


def scenario_stick(cfg: PatchConfig, mu: float = 0.3, speed: float = 0.01,
                   steps: int = 20) -> dict:
    """Elastic regime: F_t = k_t * v * t while the slip stays below a_stick."""
    patch = ContactPatch(cfg, mu=mu)
    records = patch.drive(speed=speed, steps=steps)
    last = records[-1]
    slip = speed * cfg.dt * steps
    # Force on particle 1 opposes its +y motion: F_t = -k_t * slip.
    reference = -cfg.contact_stiffness * slip
    return {
        "slip_m": slip,
        "a_stick_m": _stick_limit(cfg, mu),
        "stick_ratio": slip / _stick_limit(cfg, mu),
        "f_tangent_measured_N": last["f_tangent"],
        "f_tangent_reference_N": reference,
        "rel_error": _rel_error(last["f_tangent"], reference),
        "stored_slip_measured_m": last["stored_slip"],
        "stored_slip_reference_m": slip,
        "stored_slip_rel_error": _rel_error(last["stored_slip"], slip),
    }


def scenario_sliding(cfg: PatchConfig, mus=(0.05, 0.1, 0.2, 0.3),
                     speed: float = 1.0, steps: int = 100) -> dict:
    """Saturated regime: |F_t| = mu * F_n and the persisted slip is clamped."""
    rows = []
    for mu in mus:
        patch = ContactPatch(cfg, mu=mu)
        records = patch.drive(speed=speed, steps=steps)
        last = records[-1]
        a_stick = _stick_limit(cfg, mu)
        rows.append({
            "mu": mu,
            "f_limit_N": mu * cfg.normal_force,
            "f_tangent_measured_N": last["f_tangent"],
            "f_tangent_rel_error": _rel_error(-last["f_tangent"], mu * cfg.normal_force),
            "stored_slip_measured_m": last["stored_slip"],
            "a_stick_m": a_stick,
            "stored_slip_ratio": last["stored_slip"] / a_stick,
            "unclamped_slip_m": speed * cfg.dt * steps,
        })
    return {"driven_slip_m": speed * cfg.dt * steps, "rows": rows}


def scenario_reversal(cfg: PatchConfig, mu: float = 0.1,
                      slip_durations=(0.05, 0.1, 0.2),
                      speed: float = 1.0) -> dict:
    """Slip needed to reverse a saturated friction force is ``a_stick``.

    A saturated contact stores ``delta = a_stick``; the force changes sign once
    the slip passes through zero, so one ``a_stick`` of reversed slip is needed
    (plus at most one integrator quantum ``|v|*dt`` for the discrete crossing).
    The requirement must be set by the friction limit, not by how long the
    contact has been sliding: an unclamped persistent state makes it grow
    without bound in proportion to the sliding history.
    """
    rows = []
    a_stick = _stick_limit(cfg, mu)
    for duration in slip_durations:
        steps = int(round(duration / cfg.dt))
        patch = ContactPatch(cfg, mu=mu)
        patch.drive(speed=speed, steps=steps)
        slip_before = patch.stored_slip()
        slip = 0.0
        flip = None
        for k in range(1, steps + 1):
            patch.set_velocity(tangent_axis=1, speed=-speed)
            force, _ = patch.sim.forces()
            patch.place(normal_axis=0)
            slip += speed * cfg.dt
            if force[1].tolist()[1] > 0.0:      # force now opposes the -y motion
                flip = k
                break
        rows.append({
            "sliding_duration_s": duration,
            "sliding_force_evaluations": steps,
            "slip_before_reversal_m": slip_before,
            "reversal_flip_step": flip,
            "reversal_slip_m": slip if flip is not None else float("nan"),
            "reversal_slip_reference_m": a_stick,
            "reversal_slip_rel_error": (_rel_error(slip, a_stick)
                                        if flip is not None else float("nan")),
            "reversal_slip_excess_quanta": (slip - a_stick) / (speed * cfg.dt)
                                           if flip is not None else float("nan"),
            "slip_quantum_m": speed * cfg.dt,
        })
    reversals = [r["reversal_slip_m"] for r in rows]
    spread = (max(reversals) - min(reversals)) if all(
        math.isfinite(r) for r in reversals) else float("nan")
    return {"a_stick_m": a_stick, "rows": rows,
            "reversal_slip_spread_m": spread,
            "history_independent": bool(math.isfinite(spread)
                                        and spread <= speed * cfg.dt)}


def scenario_normal_rotation(cfg: PatchConfig, mu: float = 0.2,
                             speed: float = 1.0, steps: int = 100) -> dict:
    """After the contact normal rotates, friction must stay tangential."""
    patch = ContactPatch(cfg, mu=mu)
    patch.drive(speed=speed, steps=steps, tangent_axis=1, normal_axis=0)
    slip_before = patch.stored_slip()
    patch.place(normal_axis=1)                 # normal now along y
    patch.sim.velocities[:] = 0.0
    force, _ = patch.sim.forces()
    comp = force[1].tolist()
    f_limit = mu * cfg.normal_force
    spurious = comp[1] - cfg.normal_force      # normal force alone should remain
    return {
        "stored_slip_before_m": slip_before,
        "f_normal_measured_N": comp[1],
        "f_normal_reference_N": cfg.normal_force,
        "spurious_normal_component_N": spurious,
        "friction_limit_N": f_limit,
        "spurious_fraction_of_limit": abs(spurious) / f_limit,
        "rel_error": _rel_error(comp[1], cfg.normal_force),
    }


def scenario_separation(cfg: PatchConfig, mu: float = 0.2, speed: float = 1.0,
                        steps: int = 100, separated_steps: int = 50) -> dict:
    """Contact memory must die with the contact."""
    patch = ContactPatch(cfg, mu=mu)
    patch.drive(speed=speed, steps=steps)
    slip_before = patch.stored_slip()
    far = 3.0 * cfg.radius
    for _ in range(separated_steps):
        patch.place(normal_axis=0, separation=far)
        patch.sim.velocities[:] = 0.0
        patch.sim.forces()
    states_after = patch.n_states()
    patch.place(normal_axis=0)
    patch.sim.velocities[:] = 0.0
    force, _ = patch.sim.forces()
    return {
        "stored_slip_before_m": slip_before,
        "states_retained_after_separation": states_after,
        "recontact_tangential_force_N": force[1].tolist()[1],
        "recontact_reference_N": 0.0,
    }


def scenario_rolling(cfg: PatchConfig, mu_r: float = 0.05, mu: float = 0.0,
                     speed: float = 1.0) -> dict:
    """Rolling resistance must oppose motion and dissipate energy."""
    patch = ContactPatch(cfg, mu=mu, mu_r=mu_r)
    patch.set_velocity(tangent_axis=1, speed=speed)
    force, _ = patch.sim.forces()
    comp = force[1].tolist()
    f_ref = -mu_r * cfg.normal_force
    power = comp[1] * speed
    control = ContactPatch(cfg, mu=mu, mu_r=0.0)
    control.set_velocity(tangent_axis=1, speed=speed)
    f_control, _ = control.sim.forces()
    return {
        "mu_r": mu_r,
        "f_tangential_measured_N": comp[1],
        "f_tangential_reference_N": f_ref,
        "rel_error": _rel_error(comp[1], f_ref),
        "f_lateral_measured_N": comp[2],
        "f_lateral_reference_N": 0.0,
        "dissipation_rate_W": power,
        "dissipation_rate_reference_W": f_ref * speed,
        "control_mu_r0_f_tangential_N": f_control[1].tolist()[1],
    }


def scenario_sliding_work(cfg: PatchConfig, mu: float = 0.2, speed: float = 1.0,
                          steps: int = 100) -> dict:
    """Sliding work splits into the elastic ramp plus Coulomb dissipation.

    The contact first loads elastically over ``a_stick`` of slip (storing
    ``0.5 * F_limit * a_stick``), then slides at the limit force. Both parts are
    known in closed form, so total work is ``F_limit * (s - 0.5 * a_stick)`` and
    the dissipated part is ``F_limit * (s - a_stick)``.
    """
    patch = ContactPatch(cfg, mu=mu)
    records = patch.drive(speed=speed, steps=steps)
    work = sum(-r["f_tangent"] * speed * cfg.dt for r in records)
    slip = speed * cfg.dt * steps
    f_limit = mu * cfg.normal_force
    a_stick = _stick_limit(cfg, mu)
    stored = 0.5 * f_limit * a_stick
    work_reference = f_limit * (slip - 0.5 * a_stick)
    dissipation_reference = f_limit * (slip - a_stick)
    return {
        "mu": mu,
        "slip_m": slip,
        "a_stick_m": a_stick,
        "work_measured_J": work,
        "work_reference_J": work_reference,
        "rel_error": _rel_error(work, work_reference),
        "elastic_stored_J": stored,
        "dissipation_measured_J": work - stored,
        "dissipation_reference_J": dissipation_reference,
        "dissipation_rel_error": _rel_error(work - stored, dissipation_reference),
    }


def scenario_work_convergence(mu: float = 0.2, speed: float = 1.0,
                              dt_list=(1e-4, 5e-5, 2.5e-5),
                              target_slip: float = 0.01) -> dict:
    """Sliding-work error versus time step at fixed total slip.

    The residual is the discrete ramp-up before saturation, so it must shrink
    as ``dt`` is refined; a non-converging residual would indicate a modelling
    error rather than a discretisation one.
    """
    rows = []
    for dt in dt_list:
        cfg = PatchConfig(dt=dt)
        steps = max(1, int(round(target_slip / (speed * dt))))
        patch = ContactPatch(cfg, mu=mu)
        records = patch.drive(speed=speed, steps=steps)
        work = sum(-r["f_tangent"] * speed * dt for r in records)
        slip = speed * dt * steps
        f_limit = mu * cfg.normal_force
        a_stick = _stick_limit(cfg, mu)
        reference = f_limit * (slip - 0.5 * a_stick)
        rows.append({
            "dt_s": dt,
            "steps": steps,
            "slip_m": slip,
            "work_measured_J": work,
            "work_reference_J": reference,
            "rel_error": _rel_error(work, reference),
        })
    first = abs(rows[0]["rel_error"])
    last = abs(rows[-1]["rel_error"])
    return {"target_slip_m": target_slip, "rows": rows,
            "error_ratio_first_to_last": (first / last) if last > 0 else None,
            "monotone_decreasing": all(
                abs(rows[k + 1]["rel_error"]) <= abs(rows[k]["rel_error"])
                for k in range(len(rows) - 1))}


def run_campaign(cfg: PatchConfig | None = None) -> dict:
    """Run every scenario and return the full campaign record."""
    cfg = cfg or PatchConfig()
    return {
        "config": {
            "radius_m": cfg.radius,
            "contact_stiffness_N_per_m": cfg.contact_stiffness,
            "overlap_m": cfg.overlap,
            "normal_force_N": cfg.normal_force,
            "dt_s": cfg.dt,
            "density_kg_per_m3": cfg.density,
        },
        "stick": scenario_stick(cfg),
        "sliding": scenario_sliding(cfg),
        "reversal": scenario_reversal(cfg),
        "normal_rotation": scenario_normal_rotation(cfg),
        "separation": scenario_separation(cfg),
        "rolling": scenario_rolling(cfg),
        "sliding_work": scenario_sliding_work(cfg),
        "work_convergence": scenario_work_convergence(),
    }