"""Analytic contact-impact benchmark for the DEM3D normal contact model.

Why this module exists
----------------------
``contact_tangential.restitution_to_damping`` maps a *target* coefficient of
restitution ``e`` to a normal damping ``gamma_n`` using the textbook
**unclamped** linear spring-dashpot identity

    e = exp(-pi*zeta / sqrt(1 - zeta**2)),   zeta = gamma_n / (2*sqrt(k_n*m_eff))

That identity is exact only for the free mass-spring-dashpot oscillator, whose
normal force is allowed to become *attractive* (``F < 0``) late in the contact.

The DEM3D solver does not allow that: every normal force branch is truncated by
``clamp_min(0)``::

    F_n = max(0, k_n*overlap - c_n*v_n)        (dem3d.py, pair / tool / platen)

The truncation removes the retarding force over the last part of the separation
phase, so the *realised* restitution of the merged solver is systematically
**higher** than the target fed into the mapping.  For ``e_target = 0.50`` the
bias is about +10 %, i.e. outside the 3 % accuracy band used for engineering
predictions.  This module provides the closed-form reference for the clamped
model, a corrected damping mapping, and a harness that measures the solver
against both references so the bias can never silently regress.

Analytic basis
--------------
Relative coordinate ``delta = 2R - s`` (overlap, ``s`` = centre separation),
effective mass ``m_eff`` (``m/2`` for an equal-mass pair), closing speed ``v0``:

    m_eff*delta'' + c*delta' + k*delta = 0      (while F > 0)

with ``zeta = c/(2*sqrt(k*m_eff))``, ``omega_n = sqrt(k/m_eff)``,
``omega_d = omega_n*sqrt(1-zeta**2)`` and ``delta(0)=0, delta'(0)=v0``:

    delta(t)  = (v0/omega_d) * exp(-zeta*omega_n*t) * sin(omega_d*t)
    delta'(t) = v0*exp(-zeta*omega_n*t)
                * [cos(theta) - (zeta/sqrt(1-zeta**2))*sin(theta)],  theta = omega_d*t

Unclamped (``e_unclamped``): contact lasts one half period ``t_c = pi/omega_d``
and ``e = -delta'(t_c)/v0 = exp(-pi*zeta/sqrt(1-zeta**2))``.

Clamped (``e_clamped``): the force vanishes as soon as
``k*delta + c*delta' = 0``, i.e. ``omega_n*delta + 2*zeta*delta' = 0``.  Dividing
by ``v0*exp(-zeta*omega_n*t)``:

    (1 - 2*zeta**2)*sin(theta*) + 2*zeta*sqrt(1-zeta**2)*cos(theta*) = 0

    => theta* = atan2(2*zeta*sqrt(1-zeta**2), 2*zeta**2 - 1)   in (pi/2, pi] for zeta < 1/sqrt(2)

    => e_clamped = -delta'(t*)/v0,   t* = theta*/omega_d

At release the overlap rate is negative and its magnitude is
``v0*exp(-zeta*omega_n*t*)``. The dimensionless release time is
``2*acos(zeta)/sqrt(1-zeta**2)`` below critical damping, 2 at critical
damping, and ``2*acosh(zeta)/sqrt(zeta**2-1)`` above it. Free separation
after release takes another ``c/k`` seconds by the zero-force condition.

All quantities are SI: N, m, s, kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "ImpactAnalytic",
    "impact_analytic",
    "unclamped_restitution",
    "clamped_restitution",
    "rk4_clamped_restitution",
    "restitution_to_damping_clamped",
    "PairImpactResult",
    "pair_impact",
]


# ===================================================================
# Analytic references
# ===================================================================

def unclamped_restitution(zeta: float) -> float:
    """Restitution of the free (possibly adhesive) spring-dashpot oscillator.

    ``e = exp(-pi*zeta/sqrt(1-zeta**2))``.  This is the identity implemented by
    ``contact_tangential.restitution_to_damping``; it is *not* what a solver
    with ``max(0, F)`` contact realises.
    """
    if zeta < 0:
        raise ValueError("zeta must be nonnegative")
    if zeta == 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0  # critically damped or beyond: no rebound
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))


def _release_time(zeta: float) -> float:
    """Dimensionless time omega_n*t at the first unloading force zero."""
    if not math.isfinite(zeta) or zeta < 0:
        raise ValueError("zeta must be finite and nonnegative")
    if zeta < 1:
        return 2*math.acos(zeta)/math.sqrt((1-zeta)*(1+zeta))
    if zeta == 1:
        return 2.
    return 2*math.acosh(zeta)/math.sqrt(zeta-1)/math.sqrt(zeta+1)


def clamped_restitution(zeta: float) -> float:
    """Rebound speed ratio of a compressive-only linear contact at any damping."""
    return math.exp(-zeta*_release_time(zeta))


def rk4_clamped_restitution(zeta: float, *, n_steps: int = 200_000, zeta_limit: float = 0.999) -> float:
    """Numerically integrate the clamped 1-DOF impact, in normalised units.

    State ``(delta, delta')`` normalised as ``x = delta*omega_n/v0`` and
    ``u = delta'/v0`` so that ``x' = u`` and ``u' = -2*zeta*u - x`` (time in
    units of ``1/omega_n``).  The clamp removes the force when
    ``x + 2*zeta*u <= 0``; the contact ends when the gap closes again
    (``x <= 0`` after the first compression).  Reference value independent of
    the theta* derivation.
    """
    if zeta < 0 or zeta > zeta_limit:
        raise ValueError(f"zeta must be in [0, {zeta_limit}]")

    x, u = 0.0, 1.0
    # Contact lasts less than one natural period; resolution is far beyond need.
    dt = 2.0 * math.pi / n_steps

    def acc(x: float, u: float) -> float:
        f = x + 2.0 * zeta * u
        return -x - 2.0 * zeta * u if f > 0.0 else 0.0

    compressed = False
    for _ in range(n_steps * 2):
        k1x, k1u = u, acc(x, u)
        k2x, k2u = u + 0.5 * dt * k1u, acc(x + 0.5 * dt * k1x, u + 0.5 * dt * k1u)
        k3x, k3u = u + 0.5 * dt * k2u, acc(x + 0.5 * dt * k2x, u + 0.5 * dt * k2u)
        k4x, k4u = u + dt * k3u, acc(x + dt * k3x, u + dt * k3u)
        x_new = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        u_new = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        if x_new <= 0.0 and compressed:
            # Linear interpolation of the zero crossing for the rebound speed.
            frac = x / (x - x_new) if x != x_new else 0.0
            u_end = u + frac * (u_new - u)
            return max(0.0, -u_end)
        compressed = compressed or x_new > 0.0
        x, u = x_new, u_new
    raise RuntimeError("clamped impact did not separate within one natural period")


@dataclass(frozen=True)
class ImpactAnalytic:
    """Analytic impact metrics for one (k, c, m_eff) combination."""

    zeta: float
    omega_n: float
    omega_d: float
    e_unclamped: float
    e_clamped: float
    contact_time_unclamped: float
    contact_time_force: float
    contact_time_geometric: float
    clamp_active: bool
    theta_star: float | None
    release_overlap_m: float = 0.
    dashpot_work_j: float = 0.
    release_elastic_energy_j: float = 0.
    peak_overlap_m: float = 0.
    time_to_peak_overlap_s: float = 0.
    peak_force_n: float = 0.
    time_to_peak_force_s: float = 0.
    compression_impulse_ns: float = 0.
    rebound_impulse_ns: float = 0.

    @property
    def e_bias(self) -> float:
        """Relative bias of the unclamped mapping: e_clamped/e_unclamped - 1."""
        return self.e_clamped / self.e_unclamped - 1.0 if self.e_unclamped else math.inf


def impact_analytic(k_n: float, c_n: float, m_eff: float, v0: float = 1.0) -> ImpactAnalytic:
    """Closed-form impact metrics of the clamped and unclamped models.

    ``v0`` is relative closing speed. It sets the length and energy scales.
    Dashpot work integrates c*delta_dot² during active contact. The spring
    energy remaining at release completes the loss of relative kinetic energy.
    """
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")
    if c_n < 0:
        raise ValueError("c_n must be nonnegative")
    if not math.isfinite(v0) or v0 <= 0:
        raise ValueError("v0 must be finite and positive")

    omega_n = math.sqrt(k_n / m_eff)
    zeta = c_n / (2.0 * math.sqrt(k_n * m_eff))
    tau = _release_time(zeta)
    omega_d = omega_n*math.sqrt((1-zeta)*(1+zeta)) if zeta < 1 else 0.
    e_unc = unclamped_restitution(zeta)
    t_unc = math.pi/omega_d if omega_d else math.inf
    e_clamped = math.exp(-zeta*tau)
    t_force = tau/omega_n
    t_geometric = t_force+c_n/k_n
    theta_star = t_force*omega_d if omega_d else None
    release_overlap = (c_n/k_n)*(v0*e_clamped)
    release_elastic_energy = .5*k_n*release_overlap**2
    kinetic_loss = .5*m_eff*v0**2*(-math.expm1(-2*(zeta*tau)))
    # Initial force slope is k*v0*(1-4*zeta²). A later maximum exists
    # only below zeta=1/2. atan2 retains accuracy near the transition.
    peak_force_time = 0.
    peak_force = c_n*v0
    if zeta < .5:
        root = math.sqrt(1-zeta*zeta)
        phase = math.atan2(root*(1-2*zeta)*(1+2*zeta), zeta*(3-4*zeta*zeta))
        peak_force_time = phase/omega_d
        peak_force = m_eff*omega_n*v0*math.exp(-zeta*phase/root)
    return ImpactAnalytic(
        zeta=zeta,
        omega_n=omega_n,
        omega_d=omega_d,
        e_unclamped=e_unc,
        e_clamped=e_clamped,
        contact_time_unclamped=t_unc,
        contact_time_force=t_force,
        contact_time_geometric=t_geometric,
        clamp_active=True,
        theta_star=theta_star,
        release_overlap_m=release_overlap,
        dashpot_work_j=kinetic_loss-release_elastic_energy,
        release_elastic_energy_j=release_elastic_energy,
        peak_overlap_m=(v0/omega_n)*math.exp(-.5*(zeta*tau)),
        time_to_peak_overlap_s=.5*t_force,
        peak_force_n=peak_force,
        time_to_peak_force_s=peak_force_time,
        compression_impulse_ns=m_eff*v0,
        rebound_impulse_ns=m_eff*v0*e_clamped,
    )


def restitution_to_damping_clamped(
    e_target: float,
    k_n: float,
    m_eff: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Damping coefficient that realises ``e_target`` for the **clamped** model.

    Uses bisection on ``zeta`` (``e_clamped(zeta)`` is strictly decreasing in
    ``zeta``), then ``c = 2*zeta*sqrt(k_n*m_eff)``.  Use this instead of
    ``contact_tangential.restitution_to_damping`` when the produced damping will
    be consumed by a solver whose normal force is truncated at zero.
    """
    if not (0.0 < e_target <= 1.0):
        raise ValueError(f"e_target must be in (0, 1], got {e_target}")
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")
    if e_target == 1.0:
        return 0.0

    lo, hi = 0.0, 1.0
    while clamped_restitution(hi) > e_target:
        lo, hi = hi, 2*hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if clamped_restitution(mid) > e_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    zeta = 0.5 * (lo + hi)
    return 2.0 * zeta * math.sqrt(k_n * m_eff)


# ===================================================================
# Solver harness: equal-mass head-on pair collision
# ===================================================================

@dataclass(frozen=True)
class PairImpactResult:
    """Solver measurements for one head-on pair impact."""

    e_measured: float
    impact_speed: float
    rebound_speed: float
    contact_steps: int
    contact_time: float
    force_steps: int
    contact_time_force: float
    max_overlap: float
    momentum_error: float
    dt: float
    steps_per_contact: float
    force_law_residual: float
    e_unclamped: float
    e_clamped: float
    contact_time_analytic: float
    contact_time_force_analytic: float
    # -- the same measurements in the form the thermal chain consumes --------
    pair_mass_kg: float = 0.0
    peak_force: float = 0.0
    dissipated_energy: float = 0.0
    thermal_state: object = None      # ice_thermal_stress.dem_coupling.ThermalState
    time_to_peak_overlap_s: float = 0.  # from first geometric touch
    time_to_peak_force_s: float = 0.
    compression_impulse_ns: float = 0.
    rebound_impulse_ns: float = 0.
    final_center_of_mass_speed: float = 0.
    final_transverse_relative_speed: float = 0.
    final_center_of_mass_velocity: tuple[float, float, float] = (0., 0., 0.)
    center_of_mass_position_error_m: float = 0.
    active_dashpot_work_j: float = 0.
    release_elastic_energy_j: float = 0.
    compression_dashpot_work_j: float = 0.
    rebound_dashpot_work_j: float = 0.
    compression_contact_work_j: float = 0.
    rebound_contact_work_j: float = 0.
    # Interpolated event times measured from first geometric contact.
    time_to_force_release_s: float = 0.
    time_to_separation_s: float = 0.

    @property
    def e_error_vs_clamped(self) -> float:
        return self.e_measured / self.e_clamped - 1.0

    @property
    def e_error_vs_unclamped(self) -> float:
        return self.e_measured / self.e_unclamped - 1.0 if self.e_unclamped else math.inf

    @property
    def contact_time_error(self) -> float:
        """Error of the geometric (overlap) contact duration."""
        return self.contact_time / self.contact_time_analytic - 1.0

    @property
    def contact_time_force_error(self) -> float:
        """Error of the force-contact duration, i.e. the time the normal force acts."""
        return self.contact_time_force / self.contact_time_force_analytic - 1.0


@dataclass(frozen=True)
class _ImpactLedger:
    """Minimal record the thermal bridge needs from a solver impact."""

    impact_speed: float
    rebound_speed: float
    pair_mass_kg: float
    peak_force: float
    contact_time_force: float
    dissipated_energy: float
    peak_overlap: float
    dt: float


def _build_pair(k_n: float, c_n: float, radius: float, density: float, dt_scale: float,
                axis=(0., 0., 1.)) -> tuple[IceDEM3D, int, int]:
    """Two free particles facing each other; every other lattice site parked and fixed."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=c_n,
        # The harness installs an *absolute* dashpot ``c_n`` [N s/m] derived from
        # the restitution mapping under review, i.e. it integrates
        # F_n = max(0, k*delta - c*v_n).  The engine default
        # (``contact_damping_model="restitution"``) ignores ``contact_damping`` on
        # the particle-particle branch and rebuilds gamma_n from
        # ``restitution_coefficient``, which silently makes the pair elastic
        # (measured e ~ 1.0 for every target).  The verification harness must
        # therefore be explicit about the convention it validates (m2626).
        contact_damping_model="absolute",
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for i in range(n):
        pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    # 1 um initial clearance (each centre one radius from the mid-plane) keeps the
    # free-flight phase negligible in step count while starting outside contact.
    offset = (radius + 1.0e-6)*pos.new_tensor(axis)
    pos[0], pos[1] = -offset, offset
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    # Break every lattice bond before the impact so the pair is free-flying.
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break; harness assumption violated")
    dem.dt = dem.dt * dt_scale
    return dem, 0, 1


def pair_impact(
    *,
    e_target: float = 0.5,
    v0: float = 0.5,
    center_of_mass_speed: float = 0.,
    impact_axis: tuple[float, float, float] = (0., 0., 1.),
    center_of_mass_velocity: tuple[float, float, float] | None = None,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt_scale: float = 1.0,
    damping: str = "unclamped_mapping",
    thermal: bool = False,
    bulk_temperature_c: float = -10.0,
    thermal_cells: int = 120,
    thermal_dt_s: float = 2.0e-6,
) -> PairImpactResult:
    """Run an equal-mass head-on impact and measure the realised restitution.

    Parameters
    ----------
    e_target : target coefficient of restitution used to build the damping.
    damping  : ``"unclamped_mapping"`` uses
               ``contact_tangential.restitution_to_damping`` (the mapping under
               review); ``"clamped_mapping"`` uses
               :func:`restitution_to_damping_clamped` (the corrected mapping).
    impact_axis : nonzero direction from the first sphere to the second in 3-D.
    center_of_mass_speed : signed common velocity along the collision axis (m/s).
               Impact and rebound speeds are per-sphere speeds in the pair frame.
    center_of_mass_velocity : optional three-component lab-frame common velocity
               in m/s, used with center_of_mass_speed left at zero.
    dt_scale : multiplier on the solver's ``recommended_dt``.
    thermal  : if true, feed the work the solver actually dissipated into
               :mod:`tensordem.ice_thermal_stress` and attach the resulting
               ``ThermalState`` to the result (``thermal_state``).  This is the
               production coupling path: the thermal package never re-derives
               the impact energetics, it consumes the solver's ledger.
    bulk_temperature_c : ice temperature used by the thermal chain.

    Returns
    -------
    PairImpactResult with the measured restitution, contact duration, momentum
    drift and the residual of the documented force law.
    """
    from .contact_tangential import restitution_to_damping

    if not math.isfinite(center_of_mass_speed):
        raise ValueError("center_of_mass_speed must be finite")
    if not math.isfinite(v0) or v0 <= 0:
        raise ValueError("v0 must be finite and positive")

    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    if damping == "unclamped_mapping":
        c_n = restitution_to_damping(e_target, k_n, m_eff)
    elif damping == "clamped_mapping":
        c_n = restitution_to_damping_clamped(e_target, k_n, m_eff)
    else:
        raise ValueError("damping must be 'unclamped_mapping' or 'clamped_mapping'")

    if len(impact_axis) != 3 or not all(math.isfinite(v) for v in impact_axis):
        raise ValueError("impact_axis must contain three finite components")
    scale = max(abs(v) for v in impact_axis)
    if scale == 0:
        raise ValueError("impact_axis must be nonzero")
    scaled = tuple(v/scale for v in impact_axis)
    length = math.hypot(*scaled)
    unit_axis = tuple(v/length for v in scaled)
    if center_of_mass_velocity is not None:
        if center_of_mass_speed != 0.:
            raise ValueError("specify center_of_mass_velocity or center_of_mass_speed")
        if len(center_of_mass_velocity) != 3 or not all(math.isfinite(v) for v in center_of_mass_velocity):
            raise ValueError("center_of_mass_velocity must contain three finite components")
    dem, i, j = _build_pair(k_n, c_n, radius, density, dt_scale, unit_axis)
    axis = dem.positions.new_tensor(unit_axis)
    dem.velocities = torch.zeros_like(dem.positions)
    drift = (center_of_mass_speed*axis if center_of_mass_velocity is None else
             dem.velocities.new_tensor(center_of_mass_velocity))
    dem.velocities[i] = drift + v0*axis
    dem.velocities[j] = drift - v0*axis
    initial_center = .5*(dem.positions[i]+dem.positions[j])
    dem.time = 0.0
    dem.step_count = 0

    # Force-law conformance at a prescribed overlap, compared with max(0, k*d - c*vn).
    probe = dem.positions.clone()
    probe[i] = -(radius - 1.0e-4)*axis
    probe[j] = (radius - 1.0e-4)*axis
    saved = dem.positions.clone()
    dem.positions = probe
    f_probe, _ = dem.forces(update_fracture=False)
    overlap = 2.0 * radius - torch.linalg.vector_norm(probe[j]-probe[i])
    vn = float(torch.dot(dem.velocities[j]-dem.velocities[i], axis))
    expected = max(0.0, k_n * float(overlap) - c_n * vn)
    force_law_residual = float(torch.linalg.vector_norm(f_probe[i]+expected*axis)) / max(expected, 1e-30)
    dem.positions = saved

    momentum_before = float(torch.dot(dem.velocities.sum(dim=0), axis))
    # Relative kinetic energy excludes the uniform translation of the pair.
    relative_speed = float(torch.dot(dem.velocities[i]-dem.velocities[j], axis))
    initial_relative_speed = relative_speed
    contact_steps = 0
    force_steps = 0
    clearance = float(torch.linalg.vector_norm(dem.positions[j]-dem.positions[i]))-2*radius
    touch_time = clearance/(2*v0)
    max_overlap = 0.0
    time_to_peak = 0.0
    peak_force = 0.0
    peak_force_time = 0.0
    # Euler uses the force from the start of each step. The initial gap
    # gives zero load; subsequent values come from the sampled pair state.
    normal_force = 0.
    compression_work = 0.
    rebound_work = 0.
    release_elastic_energy = 0.
    time_to_force_release = 0.
    overlap_now = -clearance
    compression_contact_work = 0.
    rebound_contact_work = 0.
    compression_impulse = 0.
    rebound_impulse = 0.
    for _ in range(400_000):
        v_prev = relative_speed
        dem.step()
        relative_speed = float(torch.dot(dem.velocities[i]-dem.velocities[j], axis))
        v_next = relative_speed
        impulse = normal_force*dem.dt
        if v_prev <= 0:
            compression_fraction = 0.
        elif v_next >= 0:
            compression_fraction = 1.
        else:
            # Maximum compression is where the relative velocity changes sign.
            compression_fraction = v_prev/(v_prev-v_next)
        loading_impulse = compression_fraction*impulse
        unloading_impulse = (1-compression_fraction)*impulse
        compression_impulse += loading_impulse
        rebound_impulse += unloading_impulse
        # Work of the applied step impulse uses the mean relative speed on
        # each side of the zero crossing. Both reported energies are positive.
        compression_contact_work += .5*loading_impulse*(max(v_prev, 0.)+max(v_next, 0.))
        rebound_contact_work += .5*unloading_impulse*(max(-v_prev, 0.)+max(-v_next, 0.))
        # In free flight the normal force is exactly zero, so the velocity is
        # bit-for-bit unchanged; any change means the contact force was active.
        if v_next != v_prev:
            force_steps += 1
        sep = float(torch.linalg.vector_norm(dem.positions[j]-dem.positions[i]))
        previous_overlap = overlap_now
        overlap_now = 2.0 * radius - sep
        if normal_force > 0.:
            trial_force = k_n*overlap_now + c_n*v_next
            active_fraction = 1.
            if trial_force <= 0.:
                # Locate release before the geometric overlap necessarily ends.
                active_fraction = normal_force/(normal_force-trial_force)
                time_to_force_release = dem.time-(1-active_fraction)*dem.dt-touch_time
                release_overlap = max(0., previous_overlap + active_fraction*(overlap_now-previous_overlap))
                release_elastic_energy = .5*k_n*release_overlap**2
            work_per_step = c_n*v_prev**2*dem.dt
            active_compression = min(compression_fraction, active_fraction)
            compression_work += work_per_step*active_compression
            rebound_work += work_per_step*(active_fraction-active_compression)
        if overlap_now > 0.0:
            contact_steps += 1
            if overlap_now > max_overlap:
                max_overlap = overlap_now
                time_to_peak = dem.time-touch_time
            # the documented force law, evaluated on the solver's own state
            normal_force = max(0., k_n * overlap_now + c_n * relative_speed)
            if normal_force > peak_force:
                peak_force = normal_force
                peak_force_time = dem.time-touch_time
        elif contact_steps:
            fraction = previous_overlap/(previous_overlap-overlap_now)
            time_to_separation = dem.time-(1-fraction)*dem.dt-touch_time
            break
    else:  # pragma: no cover - guarded by the physics
        raise RuntimeError("pair did not separate within the step budget")

    rebound = -.5*relative_speed
    analytic = impact_analytic(k_n, c_n, m_eff, 2*v0)
    dissipated = max(.5*m_eff*(initial_relative_speed-relative_speed)
                     *(initial_relative_speed+relative_speed), 0.)
    thermal_state = None
    if thermal:
        from .ice_thermal_stress.dem_coupling import couple_impact_result
        from .ice_thermal_stress.properties import ice_properties

        props = ice_properties(bulk_temperature_c)
        # two equal ice spheres: R* = R/2 and E* = E / (2 (1 - nu^2))
        thermal_state = couple_impact_result(
            _ImpactLedger(impact_speed=v0, rebound_speed=rebound, pair_mass_kg=m,
                          peak_force=peak_force, contact_time_force=force_steps * dem.dt,
                          dissipated_energy=dissipated, peak_overlap=max_overlap,
                          dt=dem.dt),
            bulk_temperature_c=bulk_temperature_c,
            effective_radius_m=0.5 * radius,
            effective_modulus_pa=props.youngs_modulus_pa / (2.0 * (1.0 - props.poisson ** 2)),
            peak_overlap_m=max_overlap,
            cells=thermal_cells,
            dt_s=thermal_dt_s,
        )
    final_center = .5*(dem.positions[i]+dem.positions[j])
    final_drift = .5*(dem.velocities[i]+dem.velocities[j])
    return PairImpactResult(
        time_to_force_release_s=time_to_force_release,
        time_to_separation_s=time_to_separation,
        active_dashpot_work_j=compression_work+rebound_work,
        compression_contact_work_j=compression_contact_work,
        rebound_contact_work_j=rebound_contact_work,
        compression_dashpot_work_j=compression_work,
        rebound_dashpot_work_j=rebound_work,
        release_elastic_energy_j=release_elastic_energy,
        e_measured=rebound / v0,
        impact_speed=v0,
        rebound_speed=rebound,
        contact_steps=contact_steps,
        contact_time=contact_steps * dem.dt,
        force_steps=force_steps,
        contact_time_force=force_steps * dem.dt,
        max_overlap=max_overlap,
        momentum_error=float(torch.dot(dem.velocities.sum(dim=0), axis)) - momentum_before,
        dt=dem.dt,
        steps_per_contact=contact_steps,
        force_law_residual=force_law_residual,
        e_unclamped=analytic.e_unclamped,
        e_clamped=analytic.e_clamped,
        contact_time_analytic=analytic.contact_time_geometric,
        contact_time_force_analytic=analytic.contact_time_force,
        pair_mass_kg=m,
        peak_force=peak_force,
        dissipated_energy=dissipated,
        thermal_state=thermal_state,
        time_to_peak_overlap_s=time_to_peak,
        time_to_peak_force_s=peak_force_time,
        compression_impulse_ns=compression_impulse,
        rebound_impulse_ns=rebound_impulse,
        final_transverse_relative_speed=float(torch.linalg.vector_norm(
            dem.velocities[i]-dem.velocities[j]-relative_speed*axis)),
        final_center_of_mass_speed=float(torch.dot(final_drift, axis)),
        final_center_of_mass_velocity=tuple(final_drift.tolist()),
        center_of_mass_position_error_m=float(torch.linalg.vector_norm(
            final_center-initial_center-dem.time*drift)),
    )