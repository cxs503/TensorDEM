"""Rolling resistance contact laws for DEM (ice blocks, floes, granular media).

Motivation
----------
``tensordem.contact_tangential`` covers normal restitution and Cundall-Strack
sliding friction but declares ``TangentialContactConfig.rolling_mu`` as an
unused placeholder: without a rolling-resistance torque, spherical/near-spherical
particles and ice blocks roll far too easily, so the macroscopic angle of repose
and the resistance of a broken-ice field in a channel are systematically
under-predicted.  This module supplies the missing contact law.

Implemented models (same three families assessed by Ai et al. 2011)
------------------------------------------------------------------
1. ``"cdt"`` — constant directional torque:

       M_r = - mu_r * |F_n| * r_eff * w_hat

   A saturated torque that always opposes the rolling direction.  It has no
   static branch: at |omega_roll| -> 0 the torque magnitude does not vanish, so a
   resting particle must be regularised (``omega_tol``) and the model cannot
   hold a static equilibrium on its own (creep/jitter).  Validated below.

2. ``"viscous"`` — velocity-proportional rolling resistance:

       M_r = - (mu_r * |F_n| * r_eff / omega_ref) * omega_roll

   Strictly dissipative (M_r . omega_roll <= 0 everywhere), used as the
   damping/energy-dissipation branch.

3. ``"epsd"`` — elastic-plastic spring (EPSD/EPSD2 family, Iwashita & Oda 1998,
   Jiang et al. 2005):

       M_r^{n+1} = M_r^{n} - k_r * omega_roll * dt
       |M_r^{n+1}| <= mu_r * |F_n| * r_eff     (plastic clamp)

   The rolling spring accumulates an elastic torque, so the contact can carry a
   static torque up to the rolling limit; beyond it the contact yields and the
   excess is dissipated.  This is the branch that reproduces stick behaviour.

Conventions
-----------
* ``omega_rel = omega_j - omega_i`` with ``i`` -> ``j`` along the unit normal
  ``n``.  The returned torque is the torque acting on particle ``j``; particle
  ``i`` receives ``-T`` (Newton's third law for the couple).
* Only the rolling component of the relative spin is resisted: the component
  along ``n`` (drilling/spin friction) is deliberately out of scope here.
* All quantities in SI: N, m, rad/s, N*m, s.

References
----------
Ai, J., Chen, J.-F., Rotter, J.M., Ooi, J.Y. (2011). Assessment of rolling
    resistance models in discrete element simulations. Powder Technology 206,
    269-282.  doi:10.1016/j.powtec.2010.11.003
Iwashita, K., Oda, M. (1998). Rolling resistance at contacts in simulation of
    shear band development by DEM. J. Eng. Mech. 124, 285-292.
Jiang, M.J., Yu, H.-S., Harris, D. (2005). A novel discrete model for granular
    material incorporating rolling resistance. Comput. Geotech. 32, 340-357.
Cundall, P.A., Strack, O.D.L. (1979). A discrete numerical model for granular
    assemblies. Geotechnique 29, 47-65.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

MODELS = ("cdt", "viscous", "epsd")


@dataclass(frozen=True)
class RollingResistanceConfig:
    """Rolling-resistance contact parameters.

    Attributes:
        model:    ``"cdt"``, ``"viscous"`` or ``"epsd"`` (see module docstring).
        mu_r:     Rolling-resistance coefficient (dimensionless).  ``0.0``
                  disables rolling resistance exactly.
        k_r:      Rolling spring stiffness, N*m/rad.  Required by ``"epsd"``;
                  must be > 0 there.  A modelling choice, not a material
                  constant: it sets the rolling-mode frequency
                  ``sqrt(k_r / I)``, which must be resolved by ``dt``.
        omega_ref: Regularisation scale for ``"viscous"`` (rad/s): at
                  ``|omega_roll| = omega_ref`` the torque equals the saturated
                  rolling limit ``mu_r |F_n| r_eff``.
        omega_tol: Angular-velocity tolerance for ``"cdt"`` (rad/s); below it
                  the torque is set to zero so a resting particle does not
                  receive a spurious saturated torque.
    """

    model: str = "epsd"
    mu_r: float = 0.0
    k_r: Optional[float] = None
    omega_ref: float = 1.0
    omega_tol: float = 1e-9

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if not math.isfinite(self.mu_r) or self.mu_r < 0.0:
            raise ValueError("mu_r must be finite and nonnegative")
        if self.model == "epsd":
            if self.k_r is None:
                raise ValueError("model='epsd' requires a positive rolling stiffness k_r")
            if not math.isfinite(self.k_r) or self.k_r <= 0.0:
                raise ValueError("k_r must be finite and positive")
        if not math.isfinite(self.omega_ref) or self.omega_ref <= 0.0:
            raise ValueError("omega_ref must be finite and positive")
        if not math.isfinite(self.omega_tol) or self.omega_tol < 0.0:
            raise ValueError("omega_tol must be finite and nonnegative")

    @property
    def enabled(self) -> bool:
        return self.mu_r > 0.0


def rolling_component(omega_rel: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Remove the spin (drilling) component of the relative angular velocity.

    Args:
        omega_rel: (N, 3) relative angular velocity ``omega_j - omega_i``.
        normal:    (N, 3) unit contact normal from particle ``i`` to ``j``.

    Returns:
        (N, 3) rolling (tangential) angular velocity.
    """
    if omega_rel.ndim != 2 or omega_rel.shape[1] != 3:
        raise ValueError("omega_rel must have shape (N, 3)")
    if normal.shape != omega_rel.shape:
        raise ValueError("normal must have the same shape as omega_rel")
    spin = (omega_rel * normal).sum(dim=1, keepdim=True)
    return omega_rel - spin * normal


def rolling_resistance_torque(
    omega_rel: torch.Tensor,
    normal_force_mag: torch.Tensor,
    r_eff: torch.Tensor,
    config: RollingResistanceConfig,
    *,
    normal: Optional[torch.Tensor] = None,
    torque_state: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the rolling-resistance torque acting on particle ``j``.

    Args:
        omega_rel:        (N, 3) relative angular velocity ``omega_j - omega_i``.
        normal_force_mag: (N,) magnitude of the (nonnegative) normal force.
        r_eff:            (N,) effective radius ``r_i r_j / (r_i + r_j)``; for a
                          particle on a plane use the particle radius.
        config:           :class:`RollingResistanceConfig`.
        normal:           (N, 3) unit normal, used to strip the spin component.
                          ``None`` treats the full angular velocity as rolling.
        torque_state:     (N, 3) EPSD elastic torque state from the previous step.
                          ``None`` starts from a stress-free contact.
        dt:               Time step, s.  Required by ``"epsd"``.

    Returns:
        ``(torque_j, new_torque_state)``: torque on particle ``j`` (N, 3) and the
        elastic state to carry to the next step (N, 3).  For ``"cdt"`` and
        ``"viscous"`` the returned state is always zero, since those models are
        memoryless.  The contact is deactivated and its state reset to zero
        whenever ``normal_force_mag == 0``.
    """
    if omega_rel.ndim != 2 or omega_rel.shape[1] != 3:
        raise ValueError("omega_rel must have shape (N, 3)")
    if normal_force_mag.ndim != 1 or r_eff.ndim != 1:
        raise ValueError("normal_force_mag and r_eff must be 1-D")
    if normal_force_mag.shape[0] != omega_rel.shape[0] or r_eff.shape[0] != omega_rel.shape[0]:
        raise ValueError("normal_force_mag, r_eff and omega_rel must agree in length")
    if (normal_force_mag < 0).any():
        raise ValueError("normal_force_mag must be nonnegative")
    if (r_eff < 0).any():
        raise ValueError("r_eff must be nonnegative")

    n_contacts = omega_rel.shape[0]
    dtype = omega_rel.dtype
    device = omega_rel.device
    zeros = torch.zeros_like(omega_rel)
    inactive = (normal_force_mag <= 0.0) | (r_eff <= 0.0)

    if not config.enabled:
        return zeros, torch.zeros_like(torque_state) if torque_state is not None else zeros

    omega = rolling_component(omega_rel, normal) if normal is not None else omega_rel
    limit = config.mu_r * normal_force_mag * r_eff  # (N,) saturated torque magnitude

    if config.model == "cdt":
        speed = torch.linalg.vector_norm(omega, dim=1)
        direction = omega / speed.clamp_min(torch.finfo(dtype).tiny)[:, None]
        magnitude = torch.where(speed > config.omega_tol, limit, torch.zeros_like(limit))
        torque = -magnitude[:, None] * direction
        return _deactivate(torque, inactive), zeros

    if config.model == "viscous":
        # gamma_r chosen so that |omega| = omega_ref reproduces the roll limit
        gamma = limit / config.omega_ref
        torque = -gamma[:, None] * omega
        return _deactivate(torque, inactive), zeros

    # Elastic-plastic rolling spring (EPSD).
    if dt is None or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("model='epsd' requires a positive finite dt")
    state = zeros if torque_state is None else torque_state
    if state.shape != omega_rel.shape:
        raise ValueError("torque_state must have shape (N, 3)")
    state = _deactivate(state, inactive)  # separated contacts lose their memory
    trial = state - float(config.k_r) * omega * dt
    trial_mag = torch.linalg.vector_norm(trial, dim=1)
    safe = trial_mag.clamp_min(torch.finfo(dtype).tiny)
    scale = torch.where(trial_mag > limit, limit / safe, torch.ones_like(limit))
    new_state = trial * scale[:, None]
    return new_state, new_state


def _deactivate(values: torch.Tensor, inactive: torch.Tensor) -> torch.Tensor:
    """Zero rows flagged as inactive."""
    if not bool(inactive.any()):
        return values
    mask = (~inactive).to(values.dtype)[:, None]
    return values * mask


# ===================================================================
# Analytical references used for verification
# ===================================================================

def critical_incline_angle_deg(mu_r: float) -> float:
    """Analytical critical incline angle for a pure-rolling particle.

    Torque balance for a particle at rest on an incline of angle ``theta``:
    the friction force ``f = m g sin(theta)`` produces a torque ``f R`` about
    the centre, which the rolling resistance can resist up to
    ``mu_r |F_n| R = mu_r m g cos(theta) R``.  Equilibrium therefore requires

        tan(theta) <= mu_r   ->   theta_c = arctan(mu_r).

    Independent of the moment of inertia, since nothing accelerates.
    """
    if not math.isfinite(mu_r) or mu_r < 0.0:
        raise ValueError("mu_r must be finite and nonnegative")
    return math.degrees(math.atan(mu_r))


def incline_rolling_acceleration(
    theta_deg: float,
    mu_r: float,
    *,
    gravity: float = 9.81,
    inertia_factor: float = 0.4,
) -> float:
    """Analytical downhill acceleration of a sphere rolling without slip.

    Force/moment balance with the no-slip constraint ``a = -R domega/dt`` and a
    saturated rolling torque ``mu_r m g cos(theta) R`` gives

        a = g (sin(theta) - mu_r cos(theta)) / (1 + I / (m R^2)),

    with ``I / (m R^2) = 2/5`` for a solid sphere (``inertia_factor=0.4``).
    The acceleration vanishes at ``theta = arctan(mu_r)``, consistent with
    :func:`critical_incline_angle_deg`.
    """
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    if not math.isfinite(mu_r) or mu_r < 0.0:
        raise ValueError("mu_r must be finite and nonnegative")
    if inertia_factor <= 0.0:
        raise ValueError("inertia_factor must be positive")
    theta = math.radians(theta_deg)
    return gravity * (math.sin(theta) - mu_r * math.cos(theta)) / (1.0 + inertia_factor)


def pure_rolling_feasible(
    theta_deg: float,
    mu_r: float,
    mu: float,
    *,
    inertia_factor: float = 0.4,
) -> bool:
    """Whether a sphere can roll without slip at this incline angle.

    Rolling without slip requires a tangential force
    ``f = m g (I/(mR^2) sin t + mu_r cos t) / (1 + I/(mR^2))``, so the
    friction cone ``f <= mu N = mu m g cos t`` is respected only if

        (I/(mR^2) tan t + mu_r) / (1 + I/(mR^2)) <= mu.

    Above that angle the sphere slips and :func:`incline_rolling_acceleration`
    no longer applies; use :func:`incline_sliding_acceleration` instead.
    """
    for name, value in (("theta_deg", theta_deg), ("mu_r", mu_r), ("mu", mu)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mu_r < 0.0 or mu < 0.0:
        raise ValueError("mu_r and mu must be nonnegative")
    if inertia_factor <= 0.0:
        raise ValueError("inertia_factor must be positive")
    theta = math.radians(theta_deg)
    required = (inertia_factor * math.tan(theta) + mu_r) / (1.0 + inertia_factor)
    return required <= mu + 1e-12


def incline_sliding_acceleration(
    theta_deg: float,
    mu: float,
    *,
    gravity: float = 9.81,
) -> float:
    """Analytical downhill acceleration when the contact slides steadily.

    With fully mobilised kinetic friction ``f = mu m g cos(theta)`` opposing
    the slip, the translation balance gives ``a = g (sin t - mu cos t)``.  This
    is the relevant reference above the pure-rolling limit of
    :func:`pure_rolling_feasible`, where the rotational equation only sets the
    spin-up rate and not the translational acceleration.
    """
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be finite and nonnegative")
    theta = math.radians(theta_deg)
    return gravity * (math.sin(theta) - mu * math.cos(theta))


# ===================================================================
# Reference rigid-body harness (single sphere on an inclined plane)
# ===================================================================

@dataclass(frozen=True)
class SphereInclineConfig:
    """Configuration of the single-sphere inclined-plane harness.

    The harness is a deliberately minimal rigid-body integration of one sphere
    with three contact contributions -- normal spring-dashpot, Cundall-Strack
    tangential friction and the rolling-resistance torque from this module --
    so the contact law can be checked against the closed-form references above
    without the bonded-particle solver.
    """

    radius: float = 0.01              # m
    density: float = 900.0            # kg/m^3 (ice)
    k_n: float = 2.0e5                # N/m
    gamma_n: float = 0.0              # N*s/m normal damping
    mu: float = 0.9                   # sliding friction coefficient at the contact
    k_t: Optional[float] = None       # tangential stiffness (default: k_n)
    dt: float = 2.0e-5                # s
    gravity: float = 9.81             # m/s^2

    def __post_init__(self) -> None:
        for name in ("radius", "density", "k_n", "mu", "dt", "gravity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gamma_n < 0.0:
            raise ValueError("gamma_n must be nonnegative")
        if self.k_t is not None and self.k_t <= 0.0:
            raise ValueError("k_t must be positive when given")

    @property
    def mass(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3 * self.density

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def inertia_factor(self) -> float:
        return self.inertia / (self.mass * self.radius ** 2)


def simulate_sphere_on_incline(
    angles_deg,
    rolling: RollingResistanceConfig,
    *,
    config: SphereInclineConfig = SphereInclineConfig(),
    duration: float = 0.2,
    settle_time: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Integrate N independent spheres resting on inclined planes.

    The ``N`` columns share the same material/contact parameters but use
    different incline angles, so a whole angle sweep costs one integration
    loop.

    Args:
        angles_deg:   Sequence of incline angles in degrees (one column each).
        rolling:      Rolling-resistance configuration applied to every column.
        config:       Harness parameters.
        duration:     Total simulated time, s.
        settle_time:  Time excluded from the acceleration fit, s (lets the
                      initial contact oscillation decay and the sphere reach
                      a steady rolling state).

    Returns:
        Dict of tensors with time on axis 0 and columns on axis 1:
        ``time``, ``x`` (down-slope displacement), ``v`` (down-slope velocity),
        ``omega`` (spin, positive clockwise, i.e. ``omega = -v/R`` when rolling
        without slip), ``normal_force``, ``torque``, ``slip_velocity``.
    """
    angles = torch.as_tensor(list(angles_deg), dtype=torch.float64)
    if angles.ndim != 1 or angles.numel() == 0:
        raise ValueError("angles_deg must be a nonempty 1-D sequence")
    n = angles.numel()
    dt, g, radius = config.dt, config.gravity, config.radius
    mass, inertia = config.mass, config.inertia
    k_t = config.k_n if config.k_t is None else config.k_t

    theta = torch.deg2rad(angles)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)

    # State: x (down-slope), y (height above the plane), spin omega (+z).
    x = torch.zeros(n, dtype=torch.float64)
    # Start exactly at the static normal equilibrium depth so the normal
    # contact does not have to bootstrap itself (avoids a spurious bounce).
    y = radius - mass * g * cos_t / config.k_n  # centre height above plane
    v = torch.zeros(n, dtype=torch.float64)
    vy = torch.zeros(n, dtype=torch.float64)
    omega = torch.zeros(n, dtype=torch.float64)
    xi = torch.zeros(n, dtype=torch.float64)         # Cundall-Strack tangential memory
    rolling_state = torch.zeros(n, 3, dtype=torch.float64)
    normal_vec = torch.zeros(n, 3, dtype=torch.float64)
    normal_vec[:, 1] = 1.0
    mu_r = rolling.mu_r

    steps = int(round(duration / dt))
    history = {key: [] for key in ("x", "v", "omega", "normal_force", "torque", "slip_velocity")}

    for _ in range(steps):
        penetration = radius - y
        active = penetration > 0.0
        # Normal contact: spring-dashpot, clamped at zero (no adhesion).
        f_n = torch.where(active, config.k_n * penetration - config.gamma_n * vy,
                          torch.zeros_like(penetration)).clamp_min(0.0)
        # Contact-point slip velocity along the slope: v_contact = v + R*omega.
        v_slip = v + radius * omega
        xi = torch.where(active, xi + v_slip * dt, torch.zeros_like(xi))
        # Cundall-Strack: the spring force OPPOSES the accumulated slip.
        f_limit = float(config.mu) * f_n
        f_t = (-k_t * xi).clamp(min=-f_limit, max=f_limit)
        # Rolling resistance torque (about +z), applied to the sphere.
        omega_vec = torch.zeros(n, 3, dtype=torch.float64)
        omega_vec[:, 2] = omega
        torque_vec, rolling_state = rolling_resistance_torque(
            omega_vec,
            f_n,
            torch.full((n,), radius, dtype=torch.float64),
            rolling,
            normal=normal_vec,
            torque_state=rolling_state,
            dt=dt,
        )
        torque_z = torque_vec[:, 2]

        a_x = g * sin_t + f_t / mass
        a_y = -g * cos_t + f_n / mass
        alpha = (radius * f_t + torque_z) / inertia

        # Symplectic (semi-implicit) Euler.
        v = v + a_x * dt
        vy = vy + a_y * dt
        omega = omega + alpha * dt
        x = x + v * dt
        y = y + vy * dt

        history["x"].append(x)
        history["v"].append(v)
        history["omega"].append(omega)
        history["normal_force"].append(f_n)
        history["torque"].append(torque_z)
        history["slip_velocity"].append(v_slip)

    out = {key: torch.stack(values) for key, values in history.items()}
    out["time"] = torch.arange(steps, dtype=torch.float64) * dt
    out["angles_deg"] = angles
    out["steady_mask"] = out["time"] >= settle_time
    return out


def fit_acceleration(result: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of ``v(t) = v0 + a t`` per column over the steady window.

    Returns:
        ``(acceleration, r_squared)`` each of shape (N,).
    """
    mask = result["steady_mask"]
    time = result["time"][mask]
    velocity = result["v"][mask]
    if time.numel() < 2:
        raise ValueError("steady window is too short to fit an acceleration")
    t_mean = time.mean()
    v_mean = velocity.mean(dim=0)
    dt = (time - t_mean)[:, None]
    slope = (dt * (velocity - v_mean)).sum(dim=0) / (dt ** 2).sum()
    residual = velocity - (v_mean + dt * slope)
    ss_res = (residual ** 2).sum(dim=0)
    ss_tot = ((velocity - v_mean) ** 2).sum(dim=0)
    r_squared = 1.0 - ss_res / ss_tot.clamp_min(torch.finfo(torch.float64).tiny)
    return slope, r_squared