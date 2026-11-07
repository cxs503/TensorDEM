"""Transient propeller-ice milling: blade-ice contact, spindle dynamics, debris.

Physical picture
----------------
A propeller turns inside an ice-blocked sector of the disc (the
:class:`~tensordem.propeller_ice.geometry.IceBlockGeometry` sector).  The ice body
is fed into the propeller plane at a constant rate, i.e. the ice face advances by
``s = feed_per_rev`` per propeller revolution, so each *blade pass* over a given
azimuth removes a layer of ``a_pen = s / Z``.  Per radial strip ``i`` and blade
``j`` the instantaneous engaged cut area is

    A_eng(i, j) = c(r_i) * f_eng(i, j) * a_pen

where ``f_eng in [0, 1]`` is the fraction of the strip chord actually inside the
ice sector.  ``f_eng`` ramps from 0 to 1 as the blade's leading edge sweeps the
chord into the ice and back to 0 as the trailing edge leaves, i.e. the entry and
exit transients of the cut are resolved in time.  Because the strip's azimuth is
offset by the blade **skew** ``theta_skew(r_i)``, the entry of the inner strips
leads the entry of the outer ones: skew smears the sector-entry load step over
the skew angle, which is exactly the mechanism ice-class propellers use to
suppress milling torque pulsation.

Forces come from :mod:`tensordem.propeller_ice.contact`:

* normal crushing ``F_n = p(a_pen) A_eng`` with the pressure plateau at
  ``sigma_c``;
* tangential Coulomb friction ``F_t = mu(v_slip) F_n`` with the fitted ice-steel
  friction correlation at the local sliding speed ``omega r``;
* the section is pitched by ``phi(r)``, so the blade-frame loads decompose into
  the shaft torque ``Q = r (F_n cos phi + F_t sin phi)`` and the axial thrust
  ``T = F_n sin phi - F_t cos phi``.

Spindle dynamics: ``I domega/dt = Q_drive - Q_ice - Q_hydro`` (``Q_hydro`` from
the BEM open-water model at the instantaneous revolutions, :mod:`.hydro`).  With
``governed=True`` the drive holds the set speed through an ideal servo and only
the load torque is reported; with ``governed=False`` the drive is a finite-gain
controller and the actual speed sag under the ice load is simulated.

Brittle debris: when a blade pass completes, the removed volume
``c_i a_pen dr_i`` is committed and converted into chips of the documented flake
volume ``V_flake = (0.35 c_i)(2 dr_i)(kappa a_pen)`` (``kappa`` = spall thickness
over penetration, order 1-10 for crushing-dominated milling).  Chips leave the
blade at the blade surface speed and are then integrated in water with quadratic
drag + added mass + buoyancy toward the BEM race flow; the *ejection velocity* is
recorded when a chip has travelled ``exit_distance`` from its release point.  Mass
is conserved exactly: the chip budget is drained by the committed pass volumes, so
``sum(chip mass) + pending = removed mass`` to machine precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contact import BladeIceContact
from .geometry import IceBlockGeometry, PropellerGeometry
from .hydro import OpenWaterConfig, OpenWaterState, solve_open_water

__all__ = [
    "MillingConfig",
    "MillingResult",
    "simulate_milling",
    "drag_relax_step",
    "drag_relax_analytic",
]

_GRAVITY = 9.81


@dataclass(frozen=True)
class MillingConfig:
    """Milling case definition.

    Parameters
    ----------
    propeller, ice, contact
        Geometry and contact models.
    rpm
        Nominal shaft speed [rev/min].
    advance_ratio
        Open-water advance ratio ``J = V_a / (n D)`` at ``rpm``; the tow speed is
        ``V_a = J n D`` so that a speed series keeps a fixed operating point.
    feed_per_rev
        Ice advance into the propeller plane per revolution [m/rev].
    n_strips
        Radial strip count.
    dt, duration, spin_up
        Time step [s], simulated duration [s] and the leading interval excluded
        from the steady statistics [s].
    ice_density, water_density, water_viscosity
        Material data [kg/m^3, kg/m^3, Pa s].
    spall_factor
        Flake thickness over penetration, ``kappa`` [-].
    chip_plan_factor
        Flake plan length over chord [-].
    added_mass_coefficient
        Added-mass coefficient of a chip [-] (0.5 for a sphere).
    drag_coefficient
        Quadratic drag coefficient of a chip [-].
    exit_distance
        Flight distance at which the chip speed is recorded as the ejection
        velocity [m] (default ``0.25 D``).
    max_flight_time
        Longest chip flight integrated [s].
    governed
        ``True``: ideal speed-holding servo; ``False``: finite-gain drive.
    drive_gain, inertia
        Drive gain [N m s/rad] and shaft inertia (propeller + entrained water)
        [kg m^2] used when ``governed=False``.
    """

    propeller: PropellerGeometry = field(default_factory=PropellerGeometry)
    ice: IceBlockGeometry = field(default_factory=IceBlockGeometry)
    contact: BladeIceContact = field(default_factory=BladeIceContact)
    rpm: float = 60.0
    advance_ratio: float = 0.40
    feed_per_rev: float = 3.0e-3
    n_strips: int = 32
    dt: float = 5.0e-4
    duration: float = 4.0
    spin_up: float = 1.0
    ice_density: float = 917.0
    water_density: float = 1025.0
    water_viscosity: float = 1.8e-3
    spall_factor: float = 4.0
    chip_plan_factor: float = 0.35
    spall_jitter: float = 0.0
    seed: int = 0
    added_mass_coefficient: float = 0.5
    drag_coefficient: float = 0.90
    exit_distance: float | None = None
    max_flight_time: float = 1.5
    governed: bool = True
    drive_gain: float = 3000.0
    inertia: float = 0.05
    hydro: OpenWaterConfig = field(default_factory=OpenWaterConfig)

    def __post_init__(self) -> None:
        if self.rpm <= 0.0:
            raise ValueError("rpm must be positive")
        if self.advance_ratio < 0.0:
            raise ValueError("advance_ratio must be non-negative")
        if self.feed_per_rev <= 0.0:
            raise ValueError("feed_per_rev must be positive")
        if self.dt <= 0.0 or self.duration <= 0.0:
            raise ValueError("dt and duration must be positive")
        if not 0.0 <= self.spin_up < self.duration:
            raise ValueError("spin_up must lie in [0, duration)")
        if self.spall_factor <= 0.0 or self.chip_plan_factor <= 0.0:
            raise ValueError("spall_factor and chip_plan_factor must be positive")
        if not 0.0 <= self.spall_jitter < 1.0:
            raise ValueError("spall_jitter must lie in [0, 1)")
        if not 0.0 <= self.added_mass_coefficient < 5.0:
            raise ValueError("added_mass_coefficient must be in [0, 5)")
        if self.drag_coefficient <= 0.0:
            raise ValueError("drag_coefficient must be positive")
        if self.exit_distance is not None and self.exit_distance <= 0.0:
            raise ValueError("exit_distance must be positive when given")

    # -- derived ----------------------------------------------------------
    @property
    def omega_nominal(self) -> float:
        """Nominal angular speed [rad/s]."""
        return 2.0 * math.pi * self.rpm / 60.0

    @property
    def revolutions_per_second(self) -> float:
        return self.rpm / 60.0

    @property
    def advance_velocity(self) -> float:
        """Tow speed ``V_a = J n D`` [m/s]."""
        return self.advance_ratio * self.revolutions_per_second * self.propeller.diameter

    @property
    def feed_velocity(self) -> float:
        """Ice advance rate ``v_f = feed_per_rev * n / 60`` [m/s]."""
        return self.feed_per_rev * self.revolutions_per_second

    @property
    def penetration_per_pass(self) -> float:
        """Cut depth removed by one blade pass, ``a_pen = feed_per_rev / Z`` [m]."""
        return self.feed_per_rev / self.propeller.n_blades

    @property
    def exit_distance_resolved(self) -> float:
        return 0.25 * self.propeller.diameter if self.exit_distance is None else self.exit_distance

    def as_dict(self) -> dict[str, Any]:
        return {
            "rpm": self.rpm,
            "advance_ratio": self.advance_ratio,
            "advance_velocity_m_s": self.advance_velocity,
            "feed_per_rev_m": self.feed_per_rev,
            "feed_velocity_m_s": self.feed_velocity,
            "penetration_per_pass_m": self.penetration_per_pass,
            "n_strips": self.n_strips,
            "dt_s": self.dt,
            "duration_s": self.duration,
            "spin_up_s": self.spin_up,
            "spall_factor": self.spall_factor,
            "spall_jitter": self.spall_jitter,
            "chip_plan_factor": self.chip_plan_factor,
            "exit_distance_m": self.exit_distance_resolved,
            "governed": self.governed,
            "ice_density": self.ice_density,
            "water_density": self.water_density,
        }


@dataclass
class MillingResult:
    """Time histories plus debris statistics of one milling run."""

    config: MillingConfig
    time: np.ndarray
    omega: np.ndarray
    torque_ice: np.ndarray
    torque_hydro: np.ndarray
    torque_total: np.ndarray
    thrust: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    blades_engaged: np.ndarray
    open_water: OpenWaterState
    # debris
    chip_release_time: np.ndarray
    chip_release_pos: np.ndarray
    chip_release_vel: np.ndarray
    chip_mass: np.ndarray
    chip_diameter: np.ndarray
    chip_exit_speed: np.ndarray
    chip_exit_time: np.ndarray
    chip_exit_distance: np.ndarray
    chip_paths: list[np.ndarray]
    # ledgers
    mass_removed: float
    mass_chips: float
    mass_pending: float
    ice_thickness_remaining: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def steady_mask(self) -> np.ndarray:
        """Samples inside the steady statistical window."""
        return self.time >= self.config.spin_up

    def chip_mask(self) -> np.ndarray:
        """Chips released inside the steady window (used for the statistics)."""
        return self.chip_release_time >= self.config.spin_up

    def torque_pulsation(self) -> dict[str, float]:
        """Torque pulsation metrics of ``Q_ice`` on the steady window."""
        mask = self.steady_mask()
        q = self.torque_ice[mask]
        q_mean = float(np.mean(q))
        if q_mean <= 0.0:
            raise ValueError("mean ice torque is not positive; pulsation undefined")
        q05, q50, q95 = (float(v) for v in np.percentile(q, [5.0, 50.0, 95.0]))
        q_min, q_max = float(np.min(q)), float(np.max(q))
        rms = float(np.sqrt(np.mean((q - q_mean) ** 2)))
        return {
            "Q_mean_Nm": q_mean,
            "Q_min_Nm": q_min,
            "Q_max_Nm": q_max,
            "Q_p05_Nm": q05,
            "Q_p50_Nm": q50,
            "Q_p95_Nm": q95,
            "delta_q_over_mean_p95": (q95 - q05) / q_mean,
            "delta_q_over_mean_ptp": (q_max - q_min) / q_mean,
            "delta_q_over_mean_rms": rms / q_mean,
        }

    def ejection_statistics(self) -> dict[str, float]:
        """Mass-weighted chip ejection statistics."""
        mask = self.chip_mask()
        speeds = self.chip_exit_speed[mask]
        masses = self.chip_mass[mask]
        release = np.linalg.norm(self.chip_release_vel[mask], axis=1)
        if speeds.size == 0:
            raise ValueError("no chips in the steady window")
        weights = masses / masses.sum()
        order = np.argsort(speeds)
        cumulative = np.cumsum(weights[order])
        def weighted_quantile(q: float) -> float:
            idx = int(np.searchsorted(cumulative, q))
            return float(speeds[order][min(idx, speeds.size - 1)])
        return {
            "n_chips": int(speeds.size),
            "ejection_speed_mean": float(np.mean(speeds)),
            "ejection_speed_mass_weighted": float(np.sum(weights * speeds)),
            "ejection_speed_p05": weighted_quantile(0.05),
            "ejection_speed_p50": weighted_quantile(0.50),
            "ejection_speed_p95": weighted_quantile(0.95),
            "ejection_speed_min": float(np.min(speeds)),
            "ejection_speed_max": float(np.max(speeds)),
            "release_speed_mean": float(np.mean(release)),
            "chip_mass_mean_kg": float(np.mean(masses)),
            "flight_time_mean_s": float(np.mean(self.chip_exit_time[mask])),
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready summary (histories stay in the arrays)."""
        return {
            "config": self.config.as_dict(),
            "propeller": self.config.propeller.as_dict(),
            "ice": self.config.ice.as_dict(self.config.propeller.radius),
            "contact": self.config.contact.as_dict(),
            "open_water": self.open_water.as_dict(),
            "torque_pulsation": self.torque_pulsation(),
            "ejection": self.ejection_statistics(),
            "ledger": {
                "mass_removed_kg": self.mass_removed,
                "mass_chips_kg": self.mass_chips,
                "mass_pending_kg": self.mass_pending,
                "mass_closure_rel_error": abs(self.mass_removed - (self.mass_chips + self.mass_pending))
                / max(self.mass_removed, 1e-30),
                "ice_thickness_min_remaining_m": float(np.min(self.ice_thickness_remaining)),
            },
            "meta": self.meta,
        }


def drag_relax_step(rel_velocity: np.ndarray, drag_coefficient: Any,
                    dt: float) -> np.ndarray:
    """One exact step of quadratic drag relaxation ``dv/dt = -k |w| w``.

    With a constant carrier velocity ``u`` and a constant ``k`` the solution of the
    relative velocity ``w = v - u`` is ``|w|(t) = |w0| / (1 + k |w0| t)`` with the
    direction preserved, so this update is *exact* for any ``dt`` (not a first-order
    scheme).  :func:`drag_relax_analytic` is the closed form used to verify it.
    """
    w = np.asarray(rel_velocity, dtype=float)
    magnitude = np.linalg.norm(w, axis=-1, keepdims=True)
    return w / (1.0 + np.asarray(drag_coefficient, dtype=float) * magnitude * float(dt))


def drag_relax_analytic(initial_velocity: np.ndarray, carrier_velocity: np.ndarray,
                        drag_coefficient: Any, time: float) -> np.ndarray:
    """Closed-form velocity of a chip under quadratic drag in a uniform flow."""
    v0 = np.asarray(initial_velocity, dtype=float)
    u = np.asarray(carrier_velocity, dtype=float)
    w0 = v0 - u
    magnitude = np.linalg.norm(w0, axis=-1, keepdims=True)
    return u + w0 / (1.0 + np.asarray(drag_coefficient, dtype=float) * magnitude * float(time))


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Map angles to ``[-pi, pi)``."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def simulate_milling(config: MillingConfig) -> MillingResult:
    """Run one transient propeller-ice milling case."""
    cfg = config
    geom = cfg.propeller
    strips = geom.radial_strips(cfg.n_strips)
    r = strips["r"]
    dr = strips["dr"]
    chord = strips["chord"]
    phi = strips["pitch_angle"]
    skew = strips["skew"]
    chord_angle = strips["chord_angle"]
    z_blades = geom.n_blades
    ice_mask = cfg.ice.radial_mask(r, geom.radius)
    n_strips_ice = int(np.count_nonzero(ice_mask))
    if n_strips_ice == 0:
        raise ValueError("the ice block does not overlap any radial strip")

    a_pen = cfg.penetration_per_pass
    rng = np.random.default_rng(cfg.seed)
    sector = cfg.ice.sector_width_rad
    entry = cfg.ice.entry_angle()

    open_water = solve_open_water(geom, cfg.revolutions_per_second, cfg.advance_velocity,
                                  cfg.hydro)
    race_velocity = open_water.race_axial_velocity(cfg.advance_velocity, factor=1.5)
    flow = np.array([0.0, 0.0, race_velocity])

    omega_target = cfg.omega_nominal
    omega = omega_target
    theta = 0.0

    n_steps = int(round(cfg.duration / cfg.dt)) + 1
    time = np.arange(n_steps) * cfg.dt

    torque_ice = np.zeros(n_steps)
    torque_hydro = np.zeros(n_steps)
    thrust_hist = np.zeros(n_steps)
    normal_hist = np.zeros(n_steps)
    tang_hist = np.zeros(n_steps)
    engaged_hist = np.zeros(n_steps)
    omega_hist = np.zeros(n_steps)

    ice_remaining = np.full(r.size, cfg.ice.thickness)
    pending_volume = np.zeros(r.size)
    q_hydro_ref = open_water.torque

    blade_offsets = 2.0 * np.pi * np.arange(z_blades) / z_blades
    prev_engaged = np.zeros((z_blades, r.size))
    flake_threshold = np.ones((z_blades, r.size))

    chips_t: list[float] = []
    chips_pos: list[np.ndarray] = []
    chips_vel: list[np.ndarray] = []
    chips_mass: list[float] = []
    chips_d: list[float] = []
    chips_speed: list[float] = []
    chips_exit_t: list[float] = []
    chips_exit_d: list[float] = []
    chips_paths: list[np.ndarray] = []

    capacity = 40000
    act_t = np.zeros(capacity)
    act_pos = np.zeros((capacity, 3))
    act_vel = np.zeros((capacity, 3))
    act_k = np.zeros(capacity)
    act_start = np.zeros((capacity, 3))
    act_age = np.zeros(capacity)
    act_index = np.full(capacity, -1, dtype=int)
    act_alive = np.zeros(capacity, dtype=bool)
    n_active = 0
    n_chips = 0

    removed_volume = 0.0
    friction_work = 0.0

    buoyancy_accel = _GRAVITY * (cfg.water_density - cfg.ice_density) / cfg.ice_density

    for step in range(n_steps):
        theta += omega * cfg.dt
        # ---- blade-ice engagement ---------------------------------------
        blade_azimuth = theta + blade_offsets[:, None] + skew[None, :]
        # angular distance past the sector's leading edge, in [0, 2 pi)
        x = (blade_azimuth - entry) % (2.0 * np.pi)
        engaged_fraction = (np.clip(x, 0.0, sector) - np.clip(x - chord_angle[None, :], 0.0, sector))
        engaged_fraction = np.clip(engaged_fraction / chord_angle[None, :], 0.0, 1.0)
        active = ice_mask[None, :] & (ice_remaining[None, :] > 1e-9)
        engaged_fraction = np.where(active, engaged_fraction, 0.0)

        # a new blade pass starts where the engagement goes from 0 to positive:
        # draw the brittle-flake release threshold of that pass
        starting = (engaged_fraction > 0.0) & (prev_engaged <= 0.0)
        if np.any(starting):
            fresh = 1.0 - cfg.spall_jitter * rng.random((z_blades, r.size))
            flake_threshold = np.where(starting, fresh, flake_threshold)
        released = engaged_fraction > flake_threshold
        engaged_fraction = np.where(released, 0.0, engaged_fraction)

        penetration = np.minimum(a_pen, ice_remaining)[None, :] * engaged_fraction
        f_n = cfg.contact.normal(dr[None, :], penetration)
        sliding_speed = omega * r
        f_t, mu = cfg.contact.tangential(f_n, sliding_speed[None, :])
        # blade-frame -> shaft decomposition
        q_strip = r[None, :] * (f_n * np.cos(phi)[None, :] + f_t * np.sin(phi)[None, :])
        t_strip = f_n * np.sin(phi)[None, :] - f_t * np.cos(phi)[None, :]
        torque_ice[step] = float(np.sum(q_strip))
        thrust_hist[step] = float(np.sum(t_strip))
        normal_hist[step] = float(np.sum(f_n))
        tang_hist[step] = float(np.sum(f_t))
        engaged_hist[step] = float(np.count_nonzero(engaged_fraction > 1e-12))
        friction_work += float(np.sum(np.where(engaged_fraction > 0.0,
                                               f_t * sliding_speed[None, :], 0.0))) * cfg.dt

        # ---- completed passes -> committed volume, debris ---------------
        completed = (prev_engaged > 0.0) & (engaged_fraction <= 0.0)
        if np.any(completed):
            pass_volume = (chord[None, :] * np.minimum(a_pen, ice_remaining)[None, :]
                           * dr[None, :])
            per_strip = np.sum(np.where(completed, pass_volume, 0.0), axis=0)
            pending_volume += per_strip
            removed_volume += float(np.sum(per_strip))
            ice_remaining = np.maximum(ice_remaining - a_pen * np.count_nonzero(completed, axis=0),
                                       0.0)
            # convert pending volume into chips, mass conserving
            for i in np.nonzero(per_strip > 0.0)[0]:
                v_chip = (cfg.chip_plan_factor * chord[i]) * (2.0 * dr[i]) * (cfg.spall_factor * a_pen)
                if v_chip <= 0.0:
                    continue
                while pending_volume[i] >= v_chip and n_chips < capacity - 1:
                    pending_volume[i] -= v_chip
                    # release at the blade azimuth that just swept the strip
                    az = float(blade_azimuth[0, i] % (2.0 * np.pi))
                    pos = np.array([r[i] * math.cos(az), r[i] * math.sin(az), 0.0])
                    vel = np.array([-omega * r[i] * math.sin(az),
                                    omega * r[i] * math.cos(az),
                                    cfg.feed_velocity])
                    mass = cfg.ice_density * v_chip
                    d_eq = (6.0 * v_chip / math.pi) ** (1.0 / 3.0)
                    k_drag = (0.5 * cfg.water_density * cfg.drag_coefficient
                              * math.pi * d_eq ** 2 / 4.0) / (mass * (1.0 + cfg.added_mass_coefficient))
                    chips_t.append(time[step])
                    chips_pos.append(pos)
                    chips_vel.append(vel)
                    chips_mass.append(mass)
                    chips_d.append(d_eq)
                    chips_speed.append(float("nan"))
                    chips_exit_t.append(float("nan"))
                    chips_exit_d.append(float("nan"))
                    chips_paths.append(np.empty((0, 3)))
                    act_t[n_chips] = time[step]
                    act_pos[n_chips] = pos
                    act_vel[n_chips] = vel
                    act_k[n_chips] = k_drag
                    act_start[n_chips] = pos
                    act_age[n_chips] = 0.0
                    act_index[n_chips] = n_chips
                    act_alive[n_chips] = True
                    n_chips += 1
        prev_engaged = engaged_fraction

        # ---- debris transport -------------------------------------------
        live = np.nonzero(act_alive[:n_chips])[0]
        if live.size:
            v = act_vel[live]
            w = v - flow[None, :]
            v_new = drag_relax_step(w, act_k[live][:, None], cfg.dt) + flow[None, :]
            v_new[:, 1] += buoyancy_accel * cfg.dt
            act_vel[live] = v_new
            act_pos[live] = act_pos[live] + v_new * cfg.dt
            act_age[live] += cfg.dt
            travelled = np.linalg.norm(act_pos[live] - act_start[live], axis=1)
            speed = np.linalg.norm(v_new, axis=1)
            reached = travelled >= cfg.exit_distance_resolved
            expired = act_age[live] >= cfg.max_flight_time
            done = reached | expired
            for local, global_idx in enumerate(live):
                if done[local]:
                    idx = int(act_index[global_idx])
                    chips_speed[idx] = float(speed[local])
                    chips_exit_t[idx] = float(time[step] - chips_t[idx])
                    chips_exit_d[idx] = float(travelled[local])
                    act_alive[global_idx] = False
            # store a coarse path sample for the figures
            for local, global_idx in enumerate(live):
                if done[local]:
                    continue
                idx = int(act_index[global_idx])
                if chips_paths[idx].shape[0] < 64 and (step % 4 == 0):
                    chips_paths[idx] = np.vstack([chips_paths[idx], act_pos[global_idx]])

        # ---- spindle dynamics -------------------------------------------
        n_inst = omega / (2.0 * math.pi)
        torque_hydro[step] = q_hydro_ref * (n_inst / cfg.revolutions_per_second) ** 2
        load = torque_ice[step] + torque_hydro[step]
        if cfg.governed:
            omega = omega_target
        else:
            # implicit (exact for a linear drive law and a frozen load): the
            # spindle time constant I/G is far shorter than dt for realistic
            # inertias, so an explicit update would be unstable
            drive_gain = cfg.drive_gain
            omega = ((cfg.inertia / cfg.dt) * omega + q_hydro_ref
                     + drive_gain * omega_target - load) / (cfg.inertia / cfg.dt + drive_gain)
            omega = max(omega, 0.0)
        omega_hist[step] = omega

    torque_total = torque_ice + torque_hydro

    # close the debris statistics: chips still flying record their current speed
    for idx in range(n_chips):
        if math.isnan(chips_speed[idx]):
            chips_speed[idx] = float(np.linalg.norm(act_vel[idx])) if idx < n_chips else 0.0
            chips_exit_t[idx] = float(cfg.duration - chips_t[idx])
            chips_exit_d[idx] = float(np.linalg.norm(act_pos[idx] - chips_pos[idx]))

    mass_removed = removed_volume * cfg.ice_density
    mass_chips = float(np.sum(chips_mass)) if chips_mass else 0.0
    mass_pending = float(np.sum(pending_volume)) * cfg.ice_density

    steady = time >= cfg.spin_up
    q_ice_mean = float(np.mean(torque_ice[steady]))
    q_tot_mean = float(np.mean(torque_total[steady]))
    meta = {
        "n_steps": n_steps,
        "n_strips_in_ice": n_strips_ice,
        "race_velocity_m_s": race_velocity,
        "feed_velocity_m_s": cfg.feed_velocity,
        "penetration_per_pass_m": a_pen,
        "rev_per_steady_window": float(cfg.rpm / 60.0 * (cfg.duration - cfg.spin_up)),
        "blade_passing_frequency_hz": float(cfg.rpm / 60.0 * geom.n_blades),
        "samples_per_bpf_period": float(1.0 / (cfg.rpm / 60.0 * geom.n_blades) / cfg.dt),
        "mean_ice_torque_Nm": q_ice_mean,
        "mean_total_torque_Nm": q_tot_mean,
        "mean_thrust_N": float(np.mean(thrust_hist[steady])),
        "mean_normal_force_N": float(np.mean(normal_hist[steady])),
        "mean_engaged_passes": float(np.mean(engaged_hist[steady])),
        "ice_power_W": q_ice_mean * cfg.omega_nominal,
        "friction_work_J": friction_work,
        "drive_work_J": float(np.sum(torque_total[1:] * omega_hist[1:]) * cfg.dt),
        "chip_kinetic_energy_J": float(np.sum(0.5 * np.asarray(chips_mass)
                                              * np.sum(np.asarray(chips_vel) ** 2, axis=1))
                                       ) if chips_mass else 0.0,
        "mu_mean": float(np.mean(mu)) if np.size(mu) else float("nan"),
        "ice_removal_rate_m3_s": removed_volume / cfg.duration,
        "nominal_removal_rate_m3_s": cfg.feed_velocity * float(np.sum(chord * dr * ice_mask)),
    }

    return MillingResult(
        config=cfg,
        time=time,
        omega=omega_hist,
        torque_ice=torque_ice,
        torque_hydro=torque_hydro,
        torque_total=torque_total,
        thrust=thrust_hist,
        normal_force=normal_hist,
        tangential_force=tang_hist,
        blades_engaged=engaged_hist,
        open_water=open_water,
        chip_release_time=np.asarray(chips_t, dtype=float),
        chip_release_pos=np.asarray(chips_pos, dtype=float) if chips_pos else np.zeros((0, 3)),
        chip_release_vel=np.asarray(chips_vel, dtype=float) if chips_vel else np.zeros((0, 3)),
        chip_mass=np.asarray(chips_mass, dtype=float),
        chip_diameter=np.asarray(chips_d, dtype=float),
        chip_exit_speed=np.asarray(chips_speed, dtype=float),
        chip_exit_time=np.asarray(chips_exit_t, dtype=float),
        chip_exit_distance=np.asarray(chips_exit_d, dtype=float),
        chip_paths=chips_paths,
        mass_removed=mass_removed,
        mass_chips=mass_chips,
        mass_pending=mass_pending,
        ice_thickness_remaining=ice_remaining,
        meta=meta,
    )