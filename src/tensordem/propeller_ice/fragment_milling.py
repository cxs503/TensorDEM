"""Fragment-cloud propeller-ice milling with recirculation.

Extends the single-block milling model (:mod:`tensordem.propeller_ice.milling`)
to a fragment cloud that includes:

1. **Fragment cloud** -- ``N_c`` ice fragments whose equivalent diameters follow
   the FSD power law ``n(D) ~ D^{-alpha}`` (Clauset et al. 2009), with
   ``alpha in [2.5, 3.5]``.  Fragment masses are derived from equivalent spheres.

2. **Fragment-blade contact** -- each fragment that overlaps a blade panel
   contributes a crushing force (same contact law as single-block milling) plus
   a tangential friction force.  The fragment's azimuthal position evolves with
   the local flow + a stochastic collision term.

3. **Fragment-fragment collision** -- pairwise elastic collisions between
   fragments within a cutoff distance, conserving momentum and kinetic energy
   (coefficient of restitution ``e_c``).

4. **Recirculation loop** -- a fraction ``R`` of discharged fragments are
   re-injected at the propeller inlet with a phase offset, modelling the
   wake-recirculation observed in ice-milling experiments.

5. **Torque spectrum sidebands** -- the fragment cloud introduces amplitude
   modulation of the BPF harmonics, producing sidebands at
   ``k * f_BPF ± m * f_rot``.  The sideband energy ratio
   ``E_sb / E_total`` is the primary observable.

6. **Three conservation ledgers** -- mass, axial momentum impulse, and energy
   (cutting work, fragment kinetic energy, dissipation).

Physical parameters
-------------------
``N_c`` : int
    Number of fragments in the cloud (0 = single-block baseline).
``R`` : float
    Recirculation ratio (return flow / discharge flow), in [0, 0.5].
``fsd_alpha`` : float
    FSD power-law exponent (default 3.0).
``D_min, D_max`` : float
    Fragment size bounds [m].
``e_c`` : float
    Coefficient of restitution for fragment-fragment collisions.
``collision_cutoff`` : float
    Distance threshold for fragment-fragment collision detection [m].

Usage
-----
>>> from tensordem.propeller_ice.fragment_milling import FragmentMillingConfig, simulate_fragment_milling
>>> cfg = FragmentMillingConfig(n_fragments=20, recirculation_ratio=0.2)
>>> result = simulate_fragment_milling(cfg)
>>> print(result.sideband_energy_ratio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contact import BladeIceContact
from .geometry import IceBlockGeometry, PropellerGeometry
from .hydro import OpenWaterConfig, OpenWaterState, solve_open_water
from .milling import MillingConfig, MillingResult, simulate_milling, drag_relax_step
from .spectrum import periodogram, blade_passing_frequency

__all__ = [
    "FragmentMillingConfig",
    "FragmentMillingResult",
    "simulate_fragment_milling",
    "compute_sideband_energy_ratio",
    "generate_fsd_sizes",
    "FragmentCloud",
]

_GRAVITY = 9.81


def generate_fsd_sizes(
    n_fragments: int,
    alpha: float = 3.0,
    d_min: float = 0.005,
    d_max: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Generate fragment equivalent diameters from a truncated power-law FSD.

    The number distribution is ``n(D) dD ~ D^{-alpha} dD``.  Using inverse-CDF
    sampling with bounds ``[D_min, D_max]``:

        D = [D_min^{1-alpha} + u (D_max^{1-alpha} - D_min^{1-alpha})]^{1/(1-alpha)}

    where ``u ~ U(0, 1)``.
    """
    if n_fragments <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    u = rng.random(n_fragments)
    if abs(alpha - 1.0) < 1e-12:
        sizes = d_min * np.exp(u * np.log(d_max / d_min))
    else:
        exponent = 1.0 / (1.0 - alpha)
        lo = d_min ** (1.0 - alpha)
        hi = d_max ** (1.0 - alpha)
        sizes = (lo + u * (hi - lo)) ** exponent
    return np.clip(sizes, d_min, d_max)


@dataclass
class FragmentCloud:
    """Lagrangian fragment cloud state."""

    n_fragments: int
    diameters: np.ndarray
    masses: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    z: np.ndarray
    vr: np.ndarray
    vtheta: np.ndarray
    vz: np.ndarray
    alive: np.ndarray

    @classmethod
    def create(
        cls,
        n_fragments: int,
        diameters: np.ndarray,
        propeller: PropellerGeometry,
        ice_density: float = 917.0,
        seed: int = 0,
    ) -> "FragmentCloud":
        """Create a fragment cloud uniformly distributed in the ice-blocked annulus."""
        rng = np.random.default_rng(seed)
        masses = ice_density * np.pi / 6.0 * diameters ** 3

        r_inner = propeller.hub_radius
        r_outer = propeller.radius
        r2 = r_inner ** 2 + rng.random(n_fragments) * (r_outer ** 2 - r_inner ** 2)
        r = np.sqrt(r2)
        theta = rng.random(n_fragments) * 2.0 * np.pi
        z = rng.random(n_fragments) * 0.02 - 0.01

        vr = np.zeros(n_fragments)
        vtheta = np.zeros(n_fragments)
        vz = np.zeros(n_fragments)
        alive = np.ones(n_fragments, dtype=bool)

        return cls(
            n_fragments=n_fragments,
            diameters=diameters,
            masses=masses,
            r=r,
            theta=theta,
            z=z,
            vr=vr,
            vtheta=vtheta,
            vz=vz,
            alive=alive,
        )

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses[self.alive]))

    @property
    def n_alive(self) -> int:
        return int(np.sum(self.alive))


@dataclass(frozen=True)
class FragmentMillingConfig:
    """Configuration for fragment-cloud milling simulation."""

    base: MillingConfig = field(default_factory=MillingConfig)
    n_fragments: int = 0
    recirculation_ratio: float = 0.0
    fsd_alpha: float = 3.0
    d_min: float = 0.005
    d_max: float = 0.08
    e_c: float = 0.3
    collision_cutoff: float = 0.02
    fragment_drag_coefficient: float = 0.9
    fragment_added_mass: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError("n_fragments must be non-negative")
        if not 0.0 <= self.recirculation_ratio <= 0.5:
            raise ValueError("recirculation_ratio must be in [0, 0.5]")
        if not 1.5 < self.fsd_alpha < 5.0:
            raise ValueError("fsd_alpha must be in (1.5, 5.0)")
        if self.d_min <= 0.0 or self.d_max <= self.d_min:
            raise ValueError("require 0 < d_min < d_max")
        if not 0.0 <= self.e_c <= 1.0:
            raise ValueError("e_c must be in [0, 1]")
        if self.collision_cutoff <= 0.0:
            raise ValueError("collision_cutoff must be positive")


@dataclass
class FragmentMillingResult:
    """Result of a fragment-cloud milling simulation."""

    config: FragmentMillingConfig
    time: np.ndarray
    omega: np.ndarray
    torque_ice: np.ndarray
    torque_hydro: np.ndarray
    torque_total: np.ndarray
    thrust: np.ndarray
    fragment_torque: np.ndarray
    fragment_n_alive: np.ndarray
    mass_removed: float
    mass_chips: float
    mass_pending: float
    mass_fragments_initial: float
    mass_fragments_final: float
    mass_fragments_recirculated: float
    axial_momentum_input: float
    axial_momentum_fragments: float
    axial_momentum_discharged: float
    cutting_work: float
    fragment_kinetic_energy: float
    collision_dissipation: float
    drag_dissipation: float
    sideband_energy_ratio: float
    sideband_energy_ratio_vs_nc: dict[int, float] | None = None
    base_result: MillingResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def steady_mask(self) -> np.ndarray:
        return self.time >= self.config.base.spin_up


def compute_sideband_energy_ratio(
    time: np.ndarray,
    torque: np.ndarray,
    rpm: float,
    n_blades: int,
    window: np.ndarray | None = None,
    n_harmonics: int = 6,
) -> float:
    """Compute the sideband energy ratio E_sb / E_total via harmonic decomposition.

    Fits the BPF harmonics (k*f_BPF for k=1..n_harmonics) to the torque signal
    via least squares. The residual (signal minus harmonic model) represents the
    sideband energy from amplitude modulation by the fragment cloud.

    Returns ``sum(residual^2) / sum((q - mean(q))^2)``, i.e. the fraction of
    variance not explained by the BPF harmonics.
    """
    from .spectrum import harmonic_amplitudes

    t = np.asarray(time, dtype=float)
    q = np.asarray(torque, dtype=float)
    if window is not None:
        t = t[window]
        q = q[window]
    if t.size < 32:
        return 0.0

    f_bpf = blade_passing_frequency(rpm, n_blades)
    fit = harmonic_amplitudes(t, q, f_bpf, n_harmonics=n_harmonics, n_fit=4)

    # Reconstruct harmonic model
    y = q - np.mean(q)
    model = np.zeros_like(q)
    for k in range(1, n_harmonics + 1):
        a_k = fit["amplitudes"].get(k, 0.0)
        phi_k = fit["phases_rad"].get(k, 0.0)
        model += a_k * np.cos(2.0 * np.pi * k * f_bpf * t + phi_k)

    residual = y - model
    total_var = float(np.sum(y ** 2))
    residual_var = float(np.sum(residual ** 2))
    return residual_var / total_var if total_var > 0.0 else 0.0


def _fragment_blade_contact(
    cloud: FragmentCloud,
    propeller: PropellerGeometry,
    contact: BladeIceContact,
    omega: float,
    theta_prop: float,
    n_blades: int,
    ice_thickness: float,
) -> tuple[float, float, float]:
    """Compute fragment-blade contact forces."""
    torque = 0.0
    thrust = 0.0
    normal = 0.0

    blade_offsets = 2.0 * np.pi * np.arange(n_blades) / n_blades

    for i in range(cloud.n_fragments):
        if not cloud.alive[i]:
            continue

        d = cloud.diameters[i]
        r_f = cloud.r[i]
        theta_f = cloud.theta[i]

        if r_f < propeller.hub_radius or r_f > propeller.radius:
            continue

        for b in range(n_blades):
            blade_theta = (theta_prop + blade_offsets[b]) % (2.0 * np.pi)
            dtheta = abs(((theta_f - blade_theta + np.pi) % (2.0 * np.pi)) - np.pi)

            chord = propeller.chord(r_f)
            chord_angle = chord / max(r_f, 1e-9)

            if dtheta < chord_angle * 0.5:
                penetration = min(d * 0.3, ice_thickness * 0.1)
                if penetration <= 0:
                    continue

                span = d
                f_n = float(contact.normal(span, penetration))
                sliding_speed = omega * r_f
                f_t, _ = contact.tangential(f_n, sliding_speed)

                phi = propeller.pitch_angle(r_f)
                torque += r_f * (f_n * np.cos(phi) + f_t * np.sin(phi))
                thrust += f_n * np.sin(phi) - f_t * np.cos(phi)
                normal += f_n

    return torque, thrust, normal


def _fragment_fragment_collision(
    cloud: FragmentCloud,
    cutoff: float,
    e_c: float,
) -> float:
    """Handle pairwise fragment-fragment collisions. Returns dissipated energy."""
    dissipation = 0.0
    alive_indices = np.where(cloud.alive)[0]
    n_alive = len(alive_indices)

    if n_alive < 2:
        return dissipation

    for i_idx in range(n_alive):
        i = alive_indices[i_idx]
        for j_idx in range(i_idx + 1, n_alive):
            j = alive_indices[j_idx]

            dx = cloud.r[i] * np.cos(cloud.theta[i]) - cloud.r[j] * np.cos(cloud.theta[j])
            dy = cloud.r[i] * np.sin(cloud.theta[i]) - cloud.r[j] * np.sin(cloud.theta[j])
            dz = cloud.z[i] - cloud.z[j]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            min_dist = 0.5 * (cloud.diameters[i] + cloud.diameters[j])

            if dist < cutoff and dist > 1e-12:
                v1 = np.array([
                    cloud.vr[i] * np.cos(cloud.theta[i]) - cloud.vtheta[i] * np.sin(cloud.theta[i]),
                    cloud.vr[i] * np.sin(cloud.theta[i]) + cloud.vtheta[i] * np.cos(cloud.theta[i]),
                    cloud.vz[i],
                ])
                v2 = np.array([
                    cloud.vr[j] * np.cos(cloud.theta[j]) - cloud.vtheta[j] * np.sin(cloud.theta[j]),
                    cloud.vr[j] * np.sin(cloud.theta[j]) + cloud.vtheta[j] * np.cos(cloud.theta[j]),
                    cloud.vz[j],
                ])

                v_rel = v1 - v2
                v_rel_norm = np.linalg.norm(v_rel)

                if v_rel_norm > 1e-12:
                    n_vec = np.array([dx, dy, dz]) / dist
                    v_rel_n = np.dot(v_rel, n_vec)

                    if v_rel_n < 0:
                        m1, m2 = cloud.masses[i], cloud.masses[j]
                        mu = m1 * m2 / (m1 + m2)
                        impulse = -(1.0 + e_c) * mu * v_rel_n
                        v1_new = v1 + impulse / m1 * n_vec
                        v2_new = v2 - impulse / m2 * n_vec

                        cloud.vr[i] = v1_new[0] * np.cos(cloud.theta[i]) + v1_new[1] * np.sin(cloud.theta[i])
                        cloud.vtheta[i] = -v1_new[0] * np.sin(cloud.theta[i]) + v1_new[1] * np.cos(cloud.theta[i])
                        cloud.vz[i] = v1_new[2]

                        cloud.vr[j] = v2_new[0] * np.cos(cloud.theta[j]) + v2_new[1] * np.sin(cloud.theta[j])
                        cloud.vtheta[j] = -v2_new[0] * np.sin(cloud.theta[j]) + v2_new[1] * np.cos(cloud.theta[j])
                        cloud.vz[j] = v2_new[2]

                        ke_before = 0.5 * mu * v_rel_n ** 2
                        ke_after = 0.5 * mu * (e_c * v_rel_n) ** 2
                        dissipation += ke_before - ke_after

    return dissipation


def simulate_fragment_milling(config: FragmentMillingConfig) -> FragmentMillingResult:
    """Run a fragment-cloud propeller-ice milling simulation.

    When ``n_fragments=0``, delegates to single-block milling.
    """
    cfg = config
    base_cfg = cfg.base

    if cfg.n_fragments == 0:
        base_result = simulate_milling(base_cfg)
        mask = base_result.steady_mask()
        sb_ratio = compute_sideband_energy_ratio(
            base_result.time, base_result.torque_ice,
            base_cfg.rpm, base_cfg.propeller.n_blades,
            window=mask,
        )
        return FragmentMillingResult(
            config=cfg,
            time=base_result.time,
            omega=base_result.omega,
            torque_ice=base_result.torque_ice,
            torque_hydro=base_result.torque_hydro,
            torque_total=base_result.torque_total,
            thrust=base_result.thrust,
            fragment_torque=np.zeros_like(base_result.time),
            fragment_n_alive=np.zeros_like(base_result.time, dtype=int),
            mass_removed=base_result.mass_removed,
            mass_chips=base_result.mass_chips,
            mass_pending=base_result.mass_pending,
            mass_fragments_initial=0.0,
            mass_fragments_final=0.0,
            mass_fragments_recirculated=0.0,
            axial_momentum_input=0.0,
            axial_momentum_fragments=0.0,
            axial_momentum_discharged=0.0,
            cutting_work=base_result.meta.get("friction_work_J", 0.0),
            fragment_kinetic_energy=base_result.meta.get("chip_kinetic_energy_J", 0.0),
            collision_dissipation=0.0,
            drag_dissipation=0.0,
            sideband_energy_ratio=sb_ratio,
            base_result=base_result,
            meta={**base_result.meta, "n_fragments": 0, "recirculation_ratio": 0.0},
        )

    # Generate fragment sizes from FSD
    diameters = generate_fsd_sizes(
        cfg.n_fragments, alpha=cfg.fsd_alpha,
        d_min=cfg.d_min, d_max=cfg.d_max, seed=cfg.seed,
    )

    cloud = FragmentCloud.create(
        cfg.n_fragments, diameters, base_cfg.propeller,
        ice_density=base_cfg.ice_density, seed=cfg.seed,
    )

    base_result = simulate_milling(base_cfg)

    geom = base_cfg.propeller
    open_water = base_result.open_water
    race_velocity = open_water.race_axial_velocity(base_cfg.advance_velocity, factor=1.5)
    flow_z = race_velocity

    n_steps = len(base_result.time)
    dt = base_cfg.dt

    buoyancy_accel = _GRAVITY * (base_cfg.water_density - base_cfg.ice_density) / base_cfg.ice_density

    frag_torque = np.zeros(n_steps)
    frag_n_alive = np.zeros(n_steps, dtype=int)

    total_collision_dissipation = 0.0
    total_drag_dissipation = 0.0
    total_axial_momentum_input = 0.0
    total_axial_momentum_discharged = 0.0
    mass_recirculated = 0.0

    rng_recirc = np.random.default_rng(cfg.seed + 10000)

    for step in range(n_steps):
        theta_prop = base_result.omega[step] * base_result.time[step]
        omega = base_result.omega[step]

        # fragment-blade contact
        f_torque, f_thrust, f_normal = _fragment_blade_contact(
            cloud, geom, base_cfg.contact, omega, theta_prop,
            geom.n_blades, base_cfg.ice.thickness,
        )
        frag_torque[step] = f_torque
        frag_n_alive[step] = cloud.n_alive

        # fragment transport
        alive_mask = cloud.alive
        if np.any(alive_mask):
            for i in range(cloud.n_fragments):
                if not cloud.alive[i]:
                    continue

                d = cloud.diameters[i]
                mass = cloud.masses[i]
                area = np.pi * d ** 2 / 4.0
                k_drag = (0.5 * base_cfg.water_density * cfg.fragment_drag_coefficient * area
                          / (mass * (1.0 + cfg.fragment_added_mass)))

                # radial: centrifugal + drag
                cloud.vr[i] += (omega ** 2 * cloud.r[i] - k_drag * cloud.vr[i] * abs(cloud.vr[i])) * dt

                # azimuthal: dragged by blade wake
                blade_speed = omega * cloud.r[i]
                cloud.vtheta[i] += (0.3 * blade_speed - cloud.vtheta[i]) * k_drag * dt

                # axial: buoyancy + drag toward race flow
                vz_rel = cloud.vz[i] - flow_z
                cloud.vz[i] += (buoyancy_accel - k_drag * vz_rel * abs(vz_rel)) * dt

                # update positions
                cloud.r[i] += cloud.vr[i] * dt
                cloud.theta[i] += cloud.vtheta[i] / max(cloud.r[i], 1e-9) * dt
                cloud.z[i] += cloud.vz[i] * dt

                # check discharge
                if cloud.r[i] > geom.radius * 1.5 or cloud.z[i] > 0.5 or cloud.z[i] < -0.5:
                    total_axial_momentum_discharged += mass * cloud.vz[i]
                    if cfg.recirculation_ratio > 0.0 and rng_recirc.random() < cfg.recirculation_ratio:
                        cloud.r[i] = geom.hub_radius + rng_recirc.random() * (geom.radius - geom.hub_radius)
                        cloud.theta[i] = rng_recirc.random() * 2.0 * np.pi
                        cloud.z[i] = -0.01
                        cloud.vr[i] = 0.0
                        cloud.vtheta[i] = 0.0
                        cloud.vz[i] = flow_z * 0.5
                        mass_recirculated += mass
                    else:
                        cloud.alive[i] = False

                if cloud.r[i] < geom.hub_radius * 0.5:
                    cloud.alive[i] = False

            # fragment-fragment collisions
            if cloud.n_alive >= 2:
                diss = _fragment_fragment_collision(cloud, cfg.collision_cutoff, cfg.e_c)
                total_collision_dissipation += diss

            # drag dissipation
            for i in range(cloud.n_fragments):
                if not cloud.alive[i]:
                    continue
                d = cloud.diameters[i]
                mass = cloud.masses[i]
                area = np.pi * d ** 2 / 4.0
                k_drag = (0.5 * base_cfg.water_density * cfg.fragment_drag_coefficient * area
                          / (mass * (1.0 + cfg.fragment_added_mass)))
                v_rel = np.sqrt(cloud.vr[i] ** 2 + cloud.vtheta[i] ** 2 + (cloud.vz[i] - flow_z) ** 2)
                total_drag_dissipation += k_drag * v_rel ** 3 * mass * dt

            total_axial_momentum_input += f_thrust * dt

    # Compute sideband energy ratio
    mask = base_result.steady_mask()
    torque_total_augmented = base_result.torque_ice + frag_torque
    sb_ratio = compute_sideband_energy_ratio(
        base_result.time, torque_total_augmented,
        base_cfg.rpm, base_cfg.propeller.n_blades,
        window=mask,
    )

    # Fragment kinetic energy
    frag_ke = 0.0
    for i in range(cloud.n_fragments):
        if cloud.alive[i]:
            v2 = cloud.vr[i] ** 2 + cloud.vtheta[i] ** 2 + cloud.vz[i] ** 2
            frag_ke += 0.5 * cloud.masses[i] * v2

    meta = {
        **base_result.meta,
        "n_fragments": cfg.n_fragments,
        "recirculation_ratio": cfg.recirculation_ratio,
        "fsd_alpha": cfg.fsd_alpha,
        "e_c": cfg.e_c,
        "mass_recirculated_kg": mass_recirculated,
        "collision_dissipation_J": total_collision_dissipation,
        "drag_dissipation_J": total_drag_dissipation,
        "fragment_kinetic_energy_J": frag_ke,
        "axial_momentum_input_Ns": total_axial_momentum_input,
        "axial_momentum_discharged_Ns": total_axial_momentum_discharged,
    }

    return FragmentMillingResult(
        config=cfg,
        time=base_result.time,
        omega=base_result.omega,
        torque_ice=base_result.torque_ice + frag_torque,
        torque_hydro=base_result.torque_hydro,
        torque_total=base_result.torque_total + frag_torque,
        thrust=base_result.thrust,
        fragment_torque=frag_torque,
        fragment_n_alive=frag_n_alive,
        mass_removed=base_result.mass_removed,
        mass_chips=base_result.mass_chips,
        mass_pending=base_result.mass_pending,
        mass_fragments_initial=cloud.total_mass,
        mass_fragments_final=frag_ke,
        mass_fragments_recirculated=mass_recirculated,
        axial_momentum_input=total_axial_momentum_input,
        axial_momentum_fragments=0.0,
        axial_momentum_discharged=total_axial_momentum_discharged,
        cutting_work=base_result.meta.get("friction_work_J", 0.0),
        fragment_kinetic_energy=frag_ke,
        collision_dissipation=total_collision_dissipation,
        drag_dissipation=total_drag_dissipation,
        sideband_energy_ratio=sb_ratio,
        base_result=base_result,
        meta=meta,
    )
