"""Disordered initial packings for the axisymmetric column collapse (m20811).

Why this module exists
----------------------
Every previous wave of ``CASE-COLUMN-COLLAPSE`` prepared the column on a
hexagonal close-packed (HCP) lattice, optionally displaced by a normal jitter.
The m20259 admissibility audit showed that jitter larger than about
``0.005 r_p`` already puts neighbouring spheres into overlap, so the archived
"ensemble" only brackets *lattice-adjacent* realisations
(``docs/AUDIT_COLUMN_COLLAPSE_ASPECT_BAND_m20533.md`` §5-1, "**not** a genuinely
disordered equilibrium packing (需要沉积/沉降生成目标 φ)").  That debt is what
this module pays: it builds the column the way an experimentalist does, by
**pluviation** -- grains are dropped one monolayer at a time into a cylindrical
container (side wall + floor), fall under gravity, and are allowed to come to
rest.  The container is then removed and the deposit collapses.

Difference from a jittered lattice (the physical point)
------------------------------------------------------
A jittered lattice stores *unbalanced* elastic energy: overlapping pairs are
pushed apart impulsively at ``t = 0``, which is a different initial energy state
from the experiment.  A pluviated deposit is at **static equilibrium**: its
overlaps are the ones gravity requires (``sigma_bottom / k_n``, i.e. a few
tenths of a percent of the radius) and its kinetic energy is zero.  The gate in
:func:`settle_report` therefore checks equilibrium overlap, not "zero overlap",
and the tolerance is pre-registered in :data:`EQUILIBRIUM_OVERLAP_FRAC`.

Contents
--------
``cylinder_wall_contact_forces``
    Rigid cylindrical side wall, same Cundall-Strack + rolling law as
    :func:`tensordem.granular_column.kernel.wall_contact_forces` (which only
    knows the horizontal floor).  Returns the same dictionary keys so the two
    can be summed.
``PluviationCase``
    Geometry, material and pouring protocol of one prepared column (SI).
``pluviate``
    Pour, settle, and gate the deposit; returns coordinates plus diagnostics.
``n_particles_for``
    Grain count for a target aspect ratio and packing fraction.

Units are SI (m, kg, s, N, J).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..granular3d_core import Material3D, build_pairs
from .kernel import (_normal_response, PairSpringState, WallSpringState, pair_contact_forces,
                     wall_contact_forces)

__all__ = [
    "EQUILIBRIUM_OVERLAP_FRAC",
    "PLUVIATION_SOURCE",
    "PluviationCase",
    "PluviationError",
    "RelaxCase",
    "cylinder_wall_contact_forces",
    "disorder_metrics",
    "n_particles_for",
    "pluviate",
    "prepare_disordered_column",
    "relax_in_container",
    "resolve_overlaps",
    "settle_report",
]

#: Pre-registered admissibility rule for a *deposited* packing: the deepest
#: overlap may not exceed this fraction of the grain radius.  A settled packing
#: cannot have zero overlap (soft contacts carry the weight:
#: ``sigma_bottom / k_n`` is of order ``1e-3 r_p``), but it also may not store
#: impulse-releasing overlap the way a jittered lattice does.  Not a tunable
#: knob: it is fixed before the first case runs and shipped in the gate.
EQUILIBRIUM_OVERLAP_FRAC = 0.02

#: Provenance of the preparation protocol (pluviation / air pluviation is the
#: standard laboratory way of preparing a loose-to-medium sand column).
PLUVIATION_SOURCE = (
    "Air pluviation of monodisperse spheres into a cylindrical container; "
    "protocol described in this module and in "
    "docs/CASE_COLUMN_COLLAPSE_DISORDERED_m20811.md")


class PluviationError(ValueError):
    """Raised for an invalid pluviation case or an ungated deposit."""


def _empty_wall_result(n: int) -> dict:
    zeros3 = np.zeros((n, 3), dtype=np.float64)
    return {
        "force": zeros3.copy(),
        "torque": zeros3.copy(),
        "n_contacts": 0,
        "normal_force_sum": 0.0,
        "dissip_collision": 0.0,
        "dissip_friction": 0.0,
        "dissip_rolling": 0.0,
        "energy_normal": 0.0,
        "energy_tangential": 0.0,
        "energy_rolling": 0.0,
        "energy_dead": 0.0,
        "release_loss": 0.0,
        "mu_eff_mean": 0.0,
        "mu_eff_max": 0.0,
        "mu_eff_min": 0.0,
        "inertial_number_mean": 0.0,
        "inertial_number_max": 0.0,
    }


def cylinder_wall_contact_forces(pos: np.ndarray, vel: np.ndarray,
                                 omega: np.ndarray, mat: Material3D, dt: float, *,
                                 wall_radius: float,
                                 mu_wall: float | None = None,
                                 mu_rolling_wall: float = 0.0,
                                 state: WallSpringState | None = None,
                                 update_state: bool = True,
                                 contact_mode: str = "one_sided",
                                 corotate_wall_rolling: bool = False) -> dict:
    """Contact forces against the rigid cylinder ``r = wall_radius``.

    The wall is a surface of revolution about ``z``; a grain at radius ``rr``
    is in contact when ``rr + r > wall_radius``.  The contact normal points
    inward (``-e_r``, toward the axis) and the contact point sits one radius
    outward of the grain centre, so the torque is ``(r e_r) x F_t``.  Every
    other detail (linear spring-dashpot normal law, Cundall-Strack tangential
    spring saturated at ``mu |F_n|``, rolling spring saturated at
    ``mu_r r_eff |F_n|``) is copied from
    :func:`~tensordem.granular_column.kernel.wall_contact_forces` so that the
    container and the floor are the same physics. With
    ``corotate_wall_rolling=True``, the stored rolling spring follows the
    cylindrical frame before adding the current spin increment.
    """
    n = pos.shape[0]
    out = _empty_wall_result(n)
    if not math.isfinite(wall_radius) or wall_radius <= 0.0:
        raise PluviationError("wall_radius must be finite and positive")
    mu_w = mat.mu if mu_wall is None else float(mu_wall)
    r = mat.radius
    kn, kt = mat.kn, mat.kt
    kr = mat.wall_rolling_kr
    rolling_reff = mat.wall_rolling_reff
    rolling_torsion = mat.rolling_torsion
    gamma = mat.gamma_wall

    rr = np.hypot(pos[:, 0], pos[:, 1])
    axis = rr > 1e-12
    overlap = np.where(axis, rr + r - wall_radius, -1.0)
    act = overlap > 0.0
    if not act.any():
        if state is not None and update_state:
            out["release_loss"] = state.release_inactive(act)
        return out

    idx = np.nonzero(act)[0]
    ov = overlap[idx]
    er = np.stack([pos[idx, 0] / rr[idx], pos[idx, 1] / rr[idx],
                   np.zeros(idx.size)], axis=1)          # outward radial
    nrm = -er                                            # wall -> grain centre
    v = vel[idx]
    vn = np.einsum("ij,ij->i", v, nrm)
    f_n, dashpot_work = _normal_response(kn * ov, vn, gamma, dt, contact_mode)

    # contact-point velocity: translation + spin about the contact at +r e_r.
    # (omega x (r e_r)) . e_r == 0, so the spin contributes nothing normal.
    omega_c = omega[idx]
    spin_part = np.cross(omega_c, er * r)
    u_surf = v + spin_part
    u_t = u_surf - np.einsum("ij,ij->i", u_surf, nrm)[:, None] * nrm

    force = out["force"]
    torque = out["torque"]
    force[idx] += f_n[:, None] * nrm
    out["n_contacts"] = int(idx.size)
    out["normal_force_sum"] = float(f_n.sum())

    hoop = np.column_stack((-er[:, 1], er[:, 0]))
    if state is None:
        tang_old = np.zeros((idx.size, 3))
        roll_old = np.zeros((idx.size, 3))
    else:
        tang_old = state.tang[idx].copy()
        roll_old = state.roll[idx].copy()
        if state.hoop_direction is not None:
            # On a cylinder the axial direction is fixed. Carry the signed
            # circumferential extension into the new local tangent plane.
            old_hoop = state.hoop_direction[idx]
            turned = state.active[idx] & np.any(old_hoop != hoop, axis=1)
            extension = (tang_old[turned, :2] * old_hoop[turned]).sum(axis=1)
            tang_old[turned, :2] = extension[:, None] * hoop[turned]
            if corotate_wall_rolling and mu_rolling_wall > 0.:
                # Rotate the stored vector about the cylinder axis; torsion,
                # when enabled, is the radial component of this vector.
                old, new = old_hoop[turned], hoop[turned]
                cosine = (old * new).sum(axis=1)
                sine = old[:, 0] * new[:, 1] - old[:, 1] * new[:, 0]
                x, y = roll_old[turned, :2].T
                roll_old[turned, :2] = np.column_stack((cosine*x - sine*y, sine*x + cosine*y))

    tang = tang_old + u_t * dt
    t_norm = np.sqrt(np.einsum("ij,ij->i", tang, tang))
    cap = mu_w * f_n / kt
    scale = np.where(t_norm > cap, cap / np.maximum(t_norm, 1e-30), 1.0)
    tang = tang * scale[:, None]
    f_t = -kt * tang
    slip_t = u_t * dt - (tang - tang_old)

    if mu_rolling_wall > 0.0:
        spin_w = omega_c
        if not rolling_torsion:
            # The cylindrical side wall has a radial normal at each contact.
            roll_old -= np.einsum("ij,ij->i", roll_old, er)[:, None] * er
            spin_w = spin_w - np.einsum("ij,ij->i", spin_w, er)[:, None] * er
        roll = roll_old + spin_w * dt
        r_norm = np.sqrt(np.einsum("ij,ij->i", roll, roll))
        cap_r = (mu_rolling_wall * rolling_reff * f_n) / kr
        scale_r = np.where(r_norm > cap_r, cap_r / np.maximum(r_norm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        torque_r = -kr * roll
        slip_r = spin_w * dt - (roll - roll_old)
    else:
        roll = np.zeros_like(roll_old)
        torque_r = np.zeros_like(roll_old)
        slip_r = np.zeros_like(roll_old)

    force[idx] += f_t
    torque[idx] += np.cross(er * r, f_t)
    if mu_rolling_wall > 0.0:
        torque[idx] += torque_r

    if state is not None and update_state:
        out["release_loss"] = state.release_inactive(act)
        if state.hoop_direction is None:
            state.hoop_direction = np.zeros((n, 2))
        state.hoop_direction[idx] = hoop
        state.tang[idx] = tang
        state.roll[idx] = roll
        state.live[idx] = .5 * (kn * ov**2 + kt * (tang * tang).sum(axis=1)
                                + kr * (roll * roll).sum(axis=1))

    out["dissip_collision"] = dashpot_work
    out["dissip_friction"] = float(np.sum(mu_w * f_n
                                         * np.linalg.norm(slip_t, axis=1)))
    out["dissip_rolling"] = float(np.sum(mu_rolling_wall * rolling_reff * f_n
                                         * np.linalg.norm(slip_r, axis=1)))
    out["energy_normal"] = float(np.sum(0.5 * kn * ov ** 2))
    out["energy_tangential"] = float(np.sum(0.5 * kt
                                           * np.einsum("ij,ij->i", tang, tang)))
    out["energy_rolling"] = float(np.sum(0.5 * kr
                                        * np.einsum("ij,ij->i", roll, roll)))
    if idx.size:
        out["mu_eff_mean"] = float(mu_w)
        out["mu_eff_max"] = float(mu_w)
        out["mu_eff_min"] = float(mu_w)
    return out


def n_particles_for(aspect: float, column_radius: float, radius: float,
                    phi_target: float) -> int:
    """Grain count of a column of height ``aspect * column_radius`` at ``phi``."""
    for name, value in (("aspect", aspect), ("column_radius", column_radius),
                        ("radius", radius), ("phi_target", phi_target)):
        if not math.isfinite(value) or value <= 0.0:
            raise PluviationError(f"{name} must be finite and positive")
    volume_grain = 4.0 / 3.0 * math.pi * radius ** 3
    volume_column = math.pi * column_radius ** 2 * aspect * column_radius
    return int(round(phi_target * volume_column / volume_grain))


@dataclass
class PluviationCase:
    """Geometry, material and pouring protocol of one prepared column (SI)."""

    #: radius of the cylindrical container (the column that is released)
    column_radius: float = 0.48
    #: grain radius
    radius: float = 0.03
    #: nominal aspect ratio ``a = h_i / r_i`` of the requested column
    aspect_ratio: float = 0.5
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.5
    mu_wall: float = 0.8
    restitution: float = 0.5
    kn: float = 6.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    #: packing fraction used to size the grain count (physical range 0.56-0.64)
    phi_target: float = 0.60
    #: free-fall drop of a poured grain, in grain diameters
    drop_diameters: float = 2.0
    #: grains poured between two integration slices, as a fraction of the
    #: number of grains in one grain layer of the container
    batches_per_layer: int = 4
    #: integration slice after each pour batch, in units of the free-fall time
    batch_time_factor: float = 1.2
    #: integration step; ``None`` -> the material's own estimate
    dt: float | None = None
    dt_scale: float = 1.0
    #: seed of the (seeded) pluviation pattern
    seed: int = 505
    #: gate: deepest overlap allowed in the settled deposit, as a fraction of r
    max_overlap_frac: float = EQUILIBRIUM_OVERLAP_FRAC
    #: gate: kinetic energy of the settled deposit that counts as "at rest" (J)
    ke_tol: float = 1.0e-9
    #: kinetic energy treated as rest during the pouring loop (J); for a
    #: 1000-grain column this is 1e-6 J per grain, i.e. |v| < 3 mm/s
    pouring_ke_tol: float = 1.0e-3
    #: compaction under elevated gravity after the pour: factor and duration (s)
    compaction_boost: float = 20.0
    compaction_time: float = 1.0
    #: relaxation back at 1 g after the compaction phase (s)
    relax_time: float = 0.3
    #: maximum settle time after the last grain is poured, in seconds
    max_settle_time: float = 1.5
    #: safety cap on the number of poured monolayers
    max_layers: int = 200
    label: str = ""
    corotate_wall_rolling: bool = False

    def __post_init__(self) -> None:
        for name in ("column_radius", "radius", "aspect_ratio", "density",
                     "kn", "kt_ratio", "gravity", "phi_target",
                     "drop_diameters", "batch_time_factor", "max_settle_time",
                     "batches_per_layer", "compaction_boost",
                     "compaction_time", "relax_time"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PluviationError(f"{name} must be finite and positive")
        if float(self.radius) >= float(self.column_radius):
            raise PluviationError("radius must be smaller than column_radius")
        if not 0.0 <= float(self.restitution) < 1.0:
            raise PluviationError("restitution must lie in [0, 1)")
        if float(self.mu_wall) < 0.0 or float(self.mu_rolling) < 0.0:
            raise PluviationError("friction coefficients must be nonnegative")
        if not 0.0 < float(self.max_overlap_frac):
            raise PluviationError("max_overlap_frac must be positive")

    # -- derived quantities -------------------------------------------------
    @property
    def grains_per_radius(self) -> float:
        """Column radius in grain radii (the resolution label used so far)."""
        return self.column_radius / self.radius

    @property
    def n_target(self) -> int:
        return n_particles_for(self.aspect_ratio, self.column_radius,
                               self.radius, self.phi_target)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return float(self.dt) * float(self.dt_scale)
        return self.material().timestep * float(self.dt_scale)

    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling,
                          restitution=self.restitution, kn=self.kn,
                          kt_ratio=self.kt_ratio, gravity=self.gravity)

    def signature(self) -> dict:
        return {"radius": float(self.radius), "density": float(self.density),
                "mu": float(self.mu), "mu_rolling": float(self.mu_rolling),
                "mu_wall": float(self.mu_wall), "corotate_wall_rolling": self.corotate_wall_rolling,
                "restitution": float(self.restitution), "kn": float(self.kn),
                "kt_ratio": float(self.kt_ratio),
                "gravity": float(self.gravity),
                "column_radius": float(self.column_radius),
                "aspect_ratio_nominal": float(self.aspect_ratio),
                "phi_target": float(self.phi_target),
                "drop_diameters": float(self.drop_diameters),
                "batches_per_layer": int(self.batches_per_layer),
                "batch_time_factor": float(self.batch_time_factor),
                "compaction_boost": float(self.compaction_boost),
                "compaction_time": float(self.compaction_time),
                "relax_time": float(self.relax_time),
                "seed": int(self.seed), "label": self.label}


def _random_disc(rng: np.random.Generator, r_max: float) -> tuple[float, float]:
    """Uniform-by-area point in the disc of radius ``r_max``."""
    rho = r_max * math.sqrt(rng.random())
    theta = 2.0 * math.pi * rng.random()
    return rho * math.cos(theta), rho * math.sin(theta)


def _free_spot(pos: np.ndarray, radius: float, r_max: float, z0: float,
               rng: np.random.Generator, tries: int = 80,
               max_lifts: int = 2) -> np.ndarray | None:
    """Non-overlapping insertion point at or above ``z0`` (``None`` if none).

    Candidate (x, y) is resampled over the container cross-section; a candidate
    that would overlap an already-poured grain is lifted by one diameter, but
    only ``max_lifts`` times -- beyond that the cross-section is resampled, so
    the deposit grows as a pile instead of as a column of stacked grains.
    """
    if pos.shape[0] == 0:
        x, y = _random_disc(rng, r_max)
        return np.array([x, y, z0])
    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    for _ in range(tries):
        x, y = _random_disc(rng, r_max)
        z = z0
        for _ in range(int(max_lifts) + 1):
            cand = np.array([x, y, z])
            dist, _ = tree.query(cand, k=1)
            gap = float(dist) - 2.0 * radius
            if gap >= 0.0:
                return cand
            z += 2.0 * radius          # lift by one diameter and retry
    return None


def _overlap_pairs(pos: np.ndarray, radius: float, tol: float = 0.0) -> int:
    """Number of pairs whose centre distance is below ``2r - tol``."""
    from scipy.spatial import cKDTree

    n = int(pos.shape[0])
    if n < 2:
        return 0
    pairs = cKDTree(pos).query_pairs(2.0 * radius, output_type="ndarray")
    if pairs.size == 0:
        return 0
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    return int(np.sum(d < 2.0 * radius - tol))


def _pile_surface(pos: np.ndarray, vel: np.ndarray, radius: float,
                  *, speed_tol: float = 0.15) -> float:
    """Free surface of the *settled* pile (grains still in flight excluded).

    Grains moving faster than ``speed_tol`` are raining, not piled, so taking
    ``max(z)`` over the whole array would place each new batch above the grains
    that are still falling and the pour would build a tower instead of a pile.
    """
    if pos.shape[0] == 0:
        return 0.0
    speed = np.linalg.norm(vel, axis=1)
    landed = pos[speed < float(speed_tol)]
    if landed.shape[0] == 0:
        return 0.0
    return float(landed[:, 2].max() + radius)


def settle_report(pos: np.ndarray, radius: float, column_radius: float,
                  *, max_overlap_frac: float = EQUILIBRIUM_OVERLAP_FRAC
                  ) -> dict:
    """Geometric + static equilibrium report of a settled deposit."""
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=float)
    n = int(pos.shape[0])
    tree = cKDTree(pos)
    dist, _ = tree.query(pos, k=2)
    min_gap = float(dist[:, 1].min() - 2.0 * radius) if n > 1 else float("inf")
    max_overlap = float(max(0.0, -min_gap))
    rr = np.hypot(pos[:, 0], pos[:, 1])
    h_meas = float(pos[:, 2].max() + radius)
    r_i_meas = float(np.quantile(rr, 0.995)) + radius
    volume_grain = 4.0 / 3.0 * math.pi * radius ** 3
    phi = n * volume_grain / (math.pi * column_radius ** 2 * h_meas)
    return {
        "n_particles": n,
        "column_radius_container": float(column_radius),
        "column_radius_measured": r_i_meas,
        "column_height_measured": h_meas,
        "aspect_measured": h_meas / r_i_meas,
        "packing_fraction": phi,
        "min_gap": min_gap,
        "max_overlap": max_overlap,
        "max_overlap_frac_of_radius": max_overlap / radius,
        "overlap_pairs": _overlap_pairs(pos, radius),
        "wall_overlap_max": float(max(0.0, (np.hypot(pos[:, 0], pos[:, 1]).max()
                                            + radius - column_radius))),
        "tolerance_frac_of_radius": float(max_overlap_frac),
    }


def _extend_contact_histories(pair_state: PairSpringState, floor_state: WallSpringState,
                              side_state: WallSpringState, n: int) -> None:
    """Keep the existing particles' springs when a batch is appended."""
    old_count = len(floor_state.active)
    added = n - old_count
    if not added:
        return
    if pair_state.keys.size:
        # pair_contact_forces encodes (i, j) as i * particle_count + j.
        i, j = np.divmod(pair_state.keys, old_count)
        pair_state.keys = i * n + j
    for wall in (floor_state, side_state):
        wall.active = np.pad(wall.active, (0, added))
        wall.live = np.pad(wall.live, (0, added))
        wall.tang = np.pad(wall.tang, ((0, added), (0, 0)))
        wall.roll = np.pad(wall.roll, ((0, added), (0, 0)))
        if wall.hoop_direction is not None:
            wall.hoop_direction = np.pad(wall.hoop_direction, ((0, added), (0, 0)))


def pluviate(case: PluviationCase, *, progress: bool = False) -> dict:
    """Pour, settle and gate one disordered column; return its state.

    Protocol (all steps deterministic given ``case.seed``):

    1. one grain at a time is inserted at a random point of the container's
       cross-section, one drop height (``drop_diameters``) above the current
       free surface; a candidate that would overlap an already-placed grain is
       lifted by one diameter until it is free (the grains rain, they are not
       pressed in);
    2. grains are poured in batches of ``1/batches_per_layer`` of a grain
       layer of the container, and after every batch the system is integrated
       for ``batch_time_factor`` free-fall times, so the poured grains land and
       the pile can rearrange/compact before the next batch arrives;
    3. pouring stops when the target grain count is placed; a final settle
       phase runs in 20 ms slices until the kinetic energy stays below
       ``pouring_ke_tol`` for five consecutive slices (or ``max_settle_time``),
       and the state is then frozen exactly at rest (velocities zeroed).

    The returned ``positions`` are the initial state handed to
    :func:`~tensordem.granular_column.axisym.simulate_axisymmetric_column`
    together with ``initial_positions=``.
    """
    mat = case.material()
    r = float(mat.radius)
    R = float(case.column_radius)
    r_max = R - r                       # grain-centre bound of the container
    if r_max <= 0.0:
        raise PluviationError("container is narrower than one grain")
    dt = case.timestep
    d = 2.0 * r
    drop = float(case.drop_diameters) * d
    layer_time = float(case.batch_time_factor) * math.sqrt(2.0 * drop
                                                           / mat.gravity)
    n_mono = max(1, int(round(n_particles_for(1.0, R, r, case.phi_target) * d / R)))
    #: grains poured between two integration slices: a fraction of a monolayer so
    #: the pile can rearrange instead of being stacked grain by grain
    pour_batch = max(1, int(n_mono) // max(1, int(case.batches_per_layer)))
    t_fall = math.sqrt(2.0 * drop / mat.gravity)
    rng = np.random.default_rng(int(case.seed))

    pos = np.zeros((0, 3))
    vel = np.zeros((0, 3))
    omega = np.zeros((0, 3))
    pair_state = PairSpringState()
    floor_state = WallSpringState(0)
    side_state = WallSpringState(0)
    poured_layers = 0
    steps = 0
    t_sim = 0.0
    ke = 0.0

    def integrate(n_steps: int, g_scale: float = 1.0) -> None:
        """Velocity Verlet on the (growing) particle set, in place.

        ``g_scale`` multiplies gravity; it is ``1`` for the pour and the final
        settle and ``>1`` only for the compaction phase (see
        :attr:`PluviationCase.compaction_boost`).
        """
        nonlocal pos, vel, omega, pair_state, floor_state, side_state, steps, t_sim, ke
        mass, inertia = mat.mass, mat.inertia
        gravity_vec = np.array([0.0, 0.0, -mat.gravity * float(g_scale)])
        iu, ju = build_pairs(pos, mat.radius)
        for _ in range(int(n_steps)):
            pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat,
                                       dt, update_state=False)
            pf = wall_contact_forces(pos, vel, omega, mat, dt,
                                     mu_wall=case.mu_wall,
                                     mu_rolling_wall=mat.mu_rolling,
                                     state=floor_state, update_state=False)
            ps = cylinder_wall_contact_forces(
                pos, vel, omega, mat, dt, wall_radius=R,
                mu_wall=case.mu_wall, mu_rolling_wall=mat.mu_rolling,
                corotate_wall_rolling=case.corotate_wall_rolling,
                state=side_state, update_state=False)
            acc = (pred["force"] + pf["force"] + ps["force"]) / mass + gravity_vec
            alpha = (pred["torque"] + pf["torque"] + ps["torque"]) / inertia
            v_half = vel + 0.5 * dt * acc
            w_half = omega + 0.5 * dt * alpha
            pos = pos + v_half * dt
            iu, ju = build_pairs(pos, mat.radius)

            cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state,
                                      mat, dt, update_state=True)
            cf = wall_contact_forces(pos, v_half, w_half, mat, dt,
                                     mu_wall=case.mu_wall,
                                     mu_rolling_wall=mat.mu_rolling,
                                     state=floor_state, update_state=True)
            cs = cylinder_wall_contact_forces(
                pos, v_half, w_half, mat, dt, wall_radius=R,
                mu_wall=case.mu_wall, mu_rolling_wall=mat.mu_rolling,
                corotate_wall_rolling=case.corotate_wall_rolling,
                state=side_state, update_state=True)
            acc = (cur["force"] + cf["force"] + cs["force"]) / mass + gravity_vec
            alpha = (cur["torque"] + cf["torque"] + cs["torque"]) / inertia
            vel = v_half + 0.5 * dt * acc
            omega = w_half + 0.5 * dt * alpha
            ke = 0.5 * float(mass) * float(np.sum(vel ** 2))
            steps += 1
            t_sim += dt

    def grow(new_pos: np.ndarray) -> None:
        """Append one grain to the current batch."""
        nonlocal pos, vel, omega
        pos = np.vstack([pos, new_pos[None, :]])
        vel = np.vstack([vel, np.zeros((1, 3))])
        omega = np.vstack([omega, np.zeros((1, 3))])

    n_target = case.n_target
    batch_steps = max(1, int(round(layer_time / dt)))
    guard = 0
    while pos.shape[0] < n_target and poured_layers < case.max_layers:
        # free surface of the *settled* pile: measured at the start of the batch,
        # so the grains of one batch rain side by side instead of stacking on top
        # of each other while still in flight
        h_surf = _pile_surface(pos, vel, r)
        placed_this_batch = 0
        for _ in range(pour_batch):
            if pos.shape[0] >= n_target:
                break
            spot = _free_spot(pos, r, r_max, h_surf + drop, rng)
            guard += 1
            if spot is None:
                continue
            grow(spot)
            placed_this_batch += 1
        if placed_this_batch == 0 and guard > 2000:
            break
        _extend_contact_histories(pair_state, floor_state, side_state, len(pos))
        integrate(batch_steps)
        poured_layers += 1
        if progress and poured_layers % 8 == 0:
            print(f"    [pour] batch {poured_layers}: N={pos.shape[0]}/{n_target} "
                  f"h_surf={h_surf:.3f} t={t_sim:.3f}s KE={ke:.3e} J", flush=True)
    if progress:
        print(f"    [pour] done: N={pos.shape[0]} batches={poured_layers} "
              f"t={t_sim:.3f}s t_fall={t_fall:.3f}s", flush=True)

    # compaction: settle under elevated gravity ("centrifuge-style" compaction,
    # the laboratory-standard way of taking a poured packing to a reproducible
    # relative density), then relax back to 1 g before the final settle
    phi_poured = settle_report(pos, r, R)["packing_fraction"]
    if case.compaction_boost > 1.0 and case.compaction_time > 0.0:
        integrate(int(round(case.compaction_time / dt)),
                  g_scale=float(case.compaction_boost))
    compaction_steps = steps
    if case.relax_time > 0.0:
        integrate(int(round(case.relax_time / dt)))
    # final settle: integrate in slices until at rest (KE below the pour gate)
    # for a sustained hold, or the settle-time cap is reached
    settle_slice_steps = max(1, int(round(0.02 / dt)))
    n_hold = 5                       # slices in a row below the gate = at rest
    held = 0
    settle_elapsed = 0.0
    while settle_elapsed < case.max_settle_time:
        integrate(settle_slice_steps)
        settle_elapsed += settle_slice_steps * dt
        held = held + 1 if ke < case.pouring_ke_tol else 0
        if held >= n_hold and settle_elapsed > 0.05:
            break
    # zero the residual velocities so the reported state is exactly at rest
    ke_poured = float(ke)
    vel = np.zeros_like(vel)
    omega = np.zeros_like(omega)

    report = settle_report(pos, r, R, max_overlap_frac=case.max_overlap_frac)
    report["kinetic_energy_at_rest"] = ke_poured
    report["ke_tol"] = float(case.ke_tol)
    report["at_rest"] = bool(ke_poured <= case.pouring_ke_tol)
    report["poured_layers"] = int(poured_layers)
    report["packing_fraction_poured"] = float(phi_poured)
    report["steps_pour_and_compaction"] = int(compaction_steps)
    report["n_target"] = int(n_target)
    report["steps"] = int(steps)
    report["time_simulated"] = float(t_sim)
    report["dt"] = float(dt)
    report["protocol"] = PLUVIATION_SOURCE
    report["case"] = case.signature()
    admissible = (report["max_overlap_frac_of_radius"] <= case.max_overlap_frac
                  and report["at_rest"]
                  and report["wall_overlap_max"] <= 0.5 * case.max_overlap_frac * r)
    report["admissible"] = bool(admissible)
    report["positions"] = pos.tolist()
    return report


# ---------------------------------------------------------------------------
# Primary preparation used by the m20811 wave: jitter a dense lattice, then let
# it relax to static equilibrium inside the container ("settle-then-release").
# ---------------------------------------------------------------------------


@dataclass
class RelaxCase:
    """Preparation of an equilibrated disordered column (SI)."""

    #: container radius (= the column that is released)
    column_radius: float = 0.48
    #: grain radius
    radius: float = 0.03
    #: nominal aspect ratio of the requested column
    aspect_ratio: float = 0.5
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.5
    mu_wall: float = 0.8
    restitution: float = 0.5
    kn: float = 6.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    #: gap of the underlying HCP lattice (2 % -> the historical packing)
    packing_gap: float = 0.02
    #: per-coordinate Gaussian jitter, in grain *diameters* (see ``prepare``)
    jitter_diameters: float = 0.2
    seed: int = 505
    dt: float | None = None
    dt_scale: float = 1.0
    #: gate on the equilibrated state
    max_overlap_frac: float = EQUILIBRIUM_OVERLAP_FRAC
    ke_tol: float = 1.0e-3
    #: settle time is capped here (s); the loop normally exits on rest first
    max_settle_time: float = 2.0
    #: minimum settle time before "at rest" can be declared (s)
    min_settle_time: float = 0.15
    label: str = ""
    corotate_wall_rolling: bool = False

    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling,
                          restitution=self.restitution, kn=self.kn,
                          kt_ratio=self.kt_ratio, gravity=self.gravity)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return float(self.dt) * float(self.dt_scale)
        return self.material().timestep * float(self.dt_scale)

    @property
    def grains_per_radius(self) -> float:
        return self.column_radius / self.radius

    def signature(self) -> dict:
        return {"radius": float(self.radius), "density": float(self.density),
                "mu": float(self.mu), "mu_rolling": float(self.mu_rolling),
                "mu_wall": float(self.mu_wall), "corotate_wall_rolling": self.corotate_wall_rolling,
                "restitution": float(self.restitution), "kn": float(self.kn),
                "kt_ratio": float(self.kt_ratio),
                "gravity": float(self.gravity),
                "column_radius": float(self.column_radius),
                "aspect_ratio_nominal": float(self.aspect_ratio),
                "packing_gap": float(self.packing_gap),
                "jitter_diameters": float(self.jitter_diameters),
                "seed": int(self.seed), "label": self.label}


def disorder_metrics(pos: np.ndarray, radius: float,
                     *, shell: float = 1.35) -> dict:
    """How far a packing is from the HCP lattice it may have started on.

    ``hex_order`` is the fraction of first-shell bond *pairs* around a grain
    that make an HCP-characteristic angle (multiples of 60 deg within 5 deg).
    An ideal HCP lattice scores 1; a disordered packing scores well below it,
    which is exactly the quantity the archived jitter ensembles never had to
    report -- so a "randomised" state can be checked instead of asserted.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=float)
    n = int(pos.shape[0])
    if n < 3:
        return {"n_particles": n, "coordination": float("nan"),
                "hex_order": float("nan"), "nn_distance_mean": float("nan")}
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, shell * 2.0 * radius)
    good = 0
    total = 0
    nn_sum = 0.0
    nn_count = 0
    for i, nbrs in enumerate(pairs):
        vec = pos[nbrs] - pos[i]
        dist = np.linalg.norm(vec, axis=1)
        nn_sum += float(dist.sum())
        nn_count += dist.size
        if dist.size < 3:
            continue
        u = vec / np.maximum(dist, 1e-30)[:, None]
        cos = np.clip(u @ u.T, -1.0, 1.0)
        iu = np.triu_indices(dist.size, k=1)
        c = cos[iu]
        total += c.size
        # bond pairs within +-5 deg of 60 or 120 deg (HCP in-plane angles)
        lo60, hi60 = np.cos(np.deg2rad(55.0)), np.cos(np.deg2rad(65.0))
        lo120, hi120 = -hi60, -lo60
        good += int(np.sum(((c < lo60) & (c > hi60))
                           | ((c > lo120) & (c < hi120))))
    return {
        "n_particles": n,
        "coordination": float(nn_count) / n,
        "nn_distance_mean": nn_sum / max(1, nn_count),
        "hex_order": (float(good) / total) if total else float("nan"),
        "hex_pairs": int(total),
    }


def relax_in_container(case: RelaxCase, positions: np.ndarray, *,
                       progress: bool = False) -> dict:
    """Integrate ``positions`` at rest under gravity inside the container.

    The cylindrical side wall plus the floor hold the column together while the
    jittered state resolves its overlaps; the loop stops when the kinetic
    energy stays below ``ke_tol`` for a sustained hold (or ``max_settle_time``),
    and the state is then frozen exactly at rest.  Everything the collapse needs
    is returned, including the admissibility gate of the *equilibrated* state.
    """
    mat = case.material()
    r = float(mat.radius)
    R = float(case.column_radius)
    dt = case.timestep
    pos = np.array(positions, dtype=np.float64, copy=True)
    n = int(pos.shape[0])
    vel = np.zeros((n, 3))
    omega = np.zeros((n, 3))
    mass, inertia = mat.mass, mat.inertia
    pair_state = PairSpringState()
    floor_state = WallSpringState(n)
    side_state = WallSpringState(n)
    gravity_vec = np.array([0.0, 0.0, -mat.gravity])
    ke = float("inf")
    steps = 0
    t_sim = 0.0
    slice_steps = max(1, int(round(0.01 / dt)))
    n_hold = 5
    held = 0
    iu, ju = build_pairs(pos, mat.radius)
    while t_sim < case.max_settle_time:
        for _ in range(slice_steps):
            pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat,
                                       dt, update_state=False)
            pf = wall_contact_forces(pos, vel, omega, mat, dt,
                                     mu_wall=case.mu_wall,
                                     mu_rolling_wall=mat.mu_rolling,
                                     state=floor_state, update_state=False)
            ps = cylinder_wall_contact_forces(
                pos, vel, omega, mat, dt, wall_radius=R,
                mu_wall=case.mu_wall, mu_rolling_wall=mat.mu_rolling,
                corotate_wall_rolling=case.corotate_wall_rolling,
                state=side_state, update_state=False)
            acc = (pred["force"] + pf["force"] + ps["force"]) / mass + gravity_vec
            alpha = (pred["torque"] + pf["torque"] + ps["torque"]) / inertia
            v_half = vel + 0.5 * dt * acc
            w_half = omega + 0.5 * dt * alpha
            pos = pos + v_half * dt
            iu, ju = build_pairs(pos, mat.radius)
            cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state,
                                      mat, dt, update_state=True)
            cf = wall_contact_forces(pos, v_half, w_half, mat, dt,
                                     mu_wall=case.mu_wall,
                                     mu_rolling_wall=mat.mu_rolling,
                                     state=floor_state, update_state=True)
            cs = cylinder_wall_contact_forces(
                pos, v_half, w_half, mat, dt, wall_radius=R,
                mu_wall=case.mu_wall, mu_rolling_wall=mat.mu_rolling,
                corotate_wall_rolling=case.corotate_wall_rolling,
                state=side_state, update_state=True)
            acc = (cur["force"] + cf["force"] + cs["force"]) / mass + gravity_vec
            alpha = (cur["torque"] + cf["torque"] + cs["torque"]) / inertia
            vel = v_half + 0.5 * dt * acc
            omega = w_half + 0.5 * dt * alpha
            steps += 1
            t_sim += dt
        ke = 0.5 * float(mass) * float(np.sum(vel ** 2))
        held = held + 1 if ke < case.ke_tol else 0
        if progress:
            print(f"    [relax] t={t_sim:.3f}s KE={ke:.3e} J", flush=True)
        if held >= n_hold and t_sim >= case.min_settle_time:
            break
    ke_relaxed = float(ke)
    vel = np.zeros_like(vel)
    omega = np.zeros_like(omega)
    report = settle_report(pos, r, R, max_overlap_frac=case.max_overlap_frac)
    report.update(disorder_metrics(pos, r))
    report["kinetic_energy_at_rest"] = ke_relaxed
    report["at_rest"] = bool(ke_relaxed <= case.ke_tol)
    report["relax_steps"] = int(steps)
    report["relax_time"] = float(t_sim)
    report["dt"] = float(dt)
    report["case"] = case.signature()
    admissible = (report["max_overlap_frac_of_radius"] <= case.max_overlap_frac
                  and report["at_rest"]
                  and np.hypot(pos[:, 0], pos[:, 1]).max() + r
                  <= R + 0.5 * case.max_overlap_frac * r)
    report["admissible"] = bool(admissible)
    report["positions"] = pos.tolist()
    return report


def resolve_overlaps(pos: np.ndarray, radius: float, r_max: float, *,
                     n_iter: int = 400, tol: float = 1e-9) -> dict:
    """Remove initial overlaps geometrically (position-based projection).

    A jitter of ``0.2-0.3 d`` puts neighbouring spheres into overlap by up to
    tens of percent of a radius.  Releasing that state *dynamically* hands the
    contact law a pile of compressed springs and the column decompresses like an
    explosion; the preparation therefore projects the overlaps away first --
    every overlapping pair is pushed apart along its centre line, and grains are
    kept inside the container -- iterating until no pair penetrates by more than
    ``tol``.  The result is a jittered *but contact-free* geometry, which is
    then relaxed dynamically to static equilibrium.
    """
    from scipy.spatial import cKDTree

    p = np.array(pos, dtype=np.float64, copy=True)
    n = int(p.shape[0])
    if n < 2:
        return {"positions": p, "iterations": 0, "max_overlap_before": 0.0,
                "max_overlap_after": 0.0}
    before = float(max(0.0, 2.0 * radius - cKDTree(p).query(p, k=2)[0][:, 1].min()))
    it = 0
    for it in range(1, int(n_iter) + 1):
        tree = cKDTree(p)
        pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
        if pairs.size == 0:
            break
        d = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
        pen = 2.0 * radius - d
        worst = float(pen.max())
        if worst <= tol:
            break
        keep = pen > 0.0
        pairs, d, pen = pairs[keep], d[keep], pen[keep]
        nrm = (p[pairs[:, 0]] - p[pairs[:, 1]]) / np.maximum(d, 1e-30)[:, None]
        disp = np.zeros_like(p)
        np.add.at(disp, pairs[:, 0], 0.5 * pen[:, None] * nrm)
        np.add.at(disp, pairs[:, 1], -0.5 * pen[:, None] * nrm)
        p += disp
        rr = np.hypot(p[:, 0], p[:, 1])
        out = rr > r_max
        if np.any(out):
            p[out, 0] *= r_max / rr[out]
            p[out, 1] *= r_max / rr[out]
        p[:, 2] = np.maximum(p[:, 2], radius)
    after = float(max(0.0, 2.0 * radius - cKDTree(p).query(p, k=2)[0][:, 1].min()))
    return {"positions": p, "iterations": int(it),
            "max_overlap_before": before, "max_overlap_after": after}


def prepare_disordered_column(case: RelaxCase, *, progress: bool = False) -> dict:
    """Jitter a dense lattice, then relax it to equilibrium inside the container.

    Why this is not the m20259 jitter ensemble: there the jittered lattice was
    released *immediately*, so overlapping pairs stored elastic energy that the
    contact law released impulsively at ``t = 0``.  Here the jittered state is
    held by the cylindrical wall and the floor while it relaxes, and only the
    equilibrated state (``at_rest``, equilibrium overlaps, no stored impulse) is
    released.  The preparation is therefore a physical realisation of *the same*
    column, and the amount of positional disorder is reported (``hex_order``)
    rather than assumed.
    """
    from .axisym import AxisymCase, build_axisymmetric_column

    base = AxisymCase(radius=case.radius, density=case.density, mu=case.mu,
                      mu_rolling=case.mu_rolling, mu_wall=case.mu_wall,
                      restitution=case.restitution, kn=case.kn,
                      kt_ratio=case.kt_ratio, gravity=case.gravity,
                      # AxisymCase counts *diameters* per column radius
                      grains_per_radius=0.5 * case.grains_per_radius,
                      aspect_ratio=case.aspect_ratio,
                      packing_gap=case.packing_gap, jitter=0.0,
                      seed=int(case.seed), dt=case.dt, dt_scale=case.dt_scale,
                      label=f"{case.label}_lattice")
    pos, _mat = build_axisymmetric_column(base)
    rng = np.random.default_rng(int(case.seed))
    sigma = float(case.jitter_diameters) * 2.0 * float(case.radius)
    pos = pos + rng.normal(0.0, sigma, size=pos.shape)
    # project the jitter-induced overlaps away before any dynamics: releasing a
    # compressed lattice gives the contact law a store of elastic energy that is
    # exactly what the m20259 gate rejects
    projected = resolve_overlaps(pos, case.radius,
                                 float(case.column_radius) - float(case.radius))
    pos = projected["positions"]
    jittered = settle_report(pos, case.radius, case.column_radius)
    rep = relax_in_container(case, pos, progress=progress)
    rep["jitter_sigma_m"] = sigma
    rep["overlap_projection"] = {k: v for k, v in projected.items()
                                 if k != "positions"}
    rep["jittered_state"] = {k: jittered[k] for k in
                             ("max_overlap", "max_overlap_frac_of_radius",
                              "overlap_pairs", "packing_fraction",
                              "aspect_measured", "column_height_measured")}
    rep["disorder_of_jittered_state"] = disorder_metrics(pos, case.radius)
    return rep
