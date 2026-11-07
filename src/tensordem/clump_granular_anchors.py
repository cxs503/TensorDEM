"""Macroscopic anchors for the non-spherical DEM layer.

Three independent macroscopic statements are produced here, each with the
protocol it needs (a granular packing is protocol dependent -- stating the
protocol is part of the evidence):

``measure_packing_fraction``
    A frictionless sphere packing is poured under gravity into a box and settled
    with a tap sequence.  The solid fraction is measured in horizontal slabs and
    compared with Scott & Kilgour's dense random packing value ``0.64``.
``cube_pile`` / ``measure_repose_angle``
    Cubic particles are represented as rigid 8-sphere clumps (the standard
    multi-sphere stand-in for a cube).  A column is released, the heap is allowed
    to settle, and the repose angle is read off a least-squares fit of the free
    surface profile.  Rotations are integrated with a quaternion update, so the
    cubes genuinely topple and interlock rather than sliding as spheres.
``orientation_order``
    During the pile formation the cubic-axis orientational order
    ``S(t) = 1.5 <cos^2 theta> - 0.5`` (``theta`` = angle between a cube axis and
    the settling axis) is tracked; a settling pile must reach a plateau.

Contact model for the clumps: linear normal spring + dashpot, Coulomb friction
with a viscous regularisation (``F_t = -mu F_n v_t/|v_t|``), all on the *member
spheres* of the union -- i.e. the same multi-sphere channel that
:func:`tensordem.convex_contact.clump_contact` verifies against closed forms.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CubeClumpBody",
    "ClumpPile",
    "measure_packing_fraction",
    "measure_repose_angle",
    "cube_pile",
    "scott_dense_random_packing",
]

SCOTT_DENSE_RANDOM_PACKING = 0.64


def scott_dense_random_packing() -> float:
    """Scott & Kilgour (1969) dense random packing fraction of equal spheres."""
    return SCOTT_DENSE_RANDOM_PACKING


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _random_quaternion(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


@dataclass
class CubeClumpBody:
    """Rigid body = 8 member spheres placed at the corners of a cube."""

    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    edge: float
    mass: float
    inertia: np.ndarray
    local_offsets: np.ndarray
    member_radius: float

    @classmethod
    def create(cls, position, edge: float, density: float, quaternion=None) -> "CubeClumpBody":
        radius = 0.25 * edge
        offset = 0.5 * edge - radius
        signs = np.array([[sx, sy, sz] for sx in (1.0, -1.0)
                          for sy in (1.0, -1.0) for sz in (1.0, -1.0)])
        local = signs * offset
        member_mass = density * 4.0 / 3.0 * math.pi * radius ** 3
        mass = 8.0 * member_mass
        inertia_scalar = 0.4 * member_mass * radius ** 2
        inertia = np.zeros(3)
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            inertia[axis] = np.sum(inertia_scalar + member_mass
                                   * np.sum(local[:, others] ** 2, axis=1))
        return cls(position=np.asarray(position, float),
                   quaternion=np.asarray(quaternion if quaternion is not None
                                         else np.array([1.0, 0.0, 0.0, 0.0]), float),
                   velocity=np.zeros(3), angular_velocity=np.zeros(3),
                   edge=edge, mass=mass, inertia=inertia,
                   local_offsets=local, member_radius=radius)

    def member_positions(self, rotation: np.ndarray | None = None) -> np.ndarray:
        rot = _quaternion_to_matrix(self.quaternion) if rotation is None else rotation
        return self.position + self.local_offsets @ rot.T

    def axes(self) -> np.ndarray:
        return _quaternion_to_matrix(self.quaternion)


class ClumpPile:
    """Cubic-clump granular column that collapses into a heap on a floor."""

    def __init__(self, count: int, edge: float = 0.05, density: float = 1000.0,
                 *, box_width: float = 0.45, box_depth: float = 0.20,
                 stiffness: float = 4.0e4, damping: float = 2.0,
                 friction: float = 0.5, gravity: float = 9.81, seed: int = 20261106,
                 column_width: int = 5, column_depth: int = 3) -> None:
        rng = np.random.default_rng(seed)
        bodies = []
        # Spacing must exceed the *bounding* diameter of a randomly oriented cube
        # (2 (sqrt(3) offset + r) = 0.0683 edge for edge = 0.05), otherwise the
        # initial state already overlaps and the run opens with a repulsion
        # impulse that has nothing to do with the physics being measured.
        spacing = 1.5 * edge
        catalogue = CubeClumpBody.create([0.0, 0.0, 0.0], edge, density)
        clearance = 2.0 * (np.linalg.norm(catalogue.local_offsets[0]) + catalogue.member_radius)
        spacing = max(spacing, 1.05 * clearance)
        depth_layers = int(column_depth)
        width = int(column_width)
        per_layer = depth_layers * width
        layers = max(1, int(math.ceil(count / per_layer)))
        # keep the seed column inside the box: an initial state above the top wall
        # would be pushed down by that wall and inject a spurious impulse
        layers = min(layers, max(1, int(1.1 // spacing)))
        for layer in range(layers):
            for i in range(width):
                for j in range(depth_layers):
                    if len(bodies) >= count:
                        break
                    x = (-0.5 * (width - 1) + i) * spacing
                    y = (-0.5 * (depth_layers - 1) + j) * spacing
                    z = 0.5 * clearance + layer * spacing
                    bodies.append(CubeClumpBody.create(
                        [x, y, z], edge, density, _random_quaternion(rng)))
        while len(bodies) < count:
            bodies.append(CubeClumpBody.create(
                [(-0.5 + rng.random()) * 0.2, (-0.5 + rng.random()) * 0.15,
                 0.5 * clearance + 0.1 + rng.random() * 0.1], edge, density,
                _random_quaternion(rng)))
        self.bodies = bodies[:count]
        self.box = np.array([[-0.5 * box_width, -0.5 * box_depth, 0.0],
                             [0.5 * box_width, 0.5 * box_depth, 1.2]])
        self.stiffness = stiffness
        self.damping = damping
        self.friction = friction
        self.gravity = gravity
        self.edge = edge
        self._cache_state()

    # -- helpers --------------------------------------------------------
    def _cache_state(self) -> None:
        self.positions = np.array([b.position for b in self.bodies])
        self.velocities = np.array([b.velocity for b in self.bodies])
        self.quaternions = np.array([b.quaternion for b in self.bodies])
        self.omegas = np.array([b.angular_velocity for b in self.bodies])
        self.masses = np.array([b.mass for b in self.bodies])
        self.inertias = np.array([b.inertia for b in self.bodies])
        self.radii = np.array([b.member_radius for b in self.bodies])
        self.local_offsets = self.bodies[0].local_offsets
        self.bounding_radius = float(np.linalg.norm(self.local_offsets[0]) + self.radii[0])

    def _write_back(self) -> None:
        for body, position, velocity, quaternion, omega in zip(
                self.bodies, self.positions, self.velocities, self.quaternions, self.omegas):
            body.position = position
            body.velocity = velocity
            body.quaternion = quaternion / np.linalg.norm(quaternion)
            body.angular_velocity = omega

    def member_world_positions(self) -> np.ndarray:
        """``(n_bodies, 8, 3)`` member centres in world coordinates."""
        rotations = np.array([_quaternion_to_matrix(q) for q in self.quaternions])
        return self.positions[:, None, :] + np.einsum("nij,kj->nki", rotations,
                                                      self.local_offsets)

    # -- one velocity-Verlet step with quaternion rotation --------------
    def step(self, dt: float) -> None:
        force, torque = self._forces()
        acceleration = force / self.masses[:, None] + np.array([0.0, 0.0, -self.gravity])
        velocity_half = self.velocities + 0.5 * dt * acceleration
        positions_new = self.positions + dt * velocity_half
        omega_half = self.omegas + 0.5 * dt * torque / self.inertias
        # quaternion exponential update (first order in dt, renormalised)
        angle = np.linalg.norm(omega_half, axis=1)
        delta = np.zeros_like(self.quaternions)
        small = angle < 1.0e-14
        half = 0.5 * dt * angle
        axis = np.divide(omega_half, angle[:, None], out=np.zeros_like(omega_half),
                         where=(angle > 0)[:, None])
        delta[:, 0] = np.where(small, 1.0, np.cos(half))
        delta[:, 1:] = np.where(small[:, None], 0.0,
                                0.5 * dt * axis * (np.sin(half) / np.where(small, 1.0, half))[:, None])
        quaternions_new = self._quaternion_multiply(delta, self.quaternions)
        quaternions_new /= np.linalg.norm(quaternions_new, axis=1)[:, None]

        saved = (self.positions, self.quaternions)
        self.positions = positions_new
        self.quaternions = quaternions_new
        force_new, torque_new = self._forces()
        self.positions, self.quaternions = saved

        velocities_new = velocity_half + 0.5 * dt * (
            force_new / self.masses[:, None] + np.array([0.0, 0.0, -self.gravity]))
        omegas_new = omega_half + 0.5 * dt * torque_new / self.inertias
        self.positions = positions_new
        self.quaternions = quaternions_new
        self.velocities = velocities_new
        self.omegas = omegas_new
        self._write_back()

    @staticmethod
    def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        wa, xa, ya, za = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        wb, xb, yb, zb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ], axis=1)

    # -- contact forces --------------------------------------------------
    def _forces(self):
        count = len(self.bodies)
        force = np.zeros((count, 3))
        torque = np.zeros((count, 3))

        centres = self.positions
        span = centres[:, None, :] - centres[None, :, :]
        distance = np.linalg.norm(span, axis=2)
        cutoff = 2.0 * self.bounding_radius + 1.0e-9
        mask = (distance < cutoff) & (~np.eye(count, dtype=bool))
        i_index, j_index = np.nonzero(np.triu(mask, k=1))
        members = self.member_world_positions()
        if i_index.size:
            self._pair_contacts(i_index, j_index, members, force, torque)
        self._wall_contacts(members, force, torque)
        return force, torque

    def _pair_contacts(self, i_index, j_index, members, force, torque) -> None:
        members_i = members[i_index]
        members_j = members[j_index]
        delta = members_j[:, None, :, :] - members_i[:, :, None, :]
        dist = np.linalg.norm(delta, axis=3)
        overlap = 2.0 * self.radii[i_index][:, None, None] - dist
        active = overlap > 0.0
        if not np.any(active):
            return
        inverse = np.divide(1.0, dist, out=np.zeros_like(dist), where=dist > 0.0)
        normal = delta * inverse[..., None]
        rel_vel = (self.velocities[j_index][:, None, None, :]
                   - self.velocities[i_index][:, None, None, :])
        closing = -np.einsum("pijk,pijk->pij", rel_vel, normal)
        magnitude = np.clip(self.stiffness * np.clip(overlap, 0.0, None)
                            + self.damping * np.clip(closing, 0.0, None) * active, 0.0, None)
        tang = rel_vel - rel_vel * np.einsum("pijk,pijk->pij", rel_vel, normal)[..., None] * normal
        tang_norm = np.linalg.norm(tang, axis=3)
        tang_dir = np.divide(tang, tang_norm[..., None], out=np.zeros_like(tang),
                             where=(tang_norm > 1.0e-12)[..., None])
        # Viscous regularisation of Coulomb friction: the magnitude saturates at
        # mu F_n but is linear in |v_t| below ``v_reg``, which removes the
        # direction-flip chatter that otherwise pumps energy into the pile.
        v_reg = 2.0e-3
        blend = np.tanh(tang_norm / v_reg)
        friction = -self.friction * magnitude[..., None] * blend[..., None] \
            * tang_dir * active[..., None]
        contact_force = magnitude[..., None] * normal + friction          # on body i (push away from j)
        contact_force = np.where(active[..., None], contact_force, 0.0)
        total_i = -np.sum(contact_force, axis=(1, 2))
        total_j = np.sum(contact_force, axis=(1, 2))
        np.add.at(force, i_index, total_i)
        np.add.at(force, j_index, total_j)
        # torques about the body centres
        arm_i = members_i - self.positions[i_index][:, None, :]
        arm_j = members_j - self.positions[j_index][:, None, :]
        torque_i = -np.sum(np.cross(arm_i[:, :, None, :], contact_force), axis=(1, 2))
        torque_j = np.sum(np.cross(arm_j[:, None, :, :], contact_force), axis=(1, 2))
        np.add.at(torque, i_index, torque_i)
        np.add.at(torque, j_index, torque_j)

    def _wall_contacts(self, members, force, torque) -> None:
        lo, hi = self.box[0], self.box[1]
        radii = self.radii[:, None]
        for axis in range(3):
            for sign, wall in ((1.0, lo[axis]), (-1.0, hi[axis])):
                overlap = (wall + radii - members[:, :, axis]) if sign > 0.0 else \
                          (members[:, :, axis] + radii - wall)
                active = overlap > 0.0
                if not np.any(active):
                    continue
                speed = self.velocities[:, None, axis] * sign
                magnitude = np.where(active,
                                     self.stiffness * np.clip(overlap, 0.0, None)
                                     + self.damping * np.clip(-speed, 0.0, None), 0.0)
                direction = np.zeros(3)
                direction[axis] = sign
                contact = magnitude[..., None] * direction
                np.add.at(force, np.arange(len(self.bodies)),
                          np.sum(contact, axis=1))
                arm = members - self.positions[:, None, :]
                np.add.at(torque, np.arange(len(self.bodies)),
                          np.sum(np.cross(arm, contact), axis=1))

    # -- diagnostics -----------------------------------------------------
    def surface_profile(self, bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
        x = self.positions[:, 0]
        z = self.positions[:, 2]
        edges = np.linspace(x.min() - 1e-9, x.max() + 1e-9, bins + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        heights = np.full(bins, np.nan)
        for index in range(bins):
            selected = (x >= edges[index]) & (x < edges[index + 1])
            if np.count_nonzero(selected) >= 2:
                heights[index] = np.max(z[selected])
        return centres, heights

    def orientation_order(self) -> float:
        orders = []
        for body in self.bodies:
            axis = body.axes()[:, 2]
            orders.append(axis[2] ** 2)
        orders = np.asarray(orders)
        return float(1.5 * np.mean(orders) - 0.5)


def measure_repose_angle(profile_x: np.ndarray, profile_z: np.ndarray,
                         *, flank_fraction: float = 0.35) -> dict:
    """Least-squares repose angle from the free-surface profile.

    Only the flanks are used: the top of the heap is artificially flat (the pile
    has not spread that far) and the toe is rounded, so a fit over the whole
    profile would understate the slope.
    """
    valid = np.isfinite(profile_z)
    x = profile_x[valid]
    z = profile_z[valid]
    if x.size < 4:
        raise ValueError("not enough profile bins to fit a repose angle")
    peak = int(np.argmax(z))
    left = slice(max(0, int(peak - flank_fraction * z.size)), peak)
    right = slice(peak + 1, min(z.size, peak + 1 + int(flank_fraction * z.size) + 1))
    angles = []
    for part in (left, right):
        xs, zs = x[part], z[part]
        if xs.size >= 2 and np.ptp(xs) > 0:
            slope, _ = np.polyfit(xs, zs, 1)
            angles.append(abs(math.degrees(math.atan(slope))))
    if not angles:
        raise ValueError("no usable flank")
    angle = float(np.mean(angles))
    return {"angle_deg": angle, "flank_angles_deg": [float(a) for a in angles],
            "profile_x": x.tolist(), "profile_z": z.tolist()}


def cube_pile(count: int = 60, edge: float = 0.05, steps: int = 12000,
              *, dt: float = 2.0e-4, report_every: int = 500, **kwargs) -> dict:
    """Release a cubic-particle column and follow the pile until it settles."""
    pile = ClumpPile(count=count, edge=edge, **kwargs)
    history = []
    for step in range(steps):
        pile.step(dt)
        if (step + 1) % report_every == 0:
            centres, heights = pile.surface_profile()
            history.append({"step": step + 1, "time": (step + 1) * dt,
                            "orientation_order": pile.orientation_order(),
                            "max_height": float(np.nanmax(heights)),
                            "kinetic_energy": float(0.5 * np.sum(
                                pile.masses[:, None] * pile.velocities ** 2)),
                            "profile_x": centres.tolist(),
                            "profile_z": [None if not np.isfinite(h) else float(h)
                                          for h in heights]})
    return {"pile": pile, "history": history}


def measure_packing_fraction(count: int = 320, radius: float = 0.02,
                             *, steps: int = 6000, gravity: float = 9.81,
                             stiffness: float = 4.0e4, damping: float = 50.0,
                             box_width: float = 0.42, box_height: float = 0.9,
                             taps: int = 2, tap_steps: int = 400,
                             compress_step: float = 5.0e-6,
                             pressure_ratio: float = 1.0,
                             safety: float = 0.02, seed: int = 20261107) -> dict:
    """Measure the dense random packing fraction by uniaxial compression.

    Protocol (a granular packing is protocol dependent, so the protocol is part of
    the evidence):

    1. *spawn*: a loose lattice cloud (spacing 2.1 r, so no initial overlap) in the
       lower part of the box;
    2. *settle*: rain under gravity with a normal dashpot, then tap twice
       (0.1 m/s random kicks + re-settling);
    3. *compress*: the top wall becomes a piston that descends at a constant rate
       until the wall force reaches ``pressure_ratio`` times the bed weight -- the
       quasi-static uniaxial-compression route to a dense packing, which is what
       "poured and tapped" is standing in for in the DEM literature;
    4. *measure*: with the bed filling the box under the piston, the solid
       fraction is ``N V_sphere / (A h)`` with ``h`` the current box height, so no
       free-surface or slab correction enters the number.
    """
    from tensordem.contact_timestep_ledger import SphereGas, cundall_critical_timestep

    rng = np.random.default_rng(seed)
    half = 0.5 * box_width
    box = np.array([[-half, -half, 0.0], [half, half, box_height]])
    side = int(math.ceil(count ** (1.0 / 3.0)))
    grid = 2.1 * radius
    positions = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(positions) >= count:
                    break
                positions.append([-0.5 * (side - 1) * grid + i * grid,
                                  -0.5 * (side - 1) * grid + j * grid,
                                  radius + 1.2 * radius + k * grid])
    positions = np.asarray(positions[:count]) + rng.normal(scale=0.2 * radius,
                                                           size=(count, 3))
    for axis in (0, 1):
        positions[:, axis] = np.clip(positions[:, axis], -half + radius, half - radius)
    positions[:, 2] = np.clip(positions[:, 2], radius, box_height - radius)
    velocities = rng.normal(scale=0.02, size=(count, 3))
    gas = SphereGas(positions=positions, velocities=velocities, radius=radius,
                    mass=1.0, stiffness=stiffness, box=box, gravity=gravity,
                    damping=damping)
    dt = safety * cundall_critical_timestep(1.0, stiffness)
    for _ in range(steps):
        gas.step(dt)
    for _ in range(taps):
        gas.velocities += rng.normal(scale=0.1, size=(count, 3))
        for _ in range(tap_steps):
            gas.step(dt)

    volume_sphere = 4.0 / 3.0 * math.pi * radius ** 3
    area = (2 * half) ** 2
    target_force = pressure_ratio * count * 1.0 * gravity

    def top_wall_force() -> float:
        """Elastic part of the piston load, i.e. the *static* compression pressure.

        The dashpot contribution is deliberately excluded: a single bounce would
        otherwise exceed the target load and stop the compression before it starts.
        """
        overlap = np.clip(gas.positions[:, 2] + radius - box[1, 2], 0.0, None)
        return float(gas.stiffness * np.sum(overlap))

    height = float(box[1, 2])
    compressed_steps = 0
    floor_height = max(1.5 * radius, count * volume_sphere / (area * 0.72))
    while top_wall_force() < target_force and height > floor_height:
        box[1, 2] -= compress_step
        height = float(box[1, 2])
        gas.step(dt)
        compressed_steps += 1
        if compressed_steps > 60_000:
            break
    # relax at fixed height so the measured state is quasi-static
    for _ in range(600):
        gas.step(dt)
    height = float(box[1, 2])
    fraction = count * volume_sphere / (area * height) if height > 0 else float("nan")
    return {"packing_fraction": float(fraction),
            "packing_fraction_before_compression": float("nan"),
            "scott": SCOTT_DENSE_RANDOM_PACKING,
            "relative_error": float(abs(fraction - SCOTT_DENSE_RANDOM_PACKING)
                                    / SCOTT_DENSE_RANDOM_PACKING),
            "bed_height": height, "compressed_steps": compressed_steps,
            "top_wall_force": float(top_wall_force()),
            "target_force": float(target_force),
            "count": count, "radius": radius, "steps": steps, "taps": taps, "dt": dt}
