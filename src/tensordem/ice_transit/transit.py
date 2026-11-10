"""Steady-state broken-ice transit: recycling inlet, pack boundary, energy ledger.

The laboratory frame is used throughout (the hull is kinematic and advances at
``V``); the *conveyor* follows the ship so that a transit of arbitrary length
runs with a compact floe count:

    active window   x in [x_stem(t) - L_down, x_stem(t) + L_up]
    recycling       a floe that falls behind ``x_stem - L_down`` re-enters at
                    ``x_stem + L_up`` with zero velocity and equilibrium heave

Because the window moves with the hull, ``L_up + L_down`` (200 m here) replaces
the 1 km domain that a fixed window would need for a 10 ship-length route.  The
re-injection is book-kept: the kinetic and potential energy carried out with a
recycled floe is added back on the ``removed`` side of the ledger, so the
balance stays exact.

Energy ledger
-------------
Every force acting on the ice is one of

``hull``        input: work ``W_in = int sum F_hull . v_floe dt``
``water drag``  sink: ``P_drag = sum rho Cd A |v|^3``
``heave``       buoyancy + weight (conservative, ``PE_heave``) and heave dashpot
``contacts``    floe-floe / floe-hull: elastic potential ``U`` + dashpot and
                Coulomb dissipation (both instrumented in :mod:`.laws`)
``pack wall``   elastic potential + dashpot dissipation
``recycling``    floes leaving the window carry energy out (``W_rem``)

so the audit is

    W_in + W_inj - W_rem - W_diss = d(E_ke + E_heave + E_contact)

with every term measured from the state, never assumed.  The residual of that
identity (normalised by the hull input work) is reported with the case.

Steady resistance
-----------------
``R_ice = sum_i F_hull->floe,i . e_x`` is the reaction the ice exerts on the hull
along the track.  It is averaged over the steady window (after the conveyor has
been populated, ``t >= steady_start``), with standard deviation, standard error,
peak and a linear-drift stationarity statistic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from ..ice_channel import WaterEnvironment, submerged_draught
from ..broken_ice_resistance.hull import ShipForm
from .laws import (CrushingLaw, floe_floe_contacts, hull_contacts,
                   polygon_distance, wall_contacts)

__all__ = ["SteadyTransit", "TransitResult"]

GRAVITY = 9.80665


@dataclass
class TransitResult:
    """One steady-state transit measurement."""

    spec: dict[str, Any]
    hull: dict[str, Any]
    mean_kN: float
    std_kN: float
    sem_kN: float
    peak_kN: float
    min_kN: float
    stationarity_drift: float
    contact_count_mean: float
    floe_count: int
    realised_concentration: float
    simulated_time_s: float
    steady_window_s: tuple[float, float]
    route_length_m: float
    steps: int
    wall_time_s: float
    # energy ledger [kJ]
    hull_input_kJ: float
    thrust_work_kJ: float
    slip_work_kJ: float
    dissipation_kJ: dict[str, float]
    energy_stored_change_kJ: dict[str, float]
    injected_kJ: float
    numerical_injection_kJ: float
    numerical_injection_share: float
    removed_kJ: float
    ledger_residual_kJ: float
    ledger_residual_rel: float
    ledger_sha256: str = ""
    state_digest: str = ""
    series: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "series"}
        if include_series:
            payload["series"] = self.series
        return payload


def _relax(centres: torch.Tensor, radii: torch.Tensor,
           generator: torch.Generator, iterations: int = 200) -> torch.Tensor:
    """Push overlapping disks apart until the field has no overlap.

    Deterministic pair relaxation (Jacobi style): every overlapping pair is
    moved apart by half of its overlap along the pair normal.  A tiny random
    component breaks the symmetric stalls of a perfect lattice.  The field is
    *not* re-compacted afterwards, so the realised coverage can be a fraction of
    a per cent below the requested one; it is measured and reported
    (:meth:`SteadyTransit.realised_concentration`).
    """
    n = centres.shape[0]
    if n < 2:
        return centres
    pos = centres.clone()
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
        overlap = (radii[:, None] + radii[None, :]) - dist
        iu, ju = torch.triu_indices(n, n, offset=1)
        pen = overlap[iu, ju] - 1e-6
        active = pen > 0.0
        if not bool(active.any()):
            break
        i, j = iu[active], ju[active]
        direction = delta[i, j] / dist[i, j][:, None]
        shift = 0.5 * pen[active][:, None] * direction
        adjustment = torch.zeros_like(pos)
        adjustment.index_add_(0, i, shift)
        adjustment.index_add_(0, j, -shift)
        pos = pos + adjustment
    return pos


def _max_radius_spread(concentration: float) -> float:
    """Largest unilateral radius spread ``j`` that can fill ``concentration``.

    ``C(j) = (pi/(2 sqrt 3)) (1 + j^2/3) / (1 + j)^2`` is the coverage of a
    triangular lattice of disks whose spacing clears the largest disk; it
    decreases monotonically from the hexagonal limit ``0.9069`` at ``j = 0``.
    Solved by bisection so that the field never needs an overlap to reach the
    requested concentration.
    """
    limit = math.pi / (2.0 * math.sqrt(3.0))
    if concentration >= limit:
        return 0.0
    if concentration <= 0.0:
        return 1.0

    def coverage(j: float) -> float:
        return limit * (1.0 + j * j / 3.0) / (1.0 + j) ** 2

    low, high = 0.0, 4.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if coverage(mid) > concentration:
            low = mid
        else:
            high = mid
    return low


class SteadyTransit:
    """Plan-view floe field + kinematic hull + recycling inlet + energy ledger."""

    # -- construction -------------------------------------------------------
    def __init__(self, radius: Sequence[float], thickness: Sequence[float], *,
                 hull: ShipForm, speed: float, law: CrushingLaw, friction: float,
                 water: WaterEnvironment, window_upstream: float,
                 window_downstream: float, pack_width: float, dt: float,
                 normal_damping_ratio: float = 0.2,
                 tangential_damping_ratio: float = 0.2,
                 heave_damping_ratio: float = 0.2,
                 wall_damping_ratio: float = 0.2,
                 dtype=torch.float64) -> None:
        self.dtype = dtype
        self.radius = torch.as_tensor(list(radius), dtype=dtype)
        self.thickness = torch.as_tensor(list(thickness), dtype=dtype)
        if self.radius.numel() != self.thickness.numel() or self.radius.numel() == 0:
            raise ValueError("radius and thickness must be non-empty and equally long")
        self.n = int(self.radius.numel())
        self.hull = hull
        self.speed = float(speed)
        self.law = law
        self.friction = float(friction)
        self.water = water
        self.window_upstream = float(window_upstream)
        self.window_downstream = float(window_downstream)
        self.pack_width = float(pack_width)
        self.dt = float(dt)
        self.normal_damping_ratio = float(normal_damping_ratio)
        self.tangential_damping_ratio = float(tangential_damping_ratio)
        self.heave_damping_ratio = float(heave_damping_ratio)
        self.wall_damping_ratio = float(wall_damping_ratio)

        area = math.pi * self.radius ** 2
        self.waterplane_area = area
        self.mass = water.ice_density * area * self.thickness
        self.inertia_yaw = 0.5 * self.mass * self.radius ** 2
        self.wall_stiffness = water.water_density * water.gravity * area

        self.pos = torch.zeros((self.n, 2), dtype=dtype)
        self.vel = torch.zeros((self.n, 2), dtype=dtype)
        self.omega = torch.zeros(self.n, dtype=dtype)
        self.z = self.equilibrium_heave(self.thickness)
        self.vz = torch.zeros(self.n, dtype=dtype)
        self.time = 0.0
        self.steps = 0

        # ledger accumulators [J] and dissipative powers [W]
        self.w_in = 0.0
        self.w_thrust = 0.0
        self.w_diss = 0.0
        self.w_injected = 0.0
        self.w_removed = 0.0
        self.diss_channels: dict[str, float] = {
            "water_drag": 0.0, "hull_normal_damping": 0.0, "hull_sliding": 0.0,
            "floe_normal_damping": 0.0, "floe_sliding": 0.0,
            "pack_wall": 0.0, "heave_damping": 0.0,
        }
        self.w_numerical = 0.0
        self.recycle_count = 0
        self.initial_energy = 0.0
        self._generator = torch.Generator().manual_seed(0)

    # -- helpers ------------------------------------------------------------
    def equilibrium_heave(self, thickness: torch.Tensor) -> torch.Tensor:
        ratio = self.water.ice_density / self.water.water_density
        return 0.5 * thickness - ratio * thickness

    def x_stem(self, t: Optional[float] = None) -> float:
        t = self.time if t is None else t
        return self.hull.x_start + self.speed * t

    def hull_distance(self, x_stem: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """True signed distance of every floe centre to the hull outline."""
        normals, points = self.hull.outline(self.x_stem() if x_stem is None else x_stem)
        return polygon_distance(self.pos, points.to(self.dtype), normals.to(self.dtype))

    def hull_velocity(self) -> torch.Tensor:
        return torch.tensor([self.speed, 0.0], dtype=self.dtype)

    def set_seed(self, seed: int) -> None:
        self._generator = torch.Generator().manual_seed(int(seed))

    # -- field --------------------------------------------------------------
    @classmethod
    def build(cls, *, concentration: float, mean_radius: float,
              radius_jitter: float, radius_range: tuple[float, float],
              thickness: float, hull: ShipForm, speed: float, law: CrushingLaw,
              friction: float, water: WaterEnvironment, pack_width: float,
              window_upstream: float, window_downstream: float, dt: float,
              seed: int, ice_modulus: float) -> "SteadyTransit":
        """Jittered hexagonal floe field filling the conveyor window.

        Placement follows :meth:`tensordem.ice_channel.IceChannelSimulation.
        from_field` (equal-area cells, realised coverage = requested coverage to
        <1 %, no initial overlap) but over the *moving* window
        ``[x_stem, x_stem + window_upstream] x [0, pack_width]``.
        """
        if not 0.0 <= concentration <= 1.0:
            raise ValueError("concentration must lie in [0, 1]")
        generator = torch.Generator().manual_seed(int(seed))
        length = float(window_upstream)
        width = float(pack_width)
        r_max = min(mean_radius * (1.0 + radius_jitter), radius_range[1])
        r_min = max(mean_radius * (1.0 - radius_jitter), radius_range[0])
        # Geometric limit: one disk per triangular-lattice site with a unilateral
        # radius spread j (r = rbar (1 + j u), u in [-1, 1]) fills
        #     C(j) = (pi / (2 sqrt 3)) * (1 + j^2/3) / (1 + j)^2
        # because the lattice spacing must clear the *largest* disk.  Near the
        # hexagonal limit the field must therefore be monodisperse; the realised
        # spread is reported with every case instead of silently overlapping.
        j_limit = _max_radius_spread(concentration)
        jitter = min(float(radius_jitter), j_limit)
        # the pack is populated between the walls, with a margin so that no floe
        # is born inside the compliant boundary
        x0 = r_max + 0.5
        x1 = length
        y0 = r_max + 0.5
        y1 = width - r_max - 0.5
        if x1 <= x0 or y1 <= y0:
            raise ValueError("the conveyor window is too small for the floe size")
        rbar = 0.5 * (r_min + r_max)
        cell_area = math.pi * rbar ** 2 / max(concentration, 1e-9)
        s_lat = math.sqrt(2.0 * cell_area / math.sqrt(3.0))
        n_cols = max(int((x1 - x0) // s_lat), 1)
        n_rows = max(int((y1 - y0) // (s_lat * math.sqrt(3.0) / 2.0)), 1)
        xs, ys = [], []
        for row in range(n_rows):
            y = y0 + (row + 0.5) * s_lat * math.sqrt(3.0) / 2.0
            offset = 0.5 * s_lat if row % 2 else 0.0
            for col in range(n_cols):
                x = x0 + (col + 0.5) * s_lat + offset
                if x <= x1:
                    xs.append(x)
                    ys.append(y)
        centres = torch.tensor(list(zip(xs, ys)), dtype=torch.float64)
        jitter_xy = (torch.rand(centres.shape, generator=generator, dtype=torch.float64)
                     - 0.5) * 0.15 * s_lat
        centres = centres + jitter_xy
        spread = 1.0 + jitter * (2.0 * torch.rand(
            centres.shape[0], generator=generator, dtype=torch.float64) - 1.0)
        radii = (0.5 * s_lat / (1.0 + jitter)) * spread
        # rescale once so the realised coverage of the populated pack equals the
        # requested concentration, then relax away any residual overlap: an
        # initial overlap would inject the placement's elastic energy (megajoules
        # per contact under a strength-limited law) into the transit and corrupt
        # both the resistance and the ledger.
        packed_area = (x1 - x0) * (y1 - y0)
        target_area = concentration * packed_area
        radii = radii * math.sqrt(target_area / float((math.pi * radii ** 2).sum()))
        centres = _relax(centres, radii, generator)
        keep = ((centres[:, 0] > 0.0) & (centres[:, 0] < length)
                & (centres[:, 1] > 0.0) & (centres[:, 1] < width))
        centres = centres[keep]
        radii = radii[keep]
        radii = torch.clamp(radii, min=radius_range[0], max=radius_range[1])
        thicknesses = torch.full((centres.shape[0],), float(thickness), dtype=torch.float64)

        sim = cls(radii, thicknesses, hull=hull, speed=speed, law=law,
                  friction=friction, water=water, window_upstream=window_upstream,
                  window_downstream=window_downstream, pack_width=pack_width, dt=dt)
        sim.pos = centres.clone()
        sim.set_seed(seed)
        sim.omega = torch.zeros(centres.shape[0], dtype=torch.float64)
        sim.z = sim.equilibrium_heave(sim.thickness)
        sim.vz = torch.zeros(centres.shape[0], dtype=torch.float64)
        sim.ice_modulus = float(ice_modulus)
        sim.packed_area = (x1 - x0) * (y1 - y0)
        sim.radius_spread_realised = float(jitter)
        return sim

    def realised_concentration(self) -> float:
        """Floe area / active-window area (the conveyor's coverage)."""
        area = math.pi * float((self.radius ** 2).sum())
        window = getattr(self, "packed_area", self.window_upstream * self.pack_width)
        return area / window if window > 0 else 0.0

    def max_initial_overlap(self) -> float:
        if self.n < 2:
            return 0.0
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + 1e-30)
        overlap = (self.radius[:, None] + self.radius[None, :]) - dist
        iu, ju = torch.triu_indices(self.n, self.n, offset=1)
        return float(torch.clamp(overlap[iu, ju].max(), min=0.0))

    # -- forces -------------------------------------------------------------
    def _drag(self) -> tuple[torch.Tensor, float]:
        draught = torch.clamp(submerged_draught(self.z, self.thickness), min=0.0)
        area = 2.0 * self.radius * draught
        speed = torch.linalg.norm(self.vel, dim=1, keepdim=True)
        coefficient = 0.5 * self.water.water_density * self.water.water_drag_coefficient
        force = -coefficient * area[:, None] * speed * self.vel
        power = float((coefficient * area * speed.squeeze(1) ** 3).sum())
        return force, power

    def heave_potential(self, z: torch.Tensor) -> torch.Tensor:
        """Heave potential energy [J] referenced to each floe's own equilibrium.

        ``U(z) = m g z - int F_b dz`` with the Archimedes force of a flat disk,
        ``F_b = rho_w g A clamp(t/2 - z, 0, t)``, integrated piecewise and shifted
        by its value at the Archimedes draught.  The shift makes the injected /
        removed heave energy of a recycled floe vanish at equilibrium (otherwise
        the datum of a floe that leaves the window biases the ledger) and leaves
        every *difference* of the store unchanged.
        """
        t = self.thickness
        k = self.water.water_density * self.water.gravity * self.waterplane_area
        half = 0.5 * t

        def _raw(level: torch.Tensor) -> torch.Tensor:
            deep = k * t * level
            mid = k * (0.5 * t * level - 0.5 * level ** 2) + k * t ** 2 / 8.0
            shallow = k * t ** 2 / 4.0
            integral = torch.where(level <= -half, deep,
                                   torch.where(level >= half, shallow, mid))
            return self.mass * self.water.gravity * level - integral

        return _raw(z) - _raw(self.equilibrium_heave(t))

    def _heave(self) -> tuple[torch.Tensor, float, torch.Tensor]:
        draught = torch.clamp(0.5 * self.thickness - self.z,
                              min=torch.zeros_like(self.z), max=self.thickness)
        upward = self.water.water_density * self.water.gravity \
            * self.waterplane_area * draught
        vertical = upward - self.mass * self.water.gravity
        k_heave = self.water.water_density * self.water.gravity * self.waterplane_area
        c_heave = 2.0 * self.heave_damping_ratio * torch.sqrt(
            torch.clamp(self.mass * k_heave, min=0.0))
        vertical = vertical - c_heave * self.vz
        power = float((c_heave * self.vz ** 2).sum())
        return vertical, power, self.heave_potential(self.z)

    def stored_energy(self) -> dict[str, float]:
        """Kinetic, heave, contact and pack-wall potential energies [J]."""
        ke = float((0.5 * self.mass * (self.vel ** 2).sum(dim=1)).sum()
                   + (0.5 * self.inertia_yaw * self.omega ** 2).sum())
        _, _, pe_heave = self._heave()
        ff = floe_floe_contacts(self.pos, self.vel, self.omega, self.radius,
                                self.thickness, self.mass, self.law, self.friction,
                                self.normal_damping_ratio, self.tangential_damping_ratio)
        signed, normal = self.hull_distance()
        hc = hull_contacts(self.pos, self.vel, self.omega, self.radius,
                           self.thickness, self.mass, self.law, self.friction,
                           self.normal_damping_ratio, self.tangential_damping_ratio,
                           signed, normal, self.hull_velocity())
        wc = wall_contacts(self.pos, self.vel, self.radius, self.pack_width,
                          self.wall_stiffness, self.wall_damping_ratio, self.mass)
        return {"kinetic": ke, "heave": float(pe_heave.sum()),
                "contact": float(ff["potential"]) + float(hc["potential"]),
                "pack_wall": float(wc["potential"])}

    # -- one step -----------------------------------------------------------
    def step(self, *, ledger: bool = True) -> dict[str, float]:
        """Advance one semi-implicit Euler step and accumulate the ledger."""
        dt = self.dt
        x_stem = self.x_stem()
        signed, normal = self.hull_distance(x_stem)
        hc = hull_contacts(self.pos, self.vel, self.omega, self.radius,
                           self.thickness, self.mass, self.law, self.friction,
                           self.normal_damping_ratio, self.tangential_damping_ratio,
                           signed, normal, self.hull_velocity())
        ff = floe_floe_contacts(self.pos, self.vel, self.omega, self.radius,
                                self.thickness, self.mass, self.law, self.friction,
                                self.normal_damping_ratio, self.tangential_damping_ratio)
        wc = wall_contacts(self.pos, self.vel, self.radius, self.pack_width,
                           self.wall_stiffness, self.wall_damping_ratio, self.mass)
        drag, drag_power = self._drag()
        vertical, heave_power, _ = self._heave()

        force = hc["force"] + ff["force"] + wc["force"] + drag
        torque = hc["torque"] + ff["torque"] + wc["torque"]

        # ledger: work in, thrust work, dissipation
        hull_force = hc["force"]
        resistance = float(hull_force[:, 0].sum())
        # Ledger input = *thrust work* R.V (the energy the kinematic hull puts into
        # the ice), not the floe-side work: with a moving boundary the two differ
        # by the relative-frame work that the contacts dissipate, and only the
        # thrust form closes the balance (the floe-side work is reported
        # separately as ``hull_input_kJ`` and their difference as the slip work).
        w_rim_rate = float((hull_force * self.vel).sum()
                           + (hc["torque"] * self.omega).sum())
        self.w_in += w_rim_rate * dt
        self.w_thrust += resistance * self.speed * dt
        channels = {
            "water_drag": drag_power,
            "hull_normal_damping": float(hc["p_normal_damp"]),
            "hull_sliding": float(hc["p_tangential"]),
            "floe_normal_damping": float(ff["p_normal_damp"]),
            "floe_sliding": float(ff["p_tangential"]),
            "pack_wall": float(wc["p_dissipation"]),
            "heave_damping": heave_power,
        }
        # Exact kinetic-energy increment of the semi-implicit update
        #     dKE = F.v_old dt + (1/2) |F|^2/m dt^2   (+ rotational, + heave)
        # The second term is numerical: it is booked explicitly as
        # ``integrator_injection`` so that the ledger is an identity rather than
        # a hope, and its share of the thrust work is reported with every case.
        injection = (0.5 * (force ** 2).sum(dim=1) / self.mass).sum() \
            + (0.5 * torque ** 2 / self.inertia_yaw).sum() \
            + (0.5 * vertical ** 2 / self.mass).sum()
        self.w_numerical += float(injection) * dt * dt
        step_diss = sum(channels.values())
        self.w_diss += step_diss * dt
        for key, value in channels.items():
            self.diss_channels[key] += value * dt

        # semi-implicit Euler
        self.vel = self.vel + dt * force / self.mass[:, None]
        self.omega = self.omega + dt * torque / self.inertia_yaw
        self.pos = self.pos + dt * self.vel
        self.vz = self.vz + dt * vertical / self.mass
        self.z = self.z + dt * self.vz
        self.time += dt
        self.steps += 1

        recycled = self._recycle()
        return {"resistance_N": resistance, "contacts": float(hc["count"]),
                "floe_contacts": float(ff["count"]), "recycled": float(recycled)}

    def _recycle(self, attempts: int = 40, clearance: float = 0.05) -> int:
        """Move floes that fell behind the stern back ahead of the stem.

        A recycled floe must land in a *free* spot: re-injecting it on top of
        another floe would create the elastic energy of that overlap from nothing
        (megajoules per contact under a strength-limited law) and destroy the
        energy ledger.  Candidates are drawn uniformly in the upstream half of the
        conveyor and accepted only if they clear every other floe; a floe that
        finds no free spot stays where it is and is retried on the next step.
        """
        x_stem = self.x_stem()
        behind = self.pos[:, 0] < (x_stem - self.window_downstream)
        index = behind.nonzero(as_tuple=False).squeeze(1)
        if index.numel() == 0:
            return 0
        kept = torch.ones(self.n, dtype=torch.bool)
        kept[index] = False
        moved = 0
        for slot in index.tolist():
            radius = float(self.radius[slot])
            x_lo = x_stem + 0.55 * self.window_upstream + radius
            x_hi = x_stem + self.window_upstream
            y_lo = radius + 1e-3
            y_hi = self.pack_width - radius - 1e-3
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            others = self.pos[kept]
            others_r = self.radius[kept]
            placed = None
            for _ in range(attempts):
                candidate = torch.tensor(
                    [x_lo + (x_hi - x_lo) * float(torch.rand(
                        (), generator=self._generator, dtype=self.dtype)),
                     y_lo + (y_hi - y_lo) * float(torch.rand(
                         (), generator=self._generator, dtype=self.dtype))],
                    dtype=self.dtype)
                gap = torch.linalg.norm(others - candidate, dim=1) \
                    - (others_r + radius + clearance)
                if others.shape[0] == 0 or float(gap.min()) > 0.0:
                    placed = candidate
                    break
            if placed is None:
                continue
            ke = float(0.5 * self.mass[slot] * (self.vel[slot] ** 2).sum()
                       + 0.5 * self.inertia_yaw[slot] * self.omega[slot] ** 2)
            _, _, pe = self._heave()
            self.w_removed += float(ke + pe[slot])
            self.pos[slot] = placed
            self.vel[slot] = 0.0
            self.omega[slot] = 0.0
            self.vz[slot] = 0.0
            self.z[slot] = self.equilibrium_heave(self.thickness[slot])
            kept[slot] = True
            moved += 1
        self.recycle_count += moved
        return moved

    # -- run ----------------------------------------------------------------
    def run(self, n_steps: int, *, record_every: int = 20,
            steady_start_s: float = 0.0) -> TransitResult:
        started = time.time()
        e0 = self.stored_energy()
        self.initial_energy = sum(e0.values())
        records: list[dict[str, float]] = []
        route_start = self.x_stem()
        for step in range(n_steps):
            info = self.step()
            if step % record_every == 0:
                records.append({
                    "time_s": self.time,
                    "resistance_kN": info["resistance_N"] / 1e3,
                    "contacts": info["contacts"],
                    "floe_contacts": info["floe_contacts"],
                    "mean_speed_m_s": float(torch.linalg.norm(self.vel, dim=1).mean()),
                    "route_m": self.x_stem() - route_start,
                    "recycled": float(self.recycle_count),
                })
        e1 = self.stored_energy()
        route = self.x_stem() - route_start

        times = torch.tensor([r["time_s"] for r in records], dtype=self.dtype)
        forces = torch.tensor([r["resistance_kN"] for r in records], dtype=self.dtype)
        window = times >= float(steady_start_s)
        if int(window.sum()) < 8:  # fall back to the last half of the run
            window = times >= 0.5 * float(times[-1])
        sample = forces[window]
        sample_t = times[window]
        mean = float(sample.mean())
        std = float(sample.std(unbiased=True)) if sample.numel() > 1 else 0.0
        sem = std / math.sqrt(max(sample.numel(), 1))
        slope = 0.0
        if sample.numel() > 2:
            tm = sample_t - sample_t.mean()
            slope = float((tm * (sample - sample.mean())).sum() / (tm ** 2).sum())
        drift = slope * float(sample_t[-1] - sample_t[0]) / max(abs(mean), 1e-12)

        stored_change = {k: e1[k] - e0[k] for k in e0}
        total_change = sum(stored_change.values())
        residual = (self.w_thrust + self.w_injected + self.w_numerical
                    - self.w_removed - self.w_diss) - total_change
        scale = max(abs(self.w_thrust), 1.0)
        return TransitResult(
            spec={}, hull=self.hull.as_dict(),
            mean_kN=mean, std_kN=std, sem_kN=sem,
            peak_kN=float(sample.max()) if sample.numel() else 0.0,
            min_kN=float(sample.min()) if sample.numel() else 0.0,
            stationarity_drift=drift,
            contact_count_mean=float(sum(r["contacts"] for r in records) / max(len(records), 1)),
            floe_count=self.n,
            realised_concentration=self.realised_concentration(),
            simulated_time_s=self.time,
            steady_window_s=(float(sample_t[0]) if sample_t.numel() else 0.0,
                             float(sample_t[-1]) if sample_t.numel() else 0.0),
            route_length_m=route,
            steps=self.steps,
            wall_time_s=time.time() - started,
            hull_input_kJ=self.w_in / 1e3,
            thrust_work_kJ=self.w_thrust / 1e3,
            slip_work_kJ=(self.w_thrust - self.w_in) / 1e3,
            dissipation_kJ={k: v / 1e3 for k, v in self.diss_channels.items()},
            energy_stored_change_kJ={k: v / 1e3 for k, v in stored_change.items()},
            injected_kJ=self.w_injected / 1e3,
            numerical_injection_kJ=self.w_numerical / 1e3,
            numerical_injection_share=(self.w_numerical / max(abs(self.w_thrust), 1.0)),
            removed_kJ=self.w_removed / 1e3,
            ledger_residual_kJ=residual / 1e3,
            ledger_residual_rel=abs(residual) / scale,
            series=records,
        )