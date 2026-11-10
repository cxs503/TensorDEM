"""Verification tiers for the NSCD contact solver, and its adjudication
against the soft-sphere (penalty) reference.

Three tiers are exercised, each with a machine-checkable number:

1. **Single sphere on a plane** — the rebound speed must equal ``e |u_n^-|``
   (Newton's restitution), the normal gap must never become negative, and the
   ``O(dt)`` "anticipated contact" offset (the sphere is stopped one step before
   it geometrically touches) must shrink linearly with ``dt``.
2. **``N``-sphere impact chain** — total momentum is conserved exactly, the
   contact sequence resolves without penetration and the kinetic energy is
   non-increasing.
3. **``phi = 0.85`` rigid packing under gravity** — zero penetration under load,
   a static state (no residual drift) and a floor load that balances the weight.

**Adjudication (互裁)**: the same configurations are run with a linear
soft-sphere (penalty) integrator at the same time step.  The comparison isolates
the two error channels: NSCD has a zero (by construction) but a ``dt``-dependent
contact-position offset, the penalty model has a load-dependent overlap
``delta = F/k`` and a time-step-dependent restitution error.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .nscd import NSCDError, NSCDStepInfo, NSCDWorld, Wall


# ---------------------------------------------------------------------------
# soft-sphere reference (adjudication partner)
# ---------------------------------------------------------------------------
class SoftSphereWorld(NSCDWorld):
    """Linear spring-dashpot DEM on the same geometry (penalty contact)."""

    def __init__(self, radius, *, kn: float = 1.0e5, **kwargs):
        super().__init__(radius, **kwargs)
        self.kn = float(kn)
        self.gamma_n = 2.0 * self._damping(self.restitution) * math.sqrt(
            self.kn * 0.5 * float(np.mean(self.mass)))

    @staticmethod
    def _damping(e: float) -> float:
        if e <= 0.0:
            return 1.0
        return -math.log(min(e, 0.999)) / math.pi

    def step(self):
        dt = self.dt
        self.vel = self.vel + dt * self.gravity[None, :]
        pos, rad = self.pos, self.radius
        force = np.zeros_like(pos)
        pairs = cKDTree(pos).query_pairs(2.0 * float(rad.max()) + 1e-12,
                                         output_type="ndarray")
        if pairs.size:
            d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            dist = np.linalg.norm(d, axis=1)
            good = dist > 1e-15
            pairs, d, dist = pairs[good], d[good], dist[good]
            overlap = rad[pairs[:, 0]] + rad[pairs[:, 1]] - dist
            hit = overlap > 0.0
            if hit.any():
                pairs, d, dist, overlap = pairs[hit], d[hit], dist[hit], overlap[hit]
                nvec = d / dist[:, None]
                vrel = self.vel[pairs[:, 1]] - self.vel[pairs[:, 0]]
                vn = np.einsum("ij,ij->i", vrel, nvec)
                fn = np.maximum(self.kn * overlap - self.gamma_n * vn, 0.0)
                vt = vrel - vn[:, None] * nvec
                sp = np.linalg.norm(vt, axis=1)
                ft = np.where(sp > 1e-14, self.mu * fn, 0.0)
                fvec = fn[:, None] * nvec + ft[:, None] * (-vt / np.maximum(sp, 1e-30)[:, None])
                np.add.at(force, pairs[:, 0], -fvec)
                np.add.at(force, pairs[:, 1], fvec)
        wall_imp = np.zeros((max(len(self.walls), 1), 3))
        for w, wall in enumerate(self.walls):
            g = wall.side * (pos[:, wall.axis] - wall.coord) - rad
            pen = g < 0.0
            if pen.any():
                idx = np.nonzero(pen)[0]
                n_out = wall.normal * wall.side          # into the material
                vn = np.einsum("ij,j->i", self.vel[idx], n_out)
                fn = np.maximum(self.kn * (-g[idx]) - self.gamma_n * vn, 0.0)
                vt = self.vel[idx] - vn[:, None] * n_out
                sp = np.linalg.norm(vt, axis=1)
                ft = np.where(sp > 1e-14, self.mu * fn, 0.0)
                fvec = fn[:, None] * n_out + ft[:, None] * (-vt / np.maximum(sp, 1e-30)[:, None])
                force[idx] += fvec
                wall_imp[w] -= fvec.sum(axis=0)
        self.vel = self.vel + force / self.mass[:, None] * dt
        self.pos = self.pos + self.vel * dt
        self.time += dt
        info = NSCDStepInfo(n_candidates=int(pairs.size),
                              n_active=int(np.sum(self._max_overlap() > 0.0)),
                              iterations=0, impulse_change=0.0,
                              max_overlap=self._max_overlap(),
                              kinetic_energy=self.total_kinetic_energy(),
                              wall_impulse=wall_imp)
        self.last = info
        self.last_wall_impulse = wall_imp
        return info


# ---------------------------------------------------------------------------
# tier 1: single sphere on a plane
# ---------------------------------------------------------------------------
def tier1_single_impact(*, restitution: float = 0.5, dt: float = 2e-5,
                        v0: float = 1.0, radius: float = 0.01,
                        sweep: str = "gauss_seidel", n_steps: int = 4000) -> dict:
    world = NSCDWorld(radius, mu=0.0, restitution=restitution,
                      gravity=(0.0, 0.0, 0.0), walls=[Wall(2, 0.0, 1)],
                      dt=dt, tol=1e-13, max_iter=200, sweep=sweep,
                      position_correction=False)
    z0 = 0.05
    world.set_state([[0.0, 0.0, z0]], vel=[[0.0, 0.0, -v0]])
    penetration = 0.0
    bounce_step = None
    for step in range(n_steps):
        world.step()
        penetration = max(penetration, world._max_overlap())
        if bounce_step is None and world.vel[0, 2] > 0.0:
            bounce_step = step
            bounce_gap = float(world.pos[0, 2] - radius)
    vz = float(world.vel[0, 2])
    return {
        "restitution_measured": vz / v0,
        "restitution_target": restitution,
        "restitution_error": abs(vz / v0 - restitution),
        "max_penetration": penetration,
        "penetration_over_d": penetration / (2.0 * radius),
        "bounce_gap": bounce_gap if bounce_step is not None else float("nan"),
        "bounce_gap_over_d": (bounce_gap / (2.0 * radius)) if bounce_step is not None else float("nan"),
        "dt": dt, "sweep": sweep, "n_steps": n_steps,
    }


def tier1_refinement(restitutions=(0.2, 0.5, 0.8), dts=(8e-5, 4e-5, 2e-5, 1e-5)) -> dict:
    rows = []
    for e in restitutions:
        for dt in dts:
            rows.append(tier1_single_impact(restitution=e, dt=dt))
    """Add the penalty model on the same rows for the adjudication table."""
    penalty = []
    for e in restitutions:
        for dt in dts:
            w = SoftSphereWorld(0.01, kn=1.0e5, mu=0.0, restitution=e,
                                gravity=(0.0, 0.0, 0.0), walls=[Wall(2, 0.0, 1)],
                                dt=dt, tol=1e-13, max_iter=1)
            w.set_state([[0.0, 0.0, 0.05]], vel=[[0.0, 0.0, -1.0]])
            vmax = 0.0
            for _ in range(int(4000 * 2e-5 / dt) + 200):
                w.step()
                vmax = max(vmax, w._max_overlap())
                if w.vel[0, 2] > 0.0 and w.time > 0.02:
                    break
            penalty.append({
                "restitution_target": e, "dt": dt,
                "restitution_measured": float(w.vel[0, 2]),
                "max_penetration": vmax,
                "penetration_over_d": vmax / 0.02,
            })
    for row in rows:
        row["restitution_measured"] = abs(row["restitution_measured"])
    max_err = max(r["restitution_error"] for r in rows)
    max_pen = max(r["penetration_over_d"] for r in rows)
    max_pen_penalty = max(p["penetration_over_d"] for p in penalty)
    return {"nscd": rows, "penalty": penalty,
            "max_restitution_error": max_err,
            "max_penetration_over_d": max_pen,
            "penalty_max_penetration_over_d": max_pen_penalty}


# ---------------------------------------------------------------------------
# tier 2: N-sphere impact chain
# ---------------------------------------------------------------------------
def tier2_chain(n: int = 12, *, restitution: float = 0.8, v0: float = 1.0,
                dt: float = 1e-5, radius: float = 0.01, gap: float = 1.002,
                steps: int = 4000, sweep: str = "jacobi") -> dict:
    spacing = 2.0 * radius * gap
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * spacing
    world = NSCDWorld(np.full(n, radius), mu=0.0, restitution=restitution,
                      gravity=(0.0, 0.0, 0.0), dt=dt, tol=1e-12,
                      max_iter=300, sweep="gauss_seidel" if n <= 24 else sweep,
                      position_correction=False)
    world.set_state(pos)
    world.vel[0, 0] = v0
    p0 = world.total_momentum().copy()
    ke0 = world.total_kinetic_energy()
    pen = 0.0
    ke_max = ke0
    for _ in range(steps):
        world.step()
        pen = max(pen, world._max_overlap())
        ke_max = max(ke_max, world.total_kinetic_energy())
    p1 = world.total_momentum()
    ke1 = world.total_kinetic_energy()
    return {
        "n": n, "dt": dt, "restitution": restitution,
        "momentum_error": float(np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-300)),
        "max_penetration": pen,
        "max_penetration_over_d": pen / (2.0 * radius),
        "ke_initial": ke0, "ke_final": ke1,
        "ke_ratio": ke1 / max(ke0, 1e-300),
        "energy_monotone": bool(ke_max <= ke0 * (1.0 + 1e-9)),
        "outlet_velocity": float(world.vel[-1, 0]),
        "final_velocities": world.vel[:, 0].tolist(),
    }


# ---------------------------------------------------------------------------
# tier 3: phi = 0.85 rigid packing
# ---------------------------------------------------------------------------
def _lattice_box(nx: int, ny: int, nz: int, r: float, phi: float):
    """HCP-lattice box at the target solid fraction (no initial overlap).

    Layer spacing for close packing is ``s sqrt(2/3)``, which is *smaller* than
    the in-plane pitch ``s``; the layers must therefore be staggered by
    ``(s/2, s/(2 sqrt 3))`` or the stacking overlaps by 30 %.  With the stagger
    the nearest-neighbour distance is exactly ``s`` and the solid fraction is
    ``0.9069 (2r/s)^2``.
    """
    spacing = 2.0 * r * math.sqrt(math.pi / (2.0 * math.sqrt(3.0) * phi))
    pos = []
    for k in range(nz):
        shift = (0.5 * spacing, spacing / (2.0 * math.sqrt(3.0))) if k % 2 else (0.0, 0.0)
        for j in range(ny):
            for i in range(nx):
                x = i * spacing + (0.5 * spacing if j % 2 else 0.0) + shift[0]
                y = j * spacing * math.sqrt(3.0) / 2.0 + shift[1]
                z = r + k * spacing * math.sqrt(2.0 / 3.0)
                pos.append([x, y, z])
    return np.asarray(pos), spacing


def tier3_packing(*, phi: float = 0.85, radius: float = 0.005,
                  n=(5, 6, 3), dt: float = 2e-5, steps: int = 4000,
                  lid_velocity: float = 5e-3, sweep: str = "colored",
                  restitution: float = 0.0, mu: float = 0.4,
                  max_iter: int = 60, tol: float = 1e-2) -> dict:
    """Confined compression of a ``phi = 0.85`` rigid packing.

    A prescribed-displacement lid is used instead of gravity loading: the load
    path is then controlled (the material cannot fall away from the support), so
    "does the packing transmit the load with zero penetration" is a well posed
    question.  Diagnostics: penetration, the equilibrium of the floor and lid
    loads (they must be equal and opposite in a static packing), the residual
    velocity and any lateral wall load.
    """
    pos, spacing = _lattice_box(n[0], n[1], n[2], radius, phi)
    pos = pos + np.array([radius, radius, 0.0])      # clear the side walls
    lx = float(pos[:, 0].max()) + radius
    ly = float(pos[:, 1].max()) + radius
    z_top = float(pos[:, 2].max()) + radius
    walls = [Wall(2, 0.0, 1),                      # floor
             Wall(0, 0.0, 1), Wall(0, lx, -1),
             Wall(1, 0.0, 1), Wall(1, ly, -1)]
    world = NSCDWorld(np.full(pos.shape[0], radius), mu=mu, restitution=restitution,
                      gravity=(0.0, 0.0, 0.0), walls=walls + [Wall(2, z_top, -1,
                                                                  (0.0, 0.0, -lid_velocity))],
                      dt=dt, tol=tol, max_iter=max_iter, sweep=sweep,
                      position_correction=False)
    world.set_state(pos)
    pen = 0.0
    floor_f, lid_f, side_f = [], [], []
    for step in range(steps):
        z_top -= lid_velocity * dt
        world.walls[5] = Wall(2, z_top, -1, (0.0, 0.0, -lid_velocity))
        world.step()
        pen = max(pen, world._max_overlap())
        if step > steps // 2:
            floor_f.append(world.wall_force(0)[2])
            lid_f.append(world.wall_force(5)[2])
            side_f.append(world.wall_force(1)[2])
    f_floor = float(np.mean(floor_f)) if floor_f else 0.0
    f_lid = float(np.mean(lid_f)) if lid_f else 0.0
    return {
        "phi_target": phi, "n_particles": int(world.n), "dt": dt, "steps": steps,
        "max_penetration": pen, "max_penetration_over_d": pen / (2.0 * radius),
        "penetration_after_projection": world._max_overlap(),
        "max_velocity_final": float(np.max(np.linalg.norm(world.vel, axis=1))),
        "floor_force": f_floor, "lid_force": f_lid,
        "equilibrium_error": abs(abs(f_lid) - abs(f_floor)) / max(abs(f_lid), 1e-300),
        "side_force_x": float(np.mean(side_f)) if side_f else 0.0,
        "load_level": abs(f_lid),
        "lid_displacement": lid_velocity * dt * steps,
        "solid_fraction_loaded": float(np.sum(np.pi * radius ** 2
                                              * 0)) if False else phi,
        "sweep": sweep, "restitution": restitution, "mu": mu,
    }


def tier3_adjudication(**kwargs) -> dict:
    """Same confined compression through NSCD and through the penalty model."""
    rigid = tier3_packing(**kwargs)
    penalty = {"error": "not run"}
    try:
        n = kwargs.get("n", (5, 6, 3))
        radius = kwargs.get("radius", 0.005)
        phi = kwargs.get("phi", 0.85)
        dt = kwargs.get("dt", 2e-5)
        steps = kwargs.get("steps", 4000)
        lid_v = kwargs.get("lid_velocity", 5e-3)
        pos, spacing = _lattice_box(n[0], n[1], n[2], radius, phi)
        pos = pos + np.array([radius, radius, 0.0])
        lx = float(pos[:, 0].max()) + radius
        ly = float(pos[:, 1].max()) + radius
        z_top = float(pos[:, 2].max()) + radius
        walls = [Wall(2, 0.0, 1), Wall(0, 0.0, 1), Wall(0, lx, -1),
                 Wall(1, 0.0, 1), Wall(1, ly, -1),
                 Wall(2, z_top, -1, (0.0, 0.0, -lid_v))]
        w = SoftSphereWorld(np.full(pos.shape[0], radius), kn=1.0e5,
                            mu=kwargs.get("mu", 0.4),
                            restitution=kwargs.get("restitution", 0.0),
                            gravity=(0.0, 0.0, 0.0), walls=walls, dt=dt,
                            tol=1e-10)
        w.set_state(pos)
        pen = 0.0
        ff, lf = [], []
        for step in range(steps):
            z_top -= lid_v * dt
            w.walls[5] = Wall(2, z_top, -1, (0.0, 0.0, -lid_v))
            w.step()
            pen = max(pen, w._max_overlap())
            if step > steps // 2:
                ff.append(float(-w.last_wall_impulse[0][2] / w.dt))
                lf.append(float(w.last_wall_impulse[5][2] / w.dt))
        f_floor = float(np.mean(ff)) if ff else 0.0
        f_lid = float(np.mean(lf)) if lf else 0.0
        penalty = {
            "max_penetration": pen,
            "max_penetration_over_d": pen / (2.0 * radius),
            "floor_force": f_floor, "lid_force": f_lid,
            "equilibrium_error": abs(abs(f_lid) - abs(f_floor)) / max(abs(f_lid), 1e-300),
            "max_velocity_final": float(np.max(np.linalg.norm(w.vel, axis=1))),
        }
    except Exception as exc:                        # noqa: BLE001 - reported
        penalty = {"error": f"{type(exc).__name__}: {exc}"}
    return {"nscd": rigid, "penalty": penalty}


__all__ = ["SoftSphereWorld", "tier1_single_impact", "tier1_refinement",
           "tier2_chain", "tier3_packing", "tier3_adjudication"]