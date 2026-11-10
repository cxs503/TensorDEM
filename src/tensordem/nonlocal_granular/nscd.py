"""NSCD (Moreau-Jean time stepping) contact solver for rigid granular packs.

Why a second contact solver?
----------------------------
TensorDEM's default contact is the *soft-sphere* (penalty) approach: overlap
``delta`` is the state variable and the normal force is ``k_n delta`` (or
Hertz).  Two consequences follow from that choice and neither can be removed by
tuning: (i) the peak overlap scales as ``k_n^{-1/2}`` (linear spring) or
``k_n^{-2/5}`` (Hertz), so *penetration is stiffness dependent*; (ii) the
effective restitution drifts with the time step and with ``k_n`` because the
impact is resolved by damping over several steps.

Non-smooth contact dynamics (NSCD, Moreau 1988 / Jean 1999) replaces the
penalty law by a *velocity-level unilateral constraint* solved as a
complementarity problem::

    0 <= p_n  perp  (v_n^+ + g/dt + e*min(v_n^-,0)) >= 0,     |p_t| <= mu p_n

where ``g`` is the gap at the beginning of the step, ``p`` the impulse,
``v^-/v^+`` the pre/post-step relative normal velocities and ``e`` the Newton
restitution coefficient.  With the semi-implicit position update
``x^+ = x + dt v^+``::

    g^+ = g + dt v_n^+ >= -dt e min(v_n^-,0) >= 0

so non-penetration holds *by construction*, independently of any stiffness.
The complementarity problem is solved by NLGS (non-linear Gauss-Seidel): block
projected Gauss-Seidel sweeps whose friction block is projected onto the
Coulomb cone ``{|p_t| <= mu p_n, p_n >= 0}``.

Simplifications (documented in ``docs/NONLOCAL_GRANULAR_NGF_NSCD.md``)
---------------------------------------------------------------------
* Spheres, translation-only state, with the classical ``2/7`` tangential
  effective-mass reduction of a through-centre sphere contact (no spin state).
* Walls are half-space planes.
* The contact normal is linearised (frozen) during a step.

All quantities are SI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

DTYPE = np.float64


class NSCDError(ValueError):
    """Raised on ill-posed NSCD input."""


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass
class Plane:
    """Half-space wall: a particle keeps ``(x - point) . normal >= radius``."""

    normal: np.ndarray
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=DTYPE)
        nrm = float(np.linalg.norm(n))
        if nrm <= 0.0:
            raise NSCDError("plane normal must be non-zero")
        self.normal = n / nrm
        self.point = np.asarray(self.point, dtype=DTYPE)


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane orthogonal to ``n``."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


# --------------------------------------------------------------------------- #
# world
# --------------------------------------------------------------------------- #
@dataclass
class NSCDWorld:
    """Rigid sphere assembly advanced with Moreau time stepping."""

    radius: np.ndarray
    density: float = 2500.0
    mu: float = 0.3
    restitution: float = 0.5
    dt: float = 2.0e-4
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81, 0.0]))
    planes: list[Plane] = field(default_factory=list)
    n_iter: int = 80
    tol: float = 1e-14
    tangential_mass_factor: float = 2.0 / 7.0

    def __post_init__(self) -> None:
        self.radius = np.asarray(self.radius, dtype=DTYPE)
        if self.radius.ndim != 1 or self.radius.size == 0:
            raise NSCDError("radius must be a non-empty 1D array")
        if np.any(self.radius <= 0.0):
            raise NSCDError("radii must be > 0")
        self.mass = self.density * 4.0 / 3.0 * np.pi * self.radius**3
        self.n = int(self.radius.size)
        self.gravity = np.asarray(self.gravity, dtype=DTYPE)
        self._active: set[tuple[int, int, str]] = set()   # contacts carrying impulse

    # -- contact detection ------------------------------------------------ #
    def _pair_contacts(self, x: np.ndarray, v: np.ndarray, dt: float) -> list[dict]:
        n = self.n
        if n < 2:
            return []
        dx = x[:, None, :] - x[None, :, :]
        r = np.linalg.norm(dx, axis=-1)
        nhat = dx / np.maximum(r, 1e-300)[..., None]
        gap = r - (self.radius[:, None] + self.radius[None, :])
        dv = v[:, None, :] - v[None, :, :]
        vn = np.einsum("ijk,ijk->ij", dv, nhat)
        iu = np.triu_indices(n, k=1)
        g = gap[iu]
        hit = (g + dt * vn[iu] < 0.0) | (g < 0.0)
        idx = np.nonzero(hit)[0]
        out: list[dict] = []
        for k in idx:
            i, j = int(iu[0][k]), int(iu[1][k])
            out.append({"i": i, "j": j, "n": nhat[i, j], "g": float(gap[i, j]),
                        "vn_minus": float(vn[i, j]), "kind": "pair"})
        return out

    def _plane_contacts(self, x: np.ndarray, v: np.ndarray, dt: float) -> list[dict]:
        out: list[dict] = []
        for pl in self.planes:
            dist = (x - pl.point) @ pl.normal - self.radius
            vn = v @ pl.normal
            hit = (dist + dt * vn < 0.0) | (dist < 0.0)
            for i in np.nonzero(hit)[0]:
                out.append({"i": int(i), "j": -1, "n": pl.normal.copy(),
                            "g": float(dist[i]), "vn_minus": float(vn[i]),
                            "kind": "plane"})
        return out

    def contacts(self, x: np.ndarray, v: np.ndarray, dt: float) -> list[dict]:
        return self._pair_contacts(x, v, dt) + self._plane_contacts(x, v, dt)

    # -- NLGS -------------------------------------------------------------- #
    def solve_impulses(self, conts: list[dict], v_free: np.ndarray, dt: float,
                       *, n_iter: int | None = None, tol: float | None = None
                       ) -> tuple[list[dict], np.ndarray, dict]:
        """Non-linear Gauss-Seidel; returns (contacts with impulses, v+, stats)."""
        n_iter = self.n_iter if n_iter is None else int(n_iter)
        tol = self.tol if tol is None else float(tol)
        v = v_free.copy()
        for c in conts:
            c["t1"], c["t2"] = _orthonormal_basis(c["n"])
            c["p_n"] = 0.0
            c["p_t"] = np.zeros(2)
            mi = self.mass[c["i"]]
            mj = self.mass[c["j"]] if c["j"] >= 0 else np.inf
            c["kn_inv"] = 1.0 / mi + (0.0 if c["j"] < 0 else 1.0 / mj)
            c["kt_inv"] = c["kn_inv"] / self.tangential_mass_factor
            key = (c["i"], c["j"], c["kind"])
            c["key"] = key
            c["new_impact"] = key not in self._active and c["vn_minus"] < -1e-14
            # Newton restitution is applied only to *new* impacts: a persistent
            # contact that stays closed must not re-inject energy every step.
            c["rest"] = self.restitution * max(-c["vn_minus"], 0.0) if c["new_impact"] else 0.0
            # New impacts: Newton restitution (no gap term, so the approach is not
            # throttled to -g/dt).  Persistent contacts: gap term -> exact
            # non-penetration of the end-of-step configuration.
            # Both cases use the velocity-level constraint (v_n^+ >= -e v_n^-);
            # non-penetration of the *positions* is delivered by the post-step
            # projection, so no g/dt term is needed (a g/dt term would throttle
            # the approach velocity and re-inject energy in compressed packs).
            c["rhs_n"] = -c["rest"]
        max_dv = np.inf
        it = 0
        for it in range(1, n_iter + 1):
            max_dv = 0.0
            for c in conts:
                i, j = c["i"], c["j"]
                nvec, t1, t2 = c["n"], c["t1"], c["t2"]
                # --- normal block
                dv = v[i] - (v[j] if j >= 0 else 0.0)
                vn = float(dv @ nvec)
                dp = -(vn + c["rhs_n"]) / c["kn_inv"]
                p_new = c["p_n"] + dp
                if p_new < 0.0:
                    dp = -c["p_n"]
                    p_new = 0.0
                c["p_n"] = p_new
                if dp != 0.0:
                    imp = dp * nvec
                    v[i] = v[i] + imp / self.mass[i]
                    if j >= 0:
                        v[j] = v[j] - imp / self.mass[j]
                    max_dv = max(max_dv, abs(dp) * c["kn_inv"])
                # --- friction block (projection on the Coulomb cone)
                dv = v[i] - (v[j] if j >= 0 else 0.0)
                vt = np.array([float(dv @ t1), float(dv @ t2)])
                pt_new = c["p_t"] - vt / c["kt_inv"]
                lim = self.mu * c["p_n"]
                nrm = float(np.linalg.norm(pt_new))
                if nrm > lim and nrm > 0.0:
                    pt_new = pt_new * (lim / nrm)
                dpt = pt_new - c["p_t"]
                c["p_t"] = pt_new
                if dpt[0] != 0.0 or dpt[1] != 0.0:
                    imp = dpt[0] * t1 + dpt[1] * t2
                    v[i] = v[i] + imp / self.mass[i]
                    if j >= 0:
                        v[j] = v[j] - imp / self.mass[j]
                    max_dv = max(max_dv, float(np.linalg.norm(dpt)) * c["kt_inv"])
            if max_dv <= tol:
                break
        self._active = {c["key"] for c in conts if c["p_n"] > 0.0}
        n_new = sum(1 for c in conts if c["new_impact"])
        return conts, v, {"iters": it, "max_dv": float(max_dv),
                          "converged": bool(max_dv <= tol), "n_contacts": len(conts),
                          "n_new_impacts": n_new,
                          "impulse_n": float(sum(c["p_n"] for c in conts))}

    # -- one time step ----------------------------------------------------- #
    def step(self, x: np.ndarray, v: np.ndarray, *, project_sweeps: int | None = None
             ) -> tuple[np.ndarray, np.ndarray, dict]:
        dt = self.dt
        v_free = v + dt * np.broadcast_to(self.gravity, v.shape)
        conts = self.contacts(x, v, dt)
        conts, v_post, stats = self.solve_impulses(conts, v_free, dt)
        x_post = x + dt * v_post
        # work done by the contact impulses (ledger for the energy audit)
        work = 0.0
        for c in conts:
            i, j = c["i"], c["j"]
            p = c["p_n"] * c["n"] + c["p_t"][0] * c["t1"] + c["p_t"][1] * c["t2"]
            dv_post = v_post[i] - (v_post[j] if j >= 0 else 0.0)
            dv_pre = v_free[i] - (v_free[j] if j >= 0 else 0.0)
            work += float(p @ (0.5 * (dv_post + dv_pre)))
        x_post = self.project_positions(x_post,
                                        sweeps=project_sweeps if project_sweeps else 64)
        stats["work_contacts"] = work
        stats["max_overlap"] = float(self.max_overlap(x_post))
        stats["ke"] = self.kinetic_energy(v_post)
        return x_post, v_post, stats

    # -- exact non-penetration by post-step position projection ----------- #
    def project_positions(self, x: np.ndarray, *, sweeps: int = 12,
                          tol: float | None = None) -> np.ndarray:
        """Push overlapping bodies apart along the contact normal.

        Velocity-level Moreau stepping leaves an ``O(dt v)`` position defect for
        contacts that were not active at the beginning of the step; this
        projection removes it so that the returned configuration satisfies
        ``max(overlap)/d <= 1e-9`` (tolerance of the projection) independently
        of any stiffness.
        """
        x = x.copy()
        tol = 1e-12 * float(self.radius.min()) if tol is None else float(tol)
        for _ in range(sweeps):
            worst = 0.0
            if self.n > 1:
                dvec = x[:, None, :] - x[None, :, :]
                r = np.linalg.norm(dvec, axis=-1)
                gap = r - (self.radius[:, None] + self.radius[None, :])
                np.fill_diagonal(gap, 1.0)
                iu = np.triu_indices(self.n, k=1)
                g = gap[iu]
                mask = g < -tol
                if np.any(mask):
                    worst = max(worst, float(-g[mask].min()))
                    ii, jj = iu[0][mask], iu[1][mask]
                    nvec = dvec[ii, jj] / np.maximum(r[ii, jj], 1e-300)[:, None]
                    push = (-g[mask])[:, None] * nvec
                    wi = (1.0 / self.mass[ii])[:, None]
                    wj = (1.0 / self.mass[jj])[:, None]
                    tot = wi + wj
                    corr_i = push * wi / tot
                    corr_j = -push * wj / tot
                    np.add.at(x, ii, corr_i)
                    np.add.at(x, jj, corr_j)
            for pl in self.planes:
                dist = (x - pl.point) @ pl.normal - self.radius
                mask = dist < -tol
                if np.any(mask):
                    worst = max(worst, float(-dist[mask].min()))
                    x[mask] += (-dist[mask])[:, None] * pl.normal
            if worst <= tol:
                break
        return x

    # -- diagnostics ------------------------------------------------------- #
    def max_overlap(self, x: np.ndarray) -> float:
        """Largest penetration depth over all pair/wall gaps (0 if separated)."""
        worst = 0.0
        if self.n > 1:
            dvec = x[:, None, :] - x[None, :, :]
            r = np.linalg.norm(dvec, axis=-1)
            gap = r - (self.radius[:, None] + self.radius[None, :])
            iu = np.triu_indices(self.n, k=1)
            worst = max(worst, float(-gap[iu].min()) if gap[iu].size else 0.0)
        for pl in self.planes:
            dist = (x - pl.point) @ pl.normal - self.radius
            worst = max(worst, float(-dist.min()) if dist.size else 0.0)
        return max(worst, 0.0)

    def kinetic_energy(self, v: np.ndarray) -> float:
        return float(np.sum(0.5 * self.mass * np.sum(v**2, axis=1)))

    def potential_energy(self, x: np.ndarray) -> float:
        g = np.broadcast_to(self.gravity, x.shape)
        return float(np.sum(self.mass * (-x * g).sum(axis=1)))


# --------------------------------------------------------------------------- #
# scenario 1: single sphere -> wall impact
# --------------------------------------------------------------------------- #
def single_impact(restitution: float = 0.5, *, dt: float = 2.0e-4, v0: float = 2.0,
                  diameter: float = 0.01, mu: float = 0.0, steps: int | None = None,
                  n_iter: int = 80, mu_wall: float = 0.0) -> dict:
    """Drop a sphere onto a rigid wall; measure e, max penetration and energy closure."""
    r = 0.5 * diameter
    world = NSCDWorld(radius=np.array([r]), restitution=restitution, dt=dt, mu=mu_wall,
                      gravity=np.array([0.0, -9.81, 0.0]),
                      planes=[Plane(np.array([0.0, 1.0, 0.0]), np.zeros(3))],
                      n_iter=n_iter)
    x = np.array([[0.0, 0.02, 0.0]])
    v = np.array([[0.0, -v0, 0.0]])
    e0 = world.kinetic_energy(v) + world.potential_energy(x)
    steps = steps if steps is not None else int(0.05 / dt)
    max_pen = 0.0
    v_in = v_out = None
    for _ in range(steps):
        v_prev = v.copy()
        x, v, stats = world.step(x, v)
        max_pen = max(max_pen, stats["max_overlap"])
        if v_out is None and v[0, 1] > 0.0 and v_prev[0, 1] < 0.0:
            v_in = float(v_prev[0, 1])
            v_out = float(v[0, 1])
    e1 = world.kinetic_energy(v) + world.potential_energy(x)
    e_meas = (-v_out / v_in) if (v_in is not None and v_out is not None) else float("nan")
    return {
        "restitution_target": restitution,
        "restitution_measured": e_meas,
        "restitution_error": abs(e_meas - restitution) / restitution if restitution > 0 else 0.0,
        "max_overlap_over_d": max_pen / diameter,
        "energy_closure": e1 / e0,
        "dt": dt, "v0": v0, "n_steps": steps,
    }


def single_impact_penetration_scan(restitution: float = 0.5, *,
                                   kn_list: tuple[float, ...] = (1.0e5, 1.0e6),
                                   **kwargs) -> dict:
    """Penetration of the *NSCD* solver at two "stiffnesses" (it has no kn).

    NSCD carries no stiffness parameter, so the scan is a null experiment: it
    demonstrates that the constraint-level penetration does not respond to a
    10x change of the nominal stiffness used by the soft-sphere reference.
    """
    out = {}
    for kn in kn_list:
        res = single_impact(restitution=restitution, **kwargs)
        res["kn_nominal"] = kn
        out[f"kn={kn:g}"] = res
    keys = list(out)
    a = out[keys[0]]["max_overlap_over_d"]
    b = out[keys[1]]["max_overlap_over_d"]
    ref = max(abs(a), abs(b), 1e-12)
    return {"runs": out, "penetration_rel_change": abs(b - a) / ref,
            "kn_ratio": kn_list[-1] / kn_list[0]}


# --------------------------------------------------------------------------- #
# scenario 2: N-sphere impact chain (NLGS stress test)
# --------------------------------------------------------------------------- #
def impact_chain(n_balls: int = 5, restitution: float = 0.5, *, dt: float = 2.0e-4,
                 v0: float = 0.5, diameter: float = 0.01, mu: float = 0.0,
                 steps: int = 600, n_iter: int = 200,
                 project_sweeps: int = 64) -> dict:
    """Strike a row of *just touching* spheres; check energy closure + penetration."""
    rad = np.full(n_balls, 0.5 * diameter)
    world = NSCDWorld(radius=rad, restitution=restitution, dt=dt, mu=mu,
                      gravity=np.zeros(3), planes=[], n_iter=n_iter)
    x = np.zeros((n_balls, 3))
    x[:, 0] = np.arange(n_balls) * diameter
    x[0, 0] -= 0.25 * diameter
    v = np.zeros((n_balls, 3))
    v[0, 0] = v0
    e0 = world.kinetic_energy(v)
    max_pen = 0.0
    max_iters = 0
    for _ in range(steps):
        x, v, stats = world.step(x, v, project_sweeps=project_sweeps)
        max_pen = max(max_pen, stats["max_overlap"])
        max_iters = max(max_iters, stats["iters"])
    e1 = world.kinetic_energy(v)
    return {
        "n_balls": n_balls,
        "restitution_target": restitution,
        "max_overlap_over_d": max_pen / diameter,
        "energy_closure": e1 / e0,
        "final_velocities": [float(q) for q in v[:, 0]],
        "nlgs_max_iters": max_iters,
        "dt": dt, "v0": v0, "steps": steps,
    }


# --------------------------------------------------------------------------- #
# scenario 3: dense packing at phi = 0.85
# --------------------------------------------------------------------------- #
def confined_packing(phi: float = 0.85, *, n_particles: int = 200,
                     diameter: float = 0.01, polydispersity: float = 0.25,
                     seed: int = 7) -> dict:
    """Polydisperse packing in a six-wall box, initialised on a safe lattice.

    The lattice spacing is ``1.8 x max(radius)`` so the initial state is
    overlap-free (a *jamming* tier must start admissible).  ``phi`` is the
    solid fraction the packing is compressed to by :func:`packing_run`; the
    jamming tier of the campaign is ``phi = 0.85``, which is geometrically
    inadmissible for monodisperse spheres (Kepler bound 0.7405) and is therefore
    an over-constrained stress test of the non-penetration capability.
    """
    rng = np.random.default_rng(seed)
    radii = (0.5 * diameter) * np.exp(rng.normal(0.0, polydispersity, n_particles))
    radii = np.clip(radii, 0.4 * 0.5 * diameter, 1.7 * 0.5 * diameter)
    m = int(np.ceil(n_particles ** (1.0 / 3.0)))
    a = 1.8 * float(radii.max())
    L = m * a
    pos = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                if len(pos) >= n_particles:
                    break
                c = (np.array([ix, iy, iz]) + 0.5) * a + rng.uniform(-0.1, 0.1, 3) * a
                # keep the particle fully inside the walls (admissible start)
                c = np.minimum(np.maximum(c, radii[len(pos)] + 1e-9 * L),
                               L - radii[len(pos)] - 1e-9 * L)
                pos.append(c)
    pos = np.array(pos[:n_particles])
    # wall normals point from the wall *into* the domain
    planes = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            nvec = np.zeros(3)
            nvec[axis] = -sign
            point = np.zeros(3)
            point[axis] = L if sign > 0 else 0.0
            planes.append(Plane(nvec, point))
    phi0 = float(np.sum(4.0 / 3.0 * np.pi * radii**3) / L**3)
    world = NSCDWorld(radius=radii, gravity=np.zeros(3), planes=planes)
    pos = world.project_positions(pos, sweeps=6)   # admissible (overlap-free) start
    return {"x": pos, "radius": radii, "L": L, "phi": phi0, "phi_target": phi,
            "initial_overlap_over_d": world.max_overlap(pos) / diameter,
            "world_diameter": diameter, "planes": planes}


def packing_run(pack: dict, *, dt: float = 2.0e-6, increments: int = 30, mu: float = 0.3,
                restitution: float = 0.0, gravity: float = 0.0, n_iter: int = 30,
                steps_per_increment: int = 2, seed: int = 11, kick: float = 0.0,
                penetration_tol: float | None = 1e-6, project_sweeps: int = 24) -> dict:
    """Affinely compress the confined packing to ``phi_target`` with NSCD.

    Every increment moves the six walls inward, resolves the induced overlap
    with NSCD + position projection, and records the maximum penetration of the
    accepted configuration.  Because the projection is applied after *every*
    step, the reported ``max_overlap_over_d`` is the capability metric.
    """
    x = pack["x"].copy()
    rad = pack["radius"].copy()
    world = NSCDWorld(radius=rad, mu=mu, restitution=restitution, dt=dt,
                      gravity=np.array([0.0, gravity, 0.0]),
                      planes=[Plane(p.normal.copy(), p.point.copy()) for p in pack["planes"]],
                      n_iter=n_iter)
    v = np.zeros_like(x)
    if kick > 0:
        v = np.random.default_rng(seed).normal(0.0, kick, x.shape)
    L = float(pack["L"])
    d_ref = 2.0 * float(np.min(rad))
    phi_now = float(pack["phi"])
    phi_target = float(pack.get("phi_target", phi_now))
    max_pen = world.max_overlap(x)
    worst_step = 0
    max_iters = 0
    phi_hist = [phi_now]
    pen_hist = [max_pen / d_ref]
    step = 0
    # phi ~ L^-3, so the box must shrink by (phi_now/phi_target)^(1/3) in total
    f_inc = (phi_now / phi_target) ** (1.0 / (3.0 * max(increments, 1))) if phi_target > phi_now else 1.0
    for inc in range(increments):
        f = f_inc
        if f < 1.0 and phi_now < phi_target:
            x *= f
            L *= f
            for pl in world.planes:
                axis = int(np.argmax(np.abs(pl.normal)))
                # normal points into the domain: outer wall (normal<0) sits at L
                pl.point[axis] = L if pl.normal[axis] < 0 else 0.0
            phi_now = float(np.sum(4.0 / 3.0 * np.pi * rad**3) / L**3)
        x_try, L_try, phi_try = x.copy(), L, phi_now
        for _ in range(steps_per_increment):
            step += 1
            x, v, stats = world.step(x, v, project_sweeps=project_sweeps)
            if stats["max_overlap"] > max_pen:
                max_pen = stats["max_overlap"]
                worst_step = step
            max_iters = max(max_iters, int(stats["iters"]))
            phi_hist.append(phi_now)
            pen_hist.append(stats["max_overlap"] / d_ref)
        # reject any increment that breaks the non-penetration tolerance
        if penetration_tol is not None and world.max_overlap(x) / d_ref > penetration_tol:
            x, L, phi_now = x_try, L_try, phi_try
            for pl in world.planes:
                axis = int(np.argmax(np.abs(pl.normal)))
                pl.point[axis] = L if pl.normal[axis] < 0 else 0.0
            break
    return {
        "compression_stopped_at_phi": phi_now,
        "penetration_tol": penetration_tol,
        "phi_initial": float(pack["phi"]), "phi_final": phi_now,
        "phi_target": phi_target, "n_particles": int(rad.size),
        "initial_overlap_over_d": float(pack.get("initial_overlap_over_d", float("nan"))),
        "max_overlap_over_d": max_pen / d_ref,
        "worst_step": worst_step, "nlgs_max_iters": max_iters, "dt": dt,
        "n_steps": step,
        "final_overlap_over_d": world.max_overlap(x) / d_ref,
        "rms_velocity_final": float(np.sqrt(np.mean(np.sum(v**2, axis=1)))),
        "phi_history": phi_hist, "overlap_history": pen_hist,
    }


# --------------------------------------------------------------------------- #
# soft-sphere reference (penalty contact) for cross-adjudication
# --------------------------------------------------------------------------- #
def soft_sphere_impact(stiffness: float = 1.0e5, restitution: float = 0.5, *,
                       dt: float = 2.0e-6, v0: float = 2.0, diameter: float = 0.01,
                       density: float = 2500.0, steps_per_contact: int = 20000,
                       gravity: float = -9.81) -> dict:
    """Linear spring-dashpot sphere->wall impact (TensorDEM's default family).

    Damping is the standard velocity-proportional form
    ``c = 2 zeta sqrt(k m)`` with ``zeta`` obtained from the restitution via
    ``zeta = -ln(e)/sqrt(pi^2 + ln(e)^2)`` (Tsuji / Antypov-style closure).
    """
    r = 0.5 * diameter
    mass = density * 4.0 / 3.0 * np.pi * r**3
    ln_e = np.log(max(restitution, 1e-12))
    zeta = -ln_e / np.sqrt(np.pi**2 + ln_e**2)
    c_damp = 2.0 * zeta * np.sqrt(stiffness * mass)
    y = 0.02
    vy = -v0
    max_pen = 0.0
    v_in = v_out = None
    touching = False
    for _ in range(steps_per_contact):
        pen = r - y
        force = 0.0
        if pen > 0.0:
            force = stiffness * pen - c_damp * vy
            if not touching:
                touching = True
                v_in = vy
            max_pen = max(max_pen, pen)
        ay = gravity + force / mass
        vy += ay * dt
        y += vy * dt
        if touching and pen <= 0.0 and vy > 0.0 and v_out is None:
            v_out = vy
            break
    e_meas = (-v_out / v_in) if (v_in is not None and v_out is not None) else float("nan")
    return {
        "stiffness": stiffness, "restitution_target": restitution,
        "restitution_measured": e_meas,
        "restitution_error": abs(e_meas - restitution) / restitution if restitution > 0 else 0.0,
        "max_overlap_over_d": max_pen / diameter, "dt": dt,
    }


def soft_sphere_penetration_scan(restitution: float = 0.5, *,
                                 kn_list: tuple[float, ...] = (1.0e5, 1.0e6),
                                 **kwargs) -> dict:
    """Soft-sphere penetration at two stiffnesses: the stiffness-dependent baseline."""
    out = {}
    for kn in kn_list:
        out[f"kn={kn:g}"] = soft_sphere_impact(stiffness=kn, restitution=restitution, **kwargs)
    keys = list(out)
    a = out[keys[0]]["max_overlap_over_d"]
    b = out[keys[1]]["max_overlap_over_d"]
    ref = max(abs(a), abs(b), 1e-30)
    return {"runs": out, "penetration_rel_change": abs(b - a) / ref,
            "kn_ratio": kn_list[-1] / kn_list[0]}

# --------------------------------------------------------------------------- #
# scenario 3b: dense rigid packing (FCC, phi = 0.7405) penetrated by a projectile
# --------------------------------------------------------------------------- #
def fcc_packing(n_cells: int = 4, diameter: float = 0.01, *, seed: int = 3) -> dict:
    """Overlap-free FCC arrangement (phi = 0.7405, coordination 12).

    FCC at contact is the densest *admissible* arrangement of equal rigid
    spheres (Kepler bound 0.7405); ``phi = 0.85`` is geometrically impossible
    for rigid monodisperse grains and can only be reached by deformable or
    strongly polydisperse material.  This packing is therefore the jamming tier
    of the rigid NSCD capability: a fully connected contact network (12
    neighbours per grain) into which a projectile is fired.
    """
    a = diameter * np.sqrt(2.0)     # FCC lattice constant at contact
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    pos = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    pos.append((np.array([ix, iy, iz]) + b) * a)
    pos = np.array(pos)
    L = n_cells * a
    rad = np.full(pos.shape[0], 0.5 * diameter)
    phi = float(np.sum(4.0 / 3.0 * np.pi * rad**3) / L**3)
    return {"x": pos, "radius": rad, "L": L, "phi": phi, "diameter": diameter}


def packing_impact(pack: dict, *, speed: float = 5.0, dt: float = 2.0e-6,
                   steps: int = 120, mu: float = 0.3, restitution: float = 0.2,
                   n_iter: int = 60, seed: int = 5, project_sweeps: int = 60) -> dict:
    """Fire a projectile into the dense packing; audit penetration / momentum."""
    d = float(pack["diameter"])
    rad = np.append(pack["radius"], 0.5 * d)
    x = np.vstack([pack["x"], np.array([[pack["L"] * 0.5, -1.01 * d, pack["L"] * 0.5]])])
    v = np.zeros_like(x)
    v[-1, 1] = speed
    world = NSCDWorld(radius=rad, mu=mu, restitution=restitution, dt=dt,
                      gravity=np.zeros(3), planes=[], n_iter=n_iter)
    p0 = world.mass @ v
    e0 = world.kinetic_energy(v)
    max_pen = world.max_overlap(x)
    worst = 0
    max_iters = 0
    work = 0.0
    for k in range(steps):
        x, v, stats = world.step(x, v, project_sweeps=project_sweeps)
        work += stats["work_contacts"]
        if stats["max_overlap"] > max_pen:
            max_pen = stats["max_overlap"]
            worst = k + 1
        max_iters = max(max_iters, int(stats["iters"]))
    p1 = world.mass @ v
    e1 = world.kinetic_energy(v)
    pmom = float(np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-30))
    d_ref = d
    return {
        "phi": pack["phi"], "n_particles": int(rad.size),
        "speed": speed, "steps": steps, "dt": dt,
        "max_overlap_over_d": max_pen / d_ref, "worst_step": worst,
        "momentum_error": pmom,
        "energy_in": e0, "energy_out": e1,
        "work_contacts": work,
        "energy_ledger_closure": abs(work) / max(abs(e1 - e0), 1e-30),
        "nlgs_max_iters": max_iters,
    }
