"""Buoyant rigid-floe DEM for ice rafting and ridging (2-D, vertical plane).

Why this kernel
---------------
The repository already carries quasi-static ridge models
(``tensordem.ice_ridge_formation``) and granular ice-rubble micro-structure
models, but nothing that integrates *free-floating* rigid floes: hull/contact
kernels in ``tensordem.granular2d`` have gravity but no hydrostatics, and the
granular ice kernels have no free surface.  Rafting is the one ice-formation
process where hydrostatics is the whole physics -- a floe must be lifted out of
the water -- so this module adds a rigid-body discrete-element kernel with

* hydrostatic buoyancy from the exact submerged polygon (Sutherland-Hodgman clip
  of the convex floe against the waterline, so the centre of buoyancy moves as
  the floe tilts),
* quadratic water drag on the submerged envelope,
* node-discretised contact springs with regularised Coulomb friction
  (unconditionally dissipative, hence stable with explicit integration),
* a prescribed piston and a rigid anchor wall, i.e. a confined compression cell.

Physics of the rafting threshold
--------------------------------
For two equal floes of thickness ``h`` and length ``L`` the overriding floe has
to be lifted out of the water by the full step height ``h`` (from draft
``-d = -h rho_i/rho_w`` up to the top surface of the under floe at
``h - d``).  The force-displacement law of a floating body lifted from its
equilibrium position is linear, ``F(z) = rho_w g A_wp z`` (``A_wp = L`` per unit
width), saturating at the floe weight ``rho_i g h L`` when the floe is fully
emerged (``z = d``).  The work to ride up is therefore

    W = 0.5 * rho_i g h L * d = 0.5 * rho_i g h^2 L * (rho_i / rho_w),

and the horizontal convergence that accompanies a unit climb of the leading
face is ``h`` itself (the leading face slides up the square edge of the under
floe), hence

    N_raft = W / h = 0.5 * rho_i g h L (rho_i/rho_w),    tau_raft = N_raft / h.

``tau_raft`` is a *sheet* stress (line load divided by thickness).  Two limits
matter and are reported separately:

* rigid-floe limit (this DEM, and the closed form above): the whole floe must be
  lifted, so ``tau_raft ~ 0.5 rho_i (rho_i/rho_w) g L`` grows with the *floe
  length* and is independent of thickness;
* flexural limit: a long floe cannot be lifted over its whole length; it bends
  and fails, and the participating length is the elastic-foundation length
  ``l_c = (D / (rho_w g))^(1/4)``, giving a much smaller threshold.

The two limits bracket the observed rafting/ridging onset pressures, which is
exactly the diagnostic the verification face reports (see ``anchors.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "IceProperties",
    "FloeSpec",
    "analytic_rafting_threshold",
    "flexural_length",
    "submerged_properties",
    "clip_polygon_below",
    "floe_polygon",
    "FloeWorld",
    "run_rafting_pair",
    "run_ridge",
    "ridge_geometry",
]


@dataclass(frozen=True)
class IceProperties:
    """Material and water properties."""

    rho_ice: float = 917.0
    rho_water: float = 1025.0
    gravity: float = 9.81
    young: float = 5.0e9          # effective elastic modulus of sea ice [Pa]
    poisson: float = 0.3
    flexural_strength: float = 5.0e5   # [Pa]

    def __post_init__(self) -> None:
        for name in ("rho_ice", "rho_water", "gravity", "young", "flexural_strength"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 <= self.poisson < 0.5):
            raise ValueError("poisson must lie in [0, 0.5)")

    @property
    def draft_ratio(self) -> float:
        return self.rho_ice / self.rho_water


def flexural_length(h: float, ice: IceProperties) -> float:
    """Elastic-foundation (characteristic) length of a floating ice plate.

    ``l_c = (D / (rho_w g))^(1/4)`` with ``D = E h^3 / (12 (1 - nu^2))``.
    """
    if h <= 0.0:
        raise ValueError("thickness must be positive")
    d = ice.young * h**3 / (12.0 * (1.0 - ice.poisson**2))
    return (d / (ice.rho_water * ice.gravity)) ** 0.25


def ramp_factor(bevel: float, mu: float) -> float:
    """Friction-corrected wedge factor for riding up a slope.

    The leading edge climbs a ramp of angle ``a = atan(2*bevel)`` (``bevel`` is
    the bevel cut length per unit thickness).  Balancing along the ramp,
    ``N cos a = W (sin a + mu cos a) + mu N sin a`` gives
    ``N = W (sin a + mu cos a) / (cos a - mu sin a)``.
    """
    a = math.atan(min(max(2.0 * bevel, 0.0), 20.0))
    den = math.cos(a) - mu * math.sin(a)
    if den <= 0.0:
        raise ValueError("friction too large for this bevel: the floe locks")
    return (math.sin(a) + mu * math.cos(a)) / den


def analytic_rafting_threshold(h: float, length: float, ice: IceProperties | None = None,
                               *, limit_length: bool = False, bevel: float = 0.5,
                               mu: float = 0.0) -> dict:
    """Closed-form rafting threshold (see the module docstring).

    ``limit_length=True`` replaces the floe length by the elastic-foundation
    length, i.e. returns the flexural-limited variant.
    """
    ice = ice or IceProperties()
    if h <= 0.0 or length <= 0.0:
        raise ValueError("h and length must be positive")
    l_eff = min(length, flexural_length(h, ice)) if limit_length else length
    factor = ramp_factor(bevel, mu) if bevel > 0.0 else float("inf")
    n_raft = 0.5 * ice.rho_ice * ice.gravity * h * l_eff * ice.draft_ratio * factor
    return {
        "h": float(h),
        "length": float(length),
        "l_eff": float(l_eff),
        "line_load": float(n_raft),
        "stress": float(n_raft / h),
        "lift_work": float(n_raft * h),
        "flexural_length": float(flexural_length(h, ice)),
        "ramp_factor": float(factor),
        "bevel": float(bevel),
        "mu": float(mu),
        "limit": "flexural" if limit_length else "rigid",
    }


def rafting_resistance(climb: float, h: float, length: float,
                       ice: IceProperties | None = None, *, bevel: float = 0.5,
                       mu: float = 0.0) -> dict:
    """Instantaneous rafting resistance at a given climb ``z`` of the leading edge.

    A floating floe lifted quasi-statically by ``z`` from its equilibrium position
    needs a vertical force ``rho_w g L z`` (linear regime, valid while
    ``z <= draft = h rho_i/rho_w``); beyond full emergence it needs its whole
    weight ``rho_i g h L``.  Sliding up a ramp of angle ``a = atan(2 bevel)`` with
    ice-ice friction ``mu`` multiplies that by :func:`ramp_factor`.

    This is the *instantaneous* law; its average over ``z`` in ``[0, h]`` gives
    the energy-based form returned by :func:`analytic_rafting_threshold`.
    """
    ice = ice or IceProperties()
    if h <= 0.0 or length <= 0.0:
        raise ValueError("h and length must be positive")
    draft = h * ice.draft_ratio
    z = max(float(climb), 0.0)
    vertical = ice.rho_water * ice.gravity * length * min(z, draft)
    if z > draft:
        vertical += ice.rho_ice * ice.gravity * length * (z - draft)
    factor = ramp_factor(bevel, mu) if bevel > 0.0 else float("inf")
    n = vertical * factor
    return {"climb": z, "climb_ratio": z / h, "vertical_force": float(vertical),
            "line_load": float(n), "stress": float(n / h),
            "ramp_factor": float(factor), "draft": float(draft)}


# ---------------------------------------------------------------------------
# Hydrostatics
# ---------------------------------------------------------------------------

def clip_polygon_below(poly: np.ndarray, waterline: float = 0.0) -> np.ndarray:
    """Sutherland-Hodgman clip of a closed convex polygon to ``y <= waterline``."""
    pts = np.asarray(poly, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("polygon needs at least three vertices")
    out: list[np.ndarray] = []
    n = pts.shape[0]
    for k in range(n):
        cur = pts[k]
        nxt = pts[(k + 1) % n]
        cur_in = cur[1] <= waterline
        nxt_in = nxt[1] <= waterline
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = (waterline - cur[1]) / (nxt[1] - cur[1])
            out.append(cur + t * (nxt - cur))
    if len(out) < 3:
        return np.zeros((0, 2))
    return np.asarray(out)


def submerged_properties(poly: np.ndarray, waterline: float = 0.0) -> tuple[float, float, float]:
    """Submerged area and its centroid ``(area, cx, cy)``."""
    clipped = clip_polygon_below(poly, waterline)
    if len(clipped) < 3:
        return 0.0, float("nan"), float("nan")
    x, y = clipped[:, 0], clipped[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-18:
        return 0.0, float("nan"), float("nan")
    cx = float(np.sum((x + x1) * cross) / (6.0 * area))
    cy = float(np.sum((y + y1) * cross) / (6.0 * area))
    return abs(area), cx, cy


def floe_polygon(cx: float, cy: float, length: float, thickness: float,
                 theta: float = 0.0) -> np.ndarray:
    """Rectangle centred at ``(cx, cy)``, rotated by ``theta`` (radians)."""
    if length <= 0.0 or thickness <= 0.0:
        raise ValueError("length and thickness must be positive")
    hl, hh = 0.5 * length, 0.5 * thickness
    local = np.array([[-hl, -hh], [hl, -hh], [hl, hh], [-hl, hh]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


@dataclass(frozen=True)
class FloeSpec:
    """Rectangular floe, optionally with a beveled leading-bottom edge.

    ``bevel`` is the cut length of the leading (left) bottom corner as a
    fraction of the thickness.  A real broken floe edge is sloped, and the bevel
    is what makes rafting possible at all: with two square edges the contact
    normal is horizontal, the pair is geometrically locked and no lift can be
    generated (this DEM reproduces that lock, see the report).  ``bevel = 0``
    therefore gives a deliberately locking geometry (negative control);
    ``bevel = 0.5`` is a 45-degree edge.
    """

    length: float = 6.0
    thickness: float = 0.5
    n_nodes_per_edge: int = 5
    density: float = 917.0
    bevel: float = 0.5

    def local_polygon(self) -> np.ndarray:
        hl, hh = 0.5 * self.length, 0.5 * self.thickness
        c = float(self.bevel) * self.thickness
        if c <= 0.0:
            return np.array([[-hl, -hh], [hl, -hh], [hl, hh], [-hl, hh]], dtype=float)
        if c > 0.9 * self.thickness:
            raise ValueError("bevel must be at most 0.9 of the thickness")
        return np.array([[-hl + c, -hh], [hl, -hh], [hl, hh], [-hl, hh],
                         [-hl, -hh + c]], dtype=float)


class _Floe:
    """A free-floating rigid rectangular floe (3 DOF plane strain)."""

    __slots__ = ("spec", "pos", "theta", "vel", "omega", "mass", "inertia",
                 "nodes_local", "node_radius", "area", "thickness", "length",
                 "poly_local", "centroid_local")

    def __init__(self, spec: FloeSpec, pos, theta: float, ice: IceProperties,
                 out_of_plane: float = 1.0) -> None:
        self.spec = spec
        self.pos = np.array(pos, dtype=float)
        self.theta = float(theta)
        self.vel = np.zeros(2)
        self.omega = 0.0
        self.length = float(spec.length)
        self.thickness = float(spec.thickness)
        self.poly_local = spec.local_polygon()
        # exact first moments of the local polygon -> centroid & inertia
        a, cx, cy = submerged_properties(self.poly_local, waterline=1e18)
        self.area = float(a)
        self.centroid_local = np.array([cx, cy])
        self.poly_local = self.poly_local - self.centroid_local
        pts = self.poly_local
        inertia = 0.0
        for k in range(len(pts)):
            p0 = pts[k]
            p1 = pts[(k + 1) % len(pts)]
            cr = p0[0] * p1[1] - p1[0] * p0[1]
            inertia += cr * (p0 @ p0 + p0 @ p1 + p1 @ p1)
        self.mass = ice.rho_ice * self.area * out_of_plane
        self.inertia = abs(ice.rho_ice * out_of_plane * inertia / 12.0)
        m = max(int(spec.n_nodes_per_edge), 2)
        nodes = []
        for k in range(len(pts)):
            p0 = pts[k]
            p1 = pts[(k + 1) % len(pts)]
            for t in np.linspace(0.0, 1.0, m, endpoint=False):
                nodes.append(tuple(p0 + t * (p1 - p0)))
        self.nodes_local = np.asarray(nodes, dtype=float)
        perim = float(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)))
        # contact skin: nodes are spaced along the perimeter, the contact
        # radius is a fraction of the spacing so the discretised body is only
        # slightly inflated (0.2 of the spacing, i.e. a few percent of h)
        self.node_radius = 0.2 * perim / len(nodes)

    def vertices(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        return self.poly_local @ rot.T + self.pos

    def nodes_world(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        return self.nodes_local @ rot.T + self.pos


class FloeWorld:
    """Two-dimensional buoyant rigid-floe assembly with a piston cell.

    Boundaries
    ----------
    ``x_left``   rigid vertical anchor wall (left).
    ``piston_x`` prescribed vertical plate moving at ``piston_velocity``
    (negative = compressing to the left).
    The waterline is ``y = 0``; there is no floor: the floes float.
    """

    def __init__(self, floes: list[_Floe], ice: IceProperties | None = None, *,
                 x_left: float = 0.0, piston_x: float | None = None,
                 piston_velocity: float = -0.05, kn: float = 5.0e7,
                 zeta: float = 0.6, mu: float = 0.3, dt: float | None = None,
                 drag_coefficient: float = 1.0, static_segments=None) -> None:
        self.ice = ice or IceProperties()
        self.floes = floes
        self.x_left = float(x_left)
        self.piston_x = float(piston_x) if piston_x is not None else None
        self.piston_velocity = float(piston_velocity)
        self.kn = float(kn)
        self.zeta = float(zeta)
        self.mu = float(mu)
        self.drag_coefficient = float(drag_coefficient)
        segs = [] if static_segments is None else [np.asarray(sg, dtype=float).reshape(2, 2)
                                                   for sg in static_segments]
        self.static_segments = segs
        m_min = max(min(f.mass for f in floes), 1e-9)
        n_nodes = max(len(floes[0].nodes_local), 1)
        omega = math.sqrt(self.kn / (m_min / n_nodes))
        self.dt = dt if dt is not None else 0.05 * 2.0 * math.pi / omega
        self.time = 0.0
        self.piston_force = 0.0
        self._relink()
        self.piston_force_history: list[tuple[float, float]] = []

    # -- geometry / contacts -------------------------------------------
    def _relink(self) -> None:
        pts = []
        owner = []
        for i, f in enumerate(self.floes):
            nw = f.nodes_world()
            pts.append(nw)
            owner.append(np.full(len(nw), i, dtype=np.int64))
        self._node_pos = np.vstack(pts)
        self._node_owner = np.concatenate(owner)
        self._node_r = np.array([f.node_radius for f in self.floes])[self._node_owner]
        self._tree = cKDTree(self._node_pos)
        pairs = self._tree.query_pairs(2.0 * float(self._node_r.max()), output_type="ndarray")
        if len(pairs):
            keep = self._node_owner[pairs[:, 0]] != self._node_owner[pairs[:, 1]]
            pairs = pairs[keep]
        self._pairs = pairs

    def _contact_forces(self) -> tuple[np.ndarray, np.ndarray, float]:
        n_bodies = len(self.floes)
        force = np.zeros((n_bodies, 2))
        torque = np.zeros(n_bodies)
        piston_force = 0.0
        if len(self._pairs):
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            d = self._node_pos[j] - self._node_pos[i]
            dist = np.hypot(d[:, 0], d[:, 1])
            r_sum = self._node_r[i] + self._node_r[j]
            pen = r_sum - dist
            act = pen > 0.0
            if act.any():
                ii, jj = i[act], j[act]
                dist_c = np.maximum(dist[act], 1e-12)
                normal = d[act] / dist_c[:, None]
                ov = pen[act]
                bi, bj = self._node_owner[ii], self._node_owner[jj]
                v_i = np.array([self.floes[b].vel for b in bi])
                v_j = np.array([self.floes[b].vel for b in bj])
                w_i = np.array([self.floes[b].omega for b in bi])
                w_j = np.array([self.floes[b].omega for b in bj])
                ri = self._node_pos[ii] - np.array([self.floes[b].pos for b in bi])
                rj = self._node_pos[jj] - np.array([self.floes[b].pos for b in bj])
                cp_i = v_i + np.stack([-w_i * ri[:, 1], w_i * ri[:, 0]], axis=1)
                cp_j = v_j + np.stack([-w_j * rj[:, 1], w_j * rj[:, 0]], axis=1)
                v_rel = cp_j - cp_i
                v_n = np.einsum("ab,ab->a", v_rel, normal)
                tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
                v_t = v_rel - v_n[:, None] * normal
                v_t_mag = np.hypot(v_t[:, 0], v_t[:, 1])
                m_node = float(np.mean([fl.mass / len(fl.nodes_local) for fl in self.floes]))
                m_eff = 0.5 * m_node
                c_n = 2.0 * self.zeta * math.sqrt(self.kn * m_eff)
                f_n = np.maximum(self.kn * ov - c_n * v_n, 0.0)
                f_t_mag = np.minimum(2.0 * self.zeta * math.sqrt(0.5 * self.kn * m_eff) * v_t_mag,
                                     self.mu * f_n)
                scale = np.where(v_t_mag > 0.0, f_t_mag / np.maximum(v_t_mag, 1e-30), 0.0)
                f_pair = -normal * f_n[:, None] + tangent * scale[:, None]
                np.add.at(force, bi, f_pair)
                np.add.at(force, bj, -f_pair)
                tq = ri[:, 0] * f_pair[:, 1] - ri[:, 1] * f_pair[:, 0]
                np.add.at(torque, bi, tq)
                tq_j = (-rj[:, 0]) * (-f_pair[:, 1]) - (-rj[:, 1]) * (-f_pair[:, 0])
                np.add.at(torque, bj, tq_j)
        # static line segments (ramps, floors, neighbour corners)
        for k, f in enumerate(self.floes):
            if not self.static_segments:
                break
            nw = f.nodes_world()
            rel = nw - f.pos
            vx = f.vel[0] - f.omega * rel[:, 1]
            vy = f.vel[1] + f.omega * rel[:, 0]
            c_n = 2.0 * self.zeta * math.sqrt(self.kn * f.mass / max(len(rel), 1))
            for seg in self.static_segments:
                a = seg[0]
                ab = seg[1] - a
                ll = float(ab @ ab)
                if ll <= 0.0:
                    continue
                tparam = np.clip(((nw - a) @ ab) / ll, 0.0, 1.0)
                closest = a + tparam[:, None] * ab
                dvec = nw - closest
                dist = np.hypot(dvec[:, 0], dvec[:, 1])
                pen = f.node_radius - dist
                sel = pen > 0.0
                if not sel.any():
                    continue
                nrm = dvec[sel] / np.maximum(dist[sel], 1e-12)[:, None]
                v_n = vx[sel] * nrm[:, 0] + vy[sel] * nrm[:, 1]
                f_n = np.maximum(self.kn * pen[sel] - c_n * v_n, 0.0)
                tan = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)
                v_t = (vx[sel] * tan[:, 0] + vy[sel] * tan[:, 1])
                f_t = -np.minimum(2.0 * self.zeta * math.sqrt(self.kn * f.mass / max(len(rel), 1))
                                  * 0.5 * np.abs(v_t), self.mu * f_n) * np.sign(v_t)
                fx = nrm[:, 0] * f_n + tan[:, 0] * f_t
                fy = nrm[:, 1] * f_n + tan[:, 1] * f_t
                force[k, 0] += float(np.sum(fx))
                force[k, 1] += float(np.sum(fy))
                torque[k] += float(np.sum(rel[sel, 0] * fy - rel[sel, 1] * fx))
        # anchor wall (x >= x_left) and piston (x <= piston_x)
        for k, f in enumerate(self.floes):
            nw = f.nodes_world()
            rel = nw - f.pos
            vx = f.vel[0] - f.omega * rel[:, 1]
            r = f.node_radius
            c_n = 2.0 * self.zeta * math.sqrt(self.kn * f.mass / max(len(rel), 1))
            pen_wall = (self.x_left + r) - nw[:, 0]
            sel = pen_wall > 0.0
            if sel.any():
                v_n = vx[sel]
                f_n = np.maximum(self.kn * pen_wall[sel] - c_n * v_n, 0.0)
                force[k, 0] += float(np.sum(f_n))
                torque[k] += float(np.sum(-rel[sel, 1] * f_n))
            if self.piston_x is not None:
                pen_p = nw[:, 0] - (self.piston_x - r)
                sel = pen_p > 0.0
                if sel.any():
                    v_pen = vx[sel] - self.piston_velocity
                    f_n = np.maximum(self.kn * pen_p[sel] - c_n * v_pen, 0.0)
                    force[k, 0] -= float(np.sum(f_n))
                    torque[k] += float(np.sum(rel[sel, 1] * f_n))
                    piston_force += float(np.sum(f_n))
        return force, torque, piston_force

    # -- dynamics -------------------------------------------------------
    def _buoyancy(self, floe: _Floe) -> tuple[np.ndarray, float, float]:
        poly = floe.vertices()
        area, cx, cy = submerged_properties(poly)
        if area <= 0.0:
            return np.zeros(2), 0.0, 0.0
        f = np.array([0.0, self.ice.rho_water * self.ice.gravity * area])
        tq = (cx - floe.pos[0]) * f[1]
        return f, tq, area

    def step(self, n_steps: int = 1) -> None:
        for k in range(n_steps):
            if k % 20 == 0:
                self._relink()
            force, torque, piston_force = self._contact_forces()
            self.piston_force = piston_force
            for idx, f in enumerate(self.floes):
                fb, tb, area_sub = self._buoyancy(f)
                total = force[idx] + fb + np.array([0.0, -f.mass * self.ice.gravity])
                # quadratic water drag on the submerged envelope
                if area_sub > 0.0:
                    speed = float(np.hypot(*f.vel))
                    if speed > 0.0:
                        cd = 0.5 * self.ice.rho_water * self.drag_coefficient * math.sqrt(area_sub)
                        total -= cd * speed * f.vel
                f.vel += total / f.mass * self.dt
                f.omega += (torque[idx] + tb) / f.inertia * self.dt
                # a little rotational drag (water)
                f.omega -= 0.02 * f.omega
                f.pos += f.vel * self.dt
                f.theta += f.omega * self.dt
            if self.piston_x is not None:
                self.piston_x += self.piston_velocity * self.dt
            self.time += self.dt

    def kinetic_energy(self) -> float:
        return float(sum(0.5 * f.mass * float(f.vel @ f.vel)
                         + 0.5 * f.inertia * f.omega**2 for f in self.floes))


def _make_pair(h: float, L: float, ice: IceProperties, *, gap: float, mu: float,
               kn: float, zeta: float, dt: float | None,
               piston_velocity: float, n_nodes: int,
               bevel: float = 0.5, start_climb: float = 0.6) -> tuple[FloeWorld, _Floe, _Floe]:
    draft = h * ice.draft_ratio
    cy = 0.5 * h - draft          # waterline at mid-plane: bottom at -draft
    left = _Floe(FloeSpec(L, h, n_nodes, ice.rho_ice, bevel), (-0.5 * L - gap, cy), 0.0, ice)
    right = _Floe(FloeSpec(L, h, n_nodes, ice.rho_ice, bevel), (0.5 * L + gap, cy), 0.0, ice)
    sep_needed = 4.0 * left.node_radius
    if gap < sep_needed:            # never start with overlapping node skins
        shift = 0.5 * (sep_needed - gap)
        left.pos[0] -= shift
        right.pos[0] += shift
    # Rafting initiation: two equal floes floating side by side touch keel-to-keel
    # and stem-to-stem, which is geometrically locked (no lift component).  The
    # overriding floe is therefore started one thickness higher, resting on the
    # top edge of the under floe -- the configuration a real raft reaches after
    # its edge has crushed/bent over the neighbour.
    if start_climb > 0.0:
        right.pos[1] += start_climb * h
        right.pos[0] += 0.5 * left.node_radius
    # the boundary nodes carry a contact radius, so the rigid wall and the piston
    # must be parked one node radius outside the polygon to start contact-free
    r_off = left.node_radius
    x_left = float(np.min(left.nodes_world()[:, 0])) - r_off
    piston_x = float(np.max(right.nodes_world()[:, 0])) + 1.001 * r_off
    world = FloeWorld([left, right], ice, x_left=x_left, piston_x=piston_x,
                      piston_velocity=piston_velocity, kn=kn, zeta=zeta, mu=mu, dt=dt)
    return world, left, right


def run_rafting_pair(h: float = 0.5, length: float = 6.0, ice: IceProperties | None = None,
                     *, gap: float = 0.05, mu: float = 0.3, kn: float = 5.0e7,
                     zeta: float = 0.6, dt: float | None = None,
                     piston_velocity: float = -0.2, n_nodes_per_edge: int = 11,
                     bevel: float = 0.5, start_climb: float = 0.6,
                     max_steps: int = 40000, sample_every: int = 40,
                     settle_steps: int = 4000) -> dict:
    """Squeeze two floes until one has ridden up on the other.

    Returns the piston load history, the climb history and the stress at
    mid-climb (the operational rafting threshold).
    """
    ice = ice or IceProperties()
    world, left, right = _make_pair(h, length, ice, gap=gap, mu=mu, kn=kn, zeta=zeta,
                                    dt=dt, piston_velocity=piston_velocity,
                                    n_nodes=n_nodes_per_edge, bevel=bevel,
                                    start_climb=start_climb)
    # settle: let both floes find their floating equilibrium before squeezing
    world.piston_velocity = 0.0
    world.step(settle_steps)
    y_top_left = float(left.vertices()[:, 1].max())
    draft = h * ice.draft_ratio
    # leading (left) bottom vertex of the overriding floe: vertex 0 of the
    # rectangle (local order is left-bottom, right-bottom, right-top, left-top)
    lead_idx = np.argsort(right.vertices()[:, 0])[:2]
    bottom0 = float(np.mean(right.vertices()[lead_idx, 1]))
    world.piston_velocity = piston_velocity
    history = []
    onset = None
    peak = 0.0
    for step in range(max_steps):
        world.step(1)
        if step % sample_every == 0:
            bottom = float(np.mean(right.vertices()[lead_idx, 1]))
            climb = bottom - bottom0
            load = world.piston_force
            peak = max(peak, load)
            history.append({"time": float(world.time), "climb": float(climb),
                            "load": float(load), "climb_ratio": float(climb / h),
                            "ke": world.kinetic_energy()})
            if onset is None and climb >= 0.9 * h:
                onset = {"climb": float(climb), "load": float(load),
                         "time": float(world.time),
                         "climb_ratio": float(climb / h)}
                break
    tau_mid = None
    for row in history:
        if row["climb_ratio"] >= 0.5:
            tau_mid = row["load"] / h
            break
    analytic = analytic_rafting_threshold(h, length, ice, bevel=bevel, mu=mu)
    # report the analytic value with the flexural-limited length as well, since a
    # 6 m floe is shorter than its 8.7 m elastic-foundation length but a 30 m one
    # is not
    return {
        "h": float(h),
        "length": float(length),
        "mu": float(mu),
        "bevel": float(bevel),
        "piston_velocity": float(piston_velocity),
        "dt": float(world.dt),
        "steps": len(history) * sample_every,
        "history": history,
        "rafting_onset": onset,
        "peak_load": float(peak),
        "tau_mid_climb": None if tau_mid is None else float(tau_mid),
        "tau_peak": float(peak / h),
        "tau_analytic": analytic["stress"],
        "tau_analytic_flexural": analytic_rafting_threshold(
            h, length, ice, limit_length=True, bevel=bevel, mu=mu)["stress"],
        "start_climb": float(start_climb),
        "ratio_dem_over_analytic": (None if tau_mid is None
                                    else float((tau_mid) / analytic["stress"])),
        "y_top_left": y_top_left,
        "draft": float(draft),
        "gest": None,
        "final_positions": [[float(p[0]), float(p[1]), float(f.theta)] for f, p in
                            zip(world.floes, [f.pos for f in world.floes])],
        "final_vertices": [f.vertices().tolist() for f in world.floes],
    }


def run_ridge(h: float = 0.5, n_floes: int = 12, ice: IceProperties | None = None,
              *, length_span: tuple[float, float] = (3.0, 10.0), gap: float = 0.05,
              mu: float = 0.3, kn: float = 5.0e7, zeta: float = 0.6,
              dt: float | None = None, piston_velocity: float = -0.2,
              n_nodes_per_edge: int = 11, bevel: float = 0.5, convergence: float = 0.18,
              settle_steps: int = 3000, sample_every: int = 200,
              max_steps: int = 200000, seed: int = 0) -> dict:
    """Converge a chain of floes into a ridge (rafting/rubbling) pile.

    ``convergence`` is the prescribed fractional shortening of the original
    chain.  The emergent geometry is measured by :func:`ridge_geometry`.
    """
    ice = ice or IceProperties()
    rng = np.random.default_rng(seed)
    lengths = np.linspace(length_span[0], length_span[1], n_floes)
    rng.shuffle(lengths)
    draft = h * ice.draft_ratio
    cy = 0.5 * h - draft
    floes = []
    # the neighbours must start with non-overlapping node skins (4r), otherwise
    # the chain explodes on the first step (a real failure mode we hit)
    probe = _Floe(FloeSpec(float(lengths[0]), h, n_nodes_per_edge, ice.rho_ice, bevel),
                  (0.0, cy), 0.0, ice)
    gap = max(gap, 4.5 * probe.node_radius)
    x = 0.0
    for L in lengths:
        floes.append(_Floe(FloeSpec(float(L), h, n_nodes_per_edge, ice.rho_ice, bevel),
                           (x + 0.5 * L, cy), 0.0, ice))
        x += L + gap
    r_off = floes[0].node_radius
    x_lo = float(min(np.min(f.nodes_world()[:, 0]) for f in floes)) - r_off
    x_hi = float(max(np.max(f.nodes_world()[:, 0]) for f in floes)) + 1.001 * r_off
    world = FloeWorld(floes, ice, x_left=x_lo, piston_x=x_hi, piston_velocity=0.0,
                      kn=kn, zeta=zeta, mu=mu, dt=dt)
    world.step(settle_steps)
    target = convergence * sum(lengths)
    world.piston_velocity = piston_velocity
    area_initial = sum(f.area for f in floes)
    moved = 0.0
    hist = []
    steps = 0
    while moved < target and steps < max_steps:
        world.step(1)
        steps += 1
        moved += abs(piston_velocity) * world.dt
        if steps % sample_every == 0:
            geom = ridge_geometry(floes, h, ice)
            hist.append({"time": float(world.time), "moved": float(moved),
                         "sail_height": geom["sail_height"],
                         "keel_depth": geom["keel_depth"],
                         "sail_width": geom["sail_width"],
                         "load": float(world.piston_force)})
    geom = ridge_geometry(floes, h, ice)
    geom.update({
        "h": float(h),
        "n_floes": int(n_floes),
        "convergence": float(moved / sum(lengths)),
        "piston_force": float(world.piston_force),
        "tau_ridge": float(world.piston_force / h),
        "area_initial": float(area_initial),
        "area_final": float(sum(f.area for f in floes)),
        "steps": int(steps),
        "dt": float(world.dt),
        "history": hist,
        "polygons": [f.vertices().tolist() for f in floes],
    })
    return geom


def ridge_geometry(floes: list[_Floe], h: float, ice: IceProperties | None = None) -> dict:
    """Sail/keel geometry of a floe pile (waterline at ``y = 0``)."""
    ice = ice or IceProperties()
    polys = [f.vertices() for f in floes]
    allv = np.vstack(polys)
    sail_height = float(max(allv[:, 1].max(), 0.0))
    keel_depth = float(max(-allv[:, 1].min(), 0.0))
    above = allv[allv[:, 1] > 0.0]
    x_lo = float(above[:, 0].min()) if len(above) else float("nan")
    x_hi = float(above[:, 0].max()) if len(above) else float("nan")
    sail_width = float(x_hi - x_lo) if len(above) else 0.0
    below = allv[allv[:, 1] < 0.0]
    kx_lo = float(below[:, 0].min()) if len(below) else float("nan")
    kx_hi = float(below[:, 0].max()) if len(below) else float("nan")
    keel_width = float(kx_hi - kx_lo) if len(below) else 0.0
    area_sub = 0.0
    for poly in polys:
        a, _, _ = submerged_properties(poly)
        area_sub += a
    return {
        "sail_height": sail_height,
        "keel_depth": keel_depth,
        "sail_width": sail_width,
        "keel_width": keel_width,
        "sail_aspect": (sail_height / sail_width) if sail_width > 0 else float("nan"),
        "keel_aspect": (keel_depth / keel_width) if keel_width > 0 else float("nan"),
        "sail_keel_ratio": (sail_height / keel_depth) if keel_depth > 0 else float("nan"),
        "submerged_area": float(area_sub),
        "draft_ratio": float(ice.draft_ratio),
    }

def run_rafting_ramp(h: float = 0.5, length: float = 6.0, ice: IceProperties | None = None,
                     *, ramp_angle_deg: float = 45.0, mu: float = 0.3,
                     kn: float = 5.0e7, zeta: float = 0.6, dt: float | None = None,
                     piston_velocity: float = -0.1, n_nodes_per_edge: int = 11,
                     bevel: float = 0.5, max_steps: int = 30000,
                     sample_every: int = 100, settle_steps: int = 3000,
                     ramp_length: float = 6.0) -> dict:
    """Ride one floating floe up a rigid ramp: the rafting resistance law.

    The ramp is a fixed 45-degree-or-whatever line under the leading bevel of the
    floe, i.e. the idealised edge of the neighbouring floe.  The piston pushes the
    floe along the ramp; the quasi-static piston load is the rafting resistance of
    a floe that has to be lifted out of the water while sliding on ice-ice
    friction.  This is the *forward* rafting problem in its cleanest form: one
    free body, real hydrostatics, real contact, no two-body locking.
    """
    ice = ice or IceProperties()
    h = float(h)
    alpha = math.radians(ramp_angle_deg)
    cy = 0.5 * h - h * ice.draft_ratio
    floe = _Floe(FloeSpec(length, h, n_nodes_per_edge, ice.rho_ice, bevel),
                 (0.0, cy), 0.0, ice)
    verts = floe.vertices()
    lead = int(np.argmin(verts[:, 0]))
    toe = verts[lead]                        # leading bottom vertex of the bevel
    r = floe.node_radius
    # ramp line through a point just below the toe, rising to the left
    normal = np.array([math.sin(alpha), math.cos(alpha)])   # points away from ramp
    base = toe - 1.02 * r * normal
    seg = np.array([[base[0] + ramp_length * math.cos(alpha),
                     base[1] - ramp_length * math.sin(alpha)],
                    [base[0] - 2.0 * math.cos(alpha),
                     base[1] + 2.0 * math.sin(alpha)]])
    piston_x = float(np.max(floe.nodes_world()[:, 0])) + 1.02 * r
    world = FloeWorld([floe], ice, x_left=-1e6, piston_x=piston_x,
                      piston_velocity=0.0, kn=kn, zeta=zeta, mu=mu, dt=dt,
                      static_segments=[seg])
    world.step(settle_steps)
    world.piston_velocity = piston_velocity
    y0 = float(np.mean(floe.vertices()[np.argsort(floe.vertices()[:, 0])[:2], 1]))
    history = []
    tau_mid = None
    peak = 0.0
    for step in range(max_steps):
        world.step(1)
        if step % sample_every == 0:
            y = float(np.mean(floe.vertices()[np.argsort(floe.vertices()[:, 0])[:2], 1]))
            climb = y - y0
            load = world.piston_force
            peak = max(peak, load)
            history.append({"time": float(world.time), "climb": float(climb),
                            "climb_ratio": float(climb / h), "load": float(load),
                            "theta": float(floe.theta)})
            if tau_mid is None and climb >= 0.45 * h:
                tau_mid = load
                break
    tanh_ref = analytic_rafting_threshold(h, length, ice, bevel=bevel, mu=mu)
    return {
        "h": h, "length": float(length), "ramp_angle_deg": float(ramp_angle_deg),
        "mu": float(mu), "bevel": float(bevel), "dt": float(world.dt),
        "history": history, "peak_load": float(peak),
        "tau_mid_climb": None if tau_mid is None else float(tau_mid / h),
        "tau_peak": float(peak / h),
        "tau_analytic": tanh_ref["stress"],
        "ramp_factor": tanh_ref["ramp_factor"],
        "ratio_dem_over_analytic": (None if tau_mid is None
                                    else float((tau_mid / h) / tanh_ref["stress"])),
        "polygon": floe.vertices().tolist(),
        "ramp": seg.tolist(),
    }
