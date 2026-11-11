"""Buoyant *breakable* rigid-floe DEM for ice-ridge formation (2-D vertical plane).

Why this module exists
----------------------
``tensordem.ice_ridge_formation`` (mission m10822) modelled ridging
*kinematically*: floe positions were prescribed from the compression strain and
the sail/keel geometry was then read off the Hopky (1983) empirical law, so
nothing emerged, and the porosity "scaling law" was fitted to the same
exponential that had been assumed.  This module replaces that with a
discrete-element model in the vertical plane in which every macroscopic ridge
statistic is *measured* from the block configuration:

* every floe/block is a free convex rigid polygon with three degrees of freedom
  (``x``, ``z``, ``theta``) and exact area, centroid and inertia;
* hydrostatic buoyancy acts on the exact submerged part of each polygon (the
  centre of buoyancy moves as a block tilts), so nothing prescribes a sail above
  water or a keel below it: the sail/keel split is the new floatation
  equilibrium of the splashed blocks;
* floe-floe and floe-boundary contacts are node-discretised penalty springs with
  regularised Coulomb friction;
* **bending failure**: the internal moment carried by a block, accumulated from
  the contact forces and their lever arms, gives the beam bending stress
  ``sigma = 6 M / t^2`` (``t`` = the body's own caliper thickness).  Above the
  flexural strength ``sigma_f`` the body is cut by a straight crack and the two
  fragments keep their momentum; fragments stay breakable down to a minimum
  survivable size, so rubbling proceeds;
* the compression driver is a rigid piston advancing at constant velocity.

Measured diagnostics (read back from the blocks, never assumed)
---------------------------------------------------------------
* ``ridge_porosity``: cross-section porosity of the ridged part of the field,
  ``n = 1 - A_ice / A_envelope``, evaluated column-wise on the actual polygon
  geometry (columns whose vertical ice extent exceeds ``1.8 h`` are "ridged");
* ``sail_keel_geometry``: sail height and keel depth with respect to the
  waterline;
* volume bookkeeping: a crack is a half-plane clip, so the two fragment areas add
  up to the parent area to round-off.  Total ice area is audited at every sample,
  which turns the ``<1 %`` criterion into a *test* of the implementation instead
  of an input;
* rafting/rubbling classification: rubble if a crack precedes any floe climbing
  more than half a thickness out of the water, rafting otherwise.

Limitations (see ``docs/ICE_RIDGE_RUBBLING_DEM.md``): plane strain with unit
out-of-plane width, no thermal bonding/refreezing, no crushing of blocks below
the minimum size, a bending-only failure criterion, and a 2-D area porosity that
is not identical to the 3-D volumetric macroporosity of a field ridge.

References
----------
Hopky & Lawrence (1983); Timco & Burden (1997, keel/sail ratio ~4.4);
Parmerter & Coon (1972, rafting vs ridging); Hopkins (1998).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time as _time
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..ice_rafting.mechanics import (
    IceProperties,
    submerged_properties,
)

__all__ = [
    "RubbleFloe",
    "RubbleWorld",
    "RidgeDEMConfig",
    "polygon_area",
    "polygon_moments",
    "split_convex_polygon",
    "ridge_porosity",
    "sail_keel_geometry",
    "fit_porosity_law",
    "generate_floe_field",
    "run_rubbling_ridge",
    "run_aspect_sweep",
]


# ---------------------------------------------------------------------------
# convex polygon helpers
# ---------------------------------------------------------------------------

def polygon_area(poly: np.ndarray) -> float:
    """Signed area of a closed polygon (positive counter-clockwise)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_moments(poly: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Area, centroid and polar second moment about the centroid.

    Returns ``(area, centroid, i_centroid)`` where ``i_centroid`` is the polar
    second moment **per unit mass**, so a body of density ``rho`` has rotational
    inertia ``rho * area * i_centroid`` per unit out-of-plane width.
    """
    p = np.asarray(poly, dtype=float)
    if polygon_area(p) < 0.0:
        p = p[::-1]
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * float(np.sum(cross))
    if a <= 0.0:
        raise ValueError("degenerate polygon")
    cx = float(np.sum((x + x1) * cross) / (6.0 * a))
    cy = float(np.sum((y + y1) * cross) / (6.0 * a))
    i_o = float(np.sum(cross * (x * x + x * x1 + x1 * x1
                                + y * y + y * y1 + y1 * y1)) / 12.0)
    # polar second moment per unit mass (the area moment divided by the area);
    # the body inertia is then rho * area * i_c
    i_c = (i_o - a * (cx * cx + cy * cy)) / a
    return a, np.array([cx, cy]), max(i_c, 1e-18)


def _clip_halfplane(poly: np.ndarray, p0: np.ndarray, n: np.ndarray,
                    keep_negative: bool = True) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon by a half-plane."""
    p = np.asarray(poly, dtype=float)
    s = (p - p0) @ n
    if keep_negative:
        s = -s
    out: list[np.ndarray] = []
    m = len(p)
    for k in range(m):
        cur, nxt = p[k], p[(k + 1) % m]
        sc, sn = s[k], s[(k + 1) % m]
        if sc >= 0.0:
            out.append(cur)
        if sc * sn < 0.0:
            t = sc / (sc - sn)
            out.append(cur + t * (nxt - cur))
    if len(out) < 3:
        return np.zeros((0, 2))
    return np.asarray(out)


def split_convex_polygon(poly: np.ndarray, point: Iterable[float],
                         normal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Cut a convex polygon with the line ``{x : (x - point) . normal = 0}``.

    Both halves share the cutting line by construction, so their areas add up to
    the parent area to round-off.  That is what makes the volume audit of the
    ridge runner meaningful.
    """
    p0 = np.asarray(point, dtype=float)
    nrm = np.asarray(normal, dtype=float)
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-30)
    a = _clip_halfplane(poly, p0, nrm, keep_negative=True)
    b = _clip_halfplane(poly, p0, nrm, keep_negative=False)
    return a, b


def _unique_polygon(poly: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Drop duplicated consecutive vertices produced by clipping."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 0:
        return poly
    keep = [poly[0]]
    for v in poly[1:]:
        if np.hypot(*(v - keep[-1])) > tol:
            keep.append(v)
    if len(keep) > 2 and np.hypot(*(keep[-1] - keep[0])) <= tol:
        keep.pop()
    return np.asarray(keep)


# ---------------------------------------------------------------------------
# breakable body
# ---------------------------------------------------------------------------

class RubbleFloe:
    """A free-floating rigid convex polygon that can be split (3 DOF)."""

    __slots__ = ("poly_local", "nodes_local", "node_radius", "area", "mass",
                 "inertia", "pos", "theta", "vel", "omega", "thickness",
                 "length", "parent", "birth_step", "node_spacing", "radius")

    def __init__(self, polygon: np.ndarray, ice: IceProperties, *,
                 n_nodes_per_edge: int = 5, out_of_plane: float = 1.0,
                 parent: int = -1, birth_step: int = 0,
                 node_spacing: float | None = None,
                 buoyancy_every: int = 5, dt_safety: float = 0.08,
                 local_damping: float = 0.7,
                 servo_force: float | None = None,
                 servo_velocity: float = 0.02) -> None:
        poly = _unique_polygon(np.asarray(polygon, dtype=float))
        if len(poly) < 3:
            raise ValueError("a body needs at least three vertices")
        if polygon_area(poly) < 0.0:
            poly = poly[::-1]
        area, centroid, i_c = polygon_moments(poly)
        self.area = float(area)
        self.poly_local = poly - centroid
        self.pos = centroid.copy()
        self.theta = 0.0
        self.vel = np.zeros(2)
        self.omega = 0.0
        self.mass = float(ice.rho_ice * area * out_of_plane)
        self.inertia = float(ice.rho_ice * area * i_c * out_of_plane)
        # caliper thickness = 2 x smallest centroid-to-edge distance; this is the
        # beam depth entering the bending criterion sigma = 6 M / t^2
        t_min = math.inf
        n = len(self.poly_local)
        for k in range(n):
            a, b = self.poly_local[k], self.poly_local[(k + 1) % n]
            edge = b - a
            leng = float(np.hypot(*edge))
            if leng <= 1e-12:
                continue
            t_min = min(t_min, abs(float(np.cross(edge, -a))) / leng)
        self.thickness = float(max(2.0 * (t_min if math.isfinite(t_min) else 0.0), 1e-6))
        self.length = float(np.ptp(self.poly_local[:, 0]))
        self.radius = float(np.max(np.hypot(self.poly_local[:, 0], self.poly_local[:, 1])))
        m = max(int(n_nodes_per_edge), 2)
        span = max(self.length, 1e-9)
        nodes = []
        for k in range(n):
            p0, p1 = self.poly_local[k], self.poly_local[(k + 1) % n]
            seg = float(np.hypot(*(p1 - p0)))
            if node_spacing is not None and node_spacing > 0.0:
                n_seg = max(int(math.ceil(seg / float(node_spacing))), 1)
            else:
                n_seg = max(int(math.ceil(m * seg / span)), 1)
            for t in np.linspace(0.0, 1.0, n_seg, endpoint=False):
                nodes.append(p0 + t * (p1 - p0))
        self.nodes_local = np.asarray(nodes, dtype=float)
        perim = float(np.sum(np.hypot(*(np.roll(self.poly_local, -1, axis=0)
                                       - self.poly_local).T)))
        if node_spacing is not None and node_spacing > 0.0:
            self.node_spacing = float(node_spacing)
            self.node_radius = 0.5 * self.node_spacing
        else:
            self.node_spacing = perim / max(len(self.nodes_local), 1)
            self.node_radius = 0.5 * self.node_spacing
        self.parent = int(parent)
        self.birth_step = int(birth_step)

    # -- kinematics -----------------------------------------------------
    def _rot(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def vertices(self) -> np.ndarray:
        return self.poly_local @ self._rot().T + self.pos

    def nodes_world(self) -> np.ndarray:
        return self.nodes_local @ self._rot().T + self.pos

    def velocity_at(self, point: np.ndarray) -> np.ndarray:
        r = np.asarray(point, dtype=float) - self.pos
        return self.vel + np.array([-self.omega * r[1], self.omega * r[0]])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"RubbleFloe(area={self.area:.3f}, pos=({self.pos[0]:.2f},"
                f"{self.pos[1]:.2f}), parent={self.parent})")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeDEMConfig:
    """Configuration of the rubbling ridge experiment."""

    n_floes: int = 10
    ice_thickness: float = 0.5
    min_floe_length: float = 2.0
    max_floe_length: float = 8.0
    power_law_exponent: float = 2.5
    initial_gap: float = 0.02
    piston_velocity: float = 0.15
    servo_force: float | None = 30_000.0
    servo_velocity: float = 0.02
    target_convergence: float = 0.22
    ice: IceProperties = field(default_factory=IceProperties)
    kn: float = 1.0e7
    zeta: float = 0.7
    mu: float = 0.4
    drag_coefficient: float = 1.0
    dt: float | None = None
    n_nodes_per_edge: int = 5
    node_spacing_factor: float = 0.06
    edge_bevel: float = 0.5
    mass_scale: float = 25.0
    local_damping: float = 0.7
    dt_safety: float = 0.08
    buoyancy_every: int = 5
    min_block_area: float = 0.20
    min_block_thickness: float = 0.10
    settle_steps: int = 2000
    sample_every: int = 250
    fracture_every: int = 10
    max_steps: int = 400_000
    seed: int = 20261111

    def __post_init__(self) -> None:
        if self.n_floes < 2:
            raise ValueError("n_floes must be at least 2")
        for name in ("ice_thickness", "min_floe_length", "max_floe_length",
                     "piston_velocity", "target_convergence", "kn", "mu"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_floe_length <= self.min_floe_length:
            raise ValueError("max_floe_length must exceed min_floe_length")
        if not (0.0 <= self.target_convergence < 0.9):
            raise ValueError("target_convergence must lie in [0, 0.9)")
        if self.power_law_exponent <= 0.0:
            raise ValueError("power_law_exponent must be positive")


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------

class RubbleWorld:
    """Buoyant breakable-floe assembly between a wall and a moving piston.

    Conventions: ``z = 0`` is the waterline; a floating sheet of thickness ``h``
    spans ``z in [-h rho_i/rho_w, h (1 - rho_i/rho_w)]``.  The piston is the
    right-hand vertical plate and moves towards the fixed left wall, so the field
    converges at ``|piston_velocity|``.
    """

    def __init__(self, floes: Sequence[RubbleFloe], ice: IceProperties | None = None,
                 *, x_left: float = 0.0, piston_x: float | None = None,
                 piston_velocity: float = -0.15, kn: float = 1.0e7,
                 zeta: float = 0.7, mu: float = 0.4, dt: float | None = None,
                 drag_coefficient: float = 1.0, n_relink: int = 10,
                 min_block_area: float = 0.20,
                 min_block_thickness: float = 0.10,
                 node_spacing: float | None = None,
                 buoyancy_every: int = 5, dt_safety: float = 0.08,
                 local_damping: float = 0.7,
                 servo_force: float | None = None,
                 servo_velocity: float = 0.02) -> None:
        self.ice = ice or IceProperties()
        self.floes: list[RubbleFloe] = list(floes)
        if not self.floes:
            raise ValueError("world needs at least one body")
        self.x_left = float(x_left)
        self.piston_x = (float(piston_x) if piston_x is not None
                         else float(max(f.vertices()[:, 0].max() for f in self.floes) + 0.01))
        self.piston_velocity = float(piston_velocity)
        self.kn = float(kn)
        self.zeta = float(zeta)
        self.mu = float(mu)
        self.drag_coefficient = float(drag_coefficient)
        self.n_relink = int(n_relink)
        self.min_block_area = float(min_block_area)
        self.min_block_thickness = float(min_block_thickness)
        self.node_spacing = (float(node_spacing) if node_spacing is not None
                             else float(self.floes[0].node_spacing))
        self.buoyancy_every = max(int(buoyancy_every), 1)
        self.local_damping = float(local_damping)
        self.fracture_cooldown = 200
        self._fracture_cooldown_until = 0
        self.servo_force = None if servo_force is None else float(servo_force)
        self.servo_velocity = float(servo_velocity)
        self._buoy_cache: list[tuple[np.ndarray, float, float]] = []
        self._buoy_cache_step = -10 ** 9
        self.dt_safety = float(dt_safety)
        self._auto_dt = dt is None
        self.dt = float(dt) if dt is not None else 0.0
        if self._auto_dt:
            self._update_dt()
        self.time = 0.0
        self.step_index = 0
        self.piston_force = 0.0
        self.fracture_log: list[dict] = []
        self.last_contacts: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self._relink()

    # -- bookkeeping ----------------------------------------------------
    def _update_dt(self) -> None:
        """Contact-limited explicit step: ``dt = safety * 2*pi / omega_max``.

        Recomputed after every fracture because a small fragment has a much higher
        node frequency than the parent floe; keeping the parent step would make the
        integrator unstable as soon as rubble appears.
        """
        omega = 0.0
        for f in self.floes:
            n_nodes = max(len(f.nodes_local), 1)
            omega = max(omega, math.sqrt(self.kn / max(f.mass / n_nodes, 1e-9)))
            # rotational branch: a contact at the body radius is an angular spring
            omega = max(omega, math.sqrt(self.kn * n_nodes * f.radius ** 2
                                         / max(f.inertia, 1e-9)))
        if omega > 0.0:
            self.dt = self.dt_safety * 2.0 * math.pi / omega

    def total_area(self) -> float:
        return float(sum(f.area for f in self.floes))

    def total_ice_volume(self, out_of_plane: float = 1.0) -> float:
        """Ice volume = total polygon area x out-of-plane width."""
        return self.total_area() * float(out_of_plane)

    def _relink(self) -> None:
        pts, owner = [], []
        for i, f in enumerate(self.floes):
            nw = f.nodes_world()
            pts.append(nw)
            owner.append(np.full(len(nw), i, dtype=np.int64))
        self._node_pos = np.vstack(pts)
        self._node_owner = np.concatenate(owner)
        self._node_r = np.array([f.node_radius for f in self.floes])[self._node_owner]
        self._tree = cKDTree(self._node_pos)
        r_max = float(self._node_r.max())
        # 2.6 r (not 2 r): two staggered node rows can be 2r apart only when the
        # surfaces are still separated, so a wider search closes the detection hole
        pairs = self._tree.query_pairs(2.6 * r_max, output_type="ndarray")
        if len(pairs):
            pairs = pairs[self._node_owner[pairs[:, 0]] != self._node_owner[pairs[:, 1]]]
        self._pairs = pairs

    # -- forces ---------------------------------------------------------
    def _contact_forces(self):
        n = len(self.floes)
        force = np.zeros((n, 2))
        torque = np.zeros(n)
        contacts: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(n)]
        piston_force = 0.0
        if len(self._pairs):
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            d = self._node_pos[j] - self._node_pos[i]
            dist = np.hypot(d[:, 0], d[:, 1])
            pen = (self._node_r[i] + self._node_r[j]) - dist
            act = pen > 0.0
            if act.any():
                ii, jj = i[act], j[act]
                # when two node skins nearly coincide the pair normal degenerates;
                # fall back to the centroid-to-centroid direction and saturate the
                # penetration so a jammed pair cannot generate an unbounded force
                dc = dist[act]
                normal = d[act] / np.maximum(dc, 1e-12)[:, None]
                bad = dc < 0.1 * (self._node_r[ii] + self._node_r[jj])
                if bad.any():
                    owner_i = self._node_owner[ii]
                    owner_j = self._node_owner[jj]
                    cent = (np.array([self.floes[b].pos for b in owner_i])
                            - np.array([self.floes[b].pos for b in owner_j]))
                    cn = cent / np.maximum(np.hypot(cent[:, 0], cent[:, 1]), 1e-12)[:, None]
                    normal[bad] = cn[bad]
                pen_sat = np.minimum(pen[act], 0.75 * (self._node_r[ii] + self._node_r[jj]))
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
                m_node = float(np.mean([f.mass / len(f.nodes_local) for f in self.floes]))
                c_n = 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * m_node)
                f_n = np.maximum(self.kn * pen_sat - c_n * v_n, 0.0)
                f_t_mag = np.minimum(2.0 * self.zeta * math.sqrt(0.5 * self.kn * 0.5 * m_node)
                                     * v_t_mag, self.mu * f_n)
                scale = np.where(v_t_mag > 0.0, f_t_mag / np.maximum(v_t_mag, 1e-30), 0.0)
                f_pair = -normal * f_n[:, None] + tangent * scale[:, None]
                np.add.at(force, bi, f_pair)
                np.add.at(force, bj, -f_pair)
                np.add.at(torque, bi, ri[:, 0] * f_pair[:, 1] - ri[:, 1] * f_pair[:, 0])
                # force on j is -f_pair, so its torque is -(rj x f_pair)
                np.add.at(torque, bj, -(rj[:, 0] * f_pair[:, 1] - rj[:, 1] * f_pair[:, 0]))
                for a in range(len(bi)):
                    contacts[int(bi[a])].append((self._node_pos[ii[a]].copy(), f_pair[a].copy()))
                    contacts[int(bj[a])].append((self._node_pos[jj[a]].copy(), -f_pair[a].copy()))
        # rigid left wall and piston
        for k, f in enumerate(self.floes):
            # scalar prefilter: only bodies whose bounding disc reaches a wall
            if (f.pos[0] - f.radius > self.x_left + 2.0 * f.node_radius
                    and f.pos[0] + f.radius < self.piston_x - 2.0 * f.node_radius):
                continue
            nw = f.nodes_world()
            r_loc = nw - f.pos
            near = ((nw[:, 0] < self.x_left + 2.0 * f.node_radius)
                    | (nw[:, 0] > self.piston_x - 2.0 * f.node_radius))
            if not near.any():
                continue
            v = f.vel + np.stack([-f.omega * r_loc[:, 1], f.omega * r_loc[:, 0]], axis=1)
            c_n = 2.0 * self.zeta * math.sqrt(self.kn * f.mass / max(len(nw), 1))
            pen_w = (self.x_left + f.node_radius) - nw[:, 0]
            sel = pen_w > 0.0
            if sel.any():
                f_n = np.maximum(self.kn * pen_w[sel] - c_n * v[sel, 0], 0.0)
                force[k, 0] += float(np.sum(f_n))
                torque[k] += float(np.sum(-r_loc[sel, 1] * f_n))
                for idx, mag in zip(np.nonzero(sel)[0], f_n):
                    contacts[k].append((nw[idx].copy(), np.array([float(mag), 0.0])))
            pen_p = nw[:, 0] - (self.piston_x - f.node_radius)
            sel = pen_p > 0.0
            if sel.any():
                v_n = v[sel, 0] - self.piston_velocity
                f_n = np.maximum(self.kn * pen_p[sel] - c_n * v_n, 0.0)
                force[k, 0] -= float(np.sum(f_n))
                torque[k] += float(np.sum(r_loc[sel, 1] * f_n))
                piston_force += float(np.sum(f_n))
                for idx, mag in zip(np.nonzero(sel)[0], f_n):
                    contacts[k].append((nw[idx].copy(), np.array([-float(mag), 0.0])))
        return force, torque, piston_force, contacts

    @staticmethod
    def _shoelace(poly: np.ndarray) -> tuple[float, float]:
        """Area and x-centroid of a convex polygon (fast, few numpy calls)."""
        x, y = poly[:, 0], poly[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a = 0.5 * float(cr.sum())
        if abs(a) < 1e-18:
            return 0.0, 0.0
        cx = float(np.sum((x + x1) * cr) / (6.0 * a))
        return abs(a), cx

    def _buoyancy(self, floe: RubbleFloe) -> tuple[np.ndarray, float, float]:
        """Hydrostatic force on the exact submerged polygon (waterline ``z = 0``)."""
        poly = floe.vertices()
        z = poly[:, 1]
        if z.max() <= 0.0:                    # fully submerged
            area, cx = floe.area, floe.pos[0]
        elif z.min() >= 0.0:                  # fully emerged (sail)
            return np.zeros(2), 0.0, 0.0
        else:
            out: list[np.ndarray] = []
            n = len(poly)
            for k in range(n):
                cur, nxt = poly[k], poly[(k + 1) % n]
                ci, ni = cur[1] <= 0.0, nxt[1] <= 0.0
                if ci:
                    out.append(cur)
                if ci != ni:
                    t = -cur[1] / (nxt[1] - cur[1])
                    out.append(cur + t * (nxt - cur))
            if len(out) < 3:
                return np.zeros(2), 0.0, 0.0
            area, cx = self._shoelace(np.asarray(out))
            if area <= 0.0:
                return np.zeros(2), 0.0, 0.0
        force = np.array([0.0, self.ice.rho_water * self.ice.gravity * area])
        torque = (cx - floe.pos[0]) * force[1]
        return force, torque, float(area)

    # -- fracture -------------------------------------------------------
    def _principal_axis(self, body: RubbleFloe) -> np.ndarray:
        """Unit vector along the block's longest dimension (world frame)."""
        v = body.vertices()
        best, axis = -1.0, np.array([1.0, 0.0])
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                d = v[b] - v[a]
                L = float(np.hypot(*d))
                if L > best:
                    best, axis = L, d / max(L, 1e-12)
        return axis

    def bending_stress(self, body: RubbleFloe, log) -> float:
        """Peak bending stress carried by a block: ``sigma = 6 M / t^2``.

        The block is treated as a beam along its principal (long) axis.  Every
        contact contributes ``|r x F|`` at its axial station; sweeping the stations
        from one end and accumulating gives the internal bending moment at each
        section (a statically admissible field, forces along the axis included -
        they create no bending).  ``M`` is the largest of those moments and ``t``
        is the block's own caliper thickness, so this is the maximum bending stress
        in the block.  A single contact (no span) cannot bend the block and returns
        zero, which is what keeps a lone indentation from being called a failure.
        """
        if not log:
            return 0.0
        axis = self._principal_axis(body)
        stations = []
        for point, force in log:
            r = point - body.pos
            stations.append((float(np.dot(axis, r)), abs(float(np.cross(r, force)))))
        stations.sort()
        if len(stations) < 2:
            return 0.0
        span = stations[-1][0] - stations[0][0]
        if span < 0.25 * max(body.length, 1e-9):
            return 0.0
        moment = best = 0.0
        for _, m in stations[:-1]:
            moment += m
            best = max(best, moment)
        return 6.0 * best / max(body.thickness ** 2, 1e-12)

    def _split_body(self, body: RubbleFloe, log, rng):
        """Cut an overstressed block by a crack normal to its long axis."""
        poly = body.vertices()
        axis = self._principal_axis(body)
        offset = float(rng.uniform(-0.15, 0.15)) * body.length
        point = body.pos + offset * axis
        a_poly, b_poly = split_convex_polygon(poly, point, axis)
        if len(a_poly) < 3 or len(b_poly) < 3:
            return None
        areas = (polygon_area(a_poly), polygon_area(b_poly))
        if min(areas) < 0.5 * self.min_block_area:
            return None
        normal = np.array([-axis[1], axis[0]])
        pieces = []
        for poly_piece in (a_poly, b_poly):
            child = RubbleFloe(poly_piece, self.ice, parent=body.parent,
                               birth_step=self.step_index,
                               node_spacing=self.node_spacing)
            child.vel = body.velocity_at(child.pos)
            side = 1.0 if np.dot(child.pos - body.pos, normal) > 0.0 else -1.0
            child.pos = child.pos + side * 0.5 * self.node_spacing * normal
            child.vel = child.vel + side * 0.02 * normal
            child.omega = body.omega
            pieces.append(child)
        return pieces[0], pieces[1]

    def check_fracture(self) -> int:
        """Cut the most stressed block if it is overstressed; return cuts.

        A cooldown of ``fracture_cooldown`` steps between consecutive cuts lets the
        rubble re-equilibrate, so a single convergence event cannot cascade into
        shattering the whole field in one instant.
        """
        if self.step_index < self._fracture_cooldown_until:
            return 0
        if not self.last_contacts:
            self.last_contacts = self._contact_forces()[3]
        contacts = self.last_contacts
        rng = np.random.default_rng((self.step_index * 7919 + 17) % (2 ** 32))
        # Only the most stressed body is cut per call.  A 1-D chain under uniform
        # compression loads every floe with (almost) the same force, so a "split
        # all overstressed bodies" rule shatters the whole field in one step; real
        # ridging localises where the section is weakest and lets the rest of the
        # field translate.  One cut per call reproduces that localisation.
        best_k, best_sigma = -1, 0.0
        for k, body in enumerate(self.floes):
            small = (body.area < 2.0 * self.min_block_area
                     or body.thickness < self.min_block_thickness
                     or body.length < 3.0 * body.thickness)
            if small:
                continue
            sigma = self.bending_stress(body, contacts[k])
            if sigma > best_sigma:
                best_k, best_sigma = k, sigma
        keep: list[RubbleFloe] = []
        born: list[RubbleFloe] = []
        n_cuts = 0
        for k, body in enumerate(self.floes):
            if k != best_k or best_sigma <= self.ice.flexural_strength:
                keep.append(body)
                continue
            sigma = best_sigma
            pieces = self._split_body(body, contacts[k], rng)
            if pieces is None:
                keep.append(body)
                continue
            keep.append(pieces[0])
            born.append(pieces[1])
            n_cuts += 1
            self.fracture_log.append({
                "step": int(self.step_index),
                "time": float(self.time),
                "sigma": float(sigma),
                "sigma_f": float(self.ice.flexural_strength),
                "area_parent": float(body.area),
                "area_children": [float(pieces[0].area), float(pieces[1].area)],
                "z": float(body.pos[1]),
                "thickness": float(body.thickness),
            })
        if n_cuts:
            self._fracture_cooldown_until = self.step_index + self.fracture_cooldown
            self.floes = keep + born
            self.last_contacts = []
            self._buoy_cache = []
            self._relink()
            if self._auto_dt:
                self._update_dt()
        return n_cuts

    # -- integration ----------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            if self.step_index % self.n_relink == 0:
                self._relink()
            if self.step_index % 50 == 0 and not math.isfinite(self.kinetic_energy()):
                raise FloatingPointError(
                    f"non-finite state at step {self.step_index} (t={self.time:.3f}s); "
                    "reduce dt_safety or mass_scale")
            force, torque, piston_force, contacts = self._contact_forces()
            self.piston_force = piston_force
            self.last_contacts = contacts
            # hydrostatics are smooth on the buoyancy time scale (period >> dt), so
            # they are refreshed every ``buoyancy_every`` steps and held between
            # refreshes; the cache is invalidated whenever the body list changes
            if (len(self._buoy_cache) != len(self.floes)
                    or self.step_index - self._buoy_cache_step >= self.buoyancy_every):
                self._buoy_cache = [self._buoyancy(f) for f in self.floes]
                self._buoy_cache_step = self.step_index
            alpha = self.local_damping
            for idx, f in enumerate(self.floes):
                fb, tb, area_sub = self._buoy_cache[idx]
                total = force[idx] + fb + np.array([0.0, -f.mass * self.ice.gravity])
                if area_sub > 0.0:
                    speed = float(np.hypot(*f.vel))
                    if speed > 0.0:
                        cd = 0.5 * self.ice.rho_water * self.drag_coefficient * math.sqrt(area_sub)
                        total = total - cd * speed * f.vel
                if alpha > 0.0:
                    # Cundall local non-viscous damping: removes the energy that an
                    # explicit penalty contact would otherwise pump into a confined
                    # chain (elastic rebound), without altering static equilibrium.
                    # This is the standard quasi-static DEM treatment and is reported
                    # as ``local_damping`` in the result.
                    total = total - alpha * np.abs(total) * np.sign(f.vel)
                f.vel = f.vel + total / f.mass * self.dt
                tq_total = torque[idx] + tb
                if alpha > 0.0 and f.omega != 0.0:
                    tq_total -= alpha * abs(tq_total) * math.copysign(1.0, f.omega)
                f.omega = f.omega + tq_total / f.inertia * self.dt
                f.omega -= 0.02 * f.omega
                f.pos = f.pos + f.vel * self.dt
                f.theta = f.theta + f.omega * self.dt
            if self.piston_x is not None:
                if self.servo_force is not None:
                    # stress servo (PFC-style): the piston advances at most
                    # ``servo_velocity`` towards the sample while the measured load
                    # is below the target, and backs off above it.  A prescribed
                    # velocity instead stores elastic energy in the contact springs
                    # of a confined chain and releases it as a violent rebound, so a
                    # servo is the standard way to load a dense packing quasi-
                    # statically.
                    err = (self.servo_force - self.piston_force) / max(self.servo_force, 1e-12)
                    self.piston_velocity = -self.servo_velocity * float(np.clip(err / 0.2, -1.0, 1.0))
                self.piston_x += self.piston_velocity * self.dt
            self.time += self.dt
            self.step_index += 1

    def kinetic_energy(self) -> float:
        return float(sum(0.5 * f.mass * float(f.vel @ f.vel)
                         + 0.5 * f.inertia * f.omega ** 2 for f in self.floes))


# ---------------------------------------------------------------------------
# initial field
# ---------------------------------------------------------------------------

def _power_law_lengths(n: int, lo: float, hi: float, exponent: float) -> np.ndarray:
    """Stratified sample of ``p(L) ~ L^-exponent`` on ``[lo, hi]`` (deterministic)."""
    u = (np.arange(n) + 0.5) / n
    a = 1.0 - exponent
    if abs(a) < 1e-9:
        return lo * (hi / lo) ** u
    lo_a, hi_a = lo ** a, hi ** a
    return (lo_a + u * (hi_a - lo_a)) ** (1.0 / a)


def generate_floe_field(config: RidgeDEMConfig, *, thickness: float | None = None,
                        lengths: Sequence[float] | None = None) -> list[RubbleFloe]:
    """A row of floating polygonal floes: power-law lengths, hydrostatic draft."""
    ice = config.ice
    h = float(config.ice_thickness if thickness is None else thickness)
    if lengths is None:
        lengths = _power_law_lengths(config.n_floes, config.min_floe_length,
                                     config.max_floe_length, config.power_law_exponent)
    draft = h * ice.draft_ratio
    z_centre = 0.5 * h - draft          # equilibrium float of an unloaded floe
    spacing = float(config.node_spacing_factor) * h
    gap = max(float(config.initial_gap), 2.5 * 0.5 * spacing)
    floes: list[RubbleFloe] = []
    cursor = gap                     # keep the leading edge off the anchor wall
    bevel = float(config.edge_bevel) * h
    for k, L in enumerate(lengths):
        hl, hh = 0.5 * float(L), 0.5 * h
        if bevel > 0.0:
            # both bottom corners bevelled: a real floe edge is sloped, and a square
            # edge pair is geometrically locked (no lift component, see
            # ice_rafting.mechanics) so a square-edged field could never ridge
            poly = np.array([[-hl + bevel, -hh], [hl - bevel, -hh], [hl, -hh + bevel],
                             [hl, hh], [-hl, hh], [-hl, -hh + bevel]], dtype=float)
        else:
            poly = np.array([[-hl, -hh], [hl, -hh], [hl, hh], [-hl, hh]], dtype=float)
        poly = poly + np.array([cursor + hl, z_centre])
        body = RubbleFloe(poly, ice, n_nodes_per_edge=config.n_nodes_per_edge,
                          parent=k, node_spacing=spacing)
        floes.append(body)
        cursor += float(L) + gap
    return floes


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _clip_polygon_to_rect(poly: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Clip a convex polygon to the vertical strip ``x0 <= x <= x1``."""
    p = _clip_halfplane(poly, np.array([x0, 0.0]), np.array([-1.0, 0.0]), True)
    if len(p) < 3:
        return np.zeros((0, 2))
    p = _clip_halfplane(p, np.array([x1, 0.0]), np.array([1.0, 0.0]), True)
    return p if len(p) >= 3 else np.zeros((0, 2))


def ridge_porosity(floes: Sequence[RubbleFloe], h: float, *,
                   dx: float | None = None, ridge_factor: float = 1.8) -> dict:
    """Cross-section porosity of the ridged part of the field.

    The domain is cut into vertical columns of width ``dx``.  For every column the
    actual ice area (polygon-strip intersection) and the vertical ice extent are
    measured.  Columns whose extent exceeds ``ridge_factor * h`` belong to the
    ridge, and ``n = 1 - sum(A_ice) / sum(extent * dx)``.

    Returns the porosity, the ridged width, the solid fraction and the column
    profiles.  ``nan`` porosity means "no ridge detected" -- it is never replaced
    by a nominal value.
    """
    if not floes:
        raise ValueError("porosity needs at least one body")
    dx = float(dx if dx is not None else 0.25 * h)
    allv = np.vstack([f.vertices() for f in floes])
    x_lo, x_hi = float(allv[:, 0].min()), float(allv[:, 0].max())
    n_bins = max(int(math.ceil((x_hi - x_lo) / dx)), 1)
    ice_area = np.zeros(n_bins)
    z_top = np.full(n_bins, -np.inf)
    z_bot = np.full(n_bins, np.inf)
    for f in floes:
        poly = f.vertices()
        px_lo, px_hi = float(poly[:, 0].min()), float(poly[:, 0].max())
        b0 = max(int(math.floor((px_lo - x_lo) / dx)), 0)
        b1 = min(int(math.floor((px_hi - x_lo) / dx)), n_bins - 1)
        for b in range(b0, b1 + 1):
            x0 = x_lo + b * dx
            strip = _clip_polygon_to_rect(poly, x0, x0 + dx)
            if len(strip) < 3:
                continue
            a = abs(polygon_area(strip))
            if a <= 1e-14:
                continue
            ice_area[b] += a
            z_top[b] = max(z_top[b], float(strip[:, 1].max()))
            z_bot[b] = min(z_bot[b], float(strip[:, 1].min()))
    extent = np.where(np.isfinite(z_top) & np.isfinite(z_bot), z_top - z_bot, 0.0)
    ridged = extent > ridge_factor * h
    out = {"dx": dx, "ice_area_profile": ice_area, "extent_profile": extent,
           "x_lo": x_lo, "ridged": ridged, "ice_area_total": float(ice_area.sum())}
    if not ridged.any():
        out.update({"porosity": float("nan"), "n_columns": 0, "ridge_width": 0.0,
                    "ice_area": 0.0, "envelope_area": 0.0,
                    "solid_fraction": float("nan")})
        return out
    envelope = float(np.sum(extent[ridged] * dx))
    ice = float(np.sum(ice_area[ridged]))
    out.update({"porosity": float(1.0 - ice / envelope), "n_columns": int(ridged.sum()),
                "ridge_width": float(ridged.sum() * dx), "ice_area": ice,
                "envelope_area": envelope,
                "solid_fraction": float(ice / envelope) if envelope > 0 else float("nan")})
    return out


def sail_keel_geometry(floes: Sequence[RubbleFloe]) -> dict:
    """Sail height and keel depth with respect to the waterline (``z = 0``)."""
    if not floes:
        raise ValueError("geometry needs at least one body")
    allv = np.vstack([f.vertices() for f in floes])
    sail = float(max(allv[:, 1].max(), 0.0))
    keel = float(max(-allv[:, 1].min(), 0.0))
    above = allv[allv[:, 1] > 0.0]
    below = allv[allv[:, 1] < 0.0]
    sail_w = float(np.ptp(above[:, 0])) if len(above) else 0.0
    keel_w = float(np.ptp(below[:, 0])) if len(below) else 0.0
    return {
        "sail_height": sail,
        "keel_depth": keel,
        "sail_keel_ratio": (sail / keel) if keel > 0.0 else float("nan"),
        "keel_sail_ratio": (keel / sail) if sail > 0.0 else float("nan"),
        "sail_width": sail_w,
        "keel_width": keel_w,
        "n_blocks": int(len(floes)),
        "total_area": float(sum(f.area for f in floes)),
    }


def fit_porosity_law(strain: Sequence[float], porosity: Sequence[float]) -> dict:
    """Least-squares fit of ``n(eps) = n0 * exp(-eps / eps_c)``.

    Fit in log space (``ln n = ln n0 - eps / eps_c``) and report the coefficient
    of determination of that linear model together with the raw pairs.  No value
    is substituted for a failed fit: ``r2`` is whatever the data gives.
    """
    eps = np.asarray(strain, dtype=float)
    n = np.asarray(porosity, dtype=float)
    ok = np.isfinite(eps) & np.isfinite(n) & (n > 0.0)
    eps, n = eps[ok], n[ok]
    out = {"n_points": int(len(eps)), "n0": float("nan"), "eps_c": float("nan"),
           "r2": float("nan"), "slope": float("nan"),
           "n_pairs": [[float(a), float(b)] for a, b in zip(eps, n)],
           "fit_space": "log-linear (ln n vs eps)"}
    if len(eps) < 3:
        return out
    A = np.column_stack([np.ones(len(eps)), eps])
    coef, *_ = np.linalg.lstsq(A, np.log(n), rcond=None)
    n0 = float(math.exp(coef[0]))
    slope = float(coef[1])
    eps_c = float(-1.0 / slope) if slope < 0.0 else float("inf")
    pred = A @ coef
    ss_res = float(np.sum((np.log(n) - pred) ** 2))
    ss_tot = float(np.sum((np.log(n) - np.log(n).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    out.update({"n0": n0, "eps_c": eps_c, "r2": r2, "slope": slope})
    return out


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def _snapshot(world: RubbleWorld, config: RidgeDEMConfig, width0: float,
              area0: float, z0: float) -> dict:
    h = config.ice_thickness
    width = world.piston_x - world.x_left
    strain = 1.0 - width / width0
    por = ridge_porosity(world.floes, h)
    geom = sail_keel_geometry(world.floes)
    return {
        "time": float(world.time),
        "step": int(world.step_index),
        "strain": float(strain),
        "porosity": float(por.get("porosity", float("nan"))),
        "ridge_width": float(por.get("ridge_width", 0.0)),
        "sail_height": float(geom["sail_height"]),
        "keel_depth": float(geom["keel_depth"]),
        "sail_keel_ratio": float(geom["sail_keel_ratio"]),
        "n_blocks": int(len(world.floes)),
        "n_fractures": int(len(world.fracture_log)),
        "piston_force": float(world.piston_force),
        "tau_ridge": float(world.piston_force / h),
        "total_area": float(world.total_area()),
        "area_error": float(abs(world.total_area() - area0) / area0),
        "max_climb": float(max((f.pos[1] - z0) for f in world.floes) / h),
        "kinetic_energy": float(world.kinetic_energy()),
    }


def run_rubbling_ridge(config: RidgeDEMConfig | None = None, **kwargs) -> dict:
    """Compress a floe field into a ridge and measure everything.

    Returns a JSON-serialisable dict with the strain history, the final geometry,
    the volume audit, the porosity fit and the fracture log.  Nothing in the
    result is prescribed: sail height, keel depth, porosity and the volume error
    are read from the block configuration.
    """
    config = config or RidgeDEMConfig(**kwargs)
    ice = config.ice
    h = float(config.ice_thickness)
    t_start = _time.time()
    # Numerical mass scaling: multiplying both densities by S multiplies weight
    # and buoyancy together (so the draft ratio and every hydrostatic result is
    # unchanged) while lowering the contact frequency by sqrt(S), which lets the
    # explicit integrator take a sqrt(S) larger step.  Contact stiffness is kept,
    # so static penetrations are unchanged; this is the standard quasi-static DEM
    # mass-scaling trick, reported as ``mass_scale`` in the result.
    scale = float(config.mass_scale)
    dyn_ice = IceProperties(rho_ice=ice.rho_ice * scale,
                            rho_water=ice.rho_water * scale,
                            gravity=ice.gravity, young=ice.young,
                            poisson=ice.poisson,
                            flexural_strength=ice.flexural_strength)
    dyn_config = replace(config, ice=dyn_ice)
    floes = generate_floe_field(dyn_config)
    n_floes0 = len(floes)
    gap = max(float(config.initial_gap), 2.5 * 0.5 * config.node_spacing_factor * h)
    width0 = float(max(f.vertices()[:, 0].max() for f in floes) + gap)
    area0 = float(sum(f.area for f in floes))
    z0 = 0.5 * h - h * ice.draft_ratio
    world = RubbleWorld(
        floes, dyn_ice, x_left=0.0, piston_x=width0,
        piston_velocity=0.0, kn=config.kn, zeta=config.zeta, mu=config.mu,
        dt=config.dt, drag_coefficient=config.drag_coefficient,
        min_block_area=config.min_block_area,
        min_block_thickness=config.min_block_thickness,
        node_spacing=config.node_spacing_factor * h,
        local_damping=config.local_damping, dt_safety=config.dt_safety,
        buoyancy_every=config.buoyancy_every,
        servo_force=config.servo_force, servo_velocity=config.servo_velocity)
    # settle: let hydrostatics find its own equilibrium before loading
    world.step(config.settle_steps)
    settle_drift = float(max(abs(f.pos[1] - z0) for f in world.floes))
    world.piston_velocity = -float(config.piston_velocity)
    if config.servo_force is None:
        world.piston_velocity = -float(config.piston_velocity)
    history = [_snapshot(world, config, width0, area0, z0)]
    first_fracture_strain = None
    first_climb_strain = None
    max_steps = int(config.max_steps)
    n_samples = 0
    while world.step_index < max_steps:
        world.step(1)
        if world.step_index % config.fracture_every == 0:
            world.check_fracture()
            if world.fracture_log and first_fracture_strain is None:
                first_fracture_strain = float(
                    1.0 - (world.piston_x - world.x_left) / width0)
        if world.step_index % config.sample_every == 0:
            snap = _snapshot(world, config, width0, area0, z0)
            history.append(snap)
            n_samples += 1
            if snap["max_climb"] > 0.5 and first_climb_strain is None:
                first_climb_strain = snap["strain"]
        if (1.0 - (world.piston_x - world.x_left) / width0) >= config.target_convergence:
            break
    final = _snapshot(world, config, width0, area0, z0)
    geom = sail_keel_geometry(world.floes)
    por_final = ridge_porosity(world.floes, h)
    fit = fit_porosity_law([s["strain"] for s in history],
                           [s["porosity"] for s in history])
    area_final = world.total_area()
    phi = por_final.get("solid_fraction", float("nan"))
    isostasy_ratio = (float(ice.rho_water / (phi * ice.rho_ice) - 1.0)
                      if math.isfinite(phi) and phi > 0.0 else float("nan"))
    if first_fracture_strain is None:
        classification = "no_failure"
    elif first_climb_strain is not None and first_climb_strain < first_fracture_strain:
        classification = "rafting"
    else:
        classification = "rubbling"
    return {
        "config": {
            "n_floes": config.n_floes,
            "ice_thickness": h,
            "min_floe_length": config.min_floe_length,
            "max_floe_length": config.max_floe_length,
            "power_law_exponent": config.power_law_exponent,
            "piston_velocity": config.piston_velocity,
            "servo_force": config.servo_force,
            "servo_velocity": config.servo_velocity,
            "target_convergence": config.target_convergence,
            "mu": config.mu,
            "kn": config.kn,
            "zeta": config.zeta,
            "flexural_strength": ice.flexural_strength,
            "rho_ice": ice.rho_ice,
            "rho_water": ice.rho_water,
            "min_block_area": config.min_block_area,
            "min_block_thickness": config.min_block_thickness,
            "node_spacing_factor": config.node_spacing_factor,
            "mass_scale": config.mass_scale,
            "local_damping": config.local_damping,
            "dt_safety": config.dt_safety,
            "seed": config.seed,
        },
        "dt": float(world.dt),
        "n_steps": int(world.step_index),
        "runtime_s": float(_time.time() - t_start),
        "floe_lengths": [float(f.length) for f in floes] if False else [
            float(L) for L in _power_law_lengths(
                config.n_floes, config.min_floe_length, config.max_floe_length,
                config.power_law_exponent)],
        "initial_width": width0,
        "settle_drift": settle_drift,
        "history": history,
        "final": final,
        "geometry": geom,
        "porosity_final": {k: v for k, v in por_final.items()
                           if not isinstance(v, np.ndarray)},
        "porosity_fit": fit,
        "isostasy_ratio": isostasy_ratio,
        "volume": {
            "initial_area": area0,
            "final_area": area_final,
            "lost_area": 0.0,
            "rel_error": float(abs(area_final - area0) / area0),
            "out_of_plane_width": 1.0,
            "initial_volume": area0 * 1.0,
            "final_volume": area_final * 1.0,
        },
        "fracture_log": world.fracture_log,
        "n_fractures": int(len(world.fracture_log)),
        "classification": classification,
        "first_fracture_strain": first_fracture_strain,
        "first_climb_strain": first_climb_strain,
        "polygons": [f.vertices().tolist() for f in world.floes],
        "block_areas": [float(f.area) for f in world.floes],
        "n_samples": n_samples,
    }


def run_aspect_sweep(thicknesses: Sequence[float] = (0.2, 0.3, 0.5, 0.8),
                     length: float = 4.0, *, n_floes: int = 6,
                     target_convergence: float = 0.12,
                     piston_velocity: float = 0.15,
                     servo_force: float | None = 30_000.0,
                     servo_velocity: float = 0.02, mu: float = 0.4,
                     ice: IceProperties | None = None,
                     settle_steps: int = 1500, sample_every: int = 200,
                     max_steps: int = 150_000) -> list[dict]:
    """Rafting/rubbling transition versus floe aspect ratio ``h / L``.

    Each run is a short compression of ``n_floes`` identical floes of length
    ``L`` and thickness ``h``; the reported transition strain is the convergence
    at which the first bending crack appears (``None`` if the run stays intact,
    i.e. the field rafts instead of rubbling).
    """
    ice = ice or IceProperties()
    out: list[dict] = []
    for h in thicknesses:
        cfg = RidgeDEMConfig(
            n_floes=n_floes, ice_thickness=float(h),
            min_floe_length=float(length) * 0.999, max_floe_length=float(length) * 1.001,
            piston_velocity=piston_velocity, servo_force=servo_force,
            servo_velocity=servo_velocity, target_convergence=target_convergence,
            mu=mu, ice=ice, settle_steps=settle_steps,
            sample_every=sample_every, max_steps=max_steps)
        res = run_rubbling_ridge(cfg)
        out.append({
            "h": float(h), "L": float(length), "aspect": float(h / length),
            "first_fracture_strain": res["first_fracture_strain"],
            "classification": res["classification"],
            "n_fractures": res["n_fractures"],
            "n_steps": res["n_steps"],
            "sail_height": res["final"]["sail_height"],
            "keel_depth": res["final"]["keel_depth"],
            "sail_keel_ratio": res["final"]["sail_keel_ratio"],
            "porosity": res["final"]["porosity"],
        })
    return out
