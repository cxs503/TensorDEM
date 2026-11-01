"""Minimal vectorised soft-sphere granular DEM used by the ice-intake studies.

The module is deliberately self-contained (numpy only) and works in *consistent
units*: every quantity is expressed in the unit system the caller chooses
(SI for engineering tables, or ``r = rho = g = 1`` for scale-free
dimensionless studies).  Contact laws are the textbook linear spring-dashpot
normal model with a memoryless viscous-Coulomb tangential model and rolling
resistance:

    F_n = max(0, k_n * delta - c_n * v_n)
    F_t = -min(mu * F_n, c_t * |v_t|) * v_t_hat
    T   = -min(mu_r * r * F_n, I * |omega| / dt) * omega_hat

The tangential model is rate-dependent (no Cundall-Strack history), which keeps
the solver fully vectorised; see ``docs/ice_intake/CASE.md`` for the
limitations this implies for arch statistics.

Contact search uses a uniform cell grid with the classic
"sort / bincount / repeat" vectorisation, so a 10^5-particle configuration can
be advanced on a single CPU core without Python-level loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "GranularConfig",
    "PlaneWall",
    "BoxWalls",
    "HoledFloor",
    "FunnelWalls2D",
    "GranularSystem",
    "hex_lattice",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GranularConfig:
    """Contact/derived properties of a monodisperse granular assembly.

    ``dimensionality`` only changes the mass/inertia definition (solid sphere
    vs. disk of thickness ``2 r``) and the mass of a particle; the dynamics
    always run in 3-D with the third coordinate pinned for planar cases.
    """

    radius: float
    density: float = 1.0
    normal_stiffness: float = 2.0e4
    normal_damping_ratio: float = 0.30
    tangential_damping_ratio: float = 0.30
    friction: float = 0.35
    rolling_friction: float = 0.05
    gravity: float = 1.0
    dimensionality: int = 3

    def __post_init__(self) -> None:
        if not (self.radius > 0):
            raise ValueError("radius must be positive")
        if not (self.density > 0):
            raise ValueError("density must be positive")
        if not (self.normal_stiffness > 0):
            raise ValueError("normal_stiffness must be positive")
        if self.friction < 0 or self.rolling_friction < 0:
            raise ValueError("friction coefficients must be non-negative")
        if self.dimensionality not in (2, 3):
            raise ValueError("dimensionality must be 2 or 3")

    # -- derived -----------------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def volume(self) -> float:
        if self.dimensionality == 2:
            # disk of thickness 2 r
            return math.pi * self.radius**2 * (2.0 * self.radius)
        return 4.0 / 3.0 * math.pi * self.radius**3

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def inertia_factor(self) -> float:
        """I / (m r^2): 2/5 for a solid sphere, 1/2 for a disk."""
        return 0.5 if self.dimensionality == 2 else 0.4

    def recommended_dt(self, n_contacts: int = 6) -> float:
        """Stable explicit time step from a frequency bound.

        For a linear spring system with mass ``m`` and up to ``n_contacts``
        simultaneous contacts the highest eigenfrequency is bounded by
        ``sqrt(2 n m_eff ... )``; a conservative estimate
        ``omega_max = 2 sqrt(n k_n / m)`` has proven adequate for the
        benchmarks in ``tests/test_granular.py`` (energy stays bounded and
        wall penetration is below 1e-3 r).
        """
        omega = 2.0 * math.sqrt(n_contacts * self.normal_stiffness / self.mass)
        return 0.05 * 2.0 * math.pi / omega


# ---------------------------------------------------------------------------
# walls
# ---------------------------------------------------------------------------


class _Wall:
    """Interface for rigid boundary contributions."""

    def forces(self, x, v, w, r, cfg: GranularConfig):
        raise NotImplementedError


class PlaneWall(_Wall):
    """Half-space boundary ``n . x >= offset`` (``n`` points into the domain).

    ``mask`` optionally selects the particles the wall acts on, which is how
    the square outlet of the silo floor is carved out of an infinite plane.
    """

    def __init__(self, normal, offset: float, mask=None):
        n = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("zero normal")
        self.normal = n / norm
        self.offset = float(offset) * norm
        self.mask = mask
        self._dt = None

    def _penetration(self, x, r):
        s = x @ self.normal - self.offset
        if self.mask is not None:
            active = self.mask(x)
        else:
            active = np.ones(x.shape[0], dtype=bool)
        return r - s, active

    def forces(self, x, v, w, r, cfg: GranularConfig):
        delta, active = self._penetration(x, r)
        active = active & (delta > 0.0)
        return _contact_response(
            x, v, w, r, cfg, self.normal, delta, active, wall=True, dt=self._dt
        )


class BoxWalls(_Wall):
    """Axis-aligned box: closed in all directions where ``open`` is False."""

    def __init__(self, lower, upper, open_dirs=()):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if np.any(self.upper <= self.lower):
            raise ValueError("upper must exceed lower")
        self.open_dirs = tuple(open_dirs)
        walls = []
        for axis in range(3):
            if ("-", axis) not in self.open_dirs:
                normal = np.zeros(3)
                normal[axis] = 1.0
                walls.append(PlaneWall(normal, self.lower[axis]))
            if ("+", axis) not in self.open_dirs:
                normal = np.zeros(3)
                normal[axis] = -1.0
                walls.append(PlaneWall(normal, -self.upper[axis]))
        self.walls = walls

    def forces(self, x, v, w, r, cfg: GranularConfig):
        return _combine(
            [wall.forces(x, v, w, r, cfg) for wall in self.walls], x.shape[0]
        )


def _combine(results, n):
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    for f, t in results:
        force += f
        torque += t
    return force, torque


class HoledFloor(PlaneWall):
    """Horizontal floor at ``z = z0`` with a square/rectangular outlet."""

    def __init__(self, z0: float = 0.0, half_x: float = 1.0, half_y: float = 1.0):
        self.half_x = float(half_x)
        self.half_y = float(half_y)
        self.z0 = float(z0)

        def mask(x):
            inside = (np.abs(x[:, 0]) <= self.half_x) & (np.abs(x[:, 1]) <= self.half_y)
            return ~inside

        super().__init__((0.0, 0.0, 1.0), z0, mask=mask)


class FunnelWalls2D(_Wall):
    """Planar funnel (2-D) with an outlet of width ``aperture``.

    ``inclination`` is the angle of the wall measured from the horizontal:
    90 deg gives vertical walls (a silo), smaller angles give a converging
    funnel.  Both walls are infinite half-spaces of the form
    ``+-x sin(theta) + z cos(theta) >= -(aperture/2) sin(theta)``.
    """

    def __init__(self, aperture: float, inclination_deg: float, z_out: float = 0.0):
        theta = math.radians(inclination_deg)
        s, c = math.sin(theta), math.cos(theta)
        if s <= 0:
            raise ValueError("inclination must be in (0, 90] degrees")
        self.aperture = aperture
        self.inclination_deg = inclination_deg
        self.z_out = z_out
        self.left = PlaneWall((s, 0.0, c), -0.5 * aperture * s)
        self.right = PlaneWall((-s, 0.0, c), -0.5 * aperture * s)

    def forces(self, x, v, w, r, cfg: GranularConfig):
        return _combine([self.left.forces(x, v, w, r, cfg),
                         self.right.forces(x, v, w, r, cfg)], x.shape[0])


# ---------------------------------------------------------------------------
# contact response shared by particle-particle and particle-wall pairs
# ---------------------------------------------------------------------------


def _contact_response(x, v, w, r, cfg, normal, delta, active, wall=False, dt=None):
    """Return (force, torque) arrays for the active subset only.

    For particle-particle pairs ``x`` contains both partners interleaved
    (even index = i, odd = j); for walls ``x`` holds the particle centres and
    ``normal`` is broadcast.
    """
    n_p = x.shape[0]
    force = np.zeros((n_p, 3))
    torque = np.zeros((n_p, 3))
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return force, torque

    if wall:
        xi = x[idx]
        vi = v[idx]
        wi = w[idx]
        ri = r[idx]
        nj = normal if normal.ndim == 1 else normal[idx]
        nj = np.atleast_2d(nj)
        d = delta[idx][:, None]
        v_contact = vi + np.cross(wi, ri[:, None] * nj)
        vn = np.einsum("ij,ij->i", v_contact, nj)
        vt = v_contact - vn[:, None] * nj
        vt_mag = np.linalg.norm(vt, axis=1)

        cn = 2.0 * cfg.normal_damping_ratio * math.sqrt(cfg.normal_stiffness * cfg.mass)
        fn = cfg.normal_stiffness * d[:, 0] - cn * vn
        fn = np.maximum(fn, 0.0)
        ct = 2.0 * cfg.tangential_damping_ratio * math.sqrt(cfg.normal_stiffness * cfg.mass)
        ft_mag = np.minimum(cfg.friction * fn, ct * vt_mag)
        ft_hat = np.zeros_like(vt)
        nz = vt_mag > 1e-300
        ft_hat[nz] = vt[nz] / vt_mag[nz, None]
        total = fn[:, None] * nj - ft_mag[:, None] * ft_hat
        force[idx] = total
        torque[idx] = np.cross(-ri[:, None] * nj, total)
        torque[idx] += _rolling_torque(fn, wi, ri, cfg, dt)
        return force, torque

    # particle-particle (i: even, j: odd)
    xi = x[idx[0::2]]
    xj = x[idx[1::2]]
    vi = v[idx[0::2]]
    vj = v[idx[1::2]]
    wi = w[idx[0::2]]
    wj = w[idx[1::2]]
    ri = r[idx[0::2]]
    rj = r[idx[1::2]]
    dvec = xj - xi
    dist = np.linalg.norm(dvec, axis=1)
    nvec = dvec / dist[:, None]
    v_rel = (vj + np.cross(wj, -rj[:, None] * nvec)) - (
        vi + np.cross(wi, ri[:, None] * nvec)
    )
    vn = np.einsum("ij,ij->i", v_rel, nvec)
    vt = v_rel - vn[:, None] * nvec
    vt_mag = np.linalg.norm(vt, axis=1)

    cn = 2.0 * cfg.normal_damping_ratio * math.sqrt(
        cfg.normal_stiffness * 0.5 * cfg.mass
    )
    # every pair passed here is active, so ``delta`` is already compact
    fn = cfg.normal_stiffness * delta - cn * vn
    fn = np.maximum(fn, 0.0)
    ct = 2.0 * cfg.tangential_damping_ratio * math.sqrt(
        cfg.normal_stiffness * 0.5 * cfg.mass
    )
    ft_mag = np.minimum(cfg.friction * fn, ct * vt_mag)
    ft_hat = np.zeros_like(vt)
    nz = vt_mag > 1e-300
    ft_hat[nz] = vt[nz] / vt_mag[nz, None]
    total = fn[:, None] * nvec - ft_mag[:, None] * ft_hat

    force[idx[0::2]] += -total
    force[idx[1::2]] += total
    torque[idx[0::2]] += np.cross(ri[:, None] * nvec, -total)
    torque[idx[1::2]] += np.cross(-rj[:, None] * nvec, total)
    t_roll_i = _rolling_torque(fn, wi, ri, cfg, dt)
    t_roll_j = _rolling_torque(fn, wj, rj, cfg, dt)
    torque[idx[0::2]] += t_roll_i
    torque[idx[1::2]] += t_roll_j
    return force, torque


def _rolling_torque(fn, omega, radius, cfg, dt=None):
    """Rolling resistance, capped so the discrete update can never reverse spin.

    Without the ``I |omega| / dt`` cap an explicit rolling torque injects
    energy once a particle is almost at rest; the cap makes the scheme
    unconditionally dissipative (see ``tests/test_granular.py``).
    """
    mag = cfg.rolling_friction * radius * fn
    om = np.linalg.norm(omega, axis=1)
    if dt is not None and dt > 0:
        inertia = cfg.inertia_factor * cfg.mass * radius**2
        mag = np.minimum(mag, inertia * om / dt)
    scale = np.zeros_like(om)
    nz = om > 1e-300
    scale[nz] = -mag[nz] / om[nz]
    return scale[:, None] * omega


# ---------------------------------------------------------------------------
# neighbour search
# ---------------------------------------------------------------------------


_OFFSETS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if not (dx < 0 or (dx == 0 and (dy < 0 or (dy == 0 and dz < 0))))],
    dtype=np.int64,
)


def _cell_pairs(x: np.ndarray, r_max: float, lo: np.ndarray, hi: np.ndarray,
                cell: float, skin: float):
    """Return candidate pair indices ``(i, j)`` (each unordered pair once).

    All 14 half-stencil offsets are handled in a single vectorised pass: the
    ``(n, 14)`` table of target cells is flattened into one ``repeat``/
    ``cumsum`` gather, which keeps the Python-level numpy call count at a
    handful per time step instead of one set per offset.
    """
    n = x.shape[0]
    dims = np.maximum(np.floor((hi - lo) / cell).astype(np.int64) + 1, 1)
    c = np.floor((x - lo) / cell).astype(np.int64)
    np.clip(c, 0, dims - 1, out=c)
    lin = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    order = np.argsort(lin, kind="stable")
    n_cell = int(dims[0] * dims[1] * dims[2])
    counts = np.bincount(lin[order], minlength=n_cell)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # --- self cell: keep i < j only ---------------------------------------
    out_i, out_j = [], []
    self_cnt = counts[lin]
    if self_cnt.sum() > 1:
        i_idx = np.repeat(np.arange(n), self_cnt)
        cum = np.cumsum(self_cnt) - self_cnt
        j_idx = order[np.repeat(starts[lin], self_cnt)
                      + (np.arange(i_idx.size) - np.repeat(cum, self_cnt))]
        keep = i_idx < j_idx
        out_i.append(i_idx[keep])
        out_j.append(j_idx[keep])

    # --- 13 neighbour offsets in one pass ---------------------------------
    offs = _OFFSETS[1:]  # drop the self offset
    tc = c[:, None, :] + offs[None, :, :]
    valid = np.all((tc >= 0) & (tc < dims), axis=2)
    tlin = np.where(valid,
                    (tc[..., 0] * dims[1] + tc[..., 1]) * dims[2] + tc[..., 2], 0)
    cnt = np.where(valid, counts[tlin], 0).ravel()
    sel = cnt > 0
    if sel.any():
        cnt = cnt[sel]
        tlin = tlin.ravel()[sel]
        owner = np.repeat(np.arange(n), offs.shape[0])[sel]
        total = int(cnt.sum())
        i_idx = np.repeat(owner, cnt)
        cum = np.cumsum(cnt) - cnt
        j_idx = order[np.repeat(starts[tlin], cnt)
                      + (np.arange(total) - np.repeat(cum, cnt))]
        out_i.append(i_idx)
        out_j.append(j_idx)

    i_all = np.concatenate(out_i) if out_i else np.empty(0, dtype=np.int64)
    j_all = np.concatenate(out_j) if out_j else np.empty(0, dtype=np.int64)
    if i_all.size == 0:
        return i_all, j_all

    d = x[j_all] - x[i_all]
    keep = np.einsum("ij,ij->i", d, d) < (2.0 * r_max + skin) ** 2
    i_all, j_all = i_all[keep], j_all[keep]
    # neighbour-cell pairs come out in arbitrary order; normalise to i < j so
    # the output is deterministic and symmetric diagnostics are trivial
    return np.minimum(i_all, j_all), np.maximum(i_all, j_all)


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------


class GranularSystem:
    """Monodisperse soft-sphere assembly with rigid walls."""

    def __init__(self, x, cfg: GranularConfig, walls=(), skin_fraction: float = 0.1):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must be (N, 3)")
        self.cfg = cfg
        self.x = x.copy()
        self.v = np.zeros_like(self.x)
        self.w = np.zeros_like(self.x)
        self.r = np.full(self.x.shape[0], cfg.radius)
        self.m = np.full(self.x.shape[0], cfg.mass)
        self.walls = list(walls)
        self.skin = skin_fraction * cfg.radius
        self.removed_mass = 0.0
        self.removed_indices: list[int] = []
        self.initial_mass = float(self.m.sum())
        self.initial_particles = int(self.x.shape[0])
        self.steps = 0
        self._g = np.array([0.0, 0.0, -cfg.gravity])
        self._dt = None

    # -- helpers -----------------------------------------------------------
    def _bounds(self):
        pad = 4.0 * self.cfg.radius
        lo = self.x.min(axis=0) - pad
        hi = self.x.max(axis=0) + pad
        return lo, hi

    def forces(self):
        cfg = self.cfg
        force = np.zeros_like(self.x)
        torque = np.zeros_like(self.x)
        force += self.m[:, None] * self._g
        for wall in self.walls:
            f, t = wall.forces(self.x, self.v, self.w, self.r, cfg)
            force += f
            torque += t
        lo, hi = self._bounds()
        cell = 2.0 * cfg.radius + self.skin
        i, j = _cell_pairs(self.x, cfg.radius, lo, hi, cell, self.skin)
        if i.size:
            dvec = self.x[j] - self.x[i]
            dist = np.linalg.norm(dvec, axis=1)
            overlap = 2.0 * cfg.radius - dist
            near = overlap > 0.0
            if near.any():
                i, j = i[near], j[near]
                inter = np.empty(2 * i.size, dtype=np.int64)
                inter[0::2] = i
                inter[1::2] = j
                f, t = _contact_response(
                    self.x[inter], self.v[inter], self.w[inter], self.r[inter],
                    cfg, None, overlap[near],
                    np.ones(2 * i.size, dtype=bool), dt=self._dt,
                )
                # scatter: contact indices repeat, so accumulate with bincount
                _scatter_add(force, i, f[0::2])
                _scatter_add(force, j, f[1::2])
                _scatter_add(torque, i, t[0::2])
                _scatter_add(torque, j, t[1::2])
        return force, torque

    def _broadcast_dt(self, dt):
        self._dt = dt
        for wall in self.walls:
            for sub in getattr(wall, "walls", [wall]):
                sub._dt = dt
            for sub in (getattr(wall, "left", None), getattr(wall, "right", None)):
                if sub is not None:
                    sub._dt = dt

    def step(self, dt: float, drag: float = 0.0):
        self._broadcast_dt(dt)
        f, t = self.forces()
        self.v += dt * f / self.m[:, None]
        if drag > 0:
            self.v *= (1.0 - drag * dt)
        inertia = self.cfg.inertia_factor * self.m * self.cfg.radius**2
        self.w += dt * t / inertia[:, None]
        self.x += dt * self.v
        self.steps += 1

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.m * np.einsum("ij,ij->i", self.v, self.v)))

    def potential_energy(self) -> float:
        return float(np.sum(self.m * self.cfg.gravity * self.x[:, 2]))

    def mass_residual(self) -> float:
        """Signed ledger residual: (present + removed) - initial."""
        return float(self.m.sum() + self.removed_mass - self.initial_mass)

    def max_wall_penetration(self) -> float:
        worst = 0.0
        for wall in self.walls:
            if isinstance(wall, PlaneWall):
                s = self.x @ wall.normal - wall.offset
                if wall.mask is not None:
                    act = wall.mask(self.x)
                else:
                    act = np.ones(self.x.shape[0], dtype=bool)
                pen = np.where(act, self.r - s, -np.inf)
                worst = max(worst, float(pen.max()) if pen.size else 0.0)
            elif isinstance(wall, BoxWalls):
                for sub in wall.walls:
                    s = self.x @ sub.normal - sub.offset
                    worst = max(worst, float((self.r - s).max()))
            elif isinstance(wall, FunnelWalls2D):
                for sub in (wall.left, wall.right):
                    s = self.x @ sub.normal - sub.offset
                    worst = max(worst, float((self.r - s).max()))
        return worst

    def remove(self, index: np.ndarray):
        if index.size == 0:
            return
        self.removed_mass += float(self.m[index].sum())
        self.removed_indices.extend(int(k) for k in index)
        keep = np.ones(self.x.shape[0], dtype=bool)
        keep[index] = False
        self.x = self.x[keep]
        self.v = self.v[keep]
        self.w = self.w[keep]
        self.r = self.r[keep]
        self.m = self.m[keep]

    # -- diagnostics -------------------------------------------------------
    def summary(self) -> dict:
        return {
            "particles": int(self.x.shape[0]),
            "initial_particles": self.initial_particles,
            "steps": self.steps,
            "kinetic_energy": self.kinetic_energy(),
            "potential_energy": self.potential_energy(),
            "mass_residual": self.mass_residual(),
            "removed_mass": self.removed_mass,
            "max_wall_penetration": self.max_wall_penetration(),
            "finite": bool(np.all(np.isfinite(self.x))),
        }


def _scatter_add(target: np.ndarray, index: np.ndarray, values: np.ndarray):
    """``target[index] += values`` that also works with repeated indices."""
    n = target.shape[0]
    for comp in range(target.shape[1]):
        target[:, comp] += np.bincount(index, weights=values[:, comp], minlength=n)


# ---------------------------------------------------------------------------
# packing helpers
# ---------------------------------------------------------------------------


def hex_lattice(count: int, radius: float, spacing: float | None = None,
                rng: np.random.Generator | None = None, jitter: float = 0.02,
                planar: bool = False, layers: int | None = None) -> np.ndarray:
    """Return ``count`` centres on a close-packed (jittered) lattice.

    ``planar=True`` lays the particles out in the x-z plane (y = 0), which is
    what the 2-D funnel study needs; otherwise an fcc-like stacking is used.
    """
    rng = rng or np.random.default_rng(0)
    d = spacing if spacing is not None else 2.05 * radius
    pts = []
    if planar:
        nz = layers or max(1, int(math.ceil(math.sqrt(count))))
        ny = max(1, int(math.ceil(count / nz)))
        for k in range(nz):
            for j in range(ny):
                pts.append([(j + 0.5 * (k % 2)) * d, 0.0, (k + 0.5) * d * math.sqrt(3) / 2])
        pts = np.array(pts)
    else:
        # simple-cubic lattice with lattice constant ``spacing`` (>= 2 r), so the
        # initial configuration is strictly overlap free
        n_side = max(1, int(math.ceil(count ** (1.0 / 3.0))))
        for k in range(n_side):
            for j in range(n_side):
                for i in range(n_side):
                    pts.append([i * d, j * d, k * d])
        pts = np.array(pts)
    if pts.shape[0] < count:
        raise ValueError("lattice smaller than requested count")
    pts = pts[:count]
    if jitter:
        pts = pts + rng.normal(scale=jitter * radius, size=pts.shape)
    return pts