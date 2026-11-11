"""Ice-floe pile-up against a vertical structure: DEM load history, ISO 19906
benchmark and porosity-resistance scaling law (campaign p227w2-N04).

Physical problem
----------------
A drifting broken-ice field (thickness ``h``, compressive strength ``sigma_c``,
drift speed ``V``) is driven against a fixed vertical structure of characteristic
width ``D`` (cylinder diameter or square column side).  Ice floes are pushed
against the structure, crush at the contacts, and pile up in front of it.  The
horizontal load transmitted through the rubble pile is recorded as a time
history; its peak ``F_max`` and its steady (rubble-jam) level ``F_steady`` are
compared with ISO 19906 and used to test the porosity-resistance law
``F ~ (1 - n)**alpha``.

Model
-----
A **2-D plan-view (depth-averaged) soft-disc DEM**.  Each floe is a vertical-faced
disc of radius ``r_i`` and thickness ``h``:

* inertia is realistic: ``m_i = rho_ice * pi r_i**2 * h``;
* ice-ice and ice-structure contacts use a **crushing-limited normal law**
  ``Fn = min(k_n * delta, sigma_c * h * b(delta))``, where ``b(delta)`` is the
  chord of the contact zone (a growing contact area is what makes crushing
  pressure -- and therefore the structure load -- scale with ``sigma_c * h``,
  not with an arbitrary spring stiffness);
* Coulomb friction with coefficient ``mu`` on all contacts;
* quadratic hydrodynamic drag from still water;
* the ice field is driven by a **kinematic platen** (the far-field ice sheet)
  advancing at the drift speed ``V`` from upstream; its travel is bounded
  (``platen_travel_factor * D``) because in a 2-D plan view the platen's swept
  area has nowhere to go -- the sheet's third dimension (rubble thickening /
  ride-up) is not resolved;
* the structure sits in a **confining channel** (rough walls at ``y = +-y_half``)
  closed downstream, so the jam is retained and the load can build to the
  crushing limit; an *open* channel lets the sheet simply flow past the
  structure and produces no rubble load at all (checked, see the campaign doc);
* a floe that penetrates a wall/structure beyond the crushable-zone depth
  (``0.2 r_eff``) is either stopped by a stiff core (default) or eroded
  (``comminution=True``: the crushed surface layer is destroyed and tracked as
  a mass-loss audit quantity).

The rubble-pile porosity is **measured from the simulated packing** (binary
mask of the discs -> morphological closing/fill envelope -> void fraction), not
prescribed.  It is a 2-D plan-view (area) porosity; see
``docs/p227w2_N04_vertical_structure_pileup.md`` for the mapping caveat to the
3-D rubble porosity of a prototype ridge.

ISO 19906 reference load
------------------------
ISO 19906:2010 global ice pressure on a vertical structure:

    p_G = C_R * (h / h_ref)**n * (w / h)**m ,      p_G <= sigma_c
    F   = p_G * w * h

with ``C_R = 2.8 MPa``, ``h_ref = 1 m``, ``n = -0.5 + h/10`` (not below -0.25),
``m = -0.16`` (not below -0.10) and ``w`` the contact width.  Written in the
form used by this campaign, ``F = C * sigma_c * h**2`` with

    C = (p_G / sigma_c) * (w / h) ,

so the "C" of the campaign formula is not a free knob: it is fixed by the ISO
pressure factors, the structure width and the ice thickness.

Units: SI throughout (m, kg, s, N, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = [
    "RHO_ICE",
    "RHO_WATER",
    "GRAVITY",
    "ISO19906Result",
    "iso19906_load",
    "PileupConfig",
    "PileupResult",
    "make_floe_field",
    "run_pileup_case",
    "measure_pile_porosity",
    "fit_porosity_scaling",
    "load_statistics",
]

RHO_ICE = 900.0      # kg/m3
RHO_WATER = 1025.0   # kg/m3
GRAVITY = 9.81       # m/s2
CD_WATER = 1.2       # drag coefficient of a bluff body in a current


# ---------------------------------------------------------------------------
# ISO 19906 reference load
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ISO19906Result:
    """ISO 19906 nominal global ice load on a vertical structure."""

    sigma_c: float
    h: float
    w_contact: float
    p_G: float          # global ice pressure, Pa
    F_ISO: float        # N
    C_task: float       # coefficient of the campaign form F = C sigma_c h^2
    n_exp: float
    m_exp: float
    pressure_capped: bool

    def as_dict(self) -> dict:
        return asdict(self)


def iso19906_load(
    sigma_c: float,
    h: float,
    D: float,
    shape: str = "cylinder",
    h_ref: float = 1.0,
    C_R: float = 2.8e6,
    contact_width: float | None = None,
) -> ISO19906Result:
    """ISO 19906 global ice pressure and horizontal load on a vertical structure.

    Parameters
    ----------
    sigma_c : uniaxial compressive strength of ice [Pa]
    h : ice thickness [m]
    D : structure width (diameter for a cylinder, side for a square column) [m]
    shape : ``"cylinder"`` or ``"square"`` -- only the contact-width convention
        differs: the full projected width ``D`` is used for both, which is the
        ISO narrow-structure convention.
    contact_width : optional override of the contact width ``w`` [m].
    """
    if sigma_c <= 0 or h <= 0 or D <= 0:
        raise ValueError("sigma_c, h and D must be positive")
    if shape not in ("cylinder", "square"):
        raise ValueError(f"unsupported structure shape: {shape!r}")

    w = float(contact_width if contact_width is not None else D)

    n_exp = max(-0.5 + h / 10.0, -0.25)
    m_exp = max(-0.16, -0.10)  # ISO limit; kept explicit for the audit trail
    m_exp = -0.16

    p_G = C_R * (h / h_ref) ** n_exp * (w / h) ** m_exp
    capped = bool(p_G > sigma_c)
    p_G = min(p_G, sigma_c)

    F = p_G * w * h
    C_task = (p_G / sigma_c) * (w / h)
    return ISO19906Result(
        sigma_c=float(sigma_c),
        h=float(h),
        w_contact=w,
        p_G=float(p_G),
        F_ISO=float(F),
        C_task=float(C_task),
        n_exp=float(n_exp),
        m_exp=float(m_exp),
        pressure_capped=capped,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PileupConfig:
    """Configuration of one 2-D plan-view ice pile-up DEM run."""

    sigma_c: float = 3.0e6     # Pa
    h: float = 1.0             # m
    D: float = 5.0             # m
    shape: str = "cylinder"    # "cylinder" | "square"
    drift_speed: float = 0.5   # m/s
    duration: float = 9.0      # s of physical time
    seed: int = 7
    # floe size distribution (lognormal, truncated)
    floe_radius_median: float = 0.40          # m
    floe_radius_sigma_ln: float = 0.30
    floe_radius_bounds: tuple[float, float] = (0.22, 0.75)
    # ice supply: initial plan-view ice concentration of the upstream field
    ice_concentration: float = 0.78
    # contact parameters
    mu: float = 0.4                 # ice-ice / ice-steel friction coefficient
    k_lin_factor: float = 30.0      # k_n = k_lin_factor * sigma_c * h  [N/m]
    damping_ratio: float = 0.8
    # geometric layout
    field_length_factor: float = 1.3      # upstream field length = factor * D
    downstream_length_factor: float = 0.6  # downstream field length = factor * D
    lateral_factor: float = 0.80          # channel half-width = factor * D
    lateral_wall: bool = True             # confining channel walls at y=+-y_half
    downstream_wall: bool = True          # closed downstream end
    comminution: bool = False             # structure contact consumes crushed ice
    platen_travel_factor: float = 0.50    # platen advance limit = factor * D
    grid_cell: float = 0.03               # porosity grid [m]
    sample_dt: float = 0.002              # load-history sampling [s]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["floe_radius_bounds"] = list(self.floe_radius_bounds)
        return d


# ---------------------------------------------------------------------------
# Floe field generation
# ---------------------------------------------------------------------------
def _lognormal_radii(n: int, cfg: PileupConfig, rng: np.random.Generator) -> np.ndarray:
    lo, hi = cfg.floe_radius_bounds
    out = np.empty(0)
    while out.size < n:
        draw = rng.lognormal(math.log(cfg.floe_radius_median), cfg.floe_radius_sigma_ln,
                             size=4 * n)
        draw = draw[(draw >= lo) & (draw <= hi)]
        out = np.concatenate([out, draw])
    return out[:n]


def _structure_mask(x: np.ndarray, y: np.ndarray, cfg: PileupConfig,
                    inflate: float = 0.0) -> np.ndarray:
    """True where a point lies inside the vertical structure footprint."""
    if cfg.shape == "cylinder":
        return np.hypot(x, y) <= 0.5 * cfg.D + inflate
    a = 0.5 * cfg.D
    dx = np.abs(x) - a
    dy = np.abs(y) - a
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside_dist = np.minimum(np.maximum(dx, dy), 0.0)
    return (outside + inside_dist) <= inflate


def make_floe_field(cfg: PileupConfig, rng: np.random.Generator):
    """Build the upstream broken-ice sheet and return ``(pos, rad, concentration)``.

    A real drifting ice sheet is *continuous*: the plan-view concentration is
    high (0.7-1.0) and the open water is limited to narrow leads between floes.
    The field is therefore generated as a **jittered hexagonal floe mosaic**:
    floe centres sit on a triangular lattice whose spacing ``s`` is chosen so
    that ``pi <r^2> / (s^2 sqrt(3)/2)`` equals ``cfg.ice_concentration``;
    radii are lognormal draws truncated at ``0.5 s`` so that the mosaic is
    overlap-free, and centres are jittered by +/-5 % of ``s``.

    This gives the platen a force-transmitting (already jammed) sheet to push,
    which is the correct initial condition for a pile-up problem -- a dilute
    cloud of floes would simply be herded without generating rubble.
    """
    x_lo = -0.5 * cfg.D - cfg.D * cfg.field_length_factor
    x_hi = 0.5 * cfg.D + cfg.downstream_length_factor * cfg.D
    y_half = cfg.lateral_factor * cfg.D
    band_area = (x_hi - x_lo) * 2.0 * y_half
    phi = float(np.clip(cfg.ice_concentration, 0.05, 0.86))

    r_med = cfg.floe_radius_median
    # lattice pitch: 1.15 median radii fit inside a hexagon half-pitch, i.e. the
    # floes almost touch (a continuous sheet with narrow leads)
    spacing = 2.02 * r_med
    r_cap = 0.49 * spacing
    dx = spacing
    dy = spacing * math.sqrt(3.0) / 2.0
    xs: list[float] = []
    ys: list[float] = []
    rs: list[float] = []
    n_rows = int(2.0 * y_half / dy) + 2
    n_cols = int((x_hi - x_lo) / dx) + 2
    n_sites = 0
    for j in range(n_rows):
        y0 = -y_half - dy + j * dy
        for i in range(n_cols):
            x0 = x_lo - dx + i * dx + (0.5 * dx if j % 2 else 0.0)
            x = x0 + rng.uniform(-0.05, 0.05) * spacing
            y = y0 + rng.uniform(-0.05, 0.05) * spacing
            r = float(min(rng.lognormal(math.log(r_med), cfg.floe_radius_sigma_ln),
                          r_cap))
            if x + r > x_hi or x - r < x_lo or abs(y) + r > y_half:
                continue
            if _structure_mask(np.array([x]), np.array([y]), cfg, inflate=r)[0]:
                continue
            n_sites += 1
            xs.append(x)
            ys.append(y)
            rs.append(r)

    pos = np.column_stack([xs, ys]) if xs else np.zeros((0, 2))
    rad = np.asarray(rs, dtype=float)
    # local (in-band) sheet concentration: covered area / mosaic cell area
    band_area = max(n_sites, 1) * (dx * dy)
    realised = float(np.sum(math.pi * rad ** 2) / band_area) if rad.size else 0.0
    # thin the sheet with random open leads down to the requested concentration
    if rad.size and realised > phi:
        keep_n = max(20, int(round(phi / realised * rad.size)))
        idx = np.sort(rng.choice(rad.size, size=min(keep_n, rad.size), replace=False))
        pos, rad = pos[idx], rad[idx]
        realised = float(np.sum(math.pi * rad ** 2) / band_area)
    return pos, rad, realised


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PileupResult:
    """Outcome of one ice pile-up DEM run."""

    cfg: PileupConfig
    time: np.ndarray
    load_x: np.ndarray
    load_y: np.ndarray
    F_max: float
    F_steady: float
    F_mean_all: float
    F_sustained: float
    porosity: float
    solids_fraction: float
    pile_length: float
    pile_width: float
    pile_area: float
    n_floes_initial: int
    n_floes_pile: int
    ice_concentration_realised: float
    iso: ISO19906Result
    ratio_max: float
    ratio_steady: float
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pile_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    pile_extent: tuple = (0.0, 0.0, 0.0, 0.0)
    crushed_area: float = 0.0
    platen_travel: float = 0.0
    dt: float = 0.0
    n_steps: int = 0

    def as_dict(self, with_series: bool = True) -> dict:
        d = {
            "F_max": self.F_max,
            "F_steady": self.F_steady,
            "F_mean_all": self.F_mean_all,
            "F_sustained": self.F_sustained,
            "porosity": self.porosity,
            "solids_fraction": self.solids_fraction,
            "pile_length": self.pile_length,
            "pile_width": self.pile_width,
            "pile_area": self.pile_area,
            "n_floes_initial": self.n_floes_initial,
            "n_floes_pile": self.n_floes_pile,
            "ice_concentration_realised": self.ice_concentration_realised,
            "F_ISO": self.iso.F_ISO,
            "p_G": self.iso.p_G,
            "C_task": self.iso.C_task,
            "ratio_max": self.ratio_max,
            "ratio_steady": self.ratio_steady,
            "platen_travel": self.platen_travel,
            "crushed_area": self.crushed_area,
            "crushed_area_fraction": self.crushed_area / max(self.pile_area, 1e-9),
            "dt": self.dt,
            "n_steps": self.n_steps,
            "config": self.cfg.as_dict(),
        }
        if with_series:
            d["time"] = self.time.tolist()
            d["load_x"] = self.load_x.tolist()
            d["load_y"] = self.load_y.tolist()
        return d


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _chord(r_eff: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Chord length of a disc of radius ``r_eff`` cut at depth ``delta``.

    This chord is the width ``b`` of the contact zone; the contact area is
    ``h * b`` and the crushing force capacity is ``sigma_c * h * b``.
    """
    rb = np.clip(delta, 0.0, r_eff)
    return 2.0 * np.sqrt(np.maximum(r_eff * r_eff - (r_eff - rb) ** 2, 0.0))


def comminute(rad: np.ndarray, idx: np.ndarray, excess: np.ndarray,
              share: float, r_min: float) -> float:
    """Absorb over-penetration by eroding the crushed floe surface.

    Ice does not behave like a rigid body once the contact pressure reaches its
    crushing strength: the crushed layer is destroyed.  ``excess`` is the
    penetration beyond the crushable-zone depth; it is absorbed by reducing the
    floe radii (``share`` of the excess per floe for a pair contact), and the
    area removed is returned for the mass-loss audit.  This is the 2-D
    plan-view analogue of rubble comminution / vertical extrusion, and it is
    what keeps the contact force -- and therefore the structure load -- at the
    crushing capacity instead of letting it grow without bound in a confined
    domain.
    """
    if idx.size == 0:
        return 0.0
    r_old = rad[idx]
    r_new = np.maximum(r_old - share * excess, r_min)
    rad[idx] = r_new
    return float(np.sum(math.pi * (r_old ** 2 - r_new ** 2)))


def normal_force(sigma_c: float, h: float, r_eff: np.ndarray, delta: np.ndarray,
                 k_n: float, delta_cap_factor: float = 0.20,
                 k_hard_factor: float = 4.0) -> np.ndarray:
    """Crushing-limited normal contact force of an ice floe.

    ``Fn = min(k_n * delta, sigma_c * h * b(delta))`` -- the ice cannot transmit
    more than its crushing strength over the contact area.  A floe is allowed to
    penetrate by ``delta_cap = delta_cap_factor * r_eff`` (the crushed zone);
    beyond that a stiff core (``k_hard_factor * k_n``) prevents tunnelling
    through a wall.  The crushing branch is what makes the structure load scale
    with ``sigma_c * h`` instead of with an arbitrary spring stiffness.
    """
    delta = np.asarray(delta, dtype=float)
    r_eff = np.asarray(r_eff, dtype=float)
    delta_cap = delta_cap_factor * r_eff
    f_cap = sigma_c * h * _chord(r_eff, np.minimum(delta, delta_cap))
    soft = np.minimum(k_n * delta, f_cap)
    hard = f_cap + k_hard_factor * k_n * np.maximum(delta - delta_cap, 0.0)
    return np.where(delta <= delta_cap, soft, hard)


def run_pileup_case(cfg: PileupConfig) -> PileupResult:
    """Run one 2-D plan-view ice pile-up DEM case against a vertical structure.

    Boundary conditions
    -------------------
    * ``x = x_platen(t)``: the upstream **kinematic platen** (the far-field ice
      sheet) advances at the drift speed ``V``.  It is the only energy/force
      source, exactly as the far-field ice sheet is in reality; it stops when it
      reaches the structure's upstream face (it represents an ice sheet that can
      never pass through the structure).
    * ``y = +/- y_half``: rough, confining channel walls (ice keeps piling up
      laterally); floes can leave through the lateral gaps beside the structure.
    * downstream is open: floes that pass the structure are removed.
    """
    rng = np.random.default_rng(cfg.seed)

    pos, rad, conc = make_floe_field(cfg, rng)
    n = pos.shape[0]
    if n < 12:
        raise RuntimeError(f"floe field too sparse (n={n}); increase ice_concentration")
    vel = np.zeros_like(pos)
    m = RHO_ICE * math.pi * rad ** 2 * cfg.h
    r_max = float(rad.max())
    r_mean = float(rad.mean())

    # contact stiffness / damping (soft-sphere, crushing-limited normal law)
    k_n = cfg.k_lin_factor * cfg.sigma_c * cfg.h
    m_eff = float(np.percentile(m, 10))
    omega = math.sqrt(k_n / m_eff)
    gamma = 2.0 * cfg.damping_ratio * math.sqrt(k_n * m_eff)
    omega_hard = math.sqrt(4.0 * k_n / m_eff)  # stiff core of the contact law
    dt_stab = 0.08 * 2.0 * math.pi / omega_hard
    n_sub = max(1, int(round(cfg.sample_dt / dt_stab)))
    dt = cfg.sample_dt / n_sub

    y_half = cfg.lateral_factor * cfg.D
    x_platen = float(pos[:, 0].min()) - 2.0 * r_max
    x_platen_stop = max(-0.5 * cfg.D - 2.5 * r_mean,
                        x_platen + cfg.platen_travel_factor * cfg.D)
    n_steps = int(round(cfg.duration / dt))
    n_out = n_steps // n_sub

    c_w = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad * cfg.h  # N per (m/s)^2

    t_hist = np.zeros(n_out)
    Fx_hist = np.zeros(n_out)
    Fy_hist = np.zeros(n_out)

    pos = pos.copy()
    rad = rad.copy()
    vel = vel.copy()
    m = m.copy()
    load_x = np.zeros(pos.shape[0])
    load_y = np.zeros(pos.shape[0])
    crushed_area = 0.0
    r_min = 0.12 * float(cfg.floe_radius_median)
    R = 0.5 * cfg.D
    a_sq = 0.5 * cfg.D
    platen_travel = 0.0

    for step in range(n_steps):
        # ---- platen position (kinematic far-field ice sheet)
        target = min(cfg.drift_speed * (step + 1) * dt, x_platen_stop - x_platen)
        travel = max(target, 0.0)
        xp = x_platen + travel
        platen_travel = travel

        force = np.zeros_like(pos)
        tree = cKDTree(pos)

        # ---------------- floe-floe contacts ----------------
        pairs = tree.query_pairs(2.0 * r_max, output_type="ndarray")
        if pairs.size:
            i, j = pairs[:, 0], pairs[:, 1]
            dvec = pos[j] - pos[i]
            d = np.hypot(dvec[:, 0], dvec[:, 1])
            d = np.where(d < 1e-12, 1e-12, d)
            delta = rad[i] + rad[j] - d
            act = delta > 0.0
            if np.any(act):
                i, j, d, delta = i[act], j[act], d[act], delta[act]
                nvec = dvec[act] / d[:, None]
                r_eff = 2.0 * rad[i] * rad[j] / (rad[i] + rad[j])
                fn = normal_force(cfg.sigma_c, cfg.h, r_eff, delta, k_n)
                vrel = vel[j] - vel[i]
                vn = np.einsum("ij,ij->i", vrel, nvec)
                fn = np.maximum(fn + gamma * vn, 0.0)
                tvec = vrel - vn[:, None] * nvec
                tnorm = np.linalg.norm(tvec, axis=1)
                ft = np.minimum(cfg.mu * fn, 0.5 * gamma * tnorm)
                ftvec = -ft[:, None] * tvec / np.maximum(tnorm, 1e-12)[:, None]
                fij = fn[:, None] * nvec + ftvec
                np.add.at(force, i, -fij)
                np.add.at(force, j, fij)

        # ---------------- structure contacts ----------------
        load_x = np.zeros_like(load_x)
        load_y = np.zeros_like(load_y)
        if cfg.shape == "cylinder":
            cx, cy = pos[:, 0], pos[:, 1]
            dist = np.hypot(cx, cy)
            ds_all = R + rad - dist
            hit = ds_all > 0.0
            if np.any(hit):
                dnorm = np.maximum(dist[hit], 1e-12)
                # outward normal: structure centre -> floe centre
                nx = cx[hit] / dnorm
                ny = cy[hit] / dnorm
                ds = ds_all[hit]
                fn = normal_force(cfg.sigma_c, cfg.h, rad[hit], ds, k_n)
                idx_h = np.nonzero(hit)[0]
                if cfg.comminution:
                    crushed_area += comminute(rad, idx_h,
                                              np.maximum(ds - 0.20 * rad[hit], 0.0),
                                              1.0, r_min)
                vn = vel[hit, 0] * nx + vel[hit, 1] * ny
                fn = np.maximum(fn - gamma * vn, 0.0)
                np.add.at(force, np.nonzero(hit)[0],
                          np.column_stack([fn * nx, fn * ny]))
                load_x[hit] = -fn * nx
                load_y[hit] = -fn * ny
        else:  # square column: exact signed distance to the square footprint
            cx_, cy_ = np.clip(pos[:, 0], -a_sq, a_sq), np.clip(pos[:, 1], -a_sq, a_sq)
            dvec = np.column_stack([pos[:, 0] - cx_, pos[:, 1] - cy_])
            dist = np.hypot(dvec[:, 0], dvec[:, 1])
            inside = dist < 1e-9
            if np.any(inside):
                # a floe centre inside the footprint: push out through the
                # nearest face (numerical guard, should not happen once jammed)
                qx = a_sq - np.abs(pos[inside, 0])
                qy = a_sq - np.abs(pos[inside, 1])
                use_x = qx < qy
                dvec[inside, 0] = np.where(use_x, np.sign(pos[inside, 0]) * 1e-6, 0.0)
                dvec[inside, 1] = np.where(use_x, 0.0, np.sign(pos[inside, 1]) * 1e-6)
                dist[inside] = 1e-6
            ds_all = rad - dist
            hit = ds_all > 0.0
            if np.any(hit):
                ds = ds_all[hit]
                nx = dvec[hit, 0] / dist[hit]
                ny = dvec[hit, 1] / dist[hit]
                fn = normal_force(cfg.sigma_c, cfg.h, rad[hit], ds, k_n)
                idx = np.nonzero(hit)[0]
                if cfg.comminution:
                    crushed_area += comminute(rad, idx,
                                              np.maximum(ds - 0.20 * rad[hit], 0.0),
                                              1.0, r_min)
                vn = vel[hit, 0] * nx + vel[hit, 1] * ny
                fn = np.maximum(fn - gamma * vn, 0.0)
                np.add.at(force, idx, np.column_stack([fn * nx, fn * ny]))
                load_x[hit] = -fn * nx
                load_y[hit] = -fn * ny

        # ---------------- platen contact ----------------
        delta_p = xp - (pos[:, 0] - rad)
        hit_p = delta_p > 0.0
        if np.any(hit_p):
            ds = delta_p[hit_p]
            fn = normal_force(cfg.sigma_c, cfg.h, rad[hit_p], ds, k_n)
            vn = vel[hit_p, 0]
            fn = np.maximum(fn + gamma * (cfg.drift_speed - vn), 0.0)
            force[hit_p, 0] += fn

        # ---------------- lateral channel walls ----------------
        for sign in ((-1.0, 1.0) if cfg.lateral_wall else ()):
            # penetration of the floe into the wall at y = sign*y_half
            ds_all = rad + sign * pos[:, 1] - y_half
            hit = ds_all > 0.0
            if not np.any(hit):
                continue
            ds = ds_all[hit]
            fn = normal_force(cfg.sigma_c, cfg.h, rad[hit], ds, k_n)
            vn = sign * vel[hit, 1]
            fn = np.maximum(fn - gamma * vn, 0.0)
            force[hit, 1] += -sign * fn

        # ---------------- water drag (still water) ----------------
        speed = np.hypot(vel[:, 0], vel[:, 1])
        force -= (c_w * speed)[:, None] * vel

        # ---------------- integrate (semi-implicit Euler) ----------------
        vel += (force / m[:, None]) * dt
        v_cap = 4.0 * cfg.drift_speed + 1.0
        np.clip(vel, -v_cap, v_cap, out=vel)
        pos += vel * dt

        # ---------------- downstream wall (the sheet continues downstream) ---
        x_right = 0.5 * cfg.D + cfg.downstream_length_factor * cfg.D
        ds_r = rad - (x_right - pos[:, 0]) if cfg.downstream_wall else np.zeros_like(rad)
        hit_r = ds_r > 0.0
        if np.any(hit_r):
            ds = ds_r[hit_r]
            fn = normal_force(cfg.sigma_c, cfg.h, rad[hit_r], ds, k_n)
            vn = -vel[hit_r, 0]
            fn = np.maximum(fn - gamma * vn, 0.0)
            force[hit_r, 0] -= fn

        # fully comminuted floes leave the field (crushed fines are carried away)
        if step % 25 == 0:
            dead = rad <= 1.001 * r_min
            if np.any(dead):
                keep = ~dead
                pos, rad, vel, m = pos[keep], rad[keep], vel[keep], m[keep]
                load_x, load_y = load_x[keep], load_y[keep]
                c_w = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad * cfg.h

        if (step + 1) % n_sub == 0:
            k = (step + 1) // n_sub - 1
            if 0 <= k < n_out:
                t_hist[k] = (step + 1) * dt
                Fx_hist[k] = load_x.sum()
                Fy_hist[k] = load_y.sum()

    F_mean_all = float(np.mean(Fx_hist))
    t_drive_end = min(cfg.duration, cfg.platen_travel_factor * cfg.D / cfg.drift_speed)
    F_max, F_steady, F_sustained = _peak_and_steady(Fx_hist, cfg.sample_dt, t_drive_end)

    geom = measure_pile_porosity(pos, rad, cfg)
    iso = iso19906_load(cfg.sigma_c, cfg.h, cfg.D, cfg.shape)

    return PileupResult(
        cfg=cfg,
        time=t_hist,
        load_x=Fx_hist,
        load_y=Fy_hist,
        F_max=float(F_max),
        F_steady=float(F_steady),
        F_mean_all=F_mean_all,
        F_sustained=float(F_sustained),
        porosity=float(geom["porosity"]),
        solids_fraction=float(geom["solids_fraction"]),
        pile_length=float(geom["pile_length"]),
        pile_width=float(geom["pile_width"]),
        pile_area=float(geom["window_area"]),
        n_floes_initial=int(n),
        n_floes_pile=int(geom["n_floes_pile"]),
        ice_concentration_realised=float(conc),
        iso=iso,
        ratio_max=float(F_max / iso.F_ISO),
        ratio_steady=float(F_steady / iso.F_ISO),
        positions=pos.copy(),
        radii=rad.copy(),
        pile_mask=geom["mask"],
        pile_extent=geom["extent"],
        crushed_area=float(crushed_area),
        platen_travel=float(platen_travel),
        dt=float(dt),
        n_steps=int(n_steps),
    )


def _peak_and_steady(series: np.ndarray, sample_dt: float,
                     t_drive_end: float) -> tuple[float, float, float]:
    """Peak load, sustained (driven) load and relaxed load.

    * ``F_max``      : peak of the 50 ms moving average -- the peak jam load;
    * ``F_steady``   : mean over the last 25 % of the record, i.e. the load of
      the standing rubble jam once the platen has stopped driving;
    * ``F_sustained``: mean over the last 40 % of the driven interval -- the load
      while the sheet is still being pushed forward (jam/slip regime, lower
      because the pile is continuously shearing).
    """
    if series.size == 0:
        return 0.0, 0.0, 0.0
    n = series.size
    n_drive = int(np.clip(round(t_drive_end / sample_dt), 2, n))
    lo = int(0.6 * n_drive)
    f_steady = float(np.mean(series[lo:max(n_drive, lo + 1)]))
    f_relax = float(np.mean(series[int(0.75 * n):]))
    # peak of a 50 ms moving average: the physical jam peak, immune to
    # single-step contact-solver spikes
    w = max(1, int(round(0.05 / sample_dt)))
    if w > 1 and n > w:
        kern = np.ones(w) / w
        smooth = np.convolve(series, kern, mode="valid")
        f_max = float(np.max(smooth[max(int(0.1 * smooth.size), 1):]))
    else:
        f_max = float(np.max(series))
    return f_max, f_relax, f_steady  # peak, steady (relaxed jam), sustained


# ---------------------------------------------------------------------------
# Pile geometry and porosity
# ---------------------------------------------------------------------------
def measure_pile_porosity(pos: np.ndarray, rad: np.ndarray, cfg: PileupConfig) -> dict:
    """Measure the rubble-pile void fraction from the simulated packing.

    The pile is the material jammed **in front of the structure's upstream
    face**: the measurement window is ``x in [-R - D, -R]`` (``R = D/2``),
    ``|y| <= 0.8 * D``.  The window is rasterised on a ``cfg.grid_cell`` grid,
    the discs are stamped into it, and

        n = 1 - (area covered by discs) / (window area) .

    Because the ice field upstream is continuous, the window is normally full of
    material; ``n`` is therefore the plan-view (depth-averaged) void fraction of
    the rubble pile, which is what the porosity-resistance law is tested on.
    """
    s = cfg.grid_cell
    R = 0.5 * cfg.D
    x_hi = -R
    x_lo = -R - cfg.D
    y_half = cfg.lateral_factor * cfg.D
    gx = np.arange(x_lo, x_hi, s)
    gy = np.arange(-y_half, y_half, s)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    solid = np.zeros_like(X, dtype=bool)
    n_pile = 0
    for (px, py), rr in zip(pos, rad):
        if px + rr < x_lo or px - rr > x_hi or abs(py) - rr > y_half:
            continue
        stamp = (X - px) ** 2 + (Y - py) ** 2 <= rr * rr
        solid |= stamp
        n_pile += 1
    window_area = X.size * s * s
    solid_area = float(solid.sum()) * s * s
    porosity_window = 1.0 - solid_area / window_area if window_area > 0 else 0.0

    # rubble-pile void fraction: void space *inside the pile envelope*, i.e.
    # morphological closing + hole filling of the packed material (this removes
    # open water in front of the pile from the denominator, so that the number is
    # a packing porosity and not a coverage fraction)
    r_mean_ = float(np.mean(rad)) if rad.size else cfg.floe_radius_median
    kk = max(1, int(round(0.55 * r_mean_ / s)))
    st = np.zeros((2 * kk + 1, 2 * kk + 1), dtype=bool)
    yy, xx = np.mgrid[-kk:kk + 1, -kk:kk + 1]
    st[xx * xx + yy * yy <= kk * kk] = True
    env = ndimage.binary_closing(solid, structure=st)
    env = ndimage.binary_fill_holes(env)
    lab, nlab = ndimage.label(env)
    if nlab > 1:
        sizes = ndimage.sum(env, lab, range(1, nlab + 1))
        env = lab == (int(np.argmax(sizes)) + 1)
    env_area = float(env.sum()) * s * s
    solid_in_env = float((solid & env).sum()) * s * s
    porosity = 1.0 - solid_in_env / env_area if env_area > 0 else porosity_window

    # upstream extent of the material connected to the structure (pile length)
    lab, nlab = ndimage.label(solid)
    pile_length = 0.0
    extent = (float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1]))
    if nlab:
        # component touching the downstream (structure) edge of the window
        edge = np.unique(lab[-1, :])
        edge = edge[edge > 0]
        if edge.size:
            comp = np.isin(lab, edge)
            ii, jj = np.nonzero(comp)
            pile_length = float(gx[ii.max()] - gx[ii.min()])
            extent = (float(gx[ii.min()]), float(gx[ii.max()]),
                      float(gy[jj.min()]), float(gy[jj.max()]))
    return {
        "porosity": float(porosity),
        "porosity_window": float(porosity_window),
        "solids_fraction": float(1.0 - porosity),
        "envelope_area": float(env_area),
        "window_area": float(window_area),
        "pile_length": pile_length,
        "pile_width": extent[3] - extent[2],
        "n_floes_pile": int(n_pile),
        "mask": solid,
        "extent": extent,
    }


# ---------------------------------------------------------------------------
# Porosity-resistance scaling fit
# ---------------------------------------------------------------------------
def fit_porosity_scaling(porosity: Sequence[float], force: Sequence[float],
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """Least-squares fit of ``F = A * (1 - n)**alpha`` in the log domain.

    Returns the exponent, amplitude, coefficient of determination (on log F),
    a bootstrap 95 % interval of ``alpha`` and the number of points.
    """
    n = np.asarray(porosity, dtype=float)
    F = np.asarray(force, dtype=float)
    ok = np.isfinite(n) & np.isfinite(F) & (F > 0) & (n < 0.98)
    n, F = n[ok], F[ok]
    if n.size < 3:
        raise ValueError("need at least 3 valid (porosity, force) points")
    x = np.log(1.0 - n)
    y = np.log(F)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, alpha = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n.size, n.size)
        Ab = np.column_stack([np.ones(idx.size), x[idx]])
        try:
            boots[b] = np.linalg.lstsq(Ab, y[idx], rcond=None)[0][1]
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            boots[b] = np.nan
    boots = boots[np.isfinite(boots)]
    return {
        "alpha": alpha,
        "alpha_ci95": [float(np.percentile(boots, 2.5)),
                       float(np.percentile(boots, 97.5))] if boots.size else [alpha, alpha],
        "alpha_std_boot": float(np.std(boots)) if boots.size else 0.0,
        "amplitude": float(math.exp(intercept)),
        "R2": r2,
        "n_points": int(n.size),
        "porosity_range": [float(n.min()), float(n.max())],
        "force_range": [float(F.min()), float(F.max())],
    }


def load_statistics(series: np.ndarray, sample_dt: float) -> dict:
    """Descriptive statistics of a load time history."""
    s = np.asarray(series, dtype=float)
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "max": float(s.max()),
        "min": float(s.min()),
        "rms": float(np.sqrt(np.mean(s ** 2))),
        "duration": float(s.size * sample_dt),
        "n_samples": int(s.size),
    }