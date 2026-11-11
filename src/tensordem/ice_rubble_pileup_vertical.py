"""Ice rubble pile-up against a vertical structure: 2-D vertical-section DEM.

Physical problem
----------------
A drifting broken-ice sheet (thickness ``h``, compressive strength ``sigma_c``,
drift speed ``V``) runs into a fixed vertical structure of width ``D``
(cylinder diameter or square-column side).  Blocks crush at the contact, ride
up and accumulate into a rubble pile in front of the structure.  This module
resolves the *vertical* cross-section ``(x, z)`` -- the standard idealisation
for rubble building against a structure (Hopkins 1998; Paavilainen & Tuhkuri
2012) -- so pile face angle, rubble porosity and the load transmitted to the
structure all come out of the same simulation.  Loads are per unit width
``[N/m]``; a structure of width ``D`` carries ``F_3D ~ F' * D``.

Model
-----
* Bodies: rigid circular blocks (rubble-block disc idealisation, no spin DOF),
  radius lognormal around ``0.3 h``; ``n_rows`` rows reproduce a sheet of
  thickness ``h``.
* Water line ``z = 0``: each block carries its buoyant weight
  ``F_z = rho_w g A_sub(z) - rho_i g A`` (analytic ``A_sub``, see
  :func:`submerged_area`) plus quadratic water drag.  Freeboard is not
  imposed -- it follows from buoyancy.
* Contact law: soft-sphere with an ice **crushing cap**

      F_n = min(k_n * delta, sigma_c * b(delta)),   b = chord of the contact

  i.e. ice cannot transmit more than its crushing strength over the contact
  area -- the same law as :mod:`tensordem.ice_vertical_structure_pileup`.
  Tangential force is regularised Coulomb friction
  ``F_t = -mu F_n tanh(v_t / v_eps)`` (static friction -> a stable pile).
* Driving: the far-field intact sheet is a kinematic vertical platen moving at
  ``V`` (thickness ``h`` around the water line); the structure is a rigid
  vertical wall at ``x = 0``.
* Predicted load: horizontal force on the wall, ``F'(t)`` [N/m].

ISO 19906 reference
-------------------
:func:`iso_19906_reference` re-uses
:func:`tensordem.ice_vertical_structure_pileup.iso19906_load` (ISO 19906:2010
A.8.2.6 global pressure ``p_G = C_R (h/h_0)^n (w/h)^m`` capped at ``sigma_c``,
``F = p_G w h``) and returns the 3-D load and the per-unit-width load
``F/D`` used for the DEM comparison.  The campaign form ``F = C sigma_c h^2``
is reported as ``C_task = F' / (sigma_c h^2)``.

Units: SI (m, s, N, Pa); loads per unit width in N/m.

References
----------
ISO 19906:2010, Arctic offshore structures, Annex A.8 (global ice pressure).
Hopkins, M.A. (1998), Four stages of pressure ridging, J. Geophys. Res. 103.
Timco, G.W., Weeks, W.F. (2010), A review of the engineering properties of sea
    ice, Cold Reg. Sci. Technol. 60, 107-129.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .ice_vertical_structure_pileup import iso19906_load

__all__ = [
    "RHO_ICE",
    "RHO_WATER",
    "GRAVITY",
    "FLOATING_CENTRE_OFFSET",
    "PileupVerticalConfig",
    "PileupVerticalResult",
    "submerged_area",
    "buoyant_vertical_force",
    "make_floe_sheet",
    "normal_force",
    "cap_binds",
    "run_vertical_pileup_case",
    "measure_pile_geometry",
    "fit_porosity_scaling",
    "iso_19906_reference",
    "main",
]

RHO_ICE = 900.0
RHO_WATER = 1025.0
GRAVITY = 9.81
CD_WATER = 1.2

#: centre height / radius of a floating disc for rho_ice/rho_water = 0.878
FLOATING_CENTRE_OFFSET = -0.641


# ---------------------------------------------------------------------------
# Buoyancy
# ---------------------------------------------------------------------------
def submerged_area(radius, z_centre):
    """Submerged area of a disc below the water line ``z = 0``.

    ``A_sub = pi r^2 - r^2 (alpha - sin(alpha) cos(alpha))`` with area of the
    circular cap above the water line ``r^2 (alpha - sin cos)`` and
    ``alpha = arccos(-z_c / r)``.

    Limits: ``z_c = +r`` -> 0 (block entirely above the water);
    ``z_c = -r`` -> ``pi r^2`` (entirely submerged); ``z_c = 0`` -> half.
    """
    radius = np.asarray(radius, dtype=float)
    z_centre = np.asarray(z_centre, dtype=float)
    r = np.maximum(radius, 1e-12)
    cos_alpha = np.clip(-z_centre / r, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    cap = r * r * (alpha - np.sin(alpha) * cos_alpha)
    return np.pi * r * r - cap


def buoyant_vertical_force(radius, z_centre, rho_ice: float = RHO_ICE,
                           rho_water: float = RHO_WATER,
                           gravity: float = GRAVITY):
    """Net vertical force per unit width [N/m] on a floating disc (+ up)."""
    radius = np.asarray(radius, dtype=float)
    z_centre = np.asarray(z_centre, dtype=float)
    area = np.pi * radius * radius
    return (rho_water * gravity * submerged_area(radius, z_centre)
            - rho_ice * gravity * area)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PileupVerticalConfig:
    """Configuration of one vertical-section pile-up DEM case."""

    sigma_c: float = 3.0e6        # Pa, uniaxial compressive strength
    h: float = 1.0                # m, ice sheet thickness
    D: float = 5.0                # m, structure width (cylinder dia / square side)
    shape: str = "cylinder"       # "cylinder" | "square"
    drift_speed: float = 0.5      # m/s
    seed: int = 11
    # block size distribution, in multiples of h
    r_median_factor: float = 0.12
    r_sigma_ln: float = 0.25
    r_bounds_factor: tuple[float, float] = (0.08, 0.18)
    # contact parameters
    mu: float = 0.45
    k_lin_factor: float = 3.0      # k_n = factor * sigma_c  [Pa]
    hard_factor: float = 2.0
    delta_cap_factor: float = 0.40  # crushable-zone depth / r_eff
    damping_ratio: float = 0.35
    v_eps: float = 0.01             # m/s, Coulomb regularisation speed
    comminution: bool = False       # optional radius erosion (mass-losing; off by default)
    r_min_factor: float = 0.35      # eroded radius floor, x initial radius
    platen_force_cap_factor: float = 3.0  # platen thrust cap / (sigma_c h)
    # layout
    sheet_length_factor: float = 8.0
    n_rows: int = 2
    platen_travel_factor: float = 10.0
    wall_top_factor: float = 4.0
    wall_bottom_factor: float = 3.0
    platen_top_factor: float = 0.15
    sheet_plane_factor: float = 0.95  # lower boundary (keel suppressed)
    # numerics / output
    dt: float = 0.0
    sample_dt: float = 5.0e-3
    grid_cell: float = 0.05         # pile rasterisation cell / h
    settle_time: float = 0.6        # s, buoyant settling before the drive
    max_speed: float = 6.0          # m/s numerical safety clip

    def as_dict(self) -> dict:
        d = asdict(self)
        d["r_bounds_factor"] = list(self.r_bounds_factor)
        return d


@dataclass
class PileupVerticalResult:
    """Outcome of one vertical-section pile-up case."""

    F_max: float                  # N/m, peak wall load
    F_steady: float               # N/m, mean over the tail of the window
    F_mean: float                 # N/m, mean over the measurement window
    F_ISO: float                  # N, ISO 19906 3-D load
    F_ISO_per_width: float        # N/m
    p_G: float                    # Pa
    ratio_max: float              # F_max / F_ISO_per_width
    ratio_steady: float
    C_task: float                 # F_max / (sigma_c h^2)
    porosity: float
    solids_fraction: float
    pile_height: float            # m above the water line
    pile_length: float            # m, wall face to toe
    pile_angle_deg: float
    pile_area: float              # m^2 solid block area in the envelope
    envelope_area: float          # m^2
    n_blocks_initial: int
    n_blocks_pile: int
    platen_travel: float
    pressure_capped: bool = False
    crushed_area: float = 0.0
    time: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    load: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    pos: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))
    rad: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    cfg: PileupVerticalConfig | None = None
    dt: float = 0.0
    n_steps: int = 0

    def as_dict(self, with_series: bool = True) -> dict:
        d = {
            "F_max": self.F_max,
            "F_steady": self.F_steady,
            "F_mean": self.F_mean,
            "F_ISO": self.F_ISO,
            "F_ISO_per_width": self.F_ISO_per_width,
            "p_G": self.p_G,
            "ratio_max": self.ratio_max,
            "ratio_steady": self.ratio_steady,
            "C_task": self.C_task,
            "pressure_capped": self.pressure_capped,
            "porosity": self.porosity,
            "solids_fraction": self.solids_fraction,
            "pile_height": self.pile_height,
            "pile_length": self.pile_length,
            "pile_angle_deg": self.pile_angle_deg,
            "pile_area": self.pile_area,
            "envelope_area": self.envelope_area,
            "n_blocks_initial": self.n_blocks_initial,
            "n_blocks_pile": self.n_blocks_pile,
            "platen_travel": self.platen_travel,
            "crushed_area": self.crushed_area,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "config": self.cfg.as_dict() if self.cfg is not None else None,
        }
        if with_series:
            d["time"] = np.asarray(self.time).tolist()
            d["load"] = np.asarray(self.load).tolist()
            d["pos"] = np.asarray(self.pos).tolist()
            d["rad"] = np.asarray(self.rad).tolist()
        return d


# ---------------------------------------------------------------------------
# ISO 19906 reference
# ---------------------------------------------------------------------------
def iso_19906_reference(cfg: PileupVerticalConfig) -> dict:
    """ISO 19906 load for this configuration plus its per-unit-width form."""
    iso = iso19906_load(cfg.sigma_c, cfg.h, cfg.D, shape=cfg.shape)
    per_width = iso.F_ISO / cfg.D
    return {
        "F_ISO": float(iso.F_ISO),
        "F_ISO_per_width": float(per_width),
        "p_G": float(iso.p_G),
        "C_task_iso": float(per_width / (cfg.sigma_c * cfg.h * cfg.h)),
        "C_R": 2.8e6,
        "n_exp": float(iso.n_exp),
        "m_exp": float(iso.m_exp),
        "pressure_capped": bool(iso.pressure_capped),
        "form": "F = p_G w h, p_G = C_R (h/h0)^n (w/h)^m  [ISO 19906 A.8.2.6]",
    }


# ---------------------------------------------------------------------------
# Contact law
# ---------------------------------------------------------------------------
def _chord(r_eff: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Contact chord width ``b`` of a disc of radius ``r_eff`` cut at depth
    ``delta``."""
    rb = np.clip(delta, 0.0, r_eff)
    return 2.0 * np.sqrt(np.maximum(r_eff * r_eff - (r_eff - rb) ** 2, 0.0))


def normal_force(sigma_c: float, r_eff, delta, k_n: float,
                 delta_cap_factor: float = 0.25,
                 hard_factor: float = 2.0) -> np.ndarray:
    """Crushing-capped normal contact force per unit width [N/m].

    ``F_n = min(k_n delta, sigma_c b(delta))`` on the crushable branch; beyond
    ``delta_cap = delta_cap_factor r_eff`` a stiff core
    (``hard_factor k_n``) prevents tunnelling.
    """
    delta = np.asarray(delta, dtype=float)
    r_eff = np.asarray(r_eff, dtype=float)
    delta_cap = delta_cap_factor * r_eff
    f_cap = sigma_c * _chord(r_eff, np.minimum(delta, delta_cap))
    soft = np.minimum(k_n * delta, f_cap)
    hard = f_cap + hard_factor * k_n * np.maximum(delta - delta_cap, 0.0)
    return np.where(delta <= delta_cap, soft, hard)


def cap_binds(cfg: PileupVerticalConfig) -> bool:
    """True when the crushing cap is reached *inside* the crushable zone.

    Requirement: the soft-spring force at ``delta_cap = delta_cap_factor * r``
    must reach the crushing capacity ``sigma_c * b(delta_cap)``; otherwise the
    structure load is controlled by the contact stiffness rather than by the
    ice strength, and ``F_max`` is a numerical artefact.  Checked for the
    smallest block of the distribution (worst case).
    """
    r = cfg.r_bounds_factor[0] * cfg.h
    cap = cfg.delta_cap_factor * r
    b = 2.0 * math.sqrt(max(r * r - (r - cap) ** 2, 0.0))
    return bool(cfg.k_lin_factor * cap >= b)


# ---------------------------------------------------------------------------
# Initial sheet
# ---------------------------------------------------------------------------
def _draw_radii(n: int, cfg: PileupVerticalConfig,
                rng: np.random.Generator) -> np.ndarray:
    lo, hi = (f * cfg.h for f in cfg.r_bounds_factor)
    out: list[float] = []
    while len(out) < n:
        draw = rng.lognormal(math.log(cfg.r_median_factor * cfg.h),
                             cfg.r_sigma_ln, size=4 * n)
        out.extend(draw[(draw >= lo) & (draw <= hi)].tolist())
    return np.array(out[:n], dtype=float)


def make_floe_sheet(cfg: PileupVerticalConfig, rng: np.random.Generator):
    """Build the incoming broken-ice sheet as a dense random block packing.

    Candidate block centres are drawn on a jittered grid inside a band of
    thickness ``h`` around the water line and accepted greedily when they do
    not overlap an already accepted block (random sequential adsorption).  Each
    block sits at its free-floating centre height ``-0.641 r`` plus a small
    offset, so the freeboard is a buoyancy result, not an input.  Returns
    ``(pos, rad)`` with ``pos`` in ``(x, z)`` [m].
    """
    h = cfg.h
    length = cfg.sheet_length_factor * h
    z_low, z_high = -0.90 * h, 0.10 * h
    r_med = cfg.r_median_factor * h
    cell = 0.35 * r_med

    xs = np.arange(-length, -0.35 * h, cell)
    zs = np.arange(z_low + 0.6 * r_med, z_high - 0.3 * r_med, cell)
    cand = np.array([(x + rng.uniform(-0.4, 0.4) * cell,
                      z + rng.uniform(-0.4, 0.4) * cell)
                     for x in xs for z in zs], dtype=float)
    rng.shuffle(cand)
    radii = _draw_radii(min(400, 4 * cand.shape[0]), cfg, rng)

    pos, rad = [], []
    for (xc, zc), r in zip(cand, radii):
        if zc - r < z_low or zc + r > z_high:
            continue
        if pos:
            pp = np.asarray(pos)
            rr = np.asarray(rad)
            if np.any(np.hypot(pp[:, 0] - xc, pp[:, 1] - zc) < 0.995 * (rr + r)):
                continue
        pos.append((xc, zc))
        rad.append(r)
    if len(pos) < 10:
        raise RuntimeError("could not build a dense floe sheet")
    order = rng.permutation(len(pos))
    return np.asarray(pos)[order], np.asarray(rad)[order]


# ---------------------------------------------------------------------------
# Pile geometry
# ---------------------------------------------------------------------------
def measure_pile_geometry(pos: np.ndarray, rad: np.ndarray,
                          cfg: PileupVerticalConfig, z_sheet_top: float) -> dict:
    """Rubble geometry: porosity, sail height, toe length and flank angle.

    Definitions (all measured from the DEM block packing):

    * **pile blocks**: blocks whose top rises above the undisturbed incoming
      sheet, ``z_i + r_i > z_sheet_top``;
    * **toe**: leftmost extent of the raised rubble, ``min(x_i - r_i)``;
    * **envelope**: convex hull of the block extremes (``x +- r``, ``z +- r``);
      **porosity** ``n = 1 - A_blocks / A_hull`` -- the standard void-fraction
      definition for a rubble body, blocks counted at full area (so ``n`` is a
      lower bound: an overlap-free packing is assumed);
    * **sail height**: highest block top;
    * **flank angle**: slope of the outer boundary ``x_min(z)`` measured in
      horizontal bands between the sheet top and 90 % of the sail height --
      the angle of repose of the rubble flank, which is the quantity that can
      be compared with the 30-45 deg range quoted for ice rubble.
    """
    h = cfg.h
    empty = {"porosity": float("nan"), "solids_fraction": float("nan"),
             "pile_height": 0.0, "pile_length": 0.0, "pile_angle_deg": float("nan"),
             "pile_area": 0.0, "envelope_area": 0.0, "n_blocks_pile": 0,
             "toe_x": float("nan"), "flank_z": np.zeros(0), "flank_x": np.zeros(0)}
    if pos.shape[0] == 0:
        return empty

    tops = pos[:, 1] + rad
    pile = tops > z_sheet_top
    if np.count_nonzero(pile) < 4:
        return empty
    pp, rr = pos[pile], rad[pile]

    toe_x = float(np.min(pp[:, 0] - rr))
    sail = float(np.max(pp[:, 1] + rr))

    # ---- envelope: convex hull of the block extremes ----
    from scipy.spatial import ConvexHull
    ext = np.vstack([np.column_stack([pp[:, 0] + rr, pp[:, 1]]),
                     np.column_stack([pp[:, 0] - rr, pp[:, 1]]),
                     np.column_stack([pp[:, 0], pp[:, 1] + rr]),
                     np.column_stack([pp[:, 0], pp[:, 1] - rr])])
    try:
        hull = ConvexHull(ext)
        hull_area = float(hull.volume)          # 2-D: "volume" is the area
    except Exception:
        return empty
    solid_area = float(np.sum(np.pi * rr ** 2))
    porosity = float(min(max(1.0 - solid_area / max(hull_area, 1e-12), 0.0), 1.0))

    # ---- flank angle from the outer boundary x_min(z) ----
    z_lo = z_sheet_top
    z_hi = 0.9 * sail
    bands = np.linspace(z_lo, z_hi, 6)
    bz, bx = [], []
    for k in range(len(bands) - 1):
        sel = (pp[:, 1] >= bands[k]) & (pp[:, 1] < bands[k + 1])
        if np.any(sel):
            bz.append(0.5 * (bands[k] + bands[k + 1]))
            bx.append(float(np.min(pp[sel, 0] - rr[sel])))
    angle = float("nan")
    if len(bx) >= 3 and (max(bx) - min(bx)) > 1e-6:
        slope = float(np.polyfit(bx, bz, 1)[0])   # dz/dx
        angle = float(math.degrees(math.atan(abs(slope))))

    return {
        "porosity": porosity,
        "solids_fraction": 1.0 - porosity,
        "pile_height": sail,
        "pile_length": float(abs(toe_x)),
        "pile_angle_deg": angle,
        "pile_area": solid_area,
        "envelope_area": hull_area,
        "n_blocks_pile": int(np.count_nonzero(pile)),
        "toe_x": toe_x,
        "flank_z": np.asarray(bz),
        "flank_x": np.asarray(bx),
    }


# ---------------------------------------------------------------------------
# Comminution (crushed-layer erosion)
# ---------------------------------------------------------------------------
def comminute(rad: np.ndarray, idx: np.ndarray, excess: np.ndarray,
              r_min: np.ndarray, share: float = 0.5) -> float:
    """Absorb over-penetration by eroding the crushed block surface.

    Ice does not behave as a rigid body once the contact pressure reaches its
    crushing strength: the crushed layer is destroyed.  ``excess`` is the
    penetration beyond the crushable-zone depth; it is absorbed by reducing the
    radii (``share`` of the excess per block of a pair), never below
    ``r_min``.  Returns the crushed (removed) area for the mass audit.
    """
    if idx.size == 0:
        return 0.0
    r_old = rad[idx].copy()
    rad[idx] = np.maximum(r_old - share * excess, r_min[idx])
    return float(np.sum(np.pi * (r_old ** 2 - rad[idx] ** 2)))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_vertical_pileup_case(cfg: PileupVerticalConfig) -> PileupVerticalResult:
    """Run one vertical-section ice pile-up DEM case.

    Boundary conditions: rigid vertical structure wall at ``x = 0`` spanning
    ``z in [-1.5h, +4h]``; kinematic platen (far-field intact sheet) advancing
    at ``V``; water free surface at ``z = 0`` with buoyancy and quadratic drag.
    """
    if cfg.sigma_c <= 0 or cfg.h <= 0 or cfg.D <= 0 or cfg.drift_speed <= 0:
        raise ValueError("sigma_c, h, D and drift_speed must be positive")

    rng = np.random.default_rng(cfg.seed)
    pos, rad = make_floe_sheet(cfg, rng)
    n = pos.shape[0]
    if n < 10:
        raise RuntimeError(f"sheet too sparse (n={n})")

    area = math.pi * rad ** 2
    mass = RHO_ICE * area                      # kg/m
    vel = np.zeros_like(pos)
    force = np.zeros_like(pos)

    z_base = float(np.min(pos[:, 1] - rad))
    z_sheet_top = float(np.max(pos[:, 1] + rad))

    k_n = cfg.k_lin_factor * cfg.sigma_c       # Pa (= N/m per m)
    m_min = float(np.min(mass))
    omega_hard = math.sqrt(cfg.hard_factor * k_n / m_min)
    dt = float(cfg.dt) if cfg.dt > 0 else 0.05 * 2.0 * math.pi / omega_hard
    dt = min(dt, cfg.sample_dt / 4.0)
    gamma = 2.0 * cfg.damping_ratio * math.sqrt(k_n * m_min)

    r_max = float(rad.max())
    r_min = cfg.r_min_factor * rad.copy()
    c_drag = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad     # N per (m/s)^2
    platen_cap = cfg.platen_force_cap_factor * cfg.sigma_c * cfg.h
    crushed_area = 0.0

    z_wall_top = cfg.wall_top_factor * cfg.h
    z_wall_bottom = -cfg.wall_bottom_factor * cfg.h
    z_platen_top = cfg.platen_top_factor * cfg.h
    z_floor = -cfg.sheet_plane_factor * cfg.h
    x_platen = float(pos[:, 0].min()) - 2.0 * r_max
    travel = cfg.platen_travel_factor * cfg.h
    x_p_stop = x_platen + travel

    settle_steps = int(round(cfg.settle_time / dt))
    drive_time = travel / cfg.drift_speed
    n_steps = settle_steps + int(round(drive_time / dt))
    sample_every = max(1, int(round(cfg.sample_dt / dt)))
    n_out = n_steps // sample_every + 2
    t_hist = np.zeros(n_out)
    load_hist = np.zeros(n_out)
    k_out = 0
    platen_force_prev = 0.0
    pairs = np.zeros((0, 2), dtype=int)
    patch_every = 4

    for step in range(n_steps):
        t = step * dt
        if step >= settle_steps and x_platen < x_p_stop and platen_force_prev < platen_cap:
            x_platen += cfg.drift_speed * dt

        force[:] = 0.0
        force[:, 1] += buoyant_vertical_force(rad, pos[:, 1])

        # ---------------- block-block contacts ----------------
        if step % patch_every == 0:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(2.0 * r_max + 0.05 * cfg.h, output_type="ndarray")
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
                fn = normal_force(cfg.sigma_c, r_eff, delta, k_n,
                                  cfg.delta_cap_factor, cfg.hard_factor)
                vrel = vel[j] - vel[i]
                vn = np.einsum("ij,ij->i", vrel, nvec)
                fn = np.maximum(fn + gamma * np.minimum(vn, 0.0), 0.0)
                vt = vrel - vn[:, None] * nvec
                vtn = np.linalg.norm(vt, axis=1)
                ft_mag = cfg.mu * fn * np.tanh(vtn / cfg.v_eps)
                ft = -ft_mag[:, None] * vt / np.maximum(vtn, 1e-12)[:, None]
                fij = fn[:, None] * nvec + ft
                np.add.at(force, i, -fij)
                np.add.at(force, j, fij)
                if cfg.comminution:
                    exc = np.maximum(delta - cfg.delta_cap_factor * r_eff, 0.0)
                    hot = exc > 0.0
                    if np.any(hot):
                        ih, jh = i[hot], j[hot]
                        r_old = rad.copy()
                        crushed_area += comminute(rad, ih, exc[hot], r_min)
                        crushed_area += comminute(rad, jh, exc[hot], r_min)
                        changed = np.union1d(ih, jh)
                        area[changed] = math.pi * rad[changed] ** 2
                        mass[changed] = RHO_ICE * area[changed]
                        c_drag[changed] = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad[changed]
                        r_max = float(rad.max())
                        del r_old

        # ---------------- structure wall contacts ----------------
        dx = pos[:, 0] + rad                       # penetration into x > 0
        hit = (dx > 0.0) & (pos[:, 1] + rad > z_wall_bottom) & (pos[:, 1] - rad < z_wall_top)
        load_now = 0.0
        if np.any(hit):
            ds = dx[hit]
            fn = normal_force(cfg.sigma_c, rad[hit], ds, k_n,
                              cfg.delta_cap_factor, cfg.hard_factor)
            vn = -vel[hit, 0]
            fn = np.maximum(fn - gamma * np.minimum(vn, 0.0), 0.0)
            ft_mag = cfg.mu * fn * np.tanh(np.abs(vel[hit, 1]) / cfg.v_eps)
            idxh = np.nonzero(hit)[0]
            np.add.at(force, idxh, np.column_stack([-fn, -ft_mag * np.sign(vel[hit, 1])]))
            load_now = float(np.sum(fn))
            if cfg.comminution:
                exc = np.maximum(ds - cfg.delta_cap_factor * rad[hit], 0.0)
                hot = exc > 0.0
                if np.any(hot):
                    ih = idxh[hot]
                    crushed_area += comminute(rad, ih, exc[hot], r_min)
                    area[ih] = math.pi * rad[ih] ** 2
                    mass[ih] = RHO_ICE * area[ih]
                    c_drag[ih] = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad[ih]
        if step % sample_every == 0 and k_out < n_out:
            load_hist[k_out] = load_now
            t_hist[k_out] = t
            k_out += 1

        # ---------------- platen contacts ----------------
        pen = x_platen - (pos[:, 0] - rad)
        hit_p = (pen > 0.0) & (pos[:, 1] - rad < z_platen_top)
        platen_force = 0.0
        if np.any(hit_p):
            ds = pen[hit_p]
            fn = normal_force(cfg.sigma_c, rad[hit_p], ds, k_n,
                              cfg.delta_cap_factor, cfg.hard_factor)
            vn = cfg.drift_speed - vel[hit_p, 0]
            fn = np.maximum(fn + gamma * np.minimum(vn, 0.0), 0.0)
            force[hit_p, 0] += fn
            platen_force = float(np.sum(fn))
            if cfg.comminution:
                exc = np.maximum(ds - cfg.delta_cap_factor * rad[hit_p], 0.0)
                hot = exc > 0.0
                if np.any(hot):
                    ih = np.nonzero(hit_p)[0][hot]
                    crushed_area += comminute(rad, ih, exc[hot], r_min)
                    area[ih] = math.pi * rad[ih] ** 2
                    mass[ih] = RHO_ICE * area[ih]
                    c_drag[ih] = 0.5 * RHO_WATER * CD_WATER * 2.0 * rad[ih]

        # ---------------- sheet plane (lower boundary, keel suppressed) ----
        pen_f = z_floor - (pos[:, 1] - rad)
        hit_f = pen_f > 0.0
        if np.any(hit_f):
            fn = normal_force(cfg.sigma_c, rad[hit_f], pen_f[hit_f], k_n,
                              cfg.delta_cap_factor, cfg.hard_factor)
            vn = -vel[hit_f, 1]
            fn = np.maximum(fn - gamma * np.minimum(vn, 0.0), 0.0)
            ft_mag = cfg.mu * fn * np.tanh(np.abs(vel[hit_f, 0]) / cfg.v_eps)
            idxf = np.nonzero(hit_f)[0]
            np.add.at(force, idxf, np.column_stack([-ft_mag * np.sign(vel[hit_f, 0]), fn]))

        platen_force_prev = platen_force

        # ---------------- water drag ----------------
        speed = np.hypot(vel[:, 0], vel[:, 1])
        force -= (c_drag * speed)[:, None] * vel

        # ---------------- integrate (symplectic Euler) ----------------
        vel += (force / mass[:, None]) * dt
        np.clip(vel, -cfg.max_speed, cfg.max_speed, out=vel)
        pos += vel * dt
        np.minimum(pos[:, 0], -0.2 * rad, out=pos[:, 0])

    # ---- post-processing ----
    t_hist = t_hist[:k_out]
    load_hist = load_hist[:k_out]
    t0 = cfg.settle_time + 0.50 * drive_time
    win = t_hist >= t0
    if not np.any(win):
        win = np.ones_like(t_hist, dtype=bool)
    lw = load_hist[win]
    tw = t_hist[win]
    F_max = float(np.max(lw)) if lw.size else 0.0
    F_mean = float(np.mean(lw)) if lw.size else 0.0
    # sustained (steady) load: mean over the plateau where a 10-point moving
    # average stays within 80 % of its own maximum -- robust to a single
    # numerical spike and independent of where the peak sits in the record.
    if lw.size >= 10:
        k = 10
        smooth = np.convolve(lw, np.ones(k) / k, mode="valid")
        ref = float(np.max(smooth))
        plat = smooth >= 0.80 * ref if ref > 0 else np.zeros_like(smooth, bool)
        F_steady = float(np.mean(lw[k - 1:][plat])) if np.any(plat) else float(ref)
    else:
        F_steady = float(np.mean(lw)) if lw.size else 0.0

    geo = measure_pile_geometry(pos, rad, cfg, z_sheet_top)
    ref = iso_19906_reference(cfg)

    return PileupVerticalResult(
        F_max=F_max,
        F_steady=F_steady,
        F_mean=F_mean,
        F_ISO=ref["F_ISO"],
        F_ISO_per_width=ref["F_ISO_per_width"],
        p_G=ref["p_G"],
        ratio_max=F_max / ref["F_ISO_per_width"],
        ratio_steady=F_steady / ref["F_ISO_per_width"],
        C_task=F_max / (cfg.sigma_c * cfg.h * cfg.h),
        porosity=float(geo["porosity"]),
        solids_fraction=float(geo["solids_fraction"]),
        pile_height=float(geo["pile_height"]),
        pile_length=float(geo["pile_length"]),
        pile_angle_deg=float(geo["pile_angle_deg"]),
        pile_area=float(geo["pile_area"]),
        envelope_area=float(geo["envelope_area"]),
        n_blocks_initial=int(n),
        n_blocks_pile=int(geo["n_blocks_pile"]),
        platen_travel=float(travel),
        crushed_area=float(crushed_area),
        pressure_capped=bool(ref["pressure_capped"]),
        time=t_hist,
        load=load_hist,
        pos=pos,
        rad=rad,
        cfg=cfg,
        dt=dt,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Porosity-resistance scaling law
# ---------------------------------------------------------------------------
def fit_porosity_scaling(solids_fraction, normalised_load) -> dict:
    """Fit ``F/(sigma_c h^2) = K (1 - n)^alpha`` in log-log space.

    Parameters
    ----------
    solids_fraction : sequence of ``1 - n`` (rubble solids fraction)
    normalised_load : sequence of ``F_max / (sigma_c h^2)``

    Returns ``alpha``, ``K``, ``R2`` and the bootstrap 95 % interval of
    ``alpha`` plus the log-lever arm (diagnostic: a small lever arm means the
    exponent is not identifiable, however good ``R2`` looks).
    """
    x = np.asarray(solids_fraction, dtype=float)
    y = np.asarray(normalised_load, dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 points to fit alpha")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x and y must be positive")
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # residual bootstrap for the exponent's uncertainty
    rng = np.random.default_rng(12345)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, x.size, x.size)
        if np.unique(lx[idx]).size < 2:
            continue
        boots.append(np.polyfit(lx[idx], ly[idx], 1)[0])
    boots = np.array(boots)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots.size else (float("nan"), float("nan"))

    return {
        "alpha": float(slope),
        "K": float(math.exp(intercept)),
        "R2": float(r2),
        "alpha_ci95": list(ci),
        "n_points": int(x.size),
        "log_lever_arm": float(lx.max() - lx.min()),
        "solids_fraction_range": [float(x.min()), float(x.max())],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _quick_config(**kw) -> PileupVerticalConfig:
    base = dict(h=0.6, D=3.0, sigma_c=3.0e6, drift_speed=0.5,
                sheet_length_factor=12.0, platen_travel_factor=2.0)
    base.update(kw)
    return PileupVerticalConfig(**base)


def main(argv=None) -> int:
    """``python -m tensordem.ice_rubble_pileup_vertical`` entry point."""
    ap = argparse.ArgumentParser(
        description="2-D vertical-section DEM of ice rubble pile-up against a "
                    "vertical structure with an ISO 19906 check")
    ap.add_argument("--sigma-c", type=float, default=3.0e6, help="ice strength [Pa]")
    ap.add_argument("--h", type=float, default=1.0, help="ice thickness [m]")
    ap.add_argument("--D", type=float, default=5.0, help="structure width [m]")
    ap.add_argument("--shape", choices=["cylinder", "square"], default="cylinder")
    ap.add_argument("--speed", type=float, default=0.5, help="drift speed [m/s]")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--quick", action="store_true", help="shorter, cheaper case")
    ap.add_argument("--out", type=str, default="", help="write JSON result here")
    args = ap.parse_args(argv)

    cfg = PileupVerticalConfig(sigma_c=args.sigma_c, h=args.h, D=args.D,
                               shape=args.shape, drift_speed=args.speed,
                               seed=args.seed)
    if args.quick:
        cfg = _quick_config(sigma_c=args.sigma_c, h=args.h, D=args.D,
                            shape=args.shape, drift_speed=args.speed,
                            seed=args.seed)
    res = run_vertical_pileup_case(cfg)
    payload = res.as_dict(with_series=False)
    payload["iso"] = iso_19906_reference(cfg)
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())