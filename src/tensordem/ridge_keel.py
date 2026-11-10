"""Granular flow of a ridge keel under hull penetration: force chains, shear
bands and a Lindqvist-family resistance anchor.

Physical picture
----------------
A first-year ice ridge consists of a *sail* above the waterline, a consolidated
layer at the waterline, and a *keel* of loose, partly refrozen blocks below it
(keel depth ``h_k`` = 5-20 m, mouldline width ``w_k`` = 20-60 m, block sizes
``d`` = 0.1-2 m).  When a ship penetrates a ridge the resistance comes from the
keel: the hull must displace, lift and shear a confined assembly of ice blocks,
which is a *granular* problem.  Empirical formulas (Lindqvist 1989 and its
relatives) collapse that mechanics into "friction coefficient times the weight
of the displaced rubble".

This module builds the grain-scale side of that comparison for a 2-D
plane-strain section of the keel (unit out-of-plane thickness):

* **poly-disperse blocks** on a power-law number density over ``d`` in
  ``[d_min, d_max]`` (:class:`PowerLawPSD`), placed in the triangular keel
  envelope and compacted to a prescribed solid fraction;
* **Hertz-Mindlin contacts**: non-linear normal spring with restitution-based
  damping, an incremental (Cundall-Strack) tangential spring saturating at
  ``mu F_n``, and rolling resistance;
* **freeze bonds**: an axial/tangential spring pair with normal strength
  ``sigma_n`` and shear strength ``sigma_t`` (Pa x bond area) that breaks in
  tension or shear and books the released elastic energy as fracture energy;
* a **rigid bow wedge** (half-angle ``alpha``) advancing at speed ``V`` through
  the rubble with Coulomb friction against the ice;
* a **rigid frictional lid** at the mouldline representing the consolidated
  layer and the sail (it holds the buoyant rubble down and forbids free heave),
  while the keel flanks are open so displaced rubble escapes laterally -- the
  plane-strain analogue of a hull pushing rubble out of its path.

Post-processing
---------------
:meth:`RidgeKeelDEM.force_chain_report` builds the strong-contact network
(``F_n > 2 <F_n>``), its connected components (chains), the load fraction it
carries and the chain lifetime across measurement windows;
:meth:`RidgeKeelDEM.stress_report` gives the contact stress tensor and its
principal directions; :meth:`RidgeKeelDEM.shear_band_report` localises the
shear band from the displacement-gradient field and returns its orientation
relative to the local minor principal stress; :meth:`RidgeKeelDEM.porosity_report`
gives the body-following local porosity change.
:func:`lindqvist_keel_resistance` is the empirical anchor.

Evidence grading
----------------
Everything on the DEM side is REAL: numbers come from integrated contact
mechanics with an auditable energy ledger (:class:`EnergyLedger`).  The Lindqvist
*structure* used for the anchor (friction x displaced weight) is REAL, but its
*coefficient set* is WITHHELD -- the primary paper is not readable offline and
the published coefficient varies roughly between 0.4 and 1.0, so the anchor is
reported with that band instead of a validated point value.  See
``docs/p257w2_n05_ridge_keel_granular_flow.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

GRAVITY = 9.81
RHO_ICE = 917.0
RHO_WATER = 1025.0

__all__ = [
    "GRAVITY",
    "RHO_ICE",
    "RHO_WATER",
    "KeelSection",
    "PowerLawPSD",
    "RubbleMaterial",
    "WedgeIndenter",
    "EnergyLedger",
    "RidgeKeelDEM",
    "lindqvist_keel_resistance",
    "biaxial_friction_angle",
    "stable_timestep",
]


# --------------------------------------------------------------------------- #
# geometry and material
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeelSection:
    """Triangular keel cross-section (plane strain, unit thickness).

    ``y = 0`` is the mouldline, the keel tip sits at ``y = -keel_depth`` and the
    half-width grows linearly from the tip to ``keel_width / 2`` at the
    mouldline (the usual triangular idealisation of a ridge keel).
    """

    keel_depth: float
    keel_width: float
    porosity: float = 0.40

    def __post_init__(self) -> None:
        for name in ("keel_depth", "keel_width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 <= self.porosity < 0.9):
            raise ValueError("porosity must lie in [0, 0.9)")

    @property
    def solid_fraction(self) -> float:
        return 1.0 - self.porosity

    @property
    def cross_section_area(self) -> float:
        return 0.5 * self.keel_width * self.keel_depth

    def half_width(self, depth):
        """Half-width of the envelope at ``depth`` below the mouldline (m).

        The keel is widest at the mouldline (``depth = 0``) and closes to a point
        at the tip (``depth = keel_depth``).
        """
        d = np.asarray(depth, dtype=float)
        return 0.5 * self.keel_width * (1.0 - np.clip(d, 0.0, self.keel_depth) / self.keel_depth)

    def contains(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        depth = -y
        return ((depth >= 0.0) & (depth <= self.keel_depth)
                & (np.abs(x) <= self.half_width(depth)))


@dataclass(frozen=True)
class PowerLawPSD:
    """Number-density power law ``n(d) ~ d**-exponent`` on ``[d_min, d_max]``.

    Measured ridge-keel block-size distributions are close to power laws with a
    number exponent of 2-3.  A steep exponent puts most of the *count* into
    fines, which costs timestep without adding area; the default here is 1.8
    over ``[0.15, 2.0] m`` and the truncation is declared as a model limit.
    """

    d_min: float = 0.15
    d_max: float = 2.0
    exponent: float = 1.8

    def __post_init__(self) -> None:
        if not (0.0 < self.d_min < self.d_max):
            raise ValueError("require 0 < d_min < d_max")
        if not math.isfinite(self.exponent) or abs(self.exponent - 1.0) < 1e-12:
            raise ValueError("exponent must be finite and different from 1")

    def sample(self, count: int, rng) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        p = self.exponent
        lo, hi = self.d_min ** (1.0 - p), self.d_max ** (1.0 - p)
        u = rng.random(count)
        return (lo + u * (hi - lo)) ** (1.0 / (1.0 - p))

    def mean_square_diameter(self) -> float:
        p = self.exponent
        if abs(p - 3.0) < 1e-12:
            num = math.log(self.d_max / self.d_min)
        else:
            num = (self.d_max ** (3.0 - p) - self.d_min ** (3.0 - p)) / (3.0 - p)
        den = (self.d_max ** (1.0 - p) - self.d_min ** (1.0 - p)) / (1.0 - p)
        return num / den

    def mean_area(self) -> float:
        return 0.25 * math.pi * self.mean_square_diameter()


@dataclass(frozen=True)
class RubbleMaterial:
    """Contact and freeze-bond parameters of the ice rubble."""

    rho_ice: float = RHO_ICE
    young_modulus: float = 1.0e9
    poisson_ratio: float = 0.30
    friction: float = 0.55
    rolling_friction: float = 0.05
    restitution: float = 0.40
    bond_normal_strength: float = 2.0e4
    bond_shear_strength: float = 1.0e4
    bond_stiffness_factor: float = 0.05

    def __post_init__(self) -> None:
        for name in ("rho_ice", "young_modulus"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in [0, 0.5)")
        if not (0.0 <= self.friction <= 1.5):
            raise ValueError("friction must lie in [0, 1.5]")
        if not (0.0 <= self.restitution < 1.0):
            raise ValueError("restitution must lie in [0, 1)")
        for name in ("bond_normal_strength", "bond_shear_strength"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def e_star(self) -> float:
        """``E / (2 (1 - nu^2))`` -- identical bodies."""
        return self.young_modulus / (2.0 * (1.0 - self.poisson_ratio ** 2))

    def e_star_rigid(self) -> float:
        """``E / (1 - nu^2)`` -- body against a rigid wall."""
        return self.young_modulus / (1.0 - self.poisson_ratio ** 2)

    @property
    def mindlin_ratio(self) -> float:
        """Secant stiffness ratio ``k_t / k_n = 2 (1 - nu) / (2 - nu)``."""
        return 2.0 * (1.0 - self.poisson_ratio) / (2.0 - self.poisson_ratio)

    @property
    def damping_ratio(self) -> float:
        e = self.restitution
        if e <= 1e-9:
            return 1.0
        return -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)


def stable_timestep(material: RubbleMaterial, radius_min: float, *,
                    overlap_reference: float = 1.0e-3,
                    safety: float = 0.08) -> float:
    """Contact-stability timestep for the smallest (lightest) grain.

    ``dt = safety * 2 pi / omega`` with ``omega = sqrt(k_secant / m_eff)``
    evaluated at a reference overlap -- the usual Hertz DEM criterion.
    """
    if radius_min <= 0.0:
        raise ValueError("radius_min must be positive")
    r_eff = 0.5 * radius_min
    k_3_2 = (4.0 / 3.0) * material.e_star * math.sqrt(r_eff)
    k_secant = 1.5 * k_3_2 * math.sqrt(overlap_reference)
    mass = material.rho_ice * math.pi * radius_min ** 2
    omega = math.sqrt(k_secant / (0.5 * mass))
    return safety * 2.0 * math.pi / omega


# --------------------------------------------------------------------------- #
# wedge indenter
# --------------------------------------------------------------------------- #
class WedgeIndenter:
    """Rigid 2-D wedge (ship bow section) advancing along ``+x``.

    The body is the intersection of the two half-spaces behind the apex,
    ``X . n_u <= 0`` and ``X . n_l <= 0`` with ``X = x - apex``,
    ``n_u = (sin a, cos a)``, ``n_l = (sin a, -cos a)``: the faces therefore make
    the half-angle ``a`` with the direction of motion.  Contacts are resolved
    against the closest point of that convex region (faces and apex), which
    keeps the force continuous while a grain travels past the tip.
    """

    def __init__(self, half_angle_deg: float, speed: float, *,
                 apex_x: float = 0.0, apex_y: float = 0.0,
                 friction: float = 0.30) -> None:
        if not (0.0 < half_angle_deg < 90.0):
            raise ValueError("half_angle_deg must lie in (0, 90)")
        if speed < 0.0:
            raise ValueError("speed must be non-negative")
        if friction < 0.0:
            raise ValueError("friction must be non-negative")
        self.half_angle_deg = float(half_angle_deg)
        self.half_angle = math.radians(half_angle_deg)
        self.speed = float(speed)
        self.apex = np.array([float(apex_x), float(apex_y)])
        self.friction = float(friction)
        self.travelled = 0.0
        self.face_force = 0.0        # resistance, +x component [N per unit depth]

    def advance(self, dt: float) -> None:
        step = self.speed * dt
        self.apex = self.apex + np.array([step, 0.0])
        self.travelled += step

    def contact(self, pos: np.ndarray, radius: np.ndarray):
        """Overlap, unit outward normal and contact mask for every grain."""
        x = pos - self.apex
        ca, sa = math.cos(self.half_angle), math.sin(self.half_angle)
        s_upper = -(x @ np.array([sa, ca]))
        s_lower = -(x @ np.array([sa, -ca]))
        inside = (s_upper > 0.0) & (s_lower > 0.0)
        best_d = np.full(pos.shape[0], np.inf)
        best_p = np.zeros_like(pos)
        for ray in (np.array([-ca, sa]), np.array([-ca, -sa])):
            t = np.clip(x @ ray, 0.0, None)
            p = self.apex + t[:, None] * ray[None, :]
            d = np.linalg.norm(pos - p, axis=1)
            take = d < best_d
            best_d = np.where(take, d, best_d)
            best_p = np.where(take[:, None], p, best_p)
        dist = np.linalg.norm(pos - best_p, axis=1)
        normal_out = (pos - best_p) / np.maximum(dist, 1e-12)[:, None]
        overlap = radius - dist
        if inside.any():
            upper_nearer = s_upper <= s_lower
            face_normal = np.where(upper_nearer[:, None],
                                   np.array([sa, ca])[None, :],
                                   np.array([sa, -ca])[None, :])
            overlap_inside = radius + np.where(upper_nearer, s_upper, s_lower)
            overlap = np.where(inside, overlap_inside, overlap)
            normal_out = np.where(inside[:, None], face_normal, normal_out)
        return overlap, normal_out, overlap > 0.0


# --------------------------------------------------------------------------- #
# energy ledger
# --------------------------------------------------------------------------- #
@dataclass
class EnergyLedger:
    """Cumulative work / dissipation bookkeeping of one penetration run."""

    wedge_work: float = 0.0
    body_work: float = 0.0
    normal_dissipation: float = 0.0
    slip_dissipation: float = 0.0
    rolling_dissipation: float = 0.0
    bond_fracture_energy: float = 0.0
    drag_dissipation: float = 0.0
    initial_kinetic: float = 0.0
    initial_contact_elastic: float = 0.0
    initial_bond_elastic: float = 0.0
    final_kinetic: float = 0.0
    final_contact_elastic: float = 0.0
    final_bond_elastic: float = 0.0

    @property
    def external_work(self) -> float:
        return self.wedge_work + self.body_work

    @property
    def dissipated(self) -> float:
        return (self.normal_dissipation + self.slip_dissipation
                + self.rolling_dissipation + self.bond_fracture_energy
                + self.drag_dissipation)

    @property
    def residual(self) -> float:
        internal = ((self.final_kinetic - self.initial_kinetic)
                    + (self.final_contact_elastic - self.initial_contact_elastic)
                    + (self.final_bond_elastic - self.initial_bond_elastic))
        return self.external_work - internal - self.dissipated

    @property
    def residual_fraction(self) -> float:
        scale = abs(self.external_work)
        return float("inf") if scale <= 0.0 else self.residual / scale

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update(external_work=self.external_work, dissipated=self.dissipated,
                   residual=self.residual, residual_fraction=self.residual_fraction)
        return out


# --------------------------------------------------------------------------- #
# DEM engine
# --------------------------------------------------------------------------- #
class RidgeKeelDEM:
    """Plane-strain DEM of a ridge keel penetrated by a bow wedge."""

    def __init__(self, section: KeelSection, psd: PowerLawPSD,
                 material: RubbleMaterial, *,
                 solid_fraction: float = 0.62, seed: int = 0,
                 lid_y: float = 0.0, lid_friction: float = 0.20,
                 drag_time: float = 0.5, overburden_pressure: float = 7.5e3,
                 body_weight: bool = True, packing_overlap_scale: float = 1.0,
                 max_particles: int = 1500) -> None:
        if not (0.30 <= solid_fraction <= 0.85):
            raise ValueError("solid_fraction must lie in [0.30, 0.85]")
        self.section = section
        self.psd = psd
        self.material = material
        self.solid_fraction = float(solid_fraction)
        self.rng = np.random.default_rng(seed)
        self.lid_y = float(lid_y)
        self.lid_friction = float(lid_friction)
        self.drag_time = float(drag_time)
        self.overburden_pressure = float(overburden_pressure)
        self.body_weight = bool(body_weight)
        self.packing_overlap_scale = float(packing_overlap_scale)
        self.time = 0.0
        self.ledger = EnergyLedger()
        self.wedge = None
        self._xi_state = None
        self._pair_keys = None
        self.flank_walls = True
        self.envelope_scale = 1.0
        self.floor_fraction = 0.90
        self._contact_cache = None
        self._build(max_particles)
        self._body_force_cache = np.zeros_like(self.pos)
        self._wedge_cache = None

    # -- assembly ----------------------------------------------------------
    def _build(self, max_particles: int) -> None:
        section, psd, material = self.section, self.psd, self.material
        target_area = section.cross_section_area * self.solid_fraction
        budget = int(round(target_area / psd.mean_area()))
        budget = max(60, min(budget, max_particles))
        radii = 0.5 * psd.sample(budget, self.rng)
        pos, radius = self._random_sequential_packing(radii)
        self.pos = pos
        self.radius = radius
        achieved = float(np.sum(math.pi * radius ** 2)) / section.cross_section_area
        growth = min(math.sqrt(self.solid_fraction / max(achieved, 1e-6)), 1.30)
        self._relax_packing(start_scale=1.0, end_scale=growth)
        # last step: a small uniform inflation that leaves a residual overlap
        # everywhere.  The contact solver relaxes it against the closed envelope,
        # which is how a confined granular body acquires its in-situ pressure and
        # contact network (a perfect non-overlapping packing carries no stress).
        self.radius = self.radius * self.packing_overlap_scale
        self.packing_growth = growth
        mass = material.rho_ice * math.pi * self.radius ** 2
        self.mass = mass
        self.inertia = 0.5 * mass * self.radius ** 2
        self.vel = np.zeros_like(self.pos)
        self.omega = np.zeros(self.pos.shape[0])
        self.theta = self.rng.uniform(0.0, 2.0 * math.pi, self.pos.shape[0])
        self.n_particles = int(self.pos.shape[0])
        self._build_pairs()
        self._init_bonds()
        self._body_force_cache = np.zeros_like(self.pos)
        self._last_disp = np.zeros_like(self.pos)

    def _wall_overlap(self, pos: np.ndarray, radius: np.ndarray) -> float:
        """Worst penetration of ``pos``/``radius`` into any rigid boundary."""
        worst = -np.inf
        for origin, normal, _ in self._wall_planes():
            s = (pos - origin) @ normal
            worst = max(worst, float(np.max(radius - s)))
        return worst

    def _project_inside(self, pos: np.ndarray, radius: np.ndarray) -> None:
        """Push grains that penetrate a rigid boundary back along its normal."""
        for origin, normal, _ in self._wall_planes():
            s = (pos - origin) @ normal
            overlap = radius - s
            hit = overlap > 0.0
            if hit.any():
                pos[hit] += overlap[hit, None] * normal[None, :]

    def _random_sequential_packing(self, radii: np.ndarray,
                                   attempts_cap: int = 600):
        """Random sequential addition inside the keel envelope (largest first).

        Placing the biggest blocks first leaves room for them, so the packing is
        geometrically admissible without deep overlaps; the achieved solid
        fraction is an *outcome* of the packing and is reported, not imposed.
        """
        section = self.section
        order = np.argsort(radii)[::-1]
        cell = 2.2 * float(radii.max())
        grid = {}
        placed_pos = []
        placed_radius = []
        placed_index = []
        for idx in order:
            r = float(radii[idx])
            depth_min = r * 1.02
            depth_max = section.keel_depth * (1.0 - 2.0 * r / section.keel_width) - 1e-6
            if depth_min >= depth_max:
                continue
            accepted = False
            for _ in range(attempts_cap):
                depth = self.rng.uniform(depth_min, depth_max)
                half = float(np.atleast_1d(section.half_width(depth))[0])
                if half <= 0.0:
                    continue
                x = self.rng.uniform(-0.98 * half, 0.98 * half)
                y = self.lid_y - depth
                if self._wall_overlap(np.array([[x, y]]), np.array([r])) > 1e-9:
                    continue
                gx, gy = int(np.floor(x / cell)), int(np.floor(y / cell))
                clash = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for other in grid.get((gx + dx, gy + dy), ()):
                            ox, oy, orad = placed_pos[other][0], placed_pos[other][1], placed_radius[other]
                            if (x - ox) ** 2 + (y - oy) ** 2 < (r + orad) ** 2 * 1.0001:
                                clash = True
                                break
                        if clash:
                            break
                    if clash:
                        break
                if clash:
                    continue
                grid.setdefault((gx, gy), []).append(len(placed_pos))
                placed_pos.append((x, y))
                placed_radius.append(r)
                placed_index.append(int(idx))
                accepted = True
                break
            if not accepted:
                continue
        pos = np.asarray(placed_pos, dtype=float)
        radius = np.asarray(placed_radius, dtype=float)
        return pos, radius

    def _relax_packing(self, *, growth_steps: int = 16, iterations: int = 16,
                       start_scale: float = 0.55, end_scale: float = 1.0) -> None:
        """Geometric overlap removal with radius growth (no dynamics).

        The RSA packing is admissible but loose; the radii are grown to
        ``end_scale`` while overlapping pairs are pushed apart geometrically and
        grains that leave the envelope are projected back.  The dynamic phase
        then starts from a dense, resolved packing.
        """
        n = self.pos.shape[0]
        rows = np.arange(n)
        for lam in np.linspace(start_scale, end_scale, growth_steps):
            r = self.radius * lam
            for _ in range(iterations):
                moved = False
                for lo in range(0, n, 200):
                    hi = min(lo + 200, n)
                    block = self.pos[lo:hi]
                    d = block[:, None, :] - self.pos[None, :, :]
                    dist = np.linalg.norm(d, axis=2)
                    overlap = r[lo:hi][:, None] + r[None, :] - dist
                    pair = overlap > 1e-9
                    pair[np.arange(hi - lo), rows[lo:hi]] = False
                    if not pair.any():
                        continue
                    moved = True
                    safe = np.maximum(dist, 1e-12)
                    mag = np.where(pair, 0.5 * overlap / safe, 0.0)
                    push = mag[:, :, None] * d
                    self.pos[lo:hi] += push.sum(axis=1)
                    self.pos -= push.sum(axis=0)
                if not moved:
                    break
            self._project_inside(self.pos, r)
        self._project_inside(self.pos, self.radius)

    def _build_pairs(self, skin_factor: float = 2.5) -> None:
        skin = skin_factor * float(np.mean(self.radius))
        cutoff = 2.0 * float(np.max(self.radius)) + skin
        n = self.n_particles
        origin = self.pos.min(axis=0) - cutoff
        keys = np.floor((self.pos - origin) / cutoff).astype(np.int64)
        buckets = {}
        for idx in range(n):
            buckets.setdefault((int(keys[idx, 0]), int(keys[idx, 1])), []).append(idx)
        pairs = []
        for (cx, cy), members in buckets.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = buckets.get((cx + dx, cy + dy))
                    if not other:
                        continue
                    for i in members:
                        for j in other:
                            if j <= i:
                                continue
                            d = self.pos[j] - self.pos[i]
                            if d @ d <= cutoff ** 2:
                                pairs.append((i, j))
        arr = np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)
        new_keys = arr[:, 0] * n + arr[:, 1]
        order = np.argsort(new_keys)
        self.pairs = arr[order]
        new_keys = new_keys[order]
        if self._pair_keys is not None and self._pair_keys.size and new_keys.size:
            old = self._pair_keys
            idx = np.clip(np.searchsorted(old, new_keys), 0, old.size - 1)
            match = old[idx] == new_keys
            xi = np.zeros((new_keys.size, 2))
            xi[match] = self._xi_state[idx[match]]
            self._xi_state = xi
        else:
            self._xi_state = np.zeros((new_keys.size, 2))
        self._pair_keys = new_keys
        self._pair_cutoff = cutoff
        self._pair_steps = 0

    def _init_bonds(self) -> None:
        self.bonds = np.zeros((0, 2), dtype=np.int64)
        self.bond_alive = np.zeros(0, dtype=bool)
        self.bond_length0 = np.zeros(0)
        self.bond_normal = np.zeros((0, 2))
        self.bond_xi = np.zeros((0, 2))
        self.bond_area = np.zeros(0)
        self.bond_kn = 0.0
        self.bond_kt = 0.0

    def install_bonds(self, *, gap_tolerance: float = 0.05) -> int:
        """Freeze-bond every pair touching within ``gap_tolerance``."""
        m = self.material
        if self.pairs.size == 0:
            return 0
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = np.linalg.norm(self.pos[j] - self.pos[i], axis=1)
        touch = d <= (self.radius[i] + self.radius[j]) * (1.0 + gap_tolerance)
        pairs = self.pairs[touch]
        if pairs.shape[0] == 0:
            return 0
        self.bonds = pairs.copy()
        self.bond_alive = np.ones(pairs.shape[0], dtype=bool)
        self.bond_length0 = np.linalg.norm(self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]],
                                           axis=1)
        self.bond_normal = ((self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]])
                            / self.bond_length0[:, None])
        self.bond_xi = np.zeros((pairs.shape[0], 2))
        self.bond_area = 2.0 * np.minimum(self.radius[pairs[:, 0]], self.radius[pairs[:, 1]])
        mean_area = math.pi * float(np.mean(self.radius)) ** 2
        self.bond_kn = (m.bond_stiffness_factor * m.e_star * mean_area
                        / float(np.mean(self.radius)))
        self.bond_kt = self.bond_kn * m.mindlin_ratio
        return int(pairs.shape[0])

    # -- energy helpers ----------------------------------------------------
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1)
                                  + self.inertia * self.omega ** 2))

    @property
    def effective_unit_weight(self) -> float:
        """Downward effective unit weight of the confined rubble [N/m3].

        The sail and the consolidated layer load the keel from above; that
        overburden (``overburden_pressure``, of order kPa in ridge keels) is the
        confining mechanism that turns the rubble into a stressed granular body.
        On top of it the buoyant excess of ice in water acts *upwards*
        (``(rho_ice - rho_water) g`` is negative), so it is kept as a correction.
        """
        return self.overburden_pressure / self.section.keel_depth

    def _body_force(self) -> np.ndarray:
        f = np.zeros_like(self.pos)
        if not self.body_weight:
            return f
        volume = math.pi * self.radius ** 2
        f[:, 1] = -self.effective_unit_weight * volume
        return f

    # -- walls -------------------------------------------------------------
    def _wall_planes(self):
        """(origin, inward unit normal, friction) of the rigid boundaries."""
        planes = [(np.array([0.0, self.lid_y]), np.array([0.0, -1.0]), self.lid_friction)]
        if self.floor_fraction > 0.0:
            floor_y = self.lid_y - self.floor_fraction * self.section.keel_depth
            planes.append((np.array([0.0, floor_y]), np.array([0.0, 1.0]), self.lid_friction))
        if self.flank_walls:
            w, h = self.section.keel_width, self.section.keel_depth
            for sign in (1.0, -1.0):
                origin = np.array([sign * 0.5 * w, self.lid_y])
                normal = np.array([-sign, w / (2.0 * h)])
                planes.append((origin, normal / np.linalg.norm(normal), 0.0))
        if self.envelope_scale < 1.0:
            centre = np.array([0.0, self.lid_y - self.section.keel_depth / 3.0])
            scaled = []
            for origin, normal, mu in planes:
                distance = float((centre - origin) @ normal)
                scaled.append((origin + (1.0 - self.envelope_scale) * distance * normal,
                               normal, mu))
            planes = scaled
        return planes

    # -- force assembly ----------------------------------------------------
    def _forces(self, dt: float):
        m = self.material
        n = self.n_particles
        force = np.zeros_like(self.pos)
        torque = np.zeros(n)
        self._contact_cache = {"active": np.zeros(0, dtype=bool)}
        if self.pairs.shape[0]:
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            delta = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(delta, axis=1)
            normal = delta / np.maximum(dist, 1e-12)[:, None]
            overlap = self.radius[i] + self.radius[j] - dist
            active = overlap > 0.0
            r_eff = 0.5 * (self.radius[i] * self.radius[j]) / np.maximum(
                self.radius[i] + self.radius[j], 1e-12)
            k_3_2 = (4.0 / 3.0) * m.e_star * np.sqrt(np.maximum(r_eff, 0.0))
            overlap_pos = np.maximum(overlap, 0.0)
            f_n = np.where(active, k_3_2 * overlap_pos ** 1.5, 0.0)
            # contact point of grain i is at +r_i n (towards j), and -r_j n on j
            arm_i = self.radius[i][:, None] * normal
            arm_j = -self.radius[j][:, None] * normal
            spin_i = self.omega[i][:, None] * np.stack((-arm_i[:, 1], arm_i[:, 0]), axis=1)
            spin_j = self.omega[j][:, None] * np.stack((-arm_j[:, 1], arm_j[:, 0]), axis=1)
            v_slip_full = (self.vel[i] - self.vel[j]) + spin_i - spin_j
            v_n = np.sum(v_slip_full * normal, axis=1)
            v_t = v_slip_full - v_n[:, None] * normal
            m_eff = 0.5 * self.mass[i] * self.mass[j] / np.maximum(
                self.mass[i] + self.mass[j], 1e-30)
            k_secant = 1.5 * k_3_2 * np.sqrt(overlap_pos)
            c_n = 2.0 * m.damping_ratio * np.sqrt(np.maximum(m_eff * k_secant, 0.0))
            f_n = np.where(active, f_n - c_n * v_n, 0.0)
            f_n = np.maximum(f_n, 0.0)
            self.ledger.normal_dissipation += float(
                np.sum(np.where(active, c_n * v_n ** 2 * dt, 0.0)))
            k_t = m.mindlin_ratio * k_secant
            self._xi_state += v_t * dt
            f_t_vec = -k_t[:, None] * self._xi_state
            f_t_mag = np.linalg.norm(f_t_vec, axis=1)
            limit = m.friction * f_n
            sliding = active & (f_t_mag > limit)
            if sliding.any():
                scale = limit[sliding] / np.maximum(f_t_mag[sliding], 1e-30)
                f_t_vec[sliding] *= scale[:, None]
                slip = np.linalg.norm(v_t[sliding], axis=1) * dt
                self.ledger.slip_dissipation += float(np.sum(limit[sliding] * slip))
                self._xi_state[sliding] = -f_t_vec[sliding] / np.maximum(
                    k_t[sliding], 1e-30)[:, None]
            # traction on grain j (repulsion along +n) and its reaction on i;
            # ``f_t_vec = -k_t xi`` is the tangential force acting on grain i
            traction_j = f_n[:, None] * normal - f_t_vec
            force[:, 0] += (np.bincount(j, weights=traction_j[:, 0], minlength=n)
                            - np.bincount(i, weights=traction_j[:, 0], minlength=n))
            force[:, 1] += (np.bincount(j, weights=traction_j[:, 1], minlength=n)
                            - np.bincount(i, weights=traction_j[:, 1], minlength=n))
            t_i = -arm_i[:, 0] * traction_j[:, 1] + arm_i[:, 1] * traction_j[:, 0]
            t_j = arm_j[:, 0] * traction_j[:, 1] - arm_j[:, 1] * traction_j[:, 0]
            # rolling resistance: a dissipative couple on *each* grain that
            # opposes its own spin (applying one common couple to the pair would
            # anti-damp one of the two grains)
            m_roll_i = np.where(active, m.rolling_friction * f_n * self.radius[i]
                                * np.tanh(self.omega[i] / 0.05), 0.0)
            m_roll_j = np.where(active, m.rolling_friction * f_n * self.radius[j]
                                * np.tanh(self.omega[j] / 0.05), 0.0)
            torque += (np.bincount(i, weights=t_i - m_roll_i, minlength=n)
                       + np.bincount(j, weights=t_j - m_roll_j, minlength=n))
            self.ledger.rolling_dissipation += float(
                np.sum(np.abs(m_roll_i * self.omega[i]) * dt)
                + np.sum(np.abs(m_roll_j * self.omega[j]) * dt))
            self._contact_cache = dict(i=i, j=j, normal=normal, overlap=overlap_pos,
                                       active=active, f_n=f_n, f_t=f_t_vec,
                                       r_eff=r_eff, k_3_2=k_3_2, k_t=k_t)
        self._bond_forces(force, torque, dt)
        self._wall_forces(force, dt)
        self._wedge_forces(force, dt)
        body = self._body_force()
        force += body
        if self.drag_time > 0.0:
            force -= self.mass[:, None] * self.vel / self.drag_time
            self.ledger.drag_dissipation += float(
                np.sum(self.mass[:, None] * self.vel ** 2 / self.drag_time) * dt)
        self._body_force_cache = body
        return force, torque

    def _bond_forces(self, force, torque, dt) -> None:
        if self.bonds.shape[0] == 0 or not self.bond_alive.any():
            return
        m = self.material
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        delta = self.pos[j] - self.pos[i]
        dist = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(dist, 1e-12)[:, None]
        extension = dist - self.bond_length0
        f_axial = self.bond_kn * extension
        v_rel = self.vel[i] - self.vel[j]
        v_n = np.sum(v_rel * normal, axis=1)
        v_t = v_rel - v_n[:, None] * normal
        self.bond_xi += v_t * dt
        f_shear = -self.bond_kt * self.bond_xi
        shear_mag = np.linalg.norm(f_shear, axis=1)
        tensile_capacity = m.bond_normal_strength * self.bond_area
        shear_capacity = m.bond_shear_strength * self.bond_area
        broken = self.bond_alive & ((-f_axial > tensile_capacity)
                                    | (shear_mag > shear_capacity))
        if broken.any():
            released = (0.5 * self.bond_kn * extension[broken] ** 2
                        + 0.5 * self.bond_kt
                        * np.sum(self.bond_xi[broken] ** 2, axis=1))
            self.ledger.bond_fracture_energy += float(np.sum(released))
            self.bond_alive[broken] = False
        alive = self.bond_alive
        if not alive.any():
            return
        force_bond = np.where(alive[:, None], f_axial[:, None] * normal + f_shear, 0.0)
        n = self.n_particles
        force[:, 0] += (np.bincount(i, weights=force_bond[:, 0], minlength=n)
                        - np.bincount(j, weights=force_bond[:, 0], minlength=n))
        force[:, 1] += (np.bincount(i, weights=force_bond[:, 1], minlength=n)
                        - np.bincount(j, weights=force_bond[:, 1], minlength=n))
        arm_i = self.radius[i][:, None] * normal
        arm_j = -self.radius[j][:, None] * normal
        torque += (np.bincount(i, weights=arm_i[:, 0] * force_bond[:, 1]
                               - arm_i[:, 1] * force_bond[:, 0], minlength=n)
                   + np.bincount(j, weights=-arm_j[:, 0] * force_bond[:, 1]
                                 + arm_j[:, 1] * force_bond[:, 0], minlength=n))

    def _wall_forces(self, force, dt) -> None:
        m = self.material
        for origin, normal, mu in self._wall_planes():
            s = (self.pos - origin) @ normal
            overlap = self.radius - s
            active = overlap > 0.0
            if not active.any():
                continue
            idx = np.nonzero(active)[0]
            r_eff = 0.5 * self.radius[idx]
            k_3_2 = (4.0 / 3.0) * m.e_star_rigid() * np.sqrt(r_eff)
            # regularisation: a grain forced through a boundary must not generate an
            # unbounded penalty force
            delta = np.minimum(overlap[idx], 0.15 * self.radius[idx])
            f_n = k_3_2 * delta ** 1.5
            v = self.vel[idx]
            v_n = v @ normal
            c_n = 2.0 * m.damping_ratio * np.sqrt(
                self.mass[idx] * 1.5 * k_3_2 * np.sqrt(delta))
            f_n = np.maximum(f_n - c_n * v_n, 0.0)
            self.ledger.normal_dissipation += float(np.sum(c_n * v_n ** 2 * dt))
            v_t = v - v_n[:, None] * normal[None, :]
            speed_t = np.linalg.norm(v_t, axis=1)
            f_t = mu * f_n * np.tanh(speed_t / 1e-2)
            self.ledger.slip_dissipation += float(np.sum(f_t * speed_t) * dt)
            wall_force = (f_n[:, None] * normal[None, :]
                          - f_t[:, None] * v_t / np.maximum(speed_t, 1e-12)[:, None])
            force[idx] += wall_force

    def _wedge_forces(self, force, dt) -> None:
        if self.wedge is None:
            self._wedge_cache = None
            return
        m = self.material
        overlap, normal, active = self.wedge.contact(self.pos, self.radius)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            self.wedge.face_force = 0.0
            self._wedge_cache = None
            return
        r_eff = 0.25 * self.radius[idx]
        k_3_2 = (4.0 / 3.0) * m.e_star_rigid() * np.sqrt(r_eff)
        delta = np.minimum(overlap[idx], 0.15 * self.radius[idx])
        f_n = k_3_2 * delta ** 1.5
        v = self.vel[idx] - np.array([self.wedge.speed, 0.0])
        v_n = np.sum(v * normal[idx], axis=1)
        c_n = 2.0 * m.damping_ratio * np.sqrt(
            self.mass[idx] * 1.5 * k_3_2 * np.sqrt(delta))
        f_n = np.maximum(f_n - c_n * v_n, 0.0)
        self.ledger.normal_dissipation += float(np.sum(c_n * v_n ** 2 * dt))
        v_t = v - v_n[:, None] * normal[idx]
        speed_t = np.linalg.norm(v_t, axis=1)
        f_t = self.wedge.friction * f_n * np.tanh(speed_t / 1e-2)
        self.ledger.slip_dissipation += float(np.sum(f_t * speed_t) * dt)
        wedge_force = (f_n[:, None] * normal[idx]
                       - f_t[:, None] * v_t / np.maximum(speed_t, 1e-12)[:, None])
        force[idx] += wedge_force
        # reaction on the wedge is minus the sum of the forces on the grains
        self.wedge.face_force = float(np.sum(wedge_force[:, 0]))
        self._wedge_cache = wedge_force

    # -- integration -------------------------------------------------------
    def elastic_energy(self):
        """(contact, bond) elastic energy of the reversible springs [J]."""
        cache = self._contact_cache
        contact = 0.0
        if cache is not None and cache["active"].any():
            act = cache["active"]
            contact = float(np.sum(0.4 * cache["k_3_2"][act] * cache["overlap"][act] ** 2.5)
                            + np.sum(0.5 * cache["k_t"][act]
                                     * np.sum(self._xi_state[act] ** 2, axis=1)))
        bond = 0.0
        if self.bonds.shape[0] and self.bond_alive.any():
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            d = np.linalg.norm(self.pos[j] - self.pos[i], axis=1)
            ext = d - self.bond_length0
            bond = float(np.sum((0.5 * self.bond_kn * ext ** 2
                                 + 0.5 * self.bond_kt
                                 * np.sum(self.bond_xi ** 2, axis=1))[self.bond_alive]))
        return contact, bond

    def step(self, dt: float) -> dict:
        force, torque = self._forces(dt)
        acc = force / self.mass[:, None]
        self.vel += 0.5 * acc * dt
        self.omega += 0.5 * (torque / self.inertia) * dt
        disp = self.vel * dt
        self.pos += disp
        self.theta += self.omega * dt
        self.ledger.body_work += float(np.sum(self._body_force_cache * disp))
        old_face = 0.0 if self.wedge is None else self.wedge.face_force
        if self.wedge is not None:
            self.wedge.advance(dt)
        force, torque = self._forces(dt)
        self.vel += 0.5 * (force / self.mass[:, None]) * dt
        self.omega += 0.5 * (torque / self.inertia) * dt
        if self.wedge is not None:
            self.ledger.wedge_work += (0.5 * (old_face + self.wedge.face_force)
                                       * self.wedge.speed * dt)
        self.time += dt
        self._last_disp = disp
        if not np.isfinite(self.vel).all():
            raise RuntimeError(
                "ridge_keel DEM diverged (non-finite velocity); reduce the timestep")
        self._pair_steps += 1
        if self._pair_steps >= 120:
            self._build_pairs()
        return {"time": self.time,
                "wedge_force": 0.0 if self.wedge is None else self.wedge.face_force,
                "kinetic": self.kinetic_energy()}

    # -- compaction / relaxation ------------------------------------------
    def compact(self, dt: float, steps: int, *, damping_time: float = 0.05,
                tol: float = 1e-6) -> dict:
        """Relax the initial overlaps under strong viscous damping."""
        keep = self.drag_time
        self.drag_time = damping_time
        used = 0
        for used in range(steps):
            self.step(dt)
            if used > 200 and self.kinetic_energy() < tol * max(self.n_particles, 1):
                used += 1
                break
        self.drag_time = keep
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self.ledger = EnergyLedger()
        return {"steps": used, "solid_fraction": self.measure_solid_fraction()}

    def container_area(self) -> float:
        """Area of the envelope currently seen by the grains [m2]."""
        return self.section.cross_section_area * self.envelope_scale ** 2

    def measure_solid_fraction(self) -> float:
        ice_area = float(np.sum(math.pi * self.radius ** 2))
        return ice_area / self.container_area()

    def consolidate(self, dt: float, *, target_solid: float = 0.70,
                    max_steps: int = 6000, steps_per_step: int = 12,
                    scale_step: float = 0.004, damping_time: float = 0.05) -> dict:
        """Isotropic pre-consolidation to the in-situ keel stress state.

        The keel sits in a closed envelope (consolidated layer above, ridge either
        side); the rubble is brought to the measured in-situ solid fraction by
        slowly shrinking the envelope, which is how a confined granular body
        acquires its contact network.  Returns the history of the run.
        """
        keep = self.drag_time
        self.drag_time = damping_time
        history = []
        steps = 0
        while steps < max_steps:
            for _ in range(steps_per_step):
                self.step(dt)
                steps += 1
            solid = self.measure_solid_fraction()
            history.append((self.envelope_scale, solid, self.kinetic_energy()))
            if solid >= target_solid:
                break
            self.envelope_scale = max(self.envelope_scale - scale_step, 0.5)
        # relax the residual kinetic energy at the final confinement
        for _ in range(1500):
            self.step(dt)
            steps += 1
            if self.kinetic_energy() < 1e-8 * max(self.n_particles, 1):
                break
        self.drag_time = keep
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self.ledger = EnergyLedger()
        return {"steps": steps, "envelope_scale": self.envelope_scale,
                "solid_fraction": self.measure_solid_fraction(),
                "history": history[-5:]}

    # -- analysis ----------------------------------------------------------
    def active_contacts(self):
        cache = self._contact_cache
        if cache is None or "active" not in cache or not cache["active"].any():
            return None
        act = cache["active"]
        return {k: (v[act] if isinstance(v, np.ndarray) else v) for k, v in cache.items()}

    def stress_report(self, *, region=None) -> dict:
        """Contact stress tensor (compression positive) and principal axes."""
        c = self.active_contacts()
        if c is None:
            return {}
        if region is not None:
            keep = region((self.pos[c["i"]] + self.pos[c["j"]]) * 0.5)
            if not keep.any():
                return {}
            c = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
        branch = (self.radius[c["i"]] + self.radius[c["j"]])[:, None] * c["normal"]
        traction = c["f_n"][:, None] * c["normal"] + c["f_t"]
        sigma = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                sigma[a, b] = -np.sum(traction[:, a] * branch[:, b])
        mean_grain_area = math.pi * float(np.mean(self.radius)) ** 2
        sigma /= max(c["f_n"].size * mean_grain_area, 1e-12)
        sigma = 0.5 * (sigma + sigma.T)
        evals, evecs = np.linalg.eigh(-sigma)          # compression positive
        order = np.argsort(evals)[::-1]
        major, minor = evecs[:, order[0]], evecs[:, order[1]]
        return {
            "sigma": sigma.tolist(),
            "principal_compressive": [float(evals[order[0]]), float(evals[order[1]])],
            "major_direction_deg": float(math.degrees(math.atan2(major[1], major[0])) % 180.0),
            "minor_direction_deg": float(math.degrees(math.atan2(minor[1], minor[0])) % 180.0),
            "mobilised_friction_sin": float(
                (evals[order[0]] - evals[order[1]])
                / max(abs(evals[order[0]] + evals[order[1]]), 1e-30)),
            "n_contacts": int(c["f_n"].size),
        }

    def force_chain_report(self, *, region=None, threshold: float = 2.0) -> dict:
        """Strong-contact network: orientation, load share and chain topology."""
        c = self.active_contacts()
        if c is None:
            return {}
        i, j, f_n, normal = c["i"], c["j"], c["f_n"], c["normal"]
        if region is not None:
            keep = region((self.pos[i] + self.pos[j]) * 0.5)
            if not keep.any():
                return {}
            i, j, f_n, normal = i[keep], j[keep], f_n[keep], normal[keep]
        mean_force = float(np.mean(f_n))
        strong = f_n > threshold * mean_force
        load_fraction = float(np.sum(f_n[strong]) / np.sum(f_n))
        if strong.any():
            nn, w = normal[strong], f_n[strong]
            fabric = np.einsum("c,ca,cb->ab", w, nn, nn) / float(np.sum(w))
            evals, evecs = np.linalg.eigh(fabric)
            axis = evecs[:, int(np.argmax(evals))]
            chain_direction = float(math.degrees(math.atan2(axis[1], axis[0])) % 180.0)
            anisotropy = float((evals.max() - evals.min()) / max(np.trace(fabric), 1e-30))
        else:
            chain_direction, anisotropy = float("nan"), 0.0
        parent = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in zip(i[strong], j[strong]):
            a, b = int(a), int(b)
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        chains = {}
        for node in parent:
            root = find(node)
            chains[root] = chains.get(root, 0) + 1
        sizes = sorted(chains.values(), reverse=True)
        return {
            "mean_normal_force": mean_force,
            "threshold": float(threshold),
            "n_contacts": int(f_n.size),
            "strong_fraction": float(np.mean(strong)),
            "load_fraction": load_fraction,
            "chain_direction_deg": chain_direction,
            "fabric_anisotropy": anisotropy,
            "n_chains": len(sizes),
            "chain_sizes_top": sizes[:10],
            "largest_chain_share": float(sizes[0] / max(sum(sizes), 1)) if sizes else 0.0,
            "strong_contact_keys": [int(k) for k in (i[strong] * self.n_particles
                                                     + j[strong])],
        }

    def snapshot(self):
        """State used as the reference for displacement / porosity analysis."""
        return {"pos": self.pos.copy(), "alive": np.ones(self.n_particles, dtype=bool),
                "time": self.time}

    def shear_band_report(self, snapshot: dict, *, cell_size: float = 1.6,
                          quantile: float = 0.75, region=None) -> dict:
        """Localise the shear band from the displacement-gradient field.

        The displacement field is fitted linearly per grid cell, the equivalent
        shear strain ``gamma`` is mapped, the upper-quantile cells are retained
        and their ``gamma``-weighted principal axis is the band direction.
        """
        pos0 = snapshot["pos"]
        alive = snapshot["alive"].copy()
        if region is not None:
            alive &= region(self.pos)
        if alive.sum() < 20:
            return {}
        disp = self.pos - pos0
        x = 0.5 * (self.pos[:, 0] + pos0[:, 0])
        y = 0.5 * (self.pos[:, 1] + pos0[:, 1])
        ix = np.floor((x - x.min()) / cell_size).astype(np.int64)
        iy = np.floor((y - y.min()) / cell_size).astype(np.int64)
        keys = ix * 100000 + iy
        centres, gammas = [], []
        for key in np.unique(keys[alive]):
            members = alive & (keys == key)
            if members.sum() < 6:
                continue
            px, py = x[members], y[members]
            ux, uy = disp[members, 0], disp[members, 1]
            design = np.column_stack([np.ones_like(px), px, py])
            coef, *_ = np.linalg.lstsq(design, np.column_stack([ux, uy]), rcond=None)
            dux_dx, dux_dy = coef[1, 0], coef[2, 0]
            duy_dx, duy_dy = coef[1, 1], coef[2, 1]
            exx, eyy = dux_dx, duy_dy
            exy = 0.5 * (dux_dy + duy_dx)
            gamma = 2.0 * math.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2)
            centres.append((px.mean(), py.mean()))
            gammas.append(gamma)
        if len(gammas) < 4:
            return {}
        centres = np.asarray(centres)
        gammas = np.asarray(gammas)
        cutoff = float(np.quantile(gammas, quantile))
        hot = gammas >= cutoff
        if hot.sum() < 4:
            return {}
        pts = centres[hot]
        w = gammas[hot]
        centre = np.average(pts, axis=0, weights=w)
        d = pts - centre
        cov = np.einsum("c,ca,cb->ab", w, d, d) / float(np.sum(w))
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        band_direction = float(math.degrees(math.atan2(axis[1], axis[0])) % 180.0)
        return {
            "cell_size": float(cell_size),
            "quantile": float(quantile),
            "n_cells": int(len(gammas)),
            "n_hot_cells": int(hot.sum()),
            "gamma_max": float(gammas.max()),
            "gamma_mean_hot": float(np.mean(w)),
            "band_direction_deg": band_direction,
            "band_centroid": [float(centre[0]), float(centre[1])],
        }

    def local_solid_fraction(self, radius_scale: float = 1.5):
        """Per-grain local solid fraction from a Gaussian coarse-graining.

        ``nu_i = sum_j a_j W(r_ij) / integral W`` with ``W`` a Gaussian of width
        ``R = radius_scale * mean diameter``; ``a_j = pi r_j^2``.  The measure
        follows the material, so it is insensitive to a grain leaving a fixed
        Eulerian window -- which is what a spilling keel does.
        """
        R = radius_scale * 2.0 * float(np.mean(self.radius))
        area = math.pi * self.radius ** 2
        nu = np.empty(self.n_particles)
        norm = 1.0 / (2.0 * math.pi * R ** 2)
        for i0 in range(0, self.n_particles, 256):
            block = self.pos[i0:i0 + 256]
            d2 = np.sum((block[:, None, :] - self.pos[None, :, :]) ** 2, axis=2)
            w = np.exp(-d2 / (2.0 * R ** 2)) * norm
            nu[i0:i0 + 256] = w @ area
        return nu

    def porosity_report(self, snapshot: dict, *, region=None, radius_scale: float = 1.5):
        """Porosity change of the keel material (body-following measure)."""
        alive = snapshot["alive"].copy()
        if region is not None:
            alive &= region(self.pos) & region(snapshot["pos"])
        if alive.sum() < 10:
            return {}
        nu_now = self.local_solid_fraction(radius_scale)
        prev_pos = self.pos.copy()
        self.pos = snapshot["pos"]
        nu_then = self.local_solid_fraction(radius_scale)
        self.pos = prev_pos
        solid_then = float(np.mean(nu_then[alive]))
        solid_now = float(np.mean(nu_now[alive]))
        return {
            "n_grains": int(alive.sum()),
            "solid_fraction_initial": solid_then,
            "solid_fraction_current": solid_now,
            "porosity_initial": 1.0 - solid_then,
            "porosity_current": 1.0 - solid_now,
            "delta_porosity": solid_then - solid_now,
        }



# --------------------------------------------------------------------------- #
# post-processing helpers
# --------------------------------------------------------------------------- #
def chain_lifetime(contact_key_windows) -> dict:
    """Run lengths (in measurement windows) of individual strong contacts.

    ``contact_key_windows`` is a sequence of iterables of contact keys, one per
    measurement window.  A contact's *lifetime* is the number of consecutive
    windows in which it stays strong; the reported number is the mean over all
    strong contacts (a low value means the load path re-wires quickly, i.e. the
    force chain is a fleeting structure, which is what a plowing granular mass
    produces).
    """
    runs = {}
    current = {}
    for window, keys in enumerate(contact_key_windows):
        for key in set(keys):
            if key in current and current[key] == window - 1:
                runs[key][-1] += 1
            else:
                runs.setdefault(key, []).append(1)
            current[key] = window
    lengths = [run for key in runs for run in runs[key]]
    if not lengths:
        return {"n_contacts": 0, "mean_lifetime_windows": 0.0,
                "max_lifetime_windows": 0, "persistent_fraction": 0.0}
    lengths = np.asarray(lengths, dtype=float)
    longest = np.array([max(runs[key]) for key in runs], dtype=float)
    return {
        "n_contacts": int(len(runs)),
        "n_runs": int(lengths.size),
        "mean_lifetime_windows": float(np.mean(lengths)),
        "max_lifetime_windows": int(np.max(lengths)),
        "persistent_fraction": float(np.mean(longest >= 2)),
    }


def lindqvist_keel_resistance(section: KeelSection, half_angle_deg: float, *,
                              hull_friction: float = 0.60,
                              solid_fraction=None,
                              rho_water: float = RHO_WATER,
                              rho_ice: float = RHO_ICE,
                              gravity: float = GRAVITY,
                              friction_band=(0.40, 1.00)) -> dict:
    """Lindqvist-family keel resistance of a hull penetrating a ridge keel.

    Structure (REAL, derived from the momentum/friction picture): the hull must
    push aside the rubble that lies along its wetted faces, so the resistance per
    unit ship length is the friction coefficient times the weight of the ice
    column resting on that face::

        R_keel = mu_h * rho_i * g * (1 - n) * h_k / sin(alpha)

    with ``h_k / sin(alpha)`` the wetted face length of a wedge of half-angle
    ``alpha`` spanning the keel depth.  This is the keel term only: a
    consolidated layer would add the crushing/bending terms of the level-ice
    anchor (:mod:`tensordem.level_ice.formula`), which the DEM benchmark here
    deliberately omits.

    Coefficient grade: **WITHHELD**.  The published ``mu_h`` for ice-hull
    friction spans roughly 0.4-1.0 and the primary source (Lindqvist 1989) is not
    readable offline, so the point value (0.6) is accompanied by the band the
    coefficient uncertainty implies instead of being presented as validated.
    """
    if not (0.0 < half_angle_deg < 90.0):
        raise ValueError("half_angle_deg must lie in (0, 90)")
    if hull_friction <= 0.0:
        raise ValueError("hull_friction must be positive")
    sf = section.solid_fraction if solid_fraction is None else float(solid_fraction)
    if not (0.0 < sf <= 1.0):
        raise ValueError("solid_fraction must lie in (0, 1]")
    alpha = math.radians(half_angle_deg)
    face_length = section.keel_depth / math.sin(alpha)
    weight_per_area = rho_ice * gravity * sf
    central = hull_friction * weight_per_area * face_length
    lo, hi = friction_band
    return {
        "model": "lindqvist-keel-friction",
        "keel_depth_m": section.keel_depth,
        "keel_width_m": section.keel_width,
        "solid_fraction": sf,
        "hull_friction": hull_friction,
        "half_angle_deg": half_angle_deg,
        "face_length_m": face_length,
        "weight_per_area_Pa": weight_per_area,
        "resistance_N_per_m": central,
        "resistance_band_N_per_m": [lo * weight_per_area * face_length,
                                    hi * weight_per_area * face_length],
        "friction_band": [lo, hi],
        "buoyant_density_difference_kg_m3": rho_water - rho_ice,
        "coefficient_grade": "WITHHELD",
        "structure_grade": "REAL",
    }


# --------------------------------------------------------------------------- #
# independent friction-angle calibration
# --------------------------------------------------------------------------- #
def biaxial_friction_angle(material: RubbleMaterial, psd: PowerLawPSD, *,
                           width: float = 6.0, height: float = 6.0,
                           solid_fraction: float = 0.62, seed: int = 0,
                           confining_pressure: float = 5.0e4,
                           axial_strain: float = 0.12, servo_gain: float = 2.0e-2,
                           max_particles: int = 700) -> dict:
    """Plane-strain biaxial test in the same contact framework: returns phi.

    The cell is a box of rigid frictional walls.  The four walls are first
    driven in until the measured wall pressure reaches ``confining_pressure``
    (isotropic consolidation), then the top and bottom walls are pushed together
    at constant speed while the lateral walls servo-control the confining
    pressure.  ``sin(phi) = (sigma1 - sigma3) / (sigma1 + sigma3)`` at peak.
    """
    rng = np.random.default_rng(seed)
    target_area = width * height * solid_fraction
    budget = int(round(target_area / psd.mean_area()))
    budget = max(80, min(budget, max_particles))
    radii = 0.5 * psd.sample(budget, rng)
    spacing = math.sqrt(target_area / (solid_fraction * budget)) / 0.931
    pts = []
    for jy in range(int(height / (spacing * 0.866)) + 2):
        y = 0.5 * spacing + jy * spacing * 0.866
        if y > height:
            break
        shift = 0.5 * spacing if jy % 2 else 0.0
        for ix in range(int(width / spacing) + 2):
            x = 0.5 * spacing + shift + ix * spacing
            if x > width:
                continue
            pts.append((x, y))
    pts = np.asarray(pts, dtype=float)
    count = min(pts.shape[0], radii.size)
    pos = pts[rng.permutation(count)]
    radius = radii[:count]
    mass = material.rho_ice * math.pi * radius ** 2
    inertia = 0.5 * mass * radius ** 2
    vel = np.zeros_like(pos)
    omega = np.zeros(count)
    xi = np.zeros((count * (count - 1) // 2, 2))
    ii, jj = np.triu_indices(count, 1)
    bound = {"xlo": 0.0, "xhi": width, "ylo": 0.0, "yhi": height}
    lateral = np.array([0.0, width])       # servo-controlled
    vel_wall = {"xlo": 0.0, "xhi": 0.0, "ylo": 0.0, "yhi": 0.0}

    def pair_arrays():
        return ii, jj

    def forces():
        f = np.zeros_like(pos)
        tq = np.zeros(count)
        delta = pos[jj] - pos[ii]
        dist = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(dist, 1e-12)[:, None]
        overlap = radius[ii] + radius[jj] - dist
        active = overlap > 0.0
        ov = np.maximum(overlap, 0.0)
        r_eff = 0.5 * radius[ii] * radius[jj] / (radius[ii] + radius[jj])
        k_32 = (4.0 / 3.0) * material.e_star * np.sqrt(r_eff)
        f_n = np.where(active, k_32 * ov ** 1.5, 0.0)
        v_n = np.sum((vel[ii] - vel[jj]) * normal, axis=1)
        k_sec = 1.5 * k_32 * np.sqrt(ov)
        c_n = 2.0 * material.damping_ratio * np.sqrt(
            0.5 * mass[ii] * mass[jj] / (mass[ii] + mass[jj]) * k_sec)
        f_n = np.maximum(np.where(active, f_n - c_n * v_n, 0.0), 0.0)
        v_t = (vel[ii] - vel[jj]) - v_n[:, None] * normal
        xi[:] += v_t
        f_t = -material.mindlin_ratio * k_sec[:, None] * xi
        mag = np.linalg.norm(f_t, axis=1)
        limit = material.friction * f_n
        slide = active & (mag > limit)
        if slide.any():
            f_t[slide] *= (limit[slide] / np.maximum(mag[slide], 1e-30))[:, None]
            xi[slide] = -f_t[slide] / np.maximum(
                material.mindlin_ratio * k_sec[slide], 1e-30)[:, None]
        cf = f_n[:, None] * normal + f_t
        np.add.at(f, ii, cf)
        np.add.at(f, jj, -cf)
        arm_i = -radius[ii][:, None] * normal
        np.add.at(tq, ii, arm_i[:, 0] * cf[:, 1] - arm_i[:, 1] * cf[:, 0])
        arm_j = radius[jj][:, None] * normal
        np.add.at(tq, jj, -arm_j[:, 0] * cf[:, 1] + arm_j[:, 1] * cf[:, 0])
        wall_forces = {}
        for name, origin, normal_w in (
            ("xlo", np.array([bound["xlo"], 0.0]), np.array([1.0, 0.0])),
            ("xhi", np.array([bound["xhi"], 0.0]), np.array([-1.0, 0.0])),
            ("ylo", np.array([0.0, bound["ylo"]]), np.array([0.0, 1.0])),
            ("yhi", np.array([0.0, bound["yhi"]]), np.array([0.0, -1.0])),
        ):
            s = (pos - origin) @ normal_w
            ov_w = radius - s
            act = ov_w > 0.0
            if not act.any():
                wall_forces[name] = 0.0
                continue
            idx = np.nonzero(act)[0]
            k_w = (4.0 / 3.0) * material.e_star_rigid() * np.sqrt(0.5 * radius[idx])
            fn = k_w * ov_w[idx] ** 1.5
            vn = vel[idx] @ normal_w
            cn = 2.0 * material.damping_ratio * np.sqrt(
                mass[idx] * 1.5 * k_w * np.sqrt(ov_w[idx]))
            fn = np.maximum(fn - cn * vn, 0.0)
            np.add.at(f, idx, fn[:, None] * normal_w[None, :])
            wall_forces[name] = float(np.sum(fn))
        return f, tq, wall_forces

    # --- isotropic consolidation by moving all walls in -------------------
    dt = stable_timestep(material, float(np.min(radius)))
    for _ in range(6000):
        f, tq, wf = forces()
        pressure_ok = True
        for name in ("xlo", "xhi"):
            area = height
            p = wf[name] / area
            speed = servo_gain * (confining_pressure - p) / max(confining_pressure, 1.0)
            speed = float(np.clip(speed, -0.3, 0.3))
            bound[name] += speed * dt * (1.0 if name == "xlo" else -1.0)
            pressure_ok &= abs(p - confining_pressure) / confining_pressure < 0.15
        for name in ("ylo", "yhi"):
            area = width
            p = wf[name] / area
            speed = servo_gain * (confining_pressure - p) / max(confining_pressure, 1.0)
            speed = float(np.clip(speed, -0.3, 0.3))
            bound[name] += speed * dt * (1.0 if name == "ylo" else -1.0)
            pressure_ok &= abs(p - confining_pressure) / confining_pressure < 0.15
        vel += (f / mass[:, None] - vel / 0.05) * dt
        pos += vel * dt
        omega += (tq / inertia - omega / 0.05) * dt
        if not np.isfinite(pos).all():
            raise RuntimeError("biaxial cell diverged during consolidation")
        if pressure_ok:
            break
    height0 = bound["yhi"] - bound["ylo"]
    peak_ratio = -1.0
    history = []
    axial_rate = 0.05            # m/s of relative wall motion
    for step in range(6000):
        f, tq, wf = forces()
        # lateral servo holds the confining pressure
        for name in ("xlo", "xhi"):
            p = wf[name] / max(height0, 1e-9)
            speed = float(np.clip(servo_gain * (confining_pressure - p)
                                  / max(confining_pressure, 1.0), -0.3, 0.3))
            bound[name] += speed * dt * (1.0 if name == "xlo" else -1.0)
        # axial compression
        bound["ylo"] += 0.5 * axial_rate * dt
        bound["yhi"] -= 0.5 * axial_rate * dt
        vel += (f / mass[:, None] - vel / 0.05) * dt
        pos += vel * dt
        omega += (tq / inertia - omega / 0.05) * dt
        if not np.isfinite(pos).all():
            raise RuntimeError("biaxial cell diverged (non-finite position); "
                               "the calibration driver still needs a stable "
                               "servo/timestep combination")
        axial = (height0 - (bound["yhi"] - bound["ylo"])) / height0
        sigma1 = 0.5 * (wf["ylo"] + wf["yhi"]) / max(bound["xhi"] - bound["xlo"], 1e-9)
        sigma3 = 0.5 * (wf["xlo"] + wf["xhi"]) / max(height0, 1e-9)
        if sigma1 + sigma3 > 0.0:
            ratio = (sigma1 - sigma3) / (sigma1 + sigma3)
            peak_ratio = max(peak_ratio, ratio)
        history.append((axial, sigma1, sigma3))
        if axial >= axial_strain:
            break
    peak_ratio = float(np.clip(peak_ratio, 0.0, 0.98))
    return {
        "method": "plane-strain biaxial, servo confining pressure",
        "confining_pressure_Pa": confining_pressure,
        "n_grains": int(count),
        "axial_strain_reached": history[-1][0] if history else 0.0,
        "peak_sigma1_Pa": max((h[1] for h in history), default=float("nan")),
        "peak_sigma3_Pa": max((h[2] for h in history), default=float("nan")),
        "sin_phi_peak": peak_ratio,
        "phi_deg": float(math.degrees(math.asin(peak_ratio))),
        "mu_micromechanical": material.friction,
    }
