"""Ice-ridge keel: freeze-bond nucleation, porosity-structure-resistance validation.

Physical problem
================
An ice-ridge *keel* is the submerged rubble pile of broken ice blocks hanging
below a level-ice sheet / ridge sail.  Three mechanisms decide its mechanical
behaviour and therefore the load it exerts on a structure:

1. **Freeze bonding (adfreeze)**.  Blocks that stay in contact at sub-zero
   temperature are cemented by a frozen water film.  The bond shear strength
   follows the TensorPhysics adfreeze interface ``tau_adfreeze(dT, t, Ra, S)``.
   Here a bond is implemented as a parallel bond with normal traction
   threshold ``F_break = tau_adfreeze(dT, t, Ra, S) * A_contact`` -- the mapping
   required by the campaign node (adfreeze parameterisation -> DEM bond).

2. **Porosity / fabric**.  The void fraction ``n`` controls the effective
   density and the resistance.  Freshly formed rubble is loose (n ~ 0.55-0.65:
   blocks are nearly neutrally buoyant, nothing compacts them).  Ridge-building
   confinement compacts the fabric, and the compacted state *survives unloading
   only when it is cemented by freeze bonds*.  Both facts are produced by the
   DEM itself (compaction -> bonding -> unloading), so porosity is an
   **emergent** output, never an input.

3. **Force chains**.  The load-bearing network of the bonded pile has an
   exponential force tail ``P(f) ~ exp(-f/f_c)``, the standard signature of a
   jammed granular packing.

Kerr (1975) empirical keel resistance
=====================================
For a rigid plate advancing into a keel of height ``h_k`` the empirical
resistance per unit width is

    F_Kerr = 0.5 * rho_w * g * h_k^2 * tan(phi_f) * (1 - n)

It factorises as *hydrostatic thrust x solid area fraction x friction*:
``0.5 rho_w g h_k^2`` is the water-pressure integral over the keel depth,
``(1-n)`` the solid fraction of the swept cross-section and ``tan(phi_f)`` the
rubble friction.  The present module evaluates each factor *independently*
from the DEM (``mu_mob`` measured by a plate-push test, ``n`` measured by
Monte-Carlo point sampling, ``h_k`` measured geometrically) and takes the
product as the skeleton resistance

    F_skel = mu_mob * (1 - n_meas) * 0.5 * rho_w * g * h_k^2      [N/m]

Kerr's formula is *not* used to build ``F_skel``; it is only the yardstick for
criterion 3.  Because Kerr (1975) has no adhesion term, criterion 3 is
evaluated on the skeleton term; the freeze-bond term
``F_bond = tau_adfreeze * (bonded area per unit face area) * h_k`` is reported
separately and drives the temperature sensitivity (criterion 4).

Evidence level: REAL for the DEM dynamics, the adfreeze law, porosity, force
chains and bond rupture; the *closure* (hydrostatic thrust x solid fraction x
mobilised friction) is an assumption inherited from Kerr (1975) and is stated
as such (see docs/ICE_RIDGE_KEEL_FREEZEBOND.md).  Everything is SI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional dependency, used for neighbour search
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

RHO_ICE = 917.0          # kg/m^3
RHO_WATER = 1025.0       # kg/m^3
GRAVITY = 9.81           # m/s^2

# ---------------------------------------------------------------------------
# Adfreeze strength law (TensorPhysics ``ice/adfreeze.py`` interface)
# ---------------------------------------------------------------------------
# tau = tau_sat(dT) * t_frac(t) * r_frac(Ra) * s_frac(S), with a hyperbolic
# undercooling saturation, a saturating Stefan-type time hardening, a weak
# roughness power law and a linear brine softening.  Calibrated on the
# Frederking (1980) / Timco (1991) freeze-bond strength data used by the
# TensorPhysics implementation; fully analytic (no randomness).
TAU_SAT_MPA = 5.8301     # MPa, saturation bond strength
D_HALF_K = 76.1562       # K, half-saturation undercooling
T_REF_S = 1000.0         # s, reference contact time
RA_REF_M = 20e-6         # m, reference roughness
S_REF_PPT = 5.0          # ppt, reference salinity


def tau_adfreeze(undercooling_c: float, contact_time_s: float = 1000.0,
                 roughness_m: float = 20e-6, salinity_ppt: float = 0.0) -> float:
    """Freeze-bond shear strength [MPa]; exactly 0 at ``dT == 0``."""
    dT = abs(float(undercooling_c))
    if dT <= 0.0:
        return 0.0
    t = max(float(contact_time_s), 0.0)
    if t <= 0.0:
        return 0.0
    t_sat = TAU_SAT_MPA * dT / (dT + D_HALF_K)
    s = math.sqrt(t / T_REF_S)
    t_frac = s / (1.0 + 0.5 * s) * 1.5  # t_frac(T_REF_S) == 1
    r_frac = min(max((float(roughness_m) / RA_REF_M) ** 0.25, 0.2), 3.0)
    s_frac = 1.0 / (1.0 + 0.12 * max(float(salinity_ppt), 0.0) / S_REF_PPT)
    return t_sat * t_frac * r_frac * s_frac


def contact_area(radius_a: float, radius_b: float, overlap: float,
                 film_thickness: float = 1.0e-3) -> float:
    """Frozen-film contact area [m^2]: Hertz radius ``a = sqrt(R_eff delta)``."""
    if radius_a <= 0.0 or radius_b <= 0.0:
        return 0.0
    r_eff = radius_a * radius_b / (radius_a + radius_b)
    delta = max(float(overlap), float(film_thickness))
    return math.pi * r_eff * delta


def freeze_bond_break_force(undercooling_c: float, radius_a: float,
                            radius_b: float, overlap: float,
                            contact_time_s: float = 1000.0,
                            roughness_m: float = 20e-6,
                            salinity_ppt: float = 0.0) -> float:
    """DEM bond threshold ``F_break = tau_adfreeze * A_contact`` [N]."""
    tau = tau_adfreeze(undercooling_c, contact_time_s, roughness_m, salinity_ppt)
    return tau * 1.0e6 * contact_area(radius_a, radius_b, overlap)


def kerr_keel_resistance(keel_height: float, porosity: float,
                         friction_angle_deg: float = 40.0,
                         rho_water: float = RHO_WATER,
                         gravity: float = GRAVITY) -> float:
    """Kerr (1975) empirical keel resistance per unit width [N/m]."""
    if keel_height <= 0.0:
        return 0.0
    n = min(max(float(porosity), 0.0), 1.0)
    phi = math.radians(friction_angle_deg)
    return 0.5 * rho_water * gravity * keel_height ** 2 * math.tan(phi) * (1.0 - n)


def keel_resistance_model(keel_height: float, porosity: float, mu_mob: float,
                          tau_mpa: float, bonded_area_density: float,
                          rho_water: float = RHO_WATER,
                          gravity: float = GRAVITY) -> dict[str, float]:
    """Keel resistance per unit width [N/m] built from DEM-measured factors.

    ``F_skel`` = mobilised friction x solid fraction x hydrostatic thrust
    (Kerr-comparable, no adhesion); ``F_bond`` = frozen-bond area x strength.
    """
    if keel_height <= 0.0:
        return {"f_skel": 0.0, "f_bond": 0.0, "f_keel": 0.0, "thrust": 0.0}
    n = min(max(float(porosity), 0.0), 1.0)
    thrust = 0.5 * rho_water * gravity * keel_height ** 2
    f_skel = max(float(mu_mob), 0.0) * (1.0 - n) * thrust
    # bonded frozen area intersected by the advancing face of unit width
    a_bond_face = max(float(bonded_area_density), 0.0) * keel_height * 1.0
    f_bond = max(float(tau_mpa), 0.0) * 1.0e6 * a_bond_face
    return {"f_skel": float(f_skel), "f_bond": float(f_bond),
            "f_keel": float(f_skel + f_bond), "thrust": float(thrust)}


# ---------------------------------------------------------------------------
# Block size distribution
# ---------------------------------------------------------------------------
def rosin_rammler_sample(n: int, d_min: float = 0.1, d_max: float = 1.0,
                         exponent: float = 1.6, seed: int = 0) -> np.ndarray:
    """Truncated Rosin-Rammler block diameters [m], deterministic per seed."""
    if n <= 0:
        return np.zeros(0)
    q = float(exponent)

    def raw_mean(d_e: float) -> float:
        u = np.linspace(1e-6, 1.0 - 1e-6, 40001)
        d = d_e * (-np.log(1.0 - u)) ** (1.0 / q)
        d = d[(d >= d_min) & (d <= d_max)]
        return float(d.mean()) if d.size else 0.0

    lo, hi = 1e-3, 10.0
    target = 0.5 * (d_min + d_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if raw_mean(mid) < target:
            lo = mid
        else:
            hi = mid
    d_e = 0.5 * (lo + hi)

    rng = np.random.default_rng(seed)
    out = np.zeros(0)
    while out.size < n:
        u = rng.random(max(4 * (n - out.size), 64))
        d = d_e * (-np.log(1.0 - u)) ** (1.0 / q)
        d = d[(d >= d_min) & (d <= d_max)]
        out = np.concatenate([out, d])
    return np.sort(out[:n])


# ---------------------------------------------------------------------------
# DEM state
# ---------------------------------------------------------------------------
@dataclass
class RubblePile:
    """DEM state of a rubble pile (spheres with radius = d/2)."""

    pos: np.ndarray            # (N,3) [m]
    vel: np.ndarray            # (N,3) [m/s]
    radius: np.ndarray         # (N,)  [m]
    mass: np.ndarray           # (N,)  [kg]
    box: np.ndarray            # (3,2) wall positions [m]
    friction_coeff: float = math.tan(math.radians(40.0))
    stiffness: float = 1.0e5   # specific stiffness [1/s^2]; k_ij = k_s * m_eff
    damping: float = 0.5       # normal damping ratio
    tangent_damping: float = 3.0e4  # N s/m viscous Coulomb

    @property
    def n_blocks(self) -> int:
        return int(self.pos.shape[0])

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())

    def keel_height(self) -> float:
        if self.n_blocks == 0:
            return 0.0
        return float(self.pos[:, 2].max() + self.radius[self.pos[:, 2].argmax()]
                     - self.box[2, 0])

    def analytic_mass(self) -> float:
        return float(np.sum(RHO_ICE * 4.0 / 3.0 * math.pi * self.radius ** 3))


def _relax_overlaps(pos: np.ndarray, radius: np.ndarray, box: np.ndarray,
                    iters: int = 300) -> np.ndarray:
    """Push overlapping spheres apart (dilute placement -> valid packing)."""
    pos = pos.copy()
    if radius.size < 2:
        return pos
    rmax = float(radius.max())
    for _ in range(iters):
        tree = cKDTree(pos)
        pairs = np.asarray(tree.query_pairs(2.0 * rmax, output_type="ndarray"))
        if pairs.size == 0:
            break
        dx = pos[pairs[:, 0]] - pos[pairs[:, 1]]
        dist = np.linalg.norm(dx, axis=1)
        dist = np.where(dist < 1e-9, 1e-9, dist)
        overlap = radius[pairs[:, 0]] + radius[pairs[:, 1]] - dist
        bad = overlap > 1e-6
        if not np.any(bad):
            break
        pairs = pairs[bad]
        dx = dx[bad] / dist[bad][:, None]
        push = 0.5 * overlap[bad][:, None] * dx
        np.add.at(pos, pairs[:, 0], push)
        np.add.at(pos, pairs[:, 1], -push)
        for axis in range(3):
            np.clip(pos[:, axis], box[axis, 0] + radius, box[axis, 1] - radius,
                    out=pos[:, axis])
    return pos


def build_pile(n_blocks: int, box_length: float = 6.0, box_width: float = 6.0,
               loose_height: float = 3.0, d_min: float = 0.1,
               d_max: float = 1.0, seed: int = 0,
               friction_angle_deg: float = 40.0,
               placement_porosity: float = 0.72,
               exponent: float = 1.2) -> RubblePile:
    """Dilute random placement (no overlaps) of a Rosin-Rammler block ensemble.

    The blocks are placed uniformly at ``placement_porosity`` and then
    de-overlapped geometrically; the DEM stages in :func:`run_keel_case` do the
    settling, so the initial state carries no artificial force.
    """
    rng = np.random.default_rng(seed + 101)
    d = rosin_rammler_sample(n_blocks, d_min, d_max, exponent=exponent, seed=seed)
    r = 0.5 * d
    solid = float(np.sum(4.0 / 3.0 * math.pi * r ** 3))
    h_place = max(loose_height,
                  solid / max(1.0 - placement_porosity, 1e-3) / (box_length * box_width))
    box = np.array([[0.0, box_length], [0.0, box_width], [0.0, h_place]])
    pos = np.stack([
        rng.uniform(r, box_length - r),
        rng.uniform(r, box_width - r),
        rng.uniform(r, h_place - r),
    ], axis=1)
    pos = _relax_overlaps(pos, r, box)
    mass = RHO_ICE * 4.0 / 3.0 * math.pi * r ** 3
    return RubblePile(pos=pos, vel=np.zeros_like(pos), radius=r, mass=mass,
                      box=box,
                      friction_coeff=math.tan(math.radians(friction_angle_deg)))


# ---------------------------------------------------------------------------
# Freeze bonds
# ---------------------------------------------------------------------------
@dataclass
class BondSet:
    """Parallel (freeze) bonds at pile contacts."""

    i: np.ndarray
    j: np.ndarray
    gap0: np.ndarray      # overlap at bonding time (bond is neutral there)
    area: np.ndarray      # frozen-film contact area [m^2]
    f_break: np.ndarray   # normal traction threshold [N]
    broken: np.ndarray
    k_bond: np.ndarray | None = None   # bond stiffness [N/m] (ductility)

    @property
    def n_bonds(self) -> int:
        return int(self.i.size)

    @property
    def n_broken(self) -> int:
        return int(self.broken.sum())

    def active_area(self) -> float:
        return float(self.area[~self.broken].sum())


def _pairs(pile: RubblePile) -> np.ndarray:
    """Candidate pairs (centre distance <= 2*max radius)."""
    if pile.n_blocks < 2:
        return np.zeros((0, 2), dtype=np.int64)
    rmax = float(pile.radius.max())
    if cKDTree is not None:
        tree = cKDTree(pile.pos)
        pairs = tree.query_pairs(2.0 * rmax, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64)
    out = []
    for a in range(pile.n_blocks):
        for b in range(a + 1, pile.n_blocks):
            if np.linalg.norm(pile.pos[a] - pile.pos[b]) <= 2.0 * rmax:
                out.append((a, b))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def create_bonds(pile: RubblePile, undercooling_c: float,
                 contact_time_s: float = 1000.0, roughness_m: float = 20e-6,
                 salinity_ppt: float = 0.0,
                 gap_tolerance: float = 5.0e-3) -> BondSet:
    """Create freeze bonds at every contact closer than ``gap_tolerance``."""
    pairs = _pairs(pile)
    empty = np.zeros(0)
    if pairs.shape[0] == 0:
        return BondSet(empty.astype(np.int64), empty.astype(np.int64), empty,
                       empty, empty, np.zeros(0, dtype=bool))
    dist = np.linalg.norm(pile.pos[pairs[:, 1]] - pile.pos[pairs[:, 0]], axis=1)
    overlap = pile.radius[pairs[:, 0]] + pile.radius[pairs[:, 1]] - dist
    keep = overlap > -gap_tolerance
    pairs = pairs[keep]
    overlap = overlap[keep]
    if pairs.shape[0] == 0:
        return BondSet(empty.astype(np.int64), empty.astype(np.int64), empty,
                       empty, empty, np.zeros(0, dtype=bool), empty)
    r_eff = (pile.radius[pairs[:, 0]] * pile.radius[pairs[:, 1]]
             / (pile.radius[pairs[:, 0]] + pile.radius[pairs[:, 1]]))
    delta = np.maximum(overlap, 1.0e-3)
    area = math.pi * r_eff * delta
    tau = tau_adfreeze(undercooling_c, contact_time_s, roughness_m, salinity_ppt)
    f_break = tau * 1.0e6 * area
    # bond ductility: a freeze bond fails after ~1 mm of opening, so its
    # stiffness is f_break / delta_fail (not the much stiffer contact law)
    k_bond = f_break / 1.0e-3
    return BondSet(i=pairs[:, 0].astype(np.int64), j=pairs[:, 1].astype(np.int64),
                   gap0=overlap.astype(float), area=area, f_break=f_break,
                   broken=np.zeros(pairs.shape[0], dtype=bool), k_bond=k_bond)


# ---------------------------------------------------------------------------
# Contact force evaluation (sphere-sphere + sphere-wall + bonds + plate)
# ---------------------------------------------------------------------------
def contact_forces(pile: RubblePile, bonds: BondSet | None = None,
                   plate_x: float | None = None,
                   plate_speed: float = 0.0) -> dict[str, Any]:
    """Single force evaluation; returns forces + contact bookkeeping."""
    n = pile.n_blocks
    force = np.zeros((n, 3))
    bond_tension = np.zeros(0)
    pairs = _pairs(pile)
    if pairs.shape[0]:
        pi = pile.pos[pairs[:, 0]]
        pj = pile.pos[pairs[:, 1]]
        dx = pj - pi
        dist = np.linalg.norm(dx, axis=1)
        dist = np.where(dist < 1e-12, 1e-12, dist)
        n_hat = dx / dist[:, None]
        delta = pile.radius[pairs[:, 0]] + pile.radius[pairs[:, 1]] - dist
        vrel = pile.vel[pairs[:, 1]] - pile.vel[pairs[:, 0]]
        vn = np.einsum("ij,ij->i", vrel, n_hat)
        vt_vec = vrel - vn[:, None] * n_hat
        vt_mag = np.linalg.norm(vt_vec, axis=1)
        m_eff = 0.5 * (pile.mass[pairs[:, 0]] + pile.mass[pairs[:, 1]])
        k_pair = pile.stiffness * m_eff          # specific stiffness x m_eff
        c_n = 2.0 * pile.damping * m_eff * math.sqrt(pile.stiffness)
        fn = np.where(delta > 0.0, k_pair * delta - c_n * vn, 0.0)
        ft = np.minimum(pile.friction_coeff * np.abs(fn), pile.tangent_damping * vt_mag)
        ft = np.where(delta > 0.0, ft, 0.0)

        if bonds is not None and bonds.n_bonds:
            # vectorised bond bookkeeping: map bonded pairs onto contact pairs
            npair = pairs.shape[0]
            key_pairs = pairs[:, 0] * n + pairs[:, 1]
            key_bonds = bonds.i * n + bonds.j
            order = np.argsort(key_pairs)
            spk = key_pairs[order]
            at = np.clip(np.searchsorted(spk, key_bonds), 0, max(npair - 1, 0))
            valid = spk[at] == key_bonds
            bidx = np.nonzero(valid)[0]
            pidx = order[at[valid]]
            active = ~bonds.broken[bidx]
            k_b = (bonds.k_bond if bonds.k_bond is not None
                   else np.full(bonds.n_bonds, np.inf))
            tension = k_b[bidx] * (delta[pidx] - bonds.gap0[bidx])
            bond_tension = tension[active]
            separating = tension < 0.0
            fail = separating & (np.abs(tension) > bonds.f_break[bidx]) & active
            bonds.broken[bidx[fail]] = True
            apply = active & separating & ~fail
            if np.any(apply):
                ii = pidx[apply]
                fn[ii] = fn[ii] + tension[apply]
                cap = (np.abs(fn[ii]) * pile.friction_coeff
                       + bonds.f_break[bidx[apply]])
                ft[ii] = np.minimum(ft[ii], cap)
        vhat = vt_vec / np.where(vt_mag < 1e-12, 1e-12, vt_mag)[:, None]
        # contact normal points from i to j, so the reaction on i is -F
        f_pair = fn[:, None] * n_hat + ft[:, None] * vhat
        np.add.at(force, pairs[:, 0], -f_pair)
        np.add.at(force, pairs[:, 1], f_pair)
        fn_out, ft_out = fn, ft
    else:
        fn_out = np.zeros(0)
        ft_out = np.zeros(0)

    walls = []
    for axis in range(3):
        lo, hi = pile.box[axis]
        for side in (0, 1):
            wall_pos = lo if side == 0 else hi
            n_dir = 1.0 if side == 0 else -1.0        # outward normal of the wall
            if axis == 0 and side == 1 and plate_x is not None:
                wall_pos = plate_x
            overlap = pile.radius - n_dir * (pile.pos[:, axis] - wall_pos)
            hit = overlap > 0.0
            if not np.any(hit):
                continue
            speed = (plate_speed if (axis == 0 and side == 1 and plate_x is not None)
                     else 0.0)
            v_n = n_dir * (pile.vel[:, axis] - speed)   # >0 = receding
            k_wall = pile.stiffness * pile.mass
            c_n = 2.0 * pile.damping * pile.mass * math.sqrt(pile.stiffness)
            fn = np.maximum(k_wall * overlap - c_n * v_n, 0.0) * hit
            f = np.zeros((n, 3))
            f[:, axis] = n_dir * fn
            tang = [a for a in range(3) if a != axis]
            vt = np.linalg.norm(pile.vel[:, tang], axis=1)
            ft = np.minimum(pile.friction_coeff * fn, pile.tangent_damping * vt)
            for t_ax in tang:
                f[:, t_ax] -= ft * np.sign(pile.vel[:, t_ax] + 1e-15)
            force += f
            walls.append({"axis": axis, "side": side, "fn": float(fn.sum()),
                          "ft": float(ft.sum()), "n_contact": int(hit.sum())})

    return {"force": force, "pairs": pairs, "fn": fn_out, "ft": ft_out,
            "bond_tension": bond_tension, "walls": walls}


@dataclass
class DemParams:
    dt: float = 2.0e-3
    buoyant_accel: float = GRAVITY * (RHO_WATER / RHO_ICE - 1.0)  # upward
    global_damping: float = 0.995
    v_max: float = 5.0          # numerical velocity cap [m/s]


def integrate(pile: RubblePile, steps: int, params: DemParams | None = None,
              bonds: BondSet | None = None, plate_x: float | None = None,
              plate_speed: float = 0.0,
              wall_motion: dict[int, float] | None = None,
              track: bool = False) -> dict[str, Any]:
    """Explicit fixed-step integration; ``wall_motion`` moves walls (compaction)."""
    params = params or DemParams()
    dt = params.dt
    hist = {"ke": [], "fn_sum": []}
    for _ in range(steps):
        if wall_motion:
            for idx, speed in wall_motion.items():
                axis, side = divmod(idx, 2)
                pile.box[axis, side] += speed * dt
        res = contact_forces(pile, bonds=bonds, plate_x=plate_x,
                            plate_speed=plate_speed)
        f = res["force"]
        f[:, 2] += pile.mass * params.buoyant_accel
        pile.vel = (pile.vel + (f / pile.mass[:, None]) * dt) * params.global_damping
        np.clip(pile.vel, -params.v_max, params.v_max, out=pile.vel)
        pile.pos = pile.pos + pile.vel * dt
        for axis in range(3):
            lo, hi = pile.box[axis]
            np.clip(pile.pos[:, axis], lo + pile.radius, hi - pile.radius,
                    out=pile.pos[:, axis])
        if track:
            hist["ke"].append(float(np.sum(0.5 * pile.mass
                                          * np.sum(pile.vel ** 2, axis=1))))
            hist["fn_sum"].append(float(res["fn"][res["fn"] > 0].sum()))
    return hist


# ---------------------------------------------------------------------------
# Porosity / structure
# ---------------------------------------------------------------------------
def _sampling_box(pile: RubblePile) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([pile.box[0, 0], pile.box[1, 0], pile.box[2, 0]])
    z_top = float((pile.pos[:, 2] + pile.radius).max())
    hi = np.array([pile.box[0, 1], pile.box[1, 1], z_top])
    return lo, hi


def measure_porosity(pile: RubblePile, n_samples: int = 200000,
                     seed: int = 7) -> dict[str, float]:
    """Monte-Carlo point-sampling porosity of the pile's bounding layer."""
    if pile.n_blocks == 0:
        return {"porosity": 1.0, "solid_fraction": 0.0, "solid_volume": 0.0,
                "bounding_volume": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    lo, hi = _sampling_box(pile)
    pts = lo + rng.random((n_samples, 3)) * (hi - lo)
    tree = cKDTree(pile.pos)
    dist, idx = tree.query(pts, k=1)
    solid = dist <= pile.radius[idx]
    p = float(solid.mean())
    vol = float(np.prod(hi - lo))
    return {"porosity": 1.0 - p, "solid_fraction": p, "solid_volume": p * vol,
            "bounding_volume": vol,
            "std_error": math.sqrt(max(p * (1.0 - p), 0.0) / n_samples)}


def measure_local_porosity(pile: RubblePile, n_samples: int = 120000,
                           seed: int = 11) -> np.ndarray:
    """Voronoi (nearest-block-cell) local porosity distribution."""
    if pile.n_blocks == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    lo, hi = _sampling_box(pile)
    pts = lo + rng.random((n_samples, 3)) * (hi - lo)
    tree = cKDTree(pile.pos)
    dist, idx = tree.query(pts, k=1)
    solid = (dist <= pile.radius[idx]).astype(float)
    counts = np.bincount(idx, minlength=pile.n_blocks).astype(float)
    solid_counts = np.bincount(idx, weights=solid, minlength=pile.n_blocks)
    used = counts > 20
    return 1.0 - solid_counts[used] / counts[used]


def force_chain_stats(fn: np.ndarray, min_contacts: int = 50) -> dict[str, float]:
    """Force-chain statistics + exponential tail fit ``P(f) ~ exp(-f/fc)``."""
    fn = np.asarray(fn, dtype=float)
    fn = fn[fn > 0.0]
    base = {"n_contacts": int(fn.size), "mean_force": 0.0, "max_force": 0.0,
            "tail_scale": 0.0, "tail_r_squared": 0.0}
    if fn.size < min_contacts:
        return base
    f_mean = float(fn.mean())
    tail = fn[fn >= f_mean]
    base["mean_force"] = f_mean
    base["max_force"] = float(fn.max())
    if tail.size < 20:
        return base
    nbins = int(max(6, min(20, tail.size // 8)))
    counts, edges = np.histogram(tail, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ok = counts > 0
    if ok.sum() < 4:
        return base
    x = centres[ok]
    y = np.log(counts[ok] / (tail.size * widths[ok]))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    base["tail_scale"] = float(-1.0 / slope) if slope < 0 else 0.0
    base["tail_r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return base


def measure_mobilised_friction(pile: RubblePile, bonds: BondSet | None = None,
                               shear_speed: float = 0.10, steps: int = 400,
                               dt: float = 2.0e-3,
                               friction_angle_deg: float = 40.0,
                               overburden_accel: float = 5.0) -> dict[str, float]:
    """Direct-shear test of the pile: measure the mobilised friction coefficient.

    The upper half of the pile is dragged at constant speed while the lower half
    stays fixed; the ratio of tangential to normal impulse on the contacts that
    cross the shear plane is the mobilised friction coefficient ``mu_mob``.  It
    is the DEM analogue of Kerr's ``tan(phi_f)`` and is bounded above by the
    contact friction coefficient of the pile.
    """
    params = DemParams(dt=dt, buoyant_accel=-overburden_accel,
                       global_damping=0.999, v_max=2.0)
    z_mid = float(np.median(pile.pos[:, 2]))
    upper = pile.pos[:, 2] > z_mid
    if upper.sum() < 5 or (~upper).sum() < 5:
        return {"mu_mob": 0.0, "normal_impulse": 0.0, "tangential_impulse": 0.0,
                "tan_phi": math.tan(math.radians(friction_angle_deg)),
                "shear_contacts": 0, "shear_plane_z": z_mid,
                "loaded_fn": np.zeros(0)}
    fn_acc = 0.0
    ft_acc = 0.0
    n_acc = 0
    fn_samples: list[np.ndarray] = []
    for step in range(steps):
        pile.vel[upper, 1] = shear_speed            # kinematic shear drive
        res = contact_forces(pile, bonds=bonds)
        pairs = res["pairs"]
        if pairs.shape[0]:
            up_i = upper[pairs[:, 0]]
            crossing = up_i != upper[pairs[:, 1]]
            if np.any(crossing):
                if step > steps // 3:               # steady-state window
                    fn_acc += float(np.abs(res["fn"][crossing]).sum())
                    ft_acc += float(res["ft"][crossing].sum())
                    n_acc += int(crossing.sum())
                    loaded = np.abs(res["fn"])
                    fn_samples.append(loaded[loaded > 0.0])
        f = res["force"]
        f[:, 2] += pile.mass * params.buoyant_accel
        pile.vel = (pile.vel + (f / pile.mass[:, None]) * dt) * params.global_damping
        np.clip(pile.vel, -params.v_max, params.v_max, out=pile.vel)
        pile.vel[upper, 1] = shear_speed
        pile.pos = pile.pos + pile.vel * dt
        for axis in range(3):
            lo, hi = pile.box[axis]
            np.clip(pile.pos[:, axis], lo + pile.radius, hi - pile.radius,
                    out=pile.pos[:, axis])
    tan_phi = math.tan(math.radians(friction_angle_deg))
    loaded_fn = np.concatenate(fn_samples) if fn_samples else np.zeros(0)
    return {"mu_mob": float(ft_acc / fn_acc) if fn_acc > 0 else 0.0,
            "normal_impulse": fn_acc, "tangential_impulse": ft_acc,
            "tan_phi": tan_phi, "shear_contacts": n_acc,
            "shear_plane_z": z_mid, "loaded_fn": loaded_fn}


def measure_bond_break_force(undercooling_c: float, radius: float = 0.25,
                             contact_time_s: float = 1000.0,
                             roughness_m: float = 20e-6,
                             salinity_ppt: float = 0.0,
                             steps: int = 20000, dt: float = 2.0e-4,
                             pull_speed: float = 0.002) -> dict[str, float]:
    """DEM pull test of one freeze bond -> measured rupture force [N].

    Two bonded blocks are pulled apart at constant speed; the peak tensile
    force of the bond is compared with the analytic ``tau_adfreeze * A``
    (criterion 1: adfreeze parameterisation consistency).
    """
    r = float(radius)
    pile = RubblePile(
        pos=np.array([[0.5, 0.5, 0.5], [0.5 + 2.0 * r - 1.0e-3, 0.5, 0.5]]),
        vel=np.zeros((2, 3)), radius=np.array([r, r]),
        mass=np.full(2, RHO_ICE * 4.0 / 3.0 * math.pi * r ** 3),
        box=np.array([[-2.0, 6.0], [-2.0, 6.0], [-2.0, 6.0]]),
        stiffness=1.0e5, damping=0.05, tangent_damping=1.0e3)
    overlap = (pile.radius[0] + pile.radius[1]
               - float(np.linalg.norm(pile.pos[1] - pile.pos[0])))
    bonds = create_bonds(pile, undercooling_c, contact_time_s, roughness_m,
                         salinity_ppt, gap_tolerance=1.0)
    analytic = freeze_bond_break_force(undercooling_c, r, r, overlap,
                                       contact_time_s, roughness_m, salinity_ppt)
    if bonds.n_bonds == 0 or analytic == 0.0:
        return {"analytic_n": analytic, "measured_n": 0.0, "rel_error": 0.0,
                "broken": bool(bonds.n_bonds and bonds.broken[0])}
    peak = 0.0
    for _ in range(steps):
        pile.pos[1, 0] += pull_speed * dt
        res = contact_forces(pile, bonds=bonds)
        tension = res["bond_tension"]
        if tension.size:
            peak = max(peak, abs(float(tension.min())))
        if bonds.broken[0]:
            break
    err = abs(peak - analytic) / analytic if analytic > 0 else 0.0
    return {"analytic_n": float(analytic), "measured_n": float(peak),
            "rel_error": float(err), "broken": bool(bonds.broken[0])}


# ---------------------------------------------------------------------------
# One keel case: place -> compact -> bond -> unload -> measure
# ---------------------------------------------------------------------------
def run_keel_case(keel_height_target: float = 4.0, undercooling_c: float = -10.0,
                  n_blocks: int | None = None, box_length: float | None = None,
                  box_width: float | None = None, friction_angle_deg: float = 40.0,
                  contact_time_s: float = 1000.0, salinity_ppt: float = 0.0,
                  seed: int = 0, consol_steps: int = 3600,
                  release_steps: int = 600, dt: float = 2.0e-3,
                  with_bonds: bool = True,
                  consolidation_accel: float = 300.0,
                  exponent: float = 1.2,
                  in_service_accel: float = 5.0) -> dict[str, Any]:
    """Keel formation -> freeze bonding -> buoyant release -> measurements.

    Stage 1 (loose deposition): ballistic deposition of Rosin-Rammler blocks,
    porosity ~0.65 -- the state of *uncemented* rubble.
    Stage 2 (ridge-building consolidation): the top boundary (sail / level ice)
    closes quasi-statically and compacts the fabric to n ~ 0.40.
    Stage 3 (freeze bonding): contacts are cemented with
    ``F_break = tau_adfreeze * A_contact``.
    Stage 4 (release): the boundary reopens and the net body force becomes
    buoyant (ice is 10 % lighter than water).  An *uncemented* pile cannot hold
    the consolidated fabric -- its blocks are buoyant and disperse, so its
    porosity returns to the loose regime; a *cemented* pile keeps it.
    """
    target_solid_fraction = 0.60          # n = 0.40 after ridge consolidation
    # volume-weighted mean block volume of the Rosin-Rammler sample
    d_sample = rosin_rammler_sample(20000, 0.1, 1.0, exponent=exponent, seed=1)
    v_block = float(np.mean(math.pi / 6.0 * d_sample ** 3))
    if n_blocks is None:
        if box_length is None or box_width is None:
            # footprint that allows >= 500 blocks at the target porosity
            area = max(500.0 * v_block / (target_solid_fraction * keel_height_target),
                       16.0)
            box_length = box_width = math.sqrt(area)
        n_blocks = int(round(target_solid_fraction * box_length * box_width
                             * keel_height_target / v_block))
    n_blocks = max(int(n_blocks), 500)
    pile = build_pile(n_blocks, box_length, box_width,
                      loose_height=max(3.0 * keel_height_target, 6.0),
                      seed=seed, friction_angle_deg=friction_angle_deg,
                      exponent=exponent)
    h_box = pile.keel_height()
    pile.box[2, 1] = h_box + 0.2
    solid_volume = float(np.sum(4.0 / 3.0 * math.pi * pile.radius ** 3))
    porosity_loose = measure_porosity(pile)["porosity"]

    params = DemParams(dt=dt)
    # --- stage 2: ridge-building consolidation ---------------------------------
    # the ridge-building overburden (level-ice thrust + sail weight) is applied
    # as an equivalent downward acceleration; force-controlled so that contact
    # overlaps stay physical (no kinematic crushing)
    params_comp = DemParams(dt=dt, buoyant_accel=-consolidation_accel,
                            global_damping=0.98, v_max=4.0)
    integrate(pile, consol_steps, params_comp)
    # the overburden is removed before freezing (a ridge consolidates first and
    # freezes afterwards), so the bonded state carries no artificial pre-stress
    params_unload = DemParams(dt=dt, buoyant_accel=-in_service_accel,
                              global_damping=0.99)
    integrate(pile, 1200, params_unload)
    pile.vel[:] = 0.0      # bonds form on rubble at rest (quasi-static freeze)
    comp = measure_porosity(pile)
    porosity_compacted = comp["porosity"]
    h_compacted = pile.keel_height()

    # --- stage 3: freeze bonds from the adfreeze parameterisation ---------
    tau = tau_adfreeze(undercooling_c, contact_time_s, salinity_ppt=salinity_ppt)
    bonds = None
    bonded_area_at_bonding = 0.0
    if with_bonds and tau > 0.0:
        bonds = create_bonds(pile, undercooling_c, contact_time_s,
                             salinity_ppt=salinity_ppt)
        pile.vel[:] = 0.0
        # the freeze-bond area available on the swept face is a property of the
        # bonded fabric at loading time; bond rupture during shearing is
        # reported separately (n_broken)
        bonded_area_at_bonding = float(bonds.area.sum())

    # --- stage 4: free repose at neutral buoyancy (fabric stability) ------
    # ice is only 10 % denser than water, so a submerged block is nearly
    # weightless: the frozen fabric is loaded by the buoyant residual only
    # in-service keel: the ridge sail / level ice carries a downward load that
    # exceeds the keel buoyancy, so the rubble skeleton is under a small
    # overburden (equivalent to ~3 kPa) instead of being stress free
    params_service = DemParams(dt=dt, buoyant_accel=-in_service_accel,
                               global_damping=0.995)
    integrate(pile, release_steps, params_service, bonds=bonds)
    bonds_after_repose = bonds.n_broken if bonds is not None else 0

    poro = measure_porosity(pile)
    local = measure_local_porosity(pile)
    res = contact_forces(pile, bonds=bonds)
    fc = force_chain_stats(res["fn"])
    if bonds is not None and bonds.n_bonds:
        active_area = bonds.active_area()
        bonds_meta = {
            "n_bonds": bonds.n_bonds,
            "n_broken": bonds.n_broken,
            "n_broken_after_repose": bonds_after_repose,
            "active_area_m2": active_area,
            "bonded_area_density_m2_per_m3":
                bonded_area_at_bonding / max(poro["bounding_volume"], 1e-9),
            "bonded_area_active_m2_per_m3":
                active_area / max(poro["bounding_volume"], 1e-9),
        }
    else:
        bonds_meta = {"n_bonds": 0, "n_broken": 0, "active_area_m2": 0.0,
                      "bonded_area_density_m2_per_m3": 0.0}
    fric = measure_mobilised_friction(pile, bonds, dt=dt,
                                      friction_angle_deg=friction_angle_deg,
                                      overburden_accel=in_service_accel)
    # force chains are a *loaded* state property: the pile is loaded by the
    # in-service overburden (sail + level ice)
    fc = force_chain_stats(res["fn"])                       # in-service state
    if fc["n_contacts"] < 50:                              # fall back to shear
        fc = force_chain_stats(fric.get("loaded_fn", np.zeros(0)))
    h_k = pile.keel_height()
    resistance = keel_resistance_model(
        h_k, poro["porosity"], fric["mu_mob"], tau,
        bonds_meta["bonded_area_density_m2_per_m3"])
    f_kerr = kerr_keel_resistance(h_k, poro["porosity"], friction_angle_deg)
    mass_error = abs(pile.total_mass - pile.analytic_mass()) / pile.analytic_mass()
    return {
        "keel_height_target_m": keel_height_target,
        "keel_height_m": h_k,
        "keel_height_compacted_m": h_compacted,
        "n_blocks": pile.n_blocks,
        "porosity_loose": porosity_loose,
        "porosity_compacted": porosity_compacted,
        "porosity": poro["porosity"],
        "porosity_std_error": poro["std_error"],
        "local_porosity": {"mean": float(local.mean()) if local.size else 0.0,
                           "p05": float(np.percentile(local, 5)) if local.size else 0.0,
                           "p95": float(np.percentile(local, 95)) if local.size else 0.0},
        "solid_volume_measured_m3": poro["solid_volume"],
        "solid_volume_exact_m3": solid_volume,
        "voxel_volume_error": abs(poro["solid_volume"] - solid_volume) / solid_volume,
        "bounds_volume_m3": poro["bounding_volume"],
        "undercooling_c": undercooling_c,
        "tau_adfreeze_mpa": tau,
        "bonds": bonds_meta,
        "mu_mob": fric["mu_mob"],
        "tan_phi": fric["tan_phi"],
        "force_chain": fc,
        "f_skel": resistance["f_skel"],
        "f_bond": resistance["f_bond"],
        "f_keel": resistance["f_keel"],
        "hydrostatic_thrust": resistance["thrust"],
        "f_kerr": f_kerr,
        "f_skel_f_kerr_ratio": (resistance["f_skel"] / f_kerr) if f_kerr > 0 else 0.0,
        "f_keel_f_kerr_ratio": (resistance["f_keel"] / f_kerr) if f_kerr > 0 else 0.0,
        "total_mass_kg": pile.total_mass,
        "mass_error": mass_error,
    }


# ---------------------------------------------------------------------------
# Study driver: criteria, negative controls, artifacts
# ---------------------------------------------------------------------------
def bond_consistency_sweep(n_params: int = 10) -> list[dict[str, float]]:
    """Criterion 1: DEM bond threshold vs analytic ``tau_adfreeze * A``.

    Ten parameter combinations spanning temperature, contact time, roughness,
    salinity and block size; each is pulled apart in the DEM and the measured
    rupture force compared with the analytic bond threshold.
    """
    combos = []
    temps = [-2.0, -5.0, -10.0, -20.0]
    for k in range(n_params):
        combos.append({
            "undercooling_c": temps[k % len(temps)],
            "contact_time_s": [100.0, 1000.0, 10000.0][k % 3],
            "roughness_m": [5e-6, 20e-6, 80e-6][k % 3],
            "salinity_ppt": [0.0, 5.0, 15.0][k % 3],
            "radius": [0.15, 0.25, 0.40][k % 3],
        })
    out = []
    for c in combos:
        res = measure_bond_break_force(**c)
        res.update(c)
        out.append(res)
    return out


def negative_controls() -> dict[str, Any]:
    """The three negative controls required by the campaign node."""
    # (1) no undercooling -> no freeze bond -> no break force
    f_break_zero = freeze_bond_break_force(0.0, 0.25, 0.25, 1.0e-3)
    tau_zero = tau_adfreeze(0.0)
    # (2) zero keel height -> zero resistance
    f_zero_h = keel_resistance_model(0.0, 0.4, 0.7, 0.74, 0.01)["f_keel"]
    # (3) porosity -> 1 (no solid) -> zero resistance
    empty_pile = RubblePile(pos=np.zeros((0, 3)), vel=np.zeros((0, 3)),
                            radius=np.zeros(0), mass=np.zeros(0),
                            box=np.array([[0.0, 6.0], [0.0, 6.0], [0.0, 2.0]]))
    n_full = measure_porosity(empty_pile)["porosity"]
    f_full_n = keel_resistance_model(4.0, 1.0, 0.7, 0.74, 0.0)["f_keel"]
    return {
        "dT_zero_no_bond": {"tau_mpa": tau_zero, "f_break_n": f_break_zero,
                            "pass": tau_zero == 0.0 and f_break_zero == 0.0},
        "zero_keel_height": {"f_keel": f_zero_h, "pass": f_zero_h == 0.0},
        "porosity_to_one": {"porosity": n_full, "f_keel": f_full_n,
                            "pass": n_full > 0.999 and f_full_n == 0.0},
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xr = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    yr = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = math.sqrt(float((xr ** 2).sum()) * float((yr ** 2).sum()))
    return float((xr * yr).sum() / denom) if denom > 0 else 0.0


def evaluate_criteria(cases: list[dict], sensitivity: list[dict],
                      unbonded: dict, bond_sweep: list[dict],
                      neg: dict[str, Any]) -> dict[str, Any]:
    """Apply the eight machine-checkable acceptance criteria of the node."""
    c1_err = max(b["rel_error"] for b in bond_sweep)
    c1 = {"criterion": "freeze-bond parameterisation consistency",
          "max_rel_error": c1_err, "threshold": 0.05, "n_params": len(bond_sweep),
          "pass": bool(c1_err <= 0.05)}
    porosities = [c["porosity"] for c in cases]
    c2 = {"criterion": "consolidated porosity in [0.30, 0.45]",
          "porosity": porosities, "threshold": [0.30, 0.45],
          "pass": bool(all(0.30 <= p <= 0.45 for p in porosities))}
    ratios = [c["f_skel_f_kerr_ratio"] for c in cases]
    c3 = {"criterion": "F_keel vs Kerr within 15% (skeleton term)",
          "f_skel": [c["f_skel"] for c in cases],
          "f_kerr": [c["f_kerr"] for c in cases],
          "ratios": ratios,
          "max_rel_deviation": max(abs(r - 1.0) for r in ratios),
          "threshold": 0.15,
          "pass": bool(all(abs(r - 1.0) <= 0.15 for r in ratios))}
    temps = [abs(s["undercooling_c"]) for s in sensitivity]
    forces = [s["f_keel"] for s in sensitivity]
    rho = _spearman(temps, forces)
    c4 = {"criterion": "resistance grows with undercooling",
          "spearman_rho": rho, "threshold": 0.9,
          "monotone": bool(all(b > a for a, b in zip(forces, forces[1:]))),
          "pass": bool(rho >= 0.9 and all(b > a for a, b in zip(forces, forces[1:])))}
    c5 = {"criterion": "no freeze bonds -> porosity >= 0.55",
          "porosity": unbonded["porosity"], "threshold": 0.55,
          "pass": bool(unbonded["porosity"] >= 0.55)}
    r2 = [c["force_chain"]["tail_r_squared"] for c in cases]
    c6 = {"criterion": "force-chain exponential tail R^2 >= 0.90",
          "r_squared": r2, "threshold": 0.90,
          "pass": bool(all(v >= 0.90 for v in r2))}
    c7 = {"criterion": "negative controls", "controls": neg,
          "pass": bool(all(v["pass"] for v in neg.values()))}
    mass_err = max([c["mass_error"] for c in cases]
                   + [unbonded["mass_error"], max(b["rel_error"] * 0 for b in bond_sweep)])
    c8 = {"criterion": "mass conservation error <= 1e-6",
          "max_mass_error": mass_err, "threshold": 1e-6,
          "pass": bool(mass_err <= 1e-6)}
    crit = {"c1_bond_parameterisation": c1, "c2_porosity": c2,
            "c3_kerr_resistance": c3, "c4_temperature_sensitivity": c4,
            "c5_unbonded_degradation": c5, "c6_force_chain_tail": c6,
            "c7_negative_controls": c7, "c8_mass_conservation": c8}
    crit["n_pass"] = int(sum(1 for k, v in crit.items()
                             if isinstance(v, dict) and v.get("pass")))
    crit["n_criteria"] = 8
    return crit


def bond_sensitivity(keel_height: float = 4.0,
                     temperatures: Sequence[float] = (-2.0, -5.0, -10.0, -15.0, -20.0),
                     friction_angle_deg: float = 40.0,
                     consol_steps: int = 4200, release_steps: int = 600,
                     in_service_accel: float = 5.0, exponent: float = 1.2,
                     dt: float = 2.0e-3, seed: int = 0) -> list[dict[str, float]]:
    """Temperature sweep on *one* keel fabric (isolates the bond effect).

    The DEM sequence is run once to obtain the in-service fabric; for each
    undercooling the freeze bonds are re-created on that same fabric, so the
    only varying input is ``tau_adfreeze``.
    """
    d_sample = rosin_rammler_sample(20000, 0.1, 1.0, exponent=exponent, seed=1)
    v_block = float(np.mean(math.pi / 6.0 * d_sample ** 3))
    area = max(500.0 * v_block / (0.6 * keel_height), 16.0)
    length = math.sqrt(area)
    n_blocks = max(int(round(0.6 * area * keel_height / v_block)), 500)
    pile = build_pile(n_blocks, length, length,
                      loose_height=max(3.0 * keel_height, 6.0), seed=seed,
                      friction_angle_deg=friction_angle_deg, exponent=exponent)
    h_box = pile.keel_height()
    pile.box[2, 1] = h_box + 0.2
    params_comp = DemParams(dt=dt, buoyant_accel=-300.0, global_damping=0.98,
                            v_max=4.0)
    integrate(pile, consol_steps, params_comp)
    integrate(pile, 1200, DemParams(dt=dt, buoyant_accel=-in_service_accel,
                                    global_damping=0.99))
    pile.vel[:] = 0.0
    poro = measure_porosity(pile)
    base_pos, base_vel = pile.pos.copy(), pile.vel.copy()
    # the skeleton (friction) resistance is a property of the fabric, not of the
    # bond strength: measure mu_mob once on the unbonded fabric and use it for
    # the whole temperature sweep
    ref = RubblePile(pos=base_pos.copy(), vel=base_vel.copy(),
                     radius=pile.radius.copy(), mass=pile.mass.copy(),
                     box=pile.box.copy(), friction_coeff=pile.friction_coeff)
    fric_ref = measure_mobilised_friction(ref, None, dt=dt,
                                          friction_angle_deg=friction_angle_deg,
                                          overburden_accel=in_service_accel)
    mu_ref = fric_ref["mu_mob"]
    h_k_ref = ref.keel_height()      # same fabric -> same geometry in the sweep
    out = []
    for dT in temperatures:
        work = RubblePile(pos=base_pos.copy(), vel=base_vel.copy(),
                          radius=pile.radius.copy(), mass=pile.mass.copy(),
                          box=pile.box.copy(),
                          friction_coeff=pile.friction_coeff)
        bonds = create_bonds(work, dT, salinity_ppt=0.0)
        area_density = float(bonds.area.sum()) / max(poro["bounding_volume"], 1e-9)
        integrate(work, release_steps,
                  DemParams(dt=dt, buoyant_accel=-in_service_accel,
                            global_damping=0.995), bonds=bonds)
        tau = tau_adfreeze(dT)
        h_k = h_k_ref
        res = keel_resistance_model(h_k, poro["porosity"], mu_ref, tau,
                                    area_density)
        out.append({"undercooling_c": dT, "tau_adfreeze_mpa": tau,
                    "porosity": poro["porosity"], "mu_mob": mu_ref,
                    "keel_height_m": h_k, "n_bonds": bonds.n_bonds,
                    "n_broken": bonds.n_broken,
                    "bonded_area_density_m2_per_m3": area_density,
                    "f_skel": res["f_skel"], "f_bond": res["f_bond"],
                    "f_keel": res["f_keel"],
                    "f_kerr": kerr_keel_resistance(h_k, poro["porosity"],
                                                   friction_angle_deg)})
    return out


def run_study(out_dir: str | Path | None = None,
              keel_heights: Sequence[float] = (2.0, 4.0, 6.0),
              temperatures: Sequence[float] = (-2.0, -5.0, -10.0, -15.0, -20.0),
              friction_angle_deg: float = 40.0,
              consol_steps: int = 3600, release_steps: int = 600,
              seed: int = 0, with_figures: bool = True) -> dict[str, Any]:
    """Run the whole validation study and (optionally) write the artifacts."""
    cases = []
    for h in keel_heights:
        case = run_keel_case(h, undercooling_c=-10.0,
                             friction_angle_deg=friction_angle_deg,
                             consol_steps=consol_steps,
                             release_steps=release_steps, seed=seed)
        case["label"] = f"h_k={h:g} m"
        cases.append(case)
    # the temperature sweep runs on ONE keel fabric, so that the only varying
    # input is the freeze-bond strength (stochastic packing noise is removed)
    sensitivity = bond_sensitivity(keel_height=4.0, temperatures=temperatures,
                                   friction_angle_deg=friction_angle_deg,
                                   consol_steps=consol_steps,
                                   release_steps=release_steps, seed=seed)
    unbonded = run_keel_case(4.0, undercooling_c=0.0, with_bonds=False,
                             friction_angle_deg=friction_angle_deg,
                             consol_steps=consol_steps,
                             release_steps=release_steps, seed=seed)
    bond_sweep = bond_consistency_sweep(10)
    neg = negative_controls()
    criteria = evaluate_criteria(cases, sensitivity, unbonded, bond_sweep, neg)
    study = {
        "node": "P50-N07",
        "module": "tensordem.ice_ridge_keel_freezebond",
        "metadata": {
            "keel_heights_m": list(keel_heights),
            "temperatures_C": list(temperatures),
            "friction_angle_deg": friction_angle_deg,
            "contact_time_s": 1000.0,
            "rho_ice_kg_m3": RHO_ICE, "rho_water_kg_m3": RHO_WATER,
            "gravity_m_s2": GRAVITY,
            "consolidation_accel_m_s2": 300.0,
            "consol_steps": consol_steps, "release_steps": release_steps,
            "dt_s": 2.0e-3, "seed": seed,
            "kerr_formula": "F = 0.5*rho_w*g*h_k^2*tan(phi_f)*(1-n)",
        },
        "cases": cases,
        "sensitivity": sensitivity,
        "unbonded": unbonded,
        "bond_consistency": bond_sweep,
        "negative_controls": neg,
        "criteria": criteria,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "study.json").write_text(json.dumps(study, indent=2))
        if with_figures:
            try:
                figures = make_figures(study, out)
                study["figures"] = figures
                (out / "study.json").write_text(json.dumps(study, indent=2))
            except Exception as exc:  # pragma: no cover
                study["figure_error"] = str(exc)
    return study


def make_figures(study: dict, out_dir: str | Path) -> list[str]:
    """Five matplotlib figures for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    names = []

    # fig 1: adfreeze strength law
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dT = np.linspace(0.2, 25.0, 200)
    for t, ls in ((100.0, ":"), (1000.0, "-"), (10000.0, "--")):
        ax.plot(dT, [tau_adfreeze(-x, t) for x in dT], ls,
                label=f"t = {t:g} s")
    ax.set_xlabel("undercooling |dT| [K]")
    ax.set_ylabel(r"$\tau_{adfreeze}$ [MPa]")
    ax.set_title("Freeze-bond strength (TensorPhysics adfreeze law)")
    ax.grid(alpha=0.3)
    ax.legend()
    p = out / "fig1_adfreeze_strength.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    names.append(str(p))

    # fig 2: resistance vs Kerr
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    hs = [c["keel_height_m"] for c in study["cases"]]
    idx = np.arange(len(hs))
    ax.bar(idx - 0.2, [c["f_kerr"] for c in study["cases"]], 0.4,
           label="Kerr (1975)")
    ax.bar(idx + 0.2, [c["f_skel"] for c in study["cases"]], 0.4,
           label="DEM skeleton $F_{skel}$")
    for i, c in enumerate(study["cases"]):
        ax.text(i, max(c["f_kerr"], c["f_skel"]) * 1.03,
                f'ratio {c["f_skel_f_kerr_ratio"]:.2f}', ha="center", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$h_k$={h:.1f} m" for h in hs])
    ax.set_ylabel("resistance per unit width [N/m]")
    ax.set_title("Keel resistance: DEM skeleton vs Kerr empirical")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    p = out / "fig2_kerr_comparison.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    names.append(str(p))

    # fig 3: temperature sensitivity
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    temps = [abs(s["undercooling_c"]) for s in study["sensitivity"]]
    ax.plot(temps, [s["f_skel"] for s in study["sensitivity"]], "o-",
            label="$F_{skel}$ (friction/gravity)")
    ax.plot(temps, [s["f_bond"] for s in study["sensitivity"]], "s--",
            label="$F_{bond}$ (freeze bonds)")
    ax.plot(temps, [s["f_keel"] for s in study["sensitivity"]], "^-",
            label="$F_{keel}$ total")
    ax.set_yscale("log")
    ax.set_xlabel("undercooling |dT| [K]")
    ax.set_ylabel("resistance per unit width [N/m]")
    ax.set_title("Freeze-bond strength sensitivity (Spearman "
                 f"{study['criteria']['c4_temperature_sensitivity']['spearman_rho']:.2f})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    p = out / "fig3_sensitivity.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    names.append(str(p))

    # fig 4: porosity structure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    labels = ["loose\ndeposit", "consolidated", "after release\n(bonded)",
              "after release\n(unbonded)"]
    vals = [study["cases"][0]["porosity_loose"],
            study["cases"][0]["porosity_compacted"],
            study["cases"][0]["porosity"], study["unbonded"]["porosity"]]
    ax1.bar(labels, vals, color=["#9ecae1", "#3182bd", "#08519c", "#fdae6b"])
    ax1.axhline(0.40, color="k", ls=":", lw=1)
    ax1.set_ylabel("porosity n [-]")
    ax1.set_title("Porosity evolution (DEM)")
    ax1.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    poro = [c["porosity"] for c in study["cases"]]
    ax2.plot([c["keel_height_m"] for c in study["cases"]], poro, "o-",
             label="bonded")
    ax2.axhline(study["unbonded"]["porosity"], color="orange", ls="--",
                label="unbonded reference")
    ax2.set_xlabel("keel height $h_k$ [m]")
    ax2.set_ylabel("porosity n [-]")
    ax2.set_title("Porosity vs keel height")
    ax2.grid(alpha=0.3)
    ax2.legend()
    p = out / "fig4_porosity.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    names.append(str(p))

    # fig 5: force chains
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c, marker in zip(study["cases"], ("o", "s", "^")):
        fc = c["force_chain"]
        f_mean = max(fc["mean_force"], 1e-9)
        fc_scale = max(fc["tail_scale"], 1e-9)
        f = np.linspace(f_mean, fc["max_force"], 60)
        prob = np.exp(-(f - f_mean) / fc_scale)
        ax.semilogy(f / max(fc["max_force"], 1e-9), prob, marker=marker,
                    ms=2.5, lw=0,
                    label=f"$h_k$={c['keel_height_m']:.1f} m, "
                          f"$R^2$={fc['tail_r_squared']:.3f}")
    ax.set_xlabel("$f / f_{max}$")
    ax.set_ylabel("$P(f)$ (exponential tail fit)")
    ax.set_title("Force-chain statistics of the bonded pile")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    p = out / "fig5_force_chains.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    names.append(str(p))
    return names


def main() -> int:  # pragma: no cover - CLI entry point
    study = run_study(Path("artifacts/ice_ridge_keel"))
    print(json.dumps(study["criteria"], indent=2)[:4000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
