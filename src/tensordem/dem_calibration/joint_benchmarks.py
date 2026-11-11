"""The three macroscopic DEM benchmarks that drive the joint calibration.

=================  ==========================================  =================
benchmark          protocol (DEM)                              measured response
=================  ==========================================  =================
``repose``         point-source pouring onto a rough floor     free-surface slope
                   (funnel-discharge analogue), settled        theta [deg], phi
``shear``          dead-load direct-shear box (rough floor,     tau(gamma) curve,
                   rough weighted top platen, constant          peak / residual
                   normal load), strain-controlled shear         shear stress
``uniaxial``       bonded cylindrical specimen compressed       sigma(eps) curve,
                   between two platens, loaded past peak        peak stress, E
=================  ==========================================  =================

Why three tests instead of one: a single repose test only sees ``mu`` and weakly
``e``.  The heap-only calibration of
:mod:`tensordem.dem_calibration.pipeline` FAILED on exactly that
(``docs/DEM_AUTO_CALIBRATION.md``: 9.75 deg repose error for sand, non-monotone
theta(mu)).  The shear curve pins ``mu`` together with ``k_t`` (the tangential
stiffness sets the peak-to-residual drop) and the bonded uniaxial test is the
only benchmark that sees ``sigma_bond``.

The contact engine (walls, bonds, integrator) is
:mod:`tensordem.dem_calibration.contact_engine`; particle contacts are the
verified :func:`tensordem.granular3d_core.contact_forces` kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..granular3d_core import ContactState, hcp_positions_in_box, hcp_positions_in_cylinder
from .contact_engine import (
    ContactParams,
    GrainMaterial,
    WallState,
    hash_positions,
    integrate,
    make_bonds,
    moving_average,
)

__all__ = [
    "ReposeCase",
    "ShearCase",
    "UniaxialCase",
    "repose_angle",
    "run_repose",
    "run_shear",
    "run_uniaxial",
]


# ---------------------------------------------------------------------------
# benchmark 1: point-source pouring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReposeCase:
    """Angle of repose from the collapse of a dense column on a rough floor.

    Protocol (Lube et al. 2004 column-collapse geometry, used here as a
    flowability/repose benchmark): a dense HCP block ``block_width`` grains wide
    and ``block_layers`` grains tall is released on a rough floor, spreads and
    comes to rest; the free-surface slope of the deposit is the repose angle.

    Why collapse instead of pouring: pouring grain-by-grain or in batches into a
    small DEM specimen is *not* a well-posed repose test at this particle count.
    Both failure modes were measured on this engine -- an under-settled pour
    builds a vertical tower (theta = 60-90 deg) while a splashy pour spreads the
    grains into a monolayer with a non-monotone theta(mu).  The dense-column
    collapse is reproducible, dense (phi ~ 0.5), and gives a monotonically
    increasing theta(mu) (see ``docs/P284W3_N08_JOINT_CALIBRATION.md``).
    """

    material: GrainMaterial = field(default_factory=GrainMaterial)
    block_width: int = 7          # grains across the dense block
    block_layers: int = 4         # grains tall
    settle_steps: int = 1600      # relaxation to the final deposit
    packing_gap: float = 0.02
    jitter: float = 0.02
    floor_mu: float | None = None  # None -> mu (rough floor)
    seed: int = 5

    def __post_init__(self) -> None:
        if self.block_width < 4 or self.block_layers < 2:
            raise ValueError("block must be at least 4 grains wide and 2 tall")


def run_repose(case: ReposeCase, params: ContactParams) -> dict:
    """Collapse a dense column and measure the free-surface slope of the deposit."""
    mat = case.material
    core = mat.core(params)
    r = mat.radius
    dt = core.timestep
    floor_mu = params.mu if case.floor_mu is None else case.floor_mu

    width = case.block_width * 2.2 * r
    pos = hcp_positions_in_box(r, width, width, case.block_layers * 2.2 * r,
                               packing_gap=case.packing_gap, jitter=case.jitter,
                               seed=case.seed)
    pos = np.column_stack([pos[:, 0] - pos[:, 0].mean(),
                           pos[:, 1] - pos[:, 1].mean(),
                           pos[:, 2] - pos[:, 2].min() + r])
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    state = ContactState()
    floor_state = WallState()
    floor_point = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    zero = np.zeros(3)

    walls = {"floor": floor_state, "sides": [], "platen": WallState()}
    n_steps = integrate(pos, vel, omega, state, floor_state, core, mat, params, dt,
                        case.settle_steps, floor_mu, floor_point, up, zero,
                        walls=walls)

    ke = 0.5 * mat.mass * float(np.sum(vel ** 2))
    pe = mat.mass * mat.gravity * float(np.sum(np.maximum(pos[:, 2], 0.0)))
    try:
        fit = repose_angle(pos, r)
    except ValueError as exc:                      # unmeasurable deposit -> NaN
        fit = {"theta_deg": float("nan"), "r2": 0.0, "n_fit_bins": 0,
               "phi_solid": float("nan"), "error": str(exc)}
    out = dict(fit)
    out.update({
        "n_grains": int(len(pos)),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "ke_final_J": ke,
        "pe_final_J": pe,
        "ke_ratio": ke / pe if pe > 0 else float("nan"),
        "max_speed_m_s": float(np.linalg.norm(vel, axis=1).max()),
        "digest": hash_positions(pos),
        "positions": pos,
        "params": params.as_dict(),
    })
    return out


def repose_angle(pos: np.ndarray, radius: float, *, z_floor: float = 0.0,
                 bin_width_factor: float = 1.6, toe_quantile: float = 0.97,
                 flank_lo: float = 0.20, flank_hi: float = 0.85,
                 top_k: int = 3) -> dict:
    """Free-surface slope of a heap by a robust radial top-surface fit.

    The surface height in each annulus of width ``bin_width_factor * r`` is the
    mean of the ``top_k`` highest grain centres (robust against a single splash
    grain); a straight line is fitted to the flank between ``flank_lo`` and
    ``flank_hi`` of the toe radius.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 10:
        raise ValueError("need at least 10 grains to fit a repose angle")
    rho = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2] - z_floor
    toe = float(np.quantile(rho, toe_quantile))
    if toe <= 0.0:
        raise ValueError("degenerate pile (zero toe radius)")
    width = bin_width_factor * radius
    n_bins = max(int(math.floor(toe / width)), 3)
    edges = np.linspace(0.0, toe, n_bins + 1)
    centres, heights = [], []
    for k in range(n_bins):
        sel = (rho >= edges[k]) & (rho < edges[k + 1])
        n_sel = int(sel.sum())
        if n_sel < 3:
            continue
        zk = np.sort(z[sel])[::-1][:min(top_k, n_sel)]
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        heights.append(float(zk.mean()))
    centres = np.asarray(centres)
    heights = np.asarray(heights)
    if centres.size < 4:
        raise ValueError("fewer than four usable radial bins")
    sel = (centres >= flank_lo * toe) & (centres <= flank_hi * toe)
    if int(sel.sum()) < 3:
        raise ValueError("fewer than three flank bins")
    slope, intercept = np.polyfit(centres[sel], heights[sel], 1)
    pred = slope * centres[sel] + intercept
    ss_res = float(np.sum((heights[sel] - pred) ** 2))
    ss_tot = float(np.sum((heights[sel] - heights[sel].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if slope >= 0.0:
        raise ValueError("free surface is not descending: no pile flank")
    theta = math.degrees(math.atan(-slope))
    volume = math.pi / 3.0 * toe ** 2 * max(intercept, 0.0)
    grain_volume = len(pos) * 4.0 / 3.0 * math.pi * radius ** 3
    phi = grain_volume / volume if volume > 0.0 else float("nan")
    return {
        "theta_deg": float(theta),
        "r2": float(r2),
        "n_fit_bins": int(sel.sum()),
        "toe_radius": toe,
        "apex_z": float(np.max(z)),
        "phi_solid": float(min(max(phi, 0.0), 1.0)),
    }


# ---------------------------------------------------------------------------
# benchmark 2: direct shear box under a force-servoed dead load
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShearCase:
    """Constant-normal-load direct-shear box, strain-controlled shear."""

    material: GrainMaterial = field(default_factory=GrainMaterial)
    box_width: float = 0.060      # x (shear direction)
    box_depth: float = 0.040      # y
    fill_height: float = 0.030    # z
    normal_stress: float = 1.0e3  # Pa -- reference level for the (measured) load
    consolidation_strain: float = 0.02  # press depth / bed height (constant volume)
    shear_strain_target: float = 0.12
    shear_velocity: float = 0.012  # m/s platen speed
    settle_steps: int = 250        # pre-consolidation under gravity
    consol_steps: int = 300        # kinematic press steps
    side_mu: float = 0.0           # frictionless side walls
    floor_mu: float | None = None  # None -> mu (rough floor)
    platen_mu: float | None = None  # None -> mu (rough platen)
    packing_gap: float = 0.02
    servo_min_contacts: int = 4
    seed: int = 7

    @property
    def area(self) -> float:
        return self.box_width * self.box_depth


def run_shear(case: ShearCase, params: ContactParams) -> dict:
    """Constant-volume direct shear; returns the ``tau(gamma)`` curve.

    Protocol: the bed is settled under gravity, the top platen is pressed
    kinematically to a fixed height (constant-volume / undrained shear, the
    standard dense-packing direct-shear test), then the platen is dragged in
    ``+x`` at constant speed.  The normal stress is *measured* (an output, not a
    servo set point) -- a force/depth servo on the platen turned out to be the
    numerically fragile part of this device on a 96-grain bed (the free platen
    DOF is stiff compared with the grain mass and the bed expanded under load),
    so the normal load is prescribed by volume instead.
    """
    mat = case.material
    core = mat.core(params)
    r = mat.radius
    dt = core.timestep
    floor_mu = params.mu if case.floor_mu is None else case.floor_mu
    platen_mu = params.mu if case.platen_mu is None else case.platen_mu

    pos = hcp_positions_in_box(r, case.box_width, case.box_depth, case.fill_height,
                               packing_gap=case.packing_gap, jitter=0.02,
                               seed=case.seed)
    z = pos[:, 2] - pos[:, 2].min()
    pos = np.column_stack([pos[:, 0] - pos[:, 0].mean(),
                           pos[:, 1] - pos[:, 1].mean(), z + r])
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    state = ContactState()
    floor_state = WallState()
    floor_point = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    zero = np.zeros(3)

    n_particles = len(pos)
    half_w, half_d = 0.5 * case.box_width, 0.5 * case.box_depth
    side_walls = [
        (np.array([1.0, 0.0, 0.0]), np.array([half_w, 0.0, 0.0]), case.side_mu),
        (np.array([-1.0, 0.0, 0.0]), np.array([-half_w, 0.0, 0.0]), case.side_mu),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, half_d, 0.0]), case.side_mu),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, -half_d, 0.0]), case.side_mu),
    ]

    walls = {"floor": floor_state, "sides": [WallState() for _ in side_walls],
             "platen": WallState()}
    platen = {
        "normal": -up,
        "point": np.array([0.0, 0.0, float(pos[:, 2].max()) + r + 0.1 * r]),
        "velocity": np.zeros(3),
        "mu": platen_mu,
        "force_on_platen": np.zeros(3),
        "f_n_total": 0.0,
        "n_contacts": 0,
        "free_vertical": False,
    }

    integrate(pos, vel, omega, state, floor_state, core, mat, params, dt,
              case.settle_steps, floor_mu, floor_point, up, zero,
              side_walls=side_walls, platen=platen, walls=walls)

    # --- press to the target height (constant volume), slowly ------------- #
    bed_top = float(pos[:, 2].max()) + r
    target = bed_top - case.consolidation_strain * bed_top
    n_press = max(case.consol_steps, 8)
    for k in range(n_press):
        frac = (k + 1) / n_press
        platen["point"] = np.array([0.0, 0.0, bed_top + (target - bed_top) * frac])
        platen["velocity"] = np.array([0.0, 0.0, (target - bed_top) / n_press / dt])
        integrate(pos, vel, omega, state, floor_state, core, mat, params, dt, 1,
                  floor_mu, floor_point, up, zero,
                  side_walls=side_walls, platen=platen, walls=walls)
    height = float(pos[:, 2].max()) + r
    sigma_n_consolidated = float(platen["f_n_total"] / case.area)

    # --- strain-controlled shear ------------------------------------------ #
    n_shear = max(int(round(case.shear_strain_target * height /
                            (case.shear_velocity * dt))), 8)
    disp, tau, sigma_n = [], [], []
    x_platen = 0.0
    for step in range(n_shear + 1):
        if step:
            x_platen += case.shear_velocity * dt
            platen["point"] = np.array([x_platen, 0.0, target])
            platen["velocity"] = np.array([case.shear_velocity, 0.0, 0.0])
            integrate(pos, vel, omega, state, floor_state, core, mat, params, dt, 1,
                      floor_mu, floor_point, up, zero,
                      side_walls=side_walls, platen=platen, walls=walls)
        disp.append(float(x_platen))
        tau.append(float(abs(platen["force_on_platen"][0]) / case.area))
        sigma_n.append(float(platen["f_n_total"] / case.area))

    disp = np.asarray(disp)
    tau = np.asarray(tau)
    sigma_n = np.asarray(sigma_n)
    strain = disp / height
    smooth = moving_average(tau, max(min(21, len(tau) // 4), 1))
    peak_idx = int(np.argmax(smooth))
    n_tail = max(int(0.2 * len(smooth)), 3)
    residual = float(np.mean(smooth[-n_tail:]))
    escaped = bool(np.any(pos[:, 2] < -1.5 * r)
                   or np.any(np.abs(pos[:, 0]) > half_w + r)
                   or np.any(np.abs(pos[:, 1]) > half_d + r))
    return {
        "displacement": disp,
        "strain": strain,
        "tau": tau,
        "tau_smooth": smooth,
        "sigma_n": sigma_n,
        "height": height,
        "peak_tau": float(smooth[peak_idx]),
        "peak_strain": float(strain[peak_idx]),
        "residual_tau": residual,
        "sigma_n_consolidated": sigma_n_consolidated,
        "sigma_n_measured": float(np.mean(sigma_n)),
        "sigma_n_rel_err": float(np.mean(sigma_n) / case.normal_stress - 1.0),
        "mu_macro": residual / float(np.mean(sigma_n)) if np.mean(sigma_n) > 0 else float("nan"),
        "n_particles": int(n_particles),
        "n_steps": int(case.settle_steps + n_press + n_shear),
        "dt": float(dt),
        "escaped": escaped,
        "digest": hash_positions(pos),
        "params": params.as_dict(),
    }


# ---------------------------------------------------------------------------
# benchmark 3: bonded uniaxial compression (UCS)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniaxialCase:
    """Cylindrical bonded specimen compressed between two platens."""

    material: GrainMaterial = field(default_factory=GrainMaterial)
    specimen_radius: float = 0.018   # m
    fill_height: float = 0.070       # m
    packing_gap: float = 0.02
    strain_target: float = 0.08
    loading_velocity: float = 0.010  # m/s platen speed
    settle_steps: int = 200
    shear_bond_ratio: float = 0.5    # shear/tensile bond strength
    bond_scatter: float = 0.35       # log-normal bond-strength disorder
    jitter: float = 0.03             # packing disorder [x r]
    seed: int = 3


def run_uniaxial(case: UniaxialCase, params: ContactParams) -> dict:
    """Load a bonded specimen to post-peak and return ``sigma(eps)``."""
    mat = case.material
    core = mat.core(params)
    r = mat.radius
    dt = core.timestep

    pos = hcp_positions_in_cylinder(r, case.specimen_radius, case.fill_height,
                                   packing_gap=case.packing_gap, jitter=case.jitter,
                                   seed=case.seed)
    pos = np.column_stack([pos[:, 0] - pos[:, 0].mean(),
                           pos[:, 1] - pos[:, 1].mean(),
                           pos[:, 2] - pos[:, 2].min() + r])
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    state = ContactState()
    floor_state = WallState()
    floor_point = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    zero = np.zeros(3)
    bonds = make_bonds(pos, r, params, case.shear_bond_ratio,
                       scatter=case.bond_scatter, seed=case.seed)
    n_bonds0 = bonds.n_intact

    platen = {
        "normal": -up,
        "point": np.array([0.0, 0.0, float(pos[:, 2].max()) + r + 0.5 * r]),
        "velocity": np.zeros(3),
        "mu": params.mu,
        "force_on_platen": np.zeros(3),
        "f_n_total": 0.0,
        "free_vertical": False,
    }

    walls = {"floor": floor_state, "sides": [], "platen": WallState()}
    # settle the specimen (bonds active, no external load), then seat the platen
    integrate(pos, vel, omega, state, floor_state, core, mat, params, dt,
              case.settle_steps, params.mu, floor_point, up, zero,
              platen=platen, bonds=bonds, walls=walls)
    height0 = float(pos[:, 2].max()) + r
    platen["point"] = np.array([0.0, 0.0, height0])
    area0 = math.pi * case.specimen_radius ** 2

    n_load = max(int(round(case.strain_target * height0 /
                           (case.loading_velocity * dt))), 8)
    strain, sigma, intact = [], [], []
    for step in range(n_load + 1):
        if step:
            z_p = platen["point"][2] - case.loading_velocity * dt
            platen["point"] = np.array([0.0, 0.0, z_p])
            platen["velocity"] = np.array([0.0, 0.0, -case.loading_velocity])
            integrate(pos, vel, omega, state, floor_state, core, mat, params, dt, 1,
                      params.mu, floor_point, up, zero, platen=platen, bonds=bonds,
                      walls=walls)
        disp = height0 - float(platen["point"][2])
        strain.append(float(disp / height0))
        sigma.append(float(platen["force_on_platen"][2] / area0))
        intact.append(int(bonds.n_intact))

    strain = np.asarray(strain)
    sigma = np.asarray(sigma)
    smooth = moving_average(sigma, max(min(21, len(sigma) // 4), 1))
    peak_idx = int(np.argmax(smooth))
    modulus = float("nan")
    if peak_idx >= 3:
        lin = (strain >= 0.15 * strain[peak_idx]) & (strain <= 0.60 * strain[peak_idx])
        if int(lin.sum()) >= 3:
            modulus = float(np.polyfit(strain[lin], smooth[lin], 1)[0])
    return {
        "strain": strain,
        "stress": sigma,
        "stress_smooth": smooth,
        "peak_stress": float(smooth[peak_idx]),
        "peak_strain": float(strain[peak_idx]),
        "modulus": modulus,
        "undeformed_height": height0,
        "area": area0,
        "n_bonds_initial": int(n_bonds0),
        "n_bonds_intact": int(bonds.n_intact),
        "intact_fraction": float(bonds.n_intact / max(n_bonds0, 1)),
        "bond_strength_N": float(np.mean(bonds.tensile_strength)),
        "n_particles": int(len(pos)),
        "n_steps": int(case.settle_steps + n_load),
        "dt": float(dt),
        "digest": hash_positions(pos),
        "params": params.as_dict(),
    }