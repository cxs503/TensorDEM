"""Quasi-static sphere penetration into a granular bed (Brzinski 2013 anchor).

Why this module exists
----------------------
Mission m15367 built a sphere-penetration case on
:class:`tensordem.sphere_pack_dem.SpherePack` but recorded an honest FAIL: the
*loading protocol* injected energy instead of resolving the quasi-static force.
The measured system kinetic energy rose monotonically to 1e2 J (the whole bed
weighs ~0.8 kg, so 1e2 J is the energy of a ~13 m free fall) and the intruder
force jittered between 3 N and 272 N.

The cause is a *scale error in the loading path*, not a calibration error::

    contact stiffness (Hertz, R_eff = r/2)     k ~ 3e7 N/m
    quasi-static resistance of the intruder    F ~ 1-5 N
    => elastic overlap at that load            delta ~ 20 um (0.5 % of d)
    per sub-increment imposed by m15367        0.5 mm = 25 x delta

so every increment threw the bed 25 contact-deformation lengths out of
equilibrium: the elastic energy stored in one step (0.5 J per contact) dwarfs
the gravitational energy scale of the bed (2e-3 J), grains are launched, and the
recorded "equilibrium force" is the force of a shock, not of a quasi-static
indentation.

This module replaces the *displacement-teleport* protocol with standard DEM
quasi-static practice:

1. **Velocity (kinematic) boundary condition** on the indenter.  The indenter is
   a ``fixed`` body whose velocity is prescribed to ``-v_pen``; the integrator
   advances its position by ``v_pen * dt`` per step and evaluates the contact
   force at the drifted position (the engine's half-step position correction).
   The per-step advance (~1e-6 m) is far below the elastic overlap scale, so no
   contact is ever thrown out of equilibrium.
2. **Rate-independent contact law for the measurement.**  The contact dashpot is
   switched off (``restitution = 1``) so the force read off the indenter is a
   pure function of geometry and friction; dissipation is provided by the
   background viscous drag ``damping_global``, which acts on the *grain* force
   balance and therefore does not enter the indenter's contact force.
3. **Quasi-staticity is measured, not assumed.**  The driver reports the kinetic
   energy maximum over the loading stage, the maximum contact-overlap ratio, the
   speed ratio ``v/sqrt(gR)``, and it supports rate-doubling runs so that rate
   independence is demonstrated rather than asserted.
4. **A negative control lives in the module** (:class:`TeleportControl`): it
   reproduces the m15367 path (teleport by ``0.125 d`` then relax) on the *same*
   bed, so the energy blow-up is a reproducible before/after contrast.

Reference law (Brzinski III, Mayor & Durian, PRL 111, 168002 (2013), arXiv:1307.4638v2)
-------------------------------------------------------------------------------------
Local friction law, p. 1 Eq. (2)::

    dF = alpha * mu * (rho_bulk * g * z) * dA,        alpha = 35 +- 5

integrated over the wetted spherical cap, p. 2 Eq. (3c)/(3d)::

    z <= R:  F(z) / (alpha mu rho_bulk g) = pi (R - z/3) z**2
    z >= R:  F(z) / (alpha mu rho_bulk g) = pi (z - R/3) R**2

so for ``z >= R`` the resistance is linear in depth with

    dF/dz = alpha * mu * rho_bulk * g * pi * R**2
    =>  alpha = (dF/dz) / (mu * rho_bulk * g * pi * R**2).

``rho_bulk`` is *measured* from the simulated packing (``phi * rho_particle``,
with ``phi`` from the particle count and the measured free-surface height);
only ``alpha``, ``mu`` and the functional form are taken from the paper.

Scaling note (honest): the DEM cannot afford R/d ~ 42 (the experiment's grains
are 300 um), so runs are coarse-grained to R/d = 2-4; the alpha sensitivity to
R/d is part of the deliverable, not an afterthought.

Usage
-----
    cfg = PenetrationConfig(intruder_radius_over_d=3.0)
    result = QuasiStaticPenetration(cfg).run(verbose=True)
    print(result["metrics"]["alpha_fit"], result["acceptance"])

Command line (one case; writes ``result.json`` and optional figures)::

    python -m tensordem.penetration_quasistatic run --preset base --out artifacts/case_penetration/base
    python -m tensordem.penetration_quasistatic run --preset rate2x --out artifacts/case_penetration/rate2x
    python -m tensordem.penetration_quasistatic control --preset base --out artifacts/case_penetration/teleport_control
    python -m tensordem.penetration_quasistatic plot --input artifacts/case_penetration/base/result.json --out artifacts/case_penetration/base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .granular3d_core import hcp_positions_in_box
from .sphere_pack_dem import SpherePack, SpherePackConfig, SphereWall
from .superquadric_kernel import ContactMaterial

G = 9.81

# --- reference constants, quoted from Brzinski et al. (2013) -----------------
REFERENCE = {
    "source": "Brzinski III, Mayor & Durian, PRL 111, 168002 (2013), arXiv:1307.4638v2",
    "eq2": "dF = alpha mu (rho_bulk g z) dA          (p.1, Eq.2)",
    "eq3c": "F/(alpha mu rho g) = pi (R - z/3) z^2      (p.2, Eq.3c, z<=R)",
    "eq3d": "F/(alpha mu rho g) = pi (z - R/3) R^2      (p.2, Eq.3d, z>=R)",
    "alpha": 35.0,          # p.1, "alpha = 35 +- 5"
    "alpha_err": 5.0,
    "mu": 0.40,             # p.1, tan(22 deg) repose angle of the glass beads
    "phi": 0.59,            # p.1
    "rho_bulk": 1480.0,     # p.1, 1.48 g/cm^3
    "grain_diameter": 3.0e-4,
    "intruder_radii": [1.27e-2, 2.54e-2, 5.05e-2],  # p.2
}

# --- acceptance thresholds (mission m15404) ----------------------------------
ACCEPTANCE = {
    "kinetic_energy_max_J": 0.10,            # <= 0.1 J during the loading stage
    "force_variation_coefficient": 0.10,     # <= 10 % of the mean force
    "penetration_force_error": 0.15,         # <= 15 % vs Brzinski Eq.(2)/(3d)
    "max_overlap_ratio": 0.05,               # engineering sanity: overlap < 5 % d
}


# ---------------------------------------------------------------------------
# Reference law
# ---------------------------------------------------------------------------

def brzinski_shape(z, radius: float) -> np.ndarray:
    """``F/(alpha mu rho_bulk g)`` of Eq. (3c)/(3d) in m^3.

    ``pi (R - z/3) z**2`` for ``z <= R`` and ``pi (z - R/3) R**2`` above; the two
    branches match in value *and* slope at ``z = R`` (C1), which the test suite
    asserts.
    """
    z = np.asarray(z, dtype=float)
    zc = np.clip(z, 0.0, None)
    shallow = math.pi * (radius - zc / 3.0) * zc ** 2
    deep = math.pi * (zc - radius / 3.0) * radius ** 2
    return np.where(zc <= radius, shallow, deep)


def brzinski_force(z, radius: float, mu: float = REFERENCE["mu"],
                   rho_bulk: float = REFERENCE["rho_bulk"],
                   alpha: float = REFERENCE["alpha"],
                   g: float = G) -> np.ndarray:
    """Depth-dependent penetration resistance ``F(z)`` of Eq. (3c)/(3d) in N."""
    return alpha * mu * rho_bulk * g * brzinski_shape(z, radius)


def alpha_from_slope(slope_n_per_m: float, radius: float, mu: float,
                     rho_bulk: float, g: float = G) -> float:
    """Invert Eq. (3d) for the friction amplification factor ``alpha``."""
    return slope_n_per_m / (mu * rho_bulk * g * math.pi * radius ** 2)


def kinetic_energy_free(sp) -> float:
    """Kinetic energy of the *free* bodies only.

    A ``fixed`` drive body carries a prescribed velocity, so its ``0.5 m v^2`` is
    bookkeeping of the boundary condition, not energy of the medium.  Including it
    would make the acceptance metric depend on the indenter speed trivially.
    """
    m = sp.mass[sp.moving]
    v = sp.vel[sp.moving]
    inertia = sp.inertia[sp.moving]
    omega = sp.omega[sp.moving]
    return (0.5 * float(np.sum(m * np.einsum("ij,ij->i", v, v)))
            + 0.5 * float(np.sum(inertia[:, None] * omega ** 2)))


def kinetic_energy_drive(sp) -> float:
    """Kinetic energy carried by the fixed (prescribed) bodies."""
    m = sp.mass[sp.fixed]
    v = sp.vel[sp.fixed]
    if len(m) == 0:
        return 0.0
    return 0.5 * float(np.sum(m * np.einsum("ij,ij->i", v, v)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PenetrationConfig:
    """Geometric / material / protocol parameters of one penetration case.

    Lengths are expressed in grain diameters (``d``) or in intruder radii (``R``)
    so the *dimensionless* set -- the only thing the continuum law cares about --
    is explicit.
    """
    grain_diameter: float = 4.0e-3
    intruder_radius_over_d: float = 3.0
    wall_half_width_over_R: float = 4.0     # box centre -> wall clearance
    bed_depth_over_d: float = 8.0           # initial fill height
    density_particle: float = 2500.0
    mu_grain: float = 0.40
    mu_rolling: float = 0.05
    # Contact dashpot OFF by default: with restitution < 1 the normal force
    # carries a -c_n v_n term, and at v_pen = 0.5 m/s that term (O(1) N per
    # contact) is as large as the whole quasi-static resistance.  Dissipation is
    # supplied by damping_global instead, which does not enter the force read off
    # the intruder.
    restitution: float = 1.0
    e_star: float = 1.0e9
    g_star: float = 4.0e8
    damping: float = 500.0                  # background drag, 1/s
    damping_rotational: float = 500.0
    packing_gap: float = 0.08
    settle_jitter_over_d: float = 0.05
    # Bed preparation uses its own dissipation: the HCP lattice carries a 2 %
    # surface gap, so grains must actually *fall* onto each other and form a
    # self-supporting contact network before the indentation starts.  With the
    # loading-phase drag alone (500 1/s) the terminal speed is g/gamma = 2 cm/s
    # and 1500 steps close none of the 80 um gaps (measured: zero contacts after
    # settling -> the indenter pushed a floating lattice, n_contacts 1 -> 1586
    # and KE 60 J).  Contact damping *during preparation only* compacts the bed;
    # it cannot bias the measurement because the loading phase runs with the
    # dashpot off and starts from rest.
    settle_restitution: float = 0.4
    settle_damping: float = 50.0
    settle_steps: int = 15000
    # Recipe chosen by artifacts/case_penetration/tools/bed_recipes.py: the HCP
    # lattice dilated by 8 % deposits to phi = 0.597 / rho_bulk = 1493 kg/m3
    # after 15 000 steps (reference medium: phi = 0.59, rho_bulk = 1480), with a
    # self-supporting network (157 contacts at rest).  The denser 2 % variant
    # (phi = 0.654) and the looser 10 % variant (still falling, 0 contacts) are
    # recorded in that script's log.
    penetration_speed: float = 0.05         # m/s, indenter velocity (kinematic BC)
    penetrate_over_r: float = 1.6
    fit_from_over_r: float = 0.5
    fit_to_over_r: Optional[float] = None   # default: penetrate_over_r
    smoothing_over_r: float = 0.15          # moving-average window, in units of R
    sample_every_steps: int = 5
    abort_overlap_ratio: float = 0.25       # honest abort: contact overlap > 25 % d
    seed: int = 20261112
    label: str = "base"

    # -- derived ----------------------------------------------------------
    @property
    def d(self) -> float:
        return self.grain_diameter

    @property
    def r(self) -> float:
        return 0.5 * self.grain_diameter

    @property
    def R(self) -> float:
        return self.intruder_radius_over_d * self.grain_diameter

    @property
    def box_width(self) -> float:
        return 2.0 * self.wall_half_width_over_R * self.R

    @property
    def bed_depth(self) -> float:
        return self.bed_depth_over_d * self.grain_diameter

    @property
    def fit_to(self) -> float:
        return self.penetrate_over_r if self.fit_to_over_r is None else self.fit_to_over_r

    def validate(self) -> None:
        if self.intruder_radius_over_d < 2.0:
            raise ValueError("R/d < 2 is not a coarse-grained sphere: pick R/d >= 2")
        if self.bed_depth < self.penetrate_over_r * self.R + 3.0 * self.d:
            raise ValueError(
                "bed too shallow: need bed_depth >= penetrate_over_r*R + 3d "
                f"(have {self.bed_depth:.4f} m, need "
                f"{self.penetrate_over_r * self.R + 3.0 * self.d:.4f} m)")
        if self.box_width / 2.0 < 3.0 * self.R:
            raise ValueError("wall clearance < 3R: Janssen confinement dominates")
        if self.penetration_speed <= 0.0:
            raise ValueError("penetration_speed must be positive")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class QuasiStaticPenetration:
    """Velocity-controlled, overdamped sphere-penetration case."""

    def __init__(self, config: Optional[PenetrationConfig] = None) -> None:
        self.cfg = config or PenetrationConfig()
        self.cfg.validate()
        self.bed_info: Dict[str, float] = {}

    # -- helpers ----------------------------------------------------------
    def _walls(self, width: float) -> List[SphereWall]:
        return [
            SphereWall(np.array([1.0, 0.0, 0.0]), 0.0, label="x0"),
            SphereWall(np.array([-1.0, 0.0, 0.0]), -width, label="xL"),
            SphereWall(np.array([0.0, 1.0, 0.0]), 0.0, label="y0"),
            SphereWall(np.array([0.0, -1.0, 0.0]), -width, label="yL"),
            SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, label="floor"),
        ]

    def _material(self) -> ContactMaterial:
        cfg = self.cfg
        return ContactMaterial(e_star=cfg.e_star, g_star=cfg.g_star, mu=cfg.mu_grain,
                               mu_rolling=cfg.mu_rolling, restitution=cfg.restitution)

    def _engine_config(self) -> SpherePackConfig:
        cfg = self.cfg
        return SpherePackConfig(material=self._material(),
                                damping_global=cfg.damping,
                                damping_rotational=cfg.damping_rotational,
                                spin_slip=True)

    def build_bed(self, verbose: bool = False) -> Dict[str, float]:
        """HCP bed + gravitational settling; returns the *measured* bed state."""
        cfg = self.cfg
        r = cfg.r
        width = cfg.box_width
        pos = hcp_positions_in_box(r, width, width, cfg.bed_depth,
                                   packing_gap=cfg.packing_gap,
                                   jitter=cfg.settle_jitter_over_d * cfg.d, seed=cfg.seed,
                                   margin=0.2 * r)
        n = len(pos)
        rad = np.full(n, r)
        settle_cfg = SpherePackConfig(
            material=ContactMaterial(e_star=cfg.e_star, g_star=cfg.g_star,
                                     mu=cfg.mu_grain, mu_rolling=cfg.mu_rolling,
                                     restitution=cfg.settle_restitution),
            damping_global=cfg.settle_damping,
            damping_rotational=cfg.settle_damping, spin_slip=True)
        self.bed = SpherePack(pos, rad, cfg.density_particle, self._walls(width), settle_cfg)
        t0 = time.time()
        self.bed.run(cfg.settle_steps)
        self.bed.contact_forces()          # populate n_contacts for the report
        settle_ke = kinetic_energy_free(self.bed)
        zz = np.sort(self.bed.pos[:, 2])
        top = zz[max(0, int(0.95 * n)):]
        z_surface = float(np.mean(top)) - r
        phi = float(n * 4.0 / 3.0 * math.pi * r ** 3 / (width * width * z_surface))
        self.z_surface = z_surface
        self.bed_info = dict(
            n_grains=n, box_width=width, bed_depth_initial=cfg.bed_depth,
            z_surface=z_surface, packing_fraction=phi,
            bulk_density=phi * cfg.density_particle,
            density_particle=cfg.density_particle,
            settle_ke_J=settle_ke, settle_s=time.time() - t0,
            settle_steps=cfg.settle_steps, time_step=float(self.bed.dt),
            settle_contacts=int(self.bed.n_contacts),
            max_overlap_ratio_settle=float(self.bed.max_penetration_ratio),
            settle_restitution=cfg.settle_restitution,
            settle_damping=cfg.settle_damping,
        )
        if verbose:
            print(f"[bed] N={n} L={width*1e3:.1f}mm phi={phi:.4f} "
                  f"rho_bulk={phi*cfg.density_particle:.0f} kg/m3 "
                  f"KE={settle_ke:.2e} J dt={self.bed.dt:.3e} s", flush=True)
        return self.bed_info

    # -- main -------------------------------------------------------------
    def run(self, verbose: bool = False) -> Dict[str, object]:
        cfg = self.cfg
        if not hasattr(self, "bed"):
            self.build_bed(verbose=verbose)
        d, R = cfg.d, cfg.R

        # --- insert the indenter just above the highest grain -------------
        # the *sphere* of the highest grain, not just its centre, must clear the
        # indenter bottom: a surface grain one radius off axis otherwise starts
        # the run 1.8 mm (0.4 d) inside the indenter (measured: F = 4 kN at
        # step 1, i.e. exactly the m15367 failure mode re-entering through the
        # initial condition)
        z_top = float(self.bed.pos[:, 2].max()) + cfg.r
        z0 = z_top + R + 0.05 * d
        width = cfg.box_width
        pos2 = np.vstack([[0.5 * width, 0.5 * width, z0], self.bed.pos])
        rad2 = np.concatenate([[R], self.bed.rad])
        fixed2 = np.zeros(len(pos2), dtype=bool)
        fixed2[0] = True
        sp = SpherePack(pos2, rad2, cfg.density_particle, self._walls(width),
                        self._engine_config(), fixed=fixed2)
        # loading starts from rest: the settling transient of the bed is not part
        # of the penetration history
        sp.vel[:] = 0.0
        sp.omega[:] = 0.0
        v_pen = cfg.penetration_speed
        sp.vel[0, 2] = -v_pen            # kinematic BC: the integrator moves it
        dt = float(sp.dt)
        z_bottom0 = z0 - R
        target_depth = cfg.penetrate_over_r * R
        # The indenter starts above the free surface; initial_depth is negative.
        # Total travel = target_depth - initial_depth = target_depth + |initial_depth|.
        initial_depth = float(self.z_surface - z_bottom0)
        total_travel = target_depth - initial_depth  # = target_depth + |initial_depth|
        n_block = max(1, int(cfg.sample_every_steps))
        max_steps = int(math.ceil(total_travel / (v_pen * dt) * 1.2)) + 2000

        rec: Dict[str, List[float]] = dict(step=[], z=[], F=[], ke=[], ke_drive=[],
                                           n_contacts=[])
        ke_max = 0.0
        ke_drive_max = 0.0
        overlap_max = 0.0
        aborted = ""
        t0 = time.time()
        step = 0
        next_report = 0
        while True:
            sp.run(n_block)
            step += n_block
            # depth of the indenter bottom below the *free surface* (the same
            # convention as the reference law: z is the wetted-cap depth).  An
            # earlier version measured it from the start position, which offsets
            # z by exactly R and therefore biases dF/dz.
            depth = float(self.z_surface - (sp.pos[0, 2] - R))
            force, _ = sp.contact_forces()
            fz = float(force[0, 2])
            ke = kinetic_energy_free(sp)
            ke_drive = kinetic_energy_drive(sp)
            overlap = float(sp.max_penetration_ratio)
            ke_max = max(ke_max, ke)
            ke_drive_max = max(ke_drive_max, ke_drive)
            overlap_max = max(overlap_max, overlap)
            if depth > 0.0:
                rec["step"].append(step)
                rec["z"].append(depth)
                rec["F"].append(fz)
                rec["ke"].append(ke)
                rec["ke_drive"].append(ke_drive)
                rec["n_contacts"].append(int(sp.n_contacts))
            if verbose and step >= next_report:
                print(f"  step {step:7d} z/R={depth / R:5.2f} F={fz:8.4f} N "
                      f"KE={ke:9.3e} J overlap={overlap:.3f}", flush=True)
                next_report = step + max(n_block, 500)
            if overlap > cfg.abort_overlap_ratio:
                aborted = f"contact overlap ratio {overlap:.3f} > {cfg.abort_overlap_ratio}"
                break
            if depth >= target_depth:
                break
            if step >= max_steps:
                aborted = "step budget exhausted before the target depth"
                break
        wall = time.time() - t0

        z_arr = np.asarray(rec["z"], dtype=float)
        F_arr = np.asarray(rec["F"], dtype=float)
        ke_arr = np.asarray(rec["ke"], dtype=float)
        metrics, acceptance, fit = evaluate_profile(
            z_arr, F_arr, radius=R, mu=cfg.mu_grain,
            rho_bulk=self.bed_info["bulk_density"],
            fit_from=cfg.fit_from_over_r * R, fit_to=cfg.fit_to * R,
            smoothing=cfg.smoothing_over_r * R, ke=ke_arr)
        metrics["kinetic_energy_max_J"] = float(
            max(ke_max, metrics.get("kinetic_energy_max_sampled_J", 0.0)))
        metrics["kinetic_energy_max_including_drive_J"] = float(ke_drive_max)
        metrics["max_overlap_ratio"] = float(overlap_max)
        metrics["sample_count"] = int(len(z_arr))
        metrics["speed_ratio_v_over_sqrt_gR"] = float(v_pen / math.sqrt(G * R))
        metrics["contact_dashpot"] = "off (restitution = 1)"
        acceptance["max_overlap_ratio"] = bool(overlap_max <= ACCEPTANCE["max_overlap_ratio"])
        acceptance["no_injection_abort"] = bool(aborted == "")
        failed = [k for k, ok in acceptance.items() if not ok]

        profile = {k: [float(x) for x in v] for k, v in rec.items()}
        return dict(
            case="CASE-PENETRATION (quasi-static sphere penetration)",
            label=cfg.label,
            status="PASS" if not failed else "FAIL",
            failed_criteria=failed,
            aborted=aborted,
            reference=REFERENCE,
            acceptance_thresholds=ACCEPTANCE,
            config=dict(asdict(cfg)),
            geometry=dict(
                grain_diameter=d, grain_radius=cfg.r, intruder_radius=R,
                radius_over_d=cfg.intruder_radius_over_d,
                wall_half_width_over_R=cfg.wall_half_width_over_R,
                box_width=width, bed_depth=cfg.bed_depth,
                penetration_depth=float(z_arr[-1]) if len(z_arr) else 0.0,
                time_step=dt),
            bed=self.bed_info,
            metrics=metrics,
            fit=fit,
            acceptance=acceptance,
            profile=profile,
            timing=dict(loading_s=wall, steps=step,
                        s_per_step=wall / max(step, 1)),
        )


# ---------------------------------------------------------------------------
# Profile analysis (pure functions: unit-testable without any simulation)
# ---------------------------------------------------------------------------

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centred boxcar average with edge truncation (no padding artefacts)."""
    y = np.asarray(y, dtype=float)
    window = max(1, int(window))
    if window < 2 or len(y) < 2:
        return y.copy()
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(len(y))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(y), idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def evaluate_profile(z: np.ndarray, F: np.ndarray, radius: float, mu: float,
                     rho_bulk: float, fit_from: float, fit_to: float,
                     smoothing: float, ke: Optional[np.ndarray] = None,
                     alpha_ref: float = REFERENCE["alpha"]):
    """Fit Eq. (3d) to a force-depth profile and build the acceptance verdict.

    Two independent estimators of ``alpha`` are reported:

    * ``alpha_fit`` -- **constrained** one-parameter fit of the Brzinski shape,
      ``F = A (z - R/3)`` with ``A = alpha mu rho_bulk g pi R^2``.  The intercept
      is fixed by the theory (``-A R/3``), which removes the ambiguity of a free
      linear fit on a short window.
    * ``alpha_fit_free`` -- free linear ``F = s z + c`` (the estimator used by
      m15367), kept for continuity; its intercept is reported so the theory's
      ``c = -s R/3`` can be checked.

    ``force_variation_coefficient`` is the residual rms of the *smoothed* curve
    about the constrained fit divided by the mean measured force in the window:
    it measures jitter, not the (physical, wanted) depth trend.  The raw
    (unsmoothed) counterpart is reported as ``force_variation_coefficient_raw``.
    """
    z = np.asarray(z, dtype=float)
    F = np.asarray(F, dtype=float)
    out: Dict[str, object] = {}
    ok = np.isfinite(z) & np.isfinite(F)
    z, F = z[ok], F[ok]
    if ke is not None:
        ke = np.asarray(ke, dtype=float)[ok]
        out["kinetic_energy_max_sampled_J"] = float(np.max(ke)) if len(ke) else float("nan")
        out["kinetic_energy_p95_J"] = float(np.percentile(ke, 95)) if len(ke) else float("nan")
    dz = np.abs(np.diff(z))
    dz_mean = float(np.median(dz)) if len(dz) else 0.0
    win = max(1, int(round(smoothing / dz_mean))) if dz_mean > 0 else 1
    Fs = moving_average(F, win)
    sel = (z >= fit_from) & (z <= fit_to)
    out["window"] = dict(z_from=float(fit_from), z_to=float(fit_to),
                         smoothing_window_samples=int(win),
                         n_samples=int(np.count_nonzero(sel)))
    if np.count_nonzero(sel) < 5 or float(np.ptp(z[sel])) <= 0.0:
        out.update(degenerate=True, alpha_fit=float("nan"),
                   alpha_fit_free=float("nan"),
                   force_variation_coefficient=float("nan"),
                   penetration_force_error=float("nan"))
        return out, {"penetration_force_error": False,
                     "force_variation_coefficient": False}, out
    zs, Fw = z[sel], Fs[sel]
    a = zs - radius / 3.0
    A = float(np.dot(a, Fw) / np.dot(a, a))               # deep-branch slope
    # primary estimator: scale the *exact* piecewise shape of Eq. (3c)/(3d),
    # alpha = argmin ||F - alpha mu rho g shape(z)|| -- unbiased for a window that
    # straddles z = R, unlike the deep-branch linear form (2.9 % low on
    # [0.5R, 1.6R], see tests).
    S = mu * rho_bulk * G * brzinski_shape(zs, radius)
    alpha_shape = float(np.dot(S, Fw) / np.dot(S, S))
    resid_shape = Fw - alpha_shape * S
    resid = Fw - A * a
    mean_F = float(np.mean(Fw))
    cv = float(np.sqrt(np.mean(resid ** 2)) / max(abs(mean_F), 1e-30))
    slope, intercept = np.polyfit(zs, Fw, 1)
    diff_free = F[sel] - np.polyval(np.polyfit(zs, F[sel], 1), z[sel])
    cv_raw = float(np.sqrt(np.mean(diff_free ** 2)) / max(abs(float(np.mean(F[sel]))), 1e-30))
    alpha_fit = alpha_from_slope(A, radius, mu, rho_bulk)
    alpha_free = alpha_from_slope(float(slope), radius, mu, rho_bulk)
    f_ref = brzinski_force(zs, radius, mu=mu, rho_bulk=rho_bulk, alpha=alpha_ref)
    force_error = float(np.mean(np.abs(Fw - f_ref)) / max(float(np.mean(f_ref)), 1e-30))
    out.update(
        degenerate=False,
        alpha_fit=float(alpha_shape),
        alpha_fit_deep=float(alpha_fit),
        force_variation_coefficient_shape=float(np.sqrt(np.mean(resid_shape ** 2))
                                                / max(abs(mean_F), 1e-30)),
        alpha_fit_free=float(alpha_free),
        alpha_fit_free_intercept_N=float(intercept),
        expected_intercept_N=float(-A * radius / 3.0),
        penetration_force_error=float(abs(alpha_shape - alpha_ref) / alpha_ref),
        penetration_force_error_free=float(abs(alpha_free - alpha_ref) / alpha_ref),
        force_variation_coefficient=float(cv),
        force_variation_coefficient_raw=float(cv_raw),
        mean_force_N=float(mean_F),
        fit_rms_N=float(np.sqrt(np.mean(resid ** 2))),
        force_error_vs_reference=float(force_error),
        reference_force_mean_N=float(np.mean(f_ref)),
        reference_alpha=float(alpha_ref),
        alpha_ref_band=[alpha_ref - REFERENCE["alpha_err"], alpha_ref + REFERENCE["alpha_err"]],
        in_reference_band=bool(abs(alpha_shape - alpha_ref) <= REFERENCE["alpha_err"]),
    )
    acc = dict(
        kinetic_energy=bool(out.get("kinetic_energy_max_sampled_J", 0.0)
                            <= ACCEPTANCE["kinetic_energy_max_J"]),
        force_variation_coefficient=bool(cv <= ACCEPTANCE["force_variation_coefficient"]),
        penetration_force_error=bool(abs(alpha_shape - alpha_ref) / alpha_ref
                                     <= ACCEPTANCE["penetration_force_error"]),
    )
    return out, acc, out


# ---------------------------------------------------------------------------
# Negative control: the m15367 teleport protocol, on the same bed
# ---------------------------------------------------------------------------

class TeleportControl(QuasiStaticPenetration):
    """m15367-era loading path: teleport the indenter, then relax.

    Kept as a *falsification baseline*: on the identical bed it must reproduce
    the energy blow-up that made m15367 an honest FAIL, otherwise the diagnosis
    ("the loading path, not the bed, injects the energy") is wrong.
    """

    def __init__(self, config: Optional[PenetrationConfig] = None,
                 increment_over_d: float = 0.125, relax_steps: int = 100) -> None:
        super().__init__(config)
        self.increment_over_d = increment_over_d
        self.relax_steps = relax_steps

    def run(self, verbose: bool = False, max_increments: int = 64) -> Dict[str, object]:  # type: ignore[override]
        cfg = self.cfg
        if not hasattr(self, "bed"):
            self.build_bed(verbose=verbose)
        d, R = cfg.d, cfg.R
        z0 = float(self.bed.pos[:, 2].max()) + cfg.r + R + 0.05 * d
        width = cfg.box_width
        pos2 = np.vstack([[0.5 * width, 0.5 * width, z0], self.bed.pos])
        rad2 = np.concatenate([[R], self.bed.rad])
        fixed2 = np.zeros(len(pos2), dtype=bool)
        fixed2[0] = True
        sp = SpherePack(pos2, rad2, cfg.density_particle, self._walls(width),
                        self._engine_config(), fixed=fixed2)
        sp.vel[:] = 0.0
        sp.omega[:] = 0.0
        ke0 = kinetic_energy_free(sp)
        z_bottom0 = z0 - R
        dz = self.increment_over_d * d
        rec: Dict[str, List[float]] = dict(step=[], z=[], F=[], ke=[], n_contacts=[])
        ke_max = 0.0
        for k in range(max_increments):
            sp.pos[0, 2] -= dz                # <-- the teleport
            sp._build_neighbour_list()
            sp.run(self.relax_steps)
            force, _ = sp.contact_forces()
            dep = float(self.z_surface - (sp.pos[0, 2] - R))
            ke = kinetic_energy_free(sp)
            ke_max = max(ke_max, ke)
            rec["step"].append((k + 1) * self.relax_steps)
            rec["z"].append(dep)
            rec["F"].append(float(force[0, 2]))
            rec["ke"].append(ke)
            rec["n_contacts"].append(int(sp.n_contacts))
            if verbose:
                print(f"  [teleport] inc {k+1:3d} z/R={dep / R:5.2f} "
                      f"F={float(force[0, 2]):9.3f} N KE={ke:9.3e} J", flush=True)
            if dep >= cfg.penetrate_over_r * R:
                break
        profile = {k: [float(x) for x in v] for k, v in rec.items()}
        F = np.asarray(profile["F"], float)
        return dict(
            case="CASE-PENETRATION negative control (teleport loading path)",
            label="teleport_control",
            status="expected_energy_injection",
            geometry=dict(intruder_radius=R, grain_diameter=d, box_width=width,
                          increment=dz, relax_steps=self.relax_steps,
                          penetration_depth=float(profile["z"][-1]) if profile["z"] else 0.0),
            bed=self.bed_info,
            metrics=dict(
                kinetic_energy_max_J=float(ke_max),
                kinetic_energy_initial_J=float(ke0),
                force_max_N=float(np.max(F)) if len(F) else float("nan"),
                force_std_N=float(np.std(F)) if len(F) else float("nan"),
                max_overlap_ratio=float(sp.max_penetration_ratio)),
            profile=profile,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def preset(name: str) -> PenetrationConfig:
    """Named configurations used by the case pool (reproducible by name)."""
    table = {
        "base": PenetrationConfig(label="base"),
        "rd2": PenetrationConfig(intruder_radius_over_d=2.0, bed_depth_over_d=7.0,
                                 label="rd2"),
        "rd4": PenetrationConfig(intruder_radius_over_d=4.0, bed_depth_over_d=10.0,
                                 label="rd4"),
        "slow": PenetrationConfig(penetration_speed=0.25, label="slow"),
        "rate2x": PenetrationConfig(penetration_speed=1.0, label="rate2x"),
        "damp2000": PenetrationConfig(damping=2000.0, damping_rotational=2000.0,
                                      label="damp2000"),
        "dashpot": PenetrationConfig(restitution=0.4, label="dashpot"),
    }
    if name not in table:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(table)}")
    return table[name]


def make_figures(result: Dict[str, object], out_dir: str) -> List[str]:
    """Publication figures: force-depth vs Eq. (3c)/(3d), and the energy trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    prof = result["profile"]                       # type: ignore[index]
    z = np.asarray(prof["z"], float)
    F = np.asarray(prof["F"], float)
    ke = np.asarray(prof["ke"], float)
    R = float(result["geometry"]["intruder_radius"])      # type: ignore[index]
    rho_bulk = float(result["bed"]["bulk_density"])       # type: ignore[index]
    mu = float(result["config"]["mu_grain"])              # type: ignore[index]
    fit = result["fit"]                                   # type: ignore[index]
    cfg = result["config"]                                # type: ignore[index]
    paths: List[str] = []
    if len(z) < 5:
        return paths

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dzm = float(np.median(np.abs(np.diff(z))))
    win = max(1, int(round(float(cfg["smoothing_over_r"]) * R / dzm))) if dzm > 0 else 1
    ax[0].plot(z / R, F, lw=0.6, color="0.65", label="DEM (raw)")
    ax[0].plot(z / R, moving_average(F, win), lw=1.7, color="C0",
               label=f"DEM (moving avg, {float(cfg['smoothing_over_r']):.2f} R)")
    zz = np.linspace(0.0, float(np.max(z)) / R, 400)
    ax[0].plot(zz, brzinski_force(zz * R, R, mu=mu, rho_bulk=rho_bulk), "k--", lw=1.4,
               label=r"Brzinski Eq.(3c)/(3d), $\alpha=35\pm5$")
    ax[0].axvspan(float(cfg["fit_from_over_r"]), float(cfg["fit_to_over_r"] or cfg["penetrate_over_r"]),
                  color="C0", alpha=0.06, label="fit window")
    ax[0].set_xlabel("z / R")
    ax[0].set_ylabel("penetration resistance F (N)")
    ax[0].set_title(f"F(z):  alpha_fit = {float(fit['alpha_fit']):.1f}  (ref 35 $\\pm$ 5)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(z / R, np.maximum(ke, 1e-30), lw=0.8, color="C3")
    ax[1].axhline(ACCEPTANCE["kinetic_energy_max_J"], color="k", ls=":",
                  label="acceptance 0.1 J")
    ax[1].set_xlabel("z / R")
    ax[1].set_ylabel("system kinetic energy (J)")
    ax[1].set_title(f"KE$_{{max}}$ = {float(result['metrics']['kinetic_energy_max_J']):.2e} J")  # type: ignore[index]
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_force_depth_and_energy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="quasi-static sphere penetration case")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--preset", default="base")
        p.add_argument("--speed", type=float, default=None,
                       help="override penetration_speed (m/s)")
        p.add_argument("--damping", type=float, default=None)
        p.add_argument("--rd", type=float, default=None,
                       help="override intruder_radius_over_d")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--out", required=True)
        p.add_argument("--figures", action="store_true")

    add_common(sub.add_parser("run", help="kinematic (velocity BC) loading"))
    add_common(sub.add_parser("control", help="m15367 teleport negative control"))
    pp = sub.add_parser("plot", help="figures from an existing result.json")
    pp.add_argument("--input", required=True)
    pp.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.cmd == "plot":
        with open(args.input) as fh:
            result = json.load(fh)
        paths = make_figures(result, args.out)
        print(json.dumps({"figures": paths}, indent=2))
        return 0

    cfg = preset(args.preset)
    if args.speed is not None:
        cfg.penetration_speed = float(args.speed)
        cfg.label = f"{cfg.label}_v{args.speed:g}"
    if args.damping is not None:
        cfg.damping = float(args.damping)
        cfg.damping_rotational = float(args.damping)
    if args.rd is not None:
        cfg.intruder_radius_over_d = float(args.rd)
        cfg.bed_depth_over_d = max(cfg.bed_depth_over_d, cfg.penetrate_over_r * args.rd + 3.0)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    t0 = time.time()
    if args.cmd == "control":
        result = TeleportControl(cfg).run(verbose=True)
    else:
        result = QuasiStaticPenetration(cfg).run(verbose=True)
    result["wall_clock_s"] = time.time() - t0

    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    if args.figures:
        result["figures"] = make_figures(result, args.out)
        with open(os.path.join(args.out, "figures.json"), "w") as fh:
            json.dump({"figures": result["figures"]}, fh, indent=2)
    summary = dict(label=result.get("label"), status=result["status"],
                   failed=result.get("failed_criteria"),
                   metrics={k: v for k, v in result["metrics"].items()
                            if isinstance(v, (int, float))},
                   acceptance=result["acceptance"],
                   timing=result.get("timing"))
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "PASS" else (0 if args.cmd == "control" else 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
