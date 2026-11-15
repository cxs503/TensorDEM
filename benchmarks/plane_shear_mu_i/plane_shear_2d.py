"""CASE-DEM-PLANE-SHEAR-MUIDI round 24: a *2-D disk* plane-shear device.

Why this device exists
----------------------
Rounds 1-23 measured ``mu(I)`` on a **3-D sphere** cell (294 spheres, 0.12 x 0.05
x 0.06 m) and judged it against the **2-D disk** law of da Cruz et al. 2005
(``mu = 0.22 + 1.00 I``, ``phi = 0.81 - 0.30 I``, valid for ``I <= 0.2``).  Round
23 closed the seed question (the +29 % offset at ``I = 0.03`` is not one
realisation, CV up to 21 %) and named the only surviving candidate mechanism for
the offset: **dimension**.  A 3-D packing has a larger quasi-static ``mu_s`` than
a 2-D one at the same ``mu_p``, so part of the 16-57 % gap could be the
dimensionality of the measuring device rather than an error of the platform.

Round 23 recommendation (2) was "if a 3-D reference is to be used, archive a 3-D
``mu(I)`` paper".  This module takes the cheaper, falsifiable branch of the same
question: **build the device the law was measured on**.  A monodisperse 2-D disk
cell with rough (grain-glued) walls, constant confining stress and a prescribed
shear rate is dimensionally commensurable with the archived law - both ``mu`` and
``phi`` become the same quantities the paper reports (area fraction, not a 3-D
solid fraction with a slab-volume definition).

Device (all SI, ``thickness = d`` so that stress is a true Pa and the 2-D
elastic/inertial scaling is inherited from the 3-D engine unchanged)
---------------------------------------------------------------------
* free monodisperse disks, ``d = 10 mm``; cell ``W x h`` periodic in ``x``;
* rough walls = two **pinned** rows of disks of the same diameter, spaced
  ``d (1 + 1e-3)`` so a free disk can never squeeze between wall grains (that is
  what makes the wall rough, and it removes wall slip as a free parameter);
* bottom row moves at ``+V/2``, top row at ``-V/2`` in ``x`` -> mean shear rate;
* the top row is a **servo** in ``y``: it moves to hold the measured wall normal
  stress at ``P`` (constant-stress, as da Cruz impose);
* contacts are the repository's own 2-D engine (``tensordem.granular2d``,
  linear/Hertz normal + Cundall-Strack tangential + rolling resistance, one-sided
  history per contact key), so the contact physics is not re-implemented here;
* the pinned rows are integrated out: :meth:`RoughWallShearCell.advance` integrates
  the free degrees of freedom only and re-imposes the wall kinematics, which is
  the exact discrete counterpart of "glued wall grains".

Measured ordinates
------------------
* ``mu_wall``: ``|sum Fx| / |sum Fy|`` over both pinned rows (wall traction);
* ``mu_bulk``: from the Love-Weber sum over **all** contacts, normalised by the
  control volume ``W x h_eff x t``; because the wall disks are particles, the
  wall load is inside the sum, so ``-sigma_yy / P -> 1`` is an analytic
  calibration of the estimator that uses no literature constant (the round-8
  identity, now on a device where the wall load is not lost);
* ``phi_gap``: free-disk area over ``W x h_eff`` (``h_eff`` = wall-grain
  surface-to-surface distance).  **Pre-registered** as the primary ``phi`` (see
  ``prereg_r24.json``): the pinned rows carry no free volume, so the control
  volume of the sheared material is surface-to-surface.  ``phi_cell``
  (centre-to-centre, i.e. the wall grains' half layers included) is reported as
  the secondary definition so that the choice can be audited, not assumed;
* ``I_bulk``: ``gammadot d / sqrt(P/rho)`` with the *measured* shear rate
  (``2 V / h_eff``) and the *measured* confining stress.

Usage::

    python benchmarks/plane_shear_mu_i/plane_shear_2d.py --I 0.03 --seed 0
    python benchmarks/plane_shear_mu_i/plane_shear_2d.py --I 0.03 --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.granular2d.engine import DiskConfig, DiskSystem  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Case definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlaneShear2DCase:
    """A 2-D disk plane-shear cell held at constant confining stress."""

    I_target: float = 0.03
    n_particles: int = 1220
    radius: float = 5.0e-3
    width: float = 0.40
    gap: float = 0.30
    normal_stress: float = 1.0e4
    density: float = 2500.0
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    zeta: float = 0.2
    mu_p: float = 0.4          # da Cruz 2005 use mu_p = 0.4 for the 2-D disks
    mu_rolling: float = 0.0    # disks, no rolling resistance in the reference setup
    seed: int = 0
    settle_time_units: float = 60.0  # compaction time in d/sqrt(P/rho) units
    shear_settle_strain: float = 0.5   # shear strain before the measurement window
    strain_measure: float = 1.0        # width of the measurement window
    servo_gain: float = 0.05
    phi_init: float = 0.80
    sample_every: int = 40

    # -- derived quantities -------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def thickness(self) -> float:
        return self.diameter

    @property
    def velocity_scale(self) -> float:
        """``sqrt(P/rho)``: the only velocity the inertial number can be built on."""
        return math.sqrt(self.normal_stress / self.density)

    def shear_rate_for(self, inertial_number: float) -> float:
        """``gammadot`` that realises ``I = gammadot d / sqrt(P/rho)``."""
        return inertial_number * self.velocity_scale / self.diameter

    @property
    def gammadot(self) -> float:
        return self.shear_rate_for(self.I_target)

    @property
    def wall_speed(self) -> float:
        """Top/bottom row speed: ``2 V / h = gammadot`` at the initial gap."""
        return 0.5 * self.gammadot * self.gap

    def config(self) -> DiskConfig:
        return DiskConfig(kn=self.kn, kt_ratio=self.kt_ratio, zeta=self.zeta,
                          mu=self.mu_p, mu_rolling=self.mu_rolling,
                          density=self.density, thickness=self.thickness,
                          gravity=(0.0, 0.0), dt=None, skin=0.35 * self.radius)

    def validate(self) -> None:
        if not (self.I_target > 0.0 and math.isfinite(self.I_target)):
            raise ValueError("I_target must be finite and positive")
        if self.gap < 3.0 * self.diameter:
            raise ValueError("gap must hold at least 3 diameters")
        if self.width < 5.0 * self.diameter:
            raise ValueError("width must hold at least 5 diameters")
        if not (0.0 < self.phi_init < 0.84):
            raise ValueError("phi_init must be below the 2-D close packing 0.84")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.mu_p <= 0.0 or self.kn <= 0.0:
            raise ValueError("mu_p and kn must be positive")


# --------------------------------------------------------------------------- #
# 2. Initial state: pinned rows + lattice packing with a seeded jitter
# --------------------------------------------------------------------------- #
def _wall_row(case: PlaneShear2DCase, y: float, offset: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """A touching row of disks at height ``y``; spacing keeps them just in contact."""
    spacing = case.diameter * (1.0 + 1.0e-3)
    n = int(math.floor(case.width / spacing))
    if n < 4:
        raise ValueError("wall row needs at least 4 grains")
    x = (np.arange(n) + offset) * spacing
    x = np.mod(x, case.width)
    pos = np.stack([x, np.full(n, y)], axis=1)
    rad = np.full(n, case.radius)
    return pos, rad


def _lattice_packing(case: PlaneShear2DCase, y_lo: float, y_hi: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Triangular-ish lattice at ``phi_init`` with a seeded sub-radius jitter."""
    rng = np.random.default_rng(case.seed)
    spacing = case.diameter * (1.0 + 1.0e-3)
    dy = spacing * math.sqrt(3.0) / 2.0
    n_y = max(1, int(math.floor((y_hi - y_lo) / dy)))
    n_x = max(1, int(math.floor(case.width / spacing)))
    pts = []
    for k in range(n_y):
        y = y_lo + (k + 0.5) * (y_hi - y_lo) / n_y
        shift = 0.5 * spacing if k % 2 else 0.0
        for m in range(n_x):
            x = m * spacing + shift + 0.5 * spacing
            pts.append((x, y))
    pos = np.asarray(pts, dtype=np.float64)
    pos[:, 0] = np.mod(pos[:, 0], case.width)
    if len(pos) < case.n_particles:
        raise ValueError(f"lattice provides {len(pos)} sites < n_particles "
                         f"{case.n_particles}; enlarge the gap/width")
    idx = np.sort(rng.permutation(len(pos))[: case.n_particles])
    pos = pos[idx]
    pos += rng.uniform(-0.05, 0.05, size=pos.shape) * case.diameter
    pos[:, 1] = np.clip(pos[:, 1], y_lo, y_hi)
    rad = np.full(len(pos), case.radius)
    return pos, rad


# --------------------------------------------------------------------------- #
# 3. The cell
# --------------------------------------------------------------------------- #
class RoughWallShearCell:
    """Free disks sheared between two pinned grain rows, top row servoed in ``y``."""

    def __init__(self, case: PlaneShear2DCase) -> None:
        case.validate()
        self.case = case
        r = case.radius
        bottom, rad_b = _wall_row(case, 1.0 * r, 0.0)
        top, rad_t = _wall_row(case, 1.0 * r + case.gap + 2.0 * r, 0.5)
        free_pos, rad_f = _lattice_packing(case, 3.0 * r, case.gap + r)
        # free disks live strictly between the two rows' surfaces
        y_lo = bottom[0, 1] + 2.0 * r
        y_hi = top[0, 1] - 2.0 * r
        free_pos[:, 1] = np.clip(free_pos[:, 1], y_lo, y_hi)
        pos = np.vstack([free_pos, bottom, top])
        radii = np.concatenate([rad_f, rad_b, rad_t])
        n_free = len(free_pos)
        self.n_free = n_free
        self.n_wall = len(bottom)
        self.bottom_idx = np.arange(n_free, n_free + len(bottom))
        self.top_idx = np.arange(n_free + len(bottom), len(radii))
        self.free_idx = np.arange(n_free)
        self.pinned = np.zeros(len(radii), dtype=bool)
        self.pinned[self.bottom_idx] = True
        self.pinned[self.top_idx] = True

        self.system = DiskSystem(pos, radii, case.config(), periodic_x=case.width)
        self.dt = float(self.system.dt)

        self.y_bottom = float(bottom[0, 1])
        self.y_top = float(top[0, 1])
        self.v_wall = case.wall_speed
        self.pinned_pos = self.system.pos[self.pinned].copy()
        self.time = 0.0
        self.steps = 0
        self._set_wall_kinematics(0.0)

    # -- wall bookkeeping ---------------------------------------------------
    def _set_wall_kinematics(self, servo_vy: float) -> None:
        pos = self.system.pos
        pos[self.bottom_idx, 1] = self.y_bottom
        pos[self.top_idx, 1] = self.y_top
        self.system.vel[self.bottom_idx] = (self.v_wall, 0.0)
        self.system.vel[self.top_idx] = (-self.v_wall, servo_vy)
        self.system.omega[self.bottom_idx] = 0.0
        self.system.omega[self.top_idx] = 0.0

    @property
    def gap_eff(self) -> float:
        """Wall-grain surface-to-surface distance."""
        return self.y_top - self.y_bottom - 2.0 * self.case.radius

    @property
    def shear_rate(self) -> float:
        return 2.0 * self.v_wall / self.gap_eff

    # -- one block of steps -------------------------------------------------
    def advance(self, n_steps: int, *, servo: bool = True) -> None:
        case = self.case
        sysm = self.system
        free = self.free_idx
        mass, inertia = sysm.mass, sysm.inertia
        area_wall = case.width * case.thickness
        v_scale = case.velocity_scale
        for _ in range(n_steps):
            force, torque, _ = sysm._evaluate()
            sysm.vel[free] += self.dt * (force[free] / mass[free, None])
            sysm.omega[free] += self.dt * (torque[free] / inertia[free])
            sysm.pos[free] += self.dt * sysm.vel[free]
            sysm.theta[free] += self.dt * sysm.omega[free]
            sysm.pos[free, 0] = np.mod(sysm.pos[free, 0], case.width)
            sysm.time += self.dt
            self.steps += 1
            servo_vy = 0.0
            if servo:
                p_meas = abs(float(np.sum(force[self.top_idx, 1]))) / area_wall
                servo_vy = case.servo_gain * (p_meas / case.normal_stress - 1.0) * v_scale
                # bounded so that a transient cannot blow the cell open or shut
                servo_vy = float(np.clip(servo_vy, -0.05 * v_scale, 0.05 * v_scale))
                self.y_top += servo_vy * self.dt
                lo = self.y_bottom + 2.0 * case.radius + 4.0 * case.diameter
                hi = self.y_bottom + 2.0 * case.radius + 4.0 * case.gap
                self.y_top = min(max(self.y_top, lo), hi)
            self._set_wall_kinematics(servo_vy)
        self.time += n_steps * self.dt

    # -- diagnostics --------------------------------------------------------
    def measure(self) -> dict[str, float]:
        case = self.case
        sysm = self.system
        force = sysm._contact_force
        area_wall = case.width * case.thickness
        # the two rows carry equal and opposite tractions (internal balance), so
        # each row is measured separately and the pair is averaged
        fy_top = float(np.sum(force[self.top_idx, 1]))
        fy_bot = float(np.sum(force[self.bottom_idx, 1]))
        fx_top = float(np.sum(force[self.top_idx, 0]))
        fx_bot = float(np.sum(force[self.bottom_idx, 0]))
        p_wall = 0.5 * (abs(fy_top) + abs(fy_bot)) / area_wall
        tau_wall = 0.5 * (abs(fx_top) + abs(fx_bot)) / area_wall

        # Love-Weber over every disk-disk contact (the pinned rows are disks, so
        # the wall load is inside the sum); branch through the periodic image.
        contacts = sysm.contacts
        sigma = np.zeros((2, 2))
        if contacts is not None and len(contacts.get("i", [])):
            f = np.asarray(contacts["f_on_i"], dtype=np.float64)
            i_idx = np.asarray(contacts["i"], dtype=np.int64)
            j_idx = np.asarray(contacts["j"], dtype=np.int64)
            branch = sysm.pos[j_idx] - sysm.pos[i_idx]
            branch[:, 0] -= case.width * np.round(branch[:, 0] / case.width)
            # traction convention: sigma_ab = (1/V) sum r_a f_b, force on i from j
            vol = case.width * self.gap_eff * case.thickness
            sigma = np.einsum("ma,mb->ab", branch, f) / vol

        p_bulk = -0.5 * float(sigma[0, 0] + sigma[1, 1])
        tau_bulk = 0.5 * float(abs(sigma[0, 1]) + abs(sigma[1, 0]))
        disk_area = self.n_free * math.pi * case.radius**2
        phi_gap = disk_area / (case.width * self.gap_eff)
        phi_cell = disk_area / (case.width * (self.gap_eff + 2.0 * case.radius))
        p = p_bulk if p_bulk > 0 else p_wall
        return {
            "time": self.time,
            "strain": self.time * self.shear_rate,
            "y_top": self.y_top,
            "gap_eff": self.gap_eff,
            "phi_gap": phi_gap,
            "phi_cell": phi_cell,
            "p_wall": p_wall,
            "p_bulk": p_bulk,
            "tau_wall": tau_wall,
            "tau_bulk": tau_bulk,
            "mu_wall": tau_wall / p_wall if p_wall > 0 else float("nan"),
            "mu_bulk": tau_bulk / p_bulk if p_bulk > 0 else float("nan"),
            "sigma_yy_over_P": -sigma[1, 1] / case.normal_stress,
            "I_bulk": self.shear_rate * case.diameter
                      / math.sqrt(max(p, 1e-12) / case.density),
            "ke": sysm.kinetic_energy(),
            "n_contacts": int(len(contacts["i"])) if contacts is not None else 0,
        }


# --------------------------------------------------------------------------- #
# 4. Prototype: shear a cell through the pre-registered strain budget
# --------------------------------------------------------------------------- #
def simulate(case: PlaneShear2DCase, *, progress: bool = False) -> dict:
    """Compact at constant ``P``, then shear and measure.

    Protocol (fixed before the runs, see ``prereg_r24.json``):

    1. **compaction** - the two rows are held still (``V = 0``) and the servo
       closes the gap until the confining stress reaches ``P``.  The duration is
       ``settle_time_units * d / sqrt(P/rho)``; the final part of it is required
       to be quiescent (see ``ke_settle`` in the returned summary) so that the
       shear starts from a static packing, not from a compaction wave.
    2. **shear transient** - the rows are driven at ``+-V`` and the cell is
       sheared through ``shear_settle_strain`` (this is where the packing finds
       its critical state at the imposed ``I``).
    3. **measurement window** - the next ``strain_measure`` of shear strain is
       sampled; means, sample sds and a first-half/second-half stationarity
       statistic are reported.
    """
    cell = RoughWallShearCell(case)
    v_wall = cell.v_wall
    t_inertial = case.diameter / case.velocity_scale
    settle_steps = int(math.ceil(case.settle_time_units * t_inertial / cell.dt))
    block = max(1, int(round(case.sample_every)))
    samples: list[dict[str, float]] = []
    t0 = time.time()

    # --- 1. compaction -----------------------------------------------------
    cell.v_wall = 0.0
    cell._set_wall_kinematics(0.0)
    done = 0
    while done < settle_steps:
        n = min(block, settle_steps - done)
        cell.advance(n)
        done += n
        m = cell.measure()
        m["phase"] = "settle"
        samples.append(m)
        if progress and len(samples) % 500 == 0:
            print(f"  settle {done}/{settle_steps} phi {m['phi_gap']:.3f} "
                  f"P_wall {m['p_wall']:.0f} P_bulk {m['p_bulk']:.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    settle_tail = [r for r in samples[-max(1, len(samples) // 5):]]
    ke_settle = float(np.median([r["ke"] for r in settle_tail]))
    n_settle = len(samples)

    # --- 2./3. shear transient + measurement window ------------------------
    cell.v_wall = v_wall
    cell._set_wall_kinematics(0.0)
    t_shear_start = cell.time
    gamma = 0.0
    while gamma < case.shear_settle_strain + case.strain_measure:
        cell.advance(block)
        m = cell.measure()
        gamma = (cell.time - t_shear_start) * cell.shear_rate
        m["strain"] = gamma
        m["phase"] = "measure" if gamma >= case.shear_settle_strain else "transient"
        samples.append(m)
        if progress and len(samples) % 500 == 0:
            print(f"  shear strain {gamma:.2f} phi {m['phi_gap']:.3f} "
                  f"mu_bulk {m['mu_bulk']:.3f} P_bulk {m['p_bulk']:.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    win = [r for r in samples if r["phase"] == "measure"]
    trans = [r for r in samples if r["phase"] == "transient"]
    first = win[: len(win) // 2]
    second = win[len(win) // 2:]

    def avg(key, rows):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if len(vals) else float("nan")

    def sd(key, rows):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")

    summary = {
        "case": asdict(case),
        "dt": cell.dt,
        "settle_steps": settle_steps,
        "measure_samples": len(win),
        "duration_s": time.time() - t0,
        "n_free": cell.n_free,
        "n_wall_per_row": cell.n_wall,
        "ke_settle": ke_settle,
        "p_compaction_end": avg("p_wall", settle_tail),
        "gap_eff_final": avg("gap_eff", win),
        "config_digest": cell.system.digest(),
        "seconds_per_step": (time.time() - t0) / max(1, len(samples) * block),
    }
    for key in ("mu_wall", "mu_bulk", "phi_gap", "phi_cell", "p_wall", "p_bulk",
                "sigma_yy_over_P", "I_bulk", "tau_wall", "tau_bulk"):
        summary[key] = avg(key, win)
        summary[key + "_sd"] = sd(key, win)
        summary[key + "_first_half"] = avg(key, first)
        summary[key + "_second_half"] = avg(key, second)
        summary[key + "_transient"] = avg(key, trans)
    summary["I_target"] = case.I_target
    summary["p_wall_over_P"] = summary["p_wall"] / case.normal_stress
    summary["shear_rate_measured"] = 2.0 * v_wall / summary["gap_eff_final"]

    def drift(key):
        a, b = summary[key + "_first_half"], summary[key + "_second_half"]
        s = summary[key + "_sd"]
        n_half = max(1, len(win) // 2)
        denom = s / math.sqrt(n_half) if (s and math.isfinite(s) and s > 0) else float("nan")
        return (b - a) / denom if math.isfinite(denom) and denom > 0 else float("nan")

    summary["drift_z_mu_bulk"] = drift("mu_bulk")
    summary["drift_z_mu_wall"] = drift("mu_wall")
    summary["drift_z_phi_gap"] = drift("phi_gap")
    summary["samples"] = samples
    return summary


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--I", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-p", type=float, default=0.4)
    ap.add_argument("--settle-units", type=float, default=60.0)
    ap.add_argument("--shear-setup-strain", type=float, default=0.5)
    ap.add_argument("--strain-measure", type=float, default=1.0)
    ap.add_argument("--np", dest="n_particles", type=int, default=1220)
    ap.add_argument("--width", type=float, default=0.40)
    ap.add_argument("--gap", type=float, default=0.30)
    ap.add_argument("--quick", action="store_true", help="small/short smoke run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    case = PlaneShear2DCase(I_target=args.I, seed=args.seed, mu_p=args.mu_p,
                            width=args.width, gap=args.gap,
                            settle_time_units=args.settle_units,
                            shear_settle_strain=args.shear_setup_strain,
                            strain_measure=args.strain_measure,
                            n_particles=args.n_particles)
    if args.quick:
        case = replace(case, width=0.24, gap=0.12, n_particles=260,
                       settle_time_units=40.0, shear_settle_strain=0.4,
                       strain_measure=0.4, sample_every=40)
    out = simulate(case, progress=True)
    out["argv"] = sys.argv[1:]
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path}")
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())