"""Synthetic granular system ``test_granular_01`` + per-case compute budget.

The identifiability face of this package is *domain neutral*: it is defined on a
synthetic, fully specified granular system (frictional disks, generic density
2500 kg/m3, no material semantics attached to the case name) so that every
number in the acceptance report is reproducible from the specification alone.
The observables are standard granular-DEM calibration measurables: the critical
tilt (avalanche onset) and packing fraction of a settled bed, and the
restitution / contact duration of a single-particle impact.

Budget discipline
-----------------
Every simulation is charged against a :class:`CaseBudget` (particles, steps,
wall-clock).  Exceeding a limit raises :class:`BudgetExceeded` instead of
silently spending more compute, so a "single case" cost is a hard, testable
property of the module rather than a hope.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..granular2d.engine import DiskConfig, DiskSystem

__all__ = [
    "BudgetExceeded",
    "CaseBudget",
    "GranularSystemSpec",
    "TEST_GRANULAR_01",
    "hex_pack",
    "disk_config",
    "bed_walls",
    "build_bed_system",
    "clone_with_gravity",
    "build_drop_system",
    "build_grain_system",
    "run_for",
]


class BudgetExceeded(RuntimeError):
    """Raised when a single-case simulation would exceed its hard budget."""


@dataclass
class CaseBudget:
    """Hard per-case compute budget (particles / steps / wall-clock seconds)."""

    max_particles: int = 400
    max_steps: int = 40000
    max_wall_seconds: float = 60.0
    particles: int = 0
    steps: int = 0
    wall_seconds: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.max_particles <= 0 or self.max_steps <= 0 or self.max_wall_seconds <= 0:
            raise ValueError("budget limits must be positive")

    def start(self) -> "CaseBudget":
        self._t0 = time.perf_counter()
        return self

    def charge_particles(self, n: int) -> None:
        self.particles = max(self.particles, int(n))
        if self.particles > self.max_particles:
            raise BudgetExceeded(
                f"case needs {self.particles} particles > budget {self.max_particles}"
            )

    def charge_steps(self, n: int) -> None:
        if self._t0 == 0.0:  # lazily started (direct use without .start())
            self._t0 = time.perf_counter()
        self.steps += int(n)
        if self.steps > self.max_steps:
            raise BudgetExceeded(
                f"case accumulated {self.steps} steps > budget {self.max_steps}"
            )
        self.wall_seconds = time.perf_counter() - self._t0
        if self.wall_seconds > self.max_wall_seconds:
            raise BudgetExceeded(
                f"case wall time {self.wall_seconds:.2f}s > budget {self.max_wall_seconds}s"
            )

    def report(self) -> dict:
        return {
            "max_particles": self.max_particles,
            "max_steps": self.max_steps,
            "max_wall_seconds": self.max_wall_seconds,
            "particles": self.particles,
            "steps": self.steps,
            "wall_seconds": round(self.wall_seconds, 4),
        }


@dataclass(frozen=True)
class GranularSystemSpec:
    """Complete specification of the synthetic system ``test_granular_01``.

    All quantities are SI.  The geometry is chosen so that one forward
    evaluation (four observables, two simulations) costs a couple of seconds on
    one CPU core, which is what makes a Jacobian / inversion study affordable
    inside a single case budget.
    """

    name: str = "test_granular_01"
    # particles ------------------------------------------------------------
    radius: float = 0.018
    polydispersity: float = 0.05
    density: float = 2500.0
    thickness: float = 0.10
    kt_ratio: float = 0.5
    mu_rolling: float = 0.02
    gravity: float = 9.81
    dt: float = 1.7e-4
    seed: int = 20261110

    # settled bed (packing fraction + tilt/avalanche onset) -----------------
    bed_width: float = 0.28
    bed_span: float = 0.28
    bed_fill_height: float = 0.20
    bed_settle: float = 0.300
    tilt_probe_time: float = 0.060
    tilt_max_deg: float = 55.0
    tilt_bisections: int = 7
    tilt_threshold_speed: float = 0.050

    # single-grain confined shear: friction ratio ---------------------------
    grain_interference: float = 2.0e-4
    grain_slide_speed: float = 0.05
    grain_slide_time: float = 0.050
    grain_wall_half_length: float = 0.20

    # drop / impact test: restitution + contact duration --------------------
    drop_height: float = 0.05

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.density <= 0.0 or self.thickness <= 0.0:
            raise ValueError("radius, density and thickness must be positive")
        if not (0.0 <= self.polydispersity < 0.5):
            raise ValueError("polydispersity must lie in [0, 0.5)")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.tilt_bisections < 1:
            raise ValueError("tilt_bisections must be >= 1")

    def with_dt_divisor(self, divisor: int) -> "GranularSystemSpec":
        """Same physical case integrated with ``dt / divisor`` (convergence ladder)."""
        if divisor < 1:
            raise ValueError("divisor must be >= 1")
        return GranularSystemSpec(**{**self.__dict__, "dt": self.dt / divisor})


TEST_GRANULAR_01 = GranularSystemSpec()


def hex_pack(
    spec: GranularSystemSpec,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    seed_offset: int,
    jitter: float = 0.01,
    pitch_factor: float = 1.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic hexagonal packing with alternating (bimodal) radii.

    Radii alternate between ``r (1 +- polydispersity)`` on a checkerboard so
    that *every* nearest-neighbour pair sums to ``2 r``; the lattice pitch is
    then ``2 r * pitch_factor`` with ``pitch_factor = 1.06`` -- a 6 % clearance
    and an initial solid fraction of ``0.907 / 1.06^2 = 0.807``, a standard loose
    packing that compacts by a fraction of a particle radius.  A 1 % positional
    jitter removes the perfect-lattice symmetry that would otherwise hide
    contact anisotropy, and a fixed seed makes the packing bit-reproducible.
    """
    rng = np.random.default_rng(spec.seed + seed_offset)
    r_mean = spec.radius
    pitch = 2.0 * r_mean * pitch_factor
    dy = pitch * math.sqrt(3.0) / 2.0
    xs: list[float] = []
    ys: list[float] = []
    rs: list[float] = []
    n_rows = int(max(0.0, (y1 - y0) - 2.0 * r_mean) // dy) + 1
    for j in range(n_rows):
        y = y0 + r_mean + j * dy
        if y > y1 - r_mean + 1e-12:
            break
        shift = 0.5 * pitch if j % 2 else 0.0
        n_cols = int(max(0.0, (x1 - x0) - 2.0 * r_mean - shift) // pitch) + 1
        for i in range(n_cols):
            x = x0 + r_mean + shift + i * pitch
            if x > x1 - r_mean + 1e-12:
                break
            sign = 1.0 if (i + j) % 2 == 0 else -1.0
            r = r_mean * (1.0 + spec.polydispersity * sign)
            r *= 1.0 + 0.01 * (2.0 * rng.random() - 1.0)
            jx = jitter * r_mean * (2.0 * rng.random() - 1.0)
            jy = jitter * r_mean * (2.0 * rng.random() - 1.0)
            xs.append(x + jx)
            ys.append(y + jy)
            rs.append(r)
    if not xs:
        raise ValueError("packing region too small for the particle radius")
    return np.column_stack([xs, ys]), np.asarray(rs)


def disk_config(params: dict, spec: GranularSystemSpec, *, dt: float | None = None) -> DiskConfig:
    """Build the engine configuration from the three identified parameters."""
    from .targets import restitution_to_zeta  # local import avoids a cycle

    return DiskConfig(
        kn=float(params["stiffness"]),
        kt_ratio=spec.kt_ratio,
        zeta=restitution_to_zeta(float(params["restitution"])),
        mu=float(params["friction"]),
        mu_rolling=spec.mu_rolling,
        density=spec.density,
        thickness=spec.thickness,
        gravity=(0.0, -spec.gravity),
        dt=spec.dt if dt is None else float(dt),
    )


def bed_walls(spec: GranularSystemSpec) -> np.ndarray:
    """Confined rectangular cell: floor, two side walls, open top."""
    w = spec.bed_width
    top = 1.6 * spec.bed_fill_height
    return np.asarray(
        [
            [[-0.5 * w, 0.0], [0.5 * w, 0.0]],
            [[-0.5 * w, 0.0], [-0.5 * w, top]],
            [[0.5 * w, 0.0], [0.5 * w, top]],
        ],
        dtype=np.float64,
    )


def build_bed_system(
    params: dict, spec: GranularSystemSpec, *, dt: float | None = None
) -> DiskSystem:
    """Rectangular cell with the bed filling its left half.

    The free half-width is the run-out space an avalanche needs; with the cell
    filled edge to edge a tilted bed is simply walled in and cannot flow.
    """
    w = spec.bed_width
    pos, rad = hex_pack(spec, -0.5 * w, -0.5 * w + spec.bed_span, 0.0,
                        spec.bed_fill_height, seed_offset=307)
    return DiskSystem(pos, rad, disk_config(params, spec, dt=dt), walls=bed_walls(spec))


def clone_with_gravity(system: DiskSystem, gravity: tuple[float, float]) -> DiskSystem:
    """Copy a system's state into a new engine instance with a rotated gravity.

    ``DiskConfig`` is frozen, so a tilt step is a rebuild: the particle state
    (positions, velocities, orientation) and the wall geometry are preserved and
    only the gravity vector changes -- the same relative configuration as
    tilting the cell.  Contact history (accumulated tangential displacement) is
    deliberately *not* carried over; a fresh contact state is the conservative
    choice for a flow/no-flow test.
    """
    cfg = dataclasses.replace(system.config, gravity=(float(gravity[0]), float(gravity[1])))
    new = DiskSystem(system.pos.copy(), system.radii.copy(), cfg, walls=system.walls.copy())
    new.vel = system.vel.copy()
    new.theta = system.theta.copy()
    new.omega = system.omega.copy()
    new.time = system.time
    new.steps = system.steps
    return new


def build_grain_system(
    params: dict, spec: GranularSystemSpec, *, dt: float | None = None
) -> DiskSystem:
    """Single grain squeezed between two parallel plates (one of them sliding).

    The plate separation is ``2 r - interference``, so the grain carries a
    well-defined normal load ``k_n * interference`` on both plates; translating
    one plate at constant velocity drives the grain to steady slip, and the
    force ratio there is the Coulomb friction coefficient -- a single-grain
    friction test with no packing statistics involved.
    """
    r = spec.radius
    gap = 2.0 * r - spec.grain_interference
    half = spec.grain_wall_half_length
    walls = np.asarray(
        [
            [[-0.5 * gap, -half], [-0.5 * gap, half]],
            [[0.5 * gap, -half], [0.5 * gap, half]],
        ],
        dtype=np.float64,
    )
    return DiskSystem(np.asarray([[0.0, 0.0]]), np.asarray([r]),
                      disk_config(params, spec, dt=dt), walls=walls)


def build_drop_system(
    params: dict, spec: GranularSystemSpec, *, dt: float | None = None
) -> DiskSystem:
    floor = np.asarray([[[-0.1, 0.0], [0.1, 0.0]]], dtype=np.float64)
    pos = np.asarray([[0.0, spec.radius + spec.drop_height]])
    rad = np.asarray([spec.radius])
    return DiskSystem(pos, rad, disk_config(params, spec, dt=dt), walls=floor)


def run_for(system: DiskSystem, duration: float, budget: CaseBudget, *, chunk: int = 50) -> None:
    """Advance ``system`` for ``duration`` seconds in budget-charged chunks."""
    n_steps = int(round(duration / system.dt))
    done = 0
    while done < n_steps:
        n = min(chunk, n_steps - done)
        system.step(n)
        done += n
        budget.charge_steps(n)