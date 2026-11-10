"""Two-dimensional granular column collapse: runout scaling, energy ledger, mu(I).

Physical background
-------------------
A column of loose granular material of initial length ``L_i`` and height ``H_i``
is released from rest on a rough horizontal floor.  The runout

.. math:: \\Delta L^* = (L_f - L_i) / L_i

collapses onto a one-parameter law in the initial aspect ratio ``a_0 = H_i/L_i``
(Lube, Huppert, Sparks & Freundt, JGR 109, B10203, 2004 for axisymmetric
columns; Lube, Huppert, Sparks & Freundt, PRE 72, 041301, 2005 for
two-dimensional columns):

    a_0 <~ 1          :  Delta L* grows linearly in a_0,
    a_0 >~ 1          :  Delta L* ~ c a_0^(2/3)      (``c`` is geometry dependent)

The 2/3 exponent is the fingerprint of the balance between the potential energy
released by the column and the frictional dissipation in the spreading layer;
it is *independent* of the grain size and of gravity (the collapse is
scale-free apart from ``a_0``).  Reproducing that exponent from a bare
contact-mechanics DEM (no fitting) is the strongest available validation of the
contact/friction model of :mod:`tensordem.granular2d.engine`.

During the flow the material is dilatant and rate dependent; the local state is
described by the inertial number

.. math:: I = \\dot\\gamma d / \\sqrt{P / \\rho}, \\qquad \\mu(I) = \\tau/P

with ``mu(I)`` increasing from the static value ``mu_s`` to ``mu_2`` over a
characteristic ``I_0`` (Jop, Forterre & Pouliquen, Nature 441, 727, 2006).  The
same snapshots therefore give an ``I``-``phi`` (compaction/dilatancy) and an
``I``-``mu`` (rheology) data set at no extra cost.

Energy ledger
-------------
The collapse is dissipative from rest to rest, so the initial potential energy
``E_p0`` must be accounted for by the contact dissipation integrated over the
run (:meth:`tensordem.granular2d.engine.DiskSystem.dissipated_work`) plus the
small residual kinetic and elastic energies.  The closure residual of that
ledger is reported per run and is the acceptance quantity of criterion (e).

Units are SI throughout (m, kg, s, N); stresses are 2D (N/m) and are converted
with the slice thickness ``thickness`` when an inertial number is formed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .engine import DiskConfig, DiskSystem, box_walls
from .stress import lattice_stress

__all__ = [
    "ColumnSpec",
    "ColumnResult",
    "build_column",
    "run_collapse",
    "flow_samples",
    "fit_runout_scaling",
    "fit_phi_inertial",
    "runout_scan",
    "LUBE_TWO_THIRDS",
]

#: Lube et al. (2004, 2005) large-``a_0`` runout exponent (2D and axisymmetric).
LUBE_TWO_THIRDS = 2.0 / 3.0


class ColumnCollapseError(ValueError):
    """Raised for inconsistent column-collapse inputs."""


@dataclass(frozen=True)
class ColumnSpec:
    """Geometry and material parameters of one collapse run (SI)."""

    aspect_ratio: float = 2.0
    width: float = 0.06            # initial column length L_i (gate position) [m]
    radius: float = 0.003          # disk radius [m]
    density: float = 2500.0        # grain density [kg/m^3]
    thickness: float = 0.02        # out-of-plane slice thickness (mass only) [m]
    kn: float = 2.0e5              # normal stiffness [N/m]
    kt_ratio: float = 0.5
    zeta: float = 0.2              # normal dashpot damping ratio
    mu: float = 0.5                # grain-grain and grain-floor friction
    mu_rolling: float = 0.02       # rolling resistance coefficient
    gravity: float = 9.81          # [m/s^2]
    gap: float = 2.0e-4            # initial lattice gap between disks [m]
    jitter: float = 0.0            # deterministic position jitter fraction of r
    seed: int = 7006
    max_time: float = 4.0          # [s]
    stop_speed: float = 2.0e-3     # rest criterion on the mean speed [m/s]
    hold_steps: int = 400          # steps that must stay below stop_speed
    snapshot_interval: float = 0.02  # [s]
    bins: int = 10                 # vertical coarse-graining bins for mu(I)
    floor_extension: float = 8.0   # floor length beyond the gate, in units of H_i

    def __post_init__(self) -> None:
        for name in ("aspect_ratio", "width", "radius", "density", "thickness",
                     "kn", "gravity", "max_time", "snapshot_interval"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ColumnCollapseError(f"{name} must be finite and positive")
        for name in ("kt_ratio", "zeta", "mu", "mu_rolling", "gap", "jitter",
                     "stop_speed"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ColumnCollapseError(f"{name} must be finite and nonnegative")
        if int(self.bins) < 3:
            raise ColumnCollapseError("bins must be >= 3")
        if int(self.hold_steps) < 1:
            raise ColumnCollapseError("hold_steps must be >= 1")
        if self.width < 4.0 * self.radius:
            raise ColumnCollapseError("column must be at least 4 grains wide")
        if not math.isfinite(float(self.floor_extension)) or self.floor_extension < 2.0:
            raise ColumnCollapseError("floor_extension must be finite and >= 2")

    @property
    def height(self) -> float:
        """Nominal initial column height ``a_0 L_i`` [m]."""
        return self.aspect_ratio * self.width

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def particle_mass(self) -> float:
        return self.density * math.pi * self.radius ** 2 * self.thickness

    def disk_config(self) -> DiskConfig:
        return DiskConfig(
            kn=self.kn, kt_ratio=self.kt_ratio, zeta=self.zeta, mu=self.mu,
            mu_rolling=self.mu_rolling, density=self.density,
            thickness=self.thickness, gravity=(0.0, -self.gravity),
            skin=max(2.0 * self.radius, 4.0 * self.radius),
        )


@dataclass
class ColumnResult:
    """Outcome of one collapse run with its energy ledger and flow samples."""

    spec: ColumnSpec
    aspect_ratio: float
    length_initial: float
    height_initial: float
    length_final: float
    delta_length: float
    runout_dimensionless: float
    n_particles: int
    energy_potential_initial: float
    energy_potential_final: float
    energy_kinetic_final: float
    energy_elastic_final: float
    energy_dissipated: float
    energy_residual_relative: float
    energy_dissipated_relative: float
    kinetic_peak: float
    duration: float
    steps: int
    dt: float
    front_history: list[tuple[float, float]] = field(default_factory=list)
    kinetic_history: list[tuple[float, float]] = field(default_factory=list)
    mu_i_points: list[dict[str, float]] = field(default_factory=list)
    phi_i_points: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "aspect_ratio": self.aspect_ratio,
            "length_initial": self.length_initial,
            "height_initial": self.height_initial,
            "length_final": self.length_final,
            "delta_length": self.delta_length,
            "runout_dimensionless": self.runout_dimensionless,
            "n_particles": self.n_particles,
            "energy_potential_initial": self.energy_potential_initial,
            "energy_potential_final": self.energy_potential_final,
            "energy_kinetic_final": self.energy_kinetic_final,
            "energy_elastic_final": self.energy_elastic_final,
            "energy_dissipated": self.energy_dissipated,
            "energy_residual_relative": self.energy_residual_relative,
            "energy_dissipated_relative": self.energy_dissipated_relative,
            "kinetic_peak": self.kinetic_peak,
            "duration": self.duration,
            "steps": self.steps,
            "dt": self.dt,
            "n_mu_i_points": len(self.mu_i_points),
            "n_phi_i_points": len(self.phi_i_points),
        }


def _hex_column_positions(spec: ColumnSpec) -> np.ndarray:
    """Hexagonal filling whose outer envelope is exactly ``L_i x H_i``.

    The column spacing is fitted to the requested width and height (never below
    the 2R no-overlap limit), so the *measured* initial geometry equals the
    nominal one and the aspect ratio carries no lattice-quantisation error.
    """
    r = spec.radius
    dx_min = 2.0 * r
    dy_min = math.sqrt(3.0) * r
    if spec.width - 2.0 * r < dx_min * 0.5 or spec.height - 2.0 * r < dy_min * 0.5:
        raise ColumnCollapseError("the column is too small for its grain radius")
    nx = int(math.floor((spec.width - 2.0 * r) / dx_min)) + 1
    ny = int(math.floor((spec.height - 2.0 * r) / dy_min)) + 1
    nx, ny = max(nx, 2), max(ny, 2)
    dx = (spec.width - 2.0 * r) / (nx - 1)
    dy = (spec.height - 2.0 * r) / (ny - 1)
    rng = np.random.default_rng(spec.seed)
    pts: list[tuple[float, float]] = []
    for row in range(ny):
        y = r + row * dy
        if row % 2 == 0:
            xs = [r + k * dx for k in range(nx)]
        else:
            xs = [r + 0.5 * dx + k * dx for k in range(nx - 1)]
        pts.extend((x, y) for x in xs)
    pos = np.asarray(pts, dtype=np.float64)
    if spec.jitter > 0.0:
        pos = pos + spec.jitter * r * rng.uniform(-1.0, 1.0, size=pos.shape)
        pos[:, 0] = np.clip(pos[:, 0], r, spec.width - r)
        pos[:, 1] = np.clip(pos[:, 1], r, spec.height - r)
    # reject the (pathological) input where the fitted lattice would overlap
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    if float(np.min(dist)) < dx_min * (1.0 - 1e-3):
        raise ColumnCollapseError("fitted lattice overlaps; widen the column or shrink the grain")
    return pos


def build_column(spec: ColumnSpec) -> tuple[DiskSystem, dict[str, float]]:
    """Build the resting column retained by a gate at ``x = L_i``.

    The floor and the back (left) wall are rough static walls; the gate is
    represented by *not* passing a right wall, so releasing the column is simply
    the absence of that boundary.  The returned dictionary holds the measured
    initial geometry (``L_i``, ``H_i``, ``a_0``) used by the runout definitions.
    """
    pos = _hex_column_positions(spec)
    radii = np.full(len(pos), spec.radius)
    # the floor must be long enough for the whole runout; a too short floor lets
    # the leading grains run off the edge and fall ballistically (that is not a
    # granular collapse and used to fake arbitrarily large runouts).
    floor_length = spec.width + spec.floor_extension * spec.height
    walls = box_walls(floor_length,
                      max(spec.height, 1.0) + 4.0 * spec.radius,
                      left=True, right=False, bottom=True, top=False,
                      margin=2.0 * spec.radius)
    system = DiskSystem(pos, radii, spec.disk_config(), walls=walls)
    x_left = float(np.min(system.pos[:, 0] - radii))
    x_right = float(np.max(system.pos[:, 0] + radii))
    y_top = float(np.max(system.pos[:, 1] + radii))
    geometry = {
        "length_initial": x_right - x_left,
        "height_initial": y_top,
        "front_initial": x_right,
        "back_wall": x_left,
        "aspect_ratio_measured": y_top / max(x_right - x_left, 1e-30),
    }
    return system, geometry


def _local_flow_sample(system: DiskSystem, spec: ColumnSpec) -> dict[str, list]:
    """Coarse-grained ``I``, ``mu`` and ``phi`` in vertical bins of the flow.

    The vertical velocity profile ``v_x(y)`` gives the shear rate; the virial
    stress (:func:`tensordem.granular2d.stress.lattice_stress`) gives ``P`` and
    ``tau``; the areal grain fraction gives ``phi``.  Bins with fewer than four
    disks or without a resolved shear rate are dropped.
    """
    y_top = float(np.max(system.pos[:, 1] + system.radii))
    y_bottom = float(np.min(system.pos[:, 1] - system.radii))
    x_min = float(np.min(system.pos[:, 0] - system.radii))
    x_max = float(np.max(system.pos[:, 0] + system.radii))
    if y_top - y_bottom <= 0.0 or x_max - x_min <= 0.0:
        return {"mu_i": [], "phi_i": []}
    nb = int(spec.bins)
    field = lattice_stress(system, nx=1, ny=nb,
                           bounds=(x_min, y_bottom, x_max, y_top))
    sigma = field["sigma"][0]
    counts = field["counts"][0]
    dy = float(field["dy"][0])
    dx = float(field["dx"][0])
    edges = np.linspace(y_bottom, y_top, nb + 1)
    idx = np.clip(((system.pos[:, 1] - y_bottom) / dy).astype(int), 0, nb - 1)

    vx = np.zeros(nb)
    n_bin = np.zeros(nb, dtype=int)
    area = np.zeros(nb)
    for b in range(nb):
        sel = idx == b
        n_bin[b] = int(np.count_nonzero(sel))
        if n_bin[b]:
            vx[b] = float(np.mean(system.vel[sel, 0]))
            area[b] = float(np.sum(math.pi * system.radii[sel] ** 2))
    y_center = 0.5 * (edges[:-1] + edges[1:])
    cell_area = dx * dy
    phi = area / cell_area

    mu_i: list[dict[str, float]] = []
    phi_i: list[dict[str, float]] = []
    rho_2d_solid = spec.density * spec.thickness
    d = spec.diameter
    for b in range(1, nb - 1):
        if n_bin[b] < 4 or n_bin[b - 1] < 2 or n_bin[b + 1] < 2:
            continue
        gamma_dot = (vx[b + 1] - vx[b - 1]) / (2.0 * dy)
        pressure = -float(sigma[b, 1, 1])
        tau = 0.5 * (float(sigma[b, 0, 1]) + float(sigma[b, 1, 0]))
        if pressure <= 0.0 or counts[b] < 3 or phi[b] <= 0.0:
            continue
        if phi[b] > 0.92:  # unphysical bin average (grain overlap / tiny cell)
            continue
        inertial = abs(gamma_dot) * d / math.sqrt(pressure / (phi[b] * rho_2d_solid))
        mu_local = abs(tau) / pressure
        if not (math.isfinite(inertial) and math.isfinite(mu_local)):
            continue
        phi_i.append({"inertial_number": inertial, "packing_fraction": float(phi[b]),
                      "bin_height": float(y_center[b]), "counts": float(counts[b])})
        mu_i.append({"inertial_number": inertial, "mu": mu_local,
                     "shear_rate": gamma_dot, "pressure": pressure,
                     "bin_height": float(y_center[b]), "counts": float(counts[b])})
    return {"mu_i": mu_i, "phi_i": phi_i}


def run_collapse(spec: ColumnSpec, *, sample_flow: bool = True) -> ColumnResult:
    """Release the column, integrate to rest and close the energy ledger."""
    system, geo = build_column(spec)
    gravity = float(spec.gravity)
    e_p0 = float(np.sum(system.mass * gravity * system.pos[:, 1]))
    ledger = system.enable_energy_ledger()
    energies0 = system.energies()
    x_front0 = geo["front_initial"]
    front_history: list[tuple[float, float]] = []
    kinetic_history: list[tuple[float, float]] = []
    mu_i_points: list[dict[str, float]] = []
    phi_i_points: list[dict[str, float]] = []
    kinetic_peak = 0.0
    next_snapshot = 0.0
    below = 0
    while system.time < spec.max_time:
        system.step(1)
        ke = system.kinetic_energy()
        kinetic_peak = max(kinetic_peak, ke)
        if system.time >= next_snapshot:
            next_snapshot += spec.snapshot_interval
            front = float(np.max(system.pos[:, 0] + system.radii))
            front_history.append((system.time, front))
            kinetic_history.append((system.time, ke))
            if sample_flow and ke > 1.0e-6:
                sample = _local_flow_sample(system, spec)
                mu_i_points.extend(sample["mu_i"])
                phi_i_points.extend(sample["phi_i"])
        if ke < 1.0e-6 and system.time > 0.15:
            below += 1
            if below >= spec.hold_steps:
                break
        else:
            below = 0
    energies1 = system.energies()
    x_front = float(np.max(system.pos[:, 0] + system.radii))
    e_p1 = float(np.sum(system.mass * gravity * system.pos[:, 1]))
    elastic0 = energies0["elastic_normal"] + energies0["elastic_tangential"]
    elastic1 = energies1["elastic_normal"] + energies1["elastic_tangential"]
    ke1 = energies1["kinetic"]
    dissipated = system.dissipated_work()
    residual = (e_p0 + elastic0) - (e_p1 + elastic1 + ke1 + dissipated)
    delta_length = x_front - x_front0
    return ColumnResult(
        spec=spec,
        aspect_ratio=geo["aspect_ratio_measured"],
        length_initial=geo["length_initial"],
        height_initial=geo["height_initial"],
        length_final=x_front - geo["back_wall"],
        delta_length=delta_length,
        runout_dimensionless=delta_length / geo["length_initial"],
        n_particles=system.n,
        energy_potential_initial=e_p0,
        energy_potential_final=e_p1,
        energy_kinetic_final=ke1,
        energy_elastic_final=elastic1,
        energy_dissipated=dissipated,
        energy_residual_relative=residual / max(e_p0, 1e-30),
        energy_dissipated_relative=dissipated / max(e_p0, 1e-30),
        kinetic_peak=kinetic_peak,
        duration=system.time,
        steps=system.steps,
        dt=system.dt,
        front_history=front_history,
        kinetic_history=kinetic_history,
        mu_i_points=mu_i_points,
        phi_i_points=phi_i_points,
    )


def flow_samples(result: ColumnResult) -> dict[str, np.ndarray]:
    """Stack the inertial-number samples of one run into arrays."""
    if not result.mu_i_points:
        return {"inertial_number": np.zeros(0), "mu": np.zeros(0),
                "packing_fraction": np.zeros(0)}
    I_mu = np.array([p["inertial_number"] for p in result.mu_i_points])
    mu = np.array([p["mu"] for p in result.mu_i_points])
    phi = np.array([p["packing_fraction"] for p in result.phi_i_points])
    n = min(len(I_mu), len(phi))
    return {"inertial_number": I_mu[:n], "mu": mu[:n], "packing_fraction": phi[:n]}


def fit_runout_scaling(results: list[ColumnResult], *,
                       aspect_min: float = 1.0) -> dict[str, float]:
    """Least-squares fit of ``Delta L* = c a_0^n`` on the large-``a_0`` branch."""
    keep = [r for r in results if r.aspect_ratio >= aspect_min
            and r.runout_dimensionless > 0.0]
    if len(keep) < 3:
        raise ColumnCollapseError("at least three runs with a_0 >= aspect_min are needed")
    a = np.log([r.aspect_ratio for r in keep])
    y = np.log([r.runout_dimensionless for r in keep])
    slope, intercept = np.polyfit(a, y, 1)
    pred = slope * a + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    n = len(keep)
    return {
        "exponent": float(slope),
        "prefactor": float(math.exp(intercept)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "exponent_reference": LUBE_TWO_THIRDS,
        "exponent_error_relative": abs(slope - LUBE_TWO_THIRDS) / LUBE_TWO_THIRDS,
        "aspect_ratios": [r.aspect_ratio for r in keep],
        "runouts": [r.runout_dimensionless for r in keep],
        "n_points": n,
        "aspect_min": aspect_min,
    }


def fit_phi_inertial(points: list[dict[str, float]]) -> dict[str, float]:
    """Fit the dilatancy law ``phi = phi_c - alpha I^beta`` to sampled bins."""
    if len(points) < 4:
        raise ColumnCollapseError("at least four samples are required")
    I = np.array([p["inertial_number"] for p in points])
    phi = np.array([p["packing_fraction"] for p in points])
    order = np.argsort(I)
    I, phi = I[order], phi[order]
    # non-linear least squares on (phi_c, alpha, beta); beta > 0 enforced
    from scipy.optimize import curve_fit

    def law(x, phi_c, alpha, beta):
        return phi_c - alpha * np.power(np.maximum(x, 1e-12), beta)

    p0 = [max(float(np.max(phi)), 0.7), 0.05, 0.3]
    bounds = ([0.5, 0.0, 0.05], [1.0, 1.0, 2.0])
    popt, _ = curve_fit(law, I, phi, p0=p0, bounds=bounds, maxfev=20000)
    pred = law(I, *popt)
    ss_res = float(np.sum((phi - pred) ** 2))
    ss_tot = float(np.sum((phi - np.mean(phi)) ** 2))
    rank = _spearman(np.log(np.maximum(I, 1e-12)), phi)
    return {
        "phi_c": float(popt[0]),
        "alpha": float(popt[1]),
        "beta": float(popt[2]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "spearman_logI_phi": float(rank),
        "n_points": int(len(points)),
        "inertial_range": [float(I.min()), float(I.max())],
        "phi_range": [float(phi.min()), float(phi.max())],
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx)) * float(np.sum(ry * ry)))
    return 0.0 if denom == 0.0 else float(np.sum(rx * ry) / denom)


def fit_mu_inertial(points: list[dict[str, float]]) -> dict[str, float]:
    """Fit the Jop ``mu(I)`` law to the collapse samples (log-free LSQ)."""
    if len(points) < 4:
        raise ColumnCollapseError("at least four samples are required")
    from scipy.optimize import curve_fit

    I = np.array([p["inertial_number"] for p in points])
    mu = np.array([p["mu"] for p in points])

    def law(x, mu_s, mu_2, I_0):
        return mu_s + (mu_2 - mu_s) / (I_0 / np.maximum(x, 1e-12) + 1.0)

    p0 = [0.35, 0.7, 0.3]
    bounds = ([0.05, 0.3, 0.01], [0.9, 1.5, 3.0])
    popt, _ = curve_fit(law, I, mu, p0=p0, bounds=bounds, maxfev=40000)
    pred = law(I, *popt)
    ss_res = float(np.sum((mu - pred) ** 2))
    ss_tot = float(np.sum((mu - np.mean(mu)) ** 2))
    return {
        "mu_s": float(popt[0]),
        "mu_2": float(popt[1]),
        "I_0": float(popt[2]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "spearman_logI_mu": _spearman(np.log(np.maximum(I, 1e-12)), mu),
        "n_points": int(len(points)),
        "inertial_range": [float(I.min()), float(I.max())],
        "mu_range": [float(mu.min()), float(mu.max())],
    }


def runout_scan(aspect_ratios, *, base: ColumnSpec | None = None,
                sample_flow: bool = False) -> list[ColumnResult]:
    """Run one collapse per aspect ratio in ``aspect_ratios``."""
    base = base or ColumnSpec()
    out = []
    for a in aspect_ratios:
        spec = ColumnSpec(**{**base.__dict__, "aspect_ratio": float(a)})
        out.append(run_collapse(spec, sample_flow=sample_flow))
    return out