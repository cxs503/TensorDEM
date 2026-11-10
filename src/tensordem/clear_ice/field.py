"""The clear-ice floe field: polydisperse buoyant disks compressed on a hull face.

Physical scenario
-----------------
A broken/clear-ice field of ambient areal concentration ``c`` (i.e. free-water
porosity ``n = 1 - c``) fills a closed channel ``[0, Lx] x [0, Ly]`` with
periodic side walls in ``y`` and two rigid plane faces in ``x``.  The field
drifts with the ice velocity ``u_c`` (+x, the ship frame: the water/ice flow
approaches the hull) and compacts against the **hull face** at ``x = Lx``.  The
run therefore produces the two things the clear-ice problem is about:

* a **jammed layer** at the hull carrying the ice load through force chains,
  whose normal-force distribution, coordination number and porosity are the
  reported statistics;
* the **clear-ice load** on the hull face (N, and N per metre of face), which
  must satisfy the momentum balance ``<F_wall> + <sum F_drag> = 0`` in the
  steady state -- a check that no artificial momentum source exists.

Why a buoyant planar model
--------------------------
Floes of the thickness and size of a broken-ice field are in hydrostatic
equilibrium to within a fraction of a millimetre of draught change; heave is
stiff (``rho_w g A_wp``) with a period of seconds, well separated from the
contact time (milliseconds).  The vertical degree of freedom is therefore
*relaxed* (flat-lying assumption, as stated in :mod:`tensordem.ice_channel`) and
the simulation is planar: ``x, y`` translation plus yaw.  This removes floe
subduction/ride-up, which is a level-ice mechanism and out of scope here.

Drift forcing
-------------
Water drag ``F = 0.5 rho_w C_d A |u_rel| u_rel`` with ``A = 2 r t`` (frontal
submerged area) drives each floe towards the imposed ice velocity; the slip
velocity is an output.  In the steady state the drag of the whole population is
transferred to the hull face by the contact network.

Integration and the repaired defect
-----------------------------------
Velocity-Verlet (kick-drift-kick), one force evaluation per step, Mindlin
friction history advanced by the step displacement.  The step is **limited by
the contact time of the smallest floe** (``dt <= safety_factor * t_c``), and the
ratio is reported for every run: the legacy failure (P103W2B-N03) was a step
that did not resolve the contact, which showed up as a grid-frequency velocity
oscillation.  :mod:`tensordem.clear_ice.kinematics` quantifies that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .contact import STEEL, ContactMaterial, ContactSolver, contact_time, effective_modulus

__all__ = ["FieldConfig", "ClearIceField", "RunOutput"]


@dataclass
class FieldConfig:
    """Geometry, material and numerical parameters of one clear-ice run."""

    # population
    n_floes: int = 140
    mean_radius: float = 0.60
    radius_log_sigma: float = 0.30        # lognormal polydispersity of radii
    thickness: float = 0.30
    ice_concentration: float = 0.65       # ambient areal concentration c
    # channel: doubly periodic in x and y, with a rigid hull gate (segment)
    domain_x: float = 24.0
    domain_y: float = 18.0
    gate_width: float = 18.0              # hull face span (full channel width)
    gate_x_fraction: float = 1.00         # gate position / Lx (at the far face)
    # forcing
    drift_speed: float = 0.80             # imposed ice drift speed (m/s), +x
    water_density: float = 1025.0
    drag_coefficient: float = 1.2
    # numerics
    dt: float = 5.0e-4
    duration: float = 8.0
    warmup: float = 4.0                   # statistics window starts here
    kinematics_window: tuple = (0.5, 2.5)  # window for the resolution metrics
    sample_every: float = 0.02
    safety_factor: float = 0.10           # dt <= safety_factor * t_contact
    seed: int = 20261109
    material: ContactMaterial = field(default_factory=ContactMaterial)
    wall_material: ContactMaterial = field(default_factory=lambda: STEEL)

    def __post_init__(self) -> None:
        if self.n_floes < 2:
            raise ValueError("n_floes must be at least 2")
        for name in ("mean_radius", "thickness", "domain_x", "domain_y",
                     "dt", "duration"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.drift_speed < 0.0:
            raise ValueError("drift_speed must be non-negative")
        if not 0.0 < self.ice_concentration < 0.9:
            raise ValueError("ice_concentration must lie in (0, 0.9)")
        if self.duration <= self.warmup:
            raise ValueError("duration must exceed warmup")

    @property
    def domain_area(self) -> float:
        return self.domain_x * self.domain_y

    @property
    def gate_centre(self) -> np.ndarray:
        return np.array([self.gate_x_fraction * self.domain_x, 0.5 * self.domain_y])

    def gate_segment(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.gate_centre
        half = 0.5 * self.gate_width
        return (np.array([c[0], c[1] - half]), np.array([c[0], c[1] + half]))


@dataclass
class RunOutput:
    """Everything one clear-ice run produces (arrays, no hidden state)."""

    config: FieldConfig
    radii: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    dt: float
    dt_over_tc: float
    time: np.ndarray
    wall_force: np.ndarray             # hull-face load along +x (N), per step
    wall_force_y: np.ndarray           # tangential (friction) part, per step
    drag_force_x: np.ndarray           # total water drag along +x (N), per step
    kinetic_energy: np.ndarray
    mean_speed: np.ndarray
    coordination: np.ndarray
    sample_times: np.ndarray
    positions: np.ndarray              # (S, N, 2)
    velocities: np.ndarray             # (S, N, 2)
    contact_fn: list
    contact_i: list
    contact_j: list
    max_overlap_ratio: float
    steps: int

    @property
    def n_floes(self) -> int:
        return len(self.radii)


class ClearIceField:
    """Polydisperse clear-ice floe field with Hertz-Mindlin contacts."""

    def __init__(self, config: FieldConfig = FieldConfig(),
                 population: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
        self.config = cfg = config
        rng = np.random.default_rng(cfg.seed)
        box = np.array([cfg.domain_x, cfg.domain_y])

        if population is None:
            self.radius = self._radii(rng)
        else:
            x0, r0, t0 = population
            x0 = np.asarray(x0, dtype=float).reshape(-1, 2)
            r0 = np.asarray(r0, dtype=float).ravel()
            if len(x0) != len(r0):
                raise ValueError("population positions and radii differ in length")
            if np.any(r0 <= 0.0):
                raise ValueError("population radii must be positive")
            if np.any(t0):
                cfg = replace(cfg, thickness=float(np.mean(t0)))
                self.config = cfg
            self.radius = r0.copy()

        self.solver = ContactSolver(
            cfg.material, box=(cfg.domain_x, cfg.domain_y),
            skin=max(1.0e-3, 5.0e-3 * float(self.radius.mean())))

        if population is None:
            self.x = self._place(rng)
        else:
            self.x = x0.copy() % box

        # the field is handed over already drifting with the ice: initialising
        # at rest would add a drag spin-up transient of ~5 quadratic-drag time
        # constants before the statistics window can open
        self.v = np.zeros((len(self.radius), 2))
        self.v[:, 0] = cfg.drift_speed
        self.theta = np.zeros(len(self.radius))
        self.omega = np.zeros(len(self.radius))
        self.mass = cfg.material.density * np.pi * self.radius ** 2 * cfg.thickness
        self.inertia = 0.5 * self.mass * self.radius ** 2

        # contact-time stability diagnostic: the *smallest* floe sets the limit
        # (t_c ~ r^{3/5} at fixed thickness, so the smallest floe is fastest)
        i_small = int(np.argmin(self.radius))
        e_eff = effective_modulus(cfg.material, cfg.material)
        self.tc = float(contact_time(0.5 * self.mass[i_small], 0.5 * self.radius[i_small],
                                     e_eff, cfg.drift_speed))
        self.dt_configured = float(cfg.dt)
        self.dt = float(min(cfg.dt, cfg.safety_factor * self.tc))
        self.dt_over_tc = self.dt / self.tc

        self._gate = cfg.gate_segment()
        self._acc = np.zeros_like(self.x)
        self._alpha = np.zeros(len(self.radius))
        self._acc, self._alpha, _, _ = self._accelerations(self.x, self.v, self.omega, self.dt)

    # -- population generation -------------------------------------------
    def _radii(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        shape = rng.lognormal(mean=0.0, sigma=cfg.radius_log_sigma, size=cfg.n_floes)
        shape /= shape.mean()
        radii = cfg.mean_radius * shape
        area_needed = cfg.ice_concentration * cfg.domain_area
        radii *= np.sqrt(area_needed / (np.pi * np.sum(radii ** 2)))
        return radii

    def _place(self, rng: np.random.Generator) -> np.ndarray:
        """Random placement + position-based overlap removal (geometric only).

        The projection separates each overlapping pair by half of its overlap
        (capped), iterated until the largest relative overlap falls below
        ``1e-4``.  No energy is invented; the residual overlap is at most
        ``1e-4 r`` and the contact solver removes it during the warm-up.
        """
        cfg = self.config
        self._gate = cfg.gate_segment()
        x = np.empty((cfg.n_floes, 2))
        x = rng.random((cfg.n_floes, 2)) * np.array([cfg.domain_x, cfg.domain_y])
        r = self.radius
        box = np.array([cfg.domain_x, cfg.domain_y])
        cap = 0.10 * float(r.mean())
        for _ in range(3000):
            i, j = self.solver.pair_list(x, r)
            if len(i) == 0:
                break
            d = x[j] - x[i]
            d -= box * np.round(d / box)
            dist = np.linalg.norm(d, axis=1)
            overlap = r[i] + r[j] - dist
            act = overlap > 1.0e-4 * float(r.mean())
            if not np.any(act):
                break
            i, j, overlap = i[act], j[act], overlap[act]
            d, dist = d[act], dist[act]
            nrm = d / np.maximum(dist, 1e-12)[:, None]
            corr = 0.5 * overlap[:, None] * nrm
            disp = np.zeros_like(x)
            cnt = np.zeros(len(x))
            for axis in (0, 1):
                disp[:, axis] = (np.bincount(j, weights=corr[:, axis], minlength=len(x))
                                 - np.bincount(i, weights=corr[:, axis], minlength=len(x)))
            cnt = (np.bincount(i, minlength=len(x)) + np.bincount(j, minlength=len(x))).astype(float)
            step = disp / np.maximum(cnt, 1.0)[:, None]
            mag = np.linalg.norm(step, axis=1)
            scale = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
            # keep the packing out of the rigid gate (a floe initialised inside
            # the gate would take a non-physical ejection impulse)
            p0, p1 = self._gate
            ab = p1 - p0
            tt = np.clip(((x - p0) @ ab) / max(float(ab @ ab), 1e-30), 0.0, 1.0)
            rel = x - (p0 + tt[:, None] * ab)
            dist_w = np.linalg.norm(rel, axis=1)
            pen = r - dist_w
            on_wall = pen > 1.0e-6
            if np.any(on_wall):
                nw = rel[on_wall] / np.maximum(dist_w[on_wall], 1e-12)[:, None]
                disp[on_wall] += nw * pen[on_wall, None]
                cnt[on_wall] += 1.0
                step = disp / np.maximum(cnt, 1.0)[:, None]
                mag = np.linalg.norm(step, axis=1)
                scale = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
                step = step * scale[:, None]
            x = x + step * scale[:, None]
            x %= box
        return x

    # -- forcing ---------------------------------------------------------
    def _drag(self, v: np.ndarray) -> np.ndarray:
        cfg = self.config
        area = 2.0 * self.radius * cfg.thickness
        rel = v - np.array([cfg.drift_speed, 0.0])
        speed = np.linalg.norm(rel, axis=1)
        return -0.5 * cfg.water_density * cfg.drag_coefficient * area[:, None] \
            * speed[:, None] * rel

    def _accelerations(self, x, v, omega, dt):
        cfg = self.config
        f, tau, diag = self.solver.update(x, v, self.radius, self.mass, omega, dt,
                                          cfg.thickness)
        fw, tw = self.solver.wall_contacts(
            x, v, self.radius, omega, self.mass, self._gate[0], self._gate[1],
            cfg.wall_material, cfg.thickness)
        total = f + fw + self._drag(v)
        self.last_diag = diag
        self.last_wall = (fw, tw)
        return total / self.mass[:, None], (tau + tw) / self.inertia, diag, fw

    def wall_load(self, fw: np.ndarray) -> tuple[float, float]:
        return float(fw[:, 0].sum()), float(fw[:, 1].sum())

    # -- time integration ------------------------------------------------
    def run(self) -> RunOutput:
        cfg = self.config
        dt = self.dt
        n_steps = int(round(cfg.duration / dt))
        box = np.array([cfg.domain_x, cfg.domain_y])

        time = np.arange(n_steps + 1) * dt
        wall_force = np.zeros(n_steps + 1)
        wall_force_y = np.zeros(n_steps + 1)
        drag_x = np.zeros(n_steps + 1)
        kinetic = np.zeros(n_steps + 1)
        mean_speed = np.zeros(n_steps + 1)
        coord = np.zeros(n_steps + 1)
        samples = {"t": [], "x": [], "v": [], "fn": [], "ci": [], "cj": []}
        max_overlap_ratio = 0.0

        def record(k: int) -> None:
            nonlocal max_overlap_ratio
            diag = self.last_diag
            wx, wy = self.wall_load(self.last_wall[0])
            wall_force[k], wall_force_y[k] = wx, wy
            drag_x[k] = float(self._drag(self.v)[:, 0].sum())
            kinetic[k] = 0.5 * float(np.sum(self.mass * np.sum(self.v ** 2, axis=1))
                                     + np.sum(self.inertia * self.omega ** 2))
            mean_speed[k] = float(np.mean(np.linalg.norm(self.v, axis=1)))
            deg = (np.bincount(diag.i, minlength=len(self.radius))
                   + np.bincount(diag.j, minlength=len(self.radius)))
            coord[k] = float(deg.mean())
            if len(diag):
                max_overlap_ratio = max(
                    max_overlap_ratio,
                    float(np.max(diag.overlap / self.radius[np.minimum(diag.i, diag.j)])))

        record(0)
        next_sample = 0.0
        for k in range(n_steps):
            self.v += 0.5 * dt * self._acc
            self.omega += 0.5 * dt * self._alpha
            self.x = (self.x + dt * self.v) % box
            self.theta += dt * self.omega
            acc, alpha, _, _ = self._accelerations(self.x, self.v, self.omega, dt)
            self._acc, self._alpha = acc, alpha
            self.v += 0.5 * dt * acc
            self.omega += 0.5 * dt * alpha
            record(k + 1)
            if (k + 1) * dt + 1e-12 >= next_sample:
                samples["t"].append((k + 1) * dt)
                samples["x"].append(self.x.copy())
                samples["v"].append(self.v.copy())
                diag = self.last_diag
                samples["fn"].append(diag.fn.copy())
                samples["ci"].append(diag.i.copy())
                samples["cj"].append(diag.j.copy())
                next_sample += cfg.sample_every

        return RunOutput(
            config=cfg, radii=self.radius.copy(), mass=self.mass.copy(),
            inertia=self.inertia.copy(), dt=dt, dt_over_tc=self.dt_over_tc,
            time=time, wall_force=wall_force, wall_force_y=wall_force_y,
            drag_force_x=drag_x, kinetic_energy=kinetic, mean_speed=mean_speed,
            coordination=coord, sample_times=np.asarray(samples["t"]),
            positions=np.asarray(samples["x"]), velocities=np.asarray(samples["v"]),
            contact_fn=samples["fn"], contact_i=samples["ci"], contact_j=samples["cj"],
            max_overlap_ratio=max_overlap_ratio, steps=n_steps,
        )