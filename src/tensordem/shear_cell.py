"""Direct-shear cell for macroscopic-friction measurement (numpy kernel).

This module is the *measurement device* for the macroscopic (bulk) friction
coefficient of a sphere assembly: a rough floor, frictionless side walls and a
stress-servo top platen that is dragged sideways.  The measured pair
``(sigma_n, tau)`` at several normal-stress levels gives the Mohr-Coulomb
envelope ``tau = c + sigma_n tan(phi)`` and hence the macroscopic friction
coefficient ``mu_macro = tau / sigma_n``.

Protocol
--------
1. HCP fill of the box, gravity settle (platen parked above the bed).
2. **Consolidation under a stress servo**: the platen velocity is driven by the
   force error ``N_target - N_measured`` so the normal stress is *held* at the
   target.  A dead-load platen loses contact when the bed dilates under shear
   (``sigma_n -> 0``, ``mu`` undefined); the servo cannot.
3. Shear: the platen advances at ``shear_velocity`` while the servo keeps
   ``sigma_n`` on target; ``tau`` and ``sigma_n`` are sampled.
4. Steady-state means over the last part of the shear window, with a
   block-average standard error, plus a floor-traction cross-check.

Known defect fixed here (mission m13798, 2026-11-11)
---------------------------------------------------
The previous revision never executed: ``mass[:, None]`` indexed a Python float
(``TypeError: 'float' object is not subscriptable`` at the first step).  It also
fitted a Mohr-Coulomb *slope* inside a single run, where ``sigma_n`` is
essentially constant -- an ill-posed regression that turns normal-stress
noise into a "friction angle", and it used ``sum(|f_z|)`` as the applied normal
stress.  The device now measures ``tau/sigma`` per level and only fits the
envelope across *distinct* stress levels (:func:`measure_macro_friction`).

All quantities are SI (m, kg, s, N, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .granular3d_core import (
    Material3D,
    ContactState,
    build_pairs,
    contact_forces,
    hcp_positions_in_box,
)


def inertial_number(shear_rate: float, radius: float, density: float,
                    normal_stress: float) -> float:
    """Inertial number ``I = gamma_dot * d * sqrt(rho / sigma)``, ``d = 2 r``.

    da Cruz et al. (2005) *Phys. Rev. E* 72:021309 / GDR MiDi (2004) definition.
    Returns ``nan`` when any input cannot define it (non-positive stress, size,
    density or rate), so an unloaded record can never manufacture an ``I``.
    """
    if not (shear_rate > 0.0 and radius > 0.0 and density > 0.0
            and normal_stress > 0.0):
        return float("nan")
    return shear_rate * 2.0 * radius / math.sqrt(normal_stress / density)


@dataclass(frozen=True)
class ShearCellCase:
    """Direct-shear cell parameters (SI)."""
    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    gravity: float = 9.81
    species: str = "glass_bead"      # free-form label carried into the results
    # geometry
    box_width: float = 0.08     # x direction (shear direction)
    box_depth: float = 0.032    # y direction
    fill_height: float = 0.05   # z direction
    packing_gap: float = 0.02
    jitter: float = 0.01
    # numerics
    dt: Optional[float] = None
    seed: int = 6107
    # protocol
    settle_time: float = 0.02        # minimum bed-prep time (s)
    settle_v_tol: float = 0.01       # bed-prep stops once max |v| < tol (m/s)
    settle_max_time: float = 0.80    # hard cap on bed prep (s)
    consol_time: float = 0.03
    normal_stress: float = 800.0     # Pa, servo target
    shear_velocity: float = 0.05     # m/s, platen lateral speed
    shear_strain_target: float = 0.40
    wall_mu: float = 0.0             # side walls -- frictionless by default
    floor_mu: Optional[float] = None  # None -> mu (rough floor)
    platen_mu: Optional[float] = None  # None -> mu (rough platen)
    sample_every: int = 25
    diagnostics: bool = False        # record per-sample internal counters
    # m14840: an opt-in steadiness criterion.  ``tau`` in a short shear window
    # is still growing (measured relative drift over the averaged window up to
    # +0.8 at gamma = 0.30), so a mean tau is a window average, not a steady
    # state.  Set ``stationarity_tol`` (e.g. 0.15) to make the validity gate
    # reject a run whose tau drifted by more than that fraction of its mean.
    stationarity_tol: Optional[float] = None
    # load-controlled platen actuator: M a = N_target - N_measured (N, kg, s).
    # A velocity-rate actuator cannot follow the bed (the previous 0.05 m/s cap
    # left the platen in free fall behind a collapsing bed -- see m14721 doc).
    servo_plate_mass_frac: float = 1.0   # M_plate = frac * (bed mass)
    servo_damping_time: float = 5.0e-4   # s, actuator velocity relaxation
    servo_v_limit: float = 0.5           # m/s, |v_z| safety cap
    # m14840: the m14721 actuator damps the platen velocity against the *lab
    # frame*, so its discrete steady state is
    # ``N_measured = N_target - M_plate*v_z*(1-exp(-dt/tau))/dt``: it holds the
    # target load only for a platen at rest and leaks a load offset
    # proportional to the creep velocity it must follow (measured: +3.1..+4.6 %
    # of N_target).  ``servo_mode="bedref"`` damps against the velocity of the
    # bed the platen is riding on instead -- at steady creep the dashpot then
    # carries no force, so the equilibrium moves to N_measured == N_target.
    # ``"drag"`` reproduces the m14721 law bit-for-bit (regression reference).
    servo_mode: str = "bedref"           # "bedref" (zero offset) | "drag" (m14721)

    # ------------------------------------------------------------------ props
    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kr(self) -> float:
        return self.kt * self.radius ** 2

    @property
    def area(self) -> float:
        return self.box_width * self.box_depth

    @property
    def zeta(self) -> float:
        e = max(min(self.restitution, 1.0 - 1e-12), 1e-6)
        ln_e = math.log(e)
        return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)

    @property
    def gamma_pair(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * self.mass)

    @property
    def gamma_wall(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * self.mass)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        period = 2.0 * math.pi * math.sqrt((0.5 * self.mass) / self.kn)
        return period / 20.0

    @property
    def shear_time(self) -> float:
        """Time to reach the target engineering shear strain."""
        return self.shear_strain_target * self.fill_height / self.shear_velocity

    @property
    def shear_rate(self) -> float:
        """Engineering shear rate ``gamma_dot = v / H`` (1/s)."""
        return self.shear_velocity / self.fill_height

    @property
    def floor_mu_value(self) -> float:
        return self.mu if self.floor_mu is None else self.floor_mu

    @property
    def platen_mu_value(self) -> float:
        return self.mu if self.platen_mu is None else self.platen_mu


# ---------------------------------------------------------------------------
# plane-wall contact (numpy)
# ---------------------------------------------------------------------------
class _WallState:
    """Per-particle tangential/rolling spring state for one wall."""

    __slots__ = ("tang", "roll")

    def __init__(self) -> None:
        self.tang: dict[int, np.ndarray] = {}
        self.roll: dict[int, np.ndarray] = {}


def _plane_contact(case: ShearCellCase, pos: np.ndarray, vel: np.ndarray,
                   omega: np.ndarray, state: _WallState, normal: np.ndarray,
                   point: np.ndarray, surf_vel: np.ndarray, mu: float,
                   mu_roll: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contact force/torque from one infinite plane wall.

    ``normal`` points *into* the particle domain, so a particle is in contact
    when ``(pos - point) . normal < radius``.  Returns
    ``(force (n,3), torque (n,3), f_n (n,) >= 0)``.
    """
    n_part = len(pos)
    force = np.zeros((n_part, 3), dtype=np.float64)
    torque = np.zeros((n_part, 3), dtype=np.float64)
    f_norm = np.zeros(n_part, dtype=np.float64)
    r = case.radius
    d = (pos - point) @ normal
    active = d < r
    if not active.any():
        state.tang.clear()
        state.roll.clear()
        return force, torque, f_norm
    idx = np.nonzero(active)[0]
    pen = r - d[idx]
    sv = surf_vel[idx] if surf_vel.ndim == 2 else np.broadcast_to(surf_vel, (n_part, 3))[idx]
    v_rel = vel[idx] - sv
    v_n = v_rel @ normal
    f_n = np.maximum(case.kn * pen - case.gamma_wall * v_n, 0.0)
    f_norm[idx] = f_n

    f = f_n[:, None] * normal[None, :]
    if mu > 0.0:
        v_t = v_rel - v_n[:, None] * normal[None, :]
        tang = np.array([state.tang.get(int(i), np.zeros(3)) for i in idx])
        tang = tang + v_t * dt
        tnorm = np.linalg.norm(tang, axis=1)
        cap = mu * f_n / case.kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        for k, i in enumerate(idx):
            state.tang[int(i)] = tang[k]
        for i in list(state.tang):
            if not active[i]:
                del state.tang[i]
        f_t = -case.kt * tang
        f = f + f_t
        # contact point offset from the particle centre is opposite the normal
        torque[idx] += np.cross(np.broadcast_to(-normal * r, f_t.shape), f_t)
    else:
        state.tang.clear()

    if mu_roll > 0.0:
        spin = omega[idx]
        roll = np.array([state.roll.get(int(i), np.zeros(3)) for i in idx])
        roll = roll + spin * dt
        rnorm = np.linalg.norm(roll, axis=1)
        cap_r = mu_roll * r * f_n / case.kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        for k, i in enumerate(idx):
            state.roll[int(i)] = roll[k]
        torque[idx] += -case.kr * roll
    else:
        state.roll.clear()

    force[idx] = f
    return force, torque, f_norm


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
def _steady(hist: Sequence[float], frac: float = 0.5, n_blocks: int = 10) -> dict:
    """Mean + block standard error over the last ``frac`` of a history."""
    arr = np.asarray(hist, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    head = int(math.floor((1.0 - frac) * arr.size))
    win = arr[head:]
    if win.size < 2:
        return {"mean": float(win.mean()) if win.size else float("nan"),
                "sem": float("nan"), "n": int(win.size)}
    blocks = np.array_split(win, min(n_blocks, win.size))
    means = np.array([b.mean() for b in blocks])
    sem = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
    return {"mean": float(win.mean()), "sem": sem, "n": int(win.size)}


def prepare_bed(case: ShearCellCase) -> dict:
    """Gravity-settle the initial lattice into a quasi-static bed.

    The jittered HCP fill is dilated (bulk packing fraction ~0.3): under gravity
    it collapses by ~25 % in volume at 0.1-0.4 m/s, which is *faster than any
    stress servo can follow* -- that is the mechanism behind the m13798
    honest-FAIL (sigma_n collapsed to 0).  This function prepares the bed once,
    so every normal-stress level of a sweep starts from the **same** bed (a
    controlled variable), and the ~14k preparation steps are paid once, not per
    level.

    Returns ``{"pos", "vel", "omega", "steps", "bed_top", "packing_fraction"}``.
    """
    r = case.radius
    mass = case.mass
    dt = case.timestep
    pos = hcp_positions_in_box(r, case.box_width, case.box_depth, case.fill_height,
                              packing_gap=case.packing_gap, jitter=case.jitter,
                              seed=case.seed, margin=0.2 * r)
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)
    state = ContactState()
    mat = Material3D(radius=r, density=case.density, mu=case.mu,
                     mu_rolling=case.mu_rolling, restitution=case.restitution,
                     kn=case.kn, kt_ratio=case.kt_ratio, gravity=case.gravity)

    floor_n = np.array([0.0, 0.0, 1.0])
    floor_p = np.zeros(3)
    side_walls = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([case.box_width, 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, case.box_depth, 0.0])),
    ]
    floor_state = _WallState()
    side_state = _WallState()
    z_platen = float(pos[:, 2].max()) + 1.02 * r

    n_settle = max(int(round(case.settle_time / dt)), 50)
    n_settle_max = max(n_settle, int(round(case.settle_max_time / dt)))
    quiet = 0
    steps_done = n_settle_max
    for k in range(n_settle_max):
        iu, ju = build_pairs(pos, r)
        force, torque = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)
        force[:, 2] -= mass * case.gravity
        f_f, t_f, _ = _plane_contact(case, pos, vel, omega, floor_state, floor_n,
                                     floor_p, np.zeros(3), case.floor_mu_value,
                                     case.mu_rolling, dt)
        force += f_f
        torque += t_f
        for normal, point in side_walls:
            f_s, t_s, _ = _plane_contact(case, pos, vel, omega, side_state, normal,
                                         point, np.zeros(3), case.wall_mu, 0.0, dt)
            force += f_s
            torque += t_s
        vel = vel + force / mass * dt
        omega = omega + torque / case.inertia * dt
        pos = pos + vel * dt
        z_platen = max(z_platen, float(pos[:, 2].max()) + 1.02 * r)
        if k >= n_settle and k % 100 == 0:
            v_max = float(np.linalg.norm(vel, axis=1).max())
            quiet = quiet + 1 if v_max < case.settle_v_tol else 0
            if quiet >= 5:                      # 500 quiet steps
                steps_done = k
                break
    bed_top = float(pos[:, 2].max())
    area = case.box_width * case.box_depth
    return {"pos": pos, "vel": vel, "omega": omega, "steps": int(steps_done),
            "bed_top": bed_top,
            "packing_fraction": float(n * (4.0 / 3.0 * math.pi * r ** 3)
                                      / (area * max(bed_top + r, 1e-12))),
            "n_particles": n}


def simulate_shear_cell(case: ShearCellCase, bed: Optional[dict] = None) -> dict:
    """Run one direct-shear test at a single normal-stress level.

    ``bed`` is an optional :func:`prepare_bed` result; when given, the test
    starts from that (quasi-static) bed instead of the raw lattice, and the
    gravity-settle phase is skipped.

    Returns the steady-state traction pair and the raw histories.
    """
    r = case.radius
    mass = case.mass
    inertia = case.inertia
    dt = case.timestep
    g = case.gravity

    if bed is None:
        bed = prepare_bed(case)
    pos = np.array(bed["pos"], dtype=np.float64, copy=True)
    vel = np.array(bed["vel"], dtype=np.float64, copy=True)
    omega = np.array(bed["omega"], dtype=np.float64, copy=True)
    n = len(pos)
    state = ContactState()

    mat = Material3D(radius=r, density=case.density, mu=case.mu,
                     mu_rolling=case.mu_rolling, restitution=case.restitution,
                     kn=case.kn, kt_ratio=case.kt_ratio, gravity=g)

    floor_state = _WallState()
    side_state = _WallState()
    platen_state = _WallState()

    floor_n = np.array([0.0, 0.0, 1.0])
    floor_p = np.array([0.0, 0.0, 0.0])
    side_walls = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([case.box_width, 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, case.box_depth, 0.0])),
    ]
    platen_n = np.array([0.0, 0.0, -1.0])
    area = case.area
    n_target = case.normal_stress * area
    plate_mass = max(case.servo_plate_mass_frac * n * mass, 1e-9)

    z_platen = float(pos[:, 2].max()) + 1.02 * r
    vz_platen = 0.0
    vx_platen = 0.0

    def step(drive: bool, servo: bool) -> tuple[float, float, float, int, int]:
        """One velocity-Verlet step.

        Returns ``(tau, sigma, tau_floor, n_contact_platen, n_above_platen)``.
        ``n_above_platen`` counts particles that ended up on the *wrong* side of
        the platen (a plane wall force cannot stop tunnelling) -- diagnostics
        only, it is never averaged into a traction.
        """
        nonlocal pos, vel, omega, z_platen, vz_platen
        iu, ju = build_pairs(pos, r)
        force, torque = contact_forces(pos, vel, omega, iu, ju, state, mat, dt)
        force[:, 2] -= mass * g

        f_f, t_f, fn_floor = _plane_contact(
            case, pos, vel, omega, floor_state, floor_n, floor_p,
            np.zeros(3), case.floor_mu_value, case.mu_rolling, dt)
        force += f_f
        torque += t_f

        for normal, point in side_walls:
            f_s, t_s, _ = _plane_contact(
                case, pos, vel, omega, side_state, normal, point,
                np.zeros(3), case.wall_mu, 0.0, dt)
            force += f_s
            torque += t_s

        surf = np.array([vx_platen, 0.0, vz_platen])
        f_p, t_p, fn_p = _plane_contact(
            case, pos, vel, omega, platen_state, platen_n,
            np.array([0.0, 0.0, z_platen]), surf,
            case.platen_mu_value, case.mu_rolling, dt)
        force += f_p
        torque += t_p

        # ---- servo on the platen normal stress ---------------------------
        n_meas = float(fn_p.sum())
        loaded = fn_p > 0.0
        # Creep velocity of the bed the platen sits on: the mean normal velocity
        # of the particles that actually carry the platen (0 when it is free).
        v_bed = float(vel[loaded, 2].mean()) if loaded.any() else 0.0
        if servo:
            # Load-controlled actuator: the platen is a (fictitious) mass driven
            # by the *force* deficit N_target - N_measured and damped at a short
            # relaxation time.  A steady state therefore requires N_measured ==
            # N_target (zero offset) and the platen tracks the compacting bed at
            # whatever speed the bed demands, instead of lagging behind a fixed
            # velocity cap.
            # z is up: a positive force deficit must accelerate the platen
            # *downwards* (the bed greift nach unten) -- sign fixed in m14721.
            deficit = n_target - n_meas
            vz_platen = vz_platen - deficit / plate_mass * dt
            # Dashpot reference: "drag" damps against the lab frame (m14721),
            # "bedref" against the bed creep velocity (m14840).  The two are the
            # same expression with ref = 0, so "drag" stays bit-identical.
            ref = v_bed if case.servo_mode == "bedref" else 0.0
            vz_platen -= (vz_platen - ref) * (1.0 - math.exp(-dt / case.servo_damping_time))
            vz_platen = float(np.clip(vz_platen, -case.servo_v_limit, case.servo_v_limit))
        else:
            vz_platen = 0.0

        # ---- integrate (symplectic Euler; forces already at t) -----------
        acc = force / mass
        ang_acc = torque / inertia
        vel = vel + acc * dt
        omega = omega + ang_acc * dt
        pos = pos + vel * dt
        z_platen += vz_platen * dt

        # ---- measured tractions ------------------------------------------
        # The platen drags the bed in +x, so the platen *pushes the bed* in +x
        # and the bed's floor reaction is -x: tau = +sum(f_p,x)/A.  The previous
        # revision carried a spurious minus sign here (tau < 0 => mu < 0); it
        # went unnoticed because the platen never carried load (m13798).
        tau = float(f_p[:, 0].sum()) / area       # traction transmitted to the bed
        sigma = n_meas / area
        tau_floor = -float(f_f[:, 0].sum()) / area   # same traction, floor side
        n_contact = int((fn_p > 0.0).sum())
        n_above = int((pos[:, 2] > z_platen).sum())
        return tau, sigma, tau_floor, n_contact, n_above

    diag: dict[str, list] = {k: [] for k in (
        "phase", "step", "sigma", "tau", "platen_z", "bed_top", "bed_mean_z",
        "n_contact", "n_above", "vmax", "ke", "vz")}

    def _sample(phase: str, k: int, sigma: float, tau: float,
                n_contact: int, n_above: int) -> None:
        if not case.diagnostics:
            return
        diag["phase"].append(phase)
        diag["step"].append(int(k))
        diag["sigma"].append(float(sigma))
        diag["tau"].append(float(tau))
        diag["platen_z"].append(float(z_platen))
        diag["bed_top"].append(float(pos[:, 2].max()))
        diag["bed_mean_z"].append(float(pos[:, 2].mean()))
        diag["n_contact"].append(int(n_contact))
        diag["n_above"].append(int(n_above))
        diag["vmax"].append(float(np.linalg.norm(vel, axis=1).max()))
        diag["vz"].append(float(vz_platen))

    # ---- phase 1: bed is already prepared (see prepare_bed) -------------
    n_settle_done = int(bed.get("steps", 0))
    n_settle = n_settle_done
    # park the platen just in contact: the servo only has to take up the
    # consolidation, not chase a detached platen
    z_platen = float(pos[:, 2].max()) + 0.995 * r

    # ---- phase 2: consolidation under the stress servo ------------------
    n_consol = max(int(round(case.consol_time / dt)), 200)
    sigma_rel = float("inf")
    settled = 0
    for consol_step in range(n_consol):
        tau, sigma, _, n_contact, n_above = step(drive=False, servo=True)
        sigma_rel = abs(sigma / case.normal_stress - 1.0) if case.normal_stress > 0 else 0.0
        if settled == 0 and consol_step % case.sample_every == 0:
            _sample("consol", consol_step, sigma, tau, n_contact, n_above)
        settled = settled + 1 if sigma_rel < 0.02 else 0
        if settled >= 100:
            break

    # ---- phase 3: shear -------------------------------------------------
    vx_platen = case.shear_velocity
    n_shear = max(int(round(case.shear_time / dt)), 200)
    tau_hist: list[float] = []
    sig_hist: list[float] = []
    floor_hist: list[float] = []
    disp_hist: list[float] = []
    bed_hist: list[float] = []
    platen_hist: list[float] = []
    disp = 0.0
    for k in range(n_shear):
        tau, sigma, tau_floor, n_contact, n_above = step(drive=True, servo=True)
        disp += case.shear_velocity * dt
        if k % case.sample_every == 0:
            tau_hist.append(tau)
            sig_hist.append(sigma)
            floor_hist.append(tau_floor)
            disp_hist.append(disp / case.fill_height)
            bed_hist.append(float(pos[:, 2].max()))
            platen_hist.append(float(z_platen))
            _sample("shear", k, sigma, tau, n_contact, n_above)

    tau_s = _steady(tau_hist)
    sig_s = _steady(sig_hist)
    # m14840: the *actual* shear rate is v divided by the height of the bed the
    # platen sits on, which after compaction is ~1.7x smaller than the nominal
    # ``fill_height`` the case advertises.  Reporting the nominal value
    # under-states gamma_dot (and hence the inertial number) by that ratio.
    bed_s = _steady(bed_hist)
    bed_height = float(bed_s["mean"])
    shear_rate_actual = (case.shear_velocity / bed_height) if bed_height > 0 else float("nan")
    inertial = inertial_number(shear_rate_actual, r, case.density, sig_s["mean"])
    floor_s = _steady(floor_hist)
    mu_macro = tau_s["mean"] / sig_s["mean"] if sig_s["mean"] > 0 else float("nan")
    phi_deg = math.degrees(math.atan(mu_macro)) if mu_macro == mu_macro else float("nan")
    # stationarity diagnostic: linear drift fit of tau over the steady window
    win = np.asarray(tau_hist[int(math.floor(0.5 * len(tau_hist))):], dtype=float)
    if win.size >= 3:
        gam = np.arange(win.size, dtype=float)
        coef = np.polyfit(gam, win, 1)
        pred = np.polyval(coef, gam)
        ss_res = float(((win - pred) ** 2).sum())
        ss_tot = float(((win - win.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        drift_rel = float(coef[0] * win.size / max(abs(win.mean()), 1e-30))
    else:
        r2, drift_rel = float("nan"), float("nan")

    # ---- validity gate ---------------------------------------------------
    # A friction coefficient is only defined when the platen actually carries
    # the target normal stress.  The gate is explicit so that a detached-platen
    # run can never be silently averaged into a "friction angle".
    valid, invalid_reason = _validity_gate(sig_s["mean"], tau_s["mean"],
                                           case.normal_stress,
                                           drift_rel=drift_rel,
                                           stationarity_tol=case.stationarity_tol)

    return {
        "valid": valid,
        "invalid_reason": invalid_reason,
        "normal_stress_applied": case.normal_stress,
        "normal_stress_measured": sig_s["mean"],
        "normal_stress_sem": sig_s["sem"],
        "normal_stress_rel_err": float(sig_s["mean"] / case.normal_stress - 1.0),
        "shear_stress": tau_s["mean"],
        "shear_stress_sem": tau_s["sem"],
        "floor_shear_stress": floor_s["mean"],
        "tau_platen_vs_floor_rel": float(abs(floor_s["mean"] / tau_s["mean"] - 1.0))
        if tau_s["mean"] > 0 else float("nan"),
        "mu_macro": mu_macro,
        "friction_angle_deg": phi_deg,
        # m14840 rate bookkeeping (measured bed height, not the nominal fill one)
        "bed_height_mean": bed_height,
        "bed_height_sem": bed_s["sem"],
        "shear_rate_nominal": case.shear_rate,
        "shear_rate_actual": shear_rate_actual,
        "inertial_number": inertial,
        "r_squared": r2,
        "shear_stress_drift_rel": drift_rel,
        "consolidation_sigma_rel_err": sigma_rel,
        "shear_strain_history": disp_hist,
        "shear_stress_history": tau_hist,
        "normal_stress_history": sig_hist,
        "floor_shear_history": floor_hist,
        "bed_top_history": bed_hist,
        "platen_z_history": platen_hist,
        "n_particles": n,
        "diagnostics": diag if case.diagnostics else None,
        "total_steps": n_settle_done + n_consol + n_shear,
        "bed_prep_steps": n_settle_done,
        "timestep": dt,
        "shear_rate": case.shear_rate,
        # Nominal (pre-compaction) reference value, kept under a *distinct* key.
        # This entry used to be spelled ``inertial_number`` as well, so the dict
        # literal silently overwrote the measured-bed-height value above and
        # every consumer of ``results["inertial_number"]`` got the nominal
        # (gamma_dot_nominal, sigma_target) number -- i.e. the m14840 rate
        # bookkeeping fix was dead at the result level.
        "inertial_number_nominal": inertial_number(
            case.shear_rate, r, case.density, case.normal_stress),
    }


def _validity_gate(sigma_mean: float, tau_mean: float,
                   normal_stress: float,
                   drift_rel: float = float("nan"),
                   stationarity_tol: Optional[float] = None) -> tuple[bool, str]:
    """Is a friction coefficient defined for this run?

    ``mu = tau/sigma`` is only meaningful when the platen carried a real normal
    stress (at least half the servo target) and the bed resisted with a
    positive traction.  The gate never returns a number for an unloaded run --
    that is the difference between "mu is undefined" and "mu = 0".

    m14840 adds an optional *steadiness* leg: when ``stationarity_tol`` is set,
    a run whose shear traction drifts by more than that fraction of its mean
    across the averaged window is rejected as well, so a window average is not
    silently reported as a friction coefficient.
    """
    if not (sigma_mean > 0.5 * normal_stress):
        return False, (
            "platen lost contact or never loaded: measured sigma_n = %.4g Pa vs "
            "target %.4g Pa (mu is undefined, not 0)" % (sigma_mean, normal_stress))
    if not tau_mean > 0.0:
        return False, (
            "no positive traction on the platen: tau = %.4g Pa (mu is undefined, "
            "not 0)" % tau_mean)
    if stationarity_tol is not None and drift_rel == drift_rel:
        if abs(drift_rel) > stationarity_tol:
            return False, (
                "shear traction not steady over the averaged window: relative "
                "drift %.3f > tolerance %.3f (window mean is not a steady-state "
                "mu; shear further before averaging)" % (drift_rel, stationarity_tol))
    return True, ""


def _fit_mohr_coulomb(sigmas: Sequence[float], taus: Sequence[float]) -> dict:
    """Least-squares ``tau = c + sigma tan(phi)`` over *distinct* stress levels."""
    s = np.asarray(sigmas, dtype=float)
    t = np.asarray(taus, dtype=float)
    if s.size < 2:
        raise ValueError("Mohr-Coulomb needs >= 2 normal-stress levels; a single "
                         "level only gives the point value mu = tau/sigma")
    if np.ptp(s) <= 1e-12:
        raise ValueError(f"normal-stress levels must be distinct, got {s.tolist()}")
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    tan_phi, cohesion = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mu_levels = t / s
    return {
        "tan_phi": tan_phi,
        "friction_angle_deg": math.degrees(math.atan(tan_phi)),
        "cohesion_pa": cohesion,
        "r_squared": float(r2),
        "mu_per_level": [float(x) for x in mu_levels],
        "mu_level_spread_rel": float(np.std(mu_levels) / np.mean(mu_levels))
        if np.mean(mu_levels) > 0 else float("nan"),
        "cohesion_over_mean_tau": float(abs(cohesion / t.mean())) if t.mean() else float("nan"),
    }


def measure_macro_friction(case: ShearCellCase,
                           normal_stresses: Sequence[float] = (400.0, 800.0, 1600.0, 3200.0),
                           **overrides) -> dict:
    """Run the shear cell at several normal stresses and fit the envelope.

    Returns the per-level ``(sigma, tau, mu)`` table, the Mohr-Coulomb fit and
    the stress-independence diagnostic.  ``overrides`` are applied to ``case``
    (e.g. ``seed=6108``).
    """
    base = replace(case, **overrides) if overrides else case
    bed = prepare_bed(base)     # identical bed at every level (controlled var)
    levels = []
    for sigma in normal_stresses:
        res = simulate_shear_cell(replace(base, normal_stress=float(sigma)), bed=bed)
        levels.append({
            "sigma_target_pa": float(sigma),
            "sigma_pa": res["normal_stress_measured"],
            "sigma_rel_err": res["normal_stress_rel_err"],
            "tau_pa": res["shear_stress"],
            "tau_sem_pa": res["shear_stress_sem"],
            "mu": res["mu_macro"],
            "tau_floor_pa": res["floor_shear_stress"],
            "tau_platen_vs_floor_rel": res["tau_platen_vs_floor_rel"],
            "n_particles": res["n_particles"],
            "total_steps": res["total_steps"],
            "valid": res["valid"],
        })
    invalid = [l["sigma_target_pa"] for l in levels if not l["valid"]]
    if invalid:
        # never report an envelope assembled from levels where the platen was
        # unloaded: that is how a NaN "friction angle" used to enter reports
        return {"valid": False,
                "invalid_levels_pa": invalid,
                "reason": "platen did not hold sigma_n at levels %s; refusing to fit "
                          "a Mohr-Coulomb envelope" % invalid,
                "levels": levels}
    fit = _fit_mohr_coulomb([l["sigma_pa"] for l in levels], [l["tau_pa"] for l in levels])
    fit.update({
        "levels": levels,
        "bed": {"steps": bed["steps"], "bed_top": bed["bed_top"],
                "packing_fraction": bed["packing_fraction"],
                "n_particles": bed["n_particles"]},
        "case": {
            "radius": base.radius, "density": base.density, "mu": base.mu,
            "mu_rolling": base.mu_rolling, "kn": base.kn, "fill_height": base.fill_height,
            "shear_velocity": base.shear_velocity, "shear_rate": base.shear_rate,
            "shear_strain_target": base.shear_strain_target, "seed": base.seed,
            "species": base.species,
        },
    })
    return fit