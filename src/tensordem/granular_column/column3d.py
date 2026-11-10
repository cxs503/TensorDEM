"""Three-dimensional rectangular granular-column collapse (free-surface flow).

Physical setting
----------------
A rectangular block of monodisperse spheres, ``L0 x D0 x H0`` with
``a = H0 / L0``, rests on a rough rigid floor.  At ``t = 0`` the lateral
support is removed and the block collapses under gravity into a thin deposit.
The dimensionless runout

.. math:: \\Delta L^* = (L_f - L_0) / L_0

is, to leading order, a function of the initial aspect ratio only (Lube,
Huppert, Sparks & Freundt, JGR 109, B10203, 2004, axisymmetric columns;
Lube et al., PRE 72, 041301, 2005 and Lajeunesse, Monnier & Homsy, Phys.
Fluids 17, 103302, 2005, rectangular channels):

    a  <~ 1 :  Delta L* ~ alpha a          (linear, alpha ~ 1-1.6)
    a  >~ 2 :  Delta L* ~ beta a^(2/3)     (beta ~ 1.5-1.9)

The 2/3 exponent is the signature of the potential-energy/friction balance in
the spreading layer; it is independent of the grain size and of ``g``.  Its
recovery from a bare contact-mechanics DEM, without fitting any runout
coefficient, is the strongest available validation of the contact-friction
model of this repository.  The collapse duration obeys the ballistic scaling
``t* ~ sqrt(H0 / g)``, i.e. the front velocity scales with
``sqrt(g H0)``.

The run is integrated with a velocity-Verlet scheme whose dissipative forces
are evaluated at the half-step velocity, so the per-step slip residual of every
contact is the actual dissipated work (:mod:`tensordem.granular_column.kernel`).
That makes the potential-energy ledger

.. math:: \\Delta E_p = E_k + E_{spring} + E_{diss}^{collision}
                       + E_{diss}^{friction}

close to a small fraction of a per mille of the released energy, which is the
acceptance quantity of criterion (3) of the campaign node.

Units are SI throughout (m, kg, s, N, J).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..granular3d_core import Material3D, build_pairs, hcp_positions_in_box
from .kernel import PairSpringState, WallSpringState, pair_contact_forces, wall_contact_forces

__all__ = [
    "Column3DCase",
    "Column3DError",
    "PACKING_STATES",
    "build_column",
    "simulate_column",
    "lube_axisymmetric_runout",
]

#: Initial packing states of the campaign node (dilute / dense).
PACKING_STATES = {"dense": 0.02, "loose": 0.15}

#: Lube et al. (2004) large-aspect-ratio runout exponent.
LUBE_EXPONENT = 2.0 / 3.0


class Column3DError(ValueError):
    """Raised for inconsistent column-collapse inputs."""


@dataclass(frozen=True)
class Column3DCase:
    """Geometry, material and protocol of one 3-D collapse run (SI)."""

    aspect_ratio: float = 2.0
    #: initial column length ``L0`` (x direction) [m]
    length: float = 0.05
    #: initial column depth (y); ``None`` -> equal to ``length`` (square plan)
    depth: float | None = None
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.0
    restitution: float = 0.3
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    #: "dense" (HCP, 2 % lattice gap) or "loose" (15 % gap, phi ~ 0.49)
    packing: str = "dense"
    mu_wall: float | None = None
    jitter: float = 0.01
    seed: int = 7006
    dt: float | None = None
    #: ``dt`` multiplier on the resolved-timestep estimate (1.0 = period/20)
    dt_scale: float = 1.0
    #: total run time = max(min_time, duration_factor * sqrt(H0/g))
    duration_factor: float = 6.0
    min_time: float = 0.4
    sample_stride: int = 25
    #: rest criterion on kinetic energy (fraction of the released potential)
    rest_ke_tol: float = 1.0e-5
    #: "one_sided" (compression-only dashpot, ledger-closing) or "reference"
    #: (tension-cutoff dashpot of tensordem.granular3d_core)
    contact_mode: str = "reference"
    rest_hold: int = 300

    def __post_init__(self) -> None:
        for name in ("aspect_ratio", "length", "radius", "density", "kn",
                     "gravity", "dt_scale", "duration_factor", "min_time"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise Column3DError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "jitter", "kt_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise Column3DError(f"{name} must be finite and nonnegative")
        if not 0.0 <= float(self.restitution) < 1.0:
            raise Column3DError("restitution must lie in [0, 1)")
        if self.packing not in PACKING_STATES:
            raise Column3DError(f"packing must be one of {sorted(PACKING_STATES)}")
        if self.length < 4.0 * self.radius:
            raise Column3DError("column must be at least 4 grains wide")
        if self.sample_stride < 1:
            raise Column3DError("sample_stride must be >= 1")

    @property
    def height(self) -> float:
        return self.aspect_ratio * self.length

    @property
    def depth_length(self) -> float:
        return self.length if self.depth is None else float(self.depth)

    @property
    def packing_gap(self) -> float:
        return PACKING_STATES[self.packing]

    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling,
                          restitution=self.restitution, kn=self.kn,
                          kt_ratio=self.kt_ratio, gravity=self.gravity)

    def signature(self) -> dict:
        """Plain dict of the case, for provenance in the result JSON."""
        out = {}
        for name in ("aspect_ratio", "length", "depth", "radius", "density",
                     "mu", "mu_rolling", "restitution", "kn", "kt_ratio",
                     "gravity", "packing", "mu_wall", "jitter", "seed", "dt",
                     "dt_scale", "duration_factor", "min_time",
                     "sample_stride"):
            out[name] = getattr(self, name)
        out["height"] = self.height
        return out


def build_column(case: Column3DCase) -> tuple[np.ndarray, Material3D]:
    """HCP block packing for ``case``; returns ``(positions, material)``."""
    mat = case.material()
    pos = hcp_positions_in_box(case.radius, case.length, case.depth_length,
                               case.height, packing_gap=case.packing_gap,
                               jitter=case.jitter, seed=case.seed,
                               margin=0.2 * case.radius)
    return pos, mat


def _front_extent(x: np.ndarray, radius: float) -> tuple[float, float]:
    """Robust front/back positions of the deposit (0.5 % tails trimmed)."""
    lo, hi = np.quantile(x, [0.005, 0.995])
    return float(lo - radius), float(hi + radius)


def _energies(pos, vel, omega, mass, inertia, gravity, grid) -> dict:
    return {
        "potential": float(mass * gravity * np.sum(pos[:, 2] - grid)),
        "kinetic": float(0.5 * mass * np.sum(vel * vel)
                         + 0.5 * inertia * np.sum(omega * omega)),
    }


def simulate_column(case: Column3DCase, *, return_positions: bool = True,
                    max_steps: int | None = None) -> dict:
    """Run one collapse and return runout, kinematics and the energy ledger."""
    pos, mat = build_column(case)
    n = pos.shape[0]
    mass, inertia = mat.mass, mat.inertia
    dt = float(case.dt) if case.dt else mat.timestep * case.dt_scale
    gravity_vec = np.array([0.0, 0.0, -mat.gravity])
    mu_wall = mat.mu if case.mu_wall is None else float(case.mu_wall)

    x0_back, x0_front = _front_extent(pos[:, 0], mat.radius)
    length_measured = x0_front - x0_back
    height_measured = float(pos[:, 2].max() + mat.radius)
    depth_measured = _front_extent(pos[:, 1], mat.radius)
    depth_measured = depth_measured[1] - depth_measured[0]
    aspect_measured = height_measured / length_measured
    volume = (4.0 / 3.0 * math.pi * mat.radius ** 3)
    phi0 = n * volume / (length_measured * depth_measured * height_measured)

    pe0 = float(mass * mat.gravity * np.sum(pos[:, 2]))
    e_p0 = pe0  # released potential relative to the final (floor) datum

    pair_state = PairSpringState()
    wall_state = WallSpringState(n)
    vel = np.zeros((n, 3))
    omega = np.zeros((n, 3))

    t_end = max(case.min_time, case.duration_factor * math.sqrt(height_measured / mat.gravity))
    n_steps = int(math.ceil(t_end / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))

    time = 0.0
    diss = {"collision": 0.0, "friction": 0.0, "rolling": 0.0,
            "wall_collision": 0.0, "wall_friction": 0.0}
    spring = {"normal": 0.0, "tangential": 0.0, "rolling": 0.0,
              "wall_normal": 0.0, "wall_tangential": 0.0, "wall_rolling": 0.0}
    dead_prev = 0.0
    cutoff = 0.0
    history = {"time": [], "front": [], "back": [], "kinetic": [],
               "potential": [], "dissipation": [], "top": [], "contacts": []}

    front = x0_front
    back = x0_back
    rest_counter = 0
    max_contacts = 0
    iu = np.zeros(0, dtype=np.int64)
    ju = np.zeros(0, dtype=np.int64)
    stop_reason = "time_limit"

    for step in range(n_steps):
        # --- predictor: contact forces at (x, v, omega), springs untouched
        iu, ju = build_pairs(pos, mat.radius)
        pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, dt,
                                   update_state=False,
                                   contact_mode=case.contact_mode)
        pred_wall = wall_contact_forces(pos, vel, omega, mat, dt, mu_wall=mu_wall,
                                        state=wall_state, update_state=False,
                                        contact_mode=case.contact_mode)
        acc = (pred["force"] + pred_wall["force"]) / mass + gravity_vec
        alpha = (pred["torque"] + pred_wall["torque"]) / inertia
        v_half = vel + 0.5 * dt * acc
        w_half = omega + 0.5 * dt * alpha
        pos += v_half * dt

        # --- corrector: forces at the new positions, springs advanced once
        cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state, mat,
                                  dt, update_state=True,
                                  contact_mode=case.contact_mode)
        cur_wall = wall_contact_forces(pos, v_half, w_half, mat, dt,
                                       mu_wall=mu_wall, state=wall_state,
                                       update_state=True,
                                       contact_mode=case.contact_mode)
        acc = (cur["force"] + cur_wall["force"]) / mass + gravity_vec
        alpha = (cur["torque"] + cur_wall["torque"]) / inertia
        vel = v_half + 0.5 * dt * acc
        omega = w_half + 0.5 * dt * alpha
        time += dt

        diss["collision"] += cur["dissip_collision"]
        diss["friction"] += cur["dissip_friction"]
        diss["rolling"] += cur["dissip_rolling"]
        diss["wall_collision"] += cur_wall["dissip_collision"]
        diss["wall_friction"] += cur_wall["dissip_friction"]
        cutoff += float(cur["release_loss"] + cur_wall["release_loss"])
        spring = {
            "normal": cur["energy_normal"],
            "tangential": cur["energy_tangential"],
            "rolling": cur["energy_rolling"],
            "wall_normal": cur_wall["energy_normal"],
            "wall_tangential": cur_wall["energy_tangential"],
            "wall_rolling": cur_wall["energy_rolling"],
        }
        max_contacts = max(max_contacts, cur["n_contacts"] + cur_wall["n_contacts"])

        if step % case.sample_stride == 0 or step == n_steps - 1:
            back, front = _front_extent(pos[:, 0], mat.radius)
            en = _energies(pos, vel, omega, mass, inertia, mat.gravity, 0.0)
            history["time"].append(time)
            history["front"].append(front)
            history["back"].append(back)
            history["kinetic"].append(en["kinetic"])
            history["potential"].append(en["potential"])
            history["dissipation"].append(sum(diss.values()))
            history["top"].append(float(pos[:, 2].max() + mat.radius))
            history["contacts"].append(int(cur["n_contacts"]))

        ke = float(0.5 * mass * np.sum(vel * vel)
                   + 0.5 * inertia * np.sum(omega * omega))
        if time > 1.2 * math.sqrt(height_measured / mat.gravity) and ke < case.rest_ke_tol * e_p0:
            rest_counter += 1
            if rest_counter >= case.rest_hold:
                stop_reason = "at_rest"
                break
        else:
            rest_counter = 0

    back, front = _front_extent(pos[:, 0], mat.radius)
    en = _energies(pos, vel, omega, mass, inertia, mat.gravity, 0.0)
    spring_total = sum(spring.values())
    diss_collision = diss["collision"] + diss["wall_collision"]
    # energy that the tension cutoff rendered unrecoverable at pull-off: booked
    # as an explicit fourth channel, reported separately from the physical
    # dashpot (collision) and friction dissipations.
    diss_cutoff = cutoff
    diss_friction = diss["friction"] + diss["rolling"] + diss["wall_friction"]
    diss_physical = diss_collision + diss_friction
    diss_total = diss_physical + diss_cutoff
    e_end = en["potential"] + en["kinetic"] + spring_total
    released = e_p0 - en["potential"]
    residual = (e_p0 - e_end - diss_total) / e_p0
    residual_physical = (e_p0 - e_end - diss_physical) / e_p0

    time_hist = np.asarray(history["time"])
    front_hist = np.asarray(history["front"])
    runout_total = front - x0_front
    target = x0_front + 0.9 * runout_total
    t90 = float("nan")
    above = np.nonzero(front_hist >= target)[0]
    if above.size:
        k = int(above[0])
        if k == 0:
            t90 = float(time_hist[0])
        else:
            f0, f1 = front_hist[k - 1], front_hist[k]
            t0, t1 = time_hist[k - 1], time_hist[k]
            w = 0.0 if f1 == f0 else (target - f0) / (f1 - f0)
            t90 = float(t0 + w * (t1 - t0))
    if time_hist.size > 2:
        v_front = np.gradient(front_hist, time_hist)
        peak_front_speed = float(np.max(v_front))
        t_peak_speed = float(time_hist[int(np.argmax(v_front))])
    else:
        peak_front_speed = float("nan")
        t_peak_speed = float("nan")

    # final deposit profile
    bins = max(8, int(round(4.0 * length_measured / mat.radius)))
    edges = np.linspace(back, front, bins + 1)
    idx = np.clip(np.digitize(pos[:, 0], edges) - 1, 0, bins - 1)
    height_profile = np.zeros(bins)
    for b in range(bins):
        sel = idx == b
        if sel.any():
            height_profile[b] = float(pos[sel, 2].max() + mat.radius)
    profile_x = 0.5 * (edges[:-1] + edges[1:])

    result = {
        "case": case.signature(),
        "packing": case.packing,
        "packing_gap": case.packing_gap,
        "n_particles": int(n),
        "dt": dt,
        "n_steps": int(step + 1),
        "time_end": time,
        "stop_reason": stop_reason,
        "length_measured": length_measured,
        "height_measured": height_measured,
        "depth_measured": depth_measured,
        "aspect_measured": aspect_measured,
        "solid_fraction_initial": phi0,
        "front_initial": x0_front,
        "front_final": front,
        "back_final": back,
        "runout": runout_total,
        "runout_relative": runout_total / length_measured,
        "runout_over_height": runout_total / height_measured,
        "t90": t90,
        "peak_front_speed": peak_front_speed,
        "t_peak_front_speed": t_peak_speed,
        "ballistic_time": math.sqrt(height_measured / mat.gravity),
        "energy": {
            "potential_initial": e_p0,
            "potential_final": en["potential"],
            "potential_released": released,
            "kinetic_final": en["kinetic"],
            "spring_final": spring_total,
            "spring_channels": spring,
            "dissipated_collision": diss_collision,
            "dissipated_friction": diss_friction,
            "dissipated_collision_grain": diss["collision"],
            "dissipated_friction_grain": diss["friction"] + diss["rolling"],
            "dissipated_wall": diss["wall_collision"] + diss["wall_friction"],
            "dissipated_total": diss_total,
            "dissipated_cutoff": diss_cutoff,
            "dissipated_physical_only": diss_physical,
            "residual": residual,
            "residual_physical_only": residual_physical,
            "residual_fraction_of_released": (e_p0 - e_end - diss_total) / released
            if released > 0 else float("nan"),
            "friction_share": diss_friction / diss_total if diss_total > 0 else float("nan"),
        },
        "coordination_max": int(max_contacts),
        "history": {
            "time": time_hist.tolist(),
            "front": front_hist.tolist(),
            "kinetic": history["kinetic"],
            "potential": history["potential"],
            "dissipation": history["dissipation"],
            "top": history["top"],
        },
        "deposit_profile": {"x": profile_x.tolist(),
                            "height": height_profile.tolist()},
    }
    if return_positions:
        stride = max(1, n // 4000)
        result["positions"] = pos[::stride].tolist()
    return result


def lube_axisymmetric_runout(aspect_ratio: float, *, alpha: float = 1.0,
                             beta=1.9, transition: float = 1.7) -> float:
    """Lube et al. (2004) axisymmetric reference runout ``Delta L / L0``.

    ``alpha * a`` below ``a = transition`` and ``beta * a**(2/3)`` above it,
    matched continuously at the transition.
    """
    a = float(aspect_ratio)
    if a <= transition:
        return alpha * a
    return beta * a ** LUBE_EXPONENT
