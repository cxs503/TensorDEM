"""Axisymmetric granular-column collapse on a rough rigid floor (3-D DEM).

Case: **CASE-COLUMN-COLLAPSE** (算例池, owner 2026-11-12).

Reference experiment
--------------------
Lube, G., Huppert, H. E., Sparks, R. S. J. & Freundt, A. (2004),
*Axisymmetric collapses of granular columns*, Journal of Fluid Mechanics
**508**, 175-199, doi:10.1017/S0022112004009036.  The relations below are the
ones stated in that paper's abstract (quote, with ``r_inf`` written ``r`` in the
typeset abstract)::

    "For a < 1.7, the measured final runout radius, r_inf, is related to the
     initial radius by r_inf = r_i(1 + 1.24a); while for 1.7 < a the
     corresponding relationship is r_inf = r_i(1 + 1.6a^(1/2)).  The time,
     t_inf, taken for the grains to reach r_inf is given by
     t_inf = 3(h_i/g)^(1/2) = 3(r_i/g)^(1/2) a^(1/2), where g is the
     gravitational acceleration."

with ``a = h_i / r_i`` the initial aspect ratio of the cylindrical column.  The
targets used by the benchmark are **evaluated from those formulas at run time**
(:func:`lube_2004_runout_ratio`, :func:`lube_2004_emplacement_time`); no target
number is typed by hand anywhere in this module or in the runner.

Geometry and integration
------------------------
A cylindrical block of monodisperse spheres, radius ``r_i`` and height
``h_i = a r_i``, is packed on an HCP lattice
(:func:`tensordem.granular3d_core.hcp_positions_in_cylinder`) and released at
``t = 0`` (the lateral support of the experiment is removed).  Contacts are the
shared 3-D kernel of :mod:`tensordem.granular_column.kernel` (linear
spring-dashpot normal force, Cundall-Strack tangential spring saturated at
``mu |F_n|``, rolling spring saturated at ``mu_r R |F_n|``; floor contact from
:func:`~tensordem.granular_column.kernel.wall_contact_forces`).  Integration is
velocity Verlet with the dissipative forces evaluated at the half-step
velocity, exactly as in :func:`tensordem.granular_column.column3d.simulate_column`;
only the geometry (cylinder instead of box) and the runout metric (radial
extent instead of the ``x`` front) differ, which is what the reference
experiment requires.

Runout metric
-------------
``r_max`` is the outermost grain surface, ``r_p995`` the 99.5 % radial
quantile plus one grain radius (the same 0.5 % tail trim as the rectangular
benchmark).  The primary metric is ``r_p995``; the sensitivity of the verdict
to that choice is reported with the result.

Units are SI (m, kg, s, N, J).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..granular3d_core import Material3D, build_pairs, hcp_positions_in_cylinder
from .inertial_friction import InertialFriction
from .kernel import (PairSpringState, WallSpringState, pair_contact_forces,
                     wall_contact_forces)

__all__ = [
    "LUBE2004_DOI",
    "LUBE2004_REGIME_BREAK",
    "LUBE2004_SOURCE",
    "AxisymCase",
    "AxisymError",
    "build_axisymmetric_column",
    "lube_2004_emplacement_time",
    "lube_2004_runout_ratio",
    "radial_extent",
    "simulate_axisymmetric_column",
    "verdict",
]

#: DOI of the reference experiment (JFM 508, 175-199, 2004).
LUBE2004_DOI = "10.1017/S0022112004009036"
#: Aspect ratio separating the linear and the inertial runout regimes.
LUBE2004_REGIME_BREAK = 1.7
#: Linear-regime prefactor: ``r_inf / r_i = 1 + 1.24 a`` for ``a < 1.7``.
LUBE2004_LINEAR_PREFACTOR = 1.24
#: Inertial-regime prefactor: ``r_inf / r_i = 1 + 1.6 a^(1/2)`` for ``1.7 < a``.
LUBE2004_INERTIAL_PREFACTOR = 1.6
#: ``t_inf = 3 (h_i/g)^(1/2)``.
LUBE2004_TIME_PREFACTOR = 3.0
#: Verbatim provenance string (paper, volume, page, sentence used).
LUBE2004_SOURCE = (
    "Lube, Huppert, Sparks & Freundt (2004) JFM 508, 175-199, "
    "doi:10.1017/S0022112004009036, abstract: "
    "r_inf = r_i(1 + 1.24a) for a < 1.7; r_inf = r_i(1 + 1.6a^(1/2)) for "
    "1.7 < a; t_inf = 3(h_i/g)^(1/2) = 3(r_i/g)^(1/2)a^(1/2)"
)


class AxisymError(ValueError):
    """Raised for inconsistent axisymmetric-column inputs."""


def lube_2004_runout_ratio(aspect_ratio: float) -> float:
    """``r_inf / r_i`` of Lube et al. (2004), evaluated from the abstract."""
    a = float(aspect_ratio)
    if not math.isfinite(a) or a <= 0.0:
        raise AxisymError("aspect ratio must be finite and positive")
    if a < LUBE2004_REGIME_BREAK:
        return 1.0 + LUBE2004_LINEAR_PREFACTOR * a
    return 1.0 + LUBE2004_INERTIAL_PREFACTOR * math.sqrt(a)


def lube_2004_emplacement_time(aspect_ratio: float, column_radius: float,
                               gravity: float = 9.81) -> float:
    """``t_inf = 3 (h_i/g)^(1/2)`` with ``h_i = a r_i`` [s]."""
    a = float(aspect_ratio)
    if not math.isfinite(a) or a <= 0.0:
        raise AxisymError("aspect ratio must be finite and positive")
    if column_radius <= 0.0 or gravity <= 0.0:
        raise AxisymError("column_radius and gravity must be positive")
    return LUBE2004_TIME_PREFACTOR * math.sqrt(a * float(column_radius) / float(gravity))


@dataclass(frozen=True)
class AxisymCase:
    """Geometry, material and protocol of one axisymmetric collapse run (SI)."""

    #: ``a = h_i / r_i``
    aspect_ratio: float = 2.0
    #: grains per column radius (resolution): ``r_i = grains_per_radius * 2 r``
    grains_per_radius: float = 12.0
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.0
    #: optional ``(mu_s, mu_2, i_0)`` -> rate-dependent ``mu(I_c)`` contact
    #: friction (:class:`~.inertial_friction.InertialFriction`) instead of the
    #: constant ``mu``; ``None`` keeps the constant-friction law.
    mu_inertial: tuple[float, float, float] | None = None
    #: floor (plane) friction coefficient; ``None`` -> the grain-grain ``mu``.
    #: Lube et al. (2004) used a *glued* (fully rough) base, which a plane wall
    #: model only reproduces if ``mu_wall`` is at least the grain friction.
    mu_wall: float | None = None
    restitution: float = 0.5
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    gravity: float = 9.81
    #: lattice gap of the initial HCP packing (0.02 -> phi ~ 0.60)
    packing_gap: float = 0.02
    jitter: float = 0.0
    seed: int = 12345
    dt: float | None = None
    dt_scale: float = 1.0
    #: total run time = max(min_time, duration_factor * sqrt(h_i/g))
    duration_factor: float = 6.0
    min_time: float = 0.3
    sample_stride: int = 20
    rest_ke_tol: float = 1.0e-5
    rest_hold: int = 160
    #: fraction of the final runout that defines the emplacement time
    t_reach_fraction: float = 0.99
    contact_mode: str = "reference"
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("aspect_ratio", "grains_per_radius", "radius", "density",
                     "kn", "kt_ratio", "gravity", "duration_factor", "min_time"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise AxisymError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "jitter", "packing_gap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise AxisymError(f"{name} must be finite and nonnegative")
        if not 0.0 <= float(self.restitution) < 1.0:
            raise AxisymError("restitution must lie in [0, 1)")
        if not 0.0 < float(self.t_reach_fraction) <= 1.0:
            raise AxisymError("t_reach_fraction must lie in (0, 1]")
        if self.mu_wall is not None:
            wall = float(self.mu_wall)
            if not math.isfinite(wall) or wall < 0.0:
                raise AxisymError("mu_wall must be finite and nonnegative")
        if self.mu_inertial is not None:
            values = tuple(float(v) for v in self.mu_inertial)
            if len(values) != 3:
                raise AxisymError("mu_inertial must be a (mu_s, mu_2, i_0) triple")
            InertialFriction(*values)  # validates the triple

    @property
    def column_radius(self) -> float:
        return float(self.grains_per_radius) * 2.0 * float(self.radius)

    @property
    def column_height(self) -> float:
        return float(self.aspect_ratio) * self.column_radius

    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling, restitution=self.restitution,
                          kn=self.kn, kt_ratio=self.kt_ratio, gravity=self.gravity)

    def signature(self) -> dict:
        return {
            "case": "CASE-COLUMN-COLLAPSE",
            "reference": "Lube et al. 2004 JFM 508:175-199",
            "doi": LUBE2004_DOI,
            "aspect_ratio": float(self.aspect_ratio),
            "grains_per_radius": float(self.grains_per_radius),
            "radius": float(self.radius),
            "column_radius": self.column_radius,
            "column_height": self.column_height,
            "density": float(self.density),
            "mu": float(self.mu),
            "mu_rolling": float(self.mu_rolling),
            "mu_inertial": (None if self.mu_inertial is None
                            else [float(v) for v in self.mu_inertial]),
            "mu_law": ("constant" if self.mu_inertial is None
                       else "mu(I_c) Jop et al. (2006)"),
            "mu_wall": (None if self.mu_wall is None else float(self.mu_wall)),
            "restitution": float(self.restitution),
            "kn": float(self.kn),
            "kt_ratio": float(self.kt_ratio),
            "gravity": float(self.gravity),
            "packing_gap": float(self.packing_gap),
            "jitter": float(self.jitter),
            "seed": int(self.seed),
            "dt_scale": float(self.dt_scale),
            "duration_factor": float(self.duration_factor),
            "min_time": float(self.min_time),
            "t_reach_fraction": float(self.t_reach_fraction),
            "contact_mode": self.contact_mode,
            "label": self.label,
        }


def build_axisymmetric_column(case: AxisymCase) -> tuple[np.ndarray, Material3D]:
    """HCP cylindrical packing for ``case``; returns ``(positions, material)``."""
    pos = hcp_positions_in_cylinder(case.radius, case.column_radius,
                                    case.column_height,
                                    packing_gap=case.packing_gap,
                                    jitter=case.jitter, seed=case.seed)
    return pos, case.material()


def radial_extent(pos: np.ndarray, radius: float) -> dict:
    """Radial extent of a deposit: 99.5 % quantile (+r) and outermost grain."""
    rr = np.hypot(pos[:, 0], pos[:, 1])
    return {"p995": float(np.quantile(rr, 0.995)) + radius,
            "max": float(rr.max()) + radius,
            "mean": float(rr.mean())}


def _time_to_reach(time_hist: np.ndarray, front_hist: np.ndarray,
                   r_i: float, front_final: float, fraction: float) -> float:
    """Time at which the radial front first reaches ``r_i + fraction*runout``."""
    if time_hist.size == 0:
        return float("nan")
    target = r_i + float(fraction) * (front_final - r_i)
    above = np.nonzero(front_hist >= target)[0]
    if not above.size:
        return float("nan")
    k = int(above[0])
    if k == 0:
        return float(time_hist[0])
    f0, f1 = front_hist[k - 1], front_hist[k]
    t0, t1 = time_hist[k - 1], time_hist[k]
    w = 0.0 if f1 == f0 else (target - f0) / (f1 - f0)
    return float(t0 + w * (t1 - t0))


def _inertial_stats(case: AxisymCase, time_hist: np.ndarray, history: dict) -> dict:
    """Time statistics of the mobilised friction coefficient and ``I_c``.

    Reported for both friction laws so that a constant-friction run and a
    ``mu(I_c)`` run are directly comparable: for the constant law ``mu_bar`` is
    the constant ``mu`` and ``I_c`` is still measured (diagnostic only, it does
    not feed back into the force).
    """
    mu = np.asarray(history.get("mu_eff", []), dtype=np.float64)
    inc = np.asarray(history.get("inertial_number", []), dtype=np.float64)
    mu = mu[np.isfinite(mu)]
    inc = inc[np.isfinite(inc)]
    return {
        "law": ("constant" if case.mu_inertial is None
                else "mu(I_c) = mu_s + (mu_2-mu_s) I_c/(i_0+I_c)"),
        "mu_inertial_params": (None if case.mu_inertial is None
                               else [float(v) for v in case.mu_inertial]),
        "mu_eff_mean": float(mu.mean()) if mu.size else float("nan"),
        "mu_eff_min": float(mu.min()) if mu.size else float("nan"),
        "mu_eff_max": float(mu.max()) if mu.size else float("nan"),
        "inertial_number_mean": float(inc.mean()) if inc.size else float("nan"),
        "inertial_number_max": float(inc.max()) if inc.size else float("nan"),
        "n_samples": int(mu.size),
    }


def simulate_axisymmetric_column(case: AxisymCase, *, return_positions: bool = True,
                                 max_steps: int | None = None,
                                 progress: bool = False) -> dict:
    """Integrate one axisymmetric collapse and return runout, kinematics, ledger."""
    pos, mat = build_axisymmetric_column(case)
    n = pos.shape[0]
    mass, inertia = mat.mass, mat.inertia
    dt = float(case.dt) if case.dt else mat.timestep * case.dt_scale
    gravity_vec = np.array([0.0, 0.0, -mat.gravity])
    mu_wall = mat.mu if case.mu_wall is None else float(case.mu_wall)
    mu_i = None if case.mu_inertial is None else InertialFriction(*case.mu_inertial)

    ext0 = radial_extent(pos, mat.radius)
    r_i_measured = ext0["p995"]
    h_i_measured = float(pos[:, 2].max() + mat.radius)
    a_measured = h_i_measured / r_i_measured
    volume = 4.0 / 3.0 * math.pi * mat.radius ** 3
    phi0 = n * volume / (math.pi * r_i_measured ** 2 * h_i_measured)
    pe0 = float(mass * mat.gravity * np.sum(pos[:, 2]))

    pair_state = PairSpringState()
    wall_state = WallSpringState(n)
    vel = np.zeros((n, 3))
    omega = np.zeros((n, 3))

    t_end = max(case.min_time, case.duration_factor * math.sqrt(h_i_measured / mat.gravity))
    n_steps = int(math.ceil(t_end / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))

    time = 0.0
    diss = {"collision": 0.0, "friction": 0.0, "rolling": 0.0,
            "wall_collision": 0.0, "wall_friction": 0.0}
    spring = {"normal": 0.0, "tangential": 0.0, "rolling": 0.0,
              "wall_normal": 0.0, "wall_tangential": 0.0, "wall_rolling": 0.0}
    cutoff = 0.0
    history = {"time": [], "front": [], "front_max": [], "kinetic": [],
               "potential": [], "contacts": [], "top": [], "mu_eff": [],
               "inertial_number": []}
    front = ext0["p995"]
    front_max = ext0["max"]
    max_contacts = 0
    rest_counter = 0
    stop_reason = "time_limit"
    step = -1

    for step in range(n_steps):
        iu, ju = build_pairs(pos, mat.radius)
        pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, dt,
                                   mu_i=mu_i,
                                   update_state=False, contact_mode=case.contact_mode)
        pred_wall = wall_contact_forces(pos, vel, omega, mat, dt, mu_wall=mu_wall,
                                        mu_i=mu_i,
                                        mu_rolling_wall=mat.mu_rolling,
                                        state=wall_state, update_state=False,
                                        contact_mode=case.contact_mode)
        acc = (pred["force"] + pred_wall["force"]) / mass + gravity_vec
        alpha = (pred["torque"] + pred_wall["torque"]) / inertia
        v_half = vel + 0.5 * dt * acc
        w_half = omega + 0.5 * dt * alpha
        pos += v_half * dt

        cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state, mat, dt,
                                  mu_i=mu_i,
                                  update_state=True, contact_mode=case.contact_mode)
        cur_wall = wall_contact_forces(pos, v_half, w_half, mat, dt, mu_wall=mu_wall,
                                       mu_i=mu_i,
                                       mu_rolling_wall=mat.mu_rolling,
                                       state=wall_state, update_state=True,
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
            ext = radial_extent(pos, mat.radius)
            front, front_max = ext["p995"], ext["max"]
            ke = float(0.5 * mass * np.sum(vel * vel)
                       + 0.5 * inertia * np.sum(omega * omega))
            history["time"].append(time)
            history["front"].append(front)
            history["front_max"].append(front_max)
            history["kinetic"].append(ke)
            history["potential"].append(float(mass * mat.gravity * np.sum(pos[:, 2])))
            history["contacts"].append(int(cur["n_contacts"]))
            history["top"].append(float(pos[:, 2].max() + mat.radius))
            # contact-count-weighted mean of the (possibly rate-dependent)
            # friction coefficient actually mobilised in this sample
            n_pair = int(cur["n_contacts"])
            n_wall = int(cur_wall["n_contacts"])
            n_tot = n_pair + n_wall
            if n_tot > 0:
                mu_bar = (n_pair * float(cur["mu_eff_mean"])
                          + n_wall * float(cur_wall["mu_eff_mean"])) / n_tot
                i_pair, i_wall = cur["inertial_number_mean"], cur_wall["inertial_number_mean"]
                i_bar = sum(float(v) * w for v, w in ((i_pair, n_pair), (i_wall, n_wall))
                            if math.isfinite(float(v))) / max(
                                1, sum(n for v, n in ((i_pair, n_pair), (i_wall, n_wall))
                                       if math.isfinite(float(v))))
            else:
                mu_bar = float("nan")
                i_bar = float("nan")
            history["mu_eff"].append(mu_bar)
            history["inertial_number"].append(i_bar)

        ke = float(0.5 * mass * np.sum(vel * vel)
                   + 0.5 * inertia * np.sum(omega * omega))
        if time > 1.2 * math.sqrt(h_i_measured / mat.gravity) and ke < case.rest_ke_tol * pe0:
            rest_counter += 1
            if rest_counter >= case.rest_hold:
                stop_reason = "at_rest"
                break
        else:
            rest_counter = 0
        if progress and step % 1000 == 0:
            print(f"  step {step}/{n_steps} t={time:.3f}s front={front:.4f} "
                  f"contacts={cur['n_contacts']}", flush=True)

    ext = radial_extent(pos, mat.radius)
    front, front_max = ext["p995"], ext["max"]
    pe_end = float(mass * mat.gravity * np.sum(pos[:, 2]))
    ke_end = float(0.5 * mass * np.sum(vel * vel) + 0.5 * inertia * np.sum(omega * omega))
    spring_total = sum(spring.values())
    diss_collision = diss["collision"] + diss["wall_collision"]
    diss_friction = diss["friction"] + diss["rolling"] + diss["wall_friction"]
    diss_physical = diss_collision + diss_friction
    diss_total = diss_physical + cutoff
    released = pe0 - pe_end
    residual = (pe0 - pe_end - ke_end - spring_total - diss_total) / pe0

    time_hist = np.asarray(history["time"])
    front_hist = np.asarray(history["front"])
    runout_total = front - r_i_measured
    t_by_fraction = {f"{frac:g}": _time_to_reach(time_hist, front_hist,
                                                 r_i_measured, front, frac)
                     for frac in (0.90, 0.95, 0.99)}
    t_reach = _time_to_reach(time_hist, front_hist, r_i_measured, front,
                             case.t_reach_fraction)
    if time_hist.size > 2:
        v_front = np.gradient(front_hist, time_hist)
        peak_front_speed = float(np.max(v_front))
    else:
        peak_front_speed = float("nan")

    result = {
        "case": case.signature(),
        "reference": LUBE2004_SOURCE,
        "n_particles": int(n),
        "dt": dt,
        "n_steps": int(step + 1),
        "time_end": time,
        "stop_reason": stop_reason,
        "column_radius_measured": r_i_measured,
        "column_height_measured": h_i_measured,
        "aspect_measured": a_measured,
        "solid_fraction_initial": phi0,
        "runout_radius_p995": front,
        "runout_radius_max": front_max,
        "runout_ratio_p995": front / r_i_measured,
        "runout_ratio_max": front_max / r_i_measured,
        "runout_relative_p995": runout_total / r_i_measured,
        "t_reach": t_reach,
        "t_reach_by_fraction": t_by_fraction,
        "peak_front_speed": peak_front_speed,
        "ballistic_time": math.sqrt(h_i_measured / mat.gravity),
        "reference_runout_ratio": lube_2004_runout_ratio(case.aspect_ratio),
        "reference_emplacement_time": lube_2004_emplacement_time(
            case.aspect_ratio, case.column_radius, case.gravity),
        "energy": {
            "potential_initial": pe0,
            "potential_final": pe_end,
            "potential_released": released,
            "kinetic_final": ke_end,
            "spring_final": spring_total,
            "spring_channels": spring,
            "dissipated_collision": diss_collision,
            "dissipated_friction": diss_friction,
            "dissipated_total": diss_total,
            "dissipated_cutoff": cutoff,
            "dissipated_physical_only": diss_physical,
            "residual_fraction_of_potential": residual,
        },
        "inertial": _inertial_stats(case, time_hist, history),
        "coordination_max": int(max_contacts),
        "history": {"time": history["time"], "front": history["front"],
                    "front_max": history["front_max"],
                    "kinetic": history["kinetic"],
                    "potential": history["potential"],
                    "contacts": history["contacts"]},
    }
    if return_positions:
        stride = max(1, n // 4000)
        result["positions"] = pos[::stride].tolist()
    return result


def verdict(reference: float, measured: float, tolerance: float = 0.15) -> dict:
    """Relative-error verdict of one measured quantity against the reference."""
    if reference == 0.0 or not math.isfinite(reference):
        raise AxisymError("reference value must be finite and nonzero")
    rel = (measured - reference) / reference
    return {"reference": float(reference), "measured": float(measured),
            "relative_error": float(rel), "abs_relative_error": abs(rel),
            "tolerance": float(tolerance), "pass": abs(rel) <= tolerance}