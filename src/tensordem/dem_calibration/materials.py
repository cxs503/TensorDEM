"""Material classes, experiment definitions and in-silico "measurements".

Three granular classes are used, ordered by grain size and shape regularity:

===============  ========  =========  ==========  ===========================
class            r [mm]    rho        mu_r prior  literature repose band [deg]
===============  ========  =========  ==========  ===========================
glass beads      3.0       2500       0.05        20 - 30
sand             2.0       2650       0.10        30 - 37
gravel           5.0       2650       0.15        35 - 42
===============  ========  =========  ==========  ===========================

Each class has its own **calibration experiment** (container radius, fill
height, protocol) and an **independent validation experiment** with a different
grain radius, container radius and fill height; the calibrated parameters are
transferred to the validation geometry without any refitting, which is what the
generalisation criterion measures.

Where the targets come from
---------------------------
There is no laboratory in this repository, so the "measurements" are *twin
experiments*: the target responses are the forward DEM model evaluated at a
known ("true") parameter set, plus a declared observation noise
(``OBSERVATION_SIGMA``).  This verifies the inverse machinery end to end -
recovery, prediction error, credible intervals, reproducibility - and is the
standard way to qualify a calibration pipeline before it is pointed at real
data.  It does **not** prove that the forward model reproduces a real material;
that residual question is reported separately as a literature cross-check
(``literature_theta_deg``) and is stated as a limitation in the report.  The
twin truth values are *not* the literature targets: they are plausible
parameter triples inside the calibrated box, chosen so that the three classes
differ in friction and restitution as they do in reality.

Provenance of the parameter box and of the priors
-------------------------------------------------
``mu_pp in [0.1, 0.8]``, ``e in [0.1, 0.9]``, ``k_n/k_s in [1, 3]`` as
specified by the campaign node; ``mu_r`` prior per class from the rolling-
resistance ranges of Ai et al. (2011) for glass/sand/gravel DEM calibrations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .heap import HeapCase

__all__ = [
    "MaterialSpec",
    "MATERIALS",
    "OBSERVATION_SIGMA",
    "RESPONSE_WEIGHTS",
    "case_for",
]

#: Declared 1-sigma observation noise of the twin experiments.
OBSERVATION_SIGMA = {"theta_r_deg": 0.5, "phi_solid": 0.01}

#: Objective weights, ``J = sum_i w_i |sim_i - exp_i| / |exp_i|``.
RESPONSE_WEIGHTS = {"theta_r_deg": 0.8, "phi_solid": 0.2}


@dataclass(frozen=True)
class MaterialSpec:
    """One granular class: material, experiments and the twin-experiment truth."""

    name: str
    radius: float
    density: float
    mu_rolling: float
    truth: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    literature_theta_deg: tuple[float, float] = (0.0, 90.0)

    def case(self, condition: str, params: dict | None = None) -> HeapCase:
        return case_for(self, condition, params or self.truth)


def case_for(spec: MaterialSpec, condition: str, params: dict) -> HeapCase:
    """Build the :class:`HeapCase` of ``condition`` with ``params`` applied.

    ``condition`` is ``"calibration"`` or ``"validation"``; the condition dict
    holds geometry/protocol overrides (grain radius, container, protocol).
    """
    if condition not in ("calibration", "validation"):
        raise ValueError("condition must be 'calibration' or 'validation'")
    overrides = dict(spec.calibration if condition == "calibration" else spec.validation)
    case_params = {
        "mu": float(params["mu"]),
        "restitution": float(params["restitution"]),
        "stiffness_ratio": float(params["stiffness_ratio"]),
        "density": spec.density,
        "mu_rolling": spec.mu_rolling,
    }
    ratios = {k: overrides.pop(k) for k in ("lift_ratio", "settle_ratio") if k in overrides}
    # container/fill geometry is irrelevant for the growth protocol
    if overrides.get("shape") == "pluviation":
        overrides.pop("fill_height", None)
    case_params.update(overrides)
    case = HeapCase(**case_params)
    lift_ratio = ratios.get("lift_ratio")
    settle_ratio = ratios.get("settle_ratio")
    if lift_ratio is None and settle_ratio is None:
        return case
    # The protocol is defined in *ballistic* units: the container is raised
    # over ``lift_ratio * sqrt(H/g)`` and the heap is then given
    # ``settle_ratio * sqrt(H/g)`` to relax, so the same dimensionless driving
    # is used at every grain size and container size.
    tau = math.sqrt(case.column_height / case.gravity)
    return replace(case,
                   lift_time=float(lift_ratio) * tau if lift_ratio else case.lift_time,
                   settle_time=float(settle_ratio) * tau if settle_ratio
                   else case.settle_time)


#: Pluviation ("rain") protocol: 240 grains poured in layers of 12 onto the
#: growing heap, 200 relaxation steps per layer.  Chosen because the two
#: release protocols (retreating wall, over-steep cone) do not produce a usable
#: repose response at this system size - see docs/DEM_AUTO_CALIBRATION.md.
_PROTOCOL = dict(kn=800.0, hold_time=0.0, lift_ratio=1.0, settle_ratio=1.0,
                 packing_gap=0.03, jitter=0.02, shape="pluviation",
                 n_grains=240, layer_size=12, settle_per_layer=200,
                 inject_radius_factor=7.0, drop_height_factor=1.5)

MATERIALS: tuple[MaterialSpec, ...] = (
    MaterialSpec(
        name="glass_beads",
        radius=0.004,
        density=2500.0,
        mu_rolling=0.05,
        truth={"mu": 0.34, "restitution": 0.62, "stiffness_ratio": 1.4},
        calibration=dict(radius=0.004, cylinder_radius=0.028, fill_height=0.056,
                         seed=31001, **_PROTOCOL),
        validation=dict(radius=0.0035, cylinder_radius=0.0245, fill_height=0.049,
                        seed=31002, **_PROTOCOL),
        literature_theta_deg=(20.0, 30.0),
    ),
    MaterialSpec(
        name="sand",
        radius=0.003,
        density=2650.0,
        mu_rolling=0.10,
        truth={"mu": 0.52, "restitution": 0.38, "stiffness_ratio": 2.2},
        calibration=dict(radius=0.003, cylinder_radius=0.021, fill_height=0.042,
                         seed=32001, **_PROTOCOL),
        validation=dict(radius=0.0035, cylinder_radius=0.0245, fill_height=0.049,
                        seed=32002, **_PROTOCOL),
        literature_theta_deg=(30.0, 37.0),
    ),
    MaterialSpec(
        name="gravel",
        radius=0.005,
        density=2650.0,
        mu_rolling=0.15,
        truth={"mu": 0.68, "restitution": 0.18, "stiffness_ratio": 2.8},
        calibration=dict(radius=0.005, cylinder_radius=0.035, fill_height=0.070,
                         seed=33001, **_PROTOCOL),
        validation=dict(radius=0.004, cylinder_radius=0.028, fill_height=0.056,
                        seed=33002, **_PROTOCOL),
        literature_theta_deg=(35.0, 42.0),
    ),
)


def spec_by_name(name: str) -> MaterialSpec:
    for spec in MATERIALS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown material {name!r}; known: {[m.name for m in MATERIALS]}")
