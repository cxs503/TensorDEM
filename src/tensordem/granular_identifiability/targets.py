"""Forward observables of the synthetic granular system ``test_granular_01``.

Five measurables, all standard granular-DEM calibration targets:

``tilt_onset_deg``
    Critical tilt (deg) at which a settled bed starts to flow -- the avalanche /
    tilting-box onset.  Obtained by bisection on the bed displacement over a
    fixed probe interval, so the observable is a threshold angle with a
    quantisation of ``tilt_max_deg / 2**(bisections+1)``.
``packing_fraction``
    Solid area fraction of the settled bed (``1 - porosity``) measured in the
    same cell before tilting, so it costs no extra run; the vertical porosity
    profile is returned as well.
``discharge_rate``
    Steady mass flow rate through a silo orifice (kg/s), measured after the bed
    has been settled behind a closed bottom.  **Rejected** in the acceptance
    study (see ``study``/docs): in this small-system regime its structural
    jitter exceeds its parameter sensitivity, so it is computed for the record
    but excluded from the identified observable set.
``restitution``
    Impact coefficient of restitution ``|v_out| / |v_in|`` of a single particle.
``impact_duration_s``
    Contact duration of that impact (s).  For a linear spring-dashpot contact the
    half period is ``tau = pi / (omega sqrt(1 - zeta^2))`` with
    ``omega = sqrt(k_n / m)``, so the contact time is the standard
    impact-resonance measurement of the contact stiffness.

Every routine is deterministic (fixed seeds, float64) and charges a
:class:`~tensordem.granular_identifiability.system.CaseBudget`.
"""

from __future__ import annotations

import math

import numpy as np

from .system import (
    BudgetExceeded,
    CaseBudget,
    GranularSystemSpec,
    TEST_GRANULAR_01,
    build_bed_system,
    build_drop_system,
    build_grain_system,
    clone_with_gravity,
    run_for,
)

__all__ = [
    "TARGET_NAMES",
    "IDENTIFIED_TARGETS",
    "TARGET_SPECS",
    "restitution_to_zeta",
    "zeta_to_restitution",
    "surface_profile",
    "bed_packing_fraction",
    "bed_observables",
    "grain_observables",
    "tilt_onset_deg",
    "drop_observables",
    "impact_stiffness",
    "evaluate_targets",
    "target_vector",
    "target_sigma",
]

TARGET_NAMES = (
    "grain_friction_ratio",
    "packing_fraction",
    "restitution",
    "impact_duration_s",
)

#: measurement-noise model of each observable: ``sigma = abs + rel * |y|``
TARGET_SPECS = {
    "grain_friction_ratio": {"unit": "-", "sigma_abs": 0.020, "sigma_rel": 0.0},
    "packing_fraction": {"unit": "-", "sigma_abs": 0.010, "sigma_rel": 0.0},
    "restitution": {"unit": "-", "sigma_abs": 0.0, "sigma_rel": 0.03},
    "impact_duration_s": {"unit": "s", "sigma_abs": 0.0, "sigma_rel": 0.02},
}


def target_sigma(name: str, value: float) -> float:
    """``1-sigma`` measurement uncertainty of observable ``name`` at ``value``."""
    spec = TARGET_SPECS[name]
    return float(spec["sigma_abs"] + spec["sigma_rel"] * abs(value))


def restitution_to_zeta(restitution: float) -> float:
    """Damping ratio of the linear spring-dashpot pair for a target restitution.

    Inverts ``e = exp(-pi zeta / sqrt(1 - zeta^2))``.  The *observed* restitution
    of a compressive-only dashpot is slightly larger than this classic relation;
    the measured value is used in the inversion and the mapping bias is reported
    instead of being assumed away.
    """
    e = float(restitution)
    if not (0.0 < e <= 1.0):
        raise ValueError(f"restitution must lie in (0, 1], got {restitution}")
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


def zeta_to_restitution(zeta: float) -> float:
    """Forward classic relation ``e(zeta)`` (inverse of :func:`restitution_to_zeta`)."""
    z = float(zeta)
    if not (0.0 <= z < 1.0):
        raise ValueError(f"zeta must lie in [0, 1), got {zeta}")
    return math.exp(-math.pi * z / math.sqrt(1.0 - z * z))


# --------------------------------------------------------------------------
# geometry post-processing
# --------------------------------------------------------------------------
def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # pragma: no cover - numpy < 2.0


def surface_profile(
    pos: np.ndarray, rad: np.ndarray, *, bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-of-material height profile ``h(x)`` of a bed plus its porosity.

    Returns ``(x_centers, h, porosity)`` for every occupied bin; ``h`` is the
    highest material point in the bin and ``porosity`` is
    ``1 - (disk area in the bin) / (bin_width * h)``, i.e. the vertical-voidage
    profile a granular experiment would report.
    """
    x_lo = float(np.min(pos[:, 0] - rad)) - 1e-9
    x_hi = float(np.max(pos[:, 0] + rad)) + 1e-9
    n_bins = max(4, int(math.ceil((x_hi - x_lo) / bin_width)))
    idx = np.clip(((pos[:, 0] - x_lo) / bin_width).astype(int), 0, n_bins - 1)
    h = np.zeros(n_bins)
    area = np.zeros(n_bins)
    np.maximum.at(h, idx, pos[:, 1] + rad)
    disk_area = math.pi * rad ** 2
    np.add.at(area, idx, disk_area)
    occupied = area > 0.0
    x_centers = x_lo + (np.arange(n_bins) + 0.5) * bin_width
    porosity = np.full(n_bins, np.nan)
    valid = occupied & (h > 0.0)
    porosity[valid] = 1.0 - area[valid] / (bin_width * h[valid])
    return x_centers[occupied], h[occupied], porosity[occupied]


def bed_packing_fraction(pos: np.ndarray, rad: np.ndarray, *, width: float) -> float:
    """Solid area fraction of a settled bed ``sum(pi r^2) / (width * mean h)``.

    The bed height is the mean of the three highest material points, i.e. the
    free surface with single-particle roughness averaged out.
    """
    tops = np.sort(pos[:, 1] + rad)[-3:]
    height = float(np.mean(tops))
    if height <= 0.0:
        return float("nan")
    return float(np.sum(math.pi * rad ** 2) / (width * height))


# --------------------------------------------------------------------------
# observable evaluations
# --------------------------------------------------------------------------
def tilt_onset_deg(
    settled,
    params: dict,
    *,
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    budget: CaseBudget | None = None,
) -> dict:
    """Critical tilt of the settled bed by bisection on the flow/no-flow test.

    A tilted configuration is emulated by rotating the gravity vector (identical
    to tilting the cell).  At each probe the state is restored from the settled
    configuration, relaxed for ``tilt_probe_time`` and classified as *flowing*
    when the fastest particle exceeds ``tilt_threshold_speed``.  The probe is a
    dynamic threshold, not the static friction angle: a bed only a fraction of a
    degree above the static limit accelerates too little in the probe window to
    be detected, so the reported onset sits a couple of degrees above
    ``atan(mu)``.  It is monotone in ``mu`` and reproducible, which is what the
    identifiability study needs; the bias is stated instead of being corrected
    by an empirical fudge.
    """
    budget = budget or CaseBudget().start()
    threshold = float(spec.tilt_threshold_speed)
    g = spec.gravity
    probes = 0

    def peak_speed(angle_deg: float) -> float:
        nonlocal probes
        rad = math.radians(angle_deg)
        tilted = clone_with_gravity(settled, (g * math.sin(rad), -g * math.cos(rad)))
        run_for(tilted, spec.tilt_probe_time, budget)
        probes += 1
        return float(np.max(np.linalg.norm(tilted.vel, axis=1)))

    if peak_speed(0.0) > threshold:
        return {"tilt_onset_deg": 0.0, "censored": "lo", "probes": probes,
                "threshold_speed": threshold}
    if peak_speed(spec.tilt_max_deg) <= threshold:
        return {"tilt_onset_deg": float(spec.tilt_max_deg), "censored": "hi",
                "probes": probes, "threshold_speed": threshold}
    lo, hi = 0.0, float(spec.tilt_max_deg)
    for _ in range(spec.tilt_bisections):
        mid = 0.5 * (lo + hi)
        if peak_speed(mid) > threshold:
            hi = mid
        else:
            lo = mid
    return {
        "tilt_onset_deg": float(0.5 * (lo + hi)),
        "censored": None,
        "probes": probes,
        "threshold_speed": threshold,
        "bracket": [lo, hi],
        "resolution": float(spec.tilt_max_deg / 2 ** (spec.tilt_bisections + 1)),
    }


def grain_observables(
    params: dict,
    *,
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    budget: CaseBudget | None = None,
    dt: float | None = None,
) -> dict:
    """Single-grain confined shear -> apparent friction ratio ``sum|f_t|/sum f_n``.

    One plate is translated at constant velocity; the force ratio is averaged
    over the second half of the sliding window, i.e. once steady slip is
    established and the ratio has saturated at the Coulomb value.
    """
    budget = budget or CaseBudget().start()
    budget.start()
    sys = build_grain_system(params, spec, dt=dt)
    budget.charge_particles(sys.n)
    speed = spec.grain_slide_speed
    wv = np.zeros((len(sys.walls), 2))
    wv[1] = (0.0, -speed)
    sys.wall_velocity = wv
    normals: list[float] = []
    tangentials: list[float] = []
    n_steps = int(round(spec.grain_slide_time / sys.dt))
    for _ in range(n_steps):
        sys.walls[1, :, 1] -= speed * sys.dt
        sys.step(1)
        budget.charge_steps(1)
        c = sys.contact_forces()
        mask = c["kind"] == 1
        if mask.any():
            normals.append(float(np.sum(c["fn"][mask])))
            tangentials.append(float(np.sum(np.abs(c["ft"][mask]))))
    n_tail = max(1, len(normals) // 2)
    fn = float(np.mean(normals[-n_tail:])) if normals else 0.0
    ft = float(np.mean(tangentials[-n_tail:])) if tangentials else 0.0
    return {
        "grain_friction_ratio": ft / fn if fn > 0.0 else float("nan"),
        "normal_load": fn,
        "tangential_load": ft,
        "slip_samples": len(normals),
        "particles": int(sys.n),
    }


def bed_observables(
    params: dict,
    *,
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    budget: CaseBudget | None = None,
    dt: float | None = None,
    with_tilt: bool = False,
) -> dict:
    """Settled bed -> packing fraction (+ vertical porosity profile).

    ``with_tilt=True`` additionally measures the avalanche onset (used in the
    study's rejected-observable diagnostic; it is not part of the identified
    observable set because a confined bed at this size jams instead of flowing).
    """
    budget = budget or CaseBudget().start()
    budget.start()
    sys = build_bed_system(params, spec, dt=dt)
    budget.charge_particles(sys.n)
    run_for(sys, spec.bed_settle, budget)
    width = spec.bed_span
    bw = 2.5 * sys.mean_radius
    x, h, porosity = surface_profile(sys.pos, sys.radii, bin_width=bw)
    packing = bed_packing_fraction(sys.pos, sys.radii, width=width)
    tilt = tilt_onset_deg(sys, params, spec=spec, budget=budget) if with_tilt else None
    out = {
        "packing_fraction": packing,
        "tilt": tilt,
        "profile": {"x": x.tolist(), "height": h.tolist(), "porosity": porosity.tolist()},
        "kinetic_energy": float(sys.kinetic_energy()),
        "particles": int(sys.n),
    }
    if tilt is not None:
        out["tilt_onset_deg"] = tilt["tilt_onset_deg"]
    return out


def drop_observables(
    params: dict,
    *,
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    budget: CaseBudget | None = None,
    dt: float | None = None,
) -> dict:
    """Single-particle impact -> restitution and contact duration.

    The overlap with the floor is sampled every step and the two zero crossings
    that bound the contact episode are located by linear interpolation, so the
    contact duration is resolved well below one time step.  ``restitution`` is
    the speed ratio across the episode; ``impact_stiffness`` is the
    impact-resonance estimate ``k_n = m (pi / tau)^2 / (1 - zeta^2)`` using the
    measured restitution for ``zeta`` (a derived diagnostic, not a target).
    """
    budget = budget or CaseBudget().start()
    budget.start()
    sys = build_drop_system(params, spec, dt=dt)
    budget.charge_particles(sys.n)
    r = float(sys.radii[0])
    m = float(sys.mass[0])
    t_end = 1.6 * math.sqrt(2.0 * spec.drop_height / spec.gravity)
    n_steps = int(round(t_end / sys.dt))
    overlap = np.empty(n_steps)
    vy = np.empty(n_steps)
    for k in range(n_steps):
        sys.step(1)
        budget.charge_steps(1)
        overlap[k] = r - float(sys.pos[0, 1])
        vy[k] = float(sys.vel[0, 1])
    contact = overlap > 0.0
    if not contact.any():
        return {
            "restitution": float("nan"),
            "impact_duration_s": float("nan"),
            "impact_stiffness": float("nan"),
            "impact_speed": float("nan"),
            "rebound_speed": float("nan"),
            "particles": int(sys.n),
        }
    first = int(np.argmax(contact))
    last = int(len(contact) - 1 - np.argmax(contact[::-1]))
    if first == 0 or last == n_steps - 1:
        raise RuntimeError("impact episode touches the integration window boundary")
    dt_step = sys.dt
    t_in = (first - 1) * dt_step + dt_step * overlap[first - 1] / (
        overlap[first - 1] - overlap[first]
    )
    t_out = last * dt_step + dt_step * overlap[last] / (overlap[last] - overlap[last + 1])
    duration = float(t_out - t_in)
    v_in = float(abs(vy[first - 1]))
    v_out = float(max(0.0, vy[last + 1]))
    restitution = v_out / v_in if v_in > 0.0 else float("nan")
    return {
        "restitution": restitution,
        "impact_duration_s": duration,
        "impact_stiffness": impact_stiffness(
            {"impact_duration_s": duration, "restitution": restitution}, m
        ),
        "impact_speed": v_in,
        "rebound_speed": v_out,
        "particles": int(sys.n),
    }


def impact_stiffness(result: dict, particle_mass: float) -> float:
    """Impact-resonance stiffness estimate (N/m) from an impact record."""
    tau = float(result["impact_duration_s"])
    e = float(result["restitution"])
    if not (tau > 0.0 and 0.0 < e <= 1.0):
        return float("nan")
    zeta = restitution_to_zeta(e)
    omega = math.pi / (tau * math.sqrt(max(1e-12, 1.0 - zeta * zeta)))
    return float(particle_mass * omega ** 2)


# --------------------------------------------------------------------------
# case driver
# --------------------------------------------------------------------------
_SUBCASES = {
    "bed": bed_observables,
    "grain": grain_observables,
    "drop": drop_observables,
}


def _run_subcase(item):
    kind, kwargs = item
    template = kwargs["budget"]
    budget = CaseBudget(
        max_particles=template.max_particles,
        max_steps=template.max_steps,
        max_wall_seconds=template.max_wall_seconds,
    ).start()
    out = _SUBCASES[kind](kwargs["params"], spec=kwargs["spec"], budget=budget, dt=kwargs["dt"])
    out["_usage"] = budget.report()
    return kind, out


def evaluate_targets(
    params: dict,
    *,
    spec: GranularSystemSpec = TEST_GRANULAR_01,
    budget: CaseBudget | None = None,
    dt: float | None = None,
    detail: bool = False,
    workers: int = 1,
) -> dict:
    """Full forward evaluation: observables from three contact parameters.

    ``params`` must provide ``friction`` (Coulomb coefficient, -),
    ``restitution`` (-) and ``stiffness`` (pair normal stiffness, N/m).
    Returns ``{"targets": {...}, "usage": {...}}`` (plus per-case detail when
    ``detail=True``).

    ``workers > 1`` evaluates the two independent simulations in a process pool.
    The results are bit-identical to the serial path (every sub-case is
    deterministic); only the wall-clock differs, and the per-case budget is
    enforced *inside* each sub-case so parallelism cannot be used to smuggle in
    extra compute.
    """
    for key in ("friction", "restitution", "stiffness"):
        if key not in params:
            raise KeyError(f"missing parameter {key!r}")
    if float(params["friction"]) <= 0.0:
        raise ValueError("friction must be positive")
    template = budget or CaseBudget()
    budget = CaseBudget(
        max_particles=template.max_particles,
        max_steps=template.max_steps,
        max_wall_seconds=template.max_wall_seconds,
    ).start()
    jobs = {
        kind: {"params": dict(params), "spec": spec, "dt": dt, "budget": budget}
        for kind in ("bed", "grain", "drop")
    }
    if workers and workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(int(workers), len(jobs))) as pool:
            results = dict(pool.map(_run_subcase, list(jobs.items())))
    else:
        results = {kind: _run_subcase((kind, kwargs))[1] for kind, kwargs in jobs.items()}
    for out in results.values():
        usage = out.pop("_usage")
        budget.particles = max(budget.particles, usage["particles"])
        budget.steps += usage["steps"]
        budget.wall_seconds = max(budget.wall_seconds, usage["wall_seconds"])
    if budget.particles > budget.max_particles:
        raise BudgetExceeded(
            f"case needs {budget.particles} particles > budget {budget.max_particles}"
        )
    if budget.steps > budget.max_steps:
        raise BudgetExceeded(f"case accumulated {budget.steps} steps > budget {budget.max_steps}")
    bed = results["bed"]
    grain = results["grain"]
    drop = results["drop"]
    targets = {
        "grain_friction_ratio": grain["grain_friction_ratio"],
        "packing_fraction": bed["packing_fraction"],
        "restitution": drop["restitution"],
        "impact_duration_s": drop["impact_duration_s"],
    }
    out = {"targets": targets, "usage": budget.report(), "params": dict(params)}
    if detail:
        out.update({"bed": {k: v for k, v in bed.items() if k != "profile"},
                    "profile": bed["profile"], "grain": grain, "drop": drop})
    return out


def target_vector(result: dict, names=None) -> np.ndarray:
    """Ordered observable vector of an :func:`evaluate_targets` result."""
    return np.asarray([result["targets"][k] for k in (names or TARGET_NAMES)], dtype=float)