"""Engine-level closure of the DEM3D Hertz restitution mapping.

Question answered here: **with ``contact_model="hertz"``, is the coefficient of
restitution actually controllable through a physical parameter?**

Before this script the only engine-level restitution evidence covered the
*linear* penalty branch (``docs/dem3d_restitution_verification.md``).  The Hertz
branch derives its dashpot from the local secant stiffness,

    gamma_n(delta) = 2 zeta sqrt(m_eff k_s(delta)),   k_s(delta) = 2 E* sqrt(R_eff) sqrt(delta),

with ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)`` taken from *linear* spring-dashpot
theory.  Because ``gamma ~ delta^(1/4)`` this is a Tsuji-type non-linear damper
that should keep ``e`` velocity independent -- but that was never measured in the
solver.

Three references that do not depend on the implementation:

R1  independent RK4 integration of the *same* contact law in the relative
    coordinate (code path disjoint from ``dem3d``); its own time-step
    convergence is checked first.
R2  analytic scaling: a *constant* dashpot on a Hertz spring gives
    ``delta_max ~ v^(4/5)`` and ``t_c ~ v^(-1/5)``, hence ``dE ~ c v^(9/5)`` and
    ``1 - e ~ c v^(-1/5)``: log-log slope **-0.2**.
R3  elastic limit ``c = 0``: ``e = 1`` exactly and the peak overlap obeys
    ``delta_max = (15 m_eff v^2 / (16 E* sqrt(R_eff)))^(2/5)``.

Run:  PYTHONPATH=src python scripts/verify_dem3d_hertz_restitution.py [--quick] [--render]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)

PROTOCOL = "tensordem-dem3d-hertz-restitution-closure-v1"

RADIUS = 0.025
DENSITY = 917.0
YOUNG_MODULUS = 1.0e9
POISSON_RATIO = 0.3
PARK = 1.0

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)        # m/s, per-particle
TARGETS = (0.5, 0.8)                        # restitution targets
ABSOLUTE_DASH = 60.0                        # N s/m, scaling-anchor dashpot
SCALE_SPEEDS = (0.2, 0.4, 0.8, 1.6, 3.0)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------
def pair_config(*, dt_factor: float = 0.6, **kwargs) -> DEM3DConfig:
    """2x2x2 lattice used only as a container for one free contact pair."""
    params = dict(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        contact_model="hertz", young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO,
        contact_damping=0.0, drag=0.0,
        fix_bottom=False, fix_x_edges=False,
        tool_speed=1.0, tool_gap=10.0,   # indenter parked far away
    )
    params.update(kwargs)                       # case overrides win
    base = DEM3DConfig(**params)
    return DEM3DConfig(**{**base.__dict__, "dt": dt_factor * base.recommended_dt})


def build_pair(speed: float, cfg: DEM3DConfig, gap_fraction: float = 0.05) -> IceDEM3D:
    """Two free spheres approaching head-on along z, closing speed ``2*speed``."""
    sim = IceDEM3D(cfg)
    sim.alive[:] = False                       # isolate the unbonded contact pair
    sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([0.0, 0.0, 2.0 * RADIUS * (1.0 + gap_fraction)],
                                    dtype=sim.dtype)
    for k in range(2, len(sim.positions)):
        sim.positions[k] = torch.tensor([PARK * (k - 1), PARK * (k - 1), 3.0 * PARK],
                                        dtype=sim.dtype)
    sim.velocities.zero_()
    sim.velocities[0, 2] = +speed
    sim.velocities[1, 2] = -speed
    return sim


def _pair_state(sim: IceDEM3D) -> tuple[float, float]:
    """``(separation, closing velocity)`` of the pair (0, 1)."""
    r = sim.positions[1] - sim.positions[0]
    separation = float(torch.linalg.vector_norm(r))
    normal = r / separation
    closing = -float((sim.velocities[1] - sim.velocities[0]) @ normal)
    return separation, closing


def measure_engine(speed: float, *, dt_factor: float = 0.6,
                   max_steps: int = 200_000, **kwargs) -> dict:
    """Integrate one impact; return the measured restitution and diagnostics."""
    cfg = pair_config(dt_factor=dt_factor, **kwargs)
    sim = build_pair(speed, cfg)
    separation0, approach = _pair_state(sim)
    min_separation = separation0
    in_contact = False
    rebounded = None
    steps = 0
    while steps < max_steps:
        sim.step()
        steps += 1
        separation, closing = _pair_state(sim)
        min_separation = min(min_separation, separation)
        if separation < 2.0 * RADIUS:
            in_contact = True
        elif in_contact:
            rebounded = abs(closing)
            break
    if rebounded is None:
        raise RuntimeError("impact did not separate within max_steps")
    return {
        "speed_in_m_per_s": approach,
        "speed_out_m_per_s": rebounded,
        "restitution": rebounded / approach,
        "peak_overlap_m": 2.0 * RADIUS - min_separation,
        "contact_time_s": steps * float(cfg.dt),
        "steps": steps,
        "dt_s": float(cfg.dt),
    }


# ---------------------------------------------------------------------------
# Reference solutions
# ---------------------------------------------------------------------------
def law_parameters(**kwargs) -> dict:
    """Reduced mass and Hertz coefficients of the equal-sphere pair."""
    cfg = pair_config(**kwargs)
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    return {
        "mass_kg": cfg.mass,
        "m_eff_kg": cfg.mass / 2.0,
        "e_star_Pa": e_star,
        "r_eff_m": r_eff,
        "k_s_coefficient": 2.0 * e_star * math.sqrt(r_eff),
    }


def reference_rk4(speed: float, *, rest_target: float = 1.0,
                  absolute_dash: float | None = None, dt: float = 1.0e-7,
                  max_time: float = 5.0e-3) -> dict:
    """Independent RK4 integration of the relative-coordinate contact law.

    ``absolute_dash`` selects a constant dashpot; otherwise the dashpot is the
    engine's ``gamma(delta) = 2 zeta sqrt(m_eff k_s(delta))``.
    """
    p = law_parameters()
    m_eff = p["m_eff_kg"]
    k_coef = p["k_s_coefficient"]
    zeta = 0.0
    if absolute_dash is None and rest_target < 1.0:
        ln_e = math.log(rest_target)
        zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)

    def accel(overlap: float, rate: float) -> float:
        if overlap <= 0.0:
            return 0.0
        elastic = k_coef * overlap * math.sqrt(overlap)
        gamma = absolute_dash if absolute_dash is not None else \
            2.0 * zeta * math.sqrt(m_eff * k_coef * math.sqrt(overlap))
        # ``rate`` is d(overlap)/dt > 0 while approaching, so the separating
        # normal velocity is -rate and the dissipative part adds gamma*rate.
        force = elastic + gamma * rate
        return -force / m_eff if force > 0.0 else 0.0

    overlap, rate = 0.0, speed
    peak_overlap = 0.0
    t = 0.0
    in_contact = False
    while t < max_time:
        k1d, k1r = rate, accel(overlap, rate)
        k2d = rate + 0.5 * dt * k1r
        k2r = accel(overlap + 0.5 * dt * k1d, k2d)
        k3d = rate + 0.5 * dt * k2r
        k3r = accel(overlap + 0.5 * dt * k2d, k3d)
        k4d = rate + dt * k3r
        k4r = accel(overlap + dt * k3d, k4d)
        overlap += dt * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
        rate += dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
        peak_overlap = max(peak_overlap, overlap)
        if overlap > 0.0:
            in_contact = True
        elif in_contact:
            break
        t += dt
    return {"restitution": -rate / speed, "peak_overlap_m": peak_overlap,
            "contact_time_s": t, "zeta": zeta, "dt_s": dt}


def linear_reference(speed: float, target: float) -> float:
    """Independent RK4 for the linear spring-dashpot pair (same reduced mass)."""
    p = law_parameters()
    m_eff = p["m_eff_kg"]
    k = 5_000.0
    ln_e = math.log(target)
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    c = 2.0 * zeta * math.sqrt(m_eff * k)
    dt = 1.0e-8
    overlap, rate = 0.0, speed
    in_contact = False
    for _ in range(2_000_000):
        def acc(o: float, r: float) -> float:
            if o <= 0.0:
                return 0.0
            f = k * o + c * r
            return -f / m_eff if f > 0.0 else 0.0
        k1d, k1r = rate, acc(overlap, rate)
        k2d = rate + 0.5 * dt * k1r
        k2r = acc(overlap + 0.5 * dt * k1d, k2d)
        k3d = rate + 0.5 * dt * k2r
        k3r = acc(overlap + 0.5 * dt * k2d, k3d)
        k4d = rate + dt * k3r
        k4r = acc(overlap + dt * k3d, k4d)
        overlap += dt * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
        rate += dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
        if overlap > 0.0:
            in_contact = True
        elif in_contact:
            break
    return -rate / speed


def analytic_peak_overlap(speed: float) -> float:
    """``delta_max`` of a purely elastic equal-sphere Hertz impact."""
    p = law_parameters()
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    return ((15.0 * p["m_eff_kg"] * speed ** 2)
            / (16.0 * e_star * math.sqrt(r_eff))) ** 0.4


def loglog_slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def case_elastic_limit() -> dict:
    rows, worst_e, worst_ov = [], 0.0, 0.0
    for speed in (0.1, 0.5, 1.0):
        got = measure_engine(speed, restitution_coefficient=1.0)
        ref = analytic_peak_overlap(speed)
        err_ov = abs(got["peak_overlap_m"] - ref) / ref
        worst_e = max(worst_e, abs(got["restitution"] - 1.0))
        worst_ov = max(worst_ov, err_ov)
        rows.append({"speed_m_per_s": speed, "restitution": got["restitution"],
                     "peak_overlap_m": got["peak_overlap_m"],
                     "peak_overlap_reference_m": ref, "peak_overlap_rel_error": err_ov})
    return {"case": "elastic_limit", "metric": "|e-1| and peak-overlap error vs closed form",
            "measured": worst_e, "reference": 0.0, "relative_error": worst_e,
            "worst_overlap_rel_error": worst_ov, "tolerance": 5.0e-3,
            "passed": bool(worst_e <= 5.0e-3 and worst_ov <= 5.0e-2), "samples": rows}


def case_absolute_scaling() -> dict:
    rows, eng, ref_d = [], [], []
    for speed in SCALE_SPEEDS:
        got = measure_engine(speed, contact_damping_model="absolute",
                             contact_damping=ABSOLUTE_DASH, restitution_coefficient=1.0)
        ref = reference_rk4(speed, absolute_dash=ABSOLUTE_DASH)
        eng.append(1.0 - got["restitution"])
        ref_d.append(1.0 - ref["restitution"])
        rows.append({"speed_m_per_s": speed,
                     "restitution_engine": got["restitution"],
                     "restitution_reference": ref["restitution"],
                     "restitution_rel_error": abs(got["restitution"] - ref["restitution"]) / ref["restitution"],
                     "dissipation_engine": 1.0 - got["restitution"],
                     "dissipation_reference": 1.0 - ref["restitution"]})
    slope_eng = loglog_slope(list(SCALE_SPEEDS), eng)
    slope_ref = loglog_slope(list(SCALE_SPEEDS), ref_d)
    analytic = -0.2
    err = abs(slope_eng - analytic) / abs(analytic)
    err_ref = abs(slope_ref - analytic) / abs(analytic)
    worst = max(r["restitution_rel_error"] for r in rows)
    return {"case": "absolute_dashpot_scaling",
            "metric": "log-log slope of (1-e) vs v; analytic -1/5",
            "measured": slope_eng, "reference": analytic, "relative_error": err,
            "slope_reference_rk4": slope_ref, "slope_reference_rel_error": err_ref,
            "worst_engine_vs_reference_rel_error": worst, "tolerance": 5.0e-2,
            "passed": bool(err <= 0.05 and err_ref <= 0.03 and worst <= 0.05),
            "abs_dashpot_N_s_per_m": ABSOLUTE_DASH, "samples": rows}


def case_restitution_closure() -> dict:
    rows, worst, worst_ref = [], 0.0, 0.0
    for target in TARGETS:
        e_list = []
        for speed in SPEEDS:
            got = measure_engine(speed, restitution_coefficient=target)
            ref = reference_rk4(speed, rest_target=target)
            err = abs(got["restitution"] - target) / target
            err_ref = abs(got["restitution"] - ref["restitution"]) / ref["restitution"]
            worst = max(worst, err)
            worst_ref = max(worst_ref, err_ref)
            e_list.append(got["restitution"])
            rows.append({"target": target, "speed_m_per_s": speed,
                         "restitution_engine": got["restitution"],
                         "restitution_reference": ref["restitution"],
                         "rel_error_vs_target": err, "rel_error_vs_reference": err_ref,
                         "peak_overlap_m": got["peak_overlap_m"]})
        rows.append({"target": target, "speed_m_per_s": None,
                     "restitution_engine": None, "restitution_reference": None,
                     "rel_error_vs_target": (max(e_list) - min(e_list)) / target,
                     "rel_error_vs_reference": None,
                     "note": "velocity spread of measured e over the sweep"})
    return {"case": "restitution_physical_closure",
            "metric": "max |e_measured - e_target| / e_target over speeds",
            "measured": worst, "reference": 0.0, "relative_error": worst,
            "worst_engine_vs_reference_rel_error": worst_ref, "tolerance": 3.0e-2,
            "passed": bool(worst <= 3.0e-2), "samples": rows}


def case_linear_control() -> dict:
    rows, worst = [], 0.0
    for speed in (0.25, 0.5, 1.0):
        got = measure_engine(speed, contact_model="linear",
                             restitution_coefficient=0.5, contact_stiffness=5_000.0)
        err = abs(got["restitution"] - 0.5) / 0.5
        worst = max(worst, err)
        rows.append({"speed_m_per_s": speed, "restitution_engine": got["restitution"],
                     "reference_linear_law": linear_reference(speed, 0.5),
                     "rel_error_vs_target": err})
    return {"case": "linear_branch_control",
            "metric": "max |e - 0.5| / 0.5 on the linear branch",
            "measured": worst, "reference": 0.0, "relative_error": worst,
            "tolerance": 3.0e-2, "passed": bool(worst <= 3.0e-2), "samples": rows}


def case_reference_convergence() -> dict:
    rows = []
    for dt in (2.0e-7, 1.0e-7, 5.0e-8):
        rows.append({"dt_s": dt,
                     "restitution": reference_rk4(0.5, rest_target=0.5, dt=dt)["restitution"]})
    fine = rows[-1]["restitution"]
    worst = 0.0
    for r in rows:
        r["rel_error_vs_finest"] = abs(r["restitution"] - fine) / fine
        worst = max(worst, r["rel_error_vs_finest"])
    order = math.log2(rows[0]["rel_error_vs_finest"] / max(rows[1]["rel_error_vs_finest"], 1e-18))
    return {"case": "reference_convergence",
            "metric": "RK4 restitution drift under dt refinement",
            "measured": worst, "reference": 0.0, "relative_error": worst,
            "observed_order": order, "tolerance": 1.0e-3,
            "passed": bool(worst <= 1.0e-3 and order > 1.5), "samples": rows}


def case_engine_dt_convergence() -> dict:
    rows, worst = [], 0.0
    ref = reference_rk4(0.5, rest_target=0.5)
    for factor in (0.6, 0.3, 0.15):
        got = measure_engine(0.5, dt_factor=factor, restitution_coefficient=0.5)
        err = abs(got["restitution"] - ref["restitution"]) / ref["restitution"]
        worst = max(worst, err)
        rows.append({"dt_factor_of_recommended": factor, "dt_s": got["dt_s"],
                     "restitution": got["restitution"], "rel_error_vs_reference": err})
    return {"case": "engine_dt_convergence",
            "metric": "engine vs RK4 restitution at three time steps",
            "measured": worst, "reference": 0.0, "relative_error": worst,
            "tolerance": 3.0e-2, "passed": bool(worst <= 3.0e-2), "samples": rows}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def make_figures(results: dict, outdir: pathlib.Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                   # pragma: no cover
        return []
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    by = {c["case"]: c for c in results["cases"]}
    made = []

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for target in TARGETS:
        samples = [s for s in by["restitution_physical_closure"]["samples"]
                   if s["target"] == target and s["speed_m_per_s"] is not None]
        xs = [s["speed_m_per_s"] for s in samples]
        ax.plot(xs, [s["restitution_engine"] for s in samples], "o-",
                label=f"engine, e_target={target}")
        ax.plot(xs, [s["restitution_reference"] for s in samples], "x--",
                label=f"RK4, e_target={target}")
    ax.set_xscale("log")
    ax.set_xlabel("impact speed v [m/s]")
    ax.set_ylabel("measured restitution e")
    ax.set_title("Hertz branch: restitution versus impact speed")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    made.append(str(figdir / "restitution_vs_speed.png"))
    fig.savefig(made[-1], dpi=130)
    plt.close(fig)

    case = by["absolute_dashpot_scaling"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [s["speed_m_per_s"] for s in case["samples"]]
    ax.loglog(xs, [s["dissipation_engine"] for s in case["samples"]], "o",
              label=f"engine (slope {case['measured']:+.3f})")
    ax.loglog(xs, [s["dissipation_reference"] for s in case["samples"]], "x",
              label=f"RK4 (slope {case['slope_reference_rk4']:+.3f})")
    ax.loglog(xs, [case["samples"][0]["dissipation_reference"] * (v / xs[0]) ** (-0.2)
                   for v in xs], "k--", label="analytic slope -1/5")
    ax.set_xlabel("impact speed v [m/s]")
    ax.set_ylabel("1 - e")
    ax.set_title("Constant dashpot + Hertz spring: 1-e ~ v^-1/5")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    made.append(str(figdir / "absolute_dashpot_scaling.png"))
    fig.savefig(made[-1], dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for target in TARGETS:
        samples = [s for s in by["restitution_physical_closure"]["samples"]
                   if s["target"] == target and s["speed_m_per_s"] is not None]
        ax.semilogx([s["speed_m_per_s"] for s in samples],
                    [100.0 * s["rel_error_vs_target"] for s in samples], "o-",
                    label=f"e_target={target}")
    ax.axhline(3.0, color="k", linestyle="--", label="3 % acceptance")
    ax.set_xlabel("impact speed v [m/s]")
    ax.set_ylabel("|e - e_target| / e_target [%]")
    ax.set_title("Physical-parameter closure error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    made.append(str(figdir / "closure_error.png"))
    fig.savefig(made[-1], dpi=130)
    plt.close(fig)
    return made


def write_outputs(results: dict, outdir: pathlib.Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in results["cases"]:
        for sample in case.get("samples", []):
            rows.append({"case": case["case"], **sample})
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (outdir / "checks.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    figures = make_figures(results, outdir)
    results["figures"] = figures
    (outdir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/dem3d-hertz-restitution")
    parser.add_argument("--render", action="store_true", help="write the markdown report too")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    global SPEEDS
    if args.quick:
        SPEEDS = (0.5,)

    cases = [case_elastic_limit(), case_absolute_scaling(), case_restitution_closure(),
             case_linear_control(), case_reference_convergence(),
             case_engine_dt_convergence()]
    results = {
        "protocol": PROTOCOL,
        "radius_m": RADIUS, "density_kg_per_m3": DENSITY,
        "young_modulus_Pa": YOUNG_MODULUS, "poisson_ratio": POISSON_RATIO,
        "speeds_m_per_s": list(SPEEDS), "targets": list(TARGETS),
        "absolute_dashpot_N_s_per_m": ABSOLUTE_DASH,
        "cases": cases, "all_passed": all(c["passed"] for c in cases),
    }
    outdir = pathlib.Path(args.outdir)
    write_outputs(results, outdir)
    if args.render:
        from render_dem3d_hertz_restitution import render  # noqa: PLC0415
        (pathlib.Path("docs") / "dem3d_hertz_restitution_closure.md").write_text(render(results))
    for case in cases:
        print(f"{case['case']:32s} measured={case['measured']!r:>12} "
              f"rel_err={case['relative_error']!r:>12} passed={case['passed']}")
    print(f"all_passed={results['all_passed']}  outdir={outdir}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())