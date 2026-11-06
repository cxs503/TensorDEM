"""Fixed-support restitution: the pair dashpot must use the *rigid-partner* mass.

Question answered here: when a free sphere hits a **pinned** (hard-support) sphere,
does ``restitution_coefficient`` still map to the realised coefficient of
restitution, or is the pair dashpot scaled with the wrong effective mass?

The relative coordinate of a contact pair obeys one clipped oscillator

    m_eff * delta'' = -max(k*delta + gamma*delta', 0),   zeta = |ln e|/sqrt(pi^2+ln^2 e),
    gamma = 2 zeta sqrt(k m_eff)        (linear)  or
    gamma = 2 zeta sqrt(k_s(delta) m_eff), k_s = (4/3)E* sqrt(R*) sqrt(delta)  (Hertz)

* two **mobile** equal spheres: ``m_eff = m/2`` (both accelerate, opposite signs);
* one **pinned** sphere: the partner is infinitely massive, only the free
  particle accelerates, ``m_eff = m`` — ``tensordem.restitution.effective_mass(m, None)``
  and the rigid-counterpart convention the tool/platen branches already use.

``dem3d`` used ``m_eff = cfg.mass/2`` for every pair, so a pinned pair was
under-damped by ``sqrt(2)`` and the realised ``e`` overshot ``e_target``.
References used here (none derived from the implementation under test):

R1  independent RK4 integration of the *same* clipped law in the relative
    coordinate, run twice: with ``m_eff = m`` (correct) and ``m_eff = m/2``
    (the pre-fix convention).  The engine must track the first.
R2  the free-free pair measured in the same harness: after the fix the pinned
    and free-free realisations of the same target must agree, because the mass
    convention is the only difference between the two runs.
R3  the pre-fix engine itself, re-run from the archived source of the parent
    commit (``--legacy-src``), must reproduce the ``m_eff = m/2`` branch of R1.

Run: PYTHONPATH=src python scripts/verify_dem3d_fixed_support_restitution.py \
        [--quick] [--render] [--legacy-src DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# ``TENSORDEM_SRC`` lets the same script drive an archived (e.g. pre-fix) source
# tree without copying files; it is how the legacy control below is produced.
_SRC_OVERRIDE = pathlib.Path(os.environ["TENSORDEM_SRC"]) if os.environ.get("TENSORDEM_SRC") \
    else ROOT / "src"
sys.path.insert(0, str(_SRC_OVERRIDE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from tensordem.dem3d import (  # noqa: E402
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)
from verify_dem3d_hertz_restitution import (  # noqa: E402
    DENSITY,
    POISSON_RATIO,
    RADIUS,
    YOUNG_MODULUS,
    build_pair,
    pair_config,
)

PROTOCOL = "tensordem-dem3d-fixed-support-restitution-v1"
OUT = ROOT / "docs" / "artifacts" / "fixed-support-restitution"

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)   # m/s, closing speed of the impact
TARGETS = (0.5, 0.8)                  # restitution targets


# ---------------------------------------------------------------------------
# Case construction (reuses the equal-sphere harness)
# ---------------------------------------------------------------------------
def build_impact(speed: float, *, pinned: bool, e_target: float, model: str,
                 dt_factor: float = 0.6) -> tuple[IceDEM3D, float]:
    """One head-on impact; returns ``(simulation, closing speed)``.

    ``pinned=False``: the two spheres approach symmetrically (``+/-speed``),
    closing speed ``2*speed``, ``m_eff = m/2``.
    ``pinned=True``: sphere 1 is a hard support (velocity zeroed / position reset
    every step), sphere 0 approaches at ``speed``.
    """
    kwargs = dict(restitution_coefficient=e_target, contact_model=model)
    if model == "hertz":
        kwargs.update(young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO)
    cfg = pair_config(dt_factor=dt_factor, **kwargs)
    sim = build_pair(speed, cfg)
    if pinned:
        sim.fixed[1] = True
        sim.initial_positions[1] = sim.positions[1].clone()
        sim.velocities.zero_()
        sim.velocities[0, 2] = speed
        return sim, speed
    return sim, 2.0 * speed


def measure(sim: IceDEM3D, closing_in: float, max_steps: int = 400_000) -> dict:
    """Integrate to separation; return ``e`` plus contact diagnostics."""
    in_contact = False
    closest = float("inf")
    steps = 0
    while steps < max_steps:
        sim.step()
        steps += 1
        r = sim.positions[1] - sim.positions[0]
        separation = float(torch.linalg.vector_norm(r))
        if separation < 2.0 * RADIUS:
            in_contact = True
            closest = min(closest, separation)
        elif in_contact:
            closing_out = -float((sim.velocities[1] - sim.velocities[0]) @ (r / separation))
            return {
                "closing_in_m_per_s": closing_in,
                "closing_out_m_per_s": abs(closing_out),
                "restitution": abs(closing_out) / closing_in,
                "peak_overlap_m": 2.0 * RADIUS - closest,
                "contact_time_s": steps * float(sim.config.dt),
                "steps": steps,
                "dt_s": float(sim.config.dt),
            }
    raise RuntimeError("impact did not separate within max_steps")


# ---------------------------------------------------------------------------
# R1: independent RK4 of the clipped law in the relative coordinate
# ---------------------------------------------------------------------------
def _gamma(zeta: float, m_eff: float, *, model: str, delta: float) -> float:
    if model == "hertz":
        e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
        r_eff = hertz_effective_radius(RADIUS)
        k_s = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * math.sqrt(max(delta, 0.0))
        return 2.0 * zeta * math.sqrt(k_s * m_eff)
    return 2.0 * zeta * math.sqrt(5_000.0 * m_eff)


def reference_rk4(*, e_target: float, model: str, m_eff: float, closing_speed: float,
                  damping_m_eff: float | None = None,
                  dt: float = 1.0e-7, max_time: float = 5.0e-2) -> dict:
    """Integrate ``m_eff delta'' = -max(k delta + gamma delta', 0)`` with RK4.

    ``damping_m_eff`` is the mass used to build ``gamma = 2 zeta sqrt(k m_eff)``.
    It defaults to ``m_eff`` (self-consistent).  Passing a *smaller* value models
    the pre-fix engine, where the damping was built from ``m/2`` while the
    dynamics of a pinned pair used ``m``.  Note that the realised restitution of
    the self-consistent law is invariant to ``m_eff`` (it depends on ``zeta``
    only), so the mismatch -- not the size of ``m_eff`` -- is what shows up.
    """
    damp_mass = m_eff if damping_m_eff is None else damping_m_eff
    ln_e = math.log(e_target)
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)
    stiffness = 5_000.0

    def accel(delta: float, rate: float) -> float:
        gamma = _gamma(zeta, damp_mass, model=model, delta=delta)
        elastic = _hertz_force(delta) if model == "hertz" else stiffness * delta
        return -max(elastic + gamma * rate, 0.0) / m_eff

    delta, rate, t, peak = 0.0, closing_speed, 0.0, 0.0
    while t < max_time:
        k1u, k1v = rate, accel(delta, rate)
        k2u = rate + 0.5 * dt * k1v
        k2v = accel(delta + 0.5 * dt * k1u, k2u)
        k3u = rate + 0.5 * dt * k2v
        k3v = accel(delta + 0.5 * dt * k2u, k3u)
        k4u = rate + dt * k3v
        k4v = accel(delta + dt * k3u, k4u)
        delta += dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        rate += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        t += dt
        peak = max(peak, delta)
        if delta <= 0.0 and rate < 0.0:
            return {"restitution": abs(rate) / closing_speed, "peak_overlap_m": peak,
                    "contact_time_s": t}
    raise RuntimeError("RK4 reference did not separate")


def _hertz_force(delta: float) -> float:
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff) * max(delta, 0.0) ** 1.5


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(*, speeds=SPEEDS, targets=TARGETS, models=("linear", "hertz"),
          dt_factor: float = 0.6) -> list[dict]:
    cfg = DEM3DConfig(nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY)
    mass = cfg.mass
    rows: list[dict] = []
    for model in models:
        for e_target in targets:
            for speed in speeds:
                free = measure(*build_impact(speed, pinned=False, e_target=e_target,
                                             model=model, dt_factor=dt_factor))
                pin = measure(*build_impact(speed, pinned=True, e_target=e_target,
                                            model=model, dt_factor=dt_factor))
                ref_free = reference_rk4(e_target=e_target, model=model,
                                         m_eff=mass / 2.0, closing_speed=2.0 * speed)
                ref_pin = reference_rk4(e_target=e_target, model=model, m_eff=mass,
                                        closing_speed=speed)
                # Pre-fix engine: pinned pair dynamics with m, damping built
                # from m/2 -> zeta too small by sqrt(2).
                ref_wrong = reference_rk4(e_target=e_target, model=model, m_eff=mass,
                                          damping_m_eff=mass / 2.0,
                                          closing_speed=speed)
                rows.append({
                    "model": model, "e_target": e_target, "speed_in_m_per_s": speed,
                    "e_free_free": free["restitution"],
                    "e_pinned": pin["restitution"],
                    "e_target_rel_error_pinned": pin["restitution"] / e_target - 1.0,
                    "e_target_rel_error_free": free["restitution"] / e_target - 1.0,
                    "pinned_vs_free_rel": pin["restitution"] / free["restitution"] - 1.0,
                    "rk4_m_eff_half_free": ref_free["restitution"],
                    "rk4_m_eff_full_pinned": ref_pin["restitution"],
                    "rk4_pre_fix_inconsistent": ref_wrong["restitution"],
                    "engine_pinned_vs_rk4_full": pin["restitution"] / ref_pin["restitution"] - 1.0,
                    "engine_free_vs_rk4_half": free["restitution"] / ref_free["restitution"] - 1.0,
                    "peak_overlap_pinned_m": pin["peak_overlap_m"],
                    "peak_overlap_rk4_m": ref_pin["peak_overlap_m"],
                    "contact_time_pinned_s": pin["contact_time_s"],
                    "contact_time_rk4_s": ref_pin["contact_time_s"],
                    "dt_s": pin["dt_s"],
                })
    return rows


def render(rows: list[dict], legacy: list[dict] | None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []

    # Figure 1: realised restitution vs impact speed, pinned vs free-free, per model.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    for ax, model in zip(axes, ("linear", "hertz")):
        sub = [r for r in rows if r["model"] == model]
        for e_target, marker in ((0.5, "o"), (0.8, "s")):
            pts = sorted([r for r in sub if r["e_target"] == e_target],
                         key=lambda r: r["speed_in_m_per_s"])
            v = [r["speed_in_m_per_s"] for r in pts]
            ax.plot(v, [r["e_pinned"] for r in pts], marker + "-",
                    label=f"pinned support, e_target={e_target}")
            ax.plot(v, [r["e_free_free"] for r in pts], marker + "--", alpha=0.75,
                    label=f"free-free pair, e_target={e_target}")
            ax.axhline(e_target, color="k", lw=0.8, ls=":", alpha=0.6)
        if legacy:
            lg = sorted([r for r in legacy if r["model"] == model and r["e_target"] == 0.5],
                        key=lambda r: r["speed_in_m_per_s"])
            if lg:
                ax.plot([r["speed_in_m_per_s"] for r in lg],
                        [r["e_pinned"] for r in lg], "x-", color="crimson", alpha=0.8,
                        label="pre-fix (m_eff = m/2), e_target=0.5")
        ax.set_xscale("log")
        ax.set_xlabel("closing speed [m/s]")
        ax.set_ylabel("realised coefficient of restitution")
        ax.set_title(f"{model} contact law")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Fixed-support impact: m_eff = m/2 (pre-fix) vs m_eff = m (fixed)")
    fig.tight_layout()
    path = OUT / "restitution_vs_speed.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    produced.append(str(path.relative_to(ROOT)))

    # Figure 2: attribution — engine vs the two RK4 mass conventions.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    models = sorted({r["model"] for r in rows})
    for ax, model in zip(axes, models):
        sub = [r for r in rows if r["model"] == model and r["e_target"] == 0.5]
        sub.sort(key=lambda r: r["speed_in_m_per_s"])
        x = range(len(sub))
        ax.plot(x, [r["e_pinned"] for r in sub], "o-", label="engine, pinned support")
        ax.plot(x, [r["rk4_m_eff_full_pinned"] for r in sub], "^--",
                label="RK4, m_eff = m (correct)")
        ax.plot(x, [r["rk4_pre_fix_inconsistent"] for r in sub], "v:",
                label="RK4, dynamics m / damping m/2 (pre-fix)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{r['speed_in_m_per_s']:.2f}" for r in sub])
        ax.set_xlabel("closing speed [m/s]")
        ax.set_ylabel("realised restitution (e_target = 0.5)")
        ax.set_title(f"{model} contact law")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Attribution: the pre-fix overshoot is exactly the m_eff = m/2 branch")
    fig.tight_layout()
    path = OUT / "mass_convention_attribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    produced.append(str(path.relative_to(ROOT)))
    return produced


def legacy_probe(legacy_src: str, *, quick: bool) -> list[dict]:
    """Run the identical sweep against an archived (pre-fix) source tree."""
    import os
    import subprocess

    env = dict(os.environ)
    env["TENSORDEM_SRC"] = legacy_src
    out = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                          "--emit-json", "--quick" if quick else "--full"],
                         capture_output=True, text=True, env=env, check=True)
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    return payload["rows"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--emit-json", action="store_true")
    ap.add_argument("--legacy-src", default=None)
    args = ap.parse_args()

    speeds = SPEEDS[:2] if args.quick else SPEEDS
    targets = (0.5,) if args.quick else TARGETS
    rows = sweep(speeds=speeds, targets=targets,
                 models=("linear", "hertz"))
    payload = {"protocol": PROTOCOL, "engine": "fixed", "rows": rows}
    if args.emit_json:
        print(json.dumps(payload))
        return 0

    legacy = legacy_probe(args.legacy_src, quick=args.quick) if args.legacy_src else None
    if legacy is not None:
        payload["legacy_rows"] = legacy
        worst = max((abs(r["pinned_vs_free_rel"]) for r in legacy), default=0.0)
        print(f"pre-fix (m_eff = m/2) worst pinned-vs-free-free deviation: {100*worst:.1f}%")
        for r in legacy:
            print(f"  legacy {r['model']:6s} e_t={r['e_target']} v={r['speed_in_m_per_s']}"
                  f"  e_pinned={r['e_pinned']:.4f}"
                  f"  ({100*r['e_target_rel_error_pinned']:+.1f}% of target)")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sweep.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    figures = render(rows, legacy) if args.render else []
    payload["figures"] = figures
    (OUT / "sweep.json").write_text(json.dumps(payload, indent=2))

    print(f"{'model':6s} {'e_t':>4s} {'v[m/s]':>7s} {'e_pinned':>9s} {'e_free':>8s} "
          f"{'pin/free-1':>11s} {'pin vs RK4(m)':>14s} {'pin vs e_t':>11s}")
    for r in rows:
        print(f"{r['model']:6s} {r['e_target']:4.1f} {r['speed_in_m_per_s']:7.2f} "
              f"{r['e_pinned']:9.4f} {r['e_free_free']:8.4f} "
              f"{100*r['pinned_vs_free_rel']:10.2f}% {100*r['engine_pinned_vs_rk4_full']:13.2f}% "
              f"{100*r['e_target_rel_error_pinned']:10.2f}%")
    print(f"worst |pinned vs free-free|: "
          f"{100*max(abs(r['pinned_vs_free_rel']) for r in rows):.3f}%")
    print(f"worst |engine pinned vs RK4(m_eff=m)|: "
          f"{100*max(abs(r['engine_pinned_vs_rk4_full']) for r in rows):.3f}%")
    print(f"artifacts -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())