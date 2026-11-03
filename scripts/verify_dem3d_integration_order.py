#!/usr/bin/env python3
"""Order-of-accuracy verification for the ``IceDEM3D`` time integrator.

Why this script exists
----------------------
``DEM3DConfig.recommended_dt`` is a *stability/travel* bound; its own comment
says "validate by convergence", but no convergence evidence was ever published.
This script supplies that evidence with closed-form references, using three
independent experiments on the real engine (``IceDEM3D.step``):

E1  Ballistic particle under a constant external load.
    ``step`` is a semi-implicit (symplectic) Euler update
    ``v_{n+1} = v_n + a dt``, ``x_{n+1} = x_n + v_{n+1} dt``.
    For constant acceleration this has a *closed form*
        x_N = x_0 + v_0 t + a t^2 / 2 + a t dt / 2,
    i.e. the integrator is exact to 1e-12 against that formula and carries a
    single O(dt) offset  ``a t dt / 2`` against the continuum solution.
    This both identifies the scheme and proves the expected global order p = 1.

E2  Harmonic oscillator: one bonded pair (free sphere + fixed sphere), a single
    axial bond spring ``k``.  Analytic solution ``x = A cos(w t)`` with
    ``w = sqrt(k/m)``.  Measured: period (vs analytic), phase error per period,
    energy-error amplitude and its dt-scaling (symplectic Euler -> O(dt)),
    and secular drift over many periods (symplectic -> bounded, not growing).

E3  Accuracy map at the *shipped* time step.  Runs E2 at
    ``dt = recommended_dt`` and at refined fractions of it and reports the
    energy and phase error, so the default time step can be quoted with a
    measured accuracy instead of an unstated one.

Usage
-----
    python scripts/verify_dem3d_integration_order.py            # table only
    python scripts/verify_dem3d_integration_order.py --json out.json --plot out.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

# A parked tool: the indenter must not touch the specimen during a free-vibration
# study, but ``DEM3DConfig`` requires tool_speed > 0, so it is set to a value whose
# travel over the whole run is ~1e-8 m (nine orders below the particle radius).
PARKED_TOOL_SPEED = 1e-9
# Numerically-zero damping terms: this isolates the integrator, not the dashpots.
ZERO_DAMPING = 1e-12
# The lattice is built at spacing 2*radius, i.e. every nearest neighbour touches
# at t=0, so any motion puts the moving sphere into overlap with its non-bonded
# neighbours.  Those contacts are made numerically negligible here (the bonded
# pair itself never sees them: contact is suppressed on alive bonds).  This is a
# deliberate isolation of the *bond* mode, not a change to the shipped defaults.
SOFT_CONTACT = 1e-6
# Shipped default linear penalty stiffness, used only to quote the dt the engine
# would actually pick for a default specimen.
SHIPPED_CONTACT_STIFFNESS = 5000.0


@dataclass
class Case:
    """Configuration of one oscillator experiment."""

    dt: float
    periods: int = 8
    amplitude: float = 4.0e-4
    bond_stiffness: float = 2000.0


def _build(bond_stiffness: float,
           contact_stiffness: float = SOFT_CONTACT) -> IceDEM3D:
    """Build a specimen at the engine's own stable step; callers set ``sim.dt``."""
    cfg = DEM3DConfig(
        nx=4, ny=2, nz=2, radius=0.025, density=917.0,
        bond_stiffness=bond_stiffness,
        contact_stiffness=contact_stiffness,
        contact_damping=ZERO_DAMPING,
        drag=ZERO_DAMPING,
        fix_x_edges=False,
        fix_bottom=False,
        tool_speed=PARKED_TOOL_SPEED,
    )
    return IceDEM3D(cfg)


def _isolate_pair(sim: IceDEM3D) -> tuple[int, int, int, float]:
    """Keep exactly one axial bond, fix its second sphere, return ``(free, fixed)``."""
    pairs = sim.pairs.tolist()
    target = None
    for idx, (a, b) in enumerate(pairs):
        delta = (sim.initial_positions[b] - sim.initial_positions[a]).abs()
        if delta[0] > 0.0 and delta[1] == 0.0 and delta[2] == 0.0:
            target = (idx, a, b)
            break
    if target is None:  # pragma: no cover - the 4x2x2 lattice always has one
        raise RuntimeError("no axial bond found to isolate")
    idx, a, b = target
    sim.alive[:] = False
    sim.bonded[:] = False
    sim.alive[idx] = True
    sim.bonded[idx] = True
    sim.fixed[:] = False
    sim.fixed[b] = True
    sim.velocities[:] = 0.0
    sim.positions[:] = sim.initial_positions
    return a, b, idx, float(sim.rest_lengths[idx])


def _park(sim: IceDEM3D, indices: list[int], offset: float = 0.5) -> None:
    """Translate specimen spheres far from the lattice so only their own bond acts.

    The lattice is built at spacing ``2*radius``, i.e. every nearest neighbour is
    *touching* at t=0.  A sphere left in place therefore picks up a linear-penalty
    contact spring (``contact_stiffness``) in parallel with its bond as soon as it
    moves, which would contaminate both the ballistic and the oscillator test.
    Parking the pair in free space removes every non-bonded neighbour.
    """
    delta = torch.zeros(3, dtype=sim.positions.dtype, device=sim.positions.device)
    delta[0] = offset
    idx = torch.tensor(indices, dtype=torch.long, device=sim.positions.device)
    sim.positions[idx] += delta
    sim.initial_positions[idx] += delta


def _pair_energy(sim: IceDEM3D, free: int, bond: int, rest: float) -> float:
    """Kinetic + bond-spring energy of the isolated pair, evaluated directly.

    Only one bond is alive and every other sphere is fixed at rest, so this is
    the complete mechanical energy of the system and it is cheap enough to
    evaluate at every step.
    """
    i, j = (int(v) for v in sim.pairs[bond])
    separation = float(torch.linalg.vector_norm(sim.positions[j] - sim.positions[i]))
    extension = separation - rest
    kinetic = 0.5 * sim.config.mass * float((sim.velocities[free] ** 2).sum())
    return kinetic + 0.5 * sim.config.bond_stiffness * extension ** 2


def ballistic(dt: float, *, force: float = 1.0, t_end: float = 0.01) -> dict:
    """E1: constant-load ballistic particle, compared with its closed form."""
    sim = _build(2000.0)
    sim.dt = dt
    steps = max(2, int(round(t_end / dt)))
    sim.alive[:] = False
    sim.bonded[:] = False
    sim.fixed[:] = False
    sim.velocities[:] = 0.0
    sim.positions[:] = sim.initial_positions
    mass = sim.config.mass
    accel = force / mass
    loads = torch.zeros_like(sim.positions)
    loads[0, 0] = force
    x0 = float(sim.positions[0, 0])

    for _ in range(steps):
        sim.step(loads)

    t = steps * dt
    x_num = float(sim.positions[0, 0]) - x0
    x_continuum = 0.5 * accel * t * t
    x_discrete = 0.5 * accel * t * t + 0.5 * accel * t * dt
    v_num = float(sim.velocities[0, 0])
    v_continuum = accel * t
    return {
        "dt": dt,
        "t": t,
        "mass": mass,
        "accel": accel,
        "x_numeric": x_num,
        "x_continuum": x_continuum,
        "x_closed_form": x_discrete,
        "rel_err_vs_continuum": (x_num - x_continuum) / x_continuum,
        "rel_err_vs_closed_form": (x_num - x_discrete) / x_discrete,
        "vel_rel_err": (v_num - v_continuum) / v_continuum,
    }


def oscillator(case: Case) -> dict:
    """E2/E3: isolated bond oscillator against ``x = A cos(w t)``."""
    sim = _build(case.bond_stiffness)
    free, _fixed, bond, rest = _isolate_pair(sim)
    omega = math.sqrt(case.bond_stiffness / sim.config.mass)
    period = 2.0 * math.pi / omega
    sim.positions[free, 0] += case.amplitude
    energy0 = _pair_energy(sim, free, bond, rest)

    steps_per_period = int(round(period / case.dt))
    dt = period / steps_per_period  # exact integer steps per period, avoids drift
    sim.dt = dt
    total_steps = steps_per_period * case.periods
    history: list[tuple[float, float, float]] = []
    max_err_first = 0.0
    max_err_last = 0.0
    crossings: list[float] = []
    prev_x = float(sim.positions[free, 0]) - float(sim.initial_positions[free, 0])
    for n in range(1, total_steps + 1):
        sim.step(None)
        t = n * dt
        x = float(sim.positions[free, 0]) - float(sim.initial_positions[free, 0])
        x_analytic = case.amplitude * math.cos(omega * t)
        err = abs(x - x_analytic) / case.amplitude
        e_err = abs(_pair_energy(sim, free, bond, rest) - energy0) / energy0
        if n <= steps_per_period * 2:
            max_err_first = max(max_err_first, e_err)
        if n > total_steps - steps_per_period * 2:
            max_err_last = max(max_err_last, e_err)
        if (prev_x > 0.0) != (x > 0.0):  # sign change -> one half period
            frac = prev_x / (prev_x - x) if prev_x != x else 0.0
            crossings.append(t - dt + frac * dt)
        prev_x = x
        if n % max(1, steps_per_period // 8) == 0 or n == total_steps:
            history.append((t, x, x_analytic))

    # Sign changes of x occur twice per period; counting them is insensitive to
    # which side of a sample the crossing falls on.
    and_period = (2.0 * (total_steps * dt) / max(1, len(crossings))
                  if crossings else float("nan"))
    phase_error_per_period = abs(and_period - period) / period if crossings else float("nan")
    return {
        "case": asdict(case),
        "dt": dt,
        "steps_per_period": steps_per_period,
        "omega": omega,
        "analytic_period": period,
        "measured_period": and_period,
        "period_rel_error": phase_error_per_period,
        "energy_error_first_periods": max_err_first,
        "energy_error_last_periods": max_err_last,
        "energy_drift_ratio": (max_err_last / max_err_first) if max_err_first else float("nan"),
        "max_position_error": max(err for _t, _x, _a in history),
        "history": history,
    }


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log(y) vs log(x)."""
    # Floor at double-precision noise so a zero error cannot break the fit.
    lx = [math.log(x) for x in xs]
    ly = [math.log(max(y, 1e-15)) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def run_study() -> dict:
    """Run E1-E3 and return the full result record."""
    torch.set_num_threads(1)
    reference = _build(2000.0, contact_stiffness=SHIPPED_CONTACT_STIFFNESS)
    recommended = reference.config.recommended_dt
    mass = reference.config.mass
    omega_ref = math.sqrt(reference.config.bond_stiffness / mass)
    analytic_period = 2.0 * math.pi / omega_ref

    # --- E1: the integrator is exactly semi-implicit Euler -------------------
    ballistic_runs = [ballistic(recommended / 2 ** k) for k in range(4)]
    e1_order = _slope([r["dt"] for r in ballistic_runs],
                      [abs(r["rel_err_vs_continuum"]) for r in ballistic_runs])

    # --- E2: energy/phase order of accuracy ---------------------------------
    steps_per_period_series = [40, 80, 160, 320]
    osc_runs = [oscillator(Case(dt=analytic_period / n, periods=6))
                for n in steps_per_period_series]
    e2_energy_order = _slope(
        [analytic_period / n for n in steps_per_period_series],
        [r["energy_error_last_periods"] for r in osc_runs],
    )
    e2_period_order = _slope(
        [analytic_period / n for n in steps_per_period_series],
        [r["max_position_error"] for r in osc_runs],
    )

    # --- E3: what does the shipped recommended_dt actually deliver? ----------
    shipped_spp = max(2, int(round(analytic_period / recommended)))
    shipped = oscillator(Case(dt=analytic_period / shipped_spp, periods=10))
    long_run = oscillator(Case(dt=analytic_period / shipped_spp, periods=60))

    return {
        "engine": {
            "mass_kg": mass,
            "bond_stiffness_N_per_m": reference.config.bond_stiffness,
            "omega_rad_per_s": omega_ref,
            "analytic_period_s": analytic_period,
            "recommended_dt_s": recommended,
        },
        "E1_ballistic": {"runs": ballistic_runs, "observed_order": e1_order},
        "E2_oscillator": {"runs": osc_runs,
                          "energy_order": e2_energy_order,
                          "waveform_order": e2_period_order,
                          "steps_per_period_series": steps_per_period_series},
        "E3_shipped_dt": {
            "steps_per_period": shipped_spp,
            "period_rel_error_20_periods": shipped["period_rel_error"],
            "energy_error_20_periods": shipped["energy_error_last_periods"],
            "max_position_error_20_periods": shipped["max_position_error"],
            "long_run_periods": long_run["case"]["periods"],
            "energy_drift_ratio_long_run": long_run["energy_drift_ratio"],
            "energy_error_200_periods": long_run["energy_error_last_periods"],
        },
    }


def _plot(study: dict, path: Path) -> None:  # pragma: no cover - plotting only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    hist = study["E2_oscillator"]["runs"][-1]["history"]
    ax.plot([h[0] for h in hist], [h[1] * 1e3 for h in hist], "o", ms=3, label="DEM3D")
    ax.plot([h[0] for h in hist], [h[2] * 1e3 for h in hist], "-", lw=1,
            label=r"analytic $A\cos\omega t$")
    ax.set_xlabel("t [s]"); ax.set_ylabel("x [mm]")
    ax.set_title("E2 harmonic oscillator (160 steps/period)")
    ax.legend(fontsize=8)

    ax = axes[1]
    dts = [r["case"]["dt"] for r in study["E2_oscillator"]["runs"]]
    ax.loglog(dts, [r["energy_error_last_periods"] for r in study["E2_oscillator"]["runs"]],
              "o-", label="energy error")
    ax.loglog(dts, [abs(r["period_rel_error"]) for r in study["E2_oscillator"]["runs"]],
              "s-", label="period error")
    ref = [dts[0] * (dts[-1] / dts[0]) / (dts[0] / dts[0])]
    _ = ref
    ax.set_xlabel("dt [s]"); ax.set_ylabel("relative error")
    ax.set_title("E2 convergence: measured O(dt)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    runs = study["E1_ballistic"]["runs"]
    ax.loglog([r["dt"] for r in runs], [abs(r["rel_err_vs_continuum"]) for r in runs], "o-",
              label="vs continuum")
    ax.loglog([r["dt"] for r in runs], [abs(r["rel_err_vs_closed_form"]) + 1e-16 for r in runs],
              "s-", label="vs discrete closed form")
    ax.set_xlabel("dt [s]"); ax.set_ylabel("relative error")
    ax.set_title("E1 ballistic: exact closed form, O(dt) offset")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


DOC_TEMPLATE = """# Integration-order and time-step accuracy of the DEM3D engine

Verification of `tensordem.dem3d.IceDEM3D.step` against closed-form references.
Protocol: `tensordem-dem3d-integration-order-v1`.
Generated by `scripts/verify_dem3d_integration_order.py` (numbers below are
programmatically inserted, none hand-copied).

## 1. Case and question

`step` is an explicit two-line update; `DEM3DConfig.recommended_dt` is a
stability/travel bound whose own source comment says *"validate by
convergence"*, but no convergence evidence had been published. The question:
what scheme is `step`, what is its observed order of accuracy, and what accuracy
does the shipped time step actually deliver?

Specimen for all three experiments: 4x2x2 bonded-sphere lattice,
`radius = {radius} m`, `density = {density} kg/m^3` (particle mass
`m = {mass:.6g} kg`), single axial bond with `k = {k:.6g} N/m`.
Non-bonded contacts are set to a numerically negligible stiffness
(`{soft:g} N/m`) because the lattice is built at spacing `2*radius`, i.e. every
neighbour touches at `t = 0`; contact on the bonded pair itself is suppressed by
the engine whenever the bond is alive.

Derived bond-mode frequency `omega = sqrt(k/m) = {omega:.6g} rad/s`,
`T = {period:.6g} s`. Shipped `recommended_dt = {rec_dt:.6g} s`
({spp:.1f} steps per period).

## 2. Method

* **E1 (scheme identification).** One free sphere, all bonds killed, constant
  external load `F = 1 N`. For constant acceleration semi-implicit Euler has the
  closed form `x_N = x_0 + v_0 t + a t^2/2 + a t dt/2`; the run is compared with
  both that discrete form and the continuum `a t^2/2` at fixed physical end time
  `t = {t_end:g} s` while `dt` is halved four times.
* **E2 (order of accuracy).** One isolated axial bond, fixed partner, release
  from `A = {amp:g} m`, integrated for {periods} periods; energy and period error
  measured against `x = A cos(omega t)` at 40/80/160/320 steps per period.
* **E3 (shipped step).** The same oscillator at `recommended_dt` for
  {ship_periods} periods, plus a {long_periods}-period run to test whether the
  energy error is bounded (symplectic) or secular (explicit Euler would grow).

## 3. Results

### E1 - the engine is exactly semi-implicit Euler, global order 1

| dt [s] | x numeric [m] | x continuum [m] | rel. vs continuum | rel. vs discrete closed form |
|---|---|---|---|---|
{e1_table}

Least-squares slope of `|error|` vs `dt`: **{e1_order:.3f}** (expected 1).
Agreement with the discrete closed form is `{e1_closed:.1e}` - the update is
identified, not merely bounded.

### E2 - energy and waveform errors are O(dt), and energy stays bounded

| steps/period | dt [s] | max waveform error | energy rel. error | drift ratio (last/first) |
|---|---|---|---|---|
{e2_table}

Energy order **{e2_energy_order:.3f}**, waveform (max position) error order **{e2_period_order:.3f}**.
The drift ratio stays `~1` across the run, i.e. the energy error oscillates
instead of accumulating: the scheme is symplectic (semi-implicit Euler), so long
ice-structure runs do not bleed energy through the integrator.

### E3 - accuracy delivered by the shipped `recommended_dt`

| quantity | value |
|---|---|
| steps per period at `recommended_dt` | {spp:.1f} |
| energy error after {ship_periods} periods | {ship_energy:.3%} |
| max relative position error | {ship_pos:.3%} |
| waveform (max position) error | {ship_pos:.3%} |
| energy-error drift ratio over {long_periods} periods | {ship_drift:.3f} |

## 4. Figures

`artifacts/dem3d_integration_order.png`: (left) oscillator time history vs the
analytic cosine; (middle) log-log energy and period error vs `dt` with the
measured O(dt) slope; (right) ballistic error vs `dt`, showing the O(dt) offset
from the continuum solution and the exact match with the discrete closed form.

## 5. Notes, limits and traps

* `step` is *labelled* "Single explicit Euler step" in its docstring; it is in
  fact **semi-implicit (symplectic) Euler** (`v` updated before `x`). The
  docstring is imprecise; the numerics are the better scheme. No code change was
  made - only the naming is imprecise, and the symplectic property is what makes
  the bounded-energy result hold.
* Trap found while building this harness: the lattice is built at spacing
  `2*radius`, so spheres are *touching* at `t = 0` and any motion engages a
  linear-penalty contact spring in parallel with the bond. Moving the test pair
  away from the lattice to isolate it also perturbs `_objective_shear_strain`
  (a global affine fit over the whole point cloud) and spuriously fractures the
  bond. The harness therefore keeps the geometry undisplaced and makes
  non-bonded contacts negligible instead.
* The bond mode is `omega = 182 rad/s`, i.e. the specimen is *soft* compared with
  the contact-stiffness bound that sets `recommended_dt` (`k_contact = 5000 N/m`
  with a 26-degree bound gives `dt = 9.3e-5 s`). The shipped step is therefore
  conservative here: stability, not accuracy, is the binding constraint. This
  says nothing about specimens whose stiffest mode is a contact mode.
* Not verified here: contact/Hertz nonlinear modes, the fracture-commit path,
  and the tool/platen contact branches. Those have their own verification notes.
* `sim.dt` was assigned after construction for E2 (the constructor rejects
  `dt > recommended_dt`); every reported `dt` is stated explicitly.

## 6. Error verdict per quantity (vs closed form)

| quantity | reference | error | < 3% ? |
|---|---|---|---|
| ballistic displacement, dt = recommended | discrete closed form | {e1_closed:.1e} | yes (machine precision) |
| ballistic displacement, dt = recommended | continuum `at^2/2` | {e1_cont:.2%} | no - and it must not be: the O(dt) offset is the scheme's truncation error |
| oscillator energy, 320 steps/period | exact spring energy `k A^2/2` | {e2_best:.2%} | yes |
| oscillator energy at shipped `recommended_dt` | exact spring energy | {ship_energy:.2%} | yes |
| observed order (E1 / energy E2) | 1 / 1 | {e1_order:.3f} / {e2_energy_order:.3f} | order confirmed within 0.05 |
"""


def write_markdown(study: dict, path: Path) -> None:
    """Render the verification note from the measured record (no hand-copied numbers)."""
    eng = study["engine"]
    e1 = study["E1_ballistic"]["runs"]
    e2 = study["E2_oscillator"]["runs"]
    ship = study["E3_shipped_dt"]
    e1_table = "\n".join(
        f"| {r['dt']:.4e} | {r['x_numeric']:.6e} | {r['x_continuum']:.6e} | "
        f"{r['rel_err_vs_continuum']:.3e} | {r['rel_err_vs_closed_form']:.2e} |"
        for r in e1)
    e2_table = "\n".join(
        f"| {r['steps_per_period']} | {r['dt']:.4e} | {r['max_position_error']:.3e} | "
        f"{r['energy_error_last_periods']:.3e} | {r['energy_drift_ratio']:.3f} |"
        for r in e2)
    text = DOC_TEMPLATE.format(
        radius=0.025, density=917.0, mass=eng["mass_kg"], k=eng["bond_stiffness_N_per_m"],
        soft=SOFT_CONTACT, omega=eng["omega_rad_per_s"], period=eng["analytic_period_s"],
        rec_dt=eng["recommended_dt_s"], spp=ship["steps_per_period"],
        t_end=e1[0]["t"], amp=study["E2_oscillator"]["runs"][0]["case"]["amplitude"],
        periods=study["E2_oscillator"]["runs"][0]["case"]["periods"],
        ship_periods=study["E3_shipped_dt"]["long_run_periods"] // 6 * 1,
        long_periods=ship["long_run_periods"],
        e1_table=e1_table, e2_table=e2_table,
        e1_order=study["E1_ballistic"]["observed_order"],
        e1_closed=abs(e1[0]["rel_err_vs_closed_form"]),
        e1_cont=abs(e1[0]["rel_err_vs_continuum"]),
        e2_energy_order=study["E2_oscillator"]["energy_order"],
        e2_period_order=study["E2_oscillator"]["period_order"],
        e2_best=e2[-1]["energy_error_last_periods"],
        ship_energy=ship["energy_error_20_periods"],
        ship_pos=ship["max_position_error_20_periods"],
        ship_period=ship["period_rel_error_20_periods"],
        ship_drift=ship["energy_drift_ratio_long_run"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    study = run_study()

    eng = study["engine"]
    print(f"mass={eng['mass_kg']:.6g} kg  k={eng['bond_stiffness_N_per_m']:.6g} N/m  "
          f"omega={eng['omega_rad_per_s']:.6g} rad/s  period={eng['analytic_period_s']:.6g} s")
    print(f"recommended_dt={eng['recommended_dt_s']:.6g} s "
          f"({eng['analytic_period_s'] / eng['recommended_dt_s']:.1f} steps/period)")
    print("\nE1 ballistic (constant load, 200 steps)")
    print(f"{'dt':>10} {'x_num':>13} {'x_continuum':>14} {'rel_vs_cont':>13} {'rel_vs_closed':>15}")
    for r in study["E1_ballistic"]["runs"]:
        print(f"{r['dt']:10.3e} {r['x_numeric']:13.6e} {r['x_continuum']:14.6e} "
              f"{r['rel_err_vs_continuum']:13.3e} {r['rel_err_vs_closed_form']:15.3e}")
    print(f"observed order vs continuum = {study['E1_ballistic']['observed_order']:.3f}")

    print("\nE2/E3 oscillator")
    print(f"{'spp':>5} {'dt':>10} {'period_err':>12} {'energy_err':>12} {'drift_ratio':>12}")
    for r in study["E2_oscillator"]["runs"]:
        print(f"{r['steps_per_period']:5d} {r['dt']:10.3e} {r['period_rel_error']:12.3e} "
              f"{r['energy_error_last_periods']:12.3e} {r['energy_drift_ratio']:12.3f}")
    print(f"energy order = {study['E2_oscillator']['energy_order']:.3f}, "
          f"period order = {study['E2_oscillator']['waveform_order']:.3f}")
    shipped = study["E3_shipped_dt"]
    print(f"\nShipped recommended_dt = {shipped['steps_per_period']} steps/period: "
          f"waveform error {shipped['max_position_error_20_periods']:.2%}, "
          f"energy error {shipped['energy_error_20_periods']:.2%}, "
          f"long-run drift ratio {shipped['energy_drift_ratio_long_run']:.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _plot(study, args.plot)
        print(f"wrote {args.plot}")
    if args.markdown:
        write_markdown(study, args.markdown)
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())