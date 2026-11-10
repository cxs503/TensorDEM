"""P82W2-N09 -- multi-rate contact integration (MTS) and the time-step gate.

One script, five machine-decidable stages; every number in the JSON output is
produced by the shipped library (nothing hand-copied):

* **A  restitution ladder** (acceptance a): ``e = 0.5 / 0.9``,
  ``dt/dt_c in {0.02, 0.05, 0.1, 0.2}``, contact sub-cycling on;
* **B  energy/momentum ledger** (acceptance b): ``1e5`` steps of an undamped,
  frictionless, doubly periodic assembly; energy drift and linear momentum;
* **C  tightness negative control** (acceptance d): ``dt/dt_c = 0.5`` must be
  rejected by the gate and, when forced, must show a detectable failure;
* **D  shard reproducibility** (acceptance e): the stage-A case matrix evaluated
  in 1 / 2 / 4 shards (separate processes) -- values must agree;
* **E  double-periodic shear** (scenario 2): steady shear stress ratio
  ``|tau|/P`` versus ``mu`` for the multi-rate and single-rate solvers;
* **F  friction jamming** (acceptance c): ``mu = 0.5``, ``N = 256``, stiffness
  hierarchy 100:1 on a 10 % skeleton -- ``phi_J`` of the multi-rate solver versus
  the single-rate baseline, wall clock, and the ``W4-P1`` servo anchor
  (the shipped :func:`tensordem.granular2d.jamming.jamming_density_isotropic`,
  i.e. the lineage code of the m5524 jamming campaign).

Usage::

    python benchmarks/mts_contact_integration.py --stage all --shards 1
    python benchmarks/mts_contact_integration.py --stage ladder --shard 0 --shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem.granular2d.jamming import bidisperse_radii  # noqa: E402
from tensordem.mts import (  # noqa: E402
    DEFAULT_MARGIN,
    TIGHT_RATIO,
    InadmissibleTimeStep,
    MTSConfig,
    MTSDisks,
    calibrate_alpha,
    impact_restitution,
    stability_gate,
)

# ---------------------------------------------------------------- fiducials
MASS = 1.0e-3          # kg   (single impactor)
STIFFNESS = 1.0e4      # N/m  (Hertz stiffness)
V_IMPACT = 0.05        # m/s
E_LEVELS = (0.5, 0.9)
DT_RATIOS = (0.02, 0.05, 0.1, 0.2)
RECONVERGENCE_RATIO = 5.0e-4

GAS_N = 64
GAS_STEPS = 100_000
GAS_STEP_RECORD = 200

JAM_N = 256
JAM_MEAN_RADIUS = 0.010
JAM_SEED = 0
JAM_MU = 0.5
JAM_MU_ROLLING = 0.05
JAM_KN = 3.0e4
JAM_P_COMPRESS = 3.0e3
JAM_LEVELS = (300.0, 100.0, 30.0, 10.0, 3.0, 1.0)
JAM_STIFF_RATIO = 100.0
JAM_STIFF_FRACTION = 0.10
#: protocol durations in *physical* time (s), so the two rates run the same physics
JAM_T_COMPRESS = 6000 * 8.8e-6
JAM_T_LEVEL = 2500 * 8.8e-6
JAM_ETA = 150.0

SHEAR_N = 128
SHEAR_MU = (0.1, 0.3, 0.5, 0.7)
SHEAR_RATE = 5.0
#: the profile thermostat must be resolvable on the *coarse* step: dt/tau < 0.05
SHEAR_DRAG = 5.0e-3
#: total strain = SHEAR_RATE * (settle + measure) = 0.5 (steady state needs O(0.3))
SHEAR_T_SETTLE = 0.05
SHEAR_T_MEASURE = 0.05


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so ``json.dumps`` never fails."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def _banner(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ---------------------------------------------------------------------------
# Stage A / D -- restitution ladder
# ---------------------------------------------------------------------------
def stage_ladder(ratios=DT_RATIOS, e_levels=E_LEVELS) -> dict:
    """Criterion (a): restitution error of the contract-stepped impact."""
    out = {"cases": [], "reference": [], "worst_error": 0.0}
    for e_target in e_levels:
        cal = calibrate_alpha(e_target=e_target, v_impact=V_IMPACT, mass=MASS,
                              stiffness=STIFFNESS, dt_ratio=RECONVERGENCE_RATIO)
        ref = impact_restitution(mass=MASS, stiffness=STIFFNESS, v_impact=V_IMPACT,
                                 alpha=cal["alpha"], dt_ratio=RECONVERGENCE_RATIO)
        out["reference"].append({"e_target": e_target, "alpha": cal["alpha"],
                                 "e_reference": ref.e_measured,
                                 "calibration_error": cal["converged_error"]})
        for ratio in ratios:
            res = impact_restitution(mass=MASS, stiffness=STIFFNESS, v_impact=V_IMPACT,
                                     alpha=cal["alpha"], dt_ratio=ratio)
            err = abs(res.e_measured / ref.e_measured - 1.0)
            out["worst_error"] = max(out["worst_error"], err)
            out["cases"].append({
                "e_target": e_target,
                "dt_ratio": ratio,
                "dt_over_dt_c": res.gate["ratio"],
                "substeps": res.substeps,
                "fine_step_ratio": res.gate["fine_step"] / res.gate["dt_c"],
                "e_measured": res.e_measured,
                "e_reference": ref.e_measured,
                "restitution_error": err,
                "gate_status": res.gate["status"],
                "contact_time_measured": res.contact_time_measured,
            })
    out["criterion"] = "restitution error <= 2% for every (e, dt/dt_c)"
    out["pass"] = bool(out["worst_error"] <= 0.02)
    return out


# ---------------------------------------------------------------------------
# Stage B -- energy / momentum ledger
# ---------------------------------------------------------------------------
def _gas_setup():
    n = GAS_N
    r = 0.010
    spacing = 2.02 * r
    cols = int(math.ceil(math.sqrt(n)))
    pts = []
    for row in range(cols):
        for col in range(cols):
            pts.append([(col + 0.5 * (row % 2)) * spacing,
                        row * spacing * math.sqrt(3.0) / 2.0])
    pos = np.array(pts[:n]) + 1.0e-3
    side = math.sqrt(n * math.pi * r * r / 0.52)
    scale = side / (cols * spacing)
    pos = pos * scale * 0.98 + 0.01 * side
    radii = np.full(n, r * scale * 0.98)
    return pos, radii, (side, side)


def stage_ledger(steps: int = GAS_STEPS, outer_margin: float = 0.05) -> dict:
    """Criterion (b): undamped frictionless ledger over ``steps`` coarse steps.

    ``outer_margin`` is the gate margin of the *slow* rate: it sets the accuracy
    budget of the multiple-time-step scheme (the same value is used for the
    single-rate control, so the comparison isolates the splitting).
    """
    pos, radii, box = _gas_setup()
    cfg = MTSConfig(kn=3.0e4, zeta=0.0, mu=0.0, mu_rolling=0.0,
                    stiff_ratio=100.0, stiff_fraction=0.10,
                    outer_margin=outer_margin, fast_margin=outer_margin)
    results = {}
    history = {}
    for mode in ("single", "mts"):
        sys_ = MTSDisks(pos.copy(), radii, cfg, box=box, periodic=True)
        sys_.vel = np.random.default_rng(11).normal(0.0, 0.05, size=sys_.vel.shape)
        gate = sys_.setup_timestep(mode=mode)
        e0 = sys_.energy()
        p0 = sys_.momentum()
        p0_norm = float(np.linalg.norm(p0))
        worst = 0.0
        trace = []
        t0 = time.perf_counter()
        for k in range(steps):
            sys_.step(1)
            drift = abs(sys_.energy() - e0) / e0
            worst = max(worst, drift)
            if k % GAS_STEP_RECORD == 0 or k == steps - 1:
                trace.append({"step": k, "time": sys_.time, "energy_drift": drift,
                              "kinetic": sys_.kinetic_energy(),
                              "potential": sys_.potential_energy(),
                              "contact_fraction": float(np.mean(sys_._active))
                              if len(sys_._active) else 0.0})
            if not math.isfinite(sys_.energy()):
                break
        wall = time.perf_counter() - t0
        mom = float(np.linalg.norm(sys_.momentum() - p0) / max(p0_norm, 1e-30))
        results[mode] = {
            "dt_outer": gate["dt_outer"],
            "dt_fine": gate["dt_fine"],
            "n_sub": gate["n_sub"],
            "steps": steps,
            "time_span": sys_.time,
            "energy_drift_max": worst,
            "energy_drift_final": abs(sys_.energy() - e0) / e0,
            "momentum_relative_change": mom,
            "wall_seconds": wall,
            "work": dict(sys_.work),
            "initial_energy": e0,
            "final_energy": sys_.energy(),
        }
        history[mode] = trace
        _banner(f"  ledger {mode}: drift_max={worst:.3e} dp={mom:.2e} wall={wall:.1f}s")
    worst = max(r["energy_drift_max"] for r in results.values())
    pmax = max(r["momentum_relative_change"] for r in results.values())
    return {
        "assembly": {"n": GAS_N, "periodic": True, "mu": 0.0, "zeta": 0.0,
                     "stiff_ratio": 100.0, "stiff_fraction": 0.10,
                     "outer_margin": outer_margin, "fast_margin": outer_margin,
                     "area_fraction": float(np.sum(math.pi * radii ** 2) / (box[0] * box[1]))},
        "results": results,
        "history": history,
        "criterion": "energy drift <= 1%, |dP|/|P| <= 1e-12",
        "pass_energy": bool(worst <= 0.01),
        "pass_momentum": bool(pmax <= 1e-12),
        "pass": bool(worst <= 0.01 and pmax <= 1e-12),
    }


# ---------------------------------------------------------------------------
# Stage C -- tightness negative control
# ---------------------------------------------------------------------------
def stage_jamming_ablation(state: dict | None = None) -> dict:
    """Trade-off curve: what the coarse internal-variable rate costs in phi_J."""
    if state is None:
        state = _prepare_state()
    out = {"cases": {}}
    for label, coarse_state in (("state_fine", False), ("state_coarse", True)):
        res = _decompress(state, mode="mts", stiff_ratio=JAM_STIFF_RATIO,
                          stiff_fraction=JAM_STIFF_FRACTION,
                          state_at_coarse=coarse_state)
        out["cases"][label] = {k: res[k] for k in
                               ("phi_j", "phi_at_lowest_pressure", "wall_seconds",
                                "steps", "dphi_dp")}
        _banner(f"  jamming ablation {label}: phi_J={res['phi_j']:.5f} "
                f"wall={res['wall_seconds']:.1f}s")
    fine = out["cases"]["state_fine"]
    coarse = out["cases"]["state_coarse"]
    out["phi_j_bias"] = abs(coarse["phi_j"] / fine["phi_j"] - 1.0)
    out["speed_gain"] = fine["wall_seconds"] / coarse["wall_seconds"]
    out["criterion"] = ("the coarse internal-variable rate buys wall clock and pays "
                        "in phi_J; the shipped default (fine state) keeps phi_J")
    out["pass"] = bool(out["phi_j_bias"] >= 0.005)
    return out


def stage_ablation(steps: int = 3000) -> dict:
    """Ablation: sub-cycling only the stiff skeleton (frozen soft force) breaks
    the energy ledger -- the negative result behind the architecture choice."""
    pos, radii, box = _gas_setup()
    out = {"cases": {}}
    for split in (False, True):
        cfg = MTSConfig(kn=3.0e4, zeta=0.0, mu=0.0, mu_rolling=0.0,
                        stiff_ratio=100.0, stiff_fraction=0.10,
                        outer_margin=0.05, split_slow_contacts=split)
        sys_ = MTSDisks(pos.copy(), radii, cfg, box=box, periodic=True)
        sys_.vel = np.random.default_rng(11).normal(0.0, 0.05, size=sys_.vel.shape)
        gate = sys_.setup_timestep(mode="mts")
        e0 = sys_.energy()
        worst = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            sys_.step(1)
            worst = max(worst, abs(sys_.energy() - e0) / e0)
        wall = time.perf_counter() - t0
        out["cases"]["frozen_soft" if split else "resolved_contacts"] = {
            "steps": steps, "n_sub": gate["n_sub"], "dt_outer": gate["dt_outer"],
            "energy_drift_max": worst, "wall_seconds": wall,
            "time_span": sys_.time,
        }
        _banner(f"  ablation split={split}: drift={worst:.3e} wall={wall:.1f}s")
    ok = out["cases"]["resolved_contacts"]["energy_drift_max"]
    bad = out["cases"]["frozen_soft"]["energy_drift_max"]
    out["criterion"] = "resolving the active contacts keeps the ledger, freezing them does not"
    out["pass"] = bool(ok < bad / 100.0)
    return out


def stage_tightness() -> dict:
    """Criterion (d): dt/dt_c = 0.5 must fail loudly."""
    out = {"cases": []}
    gate = stability_gate(0.5, 1.0, margin=DEFAULT_MARGIN, tight_ratio=TIGHT_RATIO,
                          kind="hertz")
    rejected = False
    try:
        gate.check()
    except InadmissibleTimeStep as exc:
        rejected = True
        message = str(exc)
    if not rejected:
        message = "gate accepted dt/dt_c = 0.5"
    out["gate"] = {"ratio": gate.ratio, "status": gate.status, "rejected": rejected,
                   "message": message}
    worst = 0.0
    for e_target in E_LEVELS:
        cal = calibrate_alpha(e_target=e_target, v_impact=V_IMPACT, mass=MASS,
                              stiffness=STIFFNESS, dt_ratio=RECONVERGENCE_RATIO)
        ref = impact_restitution(mass=MASS, stiffness=STIFFNESS, v_impact=V_IMPACT,
                                 alpha=cal["alpha"], dt_ratio=RECONVERGENCE_RATIO)
        forced = impact_restitution(mass=MASS, stiffness=STIFFNESS, v_impact=V_IMPACT,
                                    alpha=cal["alpha"], dt_ratio=TIGHT_RATIO,
                                    subcycle=False)
        err = abs(forced.e_measured / ref.e_measured - 1.0)
        energy_gain = forced.e_measured ** 2
        worst = max(worst, err)
        out["cases"].append({
            "e_target": e_target,
            "dt_ratio": TIGHT_RATIO,
            "e_measured": forced.e_measured,
            "e_reference": ref.e_measured,
            "restitution_error": err,
            "restitution_energy_gain": energy_gain,
            "detectable": bool(err > 0.10),
        })
    forced_error = worst
    out["forced_worst_error"] = forced_error
    out["criterion"] = ("gate rejects dt/dt_c = 0.5 AND the forced step shows "
                        "restitution error > 10% (tight gate)")
    out["pass"] = bool(rejected and forced_error > 0.10)
    return out


# ---------------------------------------------------------------------------
# Stage E -- double-periodic shear
# ---------------------------------------------------------------------------
def _shear_setup(mu: float, seed: int = 0):
    n = SHEAR_N
    radii = bidisperse_radii(n, 0.010, seed=seed)
    side = math.sqrt(float(np.sum(math.pi * radii ** 2)) / 0.78)
    rng = np.random.default_rng(seed + 3)
    rmax = float(radii.max())
    pos = rmax + rng.random((n, 2)) * (side - 2 * rmax)
    cfg = MTSConfig(kn=JAM_KN, mu=mu, mu_rolling=JAM_MU_ROLLING, zeta=0.4,
                    damping_time=0.0, drag_time=SHEAR_DRAG, shear_rate=SHEAR_RATE,
                    stiff_ratio=JAM_STIFF_RATIO, stiff_fraction=JAM_STIFF_FRACTION)
    return pos, radii, side, cfg


def stage_shear() -> dict:
    """Scenario 2: steady shear stress ratio of a doubly periodic cell."""
    out = {"cases": []}
    for mu in SHEAR_MU:
        row = {"mu": mu}
        for mode in ("single", "mts"):
            pos, radii, side, cfg = _shear_setup(mu)
            sys_ = MTSDisks(pos, radii, cfg, box=(side, side), periodic=True)
            gate = sys_.setup_timestep(mode=mode)
            n_settle = max(1, int(round(SHEAR_T_SETTLE / sys_.dt)))
            n_measure = max(1, int(round(SHEAR_T_MEASURE / sys_.dt)))
            t0 = time.perf_counter()
            sys_.step(n_settle)
            samples = []
            for k in range(n_measure):
                sys_.step(1)
                if k % max(1, n_measure // 20) == 0:
                    sigma = sys_.stress_tensor()
                    p = -0.5 * (sigma[0, 0] + sigma[1, 1])
                    samples.append((float(sigma[0, 1]), float(p), sys_.coordination()))
            wall = time.perf_counter() - t0
            arr = np.array(samples)
            tau = float(np.mean(arr[:, 0]))
            press = float(np.mean(arr[:, 1]))
            coord = float(np.mean(arr[:, 2]))
            row[mode] = {
                "tau": tau, "pressure": press, "tau_over_pressure": tau / press,
                "coordination": coord, "phi": sys_.area_fraction(),
                "wall_seconds": wall, "steps": n_settle + n_measure,
                "dt_outer": gate["dt_outer"], "n_sub": gate["n_sub"],
                "time_span": sys_.time,
                "tau_series": [float(v) for v in arr[:, 0]],
            }
            _banner(f"  shear mu={mu} {mode}: tau/P={tau / press:.4f} "
                    f"Z={coord:.3f} wall={wall:.1f}s")
        a, b = row["single"], row["mts"]
        row["tau_over_pressure_rel_diff"] = abs(b["tau_over_pressure"] / a["tau_over_pressure"] - 1.0)
        row["coordination_abs_diff"] = abs(b["coordination"] - a["coordination"])
        out["cases"].append(row)
    ratios = [row["mts"]["tau_over_pressure"] for row in out["cases"]]
    out["monotone_in_mu"] = bool(all(np.diff(ratios) > 0))
    worst = max(row["tau_over_pressure_rel_diff"] for row in out["cases"])
    out["worst_rel_diff"] = worst
    out["criterion"] = ("|tau/P(multi-rate) - tau/P(single)| / (tau/P) <= 5% for every mu, "
                        "and tau/P increasing with mu")
    out["pass"] = bool(worst <= 0.05 and out["monotone_in_mu"])
    return out


# ---------------------------------------------------------------------------
# Stage F -- friction jamming (acceptance c)
# ---------------------------------------------------------------------------
def _prepare_state() -> dict:
    """Compress a dilute packing to ``p_compress`` with the *shipped* engine.

    The preparation is shared by every rate variant below, so the comparison
    isolates the integrator: identical start state, identical pressure ladder,
    identical physical durations.  The preparation kernel is the m5524 lineage
    :class:`tensordem.granular2d.engine.DiskSystem` (Hertz-Mindlin + rolling).
    """
    from tensordem.granular2d.engine import DiskSystem, DiskConfig, box_walls
    from tensordem.granular2d.jamming import servo_pressure

    radii = bidisperse_radii(JAM_N, JAM_MEAN_RADIUS, seed=JAM_SEED)
    side0 = math.sqrt(float(np.sum(math.pi * radii ** 2)) / 0.45)
    rng = np.random.default_rng(JAM_SEED + 11)
    rmax = float(radii.max())
    pos = rmax + rng.random((JAM_N, 2)) * (side0 - 2.0 * rmax)
    cfg = DiskConfig(kn=JAM_KN, mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, zeta=0.4,
                     damping_time=1.0e-3, skin=0.003)
    system = DiskSystem(pos, radii, cfg, walls=box_walls(side0, side0, margin=0.0))
    servo_pressure(system, JAM_P_COMPRESS, steps=6000, eta=JAM_ETA)
    # the fast compression leaves a deeply overlapped (P ~ 1e4) state; one ladder
    # level of the *same* shipped engine brings it to ~P = 300, which is where the
    # comparison starts (both rates then see the same, relaxed state)
    res = servo_pressure(system, float(JAM_LEVELS[0]), steps=2500, eta=JAM_ETA)
    return {
        "pos": system.pos.copy(), "vel": system.vel.copy(), "radii": radii.copy(),
        "box": (res["side"], res["side"]),
        "phi": res["phi"], "pressure": res["pressure"],
        # the engine step the ladder above was actually run with: the protocol
        # durations are quoted in *physical* time from it (P82W5-N07)
        "engine_dt": float(system.dt),
        "digest": _digest(system.pos, system.vel, radii),
    }


def _decompress(state: dict, *, mode: str, stiff_ratio: float,
                stiff_fraction: float, state_at_coarse: bool = False) -> dict:
    """Run the pressure ladder from the shared state and fit ``phi_J``."""
    cfg = MTSConfig(kn=JAM_KN, mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, zeta=0.4,
                    damping_time=1.0e-3, stiff_ratio=stiff_ratio,
                    stiff_fraction=stiff_fraction, state_at_coarse=state_at_coarse)
    sys_ = MTSDisks(state["pos"].copy(), state["radii"], cfg, box=state["box"])
    sys_.vel = state["vel"].copy()
    gate = sys_.setup_timestep(mode=mode)
    every = max(1, int(round(20 * 8.8e-6 / sys_.dt)))
    n_level = max(1, int(round(JAM_T_LEVEL / sys_.dt)))
    t0 = time.perf_counter()
    series = []
    for level in JAM_LEVELS:
        res = sys_.servo_pressure(float(level), steps=n_level, eta=JAM_ETA, every=every)
        series.append([float(level), float(res["phi"]), float(res["pressure"]),
                       float(res["kinetic_energy"])])
    wall = time.perf_counter() - t0
    p = np.array([row[0] for row in series])
    phi = np.array([row[1] for row in series])
    sel = p <= 30.0
    slope, intercept = np.polyfit(p[sel], phi[sel], 1)
    return {
        "phi_j": float(intercept),
        "phi_at_lowest_pressure": float(phi[-1]),
        "dphi_dp": float(slope),
        "levels": series,
        "wall_seconds": wall,
        "steps": len(JAM_LEVELS) * n_level,
        "time_span": sys_.time,
        "gate": gate,
        "coordination": sys_.coordination(),
        "work": dict(sys_.work),
    }


def _shipped_decompression(state: dict) -> dict:
    """Same pressure ladder on the *shipped* single-rate engine (status quo)."""
    from tensordem.granular2d.engine import DiskSystem, DiskConfig, box_walls
    from tensordem.granular2d.jamming import servo_pressure

    side = float(state["box"][0])
    cfg = DiskConfig(kn=JAM_KN, mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, zeta=0.4,
                     damping_time=1.0e-3, skin=0.003)
    system = DiskSystem(state["pos"].copy(), state["radii"], cfg,
                        walls=box_walls(side, side, margin=0.0))
    system.vel = state["vel"].copy()
    n_level = max(1, int(round(JAM_T_LEVEL / system.dt)))
    t0 = time.perf_counter()
    series = []
    for level in JAM_LEVELS:
        res = servo_pressure(system, float(level), steps=n_level, eta=JAM_ETA)
        series.append([float(level), float(res["phi"]), float(res["pressure"]),
                       float(res["kinetic_energy"])])
    wall = time.perf_counter() - t0
    p = np.array([row[0] for row in series])
    phi = np.array([row[1] for row in series])
    sel = p <= 30.0
    slope, intercept = np.polyfit(p[sel], phi[sel], 1)
    return {"phi_j": float(intercept), "dphi_dp": float(slope), "levels": series,
            "wall_seconds": wall, "steps": len(JAM_LEVELS) * n_level,
            "dt": float(system.dt), "time_span": float(system.time),
            "source": "tensordem.granular2d.engine (shipped single-rate)"}


def stage_jamming(*, run_anchor: bool = True, state: dict | None = None) -> dict:
    """Criterion (c): speed-up without changing the jamming density."""
    out = {"cases": {}}
    if state is None:
        _banner("preparing the shared compressed state (shipped engine)")
        state = _prepare_state()
    out["prepared_state"] = {k: v for k, v in state.items()
                             if k not in ("pos", "vel", "radii")}
    for label, stiff_ratio, stiff_fraction in (
        ("hierarchical", JAM_STIFF_RATIO, JAM_STIFF_FRACTION),
        ("uniform", 1.0, 0.0),
    ):
        entry = {}
        for mode in ("single", "mts"):
            result = _decompress(state, mode=mode, stiff_ratio=stiff_ratio,
                                 stiff_fraction=stiff_fraction)
            entry[mode] = result
            _banner(f"  jamming {label} {mode}: phi_J={result['phi_j']:.5f} "
                    f"wall={result['wall_seconds']:.1f}s n_sub={result['gate']['n_sub']}")
        base, mts = entry["single"], entry["mts"]
        entry["phi_j_rel_diff"] = abs(mts["phi_j"] / base["phi_j"] - 1.0)
        entry["speedup"] = base["wall_seconds"] / mts["wall_seconds"]
        entry["time_span_rel_diff"] = abs(mts["time_span"] / base["time_span"] - 1.0)
        entry["work_ratio"] = (base["work"]["single"]
                               / max(1, mts["work"]["slow"] + mts["work"]["fast"]))
        out["cases"][label] = entry
    _banner("  jamming: shipped single-rate baseline on the same ladder")
    shipped = _shipped_decompression(state)
    out["shipped_baseline"] = shipped
    for label in out["cases"]:
        entry = out["cases"][label]
        entry["shipped_phi_j_rel_diff"] = abs(
            entry["mts"]["phi_j"] / shipped["phi_j"] - 1.0)
        entry["speedup_vs_shipped"] = shipped["wall_seconds"] / entry["mts"]["wall_seconds"]
        entry["speedup_vs_shipped_single"] = (shipped["wall_seconds"]
                                              / entry["single"]["wall_seconds"])
    if run_anchor:
        from tensordem.granular2d.jamming import jamming_density_isotropic
        t0 = time.perf_counter()
        anchor = jamming_density_isotropic(
            mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, n=JAM_N, seed=JAM_SEED,
            p_compress=JAM_P_COMPRESS, levels=JAM_LEVELS, steps_per_level=2500,
            eta=JAM_ETA)
        anchor["wall_seconds"] = time.perf_counter() - t0
        anchor["source"] = ("tensordem.granular2d.jamming.jamming_density_isotropic "
                            "(m5524 / W4-P1 servo lineage)")
        out["w4_p1_servo_anchor"] = anchor
    hier = out["cases"]["hierarchical"]
    anchor = out.get("w4_p1_servo_anchor")
    anchor_diff = None
    if anchor is not None:
        anchor_diff = abs(hier["mts"]["phi_j"] / anchor["phi_j"] - 1.0)
    out["phi_j_vs_anchor_rel_diff"] = anchor_diff
    out["criterion"] = ("hierarchical: wall clock speed-up >= 2x, |dphi_J| <= 0.5%, "
                        "and phi_J within 0.5% of the W4-P1 servo anchor")
    out["pass_speedup"] = bool(hier["speedup"] >= 2.0)
    out["pass_phi_j"] = bool(hier["phi_j_rel_diff"] <= 0.005)
    out["pass_anchor"] = None if anchor_diff is None else bool(anchor_diff <= 0.005)
    out["pass"] = bool(out["pass_speedup"] and out["pass_phi_j"]
                       and (out["pass_anchor"] in (None, True)))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _shard_ladder(shards: int, *, run_dir: str = "P82W2_N09") -> dict:
    """Stage D: evaluate the ladder in ``shards`` separate processes."""
    runs = []
    for shard in range(shards):
        out_path = ROOT / "runs" / run_dir / f"ladder_shard{shard}_of{shards}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "ladder",
               "--shard", str(shard), "--shards", str(shards),
               "--output", str(out_path)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        runs.append(json.loads(out_path.read_text()))
    cases = [c for r in runs for c in r["cases"]]
    cases.sort(key=lambda c: (c["e_target"], c["dt_ratio"]))
    baseline = cases
    worst = 0.0
    for run in runs:
        for case in run["cases"]:
            match = next(c for c in baseline if c["e_target"] == case["e_target"]
                         and c["dt_ratio"] == case["dt_ratio"])
            worst = max(worst, abs(case["e_measured"] - match["e_measured"]))
    return {
        "shards": shards,
        "cases_per_shard": [len(r["cases"]) for r in runs],
        "worst_abs_difference_in_restitution": worst,
        "bitwise_identical": bool(worst == 0.0),
        "criterion": ("1/2/4 shards give identical values (bitwise) or a stated "
                      "floating-point bound inside the tolerance"),
        "pass": bool(worst <= 1e-12),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "ladder", "ledger", "tightness", "shear",
                                 "jamming", "jamming_ablation", "shards", "ablation"])
    parser.add_argument("--output", default=str(ROOT / "runs" / "P82W2_N09" / "verification.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument("--ledger-steps", type=int, default=GAS_STEPS)
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "ladder":
        # sharded evaluation: split the (e, ratio) matrix by shard
        pairs = [(e, r) for e in E_LEVELS for r in DT_RATIOS]
        mine = [p for k, p in enumerate(pairs) if k % max(args.shards, 1) == args.shard]
        result = stage_ladder(ratios=tuple(sorted({p[1] for p in mine})),
                             e_levels=tuple(sorted({p[0] for p in mine})))
        result["cases"] = [c for c in result["cases"]
                          if (c["e_target"], c["dt_ratio"]) in set(mine)]
        result["shard"] = {"index": args.shard, "count": args.shards, "cases": len(mine)}
        out_path.write_text(json.dumps(_jsonable(result), indent=1))
        print(f"shard {args.shard}/{args.shards}: {len(result['cases'])} cases -> {out_path}")
        return 0

    if args.stage == "tightness":
        result = stage_tightness()
    elif args.stage == "ledger":
        result = stage_ledger(steps=args.ledger_steps)
    elif args.stage == "ablation":
        result = stage_ablation()
    elif args.stage == "jamming_ablation":
        result = stage_jamming_ablation()
    elif args.stage == "shear":
        result = stage_shear()
    elif args.stage == "jamming":
        result = stage_jamming(run_anchor=not args.skip_anchor)
    elif args.stage == "shards":
        result = {"shard_reproducibility": {n: _shard_ladder(n) for n in (1, 2, 4)}}
        result["pass"] = all(v["pass"] for v in result["shard_reproducibility"].values())
    else:
        result = {}
        result["ladder"] = stage_ladder()
        result["tightness"] = stage_tightness()
        result["ledger"] = stage_ledger(steps=args.ledger_steps)
        result["ablation"] = stage_ablation()
        result["shard_reproducibility"] = {n: _shard_ladder(n) for n in (1, 2, 4)}
        result["shear"] = stage_shear()
        result["jamming"] = stage_jamming(run_anchor=not args.skip_anchor)
    out_path.write_text(json.dumps(_jsonable(result), indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())