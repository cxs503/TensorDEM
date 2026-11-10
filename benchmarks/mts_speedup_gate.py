"""P82W5-N07 -- the multi-rate contact-integration *acceptance gate*.

This is the maturation runner of the MTS node.  It does not re-derive the
scheme (that is :mod:`tensordem.mts`, shipped by P82W2-N09); it turns the node
into a machine-decidable acceptance gate and closes the criterion that the
first pass left open -- the **cost** side of the contract:

    sub-cycling the stiff skeleton must recover a factor >= 2 in wall clock on
    the frictional jamming state (``mu = 0.5``, ``N = 256``) **while the jamming
    density stays put** (``|d phi_J| <= 0.5 %`` versus the single-rate solver of
    the same engine *and* versus the ``W4-P1`` servo anchor).

Two engineering changes make that possible, and both are measured here against
the pre-change behaviour (``wall_rate="fast"`` + ``fast_kernel="reference"`` is
the ablation that reproduces it):

1. **Rate assignment of the walls.**  A wall spring is ``kn`` -- the *soft*
   class.  The old code re-evaluated the walls in every sub-step (``n_sub``
   times the wall work, for no accuracy); they now live on the slow arm.
2. **The fused fast-arm kernel.**  The pair-static quantities (stiffness,
   effective mass, dashpot coefficient, tangential/rolling moduli, lever radii)
   are hoisted out of the sub-step loop (``_prepare_fast``), and the
   geometry/kinematics/friction passes are fused into one sweep
   (``_fast_step``).  The contact laws are unchanged: the regression test
   ``tests/test_mts_speedup_gate.py`` shows the fused kernel reproduces the
   reference composition to round-off (``< 1e-12`` relative on force, torque and
   the internal variables).

Acceptance criteria produced by this file (all numbers come from the library,
nothing is hand-copied):

* **a** restitution ladder: ``e = 0.5 / 0.9``, ``dt/dt_c in {0.02, 0.05, 0.1,
  0.2}``, error ``<= 2 %``;
* **b** energy / momentum ledger: ``1e5`` coarse steps, undamped frictionless
  doubly periodic gas, energy drift ``<= 1 %`` and ``|dP|/|P| <= 1e-12``;
* **c** cost and physics on the same state: wall-clock speed-up ``>= 2x`` at
  ``mu = 0.5``, ``N = 256``; ``phi_J`` within ``0.5 %`` of the single-rate arm
  and of the ``W4-P1`` servo anchor (protocol-parity run);
* **d** tightness negative control: ``dt/dt_c = 0.5`` is inadmissible and, when
  forced, shows a restitution error ``> 10 %``;
* **e** shard reproducibility: the ladder matrix in 1 / 2 / 4 processes agrees.

Usage::

    python benchmarks/mts_speedup_gate.py --stage all \\
        --output artifacts/P82W5_N07/acceptance.json

Stages are individually addressable (``--stage ladder|ledger|tightness|shards|
speed|jamming|anchor_parity``) so the expensive ones can be re-run alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import mts_contact_integration as BASE  # noqa: E402  (the P82W2-N09 runner)
from tensordem.mts import MTSConfig, MTSDisks  # noqa: E402

MISSION = "P82W5_N07"

#: geometry / material of the cost experiment -- identical to the P82W2-N09
#: jamming state so the two nodes are comparable number for number
JAM_N = BASE.JAM_N
JAM_MU = BASE.JAM_MU
JAM_MU_ROLLING = BASE.JAM_MU_ROLLING
JAM_KN = BASE.JAM_KN
JAM_SEED = BASE.JAM_SEED
JAM_MEAN_RADIUS = BASE.JAM_MEAN_RADIUS
JAM_P_COMPRESS = BASE.JAM_P_COMPRESS
JAM_LEVELS = BASE.JAM_LEVELS
JAM_ETA = BASE.JAM_ETA
JAM_STIFF_RATIO = BASE.JAM_STIFF_RATIO
JAM_STIFF_FRACTION = BASE.JAM_STIFF_FRACTION
JAM_STEPS_PER_LEVEL = 2500      # the W4-P1 servo protocol
JAM_SERVO_EVERY = 20            # the W4-P1 servo refresh interval (engine steps)
JAM_MAX_STEP = 3.0e-4           # the W4-P1 servo box-scaling clip per engine step

#: the four rate-assignment / kernel variants of the ablation
VARIANTS = (
    ("legacy", {"wall_rate": "fast", "fast_kernel": "reference"}),
    ("walls_fast_lean", {"wall_rate": "fast", "fast_kernel": "lean"}),
    ("walls_slow_reference", {"wall_rate": "slow", "fast_kernel": "reference"}),
    ("matured", {"wall_rate": "slow", "fast_kernel": "lean"}),
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _banner(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ---------------------------------------------------------------------------
# The shared state of the cost experiment: the W4-P1 servo protocol
# ---------------------------------------------------------------------------
def _live_box(system) -> float:
    """Side length of the walled box as it stands *now*.

    ``isotropic_packing_at_pressure`` returns the *initial* (``phi = 0.45``) side,
    not the compressed one, so the box has to be read back from the walls.
    """
    pts = system.walls.reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return float(hi[0] - lo[0])


def anchor_state(*, conditioned: bool = True) -> dict:
    """The state the ``W4-P1`` servo anchor is built from.

    ``isotropic_packing_at_pressure`` is the shipped preparation step of
    :func:`tensordem.granular2d.jamming.jamming_density_isotropic`, so taking it
    verbatim makes the ``phi_J`` comparison a *protocol-parity* one.

    With ``conditioned=True`` the state is first driven through the first rung of
    the ladder (``P = 3e3 -> 300``) **on the shipped single-rate engine**, which
    is exactly the anchor's own endpoint of that rung.  Both integrators are then
    branched from one and the same state, so what is left to differ is the
    integrator only.  This matters: at ``P = 3e3`` the packing carries ``phi =
    0.98`` (heavy overlaps) and the first rung is a violent decompression -- a
    *non-equilibrium* transient during which two integrators seed slightly
    different packings and a marginally jammed state keeps the memory
    (``c3_transient_sensitivity`` in the output measures that effect and is
    reported, not hidden).
    """
    from tensordem.granular2d.jamming import (isotropic_packing_at_pressure,
                                              servo_pressure)

    system, _side0 = isotropic_packing_at_pressure(
        mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, n=JAM_N,
        mean_radius=JAM_MEAN_RADIUS, seed=JAM_SEED, p_compress=JAM_P_COMPRESS)
    levels = JAM_LEVELS
    if conditioned:
        servo_pressure(system, float(JAM_LEVELS[0]), steps=JAM_STEPS_PER_LEVEL,
                       eta=JAM_ETA)
        levels = JAM_LEVELS[1:]
    side = _live_box(system)
    return {
        "pos": system.pos.copy(),
        "vel": system.vel.copy(),
        "radii": np.asarray(system.radii, dtype=np.float64).copy(),
        "box": (side, side),
        "engine_dt": float(system.dt),
        "phi_prepared": float(np.sum(math.pi * np.asarray(system.radii) ** 2) / side ** 2),
        "levels": tuple(float(v) for v in levels),
        "conditioned": bool(conditioned),
    }


def _mts_system(state: dict, mode: str, variant: str, *,
                stiff_ratio: float = JAM_STIFF_RATIO,
                stiff_fraction: float = JAM_STIFF_FRACTION) -> tuple[MTSDisks, dict]:
    opts = dict(VARIANTS)[variant]
    cfg = MTSConfig(kn=JAM_KN, mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, zeta=0.4,
                    damping_time=1.0e-3, stiff_ratio=stiff_ratio,
                    stiff_fraction=stiff_fraction,
                    # the W4-P1 servo controls the disk-disk virial (the shipped
                    # virial_tensor convention); the servo has to see the same
                    # quantity here or the two ladders control different pressures
                    pressure_includes_walls=False, **opts)
    sys_ = MTSDisks(state["pos"].copy(), state["radii"], cfg, box=state["box"])
    sys_.vel = state["vel"].copy()
    gate = sys_.setup_timestep(mode=mode)
    return sys_, gate


def _ladder(sys_: MTSDisks, state: dict, *, levels=None) -> dict:
    """The W4-P1 pressure ladder on ``sys_``, in *physical* durations.

    ``JAM_STEPS_PER_LEVEL`` engine steps of ``state["engine_dt"]`` per level, so
    the single-rate, the multi-rate and the anchor run the same physical
    protocol and only the integrator differs.
    """
    levels = state["levels"] if levels is None else levels
    t_level = JAM_STEPS_PER_LEVEL * state["engine_dt"]
    every = max(1, int(round(JAM_SERVO_EVERY * state["engine_dt"] / sys_.dt)))
    n_level = max(1, int(round(t_level / sys_.dt)))
    # protocol parity: the shipped servo clips the box scaling at
    # max_step = 3e-4 per *engine* step, i.e. at a compression rate of
    # 3e-4 / dt_engine.  Using the bare rate clip would let the coarse arm
    # deform the box up to n_sub times faster -- a protocol difference that has
    # nothing to do with the integrator.
    max_rate = JAM_MAX_STEP / state["engine_dt"]
    series = []
    t0 = time.perf_counter()
    for level in levels:
        res = sys_.servo_pressure(float(level), steps=n_level, eta=JAM_ETA,
                                  every=every, max_rate=max_rate)
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
        "wall_seconds": float(wall),
        "steps": int(n_level * len(levels)),
        "n_level_steps": int(n_level),
        "servo_every": int(every),
        "dt": float(sys_.dt),
        "time_span": float(sys_.time),
        "gate": {"n_sub": int(sys_.n_sub), "dt_outer": float(sys_.dt),
                 "dt_fine": float(sys_.dt / sys_.n_sub)},
        "work": dict(sys_.work),
        "coordination": float(sys_.coordination()),
    }


# ---------------------------------------------------------------------------
# Stage: cost (criterion c, the wall-clock half)
# ---------------------------------------------------------------------------
def stage_speed(*, reps: int = 3, span_steps: int = 4000) -> dict:
    """Wall clock of the same physical span, single-rate versus multi-rate.

    Both arms advance the *same* physical time on the *same* prepared state, so
    the ratio is a like-for-like speed-up and not a comparison of different
    amounts of physics.  The measurement is interleaved (single, mts, single,
    mts, ...) and the median is reported, so a cold cache cannot masquerade as a
    speed-up.
    """
    state = anchor_state()
    out = {"state": {k: state[k] for k in ("box", "engine_dt")}, "variants": {},
           "reps": int(reps), "span_steps_single": int(span_steps)}
    out["state"]["n"] = int(len(state["radii"]))
    out["state"]["phi_prepared"] = float(state["phi_prepared"])
    for variant, _opts in VARIANTS:
        single, g_single = _mts_system(state, "single", variant)
        mts, g_mts = _mts_system(state, "mts", variant)
        n_sub = int(g_mts["n_sub"])
        n_single = int(span_steps)
        n_mts = max(1, int(round(n_single * single.dt / mts.dt)))
        span_single = n_single * single.dt
        span_mts = n_mts * mts.dt
        # warm-up (neighbour lists, first-touch allocations)
        single.step(min(50, n_single))
        mts.step(min(50, n_mts))
        t_single, t_mts = [], []
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            single.step(n_single)
            t_single.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            mts.step(n_mts)
            t_mts.append(time.perf_counter() - t0)
        ws = float(np.median(t_single))
        wm = float(np.median(t_mts))
        out["variants"][variant] = {
            "n_sub": n_sub,
            "dt_single": float(single.dt),
            "dt_mts": float(mts.dt),
            "steps_single": n_single,
            "steps_mts": n_mts,
            "span_single": float(span_single),
            "span_mts": float(span_mts),
            "span_rel_diff": float(abs(span_mts / span_single - 1.0)),
            "wall_single": ws,
            "wall_mts": wm,
            "wall_single_all": t_single,
            "wall_mts_all": t_mts,
            "ms_per_step_single": 1e3 * ws / n_single,
            "ms_per_coarse_step_mts": 1e3 * wm / n_mts,
            "cost_ratio_per_coarse_step": (wm / n_mts) / (ws / n_single),
            "speedup": ws / wm,
            "work_single": dict(single.work),
            "work_mts": dict(mts.work),
            "same_physics": {
                "phi_single": float(single.area_fraction()),
                "phi_mts": float(mts.area_fraction()),
                "pressure_single": float(single.pressure()),
                "pressure_mts": float(mts.pressure()),
            },
        }
        row = out["variants"][variant]
        _banner(f"  speed {variant:22s} n_sub={n_sub:2d} single {ws:6.2f}s "
                f"mts {wm:6.2f}s -> {row['speedup']:5.2f}x "
                f"({row['ms_per_coarse_step_mts']:.2f} ms/coarse step)")
    matured = out["variants"]["matured"]
    legacy = out["variants"]["legacy"]
    out["speedup"] = matured["speedup"]
    out["speedup_legacy"] = legacy["speedup"]
    out["gain_vs_legacy"] = matured["speedup"] / legacy["speedup"]
    out["criterion"] = ("wall clock of the same physical span >= 2x faster than the "
                        "single-rate baseline at mu=0.5, N=256, stiffness 100:1 on 10%")
    out["pass"] = bool(out["speedup"] >= 2.0)
    return out


# ---------------------------------------------------------------------------
# Stage: jamming density at protocol parity (criterion c, the physics half)
# ---------------------------------------------------------------------------
def _jamming_cases(state: dict, label_prefix: str = "") -> dict:
    """The ladder on one state, run by both integrators, on both hierarchies."""
    cases = {}
    for label, stiff_ratio, stiff_fraction in (
        ("hierarchical", JAM_STIFF_RATIO, JAM_STIFF_FRACTION),
        ("uniform", 1.0, 0.0),
    ):
        entry = {}
        for mode in ("single", "mts"):
            sys_, gate = _mts_system(state, mode, "matured",
                                     stiff_ratio=stiff_ratio,
                                     stiff_fraction=stiff_fraction)
            res = _ladder(sys_, state)
            entry[mode] = res
            _banner(f"  {label_prefix}phi_J {label:12s} {mode:6s}: {res['phi_j']:.5f} "
                    f"n_sub={res['gate']['n_sub']:2d} wall={res['wall_seconds']:.1f}s "
                    f"levels={[round(row[0]) for row in res['levels']]}")
        base, mts = entry["single"], entry["mts"]
        entry["phi_j_rel_diff"] = float(abs(mts["phi_j"] / base["phi_j"] - 1.0))
        entry["speedup"] = float(base["wall_seconds"] / mts["wall_seconds"])
        entry["span_rel_diff"] = float(abs(mts["time_span"] / base["time_span"] - 1.0))
        cases[label] = entry
    return cases


def stage_jamming(*, run_anchor: bool = True) -> dict:
    """``phi_J`` on one protocol, three integrators, plus the W4-P1 anchor."""
    state = anchor_state()
    out = {"state": {k: state[k] for k in ("box", "engine_dt")},
           "protocol": {"full_levels": list(JAM_LEVELS),
                        "branch_levels": list(state["levels"]),
                        "branch_pressure": float(JAM_LEVELS[0]),
                        "conditioning": ("one W4-P1 rung (P=3e3 -> 300) on the "
                                         "shipped single-rate engine; both "
                                         "integrators branch from that state"),
                        "eta": JAM_ETA,
                        "servo_max_step": JAM_MAX_STEP,
                        "steps_per_level": JAM_STEPS_PER_LEVEL,
                        "servo_every_engine_steps": JAM_SERVO_EVERY},
           "cases": {}}
    out["state"]["n"] = int(len(state["radii"]))
    out["state"]["phi_at_branch"] = float(state["phi_prepared"])
    out["cases"] = _jamming_cases(state)
    if run_anchor:
        from tensordem.granular2d.jamming import jamming_density_isotropic

        t0 = time.perf_counter()
        anchor = jamming_density_isotropic(
            mu=JAM_MU, mu_rolling=JAM_MU_ROLLING, n=JAM_N, seed=JAM_SEED,
            p_compress=JAM_P_COMPRESS, levels=JAM_LEVELS,
            steps_per_level=JAM_STEPS_PER_LEVEL, eta=JAM_ETA)
        anchor["wall_seconds"] = float(time.perf_counter() - t0)
        anchor["source"] = ("tensordem.granular2d.jamming.jamming_density_isotropic "
                            "(m5524 / W4-P1 servo lineage)")
        out["w4_p1_servo_anchor"] = anchor
        for label, entry in out["cases"].items():
            entry["anchor_rel_diff"] = {
                mode: float(abs(entry[mode]["phi_j"] / anchor["phi_j"] - 1.0))
                for mode in ("single", "mts")}
    # history sensitivity: the same comparison without the conditioning rung.
    # A marginally jammed packing remembers how it got there, so this number
    # bounds what the protocol (not the integrator) contributes.
    transient_state = anchor_state(conditioned=False)
    transient = {"state": {"phi_prepared": float(transient_state["phi_prepared"]),
                           "levels": list(transient_state["levels"])},
                 "cases": _jamming_cases(transient_state, label_prefix="transient ")}
    trans_anchor = None
    if "w4_p1_servo_anchor" in out:
        anchor_phi = out["w4_p1_servo_anchor"]["phi_j"]
        for label, entry in transient["cases"].items():
            entry["anchor_rel_diff"] = {
                mode: float(abs(entry[mode]["phi_j"] / anchor_phi - 1.0))
                for mode in ("single", "mts")}
        trans_anchor = transient["cases"]["hierarchical"]["anchor_rel_diff"]["mts"]
    transient["phi_j_vs_anchor_rel_diff"] = trans_anchor
    transient["note"] = ("non-equilibrium branch: the first rung (P=3e3 -> 300) "
                        "is a violent decompression; the two integrators seed "
                        "different packings there and the jammed state keeps the "
                        "memory. Reported as the protocol's history sensitivity.")
    out["c3_transient_sensitivity"] = transient
    hier = out["cases"]["hierarchical"]
    anchor_diff = hier.get("anchor_rel_diff", {}).get("mts")
    out["phi_j_rel_diff"] = hier["phi_j_rel_diff"]
    out["phi_j_vs_anchor_rel_diff"] = anchor_diff
    out["predicted_cost_ratio"] = {
        "model": "n_sub / (1 + n_sub * f) with f = stiff contact-work fraction",
        "n_sub": hier["mts"]["gate"]["n_sub"],
    }
    out["criterion"] = ("|dphi_J| <= 0.5% multi-rate vs single-rate on the same protocol "
                        "AND vs the W4-P1 servo anchor")
    out["pass_phi_j"] = bool(hier["phi_j_rel_diff"] <= 0.005)
    out["pass_anchor"] = None if anchor_diff is None else bool(anchor_diff <= 0.005)
    out["pass"] = bool(out["pass_phi_j"] and out["pass_anchor"] in (None, True))
    return out


def stage_legacy_protocol() -> dict:
    """Sensitivity: the P82W2-N09 preparation (``eta = 150``) versus the anchor.

    Kept because it is the number the previous node reported.  The residual gap
    to the anchor here is the *preparation* protocol, not the integrator: the
    same single-rate engine on the same state differs from the anchor by the
    same margin (see ``cases.hierarchical.single``).
    """
    state = dict(BASE._prepare_state())
    state.setdefault("levels", JAM_LEVELS)          # the P82W2 ladder verbatim
    out = {"cases": {}, "engine_dt": state["engine_dt"]}
    for mode in ("single", "mts"):
        sys_, _gate = _mts_system(state, mode, "matured")
        out["cases"][mode] = _ladder(sys_, state)
        _banner(f"  legacy-protocol {mode}: phi_J={out['cases'][mode]['phi_j']:.5f} "
                f"wall={out['cases'][mode]['wall_seconds']:.1f}s")
    out["phi_j_rel_diff"] = float(
        abs(out["cases"]["mts"]["phi_j"] / out["cases"]["single"]["phi_j"] - 1.0))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(stage: str, *, ledger_steps: int, run_anchor: bool,
          speed_reps: int) -> dict:
    out: dict = {"mission": MISSION,
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "python": sys.version.split()[0],
                 "numpy": np.__version__}
    if stage in ("all", "a", "ladder"):
        out["a_restitution_ladder"] = BASE.stage_ladder()
    if stage in ("all", "b", "ledger"):
        out["b_energy_ledger"] = BASE.stage_ledger(steps=ledger_steps)
    if stage in ("all", "d", "tightness"):
        out["d_tightness"] = BASE.stage_tightness()
    if stage in ("all", "e", "shards"):
        out["e_shards"] = {"shard_reproducibility":
                           {n: BASE._shard_ladder(n, run_dir=MISSION)
                            for n in (1, 2, 4)}}
        out["e_shards"]["pass"] = all(
            v["pass"] for v in out["e_shards"]["shard_reproducibility"].values())
    if stage in ("all", "speed"):
        out["c1_speed"] = stage_speed(reps=speed_reps)
    if stage in ("all", "jamming", "anchor_parity"):
        out["c2_jamming"] = stage_jamming(run_anchor=run_anchor)
        out["c2b_legacy_protocol"] = stage_legacy_protocol()
    keys = {"a": "a_restitution_ladder", "b": "b_energy_ledger",
            "c_speed": "c1_speed", "d": "d_tightness", "e": "e_shards"}
    verdict = {}
    for name, key in keys.items():
        block = out.get(key)
        if block is None:
            continue
        verdict[name] = bool(block.get("pass"))
    if "c2_jamming" in out:
        jam = out["c2_jamming"]
        verdict["c_phi_j_vs_single"] = bool(jam.get("pass_phi_j"))
        verdict["c_phi_j_vs_anchor"] = bool(jam.get("pass_anchor"))
        out["honest_fail"] = [k for k, v in verdict.items() if v is False]
    out["verdict"] = verdict
    core = ("a", "b", "c_speed", "c_phi_j_vs_single", "d", "e")
    out["pass_core"] = all(verdict[k] for k in core if k in verdict)
    out["pass_all"] = bool(out["pass_core"] and verdict.get("c_phi_j_vs_anchor", True))
    out["pass"] = out["pass_all"]
    return _jsonable(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "a", "b", "c", "d", "e", "ladder", "ledger",
                                 "speed", "jamming", "anchor_parity", "tightness",
                                 "shards"])
    parser.add_argument("--output",
                        default=str(ROOT / "artifacts" / MISSION / "acceptance.json"))
    parser.add_argument("--ledger-steps", type=int, default=BASE.GAS_STEPS)
    parser.add_argument("--speed-reps", type=int, default=3)
    parser.add_argument("--skip-anchor", action="store_true")
    args = parser.parse_args(argv)
    stage = {"a": "ladder", "b": "ledger", "c": "jamming", "d": "tightness",
             "e": "shards"}.get(args.stage, args.stage)
    result = build(stage, ledger_steps=args.ledger_steps,
                   run_anchor=not args.skip_anchor, speed_reps=args.speed_reps)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    _banner(f"wrote {out_path}")
    if "verdict" in result:
        print(json.dumps(result["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())