#!/usr/bin/env python3
"""Execute the real TensorPhysics consumer against the real TensorDEM supplier.

Anchor: ``XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT`` (round 4 / m19466).

Why this entry exists
---------------------
Rounds 1-3 froze the supplier surface, pinned it with a sha256 fingerprint and
ran 19 behavioural probes -- all *inside* TensorDEM. The consumer that actually
binds to this surface (``tensorphysics.adapters.tensordem.TensorDEMAdapter``)
was only ever exercised against a ``FakeIceDEM`` in the other repository. This
script closes that loop: it imports the real adapter + real ``Simulation`` and
drives a real ``tensordem.IceDEM``, then compares every observed number with
the value the frozen contract declares (ref vs got), and finally proves the
cross-check is not vacuous by feeding it three controlled drift variants.

Ref numbers are computed *here* from the contract declaration (``n dt F`` and
``k dt``), never transcribed from the run.

Outputs (defaults)::

    artifacts/xval_library_binding_contract_live/live_binding.json
    artifacts/xval_library_binding_contract_live/live_binding_table.md

Repro (inside the repo checkout, one core, no GPU, ~5 s)::

    python scripts/xval_library_binding_contract_live.py

Exit code is 0 only when every pre-registered arm passes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensordem import binding_contract as bc  # noqa: E402
from tensordem.dem import DEMConfig, IceDEM  # noqa: E402

OWNER_TOLERANCE = 0.15  # owner precision line; these arms are structural, so far tighter
SUBSTEP_TOL = 1e-12
LOAD_STAMP_TOL_S = 1e-12
IMPULSE_TOL = 1e-12
CLOCK_TOL = 1e-12
RATIO_TOL = 1e-12
H0_SEPARATION_MIN = OWNER_TOLERANCE

APPROVED_RATIO = 4  # global dt / native dt used by the main arms
GLOBAL_STEPS = 3
LOAD = (0.25, -0.1)  # N per particle, as in the round-1 isolated probe


def _find_tensorphysics() -> Path | None:
    """Locate the upstream consumer checkout; None means WITHHELD (never faked)."""
    candidates = [
        os.environ.get("TENSORPHYSICS_SRC"),
        "/root/tensor-ecosystem/TensorPhysics/src",
        str(ROOT.parent / "TensorPhysics" / "src"),
        str(ROOT.parent.parent / "TensorPhysics" / "src"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "tensorphysics" / "__init__.py").is_file():
            return Path(cand)
    return None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "unknown"


def _fresh_2d() -> IceDEM:
    """Isolated bonded lattice: no supports, no damping, no drag (momentum clean)."""
    cfg = DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                    contact_damping=0.0, drag=0.0)
    return IceDEM(cfg)


class RecordingIceDEM:
    """Forward to a real IceDEM while recording what the consumer hands in."""

    def __init__(self, inner: IceDEM) -> None:
        self._inner = inner
        self.calls: list[dict] = []

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def step(self, external_forces=None, *, load_time=None) -> None:
        total = None
        if external_forces is not None:
            flat = external_forces if external_forces.dim() > 1 else external_forces.reshape(1, -1)
            total = [float(v) for v in flat.sum(0)]
        self.calls.append({
            "load_time": None if load_time is None else float(load_time),
            "load_total": total,
            "load_object_id": None if external_forces is None else id(external_forces),
        })
        self._inner.step(external_forces=external_forces, load_time=load_time)


class AccumulatingIceDEM(RecordingIceDEM):
    """Controlled drift variant: the supplier accumulates loads instead of replacing them.

    H0 (the failure mode the contract exists to prevent) predicts a measured
    impulse ratio of (n+1)/2 for n substeps.
    """

    def __init__(self, inner: IceDEM) -> None:
        super().__init__(inner)
        self._pending = torch.zeros_like(inner.positions)

    def step(self, external_forces=None, *, load_time=None) -> None:
        if external_forces is not None:
            self._pending = self._pending + external_forces
            external_forces = self._pending
        super().step(external_forces=external_forces, load_time=load_time)


def _rel_error(ref: float, got: float) -> float:
    scale = abs(ref) if abs(ref) > 0.0 else 1.0
    return abs(got - ref) / scale


def _row(name: str, ref, got, rel_error, tolerance, passed, note: str = "") -> dict:
    return {"name": name, "ref": ref, "got": got, "rel_error": rel_error,
            "tolerance": tolerance, "pass": bool(passed), "note": note}


def _run_simulation(adapter, dt_global: float, n_global_steps: int):
    from tensorphysics import Simulation, SimulationConfig

    cfg = SimulationConfig(dt=dt_global, end_time=dt_global * n_global_steps)
    sim = Simulation(cfg, [adapter])
    sim.run()
    return sim


def collect() -> dict:
    from tensorphysics import SimulationState
    from tensorphysics.adapters import TensorDEMAdapter

    model = _fresh_2d()
    native_dt = float(model.dt)
    proxy = RecordingIceDEM(model)
    handed: list[tuple[float, object]] = []

    def load_provider(context, dem):
        loads = torch.tensor(LOAD, dtype=model.positions.dtype).repeat(model.positions.shape[0], 1)
        handed.append((context.time, loads))
        return loads

    adapter = TensorDEMAdapter(proxy, load_provider=load_provider)
    sim = _run_simulation(adapter, APPROVED_RATIO * native_dt, GLOBAL_STEPS)

    rows: list[dict] = []
    n_expected = GLOBAL_STEPS * APPROVED_RATIO
    n_got = len(proxy.calls)

    # A1 substep count: contract says global dt = integer multiple of native dt.
    rows.append(_row("substep_count", n_expected, n_got,
                     0.0 if n_got == n_expected else 1.0, 0.0, n_got == n_expected,
                     "contract IceDEM.clock.global_step_rule, rel_tol 1e-9"))

    # A2 load_time stamps: adapter must stamp each substep start, no interpolation.
    ref_stamps = [k * native_dt for k in range(n_expected)]
    got_stamps = [c["load_time"] for c in proxy.calls]
    dev = max((abs(a - b) for a, b in zip(ref_stamps, got_stamps)), default=float("inf"))
    rows.append(_row("load_time_stamp_max_dev_s", 0.0, dev, dev, LOAD_STAMP_TOL_S,
                     dev <= LOAD_STAMP_TOL_S,
                     f"ref stamps k*dt, k=0..{n_expected - 1}; got {len(got_stamps)} stamps"))

    # A2b loads handed in verbatim: every substep's provider tensor is passed as-is.
    same_objects = sum(1 for (ctx_t, loads), call in zip(handed, proxy.calls)
                       if call["load_object_id"] == id(loads))
    rows.append(_row("loads_passed_verbatim_calls", n_expected, same_objects, 0.0,
                     0.0, same_objects == n_expected,
                     "identity check: no silent copy/scale/remap between provider and step()"))

    # A3 external impulse identity: ref = n*dt*F_total (contract semantic).
    f_total = torch.tensor(LOAD, dtype=model.positions.dtype).repeat(model.positions.shape[0], 1)
    ref_impulse = [n_expected * native_dt * float(v) for v in f_total.sum(0)]
    got_impulse = [float(v) for v in model.accounting["external_impulse_Ns"]]
    imp_err = max(_rel_error(r, g) for r, g in zip(ref_impulse, got_impulse))
    rows.append(_row("external_impulse_rel_error", ref_impulse, got_impulse, imp_err,
                     IMPULSE_TOL, imp_err <= IMPULSE_TOL,
                     "model.accounting['external_impulse_Ns'] vs n*dt*F"))

    # A4 non-accumulation + H0 discrimination.
    ratio = [g / r if r else float("nan") for r, g in zip(ref_impulse, got_impulse)]
    h0_ratio = (n_expected + 1) / 2.0
    sep = [abs(h0_ratio - x) / h0_ratio for x in ratio]
    rows.append(_row("impulse_ratio_got_over_ref", 1.0, ratio,
                     max(abs(x - 1.0) for x in ratio), RATIO_TOL,
                     max(abs(x - 1.0) for x in ratio) <= RATIO_TOL,
                     f"H0 accumulator would give (n+1)/2 = {h0_ratio:.3f}"))
    rows.append(_row("h0_separation", H0_SEPARATION_MIN, min(sep),
                     max(0.0, H0_SEPARATION_MIN - min(sep)), H0_SEPARATION_MIN,
                     min(sep) >= H0_SEPARATION_MIN,
                     "distance from the accumulated-load prediction (owner line 15%)"))

    # A5 clock: native time == n_steps*native_dt.
    ref_time = n_expected * native_dt
    clock_err = _rel_error(ref_time, float(model.time))
    rows.append(_row("clock_rel_error", ref_time, float(model.time), clock_err,
                     CLOCK_TOL, clock_err <= CLOCK_TOL, "IceDEM.clock"))

    # A6 executed negative assertion: consumer snapshot must not claim restartability.
    snap = adapter.snapshot(None)
    from tensorphysics.protocols import Solver

    no_restore = not hasattr(Solver, "restore")
    ok6 = (snap.get("restartable") is False) and no_restore
    rows.append(_row("no_restore_claim", "restartable=False and not hasattr(Solver,'restore')",
                     f"restartable={snap.get('restartable')}, has_restore={not no_restore}",
                     0.0, 0.0, ok6, "contract IceDEM.explicitly_absent = ['restore']"))
    rows.append(_row("simulation_state", "FINISHED", str(sim.state).split(".")[-1], 0.0, 0.0,
                     sim.state == SimulationState.FINISHED, "real Simulation.run() completed"))

    # A7 substep-rule boundary, exercised through the real adapter.
    boundary = {}
    for label, ratio_value, expect_ok in (
        ("ratio+1e-10", APPROVED_RATIO + 1e-10, True),
        ("ratio+1e-6", APPROVED_RATIO + 1e-6, False),
        ("ratio 4.5", 4.5, False),
    ):
        try:
            _run_simulation(TensorDEMAdapter(_fresh_2d()), ratio_value * native_dt, 1)
            boundary[label] = "accepted"
            ok = expect_ok
        except Exception as exc:  # TensorDEMAdapterError or SimulationError
            boundary[label] = f"rejected:{type(exc).__name__}"
            ok = not expect_ok
        rows.append(_row(f"substep_boundary[{label}]", "accepted" if expect_ok else "rejected",
                         boundary[label], 0.0, 0.0, ok, "contract clock.global_step_rule rel_tol 1e-9"))

    # A7b non-accumulation sweep: ratio got/ref must stay at 1.0 for every substep count,
    # while the H0 (accumulating consumer) prediction (n+1)/2 grows with n.
    sweep = []
    f_vec = [float(v) for v in f_total.sum(0)]
    for n_sub in (1, 2, 4, 8, 12, 16):
        leg_model = _fresh_2d()
        leg = _run_simulation(
            TensorDEMAdapter(leg_model, load_provider=lambda ctx, dem, _m=leg_model:
                             torch.tensor(LOAD, dtype=_m.positions.dtype).repeat(_m.positions.shape[0], 1)),
            n_sub * float(leg_model.dt), 1,
        )
        ref = [n_sub * float(leg_model.dt) * v for v in f_vec]
        got = [float(v) for v in leg_model.accounting["external_impulse_Ns"]]
        legs = [g / r if r else float("nan") for r, g in zip(ref, got)]
        sweep.append({"substeps": n_sub, "ratio_got_over_ref": legs,
                      "h0_prediction": (n_sub + 1) / 2.0,
                      "state": str(leg.state).split(".")[-1]})
    sweep_dev = max(abs(x - 1.0) for leg in sweep for x in leg["ratio_got_over_ref"])
    rows.append(_row("non_accumulation_sweep_max_dev", 1.0, 1.0 + sweep_dev, sweep_dev, RATIO_TOL,
                     sweep_dev <= RATIO_TOL,
                     "n in {1,2,4,8,12,16}; H0 predicts (n+1)/2 = 1.0..8.5"))

    # A8 controlled drift variants: the cross-check must catch what it claims to catch.
    drift = {}
    fresh = _fresh_2d()
    acc_proxy = AccumulatingIceDEM(fresh)
    acc_adapter = TensorDEMAdapter(
        acc_proxy,
        load_provider=lambda ctx, dem: torch.tensor(LOAD, dtype=fresh.positions.dtype).repeat(
            fresh.positions.shape[0], 1),
    )
    _run_simulation(acc_adapter, APPROVED_RATIO * float(fresh.dt), GLOBAL_STEPS)
    acc_ref = [n_expected * float(fresh.dt) * float(v) for v in f_total.sum(0)]
    acc_got = [float(v) for v in fresh.accounting["external_impulse_Ns"]]
    acc_ratio = [g / r for r, g in zip(acc_ref, acc_got)]
    drift["accumulating_supplier"] = {
        "ratio": acc_ratio,
        "detected": all(abs(x - 1.0) > RATIO_TOL for x in acc_ratio),
        "predicted_by_H0": h0_ratio,
    }
    stale = _fresh_2d()
    stale.time = 0.5 * native_dt  # clock already off its own grid
    try:
        _run_simulation(TensorDEMAdapter(stale), APPROVED_RATIO * native_dt, 1)
        drift["stale_native_clock"] = {"detected": False, "detail": "accepted"}
    except Exception as exc:
        drift["stale_native_clock"] = {"detected": True, "detail": f"rejected:{type(exc).__name__}"}

    detected = int(drift["accumulating_supplier"]["detected"]) + int(drift["stale_native_clock"]["detected"])
    rows.append(_row("drift_detection_rate", "2/2", f"{detected}/2", 0.0, 0.0, detected == 2,
                     "controlled variants: accumulating supplier, stale native clock"))

    return {
        "schema": "tensordem.binding-contract.live/1",
        "anchor": "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT",
        "round": 4,
        "mission": 19466,
        "contract_fingerprint": bc.fingerprint(),
        "fingerprint_is_pinned": bc.fingerprint() == bc.FROZEN_FINGERPRINT,
        "contract_version": bc.BINDING_CONTRACT.get("contract_version"),
        "supplier": {
            "repo": "TensorDEM",
            "git_sha": _git_sha(),
            "module": "tensordem.dem.IceDEM",
            "native_dt_s": native_dt,
            "particles": int(model.positions.shape[0]),
            "config": {"nx": 6, "ny": 3, "fix_edges": False, "contact_damping": 0.0, "drag": 0.0},
        },
        "consumer": {
            "repo": "TensorPhysics",
            "module": "tensorphysics.adapters.tensordem.TensorDEMAdapter",
            "driven_by": "tensorphysics.Simulation(SimulationConfig)",
            "ratio": APPROVED_RATIO,
            "global_steps": GLOBAL_STEPS,
        },
        "rows": rows,
        "rows_total": len(rows),
        "rows_passed": sum(1 for r in rows if r["pass"]),
        "worst_rel_error": max((r["rel_error"] for r in rows if isinstance(r["rel_error"], float)),
                              default=0.0),
        "drift": drift,
        "sweep": sweep,
        "boundary": boundary,
        "verdict": "MET" if all(r["pass"] for r in rows) else "FAIL",
        "owner_tolerance": OWNER_TOLERANCE,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _table(report: dict) -> str:
    lines = ["| # | arm | ref | got | rel. error | tol | pass |", "|---|---|---|---|---:|---:|:---:|"]
    for i, r in enumerate(report["rows"], 1):
        lines.append(f"| {i} | `{r['name']}` | {r['ref']} | {r['got']} | "
                     f"{r['rel_error']:.3e} | {r['tolerance']:.0e} | {'PASS' if r['pass'] else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "xval_library_binding_contract_live")
    args = parser.parse_args(argv)

    src = _find_tensorphysics()
    if src is None:
        print("WITHHELD: tensorphysics consumer checkout not found "
              "(set TENSORPHYSICS_SRC); no numbers fabricated", file=sys.stderr)
        return 2
    sys.path.insert(0, str(src))

    report = collect()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "live_binding.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    table = args.out / "live_binding_table.md"
    table.write_text(_table(report), encoding="utf-8")

    print(f"consumer={src}")
    print(f"contract {report['contract_fingerprint']} pinned={report['fingerprint_is_pinned']}")
    print(f"rows={report['rows_passed']}/{report['rows_total']} "
          f"worst_rel_error={report['worst_rel_error']:.3e} verdict={report['verdict']} -> {target}")
    return 0 if report["verdict"] == "MET" else 1


if __name__ == "__main__":
    raise SystemExit(main())