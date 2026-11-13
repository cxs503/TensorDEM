#!/usr/bin/env python3
"""XVAL run: TensorDEM's frozen binding contract vs the TensorPhysics practice.

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT (round 1).

Emits ``artifacts/xval_library_binding_contract/result.json`` plus figures.
Every number in the JSON is produced here (no hand transcription); the report
and the paper table read this file.

Usage::

    python scripts/xval_library_binding_contract.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import binding_contract as bc  # noqa: E402

OWNER_TOLERANCE = 0.15  # owner accuracy standard: >15 % is not acceptable


def _public_surface_coverage() -> dict:
    from tensordem.dem import IceDEM
    from tensordem.dem3d import IceDEM3D

    out = {}
    for name, cls in (("IceDEM", IceDEM), ("IceDEM3D", IceDEM3D)):
        public = [
            attr for attr in dir(cls)
            if not attr.startswith("_")
            and (callable(getattr(cls, attr, None))
                 or isinstance(inspect.getattr_static(cls, attr, None), property))
        ]
        entry = bc.BINDING_CONTRACT["entries"][name]
        frozen = (set(entry["methods"]) | set(entry["properties"])) & set(public)
        out[name] = {
            "public_members": sorted(public),
            "public_count": len(public),
            "frozen_count": len(frozen),
            "frozen_fraction": len(frozen) / len(public) if public else None,
            "unfrozen_members": sorted(set(public) - frozen),
        }
    return out


def _name_only_baseline() -> dict:
    """Control: a presence-only inventory check (the pre-contract baseline).

    The baseline answers "is the frozen *name* still there?".  It cannot see
    signature kinds, requiredness, units or negative assertions, which is
    exactly the drift class that breaks binders silently.
    """
    rows = []
    mutants = bc._mutants()  # controlled mutants, one per drift class
    for key, mutant in mutants.items():
        cls = type(mutant)
        missing = []
        for entry_name, entry in bc.BINDING_CONTRACT["entries"].items():
            if entry_name != "IceDEM":
                continue
            names = (list(entry["state"]) + list(entry["properties"])
                     + list(entry["methods"]))
            for name in names:
                if not hasattr(mutant, name):
                    missing.append(name)
        rows.append({"key": key, "baseline_detected": bool(missing),
                     "missing_names": missing})
    detected = sum(r["baseline_detected"] for r in rows)
    return {"baseline": "presence-only inventory (names exist)",
            "detected": detected, "total": len(rows),
            "rate": detected / len(rows),
            "missed": [r["key"] for r in rows if not r["baseline_detected"]],
            "rows": rows}


def _impulse_table() -> list[dict]:
    """Impulse-momentum sweep: analytic ``n*dt*F`` vs measured ``m*sum(dv)``."""
    import torch

    from tensordem.dem import DEMConfig, IceDEM

    rows = []
    for n_steps in (1, 2, 4, 8, 16, 32, 64, 128):
        model = IceDEM(DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                                 contact_damping=0.0, drag=0.0))
        mass = model.config.mass
        load = torch.zeros_like(model.positions)
        load[:, 0] = 0.25
        load[:, 1] = -0.1
        v0 = model.velocities.clone()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model.step(external_forces=load, load_time=model.time)
        wall = time.perf_counter() - t0
        dp = mass * (model.velocities - v0).sum(dim=0)
        ref = n_steps * model.dt * load.sum(dim=0)
        rel = float((dp - ref).abs().max().item() / ref.abs().max().item())
        ratio = float((dp @ ref) / (ref @ ref))
        rows.append({
            "n_steps": n_steps,
            "native_dt_s": float(model.dt),
            "load_N_per_particle": [0.25, -0.1],
            "particles": int(model.positions.shape[0]),
            "analytic_impulse_Ns": ref.tolist(),
            "measured_momentum_Ns": dp.tolist(),
            "rel_error": rel,
            "impulse_ratio": ratio,
            "wall_s": wall,
        })
    return rows


def _accumulation_discriminator() -> dict:
    """H0 (binder keeps and re-adds loads) vs H1 (instantaneous loads).

    A naive accumulator that re-sends the *cumulative* load each native step
    produces ``sum_{k=1..n} k = n(n+1)/2`` times the exact impulse.  The
    measured ratio must sit on H1 (1.0) and be separated from H0 by far more
    than the 15 % owner tolerance.
    """
    rows = []
    for n_steps in (2, 4, 8, 16):
        h0_ratio = (n_steps + 1) / 2.0
        rows.append({"n_steps": n_steps, "h0_accumulating_ratio": h0_ratio,
                     "separation_vs_h1": h0_ratio - 1.0})
    probe = next(r for r in bc.probe_semantics() if r["name"] == "loads_not_accumulated")
    measured = float(probe["got"].split("=")[1])
    return {"measured_ratio": measured,
            "h1_prediction": 1.0,
            "h1_supported": abs(measured - 1.0) < 1e-12,
            "rows": rows}


def _clock_and_restart() -> dict:
    import torch

    from tensordem.dem import DEMConfig, IceDEM

    model = IceDEM(DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                             contact_damping=0.0, drag=0.0))
    load = torch.zeros_like(model.positions)
    load[:, 1] = 0.3
    steps = 10
    for _ in range(steps):
        model.step(external_forces=load, load_time=model.time)
    ref_time = steps * float(model.dt)
    clock_err = abs(model.time - ref_time) / ref_time

    snap = model.snapshot()
    clone = IceDEM.from_snapshot(snap)
    positions_equal = bool(clone.positions.equal(model.positions))
    # continuation determinism: one more step must agree bitwise
    model.step(external_forces=load, load_time=model.time)
    clone.step(external_forces=load, load_time=clone.time)
    continuation_equal = bool(clone.positions.equal(model.positions))
    drift = float((clone.positions - model.positions).abs().max().item())
    return {
        "steps": steps,
        "native_dt_s": float(model.dt),
        "clock_rel_error": clock_err,
        "step_count": int(model.step_count),
        "restart_schema": snap["schema"],
        "restart_payload_sha256": hashlib.sha256(
            json.dumps(snap, sort_keys=True).encode()).hexdigest(),
        "restart_positions_bitwise_equal": positions_equal,
        "restart_continuation_bitwise_equal": continuation_equal,
        "restart_continuation_max_position_drift_m": drift,
    }


def _figures(outdir: Path, result: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figure path only
        print(f"matplotlib unavailable: {exc}")
        return []

    made = []

    # 1. impulse error vs substep count ------------------------------------
    rows = result["impulse_momentum_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ns = [r["n_steps"] for r in rows]
    err = [max(r["rel_error"], 1e-18) for r in rows]
    ax.loglog(ns, err, "o-", label="measured |Δp − n·dt·F| / |n·dt·F|")
    ax.axhline(OWNER_TOLERANCE, color="crimson", ls="--",
               label="owner tolerance 15 %")
    ax.axhline(1e-12, color="grey", ls=":", label="contract floor 1e-12")
    ax.set_xlabel("native substeps n")
    ax.set_ylabel("relative impulse error")
    ax.set_title("DEM binding: impulse-momentum identity vs substep count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_impulse_error.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 2. mutation detection matrix -----------------------------------------
    muts = result["mutation_matrix"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [m["key"].replace("_", " ") for m in muts]
    severities = {"breaking": 2.0, "narrowing": 1.5, "additive": 1.0, "doc": 0.5}
    values = [severities.get(m["observed_severity"], 0.0) if m["detected"] else 0.0
              for m in muts]
    colors = ["#c0392b" if m["observed_severity"] == "breaking" else "#f39c12"
              for m in muts]
    ax.barh(range(len(muts)), values, color=colors)
    ax.set_yticks(range(len(muts)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 2.4)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["missed", "doc", "additive", "narrowing", "breaking"], fontsize=8)
    ax.set_title(f"drift detector: {sum(m['detected'] for m in muts)}/{len(muts)} mutants classified")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = outdir / "fig_mutation_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 3. contract surface coverage -----------------------------------------
    cov = result["surface_coverage"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(cov)
    frozen = [cov[n]["frozen_count"] for n in names]
    unfrozen = [cov[n]["public_count"] - cov[n]["frozen_count"] for n in names]
    ax.bar(names, frozen, label="frozen in contract", color="#2980b9")
    ax.bar(names, unfrozen, bottom=frozen, label="not frozen (out of scope)",
           color="#bdc3c7")
    for idx, name in enumerate(names):
        ax.text(idx, frozen[idx] / 2, f"{cov[name]['frozen_fraction']:.0%}",
                ha="center", va="center", color="white", fontsize=9)
    ax.set_ylabel("public members")
    ax.set_title("binding-contract coverage of the public solver surface")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_coverage.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 4. substep acceptance boundary ---------------------------------------
    rows = result["substep_rule"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ratios = [r["ratio"] for r in rows]
    accepted = [r["accepted"] for r in rows]
    ax.plot(ratios, [1 if a else 0 for a in accepted], "o", label="accept decision")
    ax.axvline(round(ratios[0]), color="grey", ls=":", alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["reject", "accept"])
    ax.set_xlabel("global_dt / native_dt")
    ax.set_title("integer-multiple substep rule (rel_tol = abs_tol = 1e-9)")
    ax.set_xlim(0.5, 5.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_substep_boundary.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(
        ROOT / "artifacts" / "xval_library_binding_contract"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import torch

    t0 = time.perf_counter()
    verify_report = bc.verify()
    probes = bc.probe_semantics()
    mutations = bc.mutation_matrix()
    substep = bc.substep_rule_check()
    impulse = _impulse_table()
    discriminator = _accumulation_discriminator()
    baseline = _name_only_baseline()
    restart = _clock_and_restart()
    coverage = _public_surface_coverage()
    # Round 2 (m17960): the 3-D entry of the frozen contract gets its own
    # numeric anchor, row by row (ref vs got + tolerance + pass).
    benchmark_3d = bc.benchmark_physics_3d()

    impulse_errors = [r["rel_error"] for r in impulse]
    detection_rate = sum(m["detected"] for m in mutations) / len(mutations)
    probe_pass = sum(p["status"] == "PASS" for p in probes) / len(probes)

    verdicts = [
        {"metric": "impulse-momentum relative error (max over n_steps)",
         "reference": "analytic n*dt*F", "got": max(impulse_errors),
         "tolerance": OWNER_TOLERANCE,
         "met": max(impulse_errors) < OWNER_TOLERANCE},
        {"metric": "3-D impulse-momentum relative error",
         "reference": "analytic n*dt*F",
         "got": max(p["error"] for p in probes
                    if p["name"] == "impulse_momentum_identity_3d"
                    and p["error"] is not None),
         "tolerance": OWNER_TOLERANCE,
         "met": max(p["error"] for p in probes
                    if p["name"] == "impulse_momentum_identity_3d"
                    and p["error"] is not None) < OWNER_TOLERANCE},
        {"metric": "ledger (diagnostics) vs analytic impulse relative error",
         "reference": "analytic n*dt*F",
         "got": max(p["error"] for p in probes if p["name"] == "ledger_cross_path"),
         "tolerance": OWNER_TOLERANCE,
         "met": max(p["error"] for p in probes
                    if p["name"] == "ledger_cross_path") < OWNER_TOLERANCE},
        {"metric": "native clock vs n*dt relative error",
         "reference": "analytic n*dt", "got": restart["clock_rel_error"],
         "tolerance": OWNER_TOLERANCE,
         "met": restart["clock_rel_error"] < OWNER_TOLERANCE},
        {"metric": "restart drift (max |Δposition| after restart+1 step)",
         "reference": "0 m (bitwise)", "got": restart["restart_continuation_max_position_drift_m"],
         "tolerance": 0.0,
         "met": restart["restart_continuation_bitwise_equal"]},
        {"metric": "drift-detector classification rate over 7 mutation classes",
         "reference": "7/7 expected classifications", "got": detection_rate,
         "tolerance": 1.0, "met": detection_rate == 1.0},
        {"metric": "semantic probe pass rate",
         "reference": "all declared probes", "got": probe_pass,
         "tolerance": 1.0, "met": probe_pass == 1.0},
        {"metric": "3-D numeric benchmark pass rate (ref vs got rows)",
         "reference": "all rows within their pre-registered tolerance",
         "got": benchmark_3d["rows_passed"] / benchmark_3d["rows_total"],
         "tolerance": 1.0, "met": benchmark_3d["verdict"] == "MET"},
        {"metric": "3-D worst row relative error",
         "reference": "0 (exact analytic rows); owner acceptance 0.15",
         "got": benchmark_3d["worst_rel_error"],
         "tolerance": OWNER_TOLERANCE,
         "met": benchmark_3d["worst_rel_error"] < OWNER_TOLERANCE},
        {"metric": "contract conformance (introspection checks matched)",
         "reference": "all frozen checks",
         "got": verify_report["matched"] / verify_report["checks"],
         "tolerance": 1.0, "met": verify_report["status"] == "CONFORMS"},
    ]

    result = {
        "schema": "tensordem.xval.library-binding-contract/1",
        "anchor": "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT",
        "round": 1,
        "reference_practice": {
            "repo": "TensorPhysics",
            "artifacts": ["docs/adapter-contracts.md", "docs/adapter-inventory.md",
                          "src/tensorphysics/adapters/tensordem.py",
                          "src/tensorphysics/protocols.py",
                          "tests/test_adapter_inventory_meta.py",
                          "tests/test_tensordem_adapter.py"],
            "binding": "Solver Protocol + TensorDEMAdapter (2-D IceDEM, substep clock)",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "contract_schema": bc.CONTRACT_SCHEMA,
            "contract_version": bc.CONTRACT_VERSION,
            "contract_fingerprint": bc.fingerprint(),
            "frozen_fingerprint": bc.FROZEN_FINGERPRINT,
            "frozen_fingerprint_match": bc.fingerprint() == bc.FROZEN_FINGERPRINT,
        },
        "verify": verify_report,
        "semantic_probes": probes,
        "mutation_matrix": mutations,
        "baseline_comparison": {
            "control": baseline,
            "treatment": {"detector": "frozen contract (names + signature kinds "
                                      "+ requiredness + annotations + negative assertions)",
                          "detected": sum(m["detected"] for m in mutations),
                          "total": len(mutations),
                          "rate": detection_rate},
            "delta_rate": detection_rate - baseline["rate"],
        },
        "substep_rule": substep,
        "impulse_momentum_sweep": impulse,
        "accumulation_discriminator": discriminator,
        "clock_and_restart": restart,
        "surface_coverage": coverage,
        "benchmark_physics_3d": benchmark_3d,
        "verdicts": verdicts,
        "wall_s": time.perf_counter() - t0,
    }
    result["figures"] = _figures(outdir, result)

    path = outdir / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: result[k] for k in
                      ("environment", "verdicts", "wall_s", "figures")}, indent=2)[:4000])
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())