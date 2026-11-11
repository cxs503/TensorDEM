"""Ice model-test scale cascade: two-scale Cauchy collapse + mismatch law.

Runnable in three stages (the DEM sweep and the analysis are deliberately
separate so the analysis never re-runs the DEM):

``sweep``      run every cascade member once, one JSON record per case under
               ``runs/ice_scale_cascade/cases/``.
``analysis``   read those records and write the fits, the criteria verdicts,
               the report numbers and the figures into
               ``artifacts/ice_scale_cascade/``.
``all``        sweep then analysis.  ``--only <kind>`` restricts the sweep to
               one design family (``cauchy``/``froude``/``mismatch``/
               ``resolution``/``contacts``) which is what the cluster shards use.

Usage
-----
``python benchmarks/ice_scale_cascade.py --stage all --jobs 32``
``python benchmarks/ice_scale_cascade.py --stage sweep --only cauchy --jobs 8``
``python benchmarks/ice_scale_cascade.py --stage analysis``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tensordem.ice_scale_cascade import (  # noqa: E402
    DEFAULT_LAMBDAS,
    PROTOCOL_CAUCHY,
    PROTOCOL_FROUDE,
    PROTOCOL_MISMATCH,
    ScaleCase,
    build_case,
    collapse_deviation,
    collapse_statistics,
    contact_sensitivity_pairs,
    convergence_check,
    evaluate_criteria,
    exact_covariance_pairs,
    fit_deviation_law,
    simulate_scale_case,
)

RUN_DIR = _REPO / "runs" / "ice_scale_cascade"
CASE_DIR = RUN_DIR / "cases"
OUT_DIR = _REPO / "artifacts" / "ice_scale_cascade"
FIG_DIR = OUT_DIR / "figures"

#: Engineering-protocol modulus exponent (declared assumption: ``E_m = E_p
#: lam^-0.5``) and the two ends of the admissible range used as a sensitivity
#: family.
FROUDE_MODULUS_EXPONENT = 0.5
#: ``Mod`` mismatch levels of the P-mismatch family (task book: +-30 %).
MISMATCH_LEVELS = (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30)
#: Resolution ladder (contact-CFL fraction) for the convergence criterion.
DT_LEVELS = (0.5, 0.25, 0.125, 0.0625)
#: Realisation seeds of the ensemble family: the scatter across these is the
#: numerical noise floor of any single-realisation collapse statement.
ENSEMBLE_SEEDS = (20261111, 20261112, 20261113)
#: Contact-count perturbations at fixed physics (numerical knobs only).
CONTACT_KNOBS = (
    {"radius_jitter": 0.02, "knob": "field_jitter_coarse"},
    {"radius_jitter": 0.16, "knob": "field_jitter_fine"},
    {"seed": 20261111, "knob": "seed_a"},
    {"seed": 20261112, "knob": "seed_b"},
    {"dt_scale": 0.125, "knob": "dt_half"},
)


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def design_tasks() -> list[dict]:
    """The predeclared cascade design (every task is a ``ScaleCase`` patch)."""
    tasks: list[dict] = []

    for lam in DEFAULT_LAMBDAS:
        case = build_case(PROTOCOL_CAUCHY, lam, label=f"cauchy_lam{lam:g}")
        tasks.append({"kind": "cauchy", "id": case.id(), "case": case})

    for lam in DEFAULT_LAMBDAS[1:]:
        case = build_case(PROTOCOL_FROUDE, lam, modulus_exponent=FROUDE_MODULUS_EXPONENT,
                          label=f"froude_lam{lam:g}")
        tasks.append({"kind": "froude", "id": case.id(), "case": case})
    # engineering-protocol assumption sensitivity: unscaled modulus (p_E = 0)
    for lam in (10.0, 20.0):
        case = build_case(PROTOCOL_FROUDE, lam, modulus_exponent=0.0,
                          label=f"froudeE0_lam{lam:g}")
        tasks.append({"kind": "froude", "id": case.id(), "case": case})

    for dmod in MISMATCH_LEVELS:
        case = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.0 + dmod,
                          label=f"mismatch_dm{dmod:+.2f}")
        tasks.append({"kind": "mismatch", "id": case.id(), "case": case})

    # ensemble: field-realisation scatter at fixed physics
    for proto, lam, q in ((PROTOCOL_CAUCHY, 1.0, None), (PROTOCOL_CAUCHY, 10.0, None),
                          (PROTOCOL_CAUCHY, 20.0, None),
                          (PROTOCOL_MISMATCH, 10.0, 0.7)):
        for seed in ENSEMBLE_SEEDS:
            tag = f"{'cauchy' if q is None else 'mismatchqm0.7'}_lam{lam:g}_s{seed}"
            case = build_case(proto, lam, modulus_factor=q, seed=seed, label=tag)
            tasks.append({"kind": "ensemble", "id": case.id(), "case": case})

    for dt_scale in DT_LEVELS:
        case = build_case(PROTOCOL_CAUCHY, 10.0, dt_scale=dt_scale,
                          label=f"res_dt{dt_scale:g}")
        tasks.append({"kind": "resolution", "id": case.id(), "case": case})

    base = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.3,
                      label="contacts_base")
    tasks.append({"kind": "contacts", "id": base.id(), "case": base})
    for knob in CONTACT_KNOBS:
        overrides = {k: v for k, v in knob.items() if k != "knob"}
        case = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.3,
                          label=f"contacts_{knob['knob']}", **overrides)
        tasks.append({"kind": "contacts", "id": case.id(), "case": case,
                      "knob": knob["knob"]})
    return tasks


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _record_path(case_id: str) -> Path:
    return CASE_DIR / f"{case_id}.json"


def _run_task(task: dict) -> dict:
    """Run one case, persist its record and echo it on stdout.

    The stdout echo (``RECORD {json}``) is what makes the cluster workflow
    lossless: the sweep log alone is enough to rebuild every case file
    (``--ingest``), so a lost work directory cannot destroy the evidence.
    """
    record = simulate_scale_case(task["case"])
    record["kind"] = task["kind"]
    record["knob"] = task.get("knob", "")
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    _record_path(task["id"]).write_text(json.dumps(record))
    print("RECORD " + json.dumps(record), flush=True)
    return {"id": task["id"], "kind": task["kind"], "R_star": record["R_star"],
            "delta_contacts": record["contacts_per_floe"],
            "wall_time_s": record["wall_time_s"]}


def ingest_log(path: str | Path) -> int:
    """Rebuild case records from a sweep log's ``RECORD`` lines."""
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in Path(path).read_text().splitlines():
        if not line.startswith("RECORD "):
            continue
        record = json.loads(line[len("RECORD "):])
        _record_path(record["id"]).write_text(json.dumps(record))
        n += 1
    print(f"[ingest] {n} records from {path}", flush=True)
    return n


def stage_sweep(jobs: int, only: str | None = None) -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = design_tasks()
    if only:
        kinds = {k.strip() for k in only.split(",") if k.strip()}
        tasks = [t for t in tasks if t["kind"] in kinds]
    tasks = [t for t in tasks if not (CASE_DIR / f"{t['id']}.json").exists()]
    print(f"[sweep] {len(tasks)} cases, jobs={jobs}", flush=True)
    started = time.time()
    if jobs <= 1:
        for task in tasks:
            print("DONE " + json.dumps(_run_task(task)), flush=True)
        return
    with mp.Pool(jobs) as pool:
        for out in pool.imap_unordered(_run_task, tasks):
            print("DONE " + json.dumps(out), flush=True)
    print(f"[sweep] done in {time.time() - started:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    if not CASE_DIR.exists():
        raise SystemExit(f"no case records under {CASE_DIR}; run --stage sweep first")
    records = [json.loads(p.read_text()) for p in sorted(CASE_DIR.glob("*.json"))]
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: A *condition* is everything except the field seed: changing the seed is a new
#: realisation of the same physics, and the scatter across realisations is the
#: numerical noise floor of the collapse statement.
COND_FIELDS = ("protocol", "lam", "modulus_factor", "modulus_exponent",
               "dt_scale", "radius_jitter")
CANONICAL_DT_SCALE = 0.25
CANONICAL_JITTER = 0.08


def condition_key(record: dict) -> tuple:
    case = record["case"]
    return (case["protocol"], float(case["lam"]),
            round(float(case["modulus_factor"]), 9),
            round(float(case["modulus_exponent"]), 9),
            float(case["dt_scale"]), float(case["radius_jitter"]))


def build_conditions(records: list[dict]) -> list[dict]:
    """Group realisations into conditions and report mean / s.d. / s.e. of ``R*``."""
    buckets: dict[tuple, dict[int, dict]] = {}
    for r in records:
        # de-duplicate identical runs of the same case (e.g. a case that is both
        # the Cauchy base point and the coarsest resolution level)
        buckets.setdefault(condition_key(r), {})[int(r["case"]["seed"])] = r
    conditions = []
    for key, by_seed in buckets.items():
        rs = list(by_seed.values())
        values = np.asarray([float(r["R_star"]) for r in rs], dtype=float)
        n = int(values.size)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        sem = std / math.sqrt(n) if n > 1 else float("nan")
        conditions.append({
            "protocol": key[0], "lam": key[1], "modulus_factor": key[2],
            "modulus_exponent": key[3], "dt_scale": key[4],
            "radius_jitter": key[5], "n": n,
            "R_star_mean": mean, "R_star_std": std, "R_star_sem": sem,
            "R_star_rel_scatter": std / mean if n > 1 and mean else float("nan"),
            "contacts_per_floe": float(np.mean([r["contacts_per_floe"] for r in rs])),
            "ids": sorted(r["id"] for r in rs),
            "seeds": sorted(int(r["case"]["seed"]) for r in rs),
        })
    conditions.sort(key=lambda c: (c["protocol"], c["lam"], c["modulus_factor"],
                                   c["dt_scale"], c["radius_jitter"]))
    return conditions


def _canonical(conditions: list[dict], protocol: str) -> list[dict]:
    sel = [c for c in conditions if c["protocol"] == protocol
           and abs(c["dt_scale"] - CANONICAL_DT_SCALE) < 1e-12
           and abs(c["radius_jitter"] - CANONICAL_JITTER) < 1e-12]
    return sorted(sel, key=lambda c: (c["lam"], c["modulus_factor"]))


def noise_floor(conditions: list[dict]) -> dict:
    """Pooled realisation scatter over every condition with n >= 2."""
    multi = [c for c in conditions if c["n"] >= 2 and math.isfinite(c["R_star_std"])]
    if not multi:
        return {"n_conditions": 0}
    var = float(np.mean([c["R_star_std"] ** 2 for c in multi]))
    rel = float(np.mean([c["R_star_std"] / c["R_star_mean"] for c in multi]))
    return {"n_conditions": len(multi),
            "n_realisations": sum(c["n"] for c in multi),
            "pooled_std_abs": math.sqrt(var),
            "pooled_rel_scatter": rel,
            "per_condition": [{"ids": c["ids"], "lam": c["lam"],
                               "protocol": c["protocol"], "n": c["n"],
                               "R_star_mean": c["R_star_mean"],
                               "R_star_std": c["R_star_std"],
                               "R_star_sem": c["R_star_sem"],
                               "rel_scatter": c["R_star_rel_scatter"]}
                              for c in multi]}


def analyse(records: list[dict]) -> dict:
    """Turn the case records into the collapse table, the criteria and the law."""
    by_kind: dict[str, list[dict]] = {}
    for r in records:
        by_kind.setdefault(r.get("kind", ""), []).append(r)
    conditions = build_conditions(records)

    cauchy_conds = _canonical(conditions, PROTOCOL_CAUCHY)
    if not cauchy_conds:
        raise SystemExit("no canonical P-Cauchy condition: criterion (a) is undefined")
    anchor = min(cauchy_conds, key=lambda c: c["lam"])
    if abs(anchor["lam"] - 1.0) > 1e-9:
        raise SystemExit(f"the lambda = 1 anchor is missing (lowest lambda "
                         f"{anchor['lam']:g}); the deviation reference is undefined")
    reference = float(anchor["R_star_mean"])

    def _delta(value: float) -> float:
        return (float(value) - reference) / reference

    # single-realisation deviations (seed 20261110 only) for the audit trail
    single_seed = 20261110
    def _single(lam: float) -> float | None:
        for r in records:
            c = r["case"]
            if (c["protocol"] == PROTOCOL_CAUCHY and abs(float(c["lam"]) - lam) < 1e-12
                    and abs(float(c["modulus_factor"]) - 1.0) < 1e-12
                    and abs(float(c["dt_scale"]) - CANONICAL_DT_SCALE) < 1e-12
                    and abs(float(c["radius_jitter"]) - CANONICAL_JITTER) < 1e-12
                    and int(c["seed"]) == single_seed):
                return float(r["R_star"])
        return None

    single_ref = _single(anchor["lam"])
    cauchy_rows = []
    for c in cauchy_conds:
        d = _delta(c["R_star_mean"])
        d_sem = ((c["R_star_sem"] / reference)
                 if math.isfinite(c["R_star_sem"]) else float("nan"))
        single = _single(c["lam"])
        single_delta = (100.0 * (single - single_ref) / single_ref
                        if (single is not None and single_ref) else None)
        cauchy_rows.append({
            "id": c["ids"][0], "ids": c["ids"], "lam": c["lam"], "n": c["n"],
            "R_star": c["R_star_mean"], "R_star_sem": c["R_star_sem"],
            "contacts_per_floe": c["contacts_per_floe"],
            "delta": d, "delta_pct": 100.0 * d, "delta_sem_pct": 100.0 * d_sem,
            "single_realisation_delta_pct": single_delta})

    exact_pairs = exact_covariance_pairs(records)
    ensemble_rows = [r for r in cauchy_rows if r["n"] > 1]

    froude_conds = [c for c in _canonical(conditions, PROTOCOL_FROUDE)
                    if c["lam"] > 1.0]
    froude_rows = [{"id": c["ids"][0], "lam": c["lam"], "n": c["n"],
                    "modulus_exponent": c["modulus_exponent"],
                    "q": c["modulus_factor"], "R_star": c["R_star_mean"],
                    "delta_pct": 100.0 * _delta(c["R_star_mean"])}
                   for c in froude_conds]
    froude_default = [r for r in froude_rows
                      if abs(r["modulus_exponent"] - FROUDE_MODULUS_EXPONENT) < 1e-12]
    froude_e0 = [r for r in froude_rows if r["modulus_exponent"] == 0.0]

    mismatch_rows = [{"id": c["ids"][0], "lam": c["lam"], "n": c["n"],
                      "dMod": c["modulus_factor"] - 1.0,
                      "R_star": c["R_star_mean"],
                      "delta_pct": 100.0 * _delta(c["R_star_mean"])}
                     for c in _canonical(conditions, PROTOCOL_MISMATCH)
                     if c["lam"] == 10.0 and abs(c["modulus_factor"] - 1.0) > 1e-9]

    contact_records = [r for r in by_kind.get("contacts", [])]
    contact_base = [r for r in contact_records if not r.get("knob")]
    contact_vars = [r for r in contact_records if r.get("knob")]
    for r in contact_records:
        r["delta_pct"] = 100.0 * _delta(r["R_star"])
    contact_sens = (contact_sensitivity_pairs(contact_base[0], contact_vars)
                    if contact_base and contact_vars else {"pairs": []})
    # a *field preserving* subset: only the time step moved, the floe field is
    # bit-identical, so the contact census change is a pure discretisation knob
    field_preserving = [p for p in contact_sens.get("pairs", [])
                        if p["knob"].startswith("dt")]

    resolution = by_kind.get("resolution", [])
    convergence = (convergence_check(resolution) if len(resolution) >= 3
                   else {"levels": [], "rel_change_finest": float("nan")})

    criteria = evaluate_criteria(
        cauchy=[r for r in cauchy_rows if r["lam"] > 1.0], froude=froude_default,
        mismatch=mismatch_rows, convergence=convergence,
        contact_sensitivity=contact_sens)
    c_sub = evaluate_criteria(
        cauchy=[r for r in cauchy_rows if r["lam"] > 1.0], froude=froude_default,
        mismatch=mismatch_rows,
        convergence=convergence,
        contact_sensitivity={"pairs": field_preserving})
    criteria["a"]["max_abs_delta_single_realisation_pct"] = max(
        [abs(r["single_realisation_delta_pct"]) for r in cauchy_rows
         if r["single_realisation_delta_pct"] is not None], default=float("nan"))
    criteria["a"]["ensemble_max_abs_delta_pct"] = max(
        [abs(r["delta_pct"]) for r in ensemble_rows], default=float("nan"))
    criteria["a"]["ensemble_pass"] = bool(
        ensemble_rows and criteria["a"]["ensemble_max_abs_delta_pct"] < 15.0)
    criteria["a"]["exact_pairs"] = exact_pairs
    criteria["a"]["exact_pairs_max_rel_diff"] = max(
        [p["rel_diff"] for p in exact_pairs], default=float("nan"))
    criteria["c"]["field_preserving_pass"] = c_sub["c"]["pass"]
    criteria["c"]["field_preserving_pairs"] = field_preserving
    criteria["c"]["note"] = (
        "all numerical knobs are evaluated literally; the seed/jitter knobs also "
        "change the field realisation, so their delta change is dominated by "
        "realisation scatter (see noise_floor), not by the contact census. The "
        "field_preserving_* fields isolate the pure contact-count knob.")

    law = fit_deviation_law(mismatch_rows) if len(mismatch_rows) >= 3 else None
    floor = noise_floor(conditions)

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": {"id": anchor["ids"][0], "lam": anchor["lam"],
                      "R_star": reference, "n": anchor["n"],
                      "case": anchor["case"] if "case" in anchor else None},
        "n_cases": len(records),
        "criteria": criteria,
        "noise_floor": floor,
        "exact_covariance": exact_pairs,
        "cauchy_collapse": cauchy_rows,
        "froude": froude_rows,
        "froude_assumption_sensitivity": froude_e0,
        "mismatch": mismatch_rows,
        "deviation_law": law,
        "conditions": conditions,
        "cauchy_spread": collapse_statistics([c["R_star_mean"] for c in cauchy_conds]),
        "contact_census": [
            {"id": r["id"], "lam": r["lam"], "protocol": r["protocol"],
             "contacts_per_floe": r["contacts_per_floe"],
             "floe_count": r["floe_count"]} for r in records],
        "cases": [{"id": r["id"], "kind": r.get("kind", ""), "knob": r.get("knob", ""),
                   "protocol": r["protocol"], "lam": r["lam"],
                   "modulus_factor": r["modulus_factor"],
                   "modulus_exponent": r["case"]["modulus_exponent"],
                   "R_star": r["R_star"], "R_rel_sem": r.get("R_rel_sem", float("nan")),
                   "contacts_per_floe": r["contacts_per_floe"],
                   "floe_count": r["floe_count"],
                   "dt_scale": r["case"]["dt_scale"],
                   "radius_jitter": r["case"]["radius_jitter"],
                   "seed": r["case"]["seed"],
                   "groups": r["groups"]} for r in records],
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "machine": platform.machine(), "torch": _torch_version()},
        "audit": {"case_files": len(records),
                  "case_dir": str(CASE_DIR.relative_to(_REPO)),
                  "sha256": {p.name: _sha256(p) for p in sorted(CASE_DIR.glob("*.json"))}},
    }
    anchor_records = [r for r in records
                      if condition_key(r) == (PROTOCOL_CAUCHY, anchor["lam"], 1.0, 0.5,
                                              CANONICAL_DT_SCALE, CANONICAL_JITTER)]
    if anchor_records:
        summary["reference"]["case"] = anchor_records[0]["case"]
    return summary


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:  # pragma: no cover
        return "unavailable"


def stage_analysis() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    summary = analyse(records)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    make_figures(summary)
    (OUT_DIR / "report.md").write_text(render_report(summary))
    crit = summary["criteria"]
    print(json.dumps({k: crit[k]["pass"] for k in ("a", "b", "c", "d")}
                     | {"all_pass": crit["all_pass"]}, indent=1))
    return summary


# ---------------------------------------------------------------------------
# figures + report
# ---------------------------------------------------------------------------

def make_figures(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = float(summary["reference"]["R_star"])
    cauchy = [r for r in summary["cauchy_collapse"] if r["lam"] > 1.0]
    froude = [r for r in summary["froude"]
              if abs(r["modulus_exponent"] - FROUDE_MODULUS_EXPONENT) < 1e-12
              and r["lam"] > 1.0]
    lam0 = summary["reference"]["lam"]

    # -- fig 1: collapse + deviation --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    ax.axhspan(ref * 0.85, ref * 1.15, color="#1b6ca8", alpha=0.10,
               label=r"$\pm15\%$ Cauchy band")
    ax.axhline(ref, color="#444444", lw=0.8, ls="--", label="full-scale $R^*$")
    if cauchy:
        ax.errorbar([lam0] + [r["lam"] for r in cauchy],
                    [ref] + [r["R_star"] for r in cauchy],
                    yerr=[0.0] + [r["R_star_sem"] for r in cauchy],
                    fmt="o", color="#1b6ca8", capsize=3,
                    label=r"P-Cauchy (ensemble mean $\pm$ s.e.)")
    for r in summary["cases"]:
        if r["kind"] in ("cauchy", "ensemble") and r["protocol"] == "P-Cauchy":
            ax.plot(r["lam"], r["R_star"], ".", color="#1b6ca8", alpha=0.35, ms=4)
    if froude:
        ax.errorbar([r["lam"] for r in froude], [r["R_star"] for r in froude],
                    fmt="s", color="#c0392b", capsize=3,
                    label=r"P-Froude+strength ($p_E=0.5$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"scale ratio $\lambda = L_p/L_m$")
    ax.set_ylabel(r"$R^* = R/(\sigma_f t^2)$")
    ax.set_title("dimensionless resistance vs scale ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")

    ax = axes[1]
    ax.axhspan(-15, 15, color="#1b6ca8", alpha=0.10)
    ax.axhline(0.0, color="#444444", lw=0.8, ls="--")
    if cauchy:
        ax.errorbar([r["lam"] for r in cauchy], [r["delta_pct"] for r in cauchy],
                    yerr=[r["delta_sem_pct"] for r in cauchy], fmt="o-",
                    color="#1b6ca8", capsize=3, label="P-Cauchy")
    if froude:
        ax.plot([r["lam"] for r in froude], [r["delta_pct"] for r in froude],
                "s--", color="#c0392b", label="P-Froude+strength")
    ax.set_xscale("log")
    ax.set_xlabel(r"scale ratio $\lambda$")
    ax.set_ylabel(r"$\delta$ [%]")
    ax.set_title("scale-effect deviation from the full-scale anchor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cauchy_collapse.png", dpi=170)
    plt.close(fig)

    # -- fig 2: deviation law ---------------------------------------------
    law = summary.get("deviation_law")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.axvline(0.0, color="#888888", lw=0.8)
    floor = summary.get("noise_floor", {})
    if floor.get("pooled_rel_scatter"):
        band = 100.0 * float(floor["pooled_rel_scatter"]) / math.sqrt(
            max(1, min(c["n"] for c in floor["per_condition"])))
        ax.axhspan(-band, band, color="#27ae60", alpha=0.10,
                   label=r"$\pm1$ s.e. of a "
                         f"{min(c['n'] for c in floor['per_condition'])}-seed ensemble")
    if law:
        pts = sorted(law["points"], key=lambda q: q["dMod"])
        ax.plot([q["dMod"] for q in pts], [q["delta_pct"] for q in pts], "o",
                color="#c0392b", label=r"P-mismatch DEM ($\lambda=10$)")
        xs = np.linspace(min(-0.36, min(q["dMod"] for q in pts) * 1.2),
                         max(0.36, max(q["dMod"] for q in pts) * 1.2), 300)
        form, fits = law["form"], law["fits"]
        if form == "linear":
            ys = fits["linear"]["a"] * xs
            eq = (r"$\delta = %.2f\,\Delta$, $R^2=%.3f$"
                  % (fits["linear"]["a"], fits["linear"]["r2"]))
        elif form == "quadratic":
            f = fits["quadratic"]
            ys = f["a"] * xs + f["b"] * xs ** 2
            eq = (r"$\delta = %.2f\Delta %+.2f\Delta^2$, $R^2=%.3f$"
                  % (f["a"], f["b"], f["r2"]))
        else:
            f = fits["power"]
            ys = np.sign(xs) * f["A"] * np.abs(xs) ** f["p"]
            eq = r"$|\delta| = %.2f\,|\Delta|^{%.2f}$, $R^2=%.3f$" % (
                f["A"], f["p"], f["r2"])
        ax.plot(xs, ys, "-", color="#1b6ca8", lw=1.5, label=f"{form}: " + eq)
    ax.set_xlabel(r"$\Delta(E/\sigma_f) = q-1$")
    ax.set_ylabel(r"$\delta$ [%]")
    ax.set_title("deviation law in the modulus/strength mismatch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_deviation_law.png", dpi=170)
    plt.close(fig)

    # -- fig 3: resolution ladder + contact census ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    levels = summary["criteria"]["d"]["levels"]
    ax = axes[0]
    if levels:
        ax.plot([lv["dt_scale"] for lv in levels], [lv["R_star"] for lv in levels],
                "o-", color="#1b6ca8")
        ax.set_xscale("log")
        ax.set_xlabel(r"time-step CFL fraction $dt_{scale}$")
        ax.set_ylabel(r"$R^*$")
        ax.set_title(r"resolution ladder ($\lambda=10$)")
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.3)
    ax = axes[1]
    rows = ([{"lam": lam0, "contacts_per_floe": summary["conditions"][0]["contacts_per_floe"]}]
            + cauchy) if cauchy else []
    if rows:
        ax.plot([r["lam"] for r in rows], [r["contacts_per_floe"] for r in rows],
                "o-", color="#27ae60")
        ax.set_xscale("log")
        ax.set_xlabel(r"scale ratio $\lambda$")
        ax.set_ylabel("contacts per floe (mean over the transit)")
        ax.set_title("contact census across the cascade")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_resolution_contacts.png", dpi=170)
    plt.close(fig)

    # -- fig 4: which groups each protocol actually matches ---------------
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    groups = ("Fn", "Sigma", "Mod", "t_over_L")
    labels = {"Fn": "Fn", "Sigma": r"$\sigma_f/(\rho_i g t)$",
              "Mod": r"$E/\sigma_f$", "t_over_L": "t/L"}
    ref_groups = summary["reference"]["case"]["groups"]
    xpos = np.arange(len(groups))
    width = 0.36
    for i, proto in enumerate(("P-Cauchy", "P-Froude+strength")):
        sel = [c for c in summary["cases"]
               if c["protocol"] == proto
               and abs(c["dt_scale"] - CANONICAL_DT_SCALE) < 1e-12
               and abs(c["radius_jitter"] - CANONICAL_JITTER) < 1e-12
               and c["lam"] > 1.0
               and (proto != "P-Froude+strength"
                    or abs(summary_groups_exponent(c) - FROUDE_MODULUS_EXPONENT) < 1e-12)]
        if not sel:
            continue
        worst = [max(abs(c["groups"][g] / ref_groups[g] - 1.0) for c in sel)
                 for g in groups]
        ax.bar(xpos + (i - 0.5) * width, [100.0 * w for w in worst], width, label=proto)
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels[g] for g in groups])
    ax.set_ylabel(r"max $|group/group_{ref} - 1|$ [%]")
    ax.set_title("group matching error of each protocol")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_group_matching.png", dpi=170)
    plt.close(fig)


def summary_groups_exponent(row: dict) -> float:
    """The Froude-protocol modulus exponent of a summary case row."""
    return float(row.get("modulus_exponent", FROUDE_MODULUS_EXPONENT))


def _f(value, fmt: str) -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "-"


def render_report(summary: dict) -> str:
    crit = summary["criteria"]
    law = summary.get("deviation_law")
    ref = summary["reference"]
    floor = summary.get("noise_floor", {})
    b_fit = crit["b"]["fit"]
    a_pass = "PASS" if crit["a"]["pass"] else "FAIL"
    b_pass = "PASS" if crit["b"]["pass"] else "FAIL"
    c_pass = "PASS" if crit["c"]["pass"] else "FAIL"
    d_pass = "PASS" if crit["d"]["pass"] else "FAIL"
    b_num = (f"R2={b_fit['r2']:.3f} (exponent {b_fit['exponent']:.3f})"
             if b_fit else "not fittable")
    lines = [
        "# Ice model-test scale cascade - programme-generated numbers",
        "",
        f"generated {summary['generated_utc']} - {summary['n_cases']} DEM cases - "
        f"anchor `{ref['id']}` (lambda={ref['lam']:g}, n={ref['n']}, "
        f"R*={ref['R_star']:.5f})",
        "",
        "| criterion | verdict | number |",
        "|---|---|---|",
        f"| (a) P-Cauchy max |delta| < 15 pct (all conditions) | {a_pass} | "
        f"{crit['a']['max_abs_delta_pct']:.2f} pct |",
        f"| (a-ens) same, ensemble-mean conditions only | "
        f"{'PASS' if crit['a'].get('ensemble_pass') else 'FAIL'} | "
        f"{crit['a'].get('ensemble_max_abs_delta_pct', float('nan')):.2f} pct |",
        f"| (a-exact) lambda ratio 4^k pairs must be bit identical | "
        f"{'PASS' if crit['a'].get('exact_pairs_max_rel_diff', 1.0) <= 1e-12 else 'FAIL'} "
        f"| max rel. diff = "
        f"{crit['a'].get('exact_pairs_max_rel_diff', float('nan')):.3e} |",
        f"| (b) P-Froude+strength |delta| ~ lambda, R2 > 0.90 | {b_pass} | {b_num} |",
        f"| (c) contacts-per-floe < 10 pct -> |delta| < 5 pct | {c_pass} | "
        f"max |d delta| = {crit['c']['max_abs_delta_change_pct']:.2f} pp "
        f"(field-preserving subset: "
        f"{'PASS' if crit['c'].get('field_preserving_pass') else 'FAIL'}) |",
        f"| (d) resolution |dR|/R < 3 pct | {d_pass} | "
        f"{100.0 * crit['d']['rel_change_finest']:.2f} pct |",
        "",
        "Field-realisation noise floor: pooled relative scatter "
        f"{100.0 * float(floor.get('pooled_rel_scatter', float('nan'))):.1f} pct over "
        f"{floor.get('n_realisations', 0)} realisations in "
        f"{floor.get('n_conditions', 0)} conditions.",
        "",
        "## Cauchy collapse (ensemble mean where n > 1)",
        "",
        "| lambda | n | R* | s.e. | delta pct | single-realisation delta pct "
        "| contacts/floe |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["cauchy_collapse"]:
        seed_delta = r.get("single_realisation_delta_pct")
        lines.append(
            f"| {r['lam']:g} | {r['n']} | {r['R_star']:.5f} | "
            f"{_f(r['R_star_sem'], '.5f') if r['n'] > 1 else '-'} | "
            f"{r['delta_pct']:+.2f} | "
            f"{_f(seed_delta, '+.2f') if seed_delta is not None else '-'} | "
            f"{r['contacts_per_floe']:.4f} |")
    if summary.get("exact_covariance"):
        lines += ["", "## Power-of-two scale pairs (exactness of the covariance)", "",
                  "| pair | lambda ratio | R* a | R* b | rel. diff |", "|---|---|---|---|---|"]
        for pr in summary["exact_covariance"]:
            lines.append(f"| {pr['a']} / {pr['b']} | {pr['lam_ratio']:g} | "
                         f"{pr['R_star_a']:.10f} | {pr['R_star_b']:.10f} | "
                         f"{pr['rel_diff']:.3e} |")
    lines += ["", "## Engineering protocol (Froude + strength)", "",
              "| lambda | p_E | q = E/sigma ratio | delta pct |", "|---|---|---|---|"]
    for r in sorted(summary["froude"], key=lambda q: (q["lam"], q["modulus_exponent"])):
        lines.append(f"| {r['lam']:g} | {r['modulus_exponent']:g} | {r['q']:.3f} | "
                     f"{r['delta_pct']:+.2f} |")
    if law:
        lines += ["", f"## Deviation law (selected form: {law['form']})", "",
                  "| d(E/sigma_f) | delta pct |", "|---|---|"]
        for q in law["points"]:
            lines.append(f"| {q['dMod']:+.2f} | {q['delta_pct']:+.3f} |")
        for name, fit in law["fits"].items():
            params = ", ".join(f"{k}={v:.4g}" for k, v in fit.items()
                               if k in ("a", "b", "A", "p"))
            lines += ["", f"* {name}: {params}, R2={fit['r2']:.4f}, "
                          f"SSres={fit['ss_res']:.4g}"]
    lines += ["", "## Resolution ladder", "",
              "| dt_scale | dt [s] | steps | R* |", "|---|---|---|---|"]
    for lv in crit["d"]["levels"]:
        lines.append(f"| {lv['dt_scale']:g} | {lv['dt_s']:.3e} | {lv['steps']} | "
                     f"{lv['R_star']:.5f} |")
    lines += ["", "## Contact-count perturbations", "",
              "| knob | contacts/floe | d(contacts) pct | delta pct | d(delta) pp |",
              "|---|---|---|---|---|"]
    for pr in crit["c"]["pairs"]:
        lines.append(f"| {pr['knob']} | {pr['contacts_per_floe']:.4f} | "
                     f"{pr['contact_change_pct']:+.2f} | {pr['delta_pct']:+.2f} | "
                     f"{pr['delta_change_pct']:+.2f} |")
    lines += ["", "## Ensemble scatter per condition", "",
              "| protocol | lambda | n | R* mean | s.d. | rel. scatter pct |",
              "|---|---|---|---|---|---|"]
    for c in floor.get("per_condition", []):
        lines.append(f"| {c['protocol']} | {c['lam']:g} | {c['n']} | "
                     f"{c['R_star_mean']:.5f} | {c['R_star_std']:.5f} | "
                     f"{100.0 * c['rel_scatter']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "analysis", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--only", default=None,
                        help="comma separated design families for the sweep")
    parser.add_argument("--ingest", default=None,
                        help="rebuild case records from a sweep log, then exit")
    args = parser.parse_args(argv)
    if args.ingest:
        ingest_log(args.ingest)
        return 0
    if args.stage in ("sweep", "all"):
        stage_sweep(args.jobs, args.only)
    if args.stage in ("analysis", "all"):
        stage_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())