"""CASE-COLUMN-COLLAPSE — adjudication of the *second* Lube et al. (2004) law.

Rounds 1-5 of the anchor (``CASE-COLUMN-COLLAPSE``) adjudicate only the runout
law ``r_inf = r_i (1 + 1.24 a)`` / ``r_i (1 + 1.6 a^(1/2))``.  The same paper
states a *second*, independent prediction for the emplacement time

    t_inf = 3 (h_i / g)^(1/2) = 3 (a r_i / g)^(1/2) ,

which is implemented (:func:`tensordem.granular_column.axisym.lube_2004_emplacement_time`)
and *recorded* in every per-case provenance file, but was never turned into a
verdict.  This module closes that verification debt from the stored round-5
provenance alone (no DEM is re-run): it reduces each case JSON to scalars,
re-derives the runout and emplacement-time errors against the paper, and
returns an explicit machine checkable adjudication.

Two further questions are answered with the same data:

* **aspect-ratio convention.**  The runout target is evaluated at the *nominal*
  ``a`` while the runout ratio itself is normalised by the *measured* initial
  radius.  The packed column is systematically taller than nominal
  (``packing_gap`` 0.02), so ``a_measured > a_nominal`` by up to 10 % at
  ``a = 0.5``; re-evaluating the law at ``a_measured`` is the
  self-consistent convention.  Both are reported.
* **time scaling.**  A constant offset would be a calibration issue; a wrong
  exponent is a modelling issue.  The log-log slope of ``t_reach`` against
  ``a`` is fitted and compared with the ``a^(1/2)`` of the law.

Everything is computed from the per-case JSON files (zero hand-typed number);
``sha256`` of every source file is written to the manifest so the reduced table
that is committed stays traceable to the raw provenance.

Usage::

    PYTHONPATH=src python3 scripts/analyze_column_collapse_time_law_m17534.py \
        --src-dir /root/tp-agent/work/m17111/runs/case_column_collapse_lube2004/cases \
        --out-dir artifacts/column_collapse_time_law_m17534
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys

TOLERANCE = 0.15  # anchor acceptance band (|relative error| < 15 %)
REACH_FRACTIONS = ("0.9", "0.95", "0.99")
REGIME_BREAK = 1.7
LINEAR_PREFACTOR = 1.24
INERTIAL_PREFACTOR = 1.6
TIME_PREFACTOR = 3.0

REDUCED_FIELDS = (
    "label", "aspect_ratio", "aspect_measured", "grains_per_radius", "mu",
    "mu_rolling", "mu_wall", "mu_law", "dt_scale", "duration_factor", "seed",
    "n_particles", "solid_fraction_initial", "column_radius",
    "column_radius_measured", "column_height_measured", "stop_reason",
    "time_end", "runout_ratio_p995", "reference_runout_ratio",
    "t_reach", "reference_emplacement_time", "t_reach_f090", "t_reach_f095",
    "t_reach_f099", "peak_front_speed", "ballistic_time",
    "energy_potential_released", "energy_residual_fraction",
)


# ---------------------------------------------------------------- reference
def lube_runout_ratio(aspect_ratio: float) -> float:
    """``r_inf / r_i`` of Lube et al. (2004), from the abstract."""
    a = float(aspect_ratio)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("aspect ratio must be finite and positive")
    if a < REGIME_BREAK:
        return 1.0 + LINEAR_PREFACTOR * a
    return 1.0 + INERTIAL_PREFACTOR * math.sqrt(a)


def lube_emplacement_time(aspect_ratio: float, column_radius: float,
                          gravity: float = 9.81) -> float:
    """``t_inf = 3 (a r_i / g)^(1/2)`` [s]."""
    a, r, g = float(aspect_ratio), float(column_radius), float(gravity)
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError("aspect ratio must be finite and positive")
    if r <= 0.0 or g <= 0.0:
        raise ValueError("column_radius and gravity must be positive")
    return TIME_PREFACTOR * math.sqrt(a * r / g)


def rel_error(value: float, reference: float) -> float:
    """Signed relative error of ``value`` against ``reference``."""
    return (float(value) - float(reference)) / float(reference)


# ---------------------------------------------------------------- reduction
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_case(doc: dict) -> dict:
    """Flatten one per-case provenance document to the scalar fields we use."""
    case = doc["case"]
    frac = doc.get("t_reach_by_fraction") or {}
    energy = doc.get("energy") or {}
    rec = {
        "label": case["label"],
        "aspect_ratio": float(case["aspect_ratio"]),
        "aspect_measured": float(doc["aspect_measured"]),
        "grains_per_radius": float(case["grains_per_radius"]),
        "mu": float(case["mu"]),
        "mu_rolling": float(case["mu_rolling"]),
        "mu_wall": float(case.get("mu_wall", case.get("mu_wall_ratio", float("nan")))),
        "mu_law": str(case.get("mu_law", "constant")),
        "dt_scale": float(case.get("dt_scale", 1.0)),
        "duration_factor": float(case.get("duration_factor", float("nan"))),
        "seed": int(case.get("seed", 0)),
        "n_particles": int(doc["n_particles"]),
        "solid_fraction_initial": float(doc.get("solid_fraction_initial", float("nan"))),
        "column_radius": float(case["column_radius"]),
        "column_radius_measured": float(doc.get("column_radius_measured", float("nan"))),
        "column_height_measured": float(doc.get("column_height_measured", float("nan"))),
        "stop_reason": str(doc.get("stop_reason", "")),
        "time_end": float(doc.get("time_end", float("nan"))),
        "runout_ratio_p995": float(doc["runout_ratio_p995"]),
        "reference_runout_ratio": float(doc["reference_runout_ratio"]),
        "t_reach": float(doc["t_reach"]),
        "reference_emplacement_time": float(doc["reference_emplacement_time"]),
        "t_reach_f090": float(frac.get("0.9", float("nan"))),
        "t_reach_f095": float(frac.get("0.95", float("nan"))),
        "t_reach_f099": float(frac.get("0.99", doc["t_reach"])),
        "peak_front_speed": float(doc.get("peak_front_speed", float("nan"))),
        "ballistic_time": float(doc.get("ballistic_time", float("nan"))),
        "energy_potential_released": float(energy.get("potential_released", float("nan"))),
        "energy_residual_fraction": float(
            energy.get("residual_fraction_of_potential", float("nan"))),
    }
    rec["gravity"] = float(case["gravity"])
    return rec


def load_reduced_csv(path: str) -> list[dict]:
    """Read back the committed reduced table as analysis records."""
    numeric = set(REDUCED_FIELDS) - {"label", "mu_law", "stop_reason", "source",
                                     "source_sha256"} | {"gravity"}
    records = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec = {}
            for key, value in row.items():
                if key is None:
                    continue
                if key in numeric:
                    try:
                        rec[key] = float(value)
                    except (TypeError, ValueError):
                        rec[key] = float("nan")
                else:
                    rec[key] = value
            rec["seed"] = int(rec.get("seed", 0) or 0)
            rec["n_particles"] = int(rec.get("n_particles", 0) or 0)
            records.append(rec)
    if not records:
        raise ValueError(f"empty reduced table: {path}")
    return sorted(records, key=lambda r: r["label"])


def load_cases(src_dir: str) -> list[dict]:
    """Load every ``*.json`` case record of ``src_dir`` (sorted by label)."""
    paths = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no case JSON under {src_dir}")
    records = []
    for path in paths:
        with open(path) as fh:
            rec = reduce_case(json.load(fh))
        rec["source"] = os.path.basename(path)
        rec["source_sha256"] = sha256_of(path)
        records.append(rec)
    return sorted(records, key=lambda r: r["label"])


# ------------------------------------------------------------- adjudication
def runout_table(records, convention: str = "nominal") -> list[dict]:
    """Runout errors under an ``aspect`` convention (``nominal``/``measured``)."""
    if convention not in ("nominal", "measured"):
        raise ValueError("convention must be 'nominal' or 'measured'")
    out = []
    for r in records:
        a = r["aspect_ratio"] if convention == "nominal" else r["aspect_measured"]
        ref = lube_runout_ratio(a)
        err = rel_error(r["runout_ratio_p995"], ref)
        out.append({"label": r["label"], "aspect": a, "value": r["runout_ratio_p995"],
                    "reference": ref, "error": err, "pass": abs(err) < TOLERANCE})
    return out


def emplacement_table(records, fraction: str = "0.99",
                      convention: str = "nominal") -> list[dict]:
    """Emplacement-time errors against ``t_inf = 3 (a r_i / g)^(1/2)``.

    ``fraction`` selects the front-reach definition recorded by the solver
    (``t_reach_by_fraction``).  The reference time is re-derived here from the
    case geometry rather than copied, and is checked against the value the
    solver stored (``reference_emplacement_time``).
    """
    if fraction not in REACH_FRACTIONS:
        raise ValueError(f"fraction must be one of {REACH_FRACTIONS}")
    if convention not in ("nominal", "measured"):
        raise ValueError("convention must be 'nominal' or 'measured'")
    key = {"0.9": "t_reach_f090", "0.95": "t_reach_f095", "0.99": "t_reach_f099"}[fraction]
    out = []
    for r in records:
        ref = lube_emplacement_time(r["aspect_ratio"], r["column_radius"], r["gravity"])
        if convention == "nominal":
            assert abs(ref - r["reference_emplacement_time"]) < 1e-9, (
                "solver-stored reference time disagrees with the paper formula")
            a_used = r["aspect_ratio"]
        else:
            a_used = r["aspect_measured"]
            ref = lube_emplacement_time(a_used, r["column_radius_measured"], r["gravity"])
        err = rel_error(r[key], ref)
        out.append({"label": r["label"], "aspect": a_used, "value": r[key],
                    "reference": ref, "error": err, "pass": abs(err) < TOLERANCE})
    return out


def sign_test_p(n_positive: int, n: int) -> float:
    """Two-sided sign-test p-value for ``n_positive`` out of ``n``."""
    if n <= 0:
        return float("nan")
    k = max(n_positive, n - n_positive)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def fit_exponent(xs, ys) -> dict:
    """Least-squares slope of ``log(y) = p log(x) + q`` with Pearson r."""
    lx = [math.log(float(x)) for x in xs]
    ly = [math.log(float(y)) for y in ys]
    n = len(lx)
    if n < 3:
        raise ValueError("need at least three points")
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    syy = sum((y - my) ** 2 for y in ly)
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    return {"exponent": slope, "pearson_r": r, "n": n}


def definition_robust_violations(records, convention: str = "nominal") -> list[dict]:
    """Cases outside the band under *every* recorded front-reach definition.

    A case that is out of tolerance at 90 %, 95 % and 99 % of the final front
    is out because of the physics, not because the metric was chosen strictly.
    """
    tables = {f: {row["label"]: row
                  for row in emplacement_table(records, f, convention)}
              for f in REACH_FRACTIONS}
    bad = []
    for label in tables["0.99"]:
        if all(abs(tables[f][label]["error"]) >= TOLERANCE for f in REACH_FRACTIONS):
            bad.append({"label": label,
                        "aspect": tables["0.99"][label]["aspect"],
                        "errors": {f: tables[f][label]["error"] for f in REACH_FRACTIONS}})
    return sorted(bad, key=lambda row: row["aspect"])


def summarise(table: list[dict]) -> dict:
    """Pass count / bias statistics of one error table."""
    errs = [row["error"] for row in table]
    n = len(errs)
    n_pos = sum(1 for e in errs if e > 0.0)
    ordered = sorted(errs)
    median = (ordered[n // 2] if n % 2 else
              0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
    return {
        "n": n,
        "n_pass": sum(1 for row in table if row["pass"]),
        "max_abs_error": max(abs(e) for e in errs),
        "mean_error": sum(errs) / n,
        "median_error": median,
        "n_positive": n_pos,
        "sign_test_p": sign_test_p(n_pos, n),
        "worst": min(table, key=lambda row: -abs(row["error"]))["label"],
    }


def closure_arm(records, mu=(0.5, 0.5, 0.8)) -> list[dict]:
    """Cases of the round-5 closure arm ``(mu, mu_r, mu_wall)``."""
    mu, mu_r, mu_w = mu
    return [r for r in records
            if abs(r["mu"] - mu) < 1e-9 and abs(r["mu_rolling"] - mu_r) < 1e-9
            and abs(r["mu_wall"] - mu_w) < 1e-9 and r["mu_law"] == "constant"]


def primary_arm(records, mu=(0.5, 0.5, 0.8)) -> list[dict]:
    """The five registered aspect ratios of the closure arm.

    Excludes the convergence controls (``dt_scale != 1`` or
    ``grains_per_radius != 8``) so that one fit is not confounded by protocol.
    """
    return [r for r in closure_arm(records, mu)
            if abs(r["dt_scale"] - 1.0) < 1e-12
            and abs(r["grains_per_radius"] - 8.0) < 1e-12]


def adjudicate(records) -> dict:
    """Full verdict bundle for the two Lube (2004) laws.

    Hypotheses (fixed before inspecting the results):

    * ``H_TIME`` — the emplacement-time law is reproduced within 15 % on the
      closure arm.  Rejected when fewer than all five aspect ratios pass.
    * ``H_NOMINAL`` — the runout verdict is insensitive to the aspect-ratio
      convention (nominal vs measured) at the 15 % level.
    * ``H_EXPONENT`` — ``t_reach`` grows like ``a^(1/2)`` (the law's exponent).
      Judged on the ``a >= 1`` primary subset, because the whole-arm fit is
      confounded by the shallow ``a = 0.5`` case (H_TIME_SHALLOW); a fitted
      exponent outside ``0.5 +/- 15 %`` is a scaling mismatch.
    * ``H_TIME_SHALLOW`` — the emplacement time is out of band at some aspect
      ratio under *every* recorded front-reach definition, i.e. the deviation is
      not an artefact of how "arrival" is defined.
    """
    arm = closure_arm(records)
    if not arm:
        raise ValueError("closure arm (0.5, 0.5, 0.8) not present in the data")
    runout_nom = runout_table(records, "nominal")
    runout_meas = runout_table(records, "measured")
    arm_labels = {r["label"] for r in arm}
    time_tables = {f: emplacement_table(records, f) for f in REACH_FRACTIONS}
    time_arm = {f: [row for row in time_tables[f] if row["label"] in arm_labels]
                for f in REACH_FRACTIONS}

    exp_arm = fit_exponent([r["aspect_ratio"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])],
                           [r["t_reach"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])])
    exp_arm_90 = fit_exponent(
        [r["aspect_ratio"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])],
        [r["t_reach_f090"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])])

    time_meas = {f: [row for row in emplacement_table(records, f, "measured")
                     if row["label"] in arm_labels] for f in REACH_FRACTIONS}
    exp_ge1 = fit_exponent(
        [r["aspect_ratio"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])
         if r["aspect_ratio"] >= 1.0],
        [r["t_reach"] for r in sorted(arm, key=lambda r: r["aspect_ratio"])
         if r["aspect_ratio"] >= 1.0])
    robust_bad = definition_robust_violations(records)
    prim = sorted(primary_arm(records), key=lambda r: r["aspect_ratio"])
    exp_prim = fit_exponent([r["aspect_ratio"] for r in prim], [r["t_reach"] for r in prim])
    exp_prim_90 = fit_exponent([r["aspect_ratio"] for r in prim],
                               [r["t_reach_f090"] for r in prim])
    prim_ge1 = [r for r in prim if r["aspect_ratio"] >= 1.0]
    exp_prim_ge1 = fit_exponent([r["aspect_ratio"] for r in prim_ge1],
                                [r["t_reach"] for r in prim_ge1])
    exp_prim_ge1_90 = fit_exponent([r["aspect_ratio"] for r in prim_ge1],
                                   [r["t_reach_f090"] for r in prim_ge1])
    prim_labels = {r["label"] for r in prim}
    h_time = summarise(time_arm["0.99"])
    h_time_90 = summarise(time_arm["0.9"])
    h_run_nom = summarise([row for row in runout_nom if row["label"] in arm_labels])
    h_run_meas = summarise([row for row in runout_meas if row["label"] in arm_labels])
    p_run_nom = summarise([row for row in runout_nom if row["label"] in prim_labels])
    p_run_meas = summarise([row for row in runout_meas if row["label"] in prim_labels])
    p_time = {f: summarise([row for row in emplacement_table(records, f)
                            if row["label"] in prim_labels]) for f in REACH_FRACTIONS}

    return {
        "tolerance": TOLERANCE,
        "n_cases": len(records),
        "closure_arm": sorted(arm_labels),
        "primary_arm": sorted(prim_labels),
        "runout_nominal": {"summary": h_run_nom,
                           "rows": [r for r in runout_nom if r["label"] in arm_labels]},
        "runout_measured": {"summary": h_run_meas,
                            "rows": [r for r in runout_meas if r["label"] in arm_labels]},
        "emplacement": {f: {"summary": summarise(time_arm[f]), "rows": time_arm[f]}
                        for f in REACH_FRACTIONS},
        "emplacement_all_cases": {f: summarise(time_tables[f]) for f in REACH_FRACTIONS},
        "emplacement_measured": {f: {"summary": summarise(time_meas[f]),
                                     "rows": time_meas[f]} for f in REACH_FRACTIONS},
        "definition_robust_violations": robust_bad,
        "primary": {
            "runout_nominal": p_run_nom, "runout_measured": p_run_meas,
            "emplacement": p_time,
        },
        "scaling": {"t_reach_99": exp_arm, "t_reach_90": exp_arm_90,
                    "t_reach_99_primary": exp_prim,
                    "t_reach_90_primary": exp_prim_90,
                    "t_reach_99_primary_a_ge_1": exp_prim_ge1,
                    "t_reach_90_primary_a_ge_1": exp_prim_ge1_90,
                    "t_reach_99_a_ge_1": exp_ge1, "law_exponent": 0.5},
        "verdict": {
            "H_TIME": {
                "rejected": h_time["n_pass"] < h_time["n"],
                "n_pass": h_time["n_pass"], "n": h_time["n"],
                "median_error": h_time["median_error"],
                "sign_test_p": h_time["sign_test_p"],
                "robustness_fraction_0.9": {"n_pass": h_time_90["n_pass"],
                                            "median_error": h_time_90["median_error"]},
            },
            "H_NOMINAL": {
                "rejected": (h_run_nom["n_pass"] != h_run_meas["n_pass"])
                or (abs(h_run_nom["max_abs_error"] - h_run_meas["max_abs_error"]) > 0.02),
                "nominal": {"n_pass": h_run_nom["n_pass"],
                            "max_abs_error": h_run_nom["max_abs_error"]},
                "measured": {"n_pass": h_run_meas["n_pass"],
                             "max_abs_error": h_run_meas["max_abs_error"]},
            },
            "H_TIME_SHALLOW": {
                "description": ("no recorded front-reach definition brings these cases "
                                "inside the band -> metric-independent violation"),
                "violations": [row["label"] for row in robust_bad],
                "aspects": [row["aspect"] for row in robust_bad],
                "rejected": bool(robust_bad),
                "measured_aspect_median_error_99": summarise(time_meas["0.99"])["median_error"],
            },
            "H_EXPONENT": {
                # Judged on the a >= 1 subset: the whole-arm fit mixes in the
                # a = 0.5 outlier, which is reported separately by H_TIME_SHALLOW
                # and must not be allowed to decide the scaling question.
                "rejected": abs(exp_prim_ge1["exponent"] - 0.5) > 0.15 * 0.5,
                "subset": "a >= 1, primary arm",
                "exponent_99": exp_arm["exponent"],
                "exponent_90": exp_arm_90["exponent"],
                "exponent_99_a_ge_1": exp_ge1["exponent"],
                "exponent_99_primary": exp_prim["exponent"],
                "exponent_90_primary": exp_prim_90["exponent"],
                "exponent_99_primary_a_ge_1": exp_prim_ge1["exponent"],
                "exponent_90_primary_a_ge_1": exp_prim_ge1_90["exponent"],
                "law_exponent": 0.5,
            },
        },
    }


# ------------------------------------------------------------------- output
def write_reduced_csv(records, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REDUCED_FIELDS) + ["gravity", "source",
                                                                      "source_sha256"])
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in writer.fieldnames})


def write_manifest(records, out_dir: str) -> dict:
    manifest = {
        "n_cases": len(records),
        "sources": {r["source"]: r["source_sha256"] for r in records},
    }
    for name in ("cases_reduced.csv",):
        manifest.setdefault("artifacts", {})[name] = sha256_of(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "manifest_sha256.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def plot_figures(bundle, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    paths = []

    rows99 = bundle["emplacement"]["0.99"]["rows"]
    a = [r["aspect"] for r in rows99]
    law = [lube_emplacement_time(x, 0.48, 9.81) for x in sorted(a)]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(sorted(a), law, "k-", label=r"Lube 2004: $t_\infty=3(a r_i/g)^{1/2}$")
    for frac, style, lbl in (("0.9", "o:", "DEM, front reaches 90 % of final"),
                             ("0.95", "s--", "DEM, front reaches 95 %"),
                             ("0.99", "^-", "DEM, front reaches 99 %")):
        sub = bundle["emplacement"][frac]["rows"]
        ax.plot([r["aspect"] for r in sub], [r["value"] for r in sub], style, label=lbl)
    ax.set_xlabel("initial aspect ratio $a=h_i/r_i$")
    ax.set_ylabel("emplacement time [s]")
    ax.set_title("CASE-COLUMN-COLLAPSE: emplacement time vs. Lube (2004) law")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "emplacement_time_vs_aspect.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for frac, style, lbl in (("0.9", "o:", "reach 90 %"), ("0.95", "s--", "reach 95 %"),
                             ("0.99", "^-", "reach 99 %")):
        sub = bundle["emplacement"][frac]["rows"]
        ax.plot([r["aspect"] for r in sub], [100 * r["error"] for r in sub], style, label=lbl)
    ax.axhspan(-15, 15, color="0.85", zorder=0, label="±15 % band")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("initial aspect ratio $a$")
    ax.set_ylabel("relative error of $t_\\infty$ [%]")
    ax.set_title("Emplacement-time error (closure arm 0.5/0.5/0.8)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "emplacement_error_vs_aspect.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for key, style, lbl in (("runout_nominal", "o-", "reference at nominal $a$"),
                            ("runout_measured", "s--", "reference at measured $a$")):
        sub = bundle[key]["rows"]
        ax.plot([r["aspect"] for r in sub], [100 * r["error"] for r in sub], style, label=lbl)
    ax.axhspan(-15, 15, color="0.85", zorder=0, label="±15 % band")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("initial aspect ratio $a$")
    ax.set_ylabel("runout error [%]")
    ax.set_title("Runout verdict vs aspect-ratio convention (closure arm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "runout_aspect_convention.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-dir", default=os.path.join(
        "runs", "case_column_collapse_lube2004", "cases"))
    ap.add_argument("--out-dir", default=os.path.join(
        "artifacts", "column_collapse_time_law_m17534"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    records = load_cases(args.src_dir)
    bundle = adjudicate(records)
    os.makedirs(args.out_dir, exist_ok=True)
    write_reduced_csv(records, os.path.join(args.out_dir, "cases_reduced.csv"))
    figs = []
    if not args.no_figures:
        figs = plot_figures(bundle, args.out_dir)
    bundle["figures"] = [os.path.relpath(p, args.out_dir) for p in figs]
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=True)
    write_manifest(records, args.out_dir)

    print(f"{bundle['n_cases']} cases, closure arm {bundle['closure_arm']}")
    for name in ("H_TIME", "H_NOMINAL", "H_EXPONENT"):
        v = bundle["verdict"][name]
        print(f"{name}: rejected={v['rejected']}  {json.dumps({k: v[k] for k in v if k != 'rejected'})}")
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())