"""Reduce + adjudicate the CASE-COLUMN-COLLAPSE completion wave (m20533).

Inputs (all machine-read, nothing retyped)
------------------------------------------
* ``runs/case_column_collapse_admissible_completion_m20533/cases/*.json`` — this
  mission's wave: lattice control + the four admissible members of m20259 at
  ``a in {1.0, 2.0}`` (the aspects m20259 declared ``incomplete_aspects``).
* ``docs/artifacts/column_collapse_ensemble_m20259/cases_ensemble.csv`` — the
  archived ``a = 0.5`` rows (lattice control + the same four members), read back
  so that the three-aspect picture uses one reduction path for every aspect.

Outputs
-------
``docs/artifacts/column_collapse_completion_m20533/{cases_completion.csv,
summary.json, manifest_sha256.json, fig_band_margin.png,
fig_runout_vs_aspect.png}``

Reference variants (the "aspect bookkeeping" axis)
--------------------------------------------------
The module normalises runout by the *measured* initial extent and reports the
*measured* aspect ratio ``a_meas = h_meas / r_i_meas``, while m19922/m20259
evaluated the Lube (2004) law at the *nominal* ``a``.  Lube et al. define
``a = h_i / r_i`` of the column actually released, so the paper-consistent
reference is the one evaluated at ``a_meas``.  Both are reported per case:

``err_nom``   ``r_inf/r_i`` vs ``lube_2004_runout_ratio(a_nominal)``
              (the archived convention, kept for comparability)
``err_meas``  ``r_inf/r_i`` vs ``lube_2004_runout_ratio(a_meas)``
              (**paper-consistent**, primary criterion)

Pre-registered hypotheses (fixed in the benchmark script before the first run)
-----------------------------------------------------------------------------
``C_ADMISS``  every member starts with ``overlap_pairs == 0``.
``C_BAND``    every member is within 15 % of the Lube law at ``a_meas``.
``C_SPREAD``  1-sigma / mean of ``r_inf/r_i`` across the four admissible members
              <= 5 %, at each aspect ratio.
``C_SCALE``   |shift| from the lattice control to the largest admissible jitter
              <= 5 %, at each aspect ratio.
``B_ASPECT_BOOK``  the nominal-``a`` and measured-``a`` variants are both
              reported; a per-case verdict that flips between them is flagged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.axisym import lube_2004_runout_ratio  # noqa: E402

RUN_COMPLETION = os.path.join(ROOT, "runs",
                              "case_column_collapse_admissible_completion_m20533",
                              "cases")
ARCHIVE_CSV = os.path.join(ROOT, "docs", "artifacts",
                           "column_collapse_ensemble_m20259", "cases_ensemble.csv")
ART_DIR = os.path.join(ROOT, "docs", "artifacts",
                       "column_collapse_completion_m20533")

ASPECTS = (0.5, 1.0, 2.0)
BAND_TOL = 0.15
SPREAD_TOL = 0.05
SCALE_TOL = 0.05

COLUMNS = ["wave", "case", "member", "jitter", "seed", "aspect_nominal",
           "aspect_measured", "n_particles", "stop_reason", "runout_ratio_p995",
           "ref_nominal", "ref_measured", "err_nom", "err_meas", "verdict_nom",
           "verdict_meas", "flip", "overlap_pairs", "max_overlap", "admissible",
           "branch"]


def _branch(a: float) -> str:
    return "linear (a<1.7)" if a < 1.7 else "sqrt (a>1.7)"


def _row(wave: str, case: str, member: str, jitter: float, seed: int,
         a_nom: float, a_meas: float, n: int, stop: str, runout: float,
         overlap_pairs, max_overlap, admissible) -> dict:
    ref_nom = lube_2004_runout_ratio(a_nom)
    ref_meas = lube_2004_runout_ratio(a_meas)
    err_nom = runout / ref_nom - 1.0
    err_meas = runout / ref_meas - 1.0
    ok_nom = abs(err_nom) <= BAND_TOL
    ok_meas = abs(err_meas) <= BAND_TOL
    return {"wave": wave, "case": case, "member": member, "jitter": float(jitter),
            "seed": int(seed), "aspect_nominal": float(a_nom),
            "aspect_measured": float(a_meas), "n_particles": int(n),
            "stop_reason": stop, "runout_ratio_p995": float(runout),
            "ref_nominal": float(ref_nom), "ref_measured": float(ref_meas),
            "err_nom": float(err_nom), "err_meas": float(err_meas),
            "verdict_nom": "pass" if ok_nom else "fail",
            "verdict_meas": "pass" if ok_meas else "fail",
            "flip": bool(ok_nom != ok_meas),
            "overlap_pairs": overlap_pairs, "max_overlap": max_overlap,
            "admissible": admissible, "branch": _branch(a_meas)}


def load_completion() -> list[dict]:
    rows = []
    if not os.path.isdir(RUN_COMPLETION):
        return rows
    for name in sorted(os.listdir(RUN_COMPLETION)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RUN_COMPLETION, name)) as fh:
            raw = json.load(fh)
        ens = raw.get("ensemble", {})
        st = raw.get("admissibility", {})
        rows.append(_row(
            "completion", raw.get("label", name[:-5]), ens.get("member", ""),
            float(ens.get("jitter", 0.0)), int(ens.get("seed", 0)),
            float(ens.get("aspect", raw.get("aspect_nominal", 0.0))),
            float(raw["aspect_measured"]), int(raw["n_particles"]),
            raw.get("stop_reason", ""), float(raw["runout_ratio_p995"]),
            st.get("overlap_pairs"), st.get("max_overlap"),
            st.get("admissible")))
    return rows


def load_archive() -> list[dict]:
    """a = 0.5 rows of m20259 (admissible wave + its lattice control)."""
    if not os.path.exists(ARCHIVE_CSV):
        return []
    rows = []
    with open(ARCHIVE_CSV) as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["aspect_nominal"]) - 0.5) > 1e-9:
                continue
            if float(r["jitter"]) != 0.0 and r["wave"] != "admissible":
                continue          # skip the geometrically inadmissible members
            wave = "archive_lattice" if float(r["jitter"]) == 0.0 else "archive_admissible"
            rows.append(_row(
                wave, r["case"], r["member"], float(r["jitter"]), int(r["seed"]),
                float(r["aspect_nominal"]), float(r["aspect_measured"]),
                int(r["n_particles"]), r["stop_reason"],
                float(r["runout_ratio_p995"]),
                int(r["overlap_pairs"]), float(r["max_overlap"]),
                r["admissible"] == "True"))
    return rows


def _stats(values) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _tri(ok: bool | None, n: int, need: int) -> str:
    if n < need:
        return "insufficient"
    return "pass" if ok else "fail"


def adjudicate_aspect(rows: list[dict], a: float) -> dict:
    """Per-aspect tri-state adjudication (fail dominates insufficient dominates pass)."""
    grp = [r for r in rows if abs(r["aspect_nominal"] - a) < 1e-9]
    members = [r for r in grp if r["jitter"] > 0.0]
    control = [r for r in grp if r["jitter"] == 0.0]
    out: dict = {"n_members": len(members), "n_control": len(control)}
    if not grp:
        return {"status": "insufficient", "reason": "no rows"}

    bad = [r["case"] for r in members if r["overlap_pairs"] != 0]
    out["C_ADMISS"] = {"status": _tri(not bad, len(members), 1),
                       "n": len(members), "required_n": 1,
                       "detail": {"members_with_overlaps": bad}}

    worst_meas = max((abs(r["err_meas"]) for r in members), default=float("nan"))
    worst_nom = max((abs(r["err_nom"]) for r in members), default=float("nan"))
    # signed error of the member that is furthest out of band (for reporting)
    signed = min((r["err_meas"] for r in members), key=abs,
                  default=float("nan")) if members else float("nan")
    if members:
        signed = max(members, key=lambda r: abs(r["err_meas"]))["err_meas"]
    flips = [r["case"] for r in grp if r["flip"]]
    out["C_BAND"] = {"status": _tri(worst_meas <= BAND_TOL, len(members), 1),
                     "n": len(members), "required_n": 1,
                     "detail": {"worst_abs_err_measured_a": worst_meas,
                                "worst_signed_err_measured_a": signed,
                                "worst_abs_err_nominal_a": worst_nom,
                                "variant_verdict_flips": flips,
                                "band_tol": BAND_TOL}}
    out["B_ASPECT_BOOK"] = {
        "status": "pass",
        "detail": {"delta_worst_pp": 100.0 * (worst_meas - worst_nom),
                   "n_flips": len(flips), "flips": flips,
                   "variant": "primary = measured-a (paper-consistent)"}}

    mean, sigma = _stats([r["runout_ratio_p995"] for r in members])
    rel = sigma / mean if math.isfinite(sigma) and mean else float("nan")
    out["C_SPREAD"] = {"status": _tri(math.isfinite(rel) and rel <= SPREAD_TOL,
                                      len(members), 2),
                       "n": len(members), "required_n": 2,
                       "detail": {"mean": mean, "sigma": sigma, "sigma_rel": rel,
                                  "tol": SPREAD_TOL}}
    if control and members:
        c = control[0]["runout_ratio_p995"]
        shift = max(abs(r["runout_ratio_p995"] / c - 1.0) for r in members)
        out["C_SCALE"] = {"status": _tri(shift <= SCALE_TOL, len(members) + 1, 2),
                          "n": len(members) + 1, "required_n": 2,
                          "detail": {"max_abs_shift_vs_lattice": shift,
                                     "lattice_runout": c, "tol": SCALE_TOL}}
    else:
        out["C_SCALE"] = {"status": "insufficient", "n": 0, "required_n": 2,
                          "detail": None}
    # anchor-level statement for this aspect: does *any* row leave the 15 % band?
    out["anchor_within_15pct_all_rows"] = all(
        abs(r["err_meas"]) <= BAND_TOL for r in grp)
    out["max_abs_err_measured_a_all_rows"] = max(
        (abs(r["err_meas"]) for r in grp), default=float("nan"))
    return out


def lattice_reproducibility(rows: list[dict]) -> dict:
    """Deterministic lattice controls: distinct aspects must not collide, and
    every lattice run must have reproduced the archived a = 0.5 value."""
    ctrl = {r["aspect_nominal"]: r for r in rows if r["jitter"] == 0.0}
    out = {"by_aspect": {f"{a:g}": ctrl[a]["runout_ratio_p995"] for a in sorted(ctrl)},
           "n_control": len(ctrl)}
    a05 = [r for r in rows if r["jitter"] == 0.0
           and abs(r["aspect_nominal"] - 0.5) < 1e-9]
    out["a05_from_archive"] = bool(a05)
    return out


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["aspect_nominal"], r["jitter"], r["case"])):
            w.writerow({k: r[k] for k in COLUMNS})


def _figures(rows: list[dict], summ: dict, outdir: str) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    aa = np.linspace(0.2, 3.2, 400)
    law = np.array([lube_2004_runout_ratio(float(a)) for a in aa])

    # fig 1: runout vs measured aspect ratio, both Lube branches
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(aa, law, "k-", lw=1.6, label="Lube et al. (2004) closed form")
    ax.axvline(1.7, color="0.7", ls=":", lw=1.0)
    lattice = [r for r in rows if r["jitter"] == 0.0]
    members = [r for r in rows if r["jitter"] > 0.0]
    ax.plot([r["aspect_measured"] for r in lattice],
            [r["runout_ratio_p995"] for r in lattice], "o", color="tab:blue",
            ms=6, label="lattice control (deterministic)")
    ax.plot([r["aspect_measured"] for r in members],
            [r["runout_ratio_p995"] for r in members], "o", color="tab:red",
            ms=5, mfc="none", label="admissible members (jitter <= 0.005)")
    ax.set_xlabel(r"measured aspect ratio $a = h_i/r_i$")
    ax.set_ylabel(r"final runout ratio $r_\infty/r_i$")
    ax.set_title("CASE-COLUMN-COLLAPSE: admissible ensemble vs Lube (2004)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p1 = os.path.join(outdir, "fig_runout_vs_aspect.png")
    fig.tight_layout(); fig.savefig(p1, dpi=160); plt.close(fig)
    figs.append({"name": "fig_runout_vs_aspect.png",
                 "caption": "Runout ratio against the measured aspect ratio, both "
                            "admissible members and the deterministic lattice control, "
                            "with the two-branch Lube et al. (2004) law."})

    # fig 2: error margin vs the +-15 % band, both reference variants
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(-15, 15, color="tab:green", alpha=0.10, label="anchor band +-15 %")
    for offs, key, col, mk in ((-0.035, "err_meas", "tab:blue", "o"),
                               (0.035, "err_nom", "tab:orange", "s")):
        pts = [r for r in rows]
        ax.plot([r["aspect_measured"] + offs for r in pts],
                [100.0 * r[key] for r in pts], mk, color=col, ms=5,
                label=("vs law at measured a (paper-consistent)" if key == "err_meas"
                       else "vs law at nominal a (archived convention)"))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"measured aspect ratio $a$")
    ax.set_ylabel("runout error [%]")
    ax.set_title("Band margin: reference-variant sensitivity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p2 = os.path.join(outdir, "fig_band_margin.png")
    fig.tight_layout(); fig.savefig(p2, dpi=160); plt.close(fig)
    figs.append({"name": "fig_band_margin.png",
                 "caption": "Per-case runout error against the 15 % anchor band, "
                            "plotted for both reference variants (law evaluated at the "
                            "measured vs the nominal aspect ratio)."})
    del summ
    return figs


def _digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    os.makedirs(ART_DIR, exist_ok=True)
    rows = load_completion() + load_archive()
    if not rows:
        print("no case JSONs found", file=sys.stderr)
        return 2

    verdicts = {f"{a:g}": adjudicate_aspect(rows, a) for a in ASPECTS}
    complete = [a for a in ASPECTS
                if verdicts[f"{a:g}"].get("C_BAND", {}).get("status") != "insufficient"]
    incomplete = [a for a in ASPECTS
                  if verdicts[f"{a:g}"].get("C_BAND", {}).get("status") == "insufficient"]

    all_members = [r for r in rows if r["jitter"] > 0.0]
    worst_meas = max(abs(r["err_meas"]) for r in all_members)
    worst_nom = max(abs(r["err_nom"]) for r in all_members)
    summ = {
        "study": "CASE-COLUMN-COLLAPSE completion of the admissible ensemble wave",
        "anchor": "CASE-COLUMN-COLLAPSE",
        "reference": "Lube et al. 2004 JFM 508:175-199, doi:10.1017/S0022112004009036",
        "tolerance": BAND_TOL,
        "thresholds": {"BAND_TOL": BAND_TOL, "SPREAD_TOL": SPREAD_TOL,
                       "SCALE_TOL": SCALE_TOL},
        "primary_reference_variant": "measured-a (paper-consistent)",
        "aspects": [f"{a:g}" for a in ASPECTS],
        "aspects_complete": [f"{a:g}" for a in complete],
        "incomplete_aspects": [f"{a:g}" for a in incomplete],
        "n_cases": len(rows),
        "n_members": len(all_members),
        "verdicts": verdicts,
        "worst_abs_err_measured_a": worst_meas,
        "worst_abs_err_nominal_a": worst_nom,
        "anchor_max_err_pp": 100.0 * worst_meas,
        "anchor_within_15pct_all_members": bool(worst_meas <= BAND_TOL),
        "lattice_reproducibility": lattice_reproducibility(rows),
        "rows": rows,
    }

    write_csv(rows, os.path.join(ART_DIR, "cases_completion.csv"))
    figs = _figures(rows, summ, ART_DIR)
    summ["figures"] = [f["name"] for f in figs]
    with open(os.path.join(ART_DIR, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    manifest = {name: _digest(os.path.join(ART_DIR, name))
                for name in sorted(os.listdir(ART_DIR))
                if name.endswith((".csv", ".json", ".png"))
                and name != "manifest_sha256.json"}
    with open(os.path.join(ART_DIR, "manifest_sha256.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"cases={len(rows)} members={len(all_members)} "
          f"complete={summ['aspects_complete']} incomplete={summ['incomplete_aspects']}")
    for a in ASPECTS:
        v = verdicts[f"{a:g}"]
        if "C_BAND" not in v:
            print(f"  a={a:g}: {v}")
            continue
        print(f"  a={a:g}: C_BAND={v['C_BAND']['status']} "
              f"worst_meas={100 * v['C_BAND']['detail']['worst_abs_err_measured_a']:+.1f}% "
              f"worst_nom={100 * v['C_BAND']['detail']['worst_abs_err_nominal_a']:+.1f}% "
              f"spread={100 * v['C_SPREAD']['detail']['sigma_rel']:.2f}% "
              f"scale={100 * v['C_SCALE']['detail']['max_abs_shift_vs_lattice']:.2f}% "
              f"flips={len(v['C_BAND']['detail']['variant_verdict_flips'])}")
    print(f"  anchor all-members worst |err| = {100 * worst_meas:.1f}% "
          f"(nominal-a variant {100 * worst_nom:.1f}%)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())