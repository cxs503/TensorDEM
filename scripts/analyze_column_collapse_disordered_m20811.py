"""Reduction + pre-registered verdicts for the disordered-column wave (m20811).

Reads ``runs/case_column_collapse_disordered_m20811/cases/*.json`` (written by
``benchmarks/case_column_collapse_disordered_m20811.py``), writes
``docs/artifacts/column_collapse_disordered_m20811/``:

``cases_disordered.csv``   one row per case (all numbers machine-derived)
``summary.json``           pre-registered criteria -> verdict, plus the
                           comparison against the m20533 lattice-ensemble band
``fig_runout_disorder.png`` runout ratio vs measured aspect ratio, with the
                           Lube (2004) law and both preparation families
``fig_err_band.png``       per-case relative error against the +-15 % band

Criteria (fixed in the benchmark module before the first case ran; repeated in
``CRITERIA`` below so a reader does not have to trust the order of commits).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np

from tensordem.granular_column.axisym import lube_2004_runout_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_DIR = os.path.join(ROOT, "runs", "case_column_collapse_disordered_m20811",
                       "cases")
OUT_DIR = os.path.join(ROOT, "docs", "artifacts",
                       "column_collapse_disordered_m20811")
ARCHIVE = os.path.join(ROOT, "docs", "artifacts",
                       "column_collapse_completion_m20533", "cases_ensemble.csv")

CRITERIA = {
    "C_PHI": "0.52 <= phi <= 0.68 for every released state",
    "C_EQ": "at_rest (KE <= 1e-3 J) and max_overlap <= 0.02 r_p",
    "C_DISORDER": "hex_order <= 0.8 * hex_order(lattice control, same aspect)",
    "C_BAND": "every member within 15 % of the Lube law evaluated at a_meas",
    "C_SPREAD": "sigma/mean of r_inf/r_i across members <= 5 % per aspect",
    "B_LATTICE": "lattice control re-run in this wave (baseline in-run)",
}


def load_cases() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "*.json"))):
        with open(path) as fh:
            res = json.load(fh)
        prep = res["preparation"]
        member = res["member"]
        a = float(res["aspect_measured"])
        ref = lube_2004_runout_ratio(a)
        got = float(res["runout_ratio_p995"])
        rows.append({
            "case": member["tag"],
            "phase": member["phase"],
            "aspect_nominal": float(member["aspect_nominal"]),
            "aspect_measured": a,
            "jitter_diameters": float(member["jitter_diameters"]),
            "seed": int(member["seed"]),
            "n_particles": int(res["n_particles"]),
            "packing_fraction": float(prep["packing_fraction"]),
            "packing_fraction_poured": float(prep.get("packing_fraction_poured",
                                                      float("nan"))),
            "max_overlap_frac": float(prep["max_overlap_frac_of_radius"]),
            "ke_at_rest": float(prep["kinetic_energy_at_rest"]),
            "at_rest": bool(prep["at_rest"]),
            "hex_order": float(prep["hex_order"]),
            "coordination": float(prep["coordination"]),
            "relax_time": float(prep["relax_time"]),
            "runout_ratio": got,
            "reference": ref,
            "err": got / ref - 1.0,
            "stop_reason": res["stop_reason"],
            "lattice_hex_order": (float(res["lattice_disorder"]["hex_order"])
                                  if "lattice_disorder" in res else float("nan")),
            "file": os.path.basename(path),
        })
    return rows


def _spread(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean())


def adjudicate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_cases": 0, "verdicts": {}, "incomplete": True}
    by_aspect = {}
    for r in rows:
        by_aspect.setdefault(r["aspect_nominal"], []).append(r)
    per_aspect = {}
    for a, group in sorted(by_aspect.items()):
        members = [g for g in group if g["phase"] == "disordered"]
        control = [g for g in group if g["phase"] == "lattice_control"]
        hex_lat = control[0]["lattice_hex_order"] if control else float("nan")
        errs = [m["err"] for m in members]
        worst = max(members, key=lambda m: abs(m["err"])) if members else None
        per_aspect[a] = {
            "n_members": len(members),
            "err_min": min(errs) if errs else float("nan"),
            "err_max": max(errs) if errs else float("nan"),
            "worst_abs_err": abs(worst["err"]) if worst else float("nan"),
            "worst_member": worst["case"] if worst else "",
            "spread": _spread([m["runout_ratio"] for m in members]),
            "phi": [round(m["packing_fraction"], 4) for m in members],
            "hex_order": [round(m["hex_order"], 4) for m in members],
            "hex_order_lattice_control": hex_lat,
            "hex_order_ratio_max": (max(m["hex_order"] for m in members) / hex_lat
                                    if members and hex_lat and not math.isnan(hex_lat)
                                    else float("nan")),
            "lattice_control_runout": (control[0]["runout_ratio"] if control
                                       else float("nan")),
            "lattice_control_err": (control[0]["err"] if control else float("nan")),
            "C_BAND": bool(members and all(abs(e) <= 0.15 for e in errs)),
            "C_SPREAD": bool(members and _spread([m["runout_ratio"] for m in members])
                             <= 0.05),
        }
    verdicts = {
        "C_PHI": all(0.52 <= r["packing_fraction"] <= 0.68 for r in rows),
        "C_EQ": all(r["at_rest"] and r["max_overlap_frac"] <= 0.02 for r in rows),
        "C_DISORDER": all(v["hex_order_ratio_max"] <= 0.8
                          for v in per_aspect.values()
                          if not math.isnan(v["hex_order_ratio_max"])),
        "C_BAND": all(v["C_BAND"] for v in per_aspect.values()),
        "C_SPREAD": all(v["C_SPREAD"] for v in per_aspect.values()),
    }
    return {"n_cases": len(rows), "per_aspect": per_aspect, "verdicts": verdicts,
            "incomplete": False}


def archived_baseline() -> dict:
    """Worst admissible-member error of the m20533 lattice ensemble, per aspect."""
    if not os.path.exists(ARCHIVE):
        return {}
    out = {}
    with open(ARCHIVE) as fh:
        for row in csv.DictReader(fh):
            try:
                a = float(row.get("aspect_nominal") or row.get("aspect") or "nan")
                err = float(row.get("err_meas") or row.get("err") or "nan")
            except (TypeError, ValueError):
                continue
            if not math.isfinite(a) or not math.isfinite(err):
                continue
            key = round(a, 6)
            out[key] = max(out.get(key, 0.0), abs(err))
    return out


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def figures(rows: list[dict], summary: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made = []
    aspects = sorted({r["aspect_nominal"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    aa = np.linspace(0.2, max(2.2, max(aspects) + 0.2), 200)
    ax.plot(aa, [lube_2004_runout_ratio(a) for a in aa], "k-", lw=1.4,
            label="Lube et al. (2004)")
    for r in rows:
        style = dict(marker="s" if r["phase"] == "lattice_control" else "o",
                     facecolors="none" if r["phase"] != "lattice_control" else None)
        ax.scatter(r["aspect_measured"], r["runout_ratio"], s=46,
                   label=(r["phase"].replace("_", " ") if r["case"].startswith("m0")
                          else None), **style,
                   edgecolors="C0" if r["phase"] == "lattice_control" else "C1")
    ax.set_xlabel("measured aspect ratio  $a = h_i/r_i$")
    ax.set_ylabel("$r_\\infty / r_i$")
    ax.set_title("Disordered (settled) columns vs Lube (2004)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip([l or "" for l in labels], handles))
    ax.legend([h for k, h in uniq.items() if k], [k for k in uniq if k], fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_runout_disorder.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [f"{r['case']}\na={r['aspect_nominal']:g}" for r in rows]
    errs = [100 * r["err"] for r in rows]
    colors = ["C0" if r["phase"] == "lattice_control" else "C1" for r in rows]
    ax.bar(range(len(rows)), errs, color=colors)
    ax.axhline(15, ls="--", c="k", lw=1)
    ax.axhline(-15, ls="--", c="k", lw=1)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("runout error vs Lube at $a_{meas}$  [%]")
    ax.set_title("Disordered columns: band margin (blue = lattice control)")
    ax.grid(axis="y", alpha=0.3)
    p = os.path.join(out_dir, "fig_err_band.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)
    return made


def main() -> int:
    rows = load_cases()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not rows:
        print("no cases found in", RUN_DIR)
        return 1
    write_csv(rows, os.path.join(OUT_DIR, "cases_disordered.csv"))
    summary = adjudicate(rows)
    summary["criteria"] = CRITERIA
    summary["archive_m20533_worst_abs_err"] = archived_baseline()
    figs = figures(rows, summary, OUT_DIR)
    summary["figures"] = [os.path.basename(p) for p in figs]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"verdicts": summary["verdicts"],
                      "per_aspect": summary["per_aspect"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
