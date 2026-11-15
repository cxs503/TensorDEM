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

# Three-valued verdicts.  A criterion that has no evidence must never be
# reported as ``pass``: an ``all()`` over an empty set is vacuously true, which
# is exactly how m20811 reported C_DISORDER = true with no in-run lattice
# control at either aspect (see docs/AUDIT_COLUMN_COLLAPSE_DISORDERED_VERDICT_m21886.md).
PASS = "pass"
FAIL = "fail"
UNEVALUABLE = "unevaluable"

PHI_LO, PHI_HI = 0.52, 0.68
OVERLAP_MAX = 0.02
BAND = 0.15
SPREAD_MAX = 0.05
DISORDER_RATIO_MAX = 0.8
MIN_MEMBERS_FOR_SPREAD = 2

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


BUNDLE_CSV = os.path.join(OUT_DIR, "cases_disordered.csv")

_CSV_FLOAT = ("aspect_nominal", "aspect_measured", "jitter_diameters",
              "packing_fraction", "packing_fraction_poured", "max_overlap_frac",
              "ke_at_rest", "hex_order", "coordination", "relax_time",
              "runout_ratio", "reference", "err", "lattice_hex_order")
_CSV_BOOL = ("at_rest", "admissible")
_CSV_INT = ("seed", "n_particles")
_CSV_SET = set(_CSV_FLOAT) | set(_CSV_BOOL) | set(_CSV_INT)
# ``admissible`` is *derived* by :func:`adjudicate`; a bundle that predates it
# stays readable, and a stale stored value is overwritten rather than trusted.
_CSV_REQUIRED = _CSV_SET - {"admissible"}


def _as_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(text: str) -> bool:
    t = str(text).strip().lower()
    if t in ("true", "1", "yes"):
        return True
    if t in ("false", "0", "no"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def load_bundled_cases(path: str | None = None) -> list[dict]:
    """Read the *committed* bundle CSV and re-derive every derived column.

    The bundle is the only evidence that survives a fresh checkout (the per-case
    JSONs live under ``runs/`` and are not tracked), so the reduction has to be
    able to start from it.  The stored ``reference`` and ``err`` columns are
    **not trusted**: both are recomputed from ``aspect_measured`` and
    ``runout_ratio`` through :func:`lube_2004_runout_ratio`, and a mismatch
    larger than 1e-9 (relative for ``reference``, absolute for ``err``) raises
    :class:`ValueError` instead of being silently analysed.
    """
    path = path or BUNDLE_CSV
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, value in raw.items():
                if key in _CSV_FLOAT:
                    row[key] = _as_float(value)
                elif key in _CSV_BOOL:
                    row[key] = _as_bool(value)
                elif key in _CSV_INT:
                    row[key] = int(float(value))
                else:
                    row[key] = value
            missing = _CSV_REQUIRED - set(row)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            a = row["aspect_measured"]
            if not (a > 0.0 and math.isfinite(a)):
                raise ValueError(f"{path}: aspect_measured={a} is not a length ratio")
            ref = lube_2004_runout_ratio(a)
            if abs(row["reference"] - ref) > 1e-9 * abs(ref):
                raise ValueError(
                    f"{path}: row {row['case']} stores reference="
                    f"{row['reference']!r} but Lube(a_meas={a!r})={ref!r}")
            err = row["runout_ratio"] / ref - 1.0
            if abs(row["err"] - err) > 1e-9:
                raise ValueError(
                    f"{path}: row {row['case']} stores err={row['err']!r} but "
                    f"runout/reference-1={err!r}")
            if not 0.0 < row["packing_fraction"] < 1.0:
                raise ValueError(f"{path}: row {row['case']} packing_fraction="
                                 f"{row['packing_fraction']!r} outside (0, 1)")
            rows.append(row)
    return rows


def bundle_sha256(path: str | None = None) -> str:
    import hashlib
    path = path or BUNDLE_CSV
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _spread(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean())


def _verdict(state: str, reason: str, evidence: int = 0) -> dict:
    if state not in (PASS, FAIL, UNEVALUABLE):
        raise ValueError(f"bad verdict state {state!r}")
    return {"state": state, "reason": reason, "evidence": int(evidence)}


def _combine(verdicts: list[dict], what: str) -> dict:
    """Aggregate per-aspect verdicts: fail > unevaluable > pass.

    A criterion can only be ``pass`` when at least one aspect supplied
    evidence *and* no aspect failed or was unevaluable -- so an empty input can
    never produce ``pass``.
    """
    if not verdicts:
        return _verdict(UNEVALUABLE, f"no aspect supplied evidence for {what}", 0)
    failed = [v for v in verdicts if v["state"] == FAIL]
    if failed:
        return _verdict(FAIL, "; ".join(v["reason"] for v in failed),
                        sum(v["evidence"] for v in verdicts))
    unknown = [v for v in verdicts if v["state"] == UNEVALUABLE]
    if unknown:
        return _verdict(UNEVALUABLE, "; ".join(v["reason"] for v in unknown),
                        sum(v["evidence"] for v in verdicts))
    return _verdict(PASS, f"all {len(verdicts)} aspect(s) pass",
                    sum(v["evidence"] for v in verdicts))


def _member_admissible(row: dict) -> bool:
    """C_EQ applied to one released state (at rest, no residual overlap)."""
    return bool(row["at_rest"] and row["max_overlap_frac"] <= OVERLAP_MAX)


def adjudicate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_cases": 0, "verdicts": {}, "verdict_states": {},
                "incomplete": True}
    for r in rows:
        r["admissible"] = _member_admissible(r)
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
        spread = _spread([m["runout_ratio"] for m in members])
        ratio = (max(m["hex_order"] for m in members) / hex_lat
                 if members and math.isfinite(hex_lat) and hex_lat > 0.0
                 else float("nan"))
        adm = [m for m in members if m["admissible"]]
        per_aspect[a] = {
            "n_members": len(members),
            "n_admissible": len(adm),
            "err_min": min(errs) if errs else float("nan"),
            "err_max": max(errs) if errs else float("nan"),
            "worst_abs_err": abs(worst["err"]) if worst else float("nan"),
            "worst_member": worst["case"] if worst else "",
            "spread": spread,
            "phi": [round(m["packing_fraction"], 4) for m in members],
            "hex_order": [round(m["hex_order"], 4) for m in members],
            "hex_order_lattice_control": hex_lat,
            "hex_order_ratio_max": ratio,
            "lattice_control_runout": (control[0]["runout_ratio"] if control
                                       else float("nan")),
            "lattice_control_err": (control[0]["err"] if control else float("nan")),
            "band_on_admissible": ([bool(abs(m["err"]) <= BAND) for m in adm]
                                   if adm else None),
            "C_BAND": (_verdict(PASS, f"{len(members)} member(s) within "
                                       f"{100 * BAND:.0f} % of Lube(a_meas)",
                                len(members))
                       if members and all(abs(e) <= BAND for e in errs)
                       else _verdict(FAIL, "member(s) outside the band: "
                                     + ", ".join(f"{m['case']}={100 * m['err']:+.1f} %"
                                                 for m in members
                                                 if abs(m["err"]) > BAND),
                                     len(members))
                       if members
                       else _verdict(UNEVALUABLE, "no disordered member", 0)),
            "C_SPREAD": (_verdict(PASS, f"sigma/mean={spread:.4f} <= {SPREAD_MAX}",
                                  len(members))
                         if len(members) >= MIN_MEMBERS_FOR_SPREAD
                         and spread <= SPREAD_MAX
                         else _verdict(FAIL, f"sigma/mean={spread:.4f} > {SPREAD_MAX}",
                                       len(members))
                         if len(members) >= MIN_MEMBERS_FOR_SPREAD
                         else _verdict(UNEVALUABLE,
                                       f"{len(members)} member(s) < "
                                       f"{MIN_MEMBERS_FOR_SPREAD}: spread undefined",
                                       len(members))),
            "C_DISORDER": (_verdict(PASS, f"hex_order/(lattice control)="
                                          f"{ratio:.3f} <= {DISORDER_RATIO_MAX}",
                                    len(members))
                           if math.isfinite(ratio) and ratio <= DISORDER_RATIO_MAX
                           else _verdict(FAIL, f"hex_order/(lattice control)="
                                               f"{ratio:.3f} > {DISORDER_RATIO_MAX}",
                                         len(members))
                           if math.isfinite(ratio)
                           else _verdict(UNEVALUABLE,
                                         "no in-run lattice control at this aspect: "
                                         "no disorder baseline", len(members))),
            # Pre-registered in CRITERIA since m20811 but never evaluated:
            # the wave was supposed to re-run the lattice control in-wave so
            # that C_DISORDER has an in-run baseline.
            "B_LATTICE": (_verdict(PASS, "in-run lattice control present", 1)
                          if control and math.isfinite(hex_lat)
                          else _verdict(FAIL, "no in-run lattice control row",
                                        0)),
        }
    criteria_per_aspect = {}
    for crit in ("C_BAND", "C_SPREAD", "C_DISORDER", "B_LATTICE"):
        tagged = []
        for aspect, agg in per_aspect.items():
            v = dict(agg[crit])
            v["reason"] = f"a={aspect:g}: {v['reason']}"
            tagged.append(v)
        criteria_per_aspect[crit] = tagged
    phi_bad = [r["case"] for r in rows
               if not (math.isfinite(r["packing_fraction"])
                       and PHI_LO <= r["packing_fraction"] <= PHI_HI)]
    phi_ok = all(math.isfinite(r["packing_fraction"]) for r in rows)
    overlap_bad = [r["case"] for r in rows if not r["admissible"]]
    verdicts = {
        "C_PHI": (_verdict(PASS, f"{len(rows)} state(s) in "
                                 f"[{PHI_LO}, {PHI_HI}]", len(rows))
                  if phi_ok and not phi_bad
                  else _verdict(FAIL, "phi outside the bracket: " + ", ".join(phi_bad),
                                len(rows))
                  if phi_ok
                  else _verdict(UNEVALUABLE, "non-finite packing fraction", len(rows))),
        "C_EQ": (_verdict(PASS, f"{len(rows)} state(s) at rest without overlap",
                          len(rows))
                 if not overlap_bad
                 else _verdict(FAIL, "state(s) violating at_rest/overlap: "
                               + ", ".join(overlap_bad), len(rows))),
        "C_DISORDER": _combine(criteria_per_aspect["C_DISORDER"], "C_DISORDER"),
        "C_BAND": _combine(criteria_per_aspect["C_BAND"], "C_BAND"),
        "C_SPREAD": _combine(criteria_per_aspect["C_SPREAD"], "C_SPREAD"),
        "B_LATTICE": _combine(criteria_per_aspect["B_LATTICE"], "B_LATTICE"),
    }
    members = [r for r in rows if r["phase"] == "disordered"]
    coverage = {
        "n_rows": len(rows),
        "n_members": len(members),
        "aspects": sorted(by_aspect),
        "members_per_aspect": {str(a): v["n_members"]
                               for a, v in per_aspect.items()},
        "spread_evaluable": [str(a) for a, v in per_aspect.items()
                             if v["n_members"] >= MIN_MEMBERS_FOR_SPREAD],
        "disorder_evaluable": [str(a) for a, v in per_aspect.items()
                               if math.isfinite(v["hex_order_ratio_max"])],
    }
    return {"n_cases": len(rows),
            "n_admissible": sum(1 for r in rows if r["admissible"]),
            "per_aspect": per_aspect,
            "verdicts": verdicts,
            "verdict_states": {k: v["state"] for k, v in verdicts.items()},
            "evidence_coverage": coverage,
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

    # ---- verdict / evidence coverage -------------------------------------
    states = summary.get("verdict_states", {})
    colour = {PASS: "#2a9d3f", FAIL: "#c62828", UNEVALUABLE: "#9e9e9e"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    aspects = sorted(summary.get("per_aspect", {}))
    counts = [summary["per_aspect"][a]["n_members"] for a in aspects]
    ax0.bar(range(len(aspects)), counts, color="#5c6bc0")
    ax0.axhline(MIN_MEMBERS_FOR_SPREAD, ls="--", c="k", lw=1,
                label=f"spread needs >= {MIN_MEMBERS_FOR_SPREAD} members")
    ax0.set_xticks(range(len(aspects)))
    ax0.set_xticklabels([f"a={a:g}" for a in aspects])
    ax0.set_ylabel("committed members per aspect")
    ax0.set_ylim(0, max(2.0, max(counts) + 1.0))
    ax0.set_title("Evidence coverage of the committed bundle")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(axis="y", alpha=0.3)

    names = ["C_PHI", "C_EQ", "C_DISORDER", "C_BAND", "C_SPREAD"]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.6, len(names) - 0.4)
    ax1.invert_yaxis()
    for i, name in enumerate(names):
        st = states.get(name, UNEVALUABLE)
        ax1.barh(i, 1.0, color=colour[st], alpha=0.85)
        ax1.text(0.02, i, f"{name}: {st}", va="center", ha="left",
                 color="white", fontsize=9, fontweight="bold")
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_title("Pre-registered criteria, three-valued adjudication")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    p = os.path.join(out_dir, "fig_verdict_evidence.png")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)
    return made


SUPERSEDES = (
    "m20811 summary.json reported C_DISORDER=pass from an all() over an empty "
    "set (no in-run lattice control) and C_SPREAD=fail from a spread that is "
    "undefined for a single member; both are re-adjudicated here as "
    "'unevaluable' (m21886)."
)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("runs", "bundle"), default="runs",
                    help="runs = per-case JSONs under runs/ (default); "
                         "bundle = the committed cases_disordered.csv")
    args = ap.parse_args(argv)
    if args.source == "bundle":
        rows = load_bundled_cases()
    else:
        rows = load_cases()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not rows:
        print("no cases found in", RUN_DIR)
        return 1
    summary = adjudicate(rows)          # derives the per-row ``admissible`` flag
    write_csv(rows, os.path.join(OUT_DIR, "cases_disordered.csv"))
    summary["criteria"] = CRITERIA
    summary["source"] = {"mode": args.source,
                         "bundle_csv": os.path.relpath(BUNDLE_CSV, ROOT),
                         "bundle_csv_sha256": bundle_sha256(),
                         "run_dir": (os.path.relpath(RUN_DIR, ROOT)
                                     if args.source == "runs" else "")}
    summary["supersedes"] = SUPERSEDES
    summary["archive_m20533_worst_abs_err"] = archived_baseline()
    figs = figures(rows, summary, OUT_DIR)
    summary["figures"] = [os.path.basename(p) for p in figs]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"verdict_states": summary["verdict_states"],
                      "evidence_coverage": summary["evidence_coverage"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
