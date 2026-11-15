"""Round 22: re-judge the archived plane-shear points against the **archived**
da Cruz 2005 law, and measure how far the two in-repo Jop triples are from it.

Inputs (all committed, none regenerated here)
---------------------------------------------
``runs/provenance_archive/sources.json``  archived papers + parsed constants
``runs/full_cell/point_*.json``           the four servo-cell points (round 14)

Outputs
-------
``runs/round22/analysis_r22.json``, ``runs/round22/tables_r22.md``

The comparison is deliberately *not* "which triple rescues the acceptance": the
round-13 envelope already answered that (none does).  The question here is the
one the ledger actually asks: does the measurement reproduce the law the cited
paper publishes - and if not, is the shortfall a constant error (fixable by
re-fitting the contact friction) or a shape/estimator error (not fixable that
way).
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reference_archive import (  # noqa: E402
    ATTRIBUTED_TRIPLES, archived_law, fit_jop_to_law, jop_mu, load_manifest,
)

RUNS = os.path.join(HERE, "runs")
OUTDIR = os.path.join(RUNS, "round22")
TOL = 0.10

#: ordinates that the archive actually carries, in reporting order
MU_ORDINATES = ("mu_full_zz_final", "mu_full_final", "mu_wall_final",
                "mu_slab_mid_final", "mu_slab_both_final")
PHI_ORDINATES = ("phi_slab", "phi_bed_steady")


def _points() -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "full_cell", "point_*.json"))):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d["_file"] = os.path.relpath(path, HERE)
        out.append(d)
    out.sort(key=lambda d: d["I_target"])
    return out


def _law_value(name, law, I):
    """Return (value, in_validity) for candidate ``name`` at ``I``."""
    if name == "DACRUZ_LINEAR":
        v = law.mu(I)
        return (law.mu_extrapolated(I) if v is None else v), v is not None
    ms, m2, i0 = ATTRIBUTED_TRIPLES[name]
    return jop_mu(I, ms, m2, i0), True


def evaluate() -> dict:
    manifest = load_manifest()
    law = archived_law(manifest)
    pts = _points()
    if not pts:
        raise SystemExit("no archived points under runs/full_cell/")

    candidates = ("DACRUZ_LINEAR",) + tuple(ATTRIBUTED_TRIPLES)
    report = {
        "kind": "round22-rejudge",
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "tol": TOL,
        "archived_law": {
            "mu_min": law.mu_min, "b": law.b, "nu_max": law.nu_max, "a": law.a,
            "i_saturation": law.i_saturation,
            "source": "runs/provenance_archive/sources.json (da Cruz 2005 Eq. 13/14)",
        },
        "points": [],
        "candidates": {},
        "phi": {},
    }

    for p in pts:
        I = p["I_bulk"]
        row = {
            "I_target": p["I_target"], "I_bulk": I, "file": p["_file"],
            "mu": {k: p.get(k) for k in MU_ORDINATES},
            "phi": {k: p.get(k) for k in PHI_ORDINATES},
            "rel_err": {},
            "law_in_validity": bool(I <= law.i_saturation),
        }
        for name in candidates:
            v, in_valid = _law_value(name, law, I)
            entry = {"law_mu": v, "law_in_validity": in_valid, "rel_err": {}}
            for k in MU_ORDINATES:
                m = p.get(k)
                if m is None:
                    continue
                entry["rel_err"][k] = (m - v) / v
            row["rel_err"][name] = entry
        report["points"].append(row)

    for name in candidates:
        stats = {}
        for k in MU_ORDINATES:
            errs, valid_only = [], []
            for row in report["points"]:
                e = row["rel_err"][name]["rel_err"].get(k)
                if e is None:
                    continue
                errs.append(e)
                if row["law_in_validity"]:
                    valid_only.append(e)
            if not errs:
                continue
            stats[k] = {
                "n": len(errs),
                "rel_err_all": errs,
                "max_abs_rel_err_all": max(abs(e) for e in errs),
                "max_abs_rel_err_in_validity": (max(abs(e) for e in valid_only)
                                                if valid_only else None),
                "n_within_tol_all": sum(abs(e) <= TOL for e in errs),
                "n_in_validity": len(valid_only),
                "n_within_tol_in_validity": sum(abs(e) <= TOL for e in valid_only),
                "passes_all_points_in_validity": bool(valid_only) and
                                                 all(abs(e) <= TOL for e in valid_only),
            }
        report["candidates"][name] = stats

    # ---- phi: archived dilatancy law vs the two archived phi ordinates -------
    phi_stats = {}
    for k in PHI_ORDINATES:
        errs, valid = [], []
        for row in report["points"]:
            v = row["phi"][k]
            if v is None:
                continue
            target = law.phi(row["I_bulk"])
            if target is None:          # I > saturation: law not quoted there
                target = law.phi_extrapolated(row["I_bulk"])
                in_valid = False
            else:
                in_valid = True
            errs.append((v - target) / target)
            if in_valid:
                valid.append((v - target) / target)
        phi_stats[k] = {
            "n": len(errs), "rel_err": errs,
            "max_abs_rel_err_in_validity": max((abs(e) for e in valid), default=None),
            "n_within_tol_in_validity": sum(abs(e) <= TOL for e in valid),
            "n_in_validity": len(valid),
        }
    report["phi"] = {
        "law": "phi(I) = nu_max - a*I (da Cruz 2005 Eq. 13, mu_p = 0.4)",
        "ordinates": phi_stats,
    }

    # ---- monotonicity: the measured mu must rise with I for any mu(I) law ---
    mono = {}
    for k in MU_ORDINATES:
        seq = [(row["I_bulk"], row["mu"][k]) for row in report["points"]
               if row["mu"][k] is not None]
        seq.sort()
        drops = [(seq[i][0], seq[i + 1][0], (seq[i + 1][1] - seq[i][1]) / seq[i][1])
                 for i in range(len(seq) - 1) if seq[i + 1][1] < seq[i][1]]
        mono[k] = {
            "sequence": seq,
            "n_decreasing_steps": len(drops),
            "largest_drop_frac": min((d[2] for d in drops), default=0.0),
            "monotone_increasing": not drops,
        }
    report["monotonicity"] = mono

    # ---- reconciliation: Jop form fitted to the archived law ----------------
    fit = fit_jop_to_law(law)
    report["jop_reconciliation"] = {
        "fit": {k: fit[k] for k in ("mu_s", "mu_2", "I_0", "max_abs_rel_err",
                                    "i_range", "tol", "reproduces_law_within_tol")},
        "distance_of_in_repo_triples": {
            name: {
                "mu_s_rel": (t[0] - fit["mu_s"]) / fit["mu_s"],
                "mu_2_rel": (t[1] - fit["mu_2"]) / fit["mu_2"],
                "I_0_rel": (t[2] - fit["I_0"]) / fit["I_0"],
                "max_abs_rel_err_vs_archived_law": max(
                    abs(jop_mu(I, *t) - (law.mu_min + law.b * I)) /
                    (law.mu_min + law.b * I)
                    for I in fit["I_grid"]),
            } for name, t in ATTRIBUTED_TRIPLES.items()
        },
    }

    # ---- verdict -----------------------------------------------------------
    dc = report["candidates"]["DACRUZ_LINEAR"]["mu_full_zz_final"]
    report["verdict"] = {
        "primary_ordinate": "mu_full_zz_final",
        "archived_law_passes_in_validity":
            report["candidates"]["DACRUZ_LINEAR"]["mu_full_zz_final"]["passes_all_points_in_validity"],
        "n_within_tol_in_validity": dc["n_within_tol_in_validity"],
        "n_in_validity": dc["n"],
        "max_abs_rel_err_in_validity": dc["max_abs_rel_err_in_validity"],
        "any_candidate_passes": any(
            st["passes_all_points_in_validity"]
            for name, cand in report["candidates"].items()
            for st in cand.values()),
        "phi_passes_in_validity": all(
            v["max_abs_rel_err_in_validity"] is not None and
            v["max_abs_rel_err_in_validity"] <= TOL
            for v in phi_stats.values()),
        "mu_monotone": all(v["monotone_increasing"] for v in mono.values()),
    }
    report["verdict"]["ledger"] = (
        "FAIL" if not (report["verdict"]["archived_law_passes_in_validity"] and
                       report["verdict"]["phi_passes_in_validity"]) else "PASS")
    return report


def _pct(x):
    return "n/a" if x is None else "%+.1f%%" % (100 * x)


def markdown(report: dict) -> str:
    law = report["archived_law"]
    L = []
    L.append("### Table A - archived reference law (parsed, not transcribed)\n")
    L.append("| constant | value | source |")
    L.append("|---|---|---|")
    L.append("| `mu_min` | %.3f | da Cruz 2005 Eq. (14) |" % law["mu_min"])
    L.append("| `b` | %.3f | da Cruz 2005 Eq. (14) |" % law["b"])
    L.append("| `nu_max` | %.3f | da Cruz 2005 Eq. (13) |" % law["nu_max"])
    L.append("| `a` | %.3f | da Cruz 2005 Eq. (13) |" % law["a"])
    L.append("| `I_sat` | %.3f | da Cruz 2005 Sec. IV B (fit breaks down above) |" % law["i_saturation"])
    L.append("\n`mu(I) = %.2f + %.2f I` (I <= %.1f), `phi(I) = %.2f - %.2f I`.\n"
             % (law["mu_min"], law["b"], law["i_saturation"], law["nu_max"], law["a"]))

    L.append("### Table B - mu: measured vs each candidate reference\n")
    hdr = ["I_bulk", "in validity", "mu (full_zz)"] + \
          ["%s err" % c for c in report["candidates"]]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for row in report["points"]:
        cells = ["%.4f" % row["I_bulk"], "yes" if row["law_in_validity"] else "no",
                 "%.4f" % row["mu"]["mu_full_zz_final"]]
        for c in report["candidates"]:
            cells.append(_pct(row["rel_err"][c]["rel_err"].get("mu_full_zz_final")))
        L.append("| " + " | ".join(cells) + " |")

    L.append("\n### Table C - pass/fail per candidate and per stress ordinate\n")
    L.append("| candidate | ordinate | n | in-validity max abs err | within 10% (validity) | passes |")
    L.append("|---|---|---|---|---|---|")
    for c, stats in report["candidates"].items():
        for k, st in stats.items():
            L.append("| %s | `%s` | %d | %s | %d/%d | %s |"
                     % (c, k, st["n"], _pct(st["max_abs_rel_err_in_validity"]),
                        st["n_within_tol_in_validity"], st["n_in_validity"],
                        "yes" if st["passes_all_points_in_validity"] else "no"))

    L.append("\n### Table D - phi vs the archived dilatancy law\n")
    L.append("| ordinate | n | in-validity max abs err | within 10% |")
    L.append("|---|---|---|---|")
    for k, st in report["phi"]["ordinates"].items():
        L.append("| `%s` | %d | %s | %d/%d |"
                 % (k, st["n"], _pct(st["max_abs_rel_err_in_validity"]),
                    st["n_within_tol_in_validity"], st["n_in_validity"]))

    L.append("\n### Table E - monotonicity of the measured mu\n")
    L.append("| ordinate | sequence (I, mu) | decreasing steps | largest drop |")
    L.append("|---|---|---|---|")
    for k, st in report["monotonicity"].items():
        seq = ", ".join("(%.3f, %.3f)" % s for s in st["sequence"])
        L.append("| `%s` | %s | %d | %s |"
                 % (k, seq, st["n_decreasing_steps"], _pct(st["largest_drop_frac"])))

    jr = report["jop_reconciliation"]
    L.append("\n### Table F - can a Jop triple ever represent da Cruz?\n")
    f = jr["fit"]
    L.append("Least-squares Jop-form fit to the archived linear law over "
             "I = %g..%g: mu_s = %.3f, mu_2 = %.3f, I_0 = %.3f "
             "(max abs err %.2f%%).\n"
             % (f["i_range"][0], f["i_range"][1], f["mu_s"], f["mu_2"], f["I_0"],
                100 * f["max_abs_rel_err"]))
    L.append("| in-repo triple | mu_s err | mu_2 err | I_0 err | max mu err vs archived law |")
    L.append("|---|---|---|---|---|")
    for name, d in jr["distance_of_in_repo_triples"].items():
        L.append("| %s | %s | %s | %s | %s |"
                 % (name, _pct(d["mu_s_rel"]), _pct(d["mu_2_rel"]), _pct(d["I_0_rel"]),
                    _pct(d["max_abs_rel_err_vs_archived_law"])))
    return "\n".join(L) + "\n"


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    report = evaluate()
    with open(os.path.join(OUTDIR, "analysis_r22.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    md = markdown(report)
    with open(os.path.join(OUTDIR, "tables_r22.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print("verdict:", json.dumps(report["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())