"""Thin- vs thick-layer branch audit for CASE-DEM-PLANE-SHEAR-MUIDI.

Why this module exists
----------------------
The ledger acceptance compares the device's ``phi(I)`` against da Cruz et al.
2005 Eqs. (13)-(14), as transcribed in :mod:`reference_2d`::

    nu(I) = nu_max - a I,   nu_max = 0.81,  a = 0.3   (mu = 0.4)

Rounds 24-31 use ``nu_max = 0.81`` unconditionally.  The archived paper text in
``runs/provenance_archive/dacruz2005_excerpts.txt`` (key ``thin_layer_packing``)
says that number is the **thick-layer** value and that a thin layer has a
*different* one::

    "The parameters remain the same, except for the solid fraction which is
     smaller than for a thick layer: nu_max = 0.82 instead of 0.84 for mu = 0
     and nu_max = 0.78 instead of 0.81 for mu = 0.4."

The rounds-24..31 cell is ``gap = 0.15`` with ``radius = 5e-3``, i.e.
``gap_eff / d = 14`` free-disk layers - neither obviously "thick" nor "thin",
and the archived excerpt does not state the layer count that separates them.
No previous round adjudicated *which* branch the cell belongs to, so the ``phi``
comparison has an unquantified reference-side systematic.

Two facts from the same excerpt decide how much this can move the verdict:

1. the branch is **specific to the solid fraction** - the friction law is
   stated to be thickness-invariant ("the dilatancy and friction laws are not
   affected"), so the ``mu(I)`` acceptance cannot be rescued or broken by the
   branch choice;
2. for the solid fraction the branch is a constant offset of ``0.03`` in
   ``nu_max``, i.e. a shift of ~3.7 % of ``nu`` that is comparable to the 10 %
   acceptance band at low ``I`` and small at high ``I``.

This module therefore does *not* pick a branch.  It evaluates every measured
point under **both** branches and reports whether the ledger verdict is
branch-sensitive, which is the only honest statement the archived text supports.

Everything here is pure arithmetic over already-archived run files; no contact
physics is touched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "THICK_NU_MAX",
    "THIN_NU_MAX",
    "A_DILATANCY",
    "LAYER_SEPARATION_UNSTATED",
    "BranchVerdict",
    "layer_count",
    "phi_ref_branch",
    "judge_point",
    "judge_points",
    "compare_branches",
    "load_points",
    "render_tables",
]

#: ``nu_max`` for ``mu = 0.4`` as printed for a thick layer (Eq. 13 / Eq. 14 set)
THICK_NU_MAX: float = 0.81
#: the same constant as printed for a thin layer (``thin_layer_packing`` excerpt)
THIN_NU_MAX: float = 0.78
#: ``a`` from Eq. (13); the excerpt gives ``a = 0.31`` for ``mu = 0.4`` as a
#: measured refinement, so both are carried and the larger one is used as the
#: conservative (smaller-``nu_ref``) choice where they differ.
A_DILATANCY: float = 0.3
A_DILATANCY_MEASURED_MU04: float = 0.31

#: the archived excerpt does not state the layer count separating the branches
LAYER_SEPARATION_UNSTATED: str = (
    "the archived excerpt does not give the layer count that separates its "
    "'thin' from its 'thick' case; the branch is therefore carried as an "
    "envelope, not resolved")

BRANCHES: Mapping[str, float] = {"thick": THICK_NU_MAX, "thin": THIN_NU_MAX}


def layer_count(gap_eff: float, diameter: float) -> float:
    """Free-disk layers across the sheared gap: ``gap_eff / d``."""
    gap_eff = float(gap_eff)
    diameter = float(diameter)
    if not (diameter > 0.0):
        raise ValueError("diameter must be positive")
    return gap_eff / diameter


def phi_ref_branch(I: float, *, nu_max: float, a: float = A_DILATANCY) -> float:
    """``nu(I) = nu_max - a I`` for one branch (scalar)."""
    return float(nu_max) - float(a) * float(I)


def judge_point(
    *,
    I: float,
    phi: float,
    tolerance: float = 0.10,
    a: float = A_DILATANCY,
) -> dict:
    """Judge one ``(I, phi)`` point under every branch.

    Returns a dict with a ``branches`` mapping (``thick``/``thin``) holding
    ``nu_ref``, ``err`` (signed relative error, measured vs reference) and
    ``within`` (``|err| <= tolerance``), plus the branch-sensitivity summary.
    """
    I = float(I)
    phi = float(phi)
    out: dict = {"I": I, "phi": phi, "tolerance": tolerance, "a": a,
                 "branches": {}}
    for name, nu_max in BRANCHES.items():
        nu_ref = phi_ref_branch(I, nu_max=nu_max, a=a)
        err = (phi - nu_ref) / nu_ref if nu_ref else float("nan")
        out["branches"][name] = {
            "nu_max": nu_max,
            "nu_ref": nu_ref,
            "err": err,
            "err_pct": 100.0 * err if math.isfinite(err) else float("nan"),
            "within": bool(math.isfinite(err) and abs(err) <= tolerance),
        }
    thick = out["branches"]["thick"]
    thin = out["branches"]["thin"]
    out["branch_sensitive"] = bool(thick["within"] != thin["within"])
    out["err_shift_pct"] = thin["err_pct"] - thick["err_pct"]
    return out


def judge_points(points: Iterable[Mapping[str, float]], *,
                 tolerance: float = 0.10, a: float = A_DILATANCY) -> dict:
    """Judge a whole set of points and summarise the acceptance per branch."""
    rows = [judge_point(I=p["I"], phi=p["phi"], tolerance=tolerance, a=a)
            for p in points]
    summary: dict = {"n_points": len(rows), "tolerance": tolerance, "a": a,
                     "rows": rows}
    if not rows:
        summary["verdict"] = {name: "NO_POINTS" for name in BRANCHES}
        summary["branch_sensitive"] = False
        summary["worst_err_pct"] = {name: float("nan") for name in BRANCHES}
        return summary
    verdict = {}
    worst = {}
    for name in BRANCHES:
        errs = [abs(r["branches"][name]["err"]) for r in rows
                if math.isfinite(r["branches"][name]["err"])]
        worst[name] = 100.0 * max(errs) if errs else float("nan")
        verdict[name] = ("PASS" if errs and max(errs) <= tolerance else
                         "FAIL" if errs else "NO_POINTS")
    summary["verdict"] = verdict
    summary["worst_err_pct"] = worst
    summary["branch_sensitive"] = bool(verdict["thick"] != verdict["thin"])
    return summary


def compare_branches(summary: Mapping[str, object]) -> dict:
    """Turn a :func:`judge_points` summary into the ledger-facing statement."""
    verdict = summary["verdict"]
    sensitive = bool(summary.get("branch_sensitive"))
    if sensitive:
        statement = (
            "BRANCH-SENSITIVE: the phi acceptance flips between the thick "
            "(%s) and thin (%s) nu_max branches, so the phi criterion cannot be "
            "reported as PASS/FAIL from this device without the layer "
            "separation the excerpt does not give."
            % (verdict["thick"], verdict["thin"]))
    else:
        statement = (
            "BRANCH-INSENSITIVE: both nu_max branches give %s, so the "
            "thin/thick ambiguity does not change the phi verdict."
            % verdict["thick"])
    return {
        "phi_verdict_by_branch": dict(verdict),
        "phi_worst_abs_err_pct_by_branch": dict(summary["worst_err_pct"]),
        "branch_sensitive": sensitive,
        "phi_statement": statement,
        "mu_statement": (
            "the friction law is thickness-invariant in the archived text "
            "('the dilatancy and friction laws are not affected'), so no mu "
            "verdict may be attributed to the branch choice"),
        "layer_separation": LAYER_SEPARATION_UNSTATED,
    }


# --------------------------------------------------------------------------- #
# I/O over archived run files
# --------------------------------------------------------------------------- #
def load_points(run_dir: Path, *, pattern: str = "prof_r31_w*_I*_s[0-9].json",
                use_measured_I: bool = True) -> list:
    """Read ``(I, phi)`` pairs out of archived R31 run files.

    ``use_measured_I`` selects the profile-derived abscissa
    (``profile_derived.I_measured``) when present and finite, else the device's
    own ``I_bulk``.  The choice is recorded per point.
    """
    run_dir = Path(run_dir)
    pts = []
    for path in sorted(run_dir.glob(pattern)):
        data = json.loads(path.read_text())
        phi = data.get("phi_gap")
        if phi is None or not math.isfinite(float(phi)):
            continue
        der = data.get("profile_derived") or {}
        I_meas = der.get("I_measured")
        I_bulk = data.get("I_bulk")
        I = None
        source = None
        if use_measured_I and I_meas is not None and math.isfinite(float(I_meas)):
            I, source = float(I_meas), "profile_derived.I_measured"
        elif I_bulk is not None and math.isfinite(float(I_bulk)):
            I, source = float(I_bulk), "I_bulk"
        if I is None:
            continue
        pts.append({"I": I, "phi": float(phi), "file": path.name,
                    "I_source": source,
                    "I_target": data.get("I_target"),
                    "w": data.get("polydispersity")})
    return pts


def render_tables(audit: Mapping[str, object]) -> str:
    """Markdown tables for the audit (numbers interpolated, never hand-typed)."""
    geo = audit["geometry"]
    lines = []
    lines.append("### 表 A1 — 层数与分支（该分支由归档原文给出，未由任何轮次判定）\n")
    lines.append("| 量 | 值 | 依据 |")
    lines.append("|---|---|---|")
    lines.append("| $h_{eff}/d$（自由盘层数） | %.2f | 算例 `gap=%.3f`, $d=%.4f$ |"
                 % (geo["layers"], geo["gap"], geo["diameter"]))
    lines.append("| $\\nu_{max}$ 厚层（$\\mu=0.4$） | %.2f | Eq.13 / `reference_2d` |"
                 % geo["nu_max_thick"])
    lines.append("| $\\nu_{max}$ 薄层（$\\mu=0.4$） | %.2f | 归档原文 `thin_layer_packing` |"
                 % geo["nu_max_thin"])
    lines.append("| 分支分隔层数 | 原文未给出 | %s |" % geo["layer_separation"])
    lines.append("")

    summary = audit["summary"]
    lines.append("### 表 A2 — $\\phi$ 逐点在两个分支下的判定（容差 %s%%）\n"
                 % ("%.0f" % (100.0 * summary["tolerance"])))
    lines.append("| 文件 | $I$ | $I$ 来源 | $\\phi$ | "
                 "$\\nu_{ref}$ 厚 | err % | 厚 | $\\nu_{ref}$ 薄 | err % | 薄 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["rows"]:
        t = r["branches"]["thick"]
        h = r["branches"]["thin"]
        lines.append("| %s | %.5f | %s | %.4f | %.4f | %+.2f | %s | %.4f | %+.2f | %s |"
                     % (r.get("file", "–"), r["I"], r.get("I_source", "–"), r["phi"],
                        t["nu_ref"], t["err_pct"], "PASS" if t["within"] else "FAIL",
                        h["nu_ref"], h["err_pct"], "PASS" if h["within"] else "FAIL"))
    lines.append("")
    comp = audit["comparison"]
    lines.append("### 表 A3 — 分支敏感性结论\n")
    lines.append("| 项 | 结论 |")
    lines.append("|---|---|")
    lines.append("| $\\phi$ 判定（厚/薄） | %s / %s |"
                 % (comp["phi_verdict_by_branch"]["thick"],
                    comp["phi_verdict_by_branch"]["thin"]))
    lines.append("| $\\phi$ 最差绝对误差（厚/薄） | %.2f %% / %.2f %% |"
                 % (comp["phi_worst_abs_err_pct_by_branch"]["thick"],
                    comp["phi_worst_abs_err_pct_by_branch"]["thin"]))
    lines.append("| 分支敏感？ | %s |" % ("是" if comp["branch_sensitive"] else "否"))
    lines.append("| $\\phi$ 结论 | %s |" % comp["phi_statement"])
    lines.append("| $\\mu$ 结论 | %s |" % comp["mu_statement"])
    lines.append("")
    return "\n".join(lines)


def build_audit(points: Sequence[Mapping[str, float]], *, gap_eff: float,
                diameter: float, tolerance: float = 0.10,
                a: float = A_DILATANCY) -> dict:
    """Assemble the full audit record (geometry + per-point + comparison)."""
    summary = judge_points(points, tolerance=tolerance, a=a)
    for row, pt in zip(summary["rows"], points):
        row["file"] = pt.get("file")
        row["I_source"] = pt.get("I_source")
    return {
        "geometry": {
            "gap": float(gap_eff),
            "diameter": float(diameter),
            "layers": layer_count(gap_eff, diameter),
            "nu_max_thick": THICK_NU_MAX,
            "nu_max_thin": THIN_NU_MAX,
            "a": float(a),
            "layer_separation": LAYER_SEPARATION_UNSTATED,
        },
        "summary": summary,
        "comparison": compare_branches(summary),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(Path(__file__).resolve().parent
                                             / "runs" / "plane_shear_2d"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent
                                             / "runs" / "thin_layer_branch"))
    ap.add_argument("--pattern", default="prof_r31_w*_I*_s[0-9].json")
    ap.add_argument("--nominal-I", action="store_true",
                    help="use I_bulk instead of the profile-derived abscissa")
    ap.add_argument("--gap", type=float, default=0.15,
                    help="total wall-to-wall gap (gap_eff = gap - diameter)")
    ap.add_argument("--diameter", type=float, default=1.0e-2)
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = load_points(Path(args.run_dir), pattern=args.pattern,
                      use_measured_I=not args.nominal_I)
    audit = build_audit(pts, gap_eff=args.gap - args.diameter,
                        diameter=args.diameter, tolerance=args.tolerance)
    (out / "thin_layer_branch_audit.json").write_text(
        json.dumps(audit, indent=1))
    (out / "thin_layer_branch_tables.md").write_text(render_tables(audit))
    comp = audit["comparison"]
    print("layers=%.2f  n_points=%d" % (audit["geometry"]["layers"],
                                        audit["summary"]["n_points"]))
    print("phi verdict thick/thin: %s / %s"
          % (comp["phi_verdict_by_branch"]["thick"],
             comp["phi_verdict_by_branch"]["thin"]))
    print("worst |err| thick/thin: %.2f %% / %.2f %%"
          % (comp["phi_worst_abs_err_pct_by_branch"]["thick"],
             comp["phi_worst_abs_err_pct_by_branch"]["thin"]))
    print("branch_sensitive:", comp["branch_sensitive"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())