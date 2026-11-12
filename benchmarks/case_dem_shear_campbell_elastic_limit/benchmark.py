#!/usr/bin/env python3
"""CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT -- reference-vs-got benchmark.

Reads the *shipped* run artifacts (``point_*.json`` + ``scaling.json``), the
reference anchors captured in ``reference_anchors.json`` and emits

* ``benchmark.json`` -- machine-readable leg table (ref value, got value,
  absolute/relative deviation, verdict),
* ``benchmark.md``   -- the same table as it appears in the report.

Nothing here is hand-copied: every ``got`` number is recomputed from the point
JSONs written by ``run_case.py --production``; every ``ref`` number carries the
anchor identifier of the reference statement it comes from.

The legs and their tolerances are declared *before* the numbers are read (see
``LEGS``); the elastic/quasi-static reference value is an exponent n = 0, so the
case's "<15 %" criterion is applied as the absolute band
``|n_got| <= 0.15 * 2 = 0.30`` on the exponent (2.0 being the inertial span the
exponent is normalised against).

Usage
-----
    python benchmarks/case_dem_shear_campbell_elastic_limit/benchmark.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = REPO / "artifacts" / "case_dem_shear_campbell_elastic_limit"

# ---------------------------------------------------------------------------
# Pre-declared legs.  ``ref`` is a key into reference_anchors.json; ``tol`` is
# the acceptance band applied to ``abs_dev`` (or to ``ratio`` when given).
# ---------------------------------------------------------------------------
TOL_EXPONENT_ABS = 0.30      # 15 % of the inertial exponent span n in [0, 2]
TOL_BETA_BAND = (0.0, 1.0)   # elastic-inertial: 0 < beta < 1 (strict)
TOL_KAPPA_FACTOR = 3.0       # order of magnitude => factor-3 band
BETA_CROSS = 0.95   # beta value whose log-kappa crossing defines kappa_c


LEGS = [
    {
        "leg": "R1_elastic_quasi_static_exponent",
        "quantity": "d ln tau / d ln gdot in the elastic-quasi-static limit",
        "ref_anchor": "campbell2002_abstract_elastic_quasi_static",
        "ref": 0.0,
        "tol": TOL_EXPONENT_ABS,
        "tol_kind": "abs",
        "source": "Campbell (2002) abstract: 'a non-inertial quasi-static regime in which"
                  " the stresses scale purely elastically (elastic-quasi-static)'"
                  " => tau independent of gdot => n = 0",
    },
    {
        "leg": "R2_elastic_inertial_branch",
        "quantity": "beta = d ln tau* / d ln kappa over the gated ladder",
        "ref_anchor": "campbell2002_abstract_elastic_inertial",
        "ref": "0 < beta < 1",
        "tol": TOL_BETA_BAND,
        "tol_kind": "band",
        "source": "Campbell (2002) abstract: 'one for which the stresses scale with both"
                  " elastic and inertial properties (elastic-inertial regime)'",
    },
    {
        "leg": "R1b_direct_rate_exponent",
        "quantity": "n = d ln tau / d ln gdot measured by the shear-rate sweep at"
                    " the stiffest swept stiffness (kn = 1e4)",
        "ref_anchor": "campbell2002_abstract_elastic_quasi_static",
        "ref": 0.0,
        "tol": TOL_EXPONENT_ABS,
        "tol_kind": "abs",
        "source": "same anchor as R1: in the elastic-quasi-static regime the stress"
                  " does not depend on the shear rate (n = 0)",
    },
    {
        "leg": "R3_stiffness_free_threshold",
        "quantity": "kappa at which the measured local beta crosses 0.95"
                    " (log-kappa interpolation)",
        "ref_anchor": "arxiv1712_02693_kappa_1e4",
        "ref": 1.0e4,
        "tol": TOL_KAPPA_FACTOR,
        "tol_kind": "factor",
        "source": "arXiv:1712.02693 (restating Campbell's map): 'contact stiffness is"
                  " often assumed to be very large in mu(I) rheology by only focusing on"
                  " the cases that has kappa > 1e4'",
    },
    {
        "leg": "R4_coefficient_of_the_elastic_branch",
        "quantity": "coefficient C in tau* = C * kappa (elastic-quasi-static plateau)",
        "ref_anchor": "campbell2002_tables",
        "ref": None,
        "tol": None,
        "tol_kind": "none",
        "source": "Campbell (2002) JFM 465:261-291 tables/figures: full text CLOSED"
                  " (Unpaywall best_oa_location=null); the coefficient of the elastic"
                  " branch is NOT accessible in this sandbox -> WITHHELD, not"
                  " benchmarked (our value is reported for the record only)",
    },
    {
        "leg": "R5_bagnold_inertial_exponent",
        "quantity": "n = d ln tau / d ln gdot in the inertial limit (kappa << 1)",
        "ref_anchor": "campbell2002_abstract_no_rate_path",
        "ref": 2.0,
        "tol": TOL_EXPONENT_ABS,
        "tol_kind": "abs",
        "source": "Bagnold 1954 / Campbell (2002) inertial branch (tau ~ rho d^2 gdot^2)."
                  " Campbell further states there is no path from inertial to"
                  " quasi-static flow by varying the shear rate at fixed concentration;"
                  " this linear-spring cell can only reach kappa << 1 with contact"
                  " overlaps of order d, so the leg is reported and gated honestly",
    },
]


def _load_points() -> list[dict]:
    pts = [json.loads(f.read_text()) for f in sorted(ART.glob("point_*.json"))]
    if not pts:
        raise SystemExit(f"no point artifacts in {ART}; run run_case.py --production first")
    return pts


def _loglog_slope(xs, ys):
    x = [math.log(v) for v in xs]
    y = [math.log(v) for v in ys]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0.0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx


def _gated(pts):
    return [p for p in pts if p.get("gates", {}).get("all")]


def _ladder(pts):
    """Gated points of the kappa ladder (fixed shear rate 1, nominal N=400)."""
    return sorted([p for p in _gated(pts)
                   if p["case"]["shear_rate"] == 1.0
                   and p["case"]["n_particles"] == 400],
                  key=lambda p: p["kappa"])


def compute(pts: list[dict]) -> dict:
    lat = _ladder(pts)
    out: dict = {"n_points_total": len(pts), "n_points_gated": len(_gated(pts)),
                 "ladder": [{"kn": p["case"]["kn"], "kappa": p["kappa"],
                             "tau_star": p["tau_star"], "tau": p["tau"],
                             "pressure": p["pressure"], "mu_eff": p["mu_eff"],
                             "I": p["I"], "max_overlap": p["max_overlap_ratio"],
                             "coord": p["coordination"]}
                            for p in lat],
                 "legs": []}
    if len(lat) < 3:
        out["status"] = "refused"
        out["reason"] = f"only {len(lat)} gated ladder points; no exponent is extractable"
        return out

    kappa = [p["kappa"] for p in lat]
    tstar = [p["tau_star"] for p in lat]
    beta_global = _loglog_slope(kappa, tstar)
    out["beta_global"] = beta_global
    out["n_global"] = 2.0 * (1.0 - beta_global)
    out["local"] = []
    for a, b in zip(lat[:-1], lat[1:]):
        slope = _loglog_slope([a["kappa"], b["kappa"]], [a["tau_star"], b["tau_star"]])
        out["local"].append({"kappa_low": a["kappa"], "kappa_high": b["kappa"],
                             "beta": slope, "n": 2.0 * (1.0 - slope)})

    # ---- R1: exponent of the two stiffest gated points (elastic end) -------
    top = out["local"][-1]
    n_elastic = top["n"]
    # ---- R2: is the ladder inside the elastic-inertial band? --------------
    # ---- R3: kappa at which the branch becomes rate independent -----------
    kappa_c = None
    segs = [sg for sg in out["local"] if sg["beta"] is not None]
    for a, b in zip(segs[:-1], segs[1:]):
        if a["beta"] < BETA_CROSS <= b["beta"]:
            # log-linear interpolation of the beta crossing in log10(kappa)
            t = (BETA_CROSS - a["beta"]) / (b["beta"] - a["beta"])
            xa = math.log10(math.sqrt(a["kappa_low"] * a["kappa_high"]))
            xb = math.log10(math.sqrt(b["kappa_low"] * b["kappa_high"]))
            kappa_c = 10.0 ** (xa + t * (xb - xa))
            break
    if kappa_c is None and segs and segs[0]["beta"] >= BETA_CROSS:
        kappa_c = segs[0]["kappa_low"]

    # ---- R4: coefficient of the elastic plateau (for the record) ----------
    coeff = None
    if kappa_c is not None:
        stiff = [p for p in lat if p["kappa"] >= kappa_c]
        if len(stiff) >= 2:
            coeff = sum(p["tau_star"] / p["kappa"] for p in stiff) / len(stiff)

    # ---- R5: inertial exponent from the shear-rate sweeps -----------------
    rate_legs = {}
    by_kn: dict[float, list[dict]] = {}
    for p in _gated(pts):
        if p["case"]["n_particles"] != 400:
            continue
        by_kn.setdefault(p["case"]["kn"], []).append(p)
    for kn, sub in sorted(by_kn.items()):
        rates = sorted(p["case"]["shear_rate"] for p in sub)
        if len(set(rates)) < 3:
            continue
        sub = sorted(sub, key=lambda p: p["case"]["shear_rate"])
        n_rate = _loglog_slope([p["case"]["shear_rate"] for p in sub],
                               [p["tau"] for p in sub])
        rate_legs[f"kn{kn:.0e}"] = {"n_rate": n_rate,
                                    "rates": [p["case"]["shear_rate"] for p in sub],
                                    "kappa": [p["kappa"] for p in sub],
                                    "tau": [p["tau"] for p in sub]}
    out["rate_sweeps"] = rate_legs

    got = {
        "R1_elastic_quasi_static_exponent": n_elastic,
        "R1b_direct_rate_exponent": (rate_legs.get("kn1e+04") or {}).get("n_rate"),
        "R2_elastic_inertial_branch": beta_global,
        "R3_stiffness_free_threshold": kappa_c,
        "R4_coefficient_of_the_elastic_branch": coeff,
        "R5_bagnold_inertial_exponent": (rate_legs.get("kn1e+00") or {}).get("n_rate"),
    }

    for leg in LEGS:
        rec = {k: leg[k] for k in ("leg", "quantity", "ref", "ref_anchor", "source")}
        g = got.get(leg["leg"])
        rec["got"] = g
        rec["tol"] = leg["tol"]
        if leg["tol_kind"] == "none" or g is None:
            rec["verdict"] = "WITHHELD" if leg["tol_kind"] == "none" else "NOT_MEASURED"
            rec["note"] = ("reference tables closed; ours reported for the record"
                           if leg["tol_kind"] == "none"
                           else "no gated point supports this leg")
        elif leg["tol_kind"] == "abs":
            dev = g - leg["ref"]
            rec["abs_dev"] = dev
            rec["verdict"] = "PASS" if abs(dev) <= leg["tol"] else "FAIL"
        elif leg["tol_kind"] == "band":
            lo, hi = leg["tol"]
            rec["verdict"] = "PASS" if lo < g < hi else "FAIL"
        elif leg["tol_kind"] == "factor":
            ratio = g / leg["ref"] if g else float("inf")
            rec["ratio"] = ratio
            rec["verdict"] = "PASS" if (1.0 / leg["tol"]) <= ratio <= leg["tol"] else "FAIL"
        out["legs"].append(rec)

    benchable = [r for r in out["legs"] if r["verdict"] in ("PASS", "FAIL")]
    out["status"] = "ok"
    out["legs_passed"] = [r["leg"] for r in benchable if r["verdict"] == "PASS"]
    out["legs_failed"] = [r["leg"] for r in benchable if r["verdict"] == "FAIL"]
    out["overall"] = "PASS" if not out["legs_failed"] else "FAIL"
    return out


def to_markdown(res: dict) -> str:
    lines = ["# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — reference vs got", "",
             f"* points: {res['n_points_gated']}/{res['n_points_total']} gated,",
             f"* beta_global = {res.get('beta_global'):.4f} "
             f"-> n_global = {res.get('n_global'):.4f}", "",
             "| leg | quantity | ref | got | dev | tol | verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in res["legs"]:
        ref = r["ref"] if not isinstance(r["ref"], float) else f"{r['ref']:.3g}"
        got = r["got"]
        got_s = f"{got:.4g}" if isinstance(got, (int, float)) else str(got)
        dev = r.get("abs_dev", r.get("ratio"))
        dev_s = f"{dev:.4g}" if isinstance(dev, (int, float)) else "—"
        tol = r["tol"] if not isinstance(r["tol"], tuple) else f"({r['tol'][0]}, {r['tol'][1]})"
        tol_s = f"{tol:.3g}" if isinstance(tol, (int, float)) else str(tol)
        mark = {"PASS": "✅", "FAIL": "❌", "WITHHELD": "⏸", "NOT_MEASURED": "—"}[r["verdict"]]
        lines.append(f"| {r['leg']} | {r['quantity']} | {ref} | {got_s} | {dev_s} | {tol_s} "
                     f"| {mark} {r['verdict']} |")
    lines += ["", "## ladder", "",
              "| kn | kappa | tau* | tau | P | mu_eff | I | max_ov | Z |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in res["ladder"]:
        lines.append(f"| {p['kn']:.0e} | {p['kappa']:.4g} | {p['tau_star']:.4g} | "
                     f"{p['tau']:.4g} | {p['pressure']:.4g} | {p['mu_eff']:.3f} | "
                     f"{p['I']:.3g} | {p['max_overlap']:.3f} | {p['coord']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    pts = _load_points()
    res = compute(pts)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "benchmark.json").write_text(json.dumps(res, indent=1, default=float))
    (ART / "benchmark.md").write_text(to_markdown(res))
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("status", "beta_global", "n_global", "overall",
                               "legs_passed", "legs_failed")}, indent=1))
    for r in res["legs"]:
        print(f"  {r['leg']:<40s} ref={r['ref']!s:<18s} got={r['got']!s:<12s}"
              f" {r['verdict']}")
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())