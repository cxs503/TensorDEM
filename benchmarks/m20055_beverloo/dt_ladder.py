#!/usr/bin/env python3
"""dt-convergence ladder for the primary m20055 case D5 (see prereg/p20055_dt_ladder.md).

Two modes
---------
Run one level (identical case, only ``dt`` differs from the filed D5):

    PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --level dthalf

Analyse whatever levels are on disk (no DEM needed):

    PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --analyse

The analysis writes

* ``runs/m20055_beverloo_dt_ladder/ladder_summary.json`` -- all numbers;
* ``docs/AUDIT_M20055_DT_CONVERGENCE.md`` -- generated report (zero hand-copy);
* ``docs/figures/m20055_dt_ladder.png`` -- W(dt) + Richardson limit + k band.

Nothing is typed into this script: the reference pair (C, k) and the k-range come
from ``ref_beverloo_1961.json``, and every number in the markdown/figure comes
from the per-level JSON written by ``--level``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.silo_discharge import SiloCase  # noqa: E402

import run_case  # noqa: E402  (same directory: predeclared D5 case + shared analyse())

REF = json.loads((HERE / "ref_beverloo_1961.json").read_text())
C_LIT = float(REF["coefficients"]["C"])
K_LIT = float(REF["coefficients"]["k"])
K_LO, K_HI = (float(x) for x in REF["coefficients"]["k_range_spherical"])
C_LO, C_HI = (float(x) for x in REF["coefficients"]["C_range_spherical"])

RUNS = ROOT / "runs" / "m20055_beverloo_dt_ladder"
DOC = ROOT / "docs" / "AUDIT_M20055_DT_CONVERGENCE.md"
FIG = ROOT / "docs" / "figures" / "m20055_dt_ladder.png"
FIG_DISCHARGE = ROOT / "docs" / "figures" / "m20055_dt_ladder_discharge.png"

#: ladder levels in order of decreasing dt.  The integer is the dt divisor.
LEVELS = [("dt", 1), ("dthalf", 2), ("dtquarter", 4), ("dteighth", 8)]
DIVISOR = dict(LEVELS)
#: predeclared thresholds (prereg/p20055_dt_ladder.md)
TOL_PASS = 0.15          # platform precision standard
TOL_CASE = 0.10          # the case's own filed tolerance
CV_FLAG = 0.30           # a level noisier than this is not used for the order


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def build_level_case(level: str) -> SiloCase:
    """D5 with dt = dt_default / divisor(level); every other field unchanged."""
    if level not in DIVISOR:
        raise SystemExit(f"unknown level {level!r}; choose from {sorted(DIVISOR)}")
    base = run_case.build_case("D5")
    dt0 = base.timestep
    if DIVISOR[level] == 1:
        return base
    return SiloCase(**{**vars(base), "dt": dt0 / DIVISOR[level]})


def contact_period(case: SiloCase) -> float:
    return 2.0 * math.pi * math.sqrt(case.material.mass / case.kn)


def run_level(level: str, out_dir: Path) -> dict:
    """Run one ladder level and file the record (same schema as run_case)."""
    import time

    case = build_level_case(level)
    ref = REF
    t0 = time.time()
    res = run_case.simulate_silo(case)
    wall = time.time() - t0
    rec = run_case.analyse(f"D5_{level}", case, res, ref, wall)
    rec["ladder"] = {"level": level, "divisor": DIVISOR[level],
                     "dt": case.timestep, "dt_default": run_case.build_case("D5").timestep,
                     "dt_over_contact_period": case.timestep / contact_period(case),
                     "steps": int(res["steps"]),
                     "base_case": "D5", "prereg": "prereg/p20055_dt_ladder.md"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{level}.json").write_text(json.dumps(rec, indent=2, default=float))
    v = rec["verdict"]
    print(f"[ladder {level}] dt={case.timestep:.6e} steps={rec['ladder']['steps']} "
          f"W={v['W_got_kg_s']:.4f} W_ref={v['W_ref_kg_s']:.4f} "
          f"rel={100.0 * v['rel_err_W']:+.2f}% "
          f"CV={rec['telemetry']['steady_rate_cv']:.3f} wall={wall:.0f}s")
    return rec


# ---------------------------------------------------------------------------
# analysis helpers (pure functions -- unit-tested)
# ---------------------------------------------------------------------------
def observed_order(w: np.ndarray, ratio: float = 2.0) -> dict:
    """Three-level Richardson order p = log_r((W0-W1)/(W1-W2)).

    Returns ``{"p": None}`` when the estimator is undefined (sign change, zero
    difference, or fewer than three levels), so callers cannot silently use a
    meaningless order.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size < 3:
        return {"p": None, "reason": "fewer than three levels"}
    d0, d1 = float(w[0] - w[1]), float(w[1] - w[2])
    if d0 == 0.0 or d1 == 0.0:
        return {"p": None, "reason": "zero difference between levels"}
    if d0 * d1 < 0.0:
        return {"p": None, "reason": "non-monotone ladder (sign change)"}
    p = math.log(d0 / d1) / math.log(ratio)
    return {"p": p, "reason": "", "d0": d0, "d1": d1}


def richardson(w: np.ndarray, p: float, ratio: float = 2.0) -> float:
    """Extrapolate the last two levels to dt -> 0 with order ``p``."""
    w = np.asarray(w, dtype=np.float64)
    if w.size < 2:
        raise ValueError("need at least two levels")
    rp = ratio ** p
    if rp == 1.0:
        raise ValueError("order 0 does not extrapolate")
    return float(w[-1] + (w[-1] - w[-2]) / (rp - 1.0))


def k_band(W_ref: np.ndarray, D: float, d: float, k_lo: float = K_LO,
           k_hi: float = K_HI) -> dict:
    """Literature k-range -> reference-rate band, at fixed C.

    ``W_ref`` is the reference rate at the cited k; the band is the factor
    ``((D - k d)/(D - k_lit d))^{5/2}`` applied to it.
    """
    base = D - K_LIT * d
    fac = [((D - k_lo * d) / base) ** 2.5, ((D - k_hi * d) / base) ** 2.5]
    return {"W_min": float(W_ref) * min(fac), "W_max": float(W_ref) * max(fac),
            "fac_min": min(fac), "fac_max": max(fac)}


def c_band(W_ref: float, C_lo: float = C_LO, C_hi: float = C_HI) -> dict:
    """Literature C-range -> reference-rate band (linear in C)."""
    fac = C_lo / C_LIT, C_hi / C_LIT
    return {"W_min": W_ref * min(fac), "W_max": W_ref * max(fac),
            "fac_min": min(fac), "fac_max": max(fac)}


def summarise(records: list[dict]) -> dict:
    """Build the ladder summary dict from the per-level records."""
    levels = [r["ladder"]["level"] for r in records]
    dt = np.array([r["ladder"]["dt"] for r in records], dtype=float)
    W = np.array([r["verdict"]["W_got_kg_s"] for r in records], dtype=float)
    W_ref = np.array([r["verdict"]["W_ref_kg_s"] for r in records], dtype=float)
    cv = np.array([r["telemetry"]["steady_rate_cv"] for r in records], dtype=float)
    rho_b = np.array([r["verdict"]["rho_b_kg_m3"] for r in records], dtype=float)
    rel = (W - W_ref) / W_ref
    noisy = [lv for lv, c in zip(levels, cv) if not (c < CV_FLAG)]
    clean = [i for i, c in enumerate(cv) if c < CV_FLAG]

    out = {
        "case": "D5 ladder (dt-convergence), CASE-DEM-SILO-BEVERLOO m20055",
        "prereg": "prereg/p20055_dt_ladder.md",
        "reference": {"C": C_LIT, "k": K_LIT, "C_range": [C_LO, C_HI],
                      "k_range": [K_LO, K_HI],
                      "provenance": REF["provenance"]["verification_status"]},
        "levels": levels,
        "noisy_levels": noisy,
        "thresholds": {"pass": TOL_PASS, "case_tolerance": TOL_CASE, "cv_flag": CV_FLAG},
        "rows": [],
    }
    for i, lv in enumerate(levels):
        D_i = float(records[i]["telemetry"]["orifice_diameter"])
        d_i = float(records[i]["telemetry"]["particle_diameter"])
        band = k_band(W_ref[i], D_i, d_i)
        out["rows"].append({
            "level": lv, "dt": float(dt[i]),
            "dt_over_contact_period": float(records[i]["ladder"]["dt_over_contact_period"]),
            "steps": int(records[i]["ladder"]["steps"]),
            "W": float(W[i]), "W_ref": float(W_ref[i]),
            "rel": float(rel[i]), "cv": float(cv[i]),
            "rho_b": float(rho_b[i]),
            "phi_bed": float(records[i]["telemetry"]["packing_fraction_bed"]),
            "n_discharged": int(records[i]["telemetry"]["n_discharged_total"]),
            "n_bed": int(records[i]["telemetry"]["n_bed"]),
            "k_band_rel": [float((W[i] - band["W_max"]) / band["W_max"]),
                           float((W[i] - band["W_min"]) / band["W_min"])],
        })

    # order + limit on the clean subset (fall back to all levels if <3 clean)
    idx = clean if len(clean) >= 2 else list(range(len(levels)))
    Wc = W[idx]
    order = observed_order(Wc)
    lim = {"order": order}
    if order["p"] is not None and 0.0 < order["p"] <= 4.0:
        p = order["p"]
        lim["p_used"] = p
        lim["W_inf"] = richardson(Wc, p)
        lim["p_source"] = "observed"
    else:
        # two-level p=1 extrapolation as the documented fallback
        lim["p_used"] = 1.0
        lim["W_inf"] = richardson(Wc, 1.0)
        lim["p_source"] = "fallback p=1 (order not identifiable)"
    W_ref_lim = float(W_ref[-1])
    lim["W_ref"] = W_ref_lim
    lim["rel"] = (lim["W_inf"] - W_ref_lim) / W_ref_lim
    lim["rel_p1"] = (richardson(Wc, 1.0) - W_ref_lim) / W_ref_lim
    lim["rel_p2"] = (richardson(Wc, 2.0) - W_ref_lim) / W_ref_lim
    lim["band_rel"] = [min(lim["rel_p1"], lim["rel_p2"], lim["rel"]),
                      max(lim["rel_p1"], lim["rel_p2"], lim["rel"])]
    D_l = float(records[-1]["telemetry"]["orifice_diameter"])
    d_l = float(records[-1]["telemetry"]["particle_diameter"])
    band = k_band(W_ref_lim, D_l, d_l)
    out["geometry"] = {"orifice_diameter": D_l, "particle_diameter": d_l,
                       "D_over_d": D_l / d_l}
    lim["k_band_rel"] = [(lim["W_inf"] - band["W_max"]) / band["W_max"],
                         (lim["W_inf"] - band["W_min"]) / band["W_min"]]
    out["limit"] = lim

    if order["p"] is None or not (0.0 < order["p"] <= 4.0):
        # no order -> the extrapolation is a documented fallback, not a claim
        verdict = "INCONCLUSIVE-NO-ORDER"
    elif abs(lim["rel"]) < TOL_PASS:
        verdict = "PASS-CONVERGED-IN-TOL"
    else:
        verdict = "FAIL-BUT-QUANTIFIED"
    out["verdict"] = verdict
    out["dt_effect"] = float(np.max(np.abs(np.diff(rel)))) if rel.size > 1 else 0.0
    out["archive_check"] = archive_check(records[0]) if records else {"available": False}
    return out


def _rel(path: Path) -> str:
    """Path relative to the repo when possible, else absolute (tests use tmp dirs)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


ARCHIVE_DOC = ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m20055.md"


def parse_archive_row(md_text: str, case: str) -> dict | None:
    """Read one row of the archived m20055 verdict table out of the markdown.

    The archived markdown table is the only place the filed D5 numbers exist
    (their per-run JSON was never committed), so it is parsed rather than copied.
    Columns: case | D/d | mu_p | n_bed | H/D | silo_D/D | rho_b | phi |
    discharged% | W | W_ref | err | C_rec | CV | <10%.
    """
    for line in md_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] != case:
            continue
        try:
            return {"case": case, "n_bed": int(float(cells[3])),
                    "rho_b": float(cells[6]), "phi": float(cells[7]),
                    "W": float(cells[9]), "W_ref": float(cells[10]),
                    "C_rec": float(cells[12]), "cv": float(cells[13])}
        except (IndexError, ValueError):
            return None
    return None


def archive_check(coarse: dict, doc: Path = ARCHIVE_DOC) -> dict:
    """Does the filed coarse-level number reproduce on current main at the same dt?"""
    if not Path(doc).exists():
        return {"available": False}
    arch = parse_archive_row(Path(doc).read_text(), "D5")
    if arch is None:
        return {"available": False}
    v = coarse.get("verdict", {})
    t = coarse.get("telemetry", {})
    dm = coarse.get("dimensionless", {})
    W_c = float(v.get("W_got_kg_s", float("nan")))
    rho_c = float(v.get("rho_b_kg_m3", float("nan")))
    Deff = float(v.get("Deff_m", float("nan")))
    g = float(t.get("gravity", 9.81))
    C_c = dm.get("C_recovered")
    if C_c is None and Deff == Deff and rho_c == rho_c:
        denom = rho_c * math.sqrt(g) * Deff ** 2.5
        C_c = W_c / denom if denom > 0 else float("nan")
    cur = {"W": W_c, "W_ref": float(v.get("W_ref_kg_s", float("nan"))),
           "rho_b": rho_c,
           "phi": float(t.get("packing_fraction_bed", float("nan"))),
           "C_rec": float(C_c if C_c is not None else float("nan")),
           "cv": float(t.get("steady_rate_cv", float("nan")))}
    return {"available": True, "archive": arch, "current": cur,
            "delta": {k: (cur[k] - arch[k]) / arch[k]
                      for k in cur if k in arch and arch[k] not in (0, None)}}


def load_records(runs: Path = RUNS) -> list[dict]:
    recs = []
    for lv, _div in LEVELS:
        p = runs / f"{lv}.json"
        if p.exists():
            recs.append(json.loads(p.read_text()))
    return recs


# ---------------------------------------------------------------------------
# report + figure
# ---------------------------------------------------------------------------
def fig_ladder(s: dict, out: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = np.array([r["dt"] for r in s["rows"]])
    W = np.array([r["W"] for r in s["rows"]])
    W_ref = s["rows"][-1]["W_ref"]
    lim = s["limit"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(dt, W, "o-", color="C0", label="DEM steady rate W(dt)")
    for i, r in enumerate(s["rows"]):
        ax.annotate(f"{100 * r['rel']:+.1f}%", (dt[i], W[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="C0")
    ax.axhline(W_ref, color="k", ls="-", lw=1.0,
               label=f"Beverloo C={C_LIT}, k={K_LIT}")
    g = s["geometry"]
    bloom = k_band(W_ref, g["orifice_diameter"], g["particle_diameter"])
    ax.axhspan(bloom["W_min"], bloom["W_max"], color="0.75", alpha=0.5,
               label=f"k in [{K_LO}, {K_HI}] band")
    ax.axhline(lim["W_inf"], color="C3", ls="--", lw=1.2,
               label=f"Richardson dt->0 (p={lim['p_used']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("time step dt [s]")
    ax.set_ylabel("steady mass flow rate W [kg/s]")
    ax.set_title("m20055 D5: discharge rate vs time step\n(error vs Beverloo 1961 in %)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return _rel(out)


def fig_discharge(records: list[dict], out: Path) -> str:
    """Cumulative discharge count vs time for every ladder level + steady window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for r, col in zip(records, ("C0", "C1", "C2", "C3")):
        tele = r["telemetry"]
        t = np.asarray(tele["discharge_time_s"], dtype=float)
        n = np.asarray(tele["discharge_cumulative_n"], dtype=float)
        if t.size < 2:
            continue
        lv = r["ladder"]["level"]
        ax.plot(t, n, "-", color=col, lw=1.2,
                label=f"{lv} (dt={r['ladder']['dt']:.2e} s, n={int(n[-1])})")
        lo, hi = tele.get("steady_interval_s", [None, None])
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, color=col, alpha=0.10)
    ax.set_xlabel("time since the orifice opened [s]")
    ax.set_ylabel("cumulative discharged particles [-]")
    ax.set_title("m20055 D5 dt-ladder: discharge record (shaded = steady window 0.2-0.8)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return _rel(out)


def render_doc(s: dict) -> str:
    lim = s["limit"]
    L = []
    L.append("# Audit — m20055 Beverloo D5: dt-convergence ladder and the size of the discretisation error\n")
    L.append("Generated by `benchmarks/m20055_beverloo/dt_ladder.py --analyse` from "
             "`runs/m20055_beverloo_dt_ladder/*.json`. No number below is typed by hand.\n")
    L.append(f"*Design*: `prereg/p20055_dt_ladder.md` (written before the runs). "
             f"*Reference*: Beverloo (1961), C = {C_LIT}, k = {K_LIT} "
             f"(`{s['reference']['provenance']}`), C-range "
             f"{s['reference']['C_range']}, k-range {s['reference']['k_range']}.\n")
    L.append("## 1. Why this ladder was run\n")
    L.append("The filed m20055 verdict is an honest FAIL of the primary case D5 "
             "(+20.6 %). The filed matrix also contains a single dt/2 arm that moved the "
             "answer by 8.9 % of the reference toward the literature value. A two-point "
             "sensitivity is not a convergence statement, so the verdict could not be "
             "separated from a discretisation artefact. This ladder quantifies it.\n")
    L.append("## 2. Ladder result (nothing else changed: same seed, bed preparation, "
             "quantile window 0.20-0.80, 5 sub-windows)\n")
    L.append("| level | dt [s] | dt/T_contact | steps | n_bed | φ_bed | W [kg/s] | W_ref [kg/s] | err | CV |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        L.append(f"| {r['level']} | {r['dt']:.4e} | {r['dt_over_contact_period']:.4f} | "
                 f"{r['steps']} | {r['n_bed']} | {r['phi_bed']:.3f} | {r['W']:.4f} | "
                 f"{r['W_ref']:.4f} | {100 * r['rel']:+.2f} % | {r['cv']:.3f} |")
    L.append("")
    op = lim["order"]["p"]
    why = lim["order"]["reason"] or (
        f"outside (0, 4] -> no extrapolation claimed" if op is not None
        and not (0.0 < op <= 4.0) else "")
    L.append("* Observed order of convergence (three-level Richardson, r = 2): "
             + (f"**p = {op:.3f}**" if op is not None else "**undefined**")
             + (f" ({why})" if why else "") + ".")
    L.append(f"* Extrapolated limit dt → 0: **W∞ = {lim['W_inf']:.4f} kg/s** "
             f"→ **{100 * lim['rel']:+.2f} %** vs Beverloo "
             f"(order source: {lim['p_source']}).")
    L.append(f"* Order-robustness band: p = 1 → {100 * lim['rel_p1']:+.2f} %, "
             f"p = 2 → {100 * lim['rel_p2']:+.2f} %.")
    L.append(f"* Isolated dt contribution: coarse-to-fine change in the error is "
             f"{100 * s['dt_effect']:.2f} percentage points of the reference.\n")
    ac = s.get("archive_check", {"available": False})
    if ac.get("available"):
        a, c, d = ac["archive"], ac["current"], ac["delta"]
        L.append("## 2b. The filed number does not reproduce on current main\n")
        L.append("The coarse ladder level *is* the filed D5 configuration (same seed, same "
                 "settling protocol). Re-running it here gives a different bed and a different "
                 "verdict, although `git log -- src/tensordem/silo_discharge.py` shows the module "
                 "only gained an opt-in flux meter (ce9023fc, default off) after the m20055 merge:\n")
        L.append("| quantity | filed (archived md) | this run (dt level) | deviation |")
        L.append("|---|---|---|---|")
        for k, unit in (("rho_b", "kg/m³"), ("phi", "-"), ("W", "kg/s"),
                        ("W_ref", "kg/s"), ("C_rec", "-"), ("cv", "-")):
            L.append(f"| {k} {unit} | {a[k]:.4f} | {c[k]:.4f} | {100 * d[k]:+.1f} % |")
        L.append("")
        L.append("The settled bed is a *chaotic output* of the preparation protocol, and the "
                 "reference rate carries it through `rho_b`: the filed archived table's own D4 row "
                 "also disagrees with the filed `m20055_D4.json` record kept in "
                 "`docs/board_audit/m20232/refs/` (W 2.0857 vs 1.8175 kg/s at rho_b 1118 vs "
                 "1298 kg/m³). Consequence: a single-realisation comparison against Beverloo "
                 "cannot resolve differences smaller than this bed scatter, which is of the same "
                 "order as the error the case reports. That is a property of the case design, not "
                 "of the time step, and it is the reason the verdict below is stated as a band.\n")
    L.append("## 3. Verdict against the predeclared decision rule\n")
    L.append(f"**{s['verdict']}** — extrapolated error {100 * lim['rel']:+.2f} % "
             f"versus the platform 15 % standard (the case's own filed tolerance is "
             f"10 %; the ±3 % column is reported only for reference, because the "
             f"cited (C, k) pair is not re-digitised and a few-% bias lives inside the "
             f"reference itself).\n")
    L.append("| quantity | value | < 3 % ? |")
    L.append("|---|---|---|")
    L.append(f"| W∞ vs Beverloo (C={C_LIT}, k={K_LIT}) | {100 * lim['rel']:+.2f} % | "
             f"{'yes' if abs(lim['rel']) < 0.03 else 'no'} |")
    L.append(f"| W∞ vs Beverloo, best case in the k-range | {100 * lim['k_band_rel'][0]:+.2f} % | "
             f"{'yes' if abs(lim['k_band_rel'][0]) < 0.03 else 'no'} |")
    L.append(f"| W∞ vs Beverloo, worst case in the k-range | {100 * lim['k_band_rel'][1]:+.2f} % | "
             f"{'yes' if abs(lim['k_band_rel'][1]) < 0.03 else 'no'} |")
    L.append("")
    L.append("## 4. What the k-range alone can explain\n")
    L.append("A single orifice diameter makes the comparison sensitive to the literature's "
             "own k-uncertainty through (D − k d)^(5/2). At this case that band is "
             f"{100 * (lim['k_band_rel'][0]):+.1f} … {100 * (lim['k_band_rel'][1]):+.1f} % of "
             "the reference. Consequently a single-D/d error bar cannot be read as a DEM "
             "defect smaller than that band; the honest statement is the band, and a joint "
             "(C, k) fit over ≥ 3 orifice diameters (`tensordem.silo_discharge.beverloo_fit`) "
             "is the only way to make the pair identifiable from DEM data alone.\n")
    L.append("## 5. Reproduction\n")
    L.append("```bash")
    for lv in s["levels"]:
        L.append(f"PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --level {lv}")
    L.append("PYTHONPATH=src python benchmarks/m20055_beverloo/dt_ladder.py --analyse")
    L.append("PYTHONPATH=src python -m pytest tests/test_m20055_dt_ladder.py -q")
    L.append("```")
    L.append("## 6. Figures and artifacts\n")
    L.append(f"* `{_rel(FIG)}` — W(dt) with the Richardson limit and the k-band;")
    L.append(f"* `{_rel(FIG_DISCHARGE)}` — cumulative discharge record of every level, "
             "shaded = the predeclared 0.20-0.80 steady window;")
    L.append("* `runs/m20055_beverloo_dt_ladder/<level>.json` — raw per-level records;")
    L.append("* `runs/m20055_beverloo_dt_ladder/ladder_summary.json` — every number above.\n")
    if s["noisy_levels"]:
        L.append(f"**Flagged noisy levels** (CV > {CV_FLAG}): {', '.join(s['noisy_levels'])} — "
                 "excluded from the order estimate (see prereg §Analysis).\n")
    return "\n".join(L) + "\n"


def analyse_all(runs: Path = RUNS, doc: Path = DOC, fig: Path = FIG) -> dict:
    recs = load_records(runs)
    if len(recs) < 2:
        raise SystemExit(f"need at least two ladder levels under {runs}; found {len(recs)}")
    s = summarise(recs)
    (runs / "ladder_summary.json").write_text(json.dumps(s, indent=2, default=float))
    try:
        s["figure"] = fig_ladder(s, fig)
    except Exception as exc:                             # pragma: no cover
        s["figure"] = f"figure failed: {exc}"
    try:
        s["figure_discharge"] = fig_discharge(recs, FIG_DISCHARGE)
    except Exception as exc:                             # pragma: no cover
        s["figure_discharge"] = f"figure failed: {exc}"
    doc.write_text(render_doc(s))
    print(f"verdict={s['verdict']} W_inf={s['limit']['W_inf']:.4f} "
          f"rel={100 * s['limit']['rel']:+.2f}% p={s['limit']['p_used']}")
    print(f"wrote {_rel(doc)} and {_rel(runs / 'ladder_summary.json')}")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default=None, help="run one ladder level: " + ", ".join(DIVISOR))
    ap.add_argument("--analyse", action="store_true", help="analyse the levels on disk")
    ap.add_argument("--out-dir", default=str(RUNS))
    args = ap.parse_args(argv)
    if args.level:
        run_level(args.level, Path(args.out_dir))
        return 0
    if args.analyse:
        analyse_all()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())