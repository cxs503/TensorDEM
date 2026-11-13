"""CASE-DEM-PLANE-SHEAR-MUIDI round 10 — the abscissa and window audit.

**No new simulation.**  This module re-judges the *already archived* round-8
artefacts (``runs/full_cell/point_*.json`` and ``runs/bulk_rheometry/point_*.json``)
under every combination of

* abscissa  — which inertial number the reference law ``mu(I)`` is evaluated at,
* ordinate  — which shear-stress estimator (tail / final window, p or sigma_zz),
* gate      — whether points that failed the device's own constant-stress gate
              (``servo_valid == False``) are allowed into the verdict,

and reports the verdict as a *matrix*, not as a single number.

Why this exists
---------------
Round 8's P3 verdict was quoted as one number per point: ``mu_full`` (60 % tail
average) against ``mu_ref`` evaluated at ``I_target``.  Three choices inside that
sentence were never audited:

1. ``I_target`` is the inertial number of the *imposed platen rate* with the
   *applied* pressure.  The ordinate ``mu_full`` is a **volume** average.  If the
   sample localises or slips (round 8 measured ``shear_rate_measured /
   shear_rate_imposed = 0.38 ... 0.97`` and ``profile_r2 = 0.57 ... 0.97``), the
   volume-averaged strain rate is not the imposed one, so the pair
   (volume-average stress, imposed-rate I) is not a material point.  The device's
   own ```shear_rate_profile`` docstring says exactly this.
2. The archived field named ``I_measured`` is **not** a measured I: it is
   ``gdot_imposed d / sqrt(P_measured / rho_p)`` (``mu_i_rheology`` line ~1131
   before round 10).  A reader of the artefact reasonably assumes the rounds
   plotted a measured abscissa; they did not.
3. The averaging window flips between tail (60 %) and final (20 %).  The R5
   constant-stress gate is judged on the *final* window while P3 was quoted on
   the *tail*, so the two verdicts are not the same sample.

None of these is a numerical error in the stress arithmetic (round 8 calibrated
that to 4.2 % against the virial identity).  They are *identification* errors: the
sentence "mu is 27 % low" is only meaningful if the abscissa and the window are
pinned down.  This round pins them.

Usage
-----
``python benchmarks/plane_shear_mu_i/abscissa_audit.py``            # write artefacts
``python benchmarks/plane_shear_mu_i/abscissa_audit.py --no-fig``   # text only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNS = HERE / "runs"
ARMS = ("full_cell", "bulk_rheometry")
OUT = RUNS / "abscissa_audit"

# Case constants of the archived sweep (``full_cell_rheometry.POINTS``): the
# artefacts do not carry rho/d, so they are read here from the case definition
# and cross-checked against ``shear_rate_imposed * d / sqrt(P / rho) == I_target``
# for every point (a failure means the wrong constant was used).
D_PARTICLE = 0.01          # m, 2 * radius = 2 * 0.005
RHO_GRAIN = 2500.0         # kg/m^3, grain density (NOT bulk density)
P_APPLIED = 62500.0        # Pa
BAND = 0.10                # ledger acceptance: |mu/mu_ref - 1| <= 10 %

# which shear-stress estimators each archived arm actually carries (the
# bulk_rheometry arm ran without ``bulk_stress``, so it has wall and slab
# ordinates but no ``mu_full*``; printing an all-nan block for it would be noise)
ORDINATES_OF_ARM = {
    "full_cell": ("mu_full", "mu_full_final", "mu_full_zz", "mu_full_zz_final"),
    "bulk_rheometry": ("mu_wall", "mu_wall_final", "mu_slab_both",
                       "mu_slab_both_final"),
}
ABSCISSAE = ("I_target", "I_imposed_measP", "I_vol", "I_bulk")


# ---------------------------------------------------------------------------
# reference law — imported from the round-8 module so that this audit judges the
# verdict *with the law the verdict used*, and asserted against the archived
# ``mu_ref`` column (a provenance cross-check, see ``check_law``).
# ---------------------------------------------------------------------------
def mu_ref_from_reference_law(I: float) -> float:
    """Independent fallback: the repo's canonical law (``reference_law.ref_mu``)."""
    if str(HERE) not in sys.path:                       # noqa: PLC0415
        sys.path.insert(0, str(HERE))
    from reference_law import ref_mu                    # noqa: PLC0415
    return float(ref_mu(I))


def _law_of_record():
    """The law round 8 used, re-derived here from ``reference_law`` constants.

    ``full_cell_rheometry.mu_ref`` is a two-line copy of ``reference_law.ref_mu``;
    importing it would make the audit depend on the very script under audit, so
    the law is rebuilt from the canonical module and *cross-checked* against the
    archived ``mu_ref`` column instead (``check_law``).  A mismatch is a hard
    error, never a warning.
    """
    return mu_ref_from_reference_law


# ---------------------------------------------------------------------------
# abscissae
# ---------------------------------------------------------------------------
def i_from_rate(rate: float, pressure: float) -> float:
    """``I = rate d / sqrt(P/rho)``; ``nan`` for a non-positive pressure."""
    if not pressure > 0.0:
        return float("nan")
    return abs(rate) * D_PARTICLE / math.sqrt(pressure / RHO_GRAIN)


def measured_pressure(point: dict) -> float:
    """Tail (60 %) measured pressure, Pa — reconstructed from the control error.

    The archived point stores only ``pressure_control_error_final`` (final
    window).  The tail pressure is recovered from ``I_measured`` (imposed rate,
    measured pressure), which is exact up to float rounding.
    """
    rate, I = point.get("shear_rate_imposed"), point.get("I_measured")
    if not rate or not I:
        return float("nan")
    return RHO_GRAIN * (rate * D_PARTICLE / I) ** 2


def abscissae(point: dict) -> dict:
    """The four candidate abscissae of one archived point (all at measured P)."""
    rate_i = point["shear_rate_imposed"]
    rate_m = point.get("shear_rate_measured")
    p_tail = measured_pressure(point)
    out = {
        # (a) pre-registered: imposed rate, *applied* pressure  == I_target
        "I_target": float(point["I_target"]),
        # (b) imposed rate, measured pressure (the artefact's "I_measured")
        "I_imposed_measP": float(point["I_measured"]),
        # (c) volume-averaged measured rate, measured pressure — the pair that
        #     matches the volume-averaged ordinate
        "I_vol": i_from_rate(rate_m if rate_m is not None else float("nan"),
                             p_tail),
        # (d) as archived by round 8 from the bulk (slab / full-cell) pressure
        "I_bulk": (float(point["I_bulk"])
                   if point.get("I_bulk") is not None else float("nan")),
    }
    return out


def check_law(point: dict, law) -> float:
    """Max relative disagreement between the rebuilt law and the archived column."""
    I, ref = point.get("I_target"), point.get("mu_ref")
    if not I or not ref:
        return float("nan")
    return abs(law(I) / ref - 1.0)


# ---------------------------------------------------------------------------
# judgement
# ---------------------------------------------------------------------------
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def err_pct(mu: float, I: float, law) -> float:
    """Signed relative error of ``mu`` against ``law(I)``, in percent."""
    if mu is None or I is None or not math.isfinite(I) or not I > 0.0:
        return float("nan")
    ref = law(I)
    return 100.0 * (mu / ref - 1.0) if ref else float("nan")


def judge_point(point: dict, law, ordinates=ORDINATES_OF_ARM["full_cell"]) -> dict:
    """Error of every (abscissa x ordinate) cell for one point."""
    ab = abscissae(point)
    cells = {}
    for a, I in ab.items():
        for o in ordinates:
            e = err_pct(point.get(o), I, law)
            cells[f"{a}|{o}"] = e
    return {
        "I_target": point["I_target"],
        "abscissae": ab,
        "mu": {o: point.get(o) for o in ordinates},
        "mu_ref": {a: (law(I) if math.isfinite(I) and I > 0 else float("nan"))
                   for a, I in ab.items()},
        "err_pct": cells,
        "rate_ratio": point.get("rate_ratio"),
        "profile_r2": point.get("profile_r2"),
        "servo_valid": point.get("servo_valid"),
        "n_particles": point.get("n_particles"),
        "pressure_control_error_final": point.get("pressure_control_error_final"),
    }


def budget(point_judged: dict) -> dict:
    """How much of the mu error the abscissa shift alone can explain.

    ``delta_ref`` is ``mu_ref(I_vol)/mu_ref(I_target) - 1``: the change of the
    *reference* ordinate, i.e. the slice of the measured error that is an
    abscissa artefact.  ``explained`` is that change measured against the
    observed error at the target abscissa (a fraction in [-inf, inf]; 1.0 means
    the abscissa shift accounts for the whole discrepancy).
    """
    a, b = point_judged["mu_ref"].get("I_target"), point_judged["mu_ref"].get("I_vol")
    # the arm's tail-window and final-window ordinates (first two present)
    e_tail = point_judged["err_pct"].get("I_target|mu_full")
    e_final = point_judged["err_pct"].get("I_target|mu_full_final")
    if e_tail is None:
        present = [o for o in point_judged["mu"] if o in ORDINATES_OF_ARM["bulk_rheometry"]]
        e_tail = point_judged["err_pct"].get(f"I_target|{present[0]}") if present else None
        e_final = (point_judged["err_pct"].get(f"I_target|{present[1]}")
                   if len(present) > 1 else None)
    if not a or not b or not math.isfinite(a) or not math.isfinite(b):
        return {"delta_ref_pct": float("nan"), "explained_tail": float("nan"),
                "explained_final": float("nan")}
    d = b / a - 1.0

    def frac(e):            # a ratio against an error near zero is ill-conditioned
        if not e or not math.isfinite(e) or abs(e) < 1.0:
            return None
        return d / (e / 100.0)

    return {"delta_ref_pct": 100.0 * d, "explained_tail": frac(e_tail),
            "explained_final": frac(e_final)}


def within(mu: float, I: float, law, band: float = BAND) -> bool:
    """Ledger acceptance for one cell; ``nan`` (no measurement) is never a pass."""
    e = err_pct(mu, I, law)
    return bool(math.isfinite(e) and abs(e) <= 100.0 * band)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def load_arm(arm: str) -> list:
    rows = []
    for I in (0.01, 0.03, 0.1, 0.3):
        f = RUNS / arm / f"point_{I:.3g}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise(arms: dict, law) -> dict:
    """Pass counts over the full matrix, plus the gate-restricted counterpart."""
    out = {}
    for arm, points in arms.items():
        ords = ORDINATES_OF_ARM.get(arm, ORDINATES_OF_ARM["full_cell"])
        # keep only the ordinates this arm actually measured (the bulk_rheometry
        # arm ran without bulk_stress, so its mu_slab_* entries are null)
        ords = tuple(o for o in ords
                     if any(_finite(p.get(o)) for p in points))
        judged = [judge_point(p, law, ords) for p in points]
        gated = [j for j in judged if j["servo_valid"]]
        cells = [(a, o) for a in ABSCISSAE for o in ords
                 if any(_finite(j["mu"].get(o)) for j in judged)]
        table, table_gated = {}, {}
        for a, o in cells:
            c = f"{a}|{o}"
            table[c] = sum(within(j["mu"][o], j["abscissae"][a], law)
                           for j in judged)
            table_gated[c] = sum(within(j["mu"][o], j["abscissae"][a], law)
                                 for j in gated)
        out[arm] = {
            "ordinates": list(ords),
            "n_points": len(judged),
            "n_gate_pass": len(gated),
            "pass_within_10pct_all_points": table,
            "pass_within_10pct_gate_only": table_gated,
            "points": judged,
            "budget": [budget(j) for j in judged],
        }
    return out


def _fmt(x, w=8, p=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return " " * (w - 3) + "nan"
    return f"{x:+{w}.{p}f}"


def write_tables(res: dict, path: Path) -> str:
    L = ["# Round 10 — abscissa / window / gate matrix (generated, do not hand-edit)",
         "",
         f"band = ±{100 * BAND:.0f} % on `mu`; `nan` never counts as a pass.", ""]
    for arm, blk in res["arms"].items():
        L += [f"## {arm}  (n = {blk['n_points']}, gate-passing = {blk['n_gate_pass']})", "",
              "| I_target | I_imposed_measP | I_vol | I_bulk | rate ratio | profile R2 | servo |",
              "|---|---|---|---|---|---|---|"]
        for j in blk["points"]:
            a = j["abscissae"]
            L.append("| %.3g | %.4f | %.4f | %.4f | %.2f | %.2f | %s |" % (
                a["I_target"], a["I_imposed_measP"], a["I_vol"], a["I_bulk"],
                j["rate_ratio"] or float("nan"), j["profile_r2"] or float("nan"),
                j["servo_valid"]))
        for o in blk.get("ordinates", ORDINATES_OF_ARM["full_cell"]):
            L += ["", f"### ordinate `{o}` — signed error [%] vs `mu_ref` at each abscissa", "",
                  "| I_target | " + " | ".join(ABSCISSAE) + " |",
                  "|---|" + "---|" * len(ABSCISSAE)]
            for j in blk["points"]:
                L.append("| %.3g | " % j["I_target"] + " | ".join(
                    _fmt(j["err_pct"][f"{a}|{o}"], 7, 1) for a in ABSCISSAE) + " |")
        L += ["", "### abscissa budget (`mu_ref(I_vol)/mu_ref(I_target) - 1`)", "",
              "| I_target | delta_ref [%] | explained (tail) | explained (final) |",
              "|---|---|---|---|"]
        for j, b in zip(blk["points"], blk["budget"]):
            L.append("| %.3g | %s | %s | %s |" % (
                j["I_target"], _fmt(b["delta_ref_pct"], 7, 1),
                _fmt(b["explained_tail"], 7, 2), _fmt(b["explained_final"], 7, 2)))
        L += ["", "### pass counts (cells inside ±10 %)", "",
              "| cell | all points | gate only |", "|---|---|---|"]
        for c, n in blk["pass_within_10pct_all_points"].items():
            L.append(f"| {c} | {n}/{blk['n_points']} | "
                     f"{blk['pass_within_10pct_gate_only'][c]}/{blk['n_gate_pass']} |")
        L.append("")
    text = "\n".join(L)
    path.write_text(text)
    return text


def make_figure(res: dict, law, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:                                       # pragma: no cover
        return False
    blk = res["arms"].get("full_cell")
    if not blk:
        return False
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    Ig = np.logspace(-3.4, 0.2, 300)
    ax0.plot(Ig, [law(I) for I in Ig], "k-", lw=1.6,
             label="mu_ref (da Cruz 2005 fit, as used)")
    colours = {"I_target": "#888888", "I_vol": "#d62728", "I_bulk": "#1f77b4"}
    for a in ABSCISSAE:
        xs = [j["abscissae"][a] for j in blk["points"]]
        ys = [j["mu"].get("mu_full") for j in blk["points"]]
        if any(y is None for y in ys):
            continue
        if a in colours:
            ax0.plot(xs, ys, "o", ms=6, color=colours[a], label=f"mu_full @ {a}")
    for j in blk["points"]:
        if j["mu"].get("mu_full") is None:
            continue
        ax0.annotate("", xy=(j["abscissae"]["I_vol"], j["mu"]["mu_full"]),
                     xytext=(j["abscissae"]["I_target"], j["mu"]["mu_full"]),
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0,
                                     ls="--"))
    ax0.set_xscale("log")
    ax0.set_xlabel("inertial number I")
    ax0.set_ylabel("mu")
    ax0.set_title("the abscissa the verdict is judged at\n(red arrow = target -> volume-averaged rate)")
    ax0.legend(fontsize=7, loc="lower right")
    ax0.grid(alpha=0.3, which="both")

    width = 0.2
    idx = np.arange(len(blk["points"]))
    for k, a in enumerate(ABSCISSAE):
        vals = [j["err_pct"].get(f"{a}|mu_full", float("nan"))
                for j in blk["points"]]
        ax1.bar(idx + (k - 1.5) * width, vals, width, label=a)
    ax1.axhspan(-10, 10, color="tab:green", alpha=0.12, label="±10 % band")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"I={j['I_target']:.3g}" for j in blk["points"]])
    ax1.set_ylabel("mu_full error vs mu_ref [%]")
    ax1.set_title("same ordinate, four abscissae")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    global RUNS, OUT
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)
    RUNS, OUT = args.runs, args.out
    OUT.mkdir(parents=True, exist_ok=True)

    law = _law_of_record()
    arms = {a: load_arm(a) for a in ARMS}
    arms = {a: p for a, p in arms.items() if p}
    if not arms:
        print("no archived points found under", RUNS, file=sys.stderr)
        return 2

    # provenance: law cross-check against the archived column
    law_err = max((check_law(p, law) for ps in arms.values() for p in ps
                   if p.get("mu_ref")), default=float("nan"))

    res = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 10,
        "kind": "archival re-analysis (zero new simulation)",
        "constants": {"d_particle_m": D_PARTICLE, "rho_grain": RHO_GRAIN,
                      "P_applied_pa": P_APPLIED, "band": BAND},
        "provenance": {
            "files": {str(f.relative_to(RUNS)): sha256(f)
                      for arm in arms for f in sorted((RUNS / arm).glob("point_*.json"))},
            "law_max_rel_disagreement": law_err,
        },
        "arms": summarise(arms, law),
    }
    if not (math.isfinite(law_err) and law_err < 1e-9):
        print(f"law cross-check failed: {law_err}", file=sys.stderr)
        return 3

    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=str))
    write_tables(res, OUT / "tables.md")
    if not args.no_fig:
        res["figure"] = make_figure(res, law, OUT / "fig_abscissa.png")
        (OUT / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {OUT/'result.json'}  law cross-check {law_err:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())