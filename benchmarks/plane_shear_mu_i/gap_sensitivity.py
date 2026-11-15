"""CASE-DEM-PLANE-SHEAR-MUIDI round 21: finite-gap (wall-layer) sensitivity.

Motivation
----------
Twenty rounds judged the ledger acceptance (mu(I) and phi(I) within 10 % of the
quoted da Cruz / GDR-MiDi reference over I = 1e-3 ... 0.3) on a bed that is
4-9 particle diameters thick.  Every one of those rounds changed the *lateral*
size of the cell to raise N (box_width / box_depth) and changed the gap only
*jointly* with the depth (``run_stationarity.ARMS``: ``thick`` = depth 0.08 +
gap 0.09), so **the shear-direction finite-size effect was never isolated**.
The repeated finding of that corpus is a one-sided bias: mu is 17-36 % *below*
the reference law and phi is 5-14 % *below* it (rounds 3, 5, 11, 16, 20).

Both deficits have the same geometric signature.  A flat wall cannot host a
particle centre closer than r, so a layer of thickness ~d/2 next to each platen
carries no particle centres; in a gap of H = n_gap * d the container-averaged
solid fraction is therefore

    phi(H) = phi_inf * (H - 2*delta) / H ,      delta ~ d/2

i.e. ``phi`` is biased *low* by 2*delta/H -- 25 % at H = 4d, 8 % at H = 12d.
The same dead layer is not sheared at the bulk rate, so the gap-averaged shear
rate used for the abscissa is biased low and mu, read at the platen, mixes two
boundary layers whose combined thickness is a large fraction of the gap
(round 8 already flagged this qualitatively, never measured it).

This round measures the trend and fits the correction.  Nothing about the
contact law, the device or the reference constants changes: the *only* varied
axis is the number of particle diameters between the platens, at fixed lateral
extent (W = 0.12, D = 0.05) and fixed P / kn / mu / seed.

Protocol parity with rounds 5/16/20: ``ServoShearCase``, smooth walls,
P = 6.25e4 Pa, kn = 6.25e4 N/m, mu = 0.5, mu_roll = 0.05, e = 0.3, r = 0.005 m,
rho = 2500 kg/m^3, seed 5279.  ``max_recover_strain`` (round 17) bounds the
sheared strain so a point cannot be quoted beyond its own budget.

Usage
-----
    python gap_sensitivity.py --run --I 0.01 --fill 0.04     # one shard
    python gap_sensitivity.py --assemble                      # fit + figures

Output: ``runs/gap_sensitivity/gap<I>__fill<H>.json`` per point,
``runs/gap_sensitivity/analysis.json`` + figures from ``--assemble``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402
from reference_law import (I_0, PHI_REF_STATUS, MU_REF_STATUS,  # noqa: E402
                           ref_mu, ref_phi, reference)

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "gap_sensitivity"

# --- protocol (rounds 5/16/20 parity, gap is the varied axis) --------------
P = 6.25e4            # Pa
KN = 6.25e4           # N/m  (P/kn = 1)
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E_REST = 0.3
R = 0.005             # m  -> d = 0.01 m
RHO = 2500.0
W = 0.12              # m, lateral (x, periodic)
D = 0.05              # m, lateral (y, smooth walls) -- held FIXED
SEED = 5279
STRAIN_TARGET = 4.0   # round 16: phi converged by 2, mu needs more
FILLS = (0.04, 0.08, 0.14, 0.22)   # nominal fill -> measured gap 3..15 d
I_TARGETS = (0.01, 0.03, 0.1)


# ---------------------------------------------------------------- correction
def fit_wall_correction(points: list[dict], key: str) -> dict:
    """Fit ``v(H) = v_inf * (1 - 2*delta/H)`` to ``(gap_d, value)`` points.

    ``points`` rows must carry ``gap_d`` (measured gap in particle diameters)
    and ``key``.  The model is linear in ``1/H``:

        v = v_inf + b/H ,   v_inf = intercept,   delta = -b / (2 * v_inf)

    so an ordinary least-squares line in ``1/H`` identifies both the
    unbounded-gap value and the dead-layer thickness ``delta`` (in metres).
    Fewer than two usable points, a degenerate abscissa, or a non-positive
    ``v_inf`` yields ``identified: False`` rather than a quoted number: the
    correction is only a measurement when it is over-determined.
    """
    rows = [(float(p["gap_d"]), float(p[key])) for p in points
            if p.get(key) is not None and math.isfinite(float(p[key]))
            and p.get("gap_d")]
    out = {"key": key, "n_points": len(rows), "identified": False,
           "v_inf": float("nan"), "slope_per_d": float("nan"),
           "delta_over_d": float("nan"), "delta_m": float("nan"),
           "r2": float("nan")}
    if len(rows) < 2:
        return out
    h = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    x = 1.0 / h
    if np.ptp(x) <= 0.0:
        return out
    # v = a + b*x  with b = -2*delta_d*v_inf  (delta_d = delta in diameters)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ np.array([a, b])
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    out.update({
        "v_inf": float(a),
        "slope_per_d": float(b),
        "delta_over_d": float(-b / (2.0 * a)) if a != 0.0 else float("nan"),
        "delta_m": float(-b / (2.0 * a)) * (2.0 * R) if a != 0.0 else float("nan"),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "identified": bool(a > 0.0 and abs(a) > 1e-12),
    })
    return out


def residual_table(points: list[dict], key: str, fit: dict) -> list[dict]:
    """Per-point error of ``key`` against the reference *and* against the fit."""
    rows = []
    for p in points:
        if p.get(key) is None:
            continue
        i = p.get("I_abscissa")
        ref = None if i is None else (
            ref_phi(i) if key.startswith("phi") else ref_mu(i))
        rows.append({
            "gap_d": p["gap_d"],
            "I": i,
            "value": float(p[key]),
            "ref": None if ref is None else float(ref),
            "err_pct": (None if not ref else
                        100.0 * (float(p[key]) / float(ref) - 1.0)),
            "fit_inf": fit.get("v_inf"),
        })
    return rows


# ---------------------------------------------------------------------- run
def run_point(I: float, fill: float, *, strain: float = STRAIN_TARGET,
              settle_steps: int = 4000, sample_stride: int = 20,
              wall_mode: str = "smooth", box_width: float = W,
              box_depth: float = D) -> dict:
    """Run one gap point and reduce it to the fields the fit needs."""
    case = ServoShearCase(
        inertial_number=I, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E_REST, kn=KN, normal_stress=P, shear_strain=strain,
        settle_steps=settle_steps, min_shear_steps=1,
        # round 17 ceiling: the recovery gate may shear at most one further
        # ``strain`` unit past the request, so the worst case is 2x.  The
        # delivered strain and the budget flag are archived per point, so a
        # point that spends the allowance is visible in the fit table.
        max_recover_strain=strain,
        sample_stride=sample_stride, packing_gap=0.02, jitter=0.01, seed=SEED,
        radius=R, density=RHO, box_width=box_width, box_depth=box_depth,
        fill_height=fill, wall_mode=wall_mode,
    )
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wall = time.perf_counter() - t0
    series = res["series"]
    n = len(series)
    n_tail = max(1, int(0.4 * n))
    tail = series[-n_tail:]

    def _tail(k: str) -> float:
        vals = [s.get(k) for s in tail if s.get(k) is not None]
        vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    mu_wall = _tail("mu")
    mu_bulk = _tail("bulk_mu")
    mu_full = _tail("full_mu")
    p_meas = float(res.get("normal_stress_measured_pa") or float("nan"))
    plate_h = float(res.get("plate_height_steady") or float("nan"))
    d = 2.0 * R
    # block drift of the reading over the last half of the shear phase
    half = series[n // 2:]
    blocks = []
    if len(half) >= 4:
        cut = len(half) // 2
        for chunk in (half[:cut], half[cut:]):
            vals = [s["mu"] for s in chunk if s.get("mu") is not None]
            blocks.append(float(np.mean(vals)) if vals else float("nan"))
    drift = (100.0 * (blocks[1] / blocks[0] - 1.0)
             if len(blocks) == 2 and blocks[0] else float("nan"))

    phi_bed = float(res.get("phi_bed_steady") or float("nan"))
    phi_extent = float(res.get("phi_extent_final") or float("nan"))
    i_meas = res.get("inertial_number_measured_rate_final")
    i_imp = res.get("inertial_number_imposed_rate")
    rate_meas = res.get("shear_rate_measured_final")
    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 21,
        "experiment": "finite_gap_wall_layer_sensitivity",
        "I_target": I,
        "fill_height": fill,
        "box_width": box_width,
        "box_depth": box_depth,
        "wall_mode": wall_mode,
        "n_particles": int(res.get("n_particles") or 0),
        "gap_m": plate_h,
        "gap_d": plate_h / d if math.isfinite(plate_h) else float("nan"),
        "strain_target": strain,
        "strain_achieved": float(res.get("shear_strain_achieved") or float("nan")),
        "strain_budget_respected": bool(
            res.get("strain_budget_respected", True)),
        "mu_wall": mu_wall,
        "mu_bulk": mu_bulk,
        "mu_full": mu_full,
        "mu_drift_pct_halfsplit": drift,
        "phi_bed": phi_bed,
        "phi_extent": phi_extent,
        "phi_container_final": float(res.get("phi_container_final") or float("nan")),
        "coordination": float(res.get("coordination_number") or float("nan")),
        "p_measured_pa": p_meas,
        "pressure_control_error_pct": (100.0 * (p_meas / P - 1.0)
                                       if math.isfinite(p_meas) else float("nan")),
        "shear_rate_measured": rate_meas,
        "shear_rate_imposed": float(res.get("shear_rate_imposed") or float("nan")),
        "shear_rate_ratio": (rate_meas / res["shear_rate_imposed"]
                             if rate_meas is not None
                             and res.get("shear_rate_imposed") else float("nan")),
        "profile_r2": float(res.get("profile_r2") or float("nan")),
        "I_imposed_rate": i_imp,
        "I_measured_rate": i_meas,
        "I_abscissa": i_imp,          # imposed rate is the self-consistent one
        "mu_full_over_wall": res.get("mu_full_over_wall"),
        "full_momentum_balance": res.get("full_momentum_balance_final"),
        "servo_valid": bool(res.get("servo_valid", False)),
        "wall_clock_s": wall,
        "reference": reference(),     # provenance travels with every artifact
    }


# ------------------------------------------------------------------ assemble
def assemble(points: list[dict]) -> dict:
    """Fit the wall-layer correction at every I target and test the acceptance."""
    by_I: dict[float, list[dict]] = {}
    for p in points:
        by_I.setdefault(float(p["I_target"]), []).append(p)
    out = {"experiment": "finite_gap_wall_layer_sensitivity", "round": 21,
           "protocol": {"P_pa": P, "kn": KN, "mu": MU_CONTACT,
                        "mu_rolling": MU_ROLLING, "restitution": E_REST,
                        "radius_m": R, "density": RHO, "box_width_m": W,
                        "box_depth_m": D, "seed": SEED,
                        "strain_target": STRAIN_TARGET},
           "reference": reference(),
           "reference_status": {"mu": MU_REF_STATUS, "phi": PHI_REF_STATUS},
           "per_I": {}}
    for I in sorted(by_I):
        rows = sorted(by_I[I], key=lambda r: r["gap_d"])
        phi_fit = fit_wall_correction(rows, "phi_bed")
        mu_fit = fit_wall_correction(rows, "mu_wall")
        mu_bulk_fit = fit_wall_correction(rows, "mu_bulk")
        i_ab = float(np.mean([r["I_abscissa"] for r in rows
                              if r.get("I_abscissa")]))
        ref_m, ref_p = float(ref_mu(i_ab)), float(ref_phi(i_ab))
        def _err(v, ref):
            return (None if not (v and math.isfinite(v) and ref)
                    else 100.0 * (v / ref - 1.0))
        out["per_I"][f"{I:g}"] = {
            "I_abscissa": i_ab,
            "n_points": len(rows),
            "gaps_d": [r["gap_d"] for r in rows],
            "phi_bed": [r["phi_bed"] for r in rows],
            "mu_wall": [r["mu_wall"] for r in rows],
            "mu_bulk": [r["mu_bulk"] for r in rows],
            "mu_drift_pct_halfsplit": [r.get("mu_drift_pct_halfsplit")
                                       for r in rows],
            "mu_drift_max_abs_pct": float(np.nanmax(
                [abs(r["mu_drift_pct_halfsplit"]) for r in rows
                 if r.get("mu_drift_pct_halfsplit") is not None
                 and math.isfinite(r["mu_drift_pct_halfsplit"])] or [float("nan")])),
            "strain_budget_respected_all": all(
                r.get("strain_budget_respected", False) for r in rows),
            "phi_fit": phi_fit,
            "mu_wall_fit": mu_fit,
            "mu_bulk_fit": mu_bulk_fit,
            "reference_at_I": {"mu": ref_m, "phi": ref_p},
            "err_pct_thinnest_gap": {"phi": _err(rows[0]["phi_bed"], ref_p),
                                     "mu_wall": _err(rows[0]["mu_wall"], ref_m),
                                     "mu_bulk": _err(rows[0]["mu_bulk"], ref_m)},
            "err_pct_gap_extrapolated": {
                "phi": _err(phi_fit["v_inf"], ref_p),
                "mu_wall": _err(mu_fit["v_inf"], ref_m),
                "mu_bulk": _err(mu_bulk_fit["v_inf"], ref_m)},
            "residuals": {"phi_bed": residual_table(rows, "phi_bed", phi_fit),
                          "mu_wall": residual_table(rows, "mu_wall", mu_fit),
                          "mu_bulk": residual_table(rows, "mu_bulk", mu_bulk_fit)},
        }
    return out


def verdict(analysis: dict, band: float = 10.0) -> dict:
    """Does the correction move the point inside the +/-10 % acceptance band?"""
    rows = []
    for I, blk in analysis["per_I"].items():
        for q in ("phi", "mu_wall", "mu_bulk"):
            thin = blk["err_pct_thinnest_gap"][q]
            ext = blk["err_pct_gap_extrapolated"][q]
            if thin is None or ext is None:
                continue
            rows.append({"I": float(I), "quantity": q,
                         "err_thinnest_gap_pct": thin,
                         "err_extrapolated_pct": ext,
                         "inside_band_thin": abs(thin) < band,
                         "inside_band_extrapolated": abs(ext) < band,
                         "bias_removed_pct": abs(thin) - abs(ext)})
    return {
        "band_pct": band,
        "rows": rows,
        "n_inside_band_thinnest": sum(r["inside_band_thin"] for r in rows),
        "n_inside_band_extrapolated": sum(r["inside_band_extrapolated"]
                                          for r in rows),
        "n_comparisons": len(rows),
        "correction_improves_majority": bool(
            rows and sum(r["bias_removed_pct"] > 0 for r in rows) > 0.5 * len(rows)),
    }


# ------------------------------------------------------------------- figures
def make_figures(analysis: dict, verd: dict, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

    # (a) phi vs 1/H with the linear wall-layer fit
    ax = axes[0][0]
    for k, (I, blk) in enumerate(sorted(analysis["per_I"].items(),
                                        key=lambda kv: float(kv[0]))):
        c = cmap(k)
        inv = [1.0 / g for g in blk["gaps_d"]]
        ax.plot(inv, blk["phi_bed"], "o-", color=c, label=f"I={I}")
        f = blk["phi_fit"]
        if f["identified"] and math.isfinite(f["v_inf"]):
            xx = np.linspace(0.0, max(inv) * 1.05, 50)
            ax.plot(xx, f["v_inf"] + f["slope_per_d"] * xx, ":", color=c, lw=1.2)
            ax.plot([0.0], [f["v_inf"]], "D", color=c, ms=7)
    ax.set_xlabel("d / H   (inverse gap, gap in particle diameters)")
    ax.set_ylabel(r"$\phi_{bed}$")
    ax.set_title("(a) $\\phi$ vs inverse gap; diamonds = fitted $\\phi_\\infty$")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # (b) mu vs 1/H, wall and bulk estimators
    ax = axes[0][1]
    for k, (I, blk) in enumerate(sorted(analysis["per_I"].items(),
                                        key=lambda kv: float(kv[0]))):
        c = cmap(k)
        inv = [1.0 / g for g in blk["gaps_d"]]
        ax.plot(inv, blk["mu_wall"], "s-", color=c, label=f"wall I={I}")
        ax.plot(inv, blk["mu_bulk"], "^--", color=c, alpha=.65,
                label=f"bulk I={I}")
        f = blk["mu_wall_fit"]
        if f["identified"] and math.isfinite(f["v_inf"]):
            xx = np.linspace(0.0, max(inv) * 1.05, 50)
            ax.plot(xx, f["v_inf"] + f["slope_per_d"] * xx, ":", color=c, lw=1.2)
    ax.set_xlabel("d / H")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("(b) $\\mu$ vs inverse gap (platen vs bulk slab)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    # (c) error against the reference law: thinnest gap vs extrapolated
    ax = axes[1][0]
    rows = verd["rows"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["err_thinnest_gap_pct"] for r in rows], 0.4,
           label="thinnest measured gap")
    ax.bar(x + 0.2, [r["err_extrapolated_pct"] for r in rows], 0.4,
           label="H -> inf extrapolated")
    ax.axhline(0.0, color="k", lw=.8)
    for sv in (-10.0, 10.0):
        ax.axhline(sv, color="r", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['quantity']}\nI={r['I']:g}" for r in rows],
                       fontsize=6, rotation=45)
    ax.set_ylabel("error vs reference law, %")
    ax.set_title("(c) acceptance error before/after the correction")
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    # (d) mu stationarity proxy vs gap
    ax = axes[1][1]
    for k, (I, blk) in enumerate(sorted(analysis["per_I"].items(),
                                        key=lambda kv: float(kv[0]))):
        dr = [abs(v) if (v is not None and math.isfinite(v)) else np.nan
              for v in blk["mu_drift_pct_halfsplit"]]
        ax.plot(blk["gaps_d"], dr, "o-", color=cmap(k), label=f"I={I}")
    ax.set_xlabel("gap H / d")
    ax.set_ylabel("|mu drift| last-half split, %")
    ax.set_title("(d) stationarity proxy vs gap")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout()
    p1 = outdir / "fig_gap_sensitivity.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    figs.append({"file": str(p1.relative_to(HERE)),
                 "caption": "(a) phi vs 1/(gap in diameters) with the linear "
                            "wall-layer fit; diamonds are the H->inf intercept; "
                            "(b) mu from the platen (solid) and the bulk slab "
                            "(dashed) on the same abscissa; (c) error against "
                            "the reference law at the thinnest measured gap vs "
                            "the extrapolated unbounded-gap value (red lines = "
                            "the +/-10 % acceptance band); (d) half-split drift "
                            "of mu as a stationarity proxy, per gap."})

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
    for k, (I, blk) in enumerate(sorted(analysis["per_I"].items(),
                                        key=lambda kv: float(kv[0]))):
        c = cmap(k)
        inv = [1.0 / g for g in blk["gaps_d"]]
        ax2.plot(inv, blk["phi_bed"], "o", color=c, alpha=.55)
        f = blk["phi_fit"]
        if f["identified"] and math.isfinite(f["v_inf"]):
            ax2.plot([0.0], [f["v_inf"]], "D", color=c, ms=9,
                     label=f"I={I}: $\\phi_\\infty$={f['v_inf']:.3f}, "
                           f"$\\delta/d$={f['delta_over_d']:.2f}")
    ax2.set_xlabel("d / H")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_title("Measured $\\phi$ (circles) and fitted $\\phi_\\infty$ "
                  "(diamonds) with identified dead-layer thickness")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = outdir / "fig_phi_extrapolation.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    figs.append({"file": str(p2.relative_to(HERE)),
                 "caption": "Measured phi(d/H) (circles) and the fitted "
                            "unbounded-gap value phi_inf (diamonds), with the "
                            "identified dead-layer thickness delta/d per I."})
    return figs


# ---------------------------------------------------------------------- main
def _load_points(outdir: Path = OUT) -> list[dict]:
    pts = []
    for f in sorted(outdir.glob("gap*.json")):
        try:
            pts.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return pts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="run one gap point")
    ap.add_argument("--assemble", action="store_true", help="fit + figures")
    ap.add_argument("--I", type=float, default=0.01)
    ap.add_argument("--fill", type=float, default=0.04)
    ap.add_argument("--strain", type=float, default=STRAIN_TARGET)
    ap.add_argument("--settle", type=int, default=4000)
    ap.add_argument("--sample-stride", type=int, default=20)
    ap.add_argument("--box-width", type=float, default=W)
    ap.add_argument("--box-depth", type=float, default=D)
    ap.add_argument("--wall-mode", default="smooth")
    ap.add_argument("--matrix", default="full", choices=("full", "small"),
                    help="full = (0.01,0.03,0.1) x 4 fills in the 0.12x0.05 cell; "
                         "small = I=0.1 only in the cheap 0.06x0.03 cell")
    ap.add_argument("--shard", default=None,
                    help="i/n: run the i-th strided slice of the (I, fill) matrix")
    args = ap.parse_args(argv)

    if args.shard:
        i_s, n_s = args.shard.split("/")
        i_sh, n_sh = int(i_s) - 1, int(n_s)
        if args.matrix == "small":
            matrix = [(0.1, f) for f in FILLS]
            args.box_width, args.box_depth = 0.06, 0.03
        else:
            matrix = [(I, f) for I in I_TARGETS for f in FILLS]
        for k in range(i_sh, len(matrix), n_sh):
            I, f = matrix[k]
            rec = run_point(I, f, strain=args.strain, settle_steps=args.settle,
                            sample_stride=args.sample_stride,
                            wall_mode=args.wall_mode,
                            box_width=args.box_width,
                            box_depth=args.box_depth)
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"gap{I:g}__fill{f:.3f}.json"
            path.write_text(json.dumps(rec, indent=1))
            print(f"[gap_sensitivity] I={I:g} fill={f:.3f}: gap/d={rec['gap_d']:.2f} "
                  f"N={rec['n_particles']} mu_wall={rec['mu_wall']:.4f} "
                  f"phi={rec['phi_bed']:.4f} "
                  f"drift={rec['mu_drift_pct_halfsplit']:.2f}% "
                  f"wall={rec['wall_clock_s']:.0f}s")
        return 0

    if args.run:
        OUT.mkdir(parents=True, exist_ok=True)
        rec = run_point(args.I, args.fill, strain=args.strain,
                        settle_steps=args.settle,
                        sample_stride=args.sample_stride,
                        wall_mode=args.wall_mode,
                        box_width=args.box_width, box_depth=args.box_depth)
        tag = f"gap{args.I:g}__fill{args.fill:.3f}"
        path = OUT / f"{tag}.json"
        path.write_text(json.dumps(rec, indent=1))
        print(f"[gap_sensitivity] {tag}: gap/d={rec['gap_d']:.2f} "
              f"N={rec['n_particles']} mu_wall={rec['mu_wall']:.4f} "
              f"phi={rec['phi_bed']:.4f} drift={rec['mu_drift_pct_halfsplit']:.2f}% "
              f"wall={rec['wall_clock_s']:.0f}s -> {path}")
        return 0

    if args.assemble:
        pts = _load_points()
        if not pts:
            print("[gap_sensitivity] no point files; run shards first")
            return 1
        analysis = assemble(pts)
        verd = verdict(analysis)
        figs = make_figures(analysis, verd, OUT)
        analysis["verdict"] = verd
        analysis["figures"] = figs
        (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))
        for I, blk in analysis["per_I"].items():
            print(f"I={I}: gaps={['%.2f' % g for g in blk['gaps_d']]} "
                  f"phi_inf={blk['phi_fit']['v_inf']:.4f} "
                  f"(delta/d={blk['phi_fit']['delta_over_d']:.2f}) "
                  f"mu_wall_inf={blk['mu_wall_fit']['v_inf']:.4f} "
                  f"mu_bulk_inf={blk['mu_bulk_fit']['v_inf']:.4f} "
                  f"drift_max={blk['mu_drift_max_abs_pct']:.1f}%")
        print(f"inside band: thin={verd['n_inside_band_thinnest']}"
              f"/{verd['n_comparisons']} "
              f"extrapolated={verd['n_inside_band_extrapolated']}"
              f"/{verd['n_comparisons']}  "
              f"improves_majority={verd['correction_improves_majority']}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())