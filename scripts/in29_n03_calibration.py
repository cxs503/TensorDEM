"""IN29-N03: DEM ice-load time-history statistics calibration campaign.

Runs, in order,

1. the DEM indentation anchor (real ``IceDEM`` runs at three contact dampings),
2. the four-factor Latin-hypercube sweep + quadratic-surrogate calibration,
3. the exact closed-loop validation of the calibrated record (whipping stress
   share, two damage-share definitions, four-segment dispersion),
4. the spectral-vs-exact response cross-check,
5. a record-length convergence study of the damage-share statistics,
6. the identifiability (Fisher information) analysis of the four controls.

Everything lands in ``artifacts/in29_n03/`` as JSON/CSV plus PNG figures.

Usage::

    python scripts/in29_n03_calibration.py            # full campaign
    python scripts/in29_n03_calibration.py --quick     # skip DEM anchor
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.whipping import load_calibration as lc        # noqa: E402
from tensordem.whipping import load_stats as ls             # noqa: E402

OUT = ROOT / "artifacts" / "in29_n03"


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, lc.IceLoadControls):
        return obj.as_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# --------------------------------------------------------------------------- #
# 1. DEM anchor
# --------------------------------------------------------------------------- #
def run_anchor(dampings=(5.0, 15.0), duration: float = 1.2) -> dict:
    # The DEM explicit step is bounded by the *total* dashpot damping
    # (dt = 0.2 m / c_total), so contact dampings far above the default
    # 5 N s/m are computationally prohibitive on the 21x5 lattice.  The anchor
    # therefore uses a 13x5 lattice (65 disks) and two dampings; the fitted
    # exponent is a two-point slope and is reported as such.
    t0 = time.time()
    anchor = lc.dem_anchor_table(dampings=dampings, duration=duration,
                                 nx=13, ny=5, start_x=-0.5)
    anchor["wall_time_s"] = round(time.time() - t0, 1)
    anchor["note"] = ("model-scale pulse statistics of the repository ice-indentation DEM; "
                      "Froude time scaling lambda_t = sqrt(L_ship / L_model) maps them to "
                      "the 150 m hull")
    write_json("dem_anchor.json", anchor)
    return anchor


# --------------------------------------------------------------------------- #
# 2. sweep + calibration
# --------------------------------------------------------------------------- #
def run_calibration(n_points: int = 24, rounds: int = 3, seed: int = 0) -> dict:
    t0 = time.time()
    out = lc.calibrate(n_points=n_points, seed=seed, rounds=rounds)
    payload = {
        "wall_time_s": round(time.time() - t0, 1),
        "n_lhs_points": n_points,
        "history": out["history"],
        "surrogate": {
            "r2": out["surrogate"]["r2"],
            "sigma2": out["surrogate"]["sigma2"],
            "design_rank": out["surrogate"]["design_rank"],
            "n_points": out["surrogate"]["n_points"],
        },
        "sweep": [{
            "index": p["index"],
            "unit": p["unit"],
            "controls": p["controls"],
            "statistics": lc.statistic_vector(p),
            "load": {k: v for k, v in p["load"].items() if isinstance(v, (int, float))},
            "response_spectral": {k: v for k, v in p["response"].items()
                                  if isinstance(v, (int, float))},
        } for p in out["points"]],
        "calibrated_controls": out["best"]["controls"].as_dict(),
        "calibrated_statistics": lc.statistic_vector(out["best"]["result"]),
        "calibrated_unit": out["best"]["unit"].tolist(),
        "target_bands": lc.CALIBRATION_TARGETS,
    }
    write_json("calibration.json", payload)

    # CSV of the sweep for plotting / external use
    rows = ["index,crushing_scale,contact_damping,floe_geometry,froude,ratio_R,"
            "rise_time_median_s,spectral_slope,whipping_share_spectral"]
    for p in out["points"]:
        c, s = p["controls"], lc.statistic_vector(p)
        rows.append(",".join(str(x) for x in [
            p["index"], f"{c['crushing_scale']:.5f}", f"{c['contact_damping']:.5f}",
            f"{c['floe_geometry']:.5f}", f"{c['froude']:.5f}",
            f"{s['ratio_R']:.5f}", f"{s['rise_time_median_s']:.6f}",
            f"{s['spectral_slope_decade_per_decade']:.5f}",
            f"{s['whipping_stress_share']:.5f}"]))
    (OUT / "lhs_sweep.csv").write_text("\n".join(rows) + "\n")
    payload["best"] = {"direct_cost": out["best"]["direct_cost"],
                       "unit": out["best"]["unit"].tolist()}
    return payload


# --------------------------------------------------------------------------- #
# 3. exact closed-loop validation + 4. spectral cross-check + 5. convergence
# --------------------------------------------------------------------------- #
def run_validation(controls: lc.IceLoadControls, case=None) -> dict:
    case = lc.ShipCase() if case is None else case
    t, f, meta = lc.synthesise(controls)
    stats = ls.record_statistics(t, f)
    rise = lc.conditioned_rise_statistics(t, f)
    # the campaign's t_r is the conditioned (macroscale) rise time, consistent
    # with the calibration vector; keep the raw value for the audit trail
    stats["rise_time_median_s_raw"] = stats["rise_time_median_s"]
    stats.update({k: v for k, v in rise.items()})
    segments = ls.segment_statistics(t, f, n_segments=4)
    t0 = time.time()
    exact = lc.closed_loop(t, f, case)
    wall = time.time() - t0
    spectral = lc.spectral_response_metrics(t, f, case)
    # The repository copy is decimated to 200 Hz (the 1 kHz record is 3 MB and
    # is regenerated in seconds by this script); the statistics above are all
    # computed on the full 1 kHz record.
    step = 5
    np.savetxt(OUT / "load_record_200Hz.csv",
               np.column_stack([t[::step], f[::step]]), delimiter=",",
               header="time_s,force_N", comments="")
    payload = {
        "controls": controls.as_dict(),
        "meta": {k: v for k, v in meta.items() if k != "components"},
        "load_statistics": stats,
        "conditioned_rise_statistics": rise,
        "load_segments": [{k: v for k, v in s.items() if isinstance(v, (int, float))}
                          for s in segments],
        "load_segment_ratio_R_dispersion": ls.dispersion(
            [s["ratio_R"] for s in segments]),
        "load_segment_slope_dispersion": ls.dispersion(
            [s["spectral_slope_decade_per_decade"] for s in segments]),
        "response_exact": exact,
        "response_spectral": spectral,
        "spectral_vs_exact_share_bias": (spectral["whipping_stress_share"]
                                         / exact["whipping_stress_share"]),
        "spectral_vs_exact_std_bias": (spectral["stress_std_Pa"]
                                       / exact["stress_std_Pa"]),
        "exact_wall_time_s": round(wall, 1),
    }
    write_json("validation.json", payload)
    return payload


def run_spectral_crosscheck(controls_list, case=None) -> dict:
    case = lc.ShipCase() if case is None else case
    rows = []
    for i, c in enumerate(controls_list):
        t, f, _ = lc.synthesise(c)
        exact = lc.closed_loop(t, f, case)
        spec = lc.spectral_response_metrics(t, f, case)
        rows.append({
            "case": i,
            "controls": c.as_dict(),
            "share_exact": exact["whipping_stress_share"],
            "share_spectral": spec["whipping_stress_share"],
            "std_exact_Pa": exact["stress_std_Pa"],
            "std_spectral_Pa": spec["stress_std_Pa"],
            "share_bias": spec["whipping_stress_share"] / exact["whipping_stress_share"],
        })
    biases = [r["share_bias"] for r in rows]
    payload = {"rows": rows, "mean_share_bias": float(np.mean(biases)),
               "max_abs_share_error": float(max(abs(b - 1.0) for b in biases))}
    write_json("spectral_crosscheck.json", payload)
    return payload


def run_convergence(controls: lc.IceLoadControls, durations=(60.0, 120.0, 240.0),
                    case=None) -> dict:
    """Dispersion of the damage-share statistics against record length."""
    case = lc.ShipCase() if case is None else case
    rows = []
    for T in durations:
        c = lc.controls_from_dict(controls.as_dict(), duration_s=T)
        t, f, _ = lc.synthesise(c)
        r = lc.closed_loop(t, f, case)
        rows.append({
            "duration_s": T,
            "whipping_stress_share": r["whipping_stress_share"],
            "damage_share_band": r["whipping_damage_share_band"],
            "damage_share_ideal": r["whipping_damage_share_ideal_split"],
            "damage_share_modal": r["whipping_damage_share_modal"],
            "segment_damage_share": r["segment_damage_share"],
            "segment_damage_share_dispersion": r["segment_damage_share_dispersion"],
            "segment_ratio_R_dispersion": r["segment_ratio_R_dispersion"],
            "rainflow_cycles_stress": r["rainflow_cycles_stress"],
        })
    payload = {"controls": controls.as_dict(), "rows": rows,
               "note": ("four equal-time windows per record; the damage share is a "
                        "range^5 statistic and therefore converges slowly in the "
                        "number of extreme cycles")}
    write_json("convergence.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# 6. identifiability
# --------------------------------------------------------------------------- #
def run_identifiability(controls: lc.IceLoadControls, result: dict, case=None) -> dict:
    case = lc.ShipCase() if case is None else case
    ident = lc.identifiability(controls, result, case)
    write_json("identifiability.json", ident)
    return ident


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(validation: dict, calibration: dict, anchor: dict,
                 convergence: dict, ident: dict) -> list[str]:
    plt = _pyplot()
    OUT.mkdir(parents=True, exist_ok=True)
    names = []

    # --- figure 1: calibrated load record + PSD ------------------------- #
    data = np.loadtxt(OUT / "load_record_200Hz.csv", delimiter=",", skiprows=1)
    t, f = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(t, f / 1e6, lw=0.35)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("bow load [MN]")
    ax[0].set_title("calibrated 60 s ice-load record (1 kHz)")
    ax[0].grid(alpha=0.3)
    freq, psd = ls.welch_psd(t, f)
    ax[1].loglog(freq[1:], psd[1:], lw=0.9, label="synthesised record")
    band = (freq >= 0.5) & (freq <= 10.0) & (psd > 0)
    slope = validation["load_statistics"]["spectral_slope_decade_per_decade"]
    ref = psd[band][0] * (freq[band] / freq[band][0]) ** slope
    ax[1].loglog(freq[band], ref, "k--", lw=1.0,
                 label=f"fit: {slope:+.2f} decade/decade")
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel("PSD [N$^2$/Hz]")
    ax[1].set_title("load spectrum, 0.5-10 Hz slope")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_record_and_spectrum.png", dpi=140)
    plt.close(fig)
    names.append("fig1_record_and_spectrum.png")

    # --- figure 2: sweep sensitivity ------------------------------------ #
    sweep = calibration["sweep"]
    params = ["crushing_scale", "contact_damping", "floe_geometry", "froude"]
    metrics = [("ratio_R", "R = mean/peak"),
               ("rise_time_median_s", "t$_r$ [s]"),
               ("spectral_slope_decade_per_decade", "PSD slope"),
               ("whipping_stress_share", "whipping share")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        vals = [p["statistics"][key] for p in sweep]
        ax.scatter(vals, [p["statistics"]["ratio_R"] for p in sweep], s=16)
        lo, hi = calibration["target_bands"][key]["band"]
        ax.axvspan(lo, hi, color="tab:green", alpha=0.15)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R (colour = LHS point)")
    fig.suptitle("four-factor Latin hypercube: statistics vs target bands (green)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lhs_sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    names.append("fig2_lhs_sweep.png")

    # --- figure 3: sensitivity / Fisher information ---------------------- #
    names_p = ident["parameters"]
    J = np.array(ident["jacobian"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(names_p))
    width = 0.2
    for k, (label, colour) in enumerate(zip(
            ["dR", "d t$_r$", "d slope", "d whipping share"],
            ["tab:blue", "tab:orange", "tab:green", "tab:red"])):
        ax.bar(x + (k - 1.5) * width, J[k], width, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in names_p], fontsize=8)
    ax.set_ylabel("d(normalised statistic) / d ln p")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("which parameter controls which statistic")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sensitivity.png", dpi=140)
    plt.close(fig)
    names.append("fig3_sensitivity.png")

    # --- figure 4: segments + convergence -------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    seg = validation["response_exact"]["segment_damage_share"]
    segR = validation["response_exact"]["segment_ratio_R"]
    x = np.arange(1, len(seg) + 1)
    ax[0].bar(x - 0.18, segR, 0.36, label="R per segment")
    ax[0].bar(x + 0.18, seg, 0.36, label="damage share per segment")
    ax[0].axhline(0.30, color="tab:blue", ls=":", lw=1)
    ax[0].axhline(0.50, color="tab:blue", ls=":", lw=1)
    ax[0].set_xlabel("15 s segment")
    ax[0].set_title("per-segment R and whipping damage share")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    rows = convergence["rows"]
    T = [r["duration_s"] for r in rows]
    ax[1].semilogy(T, [r["segment_damage_share_dispersion"] for r in rows], "o-",
                   label="damage-share dispersion")
    ax[1].semilogy(T, [r["segment_ratio_R_dispersion"] for r in rows], "s-",
                   label="R dispersion")
    ax[1].axhline(0.10, color="k", ls="--", lw=1, label="10 % criterion")
    ax[1].set_xlabel("record length T [s]")
    ax[1].set_title("segment dispersion vs record length")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_segments_convergence.png", dpi=140)
    plt.close(fig)
    names.append("fig4_segments_convergence.png")

    # --- figure 5: damage-share definitions ------------------------------ #
    resp = validation["response_exact"]
    labels = ["butterworth\nband split", "ideal FFT\nsplit", "modal split",
              "removal\ntechnique"]
    values = [resp["whipping_damage_share_band"],
              resp["whipping_damage_share_ideal_split"],
              resp["whipping_damage_share_modal"],
              resp["whipping_damage_share_removal"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:cyan", "tab:green", "tab:red"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("whipping damage share [-]")
    ax.set_title("four damage-share estimators on the same calibrated record")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_damage_definitions.png", dpi=140)
    plt.close(fig)
    names.append("fig5_damage_definitions.png")

    # --- figure 6: DEM anchor -------------------------------------------- #
    runs = anchor["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    zeta = [r["contact_damping"] for r in runs]
    rises = [r["rise_time_median_s_model_scale"] * 1e3 for r in runs]
    full = anchor["rise_time_median_s_full_scale"]
    ax[0].loglog(zeta, rises, "o-", label="model scale")
    ax[0].loglog(zeta, [r * 1e3 for r in full], "s--", label="Froude-scaled (150 m)")
    ax[0].axhspan(50, 250, color="tab:green", alpha=0.15, label="target band")
    ax[0].set_xlabel("DEM contact damping [N s/m]")
    ax[0].set_ylabel("median 10-90 % rise time [ms]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].bar([str(z) for z in zeta], [r["ratio_R"] for r in runs],
              color="tab:gray")
    ax[1].axhspan(0.30, 0.50, color="tab:green", alpha=0.15)
    ax[1].set_xlabel("DEM contact damping [N s/m]")
    ax[1].set_title("R of the raw DEM record (target band shaded)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("DEM anchor: prototype pulse statistics and Froude scaling", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_dem_anchor.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    names.append("fig6_dem_anchor.png")
    return names


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the DEM anchor runs")
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-convergence", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    print("== DEM anchor ==", flush=True)
    if args.quick and (OUT / "dem_anchor.json").exists():
        anchor = json.loads((OUT / "dem_anchor.json").read_text())
        print("   reused artifacts/in29_n03/dem_anchor.json")
    else:
        anchor = run_anchor()

    print("== LHS sweep + calibration ==", flush=True)
    cal = run_calibration(args.points, args.rounds, args.seed)
    controls = cal["calibrated_controls"]
    print("   calibrated controls:", {k: round(v, 4) for k, v in controls.items()
                                     if k in ("crushing_scale", "contact_damping",
                                              "floe_geometry", "froude")})

    print("== exact closed-loop validation ==", flush=True)
    validation = run_validation(lc.controls_from_dict(controls))

    print("== spectral vs exact cross-check ==", flush=True)
    variants = [
        lc.controls_from_dict(controls, seed=1),
        lc.controls_from_dict(controls, seed=2),
        lc.controls_from_dict(controls, floe_geometry=controls["floe_geometry"] * 1.2),
        lc.controls_from_dict(controls, crushing_scale=controls["crushing_scale"] * 1.3),
    ]
    cross = run_spectral_crosscheck(variants)

    print("== convergence study ==", flush=True)
    if args.skip_convergence:
        convergence = json.loads((OUT / "convergence.json").read_text())
    else:
        convergence = run_convergence(lc.controls_from_dict(controls))

    print("== identifiability ==", flush=True)
    base_result = lc.evaluate(lc.controls_from_dict(controls))
    ident = run_identifiability(lc.controls_from_dict(controls), base_result)

    print("== figures ==", flush=True)
    figures = make_figures(validation, cal, anchor, convergence, ident)

    print("== acceptance ==", flush=True)
    lock_path = OUT / "regression_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {
        "passed": None, "failed": None}
    report = {
        "load": validation["load_statistics"],
        "response_exact": validation["response_exact"],
        "identifiability": ident,
        "regression_lock": lock,
        "test_count": lock.get("test_count", 0),
    }
    criteria = lc.acceptance(report)
    criteria["spectral_share_bias"] = cross["mean_share_bias"]
    criteria["spectral_share_max_error"] = cross["max_abs_share_error"]
    criteria["validation_record"] = validation["controls"]
    criteria["calibration_cost"] = cal["history"][-1]["direct_cost"]
    criteria["surrogate_r2"] = cal["surrogate"]["r2"]
    write_json("acceptance.json", criteria)
    print(json.dumps({k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                      for k, v in criteria.items()}, indent=2, default=_json_default))
    print("figures:", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())