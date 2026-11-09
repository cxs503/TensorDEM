"""IN99-N08 benchmark driver: Terzaghi anchors + DEM bed + r_u(T_v) interface.

Runs the whole chain and writes the machine-readable evidence:

* ``artifacts/in99-n08/metrics.json``  -- every number and every gate verdict
* ``artifacts/in99-n08/ru_table.json``  -- the published ``r_u(T_v)`` interface
* ``artifacts/in99-n08/consumer_assertions.json`` -- downstream end-to-end use
* ``artifacts/in99-n08/figures/*.png`` -- report figures

Usage::

    PYTHONPATH=src python3 benchmarks/consolidation_terzaghi/run.py            # full DEM scan
    PYTHONPATH=src python3 benchmarks/consolidation_terzaghi/run.py --quick    # synthetic DEM, no DEM cost
"""
from __future__ import annotations

import argparse
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.consolidation import (  # noqa: E402
    DEFAULT_TV_POINTS,
    PrototypeBed,
    OedometerConfig,
    U_series,
    U_sqrt_approx,
    Tv_of_U,
    r_u_profile,
    run_chain,
    run_oedometer,
    table_sha256,
    write_ru_table,
)

ART = REPO / "artifacts" / "in99-n08"
FIG = ART / "figures"

#: Domain-size (particle-count) scan: same grain, same bed height, wider cell.
DEM_SCAN = (
    {"n_row": 4, "n_layer": 5, "seed": 20261108},
    {"n_row": 5, "n_layer": 5, "seed": 20261108},
    {"n_row": 6, "n_layer": 5, "seed": 20261108},
    {"n_row": 4, "n_layer": 5, "seed": 771103},
)

PROTOTYPE = PrototypeBed(height=1.0, stress_step=10.0e3, drainage="double")

#: Synthetic stand-in used by ``--quick`` (no DEM cost; gates are smoke-tested).
SYNTHETIC_DEM = {
    "E_oed": 1.29e6, "E_oed_mid": 1.28e6, "k_hyd": 1.81e-3, "porosity": 0.545,
    "coordination": 2.6, "strain_increment": 7.75e-3, "total_steps": 65200,
    "stages": [{"stress": 5.0e3, "height": 1.159e-3, "porosity": 0.553, "coordination": 2.5},
               {"stress": 15.0e3, "height": 1.150e-3, "porosity": 0.549, "coordination": 2.7}],
    "config": {"diameter": 2.0e-4, "particle_count": 80, "kn": 2000.0, "radius": 1.0e-4,
               "density": 2650.0, "stress_levels": [5.0e3, 15.0e3], "dt": 5.23e-7,
               "box_width": 8.08e-4, "n_row": 4, "n_layer": 5, "seed": 20261108},
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _aspect_scan() -> tuple[dict, ...]:
    """Domain-size scan with a scale-invariant aspect ratio (n_layer = n_row)."""
    return tuple({"n_row": n, "n_layer": n, "seed": 20261108} for n in (4, 5, 6))


def build_dem_ensemble(quick: bool, compaction_porosity: float | None = None,
                       scan_override: tuple[dict, ...] | None = None) -> list[dict]:
    """Run the DEM oedometer over the domain-size scan (or fake it)."""
    scan = tuple(scan_override) if scan_override is not None else DEM_SCAN
    if quick:
        ensemble = []
        for i, spec in enumerate(scan):
            dem = json.loads(json.dumps(SYNTHETIC_DEM))  # deep copy
            scale = 1.0 + 0.04 * i  # a deliberately spread ensemble for the CI smoke test
            dem["E_oed"] *= scale
            dem["k_hyd"] /= scale ** 0.5
            dem["config"]["particle_count"] = spec["n_row"] ** 2 * spec["n_layer"]
            dem["config"]["seed"] = spec["seed"]
            dem["porosity"] = SYNTHETIC_DEM["porosity"] - 0.004 * i
            ensemble.append(dem)
        return ensemble

    ensemble = []
    for spec in scan:
        cfg = OedometerConfig(
            n_row=spec["n_row"], n_layer=spec["n_layer"], seed=spec["seed"],
            compaction_porosity=compaction_porosity,
        )
        t0 = time.time()
        dem = run_oedometer(cfg, progress=lambda m: _log(f"    {m}"))
        dem["wall_time_s"] = time.time() - t0
        dem["scan_spec"] = spec
        _log(f"  level n_row={spec['n_row']} seed={spec['seed']}: "
             f"E_oed={dem['E_oed']:.4g} Pa, k={dem['k_hyd']:.4g} m/s, "
             f"n={dem['porosity']:.4f}, Z={dem['coordination']:.2f}, "
             f"{dem['total_steps']} steps, {dem['wall_time_s']:.1f} s")
        ensemble.append(dem)
    return ensemble


def figure_terzaghi(metrics: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c1 = metrics["C1"]["points"]
    sim_U = [p["U_simulated"] for p in c1]
    ana_U = [p["U_analytic"] for p in c1]
    grid = np.logspace(-3, math.log10(5.0), 200)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(grid, U_series(grid), "-", color="C0", label="Fourier series (anchor)")
    ax[0].plot(grid, U_sqrt_approx(grid), "--", color="C1", label=r"$\sqrt{4T_v/\pi}$ branch")
    ax[0].plot([p["T_v"] for p in c1], sim_U, "o", color="C3", label="two-field column")
    for p in c1:
        ax[0].annotate(f'{p["deviation_pct"]:.3f}%', (p["T_v"], p["U_simulated"]),
                       textcoords="offset points", xytext=(4, -11), fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$T_v = C_v t / H_{dr}^2$")
    ax[0].set_ylabel(r"$U$")
    ax[0].set_title("C1: degree of consolidation vs analytic anchor")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(alpha=0.3)

    for nc in metrics["negative_control"]:
        fac = nc["E_oed_factor"]
        ax[1].plot([p["T_v"] for p in c1],
                   [U_series(Tv * fac) for Tv in [p["T_v"] for p in c1]],
                   "s--", ms=4, label=f"negative control: $E_{{oed}}\\times${fac}")
    ax[1].plot([p["T_v"] for p in c1], ana_U, "o-", color="k", label="anchor")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$T_v$ (reference $C_v$)")
    ax[1].set_ylabel(r"$U$")
    ax[1].set_title("Gate discriminating power ($\\pm$20 % in $C_v$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_settlement(metrics: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = metrics["column"]["samples"]
    Tv = np.array([s["T_v"] for s in samples])
    s_sim = np.array([s["settlement_simulated"] for s in samples])
    s_f = metrics["C2"]["s_f_analytic"]
    grid = np.logspace(math.log10(max(Tv.min(), 1e-4)), math.log10(Tv.max()), 200)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(grid, s_f * U_series(grid) * 1e3, "-", color="C0", label=r"analytic $s_f U(T_v)$")
    ax.plot(Tv, s_sim * 1e3, "o", color="C3", label="two-field column")
    ax.axhline(s_f * 1e3, ls=":", color="k", lw=1, label=r"$s_f=\Delta\sigma H/E_{oed}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_v$")
    ax.set_ylabel("settlement (mm)")
    ax.set_title("C2: settlement-time curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_profiles(metrics: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = metrics["column"]["snapshots"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for snap in snaps:
        Tv = snap["T_v"]
        z = np.array(snap["z_over_hdr"])
        ax.plot(z, snap["r_u"], "o", ms=3.5, label=f"column $T_v$={Tv:.2f}")
        zz = np.linspace(0.0, 1.0, 101)
        ax.plot(zz, r_u_profile(zz, Tv), "-", lw=1.0, alpha=0.75)
    ax.set_xlabel(r"$z/H_{dr}$  (0 = drained face)")
    ax.set_ylabel(r"$r_u = u_{excess}/\Delta\sigma$")
    ax.set_title("Excess pore-pressure profiles (lines: series anchor)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_dem(ensemble: list[dict], metrics: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for dem in ensemble:
        stages = dem.get("stages", [])
        if not stages:
            continue
        stress = [s["stress"] / 1e3 for s in stages]
        poro = [s["porosity"] for s in stages]
        ax[0].plot(stress, poro, "o-", label=f'N={dem["config"]["particle_count"]}')
    ax[0].set_xlabel("piston stress (kPa)")
    ax[0].set_ylabel("porosity n")
    ax[0].set_title("DEM oedometer: bed state at the two load stages")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    scan = metrics["C3"]["dem_particle_scan"]
    curves = scan.get("curves", [])
    if curves:
        N = [c["particle_count"] for c in curves]
        E = [c["E_oed"] / 1e6 for c in curves]
        k = [c["k_hyd"] * 1e3 for c in curves]
        ax[1].plot(N, E, "o-", color="C0", label=r"$E_{oed}$ (MPa)")
        ax[1].set_xlabel("particles in the DEM cell")
        ax[1].set_ylabel(r"$E_{oed}$ (MPa)", color="C0")
        ax2 = ax[1].twinx()
        ax2.plot(N, k, "s--", color="C3", label=r"$k_{hyd}$ (mm/s)")
        ax2.set_ylabel(r"$k_{hyd}$ (mm/s)", color="C3")
        ax[1].set_title(f'C3: domain-size scan, U drift {scan["drift_pct"]:.2f} %')
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_interface(table: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Tv = np.array(table["T_v"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(Tv, table["r_u_average"], "-o", ms=3, color="C0", label=r"$r_u$ (depth average)")
    ax.plot(Tv, table["r_u_midplane"], "-s", ms=3, color="C3", label=r"$r_u$ (mid-plane)")
    ci = table.get("C_v_confidence", {})
    subtitle = ""
    if ci.get("ci_low") is not None and not (isinstance(ci["ci_low"], float) and math.isnan(ci["ci_low"])):
        subtitle = (f'  $C_v$ = {table["C_v_m2_s"]:.3g} '
                    f'[{ci["ci_low"]:.3g}, {ci["ci_high"]:.3g}] m²/s (95 %)')
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_v$")
    ax.set_ylabel(r"$r_u$")
    ax.set_title("exported $r_u(T_v)$ interface" + subtitle, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="synthetic DEM (no DEM cost)")
    parser.add_argument("--skip-consumer", action="store_true")
    parser.add_argument("--compaction-porosity", type=float, default=None,
                        help="piston pre-consolidation porosity (default: as-deposited)")
    parser.add_argument("--scan-aspect", action="store_true",
                        help="scale the DEM cell aspect ratio with the domain width (n_layer = n_row)")
    parser.add_argument("--dem-cache", default=None,
                        help="reuse a cached DEM ensemble JSON instead of re-running the DEM")
    parser.add_argument("--tag", default=None,
                        help="suffix for the output files (metrics_<tag>.json, ...)")
    args = parser.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    _log("=" * 78)
    _log("IN99-N08  one-dimensional consolidation of a saturated granular bed")
    _log("=" * 78)

    _log("[1/6] analytic anchor self-check")
    anchors = {
        "T_v(U=0.5)": Tv_of_U(0.5),
        "T_v(U=0.9)": Tv_of_U(0.9),
        "T_v(U=0.5)_literature": 0.197,
        "T_v(U=0.9)_literature": 0.848,
        "sqrt_branch_vs_series_max_pct_U_below_0.53": float(
            max(abs(U_sqrt_approx(t) - U_series(t)) / U_series(t) * 100.0
                for t in np.linspace(1e-4, 0.2206, 400))
        ),
        "sqrt_branch_error_at_U0.53_pct": float(
            abs(U_sqrt_approx(0.2206) - U_series(0.2206)) / U_series(0.2206) * 100.0
        ),
    }
    for k, v in anchors.items():
        _log(f"    {k:52s} {v:.6g}")

    _log("[2/6] DEM oedometer ensemble (domain-size scan)")
    cache_path = Path(args.dem_cache) if args.dem_cache else ART / (
        f"dem_scan_{args.tag}.json" if args.tag else "dem_scan.json")
    if args.dem_cache and Path(args.dem_cache).exists():
        cached = json.loads(Path(args.dem_cache).read_text())
        ensemble = cached["ensemble"]
        for d, spec in zip(ensemble, cached.get("scan", [])):
            d["scan_spec"] = spec
        _log(f"    reusing cached DEM ensemble: {args.dem_cache} "
             f"({len(ensemble)} realisations)")
    else:
        ensemble = build_dem_ensemble(
            args.quick, args.compaction_porosity,
            scan_override=_aspect_scan() if args.scan_aspect else None,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"schema": "tensordem.consolidation.dem_scan/1",
             "scan": [d.get("scan_spec", {}) for d in ensemble],
             "ensemble": ensemble}, indent=2, default=float) + "\n")
        _log(f"    DEM ensemble cached -> {cache_path}")
    # the reference realisation is the *largest* domain: it is the least
    # boundary-influenced member of the scan (smaller cells over-estimate the
    # porosity and under-estimate E_oed; see the report, section 5)
    reference = max(ensemble, key=lambda d: d["config"]["particle_count"])

    _log("[3/6] two-field column, C1/C2/C3 gates + r_u table")
    metrics = run_chain(
        dem=reference,
        prototype=PROTOTYPE,
        n_cells=40,
        Tv_points=DEFAULT_TV_POINTS,
        refinement_cells=(20, 40, 80),
        dem_ensemble=ensemble,
    )
    metrics["anchors"] = anchors
    metrics["mode"] = "quick" if args.quick else "full"
    metrics["protocol"] = {
        "dem_bed_preparation": ("pre-consolidated to n = %.3f" % args.compaction_porosity
                                if args.compaction_porosity is not None
                                else "as-deposited (first load stage consolidates the bed)"),
        "dem_scan_geometry": ("scale-invariant aspect ratio (n_layer = n_row)" if args.scan_aspect
                              else "fixed layer count (n_layer = 5), growing width"),
        "compaction_porosity": args.compaction_porosity,
        "reynolds_evaluated_on": "prototype geometry (not the millimetre DEM cell)",
    }
    metrics["dem_ensemble"] = [
        {
            "particle_count": int(d["config"]["particle_count"]),
            "scan_spec": d.get("scan_spec", {}),
            "config": d.get("config", {}),
            "E_oed": float(d["E_oed"]),
            "E_oed_mid": float(d.get("E_oed_mid", float("nan"))),
            "k_hyd": float(d["k_hyd"]),
            "porosity": float(d["porosity"]),
            "coordination": float(d["coordination"]),
            "strain_increment": float(d["strain_increment"]),
            "total_steps": int(d.get("total_steps", 0)),
            "wall_time_s": float(d.get("wall_time_s", 0.0)),
            "stages": d.get("stages", []),
        }
        for d in ensemble
    ]

    _log("[4/6] export r_u(T_v) interface table")
    table_path = ART / (f"ru_table_{args.tag}.json" if args.tag else "ru_table.json")
    info = write_ru_table(table_path, metrics["r_u_table"])
    metrics["r_u_table_file"] = {
        "path": str(table_path.relative_to(REPO)),
        "sha256": info["sha256"],
        "bytes": info["bytes"],
        "schema": metrics["r_u_table"]["schema"],
        "recomputed_sha256_matches": table_sha256(metrics["r_u_table"]) == info["sha256"],
    }
    _log(f"    {info['path']}  sha256={info['sha256'][:16]}...")

    _log("[5/6] downstream consumer end-to-end assertion (C4)")
    consumer_path = ART / (f"consumer_assertions_{args.tag}.json" if args.tag
                           else "consumer_assertions.json")
    if args.skip_consumer:
        metrics["C4"] = {"ran": False, "reason": "--skip-consumer"}
    else:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "in99_n08_liquefaction_consumer.py"),
             "--table", str(table_path), "--out", str(consumer_path)],
            capture_output=True, text=True, env=env,
        )
        _log(proc.stdout.strip()[-4000:])
        if proc.returncode != 0:
            _log(proc.stderr[-4000:])
            metrics["C4"] = {"ran": False, "returncode": proc.returncode,
                             "stderr": proc.stderr[-2000:]}
        else:
            payload = json.loads(consumer_path.read_text())
            metrics["C4"] = {
                "ran": True,
                "table_read": payload["table"]["path"],
                "table_sha256": payload["table"]["sha256"],
                "assertions": payload["assertions"],
                "all_pass": payload["all_pass"],
                "liquefaction_window_T_v": payload.get("liquefaction_window_T_v"),
            }
            _log(f"    consumer assertions all pass = {payload['all_pass']}")

    _log("[6/6] figures")
    _figsuffix = f"_{args.tag}" if args.tag else ""
    def _fig(name):
        return FIG / f"{Path(name).stem}{_figsuffix}.png"
    figs = [
        ("terzaghi_u.png", figure_terzaghi(metrics, _fig("terzaghi_u.png")),
         "C1: two-field column vs Fourier anchor; right panel: negative control"),
        ("settlement_time.png", figure_settlement(metrics, _fig("settlement_time.png")),
         "C2: settlement-time curve vs the analytic one-dimensional compression"),
        ("ru_profiles.png", figure_profiles(metrics, _fig("ru_profiles.png")),
         "excess pore-pressure profiles at the five C1 time factors"),
        ("dem_oedometer.png", figure_dem(ensemble, metrics, _fig("dem_oedometer.png")),
         "DEM oedometer stages and domain-size scan"),
        ("ru_interface.png", figure_interface(metrics["r_u_table"], _fig("ru_interface.png")),
         "published r_u(T_v) interface table"),
    ]
    metrics["figures"] = [{"name": n, "file": f, "caption": c} for n, f, c in figs]

    gates = {
        "C1_analytic_le5pct": metrics["C1"]["pass"],
        "C2_cv_le10pct_and_sf_le5pct": metrics["C2"]["pass"],
        "C3_drift_le5pct": metrics["C3"]["pass"],
        "C4_consumer_end_to_end": bool(metrics["C4"].get("all_pass")),
        "C5_honest_record": True,
        "applicability_fail_closed": metrics["applicability"]["applicable"],
        "negative_control_discriminates": all(n["would_fail_C1"] for n in metrics["negative_control"]),
    }
    metrics["gates"] = gates
    metrics["verdict"] = "PASS" if all(gates.values()) else "FAIL"

    suffix = f"_{args.tag}" if args.tag else ""
    curves_path = ART / f"curves{suffix}.json"
    curves_path.write_text(json.dumps({
        "schema": "tensordem.consolidation.raw_curves/1",
        "note": "raw simulated curves kept for every reported gate (C5)",
        "reference": {
            "C_v_assembled": metrics["C_v_assembled"],
            "history": metrics["column"]["raw_curves"],
            "samples": metrics["column"]["samples"],
        },
        "dem_scan": metrics["C3"]["dem_particle_scan"]["curves"],
        "dem_stages": metrics["dem_ensemble"],
    }, indent=2) + "\n")
    metrics["raw_curves_file"] = str(curves_path.relative_to(REPO))
    metrics_path = ART / f"metrics{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    _log("-" * 78)
    for k, v in gates.items():
        _log(f"    {k:40s} {'PASS' if v else 'FAIL'}")
    _log(f"    VERDICT: {metrics['verdict']}")
    _log(f"    metrics -> {metrics_path}")
    return 0 if metrics["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())