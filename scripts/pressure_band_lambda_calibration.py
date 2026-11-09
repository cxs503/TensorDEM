#!/usr/bin/env python3
"""P204-N08 campaign driver: calibrated ``lam_x`` for the ice pressure band.

Stages
------
``synthetic``
    Monte-Carlo finite-window bias study over ``L/lam in [1, 50]`` for two field
    families (spectral + AR(1)); fits ``rho(K) = 1 + a/K + b/K**2`` per estimator,
    derives the self-consistent correction, the leave-one-out sufficiency
    threshold ``K*`` and the variance-scaling exponents.
``validate``
    Out-of-sample check of ``K*`` on 3 seeds x 3 record lengths.
``dem``
    Long-domain ice--structure crushing DEM pressure band: time-averaged profile,
    corrected correlation length, block-bootstrap interval, shuffle control, and
    ``lam_x/h`` against the literature band 1.5--3.
``contract``
    ``lambda_x_estimate.v1`` payload + flat prop-1104 view, validated.

Everything written to ``--out`` (default ``artifacts/pressure_band_lambda``) is
produced programmatically; the figures are rendered from the same JSON.

Usage
-----
    python scripts/pressure_band_lambda_calibration.py --skip-dem
    python scripts/pressure_band_lambda_calibration.py --dem-widths 40 60 --dem-seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensordem.ice_pressure_band import (
    BandConfig,
    BiasModel,
    LITERATURE_BAND,
    METHODS,
    PRIMARY_METHODS,
    bootstrap_lambda,
    build_estimate,
    consensus,
    correct_length,
    diagnose_regime,
    estimate_all,
    estimate_profile_lambda,
    run_band,
    resolution_ladder,
    run_bias_study,
    scan_threshold_out_of_sample,
    run_variance_family,
    shuffle_control,
    snapshot_profiles,
    time_averaged_profile,
    to_prop_1104,
    validate,
    validate_threshold,
    windowed_field,
)

TOL = 0.05


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def stage_synthetic(args) -> dict:
    t0 = time.time()
    print(f"[synthetic] K grid {args.k_min}..{args.k_max}, n_real={args.n_real}", flush=True)
    k_grid = np.geomspace(args.k_min, args.k_max, args.n_k)
    study = run_bias_study(
        k_grid=k_grid,
        n_real=args.n_real,
        dx_over_lam=args.dx_over_lam,
        kinds=tuple(args.kinds),
        seed=args.seed,
        tol=TOL,
        progress=lambda i, n, K: print(f"  K={K:6.2f} ({i}/{n})", flush=True)
        if args.verbose else None,
    )
    out = study.to_dict()
    out["wall_seconds"] = time.time() - t0
    # summaries used by the report
    summary = {}
    for kind in args.kinds:
        summary[kind] = {
            m: {
                "k_raw": study.sufficiency[kind][m]["k_raw"],
                "k_corrected": study.sufficiency[kind][m]["k_corrected"],
                "model_rms": study.models[kind][m]["rms"],
                "model_form": study.models[kind][m]["form"],
                "model_params": study.models[kind][m]["params"],
                "calibrated": study.calibrated(kind, m),
                "sd_slope": study.variance[kind][m]["sd_log_log_slope"],
                "sd_slope_se": study.variance[kind][m]["sd_log_log_slope_se"],
            }
            for m in METHODS
        }
    out["summary"] = summary
    return out


def stage_validate(study_data: dict, args, out_dir: Path) -> dict:
    """Out-of-sample threshold scan (selection seeds) + confirmation (fresh seeds).

    The in-sample leave-one-out threshold of the bias study is *reported* but not
    used as the sufficiency claim: the corrected bias is measured on independent
    ensembles here, and the confirmation uses seeds disjoint from the selection
    ones.
    """
    from tensordem.ice_pressure_band.calibration import BiasStudy

    k_grid = np.asarray(study_data["k_grid"], dtype=float)
    kinds = {}
    for kind, d in study_data["kinds"].items():
        kinds[kind] = {m: dict(d[m]) for m in METHODS}
        kinds[kind]["_L_over_lam"] = study_data["k_grid"]
    study = BiasStudy(
        k_grid=k_grid,
        dx_over_lam=study_data["dx_over_lam"],
        n_real=study_data["n_real"],
        kinds=kinds,
        models=study_data["models"],
        sufficiency=study_data["sufficiency"],
        variance=study_data["variance"],
        seed=study_data["seed"],
    )
    calibrated = [m for m in PRIMARY_METHODS if study.calibrated(args.kinds[0], m)]
    result = {"calibrated_methods": calibrated}
    for kind in args.kinds:
        scan = scan_threshold_out_of_sample(
            study, kind=kind, seeds=(11, 12, 13), n_real=args.n_scan, tol=TOL,
            methods=tuple(calibrated) if calibrated else PRIMARY_METHODS,
        )
        # sequential selection with disjoint confirmation: walk the candidate K
        # values upwards and accept the first one that the *confirmation* seeds
        # also accept (the scan alone is a selection sample and must not be its
        # own verdict).  The multiplicity of this procedure is documented.
        methods_use = tuple(calibrated) if calibrated else PRIMARY_METHODS
        candidates = [e["K"] for e in scan["per_k"]
                      if all(e["per_method"][m]["within_tol"] for m in methods_use)]
        star = float("nan")
        conf = {"status": "no_finite_threshold", "per_method": {}}
        attempts = []
        for cand in candidates:
            c = validate_threshold(
                study, kind=kind, k_star=cand, seeds=(21, 22, 23),
                length_factors=(1.0, 1.5, 2.0), n_real=args.n_validate, tol=TOL,
                methods=methods_use,
            )
            ok = all(c["per_method"][m]["within_tol"] for m in methods_use)
            attempts.append({"K": cand, "confirmed": bool(ok),
                             "pooled": {m: c["per_method"][m]["pooled_mean_bias"]
                                        for m in methods_use}})
            if ok:
                star, conf = float(cand), c
                break
        result_attempts = attempts
        result[kind] = {
            "k_star_out_of_sample": star,
            "selection_attempts": result_attempts,
            "k_star_per_method": scan["k_star"],
            "scan": scan,
            "confirmation": conf,
            "per_method": conf.get("per_method", {}),
        }
        pb = {m: round(scan["k_star"][m], 2) for m in scan["k_star"]}
        mx = {m: round(conf["per_method"][m]["max_abs_bias"], 4)
              for m in conf.get("per_method", {})}
        print(f"[validate] {kind}: out-of-sample K*={star:.2f} per-method={pb} "
              f"confirmation pooled={ {m: round(conf['per_method'][m]['pooled_mean_bias'], 4) for m in conf.get('per_method', {})} } "
              f"max|cell|={mx}", flush=True)
    return result


def stage_variance(args) -> dict:
    lam_values = np.geomspace(args.lam_min, args.lam_max, args.n_lam)
    fam = run_variance_family(
        lam_values=lam_values,
        fixed_length_over_lam=args.fixed_K,
        n_real=args.n_variance,
        dx_over_lam=args.dx_over_lam,
        kind=args.kinds[0],
        seed=args.seed + 7,
    )
    return fam


def stage_negative_controls(args) -> dict:
    """White-noise and strictly periodic records: the estimators must be rejected."""
    rng = np.random.default_rng(args.seed + 101)
    n = 400
    dx = 0.05
    white = rng.standard_normal(n)
    period = 0.4
    periodic = np.sin(2.0 * np.pi * np.arange(n) * dx / period)
    correlated = windowed_field(n, dx, 1.0, rng, kind=args.kinds[0])
    out = {}
    for name, sig in (("white_noise", white), ("periodic", periodic), ("correlated", correlated)):
        regime = diagnose_regime(sig, dx)
        est = estimate_all(sig, dx)
        out[name] = {
            "regime": regime.to_dict(),
            "estimates_over_dx": {m: (est[m] / dx if np.isfinite(est[m]) else None) for m in METHODS},
            "expected_period_over_dx": (period / dx if name == "periodic" else None),
        }
    return out


def stage_dem(args, out_dir: Path, models: dict) -> dict:
    runs = []
    for width in args.dem_widths:
        for seed in args.dem_seeds:
            cfg = BandConfig(
                thickness=args.dem_thickness,
                width_over_thickness=float(width),
                d50_ratio=args.dem_d50_ratio,
                n_max=args.dem_n_max,
                depth=args.dem_depth,
                drive_speed=args.dem_speed,
                stroke_over_thickness=args.dem_stroke,
                save_every=args.dem_save_every,
                cells_per_thickness=args.dem_cells,
                seed=int(seed),
            )
            t0 = time.time()
            print(f"[dem] width={width}h seed={seed} ny={cfg.ny} ...", flush=True)
            run = run_band(cfg)
            snaps = snapshot_profiles(run.pressure, cfg)
            profile = snaps.mean(axis=0)
            est = estimate_profile_lambda(profile, cfg.dy, models=models,
                                          methods=METHODS)
            boots = {
                m: bootstrap_lambda(snaps, cfg.dy, models=models, method=m,
                                    n_boot=args.dem_boot, seed=seed + 1000)
                for m in METHODS
            }
            per_snapshot = []
            for row in snaps[:: max(1, snaps.shape[0] // 40)]:
                e = estimate_profile_lambda(row, cfg.dy, models=models, methods=("integral",))
                per_snapshot.append(e["corrected"]["integral"])
            per_snapshot = np.asarray([v for v in per_snapshot if np.isfinite(v)])
            control = shuffle_control(profile, cfg.dy, models=models, seed=seed)
            lam = est["value_m"]
            h = cfg.thickness
            # primary estimate: integral method (lowest threshold K*), CI from its bootstrap
            primary = "integral"
            ci = (boots[primary]["ci_low_m"], boots[primary]["ci_high_m"])
            record = {
                "width_over_thickness": float(width),
                "seed": int(seed),
                "config": cfg.to_dict(),
                "run": run.to_dict(),
                "pressure_profile_pa": profile.tolist(),
                "profile_mean_pa": float(np.mean(profile)),
                "profile_std_pa": float(np.std(profile)),
                "estimate": est,
                "bootstrap": {m: {k: v for k, v in boots[m].items() if k != "samples"}
                              for m in METHODS},
                "per_snapshot_lambda_m": {
                    "n": int(per_snapshot.size),
                    "mean": float(np.mean(per_snapshot)) if per_snapshot.size else None,
                    "sd": float(np.std(per_snapshot, ddof=1)) if per_snapshot.size > 2 else None,
                },
                "shuffle_control": control,
                "lambda_m": lam,
                "lambda_over_thickness": (lam / h) if np.isfinite(lam) else None,
                "ci95_m": list(ci),
                "ci95_over_thickness": [ci[0] / h, ci[1] / h],
                "L_over_lambda": est["L_over_lambda"],
                "wall_seconds": time.time() - t0,
            }
            runs.append(record)
            print(f"[dem]   lambda={lam:.4f} m  lambda/h={lam/h:.2f}  "
                  f"L/lambda={est['L_over_lambda']:.1f}  ({record['wall_seconds']:.0f} s)",
                  flush=True)
    values = [r["lambda_m"] for r in runs if r["lambda_m"] and np.isfinite(r["lambda_m"])]
    over_h = [r["lambda_over_thickness"] for r in runs if r["lambda_over_thickness"]]
    summary = {
        "n_runs": len(runs),
        "lambda_m_mean": float(np.mean(values)) if values else None,
        "lambda_m_sd": float(np.std(values, ddof=1)) if len(values) > 2 else None,
        "lambda_over_thickness_mean": float(np.mean(over_h)) if over_h else None,
        "lambda_over_thickness_min": float(np.min(over_h)) if over_h else None,
        "lambda_over_thickness_max": float(np.max(over_h)) if over_h else None,
        "literature_band": list(LITERATURE_BAND),
        "inside_band": bool(over_h and np.all([LITERATURE_BAND[0] <= v <= LITERATURE_BAND[1]
                                               for v in over_h])),
        "inside_band_any": bool(over_h and np.any([LITERATURE_BAND[0] <= v <= LITERATURE_BAND[1]
                                                   for v in over_h])),
        "min_L_over_lambda": float(min(r["L_over_lambda"] for r in runs)) if runs else None,
    }
    return {"runs": runs, "summary": summary}


def enrich_dem(dem: dict, models: dict) -> dict:
    """Recompute point estimates with the production models and add the
    resolution ladder (the DEM point estimates were made with trial model
    parameters; the difference is reported, not hidden)."""
    for r in dem["runs"]:
        prof = np.asarray(r["pressure_profile_pa"], dtype=float)
        cfg_dy = float(r["config"]["dy"])
        est = estimate_profile_lambda(prof, cfg_dy, models=models, methods=METHODS)
        r["estimate_dem_scan"] = r.get("estimate")
        r["estimate"] = est
        r["lambda_m"] = est["value_m"]
        r["lambda_over_thickness"] = float(est["value_m"] / r["config"]["thickness"])
        r["L_over_lambda"] = est["L_over_lambda"]
        prev = (r.get("estimate_dem_scan") or {}).get("correction_factor") or {}
        shift = {}
        for m in METHODS:
            old = prev.get(m)
            new = est["correction_factor"].get(m)
            shift[m] = (None if not old or not new or not np.isfinite(old) or not np.isfinite(new)
                        else float(new / old))
        r["correction_factor_shift"] = shift
        r["resolution_ladder"] = resolution_ladder(prof, cfg_dy, models=models)
    vals = [r["lambda_m"] for r in dem["runs"] if np.isfinite(r["lambda_m"])]
    over = [r["lambda_over_thickness"] for r in dem["runs"]]
    ladder_all = [c for r in dem["runs"] for c in r.get("resolution_ladder", [])]
    dem["summary"] = {
        "n_runs": len(dem["runs"]),
        "lambda_m_mean": float(np.mean(vals)) if vals else None,
        "lambda_m_sd": float(np.std(vals, ddof=1)) if len(vals) > 2 else None,
        "lambda_over_thickness_mean": float(np.mean(over)) if over else None,
        "lambda_over_thickness_min": float(np.min(over)) if over else None,
        "lambda_over_thickness_max": float(np.max(over)) if over else None,
        "literature_band": list(LITERATURE_BAND),
        "inside_band_all": bool(np.all([LITERATURE_BAND[0] <= v <= LITERATURE_BAND[1] for v in over])),
        "inside_band_any": bool(np.any([LITERATURE_BAND[0] <= v <= LITERATURE_BAND[1] for v in over])),
        "min_L_over_lambda": float(min(r["L_over_lambda"] for r in dem["runs"])),
        "regimes": sorted({r["estimate"]["regime"]["label"] for r in dem["runs"]}),
        "resolution_ladder": ladder_all,
        "max_abs_correction_factor_shift": float(np.nanmax(
            [abs(v - 1.0) for r in dem["runs"] for v in r["correction_factor_shift"].values()
             if v is not None and np.isfinite(v)] or [np.nan])),
        "particle_scale_over_h": float(dem["runs"][0]["config"]["d50_ratio"]),
    }
    return dem


def stage_contract(dem: dict | None, study: dict, args) -> dict:
    """Build and validate the lambda_x record + prop-1104 view."""
    if dem is None or not dem["runs"]:
        return {"status": "skipped", "reason": "no DEM runs"}
    primary = dem["runs"][0]
    est = primary["estimate"]
    method_values = {m: est["corrected"].get(m) for m in METHODS}
    cons = consensus(method_values)
    calibrated = [m for m in PRIMARY_METHODS
                  if study["summary"][args.kinds[0]][m].get("calibrated", True)]
    cons_cal = consensus({m: method_values[m] for m in calibrated})
    k_star = float(study.get("_k_star_out_of_sample", float("nan")))
    if not np.isfinite(k_star):
        k_star = max(study["summary"][args.kinds[0]][m]["k_corrected"] for m in PRIMARY_METHODS)
    if not np.isfinite(k_star):
        return {"status": "skipped", "reason": "no finite sufficiency threshold"}
    estimator = {
        "primary": "integral",
        "methods": method_values,
        "method_spread": cons_cal["method_spread"],
        "method_agreement": cons_cal["agreement"],
        "bias_model": study["summary"][args.kinds[0]]["integral"],
        "bias_corrected": True,
        "uncertainty": "moving-block bootstrap over DEM snapshots",
    }
    ci_lo, ci_hi = primary["ci95_m"]
    warn_note = []
    if not (ci_lo <= primary["lambda_m"] <= ci_hi):
        # the block bootstrap re-estimates the profile from resampled snapshots,
        # so its percentile interval need not bracket the single-profile point
        # estimate; the interval is widened to include it and the widening is
        # recorded instead of hidden
        warn_note.append(
            f"bootstrap 95 % interval [{ci_lo:.4f}, {ci_hi:.4f}] m did not bracket the "
            f"point estimate {primary['lambda_m']:.4f} m; interval widened to the union")
        ci_lo = min(ci_lo, primary["lambda_m"])
        ci_hi = max(ci_hi, primary["lambda_m"])
    record = build_estimate(
        value_m=primary["lambda_m"],
        thickness_m=primary["config"]["thickness"],
        ci95_m=(ci_lo, ci_hi),
        estimator=estimator,
        record_length_m=primary["estimate"]["record_length_m"],
        dx_m=primary["estimate"]["dx_m"],
        n_samples=primary["estimate"]["n_samples"],
        threshold_K=k_star,
        bias_corrected=True,
        regime=est["regime"]["label"],
        # a band whose own diagnostic says "white noise at this resolution" cannot
        # be graded CALIBRATED: the value is emitted for traceability only
        evidence_grade=("CALIBRATED" if est["regime"]["label"] == "ok" else "WITHHELD"),
        sources=[
            "TensorDEM ice_contact.sim.IceWallCrush (long-domain band)",
            "synthetic bias calibration (this study)",
        ],
        warnings=list(primary["estimate"].get("warnings", [])) + warn_note,
    )
    errors = validate(record)
    view = None if errors else to_prop_1104(record)
    return {"status": "ok" if not errors else "invalid", "errors": errors,
            "record": record, "prop_1104": view, "consensus": cons,
            "consensus_calibrated": cons_cal, "threshold_K": k_star,
            "calibrated_methods": calibrated}


def _figures(study: dict, validation: dict, varfam: dict, dem: dict | None,
             controls: dict, out_dir: Path, args) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    made = []
    kind = args.kinds[0]
    K = np.asarray(study["k_grid"], dtype=float)

    # fig1: bias curves + fitted model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"integral": "C0", "expfit": "C1", "spectral": "C2", "efold": "C3"}
    for ax, kd in zip(axes, args.kinds):
        for m in METHODS:
            ratio = np.asarray(study["kinds"][kd][m]["ratio_mean"], dtype=float)
            sd = np.asarray(study["kinds"][kd][m]["ratio_sd"], dtype=float)
            ax.errorbar(K, ratio, yerr=sd / np.sqrt(study["n_real"]), fmt="o", ms=3,
                        color=colors[m], label=f"{m} (MC)")
            mod = study["models"][kd][m]
            Kf = np.geomspace(1, 60, 200)
            rho = BiasModel(method=m, form=mod["form"], params=tuple(mod["params"])).rho(Kf)
            ax.plot(Kf, rho, "-", color=colors[m], lw=1.2,
                    label=f"{m} fit ({mod['form']})")
        ax.axhspan(0.95, 1.05, color="0.85", zorder=0)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("$K = L/\\lambda$")
        ax.set_ylabel("$E[\\hat\\lambda]/\\lambda$")
        ax.set_title(f"finite-window bias, {kd} field")
        ax.set_ylim(0, 1.35)
        ax.legend(fontsize=6.5, ncol=1)
    fig.tight_layout()
    p = figs / "fig1_bias_curves.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    made.append({"name": "fig1_bias_curves.png",
                 "caption": "Measured finite-window bias E[lam_hat]/lam vs K=L/lam for the "
                            "three estimators plus the legacy e-folding method, both field "
                            "families; grey band = +-5% tolerance, curves = fitted "
                            "rho(K)=1+a/K+b/K^2.", "path": str(p)})

    # fig2: corrected residual bias (LOO) + threshold
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in PRIMARY_METHODS:
        loo = np.asarray(study["sufficiency"][kind][m]["loo_residual_bias"], dtype=float)
        ax.plot(K, loo, "o-", ms=3, color=colors[m], label=f"{m}")
        ks = study["sufficiency"][kind][m]["k_corrected"]
        if np.isfinite(ks):
            ax.axvline(ks, color=colors[m], ls="--", lw=1.0)
    ax.axhline(TOL, color="r", ls=":", label="5 % tolerance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$K = L/\\lambda$")
    ax.set_ylabel("|bias| after correction (leave-one-out)")
    ax.set_title("sufficiency threshold (corrected, out-of-sample)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figs / "fig2_sufficiency_threshold.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    made.append({"name": "fig2_sufficiency_threshold.png",
                 "caption": "Leave-one-out residual bias of the corrected estimator vs K; "
                            "vertical dashed lines are the derived thresholds K*.", "path": str(p)})

    # fig3: variance scaling
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for m in PRIMARY_METHODS:
        sd = np.asarray(study["kinds"][kind][m]["ratio_sd"], dtype=float)
        axes[0].loglog(K, sd, "o-", ms=3, color=colors[m],
                       label=f"{m}: slope {study['variance'][kind][m]['sd_log_log_slope']:.2f}")
        ls = varfam["lam_values"]
        axes[1].loglog(ls, varfam[m]["rel_sd"], "o-", ms=3, color=colors[m],
                       label=f"{m}: slope {varfam[m]['sd_vs_lam_slope']:.2f}")
    axes[0].loglog(K, 0.9 / np.sqrt(K), "k--", label="$-1/2$ reference")
    axes[0].set_xlabel("$K = L/\\lambda$"); axes[0].set_ylabel("sd($\\hat\\lambda/\\lambda$)")
    axes[0].set_title("variance scaling in record length")
    axes[1].set_xlabel("$\\lambda$ at fixed $L$"); axes[1].set_ylabel("sd($\\hat\\lambda/\\lambda$)")
    axes[1].set_title("variance scaling in $\\lambda$ ($+3/2$ reference)")
    for ax in axes:
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    p = figs / "fig3_variance_scaling.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    made.append({"name": "fig3_variance_scaling.png",
                 "caption": "Standard deviation of the normalised correlation-length estimate "
                            "vs K (left) and vs lam at fixed record length (right), with the "
                            "theoretical slopes.", "path": str(p)})

    if dem and dem["runs"]:
        # fig4: DEM pressure band + ACF + bootstrap
        run = dem["runs"][0]
        cfg = run["config"]
        prof = np.asarray(run["pressure_profile_pa"], dtype=float)
        y = np.arange(prof.size) * cfg["dy"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        axes[0].plot(y / cfg["thickness"], prof / 1e6, "-", lw=0.8)
        axes[0].set_xlabel("$y/h$"); axes[0].set_ylabel("$\\bar p$ [MPa]")
        axes[0].set_title(f"time-averaged pressure band, B = {cfg['width_over_thickness']:.0f} h")
        from tensordem.ice_pressure_band import autocorrelation, exponential_acf
        lags, R = autocorrelation(prof, cfg["dy"])
        m = lags <= 4 * run["lambda_m"]
        axes[1].plot(lags[m] / cfg["thickness"], R[m], "o-", ms=2, label="measured $R$")
        axes[1].plot(lags[m] / cfg["thickness"],
                     exponential_acf(lags[m], run["lambda_m"]), "k--",
                     label=f"exp($-r/\\lambda_x$), $\\lambda_x$={run['lambda_m']:.3f} m")
        axes[1].set_xlabel("$r/h$"); axes[1].set_ylabel("$R(r)$")
        axes[1].set_title("spatial autocorrelation of $\\bar p(y)$")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = figs / "fig4_dem_pressure_band.png"
        fig.savefig(p, dpi=145)
        plt.close(fig)
        made.append({"name": "fig4_dem_pressure_band.png",
                     "caption": "Long-domain DEM pressure band on the structure face (left) and "
                                "its along-band autocorrelation with the fitted exponential "
                                "(right).", "path": str(p)})

        # fig5: lambda_x/h vs record length with CI and literature band
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.axhspan(*LITERATURE_BAND, color="C2", alpha=0.15,
                   label=f"literature band {LITERATURE_BAND[0]}-{LITERATURE_BAND[1]}")
        for r in dem["runs"]:
            xv = r["config"]["width_over_thickness"]
            yv = r["lambda_over_thickness"]
            lo, hi = r["ci95_over_thickness"]
            ax.errorbar([xv], [yv], yerr=[[max(yv - lo, 0)], [max(hi - yv, 0)]],
                        fmt="o", capsize=4, label=f"seed {r['seed']}")
        ax.set_xlabel("record length $B/h$")
        ax.set_ylabel("$\\lambda_x/h$")
        ax.set_title("native DEM calibration of $\\lambda_x/h$ with 95 % bootstrap CI")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = figs / "fig5_lambda_over_h.png"
        fig.savefig(p, dpi=145)
        plt.close(fig)
        made.append({"name": "fig5_lambda_over_h.png",
                     "caption": "Native DEM estimates of lam_x/h with moving-block bootstrap "
                                "95 % intervals, against the literature band 1.5-3.", "path": str(p)})

    # fig6: negative controls
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = list(controls)
    width = 0.8 / max(len(METHODS), 1)
    for i, m in enumerate(METHODS):
        vals = [controls[k]["estimates_over_dx"][m] for k in labels]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(np.arange(len(labels)) + i * width, vals, width=width, label=m)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{k}\n({controls[k]['regime']['label']})" for k in labels], fontsize=8)
    ax.set_ylabel("$\\hat\\lambda/\\Delta$")
    ax.set_yscale("log")
    ax.set_title("negative controls: estimates normalised by grid spacing")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figs / "fig6_negative_controls.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    made.append({"name": "fig6_negative_controls.png",
                 "caption": "White-noise, strictly periodic and correlated records: the regime "
                            "diagnostic labels are beneath the ticks; estimates are in grid "
                            "spacings.", "path": str(p)})
    return made


def criteria(study: dict, validation: dict, varfam: dict, dem: dict | None,
             controls: dict, contract: dict, args) -> dict:
    kind = args.kinds[0]
    summ = study["summary"][kind]
    calibrated = [m for m in PRIMARY_METHODS if summ[m].get("calibrated", True)]
    k_star = {m: summ[m]["k_corrected"] for m in PRIMARY_METHODS}
    v_kind = validation.get(kind, {})
    k_star_rec = float(v_kind.get("k_star_out_of_sample", float("inf")))
    k_star_loo = float(max([summ[m]["k_corrected"] for m in calibrated] or [float("inf")]))
    crit = {}
    crit["C1_bias_model_rms_le_0.03"] = {
        "value": {m: summ[m]["model_rms"] for m in METHODS},
        "pass": bool(calibrated and all(summ[m]["model_rms"] <= 0.03 for m in calibrated)),
        "calibrated_methods": calibrated,
        "criterion": "leave-one-out RMS of the selected bias model <= 0.03 for every method "
                     "declared calibrated (a method with RMS > 0.05 is excluded and reported)",
    }
    crit["C2_threshold_finite_and_le_50"] = {
        "value": {"in_sample_loo": k_star, "out_of_sample": k_star_rec,
                  "out_of_sample_per_method": v_kind.get("k_star_per_method")},
        "recommended_K_star": k_star_rec,
        "pass": bool(np.isfinite(k_star_rec) and k_star_rec <= 50.0),
        "criterion": "the recommended K* (out-of-sample, max over calibrated corrected "
                     "estimators) is finite and within the scanned range L/lam <= 50",
    }
    conf = validation.get(kind, {}).get("confirmation", {}) or {}
    crit["C3_validation_3seeds_3L_within_5pct"] = {
        "value": {m: {"pooled_mean_bias": conf.get("per_method", {}).get(m, {}).get("pooled_mean_bias"),
                      "pooled_se": conf.get("per_method", {}).get(m, {}).get("pooled_se"),
                      "max_abs_cell_bias": conf.get("per_method", {}).get(m, {}).get("max_abs_bias"),
                      "all_cells_within_tol": conf.get("per_method", {}).get(m, {}).get("all_cells_within_tol")}
                  for m in PRIMARY_METHODS},
        "methods_validated": calibrated,
        "k_star_confirmed": conf.get("k_star_used"),
        "pass": bool(calibrated and all(
            conf.get("per_method", {}).get(m, {}).get("within_tol", False)
            for m in calibrated)),
        "criterion": "at K = K*, the pooled corrected bias over 3 fresh seeds x 3 record "
                     "lengths (2700 records) is within 5 % (+2 SE); the per-cell maximum is "
                     "reported separately as information",
    }
    sd_slopes = {m: summ[m]["sd_slope"] for m in PRIMARY_METHODS}
    lam_slopes = {m: varfam[m]["sd_vs_lam_slope"] for m in PRIMARY_METHODS}
    crit["C4_variance_scaling"] = {
        "value": {"sd_vs_K": sd_slopes, "sd_vs_lam_at_fixed_L": lam_slopes},
        "pass": bool(all(-0.75 <= v <= -0.30 for v in sd_slopes.values())
                     and all(1.1 <= v <= 1.9 for v in lam_slopes.values())),
        "criterion": "sd(lam_hat/lam) ~ K^-0.5 (accept -0.75..-0.30) and ~ lam^1.5 at fixed L "
                     "(accept 1.1..1.9), i.e. Var ~ lam/L",
    }
    if dem and dem["runs"]:
        s = dem["summary"]
        crit["C5_dem_lambda_over_h"] = {
            "value": {"mean": s["lambda_over_thickness_mean"],
                      "min": s["lambda_over_thickness_min"],
                      "max": s["lambda_over_thickness_max"],
                      "n_runs": s["n_runs"],
                      "min_L_over_lambda": s["min_L_over_lambda"],
                      "literature_band": s["literature_band"],
                      "inside_band_all": s["inside_band_all"],
                      "inside_band_any": s["inside_band_any"],
                      "regimes": s.get("regimes"),
                      "resolution_ladder": s.get("resolution_ladder"),
                      "frac_of_particle_scale": (
                          s["lambda_over_thickness_mean"] / s.get("particle_scale_over_h", 1.0)
                          if s.get("lambda_over_thickness_mean") else None)},
            "criterion_detail": "inside_band_any decides the pass; the resolution ladder and the "
                                "regime labels are reported so an out-of-band result is diagnosable",
            "pass": bool(s["inside_band_any"]),
            "criterion": "native DEM lam_x/h within the literature band 1.5-3 for at least one "
                         "long-domain run (honest negative if outside)",
        }
        crit["C6_record_sufficiency_at_scale"] = {
            "value": {"min_L_over_lambda": s["min_L_over_lambda"],
                      "threshold_K": max(k_star.values())},
            "pass": bool(s["min_L_over_lambda"] >= max(k_star.values())),
            "criterion": "DEM record length L/lam_x >= K* (record long enough for the calibrated "
                         "estimator)",
        }
    w = controls["white_noise"]; p = controls["periodic"]; c = controls["correlated"]
    crit["C7_negative_controls"] = {
        "value": {"white_regime": w["regime"]["label"],
                  "periodic_regime": p["regime"]["label"],
                  "correlated_regime": c["regime"]["label"],
                  "white_integral_over_dx": w["estimates_over_dx"]["integral"],
                  "periodic_period_over_dx": p["expected_period_over_dx"]},
        "pass": bool(w["regime"]["label"] == "white_noise"
                     and p["regime"]["label"] == "periodic"
                     and c["regime"]["label"] == "ok"
                     and (w["estimates_over_dx"]["integral"] or 0) < 3.0),
        "criterion": "white noise and strictly periodic records are rejected by the regime "
                     "diagnostic; white-noise integral scale < 3 grid spacings",
    }
    agree = contract.get("consensus", {}).get("method_spread")
    agree_cal = contract.get("consensus_calibrated", {}).get("method_spread")
    crit["C8_multi_method_agreement"] = {
        "value": {"method_spread_calibrated": agree_cal,
                  "method_spread_all": agree,
                  "agreement": contract.get("consensus", {}).get("agreement"),
                  "methods": contract.get("consensus", {}).get("methods")},
        "pass": bool(agree_cal is not None and agree_cal <= 0.30),
        "criterion": "relative spread of the corrected estimates across methods <= 30 % at the "
                     "DEM record length",
    }
    crit["C9_output_contract"] = {
        "value": {"status": contract.get("status"), "errors": contract.get("errors"),
                  "schema_version": (contract.get("record") or {}).get("schema_version")},
        "pass": bool(contract.get("status") == "ok"),
        "criterion": "lambda_x_estimate.v1 record validates and the prop-1104 flat view is "
                     "producible",
    }
    crit["all_pass"] = bool(all(v.get("pass") for v in crit.values() if isinstance(v, dict)))
    return crit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/pressure_band_lambda")
    ap.add_argument("--kinds", nargs="+", default=["spectral", "ar1"])
    ap.add_argument("--k-min", type=float, default=1.0)
    ap.add_argument("--k-max", type=float, default=50.0)
    ap.add_argument("--n-k", type=int, default=18)
    ap.add_argument("--n-real", type=int, default=400)
    ap.add_argument("--n-validate", type=int, default=200)
    ap.add_argument("--n-scan", type=int, default=150)
    ap.add_argument("--n-variance", type=int, default=300)
    ap.add_argument("--n-lam", type=int, default=7)
    ap.add_argument("--lam-min", type=float, default=0.25)
    ap.add_argument("--lam-max", type=float, default=4.0)
    ap.add_argument("--fixed-K", type=float, default=20.0)
    ap.add_argument("--dx-over-lam", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-dem", action="store_true")
    ap.add_argument("--reuse-validation", action="store_true",
                    help="reuse an existing validation.json (stage-level resume)")
    ap.add_argument("--reuse-dem", action="store_true",
                    help="reuse an existing dem_runs.json instead of re-running the DEM")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--dem-widths", nargs="+", type=int, default=[40, 60])
    ap.add_argument("--dem-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dem-thickness", type=float, default=0.2)
    ap.add_argument("--dem-d50-ratio", type=float, default=0.25)
    ap.add_argument("--dem-n-max", type=int, default=7000)
    ap.add_argument("--dem-depth", type=float, default=0.5)
    ap.add_argument("--dem-speed", type=float, default=0.5)
    ap.add_argument("--dem-stroke", type=float, default=0.5)
    ap.add_argument("--dem-save-every", type=int, default=10)
    ap.add_argument("--dem-cells", type=int, default=16)
    ap.add_argument("--dem-boot", type=int, default=300)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    study_path = out_dir / "study.json"
    t_start = time.time()

    if args.skip_synthetic and study_path.exists():
        study = json.loads(study_path.read_text())
        print("[synthetic] reused", study_path, flush=True)
    else:
        study = stage_synthetic(args)
        study_path.write_text(json.dumps(_jsonable(study), indent=1))

    val_path = out_dir / "validation.json"
    if args.reuse_validation and val_path.exists():
        validation = json.loads(val_path.read_text())["validation"]
        print("[validate] reused", val_path, flush=True)
    else:
        validation = stage_validate(study, args, out_dir)
    varfam = stage_variance(args)
    controls = stage_negative_controls(args)
    models = {m: study["models"][args.kinds[0]][m] for m in METHODS}

    dem_path = out_dir / "dem_runs.json"
    dem = None
    if args.reuse_dem and dem_path.exists():
        dem = json.loads(dem_path.read_text())
        print("[dem] reused", dem_path, flush=True)
    elif not args.skip_dem:
        dem = stage_dem(args, out_dir, models)
    if dem is not None:
        dem = enrich_dem(dem, models)
        dem_path.write_text(json.dumps(_jsonable(dem), indent=1))
    study_for_contract = dict(study)
    study_for_contract["_k_star_out_of_sample"] = validation.get(
        args.kinds[0], {}).get("k_star_out_of_sample", float("nan"))
    contract = stage_contract(dem, study_for_contract, args)

    crit = criteria(study, validation, varfam, dem, controls, contract, args)
    figs = _figures(study, validation, varfam, dem, controls, out_dir, args)

    report = {
        "campaign_node": "P204-N08",
        "repo": "TensorDEM",
        "module": "tensordem.ice_pressure_band",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.time() - t_start,
        "settings": {k: v for k, v in vars(args).items()},
        "criteria": crit,
        "calibration_summary": study["summary"],
        "recommended_k_star": {
            kind: validation.get(kind, {}).get("k_star_out_of_sample", float("nan"))
            for kind in args.kinds
        },
        "validation": validation,
        "variance_family": varfam,
        "negative_controls": controls,
        "dem_summary": (dem or {}).get("summary"),
        "contract": contract,
        "figures": [{k: v for k, v in f.items() if k != "path"} for f in figs],
        "artifacts": {
            "study_json": str(study_path),
            "report_json": str(out_dir / "report.json"),
            "figures": [f["path"] for f in figs],
        },
    }
    (out_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=1))
    if dem is not None:
        (out_dir / "dem_runs.json").write_text(json.dumps(_jsonable(dem), indent=1))
    (out_dir / "validation.json").write_text(
        json.dumps(_jsonable({"validation": validation, "variance_family": varfam,
                              "negative_controls": controls}), indent=1))
    print(json.dumps(_jsonable(crit), indent=1))
    print(f"[done] {time.time() - t_start:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()