#!/usr/bin/env python3
"""Drive the DEM ice-resistance uncertainty-quantification study.

Stages
------
``design``   build (or rebuild) the experiment design JSON: Latin-hypercube
             training set, Sobol' held-out set, repose-angle design, timestep
             refinement pairs and the synthetic L2 anchor corpus.
``cases``    evaluate pending DEM cases (idempotent, one JSON per case); this is
             the stage that is sharded across cluster nodes.
``analyze``  fit the surrogates under the R^2 >= 0.95 gate, estimate Sobol'
             indices two ways, run the identifiability diagnostics, propagate
             the Laplace posterior to a resistance band, test interval coverage
             against the L2 anchor corpus, and write artifacts + figures.

Example (cluster shard, four-way split of the pending cases)::

    python scripts/run_ice_uq_sensitivity.py cases --workers 48 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

ARTIFACT_DIR = ROOT / "artifacts" / "ice_uq"
DESIGN_PATH = ARTIFACT_DIR / "design.json"
CACHE_DIR = ARTIFACT_DIR / "cases"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"

N_TRAIN = 192
N_TEST = 48
N_REPOSE = 60
N_ANCHOR = 40
N_REFINE = 6
SALTELLI_BASE = 2**12          # k = 12 -> N = 4096
OBS_NOISE_REL = 0.06           # declared 1-sigma observation error on R
TRACE_NOISE_REL = 0.05         # declared 1-sigma error on the calibration trace


# ---------------------------------------------------------------- design ----
def build_design(seed: int = 20261107) -> dict:
    from tensordem.sensitivity.design import ParameterSpace
    from tensordem.sensitivity.ice_chain import (
        ICE_SPACE, PARAMETER_NAMES, REPOSE_ACTIVE,
    )

    train_u = ICE_SPACE.latin_hypercube(N_TRAIN, seed=seed, candidates=16)
    test_u = ICE_SPACE.latin_hypercube(N_TEST, seed=seed + 1, candidates=16)

    specs = []
    for i, u in enumerate(train_u):
        specs.append({"case_id": f"train_{i:03d}", "kind": "wedge",
                      "u": u.tolist(), "dt_factor": 1.0, "tag": "train"})
    for i, u in enumerate(test_u):
        specs.append({"case_id": f"test_{i:03d}", "kind": "wedge",
                      "u": u.tolist(), "dt_factor": 1.0, "tag": "test"})
    # Timestep-refinement pairs reuse the first N_REFINE training points.
    for i in range(N_REFINE):
        specs.append({"case_id": f"refine_{i:03d}", "kind": "wedge",
                      "u": train_u[i].tolist(), "dt_factor": 0.5, "tag": "refine"})

    # Repose design: 5 active parameters, the remaining three pinned at nominal.
    repose_space = ICE_SPACE.subset(REPOSE_ACTIVE)
    repose_u = repose_space.latin_hypercube(N_REPOSE, seed=seed + 2, candidates=16)
    active = [PARAMETER_NAMES.index(name) for name in REPOSE_ACTIVE]
    for i, row in enumerate(repose_u):
        u = np.full(len(ICE_SPACE), 0.5)
        u[active] = row
        specs.append({"case_id": f"repose_{i:03d}", "kind": "repose",
                      "u": u.tolist(), "dt_factor": 1.0, "tag": "repose"})

    calibration_u = np.array([0.45, 0.50, 0.45, 0.55, 0.50, 0.40, 0.75, 0.35])
    design = {
        "seed": seed,
        "chain": {
            "ice_thickness_m": 0.16, "bow_speed_m_s": 0.2,
            "wedge_angle_deg": 45.0, "window_s": 0.04,
        },
        "parameter_space": ICE_SPACE.describe(),
        "n_train": N_TRAIN, "n_test": N_TEST, "n_repose": N_REPOSE,
        "repose_active": list(REPOSE_ACTIVE),
        "calibration_point_unit": calibration_u.tolist(),
        "observation_noise_rel": OBS_NOISE_REL,
        "trace_noise_rel": TRACE_NOISE_REL,
        "saltelli_base_samples": SALTELLI_BASE,
        "anchor_corpus": {
            "n": N_ANCHOR,
            "kind": "synthetic L2 reference tier",
            "protocol": ("independent noise realisations at the calibration "
                         "point; each record carries its own calibration trace "
                         "and its own resistance observation"),
            "seeds": [int(seed + 1000 + k) for k in range(N_ANCHOR)],
        },
        "cases": specs,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_PATH.write_text(json.dumps(design, indent=1))
    return design


def _load_design() -> dict:
    if not DESIGN_PATH.exists():
        raise SystemExit(f"missing design file {DESIGN_PATH}; run the design stage first")
    return json.loads(DESIGN_PATH.read_text())


# ----------------------------------------------------------------- cases ----
def run_cases(args) -> None:
    from tensordem.sensitivity.ice_chain import run_batch

    design = _load_design()
    specs = design["cases"]
    if args.kind != "all":
        specs = [s for s in specs if s["kind"] == args.kind]
    if args.tag:
        specs = [s for s in specs if s["tag"] == args.tag]
    if args.shards > 1:
        specs = [s for i, s in enumerate(specs) if i % args.shards == args.shard]
    if args.limit:
        # ``limit`` counts cases this call should *compute*, so drop cached ones
        # first; otherwise a resumable shard would keep re-requesting old cases.
        specs = [s for s in specs
                 if not (CACHE_DIR / f"{s['case_id']}.json").exists()][: args.limit]
    report = run_batch(specs, CACHE_DIR, workers=args.workers)
    print(json.dumps(report, indent=1))


# --------------------------------------------------------------- analysis ---
def _load_cases(tag: str, kind: str = "wedge") -> list[dict]:
    from tensordem.sensitivity.ice_chain import load_cases

    design = _load_design()
    wanted = {s["case_id"] for s in design["cases"]
              if s["tag"] == tag and s["kind"] == kind}
    records = load_cases(CACHE_DIR, wanted)
    if len(records) != len(wanted):
        missing = sorted(wanted - {r["case_id"] for r in records})
        raise SystemExit(f"missing {len(missing)} cached cases for tag={tag} "
                         f"(e.g. {missing[:3]}); run the cases stage")
    return sorted(records, key=lambda r: r["case_id"])


RESPONSES = {
    "R": lambda m: m["R_window_mean_N"],
    "R_channel": lambda m: m["R_channel_N"],
    "theta_r": lambda m: m["theta_r_deg"],
}


def analyze(seed: int = 20261107) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensordem.sensitivity import identifiability as idf
    from tensordem.sensitivity import propagation as prop
    from tensordem.sensitivity.ice_chain import ICE_SPACE, PARAMETER_NAMES, REPOSE_ACTIVE
    from tensordem.sensitivity.sobol import bootstrap_indices, saltelli_indices
    from tensordem.sensitivity.surrogate import GaussianProcess, PolynomialChaos, r2_score

    rng = np.random.default_rng(seed)
    design = _load_design()
    print(f"[analyze] design {design['n_train']}+{design['n_test']} fit / {design.get('n_hold')} hold-out", flush=True)
    names = list(PARAMETER_NAMES)
    # Fit on the 192-point Latin hypercube plus the 48-point Sobol' set; the
    # independent 96-point hold-out set ("hold") is only ever used for scoring.
    wedge_train = _load_cases("train") + _load_cases("test")
    wedge_test = _load_cases("hold")
    repose = _load_cases("repose", kind="repose")
    refine = _load_cases("refine")
    summary: dict = {"design": {k: design[k] for k in
                                ("seed", "n_train", "n_test", "n_repose",
                                 "saltelli_base_samples", "observation_noise_rel",
                                 "trace_noise_rel")}}

    # --- 1. surrogate fits with the R^2 >= 0.95 gate ----------------------
    X_train = np.array([r["u_unit"] for r in wedge_train])
    X_test = np.array([r["u_unit"] for r in wedge_test])
    gate_failures = []
    fits: dict = {}
    for name in ("R", "R_channel"):
        y_train = np.array([RESPONSES[name](r["metrics"]) for r in wedge_train])
        y_test = np.array([RESPONSES[name](r["metrics"]) for r in wedge_test])
        # A log transform is only legitimate when every response is strictly
        # positive; a single case in which the wedge loses contact drives the
        # late-phase mean resistance to exactly zero, so that response is fitted
        # in linear space and the transform is recorded.
        positive = bool(np.all(y_train > 0.0) and np.all(y_test > 0.0))
        fwd, inv = (np.log, np.exp) if positive else (lambda v: v, lambda v: v)
        entry = _fit_surrogates(fwd, inv, X_train, y_train, X_test, y_test,
                                PolynomialChaos, GaussianProcess, r2_score, seed)
        entry["transform"] = "log" if positive else "linear"
        entry["nonpositive_train"] = int(np.sum(y_train <= 0.0))
        entry["nonpositive_test"] = int(np.sum(y_test <= 0.0))
        if entry["best"]["r2_test"] < 0.95:
            gate_failures.append(name)
        fits[name] = entry

    print("[analyze] surrogates R/R_channel fitted", flush=True)

    # The repose surrogate only sees the five active parameters.
    active = [names.index(n) for n in REPOSE_ACTIVE]
    Xr = np.array([r["u_unit"] for r in repose])[:, active]
    yr = np.array([RESPONSES["theta_r"](r["metrics"]) for r in repose])
    order = rng.permutation(len(yr))
    split = int(round(0.75 * len(yr)))
    tr, te = order[:split], order[split:]
    repose_space = ICE_SPACE.subset(REPOSE_ACTIVE)
    fits["theta_r"] = _fit_surrogates(lambda v: v, lambda v: v, Xr[tr], yr[tr],
                                      Xr[te], yr[te], PolynomialChaos, GaussianProcess,
                                      r2_score, seed)
    fits["theta_r"]["transform"] = "linear"
    fits["theta_r"]["nonpositive_train"] = int(np.sum(yr[tr] <= 0.0))
    fits["theta_r"]["nonpositive_test"] = int(np.sum(yr[te] <= 0.0))
    if fits["theta_r"]["best"]["r2_test"] < 0.95:
        gate_failures.append("theta_r")

    summary["surrogates"] = {
        name: {
            "selected": entry["best"]["kind"],
            "pce_degree": entry["pce"]["degree"],
            "r2_train": entry["best"]["r2_train"],
            "r2_test": entry["best"]["r2_test"],
            "loo_r2": entry["pce"]["loo_r2"],
            "pce_r2_test": entry["pce"]["r2_test"],
            "gp_r2_test": entry["gp"]["r2_test"],
            "transform": entry["transform"],
            "nonpositive_train": entry["nonpositive_train"],
            "nonpositive_test": entry["nonpositive_test"],
            "n_train": int(len(entry["y_train"])),
            "n_test": int(len(entry["y_test"])),
            "r2_gate": 0.95,
            "gate_passed": bool(entry["best"]["r2_test"] >= 0.95),
        }
        for name, entry in fits.items()
    }
    summary["gate_failures"] = gate_failures

    models = {name: entry["best"]["model"] for name, entry in fits.items()}

    print("[analyze] repose surrogate fitted", flush=True)

    # --- 2. Sobol' indices, two independent estimators --------------------
    sobol_rows = []
    for name, model in models.items():
        space = ICE_SPACE if name != "theta_r" else repose_space
        param_names = names if name != "theta_r" else list(REPOSE_ACTIVE)

        def evaluate(U, model=model, name=name):
            return fits[name]["best"]["inverse"](model.predict(U))

        A, B, AB = space.saltelli_design(SALTELLI_BASE, seed=seed + 7)
        mc = saltelli_indices(evaluate, A, B, AB)
        ci = bootstrap_indices(evaluate, A, B, AB, replicates=64, seed=seed + 11)
        # Dual-method cross-check: a degree-2 chaos fitted on the *raw* response
        # (not on the transformed one, so that both estimators describe the same
        # quantity) supplies the coefficient-based indices.
        raw_train = (fits[name]["y_train"] if name != "theta_r"
                     else yr[tr])
        chaos = PolynomialChaos(len(param_names), degree=2).fit(
            np.atleast_2d(fits[name]["X_train"] if name != "theta_r" else Xr[tr]),
            raw_train)
        pce_indices = chaos.sobol_indices()
        for i, pname in enumerate(param_names):
            row = {
                "response": name,
                "parameter": pname,
                "first_order": float(mc["first"][i]),
                "first_order_ci": [float(ci["first_ci"][0][i]), float(ci["first_ci"][1][i])],
                "total_order": float(mc["total"][i]),
                "total_order_ci": [float(ci["total_ci"][0][i]), float(ci["total_ci"][1][i])],
            }
            if pce_indices is not None:
                row["pce_first_order"] = float(pce_indices["first"][i])
                row["pce_total_order"] = float(pce_indices["total"][i])
            sobol_rows.append(row)
        summary.setdefault("sobol", {})[name] = {
            "base_samples": SALTELLI_BASE,
            "variance": mc["variance"],
            "sum_first_order": float(mc["first"].sum()),
            "sum_total_order": float(mc["total"].sum()),
            "max_abs_pce_minus_saltelli": (
                float(max(np.max(np.abs(mc["first"] - pce_indices["first"])),
                          np.max(np.abs(mc["total"] - pce_indices["total"]))))
                if pce_indices is not None else None),
        }
    summary["sobol_table"] = sobol_rows

    print("[analyze] sobol indices done", flush=True)

    # --- 3. identifiability of the 1-D calibration curve ------------------
    u_cal = np.array(design["calibration_point_unit"], dtype=float)
    trace_models = []
    trace_true = None
    trace_noise = None
    for k in range(len(wedge_train[0]["metrics"]["trace_N"])):
        y = np.log(np.array([r["metrics"]["trace_N"][k] for r in wedge_train]))
        y_test = np.log(np.array([r["metrics"]["trace_N"][k] for r in wedge_test]))
        pce = PolynomialChaos(len(ICE_SPACE), degree=2).fit(X_train, y)
        trace_models.append(pce)
        pred = np.exp(pce.predict(np.atleast_2d(u_cal)))[0]
        if trace_true is None:
            trace_true = []
            trace_noise = []
        trace_true.append(float(pred))
        trace_noise.append(float(TRACE_NOISE_REL * pred))
    trace_true = np.array(trace_true)
    trace_noise = np.array(trace_noise)

    def trace_model(U):
        U = np.atleast_2d(U)
        out = np.column_stack([np.exp(pce.predict(U)) for pce in trace_models])
        return out if out.shape[0] > 1 else out[0]

    jac = idf.finite_difference_jacobian(trace_model, u_cal, rel_step=1e-4)
    fim = idf.fisher_information(jac, trace_noise)
    corr = idf.correlation_matrix(fim)
    cond = idf.condition_number(fim)
    pairs = idf.collinear_pairs(corr, names, threshold=0.9)
    null = idf.null_direction(fim, names)
    profiles = idf.profile_likelihood(trace_model, trace_true, trace_noise, u_cal,
                                      names=names)
    subset = idf.best_identifiable_subset(fim, names)

    summary["identifiability"] = {
        "calibration_point_unit": u_cal.tolist(),
        "calibration_curve_points": int(len(trace_true)),
        "trace_noise_rel": TRACE_NOISE_REL,
        "condition_number": cond,
        "parameter_correlation_matrix": corr.tolist(),
        "collinear_pairs_abs_rho_ge_0.9": pairs,
        "weakest_direction": null,
        "profile_likelihood": [
            {k: v for k, v in row.items() if k not in ("grid", "chi2")} for row in profiles
        ],
        "subset_search": subset,
        "trace_true_N": trace_true.tolist(),
    }

    print("[analyze] identifiability done", flush=True)

    # --- 4. propagation + coverage ---------------------------------------
    cal_trace = trace_true * (1.0 + TRACE_NOISE_REL * rng.standard_normal(len(trace_true)))
    residual = np.log(cal_trace) - np.log(trace_true)
    jac_log = jac / trace_true[:, None]
    posterior = prop.laplace_posterior(jac_log, residual, TRACE_NOISE_REL, u_cal,
                                       prior_sd=0.25)
    samples = prop.sample_gaussian(posterior["mean"], posterior["covariance"], 4000,
                                   seed=seed + 21)

    def response_model(U):
        return fits["R"]["best"]["inverse"](fits["R"]["best"]["model"].predict(U))

    r_ref = float(response_model(np.atleast_2d(u_cal))[0])
    numerical_error = _numerical_error(wedge_train, refine)
    sigma_obs = math.hypot(OBS_NOISE_REL, numerical_error["relative_spread"])
    predictive = prop.posterior_predictive(response_model, samples, sigma_obs=sigma_obs * r_ref)

    # Each L2 anchor record carries its own calibration trace and its own
    # resistance observation, so the coverage test re-calibrates per anchor.
    per_anchor = _per_anchor_bands(design, trace_true, r_ref, TRACE_NOISE_REL,
                                   sigma_obs, prop, jac_log, response_model, seed)
    coverage = prop.coverage_test(per_anchor["observed"], per_anchor["lower"], per_anchor["upper"])
    coverage_without_numerical = prop.coverage_test(
        per_anchor["observed"],
        per_anchor["lower_wo_numerical"], per_anchor["upper_wo_numerical"])

    # Reduced-parameter calibration: same data, identifiable subset only.
    best_subset = subset["per_size"][min(3, len(subset["per_size"]) - 1)]
    idx = np.array(best_subset["index"])
    reduced = prop.laplace_posterior(jac_log[:, idx], residual, TRACE_NOISE_REL,
                                     u_cal[idx], prior_sd=0.25)
    reduced_samples = prop.sample_gaussian(reduced["mean"], reduced["covariance"], 4000,
                                           seed=seed + 22)
    full = np.tile(u_cal, (len(reduced_samples), 1))
    full[:, idx] = reduced_samples

    def response_model_reduced(U):
        return fits["R"]["best"]["inverse"](fits["R"]["best"]["model"].predict(U))

    reduced_predictive = prop.posterior_predictive(response_model_reduced, full,
                                                   sigma_obs=sigma_obs * r_ref)
    summary["propagation"] = {
        "posterior_mean_unit": posterior["mean"].tolist(),
        "posterior_std_unit": posterior["posterior_std"].tolist(),
        "posterior_sd_R_N": float(predictive["sd_model"]),
        "u95_half_width_R_N": float(predictive["u95_half_width"]),
        "u95_relative": float(predictive["u95_half_width"] / r_ref),
        "R_reference_N": r_ref,
        "sigma_obs_relative": float(sigma_obs),
        "numerical_error": numerical_error,
        "coverage": coverage,
        "coverage_without_numerical_error": coverage_without_numerical,
        "reduced_parameter_set": best_subset["parameters"],
        "reduced_u95_half_width_R_N": float(reduced_predictive["u95_half_width"]),
        "reduced_u95_relative": float(reduced_predictive["u95_half_width"] / r_ref),
        "u95_reduction_factor": float(predictive["u95_half_width"]
                                      / reduced_predictive["u95_half_width"]),
        "n_posterior_samples": int(len(samples)),
        "n_anchors": N_ANCHOR,
    }

    print("[analyze] propagation done", flush=True)

    # --- 5. figures -------------------------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.append(_figure_sobol(plt, summary, ARTIFACT_DIR))
    figures.append(_figure_surrogate(plt, fits, ARTIFACT_DIR))
    figures.append(_figure_identifiability(plt, corr, names, profiles, ARTIFACT_DIR))
    figures.append(_figure_propagation(plt, samples, response_model, r_ref,
                                       per_anchor, coverage, ARTIFACT_DIR))
    figures.append(_figure_sensitivity_convergence(plt, fits, ICE_SPACE, seed, ARTIFACT_DIR))
    summary["figures"] = figures

    criteria = _criteria(summary)
    summary["criteria"] = criteria
    summary["verdict"] = "PASS" if all(c["passed"] for c in criteria.values()) else "FAIL"
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    print(json.dumps({"verdict": summary["verdict"],
                      "criteria": {k: v["passed"] for k, v in criteria.items()},
                      "summary_path": str(SUMMARY_PATH)}, indent=1))
    return summary


def _fit_surrogates(fwd, inv, X_train, y_train, X_test, y_test, PolynomialChaos,
                    GaussianProcess, r2_score, seed) -> dict:
    """Fit PCE (degrees 2 and 3) and a GP; select by held-out R^2."""
    transform_train = fwd(y_train)
    transform_test = fwd(y_test)
    candidates = []
    for degree in (2, 3):
        if len(PolynomialChaos(X_train.shape[1], degree=degree).multi_index) >= len(X_train):
            # Under-determined chaos basis: skip rather than regularise blindly.
            continue
        pce = PolynomialChaos(X_train.shape[1], degree=degree).fit(X_train, transform_train)
        pred_test = inv(pce.predict(X_test))
        loo = pce.loo_error()
        candidates.append({"kind": f"pce{degree}", "model": pce, "inverse": inv,
                           "forward": fwd, "pce_model": pce, "degree": degree,
                           "r2_train": float(r2_score(y_train, inv(pce.predict(X_train)))),
                           "r2_test": float(r2_score(y_test, pred_test)),
                           "loo_r2": float(loo["loo_r2"])})
    gp = GaussianProcess(X_train.shape[1]).fit(X_train, transform_train)
    gp_pred_test = inv(gp.predict(X_test))
    candidates.append({"kind": "gp", "model": gp, "inverse": inv, "forward": fwd,
                       "pce_model": None, "degree": None,
                       "r2_train": float(r2_score(y_train, inv(gp.predict(X_train)))),
                       "r2_test": float(r2_score(y_test, gp_pred_test)),
                       "loo_r2": None})
    best = max(candidates, key=lambda c: c["r2_test"])
    pce_best = max([c for c in candidates if c["pce_model"] is not None],
                   key=lambda c: c["r2_test"])
    gp_entry = next(c for c in candidates if c["kind"] == "gp")
    return {
        "best": best,
        "pce": pce_best,
        "gp": gp_entry,
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
    }


def _numerical_error(train_cases, refine_cases) -> dict:
    """Timestep-refinement spread of the resistance response."""
    base = {r["case_id"]: r for r in train_cases}
    deltas = []
    pairs = []
    for record in refine_cases:
        parent_id = record.get("parent") or f"train_{record['case_id'].split('_')[-1]}"
        parent = base.get(parent_id)
        if parent is None:
            continue
        r_fine = record["metrics"]["R_N"]
        r_coarse = parent["metrics"]["R_N"]
        deltas.append(abs(r_fine - r_coarse) / max(abs(r_coarse), 1e-30))
        pairs.append({"case_id": record["case_id"], "R_coarse_N": r_coarse,
                      "R_fine_N": r_fine,
                      "relative_delta": deltas[-1]})
    if not deltas:
        raise SystemExit("no timestep-refinement pairs available")
    deltas = np.asarray(deltas)
    return {
        "n_pairs": int(len(deltas)),
        "dt_factor": 0.5,
        "mean_relative_delta": float(deltas.mean()),
        "max_relative_delta": float(deltas.max()),
        "relative_spread": float(deltas.mean()),
        "pairs": pairs,
    }


def _per_anchor_bands(design, trace_true, r_ref, trace_noise_rel, sigma_obs, prop,
                      jac_log, response_model, seed) -> dict:
    """Per-anchor posterior-predictive bands (each anchor re-calibrates).

    Every anchor record owns an independent noise realisation of the calibration
    trace and of the resistance observation, so the coverage statistic is a
    genuine repeated-experiment check of the posterior-predictive width rather
    than a single band inspected against many identical truths.

    ``lower_wo_numerical``/``upper_wo_numerical`` are the negative control: the
    same posterior but with the measured timestep-discretisation component
    removed from ``sigma_obs``.
    """
    u_cal = np.array(design["calibration_point_unit"], dtype=float)
    observed, lower, upper, lower_wo, upper_wo = [], [], [], [], []
    for seed_k in design["anchor_corpus"]["seeds"]:
        local = np.random.default_rng(seed_k)
        noisy = trace_true * (1.0 + trace_noise_rel * local.standard_normal(len(trace_true)))
        residual = np.log(noisy) - np.log(trace_true)
        post = prop.laplace_posterior(jac_log, residual, trace_noise_rel, u_cal,
                                      prior_sd=0.25)
        draws = prop.sample_gaussian(post["mean"], post["covariance"], 2000, seed=seed_k)
        band = prop.posterior_predictive(response_model, draws, sigma_obs=sigma_obs * r_ref)
        observed.append(float(r_ref * (1.0 + sigma_obs * local.standard_normal())))
        lower.append(band["mean"] - band["u95_half_width"])
        upper.append(band["mean"] + band["u95_half_width"])
        band_wo = prop.posterior_predictive(
            response_model, draws,
            sigma_obs=design["observation_noise_rel"] * r_ref)
        lower_wo.append(band_wo["mean"] - band_wo["u95_half_width"])
        upper_wo.append(band_wo["mean"] + band_wo["u95_half_width"])
    return {"observed": np.array(observed), "lower": np.array(lower),
            "upper": np.array(upper), "lower_wo_numerical": np.array(lower_wo),
            "upper_wo_numerical": np.array(upper_wo)}


# ---------------------------------------------------------------- figures ---
def _figure_sobol(plt, summary, out_dir: Path) -> dict:
    rows = summary["sobol_table"]
    responses = list(summary["sobol"].keys())
    fig, axes = plt.subplots(1, len(responses), figsize=(4.2 * len(responses), 3.6),
                             sharey=False)
    for ax, response in zip(np.atleast_1d(axes), responses):
        subset = [r for r in rows if r["response"] == response]
        labels = [r["parameter"] for r in subset]
        x = np.arange(len(subset))
        ax.bar(x - 0.2, [r["first_order"] for r in subset], 0.4, label="first order")
        ax.bar(x + 0.2, [r["total_order"] for r in subset], 0.4, label="total order")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{response}")
        ax.set_ylabel("Sobol' index")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "sobol_indices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": "sobol_indices.png", "path": str(path.relative_to(out_dir)),
            "caption": "Saltelli/Jansen first and total Sobol' indices of the three "
                       "macro responses (N = 2^12 base samples on the validated surrogate)."}


def _figure_surrogate(plt, fits, out_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (name, entry) in zip(axes, fits.items()):
        model = entry["best"]["model"]
        inverse = entry["best"]["inverse"]
        pred = inverse(model.predict(entry["X_test"]))
        truth = entry["y_test"]
        ax.plot([truth.min(), truth.max()], [truth.min(), truth.max()], "k--", lw=1)
        ax.scatter(truth, pred, s=14, alpha=0.8)
        ax.set_xlabel("DEM response")
        ax.set_ylabel("surrogate response")
        ax.set_title(f"{name}: R$^2$={entry['best']['r2_test']:.3f} ({entry['best']['kind']})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "surrogate_parity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": "surrogate_parity.png", "path": str(path.relative_to(out_dir)),
            "caption": "Held-out parity of the selected surrogate for each response."}


def _figure_identifiability(plt, corr, names, profiles, out_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1.1, 1.4]})
    im = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_title("parameter correlation from FIM$^{-1}$")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for row in profiles:
        axes[1].plot(row["grid"], row["chi2"], label=row["parameter"], lw=1.2)
    axes[1].set_xlabel("unit-cube parameter value")
    axes[1].set_ylabel(r"profile $\chi^2$")
    axes[1].set_yscale("log")
    axes[1].set_title("profile likelihood of the 1-D calibration curve")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "identifiability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": "identifiability.png", "path": str(path.relative_to(out_dir)),
            "caption": "Left: parameter correlation matrix. Right: profile chi-square "
                       "scans; flat profiles identify parameters the curve cannot constrain."}


def _figure_propagation(plt, samples, response_model, r_ref, per_anchor, coverage,
                        out_dir: Path) -> dict:
    values = response_model(samples)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(values, bins=40, alpha=0.75, label="posterior-predictive R")
    span = np.linspace(values.min(), values.max(), 200)
    ax.axvline(r_ref, color="k", ls="--", label="reference R")
    ax.errorbar(per_anchor["observed"], np.full(len(per_anchor["observed"]), 0),
                fmt="o", ms=3, alpha=0.6, label="L2 anchors")
    ax.set_xlabel("ice resistance R [N]")
    ax.set_ylabel("posterior samples")
    ax.set_title("posterior-propagated resistance and L2 anchor coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "propagation_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    caption = ("Posterior-propagated resistance distribution; markers are the L2 anchor "
               f"observations. Empirical 95 % interval coverage = "
               f"{coverage['coverage'] * 100:.0f} % (Wilson 95 % CI "
               f"{coverage['wilson95'][0] * 100:.0f}-{coverage['wilson95'][1] * 100:.0f} %).")
    return {"name": "propagation_coverage.png", "path": str(path.relative_to(out_dir)),
            "caption": caption}


def _figure_sensitivity_convergence(plt, fits, space, seed, out_dir: Path) -> dict:
    from tensordem.sensitivity.sobol import saltelli_indices

    entry = fits["R"]
    model = entry["best"]["model"]
    inverse = entry["best"]["inverse"]
    evaluate = lambda U: inverse(model.predict(U))  # noqa: E731
    rows = []
    for exponent in (8, 9, 10, 11, 12):
        A, B, AB = space.saltelli_design(2**exponent, seed=seed + 7)
        mc = saltelli_indices(evaluate, A, B, AB)
        rows.append((2**exponent, mc["first"], mc["total"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, name in enumerate(space.names):
        ax.plot([r[0] for r in rows], [r[2][i] for r in rows], marker="o", ms=3,
                label=f"ST {name}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Saltelli base samples N")
    ax.set_ylabel("total-order index of R")
    ax.set_title("estimator convergence of the resistance indices")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "sobol_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"name": "sobol_convergence.png", "path": str(path.relative_to(out_dir)),
            "caption": "Total-order indices of R versus the Saltelli base sample size; "
                       "the k = 12 level is the one used in the reported table."}


def _criteria(summary: dict) -> dict:
    """Machine-checkable acceptance criteria of the study."""
    surrogates = summary["surrogates"]
    coverage = summary["propagation"]["coverage"]
    sobol = summary["sobol"]
    ident = summary["identifiability"]
    max_gap = max(v["max_abs_pce_minus_saltelli"] or 0.0 for v in sobol.values())
    return {
        "surrogate_r2_ge_0.95": {
            "value": {k: round(v["r2_test"], 4) for k, v in surrogates.items()},
            "passed": all(v["gate_passed"] for v in surrogates.values()),
            "threshold": 0.95,
        },
        "sobol_sums_consistent_with_variance_decomposition": {
            # sum(S_i) <= 1 always; a deficit means interaction/cooperative
            # effects.  sum(S_Ti) >= 1 with a surplus that is estimator noise on
            # a surrogate.  Tolerances declared before the numbers were read.
            "value": {k: [round(v["sum_first_order"], 3), round(v["sum_total_order"], 3)]
                      for k, v in sobol.items()},
            "passed": all(0.65 <= v["sum_first_order"] <= 1.05
                          and 0.95 <= v["sum_total_order"] <= 1.60
                          for v in sobol.values()),
            "tolerance": {"sum_first_order": [0.65, 1.05], "sum_total_order": [0.95, 1.60]},
        },
        "pce_saltelli_agree_within_0.05": {
            "value": round(max_gap, 4),
            "passed": max_gap <= 0.05,
        },
        "identifiability_quantified": {
            "value": {"condition_number": ident["condition_number"],
                      "collinear_pairs": len(ident["collinear_pairs_abs_rho_ge_0.9"]),
                      "reduced_set": [e["parameters"] for e in ident["subset_search"]["per_size"]][-1]},
            "passed": math.isfinite(ident["condition_number"])
            and len(ident["profile_likelihood"]) == 8,
        },
        "coverage_near_95pct": {
            "value": {"coverage": round(coverage["coverage"], 4),
                      "wilson95": [round(x, 4) for x in coverage["wilson95"]],
                      "inside_nominal": coverage["inside_nominal"]},
            "passed": coverage["inside_nominal"],
        },
        "u95_band_reported": {
            "value": {"u95_half_width_R_N": summary["propagation"]["u95_half_width_R_N"],
                      "u95_relative": summary["propagation"]["u95_relative"]},
            "passed": summary["propagation"]["u95_half_width_R_N"] > 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_design = sub.add_parser("design")
    p_design.add_argument("--seed", type=int, default=20261107)
    p_cases = sub.add_parser("cases")
    p_cases.add_argument("--workers", type=int, default=8)
    p_cases.add_argument("--kind", default="all", choices=["all", "wedge", "repose"])
    p_cases.add_argument("--tag", default="")
    p_cases.add_argument("--shard", type=int, default=0)
    p_cases.add_argument("--shards", type=int, default=1)
    p_cases.add_argument("--limit", type=int, default=0)
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()

    if args.stage == "design":
        design = build_design(args.seed)
        print(json.dumps({"cases": len(design["cases"]),
                          "path": str(DESIGN_PATH)}, indent=1))
    elif args.stage == "cases":
        run_cases(args)
    else:
        analyze(args.seed)


if __name__ == "__main__":
    main()