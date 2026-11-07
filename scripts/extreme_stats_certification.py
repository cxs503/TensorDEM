"""Extreme-value statistics certification of the ice-abrasion wear field.

Reproduces the W10-N06 grid-convergence index of the P95 wear depth, replaces it
with a statistical certification (sampling interval + resolution scaling law +
adjudication) and writes machine-checkable evidence to
``artifacts/extreme_stats/certification.json`` plus the paper figures.

Stages
------
A. synthetic validation of the estimators (GEV/Gumbel profile intervals against
   Monte Carlo, extreme-value index and extremal-index recovery);
B. field certification: replicate statistics at four grid resolutions, paired
   resolution differences, block bootstrap, cell-average scaling law and the
   two-mechanism (activity smoothing + pit shot noise) resolution model;
C. figures and the acceptance-protocol verdict.

Usage: ``PYTHONPATH=src python3 scripts/extreme_stats_certification.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import tensordem.extreme_statistics as es
import tensordem.ice_abrasion as ab

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "extreme_stats"
FIG = OUT / "figs"
SPEC_RATE = ab.ICE_ON_STEEL_SPECIFIC_WEAR_GUESS


def _j(obj):
    """JSON-safe conversion (numpy scalars, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return [_j(v) for v in obj.tolist()]
    return obj


def load_field(cell, rep, case=None):
    name = case or ("c%g" % cell)
    path = OUT / "fields" / ("field_%s_r%02d.npz" % (name, rep))
    if not path.exists():
        return None
    data = np.load(path)
    return data["depth"].reshape(tuple(int(v) for v in data["shape"]))


def unified_records():
    """Merge the isotropic replicate table and the fine/anisotropic extension."""
    base = json.loads((OUT / "p95_replicates.json").read_text())["records"]
    fine = json.loads((OUT / "fine_replicates.json").read_text())["records"]
    records = []
    for r in base:
        records.append({"cell_m": float(r["cell_m"]), "case": "c%g" % r["cell_m"],
                        "rep": int(r["rep"]), "seed": int(r["seed"]),
                        "p95_m": float(r["p95_m"]), "p99_m": float(r["p99_m"]),
                        "p90_m": float(r["p90_m"]), "mean_m": float(r["mean_m"]),
                        "max_m": float(r["max_m"]), "n_cells": int(r["n_cells"]),
                        "n_nonzero": int(r["n_nonzero"]),
                        "volume_m3": float(r["volume_m3"])})
    for r in fine:
        iso = r["case"].startswith("iso")
        h = float(r["spec"]["cell_size_m"])
        records.append({"cell_m": h,
                        "case": ("c%g" % h) if iso else r["case"],
                        "grid_case": r["case"], "rep": int(r["rep"]),
                        "seed": int(r["seed"]), "p95_m": float(r["p95_m"]),
                        "p99_m": float(r["p99_m"]), "p90_m": float(r["p90_m"]),
                        "mean_m": float(r["mean_m"]), "max_m": float(r["max_m"]),
                        "n_cells": int(r["n_cells"]),
                        "n_nonzero": int(r["n_nonzero"]),
                        "volume_m3": float(r["volume_m3"]),
                        "dz_m": float(r["spec"].get("cell_size_z_m") or r["spec"]["cell_size_m"])})
    return records


def shape_sample():
    """Normalised pit-volume shape sample from the sliding-contact ensemble."""
    ens = ab.sample_contact_ensemble(ab.ContactEnsembleConfig())
    pit = SPEC_RATE * ens.pressure_pa * ens.slip_distance_m
    vol = (pit * math.pi * ens.radius_m ** 2).numpy()
    return vol / vol.mean(), ens


def stage_synthetic(quick: bool = False, cache: bool = True) -> dict:
    """A. Estimator validation against Monte Carlo (cached: it is the slow stage)."""
    cache_path = OUT / "synthetic_stage.json"
    if cache and not quick and cache_path.exists():
        return json.loads(cache_path.read_text())
    n_gum_rep = 120 if quick else 400
    n_gev_rep = 20 if quick else 40
    gum = es.coverage_study(es.GEVParams(0.0, 1.0, 0.0), n=200, n_rep=n_gum_rep,
                            p=0.95, methods=("profile", "delta", "order"))
    fre = es.coverage_study(es.GEVParams(1.0, 1.0, 0.25), n=400, n_rep=n_gev_rep,
                            p=0.95, methods=("profile", "delta", "order"),
                            gumbel_parent=False)
    wei = es.coverage_study(es.GEVParams(1.0, 1.0, -0.25), n=400, n_rep=n_gev_rep,
                            p=0.95, methods=("profile", "delta", "order"),
                            gumbel_parent=False)
    evi = es.evi_recovery_study(n=20000, blocks=(10, 50, 200, 500), n_rep=8)
    theta = es.theta_recovery_study(n=20000, a_values=(0.0, 0.2, 0.5, 0.7))
    out = {"coverage_gumbel": gum, "coverage_frechet": fre, "coverage_weibull": wei,
           "evi_recovery": evi, "theta_recovery": theta}
    if cache and not quick:
        (OUT / "synthetic_stage.json").write_text(json.dumps(_j(out), indent=1))
    return out


def stage_field(records, shape, ens) -> dict:
    """B. Field certification and resolution adjudication."""
    hull = ab.HullParticulars()
    act = ab.ActivityDistribution()
    iso = {0.125: 0.125, 0.25: 0.25, 0.5: 0.5, 1.0: 1.0}
    levels = sorted(iso)
    recs = [r for r in records if r["case"] == "c%g" % r["cell_m"]]
    fields = {}
    for h in levels:
        f = load_field(h, 0)
        if f is not None:
            fields[(h, 0)] = f
    cert = es.wear_field_p95_certificate(
        recs, fields, q=0.95, level=0.95, block_shape=(1, 4),
        n_boot=600 if 0.125 in levels else 2000, shape_sample=shape)

    # W10-N06 reproduction: single-realisation GCI of the three coarse levels
    rep0 = {r["cell_m"]: r["p95_m"] for r in recs if r["rep"] == 0}
    gci_rep0 = ab.grid_convergence_index(rep0[0.25], rep0[0.5], rep0[1.0])
    means = {h: float(np.mean([r["p95_m"] for r in recs if r["cell_m"] == h]))
             for h in levels}
    gci_mean = ab.grid_convergence_index(means[0.25], means[0.5], means[1.0])

    # resolution scaling law (four isotropic levels)
    by_cell = {h: [r["p95_m"] for r in recs if r["cell_m"] == h] for h in levels}
    law = es.resolution_scaling_law(by_cell)
    cert["resolution_law"] = law
    verdict = es.p95_resolution_verdict(cert)

    # two-mechanism resolution model (activity smoothing x pit shot noise)
    prof25 = es.waterline_activity_p95(act, hull, dx_m=0.25, dz_m=0.25)
    zero25 = float(np.mean([1 - r["n_nonzero"] / r["n_cells"]
                            for r in recs if r["cell_m"] == 0.25]))
    c25 = es.shot_rate_from_zero_fraction(prof25["field"], zero25)
    area25 = prof25["dx_actual_m"] * prof25["dz_actual_m"]
    mean_depth = float(np.mean([r["mean_m"] for r in recs]))
    configs = [("iso", 0.125, 0.125), ("iso", 0.25, 0.25), ("iso", 0.5, 0.5),
               ("iso", 1.0, 1.0), ("aniso_x0.25_z1.0", 0.25, 1.0),
               ("aniso_x1.0_z0.25", 1.0, 0.25), ("aniso_x0.25_z0.5", 0.25, 0.5)]
    model_rows = []
    for name, dx, dz in configs:
        prof = es.waterline_activity_p95(act, hull, dx_m=dx, dz_m=dz)
        area = prof["dx_actual_m"] * prof["dz_actual_m"]
        rate = c25 * area / area25
        pred = es.band_field_p95(prof["field"], mean_depth, rate, shape,
                                 n_realisations=150)
        if name == "iso":
            meas = means[dx]
            meas_cv = float(np.std([r["p95_m"] for r in recs
                                    if r["cell_m"] == dx], ddof=1) / meas * 100.0)
            n_rep = len([r for r in recs if r["cell_m"] == dx])
        else:
            sub = [r for r in records if r["case"] == name]
            meas = float(np.mean([r["p95_m"] for r in sub]))
            meas_cv = float(np.std([r["p95_m"] for r in sub], ddof=1) / meas * 100.0)
            n_rep = len(sub)
        model_rows.append({
            "case": name, "dx_m": dx, "dz_m": dz,
            "n_x": prof["n_x"], "n_z": prof["n_z"], "n_cells": prof["n_cells"],
            "hit_rate_per_cell": rate, "n_rep": n_rep,
            "measured_p95_m": meas, "measured_cv_pct": meas_cv,
            "model_p95_m": pred["p95_mean_m"], "model_cv_pct": pred["p95_cv_pct"],
            "model_rel_deviation_pct": (pred["p95_mean_m"] - meas) / meas * 100.0})
    # model-only continuation to the continuum limit (the model is the certified law)
    cont = []
    for dx, dz in ((0.0625, 0.0625), (0.03125, 0.03125), (0.015625, 0.015625)):
        prof = es.waterline_activity_p95(act, hull, dx_m=dx, dz_m=dz, n_sub=48)
        area = prof["dx_actual_m"] * prof["dz_actual_m"]
        pred = es.band_field_p95(prof["field"], mean_depth, c25 * area / area25,
                                 shape, n_realisations=60)
        cont.append({"dx_m": dx, "dz_m": dz, "n_cells": prof["n_cells"],
                     "model_p95_m": pred["p95_mean_m"]})

    # extreme-value description of the P95 sampling distribution (ALARP check)
    p95_025 = np.array([r["p95_m"] for r in recs if r["cell_m"] == 0.25])
    gumb_profile = es.gumbel_profile_ci(p95_025, p=0.5, level=0.95)
    gev = es.gev_fit(p95_025)
    rng = np.random.default_rng(20261106)
    xi_boot = []
    for _ in range(600):
        s = p95_025[rng.integers(0, p95_025.size, p95_025.size)]
        try:
            xi_boot.append(es.gev_fit(s).xi)
        except Exception:
            continue
    xi_ci = [float(np.quantile(xi_boot, 0.025)), float(np.quantile(xi_boot, 0.975))]
    # spatial clustering of the depth field: extremal index along the waterline
    field025 = fields.get((0.25, 0))
    theta_field = None
    if field025 is not None:
        row = int(np.argmax(field025.mean(axis=1)))
        theta_field = es.extremal_index_stability(field025[row, :])
        theta_field["row_index"] = row
        theta_field["row_mean_depth_m"] = float(field025[row, :].mean())
    theta_iid = es.extremal_index_stability(p95_025)
    return {"certificate": cert, "verdict": verdict,
            "gci_w10n06": {"single_realisation": gci_rep0,
                           "replicate_mean": gci_mean,
                           "levels_m": [1.0, 0.5, 0.25]},
            "resolution_model": {"zero_fraction_0p25": zero25,
                                 "hit_rate_per_cell_0p25": c25,
                                 "cell_area_0p25_m2": area25,
                                 "mean_depth_m": mean_depth,
                                 "rows": model_rows,
                                 "continuum_limit": cont},
            "extreme_value_description": {
                "n_rep": int(p95_025.size),
                "gumbel_profile_median_ci": gumb_profile,
                "gev_fit": gev.summary(),
                "gev_xi_bootstrap_ci": xi_ci,
                "theta_spatial_waterline_row": theta_field,
                "theta_replicates_negative_control": theta_iid}}


def main(quick: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    records = unified_records()
    shape, ens = shape_sample()
    synthetic = stage_synthetic(quick=quick, cache="--fresh" not in sys.argv)
    field = stage_field(records, shape, ens)
    protocol = es.acceptance_protocol_evaluate(
        field["certificate"],
        coverage=synthetic["coverage_gumbel"],
        theta=field["extreme_value_description"]["theta_spatial_waterline_row"])
    out = {"mission": 5446, "spec_rate_m3_per_Nm": SPEC_RATE,
           "ensemble": {"n_samples": len(ens),
                        "radius_median_m": float(ens.radius_m.median()),
                        "friction_coefficient": ens.friction_coefficient},
           "synthetic": synthetic, "field": field, "protocol": protocol}
    (OUT / "certification.json").write_text(json.dumps(_j(out), indent=1))
    return out


if __name__ == "__main__":
    res = main(quick="--quick" in sys.argv)
    print(json.dumps(_j(res["protocol"]), indent=1))
    print(json.dumps(_j(res["field"]["verdict"]), indent=1))
