"""R2 re-measurement of CASE-BIAXIAL-SHEAR-BAND with the corrected estimator.

The delivered case reported ``44.25 +- 0.8`` / ``44.67 +- 0.5`` deg and an honest
FAIL against the 60 deg dense-sand anchor.  ``run_r2_estimator_audit.py`` shows
that the pre-R2 Radon estimator was biased by up to 18 deg on the elongated
specimen of this exact case, so those numbers are not trustworthy.  This script
re-runs the two *resolved* configurations with the fixed estimator and, because
every strain snapshot is recorded, measures the *same* data twice:

* ``theta_r2``      -- production pipeline after the estimator fix;
* ``theta_pre_r2``  -- pre-R2 estimator on the identical snapshots (no run-to-run
  noise between the two numbers: a paired before/after comparison);
* ``psi_band``      -- dilation angle measured **inside** the band, next to the
  global ``psi`` that the original case used.  Roscoe's
  ``theta = 45 + psi/2`` is a *band-local* relation, so the global dilation
  angle is a category error that the original report flagged as an
  unphysical ``psi > phi``.

Outputs ``artifacts/biaxial_shear_band/r2_case_rerun.json`` and
``artifacts/biaxial_shear_band/r2_case_rerun.png``.
Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r2_case_rerun.py --workers 2``
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

THETA_REF_DEG = 60.0
TOLERANCE = 0.15
BOND_FRACTION = 0.35
DIAMETER = 1.5
HALF_WIDTH = 1.0 * DIAMETER          # +- one diameter around the band centre line

CONFIGS = [
    {"name": "r24x48_p100", "columns": 24, "rows": 48, "confining_stress": 100.0e3},
    {"name": "r24x48_p200", "columns": 24, "rows": 48, "confining_stress": 200.0e3},
]


def _run_one(config: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from tensordem.biaxial_band_diagnostics import (
        band_local_dilation,
        legacy_radon_band_angle,
    )
    from tensordem.biaxial_shear_band import (
        local_shear_strain,
        measure_band,
        run_biaxial_shear_band,
    )

    started = time.time()
    result = run_biaxial_shear_band(
        confining_stress=config["confining_stress"],
        columns=config["columns"], rows=config["rows"],
        strain_rate=0.05, strain_target=0.12,
        consolidation_steps=2000, snapshot_strain=0.01,
        servo_gain=3.0e-4, servo_velocity_limit=2.0,
        material_overrides={"freeze_bond_fraction": BOND_FRACTION},
        record_snapshots=True)
    snapshots = result["snapshots"]
    peak_strain = float(result["peak"]["axial_strain"])
    final_strain = float(snapshots[-1]["strain"])
    tail_start = peak_strain + 0.4 * max(final_strain - peak_strain, 0.0)
    late = [s for s in snapshots if s["strain"] >= tail_start]

    fixed_angles, legacy_angles, aspects = [], [], []
    for ref, now in zip(late[:-1], late[1:]):
        window = ref["window"] or now["window"]
        fixed_angles.append(float(measure_band(ref["positions"], now["positions"],
                                               bounds=window)["folded_angle_deg"]))
        xmin, xmax, ymin, ymax = window
        ref_pos = ref["positions"]
        inside = ((ref_pos[:, 0] >= xmin) & (ref_pos[:, 0] <= xmax)
                  & (ref_pos[:, 1] >= ymin) & (ref_pos[:, 1] <= ymax))
        gamma = local_shear_strain(ref_pos[inside], now["positions"][inside], k=8)
        legacy_angles.append(float(legacy_radon_band_angle(
            ref_pos[inside, 0], ref_pos[inside, 1], gamma)["folded_angle_deg"]))
        aspects.append((float(np.ptp(ref_pos[inside, 0])),
                        float(np.ptp(ref_pos[inside, 1]))))

    theta_r2 = float(np.mean(fixed_angles))
    theta_pre = float(np.mean(legacy_angles))
    dilation = band_local_dilation(late, angle_deg=theta_r2, half_width=HALF_WIDTH)

    phi = float(result["peak"]["friction_angle_deg"])
    psi_global = float(result["peak"]["dilation_angle_deg"])
    theory_band = {name: 45.0 + coef_a * phi + coef_b * dilation["psi_band_deg"]
                   for name, (coef_a, coef_b) in
                   {"mohr_coulomb": (0.5, 0.0), "roscoe": (0.0, 0.5),
                    "arthur": (0.25, 0.25)}.items()}
    theory_global = {name: 45.0 + coef_a * phi + coef_b * psi_global
                     for name, (coef_a, coef_b) in
                     {"mohr_coulomb": (0.5, 0.0), "roscoe": (0.0, 0.5),
                      "arthur": (0.25, 0.25)}.items()}

    # cumulative field of the last late snapshot, kept for the figure
    field = None
    if late:
        window = late[-1]["window"]
        xmin, xmax, ymin, ymax = window
        pos = late[0]["positions"]
        inside = ((pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                  & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax))
        gamma = local_shear_strain(pos[inside], late[-1]["positions"][inside], k=8)
        field = {"x": pos[inside, 0].tolist(), "y": pos[inside, 1].tolist(),
                 "gamma": gamma.tolist()}

    return {
        "name": config["name"],
        "confining_stress": float(config["confining_stress"]),
        "theta_r2_deg": theta_r2,
        "theta_r2_std_deg": float(np.std(fixed_angles)),
        "theta_pre_r2_deg": theta_pre,
        "theta_pre_r2_std_deg": float(np.std(legacy_angles)),
        "late_snapshots": len(fixed_angles),
        "relative_error_r2": abs(theta_r2 - THETA_REF_DEG) / THETA_REF_DEG,
        "relative_error_pre_r2": abs(theta_pre - THETA_REF_DEG) / THETA_REF_DEG,
        "within_tolerance_r2": bool(abs(theta_r2 - THETA_REF_DEG) / THETA_REF_DEG <= TOLERANCE),
        "within_tolerance_pre_r2": bool(abs(theta_pre - THETA_REF_DEG) / THETA_REF_DEG <= TOLERANCE),
        "aspect_ratio_window": (aspects[-1][0] / aspects[-1][1]) if aspects else math.nan,
        "phi_deg": phi,
        "psi_global_deg": psi_global,
        "dilation_band": dilation,
        "theory_from_band_psi_deg": theory_band,
        "theory_from_global_psi_deg": theory_global,
        "band_history_theta_deg": result["band"]["angle_deg"],
        "band_history_resolved": bool(result["band"]["resolved"]),
        "servo_relative_offset": float(np.mean(result["stress_path"]["sigma_h"])
                                       / config["confining_stress"] - 1.0),
        "broken_bonds": int(result["broken_bonds"]),
        "wall_time_s": time.time() - started,
        "gamma_field": field,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r2_case_rerun.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r2_case_rerun.png")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if here not in sys.path:
        sys.path.insert(0, os.path.join(here, "src"))

    results = []
    if args.workers > 1:
        with mp.get_context("fork").Pool(min(args.workers, len(CONFIGS))) as pool:
            for result in pool.imap_unordered(_run_one, CONFIGS):
                results.append(result)
                print(f"[done] {result['name']} theta_r2={result['theta_r2_deg']:.2f} "
                      f"theta_pre={result['theta_pre_r2_deg']:.2f} deg", flush=True)
    else:
        for config in CONFIGS:
            try:
                results.append(_run_one(config))
            except Exception as exc:                        # pragma: no cover - reported
                results.append({"name": config["name"], "error": repr(exc),
                                "traceback": traceback.format_exc()})
            print(f"[done] {results[-1].get('name')}", flush=True)

    ok = [r for r in results if "error" not in r]
    payload = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "stage": "R2 re-measurement (estimator fix)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor": {"theta_ref_deg": THETA_REF_DEG, "tolerance": TOLERANCE,
                   "source": ("Desrues & Viggiani (2004) I/JNUMAG 28:279-321, "
                              "dense Hostun sand plane-strain biaxial 55-60 deg")},
        "settings": {"bond_fraction": BOND_FRACTION, "band_half_width": HALF_WIDTH},
        "runs": results,
        "summary": {
            "runs": len(results),
            "resolved": len(ok),
            "theta_r2_deg": [round(r["theta_r2_deg"], 2) for r in ok],
            "theta_pre_r2_deg": [round(r["theta_pre_r2_deg"], 2) for r in ok],
            "rel_err_r2": [round(r["relative_error_r2"], 4) for r in ok],
            "rel_err_pre_r2": [round(r["relative_error_pre_r2"], 4) for r in ok],
            "psi_band_deg": [round(r["dilation_band"]["psi_band_deg"], 2) for r in ok],
            "psi_global_deg": [round(r["psi_global_deg"], 2) for r in ok],
            "passed_r2": sum(1 for r in ok if r["within_tolerance_r2"]),
        },
    }
    payload["summary"]["all_within_tolerance_r2"] = (
        len(ok) == len(results) and payload["summary"]["passed_r2"] == len(results))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = {r["name"]: r.pop("gamma_field") for r in results}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    print(json.dumps(payload["summary"], indent=1))
    print(f"[written] {args.out}")
    _figure(payload, fields, args.figure)
    return 0 if payload["summary"]["all_within_tolerance_r2"] else 1


def _figure(payload: dict, fields: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in payload["runs"] if "error" not in r]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    axes[0].axhspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
                    color="tab:green", alpha=0.12, label="+/-15 % band")
    axes[0].axhline(THETA_REF_DEG, color="tab:green", lw=1.0)
    names = [r["name"] for r in ok]
    axes[0].plot(names, [r["theta_pre_r2_deg"] for r in ok], "s--", color="tab:red",
                 label="pre-R2 estimator")
    axes[0].plot(names, [r["theta_r2_deg"] for r in ok], "o-", color="tab:blue",
                 label="R2 estimator")
    for i, r in enumerate(ok):
        axes[0].annotate(f"{r['theta_pre_r2_deg']:.1f}", (i, r["theta_pre_r2_deg"]),
                         textcoords="offset points", xytext=(4, -12), fontsize=8,
                         color="tab:red")
        axes[0].annotate(f"{r['theta_r2_deg']:.1f}", (i, r["theta_r2_deg"]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8,
                         color="tab:blue")
    axes[0].set_ylabel("band inclination [deg from $\\sigma_3$]")
    axes[0].set_title("(a) same run, two estimators")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    theory = {"roscoe": lambda p, s: 45 + s / 2,
              "arthur": lambda p, s: 45 + (p + s) / 4,
              "mohr_coulomb": lambda p, s: 45 + p / 2}
    styles = {"roscoe": "tab:purple", "arthur": "tab:orange", "mohr_coulomb": "tab:brown"}
    for name, fn in theory.items():
        xs = np.linspace(0.0, 25.0, 50)
        axes[1].plot(xs, [fn(phi, s) for phi in [30.0] for s in xs], "-",
                     color=styles[name], lw=1.2,
                     label=f"{name} (phi=30 deg)")
    psi_band = [r["dilation_band"]["psi_band_deg"] for r in ok]
    phi_deg = [r["phi_deg"] for r in ok]
    axes[1].plot(psi_band, [r["theta_r2_deg"] for r in ok], "o", color="tab:blue",
                 ms=8, label="measured (psi band)")
    axes[1].plot([r["psi_global_deg"] for r in ok], [r["theta_r2_deg"] for r in ok],
                 "x", color="tab:red", ms=9, label="measured (psi global, old)")
    for r, x in zip(ok, psi_band):
        axes[1].annotate(f"phi={r['phi_deg']:.0f}", (x, r["theta_r2_deg"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    axes[1].set_xlabel("dilation angle [deg]")
    axes[1].set_ylabel("band inclination [deg]")
    axes[1].set_title("(b) Roscoe / Arthur against band-local dilatancy")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    key = next((k for k in fields if fields[k]), None)
    if key is not None:
        field = fields[key]
        sc = axes[2].scatter(field["x"], field["y"], c=field["gamma"], s=14, cmap="magma")
        centre = (float(np.mean(field["x"])), float(np.mean(field["y"])))
        run = next(r for r in ok if r["name"] == key)
        for angle, colour, label in ((run["theta_r2_deg"], "cyan", "R2"),
                                     (run["theta_pre_r2_deg"], "lime", "pre-R2")):
            rad = math.radians(angle)
            length = float(np.ptp(field["y"])) * 0.45
            dx, dy = math.cos(rad) * length, math.sin(rad) * length
            axes[2].plot([centre[0] - dx, centre[0] + dx], [centre[1] - dy, centre[1] + dy],
                         "-", color=colour, lw=1.8, label=f"{label} {angle:.1f} deg")
        fig.colorbar(sc, ax=axes[2], label="local max shear strain $\\gamma$")
        axes[2].set_aspect("equal")
        axes[2].set_title(f"(c) late-window $\\gamma$ field, {key}")
        axes[2].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figure] {path}")


if __name__ == "__main__":
    raise SystemExit(main())