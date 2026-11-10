#!/usr/bin/env python3
"""Ice-particle Mohr-Coulomb campaign (campaign node p246w2-n02, TensorDEM).

Runs the four experiments of the verification surface, evaluates the machine
checks C1..C7, writes ``artifacts/ice_particle_mc/report.json`` and renders the
figures into ``artifacts/ice_particle_mc/figures``.

    python scripts/run_ice_particle_mc_campaign.py               # full campaign
    python scripts/run_ice_particle_mc_campaign.py --quick       # smoke (small)

Every number in the JSON is computed by the drivers in
``tensordem.ice_particle_mc.study`` from the simulation state; the acceptance
booleans are evaluated here and the raw deviations are kept next to them, so a
FAIL cannot be hidden by a formatting choice.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_particle_mc import analysis, study  # noqa: E402
from tensordem.ice_particle_mc.study import (IndentationSpec, UniaxialSpec,  # noqa: E402
                                             force_chain_stats)

ARTIFACTS = ROOT / "artifacts" / "ice_particle_mc"
FIGURES = ARTIFACTS / "figures"

#: three (c, phi) groups: the C1 acceptance runs over exactly these
MC_GROUPS = ((0.80e6, 25.0), (1.50e6, 30.0), (2.20e6, 35.0))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_uniaxial_group(cohesion, phi, seed, quick):
    spec = UniaxialSpec(cohesion_pa=cohesion, friction_angle_deg=phi, seed=seed,
                        strain_target=1.5e-3 if quick else 3.0e-3)
    result = study.run_uniaxial(spec)
    return result


def proportionality_run(quick):
    """E must scale linearly with the input bond modulus (same geometry/seed)."""
    base = study.run_uniaxial(UniaxialSpec(cohesion_pa=1.5e6, friction_angle_deg=30.0,
                                           seed=0, bond_modulus=9.0e9,
                                           strain_target=1.5e-3 if quick else 3.0e-3))
    half = study.run_uniaxial(UniaxialSpec(cohesion_pa=1.5e6, friction_angle_deg=30.0,
                                           seed=0, bond_modulus=4.5e9,
                                           strain_target=1.5e-3 if quick else 3.0e-3))
    ratio = half["E_measured_Pa"] / base["E_measured_Pa"] if base["E_measured_Pa"] else float("nan")
    return {"E_full_Pa": base["E_measured_Pa"], "E_half_Pa": half["E_measured_Pa"],
            "ratio": ratio, "expected_ratio": 0.5,
            "error_vs_0p5": abs(ratio - 0.5) / 0.5 if math.isfinite(ratio) else float("nan")}


def figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. stress-strain curves with the MC prediction
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for group in results["C1"]["groups"]:
        ax.plot(np.array(group["strain"]) * 1e3, np.array(group["stress"]) / 1e6,
                label=f"c={group['spec']['cohesion_pa']/1e6:.2f} MPa, phi={group['spec']['friction_angle_deg']:.0f} deg")
        ax.axhline(group["sigma_mc_Pa"] / 1e6, ls=":", lw=1.0,
                   color=ax.lines[-1].get_color())
    ax.set_xlabel("engineering axial strain [1e-3]")
    ax.set_ylabel("axial stress [MPa]")
    ax.set_title("Uniaxial compression (D=50 mm, H=100 mm): dotted = MC prediction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = FIGURES / "uniaxial_stress_strain.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    made.append(path.name)

    # 2. pressure-area law
    fit = results["C3"]
    if fit.get("areas"):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.loglog(fit["areas"], fit["pressures"], "o", label="DEM punch peak pressure")
        a = np.array(fit["areas"], dtype=float)
        p0 = fit["fit"].get("prefactor")
        n = fit["exponent"]
        if p0 and math.isfinite(n):
            ax.loglog(a, p0 * a ** n, "-",
                      label=f"fit: P ~ A^{n:.3f} (R2={fit['r2']:.3f})")
        ref = fit["pressures"][0]
        ax.loglog(a, ref * (a / a[0]) ** -0.5, "--", label="Korzhavin A^(-0.5) reference")
        ax.set_xlabel("punch contact area [m^2]")
        ax.set_ylabel("peak contact pressure [Pa]")
        ax.set_title("Pressure-area relation from punch indentation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        path = FIGURES / "pressure_area_law.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        made.append(path.name)

    # 3. force-chain statistics on the crushed specimen
    fc = results["C5"]
    if fc.get("centres"):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.semilogy(fc["centres"], fc["densities"], "o", label="P(f) DEM contacts")
        c = np.array(fc["centres"], dtype=float)
        slope = fc.get("slope")
        if slope is not None and math.isfinite(slope):
            ax.semilogy(c, np.exp(fc["slope"] * c + math.log(fc["densities"][0])
                                  - fc["slope"] * fc["centres"][0]),
                        "-", label=f"exp fit: lambda={fc['decay_length']:.2f}<f>, R2={fc['r2']:.3f}")
        ax.set_xlabel("f / <f>")
        ax.set_ylabel("P(f/<f>)")
        ax.set_title("Contact force distribution (crushed ice specimen)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        path = FIGURES / "force_chain_distribution.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        made.append(path.name)

    # 4. friction sensitivity
    rs = results["C4"]
    if rs.get("cases"):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        mus = [c["mu"] for c in rs["cases"]]
        forces = [c["R_ind_N"] for c in rs["cases"]]
        ax.plot(mus, forces, "o-")
        ax.set_xlabel("contact friction coefficient mu")
        ax.set_ylabel("peak indentation resistance [N]")
        ax.set_title("Resistance sensitivity to friction")
        ax.grid(alpha=0.3)
        path = FIGURES / "friction_sensitivity.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        made.append(path.name)

    # 5. energy ledger
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names = [g["label"] for g in results["C6"]["cases"]]
    residuals = [abs(g["residual"]) * 100.0 for g in results["C6"]["cases"]]
    ax.bar(range(len(names)), residuals)
    ax.axhline(5.0, color="r", ls="--", label="5 % acceptance")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, fontsize=8)
    ax.set_ylabel("|energy residual| / W_ext  [%]")
    ax.set_title("Energy ledger closure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    path = FIGURES / "energy_ledger.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    made.append(path.name)
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-indentation", action="store_true")
    parser.add_argument("--out", default=str(ARTIFACTS / "report.json"))
    args = parser.parse_args()
    started = time.time()
    results: dict = {}

    # ---------------- C1 / C2 / C6 : uniaxial -------------------------------
    groups = []
    for cohesion, phi in MC_GROUPS:
        print(f"[uniaxial] c={cohesion/1e6:.2f} MPa phi={phi} deg", flush=True)
        try:
            groups.append(run_uniaxial_group(cohesion, phi, 0, args.quick))
        except Exception as exc:            # pragma: no cover - defensive
            groups.append({"error": repr(exc), "spec": {"cohesion_pa": cohesion,
                                                        "friction_angle_deg": phi}})
    ok_groups = [g for g in groups if "sigma_peak_Pa" in g]
    c1_ok = bool(ok_groups) and all(g["sigma_error"] <= 0.10 for g in ok_groups)
    c2_errors = [g["E_error"] for g in ok_groups if math.isfinite(g.get("E_error", float("nan")))]
    c2_ok = bool(c2_errors) and max(c2_errors) <= 0.05
    results["C1"] = {
        "criterion": "sigma_peak vs 2c cos(phi)/(1-sin phi), error <= 10 % (3 groups)",
        "groups": groups,
        "errors": [g.get("sigma_error") for g in ok_groups],
        "max_error": max((g["sigma_error"] for g in ok_groups), default=float("nan")),
        "pass": c1_ok,
    }
    results["C2"] = {
        "criterion": "E from the pre-peak linear segment vs the affine lattice prediction, error <= 5 %",
        "E_measured_Pa": [g.get("E_measured_Pa") for g in ok_groups],
        "E_affine_Pa": [g.get("E_affine_Pa") for g in ok_groups],
        "E_error": c2_errors,
        "E_r2": [g.get("E_r2") for g in ok_groups],
        "pass_vs_affine_prediction": c2_ok,
        "proportionality": proportionality_run(args.quick),
    }
    results["C2"]["pass"] = bool(results["C2"]["proportionality"]["error_vs_0p5"] <= 0.05
                                 and c2_ok)

    # ---------------- C3 : pressure-area law ------------------------------
    if args.skip_indentation:
        results["C3"] = {"pass": False, "note": "skipped by flag"}
    else:
        print("[indentation] pressure-area sweep", flush=True)
        try:
            ind = study.run_indentation(IndentationSpec(
                punch_widths=(0.06, 0.10, 0.14) if args.quick else (0.05, 0.075, 0.10, 0.125, 0.15),
                penetration=0.006 if args.quick else 0.008,
                step_displacement=4.0e-6 if args.quick else 3.0e-6))
            pressures = ind["pressures"]
            plausible = all(p < 1.0e8 for p in pressures if p)   # < 100 MPa
            results["C3"] = dict(ind)
            results["C3"]["pressures_below_100MPa"] = plausible
            results["C3"]["criterion"] = "P(A) = P0 (A/A0)^n with n = -0.5 +- 0.15"
            results["C3"]["pass"] = bool(math.isfinite(ind["exponent"])
                                         and ind["exponent_error_vs_m05"] <= 0.15
                                         and plausible)
        except Exception as exc:            # pragma: no cover - defensive
            results["C3"] = {"pass": False, "error": repr(exc)}

    # ---------------- C4 : friction sensitivity ---------------------------
    print("[friction] resistance sensitivity", flush=True)
    try:
        rs = study.run_ice_resistance(study.ResistanceSpec(
            friction_coefficients=(0.1, 0.3),
            penetration=0.006 if args.quick else 0.008,
            step_displacement=4.0e-6 if args.quick else 3.0e-6))
        results["C4"] = rs
        results["C4"]["criterion"] = ("mu 0.1 -> 0.3 changes the ice crushing resistance "
                                      "by >= 20 % (indentation proxy for R_ice)")
        results["C4"]["pass"] = bool(rs["R_ind_change"] >= 0.20)
    except Exception as exc:                # pragma: no cover - defensive
        results["C4"] = {"pass": False, "error": repr(exc)}

    # ---------------- C5 : force-chain statistics -------------------------
    sample = None
    for case in (results.get("C4", {}).get("cases") or []):
        sample = case.get("force_snapshot")
    if sample is not None:
        stats = force_chain_stats(np.asarray(sample, dtype=float))
        results["C5"] = dict(stats)
        results["C5"]["criterion"] = "P(f) ~ exp(-f/<f>) for f > <f>, R2 >= 0.9"
        results["C5"]["pass"] = bool(math.isfinite(stats.get("r2", float("nan")))
                                     and stats["r2"] >= 0.9)
        results["C5"]["source"] = "contact force sample of the crushed punch specimen"
    else:                                    # pragma: no cover - defensive
        results["C5"] = {"pass": False, "note": "no contact sample"}

    # ---------------- C6 : energy ledger ----------------------------------
    cases = []
    for group in ok_groups:
        cases.append({"label": f"ucs {group['spec']['cohesion_pa']/1e6:.2f}MPa",
                      "residual": group["energy_residual_relative"],
                      "external_work_J": group["energy"]["external_work"]})
    for case in (results.get("C3", {}).get("cases") or []):
        cases.append({"label": f"punch {case['punch_area_m2']*1e4:.1f}cm2",
                      "residual": case.get("energy_residual_relative"),
                      "external_work_J": None})
    worst = max((abs(c["residual"]) for c in cases
                 if c["residual"] is not None and math.isfinite(c["residual"])),
                default=float("nan"))
    results["C6"] = {"criterion": "external work = elastic + kinetic + friction + fracture, residual <= 5 %",
                     "cases": cases, "worst_abs_residual": worst,
                     "pass": bool(math.isfinite(worst) and worst <= 0.05)}

    # ---------------- C7 : honesty gate -----------------------------------
    results["C7"] = {
        "criterion": "rate/temperature dependence of real ice declared, not modelled",
        "declaration": ("The bond law is isothermal and rate independent: no creep, no "
                        "strain-rate strengthening, no temperature dependence, no "
                        "pressure melting. Real ice strength varies by a factor of ~2-3 "
                        "over the 1e-5..1e-1 1/s strain-rate range and by ~2x over "
                        "-5..-30 C; those are NOT modelled here and no result in this "
                        "report may be read as a rate- or temperature-dependent ice "
                        "strength. The bond cohesion/friction transfer to the macro "
                        "parameters is calibrated through the affine fabric factor "
                        "(reported per group), which is a geometric, not a physical, "
                        "calibration."),
        "pass": True,
    }

    results["meta"] = {
        "node": "p246w2-n02",
        "repo": "TensorDEM",
        "module": "tensordem.ice_particle_mc",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - started, 1),
        "quick": bool(args.quick),
        "summary_pass": {k: results[k].get("pass") for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    try:
        made = figures(results)
        results["meta"]["figures"] = made
    except Exception as exc:                         # pragma: no cover - defensive
        results["meta"]["figures_error"] = repr(exc)
    Path(args.out).write_text(json.dumps(_jsonable(results), indent=1))
    print(json.dumps(results["meta"]["summary_pass"], indent=1))
    print("written", args.out, "wall", results["meta"]["wall_seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())