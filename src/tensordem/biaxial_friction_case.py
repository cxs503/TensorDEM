"""Biaxial friction-angle benchmark (CASE-DEM-BIAXIAL-FRICTION).

Measures the peak Mohr-Coulomb friction angle from a plane-strain biaxial
compression test and compares it against the classical critical-state reference
for dense sand.

Reference
---------
Bolton (1986) "The strength and dilatancy of sands", *Geotechnique* **36**(1):
65-78.  Table 2 (p. 72) gives the plane-strain peak friction angle for dense
Hostun sand as **38 deg** (phi_ps = 38 deg at D_r = 80 %, sigma_3 = 100 kPa).
This is the canonical reference for dense-sand plane-strain strength.

Acceptance criterion
--------------------
Relative error |phi_got - phi_ref| / phi_ref < 8 %.

Why wall_friction = 0
---------------------
The lateral servo walls of a biaxial cell carry a vertical shear traction when
they are rough, which biases the axial stress measured from the top platen
alone.  OSS triaxial/biaxial practice is to make the confining walls
frictionless so that the boundary traction is a pure normal stress.  This case
uses ``wall_friction=0.0`` for that reason.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from typing import Any

import numpy as np

CASE_KEY = "CASE-DEM-BIAXIAL-FRICTION"

# ---------------------------------------------------------------------------
# Pre-declared reference and tolerance
# ---------------------------------------------------------------------------
PHI_REF_DEG = 38.0  # Bolton (1986) Geotechnique 36(1):65-78, Table 2, p.72
TOLERANCE = 0.08    # 8 % relative

REF_SOURCE = (
    "Bolton (1986) Geotechnique 36(1):65-78, Table 2, p.72: "
    "dense Hostun sand plane-strain peak friction angle phi_ps = 38 deg "
    "(D_r = 80 %, sigma_3 = 100 kPa)"
)

# ---------------------------------------------------------------------------
# Anchor configuration
# ---------------------------------------------------------------------------
ANCHOR_CONFIG: dict[str, Any] = {
    "columns": 24,
    "rows": 48,
    "confining_stress": 100.0e3,   # 100 kPa
    "wall_friction": 0.0,          # frictionless lateral walls (OSS practice)
    "consolidation_steps": 4000,
    "strain_rate": 0.05,
    "strain_target": 0.15,
    "snapshot_strain": 0.01,
    "servo_gain": 3.0e-4,
    "servo_velocity_limit": 2.0,
    "radius": 0.75,
    "freeze_bond_fraction": 0.35,
}

QUICK_CONFIG: dict[str, Any] = {
    "columns": 8,
    "rows": 16,
    "confining_stress": 100.0e3,
    "wall_friction": 0.0,
    "consolidation_steps": 300,
    "strain_rate": 0.05,
    "strain_target": 0.05,
    "snapshot_strain": 0.01,
    "servo_gain": 3.0e-4,
    "servo_velocity_limit": 2.0,
    "radius": 0.75,
    "freeze_bond_fraction": 0.35,
}


def friction_angle_deg(sigma1: float, sigma3: float) -> float:
    """Mohr-Coulomb friction angle (c = 0) from principal stresses."""
    s1 = float(sigma1)
    s3 = float(sigma3)
    denom = s1 + s3
    if not (math.isfinite(s1) and math.isfinite(s3)) or denom <= 0.0:
        return float("nan")
    s = (s1 - s3) / denom
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def judge_friction(phi_got: float, *, phi_ref: float = PHI_REF_DEG,
                   tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Pre-declared acceptance arithmetic for the friction-angle benchmark."""
    rel = abs(phi_got - phi_ref) / phi_ref if math.isfinite(phi_got) else float("nan")
    return {
        "ref_deg": float(phi_ref),
        "got_deg": phi_got,
        "relative_error": rel,
        "tolerance": float(tolerance),
        "within_tolerance": bool(math.isfinite(rel) and rel <= tolerance),
        "passed": bool(math.isfinite(rel) and rel <= tolerance),
    }


def run_biaxial_friction_case(*, quick: bool = False,
                               **overrides: Any) -> dict[str, Any]:
    """Run the biaxial friction-angle benchmark and return the result dict."""
    from tensordem.biaxial_shear_band import run_biaxial_shear_band

    config = dict(QUICK_CONFIG if quick else ANCHOR_CONFIG)
    config.update({k: v for k, v in overrides.items() if v is not None})
    freeze_bond = float(config.pop("freeze_bond_fraction", 0.35))
    radius = float(config.pop("radius", 0.75))

    started = time.time()
    result = run_biaxial_shear_band(
        confining_stress=float(config["confining_stress"]),
        columns=int(config["columns"]), rows=int(config["rows"]),
        radius=radius,
        strain_rate=float(config["strain_rate"]),
        strain_target=float(config["strain_target"]),
        consolidation_steps=int(config["consolidation_steps"]),
        snapshot_strain=float(config["snapshot_strain"]),
        servo_gain=float(config["servo_gain"]),
        servo_velocity_limit=float(config["servo_velocity_limit"]),
        wall_friction=float(config["wall_friction"]),
        material_overrides={"freeze_bond_fraction": freeze_bond},
        record_snapshots=False,
    )
    elapsed = time.time() - started

    phi_peak = float(result["peak"]["friction_angle_deg"])
    phi_internal = float(result["stress_definitions"]["friction_angle_internal_deg"])

    judgement = judge_friction(phi_peak)

    # Dimensionless numbers
    sigma3 = float(config["confining_stress"])
    strain_rate = float(config["strain_rate"])
    diameter = 2.0 * radius
    # Inertial number I = eps_dot * d * sqrt(rho / sigma3)
    # Use ice density as the particle density (ridge_dem default)
    rho = 917.0  # kg/m3, ice density
    inertial_number = strain_rate * diameter * math.sqrt(rho / sigma3)

    out = {
        "case_key": CASE_KEY,
        "reference": REF_SOURCE,
        "config": {**config, "freeze_bond_fraction": freeze_bond, "radius": radius},
        "specimen": result["specimen"],
        "loading": result["loading"],
        "peak": result["peak"],
        "band": {k: v for k, v in result["band"].items() if k != "history"},
        "stress_definitions": result["stress_definitions"],
        "theory": result["theory"],
        "friction_angle": {
            "phi_peak_wall_deg": phi_peak,
            "phi_peak_internal_deg": phi_internal,
            "phi_ref_deg": PHI_REF_DEG,
            "tolerance": TOLERANCE,
        },
        "judgement": judgement,
        "dimensionless": {
            "inertial_number": inertial_number,
            "strain_rate_non-dimensional": strain_rate * diameter / math.sqrt(sigma3 / rho),
            "confining_stress_kPa": sigma3 / 1e3,
        },
        "elapsed_s": round(elapsed, 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repro_cmd": (
            f"TENSORDEM_THREADS=4 PYTHONPATH=src python "
            f"benchmarks/case_dem_biaxial_friction/run_case.py"
        ),
    }
    return out


def render_figure(result: dict[str, Any], out_path: str) -> str:
    """Generate a two-panel figure: stress path + friction-angle comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = result.get("peak", {})
    judgement = result.get("judgement", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: stress path (deviator vs axial strain)
    stress_path = result.get("stress_path", {})
    if stress_path and "axial_strain" in stress_path:
        eps = np.asarray(stress_path["axial_strain"])
        dev = np.asarray(stress_path.get("deviator", []))
        ax1.plot(eps, dev / 1e3, "b-", linewidth=1.5, label="deviator stress")
        peak_idx = int(np.argmax(dev))
        ax1.axvline(x=float(hist.get("axial_strain", 0)), color="r",
                    linestyle="--", alpha=0.5, label="peak")
        ax1.set_xlabel("Axial strain [-]")
        ax1.set_ylabel("Deviator stress [kPa]")
        ax1.set_title("Stress path")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

    # Panel 2: friction angle comparison
    phi_got = judgement.get("got_deg", float("nan"))
    phi_ref = judgement.get("ref_deg", float("nan"))
    rel_err = judgement.get("relative_error", float("nan"))

    bars_x = [0, 1]
    bars_h = [phi_ref, phi_got]
    bars_label = ["Reference\n(Bolton 1986)", "TensorDEM\n(got)"]
    colors = ["#2196F3", "#FF9800"]

    bars = ax2.bar(bars_x, bars_h, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(bars_x)
    ax2.set_xticklabels(bars_label, fontsize=10)
    ax2.set_ylabel("Friction angle [deg]")
    ax2.set_title(f"phi_peak: {phi_got:.1f} deg  (ref {phi_ref:.1f} deg, err {rel_err*100:.1f}%)")

    # Tolerance band
    tol_upper = phi_ref * (1 + TOLERANCE)
    tol_lower = phi_ref * (1 - TOLERANCE)
    ax2.axhspan(tol_lower, tol_upper, alpha=0.15, color="green", label=f"±{TOLERANCE*100:.0f}% tolerance")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Verdict text
    verdict = "PASS" if judgement.get("passed", False) else "FAIL"
    verdict_color = "green" if verdict == "PASS" else "red"
    ax2.text(0.02, 0.95, f"Verdict: {verdict}", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", color=verdict_color,
             verticalalignment="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def run_case(*, quick: bool = False, out_dir: str | None = None,
             **overrides: Any) -> dict[str, Any]:
    """Full case run: compute + judge + save artefacts."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                               "artifacts", CASE_KEY)
    os.makedirs(out_dir, exist_ok=True)

    result = run_biaxial_friction_case(quick=quick, **overrides)

    # Save JSON
    json_path = os.path.join(out_dir, "run.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save figure
    fig_path = os.path.join(out_dir, "biaxial_friction.png")
    render_figure(result, fig_path)

    verdict = "PASS" if result["judgement"]["passed"] else "FAIL"
    print(f"[{CASE_KEY}] phi_peak = {result['friction_angle']['phi_peak_wall_deg']:.2f} deg  "
          f"(ref {PHI_REF_DEG} deg, err {result['judgement']['relative_error']*100:.1f}%)  "
          f"-> {verdict}")
    print(f"  Artefacts: {json_path}, {fig_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CASE-DEM-BIAXIAL-FRICTION benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (not a result)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()
    run_case(quick=args.quick, out_dir=args.out_dir)
