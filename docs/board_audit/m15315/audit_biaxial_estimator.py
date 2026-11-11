"""Board audit (m15315) of m15279 / CASE-BIAXIAL-SHEAR-BAND: is the reported
26 % shear-band-inclination error a *material* result or a *measurement-operator*
artefact?

The audited artefact is the module delivered by mission m15279 (commit
5060946e, branch agent/m15279, TensorDEM):
``src/tensordem/biaxial_shear_band.py`` + ``benchmarks/biaxial_shear_band/``.

Predeclared decision rule (written before the probes were run)
-------------------------------------------------------------
P1  Operator recovery on synthetic bands of known inclination, in the *same*
    measurement window geometry as the reported run (24x48 and 16x32 lattices):
    if the operator returns theta_true within 3 deg the estimator is unbiased
    and the 44.25/44.67 deg result stands as a material property.  A systematic
    deviation larger than 3 deg falsifies the "material" interpretation and the
    benchmark verdict as filed.
P2  Root cause: one-line isolation.  The operator normalises x and y by their
    own extents before projecting.  For a window of aspect ratio A = ptp(y)/ptp(x)
    this maps a true inclination theta to arctan(tan(theta)/A).  If replacing the
    per-axis normalisation by a single isotropic scale removes the deviation,
    the artefact is the normalisation and nothing else.
P3  Re-measurement of the mission's own stored field
    (``artifacts/biaxial_shear_band/final_field.npz``, produced by m15279
    ``make_figure.py``) with the isotropic operator.  The corrected inclination
    is compared with the 60 deg anchor and the 15 % tolerance.

Usage
-----
    TENSORDEM_REPO=/root/tp-agent/work/m15279 \\
        python docs/board_audit/m15315/audit_biaxial_estimator.py

``TENSORDEM_REPO`` must contain the m15279 deliverable (checkout of 5060946e);
the audit itself never imports anything outside that tree.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO = os.environ.get("TENSORDEM_REPO", "/root/tp-agent/work/m15279")
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

from tensordem.biaxial_shear_band import (  # noqa: E402
    local_shear_strain,
    radon_band_angle,
)

torch.set_num_threads(1)

THETA_REF = 60.0
TOLERANCE = 0.15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_biaxial_estimator.json")


def lattice(columns: int, rows: int, spacing: float = 1.0, jitter: float = 0.03,
            seed: int = 11) -> np.ndarray:
    """Jittered triangular lattice, same construction as the m15279 specimen."""
    rng = np.random.default_rng(seed)
    points = []
    for row in range(rows):
        for col in range(columns):
            points.append((col * spacing + 0.5 * (row % 2) * spacing,
                           row * spacing * math.sqrt(3) / 2))
    pts = np.asarray(points, dtype=float)
    return pts + jitter * spacing * rng.standard_normal(pts.shape)


def ridge(pts: np.ndarray, theta_deg: float, width: float = 2.0) -> np.ndarray:
    """Gaussian ridge of prescribed inclination (the m15279 test construction)."""
    rad = math.radians(theta_deg)
    s = -pts[:, 0] * math.sin(rad) + pts[:, 1] * math.cos(rad)
    return np.exp(-(s / width) ** 2)


def radon_isotropic(x, y, w, *, resolution_deg: float = 0.5, n_bins: int | None = None) -> dict:
    """The audited operator with ONE change: isotropic instead of per-axis scaling.

    Criterion (eta2 of the bin means), bin count and folding are identical to
    ``tensordem.biaxial_shear_band.radon_band_angle``.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    w = np.clip(np.asarray(w, dtype=float).ravel(), 0.0, None)
    w = w / w.sum()
    angles = np.arange(0.0, 180.0, float(resolution_deg))
    bins = int(n_bins) if n_bins else max(8, min(40, int(round(math.sqrt(xs.size)))))
    scale = max(float(np.ptp(xs)), float(np.ptp(ys)), 1e-12)   # <-- isotropic
    xs = (xs - xs.min()) / scale
    ys = (ys - ys.min()) / scale
    mu = float(w.mean())
    total = float(np.sum((w - mu) ** 2))
    eta = np.empty_like(angles)
    for i, alpha in enumerate(angles):
        rad = math.radians(alpha)
        s = xs * (-math.sin(rad)) + ys * math.cos(rad)
        s = (s - s.min()) / max(float(np.ptp(s)), 1e-12)
        index = np.minimum((s * bins).astype(np.int64), bins - 1)
        count = np.bincount(index, minlength=bins).astype(float)
        wsum = np.bincount(index, weights=w, minlength=bins)
        occupied = count > 0
        bin_mean = np.zeros(bins)
        bin_mean[occupied] = wsum[occupied] / count[occupied]
        eta[i] = float(np.sum(count[occupied] * (bin_mean[occupied] - mu) ** 2)) / total
    best = int(np.argmax(eta))
    alpha = float(angles[best])
    folded = 180.0 - alpha if alpha > 90.0 else alpha
    # keep the m15279 z-score definition (peak vs the far-angle null level)
    separation = np.abs(((angles - alpha + 90.0) % 180.0) - 90.0)
    far = separation > 10.0
    null_mean = float(np.mean(eta[far]))
    null_std = float(np.std(eta[far]))
    return {"angle_deg": alpha, "folded_angle_deg": folded, "eta2": float(eta[best]),
            "z_score": (float(eta[best]) - null_mean) / null_std if null_std > 0 else 0.0,
            "angles_deg": angles, "eta2_curve": eta}


def main() -> dict:
    report: dict = {"audited_repo": REPO, "theta_ref_deg": THETA_REF,
                    "tolerance": TOLERANCE, "probes": {}}

    # --- P1: operator recovery vs window aspect -----------------------------
    p1 = []
    for columns, rows in ((16, 32), (24, 48), (48, 48), (48, 24)):
        pts = lattice(columns, rows)
        aspect = float(np.ptp(pts[:, 1]) / np.ptp(pts[:, 0]))
        for theta in (45.0, 55.0, 60.0, 65.0, 70.0):
            w = ridge(pts, theta)
            lib = radon_band_angle(pts[:, 0], pts[:, 1], w, resolution_deg=0.5)
            iso = radon_isotropic(pts[:, 0], pts[:, 1], w)
            p1.append({"window": f"{columns}x{rows}", "aspect": round(aspect, 3),
                       "theta_true": theta,
                       "library_deg": round(lib["folded_angle_deg"], 2),
                       "library_bias_deg": round(lib["folded_angle_deg"] - theta, 2),
                       "isotropic_deg": round(iso["folded_angle_deg"], 2),
                       "isotropic_bias_deg": round(iso["folded_angle_deg"] - theta, 2),
                       "library_error_if_judged": round(
                           abs(lib["folded_angle_deg"] - theta) / theta, 4)})
    report["probes"]["P1_operator_recovery"] = p1
    report["P1_max_abs_bias_library"] = max(abs(r["library_bias_deg"]) for r in p1)
    report["P1_max_abs_bias_isotropic"] = max(abs(r["isotropic_bias_deg"]) for r in p1)

    # --- P2: root cause = per-axis (anisotropic) normalisation --------------
    p2 = []
    for r in p1:
        predicted = math.degrees(math.atan(math.tan(math.radians(r["theta_true"])) / r["aspect"]))
        p2.append({"window": r["window"], "aspect": r["aspect"], "theta_true": r["theta_true"],
                   "arctan_tan_over_aspect_predicted_deg": round(predicted, 2),
                   "library_deg": r["library_deg"],
                   "residual_deg": round(r["library_deg"] - predicted, 2)})
    report["probes"]["P2_root_cause"] = p2
    report["P2_max_abs_residual_deg"] = max(abs(r["residual_deg"]) for r in p2)

    # --- P3: re-measure the mission's own stored field ----------------------
    npz_path = os.path.join(REPO, "artifacts", "biaxial_shear_band", "final_field.npz")
    field = np.load(npz_path)
    x, y, gamma = field["x"], field["y"], field["gamma"]
    aspect = float(np.ptp(y) / np.ptp(x))
    lib_field = radon_band_angle(x, y, gamma, resolution_deg=1.0)
    iso_field = radon_isotropic(x, y, gamma, resolution_deg=1.0)
    # calibration of the *library* operator on this exact window geometry
    calibration = []
    for theta in (50.0, 55.0, 60.0, 65.0, 70.0):
        w = ridge(np.column_stack([x, y]), theta)
        calibration.append({"theta_true": theta,
                            "library_deg": round(radon_band_angle(x, y, w, resolution_deg=1.0)
                                                 ["folded_angle_deg"], 2),
                            "isotropic_deg": round(radon_isotropic(x, y, w)["folded_angle_deg"], 2)})
    corrected = iso_field["folded_angle_deg"]
    report["probes"]["P3_real_field"] = {
        "npz": npz_path,
        "particles": int(x.size),
        "window_aspect": round(aspect, 3),
        "stored_angle_deg_m15279": float(field["angle_deg"]),
        "library_radon_deg": round(lib_field["folded_angle_deg"], 2),
        "isotropic_radon_deg": round(corrected, 2),
        "isotropic_eta2": round(iso_field["eta2"], 3),
        "isotropic_z": round(iso_field["z_score"], 2),
        "relative_error_vs_anchor": round(abs(corrected - THETA_REF) / THETA_REF, 4),
        "within_tolerance": bool(abs(corrected - THETA_REF) / THETA_REF <= TOLERANCE),
        "calibration_on_same_window": calibration,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=1))
    print(json.dumps(report["probes"]["P3_real_field"], indent=1))
    print("[written]", OUT)
    return report


if __name__ == "__main__":
    main()