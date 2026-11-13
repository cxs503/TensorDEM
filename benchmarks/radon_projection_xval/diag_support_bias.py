"""Post-hoc diagnostic (NOT part of the frozen case matrix): why does the OSS
image-domain projection criterion lock onto 90 deg on the elongated specimen?

Measurement, per projection angle ``theta`` of the sinogram:

* ``support_90``  number of bins that carry 90 % of the projected mass
                  (i.e. how *long* the projection is);
* ``energy``      the prereg's primary focus criterion  ``sum(p_hat**2)``;
* ``energy_norm`` ``sum(p_hat**2) * support_90`` -- the same concentration
                  divided by the projection length, i.e. "sharpness per unit
                  support";
* ``eta2_analog`` ``sum(p_hat**2) * m2`` with ``m2`` the second central moment
                  of the projection (the variance-weighted concentrate).

If ``support_90`` alone reproduces the 90 deg maximum of ``energy``, the failure
is a support-length bias of the criterion, not a defect of the OSS transform.
Nothing here changes the prereg; it only explains a measured failure mode.

Usage: python benchmarks/radon_projection_xval/diag_support_bias.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "radon_projection_xval"))

from run_xval import gaussian_ridge, lattice, production_field  # noqa: E402
from tensordem.radon_reference import radon_sinogram, rasterize_weights  # noqa: E402

OUT = ROOT / "artifacts" / "XVAL-DEM-RADON-PROJECTION-MEASUREMENT"


def criteria(sino: np.ndarray) -> dict[str, np.ndarray]:
    p = sino
    total = p.sum(axis=0, keepdims=True)
    n = p / total
    bins = np.arange(p.shape[0])[:, None]
    mu = (n * bins).sum(axis=0)
    m2 = (n * (bins - mu) ** 2).sum(axis=0)
    energy = (n ** 2).sum(axis=0)
    cumulative = np.cumsum(p, axis=0) / total
    support = (cumulative < 0.95).sum(axis=0) - (cumulative < 0.05).sum(axis=0)
    support = np.clip(support.astype(float), 1.0, None)
    return {"energy": energy, "support_90": support,
            "energy_norm": energy * support, "eta2_analog": energy * m2,
            "entropy": (n * np.log(np.where(n > 0, n, 1.0))).sum(axis=0)}


def main() -> int:
    points = lattice(24, 48, seed=11)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
    theta = np.arange(0.0, 180.0, 0.5)
    rows = []
    for field, angle in [("gaussian_ridge", 45.0), ("gaussian_ridge", 70.0),
                         ("production_strain", 40.0), ("production_strain", 45.0),
                         ("production_strain", 60.0)]:
        if field == "gaussian_ridge":
            weight = gaussian_ridge(points, angle, sigma=2.0 * spacing)
        else:
            weight = production_field(points, angle, spacing=spacing,
                                      band_width=3.0 * spacing)
        raster = rasterize_weights(points[:, 0], points[:, 1], weight,
                                   pixel_size=0.5, deposit_sigma=spacing)
        crit = criteria(radon_sinogram(raster["image"], theta))
        ridge = lambda x: float((theta[int(np.argmax(x))] - 90.0) % 180.0)
        rows.append({
            "field": field, "injected_deg": angle,
            "argmax": {k: ridge(v) for k, v in crit.items()},
            "support_90_range": [float(crit["support_90"].min()),
                                 float(crit["support_90"].max())],
            "support_90_at_90deg": float(crit["support_90"][int(np.argmin(abs(theta - 180.0)))]),
            "support_90_at_180deg": float(crit["support_90"][int(np.argmin(abs(theta - 179.5)))]),
            "support_90_at_0deg": float(crit["support_90"][0]),
            "corr_energy_support": float(np.corrcoef(crit["energy"], crit["support_90"])[0, 1]),
            "curve": {"theta_deg": theta.tolist(),
                      "energy": crit["energy"].tolist(),
                      "support_90": crit["support_90"].tolist()},
        })
        print(json.dumps({k: rows[-1][k] for k in
                          ("field", "injected_deg", "argmax", "support_90_range",
                           "corr_energy_support")}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag_support_bias.json").write_text(json.dumps({"rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())