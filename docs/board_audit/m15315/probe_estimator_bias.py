"""Independent audit probe: does m15279's band estimator recover a *known*
inclination through the full pipeline (local_shear_strain -> radon)?

Predeclared (written before running):
  - lattice: jittered triangular disc lattice, 24 x 48 columns/rows, spacing 1.0,
    jitter 3% (seed 11) -> ~1128 particles, same size as the m15279 resolved runs.
  - injected band: u = gamma0 * exp(-(s/w)^2) * t, s = -x sin(th) + y cos(th),
    t = (cos th, sin th); gamma0 = 0.02, w in {1.0, 2.0, 4.0} particle spacings.
  - optional homogeneous axial compression eps = 1e-2 added to the displacement.
  - decision rule: |recovered - th| <= 3 deg at w <= 2 -> pipeline is unbiased at
    the 3 deg level; a systematic drift towards 45 deg at narrow bands falsifies
    the claim that the measured 44.5 deg is a material property.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/tp-agent/work/m15279/src")
from tensordem.biaxial_shear_band import local_shear_strain, measure_band, radon_band_angle  # noqa: E402

torch.set_num_threads(1)


def lattice(columns=24, rows=48, spacing=1.0, jitter=0.03, seed=11):
    rng = np.random.default_rng(seed)
    pts = []
    for row in range(rows):
        for col in range(columns):
            x = col * spacing + 0.5 * (row % 2) * spacing
            y = row * spacing * math.sqrt(3) / 2
            pts.append((x, y))
    pts = np.asarray(pts, float)
    pts += jitter * spacing * rng.standard_normal(pts.shape)
    return pts


def band_field(x_ref, theta_deg, gamma0=0.02, width=2.0, eps=0.0):
    th = math.radians(theta_deg)
    s = -x_ref[:, 0] * math.sin(th) + x_ref[:, 1] * math.cos(th)
    t = np.array([math.cos(th), math.sin(th)])
    u = gamma0 * np.exp(-(s / width) ** 2)[:, None] * t[None, :]
    if eps:
        u = u + np.stack([-eps * x_ref[:, 0], +eps * x_ref[:, 1]], axis=1)
    return x_ref + u


def residual_without_affine_removal(x_ref, x_now, k=8):
    """Same local-strain formula, but skipping the global affine subtraction."""
    disp = x_now - x_ref
    disp = disp - disp.mean(axis=0)
    xt = torch.as_tensor(x_ref)
    dist = torch.cdist(xt, xt)
    idx = torch.topk(dist, k, largest=False).indices
    nb = x_ref[idx]
    dx = nb - nb.mean(axis=1, keepdims=True)
    du = disp[idx] - disp[idx].mean(axis=1, keepdims=True)
    a = np.einsum("nki,nkj->nij", dx, dx)
    b = np.einsum("nki,nkj->nij", du, dx)
    grad = np.linalg.solve(a.transpose(0, 2, 1), b.transpose(0, 2, 1)).transpose(0, 2, 1)
    e_xx, e_yy = grad[:, 0, 0], grad[:, 1, 1]
    e_xy = 0.5 * (grad[:, 0, 1] + grad[:, 1, 0])
    return np.sqrt((e_xx - e_yy) ** 2 + 4 * e_xy ** 2)


def main():
    ref = lattice()
    rows = []
    for theta in (45.0, 50.0, 55.0, 60.0, 65.0, 70.0):
        for width in (1.0, 2.0, 4.0):
            for eps in (0.0, 1e-2):
                now = band_field(ref, theta, width=width, eps=eps)
                out_full = measure_band(ref, now, k=8)
                # same window/field but without the global affine subtraction
                gamma_naked = residual_without_affine_removal(ref, now, k=8)
                out_naked = radon_band_angle(ref[:, 0], ref[:, 1], gamma_naked,
                                             resolution_deg=0.5)
                rows.append({
                    "theta_true": theta, "width": width, "eps": eps,
                    "full_pipeline_deg": round(out_full["folded_angle_deg"], 2),
                    "full_bias_deg": round(out_full["folded_angle_deg"] - theta, 2),
                    "full_eta2": round(out_full["eta2"], 3),
                    "no_affine_removal_deg": round(out_naked["folded_angle_deg"], 2),
                    "no_affine_bias_deg": round(out_naked["folded_angle_deg"] - theta, 2),
                })
    summary = {
        "particles": int(ref.shape[0]),
        "max_abs_bias_full_pipeline": max(abs(r["full_bias_deg"]) for r in rows),
        "max_abs_bias_no_affine": max(abs(r["no_affine_bias_deg"]) for r in rows),
        "rows": rows,
    }
    out = "/root/tp-agent/work/m15315/docs/board_audit/m15315/estimator_bias.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    for r in rows:
        print(f"th={r['theta_true']:5.1f} w={r['width']:.1f} eps={r['eps']:g} "
              f"full={r['full_pipeline_deg']:6.2f} ({r['full_bias_deg']:+5.2f}) "
              f"eta2={r['full_eta2']:.2f} naked={r['no_affine_removal_deg']:6.2f} "
              f"({r['no_affine_bias_deg']:+5.2f})")
    print("max |bias| full pipeline:", summary["max_abs_bias_full_pipeline"],
          "| without affine removal:", summary["max_abs_bias_no_affine"])


if __name__ == "__main__":
    main()
