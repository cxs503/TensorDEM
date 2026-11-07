"""Macroscopic anchors for the non-spherical DEM layer.

Runs the two granular protocols and writes
``artifacts/nonspherical-contact-timestep/anchors.json`` plus
``docs/nonspherical-contact-timestep/granular_anchors.png``.

    python scripts/verify_granular_macro_anchors.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.clump_granular_anchors import (  # noqa: E402
    cube_pile, measure_packing_fraction, measure_repose_angle,
    scott_dense_random_packing,
)

ARTIFACT_DIR = ROOT / "artifacts" / "nonspherical-contact-timestep"
FIGURE_DIR = ROOT / "docs" / "nonspherical-contact-timestep"

# Experimental band for near-cubic (aspect ratio 1) granular particles.  Published
# repose angles for cubes/sugar/dice cluster around 30-35 deg; the acceptance band
# is that range widened by the campaign's +/-3 deg tolerance.
CUBE_REPOSE_BAND_DEG = (30.0, 35.0)
REPOSE_TOLERANCE_DEG = 3.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    steps = 3_000 if args.quick else 6_000
    packing_steps = 1_500 if args.quick else 3_000

    payload = {"environment": {"python": platform.python_version(),
                               "numpy": np.__version__,
                               "platform": platform.platform()}}

    t0 = time.time()
    print("[1/2] cubic-particle pile", flush=True)
    # 60 cubic particles is the largest pile this dense-pair reference
    # implementation settles inside the campaign compute budget (see the
    # limitation note in the document); the protocol is what matters here.
    pile = cube_pile(count=60, edge=0.05, steps=steps, dt=1.0e-4, report_every=500,
                     box_width=0.30, box_depth=0.12, friction=0.6, seed=20261108)
    history = pile["history"]
    profile = measure_repose_angle(
        np.array(history[-1]["profile_x"]),
        np.array([np.nan if v is None else v for v in history[-1]["profile_z"]]))
    payload["pile"] = {
        "count": 60, "edge": 0.05, "friction": 0.6, "dt": 1.0e-4, "steps": steps,
        "history": history, "repose": profile,
        "repose_band_deg": list(CUBE_REPOSE_BAND_DEG),
        "repose_tolerance_deg": REPOSE_TOLERANCE_DEG,
        "repose_within_band": bool(
            CUBE_REPOSE_BAND_DEG[0] - REPOSE_TOLERANCE_DEG
            <= profile["angle_deg"] <= CUBE_REPOSE_BAND_DEG[1] + REPOSE_TOLERANCE_DEG),
    }
    print("    repose angle", round(profile["angle_deg"], 3), "deg", flush=True)

    print("[2/2] packing fraction", flush=True)
    packing = measure_packing_fraction(count=150, radius=0.02, steps=packing_steps,
                                       taps=2, tap_steps=300, box_width=0.34,
                                       box_height=0.5, compress_step=2.0e-5,
                                       pressure_ratio=5.0)
    payload["packing"] = {**packing, "scott": scott_dense_random_packing(),
                          "within_1pct": bool(packing["relative_error"] <= 0.01)}
    print("    packing fraction", round(packing["packing_fraction"], 4), flush=True)

    order = [h["orientation_order"] for h in history]
    tail = order[len(order) // 2:]
    payload["orientation_order"] = {
        "series": order,
        "steps": [h["step"] for h in history],
        "final": float(order[-1]),
        "tail_mean": float(np.mean(tail)),
        "tail_std": float(np.std(tail)),
        "converged": bool(np.std(tail) < 0.02),
    }
    payload["criteria"] = {
        "repose_angle_deg": profile["angle_deg"],
        "repose_band_deg": list(CUBE_REPOSE_BAND_DEG),
        "repose_within_band_pm3deg": payload["pile"]["repose_within_band"],
        "packing_fraction": packing["packing_fraction"],
        "packing_scott": scott_dense_random_packing(),
        "packing_relative_error": packing["relative_error"],
        "packing_within_1pct": payload["packing"]["within_1pct"],
        "orientation_order_tail_std": payload["orientation_order"]["tail_std"],
        "orientation_order_converged": payload["orientation_order"]["converged"],
    }
    payload["runtime_seconds"] = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    axes[0].plot(payload["orientation_order"]["steps"], order, "o-")
    axes[0].axhline(payload["orientation_order"]["tail_mean"], color="k", ls="--", lw=1.0,
                    label=f"tail mean {payload['orientation_order']['tail_mean']:.3f}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("S(t) = 1.5<cos^2 theta> - 0.5")
    axes[0].set_title("orientational order during pile formation", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    x = np.array(profile["profile_x"])
    z = np.array(profile["profile_z"])
    axes[1].plot(x, z, "o-")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("free-surface height [m]")
    axes[1].set_title(f"heap profile, repose {profile['angle_deg']:.1f} deg", fontsize=10)
    axes[1].grid(alpha=0.3)

    series = [h["max_height"] for h in history]
    axes[2].plot(payload["orientation_order"]["steps"], series, "o-")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("heap height [m]")
    axes[2].set_title(f"packing fraction {packing['packing_fraction']:.3f} "
                      f"(Scott 0.64)", fontsize=10)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "granular_anchors.png", dpi=160)
    plt.close(fig)
    payload["figures"] = ["docs/nonspherical-contact-timestep/granular_anchors.png"]

    (ARTIFACT_DIR / "anchors.json").write_text(json.dumps(payload, indent=1, default=float),
                                               encoding="utf-8")
    print(json.dumps(payload["criteria"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())