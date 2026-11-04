"""Is the Brazilian centre-field acceptance criterion measurable at all?

Why this script exists
----------------------
``scripts/brazilian_hondros_refinement.py`` refines the lattice and finds the
centre-field error does **not** fall monotonically: +30.5%, -15.9%, +25.9%,
-4.2% for ``N = 15, 20, 25, 30`` grains across the diameter.  A discretisation
error that is being resolved behaves like ``O(h^p)``; this does not.  So either
the physics is not converging, or the **estimator** is not.

The estimator is the thing actually being graded by the closure doc: the mean
bond virial stress over particles inside the square window ``|x|,|z| <= D/10``.
Two failure modes are possible and they call for opposite responses:

* **noise** -- the per-particle virial stress is a discrete estimator with large
  scatter, so a finite window mean has a sampling error ``sigma/sqrt(n)`` that
  shrinks only slowly.  Then the acceptance criterion is *unmeasurable*, and the
  right fix is a better estimator, not a finer lattice.
* **structure** -- the window mean is a stable number and it simply differs from
  Hondros because the lattice is anisotropic.  Then the FAIL verdict stands.

This script separates the two on a *single* relaxed configuration, so the answer
costs one relaxation rather than a refinement campaign.  It re-measures the same
equilibrium at a sweep of window sizes ``w`` and reports, per window, the mean,
the per-particle standard deviation and the standard error of that mean.  If the
window-to-window swings are comparable to the analytic target (15%) while each
individual standard error is far below them, the swing is *systematic*; if the
swings sit inside a few standard errors, it is *noise* and no window is
trustworthy.

A third diagnostic is included because it is the physically defensible
alternative: the Hondros tensile stress ``2P/(pi D t)`` is **uniform along the
whole loaded diameter**, so the natural discrete analogue is not a small square
at the centre but the mean over a thin diametral strip spanning the full
diameter.  Its scatter tells us whether a *measurable* criterion exists.

Scope: numerical verification of the apparatus.  Nothing here calibrates an ice
material or certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import (  # noqa: E402
    MANIFEST,
    contact_arc,
    virial_stress,
)
from brazilian_hondros_refinement import (  # noqa: E402
    DIAMETER_M,
    THICKNESS_M,
    TRAVEL_RATIO,
    anisotropy_metrics,
    case_at_resolution,
)
from brazilian_lattice_static import (  # noqa: E402
    lattice_moduli,
    solve_static_equilibrium,
)

#: Centre-window half-widths ``w/D`` swept by the estimator study.  ``0.1`` is
#: the value hard-wired in the published closure run, so it is the anchor.
WINDOW_RATIOS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
#: Strip half-widths ``a/D`` for the "uniform tension" estimator: a thin band
#: about the loaded diameter, spanning the full diameter in z.
STRIP_RATIOS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)


def _field(stress: np.ndarray, diameter: float, w_ratio: float) -> dict[str, Any]:
    """Mean/σ/SEM of the centre stress over the square window ``|x|,|z| <= w``."""
    w = w_ratio * diameter
    mask = (np.abs(stress["x"]) <= w) & (np.abs(stress["z"]) <= w)
    return _stats(stress, mask, {"window_ratio": w_ratio, "window_m": w})


def _strip(stress: np.ndarray, diameter: float, a_ratio: float) -> dict[str, Any]:
    """Mean/σ/SEM of σ_xx over a diametral strip ``|x| <= a``, all z."""
    a = a_ratio * diameter
    mask = np.abs(stress["x"]) <= a
    return _stats(stress, mask, {"strip_ratio": a_ratio, "strip_m": a})


def _stats(stress: np.ndarray, mask: np.ndarray, extra: dict[str, float]
           ) -> dict[str, Any]:
    count = int(mask.sum())
    out: dict[str, Any] = {**extra, "particle_count": count}
    if count == 0:
        out.update({"sigma_xx_Pa": None, "sigma_xx_sd_Pa": None,
                    "sigma_xx_sem_Pa": None, "sigma_zz_Pa": None,
                    "principal_ratio": None, "sem_over_abs_mean": None})
        return out
    sxx = stress["sxx"][mask]
    szz = stress["szz"][mask]
    mean_xx = float(sxx.mean())
    mean_zz = float(szz.mean())
    sd = float(sxx.std(ddof=1)) if count > 1 else float("nan")
    sem = sd / math.sqrt(count) if count > 1 else float("nan")
    out.update({
        "sigma_xx_Pa": mean_xx,
        "sigma_xx_sd_Pa": sd,
        "sigma_xx_sem_Pa": sem,
        "sigma_zz_Pa": mean_zz,
        "principal_ratio": (mean_zz / mean_xx) if mean_xx else None,
        "sem_over_abs_mean": (sem / abs(mean_xx)) if mean_xx and sem == sem else None,
    })
    return out


def analyse(base: dict[str, Any], resolution: int, *,
            increments: int = 20, max_iterations: int = 4000) -> dict[str, Any]:
    """Relax once, then re-measure the field at every estimator volume."""
    case = case_at_resolution(base, resolution)
    travel = TRAVEL_RATIO * DIAMETER_M
    result = solve_static_equilibrium(
        case, travel, increments=increments, max_iterations=max_iterations,
        return_configuration=True,
    )
    config = result.pop("configuration")
    sim = config["sim"]
    positions = torch.as_tensor(config["positions"], dtype=sim.dtype, device=sim.device)
    sim.positions = positions
    radius = float(case["config"]["radius_m"])
    spacing = 2.0 * radius

    stress_t = virial_stress(sim, radius)
    pos = positions.detach().cpu().numpy().astype(float)
    st = {
        "x": pos[:, 0], "z": pos[:, 2],
        "sxx": stress_t[:, 0, 0].detach().cpu().numpy().astype(float),
        "szz": stress_t[:, 2, 2].detach().cpu().numpy().astype(float),
    }
    diameter = result["specimen_diameter_m"]
    split = result["split_stress_Pa"]
    top_z = sim.top_platen_start_z + sim.time * sim.config.top_platen_velocity
    arc = contact_arc(sim, radius, diameter, top_z)

    windows = [_field(st, diameter, r) for r in WINDOW_RATIOS]
    strips = [_strip(st, diameter, r) for r in STRIP_RATIOS]
    for entry in windows + strips:
        if entry.get("sigma_xx_Pa") is not None:
            entry["tension_rel_error"] = (entry["sigma_xx_Pa"] - split) / split
            entry["principal_ratio_rel_error"] = (
                abs(entry["principal_ratio"] + 3.0) / 3.0
                if entry["principal_ratio"] is not None else None)

    # The window sweep is only meaningful if it brackets the D/10 anchor.
    anchor = next(w for w in windows if w["window_ratio"] == 0.10)
    wide = [w for w in windows if w["window_ratio"] >= 0.06]
    errs = [w["tension_rel_error"] for w in wide]
    sems = [w["sem_over_abs_mean"] for w in wide if w["sem_over_abs_mean"] is not None]
    systematic = max(errs) - min(errs)
    noise = 2.0 * max(sems) if sems else float("nan")

    moduli = lattice_moduli(case)
    return {
        "resolution_across_diameter": resolution,
        "grain_radius_m": radius,
        "grain_spacing_m": spacing,
        "specimen_diameter_m": diameter,
        "specimen_thickness_m": THICKNESS_M,
        "travel_m": travel,
        "particle_count": result["particle_count"],
        "bond_count": result["bond_count"],
        "equilibrium_converged": result["equilibrium_converged"],
        "equilibrium_residual_rel": result["equilibrium_residual_rel"],
        "mean_platen_load_N": result["mean_platen_load_N"],
        "split_stress_Pa": split,
        "contact_arc_ratio": arc["contact_arc_ratio"],
        "anchor_window": anchor,
        "windows": windows,
        "strips": strips,
        "anisotropy": anisotropy_metrics(moduli),
        "poisson_ratio": moduli.get("poisson_ratio"),
        "window_sweep": {
            "systematic_swing_in_tension_error": systematic,
            "two_sem_noise_band": noise,
            "swing_over_noise": (systematic / noise) if noise and noise == noise else None,
            "noise_dominated": bool(noise == noise and systematic <= noise),
            "swing_exceeds_15pct": bool(systematic > 0.15),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="brazil_elastic_field")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[15])
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/dem3d_brazilian_refinement"))
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = next(c for c in manifest["cases"] if c["id"] == args.case)

    cases = []
    for resolution in args.resolutions:
        row = analyse(base, resolution, increments=args.increments,
                      max_iterations=args.max_iterations)
        cases.append(row)
        print(f"N={resolution:3d} anchor(w/D=0.10) sigma_xx={row['anchor_window']['sigma_xx_Pa']:.3f} Pa "
              f"err={row['anchor_window']['tension_rel_error']:+.2%} "
              f"sem={row['anchor_window']['sem_over_abs_mean']:.2%} | "
              f"swing={row['window_sweep']['systematic_swing_in_tension_error']:.2%} "
              f"noise={row['window_sweep']['two_sem_noise_band']:.2%} "
              f"ratio={row['window_sweep']['swing_over_noise']}", flush=True)

    payload = {
        "method": ("single static relaxation (L-BFGS, 20-step load continuation) re-measured "
                   "at a sweep of virial averaging volumes; N grains across a fixed "
                   "D=%.2f m / t=%.2f m specimen" % (DIAMETER_M, THICKNESS_M)),
        "window_ratios": list(WINDOW_RATIOS),
        "strip_ratios": list(STRIP_RATIOS),
        "reference": manifest["reference"],
        "acceptance": manifest["acceptance"],
        "cases": cases,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "brazilian_estimator.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()