#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498) negative-input probes for TensorDEM.

Question this script answers
----------------------------
Does a nonsensical input raise, or does it silently return a number that a
downstream consumer cannot distinguish from a valid physical result?

Each probe declares the expected behaviour *before* the call.  Three outcomes
are recorded:

* ``REJECTED`` - the API raised (expected for invalid input) -> guard present.
* ``SILENT``   - no exception, produced a finite/NaN number -> **audit finding**
  (a caller can unknowingly consume a placeholder).
* ``CRASH_UNEXPECTED`` - raised on input the API is supposed to accept.

The script never modifies library code; it only probes the public surface.
Outputs ``runs/negative_probe/results.json`` plus a readable log on stdout.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import (  # noqa: E402
    DEM3DConfig,
    IceDEM3D,
    hertz_normal_force,
    hertz_reduced_modulus,
)
from tensordem.brash_ice_rheology.material import BrashIceState  # noqa: E402
from tensordem.brash_ice_rheology.model import brash_resistance  # noqa: E402
from tensordem.brash_ice_rheology.fullscale_validation import (  # noqa: E402
    MT_UIKKU_HULL,
    OperatingPoint,
    dem_resistance,
)

RESULTS: list[dict] = []


def probe(pid: str, what: str, expect: str, fn) -> None:
    """Run ``fn`` and classify the outcome against ``expect``.

    ``expect`` is ``"raise"`` (invalid input must raise) or ``"accept"``
    (valid input must work; a raise here is an unexpected crash).
    """
    try:
        value = fn()
        outcome, detail = "no-raise", _fmt(value)
    except Exception as exc:  # noqa: BLE001 - probe wants every exception class
        outcome, detail = "raise", f"{type(exc).__name__}: {exc}"
    if expect == "raise":
        verdict = "REJECTED" if outcome == "raise" else "SILENT"
    else:
        verdict = "ACCEPTED" if outcome == "no-raise" else "CRASH_UNEXPECTED"
    RESULTS.append(
        dict(probe=pid, what=what, expected=expect, outcome=outcome,
             observed=detail, verdict=verdict)
    )
    print(f"[{verdict:15s}] {pid}: {what} -> {detail}")


def _fmt(value) -> str:
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all())
        return f"tensor(shape={tuple(value.shape)}, all_finite={finite}, min={float(value.min()):.6g})"
    if isinstance(value, float):
        return f"{value:.6g} (finite={math.isfinite(value)})"
    return str(value)


def dem3d(**overrides):
    def _build():
        cfg = DEM3DConfig(**overrides)
        sim = IceDEM3D(cfg)
        for _ in range(2):
            sim.step()
        return sim.broken_bonds  # property: touch a stepped state
    return _build


def main() -> int:
    # --- 1. contact-law primitives -------------------------------------
    probe("hertz-neg-overlap",
          "hertz_normal_force(overlap=-1e-5) must be exactly 0 (no adhesion branch)",
          "accept",
          lambda: hertz_normal_force(-1e-5, 1e9, 0.01))
    probe("hertz-nan-overlap",
          "hertz_normal_force(overlap=nan) should be rejected, not propagate NaN",
          "raise",
          lambda: hertz_normal_force(float("nan"), 1e9, 0.01))
    probe("hertz-int-tensor",
          "hertz_normal_force(integer tensor) must raise TypeError",
          "raise",
          lambda: hertz_normal_force(torch.tensor([1, 2]), 1e9, 0.01))
    probe("hertz-poisson-1",
          "hertz_reduced_modulus(poisson_ratio=1.0) is singular (1-nu^2=0)",
          "raise",
          lambda: hertz_reduced_modulus(7e10, 1.0))
    probe("hertz-negative-E",
          "hertz_reduced_modulus(young_modulus=-7e10) is unphysical",
          "raise",
          lambda: hertz_reduced_modulus(-7e10, 0.23))

    # --- 2. DEM3D configuration ----------------------------------------
    probe("dem3d-negative-radius", "DEM3DConfig(radius=-0.025)", "raise",
          dem3d(radius=-0.025))
    probe("dem3d-zero-radius", "DEM3DConfig(radius=0.0)", "raise", dem3d(radius=0.0))
    probe("dem3d-empty-lattice", "DEM3DConfig(nx=0)", "raise", dem3d(nx=0))
    probe("dem3d-zero-dt", "DEM3DConfig(dt=0.0)", "raise", dem3d(dt=0.0))
    probe("dem3d-negative-damping", "DEM3DConfig(contact_damping=-5.0)", "raise",
          dem3d(contact_damping=-5.0))
    probe("dem3d-negative-breaking-strain",
          "DEM3DConfig(breaking_strain=-0.015)", "raise",
          dem3d(breaking_strain=-0.015))
    probe("dem3d-valid-baseline", "DEM3DConfig() default must build and step", "accept",
          dem3d())

    # --- 3. NaN poisoning of a running state ---------------------------
    def nan_poison():
        sim = IceDEM3D(DEM3DConfig(nx=4, ny=3, nz=2))
        sim.positions[0, 0] = float("nan")
        sim.step()
        return float(torch.isfinite(sim.positions).all())

    probe("dem3d-nan-position",
          "NaN injected into a particle position should be detected and raise",
          "raise", nan_poison)

    # --- 4. brash-ice resistance input domain --------------------------
    def op(**over):
        base = dict(id="PROBE", ship=MT_UIKKU_HULL, ice_thickness_m=0.8,
                    ice_concentration=0.6, speed_m_s=3.0,
                    measured_resistance_kN=385.0)
        base.update(over)
        return OperatingPoint(**base)

    probe("brash-negative-concentration",
          "dem_resistance(C_ice=-0.5): a negative coverage must not yield R=0",
          "raise", lambda: dem_resistance(op(ice_concentration=-0.5)))
    probe("brash-concentration-1.5",
          "dem_resistance(C_ice=1.5) (impossible packing)", "raise",
          lambda: dem_resistance(op(ice_concentration=1.5)))
    probe("brash-negative-thickness",
          "dem_resistance(H=-1.0 m)", "raise",
          lambda: dem_resistance(op(ice_thickness_m=-1.0)))
    probe("brash-negative-speed",
          "brash_resistance(speed=-3.0 m/s)", "raise",
          lambda: brash_resistance(speed=-3.0))
    probe("brash-porosity-1", "BrashIceState(porosity=1.0)", "raise",
          lambda: BrashIceState(porosity=1.0))
    probe("brash-zero-thickness", "BrashIceState(layer_thickness_m=0.0)", "raise",
          lambda: BrashIceState(layer_thickness_m=0.0))
    probe("brash-negative-floe", "BrashIceState(floe_diameter_m=-0.5)", "raise",
          lambda: BrashIceState(floe_diameter_m=-0.5))

    # --- 5. zero-concentration negative control (documented criterion f)
    probe("brash-zero-concentration-control",
          "dem_resistance(C_ice=0) must return exactly 0 (documented criterion f)",
          "accept", lambda: dem_resistance(op(ice_concentration=0.0)))

    counts = {}
    for row in RESULTS:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    out = ROOT / "runs" / "negative_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"probes": RESULTS, "counts": counts}, indent=2) + "\n",
        encoding="utf-8")
    print("\nSUMMARY:", json.dumps(counts))
    print("findings (SILENT = invalid input silently accepted):")
    for row in RESULTS:
        if row["verdict"] in ("SILENT", "CRASH_UNEXPECTED"):
            print(f"  - {row['probe']}: {row['what']} -> {row['observed']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover - loud failure, no silent exit 0
        traceback.print_exc()
        raise SystemExit(2)