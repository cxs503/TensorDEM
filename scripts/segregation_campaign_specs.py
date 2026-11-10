#!/usr/bin/env python3
"""Generate the case matrix of the size-segregation validation campaign.

The matrix is written to a JSON file consumed by
``scripts/run_segregation_campaign.py`` (one process per case).  Generating it
from code keeps the design reproducible: the families below are the whole
design, and the resulting file is a build artefact, not a hand-edited list.

Families
--------
``ratio``   diameter-ratio sweep at a fixed drive, three seeds each (criteria b, f)
``gamma``   drive sweep at fixed ratio (criterion d: threshold and saturation)
``fraction`` large-species loading sweep (design parameter of the node)
``freq``    frequency sweep at fixed ``Gamma`` (drive is not a single variable)
``pour``    gravity pour, gate release (independent driver, heap morphology)

Units are SI.  All cases of one family share the contact model and timestep, so
the only difference between them is the parameter under test.
"""

from __future__ import annotations

import argparse
import json
import os

GRAVITY = 9.81


def case(name: str, *, drive: str = "vib", ratio: float = 2.0,
         large_fraction: float = 0.30, gamma: float = 3.0, frequency: float = 20.0,
         seed: int = 0, n_particles: int = 4000, small_radius: float = 4.0e-3,
         width: float = 0.60, height: float = 1.0, dt: float = 1.8e-5,
         cycles: float = 50.0, settle_time: float = 4.0,
         sample_per_cycle: int = 10, **extra) -> dict:
    """One campaign point.

    ``cycles`` fixes the duration in *vibration periods* rather than seconds, so
    the same family comparison is not biased by the frequency sweep.  The pour
    cases read it as a plain duration in seconds.
    """
    if drive == "vib":
        run_time = cycles / frequency
        sample_every = max(25, int(round(1.0 / (frequency * dt) / sample_per_cycle)))
    else:
        run_time = cycles
        sample_every = max(25, int(round(2.0e-4 / dt)))
    spec = dict(
        name=name, drive=drive, ratio=ratio, large_fraction=large_fraction,
        gamma=gamma, frequency=frequency, seed=seed, n_particles=n_particles,
        small_radius=small_radius, width=width, height=height, dt=dt,
        kn=2.0e5, kt_ratio=0.5, zeta=0.3, mu=0.4, mu_rolling=0.05,
        settle_time=settle_time, run_time=run_time, settle_damping_time=1.0e-2,
        sample_every=sample_every, top_fraction=0.25, n_z_bins=240, n_x_zones=8,
        gate_fraction=0.35, fit_start_fraction=0.15, stiffness_scale=1.0,
        loose_fraction=0.55, n_blocks=8, snapshot_sample=4000,
    )
    spec.update(extra)
    return spec


def build(full: bool = True) -> list[dict]:
    out: list[dict] = []
    ratios = (1.0, 1.5, 2.0, 3.0, 4.0)
    seeds = (0, 1, 2) if full else (0,)
    gamma_drive = 3.0
    freq = 15.0

    # ---- ratio family (three seeds) ----------------------------------
    for ratio in ratios:
        for seed in seeds:
            out.append(case(f"vib_R{ratio:.1f}_g{gamma_drive:.0f}_f{freq:.0f}_s{seed}",
                            ratio=ratio, gamma=gamma_drive, frequency=freq, seed=seed))
    # damped settling variant: does the driven segregation depend on a
    # segregation that already happened during the gravity compaction?
    out.append(case(f"vib_R2.0_g{gamma_drive:.0f}_f{freq:.0f}_damp", ratio=2.0,
                    gamma=gamma_drive, frequency=freq, settle_damping_time=2.0e-3))

    # ---- drive family (threshold + saturation) ------------------------
    for gamma in (0.75, 1.0, 1.5, 2.0, 2.5, 4.0):
        out.append(case(f"vib_R2.0_g{gamma:.2f}_f{freq:.0f}_s0", ratio=2.0,
                        gamma=gamma, frequency=freq, seed=0,
                        cycles=70.0 if gamma < 1.0 else 50.0))

    # ---- large-fraction family ---------------------------------------
    if full:
        for fraction in (0.20, 0.40):
            for ratio in (2.0, 4.0):
                out.append(case(
                    f"vib_R{ratio:.1f}_g{gamma_drive:.0f}_f{freq:.0f}_phi{fraction:.1f}_s0",
                    ratio=ratio, gamma=gamma_drive, frequency=freq, seed=0,
                    large_fraction=fraction))

    # ---- frequency family at fixed Gamma -----------------------------
    for f in (25.0, 40.0, 60.0):
        out.append(case(f"vib_R2.0_g{gamma_drive:.0f}_f{f:.0f}_s0", ratio=2.0,
                        gamma=gamma_drive, frequency=f, seed=0))

    # ---- gravity pour (independent driver) ---------------------------
    pour_seeds = (0, 1) if full else (0,)
    for ratio in ratios:
        for seed in pour_seeds:
            out.append(case(f"pour_R{ratio:.1f}_s{seed}", drive="pour", ratio=ratio,
                            seed=seed, height=1.4, gate_fraction=0.8,
                            cycles=2.2, settle_time=4.0))

    # de-duplicate by name, keeping the first definition
    seen: dict[str, dict] = {}
    for spec in out:
        seen.setdefault(spec["name"], spec)
    return sorted(seen.values(), key=lambda s: s["name"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="one seed, no fraction family")
    args = ap.parse_args(argv)
    specs = build(full=not args.quick)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(specs, fh, indent=1)
    print(f"{len(specs)} cases -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())