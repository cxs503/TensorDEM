"""Emit the JSON spec list of the size-segregation validation campaign.

The matrix is organised so that every acceptance criterion of the node is backed
by a dedicated family, and every family shares the same preparation protocol
(same packing algorithm, same settle time, same sampling) so that differences
between cases are differences of physics, not of protocol.

Families
--------
``ctl``    ``d_l/d_s = 1`` equal-size negative control (3 seeds)
``ratio``  ``d_l/d_s in {1.5, 2, 3, 4}`` at fixed drive, 3 seeds
``gamma``  drive sweep ``Gamma in {0.5 ... 5}`` at 60 Hz
``freq``   frequency sweep 15-60 Hz at ``Gamma = 1.5``
``frac``   large-species volume fraction 0.2 / 0.4
``stiff``  contact-stiffness sensitivity (kn x 4, dt / 2)
``pour``   gravity pour of a gated block (heap morphology family)
``size``   particle-count check at 6000 particles

Geometry note
-------------
The vibrated trough is deliberately **wide and shallow** (1.2 m x 0.18 m for
2x10^4 disks of 1-4 mm): a cargo-hold-like layer, no side-wall convection roll,
so whatever segregation is measured comes from size percolation and not from a
convection cell.  This is stated in the report as a scope limitation.
"""

from __future__ import annotations

import argparse
import json

BASE = dict(
    small_radius=1.0e-3,
    dt=1.5e-5,
    kn=1.20e4,
    mu=0.40,
    mu_rolling=0.05,
    zeta=0.30,
    settle_time=0.22,
    settle_damping_time=0.0,
    run_time=0.40,
    sample_every=150,
    top_fraction=0.25,
    n_z_bins=240,
    n_x_zones=8,
    n_particles=20000,
    width=1.20,
    height=0.18,
    loose_fraction=0.50,
)

POUR = dict(
    BASE,
    height=0.40,
    run_time=0.75,
    gate_fraction=0.50,
    sample_every=200,
)

SHOWCASE = ("ratio_R3.0_G3.0_f60_s0", "ctl_R1.0_G3.0_f60_s0", "pour_R1.0_s0", "pour_R3.0_s0")


def build() -> list[dict]:
    specs: list[dict] = []

    def add(**kw) -> None:
        spec = dict(BASE)
        spec.update(kw)
        spec.setdefault("name", "case")
        if spec["name"] in SHOWCASE:
            spec["snapshot_sample"] = 4000
        specs.append(spec)

    # A. equal-size negative control
    for seed in (0, 1, 2):
        add(name=f"ctl_R1.0_G3.0_f60_s{seed}", ratio=1.0, gamma=3.0, frequency=60.0, seed=seed)

    # B. diameter-ratio family
    for ratio in (1.5, 2.0, 3.0, 4.0):
        for seed in (0, 1, 2):
            add(name=f"ratio_R{ratio:.1f}_G3.0_f60_s{seed}", ratio=ratio, gamma=3.0,
                frequency=60.0, seed=seed)

    # C. drive sweep at fixed frequency
    for gamma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for seed in (0, 1):
            add(name=f"gamma_R2.0_G{gamma:.1f}_f60_s{seed}", ratio=2.0, gamma=gamma,
                frequency=60.0, seed=seed)

    # D. frequency sweep at fixed drive (the 60 Hz point already exists)
    for freq in (15.0, 20.0, 30.0, 45.0):
        add(name=f"freq_R2.0_G1.5_f{freq:.0f}_s0", ratio=2.0, gamma=1.5, frequency=freq,
            seed=0)

    # E. composition sensitivity
    for frac in (0.2, 0.4):
        for seed in (0, 1):
            add(name=f"frac_R3.0_fv{frac:.1f}_G3.0_f60_s{seed}", ratio=3.0,
                large_fraction=frac, gamma=3.0, frequency=60.0, seed=seed)

    # F. contact-stiffness sensitivity (kn x 4 -> dt / 2 for stability)
    add(name="stiff_R3.0_G3.0_f60_s0", ratio=3.0, gamma=3.0, frequency=60.0, seed=0,
        stiffness_scale=4.0, dt=7.5e-6, run_time=0.30)

    # G. gravity pour
    for ratio in (1.0, 2.0, 3.0, 4.0):
        for seed in (0, 1):
            spec = dict(POUR)
            spec.update(dict(name=f"pour_R{ratio:.1f}_s{seed}", drive="pour", ratio=ratio,
                             gamma=0.0, frequency=0.0, seed=seed))
            if spec["name"] in SHOWCASE:
                spec["snapshot_sample"] = 4000
            specs.append(spec)

    # H. particle-count check (same physics, quarter of the particles)
    add(name="size_R3.0_G3.0_f60_N6000_s0", ratio=3.0, gamma=3.0, frequency=60.0, seed=0,
        n_particles=6000, width=0.65, height=0.12, loose_fraction=0.50)
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    specs = build()
    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate case names")
    with open(args.out, "w") as fh:
        json.dump(specs, fh, indent=1)
    print(f"wrote {len(specs)} case specs to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())