#!/usr/bin/env python3
"""CASE-DEM-BRAZIL-NUT-SEGREGATION: one DEM point of the Möbius et al. (2001) case.

Reference (primary source, read from the publisher's figure page, not copied from
a secondary summary):

  M.E. Möbius, B.E. Lauderdale, S.R. Nagel & H.M. Jaeger,
  "Size separation of granular particles", Nature 414, 270 (2001),
  DOI 10.1038/35104697 -- Fig. 1 ("Density and size dependence of the Brazil-nut
  effect").  Caption parameters: bed of d = 0.5 mm glass spheres
  (rho_m = 2.4 g/ml), fill height 8.6 cm, container diameter 8.2 cm,
  well-separated sinusoidal taps with Gamma = A(2 pi f)^2/g = 5 at f = 13 Hz,
  spherical intruder of diameter D = 2.54 cm (D/d = 50.8) whose TOP starts
  z0 = 4.6 cm below the free surface; the measured quantity is the rising time
  T_rise versus density ratio rho/rho_m and size ratio D/d.

The experiment is not reproduced at 1:1 scale (D/d = 50.8 would need ~1e4 bed
particles across the depth); this harness runs a *geometrically similar* 2D
cell at reduced D/d, using the existing TensorDEM 2D frictional-disk engine
(:mod:`tensordem.granular2d.engine`) and the vibrated bed of
:mod:`tensordem.segregation.bed2d` -- no new solver is introduced.

Similarity controls (all dimensionless, recorded in the artefact):

* ``Gamma = A omega^2 / g``           -- 5.0   (same as the reference)
* ``D/H_bed``                         -- 0.295 (reference 2.54/8.6 = 0.295)
* ``W/H_bed``                         -- 0.95  (reference 8.2/8.6 = 0.95)
* ``(depth of intruder top)/H_bed``   -- 0.535 (reference 4.6/8.6 = 0.535)
* ``rho/rho_m``                       -- swept (the reference's abscissa)
* ``D/d``                             -- reduced, declared as the scale gap

Output: one JSON record per case in the run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular2d.engine import DiskConfig  # noqa: E402
from tensordem.segregation.bed2d import BidisperseBed, channel_walls  # noqa: E402

G = 9.81
RHO_BED = 2400.0          # kg/m^3, glass beads of the reference (2.4 g/ml)
# reference geometry ratios
D_OVER_H = 2.54 / 8.6
W_OVER_H = 8.2 / 8.6
TOP_DEPTH_OVER_H = 4.6 / 8.6
GAMMA = 5.0
FREQ_HZ = 13.0


# --------------------------------------------------------------------- packing
def rsa_pack(r_s: float, width: float, height: float, *, n_target: int,
             seed: int, clearance: float = 0.0) -> np.ndarray:
    """Random sequential adsorption of equal disks (loose specimen before settling)."""
    rng = np.random.default_rng(seed)
    target = int(n_target)
    pos = np.empty((0, 2))
    attempts = 0
    max_attempts = 200 * target + 10000
    while len(pos) < target and attempts < max_attempts:
        attempts += 1
        cand = np.array([rng.uniform(r_s + clearance, width - r_s - clearance),
                         rng.uniform(r_s, height - r_s)])
        if len(pos):
            d2 = np.sum((pos - cand) ** 2, axis=1)
            if d2.min() < (2.0 * r_s + clearance) ** 2:
                continue
        pos = np.vstack([pos, cand])
    if len(pos) < target:
        raise RuntimeError(f"RSA stalled at {len(pos)}/{target} disks")
    return pos


def _surface_height(pos: np.ndarray, radii: np.ndarray, mask: np.ndarray) -> float:
    tops = pos[mask, 1] + radii[mask]
    return float(np.quantile(tops, 0.995))


def build_bed(*, d_small: float, size_ratio: float, seed: int = 0,
              loose_fraction: float = 0.52, settle_time: float = 0.30,
              settle_damping_time: float = 5.0e-4, kn: float = 3.0e4,
              dt: float = 2.0e-5, **kwargs) -> dict:
    """Pack + gravity-compact a bed of small disks, then embed one large intruder."""
    r_s = 0.5 * d_small
    r_l = 0.5 * size_ratio * d_small
    # Cell sized so that W/H_bed and D/H_bed match the reference.  The number of
    # small disks follows from the *settled* solid fraction; the box is made
    # 1.7 H tall so the specimen can be packed loose (RSA) and then compacted.
    phi_settled_target = 0.82
    D = 2.0 * r_l
    H = D / D_OVER_H
    W = W_OVER_H * H
    n_target = int(round(phi_settled_target * W * H / (math.pi * r_s**2)))
    h_box = 1.7 * H
    pos = rsa_pack(r_s, W, h_box, n_target=n_target, seed=seed)

    walls, base, gate = channel_walls(W, h_box, margin=1.0e-3)
    radii = np.full(len(pos), r_s)
    is_large = np.zeros(len(pos), dtype=bool)
    cfg = DiskConfig(kn=kn, kt_ratio=0.5, zeta=0.3, mu=0.4, mu_rolling=0.05,
                     density=RHO_BED, thickness=1.0, gravity=(0.0, -G),
                     skin=3.0 * r_s, dt=dt, damping_time=settle_damping_time)
    bed = BidisperseBed(pos, radii, is_large, cfg, walls=walls,
                        base_index=base, gate_index=gate)
    bed.step(int(round(settle_time / dt)))
    # pin the (static) walls back to a box that hugs the settled specimen so that
    # the free surface is at a known height
    z_surf = _surface_height(bed.pos, bed.radii, np.ones(bed.n, dtype=bool))
    return {"bed": bed, "r_s": r_s, "r_l": r_l, "W": W, "h_box": h_box,
            "z_surf_pre": z_surf, "n_small": int(bed.n), "kn": kn, "dt": dt}


def embed_intruder(bed_state: dict, *, rho_ratio: float, seed: int = 0) -> dict:
    """Replace the small disks under the free surface by one large intruder.

    The intruder top is placed ``TOP_DEPTH_OVER_H * H`` below the measured free
    surface, i.e. the reference's initial depth.  Removed small disks are
    reported (their count and area) so the mass bookkeeping of the preparation
    is auditable.
    """
    bed = bed_state["bed"]
    r_s, r_l = bed_state["r_s"], bed_state["r_l"]
    z_surf = bed_state["z_surf_pre"]
    H = z_surf - bed.radii.min()
    z_c = z_surf - TOP_DEPTH_OVER_H * H - r_l
    rho_l = rho_ratio * RHO_BED
    keep = np.ones(bed.n, dtype=bool)
    dist = np.hypot(bed.pos[:, 0] - 0.5 * bed_state["W"], bed.pos[:, 1] - z_c)
    keep &= dist >= (r_l + r_s)
    removed = int((~keep).sum())
    removed_area = float(np.sum(bed.measure[~keep]))

    pos = np.vstack([bed.pos[keep], np.array([[0.5 * bed_state["W"], z_c]])])
    radii = np.concatenate([bed.radii[keep], [r_l]])
    is_large = np.zeros(len(pos), dtype=bool)
    is_large[-1] = True
    cfg = bed.config
    bed2 = BidisperseBed(pos, radii, is_large, cfg, walls=bed.walls,
                         base_index=bed.base_index, gate_index=bed.gate_index)
    bed2.time = bed.time
    # per-particle densities: the intruder's mass is set from rho_l
    thickness = cfg.thickness
    mass = RHO_BED * math.pi * bed2.radii**2 * thickness
    mass[-1] = rho_l * math.pi * r_l**2 * thickness
    bed2.mass = mass
    bed2.inertia = 0.5 * mass * bed2.radii**2
    return {"bed": bed2, "z_c0": float(z_c), "removed_small": removed,
            "removed_area": removed_area, "rho_l": rho_l, "H0": float(H)}


# ---------------------------------------------------------------------- driving
def tap_drive(bed: BidisperseBed, *, amplitude: float, frequency: float,
              dt: float, taps: int, rest_fraction: float = 0.5,
              sample_every: int = 50, stop_when_surfaced: bool = True) -> dict:
    """Well-separated sinusoidal taps, one period each, plus a quiet rest phase.

    One tap = ``1/f`` of base motion ``A sin(omega t)`` starting (and ending) at
    the bottom of the stroke, followed by ``rest_fraction/f`` of rest.  This is
    the protocol of the reference (``Gamma`` is held fixed); a continuous sine
    would instead test a fluidised steady state.
    """
    omega = 2.0 * math.pi * frequency
    steps_tap = int(round((1.0 / frequency) / dt))
    steps_rest = int(round(rest_fraction / frequency / dt))
    is_large = bed.is_large
    series = {"tap": [], "time_s": [], "z_intruder_m": [], "z_small_mean_m": [],
              "surface_m": [], "ke_J": [], "ke_samples": []}
    t0 = bed.time
    surfaced_at = None
    snapshots = {"initial": _snapshot(bed)}
    for k in range(1, taps + 1):
        # phase so that the stroke starts at the bottom of the oscillation
        bed.set_vibration(amplitude=amplitude, frequency=omega,
                          phase=-omega * bed.time)
        bed.step(steps_tap)
        bed.stop_vibration()
        n_chunks = max(1, steps_rest // sample_every)
        for _ in range(n_chunks):
            bed.step(sample_every)
            series["ke_samples"].append(bed.kinetic_energy())
        pos, radii = bed.pos, bed.radii
        z_l = float(pos[is_large, 1].mean())
        z_s = float(pos[~is_large, 1].mean())
        surf = _surface_height(pos, radii, ~is_large)
        series["tap"].append(k)
        series["time_s"].append(float(bed.time - t0))
        series["z_intruder_m"].append(z_l)
        series["z_small_mean_m"].append(z_s)
        series["surface_m"].append(surf)
        series["ke_J"].append(float(bed.kinetic_energy()))
        if surfaced_at is None and (z_l + radii[is_large][0]) >= surf:
            surfaced_at = k
            if stop_when_surfaced:
                break
    snapshots["final"] = _snapshot(bed)
    return {"series": series, "surfaced_at_tap": surfaced_at, "snapshots": snapshots,
            "steps_tap": steps_tap, "steps_rest": steps_rest,
            "amplitude_m": amplitude, "frequency_hz": frequency}


def _snapshot(bed: BidisperseBed) -> dict:
    """Particle configuration of one instant (for the field figure)."""
    return {"pos": np.asarray(bed.pos).tolist(),
            "radii": np.asarray(bed.radii).tolist(),
            "is_large": np.asarray(bed.is_large).tolist(),
            "time_s": float(bed.time)}


def _digest(bed: BidisperseBed) -> str:
    h = hashlib.sha256()
    for arr in (bed.pos, bed.vel, bed.radii, bed.mass):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def run_case(*, size_ratio: float, rho_ratio: float, d_small: float = 2.0e-3,
             seed: int = 0, max_taps: int = 40, kn: float = 3.0e4,
             dt: float = 2.0e-5, settle_time: float = 0.30,
             gamma: float = GAMMA, frequency: float = FREQ_HZ) -> dict:
    t_wall = time.time()
    state = build_bed(d_small=d_small, size_ratio=size_ratio, seed=seed,
                      settle_time=settle_time, kn=kn, dt=dt)
    emb = embed_intruder(state, rho_ratio=rho_ratio, seed=seed)
    bed = emb["bed"]
    amplitude = gamma * G / (2.0 * math.pi * frequency) ** 2
    drive = tap_drive(bed, amplitude=amplitude, frequency=frequency, dt=dt,
                      taps=max_taps)
    pos = bed.pos
    is_large = bed.is_large
    h0 = emb["H0"]
    om = 2.0 * math.pi * frequency
    tstep = 1.0 / frequency
    rec = {
        "case": {
            "name": f"brazil_nut_R{size_ratio:g}_rho{rho_ratio:g}_s{seed}",
            "size_ratio": float(size_ratio),
            "rho_ratio": float(rho_ratio),
            "d_small_m": float(d_small),
            "D_intruder_m": float(2 * state["r_l"]),
            "W_m": float(state["W"]),
            "H0_m": float(h0),
            "seed": int(seed),
        },
        "dimensionless": {
            "Gamma": float(gamma),
            "f_Hz": float(frequency),
            "A_over_d": float(amplitude / d_small),
            "A_over_D": float(amplitude / (2 * state["r_l"])),
            "D_over_d": float(size_ratio),
            "D_over_H0": float(2 * state["r_l"] / h0),
            "W_over_H0": float(state["W"] / h0),
            "rho_over_rho_m": float(rho_ratio),
            "m_intruder_over_m_small": float(
                (rho_ratio * state["r_l"] ** 2) / state["r_s"] ** 2),
            "a_peak_over_g": float(amplitude * om**2 / G),
            "t_tap_over_sqrt_d_over_g": float(tstep / math.sqrt(d_small / G)),
            "dt_over_contact_time": float(
                dt / (math.pi * math.sqrt(0.5 * RHO_BED * math.pi * state["r_s"] ** 2
                                          / kn))),
        },
        "bed": {
            "n_particles": int(bed.n),
            "n_small_after_embed": int((~is_large).sum()),
            "removed_small_for_intruder": emb["removed_small"],
            "removed_area_m2": emb["removed_area"],
            "area_fraction": float(np.sum(bed.measure) / (state["W"] * bed.walls[:, 1, 1].max())),
        },
        "preparation": {
            "kn_N_m": float(kn), "dt_s": float(dt), "settle_time_s": float(settle_time),
            "z_intruder_center_initial_m": emb["z_c0"],
            "surface_initial_m": float(state["z_surf_pre"]),
            "top_depth_initial_m": float(state["z_surf_pre"] - emb["z_c0"]
                                         - state["r_l"]),
        },
        "drive": {k: v for k, v in drive.items()
                  if k not in ("series", "snapshots")},
        "series": drive["series"],
        "snapshots": drive["snapshots"],
        "audit": {
            "wall_clock_s": time.time() - t_wall,
            "digest": _digest(bed),
            "final_surface_m": float(_surface_height(pos, bed.radii, ~is_large)),
            "final_z_intruder_m": float(pos[is_large, 1].mean()),
            "final_z_small_mean_m": float(pos[~is_large, 1].mean()),
            "n_steps_total": int(round(bed.time / dt)),
        },
        "metrics": summarise(drive["series"], bed, emb),
    }
    return rec


def summarise(series: dict, bed: BidisperseBed, emb: dict) -> dict:
    """Rise-time and layering descriptors (defined before the sweep was run)."""
    tap = np.asarray(series["tap"], dtype=float)
    z = np.asarray(series["z_intruder_m"], dtype=float)
    surf = np.asarray(series["surface_m"], dtype=float)
    zs = np.asarray(series["z_small_mean_m"], dtype=float)
    r_l = float(bed.radii[bed.is_large][0])
    d_s = float(2.0 * bed.radii[~bed.is_large].mean())
    H0 = emb["H0"]
    depth = surf - (z + r_l)                      # >0: intruder still buried
    n = min(len(tap), len(depth))
    rose = 0
    for k in range(n):
        if depth[k] <= 0.0:
            rose = int(tap[k])
            break
    dz = np.diff(z)
    dz_over_d = dz / d_s
    return {
        "z_intruder_over_H0_initial": float((z[0] - bed.radii.min()) / H0),
        "z_intruder_over_H0_final": float((z[-1] - bed.radii.min()) / H0),
        "surface_over_H0_final": float(surf[-1] / H0),
        "depth_initial_over_D": float(depth[0] / (2 * r_l)),
        "depth_final_over_D": float(depth[-1] / (2 * r_l)),
        "net_rise_m": float(z[-1] - z[0]),
        "net_rise_over_d": float((z[-1] - z[0]) / d_s),
        "rise_per_tap_over_d_mean": float(np.mean(dz_over_d)) if dz.size else 0.0,
        "rise_per_tap_over_d_median": float(np.median(dz_over_d)) if dz.size else 0.0,
        "rise_per_tap_over_d_std": float(np.std(dz_over_d)) if dz.size else 0.0,
        "n_taps_run": int(len(tap)),
        "surfaced_at_tap": (int(rose) if rose else None),
        # layering index: excess height of the intruder over the small-species
        # centroid, in bed heights. 0 = fully mixed, ->0.5 = intruder on top.
        "layering_index_initial": float((z[0] - zs[0]) / H0),
        "layering_index_final": float((z[-1] - zs[-1]) / H0),
        "direction": ("rise" if z[-1] - z[0] > 0 else "sink"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size-ratio", type=float, required=True)
    p.add_argument("--rho-ratio", type=float, required=True)
    p.add_argument("--d-small", type=float, default=2.0e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taps", type=int, default=40)
    p.add_argument("--kn", type=float, default=3.0e4)
    p.add_argument("--dt", type=float, default=2.0e-5)
    p.add_argument("--settle-time", type=float, default=0.30)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args(argv)
    rec = run_case(size_ratio=a.size_ratio, rho_ratio=a.rho_ratio,
                   d_small=a.d_small, seed=a.seed, max_taps=a.taps,
                   kn=a.kn, dt=a.dt, settle_time=a.settle_time)
    out = a.out or os.path.join(HERE, "runs", rec["case"]["name"] + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({"case": rec["case"], "metrics": rec["metrics"],
                      "wall_clock_s": rec["audit"]["wall_clock_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())