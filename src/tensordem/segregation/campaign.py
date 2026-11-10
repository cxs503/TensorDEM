"""Validation campaign driver for the size-segregation surface.

A *case* is one (configuration, driver, parameter, seed) point.  Every case
writes one JSON record so that the campaign can be sharded across cores (each
shard runs a disjoint subset of case names) and post-processed independently of
the compute that produced it.  No number in a report is typed by hand: the
records are the only source.

Case families
-------------
``vib``   vertically vibrated trough at a prescribed drive
          ``Gamma = A w^2 / g`` and frequency; used for the segregation index
          ``S(Gamma)``, the Brazil-nut rise velocity ``v_up`` and the threshold
          search.
``pour``  gravity pour: a gated block of the same mixture collapses into a heap;
          used for the heap-morphology quantities consumed by the ship side
          (surface/core split, run-out, floor pressure peaks).

The counters that make the run auditable live in ``record["audit"]``: total
mass and centre of mass (per-particle vs binned vs zonal reconstruction), the
containment check, the deterministic state digest, and the wall-clock cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..granular2d.engine import DiskConfig
from .bed2d import BidisperseBed, channel_walls
from .metrics import BedStatistics, mass_moment_audit, rise_velocity, segregation_indices
from .packing import bidisperse_radii, mixture_area, random_sequential_pack

GRAVITY = 9.81

__all__ = ["CaseSpec", "run_case", "build_spec", "main"]


@dataclass(frozen=True)
class CaseSpec:
    """One campaign point (all lengths in m, times in s)."""

    name: str
    drive: str = "vib"                 # "vib" | "pour"
    ratio: float = 2.0                 # d_large / d_small
    large_fraction: float = 0.30       # large-species solid-volume fraction
    gamma: float = 2.0                 # dimensionless vibration drive
    frequency: float = 30.0            # Hz
    seed: int = 0
    n_particles: int = 12000
    small_radius: float = 1.0e-3
    width: float = 0.40
    height: float = 0.18
    dt: float = 1.5e-5
    kn: float = 1.20e4
    kt_ratio: float = 0.50
    zeta: float = 0.30
    mu: float = 0.40
    mu_rolling: float = 0.05
    settle_time: float = 0.15
    run_time: float = 0.45
    settle_damping_time: float = 0.0
    sample_every: int = 150
    top_fraction: float = 0.25
    n_z_bins: int = 240
    n_x_zones: int = 8
    gate_fraction: float = 0.35        # pour: initial block width fraction
    fit_start_fraction: float = 0.10   # rise-velocity fit window (of run_time)
    stiffness_scale: float = 1.0       # kn multiplier (stiffness sensitivity)
    loose_fraction: float = 0.52        # RSA target before gravity compaction
    n_blocks: int = 8                  # temporal blocks for the sampling error of S
    snapshot_sample: int = 0            # >0: store a position subsample for figures

    def to_json(self) -> dict:
        return asdict(self)


def build_spec(**kwargs) -> CaseSpec:
    return CaseSpec(**kwargs)


def _digest(bed: BidisperseBed) -> str:
    h = hashlib.sha256()
    for arr in (bed.pos, bed.vel, bed.theta, bed.omega):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.float64(bed.time).tobytes())
    return h.hexdigest()


def _pack_loose(radii, box_width, seed, *, fractions, height_limit):
    """RSA packing of a loose specimen, backing off the target if it stalls.

    A high area fraction is unreachable by random sequential adsorption for some
    diameter ratios (the ``d_l/d_s = 3`` mixture stalls near 0.55), so the target
    is lowered step by step.  The realised fraction is returned and recorded: the
    preparation protocol must be visible in the artefact, not assumed.
    """
    area = mixture_area(radii)
    last = None
    for frac in fractions:
        height = area / (frac * box_width)
        if height > height_limit:
            last = f"target {frac:.2f} needs {height:.3f} m > {height_limit:.3f} m"
            continue
        try:
            pos = random_sequential_pack(radii, box_width, height, seed=seed,
                                         target_fraction=frac)
            return pos, height, frac
        except RuntimeError as exc:
            last = str(exc)
    raise RuntimeError(f"no admissible loose packing: {last}")


def _make_bed(spec: CaseSpec) -> tuple[BidisperseBed, np.ndarray, np.ndarray]:
    rng_key = spec.seed
    radii, is_large = bidisperse_radii(
        spec.n_particles, spec.ratio, spec.large_fraction,
        small_radius=spec.small_radius, seed=rng_key,
    )
    area = mixture_area(radii)
    fractions = [spec.loose_fraction, spec.loose_fraction - 0.04,
                 spec.loose_fraction - 0.08, spec.loose_fraction - 0.12]
    if spec.drive == "pour":
        block_w = spec.gate_fraction * spec.width
        pos, height, frac = _pack_loose(radii, block_w, rng_key + 1,
                                        fractions=fractions, height_limit=spec.height)
        walls, base, gate = channel_walls(spec.width, spec.height, margin=2e-3,
                                          gate_x=block_w)
    else:
        pos, height, frac = _pack_loose(radii, spec.width, rng_key + 1,
                                        fractions=fractions, height_limit=spec.height)
        walls, base, gate = channel_walls(spec.width, spec.height, margin=2e-3)
    cfg_settle = DiskConfig(
        kn=spec.kn * spec.stiffness_scale, kt_ratio=spec.kt_ratio, zeta=spec.zeta,
        mu=spec.mu, mu_rolling=spec.mu_rolling, density=1000.0, thickness=1.0,
        gravity=(0.0, -GRAVITY), skin=3.0e-4, dt=spec.dt,
        damping_time=spec.settle_damping_time,
    )
    bed = BidisperseBed(pos, radii, is_large, cfg_settle, walls=walls,
                        base_index=base, gate_index=gate)
    bed.prep_loose_height_m = float(height)
    bed.prep_loose_fraction = float(frac)
    return bed, radii, is_large


def _run_cfg(spec: CaseSpec) -> DiskConfig:
    return DiskConfig(
        kn=spec.kn * spec.stiffness_scale, kt_ratio=spec.kt_ratio, zeta=spec.zeta,
        mu=spec.mu, mu_rolling=spec.mu_rolling, density=1000.0, thickness=1.0,
        gravity=(0.0, -GRAVITY), skin=3.0e-4, dt=spec.dt, damping_time=0.0,
    )


def run_case(spec: CaseSpec) -> dict:
    """Run one case and return its record (JSON-serialisable)."""
    t_wall = time.time()
    bed, radii, is_large = _make_bed(spec)
    steps_settle = int(round(spec.settle_time / spec.dt))
    if spec.settle_damping_time > 0.0:
        # Damped, quasi-static compaction: the specimen is built by random
        # sequential adsorption at a loose fraction (~0.5), so a free-fall
        # collapse to the settled fraction (~0.82) would reach impact speeds at
        # which a soft contact (k_n chosen for a 2 % static overlap) is
        # penetrated by more than the particle radius, i.e. the run would blow
        # up for a numerical reason unrelated to the physics under test.  The
        # viscous term caps the settling speed at ``g * settle_damping_time``.
        # Settling stops early once the bed is quiescent *and* its height has
        # stopped changing, so a case never pays for settling time it does not
        # need.
        chunk = 4000
        done = 0
        h_prev = None
        while done < steps_settle:
            n = min(chunk, steps_settle - done)
            bed.step(n)
            done += n
            v_max = float(np.max(np.abs(bed.vel)))
            h_now = float(np.quantile(bed.pos[:, 1] + bed.radii, 0.995))
            if h_prev is not None and v_max < 0.02 and abs(h_now - h_prev) < 1.0e-3:
                break
            h_prev = h_now
        steps_settle = done
    else:
        bed.step(steps_settle)
    bed.config = _run_cfg(spec)
    bed.vel *= 0.0
    bed.omega *= 0.0

    z_edges = np.linspace(0.0, spec.height, spec.n_z_bins + 1)
    x_edges = np.linspace(0.0, spec.width, spec.n_x_zones + 1)
    stats = BedStatistics(z_edges=z_edges, x_edges=x_edges)
    # Temporal blocks: the bed is sampled into ``n_blocks`` consecutive windows so
    # that the segregation index carries a *measured* sampling error instead of an
    # assumed one.  The blocks are contiguous in time (not interleaved), so their
    # spread also exposes a slow drift (a bed that has not reached its steady
    # segregation state).
    n_blocks = max(1, int(spec.n_blocks))
    block_stats = [BedStatistics(z_edges=z_edges, x_edges=x_edges)
                   for _ in range(n_blocks)]
    base_sum = np.zeros(spec.n_x_zones)
    base_peak = np.zeros(spec.n_x_zones)
    base_cnt = np.zeros(spec.n_x_zones)
    n_samples = 0
    times: list[float] = []
    mean_z_large: list[float] = []
    mean_z_small: list[float] = []
    ke: list[float] = []
    tg: list[float] = []

    reference_measure = float(np.sum(bed.measure))
    z_com_initial = float(np.sum(bed.measure * bed.pos[:, 1]) / reference_measure)
    # Composition *before* the drive is switched on.  Gravity settling alone can
    # already produce a size gradient (kinetic sieving during the collapse), so
    # the initial state has to be measured instead of assumed mixed: the driven
    # segregation is then reported as Delta S = S(end) - S(start).
    stats_init = BedStatistics(z_edges=z_edges, x_edges=x_edges)
    stats_init.add(bed.pos, bed.measure, bed.is_large)
    settled_height = float(np.quantile(bed.pos[:, 1] + bed.radii, 0.995))
    started = bed.time
    if spec.drive == "vib":
        amplitude = bed.set_vibration_gamma(spec.gamma, spec.frequency, gravity=GRAVITY)
    else:
        amplitude = 0.0
        bed.open_gate()

    steps_run = int(round(spec.run_time / spec.dt))
    n_chunks = max(1, steps_run // spec.sample_every)
    for _ in range(n_chunks):
        bed.step(spec.sample_every)
        stats.add(bed.pos, bed.measure, bed.is_large)
        # block index from the *sample* counter (the samples are already spaced by
        # ``sample_every``, so dividing by it again would pin every sample to the
        # first block and silently destroy the measured sampling error)
        block_stats[min(n_samples * n_blocks // max(n_chunks, 1),
                        n_blocks - 1)].add(bed.pos, bed.measure, bed.is_large)
        times.append(bed.time)
        zl, zs = bed.species_mean_height()
        mean_z_large.append(zl)
        mean_z_small.append(zs)
        ke.append(bed.kinetic_energy())
        tg.append(bed.granular_temperature())
        if len(bed.base_contact_x):
            ix = np.clip(np.digitize(bed.base_contact_x, x_edges) - 1, 0, spec.n_x_zones - 1)
            zone_load = np.zeros(spec.n_x_zones)
            np.add.at(zone_load, ix, bed.base_contact_fn)
            base_sum += zone_load
            base_peak = np.maximum(base_peak, zone_load)
            base_cnt += np.bincount(ix, minlength=spec.n_x_zones)
        n_samples += 1

    times_a = np.asarray(times)
    zl_a = np.asarray(mean_z_large)
    seg = 1.0 / (spec.width * spec.n_x_zones)

    record = {
        "spec": spec.to_json(),
        "drive": spec.drive,
        "amplitude_m": float(amplitude),
        "gamma_realised": float(amplitude * (2 * math.pi * spec.frequency) ** 2 / GRAVITY)
        if spec.drive == "vib" else 0.0,
        "dt_s": bed.dt,
        "steps_settle": steps_settle,
        "steps_run": n_chunks * spec.sample_every,
        "time_start_s": float(started),
        "time_end_s": float(bed.time),
        "n_particles": int(bed.n),
        "n_large": int(bed.is_large.sum()),
        "solid_area_m2": reference_measure,
        "area_fraction": float(np.sum(bed.measure) / (spec.width * spec.height)),
        "prep_loose_fraction_requested": spec.loose_fraction,
        "prep_loose_fraction_realised": float(getattr(bed, "prep_loose_fraction", float("nan"))),
        "prep_loose_height_m": float(getattr(bed, "prep_loose_height_m", float("nan"))),
        "settled_bed_height_m": settled_height,
        "initial_z_com_m": z_com_initial,
        "series": {
            "time_s": times_a.tolist(),
            "mean_z_large_m": zl_a.tolist(),
            "mean_z_small_m": np.asarray(mean_z_small).tolist(),
            "kinetic_energy_J": ke,
            "granular_temperature_m2_s2": tg,
        },
        "base_load": {
            "zone_sum_N": base_sum.tolist(),
            "zone_peak_N": base_peak.tolist(),
            "zone_contact_samples": base_cnt.tolist(),
            "zone_length_m": [seg] * spec.n_x_zones,
            "mean_pressure_Pa": (base_sum / max(n_samples, 1) / seg).tolist(),
            "peak_pressure_Pa": (base_peak / seg).tolist(),
        },
        "stats_initial": {
            "z_area": stats_init.z_area.tolist(),
            "z_large": stats_init.z_large.tolist(),
            "n_samples": stats_init.n_samples,
        },
        "stats": {
            "z_area": stats.z_area.tolist(),
            "z_large": stats.z_large.tolist(),
            "z_moment": stats.z_moment.tolist(),
            "x_area": stats.x_area.tolist(),
            "x_large": stats.x_large.tolist(),
            "x_moment": stats.x_moment.tolist(),
            "x_moment2": stats.x_moment2.tolist(),
            "x_z_moment": stats.x_z_moment.tolist(),
            "large_z_sum": stats.large_z_sum,
            "large_count": stats.large_count,
            "small_z_sum": stats.small_z_sum,
            "small_count": stats.small_count,
            "total_measure": stats.total_measure,
            "total_moment": stats.total_moment,
            "n_samples": stats.n_samples,
            "z_edges": z_edges.tolist(),
            "x_edges": x_edges.tolist(),
        },
        "stats_blocks": [
            {
                "z_area": b.z_area.tolist(),
                "z_large": b.z_large.tolist(),
                "z_moment": b.z_moment.tolist(),
                "x_area": b.x_area.tolist(),
                "total_measure": b.total_measure,
                "n_samples": b.n_samples,
            }
            for b in block_stats
        ],
    }
    z_com_final = float(np.sum(bed.measure * bed.pos[:, 1]) / np.sum(bed.measure))
    if spec.snapshot_sample > 0:
        rng = np.random.default_rng(spec.seed + 7)
        take = rng.choice(bed.n, size=min(spec.snapshot_sample, bed.n), replace=False)
        record["snapshot"] = {
            "pos": bed.pos[take].tolist(),
            "radii": bed.radii[take].tolist(),
            "is_large": bed.is_large[take].tolist(),
            "width_m": spec.width,
            "settled_bed_height_m": record["settled_bed_height_m"],
        }
    record["audit"] = {
        "digest": _digest(bed),
        "wall_clock_s": time.time() - t_wall,
        "final_bed_height_m": float(np.quantile(bed.pos[:, 1] + bed.radii, 0.995)),
        "z_com_initial_m": z_com_initial,
        "z_com_final_m": z_com_final,
    }
    record["metrics"] = derive_metrics(record)
    return record


def segregation_blocks(record: dict, spec: "CaseSpec") -> dict:
    """Segregation index and rise speed evaluated on the temporal blocks.

    The mean of the per-block values is the best estimate of the steady value;
    their spread is the *measured* sampling error of the case, which is what the
    negative control of criterion (a) has to be judged against (a fixed 0.02
    tolerance is only meaningful if the estimator noise is smaller than it).
    """
    blocks = record.get("stats_blocks") or []
    s_fine: list[float] = []
    s_bn: list[float] = []
    times = np.asarray(record["series"]["time_s"], dtype=np.float64)
    zl = np.asarray(record["series"]["mean_z_large_m"], dtype=np.float64)
    n_chunks = times.size
    n_blk = max(1, len(blocks))
    v_up: list[float] = []
    for i, blk in enumerate(blocks):
        if blk.get("n_samples", 0) < 1:
            continue
        st = BedStatistics(z_edges=np.asarray(record["stats"]["z_edges"], dtype=np.float64),
                           x_edges=np.asarray(record["stats"]["x_edges"], dtype=np.float64))
        st.z_area = np.asarray(blk["z_area"], dtype=np.float64)
        st.z_large = np.asarray(blk["z_large"], dtype=np.float64)
        st.z_moment = np.asarray(blk["z_moment"], dtype=np.float64)
        st.x_area = np.asarray(blk["x_area"], dtype=np.float64)
        st.total_measure = float(blk["total_measure"])
        st.total_moment = float(np.sum(st.z_moment))
        st.n_samples = int(blk["n_samples"])
        try:
            idx = segregation_indices(st, top_fraction=spec.top_fraction,
                                      h_ref_m=record.get("settled_bed_height_m"))
        except ValueError:
            # The fixed geometric reference (the settled height) can be above the
            # whole bed in a block where the specimen is still compacting: the
            # block then falls back to its own detected surface instead of being
            # dropped from the sample.
            try:
                idx = segregation_indices(st, top_fraction=spec.top_fraction)
            except ValueError:
                continue
        s_fine.append(idx["S_fine"])
        s_bn.append(idx["S_bn"])
        lo = i * n_chunks // n_blk
        hi = max(lo + 2, (i + 1) * n_chunks // n_blk)
        if hi <= n_chunks:
            try:
                v_up.append(rise_velocity(times[lo:hi], zl[lo:hi], t_start=times[lo])["v_up_m_s"])
            except ValueError:
                pass
    out: dict = {"n_blocks": len(blocks), "S_fine": s_fine, "S_bn": s_bn,
                 "v_up_m_s": v_up}
    for key in ("S_fine", "S_bn", "v_up_m_s"):
        arr = np.asarray(out[key], dtype=np.float64)
        if arr.size:
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            # a single block carries no information about the spread: report the
            # sampling error as undefined rather than as a fake zero
            out[f"{key}_stderr"] = (out[f"{key}_std"] / math.sqrt(arr.size)
                                    if arr.size > 1 else float("nan"))
    return out


def stats_from_record(record: dict) -> BedStatistics:
    """Rebuild the :class:`BedStatistics` container from a record (no physics)."""
    st = record["stats"]
    stats = BedStatistics(
        z_edges=np.asarray(st["z_edges"], dtype=np.float64),
        x_edges=np.asarray(st["x_edges"], dtype=np.float64),
    )
    for name in ("z_area", "z_large", "z_moment", "x_area", "x_large",
                 "x_moment", "x_moment2", "x_z_moment"):
        setattr(stats, name, np.asarray(st[name], dtype=np.float64))
    stats.large_z_sum = float(st["large_z_sum"])
    stats.large_count = int(st["large_count"])
    stats.small_z_sum = float(st["small_z_sum"])
    stats.small_count = int(st["small_count"])
    stats.total_measure = float(st["total_measure"])
    stats.total_moment = float(st["total_moment"])
    stats.n_samples = int(st["n_samples"])
    return stats


def derive_metrics(record: dict) -> dict:
    """All derived quantities of a case, computed from the record alone.

    Separated from the integration so that a post-processing change never has to
    re-run a simulation, and so a sharded run writes a self-describing record.
    """
    spec = CaseSpec(**record["spec"])
    stats = stats_from_record(record)
    times = np.asarray(record["series"]["time_s"], dtype=np.float64)
    zl = np.asarray(record["series"]["mean_z_large_m"], dtype=np.float64)
    zs = np.asarray(record["series"]["mean_z_small_m"], dtype=np.float64)
    fit_start = spec.fit_start_fraction * (times[-1] - times[0]) + times[0]
    seg_idx = segregation_indices(
        stats, top_fraction=spec.top_fraction,
        h_ref_m=record.get("settled_bed_height_m"),
    )
    rise = rise_velocity(times, zl, t_start=fit_start)
    rise_small = rise_velocity(times, zs, t_start=fit_start)
    blocks = segregation_blocks(record, spec)
    seg_initial = {}
    st0 = record.get("stats_initial")
    if st0 and st0.get("n_samples", 0) > 0:
        b0 = BedStatistics(z_edges=stats.z_edges, x_edges=stats.x_edges)
        b0.z_area = np.asarray(st0["z_area"], dtype=np.float64)
        b0.z_large = np.asarray(st0["z_large"], dtype=np.float64)
        b0.n_samples = int(st0["n_samples"])
        try:
            seg_initial = segregation_indices(
                b0, top_fraction=spec.top_fraction,
                h_ref_m=record.get("settled_bed_height_m"))
        except ValueError:
            seg_initial = {}
    audit = mass_moment_audit(stats, density=1000.0, thickness=1.0)
    profile = stats.profile()
    return {
        "segregation": seg_idx,
        "segregation_initial": seg_initial,
        "delta_S_fine": (float(seg_idx["S_fine"] - seg_initial["S_fine"])
                         if seg_initial else None),
        "delta_S_bn": (float(seg_idx["S_bn"] - seg_initial["S_bn"])
                       if seg_initial else None),
        "segregation_blocks": blocks,
        "rise_large": rise,
        "rise_small": rise_small,
        "audit": audit,
        "initial_z_com_m": record["audit"]["z_com_initial_m"],
        "final_z_com_instant_m": record["audit"]["z_com_final_m"],
        "dz_com_m": record["audit"]["z_com_final_m"] - record["audit"]["z_com_initial_m"],
        "mass_conservation_rel": float(audit["mass_conservation_rel"]),
        "z_com_consistency_rel": float(audit["z_com_consistency_rel"]),
        "mean_granular_temperature": float(
            np.mean(record["series"]["granular_temperature_m2_s2"])
        ),
        "profile_z_centres": profile["z_centres"].tolist(),
        "profile_solid_fraction": profile["solid_fraction"].tolist(),
        "profile_large_fraction": profile["large_fraction_of_solid"].tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorDEM size-segregation campaign")
    ap.add_argument("--spec", required=True, help="JSON file with a list of case specs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--shard", default="0/1", help="i/n shard selector")
    ap.add_argument("--only", default=None, help="comma-separated case names")
    args = ap.parse_args(argv)

    i, n = (int(v) for v in args.shard.split("/"))
    with open(args.spec) as fh:
        specs = [CaseSpec(**s) for s in json.load(fh)]
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s.name in keep]
    specs = [s for k, s in enumerate(sorted(specs, key=lambda s: s.name)) if k % n == i]
    os.makedirs(args.out, exist_ok=True)
    for spec in specs:
        path = os.path.join(args.out, f"{spec.name}.json")
        if os.path.exists(path):
            print(f"[skip] {spec.name}", flush=True)
            continue
        record = run_case(spec)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(record, fh)
        os.replace(tmp, path)
        print(f"[done] {spec.name} wall={record['audit']['wall_clock_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())