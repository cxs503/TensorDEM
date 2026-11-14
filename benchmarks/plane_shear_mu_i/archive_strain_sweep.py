"""Round 16 (m20855) -- archive the round-15 strain-convergence sweep.

Round 15 launched six live strain-convergence runs (``phi_estimator_audit.py
--live --I ... --strain 2.0``, seeds 5279, smooth walls) as cluster jobs
``cj2950 .. cj2955``.  They *all succeeded* but the session ended before the
outputs were collected, so round 15's document declared the measurement
"not returned" and made no claim about steady-state ``phi(I)``.  This module
pulls those six artifacts out of the cluster tarballs, records their
provenance, and downsamples the long time series so they can live in the repo.

Why a downsample at all: the device samples every ``sample_stride`` steps with
``sample_stride = max(1, steps // 400)``, where ``steps`` is the *planned* shear
budget.  Because that budget is small at high ``I`` (515 steps at ``I = 0.3``)
the stride collapses to 1 and the run emits one sample per step -- 20 309
samples, a 4.7 MB JSON.  That is collateral damage of the same strain-budget
defect this round documents (see ``strain_convergence_audit.py``), not a
deliberate choice.  For archival we keep every scalar verbatim and thin the
series to <= ``KEEP_POINTS`` uniformly, recording the exact factor and the
original length so the thinning is reversible in provenance terms.

Usage
-----
    python benchmarks/plane_shear_mu_i/archive_strain_sweep.py \
        --src /tmp/cj2950 /tmp/cj2951 ... --out benchmarks/plane_shear_mu_i/runs/phi_audit
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

KEEP_POINTS = 500
JOBS = {2950: 0.001, 2951: 0.003, 2952: 0.01, 2953: 0.03, 2954: 0.1, 2955: 0.3}

# scalar fields copied verbatim (no rounding, no re-derivation)
SCALAR_KEYS = (
    "I_target", "wall_mode", "strain_requested", "strain_achieved",
    "n_particles", "dt", "shear_steps_executed", "wall_clock_s",
    "phi_container_final", "phi_extent_final", "phi_centres_final",
    "phi_bed_steady", "slab_over_bed_final", "boundary_layer_factor",
    "mu_full_final", "mu_wall_final", "I_imposed", "I_bulk_final",
    "pressure_control_error_final", "servo_valid", "coordination",
    "steady_phi", "steady_mu", "drift_abs", "drift_rel_pct",
)
SERIES_KEYS = ("strain", "solid_fraction", "phi_bed_now", "phi_extent",
               "slab_over_bed", "full_mu", "mu", "plate_z",
               "normal_stress_pa")


def find_run(src_dir: str) -> str:
    hits = glob.glob(os.path.join(
        src_dir, "**", "plane_shear_mu_i", "runs", "phi_audit",
        "I*_smooth_*.json"), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"{src_dir}: expected 1 run json, found {len(hits)}")
    return hits[0]


def thin(series: list, factor: int) -> list:
    """Uniform decimation that always keeps the first and last sample."""
    if factor <= 1 or len(series) <= KEEP_POINTS:
        return list(series)
    out = series[::factor]
    if out and out[-1] is not series[-1]:
        out.append(series[-1])
    return out


def archive_one(src_dir: str, job: int, out_dir: str) -> dict:
    path = find_run(src_dir)
    with open(path) as fh:
        raw = json.load(fh)

    series = raw.pop("series", {}) or {}
    n_orig = max((len(v) for v in series.values()), default=0)
    factor = max(1, math.ceil(n_orig / KEEP_POINTS))

    rec = {k: raw[k] for k in SCALAR_KEYS if k in raw}
    rec["provenance"] = {
        "cluster_job": f"cj{job}",
        "source_artifact": os.path.relpath(path, src_dir),
        "archived_by": "archive_strain_sweep.py (round 16, m20855)",
        "series_downsample_factor": factor,
        "series_length_original": n_orig,
        "series_length_archived": len(thin(series.get("strain", []), factor)),
        "note": ("scalars verbatim; series uniformly decimated to <= "
                 f"{KEEP_POINTS} points keeping first/last sample"),
    }
    rec["series"] = {k: thin(v, factor) for k, v in series.items()
                     if k in SERIES_KEYS}

    tag = f"I{rec['I_target']:g}_smooth_s5279".replace("I0.", "I0.")
    name = f"I{rec['I_target']:g}_smooth_s5279.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    rec["_archived_as"] = name
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="extracted cjNNNN tarball directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--job-ids", nargs="*", type=int,
                    help="job ids, parallel to --src (default cj2950..2955)")
    args = ap.parse_args(argv)

    jobs = args.job_ids or sorted(JOBS)
    if len(jobs) != len(args.src):
        raise SystemExit("--job-ids and --src must have equal length")

    rows = []
    for src, job in zip(args.src, jobs):
        r = archive_one(src, job, args.out)
        rows.append(r)
        p = r["provenance"]
        print(f"cj{job} I={r['I_target']:g} strain={r['strain_achieved']:7.2f} "
              f"series {p['series_length_original']} -> "
              f"{p['series_length_archived']} (/{p['series_downsample_factor']})"
              f" -> {r['_archived_as']}")

    with open(os.path.join(args.out, "strain_sweep_archive_index.json"), "w") as fh:
        json.dump({"jobs": [r["provenance"]["cluster_job"] for r in rows],
                   "points": [{"I_target": r["I_target"],
                               "strain_achieved": r["strain_achieved"],
                               "file": r["_archived_as"]} for r in rows]},
                  fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())