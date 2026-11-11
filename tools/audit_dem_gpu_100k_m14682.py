#!/usr/bin/env python3
"""Board-audit reproducer for mission m14682 (GAP-DEM-GPU-100K).

Independent re-measurement of the two claims that had **no raw run artifact** in
m14682 (they existed only as prose in artifacts/gap_dem_gpu_100k/attempt_ledger.json):

  * "step_time_cpu_1e5_dem3d": 6.645 s (legacy) -> 3.570 s (device backends)
  * "contact_search_speedup_cpu_1e5": 3.267 s -> 0.726 s (4.5x), exact same pair set

Design (predeclared before running):
  T1 timing @ 101 614 particles (47x47x46, the mission's large lattice):
     arm A = DEM3DConfig(neighbor_search="legacy",   tangential_store="dict")
     arm B = DEM3DConfig(neighbor_search="vectorized", tangential_store="dense")
     1 warm-up step + 2 timed steps per arm, median reported; both arms get the
     same seed, lattice and 0.995 pre-compression.
  T2 contact-set equality: legacy cell list vs device-native candidate_pairs, and
     both vs an exact O(N^2) distance criterion at N=2000 (a 1e5^2 float64 matrix
     would need ~82 GB, so the exact reference is only taken at small N — the
     mission's own test does the same).
  T3 GPU leg: probe only (no CUDA in this container), so the GAP criterion
     "median step at >=1e5 particles <= 2 s on GPU" is NOT_RUN here too.

Usage:
  python3 tools/audit_dem_gpu_100k_m14682.py \
      --repo-root /root/tp-agent/work/m14682 \
      --outdir artifacts/audit_dem_gpu_100k
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True,
                    help="worktree that contains the m14682 branch code (src/tensordem)")
    ap.add_argument("--outdir", default="artifacts/audit_dem_gpu_100k")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--small-n", type=int, default=2000)
    ap.add_argument("--lattice", default="47,47,46",
                    help="nx,ny,nz (default = the mission's 101 614 lattice)")
    ap.add_argument("--only", default="both", choices=("both", "legacy", "vectorized"),
                    help="run a single arm (keeps one invocation under the 120 s shell cap)")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root / "src"))

    import torch  # noqa: E402  (after sys.path so the audited repo wins)
    from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
    from tensordem.dem3d_pair_search import candidate_pairs  # noqa: E402

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    radius, dt, compression = 0.02, 1e-7, 0.995
    lat = tuple(int(x) for x in args.lattice.split(","))

    def build(nx: int, ny: int, nz: int, neighbor: str, tangential: str) -> "IceDEM3D":
        cfg = DEM3DConfig(nx=nx, ny=ny, nz=nz, radius=radius, dt=dt, device="cpu",
                          friction_coefficient=0.3,
                          neighbor_search=neighbor, tangential_store=tangential)
        sim = IceDEM3D(cfg)
        sim.positions *= compression
        return sim

    def timed(sim: "IceDEM3D") -> dict:
        for _ in range(args.warmup):
            sim.step()
        per = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            sim.step()
            per.append(time.perf_counter() - t0)
        return {"step_seconds": [round(v, 6) for v in per],
                "median": round(statistics.median(per), 6),
                "mean": round(statistics.mean(per), 6),
                "min": round(min(per), 6), "max": round(max(per), 6)}

    out: dict = {
        "audit_of_mission": 14682,
        "gap": "GAP-DEM-GPU-100K",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(root),
        "hardware": {"hostname": platform.node(), "machine": platform.machine(),
                     "torch_num_threads": torch.get_num_threads(),
                     "torch": torch.__version__, "python": platform.python_version(),
                     "device": "cpu", "cuda_available": torch.cuda.is_available()},
        "protocol": {"timing_lattice": [lat[0], lat[1], lat[2]],
                     "particles": lat[0] * lat[1] * lat[2],
                     "radius": radius, "dt": dt, "compression": compression,
                     "warmup_steps": args.warmup, "timed_steps": args.steps,
                     "statistic": "median of timed steps"},
        "arms": {},
    }

    # ---- T1: 1e5 timing, legacy vs device-native (same lattice/seed) ----
    arms = {"legacy": ("legacy", "dict"), "vectorized": ("vectorized", "dense")}
    if args.only != "both":
        arms = {args.only: arms[args.only]}
    for arm, (nb, ts) in arms.items():
        t0 = time.perf_counter()
        sim = build(lat[0], lat[1], lat[2], nb, ts)
        rec = {"build_seconds": round(time.perf_counter() - t0, 3),
               "particles": int(sim.positions.shape[0])}
        rec.update(timed(sim))
        cutoff = 2.0 * radius
        t0 = time.perf_counter()
        pairs = (candidate_pairs(sim.positions, cutoff).to(torch.long)
                 if nb == "vectorized" else sim._contact_candidate_pairs())
        rec["pair_search_seconds"] = round(time.perf_counter() - t0, 6)
        rec["n_candidate_pairs"] = int(pairs.shape[0])
        out["arms"][arm] = rec

    # ---- T2: exact-reference pair-set check at small N ----
    small_legacy = build(12, 13, 13, "legacy", "dict")          # 2028 particles
    small_device = build(12, 13, 13, "vectorized", "dense")
    width = 2.0 * radius
    pairs_legacy = small_legacy._contact_candidate_pairs()
    pairs_new = candidate_pairs(small_device.positions, width).to(torch.long)
    pos = small_device.positions.double()
    d = torch.cdist(pos, pos)
    iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    mask = d[iu[0], iu[1]] < width
    brute = torch.stack([iu[0][mask], iu[1][mask]], dim=1)

    def rowset(p: torch.Tensor) -> set:
        p = p.cpu()
        lo = torch.minimum(p[:, 0], p[:, 1])
        hi = torch.maximum(p[:, 0], p[:, 1])
        return set(zip(lo.tolist(), hi.tolist()))

    s_new, s_brute, s_legacy = rowset(pairs_new), rowset(brute), rowset(pairs_legacy)
    out["pair_set_check_small_n"] = {
        "particles": int(pos.shape[0]),
        "n_pairs_legacy": int(pairs_legacy.shape[0]),
        "n_pairs_vectorized": int(pairs_new.shape[0]),
        "n_pairs_brute": int(brute.shape[0]),
        "vectorized_equals_brute": s_new == s_brute,
        "only_in_vectorized": len(s_new - s_brute),
        "missing_from_vectorized": len(s_brute - s_new),
        "legacy_subset_of_brute": s_legacy <= s_brute,
        "legacy_subset_of_vectorized": s_legacy <= s_new,
        "pair_count_1e5_legacy_vs_vectorized": [
            out["arms"]["legacy"]["n_candidate_pairs"] if "legacy" in out["arms"] else None,
            out["arms"]["vectorized"]["n_candidate_pairs"] if "vectorized" in out["arms"] else None],
        "same_count_at_1e5": (len(out["arms"]) == 2 and
                              out["arms"]["legacy"]["n_candidate_pairs"]
                              == out["arms"]["vectorized"]["n_candidate_pairs"]),
    }

    out["comparison"] = {
        "step_speedup_x": (round(out["arms"]["legacy"]["median"]
                                 / max(out["arms"]["vectorized"]["median"], 1e-9), 2)
                          if len(out["arms"]) == 2 else None),
        "pair_search_speedup_x": (round(out["arms"]["legacy"]["pair_search_seconds"]
                                        / max(out["arms"]["vectorized"]["pair_search_seconds"], 1e-9), 2)
                                  if len(out["arms"]) == 2 else None),
        "target_step_seconds_at_1e5_gpu": 2.0,
        "gpu_step_seconds": "NOT_RUN (torch.cuda.is_available()=False in this container)",
        "gap_criterion_judged": False,
        "doc_claim_legacy_step_seconds": 6.645,
        "doc_claim_vectorized_step_seconds": 3.570,
    }

    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    fp = od / f"audit_repro_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(json.dumps({"comparison": out["comparison"],
                      "pair_set_check_small_n": out["pair_set_check_small_n"]}, indent=2))
    print("wrote", fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())