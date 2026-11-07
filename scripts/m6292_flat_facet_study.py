"""m6292: how wrong is the superquadric fixed-point contact on flat facets?

Programmatic evidence for ``docs/superquadric_flat_facet_reference.md``.
Everything is written to ``artifacts/m6292/`` (JSON + PNG); the doc quotes only
those numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tensordem.convex_contact import Box, contact as gjk_epa_contact
from tensordem.superquadric import (
    SuperquadricBatch,
    _quat_to_matrix,
    superquadric_contact_batch,
)
from tensordem.superquadric_convex import (
    SuperquadricShape,
    contact_by_direction_min,
    superquadric_shape,
    support_distance,
    support_point,
    analytic_sphere_limit,
    analytic_block_limit,
)

R = 0.01
OUT = Path("artifacts/m6292")
SWEEPS = (1, 2, 3, 4, 6, 8, 12, 20)


def _batch(ext, n, center, rot):
    return SuperquadricBatch(np.array([ext[0]]), np.array([ext[1]]), np.array([ext[2]]),
                             np.array([n]), np.array([n]), np.asarray(center)[None],
                             np.asarray(rot)[None])


def _shapes(ext, n, ca, cb, ra, rb):
    return (superquadric_shape(*ext, n, n, ca, ra), superquadric_shape(*ext, n, n, cb, rb))


def e1_anchors() -> dict:
    """The reference against the two closed-form limits."""
    rng = np.random.default_rng(3)
    sphere, block = [], []
    for _ in range(8):
        ra = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
        rb = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
        ca = rng.normal(size=3) * 0.05
        cb = ca + rng.normal(size=3) * 0.05
        sa, sb = _shapes((R, R, R), 2.0, ca, cb, ra, rb)
        got = contact_by_direction_min(sa, sb, n_dirs=8000)
        ref = analytic_sphere_limit(sa, sb)
        sphere.append(abs(got["gap"] + ref["gap"]) / R)

        ext = (R * rng.uniform(0.8, 2.0), R, R * rng.uniform(0.6, 1.0))
        cb2 = ca + rng.normal(size=3) * 0.6 * R
        sa, sb = _shapes(ext, 60.0, ca, cb2, ra, rb)
        got = contact_by_direction_min(sa, sb, n_dirs=8000)
        ref = analytic_block_limit(sa, sb)
        block.append(abs(got["gap"] + ref["gap"]) / R)
    return {"sphere_limit_max_err_over_R": float(max(sphere)),
            "block_limit_max_err_over_R": float(max(block)),
            "n_cases_each": 8, "grid_dirs": 20000}


def _study_cases(n_cases: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        n = float(rng.choice([2.0, 3.0, 6.0, 8.0, 16.0]))
        ext = np.array([R * rng.uniform(1.0, 3.0), R, R * rng.uniform(0.5, 1.0)])
        ra = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
        rb = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
        ca = rng.normal(size=3) * 0.02
        cb = ca + rng.normal(size=3) * 0.3 * R + np.array([0.0, 0.0, 2.0 * R * 0.98])
        cases.append((n, ext, ca, cb, ra, rb))
    return cases


def e2_kernel_debt() -> dict:
    """Fixed point (production sweeps) and EPA, both scored on the reference."""
    rows = []
    for n, ext, ca, cb, ra, rb in _study_cases():
        ba = _batch(ext, n, ca, ra)
        bb = _batch(ext, n, cb, rb)
        sweep = [float(superquadric_contact_batch(ba, bb, n_iter=it)["overlap"][0])
                 for it in SWEEPS]
        sa, sb = _shapes(ext, n, ca, cb, ra, rb)
        ref = contact_by_direction_min(sa, sb, n_dirs=3000)
        epa = gjk_epa_contact(Box(ca, ext, ra), Box(cb, ext, rb))
        epa_gap = epa.depth if not epa.separated else -epa.distance
        rows.append({
            "n": n, "ext": ext.tolist(), "overlap_by_sweep": sweep,
            "reference_gap": ref["gap"], "epa_gap": float(epa_gap),
            "err_n4_over_R": abs(sweep[SWEEPS.index(4)] - ref["gap"]) / R,
            "err_n20_over_R": abs(sweep[SWEEPS.index(20)] - ref["gap"]) / R,
            "err_epa_over_R": abs(float(epa_gap) - ref["gap"]) / R,
            "spread_after_3_over_R": (max(sweep[2:]) - min(sweep[2:])) / R,
            "sign_flip": bool((sweep[SWEEPS.index(4)] > 0.0) != (ref["gap"] > 0.0)),
        })
    def agg(key):
        return float(np.max([r[key] for r in rows]))
    return {
        "rows": rows,
        "max_err_n4_over_R": agg("err_n4_over_R"),
        "max_err_n20_over_R": agg("err_n20_over_R"),
        "max_err_epa_over_R": agg("err_epa_over_R"),
        "max_spread_after_3_over_R": agg("spread_after_3_over_R"),
        "median_err_n4_over_R": float(np.median([r["err_n4_over_R"] for r in rows])),
        "sign_flips_vs_reference": int(sum(r["sign_flip"] for r in rows)),
        "contacts_gt_5pct_R_n4": int(sum(r["err_n4_over_R"] > 0.05 for r in rows)),
        "contacts_gt_5pct_R_epa": int(sum(r["err_epa_over_R"] > 0.05 for r in rows)),
    }


def e3_penetration_sweep() -> dict:
    """Sweep-count spread of the fixed point as a function of the overlap/R."""
    rng = np.random.default_rng(19)
    rows = []
    for n in (8.0, 16.0):
        for ratio in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
            spreads, ref_errs = [], []
            for _ in range(3):
                ra = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
                rb = _quat_to_matrix(rng.normal(size=(1, 4)))[0]
                ca = np.zeros(3)
                cb = np.array([0.3, -0.2, 0.0]) * R + np.array([0.0, 0.0, (2.0 - ratio) * R])
                ext = (R * 1.2, R, R * 0.9)
                ba, bb = _batch(ext, n, ca, ra), _batch(ext, n, cb, rb)
                vals = [float(superquadric_contact_batch(ba, bb, n_iter=it)["overlap"][0])
                        for it in SWEEPS]
                spreads.append((max(vals[2:]) - min(vals[2:])) / R)
                sa, sb = _shapes(ext, n, ca, cb, ra, rb)
                ref = contact_by_direction_min(sa, sb, n_dirs=2000)
                ref_errs.append(abs(vals[SWEEPS.index(4)] - ref["gap"]) / R)
            rows.append({"n": n, "overlap_over_R": ratio,
                         "spread_after_3_over_R": float(max(spreads)),
                         "err_at_4_over_R": float(max(ref_errs))})
    return {"rows": rows}


def e4_cost() -> dict:
    ext = (R, R, R)
    ba = _batch(ext, 16.0, np.zeros(3), np.eye(3), np.eye(3))
    bb = _batch(ext, 16.0, np.array([0.0, 0.0, 1.98 * R]), np.eye(3), np.eye(3))
    sa, sb = _shapes(ext, 16.0, np.zeros(3), np.array([0.0, 0.0, 1.98 * R]), np.eye(3), np.eye(3))
    t0 = time.perf_counter()
    for _ in range(200):
        superquadric_contact_batch(ba, bb, n_iter=4, compute_radius=False)
    kernel_us = (time.perf_counter() - t0) / 200 * 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        gjk_epa_contact(Box(np.zeros(3), ext), Box(np.array([0.0, 0.0, 1.98 * R]), ext))
    epa_us = (time.perf_counter() - t0) / 20 * 1e6
    t0 = time.perf_counter()
    for _ in range(5):
        contact_by_direction_min(sa, sb, n_dirs=3000)
    ref_us = (time.perf_counter() - t0) / 5 * 1e6
    return {"kernel_us_per_pair": kernel_us, "epa_us_per_pair": epa_us,
            "direction_min_6000_us_per_pair": ref_us}


def figures(study) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = study["e2"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    idx = np.arange(len(rows))
    eps = 1e-12
    ax.semilogy(idx, [max(r["err_n4_over_R"], eps) for r in rows], "o", ms=4,
                label="fixed point, n_iter=4 (production)")
    ax.semilogy(idx, [max(r["err_n20_over_R"], eps) for r in rows], "s", ms=3.5,
                label="fixed point, n_iter=20")
    ax.semilogy(idx, [max(r["err_epa_over_R"], eps) for r in rows], "^", ms=3.5,
                label="GJK/EPA (convex_contact)")
    ax.axhline(0.01, color="k", ls="--", lw=1, label="1 % of R")
    ax.set_xlabel("case index (40 random poses, contact_convergence_study setup)")
    ax.set_ylabel("|reported gap - reference gap| / R")
    ax.set_title("Superquadric contact error against the direction-minimisation reference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_reference.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for n, style in ((8.0, "o-"), (16.0, "s--")):
        pts = [r for r in study["e3"]["rows"] if r["n"] == n]
        ax.plot([p["overlap_over_R"] for p in pts], [p["spread_after_3_over_R"] for p in pts],
                style, ms=5, label=f"n1=n2={n:g}: sweep spread (n_iter>=3)")
        ax.plot([p["overlap_over_R"] for p in pts], [p["err_at_4_over_R"] for p in pts],
                style, ms=3, alpha=0.55, label=f"n1=n2={n:g}: |err| at n_iter=4")
    ax.axhline(1e-3, color="k", ls="--", lw=1, label="documented criterion 1e-3 R")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("overlap / R")
    ax.set_ylabel("overlap error / R")
    ax.set_title("Flat-facet error grows with penetration: the fixed point is not contractive")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "spread_vs_penetration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    worst = sorted(rows, key=lambda r: -r["spread_after_3_over_R"])[:4]
    for k, r in enumerate(worst):
        ax.plot(SWEEPS, np.array(r["overlap_by_sweep"]) / R, "o-", ms=3,
                label=f"case n={r['n']:g} (ref {r['reference_gap'] / R:+.3f} R)")
        ax.axhline(r["reference_gap"] / R, color=f"C{k}", ls=":", lw=1)
    ax.set_xlabel("fixed-point sweeps")
    ax.set_ylabel("reported overlap / R")
    ax.set_title("Same pair, different sweep count: overlap wanders (dotted = reference)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_traces.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    study = {
        "e1_anchors": e1_anchors(),
        "e2_kernel_debt": e2_kernel_debt(),
        "e3_penetration_sweep": e3_penetration_sweep(),
        "e4_cost": e4_cost(),
    }
    figures(study)
    payload = {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v)
               for k, v in study.items()}
    (OUT / "flat_facet_study_summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "flat_facet_study.json").write_text(json.dumps(study, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
