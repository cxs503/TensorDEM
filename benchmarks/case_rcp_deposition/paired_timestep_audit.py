#!/usr/bin/env python3
"""Common-random-numbers (paired) estimator for the RCP timestep-bias gate.

Why this module exists
----------------------
``AUDIT_m22824`` §3 judged the temporal-convergence gate at the four
``(mu, k_n)`` levels of ``CASE-DEM-RCP-PACKING-FRACTION`` and returned
``UNDECIDABLE`` on three of them.  The estimator it used is the **unpaired**
Welch interval on ``Delta = phi(dt/2) - phi(dt)``, i.e. the two timestep levels
were treated as two independent populations.  They are not independent.

:func:`tensordem.deposition_packing.DepositionSpec.box_width` and
``initial_height`` are functions of ``(n, radius, target_phi_guess,
target_aspect, initial_phi)`` only -- they do not involve ``dt_frac`` -- and
:func:`tensordem.deposition_packing.initial_configuration` consumes only
``(n, box_width, height, radius, seed, min_sep_factor)``.  So a coarse run and a
fine run **at the same seed start from the identical particle configuration**:
the seed is a common random number, and the seed-to-seed scatter (the whole
reason m22558 needed so many seeds) is a shared nuisance parameter that
*cancels* in the per-seed difference

    Delta_i = phi_fine(seed i) - phi_coarse(seed i).

That is the estimator implemented here.  It is not a new physical claim and it
does not relax any tolerance: ``TOL_BIAS``, ``CONF`` and the three-way verdict
vocabulary are imported unchanged from :mod:`seed_bias_audit` (m22558 /
m21401), so a level can only move from ``UNDECIDABLE`` to ``CONVERGED`` /
``BIASED`` because the *interval got narrower*, not because the band moved.

Pairing premise is checked, not assumed
---------------------------------------
A paired interval is only valid if the two runs of a pair really share their
initial condition.  Two independent checks are applied to every pair:

1. **code-level**: ``DepositionSpec.box_width`` / ``initial_height`` are equal
   at the coarse and fine ``dt_frac`` (unit-tested in
   ``tests/test_rcp_paired_timestep.py``);
2. **artefact-level**: the run JSON's ``placement`` record -- which carries
   ``drop_height``, ``initial_height`` and ``rejected_candidates``, the latter
   being a very sharp fingerprint of the random sequential addition -- must be
   numerically identical for the two runs of a pair.

A pair failing (2) is dropped from the estimator and reported
``PAIRING_PREMISE_FAILED``; it is never silently averaged in.

Cost/benefit of pairing
-----------------------
:func:`variance_reduction` reports, on the same data, the paired standard error
against the unpaired Welch one, and :func:`seeds_needed_paired` gives the seed
budget the paired estimator needs for a stated margin.  Both numbers are part of
the audit output so a reader can check the claim "pairing halves the interval"
instead of taking it on faith.

Usage
-----
::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/paired_timestep_audit.py --json
    PYTHONPATH=src python benchmarks/case_rcp_deposition/paired_timestep_audit.py \
        --write benchmarks/case_rcp_deposition/runs/paired_timestep_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from seed_bias_audit import (  # noqa: E402  (path set above)
    CONF,
    DT_COARSE,
    DT_FINE,
    N_PRIMARY,
    TOL_BIAS,
    classify_bias,
    levels_present,
    load_runs,
    seeds_needed,
    welch_diff,
)

OUT_JSON = os.path.join(HERE, "runs", "paired_timestep_audit.json")

#: verdict vocabulary is *not* extended here; this one is a premise failure
PAIRING_PREMISE_FAILED = "PAIRING_PREMISE_FAILED"

__all__ = [
    "PAIRING_PREMISE_FAILED",
    "placement_fingerprint",
    "pair_by_seed",
    "paired_diff",
    "variance_reduction",
    "seeds_needed_paired",
    "audit_level",
    "audit",
]

#: placement keys compared for the artefact-level pairing check
FINGERPRINT_KEYS = ("drop_height", "initial_height", "min_separation",
                    "placement", "placement_phi", "requested_initial_phi",
                    "rejected_candidates")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _raw_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def placement_fingerprint(path: str) -> dict:
    """The placement record of one run, keyed by the fields that must agree.

    ``rejected_candidates`` is included because it counts every rejected
    random sequential addition candidate: it changes if the seed, the box or
    the separation rule changes, and it is insensitive to the integrator.
    """
    d = _raw_json(path)
    rec = d.get("placement") or {}
    return {k: rec.get(k) for k in FINGERPRINT_KEYS}


def _same_fingerprint(a: dict, b: dict) -> bool:
    for k in FINGERPRINT_KEYS:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not math.isclose(float(va), float(vb), rel_tol=0.0,
                                abs_tol=1e-12):
                return False
        elif va != vb:
            return False
    return True


def pair_by_seed(runs, mu: float, kn: float, n: int = N_PRIMARY,
                 check_premise: bool = True) -> list:
    """Per-seed ``(coarse, fine)`` pairs at one ``(mu, k_n)`` level.

    Each pair carries ``delta = phi_fine - phi_coarse`` and a ``premise_ok``
    flag.  Runs whose placement fingerprint disagrees (or is missing) are
    returned with ``premise_ok=False`` so the caller can count them rather
    than lose them silently.
    """
    sel = [r for r in runs if r["n"] == n and r["mu"] == mu and r["kn"] == kn]
    by_seed: dict = {}
    for r in sel:
        by_seed.setdefault(r["seed"], {})[r["dt_frac"]] = r
    pairs = []
    for seed in sorted(by_seed):
        coarse = by_seed[seed].get(DT_COARSE)
        fine = by_seed[seed].get(DT_FINE)
        if coarse is None or fine is None:
            continue
        if check_premise:
            fa = placement_fingerprint(os.path.join(ROOT, coarse["path"]))
            fb = placement_fingerprint(os.path.join(ROOT, fine["path"]))
            ok = _same_fingerprint(fa, fb)
        else:  # pragma: no cover - explicit opt-out, used only by unit tests
            fa = fb = {}
            ok = True
        pairs.append({
            "seed": int(seed),
            "phi_coarse": float(coarse["phi"]),
            "phi_fine": float(fine["phi"]),
            "delta": float(fine["phi"] - coarse["phi"]),
            "settled_coarse": bool(coarse["settled"]),
            "settled_fine": bool(fine["settled"]),
            "premise_ok": bool(ok),
            "placement_coarse": fa,
            "placement_fine": fb,
            "path_coarse": coarse["path"],
            "path_fine": fine["path"],
        })
    return pairs


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_diff(pairs, tol: float = TOL_BIAS, conf: float = CONF) -> dict:
    """Paired interval on the mean per-seed difference ``Delta``.

    Only pairs with ``premise_ok`` enter the estimate; the count of dropped
    pairs is reported.  With fewer than two usable pairs the standard error is
    undefined and the verdict degenerates to the point estimate -- the same
    convention ``seed_bias_audit.classify_bias`` uses, flagged here through
    ``resolved_from="point"``.
    """
    usable = [p for p in pairs if p.get("premise_ok", True)]
    dropped = [p["seed"] for p in pairs if not p.get("premise_ok", True)]
    d = np.array([p["delta"] for p in usable], dtype=float)
    n = int(d.size)
    out = {
        "error_model": "paired_crn",
        "n_pairs": n,
        "seeds": [int(p["seed"]) for p in usable],
        "deltas": [float(x) for x in d],
        "dropped_seeds_premise_failed": dropped,
        "tol": float(tol),
        "conf": float(conf),
        "mean_coarse": float(np.mean([p["phi_coarse"] for p in usable]))
        if n else None,
        "mean_fine": float(np.mean([p["phi_fine"] for p in usable]))
        if n else None,
    }
    if n == 0:
        out.update(delta=None, sem=None, df=None, verdict="NOT_RUN",
                   half_width=None, lo=None, hi=None,
                   resolved_from="none")
        return out
    delta = float(d.mean())
    out["delta"] = delta
    if n < 2:
        out.update(sem=None, df=None, half_width=None, lo=None, hi=None)
        out.update(classify_bias(delta, None, None, tol=tol, conf=conf))
        out["std_paired"] = None
        return out
    sd = float(d.std(ddof=1))
    sem = sd / math.sqrt(n)
    df = float(n - 1)
    out.update(std_paired=sd, sem=sem, df=df)
    out.update(classify_bias(delta, sem, df, tol=tol, conf=conf))
    return out


def variance_reduction(pairs, tol: float = TOL_BIAS, conf: float = CONF) -> dict:
    """Paired vs unpaired (Welch) standard error on the *same* pairs.

    ``ratio = sem_unpaired / sem_paired`` is the interval-width factor the
    pairing buys.  A ratio near 1 means the two runs of a pair do not share
    their noise (the premise is not buying anything, even if it holds
    formally); a ratio > 1 is the expected cancellation.
    """
    usable = [p for p in pairs if p.get("premise_ok", True)]
    n = len(usable)
    if n < 2:
        return {"n_pairs": n, "sem_paired": None, "sem_unpaired": None,
                "ratio": None}
    d = np.array([p["delta"] for p in usable], dtype=float)
    sem_paired = float(d.std(ddof=1) / math.sqrt(n))
    a = np.array([p["phi_coarse"] for p in usable], dtype=float)
    b = np.array([p["phi_fine"] for p in usable], dtype=float)
    unpaired = welch_diff(a, b)
    sem_unpaired = unpaired.get("sem")
    ratio = (float(sem_unpaired) / sem_paired
             if sem_unpaired not in (None, 0.0) and sem_paired > 0 else None)
    return {"n_pairs": n, "sem_paired": sem_paired,
            "sem_unpaired": sem_unpaired, "ratio": ratio,
            "unpaired_error_model": unpaired.get("error_model")}


def seeds_needed_paired(sigma: float, margin: float, conf: float = CONF,
                        n_max: int = 200) -> int:
    """Pairs needed so the paired 95 % half-width falls below ``margin``.

    ``t_{n-1} * sigma / sqrt(n) < margin``.  Returns ``n_max`` when the margin
    is not reachable within ``n_max``, and 1 for a degenerate/zero scatter.
    """
    if not np.isfinite(sigma) or sigma <= 0.0 or margin <= 0.0:
        return 1
    for n in range(2, n_max + 1):
        hw = float(stats.t.ppf(0.5 + conf / 2.0, n - 1)) * sigma / math.sqrt(n)
        if hw < margin:
            return n
    return n_max


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit_level(runs, mu: float, kn: float, n: int = N_PRIMARY,
                tol: float = TOL_BIAS, conf: float = CONF) -> dict:
    """One ``(mu, k_n)`` level: pairs, paired verdict, unpaired verdict, budget."""
    pairs = pair_by_seed(runs, mu, kn, n=n)
    paired = paired_diff(pairs, tol=tol, conf=conf)
    vr = variance_reduction(pairs, tol=tol, conf=conf)
    sel = [r for r in runs if r["n"] == n and r["mu"] == mu and r["kn"] == kn]
    coarse = np.array(sorted(r["phi"] for r in sel
                             if r["dt_frac"] == DT_COARSE), dtype=float)
    fine = np.array(sorted(r["phi"] for r in sel
                           if r["dt_frac"] == DT_FINE), dtype=float)
    unpaired = (welch_diff(coarse, fine)
                if coarse.size and fine.size else
                {"delta": None, "sem": None, "df": None, "error_model": "none"})
    if unpaired.get("delta") is not None and fine.size and coarse.size:
        unpaired.update(classify_bias(unpaired["delta"], unpaired["sem"],
                                      unpaired["df"], tol=tol, conf=conf))
    else:
        unpaired.update(verdict="NOT_RUN", resolved_from="none")
    sigma = paired.get("std_paired")
    budget = {
        "seeds_needed_paired": (None if sigma in (None, 0.0)
                                else seeds_needed_paired(sigma, tol, conf)),
        "seeds_needed_unpaired": (None if unpaired.get("sem") is None
                                  else seeds_needed(
                                      max(float(np.std(coarse, ddof=1))
                                          if coarse.size > 1 else 0.0,
                                          float(np.std(fine, ddof=1))
                                          if fine.size > 1 else 0.0),
                                      tol, conf)),
        "margin": float(tol),
    }
    return {
        "mu": float(mu), "kn": float(kn), "n": int(n),
        "n_seeds_coarse": int(coarse.size), "n_seeds_fine": int(fine.size),
        "phi_coarse_mean": float(coarse.mean()) if coarse.size else None,
        "phi_fine_mean": float(fine.mean()) if fine.size else None,
        "phi_coarse_std": (float(coarse.std(ddof=1))
                           if coarse.size > 1 else None),
        "phi_fine_std": (float(fine.std(ddof=1)) if fine.size > 1 else None),
        "pairs": pairs,
        "paired": paired,
        "unpaired": unpaired,
        "variance_reduction": vr,
        "seed_budget": budget,
        # A level is certified only when the verdict came from an *interval*
        # that resolves the band.  A point estimate (n_pairs < 2, or a
        # degenerate scatter) can never certify: calling that CONVERGED is the
        # over-claim m22824 §3 recorded as 不可认证.
        "certified": (paired.get("verdict") == "CONVERGED"
                      and paired.get("resolved_from") == "interval"),
    }


def audit(runs_dir: str | None = None, n: int = N_PRIMARY,
          tol: float = TOL_BIAS, conf: float = CONF) -> dict:
    """All levels that have both timestep arms, with paired and unpaired verdicts."""
    runs = load_runs(runs_dir or os.path.join(HERE, "runs"))
    levels = levels_present(runs, n=n)
    rows = [audit_level(runs, mu, kn, n=n, tol=tol, conf=conf)
            for mu, kn in levels]
    certified = [r for r in rows if r["certified"]]
    point_only = [r for r in rows
                  if r["paired"]["verdict"] == "CONVERGED" and not r["certified"]]
    return {
        "n_particles": int(n),
        "tol_bias": float(tol),
        "conf": float(conf),
        "estimator": "paired_crn",
        "n_runs_total": len(runs),
        "levels": rows,
        "summary": {
            "n_levels": len(rows),
            "n_certified_paired": len(certified),
            "n_converged_by_point_estimate_only": len(point_only),
            "verdicts_paired": {v: sum(1 for r in rows
                                       if r["paired"]["verdict"] == v)
                                for v in ("CONVERGED", "BIASED",
                                          "UNDECIDABLE", "NOT_RUN")},
            "verdicts_unpaired": {v: sum(1 for r in rows
                                         if r["unpaired"]["verdict"] == v)
                                  for v in ("CONVERGED", "BIASED",
                                            "UNDECIDABLE", "NOT_RUN")},
            "n_pairs_premise_failed": sum(
                len(r["paired"]["dropped_seeds_premise_failed"]) for r in rows),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--n", type=int, default=N_PRIMARY)
    ap.add_argument("--json", action="store_true", help="print the audit JSON")
    ap.add_argument("--write", default=None, help="write the audit JSON here")
    args = ap.parse_args(argv)
    res = audit(args.runs_dir, n=args.n)
    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)) or ".",
                    exist_ok=True)
        with open(args.write, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(f"wrote {args.write}")
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
    else:
        for r in res["levels"]:
            p, u = r["paired"], r["unpaired"]
            print(f"mu={r['mu']:<4g} kn={r['kn']:<8g} pairs={p['n_pairs']} "
                  f"Delta_paired={p['delta']!r:>12} "
                  f"CI=[{p['lo']}, {p['hi']}] -> {p['verdict']:<11} | "
                  f"unpaired Delta={u['delta']!r} -> {u['verdict']}")
        print("summary:", json.dumps(res["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())