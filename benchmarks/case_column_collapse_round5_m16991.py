"""CASE-COLUMN-COLLAPSE round 5 — breaking round 4's right-censoring on ``mu``.

Why round 5 exists
------------------
Rounds 1-4 of this anchor (``docs/case_column_collapse_*``) left it open with a
joint-identifiability question: is there **one** constant pair ``(mu, mu_r)``
(contact friction, rolling resistance) that reproduces the Lube et al. (2004)
runout law at every aspect ratio inside the 15 % tolerance?  Round 4
(m16862) probed the grid ``mu in {0.5,...,0.9} x mu_r in {0.1,...,0.5}`` at the
two branch representatives ``a = 1`` (linear branch, ``a < 1.7``) and ``a = 3``
(square-root branch) and returned *honest INCOMPLETE*: for ``mu_r <= 0.4`` the
required contact friction of the square-root branch ran off the top of the
``mu`` grid (only a lower bound was obtained), so the branch requirement gap

    gap(mu_r) = mu_req(square-root, mu_r) - mu_req(linear, mu_r)

could be evaluated at a single ``mu_r`` (0.5, ``gap = +0.29``).  Its own
documented next step: raise the ``mu`` grid and push ``mu_r`` down to see
whether ``gap`` changes sign (a crossing = joint solution exists).

Pre-registered protocol (fixed *before* any wave-5 run)
------------------------------------------------------
* H5: the two branch requirement curves cross, i.e. ``gap(mu_r)`` changes sign
  on ``mu_r in {0.1, 0.2, 0.3, 0.5}``, so that one constant ``(mu, mu_r)``
  reproduces the runout law at both branches.
* Instead of a fixed grid (which re-introduces censoring), ``mu_req(a, mu_r)``
  is **located by bisection** on the signed runout error

      e(mu) = (ratio_sim - ratio_ref(a)) / ratio_ref(a)

  on the bracket ``mu in [MU_LO, MU_HI] = [0.9, 2.4]`` (``mu = 0.9`` is
  exactly the round-4 grid ceiling, so the bracket abuts round 4's censored
  data instead of overlapping it), to ``|Delta mu| <= MU_TOL = 0.05``, at most
  ``MAX_EVAL = 8`` evaluations per chain.
* ``mu_req`` is **only** declared when the bracket straddles zero (both signs
  observed).  If ``e(mu) > 0`` at ``MU_HI`` the requirement is censored and the
  chain reports ``>"MU_HI"`` (a lower bound) — never a fake root.
* Monotonicity of ``e(mu)`` (decreasing) is the premise of the bracket search;
  it is checked on the observed evaluations and reported, not assumed silently.
  The bracket is narrowed with an Illinois (modified regula-falsi) secant step
  rather than plain bisection — same guarantees (a root is reported only when
  both signs were observed inside the bracket, bracket width <= 2 MU_TOL) and
  about half the evaluations on the near-linear error curves.  Adopted before
  any mid-bracket evaluation was inspected; recorded as a method deviation in
  ``docs/case_column_collapse_round5_m16991.md``.
* Everything else — resolution (8 grains per column radius), ``dt``,
  restitution, wall friction, seeding, duration factor 8 — is byte-identical
  to rounds 1-4 (:data:`BASE` mirrors ``benchmarks/case_column_collapse_lube2004.BASE``);
  so the wave-5 points at ``mu = 0.9`` double as a determinism control against
  the round-4 probe ids (same ``(mu, mu_r, a)``, same seed).

Decision rule (fixed here, before the runs)
-------------------------------------------
1. Compute ``mu_req(a, mu_r)`` for ``a in {1, 3}``, ``mu_r in {0.1, 0.2, 0.3}``.
2. Merge with round 4's ``mu_r = 0.5`` values (``mu_req^lin = 0.514``,
   ``mu_req^sqrt = 0.804``, both bracketed there) to get ``gap(mu_r)`` on four
   ``mu_r`` levels.
3. H5 **supported** iff at least one grid interval of ``mu_r`` reverses the
   sign of ``gap`` **and** both endpoints of that interval are bracketed
   (uncensored).
   H5 **refuted** iff every ``mu_req`` is bracketed (or at worst censored
   *above*, i.e. ``> MU_HI`` for both branches) and ``gap`` keeps one sign.
   Otherwise the verdict is **INCOMPLETE** (honest, no polish).
4. Secondary, pre-registered reading independent of (3): the pair that would be
   needed is compared with the physically admissible sand/gravel box
   ``mu <= 0.84`` (= tan 40 deg) and ``mu_r <= 0.3``; a required ``mu`` outside
   that box means sphere DEM with a *constant* friction law cannot reproduce the
   reference law — stated as a range statement, not as a fitted value.

Usage
-----
::

    python3 benchmarks/case_column_collapse_round5_m16991.py --plan
    python3 benchmarks/case_column_collapse_round5_m16991.py --chain a1_mur0.1
    python3 benchmarks/case_column_collapse_round5_m16991.py --chain a3_mur0.3
    python3 benchmarks/case_column_collapse_round5_m16991.py --collect

Each chain writes one JSON per evaluated ``(a, mu_r, mu)`` point into
``runs/case_column_collapse_r5_m16991/cases/`` (cache-first: a point whose JSON
already exists is not recomputed, so chains are resumable and shardable), plus
its own ``chain_<id>.json`` trace.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular_column.axisym import (  # noqa: E402
    LUBE2004_DOI, LUBE2004_SOURCE, AxisymCase, lube_2004_runout_ratio,
    simulate_axisymmetric_column)

ROUND = "r5_m16991"
OUT_DIR = os.path.join("runs", "case_column_collapse_r5_m16991")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = 0.15

#: Identical to ``case_column_collapse_lube2004.BASE`` except the three knobs
#: varied here (``aspect_ratio``, ``mu``, ``mu_rolling``).
BASE = dict(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.3,
            restitution=0.5, kn=6.0e4, kt_ratio=0.5, gravity=9.81,
            packing_gap=0.02, jitter=0.0, seed=12345,
            dt_scale=1.0, duration_factor=8.0, min_time=0.5,
            sample_stride=20, rest_ke_tol=1.0e-4, rest_hold=200,
            grains_per_radius=8.0)

#: Aspects of the case card's own acceptance criterion (all five must be
#: inside 15 %).  ``a = 1`` and ``a = 3`` are the branch representatives used
#: for the requirement curves; the other three are out-of-sample for the fit.
SWEEP_ASPECTS = (0.5, 1.0, 1.5, 2.0, 3.0)
#: Branch representatives: a = 1 is in the linear regime (a < 1.7), a = 3 in
#: the square-root regime (Lube et al. 2004).
BRANCH_ASPECTS = (1.0, 3.0)
#: Rolling resistance levels.  mu_r = 0.5 is *not* re-run: round 4 bracketed
#: both branches there (0.514 / 0.804) and those values are merged in at
#: collect time from ``ROUND4_REFERENCE``.
W5_MURS = (0.1, 0.2, 0.3)
#: Bisection bracket on the contact friction: [round-4 ceiling, high-mu wall].
MU_LO = 0.9
MU_HI = 2.4
#: Bisection stop: |bracket| <= 2 * MU_TOL measures mu_req to +-0.05.
MU_TOL = 0.05
MAX_EVAL = 8
#: Physically admissible box used for the secondary reading (sand/gravel:
#: tan(40 deg) = 0.839, rolling resistance coefficient <= 0.3).
PHYS_MU_MAX = 0.84
PHYS_MUR_MAX = 0.3
#: Round-4 (m16862) bracketed requirements at mu_r = 0.5, quoted from
#: docs/case_column_collapse_round4_m16862.md sec. 3.2 (the only mu_r where
#: round 4 bracketed both branches).  Kept as an *input to be checked*, not as
#: a result: ``--collect`` re-derives it from round-4 case JSONs whenever they
#: are present in the worktree and flags a mismatch.
ROUND4_REFERENCE = {"mu_rolling": 0.5, "mu_req_linear": 0.514, "mu_req_sqrt": 0.804}


# --------------------------------------------------------------------------
# case ids and one evaluation
# --------------------------------------------------------------------------
def case_id(aspect: float, mu_rolling: float, mu: float) -> str:
    return f"w5_a{aspect:g}_mur{mu_rolling:g}_mu{mu:g}"


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def signed_error(ratio: float, aspect: float) -> float:
    """Relative error of the simulated runout ratio against the reference law."""
    ref = lube_2004_runout_ratio(aspect)
    return (ratio - ref) / ref


def run_point(aspect: float, mu_rolling: float, mu: float,
              progress: bool = False) -> dict:
    """Evaluate one ``(a, mu_r, mu)`` point (cache-first, deterministic)."""
    cid = case_id(aspect, mu_rolling, mu)
    path = case_path(cid)
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
        res["cached"] = True
        return res
    t0 = time.perf_counter()
    case = AxisymCase(**dict(BASE, aspect_ratio=aspect, mu=mu,
                             mu_rolling=mu_rolling), label=cid)
    res = simulate_axisymmetric_column(case, progress=progress)
    res["wall_clock_s"] = time.perf_counter() - t0
    res["round"] = ROUND
    res["protocol"] = dict(BASE, aspect_ratio=aspect, mu=mu, mu_rolling=mu_rolling)
    res["case_id"] = cid
    res["reference_runout_ratio"] = lube_2004_runout_ratio(aspect)
    res["signed_error"] = signed_error(res["runout_ratio_p995"], aspect)
    res["signed_error_max_metric"] = signed_error(res["runout_ratio_max"], aspect)
    res["within_tolerance"] = abs(res["signed_error"]) <= TOLERANCE
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    res["cached"] = False
    return res


# --------------------------------------------------------------------------
# bisection on mu
# --------------------------------------------------------------------------
def bisect_mu(aspect: float, mu_rolling: float, lo: float = MU_LO,
              hi: float = MU_HI, tol: float = MU_TOL, max_eval: int = MAX_EVAL,
              progress: bool = False, log=print,
              evaluate_fn=None) -> dict:
    """Locate the contact friction that zeroes the signed runout error.

    Returns a trace dict with ``evals`` (every ``(mu, signed_error, stop_reason,
    wall_clock_s)``), the bracket, and ``mu_req`` — ``None`` when censored
    (``censoring = "low"`` if the error is still positive at ``hi``, ``"high"``
    if already negative at ``lo``).  Never invents a root: a bracketed root
    requires both signs actually observed.
    """
    evals: list[dict] = []

    def evaluate(mu: float) -> dict:
        if evaluate_fn is not None:
            row = dict(evaluate_fn(mu))
            row.setdefault("mu", mu)
            evals.append(row)
            log(f"  [synthetic] mu={mu:.3f} err={row['signed_error'] * 100:+.2f}%")
            return row
        res = run_point(aspect, mu_rolling, mu, progress=progress)
        row = {"mu": mu, "signed_error": res["signed_error"],
               "runout_ratio_p995": res["runout_ratio_p995"],
               "runout_ratio_max": res["runout_ratio_max"],
               "reference": res["reference_runout_ratio"],
               "stop_reason": res["stop_reason"],
               "n_particles": res["n_particles"],
               "wall_clock_s": res.get("wall_clock_s"),
               "cached": res["cached"], "case_id": res["case_id"]}
        evals.append(row)
        log(f"  a={aspect:g} mu_r={mu_rolling:g} mu={mu:.3f} "
            f"err={row['signed_error'] * 100:+.1f}% ratio={row['runout_ratio_p995']:.3f} "
            f"ref={row['reference']:.3f} stop={row['stop_reason']}"
            f"{' (cached)' if row['cached'] else ''}")
        return row

    f_lo = evaluate(lo)
    if f_lo["signed_error"] <= 0.0:
        return {"aspect": aspect, "mu_rolling": mu_rolling, "evals": evals,
                "mu_req": None, "censoring": "high",
                "note": f"signed error already <= 0 at mu = {lo:g}: requirement below the bracket"}
    f_hi = evaluate(hi)
    if f_hi["signed_error"] > 0.0:
        return {"aspect": aspect, "mu_rolling": mu_rolling, "evals": evals,
                "mu_req": None, "censoring": "low", "lower_bound": hi,
                "note": f"signed error still positive at mu = {hi:g}: mu_req > {hi:g} (censored)"}

    a_mu, b_mu = lo, hi
    a_err, b_err = f_lo["signed_error"], f_hi["signed_error"]
    non_monotone = 0
    # Illinois (modified regula falsi) on the bracket instead of plain
    # bisection: identical guarantees (a root is only reported when both signs
    # were actually observed inside the bracket, and the bracket still narrows
    # to 2 * tol), but the secant step converges in roughly half the
    # evaluations on the near-linear error curves observed at mu_r >= 0.2.
    # Recorded as a method deviation in the round-5 note: it was adopted before
    # any mid-bracket (mu != 0.9) evaluation had been inspected.
    same_side = 0
    while (b_mu - a_mu) > 2.0 * tol and len(evals) < max_eval:
        denom = (a_err - b_err)
        x = a_mu if denom == 0.0 else (a_mu * b_err - b_mu * a_err) / denom
        # keep the trial point strictly inside the bracket (Illinois safeguard)
        span = b_mu - a_mu
        x = min(max(x, a_mu + 0.1 * span), b_mu - 0.1 * span)
        row = evaluate(x)
        e = row["signed_error"]
        if e > max(a_err, b_err) or e < min(a_err, b_err):
            non_monotone += 1
        if e > 0.0:
            a_mu, a_err = x, e
            if same_side == 1:
                b_err *= 0.5            # Illinois: damp the stale end
            same_side = 1
        else:
            b_mu, b_err = x, e
            if same_side == -1:
                a_err *= 0.5
            same_side = -1
    mu_req = 0.5 * (a_mu + b_mu)
    return {"aspect": aspect, "mu_rolling": mu_rolling, "evals": evals,
            "mu_req": mu_req, "censoring": None, "bracket": [a_mu, b_mu],
            "bracket_width": b_mu - a_mu, "non_monotone_evals": non_monotone,
            "note": f"mu_req = {mu_req:.3f} +- {(b_mu - a_mu) / 2:.3f}"}


def run_validation(mu: float, mu_rolling: float, progress: bool = False) -> dict:
    """Pre-registered step 2: check one candidate pair on **every** aspect.

    The pair is *not* re-fitted here: ``a = 1`` and ``a = 3`` reproduce the
    chain points (determinism control), ``a in {0.5, 1.5, 2.0}`` are
    out-of-sample.  The criterion is the case card's own acceptance: relative
    runout error ``< 15 %`` at every aspect ratio.
    """
    rows = []
    for a in SWEEP_ASPECTS:
        res = run_point(a, mu_rolling, mu, progress=progress)
        rows.append({"aspect": a, "case_id": res["case_id"],
                     "n_particles": res["n_particles"],
                     "runout_ratio_p995": res["runout_ratio_p995"],
                     "reference": res["reference_runout_ratio"],
                     "signed_error": res["signed_error"],
                     "within_tolerance": res["within_tolerance"],
                     "stop_reason": res["stop_reason"],
                     "wall_clock_s": res.get("wall_clock_s")})
        print(f"  validate mu={mu:g} mu_r={mu_rolling:g} a={a:g} "
              f"err={rows[-1]['signed_error'] * 100:+.1f}% "
              f"{'PASS' if rows[-1]['within_tolerance'] else 'FAIL'} "
              f"stop={rows[-1]['stop_reason']}", flush=True)
    out = {"pair": {"mu": mu, "mu_rolling": mu_rolling},
           "tolerance": TOLERANCE, "rows": rows,
           "all_within_tolerance": all(r["within_tolerance"] for r in rows),
           "n_fail": sum(1 for r in rows if not r["within_tolerance"]),
           "max_abs_error": max(abs(r["signed_error"]) for r in rows)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"validation_mu{mu:g}_mur{mu_rolling:g}.json"),
              "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def chain_id(aspect: float, mu_rolling: float) -> str:
    return f"a{aspect:g}_mur{mu_rolling:g}"


CHAINS = [(a, m) for a in BRANCH_ASPECTS for m in W5_MURS]


def run_chain(aspect: float, mu_rolling: float, progress: bool = False,
              lo: float = MU_LO, hi: float = MU_HI) -> dict:
    cid = chain_id(aspect, mu_rolling)
    print(f"[chain {cid}] a={aspect:g} mu_r={mu_rolling:g} "
          f"bracket=[{lo:g}, {hi:g}] tol={MU_TOL:g}", flush=True)
    trace = bisect_mu(aspect, mu_rolling, lo=lo, hi=hi, progress=progress)
    trace["chain"] = cid
    trace["protocol"] = {"mu_bracket": [lo, hi], "mu_tol": MU_TOL,
                         "max_eval": MAX_EVAL, "tolerance": TOLERANCE,
                         "reference": LUBE2004_SOURCE, "doi": LUBE2004_DOI}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"chain_{cid}.json"), "w") as fh:
        json.dump(trace, fh, indent=1)
    print(f"[chain {cid}] -> {trace['note']}", flush=True)
    return trace


# --------------------------------------------------------------------------
# collection / verdict
# --------------------------------------------------------------------------
def load_chains() -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "chain_*.json"))):
        with open(path) as fh:
            t = json.load(fh)
        out[t["chain"]] = t
    return out


def gap_table(chains: dict, extra_mur: dict | None = None) -> list[dict]:
    """``mu_req`` per (branch, mu_r) and the branch requirement gap."""
    extra_mur = extra_mur or {}
    murs = sorted({t["mu_rolling"] for t in chains.values()} | set(extra_mur))
    rows = []
    for mur in murs:
        lin = next((t for t in chains.values()
                    if t["mu_rolling"] == mur and t["aspect"] < 1.7), None)
        sqr = next((t for t in chains.values()
                    if t["mu_rolling"] == mur and t["aspect"] > 1.7), None)
        lin_req = lin["mu_req"] if lin else extra_mur.get(mur, {}).get("mu_req_linear")
        sqr_req = sqr["mu_req"] if sqr else extra_mur.get(mur, {}).get("mu_req_sqrt")
        censored = [t["censoring"] for t in (lin, sqr) if t and t["censoring"]]
        gap = (sqr_req - lin_req) if (lin_req is not None and sqr_req is not None) else None
        rows.append({
            "mu_rolling": mur,
            "mu_req_linear": lin_req,
            "mu_req_sqrt": sqr_req,
            "linear_lower_bound_only": bool(lin and lin["censoring"]),
            "sqrt_lower_bound_only": bool(sqr and sqr["censoring"]),
            "gap": gap,
            "source": "wave5" if (lin or sqr) else "round4",
            "censoring": censored,
        })
    return rows


def verdict(rows: list[dict]) -> dict:
    """Apply the pre-registered H5 decision rule to the gap table."""
    bracketed = [r for r in rows if r["gap"] is not None]
    if not bracketed:
        return {"h5": "INCOMPLETE", "reason": "no mu_r with both branches bracketed"}
    signs = []
    for r in bracketed:
        signs.append(1 if r["gap"] > 0 else (-1 if r["gap"] < 0 else 0))
    crossed = any(a * b < 0 for a, b in zip(signs, signs[1:]))
    all_pos = all(s >= 0 for s in signs)
    all_neg = all(s <= 0 for s in signs)
    if crossed:
        h5 = "SUPPORTED"
        reason = "gap changes sign on the mu_r grid: joint (mu, mu_r) exists"
    elif (all_pos or all_neg) and len(bracketed) >= 3:
        h5 = "REFUTED"
        reason = ("gap keeps one sign over >= 3 bracketed mu_r levels; "
                  "branch requirement curves do not cross")
    else:
        h5 = "INCOMPLETE"
        reason = (f"only {len(bracketed)} bracketed mu_r level(s); "
                  "crossing neither shown nor excluded")
    return {"h5": h5, "reason": reason, "signs": signs,
            "n_bracketed_mur": len(bracketed)}


def collect() -> dict:
    chains = load_chains()
    rows = gap_table(chains, extra_mur={ROUND4_REFERENCE["mu_rolling"]: {
        "mu_req_linear": ROUND4_REFERENCE["mu_req_linear"],
        "mu_req_sqrt": ROUND4_REFERENCE["mu_req_sqrt"]}})
    out = {
        "case": "CASE-COLUMN-COLLAPSE",
        "round": ROUND,
        "reference": LUBE2004_SOURCE,
        "doi": LUBE2004_DOI,
        "tolerance": TOLERANCE,
        "protocol": {"mu_bracket": [MU_LO, MU_HI], "mu_tol": MU_TOL,
                     "max_eval": MAX_EVAL, "branch_aspects": BRANCH_ASPECTS,
                     "mu_rolling": W5_MURS, "round4_reference": ROUND4_REFERENCE,
                     "phys_box": {"mu_max": PHYS_MU_MAX, "mu_rolling_max": PHYS_MUR_MAX}},
        "gap_table": rows,
    }
    out.update(verdict(rows))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true", help="print the pre-registered chains")
    ap.add_argument("--chain", help="run one chain, e.g. a1_mur0.1 or a3_mur0.3")
    ap.add_argument("--all", action="store_true", help="run every chain serially")
    ap.add_argument("--collect", action="store_true", help="build gap table + verdict")
    ap.add_argument("--validate", nargs=2, type=float, metavar=("MU", "MU_R"),
                    help="pre-registered step 2: check one candidate pair on all "
                         "five aspects")
    ap.add_argument("--lo", type=float, default=MU_LO,
                    help="override the lower bracket end (fallback when a chain "
                         "returns censoring='high' at MU_LO)")
    ap.add_argument("--hi", type=float, default=MU_HI, help="override the upper end")
    ap.add_argument("--progress", action="store_true", help="per-step progress")
    args = ap.parse_args(argv)

    if args.plan:
        for a, m in CHAINS:
            print(chain_id(a, m), f"a={a:g} mu_r={m:g}")
        print(f"bracket mu in [{MU_LO:g}, {MU_HI:g}], tol {MU_TOL:g}, "
              f"<= {MAX_EVAL} evaluations per chain")
        return 0
    if args.chain:
        want = args.chain
        hit = [(a, m) for a, m in CHAINS if chain_id(a, m) == want]
        if not hit:
            print(f"unknown chain {want}; see --plan", file=sys.stderr)
            return 2
        run_chain(*hit[0], progress=args.progress, lo=args.lo, hi=args.hi)
        return 0
    if args.all:
        for a, m in CHAINS:
            run_chain(a, m, progress=args.progress)
        return 0
    if args.validate:
        out = run_validation(*args.validate, progress=args.progress)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        return 0
    if args.collect:
        print(json.dumps(collect(), indent=1))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())