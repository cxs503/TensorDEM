"""Implementation-provenance audit of the RCP deposition campaign (m22824).

Why this script exists
----------------------
Every deposition run records ``impl_sha256`` -- the SHA-256 of
``src/tensordem/deposition_packing.py`` at the moment the run started.  An
inventory of the campaign shows **two distinct hashes** inside the same
``(mu, k_n)`` family::

    mu = 0.0, k_n = 64000:  seed1 (dt)        -> cf4b761c...
                            seed2,3 (dt)      -> b44c1914...
                            seed1,2,3 (dt/2)  -> b44c1914...
    mu = 0.1, k_n = 64000:  seed1 (dt)        -> cf4b761c...
                            seed1 (dt/2)      -> b44c1914...

The temporal-bias gate of m21401/m22558 forms ``Delta = phi(dt/2) - phi(dt)``
from those files, i.e. it differences runs created by **two revisions of the
solver**.  Nobody had checked whether the revision change is numerically inert,
so the ladder's central premise ("the only difference between the two members
of a pair is the timestep") was an *assumption*, not a verified fact.  This
module turns it into a verified fact -- or exposes it as a defect.

Method
------
1. **Inventory**: group every run on disk by ``impl_sha256`` (full hex, not a
   prefix) and tabulate which ``(mu, k_n, dt_frac)`` cells each digest covers.
2. **Behavioural equivalence certificate**: a digest pair is declared
   *numerically equivalent* only if a **reproduction run** -- same spec, same
   seed, executed under the *current* implementation -- reproduces the archived
   artefacts of the older digest.  The comparison is made on the settled
   positions (``.npz``, maximum absolute coordinate difference) and on every
   scalar metric, not on the source diff: "the diff looks harmless" is exactly
   the kind of claim an audit must not accept on inspection alone.
3. **Confound census**: for every ``(mu, k_n)`` level that has both timestep
   levels, report whether the pair spans one or two digest classes.

Verdict vocabulary (pre-registered before the reproduction runs were read):

===================  ===================================================
``EQUIVALENT``       positions identical (max |dx| = 0) and all scalar
                     metrics identical -> the digest change is inert, the
                     ladder is unconfounded and may keep using the archive
``NUMERICALLY_EQUAL``positions differ by <= 1e-12 (floating-point noise)
                     but metrics agree to 1e-12 -> inert in practice
``DIVERGENT``        anything larger -> the archived pair must be re-run
                     under one implementation before the gate is quoted
``MISSING``          no reproduction run on disk -> unknown, never "passed"
===================  ===================================================

Usage::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/provenance_audit.py --json
    PYTHONPATH=src python benchmarks/case_rcp_deposition/provenance_audit.py --json --dump
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
REPRO = os.path.join(HERE, "runs_repro_m22824")
OUT_JSON = os.path.join(RUNS, "provenance_audit.json")

#: scalar metrics compared by the reproduction certificate, in report order
CERT_FIELDS = ("phi_bulk", "phi_bulk_overlap_corrected", "coordination_bulk",
               "phi_bulk_solid", "steps", "final_time")
#: tolerances of the pre-registered verdicts
TOL_EQUIVALENT = 0.0
TOL_NUMERICALLY_EQUAL = 1e-12

__all__ = ["load_provenance", "hash_classes", "ladder_confounds",
           "compare_repro", "audit", "verdict_for", "CERT_FIELDS",
           "current_digest"]


def current_digest() -> str:
    """SHA-256 of the solver module *right now* -- the certificate's reference.

    Computed with the same routine the runs use
    (:func:`tensordem.deposition_packing.inspect_source_sha256`), so a checkout
    that has drifted from the released source shows up as a digest mismatch
    instead of silently certifying itself.
    """
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tensordem.deposition_packing import inspect_source_sha256
    return inspect_source_sha256()


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
def load_provenance(runs_dir: str = RUNS) -> list:
    """Every run JSON with its spec, metrics and ``impl_sha256``."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "deposition_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        spec, m = d["spec"], d["metrics"]
        row = {
            "path": os.path.relpath(path, ROOT),
            "tag": os.path.basename(path)[:-5],
            "n": int(spec["n_particles"]),
            "mu": float(spec["mu"]),
            "kn": float(spec["kn"]),
            "seed": int(spec["seed"]),
            "dt_frac": float(spec.get("dt_frac", 1.0 / 20.0)),
            "impl_sha256": d.get("impl_sha256", ""),
            "settled": bool(d.get("settled", False)),
        }
        for k in CERT_FIELDS:
            if k in m:
                row[k] = float(m[k])
        out.append(row)
    return out


def hash_classes(runs) -> dict:
    """``{digest: {...}}`` -- which cells each implementation digest covers."""
    classes: dict[str, dict] = {}
    for r in runs:
        d = r["impl_sha256"] or "unknown"
        c = classes.setdefault(d, {"digest": d, "n_runs": 0, "tags": [],
                                   "mus": [], "kns": [], "dt_fracs": []})
        c["n_runs"] += 1
        c["tags"].append(r["tag"])
        for k, f in (("mus", "mu"), ("kns", "kn"), ("dt_fracs", "dt_frac")):
            if r[f] not in c[k]:
                c[k].append(r[f])
    for c in classes.values():
        c["mus"].sort()
        c["kns"].sort()
        c["dt_fracs"].sort()
        c["tags"].sort()
    return classes


def ladder_confounds(runs) -> list:
    """For every level with both timestep levels: does the pair mix digests?"""
    rows = []
    keys = sorted({(r["mu"], r["kn"]) for r in runs if r["n"] == 10000})
    for mu, kn in keys:
        coarse = [r for r in runs
                  if r["n"] == 10000 and r["mu"] == mu and r["kn"] == kn
                  and r["dt_frac"] == 1.0 / 20.0]
        fine = [r for r in runs
                if r["n"] == 10000 and r["mu"] == mu and r["kn"] == kn
                and r["dt_frac"] == 1.0 / 40.0]
        if not coarse or not fine:
            continue
        ch = {r["impl_sha256"] for r in coarse}
        fh = {r["impl_sha256"] for r in fine}
        mixed = len(ch | fh) > 1
        shared = bool(ch & fh)
        rows.append({
            "mu": mu, "kn": kn,
            "n_coarse": len(coarse), "n_fine": len(fine),
            "coarse_digests": sorted(ch), "fine_digests": sorted(fh),
            "n_digests": len(ch | fh),
            "mixed_digest": mixed,
            "shared_digest_on_both_levels": shared,
            # a pair is only usable for a timestep test when *every* member of
            # the pair came out of the same implementation digest
            "clean_for_timestep_test": (not mixed),
            "coarse_seeds": sorted(r["seed"] for r in coarse),
            "fine_seeds": sorted(r["seed"] for r in fine),
        })
    return rows


# ---------------------------------------------------------------------------
# reproduction certificate
# ---------------------------------------------------------------------------
def _positions(npz_path: str):
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as z:
        return np.array(z["positions"], dtype=float)


def compare_repro(stored: dict, repro: dict, stored_npz: str = None,
                  repro_npz: str = None) -> dict:
    """Certificate of one (stored run, reproduction run) pair."""
    fields, worst = {}, 0.0
    skip = {"path", "tag", "impl_sha256", "settled"}
    for k in CERT_FIELDS:
        if k in stored and k in repro:
            a, b = float(stored[k]), float(repro[k])
            fields[k] = {"stored": a, "repro": b, "abs_diff": abs(a - b),
                         "rel_diff": (abs(a - b) / abs(a)) if a else None}
            worst = max(worst, abs(a - b))
    dpos = None
    if stored_npz and repro_npz:
        pa, pb = _positions(stored_npz), _positions(repro_npz)
        if pa is not None and pb is not None and pa.shape == pb.shape:
            dpos = float(np.max(np.abs(pa - pb)))
    return {
        "stored_tag": stored["tag"], "repro_tag": repro["tag"],
        "stored_digest": stored["impl_sha256"], "repro_digest": repro["impl_sha256"],
        "same_spec": all(stored[k] == repro[k]
                         for k in ("n", "mu", "kn", "seed", "dt_frac")),
        "fields": fields,
        "max_scalar_abs_diff": worst,
        "positions_max_abs_diff": dpos,
        "verdict": verdict_for(worst, dpos),
    }


def verdict_for(max_scalar: float, dpos: float | None) -> str:
    """Pre-registered three-way (four-way with MISSING) certificate verdict."""
    if dpos is None:
        return "MISSING"
    if max_scalar == TOL_EQUIVALENT and dpos == TOL_EQUIVALENT:
        return "EQUIVALENT"
    if max_scalar <= TOL_NUMERICALLY_EQUAL and dpos <= TOL_NUMERICALLY_EQUAL:
        return "NUMERICALLY_EQUAL"
    return "DIVERGENT"


def _match_repro(stored: dict, repro_runs) -> dict | None:
    for r in repro_runs:
        if all(stored[k] == r[k] for k in ("n", "mu", "kn", "seed", "dt_frac")):
            return r
    return None


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit(runs_dir: str = RUNS, repro_dir: str = REPRO) -> dict:
    runs = load_provenance(runs_dir)
    repro = load_provenance(repro_dir) if os.path.isdir(repro_dir) else []
    classes = hash_classes(runs)
    ladders = ladder_confounds(runs)

    current = current_digest()
    certs = []
    for old in runs:
        new = _match_repro(old, repro)
        if new is None:
            # a run produced by the *current* digest needs no certificate
            if old["impl_sha256"] == current:
                continue
            certs.append({"stored_tag": old["tag"], "repro_tag": None,
                          "stored_digest": old["impl_sha256"],
                          "repro_digest": None, "fields": {},
                          "max_scalar_abs_diff": None,
                          "positions_max_abs_diff": None,
                          "same_spec": False, "verdict": "MISSING"})
            continue
        certs.append(compare_repro(
            old, new,
            os.path.join(runs_dir, old["tag"] + ".npz"),
            os.path.join(repro_dir, new["tag"] + ".npz")))

    verdicts = {}
    for c in certs:
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
    n_mixed = sum(1 for l in ladders if l["mixed_digest"])
    clean = [l for l in ladders if l["clean_for_timestep_test"]]
    return {
        "runs_dir": os.path.relpath(runs_dir, ROOT),
        "repro_dir": os.path.relpath(repro_dir, ROOT) if os.path.isdir(repro_dir) else None,
        "n_runs": len(runs),
        "n_repro_runs": len(repro),
        "n_digests": len(classes),
        "current_digest": current,
        "n_runs_off_current_digest": sum(1 for r in runs
                                         if r["impl_sha256"] != current),
        "classes": classes,
        "ladders": ladders,
        "n_ladder_levels": len(ladders),
        "n_ladder_levels_mixed_digest": n_mixed,
        "n_ladder_levels_clean": len(clean),
        "certificates": certs,
        "certificate_verdicts": verdicts,
        "all_certified_inert": (bool(certs)
                                and all(c["verdict"] in ("EQUIVALENT",
                                                         "NUMERICALLY_EQUAL")
                                        for c in certs)),
        "tolerances": {"equivalent": TOL_EQUIVALENT,
                       "numerically_equal": TOL_NUMERICALLY_EQUAL},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="numbers only")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--repro", default=REPRO)
    ap.add_argument("--dump", action="store_true", help="write the audit JSON")
    args = ap.parse_args(argv)

    res = audit(args.runs, args.repro)
    if args.dump:
        with open(OUT_JSON, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print("wrote", os.path.relpath(OUT_JSON, ROOT))

    print(f"runs={res['n_runs']} repro_runs={res['n_repro_runs']} "
          f"digests={res['n_digests']} current={res['current_digest'][:12]} "
          f"off_current={res['n_runs_off_current_digest']}")
    for d, c in sorted(res["classes"].items()):
        print(f"  digest {d[:12]}  runs={c['n_runs']:2d}  "
              f"mu={c['mus']} kn={c['kns']}")
    print(f"ladder levels={res['n_ladder_levels']} "
          f"mixed_digest={res['n_ladder_levels_mixed_digest']} "
          f"clean={res['n_ladder_levels_clean']}")
    for l in res["ladders"]:
        print(f"  mu={l['mu']:g} kn={l['kn']:g} "
              f"seeds=({l['n_coarse']},{l['n_fine']}) "
              f"digests={l['n_digests']} "
              f"{'MIXED' if l['mixed_digest'] else 'clean'}")
    if res["certificates"]:
        print("certificates:")
        for c in res["certificates"]:
            d = c["positions_max_abs_diff"]
            print(f"  {c['stored_tag']:<58s} -> {c['verdict']:<18s} "
                  f"max|dx|={'-' if d is None else f'{d:.3e}'}")
    else:
        print("certificates: none on disk (reproduction runs not yet returned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())