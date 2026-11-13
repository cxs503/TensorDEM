"""Render the round-3 (repose-angle metrology) evidence into the committed tree.

Why this exists
---------------
``docs/H1_BROKEN_ICE_DEM_ROUND3.md`` §1 quotes a table of *recovered* cone
angles for 13 known slopes, produced (not hand-copied) by
``benchmarks/h1_broken_ice_dem/make_figures_round3.py``.  That script writes to
``artifacts/h1_broken_ice_dem/``, which is **not** part of the committed tree
(the merge gate drops ``artifacts/``), so the table the document rests on was
unverifiable from a checkout: nothing in the tree let a reader reproduce
"worst case 7.24 deg naive / 2.30 deg flank".  This renderer runs the same
benchmark generator and commits its JSON + figure under
``runs/h1_broken_ice_dem/round3/`` together with a provenance block (sha256 of
the generator and of the estimator module, git commit, timestamp), so the
numbers in the document are machine-checkable from the tree alone
(:mod:`tests.test_h1_round3_estimator_evidence`).

    PYTHONPATH=src python scripts/render_h1_round3_evidence.py

Zero hand-copying: every number in the committed JSON is written by the
generator below, and the test recomputes the same points from
``tensordem.polyhedron_floe3d.pile_surface`` to detect code drift.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BENCH = os.path.join(ROOT, "benchmarks", "h1_broken_ice_dem", "make_figures_round3.py")
DEST = os.path.join(ROOT, "runs", "h1_broken_ice_dem", "round3")
ESTIMATOR = os.path.join(ROOT, "src", "tensordem", "polyhedron_floe3d", "pile_surface.py")

sys.path.insert(0, os.path.join(ROOT, "src"))


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - git is optional for reproduction
        return ""


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("h1_round3_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    bench = _load_benchmark()
    bench.OUT = DEST  # the generator writes round3_figures.json + fig4/fig5 here
    bench.main()

    path = os.path.join(DEST, "round3_figures.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    # Ablation on the *same* profiles: the same BIC break search without the
    # "apron cannot be steeper than the flank" admissibility rule.  This is the
    # pre-fix behaviour and the baseline the fixed estimator is compared to.
    from tensordem.polyhedron_floe3d.pile_surface import fit_flank

    truth = np.arange(15.0, 45.1, 2.5)
    ablation = {"flank_clean": [], "flank_apron": []}
    for angle in truth:
        for share, key in ((0.0, "flank_clean"), (0.4, "flank_apron")):
            prof = bench.synthetic(float(angle), 400, share)
            ablation[key].append(fit_flank(prof["radius"], prof["height"],
                                           prof["count"], min_drop=0.05,
                                           require_flatter_apron=False)["angle_deg"])
    ablation["truth_deg"] = list(truth)
    ablation["note"] = ("require_flatter_apron=False: BIC split admits a break whose "
                        "outward segment descends faster than the flank (pre-fix)")
    ref = np.asarray(truth, dtype=np.float64)
    for key in ("flank_clean", "flank_apron"):
        err = np.abs(np.asarray(ablation[key], dtype=np.float64) - ref)
        ablation["worst_error_%s_deg" % key.split("_", 1)[1]] = (
            None if np.all(np.isnan(err)) else float(np.nanmax(err)))
    payload["fig1_pre_fix_ablation"] = ablation
    payload["provenance"] = {
        "generator": os.path.relpath(BENCH, ROOT),
        "generator_sha256": _sha256(BENCH),
        "estimator": os.path.relpath(ESTIMATOR, ROOT),
        "estimator_sha256": _sha256(ESTIMATOR),
        "renderer": os.path.relpath(os.path.abspath(__file__), ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("regenerate with `PYTHONPATH=src python "
                 "scripts/render_h1_round3_evidence.py`; numbers are written by "
                 "the benchmark generator, never typed by hand"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)

    fig = payload.get("fig1", {})
    abl = payload.get("fig1_pre_fix_ablation", {})
    print(json.dumps({
        "committed": os.path.relpath(path, ROOT),
        "worst_error_naive_apron_deg": fig.get("worst_error_naive_apron_deg"),
        "worst_error_flank_apron_deg": fig.get("worst_error_flank_apron_deg"),
        "pre_fix_worst_error_flank_clean_deg": abl.get("worst_error_clean_deg"),
        "fig2_present": payload.get("fig2") is not None,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())