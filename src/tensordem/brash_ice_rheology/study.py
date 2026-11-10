"""End-to-end study driver: measurements -> criteria -> artifacts -> report.

Two modes
---------
``mu-sweep``   run the DEM steady-shear sweep (~25 s per point on one core) and
               cache it as ``artifacts/brash_ice_rheology/mu_i_servo_sweep.json``;
``all``        load (or run) that sweep, evaluate criteria (a)-(g), write
               ``brash_ice_rheology_results.json``, the PNG figures and the
               markdown report ``docs/BRASH_ICE_CHANNEL_MU_I_RESISTANCE.md``.

Usage::

    python3 -m tensordem.brash_ice_rheology.study --mu-sweep
    python3 -m tensordem.brash_ice_rheology.study --all --pytest-count 22
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
from typing import Dict, List, Optional

__all__ = ["run_mu_sweep", "load_mu_sweep", "run_study", "ARTIFACT_DIR"]

ARTIFACT_DIR = "artifacts/brash_ice_rheology"
MU_SWEEP_FILE = "mu_i_servo_sweep.json"
#: Diagnostic variant: rolling resistance on (interlocking angular floes).
MU_SWEEP_ROLLING_FILE = "mu_i_servo_sweep_rolling.json"
RESULTS_FILE = "brash_ice_rheology_results.json"
REPORT_FILE = "docs/BRASH_ICE_CHANNEL_MU_I_RESISTANCE.md"


def run_mu_sweep(root: str = ".", *, targets=None,
                 contact_friction: float = 0.36,
                 rolling_friction: float = 0.0,
                 merge: bool = False,
                 out_name: str = MU_SWEEP_FILE) -> Dict[str, object]:
    """Run the DEM sweep and write the calibration cache (JSON).

    ``merge=True`` appends to an existing cache (the sweep can then be run in
    chunks that each fit a tool time limit), replacing points with the same
    inertial-number target.
    """
    from .mu_i import SWEEP_TARGETS, calibration_report, run_shear_sweep

    records = run_shear_sweep(targets or SWEEP_TARGETS,
                              contact_friction=contact_friction,
                              rolling_friction=rolling_friction)
    if merge:
        previous = load_mu_sweep(root, out_name)
        if previous:
            keep = [p for p in previous.get("points", [])
                    if all(abs(p["inertial_number_target"] - r["inertial_number_target"])
                           > 1e-12 for r in records)]
            records = sorted(keep + records,
                             key=lambda p: p["inertial_number_target"])
    report = calibration_report(records) if len(records) >= 3 else None
    payload = {
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "contact_friction": contact_friction,
        "rolling_friction": rolling_friction,
        "points": records,
        "n_points": len(records),
        "wall_clock_total_s": sum(r["wall_clock_s"] for r in records),
        "fit": report,
    }
    out = pathlib.Path(root) / ARTIFACT_DIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True))
    return payload


def load_mu_sweep(root: str = ".",
                  out_name: str = MU_SWEEP_FILE) -> Optional[Dict[str, object]]:
    """Load the cached sweep (``None`` if it has not been produced yet)."""
    path = pathlib.Path(root) / ARTIFACT_DIR / out_name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _clean(obj):
    """JSON-safe copy: non-finite floats become ``None``."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def run_study(root: str = ".", *, pytest_count: Optional[int] = None,
              mu_calibration: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Evaluate the criteria, write artifacts, figures and the markdown report."""
    from .anchors import anchor_components, scaling_diagnostic
    from .coefficients import ledger_sha256
    from .figures import fig_components, fig_mu_i, fig_porosity, fig_speed, png_base64
    from .material import STANDARD_BRASH
    from .model import STANDARD_BEAMS_M, brash_resistance, standard_hull
    from .report import build_report
    from .validation import evaluate_criteria

    state = STANDARD_BRASH
    from .mu_i import calibration_report, fit_sweep

    cache = mu_calibration if mu_calibration is not None else load_mu_sweep(root)
    mu_report = None
    if cache and len(cache.get("points", [])) >= 3:
        # always re-fit from the cached *measurements*, never from a cached fit,
        # so the fitter in the code is the single source of truth
        mu_report = calibration_report(cache["points"])
        mu_report["points"] = cache["points"]
    rolling = load_mu_sweep(root, MU_SWEEP_ROLLING_FILE)
    rolling_report = None
    if rolling and len(rolling.get("points", [])) >= 3:
        rolling_report = fit_sweep(rolling["points"])
        rolling_report["points"] = rolling["points"]

    results = evaluate_criteria(mu_report, state, pytest_count=pytest_count)
    results["calibration"] = _family_calibration_summary()
    results["standard_state"] = state.as_dict()
    results["suyuthi_note"] = ("Suyuthi et al. (2013) not readable offline; "
                               "reference = in-repository brash anchor")
    if rolling_report:
        results["mu_calibration_rolling"] = {
            "parameters": rolling_report["parameters"],
            "literature_relative_errors": rolling_report["literature_relative_errors"],
            "rms_relative": rolling_report["rms_relative"],
            "n_points": rolling_report["n_points"],
            "inertial_range": [min(p["inertial_number_measured"]
                                   for p in rolling_report["points"]),
                               max(p["inertial_number_measured"]
                                   for p in rolling_report["points"])],
        }
    if cache:
        results["mu_calibration"] = {
            "parameters": mu_report["parameters"],
            "literature_parameters": mu_report["literature"],
            "n_points": mu_report["n_points"],
            "inertial_range": [min(p["inertial_number_measured"]
                                   for p in mu_report["points"]),
                               max(p["inertial_number_measured"]
                                   for p in mu_report["points"])],
        }

    # decomposition table + per-beam anchor components (audit chain)
    decomposition = []
    for beam in STANDARD_BEAMS_M:
        hull = standard_hull(beam)
        res = brash_resistance(state, hull, 3.0)
        decomposition.append({
            "beam_m": beam, "model": res.as_dict(),
            "anchor": _clean(anchor_components(state, hull, 3.0)),
        })
    results["decomposition"] = decomposition
    results["scaling"] = scaling_diagnostic(state, STANDARD_BEAMS_M, 3.0)
    results["ledger_sha256"] = ledger_sha256()

    # figures
    figs = [
        ("brash_components", "Three-component resistance vs beam with the "
                             "reference anchor total (criterion b, c).",
         fig_components()),
        ("brash_speed", "R(v) and its log-log power-law fit over 1-5 m/s "
                        "(criterion d).", fig_speed()),
        ("brash_porosity", "Component response to layer porosity (criterion e).",
         fig_porosity()),
        ("brash_mu_i", "mu(I) closure: DEM calibration points, Jop law and the "
                       "brash-channel operating point (criterion a).",
         fig_mu_i(mu_report)),
    ]
    fig_dir = pathlib.Path(root) / ARTIFACT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures_meta: List[Dict[str, str]] = []
    for name, caption, fig in figs:
        b64 = png_base64(fig)
        (fig_dir / f"{name}.png").write_bytes(__import__("base64").b64decode(b64))
        figures_meta.append({"name": name, "caption": caption, "png_base64": b64})
    results["figures"] = [{"name": f["name"], "caption": f["caption"]}
                          for f in figures_meta]

    md = build_report(results)
    (pathlib.Path(root) / REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
    (pathlib.Path(root) / REPORT_FILE).write_text(md)
    (fig_dir / RESULTS_FILE).write_text(json.dumps(_clean(results), indent=2,
                                                   sort_keys=True))
    results["_figures_full"] = figures_meta
    results["_report_markdown"] = md
    return results


def _family_calibration_summary() -> Dict[str, object]:
    from .calibration import calibrate_family
    return calibrate_family()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--mu-sweep", action="store_true",
                        help="run the DEM steady-shear sweep and cache it")
    parser.add_argument("--all", action="store_true",
                        help="evaluate criteria, write artifacts + report")
    parser.add_argument("--pytest-count", type=int, default=None)
    parser.add_argument("--contact-friction", type=float, default=0.36)
    parser.add_argument("--rolling-friction", type=float, default=0.0)
    parser.add_argument("--targets", default=None,
                        help="comma separated inertial-number targets")
    parser.add_argument("--merge", action="store_true",
                        help="merge into an existing sweep cache")
    parser.add_argument("--rolling", action="store_true",
                        help="run the interlocking (rolling resistance) variant "
                             "into %s" % MU_SWEEP_ROLLING_FILE)
    args = parser.parse_args(argv)

    if args.mu_sweep:
        targets = None
        if args.targets:
            targets = tuple(float(x) for x in args.targets.split(",") if x.strip())
        payload = run_mu_sweep(
            args.root, targets=targets,
            contact_friction=args.contact_friction,
            rolling_friction=(0.1 if args.rolling else args.rolling_friction),
            merge=args.merge,
            out_name=(MU_SWEEP_ROLLING_FILE if args.rolling else MU_SWEEP_FILE))
        summary = {"n_points": payload["n_points"],
                   "wall_clock_total_s": payload["wall_clock_total_s"]}
        if payload["fit"]:
            summary["literature"] = payload["fit"]["a2_literature"]
            summary["parameters"] = payload["fit"]["parameters"]
        print(json.dumps(summary, indent=2))
    if args.all:
        results = run_study(args.root, pytest_count=args.pytest_count)
        print(json.dumps({"passed": results["passed"], "failed": results["failed"],
                          "not_evaluated": results["not_evaluated"],
                          "ledger_sha256": results["ledger_sha256"]}, indent=2))
    if not args.mu_sweep and not args.all:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
