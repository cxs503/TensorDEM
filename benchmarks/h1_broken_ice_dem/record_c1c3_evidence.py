"""Record the committed C1/C3 evidence for anchor H1-BROKEN-ICE-DEM.

Why this script exists
----------------------
The audit of round 2 (m15029, item 3) required the *run artifacts* to live in the
tree, and ``tests/test_h1_broken_ice_dem.py`` reads them from

    runs/h1_broken_ice_dem/results_c1c3.json

Because that file was filtered out of the earlier delivery, those three tests
silently **skipped** -- i.e. the C1 (Archimedes floating equilibrium) and C3
(channel clearing) claims were not actually re-checked by CI, only asserted in a
document.  This script regenerates the artifact from the committed code so the
tests turn from ``skip`` into real assertions.

What is kept, and why it is trimmed
-----------------------------------
``run_h1_broken_ice_dem`` serialises the full per-step time series (~13k samples
for C3).  For the evidence file we keep every ``--stride``-th sample of the C3
series: enough to re-plot the lane-concentration drop, small enough that the
merge gate does not filter the file out of the tree (which is how the previous
artifact was lost).  Nothing is retyped: every number in the file comes from the
solver run below.

Usage
-----
    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py
    # slower/higher-fidelity C3 (cluster):
    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py \
        --channel-n-floes 24 --channel-steps 6000
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensordem.integration_callers import check_h1_maturity, run_h1_broken_ice_dem  # noqa: E402

OUT = pathlib.Path("runs/h1_broken_ice_dem/results_c1c3.json")

SERIES_KEYS = ("time_s", "lane_concentration", "lane_concentration_centroid_proxy",
               "concentration", "resistance_N")


def _trim_channel(case: Dict[str, Any], stride: int) -> Dict[str, Any]:
    """Downsample the C3 time series, keep every scalar verdict untouched."""
    trimmed = {key: value for key, value in case.items() if key != "series"}
    series = case.get("series", {})
    trimmed["series"] = {key: list(series.get(key, []))[::stride] for key in SERIES_KEYS
                         if key in series}
    trimmed["series"]["stride"] = stride
    trimmed["series"]["n_samples_full"] = len(series.get("time_s", []))
    return trimmed


def _markdown_table(evidence: Dict[str, Any]) -> str:
    """Render the C1/C3 numbers as a markdown table (so docs never hand-copy)."""
    lines = ["| case | quantity | measured (`got`) | reference (`ref`) | error | verdict |",
             "|---|---|---|---|---|---|"]
    b = evidence["cases"].get("buoyancy", {})
    if b:
        ref, got = b.get("ref", {}), b.get("got", {})
        error = b.get("error", {})
        for got_key, ref_key, err_key in (
                ("submerged_fraction", "submerged_fraction_archimedes",
                 "rel_err_submerged_fraction"),
                ("draft_m", "draft_archimedes_m", "rel_err_draft")):
            rel = error.get(err_key, float("nan"))
            ok = math.isfinite(rel) and rel <= b.get("tolerance", 0.05)
            lines.append(
                f"| C1 buoyancy | {got_key} | {got.get(got_key):.6f} | "
                f"{ref.get(ref_key):.6f} | {100.0 * rel:.3f} % | "
                f"{'PASS' if ok else 'FAIL'} |")
    c = evidence["cases"].get("channel", {})
    if c:
        ref, got, summary = c.get("ref", {}), c.get("got", {}), c.get("summary", {})
        crit = c.get("criteria", {})
        for key, ref_text, crit_key in (
                ("lane_concentration_ratio", ref.get("lane_drop_threshold_fraction"),
                 "lane_halved"),
                ("hull_advance_m", ref.get("sweep_requirement"), "sweep_complete"),
                ("mass_residual", ref.get("mass_residual_tolerance"), "mass_conserved"),
                ("energy_residual", ref.get("energy_residual_tolerance"), "energy_closed")):
            value = got.get(key, float("nan"))
            ref_text = ref_text if isinstance(ref_text, (int, float)) else str(ref_text)
            lines.append(f"| C3 channel | {key} | {value:.6g} | {ref_text} | - | "
                         f"{'PASS' if crit.get(crit_key) else 'FAIL'} |")
        lines.append("")
        lines.append("criteria: " + ", ".join(f"{k}={v}" for k, v in crit.items()))
        lines.append("")
        lines.append(
            "lane concentration: "
            f"{summary.get('lane_concentration_initial', float('nan')):.4f} -> "
            f"{summary.get('lane_concentration_final', float('nan')):.4f} "
            f"(ratio {got.get('lane_concentration_ratio', float('nan')):.4f}); "
            f"hull advance {got.get('hull_advance_m', float('nan')):.3f} m; "
            f"resistance {got.get('resistance_mean_steady_N', float('nan')):.1f} N")
    m = evidence.get("maturity") or {}
    lines.append("")
    lines.append(f"L4 gate: passed={m.get('passed')} level={m.get('level_achieved')} "
                 f"gaps={m.get('gaps')}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stride", type=int, default=20,
                    help="keep every stride-th C3 sample (default 20)")
    ap.add_argument("--channel-n-floes", type=int, default=24)
    ap.add_argument("--channel-steps", type=int, default=6000)
    ap.add_argument("--work-dir", default="runs/h1_broken_ice_dem_raw",
                    help="where the untrimmed caller report is written")
    args = ap.parse_args(argv)

    t0 = time.time()
    result = run_h1_broken_ice_dem(
        cases=("buoyancy", "channel"), output_dir=args.work_dir,
        channel={"n_floes": args.channel_n_floes, "n_steps": args.channel_steps},
    )
    evidence: Dict[str, Any] = {
        "anchor": "H1-BROKEN-ICE-DEM",
        "produced_by": "benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py",
        "repro_cmd": ("PYTHONPATH=src python "
                      "benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py "
                      f"--channel-n-floes {args.channel_n_floes} "
                      f"--channel-steps {args.channel_steps} --stride {args.stride}"),
        "runtime_s": None,  # filled below: the run time of *this* recording
        "cases": {},
        "passed": {},
        "maturity": None,
    }
    for name, case in result["cases"].items():
        evidence["cases"][name] = (_trim_channel(case, args.stride)
                                   if name == "channel" else case)
        evidence["passed"][name] = bool(case["passed"])
    evidence["integration"] = result.get("integration", {})
    evidence["runtime_s"] = time.time() - t0
    # the gate is *recorded*, not enforced: a failing verdict stays visible in the
    # file (honest FAIL is data, not an error)
    evidence["maturity"] = check_h1_maturity(
        {"cases": evidence["cases"], "integration": evidence["integration"],
         "repro_cmd": evidence["repro_cmd"]}, required_level="L4")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=1, default=float)
    table_path = out.with_name(out.stem + "_table.md")
    table_path.write_text(_markdown_table(evidence), encoding="utf-8")

    channel = evidence["cases"].get("channel", {})
    print(json.dumps({
        "out": str(out),
        "table": str(table_path),
        "size_kb": round(out.stat().st_size / 1024.0, 1),
        "passed": evidence["passed"],
        "channel_criteria": channel.get("criteria"),
        "channel_summary": {k: channel.get("summary", {}).get(k) for k in
                            ("lane_concentration_initial", "lane_concentration_final",
                             "lane_concentration_ratio", "hull_advance_m",
                             "mass_residual", "energy_residual")},
        "maturity_passed": evidence["maturity"]["passed"],
        "maturity_gaps": evidence["maturity"]["gaps"],
        "runtime_s": round(evidence["runtime_s"], 1),
    }, indent=1, default=float))
    return 0 if all(evidence["passed"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())