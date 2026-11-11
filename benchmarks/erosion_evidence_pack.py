"""Build the evidence pack for node p176w2-n11 from the two run artefacts.

The pack turns the raw ``results.json`` files into (a) a machine-readable
reference-vs-got benchmark table, and (b) a run manifest that binds every claim
to an artifact path, a sha256 and the exact command that reproduces it.  No
number in the pack is typed by hand: everything is read from the JSON that the
two benchmarks wrote.

Run:
    python benchmarks/erosion_evidence_pack.py
Writes into ``artifacts/p176w2-n11/``:
    benchmark_ref_vs_got.csv / .md
    RUN_MANIFEST.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "p176w2-n11"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


def rows_main(main: dict) -> list[dict]:
    c1, c2, c3, c4, c5, c6, c7 = (main[k] for k in
                                  ("C1", "C2", "C3", "C4", "C5", "C6", "C7"))
    rows = [
        {
            "face": "standalone",
            "id": "C1a",
            "quantity": "Finnie angle-function peak [deg]",
            "reference": "ductile literature window 15-25 deg",
            "reference_value": "15-25",
            "got_value": f"{c1['finnie_peak_deg_analytic']:.4f}",
            "metric": "in-window",
            "tolerance": "15-25",
            "verdict": "PASS" if c1["finnie_peak_in_window"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C1",
        },
        {
            "face": "standalone",
            "id": "C1b",
            "quantity": "Finnie vs Oka normalised shape, peak band",
            "reference": "Oka et al. (2005) plastic angle curve",
            "reference_value": "same shape (0 %)",
            "got_value": f"{c1['shape_deviation_peak_band_40pct']:.4%}",
            "metric": "max relative difference",
            "tolerance": "<15%",
            "verdict": "PASS" if c1["shape_agreement_under_15pct"] else "HONEST_FAIL",
            "source": "artifacts/p176w2-n11/results.json:C1",
        },
        {
            "face": "standalone",
            "id": "C2",
            "quantity": "velocity exponent n",
            "reference": f"literature n = {c2.get('literature_exponent', 'n/a')}",
            "reference_value": f"{c2.get('literature_exponent', float('nan'))}",
            "got_value": f"{c2['fit']['exponent']:.4f}",
            "metric": "relative error",
            "tolerance": "n in [2.1, 2.5], err <10%",
            "verdict": "PASS" if c2["pass"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C2",
        },
        {
            "face": "standalone",
            "id": "C4",
            "quantity": "mass-loss ledger closure residual",
            "reference": "0 (exact bookkeeping)",
            "reference_value": "0",
            "got_value": f"{c4['closure_residual']:.3e}",
            "metric": "relative residual",
            "tolerance": "<5%",
            "verdict": "PASS" if c4["pass"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C4",
        },
        {
            "face": "standalone",
            "id": "C5",
            "quantity": "double monotonicity (speed up / post-peak decay)",
            "reference": "monotone",
            "reference_value": "0 violations",
            "got_value": f"{c5['velocity_monotone_increasing']['samples']} + "
                         f"{c5['angle_monotone_decaying_to_90']['samples']} points, "
                         f"{c5['velocity_monotone_increasing']['violations'] + c5['angle_monotone_decaying_to_90']['violations']} violations",
            "metric": "violations",
            "tolerance": "0 violations, >=5 points each",
            "verdict": "PASS" if c5["pass"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C5",
        },
        {
            "face": "standalone",
            "id": "C6",
            "quantity": "negative control: zero impacts / zero speed",
            "reference": "E = 0 exactly",
            "reference_value": "0",
            "got_value": f"{c6['zero_speed']['erosion_ratio_kg_kg']:.1e}",
            "metric": "absolute",
            "tolerance": "<1e-12",
            "verdict": "PASS" if c6["pass"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C6",
        },
        {
            "face": "standalone",
            "id": "C7",
            "quantity": "breakage switch changes ledger?",
            "reference": "identity (no change)",
            "reference_value": "0",
            "got_value": f"{c7['identity_vs_plain']['max_absolute_difference']:.1e}",
            "metric": "absolute difference",
            "tolerance": "<1e-6",
            "verdict": "PASS" if c7["pass"] else "FAIL",
            "source": "artifacts/p176w2-n11/results.json:C7",
        },
    ]
    for case in c3["cases"]:
        rows.append({
            "face": "standalone",
            "id": f"C3@{case['impact_speed_m_s']:.0f}m_s",
            "quantity": "erosion ratio E [kg/kg] at 90 deg",
            "reference": "DNV-RP-O501 ductile design rate",
            "reference_value": f"{case['dnv_design_ratio_kg_kg']:.6e}",
            "got_value": f"{case['stream_ratio_kg_kg']:.6e}",
            "metric": f"relative deviation {case['relative_deviation']:.4%}",
            "tolerance": "<15%",
            "verdict": "PASS" if case["relative_deviation"] < 0.15 else "HONEST_FAIL",
            "source": "artifacts/p176w2-n11/results.json:C3.cases",
        })
    return rows


def rows_integration(integ: dict) -> list[dict]:
    rows = []
    for case in integ["cases"]:
        rows.append({
            "face": "integration",
            "id": f"pipe@{case['U_mix_m_s']:.1f}m_s",
            "quantity": "impact erosion ratio E [kg/kg] (pipe wall)",
            "reference": "same impacts replayed through DNV-RP-O501",
            "reference_value": f"{case['dnv_replay']['reference_erosion_ratio_kg_kg']:.6e}",
            "got_value": f"{case['erosion_ratio_kg_kg']:.6e}",
            "metric": f"model/ref = {1 + case['dnv_replay']['relative_deviation_of_ratio']:.1f}x",
            "tolerance": "<15% (model-class, not gated here)",
            "verdict": "HONEST_FAIL",
            "source": "artifacts/p176w2-n11/integration/results.json:cases",
        })
    rows.append({
        "face": "integration",
        "id": "I1",
        "quantity": "recorder ledger closure (pipe)",
        "reference": "0",
        "reference_value": "0",
        "got_value": f"{integ['max_ledger_residual']:.3e}",
        "metric": "relative residual",
        "tolerance": "<1e-6",
        "verdict": "PASS" if integ["criteria"]["I1_real_coupling_and_ledger"] else "FAIL",
        "source": "artifacts/p176w2-n11/integration/results.json",
    })
    rows.append({
        "face": "integration",
        "id": "I2",
        "quantity": "trajectory checksum, coupled vs uncoupled",
        "reference": "identical",
        "reference_value": "identical",
        "got_value": "identical" if integ["trajectory_identical"] else "differs",
        "metric": "sha256 equality",
        "tolerance": "identical",
        "verdict": "PASS" if integ["criteria"]["I2_observer_no_flow_change"] else "FAIL",
        "source": "artifacts/p176w2-n11/integration/results.json",
    })
    return rows


def main() -> int:
    main_res = json.loads((OUT / "results.json").read_text())
    integ_path = OUT / "integration" / "results.json"
    integ = json.loads(integ_path.read_text())
    rows = rows_main(main_res) + rows_integration(integ)

    with (OUT / "benchmark_ref_vs_got.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# p176w2-n11 — reference vs got (generated, do not hand-edit)", "",
             f"standalone all_pass = **{main_res['all_pass']}**; "
             f"integration all_pass = **{integ['all_pass']}**", "",
             "| face | id | quantity | reference | ref value | got value | metric | tol | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {face} | {id} | {quantity} | {reference} | {reference_value} | "
                     "{got_value} | {metric} | {tolerance} | {verdict} |".format(**r))
    lines += ["", "## honest FAIL (not gated, not overwritten)", "",
              f"- integration absolute agreement vs DNV at the pipe operating corner: "
              f"max relative deviation "
              f"{integ['dnv_absolute_agreement_honest_fail']['max_relative_deviation_model_vs_dnv']:.1f}x "
              f"(threshold 15%). Cause: {integ['dnv_absolute_agreement_honest_fail']['cause']}"]
    (OUT / "benchmark_ref_vs_got.md").write_text("\n".join(lines) + "\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "unstaged"
    except Exception:                                   # pragma: no cover
        commit = "unknown"
    artifacts = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "RUN_MANIFEST.json":
            artifacts[str(path.relative_to(ROOT))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest = {
        "node": "p176w2-n11",
        "repository": "TensorDEM",
        "module": "tensordem.erosion_dem",
        "base_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package": "tensordem",
        "repro_cmd": {
            "standalone": "python benchmarks/erosion_dem_verification.py",
            "integration": "python benchmarks/erosion_pipe_coupled.py",
            "evidence_pack": "python benchmarks/erosion_evidence_pack.py",
            "tests": "python -m pytest tests/test_erosion_dem.py "
                     "tests/test_erosion_pipe_integration.py -q",
            "cli": "python -m tensordem.erosion_dem impact --speed 30 --angle 90",
        },
        "entry_points": {
            "module": "python -m tensordem.erosion_dem {impact,stream,pipe,verify}",
            "console_script": "tensordem-erosion = tensordem.erosion_dem.cli:main",
        },
        "integration_callers": [
            "src/tensordem/slurry_transport/pipe_dem.py::"
            "PipeSectionDEMCFD._scan_wall_impacts -> ImpactErosionRecorder.record",
        ],
        "gate_evidence": {
            "L2_impl_files": [
                "src/tensordem/erosion_dem/__init__.py",
                "src/tensordem/erosion_dem/contact.py",
                "src/tensordem/erosion_dem/crater.py",
                "src/tensordem/erosion_dem/stream.py",
                "src/tensordem/erosion_dem/recorder.py",
                "src/tensordem/erosion_dem/breakage_bridge.py",
                "src/tensordem/erosion_dem/cli.py",
                "src/tensordem/erosion_dem/__main__.py",
                "src/tensordem/erosion_dem/report.py",
            ],
            "L2_entry": [
                "src/tensordem/erosion_dem/__main__.py",
                "pyproject.toml:[project.scripts].tensordem-erosion",
            ],
            "L2_run_artifacts": sorted(
                p for p in artifacts if p.startswith("artifacts/p176w2-n11/")
            ),
            "L3_integration_callers": [
                "src/tensordem/slurry_transport/pipe_dem.py",
                "benchmarks/erosion_pipe_coupled.py",
                "tests/test_erosion_pipe_integration.py",
                "artifacts/p176w2-n11/integration/results.json",
            ],
            "L4_benchmark": [
                "artifacts/p176w2-n11/benchmark_ref_vs_got.md",
                "artifacts/p176w2-n11/benchmark_ref_vs_got.csv",
            ],
            "L4_repro_cmd": "see repro_cmd above (all three commands in one shell)",
        },
        "criteria": {
            "standalone": main_res["criteria"],
            "integration": integ["criteria"],
        },
        "honest_fail": integ["dnv_absolute_agreement_honest_fail"],
        "wall_time_s": {
            "standalone": main_res["wall_time_s"],
            "integration": integ["wall_time_s"],
        },
        "artifacts": artifacts,
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2,
                                                      ensure_ascii=False))
    # generated integration document (no hand-typed numbers)
    doc = ROOT / "docs" / "P176W2_N11_DEM_EROSION_INTEGRATION.md"
    body = ["# p176w2-n11 — integration caller, entry points and repro (generated)",
            "",
            "> Generated by `benchmarks/erosion_evidence_pack.py` from "
            "`artifacts/p176w2-n11/results.json`, "
            "`artifacts/p176w2-n11/integration/results.json` and "
            "`artifacts/p176w2-n11/RUN_MANIFEST.json`. Do not hand-edit.",
            "",
            "## 1. Entry points (L2)",
            "",
            "| kind | value |", "|---|---|"]
    for k, v in manifest["entry_points"].items():
        body.append(f"| {k} | `{v}` |")
    body += ["", "Reproduce:", "", "```bash"]
    body += [f"{k}: {v}" for k, v in manifest["repro_cmd"].items()]
    body += ["```", "", "## 2. Integration caller (L3)", ""]
    body += [f"- `{c}`" for c in manifest["integration_callers"]]
    body += ["",
             "The slurry pipe DEM-CFD solver detects rising-edge wall contacts and "
             "hands `(speed, angle, diameter, density)` to "
             "`ImpactErosionRecorder.record`. With `erosion=None` the solver is "
             "bit-for-bit unchanged; the trajectory checksum below is the proof.",
             "", "| case | U_mix [m/s] | impacts | mean v [m/s] | mean alpha [deg] | "
             "E model [kg/kg] | E DNV replay | model/DNV | ledger residual | "
             "wall loss model [mm/yr] | wall loss DNV [mm/yr] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in integ["cases"]:
        body.append(
            f"| pipe | {c['U_mix_m_s']:.1f} | {c['n_impacts']} | "
            f"{c['mean_impact_speed_m_s']:.3f} | {c['mean_impact_angle_deg']:.2f} | "
            f"{c['erosion_ratio_kg_kg']:.4e} | "
            f"{c['dnv_replay']['reference_erosion_ratio_kg_kg']:.4e} | "
            f"{1 + c['dnv_replay']['relative_deviation_of_ratio']:.1f}x | "
            f"{c['ledger_residual']:.2e} | {c['wall_loss_mm_per_year']:.4f} | "
            f"{c['dnv_equivalent_wall_loss_mm_per_year']:.4f} |")
    body += ["",
             f"Trajectory checksum identical with/without the recorder: "
             f"**{integ['trajectory_identical']}**; zero-speed negative control: "
             f"**{integ['negative_control']['pass']}**; per-impact replay identity "
             f"residual max "
             f"{max(c['replay_decomposition']['max_per_impact_identity_residual'] for c in integ['cases']):.1e}.",
             "",
             "### honest FAIL (not gated, not overwritten)",
             "",
             f"Absolute agreement with DNV at the pipe operating corner is "
             f"{integ['dnv_absolute_agreement_honest_fail']['max_relative_deviation_model_vs_dnv']:.0f}x "
             f"(threshold 15%). {integ['dnv_absolute_agreement_honest_fail']['cause']}",
             "",
             "## 3. Benchmark (L4) and gate mapping", "",
             "| gate | evidence |", "|---|---|"]
    for k, v in manifest["gate_evidence"].items():
        if isinstance(v, list):
            body.append(f"| {k} | " + "<br>".join(f"`{x}`" for x in v) + " |")
        else:
            body.append(f"| {k} | {v} |")
    body += ["", "Full reference-vs-got table: "
             "`artifacts/p176w2-n11/benchmark_ref_vs_got.md` "
             "(machine-readable: `.csv`).", "",
             f"Wall time: standalone {manifest['wall_time_s']['standalone']} s, "
             f"integration {manifest['wall_time_s']['integration']} s.",
             "", "## 4. Figures", ""]
    for fig in main_res.get("figures", []):
        body.append(f"- `artifacts/p176w2-n11/{fig['name']}` — {fig['caption']}")
    for fig in integ.get("figures", []):
        body.append(f"- `artifacts/p176w2-n11/integration/{fig['name']}` — {fig['caption']}")
    doc.write_text("\n".join(body) + "\n")
    print(f"wrote {doc}")

    print(f"wrote {OUT/'benchmark_ref_vs_got.md'}")
    print(f"wrote {OUT/'benchmark_ref_vs_got.csv'}")
    print(f"wrote {OUT/'RUN_MANIFEST.json'} ({len(artifacts)} artifacts hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())