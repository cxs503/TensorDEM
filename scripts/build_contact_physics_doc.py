"""Refresh the acceptance table of ``docs/contact_physics_module.md``.

Every number in the table is read from
``artifacts/contact_physics/integration_report.json`` (and the artifacts it
points at), so the published page cannot drift from the machine-generated
evidence -- the same anti-drift rule the repository uses elsewhere
(``scripts/verify_ridge_grounding_rederivation.py``).

Usage
-----
``python3 scripts/build_contact_physics_doc.py``
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "contact_physics_module.md"
REPORT = ROOT / "artifacts" / "contact_physics" / "integration_report.json"
MARKER = "<!-- ACCEPTANCE-TABLE -->"


def build_table(report: dict) -> str:
    acceptance = report["acceptance"]
    lines = [
        f"Generated from `artifacts/contact_physics/integration_report.json` "
        f"(git `{report['git_rev'][:12]}`, {report['generated_at_utc']}).",
        "",
        "| criterion | requirement | measured | pass | artifact |",
        "|---|---|---|---|---|",
        (f"| (a) integration | integrated run == three standalone runs, "
         f"max rel. error < {report['regimes'][next(iter(report['regimes']))]['tolerance']:.0e} | "
         f"{acceptance['a_integration']['max_relative_error']:.1e} | "
         f"{acceptance['a_integration']['passed']} | "
         f"`{acceptance['a_integration']['artifact']}` |"),
        (f"| (b) API stability | zero signature changes vs pre-integration baseline "
         f"`{report['api']['baseline_git_rev'][:12]}` | "
         f"{acceptance['b_api_stability']['signature_changes']} changes | "
         f"{acceptance['b_api_stability']['passed']} | "
         f"`{acceptance['b_api_stability']['artifact']}` |"),
        (f"| (c) performance | time and Python peak-memory overhead < 5 % | "
         f"time {acceptance['c_performance']['time_overhead_percent']:.2f} %, "
         f"memory {acceptance['c_performance']['memory_overhead_percent']:.2f} % | "
         f"{acceptance['c_performance']['passed']} | "
         f"`{acceptance['c_performance']['artifact']}` |"),
        (f"| (d) regression gate | upstream N02/N05/N08 suites green | "
         f"{acceptance['d_regression']['passed_tests']} passed, "
         f"{acceptance['d_regression']['failed_tests']} failed, "
         f"{acceptance['d_regression']['errors']} errors | "
         f"{acceptance['d_regression']['passed']} | "
         f"`{acceptance['d_regression']['artifact']}` |"),
        (f"| (e) documentation | all public API documented, example runs | "
         f"coverage {acceptance['e_documentation']['coverage'] * 100:.0f} % | "
         f"{acceptance['e_documentation']['passed']} | "
         f"`{acceptance['e_documentation']['artifact']}` |"),
        "",
        "### Per-regime parity (criterion a)",
        "",
        "| regime | steps | duration [s] | max rel. error | T_int end [degC] | mu end [-] "
        "| film end [m] | damage max | broken bonds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, regime in report["regimes"].items():
        final = regime["final"]
        lines.append(
            f"| {name} | {regime['steps']} | {regime['duration_s']:.4f} "
            f"| {regime['max_relative_error']:.1e} "
            f"| {final['interface_temperature_c']:.4f} | {final['mu']:.5f} "
            f"| {final['film_thickness_m']:.3e} | {final['damage_max']:.4f} "
            f"| {int(final['broken_bonds'])} |")
    lines += [
        "",
        "### Performance (criterion c)",
        "",
        "| quantity | integrated | standalone (three drivers) | overhead |",
        "|---|---|---|---|",
        (f"| wall time (min of {report['performance']['repeats']}, 1 thread) [s] "
         f"| {report['performance']['integrated_wall_s']:.4f} "
         f"| {report['performance']['standalone_wall_s']:.4f} "
         f"| {report['performance']['time_overhead_percent']:+.2f} % |"),
        (f"| wall time (median of {report['performance']['repeats']}) [s] "
         f"| {report['performance']['integrated_wall_median_s']:.4f} "
         f"| {report['performance']['standalone_wall_median_s']:.4f} "
         f"| {report['performance']['time_overhead_median_percent']:+.2f} % |"),
        (f"| Python peak (tracemalloc, min) [bytes] "
         f"| {report['performance']['integrated_peak_py_bytes']} "
         f"| {report['performance']['standalone_peak_py_bytes']} "
         f"| {report['performance']['memory_overhead_percent']:+.2f} % |"),
        "",
        f"All five criteria pass: **{report['all_passed']}**.  Figures: "
        + ", ".join(f"`artifacts/contact_physics/{figure['name']}`"
                    for figure in report["figures"])
        + ".",
    ]
    return "\n".join(lines)


def main() -> int:
    report = json.loads(REPORT.read_text())
    table = build_table(report)
    text = DOC.read_text()
    if MARKER not in text:
        raise SystemExit(f"marker {MARKER} missing from {DOC}")
    head, _, tail = text.partition(MARKER)
    DOC.write_text(head + MARKER + "\n\n" + table + "\n" + tail.split("\n", 1)[1])
    print(f"refreshed {DOC.relative_to(ROOT)} acceptance table "
          f"({len(table.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())