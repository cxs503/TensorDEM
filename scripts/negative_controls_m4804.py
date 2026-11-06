#!/usr/bin/env python3
"""Quantify the detection margins of the 2-D energy/fracture negative controls.

Runs the tool-indentation scenario once and answers, for every ledger channel,
the question the shipped negative controls leave open: *how large must a fault
be before the balance gate can see it?*  The gate is linear in the channels, so
the accepted set is a closed-form interval and the answer has no fitted
parameters:

    accepted:  |r0 + s_ch * (x - x0)| <= tol * S0
    resolution: min distance from x0 to the nearer edge of that interval
              = tol * S0 - |r0|            (additive, in J)
              = (tol * S0 - |r0|) / |x0|   (multiplicative, |alpha - 1|)

Each closed-form boundary is re-found by bisecting the gate itself, so the two
independent routes are cross-checked.  Outputs:

    artifacts/m4804_negative_control_margins/margins.json
    artifacts/m4804_negative_control_margins/negative_controls.log
    docs/dem2d_energy_fracture_ledger_negative_controls_m4804.md
        -> the table between the ``m4804-injected-table`` markers is rebuilt
           byte-identically on every run (no hand-transcribed numbers)

Usage::

    python3 scripts/negative_controls_m4804.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_dem2d_energy_fracture_ledger_negative import (  # noqa: E402
    ACTIVE_CHANNELS,
    LATTICE,
    SIGNS,
    TOL,
    DURATION,
    TOOL_SPEED,
    _clean,
    _energy_residual,
    _energy_scale,
    _rel_err,
    acceptance_band,
    bisect_min_detectable,
    fault_margin,
    min_detectable_fault,
    shipped_faults,
)

OUT_DIR = ROOT / "artifacts" / "m4804_negative_control_margins"
DOC = ROOT / "docs" / "dem2d_energy_fracture_ledger_negative_controls_m4804.md"
BEGIN = "<!-- BEGIN:m4804-injected-table -->"
END = "<!-- END:m4804-injected-table -->"


def build_report() -> dict:
    """Assemble every number in the ledger margin table."""
    sim = _clean()
    r0 = _energy_residual(sim)
    scale = _energy_scale(sim)
    budget = TOL * scale
    headroom = budget - abs(r0)

    channels = []
    for channel in SIGNS:
        value = float(sim.accounting[channel])
        additive = min_detectable_fault(sim, channel, "additive")
        multiplicative = min_detectable_fault(sim, channel, "multiplicative")
        numeric = bisect_min_detectable(sim, channel, "additive")
        band = acceptance_band(sim, channel)
        channels.append(
            {
                "channel": channel,
                "sign": SIGNS[channel],
                "value_J": value,
                "band_J": list(band),
                "additive_resolution_J": additive["size"],
                "additive_resolution_fraction_of_value": (
                    additive["size"] / abs(value) if value else None
                ),
                "additive_bisection_J": numeric,
                "additive_bisection_gap_rel": (
                    abs(numeric - additive["size"]) / additive["size"]
                    if additive["size"]
                    else 0.0
                ),
                "multiplicative_resolution_abs_dalpha": (
                    multiplicative["size"] if multiplicative["reachable"] else None
                ),
                "multiplicative_reachable": multiplicative["reachable"],
                "multiplicative_reason": multiplicative["reason"],
            }
        )

    controls = [fault_margin(sim, fault) for fault in shipped_faults(sim)]
    data_scale_gaps = []
    for channel in ACTIVE_CHANNELS:
        for mode in ("additive", "multiplicative"):
            clean_boundary = bisect_min_detectable(sim, channel, mode, scale="clean")
            data_boundary = bisect_min_detectable(sim, channel, mode, scale="data")
            data_scale_gaps.append(abs(data_boundary - clean_boundary) / clean_boundary)

    report = {
        "case": "dem2d tool indentation, breaks bonds",
        "lattice": list(LATTICE),
        "duration_s": DURATION,
        "tool_speed_m_per_s": TOOL_SPEED,
        "steps": int(round(DURATION / sim.dt)),
        "dt_s": float(sim.dt),
        "broken_bonds": int(sim.broken_bonds),
        "gate": {
            "tolerance_rel": TOL,
            "clean_residual_J": r0,
            "clean_rel_residual": _rel_err(sim),
            "reference_scale_J": scale,
            "budget_J": budget,
            "headroom_J": headroom,
            "closes": abs(r0) < budget,
        },
        "channels": channels,
        "controls": controls,
        "checks": {
            "all_controls_detected": all(
                c["detected_clean_scale"] and c["detected_shipped_gate"] for c in controls
            ),
            "min_control_margin": min(c["margin"] for c in controls),
            "min_control_margin_channel": min(
                controls, key=lambda c: c["margin"]
            )["channel"],
            "max_bisection_gap_rel": max(
                c["additive_bisection_gap_rel"] for c in channels
            ),
            "max_data_scale_boundary_shift_rel": max(data_scale_gaps),
        },
    }
    return report


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}e}"


def render_tables(report: dict) -> str:
    """Render the injected Markdown table (deterministic, run to run)."""
    gate = report["gate"]
    lines = [
        "| 通道 | 清洁值 $x_0$ (J) | 加性分辨率 $d^\*$ (J) | $d^\*/x_0$ | "
        "乘性分辨率 $\\lvert\\alpha^\*-1\\rvert$ | 二分复算 $d^\*$ (J) | 相对偏差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["channels"]:
        fraction = row["additive_resolution_fraction_of_value"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["channel"],
                _fmt(row["value_J"]),
                _fmt(row["additive_resolution_J"]),
                "—（休眠）" if fraction is None else f"{fraction:.2e}",
                "—（休眠通道无乘性控制）"
                if not row["multiplicative_reachable"]
                else f"{row['multiplicative_resolution_abs_dalpha']:.3e}",
                _fmt(row["additive_bisection_J"]),
                f"{row['additive_bisection_gap_rel']:.1e}",
            )
        )
    lines += [
        "",
        "判据裕度 $d^\\* = \\mathrm{tol}\\cdot S_0 - \\lvert r_0\\rvert$ = "
        f"{_fmt(gate['headroom_J'])} J"
        f"（tol = {gate['tolerance_rel']:.3g}，$S_0$ = {_fmt(gate['reference_scale_J'])} J，"
        f"$r_0$ = {_fmt(gate['clean_residual_J'])} J）。",
        "",
        "| 已装负控故障 | 通道 | 模式 | 故障幅度 | 分辨率 | 裕度 $M$ | 检出 |",
        "|---|---|---|---|---|---|---|",
    ]
    for control in report["controls"]:
        size = control["fault_size"]
        if control["mode"] == "multiplicative":
            size_txt = f"$\\alpha$ = {control['factor']:.3g}（$\\Delta\\alpha$ = {size:.3g}）"
        else:
            size_txt = f"$+${size:.3e} J"
        lines.append(
            "| {} | `{}` | {} | {} | {} | {:.3g} | {} |".format(
                "零化" if control["factor"] == 0.0
                else ("三倍" if control["factor"] == 3.0 else "加性"),
                control["channel"],
                control["mode"],
                size_txt,
                _fmt(control["min_detectable"]),
                control["margin"],
                "是" if (control["detected_clean_scale"]
                        and control["detected_shipped_gate"]) else "**否**",
            )
        )
    return "\n".join(lines)


def inject_doc(table: str) -> bool:
    """Replace the marked region of the document; idempotent and byte-stable."""
    if not DOC.exists():
        return False
    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{DOC} is missing the {BEGIN!r}/{END!r} markers")
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    new = f"{head}{BEGIN}\n{table}\n{END}{tail}"
    if new != text:
        DOC.write_text(new)
        return True
    return False


def main() -> int:
    report = build_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (OUT_DIR / "margins.json").write_text(payload)

    table = render_tables(report)
    changed = inject_doc(table)

    gate = report["gate"]
    log = [
        f"case: {report['case']}",
        f"lattice: {report['lattice']}, dt: {report['dt_s']:.6e} s, "
        f"steps: {report['steps']}, duration: {report['duration_s']} s",
        f"broken bonds: {report['broken_bonds']}",
        f"clean residual r0: {gate['clean_residual_J']:.9e} J "
        f"(rel {gate['clean_rel_residual']:.6e} < tol {gate['tolerance_rel']:.3g})",
        f"reference scale S0: {gate['reference_scale_J']:.9e} J",
        f"budget tol*S0: {gate['budget_J']:.9e} J",
        f"headroom d*: {gate['headroom_J']:.9e} J",
        "",
        "channel | value_J | additive_d*_J | rel_gap(bisection) | mult |da*|",
    ]
    for row in report["channels"]:
        log.append(
            "{} | {:.9e} | {:.9e} | {:.2e} | {}".format(
                row["channel"], row["value_J"], row["additive_resolution_J"],
                row["additive_bisection_gap_rel"],
                "dormant" if not row["multiplicative_reachable"]
                else f"{row['multiplicative_resolution_abs_dalpha']:.9e}",
            )
        )
    log.append("")
    log.append("shipped control | channel | mode | size | resolution | margin | detected")
    for control in report["controls"]:
        log.append(
            "{} | {} | {} | {:.9e} | {:.9e} | {:.3f} | {}".format(
                control["factor"] if control["factor"] is not None else "offset",
                control["channel"], control["mode"], control["fault_size"],
                control["min_detectable"], control["margin"],
                control["detected_clean_scale"] and control["detected_shipped_gate"],
            )
        )
    checks = report["checks"]
    log += [
        "",
        f"all controls detected: {checks['all_controls_detected']}",
        f"min control margin: {checks['min_control_margin']:.3f} "
        f"({checks['min_control_margin_channel']})",
        f"max bisection gap (rel): {checks['max_bisection_gap_rel']:.3e}",
        "max boundary shift from the data-derived scale convention: "
        f"{checks['max_data_scale_boundary_shift_rel']:.3e}",
        f"doc table injected/updated: {changed}",
    ]
    (OUT_DIR / "negative_controls.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())