"""Inject the acceptance numbers of node p223w2-n03 into the markdown report.

Zero hand-copying: every number in the report's "results" block is written from
``artifacts/p223w2n03/acceptance.json`` (which is itself produced by the campaign
runner from the raw run records).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "artifacts" / "p223w2n03" / "acceptance.json"
DOC = ROOT / "docs" / "P223W2N03_ICE_RAFTING_BRAZIL_NUT.md"
START = "<!-- NUMBERS:START -->"
END = "<!-- NUMBERS:END -->"


def _f(x, digits=3, scale=1.0):
    if x is None:
        return "n/a"
    try:
        return f"{float(x) * scale:.{digits}g}"
    except (TypeError, ValueError):
        return str(x)


def build_block(acc: dict) -> str:
    lines = []
    ra = acc.get("acceptance_rafting", {})
    ca = acc.get("acceptance_convection", {})
    acc1 = ra.get("AC1", {})
    acc2 = ra.get("AC2", {})
    acc3 = ca.get("AC3", {})
    acc4 = ca.get("AC4", {})
    acc5 = ca.get("AC5", {})
    on = ca.get("onset_check", {})
    ridge = acc["faces"]["rafting"]["ridge"]
    base = acc["faces"]["rafting"]["ramp_cases"][0]
    runs = acc["faces"]["convection"]["runs"]

    lines.append("| quantity | computed | anchor | rel. error | verdict |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| AC1 rafting threshold tau_raft (h=0.5 m, L=6 m, 45 deg) "
                 f"| {_f(acc1.get('value'), 4, 1e-3)} kPa "
                 f"| {_f(acc1.get('anchor'), 3, 1e-3)} kPa (range "
                 f"{_f(acc1['anchor_range'][0], 3, 1e-3)}-{_f(acc1['anchor_range'][1], 3, 1e-3)} kPa) "
                 f"| {_f(acc1.get('rel_error'), 3)} "
                 f"| **{acc1.get('verdict', 'n/a')}** (needs |err| <= 0.15) |")
    lines.append(f"| AC2 ridge sail/keel H_s/H_k "
                 f"| {_f(acc2.get('value'), 3)} "
                 f"| {_f(acc2.get('anchor'), 3)} (range {_f(acc2['anchor_range'][0], 2)}-"
                 f"{_f(acc2['anchor_range'][1], 2)}) "
                 f"| {_f(acc2.get('rel_error'), 3)} | **{acc2.get('verdict', 'n/a')}** "
                 f"(stable chain: {acc2.get('stable')}) |")
    lines.append(f"| AC2b ridge sail aspect H_s/W_s "
                 f"| {_f(acc2.get('sail_aspect', {}).get('value'), 3)} "
                 f"| {_f(acc2.get('sail_aspect', {}).get('anchor'), 3)} "
                 f"| {_f(acc2.get('sail_aspect', {}).get('rel_error'), 3)} | reported |")
    lines.append(f"| AC3 Brazil-nut segregation timescale (Gamma=6, d_l/d_s=2) "
                 f"| {_f(acc3.get('value'), 3)} cycles "
                 f"| {_f(acc3.get('anchor'), 3)} cycles (range "
                 f"{_f(acc3['anchor_range'][0], 3)}-{_f(acc3['anchor_range'][1], 3)}) "
                 f"| {_f(acc3.get('rel_error'), 3)} | **{acc3.get('verdict', 'n/a')}** "
                 f"(needs |err| <= 0.10) |")
    lines.append(f"| AC4 reverse Brazil-nut boundary rho_l/rho_s "
                 f"| {_f(acc4.get('value'), 3)} "
                 f"| {_f(acc4.get('anchor'), 3)} (range {_f(acc4['anchor_range'][0], 2)}-"
                 f"{_f(acc4['anchor_range'][1], 2)}) "
                 f"| {_f(acc4.get('rel_error'), 3)} | **{acc4.get('verdict', 'n/a')}** |")
    lines.append(f"| AC5 negative control: no vibration "
                 f"| dz = {_f(acc5.get('dz_control_mm'), 3)} mm "
                 f"vs {_f(acc5.get('dz_bne_mm'), 3)} mm shaken "
                 f"| - | ratio {_f(acc5.get('ratio'), 3)} "
                 f"| **{acc5.get('verdict', 'n/a')}** (needs < 0.25) |")
    lines.append(f"| onset check: sub-critical Gamma={_f(on.get('onset_gamma'), 3)} "
                 f"| dz = {_f(on.get('dz_subcritical_mm'), 3)} mm "
                 f"vs {_f(on.get('dz_supercritical_mm'), 3)} mm at Gamma=6 "
                 f"| - | - | **{on.get('verdict', 'n/a')}** |")
    lines.append("")
    lines.append("Raw run parameters:")
    lines.append("")
    lines.append(f"* rafting ramp: dt = {_f(base.get('dt'), 3)} s, "
                 f"{base.get('n_steps')} steps, climb reached "
                 f"{_f(base.get('climb_max'), 3)} h")
    lines.append(f"* ridge chain: convergence {_f(ridge.get('convergence'), 3)}, "
                 f"piston load {_f(ridge.get('piston_force'), 4)} N, "
                 f"sail height {_f(ridge.get('sail_height'), 4)} m, "
                 f"keel depth {_f(ridge.get('keel_depth'), 4)} m "
                 f"(h = {_f(ridge.get('h'), 2)} m)")
    lines.append(f"* convection bed: {_f(runs['bne'].get('n_particles'), 4)} particles, "
                 f"dt = {_f(runs['bne'].get('dt'), 3)} s, "
                 f"{_f(runs['bne'].get('n_cycles'), 3)} cycles at "
                 f"{_f(runs['bne'].get('frequency'), 3)} Hz, "
                 f"amplitude {_f(runs['bne'].get('amplitude'), 3, 1e3)} mm")
    lines.append(f"* Brazil-nut dz (Gamma=6): {_f(runs['bne'].get('dz_from_initial'), 3, 1e3)} mm; "
                 f"dz (no shaking): {_f(runs['control_novibration'].get('dz_from_initial'), 3, 1e3)} mm; "
                 f"dz (Gamma=1): {_f(runs['subcritical'].get('dz_from_initial'), 3, 1e3)} mm")
    lines.append("")
    lines.append("Reverse Brazil-nut boundary per shaking intensity:")
    lines.append("")
    lines.append("| Gamma | rho boundary | scan (rho ratio: dz [mm]) |")
    lines.append("|---|---|---|")
    for b in ca.get("AC4", {}).get("boundaries", []):
        scan = ", ".join(f"{r:g}: {d*1e3:+.2f}" for r, d in b.get("scan", []))
        lines.append(f"| {b['gamma']:g} | {_f(b.get('rho_boundary'), 3)} | {scan} |")
    return "\n".join(lines)


def main() -> int:
    acc = json.loads(ACC.read_text())
    block = build_block(acc)
    text = DOC.read_text()
    i, j = text.index(START), text.index(END)
    text = text[:i + len(START)] + "\n\n" + block + "\n\n" + text[j:]
    DOC.write_text(text)
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())