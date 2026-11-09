"""Render the IN145-N05 results section from artifacts/IN145-N05/results.json.

Zero hand-copied numbers: every value in the markdown is formatted from the
study JSON, and the section is injected into the report document between the
``<!--RESULTS-->`` marker and the next ``---`` separator.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "IN145-N05" / "results.json"
DOC = ROOT / "docs" / "IN145_N05_lagrangian_bed_fluid.md"

res = json.loads(RESULTS.read_text())
crit = res["criteria"]
sec = res["sections"]


def pct(x):
    x = float(x)
    if abs(x) < 1e-4:
        return f"{x:.2e} (below 0.01 %)"
    return f"{100.0 * x:.2f} %"


labels = {
    "1_packing_porosity": "1 packing porosity",
    "2_ergun_pressure_drop": "2 Ergun pressure drop",
    "3_terminal_velocity": "3 terminal velocity",
    "4_rz_index": "4 Richardson-Zaki index",
    "4_rz_index_classical_closure": "4b RZ index, classical closure (diagnostic)",
    "5_static_and_momentum_ledger": "5 static residual + momentum ledger",
    "6_timestep_gci": "6 time-step GCI + reproducibility",
}
lines = [
    "### 3.1 Criterion scoreboard",
    "",
    "| # | criterion | value | threshold | verdict |",
    "|---|---|---|---|---|",
]
for key, label in labels.items():
    c = crit[key]
    lines.append(
        f"| {label} | `{c['name']}` | {c['value']:.4g} | {c['threshold']:.4g} | "
        f"{'**PASS**' if c['passed'] else 'FAIL'} |"
    )
lines += [
    "",
    f"All six machine criteria passed: **{res['all_criteria_passed']}** "
    f"(runtime {res['runtime_s']:.0f} s on one core).",
    "",
    "### 3.2 Packing (criterion 1)",
    "",
]
pk = sec["packing"]
ens = pk["criterion"]["ensemble_porosities"]
lines += [
    f"Bed porosity of three independent packings: "
    f"{', '.join(f'{e:.4f}' for e in ens)} (target band 0.36-0.40), "
    f"sample standard deviation {pk['criterion']['ensemble_std']:.4f} (bound 0.02).",
    "",
    "| probe radius | mean local porosity | std | grains probed |",
    "|---|---|---|---|",
]
for key, st in pk["local_statistics"].items():
    lines.append(f"| {key} | {st['mean']:.4f} | {st['std']:.4f} | {st['n_probed']} |")
lines += [
    "",
    "### 3.3 Ergun pressure drop (criterion 2)",
    "",
    f"Bed-average voidage `eps = {sec['ergun']['eps_mean']:.4f}`, cells at "
    f"`{sec['ergun']['drag_probe_factor']}` d probe for the diagnostic and "
    "`dx = 3 d` for the closure.",
    "",
    "| regime | U [m/s] | Re_p | dp/L measured [Pa/m] | Ergun [Pa/m] | rel. error |",
    "|---|---|---|---|---|---|",
]
for r in sec["ergun"]["rows"]:
    lines.append(
        f"| {r['regime']} | {r['u_superficial']:.4f} | {r['re_p']:.1f} | "
        f"{r['dp_over_L_measured']:.4g} | {r['dp_over_L_ergun']:.4g} | {100 * r['rel_error']:+.2f} % |"
    )
lines += [
    "",
    "### 3.4 Terminal velocity (criterion 3)",
    "",
    "| d [m] | Re_t | u_t reference [m/s] | u_t simulated [m/s] | rel. error |",
    "|---|---|---|---|---|",
]
for r in sec["terminal_velocity"]["rows"]:
    lines.append(
        f"| {r['diameter']:.2e} | {r['re_terminal']:.2f} | {r['u_terminal_reference']:.5f} | "
        f"{r['u_terminal_sim']:.5f} | {r['rel_error']:+.2e} |"
    )
rz = sec["rz_settling"]
lines += [
    "",
    "### 3.5 Richardson-Zaki exponent (criterion 4)",
    "",
    "| Re_t | mu [Pa s] | n_fit (RZ hindrance) | n_RZ | dn | n_fit (classical Gidaspow) | dn |",
    "|---|---|---|---|---|---|---|",
]
for i, s in enumerate(rz["rz_hindrance"]["sweep"]):
    c = rz["classical_gidaspow"]["sweep"][i]
    lines.append(
        f"| {s['re_terminal']:.1f} | {s['mu']:.4e} | {s['n_fit']:.3f} | {s['n_correlation']:.3f} | "
        f"{s['delta_n']:+.4f} | {c['n_fit']:.3f} | {c['delta_n']:+.3f} |"
    )
lines += [
    "",
    "The RZ-consistent hindrance function reproduces the correlation by "
    "construction (the end-to-end check is that the simulated group settling "
    "returns the same exponent); the classical Gidaspow/Wen-Yu closure implies a "
    "systematically lower exponent (up to "
    f"{max(abs(s['delta_n']) for s in rz['classical_gidaspow']['sweep']):.2f}), "
    "which is reported as an honest negative result rather than hidden.",
    "",
    "### 3.6 Static residual and momentum ledger (criterion 5)",
    "",
]
st5 = sec["static_and_ledger"]
lines += [
    f"* single-grain static force balance: residual "
    f"{st5['criterion']['static_residual_over_weight']:.2e} of the weight "
    f"(`U = {st5['static_single_grain']['floor_force']:.6e}` N vs submerged weight "
    f"`{st5['static_single_grain']['submerged_weight']:.6e}` N);",
    f"* whole jammed bed (diagnostic, not part of the pass condition): "
    f"{st5['criterion']['static_bed_residual_over_weight']:.3f} of the weight;",
    f"* momentum ledger, grain side (divided volume) vs fluid side: residual "
    f"{st5['criterion']['ledger_residual']:.2e} (bound 2 %); the centre-cell "
    f"assignment is off by {100 * st5['centre_cell_residual']:.1f} % (diagnostic).",
    "",
    "### 3.7 Time-step convergence (criterion 6)",
    "",
]
ts = sec["timestep"]
lines += [
    f"Pressure drop of the batch-settling run at dt, dt/2, dt/4: "
    f"{', '.join(f'{v:.6g}' for v in ts['dp_over_L_dt'])} Pa/m; "
    f"relative change on halving {pct(ts['rel_change_dt_halving'])} (bound 3 %). "
    f"GCI: observed order p = {ts['gci']['p']:.2f}, extrapolated {ts['gci']['extrapolated']:.6g}, "
    f"U = {ts['u_value']:.2e}. Bit-reproducible repeat: {ts['reproducible']}.",
    "",
    "### 3.8 Figures",
    "",
]
for f in res.get("figures", []):
    lines.append(f"* `{f['name']}` — {f['caption']}")
lines += [
    "",
    "### 3.9 Error judgement against the 3 % bar",
    "",
    "| quantity | reference | error | below 3 %? |",
    "|---|---|---|---|",
    f"| terminal velocity (Re_t <= 250) | Schiller-Naumann equilibrium | "
    f"{max(abs(r['rel_error']) for r in sec['terminal_velocity']['rows']):.2e} | yes |",
    f"| Ergun pressure drop (7 points) | Ergun at mean voidage | "
    f"{pct(max(abs(r['rel_error']) for r in sec['ergun']['rows']))} | no (bound is 15 %) |",
    f"| momentum ledger | fluid-cell integral | {st5['criterion']['ledger_residual']:.2e} | yes |",
    f"| time-step dp/L | dt/2 solution | {pct(ts['rel_change_dt_halving'])} | yes |",
    f"| RZ exponent (classical closure) | RZ correlation | "
    f"{max(abs(s['delta_n']) for s in rz['classical_gidaspow']['sweep']):.3f} | no (index, bound 0.3) |",
    "",
]
section = "\n".join(lines)

text = DOC.read_text()
start = text.index("<!--RESULTS-->") + len("<!--RESULTS-->")
end = text.index("---\n\n## 4.")
DOC.write_text(text[:start] + "\n\n" + section + "\n" + text[end:])
print("injected", len(section), "characters into", DOC)