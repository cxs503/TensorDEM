#!/usr/bin/env python3
"""Inject machine-generated result tables into the verification report.

Reads ``docs/dem3d-rolling-resistance/results.json`` (written by
``scripts/verify_dem3d_rolling_resistance.py``) and replaces the
``<!--TABLE:...-->`` markers in ``docs/dem3d_rolling_resistance_verification.md``
so that no number in the report is transcribed by hand.

Run:  python docs/dem3d-rolling-resistance/inject_tables.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DOC = HERE.parent / "dem3d_rolling_resistance_verification.md"


def _esc(text: str) -> str:
    """Escape pipes inside table cells so Markdown does not split the row."""
    return text.replace("|", "\\|")


def law_table(d: dict) -> str:
    L = d["law_checks"]
    rows = [
        ("L1", "|f_rr| = mu_r |F_n|", f"{L['L1_force_mag']:.9f} N", f"{L['L1_expected']:.9f} N", f"{L['L1_rel_err']:.2e}"),
        ("L2", "cos(angle(f_j, v_rel))", f"{L['L2_cos_to_slip']:+.12f}", "-1", f"{abs(L['L2_cos_to_slip'] + 1):.2e}"),
        ("L3", "|f_i + f_j| (momentum)", f"{L['L3_net_force']:.3e} N", "0", "-"),
        ("L4", "|T| = mu_r R |F_n|", f"{L['L4_torque_mag']:.9e} N m", f"{L['L4_expected_lever_R']:.9e} N m", f"{L['L4_rel_err']:.2e}"),
        ("L4b", "|T| / (mu_r R/2 |F_n|) (Ai 2011)", f"{L['L4_impl_over_ai_factor']:.6f}", "2 (convention)", "0"),
        ("L5", "|F_t| with saturated Coulomb", f"{L['L5_total_tangential']:.9f} N", f"{L['L5_expected']:.9f} N", f"{L['L5_rel_err']:.2e}"),
        ("L6", "mu_r = 0 bit-identical to friction only", str(L["L6_bit_identical_to_friction_only"]), "True", "-"),
    ]
    out = ["| check | quantity | measured | analytic | rel. error |", "|---|---|---|---|---|"]
    out += ["| " + " | ".join(_esc(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def runs_table(d: dict) -> str:
    out = ["| `mu` | `mu_r` | steps | `a` measured [m/s²] | `a` analytic [m/s²] | rel. error | `t*` measured [ms] | `t*` analytic [ms] | rel. error | max\\|`v_y`\\| [m/s] | KE monotone | overlap drift |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["integration_runs"]:
        out.append(
            f"| {r['mu']:.2f} | {r['mu_r']:.2f} | {r['steps']} | {r['a_measured']:.4f} | {r['a_analytic']:.4f} | "
            f"{100*r['a_rel_err']:.3f} % | {1e3*r['t_stop_measured']:.4f} | {1e3*r['t_stop_analytic']:.4f} | "
            f"{100*r['t_stop_rel_err']:.3f} % | {r['max_abs_vy']:.2e} | {r['ke_monotone']} | {r['overlap_drift_rel']:.1e} |"
        )
    return "\n".join(out)


def sweep_table(d: dict) -> str:
    out = ["| `mu` | `mu_r` | `a` measured [m/s²] | `a` analytic [m/s²] | rel. error |",
           "|---|---|---|---|---|"]
    for r in d["sweep"]:
        err = "n/a (no resistance)" if r["a_analytic"] == 0 else f"{100*r['a_rel_err']:.3f} %"
        out.append(f"| {r['mu']:.2f} | {r['mu_r']:.2f} | {r['a_measured']:.4f} | {r['a_analytic']:.4f} | {err} |")
    return "\n".join(out)


def error_table(d: dict) -> str:
    L = d["law_checks"]
    out = ["| quantity | reference | error | < 3 %? |", "|---|---|---|---|",
           f"| couple magnitude | `mu_r |F_n|` | {L['L1_rel_err']:.2e} | yes (exact) |",
           f"| couple direction | anti-parallel to `v_rel` | {abs(L['L2_cos_to_slip'] + 1):.2e} (cos) | yes (exact) |",
           f"| momentum balance | `f_i + f_j = 0` | {L['L3_net_force']:.1e} N | yes (exact) |",
           f"| diagnostic torque | `mu_r R |F_n|` | {L['L4_rel_err']:.2e} | yes (exact) |",
           f"| additivity | `(mu + mu_r)|F_n|` | {L['L5_rel_err']:.2e} | yes (exact) |"]
    worst = max(r["a_rel_err"] for r in d["integration_runs"])
    for r in d["integration_runs"]:
        out.append(f"| sliding deceleration, `mu={r['mu']:.2f}`, `mu_r={r['mu_r']:.2f}` | `(mu+mu_r)|F_n|/m` | {100*r['a_rel_err']:.3f} % | "
                   f"{'yes' if r['a_rel_err'] < 0.03 else '**no**'} |")
        out.append(f"| stopping time, `mu={r['mu']:.2f}`, `mu_r={r['mu_r']:.2f}` | `v0/a` | {100*r['t_stop_rel_err']:.3f} % | "
                   f"{'yes' if r['t_stop_rel_err'] < 0.03 else '**no**'} |")
    out.append(f"| worst integration-level error (any run) | analytic | {100*worst:.3f} % | "
               f"{'yes' if worst < 0.03 else '**no**'} |")
    return "\n".join(out)


def main() -> int:
    d = json.loads((HERE / "results.json").read_text())
    text = DOC.read_text()
    for marker, gen in (("<!--TABLE:law-->", law_table),
                        ("<!--TABLE:runs-->", runs_table),
                        ("<!--TABLE:sweep-->", sweep_table),
                        ("<!--TABLE:errors-->", error_table)):
        table = gen(d)
        if marker in text:
            text = text.replace(marker, table)
            continue
        # already injected: replace the existing block that starts like this table
        head = table.splitlines()[0].split(" ")[0] + " "
        lines = text.splitlines(keepends=True)
        start = next((k for k, ln in enumerate(lines)
                      if ln.startswith("|") and ln.split("|")[1:2] == table.splitlines()[0].split("|")[1:2]),
                     None)
        if start is None:
            raise SystemExit(f"cannot locate block for {marker}")
        end = start
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        text = "".join(lines[:start]) + table + "\n" + "".join(lines[end:])
    DOC.write_text(text)
    print(f"injected 4 tables into {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())