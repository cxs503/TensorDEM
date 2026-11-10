#!/usr/bin/env python
"""Emit the markdown tables of the LW3-N06 report straight from results.json.

Zero hand-copying: every number below is read from
``artifacts/bulk_cargo/results.json``; the output is pasted verbatim into
``docs/bulk_cargo_discharge_verification.md`` (and is the same data the figures
plot).  ``python scripts/m8875_bulk_cargo_tables.py [--results ...]``.
"""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/bulk_cargo/results.json")
    args = ap.parse_args(argv)
    doc = json.load(open(args.results))
    c = doc["criteria"]
    raw = doc["raw"]
    out = []

    out.append("### Criteria verdicts\n")
    out.append("| criterion | measured | reference | deviation | verdict |")
    out.append("|---|---|---|---|---|")
    c1 = c["C1_repose_angle"]
    out.append(f"| C1 repose angle | {c1['theta_deg']:.2f} deg | "
               f"{c1['reference_deg']:.1f} deg | "
               f"{c1['rel_dev_vs_reference'] * 100:.1f} % | "
               f"{'PASS' if c1['pass'] else 'FAIL'} |")
    c2 = c["C2_beverloo_curve"]
    out.append(f"| C2 Beverloo curve (worst point) | worst {c2['worst_rel_dev'] * 100:.1f} % "
               f"| 0 % | {c2['worst_rel_dev'] * 100:.1f} % | "
               f"{'PASS' if c2['pass'] else 'FAIL'} |")
    c3 = c["C3_janssen_pressure"]
    out.append(f"| C3 Janssen profile (worst slab) | worst {c3['worst_point_rel_dev'] * 100:.1f} % | 0 % | "
               f"{c3['worst_point_rel_dev'] * 100:.1f} % | {'PASS' if c3['pass'] else 'FAIL'} |")
    out.append(f"| C3 identity p_inf/lambda = rho_b g | {c3['identity_measured']:.1f} Pa/m | "
               f"{c3['identity_theory']:.1f} Pa/m | {c3['identity_rel_dev'] * 100:.1f} % | "
               f"{'PASS' if c3['identity_rel_dev'] < 0.15 else 'FAIL'} |")
    c4a, c4b = c["C4a_frictionless_repose"], c["C4b_blocked_orifice"]
    out.append(f"| C4a frictionless repose | {c4a['theta_deg']:.2f} deg | < 5 deg | "
               f"- | {'PASS' if c4a['pass'] else 'FAIL'} |")
    out.append(f"| C4b blocked orifice W | {c4b['flow_rate_kg_s']:.4f} kg/s | "
               f"<= 5 % of the D/d=4 rate ({c4b['ratio_to_smallest_flowing_case'] * 100:.2f} %) | "
               f"- | {'PASS' if c4b['pass'] else 'FAIL'} |")
    out.append(f"| C5 end-to-end JSON keys | {', '.join(c['C5_payload']['keys'])} | "
               f"repose_angle/beverloo_curve/janssen_pressure | - | "
               f"{'PASS' if c['C5_payload']['pass'] else 'FAIL'} |")
    out.append("")

    out.append("### Poured-heap cases (repose angle)\n")
    out.append("| case | mu | mu_r | theta [deg] | tan(theta) | flank r2 | sections x/y [deg] | n particles |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted([r for r in raw if r["kind"] == "pour"], key=lambda r: r["mu_rolling"]):
        rep = r["repose"]
        sec = rep.get("section_theta_deg") or {}
        out.append(f"| {r['id']} | {r['mu']:.2f} | {r['mu_rolling']:.2f} | "
                   f"{rep['theta_deg']:.2f} | {rep['tan_theta']:.3f} | {rep['r2']:.3f} | "
                   f"{sec.get('x', float('nan')):.1f} / {sec.get('y', float('nan')):.1f} | "
                   f"{r['n_particles']} |")
    out.append("")

    out.append("### Beverloo discharge cases\n")
    out.append("| case | D [mm] | D/d | W measured [kg/s] | +/- 1 sigma | W Beverloo [kg/s] | deviation | rho_b [kg/m3] | windows [kg/s] |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for p in c["C2_beverloo_curve"]["points"]:
        out.append(f"| {p['case']} | {p['D_m'] * 1e3:.1f} | {p['D_over_d']:.1f} | "
                   f"{p['W_measured_kg_s']:.4f} | {p['W_std_kg_s']:.4f} | "
                   f"{p['W_theory_kg_s']:.4f} | {p['rel_dev'] * 100:.1f} % | "
                   f"{p['bulk_density_kg_m3']:.0f} | "
                   + ", ".join(f"{w:.3f}" for w in p["windows_kg_s"]) + " |")
    out.append("")

    out.append("### Janssen pressure profile (plugged hold, mu_w = 0.40)\n")
    out.append("| z below surface [mm] | sigma_h [Pa] | sigma_v [Pa] | Janssen theory [Pa] (lambda from measured K) | hydrostatic [Pa] | in criterion |")
    out.append("|---|---|---|---|---|---|")
    j = [r for r in raw if r["id"] == "J_mu040"][0]
    hydro = j["hydrostatic_reference_Pa"]
    for i, row in enumerate(c["C3_janssen_pressure"]["profile"]):
        z = (j["bed_top_m"] - row["z_mid"]) * 1e3
        h = hydro[i] if i < len(hydro) else float("nan")
        lam = c["C3_janssen_pressure"]["lambda_theory_m"]
        pinf = c["C3_janssen_pressure"]["p_inf_theory_Pa"]
        pred = pinf * (1.0 - pow(2.718281828459045, -(j["bed_top_m"] - row["z_mid"]) / lam))
        inside = "yes" if (j["bed_top_m"] - row["z_mid"]) >= 4.0 * j["params"]["radius"] else "no (surface slice)"
        out.append(f"| {z:.1f} | {row['sigma_h_Pa']:.0f} | {row['sigma_v_Pa']:.0f} | "
                   f"{pred:.0f} | {h:.0f} | {inside} |")
    out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())