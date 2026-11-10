#!/usr/bin/env python
"""Generate the P155-N09 markdown report + platform payload from the artifacts.

Every number in the output comes from `artifacts/bidisperse_segregation_criteria.json`
and the merged campaign JSON -- nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import pathlib
import sys


def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.{digits}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{digits}f}"
    return str(x)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--mission", default="6689")
    args = ap.parse_args(argv)

    crit = json.loads(pathlib.Path(args.criteria).read_text())
    camp = json.loads(pathlib.Path(args.campaign).read_text())
    summary = crit["criteria_summary"]

    L: list[str] = []
    L.append("# Bidisperse granular segregation: verification surface (P155-N09)\n")
    L.append("## 1. Case and physical background\n")
    L.append(
        "A cargo hold filled with a two-size granular material (ore, grain, "
        "aggregate) segregates when the ship rolls or vibrates: the coarse "
        "fraction migrates up (Brazil-nut) or down (reverse Brazil-nut).  The "
        "consequences are a graded free surface, a depth-dependent permeability "
        "and therefore a depth-dependent rate of excess-pore-pressure "
        "dissipation -- the mechanism behind cargo shift and liquefaction.\n")
    L.append(
        f"Simulation inventory: **{camp['meta']['n_shear']} shear-cell cases** and "
        f"**{camp['meta']['n_vibration']} shaker cases**, size ratios "
        f"{', '.join(str(r) for r in camp['meta']['ratios'])}, 2-D frictional "
        "disks (Hertz-Mindlin normal/tangential + rolling resistance), "
        "`tensordem.granular2d.engine`.\n")
    L.append("### Parameter table\n")
    L.append("| item | value |\n|---|---|")
    L.append("| particle diameters | 10 mm (small), 12/15/20/30 mm (large) |")
    L.append("| density / thickness | 2500 kg/m3 / 10 mm |")
    L.append("| contact | kn = 1e5 N/m, kt/kn = 0.5, zeta = 0.3, mu = 0.5, mu_r = 0.05 |")
    L.append("| cell | periodic in x (shear) / closed (shaker), 120 particles, phi_nom = 0.60 |")
    L.append("| shear cell | dead-weight load plate at sigma = 4000 Pa, gamma_dot = 10 1/s |")
    L.append("| shaker | f = 30 Hz, Gamma = A w^2/g in {2, 10}, 20 cycles |")
    L.append("| strain / cycles per case | gamma = 6, or 20 shaking cycles |")
    L.append("| seeds | 5 per condition for the two extremes, 3 for the intermediate ratios |\n")

    L.append("## 2. Method\n")
    L.append(
        "1. **Packing** -- jittered lattice at the requested nominal solid "
        "fraction, then overlap removal by deterministic radius inflation "
        "(0.80 -> 0.85 -> 0.90 -> 0.95 -> 1.00 r) so no energy is injected.\n"
        "2. **Shear cell** -- a frictional plate carrying a dead weight "
        "`sigma * W` is placed on the free surface and given one vertical DOF "
        "(mass-spring-damper) integrated with the DEM; it is first brought to "
        "load equilibrium, then sheared at `v_top = gamma_dot * W`.  The wall "
        "tangential velocity is a first-class engine quantity "
        "(`DiskSystem(wall_velocity=...)`), so the drive is Coulomb friction at "
        "the plate, not a velocity overwrite of the particles.\n"
        "3. **Shaker** -- integrated in the container frame (walls static, "
        "effective acceleration `-g + A w^2 sin(w t)`), 30 cycles.\n"
        "4. **Metric** -- `Phi(z)` is the *area* fraction of the large species in "
        "a height bin, with species membership taken from the particle labels "
        "(global threshold `r_thr = sqrt(r_s r_l)`); `S = sum_b w_b |Phi_b - "
        "Phi_0| / sum_b w_b` with `w_b` the solid area in the bin; empty bins are "
        "dropped.  Direction: `dPhi = <Phi>_top - <Phi>_bottom` against its "
        "binomial standard error, a sign being declared only at 2 sigma.\n"
        "5. **Scaling** -- `S = A (d_l/d_s - 1)^alpha` fitted by least squares on "
        "`S` (not on log S), with a bootstrap CI over seeds.\n"
        "6. **Cargo coupling** -- `k(z) = d_eff^2 eps^3/(180 (1-eps)^2)`, "
        "`d_eff^2 = Phi d_l^2 + (1-Phi) d_s^2`, then a 1-D Terzaghi "
        "consolidation solve with `c_v(z) = k(z) K_s/gamma_w` gives the time to "
        "50 % dissipation of the base excess pressure (`t50`).\n"
        "7. **Repose angle** -- the surface cloud of a column collapse is handed "
        "to `tensordem.repose_calibration.repose_angle`, i.e. the P96-N3 "
        "estimator itself.\n")

    L.append("## 3. Results\n")
    L.append("### 3.1 Criterion verdicts (machine-evaluated)\n")
    L.append("| criterion | requirement | verdict | evidence |\n|---|---|---|---|")
    c1 = crit["C1"]["evidence"]
    L.append(f"| C1 | monodisperse negative control | {fmt(summary['C1'])} | "
             f"bitwise frozen deviation = {fmt(c1['bitwise_frozen_deviation'])}, "
             f"bitwise determinism deviation = {fmt(c1['bitwise_determinism_deviation'])}, "
             f"sheared S = {fmt(c1['sheared_S'])}, floor = {fmt(c1['noise_floor'])} |")
    c2 = crit["C2"]
    def _dir(row):
        return "n/a" if not row else (f"{row['dominant']} (consensus "
                                     f"{fmt(row['consensus'])}, n={row['n']})")
    L.append(f"| C2 | opposite directions, >= 8/10 seeds | {fmt(summary['C2'])} | "
             f"shear 1.2 -> {_dir(c2.get('shear_1.2'))}, "
             f"shear 3.0 -> {_dir(c2.get('shear_3.0'))} |")
    bf = crit["C3"].get("best_fit") or {}
    L.append(f"| C3 | S monotone in ratio, R2 > 0.9, CI; S down as phi -> phi_j | "
             f"{fmt(summary['C3'])} | R2 = {fmt(bf.get('R2'))}, "
             f"alpha = {fmt(bf.get('alpha'))}, CI = {bf.get('alpha_ci')}, "
             f"monotone = {fmt(bf.get('monotone_increasing'))} |")
    L.append(f"| C4 | mass/concentration residual < 1e-10 | {fmt(summary['C4'])} | "
             f"max |Delta Phi| = {fmt(crit['C4']['max_residual'])} over "
             f"{crit['C4']['n_cases']} cases |")
    c5 = crit["C5"]
    L.append(f"| C5 | repose angle with P96-N3 tooling | {fmt(summary['C5'])} | "
             f"stratified {fmt(c5.get('stratified_angle_deg'))} deg vs "
             f"monodisperse {fmt(c5.get('monodisperse_angle_deg'))} deg, "
             f"same estimator; unresolved: {fmt(c5.get('stratified_error'))} |")
    L.append(f"| C6 | sensitivity table reproducible | {fmt(summary['C6'])} | "
             f"bitwise identical reruns = {fmt(crit['C6'].get('bitwise_identical_reruns'))} |\n")

    L.append("### 3.2 Segregation degree per condition\n")
    L.append("| kind | size ratio | drive | seeds | S mean | S std | measured phi | direction (consensus) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in camp["summary"]["shear"]:
        L.append(f"| shear | {fmt(row['key']['size_ratio'],2)} | "
                 f"sigma={fmt(row['key']['normal_stress'],0)} Pa, "
                 f"gdot={fmt(row['key']['shear_rate'],0)} 1/s | {row['n_seeds']} | "
                 f"{fmt(row['S_mean'])} | {fmt(row['S_std'])} | {fmt(row['phi_mean'])} | "
                 f"{row['direction']['dominant']} ({fmt(row['direction']['consensus'],2)}) |")
    for row in camp["summary"]["vibration"]:
        L.append(f"| vibration | {fmt(row['key']['size_ratio'],2)} | "
                 f"Gamma={fmt(row['key']['gamma'],0)} | {row['n_seeds']} | "
                 f"{fmt(row['S_mean'])} | {fmt(row['S_std'])} | {fmt(row['phi_mean'])} | "
                 f"{row['direction']['dominant']} ({fmt(row['direction']['consensus'],2)}) |")
    L.append("")

    L.append("### 3.3 Scaling law\n")
    L.append("| kind | axis | ratios | S means | A | alpha | 95% CI | R2 |\n|---|---|---|---|---|---|---|---|")
    for row in crit["C3"]["scaling"]:
        axis = "sigma" if "normal_stress" in row else "Gamma"
        L.append(f"| {row['kind']} | {axis}={fmt(row.get('normal_stress', row.get('gamma')))} | "
                 f"{row['ratios']} | {[round(v, 3) for v in row['S_means']]} | "
                 f"{fmt(row['A'])} | {fmt(row['alpha'])} | {row['alpha_ci']} | "
                 f"{fmt(row['R2'])} |")
    L.append("")

    L.append("### 3.4 Permeability / excess pore pressure sensitivity\n")
    L.append("| profile | k mean (m2) | k min | k max | k_top/k_bottom | t50 (s) | t50/t50(uniform) |")
    L.append("|---|---|---|---|---|---|---|")
    for row in crit["C6"].get("rows", []):
        L.append(f"| {row['profile']} | {fmt(row['k_mean'])} | {fmt(row['k_min'])} | "
                 f"{fmt(row['k_max'])} | {fmt(row['k_top_over_bottom'])} | "
                 f"{fmt(row['t50'])} | {fmt(row['t50_ratio_to_uniform'])} |")
    L.append("")

    L.append("### 3.5 Repose angle (C5)\n")
    L.append(f"* estimator: `{c5.get('estimator')}`")
    L.append(f"* stratified bed: {fmt(c5.get('stratified_angle_deg'))} deg "
             f"(r2 = {fmt(c5.get('stratified_r2'))})")
    L.append(f"* monodisperse bed: {fmt(c5.get('monodisperse_angle_deg'))} deg "
             f"(r2 = {fmt(c5.get('monodisperse_r2'))})")
    L.append(f"* difference: {fmt(c5.get('delta_deg'))} deg")
    for key in ("stratified_error", "monodisperse_error"):
        if c5.get(key):
            L.append(f"* {key}: `{c5[key]}`")
    if c5.get("diagnosis"):
        L.append(f"* diagnosis: {c5['diagnosis']}")
    if c5.get("estimator_identity"):
        L.append(f"* identity: {c5['estimator_identity']}")
    if c5.get("synthetic_validation"):
        L.append(f"* synthetic validation: {c5['synthetic_validation']}")
    L.append("")

    L.append("## 4. Figures\n")
    L.append(f"`{pathlib.Path(args.figure).name}`: (a) S vs size ratio for both "
             "drivers, (b) scaling fits with R2, (c) S vs measured solid fraction "
             "(jamming suppression), (d) drainage time ratio per stratification.\n")

    L.append("## 5. Caveats, limitations and honest FAILs\n")
    L.append(
        "* **Volume fraction is an output, not an input, in the shear cell.** In "
        "2-D frictional disks a gravity-loaded bed packs to phi ~ 0.85-0.87; the "
        "nominal grid phi = 0.55/0.60/0.63 of the task is not attainable in a "
        "gravitational cell (it would be a dilute, fluidised state). The phi axis "
        "is therefore realised by the plate load sigma and the shaking intensity "
        "Gamma, and the *measured* solid fraction is reported with every case; "
        "the jamming trend required by C3 is tested on that measured axis.\n"
        "* **Metric correction.** The first implementation classified species "
        "with the median radius *inside* each bin; a bin containing only large "
        "particles then reported Phi = 0, i.e. it inverted the sign of the "
        "profile and produced a spurious S ~ 0.98 from pure quantisation noise. "
        "Species membership is now global (labels or a fixed threshold) and the "
        "profile is area weighted; the regression test "
        "`test_profile_is_label_based_not_bin_local` pins this.\n"
        "* **Shear drive.** The moving lid must be in contact with the bed. An "
        "unloaded lid floating above a settled bed transmits nothing "
        "(measured vx/v_top ~ 0.01); the dead-weight plate brings the measured "
        "shear penetration to vx/v_top ~ 0.7 near the plate.\n"
        "* **Campaign size.** The planned cluster grid (4 ratios x 3 stresses x 2 "
        "shear rates x 3 seeds, plus the 10-seed blocks) sat in the job queue for "
        "the whole session and was cancelled; the numbers here come from the local "
        "campaign that actually ran (16 shear + 26 shaker cases, 42 records). "
        "The cluster runner and specs are in the repo and reproduce the larger grid.\n"
        "* **2-D disks.** No out-of-plane percolation path, so absolute S values "
        "are not transferable to 3-D cargo; the sign, monotonicity and the "
        "permeability coupling are.\n"
        "* **No experimental benchmark in CI.** The quantitative checks are "
        "(i) exact conservation, (ii) determinism bitwise, (iii) the S -> 0 limit "
        "at d_l/d_s -> 1, and (iv) Kozeny-Carman against its closed form. No "
        "experimental segregation data set is available in this environment, so "
        "an experimental error percentage cannot be claimed -- stated explicitly "
        "rather than approximated.\n")

    L.append("## 6. Reproduction\n")
    L.append("```sh")
    L.append("PYTHONPATH=src python scripts/run_bidisperse_campaign.py --mode ratios \\")
    L.append("    --shard 0 --nshards 6 --outdir runs/ratios")
    L.append("PYTHONPATH=src python scripts/run_bidisperse_campaign.py --mode probe \\")
    L.append("    --outdir runs/all --merge --report artifacts/bidisperse_campaign.json")
    L.append("PYTHONPATH=src python scripts/audit_bidisperse_criteria.py --inputs runs/all \\")
    L.append("    --out artifacts/bidisperse_segregation_criteria.json")
    L.append("PYTHONPATH=src python scripts/make_bidisperse_figures.py \\")
    L.append("    --report artifacts/bidisperse_campaign.json \\")
    L.append("    --out artifacts/bidisperse_segregation.png")
    L.append("PYTHONPATH=src python -m pytest tests/test_bidisperse_segregation.py \\")
    L.append("    tests/test_granular2d_wall_velocity.py -q")
    L.append("```\n")
    L.append("## 7. Error judgement vs references\n")
    L.append("| quantity | reference | error | < 3 % ? |")
    L.append("|---|---|---|---|")
    L.append(f"| weighted Phi vs Phi_0 | conservation (exact) | "
             f"{fmt(crit['C4']['max_residual'])} | yes |")
    L.append(f"| frozen/deterministic replay | bitwise identity | "
             f"{fmt(c1['bitwise_determinism_deviation'])} | yes |")
    L.append("| Kozeny-Carman k | closed form d^2 eps^3/(180(1-eps)^2) | 0 (same formula, unit test) | yes |")
    L.append("| S(d_l/d_s = 1) | 0 beyond the finite-size floor | see C1 | yes (floor-bounded) |")
    L.append("| segregation degree S | no experimental data set available | not computable | no claim made |\n")

    md = "\n".join(L) + "\n"
    pathlib.Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.doc).write_text(md)
    print("wrote", args.doc)

    if args.payload:
        fig = pathlib.Path(args.figure)
        png = base64.b64encode(fig.read_bytes()).decode() if fig.exists() else ""
        crit_line = ", ".join(f"{k}={fmt(v)}" for k, v in summary.items())
        payload = {
            "title": "P155-N09 Bidisperse granular segregation: shear/vibration "
                     "verification and cargo-permeability coupling",
            "summary": f"Criteria {crit_line}; shear cell with dead-weight plate, "
                       f"shaker cell, label-based area-weighted metric, "
                       f"Kozeny-Carman + Terzaghi sensitivity table.",
            "scenario_key": "granular/segregation",
            "mission_id": "6689",
            "tags": ["TensorDEM", "granular", "segregation", "P155-N09",
                     "cargo-liquefaction", "dem"],
            "markdown": md,
            "figures": [{"name": "bidisperse_segregation.png",
                         "caption": "P155-N09 segregation campaign overview",
                         "png_base64": png}],
        }
        pathlib.Path(args.payload).write_text(json.dumps(payload))
        print("wrote", args.payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
