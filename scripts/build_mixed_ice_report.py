#!/usr/bin/env python3
"""Build /tmp/report.json (platform report payload) from the archived artifact."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "mixed_ice_dem_field"
DOC = ROOT / "docs" / "mixed_ice_dem_field.md"

payload = json.loads((ART / "campaign.json").read_text())
a = payload["criterion_a"]
cb = payload["criterion_b"]
prim = cb["primary"]
c = payload["criterion_c"]
d = payload["criterion_d"]
ship = payload["ship"]
impact = payload["aging_law_impact"]

figures = []
captions = {
    "fig_mixture_rule.png":
        "Figure 1. DEM section resistance vs the linear mixture rule (1) and the "
        "harmonic/Reuss mixture, with the mixture-rule error and its 95 % CI "
        "(balanced layout, 5 m transit, 10 realisations per fraction).",
    "fig_age_strength.png":
        "Figure 2. Age-dependent material law: sigma_c (extended, temperature "
        "amplified), sigma_f (Timco & O'Brien) and E, normalised to the "
        "first-year state; the criterion (a) band [1.5, 3.0] and the saturation "
        "asymptote are shaded/marked; right: sensitivity to the temperature "
        "exponent m.",
    "fig_ship_scale.png":
        "Figure 3. Ship-scale resistance vs MY area fraction (Lindqvist, Riska) "
        "and the decomposition of the MY/FY increment into thickness, material "
        "and interaction at the class means.",
    "fig_class_field.png":
        "Figure 4. Balanced class map of one simulated field (orange MY, blue FY) "
        "with the 5 m hull transit marked, and the deviation from the linear "
        "mixture rule against the MY floe length crossed (all layouts).",
}
for name, caption in captions.items():
    path = ART / name
    figures.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    })

summary = (
    "TensorDEM: 冰龄依赖的多年冰强度律扩展（σ_c 温度放大，MY/FY 抗压强度比 1.46→2.08，"
    "落文献带 [1.5,3.0]，单调饱和）+ 在真实非均匀冰场上直接检验线性混合律。"
    "12 个构型 11 个在 ±15% 内、偏差系统性为负（约 −10%），唯一越界为「近纯 MY+薄 FY 缝」"
    "−22.7%±9.2%（最弱环）。船体尺度 0→60% MY：0.990→5.044 MN（+409%，Lindqvist，3 m/s），"
    "厚度贡献 54%。"
)

markdown = (DOC.read_text()).replace(
    "Artifact:", "Platform report payload: mission_id 7842.\n\nArtifact:")

report = {
    "title": "多年冰/混合冰场阻力模型扩展：冰龄强度律与线性混合律的直接 DEM 检验"
             "（p54w2-N01 / TensorDEM）",
    "summary": summary,
    "scenario_key": "ship-performance/ice/mixed-ice-resistance-tensordem",
    "mission_id": 7842,
    "tags": ["TensorDEM", "multi-year-ice", "mixed-ice-field", "Lindqvist",
             "DEM", "resistance", "criterion-a", "criterion-b", "honest-fail",
             "p54w2-N01"],
    "markdown": markdown,
    "figures": figures,
    "metadata": {
        "node": "p54w2-N01",
        "criteria": {
            "a_sigma_c_ratio": {
                "value": a["ratio_my_over_fy"], "band": a["band"],
                "pass": a["criterion_a_pass"],
                "artifact": "criterion_a.ratio_my_over_fy",
            },
            "b_mixture_vs_dem": {
                "worst_bias_pct": prim["max_abs_linear_error_pct"],
                "worst_ci95_pct": prim["worst_ci95_pct"],
                "configs_in_band": f"{prim['n_configs_in_band']}/{prim['n_configs_total']}",
                "pass_universal": prim["criterion_b_pass"],
                "pass_coarse_floe": prim["criterion_b_pass_coarse_only"],
                "outlier": {"regime": "random layout, traversed p=0.88",
                            "error_pct": cb["random_layout"]["rows"][4]["linear_error_pct"],
                            "ci95_pct": cb["random_layout"]["rows"][4]["mean_error_ci95_pct"]},
                "artifact": "criterion_b.primary",
            },
            "c_monotone_saturating": {
                "strictly_increasing": c["strictly_increasing"],
                "sigma_c_saturation_mpa": c["saturation_sigma_c_pa"] / 1e6,
                "residual_pct": c["residual_at_last_age_pct"],
                "pass": c["criterion_c_pass"],
                "artifact": "criterion_c",
            },
            "d_fy_degeneration": {
                "relative_difference": d["relative_difference"],
                "pass": d["criterion_d_pass"],
                "artifact": "criterion_d",
            },
        },
        "headline": {
            "my_fraction_from": 0.0, "my_fraction_to": 0.6,
            "ship_p0_MN": ship["summary"]["lindqvist"]["r_at_p0_N"] / 1e6,
            "ship_p60_MN": ship["summary"]["lindqvist"]["r_at_target_N"] / 1e6,
            "ship_change_pct": ship["summary"]["lindqvist"]["relative_change_pct"],
            "riska_change_pct": ship["summary"]["riska"]["relative_change_pct"],
            "dem_section_change_pct": prim["my_fraction_effect"].get(
                "relative_change_measured_pct"),
            "decomposition_share_pct": {
                "thickness": ship["decomposition"]["thickness_share_pct"],
                "material": ship["decomposition"]["material_share_pct"],
                "interaction": ship["decomposition"]["interaction_share_pct"],
            },
        },
        "aging_law_extension": {
            "temperature_exponent": 0.35,
            "band": [0.20, 0.50],
            "sigma_c_my_old_mpa": impact["sigma_c_my_old_pa"] / 1e6,
            "sigma_c_my_new_mpa": impact["sigma_c_my_new_pa"] / 1e6,
            "quadrature_p60_old_MN": impact["quadrature_r_at_60_old_MN"],
            "quadrature_p60_new_MN": impact["quadrature_r_at_60_new_MN"],
        },
        "artifacts": {
            "doc": "docs/mixed_ice_dem_field.md",
            "json": "artifacts/mixed_ice_dem_field/campaign.json",
            "artifact_sha256": payload["artifact_sha256"],
            "aging_ledger_sha256": payload["aging_ledger_sha256"],
            "engine": "TensorDEM @ branch agent/m7842",
            "runtime_s": payload.get("runtime_s"),
        },
        "limits": [
            "L1 plane-section lattice has one thickness: no thickness step-up "
            "inside a field (thickness mixing is anchor-only, declared).",
            "L2 5 m transit = 5 thicknesses; field-realistic regime is 'one MY "
            "floe edge per transit'; broken-bond fraction 0.7 after 18 m.",
            "L3 interface bonds take the mean of the two classes (declared).",
            "L4 crushing-dominated section (92 % of R); bending nearly absent.",
            "L8 the absolute ship resistance is not validated here.",
        ],
    },
}
Path("/tmp/report.json").write_text(json.dumps(report))
print(json.dumps({
    "title": report["title"],
    "figures": [f["name"] for f in figures],
    "bytes": Path("/tmp/report.json").stat().st_size,
}, indent=2))