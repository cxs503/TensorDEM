"""Append the machine-generated result tables of the breakage-PBM study to the doc.

Numbers are read from ``artifacts/breakage_pbm/summary.json`` (written by
``scripts/run_breakage_pbm_verification.py``); nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import sys

MARKER = "## 4. 结果"


def main() -> int:
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/breakage_pbm/summary.json"
    doc_path = sys.argv[2] if len(sys.argv) > 2 else "docs/breakage_pbm_dem_coupling.md"
    with open(summary_path) as fh:
        s = json.load(fh)

    lines = [MARKER, "",
             f"运行配置：破碎试验 `n={s['meta']['crush_samples']}`，装配体颗粒数 "
             f"`N={s['meta']['n_particles']}`，加载时长 `{s['meta']['t_end']:g} s`，"
             f"随机种子 `{s['meta']['seed']}`；总耗时 `{s.get('runtime_s', 0):.0f} s`。",
             "", "### 4.1 判据表", "",
             "| 判据 | 实测值 | 阈值 | 判定 |", "|---|---|---|---|"]
    for key, crit in s["criteria"].items():
        lines.append(f"| {key} | {crit['value']:.6g} | {crit['threshold']:.6g} | "
                     f"{'PASS' if crit['passed'] else 'FAIL'} |")
    lines += ["", f"全部判据通过：**{s['passed_all']}**",
              f"pytest 用例数（本文件 + 本验证面）：**{s.get('n_tests', 'see C7 table')}**", ""]

    cr = s.get("crushing", {})
    if cr:
        w = cr["weibull"]
        lines += ["### 4.2 单颗粒破碎强度（C1）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| 破碎试验数 | {cr['n_tests']} |",
                  f"| 子球数 | {cr['n_sub']} |",
                  f"| 峰值破碎力中位数 | {cr['median_peak_force_N']:.1f} N |",
                  f"| 破碎应力中位数 | {cr['median_strength_Pa'] / 1e6:.2f} MPa |",
                  f"| Weibull 模数 m | {w['m']:.3f} |",
                  f"| 尺度 σ0 | {w['sigma_0'] / 1e6:.3f} MPa |",
                  f"| CDF 拟合 R² | {w['r2_cdf']:.5f} |", ""]

    d = s.get("daughters", {})
    if d:
        kb = d["ks_beta_heldout"]
        lines += ["### 4.3 子代尺寸分布（C2）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| 子代样本数 | {d['n_fragments']} |",
                  f"| 每事件平均碎块数 | {d['mean_fragments_per_event']:.3f} |",
                  f"| 最大碎块占比中位数 | {d['largest_fragment_fraction_median']:.3f} |",
                  f"| 分辨率（子球） | {d['n_sub_resolution']} |",
                  f"| 拟合定律 | {d['fitted_law']} |",
                  f"| 权重 / 分量 (a, b) | "
                  + "; ".join(f"{w:.3f}·Beta({a:.3f},{b:.3f})" for w, (a, b)
                              in zip(d['fitted_weights'], d['fitted_params'])) + " |",
                  f"| 平均子代数（由 binned 律定） | {d['beta_mean_daughters']:.3f} |",
                  f"| 拟合样本 / 检验样本 | {kb['n_a']} / {kb['n_b']} |",
                  f"| KS 统计量（留出样本） | {kb['statistic']:.4f} |",
                  f"| KS p（留出样本，判据 C2） | {kb['pvalue']:.4g} |",
                  f"| KS p（对数正态，同协议） | {d['ks_lognormal_same_protocol']['pvalue']:.4g} |",
                  f"| KS p（均匀，同协议） | {d['ks_uniform_same_protocol']['pvalue']:.4g} |", ""]

    a = s.get("dpbm_analytic", {})
    if a:
        lines += ["### 4.4 分段 PBM vs 精确解（C3）", "",
                  f"S = {a['rate_1_per_s']:g} 1/s，t = {a['time_s']:g} s，τ = {a['tau']:g}；"
                  f"参考解 = {a['reference']}。", "",
                  "| 单元数 | 累积质量分数最大偏差 | 累积 L1 | 质量比 | 子代算子质量亏损 |",
                  "|---|---|---|---|---|"]
        for r in a["resolution_study"]:
            lines.append(f"| {r['n_sections']} | {r['max_cumulative_mass_deviation']:.4f} | "
                         f"{r['cumulative_l1']:.5f} | {r['mass_ratio']:.6f} | "
                         f"{r['daughter_operator_mass_defect']:.2e} |")
        lines.append("")

    asm = s.get("assembly", {})
    cv = s.get("cross_validation", {})
    if cv:
        lines += ["### 4.5 DEM–PBM 交叉验证（C4）", ""]
        if asm:
            lines += [f"接触 DEM 装配体：`N={asm['n_particles_initial']}` 颗粒，"
                      f"`{asm['n_events']}` 次破碎事件，末态颗粒数 {asm['n_particles_end']}，"
                      f"首次破碎 t = {asm['first_break_time_s']:.3f} s，"
                      f"κ = {asm['kappa_effective_stress_ratio']:.3f}，"
                      f"σ_max = {asm['stress_max_Pa'] / 1e6:.1f} MPa。", ""]
        lines += [f"PBM 单元数 {cv['n_sections']}，子代算子质量亏损 "
                  f"{cv['daughter_operator_mass_defect']:.2e}，归一化长度 d0 = "
                  f"{cv['normalisation_m'] * 1e3:.3f} mm，"
                  f"率剖面最大 S = {cv['rate_profile_max_1_per_s']:.4g} 1/s。", "",
                  "| t [s] | PBM 数均直径 [mm] | Lagrangian 颗粒数 | Lagrangian 数均直径 [mm] | "
                  "W1 [mm] | W1/d0（判据） | 接触 DEM W1/d0（诊断） |",
                  "|---|---|---|---|---|---|---|"]
        for t, rec in cv["records"].items():
            diag = cv["contact_dem_records"][t]
            lines.append(
                f"| {t} | {rec['pbm_number_mean_diameter_m'] * 1e3:.3f} | "
                f"{rec['lagrangian_n_particles']} | "
                f"{rec['lagrangian_mean_diameter_m'] * 1e3:.3f} | "
                f"{rec['wasserstein_m'] * 1e3:.4f} | {rec['wasserstein_normalised']:.4f} | "
                f"{diag['wasserstein_normalised']:.4f} |")
        lines.append("")

    nc = s.get("negative_controls", {})
    if nc:
        lines += ["### 4.6 负控（C6）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| σ0→∞ PSD 漂移 | {nc['no_breakage_drift']:.2e} |",
                  f"| σ0→∞ 最大 hazard | {nc['strong_strength_max_hazard_per_s']:.2e} 1/s |",
                  f"| σ0→0 与子代分布累积一致度 | {nc['instant_breakage_psd_agreement']:.4f} |",
                  f"| σ0→0 残余母颗粒占比 | {nc['instant_breakage_surviving_parent_ratio']:.2e} |",
                  f"| σ0→0 最大 hazard | {nc['weak_strength_max_hazard_per_s']:.2e} 1/s |", ""]

    with open(doc_path) as fh:
        doc = fh.read()
    head = doc.split(MARKER)[0]
    tail = doc.split("## 5. 局限与未完成项")[-1] if "## 5. 局限与未完成项" in doc else ""
    out = head + "\n".join(lines) + "\n## 5. 局限与未完成项" + tail
    with open(doc_path, "w") as fh:
        fh.write(out)
    print(f"wrote {doc_path}: {len(lines)} generated lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())