"""Append the machine-generated result tables of the breakage-PBM study to the doc.

Numbers are read from ``artifacts/breakage_pbm/summary.json`` (written by
``scripts/run_breakage_pbm_verification.py``); nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import os
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
    lines += ["", f"全部判据通过：**{s['passed_all']}**", ""]

    cr = s.get("crushing", {})
    if cr:
        w = cr["weibull"]
        lines += ["### 4.2 单颗粒破碎（C1）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| 破碎试验数 | {cr['n_tests']} |",
                  f"| 子球数 / 键数 | {cr['n_sub']} / — |",
                  f"| 峰值破碎力中位数 | {cr['median_peak_force_N']:.1f} N |",
                  f"| 破碎应力中位数 | {cr['median_strength_Pa']/1e6:.2f} MPa |",
                  f"| Weibull 模数 m | {w['m']:.3f} |",
                  f"| 尺度 σ0 | {w['sigma_0']/1e6:.3f} MPa |",
                  f"| CDF 拟合 R² | {w['r2_cdf']:.5f} |",
                  f"| KS 统计量 | {w['ks_statistic']:.4f} |", ""]

    d = s.get("daughters", {})
    if d:
        lines += ["### 4.3 子代分布（C2）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| 子代样本数 | {d['n_fragments']} |",
                  f"| 每事件平均碎块数 | {d['mean_fragments_per_event']:.3f} |",
                  f"| 最大碎块占比中位数 | {d['largest_fragment_fraction_median']:.3f} |",
                  f"| 子球分辨率 | {d['n_sub_resolution']} |",
                  f"| 对数正态 σ_ln（矩估计） | {d['lognormal_sigma_ln']:.4f} |",
                  f"| KS(对数正态) p | {d['ks_lognormal']['pvalue']:.4g} |",
                  f"| KS(对数正态) 自举 p | {d['ks_lognormal_bootstrap_pvalue']:.4g} |",
                  f"| KS(均匀) p | {d['ks_uniform']['pvalue']:.4g} |", ""]

    a = s.get("dpbm_analytic", {})
    if a:
        lines += ["### 4.4 群体平衡 vs 闭式解（C3）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| S | {a['rate_1_per_s']:g} 1/s |",
                  f"| τ = S·t | {a['tau']:g} |",
                  f"| 单元数 | {a['n_sections']} |",
                  f"| 累积质量分数最大偏差 | {a['cumulative_max_abs_deviation']:.4f} |",
                  f"| 累积 L1 | {a['cumulative_l1']:.4e} |",
                  f"| 分箱密度 L1（参考） | {a['l1_relative_mass_density']:.4f} |",
                  f"| 数密度矩比 | {a['number_moment_ratio_to_analytic']:.4f} |",
                  f"| 质量矩比 | {a['mass_moment_ratio_to_analytic']:.4f} |",
                  f"| 子代矩阵质量亏损 | {a['daughter_mass_defect']:.2e} |", ""]

    cv = s.get("cross_validation", {})
    if cv:
        lines += ["### 4.5 DEM–PBM 交叉验证（C4）", "",
                  f"κ（有效应力比，首个破碎事件前测得）= {cv['kappa_effective_stress_ratio']:.3f}；"
                  f"破碎事件 {cv['n_events']} 次；末态颗粒数 {cv['n_particles_end']}；"
                  f"首次破碎时刻 {cv['first_break_time_s']:.4g} s。", "",
                  "| t [s] | DEM 颗粒数 | DEM 平均直径 [mm] | PBM 平均直径 [mm] | "
                  "W1 [mm] | W1 / d0 |", "|---|---|---|---|---|---|"]
        for t, rec in cv["records"].items():
            lines.append(
                f"| {t} | {rec['dem_n_particles']} | {rec['dem_mean_diameter_m']*1e3:.3f} | "
                f"{rec['pbm_mean_diameter_m']*1e3:.3f} | "
                f"{rec['wasserstein_raw_m']*1e3:.3f} | {rec['wasserstein_normalised']:.4f} |")
        lines.append("")

    nc = s.get("negative_controls", {})
    if nc:
        lines += ["### 4.6 负控（C6）", "",
                  "| 量 | 值 |", "|---|---|",
                  f"| σ0→∞ PSD 漂移 | {nc['no_breakage_drift']:.2e} |",
                  f"| σ0→0 与子代分布累积一致度 | {nc['instant_breakage_psd_agreement']:.4f} |",
                  f"| σ0→0 残余母颗粒占比 | {nc['instant_breakage_surviving_parent_ratio']:.2e} |", ""]

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