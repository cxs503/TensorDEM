"""Inject the run numbers into ``docs/breakage_pbm_dem_coupling.md`` (zero hand copy).

The committed run artefact ``benchmarks/dem_pbm_breakage/results.json`` is the
only source of every number written here: the script regenerates the block
between the ``<!-- INJECT:... -->`` markers, so the documentation cannot drift
from the run.

Usage
-----
    python benchmarks/dem_pbm_breakage/inject_doc_numbers.py \
        benchmarks/dem_pbm_breakage/results.json docs/breakage_pbm_dem_coupling.md
"""
from __future__ import annotations

import json
import sys

BEGIN = "<!-- INJECT:REWORK-RUN -->"
END = "<!-- /INJECT:REWORK-RUN -->"


def fmt(v):
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def build(res: dict) -> str:
    lines = [BEGIN, ""]
    meta = res.get("meta", {})
    lines.append(f"运行配置：破碎试验 `n={meta.get('crush_samples')}`，装配体颗粒数 "
                 f"`N={meta.get('n_particles')}`，加载时长 `{meta.get('t_end')} s`，"
                 f"随机种子 `{meta.get('seed')}`；总耗时 `{res.get('runtime_s', 0):.0f} s`；"
                 f"率剖面时间窗 `{res['criteria']['C4_dem_pbm_wasserstein']['rate_window_s']} s`"
                 f"（{res['criteria']['C4_dem_pbm_wasserstein']['n_rate_windows']} 窗）。")
    lines.append("")
    lines.append("### 判据表（程序化注入，源 `benchmarks/dem_pbm_breakage/results.json`）")
    lines.append("")
    lines.append("| 判据 | 实测值 | 阈值 | 判定 |")
    lines.append("|---|---|---|---|")
    for key, c in res["criteria"].items():
        lines.append(f"| {key} | {fmt(c['value'])} | {fmt(c['threshold'])} | "
                     f"{fmt(bool(c['passed']))} |")
    lines.append("")
    lines.append(f"全部判据通过：**{res.get('passed_all')}**；benchmark 全部对上："
                 f"**{res.get('benchmark_passed_all')}**")
    lines.append("")
    lines.append("### benchmark（参考值 vs 计算值）")
    lines.append("")
    lines.append("| 量 | 参考 ref | 计算 got | 相对误差 | 容差 | 判定 | 来源 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in res.get("benchmark", []):
        lines.append(f"| {row['quantity']} | {fmt(row['reference'])} | {fmt(row['computed'])} "
                     f"| {row['relative_error']:.3g} | {fmt(row['tolerance'])} | "
                     f"{fmt(bool(row['passed']))} | {row['source']} |")
    lines.append("")
    c4 = res["criteria"]["C4_dem_pbm_wasserstein"]
    lines.append("### C4 逐时刻（接触 DEM vs PBM；以及 PBM vs 颗粒分辨实现）")
    lines.append("")
    lines.append("| t [s] | DEM n | DEM 数均直径 [mm] | PBM n | PBM 数均直径 [mm] "
                 "| W1/d0 (DEM vs PBM) | W1/d0 (PBM vs Lagrangian) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t, wd in c4["per_time_normalised"].items():
        rec = res["cross_validation"]["contact_dem_records"][t]
        lag = c4["pbm_vs_lagrangian_per_time_normalised"][t]
        lines.append(f"| {t} | {rec['dem_n_particles']} | "
                     f"{rec['dem_mean_diameter_m'] * 1e3:.3f} | — | — | "
                     f"{wd:.4f} | {lag:.4f} |")
    lines.append("")
    lines.append("### 负控与质量账本")
    lines.append("")
    nc = res.get("negative_controls", {})
    for key in ("no_breakage_drift", "instant_breakage_psd_agreement",
                "instant_breakage_surviving_parent_ratio"):
        if key in nc:
            lines.append(f"- `{key}` = {fmt(nc[key])}")
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    res_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/dem_pbm_breakage/results.json"
    doc_path = sys.argv[2] if len(sys.argv) > 2 else "docs/breakage_pbm_dem_coupling.md"
    res = json.load(open(res_path))
    block = build(res)
    text = open(doc_path).read()
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n## 6. 返修轮（m14603）复跑结果\n\n" + block + "\n"
    open(doc_path, "w").write(text)
    print(f"injected {len(block)} chars from {res_path} into {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())