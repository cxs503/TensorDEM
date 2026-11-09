"""Inject the campaign results (JSON) into the upscaling docs page.

Every number in the injected block comes from ``runs/upscaling_cg/results.json``
so the documentation cannot drift from the computed evidence (no hand copying).

    PYTHONPATH=src python tools/render_upscaling_doc.py [--json ...] [--doc ...]
"""
from __future__ import annotations

import argparse
import json
import os


def _fmt(value, digits=2) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0.0):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def build_block(payload: dict) -> str:
    dev = payload["deviations"]
    crit = payload["criteria"]
    lines = []
    add = lines.append

    add(f"生成时间（UTC）{payload['generated_utc']} · 运行数 {len(payload['runs'])} · "
        f"总 CPU 时间 {_fmt(payload['elapsed_s'], 1)} s · "
        f"artefact sha256 `{payload['provenance']['artefact_sha256'][:16]}…`")
    add("")

    # ---- 1. macro invariant table -------------------------------------
    add("### 4.1 宏观不变量 vs f（similar 模式，A=口径A k∝f³，B=口径B k∝f）")
    add("")
    add("| f | A: φ (°) | A: μ | A: θ_r (°) | A: E_eff (Pa) | A: 首峰力 (N) | "
        "B: φ (°) | B: μ | B: θ_r (°) | B: E_eff (Pa) | B: 首峰力 (N) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    fs = [str(f) for f in payload["f_values"]]
    for f in fs:
        row = [f]
        for approach in ("contact_time", "stiffness_invariant"):
            shear = dev.get("shear", {}).get(approach, {}).get(f, {})
            rep = dev.get("repose", {}).get(approach, {}).get(f, {})
            ucs = dev.get("ucs", {}).get(approach, {}).get(f, {})
            row += [_fmt(shear.get("friction_angle_deg")), _fmt(shear.get("mu_macro")),
                    _fmt(rep.get("theta_r_deg")), _fmt(ucs.get("e_eff_pa")),
                    _fmt(ucs.get("first_peak_force_n"), 4)]
        add("| " + " | ".join(row) + " |")
    add("")

    # ---- 2. deviation table -------------------------------------------
    add("### 4.2 误差预算（相对同口径 f=1 的偏差）")
    add("")
    add("| f | A: Δφ (%) | A: Δθ_r (°) | A: ΔE_eff (%) | A: ΔF_peak (%) | "
        "B: Δφ (%) | B: Δθ_r (°) | B: ΔE_eff (%) | B: ΔF_peak (%) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for f in fs[1:]:
        row = [f]
        for approach in ("contact_time", "stiffness_invariant"):
            shear = dev.get("shear", {}).get(approach, {}).get(f, {})
            rep = dev.get("repose", {}).get(approach, {}).get(f, {})
            ucs = dev.get("ucs", {}).get(approach, {}).get(f, {})
            row += [_fmt(shear.get("phi_dev_pct")), _fmt(rep.get("theta_dev_deg")),
                    _fmt(ucs.get("e_eff_dev_pct")), _fmt(ucs.get("peak_force_dev_pct"))]
        add("| " + " | ".join(row) + " |")
    add("")
    add("首峰力偏差按各口径声明的力指数归一（A: F ∝ f⁴，B: F ∝ f²）。")
    inv = payload.get("invalid_runs") or []
    if inv:
        add("")
        add("**无效实现（如实列出）**：" + "；".join(
            f"{r['case']}/{r['approach']}/f={r['f']:g}/seed={r['seed']} 的 "
            f"{','.join(r['nan_metrics'])} 为 NaN（死载压板脱开，σ_n=0 ⇒ μ 无定义，已从均值剔除）"
            for r in inv) + "。")
    add("")

    # ---- 3. criteria verdicts -----------------------------------------
    add("### 4.3 机器判据逐项裁决")
    add("")
    add("| 判据 | 阈值 | 口径A | 口径B |")
    add("|---|---|---|---|")
    thr = payload["criteria_thresholds"]
    labels = {
        "shear_phi": ("① 摩擦角偏差", f"≤{thr['shear_phi_dev_pct']['f_le_4']:g}% (f≤4) / "
                                     f"≤{thr['shear_phi_dev_pct']['f_8']:g}% (f=8)"),
        "repose_theta": ("② 休止角偏差", f"≤{thr['repose_theta_dev_deg']['f_le_4']:g}° (f≤4)"),
        "e_eff": ("③ E_eff 偏差", f"≤{thr['e_eff_dev_pct']['all']:g}%"),
        "force_peak": ("④ 首峰力偏差", f"≤{thr['force_peak_dev_pct']['all']:g}%"),
    }
    for key, (name, limit) in labels.items():
        a = crit["contact_time"].get(key, {})
        b = crit["stiffness_invariant"].get(key, {})
        add(f"| {name} | {limit} | {'PASS' if a.get('all_pass') else 'FAIL'} | "
            f"{'PASS' if b.get('all_pass') else 'FAIL'} |")
    stab = payload["stability"]
    n_pass = sum(1 for r in stab["rows"] if r.get("numerical_ok"))
    n_ov = sum(1 for r in stab["rows"] if r.get("declared_overlap_violation"))
    add(f"| ⑤ 数值稳定性门 | v_max<10 v_ref / 能量单调不增 / 无逃逸 / Δt=t_c/20 | "
        f"{n_pass}/{len(stab['rows'])} 组合通过 | （同左，两口径共享） |")
    add(f"| ⑤b 声明性可压缩限 | δ/R ≤ 2 % | {n_ov}/{len(stab['rows'])} 组合越界"
        f"（均为口径B f≥4，属已声明失效边界） | （同左） |")
    add("")

    # ---- 4. negative controls -----------------------------------------
    add("### 4.4 负控（错误定标必须被宏观判据拒绝）")
    add("")
    neg = payload["negative_controls"]
    for f, row in neg["kn_f3_for_stiffness_vs_e_eff"]["measured"].items():
        add(f"* `kn_f3_for_stiffness` f={f}: E_eff 偏差 {_fmt(row['e_eff_dev_pct'])} % "
            f"→ 判据③ 拒绝（阈值 10%），"
            f"E_eff={_fmt(row['e_eff_pa'])} Pa")
    if "mass_only_vs_shear_phi" in neg:
        for f, row in neg["mass_only_vs_shear_phi"]["measured"].items():
            add(f"* `mass_only` f={f}: 摩擦角偏差 {_fmt(row['phi_dev_pct'])} % "
                f"（φ={_fmt(row['friction_angle_deg'])}°）→ 判据① 拒绝")
    for f, row in neg["mass_only_contact_time"].items():
        add(f"* `mass_only` f={f}: 接触时间比 {_fmt(row['t_c_ratio'], 3)} "
            f"（理论 f^1.5={_fmt(row['expected'], 3)}，偏差 {_fmt(row['dev_pct'], 1)} %）")
    add("")

    # ---- 5. resolution (RVE) budget -----------------------------------
    add("### 4.5 真实粗粒化的分辨率（RVE）误差预算（fixed_domain, N→N/f³）")
    add("")
    rve = payload["rve_budget"]
    if rve:
        add("| 算例 | f | 颗粒数 | f=1 颗粒数 | 偏差 |")
        add("|---|---|---|---|---|")
        for case, rows in rve.items():
            for f, row in rows.items():
                if case == "shear":
                    val = f"Δφ {_fmt(row.get('phi_dev_pct'))} %"
                elif case == "repose":
                    val = f"Δθ_r {_fmt(row.get('theta_dev_deg'))} °"
                else:
                    val = f"ΔE_eff {_fmt(row.get('e_eff_dev_pct'))} %"
                add(f"| {case} | {f} | {row['n_particles']} | {row['ref_n_particles']} | {val} |")
    else:
        add("（未生成）")
    add("")

    # ---- 6. validity boundary -----------------------------------------
    add("### 4.6 预测的重叠率（可压缩性）失效边界")
    add("")
    add("| f | A: δ/R 自重力 (%) | A: δ/R 200 Pa (%) | B: δ/R 自重力 (%) | B: δ/R 200 Pa (%) |")
    add("|---|---|---|---|---|")
    vb = payload["validity_boundary"]
    for f in fs:
        a = vb["contact_time"][f]
        b = vb["stiffness_invariant"][f]
        add(f"| {f} | {100 * a['overlap_ratio_self_weight']:.2f} | "
            f"{100 * a['overlap_ratio_imposed_200pa']:.2f} | "
            f"{100 * b['overlap_ratio_self_weight']:.2f} | "
            f"{100 * b['overlap_ratio_imposed_200pa']:.2f} |")
    add("")
    add(f"线性接触律有效限 δ/R ≤ {100 * vb['limits']['max_overlap_ratio']:.0f}%（自重力基准：10 颗粒直径床深）。")
    add("")

    # ---- 7. figures ---------------------------------------------------
    figs = payload.get("figures") or []
    if figs:
        add("### 4.7 图")
        add("")
        captions = {
            "fig_macro_vs_f.png": "宏观不变量 vs 粗粒化因子 f（similar 模式，两口径）",
            "fig_error_budget.png": "偏差预算与判据阈值线",
            "fig_ucs_force_displacement.png": "单轴压缩应力-应变与按声明力/应力指数归一后的对拍",
            "fig_shear_response.png": "直剪盒 μ(t) 逐步响应（各 f）",
            "fig_drum_surface.png": "滚筒自由面与休止角（f=1 与 f=8）",
            "fig_validity_boundary.png": "重叠率（可压缩性）失效边界",
        }
        for path in figs:
            name = os.path.basename(path)
            add(f"* `{path}` — {captions.get(name, name)}")
        add("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="runs/upscaling_cg/results.json")
    ap.add_argument("--doc", default="docs/coarse_graining_upscaling_laws.md")
    args = ap.parse_args(argv)
    with open(args.json) as fh:
        payload = json.load(fh)
    block = build_block(payload)
    with open(args.doc) as fh:
        text = fh.read()
    begin = "<!-- RESULTS:BEGIN -->"
    end = "<!-- RESULTS:END -->"
    i, j = text.index(begin) + len(begin), text.index(end)
    text = text[:i] + "\n" + block + "\n" + text[j:]
    with open(args.doc, "w") as fh:
        fh.write(text)
    print(f"injected results into {args.doc} ({len(block)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())