"""Assemble the progress-report payload for IN102-N02.

Reads the campaign ledgers (`summary.json`, `summary_B.json`) and the rendered
figures, and writes `/tmp/report.json` (title / summary / scenario_key /
mission_id / tags / markdown / figures).  All numbers are injected from the
JSON, never typed by hand.

    python benchmarks/coarse_graining/make_progress_report.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIGDIR = REPO / "docs" / "verification" / "figures"
DOC = REPO / "docs" / "verification" / "COARSE_GRAINING_MACRO.md"


def fig(name: str):
    path = FIGDIR / name
    if not path.exists():
        return None
    return {
        "name": path.stem,
        "caption": {
            "cg_mu_i_curves": "μ(I) measured on the constant-stress shear box for "
                              "each coarse-graining factor k (I-matched, same device).",
            "cg_micro_invariance": "Coordination number Z and solid fraction φ vs k, "
                                   "normalised to k=1, with the predeclared ±3% band.",
            "cg_pareto": "Speedup vs maximum macro deviation; red = Pareto front.",
        }.get(path.stem, path.stem),
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    }


def main() -> None:
    payloads = {}
    for name in ("summary.json", "summary_B.json"):
        path = HERE / name
        if path.exists():
            payloads[name] = json.loads(path.read_text())

    lines = []
    add = lines.append
    add("## 1 案例介绍")
    add("")
    add("DEM 粗粒化（coarse-graining）把每个接触级参数放大到宏观可用的尺度：若几何、刚度、"
        "密度、摩擦与围压按相似律缩放后宏观流变不变，则粗粒模型可作为原型颗粒床的代理。"
        "本节点用**同一台**恒压剪切盒（`tensordem.mu_i_rheology.simulate_servo_shear`，"
        "x 方向周期、侧壁光滑、底板与顶板摩擦、顶板恒压伺服）测三个宏观锚：μ(I) 曲线、"
        "配位数 Z、孔隙率 φ=1−固相分数，外加自重柱塌落的休止角 θ_r。"
        "基准材料 r=5 mm、ρ=2500 kg/m³、μ=0.5、μ_r=0.05、e=0.3，围压 62.5 kPa，"
        "I ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}，k ∈ {1,2,4,8}。")
    add("")
    add("## 2 怎么做的")
    add("")
    add("两条相似律被当成竞争假设并列检验（同一装置、同一 I 网格、同一容差，阈值预先固定，"
        "未按结果调整）：")
    add("")
    add("| 口径 | 刚度 | 相对重叠 u = P d / k_n | 接触时间 | 剪切率 |")
    add("|---|---|---|---|---|")
    add("| A 接触时间不变 | k_n ∝ k³ | ∝ 1/k（随 k 变软→变刚） | 不变 | γ̇/k |")
    add("| B 声速不变 | k_n ∝ k | 不变 | ∝ k | γ̇/k |")
    add("")
    add("两条口径都按 I 匹配剪切率（γ̇ = I√(P/ρ)/d），因为固定 P 时 I ∝ γ̇·d，"
        "用原型剪切率跑粗粒模型会把 I 放大 k 倍，那是 I 失配而非流变检验。"
        "设备侧修正了四个坑（详见文档 §6）：摩擦弹簧符号反了导致样本被持续加热、"
        "速度正则化 Coulomb 在低 I 极限不可用、伺服用瞬时接触力会被力链噪声泵浦、"
        "平板穿透未夹紧会让粒子越过顶板后力发散。")
    add("")

    for name, label in (("summary.json", "口径 A（接触时间不变）"),
                        ("summary_B.json", "口径 B（声速不变）")):
        s = payloads.get(name)
        if not s:
            continue
        add(f"## 3 计算结果 — {label}")
        add("")
        add("| k | Δμ(I) max [%] | ΔZ [%] | Δφ [%] | Δθ_r [%] | max dev [%] | speedup |")
        add("|---|---|---|---|---|---|---|")
        for row in s["rows"]:
            rep = ("n/a" if row["repose_dev_pct"] is None
                   else f"{row['repose_dev_pct']:.2f}")
            add(f"| {row['k']:g} | {row['rheology_dev_pct']:.2f} | "
                f"{row['coordination_dev_pct']:.2f} | {row['solid_fraction_dev_pct']:.2f} | "
                f"{rep} | {row['max_deviation_pct']:.2f} | {row['speedup']:.2f}× |")
        add("")
        add(f"safe_k_upper_bound（该口径）= **{s['safe_k_upper_bound']:g}**；"
            f"容忍线：流变 {s['tolerances_pct']['rheology']:g}%、微观 "
            f"{s['tolerances_pct']['micro']:g}%、休止角 "
            f"{s['tolerances_pct']['repose']:g}%。")
        add("")

    add("## 4 结果图")
    add("")
    for name in ("cg_mu_i_curves.png", "cg_micro_invariance.png", "cg_pareto.png"):
        if (FIGDIR / name).exists():
            add(f"![{name}](figures/{name})")
            add("")

    add("## 5 注意点")
    add("")
    add("1. 口径 A 的问题不是数值噪声而是相似性缺口：k_n ∝ k³ 让相对重叠 u = P d/k_n 随 k 减小，"
        "接触网络在 k 大时更“刚”，配位数与固相分数随之漂移；口径 B 保持 u 不变，"
        "是唯一同时匹配 I 与 u 的缩放。")
    add("2. 粗粒化加速比是**域体积**加速比（同等 CPU 秒内推进的物理体积），"
        "不是同一问题的粒子数减少：几何缩放下盒内颗粒数不变。")
    add("3. 休止角为单种子锥面拟合，统计误差单列（stderr），未并入容差。")
    add("4. 单分散球、无黏结、无破碎；剪切盒零重力、无 Lees–Edwards 剪切框，"
        "测的是均匀流变而非壁面驱动的剖面。")
    add("")

    add("## 6 误差判定")
    add("")
    add("对基准 k=1，各锚逐项给出偏差（判据：流变 ≤5%、Z/φ/θ_r ≤3%）：")
    add("")
    for name, label in (("summary.json", "口径 A"), ("summary_B.json", "口径 B")):
        s = payloads.get(name)
        if not s:
            continue
        add(f"**{label}**")
        add("")
        for row in s["rows"]:
            verdict_mu = "是" if row["rheology_dev_pct"] <= 5.0 else "否"
            verdict_mi = "是" if row["micro_dev_pct"] <= 3.0 else "否"
            add(f"* k={row['k']:g}: μ(I) {row['rheology_dev_pct']:.2f}% "
                f"(<5%? {verdict_mu})、Z {row['coordination_dev_pct']:.2f}%、"
                f"φ {row['solid_fraction_dev_pct']:.2f}% (max {row['micro_dev_pct']:.2f}%,"
                f" <3%? {verdict_mi})、"
                + ("θ_r " + (f"{row['repose_dev_pct']:.2f}%" if row['repose_dev_pct'] is not None
                             else "n/a") + " (<3%? "
                   + ("是" if (row["repose_dev_pct"] or 0) <= 3.0 else "否") + ")")
                + "。")
        add("")

    add("## 7 审计链")
    add("")
    add("* 每档 k 的原始结果：`benchmarks/coarse_graining/{(B_)k<k>}/result.json`；"
        "汇总账本：`summary.json` / `summary_B.json`。")
    add("* 报告文档：`docs/verification/COARSE_GRAINING_MACRO.md`（数值由脚本注入，零手抄）。")
    add("* 回归门：`pytest tests/test_coarse_graining_scaling.py`（含设备物理门、"
        "I 匹配必要性负控、锥面拟合解析校核、宏观不变性判据）。")
    add("")

    markdown = "\n".join(lines)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    (HERE / "report_markdown.md").write_text(markdown)

    figures = [f for f in (fig("cg_mu_i_curves.png"), fig("cg_micro_invariance.png"),
                           fig("cg_pareto.png")) if f]
    payload = {
        "title": "DEM 粗粒化标定律三宏观锚：μ(I)/配位数-孔隙率/休止角跨 k 不变性与加速比 Pareto（IN102-N02）",
        "summary": ("以恒压剪切盒为统一装置，k∈{1,2,4,8}×I∈{1e-3..1e-1} 检验 DEM 粗粒化"
                    "宏观不变性；判定接触时间不变（k_n∝k³）与声速不变（k_n∝k）两条缩放律"
                    "何者宏观自洽，并给出速度-偏差 Pareto 前沿与安全 k 上界。"),
        "scenario_key": "dem_coarse_graining_macro",
        "mission_id": 8998,
        "tags": ["TensorDEM", "coarse-graining", "mu(I) rheology", "verification",
                 "IN102-N02", "DEM"],
        "markdown": markdown,
        "figures": figures,
    }
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KiB, {len(figures)} figures)")


if __name__ == "__main__":
    main()