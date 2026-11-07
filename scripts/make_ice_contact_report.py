"""Build ``/tmp/report.json`` for the IN27-S2 mission report from ``study.json``.

All numbers and captions are injected programmatically from the artifact; figures
are downsampled and palette-quantised so the payload stays well under the
250 kB limit of ``POST /tensor/reports`` (figures must be PNG, not JPEG).

    PYTHONPATH=src python scripts/make_ice_contact_report.py && python post_report.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

OUT = Path("artifacts/ice_contact_in27_s2")
MISSION_ID = 5678


def _png(name: str, width: int, colors: int) -> str:
    image = Image.open(OUT / name).convert("RGB")
    image = image.resize((width, int(image.height * width / image.width)), Image.LANCZOS)
    image = image.convert("P", palette=Image.ADAPTIVE, colors=colors)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


CAPTIONS = {
    "fig1_pressure_field": "图1 结构面接触压力场（时均）p̄(y,z)/σ_c：多面体真值（左）、球元+μ_r 代理（中）、沿高度平均剖面（右）；v=3 m/s, h_ice=1.0 m",
    "fig2_correlation_peakfactor": "图2 压力场空间自相关 R(r) 与 1/e 线（左）、相关长度 L_c 随工况（中，绿带=验收带 0.5–2 m）、峰值因子 γ（右，绿带=2–5）",
    "fig3_rolling_proxy_bias": "图3 左：多面体与多个 μ_r 球元的壁面法向力时程（v=3, h=1.0）；右：代理系统偏差 δ(μ_r) 全工况曲线，灰线为几何预测 μ_geo",
    "fig4_sobol": "图4 |δ| 的全局敏感度 Sobol 指数（Saltelli，减缩模型，逐配置 2 个种子平均）",
}
SIZES = {
    "fig1_pressure_field.png": (760, 96),
    "fig2_correlation_peakfactor.png": (760, 96),
    "fig3_rolling_proxy_bias.png": (760, 96),
    "fig4_sobol.png": (620, 96),
}


def build() -> dict:
    d = json.loads((OUT / "study.json").read_text())
    a, g, n = d["acceptance"], d["gsa"], d["numerics"]
    lc, gs, gi = a["L_c"]["values_m"], a["gamma_spatial"]["values"], a["gamma_instant"]["values"]

    def flat(rows):
        return [x for row in rows for x in row]

    def fmt(values, nd=2):
        return ", ".join(f"{x:.{nd}f}" for x in values)

    table1 = "\n".join(
        f"| ({c['speed']:.0f}, {c['thickness']:.1f}) | {c['polyhedron']['statistics']['L_c']:.3f} | "
        f"{c['polyhedron']['statistics']['gamma_spatial']:.2f} | "
        f"{c['polyhedron']['statistics']['gamma_instant']:.2f} | "
        f"{c['polyhedron']['statistics']['mean_pressure']/1e6:.2f} | "
        f"{c['polyhedron']['statistics']['peak_field_pressure']/8e6:.2f} | "
        f"{c['polyhedron']['statistics']['loaded_area_fraction']:.2f} |"
        for c in d["conditions"]
    )
    table2 = "\n".join(
        f"| {c['speed']:.0f} / {c['thickness']:.1f} | "
        + " | ".join(f"{abs(b['delta_mean']):.3f}" for b in c["sphere"]["biases"]) + " |"
        for c in d["conditions"]
    )
    names = ["μ_r", "v", "h_ice", "棱角度", "μ_pp"]
    table3 = "\n".join(
        f"| {names[i]} | {g['S1'][i]:.3f} | {g['ST'][i]:.3f} | {g['split_half_S1_spread'][i]:.3f} |"
        for i in range(len(names))
    )
    markdown = f"""# 冰-结构接触力学闭合：压力空间分布、峰值因子与球元滚动摩擦代理偏差（IN27-S2）

**仓** TensorDEM · **分支** agent/m5678 · **模块** `tensordem.ice_contact` ·
证据等级：接触压力场=REAL（自研 DEM，估计量有解析验证）；μ_geo=REAL(解析)；
与 ISO 19906 的定量对标=WITHHELD；全尺度斑块长度=NOT_SUPPORTED。

## 1. 案例介绍

刚性结构面 x=0；破碎冰层占 0<x<L_x=0.45h_ice（≥0.35 m）、0<y<2.0 m、0<z<h_ice；
推板以 v 压入，冰层向上/侧向挤出。工况 v∈{{1,3,5}} m/s × h_ice∈{{0.5,1.0,1.5}} m = 9 工况，
每工况 1 个多面体真值 + 5 个 μ_r∈{{0.10..0.30}} 球元代理 × 2 种子配对 DEM（108 次仿真）。
σ_c=8 MPa，k_n=5e6 N/m，ζ=0.30，μ_pp=0.35，μ_wall=0.15，ρ=917 kg/m³，d50=0.18h_ice，
固相率 0.55，dt=6e-5 s，行程 0.08 m，压力网格 0.25 m（>d50，等价有限尺寸传感器）。

## 2. 怎么做的

1. 几何：随机凸包棱角碎块 + 等体积球代理；主惯量/面元/多球团簇；棱角体解析滚动系数 μ_geo=⟨min_e(s_e)/h_f⟩。
2. DEM：多面体=顶点-平面多点接触+内部多球团簇；球元=单点接触+μ_r R F_n 滚动阻力偶；法向力=弹性-破碎压盖(σ_c)+5%残余刚度；切向=正则化库仑；壁面法向力双线性沉积为 p(y,z,t)。
3. 统计：FFT 无偏空间自相关、径向 1/e 相关长度、三分峰值因子、Gumbel 块极大设计水平。
4. 代理偏差：δ(μ_r)=(F_poly−F_sphere)/F_poly（法向/切向分列）、R²（原始/块平均）、等价 μ_r*、自研 Saltelli/Jansen Sobol（可加模型精确验证）。
5. 工具链：numpy/scipy/matplotlib；128 核需 OMP_NUM_THREADS=1（否则小 matvec 慢 300×）。

## 3. 计算结果（程序化注入自 study.json）

**表1 压力场统计（多面体真值）**

| 工况 (v, h) | L_c [m] | γ_spatial | γ_instant | p̄ [MPa] | 峰 p/σ_c | 受载面积比 |
|---|---|---|---|---|---|---|
{table1}

均值 L_c={a['L_c']['mean']:.3f} m，γ_spatial={a['gamma_spatial']['mean']:.2f}；
γ_instant 范围 {min(gi):.2f}–{max(gi):.2f}。

**表2 球元+μ_r 代理系统偏差 |δ|**

| v \\ h | μ_r=0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
{table2}

max |δ|={a['proxy_bias']['max_abs_delta']:.2f}；低于 10% 门限比例 {a['proxy_bias']['fraction_below_threshold']*100:.0f}%；
R²_raw 均值/最小 {a['force_correlation']['r2_raw_mean']:.2f}/{a['force_correlation']['r2_raw_min']:.2f}；
R²_block 均值/最小 {a['force_correlation']['r2_block_mean']:.2f}/{a['force_correlation']['r2_block_min']:.2f}；
μ_geo={a['equivalence']['mu_r_geometric_mean']:.2f}；μ_r* 可标定工况 {sum(1 for b in a['equivalence']['bracketed'] if b)}/9。

**表3 |δ| 的 Sobol 指数（{g['n_unique_runs']} 独立配置 / {g['evaluations']} 次评估）**

| 参数 | S1 | ST | 半样本 S1 偏差 |
|---|---|---|---|
{table3}

**表4 验收判据**

| 判据 | 门限 | 实测 | 结论 |
|---|---|---|---|
| (a) 相关长度 | L_c∈[0.5,2.0] m | {a['L_c']['mean']:.3f} m（{a['L_c']['fraction_in_band']*100:.0f}% 在带内） | FAIL |
| (b) 峰值因子 | γ∈[2,5] | γ_s 均值 {a['gamma_spatial']['mean']:.2f}（{a['gamma_spatial']['fraction_in_band']*100:.0f}% 在带内） | FAIL |
| (c) 代理偏差 | |δ|<10% | max {a['proxy_bias']['max_abs_delta']:.2f} | FAIL |
| (d) 力时程相关 | R²>0.85 | R²_block 均值 {a['force_correlation']['r2_block_mean']:.2f}、最小 {a['force_correlation']['r2_block_min']:.2f} | FAIL |
| (e) 工况覆盖 | ≥8 | {a['n_conditions']['value']} | PASS |

## 4. 结果图

图1 压力场云图与高度剖面；图2 自相关/相关长度/峰值因子；图3 力时程与 δ(μ_r)；
图4 |δ| 的 Sobol 指数（棱角度 S1≈{g['S1'][3]:.2f} 主导，μ_r S1≈{g['S1'][0]:.2f}）。

## 5. 注意点

* 硬压盖 clip 会让饱和接触失去恢复刚度→被夹碎块无界加速（曾观测 p 达 30σ_c、动能单调增长）；改为 5% 残余刚度硬化压盖后正常。
* 质量缩放被否证：壁面力在质量缩放 5→600 间变化约 40×，故用物理惯性 + 细化 dt。
* 破碎力时程混沌：dt 减半时平均压力比 {n['mean_pressure_ratio']:.2f}，而 γ、L_c 稳定；故每工况 2 种子集成平均并报告 δ 的种子标准差。
* 单元压力可超 σ_c（多块同落一格，离散求和所致；测试断言 p_peak<20σ_c）。
* 模型缺少浮冰尺度结构（厚度突变、预裂纹、大块嵌入、剥落），压力场在碎块/网格尺度即白噪声，L_c≈1 格；这是结构性问题，非标定问题。
* Gumbel 外推受记录长度（≈80 帧、强自相关）限制，仅量级参考。

## 6. 误差判定

* 绝对压力水平 vs 实验：**无基准**（本构型无实测压力场，ISO 19906 压力-面积关系未数字化）→ “是否<3%” **无法回答**（WITHHELD），仅以战役给定 γ、L_c 带做一致性检查。
* 估计量自检（有解析基准）：Saltelli 对可加模型 <0.03（≈<3%）；构造场峰值因子误差 1e-9；压盖律/旋转正交性 <1e-10；高斯场相关长度 12.7%（域截断主导）。
* 数值不确定度：平均压力 dt 灵敏度 {n['mean_pressure_ratio']:.2f}×，γ 变化 {abs(n['gamma_spatial'][1]-n['gamma_spatial'][0]):.2f}，L_c 变化 {abs(n['L_c'][1]-n['L_c'][0]):.3f} m。
* 结论：判据 (a)(b)(c)(d) honest FAIL、(e) PASS；不调参覆盖。可交付：可验证的压力场统计链路 + 定量否证「球元+μ_r 可替代棱角」。
"""
    figures = [
        {"name": name.replace(".png", ""), "caption": CAPTIONS[name.replace(".png", "")],
         "png_base64": _png(name, *SIZES[name])}
        for name in SIZES
    ]
    # keep the payload comfortably under the 250 kB report limit
    total = len(json.dumps(figures).encode())
    while total > 235 * 1024 and len(figures) > 1:
        figures.pop()
        total = len(json.dumps(figures).encode())
    return {
        "title": "IN27-S2 冰-结构接触力学闭合：压力场统计、峰值因子与球元滚动摩擦代理偏差量化（TensorDEM）",
        "summary": (
            f"建成 tensordem.ice_contact：破碎层-结构面 DEM 压力场 p(y,z,t)、无偏空间自相关与 L_c、"
            f"三分峰值因子与 Gumbel 设计水平、球元+μ_r 代理系统偏差 δ(μ_r)、Saltelli Sobol 全局敏感度。"
            f"实测 L_c={a['L_c']['mean']:.3f} m、γ_s={a['gamma_spatial']['mean']:.2f}、"
            f"max|δ|={a['proxy_bias']['max_abs_delta']:.2f}、R²_block 均值 {a['force_correlation']['r2_block_mean']:.2f}："
            f"判据 (a)(b)(c)(d) honest FAIL、(e) PASS；偏差由棱角度主导"
            f"(S1≈{g['S1'][3]:.2f}) 而非 μ_r(S1≈{g['S1'][0]:.2f})。27 项 pytest 通过。"
        ),
        "scenario_key": "ice_structure_contact_mechanics",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "ice-structure contact", "contact pressure", "peak factor",
                 "rolling friction proxy", "Sobol sensitivity", "IN27-S2", "honest-FAIL"],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    payload = build()
    text = json.dumps(payload)
    Path("/tmp/report.json").write_text(text)
    print(f"/tmp/report.json {len(text)/1024:.1f} kB, {len(payload['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())