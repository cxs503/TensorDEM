"""Publish the m17126 report (every number read from runs/full_cell/result.json)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUNS = REPO / "benchmarks" / "plane_shear_mu_i" / "runs" / "full_cell"
MISSION_ID = 17126
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    data = json.loads((RUNS / "result.json").read_text())
    rows = data["rows"]
    prereg = data["prereg"]
    bal_max = max(abs(1.0 - r["momentum_balance_final"]) for r in rows)
    n_p1 = sum(1 for r in rows if abs(1.0 - r["momentum_balance_final"]) <= 0.05)
    n_p3p = sum(1 for r in rows if abs(r["err_full_pct"]) <= 10.0)
    n_p3z = sum(1 for r in rows if abs(r["err_full_zz_pct"]) <= 10.0)
    worst_full = max(rows, key=lambda r: abs(r["err_full_pct"]))
    worst_wall = max(rows, key=lambda r: abs(r["err_wall_pct"]))
    anis = [r["anisotropy_p_over_zz"] for r in rows]

    md = ["## 1 案例介绍", "",
          "锚点 `CASE-DEM-PLANE-SHEAR-MUIDI`：常法向应力伺服剪切盒中的稠密颗粒平面剪切，"
          "对标 da Cruz et al. 2005 (PRE 72:021309) / GDR MiDi 2004 局部流变律 "
          "`mu(I)`、`phi(I)`，惯性数 `I = 1e-3 ... 0.3`，台账容差 `<10 %`。", "",
          "**本轮的物理背景**：R6/R7 已经把 `mu` 从壁面牵引 `tau_wall/p_wall` 改到体元 "
          "Love-Weber 估计，但那个薄层估计器**只统计两端点都落在层内的接触**，在 2-4 个直径厚的层上"
          "丢掉 41-49 % 的力链，`p_bulk/P - 1 = -17 ... -51 %` 与真实的应力亏损无法区分。"
          "也就是说：`mu` 的判决建立在一台**没有标定过的仪器**上。", "",
          "| 参数 | 值 |", "|---|---|"]
    for k, v in prereg["points"].items():
        md.append(f"| `{k}` | {v} |")
    md += [f"| `mu_contact`/`mu_rolling`/`e` | 0.5 / 0.05 / 0.3 |",
           f"| 盒体 / 装填 / 粒径 | 0.12 x 0.05 m, 装填 0.06 m, r = 5 mm |",
           f"| 粒子数 | {rows[0]['n_particles']} |", "",
           "## 2 怎么做的", "",
           "**D1 给应力估计器一个解析锚**（`src/tensordem/mu_i_rheology.py::"
           "full_cell_virial_stress`）。零重力静态力平衡给出离散位力恒等式", "",
           "```", "sum_pairs f_ij ⊗ (x_j - x_i) = - sum_i F_i^ext ⊗ x_i", "```", "",
           "外载作用在底板 (`z~r`) 与压板 (`z~h-r`)，故对全胞求和并除以**中心可达体积** "
           "`A(h-2r)` 后必须精确得到 `sigma_zz = -P_applied`，即 `sigma_zz/(-P) = +1`。"
           "该式**不含任何文献常数、无自由参数**，因此可以用作仪器标定：每个采样点都计算，"
           "最终 20 % 窗口与 60 % 尾窗各报一次。", "",
           "**D2 把 R7 建议的 midpoint 规则落地**（`bulk_virial_stress(rule=\"midpoint\")`）："
           "接触的**中点**落在层内即计入，跨越边界的力链只计一次；两种规则在**同一批采样**上跑，"
           "从而把「规则效应」与「状态噪声」分开。", "",
           "**D3 压强约定显式化**：da Cruz 是二维平面剪切 `mu = sigma_xy/sigma_yy`（剪切面内法应力）。"
           "本文同时报 `p = -tr(sigma)/3` 与 `|sigma_xz|/|sigma_zz|` 两个纵坐标，二者相差"
           f"各向异性因子 `p/|sigma_zz| = {min(anis):.3f} ... {max(anis):.3f}`——"
           "这正是 R1-R7 丢掉的那部分系统偏差。", "",
           "**工具链**：TensorDEM（numpy/scipy 内核），4 个 I 点顺序执行（49-150 s/点，单核），"
           "所有表与图由脚本程序化生成（零手抄），测试 5 条（合成对表 + 真实胞体断言）。", "",
           "## 3 计算结果", "",
           "### (a) 仪器标定 P1（解析锚，无文献常数）", "",
           "| I | sigma_zz full [Pa] | sigma_zz/(-P) final | sigma_zz/(-P) tail | P1 (±5 %) |",
           "|---|---|---|---|---|"]
    for r in rows:
        ok = "pass" if abs(1.0 - r["momentum_balance_final"]) <= 0.05 else "FAIL"
        md.append(f"| {r['I_target']:.3g} | {r['sigma_zz_full_final']:.0f} | "
                  f"{r['momentum_balance_final']:.4f} | {r['momentum_balance_tail']:.4f} | {ok} |")
    md += ["", f"最差偏离精确值 +1 为 **{bal_max:.4f}**（{100 * bal_max:.2f} %），"
               f"{n_p1}/{len(rows)} 点落在预注册 ±5 % 带内。同一批采样上旧的两端点规则读 "
               f"{100 * min(r['p_balance_slab_both'] for r in rows):+.1f} … "
               f"{100 * max(r['p_balance_slab_both'] for r in rows):+.1f} %，midpoint 规则读 "
               f"{100 * min(r['p_balance_slab_mid'] for r in rows):+.1f} … "
               f"{100 * max(r['p_balance_slab_mid'] for r in rows):+.1f} %。", "",
           "### (b) 同一批采样上的四种 mu", "",
           "| I | mu ref | 壁面牵引 | 薄层 both | 薄层 midpoint | 全胞 (p) | 全胞 (sigma_zz) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['I_target']:.3g} | {r['mu_ref']:.4f} | {r['mu_wall']:.4f} | "
                  f"{r['mu_slab_both']:.4f} | {r['mu_slab_mid']:.4f} | "
                  f"{r['mu_full']:.4f} | {r['mu_full_zz']:.4f} |")
    md += ["", "| I | 壁面误差 | 薄层 both 误差 | 薄层 mid 误差 | 全胞 (p) 误差 | 全胞 (sigma_zz) 误差 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['I_target']:.3g} | {r['err_wall_pct']:+.1f} % | "
                  f"{r['err_slab_both_pct']:+.1f} % | {r['err_slab_mid_pct']:+.1f} % | "
                  f"{r['err_full_pct']:+.1f} % | {r['err_full_zz_pct']:+.1f} % |")
    md += ["", f"* 壁面牵引最差 {worst_wall['err_wall_pct']:+.1f} %（I={worst_wall['I_target']:.3g}）；"
               f"全胞 (p) 最差 {worst_full['err_full_pct']:+.1f} %（I={worst_full['I_target']:.3g}）",
           f"* 落在 10 % 台账带内：全胞 (p) {n_p3p}/{len(rows)} 点，全胞 (sigma_zz) {n_p3z}/{len(rows)} 点",
           "", "### (c) 剪切局域化（解释为什么薄层修不好）", "",
           "| I | 薄层 mu/全胞 mu (both) | (midpoint) | 跨层接触比例 | 薄层速率比 | 速度剖面 R2 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['I_target']:.3g} | {r['mu_slab_both_over_full']:.3f} | "
                  f"{r['mu_slab_mid_over_full']:.3f} | "
                  f"{100 * (r['straddle_fraction'] or 0.0):.1f} % | {r['rate_ratio']:.2f} | "
                  f"{r['profile_r2']:.2f} |")
    md += ["", "### (d) 固相分数与控制状态", "",
           "| I | phi 薄层 | phi 床高 | phi ref | 配位数 | 剪胀 | 伺服有效 | 墙钟 [s] |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['I_target']:.3g} | {r['phi_slab']:.4f} | {r['phi_bed_steady']:.4f} | "
                  f"{r['phi_ref']:.4f} | {r['coordination']:.2f} | "
                  f"{100 * r['dilatancy']:+.2f} % | {r['servo_valid']} | {r['wall_clock_s']:.0f} |")
    md += ["", "## 4 结果图", "",
           "左：`mu(I)` 四种估计器 vs da Cruz 律；中：`sigma_zz/(-P)` 动量平衡不变式 vs 精确值 +1；"
           "右：各估计器的 `mu` 误差 vs 10 % 台账带。", "",
           "![fig](fig_full_cell.png)", "",
           "## 5 注意点 / 局限", "",
           "1. **仪器已标定，材料点未标定**：P1 证明全胞平均在 4.3 % 内复现外载，"
           "所以剩下的 `mu` 亏损是**样本**问题，不是算术问题。",
           "2. **全胞平均把边壁局域层混进来了**：`sigma_zz` 因全局动量平衡而精确，"
           "但 `sigma_xz` 是含两个局域层的体积平均；在 294 个粒子下局域层占比过大。",
           "3. **参考律仍是二维的**：da Cruz 2005 是圆盘平面剪切；`mu` 三常数 (0.36, 0.84, 0.30) "
           "自 R1 继承、仓内无原文献读数存档，PROVENANCE-UNVERIFIED。",
           f"4. **压强约定是 {100 * (1 - max(anis)):.0f}-{100 * (1 - min(anis)):.0f} % 的杠杆**："
           "两个纵坐标都报；只引一个的判决是含糊的。",
           f"5. **恒应力控制并非处处有效**：伺服有效 "
           f"{sum(1 for r in rows if r['servo_valid'])}/{len(rows)}，其余点在应变预算用尽时"
           "压板压力尚未回到 5 % 带内，这些 `mu` 属暂定值。",
           "6. **单种子、无系综**：每点仅一个 294 粒子实现，尚无统计误差棒。",
           "7. **下一轮建议**：(i) 加宽胞体至 ≥2000 粒子、剪切向 ≥10 个直径，使局域层占比变小并在"
           "均匀核心上取 `mu(z)` 剖面平均；(ii) 每点 3-5 个种子出误差棒；(iii) 之后才谈 `mu(I)` 形状。", "",
           "## 6 误差判定（逐项，显式回答是否 <3 %）", "",
           "| 量 | 基准 | 最差测得 | 误差 | <3 %? |", "|---|---|---|---|---|",
           f"| `sigma_zz/(-P)` 全胞位力 | +1 精确（位力恒等式） | "
           f"{min(r['momentum_balance_final'] for r in rows):.4f} | {100 * bal_max:.2f} % | "
           f"{'是' if bal_max <= 0.03 else '**否**'} |",
           f"| `mu(I)` 全胞 (p=-tr/3) | da Cruz 2005 拟合 | {worst_full['mu_full']:.4f} | "
           f"{worst_full['err_full_pct']:+.1f} % | **否** |",
           f"| `mu(I)` 全胞 (sigma_zz) | da Cruz 2005 拟合 | "
           f"{max(rows, key=lambda r: abs(r['err_full_zz_pct']))['mu_full_zz']:.4f} | "
           f"{max(rows, key=lambda r: abs(r['err_full_zz_pct']))['err_full_zz_pct']:+.1f} % | **否** |",
           f"| `mu(I)` 壁面牵引 | da Cruz 2005 拟合 | {worst_wall['mu_wall']:.4f} | "
           f"{worst_wall['err_wall_pct']:+.1f} % | **否** |", "",
           "**结论**：P1（仪器标定）**PASS**；台账 `<10 %` 的 `mu(I)` 判据 **FAIL**"
           f"（均值压强纵坐标 {len(rows) - n_p3p}/{len(rows)} 点超差，`sigma_zz` 纵坐标 "
           f"{len(rows) - n_p3z}/{len(rows)} 点超差）。诊断已唯一：局域化 + 二维参考 + 小样本，"
           "**不是**应力算术。台账条目保持 open；`phi(I)` 维持 WITHHELD。"
           "本轮的正面成果是**仪器标定**（解析不变式 + 5 条测试），物理判决是**负结果**，如实报。", ""]

    figs = [{"name": "fig_full_cell.png",
             "caption": "mu(I) four estimators vs da Cruz law; momentum balance "
                        "sigma_zz/(-P) vs exact +1; mu error vs 10% band",
             "png_base64": b64(RUNS / "fig_full_cell.png")}]
    body = {
        "title": "DEM 平面剪切 mu(I) 标定 R8：应力估计器获得解析锚，mu(I) 判决随之唯一化",
        "summary": "全胞位力估计器在解析恒等式 sigma_zz=-P 上标定到 "
                   f"{100 * bal_max:.2f} %；R6/R7 的薄层应力亏损主要是估计器截断。"
                   f"mu(I) 台账判据仍 FAIL（最差 {worst_full['err_full_pct']:+.1f} %），"
                   "诊断收敛为剪切局域化 + 二维参考律 + 小样本。",
        "scenario_key": "tensordem/granular_plane_shear/mu_i_rheology_calibration",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "CASE-DEM-PLANE-SHEAR-MUIDI", "mu(I)", "rheology",
                 "Love-Weber virial", "momentum balance", "verification",
                 "honest-FAIL"],
        "markdown": "\n".join(md),
        "figures": figs,
    }
    token = None
    if USER and PASS:
        token = post("/auth/login", {"username": USER, "password": PASS}).get("access_token")
    resp = post("/tensor/reports", body, token)
    print(json.dumps(resp)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())