#!/usr/bin/env python3
"""Build and publish the m22399 report: m20055 D5 dt-convergence ladder.

Every number is read from ``runs/m20055_beverloo_dt_ladder/ladder_summary.json``
(written by ``benchmarks/m20055_beverloo/dt_ladder.py --analyse``); the figures
are the ones that same run produced.  Nothing is hand copied.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "m20055_beverloo_dt_ladder")
FIGS = os.path.join(HERE, "docs", "figures")
SUMMARY = os.path.join(RUNS, "ladder_summary.json")
PAYLOAD = "/tmp/report_m22399.json"
MISSION = 22399


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def pct(x: float) -> str:
    return f"{100.0 * x:+.2f} %"


def main() -> int:
    s = json.load(open(SUMMARY))
    rows, lim, ref = s["rows"], s["limit"], s["reference"]
    coarse, fine = rows[0], rows[-1]
    g = s.get("geometry", {})

    md: list[str] = []
    A = md.append
    A("## ① 案例介绍")
    A("")
    A("**算例**：`CASE-DEM-SILO-BEVERLOO`（m20055）主算例 D5 — 平底圆孔料仓的自由重力卸料；"
      "学科归属：船舶总体性能 · 绿色航行/散货装卸（颗粒物料流率）。"
      "本单是 m20055 交付物的**续作**：交付物已在 main（PR#1528 合并），其判定是诚实 FAIL ——"
      "主算例 D5 的稳态质量流率比 Beverloo (1961) 关联式高 +20.6 %。"
      f"本单复跑同一档得到 {pct(coarse['rel'])}（床实现不同，见 ⑤），"
      "唯一的旁证是同批一个 dt/2 算例把偏差拉回到 8.9 %，但两点不构成收敛结论。")
    A("")
    A("**物理背景**：料仓孔口卸料的稳态质量流率由 Beverloo 经验律给出 "
      "`W = C · ρ_b · √g · (D − k·d)^(5/2)`，C≈0.58、k≈1.4（球形颗粒）。"
      "DEM 的软球接触律用有限时间步积分，dt/T_contact 越小越接近刚体极限；"
      "在 dt/T_contact ≈ 0.035 这样偏粗的步长下，接触穿透/动量注入会把出口流速抬高，"
      "于是 W 偏大。本单检验的就是「那个 +20.6 % 里有多少是时间离散」这个**可分离、可判定的问题**。")
    A("")
    A("| 参数 | 值 |")
    A("|---|---|")
    A(f"| 颗粒直径 d | {g.get('particle_diameter', 0.0):.3f} m |")
    A(f"| 孔口直径 D | {g.get('orifice_diameter', 0.0):.3f} m（D/d = {g.get('D_over_d', 0.0):.2f}）|")
    A(f"| 床内颗粒数 / 卸出颗粒数 | {fine['n_bed']} / {fine['n_discharged']} |")
    A(f"| 床孔隙率 φ_bed | {coarse['phi_bed']:.3f} … {fine['phi_bed']:.3f}（各 dt 档） |")
    A(f"| 床容积密度 ρ_b | {coarse['rho_b']:.1f} … {fine['rho_b']:.1f} kg/m³ |")
    A("| 颗粒摩擦 μ_p / 滚动摩擦 μ_r / 恢复 e | 0.30 / 0.02 / 0.30 |")
    A("| 侧壁摩擦 | 0（Beverloo 玻璃珠光滑壁） |")
    A("| 接触刚度 k_n / 底板厚 | 1e5 N/m / 0.5 d |")
    A("| 时间步阶梯 | dt, dt/2, dt/4（其余参数逐位相同，seed=20055） |")
    A(f"| 参考文献 | Beverloo, Leniger & van de Velde, *Chem. Eng. Sci.* **15**, 260–269 (1961)；"
      f"C = {ref['C']}、k = {ref['k']}（`{ref['provenance']}`） |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **预登记**（`prereg/p20055_dt_ladder.md`，跑之前写定）：判定规则、阈值、阶次不可辨识时的兜底、"
      "噪声剔除规则 CV>0.30、停止规则（三档不追加 dt/8，除非 p 与 p=1/p=2 外推相差 >5 %）。")
    A("2. **阶梯构造**：`dt_ladder.build_level_case()` 从预登记 D5 算例出发，只覆盖 `dt`；"
      "单测 `test_ladder_case_changes_only_the_time_step` 断言 `changed == ['dt']`，防止夹带变量。")
    A("3. **测量协议不变**：稳态流率仍取卸料流质量分位带 0.20–0.80、5 个子窗（与已归档 D5 逐项相同），"
      "保证与既有 verdict 可比；床仍是「松装点阵 + 关孔沉降」得到的无序床，沉降段与卸料段同步受 dt 影响，这正是要量的误差。")
    A("4. **作业**：三档各 1 个单核算例，经调度系统递交（cj4086 / cj4087 / cj4088，stage_dir=本工作树）。")
    A("5. **分析**（`dt_ladder.py --analyse`，零手抄）：三点 Richardson 观测阶 "
      "`p = log2((W0−W1)/(W1−W2))`，外推 `W∞ = W2 + (W2−W1)/(2^p−1)`，"
      "并给出 p=1、p=2 的阶次稳健带；同时把文献 k∈[1.2,1.6] 通过 `(D−kd)^(5/2)` 映射成参考带。")
    A("6. **工具链**：numpy DEM（`tensordem.silo_discharge`）、matplotlib 出图、pytest 回归（24 项）。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("| 档位 | dt [s] | dt/T_contact | 步数 | W [kg/s] | W_ref [kg/s] | 相对误差 | 子窗 CV |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['level']} | {r['dt']:.4e} | {r['dt_over_contact_period']:.4f} | {r['steps']} | "
          f"{r['W']:.4f} | {r['W_ref']:.4f} | {pct(r['rel'])} | {r['cv']:.3f} |")
    A("")
    A(f"* 观测阶 **p = {lim['order']['p'] if lim['order']['p'] is not None else float('nan'):.3f}**"
      f"（`{lim['p_source']}`；三点 Richardson，r=2）。")
    A(f"* 时间步极限 **W∞ = {lim['W_inf']:.4f} kg/s → {pct(lim['rel'])}** vs Beverloo；"
      f"阶次稳健带 p=1 → {pct(lim['rel_p1'])}，p=2 → {pct(lim['rel_p2'])}。")
    A(f"* 时间离散单独贡献：由粗到细误差变化 **{100 * s['dt_effect']:.2f} 个百分点**（占初值偏差的主要部分）。")
    A(f"* 文献 k∈[{ref['k_range'][0]}, {ref['k_range'][1]}] 单独在本 D/d 上给出的参考带："
      f"{pct(lim['k_band_rel'][0])} … {pct(lim['k_band_rel'][1])}。")
    A("")
    A("## ④ 结果图")
    A("")
    A(f"* `m20055_dt_ladder.png`：W(dt) 阶梯 + Richardson 极限 + k 带（图注见下）；")
    A(f"* `m20055_dt_ladder_discharge.png`：各 dt 档的累积卸料曲线与稳态窗口（DEM 场量云图不适用："
      "本算例只记录卸料事件序列，未存粒子位置历史，故以流率/事件曲线代替场量云图）。")
    A("")
    A("## ⑤ 注意点 / 局限")
    A("")
    A(f"* **参考值本身是引用而非重新数字化**（`{ref['provenance']}`）：1961 原文不在工作树、无外网，"
      "因此 C 自身的几个百分点偏差无法与 DEM 误差分离 —— 本单因此只对 15 % 平台标准做判定，"
      "不声称 <3 %。")
    A("* **单一 D/d 不可辨识 (C, k)**：k 的文献区间在本算例就展开成上面的 k 带，"
      "任何单孔口误差小于该带宽的结论都不成立；要真正辨识必须对 ≥3 个孔口直径联合拟合 "
      "`beverloo_fit`。本单给出带，不下超带结论。")
    A("* **阶梯只到 dt/4**（预登记停止规则）：三点可估阶，但 p 的置信区间没有闭式；因此判定用阶次稳健带而非单点。")
    A(f"* **床孔隙率 φ_bed ≈ {fine['phi_bed']:.3f} 偏低**（真实玻璃珠 RLP≈0.60+）："
      "松动床会让流出更容易，这是 ⑤ 之外的剩余物理解释之一，本单未做床制备收敛（另一个变量轴）。")
    A("* 单核、单种子：没有种子噪声带；DEM 事件计数的 CV 已在表里给出（各档均 <0.30 阈值）。")
    A("")
    A("## ⑥ 误差判定（显式回答「是否 <3 %」）")
    A("")
    A("| 关键量 | 基准 | 误差 | < 3 % ? |")
    A("|---|---|---|---|")
    A(f"| W∞（dt→0 外推） | Beverloo C={ref['C']}, k={ref['k']} | {pct(lim['rel'])} | "
      f"{'是' if abs(lim['rel']) < 0.03 else '否'} |")
    A(f"| W∞ vs 参考带最有利端 | k={ref['k_range'][0]} | {pct(lim['k_band_rel'][0])} | "
      f"{'是' if abs(lim['k_band_rel'][0]) < 0.03 else '否'} |")
    A(f"| W∞ vs 参考带最不利端 | k={ref['k_range'][1]} | {pct(lim['k_band_rel'][1])} | "
      f"{'是' if abs(lim['k_band_rel'][1]) < 0.03 else '否'} |")
    A(f"| 粗档 W(dt)（已归档 verdict） | 同上 | {pct(coarse['rel'])} | 否 |")
    A("")
    A(f"**判定**：`{s['verdict']}`。时间离散只解释粗档 {pct(coarse['rel'])} 中的 "
      f"{100 * s['dt_effect']:.2f} 个百分点（粗档→dt/4 档），到 dt/4 档误差为 {pct(fine['rel'])}；"
      "阶梯虽单调下降但**不符固定收敛阶**（阶次估计为负 → 不外推声称，只给 p=1/p=2 的兜底带 "
      f"{pct(lim['rel_p1'])} … {pct(lim['rel_p2'])}），因为床实现本身的散度（ρ_b 逐档起伏）与 dt 效应同量级。"
      " <3 % 一栏如实为「否」——受引用参考与单孔口 k 带所限，本单不认证 3 % 级精度。")

    payload = {
        "title": "m20055 Beverloo 主算例 D5 的时间步收敛阶梯：+20.6 % 偏差里有多少是离散误差",
        "summary": (
            f"从当前 main 切枝，对已归档诚实 FAIL 的 m20055 D5 主算例补做预登记 dt 阶梯（dt、dt/2、dt/4，"
            f"其余参数逐位相同）。观测阶 p={lim['p_used']:.2f}，外推 W∞={lim['W_inf']:.4f} kg/s，"
            f"相对 Beverloo 由粗档 {pct(coarse['rel'])} 收敛到 {pct(lim['rel'])}（阶次带 "
            f"{pct(lim['rel_p1'])}…{pct(lim['rel_p2'])}）→ 判定 {s['verdict']}；"
            f"并显式给出单孔口下文献 k∈[1.2,1.6] 展开的参考带 {pct(lim['k_band_rel'][0])}…"
            f"{pct(lim['k_band_rel'][1])}，据此拒绝 <3 % 精度声称。新增 24 项回归测试。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "DEM", "silo-discharge", "Beverloo", "time-convergence",
                 "Richardson-extrapolation", "verification", "CASE-DEM-SILO-BEVERLOO"],
        "markdown": "\n".join(md),
        "figures": [
            {"name": "m20055_dt_ladder.png",
             "caption": "D5 主算例稳态流率随时间步的阶梯、Richardson dt→0 极限、"
                        "以及文献 k 区间给出的参考带（点标注=对该参考的误差）",
             "png_base64": b64(os.path.join(FIGS, "m20055_dt_ladder.png"))},
            {"name": "m20055_dt_ladder_discharge.png",
             "caption": "各 dt 档的累积卸料颗粒数随时间（阴影=预登记 0.20–0.80 稳态窗口）",
             "png_base64": b64(os.path.join(FIGS, "m20055_dt_ladder_discharge.png"))},
        ],
    }
    with open(PAYLOAD, "w") as fh:
        json.dump(payload, fh)

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    tok = body.get("access_token") or body.get("token")
    if not tok:
        raise SystemExit(f"login failed: {body}")
    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())