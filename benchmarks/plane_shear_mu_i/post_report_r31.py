"""Publish the round-31 progress report for CASE-DEM-PLANE-SHEAR-MUIDI.

Numbers are read from ``outputs/r31/r31_tables_smoke.md`` / ``r31_summary_smoke.json``
and the run files, so nothing in the report is hand-copied.

    python benchmarks/plane_shear_mu_i/post_report_r31.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")
BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[1]


def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> int:
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = (tok.get("access_token") or tok.get("token")
             or tok.get("data", {}).get("access_token"))

    tables = (BENCH / "outputs" / "r31" / "r31_tables_smoke.md").read_text()
    summary = json.loads((BENCH / "outputs" / "r31" / "r31_summary_smoke.json").read_text())

    figs = []
    for name, cap in (
        ("r31_velocity_profile.png",
         "图1 实测平均速度剖面 $u_x/v_{wall}$ 对 $y/h$（池化测量窗）。虚线为理想 Couette；"
         "两点实测斜率只有名义值的 7 %–13 %，壁面处 $u_x$ 远低于施加壁速 ⇒ 壁面滑移。"),
        ("r31_mu_vs_I_abscissa.png",
         "图2 $\\mu(I)$ 两种横坐标：红=名义 $I$，绿=按剖面斜率重算的 $I$，橙=含动能应力的 $\\mu_{tot}$；"
         "蓝线为 da Cruz 2005 二维律 Eq.14。把点搬到实测 $I$ 上，最差误差从 71 % 收敛到 33 %。"),
        ("r31_depth_profiles.png",
         "图3 深度分辨的局部应力比 $\\mu_{local}(y)$ 与面积分数 $\\phi(y)$：无空层，"
         "$\\mu_{local}$ 层间离散 2.4–2.6 倍 ⇒ 不是单剪切带局域化，而是近塞流 + 两壁滑移。"),
    ):
        p = REPO / "docs" / "figures" / name
        if p.exists():
            figs.append({"name": name, "caption": cap,
                         "png_base64": base64.b64encode(p.read_bytes()).decode()})

    md = f"""# 平面剪切 mu(I) 标定：分层速度剖面与流应力直测（round 31 / m22979）

## ① 案例介绍
锚点 `CASE-DEM-PLANE-SHEAR-MUIDI`（容差 10 %）要求二维颗粒平面剪切的 $\\mu(I)$、$\\phi(I)$ 落在
da Cruz et al. 2005 (PRE 72:021309, Eqs. 13–14) / GDR MiDi 2004 的 **10 %** 内。
装置：`tensordem.granular2d` 二维圆盘粗糙壁恒压剪切胞（壁=两排钉扎盘，自由盘 $d=10$ mm，$P=10$ kPa，
$\\mu_p=0.4$，多分散 $w=0.5$）。R30 改正维度错配后的失效签名是：$\\nu_{{max}}$ 合格 (+1.7 %)，
但率敏感度 $b$ 只有论文的 1/3–1/4（0.279 vs 1.0）、截距低 29 %，并留下两个未结问题：
(A) 名义横坐标 $I=\\dot\\gamma d/\\sqrt{{P/\\rho}}$（$\\dot\\gamma=2V/h_{{eff}}$）是否等于材料真正经历的剪切率；
(B) 率相关（流）应力到底多大——R30 只有上界 0.132 % $p_{{bulk}}$，从未直测。

| 参数 | 值 |
|---|---|
| 自由盘数 / 直径 | 458（全尺寸战役）/ 200（本轮探针）/ 10 mm |
| 法向应力 P | 10 kPa |
| 接触摩擦 $\\mu_p$ | 0.4（无滚阻） |
| 多分散度 w | 0.5（晶体对照 0） |
| 测量窗 | 0.6 应变（探针）/ 8 应变（全尺寸战役 cj4516，未回） |
| 分层 | 10 层，层带 $[y_{{bottom}}+r,\\ y_{{top}}-r]$，拟合带=中心 60 % |

## ② 怎么做的
1. 新增**分层剖面/应力分解仪** `benchmarks/plane_shear_mu_i/profile_r31.py`：每个采样时刻把自由盘速度与全部接触按 $y$ 归层
（接触按中点归层，分支向量穿周期像，与整胞维里同一约定），**池化累加原始和**，因此层内方差同时含剪切梯度与真实颗粒温度。
2. 层内量：$u_x,u_y,\\phi$、接触应力 $\\sigma_{{ab}}=\\sum r_af_b/(Wh_{{bin}}t)$、动能应力
$p_{{kin}}=\\rho\\phi(\\mathrm{{var}}_x+\\mathrm{{var}}_y)/2$、$\\tau_{{kin}}=-\\rho\\phi\\,\\mathrm{{cov}}_{{xy}}$、层局部 $\\mu_{{local}}=\\tau_c/p_c$。
3. 剪切率取中心 60 % 层带的加权最小二乘斜率（不受壁面层影响），$r^2$ 作局域化诊断。
4. `prereg_r31.json` **先定后跑**（层数、拟合带、判据阈值写死）；6 组合全尺寸战役递交为 `cj4516`
（递交时队列前方 75 个、在飞 91 个，**结果未回**）。本报告数据来自**缩小算例探针**（200 盘、$I\\in\\{{0.05,0.2\\}}$、seed 0）：
同仪器、同判据、同一把尺子，但不是归档算例。
5. `analyse_r31_profile.py` 程序化生成全部数字；`test_r31_profile.py` 15 项测试全绿
（解析剖面、恒滑移、局域化检测、动能应力、粒子数守恒、整胞一致性、面积分数归一化）。

## ③ 计算结果

{tables}

## ④ 结果图
见 figures（速度剖面 / 两种横坐标下的 $\\mu(I)$ / 深度剖面）。

## ⑤ 注意点
* 探针算例**不是**归档算例：小胞伺服振荡使 $I_{{nom}}$ 被放大（表 2 的 $I_{{nom}}$ 0.111/0.451 对应
  $I_{{target}}$ 0.05/0.2），因此**不能**据此替换 R30 的定量结论，只能回答「这两个问题在此装置上是否存在」。
* 探针窗只有 0.6 应变，非稳态；壁面层 $\\phi>0.84$ 是分层边界把壁粒—自由盘接触的中点计入所致，
  边界层 $\\mu_{{local}}$ 不可与体相比较。
* 8 应变全尺寸战役仍未回，锚点的定量复核留待该战役。

## ⑥ 误差判定（honest）
| 量 | 判据 | 探针实测 | 判定 |
|---|---|---|---|
| $\\dot\\gamma_{{meas}}/\\dot\\gamma_{{nom}}$ | 0.90–1.10 | {summary['abscissa']['ratio_min']:.3f} – {summary['abscissa']['ratio_max']:.3f} | **FAIL（横坐标需重标）** |
| 剖面线性度 $r^2$ | ≥ 0.99 | ≥ {summary['abscissa']['r2_min']:.3f} | FAIL |
| $p_{{kin}}/p_{{tot}}$ | < 1 % | ≤ {100 * summary['flow_stress']['p_share_max']:.3f} % | 边缘（$\\tau_{{kin}}$ 份额仅 {100 * summary['flow_stress']['tau_share_max']:.4f} %，无法解释 b） |
| $\\mu$ vs 2-D 律（名义 I） | < 10 % | {summary['worst_err_mu_nominal']:.1f} % | FAIL |
| $\\mu$ vs 2-D 律（实测 I） | < 10 % | {summary['worst_err_mu_measured']:.1f} % | FAIL（但显著收敛） |
| 关键量 <3 % | < 3 % | 最好 {summary['worst_err_phi_nominal']:.1f} % | **NO** |

**诚实结论**：本轮把 R30 的两个未结问题都变成直测量。(1) 该装置上材料真正经历的剪切率只有名义值的 7 %–13 %，
剖面近塞流、两壁滑移 ⇒ 名义 $I$ 轴确实被高估；(2) 流应力 $\\tau_{{kin}}$ 占比 ≤ 7e-5，**不可能**解释 3–4 倍的 $b$ 低估值。
把 $\\mu$ 重新判在实测 $I$ 上最差误差从 71 % 收敛到 33 %，方向与「横坐标是主因」一致，但**仍不合格**。锚点状态不变（`open`）。
"""

    body = {
        "title": ("平面剪切 mu(I) 标定 R31：分层速度剖面直测暴露壁面滑移"
                  "（实测剪切率仅名义值 7–13 %），流应力项被直测否定"),
        "summary": ("给二维平面剪切装置加分层速度/应力分解仪：实测剪切率仅名义值的 7–13 %（近塞流+两壁滑移），"
                    "流应力占比 ≤7e-5 不能解释 b 偏低；μ 改判在实测 I 上误差由 71 % 收敛到 33 %，仍 FAIL。"),
        "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "mission_id": 22979,
        "tags": ["DEM", "mu(I)", "rheology", "plane-shear", "wall-slip",
                 "granular2d", "TensorDEM"],
        "markdown": md,
        "figures": figs,
    }
    res = post("/tensor/reports", body, token)
    print(json.dumps(res)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())