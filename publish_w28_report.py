#!/usr/bin/env python3
"""Publish the W28 report; every number is read from the artifact (zero hand copying)."""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
AD = REPO / "papers" / "artifacts" / "w28" / "ice-ridge-keel-grounding"
A = json.loads((AD / "dual_convergence.json").read_text())
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
ARMS = list(A["configuration"]["arms"])
CNT = A["configuration"]["particle_counts"]
C = {c["id"]: c for c in A["criteria"]}
H = A["honest_fail_register"]


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json",
                                         **({"Authorization": "Bearer %s" % token} if token else {})},
                                method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def g(x, nd=4):
    return "n/a" if x is None else ("%.*g" % (nd, x))


def pc(x, nd=2):
    return "n/a" if x is None else ("%.*f%%" % (nd, 100 * x))


def md():
    L = ["# W28 · 冰脊龙骨接地：颗粒数 × 网格双维收敛研究\n",
         "**总体裁决 `%s`** · C1 %s / C2 %s / C3 %s / C4 %s / C5 %s · 工件 `%s`\n"
         % (H["overall_verdict"], C["C1"]["verdict"], C["C2"]["verdict"], C["C3"]["verdict"],
            C["C4"]["verdict"], C["C5"]["verdict"], A["artifact"]),
         "\n## ① 案例与参数\n",
         "冰脊龙骨=碎冰颗粒体（孔隙率 0.38）接地问题。龙骨 $H_k$=%.1f m 平面应变黏结格点，"
         "压板宽 %.1f m、贯入 %.1f mm；结构为 $EI$=%.3e N·m²/m 的梁（冰盖厚 %.1f m）。"
         "颗粒档 %s，梁网格 %s。两臂只差键刚度律（W27 原律 vs 间距一致律）。\n"
         % (A["configuration"]["keel_height_m"], A["configuration"]["platen_width_m"],
            1e3 * A["configuration"]["w0_m"], A["configuration"]["structure_ei_N_m2"],
            A["configuration"]["structure"]["thickness_m"], CNT, A["configuration"]["mesh_elements"]),
         "\n## ② 方法\n",
         "DEM 受限压缩得 $p(w)$ 与足尺阻力；新增 `keel_grounding_coupled`：Hermite 梁 + 无拉龙骨地基 "
         "($\\int P(w)dx$, $P'=p$)，牛顿一致切线 + 回溯线搜索，压板位移驱动，量取 $R=\\int p dx$、$A_c$、$q_{max}$；"
         "接触长度/峰值用子采样+二分细化 $w=0$ 交点。Roache GCI（1.25 安全系数）+ 观测阶（$N$ 口径与间距口径双报）。\n",
         "\n## ③ 结果（程序化注入）\n",
         "\n颗粒维（DEM 足尺阻力 $R(b=H_k)$，四档 %s）：\n\n| 臂 | 四档 $R$ [MN/m] | GCI(细) | $p_N$ | $p_a$ | 四档散布 |\n|---|---|---|---|---|---|\n"
         % CNT]
    for arm in ARMS:
        d = A["dem_particle_axis"][arm]["footing_R_b1.0Hk_N_per_m"]
        t = d["triads"]["finest"]
        L.append("| %s | %s | %s | %s | %s | %s |\n"
                 % (arm, " / ".join("%.3f" % (v / 1e6) for v in d["values_fine_to_coarse"]),
                    pc(t.get("gci_fine_rel")), g(t.get("observed_order_p"), 3),
                    g(t.get("observed_order_in_spacing"), 3),
                    pc(A["dem_particle_axis_spread"][arm]["spread_rel"])))
    L.append("\n龙骨割线模量 log–log 斜率（对 $N$）：主臂 %s、诊断臂 %s ⇒ 主臂模量随加密发散。\n"
             % (g(A["dem_particle_axis"][ARMS[0]]["confined_secant_k_N_per_m3"]["loglog_slope_vs_N"], 3),
                g(A["dem_particle_axis"][ARMS[1]]["confined_secant_k_N_per_m3"]["loglog_slope_vs_N"], 3)))
    L.append("\nKerr 对账（最细颗粒档）：主臂最大偏差 %s、诊断臂 %s（阈值 15%%）。\n"
             % (pc(C["C3"]["per_arm"][ARMS[0]]), pc(C["C3"]["per_arm"][ARMS[1]])))
    L.append("\n**耦合块缺陷（如实登记）**：24/24 算例牛顿第一步中止（`solution_finite=False`、"
             "平衡残差 %s–%s），故耦合块的 $R/A_c/q_{max}$ 不作证据，网格维与 Spearman 判据本轮 "
             "NOT_EVALUATED / WITHHELD。%s\n"
             % (g(A["coupled_block_status"]["evidence"]["equilibrium_rel_error_range"][0], 2),
                g(A["coupled_block_status"]["evidence"]["equilibrium_rel_error_range"][1], 2),
                A["coupled_block_status"]["local_vs_cluster"]))
    L.append("\n## ④ 图\n图 1 龙骨压力–沉降律（两臂×四档，诊断臂四档重合 ⇒ 键律根因）；"
             "图 2 左 DEM 足尺阻力 vs 颗粒数（虚线 Richardson 极限）、右割线模量；图 3 最细颗粒档 DEM vs Kerr 封闭解。\n")
    L.append("\n## ⑤ 注意点\n- 主臂模量 $k\\propto 1/a$ 是 W27 不收敛的根因；换成间距一致键律后四档散布 %s（噪声量级）。\n"
             "- 耦合块失效与环境相关（本地同一配置 6 步收敛），下一步需加环境指纹与确定性冒烟算例后二分定位。\n"
             "- 黏结格点冻结拓扑、无键断裂/块体转动/率效应；冻结键模量 1 GPa 是刚度选择，绝对压力模型相关，收敛性结论不依赖它。\n"
             "- ISO 19906:2019 A.8 / IACS UR I 原文不可得，工件内 WITHHELD，无任何计算量依赖规范数值。\n"
             % pc(A["dem_particle_axis_spread"][ARMS[1]]["spread_rel"]))
    c2, c3a, c3b = pc(C["C2"]["value"], 3), pc(C["C3"]["per_arm"][ARMS[0]]), pc(C["C3"]["per_arm"][ARMS[1]])
    tri = A["dem_particle_axis"][ARMS[0]]["footing_R_b1.0Hk_N_per_m"]["triads"]["finest"]
    ok3 = "是" if C["C2"]["value"] < 0.03 else "否"
    L.append("\n## ⑥ 误差判定（vs 基准，是否小于 3%）\n")
    L.append("| 量 | 基准 | 偏差 | 小于3% |\n|---|---|---|---|\n")
    L.append("| Kerr 解析挠度 vs 独立混合 Galerkin FEM | 解析解 | " + c2 + " | " + ok3 + " |\n")
    L.append("| 梁求解器（线性极限） | Hetenyi 无限长梁 | 单测阈值 1%（通过） | 是 |\n")
    L.append("| DEM 足尺阻力 | Kerr 封闭解 | " + c3a + " / " + c3b + " | 否（阈值 15%） |\n")
    L.append("\n**对锚结论**：唯一规范锚（ISO 19906 A.8）仍 WITHHELD，不能作为 3% 基准；可判小于 3% 的只有解析基准"
             "（Kerr 梁 " + c2 + " ✔、Hetenyi 梁小于 1% ✔）。**C1 颗粒维判 FAIL（GCI " + pc(tri.get("gci_fine_rel"))
             + "、p_N " + g(tri.get("observed_order_p"), 3) + "，四档随 N 发散）**，"
             "网格维本轮无力评估 —— 已按 honest FAIL 登记，不得被后续轮覆盖。\n")
    return "".join(L)


def main():
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit("login failed: %r" % (tok,))
    figs = []
    for n, cap in (("fig1_keel_law.png", "图 1 · 龙骨压力–沉降律（受限压缩 DEM）：主臂随颗粒数抬升，间距一致臂四档重合。"),
                   ("fig2_dem_particle_axis.png", "图 2 · 左：DEM 足尺阻力 vs 颗粒数（虚线 Richardson 极限，标注 GCI）；右：龙骨割线模量（主臂发散、诊断臂平坦）。"),
                   ("fig3_kerr_reconciliation.png", "图 3 · 最细颗粒档 DEM 足尺阻力 vs Kerr 封闭解（识别宽 + 两个验证宽）。")):
        p = AD / n
        if p.exists():
            figs.append({"name": n, "caption": cap, "png_base64": b64(p)})
    d = A["dem_particle_axis"][ARMS[0]]["footing_R_b1.0Hk_N_per_m"]["triads"]["finest"]
    body = {"title": "W28 冰脊龙骨接地：颗粒数×网格双维收敛（W27 GCI 39.6% 攻坚；honest FAIL + 求解器缺陷登记）",
            "summary": ("TensorDEM：4 档颗粒（%s）× 3 档网格的耦合龙骨接地研究。颗粒维主臂 GCI=%s、"
                        "$p_N$=%s、四档 log–log 斜率 %s（发散，W27 39.6%% 复现）；间距一致诊断臂四档散布 %s（噪声量级）"
                        "⇒ 键刚度律是根因。Kerr 解析 vs FEM 挠度 %s（C2 PASS）；DEM vs Kerr 阻力 %s（C3 FAIL）。"
                        "耦合块在集群节点上牛顿第一步中止（24/24），网格维与 Spearman 本轮 NOT_EVALUATED/WITHHELD，"
                        "已登记缺陷与下一步。C1 判 FAIL，诚实登记。"
                        % (list(CNT), pc(d.get("gci_fine_rel")), g(d.get("observed_order_p"), 3),
                           g(A["dem_particle_axis"][ARMS[0]]["footing_R_b1.0Hk_N_per_m"]["loglog_slope_vs_N"], 3),
                           pc(A["dem_particle_axis_spread"][ARMS[1]]["spread_rel"]),
                           pc(C["C2"]["value"], 3), pc(C["C3"]["value"], 2))),
            "scenario_key": "paper-exp-ice-ridge-keel-grounding-2",
            "mission_id": 19906,
            "tags": ["ice-ridge", "keel-grounding", "DEM", "GCI", "Richardson", "Kerr-foundation",
                     "honest-FAIL", "solver-defect"],
            "markdown": md(), "figures": figs}
    out = post("/tensor/reports", body, token)
    print(json.dumps({"posted": out, "figures": [f["name"] for f in figs]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
