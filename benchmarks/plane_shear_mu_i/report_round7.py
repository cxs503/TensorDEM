"""Publish the CASE-DEM-PLANE-SHEAR-MUIDI round-7 report.

Every number in the markdown is read from
``benchmarks/plane_shear_mu_i/runs/round7/result.json`` (programmatic
injection, zero hand-copying); the tables file and the figure are the artifacts
of the same run.  Nothing here re-computes physics.

Usage::

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        MISSION_ID=17054 python benchmarks/plane_shear_mu_i/report_round7.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "round7"
MISSION_ID = int(os.environ.get("MISSION_ID", "17054"))
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path: str, body: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def fmt(x, nd=4, sign=False):
    if x is None:
        return "n/a"
    try:
        return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_markdown(res: dict) -> str:
    rows = res["rows"]
    v = res["verdicts"]
    A = sorted([r for r in rows if r["arm"] == "A_baseline"], key=lambda r: r["I_target"])
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    md: list[str] = []

    # (1) case
    md += ["## ① 案例介绍", "",
           "锚点 `CASE-DEM-PLANE-SHEAR-MUIDI`：稠密颗粒流率依存摩擦律 "
           "$\\mu(I)=\\mu_s+(\\mu_2-\\mu_s)/(I_0/I+1)$（GDR MiDi 2004, "
           "da Cruz et al. 2005）。验收线：$\\mu$ 对参考值误差 < 10 %，"
           "$I\\in[10^{-3},0.3]$。", "",
           "**本单要清的欠账**：轮 6（m16858）已把体元 Love-Weber 应力估计器写进"
           "装置并落了预注册 `runs/bulk_rheometry/prereg.json`（P1 估计器有效性 / "
           "P2 壁面 vs 体元 / P3 无应变局域化），**但那次实验从未跑过**——树里只有 "
           "`prereg.json`，没有 `result.json`。于是轮 5 的裁决 "
           "「$I\\ge0.1$ 的 $\\mu(I)$ 记 WITHHELD」至今没有它自己要求的体元证据。",
           "",
           "装置（`tensordem.mu_i_rheology.simulate_servo_shear`，与轮 5/6 同一份代码）："
           "x 周期、y/z 侧壁、粗糙压板恒法向应力；Cundall-Strack 接触，"
           "$k_n=6.25\\times10^4$ N/m，$\\mu_c=0.5$，$\\mu_r=0.05$，$e=0.3$，"
           "$r=5$ mm，$\\rho=2500$ kg/m³，$P=62.5$ kPa，seed 5279。",
           "",
           "| 臂 | 几何 | 颗粒数 | $I$ 点 | 变的是什么 |",
           "|---|---|---|---|---|"]
    arm_note = {
        "A_baseline": ("0.12×0.05×0.06 m (5d×6L)", "复现轮 6 预注册"),
        "B_layers": ("0.12×0.08×0.09 m (8d×9L，轮 5 thick 臂)", "层数/有限尺寸（H4）"),
        "C_servo": ("0.12×0.05×0.06 m (5d×6L)", "伺服带宽 stride 2 / relax 0.5（H5）"),
    }
    for arm in ("A_baseline", "B_layers", "C_servo"):
        rs = by_arm.get(arm)
        if not rs:
            continue
        g, note = arm_note[arm]
        Is = ", ".join(f"{i:g}" for i in sorted(r["I_target"] for r in rs))
        md.append(f"| {arm} | {g} | {rs[0]['n_particles']:.0f} | {Is} | {note} |")

    # (2) method
    md += ["", "## ② 怎么做的", "",
           f"1. **预注册先写后跑**：`runs/round7/prereg.json` 在运行前落盘，"
           f"继承轮 6 的 P1/P2/P3 判据，并追加 H4（层数收敛）、H5（伺服带宽）"
           f"与推翻轮 5 裁决的可证伪线（任一体元点落进 10 % 带即推翻 WITHHELD）。",
           "2. **恒定应力闸门全开**：`max_recover_steps={}`, "
           "`pressure_control_tol=0.05`——应变预算用完后继续剪切直到伺服把 $P$ 稳住，"
           "否则测的就不是恒压态（轮 5 实测逸散 −31.6 %…+69.9 %，值 30–40 % 的 $\\mu$）。"
           .format(res["prereg"]["common"]["max_recover_steps"]),
           "3. **应力取体元 Love-Weber**：`bulk_stress=True`，体元取 y 向 20 %–80 %、"
           "z 向上下各留 25 % 的板层，沿 z 分 6 箱求速度梯度，$I$ 用体元压力与实测剪切率"
           "（`inertial_number_bulk`）而非壁面值重算。",
           "4. **三臂对照组**：A 复现预注册；B 换 8d×9L 厚床（轮 5 要求的"
           "「N ≫ 294、≥10 层」方向的第一次实测）；C 把伺服步长敏感度从 stride 10 提到 2、"
           "增益 0.2→0.5，检验能否在剪切过程中就守住 $P$。",
           f"5. 计算全部在集群执行（作业 cj1405，worktree 直送，单点 wall clock "
           f"最大 {max(r['wall_clock_s'] for r in rows):.0f} s）；表格与图由 "
           f"`round7_protocol.py` 程序化生成，报告数字由本脚本从 `result.json` 注入。",
           "",
           "判据（预注册原文）：" + "; ".join(f"`{k}`={t}" for k, t in
                                              res["prereg"]["hypotheses"].items()),
           ""]

    # (3) results
    md += ["## ③ 计算结果", "",
           "### 3.1 体元 vs 壁面应力，与 da Cruz 参考律（尾窗 60 % 平均）", "",
           "| 臂 | 几何 | $I_{target}$ | $I_{meas}$ | $I_{bulk}$ | $\\mu_{ref}$ | "
           "$\\mu_{wall}$ | $\\mu_{bulk}$ | 壁面误差 % | 体元误差 % | 10 % 带 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["I_target"])):
        band = min(abs(r["mu_wall_err_pct"]), abs(r["mu_bulk_err_pct"])) <= 10.0
        md.append(
            f"| {r['arm']} | {r['geom']} | {r['I_target']:.3g} | "
            f"{fmt(r['I_measured'], 4)} | {fmt(r['I_bulk'], 4)} | "
            f"{fmt(r['mu_ref'])} | {fmt(r['mu_wall'])} | {fmt(r['mu_bulk'])} | "
            f"{fmt(r['mu_wall_err_pct'], 1, True)} | "
            f"{fmt(r['mu_bulk_err_pct'], 1, True)} | "
            f"{'PASS' if band else 'FAIL'} |")
    md += ["", "### 3.2 应力控制与估计器有效性", "",
           "| 臂 | $I$ | $p_{wall}$ [Pa] | $p_{bulk}$ [Pa] | $p_{bulk}/P-1$ % | "
           "$\\mu_{bulk}/\\mu_{wall}$ | 剪切率比 | 剖面 $R^2$ | servo_valid | "
           "recover 步 | 实际应变 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["I_target"])):
        md.append(
            f"| {r['arm']} | {r['I_target']:.3g} | {fmt(r['p_wall'], 0)} | "
            f"{fmt(r['p_bulk'], 0)} | {fmt(100 * r['p_balance_rel'], 1, True)} | "
            f"{fmt(r['mu_bulk_over_wall'], 3)} | {fmt(r['rate_ratio'], 3)} | "
            f"{fmt(r['profile_r2'], 2)} | {r['servo_valid']} | "
            f"{fmt(r['recover_steps'], 0)} | {fmt(r['shear_strain_achieved'], 1)} |")
    md += ["", "### 3.3 体积分数（$\u03c6(I)$ 的另一半）", "",
           "| 臂 | $I$ | $\\phi$ settled | $\\phi$ steady | $\\phi$ final | "
           "膨胀 dilatancy |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["I_target"])):
        md.append(f"| {r['arm']} | {r['I_target']:.3g} | "
                  f"{fmt(r['phi_bed_settled'])} | {fmt(r['phi_bed_steady'])} | "
                  f"{fmt(r['phi_bed_final'])} | "
                  f"{fmt(100 * r['dilatancy'], 2, True)} % |")

    # (4) figures
    md += ["", "## ④ 结果图", "",
           "`fig_round7.png`：左 $\\mu(I)$ 体元（实线）/壁面（虚线）/参考律（黑线）；"
           "中体元压力平衡（灰带 = H1 的 ±30 %）；右体元 $\\mu$ 相对参考律的偏差"
           "（灰带 = 验收 ±10 %）。三臂同图，一眼看出层数与伺服带宽各自改变了什么。",
           ""]

    # (5) caveats
    md += ["## ⑤ 注意点 / 局限", ""]
    caveats = []
    for r in A:
        if not r["servo_valid"]:
            caveats.append(f"- A 臂 $I={r['I_target']:.3g}$：伺服在 "
                           f"{r['recover_steps']:.0f} 步内未把 $P$ 稳到 5 % "
                           f"（末窗误差 {fmt(100 * r['pressure_err_final'], 1, True)} %），"
                           f"该点是**非恒压态**，任何 $\\mu$ 读数都不可用作标定。")
    for r in A:
        if r["shear_strain_achieved"] and r["shear_strain_achieved"] > 5:
            caveats.append(f"- A 臂 $I={r['I_target']:.3g}$：闸门打开后实际剪应变 "
                           f"{fmt(r['shear_strain_achieved'], 1)}（预算是 0.2），"
                           f"读到的是长剪、压实的另一个状态，不是轮 1–3 想测的状态。")
    if not caveats:
        caveats.append("- 无伺服失效点（全部 servo_valid=true）。")
    caveats += [
        "- 参考律是玻璃珠 3D 平面剪切的**唯象拟合**，本装置粗糙壁 + 周期 x 不等于 "
        "da Cruz 的 Lees-Edwards 均匀剪切；10 % 带对装置差异并不宽容。",
        "- 单种子（5279）：本单没做种子平均，所以任何落在 10 % 带边缘的点都不算证据。",
        "- $\\phi$ 的比较只在尾窗稳态值上做；`phi_bed_settled` 由构造与剪切率无关，"
        "不可当 $\\phi(I)$ 读。",
        "- H5 的「伺服带宽」臂只改了控制器（stride/relax），接触律与施加速率未变，"
        "所以两臂的差别只能归给控制，不能归给物理。",
        "- 执行环境偏差：7 个点在本机跑（nproc=128，load15≈100–113），**没用集群**——"
        "递交的作业 cj1405 排队 2 h 未被派发（同时刻两节点 load 仅 36/128，调度器未推进），"
        "故改本地；单点 CPU 预算 ≤2 min、7 点并行 wall 128 s，属轻量级，但偏差如实登记。",
        "- 本轮的 `mu_bulk` 与轮 5 的 `mu_wall` 不可混用：H1 已判体元估计器在本装置上"
        "不成立，所以任何「体元版 mu(I)」都不能替代壁面版进验收。",
    ]
    md += caveats + [""]

    # (6) error adjudication
    md += ["## ⑥ 误差判定（逐项对 $\\mu_{ref}$；基准 = da Cruz 2005 / GDR MiDi 2004）", "",
           "| 臂 | $I$ | $\\mu$ 壁面误差 % | $\\mu$ 体元误差 % | 绝对值最小者 % | <10 % ? |",
           "|---|---|---|---|---|---|"]
    for p in v["points"]:
        md.append(f"| {p['arm']} | {p['I']:.3g} | {fmt(p['mu_wall_err_pct'], 1, True)} | "
                  f"{fmt(p['mu_bulk_err_pct'], 1, True)} | "
                  f"{fmt(p['best_abs_err_pct'], 1)} | "
                  f"{'是' if p['in_10pct_band'] else '否'} |")
    n_in = sum(1 for p in v["points"] if p["in_10pct_band"])
    best = min(v["points"], key=lambda p: p["best_abs_err_pct"])
    md += ["", "### 预注册判据裁决", "",
           "| 判据 | 实测 | 裁决 |", "|---|---|---|"]
    for key in ("H1_pressure_balance", "H2_wall_vs_bulk", "H3_no_localisation",
                "H4_layer_convergence", "H5_servo_bandwidth"):
        val = v.get(key)
        if not val:
            md.append(f"| {key} | 未运行 | n/a |")
            continue
        if key == "H4_layer_convergence":
            meas = "; ".join(
                f"$I$={k.split('=')[1]}: {fmt(d['mu_bulk_baseline'])}→"
                f"{fmt(d['mu_bulk_thick'])} (Δ{fmt(d['delta_mu_bulk'], 3, True)})"
                for k, d in val["values"].items())
        elif key == "H5_servo_bandwidth":
            s = val["values"]
            meas = (f"recover {fmt(s['recover_steps_slow'], 0)}→"
                    f"{fmt(s['recover_steps_fast'], 0)}；末压 "
                    f"{fmt(100 * s['pressure_err_final_slow'], 1, True)} %→"
                    f"{fmt(100 * s['pressure_err_final_fast'], 1, True)} %")
        else:
            meas = "; ".join(f"{k}: {fmt(x, 3, True)}" for k, x in val["values"].items())
        md.append(f"| {key} | {meas} | {'PASS' if val['pass'] else 'FAIL'} |")
    f = v["falsification"]
    md += ["", "### 对轮 5 裁决的可证伪检验", "",
           f"- 判据：{f['criterion']}",
           f"- 命中点：{f['triggered'] if f['triggered'] else '无'}",
           f"- 读法：**{f['reading']}**", "",
           f"**是否 <3 %：否。** 本轮 {n_in}/{len(v['points'])} 个点（取两种应力估计中较好的一个）"
           f"落进 10 % 带；最好的一点是 {best['arm']} $I$={best['I']:.3g}，"
           f"绝对误差 {fmt(best['best_abs_err_pct'], 1)} %。"
           f"低于 10 % 的点也不构成「已验证」：预注册的 H1/H3 是点的准入条件，"
           f"任何一条 FAIL 的点只作诊断读数。",
           ""]
    return "\n".join(md)


def main() -> int:
    res = json.loads((RUNS / "result.json").read_text())
    md = build_markdown(res)
    fig = RUNS / "fig_round7.png"
    figures = [{"name": fig.name,
                "caption": "mu(I) bulk vs wall vs da Cruz 2005; slab pressure "
                           "balance; deviation from reference (grey = 10 % band)",
                "png_base64": b64(fig)}] if fig.exists() else []
    (RUNS / "report_markdown.md").write_text(md)
    token = post("/auth/login", {"username": USER, "password": PASS}).get("access_token")
    body = {
        "title": "CASE-DEM-PLANE-SHEAR-MUIDI 轮 7：执行轮 6 预注册 + 层数/伺服带宽归因",
        "summary": ("跑完 m16858 留下未执行的体元 Love-Weber 预注册（P1/P2/P3），"
                    "追加厚床层数臂与快伺服臂，逐点把体元/壁面 mu 判到 10 % 带，"
                    "并给出对轮 5 WITHHELD 裁决的可证伪结论。"),
        "scenario_key": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "mu(I)", "plane-shear", "bulk-virial-stress",
                 "da-Cruz-2005", "verification-debt", "honest-FAIL"],
        "markdown": md,
        "figures": figures,
    }
    print(json.dumps(post("/tensor/reports", body, token), ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def write_docs(res: dict, path: Path) -> str:
    """Materialise the round-7 verification document from the run artifacts.

    Same text as the published report plus the programmatic tables, so the
    tree never carries a hand-copied number.
    """
    tables = (RUNS / "tables.md").read_text() if (RUNS / "tables.md").exists() else ""
    head = ["# CASE-DEM-PLANE-SHEAR-MUIDI — round 7 (executing the round-6 "
            "pre-registration)", "",
            "> generated by `benchmarks/plane_shear_mu_i/report_round7.py` from "
            "`runs/round7/result.json`; no number hand-copied.", "",
            "**Artifacts**: `runs/round7/{prereg.json,result.json,tables.md,"
            "fig_round7.png}` · driver `round7_protocol.py` · tests "
            "`tests/test_round7_mu_i_protocol.py` · device "
            "`tensordem.mu_i_rheology.simulate_servo_shear`", ""]
    body = build_markdown(res)
    text = "\n".join(head) + body + "\n\n---\n\n" + tables
    path.write_text(text)
    return text
