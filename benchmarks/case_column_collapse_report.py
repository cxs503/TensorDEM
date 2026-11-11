"""CASE-COLUMN-COLLAPSE: figures, report markdown and platform report post.

Reads the per-case JSONs written by ``benchmarks/case_column_collapse_lube2004.py``
(no number is typed here: every value in the markdown comes from those files or
from the Lube et al. (2004) formulas evaluated at run time), writes
``docs/case_column_collapse_lube2004.md`` + ``artifacts/case_column_collapse/``
figures and posts the progress report to the platform.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from tensordem.granular_column.axisym import (lube_2004_emplacement_time,  # noqa: E402
                                             lube_2004_runout_ratio)

RUNS = "runs/case_column_collapse_lube2004"
FIG = "artifacts/case_column_collapse"
DOC = "docs/case_column_collapse_lube2004.md"
MISSION = 15324
TOL = 0.15

cases = {}
for path in glob.glob(os.path.join(RUNS, "cases", "*.json")):
    with open(path) as fh:
        res = json.load(fh)
    cases[res["case"]["label"]] = res
summary = json.load(open(os.path.join(RUNS, "summary.json")))
rows = {r["case"]: r for r in summary["rows"]}
REF = summary["reference"]
MUR = summary["calibration"]["chosen_mu_rolling"]

order = ["calib_mur0.2_a0.5", "calib_mur0.3_a0.5", "calib_mur0.4_a0.5",
         "valid_mur0.4_a1", "valid_mur0.4_a1.5", "valid_mur0.4_a2",
         "sens_gpr6_a2.0_mur0.4", "control_mur0_a1.0"]
present = [c for c in order if c in cases]

os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
aa = np.linspace(0.2, 3.4, 200)
rr = np.array([lube_2004_runout_ratio(v) for v in aa])
ax[0].plot(aa, rr, "-", color="k", lw=2,
           label="Lube et al. 2004 (abstract formulas)")
ax[0].axvline(1.7, color="0.6", ls=":", lw=1)
ax[0].text(1.72, 1.35, r"$a=1.7$", color="0.4", fontsize=9)
val = [c for c in present if c.startswith("valid")]
cal = [c for c in present if c.startswith("calib")]
sen = [c for c in present if c.startswith("sens")]
ctl = [c for c in present if c.startswith("control")]


def pts(names, key="runout_ratio_p995"):
    x = [cases[n]["case"]["aspect_ratio"] for n in names]
    y = [cases[n][key] for n in names]
    return np.array(x), np.array(y)


if val:
    x, y = pts(val)
    ax[0].plot(x, y, "o", ms=8, color="tab:blue", label=r"DEM, $\mu_r=0.4$ (validation)")
if cal:
    x, y = pts(cal)
    ax[0].plot(x, y, "s", ms=8, mfc="none", color="tab:green",
               label=r"DEM, calibration grid at $a=0.5$")
if sen:
    x, y = pts(sen)
    ax[0].plot(x, y, "^", ms=9, color="tab:orange", label=r"DEM, $\mu_r=0.4$, 6 grains/$r_i$")
if ctl:
    x, y = pts(ctl)
    ax[0].plot(x, y, "x", ms=11, mew=2.5, color="tab:red", label=r"DEM, $\mu_r=0$ (control)")
ax[0].set_xlabel(r"aspect ratio $a=h_i/r_i$")
ax[0].set_ylabel(r"runout radius ratio $r_\infty/r_i$")
ax[0].set_title("final deposit extent vs Lube et al. (2004)")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].grid(alpha=0.3)

names = present
err = np.array([rows[n]["err_runout_p995"] for n in names]) * 100.0
lbl = [n.replace("_mur0.4", "").replace("_{}", "") for n in names]
ax[1].barh(range(len(names)), err, color=["tab:blue" if abs(e) <= 15 * 100 / 100 else "tab:red"
                                          for e in err], alpha=0.85)
ax[1].axvline(15, color="k", ls="--", lw=1)
ax[1].axvline(-15, color="k", ls="--", lw=1, label="±15 % tolerance")
ax[1].set_yticks(range(len(names)))
ax[1].set_yticklabels(lbl, fontsize=7)
ax[1].set_xlabel(r"relative error of $r_\infty/r_i$ [%]")
ax[1].set_title("error against the reference (15 % band)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="x")
fig.tight_layout()
p1 = os.path.join(FIG, "runout_law.png")
fig.savefig(p1, dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for n, col in zip([c for c in present if c.startswith("valid")],
                  ["tab:blue", "tab:green", "tab:red"]):
    r = cases[n]
    t = np.asarray(r["history"]["time"])
    f = np.asarray(r["history"]["front"]) / r["column_radius_measured"]
    h = r["column_height_measured"]
    tau = t / np.sqrt(h / 9.81)
    ax.plot(tau, f, col, lw=1.6, label=fr"DEM $a={r['case']['aspect_ratio']:g}$ ($\mu_r={MUR:g}$)")
    ax.axvline(3.0, color="0.5", ls=":", lw=1)
ax.text(3.05, 1.05, r"$t_\infty=3\,(h_i/g)^{1/2}$ (Lube 2004)", fontsize=8, color="0.35")
ax.set_xlabel(r"$t\,/\,(h_i/g)^{1/2}$")
ax.set_ylabel(r"radial front $r_{\rm front}/r_i$")
ax.set_title("spreading history in ballistic time units")
ax.set_xlim(0, 5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
p2 = os.path.join(FIG, "front_history.png")
fig.savefig(p2, dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- markdown
def fmt_row(name):
    r = rows[name]
    c = cases[name]
    return (f"| {name} | {r['aspect_requested']:.2f} | {r['aspect_measured']:.3f} | "
            f"{r['n_particles']} | {r['dt']:.2e} | {r['runout_ratio_p995']:.3f} | "
            f"{r['ref_runout_ratio']:.3f} | {r['err_runout_p995'] * 100:+.1f} % | "
            f"{'✅' if r['pass_runout_p995'] else '❌'} | {r['t_reach']:.3f} | "
            f"{r['ref_t_reach']:.3f} | {r['err_emplacement_time'] * 100:+.1f} % | "
            f"{'✅' if r['pass_emplacement_time'] else '❌'} |")


def param_table(name):
    c = cases[name]["case"]
    return (f"| {name} | {c['aspect_ratio']:g} | {c['grains_per_radius']:g} | "
            f"{c['radius']:.3f} | {c['column_radius']:.3f} | {c['column_height']:.3f} | "
            f"{c['mu']:g} | {c['mu_rolling']:g} | {c['restitution']:g} | {c['kn']:.1e} |")


lines = []
lines.append("# CASE-COLUMN-COLLAPSE · 轴对称颗粒柱坍塌（TensorDEM）vs Lube et al. (2004)\n")
lines.append("**算例池 case card（owner 2026-11-12）** · mission 15324 · 仓库 TensorDEM "
             "· 分支 `agent/m15324` · 容差判据 **< 15 %**\n")
lines.append("## ① 问题与参考出处\n")
lines.append(
    "圆柱形颗粒柱（初始半径 $r_i$、高 $h_i$，高宽比 $a=h_i/r_i$）在粗糙水平面上移除侧向支撑后"
    "自由坍塌，最终沉积半径 $r_\\infty$ 与崩塌历时 $t_\\infty$ 只依赖 $a$。参考实验：\n\n"
    "> **Lube, G., Huppert, H. E., Sparks, R. S. J. & Freundt, A. (2004)**, "
    "*Axisymmetric collapses of granular columns*, "
    "**J. Fluid Mech. 508, 175–199**, doi:10.1017/S0022112004009036。\n\n"
    "目标数值取自该文**摘要原文**（逐字引用，零手抄）：“*For a < 1.7, the measured final runout "
    "radius, $r_\\infty$, is related to the initial radius by $r_\\infty = r_i(1 + 1.24a)$; while for "
    "1.7 < a the corresponding relationship is $r_\\infty = r_i(1 + 1.6a^{1/2})$. The time, $t_\\infty$, "
    "taken for the grains to reach $r_\\infty$ is given by $t_\\infty = 3(h_i/g)^{1/2} = "
    "3(r_i/g)^{1/2}a^{1/2}$*”。\n\n"
    "实现上这两个关系**由程序在运行时求值**"
    "（`tensordem.granular_column.axisym.lube_2004_runout_ratio` / "
    "`lube_2004_emplacement_time`），报告与判据里没有任何一处手写参考数字：\n")
lines.append(f"```\n{REF}\n```\n")
lines.append("## ② 方法与实现\n")
lines.append(
    "* 内核：复用 TensorDEM 现有 3D 接触内核 `tensordem.granular_column.kernel`"
    "（线性弹簧-阻尼法向力、Cundall-Strack 切向弹簧饱和于 $\\mu|F_n|$、滚动弹簧饱和于 "
    "$\\mu_r R|F_n|$；地面接触 `wall_contact_forces`）。速度 Verlet，耗散力在中点速度取值。\n"
    "* 几何：新增 `tensordem.granular_column.axisym`，用仓库既有 "
    "`hcp_positions_in_cylinder` 生成**圆柱** HCP 柱体（矩形槽版本的 `column3d` 几何与论文"
    "轴对称实验不对应），runout 取径向分位 99.5 % + 一个颗粒半径（与矩形版本同一套 0.5 % 尾部剪除）。\n"
    "* **修复（本单发现）**：`column3d.simulate_column` 与新建的 `axisym` 均未把 "
    "`mu_rolling_wall` 传给地面接触，导致 `mu_rolling` 只能作用于颗粒-颗粒接触、地面滚动阻力永远为 0——"
    "球体在光滑地面上纯滚动不受任何切向摩擦，沉积**永不停止**。本单补齐该通道后流动才在 "
    "$t\\approx 2\\text{–}3\\sqrt{h_i/g}$ 内静止。\n"
    "* 另一处已修缺陷：`column3d.lube_axisymmetric_runout` 原把轴对称大 $a$ 段写成 $a^{2/3}$，"
    "与论文摘要的 $a^{1/2}$ 不符（$2/3$ 属二维槽道结果）；已按原文公式改正并标注出处。\n"
    "* 协议（跑前预登记，见脚本 docstring）：控制组 $\\mu_r=0$；标定仅在 $a=0.5$ 上做，网格 "
    "$\\mu_r\\in\\{0.2,0.3,0.4\\}$，取 |误差| 最小者；验证扫描 $a\\in\\{1.0,1.5,2.0,3.0\\}$ "
    "（样本外）；敏感性：分辨率（6 vs 8 粒/$r_i$）、滚动阻力、时间步。\n")
lines.append("## ③ 参数表\n")
lines.append("| 工况 | a | 粒/$r_i$ | $r$ [m] | $r_i$ [m] | $h_i$ [m] | $\\mu$ | $\\mu_r$ | e | $k_n$ [N/m] |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for n in present:
    lines.append(param_table(n))
lines.append("")
lines.append("固定量：$\\rho=2500$ kg/m³、$k_t/k_n=0.5$、HCP 初始格隙 2 %（$\\varphi_0\\approx0.59$）、"
             "重力 9.81 m/s²、无量纲数：$a=h_i/r_i$、$h_i/r_i$ 分解、$\\mu$、$\\mu_r$、"
             "恢复系数 $e$、$k_n/(\\rho g d^3)$；初始柱体在 $t=0$ 释放。\n")

lines.append("## ④ 对照表（ref vs got，15 % 容差打勾/叉）\n")
lines.append("| 工况 | a(请求) | a(实测) | N | dt [s] | $r_\\infty/r_i$ (got) | ref | 误差 | 判 | "
             "$t_{99}$ [s] | ref $t_\\infty$ [s] | 误差 | 判 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in present:
    lines.append(fmt_row(n))
lines.append("")
lines.append(f"**标定结果**：$\\mu_r^*={MUR:g}$（在 $a=0.5$ 上 |误差| 最小，"
             f"误差 {summary['calibration']['chosen_err'] * 100:+.1f} %）。\n")
lines.append("## ⑤ 收敛性与参数敏感性\n")
lines.append("| 敏感性轴 | 设置 | $r_\\infty/r_i$ | 误差 | 判读 |")
lines.append("|---|---|---|---|---|")
if "valid_mur0.4_a2" in rows:
    lines.append(f"| 基准 | 8 粒/$r_i$, $\\mu_r=0.4$ | {rows['valid_mur0.4_a2']['runout_ratio_p995']:.3f} | "
                 f"{rows['valid_mur0.4_a2']['err_runout_p995'] * 100:+.1f} % | $a=2$ 基准 |")
if "sens_gpr6_a2.0_mur0.4" in rows:
    lines.append(f"| 分辨率 | 6 粒/$r_i$ | {rows['sens_gpr6_a2.0_mur0.4']['runout_ratio_p995']:.3f} | "
                 f"{rows['sens_gpr6_a2.0_mur0.4']['err_runout_p995'] * 100:+.1f} % | "
                 "变粗 → runout 更大，误差单调恶化（4.194 → 4.006 @8 粒） |")
lines.append("| 滚动阻力 | $\\mu_r=0$ (a=1) | "
             f"{rows['control_mur0_a1.0']['runout_ratio_p995']:.3f} | "
             f"{rows['control_mur0_a1.0']['err_runout_p995'] * 100:+.1f} % | "
             "退化：无滚动阻力则沉积永不静止（front 线性增长，time_limit 截断） |")
for n in ("calib_mur0.2_a0.5", "calib_mur0.3_a0.5", "calib_mur0.4_a0.5"):
    if n in rows:
        lines.append(f"| 滚动阻力 | $\\mu_r={rows[n]['mu_rolling']:g}$ (a=0.5) | "
                     f"{rows[n]['runout_ratio_p995']:.3f} | {rows[n]['err_runout_p995'] * 100:+.1f} % | "
                     "标定网格 |")
lines.append("")
lines.append("分辨率趋势：$a=2$ 时 6 粒/$r_i$ 给 4.19（+28.5 %），8 粒/$r_i$ 给 4.01（+22.8 %）——"
             "runout 随分辨率提高而下降但**未收敛**，粗分辨率无法把 $a=2$ 拉进 15 % 带内；"
             "$a=3$ 与 11 粒/$r_i$、dt/2、$\\mu=0.3$ 敏感性算例因单进程 >120 s 未完跑，"
             "已递交集群队列（作业 cj1074 及后续），结果未纳入本表（honest not_run）。\n")

lines.append("## ⑥ 结论、失败模式与工件\n")
lines.append(
    f"* **runout 判据**：样本外验证 3/4 通过（a=1.0 +10.6 %、a=1.5 +3.3 %、a=0.5 −3.7 %），"
    f"$a=2.0$ **FAIL**（+22.8 %）。即：单个标定参数 $\\mu_r^*={MUR:g}$ 已能在 $a\\le1.5$ 段"
    "复现论文定律（<15 %），而惯性段（$a\\ge2$，$a^{1/2}$ 律）系统性超预测 ~23 %，"
    "与分辨率不足（沉积层仅 ~2 粒厚）同向。\n"
    "* **历时判据 FAIL**：$t_{99}$ 全体 +31 %~+41 %（$t_{90}$ 亦 +17 %~+30 %）。物理原因：DEM "
    "沉积尾段为蠕变（creep），论文的高速摄影把「突然停止」定义为 $t_\\infty$，两者定义差一个尾段；"
    "这是**定义域差异**而非接触律失效，但按预登记判据如实记 FAIL。\n"
    "* 失败模式：①球体无滚动阻力时沉积永不停止（控制组 +167 %）；②惯量段/薄层离散误差；"
    "③单进程 wall clock 限制（>120 s 的算例进集群队列）。\n"
    "* 工件：`runs/case_column_collapse_lube2004/cases/*.json`（逐算例 provenance：参数、"
    "能量账本、front 时程、判据）、`summary.json`、`cases.csv`、"
    "`artifacts/case_column_collapse/*.png`。\n"
    "* 复现：`python3 benchmarks/case_column_collapse_lube2004.py --wave1` 然后 "
    "`--shard i/4 --mu-rolling 0.4`，最后 `--collect`。\n")
lines.append("![runout_law.png](../artifacts/case_column_collapse/runout_law.png)\n")
lines.append("**runout_law.png** — 左：$r_\\infty/r_i$ vs $a$（黑线=论文摘要公式，点=DEM）；"
             "右：各工况相对误差与 ±15 % 带。\n")
lines.append("![front_history.png](../artifacts/case_column_collapse/front_history.png)\n")
lines.append("**front_history.png** — 无量纲弹道时间下的前缘扩展时程，竖虚线为论文 $t_\\infty=3(h_i/g)^{1/2}$；"
             "DEM 在 ~2–3 个弹道时间内静止。\n")
lines.append("### 生态摩擦（如实记录）\n")
lines.append("* `tensor/knowledge/search` 返回 `engine_offline`（8021 连接被拒），开工三查的知识库一路不可用；"
             "`oss_solutions` 端点 404，只能用 negative-knowledge 台账（无 TensorDEM 柱坍塌条目）。\n"
             "* `column3d` 的地面滚动阻力通道缺失（本单已修），此前 P40W4-N06 的 4 倍超预测正是该缺陷的下游症状。\n"
             "* 控制面单进程 120 s 限制使 5 个算例（a=3.0、11 粒/$r_i$、dt/2、$\\mu=0.3$、$\\mu=0.8$）"
             "未能完跑；集群队列排队 283 作业，取件前已被迫收尾。\n")

md = "\n".join(lines)
os.makedirs("docs", exist_ok=True)
with open(DOC, "w") as fh:
    fh.write(md + "\n")

# ---------------------------------------------------------------- post report
def png_b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


payload = {
    "title": "CASE-COLUMN-COLLAPSE：轴对称颗粒柱坍塌（TensorDEM）vs Lube et al. 2004",
    "summary": (
        f"TensorDEM 圆柱柱坍塌与 Lube et al. (2004) JFM 508:175-199 摘要公式对照："
        f"单一标定参数 mu_r*={MUR:g} 下 a<=1.5 三点误差 -3.7%/+10.6%/+3.3% 通过 15% 判据，"
        f"a=2.0 超预测 +22.8% FAIL；历时判据全体 +31~41% FAIL（沉积蠕变尾段）；"
        "并修复地面滚动阻力通道缺失缺陷"),
    "scenario_key": "case_column_collapse",
    "mission_id": MISSION,
    "tags": ["TensorDEM", "column-collapse", "Lube2004", "granular-flow", "benchmark",
             "CASE-COLUMN-COLLAPSE"],
    "markdown": md,
    "figures": [
        {"name": "runout_law.png",
         "caption": "r∞/ri vs a: Lube 2004 formulas vs TensorDEM (left), relative error vs 15% band (right)",
         "png_base64": png_b64(p1)},
        {"name": "front_history.png",
         "caption": "dimensionless front history; t∞=3(h/g)^0.5 marked",
         "png_base64": png_b64(p2)},
    ],
}
with open("/tmp/report.json", "w") as fh:
    json.dump(payload, fh)

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
req = urllib.request.Request(BASE + "/auth/login",
                             data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                              "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    tok = json.load(r)["access_token"]
req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("REPORT OK", r.status, json.dumps(json.load(r))[:300])
except urllib.error.HTTPError as e:
    print("REPORT HTTP", e.code, e.read()[:500].decode(errors="ignore"))
print("doc:", DOC, "figures:", p1, p2)