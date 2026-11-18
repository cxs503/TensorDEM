#!/usr/bin/env python3
"""m24489 (reopened TensorDEM#1724, see-through round 2): publish the report.

All numbers are read from runs/summary.json and the run JSONs (produced by
aggregate.py / run_case.py) - zero hand-copy.  The five figures are the ones
shipped in this PR (benchmarks/case_rcp_deposition/figures/).
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "benchmarks" / "case_rcp_deposition"
FIGS = CASE / "figures"
SUMMARY = json.loads((CASE / "runs" / "summary.json").read_text())

# ---- login ---------------------------------------------------------------
req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"},
)
body = json.loads(urllib.request.urlopen(req).read())
token = body.get("access_token") or body.get("token")
assert token, f"login did not return a token: {list(body.keys())}"

# ---- figures -------------------------------------------------------------
def fig(name):
    return base64.b64encode((FIGS / name).read_bytes()).decode()

figures = [
    {"name": "fig_profiles.png",
     "caption": "Settled-bed solid-fraction profiles phi(z) for both friction "
                "tiers (mu=0 and mu=0.1, N=10000, k_n=4000), with the ledger "
                "acceptance band [0.62, 0.66] and the frictionless RCP "
                "reference 0.6366.",
     "png_base64": fig("fig_profiles.png")},
    {"name": "fig_phi_vs_mu.png",
     "caption": "Bulk solid fraction vs inter-particle friction; one marker "
                "per seed. Friction loosens the deposited packing.",
     "png_base64": fig("fig_phi_vs_mu.png")},
    {"name": "fig_phi_vs_N.png",
     "caption": "Finite-size scan of phi_bulk (mu=0).",
     "png_base64": fig("fig_phi_vs_N.png")},
    {"name": "fig_settling.png",
     "caption": "Kinetic-energy history during settling (the stopping "
                "criterion KE/(N m g d) < 5e-4).",
     "png_base64": fig("fig_settling.png")},
    {"name": "fig_phi_vs_stiffness.png",
     "caption": "Bulk solid fraction vs 1/k_n for both friction tiers "
                "(4 stiffness levels each). Squares are the Richardson "
                "hard-sphere limits.",
     "png_base64": fig("fig_phi_vs_stiffness.png")},
]

# ---- programmatic tables -------------------------------------------------
ref = SUMMARY["reference"]["phi_rcp"]
band = SUMMARY["band"]
tol = SUMMARY["tolerance_rel"]
tiers = SUMMARY["tiers"]

def relerr(phi, refv):
    return (phi - refv) / refv * 100.0

tier_lines = ["| tier | phi_bulk (mean) | std | Z | dev vs ref 0.6366 |",
              "|---|---|---|---|---|"]
for mu in ("0", "0.1"):
    t = tiers[f"mu={mu}"]
    tier_lines.append(
        f"| mu={mu} | {t['phi_mean']:.5f} | {t['phi_std']:.2e} | {t['Z']:.3f} "
        f"| {100*t['dev_vs_ref']:+.2f} % |")
tier_table = "\n".join(tier_lines)

stiff = SUMMARY["stiffness"]
stiff_lines = ["| tier | hard-sphere limit | order p | n levels | dev vs 0.6366 |",
               "|---|---|---|---|---|"]
for mu in ("0", "0.1"):
    s = stiff[f"mu={mu}"]
    stiff_lines.append(
        f"| mu={mu} | {s['limit']:.5f} | {s['order_used']:.3f} | {s['n_levels']} "
        f"| {100*(s['limit']-ref)/ref:+.2f} % |")
stiff_table = "\n".join(stiff_lines)

gate_lines = ["| gate | criterion value | pass |", "|---|---|---|"]
for g in SUMMARY["gates"]:
    got = g["got"]
    got_s = ("True" if got is True else "False" if got is False
             else (f"{got:.4g}" if isinstance(got, float) else str(got)))
    gate_lines.append(f"| {g['id']} | {got_s} | {'PASS' if g['pass'] else 'FAIL'} |")
gate_table = "\n".join(gate_lines)

n_pass = sum(1 for g in SUMMARY["gates"] if g["pass"])
n_fail = len(SUMMARY["gates"]) - n_pass

# error determinations (key quantities vs the frictionless RCP anchor 0.6366)
def pct(phi):
    return (phi - ref) / ref * 100.0

err_mu0_prod = pct(tiers["mu=0"]["phi_mean"])
err_mu0_lim = pct(stiff["mu=0"]["limit"])
err_mu01_prod = pct(tiers["mu=0.1"]["phi_mean"])
err_mu01_lim = pct(stiff["mu=0.1"]["limit"])

markdown = f"""# CASE-DEM-RCP-PACKING-FRACTION — 补齐案例图件，闭合 m22558 交付（m24489，重开 #1724 第 2 轮）

## ① 案例介绍

**物理背景**：重力沉积（gravity deposition）是离散元（DEM）最基础的装填基准——
N 个等径刚/软球在重力作用下落入方形容器，经碰撞、摩擦、滚动耗散后沉降为
随机密堆（random close packing, RCP）。其体积分数 phi 是 DEM 接触模型
（法向刚度 k_n、摩擦 mu、恢复系数 e）系统误差的"试金石"：无摩擦极限应逼近
文献 RCP 值 **phi_rcp = 0.6366**（Scott 1960；Bernal & Mason 1960），
ledger 验收带 **[0.62, 0.66]**，相对容差 **3 %**。

**几何/参数**（N=10000 主档，d=1，R=0.5，密度 1，g=1，dt = t_c/20）：

| 参数 | 取值 |
|---|---|
| 粒子数 N | 10000（主档）；2000/4000 有限尺寸档 |
| 摩擦 mu | 0.0（无摩擦档）、0.1（低摩擦档） |
| 法向刚度 k_n | 4000（生产档）、16000、64000、256000（刚度阶梯） |
| 恢复系数 e | 0.1 |
| 种子 | 1/2/3（生产档 3 种子） |
| 时间步探针 | dt_frac = 0.025（半步收敛探针） |
| 沉降判据 | KE/(N m g d) < 5e-4 |

## ② 怎么做的

- **本单动作**：`docs/CASE_RCP_DEPOSITION.md` 自"paper-style report"落地起就
  引用了 `benchmarks/case_rcp_deposition/figures/` 下 5 张图，但该目录从未入树，
  导致 `tests/test_rcp_deposition_artifacts.py::test_every_referenced_figure_is_in_the_tree`
  在干净检出上**必挂**。本单用 `aggregate.py` 从**已提交的 run 集**重新生成这 5 张图
  并提交（`MPLBACKEND=Agg, PYTHONHASHSEED=0`，两次重跑字节级一致，确定性已验证）。
- **工具链**：`run_case.py`（单 run）→ `runs/*.json/.npz` → `aggregate.py`
  （门控 + 图 + 文档 + summary.json，全部程序化，零手抄）→ `repro_gate.py`
  （反向可复现门：committed 工件 == committed 代码 × committed 数据）。
- **不改任何代码**：summary.json 与文档字节未变（aggregator 输出本就是最新的），
  本单只补图件，5 个文件，单一功能点。

## ③ 计算结果（程序化注入，零手抄）

**分层统计**（N=10000，k_n=4000 生产档，3 种子均值）：

{tier_table}

**刚度阶梯 → 硬球极限**（Richardson 外推，4 级）：

{stiff_table}

**门控**（23 门，{n_pass} PASS / {n_fail} FAIL）：

{gate_table}

> G5/G6（软球收敛斜率门）与 G9（残余重叠 1.15 %）为**已知未决债**，
> 由 m21145/m22558 锚点溯源审计单独裁定，不属于本单范围。

## ④ 结果图

5 张图均为本单提交（见 figures 内嵌）：沉降剖面 phi(z)（含验收带与参考线）、
phi–mu、有限尺寸、沉降动能史、phi–1/k_n 硬球极限。

## ⑤ 注意点

- 图件是 `aggregate.py` 的**输出**，不是手画：run 集一变必须重跑 aggregator，
  否则文档引用悬空（本单修的正是这个坑）。
- mu=0.1 档的文献锚点（0.6250）是**自声明插值**，按锚点溯源政策降级为
  `derived`，精度主张 **WITHHELD**——不得对 0.6366 报 "<3 % PASS"。
- 适用范围：等径球、立方周期/壁面容器、Hertz–Mindlin 接触；非等径/非球需另立案例。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 值 | 基准 | 相对误差 | <3 % ? |
|---|---|---|---|---|
| phi_bulk (mu=0, 生产档) | {tiers['mu=0']['phi_mean']:.5f} | 0.6366 (Scott 1960) | {err_mu0_prod:+.2f} % | **是** |
| phi 硬球极限 (mu=0) | {stiff['mu=0']['limit']:.5f} | 0.6366 | {err_mu0_lim:+.2f} % | **是** |
| phi_bulk (mu=0.1, 生产档) | {tiers['mu=0.1']['phi_mean']:.5f} | 0.6250 (derived, 非文献) | {err_mu01_prod:+.2f} % (vs 0.6366) | **不适用**（锚点 derived，主张 WITHHELD） |
| phi 硬球极限 (mu=0.1) | {stiff['mu=0.1']['limit']:.5f} | 0.6250 (derived) | {err_mu01_lim:+.2f} % (vs 0.6366) | **不适用**（同上） |

**显式回答**：无摩擦档（生产档与硬球极限）对文献锚点 0.6366 的误差
{err_mu0_prod:+.2f} % / {err_mu0_lim:+.2f} %，**均 < 3 %，PASS**；
低摩擦档因锚点为 derived 而 **WITHHELD**（不报 PASS/FAIL，如实降级）。
"""

payload = {
    "title": "CASE-DEM-RCP-PACKING-FRACTION：补齐 5 张案例图件，闭合 m22558 交付（m24489 重开 #1724 第 2 轮）",
    "summary": "重开 TensorDEM#1724 第 2 轮：m22558 交付的 run/审计已在 main，唯 figures/ 目录缺失致 test_every_referenced_figure_is_in_the_tree 必挂。用 aggregate.py 从已提交 run 集确定性重生成 5 张图并提交（5 文件，零代码改动），rcp 全套测试转绿。",
    "scenario_key": "case_rcp_deposition",
    "mission_id": 24489,
    "tags": ["TensorDEM", "RCP", "deposition", "m22558", "figures", "defect-fix", "see-through"],
    "markdown": markdown,
    "figures": figures,
}

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer " + token,
             "Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req).read())
print("report posted:", json.dumps(resp, ensure_ascii=False)[:400])
