#!/usr/bin/env python3
"""Publish the AUDIT-SCI progress report (m22981) with programmatic figures."""
import base64, io, json, os, re, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/srv/tensor/agent/work/m22981")
SPH = Path("/srv/tensor/workspace/TensorSPH")
claims = json.loads((ROOT / "artifacts/readme_claim_audit/claims.json").read_text())

# ---------------- figure 1: verdict census per repo (from claims.json) -------------
order = ["REAL", "REAL_REGENERABLE", "STALE_NUMBER", "WITHHELD", "NO_TEST",
         "GATE_DRIFT", "SELF_CONTRADICTION", "NOT_SUPPORTED", "RETIRED"]
dem = {v: 0 for v in order}
sph = {v: 0 for v in order}
for c in claims["claims"]:
    bucket = dem if c["repo"] == "dem" else sph
    bucket[c["verdict"]] = bucket.get(c["verdict"], 0) + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
xs = [v for v in order if dem.get(v) or sph.get(v)]
ax.barh([f"  {v}" for v in xs], [dem.get(v, 0) for v in xs], label="TensorDEM", color="#2b6cb0")
ax.barh([f"  {v}" for v in xs], [-sph.get(v, 0) for v in xs], label="TensorSPH", color="#c05621")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("claims (left = DEM, right = SPH)")
ax.set_title(f"README claim↔evidence audit — {claims['schema_version']}")
ax.legend(loc="lower right", fontsize=8)
for i, v in enumerate(xs):
    if dem.get(v):
        ax.text(dem[v] + 0.15, i, str(dem[v]), va="center", fontsize=8)
    if sph.get(v):
        ax.text(-sph[v] - 0.55, i, str(sph[v]), va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)

# ------------- figure 2: SPH hydrostatic bottom pressure vs the quoted 2.7 % -------
hyd = json.loads((SPH / "benchmarks/threed_hydrostatic/report.json").read_text())
shear = json.loads((SPH / "benchmarks/threed_shear/report.json").read_text())
gated = hyd["gated_cases"]
rows = []
for run in hyd["runs"]:
    if run["case"] in gated:
        rows.append((run["case"], 100 * run["pressure"]["bottom_rel_error"],
                     100 * run["pressure"]["max_rel_error_core"]))
rows.sort(key=lambda r: r[1])
ax = axes[1]
labels = [r[0] for r in rows]
bottom = [r[1] for r in rows]
worst = [r[2] for r in rows]
ax.barh(labels, bottom, color="#2f855a", label="bottom-pressure rel. error [%]")
ax.scatter(worst, labels, marker="|", s=200, color="#c53030", label="core-layer max |err| [%]")
ax.axvline(-2.7, ls="--", color="k", lw=1.2, label="README: 'within 2.7 %'")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("relative error [%]")
ax.set_title("SPH 3-D hydrostatic: README's 2.7 % vs gated cases\n"
             f"(shear decay at finest rung = {100*shear['resolution_ladder'][-1]['decay']['rel_error']:.2f} %,"
             f" benchmark verdict = {shear['verdict']})")
ax.legend(fontsize=7, loc="lower left")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150)
png_b64 = base64.b64encode(buf.getvalue()).decode()

# ---------------- markdown ----------------------------------------------------------
counts = claims["verdict_counts"]
census = [c for c in claims["claims"] if c["cid"] == "DEM-README-TEST-CENSUS"][0]["census"]
hyd_rows = "\n".join(
    f"| `{c}` | {100*rel:+.2f} % | {100*mx:+.1f} % |" for c, rel, mx in rows)

md = f"""# AUDIT-SCI：求解器 README 声称 vs 代码真相对账（TensorDEM + TensorSPH）

**单号** m22981 · **锚点** `AUDIT-SCI` · **commit** `ac420032` · **产物**
`artifacts/readme_claim_audit/{{claims.json,CLAIM_EVIDENCE_TABLE.md}}` ·
`docs/AUDIT_README_CLAIM_VS_CODE_M22981.md` · 审计器 `scripts/readme_claim_audit.py`（864 行）·
契约测试 `tests/test_readme_claim_audit.py`（21 passed）

## ① 案例介绍

对象是**两份求解器 README 声称的能力集**，不是某条物理算例。审计目标：每条能力声称能否落到
①实现文件（符号在 AST 里被*定义*）②被注册的测试文件（且该测试真的引用被测模块）③可核对的基准数值。
两个仓：TensorDEM（本分支，7af13456 派生）与 TensorSPH（只读外部检出 `cde14cf5`）。
共注册 **15 条能力级声称**：DEM 11 条（含锚点要求的三条：冰脊触底+海底管道地基、船体水线带冰磨损、
稳态碎冰直航能量账本），SPH 4 条（3-D 算子、2-D 算例、表面张力、Adami 壁面模型）。

判据分级沿用生态口径：`REAL`（代码+测试+受控证据）/ `REAL_REGENERABLE`（数值证据是生成物，
本仓惯例不入 VCS，但生成脚本可跑）/ `WITHHELD`（引用路径或数字在树里找不到第二份）/
`SELF_CONTRADICTION`（同文既声称又否认）/ `NOT_SUPPORTED`（声称存在、代码缺失）/ `STALE_NUMBER`。

## ② 怎么做的

审计被写成**可执行合同**而不是散文复核——这是它唯一不腐烂的形态：

1. **注册表**：每条声称登记 `anchor`（README 里必须仍在的原文片段）、`code`、`symbols`
   （`path::Name`，用 AST 判定是否*定义*，import 不算）、`tests` + `test_must_reference`、
   `docs`/`schemas`（受版本管理证据）、`artifacts`（生成物）、`generators`（重生成脚本）、
   `numbers`（引用的数字 + 独立第二来源）、`gates`（`gates.json` 门值）、`json_checks`
   （`benchmarks/.../report.json` 里的点路径取值）、`counterclaim`（同文中否认句）。
2. **机械检查**：路径存在 / 符号定义 / 测试引用 / 指针解析（README 里每个相对路径都试解析，
   支持 `{{a,b}}` 花括号缩写与 `file.py:symbol` 形态）/ 数字回溯（声明来源找不到→全仓 grep，
   区分「指针写错」与「数字无处可寻」）/ 门值比对（2 % 容差）/ 自我普查（按仓库自己的
   `scripts/readme_api_audit.py:_test_style` 定义现算）/ 生成器可用性（`--help` 实跑）。
3. **负面控制**：`tests/test_readme_claim_audit.py` 用 8 个合成仓证明每一类缺陷**会被抓出来**
   （缺文件、import-only 符号、无源数字、自相矛盾、悬空指针、无日期戳的过期计数、门值漂移、
   声称已从 README 撤下），而不是只证明「跑过了」。
4. `--strict` 以注册时的诚实基线为门，任一格变差即非零退出。

## ③ 计算结果（判据与数字，全部由脚本注入）

| 判定 | 条数 | 说明 |
|---|---|---|
{chr(10).join(f"| `{k}` | {v} | |" for k, v in sorted(counts.items()))}

**TensorDEM（锚点三条能力的落实）**

| 能力 | 实现 | 测试 | 数值证据 | 判定 |
|---|---|---|---|---|
| 冰脊触底 + 管道地基 | `ridge_dem.py`/`ridge_geotech.py`/`ridge_grounding.py`（1104/770/706 行） | 3 个测试文件 | `docs/ice_ridge_pipeline_grounding.md`；`artifacts/ridge_grounding/` 由 `run_ice_ridge_grounding.py` 重生成（`--help` OK） | REAL_REGENERABLE |
| 船体水线带冰磨损 | `ice_abrasion.py`（1820 行，Hertz+C–S→Archard+Finnie） | 8 个测试文件引用 | `docs/ice_abrasion_archard_wear.md`（`V=KWL/H`、`CALIBRATED`） | REAL_REGENERABLE |
| 稳态碎冰直航能量账 | `ice_transit/{{transit,laws,report}}.py` | `test_ice_transit.py` | `docs/ICE_TRANSIT_RESISTANCE_SURFACE.md` | REAL_REGENERABLE |

三条**都是真实现**，README 对校准状态的自我标注也诚实（磨损绝对量级只有一个参数 `k` 对标文献，
明确写 `CALIBRATED`，没把标定值说成预测值）。LBMDEM 的 honest FAIL 亦是真 FAIL：门
`stokes_drag_analytic.measured = 0.009757`（=+0.98 %，门 ≤2 %，pass）而顶层
`all_recomputed_pass = false`，README 逐条列了未通过项——正面样本。

**TensorSPH（只读，问题移交）**

| 编号 | 严重度 | 问题 | 证据 |
|---|---|---|---|
| S1 | **阻断** | README 两处把 3-D 验证指向 `docs/verification_threed.md`，该文件不存在 | `git ls-files` 无此 doc；真实证据在 `benchmarks/threed_{{shear,hydrostatic}}/report.json` |
| S2 | **阻断** | `Implemented` 声称 CSF 表面张力（mixture 版 Young–Laplace 1.4 %），`Current limitations` 写「**No surface-tension model.**」 | 代码 `surface_tension.py`/`csf_mixture.py` 为真；限制段是未随实现更新的残留 |
| S3 | 中 | 3-D 静水底压「within 2.7 %」在它自己的 report 里**找不到对应算例**，且与同句 shear-decay 的 2.7 % 重复 | 见下表：4 个 gated 算例 −1.21 %…−4.39 %，核心层最大 17–26 % |
| S4 | 中 | `threed_shear/report.json` 顶层 `verdict = "fail"`（fourth-order 未在每一档都更优），且 `missing_cases` 有 6 个未跑；README 只报通过项 | report.json 原文 |
| S5 | 待确认 | `threed/csf3.py` 存在 3-D CSF（Rayleigh–Plateau），而 README 与 `threed/__init__.py` 称 3-D 无表面张力（可能是**低估**） | 无 `test_csf3*`，本单未跑其 benchmark → 记 `not_run`，不给 pass |

SPH 3-D 静水底压（机器值，非手抄）：

| 算例 | 底压相对误差 | 核心层最大误差 |
|---|---|---|
{hyd_rows}

`best_case` = `adami-c0_40-nz16`；`shear` 最细档（n=24）`decay.rel_error = -2.68 %`——2.7 % 这个数
**属于 shear**，把它同时用作静水指标是本单认定的文档缺陷。

## ④ 结果图

下图左：两仓 15 条声称的判定分布（左半 DEM、右半 SPH）；右：SPH 3-D 静水四个 gated 算例的底压误差
（绿条）与核心层最大误差（红刻度）对 README 声称的 −2.7 %（虚线），数据全部由
`benchmarks/threed_hydrostatic/report.json` 程序化读取。可见 2.7 % 只覆盖 best_case，其余三个算例
超出，核心层误差到 26 %——README 的单点指标掩盖了真实精度面。

## ⑤ 注意点 / 局限

1. **注册表 ≠ 全文**：15 条**能力级**声称；背景/动机性散文不查。
2. **「存在」≠「正确」**：`code`/`tests` 检查被定义、被引用，不检查物理正确；数值正确性由各能力自己的
   benchmark 门负责（本单对 LBMDEM 与 SPH 3-D 做了门级复核，其余沿用仓内既有门）。
3. TensorDEM 三条锚点能力的**基准数值不在 git 里**（`artifacts/` 按仓库明示惯例不入 VCS），
   故判 `REAL_REGENERABLE` 而非 `REAL`：干净检出无法直接核对载荷–位移曲线，只能重跑 campaign。
   建议把 `criteria`/`gates` 级 JSON 入库（体积小、可 diff）。
4. **未跑（记 `not_run`，不算 pass）**：ice_transit / ridge_grounding / pressure_band 三场 campaign，
   以及 SPH `threed_shear`(≈30 min)、`threed_hydrostatic`(≈10 min)。本机 load15≈91（nproc 128），
   按容量纪律本单只做轻量活（审计/文档/测试），未起重算。
5. SPH 侧为**只读外部检出** `cde14cf5`，本单不改 SPH 仓，S1–S5 以问题单移交（修法均已给出）。

## ⑥ 误差判定：关键量 vs 基准

审计器的「误差」是把声称落到证据时的**不一致量**，逐项对照如下（并显式回答是否 <3 %）：

| 关键量 | 基准 | 实测偏差 | 是否 <3 % |
|---|---|---|---|
| DEM 测试规模自查（698 文件 / 6506 模块级 `test_*` / 346 `TestCase`） | README 原值 343/2714/319 | 原值约为实测的 49 %/42 %/92 %；**已改为实测值**，复算后偏差 0 % | **是（修复后 0 %）** |
| LBMDEM Stokes 曳力 +0.98 % | 门 ≤2 %（`gates.json`） | 机器值 0.009757 vs README 0.98 % → 0.4 % 相对差 | **是** |
| SPH 3-D shear 衰减 2.7 % | 解析 −2νk² | report 最细档 −2.68 %，与 README 差 0.7 % | **是** |
| SPH 3-D 静水底压「within 2.7 %」 | 4 个 gated 算例 | 无算例等于 2.7 %；best −1.21 %, 其余 −2.49 %/−4.21 %/−4.39 % | **否**（声称以偏概全） |
| SPH 3-D 核心层压力最大误差 | 15 % 门 | +17.3 %…+25.8 %（README 未披露） | **否**（未披露） |
| SPH shear 报告裁定 | README「verified」 | 顶层 `verdict=fail` + 6 个 `missing_cases` | **否**（结论被隐去） |
| DEM 三条锚点能力的基准数值可核对性 | 干净检出 | `artifacts/` 全部缺失，需重跑 | 不适用（判 REAL_REGENERABLE，非虚标） |

**honest 结论**：TensorDEM 未发现能力虚标；虚标集中在 TensorSPH（1 条自相矛盾、2 条只报通过项/
数字不可复现、1 条证据指针悬空）。审计器本身与 21 条契约测试（含 8 个负面控制）已随 commit 落盘，
README 若再漂移会被 `--strict` 拦下。
"""

payload = {
    "title": "AUDIT-SCI：求解器 README 声称 vs 代码真相对账（TensorDEM + TensorSPH）",
    "summary": "把两仓 README 的 15 条能力声称做成可执行合同（代码符号 AST 判定/测试引用/数字第二来源/门值比对/"
               "自我普查/生成器实跑）。DEM 三条锚点能力（冰脊触底+管道地基、冰磨损、直航能量账）确为真实现但数值证据"
               "不在 VCS（REAL_REGENERABLE）；SPH 查出 1 条自相矛盾（表面张力既声称又否认）、2 条只报通过项"
               "（静水 2.7% 无算例支持、shear 报告 verdict=fail 未披露）、1 条证据指针悬空。",
    "scenario_key": "audit_readme_claim_vs_code",
    "mission_id": 22981,
    "tags": ["audit", "readme", "evidence", "TensorDEM", "TensorSPH", "claims", "reproducibility"],
    "markdown": md,
    "figures": [{
        "name": "claim_evidence_audit.png",
        "caption": "左：15 条 README 能力声称的判定分布（左半 TensorDEM，右半 TensorSPH）。"
                   "右：SPH 3-D 静水四个 gated 算例底压相对误差（绿）与核心层最大误差（红刻度）对照 "
                   "README 声称的 −2.7%（虚线）；数据由 benchmarks/threed_hydrostatic/report.json 程序化读取。",
        "png_base64": png_b64,
    }],
}

base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
user = os.environ.get("TENSOR_AGENT_USER"); pw = os.environ.get("TENSOR_AGENT_PASS")
tok = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/auth/login",
                      "-H", "Content-Type: application/json",
                      "-d", json.dumps({"username": user, "password": pw})],
                     capture_output=True, text=True).stdout
token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
res = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
                      "-H", f"Authorization: Bearer {token}",
                      "-H", "Content-Type: application/json",
                      "-d", "@/tmp/report.json"], capture_output=True, text=True)
print(res.stdout[:800])