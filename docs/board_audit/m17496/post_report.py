#!/usr/bin/env python3
"""Publish the m17496 board-audit report (numbers injected from finding.json)."""
from __future__ import annotations

import base64
import json
import os
import subprocess

import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
F = json.load(open(os.path.join(REPO, "artifacts/audit_m17496/finding.json")))
R = F["result"]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")

crit = R["verdicts"]
imp = R["impulse_momentum_sweep"]
base = R["baseline_comparison"]
rows = "\n".join(
    f"| {c['metric']} | {c['reference']} | {c['got']:.3e} | {c['tolerance']} | "
    f"{'✅' if c['met'] else '❌'} |" for c in crit)
imp_rows = "\n".join(f"| {r['n_steps']} | {r['measured_momentum_Ns'][0]:.6e} | "
                     f"{r['analytic_impulse_Ns'][0]:.6e} | {r['rel_error']:.3e} | "
                     f"{r['impulse_ratio']:.12f} |" for r in imp)
probes = R["semantic_probes"]
mut = R["mutation_matrix"]
mut_rows = "\n".join(
    f"| {m['key']} | {m.get('detected')} | {next((b['baseline_detected'] for b in base['control']['rows'] if b['key']==m['key']), 'n/a')} |"
    for m in mut)
g = F["git"]
chk = F["checks"]

md = f"""# 交差核验 · m17455 = XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT（TensorDEM）— 判级 L2

**一句话**：机制真、数字真、可复现真（复跑 0 处数值差异）；但**7 个交付路径全为未提交**、**0 个 solver/平台消费方**、**登记裁判表 0 行** → L2，不是 L3/L4。verdict 已落表 `tensor_maturity_verdicts.id=572`。

## ① 案例介绍

被审单 m17455 把 TensorPhysics 的 `library-binding-frozen-contract` 做法移植到 TensorDEM 的**供应方**：把对外绑定面（原生 dt/时钟、载荷形状 `(N,2)`/单位 N、**瞬时非累加**载荷语义、整数子步规则、2-D `snapshot/from_snapshot` vs 3-D `state_dict` 重启分叉、负向断言 `not hasattr(…,'restore')`）从 docstring 里的隐含假设变成**声明式契约 + sha256 钉扎 + 内省校验 + 行为探针**。

| 项 | 值 |
|---|---|
| 锚点 | `XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT` |
| 契约 schema / 版本 | `{R['verify']['contract_schema']}` / `{R['verify']['contract_version']}` |
| 冻结指纹 | `{R['verify']['contract_fingerprint']}`（match={R['verify']['frozen_fingerprint_match']}） |
| 冻结面 | `IceDEM` 11/11 = 100%；`IceDEM3D` 8/20 = 40% |
| 环境 | python {R['environment']['python']} / torch {R['environment']['torch']} / {R['environment']['processor']}，单核，脚本 {R['wall_s']:.2f} s，无集群作业 |
| 登记裁判行 | **0**（`tensor_benchmark_cases` 54 个算例无本锚点；`tensor_xval_cases` 7 行亦无） |

## ② 怎么做的（核验方法）

审计脚本 `docs/board_audit/m17496/audit_m17455.py`，只执行被审单自己登记的复现命令（输出到 `/tmp/m17496_repro`，不碰 mission 文件）：

1. 11 个交付路径 byte-exact 拷贝 + sha256 MANIFEST；
2. 交付链：`git status` / `git cat-file -e HEAD:<p>` / `git log --all --grep=17455`；
3. 裁判查询（两表）+ 报告 16428 六段与图核对；
4. **独立复现 + 递归深比对**（float 1e-12，排除 wall_s）；
5. 测试完整性（读断言 + 重跑）；
6. **对抗探针**（审计方自造）：契约 units `N→kN` 后 `verify/assert_conforms` 是否拦；活体 `del positions` 是否被抓；
7. OSS ref 零手抄：TensorPhysics 被引文件行数与负向断言；
8. 集成性：`grep -rln binding_contract src/ benchmarks/ scripts/ tests/`。

## ③ 计算结果

### 3.1 三件套与交付链

| 检查 | 结果 | 证据 |
|---|---|---|
| run_artifacts 存在 | ✅ 11/11（result.json 19254 B `abe0efb1…`、4 png、run.json） | `docs/board_audit/m17496/refs/MANIFEST.json` |
| ref vs got 数值 | ✅ 8 判据 + 冲量扫描 n=1..128 + 7 行变异 + 19 探针 | 同上 |
| repro_cmd 可执行 | ✅ exit {F['repro']['returncode']}，**{len(F['repro_diff'])} 处数值差异** | `artifacts/audit_m17496/finding.json` |
| 已提交（branch/HEAD） | ❌ {len(g['untracked_deliverables'])}/7 路径 untracked；`--grep=17455` 提交数 0 | `git_state` |
| 登记裁判 | ❌ 0 行（benchmark_cases + xval_cases 双查） | 表查询 |

### 3.2 预登记 8 条判据 vs 审计复跑

| 判据 | ref | got | 阈值 | 判 |
|---|---|---|---|---|
{rows}

冲量扫描逐点（2-D，192 个粒子中取 x 分量，节选）：

| n 子步 | 实测 m·Σdv | 解析 n·dt·F | 相对误差 | 比值 |
|---|---|---|---|---|
{imp_rows}

变异矩阵（7 类，契约检测器 vs 仅名字基线）：检测率 **{base['treatment']['rate']:.3f}** vs 基线 **{base['control']['rate']:.3f}**（Δ={base['delta_rate']:.3f}）；19/19 探针 PASS。

| 变异类 | 契约检测 | 基线检测 |
|---|---|---|
{mut_rows}

### 3.3 对抗探针（审计方独立）

| 探针 | 期望 | 实测 | 判读 |
|---|---|---|---|
| 契约 units `N→kN` → `verify(contract=…)` | 报漂移 | `CONFORMS`/`breaking=0`/`frozen_fingerprint_match=False` | ⚠ F2：钉扎未进状态判定 |
| 活体 `del positions` → `verify(instances=…)` | 抓 | `DRIFT`/`breaking=2` | ✅ 真抓 |
| OSS ref 行数（claim 113/43） | 一致 | 113/43 且含 `assert not hasattr(Solver,'restore')` | ✅ 零手抄可核 |
| solver/平台消费方 | ≥1 | 0（仅 tests/ + scripts/） | ❌ L3 不成立 |

## ④ 结果图

- 左：7 类变异逐类命中（契约 vs 仅名字基线）；中：冲量-动量误差 vs 子步数（重跑值，叠画 1e-12 预登记线与 owner 15% 线）；右：本次核验 yes/no 门（复现 yes / 已提交 no / 消费方 no / 登记裁判 no）——这就是 L2 的依据。

## ⑤ 注意点

1. **可复现 ≠ 已交付**：7 路径全 untracked、`head_commit=null`，自动推送无提交可推 → 证据链只活在一个 worktree（F1，阻断级）。
2. **F2 指纹失配不进状态**：`verify()` 只由 `breaking` 决定 `status`，`frozen_fingerprint_match=False` 仅作字段；审计方改单位常量后仍得 `CONFORMS` → 只看 `status` 的绑定方不会收到"声明面被改"的警报。建议状态降级或让 `assert_conforms()` 抛。
3. **F4 检测率 1.0 是上界**：7 类变异与检测器同源（循环设计），不能外推为现场漂移率。
4. **F5 冲量恒等式近乎平凡**：自由飞行线性积分器上本就精确到舍入（1e-16 平线），它证明时钟/载荷记账/非累加语义，不证明接触物理；被审单 §6.2/⑥ 已如实写明。
5. **F3 3-D 仅 40% 冻结**（8/20，已披露）。
6. 本锚点无物理量，owner 15% 标准只能套在记账恒等式上（实测 1e-16），**不可宣称物理精度达标**。
7. 裁判表缺行属平台供给缺口（本锚点未登记判据），审计不代填。

## ⑥ 误差判定（显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 冲量-动量 2-D（max n=1..128） | 解析 n·dt·F | 2.66e-16 | ✅ 是 |
| 冲量-动量 3-D | 解析 n·dt·F | 3.56e-16 | ✅ 是 |
| 台账 vs 解析冲量 | 解析 n·dt·F | 1.77e-16 | ✅ 是 |
| 原生时钟 vs n·dt | 解析 n·dt | 1.91e-16 | ✅ 是 |
| 重启漂移 | 0 m | 0.0（bitwise） | ✅ 是 |
| 审计复现值 vs 归档值 | 归档 result.json | 0 处差异 | ✅ 是 |
| 漂移检测率 | 7/7 期望 | 0 漏检（基线漏 6） | ✅ 是（适用边界见 ⑤.3） |

**判级 L2**：L1 数学/单元验证 ✅；L2 独立功能实现 ✅；L3 系统集成 ❌（0 消费方、0 提交、无 CI/门消费 `assert_conforms()`）；L4 基准验证 ❌（登记裁判 0 行；基准为自洽解析恒等式，非独立外部基准）；L5 ❌。

**返修路径（升 L3）**：提交 7 个路径到 `agent/m17455`；补 1 个真实消费方（`IceDEM/__init__` 或 CI 调 `assert_conforms()`）；把 `frozen_fingerprint_match=False` 纳入 `status`。
"""

png = os.path.join(REPO, "docs/audit/figures/m17496_m17455_audit.png")
figs = [{"name": "m17496_m17455_audit.png",
         "caption": "左：7 类受控变异下契约检测器 vs 仅名字基线；中：冲量-动量相对误差 vs 子步数（审计复跑值，含 1e-12 预登记线与 15% 门）；右：审计门清单（复现=yes，已提交/消费方/登记裁判=no）→ L2 依据",
         "png_base64": base64.b64encode(open(png, "rb").read()).decode()}]

payload = {"title": "交差核验 m17455 · XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT → L2（复现真、交付断链）",
           "summary": "board-audit 复跑 m17455 的 repro_cmd：0 处数值差异；8/8 自登记判据 MET，检测率 1.0 vs 基线 0.143。但 7 个交付路径全 untracked、0 消费方、登记裁判（benchmark_cases + xval_cases）均 0 行 → 判 L2，verdict id=572。",
           "scenario_key": "board_audit_dem_library_binding_contract",
           "mission_id": 17496,
           "tags": ["board-audit", "TensorDEM", "XVAL", "library-binding", "frozen-contract", "maturity-L2", "m17455"],
           "markdown": md,
           "figures": figs}
out = post("/tensor/reports", payload, token)
print(json.dumps(out, ensure_ascii=False)[:600])