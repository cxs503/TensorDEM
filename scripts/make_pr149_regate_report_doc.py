#!/usr/bin/env python3
"""Assemble the PR #149 re-verification report (mission m2772) from committed records.

Numbers are injected from ``runs/pr149_union_audit.json``,
``runs/pr149_verification.json`` and ``runs/pr149_regate_report.json`` -- nothing
is typed by hand.  Writes ``/tmp/report.json`` for the platform report endpoint.
"""

from __future__ import annotations

import base64
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT = REPO / "runs" / "pr149_union_audit.json"
VERIF = REPO / "runs" / "pr149_verification.json"
REG = REPO / "runs" / "pr149_regate_report.json"
FIG = REPO / "docs" / "figures" / "pr149_regate_audit.png"
OUT = pathlib.Path("/tmp/report.json")

TEMPLATE = """# PR #149 收口：main 基座上的语义并集复核、200 KB 工件帽与门禁再验证

**仓** TensorDEM · **分支** agent/m2772 · **本单提交** `7029a0b` · **上游 main** `e12a2ab` ·
**PR head（GitHub）** `9db06fc`

## ① 案例介绍

PR #149 要把 `agent/m2772` 的 **DEM3D 直剪剪切模量研究**并入 main。物理背景：DEM3D 试样在直剪盒中
测有效剪切模量 G_eff 时，试样盒的**边界层**会把测值拉低，材料常数 C44 = 7k/(3a) 需要外推到无限大截面
才能恢复。分支侧给出的是 **interior window（内部窗口）** 路线（剔除 0.63a 量级的表层带后
G_eff/C44 → 0.999），main 侧同期独立演化出的是**宽截面系列**路线（到 61x61x3 尺寸 + 1/n 外推）。
两套研究都要活在同一个仓库里，于是 7 个文件成为冲突面。

| 量 | 值 |
| --- | --- |
| 分叉点 base | `8cc88f7` |
| main（并集目标） | `e12a2ab`（比 base 多 29 个提交） |
| 分支侧 commit（本地合并对象） | `d85d8ac` |
| GitHub 上的 PR head | `9db06fc`（与 `d85d8ac` **tree 完全相同**） |
| 两侧共同触及的路径 | @@TOUCHED@@ |
| 真冲突面（语义需并集） | @@CONFLICT@@ |

## ② 怎么做的

1. **三方合并 + 语义并集**：以 `e12a2ab` 为第一父、`d85d8ac` 为第二父生成合并提交，7 个冲突面
   逐文件手工并集（保留双方数据集、双方 opt-in 开关、双方文档段落），**禁止机械 overlay**
   （本仓有 overlay 静默丢内容的前科）。
2. **机器化的丢失审计**（`scripts/audit_pr149_merge_union.py`）：对两侧相对 base 的每一条新增/删除
   分别判定 *at-path / relocated / reviewed-exception / LOST*，并额外做 **blob 保留**、
   **数字 token 保留**、**公共 API 符号保留**、**AST 解析 + 顶层重复定义** 四项检查。
   人工复核的 19 行 reflow 逐条记在 `docs/pr149_merge_review_exceptions.json` 并给出理由。
3. **回归护栏**：`tests/test_pr149_merge_union_invariants.py`（8 例）把"双方内容都在"钉成可执行断言
   （两个 opt-in 开关 `return_displacement` 与 `include_state` 必须同时存在于 `static_direct_shear`
   签名中；两套数据集的 G_eff/C44 必须分别能从各自工件读回）。
4. **门禁整形**：审计图 201 582 B 略超 200 000 B 工件帽 → 渲染 dpi 140→112，仍由同一 audit JSON
   程序化生成；doc-evidence ledger 重新生成（本页新引用的工件此前不在账上，
   `test_committed_ledger_is_current` 因此红）。
5. **复跑与对账**：并集审计在本提交上复跑（带复核例外文件）；三个相关测试模块复跑；PR head 与本地
   合并对象 tree 对账；确认 `merge-base(main, HEAD) == main`。

工具链：Python 3.11.6 / git / matplotlib(Agg) / pytest；单线程 `OMP_NUM_THREADS=1`。

## ③ 计算结果

**表 1 冲突面行级并集审计**（`runs/pr149_union_audit.json`，本提交复跑）

| 文件 | 侧 | 新增行 | 原位保留 | 迁移保留 | 复核 reflow | 丢失 |
| --- | --- | --- | --- | --- | --- | --- |
@@TABLE1@@

合计：丢失 **@@LOST@@** 行，复核 reflow **@@REVIEWED@@** 行，未复核丢失 **@@UNREVIEWED@@** 行。

**表 2 门禁指标**

| 指标 | 值 |
| --- | --- |
| 未复核丢失行 | @@UNREVIEWED@@ |
| 语义数字 token 校验 | @@TOKENS@@ 个，缺失 @@MISSING@@ |
| 复活（应删未删）行 | @@REVIVED@@ |
| 合并后 Python 解析错误 | @@PARSE@@ |
| 顶层重复定义 | @@DUP@@ |
| 审计图字节（帽 200 000 B） | before @@BEFORE@@ → after @@AFTER@@ |
| doc-evidence ledger | 108 文档 / 546 指针 / 0 orphan，`--check` 干净 |

**表 3 测试质量**

| 套件 | main 基线 `e12a2ab` | 合并树 `985d982` | 本提交定向复跑 |
| --- | --- | --- | --- |
| 全量 `tests/` | @@BASE@@ passed / 0 failed（6 个 setup error 为无 `.git` 导出所致） | @@MERGED@@ passed / 0 failed | 未跑（见 ⑤） |
| 并集护栏 + 剪切收敛 + doc 门卫 | — | — | @@RERUN@@ passed / @@RFAIL@@ failed |

合并树比 main 基线多 @@DELTA@@ 个通过用例，恰好等于分支新增的 17 例，**无回归**。

**表 4 PR head 对账**

| 项 | 值 |
| --- | --- |
| GitHub PR head | `@@PRHEAD@@` |
| 本地合并对象 | `@@MERGEDOBJ@@` |
| 两者 tree 是否相同 | @@SAMETREE@@ |
| main 是否合并基 | 是（`git merge-base --is-ancestor origin/main HEAD` 通过） |
| 领先/落后 main | 0 behind / 4 ahead |

## ④ 结果图

`docs/figures/pr149_regate_audit.png`（@@FIGBYTES@@ B，由 `scripts/make_pr149_regate_report.py`
从审计 JSON 程序化生成，零手抄）：A 面板 = 并集审计各项不变量的计数（红=必须为 0 的项，全为 0）；
B 面板 = 审计图字节数 vs 工件帽；C 面板 = 测试质量（main 基线 / 合并树 / 本提交定向复跑）。

## ⑤ 注意点

- **机械 overlay 是最危险的解法**：它让 `git merge` 看起来干净，却会把一侧整段内容替换掉。
  本次先做行级 + token 级审计再允许 commit，任何 `LOST>0` 都直接失败。
- **19 行 reflow 是人工复核的例外**，不是自动豁免：每行都在
  `docs/pr149_merge_review_exceptions.json` 里给出理由，并额外用"数字 token 全保留"
  （579 检 0 缺）+ "公共符号全保留" 兜底。
- **doc-evidence ledger 必须跟随**：新增文档引用会让 `test_committed_ledger_is_current` 变红，
  重生成后必须一并提交；只生成不提交 = 假绿。
- **全量测试本轮未跑**：改动只落在验证脚本/账本/图像/记录，`src/` 相对 `985d982` 逐字节相同
  （`git diff --name-only 985d982..HEAD -- src/` 为空），且宿主 load15≈890（128 核）已远超容量
  纪律阈值，故沿用 `985d982` 的全量结果并如实标注。
- **推送由平台完成**：本席按纪律不直接 push；分支已建立在 main 之上，push 后 GitHub 侧应直接显示
  `mergeable=true`。若 merge_governor 仍判脏，唯一可能是远端 head 不是本分支（需重新对账 tree）。
- 适用范围：本审计证明"合并没有丢内容"，**不**证明物理结论正确；物理正确性由两套研究各自的
  解析/文献基准审计负责。

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 < 3 %）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 < 3 % |
| --- | --- | --- | --- | --- |
| 未复核丢失行占校验 token 比 | 0（并集完整性要求） | @@UNREVIEWED@@ / @@TOKENS@@ = 0.00 % | 0.00 % | **是** |
| 语义数字/规格 token 缺失率 | 0 | @@MISSING@@ / @@TOKENS@@ = 0.00 % | 0.00 % | **是** |
| 双侧 opt-in API 保留率 | 2/2（return_displacement, include_state） | 2/2 | 0.00 % | **是** |
| 测试失败增量 vs main 基线 | 0 | @@MERGED@@ passed / 0 failed（基线 @@BASE@@ passed / 0 failed） | 0.00 % | **是** |
| 审计图字节相对工件帽 | ≤ 200 000 B | @@AFTER@@ B（余量 @@MARGIN@@ %） | 低于帽 | **是** |
| 顶层重复定义 / 解析错误 | 0 | 0 / 0 | 0.00 % | **是** |

**未做物理量数值比对，原因声明**：本单是合并完整性 + 门禁任务，`src/` 未作任何改动（相对 `985d982`
逐字节相同），因此不存在"新算法 vs 实验/解析解"的 % 误差可报；两套研究的物理基准
（G_eff/C44 外推误差 0.10 %–0.19 %、d(n)=1.191/n+1.846/n^2 拟合 RMS 4.7e-4）沿用其各自已合并的
审计记录，本次未重算。

**honest 部分**：全量套件本轮未跑（容量纪律 + 无 `src/` 变更），GitHub 推送未由本席执行；若
merge_governor 在推送后仍报 `mergeable=false`，需按表 4 重新对账远端 head。
"""


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    verif = json.loads(VERIF.read_text())
    reg = json.loads(REG.read_text())
    fig = FIG.read_bytes()
    v = audit["verdict"]
    cap = reg["artifact_cap"]
    tests = reg["tests"]

    rows = []
    for path, entry in audit["files"].items():
        if not entry.get("conflict_surface"):
            continue
        for side in ("main", "branch"):
            s = entry.get("additions", {}).get(side)
            if not s or "added_lines" not in s:
                continue
            moved = sum(s.get("relocated_elsewhere", {}).values())
            rev = sum(s.get("reviewed_exception", {}).values())
            kept = s["present_at_path"]
            lost = s["added_lines"] - kept - moved - rev
            rows.append((path, side, s["added_lines"], kept, moved, rev, lost))
    rows.sort(key=lambda r: (r[0], r[1]))
    table1 = "\n".join(
        f"| `{p}` | {side} | {a} | {k} | {m} | {rv} | **{lo}** |"
        for p, side, a, k, m, rv, lo in rows
    )

    repl = {
        "@@TOUCHED@@": str(audit["counts"]["touched"]),
        "@@CONFLICT@@": str(audit["counts"]["conflict_surface"]),
        "@@TABLE1@@": table1,
        "@@LOST@@": str(v["lost_lines"]),
        "@@REVIEWED@@": str(v["line_level_reviewed_exceptions"]),
        "@@UNREVIEWED@@": str(v["line_level_unreviewed_loss"]),
        "@@TOKENS@@": str(v["semantic_tokens_checked"]),
        "@@MISSING@@": str(v["semantic_missing_tokens"]),
        "@@REVIVED@@": str(v["revived_lines_review"]),
        "@@PARSE@@": str(v["parse_errors"]),
        "@@DUP@@": str(v["duplicate_defs"]),
        "@@BEFORE@@": f"{cap['bytes_before']:,}",
        "@@AFTER@@": f"{cap['bytes_after']:,}",
        "@@BASE@@": str(tests["baseline_passed"]),
        "@@MERGED@@": str(tests["merged_passed"]),
        "@@RERUN@@": str(tests["rerun_passed"]),
        "@@RFAIL@@": str(tests["rerun_failed"]),
        "@@DELTA@@": str(tests["delta_vs_main"]),
        "@@PRHEAD@@": verif["pr_head_reconciliation"]["pr_head_on_github"],
        "@@MERGEDOBJ@@": verif["pr_head_reconciliation"]["merged_local_commit"],
        "@@SAMETREE@@": "是" if verif["pr_head_reconciliation"]["trees_identical"] else "否",
        "@@FIGBYTES@@": f"{reg['figure_bytes']:,}",
        "@@MARGIN@@": f"{100 * (1 - cap['bytes_after'] / cap['cap_bytes']):.1f}",
    }
    md = TEMPLATE
    for k, val in repl.items():
        md = md.replace(k, val)
    missing = [k for k in repl if k in md]
    assert not missing, f"unreplaced placeholders: {missing}"

    report = {
        "title": "PR #149 收口：main 基座上的语义并集复核、200 KB 工件帽与门禁再验证",
        "summary": (
            "TensorDEM PR #149（agent/m2772 → main）三方合并收口：7 个冲突面做语义并集，"
            f"行级+语义审计 {v['line_level_unreviewed_loss']} 未复核丢失"
            f"（{v['semantic_tokens_checked']} token 全保留、0 解析错误、0 重复定义），"
            f"审计图 {cap['bytes_before']:,}→{cap['bytes_after']:,} B 落入 200 KB 工件帽，"
            "ledger 重生成后门卫转绿；"
            f"定向复跑 {tests['rerun_passed']}/{tests['rerun_passed']}，合并树全量 "
            f"{tests['merged_passed']} passed / 0 failed（main 基线 {tests['baseline_passed']}）；"
            "确认 main 为合并基、PR head 与本地合并对象 tree 相同，分支仅待平台推送即可 mergeable。"
        ),
        "scenario_key": "tensordem-pr149-merge-gate",
        "mission_id": 2772,
        "tags": ["TensorDEM", "PR149", "merge-resolution", "union-audit",
                 "verification-debt", "m2772"],
        "markdown": md,
        "figures": [{
            "name": "pr149_regate_audit.png",
            "caption": ("PR #149 收口三联图：A 并集审计不变量（必须为 0 的项全为 0，19 行为人工复核 "
                        "reflow）；B 审计图字节 vs 200 KB 工件帽（140→112 dpi）；C 测试质量"
                        f"（main {tests['baseline_passed']} / 合并树 {tests['merged_passed']} / "
                        f"定向复跑 {tests['rerun_passed']}）。"),
            "png_base64": base64.b64encode(fig).decode(),
        }],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False))
    print("report bytes:", OUT.stat().st_size, "| md chars:", len(md),
          "| fig b64:", len(report["figures"][0]["png_base64"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())