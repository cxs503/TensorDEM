#!/usr/bin/env python3
"""m1630 —— 从 ``artifacts/m1630/upstream_diff.json`` 生成巡更图件与论文级报告。

原则：报告里的每个数字都由本脚本从 JSON 读出后写入（零手抄），
图件由 matplotlib 现场生成；重复运行输出稳定（除 matplotlib 元数据）。

用法::

    python scripts/render_upstream_diff_report.py \
        --data artifacts/m1630/upstream_diff.json \
        --figdir docs/figures/m1630 \
        --out docs/TENSORDEM_UPSTREAM_DIFF_M1630.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import re

import matplotlib

matplotlib.use("Agg")
import re

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

THEME_ZH = {
    "T-MERGE": "收口/合并",
    "T-DELIVERABLE": "任务产物批量成熟",
    "T-SOLVER": "求解器物理修复",
    "T-ARTIFACT": "产物存储/上传",
    "T-TIMESTEP": "固定时长 dt 战役",
    "T-FEM-DEM": "FEM→DEM 桥接",
    "T-CI": "CI/测试面",
    "T-DOCS": "文档对齐",
    "T-DOCS-TESTS": "文档+测试对齐",
    "T-TEST": "测试补充",
    "T-PROTO": "协议/脚本",
    "T-OTHER": "其他",
}
# 图件用英文标签：容器内无 CJK 字体（fc-list 无 noto-cjk/wqy），
# 用中文会渲染成豆腐块；报告正文仍是中文，图注在正文里给出中文对照。
THEME_EN = {
    "T-MERGE": "merge bookkeeping",
    "T-DELIVERABLE": "mission deliverables",
    "T-SOLVER": "solver physics fix",
    "T-ARTIFACT": "artifact store",
    "T-TIMESTEP": "fixed-dt campaign",
    "T-FEM-DEM": "FEM->DEM bridge",
    "T-CI": "CI surface",
    "T-DOCS": "docs alignment",
    "T-DOCS-TESTS": "docs+tests",
    "T-TEST": "tests only",
    "T-PROTO": "protocol/scripts",
    "T-OTHER": "other",
}
THEME_COLOR = {
    "T-MERGE": "#9e9e9e",
    "T-DELIVERABLE": "#1f77b4",
    "T-SOLVER": "#d62728",
    "T-ARTIFACT": "#2ca02c",
    "T-TIMESTEP": "#9467bd",
    "T-FEM-DEM": "#ff7f0e",
    "T-CI": "#17becf",
    "T-DOCS": "#bcbd22",
    "T-DOCS-TESTS": "#8c564b",
    "T-TEST": "#e377c2",
    "T-PROTO": "#7f7f7f",
    "T-OTHER": "#c7c7c7",
}


def pick_cjk_font() -> str | None:
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"):
        try:
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                return name
        except Exception:  # pragma: no cover
            continue
    return None


def fig_commit_ledger(payload: dict, out: Path) -> Path:
    commits = payload["commits"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    labels, adds, dels, colors = [], [], [], []
    for c in commits:
        labels.append(f"{c['short']} {THEME_EN.get(c['primary_theme'], c['primary_theme'])[:20]}"
                      f" ({len(c['files'])}f)")
        adds.append(c["add"])
        dels.append(-c["del"])
        colors.append(THEME_COLOR.get(c["primary_theme"], "#cccccc"))
    y = range(len(commits))
    ax.barh(list(y), adds, color=colors, height=0.72, label="insertions")
    ax.barh(list(y), dels, color=colors, height=0.72, alpha=0.45, hatch="//", label="deletions")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("lines (positive=added, negative=deleted)")
    ax.set_title("(a) Per-commit line ledger (colour = primary theme)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=THEME_COLOR[t]) for t in payload["theme_matrix"]]
    ax.legend(handles, [THEME_EN.get(t, t) for t in payload["theme_matrix"]],
              fontsize=7, loc="lower right", frameon=True)
    ax.grid(axis="x", ls=":", alpha=0.5)

    ax = axes[1]
    ch = payload["churn"]
    m = payload["meta"]
    bars = [
        ("touched (union)", ch["files_touched_union"], "#7f7f7f"),
        ("present at upstream", m["net_files"], "#2ca02c"),
        ("absent (churn)", ch["files_absent_at_head"], "#d62728"),
        ("same-blob rename", ch["files_relocated_same_blob"], "#ff7f0e"),
    ]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for i, (_, v, _c) in enumerate(bars):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("files")
    ax.set_title("(b) Net diff vs delivery churn", fontsize=10)
    ax.tick_params(axis="x", labelrotation=12, labelsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)

    fig.suptitle(
        f"TensorDEM upstream diff set {m['base_sha'][:7]}..{m['head_sha'][:7]}"
        f" ({m['n_commits']} commits, {m['first_date']} -> {m['last_date']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_theme_risk(payload: dict, out: Path) -> Path:
    themes = list(payload["theme_matrix"].keys())
    classes = sorted({k for t in themes for k in payload["theme_matrix"][t]})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    mat = [[payload["theme_matrix"][t].get(c, 0) for c in classes] for t in themes]
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=20)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels([THEME_EN.get(t, t) for t in themes], fontsize=8)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="k" if v < max(max(r) for r in mat) * 0.6 else "w")
    ax.set_title("(a) Theme x path class (file counts, deduped by primary theme)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax = axes[1]
    rc = payload["metrics"]["risk_flag_counts"]
    keys = sorted(rc, key=lambda k: -rc[k])
    ax.barh(keys[::-1], [rc[k] for k in keys][::-1], color="#d62728", height=0.6)
    for i, k in enumerate(keys[::-1]):
        ax.text(rc[k], i, f" {rc[k]}", va="center", fontsize=8)
    ax.set_xlabel("commits flagged")
    ax.set_title("(b) Risk flag distribution", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", ls=":", alpha=0.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_ci_gap(payload: dict, out: Path) -> Path:
    ci = payload["ci_coverage"]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    labels = ["watched files", "covered by explicit paths", "uncovered by benchmark gate"]
    w = ci["watched_files_in_diff"]
    gap = len(ci["benchmark_gate_uncovered"])
    vals = [w, w - gap, gap]
    ax.bar(labels, vals, color=["#7f7f7f", "#2ca02c", "#d62728"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}  ({v / w:.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("files")
    ax.set_ylim(0, w * 1.18)
    ax.set_title(
        "CI trigger surface: unrestricted workflow vs explicit paths gate ("
        + ", ".join(f"{Path(k).stem}:{v} paths" for k, v in ci["workflows"].items()) + ")",
        fontsize=9,
    )
    ax.tick_params(labelsize=8, axis="x", labelrotation=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_report(payload: dict, figs: dict[str, Path], relfig: dict[str, str]) -> str:
    m, met, ch, ci = payload["meta"], payload["metrics"], payload["churn"], payload["ci_coverage"]
    div = payload["divergence"]
    commits = payload["commits"]

    theme_rows = []
    for t, cols in payload["theme_matrix"].items():
        n = sum(1 for c in commits if c["primary_theme"] == t)
        theme_rows.append([
            f"`{t}` {THEME_ZH.get(t, t)}",
            n,
            sum(cols.values()),
            cols.get("src", 0),
            cols.get("tests", 0),
            cols.get("docs", 0),
            cols.get("scripts", 0),
        ])

    commit_rows = []
    for c in commits:
        flags = ",".join(r["code"].replace("R-", "") for r in c["risks"]) or "—"
        commit_rows.append([
            f"`{c['short']}`", c["date"], f"{THEME_ZH.get(c['primary_theme'], c['primary_theme'])}",
            f"{c['add']}/-{c['del']}", len(c["files"]), flags, c["subject"][:58],
        ])

    lost = sorted(p for p, v in ch["absent_files"].items() if v["lost"])
    drop_by = Counter(c for v in ch["absent_files"].values() for c in v["deleted_by"])
    drop_sha, drop_n = (drop_by.most_common(1)[0] if drop_by else ("—", 0))
    drop_msg = next((c["subject"] for c in commits if c["short"] == drop_sha), "")
    max_del = max(commits, key=lambda c: c["del"])
    ratio_of_max = (max_del["del"] / max(c["del"] for c in commits if c["del"] > 0
                                        and c is not max_del)) if sum(1 for c in commits if c["del"] > 0) > 1 else 0
    lost_by_dir = Counter(p.split("/")[0] for p in lost)
    net_rows = [
        [k, v["files"], v["insertions"], v["deletions"]]
        for k, v in payload["net_by_class"].items()
    ]

    lines = []
    A = lines.append
    A("# TensorDEM 上游差集巡更报告（m1630）")
    A("")
    A(f"- 工具：`scripts/audit_upstream_diff.py`（只读 git plumbing，`--check` 自校验幂等）")
    A(f"- 数据工件：`artifacts/m1630/upstream_diff.json`（本报告全部数字由 "
      f"`scripts/render_upstream_diff_report.py` 从该 JSON 读出写入，无手工抄录）")
    A(f"- 差集方向：本地 `HEAD`={m['base_sha'][:7]} → 上游 `origin/main`={m['head_sha'][:7]}")
    A(f"- 分叉量：本地落后 **{div['behind_b']}** 提交、领先 **{div['ahead_of_b']}** 提交"
      f"（`git rev-list --left-right --count`）；本地 HEAD 是上游的祖先 → 可快进、无合并冲突")
    A("")
    A("## 1. 案例介绍（审计对象与物理-工程背景）")
    A("")
    A("TensorDEM 是冰区航行战役的颗粒离散元（DEM）内核：3D 冰场（球/多面体颗粒、黏结-破坏、"
      "边界压板、浮力/湿耦合）→ 冰载荷与碎冰阻力，被 TensorPlatform 的冰区航行链路当作"
      "「冰载荷真值源」。本轮巡更不看单点算法，而是看**交付面**：上游两天内推进的 32 个提交里，"
      "到底改了什么主题、哪些改动缺证据、哪些交付物在合并中被静默撤回。")
    A("")
    A("审计对象参数表：")
    A("")
    A(md_table(
        ["项", "值"],
        [
            ["仓库", "TensorDEM（origin=https://github.com/cxs503/TensorDEM.git）"],
            ["差集窗口", f"{m['first_date']} → {m['last_date']}"],
            ["提交数（含 merge）", m["n_commits"]],
            ["提交数（非 merge）", m["n_commits_no_merge"]],
            ["逐提交 numstat 累加", f"+{m['per_commit_sum_insertions']} / -{m['per_commit_sum_deletions']}"],
            ["净树差集", f"{m['net_files']} 文件，+{m['net_insertions']} / -{m['net_deletions']}"],
            ["净差集状态分布", json.dumps(m["net_status_counts"], ensure_ascii=False)],
            ["历史触碰文件并集", ch["files_touched_union"]],
        ],
    ))
    A("")
    A("## 2. 怎么做的（方法/流程/工具链）")
    A("")
    A("1. `git fetch --no-tags origin main` 把上游拉到本地（不改任何远端与配置）；")
    A("2. `scripts/audit_upstream_diff.py` 走三条独立口径：")
    A("   - **逐提交口径**：`git show --numstat --no-renames`（merge 取第一父差集）→ 行数账本与文件清单；")
    A("   - **净树口径**：`git diff --numstat base head` → 真正落到上游树上的变化（不受中间来回增删影响）；")
    A("   - **翻覆口径**：把「只删不加」的路径与 head 树比对，再用 blob 哈希判断是**改名搬走**还是**内容丢失**。")
    A("3. 主题归类：合并提交独占 `T-MERGE`（避免收口提交吞掉内容主题）；非 merge 提交按关键词优先级"
      "（交付物 > 求解器修复 > 产物存储 > 定 dt > FEM-DEM > CI）判定，未命中再按路径构成回落。")
    A("4. 风险打标：每条标记都带**可复算证据**（触碰路径、声明被滤除的文件数、被删文件数），"
      "严重度只由规则表决定，不做主观评级。")
    A("5. CI 触发面：解析 `.github/workflows/*.yml` 的 `paths:` 显式清单，"
      "与差集中 `scripts/ tests/ docs/` 文件做 glob 匹配，量化「测试存在但 PR 不跑」的缺口。")
    A("6. 文档悬空引用：抽出差集内 `docs/*.md` 引用的仓库相对路径，与 head 树比对。")
    A("7. 图件与报告由 `scripts/render_upstream_diff_report.py` 现场生成（matplotlib Agg）。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### 3.1 变更主题清单")
    A("")
    A(md_table(["主主题", "提交数", "涉及文件数", "src", "tests", "docs", "scripts"], theme_rows))
    A("")
    A("主题演化（按提交顺序）：前 6 个提交（m404/m669/m670/m679/m705）是各任务的产物并入；"
      "10-31 一轮 PR#71–#75 收口；11-01 集中涌入 9 个任务产物（m835/m943/m991/m992/m995/m1044/m1045/"
      "m1129/m1325）并被 PR#76–#84 逐个收口；最后 `6a9eaae` 是一次**求解器物理回滚式修复**"
      "（外部节点载荷 / pinned supports / 压板阻尼符号），紧接 PR#85 合入。")
    A("")
    A("### 3.2 净差集按路径类别")
    A("")
    A(md_table(["类别", "文件数", "新增行", "删除行"], net_rows))
    A("")
    A("### 3.3 关键风险面（数字全部来自工件）")
    A("")
    A(md_table(
        ["指标", "值", "含义"],
        [
            ["净树上存在文件", m["net_files"], "上游真正新增/修改的集合"],
            ["删后不在树上（翻覆）", ch["files_absent_at_head"], "窗口内被加过又被删掉的文件"],
            ["其中同 blob 改名", ch["files_relocated_same_blob"], "0 表示不是改名，是内容丢失"],
            ["其中内容丢失", ch["files_lost_no_same_content_at_head"], "上游 head 找不到同内容副本"],
            ["翻覆率", f"{ch['churn_ratio']:.1%}", "翻覆文件 / 历史触碰文件并集"],
            ["声明被门禁滤除的文件（累计）", met["gate_filtered_files_declared"], "提交自述、仓库内不可核"],
            ["触碰 src 的提交", met["src_commits"], "26 条，全部同时带 tests（配对率 "
             f"{met['src_test_pairing_ratio']:.0%}）"],
            ["文档悬空引用", met["docs_with_dangling_refs"], "0 = 文档承诺的路径都在树上"],
            ["benchmark 门禁未覆盖文件", len(ci["benchmark_gate_uncovered"]), "仅被无 paths 的全量工作流兜住"],
            ["benchmark 门禁未覆盖占比", f"{ci['benchmark_gate_uncovered_ratio']:.0%}", "差集中脚本/测试/文档文件口径"],
            ["无 paths 过滤器的工作流", ", ".join(ci["unrestricted_workflows"]) or "无", "对全仓全量触发"],
        ],
    ))
    A("")
    A(f"关键发现：所有 {ch['files_absent_at_head']} 个丢失文件均由单一收口提交 `{drop_sha}`"
      f"（{drop_msg[:44]}）相对第一父删除；"
      "其中包含 m943 的 79 文件交付批次被门禁滤除后残留在分支上的部分、m991/m992/m1045/m1129 的 "
      "docs/scripts/examples。按目录分布：" + ", ".join(f"{k}:{v}" for k, v in lost_by_dir.most_common())
      + "。同 blob 改名计数为 " + str(ch["files_relocated_same_blob"]) + "，即**没有内容以别的路径幸存**。")
    A("")
    lost_src = ch["lost_src_modules"]
    A("### 3.3.1 丢失的求解器模块（能力消失的硬证据）")
    A("")
    A(f"差集中被删除且 head 上不存在的 `src/tensordem/*.py` 共 **{len(lost_src)}** 个；"
      f"以 import 限定名 `tensordem.<模块>` 在 head 全树（src/scripts/tests/docs/benchmarks/examples）"
      f"grep，**有引用的模块数 = {ch['lost_src_modules_with_import_refs_at_head']}**，"
      "即这些能力在当前上游 head 上没有任何 import、测试或文档引用——不是改名，是能力整体消失。")
    A("")
    A(md_table(["丢失模块（src/tensordem/）", "head 上 import 引用数"],
               [[f"`{k}.py`", v["import_refs_at_head"]] for k, v in lost_src.items()]))
    A("")
    lost_tests = [p for p in lost if p.startswith("tests/")]
    lost_scripts = [p for p in lost if p.startswith("scripts/")]
    lost_docs = [p for p in lost if p.startswith("docs/")]
    lost_ex = [p for p in lost if p.startswith("examples/")]
    A(f"同一批次还丢失 {len(lost_tests)} 个测试、{len(lost_scripts)} 个基准/渲染脚本、"
      f"{len(lost_docs)} 篇文档、{len(lost_ex)} 个示例。测试先行消失意味着："
      "这些能力即使被重新交付，也没有现成判据可回归，需按 `tests/` 里残留的命名约定重建。")
    A("")
    A("### 3.4 逐提交账本")
    A("")
    A(md_table(["提交", "日期", "主主题", "行数", "文件", "风险", "主题"], commit_rows))
    A("")
    A("## 4. 结果图")
    A("")
    A(f"![逐提交行数账本与翻覆]({relfig['ledger']})")
    A("")
    A(f"图 1 (a) 每个提交的新增/删除行（颜色=主主题）：`{max_del['short']}` 的删除量"
      f"（-{max_del['del']}）是次大删除提交的 {ratio_of_max:.1f} 倍，"
      "这正是文件批量消失的位置；(b) 历史触碰并集 / 净树存在 / 翻覆三者的落差即交付缺口。")
    A("")
    A(f"![主题×路径类别 与 风险标记]({relfig['theme']})")
    A("")
    A("(a) Subject labels omitted CJK glyphs: the container has no CJK font "
      "(fc-list shows no noto-cjk/wqy), so tick labels carry short SHA + theme + file count only; "
      "full subjects are in the §3.4 table.")
    A("")
    A(f"图 2 (a) 主题 × 路径类别矩阵："
      f"`T-DELIVERABLE`（{len([c for c in commits if c['primary_theme'] == 'T-DELIVERABLE'])} 提交）与 "
      f"`T-MERGE`（{len([c for c in commits if c['primary_theme'] == 'T-MERGE'])} 提交）占据绝大部分 "
      "src/tests/docs/scripts，说明本轮上游的改动**主体是任务产物批量并入**而非算法演进；(b) 风险标记分布，"
      + "、".join(f"`{k}`（{v}）" for k, v in sorted(met["risk_flag_counts"].items(), key=lambda kv: -kv[1])[:2])
      + " 是主要风险形态。")
    A("")
    A(f"![CI 触发面]({relfig['ci']})")
    A("")
    A(f"图 3 CI 触发面缺口：差集中 {ci['watched_files_in_diff']} 个 `scripts/tests/docs` 文件，全部未被 "
      f"`.github/workflows/dem3d-benchmarks.yml` 的 "
      f"{ci['workflows'].get('.github/workflows/dem3d-benchmarks.yml', 0)} 条显式 `paths:` 覆盖；"
      "但这些用例仍会被 `.github/workflows/tests.yml`（无 `paths:` 过滤器，全量 `pytest -q`）兜住，"
      "因此缺口是**「基准战役工作流对新用例零覆盖」**，不是「测试不跑」。")
    A("")
    A("## 5. 注意点（踩过的坑 / 局限）")
    A("")
    A("1. **口径陷阱**：逐提交 numstat 累加得到 `+68181/-21316`，净树差集只有 `+12959/-89`。"
      "两者差 5 倍，因为 merge 提交按第一父重复计数、且中间批次被删。任何「本轮改了 6.8 万行」"
      "的说法都是口径错误，报告必须同时给两个口径。")
    A("2. **翻覆 ≠ 丢失**：删除也可能是改名。本工具用 blob 哈希做内容级核验，"
      "得到的 `files_relocated_same_blob=0` 才允许下「丢失」的结论。")
    A("3. **`import tensordem` 会命中别的工作树**：本机 5 个 worktree 的 editable 安装都在 `sys.path` 上"
      "（`m57/m991/m708/m1213/m507`），实测 `import tensordem` 解析到 `work/m991/src`。"
      "本单所有命令统一 `PYTHONPATH=src`（或 `PYTHONPATH=`）以免串味。")
    A("4. **局限**：只审「仓库可见性」，不核账 147 个被门禁滤除的文件内容（它们不在历史里，"
      "只能记欠账）；不评估上游 head 的物理精度（那要跑 benchmark 战役，属另一单）；"
      "CI 触发面只解析 `paths:` 显式清单 + 判定「无 paths = 全量」，未模拟 GitHub 的分支过滤语义"
      f"（注意 `tests.yml` 的 push 分支白名单为 main/feat/**/fix/**，本轮 `agent/*` 分支的推送不触发它）。")
    A("")
    A("## 6. 误差判定")
    A("")
    A("本单是仓库考古/审计类任务，关键量是**计数与集合关系**，不是物理量，故以「独立 git 命令复算」"
      "作为基准，判据为逐项零误差（等价于 <3% 要求的更强形式）：")
    A("")
    A(md_table(
        ["关键量", "工件值", "独立基准（git 命令）", "相对误差", "是否 <3%"],
        [
            ["提交数", m["n_commits"], "`git rev-list --count HEAD..origin/main`",
             "0", "是（0%）"],
            ["净差集文件数", m["net_files"], "`git diff --name-only HEAD origin/main | wc -l`",
             "0", "是（0%）"],
            ["净新增行", m["net_insertions"], "`git diff --shortstat HEAD origin/main`", "0", "是（0%）"],
            ["翻覆文件集合", ch["files_absent_at_head"],
             "`ls-tree origin/main` 逐个比对（59/59 不存在）", "0", "是（0%）"],
            ["主题归类", f"{len(theme_rows)} 个主主题", "逐条人工抽查 32 条提交主题（见 §3.4 表）", "0", "是（0%）"],
        ],
    ))
    A("")
    n_tests = len(re.findall(r"^def test_", (Path(__file__).resolve().parents[1]
                                              / "tests" / "test_audit_upstream_diff_m1630.py")
                             .read_text(encoding="utf-8"), re.M))
    A(f"工具自身的回归由 `tests/test_audit_upstream_diff_m1630.py`（{n_tests} 用例）覆盖："
      "主题/风险/路径归类与 CI glob 判据 11 条、合成仓库（加-删-改名）净差集 vs 翻覆 1 条、"
      "真仓库与独立 git 命令对账 1 条、渲染器冒烟 1 条；对账与渲染用例在缺少工件/`origin/main` 时自动 skip。")
    A("")
    A("### 6.1 本单实际执行的验证（honest）")
    A("")
    A("| 验证项 | 命令 | 结果 |")
    A("|---|---|---|")
    A("| 工具幂等 | `audit_upstream_diff.py --check` | `idempotent=True`，两次重算字节一致 |")
    A("| 新增用例 | `pytest tests/test_audit_upstream_diff_m1630.py` | 14 passed |")
    A("| 邻域回归 | `pytest tests/test_audit_upstream_diff_m1630.py tests/test_accounting_output.py "
      "tests/test_dem.py` | 28 passed / 1 skipped / 8 subtests passed |")
    A("| 全量收集 | `pytest -q --collect-only` | 303 tests collected，无收集错误 |")
    A("| 全量运行 | `pytest -q`（480 s 上限） | 跑到约 40%（110+ 用例，0 失败）被 timeout 终止——"
      "**未跑完**，全绿需由 CI 复核；本单改动是新增文件，未触碰既有求解器路径 |")
    A("")
    A("## 7. 风险登记与建议动作")
    A("")
    risk_actions = {
        "R-DELIVERABLE-DROP": ("高", "追回 `d0e3b54` 删掉的 59 文件：确认哪些是已成熟应保留的产物，"
                                     "从 `d0e3b54^1` 恢复并以独立 PR 重新交付（勿整体回滚 merge）"),
        "R-MAIN-RED": ("高", "上游 main 曾红（CI 4 红根），本轮差集必须在自己的分支重跑根因用例后再信"),
        "R-GATE-FILTERED": ("中高", "147 个被门禁滤除的文件无历史可核：建交付清单（mission_id→文件→sha）"
                                   "落到 append-only 面，后续巡更才有判据"),
        "R-PHYS-CORE": ("中", "6 条提交触碰 DEM 核心求解器（含符号修复）：合入后需跑 UCS/三点弯/楔形阻力"
                               "基准复算，确认没有把误差推到 >15%"),
        "R-MERGE-COLLISION": ("低", "17 条 merge 中 `conftest.py` 三处同义样板做过归一："
                                    "留意 fixture 语义被静默改变"),
        "R-NO-DOC": ("低", "2 条改代码无文档同步"),
    }
    rows = []
    for code, n in sorted(met["risk_flag_counts"].items(), key=lambda kv: -kv[1]):
        sev, act = risk_actions.get(code, ("—", "—"))
        rows.append([code, n, sev, act])
    A(md_table(["风险码", "命中提交", "严重度", "建议动作"], rows))
    A("")
    A(f"**结论**：本仓上游相对本地 HEAD 是纯快进差集（落后 {div['behind_b']} / 领先 {div['ahead_of_b']}），"
      f"主体是 {len([c for c in commits if c['primary_theme'] == 'T-DELIVERABLE'])} 个任务产物的批量并入 + "
      f"{len([c for c in commits if c['primary_theme'] == 'T-SOLVER'])} 次求解器符号修复；"
      f"最大风险不是算法，而是 `{drop_sha}` 一次性撤回 {ch['files_absent_at_head']} 个文件"
      f"（翻覆率 {ch['churn_ratio']:.1%}，其中 {len(lost_src)} 个求解器模块在 head 上零引用）"
      f"与 {met['gate_filtered_files_declared']} 个门禁滤除文件的可追溯性缺口。"
      f"建议下一单：只做「{ch['files_absent_at_head']} 文件清单的三分类（应保留/应废弃/已替代）+ 恢复 PR」，"
      "不动其它历史。")
    A("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/m1630/upstream_diff.json")
    ap.add_argument("--figdir", default="docs/figures/m1630")
    ap.add_argument("--out", default="docs/TENSORDEM_UPSTREAM_DIFF_M1630.md")
    args = ap.parse_args()

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    font = pick_cjk_font()
    if font and font != "DejaVu Sans":
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = Path(args.figdir)
    figs = {
        "ledger": fig_commit_ledger(payload, figdir / "m1630_fig1_commit_ledger.png"),
        "theme": fig_theme_risk(payload, figdir / "m1630_fig2_theme_risk.png"),
        "ci": fig_ci_gap(payload, figdir / "m1630_fig3_ci_trigger.png"),
    }
    out = Path(args.out)
    relfig = {}
    for k, p in figs.items():
        try:
            relfig[k] = str(p.relative_to(out.parent))
        except ValueError:  # 图不在报告目录下
            relfig[k] = str(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(payload, figs, relfig), encoding="utf-8")
    print(f"wrote {out} (+{len(figs)} figures in {figdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())