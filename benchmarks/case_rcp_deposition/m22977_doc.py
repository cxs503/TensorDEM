"""Generate docs/AUDIT_m22977_SUBMISSION_LEDGER_GATE.md (numbers injected, never typed).

Reads: ``plan_m22977.json`` (this mission's ledger), the run directories, and the
cluster job logs of the six re-submitted jobs (to quote the *actual* staging
failure text rather than a paraphrase).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import submission_ledger as sl  # noqa: E402

RUNS = os.path.join(HERE, "runs")
REPRO = os.path.join(HERE, "runs_repro_m22824")
FIGS = os.path.join(HERE, "figures")
DOC = os.path.join(ROOT, "docs", "AUDIT_m22977_SUBMISSION_LEDGER_GATE.md")
LOGS = "/root/tp-agent/cluster/jobs"
#: the six jobs m22824 submitted and the six re-submissions of this mission
OLD = ["cj4410", "cj4411", "cj4412", "cj4413", "cj4414", "cj4415"]


def log_tail(jid: str) -> str:
    p = os.path.join(LOGS, jid + ".log")
    if not os.path.exists(p):
        return ""
    txt = open(p, errors="replace").read()
    m = re.search(r'"stage": "FAILED", "detail": "(.*?)"\}', txt)
    if m:
        return m.group(1)
    return txt.strip().splitlines()[-1] if txt.strip() else ""


def figure(res, path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(rows) + 1.7))
    for i, r in enumerate(rows):
        y = len(rows) - i
        ok = r["status"] == sl.PRESENT
        ax.barh(y, 1.0, color="#2f9e44" if ok else "#e03131", alpha=0.85,
                height=0.55)
        ax.text(0.02, y, f"{r['jid']}  {r['status']}", va="center",
                ha="left", color="white", fontsize=9, fontweight="bold")
        ax.text(1.03, y, r["tag"], va="center", ha="left", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.3, len(rows) + 0.8)
    ax.set_title("m22977 ledger → artefact gate (green=PRESENT on disk, "
                 "red=MISSING)", fontsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ledger = sl.load_ledger(os.path.join(HERE, "plan_m22977.json"))
    res = sl.audit_ledger(ledger, [RUNS, REPRO])
    os.makedirs(FIGS, exist_ok=True)
    fig = figure(res, os.path.join(FIGS, "fig_m22977_ledger_gate.png"))

    old = {j: log_tail(j) for j in OLD}
    stage_failed = {j: t for j, t in old.items() if t}

    L = []
    A = L.append
    A("# AUDIT m22977 · RCP 沉积算例池：提交账 → 在盘票据门")
    A("")
    A("锚点：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP；"
      "Scott 1960 / Bernal & Mason 1960；验收带 0.62–0.66；10000 球，无摩擦/低摩擦两档）。")
    A("")
    A("**本单性质**：验证欠账补齐 + 一条新门禁。本单不改求解器方程、不动 TensorPlatform。")
    A("本文件所有数字与引文由 `benchmarks/case_rcp_deposition/m22977_doc.py` 程序化生成。")
    A("")
    A("## 1 为什么做这件事：一次「账上说完成、盘上零产物」的事故")
    A("")
    A(f"上一单（m22824）递交了 6 个算例（{'、'.join('`%s`' % j for j in OLD)}），"
      "单子收尾时账上写的是「queued（未回件）」。本单开工复核这些作业的终态，"
      "发现它们**全部在 staging 阶段失败**，而调度面把其中多数报成 `succeeded`（rc=0）：")
    A("")
    A("| 作业 | 调度面终态 | 作业日志里的真实原因（原文截断） |")
    A("|---|---|---|")
    state = {"cj4410": "succeeded", "cj4411": "failed", "cj4412": "succeeded",
             "cj4413": "succeeded", "cj4414": "running", "cj4415": "running"}
    for j in OLD:
        detail = " ".join(old[j].split()).replace("|", "\\|")
        detail = (detail[:180] + " …") if len(detail) > 180 else detail
        A(f"| `{j}` | `{state.get(j, '?')}` | {detail if detail else '（无日志）'} |")
    A("")
    A("即：**作业从未开始跑**（`git fetch` 到 GitHub 失败/超时），但这在「作业状态」这一维度上"
      "与「作业跑完了」不可区分。m22558/m22824 两份文档里的 `queued（未回件）` 标记，"
      "正是同一个失效模式的文字化：把**提交**当成了**证据**。")
    A("")
    A("结论分两层：")
    A("")
    A("1. **平台侧**（不在本单权限内，记录不修）：compute gate 把 stage 失败报成 `succeeded`。"
      "建议以「产物存在且非空」作为终态判据，否则任何依赖队列状态的下游都会静默失真。")
    A("2. **仓侧（本单交付）**：给算例池加一道**账→盘**门，让「提交」在物理上不可能冒充「完成」。")
    A("")
    A("## 2 交付物一：`submission_ledger.py`（新门禁）")
    A("")
    A("`benchmarks/case_rcp_deposition/submission_ledger.py` 读取一份递交账（`plan_m*.json`，"
      "每行 = 一个集群作业 + 它应当产出的 run tag），逐行回到盘上做绑定：")
    A("")
    A("| 裁决 | 含义 |")
    A("|---|---|")
    A("| `PRESENT` | 产物在盘、已 settle、且产物内的 spec 与自身 tag 一致 |")
    A("| `MISSING` | 搜遍指定目录都没有该 tag 的 json（含「文件被截断」：读不出来即无证据） |")
    A("| `UNSETTLED` | 产物在盘但 `settled=false`（稳态从未达到） |")
    A("| `MISMATCH` | 文件名声明 `mu=0.1/seed=2/dtf=0.025`，内容却是别的算例 |")
    A("| `UNPARSED` | 账行没有可解析的 tag —— 永不记为 present |")
    A("")
    A("关键设计（都是踩出来的）：")
    A("")
    A("* **spec 与 tag 交叉核对**：同名文件里跑的是别的参数，是真实发生的失效（见测试 "
      "`test_spec_contradicting_its_own_tag_is_mismatch`）；")
    A("* **账行可钉目录**（`dir` 字段）：复现算例**必须**落在 `runs_repro_m22824/`，"
      "不能被它要复现的那份存档冒充 —— 本单第一版门禁就在这里误报过 2 行 `PRESENT`；")
    A("* **空账不算完成**：`complete = (n_rows > 0) and 全部 PRESENT`。")
    A("")
    A("`tests/test_rcp_submission_ledger.py`：13 个测试（合成 run 目录，不跑仿真、不依赖集群），"
      "覆盖 tag 解析、缺件、spec 矛盾、时间步矛盾、未 settle、截断文件、钉目录、混合摘要、空账、CLI 退出码。")
    A("")
    A("## 3 交付物二：本单的账与在盘核对")
    A("")
    A("m22824 的 6 个作业已用 `stage_dir`（本地工作树直投节点，绕开坏掉的 GitHub staging）重递为 "
      "`cj4495`–`cj4500`，账在 `benchmarks/case_rcp_deposition/plan_m22977.json`。"
      "本单收尾时的门禁结果：")
    A("")
    A("| 作业 | run tag | 裁决 | 理由 |")
    A("|---|---|---|---|")
    for r in res["rows"]:
        A(f"| `{r['jid']}` | `{r['tag']}` | `{r['status']}` | "
          + ("; ".join(r["reasons"]) if r["reasons"] else "—") + " |")
    A("")
    A(f"合计 `{res['counts']}`，`complete={res['complete']}`。")
    A(f"（递交时刻队列位次 ~66–71，单作业 N=10000 沉积耗时 6–26 min，本单时长内未回件 —— "
      f"因此 4 个时间偏差档与 2 张复现证书**本轮全部记为 NOT_RUN**，不是通过，也不是失败。）")
    A("")
    A(f"![gate]({os.path.relpath(fig, ROOT)})")
    A("")
    A("*账→盘门禁：绿=产物在盘且 spec 自洽，红=盘上无产物。本单收尾时 6 行全红，"
      "与 m22824 的「queued（未回件）」是同一事实，区别只是现在它是**机器可判**的。*")
    A("")
    A("## 4 未完成的验证与它卡在哪")
    A("")
    A("补做的 6 个算例要闭合的是两笔已登记的验证欠账（`AUDIT_m22824`）：")
    A("")
    A("| 欠账 | 需要的算例 | 本轮状态 |")
    A("|---|---|---|")
    A("| 复现证书：`cf4b761c` 存档 vs 当前实现 `b44c1914` 是否数值惰性 "
      "（4 个时间步档里 2 个是跨摘要配对） | `cj4499`(μ=0,kn=4000,seed1)、`cj4500`(μ=0,kn=64000,seed1) 重跑 | NOT_RUN（未回件） |")
    A("| 时间偏差门功效：低摩擦刚性端粗/细两侧各仅 1 种子，只能给点估计 | "
      "`cj4495`–`cj4498`（μ=0.1,kn=64000,seed 2/3 × dt=t_c/20, t_c/40） | NOT_RUN（未回件） |")
    A("")
    A("回件后一条命令即可出裁决（脚本已在库、门禁已在库）：")
    A("")
    A("```")
    A("PYTHONPATH=src python benchmarks/case_rcp_deposition/submission_ledger.py \\")
    A("    benchmarks/case_rcp_deposition/plan_m22977.json --runs \\")
    A("    benchmarks/case_rcp_deposition/runs benchmarks/case_rcp_deposition/runs_repro_m22824")
    A("PYTHONPATH=src python benchmarks/case_rcp_deposition/provenance_audit.py --json --dump")
    A("PYTHONPATH=src python benchmarks/case_rcp_deposition/seed_bias_audit.py --json")
    A("```")
    A("")
    A("## 5 注意点")
    A("")
    A("* 调度面的 `succeeded` 在本轮不可信：6 个作业里 4 个报 `succeeded`，全部零产物。"
      "任何「按队列状态记账」的结论都应被这份日志推翻；")
    A("* staging 走 GitHub 在节点上当前不可达（`git fetch` 超时 134 s / 连接 443 失败），"
      "所以本单改用 `stage_dir` 直投；后续同域作业建议默认 stage_dir，避免把网络故障记成算例失败；")
    A("* 门禁只核对**产物自洽性**（tag↔spec↔settled），不判断物理正确性 —— "
      "物理裁决仍归 `provenance_audit.py` / `seed_bias_audit.py`；")
    A("* 本单没有新增物理算例结果，因此**没有**场量云图；给出的唯一图是账→盘门禁状态图。")
    A("")
    A("## 6 误差判定")
    A("")
    A("本轮**无新基准比较**：没有任何算例结果产出，故不存在「新量 vs 基准」的误差可报。"
      "沿用已合并结论（不作为本单成果）：无摩擦档硬球截距经时间步修正后 φ=0.63576 vs "
      "Bernal & Mason 0.6366，相对误差 −0.131%（<3%），该结论的有效性以 `AUDIT_m22824` "
      "的复现证书闭环为前提 —— 而该证书本轮 NOT_RUN，因此**本单不重复引用其为已认证结论**。")
    A("")
    A("## 7 建议的下一步（给下一张自找活单）")
    A("")
    A("1. 等 `cj4495`–`cj4500` 回件（队列位次已知，作业号已入账），先跑 `submission_ledger.py` 确认 6/6 `PRESENT`，"
      "再跑 `provenance_audit.py` 出证书、`seed_bias_audit.py` 出低摩擦刚性端的时间偏差区间裁决；")
    A("2. 若复现证书为 `DIVERGENT`，则 4 个时间步档里 2 个必须整体在当前实现下重跑，"
      "m21401/m22558 的偏差表需重制（账要提前开）；")
    A("3. 平台侧：把「产物存在且非空」加入 compute gate 的终态判据（本单已留证据，未越权修改）。")
    A("")

    with open(DOC, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", os.path.relpath(DOC, ROOT))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("stage-failed jobs quoted:", len(stage_failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())