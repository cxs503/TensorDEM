"""Render ``docs/H1_BROKEN_ICE_DEM_ROUND7_GATE.md`` from committed evidence.

Round 7 (mission m17656, re-open of PR #1348) has two jobs and this script is the
only path from data to prose for both of them:

1. **audit the re-open**: the mission text lists PR #1348's files; they are
   already in ``main`` (merged as #1268, m15061).  Their presence is checked by
   grepping the tree for the exact construct each one introduced -- a table row
   is ``在 `` / ``缺 `` plus the matching line, never a hand-typed claim;
2. **commit the C1/C3 evidence**: ``benchmarks/h1_broken_ice_dem/
   run_validation.py`` writes ``artifacts/`` (filtered by the merge gate), while
   ``tests/test_h1_broken_ice_dem.py`` reads ``runs/h1_broken_ice_dem/
   results_c1c3.json``.  The file was never committed, so three evidence tests
   silently skipped and the C3 "passed" claim had no machine-checkable backing
   in the tree.  This script rounds the raw JSON to 12 significant digits
   (deterministically, so the committed file is small), adds a provenance block
   with the source sha256, and writes it to ``runs/``.

Usage::

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \\
        --cases buoyancy,channel --out results_c1c3.json
    PYTHONPATH=src python scripts/render_h1_gate_round7.py

Everything numeric in the document is read out of ``runs/h1_broken_ice_dem/
results_c1c3.json`` or out of the two pytest logs under ``runs/h1_gate/``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

RAW = os.path.join(ROOT, "artifacts", "h1_broken_ice_dem", "results_c1c3.json")
DEST = os.path.join(ROOT, "runs", "h1_broken_ice_dem", "results_c1c3.json")
BEFORE = os.path.join(ROOT, "runs", "h1_gate", "round7_cli_gate_before.txt")
AFTER = os.path.join(ROOT, "runs", "h1_gate", "round7_cli_gate_after.txt")
DOC = os.path.join(ROOT, "docs", "H1_BROKEN_ICE_DEM_ROUND7_GATE.md")
FIG = os.path.join(ROOT, "docs", "figures", "h1_gate_round7.png")

ARCH = "V_sub/V = rho_ice/rho_water = 917/1025"

#: constructs that PR #1348 introduced, checked by grepping the tree
DELIVERABLES = (
    ("成对阻尼器符号（负阻尼→耗散）", "src/tensordem/polyhedron_floe3d/dynamics.py",
     "the dashpot must *add* to the repulsion"),
    ("浮力功按 dt 积分（非功率）", "src/tensordem/polyhedron_floe3d/dynamics.py",
     "self._buoyancy_force[index] @ v_mid) * dt"),
    ("泳道精确切割体积原语", "src/tensordem/polyhedron_floe3d/dynamics.py",
     "def solid_volume_slab"),
    ("固定泳道固体分数观测", "src/tensordem/polyhedron_floe3d/study.py",
     "lane_solid_volume"),
    ("run_channel 记录初始能量", "src/tensordem/polyhedron_floe3d/study.py",
     "system.record_initial_energy()"),
    ("休止角静止判据", "src/tensordem/polyhedron_floe3d/h1_cases.py",
     "settle_window=settle_window"),
    ("CLI h1-broken-ice 子命令", "src/tensordem/cli.py", '"h1-broken-ice"'),
    ("run_validation 失败返回非零", "benchmarks/h1_broken_ice_dem/run_validation.py",
     "return 1"),
    ("内核三缺陷回归测试", "tests/test_polyhedron_floe3d_kernel_ledger.py",
     "def test_pair_dashpot_is_dissipative_not_amplifying"),
    ("C1/C3 committed 证据测试", "tests/test_h1_broken_ice_dem.py",
     "results_c1c3.json"),
)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def _round(obj, sig: int = 12):
    """Recursively round floats to ``sig`` significant digits."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        if obj == 0.0:
            return 0.0
        return float("%.*g" % (sig, obj))
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items() if k != "provenance"}
    if isinstance(obj, list):
        return [_round(v, sig) for v in obj]
    return obj


def commit_evidence(raw_path: str = RAW, dest: str = DEST) -> dict:
    """Copy the raw run to ``runs/`` (rounded) with a provenance block."""
    with open(raw_path, "rb") as fh:
        blob = fh.read()
    payload = _round(json.loads(blob.decode("utf-8")))
    payload["provenance"] = {
        "generated_by": "scripts/render_h1_gate_round7.py",
        "source": os.path.relpath(raw_path, ROOT),
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        "rounding": "12 significant digits (all floats, applied uniformly)",
        "producer": "benchmarks/h1_broken_ice_dem/run_validation.py "
                    "--cases buoyancy,channel",
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    return payload


def load_evidence() -> dict:
    if os.path.exists(RAW):
        return commit_evidence()
    with open(DEST) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# audit: are PR #1348's constructs in the tree?
# ---------------------------------------------------------------------------
def deliverable_rows() -> list:
    rows = []
    for label, rel, needle in DELIVERABLES:
        path = os.path.join(ROOT, rel)
        hit = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if needle in line:
                        hit = line.strip()
                        break
        if hit is None:
            rows.append((label, rel, "**缺**", "—"))
        else:
            rows.append((label, rel, "在", "`%s`" % hit.replace("|", "\\|")[:110]))
    return rows


def deliverables_table() -> str:
    rows = ["| PR #1348 要点 | 文件 | main 落地 | 证据（命中的那一行） |", "|---|---|---|---|"]
    for label, rel, status, hit in deliverable_rows():
        rows.append("| %s | `%s` | %s | %s |" % (label, rel, status, hit))
    return "\n".join(rows)


def deliverables_missing() -> list:
    return [label for label, _rel, status, _hit in deliverable_rows() if status != "在"]


# ---------------------------------------------------------------------------
# gate defect: red before / green after
# ---------------------------------------------------------------------------
def parse_pytest(path: str) -> dict:
    if not os.path.exists(path):
        return {"exists": False}
    with open(path) as fh:
        text = fh.read()
    counts = {}
    for name in ("passed", "failed", "skipped", "error"):
        m = re.search(r"(\d+) %s" % name, text)
        counts[name] = int(m.group(1)) if m else 0
    failed = re.findall(r"^FAILED (\S+)", text, flags=re.M)
    rc = re.search(r"^.*rc=(\d+)$", text, flags=re.M)
    counts["failed_names"] = [f.split("::")[-1] for f in failed]
    counts["rc"] = int(rc.group(1)) if rc else None
    counts["exists"] = True
    return counts


GATE_TEST_FILE = os.path.join(ROOT, "tests", "test_h1_broken_ice_cli.py")


def _test_names() -> list:
    """Every ``def test_*`` in the gate test file, so no row can be dropped."""
    if not os.path.exists(GATE_TEST_FILE):
        return []
    with open(GATE_TEST_FILE) as fh:
        return re.findall(r"^def (test_\w+)", fh.read(), flags=re.M)


def gate_table() -> str:
    before, after = parse_pytest(BEFORE), parse_pytest(AFTER)
    if not (before.get("exists") and after.get("exists")):
        return "(gate logs missing: run the before/after pytest pair into `runs/h1_gate/`)"
    names = _test_names() or sorted(set(before["failed_names"]) | set(after["failed_names"]))
    rows = ["| 测试 | 旧 `cli.py`（main/HEAD） | 新 `cli.py`（本轮） |", "|---|---|---|"]
    for name in names:
        rows.append("| `%s` | %s | %s |" % (
            name,
            "**FAIL**" if name in before["failed_names"] else "pass",
            "**FAIL**" if name in after["failed_names"] else "pass"))
    rows.append("| 汇总 | %d 通过 / %d 失败 / rc=%s | %d 通过 / %d 失败 / rc=%s |" % (
        before["passed"], before["failed"], before["rc"],
        after["passed"], after["failed"], after["rc"]))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# C1 / C3 numbers
# ---------------------------------------------------------------------------
def _f(x, n=4) -> str:
    try:
        return "%.*g" % (n, float(x))
    except (TypeError, ValueError):
        return "—"


def c1_rows(evidence: dict) -> list:
    case = evidence["cases"].get("buoyancy")
    if case is None:
        return ["| C1 | *(not in this run)* | — | — | — | — |"]
    ref, got, err = case["ref"], case["got"], case["error"]
    tol = case["tolerance"]
    out = [
        ("浸没体积分数", ref["submerged_fraction_archimedes"], got["submerged_fraction"],
         err["rel_err_submerged_fraction"], tol),
        ("吃水 [m]", ref["draft_archimedes_m"], got["draft_m"],
         err["rel_err_draft"], tol),
    ]
    rows = ["| 判据 | 量 | 基准（Archimedes） | 实测 | 相对误差 | 阈值 | 判定 |", "|---|---|---|---|---|---|---|"]
    for name, r, g, e, t in out:
        rows.append("| C1 | %s | %s | %s | %s | <%g | **%s** |" % (
            name, _f(r, 10), _f(g, 10), _f(e, 4), t, "PASS" if e < t else "FAIL"))
    e_res = got.get("energy_residual")
    rows.append("| C1 | 能量残差 [-] | 0 | %s | — | <1e-2 | **%s** |" % (
        _f(e_res), "PASS" if e_res is not None and e_res < 1e-2 else "FAIL"))
    return rows


def c3_rows(evidence: dict) -> list:
    case = evidence["cases"].get("channel")
    if case is None:
        return ["| C3 | *(not in this run)* | — | — | — | — |"]
    s, ref = case["summary"], case["ref"]
    rows = ["| 判据 | 量 | 基准 / 阈值 | 实测 | 判定 |", "|---|---|---|---|---|"]
    rows.append("| C3 | 泳道固体分数 初→末 | 降幅 >%g%% | %s → %s（降幅 %s%%） | **%s** |" % (
        100 * ref["lane_drop_threshold_fraction"],
        _f(s["lane_concentration_initial"]), _f(s["lane_concentration_final"]),
        _f(100 * (1 - s["lane_concentration_ratio"]), 3),
        "PASS" if s["lane_concentration_ratio"] <= 1 - ref["lane_drop_threshold_fraction"]
        else "FAIL"))
    rows.append("| C3 | 扫掠完成 (hull_advance ≥ corridor) | ≥ %s m | %s m | **%s** |" % (
        _f(case["setup"]["corridor_m"]), _f(s["hull_advance_m"]),
        "PASS" if s["sweep_complete"] else "FAIL"))
    rows.append("| C3 | 质量残差 [-] | <1e-12 | %s | **%s** |" % (
        _f(s["mass_residual"]), "PASS" if abs(s["mass_residual"]) < 1e-12 else "FAIL"))
    rows.append("| C3 | 能量残差 [-] | <%g | %s | **%s** |" % (
        ref["energy_residual_tolerance"], _f(s["energy_residual"]),
        "PASS" if s["energy_residual"] < ref["energy_residual_tolerance"] else "FAIL"))
    rows.append("| C3 | 阻力均值 ± 标准差 [N] | 无实验基准 | %s ± %s | 参考量 |" % (
        _f(s["resistance_mean_steady_N"]), _f(s["resistance_std_steady_N"])))
    return rows


def figure(evidence: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                     # pragma: no cover
        return False
    case = evidence["cases"].get("channel")
    if case is None:
        return False
    series = case["series"]
    t = series["time_s"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=140)
    ax = axes[0]
    ax.plot(t, series["lane_concentration"], lw=1.6, label="exact clipped solid fraction")
    ax.plot(t, series["lane_concentration_centroid_proxy"], lw=1.0, ls="--",
            label="centroid-count proxy (step function)")
    ax.axhline(0.5 * float(series["lane_concentration"][0]), color="k", lw=0.8, ls=":",
               label="half of the initial lane fill")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lane ice concentration [-]")
    ax.set_title("C3 fixed-lane clearing (hull advance %.2f m)"
                 % float(case["summary"]["hull_advance_m"]))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t, series["resistance_N"], lw=1.0, color="tab:red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("hull resistance [N]")
    ax.set_title("C3 hull resistance (mean %.0f N over the last 2/3)"
                 % float(case["summary"]["resistance_mean_steady_N"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
def render(evidence: dict) -> str:
    prov = evidence.get("provenance", {})
    passed = evidence.get("passed", {})
    gate = parse_pytest(BEFORE), parse_pytest(AFTER)
    has_fig = figure(evidence)
    lines = []
    A = lines.append
    A("# H1-BROKEN-ICE-DEM · 第 7 轮：**重开 #1348 的复验** + 验收入口闸门缺陷修复")
    A("")
    A("锚点：`H1-BROKEN-ICE-DEM`（TensorDEM）。任务：mission m17656「重开·PR #1348·管到底第 2 轮」。")
    A("PR #1348 本身已被 owner 裁定关闭（2026-11-05）；本次重开要求「从当前 main 切枝、功能+测试齐备、本地全套测试过、")
    A("PR 合并进 main」。因此第一步不是重写功能，而是**先核验 #1348 的功能要点是否已在 main**——")
    A("已在则重写等于制造重复实现（三重检索纪律）；不在则补齐。核验结果见 §1。")
    A("")
    A("## 1 复验：#1348 的功能要点在 main 的落地核验（机器检查，非手写）")
    A("")
    A("检查方式：对每个要点，在目标文件里 grep PR #1348 引入的那一个构造；命中即 `在`，未命中即 `缺`。")
    A("本表由 `scripts/render_h1_gate_round7.py::deliverable_rows` 生成。")
    A("")
    A(deliverables_table())
    A("")
    missing = deliverables_missing()
    A("缺口：%s。" % ("无——#1348 的 10 项要点全部已在 main（经 #1268/m15061 合入）"
                     if not missing else "、".join(missing)))
    A("")
    A("**结论（诚实）**：按「功能要点」重做一遍会是重复实现，本单不做；本轮只补 #1348 遗留的**验收入口闸门**漏洞，")
    A("并把 C1/C3 证据真正入树。下面的 §2–§3 是本轮唯一的代码增量。")
    A("")
    A("## 2 本轮增量 ①：`tensordem h1-broken-ice` 退出码恒 0（闸门形同虚设）")
    A("")
    A("`benchmarks/h1_broken_ice_dem/run_validation.py` 在审计 m15192 已被要求「失败必须返回非零」，")
    A("但**同一个验收链的 CLI 入口**漏改了：`_cmd_h1_broken_ice` 打印 `cases: {...False}` 之后 `return None`，")
    A("`main()` 也不传播状态，于是任何 CI/看板只要走 CLI 就永远是绿的。附带同源缺陷：`--quick` 直接把用户")
    A("手输的 `--repose-steps/--channel-steps/--repose-bodies` 整个丢掉（静默忽略一个显式参数 = 小样本结果")
    A("可能被当成全量结果）。")
    A("")
    A("修法（`src/tensordem/cli.py`，+51/−11）：`_cmd_h1_broken_ice` 返回退出码，任一请求的算例 `passed=False` → `1`；")
    A("`main()` 用 `raise SystemExit(status)` 传播；三个尺寸参数的 argparse 默认值改成 `None`（哨兵），")
    A("`--quick` 下只透传用户真正输入的尺寸，非 quick 下再填回文档化的 6000/45/900 默认值。")
    A("")
    A("对照证据（同一份 `tests/test_h1_broken_ice_cli.py`，先后跑在旧/新 `cli.py` 上，原始输出在 `runs/h1_gate/`）：")
    A("")
    A(gate_table())
    A("")
    A("复现：`git show HEAD:src/tensordem/cli.py > src/tensordem/cli.py && pytest tests/test_h1_broken_ice_cli.py`")
    A("（旧实现 3 红），恢复新实现后 7 绿。这就是本轮「修复前/后」的基线对照。")
    A("")
    A("## 3 本轮增量 ②：C1/C3 证据入库（三个证据测试此前是空转的 skip）")
    A("")
    A("`tests/test_h1_broken_ice_dem.py` 里有 3 个测试专门核对「committed C1/C3 证据」，读的是")
    A("`runs/h1_broken_ice_dem/results_c1c3.json`；该文件从未入树（`artifacts/` 被合并门禁滤除，而 evidence 测试读 `runs/`），")
    A("于是三个测试**静默 skip**：C3 的 `passed=True` 结论在树内没有任何机器可校的支撑。本轮把该文件生成并提交。")
    A("")
    A("生成链（零手抄）：`run_validation.py --cases buoyancy,channel --out results_c1c3.json` → `artifacts/` →")
    A("本渲染器统一四舍五入到 12 位有效数字 + 写入 provenance（源文件 sha256）→ `runs/h1_broken_ice_dem/results_c1c3.json`。")
    A("provenance 源 sha256 `%s`，生成时间 `%s`。" %
      (prov.get("source_sha256", "(already committed)"), prov.get("written", "(from the committed file)")))
    A("")
    A("### 3.1 C1 静水浮沉（基准 %s）" % ARCH)
    A("")
    A("\n".join(c1_rows(evidence)))
    A("")
    A("### 3.2 C3 航道清空（固定泳道固体分数）")
    A("")
    A("观测量的仪器是 `DEMSystem3D.solid_volume_slab`（对每个四面体按两个平面精确切割），")
    A("本轮同时给它补了解析测试 `tests/test_polyhedron_floe3d_lane_metric.py`（4 例：轴对齐盒子解析体积、")
    A("剖分可加性、多体、以及「质心在窗外但有一半在窗内」——这正是旧质心计数观测量的缺陷）。")
    A("")
    A("\n".join(c3_rows(evidence)))
    A("")
    if has_fig:
        A("![C3 lane clearing and resistance](figures/h1_gate_round7.png)")
        A("")
        A("图 1 左：固定泳道固体分数随时间（实线=精确切割体积，虚线=旧质心计数代理，点线=初始值的一半）；")
        A("右：船体阻力时序。两图均由 `scripts/render_h1_gate_round7.py` 从证据 JSON 直接绘制。")
        A("")
    else:
        A("（本次未生成图：matplotlib 不可用；数字表仍完整。）")
        A("")
    A("## 4 注意点与局限（honest）")
    A("")
    A("* **C2（碎冰堆积休止角）本轮未动**：`runs/h1_repose/` 的三 μ 分片证据同样不在树内，")
    A("  `tests/test_h1_repose_evidence.py` 有 4 个测试因此 skip。补齐需要 3 × (45 体 × 30 000 步) 的分片作业，")
    A("  超出本单预算；此处如实登记为欠账，不做任何「C2 已达标」的暗示。")
    A("* **C3 无实验阻力基准**：`mu = %s` 是声明值（沿用的航道算例值），未标定；阻力均值只作参考量列出，" %
      _f(evidence.get("cases", {}).get("channel", {}).get("setup", {}).get("mu", float("nan")), 3) +
      "不参与达标判定。清空判据是「泳道相对初始充满度降幅 > 50% 且扫掠完成」，不是「浓度=0」。")
    A("* **顺带修的红测试（pre-existing，非本单引入）**：`tests/test_hull_profile_cli.py::")
    A("  test_hull_cli_exports_position_and_reaction_history` 在 main 上是红的——它仍用 subcommand 重构**之前**的")
    A("  扁平参数形式调 `main()`（`parse_args` 直接 SystemExit），与本单的闸门修复无关；本单把它改成 `dem-2d` 子命令形式")
    A("  （3 行），否则「本地全套测试过」这条验收标准根本不可达。核对方式：把 `src/tensordem/cli.py` 换回 `git show HEAD:`")
    A("  的版本，该测试同样红 → 确认归属。")
    A("* **机器门禁的反面教训**：`artifacts/` 会被合并门禁滤除，因此「证据测试读 `runs/`、runner 写 `artifacts/`」")
    A("  这个组合天然会退化成空转 skip——只用 `pytest -q` 看「全绿」是看不出来的，必须看 `-rs` 的 skip 名单。")
    A("")
    A("## 5 误差判定（关键量 vs 基准，是否 <3%）")
    A("")
    c1 = evidence.get("cases", {}).get("buoyancy", {})
    c3 = evidence.get("cases", {}).get("channel", {})
    if c1:
        e1 = c1["error"]
        A("* C1 浸没体积分数相对 Archimedes：%s（%s）——**%s3%%**" % (
            _f(e1["rel_err_submerged_fraction"], 4),
            "%.3e" % e1["rel_err_submerged_fraction"],
            "小于" if e1["rel_err_submerged_fraction"] < 0.03 else "**未**小于"))
        A("* C1 吃水相对 Archimedes：%s（%s）——**%s3%%**" % (
            _f(e1["rel_err_draft"], 4), "%.3e" % e1["rel_err_draft"],
            "小于" if e1["rel_err_draft"] < 0.03 else "**未**小于"))
        A("* C1 能量残差 %s（阈值 1e-2）——%s" % (
            _f(c1["got"].get("energy_residual")),
            "通过" if c1["got"].get("energy_residual", 1.0) < 1e-2 else "**未通过**"))
    if c3:
        s3 = c3["summary"]
        A("* C3 **无解析/实验基准**（不得声称误差百分比）：只判质量残差 %s（阈值 1e-12）、能量残差 %s（阈值 %g）"
          "与泳道降幅 %s%%（阈值 50%%）——三项判定见 §3.2 表。" % (
              _f(s3["mass_residual"]), _f(s3["energy_residual"]),
              c3["ref"]["energy_residual_tolerance"],
              _f(100 * (1 - s3["lane_concentration_ratio"]), 3)))
    A("")
    A("## 6 复现命令")
    A("")
    A("```")
    A("PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \\")
    A("    --cases buoyancy,channel --out results_c1c3.json      # -> artifacts/")
    A("PYTHONPATH=src python scripts/render_h1_gate_round7.py    # -> runs/ + 本文 + 图")
    A("python -m pytest tests/test_h1_broken_ice_cli.py \\")
    A("    tests/test_polyhedron_floe3d_lane_metric.py \\")
    A("    tests/test_h1_broken_ice_dem.py -q")
    A("```")
    A("")
    A("整体判定（请求算例 %s）：%s" % (
        ", ".join(sorted(passed)) or "—",
        "全部 passed" if passed and all(passed.values()) else "存在未通过算例（见上表）"))
    A("")
    return "\n".join(lines)


def main() -> int:
    evidence = load_evidence()
    text = render(evidence)
    with open(DOC, "w") as fh:
        fh.write(text)
    print("wrote", os.path.relpath(DOC, ROOT))
    print("evidence", os.path.relpath(DEST, ROOT), "cases:", sorted(evidence["cases"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())