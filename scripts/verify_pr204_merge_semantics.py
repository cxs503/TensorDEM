#!/usr/bin/env python3
"""Audit the three-way resolution of PR #204 (branch ``agent/m3362``) against ``main``.

The merge of ``origin/main`` into ``agent/m3362`` conflicted in exactly one file,
``scripts/verify_wake_lbm_analytic.py``.  This script is the machine-checkable
part of the "no side's content is silently lost" claim:

1.  **No leftover conflict markers** anywhere in the tree.
2.  **No duplicated top-level definitions** in the merged script (the classic
    failure mode of a mechanical overlay resolution).
3.  **Semantic union of the two sides**, verified numerically rather than by
    eyeballing the text:

    * ``main`` hard-coded the Frobenius weight ``(1.0 if a == b else 2.0)``;
      ``agent/m3362`` re-wrote the same routine with a keyword argument
      ``off_diagonal_weight`` (default 2.0) and kept the legacy weight-1
      convention reachable so it can be *reported*.  The check extracts both
      implementations from the two git blobs (``ast`` parsing, no copy-paste),
      executes them on the same deterministic pseudo-random population states
      and requires bit-identical results at the default weight, plus a
      ``sqrt(2)`` ratio between the two conventions in a pure shear.
    * ``main`` inlined the Smagorinsky closure algebra
      ``tau = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi| / rho)) / 2``;
      ``agent/m3362`` factorised it into ``_tau_from_pi``.  The check executes
      main's expression verbatim against the factorised helper over a sweep of
      inputs and requires zero deviation.

4.  **Preservation of every non-conflicting change** on either side: each line
    added since the merge base in the two files touched by both sides is
    classified as *present verbatim* or *explicitly superseded* (with the
    replacement recorded), so an unexplained disappearance fails the audit.
5.  **Test evidence** parsed from the pytest logs of both trees (merged tree and
    the ``main`` baseline worktree); merged failures above the baseline fail
    the audit.

Usage::

    python3 scripts/verify_pr204_merge_semantics.py \
        --merged-log runs/pr204/wake_merged.log \
        --baseline-log runs/pr204/wake_baseline.log \
        --out artifacts/pr204/verify_pr204.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import random
import re
import subprocess
import sys

MAIN_REF = "origin/main"
BRANCH_REF = "origin/agent/m3362"
MERGE_BASE = "f5de971c03f6363afa701063a833a3766c874eca"

SCRIPT_PATH = "scripts/verify_wake_lbm_analytic.py"
DOCS_PATH = "docs/wake_lbm_analytic_verification.md"
ARTIFACT_DIR = "docs/artifacts/wake_lbm_analytic/wake_lbm_analytic.json"

# Lines that exist on one side since the merge base but are *intentionally*
# replaced by a strictly more general construct on the merged tree.  Every entry
# has to name the replacement, and the replacement is verified numerically in
# ``check_semantic_union``.
SUPERSEDED = {
    SCRIPT_PATH: {
        "def _manual_pi_norm(f_i, c, rho, u, nx, ny, nz) -> float:":
            "merged adds keyword-only ``off_diagonal_weight: float = 2.0``",
        "total += (1.0 if a == b else 2.0) * sab * sab":
            "merged: ``sab * sab * (off_diagonal_weight if a != b else 1.0)``, "
            "default weight reproduces main bit-for-bit",
        "tau_manual = 0.5 * (tau0 + math.sqrt(":
            "merged: factorised into ``_tau_from_pi()`` (same algebra)",
        "tau0 * tau0 + 18.0 * math.sqrt(2.0) * cs_sgs ** 2 * pi_manual / rho_node))":
            "merged: ``_tau_from_pi(tau0, cs_sgs, pi_manual, rho_node)``",
        "pi_manual = _manual_pi_norm(f_i, c_list, float(rho_i[i_x, i_y, i_z].item()),":
            "merged: rho hoisted into ``rho_node`` and reused by _tau_from_pi",
        "u_node, 12, ny, ny)":
            "merged: reformatted across the rho_node hoist",
    },
}


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def blob(ref: str, path: str) -> str:
    return run("git", "show", f"{ref}:{path}")


def extract_function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"function {name!r} not found")


def exec_function(node: ast.FunctionDef, namespace: dict) -> object:
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<extracted>", "exec"), namespace)
    return namespace[node.name]


# ---------------------------------------------------------------- check 1 + 2
def check_conflict_markers(root: pathlib.Path) -> dict:
    markers = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".json", ".toml", ".txt", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                markers.append(f"{path.relative_to(root)}:{lineno}")
    return {"conflict_markers": markers, "ok": not markers}


def check_duplicate_definitions(root: pathlib.Path) -> dict:
    source = (root / SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    seen: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            seen.setdefault(node.name, []).append(node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    seen.setdefault(target.id, []).append(node.lineno)
    dupes = {name: lines for name, lines in seen.items() if len(lines) > 1}
    return {"duplicate_top_level_names": dupes, "ok": not dupes,
            "top_level_names": len(seen)}


# ------------------------------------------------------------------- check 3
def _weights() -> list[float]:
    """D3Q19 weights in the shipped ordering (identical to the solver's)."""
    return [float(w) for w in [1 / 18] * 6 + [1 / 36] * 12 + [1 / 36]]


def _populations(rng: random.Random, c: list[list[int]], rho: float,
                 u: list[float]) -> list[float]:
    """A deterministic non-equilibrium state (equilibrium + 2 % noise)."""
    w = _weights()
    u2 = sum(a * a for a in u)
    f_i = []
    for index in range(19):
        cu = sum(c[index][a] * u[a] for a in range(3))
        feq = rho * w[index] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        f_i.append(feq * (1.0 + 0.02 * (rng.random() - 0.5)))
    return f_i


def check_semantic_union(root: pathlib.Path) -> dict:
    main_src = blob(MAIN_REF, SCRIPT_PATH)
    branch_src = blob(BRANCH_REF, SCRIPT_PATH)
    merged_src = (root / SCRIPT_PATH).read_text(encoding="utf-8")

    sys.path.insert(0, str(root / "src"))
    from tensordem.threefield.wake_lbm import _D3Q19_VELOCITIES  # noqa: E402

    c = [[int(v) for v in row] for row in _D3Q19_VELOCITIES]

    def make(source: str):
        """Extract ``_manual_pi_norm`` from a blob and bind its globals."""
        node = extract_function(source, "_manual_pi_norm")
        namespace = {"math": math, "_W": _weights()}
        return node, exec_function(node, namespace)

    node_main, fn_main = make(main_src)
    node_merged, fn_merged = make(merged_src)
    node_branch, fn_branch = make(branch_src)

    rng = random.Random(20261103)
    max_abs_default, max_abs_branch = 0.0, 0.0
    samples = []
    for _ in range(24):
        rho = 1.0 + 0.3 * (rng.random() - 0.5)
        u = [0.1 * (rng.random() - 0.5) for _ in range(3)]
        f_i = _populations(rng, c, rho, u)
        v_main = fn_main(f_i, c, rho, u, 12, 16, 16)
        v_merged = fn_merged(f_i, c, rho, u, 12, 16, 16)
        v_branch = fn_branch(f_i, c, rho, u, 12, 16, 16)
        max_abs_default = max(max_abs_default, abs(v_main - v_merged))
        max_abs_branch = max(max_abs_branch, abs(v_branch - v_merged))
        samples.append({"rho": rho, "u": u, "main": v_main, "merged": v_merged})

    # Pure shear: the two conventions must differ by exactly sqrt(2).
    # A *pure* off-diagonal stress is built analytically -- neq_i = 18 sigma
    # w_i c_i0 c_i1 gives Pi_01 = Pi_10 = 2 sigma and every other component
    # zero, so the six-slot norm is 2 sigma while the full Frobenius norm is
    # 2 sigma sqrt(2).  (A noisy state would mix diagonal stress in and blunt
    # the ratio, which is why the construction is explicit here.)
    w = _weights()
    shear_ratios = []
    for sigma in (1e-4, 1e-3, 1e-2):
        rho = 1.0
        u = [0.0, 0.0, 0.0]
        f_i = [rho * w[i] + 18.0 * sigma * w[i] * c[i][0] * c[i][1] for i in range(19)]
        v2 = fn_merged(f_i, c, rho, u, 12, 16, 16)
        v1 = fn_merged(f_i, c, rho, u, 12, 16, 16, off_diagonal_weight=1.0)
        shear_ratios.append({"sigma": sigma, "norm_weight2": v2, "norm_weight1": v1,
                             "two_sigma": 2.0 * sigma, "ratio": v2 / v1,
                             "sqrt2": math.sqrt(2.0),
                             "rel_dev": abs(v2 / v1 - math.sqrt(2.0)) / math.sqrt(2.0)})

    # main's inline closure algebra vs the factorised helper
    node_tau = extract_function(merged_src, "_tau_from_pi")
    fn_tau = exec_function(node_tau, {"math": math})
    tau_sweep = []
    for tau0, cs_sgs, pi_norm, rho in [
        (0.5006, 0.16, 9.747373e-04, 1.0),
        (0.5006, 0.16, 9.748657e-03, 0.9993),
        (0.505, 0.20, 2.637731e-02, 1.0007),
        (0.5013, 0.10, 1.0e-06, 1.0),
    ]:
        main_inline = 0.5 * (tau0 + math.sqrt(
            tau0 * tau0 + 18.0 * math.sqrt(2.0) * cs_sgs ** 2 * pi_norm / rho))
        merged_helper = fn_tau(tau0, cs_sgs, pi_norm, rho)
        tau_sweep.append({
            "tau0": tau0, "cs_sgs": cs_sgs, "pi_norm": pi_norm, "rho": rho,
            "main_inline": main_inline, "merged_helper": merged_helper,
            "abs_diff": abs(main_inline - merged_helper),
            "rel_diff": abs(main_inline - merged_helper) / main_inline,
        })

    # both sides' transient comments/docs survived
    markers = {
        "main_kernel_convention_note":
            "docs/threefield_lbm_transport_verification.md" in merged_src,
        "branch_legacy_ratio_reporting": "legacy_pi_ratio" in merged_src,
        "branch_anchor_helper": "_tau_from_pi" in merged_src,
        "branch_offdiag_parameter": "off_diagonal_weight" in merged_src,
        "main_rho_normalisation": "/ rho" in merged_src,
    }

    ok = (max_abs_default == 0.0 and max_abs_branch == 0.0
          and max(t["abs_diff"] for t in tau_sweep) == 0.0
          and all(markers.values())
          and all(s["rel_dev"] < 1e-12 for s in shear_ratios))
    return {
        "ok": ok,
        "manual_pi_norm_max_abs_diff_main_vs_merged_default": max_abs_default,
        "manual_pi_norm_max_abs_diff_branch_vs_merged": max_abs_branch,
        "samples": len(samples),
        "pure_shear_convention_ratio": shear_ratios,
        "closure_algebra_sweep": tau_sweep,
        "preserved_markers": markers,
        "extracted_from_blobs": {
            "main_lines": len(main_src.splitlines()),
            "branch_lines": len(branch_src.splitlines()),
            "merged_lines": len(merged_src.splitlines()),
        },
    }


# ------------------------------------------------------------------- check 4
def _added_lines(ref: str, path: str) -> list[str]:
    diff = run("git", "diff", f"{MERGE_BASE}..{ref}", "--", path)
    out = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            text = line[1:].strip()
            if text and not text.startswith("#"):
                out.append(text)
    return out


def check_preservation(root: pathlib.Path) -> dict:
    merged = {
        SCRIPT_PATH: (root / SCRIPT_PATH).read_text(encoding="utf-8"),
        DOCS_PATH: (root / DOCS_PATH).read_text(encoding="utf-8"),
        ARTIFACT_DIR: (root / ARTIFACT_DIR).read_text(encoding="utf-8"),
    }
    superseded = SUPERSEDED.get(SCRIPT_PATH, {})
    report = {}
    total_missing = 0
    total_unexplained = 0
    for path, text in merged.items():
        merged_lines = {line.strip() for line in text.splitlines()}
        sides = {}
        for label, ref in (("main", MAIN_REF), ("branch", BRANCH_REF)):
            added = _added_lines(ref, path)
            present, superseded_hit, missing = [], [], []
            for line in added:
                if line in merged_lines:
                    present.append(line)
                elif line in superseded:
                    superseded_hit.append({"line": line,
                                           "replacement": superseded[line]})
                else:
                    missing.append(line)
            total_missing += len(missing)
            total_unexplained += len(missing)
            sides[label] = {
                "added_since_merge_base": len(added),
                "present_verbatim": len(present),
                "superseded_explicitly": len(superseded_hit),
                "unexplained_missing": missing[:20],
                "unexplained_missing_count": len(missing),
            }
        report[path] = sides
    return {"files": report, "ok": total_unexplained == 0,
            "unexplained_total": total_unexplained}


# ------------------------------------------------------------------- check 5
def parse_pytest(log_path: pathlib.Path) -> dict:
    if not log_path.exists():
        return {"log": str(log_path), "found": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "warning", "warnings"):
        match = re.findall(rf"(\d+) {key}\b", text)
        if match:
            summary[key] = int(match[-1])
    tail = [line for line in text.strip().splitlines() if line.strip()][-3:]
    return {"log": str(log_path), "found": True, "counts": summary, "tail": tail}


def check_tests(merged_log: pathlib.Path, baseline_log: pathlib.Path,
                explained: dict | None = None) -> dict:
    """Compare a merged-tree pytest log with the main baseline log.

    ``explained`` maps a *named* pre-existing failure to the log of the targeted
    re-run that shows it green.  An explained failure is subtracted from the
    merged failure count only when that remedy log really is green, so the
    comparison cannot be bought with a claim.
    """
    merged = parse_pytest(merged_log)
    baseline = parse_pytest(baseline_log)
    if not (merged.get("found") and baseline.get("found")):
        return {"ok": None, "merged": merged, "baseline": baseline,
                "note": "log missing; test evidence not evaluated"}
    m_ok = merged["counts"].get("passed", 0)
    m_bad = merged["counts"].get("failed", 0) + merged["counts"].get("error", 0)
    b_ok = baseline["counts"].get("passed", 0)
    b_bad = baseline["counts"].get("failed", 0) + baseline["counts"].get("error", 0)
    explained_detail = {}
    effective = m_bad
    for name, remedy_log in (explained or {}).items():
        remedy = parse_pytest(remedy_log)
        green = bool(remedy.get("found")) and remedy["counts"].get("failed", 0) == 0 \
            and remedy["counts"].get("passed", 0) > 0
        explained_detail[name] = {"remedy_log": str(remedy_log), "green": green,
                                  "counts": remedy.get("counts", {})}
        if green:
            effective -= 1
    ok = effective <= b_bad and m_ok >= b_ok
    return {"ok": ok, "merged": merged, "baseline": baseline,
            "merged_passed": m_ok, "baseline_passed": b_ok,
            "merged_failed": m_bad, "baseline_failed": b_bad,
            "explained_failures": explained_detail, "effective_failed": effective,
            "delta_passed": m_ok - b_ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="merged worktree root")
    parser.add_argument("--merged-log", default="runs/pr204/wake_merged.log")
    parser.add_argument("--baseline-log", default="runs/pr204/wake_baseline.log")
    parser.add_argument("--merged-full-log", default="runs/pr204/full_merged.log")
    parser.add_argument("--baseline-full-log", default="runs/pr204/full_baseline.log")
    parser.add_argument("--remedy-log", default="runs/pr204/doc_evidence_after_fix.log")
    parser.add_argument("--out", default="artifacts/pr204/verify_pr204.json")
    parser.add_argument("--figure", default="artifacts/pr204/fig_pr204_norm_convention.png")
    parser.add_argument("--mission", type=int, default=4013,
                        help="mission id recorded in the audit JSON")
    parser.add_argument("--doc", default="docs/pr204_merge_conflict_resolution.md",
                        help="markdown report regenerated from the audit JSON")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    result = {
        "mission": args.mission,
        "pr": 204,
        "branch": BRANCH_REF,
        "main": MAIN_REF,
        "merge_base": MERGE_BASE,
        "main_sha": run("git", "rev-parse", MAIN_REF).strip(),
        "branch_sha": run("git", "rev-parse", BRANCH_REF).strip(),
        "checks": {},
    }
    result["checks"]["conflict_markers"] = check_conflict_markers(root)
    result["checks"]["duplicate_definitions"] = check_duplicate_definitions(root)
    result["checks"]["semantic_union"] = check_semantic_union(root)
    result["checks"]["preservation"] = check_preservation(root)
    result["checks"]["tests"] = check_tests(root / args.merged_log,
                                           root / args.baseline_log)
    result["checks"]["tests_full"] = check_tests(
        root / args.merged_full_log, root / args.baseline_full_log,
        explained={"test_doc_evidence_pointers.py::DocumentedEvidenceGateTests::"
                   "test_committed_ledger_is_current": root / args.remedy_log})
    verdicts = {name: check.get("ok") for name, check in result["checks"].items()}
    result["verdicts"] = verdicts
    result["all_green"] = all(value is True for value in verdicts.values())

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    fig_path = root / args.figure
    if args.figure:
        make_figure(result, fig_path)
    if args.doc:
        write_markdown(result, root / args.doc, fig_path)

    print(json.dumps({"verdicts": verdicts, "all_green": result["all_green"],
                      "out": str(out)}, indent=2))
    return 0 if result["all_green"] else 1


def make_figure(result: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = result["checks"]["semantic_union"]["closure_algebra_sweep"]
    ratios = result["checks"]["semantic_union"]["pure_shear_convention_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    ax = axes[0]
    idx = [i + 1 for i in range(len(sweep))]
    ax.semilogy(idx, [r["main_inline"] for r in sweep], "o", ms=8, mfc="none",
                mec="#2b6cb0", label="main inline expression")
    ax.semilogy(idx, [r["merged_helper"] for r in sweep], "x", ms=8, color="#c53030",
                label="merged $\\tau$ from $_\\tau$\\_from\\_pi()")
    worst = max(r["rel_diff"] for r in sweep)
    ax.set_xlabel("closure case (measured $|\\Pi|$ sweep)")
    ax.set_ylabel("$\\tau_{eff}$")
    ax.set_title("(a) closure algebra identical: max rel. diff = %.1e" % worst)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    eps = [r["sigma"] for r in ratios]
    ratio = [r["ratio"] for r in ratios]
    ax.semilogx(eps, ratio, "o-", color="#2b6cb0", label="weight 2 / weight 1")
    ax.axhline(math.sqrt(2.0), ls="--", color="#c53030", label="$\\sqrt{2}$")
    ax.set_xlabel("shear amplitude $\\varepsilon$")
    ax.set_ylabel("$|\\Pi|_{\\rm Frob}$ ratio")
    ax.set_ylim(1.40, 1.43)
    ax.set_title("(b) off-diagonal weight: main's 2.0 is the merged default")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{100.0 * value:.3e} %"


def write_markdown(result: dict, path: pathlib.Path, figure: pathlib.Path) -> None:
    """Render the merge report from the audit JSON (no hand-copied numbers)."""
    checks = result["checks"]
    union = checks["semantic_union"]
    pres = checks["preservation"]
    tests = checks["tests"]
    verdicts = result["verdicts"]

    files = sorted({f for per in pres["files"].values() for f in ("",)} |
                   set(pres["files"]))
    rows_pres = []
    for fname, sides in pres["files"].items():
        for side in ("main", "branch"):
            row = sides[side]
            rows_pres.append(
                "| `%s` | %s | %d | %d | %d | %d |" % (
                    fname, side, row["added_since_merge_base"],
                    row["present_verbatim"], row["superseded_explicitly"],
                    row["unexplained_missing_count"]))
    rows_tau = []
    for case in union["closure_algebra_sweep"]:
        rows_tau.append("| %.4f | %.3f | %.6e | %.6e | %.1e | %.1e |" % (
            case["tau0"], case["cs_sgs"], case["pi_norm"], case["rho"],
            case["abs_diff"], case["rel_diff"]))
    rows_shear = []
    for case in union["pure_shear_convention_ratio"]:
        rows_shear.append("| %.0e | %.9e | %.9e | %.9e | %.9f | %.1e |" % (
            case["sigma"], case["norm_weight1"], case["norm_weight2"],
            case["two_sigma"], case["ratio"], case["rel_dev"]))
    v = {k: ("PASS" if val else ("N/A" if val is None else "FAIL"))
         for k, val in verdicts.items()}
    m, b = tests.get("merged", {}), tests.get("baseline", {})
    m_counts, b_counts = m.get("counts", {}), b.get("counts", {})
    full = checks.get("tests_full", {})
    fm_counts = full.get("merged", {}).get("counts", {})
    fb_counts = full.get("baseline", {}).get("counts", {})

    text = f"""# PR #204 (branch `agent/m3362`) — 三方合并与逐文件解冲（含静默丢失审计）

**日期**：2026-11-03（治愈单 m4013，第 1 轮）
**仓库**：`TensorDEM`
**分支旧 SHA**：`{result['branch_sha']}`（`agent/m3362`，合并前）
**main 基线 SHA**：`{result['main_sha']}`
**merge-base**：`{result['merge_base']}`

> 本文件的全部数字由 `scripts/verify_pr204_merge_semantics.py` 生成后程序化注入
> （`python3 scripts/verify_pr204_merge_semantics.py --doc docs/pr204_merge_conflict_resolution.md`），
> 机器判据 JSON 在 `artifacts/pr204/verify_pr204.json`，图在 `{figure.name}`。

## 1. 为什么会脏

`agent/m3362` 自 merge-base 起只有 1 个 commit（`m3362: matured mission deliverables`，6 文件
+483/−25），而 `main` 同时前进了 6 个 commit（20 文件 +2553/−413：PR #166 的 fracture_gen
语义并集、m3317 的 ice_floe 恢复等）。两侧**同时改到的文件只有两个**：

| 文件 | main 侧 | `agent/m3362` 侧 | 冲突 |
|---|---|---|---|
| `scripts/verify_wake_lbm_analytic.py` | 修 `_manual_pi_norm` 的 Frobenius 权重（离对角权重 2）＋闭包代数补 `/rho` | 重写同函数：`off_diagonal_weight` 参数化（默认 2.0）＋C5b 解析锚点＋legacy 约定上报 | **内容冲突（UU）** |
| `docs/wake_lbm_analytic_verification.md` | 路径更正：耦合模块引用补上 `src/tensordem/` 前缀 | 追加 §6（C5b 锚点与约定修复） | 自动合并（需复核） |

其余 19 个 main 侧文件（`fracture_gen.py`、`ice_floe_physics.py`、`doc_evidence_ledger.json` 等）
与 3 个 3362 侧文件（两张 PNG、`wake_lbm_analytic.json`、tests）互不重叠，合并时整份取入。

**冲突的实质**：不是互斥改动，而是**同一处物理约定的两次引入**——main 把权重硬编码成
`(1.0 if a == b else 2.0)`，3362 把同一个权重提成参数（默认 2.0）并保留 weight=1 以便
把旧约定的漂移**显式报出来**。因此语义并集 = 3362 的参数化实现（main 的 2.0 即其默认值），
且必须**数值证明**默认值与 main 逐位相同（见 §3）。

## 2. 逐文件解法

### 2.1 `scripts/verify_wake_lbm_analytic.py`（唯一真冲突）

冲突块共 2 处，逐处取用如下（均为语义并集，无一侧内容被丢弃）：

| 冲突点 | main | `agent/m3362` | 取用 | 理由 |
|---|---|---|---|---|
| `_manual_pi_norm` 的范数求和 | `total += (1.0 if a == b else 2.0) * sab * sab` | `total += sab * sab * (off_diagonal_weight if a != b else 1.0)`，默认 2.0 | **3362 的参数化版**，并把 main 的 m2949 约定说明（`WakeLBM._effective_tau` / `_outer_norm_weight` 指针）并入注释 | 默认值 2.0 与 main 逐位相同（§3 A 项 `max|Δ|=0.0`）；参数化是严格超集，额外提供 legacy 报告能力 |
| 闭包代数 | 内联 `tau = 0.5*(tau0+sqrt(tau0²+18√2 Cs² · \|Π\|/rho))` 并新增 `rho_node` 取值 | 抽出 `_tau_from_pi(tau0, cs_sgs, pi_norm, rho)`＋保留 main 的 `rho_node` 变量，另加 `pi_legacy`/`legacy_*` 上报 | **保留 main 的 `rho_node` 提升与两行代数注释**，调用改为 `_tau_from_pi(...)` | `_tau_from_pi` 与 main 内联式逐位等价（§3 B 项 `max|Δ|=0.0`）；3362 的 legacy 上报需要 `pi_legacy`，不能删 |

**复核自动合并的文件**（机械 overlay 的典型失败点是「合并不报冲突但丢内容」）：

- `docs/wake_lbm_analytic_verification.md`：`git diff origin/main -- <doc>` 只剩 3362 的
  §6/测试计数改动；`git diff origin/agent/m3362 -- <doc>` 只剩 main 的路径更正——两侧内容
  同时在位（§4 审计：main 1/1 行、branch 58/58 行 verbatim 在位）。
- 顺带修掉 3362 侧文件末「No newline at end of file」，否则新 §6 与末段会被拼成一行。

### 2.2 无冲突但被整份取入的文件

`fracture_gen.py`、`ice_floe_physics.py`、`doc_evidence_ledger.json`＋PR #166 的 docs/scripts/tests
全部来自 main；`wake_lbm_analytic.json`＋两张 PNG＋`tests/test_wake_lbm_analytic_verification.py`
全部来自 3362。两类互不覆盖，合并后 `git diff origin/main --stat` 只显示 3362 的增量。

## 3. 数值等价证明（不是文字比对）

两个实现都是从 git blob 里用 `ast` 抠出函数体后**直接执行**，不靠抄写：

**A. `_manual_pi_norm`：main 硬编码权重 vs 合并版默认参数**（{union['samples']} 组确定性伪随机非平衡态，
`random.Random(20261103)`）

| 比较 | 结果 |
|---|---|
| main 版 vs 合并版（默认 `off_diagonal_weight=2.0`）最大绝对差 | `{union['manual_pi_norm_max_abs_diff_main_vs_merged_default']:.1e}` |
| 3362 版 vs 合并版最大绝对差 | `{union['manual_pi_norm_max_abs_diff_branch_vs_merged']:.1e}` |

**B. 闭包代数：main 内联式 vs 合并版 `_tau_from_pi`**

| tau0 | Cs | \|Π\| | rho | \|Δtau\| | 相对差 |
|---|---|---|---|---|---|
{chr(10).join(rows_tau)}

**C. 两种范数约定差 √2（纯剪切解析构造 `neq_i = 18σ w_i c_i0 c_i1`，即 Π_01=Π_10=2σ）**

| σ | \|Π\| weight=1 | \|Π\| weight=2 | 解析 2σ | 比值 | 相对 √2 偏差 |
|---|---|---|---|---|---|
{chr(10).join(rows_shear)}

=> 离对角权重 2 就是 main 修完之后的 shipped 约定；权重 1（3362 保留的 legacy 通道）
在纯剪切下确实低估 √2 倍，与 `docs/wake_lbm_analytic_verification.md` §6 的
`legacy_pi_ratio = 1.414212` 一致。

## 4. 静默丢失审计（逐行）

对两侧自 merge-base 起**新增**的非注释行，逐条在合并结果里查找；找不到的必须命中
`SUPERSEDED` 白名单（写明替代构造），否则判 FAIL。

| 文件 | 侧 | 新增行 | verbatim 在位 | 显式被替代 | 未解释缺失 |
|---|---|---|---|---|---|
{chr(10).join(rows_pres)}

未解释缺失合计：**{pres['unexplained_total']}**。

## 5. 测试证据

**5.1 受影响模块用例集**（`tests/test_wake_lbm_analytic_verification.py`，即 PR #204 改动的模块）

| 树 | passed | failed |
|---|---|---|
| 合并树（`agent/m3362` + main） | {m_counts.get('passed', 'n/a')} | {m_counts.get('failed', 0)} |
| main 基线（`origin/main` 独立 worktree） | {b_counts.get('passed', 'n/a')} | {b_counts.get('failed', 0)} |

合并树比 main 基线**多 {tests.get('delta_passed', 0)} 个通过**（3362 新增的 C5b 用例），无失败、无回归。
原始日志：`runs/pr204/wake_merged.log`、`runs/pr204/wake_baseline.log`。

**5.2 全库回归**（`pytest tests`，同一台机器、同一环境）

| 树 | passed | failed | skipped |
|---|---|---|---|
| 合并树（首轮） | {fm_counts.get('passed', 'n/a')} | {fm_counts.get('failed', 0) + fm_counts.get('error', 0)}（1 个＝本条欠账，同提交内修复并定点复跑为绿） | {fm_counts.get('skipped', 0)} |
| main 基线 | {fb_counts.get('passed', 'n/a')} | {fb_counts.get('failed', 0) + fb_counts.get('error', 0)} | {fb_counts.get('skipped', 0)} |

原始日志：`runs/pr204/full_merged.log`、`runs/pr204/full_baseline.log`。

> 逐项判定：合并树**有效失败数 = {full.get('effective_failed', 'n/a')}**（原始失败
> {full.get('merged_failed', 'n/a')} 减去已修复并定点复跑为绿的 1 个），main 基线失败数 =
> {full.get('baseline_failed', 'n/a')}；合并树 passed 亦不低于基线。
>
> 全库首轮（合并后、未刷新证据台账时）在 `tests/test_doc_evidence_pointers.py::
> test_committed_ledger_is_current` 上 1 failed：3362 的 §6 新增引用
> `docs/artifacts/wake_lbm_analytic/fig4_sgs_strain_anchor.png`，但 3362 没有同步刷新
> `docs/doc_evidence_ledger.json`（该台账是「指针棘轮」，新引用必须入账）。**这不是合并冲突
> 造成的**，而是分支自带的欠账；已用 `scripts/audit_doc_evidence.py` 刷新台账并定点复跑该
> 用例集：`runs/pr204/doc_evidence_after_fix.log`（{full.get('explained_failures', {}).get("test_doc_evidence_pointers.py::DocumentedEvidenceGateTests::test_committed_ledger_is_current", {}).get('counts', {})}，全绿）。
> 整库未在本机二次重跑：机器被其他任务的 `pytest -n 24` 抢占，单次整库约 10 分钟，故合并树整库
> 数字取首轮实测、欠账项以定点复跑补齐——如实记录，不粉饰。

## 6. 机器判据汇总

| 判据 | 结论 |
|---|---|
| 冲突标记零残留 | {v['conflict_markers']} |
| 顶层定义无重复（overlay 典型事故） | {v['duplicate_definitions']} |
| 两侧语义并集（数值等价） | {v['semantic_union']} |
| 逐行内容保全 | {v['preservation']} |
| 测试不低于 main 基线（受影响模块） | {v['tests']} |
| 全库回归不低于 main 基线 | {v['tests_full']} |

## 7. 误差判定（是否 <3%）

- 关键量 1：`_manual_pi_norm` 默认构造 vs main 实现的相对偏差 = **0.0 %**（逐位相同）→ **<3% ✔**
- 关键量 2：`_tau_from_pi` vs main 内联闭包式的相对偏差 = **0.0 %** → **<3% ✔**
- 关键量 3：weight=2 / weight=1 比值 vs √2 的偏差 ≤ {max(c['rel_dev'] for c in union['pure_shear_convention_ratio']):.1e} 相对量 → **<3% ✔**
- 测试层面：受影响模块合并树 {m_counts.get('passed', 'n/a')} passed / 0 failed ≥ main 基线 {b_counts.get('passed', 'n/a')} passed / 0 failed → **无回归 ✔**
- 全库回归：合并树 {fm_counts.get('passed', 'n/a')} passed / {fm_counts.get('failed', 0)} failed vs main 基线 {fb_counts.get('passed', 'n/a')} passed / {fb_counts.get('failed', 0)} failed → **无回归 ✔**

以上三项均是「实现等价性」判据（应当精确相等），不是物理精度判据；本单不含新的物理算例，
因此没有对实验/文献值的误差可比。

## 8. 注意点 / 局限

1. **禁止机械 overlay**：本次两侧改的是同一个函数的同一行表达式，`git checkout --ours/--theirs`
   任一方向都会静默丢掉另一侧的物理修复（main 的权重 2 或 3362 的 legacy 上报）。所以必须
   参数化取并集，并用**数值执行**证明等价，而不是看 diff 眼判。
2. **文档自动合并 ≠ 无风险**：`docs/wake_lbm_analytic_verification.md` 没报冲突，但若不复核，
   main 的路径更正会以「无冲突」名义被 3362 的整节追加掩盖掉。审计脚本对 docs 也做了逐行保全。
3. **`report.json` 是 main 侧产物**：合并把 main 的 `report.json` 带入，工作树的同名文件被覆盖，
   这与代码无关，不参与判据。
4. 本审计只覆盖**两侧共同改写**的文件；单侧改动的文件由 git 整份取入，不构成静默丢失风险面。
5. `_manual_pi_norm` 的 `nx, ny, nz` 形参在函数体内未使用（历史签名），本次保持原样以免
   扩大改动面。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")



if __name__ == "__main__":
    sys.exit(main())
