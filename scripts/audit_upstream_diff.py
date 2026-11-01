#!/usr/bin/env python3
"""m1630 —— TensorDEM 上游差集巡更（可复现审计工具）。

做三件事，全部走 git plumbing，输出机器可核的 JSON：

1. 差集清单：``<base>..<head>`` 的提交逐条记录（作者日、主题、增删行、文件清单、
   父提交数），并按「变更主题」归类（主题判定 = 提交前缀 + 触碰路径 + 关键词）。
2. 风险标注：对每条提交打 R-* 标记，每条标记附判定规则与命中证据
   （触碰的路径 / 数字），不做不可复算的主观评级。
3. 文档悬空引用：对差集中新增/修改的 ``docs/*.md``，抽出其中引用的仓库相对路径
   （benchmarks/ scripts/ artifacts/ runs/ src/ tests/），与 head 树比对，
   报出「文档承诺但树里不存在」的悬空引用 —— 这是交付可追溯性的硬证据。

用法::

    PYTHONPATH= python scripts/audit_upstream_diff.py \
        --base origin/main --head HEAD --out artifacts/m1630/upstream_diff.json

设计纪律：只读 git，不写仓库文件之外的东西；输出 JSON 的键顺序稳定，
同一 commit 集合重复运行字节级一致（``--check`` 会自校验一次幂等性）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- git helpers


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def resolve(ref: str) -> str:
    return git("rev-parse", "--verify", ref).strip()


def commits_between(base: str, head: str) -> list[dict]:
    """base..head 的全部提交（含 merge），按时间正序返回。"""
    fmt = "%H\x1f%h\x1f%ad\x1f%an\x1f%P\x1f%s"
    raw = git("log", f"--format={fmt}", "--date=short", "--reverse", f"{base}..{head}")
    out = []
    for line in raw.splitlines():
        sha, short, date, author, parents, subject = line.split("\x1f")
        out.append(
            {
                "sha": sha,
                "short": short,
                "date": date,
                "author": author,
                "parents": parents.split() if parents else [],
                "subject": subject,
                "files": [],
            }
        )
    return out


def attach_files(commits: list[dict], first_parent: bool = False) -> None:
    """为每条提交挂上 numstat 文件清单。merge 提交默认 -m 取第一父的差集。"""
    for c in commits:
        args = ["show", "--numstat", "--format=", "--no-renames", c["sha"]]
        if len(c["parents"]) > 1:
            args = ["diff", "--numstat", "--no-renames", c["parents"][0], c["sha"]]
        text = git(*args)
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            add, dele, path = parts
            c["files"].append(
                {
                    "path": path,
                    "add": 0 if add == "-" else int(add),
                    "del": 0 if dele == "-" else int(dele),
                }
            )


# ---------------------------------------------------------------- 分类与风险

CORE_SOLVER = {
    "src/tensordem/dem.py",
    "src/tensordem/dem3d.py",
    "src/tensordem/dem3d_wedge.py",
    "src/tensordem/dem3d_energy.py",
    "src/tensordem/calibrated_dem.py",
    "src/tensordem/buoyancy_wet.py",
    "src/tensordem/calibrated_wet_coupling.py",
    "src/tensordem/contact_tangential.py",
    "src/tensordem/hull_contact.py",
    "src/tensordem/continuum_boundary_dem.py",
    "src/tensordem/moving_plane_boundary.py",
}

THEME_PRIORITY = [
    ("T-DELIVERABLE", ("matured mission deliverables",)),
    ("T-SOLVER", ("fix(", "fix:", "hotfix", "红", "载荷", "压板", "pinned support")),
    ("T-ARTIFACT", ("artifact", "s3", "upload")),
    ("T-TIMESTEP", ("fixed-duration", "time-step", "timestep")),
    ("T-FEM-DEM", ("fem", "fracture_gen", "破坏面")),
    ("T-CI", (".github", "ci ", "pytest suite")),
]

CODE_PREFIXES = ("src/", "scripts/", "tests/", "benchmarks/")


def _subject_hit(c: dict, keys: tuple[str, ...]) -> bool:
    subj = c["subject"].lower()
    return any(k.lower() in subj for k in keys)


def theme_of(c: dict) -> dict:
    """主主题：先按 THEME_PRIORITY 的关键词命中取第一个，再退化到路径构成。

    主题只用于「变更主题清单」的归类计数；提交究竟改了什么以 files 为准。
    merge 只作为 ``is_merge`` 标记，不抢占主主题（否则收口提交会把主题表吞掉）。
    """
    if len(c["parents"]) > 1:
        return {"primary": "T-MERGE", "secondary": []}
    hits = [name for name, keys in THEME_PRIORITY if _subject_hit(c, keys)]
    code_paths = [f["path"] for f in c["files"] if f["path"].startswith(CODE_PREFIXES)]
    doc_paths = [f["path"] for f in c["files"] if f["path"].startswith("docs/")]
    if hits:
        primary = hits[0]
    elif code_paths:
        if all(p.startswith("tests/") for p in code_paths):
            primary = "T-TEST"
        elif doc_paths:
            primary = "T-DOCS-TESTS"
        else:
            primary = "T-PROTO"
    elif doc_paths:
        primary = "T-DOCS"
    else:
        primary = "T-MERGE" if len(c["parents"]) > 1 else "T-OTHER"
    return {"primary": primary, "secondary": [h for h in hits if h != primary]}


FILTERED_RE = re.compile(r"(\d+)\s*个文件被门禁滤除")
MISSION_RE = re.compile(r"^m(\d+):")


def risk_of(c: dict) -> list[dict]:
    """风险打标。每条标记给出规则名 + 证据，便于独立复核。"""
    paths = [f["path"] for f in c["files"]]
    flags = []
    core = sorted(set(paths) & CORE_SOLVER)
    tests = [p for p in paths if p.startswith("tests/")]
    docs = [p for p in paths if p.startswith("docs/")]
    ci = [p for p in paths if p.startswith(".github/")]

    if core:
        sev = "high" if not tests else "medium"
        flags.append(
            {
                "code": "R-PHYS-CORE",
                "severity": sev,
                "evidence": {"core_files": core, "tests_in_same_commit": len(tests)},
                "why": "触碰 DEM 核心求解器；同提交无回归测试时，误差只能靠后续基准战役兜底",
            }
        )
    if len(c["parents"]) > 1:
        flags.append(
            {
                "code": "R-MERGE-COLLISION",
                "severity": "low",
                "evidence": {"merged_short": c["parents"][1][:7]},
                "why": "并行分支收口；conftest/样板归一类冲突可能静默改变测试装配",
            }
        )
    m = FILTERED_RE.search(c["subject"])
    if m:
        n = int(m.group(1))
        flags.append(
            {
                "code": "R-GATE-FILTERED",
                "severity": "high" if n >= 20 else "medium",
                "evidence": {"filtered_files_declared": n},
                "why": "提交自述有产物被门禁滤除；滤除内容不在历史里，交付可追溯性缺口",
            }
        )
    if paths and not docs and any(p.startswith(("src/", "scripts/")) for p in paths):
        flags.append(
            {
                "code": "R-NO-DOC",
                "severity": "low",
                "evidence": {"code_files": sum(p.startswith(("src/", "scripts/")) for p in paths)},
                "why": "改代码/基准脚本但不改 docs；后续接手者无书面口径",
            }
        )
    drops = [f["path"] for f in c["files"] if f["del"] > 0 and f["add"] == 0]
    if len(c["parents"]) > 1 and len(drops) >= 5:
        flags.append(
            {
                "code": "R-DELIVERABLE-DROP",
                "severity": "high" if len(drops) >= 20 else "medium",
                "evidence": {"files_deleted": len(drops), "sample": drops[:5]},
                "why": "收口 merge 相对第一父删掉整批文件；若相关交付物未在别处复现，等于静默撤回",
            }
        )
    if ci:
        flags.append(
            {
                "code": "R-CI-SURFACE",
                "severity": "medium",
                "evidence": {"ci_files": ci},
                "why": "CI 触发面变更：new path 未纳入 paths 过滤时该用例在 PR 上不会跑",
            }
        )
    if any(k in c["subject"].lower() for k in ("fix(", "fix:", "红", "hotfix")):
        flags.append(
            {
                "code": "R-MAIN-RED",
                "severity": "high",
                "evidence": {"subject": c["subject"]},
                "why": "上游 main 曾红；说明合入前 CI 未全绿，本轮差集需重跑根因用例",
            }
        )
    return flags


def path_class(p: str) -> str:
    for pre in ("src/", "tests/", "docs/", "benchmarks/", "scripts/", ".github/", "prereg/", "examples/"):
        if p.startswith(pre):
            return pre.rstrip("/")
    return "other"


# ---------------------------------------------------------------- 文档悬空引用

REF_RE = re.compile(r"(?<![\w/])((?:benchmarks|scripts|artifacts|runs|src|tests|docs)/[\w./-]+)")


def dangling_refs(commits: list[dict], head_tree: set[str]) -> list[dict]:
    """docs 里引用的仓库路径若在 head 树不存在 → 悬空引用。"""
    touched_docs = sorted(
        {
            f["path"]
            for c in commits
            for f in c["files"]
            if f["path"].startswith("docs/") and f["path"].endswith(".md")
        }
    )
    out = []
    for doc in touched_docs:
        try:
            body = git("show", f"HEAD:{doc}") if doc in head_tree else ""
        except SystemExit:
            body = ""
        if not body:
            continue
        refs = sorted({m.group(1).rstrip(".,)`") for m in REF_RE.finditer(body)})
        missing = [r for r in refs if r not in head_tree]
        if missing:
            out.append({"doc": doc, "refs": len(refs), "missing": missing})
    return out




# ---------------------------------------------------------------- 净差集与翻覆（churn）

def net_diff(base: str, head: str) -> dict:
    """净树差集：只算 base 与 head 两棵树的最终差异，不受中间提交增删来回的影响。"""
    raw = git("diff", "--numstat", "--no-renames", base, head)
    rows = []
    for line in raw.splitlines():
        a, d, path = line.split("\t")
        rows.append({"path": path, "add": 0 if a == "-" else int(a), "del": 0 if d == "-" else int(d)})
    status = Counter()
    for line in git("diff", "--name-status", "--no-renames", base, head).splitlines():
        status[line.split("\t")[0][0]] += 1
    return {
        "files": len(rows),
        "insertions": sum(r["add"] for r in rows),
        "deletions": sum(r["del"] for r in rows),
        "status_counts": dict(sorted(status.items())),
        "_rows": rows,
    }


def churn(commits: list[dict], head: str) -> dict:
    """上游窗口内被删除、且最终树上不存在的文件 —— 交付物翻覆（add-then-drop）。"""
    tree = set(git("ls-tree", "-r", "--name-only", head).splitlines())
    added: dict[str, list[str]] = {}
    deleted: dict[str, list[str]] = {}
    for c in commits:
        for f in c["files"]:
            if f["add"] > 0 and f["del"] == 0:
                added.setdefault(f["path"], []).append(c["short"])
            if f["del"] > 0 and f["add"] == 0:
                deleted.setdefault(f["path"], []).append(c["short"])
    # 内容级核验：被删文件是否以同 blob 改名存在于 head（move ≠ loss）
    blob_paths: dict[str, list[str]] = {}
    for line in git("ls-tree", "-r", head).splitlines():
        meta, path = line.split("\t", 1)
        blob_paths.setdefault(meta.split()[2], []).append(path)
    gone = {}
    for p, dels in sorted(deleted.items()):
        if p in tree:
            continue
        blob = ""
        try:
            blob = git("rev-parse", f"{head}~0:{p}").strip()
        except SystemExit:
            blob = ""
        if not blob:
            for c in dels:  # 该文件在 head 上已被删，取删除前最后一版
                try:
                    blob = git("rev-parse", f"{c}^1:{p}").strip()
                except SystemExit:
                    blob = ""
                if blob:
                    break
        moved_to = [q for q in blob_paths.get(blob, []) if q != p]
        gone[p] = {
            "added_by": added.get(p, []),
            "deleted_by": dels,
            "moved_to": sorted(moved_to),
            "lost": not moved_to,
        }
    touched = {f["path"] for c in commits for f in c["files"]}
    # 丢失模块的能力消失核验：head 树里还有没有对它的引用（import / 文档 / 测试）
    lost_src = {}
    for path, info in gone.items():
        if not (path.startswith("src/") and info["lost"]):
            continue
        stem = path.rsplit("/", 1)[-1][:-3]
        # 用 import 限定名（tensordem.<stem>）计数：裸词干会被 formation/granular 这类普通词污染
        hits = subprocess.run(
            ["git", "grep", "-l", "-F", f"tensordem.{stem}", head, "--",
             "src", "scripts", "tests", "docs", "benchmarks", "examples"],
            capture_output=True, text=True,
        ).stdout.splitlines()
        lost_src[stem] = {"import_refs_at_head": len(hits), "files": hits[:3]}
    return {
        "files_touched_union": len(touched),
        "lost_src_modules": dict(sorted(lost_src.items())),
        "lost_src_modules_with_import_refs_at_head": sum(
            1 for v in lost_src.values() if v["import_refs_at_head"]),
        "files_absent_at_head": len(gone),
        "files_lost_no_same_content_at_head": sum(1 for v in gone.values() if v["lost"]),
        "files_relocated_same_blob": sum(1 for v in gone.values() if v["moved_to"]),
        "churn_ratio": round(len(gone) / len(touched), 4) if touched else None,
        "absent_files": gone,
    }

# ---------------------------------------------------------------- CI 触发面覆盖

PATH_LINE_RE = re.compile(r'^\s*-\s*["\']([^"\']+)["\']\s*$')


def ci_surface(head: str) -> tuple[dict, list[str]]:
    """抽出 .github/workflows 里 paths: 过滤器的显式路径清单（当前上游口径）。"""
    files = [f for f in git("ls-tree", "-r", "--name-only", head).splitlines()
             if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))]
    per_wf: dict[str, list[str]] = {}
    for wf in files:
        body = git("show", f"{head}:{wf}")
        paths, in_paths = [], False
        for line in body.splitlines():
            if re.match(r"^\s{4,6}paths:\s*$", line):
                in_paths = True
                continue
            if in_paths:
                m = PATH_LINE_RE.match(line)
                if m:
                    paths.append(m.group(1))
                elif line.strip() and not line.strip().startswith("#"):
                    in_paths = False
        per_wf[wf] = paths
    unrestricted = [wf for wf, paths in per_wf.items() if not paths]
    return per_wf, unrestricted


def _covered(path: str, filters: list[str]) -> bool:
    import fnmatch

    for f in filters:
        if fnmatch.fnmatch(path, f) or path == f:
            return True
        if f.endswith("/**") and path.startswith(f[:-3] + "/"):
            return True
    return False


def ci_gaps(commits: list[dict], head: str) -> dict:
    """差集里新增/改动的 scripts|tests|docs 文件，若未被任何 workflow 的 paths 覆盖，
    PR 上就不会触发对应用例——这是「测试存在但不跑」的结构性风险。"""
    per_wf, unrestricted = ci_surface(head)
    all_filters = [f for v in per_wf.values() for f in v]
    watched = sorted(
        {
            f["path"]
            for c in commits
            for f in c["files"]
            if f["path"].startswith(("scripts/", "tests/", "docs/"))
            and f["path"] not in {"/dev/null"}
        }
    )
    uncovered = [w for w in watched if not _covered(w, all_filters)]
    # 全量工作流（无 paths 过滤器）覆盖一切；缺口只在「仅靠显式枚举」的口径下成立
    gate = [w for w in watched if not _covered(w, all_filters)]
    return {
        "unrestricted_workflows": unrestricted,
        "benchmark_gate_uncovered": gate,
        "benchmark_gate_uncovered_ratio": round(len(gate) / len(watched), 4) if watched else None,
        "workflows": {k: len(v) for k, v in sorted(per_wf.items())},
        "explicit_path_filters_total": len(all_filters),
        "watched_files_in_diff": len(watched),
        "uncovered_files": uncovered,
        "uncovered_ratio_without_unrestricted": (
            0.0 if unrestricted else (round(len(uncovered) / len(watched), 4) if watched else None)
        ),
    }


# ---------------------------------------------------------------- main


def build(base_ref: str, head_ref: str) -> dict:
    base = resolve(base_ref)
    head = resolve(head_ref)
    commits = commits_between(base, head)
    attach_files(commits)
    for c in commits:
        th = theme_of(c)
        c["primary_theme"] = th["primary"]
        c["is_merge"] = len(c["parents"]) > 1
        c["secondary_themes"] = th["secondary"]
        c["risks"] = risk_of(c)
        c["classes"] = dict(sorted(Counter(path_class(f["path"]) for f in c["files"]).items()))
        c["add"] = sum(f["add"] for f in c["files"])
        c["del"] = sum(f["del"] for f in c["files"])

    tree = set(git("ls-tree", "-r", "--name-only", head).splitlines())
    counts = Counter()
    per_theme_files = defaultdict(Counter)
    for c in commits:
        for f in c["files"]:
            per_theme_files[c["primary_theme"]][path_class(f["path"])] += 1
        for r in c["risks"]:
            counts[r["code"]] += 1

    src_commits = [c for c in commits if any(f["path"].startswith("src/") for f in c["files"])]
    paired = [c for c in src_commits if any(f["path"].startswith("tests/") for f in c["files"])]
    filtered_total = sum(
        int(m.group(1)) for c in commits for m in [FILTERED_RE.search(c["subject"])] if m
    )
    misses = dangling_refs(commits, tree)
    nd = net_diff(base, head)
    ch = churn(commits, head)

    return {
        "meta": {
            "tool": "scripts/audit_upstream_diff.py",
            "base_ref": base_ref,
            "head_ref": head_ref,
            "base_sha": base,
            "head_sha": head,
            "n_commits": len(commits),
            "n_commits_no_merge": sum(1 for c in commits if len(c["parents"]) < 2),
            "files_touched_unique": len({f["path"] for c in commits for f in c["files"]}),
            "insertions": sum(c["add"] for c in commits),
            "deletions": sum(c["del"] for c in commits),
            "per_commit_sum_insertions": sum(c["add"] for c in commits),
            "per_commit_sum_deletions": sum(c["del"] for c in commits),
            "net_files": nd["files"],
            "net_insertions": nd["insertions"],
            "net_deletions": nd["deletions"],
            "net_status_counts": nd["status_counts"],
            "first_date": commits[0]["date"] if commits else None,
            "last_date": commits[-1]["date"] if commits else None,
        },
        "metrics": {
            "risk_flag_counts": dict(sorted(counts.items())),
            "src_commits": len(src_commits),
            "src_commits_with_tests": len(paired),
            "src_test_pairing_ratio": (
                round(len(paired) / len(src_commits), 4) if src_commits else None
            ),
            "gate_filtered_files_declared": filtered_total,
            "docs_with_dangling_refs": len(misses),
            "ci_benchmark_gate_uncovered": len(
                [w for w in set(f["path"] for c in commits for f in c["files"])
                 if w.startswith(("scripts/", "tests/", "docs/")) and not _covered(
                     w, [p for v in ci_surface(head)[0].values() for p in v])]
            ),
            "commits_touching_ci": sum(
                1 for c in commits if any(f["path"].startswith(".github/") for f in c["files"])
            ),
            "files_by_class": dict(
                sorted(Counter(path_class(f["path"]) for c in commits for f in c["files"]).items())
            ),
        },
        "theme_matrix": {
            t: dict(sorted(cols.items())) for t, cols in sorted(per_theme_files.items())
        },
        "net_by_class": {
            c: {"files": sum(1 for r in nd["_rows"] if path_class(r["path"]) == c),
                "insertions": sum(r["add"] for r in nd["_rows"] if path_class(r["path"]) == c),
                "deletions": sum(r["del"] for r in nd["_rows"] if path_class(r["path"]) == c)}
            for c in sorted({path_class(r["path"]) for r in nd["_rows"]})
        },
        "churn": ch,
        "dangling_refs": misses,
        "divergence": divergence(base, head),
        "ci_coverage": ci_gaps(commits, head),
        "commits": commits,
    }


def divergence(a: str, b: str) -> dict:
    """a 相对 b 的 ahead/behind（git 语义：a...b）。"""
    left, right = git("rev-list", "--left-right", "--count", f"{a}...{b}").split()
    return {"ahead_of_b": int(left), "behind_b": int(right), "a": a[:7], "b": b[:7]}


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorDEM 上游差集巡更")
    ap.add_argument("--base", default="HEAD", help="差集起点（默认本地 HEAD → 得到上游独有提交）")
    ap.add_argument("--head", default="origin/main", help="差集终点（默认上游 origin/main）")
    ap.add_argument("--out", default="artifacts/m1630/upstream_diff.json")
    ap.add_argument("--check", action="store_true", help="仅重算一次并校验幂等，不写文件")
    args = ap.parse_args(argv)

    payload = build(args.base, args.head)
    if not payload["meta"]["n_commits"]:
        print(
            "warning: base..head 为空——上游已落后或 ref 名反向；"
            "本仓约定 --base HEAD --head origin/main 取上游独有提交",
            file=sys.stderr,
        )
    text = to_json(payload)
    if args.check:
        again = to_json(build(args.base, args.head))
        same = again == text
        print(f"idempotent={same} commits={payload['meta']['n_commits']}")
        return 0 if same else 1

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    m = payload["meta"]
    print(
        f"wrote {out} commits={m['n_commits']} files={m['files_touched_unique']} "
        f"+{m['insertions']}/-{m['deletions']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())