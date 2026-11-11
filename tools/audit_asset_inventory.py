#!/usr/bin/env python3
"""audit_asset_inventory — AUDIT-ASSET-12REPO / TensorDEM 资产盘点器.

扫描 main 上近期已并 agent PR（git merge commits），对每个合并的**新增/修改文件**
做分类与静态体检，输出 JSON + markdown 表。分类：
  core    src/tensordem/** 下的算法模块
  test    tests/**
  doc     docs/** / *.md 报告
  script  scripts/** 运行器
  artifact artifacts/** 工件
  other   其余
静态体检：stub/NotImplemented/TODO 标记、是否被仓内其它文件 import（接入判定）、
与既有模块的顶层函数名重叠（重复实现判定）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

STUB_PAT = re.compile(
    r"raise\s+NotImplementedError|NotImplemented\b|#\s*(?:TODO|FIXME|XXX)\b|\bTODO\b|"
    r"\bstub\b|\bplaceholder\b|not implemented|未实现|占位",
    re.IGNORECASE,
)
DEF_PAT = re.compile(r"^def\s+([A-Za-z_][\w]*)\s*\(", re.MULTILINE)
IMPORT_PAT = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)


def sh(*args: str, cwd: str = ".") -> str:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout


def classify(path: str) -> str:
    p = Path(path)
    if path.startswith("src/tensordem/") and p.suffix == ".py":
        return "core"
    if path.startswith("tests/") and p.suffix == ".py":
        return "test"
    if path.startswith("docs/") or p.suffix == ".md":
        return "doc"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("artifacts/"):
        return "artifact"
    return "other"


def merge_window(repo: str, n: int) -> list[dict]:
    out = sh("git", "log", "--merges", "--format=%H\t%ai\t%s", "-n", str(n), cwd=repo)
    merges = []
    for line in out.strip().splitlines():
        h, date, subj = line.split("\t", 2)
        merges.append({"sha": h, "date": date, "subject": subj})
    return merges


def file_stats(repo: str, parent: str, mer: str) -> list[dict]:
    out = sh("git", "diff", "--numstat", parent, mer, cwd=repo)
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        if path.endswith(".py"):
            mode = "binary" if add == "-" else "text"
        else:
            mode = "binary" if add == "-" else "text"
        rows.append({"path": path, "added": None if add == "-" else int(add),
                     "deleted": None if dele == "-" else int(dele)})
    return rows


def static_probe(repo: str, path: str) -> dict:
    """对仓内现存文件做静态体检（工作树当前版本）。"""
    fp = Path(repo) / path
    info = {"exists": fp.exists(), "loc": 0, "stub_markers": 0, "stub_samples": [],
            "defs": 0, "imported_by": 0, "importers": []}
    if not fp.exists() or fp.suffix != ".py":
        return info
    txt = fp.read_text(errors="replace")
    info["loc"] = txt.count("\n") + 1
    hits = STUB_PAT.findall(txt)
    info["stub_markers"] = len(hits)
    for m in STUB_PAT.finditer(txt):
        line = txt[: m.start()].count("\n") + 1
        info["stub_samples"].append(f"{path}:{line}: {m.group(0)[:40]}")
        if len(info["stub_samples"]) >= 3:
            break
    info["defs"] = len(DEF_PAT.findall(txt))
    return info


def import_index(repo: str) -> dict[str, set[str]]:
    """module name -> set of files importing it (whole repo)."""
    idx: dict[str, set[str]] = defaultdict(set)
    root = Path(repo)
    for fp in root.rglob("*.py"):
        rel = str(fp.relative_to(root))
        if rel.startswith(".git"):
            continue
        try:
            txt = fp.read_text(errors="replace")
        except Exception:
            continue
        for m in IMPORT_PAT.finditer(txt):
            mod = m.group(1) or m.group(2) or ""
            if mod:
                idx[mod.split(".")[0]].add(rel)
                idx[mod].add(rel)
    return idx


def dup_index(repo: str) -> dict[str, list[str]]:
    """top-level function name -> files defining it (duplication detection)."""
    idx: dict[str, list[str]] = defaultdict(list)
    root = Path(repo)
    for fp in (root / "src").rglob("*.py"):
        rel = str(fp.relative_to(root))
        try:
            txt = fp.read_text(errors="replace")
        except Exception:
            continue
        for name in DEF_PAT.findall(txt):
            if name.startswith("_"):
                continue
            idx[name].append(rel)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("-n", "--merges", type=int, default=25)
    ap.add_argument("--out", default="artifacts/audit_m14487/inventory.json")
    args = ap.parse_args()
    repo = args.repo
    merges = merge_window(repo, args.merges)
    idx = import_index(repo)
    dup = dup_index(repo)

    records = []
    for m in merges:
        try:
            files = file_stats(repo, m["sha"] + "^1", m["sha"])
        except subprocess.CalledProcessError:
            continue
        for f in files:
            rec = {"pr": m["subject"], "date": m["date"][:10], "sha": m["sha"][:8], **f}
            rec["kind"] = classify(f["path"])
            if rec["kind"] in ("core", "script", "test"):
                rec.update(static_probe(repo, f["path"]))
                stem = Path(f["path"]).stem
                mod = f"tensordem.{stem}"
                importers = idx.get(mod, set()) | idx.get(stem, set())
                importers = {i for i in importers if i != f["path"]}
                rec["importers"] = sorted(importers)[:6]
                rec["imported_by"] = len(importers)
                rec["dup_funcs"] = {
                    n: [p for p in ps if p != f["path"]]
                    for n, ps in ((n, [p for p in dup.get(n, []) if p != f["path"]])
                                  for n in set(DEF_PAT.findall(
                                      (Path(repo) / f["path"]).read_text(errors="replace")
                                      if (Path(repo) / f["path"]).exists() else ""))
                    ) if any(ps) and n in dup
                }
            records.append(rec)

    agg = defaultdict(lambda: {"files": 0, "added": 0})
    for r in records:
        a = agg[r["kind"]]
        a["files"] += 1
        a["added"] += r["added"] or 0

    out = {"repo": repo, "merges": [m["subject"] for m in merges],
           "agg": dict(agg), "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(dict(agg), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()