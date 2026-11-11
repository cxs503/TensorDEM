"""Static import-contract audit for the TensorDEM test surface (m14846).

Motivation
----------
A test module that cannot be collected is a *published verification that cannot
run*: the documentation claims are then backed by nothing.  The classic cause in
this repository (m5754/m14450) is a package directory shadowing a same-named
module file, which silently removes the module's public symbols.

This tool re-derives, **without importing anything**, every ``from tensordem... 
import ...`` statement in ``tests/`` and resolves the requested names against the
static top-level namespace of the target module (``def``/``class``/assignment/
``import`` binding/``__all__`` re-exports).  Output is JSON + a markdown table so
the result is reproducible and auditable.

Usage
-----
    PYTHONPATH=src python scripts/m14846_import_contract_audit.py \
        --out artifacts/m14846_import_contract.json \
        --md docs/DEM_IMPORT_CONTRACT_AUDIT_M14846.md
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
_STAR = "*"  # marker: module has a star import -> namespace is open


def _module_path(name: str) -> Path | None:
    parts = name.split(".")
    if parts[0] != "tensordem":
        return None
    rel = SRC.joinpath(*parts)
    # CPython resolution order: a package directory shadows a same-named module
    # file (FileFinder tries ``dir/__init__.py`` first), so mirror that here.
    for cand in (rel / "__init__.py", rel.with_suffix(".py")):
        if cand.is_file():
            return cand
    return None


def _shadowed(name: str) -> bool:
    """True when both ``x.py`` and ``x/`` exist (the silent-shadow defect class)."""
    parts = name.split(".")
    rel = SRC.joinpath(*parts)
    return rel.with_suffix(".py").is_file() and (rel / "__init__.py").is_file()


def _top_level_names(path: Path) -> set[str]:
    """Static module-level namespace (no execution).

    Walks *all* module-scope statements, including names bound inside
    ``if``/``try`` blocks (conditional exports are still exports when the
    condition holds at import time), but never descends into function bodies.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    has_star = False

    def walk(nodes):
        nonlocal has_star
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Lambda)):
                names.add(getattr(node, "name", ""))
                continue  # do not descend into bodies
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, python_imports := (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        has_star = True
                    else:
                        names.add(alias.asname or alias.name.split(".")[0])
            for field in ("body", "orelse", "finalbody", "handlers"):
                child = getattr(node, field, None)
                if isinstance(child, list):
                    walk(child)

    walk(tree.body)
    names.discard("")
    if has_star:
        names.add(_STAR)
    return names


def _enclosing_function(tree: ast.AST, target: ast.AST) -> bool:
    """True if ``target`` lives inside a function body (deferred import)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


def audit() -> dict:
    modules: dict[str, set[str] | None] = {}

    def exports(mod: str) -> set[str] | None:
        if mod not in modules:
            p = _module_path(mod)
            modules[mod] = None if p is None else _top_level_names(p)
        return modules[mod]

    rows = []
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(test.read_text(encoding="utf-8"), filename=str(test))
        except SyntaxError as exc:  # a syntax error is itself a broken contract
            rows.append({"test": test.name, "module": "<syntax>", "missing": ["<SyntaxError>"],
                         "missing_module": False, "line": exc.lineno})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            mod = node.module or ""
            if not mod.startswith("tensordem"):
                continue
            scope = "function" if _enclosing_function(tree, node) else "module"
            ns = exports(mod)
            if ns is not None and _STAR in ns:
                continue  # open namespace (``from x import *``): cannot statically refute
            if ns is None:
                rows.append({"test": test.name, "module": mod, "missing": ["<module not found>"],
                             "missing_module": True, "line": node.lineno, "scope": scope})
                continue
            # a name may also be a *submodule* (``from pkg import sub``), which is
            # valid even though it is not in the module's top-level namespace
            missing = [a.name for a in node.names
                       if a.name != "*" and a.name not in ns
                       and _module_path(mod + "." + a.name) is None]
            if missing:
                rows.append({"test": test.name, "module": mod, "missing": missing,
                             "missing_module": False, "line": node.lineno, "scope": scope})
    shadows = sorted({r["module"] for r in rows
                      if r["missing_module"] or _shadowed(r["module"])})
    return {
        "protocol": "tensordem-import-contract-audit-v1",
        "shadowed_modules": shadows,
        "n_test_modules": len(list((ROOT / "tests").glob("test_*.py"))),
        "n_broken_contracts": len(rows),
        "broken": rows,
    }


def to_markdown(res: dict) -> str:
    lines = [
        "# TensorDEM 测试导入契约静态审计（m14846）",
        "",
        f"协议：`{res['protocol']}`　扫描测试模块：{res['n_test_modules']}　"
        f"静态不可解析导入：{res['n_broken_contracts']}　同名文件/包遮蔽："
        f"{len(res['shadowed_modules'])}",
        "",
        "方法：对 `tests/test_*.py` 做 AST 静态解析（**不 import**），把 "
        "`from tensordem... import X` 的每个名字与目标模块（`src/tensordem/**.py` 或 "
        "`**/__init__.py`）的顶层命名空间（def/class/赋值/import 绑定）求交；缺失即"
        "「已合并的检查无法运行」。",
        "",
        "| 测试模块 | 导入目标 | 缺失符号 | 行 | 判读 |",
        "|---|---|---|---|---|",
    ]
    for r in res["broken"]:
        verdict = "目标模块不存在/被同名包遮蔽" if r["missing_module"] else "符号未导出"
        lines.append(f"| {r['test']} | `{r['module']}` | {', '.join(r['missing'])} | "
                     f"{r['line']} | {verdict} |")
    if not res["broken"]:
        lines.append("| — | — | — | — | 全部可解析 |")
    lines += [
        "",
        "## 说明与局限",
        "",
        "* 这是**静态**契约检查：能抓住「名字没导出/模块被同名包遮蔽/测试语法错」这一类"
        "让 pytest 在 collection 阶段就失败的缺陷；抓不住运行期才暴露的问题（数值错、"
        "依赖缺失、参数签名变化）。",
        "* `import *` 语句被跳过（无法静态判名）。",
        "* 该审计是**门禁**：`tests/test_import_contract_audit.py` 把「不可解析导入数」"
        "钉在已知基线（含已在他单修复的条目），新增即失败。",
        "",
        "复现：",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/m14846_import_contract_audit.py \\",
        "    --out artifacts/m14846_import_contract.json \\",
        "    --md docs/DEM_IMPORT_CONTRACT_AUDIT_M14846.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "m14846_import_contract.json"))
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    res = audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(to_markdown(res), encoding="utf-8")
    print(json.dumps({"n_test_modules": res["n_test_modules"],
                      "n_broken_contracts": res["n_broken_contracts"],
                      "broken": res["broken"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()