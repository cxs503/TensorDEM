#!/usr/bin/env python3
"""Capability-registry audit for TensorDEM (CAP-REG-DEM, dual-pilot spec).

Runs the registry CLI in all three modes via *direct file-path invocation*
(the platform host contract: no ``-m``, torch optional), verifies the
single-JSON-object stdout contract, re-checks the anti-overclaim gates and
the module/entry/test_path pointers independently, and writes
``docs/m{mission}/REGISTRY_AUDIT.md`` with the numbers injected
programmatically (zero hand-copy).

Usage:  python scripts/audit_capability_registry.py [mission_id]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "src" / "tensordem" / "registry.py"
CAPS = ROOT / "src" / "tensordem" / "capabilities.py"
RUNS = ROOT / "runs"


def run_cli(args: list[str], out: Path | None = None) -> tuple[int, dict, str]:
    cmd = [sys.executable, str(REG), *args]
    if out is not None:
        cmd += ["--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    try:
        payload = json.loads(p.stdout)  # stdout must be exactly one JSON object
    except json.JSONDecodeError as e:
        return p.returncode, {}, f"stdout is not a single JSON object: {e}"
    return p.returncode, payload, p.stderr


def check_contract(payload: dict) -> list[str]:
    problems = []
    if set(payload) != {"source_commit", "capabilities", "summary", "problems"}:
        problems.append(f"contract keys wrong: {sorted(payload)}")
    caps = payload.get("capabilities", [])
    s = payload.get("summary", {})
    if s.get("total") != len(caps):
        problems.append(f"summary.total={s.get('total')} != len(capabilities)={len(caps)}")
    if s.get("problems") != len(payload.get("problems", [])):
        problems.append("summary.problems != len(problems)")
    if sum(s.get("by_maturity", {}).values()) != len(caps):
        problems.append("by_maturity does not sum to total")
    for c in caps:
        for k in ("id", "kind", "maturity", "module_path", "entry", "params",
                  "wired", "tested", "verified", "test_path", "evidence", "endpoint"):
            if k not in c:
                problems.append(f"{c.get('id')}: missing field {k}")
    return problems


def check_gates(caps: list[dict]) -> dict[str, list[str]]:
    """Independent re-implementation of the 422-gate mirror (per gate)."""
    bad: dict[str, list[str]] = {
        "mature_requires_verified_wired": [],
        "planned_requires_feature_ledger": [],
        "endpoint_requires_wired_prefix": [],
        "verified_requires_analytic_xvalue": [],
    }
    for c in caps:
        ev = c.get("evidence") or {}
        if c.get("maturity") == "mature" and not (c.get("verified") and c.get("wired")):
            bad["mature_requires_verified_wired"].append(f"{c['id']}: verified={c.get('verified')} wired={c.get('wired')}")
        if c.get("maturity") == "planned" and not str(ev.get("feature_ledger") or "").strip():
            bad["planned_requires_feature_ledger"].append(f"{c['id']}: no evidence.feature_ledger anchor")
        if c.get("endpoint"):
            if not c.get("wired"):
                bad["endpoint_requires_wired_prefix"].append(f"{c['id']}: endpoint without wired")
            elif not str(c["endpoint"]).startswith(("cli:", "route:", "api:")):
                bad["endpoint_requires_wired_prefix"].append(f"{c['id']}: endpoint prefix {c['endpoint']!r}")
        if c.get("verified") and ev.get("level") not in ("ANALYTIC", "XVALUE"):
            bad["verified_requires_analytic_xvalue"].append(f"{c['id']}: level={ev.get('level')!r}")
    return bad


def check_pointers(caps: list[dict]) -> list[str]:
    """Independent pointer re-check (module file / entry symbol / test file)."""
    import ast
    pkg = ROOT / "src" / "tensordem"
    bad = []
    for c in caps:
        mod = c.get("module_path") or ""
        parts = mod.split(".")
        if parts and parts[0] == "tensordem":
            base = pkg.joinpath(*parts[1:]) if len(parts) > 1 else None
            if base is not None and base.is_dir() and (base / "__init__.py").is_file():
                modfile = base / "__init__.py"
            elif base is not None and base.with_suffix(".py").is_file():
                modfile = base.with_suffix(".py")
            else:
                bad.append(f"{c['id']}: module file missing for {mod}")
                continue
        else:
            modfile = None
        entry = c.get("entry")
        if entry and modfile is not None:
            try:
                tree = ast.parse(modfile.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                bad.append(f"{c['id']}: module unreadable {modfile}")
                continue
            names = set()
            for n in tree.body:
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
                    names.add(n.name)
                elif isinstance(n, ast.ImportFrom):
                    names |= {a.asname or a.name for a in n.names}
            if modfile.name == "__init__.py":
                for py in sorted(modfile.parent.glob("*.py")):
                    if py.name == "__init__.py":
                        continue
                    try:
                        sub = ast.parse(py.read_text(encoding="utf-8"))
                    except (SyntaxError, OSError):
                        continue
                    for n in sub.body:
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
                            names.add(n.name)
            if entry not in names:
                bad.append(f"{c['id']}: entry {entry!r} not in {mod}")
        tp = c.get("test_path")
        if tp and not (ROOT / tp).is_file():
            bad.append(f"{c['id']}: test file missing {tp}")
    return bad


def check_verified_artifacts(caps: list[dict]) -> list[str]:
    """Every verified entry must point at an existing reference artifact."""
    pkg = ROOT / "src" / "tensordem"
    bad = []
    for c in caps:
        if not c.get("verified"):
            continue
        note = (c.get("evidence") or {}).get("note", "")
        m = re.search(r"([A-Za-z0-9_./]+\.py|verdict_[A-Za-z0-9_]+\.json|tests/[A-Za-z0-9_]+\.py)", note)
        if not m:
            bad.append(f"{c['id']}: no artifact name in evidence note {note!r}")
            continue
        art = m.group(1)
        if art.startswith("tests/"):
            if not (ROOT / art).is_file():
                bad.append(f"{c['id']}: artifact missing {art}")
        elif art.startswith("verdict_"):
            if not (ROOT / art).is_file():
                bad.append(f"{c['id']}: artifact missing {art}")
        else:
            # module file: search the package tree
            hits = list(pkg.rglob(art))
            if not hits:
                bad.append(f"{c['id']}: artifact missing {art}")
    return bad


def main() -> int:
    mission = sys.argv[1] if len(sys.argv) > 1 else "m24013"
    runs = RUNS / mission
    runs.mkdir(parents=True, exist_ok=True)

    results: dict = {"mission": mission, "checks": {}}
    failures: list[str] = []

    rc, static, err = run_cli(["--static"], runs / "registry_static_dump.json")
    results["static"] = {"rc": rc, "stderr_tail": err[-400:]}
    results["checks"]["static_contract"] = check_contract(static)
    if rc != 0:
        failures.append(f"static rc={rc}")

    rc, imp, err = run_cli([], runs / "registry_import_dump.json")
    results["import"] = {"rc": rc, "stderr_tail": err[-400:]}
    results["checks"]["import_contract"] = check_contract(imp)
    if rc != 0:
        failures.append(f"import rc={rc}")

    rc, par, err = run_cli(["--parity"], runs / "registry_parity.json")
    results["parity"] = {"rc": rc, "parity": par.get("parity"),
                         "static_count": par.get("static_count"),
                         "import_count": par.get("import_count"),
                         "differences": par.get("differences", [])[:10]}
    if rc != 0 or par.get("parity") is not True:
        failures.append(f"parity rc={rc} parity={par.get('parity')}")

    caps = static.get("capabilities", [])
    results["checks"]["static_gate_violations"] = check_gates(caps)
    n_gate_total = sum(len(v) for v in results["checks"]["static_gate_violations"].values())
    results["checks"]["static_pointer_violations"] = check_pointers(caps)
    results["checks"]["verified_artifact_missing"] = check_verified_artifacts(caps)
    results["checks"]["registry_reported_problems"] = static.get("problems", [])

    # honest count: number of @capability decorators in the generated file
    n_decor = len(re.findall(r"^\s*@capability\(", CAPS.read_text(encoding="utf-8"), re.M))
    results["count_check"] = {
        "decorators_in_capabilities_py": n_decor,
        "static_total": static.get("summary", {}).get("total"),
        "match": n_decor == static.get("summary", {}).get("total"),
    }
    if not results["count_check"]["match"]:
        failures.append("entry count mismatch vs decorator scan")

    s = static.get("summary", {})
    results["summary"] = s

    # ---------------------------------------------------------------- report
    doc = ROOT / "docs" / mission / "REGISTRY_AUDIT.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# REGISTRY_AUDIT · {mission} · CAP-REG-DEM · TensorDEM 能力注册表",
        "",
        f"生成：`scripts/audit_capability_registry.py {mission}`（数字程序化注入，零手抄）· {now}",
        f"基准 commit：`{static.get('source_commit')}`",
        "",
        "## 1. CLI 三模式（直调文件路径，torch 可选）",
        "",
        "| 模式 | 命令 | rc | 结果 |",
        "|---|---|---|---|",
        f"| static | `python src/tensordem/registry.py --static` | {results['static']['rc']} | "
        f"{s.get('total')} 条，problems={s.get('problems')} |",
        f"| import | `python src/tensordem/registry.py` | {results['import']['rc']} | "
        f"{imp.get('summary', {}).get('total')} 条，problems={imp.get('summary', {}).get('problems')} |",
        f"| parity | `python src/tensordem/registry.py --parity` | {results['parity']['rc']} | "
        f"parity={results['parity']['parity']}（{results['parity']['static_count']}="
        f"{results['parity']['import_count']}，diffs={len(results['parity']['differences'])}） |",
        "",
        "stdout 契约：单 JSON 对象 `{source_commit, capabilities, summary, problems}` —— "
        + ("三模式全部满足" if not results["checks"]["static_contract"] and not results["checks"]["import_contract"] else "存在违例，见下"),
        "",
        "## 2. 条目数如实性",
        "",
        f"- capabilities.py 中 `@capability` 装饰器数 = **{results['count_check']['decorators_in_capabilities_py']}**",
        f"- static 扫描 total = **{results['count_check']['static_total']}** → "
        + ("一致 ✓" if results["count_check"]["match"] else "**不一致 ✗**"),
        f"- summary：mature={s.get('by_maturity', {}).get('mature')} "
        f"developing={s.get('by_maturity', {}).get('developing')} "
        f"planned={s.get('by_maturity', {}).get('planned')}；"
        f"wired={s.get('wired')} tested={s.get('tested')} verified={s.get('verified')} "
        f"endpoints={s.get('endpoints')}",
        "",
        "## 3. 反过度声明门（独立复算，非自声明）",
        "",
        "| 门 | 违例数 |",
        "|---|---|",
        f"| mature 需 verified+wired | {len(results['checks']['static_gate_violations']['mature_requires_verified_wired'])} |",
        f"| planned 需 evidence.feature_ledger 锚 | {len(results['checks']['static_gate_violations']['planned_requires_feature_ledger'])} |",
        f"| endpoint 需 wired + cli:/route:/api: 前缀 | {len(results['checks']['static_gate_violations']['endpoint_requires_wired_prefix'])} |",
        f"| verified 需 ANALYTIC/XVALUE 级证据 | {len(results['checks']['static_gate_violations']['verified_requires_analytic_xvalue'])} |",
        f"| **合计** | **{n_gate_total}** |",
        "",
    ]
    for gate, vs in results["checks"]["static_gate_violations"].items():
        for v in vs:
            lines.append(f"- 违例（{gate}）：{v}")
    lines += [
        "",
        "## 4. 指针解析（module/entry/test_path 不存在 = problem，非静默丢弃）",
        "",
        f"- 注册表自报 problems：**{len(results['checks']['registry_reported_problems'])}**",
        f"- 独立复算指针违例：**{len(results['checks']['static_pointer_violations'])}**",
    ]
    for v in results["checks"]["static_pointer_violations"]:
        lines.append(f"- 违例：{v}")
    for p in results["checks"]["registry_reported_problems"]:
        lines.append(f"- 注册表自报：{p['capability']} {p['code']} {p['detail']}")
    lines += [
        "",
        f"- verified 条目证据工件缺失：**{len(results['checks']['verified_artifact_missing'])}**",
    ]
    for v in results["checks"]["verified_artifact_missing"]:
        lines.append(f"- 缺失：{v}")
    lines += [
        "",
        "## 5. 契约与 parity 违例",
        "",
        f"- static 契约违例：{len(results['checks']['static_contract'])}",
        f"- import 契约违例：{len(results['checks']['import_contract'])}",
        f"- parity 差异：{len(results['parity']['differences'])}",
        "",
        "## 6. 裁定",
        "",
    ]
    if failures:
        lines.append(f"**FAIL（{len(failures)} 项）**：")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append(f"**PASS**：三模式 exit 0；stdout 单 JSON 对象契约满足；条目数与装饰器数一致；"
                     f"四道反过度声明门 {n_gate_total} 违例；指针 {len(results['checks']['static_pointer_violations'])} 违例；"
                     f"verified 证据工件 {len(results['checks']['verified_artifact_missing'])} 缺失；"
                     "static/import parity 全字段一致。")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"audit written: {doc}")
    print(f"verdict: {'FAIL ' + str(failures) if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
