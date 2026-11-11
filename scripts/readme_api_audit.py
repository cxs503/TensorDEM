"""README / 示例 与实际代码 API 的一致性巡检（可复现、程序化、零手抄）。

审计六面，每条都是可机检断言，不靠人工判断：

A. 本地链接：README 里所有相对链接（docs/ scripts/ tests/ schemas/ artifacts/
   examples/）在仓库中真实存在。
B. 模块调用：``python -m <mod>`` 可导入；``python scripts/<f>.py`` 文件存在。
C. CLI 旗标（**逐命令归属**）：把 README 代码块按行解析成命令（续行先拼接），
   每个 ``--flag`` 只与**该命令**实际接受的旗标比对；旗标集合由 ``--help``
   实跑解析（比静态 AST 更硬：覆盖 argparse 子命令与别名）。
D. Python API：``from tensordem[...] import X`` 与 ``tensordem.X`` 中，X 要么是
   真实属性，要么是可导入子模块（README 用 ``tensordem.lbmdem`` 指模块路径）。
E. 入口点：pyproject ``[project.scripts]`` 目标可导入，且 README 复述的命令存在。
F. 默认值断言：README 复述的数值默认值（如「默认生成 21 × 5 个颗粒」）与
   argparse 默认值一致。
G. 示例可运行性（``--run-examples``）：``examples/**/*.py`` 退出码为 0。

环境断言：本仓存在多份 worktree 检出，site-packages 的 editable 安装可能指向
别的检出目录。审计先把 ``ROOT/src`` 置顶并断言导入路径在本工作树内，否则结论作废。

用法::

    python scripts/readme_api_audit.py
    python scripts/readme_api_audit.py --run-examples
    python scripts/readme_api_audit.py --out artifacts/readme_api_audit/report.json --md
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
sys.path.insert(0, str(ROOT / "src"))
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

LOCAL_PREFIXES = ("docs/", "scripts/", "tests/", "schemas/", "artifacts/", "examples/", "src/")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
CODE_BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
PY_IMPORT = re.compile(r"from\s+tensordem(?:\.([\w.]+))?\s+import\s+([^\n#]+)")
# 一条 README 命令：(kind, target)
CMD = re.compile(
    r"(?:python\s+-m\s+(?P<mod>[\w.]+)"
    r"|python\s+(?P<script>(?:scripts|examples)/[\w./-]+\.py)"
    r"|(?P<console>tensordem(?:-3d)?))\b"
)

# console 脚本 -> 等价的 -m 调用（用于取 --help 旗标）
CONSOLE_MODULE = {
    "tensordem": "tensordem",
    "tensordem-3d": "tensordem.dem3d_cli",
}


@dataclass
class Finding:
    kind: str
    claim: str
    location: str
    status: str          # OK / MISMATCH
    detail: str = ""


WAIVER = re.compile(r"<!--\s*audit-allow:\s*([\w-]+)\s*:\s*([^>]*?)-->")


@dataclass
class Audit:
    checks: list[Finding] = field(default_factory=list)
    waivers: dict[str, str] = field(default_factory=dict)

    def add(self, **kw) -> None:
        if kw.get("status") != "OK" and kw.get("kind") in self.waivers:
            kw["detail"] = f"[已声明豁免: {self.waivers[kw['kind']]}] " + kw.get("detail", "")
            kw["status"] = "OK"
        self.checks.append(Finding(**kw))

    def failures(self) -> list[Finding]:
        return [c for c in self.checks if c.status != "OK"]

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        by_kind: dict[str, dict[str, int]] = {}
        for c in self.checks:
            by_status[c.status] = by_status.get(c.status, 0) + 1
            d = by_kind.setdefault(c.kind, {})
            d[c.status] = d.get(c.status, 0) + 1
        return {"total": len(self.checks), "by_status": by_status, "by_kind": by_kind}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def line_of(text: str, needle: str, start: int = 0) -> str:
    idx = text.find(needle, start)
    return "-" if idx < 0 else f"README.md:{text.count(chr(10), 0, idx) + 1}"


def code_blocks(text: str):
    for m in CODE_BLOCK.finditer(text):
        yield m.group(1), m.group(2), text.count("\n", 0, m.start()) + 1


_HELP_CACHE: dict[str, tuple[set[str], str]] = {}


def help_flags(kind: str, target: str, timeout: int = 600) -> tuple[set[str], str]:
    """实跑 `--help` 取该命令接受的旗标集合。"""
    key = f"{kind}:{target}"
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    if kind == "mod":
        cmd = [sys.executable, "-m", target]
    elif kind == "script":
        cmd = [sys.executable, target]
    else:
        cmd = [sys.executable, "-m", CONSOLE_MODULE[target]]
    try:
        proc = subprocess.run(cmd + ["--help"], cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout, env=ENV)
        if proc.returncode != 0:
            err = (proc.stderr.strip().splitlines() or ["?"])[-1]
            res = (set(), f"--help rc={proc.returncode}: {err[:200]}")
        else:
            res = (set(FLAG.findall(proc.stdout)), "")
    except subprocess.TimeoutExpired:
        res = (set(), "--help 超时")
    _HELP_CACHE[key] = res
    return res


def warm_help_cache(text: str, workers: int = 8) -> None:
    """并行预热 `--help` 缓存（每条命令一次子进程；串行时墙钟被 torch import 主导）。"""
    from concurrent.futures import ThreadPoolExecutor

    flat = text.replace("\\\n", " ")
    jobs: list[tuple[str, str]] = []
    for m in CMD.finditer(flat):
        if m.group("mod"):
            jobs.append(("mod", m.group("mod")))
        elif m.group("script"):
            jobs.append(("script", m.group("script")))
        else:
            jobs.append(("console", m.group("console")))
    unique = list(dict.fromkeys(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda j: help_flags(*j), unique))


def audit_environment(audit: Audit) -> None:
    import tensordem
    resolved = Path(tensordem.__file__).resolve()
    inside = ROOT in resolved.parents
    audit.add(kind="environment", claim="import tensordem 来自本工作树",
              location="runtime", status="OK" if inside else "MISMATCH",
              detail=str(resolved) + ("" if inside else f" 不在 {ROOT} 内（多检出串台，审计作废）"))


def audit_links(audit: Audit, text: str) -> None:
    seen = set()
    for target in LINK.findall(text):
        if target in seen or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        seen.add(target)
        local = target.split("#", 1)[0]
        if not local.startswith(LOCAL_PREFIXES):
            continue
        exists = (ROOT / local).exists()
        audit.add(kind="link", claim=target, location=line_of(text, f"({target})"),
                  status="OK" if exists else "MISMATCH",
                  detail="" if exists else f"路径不存在: {local}")


def audit_modules_and_flags(audit: Audit, text: str) -> None:
    """逐命令解析 README 代码块：模块可导入 + 旗标归属正确。"""
    seen_cmds: set[tuple[str, str]] = set()
    flat = text.replace("\\\n", " ")          # 续行拼接
    # 代码块 + 行内反引号命令（README 有大量 `python -m x --flag` 内联写法）
    units: list[tuple[str, int]] = []
    for _, body, offset in code_blocks(flat):
        units.extend((line, offset + i) for i, line in enumerate(body.splitlines()))
    for m in re.finditer(r"`([^`\n]*python[^`\n]*)`", text):
        units.append((m.group(1), text.count(chr(10), 0, m.start()) + 1))
    if True:
        for line, ordinal in units:
            m = CMD.search(line)
            if not m:
                continue
            if m.group("mod"):
                kind, target = "mod", m.group("mod")
            elif m.group("script"):
                kind, target = "script", m.group("script")
            else:
                kind, target = "console", m.group("console")
            claim_cmd = line.strip()
            loc = f"README.md:~{ordinal}"
            # B. 目标存在 / 可导入
            if kind == "script" and not (ROOT / target).exists():
                audit.add(kind="script", claim=target, location=loc, status="MISMATCH",
                          detail="脚本文件不存在")
                continue
            if kind == "mod":
                try:
                    importlib.import_module(target)
                except Exception as exc:
                    audit.add(kind="module", claim=f"python -m {target}", location=loc,
                              status="MISMATCH", detail=f"{type(exc).__name__}: {exc}"[:200])
                    continue
                if (kind, target) not in seen_cmds:
                    seen_cmds.add((kind, target))
                    audit.add(kind="module", claim=f"python -m {target}", location=loc, status="OK")
            # C. 旗标
            flags, note = help_flags(kind, target)
            if note:
                audit.add(kind="flag", claim=claim_cmd[:90], location=loc, status="MISMATCH",
                          detail=f"无法取得旗标清单: {note}")
                continue
            for flag in sorted(set(FLAG.findall(line))):
                ok = flag in flags
                audit.add(kind="flag", claim=flag, location=loc, status="OK" if ok else "MISMATCH",
                          detail=f"命令: {claim_cmd[:70]}" if ok else
                                 f"{claim_cmd[:70]} 不接受 {flag}")


def audit_api(audit: Audit, text: str) -> None:
    import tensordem

    seen = set()

    def check(sym: str, where: str, loc: str) -> None:
        key = (where, sym)
        if key in seen:
            return
        seen.add(key)
        if where == "tensordem" and hasattr(tensordem, sym):
            ok, detail = True, "属性"
        else:
            try:
                importlib.import_module(f"{where}.{sym}")   # 例如 tensordem.ice_abrasion
                ok, detail = True, "子模块"
            except Exception:
                ok, detail = False, f"{where} 既无属性也无此子模块"
        audit.add(kind="api", claim=f"{where}:{sym}", location=loc,
                  status="OK" if ok else "MISMATCH", detail=detail if ok else detail)

    for mod, names in PY_IMPORT.findall(text):
        where = f"tensordem.{mod}" if mod else "tensordem"
        for raw in names.replace("(", " ").replace(")", " ").split(","):
            sym = raw.strip().split(" as ")[0].strip()
            if sym:
                check(sym, where, line_of(text, f"import {names.strip()[:40]}"))
    for sym in sorted(set(re.findall(r"\btensordem\.([A-Za-z_]\w*)", text))):
        if sym.startswith(("py",)):   # tensordem.py 之类的文件名词元
            continue
        check(sym, "tensordem", line_of(text, f"tensordem.{sym}"))


def audit_entrypoints(audit: Audit, text: str) -> None:
    data = tomllib.loads(read_text(ROOT / "pyproject.toml"))
    for name, target in sorted(data.get("project", {}).get("scripts", {}).items()):
        mod = target.split(":")[0]
        try:
            importlib.import_module(mod)
            audit.add(kind="entrypoint", claim=f"{name} = {target}", location="pyproject.toml",
                      status="OK")
        except Exception as exc:
            audit.add(kind="entrypoint", claim=f"{name} = {target}", location="pyproject.toml",
                      status="MISMATCH", detail=f"{type(exc).__name__}: {exc}"[:200])
        if name not in text:
            audit.add(kind="entrypoint", claim=name, location="README.md", status="OK",
                      detail="README 未提及（不算不一致，仅记录）")


def console_script_module(name: str) -> str:
    data = tomllib.loads(read_text(ROOT / "pyproject.toml"))
    return data["project"]["scripts"][name].split(":")[0]


def module_source_files(module: str) -> list[Path]:
    rel = ROOT / "src" / Path(*module.split("."))
    return [rel.with_suffix(".py")] if rel.with_suffix(".py").exists() else sorted(rel.rglob("*.py"))


def source_defaults(flags: dict[str, str],
                    files: list[Path] | None = None) -> dict[str, list[tuple[str, str, str, str]]]:
    """从源码 AST 抽取每个旗标的显式 default（--help 不显示默认值，只能读源码）。

    全仓扫描 ``add_argument`` 调用；同一旗标在不同 CLI 脚本里可能有不同默认值，
    因此返回 flag -> [(默认值, 文件, 函数作用域, 其他参数摘要)]，多值即歧义。
    """
    import ast

    out: dict[str, list[tuple[str, str, str, str]]] = {}
    for path in (files or sorted((ROOT / "src" / "tensordem").rglob("*.py"))):
        try:
            tree = ast.parse(read_text(path))
        except SyntaxError:
            continue
        rel = str(path.relative_to(ROOT))
        for fn in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            scope = getattr(fn, "name", "<module>")
            for node in ast.walk(fn) if fn is not tree else tree.body:
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name != "add_argument":
                    continue
                names = [a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("--")]
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                default = "<unset>"
                if "default" in kw:
                    try:
                        default = repr(ast.literal_eval(kw["default"]))
                    except Exception:
                        default = "<expr>"
                extra = ", ".join(sorted(k for k in kw if k in {"type", "required", "action"}))
                for flag in names:
                    if flag in flags or not flags:
                        out.setdefault(flag, []).append((default, rel, scope, extra))
    return out


def audit_defaults(audit: Audit, text: str) -> None:
    """README 复述的数值默认值必须与源码 argparse 默认值一致。"""
    defaults = source_defaults({}, module_source_files(console_script_module("tensordem")))
    m = re.search(r"默认生成\s*(\d+)\s*×\s*(\d+)\s*个颗粒", text)
    if not m:
        return
    claimed = (int(m.group(1)), int(m.group(2)))
    got = []
    for flag in ("--nx", "--ny"):
        entries = defaults.get(flag, [])
        vals = {e[0] for e in entries}
        got.append(entries)
    def as_int(entries):
        vals = {e[0] for e in entries}
        if len(vals) == 1:
            try:
                return int(float(vals.pop().strip("'")))
            except ValueError:
                return None
        return None
    actual = (as_int(got[0]), as_int(got[1]))
    audit.add(kind="default", claim=f"默认网格 {claimed[0]} × {claimed[1]}",
              location=line_of(text, m.group(0)),
              status="OK" if actual == claimed else "MISMATCH",
              detail=f"源码默认 = {actual[0]} × {actual[1]}"
                     + "" if actual == claimed else
                     f" (证据: {[e[:3] for e in got[0]]} {[e[:3] for e in got[1]]})")



BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_./{}-]+\.(?:md|py|json|csv|pt)|[A-Za-z0-9_./-]+/)`")
SIG_CLAIM = re.compile(r"\b([A-Z]\w+)\.(\w+)\(([^)]*)\)")
CTOR_CLAIM = re.compile(r"\b([A-Z]\w+)Config\(([^)]*)\)")


def expand_braces(target: str) -> list[str]:
    """`dir/{a.json,b.json}` -> 两个具体路径。"""
    m = re.match(r"^(.*)\{(.*)\}(.*)$", target)
    if not m:
        return [target]
    head, body, tail = m.groups()
    return [head + part.strip() + tail for part in body.split(",")]


def audit_backtick_paths(audit: Audit, text: str) -> None:
    """README 反引号里点名的仓库内文件（非 markdown 链接）也必须真实存在。"""
    seen = set()
    for target in BACKTICK_PATH.findall(text):
        if target in seen or target.startswith(("http", "results/", "s3://", "file://")):
            continue
        seen.add(target)
        if not target.startswith(LOCAL_PREFIXES + ("benchmarks/", "src/")):
            continue
        for base in expand_braces(target):
            exists = (ROOT / base).exists()
            detail = ""
            status = "OK"
            if not exists:
                cleaned = base.rstrip("/")
                candidates = [q for d in ("scripts", "src") for q in (ROOT / d).rglob("*.py")]
                if base.endswith("/"):        # 目录：要么写全路径，要么写 OUT = Path("...<目录>")
                    producers = [str(q.relative_to(ROOT)) for q in candidates
                                 if (cleaned + "/" in read_text(q)) or (cleaned + '"' in read_text(q))]
                else:                          # 文件：目录与文件名必须出现在同一脚本
                    parent = str(Path(cleaned).parent)
                    leaf = Path(cleaned).name
                    producers = [str(q.relative_to(ROOT)) for q in candidates
                                 if parent in read_text(q) and leaf in read_text(q)]
                if producers:
                    detail = f"生成物未入库；生产者: {producers[:2]}"
                else:
                    status, detail = "MISMATCH", f"仓库内不存在且无脚本产出: {base}"
            audit.add(kind="path", claim=base, location=line_of(text, f"`{target}`"),
                      status=status, detail=detail)


def audit_signatures(audit: Audit, text: str) -> None:
    """README 里点名的调用签名（类.方法(kw=...)）必须被真实签名接受。"""
    import inspect

    import tensordem

    seen = set()
    for cls_name, meth, args in SIG_CLAIM.findall(text):
        cls = getattr(tensordem, cls_name, None)
        if cls is None or not hasattr(cls, meth):
            continue
        key = (cls_name, meth)
        if key in seen:
            continue
        seen.add(key)
        try:
            sig = inspect.signature(getattr(cls, meth))
        except (TypeError, ValueError):
            continue
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for kw in re.findall(r"(\w+)\s*=", args):
            ok = accepts_kwargs or kw in sig.parameters
            audit.add(kind="signature", claim=f"{cls_name}.{meth}({kw}=...)",
                      location=line_of(text, f"{cls_name}.{meth}"),
                      status="OK" if ok else "MISMATCH",
                      detail=f"签名 = {sig}" if ok else f"签名不接受 {kw}; 实际 = {sig}")


def audit_config_fields(audit: Audit, text: str) -> None:
    """`DEM3DConfig(nx=..., ny=..., nz=...)` 之类的构造参数必须是真实字段。"""
    import dataclasses

    import tensordem

    seen = set()
    for name in sorted(set(re.findall(r"\b([A-Z]\w*Config)\(", text))):
        cls = getattr(tensordem, name, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        fields = {f.name for f in dataclasses.fields(cls)}
        for call in re.finditer(rf"{name}\(([^)]*)\)", text):
            args = call.group(1)
            for kw in re.findall(r"(\w+)\s*=", args):
                key = (name, kw)
                if key in seen:
                    continue
                seen.add(key)
                audit.add(kind="config", claim=f"{name}({kw}=...)",
                          location=line_of(text, call.group(0)),
                          status="OK" if kw in fields else "MISMATCH",
                          detail="" if kw in fields else
                                 f"{name} 无字段 {kw}；共有 {sorted(fields)[:8]}...")


JSON_CLAIM = re.compile(r"`?([\w./-]+\.json)`?\.([\w.]+)\s*=\s*(true|false|-?\d+(?:\.\d+)?)")


def audit_json_claims(audit: Audit, text: str) -> None:
    """README 用 `file.json.key = value` 自陈的机器可读判据，必须与文件实际内容一致。"""
    for fname, key, claimed in JSON_CLAIM.findall(text):
        # 找到这个claim在文本中的位置
        claim_pattern = f"{fname}.{key}"
        claim_idx = text.find(claim_pattern)
        if claim_idx < 0:
            claim_idx = 0
        
        # 如果 fname 包含路径分隔符，优先使用完整路径
        if "/" in fname:
            path = ROOT / fname
            if not path.exists():
                audit.add(kind="json-claim", claim=f"{fname}:{key}={claimed}",
                          location=line_of(text, fname), status="MISMATCH",
                          detail=f"引用的 json 文件不存在: {fname}")
                continue
        else:
            # 否则先向前搜索完整路径mention
            # 查找最近的对这个文件的完整路径引用
            context_start = max(0, claim_idx - 1000)
            context = text[context_start:claim_idx]
            
            # 查找形如 benchmarks/.../fname 或 artifacts/.../fname 的路径
            # 也处理 {file1.json,file2.json} 这种格式
            path_pattern = re.compile(rf"(benchmarks/[\w./-]*/{re.escape(fname)}|artifacts/[\w./-]*/{re.escape(fname)})")
            path_matches = list(path_pattern.finditer(context))
            
            if not path_matches:
                # 尝试匹配目录路径（如 benchmarks/.../ 后面跟着 {xxx,fname}）
                dir_pattern = re.compile(rf"(benchmarks/[\w./-]+/|artifacts/[\w./-]+/)")
                dir_matches = list(dir_pattern.finditer(context))
                if dir_matches:
                    # 使用最近的目录
                    dir_path = dir_matches[-1].group(1)
                    full_path = dir_path + fname
                    path = ROOT / full_path
                    if path.exists():
                        path_matches = [full_path]
            
            if path_matches:
                # 使用最近的一个完整路径
                if isinstance(path_matches[0], str):
                    full_path = path_matches[0]
                else:
                    full_path = path_matches[-1].group(1)
                path = ROOT / full_path
                if not path.exists():
                    audit.add(kind="json-claim", claim=f"{fname}:{key}={claimed}",
                              location=line_of(text, fname), status="MISMATCH",
                              detail=f"引用的 json 文件不存在: {full_path}")
                    continue
            else:
                # 按文件名搜索
                hits = list((ROOT / "benchmarks").rglob(Path(fname).name)) or \
                       list((ROOT / "artifacts").rglob(Path(fname).name))
                if not hits:
                    audit.add(kind="json-claim", claim=f"{fname}:{key}={claimed}",
                              location=line_of(text, fname), status="MISMATCH",
                              detail="引用的 json 文件在 benchmarks/ 与 artifacts/ 下均未找到")
                    continue
                path = hits[0]
        try:
            node = json.loads(read_text(path))
        except Exception as exc:
            audit.add(kind="json-claim", claim=f"{fname}:{key}", location=line_of(text, fname),
                      status="MISMATCH", detail=f"解析失败: {exc}")
            continue
        for part in key.split("."):
            node = node[part] if isinstance(node, dict) and part in node else None
        actual = json.dumps(node).lower() if node is not None else "null"
        want = claimed.lower()
        ok = actual == want or (want in {"true", "false"} and actual == want)
        audit.add(kind="json-claim", claim=f"{Path(fname).name}:{key}={claimed}",
                  location=line_of(text, fname), status="OK" if ok else "MISMATCH",
                  detail=f"文件 {path.relative_to(ROOT)} 实际 = {actual}")


def audit_torch_load(audit: Audit, text: str, run: bool) -> None:
    """README: `torch.load("results/trajectory.pt", weights_only=True)` 能读 -> 实测。"""
    import torch

    claims = {
        "results/trajectory.pt": None,
        "results-3d/trajectory_3d.pt": None,
    }
    if "weights_only=True" not in text or not run:
        audit.add(kind="torch-load", claim="torch.load(..., weights_only=True)",
                  location=line_of(text, "weights_only=True"), status="OK",
                  detail="未实测（需 --run-smoke 先产出轨迹）" if run is False else "")
        return
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="tdem_audit_"))
    jobs = [
        (["-m", "tensordem", "--nx", "5", "--ny", "3", "--steps", "10",
          "--save-every", "5", "--output", str(tmp / "2d")], "trajectory.pt"),
        (["-m", "tensordem.dem3d_cli", "--nx", "4", "--ny", "3", "--nz", "2", "--steps", "10",
          "--save-every", "5", "--output", str(tmp / "3d")], "trajectory_3d.pt"),
    ]
    for argv, fname in jobs:
        subprocess.run([sys.executable] + argv, cwd=ROOT, capture_output=True, text=True,
                       env=ENV, timeout=900)
        path = tmp / ("2d" if fname == "trajectory.pt" else "3d") / fname
        if not path.exists():
            audit.add(kind="torch-load", claim=f"torch.load({fname}, weights_only=True)",
                      location=line_of(text, "weights_only=True"), status="MISMATCH",
                      detail=f"CLI 未产出 {fname}")
            continue
        try:
            obj = torch.load(path, weights_only=True, map_location="cpu")
            keys = sorted(obj)[:6] if isinstance(obj, dict) else type(obj).__name__
            audit.add(kind="torch-load", claim=f"torch.load({fname}, weights_only=True)",
                      location=line_of(text, "weights_only=True"), status="OK",
                      detail=f"keys={keys}")
        except Exception as exc:
            audit.add(kind="torch-load", claim=f"torch.load({fname}, weights_only=True)",
                      location=line_of(text, "weights_only=True"), status="MISMATCH",
                      detail=f"{type(exc).__name__}: {str(exc)[:150]}")


def audit_documented_tests(audit: Audit, text: str) -> None:
    """README 声称「由 tests/... 执行/覆盖」的测试文件必须存在。"""
    for m in re.finditer(r"[`(]((?:tests|scripts)/[\w./-]+\.py)[`)]", text):
        target = m.group(1)
        exists = (ROOT / target).exists()
        audit.add(kind="doc-test", claim=target, location=line_of(text, target),
                  status="OK" if exists else "MISMATCH",
                  detail="" if exists else "README 声称可运行的测试/脚本不存在")



def _test_style(root: Path) -> dict[str, int]:
    """统计 tests/ 下 pytest 风格函数与 unittest TestCase 类的数量。"""
    import ast

    funcs = cases = 0
    for f in sorted((root / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(read_text(f))
        except SyntaxError:
            continue
        funcs += sum(1 for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith("test_"))
        cases += sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                     and any(getattr(b, "attr", getattr(b, "id", "")) == "TestCase"
                             for b in n.bases))
    return {"pytest_functions": funcs, "unittest_cases": cases}


def audit_test_runner(audit: Audit, text: str, root: Path | None = None,
                      style: dict[str, int] | None = None) -> None:
    """README 记录的测试命令必须能跑本仓的测试风格。

    套件以模块级 ``test_*`` 函数为主，``python -m unittest discover`` 只认
    ``unittest.TestCase`` 子类，会静默跳过其余测试并照样打印 ``OK``（假绿）。
    """
    root = root or ROOT
    style = style or _test_style(root)
    for m in re.finditer(r"python -m unittest discover[^\n`]*", text):
        if style["pytest_functions"] > style["unittest_cases"]:
            audit.add(kind="test-runner", claim=m.group(0).strip()[:80],
                      location=line_of(text, m.group(0)),
                      status="MISMATCH",
                      detail=(f"套件有 {style['pytest_functions']} 个 pytest 风格测试函数、"
                              f"{style['unittest_cases']} 个 TestCase；unittest discover 会静默"
                              "跳过前者并打印 OK（假绿）"))
        else:
            audit.add(kind="test-runner", claim=m.group(0).strip()[:80],
                      location=line_of(text, m.group(0)), status="OK")
    for m in re.finditer(r"python -m pytest[^\n`]*", text):
        audit.add(kind="test-runner", claim=m.group(0).strip()[:80],
                  location=line_of(text, m.group(0)), status="OK")


ARTIFACT_ASSERT = re.compile(r"artifacts/[\w./{}-]+")


def audit_artifact_gate_convention(audit: Audit, files: list[Path] | None = None) -> None:
    """硬断言 ``artifacts/...`` 存在、却无 skip/copy 保护的测试违背本仓惯例。

    生成物不入 VCS，干净检出里这类断言会让 ``pytest -q`` 固定红；本仓其它门
    （test_ice_ridge_grounding_rederivation 等）都用 ``skipif(not ARTIFACT.exists())``。
    """
    files = files if files is not None else sorted((ROOT / "tests").glob("test_*.py"))
    for f in files:
        text = read_text(f)
        if "artifacts/" not in text:
            continue
        has_guard = ("skipif" in text) or ("pytest.skip" in text) or ("skip(" in text)
        hard = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("assert") or ".exists()" not in stripped:
                continue
            if ARTIFACT_ASSERT.search(stripped) or "ARTIFACT" in stripped or "SUMMARY" in stripped:
                hard.append(stripped[:90])
        rel = str(f.relative_to(ROOT)) if ROOT in f.resolve().parents else f.name
        audit.add(kind="artifact-gate", claim=rel,
                  location=rel, status="OK" if (has_guard or not hard) else "MISMATCH",
                  detail=(f"{len(hard)} 条硬断言生成物存在且模块无 skip 保护: {hard[:2]}"
                          if (hard and not has_guard) else
                          ("有 skip/copy 保护" if has_guard else "无 artifacts 硬断言")))


def audit_examples(audit: Audit, run: bool, timeout: int) -> None:
    for path in sorted((ROOT / "examples").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if not run:
            audit.add(kind="example", claim=rel, location="examples/", status="OK",
                      detail="存在（未执行；--run-examples 执行）")
            continue
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, rel], cwd=ROOT, capture_output=True,
                                  text=True, timeout=timeout, env=ENV)
            rc, err = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired:
            rc, err = "timeout", ""
        dt = time.time() - t0
        audit.add(kind="example", claim=rel, location="examples/",
                  status="OK" if rc == 0 else "MISMATCH",
                  detail=f"rc={rc} {dt:.1f}s" + ("" if rc == 0 else
                                                 " | " + (err.strip().splitlines() or ["?"])[-1][:240]))


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = ["# README ↔ code API 一致性巡检（自动生成，勿手改）", "",
             f"- 检查项: {s['total']}；结果: {s['by_status']}", "",
             "| # | 类别 | 声明 | 位置 | 结果 | 说明 |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(payload["checks"], 1):
        detail = (c["detail"] or "").replace("|", "\\|")[:120]
        lines.append(f"| {i} | {c['kind']} | `{c['claim'][:80].replace('|', '/')}` | "
                     f"{c['location']} | {c['status']} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="README/API 一致性巡检")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "readme_api_audit" / "report.json")
    ap.add_argument("--md", action="store_true", help="同时写 markdown 表")
    ap.add_argument("--run-examples", action="store_true")
    ap.add_argument("--run-smoke", action="store_true",
                    help="实跑微型 CLI 断言 README 声称的产物/torch.load 可读性")
    ap.add_argument("--example-timeout", type=int, default=900)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    text = read_text(README)
    audit = Audit()
    audit.waivers = {k: v.strip() for k, v in WAIVER.findall(text)}
    warm_help_cache(text)
    audit_environment(audit)
    audit_links(audit, text)
    audit_modules_and_flags(audit, text)
    audit_api(audit, text)
    audit_entrypoints(audit, text)
    audit_defaults(audit, text)
    audit_backtick_paths(audit, text)
    audit_signatures(audit, text)
    audit_config_fields(audit, text)
    audit_documented_tests(audit, text)
    audit_json_claims(audit, text)
    audit_test_runner(audit, text)
    audit_artifact_gate_convention(audit)
    audit_torch_load(audit, text, args.run_smoke)
    audit_examples(audit, args.run_examples, args.example_timeout)

    payload = {"summary": audit.summary(), "waivers": audit.waivers,
               "failures": [asdict(f) for f in audit.failures()],
               "checks": [asdict(c) for c in audit.checks]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md:
        args.out.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    if not args.quiet:
        s = payload["summary"]
        print(f"[readme-api-audit] checks={s['total']} status={s['by_status']} -> {args.out}")
        for f in payload["failures"]:
            print(f"  FAIL {f['kind']:<12} {f['claim'][:56]:<58} {f['location']:<16} {f['detail'][:90]}")
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())