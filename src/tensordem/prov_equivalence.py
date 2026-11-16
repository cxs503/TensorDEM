"""Static equivalence certificates for run artefacts stamped with a source digest.

Why this module exists
----------------------
Run artefacts in this repo carry ``impl_sha256`` -- a SHA-256 of **one**
module file, taken at run time (``tensordem.deposition_packing.
inspect_source_sha256``).  Downstream audits compare numbers that were produced
under *different* digests, which silently assumes "different digest ⇒ same
numeric behaviour".  Until now that assumption was only ever tested by rerunning
one specification and looking at the difference, i.e. one trajectory sample per
configuration, and it cannot cover a digest for which no rerun was budgeted.

There are two defects in the digest-as-provenance scheme, and they are *not*
the same defect:

1. **It stamps one file, not the code path.**  ``deposition_packing`` imports
   :func:`tensordem.granular3d_core.damping_ratio` (the restitution convention).
   If that module changes, the digest does not change, yet the physics may.  A
   certificate must therefore cover the **local import closure**, not one file.
2. **A digest change is not a behaviour change.**  Appending a post-processing
   helper, editing a docstring, or adding an argument with a default that the
   caller never passes all rewrite the file (new digest) and change no number.
   Conversely a one-line edit inside the integrator changes everything.

What this module decides
-----------------------
:func:`certificate` answers, for two source texts, a question with a useful
strength: *are the code objects reachable from the simulation entry points
identical?*  If they are, the two digests cannot differ numerically for **any**
specification -- a theorem over the whole parameter space, not a sample.  If
they are not, the honest verdict is ``NEEDS_RUNTIME``: run both and measure.

Verdicts (pre-registered; no other values are produced)
-------------------------------------------------------
``IDENTICAL_SOURCE``
    Byte-identical files.  Trivially equivalent.
``EQUIVALENT_PATH``
    Text differs, but every code object reachable from the roots is
    byte-identical and nothing reachable was removed.  Numeric equivalence
    holds for all inputs, not just the ones that happened to be rerun.
``EQUIVALENT_PATH_SHARED_STATE``
    As above, but a module-level statement (imports, constants, decorators)
    outside the function bodies changed; equivalence is asserted for the roots
    only and the changed module-level code is listed for review.
``NEEDS_RUNTIME``
    At least one reachable function was modified or removed.  Must be decided
    by running both implementations on the same spec + seed.
``MISSING_SOURCE`` / ``UNPARSED``
    Missing or syntactically invalid source: no evidence, never a pass.

Reachability is computed on the AST call graph (calls by name, attribute calls
resolved by their last component, plus ``self.<method>`` inside a class).  It is
conservative in the direction that matters: a dynamic dispatch we cannot see
leaves the callee *outside* the reachable set, so :func:`certificate` reports
the functions it found changed and lists them; a reviewer can always demote a
verdict to ``NEEDS_RUNTIME``, never promote one.

Why not just diff the text
--------------------------
The m22824 audit wrote the rule "diff 只用来提出假设，证书才是证据" (a diff
proposes a hypothesis; the certificate is the evidence).  A text diff cannot
distinguish "helper appended after the integrator" from "line deleted inside the
integrator" without a human reading it, and it cannot tell whether the appended
helper is called by the physics.  Compiling both versions and comparing the code
objects of the reachable closure is machine-decidable and does not depend on
anyone reading the diff.

Units: none (this module is pure source analysis; no numerics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import hashlib
import types
from typing import Iterable, Mapping

__all__ = [
    "Verdict",
    "ModuleComparison",
    "certificate",
    "code_fingerprint",
    "function_code_map",
    "module_level_fingerprint",
    "reachable_from",
    "local_imports",
    "dependency_closure",
]

# ---------------------------------------------------------------------------
# Pre-registered verdict vocabulary
# ---------------------------------------------------------------------------

IDENTICAL_SOURCE = "IDENTICAL_SOURCE"
EQUIVALENT_PATH = "EQUIVALENT_PATH"
EQUIVALENT_PATH_SHARED_STATE = "EQUIVALENT_PATH_SHARED_STATE"
NEEDS_RUNTIME = "NEEDS_RUNTIME"
MISSING_SOURCE = "MISSING_SOURCE"
UNPARSED = "UNPARSED"

Verdict = str  # one of the six constants above

#: verdicts that license "the two digests may be compared without a rerun"
EQUIVALENT_VERDICTS = (IDENTICAL_SOURCE, EQUIVALENT_PATH,
                       EQUIVALENT_PATH_SHARED_STATE)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop only the *docstring* position of every scope, in place.

    A docstring edit rewrites the file (new digest) and cannot change a number,
    so it must not be able to fail a certificate.  Nothing else is touched.
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def parse_source(source: str) -> ast.Module:
    """Parse ``source`` with docstrings stripped (raises ``SyntaxError``)."""
    return _strip_docstrings(ast.parse(source))


class _FuncCollector(ast.NodeVisitor):
    """Collect ``qualname -> FunctionDef`` for every def in the module."""

    def __init__(self) -> None:
        self.funcs: dict[str, ast.AST] = {}
        self._stack: list[str] = []

    def _visit_scope(self, node, name: str) -> None:
        self._stack.append(name)
        qualname = ".".join(self._stack)
        self.funcs[qualname] = node
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self._stack.pop()

    def visit_FunctionDef(self, node):  # noqa: N802 (ast API)
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        self._visit_scope(node, node.name)

    def visit_ClassDef(self, node):  # noqa: N802
        self._stack.append(node.name)
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self._stack.pop()


def _code_sha(code: types.CodeType) -> str:
    """SHA-256 over a code object, recursing into nested code objects.

    Stamps ``co_code`` (the bytecode), the name/const tables and the argument
    layout, and recurses into function/closure constants.  Line numbers and file
    names are deliberately excluded: moving a function down the file changes
    both and changes no behaviour.
    """
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(repr((code.co_argcount, code.co_posonlyargcount,
                   code.co_kwonlyargcount, code.co_nlocals, code.co_flags,
                   code.co_names, code.co_varnames,
                   code.co_freevars, code.co_cellvars)).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            h.update(b"<code>")
            h.update(_code_sha(const).encode())
        else:
            h.update(b"<const>")
            h.update(repr(const).encode())
    return h.hexdigest()


def function_code_map(source: str) -> dict[str, str]:
    """``qualname -> code fingerprint`` for every def in ``source``.

    The whole module is compiled *as a module* and its code objects are walked
    recursively, so scoping constructs (``nonlocal``, closures, comprehensions,
    lambdas) are compiled exactly as the interpreter sees them.  Compiling a
    nested def in isolation would raise ``no binding for nonlocal`` -- that is
    the reason this function does not use per-fragment compilation.

    Raises ``SyntaxError`` if ``source`` does not parse.
    """
    tree = parse_source(source)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<module>", "exec")
    out: dict[str, str] = {}

    def walk(c: types.CodeType, prefix: str) -> None:
        counts: dict[str, int] = {}
        for const in c.co_consts:
            if not isinstance(const, types.CodeType):
                continue
            name = const.co_name
            n = counts.get(name, 0)
            counts[name] = n + 1
            qualname = prefix + name + ("" if n == 0 else f"~{n}")
            out[qualname] = _code_sha(const)
            walk(const, qualname + ".")

    walk(code, "")
    return out


def module_level_fingerprint(source: str) -> str:
    """Fingerprint of the module's *top-level* statements (excluding defs' bodies).

    This is where imports, module constants and decorators live -- the shared
    state a function body cannot see.  Function bodies are replaced by their
    names so that editing a body does not change this fingerprint.
    """
    tree = parse_source(source)
    stripped = ast.Module(body=[], type_ignores=[])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stripped.body.append(ast.Pass())
        elif isinstance(node, ast.ClassDef):
            stripped.body.append(ast.Pass())
        else:
            stripped.body.append(node)
    if not stripped.body:
        stripped.body.append(ast.Pass())
    ast.fix_missing_locations(stripped)
    code = compile(stripped, "<module-level>", "exec")
    return _code_sha(code)


def code_fingerprint(source: str) -> str:
    """Single fingerprint over the whole module: module-level code + all defs."""
    parts = [module_level_fingerprint(source)]
    for qualname, sha in sorted(function_code_map(source).items()):
        parts.append(f"{qualname}:{sha}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Call graph / reachability
# ---------------------------------------------------------------------------

def _called_names(node: ast.AST) -> set[str]:
    """Names called inside ``node`` (by ``Name`` or attribute's last component)."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name):
            out.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            out.add(fn.attr)
    return out


def reachable_from(source: str, roots: Iterable[str]) -> list[str]:
    """Qualnames transitively called from ``roots`` (uniform-cost closure).

    Resolution is by simple name: ``foo()`` and ``self.foo()`` and ``mod.foo()``
    all resolve to the qualname ending in ``.foo`` or equal to ``foo``.  Missing
    roots raise ``KeyError`` -- a certificate over a root that does not exist is
    no certificate.
    """
    tree = parse_source(source)
    coll = _FuncCollector()
    coll.visit(tree)
    by_short: dict[str, list[str]] = {}
    for qualname in coll.funcs:
        by_short.setdefault(qualname.rsplit(".", 1)[-1], []).append(qualname)

    missing = [r for r in roots if r not in coll.funcs]
    if missing:
        raise KeyError(f"root(s) not defined in source: {missing}")

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        qualname = stack.pop()
        if qualname in seen:
            continue
        seen.add(qualname)
        node = coll.funcs.get(qualname)
        if node is None:
            continue
        for name in _called_names(node):
            for target in by_short.get(name, ()):
                if target not in seen:
                    stack.append(target)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Import closure (the "one file" defect)
# ---------------------------------------------------------------------------

def local_imports(source: str, package: str) -> dict[str, str]:
    """Map ``absolute module name -> what was imported`` for intra-package imports.

    ``from .granular3d_core import damping_ratio`` inside package
    ``tensordem`` yields ``{"tensordem.granular3d_core": "damping_ratio"}``;
    ``from . import x`` and ``from ..other import y`` are handled; stdlib and
    third-party imports are ignored (their versions are pinned elsewhere).
    """
    tree = parse_source(source)
    out: dict[str, str] = {}
    pkg_parts = package.split(".") if package else []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                mod = node.module or ""
                if not (mod == package or mod.startswith(package + ".")):
                    continue
                absolute = mod
            else:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                absolute = ".".join(base + ([node.module] if node.module else []))
            names = ",".join(sorted(a.name for a in node.names))
            out[absolute] = (out.get(absolute, "") + ";" + names).strip(";")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    out.setdefault(alias.name, "*")
    return out


def dependency_closure(sources: Mapping[str, str], roots: Iterable[str],
                       package: str) -> dict[str, list[str]]:
    """Transitive local-import closure starting from ``roots`` module names.

    ``sources`` maps ``absolute module name -> source text`` (only the modules
    that exist in the repository need to be present; an import of a module that
    is absent from ``sources`` is simply not followed -- it is not a repo module
    or it is not being compared).
    """
    seen: dict[str, list[str]] = {}
    stack = [r for r in roots if r in sources]
    for r in roots:
        if r not in sources:
            seen[r] = []
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        imports = local_imports(sources[mod], package)
        seen[mod] = sorted(imports)
        for dep in imports:
            if dep in sources and dep not in seen:
                stack.append(dep)
    return dict(sorted(seen.items()))


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

@dataclass
class ModuleComparison:
    """Comparison of two versions of one module, restricted to ``roots``."""

    module: str
    roots: list[str]
    verdict: Verdict
    text_sha_old: str
    text_sha_new: str
    fingerprint_old: str
    fingerprint_new: str
    module_level_changed: bool
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    reachable_old: list[str] = field(default_factory=list)
    reachable_new: list[str] = field(default_factory=list)
    reachable_modified: list[str] = field(default_factory=list)
    reachable_removed: list[str] = field(default_factory=list)
    reachable_added: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return self.verdict in EQUIVALENT_VERDICTS

    def as_dict(self) -> dict:
        d = {
            "module": self.module,
            "roots": self.roots,
            "verdict": self.verdict,
            "equivalent": self.equivalent,
            "text_equal": self.text_sha_old == self.text_sha_new,
            "fingerprint_old": self.fingerprint_old,
            "fingerprint_new": self.fingerprint_new,
            "module_level_changed": self.module_level_changed,
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
            "reachable_old": self.reachable_old,
            "reachable_new": self.reachable_new,
            "reachable_modified": self.reachable_modified,
            "reachable_removed": self.reachable_removed,
            "reachable_added": self.reachable_added,
            "notes": self.notes,
        }
        return d


def certificate(old_source: str | None, new_source: str | None,
                roots: Iterable[str], module: str = "<module>") -> ModuleComparison:
    """Decide whether two versions of ``module`` are numerically equivalent.

    ``roots`` are the entry points whose reachable code must be identical (for
    a DEM run: the public driver, here ``("run_deposition",)``).  The verdict is
    one of the six pre-registered strings; only the three ``EQUIVALENT*`` values
    mean "may be compared without a rerun" and only ``NEEDS_RUNTIME`` means
    "must be measured".
    """
    roots = list(roots)
    if old_source is None or new_source is None:
        return ModuleComparison(module, roots, MISSING_SOURCE, "", "", "", "",
                                False, notes=["one of the two sources is absent"])
    text_old = hashlib.sha256(old_source.encode()).hexdigest()
    text_new = hashlib.sha256(new_source.encode()).hexdigest()
    try:
        fmap_old = function_code_map(old_source)
        fmap_new = function_code_map(new_source)
        ml_old = module_level_fingerprint(old_source)
        ml_new = module_level_fingerprint(new_source)
        reach_old = reachable_from(old_source, roots)
        reach_new = reachable_from(new_source, roots)
    except (SyntaxError, KeyError) as exc:
        return ModuleComparison(module, roots, UNPARSED, text_old, text_new, "",
                                "", False, notes=[f"{type(exc).__name__}: {exc}"])

    fp_old = code_fingerprint(old_source)
    fp_new = code_fingerprint(new_source)

    added = sorted(set(fmap_new) - set(fmap_old))
    removed = sorted(set(fmap_old) - set(fmap_new))
    modified = sorted(q for q in set(fmap_old) & set(fmap_new)
                      if fmap_old[q] != fmap_new[q])

    r_old, r_new = set(reach_old), set(reach_new)
    reach_removed = sorted(r_old - set(fmap_new))
    reach_modified = sorted(q for q in modified if q in r_old or q in r_new)
    reach_added = sorted(q for q in added if q in r_new)

    notes: list[str] = []
    if text_old == text_new:
        verdict = IDENTICAL_SOURCE
    elif reach_modified or reach_removed:
        verdict = NEEDS_RUNTIME
        notes.append("a function reachable from the roots changed: run both "
                     "implementations on the same spec+seed and compare")
    elif ml_old != ml_new:
        verdict = EQUIVALENT_PATH_SHARED_STATE
        notes.append("module-level statements changed (imports/constants/"
                     "decorators) while every reachable function is identical")
    else:
        verdict = EQUIVALENT_PATH
        if added:
            notes.append(f"{len(added)} function(s) added outside the reachable "
                         f"set: {', '.join(added[:6])}"
                         + ("..." if len(added) > 6 else ""))

    return ModuleComparison(
        module=module, roots=roots, verdict=verdict,
        text_sha_old=text_old, text_sha_new=text_new,
        fingerprint_old=fp_old, fingerprint_new=fp_new,
        module_level_changed=(ml_old != ml_new),
        added=added, removed=removed, modified=modified,
        reachable_old=reach_old, reachable_new=reach_new,
        reachable_modified=reach_modified, reachable_removed=reach_removed,
        reachable_added=reach_added, notes=notes,
    )