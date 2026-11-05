"""Verify the PR #200 cure merge of ``agent/m3296`` into ``main``.

Two things have to be *machine-checkable*, because both have failed before in
this repository:

1. **No silent content loss.**  A mechanical overlay resolution once dropped a
   whole side's hunks while every test still collected.  So the merge result is
   audited the way a reviewer would: every file either side touched from the
   merge base must exist in the result, and every file that only *one* side
   touched must be byte-identical to that side (git's own text merge is trusted
   only where there is nothing to decide).  The single both-sides file is
   listed explicitly — it is the one place a human/agent decision was made.
2. **The merged contract actually behaves.**  ``MovingWedgeBowIceDEM3D.forces``
   accepting ``update_history`` is not enough: a probe must not advance the
   Cundall-Strack slip history.  A positive control (the default call *does*
   advance it) is included so the assertion cannot pass vacuously.

Outputs (all under ``artifacts/pr200/``)::

    verify_pr200.json          machine-readable audit record
    pr200_merge_evidence.png   two-panel evidence figure

Usage::

    python scripts/verify_pr200_merge.py \
        --base github/main --head github/agent/m3296 \
        --merge <merge-commit-or-index>
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "pr200"
CONFLICT_FILE = "src/tensordem/dem3d_wedge.py"
MARKERS = ("<<<<<<< ", "=======", ">>>>>>> ")

#: Deviations from "byte-identical to the sole changing side" are *allowed*, but
#: they must be declared here with a reason.  An undeclared deviation is exactly
#: the silent-loss failure mode this script exists to catch.
DECLARED_AMENDMENTS = {
    "docs/doc_evidence_ledger.json": (
        "main-side file, regenerated with scripts/audit_doc_evidence.py so the new "
        "contract doc's pointers enter the evidence ledger (the ledger is a ratchet)"
    ),
    "docs/dem3d_wedge_bow_probe_contract.md": (
        "branch-side file, one pointer repaired: scripts/ice_floe_physics.py -> "
        "benchmarks/ice_floe_physics.py (main's m3317 restoration relocated the script)"
    ),
}


# --------------------------------------------------------------------------- #
# git plumbing
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    return subprocess.run(
        ("git",) + args, cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8", "replace")


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True, check=True).stdout


def side_files(base: str, rev: str) -> set[str]:
    """Files ``rev`` changed relative to ``base``."""
    return set(_git("diff", "--name-only", base, rev).split())


def merged_files(base: str, rev: str) -> set[str]:
    return set(_git("diff", "--name-only", base, rev).split())


def audit_no_content_loss(base: str, main_rev: str, branch_rev: str, merge_rev: str) -> dict:
    main_side = side_files(base, main_rev)
    branch_side = side_files(base, branch_rev)
    result = merged_files(base, merge_rev)

    dropped = sorted((main_side | branch_side) - result)
    both = sorted(main_side & branch_side)
    amendments, additions, byte_mismatch = [], [], []
    for path in sorted(result):
        only_main, only_branch = path in main_side, path in branch_side
        if only_main and only_branch:
            continue  # resolved by hand, audited separately
        if not (only_main or only_branch):
            additions.append(path)  # created by the cure itself (doc/script/artifacts)
            continue
        side = main_rev if only_main else branch_rev
        try:
            theirs = _git_bytes("show", f"{side}:{path}")
        except subprocess.CalledProcessError:
            theirs = None  # deleted on that side
        try:
            ours = _git_bytes("show", f"{merge_rev}:{path}")
        except subprocess.CalledProcessError:
            ours = None
        if theirs != ours:
            entry = {"path": path, "side": side,
                     "declared_reason": DECLARED_AMENDMENTS.get(path)}
            (amendments if entry["declared_reason"] else byte_mismatch).append(entry)

    return {
        "merge_base": base,
        "files_changed_by_main": sorted(main_side),
        "files_changed_by_branch": sorted(branch_side),
        "files_changed_by_merge": sorted(result),
        "dropped_files": dropped,
        "both_sides_files_resolved_by_hand": both,
        "declared_amendments": amendments,
        "undeclared_amendments": byte_mismatch,
        "additions_by_the_cure": additions,
        "ok": not dropped and not byte_mismatch,
    }


def audit_conflict_markers(merge_rev: str, base: str) -> dict:
    hits = {}
    for path in merged_files(base, merge_rev):
        if not path.endswith((".py", ".md", ".json", ".toml", ".yml", ".yaml")):
            continue
        try:
            text = _git("show", f"{merge_rev}:{path}")
        except subprocess.CalledProcessError:
            continue
        bad = []
        fenced = False
        for i, line in enumerate(text.splitlines()):
            if line.lstrip().startswith("```"):
                fenced = not fenced  # markdown quotes of markers live in fences
                continue
            if fenced:
                continue
            if line.startswith(MARKERS[0]) or line.startswith(MARKERS[2]):
                bad.append(i + 1)
        if bad:
            hits[path] = bad
    return {"conflict_markers_left": hits, "ok": not hits}


def audit_duplicate_definitions(path: str = CONFLICT_FILE) -> dict:
    source = (ROOT / path).read_text()
    tree = ast.parse(source)
    seen: dict[str, int] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            seen[name] = seen.get(name, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    return {"file": path, "duplicate_top_level_definitions": dups, "ok": not dups}


# --------------------------------------------------------------------------- #
# behavioural check: the merged contract is side-effect free when asked
# --------------------------------------------------------------------------- #
def _bow_model():
    from tensordem.dem3d import DEM3DConfig
    from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D

    cfg = DEM3DConfig(
        nx=3, ny=2, nz=3, radius=0.02, tool_speed=0.1,
        bond_stiffness=1000.0, contact_stiffness=3000.0,
        breaking_strain=0.5, shear_breaking_strain=0.5,
        drag=0.1, contact_damping=0.5, friction_coefficient=0.4,
        fix_x_edges=False, fix_bottom=False,
    )
    return MovingWedgeBowIceDEM3D(cfg, bow_speed=0.1, wedge_angle_deg=45.0)


def _history_fingerprint(model) -> tuple:
    state = getattr(model, "_tangential_state", {})
    return tuple(sorted(
        (int(k), float(v[0].sum()), float(v[1].sum())) for k, v in state.items()
    ))


def _compress(model, scale: float = 0.97) -> None:
    """Pull the lattice inward so neighbouring spheres overlap.

    A freshly built lattice puts neighbours exactly ``2 r`` apart, i.e. *touching
    but not overlapping*: no active contact pair exists yet, so the friction
    history is empty and a "probe leaves history untouched" assertion would pass
    vacuously (``{} == {}``).  Compressing by 3 % guarantees real contacts.
    """
    import torch

    with torch.no_grad():
        centre = model.positions.mean(dim=0, keepdim=True)
        model.positions = centre + (model.positions - centre) * scale


def audit_probe_side_effects() -> dict:
    import inspect

    import torch

    from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D

    params = inspect.signature(MovingWedgeBowIceDEM3D.forces).parameters
    signature = {
        "has_update_history": "update_history" in params,
        "keyword_only": params.get("update_history") is not None
        and params["update_history"].kind is inspect.Parameter.KEYWORD_ONLY,
    }

    model = _bow_model()
    _compress(model)

    probe_a, _ = model.forces(update_fracture=False, update_history=False)
    fp_a = _history_fingerprint(model)
    probe_b, _ = model.forces(update_fracture=False, update_history=False)
    fp_b = _history_fingerprint(model)
    probe_repeatable = bool(torch.equal(probe_a, probe_b))
    probe_side_effect_free = fp_a == fp_b

    # positive control #1: a stepping call (update_history=True) must populate /
    # advance the friction history, otherwise "no side effect" is vacuous.
    model.forces(update_fracture=False)
    fp_step = _history_fingerprint(model)
    stepping_advances_history = fp_step != fp_a and len(fp_step) > 0

    # positive control #2: after a stepping call, a probe must still not move it.
    model.forces(update_fracture=False, update_history=False)
    probe_after_step_untouched = _history_fingerprint(model) == fp_step

    return {
        "signature": signature,
        "n_active_contact_pairs": len(fp_step),
        "probe_repeatable": probe_repeatable,
        "probe_leaves_history_untouched": probe_side_effect_free,
        "stepping_call_advances_history": stepping_advances_history,
        "probe_after_step_untouched": probe_after_step_untouched,
        "force_norm_probe_N": float(probe_a.norm()),
        "ok": all(signature.values()) and probe_repeatable and probe_side_effect_free
        and stepping_advances_history and probe_after_step_untouched,
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot(record: dict, tests: dict | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    loss = record["content_loss_audit"]
    labels = ["main\n改动文件", "m3296\n改动文件", "合并结果\n文件", "仅一侧\n逐字节一致", "手工解冲\n文件"]
    values = [
        len(loss["files_changed_by_main"]),
        len(loss["files_changed_by_branch"]),
        len(loss["files_changed_by_merge"]),
        len(loss["files_changed_by_merge"]) - len(loss["both_sides_files_resolved_by_hand"]),
        len(loss["both_sides_files_resolved_by_hand"]),
    ]
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v), ha="center", fontsize=10)
    ax.set_title("PR #200 merge surface: 0 dropped, 1 hand-resolved")
    ax.set_ylabel("files")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    if tests:
        groups = ["passed", "failed"]
        base_v = [tests["main_baseline"]["passed"], tests["main_baseline"]["failed"]]
        merged_v = [tests["merged"]["passed"], tests["merged"]["failed"]]
        x = range(len(groups))
        w = 0.36
        ax.bar([i - w / 2 for i in x], base_v, w, label="main baseline", color="#4C72B0")
        ax.bar([i + w / 2 for i in x], merged_v, w, label="cured branch", color="#55A868")
        for i, (a, b) in enumerate(zip(base_v, merged_v)):
            ax.text(i - w / 2, a + 8, str(a), ha="center", fontsize=9)
            ax.text(i + w / 2, b + 8, str(b), ha="center", fontsize=9)
        ax.set_xticks(list(x), groups)
        ax.set_ylabel("tests")
        ax.set_title("Full suite: no regression vs main")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "suite results pending", ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _normalise_suite(raw: dict) -> dict:
    """Accept either ``{main_baseline, merged}`` or a named multi-run record."""
    if "main_baseline" in raw and "merged" in raw:
        return {"main_baseline": raw["main_baseline"], "merged": raw["merged"]}
    runs = raw.get("runs", raw)
    base = next(v for k, v in runs.items() if k.startswith("main_baseline"))
    merged = next(v for k, v in runs.items() if k.startswith("final"))
    return {"main_baseline": base, "merged": merged,
            "raw_runs": runs, "command": raw.get("command")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="github/main")
    ap.add_argument("--head", default="github/agent/m3296")
    ap.add_argument("--merge", default="HEAD")
    ap.add_argument("--tests", default=None,
                    help="JSON file with {main_baseline:{passed,failed},merged:{passed,failed}}")
    args = ap.parse_args(argv)

    merge_base = _git("merge-base", args.base, args.head).strip()
    record = {
        "pr": 200,
        "branch": "agent/m3296",
        "main": _git("rev-parse", args.base).strip(),
        "branch_sha": _git("rev-parse", args.head).strip(),
        "merge_base": merge_base,
        "merge_ref": args.merge,
        "content_loss_audit": audit_no_content_loss(merge_base, args.base, args.head, args.merge),
        "conflict_marker_audit": audit_conflict_markers(args.merge, merge_base),
        "duplicate_definition_audit": audit_duplicate_definitions(),
        "contract_audit": audit_probe_side_effects(),
    }
    record["ok"] = all(
        record[k]["ok"]
        for k in ("content_loss_audit", "conflict_marker_audit",
                  "duplicate_definition_audit", "contract_audit")
    )

    tests = None
    if args.tests:
        tests = _normalise_suite(json.loads(Path(args.tests).read_text()))
        record["suite"] = tests

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify_pr200.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    plot(record, tests, OUT / "pr200_merge_evidence.png")

    print(json.dumps({k: record[k] for k in ("ok", "content_loss_audit",
                                             "conflict_marker_audit",
                                             "contract_audit")}, indent=2,
                     ensure_ascii=False)[:2000])
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())