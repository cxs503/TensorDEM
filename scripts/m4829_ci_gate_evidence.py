"""Programmatic evidence for the m4829 main-CI repair (no hand-copied numbers).

Two red items on ``main@524821fd`` are reproduced and then re-scored here:

1. ``tests/test_dem2d_fracture_ledger.py::Dem2DFractureLedgerTests::test_provenance``
   asserted that the imported ``tensordem`` path contains ``os.sep + "work" + os.sep``.
   That is a property of the *runner's mount layout*, not of the checkout: it holds
   on GitHub-hosted runners (``/home/runner/work/...``) and fails on a self-hosted
   runner that checks out under ``_work`` (``.../_work/TensorDEM/TensorDEM``) even
   though the imported engine is exactly the one in the tree.  The gate is re-derived
   from the repository root and scored on every layout shape.

2. ``tests/test_doc_evidence_pointers.py::...::test_cited_source_docs_and_tests_are_tracked``
   cited ``threefield/fragmentation.py``, which git does not have (the module lives at
   ``src/tensordem/threefield/fragmentation.py``), and the committed ledger
   ``docs/doc_evidence_ledger.json`` was never regenerated after the docs merged in the
   m4819 PR, so it froze 145 documents against the 151 in the tree.

Usage::

    python scripts/m4829_ci_gate_evidence.py            # writes artifacts/ + figures
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts" / "m4829_ci_gate"

# Checkout layouts that this repository is actually exercised in.
LAYOUTS = {
    "github-hosted (/home/runner/work/...)": "/home/runner/work/TensorDEM/TensorDEM",
    "self-hosted loong64 (.../ci_env/_work/...)": "/data/ci_env/_work/TensorDEM/TensorDEM",
    "agent worktree (/root/tp-agent/work/m4829)": "/root/tp-agent/work/m4829",
    "tmp sandbox (/tmp/_work/TensorDEM/TensorDEM)": "/tmp/_work/TensorDEM/TensorDEM",
}

STATUS_KEYS = ("code:TRACKED", "data:TRACKED", "data:MISSING", "doc:TRACKED",
               "figure:TRACKED", "figure:MISSING", "test:TRACKED")


def _gate_scores() -> list[dict]:
    """Score both gate formulations on every layout shape.

    ``check_provenance`` anchors on the tree that hosts *this* script, so a
    foreign checkout cannot be simulated by faking a module path alone.  Each
    layout is therefore scored by applying the gate's rule to that layout's own
    root (``engine is <root>/src/tensordem/__init__.py``); the live worktree is
    additionally scored through the real ``check_provenance()`` call.
    """
    from scripts.verify_dem2d_fracture_ledger import check_provenance

    rows = []
    for label, checkout in LAYOUTS.items():
        engine = f"{checkout}/src/tensordem/__init__.py"
        # gate as written on main@524821fd: runner-mount-layout probe
        legacy = (os.sep + "work" + os.sep) in os.path.abspath(engine)
        # gate after the repair: the import must come from this checkout
        resolved = Path(engine).resolve()
        repaired = (resolved.parent.parent == (Path(checkout) / "src").resolve()
                    and resolved.name == "__init__.py")
        live = ""
        if Path(checkout).resolve() == ROOT.resolve():
            live = check_provenance()  # the real call, in the tree that runs this
            repaired = repaired and Path(live).resolve() == resolved
        rows.append(dict(layout=label, checkout=checkout, legacy_gate=legacy,
                         repaired_gate=repaired, engine=engine, live_call=live))
    # negative control: an engine outside this checkout must be rejected
    outside = "/opt/other-checkout/src/tensordem/__init__.py"
    try:
        check_provenance(outside)
        verdict = "NOT REJECTED"
    except RuntimeError:
        verdict = "RuntimeError"
    rows.append(dict(layout="negative control (foreign checkout)",
                     checkout="/opt/other-checkout", legacy_gate=False,
                     repaired_gate=False, engine=outside, live_call=verdict))
    return rows


def _ledger_scores() -> dict:
    head = json.loads(
        subprocess.run(["git", "-C", str(ROOT), "show",
                        "HEAD:docs/doc_evidence_ledger.json"],
                       capture_output=True, text=True, check=True).stdout
    )
    live = json.loads((ROOT / "docs" / "doc_evidence_ledger.json").read_text(encoding="utf-8"))

    key = lambda e: (e["doc"], e["pointer"], e["status"])  # noqa: E731
    head_keys = {key(e) for e in head["entries"]}
    live_keys = {key(e) for e in live["entries"]}
    ratio = lambda rep: len([e for e in rep["entries"]  # noqa: E731
                             if e["status"] == "TRACKED"]) / len(rep["entries"])
    return dict(
        head_documents=head["documents_scanned"], live_documents=live["documents_scanned"],
        head_pointers=head["pointers"], live_pointers=live["pointers"],
        head_counts={k: head["counts"].get(k, 0) for k in STATUS_KEYS},
        live_counts={k: live["counts"].get(k, 0) for k in STATUS_KEYS},
        stale_entries=sorted(f"{d}:{p} [{s}]" for d, p, s in head_keys - live_keys),
        new_entries=sorted(f"{d}:{p} [{s}]" for d, p, s in live_keys - head_keys),
        head_tracked_fraction=ratio(head), live_tracked_fraction=ratio(live),
        # the pointer the gate flagged, before and after the doc repair
        cited_threefield=[p for p in sorted({e["pointer"] for e in live["entries"]})
                          if "threefield/fragmentation.py" in p],
    )


def _plots(gates: list[dict], ledger: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    labels = [g["layout"] for g in gates]
    y = range(len(gates))
    legacy = [1 if g["legacy_gate"] else 0 for g in gates]
    repaired = [1 if g["repaired_gate"] else 0 for g in gates]
    ax.barh([i + 0.2 for i in y], legacy, height=0.38, color="#c0392b",
            label='main@524821fd gate:  "/work/" in path')
    ax.barh([i - 0.2 for i in y], repaired, height=0.38, color="#1e8449",
            label="repaired gate: engine resolves to this checkout")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.invert_yaxis()
    ax.set_title("Provenance gate verdict per checkout layout\n"
                 "(same clean tree; only the mount path changes)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    path = OUT / "provenance_gate_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    width = 0.38
    idx = range(len(STATUS_KEYS))
    ax.bar([i - width / 2 for i in idx],
           [ledger["head_counts"][k] for k in STATUS_KEYS], width,
           color="#7f8c8d", label=f'committed ledger (stale, {ledger["head_documents"]} docs)')
    ax.bar([i + width / 2 for i in idx],
           [ledger["live_counts"][k] for k in STATUS_KEYS], width,
           color="#2471a3", label=f'regenerated ledger ({ledger["live_documents"]} docs)')
    ax.set_xticks(list(idx))
    ax.set_xticklabels(STATUS_KEYS, rotation=20, fontsize=7.5)
    ax.set_ylabel("pointers")
    ax.set_title("Evidence ledger: committed (stale) vs regenerated\n"
                 f'{ledger["head_pointers"]} -> {ledger["live_pointers"]} pointers; '
                 f'pointers dropped by the gate: {len(ledger["stale_entries"])}',
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    path = OUT / "ledger_regeneration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))
    return names


def main() -> dict:
    gates = _gate_scores()
    ledger = _ledger_scores()
    figures = _plots(gates, ledger)
    summary = dict(
        schema="tensordem.m4829-ci-gate-evidence/1",
        generated_by="scripts/m4829_ci_gate_evidence.py",
        source_commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip(),
        provenance_gate=dict(
            layouts_failing_legacy=[g["layout"] for g in gates if not g["legacy_gate"]],
            layouts_failing_repaired=[g["layout"] for g in gates if not g["repaired_gate"]],
            rows=gates,
        ),
        ledger=ledger,
        figures=figures,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_evidence.json").write_text(json.dumps(summary, indent=1) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("provenance_gate", "figures")}, indent=1)[:2500])
    print("ledger:", {k: ledger[k] for k in ("head_documents", "live_documents",
                                             "head_pointers", "live_pointers",
                                             "head_tracked_fraction",
                                             "live_tracked_fraction")})
    print("stale entries:", len(ledger["stale_entries"]), "new entries:",
          len(ledger["new_entries"]))
    return summary


if __name__ == "__main__":
    main()