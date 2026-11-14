#!/usr/bin/env python3
"""XVAL round 4: contract *lineage* for the frozen binding contract.

Anchor: XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT.

Round 1/2 froze the surface and detected contract-vs-code drift; round 3 made
the surface fully accounted.  Neither could see an edit of the *frozen data*:
round 3 moved the payload from ``sha256:0cbfed8c...`` to ``sha256:048e4cf9...``
while ``CONTRACT_VERSION`` stayed ``1.0.0`` and nothing recorded the change.
Round 4 adds the lineage half (diff, release chain, binder manifests) and this
script measures it against the recorded baselines and against controlled
mutants -- including the detector-blindness baseline that motivated the work.

Every number in ``artifacts/xval_library_binding_contract/result_r4.json`` is
produced here; the report reads that file.

Usage::

    python scripts/xval_library_binding_contract_r4.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import binding_contract as bc  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "xval_library_binding_contract"
OWNER_TOLERANCE = 0.15


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _diff_row(baseline: dict) -> dict:
    payload = json.loads(bc.canonical_payload())
    findings = bc.diff_contracts(baseline["payload"], payload)
    summary = bc.summarise_findings(findings)
    tail = bc.CONTRACT_HISTORY[-1]
    return {
        "baseline_version": baseline["contract_version"],
        "baseline_fingerprint": baseline["fingerprint"],
        "findings": summary["findings"],
        "by_severity": summary["by_severity"],
        "change_classes": summary["change_classes"],
        "required_bump": bc.required_bump(findings),
        "declared_bump": tail["bump"],
        "declared_changes": tail["changes"],
        "bump_agrees": tail["bump"] == bc.required_bump(findings),
        "classes_agree": tail["changes"] == summary["change_classes"],
    }


def _blindness_baseline() -> list[dict]:
    """How much of a metadata edit does the introspection verifier see?

    ``verify()`` compares the frozen data against the live classes.  Anything
    the live classes cannot express (units, declared shapes, clock tolerances,
    capability names, ledger dispositions) is therefore invisible to it.  This
    is the measured reason round 4 needs a contract-to-contract diff.
    """
    targets = bc.default_instances()
    base = json.loads(bc.canonical_payload())

    mutations = {
        "units_retuned": lambda c: c["entries"]["IceDEM"]["state"]["dt"].update(units="ms"),
        "shape_retyped": lambda c: c["entries"]["IceDEM"]["state"]["positions"].update(
            shape="(N, 3)"),
        "dtype_retyped": lambda c: c["entries"]["IceDEM"]["state"]["fixed"].update(dtype="int8"),
        "clock_tolerance_loosened": lambda c: c["entries"]["IceDEM"]["clock"].update(
            load_time_guard_rel_tol=1e-6),
        "capability_dropped": lambda c: c["entries"]["IceDEM"]["capabilities"].pop("restart"),
        "negative_assertion_loosened": lambda c: c["entries"]["IceDEM"][
            "explicitly_absent"].remove("restore"),
        "param_units_retuned": lambda c: c["entries"]["IceDEM"]["methods"]["step"][
            "params"]["external_forces"].update(units="kN"),
        "ledger_disposition_tightened": lambda c: c["entries"]["IceDEM"]["surface_ledger"][
            "fragments"].update(disposition="excluded", reason="diagnostic"),
        "required_param_added": lambda c: c["entries"]["IceDEM"]["methods"]["step"][
            "params"].update(external_torque={"kind": "keyword_only", "required": True}),
        "method_return_retyped": lambda c: c["entries"]["IceDEM"]["methods"][
            "forces"].update(returns="torch.Tensor"),
    }
    rows = []
    for key, fn in mutations.items():
        mutated = json.loads(json.dumps(base))
        fn(mutated)
        verifier = bc.verify(targets, contract=mutated)
        findings = bc.diff_contracts(base, mutated)
        rows.append({
            "key": key,
            "verify_breaking": verifier["breaking"],
            "verify_status": verifier["status"],
            "diff_findings": len(findings),
            "diff_breaking": sum(1 for f in findings if f["severity"] == "breaking"),
            "diff_change_classes": sorted({f["change"] for f in findings}),
            "diff_detected": bool(findings),
        })
    return rows


def _participants(capsys_rows: list[dict]) -> dict:
    total = len(capsys_rows)
    detected = sum(1 for row in capsys_rows if row["detected"])
    return {"rows": total, "detected": detected,
            "rate": detected / total if total else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=str, default=str(ARTIFACTS))
    parser.add_argument("--inject-doc", type=str, default=None,
                        help="replace the R4-RESULTS block of this markdown doc")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    baseline_100 = _load(outdir / "manifest_v1.0.0.json")
    round3 = _load(outdir / "manifest_round3_unrecorded.json")

    lineage = bc.check_lineage()
    diff_row = _diff_row(baseline_100)
    unrecorded = bc.check_consumer_manifest(round3)
    clean_manifest = bc.check_consumer_manifest(bc.export_manifest())
    older_manifest = bc.check_consumer_manifest(baseline_100)

    matrix = bc.lineage_matrix()
    matrix_rows = [row for row in matrix if row["expect_change"] is not None]
    controls = [row for row in matrix if row["expect_change"] is None]
    matrix_score = _participants(matrix_rows)
    control_clean = sum(1 for row in controls if row["status"] == "CONFORMS")

    blindness = _blindness_baseline()
    verify_seen = sum(1 for row in blindness if row["verify_breaking"])
    diff_seen = sum(1 for row in blindness if row["diff_detected"])
    verify_silent_but_breaking = [
        row["key"] for row in blindness
        if row["diff_breaking"] and not row["verify_breaking"]
    ]

    physics_untouched = all(
        finding["field"].startswith("surface_ledger")
        for finding in bc.diff_contracts(baseline_100["payload"],
                                         json.loads(bc.canonical_payload()))
    )

    verdict_rows = [
        {"metric": "release chain conforms (version + re-freeze + bump discipline)",
         "reference": "no breaking lineage finding", "got": lineage["status"],
         "tolerance": 0, "pass": lineage["status"] == "CONFORMS"},
        {"metric": "lineage checks matched / run", "reference": 1.0,
         "got": lineage["matched"] / lineage["checks"], "tolerance": 0.0,
         "pass": lineage["matched"] == lineage["checks"]},
        {"metric": "declared 1.1.0 bump vs derived bump for the recorded payload diff",
         "reference": 1.0, "got": 1.0 if diff_row["bump_agrees"] else 0.0,
         "tolerance": 0.0, "pass": diff_row["bump_agrees"]},
        {"metric": "declared 1.1.0 change classes vs derived classes",
         "reference": 1.0, "got": 1.0 if diff_row["classes_agree"] else 0.0,
         "tolerance": 0.0, "pass": diff_row["classes_agree"]},
        {"metric": "candidate holes detected in the lineage matrix",
         "reference": 1.0, "got": matrix_score["rate"], "tolerance": 0.0,
         "pass": matrix_score["detected"] == matrix_score["rows"]},
        {"metric": "clean controls reporting CONFORMS", "reference": 1.0,
         "got": control_clean / len(controls), "tolerance": 0.0,
         "pass": control_clean == len(controls)},
        {"metric": "metadata mutations detected by diff_contracts",
         "reference": 1.0, "got": diff_seen / len(blindness), "tolerance": 0.0,
         "pass": diff_seen == len(blindness)},
        {"metric": "metadata mutations detected by the introspection verifier",
         "reference": "baseline (blindness measurement)",
         "got": verify_seen / len(blindness), "tolerance": 1.0,
         "pass": True},
        {"metric": "round-3 unrecorded re-freeze classified as DRIFT",
         "reference": 1.0, "got": 1.0 if unrecorded["status"] == "DRIFT" else 0.0,
         "tolerance": 0.0, "pass": unrecorded["status"] == "DRIFT"},
        {"metric": "current pinned manifest round-trips as CONFORMS",
         "reference": 1.0, "got": 1.0 if clean_manifest["status"] == "CONFORMS" else 0.0,
         "tolerance": 0.0, "pass": clean_manifest["status"] == "CONFORMS"},
        {"metric": "1.0.0 binder manifest degrades to COMPATIBLE_ADDITIVE",
         "reference": 1.0, "got": 1.0 if older_manifest["status"] == "COMPATIBLE_ADDITIVE" else 0.0,
         "tolerance": 0.0, "pass": older_manifest["status"] == "COMPATIBLE_ADDITIVE"},
        {"metric": "physics rows touched by the 1.0.0 -> 1.1.0 payload diff",
         "reference": 0, "got": 0 if physics_untouched else 1, "tolerance": 0,
         "pass": physics_untouched},
    ]

    result = {
        "schema": "tensordem.xval.binding-contract/4",
        "anchor": "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT",
        "round": 4,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owner_tolerance": OWNER_TOLERANCE,
        "contract": {
            "version": bc.CONTRACT_VERSION,
            "schema": bc.CONTRACT_SCHEMA,
            "fingerprint": bc.FROZEN_FINGERPRINT,
            "fingerprint_matches_pin": bc.fingerprint() == bc.FROZEN_FINGERPRINT,
        },
        "lineage": {k: v for k, v in lineage.items() if k != "findings"} | {
            "findings": lineage["findings"]},
        "baseline_diff": diff_row,
        "round3_unrecorded_manifest": {
            "version": round3["contract_version"],
            "fingerprint": round3["fingerprint"],
            "status": unrecorded["status"],
            "findings": [f["change"] for f in unrecorded["findings"]],
        },
        "manifests": {
            "current": {"status": clean_manifest["status"],
                        "required_bump": clean_manifest["required_bump"]},
            "v1.0.0": {"status": older_manifest["status"],
                       "required_bump": older_manifest["required_bump"],
                       "change_classes": older_manifest["change_classes"]},
        },
        "lineage_matrix": matrix,
        "lineage_matrix_score": matrix_score,
        "controls_clean": {"rows": len(controls), "clean": control_clean},
        "blindness_baseline": blindness,
        "blindness_score": {
            "rows": len(blindness),
            "verify_detected": verify_seen,
            "diff_detected": diff_seen,
            "silent_breaking_classes": verify_silent_but_breaking,
        },
        "verdict_rows": verdict_rows,
        "verdict": "MET" if all(row["pass"] for row in verdict_rows) else "FAIL",
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "wall_clock_s": None,
        },
    }
    result["environment"]["wall_clock_s"] = round(time.time() - started, 2)
    (outdir / "result_r4.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verdict", "lineage_matrix_score",
                                             "blindness_score", "controls_clean")},
                     indent=2, sort_keys=True))
    print(f"wrote {outdir / 'result_r4.json'}")

    if args.inject_doc:
        _inject_doc(result, Path(args.inject_doc))
    try:
        _figures(result, outdir)
    except Exception as exc:  # pragma: no cover - figures are best effort
        print(f"figures skipped: {exc}", file=sys.stderr)
    return 0


def _figures(result: dict, outdir: Path) -> None:  # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. detector blindness: verify() vs diff_contracts() per metadata mutation
    rows = result["blindness_baseline"]
    labels = [row["key"].replace("_", " ") for row in rows]
    verify_scores = [row["verify_breaking"] for row in rows]
    diff_scores = [row["diff_breaking"] for row in rows]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(rows) + 2))
    ax.barh([i + 0.2 for i in y], diff_scores, height=0.38,
            label="diff_contracts (round 4)", color="#1b6ca8")
    ax.barh([i - 0.2 for i in y], verify_scores, height=0.38,
            label="verify (rounds 1-3)", color="#c94f4f")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("breaking findings raised by the detector")
    ax.set_title("Contract-metadata edits: what the introspection verifier cannot see")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_detector_blindness.png", dpi=160)
    plt.close(fig)

    # 2. lineage matrix: candidate holes vs detection
    matrix = result["lineage_matrix"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(matrix) + 2))
    colors = ["#2c7a3f" if row["detected"] else "#b3341f" for row in matrix]
    ax.barh(range(len(matrix)),
            [1 if row["detected"] else 0 for row in matrix], color=colors)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([f"{row['kind']}:{row['key']}" for row in matrix], fontsize=8)
    ax.set_xlim(0, 1.15)
    for i, row in enumerate(matrix):
        ax.text(1.02, i, "detected" if row["detected"] else "MISSED",
                va="center", fontsize=7,
                color="#2c7a3f" if row["detected"] else "#b3341f")
    ax.set_xticks([0, 1])
    ax.set_xlabel("expected finding present (1) / absent (0)")
    ax.set_title("Lineage + manifest mutants: every hole must be detected")
    fig.tight_layout()
    fig.savefig(outdir / "fig_lineage_matrix.png", dpi=160)
    plt.close(fig)

    # 3. release chain: versions, fingerprints, severity budget
    history = bc.CONTRACT_HISTORY
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    versions = [entry["version"] for entry in history]
    budget = {"doc": 0, "additive": 1, "narrowing": 1, "breaking": 2}
    severity_points = [max(budget[bc.CHANGE_SEVERITY[c]] for c in entry["changes"])
                       for entry in history]
    ax1.plot(versions, severity_points, marker="o", color="#1b6ca8")
    for version, severity, entry in zip(versions, severity_points, history):
        ax1.annotate(f"{entry['bump']}\n{entry['fingerprint'][:14]}...",
                     (version, severity), textcoords="offset points",
                     xytext=(6, 8), fontsize=7)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["doc", "additive/narrowing", "breaking"], fontsize=8)
    ax1.set_title("Recorded release chain (tail must equal the pin)")
    ax1.grid(alpha=0.3)

    counts = result["baseline_diff"]["by_severity"]
    ax2.bar(list(counts), list(counts.values()),
            color=["#8a8a8a", "#4f8fbf", "#d0a24c", "#b3341f"])
    ax2.set_title("1.0.0 -> 1.1.0 payload diff\n(severity classes)", fontsize=9)
    ax2.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_release_chain.png", dpi=160)
    plt.close(fig)
    print(f"figures -> {outdir}")




BEGIN = "<!-- R4-RESULTS:BEGIN -->"
END = "<!-- R4-RESULTS:END -->"


def _inject_doc(result: dict, doc: Path) -> None:  # pragma: no cover - doc plumbing
    """Write the measured tables into the doc between the R4 markers."""
    contract = result["contract"]
    diff = result["baseline_diff"]
    lines: list[str] = [BEGIN,
                        "<!-- generated by scripts/xval_library_binding_contract_r4.py; do not hand-edit -->",
                        ""]
    lines.append(f"- contract: `{contract['version']}` / `{contract['fingerprint']}` "
                 f"(pin match = {contract['fingerprint_matches_pin']}, schema `{contract['schema']}`)")
    lineage = result["lineage"]
    lines.append(f"- release chain: {lineage['entries']} entries, {lineage['checks']} checks, "
                 f"matched {lineage['matched']}, status = **{lineage['status']}**")
    recorded = result["round3_unrecorded_manifest"]
    lines.append(f"- round-3 unrecorded re-freeze `{recorded['fingerprint']}` (declared version "
                 f"`{recorded['version']}`): consumer check = **{recorded['status']}** "
                 f"({', '.join(recorded['findings'])})")
    lines.append(f"- binder manifests: current `{result['manifests']['current']['status']}`, "
                 f"v1.0.0 `{result['manifests']['v1.0.0']['status']}` "
                 f"(required bump `{result['manifests']['v1.0.0']['required_bump']}`)")
    env = result["environment"]
    lines.append(f"- wall clock: {env['wall_clock_s']} s on {env['machine']}, "
                 f"python {env['python']} ({env['platform']})")
    lines.append("")
    lines.append("### Release chain (append-only)")
    lines.append("")
    lines.append("| version | fingerprint | bump | derived change classes | status |")
    lines.append("|---|---|---|---|---|")
    for entry in bc.CONTRACT_HISTORY:
        lines.append(f"| `{entry['version']}` | `{entry['fingerprint']}` | `{entry['bump']}` | "
                     f"{', '.join('`' + c + '`' for c in entry['changes'])} | recorded |")
    lines.append("")
    lines.append("### 1.0.0 -> 1.1.0 payload diff (derived, not declared)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| findings | {diff['findings']} |")
    for severity, count in diff["by_severity"].items():
        lines.append(f"| severity `{severity}` | {count} |")
    lines.append(f"| change classes | {', '.join('`' + c + '`' for c in diff['change_classes']) or '-'} |")
    lines.append(f"| required bump (derived) | `{diff['required_bump']}` |")
    lines.append(f"| declared bump (history tail) | `{diff['declared_bump']}` |")
    lines.append(f"| declared == derived bump | {diff['bump_agrees']} |")
    lines.append(f"| declared == derived change classes | {diff['classes_agree']} |")
    lines.append("")
    lines.append("### Detector blindness baseline: 10 metadata mutations")
    lines.append("")
    lines.append("| mutation | verify() breaking findings (rounds 1-3) | diff_contracts() classes (round 4) |")
    lines.append("|---|---|---|")
    for row in result["blindness_baseline"]:
        classes = ", ".join(f"`{c}`" for c in row["diff_change_classes"])
        lines.append(f"| `{row['key']}` | {row['verify_breaking']} | {classes} |")
    score = result["blindness_score"]
    lines.append(f"| **totals** | **{score['verify_detected']}/{score['rows']}** | "
                 f"**{score['diff_detected']}/{score['rows']}** |")
    lines.append("")
    lines.append(f"旧检查集看不出的 breaking 类（{len(score['silent_breaking_classes'])} 个）："
                 + "、".join(f"`{k}`" for k in score["silent_breaking_classes"]))
    lines.append("")
    lines.append("### Lineage / manifest mutant matrix")
    lines.append("")
    lines.append("| kind | mutant | expected finding | observed | detected | status |")
    lines.append("|---|---|---|---|---|---|")
    for row in result["lineage_matrix"]:
        observed = ", ".join(f"`{c}`" for c in row["observed"]) or "-"
        lines.append(f"| `{row['kind']}` | `{row['key']}` | "
                     f"{'`' + row['expect_change'] + '`' if row['expect_change'] else '<none>'} | "
                     f"{observed} | {row['detected']} | `{row['status']}` |")
    lines.append("")
    lines.append("### Verdict rows")
    lines.append("")
    lines.append("| metric | reference | got | tolerance | verdict |")
    lines.append("|---|---|---|---|---|")
    for row in result["verdict_rows"]:
        lines.append(f"| {row['metric']} | {row['reference']} | {row['got']} | "
                     f"{row['tolerance']} | {'MET' if row['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"overall verdict: **{result['verdict']}**")
    lines.append("")
    lines.append(END)

    text = doc.read_text(encoding="utf-8")
    start = text.index(BEGIN)
    stop = text.index(END) + len(END)
    doc.write_text(text[:start] + "\n".join(lines) + text[stop:], encoding="utf-8")
    print(f"doc blocks -> {doc}")


if __name__ == "__main__":
    raise SystemExit(main())