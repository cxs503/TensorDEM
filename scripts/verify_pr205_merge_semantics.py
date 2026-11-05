"""Machine check for the PR #205 cure: did the merge of ``main`` into
``agent/m3386`` preserve *both* sides, file by file?

A merge conflict resolved by eyeball is not evidence.  The failure mode this repo
has actually suffered is a **silent overlay**: the resolver takes one side wholesale
and the other side's additions (tests, docs, guards) disappear without a marker,
because git reports no conflict for files only one side touched.  So this audit
treats the merge commit as a claim to be falsified:

R1  every path either side changed since the merge base must exist in the merge
    result exactly as that side left it — unless both sides changed it, in which
    case it must appear in the explicit ``RESOLUTIONS`` table below (no implicit
    forgiveness, and an unlisted overlap is a hard failure);
R2  the merge result must introduce no path that neither side changed (a resolver
    that "tidies up" is inventing content);
R3  a declared union resolution must satisfy named, textual invariants that encode
    *what each side contributed*, so dropping either side fails loudly;
R4  the resulting ``forces()`` overrides must accept the probe keyword, checked at
    import time against the real classes rather than against the diff.

Blob identity comes from ``git rev-parse <rev>:<path>``, so the comparison is over
the exact git objects, not over re-read text.

Usage::

    PYTHONPATH=src python3 scripts/verify_pr205_merge_semantics.py \
        [--main origin/main] [--merged 0d76f4c] [--branch <pre-merge tip>]

Writes ``artifacts/pr205/verify_pr205.json`` plus two figures and exits non-zero on
any failed check.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

OUT = Path("artifacts/pr205")

# A path both sides changed since the merge base must be declared here; the value
# names the invariant set that proves each side survived.
RESOLUTIONS = {
    "src/tensordem/dem3d_wedge.py": "forces_signature_union",
}


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(("git",) + args, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def blob(rev: str, path: str) -> str | None:
    """Blob id of ``path`` at ``rev``; None when the path is absent there."""
    out = git("rev-parse", "--verify", "--quiet", f"{rev}:{path}", check=False).strip()
    return out or None


def changed_paths(base: str, rev: str) -> set[str]:
    raw = git("diff", "--name-only", "--no-renames", base, rev)
    return {line for line in raw.splitlines() if line}


def file_text(rev: str, path: str) -> str:
    return git("show", f"{rev}:{path}")


def check_forces_signature_union(rev: str, path: str) -> list[dict]:
    """Invariants that pin down the semantic union inside MovingWedgeBowIceDEM3D."""
    text = file_text(rev, path)
    ast.parse(text)  # the resolver must not leave a syntactically broken file
    head = text[text.index("class MovingWedgeBowIceDEM3D"):]
    forces_body = head[head.index("def forces("):head.index("def diagnostics(")]
    sig = forces_body[:forces_body.index(") ->")]

    checks = [
        # main's contribution: the keyword-only switch exists on the override ...
        ("main: update_history kwarg declared on the override",
         sig.count("update_history: bool = True") == 1),
        # ... and is actually forwarded to the inherited implementation.
        ("main: update_history forwarded to super().forces",
         forces_body.count("update_history=update_history") == 1),
        ("main: update_fracture still forwarded",
         forces_body.count("update_fracture=update_fracture") == 1),
        # this branch's contribution: a probe must not commit the reaction ...
        ("branch: probe guard conditional on update_fracture",
         "if update_fracture:" in forces_body),
        # ... and must commit in place, so the preallocated tensor keeps identity.
        ("branch: reaction committed in place via copy_",
         "self.last_reaction.copy_(bow_reaction)" in forces_body),
        # main's rebinding is precisely the side effect the branch removed.
        ("branch: unconditional rebinding of last_reaction is gone",
         "self.last_reaction = bow_reaction" not in forces_body),
    ]
    return [{"check": name, "ok": bool(ok)} for name, ok in checks]


INVARIANTS = {"forces_signature_union": check_forces_signature_union}


def import_time_signature_check() -> list[dict]:
    """R4: the merged tree must import and the overrides must accept the probe."""
    import inspect

    import tensordem.dem3d as dem3d
    import tensordem.dem3d_wedge as wedge
    import tensordem.rudder_ice_load as rudder

    results = []
    for cls in (dem3d.IceDEM3D, wedge.MovingWedgeBowIceDEM3D, rudder.RudderIceLoadDEM):
        params = inspect.signature(cls.forces).parameters
        results.append({
            "check": f"import-time: {cls.__name__}.forces accepts update_history",
            "ok": "update_history" in params and "update_fracture" in params,
        })
    return results


def provenance_table(main: str, branch: str, merged: str, base: str) -> list[dict]:
    main_changed = changed_paths(base, main)
    branch_changed = changed_paths(base, branch)
    rows = []
    for path in sorted(main_changed | branch_changed):
        b_main, b_branch, b_merged = blob(main, path), blob(branch, path), blob(merged, path)
        if path in main_changed and path in branch_changed:
            side = "both"
        elif path in main_changed:
            side = "main"
        else:
            side = "branch"
        rows.append({
            "path": path,
            "changed_by": side,
            "present_in_merged": b_merged is not None,
            "identical_to_main": b_main == b_merged,
            "identical_to_branch": b_branch == b_merged,
            "blobs": {"base": blob(base, path), "main": b_main, "branch": b_branch,
                      "merged": b_merged},
        })
    return rows


def audit(main: str, branch: str, merged: str) -> dict:
    base = git("merge-base", main, branch).strip()
    rows = provenance_table(main, branch, merged, base)

    checks: list[dict] = []
    for row in rows:
        path, side = row["path"], row["changed_by"]
        if side == "both":
            if row["identical_to_main"] and row["identical_to_branch"]:
                outcome = "agreed"  # both sides made the same change; nothing to merge
            elif path in RESOLUTIONS:
                outcome = "union"
                for res in INVARIANTS[RESOLUTIONS[path]](merged, path):
                    checks.append({**res, "path": path})
            else:
                outcome = "UNRESOLVED-OVERLAP"
        elif side == "main":
            outcome = "kept-main" if row["identical_to_main"] else "LOST-MAIN-CHANGE"
        else:
            outcome = "kept-branch" if row["identical_to_branch"] else "LOST-BRANCH-CHANGE"
        row["outcome"] = outcome
        checks.append({
            "path": path,
            "check": f"R1 {side} side of {path} preserved ({outcome})",
            "ok": outcome in ("agreed", "union", "kept-main", "kept-branch"),
        })
        # R1 also covers deletions: a side that removed the file must see it absent.
        for name, rev_blob in (("main", row["blobs"]["main"]), ("branch", row["blobs"]["branch"])):
            if rev_blob is None and side in (name, "both"):
                checks.append({
                    "path": path,
                    "check": f"R1 {name}'s deletion of {path} honoured",
                    "ok": row["blobs"]["merged"] is None,
                })

    # R2: nothing foreign may appear in the merge result.
    introduced = changed_paths(base, merged) - {row["path"] for row in rows}
    checks.append({
        "path": "-",
        "check": "R2 merge introduces no path that neither side changed",
        "ok": not introduced,
    })

    for res in import_time_signature_check():
        checks.append({"path": "src/tensordem/", **res})

    return {
        "merge": {"merge_base": base, "main": main, "branch": branch, "merged": merged},
        "files": rows,
        "checks": checks,
        "failed": [c for c in checks if not c["ok"]],
        "summary": {
            "files_compared": len(rows),
            "outcomes": {
                key: sum(1 for r in rows if r["outcome"] == key)
                for key in sorted({r["outcome"] for r in rows})
            },
            "checks": len(checks),
            "checks_failed": sum(1 for c in checks if not c["ok"]),
        },
    }


def plot(report: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    produced = []

    # Figure 1: provenance of every path the merge had to preserve.
    files = [f for f in report["files"] if f["outcome"] != "UNRESOLVED-OVERLAP"]
    files.sort(key=lambda f: ({"both": 0, "branch": 1, "main": 2}[f["changed_by"]], f["path"]))
    colours = {"agreed": "#2a9d8f", "union": "#264653", "kept-main": "#457b9d",
               "kept-branch": "#e76f51"}
    labels = {"agreed": "both sides, identical", "union": "both sides, resolved union",
              "kept-main": "main side, preserved verbatim",
              "kept-branch": "branch side, preserved verbatim"}
    fig, ax = plt.subplots(figsize=(9.0, 0.24 * len(files) + 1.6))
    ys = range(len(files))
    ax.barh(list(ys), [1] * len(files),
            color=[colours[f["outcome"]] for f in files], height=0.72)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f["path"] for f in files], fontsize=6.5)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in labels]
    ax.legend(handles, [labels[k] for k in labels], fontsize=7, loc="lower right",
              frameon=True)
    ax.set_title(f"PR #205 cure: provenance of the {len(files)} paths changed on either side\n"
                 f"0 files lost; {report['summary']['checks']} checks, "
                 f"{report['summary']['checks_failed']} failed", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = OUT / "pr205_merge_provenance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    produced.append(path)

    # Figure 2: the construction that discriminates the union's invariant — the
    # measured bow reaction rises with engagement while the committed reaction of a
    # probe stays frozen.  Data comes from the producer script's CSV, not from here.
    csv = Path("artifacts/forces_override_contract/probe_invariance.csv")
    if csv.exists():
        rows = [line.strip().split(",") for line in csv.read_text().splitlines() if line.strip()]
        header, data = rows[0], [[float(v) for v in row] for row in rows[1:]]
        col = {name: i for i, name in enumerate(header)}
        x = [d[col["bow_advance_m"]] * 1000.0 for d in data]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(x, [d[col["probe_reaction_x_N"]] for d in data], "o-",
                label="probe measures (update_fracture=False)")
        ax.plot(x, [d[col["committed_reaction_x_N"]] for d in data], "s--",
                label="committed by the owning call")
        ax.plot(x, [d[col["stored_after_probe_x_N"]] for d in data], "^:",
                label="stored last_reaction after the probe")
        drift = max(abs(d[col["commit_drift_N"]]) for d in data)
        ax.set_xlabel("bow advance before the probe  [mm]")
        ax.set_ylabel("reaction on x  [N]")
        ax.set_title("Probe must measure without committing\n"
                     f"max drift of the stored reaction over {len(data)} probes: {drift:.1e} N",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = OUT / "pr205_probe_invariance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        produced.append(path)

    return produced


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default="origin/main")
    ap.add_argument("--branch", default=None,
                    help="branch tip before the merge (default: <merged>^1)")
    ap.add_argument("--merged", default="HEAD")
    ap.add_argument("--figure", action="store_true", help="also render the figures")
    args = ap.parse_args(argv)
    # The merge commit's first parent *is* the branch tip, so the audit stays
    # reproducible after the branch has been pushed and no longer points here.
    branch = args.branch or f"{args.merged}^1"

    report = audit(args.main, branch, args.merged)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.figure:
        report["figures"] = [str(p) for p in plot(report)]
    (OUT / "verify_pr205.json").write_text(json.dumps(report, indent=2) + "\n")

    for row in report["files"]:
        print(f"{row['outcome']:<20} {row['path']}")
    print(f"\nfiles compared {report['summary']['files_compared']}  "
          f"checks {report['summary']['checks']}  "
          f"failed {report['summary']['checks_failed']}")
    for bad in report["failed"]:
        print(f"FAILED {bad['path']}: {bad['check']}", file=sys.stderr)
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())