"""Freeze the public API of the modules the contact-physics layer delegates to.

Run once, **before** the integration layer lands, so the committed baseline
records the pre-integration signatures of the upstream node modules.  The test
``tests/test_contact_physics_integration.py::test_public_signatures_are_unchanged``
then fails if any of them moves.

Fields recorded per module
--------------------------
``path``        repository-relative path of the source file
``sha256``      content hash (byte-identical check, catches body edits)
``signatures``  ``inspect.signature`` string of every public name
``git_diff_vs_head``  diff of the file against ``HEAD`` at freeze time (must be
                empty: the baseline is the pre-integration state)

Usage
-----
``python3 scripts/freeze_contact_physics_api.py``
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Modules whose behaviour the contact-physics adapters delegate to.
MODULES = (
    "tensordem.ice_bond_viscoelastic",                 # N02 bond law
    "tensordem.wet_ice_cohesion.ice_resistance",       # N02 temperature relation
    "tensordem.interface_thermal.lubrication",         # N08 friction / film
    "tensordem.interface_thermal.friction_heat",       # N08 flux / partition
    "tensordem.interface_thermal.coupling",            # N08 coupled solver
    "tensordem.ridge_dem",                             # N05 keel realisation
    "tensordem.ice_rubble_microstructure",             # N05 statistics
)

OUT = ROOT / "artifacts" / "contact_physics" / "api_baseline.json"


def public_names(module) -> list[str]:
    """``__all__`` when the module declares it, else public defs / classes."""
    declared = getattr(module, "__all__", None)
    if declared:
        return sorted(declared)
    names = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value) or inspect.isclass(value):
            if getattr(value, "__module__", None) == module.__name__:
                names.append(name)
    return sorted(names)


def main() -> int:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    modules = {}
    for name in MODULES:
        module = __import__(name, fromlist=["*"])
        relative = pathlib.Path("src", *name.split(".")).with_suffix(".py")
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        signatures = {}
        for public in public_names(module):
            target = getattr(module, public)
            try:
                signatures[public] = str(inspect.signature(target))
            except (TypeError, ValueError):  # pragma: no cover - odd descriptors
                signatures[public] = "<no signature>"
        diff = subprocess.run(
            ["git", "diff", "--", str(path.relative_to(ROOT))], cwd=ROOT,
            capture_output=True, text=True).stdout
        modules[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "signatures": signatures,
            "git_diff_vs_head": diff.strip(),
        }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_git_rev": rev,
        "purpose": ("pre-integration public API of the P121W5-N02/N05/N08 "
                    "modules, frozen for the zero-signature-change acceptance "
                    "criterion of node P121W5-N11"),
        "modules": modules,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    dirty = [name for name, record in modules.items() if record["git_diff_vs_head"]]
    print(f"wrote {OUT.relative_to(ROOT)} ({len(modules)} modules, rev {rev[:12]})")
    if dirty:
        print("WARNING: source modules already differ from HEAD:", dirty)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())