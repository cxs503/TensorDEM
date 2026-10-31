"""Pytest bootstrap: always test the sources of *this* working tree.

The package is also installed editable from another checkout, so the shared
``site-packages`` path would otherwise shadow the local ``src`` tree.  Keeping
this explicit makes the suite worktree-local and reproducible.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    text = str(candidate)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)