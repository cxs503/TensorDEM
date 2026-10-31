"""Test configuration: always import the in-tree ``src`` package.

The container also carries a site-packages copy of ``tensordem`` built from a
different work tree; the tests in this repository must exercise the sources of
this checkout, so ``src`` is prepended to ``sys.path`` before collection.
"""
from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))