"""Make the tests import the package from this checkout.

Several worktrees of this repository can share one Python environment, and the
installed ``tensordem`` may therefore point at a different checkout.  Prepending
this repository's ``src`` directory guarantees that the tests exercise the code
in the tree they belong to.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))