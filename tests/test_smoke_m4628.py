"""E2E pipeline smoke check for mission m4628.

Verifies that the repository root is importable from the test process and that
the smoke marker module carries the expected mission constant.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smoke_marker_m4628


def test_marker_constant_value():
    assert smoke_marker_m4628.PIPELINE_SMOKE_M4628 == "4628"


def test_sanity_arithmetic():
    assert 1 + 1 == 2