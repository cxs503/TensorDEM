"""Erratum pin: docs/xval_jiang2005_rolling_spring.md frequency formula.

The m17413 note carried a dimensionally inconsistent oscillation-frequency
formula ``omega = sqrt((1 + 1/f) * k_r / m)`` (units of m/s, not 1/s); the
m18522 audit (docs/xval_jiang2005_kr_rule_audit.md section 5.3) flagged it and
m20524 corrected the doc and the harness script.  These tests pin the
correction so it cannot silently regress:

* the doc and the script no longer contain the superseded formula;
* the doc's section-3.3 table values equal an *independently* evaluated
  closed form ``sqrt(k_r / (m R^2 (1 + f)))`` (the test recomputes every cell
  from first principles -- it does not trust the doc or the library);
* the two-rule ratio in the table is ``sqrt(mu_r)``.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "xval_jiang2005_rolling_spring.md"
SCRIPT = ROOT / "scripts" / "xval_jiang2005_rolling_spring.py"

# Harness parameters of the m17413 harness (must match the script constants).
K_N = 2.0e4
RADIUS = 0.01
DENSITY = 900.0
F_INERTIA = 0.4  # I/(mR^2), solid sphere
MU_R_VALUES = [0.05, 0.10, 0.20, 0.30, 0.50]


def _mass() -> float:
    return 4.0 / 3.0 * math.pi * RADIUS ** 3 * DENSITY


def _k_r(mu_r: float, rule: str) -> float:
    power = 1.0 if rule == "tensordem" else 2.0
    return 2.25 * (mu_r ** power) * K_N * RADIUS ** 2


def _freq_hz(mu_r: float, rule: str) -> float:
    """Closed form, evaluated directly here (no library import)."""
    k_r = _k_r(mu_r, rule)
    omega = math.sqrt(k_r / (_mass() * RADIUS ** 2 * (1.0 + F_INERTIA)))
    return omega / (2.0 * math.pi)


def _parse_table() -> list[tuple[float, float, float, float, float]]:
    """Parse the section-3.3 markdown table into (mu_r, td, jg, ratio, sqrt_mu)."""
    text = DOC.read_text(encoding="utf-8")
    m = re.search(r"### 3\.3 Transient Dynamics\n(.*?)\n\n", text, re.S)
    assert m, "section 3.3 table not found in the doc"
    rows = []
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0] or cells[0][0] in "- ":
            continue
        try:
            rows.append(tuple(float(c) for c in cells))
        except ValueError:
            continue
    assert len(rows) == len(MU_R_VALUES), f"expected {len(MU_R_VALUES)} rows, got {len(rows)}"
    return rows


def test_superseded_formula_is_gone_from_doc_and_script():
    doc = DOC.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    # The wrong formula, in any of its spellings, must not survive.
    for blob in (doc, script):
        assert "1 + 1/f" not in blob.replace(" ", ""), (
            "superseded formula sqrt((1+1/f) k_r / m) is back"
        )
    # The corrected formula must be present in both.
    assert "k_r / (m R² (1 + f))" in doc
    assert "k_r_val / (mass * RADIUS ** 2 * (1.0 + f_inertia))" in script


def test_doc_table_matches_independent_closed_form():
    rows = _parse_table()
    for (mu_r, td, jg, ratio, sqrt_mu), mu in zip(rows, MU_R_VALUES):
        assert mu_r == pytest.approx(mu, abs=1e-9)
        f_td = _freq_hz(mu, "tensordem")
        f_jg = _freq_hz(mu, "jiang2005")
        assert td == pytest.approx(f_td, rel=5e-3), (mu_r, td, f_td)
        assert jg == pytest.approx(f_jg, rel=5e-3), (mu_r, jg, f_jg)
        assert ratio == pytest.approx(math.sqrt(mu), rel=5e-3)
        assert sqrt_mu == pytest.approx(math.sqrt(mu), rel=5e-3)


def test_corrected_values_are_not_the_old_wrong_ones():
    """Guard against a half-fix: the retired values must not reappear as data."""
    rows = _parse_table()
    old_td = {0.05: 2.30, 0.10: 3.25, 0.20: 4.60, 0.30: 5.63, 0.50: 7.27}
    for (mu_r, td, *_), mu in zip(rows, MU_R_VALUES):
        assert td != pytest.approx(old_td[mu], rel=0.05), (
            f"mu_r={mu}: doc still carries the pre-erratum frequency"
        )


def test_old_formula_dimensional_inconsistency_is_real():
    """The superseded formula has units of m/s: for R=0.01 m it is off by
    1/(R*sqrt(4.9)) ~ 45.2x, i.e. far outside the 15% band."""
    mu = 0.2
    k_r = _k_r(mu, "tensordem")
    m = _mass()
    wrong = math.sqrt((1.0 + 1.0 / F_INERTIA) * k_r / m) / (2.0 * math.pi)
    right = _freq_hz(mu, "tensordem")
    factor = right / wrong
    assert factor == pytest.approx(1.0 / (RADIUS * math.sqrt(4.9)), rel=1e-12)
    assert abs(factor - 1.0) > 0.15  # far outside the owner's 15% red line
