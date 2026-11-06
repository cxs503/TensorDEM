"""Executable gate for ``docs/tutorials/first_verified_run.md``.

The tutorial is documentation, but every ``python`` block in it is one sequential
part of a single runnable script *with assertions*.  This module extracts those
blocks in document order and executes them, so the tutorial cannot rot into prose
that no longer runs.

Two anti-vacuity controls keep the gate honest:

* the block/assertion census proves the extracted source really contains the
  checks the tutorial advertises (a tutorial stripped of its assertions fails);
* parametrised *negative controls* mutate the extracted source -- wrong bond
  count, wrong stiffness tolerance, wrong anisotropy ratio -- and require the
  corresponding assertion to fire, proving the checks are load bearing.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "tutorials" / "first_verified_run.md"
README = ROOT / "README.md"
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)

#: ``(old, new, why)`` -- each mutation must make the tutorial's own assertions fail.
NEGATIVE_MUTATIONS = (
    (
        "assert int(sim.alive.sum()) == hand_bonds",
        "assert int(sim.alive.sum()) == hand_bonds + 1",
        "topology: engine bond count off by one",
    ),
    (
        "assert rel_err < 1e-5, f\"引擎键能与 Cauchy–Born 不符：{rel_err}\"",
        "assert rel_err < 1e-15, f\"引擎键能与 Cauchy–Born 不符：{rel_err}\"",
        "elasticity: tolerance tightened past the finite-difference error",
    ),
    (
        "assert math.isclose(young_45 / young_0, 1.6, rel_tol=1e-12)",
        "assert math.isclose(young_45 / young_0, 1.0, rel_tol=1e-12)",
        "anisotropy: pretend the square lattice is isotropic",
    ),
    (
        "assert 0.45 < fine / coarse < 0.55, fine / coarse",
        "assert 0.90 < fine / coarse < 1.10, fine / coarse",
        "energy ledger: demand zero-order residual (would hide a non-converging bug)",
    ),
)


def tutorial_blocks() -> list[str]:
    """The tutorial's fenced ``python`` blocks, in document order."""
    text = TUTORIAL.read_text(encoding="utf-8")
    blocks = [block for block in BLOCK_RE.findall(text) if block.strip()]
    assert blocks, f"{TUTORIAL} has no executable python blocks"
    return blocks


def tutorial_source() -> str:
    """All python blocks concatenated into the single script they represent."""
    return "\n".join(tutorial_blocks())


def run_tutorial(source: str) -> dict[str, object]:
    """Execute the tutorial script and return its final namespace."""
    namespace: dict[str, object] = {"__name__": "__tutorial__"}
    exec(compile(source, str(TUTORIAL), "exec"), namespace)  # noqa: S102 - the point of the test
    return namespace


def test_tutorial_blocks_are_valid_python() -> None:
    blocks = tutorial_blocks()
    assert len(blocks) >= 6, f"expected the multi-step tutorial, found {len(blocks)} blocks"
    for index, block in enumerate(blocks):
        ast.parse(block, filename=f"{TUTORIAL}#block{index}")


def test_tutorial_assertion_census() -> None:
    """The tutorial must keep a large, visible set of executable checks."""
    tree = ast.parse(tutorial_source(), filename=str(TUTORIAL))
    asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    messages = sum(1 for node in asserts if node.msg is not None)
    assert len(asserts) >= 25, f"tutorial lost assertions: only {len(asserts)}"
    assert messages >= 3, "tutorial asserts should explain what broke"


def test_readme_links_the_tutorial() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "docs/tutorials/first_verified_run.md" in readme, "README does not link the tutorial"


def test_tutorial_runs_green() -> None:
    namespace = run_tutorial(tutorial_source())
    # Spot-check that the run really reached the end and produced the calibrated numbers.
    assert namespace["rel_err"] < 1e-5
    assert namespace["run_impact"].broken == 0
    assert namespace["run_impact"].reaction > 0.0


@pytest.mark.parametrize(("old", "new", "why"), NEGATIVE_MUTATIONS, ids=[m[2] for m in NEGATIVE_MUTATIONS])
def test_injected_error_turns_tutorial_red(old: str, new: str, why: str) -> None:
    source = tutorial_source()
    assert old in source, f"negative control is stale, pattern not found: {why}"
    mutated = source.replace(old, new, 1)
    assert mutated != source
    with pytest.raises(AssertionError):
        run_tutorial(mutated)


# ---------------------------------------------------------------------------
# Evidence gates: the tutorial's *products* and its *prose numbers* are pinned.
#
# The executed-block gate above proves the code runs; it does not prove that the
# quantities the tutorial advertises in words or in
# ``artifacts/tutorial_first_verified_run/`` still belong to the current code.
# These gates close that hole: the committed summary/log must reproduce from a
# fresh run, and every numeric claim in the prose must match the executed value.
# ---------------------------------------------------------------------------

ARTIFACTS = ROOT / "artifacts" / "tutorial_first_verified_run"
SUMMARY = ARTIFACTS / "summary.json"
RUN_LOG = ARTIFACTS / "run_log.txt"
FLOAT_REL_TOL = 1e-9


@lru_cache(maxsize=1)
def collected() -> tuple[dict[str, object], str, dict[str, object]]:
    """Run the tutorial once through the collector (same namespace, plus stdout)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_tutorial_first_verified_run as collector  # noqa: PLC0415

    namespace, log = collector.run()
    return namespace, log, collector.summary(namespace, log)


def _assert_close(committed, fresh, where: str) -> None:
    if isinstance(committed, dict):
        assert set(committed) == set(fresh), f"{where}: key set drifted: {set(committed) ^ set(fresh)}"
        for key in sorted(committed):
            _assert_close(committed[key], fresh[key], f"{where}.{key}")
        return
    if isinstance(committed, list):
        assert len(committed) == len(fresh), f"{where}: length drifted"
        for index, (left, right) in enumerate(zip(committed, fresh)):
            _assert_close(left, right, f"{where}[{index}]")
        return
    if isinstance(committed, str):
        assert committed == fresh, f"{where}: {committed!r} != {fresh!r}"
        return
    assert math.isclose(float(committed), float(fresh), rel_tol=FLOAT_REL_TOL), (
        f"{where}: committed {committed!r} != fresh run {fresh!r}"
    )


def test_committed_summary_matches_fresh_run() -> None:
    """``summary.json`` is a *generated* baseline; it must equal a fresh run."""
    assert SUMMARY.exists(), f"missing generated artifact {SUMMARY}; run scripts/collect_tutorial_first_verified_run.py"
    committed = json.loads(SUMMARY.read_text(encoding="utf-8"))
    _, _, fresh = collected()
    for section in ("parameters", "static_checks", "calibration"):
        _assert_close(committed[section], fresh[section], section)
    # ``dynamic_checks`` gains the extra dt-refinement sweep during figure generation.
    committed_dyn = {key: value for key, value in committed["dynamic_checks"].items() if not isinstance(value, dict)}
    fresh_dyn = {key: value for key, value in fresh["dynamic_checks"].items() if not isinstance(value, dict)}
    _assert_close(committed_dyn, fresh_dyn, "dynamic_checks")
    _assert_close(committed["tutorial"], fresh["tutorial"], "tutorial")


def test_committed_run_log_matches_fresh_run() -> None:
    assert RUN_LOG.exists(), f"missing generated artifact {RUN_LOG}; run scripts/collect_tutorial_first_verified_run.py"
    _, log, _ = collected()
    committed = RUN_LOG.read_text(encoding="utf-8").rstrip("\n").splitlines()
    assert committed == log.rstrip("\n").splitlines()


def test_committed_figure_and_convergence_records_are_self_consistent() -> None:
    committed = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert committed["figures"] == [
        "artifacts/tutorial_first_verified_run/figures/directional_modulus.png",
        "artifacts/tutorial_first_verified_run/figures/energy_ledger_convergence.png",
    ], committed["figures"]
    sweep = committed["dynamic_checks"]["dt_refinement"]
    assert len(sweep["dt"]) == len(sweep["residual"]) == 4
    for coarse_dt, fine_dt in zip(sweep["dt"], sweep["dt"][1:]):
        assert math.isclose(fine_dt, coarse_dt / 2.0, rel_tol=1e-9)
    assert all(residual > 0.0 for residual in sweep["residual"])
    for coarse, fine in zip(sweep["residual"], sweep["residual"][1:]):
        assert 0.45 < fine / coarse < 0.55, f"ledger residual is not O(dt): {fine / coarse}"


def test_axis_bond_stiffness_calibration_matches_isotropic_lattice() -> None:
    """The recorded ``(k_axis, k_diag)`` pair must reproduce the target modulus exactly."""
    namespace, _, payload = collected()
    calibration = payload["calibration"]
    axis = calibration["axis_bond_stiffness_N_per_m"]
    diagonal = calibration["diagonal_bond_stiffness_N_per_m"]
    young = calibration["young_modulus_Pa"]
    assert math.isclose(diagonal, calibration["diagonal_to_axis_ratio"] * axis, rel_tol=1e-12)
    lattice = namespace["le"]
    thickness = float(namespace["THICKNESS"])
    # The documented pair, fed back to the lattice module, is 9.5 GPa in *every* direction.
    for row in lattice.directional_moduli([0.0, 22.5, 45.0, 90.0], axis, lattice.ISOTROPIC_DIAGONAL_SCALE):
        assert math.isclose(row["youngs_modulus_N_per_m"] / thickness, young, rel_tol=1e-9), row
    # The factor-2 trap the tutorial warns about: 2*k_axis is 19 GPa, k_axis/2 is 4.75 GPa.
    for factor in (2.0, 0.5):
        row = lattice.directional_moduli([0.0], factor * axis, lattice.ISOTROPIC_DIAGONAL_SCALE)[0]
        assert math.isclose(row["youngs_modulus_N_per_m"] / thickness, factor * young, rel_tol=1e-9)
    # Skipping the isotropic correction leaves the 60% 0 deg / 45 deg anisotropy in place.
    uncorrected = lattice.directional_moduli([0.0, 45.0], axis, lattice.DEFAULT_DIAGONAL_SCALE)
    ratio = uncorrected[1]["youngs_modulus_N_per_m"] / uncorrected[0]["youngs_modulus_N_per_m"]
    assert math.isclose(ratio, 1.6, rel_tol=1e-9), ratio


#: ``(name, prose regex with numeric captures, quantity from the executed run, abs tolerance)``.
PROSE_CLAIMS = (
    (
        "free-surface deficit",
        r"低估 ~(\d+)%",
        lambda ns: (100.0 * (1.0 - ns["raw_density"] / ns["bulk_density"]),),
        0.5,
    ),
    (
        "anisotropy 45/0",
        r"硬 (\d+)%",
        lambda ns: (100.0 * (ns["young_45"] / ns["young_0"] - 1.0),),
        0.5,
    ),
    (
        "pair-count trap",
        r"C\((\d+),2\)=(\d+)",
        lambda ns: (float(ns["n_particles"]), float(ns["n_particles"] * (ns["n_particles"] - 1) // 2)),
        0.0,
    ),
    (
        "ledger residual scale",
        r"残差 ≈ (-?\d+(?:\.\d+)?e-?\d+)",
        lambda ns: (float(ns["coarse"]),),
        2e-4,
    ),
    (
        "factor-2 calibration trap",
        r"2 k_axis = 2\.85e9` → E = (\d+) GPa",
        lambda ns: (2.0 * float(ns["ice_young"]) / 1e9,),
        0.5,
    ),
)


def check_prose_claims(text: str, namespace: dict[str, object]) -> None:
    """Every numeric claim in the prose must agree with the executed run."""
    for name, pattern, quantity, tolerance in PROSE_CLAIMS:
        matches = re.findall(pattern, text)
        assert matches, f"prose claim is gone from the tutorial: {name} ({pattern})"
        expected = quantity(namespace)  # type: ignore[operator]
        for match in matches:
            stated = match if isinstance(match, tuple) else (match,)
            assert len(stated) == len(expected), f"prose claim {name}: capture arity drifted"
            for literal, value in zip(stated, expected):
                assert abs(float(literal) - value) <= tolerance, (
                    f"prose claim {name!r} states {literal}, the executed tutorial gives {value:.6g}"
                )


def test_prose_numbers_match_execution() -> None:
    namespace, _, _ = collected()
    check_prose_claims(TUTORIAL.read_text(encoding="utf-8"), namespace)


@pytest.mark.parametrize(
    ("old", "new", "why"),
    (
        ("低估 ~23%", "低估 ~35%", "free-surface deficit inflated"),
        ("C(35,2)=595", "C(35,2)=560", "pair-count trap understated"),
        ("残差 ≈ -2e-3", "残差 ≈ -9e-3", "ledger residual order understated"),
        ("→ E = 19 GPa", "→ E = 21 GPa", "factor-2 calibration trap mis-stated"),
    ),
    ids=["deficit", "pair-count", "ledger-scale", "factor-two-trap"],
)
def test_prose_claim_gate_is_not_vacuous(old: str, new: str, why: str) -> None:
    namespace, _, _ = collected()
    text = TUTORIAL.read_text(encoding="utf-8")
    check_prose_claims(text, namespace)  # baseline is green
    assert old in text, f"stale prose mutation ({why}): {old!r} not found"
    with pytest.raises(AssertionError, match="prose claim"):
        check_prose_claims(text.replace(old, new, 1), namespace)
