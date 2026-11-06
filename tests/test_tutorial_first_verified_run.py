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
import re
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