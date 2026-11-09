"""Tests for the ISO 19906 external-anchor ledger of the ridge-keel/seabed-bearing topic.

task_key = paper-exp-ice-ridge-pipeline-seabed-bearing-1

The ledger's value is that its *mechanical* claims are machine-checkable and that
its *unverified* claims are structurally barred from being cited:

* the conversion chain must recompute to < 1e-9 relative residual on two
  independent numerical paths,
* the dimensional bookkeeping and the internal consistencies must be clean,
* no unverified (memory/convention) number may be flagged usable for a
  compliance claim,
* an anchor without verbatim text must not be silently filled in.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "anchors" / "ice_ridge_pipeline_seabed_bearing.py"
LEDGER = REPO / "benchmarks" / "anchors" / "ice-ridge-pipeline-seabed-bearing.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("ridge_seabed_anchor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def case(mod):
    return mod.extract_case_caliber()


def test_case_caliber_is_injected_from_source_not_hand_copied(mod, case):
    """口径必须来自源文件 AST 注入，并带 sha256 指纹。"""
    assert case["injection"].startswith("AST_LITERAL_MODULE_CONSTANT")
    assert len(case["source_sha256"]) == 64
    for name in mod.CASE_NAMES:
        assert name in case["values"] and case["values"][name]
    nums = mod.case_numbers(case)
    # 局部接触面积必须是管径×单位轴向长度，全局为龙骨宽×深
    assert nums["contact_area_local_m2"] == pytest.approx(nums["pipe_outer_diameter_m"] * mod.L_UNIT_M)
    assert nums["contact_area_global_m2"] == pytest.approx(nums["keel_width_m"] * nums["keel_depth_m"])


def test_conversion_chain_recomputes_below_1e_9(mod, case):
    res = mod.conversion_recompute_residual(case)
    assert res["max_relative_residual"] < 1e-9, res["worst_item"]


def test_two_paths_are_independent_and_agree(mod):
    """直接幂律 vs 对数反演，必须同值（同量级相对差 < 1e-12）。"""
    for C, n in ((3.0, -0.45), (1.2, -0.307)):
        for A in (0.25, 0.914, 20.0, 288.0):
            a = mod.p_power_law(C, A, n)
            b = mod.p_via_log_fit(C, A, n)
            assert abs(a - b) / abs(a) < 1e-12


def test_dimensionless_and_dimensionally_homogeneous(mod):
    dims = mod.dimension_checks()
    assert dims["exponent_mismatches"] == 0, [c for c in dims["checks"] if c["mismatch"]]
    # 力的对数斜率恒等 dlnF/dlnA = 1 + n
    res = mod.conversion_recompute_residual(mod.extract_case_caliber())
    assert any(item["item"].startswith("force_log_slope") for item in res["items"])


def test_internal_consistency_clean(mod, case):
    cons = mod.internal_consistency(case)
    assert cons["violations"] == 0, cons["issues"]
    assert cons["metrics"]["monotone_decreasing"] is True


def test_anchors_without_verbatim_text_are_withheld(mod):
    anchors = mod.build_anchors()
    assert len(anchors) == 2
    for a in anchors:
        assert a["verbatim_quote"] is None and a["table_or_page"] is None and a["numeric_values"] == {}
        assert a["provenance"]["status"] == "WITHHELD_NO_LICENSED_SOURCE"
        assert a["usable_for_compliance_claim"] is False


def test_candidates_never_citable_for_claims(mod):
    cands = mod.build_candidate_registry()
    assert cands
    assert all(c["provenance_level"] == "MEMORY_OR_CONVENTION" for c in cands)
    assert all(c["usable_for_compliance_claim"] is False for c in cands)


def test_domain_exponent_injected_from_doc_table(mod):
    dom = mod.extract_domain_exponent()
    assert dom["injection"].startswith("REGEX_FROM_DOC_TABLE")
    assert dom["line_number"] > 0 and len(dom["source_sha256"]) == 64
    assert 0.0 < dom["ex"] < 1.0
    assert dom["ci95"][0] < dom["ex"] < dom["ci95"][1]
    assert "primary" in dom["raw_line"]


def test_ledger_schema_and_honest_verdict():
    """工件必须存在、判据逐条带实测值与来源脚本，且不谎报 PASS。"""
    assert LEDGER.exists(), "先运行 benchmarks/anchors/ice_ridge_pipeline_seabed_bearing.py"
    doc = json.loads(LEDGER.read_text(encoding="utf-8"))
    for key in (
        "schema", "task_key", "preregistered_criteria", "case_caliber", "conversion_chain",
        "dimension_check", "internal_consistency", "anchors", "candidate_registry",
        "domain_comparison", "checks", "gate_summary", "verdict", "generator_script", "figures",
    ):
        assert key in doc, key
    assert doc["task_key"] == "paper-exp-ice-ridge-pipeline-seabed-bearing-1"
    assert [c["id"] for c in doc["checks"]] == ["C1", "C2", "C3", "C4"]
    for c in doc["checks"]:
        assert "measured" in c and "evidence" in c and c["status"] in {"PASS", "FAIL"}
    # 锚未逐字核销 -> 不得宣称 PASS
    if not any(a["verbatim_quote"] for a in doc["anchors"]):
        assert doc["verdict"] != "PASS"
        assert doc["gate_summary"]["C1_verbatim_anchor_verified"] is False
        assert doc["honest_fail_note"].strip()
    # 换算与量纲/一致性这三项必须是机检通过
    assert doc["gate_summary"]["C2_conversion_recomputable"] is True
    assert doc["gate_summary"]["C3_dimension_and_consistency"] is True
    assert doc["gate_summary"]["C4_unverified_not_citable"] is True
    # 未核候选的换算结果一律标 False
    assert all(r["usable_for_compliance_claim"] is False for r in doc["conversion_to_case_caliber"])