"""README ↔ 代码 API 巡检器自身的门：绿要有绿的理由，红要能红。

三层控制：

1. **正例**：当前工作树跑 ``scripts/readme_api_audit.py``（静态面）必须零 MISMATCH。
2. **负例（测试运行器）**：给巡检器一段把 ``python -m unittest discover`` 当测试命令
   的 README 文本，它必须判 MISMATCH —— 防止「巡检器恒绿」。
3. **负例（生成物门惯例）**：给巡检器一份硬断言 ``artifacts/... exists()`` 且无 skip
   保护的测试文件，必须判 MISMATCH；同一文件加上 ``skipif`` 后必须判 OK。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "readme_api_audit.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("_readme_api_audit", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module        # dataclasses 解析 __module__ 需要它已注册
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_current_tree_has_no_static_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, str(AUDIT_SCRIPT), "--out", str(out), "--quiet"],
                          cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, (proc.stdout[-800:] + proc.stderr[-800:])
    assert out.exists()


def test_runner_check_fires_on_unittest_command() -> None:
    """负例：README 把 unittest discover 当作跑全套测试的命令 -> 必须判红。"""
    auditor = _load_auditor()
    text = textwrap.dedent("""\
        ```bash
        python -m unittest discover -s tests -v
        ```
        """)
    audit = auditor.Audit()
    auditor.audit_test_runner(audit, text,
                              style={"pytest_functions": 2717, "unittest_cases": 319})
    failures = audit.failures()
    assert len(failures) == 1, [f.claim for f in failures]
    assert failures[0].kind == "test-runner"
    assert failures[0].status == "MISMATCH"


def test_prose_explaining_the_no_op_is_accepted_with_waiver() -> None:
    """正例：带显式豁免声明的说明性文字不算不一致（豁免理由进报告）。"""
    auditor = _load_auditor()
    text = ("<!-- audit-allow: test-runner: 说明性文字 -->\n"
            "`python -m unittest discover -s tests` 只认 TestCase 子类。\n")
    audit = auditor.Audit()
    audit.waivers = {k: v.strip() for k, v in auditor.WAIVER.findall(text)}
    auditor.audit_test_runner(audit, text, style={"pytest_functions": 2717, "unittest_cases": 319})
    assert not audit.failures(), [f.detail for f in audit.failures()]
    assert any("已声明豁免" in c.detail for c in audit.checks)


def test_artifact_gate_convention_fires_without_skip_guard(tmp_path: Path) -> None:
    """负例：硬断言 artifacts/ 生成物存在且无 skip 保护 -> 必须判红。"""
    auditor = _load_auditor()
    bad = tmp_path / "test_hard_artifact_gate.py"
    bad.write_text(textwrap.dedent('''\
        from pathlib import Path

        SUMMARY = Path("artifacts/demo/summary.json")


        def test_committed_summary() -> None:
            assert SUMMARY.exists(), "missing generated artifact"
        '''), encoding="utf-8")
    audit = auditor.Audit()
    auditor.audit_artifact_gate_convention(audit, files=[bad])
    assert [f.status for f in audit.failures()] == ["MISMATCH"]

    good = tmp_path / "test_guarded_artifact_gate.py"
    good.write_text(textwrap.dedent('''\
        from pathlib import Path

        import pytest

        SUMMARY = Path("artifacts/demo/summary.json")
        needs = pytest.mark.skipif(not SUMMARY.exists(), reason="artifact not generated")


        @needs
        def test_committed_summary() -> None:
            assert SUMMARY.exists(), "missing generated artifact"
        '''), encoding="utf-8")
    audit = auditor.Audit()
    auditor.audit_artifact_gate_convention(audit, files=[good])
    assert not audit.failures(), [f.detail for f in audit.failures()]