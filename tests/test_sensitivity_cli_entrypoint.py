"""README 记录的 ``python -m tensordem.sensitivity`` 命令必须真的可跑。

背景（2026-11 巡检）：``tensordem.sensitivity`` 被扩成 UQ 工具集后包内没有
``__main__.py``，README 的「网格/分辨率敏感性扫描」一节所写命令会以
``No module named tensordem.sensitivity.__main__`` 失败，而文档同时承诺了三个产物
（``history_nx_*.csv`` / ``summary.csv`` / ``comparison_to_finest.csv``）。
本测试把「文档命令 ↔ 代码入口 ↔ 产物名」三者钉在一起：

1. ``--help`` 退出码为 0，且 README 用到的旗标（``--nx-levels --duration --speed
   --output``）确实存在；
2. 一次最小扫描真的产出 README 声称的三个文件名。

不做物理断言（这里是入口/契约测试，不是收敛研究），因此参数取到能跑通的最小规模。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTED_FLAGS = ("--nx-levels", "--duration", "--speed", "--output")


def _run(args: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "tensordem.sensitivity"] + args,
                          cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def test_module_entrypoint_help_lists_documented_flags() -> None:
    proc = _run(["--help"], ROOT)
    assert proc.returncode == 0, f"`python -m tensordem.sensitivity --help` 失败: {proc.stderr[-400:]}"
    for flag in DOCUMENTED_FLAGS:
        assert flag in proc.stdout, f"README 使用的 {flag} 不在入口帮助里"


def test_documented_command_writes_documented_artifacts(tmp_path: Path) -> None:
    proc = _run(["--nx-levels", "5,7", "--duration", "0.02", "--speed", "0.2",
                 "--output", str(tmp_path)], ROOT)
    assert proc.returncode == 0, proc.stderr[-400:]
    produced = {p.name for p in tmp_path.iterdir()}
    for name in ("summary.csv", "comparison_to_finest.csv", "history_nx_5.csv", "history_nx_7.csv"):
        assert name in produced, f"README 承诺的 {name} 没有产出；实有 {sorted(produced)}"


@pytest.mark.parametrize("bad", [("--nx-levels", "7,7"), ("--nx-levels", "x")])
def test_invalid_levels_are_rejected(bad: tuple[str, str], tmp_path: Path) -> None:
    """入口要 fail-loud：重复/非数字列数不能被静默接受（否则产物名会互相覆盖）。"""
    proc = _run([bad[0], bad[1], "--duration", "0.01", "--output", str(tmp_path)], ROOT)
    assert proc.returncode != 0