"""m1630 巡更工具（scripts/audit_upstream_diff.py）的判据测试。

分三层，全部不依赖网络：
1. 纯函数层：主题/风险/路径归类/CI glob 覆盖（合成提交字典，逐条判据）；
2. 合成仓库层：临时 git 仓库里造「加-删-改名」三种历史，验证净差集 vs churn 口径；
3. 真仓库层：若本地存在 origin/main，则把工具输出与独立 git 命令的计数对齐（缺 ref 就 skip，
   CI 的 PR checkout 没有 origin/main 也不会红）。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "audit_upstream_diff.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("audit_upstream_diff", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    if not TOOL.exists():
        pytest.skip("audit tool missing")
    return load_tool()


# ------------------------------------------------------------------ 纯函数层


def mk(subject: str, files=(), parents=("p1",), sha="deadbeef"):
    return {
        "sha": sha,
        "short": sha[:7],
        "date": "2026-11-01",
        "author": "t",
        "parents": list(parents),
        "subject": subject,
        "files": [{"path": p, "add": a, "del": d} for p, a, d in files],
    }


def test_theme_priority_deliverable_beats_docs(tool):
    c = mk("m943: matured mission deliverables（79 个文件被门禁滤除）",
           [("docs/a.md", 1, 0), ("src/tensordem/dem.py", 1, 0)])
    assert tool.theme_of(c)["primary"] == "T-DELIVERABLE"


def test_theme_merge_is_not_primary_content(tool):
    c = mk("Merge pull request #80 from cxs503/agent/m995", [], parents=("a", "b"))
    assert tool.theme_of(c)["primary"] == "T-MERGE"


def test_theme_timestep_beats_docs(tool):
    """time-step 战役提交即使只改 docs，也应按主题归到 T-TIMESTEP（优先级高于 T-DOCS）。"""
    c = mk("Document fixed-duration DEM3D time-step studies", [("docs/x.md", 5, 1)])
    assert tool.theme_of(c)["primary"] == "T-TIMESTEP"


def test_theme_solver_fix(tool):
    c = mk("fix(dem3d): 恢复外部节点载荷 / 压板阻尼符号——CI 4 红根修",
           [("src/tensordem/dem3d.py", 100, 1)])
    c["is_merge"] = False
    assert tool.theme_of(c)["primary"] == "T-SOLVER"


def test_theme_docs_only(tool):
    # 纯文档提交（主题词表之外）：主题应回落到 T-DOCS
    c = mk("Align TensorDEM docs with science factory", [("docs/x.md", 5, 1)])
    c["is_merge"] = False
    assert tool.theme_of(c)["primary"] == "T-DOCS"


def test_risk_phys_core_high_without_tests(tool):
    c = mk("tweak", [("src/tensordem/dem3d.py", 3, 1)])
    flags = {f["code"]: f for f in tool.risk_of(c)}
    assert flags["R-PHYS-CORE"]["severity"] == "high"
    assert flags["R-PHYS-CORE"]["evidence"]["tests_in_same_commit"] == 0


def test_risk_phys_core_downgraded_with_tests(tool):
    c = mk("tweak", [("src/tensordem/dem3d.py", 3, 1), ("tests/test_dem3d.py", 10, 0)])
    flags = {f["code"]: f for f in tool.risk_of(c)}
    assert flags["R-PHYS-CORE"]["severity"] == "medium"


def test_risk_gate_filtered_severity_threshold(tool):
    small = {f["code"]: f for f in tool.risk_of(mk("m1: matured mission deliverables（2 个文件被门禁滤除）"))}
    big = {f["code"]: f for f in tool.risk_of(mk("m2: matured mission deliverables（79 个文件被门禁滤除）"))}
    assert small["R-GATE-FILTERED"]["severity"] == "medium"
    assert big["R-GATE-FILTERED"]["severity"] == "high"
    assert big["R-GATE-FILTERED"]["evidence"]["filtered_files_declared"] == 79


def test_risk_deliverable_drop_only_on_merge(tool):
    files = [(f"docs/f{i}.md", 0, 10) for i in range(6)]
    merge = mk("Merge pull request #80", files, parents=("a", "b"))
    normal = mk("prune docs", files)
    assert "R-DELIVERABLE-DROP" in {f["code"] for f in tool.risk_of(merge)}
    assert "R-DELIVERABLE-DROP" not in {f["code"] for f in tool.risk_of(normal)}


def test_path_class(tool):
    assert tool.path_class("src/tensordem/dem.py") == "src"
    assert tool.path_class("tests/test_a.py") == "tests"
    assert tool.path_class("docs/a.md") == "docs"
    assert tool.path_class("pyproject.toml") == "other"


def test_ci_glob_coverage(tool):
    filters = ["benchmarks/dem3d/**", "tests/test_dem3d_ucs.py"]
    assert tool._covered("benchmarks/dem3d/case/a.json", filters)
    assert tool._covered("tests/test_dem3d_ucs.py", filters)
    assert not tool._covered("tests/test_channel_bank_collapse.py", filters)


# ------------------------------------------------------------------ 合成仓库层


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture()
def synth_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "keep.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "tag", "base0")
    # 1) 新增并被后续提交删除（应进 churn）
    (repo / "docs").mkdir()
    (repo / "docs" / "lost.md").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "m1: matured mission deliverables（1 个文件被门禁滤除）")
    # 2) 新增后改名（同 blob -> relocated，不算 lost）
    (repo / "docs" / "moved.md").write_text("same-content\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add moved")
    _git(repo, "mv", "docs/moved.md", "docs/moved2.md")
    _git(repo, "commit", "-qm", "rename moved")
    # 3) 删除 lost.md
    (repo / "docs" / "lost.md").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "drop lost")
    return repo


def test_synthetic_net_vs_churn(tool, synth_repo: Path, monkeypatch):
    monkeypatch.chdir(synth_repo)
    payload = json.loads(tool.to_json(tool.build("base0", "HEAD")))
    assert payload["meta"]["n_commits"] == 4  # m1 / add / rename / drop
    assert payload["meta"]["net_files"] == 1  # 净：只多了一个 docs/moved2.md
    ch = payload["churn"]
    assert ch["files_absent_at_head"] == 2  # lost.md + moved.md(旧名)
    gone = ch["absent_files"]
    assert gone["docs/lost.md"]["lost"] is True
    assert gone["docs/moved.md"]["moved_to"] == ["docs/moved2.md"]
    assert ch["files_lost_no_same_content_at_head"] == 1


# ------------------------------------------------------------------ 真仓库层


def test_real_repo_counts_agree_with_git(tool, monkeypatch):
    monkeypatch.chdir(ROOT)
    r = subprocess.run(["git", "rev-parse", "--verify", "origin/main"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("origin/main 不在本 checkout（PR 浅克隆），跳过真仓库对账")
    payload = tool.build("HEAD", "origin/main")
    m = payload["meta"]
    n_indep = int(_git(ROOT, "rev-list", "--count", "HEAD..origin/main").strip())
    assert m["n_commits"] == n_indep
    files_indep = len(_git(ROOT, "diff", "--name-only", "HEAD", "origin/main").splitlines())
    assert m["net_files"] == files_indep
    tree = set(_git(ROOT, "ls-tree", "-r", "--name-only", "origin/main").splitlines())
    for path, info in payload["churn"]["absent_files"].items():
        assert path not in tree
        assert info["deleted_by"], f"{path} 被删但没记录删除提交"

# ------------------------------------------------------------------ 渲染层


def test_renderer_smoke(tmp_path: Path):
    """渲染器必须能从工件产出 3 张 png + 一份 md，且报告里不残留未替换的占位符。"""
    import subprocess

    data = ROOT / "artifacts" / "m1630" / "upstream_diff.json"
    if not data.exists():
        pytest.skip("工件缺失（artifact 未随分支交付）")
    figdir = tmp_path / "figs"
    md = tmp_path / "report.md"
    r = subprocess.run(
        ["python", str(ROOT / "scripts" / "render_upstream_diff_report.py"),
         "--data", str(data), "--figdir", str(figdir), "--out", str(md)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr[-800:]
    pngs = sorted(figdir.glob("*.png"))
    assert len(pngs) == 3 and all(p.stat().st_size > 20_000 for p in pngs)
    body = md.read_text(encoding="utf-8")
    assert "上游差集巡更报告" in body
    assert "{ch[" not in body and "None" not in body.split("## 6")[0][:200]
    # 关键数字必须来自工件（抽样对账）
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert str(payload["meta"]["n_commits"]) in body
    assert f"{payload['churn']['churn_ratio']:.1%}" in body
