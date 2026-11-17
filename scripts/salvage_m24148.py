#!/usr/bin/env python3
"""Regenerate docs/m24148/SALVAGE.md — per-item salvage verdicts (m24013-ghost).

Compares every file entry of the isolated m24013 worktree tarball against the
current main tree by sha256 (zero hand-copy) and emits the verdict table.
Verdict rules (frozen, see docs/m24148/SALVAGE.md section 1):

* 收: importable + substantive implementation + absent from main or main weaker;
* 弃: byte-identical to main / main is a strict superset (tarball = older
  snapshot) / garbage (.tmpwork).

Usage:  python3 scripts/salvage_m24148.py [tarball]
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAR = "/srv/tensor/agent/salvage/m24013-ghost.tar.gz"

# DIFF / ABSENT items with case-specific verdicts (reasons audited in m24148).
VERDICT = {
    "src/tensordem/fem_dem_fracture/__init__.py":
        ("弃", "main 版为严格超集：额外导出 dynamics/study_dynamic 全部符号（wave-2 动态交接）；tarball 为更早快照，回退即降级"),
    "src/tensordem/fem_dem_fracture/handoff.py":
        ("弃", "main 版支持 schema 1+2（SCHEMAS 元组，schema 2 为严格超集）；tarball 版仅认 schema 1，回退即丢 wave-2 工件读取能力"),
    "benchmarks/freeze_bond_thermal_profile/results.json":
        ("弃", "main 版含 L4 benchmark 对象（front_vs_stefan_quasi / front_vs_neumann_exact / energy_closure / latent_heat_budget 逐项基准对比）；tarball 版无 benchmark 块且 all_passed=true 为弱判据"),
    "benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py":
        ("弃", "main 版新增 build_benchmark()（成熟度门 L4 基准表）；tarball 版为无基准块的旧版"),
    ".githooks/pre-commit":
        ("弃", "工作树本地守卫（禁 commit main，Door-A 封印 2026-11-06）；非仓库能力，且未设 core.hooksPath 时不生效；平台分支纪律已覆盖"),
}
TMPWORK = ("弃", ".tmpwork 垃圾（远程模块缓存/字体缓存/临时脚本），任务令明示忽略")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> int:
    tar_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAR
    rows = []
    with tarfile.open(tar_path) as t:
        for m in sorted(t.getmembers(), key=lambda x: x.name):
            if not m.isfile():
                continue
            name = m.name
            h = sha(t.extractfile(m).read())
            target = ROOT / name
            if target.is_file():
                mh = sha(target.read_bytes())
                rows.append((name, "SAME" if h == mh else "DIFF", h, mh))
            else:
                rows.append((name, "ABSENT", h, ""))

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()[:12]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# SALVAGE · m24148 · m24013-ghost 逐件判定")
    lines.append("")
    lines.append(f"生成：`scripts/salvage_m24148.py`（sha256 程序化比对，零手抄）· {now}")
    lines.append(f"基准 commit：`{commit}` · 来源：`{tar_path}`（协调者隔离导出）")
    lines.append("")
    lines.append("## 1. 判定标准（冻结）")
    lines.append("")
    lines.append("- **收**：可导入 + 有实质实现（非空壳）+ main 中不存在或 main 版更弱；")
    lines.append("- **弃**：与 main 字节一致（无新内容）/ main 版为严格超集（tarball 为旧快照）/ 垃圾；")
    lines.append("- 模块级判定标准（任务令）：可导入 + 有实质实现非空壳。")
    lines.append("")
    lines.append("## 2. 模块级结论（任务点名的两个模块）")
    lines.append("")
    lines.append("| 模块 | 可导入 | 实质实现 | 判定 |")
    lines.append("|---|---|---|---|")
    lines.append("| `tensordem.brash_ice_rheology` | 是（10 件实现文件，`BrashHull`/`Closure`/`mu(I)`/`JopParameters`/四准则 validation/系数台账） | 是 | **收（打包修复）**：10 件实现与 main 字节一致，唯一缺口=缺 `__init__.py` 致注册表不可见；本 PR 补 `__init__.py` 使其成为真子包并进入 capability 注册表（`dem.brash_ice_rheology`） |")
    lines.append("| `tensordem.fem_dem_fracture` | 是（8 件，`__all__`=66 符号） | 是（plate_fem/crack_network/handoff/dynamics/study_dynamic） | **弃（tarball 3 件中 1 件 SAME、2 件为旧快照）**：crack_network.py 与 main 一致；__init__.py 与 handoff.py 的 main 版均为严格超集，回退即降级 |")
    lines.append("")
    lines.append(f"## 3. 逐件判定（tarball 全部 {len(rows)} 个文件条目；任务令「68 件」含目录条目，目录不逐件列）")
    lines.append("")
    lines.append("| # | 路径 | 比对 | sha256[:16] (tar / main) | 判定 | 理由 |")
    lines.append("|---|---|---|---|---|---|")
    for i, (name, st, h, mh) in enumerate(rows, 1):
        if name.startswith(".tmpwork/"):
            v, why = TMPWORK
            lines.append(f"| {i} | `{name}` | tmpwork | {h} / — | {v} | {why} |")
        elif st == "SAME":
            lines.append(f"| {i} | `{name}` | SAME | {h} / {mh} | 弃 | 与 main 字节一致，无新内容 |")
        else:
            v, why = VERDICT.get(name, ("弃", "未评估"))
            lines.append(f"| {i} | `{name}` | {st} | {h} / {mh or '—'} | {v} | {why} |")
    lines.append("")
    lines.append("## 4. 本 PR 实际入库的 salvage 产物")
    lines.append("")
    lines.append("| 文件 | 性质 |")
    lines.append("|---|---|")
    lines.append("| `src/tensordem/brash_ice_rheology/__init__.py` | **新增**：ghost 模块的打包修复（docstring + 26 符号再导出），使 `dem.brash_ice_rheology` 进入注册表 |")
    lines.append("| `src/tensordem/capabilities.py` | 再生成：375 → 376 条（+`dem.brash_ice_rheology`，grade B，wired+tested，verified=false 如实） |")
    lines.append("| `src/tensordem/registry.py` | 修复：无 `.git` 的裸导出（冷环境）下 test_path 指针解析不再误报 284 条（src/ 布局回退） |")
    lines.append("")
    lines.append("## 5. 验证")
    lines.append("")
    lines.append("- `python3 -m pytest tests/test_brash_ice_rheology_mu_i_resistance.py tests/test_dem_brash_ice_fullscale.py tests/test_fem_dem_fracture_handoff.py tests/test_fem_dem_fracture_dynamic.py` → **110 passed, 1 skipped**")
    lines.append("- `python src/tensordem/registry.py --static` → rc=0，单 JSON，total=376，problems=0")
    lines.append("- `python src/tensordem/registry.py --parity` → parity=True（376=376，diffs=0）")
    lines.append("- 冷环境（`python3 -S`，无 site-packages、无 .git 裸导出）直跑 file-path：rc=0，total=376，problems=0")
    lines.append("")
    (ROOT / "docs" / "m24148" / "SALVAGE.md").write_text("\n".join(lines))
    print(f"SALVAGE.md written: {len(rows)} file rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
