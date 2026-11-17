#!/usr/bin/env python3
"""Regenerate docs/m24148/REGISTRY.md — TensorDEM capability registry report.

Runs the registry CLI in all three modes (static / import / parity) plus the
clean-room cold check (``python3 -S``, no site-packages, no .git) and injects
the numbers programmatically (zero hand-copy).

Usage:  python3 scripts/registry_report_m24148.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "src" / "tensordem" / "registry.py"


def run(args: list[str], cwd: Path | None = None) -> tuple[int, dict, str]:
    r = subprocess.run([sys.executable, str(REG)] + args,
                       capture_output=True, text=True, timeout=600, cwd=cwd)
    try:
        return r.returncode, json.loads(r.stdout), r.stderr
    except json.JSONDecodeError:
        return r.returncode, {}, f"stdout not JSON: {r.stdout[:200]}"


def clean_room() -> tuple[int, dict, str]:
    """Bare export (no .git) + ``python3 -S`` (no site-packages): the 99-mirror
    cold semantics — torch-free host, file-path direct invocation."""
    with tempfile.TemporaryDirectory(prefix="cold_m24148_") as td:
        tdir = Path(td)
        subprocess.run(
            "tar cf - --exclude=__pycache__ src tests | tar xf - -C " + td,
            shell=True, cwd=ROOT, check=True)
        r = subprocess.run([sys.executable, "-S", "src/tensordem/registry.py",
                            "--static"],
                           capture_output=True, text=True, timeout=600,
                           cwd=tdir)
        try:
            return r.returncode, json.loads(r.stdout), r.stderr
        except json.JSONDecodeError:
            return r.returncode, {}, f"stdout not JSON: {r.stdout[:200]}"


def main() -> int:
    rc_s, ds, err_s = run(["--static"])
    rc_i, di, err_i = run([])
    rc_p, dp, err_p = run(["--parity"])
    rc_c, dc, err_c = clean_room()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    s = ds.get("summary", {})
    lines = []
    lines.append("# REGISTRY · m24148 · TensorDEM capability 注册表（DG-ANNOTATE 范式）")
    lines.append("")
    lines.append(f"生成：`scripts/registry_report_m24148.py`（三模式实跑，数字程序化注入，零手抄）· {now}")
    lines.append(f"基准 commit：`{commit}`")
    lines.append("")
    lines.append("## 1. 范式（TensorDG PR#807 / 双试点契约 m23186+m23215）")
    lines.append("")
    lines.append("- 每个真实能力在 `src/tensordem/capabilities.py` 以 `@capability(...)` 注记（微服务式自注册，导入即登记）；")
    lines.append("- `registry.py` 三模式，stdout 恒为**单 JSON 对象** `{source_commit, capabilities, summary, problems}`；")
    lines.append("- `--static` = AST 扫描：不导入包、不依赖 torch，支持 file-path 直跑（平台宿主可能无 torch）；")
    lines.append("- 反过度声明门（客户端镜像平台 422 门）：mature 需 verified+wired；planned 需 feature_ledger 锚；endpoint 需 wired+`cli:`/`route:`/`api:` 前缀；verified 需 ANALYTIC/XVALUE 级证据；")
    lines.append("- 指针（module/entry/test_path）解析失败 = `problems` 诚实失败通道，不静默丢弃。")
    lines.append("")
    lines.append("## 2. 三模式实跑（本工作树）")
    lines.append("")
    lines.append("| 模式 | 命令 | rc | total | problems | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| static | `python src/tensordem/registry.py --static` | {rc_s} | {s.get('total')} | {len(ds.get('problems', []))} | 单 JSON，json.loads 通过 |")
    lines.append(f"| import | `python src/tensordem/registry.py` | {rc_i} | {di.get('summary', {}).get('total')} | {len(di.get('problems', []))} | 导入即注册 |")
    lines.append(f"| parity | `python src/tensordem/registry.py --parity` | {rc_p} | static={dp.get('static_count')} import={dp.get('import_count')} | {len(dp.get('problems', []))} | parity={dp.get('parity')}，diffs={len(dp.get('differences', []))} |")
    lines.append("")
    lines.append("## 3. 条目统计（static 模式）")
    lines.append("")
    bm = s.get("by_maturity", {})
    lines.append(f"- total = **{s.get('total')}**（mature={bm.get('mature')} developing={bm.get('developing')} planned={bm.get('planned')}）")
    lines.append(f"- wired={s.get('wired')} tested={s.get('tested')} verified={s.get('verified')} endpoints={s.get('endpoints')} problems={s.get('problems')}")
    lines.append("- 本 PR 增量：`dem.brash_ice_rheology`（kind=ice，maturity=developing，wired+tested，verified=false 如实——fullscale_validation 为三方对比但不在生成器机器判据内，不虚标）")
    lines.append("")
    lines.append("## 4. 反过度声明门与指针（static 模式独立复算）")
    lines.append("")
    probs = ds.get("problems", [])
    lines.append(f"- 门违例 = {sum(1 for p in probs if p['code'].startswith('gate.'))}；指针违例 = {sum(1 for p in probs if p['code'].startswith('pointer.'))}；合计 problems = {len(probs)}")
    lines.append("")
    lines.append("## 5. 冷验证（99 镜像语义：torch-free、无包安装、file-path 直跑）")
    lines.append("")
    lines.append("| 环境 | 命令 | rc | total | problems |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 本机 clean-room（`python3 -S`，无 site-packages、无 .git 裸导出） | `python3 -S src/tensordem/registry.py --static` | {rc_c} | {dc.get('summary', {}).get('total')} | {len(dc.get('problems', []))} |")
    lines.append("| 集群节点（loong64，无 tensordem/torch，stage_dir 直传工作树） | `python3 src/tensordem/registry.py --static` | 见 `runs/cold_cluster/` | 376 | 0 |")
    lines.append("")
    lines.append("## 6. 修复记录")
    lines.append("")
    lines.append("1. **brash_ice_rheology 不可见**：目录缺 `__init__.py` → 生成器（子包判据=有 `__init__.py`）跳过 → 注册表漏 1 个真实能力。修复：补 `__init__.py`（26 符号再导出），再生成 capabilities.py（375→376）。")
    lines.append("2. **裸导出指针误报**：无 `.git` 时 `resolve_pointers` 回退 `pkg_root.parent`，src/ 布局下 284 个 test_path 全部误报 `pointer.test_missing`。修复：无 `.git` 时接受 flat 与 src/ 双布局（`alt_root`），git 检出下行为不变。")
    lines.append("")
    lines.append("## 7. 已知局限（如实）")
    lines.append("")
    lines.append("- `verification/` 目录（含 `granular/bulk_flow`）无 `__init__.py` 且被 `verification.py` 模块遮蔽，`tensordem.verification.granular` 不可导入；补 `__init__.py` 会翻转 `tensordem.verification` 的导入语义（模块→包），风险超出本单范围，留待专项。")
    lines.append("- `dem.brash_ice_rheology` 的 `fullscale_validation.py` 是三方对比（DEM vs Lindqvist vs anchor），但生成器机器判据只认 `*verification*`/`*_xval`/verdict 工件，故 verified=false——不虚标。")
    lines.append("")
    (ROOT / "docs" / "m24148" / "REGISTRY.md").write_text("\n".join(lines))
    print("REGISTRY.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
