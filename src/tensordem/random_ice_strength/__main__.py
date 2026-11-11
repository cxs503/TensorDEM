"""Entry point: ``python -m tensordem.random_ice_strength``.

Runs the acceptance study of campaign node P295W1-N07 (Weibull marginal ->
correlated strength field -> breaking-force ensemble -> width sweep), writes
the machine-readable artifacts and exits non-zero if any criterion fails, so
the module is runnable from a shell / CI without importing it from Python.

Examples
--------
    PYTHONPATH=src python -m tensordem.random_ice_strength --out-dir runs/P295W1_N07
    PYTHONPATH=src python -m tensordem.random_ice_strength --no-write --quiet
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensordem.random_ice_strength",
        description="随机冰板强度场验收研究（Weibull 边际 + 相关高斯场 + 破冰力 ensemble）",
    )
    parser.add_argument("--out-dir", default="runs/P295W1_N07",
                        help="study.json 目录（默认 runs/P295W1_N07）")
    parser.add_argument("--artifact-dir", default="artifacts/p295w1_n07",
                        help="study.json 副本目录（默认 artifacts/p295w1_n07）")
    parser.add_argument("--no-write", action="store_true",
                        help="只计算并打印判据，不写 JSON 工件")
    parser.add_argument("--quiet", action="store_true", help="不打印逐条判据")
    args = parser.parse_args(argv)

    from .study import run_study

    result = run_study(out_dir=args.out_dir, artifact_dir=args.artifact_dir,
                       write=not args.no_write)
    checks = result["checks"]
    if not args.quiet:
        for key in sorted(checks):
            chk = checks[key]
            print(f"{key:42s} pass={bool(chk['pass'])!s:5s} value={chk.get('value')}")
    n_pass = sum(1 for c in checks.values() if c["pass"])
    print(f"P295W1-N07: {n_pass}/{len(checks)} criteria pass; all_pass={result['all_pass']}")
    if not args.no_write:
        print("artifacts:", result["artifacts"].get("written", []))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
