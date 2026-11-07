"""Driver for the IN34-W3-N01 full-spectrum broken-ice validation study.

Usage
-----
    python scripts/broken_ice_micro_study.py --stage fsd
    python scripts/broken_ice_micro_study.py --stage janssen
    python scripts/broken_ice_micro_study.py --stage arch --shard 3/12
    python scripts/broken_ice_micro_study.py --stage chain --n-floes 700
    python scripts/broken_ice_micro_study.py --stage assemble
    python scripts/broken_ice_micro_study.py --stage figures

Stages are independent so that the heavy ones (``arch``, ``chain``) can be
sharded across compute nodes: every shard writes
``artifacts/broken_ice_micro_study/parts/<stage>_<shard>.json`` and
``--stage assemble`` merges them into the criteria ledger ``study.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensordem.broken_ice_micro import study  # noqa: E402

DEFAULT_OUT = _ROOT / "artifacts" / "broken_ice_micro_study"


def _parts_dir(out_dir: Path) -> Path:
    d = out_dir / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=study._json_default))
    print(f"wrote {path}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["fsd", "arch", "janssen", "chain", "assemble", "figures"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--shard", default="0/1", help="shard index/count, e.g. 2/12")
    ap.add_argument("--n-floes", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of realisations (fsd/arch/chain)")
    ap.add_argument("--alphas", default="2.3")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    parts = _parts_dir(out_dir)

    if args.stage == "fsd":
        payload = study.fsd_stage()
        _write(parts / "fsd.json", payload)
        return 0

    if args.stage == "janssen":
        payload = study.janssen_stage()
        _write(parts / "janssen.json", payload)
        return 0

    if args.stage == "arch":
        index, count = (int(v) for v in args.shard.split("/"))
        pairs = study.arch_case_pairs(seeds=args.seeds or study.ARCH_SEEDS)[index::count]
        payload = study.arch_stage(pairs=pairs)
        _write(parts / f"arch_{index}.json", payload)
        return 0

    if args.stage == "chain":
        alphas = tuple(float(v) for v in args.alphas.split(","))
        seeds = tuple(range(3, 3 + (args.seeds or 2)))
        payload = study.chain_stage(alphas=alphas, seeds=seeds, n_floes=args.n_floes)
        _write(parts / "chain.json", payload)
        return 0

    if args.stage == "assemble":
        loaded: dict = {}
        fsd = parts / "fsd.json"
        if fsd.exists():
            loaded["fsd"] = json.loads(fsd.read_text())
        janssen = parts / "janssen.json"
        if janssen.exists():
            loaded["janssen"] = json.loads(janssen.read_text())
        chain = parts / "chain.json"
        if chain.exists():
            loaded["chain"] = json.loads(chain.read_text())
        arch_files = sorted(parts.glob("arch_*.json"))
        if arch_files:
            loaded["arch"] = [json.loads(p.read_text()) for p in arch_files]
        ledger = study.assemble(loaded, out_dir=out_dir)
        print(json.dumps({k: v["passed"] for k, v in ledger["verdicts"].items()}, indent=2))
        return 0

    if args.stage == "figures":
        from scripts.broken_ice_micro_figures import make_figures  # local import
        paths = make_figures(out_dir)
        print(json.dumps([str(p) for p in paths], indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())