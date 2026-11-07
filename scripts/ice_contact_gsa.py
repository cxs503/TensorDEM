"""Run (or re-run) only the global sensitivity analysis of IN27-S2 and merge it.

The Sobol study is the expensive tail of :mod:`tensordem.ice_contact.study`
(every Saltelli evaluation is a pair of DEM runs), so it is kept in its own
entry point: it can be re-run with a different base-sample size without
redoing the campaign matrix.

    PYTHONPATH=src python scripts/ice_contact_gsa.py --n-base 20 --workers 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensordem.ice_contact.figures import render_sobol
from tensordem.ice_contact.study import _json_default, global_sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ice_contact_in27_s2"))
    parser.add_argument("--n-base", type=int, default=20)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)

    study_path = args.out / "study.json"
    payload = json.loads(study_path.read_text())
    payload["gsa"] = global_sensitivity(n_base=args.n_base, workers=args.workers)
    figure = render_sobol(payload, args.out)
    figures = list(payload.get("figures", []))
    if figure is not None and figure.name not in figures:
        figures.append(figure.name)
    payload["figures"] = figures
    study_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    gsa = payload["gsa"]
    print("S1:", np.round(gsa["S1"], 3).tolist())
    print("ST:", np.round(gsa["ST"], 3).tolist())
    print("split-half S1 spread:", np.round(gsa["split_half_S1_spread"], 3).tolist())
    print("unique DEM configurations:", gsa["n_unique_runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())