"""Run the manifest-specified Brazilian equilibration ramp (single deterministic pass).

`docs/dem3d_platen_contact_mask_audit.md` §6 left the diametral centre field as
WITHHELD because the campaign that the manifest already specifies
(`benchmarks/dem3d/brazilian_cases.json`, ``steps: 40000``) had never been run:
only 900 steps were.  This driver executes exactly that campaign once and writes
the time history so the physics checks can be evaluated at the ramp end.

It deliberately runs a *single* pass (no determinism repeat) so the campaign
fits in one scheduling window; determinism of this benchmark is already covered
by `scripts/benchmark_dem3d_brazilian.py` at short horizons.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import MANIFEST, _run_once  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="brazil_elastic_field")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dem3d_brazilian_ramp"))
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["id"] == args.case)
    steps = int(args.steps if args.steps is not None else case.get("steps", 40000))
    rows, signature, metrics = _run_once(case, steps)

    reference = manifest["reference"]
    target_ratio = float(reference["centre_principal_stress_ratio"])
    ratio = metrics.get("final_centre_principal_ratio")
    result = {
        "case_id": case["id"],
        "protocol": manifest["protocol"],
        "steps": steps,
        "signature_sha256": signature,
        "reference": reference,
        "acceptance": manifest["acceptance"],
        "metrics": metrics,
        "history": rows,
        "hondros": {
            "centre_principal_ratio": ratio,
            "centre_principal_ratio_target": target_ratio,
            "centre_principal_ratio_rel_error": (
                abs(ratio - target_ratio) / abs(target_ratio)
                if ratio is not None and math.isfinite(ratio) else None),
            "centre_tension_rel_error": metrics.get("final_centre_tension_rel_error"),
            "split_stress_Pa": rows[-1]["split_stress_Pa"] if rows else None,
            "centre_sigma_xx_Pa": metrics.get("final_centre_sigma_xx_Pa"),
            "centre_sigma_zz_Pa": metrics.get("final_centre_sigma_zz_Pa"),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{case['id']}_ramp.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / f"{case['id']}_ramp_history.csv").open("w", encoding="utf-8") as stream:
        stream.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            stream.write(",".join(str(row[key]) for key in row) + "\n")
    print(json.dumps({
        "case_id": case["id"],
        "steps": steps,
        "signature_sha256": signature,
        "final_centre_sigma_xx_Pa": metrics.get("final_centre_sigma_xx_Pa"),
        "final_centre_sigma_zz_Pa": metrics.get("final_centre_sigma_zz_Pa"),
        "final_centre_principal_ratio": ratio,
        "split_stress_Pa": rows[-1]["split_stress_Pa"] if rows else None,
        "centre_tension_rel_error": metrics.get("final_centre_tension_rel_error"),
        "final_contact_arc_ratio": metrics.get("final_contact_arc_ratio"),
        "peak_load_N": metrics.get("peak_load_N"),
    }, indent=2))


if __name__ == "__main__":
    main()