"""Determinism / no-retuning check for the m21007 frozen kink scan (m21203).

Two questions, both answered from stored provenance only:

1. **Protocol identity** — does the protocol hash of every archived case still
   equal the hash recorded by the *first* archived run (``KN_a1.6_g12.json``,
   committed by m21007)?  A drift would mean the frozen closing arm was
   silently retuned.
2. **Run-to-run determinism** — is a re-run of one already-archived case
   bit-identical in the judged quantities (runout ratio, measured aspect,
   emplacement time, particle count, step count)?

Usage::

    python3 scripts/check_reproducibility_m21203.py --reference <orig.json> --rerun <new.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007", "cases")
OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007",
                   "reproducibility.json")

JUDGED = ("runout_ratio_p995", "aspect_measured", "n_particles", "n_steps",
          "time_end", "stop_reason", "column_radius_measured",
          "column_height_measured")


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def protocol_hashes(case_dir: str = CASE_DIR) -> dict:
    hashes = {}
    for path in sorted(glob.glob(os.path.join(case_dir, "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        hashes[data["case_id"]] = data["protocol_sha256"]
    return hashes


def compare(reference: dict, rerun: dict) -> dict:
    deltas = {}
    for key in JUDGED:
        a, b = reference[key], rerun[key]
        if isinstance(a, float) and isinstance(b, float):
            deltas[key] = b - a
        else:
            deltas[key] = 0.0 if a == b else float("nan")
    t90_ref = reference["arrival"]["primary"]["fraction_0.9"]
    t90_new = rerun["arrival"]["primary"]["fraction_0.9"]
    deltas["t90"] = t90_new - t90_ref
    return {
        "same_protocol_sha256": reference["protocol_sha256"] == rerun["protocol_sha256"],
        "protocol_sha256": rerun["protocol_sha256"],
        "bitwise_identical": reference == rerun,
        "max_abs_delta": max((abs(v) for v in deltas.values()
                              if isinstance(v, float) and v == v), default=float("nan")),
        "deltas": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="original archived case JSON")
    ap.add_argument("--rerun", required=True, help="freshly recomputed case JSON")
    args = ap.parse_args(argv)
    ref, new = load(args.reference), load(args.rerun)
    payload = {
        "case_id": new["case_id"],
        "reference_file": os.path.relpath(args.reference, ROOT),
        "rerun_file": os.path.relpath(args.rerun, ROOT),
        "protocol_hashes": protocol_hashes(),
        **compare(ref, new),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: payload[k] for k in
                      ("case_id", "same_protocol_sha256", "bitwise_identical",
                       "max_abs_delta")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())