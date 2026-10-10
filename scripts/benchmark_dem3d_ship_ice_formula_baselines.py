"""Score published ice-resistance formula baselines against MT Uikku level-ice data.

This is a reference-data audit, not a DEM-versus-experiment validation. It quantifies
how published empirical formulas compare with the table-8 experimental reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "benchmarks" / "dem3d" / "ship_ice_validation" / "mt_uikku_level_ice.csv"
FORMULAS = {
    "lindqvist_prediction_kN": "Lindqvist",
    "riska_prediction_kN": "Riska",
    "jeong_prediction_kN": "Jeong",
    "keinonen_prediction_kN": "Keinonen",
}


def score_formula_baselines(reference_csv: Path, output_dir: Path) -> dict[str, Any]:
    with reference_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("reference CSV contains no cases")

    cases: list[dict[str, Any]] = []
    for row in rows:
        case_id = row.get("test_id", "").strip()
        try:
            experimental = float(row["experimental_reference_table8_kN"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"case {case_id or '<unknown>'} has no valid experimental reference") from exc
        if not math.isfinite(experimental) or experimental <= 0:
            raise ValueError(f"case {case_id} experimental reference must be finite and positive")
        scored: dict[str, Any] = {
            "test_id": case_id,
            "experimental_reference_kN": experimental,
        }
        for column, label in FORMULAS.items():
            try:
                predicted = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"case {case_id} has no valid {label} prediction") from exc
            if not math.isfinite(predicted) or predicted < 0:
                raise ValueError(f"case {case_id} {label} prediction must be finite and nonnegative")
            error = predicted - experimental
            scored[f"{label.lower()}_prediction_kN"] = predicted
            scored[f"{label.lower()}_signed_error_kN"] = error
            scored[f"{label.lower()}_absolute_percentage_error"] = abs(error) / experimental * 100.0
        cases.append(scored)

    summaries: list[dict[str, Any]] = []
    for column, label in FORMULAS.items():
        key = label.lower()
        errors = [case[f"{key}_signed_error_kN"] for case in cases]
        absolute_percentage_errors = [case[f"{key}_absolute_percentage_error"] for case in cases]
        experimental_values = [case["experimental_reference_kN"] for case in cases]
        predictions = [case[f"{key}_prediction_kN"] for case in cases]
        summaries.append({
            "formula": label,
            "case_count": len(cases),
            "mean_absolute_percentage_error_pct": mean(absolute_percentage_errors),
            "mean_signed_error_kN": mean(errors),
            "root_mean_square_error_kN": math.sqrt(mean(error * error for error in errors)),
            "mean_prediction_kN": mean(predictions),
            "mean_experimental_reference_kN": mean(experimental_values),
            "overprediction_cases": sum(error > 0 for error in errors),
            "underprediction_cases": sum(error < 0 for error in errors),
            "exact_match_cases": sum(error == 0 for error in errors),
        })

    summary = {
        "protocol": "tensordem-mt-uikku-formula-baselines-v1",
        "case_count": len(cases),
        "experimental_reference": "Hu & Zhou (2016), Table 8 experimental reference column",
        "metric_definitions": {
            "MAPE": "mean(abs(prediction - experiment) / experiment) * 100 percent",
            "mean_signed_error": "mean(prediction - experiment); positive means overprediction",
            "RMSE": "sqrt(mean((prediction - experiment)^2))",
        },
        "formula_summaries": summaries,
        "interpretation": (
            "These scores audit published empirical baselines against the cited experimental reference values. "
            "They do not validate TensorDEM and are not a fair direct comparison until case conditions, "
            "scale, geometry, friction and resistance definitions are matched."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mt_uikku_formula_baselines.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "mt_uikku_formula_baselines_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    with (output_dir / "mt_uikku_formula_baselines_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-mt-uikku-formulas"))
    args = parser.parse_args()
    print(json.dumps(score_formula_baselines(args.reference_csv, args.output), indent=2))


if __name__ == "__main__":
    main()
