"""Build the platform report payload (/tmp/report.json) for the formation campaign.

Embeds the campaign figures as base64 PNGs and uses the generated markdown report
as the body, so the published text and the artifact cannot drift apart.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]


def build(artifact_dir: Path, markdown_path: Path, mission_id: int = 1044) -> dict:
    data = json.loads((artifact_dir / "formation_campaign.json").read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    figures = []
    for entry in data["figures"]:
        png = artifact_dir / "figures" / entry["name"]
        if not png.exists():
            continue
        figures.append({
            "name": entry["name"],
            "caption": entry["caption"],
            "png_base64": base64.b64encode(png.read_bytes()).decode("ascii"),
        })
    summary = data["end_to_end"]["summary"]
    dem = data.get("dem") or {}
    payload = {
        "title": ("Multi-ship formation ice-channel breaking: DEM multi-body ice-ship "
                  "interaction, hydrodynamic interference coefficients and channel "
                  "refreezing thermodynamics"),
        "summary": (
            f"TensorDEM formation module: {summary['ships']}-ship column at {summary['speed_m_s']:.1f} m/s "
            f"in {summary['parent_thickness_m']:.1f} m level ice. Acceptance: N=1 degeneracy "
            f"{data['single_ship_degeneracy']['relative_error_percent']:+.4f}% (<3%), "
            f"refreezing vs Stefan {data['thermal']['max_abs_error_vs_stefan_percent']:.2f}% (<5%), "
            f"C_ij(d=50L)={data['interference']['min_abs_coefficient']:.2e} (<1e-3), "
            f"DEM energy residual "
            f"{(dem['energy']['relative_residual_percent'] if dem else float('nan')):+.2f}% (<5%). "
            f"Follower saves {summary['follower_saving_vs_level_ice'] * 100:.1f}% of level-ice "
            f"resistance in a fresh channel; hydrodynamic interference is only "
            f"{summary['hydrodynamic_interference_share_of_follower'] * 100:.2f}% of a follower's total "
            f"resistance, so the channel dominates. Absolute resistance law MAPE vs the MT Uikku "
            f"dataset is {data['calibration']['all_case_mape_percent']:.1f}% (thin-ice buckling "
            f"cases excluded it is {data['calibration']['thick_ice_mape_percent']:.1f}%)."),
        "scenario_key": "formation_ice_channel",
        "mission_id": mission_id,
        "tags": ["ice-going", "formation", "DEM", "Stefan", "hydrodynamic-interference",
                 "resistance", "TensorDEM", "campaign"],
        "markdown": markdown,
        "figures": figures,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(ROOT / "artifacts" / "formation"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "formation_ice_channel.md"))
    parser.add_argument("--out", default="/tmp/report.json")
    parser.add_argument("--mission-id", type=int, default=1044)
    args = parser.parse_args(argv)
    payload = build(Path(args.artifact_dir), Path(args.markdown), args.mission_id)
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.out}: {len(payload['markdown'])} chars, {len(payload['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())