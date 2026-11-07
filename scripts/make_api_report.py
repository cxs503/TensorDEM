#!/usr/bin/env python3
"""Build the platform progress report payload (``/tmp/report.json``).

The markdown body is assembled from the generated campaign artifacts, so every
number is injected programmatically; the figures are embedded as downscaled
base64 PNGs (each kept well below the platform's ~250 KB payload limit).
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

from PIL import Image

OUT = Path("artifacts/pressure_area_index")
DOC = Path("docs/pressure_area_index_dem.md")
TARGET = Path("/tmp/report.json")
MISSION_ID = 5447


def encode_png(path: Path, width: int = 900) -> dict:
    img = Image.open(path).convert("RGB")
    if img.width > width:
        img = img.resize((width, int(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > 70_000:                       # keep the payload small
        buf = io.BytesIO()
        img.convert("P", palette=Image.ADAPTIVE, colors=64).save(buf, format="PNG",
                                                                 optimize=True)
        data = buf.getvalue()
    return {"png_base64": base64.b64encode(data).decode(), "bytes": len(data)}


def main() -> int:
    summary = json.loads((OUT / "summary.json").read_text())
    fits = summary["fits"]
    diag = summary["mechanism_diagnostics"]
    prim, ctrl = fits["primary"], fits["control_uniform"]
    doc = DOC.read_text()

    in_band = 0.45 <= prim["exponent"] <= 0.55
    ctrl_zero = abs(ctrl["exponent"]) < 0.15

    figures = []
    for name, caption in (
        ("fig_pressure_area.png",
         "Left: measured elastic-limit global pressure p_G versus nominal contact "
         "area A with 30 realisations per area (error bars = 1 s.d.); dashed lines "
         "are the fitted power laws, the dotted line is the field reference A^-0.5. "
         "Right: fitted exponent with bootstrap 95 % CI for the correlated-field "
         "scans and for the simultaneous-failure (uniform) control."),
        ("fig_contact_growth.png",
         "Mechanism (i): the true contact area computed from the Hertzian contact "
         "widths grows as sqrt(penetration) (left, dotted reference) and the punch "
         "force follows delta^1.5 (right)."),
        ("fig_statistics.png",
         "Mechanism (ii): the failure non-simultaneity index per area for the "
         "correlated field versus the uniform (simultaneous) control, and the "
         "distribution of the number of bonds removed in a single failure event."),
    ):
        path = OUT / name
        if not path.exists():
            continue
        fig = encode_png(path)
        figures.append({"name": name, "caption": caption,
                        "png_base64": fig["png_base64"]})

    title = ("Pressure-area index emergence from DEM indentation: non-simultaneous "
             "failure produces p_G ~ A^-ex, simultaneous-failure negative control "
             "collapses to ex ~ 0")
    verdict = (
        f"ex = {prim['exponent']:.3f} [{prim['ci_low']:.3f}, {prim['ci_high']:.3f}] "
        f"for the correlated strength field versus ex = {ctrl['exponent']:.3f} "
        f"[{ctrl['ci_low']:.3f}, {ctrl['ci_high']:.3f}] for the perfectly correlated "
        f"(simultaneous-failure) control. Target band [0.45, 0.55]: "
        f"{'PASS' if in_band else 'HONEST FAIL'}; control -> 0: "
        f"{'PASS' if ctrl_zero else 'HONEST FAIL'}.")

    body = doc + "\n\n## 7. Acceptance-criteria JSON\n\n```json\n" + \
        json.dumps({"mission_id": MISSION_ID,
                    "exponent_primary": prim["exponent"],
                    "exponent_primary_ci": [prim["ci_low"], prim["ci_high"]],
                    "exponent_control": ctrl["exponent"],
                    "exponent_control_ci": [ctrl["ci_low"], ctrl["ci_high"]],
                    "target_band": [0.45, 0.55],
                    "target_band_met": in_band,
                    "control_zero_met": ctrl_zero,
                    "seeds_per_area": summary["seeds_per_area"],
                    "areas_m2": summary["areas_m2"],
                    "iterations": summary["fits"],
                    "diagnostics": diag,
                    "artifacts": [
                        "docs/pressure_area_index_dem.md",
                        "artifacts/pressure_area_index/summary.json",
                        "artifacts/pressure_area_index/scan_primary.json",
                        "artifacts/pressure_area_index/scan_control_uniform.json",
                        "src/tensordem/pressure_area_dem.py",
                        "tests/test_pressure_area_dem.py",
                    ]}, indent=1) + "\n```\n"

    payload = {
        "title": title,
        "summary": verdict,
        "scenario_key": "dem.pressure_area_index",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "ice", "pressure-area", "DEM", "non-simultaneous-failure",
                 "negative-control", "IN37-N03"],
        "markdown": body,
        "figures": figures,
    }
    TARGET.write_text(json.dumps(payload))
    print(f"wrote {TARGET} ({TARGET.stat().st_size/1024:.0f} KiB, "
          f"{len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())