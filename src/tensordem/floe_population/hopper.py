"""HOPPER model-test benchmark: what is required, and how far the evidence goes.

The campaign node asks for a comparison of the model's mean floe size ``<D>``
against the HOPPER ice-tank floe-size census of Lu et al. (2016) with a 30 %
tolerance.  That comparison needs the *measured* census: a table of floe
diameters and their counts from the model-test basin.

**The census is not in this repository, and it is not in the public literature in
tabular form.**  Inventing it, or substituting a curve read off a plot without a
traceable digitisation, would be data fabrication, so this module refuses to
produce a number in that case and reports the criterion as ``WITHHELD`` instead.
What it does provide is the complete, tested mechanics of the comparison, so that
dropping the measured census at :data:`HOPPER_DATA_PATH` turns the criterion into
a real quantitative check with no code change.

Two substitutes are reported next to the withheld criterion, both machine
checkable and both labelled as what they are:

``fragment_exponent``
    the power-law exponent of the pooled fragment census of the DEM impact
    campaign, compared with the broken-ice window ``2.5 <= alpha <= 3.5`` that
    this repository already accepts for FSD work
    (:data:`tensordem.broken_ice_micro.FSD_FIT_ALPHA_RANGE`).  This is a
    *literature window*, not a measurement of the HOPPER basin.

``size_scale``
    ``<D> / D_min`` of the DEM/PBE floe field against the value implied by a
    power law in the same window, ``(alpha-1)/(alpha-2)``.  It tests that the
    model's own size scale is consistent with the accepted exponent range; it is
    not a point-value agreement with the basin.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

__all__ = [
    "HOPPER_DATA_PATH",
    "HOPPER_SOURCE",
    "HOPPER_TOLERANCE",
    "HOPPER_ALPHA_WINDOW",
    "HopperBenchmark",
    "load_hopper_census",
    "compare_mean_size",
    "implied_size_ratio",
    "hopper_report",
]

#: Where the measured HOPPER census is expected.  Two columns: diameter, count.
HOPPER_DATA_PATH = Path("data/hopper_lu2016_floe_size.csv")

#: The reference the campaign names.
HOPPER_SOURCE = (
    "HOPPER ice-tank broken-ice floe-size census, Lu et al. (2016); "
    "raw tabulated census not present in this worktree"
)

#: Campaign tolerance on the mean floe size.
HOPPER_TOLERANCE = 0.30

#: Broken-ice FSD exponent window already accepted by this repository
#: (``tensordem.broken_ice_micro.fsd.FSD_FIT_ALPHA_RANGE``).
HOPPER_ALPHA_WINDOW = (2.5, 3.5)


@dataclass
class HopperBenchmark:
    """Outcome of the model-test comparison, with its evidence grade."""

    status: str
    mean_size_model: float
    mean_size_reference: float | None
    relative_error: float | None
    tolerance: float
    passes: bool | None
    source: str
    data_path: str
    reason: str
    n_reference: int = 0

    def as_dict(self) -> dict:
        return dict(
            status=self.status,
            mean_size_model=float(self.mean_size_model),
            mean_size_reference=None if self.mean_size_reference is None
            else float(self.mean_size_reference),
            relative_error=None if self.relative_error is None else float(self.relative_error),
            tolerance=float(self.tolerance),
            passes=self.passes,
            source=self.source,
            data_path=str(self.data_path),
            reason=self.reason,
            n_reference=int(self.n_reference),
            evidence_grade="REAL" if self.status == "COMPARED" else "WITHHELD",
        )


def load_hopper_census(path: Path | str = HOPPER_DATA_PATH) -> np.ndarray | None:
    """Expand a ``diameter,count`` census file into a sample of diameters.

    Returns ``None`` when the file is absent -- the honest answer for a worktree
    without the measurement -- and raises on a malformed file, because a file that
    exists but cannot be parsed must not be mistaken for "no data".
    """
    path = Path(path)
    if not path.exists():
        return None
    diameters, counts = [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        try:
            d_key = fields["diameter_m"]
            c_key = fields["count"]
        except KeyError as error:
            raise ValueError(
                f"{path} must have 'diameter_m' and 'count' columns, got {reader.fieldnames}"
            ) from error
        for row in reader:
            diameters.append(float(row[d_key]))
            counts.append(float(row[c_key]))
    if not diameters:
        raise ValueError(f"{path} contains no rows")
    sample = np.repeat(np.asarray(diameters, dtype=np.float64),
                       np.asarray(counts, dtype=np.int64))
    if sample.size == 0 or np.any(sample <= 0.0):
        raise ValueError(f"{path} contains non-positive diameters or counts")
    return sample


def compare_mean_size(model_sizes, reference_sizes, *, tolerance: float = HOPPER_TOLERANCE) -> dict:
    """Relative error of the mean floe size, with the campaign tolerance."""
    model_sizes = np.asarray(model_sizes, dtype=np.float64).reshape(-1)
    reference_sizes = np.asarray(reference_sizes, dtype=np.float64).reshape(-1)
    if model_sizes.size == 0 or reference_sizes.size == 0:
        raise ValueError("both samples must be non-empty")
    model = float(model_sizes.mean())
    reference = float(reference_sizes.mean())
    error = abs(model - reference) / reference
    return dict(
        mean_model=model,
        mean_reference=reference,
        relative_error=float(error),
        tolerance=float(tolerance),
        passes=bool(error < tolerance),
        n_model=int(model_sizes.size),
        n_reference=int(reference_sizes.size),
    )


def implied_size_ratio(alpha: float) -> float:
    """``<D>/D_min`` of a ``D**-alpha`` law truncated at its lower cutoff."""
    if alpha <= 2.0:
        raise ValueError("the first moment diverges for alpha <= 2")
    return (alpha - 1.0) / (alpha - 2.0)


def hopper_report(model_sizes, *, path: Path | str = HOPPER_DATA_PATH) -> HopperBenchmark:
    """Run the comparison, or say precisely why it cannot be run."""
    model_sizes = np.asarray(model_sizes, dtype=np.float64).reshape(-1)
    if model_sizes.size == 0:
        raise ValueError("the model census is empty")
    model_mean = float(model_sizes.mean())
    reference = load_hopper_census(path)
    if reference is None:
        return HopperBenchmark(
            status="WITHHELD",
            mean_size_model=model_mean,
            mean_size_reference=None,
            relative_error=None,
            tolerance=HOPPER_TOLERANCE,
            passes=None,
            source=HOPPER_SOURCE,
            data_path=str(path),
            reason=(
                "the measured HOPPER/Lu-2016 floe-size census is not in this worktree; "
                "the criterion cannot be evaluated and is reported as WITHHELD rather "
                "than replaced by a fabricated reference"
            ),
        )
    result = compare_mean_size(model_sizes, reference)
    return HopperBenchmark(
        status="COMPARED",
        mean_size_model=result["mean_model"],
        mean_size_reference=result["mean_reference"],
        relative_error=result["relative_error"],
        tolerance=HOPPER_TOLERANCE,
        passes=result["passes"],
        source=HOPPER_SOURCE,
        data_path=str(path),
        reason="computed against the measured census found in the worktree",
        n_reference=int(reference.size),
    )