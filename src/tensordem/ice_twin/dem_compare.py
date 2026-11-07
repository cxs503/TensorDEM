"""Online comparison of the inverted ice load against the DEM forecast.

For each ice event the twin compares the inverted total force ``f_hat(t)``
with the DEM-predicted force ``f(t)`` on the common time grid and reports

* Pearson correlation coefficient,
* peak error  ``max |f_hat - f| / max |f|``,
* energy error ``|int f_hat^2 - int f^2| / int f^2``.

If the peak or energy error exceeds the correction threshold (30 % by
campaign specification) the twin flags the DEM contact model for
recalibration (adjusting the contact stiffness / breaking strain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LoadComparison", "compare_loads", "correction_needed"]


@dataclass(frozen=True)
class LoadComparison:
    correlation: float
    peak_error: float
    energy_error: float

    def as_dict(self) -> dict:
        return {"correlation": self.correlation,
                "peak_error": self.peak_error,
                "energy_error": self.energy_error}


def compare_loads(f_hat, f, dt: float) -> LoadComparison:
    """Compare inverted vs predicted ice-load time histories (SI, same grid)."""
    a = np.asarray(f_hat, dtype=float).ravel()
    b = np.asarray(f, dtype=float).ravel()
    if a.size != b.size or a.size < 4:
        raise ValueError("histories must match and have >= 4 samples")
    if np.allclose(b, 0.0):
        return LoadComparison(0.0, 0.0 if np.allclose(a, 0.0) else 1.0,
                            0.0 if np.allclose(a, 0.0) else 1.0)
    corr = float(np.corrcoef(a, b)[0, 1])
    peak = float(np.abs(a - b).max() / np.abs(b).max())
    e_b = float(np.trapezoid(b * b, dx=dt))
    e_a = float(np.trapezoid(a * a, dx=dt))
    energy = abs(e_a - e_b) / e_b if e_b > 0 else 0.0
    return LoadComparison(corr, peak, energy)


def correction_needed(res: LoadComparison, threshold: float = 0.30) -> bool:
    """True when the DEM contact model must be recalibrated."""
    return res.peak_error > threshold or res.energy_error > threshold
