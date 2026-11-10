"""JSON output schema for keel porosity evolution results.

The output schema follows the campaign node requirements:
{
    "porosity_evolution": {
        "n_0": float,           # initial porosity
        "n_inf": float,         # asymptotic porosity
        "tau": float,           # consolidation time scale (s)
        "r_squared": float,     # fit quality
        "time_series": [        # n(t) data points
            {"t": float, "n_dem": float, "n_fit": float},
            ...
        ]
    },
    "consolidation_time": {
        "tau": float,           # primary time scale
        "t_50": float,          # time to 50% consolidation
        "t_90": float,          # time to 90% consolidation
        "wave_params": {...},   # τ(A) parameterization
        "gradation_params": {...}  # τ(gradation) parameterization
    },
    "resistance_decay": {
        "keel_resistance_factor": [...],  # R(t)/R(0) vs time
        "description": "..."
    }
}
"""
from __future__ import annotations

from typing import Dict, Any, List
import json

from .evolution import PorosityEvolutionResult
from .sensitivity import SensitivityResult

__all__ = [
    "OUTPUT_SCHEMA",
    "build_output_json",
]

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["porosity_evolution", "consolidation_time", "resistance_decay"],
    "properties": {
        "porosity_evolution": {
            "type": "object",
            "required": ["n_0", "n_inf", "tau", "r_squared", "time_series"],
            "properties": {
                "n_0": {"type": "number", "description": "Initial porosity"},
                "n_inf": {"type": "number", "description": "Asymptotic porosity"},
                "tau": {"type": "number", "description": "Consolidation time scale (s)"},
                "r_squared": {"type": "number", "description": "Fit quality R²"},
                "time_series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "t": {"type": "number"},
                            "n_dem": {"type": "number"},
                            "n_fit": {"type": "number"},
                        },
                    },
                },
            },
        },
        "consolidation_time": {
            "type": "object",
            "required": ["tau", "t_50", "t_90"],
            "properties": {
                "tau": {"type": "number"},
                "t_50": {"type": "number", "description": "Time to 50% consolidation (s)"},
                "t_90": {"type": "number", "description": "Time to 90% consolidation (s)"},
                "wave_params": {"type": "object"},
                "gradation_params": {"type": "object"},
            },
        },
        "resistance_decay": {
            "type": "object",
            "properties": {
                "keel_resistance_factor": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "t": {"type": "number"},
                            "R_ratio": {"type": "number"},
                        },
                    },
                },
                "description": {"type": "string"},
            },
        },
    },
}


def build_output_json(
    evolution: PorosityEvolutionResult,
    sensitivity: SensitivityResult = None,
) -> Dict[str, Any]:
    """Build the output JSON from evolution and sensitivity results.

    Parameters
    ----------
    evolution : PorosityEvolutionResult
        Porosity evolution analysis result.
    sensitivity : SensitivityResult, optional
        Sensitivity analysis result.

    Returns
    -------
    dict
        JSON-serializable output dictionary.
    """
    # Time series
    time_series = []
    for i in range(len(evolution.time)):
        time_series.append({
            "t": float(evolution.time[i]),
            "n_dem": float(evolution.porosity_dem[i]),
            "n_fit": float(evolution.porosity_fit[i]),
        })

    # Consolidation times
    tau = evolution.tau
    t_50 = tau * math.log(2)  # n(t_50) = n_inf + 0.5*(n_0 - n_inf)
    t_90 = tau * math.log(10)  # n(t_90) = n_inf + 0.1*(n_0 - n_inf)

    # Resistance decay (simplified model: R(t)/R(0) ∝ n(t)/n_0)
    # As porosity decreases, the keel becomes more consolidated and
    # its resistance to penetration increases
    resistance_factor = []
    for i in range(len(evolution.time)):
        n_t = evolution.porosity_fit[i]
        # Simple model: resistance increases as porosity decreases
        # R(t)/R(0) = (1 - n(t)) / (1 - n_0)
        R_ratio = (1 - n_t) / (1 - evolution.n_0)
        resistance_factor.append({
            "t": float(evolution.time[i]),
            "R_ratio": float(R_ratio),
        })

    output = {
        "porosity_evolution": {
            "n_0": float(evolution.n_0),
            "n_inf": float(evolution.n_inf),
            "tau": float(evolution.tau),
            "r_squared": float(evolution.r_squared),
            "time_series": time_series,
        },
        "consolidation_time": {
            "tau": float(tau),
            "t_50": float(t_50),
            "t_90": float(t_90),
        },
        "resistance_decay": {
            "keel_resistance_factor": resistance_factor,
            "description": (
                "Keel resistance factor R(t)/R(0) as a function of time. "
                "As the keel consolidates (porosity decreases), its resistance "
                "to ship penetration increases. Model: R(t)/R(0) = (1-n(t))/(1-n_0)."
            ),
        },
    }

    # Add sensitivity results if available
    if sensitivity is not None:
        output["consolidation_time"]["wave_params"] = sensitivity.tau_wave_params
        output["consolidation_time"]["gradation_params"] = sensitivity.tau_gradation_params
        output["sensitivity"] = {
            "wave_sweep": sensitivity.wave_sweep,
            "gradation_sweep": sensitivity.gradation_sweep,
        }

    return output


# Need math for log
import math
