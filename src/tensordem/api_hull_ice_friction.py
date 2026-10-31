"""API contract for ``POST /calculations/hull-ice-friction``.

The ship platform mounts this as a router; the handler is deliberately free of
web-framework imports so that it can be unit tested without a server::

    from tensordem.api_hull_ice_friction import handle_hull_ice_friction
    response = handle_hull_ice_friction(request_body)

and, in the platform application factory::

    from tensordem.api_hull_ice_friction import create_router
    app.include_router(create_router(prefix="/api/v1"), prefix="")

The response carries the temperature-dependent friction coefficients, the
level-ice resistance at each requested temperature, the warm/cold comparison
required by the acceptance criteria, and a machine-checkable criteria block.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from .hull_ice_friction import (
    MU_LITERATURE_HIGH,
    MU_LITERATURE_LOW,
    MU_REFERENCE_TEMPERATURE_C,
    T_MELT_C,
    _require_finite,
    friction_temperature_model,
    micro_contact_friction,
)
from .ice_resistance import (
    VesselIceGeometry,
    ice_resistance_components,
    resistance_temperature_comparison,
)

PROTOCOL = "tensordem-hull-ice-friction-v1"

DEFAULT_REQUEST: dict[str, Any] = {
    "temperatures_C": [-5.0, -10.0, -15.0, -20.0, -25.0],
    "reference_temperature_C": MU_REFERENCE_TEMPERATURE_C,
    "mu_parameters": None,
    "interface_law": {"mu_ref": MU_LITERATURE_LOW, "exponent": 0.45},
    "vessel": {},
    "comparison": {"warm_temperature_C": -5.0, "cold_temperature_C": -20.0},
}


def _parse_request(payload: Optional[dict]) -> dict:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    unknown = set(payload) - set(DEFAULT_REQUEST)
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    request = {**DEFAULT_REQUEST, **payload}

    temperatures = request["temperatures_C"]
    if not isinstance(temperatures, (list, tuple)) or not temperatures:
        raise ValueError("temperatures_C must be a non-empty list")
    for value in temperatures:
        _require_finite(value, "temperatures_C")
        if value > T_MELT_C:
            raise ValueError("every temperature must be at or below the melting point")
    request["temperatures_C"] = [float(value) for value in temperatures]

    mu_parameters = request["mu_parameters"]
    if mu_parameters is not None:
        if not isinstance(mu_parameters, dict):
            raise ValueError("mu_parameters must be an object or null")
        missing = {"mu0", "alpha", "beta"} - set(mu_parameters)
        if missing:
            raise ValueError(f"mu_parameters is missing {sorted(missing)}")
        request["mu_parameters"] = {
            key: float(mu_parameters[key]) for key in ("mu0", "alpha", "beta")
        }

    vessel = request["vessel"]
    if not isinstance(vessel, dict):
        raise ValueError("vessel must be an object")
    allowed = set(VesselIceGeometry.__dataclass_fields__)
    unknown = set(vessel) - allowed
    if unknown:
        raise ValueError(f"unknown vessel fields: {sorted(unknown)}")
    request["vessel"] = {key: float(value) for key, value in vessel.items()}

    law = request["interface_law"]
    if not isinstance(law, dict):
        raise ValueError("interface_law must be an object")
    mu_ref = law.get("mu_ref", MU_LITERATURE_LOW)
    exponent = law.get("exponent", 0.45)
    _require_finite(mu_ref, "interface_law.mu_ref")
    _require_finite(exponent, "interface_law.exponent")
    request["interface_law"] = {"mu_ref": float(mu_ref), "exponent": float(exponent)}

    comparison = request["comparison"]
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    request["comparison"] = {
        "warm_temperature_C": float(comparison.get("warm_temperature_C", -5.0)),
        "cold_temperature_C": float(comparison.get("cold_temperature_C", -20.0)),
    }
    return request


def hull_ice_friction(
    temperature_C: float,
    *,
    mu_parameters: Optional[dict] = None,
    interface_law: Optional[dict] = None,
) -> float:
    """Friction coefficient at a temperature from the calibrated law or the law."""
    _require_finite(temperature_C, "temperature_C")
    if mu_parameters is not None:
        return float(friction_temperature_model(temperature_C, **mu_parameters))
    law = interface_law or {}
    return micro_contact_friction(
        temperature_C,
        mu_ref=law.get("mu_ref", MU_LITERATURE_LOW),
        exponent=law.get("exponent", 0.45),
    )


def handle_hull_ice_friction(payload: Optional[dict] = None) -> dict:
    """Validate a request and compute the hull-ice friction calculation."""
    request = _parse_request(payload)
    geometry = VesselIceGeometry(**request["vessel"])
    mu_parameters = request["mu_parameters"]
    interface_law = request["interface_law"]

    friction = [
        {
            "temperature_C": temperature,
            "friction_coefficient": hull_ice_friction(
                temperature,
                mu_parameters=mu_parameters,
                interface_law=interface_law,
            ),
        }
        for temperature in request["temperatures_C"]
    ]
    for entry in friction:
        entry["resistance"] = ice_resistance_components(
            entry["temperature_C"],
            geometry=geometry,
            mu=entry["friction_coefficient"],
        )

    ordered = sorted(friction, key=lambda entry: entry["temperature_C"])
    # ordered is ascending in temperature: the first entry is the coldest
    monotone = bool(
        all(
            colder["friction_coefficient"] > warmer["friction_coefficient"]
            for colder, warmer in zip(ordered[:-1], ordered[1:])
        )
    )

    comparison = resistance_temperature_comparison(
        request["comparison"]["warm_temperature_C"],
        request["comparison"]["cold_temperature_C"],
        geometry=geometry,
        mu_parameters=mu_parameters if mu_parameters is not None else None,
    )
    if mu_parameters is None:
        # fall back to the interface law for the comparison states
        warm = comparison["warm"]["temperature_C"]
        cold = comparison["cold"]["temperature_C"]
        comparison = resistance_temperature_comparison(
            warm,
            cold,
            geometry=geometry,
            mu_parameters=_equivalent_power_law(interface_law),
        )

    reference_mu = hull_ice_friction(
        MU_REFERENCE_TEMPERATURE_C,
        mu_parameters=mu_parameters,
        interface_law=interface_law,
    )
    band_centre = 0.5 * (MU_LITERATURE_LOW + MU_LITERATURE_HIGH)
    within_band = MU_LITERATURE_LOW <= reference_mu <= MU_LITERATURE_HIGH

    criteria = {
        "friction_monotone_decreasing_in_temperature": monotone,
        "reference_friction_within_literature_band": {
            "temperature_C": MU_REFERENCE_TEMPERATURE_C,
            "value": reference_mu,
            "band": [MU_LITERATURE_LOW, MU_LITERATURE_HIGH],
            "relative_error_to_band_centre": abs(reference_mu - band_centre) / band_centre,
            "within_band": within_band,
        },
        "resistance_increase_cold_vs_warm": {
            "warm_temperature_C": comparison["warm"]["temperature_C"],
            "cold_temperature_C": comparison["cold"]["temperature_C"],
            "relative_increase": comparison["delta_relative_to_warm"],
            "greater_than_10_percent": comparison["delta_relative_to_warm"] > 0.10,
        },
    }

    return {
        "status": "ok",
        "protocol": PROTOCOL,
        "evidence": {
            "friction_law": "REAL: power-law adhesion-controlled ice/steel interfacial shear law",
            "friction_calibration": (
                "REAL: DEM sliding-friction calibration"
                if mu_parameters is not None
                else "literature law only (no DEM calibration supplied)"
            ),
            "resistance_model": "REAL: Lindqvist-type decomposition with explicit hull friction term",
            "absolute_resistance_validation": (
                "WITHHELD: published empirical formulas absorb hull friction into "
                "fitted coefficients; no full-scale geometry/scale conversion is "
                "included in this endpoint"
            ),
        },
        "request": {
            "temperatures_C": request["temperatures_C"],
            "mu_parameters": mu_parameters,
            "interface_law": interface_law,
            "vessel": request["vessel"],
            "comparison": request["comparison"],
        },
        "friction": friction,
        "temperature_comparison": comparison,
        "criteria": criteria,
    }


def _equivalent_power_law(interface_law: dict) -> dict:
    """Express the interfacial power law as ``mu0 + alpha |T|^beta``.

    ``mu_ref * (|T|/|T_ref|)**exponent = alpha * |T|**exponent`` with
    ``alpha = mu_ref / |T_ref|**exponent`` and ``mu0 = 0``.
    """
    mu_ref = interface_law.get("mu_ref", MU_LITERATURE_LOW)
    exponent = interface_law.get("exponent", 0.45)
    alpha = mu_ref / abs(MU_REFERENCE_TEMPERATURE_C) ** exponent
    return {"mu0": 0.0, "alpha": alpha, "beta": exponent}


def create_router(prefix: str = ""):
    """Build a FastAPI router exposing ``POST /calculations/hull-ice-friction``."""
    from fastapi import APIRouter, Body

    router = APIRouter()

    @router.post(f"{prefix}/calculations/hull-ice-friction")
    def _endpoint(payload: dict = Body(default_factory=dict)) -> dict:  # pragma: no cover
        return handle_hull_ice_friction(payload)

    return router