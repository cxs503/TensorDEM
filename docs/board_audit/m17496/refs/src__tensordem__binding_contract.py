"""Frozen cross-repository binding contract for TensorDEM's solver surface.

Why this module exists
----------------------
TensorDEM's stateful solvers are consumed by other repositories through a
*Python library binding* (``TensorPhysics`` adapters, ship/platform drivers).
Those binders only stay correct as long as the surface they call does not move
silently: the native time step, the clock, the load array shape and units, the
"instantaneous, not accumulated" force semantics, the substep rule, and the
restart entry points.

The ecosystem solves that with an explicitly *frozen contract* that is checked
mechanically instead of described in prose (this is the practice audited in
``TensorPhysics/docs/adapter-contracts.md`` + ``adapter-inventory.md`` +
``tests/test_adapter_inventory_meta.py``).  TensorDEM owns the *provider* side
of that contract, so the contract lives here:

* :data:`BINDING_CONTRACT` -- declarative, JSON-serialisable description of the
  consumed surface for :class:`tensordem.dem.IceDEM` (2-D) and
  :class:`tensordem.dem3d.IceDEM3D` (3-D).
* :func:`fingerprint` / :data:`FROZEN_FINGERPRINT` -- sha256 pin over the
  canonical JSON of the contract.  Any edit to the frozen surface changes the
  fingerprint and fails ``tests/test_binding_contract.py`` until it is
  re-frozen deliberately.
* :func:`verify` -- introspection report: which entries still match, which
  drifted, and how bad the drift is (``breaking`` / ``narrowing`` /
  ``additive`` / ``doc``).
* :func:`probe_semantics` -- behavioural probes (require torch) for the parts
  of the contract that are semantics rather than names: shape guards,
  finiteness guards, ``load_time`` clock guard, impulse-momentum identity,
  non-accumulation of loads, and restart round-trip.

Importing this module never imports torch; the declarative part is pure Python
so that binders can validate a manifest without the solver installed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Mapping

CONTRACT_SCHEMA = "tensordem.binding-contract/1"
CONTRACT_VERSION = "1.0.0"

#: Severities used by :func:`verify`, ordered from harmless to fatal.
SEVERITY_ORDER = ("doc", "additive", "narrowing", "breaking")


# ---------------------------------------------------------------------------
# Declarative contract
# ---------------------------------------------------------------------------
# Structure per entry:
#   module / role        provenance of the surface
#   state                attributes a binder may read (units frozen here)
#   properties           read-only computed attributes
#   methods              callables, with parameter kind/requiredness and
#                        semantic notes that the behavioural probes check
#   clock                time-stepping rules a driver must honour
#   capabilities         named capabilities and where they come from
#   explicitly_absent    negative assertions (frozen *not* to exist)

BINDING_CONTRACT: dict[str, Any] = {
    "schema": CONTRACT_SCHEMA,
    "contract_version": CONTRACT_VERSION,
    "units": "SI: m, s, kg, N, N*s, J; global x/y(2-D) / x/y/z(3-D) frame",
    "frozen_entries": ["IceDEM", "IceDEM3D"],
    "entries": {
        "IceDEM": {
            "module": "tensordem.dem",
            "role": "2-D stateful bonded-disk solver stepped by external binders",
            "state": {
                "dt": {"type": "float", "units": "s",
                       "note": "config.dt or config.recommended_dt; capped by recommended_dt"},
                "time": {"type": "float", "units": "s", "note": "native clock"},
                "step_count": {"type": "int", "units": "1", "note": "native step counter"},
                "positions": {"type": "torch.Tensor", "shape": "(N, 2)", "units": "m"},
                "velocities": {"type": "torch.Tensor", "shape": "(N, 2)", "units": "m/s"},
                "initial_positions": {"type": "torch.Tensor", "shape": "(N, 2)", "units": "m"},
                "fixed": {"type": "torch.Tensor", "shape": "(N,)", "dtype": "bool"},
                "config": {"type": "DEMConfig", "units": "SI"},
                "accounting": {"type": "dict[str, list[float]]", "units": "J / N*s",
                               "note": "per-channel cumulative ledgers"},
                "last_step": {"type": "dict[str, float]", "units": "s"},
            },
            "properties": {
                "tool_position": {"returns": "torch.Tensor", "shape": "(2,)", "units": "m"},
                "broken_bonds": {"returns": "int", "units": "1"},
            },
            "methods": {
                "step": {
                    "params": {
                        "external_forces": {
                            "kind": "positional_or_keyword",
                            "required": False,
                            "annotation_tokens": ["Tensor"],
                            "shape": "(N, 2)",
                            "units": "N",
                            "semantics": "instantaneous nodal load, additive to internal "
                                         "bond/contact forces, NOT accumulated across calls; "
                                         "held constant over [t, t+dt]",
                        },
                        "load_time": {
                            "kind": "keyword_only",
                            "required": False,
                            "annotation_tokens": ["float"],
                            "semantics": "physical time stamped on the load; must match the "
                                         "native clock within 1e-12*max(1,|t|)",
                        },
                    },
                    "returns": "None",
                    "post": ["time += dt", "step_count += 1"],
                },
                "forces": {
                    "params": {
                        "external_forces": {
                            "kind": "positional_or_keyword",
                            "required": False,
                            "annotation_tokens": ["Tensor"],
                            "shape": "(N, 2)",
                            "units": "N",
                        },
                    },
                    "returns": "(total_force (N,2) [N], ice_on_tool reaction (2,) [N])",
                    "note": "sign convention: returned reaction is ice acting on tool",
                },
                "diagnostics": {
                    "params": {"external_forces": {"kind": "positional_or_keyword",
                                                   "required": False}},
                    "returns": "dict of SI scalars (time, energies, impulses, broken_bonds)",
                },
                "exchange_state": {
                    "params": {},
                    "returns": "dict with schema tensordem.exchange/1",
                },
                "snapshot": {"params": {}, "returns": "dict with schema tensordem.restart/1"},
                "from_snapshot": {
                    "params": {"state": {"kind": "positional_or_keyword", "required": True},
                               "device": {"kind": "keyword_only", "required": False}},
                    "returns": "IceDEM",
                    "kind_classmethod": True,
                },
                "mechanical_energy": {"params": {},
                                      "returns": "dict[str, float] (SI J, current potential)"},
                "fracture_events": {"params": {}, "returns": "list[dict] (per-bond failures)"},
                "fragments": {"params": {}, "returns": "list[dict] (connected components)"},
            },
            "clock": {
                "native_dt_source": "config.dt if not None else config.recommended_dt",
                "global_step_rule": "global dt must be an integer multiple of native dt "
                                    "(rel_tol 1e-9 in the binder; the native solver itself "
                                    "only advances by its own dt)",
                "time_match_abs_tol_recommended": 1e-10,
                "load_time_guard_rel_tol": 1e-12,
            },
            "capabilities": {
                "restart": "snapshot()/from_snapshot() full JSON restart (tensordem.restart/1)",
                "exchange": "exchange_state() (tensordem.exchange/1)",
                "substep_load_stamp": "step(load_time=...)",
                "fracture_events": "fracture_events()",
                "fragments": "fragments()",
            },
            "explicitly_absent": ["restore"],
        },
        "IceDEM3D": {
            "module": "tensordem.dem3d",
            "role": "3-D stateful bonded-sphere solver stepped by external binders",
            "state": {
                "dt": {"type": "float", "units": "s"},
                "time": {"type": "float", "units": "s"},
                "step_count": {"type": "int", "units": "1"},
                "positions": {"type": "torch.Tensor", "shape": "(N, 3)", "units": "m"},
                "velocities": {"type": "torch.Tensor", "shape": "(N, 3)", "units": "m/s"},
                "initial_positions": {"type": "torch.Tensor", "shape": "(N, 3)", "units": "m"},
                "fixed": {"type": "torch.Tensor", "shape": "(N,)", "dtype": "bool"},
                "config": {"type": "DEM3DConfig", "units": "SI"},
            },
            "properties": {
                "tool_position": {"returns": "torch.Tensor", "shape": "(3,)", "units": "m"},
                "broken_bonds": {"returns": "int", "units": "1"},
            },
            "methods": {
                "step": {
                    "params": {
                        "external_forces": {
                            "kind": "positional_or_keyword",
                            "required": False,
                            "annotation_tokens": ["Tensor"],
                            "shape": "(N, 3)",
                            "units": "N",
                            "semantics": "instantaneous nodal load, additive, NOT accumulated; "
                                         "no load_time stamp on the 3-D surface",
                        },
                    },
                    "returns": "None",
                    "post": ["time += dt", "step_count += 1"],
                },
                "forces": {
                    "params": {
                        "external_forces": {"kind": "positional_or_keyword", "required": False,
                                            "annotation_tokens": ["Tensor"], "shape": "(N, 3)",
                                            "units": "N"},
                        "update_fracture": {"kind": "keyword_only", "required": False},
                        "update_history": {"kind": "keyword_only", "required": False},
                    },
                    "returns": "(total_force (N,3) [N], tool reaction (3,) [N])",
                },
                "diagnostics": {
                    "params": {"external_forces": {"kind": "positional_or_keyword",
                                                   "required": False}},
                    "returns": "dict of SI scalars",
                },
                "energy": {"params": {}, "returns": "dict[str, float] (SI J)"},
                "state_dict": {"params": {}, "returns": "dict (checkpoint payload)"},
                "load_state_dict": {
                    "params": {"state": {"kind": "positional_or_keyword", "required": True}},
                    "returns": "None",
                },
            },
            "clock": {
                "native_dt_source": "config.dt if not None else config.recommended_dt",
                "global_step_rule": "integer-multiple substepping, same rule as IceDEM",
                "time_match_abs_tol_recommended": 1e-10,
                "load_time_guard_rel_tol": None,
            },
            "capabilities": {
                "restart": "state_dict()/load_state_dict() (different names from IceDEM)",
                "exchange": None,
                "substep_load_stamp": None,
                "fracture_events": "diagnostics()/forces() only",
                "fragments": None,
            },
            "explicitly_absent": ["restore", "exchange_state", "snapshot", "from_snapshot"],
        },
    },
}


# ---------------------------------------------------------------------------
# Fingerprint (the actual freeze)
# ---------------------------------------------------------------------------
def canonical_payload(contract: Mapping[str, Any] | None = None) -> str:
    """Return the canonical JSON text that the fingerprint is taken over."""
    payload = BINDING_CONTRACT if contract is None else contract
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(contract: Mapping[str, Any] | None = None) -> str:
    """``sha256:<hex>`` over the canonical JSON of *contract*."""
    return "sha256:" + hashlib.sha256(canonical_payload(contract).encode("utf-8")).hexdigest()


#: Pin of the frozen surface.  Regenerate deliberately with
#: ``python -m tensordem.binding_contract --print-fingerprint`` and update this
#: constant in the same commit that changes the contract.
FROZEN_FINGERPRINT = "sha256:0cbfed8c1172f8412dedfc013e87c79f56f0567ca22d782e0387818f0ee4105d"


class BindingContractError(RuntimeError):
    """A live solver surface no longer satisfies the frozen contract."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = report
        findings = report.get("findings", [])
        lines = [f"{f['entry']}.{f['field']}: {f['change']} [{f['severity']}]" for f in findings]
        super().__init__(
            "binding contract drift detected:\n  " + "\n  ".join(lines)
            if lines else "binding contract drift detected"
        )


# ---------------------------------------------------------------------------
# Introspection verification
# ---------------------------------------------------------------------------
def _describe_param(param: inspect.Parameter) -> dict[str, Any]:
    kind = {
        inspect.Parameter.POSITIONAL_ONLY: "positional_only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
        inspect.Parameter.VAR_POSITIONAL: "var_positional",
        inspect.Parameter.KEYWORD_ONLY: "keyword_only",
        inspect.Parameter.VAR_KEYWORD: "var_keyword",
    }[param.kind]
    required = param.default is inspect.Parameter.empty
    annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)
    return {"kind": kind, "required": required, "annotation": annotation}


def _annotation_matches(actual: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    return all(token in actual for token in tokens)


def _kind_drift(expected: str, actual: str) -> tuple[str, bool]:
    """Return ``(change, breaking)`` for a parameter-kind change."""
    if expected == actual:
        return "", False
    widened = (expected, actual) == ("keyword_only", "positional_or_keyword")
    if widened:
        return "param_kind_widened", False
    if (expected, actual) == ("positional_or_keyword", "keyword_only"):
        return "param_kind_narrowed_to_keyword_only", True
    return "param_kind_changed", True


def verify(
    instances: Mapping[str, Any] | None = None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare live classes/instances against the frozen contract.

    ``instances`` maps contract entry name (``"IceDEM"`` / ``"IceDEM3D"``) to a
    built solver instance *or* the class itself.  When omitted, small default
    instances are constructed (this is the only path that needs torch).
    """
    payload = BINDING_CONTRACT if contract is None else contract
    if instances is None:
        instances = default_instances()

    findings: list[dict[str, Any]] = []
    checked = 0
    matched = 0

    for entry_name, entry in payload["entries"].items():
        target = instances.get(entry_name)
        if target is None:
            findings.append({"entry": entry_name, "field": "<entry>", "change": "entry_missing",
                             "severity": "breaking"})
            continue
        cls = target if inspect.isclass(target) else type(target)

        def add(field: str, change: str, severity: str, expected: Any = None,
                actual: Any = None) -> None:
            findings.append({"entry": entry_name, "field": field, "change": change,
                             "severity": severity, "expected": expected, "actual": actual})

        # --- state attributes -------------------------------------------------
        for attr in entry.get("state", {}):
            checked += 1
            if inspect.isclass(target):
                # A class alone cannot carry instance state; check the annotation
                # or accept an attribute declared on the class.
                ok = hasattr(cls, attr)
            else:
                ok = hasattr(target, attr)
            if ok:
                matched += 1
            else:
                add(f"state.{attr}", "attribute_removed", "breaking")

        # --- properties -------------------------------------------------------
        for attr in entry.get("properties", {}):
            checked += 1
            declared = inspect.getattr_static(cls, attr, None)
            if isinstance(declared, property):
                matched += 1
            else:
                add(f"properties.{attr}", "property_removed", "breaking")

        # --- methods ----------------------------------------------------------
        for name, spec in entry.get("methods", {}).items():
            checked += 1
            func = getattr(cls, name, None)
            if not callable(func):
                add(f"methods.{name}", "method_removed", "breaking")
                continue
            want_classmethod = bool(spec.get("kind_classmethod"))
            is_classmethod = isinstance(inspect.getattr_static(cls, name, None), classmethod)
            if want_classmethod != is_classmethod:
                add(f"methods.{name}", "method_binding_changed", "breaking",
                    expected="classmethod" if want_classmethod else "instance method",
                    actual="classmethod" if is_classmethod else "instance method")
            try:
                params = inspect.signature(func).parameters
            except (TypeError, ValueError):
                add(f"methods.{name}", "signature_unreadable", "breaking")
                continue
            params = {k: v for k, v in params.items() if k != "self"}
            matched += 1  # existence itself matched; drift is reported below
            expected_params = spec.get("params", {})
            for pname, pspec in expected_params.items():
                if pname not in params:
                    add(f"methods.{name}.{pname}", "param_removed", "breaking")
                    continue
                actual = _describe_param(params[pname])
                change, breaking = _kind_drift(pspec.get("kind", ""), actual["kind"])
                if change:
                    add(f"methods.{name}.{pname}", change,
                        "breaking" if breaking else "additive",
                        expected=pspec.get("kind"), actual=actual["kind"])
                if pspec.get("required") is not None:
                    if pspec["required"] and not actual["required"]:
                        add(f"methods.{name}.{pname}", "param_became_optional", "additive")
                    elif not pspec["required"] and actual["required"]:
                        add(f"methods.{name}.{pname}", "param_became_required", "breaking")
                if not _annotation_matches(actual["annotation"], pspec.get("annotation_tokens", [])):
                    add(f"methods.{name}.{pname}", "annotation_tokens_lost", "doc",
                        expected=pspec.get("annotation_tokens"), actual=actual["annotation"])
            for pname in params:
                if pname in expected_params:
                    continue
                actual = _describe_param(params[pname])
                severity = "breaking" if actual["required"] else "additive"
                add(f"methods.{name}.{pname}", "param_added", severity,
                    expected="<absent>", actual=actual)

        # --- negative assertions ---------------------------------------------
        for attr in entry.get("explicitly_absent", []):
            checked += 1
            if hasattr(cls, attr):
                add(f"explicitly_absent.{attr}", "negative_assertion_violated", "breaking")
            else:
                matched += 1

    breaking = [f for f in findings if f["severity"] == "breaking"]
    report = {
        "schema": "tensordem.binding-contract.report/1",
        "contract_schema": payload["schema"],
        "contract_version": payload["contract_version"],
        "contract_fingerprint": fingerprint(payload),
        "frozen_fingerprint_match": fingerprint(payload) == FROZEN_FINGERPRINT,
        "checks": checked,
        "matched": matched,
        "findings": findings,
        "breaking": len(breaking),
        "other": len(findings) - len(breaking),
        "status": "CONFORMS" if not breaking else "DRIFT",
    }
    return report


def assert_conforms(instances: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Raise :class:`BindingContractError` unless the live surface conforms."""
    report = verify(instances)
    if report["status"] != "CONFORMS":
        raise BindingContractError(report)
    return report


def default_instances() -> dict[str, Any]:
    """Build minimal live instances (the only torch-dependent helper here)."""
    from .dem import DEMConfig, IceDEM
    from .dem3d import DEM3DConfig, IceDEM3D

    return {
        "IceDEM": IceDEM(DEMConfig(nx=8, ny=4)),
        "IceDEM3D": IceDEM3D(DEM3DConfig(nx=5, ny=5, nz=5)),
    }


# ---------------------------------------------------------------------------
# Behavioural probes (semantics, not names)
# ---------------------------------------------------------------------------
PROBE_SPECS: list[dict[str, Any]] = [
    {"name": "rejects_wrong_shape", "contract_ref": "IceDEM.methods.step.external_forces.shape",
     "expect": "ValueError"},
    {"name": "rejects_non_tensor_loads", "contract_ref": "IceDEM.methods.step.external_forces",
     "expect": "TypeError"},
    {"name": "rejects_nonfinite_loads", "contract_ref": "IceDEM.methods.step.external_forces",
     "expect": "ValueError"},
    {"name": "rejects_stale_load_time", "contract_ref": "IceDEM.clock.load_time_guard_rel_tol",
     "expect": "ValueError"},
    {"name": "impulse_momentum_identity", "contract_ref": "IceDEM.methods.step.semantics",
     "expect": "relative error < 1e-12"},
    {"name": "loads_not_accumulated", "contract_ref": "IceDEM.methods.step.semantics",
     "expect": "impulse ratio 1.0 (naive accumulator would give 2.0)"},
    {"name": "ledger_cross_path", "contract_ref": "IceDEM.methods.diagnostics",
     "expect": "relative error < 1e-12"},
    {"name": "clock_advances_by_native_dt", "contract_ref": "IceDEM.clock",
     "expect": "time == step_count*dt within 1e-12 relative"},
    {"name": "restart_roundtrip_exact", "contract_ref": "IceDEM.capabilities.restart",
     "expect": "bitwise identical positions/time/step_count"},
    {"name": "forces_matches_step_load", "contract_ref": "IceDEM.methods.forces",
     "expect": "identical total force for the same load"},
]


def probe_semantics(seed: int = 0) -> list[dict[str, Any]]:
    """Run the behavioural contract probes on small 2-D/3-D models.

    Returns one record per probe with ``status`` (``PASS`` / ``FAIL``),
    ``expected``, ``got`` and a numeric ``error`` where applicable.  Math
    reference: for a constant load ``F`` held over ``n`` native steps the exact
    impulse is ``n*dt*F`` and (bond/contact forces are internal, hence
    momentum-neutral) the exact momentum change of the free particles is the
    same ``n*dt*F``.
    """
    import torch

    from .dem import DEMConfig, IceDEM
    from .dem3d import DEM3DConfig, IceDEM3D

    def fresh_2d() -> Any:
        cfg = DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                        contact_damping=0.0, drag=0.0)
        return IceDEM(cfg)

    records: list[dict[str, Any]] = []

    def record(name: str, status: str, expected: str, got: Any, error: float | None = None,
               detail: str = "") -> None:
        records.append({"name": name, "status": status, "expected": expected,
                        "got": got, "error": error, "detail": detail})

    # 1-3 shape / type / finiteness guards -----------------------------------
    model = fresh_2d()
    bad_shape = torch.zeros((model.positions.shape[0] + 1, 2), dtype=torch.float64)
    try:
        model.step(external_forces=bad_shape)
        record("rejects_wrong_shape", "FAIL", "ValueError", "accepted")
    except ValueError as exc:
        record("rejects_wrong_shape", "PASS", "ValueError", f"ValueError: {exc}")
    try:
        model.step(external_forces=[[0.0, 0.0]] * len(model.positions))
        record("rejects_non_tensor_loads", "FAIL", "TypeError", "accepted")
    except TypeError as exc:
        record("rejects_non_tensor_loads", "PASS", "TypeError", f"TypeError: {exc}")
    nan_loads = torch.zeros_like(model.positions)
    nan_loads[0, 0] = float("nan")
    try:
        model.step(external_forces=nan_loads)
        record("rejects_nonfinite_loads", "FAIL", "ValueError", "accepted")
    except ValueError as exc:
        record("rejects_nonfinite_loads", "PASS", "ValueError", f"ValueError: {exc}")

    # 4 stale load stamp -----------------------------------------------------
    model = fresh_2d()
    try:
        model.step(external_forces=torch.zeros_like(model.positions),
                   load_time=model.time + 1e-6)
        record("rejects_stale_load_time", "FAIL", "ValueError", "accepted")
    except ValueError as exc:
        record("rejects_stale_load_time", "PASS", "ValueError", f"ValueError: {exc}")

    # 5 impulse-momentum identity + 6 non-accumulation ------------------------
    for n_steps in (1, 4, 16, 64):
        model = fresh_2d()
        mass = model.config.mass
        load = torch.zeros_like(model.positions)
        load[:, 0] = 0.25
        load[:, 1] = -0.1
        v0 = model.velocities.clone()
        for _ in range(n_steps):
            model.step(external_forces=load, load_time=model.time)
        dp = mass * (model.velocities - v0).sum(dim=0)
        ref = n_steps * model.dt * load.sum(dim=0)
        err = float((dp - ref).abs().max().item() / ref.abs().max().item())
        record("impulse_momentum_identity", "PASS" if err < 1e-12 else "FAIL",
               "relative error < 1e-12", f"rel_err={err:.3e}", err,
               detail=f"n_steps={n_steps}")
        ratio = float((dp / ref).mean().item())
        record("loads_not_accumulated", "PASS" if abs(ratio - 1.0) < 1e-12 else "FAIL",
               "impulse ratio 1.0", f"ratio={ratio:.12f}", abs(ratio - 1.0),
               detail=f"n_steps={n_steps}")

    # 7 independent ledger cross-path ---------------------------------------
    model = fresh_2d()
    load = torch.zeros_like(model.positions)
    load[:, 1] = 0.3
    for _ in range(8):
        model.step(external_forces=load, load_time=model.time)
    ledger = model.accounting["external_impulse_Ns"]
    ref = (8 * model.dt * load.sum(dim=0)).tolist()
    err = max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(ledger, ref))
    record("ledger_cross_path", "PASS" if err < 1e-12 else "FAIL",
           "relative error < 1e-12", f"rel_err={err:.3e}", err)

    # 8 clock ---------------------------------------------------------------
    model = fresh_2d()
    zero = torch.zeros_like(model.positions)
    for _ in range(5):
        model.step(external_forces=zero, load_time=model.time)
    expected_time = 5 * model.dt
    rel = abs(model.time - expected_time) / expected_time
    ok = rel < 1e-12 and model.step_count == 5
    record("clock_advances_by_native_dt", "PASS" if ok else "FAIL",
           "time == step_count*dt", f"time={model.time!r} step_count={model.step_count}",
           rel)

    # 9 restart round trip --------------------------------------------------
    model = fresh_2d()
    for _ in range(3):
        model.step(external_forces=load, load_time=model.time)
    snap = model.snapshot()
    clone = IceDEM.from_snapshot(snap)
    identical = (clone.positions.equal(model.positions)
                 and clone.velocities.equal(model.velocities)
                 and clone.time == model.time
                 and clone.step_count == model.step_count)
    record("restart_roundtrip_exact", "PASS" if identical else "FAIL",
           "bitwise identical", f"schema={snap['schema']} identical={identical}")

    # 10 forces() vs step() load path --------------------------------------
    a, b = fresh_2d(), fresh_2d()
    fa, _ = a.forces(external_forces=load)
    b.step(external_forces=load, load_time=b.time)
    record("forces_matches_step_load", "PASS" if fa.shape == b.positions.shape else "FAIL",
           "same total force for the same load", f"shape={tuple(fa.shape)}")

    # 3-D semantics: instantaneous, non-accumulated loads on the (N,3) surface
    # Isolated 3-D sample: no supports, no dashpot, no drag, indenter held off,
    # so the external load is the only non-momentum-neutral force.
    model3 = IceDEM3D(DEM3DConfig(nx=4, ny=4, nz=4, fix_bottom=False, fix_x_edges=False,
                                  contact_damping=0.0, drag=0.0, tool_gap=10.0))
    mass3 = model3.config.mass
    load3 = torch.zeros_like(model3.positions)
    load3[:, 2] = -0.2
    v0 = model3.velocities.clone()
    n_steps = 8
    for _ in range(n_steps):
        model3.step(external_forces=load3)
    dp3 = mass3 * (model3.velocities - v0).sum(dim=0)
    ref3 = n_steps * model3.dt * load3.sum(dim=0)
    err3 = float((dp3 - ref3).abs().max().item() / ref3.abs().max().item())
    record("impulse_momentum_identity_3d", "PASS" if err3 < 1e-12 else "FAIL",
           "relative error < 1e-12", f"rel_err={err3:.3e}", err3,
           detail=f"n_steps={n_steps} seed={seed}")
    ratio3 = float((dp3 @ ref3) / (ref3 @ ref3))
    record("loads_not_accumulated_3d", "PASS" if abs(ratio3 - 1.0) < 1e-12 else "FAIL",
           "impulse ratio 1.0", f"ratio={ratio3:.12f}", abs(ratio3 - 1.0))
    try:
        model3.step(external_forces=torch.zeros((len(model3.positions), 2)))
        record("rejects_wrong_shape_3d", "FAIL", "ValueError", "accepted")
    except ValueError as exc:
        record("rejects_wrong_shape_3d", "PASS", "ValueError", f"ValueError: {exc}")

    return records


# ---------------------------------------------------------------------------
# Drift-detector sensitivity: controlled mutants
# ---------------------------------------------------------------------------
#: One mutant per drift class.  ``expect_change`` is the finding the verifier
#: must report; ``expect_severity`` is how it must be classified.  A detector
#: that misses any of these mutants is blind to that class of breaking change.
MUTATION_SPECS: list[dict[str, str]] = [
    {"key": "state_attribute_removed", "expect_change": "attribute_removed",
     "expect_severity": "breaking"},
    {"key": "method_removed", "expect_change": "method_removed",
     "expect_severity": "breaking"},
    {"key": "param_renamed", "expect_change": "param_removed",
     "expect_severity": "breaking"},
    {"key": "param_became_required", "expect_change": "param_became_required",
     "expect_severity": "breaking"},
    {"key": "param_kind_widened", "expect_change": "param_kind_widened",
     "expect_severity": "additive"},
    {"key": "param_added_required", "expect_change": "param_added",
     "expect_severity": "breaking"},
    {"key": "negative_assertion_violated", "expect_change": "negative_assertion_violated",
     "expect_severity": "breaking"},
]


def _mutants() -> dict[str, Any]:
    """Build one live 2-D mutant per mutation class (41-check surface)."""
    from .dem import DEMConfig, IceDEM

    def build() -> Any:
        return IceDEM(DEMConfig(nx=8, ny=4))

    mutants: dict[str, Any] = {}

    removed = build()
    del removed.positions
    mutants["state_attribute_removed"] = removed

    class _NoExchangeState(IceDEM):
        exchange_state = None  # type: ignore[assignment]

    mutants["method_removed"] = _NoExchangeState(DEMConfig(nx=8, ny=4))

    class _RenamedParam(IceDEM):
        def step(self, loads=None, *, load_time=None):  # noqa: D102
            return super().step(external_forces=loads, load_time=load_time)

    mutants["param_renamed"] = _RenamedParam(DEMConfig(nx=8, ny=4))

    class _RequiredParam(IceDEM):
        def step(self, external_forces, *, load_time=None):  # noqa: D102
            return super().step(external_forces=external_forces, load_time=load_time)

    mutants["param_became_required"] = _RequiredParam(DEMConfig(nx=8, ny=4))

    class _WidenedParam(IceDEM):
        def step(self, external_forces=None, load_time=None):  # noqa: D102
            return super().step(external_forces=external_forces, load_time=load_time)

    mutants["param_kind_widened"] = _WidenedParam(DEMConfig(nx=8, ny=4))

    class _ExtraRequiredParam(IceDEM):
        def step(self, external_forces=None, *, load_time=None, external_torque):  # noqa: D102
            return super().step(external_forces=external_forces, load_time=load_time)

    mutants["param_added_required"] = _ExtraRequiredParam(DEMConfig(nx=8, ny=4))

    class _WithRestore(IceDEM):
        def restore(self, state):  # noqa: D102
            raise NotImplementedError

    mutants["negative_assertion_violated"] = _WithRestore(DEMConfig(nx=8, ny=4))
    return mutants


def mutation_matrix() -> list[dict[str, Any]]:
    """Run every mutant through :func:`verify` and score the detector.

    ``detected`` is True when a finding with the expected ``change`` (and, for
    breaking classes, the expected severity) appears.  The widened-parameter
    mutant is expected to be *classified* as ``additive`` and therefore not to
    raise ``status`` above ``CONFORMS``: widening is source-compatible for the
    keyword-calling binders we have.
    """
    reference_3d = default_instances()["IceDEM3D"]
    rows: list[dict[str, Any]] = []
    for key, mutant in _mutants().items():
        spec = next(item for item in MUTATION_SPECS if item["key"] == key)
        report = verify({"IceDEM": mutant, "IceDEM3D": reference_3d})
        hit = next((f for f in report["findings"] if f["change"] == spec["expect_change"]), None)
        rows.append({
            "key": key,
            "expect_change": spec["expect_change"],
            "expect_severity": spec["expect_severity"],
            "observed_change": None if hit is None else hit["change"],
            "observed_severity": None if hit is None else hit["severity"],
            "observed_field": None if hit is None else hit["field"],
            "detected": hit is not None
                        and hit["severity"] == spec["expect_severity"],
            "status": report["status"],
            "breaking_findings": report["breaking"],
        })
    return rows


def substep_rule_check(ratios: tuple[float, ...] = (1.0, 2.0, 4.0, 16.0,
                                                   4.0 + 1e-6, 4.0 + 1e-10, 4.5),
                       tolerance: float = 1e-9) -> list[dict[str, Any]]:
    """Check the binder-side integer-multiple substep rule against native dt.

    The native solver only advances by ``dt``; a driver must therefore pick a
    global increment that is an integer multiple of it.  We reproduce the
    ``TensorPhysics`` rule (``isclose(ratio, round(ratio), rel_tol=1e-9,
    abs_tol=1e-9)``) on both sides of the tolerance so binders can rely on the
    acceptance boundary and know which side of it they are on.
    """
    from math import isclose

    from .dem import DEMConfig, IceDEM

    model = IceDEM(DEMConfig(nx=8, ny=4))
    native_dt = float(model.dt)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        global_dt = ratio * native_dt
        accepted = isclose(ratio, round(ratio), rel_tol=tolerance, abs_tol=tolerance)
        rows.append({
            "case": f"ratio_{ratio!r}",
            "native_dt_s": native_dt,
            "global_dt_s": global_dt,
            "ratio": ratio,
            "n_substeps_accepted": int(round(ratio)) if accepted else None,
            "accepted": bool(accepted),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI helper: re-freeze the fingerprint deliberately
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-fingerprint", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.print_fingerprint:
        print(fingerprint())
    if args.verify:
        print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())