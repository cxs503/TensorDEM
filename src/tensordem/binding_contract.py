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
# Live-object gate (used by the solver's own export paths)
# ---------------------------------------------------------------------------
# ``verify`` is a full, review-grade introspection of *both* entries.  The
# solver cannot afford that on every export, so the shipped export paths call
# :func:`assert_entry_conforms`, which checks one entry on one live object and
# caches the verdict per class: the first export pays ~1 ms, every later
# export pays one dict lookup.  A frozen-contract violation therefore raises
# from ``IceDEM.exchange_state`` / ``IceDEM3D.state_dict`` themselves instead
# of surfacing later as a numerically wrong coupled run.
_ENTRY_CACHE: dict[str, dict[str, Any]] = {}


def verify_entry(target: Any, entry_name: str, *,
                 contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Introspect one live object against one frozen entry.

    Returns the same report shape as :func:`verify` restricted to this entry,
    plus ``cache_key`` so callers can tell a cached verdict from a fresh one.
    """
    payload = BINDING_CONTRACT if contract is None else contract
    entries = payload["entries"]
    if entry_name not in entries:
        raise KeyError(f"unknown contract entry {entry_name!r}; "
                       f"frozen entries are {sorted(entries)}")
    entry = entries[entry_name]
    cls = target if inspect.isclass(target) else type(target)

    findings: list[dict[str, Any]] = []
    checked = 0
    matched = 0

    def add(field: str, change: str, severity: str) -> None:
        findings.append({"entry": entry_name, "field": field,
                         "change": change, "severity": severity})

    for attr in entry.get("state", {}):
        checked += 1
        if hasattr(target, attr) or hasattr(cls, attr):
            matched += 1
        else:
            add(f"state.{attr}", "attribute_removed", "breaking")

    for attr in entry.get("properties", {}):
        checked += 1
        if isinstance(inspect.getattr_static(cls, attr, None), property):
            matched += 1
        else:
            add(f"properties.{attr}", "property_removed", "breaking")

    for name in entry.get("methods", {}):
        checked += 1
        if callable(getattr(cls, name, None)):
            matched += 1
        else:
            add(f"methods.{name}", "method_removed", "breaking")

    for attr in entry.get("explicitly_absent", []):
        checked += 1
        if hasattr(cls, attr):
            add(f"explicitly_absent.{attr}", "negative_assertion_violated", "breaking")
        else:
            matched += 1

    breaking = [f for f in findings if f["severity"] == "breaking"]
    return {
        "schema": "tensordem.binding-contract.entry-report/1",
        "contract_schema": payload["schema"],
        "contract_version": payload["contract_version"],
        "contract_fingerprint": fingerprint(payload),
        "frozen_fingerprint_match": fingerprint(payload) == FROZEN_FINGERPRINT,
        "entry": entry_name,
        "target": f"{cls.__module__}.{cls.__qualname__}",
        "checks": checked,
        "matched": matched,
        "findings": findings,
        "breaking": len(breaking),
        "status": "CONFORMS" if not breaking else "DRIFT",
    }


def assert_entry_conforms(target: Any, entry_name: str, *,
                          contract: Mapping[str, Any] | None = None,
                          refresh: bool = False) -> dict[str, Any]:
    """Raise :class:`BindingContractError` if a live object drifted from its entry.

    The verdict is cached per ``(class, entry)``; pass ``refresh=True`` to force
    a fresh introspection (tests that mutate a single instance need this).
    """
    cls = type(target)
    cache_key = f"{cls.__module__}.{cls.__qualname__}|{entry_name}"
    report = None if refresh else _ENTRY_CACHE.get(cache_key)
    cached = report is not None
    if report is None:
        report = verify_entry(target, entry_name, contract=contract)
        # Only a conforming verdict is cached: a failure must re-verify so that
        # a transiently drifted object cannot poison later exports.
        if report["status"] == "CONFORMS":
            _ENTRY_CACHE[cache_key] = report
    report = dict(report, cache_key=cache_key, cached=cached)
    if report["status"] != "CONFORMS":
        raise BindingContractError(report)
    return report


def clear_entry_cache() -> None:
    """Drop the cached per-class verdicts (tests and hot-reload paths)."""
    _ENTRY_CACHE.clear()


# ---------------------------------------------------------------------------
# Numeric benchmark: the frozen surface vs analytic references
# ---------------------------------------------------------------------------
# The contract promises *semantics*; these rows turn each promise into a number
# with an analytic reference so that "the binding is frozen" is not a claim
# about prose.  Every row carries ``ref``, ``got`` and ``rel_error``; the owner
# acceptance line (15%) is applied to every row, and the tightest rows also
# record their own pre-registered tolerance.
BENCHMARK_ACCEPT_REL_ERROR = 0.15


def _rel(value: Any, reference: Any) -> float:
    """Relative error ``|value - reference| / |reference|`` (0 when both are 0)."""
    got = float(value)
    ref = float(reference)
    if ref == 0.0:
        return 0.0 if got == 0.0 else float("inf")
    return abs(got - ref) / abs(ref)


def _free_flight_case(dt_factor: float, total_time: float, load: tuple[float, float]) -> dict[str, Any]:
    """Uniform load on an unconstrained lattice: no internal force, pure kinematics.

    Semi-implicit Euler gives the closed form ``x_n - x_0 = a*dt^2*n(n+1)/2``
    for a constant acceleration ``a = F/m``; the continuum answer is
    ``0.5*a*(n*dt)^2``.  The first is exact for the integrator, so its error
    measures binding bookkeeping (dt, clock, per-step load consumption); the
    difference to the second is the discretisation error and must converge at
    first order.
    """
    import torch

    from .dem import DEMConfig, IceDEM

    cfg = DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                    contact_damping=0.0, drag=0.0)
    dt = cfg.recommended_dt * dt_factor
    cfg = DEMConfig(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                    contact_damping=0.0, drag=0.0, dt=dt)
    model = IceDEM(cfg)
    n_steps = max(1, int(round(total_time / dt)))
    mass = model.config.mass
    force = torch.zeros_like(model.positions)
    force[:, 0] = load[0]
    force[:, 1] = load[1]
    x0 = model.positions.clone()
    for _ in range(n_steps):
        model.step(external_forces=force, load_time=model.time)
    dx = (model.positions - x0).mean(dim=0)
    accel = (load[0] / mass, load[1] / mass)
    got = (float(dx[0]), float(dx[1]))

    def component(a: float, n: int, d: float, shape: str) -> float:
        if shape == "symplectic":
            return a * d * d * n * (n + 1) / 2.0
        return 0.5 * a * (n * d) ** 2

    sym = tuple(component(accel[i], n_steps, dt, "symplectic") for i in (0, 1))
    cont = tuple(component(accel[i], n_steps, dt, "continuum") for i in (0, 1))
    return {
        "dt_factor": dt_factor,
        "dt_s": float(dt),
        "n_steps": n_steps,
        "total_time_s": float(n_steps * dt),
        "got_displacement_m": list(got),
        "ref_symplectic_m": list(sym),
        "ref_continuum_m": list(cont),
        "rel_error_vs_symplectic": max(_rel(got[i], sym[i]) for i in (0, 1)),
        "rel_error_vs_continuum": max(_rel(got[i], cont[i]) for i in (0, 1)),
    }


def benchmark_physics(n_steps_energy: int = 2000) -> dict[str, Any]:
    """Numeric ref-vs-got benchmark of the frozen binding surface.

    Rows (all relative errors against an analytic or bitwise reference):

    1. impulse-momentum identity (2-D, isolated lattice, n = 16 steps)
    2. free-flight displacement vs the exact semi-implicit-Euler closed form
    3. free-flight displacement vs the continuum answer at dt, dt/2, dt/4
       (with the observed convergence order)
    4. mechanical-energy drift of an isolated conservative lattice
    5. Hertz normal force from the exported kernel vs ``(4/3)E*sqrt(R*)d^1.5``
    6. ``hertz_reduced_modulus`` vs the textbook equal-sphere ``E/(2(1-nu^2))``
    7. native clock ``t = n*dt``
    8. restart round-trip position drift (bitwise reference = 0 m)
    """
    import math

    import torch

    from .dem import DEMConfig, IceDEM
    from .dem3d import (hertz_effective_radius, hertz_normal_force,
                        hertz_reduced_modulus)

    rows: list[dict[str, Any]] = []

    def row(name: str, metric: str, reference: Any, got: Any,
            ref_source: str, tolerance: float = BENCHMARK_ACCEPT_REL_ERROR,
            detail: str = "", error: float | None = None) -> None:
        err = _rel(got, reference) if error is None else float(error)
        rows.append({
            "name": name,
            "metric": metric,
            "ref": reference if not isinstance(reference, tuple) else list(reference),
            "got": got if not isinstance(got, tuple) else list(got),
            "ref_source": ref_source,
            "rel_error": err,
            "tolerance": tolerance,
            "pass": bool(err <= tolerance),
            "detail": detail,
        })

    def isolated(**overrides: Any) -> Any:
        kwargs = dict(nx=6, ny=3, fix_edges=False, tool_gap=10.0,
                      contact_damping=0.0, drag=0.0)
        kwargs.update(overrides)
        return IceDEM(DEMConfig(**kwargs))

    # 1. impulse-momentum ---------------------------------------------------
    model = isolated()
    load = torch.zeros_like(model.positions)
    load[:, 0] = 0.25
    load[:, 1] = -0.1
    v0 = model.velocities.clone()
    n_impulse = 16
    for _ in range(n_impulse):
        model.step(external_forces=load, load_time=model.time)
    dp = model.config.mass * (model.velocities - v0).sum(dim=0)
    ref_impulse = n_impulse * model.dt * load.sum(dim=0)
    row("impulse_momentum_2d", "max component relative error",
        float(ref_impulse.abs().max()), float(dp.abs().max()),
        "analytic n*dt*F (bond/contact forces internal)", 1e-12,
        f"n_steps={n_impulse}",
        error=float((dp - ref_impulse).abs().max() / ref_impulse.abs().max()))

    # 2-3. free flight vs exact integrator and continuum ---------------------
    total_time = 0.01
    flight_dt = _free_flight_case(1.0, total_time, (0.25, -0.1))
    flight_half = _free_flight_case(0.5, total_time, (0.25, -0.1))
    flight_quarter = _free_flight_case(0.25, total_time, (0.25, -0.1))
    row("free_flight_vs_symplectic", "max component relative error",
        max(flight_dt["ref_symplectic_m"]), max(flight_dt["got_displacement_m"]),
        "semi-implicit Euler closed form a*dt^2*n(n+1)/2", 1e-12,
        f"t={total_time} s, n={flight_dt['n_steps']}")
    for case in (flight_dt, flight_half, flight_quarter):
        row(f"free_flight_vs_continuum_dt_x{case['dt_factor']:g}",
            "max component relative error",
            max(case["ref_continuum_m"]), max(case["got_displacement_m"]),
            "continuum 0.5*a*t^2", BENCHMARK_ACCEPT_REL_ERROR,
            f"dt={case['dt_s']:.6e} s, n={case['n_steps']}")
    orders = []
    for coarse, fine in ((flight_dt, flight_half), (flight_half, flight_quarter)):
        ratio = coarse["rel_error_vs_continuum"] / fine["rel_error_vs_continuum"]
        orders.append(math.log(ratio) / math.log(2.0))

    # 4. energy drift of a conservative lattice -----------------------------
    model = isolated()
    n = len(model.positions)
    model.velocities[:, 0] = 0.05 * torch.sin(
        torch.arange(n, dtype=torch.float64) * (2 * math.pi / model.config.nx))
    e0 = float(model.mechanical_energy()["mechanical_J"])
    worst = 0.0
    for _ in range(n_steps_energy):
        model.step()
        worst = max(worst, abs(float(model.mechanical_energy()["mechanical_J"]) - e0))
    row("energy_drift_conservative", "max |E(t)-E(0)| / E(0)",
        0.0, worst / e0, "E(0) = analytic reference; zero drift is the target",
        BENCHMARK_ACCEPT_REL_ERROR,
        f"steps={n_steps_energy}, E(0)={e0:.6e} J",
        error=worst / e0)

    # 5-6. Hertz kernel and reduced modulus ---------------------------------
    young, nu, radius, overlap = 9.0e9, 0.3, 0.025, 1.0e-4
    e_star = hertz_reduced_modulus(young, nu)
    r_eff = hertz_effective_radius(radius)
    ref_e_star = young / (2.0 * (1.0 - nu * nu))
    ref_r_eff = radius / 2.0
    ref_force = (4.0 / 3.0) * ref_e_star * math.sqrt(ref_r_eff) * overlap ** 1.5
    got_force = float(hertz_normal_force(overlap, e_star, r_eff))
    row("hertz_reduced_modulus", "relative error", ref_e_star, e_star,
        "E* = E/(2(1-nu^2)) for equal elastic spheres", 1e-12)
    row("hertz_effective_radius", "relative error", ref_r_eff, r_eff,
        "R* = R/2 for equal spheres", 1e-12)
    row("hertz_normal_force", "relative error", ref_force, got_force,
        "(4/3)E*sqrt(R*)*delta^1.5 (Hertz 1882)", 1e-12,
        f"delta={overlap} m, E={young} Pa, nu={nu}")

    # 7. native clock -------------------------------------------------------
    model = isolated()
    steps = 37
    for _ in range(steps):
        model.step()
    row("clock_n_dt", "relative error", steps * model.dt, model.time,
        "analytic n*dt", 1e-12, f"n={steps}")

    # 8. restart round-trip -------------------------------------------------
    model = isolated()
    for _ in range(10):
        model.step()
    snap = model.snapshot()
    restored = IceDEM.from_snapshot(snap)
    drift = float((restored.positions - model.positions).abs().max())
    row("restart_roundtrip_drift", "max |dposition| (m)", 0.0, drift,
        "bitwise identity of the restart payload", 0.0)

    passed = sum(1 for r in rows if r["pass"])
    return {
        "schema": "tensordem.binding-contract.benchmark/1",
        "accept_rel_error": BENCHMARK_ACCEPT_REL_ERROR,
        "rows": rows,
        "rows_total": len(rows),
        "rows_passed": passed,
        "worst_rel_error": max(r["rel_error"] for r in rows),
        "free_flight_convergence_order": orders,
        "verdict": "MET" if passed == len(rows) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Numeric benchmark: the 3-D frozen surface vs analytic references
# ---------------------------------------------------------------------------
# ``IceDEM3D`` is the entry the 3-D binders consume; freezing its *names* is not
# enough, the numbers have to be anchored too.  Each row below pairs an analytic
# (or bitwise) reference with the value the frozen surface actually produces, so
# that a silently broken 3-D binding shows up as an error, not as a green
# introspection report.
BENCHMARK_3D_ANCHORS: dict[str, str] = {
    "impulse_momentum": "m*sum(dv) = n*dt*sum(F_ext) for a momentum-neutral "
                        "internal force field (Newton's second law; exact for the "
                        "semi-implicit Euler / velocity-Verlet step)",
    "symplectic_euler": "semi-implicit Euler closed form x_n - x_0 = a*dt^2*n(n+1)/2; "
                        "cf. Hairer, Lubich & Wanner, Geometric Numerical Integration "
                        "(Springer, 2006), Sec. I.1",
    "continuum": "rigid-body limit 0.5*a*t^2 (first-order consistent in dt)",
    "hertz_contact": "Hertz, H. (1882), J. reine angew. Math. 92:156-171: "
                     "F = (4/3)E*sqrt(R*)*delta^1.5, E* = E/(2(1-nu^2)), R* = R/2 "
                     "for two equal elastic spheres",
    "energy": "isolated conservative lattice: mechanical energy is a constant of "
              "motion, so E(t) - E(0) = 0 is the analytic reference",
    "restart": "restart payload is exact: load_state_dict(state_dict()) must "
               "reproduce the trajectory bitwise",
}


def isolated_3d_model(**overrides: Any):
    """Force-free 3-D lattice used by the benchmark rows and the tests.

    No supports, no contact dashpot, no drag, indenter parked at ``tool_gap``,
    friction off: the only non-momentum-neutral force is the external load the
    caller passes.  Any other force appearing here would break the analytic
    references and is therefore a bug, not a modelling choice.
    """
    from .dem3d import DEM3DConfig, IceDEM3D

    params: dict[str, Any] = dict(
        nx=5, ny=4, nz=3, fix_bottom=False, fix_x_edges=False,
        contact_damping=0.0, drag=0.0, tool_gap=10.0,
        friction_coefficient=0.0, rolling_resistance_coefficient=0.0,
    )
    params.update(overrides)
    return IceDEM3D(DEM3DConfig(**params))


def free_flight_case_3d(dt_factor: float = 1.0, n_steps: int = 200,
                        load: tuple[float, float, float] = (0.25, -0.05, -0.1),
                        **overrides: Any) -> dict[str, Any]:
    """Uniform load on an unconstrained 3-D lattice -> pure kinematics.

    The load is uniform, so every particle gets the same acceleration and the
    bonds stay at their rest length: the internal force stays exactly zero and
    the displacement is the closed-form answer for the integrator.  Comparing
    against the semi-implicit Euler closed form measures the *binding*
    bookkeeping (dt, clock, per-step load consumption); comparing against the
    continuum answer ``0.5*a*t^2`` measures the discretisation error, which must
    shrink at first order as ``dt`` is halved.
    """
    import torch

    from .dem3d import DEM3DConfig, IceDEM3D

    params: dict[str, Any] = dict(
        nx=5, ny=4, nz=3, fix_bottom=False, fix_x_edges=False,
        contact_damping=0.0, drag=0.0, tool_gap=10.0,
        friction_coefficient=0.0, rolling_resistance_coefficient=0.0,
    )
    params.update(overrides)
    base_dt = DEM3DConfig(**params).recommended_dt
    params["dt"] = base_dt * dt_factor
    model = IceDEM3D(DEM3DConfig(**params))
    dt = model.dt
    mass = model.config.mass
    force = torch.zeros_like(model.positions, dtype=model.positions.dtype)
    for axis, value in enumerate(load):
        force[:, axis] = value
    x0 = model.positions.clone()
    for _ in range(int(n_steps)):
        model.step(external_forces=force)
    dx = (model.positions - x0).mean(dim=0)

    accel = tuple(load[i] / mass for i in range(3))
    got = tuple(float(dx[i]) for i in range(3))
    ref_sym = tuple(accel[i] * dt * dt * n_steps * (n_steps + 1) / 2.0 for i in range(3))
    ref_cont = tuple(0.5 * accel[i] * (n_steps * dt) ** 2 for i in range(3))

    def worst(values: tuple[float, ...], reference: tuple[float, ...]) -> float:
        errs = [_rel(values[i], reference[i]) for i in range(3) if reference[i] != 0.0]
        return max(errs) if errs else 0.0

    return {
        "dt_factor": float(dt_factor),
        "dt_s": float(dt),
        "n_steps": int(n_steps),
        "total_time_s": float(n_steps * dt),
        "mass_kg": float(mass),
        "accel_m_s2": list(accel),
        "got_displacement_m": list(got),
        "ref_symplectic_m": list(ref_sym),
        "ref_continuum_m": list(ref_cont),
        "symplectic_rel_error": worst(got, ref_sym),
        "continuum_rel_error": worst(got, ref_cont),
    }


def benchmark_physics_3d(n_steps_energy: int = 1000) -> dict[str, Any]:
    """Numeric ref-vs-got benchmark of the frozen **3-D** binding surface.

    Rows (``IceDEM3D``; every row carries ``ref``, ``got``, ``rel_error``,
    ``tolerance`` and ``pass``):

    1.  ``impulse_momentum_3d``          m*sum(dv) vs n*dt*sum(F) (tolerance 1e-12)
    2.  ``loads_not_accumulated_3d``     impulse projection ratio vs 1 (1e-12)
    3.  ``free_flight_symplectic_3d``    displacement vs the exact integrator form (1e-11)
    4.  ``free_flight_continuum_3d_dt_x{1,0.5,0.25}`` displacement vs 0.5*a*t^2 (owner 15%)
    5.  ``free_flight_convergence_order_3d`` first-order consistency of row 4
    6.  ``energy_drift_conservative_3d`` |E(t)-E(0)|/E(0) over a bonded lattice (2%)
    7.  ``hertz_{reduced_modulus,effective_radius,normal_force}_3d`` kernel vs Hertz (1e-12)
    8.  ``clock_n_dt_3d``                native clock vs n*dt (1e-12)
    9.  ``restart_roundtrip_3d`` / ``restart_continuation_3d`` bitwise restart (0 m)
    10. ``shape_guard_3d``               (N,2) load is rejected (boolean row)
    """
    import copy
    import math

    import torch

    from .dem3d import (hertz_effective_radius, hertz_normal_force,
                        hertz_reduced_modulus)

    rows: list[dict[str, Any]] = []

    def row(name: str, metric: str, reference: Any, got: Any,
            ref_source: str, tolerance: float = BENCHMARK_ACCEPT_REL_ERROR,
            detail: str = "", error: float | None = None) -> None:
        err = _rel(got, reference) if error is None else float(error)
        rows.append({
            "name": name,
            "metric": metric,
            "ref": list(reference) if isinstance(reference, tuple) else reference,
            "got": list(got) if isinstance(got, tuple) else got,
            "ref_source": ref_source,
            "rel_error": err,
            "tolerance": tolerance,
            "pass": bool(err <= tolerance),
            "detail": detail,
        })

    # 1-2. impulse-momentum identity and non-accumulation -------------------
    n_impulse = 8
    model = isolated_3d_model()
    mass = model.config.mass
    load = torch.zeros_like(model.positions, dtype=model.positions.dtype)
    load[:, 2] = -0.2
    v0 = model.velocities.clone()
    for _ in range(n_impulse):
        model.step(external_forces=load)
    impulse = mass * (model.velocities - v0).sum(dim=0)
    ref_impulse = n_impulse * model.dt * load.sum(dim=0)
    row("impulse_momentum_3d", "max component relative error",
        float(ref_impulse.abs().max()), float(impulse.abs().max()),
        "analytic n*dt*F (bond/contact forces internal, momentum-neutral)",
        1e-12, f"n_steps={n_impulse}",
        error=float((impulse - ref_impulse).abs().max() / ref_impulse.abs().max()))
    ratio = float((impulse @ ref_impulse) / (ref_impulse @ ref_impulse))
    row("loads_not_accumulated_3d", "impulse projection ratio", 1.0, ratio,
        "load applied once per step (accumulation would give 1+2+...+n)",
        1e-12, f"n_steps={n_impulse}")

    # 3-5. free flight vs the exact integrator and the continuum ------------
    total_time = 0.01
    base_dt = isolated_3d_model().dt
    n_base = max(1, int(round(total_time / base_dt)))
    flight_dt = free_flight_case_3d(1.0, n_base)
    flight_half = free_flight_case_3d(0.5, 2 * n_base)
    flight_quarter = free_flight_case_3d(0.25, 4 * n_base)
    row("free_flight_symplectic_3d", "max component relative error",
        max(flight_dt["ref_symplectic_m"]), max(flight_dt["got_displacement_m"]),
        "semi-implicit Euler closed form a*dt^2*n(n+1)/2", 1e-11,
        f"t={total_time} s, n={flight_dt['n_steps']}",
        error=flight_dt["symplectic_rel_error"])
    for case, factor in ((flight_dt, 1.0), (flight_half, 0.5), (flight_quarter, 0.25)):
        row(f"free_flight_continuum_3d_dt_x{factor:g}",
            "max component relative error",
            max(case["ref_continuum_m"]), max(case["got_displacement_m"]),
            "continuum 0.5*a*t^2 (rigid-body limit)",
            BENCHMARK_ACCEPT_REL_ERROR,
            f"dt={case['dt_s']:.6e} s, n={case['n_steps']}",
            error=case["continuum_rel_error"])
    orders = []
    for coarse, fine in ((flight_dt, flight_half), (flight_half, flight_quarter)):
        orders.append(math.log(coarse["continuum_rel_error"]
                               / fine["continuum_rel_error"]) / math.log(2.0))
    row("free_flight_convergence_order_3d", "observed order p in err ~ dt^p",
        1.0, sum(orders) / len(orders),
        "first-order consistency of semi-implicit Euler", 0.05,
        f"orders={[round(o, 6) for o in orders]}",
        error=abs(sum(orders) / len(orders) - 1.0))

    # 6. energy drift of a conservative bonded lattice ----------------------
    lattice = isolated_3d_model()
    n_particles = len(lattice.positions)
    lattice.velocities[:, 0] = 0.05 * torch.sin(
        torch.arange(n_particles, dtype=lattice.positions.dtype)
        * (2 * math.pi / lattice.config.nx))
    e0 = float(lattice.energy()["mechanical_energy_J"])
    worst_drift = 0.0
    for _ in range(max(1, int(n_steps_energy))):
        lattice.step()
        worst_drift = max(worst_drift,
                          abs(float(lattice.energy()["mechanical_energy_J"]) - e0))
    row("energy_drift_conservative_3d", "max |E(t)-E(0)| / E(0)", 0.0,
        worst_drift / e0, "E(0) is the analytic constant of motion", 0.02,
        f"steps={n_steps_energy}, E(0)={e0:.6e} J", error=worst_drift / e0)

    # 7. Hertz kernel and reduced modulus (3-D path) ------------------------
    young, nu, radius, overlap = 9.0e9, 0.3, 0.025, 1.0e-4
    e_star = hertz_reduced_modulus(young, nu)
    r_eff = hertz_effective_radius(radius)
    ref_e_star = young / (2.0 * (1.0 - nu * nu))
    ref_r_eff = radius / 2.0
    ref_force = (4.0 / 3.0) * ref_e_star * math.sqrt(ref_r_eff) * overlap ** 1.5
    got_force = float(hertz_normal_force(overlap, e_star, r_eff))
    row("hertz_reduced_modulus_3d", "relative error", ref_e_star, e_star,
        "E* = E/(2(1-nu^2)) for equal elastic spheres", 1e-12)
    row("hertz_effective_radius_3d", "relative error", ref_r_eff, r_eff,
        "R* = R/2 for equal spheres", 1e-12)
    row("hertz_normal_force_3d", "relative error", ref_force, got_force,
        "(4/3)E*sqrt(R*)*delta^1.5 (Hertz 1882)", 1e-12,
        f"delta={overlap} m, E={young} Pa, nu={nu}")

    # 8. native clock -------------------------------------------------------
    clock = isolated_3d_model()
    steps = 37
    for _ in range(steps):
        clock.step()
    row("clock_n_dt_3d", "relative error", steps * clock.dt, clock.time,
        "analytic n*dt", 1e-12, f"n={steps}")

    # 9. restart round-trip (state_dict / load_state_dict, the 3-D names) ---
    source = isolated_3d_model()
    load3 = torch.zeros_like(source.positions, dtype=source.positions.dtype)
    load3[:, 1] = 0.05
    for _ in range(7):
        source.step(external_forces=load3)
    payload = copy.deepcopy(source.state_dict())
    restored = isolated_3d_model()
    restored.load_state_dict(payload)
    restore_drift = float((restored.positions - source.positions).abs().max())
    row("restart_roundtrip_3d", "max |dposition| (m)", 0.0, restore_drift,
        "bitwise identity of state_dict -> load_state_dict", 0.0,
        f"steps=7, time={source.time:.6e} s")
    source.step(external_forces=load3)
    restored.step(external_forces=load3)
    continue_drift = float((restored.positions - source.positions).abs().max())
    row("restart_continuation_3d", "max |dposition| after one more step (m)",
        0.0, continue_drift, "restored solver continues the same trajectory", 0.0,
        "post-restart step with the same load")

    # 10. shape guard -------------------------------------------------------
    guard = isolated_3d_model()
    rejected = 0.0
    try:
        guard.step(external_forces=torch.zeros((len(guard.positions), 2),
                                               dtype=guard.positions.dtype))
    except ValueError:
        rejected = 1.0
    row("shape_guard_3d", "(N,2) load rejected", 1.0, rejected,
        "contract freezes a (N,3) load surface; a (N,2) load must raise", 0.0,
        "ValueError expected")

    passed = sum(1 for r in rows if r["pass"])
    return {
        "schema": "tensordem.binding-contract.benchmark-3d/1",
        "entry": "IceDEM3D",
        "accept_rel_error": BENCHMARK_ACCEPT_REL_ERROR,
        "rows": rows,
        "rows_total": len(rows),
        "rows_passed": passed,
        "worst_rel_error": max(r["rel_error"] for r in rows),
        "free_flight_convergence_order": orders,
        "anchors": BENCHMARK_3D_ANCHORS,
        "environment": {
            "dt_s": float(base_dt),
            "n_steps_energy": int(n_steps_energy),
            "particle_count": int(n_particles),
        },
        "verdict": "MET" if passed == len(rows) else "FAIL",
    }


# ---------------------------------------------------------------------------
# CLI helper: re-freeze the fingerprint deliberately
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-fingerprint", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--probes", action="store_true",
                        help="run the behavioural probes (requires torch)")
    parser.add_argument("--benchmark", action="store_true",
                        help="run the numeric ref-vs-got benchmark (requires torch)")
    parser.add_argument("--out", type=str, default=None,
                        help="write the chosen reports as JSON to this directory")
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    if args.print_fingerprint:
        print(fingerprint())
    if args.verify:
        payload["verify"] = verify()
        print(json.dumps(payload["verify"], indent=2, sort_keys=True))
    if args.probes:
        payload["probes"] = probe_semantics()
        print(json.dumps(payload["probes"], indent=2, sort_keys=True))
    if args.benchmark:
        payload["benchmark"] = benchmark_physics()
        print(json.dumps(payload["benchmark"], indent=2, sort_keys=True))
    if args.out:
        from pathlib import Path as _Path
        outdir = _Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "binding_contract_cli.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())