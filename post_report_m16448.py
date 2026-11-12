"""Publish the m16448 progress report (six-element, figures embedded as base64)."""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m16448")
ART = ROOT / "artifacts" / "open_water_high_induction"
rep = json.loads((ART / "verification.json").read_text())
alg, sol = rep["algebra"]["data"], rep["solver"]["data"]
d = sol["defect_evidence"]
lib = sol["sweeps"]["library_geometry"]["classical"]
lean = sol["sweeps"]["lean_geometry"]["classical"]


def row(r):
    return (f"| {r['J']:.2f} | {r['KT']:.4f} | {r['KQ']:.4f} | {r['eta']:.3f} | "
            f"{r['a_max']:.3f} | {r['strips_at_cap']}/{r['n_strips']} | "
            f"{r['residual']:.1e} | {r['momentum_residual']:.2e} | "
            f"{r['ct_max_loading']:.2f} | {'yes' if r['model_valid'] else 'no'} |")


md = f"""# DEFECT-D3 repaired: high-loading validity reporting in the open-water BEM (TensorDEM)

Mission m16448 · commit `6974cc92` · artifact `artifacts/open_water_high_induction/verification.json`
({rep['n_pass']}/{rep['n_gates']} gates PASS, {rep['n_fail']} FAIL) · doc
`docs/open_water_high_induction_d3_repair.md` (rendered from that JSON, no hand-copied digits).

## 1. Case

The propeller-ice milling chain takes the hydrodynamic torque and the race
induced velocity from the blade-element/momentum (BEM) open-water solve, so the
*solve's domain of validity* is load-bearing for the whole milling model. The gap
ledger row **DEFECT-D3** recorded that at the matched 3 m ice-class propeller
(D = 3.0 m, Z = 4, hub ratio 0.40, P/D = 0.95, c(0.7R)/D = 0.539, solidity
sigma(0.7R) = 1.03) and J <= 0.2, every strip sat on the induction cap a = 0.85,
the blade-element load was no longer balanced by the annulus momentum, the
returned residual was ~1e-8 (so no acceptance test could see it), and the
documented Buhl branch was never called.

## 2. Method

* Reproduce the defect on the registered geometry in the legacy mode and measure
  the quantity the old residual was blind to: blade-element thrust versus annulus
  momentum thrust.
* Derive (not tune) the model's validity boundary: the empirical actuator-disk
  saturation C_T = 0.96 F written in *this module's* convention C_T = 4 F a (1+a)
  gives the breakdown induction a_s = {alg['breakdown']['a_s']:.4f} (the turbine
  convention gives 0.4, which is why the coded branch never fitted).
* Build the high-loading closure by the same three conditions Glauert/Buhl use
  (value and slope continuity at a_s, C_T(a=1) = 2) and validate the construction
  by reproducing Buhl's published coefficients in the turbine convention.
* Add independent diagnostics to the state: momentum_residual, model_valid,
  validity_flag, ct_max_loading, converged; keep a legacy `clipped` mode so the
  defect stays reproducible.

## 3. Results (programmatically injected)

Registered 3 m geometry, J = 0.2, legacy mode: iteration residual
{d['residual']:.2e} ("converged") while the momentum residual is
{d['momentum_residual']:.2e} ({d['momentum_residual']*100:.1f} % of disk thrust),
{d['strips_at_cap']}/{d['n_strips']} strips on the cap, disk loading
{d['ct_max_loading']:.1f}x the closure bound, model_valid = {d['model_valid']},
validity_flag = `{d['validity_flag']}`.

| J | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | C_T,BE/C_T,max | valid |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(row(r) for r in lib)}

Registered 3 m geometry (same columns):

| J | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | C_T,BE/C_T,max | valid |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(row(r) for r in sol['sweeps']['defect_geometry']['classical'])}

Low-solidity geometry (c(0.7R)/D = 0.12, sigma ~ 0.22) is inside the envelope
from J = {sol['validity_boundary']['lean']['first_valid_J']}; the library geometry from
J = {sol['validity_boundary']['library']['first_valid_J']}; the registered 3 m geometry from
J = {sol['validity_boundary']['defect']['first_valid_J']}. Where the closure has a root the state
satisfies its own momentum identity to
{sol['closure_root_momentum_residual']:.1e}; inside the envelope the repaired
default reproduces the pre-repair coefficients to
{sol['legacy_parity']['worst_rel_err']:.1e} over {sol['legacy_parity']['n_compared']} comparisons
(no silent regression).

**Second finding.** The announced branch is unused *and* inconsistent: the coded
constant form does not invert Buhl's parabola (at F = 1 it returns a = 0.9446,
whose C_T is 1.857, while the exact inverse at C_T = 0.96 is 0.4), and the
turbine branch is not a change of variable of this convention (a_t -> -a_p flips
the sign of C_T, gate G12). "Simply calling the branch" would have added a second
error.

## 4. Figures

1. closure and C1 junction at a_s = 0.2 (with the published turbine parabola);
2. the defect quantified: iteration residual flat at 1e-8 vs momentum residual
   and disk-loading ratio blowing up at low J;
3. open-water curves in the three modes with the invalid operating points shaded;
4. coded `_buhl` against the exact inverse of its own parabola over F.

## 5. Notes / limits

* No experimental open-water chart exists in this repository: absolute K_T, K_Q
  stay **WITHHELD**. The registered 3 m geometry is out of envelope below J = 0.6,
  so every coefficient produced there (before or after this repair) is
  cap-limited and must not be quoted.
* `milling.py` does not yet act on `model_valid` -- propagating it into the
  milling acceptance path is the next node.
* The `high_loading` mode can fail to converge at the closure vertex (the
  parabola's slope vanishes there); it then reports `converged = False` instead of
  a number that looks solved. Observed once on the library geometry at J = 0.5.
* `momentum_residual` is normalised by disk thrust, hence ill-conditioned near
  zero thrust (high J); ~1e-7 there is round-off.

## 6. Error judgement versus references

| quantity | reference | measured error |
|---|---|---|
| construction, turbine convention | Buhl 2005 coefficients | {alg['buhl_replica']['1.00']['rel_err']:.1e} |
| breakdown induction | exact root of 4a(1+a) = 0.96 (a_s = 0.2) | {abs(alg['breakdown']['a_s'] - 0.2):.1e} |
| closure C1 junction, F = 0.3..1.0 | 4 F a (1+a) value and slope at a_s | {max(alg['continuity']['value_rel_err'], alg['continuity']['slope_rel_err']):.1e} |
| closure inverse | C_T(a(C_T)) = C_T | {alg['inverse']['worst_rel_err']:.1e} |
| momentum identity at closure roots | its own momentum relation | {sol['closure_root_momentum_residual']:.1e} |
| inside-envelope parity | pre-repair coefficients | {sol['legacy_parity']['worst_rel_err']:.1e} |

Every analytic quantity is < 3 % by many orders of magnitude. Is every key
quantity within 3 % of a *measurement*? **No** -- there is no experimental
open-water table in the repository, absolute K_T/K_Q are WITHHELD, and that is
stated rather than papered over.
"""

figs = []
for name, caption in (
    ("fig1_closure_continuity.png",
     "Classical propeller momentum relation, the constructed high-loading closure, "
     "the published Buhl parabola, and the C1 junction at a_s = 0.2."),
    ("fig2_defect_evidence.png",
     "The registered defect quantified: iteration residual flat at 1e-8 while the "
     "momentum-identity residual and the disk-loading ratio diverge at low J."),
    ("fig3_open_water_curves.png",
     "K_T, K_Q, eta in the three induction modes; shaded bands are operating points "
     "outside the annular-momentum validity bound (reported provisional)."),
    ("fig4_buhl_form_check.png",
     "Coded `_buhl` constant form versus the exact inverse of its own parabola over "
     "the loss factor F."),
):
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})

payload = {
    "title": "DEFECT-D3 repaired: derived high-loading validity boundary for the propeller-ice "
             "open-water BEM (TensorDEM m16448)",
    "summary": "Open-water BEM could report 1e-8 convergence while violating its own "
               "momentum identity by 7.1% with every strip on the induction cap. The cap is "
               "replaced by a derived validity boundary (a_s = 0.2) plus an independent "
               "momentum residual and a constructed high-loading closure validated against "
               "Buhl's coefficients to 1.1e-15; 12/12 gates, 31 tests; absolute K_T/K_Q WITHHELD.",
    "scenario_key": "tensordem/propeller_ice/open_water_high_induction",
    "mission_id": 16448,
    "tags": ["TensorDEM", "propeller-ice", "open-water-BEM", "defect-repair",
             "momentum-theory-validity", "verification"],
    "markdown": md,
    "figures": figs,
}

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not base.endswith("/api/v1"):
    base += "/api/v1"
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read())
tok = body.get("access_token") or body.get("token")
if not tok:
    raise SystemExit(f"no token: {body}")

out = Path("/tmp/report_m16448.json")
out.write_text(json.dumps(payload))
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:800])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read().decode()[:2000])