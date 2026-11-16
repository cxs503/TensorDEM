# R15-c predeclaration (m23285) — decision, alternatives, thresholds

Decision: which boundary-layer model is stated as the explanation of the
archived definitional gap (phi_centres / phi_bed), and does the audit artifact
carry a self-check that can be false?

Alternatives
  H1 (incumbent, R15): gap = plate_z/(plate_z-2r)                 [z-wall shell only]
  H2 (this round):     gap = plate_z/(plate_z-2r) * D/(D-2r)      [both walled pairs;
                                                                    x is periodic]
  H0 (no model):       gap = 1 (definitions differ only by noise)

Outcome metric: rel. error of the predicted gap vs the measured
phi_centres/phi_bed on the three analytic packings in phi_estimator_audit
(gaps 0.12 / 0.06 / 0.045 m, N = 2860 / 1430 / 1073, uniform random centres with
both wall pairs shell-depleted, true container-averaged phi = 0.5202 known by
construction).

Predeclared acceptance (before re-running the fixed code):
  * H2 accepted iff |pred/meas - 1| < 5 % on all three packings;
  * H1 must be reported as insufficient iff it exceeds 5 % on any packing;
  * the audit artifact must expose `ok` and the failing criterion, and the
    regression test must fail if `ok` goes false again.
5 % is chosen a priori from counting statistics: the slab holds ~10^2-10^3
centres, so 1/sqrt(N_slab) ~ 3-10 % on the count; 5 % is one sigma of the
coarser packing and an order of magnitude below the z-only residual.
Disclosure: the residuals were already computable from the committed R15
artifacts (this round audits committed evidence, nothing is re-simulated), so
the threshold is not blind to the data. It is stated with that caveat and was
not moved after the numbers were produced: it sits above the observed maximum
(1.3 %) and below the incumbent's minimum (9.7 %).

Device (live) arm, separate and weaker claim, predeclared as an interval:
  z-only factor bounds the measured gap from above on the device
  (measured / z-only in [0.85, 1.00]) because the grafted packing sits denser
  against the y-walls than a geometrically depleted shell argument allows;
  no two-wall claim is made for the device.
