# PR #268 CI Fix Report

## Issue
PR #268 (branch agent/m4021) CI failure: unit-tests=failure

## Root Cause
`pyproject.toml` specified `torch==2.14.1` which does not exist. Latest available: 2.13.0.

## Fix
Changed torch dependency from `torch==2.14.1` to `torch>=2.7.0`.

## Verification
- Package installs successfully
- All imports work: shear_periodic, dem3d_wedge, rudder_ice_load, contact_restitution
- Test suites pass:
  - test_dem3d_shear_periodic_closure.py: 13/13 ✓
  - test_rudder_ice_load.py: 24/24 ✓
  - test_dem3d_contact_invariants.py: 13/13 ✓

## Files Changed
- pyproject.toml: Relaxed torch version constraint (1 line)

## Impact
Minimal - dependency specification only, no code changes.
