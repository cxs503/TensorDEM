# Phi(I) Provenance & Dilatancy Audit

**Case**: CASE-DEM-PLANE-SHEAR-MUIDI  
**Sweep points**: 6  
**Full-cell points**: 4  

## Table 1 — phi(I) vs both references

| source | I | phi_meas | phi_ref_3D | err_3D [%] | phi_ref_2D | err_2D [%] |
|---|---|---|---|---|---|---|
| sweep | 0.001007 | 0.5218 | 0.6398 | -18.4 | 0.8398 | -37.9 |
| sweep | 0.003092 | 0.5218 | 0.6394 | -18.4 | 0.8395 | -37.8 |
| sweep | 0.01085 | 0.5218 | 0.6379 | -18.2 | 0.8383 | -37.7 |
| sweep | 0.03455 | 0.5218 | 0.6338 | -17.7 | 0.8348 | -37.5 |
| sweep | 0.1179 | 0.5218 | 0.6231 | -16.3 | 0.8259 | -36.8 |
| sweep | 0.3283 | 0.5218 | 0.6086 | -14.3 | 0.8139 | -35.9 |
| full_cell | 0.01 | 0.6348 | 0.6381 | -0.5 | 0.8384 | -24.3 |
| full_cell | 0.03 | 0.6590 | 0.6345 | +3.8 | 0.8355 | -21.1 |
| full_cell | 0.1 | 0.6942 | 0.6250 | +11.1 | 0.8275 | -16.1 |
| full_cell | 0.3 | 0.7224 | 0.6100 | +18.4 | 0.8150 | -11.4 |

## Table 2 — mu(I) for context

| source | I | mu_meas | mu_ref | err [%] |
|---|---|---|---|---|
| sweep | 0.001007 | 0.3295 | 0.3616 | -8.9 |
| sweep | 0.003092 | 0.3289 | 0.3649 | -9.9 |
| sweep | 0.01085 | 0.3850 | 0.3768 | +2.2 |
| sweep | 0.03455 | 0.4074 | 0.4096 | -0.5 |
| sweep | 0.1179 | 0.4302 | 0.4954 | -13.2 |
| sweep | 0.3283 | 0.4916 | 0.6108 | -19.5 |
| full_cell | 0.01 | 0.3751 | 0.3755 | -0.1 |
| full_cell | 0.03 | 0.4090 | 0.4036 | +1.3 |
| full_cell | 0.1 | 0.3772 | 0.4800 | -21.4 |
| full_cell | 0.3 | 0.4333 | 0.6000 | -27.8 |

## Dilatancy verdict

- **Sweep data**: FAIL — phi does not decrease with I
- **Full-cell data**: FAIL — phi_slab *increases* with I (servo compaction artifact)

## Provenance verdict

- **phi_3d_magnitudes**: PROVENANCE-UNVERIFIED (no in-repo literature anchor; plausible for 3D glass beads)
- **phi_2d_magnitudes**: LITERATURE-ANCHORED (da Cruz 2005, Fig. 3, 2D disks: 0.84->0.79)
- **sweep_dilatancy_sign**: FAIL
- **full_cell_dilatancy_sign**: FAIL (phi_slab increases with I — servo compaction artifact)
- **phi_acceptance_vs_3d**: WITHHELD (reference magnitudes unverified; measured phi systematically 10-15% low in sweep, but full_cell phi_slab shows wrong trend)
- **phi_acceptance_vs_2d**: FAIL (measured phi ~0.52-0.72 vs 2D ref ~0.79-0.84; different dimensionality)
- **recommendation**: The phi(I) ledger criterion should be re-scoped: (a) anchor the 3D magnitudes to a specific 3D experimental source (e.g., Boyer et al. 2011 PRL 107:198301 for 3D suspensions, or Chialvo et al. 2012 for 3D dry grains); (b) the full-cell phi_slab estimator shows a compaction artifact (phi increasing with I) that must be diagnosed — likely the servo-controlled top wall pushes particles down at high shear rate.
