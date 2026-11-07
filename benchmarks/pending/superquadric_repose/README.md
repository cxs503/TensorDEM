# Superquadric Repose Angle Benchmark

## Physical Background

Discrete Element Method (DEM) simulations of granular materials often use
sphere-clump approximations to represent non-spherical particles. While this
approach is computationally efficient, it introduces geometric and macroscopic
errors that can affect the accuracy of the simulation.

This benchmark quantifies the error introduced by the sphere-clump approximation
for superquadric particles, which generalise spheres and rounded polyhedra.

## Superquadric Geometry

A superquadric is defined by the implicit function:

```
F(x, y, z) = (|x/a|^n2 + |y/b|^n2)^(n1/n2) + |z/c|^n1 - 1
```

where:
- `a, b, c` are half-extents along the x, y, z axes
- `n1, n2` are shape exponents controlling blockiness

Special cases:
- `n1 = n2 = 2, a = b = c = R` → sphere of radius R
- `n1, n2 → ∞` → rectangular block

## Benchmark Parameters

| Parameter | Values |
|-----------|--------|
| Aspect ratio (max/min half-extent) | 1.0, 1.5, 2.0, 3.0, 5.0 |
| Shape exponent (n1 = n2) | 2.0, 4.0, 8.0, 16.0, 32.0, 50.0 |
| Number of clump spheres | 10 |
| Friction coefficient μ | 0.4 |

## Error Metrics

### Geometric Error
- **Surface distance**: Maximum distance from clump surface to true superquadric surface
- **Normal deviation**: Maximum angle between clump normal and true surface normal

### Macroscopic Error
- **Repose angle difference**: |θ_true − θ_clump|
- **Force chain deviation**: Mean deviation of force chain directions

## Disable Ratio Table

The disable ratio table specifies for which (aspect_ratio, shape_exponent) pairs
the sphere-clump approximation is acceptable. The criteria are:

| Criterion | Threshold |
|-----------|-----------|
| Maximum repose angle error | 3° |
| Maximum force chain deviation | 5° |
| Maximum geometric error | 1 mm |

## Results

### Sphere Limit (n1 = n2 = 2)
For spherical particles, the superquadric contact model reduces exactly to the
Hertz contact law. The relative error is < 1% for all tested overlaps.

### Cube Limit (n1, n2 → ∞)
For cubic particles, the contact detection works correctly, but the Hertz
formula doesn't apply (flat surfaces have zero curvature). A penalty contact
model is used instead.

### Disable Ratio Table Summary
- **Acceptable for clump**: 5/30 cases (17%)
- **First unacceptable case**: AR=1.0, n=32 (geometric error > 1 mm)

The sphere-clump approximation is only acceptable for nearly-spherical particles
with low shape exponents (n ≤ 16).

## Files

- `run_benchmark.py`: Main benchmark script
- `artifacts/dem/superquadric_repose_results.json`: Full results
- `artifacts/dem/disable_ratio_table.json`: Disable ratio table for API gate

## Running the Benchmark

```bash
cd /path/to/TensorDEM
PYTHONPATH=src python benchmarks/pending/superquadric_repose/run_benchmark.py
```

## References

- Jaklic, Leonardis, Solina: "Segments: The Shape of Objects" (Springer, 2000)
- Wellmann et al.: "A contact algorithm for rigid particles using superquadrics"
  (Computational Particle Mechanics, 2008)
- Hertz (1882), Mindlin (1949): Classical contact mechanics
