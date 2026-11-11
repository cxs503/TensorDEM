# GAP-DEM-GPU-100K Throughput Benchmark Report

## Protocol

- **Benchmark**: GAP-DEM-GPU-100K
- **Protocol**: tensordem-dem3d-gpu-throughput-v1
- **Acceptance threshold**: single step for 100K particles ≤ 2.0 s

## Hardware

### CPU
- device: cpu
- cuda_available: False
- cpu_threads: 128
- cpu_interop_threads: 128

## Results

| Scale | Particles | Device | Time/step (s) | Throughput (N·steps/s) | Pairs | Status |
|-------|-----------|--------|---------------|------------------------|-------|--------|
| 10k | 10,648 | cpu | 0.4121 | 25,841 | 125,748 |  |
| 50k | 50,653 | cpu | 1.9176 | 26,414 | 622,188 |  |
| 100k | 103,823 | cpu | 3.9736 | 26,128 | 1,290,898 | FAIL |

## Acceptance Summary

- **100K particles on cpu**: 3.9736 s/step → ❌ FAIL

## Notes

- Warmup steps are excluded from timing to allow JIT compilation / cache warming.
- GPU runs use `torch.cuda.synchronize()` before and after timing to ensure accurate wall-clock measurement.
- Bond stiffness set to 0 for pure contact-throughput measurement (bonds disabled).
- Physical parameters match the `scale_cases.json` benchmark protocol for comparability.
