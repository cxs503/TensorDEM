#!/bin/bash
# Reproducibility command for CASE-DEM-OEDOMETER-STIFF
# Run from the TensorDEM repository root

set -e

echo "=== CASE-DEM-OEDOMETER-STIFF: Reproducibility Script ==="
echo ""

# Set up environment
export PYTHONPATH="src:$PYTHONPATH"

# Step 1: Run the DEM oedometer
echo "[1/4] Running DEM oedometer (zero friction, ~90s)..."
python3 benchmarks/case_dem_oedometer_stiff/run_case.py

# Step 2: Run benchmark comparison
echo ""
echo "[2/4] Running benchmark comparison..."
python3 benchmarks/case_dem_oedometer_stiff/benchmark.py || true

# Step 3: Generate figures
echo ""
echo "[3/4] Generating figures..."
python3 benchmarks/case_dem_oedometer_stiff/make_figures.py

# Step 4: Show results
echo ""
echo "[4/4] Results summary:"
echo "  Artifacts: artifacts/case_dem_oedometer_stiff/"
ls -la artifacts/case_dem_oedometer_stiff/
echo ""
echo "  Figures: artifacts/case_dem_oedometer_stiff/figures/"
ls -la artifacts/case_dem_oedometer_stiff/figures/

echo ""
echo "=== Done ==="
