# DEM3D observation and checkpoint regression

At incoming main 6dd9e7824c4f90a8eebad1c454cc946ef04d5fdd, diagnostics on a 3×3×2 grid stretched by 3% commits 89 bond fractures at time zero. A floating alive mask containing NaN is silently converted to boolean and accepted. Exact old source and reproducer are SHA256 bound; both developer reproduction and root replay agree.

The correction makes diagnostics evaluate stored topology without updating fracture history or last reaction. Explicit forces calls retain their existing fracture behavior. Checkpoint storage dtype is checked before conversion. Fourteen targeted tests pass, including output-frequency independence, full snapshot/reaction purity, and atomic rejection of lossy dtypes. The exact test command/output is archived.

Reproduce old failures from this checkout:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/TensorFEM/.venv/bin/python docs/dem3d-observation-audit/tensordem3d-pre-fix-repro.py --repo .

These tests do not certify physical ice properties or a complete work/dissipation balance. The incoming spatial cell list and restart changes are synchronized, but material calibration, directional lattice bias, prescribed sphere geometry, absence of rotation/friction, GPU neighbor performance and time/space convergence remain open. Checkpoint semantic validation of all failure-history values and clock consistency is not established by the dtype correction.
