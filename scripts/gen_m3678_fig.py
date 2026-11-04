"""Generate the m3678 verification figure (rank-deficiency shear fix)."""
import sys, base64, io, math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensordem.dem3d import DEM3DConfig, IceDEM3D
from scripts.verify_dem3d_energy_ledger_analytic import build_subsystem, base_config, RADIUS

before = 1.0 / 3.0
cfg = base_config(drag=0.0)
coll = build_subsystem(cfg, positions=[[-RADIUS, 0, 0], [RADIUS, 0, 0]],
                       velocities=[[0, 0, 0], [0, 0, 0]], bonds=[(0, 1)],
                       rest_lengths=[2 * RADIUS])
after = float(coll._objective_shear_strain().abs().max())

lattice = IceDEM3D(DEM3DConfig())
ref = lattice.positions.clone()
strain = 0.01
lattice.positions = ref * torch.tensor([1 + strain, 1, 1], dtype=lattice.dtype)
code = float(lattice._objective_shear_strain().mean())
principal = [0.5 * ((1 + strain) ** 2 - 1), 0, 0]
mp = sum(principal) / 3
analytic = math.sqrt((2 / 3) * sum((v - mp) ** 2 for v in principal))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].bar(['before fix\n(spurious 1/3)', 'after fix\n(rank skip)'],
            [before, after], color=['crimson', 'seagreen'])
axes[0].axhline(DEM3DConfig().shear_breaking_strain, ls='--', color='k',
                label='shear limit 0.03')
axes[0].set_ylabel('shear invariant (rigid, undeformed single bond)')
axes[0].set_title('Rank-deficient neighbourhood\n(collinear single bond)')
axes[0].legend()
axes[1].bar(['code', 'analytic'], [code, analytic], color=['steelblue', 'orange'])
axes[1].set_ylabel('shear invariant (uniaxial e=0.01)')
axes[1].set_title('Full cubic lattice (unchanged)\nrel err = %.2e %%'
                  % (100 * abs(code - analytic) / analytic))
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=110)
buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
open('/tmp/fig_m3678.txt', 'w').write(b64)
print('FIG_B64_LEN', len(b64))
print('collinear after =', after, 'lattice rel err % =', 100 * abs(code - analytic) / analytic)
