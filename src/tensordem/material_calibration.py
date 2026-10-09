"""Independent infinitesimal square-lattice material qualification in SI.

Uses precisely the axial/diagonal central bond topology of IceDEM, without
contact, damage, damping or time integration. This is a material patch test,
not a simulation of fracture or a new calibrated DEM constitutive law.
"""
import math
import torch


def lattice(cells_y, *, length=0.8, height=0.4, thickness=0.2, density=917.0):
    if isinstance(cells_y, bool) or not isinstance(cells_y, int) or cells_y < 2:
        raise ValueError('cells_y must be an integer >= 2')
    for value in (length, height, thickness, density):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('geometry and density must be finite positive SI values')
    h = height / cells_y
    cells_x = round(length / h)
    if not math.isclose(cells_x * h, length, rel_tol=1e-12):
        raise ValueError('length/height must give a square lattice')
    yy, xx = torch.meshgrid(torch.arange(cells_y + 1, dtype=torch.float64),
                           torch.arange(cells_x + 1, dtype=torch.float64), indexing='ij')
    positions = torch.stack((xx.flatten() * h, yy.flatten() * h - height / 2), 1)
    indices = torch.arange(len(positions)).reshape(cells_y + 1, cells_x + 1)
    bonds = torch.cat([torch.stack((indices[:, :-1].flatten(), indices[:, 1:].flatten()), 1),
                       torch.stack((indices[:-1, :].flatten(), indices[1:, :].flatten()), 1),
                       torch.stack((indices[:-1, :-1].flatten(), indices[1:, 1:].flatten()), 1),
                       torch.stack((indices[:-1, 1:].flatten(), indices[1:, :-1].flatten()), 1)])
    weights = torch.ones_like(xx)
    weights[:, [0, -1]] *= 0.5
    weights[[0, -1], :] *= 0.5
    masses = (weights.flatten() * density * thickness * h * h)
    return positions, bonds, masses


def energy(positions, bonds, displacement, stiffness):
    """Linearized axial spring energy, not finite strain spring energy."""
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError('stiffness must be finite and positive')
    vector = positions[bonds[:, 1]] - positions[bonds[:, 0]]
    normal = vector / torch.linalg.vector_norm(vector, dim=1)[:, None]
    extension = ((displacement[bonds[:, 1]] - displacement[bonds[:, 0]]) * normal).sum(1)
    return float(0.5 * stiffness * extension.square().sum())


def qualify(cells_y, *, young_modulus=1e6, poisson_ratio=0.3, length=0.8,
            height=0.4, thickness=0.2, density=917.0):
    if not math.isfinite(young_modulus) or young_modulus <= 0 or not math.isfinite(poisson_ratio) or not -1 < poisson_ratio < 0.5:
        raise ValueError('invalid isotropic plane stress target')
    x, bonds, masses = lattice(cells_y, length=length, height=height, thickness=thickness, density=density)
    volume = length * height * thickness
    eps = 1e-4
    ux = torch.stack((eps * x[:, 0], torch.zeros(len(x))), 1)
    uy = torch.stack((torch.zeros(len(x)), eps * x[:, 1]), 1)
    shear = torch.stack((eps * x[:, 1], torch.zeros(len(x))), 1)
    c11_unit = 2 * energy(x, bonds, ux, 1.0) / (volume * eps**2)
    target_c11 = young_modulus / (1 - poisson_ratio**2)
    stiffness = target_c11 / c11_unit
    ex = energy(x, bonds, ux, stiffness)
    ey = energy(x, bonds, uy, stiffness)
    c11 = 2 * ex / (volume * eps**2)
    c22 = 2 * ey / (volume * eps**2)
    c12 = (energy(x, bonds, ux + uy, stiffness) - ex - ey) / (volume * eps**2)
    g = 2 * energy(x, bonds, shear, stiffness) / (volume * eps**2)
    target_c12 = poisson_ratio * target_c11
    target_g = young_modulus / (2 * (1 + poisson_ratio))
    curvature = 1e-3
    bending = torch.stack((-curvature * x[:, 0] * x[:, 1], 0.5 * curvature * x[:, 0]**2), 1)
    bending_actual = energy(x, bonds, bending, stiffness)
    bending_target = 0.5 * target_c11 * curvature**2 * length * thickness * height**3 / 12
    errors = dict(C11=abs(c11 / target_c11 - 1), C22=abs(c22 / target_c11 - 1),
                  C12=abs(c12 / target_c12 - 1) if target_c12 else abs(c12) / target_c11,
                  shear=abs(g / target_g - 1), bending=abs(bending_actual / bending_target - 1))
    return dict(cells_y=cells_y, particle_count=len(x), bond_count=len(bonds),
                geometry=dict(length_m=length, height_m=height, thickness_m=thickness),
                mass_kg=float(masses.sum()), mass_convention='boundary-weighted continuum control volumes; differs from IceDEM disk masses',
                target=dict(E_Pa=young_modulus, nu=poisson_ratio, C11_Pa=target_c11, C12_Pa=target_c12, G_Pa=target_g),
                stiffness_N_per_m=stiffness, measured=dict(C11_Pa=c11, C22_Pa=c22, C12_Pa=c12, G_Pa=g,
                bending_energy_J=bending_actual, target_bending_energy_J=bending_target),
                relative_errors=errors, tolerance=0.03, material_qualified=all(e <= 0.03 for e in errors.values()),
                positions_m=x.tolist(), bonds=bonds.tolist(), masses_kg=masses.tolist(),
                infinitesimal_boundary='all nodes prescribed affine strain; bending is polynomial displacement, no transverse relaxation')


def relaxed_tension(cells_y, *, corrected=False, **kwargs):
    """Actual central-spring equilibrium with transverse displacements free.

    Left/right x displacement prescribed; one y DOF removes translation.
    No contact/damage. Dense solve limits this independent qualification test.
    """
    case = qualify(cells_y, **kwargs)
    geometry = case['geometry']
    x, bonds, _ = lattice(cells_y, length=geometry['length_m'], height=geometry['height_m'],
                          thickness=geometry['thickness_m'])
    n = len(x)
    if n > 600:
        raise ValueError('dense equilibrium patch test limited to 600 nodes')
    vector = x[bonds[:, 1]] - x[bonds[:, 0]]
    normal = vector / torch.linalg.vector_norm(vector, dim=1)[:, None]
    matrix = torch.zeros((2 * n, 2 * n), dtype=torch.float64)
    coefficients = corrected_stiffness(x, bonds, case['target']['E_Pa'], geometry['thickness_m']) if corrected else torch.full((len(bonds),), case['stiffness_N_per_m'], dtype=torch.float64)
    for pair, direction, coefficient in zip(bonds, normal, coefficients):
        dofs = torch.tensor([2 * pair[0], 2 * pair[0] + 1, 2 * pair[1], 2 * pair[1] + 1])
        b = torch.cat((-direction, direction))
        matrix[dofs[:, None], dofs[None, :]] += coefficient * b[:, None] * b[None, :]
    left = torch.where(x[:, 0] == 0)[0]
    right = torch.where(x[:, 0] == geometry['length_m'])[0]
    constrained = torch.cat((2 * left, 2 * right, torch.tensor([2 * left[0] + 1])))
    free_mask = torch.ones(2 * n, dtype=torch.bool)
    free_mask[constrained] = False
    free = torch.where(free_mask)[0]
    displacement = torch.zeros(2 * n, dtype=torch.float64)
    strain = 1e-4
    displacement[2 * right] = strain * geometry['length_m']
    displacement[free] = torch.linalg.solve(matrix[free[:, None], free[None, :]],
                -matrix[free[:, None], constrained[None, :]] @ displacement[constrained])
    reaction = matrix @ displacement
    modulus = float(reaction[2 * right].sum()) / (geometry['thickness_m'] * geometry['height_m'] * strain)
    top = x[:, 1] == geometry['height_m'] / 2
    bottom = x[:, 1] == -geometry['height_m'] / 2
    transverse_strain = float(displacement.reshape(-1, 2)[top, 1].mean() - displacement.reshape(-1, 2)[bottom, 1].mean()) / geometry['height_m']
    return dict(cells_y=cells_y, measured_E_Pa=modulus, target_E_Pa=case['target']['E_Pa'],
                E_relative_error=abs(modulus / case['target']['E_Pa'] - 1),
                apparent_nu=-transverse_strain / strain, target_nu=1/3 if corrected else case['target']['nu'], corrected=corrected,
                free_dof_residual_N=float(reaction[free].abs().max()),
                displacement_m=displacement.reshape(-1, 2).tolist(),
                reaction_N=reaction.reshape(-1, 2).tolist(),
                boundary='prescribed left/right x displacement, transverse y free, one y translation fixed')


def corrected_stiffness(positions, bonds, young_modulus, thickness):
    """Per-bond N/m coefficients for realizable isotropic nu=1/3.

    Axial k=2*diagonal k; axial bonds on rectangle boundaries carry 1/2
    control-volume weight. These coefficients require a future per-bond
    IceDEM interface and are not silently passed to its scalar config.
    """
    vector = positions[bonds[:, 1]] - positions[bonds[:, 0]]
    axial = (vector[:, 0] == 0) | (vector[:, 1] == 0)
    midpoint = (positions[bonds[:, 1]] + positions[bonds[:, 0]]) / 2
    on_boundary = (midpoint[:, 0] == positions[:, 0].min()) | (midpoint[:, 0] == positions[:, 0].max()) | (midpoint[:, 1] == positions[:, 1].min()) | (midpoint[:, 1] == positions[:, 1].max())
    kd = 3 * young_modulus * thickness / 8
    return torch.where(axial, 2 * kd, kd) * torch.where(axial & on_boundary, 0.5, 1.0)


def corrected_patch(cells_y):
    x, bonds, masses = lattice(cells_y)
    coefficients = corrected_stiffness(x, bonds, 1e6, 0.2)
    vector = x[bonds[:, 1]] - x[bonds[:, 0]]
    normals = vector / torch.linalg.vector_norm(vector, dim=1)[:, None]
    def potential(u):
        extension = ((u[bonds[:, 1]] - u[bonds[:, 0]]) * normals).sum(1)
        return float(0.5 * (coefficients * extension.square()).sum())
    eps = 1e-4
    ux = torch.stack((eps*x[:,0], torch.zeros(len(x))),1)
    uy = torch.stack((torch.zeros(len(x)),eps*x[:,1]),1)
    shear = torch.stack((eps*x[:,1],torch.zeros(len(x))),1)
    factor=0.8*0.4*0.2*eps**2
    measured=dict(C11_Pa=2*potential(ux)/factor,
                  C12_Pa=(potential(ux+uy)-potential(ux)-potential(uy))/factor,
                  G_Pa=2*potential(shear)/factor)
    target=dict(C11_Pa=1125000.0,C12_Pa=375000.0,G_Pa=375000.0)
    errors={key:abs(measured[key]/target[key]-1) for key in target}
    tension=relaxed_tension(cells_y,corrected=True)
    return dict(cells_y=cells_y, mass_kg=float(masses.sum()), coefficients_N_per_m=coefficients.tolist(),
                measured=measured,target=target,relative_errors=errors,tension=tension,
                qualified=max(errors.values())<0.03 and tension['E_relative_error']<0.03 and abs(tension['apparent_nu']-1/3)<0.01,
                actual_IceDEM_backend_qualified=False)
