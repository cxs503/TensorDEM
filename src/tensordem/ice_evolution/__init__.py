"""Time evolution of a broken-ice (rubble) channel: sintering + refreezing.

Campaign node IN155-N09.  A ship repeatedly transits a channel that it has
itself broken into ice rubble.  Between transits the rubble *heals*: grain
necks sinter (Frenkel viscous growth) and the pore water refreezes (latent
heat conducted to the ambient).  The macroscopic rubble strength therefore
grows with the number of transits, and so does the ice resistance the next
transit must overcome.  This package answers, quantitatively, *how much the
resistance changes after N transits*.

Modules
-------
``sintering``
    Frenkel viscous neck growth ``a(t) = (3 gamma t / (2 eta))**(1/3)`` with a
    temperature-dependent ice viscosity, and the resulting DEM bond
    stiffness / failure force.
``refreezing``
    1-D enthalpy-method refreezing of supercooled pore water: heat conduction
    plus a latent-heat source, giving the porosity ``phi(t)`` and a closed
    energy budget (latent heat + sintering surface energy vs conduction loss).
``strength``
    Macroscopic rubble strength ``sigma(t) = sigma_ice (1 - phi)^n f_bond``
    combining the solid-fraction effect and the sintered-bond effect.
``resistance``
    A minimal 2-D bonded-disk DEM in which a rigid platen (ship bow) drives
    through a rubble bed; the platen force is the ice resistance.  Calibrates
    the ``R(sigma)`` relation used by the channel driver.
``channel``
    The N-transit driver: evolve the field between transits, evaluate
    ``sigma(N)`` and ``R_ice(N)``, and record the per-transit resistance
    history.
``sensitivity``
    Sweeps over temperature, initial porosity and transit interval, emitting
    the resistance-decay envelope.
``study``
    The driver that runs the verification chain and emits the machine-checkable
    criteria ledger plus the paper figures.

Units are SI throughout (m, kg, s, N, K, J).
"""

from .sintering import (
    ICE_SURFACE_ENERGY,
    ICE_VISCOSITY_EA,
    ICE_VISCOSITY_REF,
    ICE_VISCOSITY_T_REF,
    SinterState,
    frenkel_neck_radius,
    ice_viscosity,
    sinter_bond,
)
from .refreezing import (
    RefreezeConfig,
    RefreezeResult,
    run_refreezing,
)
from .strength import (
    StrengthConfig,
    rubble_strength,
    strength_growth,
)
from .resistance import (
    PlatenDEM,
    PlatenConfig,
    run_platen,
)
from .channel import (
    ChannelConfig,
    ChannelResult,
    run_channel_evolution,
)
from .sensitivity import (
    SensitivityConfig,
    run_sensitivity,
)
from .energy import (
    EnergyBudget,
    energy_closure,
)

__all__ = [
    "ICE_SURFACE_ENERGY",
    "ICE_VISCOSITY_EA",
    "ICE_VISCOSITY_REF",
    "ICE_VISCOSITY_T_REF",
    "SinterState",
    "frenkel_neck_radius",
    "ice_viscosity",
    "sinter_bond",
    "RefreezeConfig",
    "RefreezeResult",
    "run_refreezing",
    "StrengthConfig",
    "rubble_strength",
    "strength_growth",
    "PlatenDEM",
    "PlatenConfig",
    "run_platen",
    "ChannelConfig",
    "ChannelResult",
    "run_channel_evolution",
    "SensitivityConfig",
    "run_sensitivity",
    "EnergyBudget",
    "energy_closure",
]
