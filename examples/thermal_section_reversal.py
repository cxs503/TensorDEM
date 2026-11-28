"""Fixed square particle network with a reversing width-integrated section power."""
import argparse
import json
import math

import numpy as np

from tensordem.thermal import transient_lattice_temperatures


def measure(n, *, double_reversal=False):
    x = (np.arange(n)+.5)/n
    initial = np.empty((n+2, n))
    initial[0], initial[-1] = 1., 0.
    face = 3*n//8
    if double_reversal:
        roots = np.array([.003, .015])
        wave = np.array([3., 5.])*np.pi
        coefficients = np.linalg.solve(np.exp(-roots[:, None]*wave**2), np.ones(2))
        amplitudes = coefficients/(wave*np.cos(wave*face/n))
        axial = amplitudes @ np.sin(wave[:, None]*x)
    else:
        axial = .15*np.sin(5*np.pi*x)
    initial[1:-1] = (1-x + axial)[:, None] + (
        np.sin(3*np.pi*x)[:, None]*np.cos(4*np.pi*x))
    # Unit depth, conductivity and volumetric capacity; end links are half-length.
    rx = np.ones((n+1, n))
    rx[[0, -1]] = .5
    steps = 2*n if double_reversal else n
    time = np.linspace(0., .025 if double_reversal else .012, steps+1)
    history, links, sections = transient_lattice_temperatures(
        (rx, 1.), 1/n**2, time, initial, periodic_y=True, return_link_heat=True,
        return_directional_section_heat=True)
    net = sections["rightward"]-sections["leftward"]
    link_net = links["x"].sum(axis=2)
    if double_reversal:
        power = ((history[:, face]-history[:, face+1])/rx[face]).sum(axis=1)
        switches, = np.nonzero(np.diff(power > 0))
        assert len(switches) == 2 and power[0] > 0 and power[-1] > 0
        observed = time[switches]-power[switches]*np.diff(time)[switches]/np.diff(power)[switches]
        # A crossing step contributes to two lobes through the solver's
        # directional integrals of the affine Crank-Nicolson link power.
        i, j = switches
        forward = sections["rightward"][:, face]
        lobes = np.array([forward[:i+1].sum(), -sections["leftward"][:, face].sum(), forward[j:].sum()])
        endpoints = np.r_[0., roots, time[-1]]
        reference = np.diff(endpoints + np.expm1(-endpoints[:, None]*wave**2) @ (coefficients/wave**2))
        delta = (history[1:, 1:-1]-initial[1:-1])/n**2
        stored = np.column_stack((delta[:, :face].sum(axis=(1, 2)), delta[:, face:].sum(axis=(1, 2))))
        incoming = np.column_stack((link_net[:, 0]-link_net[:, face], link_net[:, face]-link_net[:, -1]))
        return dict(n=n, steps=steps, reversal_times_s=observed.tolist(), analytic_reversal_times_s=roots.tolist(),
                    reversal_error_s=np.max(abs(observed-roots)), signed_lobe_heat_J=lobes.tolist(),
                    analytic_signed_lobe_heat_J=reference.tolist(), directional_error_J=np.max(abs(lobes-reference)),
                    section_net_mismatch_J=np.max(abs(net-link_net)),
                    lobe_sum_residual_J=abs(lobes.sum()-link_net[:, face].sum()),
                    subdomain_balance_J=np.max(abs(stored-incoming.cumsum(axis=0))))
    left, right = [sections[key][:, face].sum() for key in ("leftward", "rightward")]
    rate = 25*np.pi**2
    amplitude = .75*np.pi*math.cos(15*np.pi/8)
    root = math.log(amplitude)/rate
    def primitive(t):
        return t+amplitude*np.expm1(-rate*t)/rate
    ref_left = -primitive(root)
    ref_right = ref_left+primitive(time[-1])
    stored = (history[:, 1:face+1]-initial[1:face+1]).sum(axis=(1, 2))/n**2
    return dict(n=n, steps=n, left_right_heat_J=[left, right], analytic_left_right_heat_J=[ref_left, ref_right],
                directional_error_J=max(abs(left-ref_left), abs(right-ref_right)),
                section_net_mismatch_J=np.max(abs(net-links["x"].sum(axis=2))),
                left_storage_residual_J=np.max(abs(stored[1:]-np.cumsum(net[:, 0]-net[:, face]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--double-reversal", action="store_true", help="verify three signed heat intervals")
    args = parser.parse_args()
    runs = [measure(n, double_reversal=args.double_reversal) for n in (16, 32, 64)]
    print(json.dumps(runs, indent=2, allow_nan=False))
    assert all(b["directional_error_J"] < a["directional_error_J"]/3.5 for a,b in zip(runs,runs[1:]))
    balance_key = "subdomain_balance_J" if args.double_reversal else "left_storage_residual_J"
    assert all(max(row["section_net_mismatch_J"], row[balance_key]) < 1e-12 for row in runs)
    if args.double_reversal:
        assert max(row["lobe_sum_residual_J"] for row in runs) < 1e-12
        assert all(b["reversal_error_s"] < a["reversal_error_s"]/3.5 for a,b in zip(runs,runs[1:]))
