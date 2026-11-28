"""Fixed square particle network with a reversing width-integrated section power."""
import json
import math

import numpy as np

from tensordem.thermal import transient_lattice_temperatures


def measure(n):
    x = (np.arange(n)+.5)/n
    initial = np.empty((n+2, n))
    initial[0], initial[-1] = 1., 0.
    initial[1:-1] = (1-x + .15*np.sin(5*np.pi*x))[:, None] + (
        np.sin(3*np.pi*x)[:, None]*np.cos(4*np.pi*x))
    # Unit depth, conductivity and volumetric capacity; end links are half-length.
    rx = np.ones((n+1, n))
    rx[[0, -1]] = .5
    time = np.linspace(0., .012, n+1)
    history, links, sections = transient_lattice_temperatures(
        (rx, 1.), 1/n**2, time, initial, periodic_y=True, return_link_heat=True,
        return_directional_section_heat=True)
    face = 3*n//8
    left, right = [sections[key][:, face].sum() for key in ("leftward", "rightward")]
    rate = 25*np.pi**2
    amplitude = .75*np.pi*math.cos(15*np.pi/8)
    root = math.log(amplitude)/rate
    def primitive(t):
        return t+amplitude*np.expm1(-rate*t)/rate
    ref_left = -primitive(root)
    ref_right = ref_left+primitive(time[-1])
    net = sections["rightward"]-sections["leftward"]
    stored = (history[:, 1:face+1]-initial[1:face+1]).sum(axis=(1, 2))/n**2
    return dict(n=n, steps=n, left_right_heat_J=[left, right], analytic_left_right_heat_J=[ref_left, ref_right],
                directional_error_J=max(abs(left-ref_left), abs(right-ref_right)),
                section_net_mismatch_J=np.max(abs(net-links["x"].sum(axis=2))),
                left_storage_residual_J=np.max(abs(stored[1:]-np.cumsum(net[:, 0]-net[:, face]))))


if __name__ == "__main__":
    runs = [measure(n) for n in (16, 32, 64)]
    print(json.dumps(runs, indent=2, allow_nan=False))
    assert all(b["directional_error_J"] < a["directional_error_J"]/3.5 for a,b in zip(runs,runs[1:]))
    assert all(max(row["section_net_mismatch_J"], row["left_storage_residual_J"]) < 1e-12 for row in runs)
