"""
sweep.py - the study that compares the three modulation schemes.

A single waveform tells you what one operating point looks like. The question
worth answering is how the schemes differ across the whole modulation range,
and that needs a sweep.

The load is deliberately left out of this sweep. The quantity being compared is
what the inverter produces, and the line-to-line voltage is algebraic: it
depends on the gate signals and the DC bus and on nothing with a state.
Dropping the load makes each point roughly forty times cheaper to compute and
changes none of the answers.
"""

from __future__ import annotations

import numpy as np

from blockset import Model, Scope, Triangle
from models.inverter_blocks import (Bridge, CarrierPWM, Modulator,
                                    harmonic_spectrum, rms, thd)


def voltage_only(p_vdc, f1, fc, m, scheme, dt, cycles):
    """Run the modulator and bridge alone, and return the line-to-line voltage."""
    mdl = Model(f"{scheme}-m{m:.3f}", dt=dt)
    ref = mdl.add(Modulator("ref", m=m, f1=f1, scheme=scheme))
    car = mdl.add(Triangle("carrier", freq_hz=fc))
    pwm = mdl.add(CarrierPWM("pwm"))
    brg = mdl.add(Bridge("bridge", vdc=p_vdc))
    scope = mdl.add(Scope("scope", ["bridge"]))
    mdl.connect(ref, pwm, port=0)
    mdl.connect(car, pwm, port=1)
    mdl.connect(pwm, brg)
    mdl.run(cycles / f1)

    t = scope.time()
    bridge = np.array(scope.data["bridge"])
    return t, bridge[:, 6], bridge[:, 3]        # v_ab, v_an


def sweep_modulation(m_values, schemes=("spwm", "thipwm", "svpwm"),
                     vdc=600.0, f1=50.0, fc=1950.0, dt=1e-6, cycles=2):
    """THD and fundamental against modulation index, per scheme.

    The reference for "how much output" is the rms of the line-to-line
    fundamental divided by the DC bus voltage, because that is the number that
    decides whether a given bus can reach a given motor voltage. Quoting the
    modulation index alone hides the difference between the schemes, since the
    same index means different things in each.
    """
    out = {s: dict(m=[], thd=[], v1_rms=[], util=[], v1_peak=[])
           for s in schemes}
    f_max = 2.5 * fc

    for scheme in schemes:
        for m in m_values:
            t, v_ab, _ = voltage_only(vdc, f1, fc, m, scheme, dt, cycles)
            freq, amp = harmonic_spectrum(t, v_ab, f1, cycles=1)
            d, v1 = thd(freq, amp, f1, f_max=f_max)
            out[scheme]["m"].append(m)
            out[scheme]["thd"].append(d)
            out[scheme]["v1_peak"].append(v1)
            out[scheme]["v1_rms"].append(v1 / np.sqrt(2))
            out[scheme]["util"].append(v1 / np.sqrt(2) / vdc)

    for s in schemes:
        for k in out[s]:
            out[s][k] = np.asarray(out[s][k], dtype=float)
    return out


def linear_limit(scheme):
    """The largest modulation index that still produces a proportional output.

    SPWM runs out at 1, when the sine reference just touches the carrier peak.
    Both injection schemes reach 2/sqrt(3) = 1.1547, because the term they add
    is common to all three phases: it moves the references down inside the
    carrier band without moving any line-to-line voltage, so the peaks fit
    where they otherwise would not.
    """
    return 1.0 if scheme == "spwm" else 2.0 / np.sqrt(3.0)
