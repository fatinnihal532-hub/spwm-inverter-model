"""
inverter_model.py - assembles the diagram and runs it.

The wiring below is the model. Everything else in this repository either feeds
it parameters or reads its scope.
"""

from __future__ import annotations

import numpy as np

from blockset import Model, Scope, Triangle
from models.inverter_blocks import (Bridge, CarrierPWM, Modulator,
                                    ThreePhaseRL, harmonic_spectrum, rms, thd)


class InverterParams:
    """One place for every number the model needs."""

    def __init__(self, vdc=600.0, f1=50.0, fc=1950.0, m=0.9,
                 scheme="spwm", R=10.0, L=20e-3, dt=1e-6):
        self.vdc, self.f1, self.fc, self.m = vdc, f1, fc, m
        self.scheme, self.R, self.L, self.dt = scheme, R, L, dt

    @property
    def mf(self):
        """Frequency modulation ratio. Odd and a multiple of 3 is the classic
        choice: odd makes the waveform half-wave symmetric so even harmonics
        vanish, and a multiple of 3 puts the carrier harmonic in the zero
        sequence where the line-to-line voltage cannot see it."""
        return self.fc / self.f1


def build(p: InverterParams):
    mdl = Model(f"3ph-{p.scheme}", dt=p.dt)

    ref = mdl.add(Modulator("ref", m=p.m, f1=p.f1, scheme=p.scheme))
    car = mdl.add(Triangle("carrier", freq_hz=p.fc))
    pwm = mdl.add(CarrierPWM("pwm"))
    brg = mdl.add(Bridge("bridge", vdc=p.vdc))
    load = mdl.add(ThreePhaseRL("load", R=p.R, L=p.L))
    scope = mdl.add(Scope("scope", ["ref", "carrier", "bridge", "load"]))

    mdl.connect(ref, pwm, port=0)
    mdl.connect(car, pwm, port=1)
    mdl.connect(pwm, brg)
    mdl.connect(brg, load)
    return mdl, scope


def simulate(p: InverterParams, cycles=6):
    """Run for a whole number of fundamental cycles and report the results."""
    mdl, scope = build(p)
    mdl.run(cycles / p.f1)

    t = scope.time()
    bridge = np.array(scope.data["bridge"])
    currents = np.array(scope.data["load"])
    refs = np.array(scope.data["ref"])

    v_ab = bridge[:, 6]
    v_an = bridge[:, 3]
    v_aO = bridge[:, 0]

    # analyse the last two cycles, by which time the load current transient
    # has died away and the answer is the steady-state one
    freq, amp = harmonic_spectrum(t, v_ab, p.f1, cycles=2)
    f_max = 50.0 * p.fc / 20.0          # a band wide enough to hold 2 x carrier
    thd_ab, v1_ab = thd(freq, amp, p.f1, f_max=f_max)

    freq_i, amp_i = harmonic_spectrum(t, currents[:, 0], p.f1, cycles=2)
    thd_i, i1 = thd(freq_i, amp_i, p.f1, f_max=f_max)

    n_last = int(round(2 / p.f1 / p.dt))
    return dict(
        t=t, refs=refs, carrier=np.array(scope.data["carrier"]),
        v_aO=v_aO, v_an=v_an, v_ab=v_ab, i=currents,
        freq=freq, amp=amp, freq_i=freq_i, amp_i=amp_i,
        thd_vab=thd_ab, thd_ia=thd_i,
        v1_ab_peak=v1_ab, i1_peak=i1,
        vab_rms=rms(v_ab[-n_last:]),
        van_rms=rms(v_an[-n_last:]),
        ia_rms=rms(currents[-n_last:, 0]),
        f_max=f_max, params=p,
    )
