"""
inverter_blocks.py - the power stage of a three-phase two-level inverter,
written as blocks for the engine in blockset/core.py.

Sign conventions, fixed once so every block below agrees:

  O     the mid-point of the DC bus. Pole voltages are measured against it,
        so a pole sits at +Vdc/2 or -Vdc/2 and never at an awkward offset.
  s_a   the switching function of leg a. 1 means the upper device conducts.
  N     the star point of the load, which floats. Its potential against O is
        the zero-sequence voltage, and it is the reason a line-to-line
        waveform is cleaner than a pole waveform.
"""

from __future__ import annotations

import numpy as np

from blockset import Block


class Modulator(Block):
    """Produces the three reference waveforms a carrier is compared against.

    Three schemes, and the difference between them is one added term that is
    common to all three phases. A common term cannot appear in any line-to-line
    voltage, because it cancels in the subtraction, and it cannot drive current
    through a floating star point. So it is free: it costs nothing at the load
    and it buys headroom at the pole.

    spwm    the reference is the sine itself. Linear up to m = 1.
    thipwm  a sixth of the third harmonic is added. The third harmonic is
            zero-sequence in a three-phase set, so it cancels at the load.
            Flattens the peaks and pushes the linear limit to 2/sqrt(3).
    svpwm   the injected term is -(max + min)/2 of the three sines, which
            centres the three references inside the carrier band. This is the
            same output space-vector modulation reaches, obtained without ever
            drawing a hexagon.
    """

    def __init__(self, name, m, f1, scheme="spwm"):
        super().__init__(name)
        self.m, self.f1, self.scheme = m, f1, scheme

    def out(self, t, x, u):
        th = 2 * np.pi * self.f1 * t
        base = self.m * np.sin(th - np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]))
        if self.scheme == "spwm":
            return base
        if self.scheme == "thipwm":
            return base + (self.m / 6.0) * np.sin(3 * th)
        if self.scheme == "svpwm":
            return base - 0.5 * (base.max() + base.min())
        raise ValueError(f"unknown scheme {self.scheme!r}")


class CarrierPWM(Block):
    """Three comparators sharing one triangular carrier.

    Input 0 is the three references, input 1 is the carrier. Output is the
    three switching functions. Comparing against a triangle rather than a
    sawtooth centres each pulse in its carrier period, which is why the first
    harmonic cluster sits at the carrier frequency instead of at half of it.
    """

    def out(self, t, x, u):
        refs, carrier = np.asarray(u[0], dtype=float), float(u[1])
        return (refs >= carrier).astype(float)


class Bridge(Block):
    """Switching functions in, voltages out.

    Returns [v_aO, v_bO, v_cO, v_an, v_bn, v_cn, v_ab], all in volts.

    The star-point voltage is the average of the three pole voltages. Removing
    it from each pole gives the voltage the load actually sees. In a two-level
    inverter that average only ever takes four values, which is why the phase
    voltage has its familiar six-step staircase underneath the switching.
    """

    def __init__(self, name, vdc):
        super().__init__(name)
        self.vdc = vdc

    def out(self, t, x, u):
        s = np.asarray(u[0], dtype=float)
        pole = (2.0 * s - 1.0) * (self.vdc / 2.0)      # +/- Vdc/2
        v_nO = pole.mean()                              # zero sequence
        phase = pole - v_nO
        return np.array([pole[0], pole[1], pole[2],
                         phase[0], phase[1], phase[2],
                         pole[0] - pole[1]])


class Pick(Block):
    """Selects one element of a vector signal, the way a Demux port does."""

    def __init__(self, name, index):
        super().__init__(name)
        self.index = index

    def out(self, t, x, u):
        return float(np.asarray(u[0], dtype=float)[self.index])


class ThreePhaseRL(Block):
    """A balanced wye RL load with a floating star point.

    L di/dt = v_phase - R i, one equation per phase. The floating star point is
    already accounted for by the bridge, which subtracts the zero-sequence
    voltage before handing the phase voltages over, so the three currents sum
    to zero on their own without a constraint being imposed.
    """

    direct_feedthrough = False
    n_states = 3

    def __init__(self, name, R, L, i0=(0.0, 0.0, 0.0)):
        super().__init__(name)
        self.R, self.L = R, L
        self.x0 = np.asarray(i0, dtype=float)

    def out(self, t, x, u):
        return x.copy()

    def deriv(self, t, x, u):
        v = np.asarray(u[0], dtype=float)[3:6]         # v_an, v_bn, v_cn
        return (v - self.R * x) / self.L


# ----------------------------------------------------------------------
# analysis helpers
# ----------------------------------------------------------------------

def harmonic_spectrum(t, y, f1, cycles):
    """One-sided amplitude spectrum over a whole number of fundamental cycles.

    Taking an exact integer number of cycles is what makes a rectangular window
    legitimate: every harmonic lands exactly on a bin, so no window function is
    needed and no energy leaks between bins. Get this wrong and the THD you
    report is the window's, not the converter's.
    """
    T = cycles / f1
    dt = t[1] - t[0]
    n = int(round(T / dt))
    seg = np.asarray(y, dtype=float)[-n:]
    spec = np.fft.rfft(seg) / n
    amp = np.abs(spec) * 2.0
    amp[0] = np.abs(spec[0])
    freq = np.fft.rfftfreq(n, dt)
    return freq, amp


def thd(freq, amp, f1, f_max):
    """THD referred to the fundamental, counting harmonics up to f_max.

    A switching converter has no finite THD if you integrate to infinity in
    theory and no repeatable one if you integrate to the solver's Nyquist
    limit in practice, so the band has to be stated. Quoting a THD without the
    band it was measured over is the most common way these numbers are misused.
    """
    k1 = int(round(f1 / (freq[1] - freq[0])))
    v1 = amp[k1]
    band = (freq > 0) & (freq <= f_max)
    band[k1] = False
    return float(np.sqrt(np.sum(amp[band] ** 2)) / v1), float(v1)


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, dtype=float) ** 2)))
