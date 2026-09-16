#!/usr/bin/env python3
"""
verify.py - checks the model against closed-form theory.

A simulation that produces plausible-looking waveforms has proved nothing. Each
check below computes a quantity two ways: once by running the model, and once
from an expression that can be derived on paper in a couple of lines. If the
two disagree by more than the stated tolerance the check fails and the script
exits non-zero, so this can be wired to CI.

Run it with:  python3 verify.py
"""

from __future__ import annotations

import sys

import numpy as np

from models.inverter_blocks import harmonic_spectrum, thd
from models.inverter_model import InverterParams, simulate
from models.sweep import linear_limit, sweep_modulation

FAILED = 0
PASSED = 0


def check(name, measured, expected, tol_pct, note=""):
    global FAILED, PASSED
    err = 100.0 * (measured - expected) / expected if expected else measured
    ok = abs(err) <= tol_pct
    PASSED += ok
    FAILED += not ok
    mark = "pass" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    print(f"         model {measured:12.5f}   theory {expected:12.5f}   "
          f"error {err:+7.3f} %   tolerance {tol_pct:.2f} %")
    if note:
        print(f"         {note}")


def check_true(name, ok, detail=""):
    """For the checks that are a yes or no rather than a number."""
    global FAILED, PASSED
    PASSED += bool(ok)
    FAILED += not ok
    print(f"  [{'pass' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def main():
    print("Three-phase SPWM inverter - model against theory")
    print("=" * 66)

    # ---------------------------------------------------------------
    print("\n1. Fundamental output in the linear range")
    print("   For m <= 1 the pole voltage fundamental is m*Vdc/2, so the")
    print("   line-to-line fundamental is sqrt(3) times that. This is the")
    print("   single most important property of carrier-based PWM: output is")
    print("   proportional to the reference, with no correction factor.")
    for m in (0.4, 0.7, 0.9, 1.0):
        p = InverterParams(m=m, scheme="spwm")
        r = simulate(p, cycles=4)
        check(f"V_ab fundamental at m = {m}", r["v1_ab_peak"],
              m * p.vdc * np.sqrt(3) / 2, tol_pct=0.5)

    # ---------------------------------------------------------------
    print("\n2. Load current follows from the load impedance alone")
    print("   The fundamental current is the fundamental phase voltage")
    print("   divided by |Z| = sqrt(R^2 + (w L)^2). If this check passes, the")
    print("   inverter and the load are being solved consistently.")
    p = InverterParams(m=0.9, scheme="spwm")
    r = simulate(p, cycles=6)
    z = np.hypot(p.R, 2 * np.pi * p.f1 * p.L)
    i1_rms_theory = (p.m * p.vdc / 2) / z / np.sqrt(2)
    check("phase current rms", r["ia_rms"], i1_rms_theory, tol_pct=1.0,
          note=f"|Z| = {z:.3f} ohm at {p.f1:.0f} Hz")

    # ---------------------------------------------------------------
    print("\n3. The DC bus utilisation gain from zero-sequence injection")
    print("   SPWM runs out at m = 1 and reaches sqrt(3)/(2 sqrt(2)) = 0.6124")
    print("   of Vdc. Injection reaches m = 2/sqrt(3) and 1/sqrt(2) = 0.7071.")
    print("   The ratio is 2/sqrt(3), the 15.47 percent that is the whole")
    print("   argument for using SVPWM over plain SPWM.")
    grid = np.array([1.0, 2.0 / np.sqrt(3.0)])
    sw = sweep_modulation(grid, schemes=("spwm", "svpwm"), cycles=2)
    u_spwm = float(np.interp(1.0, sw["spwm"]["m"], sw["spwm"]["util"]))
    u_svpwm = float(np.interp(linear_limit("svpwm"), sw["svpwm"]["m"],
                              sw["svpwm"]["util"]))
    check("SPWM utilisation at its limit", u_spwm,
          np.sqrt(3) / (2 * np.sqrt(2)), tol_pct=0.5)
    check("SVPWM utilisation at its limit", u_svpwm,
          1 / np.sqrt(2), tol_pct=0.5)
    check("ratio between them", u_svpwm / u_spwm, 2 / np.sqrt(3), tol_pct=0.5,
          note="the headline 15.47 percent")

    # ---------------------------------------------------------------
    print("\n4. Zero-sequence injection cannot reach the load")
    print("   Third-harmonic injection adds a term common to all three")
    print("   phases. A common term cancels in every line-to-line voltage, so")
    print("   the third harmonic must be absent from v_ab no matter how much")
    print("   of it was injected. This is the claim the whole scheme rests on.")
    p3 = InverterParams(m=1.1, scheme="thipwm")
    r3 = simulate(p3, cycles=4)
    freq, amp = harmonic_spectrum(r3["t"], r3["v_ab"], p3.f1, cycles=2)
    df = freq[1] - freq[0]
    a1 = amp[int(round(p3.f1 / df))]
    a3 = amp[int(round(3 * p3.f1 / df))]
    ratio = a3 / a1
    check_true("third harmonic absent from v_ab", ratio < 0.01,
               f"third / fundamental = {ratio:.6f}, must be below 0.01")

    # ---------------------------------------------------------------
    print("\n5. The frequency modulation ratio was chosen, not accepted")
    print("   mf is odd and a multiple of three. Odd makes the waveform half-")
    print("   wave symmetric, which removes the even harmonics; a multiple of")
    print("   three puts the carrier harmonic itself into the zero sequence,")
    print("   where the line-to-line voltage cannot see it.")
    p = InverterParams(m=0.9, scheme="spwm")
    mf = int(round(p.mf))
    check_true("mf is odd and a multiple of three",
               (mf % 2 == 1) and (mf % 3 == 0),
               f"mf = {mf}: odd {mf % 2 == 1}, multiple of three {mf % 3 == 0}")
    r = simulate(p, cycles=4)
    freq, amp = harmonic_spectrum(r["t"], r["v_ab"], p.f1, cycles=2)
    df = freq[1] - freq[0]
    carrier_line = amp[int(round(mf * p.f1 / df))] / amp[int(round(p.f1 / df))]
    check_true("carrier harmonic absent from v_ab", carrier_line < 0.01,
               f"carrier line = {carrier_line:.6f} of the fundamental, "
               f"which is the payoff for making mf a multiple of three")

    # ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{PASSED} checks passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
