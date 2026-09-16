#!/usr/bin/env python3
"""
make_figures.py - regenerates every figure in figures/ from the model.

No figure in this repository was drawn by hand or copied from anywhere. Run
this script and they are all rebuilt from the simulation, which means a change
to the model shows up in the pictures instead of quietly disagreeing with them.

Each figure is written twice, once for a light page and once for a dark one.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from models.inverter_blocks import harmonic_spectrum
from models.inverter_model import InverterParams, simulate
from models.plotstyle import annotate, footnote, render, tidy
from models.sweep import linear_limit, sweep_modulation

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

PHASE_LABELS = ("phase a", "phase b", "phase c")
SCHEME_LABELS = {"spwm": "SPWM", "thipwm": "Third-harmonic injection",
                 "svpwm": "SVPWM (min-max injection)"}


# ----------------------------------------------------------------------
def fig_modulation(results):
    """How the three schemes shape the reference, and why it buys headroom."""

    def build(c):
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
        for ax, (scheme, r) in zip(axes, results.items()):
            p = r["params"]
            t = r["t"]
            one = t <= 1.0 / p.f1
            # the carrier is context, not data: kept thin and faint so the
            # three references stay the thing the eye lands on
            ax.plot(t[one] * 1e3, r["carrier"][one], lw=0.7, alpha=0.5,
                    color=c["muted"], label="carrier", zorder=1)
            for i in range(3):
                ax.plot(t[one] * 1e3, r["refs"][one, i], lw=1.8,
                        color=c["series"][i], label=PHASE_LABELS[i], zorder=3)
            ax.axhline(1.0, lw=0.8, ls=(0, (4, 3)), color=c["axis"])
            ax.axhline(-1.0, lw=0.8, ls=(0, (4, 3)), color=c["axis"])
            ax.set_ylim(-1.45, 1.45)
            ax.set_ylabel("reference")
            ax.set_title(f"{SCHEME_LABELS[scheme]}   m = {p.m:.4f}", loc="left")
            tidy(ax, c)
        # legend above the first panel rather than inside it - at this carrier
        # density there is no empty corner left to put it in
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.22), ncol=4,
                       labelcolor=c["secondary"], handlelength=1.5,
                       columnspacing=1.4, borderaxespad=0.0, fontsize=8)
        axes[-1].set_xlabel("time (ms)")
        fig.suptitle("Reference waveforms against the carrier, one fundamental cycle",
                     y=1.005, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        footnote(fig,
                 "Injection flattens the peaks without touching any "
                 "line-to-line voltage, so the references fit inside the "
                 "carrier band at a modulation index of 1.1547 that plain "
                 "SPWM could not reach.",
                 c)
        return fig

    render(build, os.path.join(OUT, "modulation"))


def fig_waveforms(r):
    """The four voltages and currents, in the order you would probe them."""

    def build(c):
        p = r["params"]
        t0, t1 = 3.0 / p.f1, 5.0 / p.f1        # two settled cycles
        m = (r["t"] >= t0) & (r["t"] <= t1)
        tm = (r["t"][m] - t0) * 1e3

        fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True)

        axes[0].plot(tm, r["v_aO"][m], lw=0.9, color=c["series"][0])
        axes[0].set_ylabel("$v_{aO}$  (V)")
        axes[0].set_title("Pole voltage: two levels, nothing else", loc="left")

        axes[1].plot(tm, r["v_ab"][m], lw=0.9, color=c["series"][1])
        axes[1].set_ylabel("$v_{ab}$  (V)")
        axes[1].set_title("Line-to-line voltage: three levels, because the two "
                          "poles rarely switch together", loc="left")

        axes[2].plot(tm, r["v_an"][m], lw=0.9, color=c["series"][2])
        axes[2].set_ylabel("$v_{an}$  (V)")
        axes[2].set_title("Phase voltage at the floating star point: the "
                          "six-step staircase under the switching", loc="left")

        for i in range(3):
            axes[3].plot(tm, r["i"][m, i], lw=1.7, color=c["series"][i],
                         label=PHASE_LABELS[i])
        axes[3].set_ylabel("load current  (A)")
        axes[3].set_title("Load current: the inductance has already done the "
                          "filtering", loc="left")
        axes[3].set_xlabel("time (ms)")
        # headroom for the legend, so it never lands on a waveform
        pk = float(np.max(np.abs(r["i"][m, :])))
        axes[3].set_ylim(-pk * 1.18, pk * 1.55)

        for ax in axes:
            tidy(ax, c)
        axes[3].legend(loc="upper center", ncol=3, labelcolor=c["secondary"],
                       handlelength=1.5, columnspacing=1.4, borderaxespad=0.3)

        fig.suptitle(f"Three-phase two-level inverter, {SCHEME_LABELS[p.scheme]}"
                     f",  $m$ = {p.m},  $m_f$ = {p.mf:.0f},  $V_{{dc}}$ = {p.vdc:.0f} V",
                     y=0.997, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        footnote(fig,
                 f"The load never sees the voltage distortion: "
                 f"{r['thd_vab']*100:.0f} % THD in the line-to-line voltage "
                 f"becomes {r['thd_ia']*100:.1f} % in the current, because the "
                 f"load impedance rises with frequency and the switching "
                 f"harmonics sit near {p.mf:.0f} times the fundamental.",
                 c)
        return fig

    render(build, os.path.join(OUT, "waveforms"))


def fig_spectrum(r):
    """Where the switching energy actually goes."""

    def build(c):
        p = r["params"]
        freq, amp = r["freq"], r["amp"]
        order = freq / p.f1
        keep = (order > 0.5) & (order <= 2.6 * p.mf)
        h, a = order[keep], amp[keep] / p.vdc

        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        ax.vlines(h, 0, a, lw=1.4, color=c["series"][0])
        ax.set_xlabel("harmonic order  ($f / f_1$)")
        ax.set_ylabel("amplitude / $V_{dc}$")
        ax.set_xlim(0, 2.6 * p.mf)
        ax.set_ylim(0, max(a) * 1.18)
        tidy(ax, c)

        i1 = int(np.argmin(abs(h - 1.0)))
        annotate(ax, f"fundamental\n{r['v1_ab_peak']:.0f} V peak",
                 xy=(1.0, a[i1]), xytext=(6.0, a[i1] * 0.86), c=c)
        for k, lab in ((1, "first carrier cluster\naround $m_f$"),
                       (2, "second cluster\naround $2m_f$")):
            band = (h > k * p.mf - 5) & (h < k * p.mf + 5)
            if not band.any():
                continue
            pk = float(a[band].max())
            annotate(ax, lab, xy=(k * p.mf, pk),
                     xytext=(k * p.mf - 14, pk + max(a) * 0.14), c=c)

        ax.set_title(
            f"Line-to-line voltage spectrum   THD = {r['thd_vab']*100:.1f} % "
            f"to {r['f_max']/1e3:.0f} kHz", loc="left")
        footnote(fig,
                 "No harmonic below the carrier cluster, which is the whole "
                 "point of pushing the carrier high: the load's own inductance "
                 "filters what is left.", c)
        fig.tight_layout()
        return fig

    render(build, os.path.join(OUT, "spectrum"))


def fig_sweep(sw):
    """The comparison the project exists to make."""

    def build(c):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)

        # Third-harmonic injection and SVPWM sit almost on top of each other,
        # which is itself a result worth being able to see. A dash pattern on
        # one of them keeps both readable where they coincide, so identity
        # never rests on colour alone.
        styles = {"spwm": "-", "thipwm": (0, (5, 3)), "svpwm": "-"}
        for i, (scheme, d) in enumerate(sw.items()):
            lin = d["m"] <= linear_limit(scheme) + 1e-9
            axes[0].plot(d["m"][lin], d["thd"][lin] * 100, lw=1.9,
                         ls=styles[scheme], color=c["series"][i],
                         label=SCHEME_LABELS[scheme])
        axes[0].set_ylabel("line-to-line voltage THD  (%)")
        axes[0].set_title("Distortion, each scheme over its own linear range",
                          loc="left")
        axes[0].legend(loc="upper right", labelcolor=c["secondary"],
                       handlelength=2.2, borderaxespad=0.3)

        # The lower panel runs past m = 1 on purpose. Up to that point every
        # scheme puts out the same volts per unit of modulation index; past it
        # SPWM has nowhere left to go and its curve bends away from the line,
        # which is what overmodulation looks like when it is plotted rather
        # than described.
        sp, sv = sw["spwm"], sw["svpwm"]
        axes[1].plot(sv["m"], sv["util"], lw=1.9, color=c["series"][2],
                     label="third-harmonic injection and SVPWM")
        axes[1].plot(sp["m"], sp["util"], lw=1.9, ls=(0, (5, 3)),
                     color=c["series"][0], label="SPWM")

        u_sp = float(np.interp(1.0, sp["m"], sp["util"]))
        u_sv = float(np.interp(linear_limit("svpwm"), sv["m"], sv["util"]))
        gain = 100.0 * (u_sv / u_sp - 1.0)

        for lab, x, y, col in (("SPWM's limit", 1.0, u_sp, c["series"][0]),
                               ("injection limit", linear_limit("svpwm"),
                                u_sv, c["series"][2])):
            axes[1].plot([x], [y], "o", ms=8, color=col,
                         mec=c["surface"], mew=1.8, zorder=5)
        axes[1].axvspan(1.0, sp["m"].max(), color=c["muted"], alpha=0.10, lw=0)

        axes[1].set_ylabel("$V_{ab,1,rms} / V_{dc}$")
        axes[1].set_xlabel("modulation index  $m$")
        axes[1].set_ylim(0, u_sv * 1.34)
        axes[1].set_title(
            f"Output from the same DC bus: {u_sp:.3f} against {u_sv:.3f} "
            f"$V_{{dc}}$, a gain of {gain:.1f} %", loc="left")
        axes[1].legend(loc="upper left", labelcolor=c["secondary"],
                       handlelength=2.2, borderaxespad=0.4)
        axes[1].text(1.077, u_sv * 0.30, "SPWM is\novermodulating\nhere",
                     fontsize=8, color=c["secondary"], ha="center")

        for ax in axes:
            tidy(ax, c)

        fig.suptitle("Three modulation schemes compared",
                     y=0.998, fontsize=11, color=c["ink"], ha="left", x=0.005)
        fig.tight_layout()
        # Stated as a bus reduction the number is 1 - 1/1.1547, not the gain
        # itself. Quoting the gain in both directions is a common slip and it
        # overstates the saving by two points.
        bus_cut = 100.0 * (1.0 - u_sp / u_sv)
        footnote(fig,
                 f"The {gain:.1f} % is the whole argument for injection. Read "
                 f"the other way round, a drive that needs a given line "
                 f"voltage can reach it from a DC bus {bus_cut:.1f} % lower, "
                 f"which buys lower device voltage ratings for the same output.",
                 c)
        return fig

    render(build, os.path.join(OUT, "scheme_comparison"))


# ----------------------------------------------------------------------
def main():
    print("running the waveform cases")
    results = {}
    for scheme, m in (("spwm", 1.0), ("thipwm", 1.1547), ("svpwm", 1.1547)):
        results[scheme] = simulate(
            InverterParams(m=m, scheme=scheme), cycles=6)
        r = results[scheme]
        print(f"  {scheme:7s} V1={r['v1_ab_peak']:7.2f} V  "
              f"THD_v={r['thd_vab']*100:5.2f} %  THD_i={r['thd_ia']*100:4.2f} %")

    print("sweeping the modulation index")
    # the two linear limits are forced onto the grid rather than left to fall
    # between samples, so the numbers the figure quotes are measured at the
    # limits themselves and not interpolated towards them
    m_grid = np.unique(np.concatenate([
        np.linspace(0.1, 2.0 / np.sqrt(3.0), 18),
        [1.0, 2.0 / np.sqrt(3.0)]]))
    sw = sweep_modulation(m_grid)

    print("drawing")
    fig_modulation(results)
    fig_waveforms(simulate(InverterParams(m=0.9, scheme="spwm"), cycles=6))
    fig_spectrum(results["spwm"])
    fig_sweep(sw)
    print("done")


if __name__ == "__main__":
    main()
