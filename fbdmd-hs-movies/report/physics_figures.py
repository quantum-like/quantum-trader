#!/usr/bin/env python3
"""Physics figures for the report: fig_energy, fig_pairing, fig_structure, fig_dynamics.

Reads report/data/physics_numbers_raw.json and the cache written by physics_analysis.py, writes
report/figures/<name>.{png,svg} and the manifest report/data/physics_manifest.json.
Run: python report/physics_analysis.py && python report/physics_figures.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report"))
from figstyle import *  # noqa: F401,F403
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

apply_style()   # figstyle: 9 pt labels, 8 pt ticks/legend; no override -> 8 pt is the floor everywhere

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "report", "data")
CACHE_DIR = os.environ.get("PHYS_CACHE",
                           "/tmp/claude-0/-home-user-quantum-trader/51ecfd3f-2c9f-5b9b-ba8a-5c6a2ddea84a/scratchpad/phys_cache")
C_ION = "#A3268F"    # magenta (= red + blue): "ions" / "+/-" quantities; deliberately NOT one of the integrator colours
C_SAME = "#333333"   # same-sign (++ / --) ion pairs: dark gray, dotted
C_DARK = "#222222"
C_TXT = "#333333"
TAU_TO_S = 1.0 / (0.025 * 30.0)   # 1 tau = 1.333 s of movie at 30 fps

R = json.load(open(os.path.join(DATADIR, "physics_numbers_raw.json")))
EN = np.genfromtxt(os.path.join(DATADIR, "energy_timeseries.csv"), delimiter=",", names=True)
DE = {k: EN[f"dE_{k.replace('-', '')}_kT"] for k in ("cv-phys", "cv", "kdk-phys", "naive")}   # genfromtxt strips '-' 
PR = np.load(os.path.join(CACHE_DIR, "pairing.npz"))
GR = np.load(os.path.join(CACHE_DIR, "gr.npz"))
DY = np.load(os.path.join(CACHE_DIR, "dyn.npz"))

manifest = []


def sig(x, n=3):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    x = float(x)
    if not np.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{n}g}")


def q(value, unit, note=None):
    d = {"value": sig(value), "unit": unit}
    if note:
        d["note"] = note
    return d


def sci(x, n=1):
    """2.3e-05 -> '2.3 × 10⁻⁵' for captions."""
    m, e = f"{x:.{n}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def sci_mt(x, n=1):
    """2.3e-05 -> r'$2.3 \\times 10^{-5}$' (mathtext, for text inside the figures)."""
    m, e = f"{x:.{n}e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def mn(txt):
    """ASCII hyphen-minus before digits -> proper minus sign (captions only)."""
    import re
    return re.sub(r"(?<![\w])-(?=\d)", "−", txt)


def secax_movie(ax):
    s = ax.secondary_xaxis("top", functions=(lambda t: t * TAU_TO_S, lambda s: s / TAU_TO_S))
    s.set_xlabel("movie time (s)", fontsize=8, labelpad=2)
    s.tick_params(labelsize=8)
    s.spines["top"].set_visible(True)
    return s


# ---------------------------------------------------------------------------
# Figure 1: energy conservation, temperature, collisions
# ---------------------------------------------------------------------------
def fig_energy():
    E = R["energy"]
    t = EN["time_tau"]
    fig = plt.figure(figsize=(FULL_W, 4.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, left=0.085, right=0.985, top=0.95, bottom=0.1)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.12, height_ratios=[1, 1])
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) E(t)-E(0) for the four integrators
    order = ["naive", "cv", "kdk-phys", "cv-phys"]
    ls = {"naive": "-", "cv": "--", "kdk-phys": "-.", "cv-phys": "-"}
    lw = {"naive": 1.1, "cv": 1.2, "kdk-phys": 1.2, "cv-phys": 1.5}
    for k in order:
        ax_a.plot(t, DE[k], ls[k], color=C_INTEG[k], lw=lw[k], label=f"{k}  {E[k]['drift_kT']:+.3f}")
    ax_a.axhline(0, color=C_DARK, lw=0.6)
    ax_a.set_xlabel("t (τ)")
    ax_a.set_ylabel("E(t) − E(0)  (kT)")
    ax_a.set_xlim(0, 18)
    ax_a.set_ylim(-0.5, 3.5)          # head room: the inset sits in the empty upper-right quadrant above the naive curve (max 1.81 kT)
    ax_a.legend(loc="upper left", handlelength=1.8, labelspacing=0.25, borderaxespad=0.2, title="drift over 18 τ (kT)",
                title_fontsize=8, alignment="left")
    # inset: the three good integrators (upper right; its floor incl. tick labels is at ~1.9 kT, above the naive maximum)
    ins = ax_a.inset_axes([0.61, 0.66, 0.38, 0.32])
    for k in ["cv", "kdk-phys", "cv-phys"]:
        ins.plot(t, DE[k], ls[k], color=C_INTEG[k], lw=lw[k] * 0.8)
    ins.axhline(0, color=C_DARK, lw=0.5)
    ins.set_xlim(0, 18)
    ins.set_ylim(-0.26, 0.11)
    ins.set_yticks([-0.2, -0.1, 0, 0.1])
    ins.tick_params(labelsize=7.5, pad=1.5, length=2)
    ins.set_xticks([0, 9, 18])
    ins.grid(True, lw=0.4)
    ins.set_title("without naive", fontsize=8, pad=2)
    panel_label(ax_a, "(a)")

    # (b) cv-phys: KE / PE fluctuations (top) and E - E0 (bottom)
    KE, PE = EN["KE_cvphys_kT"], EN["PE_cvphys_kT"]
    ax_b1.plot(t, KE - KE.mean(), color=C_ACCENT, lw=0.9, label="KE − ⟨KE⟩")
    ax_b1.plot(t, PE - PE.mean(), color=C_NEUTRAL, lw=0.9, label="PE − ⟨PE⟩")
    ax_b1.set_ylabel("δKE, δPE (kT)")
    ax_b1.set_ylim(-17, 24)
    ax_b1.set_yticks([-10, 0, 10])
    ax_b1.legend(loc="upper center", ncol=2, columnspacing=1.2, handlelength=1.5, borderaxespad=0.1)
    ax_b1.text(0.02, 0.05, f"std {KE.std():.1f} kT each", transform=ax_b1.transAxes, fontsize=8, va="bottom", color=C_TXT)
    ax_b1.tick_params(labelbottom=False)
    dE = DE["cv-phys"]
    ax_b2.plot(t, dE, color=C_INTEG["cv-phys"], lw=1.2, label="cv-phys  E(t) − E(0)")
    ax_b2.axhline(0, color=C_DARK, lw=0.6)
    ax_b2.set_ylim(-0.017, 0.052)
    ax_b2.set_yticks([0, 0.02, 0.04])
    ax_b2.set_ylabel("E − E(0) (kT)")
    ax_b2.set_xlabel("t (τ)")
    ax_b2.set_xlim(0, 18)
    ax_b2.text(0.03, 0.93, f"drift {E['cv-phys']['drift_kT']:+.3f} kT = {sci_mt(E['cv-phys']['drift_rel'])} |E₀|\n"
               f"max |δE| {E['cv-phys']['max_dev_kT']:.3f} kT", transform=ax_b2.transAxes, fontsize=8, va="top", color=C_TXT,
               linespacing=1.15)
    ax_b2.legend(loc="lower right", borderaxespad=0.1)
    panel_label(ax_b1, "(b)")

    # (c) T(t) cv-phys
    T = EN["T_cvphys"]
    Tm, Ts = T.mean(), T.std()
    ax_c.axhspan(Tm - Ts, Tm + Ts, color=C_INTEG["cv-phys"], alpha=0.15, lw=0)
    ax_c.axhline(Tm, color=C_DARK, lw=0.8, ls="--")
    ax_c.plot(t, T, color=C_INTEG["cv-phys"], lw=0.9, label="T(t), cv-phys")
    ax_c.axhline(1.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax_c.set_xlabel("t (τ)")
    ax_c.set_ylabel("T = 2 KE / 3(N−1)  (kT)")
    ax_c.set_xlim(0, 18)
    ax_c.set_ylim(0.988, 1.011)
    ax_c.text(0.98, 0.96, f"⟨T⟩ = {Tm:.4f} ± {Ts:.4f} kT (band: ± std)", transform=ax_c.transAxes, ha="right", va="top",
              fontsize=8, color=C_TXT)
    ax_c.legend(loc="lower left")
    panel_label(ax_c, "(c)")

    # (d) cumulative collisions
    GE = E["neutral"]["enskog_rate_per_particle_per_tau"]
    N = 1000
    ensk = 0.5 * N * GE * t
    nn, nc = EN["ncoll_neutral"], EN["ncoll_charged"]
    ax_d.plot(t, nn / 1e3, color=C_NEUTRAL, lw=1.8, label=f"neutral run  Γ = {E['neutral']['collision_rate_per_particle_per_tau']:.2f}/τ")
    ax_d.plot(t, nc / 1e3, color=C_INTEG["cv-phys"], lw=1.0, label=f"charged run  Γ = {E['cv-phys']['collision_rate_per_particle_per_tau']:.2f}/τ")
    ax_d.plot(t, ensk / 1e3, "--", color=C_DARK, lw=0.9, label=f"Enskog  Γ_E = {GE:.2f}/τ")
    ax_d.set_xlabel("t (τ)")
    ax_d.set_ylabel("collisions (10³)")
    ax_d.set_xlim(0, 18)
    ax_d.set_ylim(0, 135)
    # the data run along the diagonal: legend in the empty lower-right triangle, inset in the empty upper-left one
    ax_d.legend(loc="lower right", borderaxespad=0.3, handlelength=2.0, labelspacing=0.3)
    ins = ax_d.inset_axes([0.15, 0.60, 0.35, 0.34])
    m = t > 1.0
    ins.plot(t[m], 100 * (nn[m] - ensk[m]) / ensk[m], color=C_NEUTRAL, lw=1.4)
    ins.plot(t[m], 100 * (nc[m] - ensk[m]) / ensk[m], color=C_INTEG["cv-phys"], lw=0.9)
    ins.axhline(0, color=C_DARK, lw=0.5, ls="--")
    ins.set_xlim(0, 18)
    ins.set_ylim(-1, 2.5)
    ins.set_xticks([0, 9, 18])
    ins.tick_params(labelsize=7.5, pad=1.5, length=2)
    ins.set_title("excess over Enskog (%)", fontsize=8, pad=2)
    ins.grid(True, lw=0.4)
    panel_label(ax_d, "(d)")

    png = savefig(fig, "fig_energy")
    nums = {}
    for k in order:
        e = E[k]
        nums[k] = {
            "run": e["run"], "integrator": e["integrator"],
            "drift_E_end_minus_E0": q(e["drift_kT"], "kT"), "drift_rel": q(e["drift_rel"], "1 (relative to |E0|)"),
            "drift_per_particle": q(e["drift_per_particle_kT"], "kT per particle"),
            "max_abs_dev": q(e["max_dev_kT"], "kT"), "max_abs_dev_rel": q(e["max_dev_rel"], "1 (relative to |E0|)"),
            "max_abs_dev_per_particle": q(e["max_dev_per_particle_kT"], "kT per particle"),
            "rms_dev_from_E0": q(e["rms_kT"], "kT"), "rms_about_mean": q(e["rms_about_mean_kT"], "kT"),
            "rms_about_mean_rel": q(e["rms_about_mean_rel"], "1 (json energy_rms_rel)"),
            "frame_to_frame_scatter_rms": q(e["frame_scatter_rms_kT"], "kT"),
            "drift_slope": q(e["slope_kT_per_tau"], "kT/τ"),
            "T_mean": q(e["T_mean"], "kT"), "T_mean_minus_1": q(e["T_mean"] - 1.0, "kT"), "T_std": q(e["T_std"], "kT"),
            "PE_mean": q(e["PE_mean_kT"], "kT"), "PE_std": q(e["PE_std_kT"], "kT"), "KE_std": q(e["KE_std_kT"], "kT"),
            "n_collisions_production": q(e["n_collisions"], "collisions in 18 τ"),
            "collision_rate": q(e["collision_rate_per_particle_per_tau"], "collisions per particle per τ"),
            "Z_virial": q(e["Z_virial"], "1 (collisional pressure PV/NkT)"), "Z_rel_dev_from_CS": q(e["Z_rel_dev"], "1"),
            "pairs_per_frame_mean": q(e["n_pairs_mean"], "+/− pairs with r < 1.2 σ"),
            "wall_seconds_sim": q(e["wall_seconds"], "s"),
        }
    nums["neutral"] = {"run": "traj/hs_neutral", "integrator": "EDMD (exact)",
                       "n_collisions_production": q(E["neutral"]["n_collisions"], "collisions in 18 τ"),
                       "collision_rate": q(E["neutral"]["collision_rate_per_particle_per_tau"], "collisions per particle per τ"),
                       "enskog_rate": q(GE, "collisions per particle per τ"),
                       "Z_virial": q(E["neutral"]["Z_virial"], "1"), "Z_CS": q(E["neutral"]["Z_CS"], "1"),
                       "Z_rel_dev_from_CS": q(E["neutral"]["Z_rel_dev"], "1"),
                       "KE_max_rel_dev": q(E["neutral"]["max_dev_rel"], "1 (KE constant to roundoff)")}
    nums["enskog_slope"] = q(0.5 * N * GE, "collisions per τ (N Γ_E / 2)")
    nums["excess_over_enskog_neutral"] = q(100 * (nn[-1] - ensk[-1]) / ensk[-1], "%")
    nums["excess_over_enskog_charged"] = q(100 * (nc[-1] - ensk[-1]) / ensk[-1], "%")
    nums["ratio_naive_to_cvphys_max_dev"] = q(E["naive"]["max_dev_kT"] / E["cv-phys"]["max_dev_kT"], "1")
    nums["ratio_cv_to_cvphys_max_dev"] = q(E["cv"]["max_dev_kT"] / E["cv-phys"]["max_dev_kT"], "1")
    nums["ratio_kdk_to_cvphys_max_dev"] = q(E["kdk-phys"]["max_dev_kT"] / E["cv-phys"]["max_dev_kT"], "1")
    nums["ratio_cv_to_cvphys_frame_scatter"] = q(E["cv"]["frame_scatter_rms_kT"] / E["cv-phys"]["frame_scatter_rms_kT"], "1")
    manifest.append(dict(
        file="fig_energy", width="full",
        caption=(f"Energy conservation and collision statistics of the production runs (N = 1000, η = 0.35, 18 τ NVE). "
                 f"(a) Total-energy deviation E(t) − E(0) for the four integrators of the charged system (20+/20−, l_B = 5 σ, Δt = 0.005 τ): "
                 f"the naive splitting drifts by {E['naive']['drift_kT']:+.2f} kT, literal collision Verlet (cv) by {E['cv']['drift_kT']:+.2f} kT, "
                 f"kdk-phys by {E['kdk-phys']['drift_kT']:+.3f} kT and the shipped cv-phys by {E['cv-phys']['drift_kT']:+.3f} kT "
                 f"({sci(E['cv-phys']['drift_rel'])} of |E₀|; inset without naive). "
                 f"(b) cv-phys: kinetic and potential energy fluctuate in antiphase with a standard deviation of {KE.std():.1f} kT each while their sum changes by {E['cv-phys']['max_dev_kT']:.3f} kT at most. "
                 f"(c) Instantaneous temperature of the cv-phys run, ⟨T⟩ = {Tm:.4f} ± {Ts:.4f} kT. "
                 f"(d) Cumulative hard-sphere collisions in the neutral and charged runs against the Enskog rate N Γ_E/2 with Γ_E = {GE:.2f} per particle per τ; "
                 f"the neutral run exceeds Enskog by {100 * (nn[-1] - ensk[-1]) / ensk[-1]:.1f} % and the charged run by {100 * (nc[-1] - ensk[-1]) / ensk[-1]:.1f} % (inset)."),
        numbers=nums))
    return png


# ---------------------------------------------------------------------------
# Figure 2: pairing
# ---------------------------------------------------------------------------
def fig_pairing():
    P = R["pairing"]
    t = PR["times"]
    npairs, fc, fl = PR["npairs"], PR["frac_contact"], PR["frac_loose"]
    fig = plt.figure(figsize=(FULL_W, 4.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.28, left=0.085, right=0.985, top=0.9, bottom=0.1)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # (a) pair count
    ax_a.step(t, npairs, where="mid", color=C_ION, lw=0.9, label="+/− contact pairs (r < 1.2 σ)")
    ax_a.axhline(npairs.mean(), color=C_DARK, ls="--", lw=0.9, label=f"mean {npairs.mean():.2f}")
    ax_a.set_xlim(0, 18)
    ax_a.set_ylim(0, 23)
    ax_a.set_yticks([0, 5, 10, 15, 20])
    ax_a.set_xlabel("t (τ)")
    ax_a.set_ylabel("number of +/− pairs")
    ax_a.legend(loc="upper left", borderaxespad=0.2, labelspacing=0.25,
                title=f"range {npairs.min()}–{npairs.max()} of 20 possible", title_fontsize=8)
    secax_movie(ax_a)
    panel_label(ax_a, "(a)")

    # (b) fractions
    # both baselines from ONE definition: 100 random 20+20 subsets of the neutral run (hard-core g(r) included);
    # the hard-core-only estimate 1 - (1 - V_shell/V)^20 = 0.567 for 2.5 sigma is kept in the numbers as a cross-check
    bl_loose, bl_contact = P["baseline_loose_neutral_subsets_mean"], P["baseline_contact_neutral_subsets_mean"]
    sem_loose, sem_contact = P["baseline_loose_neutral_subsets_std"] / 10.0, P["baseline_contact_neutral_subsets_std"] / 10.0
    ax_b.plot(t, fl, color=C_NEUTRAL, lw=0.9, label=f"opposite ion within 2.5 σ  ({fl.mean():.2f})")
    ax_b.plot(t, fc, color=C_ION, lw=0.9, label=f"in a contact pair, r < 1.2 σ  ({fc.mean():.2f})")
    ax_b.axhline(bl_loose, color=C_DARK, ls="--", lw=0.9, label=f"random ions, 2.5 σ  ({bl_loose:.3f})")
    ax_b.axhline(bl_contact, color=C_DARK, ls=":", lw=0.9, label=f"random ions, 1.2 σ  ({bl_contact:.3f})")
    ax_b.set_xlim(0, 18)
    ax_b.set_ylim(0, 1.6)
    ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xlabel("t (τ)")
    ax_b.set_ylabel("fraction of ions")
    ax_b.legend(loc="upper right", borderaxespad=0.2, handlelength=2.0, labelspacing=0.3)
    panel_label(ax_b, "(b)")

    # (c) cluster-size histogram
    sizes, frac = PR["sizes"], PR["ion_frac"]
    ax_c.bar(sizes, frac, width=0.7, color=C_ION, alpha=0.85, edgecolor="white", lw=0.8)
    for s_, f_ in zip(sizes, frac):
        ax_c.text(s_, f_ * 1.25, f"{100 * f_:.2g}%" if f_ < 0.01 else f"{100 * f_:.1f}%", ha="center", va="bottom", fontsize=8, color=C_TXT)
    ax_c.set_yscale("log")
    ax_c.set_ylim(8e-4, 2.0)
    ax_c.set_xlabel("cluster size (ions, 1.2 σ +/− bonds)")
    ax_c.set_ylabel("fraction of ions")
    ax_c.set_xticks(sizes)
    ax_c.text(0.98, 0.97, f"≥ 3 ions: {100 * P['frac_ions_in_clusters_ge3']:.1f} % of ions\nchains {P['chains_per_frame']:.2f} per frame\n"
              f"branched {P['branched_per_frame']:.3f} per frame", transform=ax_c.transAxes, ha="right", va="top", fontsize=8, color=C_TXT,
              linespacing=1.15)
    panel_label(ax_c, "(c)")

    # (d) lifetime survival function
    life = PR["life"]
    L_ = P["lifetimes"]
    xs = np.sort(life)
    surv = 1.0 - np.arange(len(xs)) / len(xs)   # P(lifetime >= x)
    ax_d.step(np.concatenate([[0], xs]), np.concatenate([[1.0], surv]), where="post", color=C_ION, lw=1.3,
              label=f"contact episodes (n = {len(life)})")
    xx = np.linspace(0, 2.6, 100)
    ax_d.plot(xx, np.exp(-xx / life.mean()), ":", color=C_DARK, lw=0.9, label="exponential with the same mean")
    for val in (L_["median_tau"], L_["mean_tau"], L_["p90_tau"]):
        ax_d.axvline(val, color=C_NEUTRAL, lw=0.7, ls="--")
    # the four labels are stacked at distinct heights to the right of the 90th-percentile line, above the survival curve
    x_lab = L_["p90_tau"] + 0.05
    for yy, txt in ((0.80, f"median  {L_['median_tau']:.2f} τ = {L_['median_tau'] * TAU_TO_S:.2f} s"),
                    (0.45, f"mean  {L_['mean_tau']:.2f} τ = {L_['mean_tau'] * TAU_TO_S:.2f} s"),
                    (0.25, f"90th pct.  {L_['p90_tau']:.2f} τ = {L_['p90_tau'] * TAU_TO_S:.2f} s"),
                    (0.142, f"longest  {L_['max_tau']:.2f} τ = {L_['max_s']:.1f} s")):
        ax_d.text(x_lab, yy, txt, fontsize=8, color=C_TXT, va="center", ha="left")
    ax_d.set_yscale("log")
    ax_d.set_ylim(2e-3, 1.3)
    ax_d.set_xlim(0, 2.6)
    ax_d.set_xlabel("contact-episode lifetime ℓ (τ)")
    ax_d.set_ylabel("P(lifetime ≥ ℓ)")
    ax_d.legend(loc="lower left", borderaxespad=0.3)
    secax_movie(ax_d)
    panel_label(ax_d, "(d)")

    png = savefig(fig, "fig_pairing")
    nums = {
        "pairs_per_frame_mean": q(P["pairs_per_frame_mean"], "+/− pairs (r < 1.2 σ)"),
        "pairs_per_frame_std": q(P["pairs_per_frame_std"], "pairs"),
        "pairs_per_frame_min": q(P["pairs_per_frame_min"], "pairs"), "pairs_per_frame_max": q(P["pairs_per_frame_max"], "pairs"),
        "pairs_first_half_mean": q(P["pairs_first_half"], "pairs"), "pairs_second_half_mean": q(P["pairs_second_half"], "pairs"),
        "pairs_match_simulator_diag": q(P["pairs_match_diag_frames"], "frames out of 721 with identical count"),
        "frac_ions_in_contact_pair_mean": q(P["frac_contact_mean"], "1"), "frac_ions_in_contact_pair_std": q(P["frac_contact_std"], "1"),
        "frac_ions_in_contact_pair_min": q(P["frac_contact_min"], "1"), "frac_ions_in_contact_pair_max": q(P["frac_contact_max"], "1"),
        "frac_ions_opposite_within_2p5_mean": q(P["frac_loose_mean"], "1"), "frac_ions_opposite_within_2p5_std": q(P["frac_loose_std"], "1"),
        "frac_ions_opposite_within_2p5_min": q(P["frac_loose_min"], "1"), "frac_ions_opposite_within_2p5_max": q(P["frac_loose_max"], "1"),
        "opposite_ions_within_2p5_per_ion": q(P["n_opposite_within_loose_per_ion"], "ions"),
        "mean_nearest_opposite_distance": q(P["mean_nearest_opposite_sigma"], "σ"),
        "baseline_random_2p5_ideal": q(P["baseline_loose_ideal"], "1 (1 − (1 − V_shell/V)^20, hard core only)"),
        "baseline_random_2p5_neutral_subsets": q(P["baseline_loose_neutral_subsets_mean"], "1 (100 random 20+20 subsets of the neutral run)"),
        "baseline_random_2p5_neutral_subsets_std": q(P["baseline_loose_neutral_subsets_std"], "1 (std over the 100 subsets)"),
        "baseline_random_2p5_neutral_subsets_sem": q(sem_loose, "1 (standard error of the subset mean, std/sqrt(100))"),
        "baseline_random_1p2_neutral_subsets": q(P["baseline_contact_neutral_subsets_mean"], "1 (100 random 20+20 subsets of the neutral run)"),
        "baseline_random_1p2_neutral_subsets_std": q(P["baseline_contact_neutral_subsets_std"], "1 (std over the 100 subsets)"),
        "baseline_random_1p2_neutral_subsets_sem": q(sem_contact, "1 (standard error of the subset mean)"),
        "baselines_plotted": "both dashed/dotted lines in (b) are the neutral-run-subset values (same definition for 2.5 σ and 1.2 σ)",
        "baseline_random_1p2_ideal": q(P["baseline_contact_ideal"], "1 (ideal gas, no g(r) enhancement)"),
        "baseline_pairs_per_frame_random": q(P["baseline_pairs_neutral_subsets_mean"], "pairs"),
        "same_sign_contacts_per_frame": q(P["same_sign_contacts_per_frame"], "++ or −− pairs with r < 1.2 σ"),
        "cluster_sizes": sizes.tolist(),
        "fraction_of_ions_by_cluster_size": [sig(v) for v in frac],
        "clusters_per_frame_by_size": [sig(v) for v in P["clusters_per_frame"]],
        "frac_ions_in_clusters_ge3": q(P["frac_ions_in_clusters_ge3"], "1"),
        "largest_cluster_mean": q(P["largest_cluster_mean"], "ions"), "largest_cluster_max": q(P["largest_cluster_max"], "ions"),
        "chains_ge3_per_frame": q(P["chains_per_frame"], "clusters (max degree 2)"),
        "branched_ge3_per_frame": q(P["branched_per_frame"], "clusters"),
        "frames_with_branched_cluster": q(P["frames_with_branched_frac"], "1"),
        "n_contact_episodes": q(L_["n"], "episodes in 18 τ (incl. 17 truncated at the run ends)"),
        "episodes_per_tau": q(P["episodes_per_tau"], "1/τ"),
        "lifetime_mean": q(L_["mean_tau"], "τ"), "lifetime_median": q(L_["median_tau"], "τ"),
        "lifetime_p90": q(L_["p90_tau"], "τ"), "lifetime_max": q(L_["max_tau"], "τ"),
        "lifetime_mean_movie": q(L_["mean_s"], "s at 30 fps"), "lifetime_median_movie": q(L_["median_s"], "s at 30 fps"),
        "lifetime_p90_movie": q(L_["p90_s"], "s at 30 fps"), "lifetime_max_movie": q(L_["max_s"], "s at 30 fps"),
        "lifetime_mean_uncensored": q(P["lifetimes_uncensored"]["mean_tau"], "τ (270 episodes not touching the run ends)"),
        "lifetime_resolution": q(0.025, "τ (one frame; lifetime = n_frames × Δt_frame)"),
        "distinct_partners_per_plus_ion_mean": q(P["partners_per_plus_mean"], "− partners over 18 τ"),
        "distinct_partners_per_plus_ion_min": q(P["partners_per_plus_min"], "partners"),
        "distinct_partners_per_plus_ion_max": q(P["partners_per_plus_max"], "partners"),
        "distinct_partners_per_minus_ion_mean": q(P["partners_per_minus_mean"], "+ partners over 18 τ"),
        "distinct_partners_per_minus_ion_min": q(P["partners_per_minus_min"], "partners"),
        "distinct_pm_pairs_ever_in_contact": q(P["n_distinct_pairs_ever"], "of 400 possible"),
        "movie_seconds_per_tau": q(TAU_TO_S, "s/τ"),
    }
    manifest.append(dict(
        file="fig_pairing", width="full",
        caption=(f"Bjerrum pairing in the charged production run (20+/20−, l_B = 5 σ, cv-phys, 721 frames). "
                 f"(a) Number of +/− contact pairs (r < 1.2 σ) per frame, mean {P['pairs_per_frame_mean']:.1f} (range {P['pairs_per_frame_min']}–{P['pairs_per_frame_max']}). "
                 f"(b) Fraction of ions in a contact pair ({100 * P['frac_contact_mean']:.0f} ± {100 * P['frac_contact_std']:.0f} %) and fraction with an opposite ion within l_B/2 = 2.5 σ "
                 f"({100 * P['frac_loose_mean']:.0f} %); dashed and dotted lines are the random-ion baselines obtained from 100 random 20+20 subsets of the neutral run, "
                 f"{bl_loose:.3f} ± {sem_loose:.3f} (2.5 σ) and {bl_contact:.3f} ± {sem_contact:.3f} (1.2 σ; standard errors of the subset mean); "
                 f"the hard-core-only estimate 1 − (1 − V_shell/V)^20 = {P['baseline_loose_ideal']:.3f} for 2.5 σ is a consistent cross-check. "
                 f"(c) Fraction of ions in connected +/− clusters of size 1–6: {100 * P['frac_ions_in_clusters_ge3']:.0f} % sit in clusters of three or more, almost all linear chains. "
                 f"(d) Survival function of contact-episode lifetimes: median {L_['median_tau']:.2f} τ, mean {L_['mean_tau']:.2f} τ, 90th percentile {L_['p90_tau']:.2f} τ, "
                 f"longest {L_['max_tau']:.2f} τ ({L_['max_s']:.1f} s of movie); the dotted curve is an exponential with the same mean."),
        numbers=nums))
    return png


# ---------------------------------------------------------------------------
# Figure 3: structure
# ---------------------------------------------------------------------------
def fig_structure():
    S = R["structure"]
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(FULL_W, 2.7), gridspec_kw=dict(wspace=0.36, left=0.07, right=0.99, top=0.93, bottom=0.17))
    rcf, g_nn = GR["rc_fine"], GR["g_nn"]
    # (a)
    g_nn_plot = np.where(rcf >= 1.0, g_nn, np.nan)          # mask the hard core: the curve starts at contact, no drop to 0
    ax_a.plot(rcf, g_nn_plot, color=C_NEUTRAL, lw=1.2, label="neutral run, 0.01 σ bins")
    ax_a.plot([1.0], [S["g_contact_CS"]], "o", ms=5.5, color=C_DARK, mfc="white", mew=1.2, label="Carnahan–Starling g(σ⁺)", zorder=5)
    ax_a.axhline(1.0, color=C_DARK, lw=0.5, ls=":")
    ax_a.set_xlim(0.9, 4.0)
    ax_a.set_ylim(0, 3.4)
    ax_a.set_xlabel("r (σ)")
    ax_a.set_ylabel("g(r)")
    ax_a.legend(loc="lower right", borderaxespad=0.3, handlelength=1.2)
    # the extrapolated contact value is quoted as text (a second marker at r = 1 would sit on top of the CS circle)
    ax_a.text(0.97, 0.62, f"g(σ⁺) = {S['g_contact_CS']:.2f}  (CS)\n{S['g_contact_neutral_fit2']:.2f}  extrapolated\n"
              f"{S['g_contact_from_Z_virial_neutral']:.2f}  from Z = {R['energy']['neutral']['Z_virial']:.3f}",
              transform=ax_a.transAxes, fontsize=8, ha="right", va="top", color=C_TXT, linespacing=1.15)
    panel_label(ax_a, "(a)")

    # (b) ion correlations, log y
    rc = GR["rc"]
    core = rc < 1.0                                            # hard core: g = 0 bins are masked (no spikes on the log axis)

    def nancore(a):
        a = np.array(a, float)
        a[core] = np.nan
        return a
    ax_b.errorbar(rc, nancore(GR["g_pm"]), yerr=nancore(GR["e_pm"]), fmt="-", color=C_ION, lw=1.2, elinewidth=0.6, capsize=0, label="g₊₋ (+/−)")
    ax_b.errorbar(rc, nancore(GR["g_pp"]), yerr=nancore(GR["e_pp"]), fmt=":", color=C_SAME, lw=1.3, elinewidth=0.6, capsize=0, label="g₊₊, g₋₋")
    ax_b.plot(rc, nancore(GR["g_in"]), "-", color=C_NEUTRAL, lw=1.6, label="ion–neutral")
    ax_b.plot(rc, nancore(GR["g_hs"]), "--", color=C_DARK, lw=0.8, label="g_HS (neutral)")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.9, 4.5)
    ax_b.set_ylim(0.06, 40)
    ax_b.set_xlabel("r (σ)")
    ax_b.set_ylabel("g(r)  (charged run, 0.05 σ bins)")
    ax_b.legend(loc="lower right", borderaxespad=0.2, handlelength=1.6, labelspacing=0.3)   # below the same-sign curve (g > 0.74 at r > 2.1)
    ax_b.text(1.13, 15.0, f"{S['g_pm_firstbin']:.1f} ± {S['g_pm_firstbin_err']:.1f}", fontsize=8, color=C_TXT, va="center")
    panel_label(ax_b, "(b)")

    # (c) PMF
    w, we = GR["w"], GR["w_err"]
    m = np.isfinite(w) & (rc >= 1.0) & (rc <= 4.0)
    ax_c.errorbar(rc[m], w[m], yerr=we[m], fmt="o-", ms=2.5, color=C_ION, lw=1.0, elinewidth=0.6, capsize=0,
                  label="−ln[g₊₋/g_HS]")
    rr = np.linspace(1.0, 4.0, 200)
    lB, kap = S["l_B_sigma"], S["kappa_sigma"]
    ax_c.plot(rr, -lB / rr, "--", color=C_DARK, lw=0.9, label="−l_B/r  (bare)")
    ax_c.plot(rr, -lB * np.exp(-kap * (rr - 1)) / (rr * (1 + kap)), "-.", color=C_NEUTRAL, lw=1.1, label="Debye–Hückel")
    ax_c.axhline(0, color=C_DARK, lw=0.5, ls=":")
    ax_c.set_xlim(0.9, 4.0)
    ax_c.set_ylim(-5.3, 0.6)
    ax_c.set_xlabel("r (σ)")
    ax_c.set_ylabel("w₊₋(r)  (kT)")
    ax_c.legend(loc="lower right", borderaxespad=0.2, handlelength=1.6, labelspacing=0.3)
    # contact value as a text block in the empty region below the bare-Coulomb curve (no arrow: it would cross that curve)
    ax_c.text(2.35, -2.3, f"w₊₋ contact bin\n{S['w_contact_firstbin_kT']:.2f} ± {S['w_contact_firstbin_err_kT']:.2f} kT\n(σ⁺: {S['w_contact_extrapolated_kT']:.2f} kT)",
              fontsize=8, color=C_TXT, va="top", ha="left", linespacing=1.1)
    panel_label(ax_c, "(c)")

    png = savefig(fig, "fig_structure")
    # same-sign g(r): where it first reaches the bulk value, its plateau and the weak second depletion
    gpp = np.asarray(GR["g_pp"], float)
    i1 = int(np.argmax((rc > 1.0) & (gpp >= 0.98)))
    ss_first_r, ss_first_g = float(rc[i1]), float(gpp[i1])
    mp = (rc > 1.95) & (rc < 2.25)
    ss_plateau = float(gpp[mp].mean())
    md = (rc > 2.3) & (rc < 2.75)
    ss_dip_mean = float(gpp[md].mean())
    ss_dip_min_r, ss_dip_min_g = float(rc[md][np.argmin(gpp[md])]), float(gpp[md].min())
    # occupancy of the 1.00-1.05 sigma bin around an ion
    v_bin = 4.0 * np.pi / 3.0 * (1.05 ** 3 - 1.0 ** 3)
    L3 = R["system"]["L"] ** 3
    ci_bin = S["g_pm_firstbin"] * (20.0 / L3) * v_bin
    nn_bin = S["g_ion_neutral_firstbin"] * (960.0 / L3) * v_bin
    nn_bin_ref = S["g_nn_charged_firstbin"] * (960.0 / L3) * v_bin
    nums = {
        "eta": q(S["eta"], "1"), "rho": q(S["rho"], "σ⁻³"), "frames_averaged_neutral": q(S["frames_neutral"], "frames"),
        "frames_averaged_charged": q(S["frames_charged"], "frames"),
        "bin_width_neutral": q(0.01, "σ"), "bin_width_charged": q(0.05, "σ"),
        "g_contact_CS": q(S["g_contact_CS"], "1 ((1−η/2)/(1−η)³)"),
        "g_contact_extrapolated_quadratic_fit": q(S["g_contact_neutral_fit2"], "1 (quadratic fit of g over 1.0–1.1 σ)"),
        "g_contact_extrapolated_linear_fit": q(S["g_contact_neutral_fit1"], "1 (linear fit over 1.0–1.1 σ)"),
        "g_first_bin_1p00_1p01": q(S["g_contact_neutral_firstbin"], "1"),
        "g_contact_from_Z_virial_neutral": q(S["g_contact_from_Z_virial_neutral"], "1 ((Z−1)/4η with Z = 5.233)"),
        "g_contact_from_Z_virial_charged": q(S["g_contact_from_Z_virial_charged"], "1 (Z = 5.224)"),
        "g_nn_first_minimum": {"r": q(S["g_nn_first_min_r"], "σ"), "g": q(S["g_nn_first_min_value"], "1")},
        "g_nn_second_peak": {"r": q(S["g_nn_second_peak_r"], "σ"), "g": q(S["g_nn_second_peak_value"], "1")},
        "g_pm_contact_bin_1p00_1p05": q(S["g_pm_firstbin"], "1"), "g_pm_contact_bin_err": q(S["g_pm_firstbin_err"], "1 (Poisson)"),
        "g_pm_contact_bin_counts": q(S["g_pm_counts_firstbin"], "pair samples"),
        "g_pm_contact_extrapolated": q(S["g_pm_contact_fit"], "1 (exp. fit of ln g over 1.0–1.15 σ, 0.01 σ bins)"),
        "g_pm_decay_length_near_contact": q(S["g_pm_fit_decay_length_sigma"], "σ"),
        "g_pm_second_peak": {"r": q(S["g_pm_second_peak_r"], "σ"), "g": q(S["g_pm_second_peak_value"], "1")},
        "g_samesign_contact_bin": q(S["g_samesign_firstbin"], "1"), "g_samesign_contact_bin_err": q(S["g_samesign_firstbin_err"], "1"),
        "g_samesign_contact_bin_counts": q(S["g_samesign_counts_firstbin"], "pair samples"),
        "g_samesign_first_bin_at_bulk": {"r": q(ss_first_r, "σ"), "g": q(ss_first_g, "1 (first bin with g ≥ 0.98)")},
        "g_samesign_plateau_1p95_2p25": q(ss_plateau, "1 (mean over 1.95–2.25 σ; ± 0.04 Poisson per bin)"),
        "g_samesign_dip_2p3_2p75_mean": q(ss_dip_mean, "1 (mean over 2.3–2.75 σ)"),
        "g_samesign_dip_2p3_2p75_min": {"r": q(ss_dip_min_r, "σ"), "g": q(ss_dip_min_g, "1")},
        "g_samesign_max_1p5_3p0": {"r": q(S["g_samesign_peak_r"], "σ"), "g": q(S["g_samesign_peak_value"], "1 (no excess over the bulk value within the Poisson error; NOT a peak)")},
        "g_ion_neutral_contact_deficit": q(100.0 * (1.0 - S["g_ion_neutral_firstbin"] / S["g_nn_charged_firstbin"]), "% below g_nn of the charged run in the 1.00–1.05 σ bin"),
        "counter_ions_per_ion_in_contact_bin": q(ci_bin, "ions (g₊₋ ρ_opp V_bin, 1.00–1.05 σ)"),
        "neutrals_per_ion_in_contact_bin": q(nn_bin, "spheres (g_ion-neutral ρ_n V_bin)"),
        "neutrals_per_neutral_in_contact_bin": q(nn_bin_ref, "spheres (g_nn ρ_n V_bin, charged run)"),
        "g_ion_neutral_contact_bin": q(S["g_ion_neutral_firstbin"], "1"),
        "g_nn_charged_run_contact_bin": q(S["g_nn_charged_firstbin"], "1"), "g_HS_neutral_run_contact_bin_0p05": q(S["g_hs_firstbin_coarse"], "1"),
        "g_large_r_mean_4p5_5p7": {"pm": q(S["g_large_r_mean_pm"], "1"), "nn": q(S["g_large_r_mean_nn"], "1"),
                                   "ion_neutral": q(S["g_large_r_mean_in"], "1"), "samesign": q(S["g_large_r_mean_samesign"], "1")},
        "w_contact_bin": q(S["w_contact_firstbin_kT"], "kT (1.00–1.05 σ)"), "w_contact_bin_err": q(S["w_contact_firstbin_err_kT"], "kT"),
        "w_contact_extrapolated": q(S["w_contact_extrapolated_kT"], "kT (−ln[g_pm(σ⁺)/g_HS(σ⁺)] from the two extrapolations)"),
        "w_vs_r_first_bins": [[sig(a), sig(b)] for a, b in S["w_at_r"]],
        "bare_coulomb_contact": q(S["bare_coulomb_contact_kT"], "kT"), "debye_hueckel_contact": q(S["debye_hueckel_contact_kT"], "kT"),
        "kappa_sigma": q(S["kappa_sigma"], "1 (κ² = 4π l_B (ρ₊+ρ₋))"), "debye_length": q(S["debye_length_sigma"], "σ"),
        "l_B": q(S["l_B_sigma"], "σ"), "rho_ion_total": q(S["rho_ion"], "σ⁻³"), "mean_ion_spacing": q(S["mean_ion_spacing_sigma"], "σ"),
        "coupling_l_B_over_sigma": q(S["l_B_sigma"], "1 (= 1/T*; DH linearisation requires ≪ 1)"),
    }
    manifest.append(dict(
        file="fig_structure", width="full",
        caption=(f"Static structure. (a) Neutral–neutral pair distribution of the hard-sphere run (η = 0.35, 0.01 σ bins, 721 frames, minimum image): "
                 f"the contact value extrapolated from 1.0–1.1 σ is g(σ⁺) = {S['g_contact_neutral_fit2']:.2f}, against the Carnahan–Starling value {S['g_contact_CS']:.2f} "
                 f"and {S['g_contact_from_Z_virial_neutral']:.2f} implied by the collisional virial Z = {R['energy']['neutral']['Z_virial']:.3f}. "
                 f"(b) Ion correlations in the charged run (0.05 σ bins, all 721 frames; error bars are Poisson): g₊₋ reaches {S['g_pm_firstbin']:.1f} ± {S['g_pm_firstbin_err']:.1f} in the contact bin "
                 f"(≈{S['g_pm_contact_fit']:.0f} extrapolated to σ⁺); same-sign pairs are depleted at contact (g ≈ {S['g_samesign_firstbin']:.1f}), recover the bulk value "
                 f"by ≈{ss_first_r:.0f} σ (the + − + chain distance; {ss_plateau:.2f} ± 0.04 over 1.95–2.25 σ) and show a weak second depletion "
                 f"(g ≈ {ss_dip_mean:.2f}) at 2.3–2.75 σ; ion–neutral correlations follow the neutral fluid beyond the first shell, while in the contact bin "
                 f"g_ion–neutral = {S['g_ion_neutral_firstbin']:.2f} against {S['g_nn_charged_firstbin']:.2f} for neutral pairs, a {100.0 * (1.0 - S['g_ion_neutral_firstbin'] / S['g_nn_charged_firstbin']):.1f} % deficit "
                 f"corresponding to the contact sites occupied by counter-ion partners ({ci_bin:.2f} per ion in that bin). "
                 f"(c) Potential of mean force w₊₋ = −ln[g₊₋/g_HS] compared with the bare Coulomb pair energy −l_B/r and the linearised Debye–Hückel screened potential "
                 f"(κσ = {S['kappa_sigma']:.2f}): the effective attraction at contact is {S['w_contact_firstbin_kT']:.2f} ± {S['w_contact_firstbin_err_kT']:.2f} kT, "
                 f"far weaker than the bare −5 kT, because the other 38 ions screen the pair; the DH curve is shown only as a scale since l_B/σ = 5 is outside its range of validity."),
        numbers=nums))
    return png


# ---------------------------------------------------------------------------
# Figure 4: dynamics
# ---------------------------------------------------------------------------
def fig_dynamics():
    D = R["dynamics"]
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(FULL_W, 2.7), gridspec_kw=dict(wspace=0.33, left=0.065, right=0.995, top=0.93, bottom=0.17))
    tl = DY["tl"]
    m = tl <= 9.0
    Dn, Dcn, Dci = (D["D"][k]["D_fit_slope"] for k in ("neutral_run_neutrals", "charged_run_neutrals", "charged_run_ions"))
    # neutral spheres of both runs in C_NEUTRAL (solid: neutral run, dashed: charged run; the curves overlap), ions in C_ION
    ax_a.loglog(tl[m], DY["msd_n"][m], "-", color=C_NEUTRAL, lw=1.8, label=f"neutral run  {Dn:.3f}")
    ax_a.loglog(tl[m], DY["msd_cn"][m], "--", color=C_NEUTRAL, lw=1.1, label=f"charged, neutrals  {Dcn:.3f}")
    ax_a.loglog(tl[m], DY["msd_ci"][m], "-", color=C_ION, lw=1.1, label=f"charged, ions  {Dci:.3f}")
    ax_a.loglog(tl[m], 6 * D["D_enskog_CS"] * tl[m], ":", color=C_DARK, lw=0.9, label=f"Enskog 6 D_E t  {D['D_enskog_CS']:.3f}")
    # guides
    tg = np.array([0.025, 0.12])
    ax_a.loglog(tg, 3 * tg ** 2 * 0.55, "-", color=C_DARK, lw=0.6)
    ax_a.text(0.06, 3 * 0.06 ** 2 * 0.3, "∝ t²", fontsize=8, color=C_TXT, ha="center", va="top")
    tg = np.array([2.0, 9.0])
    ax_a.loglog(tg, 6 * Dn * tg * 0.45, "-", color=C_DARK, lw=0.6)
    ax_a.text(4.3, 6 * Dn * 4.3 * 0.30, "∝ t", fontsize=8, color=C_TXT, ha="center", va="top")
    ax_a.set_xlim(0.02, 10)
    ax_a.set_ylim(1e-3, 500)       # head room for the legend (the data end at 6.5 σ² at 9 τ)
    ax_a.set_xlabel("lag time t (τ)")
    ax_a.set_ylabel("MSD  ⟨|Δr|²⟩  (σ²)")
    ax_a.legend(loc="upper left", borderaxespad=0.2, handlelength=1.8, labelspacing=0.3, title="D (σ²/τ), slope over 1–4 τ:",
                title_fontsize=8, alignment="left")
    panel_label(ax_a, "(a)")

    # (b) displacement histogram
    dcen, hn, hci, pmax = DY["dcen"], DY["hn"], DY["hci"], DY["pmax"]
    wbin = dcen[1] - dcen[0]
    ax_b.bar(dcen, hn, width=wbin, color=C_NEUTRAL, alpha=0.75, label="neutrals")
    ax_b.step(np.concatenate([dcen - wbin / 2, [dcen[-1] + wbin / 2]]), np.concatenate([hci, [hci[-1]]]), where="post", color=C_ION, lw=1.3,
              label="ions")
    ax_b.plot(dcen, pmax, "--", color=C_DARK, lw=0.9, label="Maxwell free flight")
    ax_b.set_xlim(0, 0.15)
    ax_b.set_ylim(0, 1.55 * max(hn.max(), pmax.max()))     # head room: legend and text clear the peak (24 σ⁻¹ at 0.035–0.045 σ)
    ax_b.set_xlabel("displacement per frame |Δr| (σ)")
    ax_b.set_ylabel("probability density (σ⁻¹)")
    ax_b.legend(loc="upper right", borderaxespad=0.2, handlelength=1.6)
    ax_b.text(0.98, 0.74, f"mean {D['disp_per_frame']['neutral_run_mean']:.3f} σ (neutrals)\nmean {D['disp_per_frame']['charged_ions_mean']:.3f} σ (ions)",
              transform=ax_b.transAxes, fontsize=8, ha="right", va="top", color=C_TXT, linespacing=1.15) if False else None
    ax_b.text(0.98, 0.71, f"mean {D['disp_per_frame']['neutral_run_mean']:.3f} σ (neutrals)\nmean {D['disp_per_frame']['charged_ions_mean']:.3f} σ (ions)",
              transform=ax_b.transAxes, fontsize=8, ha="right", va="top", color=C_TXT, linespacing=1.15)
    panel_label(ax_b, "(b)")

    # (c) free flight
    F = D["freeflight"]
    tt = DY["tt"]
    ax_c.plot(tt, DY["ff_n"], color=C_NEUTRAL, lw=0.7, label="per frame, neutral run")
    run = np.convolve(DY["ff_n"], np.ones(40) / 40, mode="valid")
    ax_c.plot(tt[20:20 + len(run)], run, color=C_DARK, lw=1.3, label="1 τ running mean")
    ax_c.axhline(F["expected_measured_rate_neutral"], color=C_ACCENT, ls="--", lw=1.1, label="exp(−2 Γ Δt_frame)")
    ax_c.set_xlim(0, 18)
    ax_c.set_ylim(0.35, 0.65)
    ax_c.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])
    ax_c.set_xlabel("t (τ)")
    ax_c.set_ylabel("free-flight fraction of neutrals")
    ax_c.text(0.98, 0.03, f"mean {F['frac_neutral_run_mean']:.3f} (neutral run)\n{F['frac_charged_neutrals_mean']:.3f} (charged-run neutrals)\n"
              f"expected {F['expected_measured_rate_neutral']:.3f},  Γ = {F['Gamma_neutral']:.2f} /τ",
              transform=ax_c.transAxes, fontsize=8, ha="right", va="bottom", color=C_TXT, linespacing=1.15)
    ax_c.legend(loc="upper left", borderaxespad=0.2, handlelength=1.8, labelspacing=0.3)
    panel_label(ax_c, "(c)")

    png = savefig(fig, "fig_dynamics")
    dd = D["D"]
    nums = {
        "D_fit_slope_1_4tau": {k: q(dd[k]["D_fit_slope"], "σ²/τ (slope of MSD vs t over 1–4 τ, /6)") for k in dd},
        "D_msd_over_6t_mean_1_4tau": {k: q(dd[k]["D_msd_over_6t"], "σ²/τ (mean of MSD/6t over 1–4 τ)") for k in dd},
        "MSD_at_1tau": {k: q(dd[k]["msd_at_1tau"], "σ²") for k in dd},
        "MSD_at_4tau": {k: q(dd[k]["msd_at_4tau"], "σ²") for k in dd},
        "D_enskog_CS": q(D["D_enskog_CS"], "σ²/τ (3/(8ρσ²g_CS(σ)) (kT/πm)^1/2)"),
        "D_over_D_enskog_neutral": q(D["D_ratio_to_enskog_neutral"], "1"),
        "rms_displacement_18tau_neutral_run": q(D["rms_disp_18tau_neutral_run"], "σ (single origin t=0→18 τ, 1000 spheres)"),
        "rms_displacement_18tau_charged_neutrals": q(D["rms_disp_18tau_charged_neutrals"], "σ (960 spheres)"),
        "rms_displacement_18tau_charged_ions": q(D["rms_disp_18tau_charged_ions"], "σ (40 ions)"),
        "mean_free_time": q(D["mean_free_time_tau"], "τ (1/Γ)"), "mean_free_path": q(D["mean_free_path_sigma"], "σ (⟨v⟩/Γ)"),
        "disp_per_frame_neutral_mean": q(D["disp_per_frame"]["neutral_run_mean"], "σ"),
        "disp_per_frame_neutral_rms": q(D["disp_per_frame"]["neutral_run_rms"], "σ"),
        "disp_per_frame_neutral_max": q(D["disp_per_frame"]["neutral_run_max"], "σ"),
        "disp_per_frame_ion_mean": q(D["disp_per_frame"]["charged_ions_mean"], "σ"),
        "disp_per_frame_ion_rms": q(D["disp_per_frame"]["charged_ions_rms"], "σ"),
        "disp_per_frame_ion_max": q(D["disp_per_frame"]["charged_ions_max"], "σ"),
        "disp_per_frame_maxwell_mean": q(D["disp_per_frame"]["maxwell_mean"], "σ ((8kT/πm)^1/2 Δt_frame, no collisions)"),
        "freeflight_fraction_neutral_run": q(F["frac_neutral_run_mean"], "1 (mean over 719 frames)"),
        "freeflight_fraction_neutral_run_frame_std": q(F["frac_neutral_run_std"], "1"),
        "freeflight_fraction_charged_neutrals": q(F["frac_charged_neutrals_mean"], "1"),
        "freeflight_fraction_charged_ions": q(F["frac_charged_ions_mean"], "1 (ions are accelerated: never ballistic)"),
        "freeflight_expected_measured_Gamma": q(F["expected_measured_rate_neutral"], "1 (exp(−2 Γ Δt_frame), Γ = 14.32/τ)"),
        "freeflight_expected_enskog_Gamma": q(F["expected_enskog"], "1 (Γ_E = 14.24/τ)"),
        "freeflight_measured_over_expected": q(F["ratio_measured_over_expected_neutral"], "1"),
        "freeflight_tolerance": q(F["tol_sigma"], "σ"),
        "freeflight_float32_roundoff_max": q(F["roundoff_cluster_max"], "σ (largest |Δr_k − Δr_k+1| in the roundoff cluster)"),
        "freeflight_frac_between_1e5_and_1e3": q(F["frac_between_1e5_and_1e3"], "1 (collisions within ~1e-3/|Δv| of a frame time)"),
        "Gamma_neutral": q(F["Gamma_neutral"], "collisions per particle per τ"), "Gamma_enskog": q(F["Gamma_enskog"], "collisions per particle per τ"),
    }
    manifest.append(dict(
        file="fig_dynamics", width="full",
        caption=(f"Single-particle dynamics (trajectories unwrapped through the periodic box). (a) Mean-squared displacement versus lag time for neutral spheres in the neutral run, "
                 f"neutral spheres in the charged run and the 40 ions, with t² (ballistic) and t (diffusive) guides and the Enskog prediction; "
                 f"the slopes over 1–4 τ give D = {Dn:.3f}, {Dcn:.3f} and {Dci:.3f} σ²/τ ({D['D_ratio_to_enskog_neutral']:.2f} × Enskog), i.e. ions and neutrals diffuse alike. "
                 f"(b) Distribution of the displacement per frame (Δt_frame = 0.025 τ): mean {D['disp_per_frame']['neutral_run_mean']:.3f} σ for neutrals, "
                 f"{D['disp_per_frame']['charged_ions_mean']:.3f} σ for ions, maximum {D['disp_per_frame']['charged_ions_max']:.2f} σ; the dashed curve is the Maxwell speed distribution times Δt_frame. "
                 f"(c) Free-flight check: fraction of neutral spheres whose displacement vector is identical over two consecutive frame intervals, "
                 f"{F['frac_neutral_run_mean']:.3f} on average against exp(−2ΓΔt_frame) = {F['expected_measured_rate_neutral']:.3f} for the measured collision rate Γ = {F['Gamma_neutral']:.2f} τ⁻¹."),
        numbers=nums))
    return png


def main():
    pngs = [fig_energy(), fig_pairing(), fig_structure(), fig_dynamics()]
    sysinfo = R["system"]
    out = dict(
        generated_by="report/physics_analysis.py + report/physics_figures.py",
        system={"N": 1000, "eta": 0.35, "L_sigma": sig(sysinfo["L"], 6), "kT": 1.0, "frames": 721, "dt_frame_tau": 0.025, "t_run_tau": 18.0,
                "ions": "20 (+e) / 20 (−e), l_B = 5 σ, Ewald", "movie_fps": 30, "movie_seconds_per_tau": sig(TAU_TO_S)},
        runs={"neutral": "traj/hs_neutral (EDMD)", "charged cv-phys": "traj/hs_charges_20p20m",
              "charged cv": "traj/_test/cv_confirmer_prod/charged_cv (reused, integrator_id cv, default production parameters)",
              "charged kdk-phys": "traj/_test/cv_confirmer_prod/charged_kdk (reused, integrator_id kdk-phys)",
              "charged naive": "traj/_test/report/charged_naive (new run)"},
        data_files=["physics_numbers_raw.json", "energy_timeseries.csv", "pairing_timeseries.csv", "cluster_hist.csv",
                    "contact_episodes.csv", "lifetime_survival.csv", "gr_neutral.csv", "gr_charged.csv", "pmf_pm.csv",
                    "msd.csv", "displacement_hist.csv", "freeflight.csv"],
        notes=[
            "All statistics from the float32 npz positions (721 frames); pair counts per frame agree with the simulator's own double-precision n_pairs_pm_1p2 column in all 721 frames.",
            "Energy panels use the per-frame _diag.csv Etot of each integrator run; cv and kdk-phys reuse traj/_test/cv_confirmer_prod/charged_{cv,kdk} (same command-line defaults as production, integrator_id checked); naive was run for this report.",
            "fig_energy (b) uses two stacked sub-panels instead of twin axes.",
            "Contact-episode lifetime = number of consecutive frames with r < 1.2 sigma times dt_frame (resolution 0.025 tau); 17 of 287 episodes touch the first or last frame and are included uncensored (excluding them changes the mean by 0.005 tau).",
            "g(r) normalised per pair type with the exact shell volume and minimum-image distances; g_HS in the PMF is the neutral run rebinned to 0.05 sigma. The PMF is a finite-N (40 ions) canonical quantity: g_pm averages 0.987 at r > 4.5 sigma.",
            "D is the slope of MSD(t) over 1-4 tau divided by 6; MSD/(6t) averaged over the same window is 10 % lower because MSD has a negative intercept after the ballistic regime.",
            "Free-flight criterion evaluated on npz float32 data: roundoff cluster ends at 2.9e-6 sigma, the 1e-5 threshold lies in a clean gap (float64 xyz on frames 0-11 gives the identical fraction 0.4969).",
            "Floats are quoted to 3 significant digits; integer counts are exact.",
            "Colour semantics: +/- pair and ion quantities are drawn in a dedicated magenta (#A3268F) that is not part of the integrator palette; same-sign ion pairs are dark gray dotted; neutral spheres are C_NEUTRAL gray in both runs (solid = neutral run, dashed = charged run). The integrator colours appear only in fig_energy.",
            "fig_pairing (b): both random baselines are the neutral-run-subset values (same definition); the hard-core-only estimate 0.567 for 2.5 sigma is kept in the numbers as a cross-check.",
            "All in-figure text is >= 8 pt (inset tick labels 7.5 pt), figstyle defaults, no rcParams override.",
        ],
        figures=manifest)
    for fg in manifest:
        fg["caption"] = mn(fg["caption"])
    with open(os.path.join(DATADIR, "physics_manifest.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for p in pngs:
        print(p)


if __name__ == "__main__":
    main()
