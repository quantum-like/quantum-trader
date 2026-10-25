#!/usr/bin/env python3
"""Physics statistics for the report figures (fig_energy, fig_pairing, fig_structure, fig_dynamics).

Reads the production trajectories traj/hs_neutral.npz and traj/hs_charges_20p20m.npz (float32 positions,
721 frames, dt_frame = 0.025 tau), the per-frame diagnostics of the four integrator runs and the json
summaries, and writes every derived quantity to report/data/*.csv|json.  Heavy per-particle intermediates
(unwrapped trajectories, MSD, displacement samples) are cached in CACHE_DIR so that physics_figures.py
can be iterated quickly.

Run from anywhere inside the venv:  python report/physics_analysis.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "report", "data")
CACHE_DIR = os.environ.get("PHYS_CACHE",
                           "/tmp/claude-0/-home-user-quantum-trader/51ecfd3f-2c9f-5b9b-ba8a-5c6a2ddea84a/scratchpad/phys_cache")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RUNS = {
    "neutral": os.path.join(ROOT, "traj", "hs_neutral"),
    "charged": os.path.join(ROOT, "traj", "hs_charges_20p20m"),
}
# the four integrators on the same charged system (20+/20-, l_B = 5, dt = 0.005, 5+15+18 tau, seed 1)
INTEG_RUNS = {
    "cv-phys": os.path.join(ROOT, "traj", "hs_charges_20p20m"),
    "cv": os.path.join(ROOT, "traj", "_test", "cv_confirmer_prod", "charged_cv"),
    "kdk-phys": os.path.join(ROOT, "traj", "_test", "cv_confirmer_prod", "charged_kdk"),
    "naive": os.path.join(ROOT, "traj", "_test", "report", "charged_naive"),
}
BOND = 1.2          # sigma, +/- contact criterion
LOOSE = 2.5         # sigma, l_B/2
FPS = 30.0
FREEFLIGHT_TOL = 1e-5


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_npz(base):
    d = np.load(base + ".npz")
    meta = json.loads(str(d["meta"]))
    return dict(pos=d["pos"], types=d["types"], charge=d["charge"], times=d["times"], L=float(d["L"]), meta=meta)


def load_diag(base):
    arr = np.genfromtxt(base + "_diag.csv", delimiter=",", names=True)
    return arr


def load_json(base):
    with open(base + ".json") as f:
        return json.load(f)


def min_image(d, L):
    d = d - L * np.round(d / L)
    return d


# ---------------------------------------------------------------------------
# 1. energy / temperature / collisions (from the diag csv + json)
# ---------------------------------------------------------------------------
def energy_stats():
    out = {}
    series = {}
    for name, base in INTEG_RUNS.items():
        dg = load_diag(base)
        js = load_json(base)
        assert js["integrator_id"] == name, (name, js["integrator_id"])
        E = dg["Etot"]
        dE = E - E[0]
        N = js["N"]
        out[name] = dict(
            integrator_id=js["integrator_id"], integrator=js["integrator"], run=os.path.relpath(base, ROOT),
            E0_kT=float(E[0]), E_end_kT=float(E[-1]),
            drift_kT=float(dE[-1]), drift_rel=float(dE[-1] / abs(E[0])), drift_per_particle_kT=float(dE[-1] / N),
            max_dev_kT=float(np.abs(dE).max()), max_dev_rel=float(np.abs(dE).max() / abs(E[0])),
            max_dev_per_particle_kT=float(np.abs(dE).max() / N),
            rms_kT=float(np.sqrt(np.mean(dE ** 2))), rms_rel=float(np.sqrt(np.mean(dE ** 2)) / abs(E[0])),
            rms_about_mean_kT=float(dE.std()), rms_about_mean_rel=float(dE.std() / abs(E[0])),
            slope_kT_per_tau=float(np.polyfit(dg["time"], dE, 1)[0]),
            frame_scatter_rms_kT=float(np.std(np.diff(E))),
            json_drift_rel=js["energy_drift_rel"], json_max_dev_rel=js["energy_max_dev_rel"],
            json_rms_rel=js["energy_rms_rel"], json_slope_rel_per_tau=js["energy_slope_per_tau_rel"],
            T_mean=float(dg["T_inst"].mean()), T_std=float(dg["T_inst"].std()),
            PE_mean_kT=float(dg["PE"].mean()), PE_std_kT=float(dg["PE"].std()),
            KE_mean_kT=float(dg["KE"].mean()), KE_std_kT=float(dg["KE"].std()),
            n_collisions=int(js["n_collisions"]),
            collision_rate_per_particle_per_tau=js["collision_rate_per_particle_per_tau"],
            Z_virial=js["Z_virial"], Z_CS=js["Z_CS"], Z_rel_dev=js["Z_rel_dev"],
            wall_seconds=js["wall_seconds"], n_pairs_mean=float(dg["n_pairs_pm_1p2"].mean()),
        )
        series[name] = dict(time=dg["time"], dE=dE, KE=dg["KE"], PE=dg["PE"], T=dg["T_inst"],
                            ncoll=dg["n_collisions"], npairs=dg["n_pairs_pm_1p2"])
    # neutral run
    dgn = load_diag(RUNS["neutral"])
    jsn = load_json(RUNS["neutral"])
    out["neutral"] = dict(integrator_id="edmd", run="traj/hs_neutral", n_collisions=int(jsn["n_collisions"]),
                          collision_rate_per_particle_per_tau=jsn["collision_rate_per_particle_per_tau"],
                          enskog_rate_per_particle_per_tau=jsn["enskog_rate_per_particle_per_tau"],
                          Z_virial=jsn["Z_virial"], Z_CS=jsn["Z_CS"], Z_rel_dev=jsn["Z_rel_dev"],
                          E0_kT=float(dgn["Etot"][0]),
                          max_dev_rel=float(np.abs(dgn["Etot"] - dgn["Etot"][0]).max() / dgn["Etot"][0]),
                          T_mean=float(dgn["T_inst"].mean()), T_std=float(dgn["T_inst"].std()))
    series["neutral"] = dict(time=dgn["time"], ncoll=dgn["n_collisions"], T=dgn["T_inst"])
    # write csv
    t = series["cv-phys"]["time"]
    cols = [t] + [series[k]["dE"] for k in INTEG_RUNS] + [series["cv-phys"]["KE"], series["cv-phys"]["PE"],
                                                        series["cv-phys"]["T"], series["neutral"]["ncoll"],
                                                        series["cv-phys"]["ncoll"]]
    hdr = "time_tau," + ",".join(f"dE_{k}_kT" for k in INTEG_RUNS) + ",KE_cvphys_kT,PE_cvphys_kT,T_cvphys,ncoll_neutral,ncoll_charged"
    np.savetxt(os.path.join(DATADIR, "energy_timeseries.csv"), np.column_stack(cols), delimiter=",", header=hdr,
               comments="", fmt="%.10g")
    return out, series


# ---------------------------------------------------------------------------
# 2. pairing / clusters / lifetimes (charged run, npz)
# ---------------------------------------------------------------------------
def components_sizes(bonds):
    """Sizes of connected components of a boolean symmetric adjacency matrix (small n)."""
    n = bonds.shape[0]
    seen = np.zeros(n, bool)
    sizes = []
    degs = bonds.sum(1)
    shapes = []  # 'chain' / 'branched' for size >= 3
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in np.nonzero(bonds[a])[0]:
                if not seen[b]:
                    seen[b] = True
                    stack.append(b)
        sizes.append(len(comp))
        if len(comp) >= 3:
            shapes.append("chain" if degs[comp].max() <= 2 else "branched")
    return sizes, shapes


def pairing_stats(ch, neu):
    pos, q, L, times = ch["pos"], ch["charge"], ch["L"], ch["times"]
    ions = np.nonzero(q != 0)[0]
    plus = np.nonzero(q > 0)[0]
    minus = np.nonzero(q < 0)[0]
    F = pos.shape[0]
    nplus, nminus = len(plus), len(minus)
    P = pos[:, plus].astype(np.float64)   # (F, 20, 3)
    M = pos[:, minus].astype(np.float64)
    d = P[:, :, None, :] - M[:, None, :, :]
    d = min_image(d, L)
    r_pm = np.sqrt((d ** 2).sum(-1))      # (F, 20, 20)
    contact = r_pm < BOND
    npairs = contact.sum((1, 2))
    # fraction of ions whose nearest opposite ion is within BOND / LOOSE
    near_plus = r_pm.min(2)   # (F, 20)
    near_minus = r_pm.min(1)
    nearest = np.concatenate([near_plus, near_minus], 1)  # (F, 40)
    frac_contact = (nearest < BOND).mean(1)
    frac_loose = (nearest < LOOSE).mean(1)
    n_opp_loose = np.concatenate([(r_pm < LOOSE).sum(2), (r_pm < LOOSE).sum(1)], 1).mean(1)
    # same-sign contacts (for completeness)
    def same(X):
        dd = min_image(X[:, :, None, :] - X[:, None, :, :], L)
        rr = np.sqrt((dd ** 2).sum(-1))
        iu = np.triu_indices(X.shape[1], 1)
        return rr[:, iu[0], iu[1]]
    r_pp = same(P)
    r_mm = same(M)
    n_same_contact = (r_pp < BOND).sum(1) + (r_mm < BOND).sum(1)

    # clusters per frame (bonds = +/- contacts only)
    nion = nplus + nminus
    maxsize = 1
    size_counts = {}
    largest = np.zeros(F, int)
    nchain = np.zeros(F, int)
    nbranched = np.zeros(F, int)
    for k in range(F):
        A = np.zeros((nion, nion), bool)
        A[:nplus, nplus:] = contact[k]
        A[nplus:, :nplus] = contact[k].T
        sizes, shapes = components_sizes(A)
        for s in sizes:
            size_counts[s] = size_counts.get(s, 0) + 1
        largest[k] = max(sizes)
        nchain[k] = shapes.count("chain")
        nbranched[k] = shapes.count("branched")
    maxsize = max(size_counts)
    sizes_arr = np.arange(1, maxsize + 1)
    clusters_per_frame = np.array([size_counts.get(s, 0) / F for s in sizes_arr])
    ion_frac = clusters_per_frame * sizes_arr / nion

    # contact episodes (consecutive frames with r < BOND) for every +/- pair
    episodes = []  # (iplus, iminus, start_frame, n_frames, censored)
    for i in range(nplus):
        for j in range(nminus):
            c = contact[:, i, j].astype(np.int8)
            if not c.any():
                continue
            dc = np.diff(np.concatenate([[0], c, [0]]))
            starts = np.nonzero(dc == 1)[0]
            ends = np.nonzero(dc == -1)[0]
            for s, e in zip(starts, ends):
                censored = int(s == 0 or e == F)
                episodes.append((i, j, int(s), int(e - s), censored))
    ep = np.array(episodes, dtype=int)
    nfr = ep[:, 3]
    dt_frame = float(times[1] - times[0])
    life = nfr * dt_frame
    uncens = ep[:, 4] == 0
    def lstats(x):
        return dict(n=int(len(x)), mean_tau=float(x.mean()), median_tau=float(np.median(x)),
                    p90_tau=float(np.percentile(x, 90)), max_tau=float(x.max()),
                    mean_s=float(x.mean() / dt_frame / FPS), median_s=float(np.median(x) / dt_frame / FPS),
                    p90_s=float(np.percentile(x, 90) / dt_frame / FPS), max_s=float(x.max() / dt_frame / FPS))
    life_stats = lstats(life)
    life_stats_uncensored = lstats(life[uncens])
    life_stats["n_censored"] = int((~uncens).sum())
    # distinct partners per + ion / per - ion over the run
    ever = contact.any(0)  # (20,20)
    partners_plus = ever.sum(1)
    partners_minus = ever.sum(0)
    # survival function
    lt_sorted = np.sort(life)
    surv_x = np.unique(lt_sorted)
    surv_y = np.array([(life >= x).mean() for x in surv_x])

    # random-placement baselines from the neutral run: random 20+20 subsets of the neutral spheres
    rng = np.random.default_rng(12345)
    posn = neu["pos"]
    Fn, Nn = posn.shape[0], posn.shape[1]
    nsub = 100
    base_loose = np.zeros(nsub)
    base_contact = np.zeros(nsub)
    base_pairs = np.zeros(nsub)
    for s in range(nsub):
        idx = rng.choice(Nn, nplus + nminus, replace=False)
        PP = posn[:, idx[:nplus]].astype(np.float64)
        MM = posn[:, idx[nplus:]].astype(np.float64)
        dd = min_image(PP[:, :, None, :] - MM[:, None, :, :], L)
        rr = np.sqrt((dd ** 2).sum(-1))
        nn = np.concatenate([rr.min(2), rr.min(1)], 1)
        base_loose[s] = (nn < LOOSE).mean()
        base_contact[s] = (nn < BOND).mean()
        base_pairs[s] = (rr < BOND).sum((1, 2)).mean()
    V = L ** 3
    v_shell = 4.0 / 3.0 * np.pi * (LOOSE ** 3 - 1.0)
    ideal_loose = 1.0 - (1.0 - v_shell / V) ** nminus   # analyze.py's hard-core-only estimate
    v_shell_c = 4.0 / 3.0 * np.pi * (BOND ** 3 - 1.0)
    ideal_contact = 1.0 - (1.0 - v_shell_c / V) ** nminus

    h = F // 2
    stats = dict(
        n_frames=F, dt_frame_tau=dt_frame, n_plus=nplus, n_minus=nminus,
        pairs_per_frame_mean=float(npairs.mean()), pairs_per_frame_std=float(npairs.std()),
        pairs_per_frame_min=int(npairs.min()), pairs_per_frame_max=int(npairs.max()),
        pairs_first_half=float(npairs[:h].mean()), pairs_second_half=float(npairs[h:].mean()),
        frac_contact_mean=float(frac_contact.mean()), frac_contact_std=float(frac_contact.std()),
        frac_contact_min=float(frac_contact.min()), frac_contact_max=float(frac_contact.max()),
        frac_loose_mean=float(frac_loose.mean()), frac_loose_std=float(frac_loose.std()),
        frac_loose_min=float(frac_loose.min()), frac_loose_max=float(frac_loose.max()),
        n_opposite_within_loose_per_ion=float(n_opp_loose.mean()),
        mean_nearest_opposite_sigma=float(nearest.mean()),
        same_sign_contacts_per_frame=float(n_same_contact.mean()),
        baseline_loose_ideal=float(ideal_loose), baseline_contact_ideal=float(ideal_contact),
        baseline_loose_neutral_subsets_mean=float(base_loose.mean()), baseline_loose_neutral_subsets_std=float(base_loose.std()),
        baseline_contact_neutral_subsets_mean=float(base_contact.mean()), baseline_contact_neutral_subsets_std=float(base_contact.std()),
        baseline_pairs_neutral_subsets_mean=float(base_pairs.mean()),
        cluster_sizes=sizes_arr.tolist(), clusters_per_frame=clusters_per_frame.tolist(), ion_fraction_by_size=ion_frac.tolist(),
        frac_ions_in_clusters_ge3=float(ion_frac[2:].sum()), largest_cluster_mean=float(largest.mean()),
        largest_cluster_max=int(largest.max()), chains_per_frame=float(nchain.mean()),
        branched_per_frame=float(nbranched.mean()), frames_with_branched_frac=float((nbranched > 0).mean()),
        lifetimes=life_stats, lifetimes_uncensored=life_stats_uncensored,
        partners_per_plus_mean=float(partners_plus.mean()), partners_per_plus_min=int(partners_plus.min()),
        partners_per_plus_max=int(partners_plus.max()),
        partners_per_minus_mean=float(partners_minus.mean()), partners_per_minus_min=int(partners_minus.min()),
        partners_per_minus_max=int(partners_minus.max()),
        n_distinct_pairs_ever=int(ever.sum()), n_episodes=int(len(life)),
        episodes_per_tau=float(len(life) / (times[-1] - times[0])),
    )
    # csv outputs
    np.savetxt(os.path.join(DATADIR, "pairing_timeseries.csv"),
               np.column_stack([times, npairs, frac_contact, frac_loose, n_opp_loose, n_same_contact, largest, nchain, nbranched]),
               delimiter=",", comments="", fmt="%.6g",
               header="time_tau,n_pm_pairs_1p2,frac_ions_contact_1p2,frac_ions_opposite_within_2p5,n_opposite_within_2p5_per_ion,"
                      "n_same_sign_contacts,largest_cluster,n_chains_ge3,n_branched_ge3")
    np.savetxt(os.path.join(DATADIR, "cluster_hist.csv"), np.column_stack([sizes_arr, clusters_per_frame, ion_frac]),
               delimiter=",", comments="", fmt="%.6g", header="cluster_size,clusters_per_frame,fraction_of_ions")
    np.savetxt(os.path.join(DATADIR, "contact_episodes.csv"), np.column_stack([ep, life]), delimiter=",", comments="",
               fmt="%d,%d,%d,%d,%d,%.4f", header="plus_index,minus_index,start_frame,n_frames,censored,lifetime_tau")
    np.savetxt(os.path.join(DATADIR, "lifetime_survival.csv"), np.column_stack([surv_x, surv_y]), delimiter=",",
               comments="", fmt="%.6g", header="lifetime_tau,survival_fraction")
    np.savez(os.path.join(CACHE_DIR, "pairing.npz"), times=times, npairs=npairs, frac_contact=frac_contact,
             frac_loose=frac_loose, sizes=sizes_arr, ion_frac=ion_frac, life=life, surv_x=surv_x, surv_y=surv_y,
             base_loose=base_loose, base_contact=base_contact, partners_plus=partners_plus)
    return stats


# ---------------------------------------------------------------------------
# 3. radial distribution functions
# ---------------------------------------------------------------------------
def pair_hist(pos, idx_a, idx_b, L, edges, same, block=200, frames=None):
    """Histogram of minimum-image pair distances between groups a and b, summed over frames.
    same=True: a == b, each unordered pair counted once."""
    F = pos.shape[0]
    frames = range(F) if frames is None else frames
    counts = np.zeros(len(edges) - 1, np.int64)
    rmax = edges[-1]
    nf = 0
    for k in frames:
        A = pos[k, idx_a].astype(np.float64)
        B = pos[k, idx_b].astype(np.float64)
        for i0 in range(0, len(idx_a), block):
            a = A[i0:i0 + block]
            d = a[:, None, :] - B[None, :, :]
            d -= L * np.round(d / L)
            r2 = np.einsum("ijk,ijk->ij", d, d)
            if same:
                # keep j > i only
                ii = np.arange(i0, min(i0 + block, len(idx_a)))
                mask = np.arange(len(idx_b))[None, :] > ii[:, None]
                r2 = r2[mask]
            r2 = r2[r2 < rmax * rmax]
            counts += np.histogram(np.sqrt(r2), bins=edges)[0]
        nf += 1
    return counts, nf


def g_from_counts(counts, nf, npairs, V, edges):
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    expected = nf * npairs * shell / V
    g = counts / expected
    err = np.sqrt(np.maximum(counts, 1)) / expected
    return g, err


def rebin(counts, edges, factor):
    n = (len(counts) // factor) * factor
    c = counts[:n].reshape(-1, factor).sum(1)
    e = edges[:n + 1:factor]
    return c, e


def structure_stats(neu, ch):
    L = neu["L"]
    V = L ** 3
    rmax = 5.7   # < L/2 = 5.718
    fine = np.round(np.arange(0.9, rmax + 1e-9, 0.01), 6)
    out = {}
    # neutral run, all 721 frames
    cache = os.path.join(CACHE_DIR, "gr_counts.npz")
    if os.path.exists(cache):
        C = dict(np.load(cache))
        log("g(r) counts loaded from cache")
    else:
        C = {}
        Nn = neu["pos"].shape[1]
        allidx = np.arange(Nn)
        t0 = time.time()
        C["nn_neutral"], C["nf_neutral"] = pair_hist(neu["pos"], allidx, allidx, L, fine, same=True)
        log(f"neutral g(r): {time.time() - t0:.1f}s")
        q = ch["charge"]
        neut = np.nonzero(q == 0)[0]
        plus = np.nonzero(q > 0)[0]
        minus = np.nonzero(q < 0)[0]
        ions = np.nonzero(q != 0)[0]
        t0 = time.time()
        C["pm"], C["nf_ch"] = pair_hist(ch["pos"], plus, minus, L, fine, same=False)
        cpp, _ = pair_hist(ch["pos"], plus, plus, L, fine, same=True)
        cmm, _ = pair_hist(ch["pos"], minus, minus, L, fine, same=True)
        C["pp"] = cpp
        C["mm"] = cmm
        C["in"], _ = pair_hist(ch["pos"], ions, neut, L, fine, same=False)
        C["nn_charged"], _ = pair_hist(ch["pos"], neut, neut, L, fine, same=True)
        log(f"charged g(r): {time.time() - t0:.1f}s")
        np.savez(cache, **C)
    Nn = neu["pos"].shape[1]
    q = ch["charge"]
    n0 = int((q == 0).sum()); npl = int((q > 0).sum()); nmi = int((q < 0).sum())
    nf_n = int(C["nf_neutral"]); nf_c = int(C["nf_ch"])
    g_nn, e_nn = g_from_counts(C["nn_neutral"], nf_n, Nn * (Nn - 1) / 2, V, fine)
    rc_fine = 0.5 * (fine[1:] + fine[:-1])
    # contact extrapolation: polynomial fit of g over [1.0, 1.1]
    eta = ch["meta"]["eta"]
    g_cs = (1 - eta / 2) / (1 - eta) ** 3
    sel = (rc_fine >= 1.0) & (rc_fine <= 1.10)
    for deg in (1, 2):
        p = np.polyfit(rc_fine[sel] - 1.0, g_nn[sel], deg)
        out[f"g_contact_neutral_fit{deg}"] = float(np.polyval(p, 0.0))
    out["g_contact_neutral_firstbin"] = float(g_nn[(rc_fine > 1.0) & (rc_fine < 1.01)][0])
    out["g_contact_CS"] = float(g_cs)
    out["eta"] = eta
    out["rho"] = Nn / V
    Zn = neu["meta"]["Z_virial"]
    out["g_contact_from_Z_virial_neutral"] = float((Zn - 1) / (4 * eta))
    out["g_contact_from_Z_virial_charged"] = float((ch["meta"]["Z_virial"] - 1) / (4 * eta))
    # peak / first minimum of g_nn
    ipk = np.argmax(g_nn)
    out["g_nn_first_peak_r"] = float(rc_fine[ipk]); out["g_nn_first_peak_value"] = float(g_nn[ipk])
    sel2 = (rc_fine > 1.2) & (rc_fine < 2.2)
    imin = np.argmin(np.where(sel2, g_nn, np.inf))
    out["g_nn_first_min_r"] = float(rc_fine[imin]); out["g_nn_first_min_value"] = float(g_nn[imin])
    sel3 = (rc_fine > 1.6) & (rc_fine < 2.6)
    i2 = np.argmax(np.where(sel3, g_nn, -np.inf))
    out["g_nn_second_peak_r"] = float(rc_fine[i2]); out["g_nn_second_peak_value"] = float(g_nn[i2])
    np.savetxt(os.path.join(DATADIR, "gr_neutral.csv"), np.column_stack([rc_fine, g_nn, e_nn, C["nn_neutral"]]),
               delimiter=",", comments="", fmt="%.6g", header="r_sigma,g_nn,g_err,pair_counts")

    # charged run, 0.05 bins
    factor = 5
    c_pm, coarse = rebin(C["pm"], fine, factor)
    c_pp, _ = rebin(C["pp"] + C["mm"], fine, factor)
    c_in, _ = rebin(C["in"], fine, factor)
    c_nnc, _ = rebin(C["nn_charged"], fine, factor)
    c_hs, _ = rebin(C["nn_neutral"], fine, factor)
    rc = 0.5 * (coarse[1:] + coarse[:-1])
    ic = int(np.argmin(np.abs(coarse - 1.0)))      # index of the contact bin [1.00, 1.05)
    assert abs(coarse[ic] - 1.0) < 1e-9
    g_pm, e_pm = g_from_counts(c_pm, nf_c, npl * nmi, V, coarse)
    g_pp, e_pp = g_from_counts(c_pp, nf_c, npl * (npl - 1) / 2 + nmi * (nmi - 1) / 2, V, coarse)
    g_in, e_in = g_from_counts(c_in, nf_c, (npl + nmi) * n0, V, coarse)
    g_nnc, e_nnc = g_from_counts(c_nnc, nf_c, n0 * (n0 - 1) / 2, V, coarse)
    g_hs, e_hs = g_from_counts(c_hs, nf_n, Nn * (Nn - 1) / 2, V, coarse)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -np.log(g_pm / g_hs)
        w_err = np.sqrt(1.0 / np.maximum(c_pm, 1) + 1.0 / np.maximum(c_hs, 1))
    w = np.where(c_pm > 0, w, np.nan)
    lB = ch["meta"]["bjerrum"]
    rho_ion = (npl + nmi) / V
    kappa = np.sqrt(4 * np.pi * lB * rho_ion)
    w_coul = -lB / rc
    w_dh = -lB * np.exp(-kappa * (rc - 1.0)) / (rc * (1 + kappa))
    np.savetxt(os.path.join(DATADIR, "gr_charged.csv"),
               np.column_stack([rc, g_pm, e_pm, g_pp, e_pp, g_in, e_in, g_nnc, e_nnc, g_hs, e_hs, c_pm, c_pp, c_in]),
               delimiter=",", comments="", fmt="%.6g",
               header="r_sigma,g_pm,g_pm_err,g_samesign,g_samesign_err,g_ion_neutral,g_ion_neutral_err,g_nn_charged,g_nn_charged_err,"
                      "g_hs_neutral_run,g_hs_err,counts_pm,counts_samesign,counts_ion_neutral")
    np.savetxt(os.path.join(DATADIR, "pmf_pm.csv"), np.column_stack([rc, w, w_err, w_coul, w_dh]), delimiter=",", comments="",
               fmt="%.6g", header="r_sigma,w_pm_kT,w_err_kT,bare_coulomb_kT,debye_hueckel_kT")
    # contact values (first 0.05 bin) and exponential extrapolation of g_pm to sigma+ from 0.01 bins over [1, 1.15]
    g_pm_fine, _ = g_from_counts(C["pm"], nf_c, npl * nmi, V, fine)
    self_ = (rc_fine >= 1.0) & (rc_fine <= 1.15) & (C["pm"] > 0)
    p = np.polyfit(rc_fine[self_] - 1.0, np.log(g_pm_fine[self_]), 1, w=np.sqrt(C["pm"][self_]))
    g_pm_contact_fit = float(np.exp(p[1]))
    g_hs_contact = out["g_contact_neutral_fit2"]
    out.update(dict(
        bins_fine_sigma=0.01, bins_coarse_sigma=0.05, frames_neutral=nf_n, frames_charged=nf_c,
        contact_bin_sigma=[float(coarse[ic]), float(coarse[ic + 1])],
        g_pm_firstbin=float(g_pm[ic]), g_pm_firstbin_err=float(e_pm[ic]), g_pm_contact_fit=g_pm_contact_fit,
        g_pm_fit_decay_length_sigma=float(-1.0 / p[0]),
        g_samesign_firstbin=float(g_pp[ic]), g_samesign_firstbin_err=float(e_pp[ic]),
        g_samesign_counts_firstbin=int(c_pp[ic]), g_pm_counts_firstbin=int(c_pm[ic]),
        g_ion_neutral_firstbin=float(g_in[ic]), g_ion_neutral_firstbin_err=float(e_in[ic]),
        g_nn_charged_firstbin=float(g_nnc[ic]), g_hs_firstbin_coarse=float(g_hs[ic]),
        g_large_r_mean_pm=float(np.nanmean(g_pm[rc > 4.5])), g_large_r_mean_nn=float(np.nanmean(g_hs[rc > 4.5])),
        g_large_r_mean_in=float(np.nanmean(g_in[rc > 4.5])), g_large_r_mean_samesign=float(np.nanmean(g_pp[rc > 4.5])),
        g_pm_peak_r=float(rc[np.nanargmax(g_pm)]), g_pm_second_peak=None,
        w_contact_firstbin_kT=float(w[ic]), w_contact_firstbin_err_kT=float(w_err[ic]),
        w_min_kT=float(np.nanmin(w)), w_min_r=float(rc[np.nanargmin(w)]),
        w_contact_extrapolated_kT=float(-np.log(g_pm_contact_fit / g_hs_contact)),
        bare_coulomb_contact_kT=float(-lB), debye_hueckel_contact_kT=float(-lB / (1 + kappa)),
        kappa_sigma=float(kappa), debye_length_sigma=float(1 / kappa), l_B_sigma=lB, rho_ion=rho_ion,
        mean_ion_spacing_sigma=float(rho_ion ** (-1 / 3)),
        w_at_r=[(float(rr), float(ww)) for rr, ww in zip(rc[ic:ic + 12], w[ic:ic + 12])],
    ))
    # second peak of g_pm (+-+ chain / second shell) between 1.6 and 2.6
    s3 = (rc > 1.6) & (rc < 2.6)
    out["g_pm_second_peak_r"] = float(rc[s3][np.nanargmax(g_pm[s3])])
    out["g_pm_second_peak_value"] = float(np.nanmax(g_pm[s3]))
    # same-sign g at 2 sigma (ion-ion second shell via a shared partner)
    s4 = (rc > 1.8) & (rc < 2.3)
    out["g_samesign_peak_r"] = float(rc[s4][np.nanargmax(g_pp[s4])])
    out["g_samesign_peak_value"] = float(np.nanmax(g_pp[s4]))
    np.savez(os.path.join(CACHE_DIR, "gr.npz"), rc_fine=rc_fine, g_nn=g_nn, e_nn=e_nn, rc=rc, g_pm=g_pm, e_pm=e_pm,
             g_pp=g_pp, e_pp=e_pp, g_in=g_in, e_in=e_in, g_nnc=g_nnc, g_hs=g_hs, w=w, w_err=w_err, w_coul=w_coul,
             w_dh=w_dh, c_pm=c_pm, c_pp=c_pp)
    return out


# ---------------------------------------------------------------------------
# 4. dynamics: unwrapped trajectories, MSD, displacements, free flight
# ---------------------------------------------------------------------------
def unwrap(pos, L):
    p = pos.astype(np.float64)
    d = np.diff(p, axis=0)
    d -= L * np.round(d / L)
    u = np.concatenate([p[:1], p[:1] + np.cumsum(d, axis=0)], 0)
    return u, d


def msd_curve(u, idx, lags):
    x = u[:, idx]
    F = x.shape[0]
    msd = np.zeros(len(lags))
    for n, k in enumerate(lags):
        dd = x[k:] - x[:F - k]
        msd[n] = np.mean((dd ** 2).sum(-1))
    return msd


def dynamics_stats(neu, ch, enstats):
    L = neu["L"]
    times = neu["times"]
    dt = float(times[1] - times[0])
    F = len(times)
    out = {}
    un, dn = unwrap(neu["pos"], L)
    uc, dc = unwrap(ch["pos"], L)
    q = ch["charge"]
    neut_c = np.nonzero(q == 0)[0]
    ions = np.nonzero(q != 0)[0]
    all_n = np.arange(neu["pos"].shape[1])
    lags = np.unique(np.round(np.logspace(0, np.log10(F - 1), 80)).astype(int))
    lags = lags[lags >= 1]
    tl = lags * dt
    t0 = time.time()
    msd_n = msd_curve(un, all_n, lags)
    msd_cn = msd_curve(uc, neut_c, lags)
    msd_ci = msd_curve(uc, ions, lags)
    log(f"MSD: {time.time() - t0:.1f}s")
    # D from 1-4 tau window
    win = (tl >= 1.0) & (tl <= 4.0)
    D = {}
    for key, m in (("neutral_run_neutrals", msd_n), ("charged_run_neutrals", msd_cn), ("charged_run_ions", msd_ci)):
        slope, icpt = np.polyfit(tl[win], m[win], 1)
        D[key] = dict(D_fit_slope=float(slope / 6), D_msd_over_6t=float(np.mean(m[win] / (6 * tl[win]))),
                      msd_at_1tau=float(np.interp(1.0, tl, m)), msd_at_4tau=float(np.interp(4.0, tl, m)),
                      msd_end=float(m[-1]), rms_disp_end_sigma=float(np.sqrt(m[-1])))
    # rms displacement over the full 18 tau (single origin, t=0 -> t=18)
    out["rms_disp_18tau_neutral_run"] = float(np.sqrt(np.mean(((un[-1] - un[0]) ** 2).sum(-1))))
    out["rms_disp_18tau_charged_neutrals"] = float(np.sqrt(np.mean(((uc[-1, neut_c] - uc[0, neut_c]) ** 2).sum(-1))))
    out["rms_disp_18tau_charged_ions"] = float(np.sqrt(np.mean(((uc[-1, ions] - uc[0, ions]) ** 2).sum(-1))))
    # Enskog D with the CS contact value
    eta = neu["meta"]["eta"]
    rho = neu["pos"].shape[1] / L ** 3
    g_cs = (1 - eta / 2) / (1 - eta) ** 3
    D_E = 3.0 / (8.0 * rho * g_cs) * np.sqrt(1.0 / np.pi)
    out["D_enskog_CS"] = float(D_E)
    out["D"] = D
    out["D_ratio_to_enskog_neutral"] = float(D["neutral_run_neutrals"]["D_fit_slope"] / D_E)
    out["mean_free_time_tau"] = float(1.0 / enstats["neutral"]["collision_rate_per_particle_per_tau"])
    out["mean_free_path_sigma"] = float(np.sqrt(8 / np.pi) * out["mean_free_time_tau"])
    np.savetxt(os.path.join(DATADIR, "msd.csv"), np.column_stack([lags, tl, msd_n, msd_cn, msd_ci]), delimiter=",",
               comments="", fmt="%.6g", header="lag_frames,lag_tau,msd_neutral_run,msd_charged_neutrals,msd_charged_ions")

    # per-frame displacements
    disp_n = np.sqrt((dn ** 2).sum(-1))            # (F-1, N)
    disp_ci = np.sqrt((dc[:, ions] ** 2).sum(-1))
    disp_cn = np.sqrt((dc[:, neut_c] ** 2).sum(-1))
    edges = np.linspace(0, 0.16, 65)
    hn = np.histogram(disp_n, bins=edges, density=True)[0]
    hci = np.histogram(disp_ci, bins=edges, density=True)[0]
    hcn = np.histogram(disp_cn, bins=edges, density=True)[0]
    # Maxwell free-flight prediction for d = v dt
    dcen = 0.5 * (edges[1:] + edges[:-1])
    v = dcen / dt
    pmax = 4 * np.pi * v ** 2 * (1 / (2 * np.pi)) ** 1.5 * np.exp(-v ** 2 / 2) / dt
    np.savetxt(os.path.join(DATADIR, "displacement_hist.csv"), np.column_stack([dcen, hn, hcn, hci, pmax]), delimiter=",",
               comments="", fmt="%.6g",
               header="disp_sigma,pdf_neutral_run,pdf_charged_neutrals,pdf_charged_ions,pdf_maxwell_free_flight")
    out["disp_per_frame"] = dict(
        neutral_run_mean=float(disp_n.mean()), neutral_run_rms=float(np.sqrt((disp_n ** 2).mean())), neutral_run_max=float(disp_n.max()),
        charged_neutrals_mean=float(disp_cn.mean()), charged_neutrals_max=float(disp_cn.max()),
        charged_ions_mean=float(disp_ci.mean()), charged_ions_rms=float(np.sqrt((disp_ci ** 2).mean())), charged_ions_max=float(disp_ci.max()),
        maxwell_mean=float(np.sqrt(8 / np.pi) * dt), maxwell_rms=float(np.sqrt(3) * dt))
    # free flight: identical displacement over two consecutive intervals
    def freeflight(d, idx):
        a = d[:-1, idx]
        b = d[1:, idx]
        diff = np.sqrt(((a - b) ** 2).sum(-1))   # (F-2, n)
        frac = (diff < FREEFLIGHT_TOL).mean(1)
        return frac, diff
    ff_n, diff_n = freeflight(dn, all_n)
    ff_cn, diff_cn = freeflight(dc, neut_c)
    ff_ci, diff_ci = freeflight(dc, ions)
    Gn = enstats["neutral"]["collision_rate_per_particle_per_tau"]
    Gc = enstats["cv-phys"]["collision_rate_per_particle_per_tau"]
    GE = enstats["neutral"]["enskog_rate_per_particle_per_tau"]
    # gap check: how many values fall in the ambiguous window [1e-6, 1e-3]
    amb = float(((diff_n > FREEFLIGHT_TOL) & (diff_n < 1e-3)).mean())
    # speed-resolved expectation: <exp(-Gamma(v) 2dt)> with Gamma(v) from the Enskog relative-speed factor
    out["freeflight"] = dict(
        tol_sigma=FREEFLIGHT_TOL, window_tau=2 * dt,
        frac_neutral_run_mean=float(ff_n.mean()), frac_neutral_run_std=float(ff_n.std()),
        frac_charged_neutrals_mean=float(ff_cn.mean()), frac_charged_neutrals_std=float(ff_cn.std()),
        frac_charged_ions_mean=float(ff_ci.mean()),
        expected_measured_rate_neutral=float(np.exp(-2 * dt * Gn)), expected_measured_rate_charged=float(np.exp(-2 * dt * Gc)),
        expected_enskog=float(np.exp(-2 * dt * GE)), Gamma_neutral=Gn, Gamma_charged=Gc, Gamma_enskog=GE,
        ratio_measured_over_expected_neutral=float(ff_n.mean() / np.exp(-2 * dt * Gn)),
        frac_between_1e5_and_1e3=amb, frac_between_3e6_and_1e5=float(((diff_n > 3e-6) & (diff_n < FREEFLIGHT_TOL)).mean()),
        roundoff_cluster_max=float(diff_n[diff_n < 3e-6].max()),
        note="float32 roundoff of the npz gives |dr_k - dr_k+1| <= ~2e-6; values between 1e-5 and 1e-3 are collisions "
             "within ~1e-3/|dv| of a frame time (expected ~1.4%); float64 xyz gives the same fraction on frames 0-11",
        frac_one_interval_free_neutral=float((np.sqrt(((dn[1:] - dn[:-1]) ** 2).sum(-1)) < FREEFLIGHT_TOL).mean()),
    )
    tt = times[1:-1]
    np.savetxt(os.path.join(DATADIR, "freeflight.csv"), np.column_stack([tt, ff_n, ff_cn, ff_ci]), delimiter=",", comments="",
               fmt="%.6g", header="time_tau,frac_free_neutral_run,frac_free_charged_neutrals,frac_free_charged_ions")
    np.savez(os.path.join(CACHE_DIR, "dyn.npz"), lags=lags, tl=tl, msd_n=msd_n, msd_cn=msd_cn, msd_ci=msd_ci,
             dcen=dcen, hn=hn, hcn=hcn, hci=hci, pmax=pmax, tt=tt, ff_n=ff_n, ff_cn=ff_cn, ff_ci=ff_ci,
             disp_n_sample=disp_n[::7].ravel(), disp_ci_all=disp_ci.ravel())
    return out


def main():
    t0 = time.time()
    log("loading")
    neu = load_npz(RUNS["neutral"])
    ch = load_npz(RUNS["charged"])
    assert neu["pos"].shape == (721, 1000, 3) and ch["pos"].shape == (721, 1000, 3)
    res = {}
    log("energy")
    res["energy"], _ = energy_stats()
    log("pairing")
    res["pairing"] = pairing_stats(ch, neu)
    # consistency with the simulator's own pair count
    dg = load_diag(RUNS["charged"])
    pr = np.load(os.path.join(CACHE_DIR, "pairing.npz"))
    res["pairing"]["pairs_match_diag_frames"] = int((pr["npairs"] == dg["n_pairs_pm_1p2"]).sum())
    res["pairing"]["pairs_max_abs_diff_vs_diag"] = int(np.abs(pr["npairs"] - dg["n_pairs_pm_1p2"]).max())
    log("structure")
    res["structure"] = structure_stats(neu, ch)
    log("dynamics")
    res["dynamics"] = dynamics_stats(neu, ch, res["energy"])
    res["system"] = dict(N=1000, L=neu["L"], eta=neu["meta"]["eta"], n_frames=721, dt_frame=0.025, t_run=18.0,
                         fps=FPS, movie_seconds_per_tau=1.0 / (0.025 * FPS),
                         bjerrum=ch["meta"]["bjerrum"], nplus=20, nminus=20)
    with open(os.path.join(DATADIR, "physics_numbers_raw.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"done in {time.time() - t0:.1f}s -> {DATADIR}/physics_numbers_raw.json")


if __name__ == "__main__":
    main()
