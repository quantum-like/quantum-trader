#!/usr/bin/env python3
"""analyze.py -- validate and characterise an hsdmd trajectory.

    python sim/analyze.py traj/<name>          (reads traj/<name>.xyz if present, else .npz)
    python sim/analyze.py traj/<name>.npz      (force a specific source)

Reports: minimum pair distance over ALL frames (hard-core check), temperature and
energy drift from <name>_diag.csv, a stationarity check (first half vs second half
of the production run, linear trends of PE and T; a WARNING means the charged
equilibration was too short), FORMAT.md contract checks (type ids, json "machine"
string) and for charged runs a per-frame pairing / cluster analysis of the ions
(1.2 sigma +/- bond criterion).
Exit status: 1 if any hard-core overlap is found, 2 on a contract violation, else 0
(stationarity problems are warnings only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sim import read_xyz  # noqa: E402

BOND = 1.2  # sigma; Bjerrum-pair proxy and cluster bond criterion
# stationarity thresholds (see README_sim.md). A WARNING is issued when
#  * the linear PE trend over the run exceeds PE_TREND_WARN*|PE| AND PE_TREND_SIGMA per-frame std devs
#    (the second condition keeps 16-ion test systems, whose relative fluctuations are large, from
#    warning on noise), or
#  * the linear T trend over the run exceeds T_DRIFT_WARN*T, or
#  * the first 2 tau of production differ from the rest by more than RELAX_SIGMA std devs of the
#    2-tau block means (an initial relaxation = too short --t-equil-charged).
PE_TREND_WARN = 0.10
PE_TREND_SIGMA = 2.0
T_DRIFT_WARN = 0.01
RELAX_SIGMA = 3.0


# ---------------------------------------------------------------------------
def load(path_or_base, source="auto"):
    base, ext = os.path.splitext(path_or_base)
    if ext in (".xyz", ".npz"):
        source = ext[1:]
    else:
        base = path_or_base
    xyz, npz = base + ".xyz", base + ".npz"
    if source == "auto":
        source = "xyz" if os.path.exists(xyz) else "npz"
    if source == "xyz":
        tr = read_xyz(xyz, dtype=np.float64)
        tr["precision"] = "float64 (xyz, %.10f text)"
        tr["tol"] = 1e-9
    else:
        d = np.load(npz)
        tr = dict(pos=d["pos"], types=d["types"], charge=d["charge"], times=d["times"], L=float(d["L"]))
        tr["precision"] = "float32 (npz cache)"
        # float32 positions carry ~L*6e-8 absolute error each; be honest about it
        tr["tol"] = 4.0 * float(tr["L"]) * np.finfo(np.float32).eps
    tr["source"] = source
    tr["base"] = base
    return tr


def min_pair_distance(pos, L, block=256):
    """Minimum image minimum pair distance of one frame, O(N^2) in row blocks (float64)."""
    pos = np.asarray(pos, dtype=np.float64)
    N = pos.shape[0]
    best = np.inf
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        d -= L * np.round(d / L)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        rows = np.arange(i0, i1)
        r2[rows - i0, rows] = np.inf  # self
        best = min(best, float(r2.min()))
    return np.sqrt(best)


def components(nion, bonds):
    """Connected components over ions given a boolean (nion, nion) bond matrix."""
    parent = list(range(nion))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(bonds, 1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(a) for a in range(nion)]
    comps = {}
    for a, r in enumerate(roots):
        comps.setdefault(r, []).append(a)
    return list(comps.values())


def pairing_frame(pos_ions, q, L):
    """Per-frame ion statistics. Returns dict."""
    d = pos_ions[:, None, :] - pos_ions[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    opposite = (q[:, None] * q[None, :]) < 0
    r_opp = np.where(opposite, r, np.inf)
    nearest_opp = r_opp.min(axis=1)
    bonds = opposite & (r < BOND)
    comps = components(len(q), bonds)
    sizes = np.array([len(c) for c in comps])
    deg = bonds.sum(axis=1)
    nchain = nbranched = 0
    for c in comps:
        if len(c) >= 3:
            if deg[c].max() <= 2:
                nchain += 1
            else:
                nbranched += 1
    return dict(frac_paired=float(np.mean(nearest_opp < BOND)),
                n_pm_pairs=int(bonds.sum() // 2),
                sizes=sizes, largest=int(sizes.max()), nchain=nchain, nbranched=nbranched,
                mean_nearest_opp=float(np.mean(nearest_opp)))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", help="traj/<name>, traj/<name>.xyz or traj/<name>.npz")
    ap.add_argument("--source", choices=["auto", "xyz", "npz"], default="auto")
    ap.add_argument("--json", default=None, help="also write the analysis as JSON to this path")
    a = ap.parse_args(argv)

    t0 = time.time()
    tr = load(a.traj, a.source)
    pos, types, charge, times, L = tr["pos"], tr["types"], tr["charge"], tr["times"], float(tr["L"])
    F, N, _ = pos.shape
    print(f"analyze: {tr['base']}  source={tr['source']} [{tr['precision']}]  frames={F} N={N} L={L:.6f}")
    out = dict(base=tr["base"], source=tr["source"], frames=F, N=N, L=L)

    # ---- metadata / diag
    meta = {}
    if os.path.exists(tr["base"] + ".json"):
        with open(tr["base"] + ".json") as f:
            meta = json.load(f)
    diag = None
    dpath = tr["base"] + "_diag.csv"
    if os.path.exists(dpath):
        diag = np.genfromtxt(dpath, delimiter=",", names=True)
        if diag.ndim == 0:
            diag = diag.reshape(1)

    # ---- hard-core check over all frames
    t1 = time.time()
    dmin_frames = np.empty(F)
    for k in range(F):
        dmin_frames[k] = min_pair_distance(pos[k], L)
    dmin = float(dmin_frames.min())
    kmin = int(dmin_frames.argmin())
    tol = tr["tol"]
    overlap = dmin < 1.0 - tol
    print(f"  min pair distance over all frames: {dmin:.12f} (frame {kmin}, t={times[kmin]:.4f}); "
          f"tolerance 1-{tol:.1e}  -> {'OVERLAP FOUND' if overlap else 'OK, no overlap'}  [{time.time() - t1:.1f}s]")
    if diag is not None:
        md_sim = float(diag["min_dist"].min())
        print(f"  simulator's own per-frame min_dist (double precision, cell list): min {md_sim:.12f}")
        out["min_dist_simulator"] = md_sim
    out.update(min_dist=dmin, min_dist_frame=kmin, overlap=bool(overlap), tolerance=tol)

    # ---- contract checks (FORMAT.md / README_sim.md)
    contract_fail = []
    if not np.all(np.isin(types, (1, 2, 3))):
        contract_fail.append(f"type ids outside {{1,2,3}}: {sorted(set(np.unique(types).tolist()) - {1, 2, 3})}")
    if np.any(np.diff(types.astype(np.int64)) < 0):
        contract_fail.append("type ids are not non-decreasing (expected neutrals, then +, then -)")
    expect_q = {1: 0, 2: 1, 3: -1}
    if not np.all(charge == np.vectorize(lambda t: expect_q.get(int(t), 99))(types)):
        contract_fail.append("charge column inconsistent with type column (1->0, 2->+1, 3->-1)")
    if meta:
        if not isinstance(meta.get("machine"), str):
            contract_fail.append(f"json 'machine' is {type(meta.get('machine')).__name__}, FORMAT.md requires a string")
        for key in ("name", "N", "eta", "L", "nplus", "nminus", "dt_frame", "n_frames", "n_collisions",
                    "wall_seconds", "Z_CS", "energy_drift_rel"):
            if key not in meta:
                contract_fail.append(f"json lacks required field {key!r}")
        if meta.get("N") != N:
            contract_fail.append(f"json N={meta.get('N')} but trajectory has N={N}")
        if meta.get("n_frames") != F:
            contract_fail.append(f"json n_frames={meta.get('n_frames')} but trajectory has {F} frames")
    if tr["source"] == "npz":
        d = np.load(tr["base"] + ".npz")
        for key, dt_ in (("pos", np.float32), ("types", np.int8), ("charge", np.int8), ("times", np.float64)):
            if d[key].dtype != dt_:
                contract_fail.append(f"npz '{key}' dtype {d[key].dtype} != {np.dtype(dt_)}")
    if F > 1:
        dtf = np.diff(times)
        if not np.allclose(dtf, dtf[0], rtol=0, atol=1e-6):
            contract_fail.append("frame times are not uniformly spaced")
    print("  contract (FORMAT.md): " + ("OK (types 1/2/3 non-decreasing, charges consistent, json fields/machine string, "
                                        "uniform frame times)" if not contract_fail else "VIOLATION: " + "; ".join(contract_fail)))
    out["contract_violations"] = contract_fail

    # ---- temperature / energy
    stationarity_warn = []
    if diag is not None:
        T = diag["T_inst"]
        E = diag["Etot"]
        KE = diag["KE"]
        PE = diag["PE"]
        E0 = E[0]
        drift = (E[-1] - E0) / abs(E0)
        maxdev = np.max(np.abs(E - E0)) / abs(E0)
        rms = np.std(E) / abs(E0)
        slope = np.polyfit(diag["time"], E, 1)[0] / abs(E0) if len(E) > 2 else 0.0
        print(f"  temperature: mean {T.mean():.6f}  std {T.std():.6f}  (kT target 1; KE mean {KE.mean():.4f}, PE mean {PE.mean():.4f})")
        print(f"  energy: E0={E0:.8f} E_end={E[-1]:.8f}  drift (E_end-E0)/|E0| = {drift:.3e}  "
              f"max|E-E0|/|E0| = {maxdev:.3e}  rms/|E0| = {rms:.3e}  slope = {slope:.3e} /tau")
        ncoll = int(diag["n_collisions"][-1])
        tspan = float(diag["time"][-1] - diag["time"][0])
        rate = 2.0 * ncoll / (N * tspan) if tspan > 0 else float("nan")
        print(f"  collisions in production: {ncoll}  ({rate:.3f} per particle per tau)")
        out.update(T_mean=float(T.mean()), T_std=float(T.std()), E0=float(E0), E_end=float(E[-1]),
                   energy_drift_rel=float(drift), energy_max_dev_rel=float(maxdev), energy_rms_rel=float(rms),
                   energy_slope_rel_per_tau=float(slope), n_collisions=ncoll, collision_rate=rate)
        if meta and meta.get("Z_virial") is not None:
            print(f"  Z_virial {meta.get('Z_virial', float('nan')):.4f} vs Z_CS {meta.get('Z_CS', float('nan')):.4f} "
                  f"({100 * meta.get('Z_rel_dev', float('nan')):+.2f}%)  [from json]")

        # ---- stationarity: first half vs second half, linear trends (detects a too-short equilibration)
        npairs_d = diag["n_pairs_pm_1p2"]
        tt = diag["time"]
        nF = len(tt)
        if nF >= 4:
            h = nF // 2
            tspan = float(tt[-1] - tt[0])
            pe_slope = np.polyfit(tt, PE, 1)[0]
            t_slope = np.polyfit(tt, T, 1)[0]
            pe_trend = abs(pe_slope) * tspan
            t_trend = abs(t_slope) * tspan
            pe_ref = max(abs(float(PE.mean())), 1e-12)
            has_pe = np.any(PE != 0)
            print(f"  stationarity (first half | second half of production): "
                  f"T {T[:h].mean():.4f} | {T[h:].mean():.4f}   "
                  f"PE {PE[:h].mean():.3f} | {PE[h:].mean():.3f}   "
                  f"+/- pairs(1.2s) {npairs_d[:h].mean():.2f} | {npairs_d[h:].mean():.2f}")
            print(f"  linear trends over the run: dPE/dt = {pe_slope:+.4f} kT/tau "
                  f"(|dPE/dt|*t_run = {pe_trend:.2f} kT = {100 * pe_trend / pe_ref:.1f}% of |PE|), "
                  f"dT/dt = {t_slope:+.2e} /tau (|dT/dt|*t_run = {100 * t_trend:.2f}% of T)")
            pe_std = float(PE.std())
            if has_pe and pe_trend > PE_TREND_WARN * pe_ref:
                if pe_trend > PE_TREND_SIGMA * pe_std:
                    stationarity_warn.append(f"PE drifts by {100 * pe_trend / pe_ref:.0f}% of |PE| = {pe_trend / pe_std:.1f} "
                                             f"per-frame std devs over the run (> {100 * PE_TREND_WARN:.0f}% and > {PE_TREND_SIGMA:.0f} sigma)")
                else:
                    print(f"  note: PE trend is {100 * pe_trend / pe_ref:.0f}% of |PE| but only {pe_trend / pe_std:.1f} per-frame std devs "
                          f"(fluctuation level for {int((charge != 0).sum())} ions), not flagged")
            if t_trend > T_DRIFT_WARN * float(T.mean()):
                stationarity_warn.append(f"T drifts by {100 * t_trend:.1f}% over the run (> {100 * T_DRIFT_WARN:.0f}%)")
            # initial relaxation: first 2 tau vs the rest, in units of the 2-tau block std
            relax = None
            if has_pe and tspan >= 8.0:
                edges = np.arange(tt[0], tt[-1] + 1e-9, 2.0)
                bm = np.array([PE[(tt >= e0) & (tt < e1)].mean() for e0, e1 in zip(edges[:-1], edges[1:])])
                if len(bm) >= 4:
                    rest = bm[1:]
                    bstd = float(rest.std(ddof=1))
                    relax = (float(bm[0]) - float(rest.mean()), bstd)
                    print(f"  initial relaxation check: PE first 2 tau {bm[0]:.2f} vs rest {rest.mean():.2f} "
                          f"(2-tau block std {bstd:.2f}; difference {relax[0] / bstd if bstd > 0 else 0:+.1f} sigma)")
                    if bstd > 0 and abs(relax[0]) > RELAX_SIGMA * bstd:
                        stationarity_warn.append(f"first 2 tau of production differ from the rest by {relax[0] / bstd:+.1f} sigma "
                                                 f"(> {RELAX_SIGMA:.0f}): pairing still relaxing at the start")
            if has_pe:
                print(f"  T_mean - 1 = {T.mean() - 1:+.4f} (NVE: set by PE at the end of equilibration; expected scatter "
                      f"+-{pe_std / (1.5 * (N - 1)):.4f} from PE fluctuations)")
            out.update(T_first_half=float(T[:h].mean()), T_second_half=float(T[h:].mean()),
                       PE_first_half=float(PE[:h].mean()), PE_second_half=float(PE[h:].mean()),
                       npairs_first_half=float(npairs_d[:h].mean()), npairs_second_half=float(npairs_d[h:].mean()),
                       PE_slope_per_tau=float(pe_slope), T_slope_per_tau=float(t_slope), PE_std=pe_std,
                       PE_first2tau_minus_rest=None if relax is None else relax[0],
                       PE_block2tau_std=None if relax is None else relax[1])
            if stationarity_warn:
                print("  WARNING: production is not stationary (" + "; ".join(stationarity_warn) +
                      "); the charged equilibration (--t-equil-charged) is too short -- use >= 10 tau")
            else:
                print("  stationarity: OK (no significant PE/T trend)")
        else:
            print("  stationarity: not assessed (fewer than 4 frames)")
    else:
        print("  (no diag csv found; skipping temperature/energy)")
    out["stationarity_warnings"] = stationarity_warn

    # ---- pairing analysis (charged runs)
    ions = np.nonzero(charge != 0)[0]
    verdict = "neutral hard spheres: no ions, free flight between collisions"
    if len(ions) > 0:
        q = charge[ions].astype(np.int64)
        fp = np.empty(F)
        npm = np.empty(F, dtype=int)
        largest = np.empty(F, dtype=int)
        nchain = np.empty(F, dtype=int)
        nbranched = np.empty(F, dtype=int)
        nearest = np.empty(F)
        maxsize = len(ions)
        hist = np.zeros(maxsize + 1)          # mean number of clusters of size s per frame
        ion_frac = np.zeros(maxsize + 1)      # fraction of ions in clusters of size s
        for k in range(F):
            st = pairing_frame(pos[k][ions].astype(np.float64), q, L)
            fp[k] = st["frac_paired"]
            npm[k] = st["n_pm_pairs"]
            largest[k] = st["largest"]
            nchain[k] = st["nchain"]
            nbranched[k] = st["nbranched"]
            nearest[k] = st["mean_nearest_opp"]
            for s_ in st["sizes"]:
                hist[s_] += 1
                ion_frac[s_] += s_
        hist /= F
        ion_frac /= F * len(ions)
        print(f"  ions: {len(ions)} (+{int((q > 0).sum())}/-{int((q < 0).sum())}), bond criterion r(+-) < {BOND} sigma")
        print(f"  fraction of ions whose nearest opposite ion is within {BOND} sigma: mean {fp.mean():.3f} "
              f"(min {fp.min():.3f}, max {fp.max():.3f}); mean nearest-opposite distance {nearest.mean():.3f} sigma")
        print(f"  +/- contact pairs per frame: mean {npm.mean():.2f} (min {npm.min()}, max {npm.max()})")
        sizes_present = [s_ for s_ in range(1, maxsize + 1) if hist[s_] > 0]
        print("  cluster-size histogram (mean clusters per frame / fraction of ions):")
        for s_ in sizes_present:
            print(f"      size {s_:2d}: {hist[s_]:7.3f} clusters   {100 * ion_frac[s_]:6.2f}% of ions")
        print(f"  largest cluster per frame: mean {largest.mean():.2f}, max {largest.max()}; "
              f"clusters of size>=3 per frame: chains (max degree 2) {nchain.mean():.3f}, branched {nbranched.mean():.3f}")
        frac_in_3plus = float(ion_frac[3:].sum())
        free = float(ion_frac[1])
        if fp.mean() < 0.25:
            verdict = f"mostly free ions ({100 * free:.0f}% unpaired on average): transient +/- encounters, few lasting Bjerrum pairs"
        elif fp.mean() < 0.6:
            verdict = (f"partially paired: on average {100 * fp.mean():.0f}% of ions sit in a +/- contact pair "
                       f"(pairs form and break; {100 * free:.0f}% free at any instant)")
        else:
            verdict = f"predominantly paired: {100 * fp.mean():.0f}% of ions in +/- contact pairs on average, {100 * free:.0f}% free"
        if frac_in_3plus > 0.2:
            verdict += (f"; frequent chains/clusters ({100 * frac_in_3plus:.0f}% of ions in clusters of >=3, "
                        f"mean largest cluster {largest.mean():.1f})")
        elif frac_in_3plus > 0.05:
            verdict += f"; occasional triplets/short chains ({100 * frac_in_3plus:.0f}% of ions in clusters of >=3)"
        else:
            verdict += "; essentially no chains (clusters of >=3 are rare)"
        out.update(n_ions=int(len(ions)), frac_paired_mean=float(fp.mean()), frac_paired_min=float(fp.min()),
                   frac_paired_max=float(fp.max()), n_pm_pairs_mean=float(npm.mean()),
                   cluster_hist_mean_per_frame={str(s_): float(hist[s_]) for s_ in sizes_present},
                   ion_fraction_by_cluster_size={str(s_): float(ion_frac[s_]) for s_ in sizes_present},
                   largest_cluster_mean=float(largest.mean()), largest_cluster_max=int(largest.max()),
                   chains_per_frame=float(nchain.mean()), branched_per_frame=float(nbranched.mean()),
                   frac_ions_in_clusters_ge3=frac_in_3plus)
    if stationarity_warn:
        verdict += " [WARNING: not stationary, see above]"
    out["verdict"] = verdict
    print(f"  verdict: {verdict}")
    print(f"  overlap check: {'FAILED' if overlap else 'passed'}; contract check: "
          f"{'FAILED' if contract_fail else 'passed'}; analysis wall time {time.time() - t0:.1f}s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    if overlap:
        return 1
    if contract_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
