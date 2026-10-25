#!/usr/bin/env python3
"""analyze.py -- validate and characterise an hsdmd trajectory.

    python sim/analyze.py traj/<name>          (reads traj/<name>.xyz if present, else .npz)
    python sim/analyze.py traj/<name>.npz      (force a specific source)

Reports: minimum pair distance over ALL frames (hard-core check), temperature and
energy drift from <name>_diag.csv, and for charged runs a per-frame pairing /
cluster analysis of the ions (1.2 sigma +/- bond criterion).
Exit status is non-zero if any hard-core overlap is found.
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

    # ---- temperature / energy
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
        if meta:
            print(f"  Z_virial {meta.get('Z_virial', float('nan')):.4f} vs Z_CS {meta.get('Z_CS', float('nan')):.4f} "
                  f"({100 * meta.get('Z_rel_dev', float('nan')):+.2f}%)  [from json]")
    else:
        print("  (no diag csv found; skipping temperature/energy)")

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
    out["verdict"] = verdict
    print(f"  verdict: {verdict}")
    print(f"  overlap check: {'FAILED' if overlap else 'passed'}; analysis wall time {time.time() - t0:.1f}s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 1 if overlap else 0


if __name__ == "__main__":
    sys.exit(main())
