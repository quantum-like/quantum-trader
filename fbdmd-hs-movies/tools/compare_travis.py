#!/usr/bin/env python3
"""Overlay a TRAVIS RDF csv (semicolon separated, distance in pm) on the g(r) computed here from traj/<run>.npz.

  python tools/compare_travis.py travis/out_hs_charges_20p20m/rdf_Na_Cl*.csv --pair Na-Cl [--npz traj/hs_charges_20p20m.npz] [--scale 4.0]
Pairs: Na-Cl (+/-), Na-Na, Cl-Cl, Ar-Ar, Na-Ar, Cl-Ar.  Writes report/figures/travis_rdf_<pair>.png.
"""
import argparse, glob, os, sys
import numpy as np

def read_travis_csv(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.replace(",", ".").split(";")]
            try:
                xs.append(float(parts[0])); ys.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(ys)

def gr_from_npz(npz, pair, rmax, nbins=120, stride=1):
    d = np.load(npz); pos = d["pos"].astype(np.float64); L = float(d["L"]); ch = d["charge"].astype(int)
    lab = {"Ar": ch == 0, "Na": ch > 0, "Cl": ch < 0}
    a, b = pair.split("-")
    ia, ib = np.where(lab[a])[0], np.where(lab[b])[0]
    same = a == b
    edges = np.linspace(0, rmax, nbins + 1); hist = np.zeros(nbins); nfr = 0
    for k in range(0, pos.shape[0], stride):
        pa, pb = pos[k, ia], pos[k, ib]
        dr = pa[:, None, :] - pb[None, :, :]; dr -= L * np.round(dr / L)
        r = np.sqrt((dr ** 2).sum(-1))
        if same:
            r = r[np.triu_indices(len(ia), 1)]
        hist += np.histogram(r.ravel(), bins=edges)[0]; nfr += 1
    rho_b = len(ib) / L ** 3
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    npairs = len(ia) * (len(ib) - 1) / 2 if same else len(ia) * len(ib)
    norm = nfr * (len(ia) if not same else len(ia)) * rho_b * shell * (0.5 if same else 1.0)
    g = hist / norm
    return 0.5 * (edges[1:] + edges[:-1]), g

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="TRAVIS rdf csv file(s) or glob")
    ap.add_argument("--pair", default="Na-Cl"); ap.add_argument("--npz", default="traj/hs_charges_20p20m.npz")
    ap.add_argument("--scale", type=float, default=4.0, help="Angstrom per sigma used in the export (default 4)")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args(argv)
    files = [f for pat in a.csv for f in glob.glob(pat)]
    if not files:
        sys.exit("no TRAVIS csv found")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report"))
    from figstyle import apply_style, savefig, FULL_W, C_ACCENT, C_PLUS
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W * 0.6, 3.0))
    rmax = 0
    for f in files:
        x, y = read_travis_csv(f); x_sigma = x / (100 * a.scale)   # pm -> sigma
        ax.plot(x_sigma, y, color=C_PLUS, lw=1.2, label=f"TRAVIS: {os.path.basename(f)}")
        rmax = max(rmax, x_sigma.max())
    r, g = gr_from_npz(a.npz, a.pair, min(rmax, 5.5), stride=a.stride)
    ax.plot(r, g, color=C_ACCENT, lw=1.2, ls="--", label=f"this work (numpy): g_{a.pair}(r)")
    ax.set_xlabel("r / σ"); ax.set_ylabel("g(r)"); ax.set_xlim(0.8, min(rmax, 5.5)); ax.legend()
    out = savefig(fig, f"travis_rdf_{a.pair.replace('-', '')}")
    print("wrote", out)

if __name__ == "__main__":
    main()
