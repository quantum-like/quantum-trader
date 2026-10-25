#!/usr/bin/env python3
"""Export a trajectory (traj/<name>.npz, FORMAT.md) in formats that VMD and TRAVIS read without surprises.

Reduced units are mapped to pseudo-Angstrom so that the standard bond-recognition heuristics of both programs
see *no* bonds between spheres (1 sigma = --scale Angstrom, default 4.0 A):  Ar = neutral hard sphere,
Na = +e, Cl = -e  (restricted-primitive-model-in-solvent analogy).  Distances in TRAVIS/VMD output are then in
pm/Angstrom and must be divided by scale (resp. 100*scale for pm) to get sigma.

Writes, for traj/<name>.npz:
  traj/<name>_elem.xyz     plain multi-frame XYZ, element labels, comment line = 'Lattice="..." Time=...'  (TRAVIS, VMD)
  traj/<name>.lammpstrj    LAMMPS dump with the periodic box in every frame (VMD reads the box natively)
  traj/<name>_frame0.pdb   first frame with CRYST1 record (VMD topology / box; TRAVIS can read it as well)
Options: --stride N (every Nth frame), --scale A (Angstrom per sigma), --unwrap (continuous coordinates for MSD in TRAVIS).
"""
import argparse, os, sys
import numpy as np

ELEM = {0: "Ar", 1: "Na", -1: "Cl"}
MASS = {"Ar": 39.948, "Na": 22.990, "Cl": 35.453}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="traj/<name>.npz (or the base name)")
    ap.add_argument("--scale", type=float, default=4.0, help="Angstrom per sigma (default 4.0)")
    ap.add_argument("--stride", type=int, default=1, help="write every Nth frame (default 1)")
    ap.add_argument("--unwrap", action="store_true", help="unwrap periodic jumps (continuous trajectories for MSD)")
    ap.add_argument("--outdir", default=None, help="output directory (default: next to the npz)")
    a = ap.parse_args(argv)
    p = a.npz if a.npz.endswith(".npz") else a.npz + ".npz"
    d = np.load(p)
    pos = d["pos"].astype(np.float64); L = float(d["L"]); ch = d["charge"].astype(int); t = d["times"]
    F, N, _ = pos.shape
    if a.unwrap:
        dis = pos[1:] - pos[:-1]; dis -= L * np.round(dis / L)
        pos = np.concatenate([pos[:1], pos[:1] + np.cumsum(dis, axis=0)], axis=0)
    sel = range(0, F, a.stride)
    s = a.scale; Ls = L * s
    base = os.path.splitext(os.path.basename(p))[0]
    outdir = a.outdir or os.path.dirname(os.path.abspath(p))
    os.makedirs(outdir, exist_ok=True)
    elem = np.array([ELEM[int(c)] for c in ch])
    types = np.where(ch == 0, 1, np.where(ch > 0, 2, 3))

    xyz = os.path.join(outdir, base + "_elem.xyz")
    with open(xyz, "w") as f:
        for k in sel:
            f.write(f"{N}\n")
            f.write(f'Lattice="{Ls:.6f} 0 0 0 {Ls:.6f} 0 0 0 {Ls:.6f}" Properties=species:S:1:pos:R:3 Time={t[k]:.6f} Frame={k} '
                    f'Note="1 sigma = {s:g} A; Ar=neutral hard sphere, Na=+e, Cl=-e"\n')
            q = pos[k] * s
            for i in range(N):
                f.write(f"{elem[i]:2s} {q[i,0]:12.6f} {q[i,1]:12.6f} {q[i,2]:12.6f}\n")

    lmp = os.path.join(outdir, base + ".lammpstrj")
    with open(lmp, "w") as f:
        for k in sel:
            q = pos[k] * s
            f.write("ITEM: TIMESTEP\n%d\n" % k)
            f.write("ITEM: NUMBER OF ATOMS\n%d\n" % N)
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 %.6f\n0 %.6f\n0 %.6f\n" % (Ls, Ls, Ls))
            f.write("ITEM: ATOMS id type element x y z q\n")
            for i in range(N):
                f.write(f"{i+1} {types[i]} {elem[i]} {q[i,0]:.6f} {q[i,1]:.6f} {q[i,2]:.6f} {ch[i]:d}\n")

    pdb = os.path.join(outdir, base + "_frame0.pdb")
    with open(pdb, "w") as f:
        f.write("REMARK   hard spheres (Ar) with +e (Na) / -e (Cl); 1 sigma = %g A; reduced units in the npz\n" % s)
        f.write("CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n" % (Ls, Ls, Ls))
        q = pos[0] * s
        for i in range(N):
            e = elem[i]; resn = {"Ar": "HS", "Na": "POS", "Cl": "NEG"}[e]
            f.write("HETATM%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n" %
                    ((i + 1) % 100000, e, resn, 1, q[i, 0], q[i, 1], q[i, 2], e.upper()))
        f.write("END\n")
    print(f"wrote {xyz}\n      {lmp}\n      {pdb}\n  frames {len(sel)} (stride {a.stride}), N={N}, box {Ls:.3f} A (= {L:.4f} sigma), scale {s:g} A/sigma"
          + (", unwrapped" if a.unwrap else ""))

if __name__ == "__main__":
    main()
