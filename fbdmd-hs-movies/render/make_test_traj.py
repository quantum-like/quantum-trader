#!/usr/bin/env python
"""Write a small synthetic trajectory for renderer tests (FORMAT.md layout):

  traj/_test/synthetic.npz  traj/_test/synthetic.xyz  traj/_test/synthetic.json

1000 spheres on a jittered simple-cubic lattice in L = 11.444 (10 x 10 x 10, spacing 1.1444,
jitter +-0.05 -> min centre distance 1.044 > sigma, i.e. no overlaps in frame 0), 40 of them
marked as ions (20 +, 20 -), `--frames` frames of slow ballistic drift (Maxwell-Boltzmann
velocities at kT = 1, no collisions!) wrapped into the periodic box. Test data only: later frames
may contain overlaps because nothing collides.

Usage:  python render/make_test_traj.py [--out-dir traj/_test] [--name synthetic] [--frames 12]
"""
import argparse
import math
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajio import write_json, write_npz, write_xyz  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def min_image_min_distance(p, L):
    d = p[:, None, :] - p[None, :, :]
    d -= L * np.rint(d / L)
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    return math.sqrt(r2.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "traj", "_test"))
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--dt-frame", type=float, default=0.025)
    ap.add_argument("--L", type=float, default=11.444)
    ap.add_argument("--n-side", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--nplus", type=int, default=20)
    ap.add_argument("--nminus", type=int, default=20)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L, n = args.L, args.n_side
    N = n ** 3
    a = L / n
    g = (np.arange(n) + 0.5) * a
    pos0 = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    pos0 = pos0 + rng.uniform(-args.jitter, args.jitter, pos0.shape)
    pos0 %= L
    dmin = min_image_min_distance(pos0, L)
    assert dmin > 1.0, f"lattice overlaps: min distance {dmin}"

    types = np.ones(N, np.int8)
    ions = rng.choice(N, args.nplus + args.nminus, replace=False)
    types[ions[:args.nplus]] = 2
    types[ions[args.nplus:]] = 3
    charge = np.where(types == 2, 1, np.where(types == 3, -1, 0)).astype(np.int8)

    vel = rng.normal(0.0, math.sqrt(args.kT), (N, 3))
    vel -= vel.mean(0)
    times = np.arange(args.frames) * args.dt_frame
    pos = (pos0[None] + vel[None] * times[:, None, None]) % L

    eta = N * math.pi / 6.0 / L ** 3
    meta = {"name": args.name, "N": N, "eta": round(eta, 5), "L": L, "kT": args.kT,
            "nplus": int((types == 2).sum()), "nminus": int((types == 3).sum()),
            "bjerrum": 0.0, "coulomb": "none",
            "integrator": "synthetic ballistic drift, no collisions (renderer test data only)",
            "dt": args.dt_frame, "dt_frame": args.dt_frame, "t_equil_neutral": 0.0, "t_equil_charged": 0.0,
            "t_run": float(times[-1]), "n_frames": args.frames, "seed": args.seed, "n_collisions": 0,
            "wall_seconds": 0.0, "min_dist_frame0": round(dmin, 5),
            "machine": f"{platform.system()} {platform.machine()} ({platform.processor() or 'cpu'})"}

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    write_npz(base + ".npz", pos, types, charge, times, L, meta)
    write_xyz(base + ".xyz", pos, types, charge, times, L)
    write_json(base + ".json", meta)
    print(f"wrote {base}.npz / .xyz / .json: N={N}, L={L}, eta={eta:.4f}, frames={args.frames}, "
          f"dt_frame={args.dt_frame}, ions {meta['nplus']}+/{meta['nminus']}-, min dist frame0 = {dmin:.4f}")
    print(f"max displacement over run: {np.abs(vel).max() * times[-1]:.3f} sigma")


if __name__ == "__main__":
    main()
