#!/usr/bin/env python3
"""order_test.py -- global energy-error order of the charged integrators (HLL 2000, Fig. 4 analogue).

Runs the hsdmd binary on the production system (N=1000, eta=0.35, 20+/20-, l_B=5 by default) for a
short NVE production of t_run (2 tau) at dt in {0.02, 0.01, 0.005, 0.0025} for each integrator
(cv-phys = the default integrator, cv = HLL exactly, kdk-phys, naive),
records max_t |E(t)-E(0)|/N and the end drift (E_end-E_0)/N from the per-frame diagnostics, fits
log-log slopes, prints a table and writes <outdir>/<name>_order.csv.

The same start is used for every (dt, integrator): the neutral equilibration is integrator-independent
and, with --t-equil-charged 0 (the default here), production starts right after the charges are
assigned.  The order test therefore does NOT run from an equilibrated ionic state (the ions fall into
pairs during the 2 tau); that is irrelevant for the convergence order but means the collision
statistics differ from the equilibrated movie runs.  --t-equil-charged > 0 is allowed (the equilibration
is then integrated with the same dt and thermostatted, so the production start differs between dts;
--t-equil-charged 15 gives the equilibrated, paired state of the movie runs).

Frame spacing is --dt-frame (0.02 by default) so that every dt divides it (the binary rounds dt to an
integer fraction of dt_frame).

    . .venv/bin/activate
    python sim/order_test.py                      # 16 runs, ~10 s on 4 cpus
    python sim/order_test.py --coulomb dsf        # smooth Coulomb model: no Ewald r_c floor at small dt
    python sim/order_test.py --N 216 --nplus 8 --nminus 8 --t-run 2 --jobs 4
    make -C sim order
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sim import PROJECT_ROOT, ensure_binary  # noqa: E402


def run_one(binary, a, integ, dt, outdir):
    tag = f"{os.path.basename(a.name)}_{integ}_dt{dt:g}"
    diag = os.path.join(outdir, tag + "_diag.csv")
    summ = os.path.join(outdir, tag + ".json")
    cmd = [binary, "--name", tag, "--N", str(a.N), "--eta", repr(a.eta), "--nplus", str(a.nplus),
           "--nminus", str(a.nminus), "--bjerrum", repr(a.bjerrum), "--coulomb", a.coulomb,
           "--integrator", integ, "--cv-q1", repr(a.cv_q1), "--cv-q2", repr(a.cv_q2), "--dt", repr(dt),
           "--t-equil-neutral", repr(a.t_equil_neutral), "--t-equil-charged", repr(a.t_equil_charged),
           "--t-run", repr(a.t_run), "--dt-frame", repr(a.dt_frame), "--seed", str(a.seed),
           "--diag", diag, "--summary", summ, "--quiet"]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\nexit {rc.returncode}\n{rc.stderr[-2000:]}")
    d = np.genfromtxt(diag, delimiter=",", names=True)
    with open(summ) as f:
        m = json.load(f)
    E = d["Etot"]
    N = a.N
    return dict(integrator=integ, dt_requested=dt, dt=m["dt"], max_abs_dE_per_N=float(np.max(np.abs(E - E[0])) / N),
                end_drift_per_N=float((E[-1] - E[0]) / N), rms_dE_per_N=float(np.std(E - E[0]) / N),
                E0=float(E[0]), n_frames=len(E), n_collisions=int(m["n_collisions"]),
                n_substeps=int(m.get("n_verlet_substeps", 0)), n_vmin=int(m.get("n_cv_vmin_bias", 0)),
                min_dist=float(m["min_dist_production"]), wall=float(m["wall_seconds_production"]))


def slope(xs, ys):
    xs, ys = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(xs, ys, 1)[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="hs_charges_20p20m")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--eta", type=float, default=0.35)
    ap.add_argument("--nplus", type=int, default=20)
    ap.add_argument("--nminus", type=int, default=20)
    ap.add_argument("--bjerrum", type=float, default=5.0)
    ap.add_argument("--coulomb", choices=["ewald", "dsf"], default="ewald")
    ap.add_argument("--cv-q1", type=float, default=1.0)
    ap.add_argument("--cv-q2", type=float, default=1.5)
    ap.add_argument("--dts", type=float, nargs="+", default=[0.02, 0.01, 0.005, 0.0025])
    ap.add_argument("--integrators", nargs="+", default=["cv-phys", "cv", "kdk-phys", "naive"])
    ap.add_argument("--t-equil-neutral", type=float, default=5.0)
    ap.add_argument("--t-equil-charged", type=float, default=0.0,
                    help="0 (default) gives the identical start for every dt/integrator; see the module docstring")
    ap.add_argument("--t-run", type=float, default=2.0)
    ap.add_argument("--dt-frame", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="traj", help="relative to the project root unless absolute; "
                    "per-run files go to <outdir>/_test/order/")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--cxx", default=None)
    a = ap.parse_args(argv)
    for dt in a.dts:
        n = a.dt_frame / dt
        if abs(n - round(n)) > 1e-9:
            ap.error(f"dt={dt} does not divide dt_frame={a.dt_frame}; the binary would round it")

    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(PROJECT_ROOT, a.outdir)
    rundir = os.path.join(outdir, "_test", "order")
    os.makedirs(rundir, exist_ok=True)
    binary = ensure_binary(force=a.rebuild, cxx=a.cxx)

    jobs = [(integ, dt) for integ in a.integrators for dt in a.dts]
    print(f"[order_test] N={a.N} eta={a.eta} +{a.nplus}/-{a.nminus} l_B={a.bjerrum} {a.coulomb}; t_equil_charged={a.t_equil_charged} "
          f"t_run={a.t_run} dt_frame={a.dt_frame}; {len(jobs)} runs on {a.jobs} threads", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda j: run_one(binary, a, j[0], j[1], rundir), jobs))

    csv_path = os.path.join(outdir, a.name + "_order.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["integrator", "dt_requested", "dt", "max_abs_dE_per_N", "end_drift_per_N", "rms_dE_per_N", "E0", "n_frames",
              "n_collisions", "n_substeps", "n_vmin", "min_dist", "wall"]
    slopes = {}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["slope_max", "slope_end"])
        w.writeheader()
        for integ in a.integrators:
            rs = [r for r in rows if r["integrator"] == integ]
            smax = slope([r["dt"] for r in rs], [max(r["max_abs_dE_per_N"], 1e-300) for r in rs]) if len(rs) >= 2 else math.nan
            send = slope([r["dt"] for r in rs], [max(abs(r["end_drift_per_N"]), 1e-300) for r in rs]) if len(rs) >= 2 else math.nan
            slopes[integ] = (smax, send)
            for r in rs:
                w.writerow({**r, "slope_max": smax, "slope_end": send})

    print(f"\n  max_t |E(t)-E(0)|/N and (E_end-E_0)/N in kT over t_run={a.t_run} tau, N={a.N}:")
    hdr = "  {:9s}".format("dt") + "".join(f" {integ:>22s}" for integ in a.integrators)
    print(hdr)
    for dt in a.dts:
        line = f"  {dt:<9g}"
        for integ in a.integrators:
            r = next(x for x in rows if x["integrator"] == integ and x["dt_requested"] == dt)
            line += f"  {r['max_abs_dE_per_N']:9.3e} ({r['end_drift_per_N']:+8.1e})"
        print(line)
    line = "  {:9s}".format("slope")
    for integ in a.integrators:
        line += f"  {slopes[integ][0]:9.2f} ({slopes[integ][1]:+8.2f})"
    print(line + "    (log-log fit of max |dE|/N, end drift in parentheses)")
    print("  collisions / Verlet sub-steps / vmin events / min_dist / wall at the smallest dt:")
    for integ in a.integrators:
        r = next(x for x in rows if x["integrator"] == integ and x["dt_requested"] == a.dts[-1])
        print(f"    {integ:9s} {r['n_collisions']} / {r['n_substeps']} / {r['n_vmin']} / {r['min_dist']:.9f} / {r['wall']:.1f}s")
    print(f"  wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
