#!/usr/bin/env python3
"""run_sim.py -- build (if needed) and run the hsdmd trajectory generator, then
convert its extended-XYZ output to the npz/json/csv files described in FORMAT.md.

Example (from the project root, inside the venv):
    python sim/run_sim.py --name hs_neutral
    python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20

Outputs:  <outdir>/<name>.xyz   extended XYZ (canonical, OVITO-readable)
          <outdir>/<name>.npz   pos float32 (F,N,3), types, charge, times, L, meta
          <outdir>/<name>.json  summary statistics + machine info
          <outdir>/<name>_diag.csv  per-frame diagnostics (written by the binary)
"""
import argparse
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SIM_DIR)
BINARY = os.path.join(SIM_DIR, "hsdmd")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def ensure_binary(force=False, cxx=None):
    """Run make in sim/ when the binary is missing or older than its sources."""
    srcs = [os.path.join(SIM_DIR, "hsdmd.cpp"), os.path.join(SIM_DIR, "Makefile")]
    stale = force or not os.path.exists(BINARY) or any(
        os.path.getmtime(s) > os.path.getmtime(BINARY) for s in srcs)
    if stale:
        cmd = ["make", "-C", SIM_DIR, "hsdmd"]
        if force:
            subprocess.run(["make", "-C", SIM_DIR, "clean"], check=True)
        if cxx:
            cmd.append(f"CXX={cxx}")
        print("[run_sim] building:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return BINARY


# ---------------------------------------------------------------------------
# extended XYZ reader (shared with analyze.py)
# ---------------------------------------------------------------------------
_LAT_RE = re.compile(r'Lattice="([^"]+)"')
_TIME_RE = re.compile(r"Time=([-+0-9.eE]+)")


def read_xyz(path, dtype=np.float64, verbose=False):
    """Read the ext-XYZ written by hsdmd.

    Returns dict(pos=(F,N,3) dtype, types=int8 (N,), charge=int8 (N,), times=(F,), L=float).
    Species/type/charge are taken from frame 0 (particle order is frame invariant).
    """
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    N = int(lines[0])
    per = N + 2
    nframes = len(lines) // per
    if nframes * per != len(lines):
        raise ValueError(f"{path}: {len(lines)} lines is not a multiple of N+2={per}")
    pos = np.empty((nframes, N, 3), dtype=dtype)
    times = np.empty(nframes, dtype=np.float64)
    L = None
    types = charge = None
    t0 = time.time()
    for k in range(nframes):
        base = k * per
        hdr = lines[base + 1].decode()
        if L is None:
            lat = [float(x) for x in _LAT_RE.search(hdr).group(1).split()]
            L = lat[0]
        times[k] = float(_TIME_RE.search(hdr).group(1))
        block = b"\n".join(lines[base + 2: base + 2 + N])
        cols = (1, 2, 3, 5, 6) if k == 0 else (1, 2, 3)
        arr = np.loadtxt(io.BytesIO(block), usecols=cols, dtype=np.float64, ndmin=2)
        if arr.shape[0] != N:
            raise ValueError(f"frame {k}: expected {N} atoms, got {arr.shape[0]}")
        pos[k] = arr[:, :3]
        if k == 0:
            types = arr[:, 3].astype(np.int8)
            charge = arr[:, 4].astype(np.int8)
        if verbose and (k % 100 == 0):
            print(f"[read_xyz] frame {k}/{nframes} ({time.time() - t0:.1f}s)", flush=True)
    return dict(pos=pos, types=types, charge=charge, times=times, L=L)


# ---------------------------------------------------------------------------
# machine info
# ---------------------------------------------------------------------------
def cpu_model():
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_info():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "node": platform.node(),
    }


# ---------------------------------------------------------------------------
# argument validation (mirrors the checks in hsdmd.cpp so bad input fails before the binary runs)
# ---------------------------------------------------------------------------
def _typed(conv, pred, what):
    def f(text):
        try:
            v = conv(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a valid {conv.__name__}")
        if not pred(v):
            raise argparse.ArgumentTypeError(f"{text!r}: must be {what}")
        return v
    f.__name__ = conv.__name__
    return f


positive_int = _typed(int, lambda v: v >= 1, ">= 1")
nonneg_int = _typed(int, lambda v: v >= 0, ">= 0")
positive_float = _typed(float, lambda v: v > 0 and math.isfinite(v), "> 0 (finite)")
nonneg_float = _typed(float, lambda v: v >= 0 and math.isfinite(v), ">= 0 (finite)")
packing_fraction = _typed(float, lambda v: 0 < v < 0.74, "in (0, 0.74)")


def load_summary(summary_tmp, keep_as):
    """Parse the binary's summary JSON; on a parse error keep the file under `keep_as` and return None."""
    try:
        with open(summary_tmp) as f:
            text = f.read()
        meta = json.loads(text)
    except FileNotFoundError:
        print(f"[run_sim] error: simulator exited 0 but wrote no summary file {summary_tmp}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        bad = lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else "<out of range>"
        print(f"[run_sim] error: summary JSON from the simulator is invalid: {e.msg} "
              f"(line {e.lineno}, col {e.colno}): {bad.strip()!r}", file=sys.stderr)
        os.replace(summary_tmp, keep_as)
        print(f"[run_sim] the offending file was kept as {keep_as}", file=sys.stderr)
        return None
    os.remove(summary_tmp)
    return meta


def fmt(x, spec):
    """format() that tolerates JSON null (None) for quantities undefined in degenerate runs."""
    return "null" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name; may contain a subdirectory, e.g. _test/neutral_small")
    ap.add_argument("--N", type=positive_int, default=1000)
    ap.add_argument("--eta", type=packing_fraction, default=0.35)
    ap.add_argument("--nplus", type=nonneg_int, default=0)
    ap.add_argument("--nminus", type=nonneg_int, default=0)
    ap.add_argument("--bjerrum", type=nonneg_float, default=5.0)
    ap.add_argument("--coulomb", choices=["ewald", "dsf"], default="ewald")
    ap.add_argument("--integrator", choices=["cv-phys", "cv", "kdk-phys", "naive"], default="cv-phys",
                    help="charged-run integrator: cv-phys (default) = collision Verlet of Houndonougbo, Laird & "
                         "Leimkuhler 2000 plus the physical-velocity rule for the outer-kick force F2 at collisions "
                         "(second order on the production system, best energy conservation); cv = HLL 2000 exactly "
                         "(first order in this ion/neutral mixture, see README_sim.md); kdk-phys = kick-drift-kick "
                         "with the physical-velocity collision rule (hsdmd <= 1.1 default); naive = HLL eq. 9 "
                         "(O(dt), for comparison)")
    ap.add_argument("--cv-q1", type=positive_float, default=1.0, help="cv splitting radius q1 in sigma (>= 1)")
    ap.add_argument("--cv-q2", type=positive_float, default=1.5, help="cv splitting radius q2 in sigma (> q1, <= L/2)")
    ap.add_argument("--dt", type=positive_float, default=0.005)
    ap.add_argument("--t-equil-neutral", type=nonneg_float, default=5.0)
    ap.add_argument("--t-equil-charged", type=nonneg_float, default=15.0,
                    help="charged (thermostatted) equilibration in tau; the +/- pairing relaxation time is "
                         "4-6 tau, so >= 10 is recommended for equilibrated production (default 15)")
    ap.add_argument("--t-run", type=nonneg_float, default=18.0)
    ap.add_argument("--dt-frame", type=positive_float, default=0.025)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="traj", help="relative to the project root unless absolute")
    ap.add_argument("--rebuild", action="store_true", help="force make clean && make")
    ap.add_argument("--cxx", default=None, help="compiler to pass to make (CXX=...)")
    ap.add_argument("--quiet", action="store_true", help="suppress the simulator's per-tau progress")
    a = ap.parse_args(argv)
    if a.nplus + a.nminus > a.N:
        ap.error(f"--nplus + --nminus = {a.nplus + a.nminus} exceeds --N = {a.N}")
    if not (a.cv_q1 >= 1.0 and a.cv_q2 > a.cv_q1):
        ap.error(f"need 1 <= --cv-q1 < --cv-q2 (got {a.cv_q1}, {a.cv_q2})")
    if a.t_run < a.dt_frame:
        print(f"[run_sim] warning: --t-run {a.t_run} < --dt-frame {a.dt_frame}: a single frame will be written and "
              "Z/collision rate are undefined (null)", file=sys.stderr)

    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(PROJECT_ROOT, a.outdir)
    base = os.path.join(outdir, a.name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    xyz, npz, js, diag = base + ".xyz", base + ".npz", base + ".json", base + "_diag.csv"
    summary_tmp = base + "_summary_tmp.json"

    binary = ensure_binary(force=a.rebuild, cxx=a.cxx)
    cmd = [binary, "--name", a.name, "--N", str(a.N), "--eta", repr(a.eta), "--nplus", str(a.nplus),
           "--nminus", str(a.nminus), "--bjerrum", repr(a.bjerrum), "--coulomb", a.coulomb,
           "--integrator", a.integrator, "--cv-q1", repr(a.cv_q1), "--cv-q2", repr(a.cv_q2), "--dt", repr(a.dt),
           "--t-equil-neutral", repr(a.t_equil_neutral), "--t-equil-charged", repr(a.t_equil_charged),
           "--t-run", repr(a.t_run), "--dt-frame", repr(a.dt_frame), "--seed", str(a.seed),
           "--xyz", xyz, "--diag", diag, "--summary", summary_tmp]
    if a.quiet:
        cmd.append("--quiet")
    print("[run_sim]", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[run_sim] simulator failed with exit code {rc}", file=sys.stderr)
        return rc
    wall_sim = time.time() - t0

    meta = load_summary(summary_tmp, base + "_summary_INVALID.json")
    if meta is None:
        return 1
    mi = machine_info()
    meta["machine_info"] = mi
    meta["machine"] = f"{mi['cpu_model']} ({mi['system']} {mi['release']}, {mi['machine']}, {mi['cpu_count']} cpus, python {mi['python']})"
    meta["machine_str"] = meta["machine"]
    meta["files"] = {"xyz": os.path.relpath(xyz, PROJECT_ROOT), "npz": os.path.relpath(npz, PROJECT_ROOT),
                     "json": os.path.relpath(js, PROJECT_ROOT), "diag": os.path.relpath(diag, PROJECT_ROOT)}
    meta["command"] = "python sim/run_sim.py " + " ".join(sys.argv[1:] if argv is None else argv)
    meta["units"] = "sigma=1, m=1, kT=1, tau=sigma*sqrt(m/kT); Coulomb U = bjerrum*kT*sum q_i q_j phi(r_ij)"

    # convert xyz -> npz
    t1 = time.time()
    tr = read_xyz(xyz, dtype=np.float64)
    if tr["pos"].shape[0] != meta["n_frames"]:
        print(f"[run_sim] warning: xyz has {tr['pos'].shape[0]} frames, summary says {meta['n_frames']}", file=sys.stderr)
    meta["wall_seconds_convert"] = time.time() - t1
    meta["wall_seconds_total_wrapper"] = time.time() - t0
    meta_json = json.dumps(meta, indent=1, sort_keys=True)
    np.savez_compressed(npz, pos=tr["pos"].astype(np.float32), types=tr["types"].astype(np.int8),
                        charge=tr["charge"].astype(np.int8), times=tr["times"], L=np.float64(tr["L"]),
                        meta=np.array(meta_json))
    with open(js, "w") as f:
        f.write(meta_json + "\n")

    nion = meta["nplus"] + meta["nminus"]
    para = (f"[run_sim] {a.name}: N={meta['N']} eta={meta['eta']} L={meta['L']:.4f} "
            f"{'+' + str(meta['nplus']) + '/-' + str(meta['nminus']) + ' ions, l_B=' + str(meta['bjerrum']) + ', ' + meta['coulomb'] if nion else 'neutral'}; "
            f"integrator {meta['integrator']}"
            f"{' dt=' + format(meta['dt'], '.6g') if nion else ''}"
            f"{' (' + str(meta['n_verlet_substeps']) + ' Verlet sub-steps, ' + str(meta['n_cv_vmin_bias']) + ' vmin events, ' + str(meta.get('n_drift_fix', 0)) + ' drift fixes)' if nion and meta.get('integrator_id', '').startswith('cv') else ''}; "
            f"{meta['n_frames']} frames at dt_frame={meta['dt_frame']} over t_run={meta['t_run']} tau. "
            f"Production collisions: {meta['n_collisions']} "
            f"({fmt(meta['collision_rate_per_particle_per_tau'], '.2f')} per particle per tau, Enskog {meta['enskog_rate_per_particle_per_tau']:.2f}); "
            f"Z_virial={fmt(meta['Z_virial'], '.4f')} vs Carnahan-Starling {meta['Z_CS']:.4f} "
            f"({fmt(None if meta['Z_rel_dev'] is None else 100 * meta['Z_rel_dev'], '+.2f')}%); "
            f"T_mean={meta['T_mean']:.5f}; total-energy drift (E_end-E_0)/|E_0|={fmt(meta['energy_drift_rel'], '.3e')}, "
            f"max |E-E_0|/|E_0|={fmt(meta['energy_max_dev_rel'], '.3e')}; |P|={meta['momentum_abs_final']:.2e}; "
            f"min pair distance in production {meta['min_dist_production']:.9f}; "
            f"wall: sim {meta['wall_seconds']:.2f}s (equil-neutral {meta['wall_seconds_equil_neutral']:.2f}s, "
            f"equil-charged {meta['wall_seconds_equil_charged']:.2f}s, production {meta['wall_seconds_production']:.2f}s), "
            f"xyz->npz {meta['wall_seconds_convert']:.2f}s, wrapper total {wall_sim + meta['wall_seconds_convert']:.2f}s "
            f"on {meta['machine_str']}. Files: {xyz}, {npz}, {js}, {diag}")
    print(para, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
