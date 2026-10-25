#!/usr/bin/env python3
"""Assemble NOTES.md (one page) from traj/*.json metadata, movie probes and machine info."""
import json, os, platform, subprocess, sys, datetime, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

def chip():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()

def probe(mp4):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
            "-of", "json", mp4], text=True)
        j = json.loads(out); s = j["streams"][0]; f = j["format"]
        return f'{s["codec_name"]} {s["width"]}x{s["height"]} @ {s["r_frame_rate"]} fps, {float(f["duration"]):.1f} s, {int(f["size"])/1e6:.1f} MB'
    except Exception as e:
        return f"(ffprobe unavailable: {e})"

def main():
    metas = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "traj", "*.json"))):
        with open(p) as f:
            m = json.load(f)
        metas[m.get("name", os.path.basename(p)[:-5])] = m
    lines = []
    lines.append("# NOTES — hard-sphere + point-charge trajectory movies\n")
    lines.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {platform.node()} "
                 f"({platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, {chip()}).\n")
    lines.append("Units: σ = 1, m = 1, kT = 1, τ = σ√(m/kT). Charges ±e on a subset of spheres; Coulomb coupling given as l_B/σ.\n")
    lines.append("\n## Runs\n")
    lines.append("| run | N | η | L/σ | kT | +/− | l_B/σ | Coulomb | integrator | Δt | frames (Δt_frame) | t_run | collisions | Z (virial) / Z_CS | ΔE/E (NVE) | sim wall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in metas.items():
        charged = (m.get("nplus", 0) + m.get("nminus", 0)) > 0
        lines.append("| {name} | {N} | {eta} | {L:.3f} | {kT} | {np}/{nm} | {lb} | {coul} | {integ} | {dt} | {nf} ({dtf}) | {trun} | {ncoll} | {Z} / {ZCS} | {drift} | {wall} |".format(
            name=name, N=m.get("N"), eta=m.get("eta"), L=float(m.get("L", float("nan"))), kT=m.get("kT", 1.0),
            np=m.get("nplus", 0), nm=m.get("nminus", 0),
            lb=m.get("bjerrum") if charged else "—", coul=m.get("coulomb") if charged else "—",
            integ=m.get("integrator", "EDMD"), dt=m.get("dt") if charged else "event-driven",
            nf=m.get("n_frames"), dtf=m.get("dt_frame"), trun=m.get("t_run"), ncoll=m.get("n_collisions"),
            Z=f'{m.get("Z_virial", float("nan")):.3f}', ZCS=f'{m.get("Z_CS", float("nan")):.3f}',
            drift=(f'{m.get("energy_drift_rel", float("nan")):.1e}' if charged else "exact (KE const)"),
            wall=f'{m.get("wall_seconds", float("nan")):.0f} s'))
    eq = {name: (m.get("t_equil_neutral"), m.get("t_equil_charged")) for name, m in metas.items()}
    lines.append("\nEquilibration (velocity rescaling to kT = 1 every 0.1 τ, then NVE production): " +
                 "; ".join(f"{n}: {a} τ neutral" + (f" + {b} τ with charges on" if (metas[n].get('nplus',0)+metas[n].get('nminus',0)) > 0 else "")
                           for n, (a, b) in eq.items()) + ". Same seed ⇒ identical neutral start for both runs.\n")
    ex = next((m for m in metas.values() if (m.get("nplus", 0) + m.get("nminus", 0)) > 0), None)
    if ex:
        lines.append(f"Charged run details: Coulomb = {ex.get('coulomb')} (Ewald α = 5.6/L, |n|∞ ≤ 6 for 40 ions), "
                     f"integrator = {ex.get('integrator')}, {ex.get('n_verlet_substeps', '?')} Verlet sub-steps, "
                     f"T_mean = {ex.get('T_mean', float('nan')):.4f}, max |δE|/|E| = {ex.get('energy_max_dev_rel', ex.get('energy_drift_rel', float('nan'))):.1e}, "
                     f"min pair distance {ex.get('min_dist_production', float('nan')):.9f} σ (no overlap).\n")
    ot = os.path.join(ROOT, "notes", "order_test.txt")
    if os.path.exists(ot):
        lines.append("\n## Integrator energy-order test (max |E(t)−E₀|/N in kT over 2 τ, equilibrated start)\n")
        body = [l for l in open(ot).read().splitlines() if l.strip().startswith(("dt", "0.0", "slope"))]
        lines.append("```\n" + "\n".join(body) + "\n```")
        lines.append("cv-phys = shipped default (HLL collision Verlet + physical-velocity rule for F₂ at collisions); cv = literal HLL; "
                     "kdk-phys = kick–drift–kick with the same rule; naive = HLL eq. 9. Slopes ≈2 / ≈0.7 / ≈1.8 / ≈1.")
    lines.append("\n## Movies\n")
    for mp4 in sorted(glob.glob(os.path.join(ROOT, "movies", "*.mp4"))):
        lines.append(f"* `{mp4}` — {probe(mp4)}")
    extra = os.path.join(ROOT, "notes", "render_timing.txt")
    lines.append("\n## Wall times\n")
    lines.append("Simulation: " + "; ".join(f"{n} {m.get('wall_seconds', float('nan')):.1f} s" for n, m in metas.items()) +
                 f" (this machine: {os.cpu_count()} CPUs, {chip()}; analysis ≈ 30 s per run).")
    if os.path.exists(extra):
        lines.append("Render + encode (from notes/render_timing.txt):\n")
        lines.append("```\n" + open(extra).read().strip() + "\n```")
    obs = os.path.join(ROOT, "notes", "observations.md")
    if os.path.exists(obs):
        lines.append("\n" + open(obs).read().strip() + "\n")
    out = os.path.join(ROOT, "NOTES.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
