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
    lines.append("\nEquilibration: neutral stage t_equil_neutral with velocity rescaling every 0.1 τ; charged runs then switch on the charges and equilibrate t_equil_charged more (rescaled), production is NVE.\n")
    lines.append("\n## Movies\n")
    for mp4 in sorted(glob.glob(os.path.join(ROOT, "movies", "*.mp4"))):
        lines.append(f"* `{mp4}` — {probe(mp4)}")
    extra = os.path.join(ROOT, "notes", "render_timing.txt")
    if os.path.exists(extra):
        lines.append("\n## Render timing\n")
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
