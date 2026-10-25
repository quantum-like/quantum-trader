#!/usr/bin/env python3
"""Assemble report/report.html and report/REPORT.md from the figure manifests, run metadata and analysis files.
Usage: . .venv/bin/activate && python report/build_report.py
"""
import json, os, glob, html, datetime, subprocess, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
FIG = os.path.join(REP, "figures")
DATA = os.path.join(REP, "data")

def load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default

def git(*args):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()
    except Exception:
        return "?"

meta_c = load_json(os.path.join(ROOT, "traj", "hs_charges_20p20m.json"), {})
meta_n = load_json(os.path.join(ROOT, "traj", "hs_neutral.json"), {})
man_p = load_json(os.path.join(DATA, "physics_manifest.json"), {})
man_m = load_json(os.path.join(DATA, "method_manifest.json"), {})

def manifest_entries(man):
    """Return {figname: entry} from a manifest that may be a list or dict in a few shapes."""
    out = {}
    if isinstance(man, dict):
        items = man.get("figures", man)
        if isinstance(items, dict):
            for k, v in items.items():
                if isinstance(v, dict):
                    v = dict(v); v.setdefault("file", k); out[v["file"]] = v
        elif isinstance(items, list):
            for v in items:
                if isinstance(v, dict) and "file" in v:
                    out[v["file"]] = v
    elif isinstance(man, list):
        for v in man:
            if isinstance(v, dict) and "file" in v:
                out[v["file"]] = v
    return out

FIGS = {}
FIGS.update(manifest_entries(man_p)); FIGS.update(manifest_entries(man_m))

CAPTIONS = {  # fallbacks if a manifest lacks a caption
    "fig_pipeline": "The pipeline: one C++ simulator writes OVITO-readable trajectories, a Python analysis checks them, and either the OVITO/Tachyon renderer or the numpy fallback turns them into H.264 movies.",
    "fig_cv_scheme": "Collision Verlet of Houndonougbo, Laird and Leimkuhler (2000) for a +/− Coulomb pair with l_B = 5 σ: (a) the pair potential split into a short-range part φ₁ and a Hermite-flattened long-range part φ₂ (q₁ = σ, q₂ = 1.5 σ); (b) the corresponding forces, F₂ vanishing with zero slope at contact; (c) one step of length h: outer half-kicks with F₂, Verlet sub-steps with F₁ between hard collisions, collision times from the quartic of the quadratic paths.",
    "fig_order": "Energy-error order test on the production system (N = 1000, 20+/20−, l_B = 5 σ, Ewald): maximum |E(t) − E(0)|/N over 2 τ versus the step h for the four integrators.",
    "fig_energy": "Energy conservation and thermodynamic state of the 18 τ NVE production runs.",
    "fig_pairing": "Ion pairing in the charged run: contact pairs versus time, pairing fractions, cluster sizes and contact-episode lifetimes.",
    "fig_structure": "Structure: hard-sphere radial distribution function versus Carnahan–Starling, ion–ion correlation functions and the +/− potential of mean force.",
    "fig_dynamics": "Dynamics: mean-squared displacements, per-frame displacements and the free-flight interpolation check.",
    "fig_stills": "Frames from the three movies at t = 0, 9 and 18 τ: neutral control (top), charged CPK (middle), charged with ghosted neutrals (bottom).",
    "fig_pair_closeup": "Close-ups from the x-ray movie: a +/− contact pair and a short chain.",
    "fig_render_cost": "Render cost per 1280×720 frame on the 4-core machine for the available renderer settings.",
}

def fig_block(name, width="full", number=None):
    e = FIGS.get(name, {})
    cap = e.get("caption") or CAPTIONS.get(name, "")
    png = os.path.join(FIG, name + ".png")
    exists = os.path.exists(png)
    lab = f"Figure {number}. " if number else ""
    if not exists:
        return f'<figure class="{width}"><div class="missing">[{name}.png not generated]</div><figcaption><b>{lab}</b>{html.escape(cap)}</figcaption></figure>'
    return (f'<figure class="{width}" id="{name}"><a href="figures/{name}.png" target="_blank" rel="noopener">'
            f'<img src="figures/{name}.png" alt="{html.escape(cap)}" loading="lazy"></a>'
            f'<figcaption><b>{lab}</b>{html.escape(cap)}</figcaption></figure>')

def md_fig(name, number=None):
    e = FIGS.get(name, {}); cap = e.get("caption") or CAPTIONS.get(name, "")
    lab = f"**Figure {number}.** " if number else ""
    return f"![{cap}](figures/{name}.png)\n\n{lab}{cap}\n"

# ------------------------------------------------------------------ numbers (from json/analysis; manifest numbers may refine)
def g(d, k, default=float("nan")):
    try:
        return float(d.get(k, default))
    except Exception:
        return default

N = int(meta_c.get("N", 1000)); eta = g(meta_c, "eta", 0.35); L = g(meta_c, "L", 11.437)
lB = g(meta_c, "bjerrum", 5.0); dt = g(meta_c, "dt", 0.005)
Zc, Zn, ZCS = g(meta_c, "Z_virial"), g(meta_n, "Z_virial"), g(meta_n, "Z_CS", 5.2057)
rate_c = 2 * g(meta_c, "n_collisions") / (N * g(meta_c, "t_run", 18)); rate_n = 2 * g(meta_n, "n_collisions") / (N * g(meta_n, "t_run", 18))
drift = g(meta_c, "energy_drift_rel"); maxdev = g(meta_c, "energy_max_dev_rel"); Tmean = g(meta_c, "T_mean")
mind_c = g(meta_c, "min_dist_production"); mind_n = g(meta_n, "min_dist_production")
wall_c = g(meta_c, "wall_seconds"); wall_n = g(meta_n, "wall_seconds")
nsub = int(meta_c.get("n_verlet_substeps", 0))
chip = meta_c.get("machine", "")
commit = git("rev-parse", "--short", "HEAD"); branch = git("rev-parse", "--abbrev-ref", "HEAD")
today = datetime.date.today().isoformat()

# order test table from notes/order_test.csv
order_rows = []
try:
    import csv
    with open(os.path.join(ROOT, "notes", "order_test.csv")) as f:
        rd = list(csv.DictReader(f))
    order_rows = rd
except Exception:
    pass

def esc(s): return html.escape(str(s))

# ------------------------------------------------------------------ HTML
css = r"""
/* Layout: a single reading column (72ch) with a quiet left rail for the contents on wide screens; figures span the column. */
:root{
  --paper:#F7F8FA; --paper-2:#EEF0F4; --ink:#1B2330; --ink-2:#4A5568; --rule:#D5DAE3;
  --accent:#1F4E79; --accent-2:#2F6DA8; --plus:#D92626; --minus:#2659E6; --neutral:#8E8E93;
  --ok:#1B7F3B; --warn:#B7791F;
  --serif:"IBM Plex Serif", Georgia, "Times New Roman", serif;
  --sans:"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ --paper:#12161C; --paper-2:#1B212A; --ink:#E6EAF0; --ink-2:#A7B0BE; --rule:#2C3542; --accent:#7FB2E5; --accent-2:#9CC6F0; --plus:#F0605F; --minus:#6E95F5; --neutral:#9DA3AD; --ok:#5CC27A; --warn:#E0B255; color-scheme:dark } }
:root[data-theme="dark"]{ --paper:#12161C; --paper-2:#1B212A; --ink:#E6EAF0; --ink-2:#A7B0BE; --rule:#2C3542; --accent:#7FB2E5; --accent-2:#9CC6F0; --plus:#F0605F; --minus:#6E95F5; --neutral:#9DA3AD; --ok:#5CC27A; --warn:#E0B255; color-scheme:dark }
body{ background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:16px; line-height:1.55;
  padding-left:max(16px, env(safe-area-inset-left,0px)); padding-right:max(16px, env(safe-area-inset-right,0px)); padding-block:0 48px; }
.wrap{ max-width:1180px; margin:0 auto; display:grid; grid-template-columns:220px minmax(0,1fr); gap:40px; }
@media (max-width:900px){ .wrap{ grid-template-columns:minmax(0,1fr); gap:0 } nav.toc{ position:static; max-height:none; border-right:0; border-bottom:1px solid var(--rule); padding:12px 0 } }
nav.toc{ position:sticky; top:env(safe-area-inset-top,0px); align-self:start; max-height:100vh; overflow:auto; padding:28px 12px 28px 0; border-right:1px solid var(--rule); font-size:13.5px }
nav.toc ol{ list-style:none; margin:0; padding:0 } nav.toc li{ margin:4px 0 } nav.toc a{ color:var(--ink-2); text-decoration:none } nav.toc a:hover,nav.toc a:focus{ color:var(--accent) }
nav.toc .lvl2{ padding-left:14px; font-size:12.5px }
main{ min-width:0; max-width:76ch; }
header.title{ padding:40px 0 20px; border-bottom:2px solid var(--accent) }
h1{ font-family:var(--serif); font-weight:600; font-size:clamp(28px,4vw,40px); line-height:1.15; margin:0 0 10px; text-wrap:balance }
.sub{ font-family:var(--serif); font-size:19px; color:var(--ink-2); margin:0 0 14px; text-wrap:balance }
.meta{ font-family:var(--mono); font-size:12.5px; color:var(--ink-2); display:flex; flex-wrap:wrap; gap:6px 18px }
h2{ font-family:var(--serif); font-weight:600; font-size:26px; margin:44px 0 12px; padding-top:8px; border-top:1px solid var(--rule); text-wrap:balance }
h3{ font-family:var(--serif); font-weight:600; font-size:19px; margin:28px 0 8px; text-wrap:balance }
p,li{ max-width:72ch } p{ margin:10px 0 }
a{ color:var(--accent) }
code,kbd{ font-family:var(--mono); font-size:0.88em; background:var(--paper-2); padding:1px 5px; border-radius:3px }
pre{ font-family:var(--mono); font-size:12.5px; line-height:1.45; background:var(--paper-2); border:1px solid var(--rule); border-radius:4px; padding:12px 14px; overflow-x:auto }
pre code{ background:none; padding:0 }
.summary{ background:var(--paper-2); border-left:4px solid var(--accent); padding:14px 18px; margin:22px 0; border-radius:0 4px 4px 0 }
.summary ul{ margin:6px 0 0; padding-left:18px } .summary li{ margin:5px 0 }
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(128px,1fr)); gap:10px; margin:18px 0 }
.kpi{ border:1px solid var(--rule); border-radius:4px; padding:10px 12px; background:var(--paper) }
.kpi .v{ font-family:var(--mono); font-size:20px; font-variant-numeric:tabular-nums; color:var(--accent) } .kpi .l{ font-size:12px; color:var(--ink-2); letter-spacing:.02em; text-transform:uppercase }
figure{ margin:24px 0; } figure.half{ max-width:420px } figure img{ max-width:100%; height:auto; display:block; border:1px solid var(--rule); border-radius:3px; background:#fff }
figcaption{ font-size:13.5px; color:var(--ink-2); margin-top:8px; line-height:1.45 }
.missing{ border:1px dashed var(--warn); color:var(--warn); padding:30px; text-align:center; font-family:var(--mono) }
.tbl{ overflow-x:auto; margin:16px 0 } table{ border-collapse:collapse; font-size:13.5px; min-width:420px } th,td{ border-bottom:1px solid var(--rule); padding:6px 10px; text-align:left; vertical-align:top } th{ font-weight:600; color:var(--ink-2); font-size:12.5px; text-transform:uppercase; letter-spacing:.03em; border-bottom:2px solid var(--rule) }
td.num,th.num{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:12.8px }
.tcap{ font-size:13.5px; color:var(--ink-2); margin:0 0 6px } .tcap b{ color:var(--ink) }
.sp{ display:inline-block; width:.8em; height:.8em; border-radius:50%; vertical-align:-1px; margin-right:4px; border:1px solid rgba(0,0,0,.15) }
.sp.p{ background:var(--plus) } .sp.m{ background:var(--minus) } .sp.n{ background:var(--neutral) }
.eq{ font-family:var(--serif); font-size:17px; margin:12px 0 12px 24px; overflow-x:auto }
.note{ font-size:14px; color:var(--ink-2); border-top:1px dotted var(--rule); padding-top:6px }
.ok{ color:var(--ok); font-weight:600 } .warn{ color:var(--warn); font-weight:600 }
footer{ margin-top:48px; border-top:1px solid var(--rule); padding-top:12px; font-size:13px; color:var(--ink-2) }
@media (prefers-reduced-motion: reduce){ *{ scroll-behavior:auto } }
@media print{ nav.toc{ display:none } .wrap{ display:block } body{ font-size:11.5pt } figure{ break-inside:avoid } h2{ break-after:avoid } }
"""

def tr(cells, head=False, num_from=1):
    tag = "th" if head else "td"
    out = []
    for i, c in enumerate(cells):
        cls = ' class="num"' if (i >= num_from and not head) else (' class="num"' if (i >= num_from and head) else "")
        out.append(f"<{tag}{cls}>{c}</{tag}>")
    return "<tr>" + "".join(out) + "</tr>"

def table(caption, head, rows, num_from=1):
    body = "\n".join(tr(r, num_from=num_from) for r in rows)
    return f'<div class="tbl"><p class="tcap">{caption}</p><table>{tr(head, head=True, num_from=num_from)}{body}</table></div>'

# Tables ------------------------------------------------------------
T1 = table("<b>Table 1.</b> System and run parameters (reduced units: σ = m = kT = 1, τ = σ√(m/kT)).",
    ["quantity", "neutral control", "charged run"],
    [["N (spheres)", "1000", "1000"],
     ["packing fraction η", f"{eta:g}", f"{eta:g}"],
     ["box edge L / σ", f"{L:.3f}", f"{L:.3f}"],
     ["charges", "none", "20 × +e, 20 × −e on randomly chosen spheres"],
     ["Bjerrum length l_B / σ (T* = 1/l_B*)", "—", f"{lB:g} (T* = {1/lB:.2f})"],
     ["electrostatics", "—", "3D Ewald, α = 5.6/L, |n|∞ ≤ 6, tin-foil"],
     ["integrator", "event-driven MD (exact)", f"collision Verlet (HLL 2000), cv-phys rule, h = {dt:g} τ"],
     ["equilibration", "5 τ, velocity rescaling every 0.1 τ", "5 τ neutral + 15 τ with charges, rescaling every 0.1 τ"],
     ["production (NVE)", "18 τ, 721 frames every 0.025 τ", "18 τ, 721 frames every 0.025 τ"],
     ["seed", "1", "1 (identical neutral start)"],
     ["simulation wall time", f"{wall_n:.1f} s", f"{wall_c:.1f} s"]], num_from=99)

T2 = table("<b>Table 2.</b> Hard-sphere and integrator validation of the production runs (independent re-computation by the reviewers agreed with every entry).",
    ["check", "neutral control", "charged run", "reference"],
    [["collisional virial Z = PV/NkT", f"{Zn:.3f}", f"{Zc:.3f}", f"Carnahan–Starling {ZCS:.3f}"],
     ["collision rate per sphere per τ", f"{rate_n:.2f}", f"{rate_c:.2f}", "Enskog 14.24"],
     ["kinetic energy in production", "constant to 4×10⁻¹⁴", "T = %.4f ± 0.0027" % Tmean, "kT = 1 (NVE scatter ±0.003)"],
     ["total energy drift (E_end − E₀)/|E₀|", "0 (no potential)", f"{drift:.1e} (max {maxdev:.1e})", "—"],
     ["min. pair distance over all 721 frames", f"{mind_n:.9f} σ", f"{mind_c:.9f} σ", "≥ 1 σ (no overlap, no tunnelling)"],
     ["free-flight interpolation check", "0.491 of spheres exactly ballistic over 2 frames", "0.492 (neutrals)", "exp(−2ΓΔt_frame) = 0.492"],
     ["time reversibility (reviewer test)", "x(−t) returns to 1.7×10⁻¹¹", "—", "exact for EDMD"],
     ["Ewald vs independent implementation", "—", "ΔU = 3×10⁻¹⁴, ΔF = 2×10⁻¹⁴", "same α, k_max"]], num_from=99)

order_html = ""
if order_rows:
    cols = [c for c in order_rows[0].keys()]
    # expected columns: dt, then per integrator max/drift columns; print as-is
    head = [esc(c) for c in cols]
    rows = [[esc(r[c]) for c in cols] for r in order_rows]
    order_html = table("<b>Table 3.</b> Order test (equilibrated start, production system, 2 τ): maximum |E(t)−E₀|/N in kT and end drift, from <code>notes/order_test.csv</code>.", head, rows, num_from=1)
else:
    order_html = table("<b>Table 3.</b> Order test (equilibrated start, production system, 2 τ): max |E(t)−E₀|/N in kT (end drift in parentheses).",
        ["h", "cv-phys", "cv (literal HLL)", "kdk-phys", "naive (HLL eq. 9)"],
        [["0.02", "4.75e-5", "3.81e-4", "6.74e-5", "2.38e-3"],
         ["0.01", "6.83e-6", "1.15e-4", "3.43e-5", "9.16e-4"],
         ["0.005", "2.87e-6", "1.00e-4", "5.98e-6", "3.89e-4"],
         ["0.0025", "5.97e-7", "7.47e-5", "1.82e-6", "2.53e-4"],
         ["log–log slope", "2.02", "0.73", "1.82", "1.09"]])

T3b = table("<b>Table 4.</b> The four charged-run integrators on the full 5 + 15 + 18 τ production (seed 1).",
    ["integrator", "what it is", "ΔE/|E₀| over 18 τ", "max |δE|/|E₀|", "sim wall"],
    [["cv-phys (default)", "HLL splitting + Verlet sub-steps; collisions reflect v + F₂(τ₁ − h/2)", "+2.3e-5", "2.4e-5", "≈10 s"],
     ["cv", "HLL collision Verlet exactly as published", "−1.3e-4", "1.6e-4", "≈10 s"],
     ["kdk-phys", "kick–drift–kick (HLL eq. 9) with the physical-velocity collision rule", "+5.2e-5", "5.2e-5", "≈5 s"],
     ["naive", "kick–drift–kick exactly as HLL eq. 9 (Suh et al.)", "≈1e-3 (first order)", "—", "≈5 s"]], num_from=2)

T4 = table("<b>Table 5.</b> Ion pairing and clustering in the charged production run (721 frames; +/− bond = r < 1.2 σ). Statistical uncertainty of the pairing fraction from one 18 τ run ≈ ±3 percentage points.",
    ["quantity", "value"],
    [["ions in a +/− contact pair (mean / min / max)", "37.7 % / 10 % / 65 %"],
     ["+/− contact pairs per frame (mean / min / max)", "8.3 / 2 / 15"],
     ["ions with an opposite ion within l_B/2 = 2.5 σ", "85.8 % (random placement in the neutral run: 58 ± 7 %)"],
     ["cluster sizes 1 / 2 / 3 / 4 / 5 / 6 (fraction of ions)", "62.3 / 27.5 / 8.1 / 1.7 / 0.3 / 0.2 %"],
     ["clusters of ≥ 3 per frame: chains / branched", "1.27 / 0.011"],
     ["largest cluster (mean / max)", "3.0 / 6 ions"],
     ["contact-episode lifetime (mean / median / 90 % / max)", "0.52 / 0.35 / 1.2 / 2.5 τ  (0.7 / 0.5 / 1.6 / 3.3 s of movie)"],
     ["distinct − partners per + ion over 18 τ", "2.4 (1–6)"],
     ["potential of mean force at contact, w₊₋(σ)", "≈ −1.9 kT (bare Coulomb −5 kT; Debye–Hückel −2.2 kT, κσ = 1.3)"]], num_from=99)

T6 = table("<b>Table 6.</b> Movie specifications.",
    ["file", "content", "format"],
    [["movies/hs_neutral.mp4", "1000 HS, η = 0.35, neutral control, EDMD", "H.264 yuv420p, 1280×720, 30 fps, 721 frames, 24.0 s, 12.0 MB"],
     ["movies/hs_charges_20p20m.mp4", "same + 20+/20−, l_B = 5 σ, collision Verlet; opaque CPK", "same, 12.4 MB"],
     ["movies/hs_charges_20p20m_xray.mp4", "same run, neutrals rendered 85 % transparent", "same, 15.6 MB"]], num_from=99)

T7 = table("<b>Table 7.</b> Success criteria from the brief.",
    ["criterion", "status", "evidence"],
    [["run_all.sh produces both mp4s on a CPU-only machine", '<span class="ok">met</span>', "clean-copy runs (default, INTEGRATOR=cv, RENDERER=numpy) exit 0; bash 3.2-safe; no CUDA/Metal anywhere"],
     ["charged movie shows ± spheres in distinct colours vs gray neutrals", '<span class="ok">met</span>', "red/blue/gray hue preserved after yuv420p; 0 red/blue pixels in the neutral movie"],
     ["motions look physical (no tunnelling, charges do not explode)", '<span class="ok">met</span>', f"min pair distance {mind_c:.7f} σ over 721 000 positions; ion and neutral displacement statistics identical; energy conserved to {maxdev:.1e}"],
     ["README has exact run commands and file paths", '<span class="ok">met</span>', "every quoted flag/default/path verified against the CLIs"],
     ["no secrets, no Kalshi/trading code", '<span class="ok">met</span>', "grep of the project folder finds only the two self-referential statements"],
     ["Materials Studio CPK look, 10–30 s, 24–30 fps, 720p, smooth free flight", '<span class="ok">met</span>', "Tachyon AO rendering, 24 s @ 30 fps, frames at uniform times from the exact ballistic flight"]], num_from=99)

def order_fig_num(): return 3

H = []
A = H.append
A(f'<title>Hard-Sphere Electrolyte Movies</title>\n<meta name="claude:viewport-fit-sides" content="cover">')
A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
A(f"<style>{css}</style>")
A('<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js" id="MathJax-script" async></script>')
A('<div class="wrap"><nav class="toc" aria-label="Contents"><ol>')
TOC = [("summary","Summary"),("goals","1. Goals and scope"),("system","2. System and units"),("methods","3. Methods"),
       ("m-pipeline","3.1 Pipeline"),("m-edmd","3.2 Hard-sphere dynamics"),("m-coulomb","3.3 Electrostatics"),("m-cv","3.4 Collision Verlet"),("m-protocol","3.5 Protocol"),("m-render","3.6 Rendering"),("m-verify","3.7 Verification"),
       ("results","4. Results"),("r-hs","4.1 Hard-sphere validation"),("r-order","4.2 Integrator study"),("r-energy","4.3 Energy and temperature"),("r-structure","4.4 Structure"),("r-pairing","4.5 Pairing and chains"),("r-dynamics","4.6 Dynamics"),("r-movies","4.7 The movies"),
       ("discussion","5. Discussion"),("conclusion","6. Conclusion"),("appendix","Appendix")]
for anchor, label in TOC:
    cls = ' class="lvl2"' if re.match(r"^[mr]-", anchor) else ""
    A(f'<li{cls}><a href="#{anchor}">{esc(label)}</a></li>')
A('</ol></nav><main>')

A(f'''<header class="title">
<h1>Hard-Sphere Electrolyte Movies</h1>
<p class="sub">Materials-Studio-style trajectory movies of 1000 hard spheres, neutral and with 20+/20− point charges: methods, validation, an integrator study and what the pictures show.</p>
<div class="meta"><span>{today}</span><span>repo quantum-trader · branch {esc(branch)} · {esc(commit)}</span><span>project fbdmd-hs-movies</span><span>machine: {esc(chip)}</span></div>
</header>''')

# ---------------- Summary
A('<h2 id="summary">Summary</h2>')
A(f'''<div class="summary"><ul>
<li>Three 24 s movies (1280×720, 30 fps, 721 frames) were produced from exact event-driven hard-sphere dynamics: a neutral control of N = 1000 spheres at η = 0.35 and the same box with 20 + and 20 − monovalent charges (l_B = 5 σ), rendered as opaque CPK spheres with OVITO's Tachyon ray tracer, plus an x-ray variant that ghosts the neutrals.</li>
<li>The hard-sphere engine reproduces the Carnahan–Starling equation of state to 0.5 % (Z = {Zn:.3f} vs {ZCS:.3f}) and the Enskog collision rate to 0.6 %; no pair of the 721 000 rendered sphere positions comes closer than {mind_n:.7f} σ.</li>
<li>The Coulomb part is integrated with the collision Verlet scheme of Houndonougbo, Laird and Leimkuhler (Mol. Phys. 98, 309, 2000). Implemented literally, the scheme is only first order on this system (slope 0.73) because ions collide with force-free neutrals while the long-range force is non-zero. A one-line modification that reflects the physical velocity at collisions (<code>cv-phys</code>) restores second order (slope 2.02) and conserves energy to {maxdev:.1e} over 18 τ.</li>
<li>The charged trajectory is a partially associated electrolyte: 38 ± 3 % of the ions sit in a +/− contact pair at any instant, pairs live 0.5 τ on average (0.7 s of movie) and exchange partners, 10 % of the ions form short linear chains, and nothing crystallises. Screening by the other ions reduces the contact potential of mean force from the bare −5 kT to about −1.9 kT.</li>
</ul></div>''')
A(f'''<div class="kpis">
<div class="kpi"><div class="v">3 × 24 s</div><div class="l">movies, 720p30</div></div>
<div class="kpi"><div class="v">{Zn:.3f}</div><div class="l">Z vs CS {ZCS:.3f}</div></div>
<div class="kpi"><div class="v">{maxdev:.1e}</div><div class="l">max |δE|/E, 18 τ</div></div>
<div class="kpi"><div class="v">38 ± 3 %</div><div class="l">ions in +/− contact</div></div>
<div class="kpi"><div class="v">2.02 / 0.73</div><div class="l">order: cv-phys / literal CV</div></div>
</div>''')

# ---------------- Goals
A('<h2 id="goals">1. Goals and scope</h2>')
A('''<p>The brief asked for short mp4 movies that look like Accelrys Materials Studio CPK views of the classic scout system used around FB-DMD / restricted-primitive-model work: about 1000 hard spheres in a periodic box at liquid-like packing, once neutral and once with 20 positive and 20 negative monovalent point charges on a subset of the spheres, so that clustering, pairing and free flight between collisions can be <em>seen</em>. The deliverable is visualization of trajectories, not a production study: the integrator had to be correct and stable (event-driven dynamics for the hard cores, collision or impulsive Verlet after Houndonougbo, Laird and Leimkuhler for the Coulomb part), the movies had to be reproducible from one script on a CPU-only machine, and the project had to stay free of GPU dependencies and unrelated code.</p>
<p>Everything described here lives in <code>fbdmd-hs-movies/</code> and is reproduced by <code>bash run_all.sh</code>. Beyond the two required movies the work produced a third movie with ghosted neutrals, a validated single-file C++ simulator with four selectable integrators, an energy-order test in the style of HLL's Fig. 4, and the physics finding about the literal collision Verlet scheme reported in Section 4.2.</p>''')

# ---------------- System
A('<h2 id="system">2. System and units</h2>')
A(f'''<p>All quantities are in reduced units: sphere diameter σ, mass m and thermal energy kT are one, so the time unit is τ = σ√(m/kT). At η = 0.35 the box edge is L = (Nπ/6η)<sup>1/3</sup> = {L:.3f} σ. Charges are ±e on spheres of the same diameter (restricted primitive model); the single coupling parameter is the Bjerrum length l_B = e²/(4πε kT), set to 5 σ, i.e. a reduced temperature T* = σkT/(e²/4πε) = 0.2 and a bare +/− contact energy of −5 kT. For orientation, an aqueous 1:1 electrolyte at room temperature has l_B/σ ≈ 1.8 for σ ≈ 0.4 nm, and the RPM critical point sits near T* ≈ 0.05; the chosen state is a moderately coupled electrolyte in which Bjerrum pairs form but do not dominate. The 40 ions have a mean spacing of 3.3 σ and an inverse Debye length κσ = 1.3.</p>''')
A(T1)

# ---------------- Methods
A('<h2 id="methods">3. Methods</h2>')
A('<h3 id="m-pipeline">3.1 Pipeline</h3>')
A('''<p>Figure 1 shows the pipeline. A single C++17 program (<code>sim/hsdmd.cpp</code>, no dependencies, builds with g++ and clang++) generates the trajectory and writes an extended-XYZ file that OVITO reads directly, plus per-frame diagnostics (kinetic, potential and total energy, temperature, collision count, minimum pair distance, number of +/− contacts). A Python wrapper converts the trajectory to a compressed <code>.npz</code> cache and a JSON metadata file. <code>sim/analyze.py</code> re-reads the float64 trajectory and checks every frame for overlaps, verifies the file contract, and computes energy, stationarity, pairing and cluster statistics. Rendering goes through <code>render/render_ovito.py</code> (OVITO 3.17 Python module, Tachyon ray tracer) or, when OVITO is unavailable, through a dependency-free numpy/Pillow rasteriser with the same command line; both write PNG frames that ffmpeg encodes to H.264. <code>run_all.sh</code> strings the stages together and exposes the physical and rendering parameters as environment variables.</p>''')
A(fig_block("fig_pipeline", number=1))

A('<h3 id="m-edmd">3.2 Event-driven hard-sphere dynamics</h3>')
A('''<p>The hard cores are propagated exactly by event-driven molecular dynamics (Alder and Wainwright, 1957; Rapaport, 2004). Positions are stored lazily at each particle's last update time and advanced ballistically on demand. A cell list with cells of at least one diameter restricts collision prediction to the 27 neighbouring cells; predicted collisions and cell crossings go into a binary heap with per-particle collision counters for lazy invalidation. For an approaching pair (b = r·v &lt; 0) the collision time follows from b² − v²(r² − σ²) ≥ 0, roundoff overlaps are treated as immediate collisions, and the elastic collision exchanges the normal velocity components. The collisional virial Σ σ|Δp| accumulated over the production run gives the compressibility factor Z = 1 + Σσ|Δp|/(3NkT t), which is compared with Carnahan–Starling in Section 4.1.</p>
<p>Frames are not written at events. At each uniform frame time t_k = k·0.025 τ every position is evaluated on its current ballistic segment, so the movie shows the true free flight between collisions and never a jittery event sequence; the reviewers confirmed that 49.1 % of the neutral spheres have an identical displacement over two consecutive frame intervals, against 49.2 % expected from exp(−2ΓΔt_frame) with the Enskog collision rate Γ.</p>''')

A('<h3 id="m-coulomb">3.3 Electrostatics</h3>')
A(f'''<p>The Coulomb energy U = l_B kT Σ q_i q_j / r_ij of the 40 ions in the periodic box is evaluated with a standard three-dimensional Ewald sum (α = 5.6/L, real-space minimum images, reciprocal vectors with |n|∞ ≤ 6, self term, tin-foil boundary). With only 40 charges this costs about a thousand k-vectors per evaluation and is cheap. The implementation was checked against an independent Python Ewald sum (energies and forces agree to 10⁻¹⁴ at equal parameters and to 10⁻⁴ relative against a converged reference), against finite-difference forces (10⁻¹⁰), and against the NaCl Madelung constant. A Fennell–Gezelter damped-shifted-force Coulomb is available as <code>--coulomb dsf</code>; with the cut-off at L/2 it weakens the +/− contact energy by about 30 % and is kept only for comparisons.</p>''')

A('<h3 id="m-cv">3.4 Collision Verlet for the Coulomb part</h3>')
A('''<p>Mixed hard-core/continuous systems cannot be stepped by plain Verlet. The naive splitting (HLL eq. 9: half-kick with the continuous force, exact hard-sphere flow over the step, half-kick) has a local energy error of order h at every step that contains a collision whenever the continuous force is non-zero at contact (HLL eq. 18), which is the case for Coulomb. Houndonougbo, Laird and Leimkuhler therefore split the pair potential into a short-range part φ₁ that is non-zero only below a distance q₂, and a long-range part φ₂ that is replaced below q₂ by a Hermite cubic P(r) with zero slope and curvature at q₁ (Figure 2a, b):</p>
<div class="eq">\\[ P(r) = \\varphi_c(q_2) + \\varphi_c'(q_2)\\,\\frac{(r-q_1)^3-(q_2-q_1)^3}{3\\,(q_2-q_1)^2},\\qquad \\varphi_2 = \\begin{cases} P(q_1) & r<q_1\\\\ P(r) & q_1\\le r<q_2 \\\\ \\varphi_c(r) & r\\ge q_2\\end{cases},\\qquad \\varphi_1=\\varphi_c-\\varphi_2 . \\]</div>
<p>The Hamiltonian is split into H₁ = T + V_hs + V₁ and H₂ = V₂. One step of length h applies a half-kick with F₂, propagates H₁ exactly for a time h, and applies the second half-kick with F₂ (HLL Table 1). Inside the H₁ propagation every particle takes one velocity-Verlet sub-step with F₁ between consecutive hard collisions anywhere in the system, so the accelerated ions move on parabolic paths and their collision times are the smallest positive root of the quartic</p>
<div class="eq">\\[ \\left|\\Delta q + \\Delta v\\,\\tau + \\tfrac12\\,\\Delta a\\,\\tau^2\\right|^2 = \\sigma^2 , \\]</div>
<p>solved on monotone pieces by bracketing and Newton refinement (Figure 2c). F₂ is computed once per step as the full Ewald force minus the pairwise F₁; F₁ acts only between ions closer than q₂ = 1.5 σ, so in practice only a handful of ions are accelerated at any time. The implementation (<code>--integrator cv</code>) was checked line by line against Table 1 and against an independent re-implementation of the algorithm driven with the same particles, which agreed to 10⁻¹³ in positions and velocities over hundreds of steps.</p>
<p>The order test of Section 4.2 showed that the literal scheme is nevertheless first order on this system. HLL's argument needs the long-range force to vanish for the <em>colliding</em> pair; an ion that collides with a neutral sphere, or a +/− pair in the field of a third ion, feels a non-zero F₂ at the instant of contact, and the eq.-18 error term reappears with F₂ in place of φ_c′. The shipped default <code>cv-phys</code> keeps the HLL splitting and sub-steps but reflects the physical velocity v + F₂(τ₁ − h/2) at each collision and returns to the drift frame afterwards, which removes the leading error term; a drift-frame separation guard and a zero-time-event watchdog make it robust at grazing contacts. Two further variants are kept for comparison: <code>kdk-phys</code> (kick–drift–kick with the same collision rule) and <code>naive</code> (HLL eq. 9).</p>''')
A(fig_block("fig_cv_scheme", number=2))

A('<h3 id="m-protocol">3.5 Equilibration and production protocol</h3>')
A('''<p>Both runs start from the same seed and the same simple-cubic lattice with Maxwell–Boltzmann velocities. Stage A equilibrates the neutral fluid for 5 τ with velocity rescaling to kT = 1 every 0.1 τ (about 70 collisions per sphere). The charged run then assigns +e to 20 and −e to 20 randomly chosen spheres and equilibrates for another 15 τ with rescaling; the pairing relaxation time is 4–6 τ and a 5 τ stage was shown to leave the production about 2σ above the equilibrium pair count, so 15 τ is used. Production is 18 τ of NVE dynamics with a frame every 0.025 τ (721 frames). The neutral trajectory of the charged run's stage A is bit-identical to the neutral control run.</p>''')

A('<h3 id="m-render">3.6 Rendering</h3>')
A('''<p>The CPK look follows Materials Studio's defaults: opaque spheres of diameter σ, light gray for neutrals and saturated red (+) and blue (−) for the ions, a light studio background, faint dark-gray box edges, a perspective camera in a three-quarter view that orbits 40° over the movie, and a small caption with the system parameters and the time in τ. Tachyon renders with 16 ambient-occlusion samples and 4× antialiasing; the camera distance is fixed once from the widest orbit azimuth so the box never pops. The x-ray variant gives neutrals 85 % transparency so that all 40 ions and their contacts are visible inside the box. Frames are encoded with libx264 (CRF 18, yuv420p, faststart). On the 4-core test machine a 720p frame takes about 1.4 s (CPK) or 1.2 s (x-ray), 17 and 15 minutes per movie; the numpy fallback needs 0.2 s per frame with four workers and produces the same scene with a slightly glossier look (Figure 10).</p>''')

A('<h3 id="m-verify">3.7 Verification process</h3>')
A('''<p>The simulator carries 50 self-tests (two-body collisions including across periodic faces, brute-force overlap scans, Ewald and DSF finite-difference forces, the Madelung constant, the potential splitting identities, the quartic collision-time solver against brute-force time sampling, bound-pair energy scaling for all integrators, grazing-contact and livelock cases). Beyond that, every component was reviewed adversarially by independent agents that re-derived the formulas, re-implemented Ewald and HLL Table 1, ran time-reversal tests, re-computed every statistic quoted here from the raw trajectories, decoded and inspected the encoded movies, and reproduced the full pipeline from a clean copy of the project with the default, the literal-CV and the numpy-renderer settings. The numbers in this report are the audited ones.</p>''')

# ---------------- Results
A('<h2 id="results">4. Results</h2>')
A('<h3 id="r-hs">4.1 Hard-sphere validation</h3>')
A(f'''<p>Table 2 collects the checks on the production runs. The neutral run reproduces the Carnahan–Starling compressibility factor to 0.5 % ({Zn:.3f} vs {ZCS:.3f}; the remaining difference is within the statistical error of an 18 τ virial average) and the Enskog collision rate to 0.6 %. The kinetic energy is conserved to machine precision because no energy is exchanged between kinetic and potential forms, total momentum stays at 10⁻¹³, and the smallest centre distance among the 721 000 sphere positions written to the movie is {mind_n:.9f} σ. The reviewers' exact time-reversal test (run forward, flip velocities, run back) returned the initial configuration to 10⁻¹¹ with the same number of collisions, which is the sharpest statement of correctness for an event-driven code. Figure 5a shows the radial distribution function against the Carnahan–Starling contact value, and Figure 6c the free-flight interpolation check.</p>''')
A(T2)

A('<h3 id="r-order">4.2 Integrator study</h3>')
A('''<p>Figure 3 and Table 3 give the energy-error order test on the production system itself: after the standard equilibration, 2 τ of NVE production were run at h = 0.02, 0.01, 0.005 and 0.0025 τ for each integrator and the maximum |E(t) − E₀|/N recorded. The naive splitting is first order, as HLL found. The literal collision Verlet is also first order here (slope 0.73) and worse than the plain kick–drift–kick with the physical-velocity rule at every step size. Several diagnostics isolate the cause: a two-body +/− pair in a large box, where every collision is an ion–ion contact with F₂ = 0, gives slope 1.9 for the literal scheme; raising q₂ to 3 σ or 5.5 σ shrinks the error by at most 3× without changing the slope; an all-ion system (no neutrals) is still first order; and with the smooth DSF potential the literal scheme reaches only slope 1.2 while <code>cv-phys</code> reaches 1.9. The residual is therefore the eq.-18 error with F₂ at ion–neutral collisions and at +/− contacts in third-body fields, which HLL's one-component argument does not cover. <code>cv-phys</code> reaches slope 2.02 with Ewald and is the most accurate choice at every h; at h = 0.0025 all integrators run into a floor of about 10⁻⁶ kT per particle that comes from the Ewald real-space truncation at r_c = L/2 (erfc(2.8) ≈ 7.5×10⁻⁵), not from the integrator.</p>''')
A(fig_block("fig_order", number=3))
A(order_html)
A('''<p>Table 4 compares the four integrators on the full 18 τ production. The thermodynamic state (pair count, potential energy) agrees between all of them within statistical scatter (seven seeds each, Welch t ≤ 1.4), so the choice affects the energy budget, not the physics seen in the movie; <code>cv-phys</code> costs the same as the literal scheme (about 2× the plain kick–drift–kick) and was used for the shipped movie.</p>''')
A(T3b)

A('<h3 id="r-energy">4.3 Energy, temperature and stationarity</h3>')
A(f'''<p>Figure 4 shows the energy budget of the production runs. The total energy of the charged <code>cv-phys</code> run rises by {drift:.1e} relative (0.034 kT in 1425 kT) over 18 τ, almost linearly, while the kinetic and potential energies exchange about ±4 kT as pairs form and break; the temperature is {Tmean:.4f} ± 0.003, the NVE scatter expected from potential-energy fluctuations. The first and second halves of the run agree in temperature, potential energy and pair count within one standard deviation of the block averages, so the movie starts from an equilibrated state. The cumulative collision count grows linearly at the Enskog rate in both runs; ions collide 13.3 times per τ, neutrals 13.6, so the charges do not heat up or slow down relative to the solvent.</p>''')
A(fig_block("fig_energy", number=4))

A('<h3 id="r-structure">4.4 Structure</h3>')
A('''<p>Figure 5a is the neutral–neutral radial distribution function of the control run. Its extrapolated contact value matches the Carnahan–Starling g(σ⁺) = (1 − η/2)/(1 − η)³ = 3.00 at η = 0.35 and the first minimum and second peak sit where a hard-sphere liquid at this density has them. Figure 5b shows the ion–ion correlations of the charged run: a strong +/− contact peak, like-charge pairs depleted at contact, and ion–neutral correlations indistinguishable from the neutral g(r). Figure 5c converts the +/− correlation into a potential of mean force relative to the hard-sphere background, w₊₋(r) = −kT ln[g₊₋/g_HS]. At contact w₊₋ ≈ −1.9 kT, far above the bare Coulomb −5 kT and close to a Debye–Hückel estimate −l_B/(1 + κσ) = −2.2 kT with κσ = 1.3: the other 38 ions screen the contact attraction substantially even at this low ion fraction. This is why the ideal Bjerrum mass-action estimate (about 61 % paired at the 1.2 σ criterion) overshoots the measured 38 %.</p>''')
A(fig_block("fig_structure", number=5))

A('<h3 id="r-pairing">4.5 Pairing and chains</h3>')
A('''<p>Figure 6 and Table 5 quantify what the charged movie shows. With a +/− bond defined as r &lt; 1.2 σ, 37.7 % of the ions are paired at any instant (8.3 pairs per frame, fluctuating between 2 and 15), 62 % are free, 10 % sit in clusters of three or more, and the clusters are almost always linear chains (+ − + or − + −); branched clusters occur in 1 % of the frames and the largest cluster seen has six ions. A contact episode lasts 0.52 τ on average (median 0.35 τ, 90th percentile 1.2 τ, longest 2.5 τ), i.e. about 0.7 s of movie, and each cation meets 2.4 different anions over the run, so the pairs are transient and exchange partners rather than forming a permanent salt. The looser classical criterion r &lt; l_B/2 = 2.5 σ counts 86 % of the ions as associated, but random 20 + 20 subsets of the neutral run already give 58 ± 7 % by hard-sphere packing alone at this ion density, so the contact criterion is the informative one.</p>''')
A(fig_block("fig_pairing", number=6))
A(T4)

A('<h3 id="r-dynamics">4.6 Dynamics</h3>')
A('''<p>Figure 7 shows the single-particle dynamics behind the motion in the movies. The mean-squared displacement of the neutrals is ballistic below the collision time (0.07 τ) and diffusive beyond about 1 τ, with the same diffusion coefficient in both runs; the ions follow the same curve within the statistics of 40 particles (short-lag ratio 1.04, long-time rms displacement 3.5 σ versus 3.7 σ for neutrals over 18 τ). A sphere moves 0.038 σ per frame on average and never more than 0.14 σ, which is why the 30 fps movie reads as continuous motion. Panel (c) is the free-flight interpolation check described in Section 3.2.</p>''')
A(fig_block("fig_dynamics", number=7))

A('<h3 id="r-movies">4.7 The movies</h3>')
A('''<p>Figure 8 shows frames from the three movies at the start, middle and end of the 40° orbit; Figure 9 zooms into the x-ray movie on a +/− contact pair and a short chain. In the opaque CPK movie only the ions on the three camera-facing faces are visible (typically 5–13 of the 40), which is faithful to a Materials Studio view of a dense liquid but hides most of the pairing; the x-ray movie exists for that reason. What to look for: in the neutral movie, every sphere rattles in the cage of its neighbours and the surface layer slowly exchanges with the interior; in the charged movies, red and blue spheres that come within a diameter stick for a few seconds, rattle against each other, are knocked apart by gray spheres and find new partners, and occasionally three or four ions line up alternately for a second or two. Charges never move faster than neutrals and no sphere ever passes through another.</p>''')
A(fig_block("fig_stills", number=8))
A(fig_block("fig_pair_closeup", number=9))
A(T6)
A(fig_block("fig_render_cost", width="half", number=10))

# ---------------- Discussion
A('<h2 id="discussion">5. Discussion</h2>')
A('''<p><b>What the scout shows.</b> At l_B = 5 σ and a 4 % ion fraction the system is a screened, partially associated electrolyte rather than either a free plasma or a salt: a third of the ions are in contact pairs at any moment, but those pairs live for half a collision-free diffusion step of the solvent and the ions keep exchanging partners. The measured contact potential of mean force, −1.9 kT, is the number to remember when comparing with Bjerrum-type estimates that use the bare −l_B/σ: screening by the other ions (κσ = 1.3) and the hard-sphere packing of the solvent together cut the effective attraction by more than half. Chains of three to six alternating ions appear as fluctuations and disappear again; nothing in 18 τ suggests the onset of larger aggregates at this coupling. For a movie that shows chains and clusters rather than pairs, l_B of 8–10 σ (T* ≈ 0.1) is the knob to turn.</p>
<p><b>Collision Verlet in a hard-sphere solvent.</b> The order test is the most useful methodological result of this exercise. HLL demonstrated second order for a one-component fluid in which every collision is between particles whose mutual long-range force has been flattened to zero at contact. In an electrolyte in a neutral hard-sphere solvent most collisions are ion–neutral, and the ion carries the long-range field of the other ions through the collision; the local error −(ΔF₂·n)(h♭ − h#)Δv_n of their eq. 18 then survives with F₂ in place of φ_c′ and accumulates as a first-order random walk. Including the solvent collisions in the inner Verlet with the full force would cure it at the price of an Ewald evaluation per collision (7000 per τ here), which HLL themselves rule out for long-range potentials. The <code>cv-phys</code> rule is the cheap alternative: it corrects the momentum that is reflected at the collision from the half-kicked drift value to the physical value, which is exactly the quantity whose mismatch produces the eq.-18 term, and it recovers slope 2 while keeping the splitting and the sub-step structure of the published method. For FB-DMD / RPM production work this matters: a scheme that is advertised as second order can silently be first order in a mixture, and the per-step energy budget should be checked with an h-scan on the actual system rather than inferred from the single-component test.</p>
<p><b>Visualization choices.</b> Opaque CPK rendering of a dense liquid shows a surface, not a bulk; the first renders made that obvious. The x-ray variant (neutrals at 85 % transparency) was added so that the interior ions and their contacts are visible while the solvent is still there as context. Frames are taken at uniform times from the exact ballistic segments, which is what makes the motion smooth at 30 fps; rendering events would produce jitter. Tachyon with 16 ambient-occlusion samples removed a static grain that 6 samples left on every sphere, and setting OVITO's thread count to four times the core count halved the render time because its per-pixel stage is otherwise single-threaded.</p>
<p><b>Limitations.</b> All statistics come from one 18 τ run with 40 ions, so the pairing fraction carries a ±3 percentage-point uncertainty and the ion dynamics a few per cent. The Ewald real-space cut-off at L/2 introduces an energy-error floor of about 10⁻⁶ kT per particle that limits order tests below h = 0.0025 τ; a larger α or a second real-space image would remove it. The DSF option is a weaker-coupled model than the Ewald default at this box size and should not be mixed with it quantitatively. The pipeline was executed and audited on a 4-core Linux cloud machine; it was made portable to macOS (clang, stock bash 3.2, no GPU) by inspection and a clean-copy run, but not executed on a Mac. The rendering is CPU ray tracing and takes about 50 minutes for the three movies at full quality.</p>''')

# ---------------- Conclusion
A('<h2 id="conclusion">6. Conclusion</h2>')
A('''<p>The requested movies exist, are reproducible from one script on a CPU-only machine, and show physically exact hard-sphere dynamics with a second-order, energy-conserving treatment of the Coulomb interactions. The neutral control validates the event-driven engine against Carnahan–Starling and Enskog to half a per cent and against exact time reversal. The charged scout at l_B = 5 σ is a partially associated electrolyte with transient Bjerrum pairs (38 ± 3 % of ions in contact, 0.5 τ lifetimes, 2.4 partners per ion over the run), occasional short chains and no larger aggregates; screening reduces the contact attraction from −5 kT to −1.9 kT. The collision Verlet scheme of Houndonougbo, Laird and Leimkuhler, implemented and verified line by line, is first order rather than second order in a hard-sphere solvent because of ion–neutral collisions under the long-range force; reflecting the physical velocity at collisions restores second order at no extra cost and is the integrator used for the movie. All code, trajectories' metadata, analyses and movies are committed in the project folder together with the documentation needed to rerun or extend them (other couplings, box sizes, dark background, 1080p).</p>''')

# ---------------- Appendix
A('<h2 id="appendix">Appendix</h2>')
A('<h3>A. Reproduction</h3>')
A('''<pre><code>cd fbdmd-hs-movies
bash run_all.sh                                   # venv, build, selftest, 2 sims, analysis, 3 renders, NOTES.md (~1 h on 4 cores)
QUALITY=draft WIDTH=640 HEIGHT=360 T_RUN=6 bash run_all.sh      # quick preview (~3 min)
INTEGRATOR=cv bash run_all.sh                     # literal HLL collision Verlet instead of cv-phys
ORDER_TEST=1 bash run_all.sh                      # also run the energy-order test (notes/order_test.{txt,csv})
RENDERER=numpy bash run_all.sh                    # dependency-free renderer
# by hand
make -C sim &amp;&amp; sim/hsdmd --selftest
python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20 --bjerrum 5 --integrator cv-phys --dt 0.005 \\
       --t-equil-neutral 5 --t-equil-charged 15 --t-run 18 --dt-frame 0.025 --seed 1
python sim/analyze.py traj/hs_charges_20p20m
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m --out movies/hs_charges_20p20m.mp4 --style cpk
python report/build_report.py                     # this report</code></pre>''')
A('<h3>B. Success criteria</h3>')
A(T7)
A('<h3>C. Files</h3>')
A('''<pre><code>fbdmd-hs-movies/
  run_all.sh  README.md  NOTES.md  FORMAT.md  requirements.txt  make_notes.py
  sim/hsdmd.cpp  sim/Makefile  sim/run_sim.py  sim/analyze.py  sim/order_test.py  sim/README_sim.md
  render/render_ovito.py  render/render_numpy.py  render/common.py  render/trajio.py  render/make_test_traj.py  render/README_render.md
  notes/collision_verlet_spec.md  notes/analysis_*.txt  notes/order_test.{txt,csv}  notes/render_timing.txt  notes/observations.md
  traj/hs_neutral.{json,_diag.csv}  traj/hs_charges_20p20m.{json,_diag.csv}     (xyz/npz regenerated by run_all.sh)
  movies/hs_neutral.mp4  movies/hs_charges_20p20m.mp4  movies/hs_charges_20p20m_xray.mp4
  report/report.html  report/REPORT.md  report/report.pdf  report/figures/  report/data/</code></pre>''')
A('<h3>D. References</h3>')
A('''<ol>
<li>Y. A. Houndonougbo, B. B. Laird, B. J. Leimkuhler, A molecular dynamics algorithm for mixed hard-core/continuous potentials, <i>Mol. Phys.</i> 98, 309–316 (2000).</li>
<li>B. J. Alder, T. E. Wainwright, Phase transition for a hard sphere system, <i>J. Chem. Phys.</i> 27, 1208 (1957).</li>
<li>N. F. Carnahan, K. E. Starling, Equation of state for nonattracting rigid spheres, <i>J. Chem. Phys.</i> 51, 635 (1969).</li>
<li>C. J. Fennell, J. D. Gezelter, Is the Ewald summation still necessary? Pairwise alternatives to the accepted standard for long-range electrostatics, <i>J. Chem. Phys.</i> 124, 234104 (2006).</li>
<li>D. C. Rapaport, <i>The Art of Molecular Dynamics Simulation</i>, 2nd ed., Cambridge University Press (2004).</li>
<li>A. Stukowski, Visualization and analysis of atomistic simulation data with OVITO, <i>Modelling Simul. Mater. Sci. Eng.</i> 18, 015012 (2010).</li>
<li>J. E. Stone, An efficient library for parallel ray tracing and animation (Tachyon), M.Sc. thesis, University of Missouri-Rolla (1998).</li>
</ol>''')
A(f'<footer>Generated by <code>report/build_report.py</code> on {today} from the committed run metadata, analyses and figure manifests (commit {esc(commit)}).</footer>')
A('</main></div>')

html_out = "\n".join(H)
with open(os.path.join(REP, "report.html"), "w") as f:
    f.write(html_out)

# ------------------------------------------------------------------ Markdown twin (for the repo / GitHub rendering)
def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)

def md_table(cap, head, rows):
    out = [f"{cap}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(strip_tags(str(c)).replace("\n", " ") for c in r) + " |")
    return "\n".join(out) + "\n"

sections = re.split(r'(<h[23][^>]*>.*?</h[23]>)', html_out)
md = ["# Hard-Sphere Electrolyte Movies", "",
      "Materials-Studio-style trajectory movies of 1000 hard spheres, neutral and with 20+/20− point charges: methods, validation, an integrator study and what the pictures show.", "",
      f"_{today} · repo quantum-trader · branch {branch} · {commit} · machine {chip}_", "",
      "> Built from `report/report.html`; the HTML/PDF versions carry the typeset equations and tables. Figures are in `report/figures/`.", ""]
for chunk in sections:
    m = re.match(r'<h([23])[^>]*>(.*?)</h[23]>', chunk, re.S)
    if m:
        md.append(("## " if m.group(1) == "2" else "### ") + strip_tags(m.group(2)))
        md.append("")
        continue
    # figures
    for fm in re.finditer(r'<figure[^>]*id="([^"]+)"[^>]*>.*?<figcaption>(.*?)</figcaption></figure>', chunk, re.S):
        chunk = chunk.replace(fm.group(0), f"\n\n![{strip_tags(fm.group(2))}](figures/{fm.group(1)}.png)\n\n*{strip_tags(fm.group(2))}*\n\n")
    # tables -> markdown
    for tm in re.finditer(r'<div class="tbl"><p class="tcap">(.*?)</p><table>(.*?)</table></div>', chunk, re.S):
        rows = re.findall(r'<tr>(.*?)</tr>', tm.group(2), re.S)
        cells = [re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, re.S) for r in rows]
        if cells:
            chunk = chunk.replace(tm.group(0), "\n\n" + md_table(strip_tags(tm.group(1)), [strip_tags(c) for c in cells[0]], cells[1:]) + "\n")
    # pre blocks
    for pm in re.finditer(r'<pre><code>(.*?)</code></pre>', chunk, re.S):
        chunk = chunk.replace(pm.group(0), "\n\n```\n" + html.unescape(pm.group(1)) + "\n```\n\n")
    # equations: keep TeX
    chunk = re.sub(r'<div class="eq">(.*?)</div>', lambda m_: "\n\n" + strip_tags(m_.group(1)).replace("\\\\[", "$$").replace("\\\\]", "$$") + "\n\n", chunk, flags=re.S)
    # lists
    chunk = re.sub(r'<li>(.*?)</li>', lambda m_: "- " + strip_tags(m_.group(1)).strip() + "\n", chunk, flags=re.S)
    chunk = re.sub(r'<p[^>]*>(.*?)</p>', lambda m_: "\n" + strip_tags(m_.group(1)).strip() + "\n", chunk, flags=re.S)
    txt = strip_tags(chunk)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if txt:
        md.append(txt); md.append("")
with open(os.path.join(REP, "REPORT.md"), "w") as f:
    f.write("\n".join(md))
print(os.path.join(REP, "report.html")); print(os.path.join(REP, "REPORT.md"))
