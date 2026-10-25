# Hard-Sphere Electrolyte Movies

Materials-Studio-style trajectory movies of 1000 hard spheres, neutral and with 20+/20− point charges: methods, validation, an integrator study and what the pictures show.

_2026-10-25 · repo quantum-trader · branch claude/fbdmd-hs-movies · 975bd1a · machine Intel(R) Xeon(R) Processor @ 2.10GHz (Linux 6.18.44-fc-v153, x86_64, 4 cpus, python 3.13.17)_

> Built from `report/report.html`; the HTML/PDF versions carry the typeset equations and tables. Figures are in `report/figures/`.

Hard-Sphere Electrolyte Movies

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
@media (max-width:900px){ nav.toc{ position:static; max-height:none; overflow:visible; border-right:0; border-bottom:1px solid var(--rule); padding:12px 0 16px; columns:2; column-gap:24px } nav.toc ol{ break-inside:avoid-column } }
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

- Summary

- 1. Goals and scope

- 2. System and units

- 3. Methods

3.1 Pipeline
3.2 Hard-sphere dynamics
3.3 Electrostatics
3.4 Collision Verlet
3.5 Protocol
3.6 Rendering
3.7 Verification
- 4. Results

4.1 Hard-sphere validation
4.2 Integrator study
4.3 Energy and temperature
4.4 Structure
4.5 Pairing and chains
4.6 Dynamics
4.7 The movies
- 5. Discussion

- 6. Conclusion

- Appendix

Hard-Sphere Electrolyte Movies

Materials-Studio-style trajectory movies of 1000 hard spheres, neutral and with 20+/20− point charges: methods, validation, an integrator study and what the pictures show.

2026-10-25repo quantum-trader · branch claude/fbdmd-hs-movies · 975bd1aproject fbdmd-hs-moviesmachine: Intel(R) Xeon(R) Processor @ 2.10GHz (Linux 6.18.44-fc-v153, x86_64, 4 cpus, python 3.13.17)

## Summary

- Three 24 s movies (1280×720, 30 fps, 721 frames) were produced from exact event-driven hard-sphere dynamics: a neutral control of N = 1000 spheres at η = 0.35 and the same box with 20 + and 20 − monovalent charges (l_B = 5 σ), rendered as opaque CPK spheres with OVITO's Tachyon ray tracer, plus an x-ray variant that ghosts the neutrals.

- The hard-sphere engine reproduces the Carnahan–Starling equation of state to 0.5 % (Z = 5.233 vs 5.206) and the Enskog collision rate to 0.5–0.6 %; no pair of the 721 000 rendered sphere positions comes closer than 1.0000001 σ.

- The Coulomb part is integrated with the collision Verlet scheme of Houndonougbo, Laird and Leimkuhler (Mol. Phys. 98, 309, 2000). Implemented literally, the scheme is only (sub-)first order on this system (slope 0.73) because ions collide with force-free neutrals while the long-range force is non-zero. A one-line modification that reflects the physical velocity at collisions (cv-phys) restores second order (slope 2.02) and conserves energy to 2.4e-05 over 18 τ.

- The charged trajectory is a partially associated electrolyte: 38 ± 3 % of the ions sit in a +/− contact pair at any instant, pairs live 0.5 τ on average (0.7 s of movie) and exchange partners, 10 % of the ions form short linear chains, and nothing crystallises. Screening by the other ions reduces the contact potential of mean force from the bare −5 kT to about −1.9 kT.

3 × 24 smovies, 720p30
5.233Z vs CS 5.206
2.4e-05max |δE|/E, 18 τ
38 ± 3 %ions in +/− contact
2.02 / 0.73order: cv-phys / literal CV

## 1. Goals and scope

The brief asked for short mp4 movies that look like Accelrys Materials Studio CPK views of the classic scout system used around FB-DMD / restricted-primitive-model work: about 1000 hard spheres in a periodic box at liquid-like packing, once neutral and once with 20 positive and 20 negative monovalent point charges on a subset of the spheres, so that clustering, pairing and free flight between collisions can be seen. The deliverable is visualization of trajectories, not a production study: the integrator had to be correct and stable (event-driven dynamics for the hard cores, collision or impulsive Verlet after Houndonougbo, Laird and Leimkuhler for the Coulomb part), the movies had to be reproducible from one script on a CPU-only machine, and the project had to stay free of GPU dependencies and unrelated code.

Everything described here lives in fbdmd-hs-movies/ and is reproduced by bash run_all.sh. Beyond the two required movies the work produced a third movie with ghosted neutrals, a validated single-file C++ simulator with four selectable integrators, an energy-order test in the style of HLL's Fig. 4, and the physics finding about the literal collision Verlet scheme reported in Section 4.2.

## 2. System and units

All quantities are in reduced units: sphere diameter σ, mass m and thermal energy kT are one, so the time unit is τ = σ√(m/kT). At η = 0.35 the box edge is L = (Nπ/6η)1/3 = 11.437 σ. Charges are ±e on spheres of the same diameter (restricted primitive model); the single coupling parameter is the Bjerrum length l_B = e²/(4πε kT), set to 5 σ, i.e. a reduced temperature T* = σkT/(e²/4πε) = 0.2 and a bare +/− contact energy of −5 kT. For orientation, an aqueous 1:1 electrolyte at room temperature has l_B/σ ≈ 1.8 for σ ≈ 0.4 nm, and the RPM critical point sits near T* ≈ 0.05; the chosen state is a moderately coupled electrolyte in which Bjerrum pairs form but do not dominate. The 40 ions have a mean spacing of 3.3 σ and an inverse Debye length κσ = 1.3.

Table 1. System and run parameters (reduced units: σ = m = kT = 1, τ = σ√(m/kT)).

| quantity | neutral control | charged run |
|---|---|---|
| N (spheres) | 1000 | 1000 |
| packing fraction η | 0.35 | 0.35 |
| box edge L / σ | 11.437 | 11.437 |
| charges | none | 20 × +e, 20 × −e on randomly chosen spheres |
| Bjerrum length l_B / σ (T* = 1/l_B*) | — | 5 (T* = 0.20) |
| electrostatics | — | 3D Ewald, α = 5.6/L, |n|∞ ≤ 6, tin-foil |
| integrator | event-driven MD (exact) | collision Verlet (HLL 2000), cv-phys rule, h = 0.005 τ |
| equilibration | 5 τ, velocity rescaling every 0.1 τ | 5 τ neutral + 15 τ with charges, rescaling every 0.1 τ |
| production (NVE) | 18 τ, 721 frames every 0.025 τ | 18 τ, 721 frames every 0.025 τ |
| seed | 1 | 1 (identical neutral start) |
| simulation wall time | 1.7 s | 10.0 s |

## 3. Methods

### 3.1 Pipeline

Figure 1 shows the pipeline. A single C++17 program (sim/hsdmd.cpp, no dependencies, builds with g++ and clang++) generates the trajectory and writes an extended-XYZ file that OVITO reads directly, plus per-frame diagnostics (kinetic, potential and total energy, temperature, collision count, minimum pair distance, number of +/− contacts). A Python wrapper converts the trajectory to a compressed .npz cache and a JSON metadata file. sim/analyze.py re-reads the float64 trajectory and checks every frame for overlaps, verifies the file contract, and computes energy, stationarity, pairing and cluster statistics. Rendering goes through render/render_ovito.py (OVITO 3.17 Python module, Tachyon ray tracer) or, when OVITO is unavailable, through a dependency-free numpy/Pillow rasteriser with the same command line; both write PNG frames that ffmpeg encodes to H.264. run_all.sh strings the stages together and exposes the physical and rendering parameters as environment variables.

![Figure 1. Pipeline of the project: run_all.sh orchestrates the C++17 simulator (event-driven hard-sphere MD with collision Verlet and Ewald/DSF Coulomb), which writes the trajectory as extended XYZ plus an npz cache, JSON metadata and a per-frame diagnostics CSV; sim/analyze.py scans every frame for overlaps and computes energy drift and pairing statistics, while render_ovito.py (Tachyon ray tracer, numpy fallback) produces 721 PNG frames per movie that ffmpeg encodes to H.264 mp4.](figures/fig_pipeline.png)

*Figure 1. Pipeline of the project: run_all.sh orchestrates the C++17 simulator (event-driven hard-sphere MD with collision Verlet and Ewald/DSF Coulomb), which writes the trajectory as extended XYZ plus an npz cache, JSON metadata and a per-frame diagnostics CSV; sim/analyze.py scans every frame for overlaps and computes energy drift and pairing statistics, while render_ovito.py (Tachyon ray tracer, numpy fallback) produces 721 PNG frames per movie that ffmpeg encodes to H.264 mp4.*

### 3.2 Event-driven hard-sphere dynamics

The hard cores are propagated exactly by event-driven molecular dynamics (Alder and Wainwright, 1957; Rapaport, 2004). Positions are stored lazily at each particle's last update time and advanced ballistically on demand. A cell list with cells of at least one diameter restricts collision prediction to the 27 neighbouring cells; predicted collisions and cell crossings go into a binary heap with per-particle collision counters for lazy invalidation. For an approaching pair (b = r·v < 0) the collision time follows from b² − v²(r² − σ²) ≥ 0, roundoff overlaps are treated as immediate collisions, and the elastic collision exchanges the normal velocity components. The collisional virial Σ σ|Δp| accumulated over the production run gives the compressibility factor Z = 1 + Σσ|Δp|/(3NkT t), which is compared with Carnahan–Starling in Section 4.1.

Frames are not written at events. At each uniform frame time t_k = k·0.025 τ every position is evaluated on its current ballistic segment, so the movie shows the true free flight between collisions and never a jittery event sequence; the reviewers confirmed that 49.1 % of the neutral spheres have an identical displacement over two consecutive frame intervals, against 49.2 % expected from exp(−2ΓΔt_frame) with the Enskog collision rate Γ.

### 3.3 Electrostatics

The Coulomb energy U = l_B kT Σ q_i q_j / r_ij of the 40 ions in the periodic box is evaluated with a standard three-dimensional Ewald sum (α = 5.6/L, real-space minimum images, reciprocal vectors with |n|∞ ≤ 6, self term, tin-foil boundary). With only 40 charges this costs about a thousand k-vectors per evaluation and is cheap. The implementation was checked against an independent Python Ewald sum (energies and forces agree to 10⁻¹⁴ at equal parameters and to 10⁻⁴ relative against a converged reference), against finite-difference forces (10⁻¹⁰), and against the NaCl Madelung constant. A Fennell–Gezelter damped-shifted-force Coulomb is available as --coulomb dsf; with the cut-off at L/2 it weakens the +/− contact energy by about 30 % and is kept only for comparisons.

### 3.4 Collision Verlet for the Coulomb part

Mixed hard-core/continuous systems cannot be stepped by plain Verlet. The naive splitting (HLL eq. 9: half-kick with the continuous force, exact hard-sphere flow over the step, half-kick) has a local energy error of order h at every step that contains a collision whenever the continuous force is non-zero at contact (HLL eq. 18), which is the case for Coulomb. Houndonougbo, Laird and Leimkuhler therefore split the pair potential into a short-range part φ₁ that is non-zero only below a distance q₂, and a long-range part φ₂ that is replaced below q₂ by a Hermite cubic P(r) with zero slope and curvature at q₁ (Figure 2a, b):

\[ P(r) = \varphi_c(q_2) + \varphi_c'(q_2)\,\frac{(r-q_1)^3-(q_2-q_1)^3}{3\,(q_2-q_1)^2},\qquad \varphi_2 = \begin{cases} P(q_1) & r<q_1\\ P(r) & q_1\le r<q_2 \\ \varphi_c(r) & r\ge q_2\end{cases},\qquad \varphi_1=\varphi_c-\varphi_2 . \]

The Hamiltonian is split into H₁ = T + V_hs + V₁ and H₂ = V₂. One step of length h applies a half-kick with F₂, propagates H₁ exactly for a time h, and applies the second half-kick with F₂ (HLL Table 1). Inside the H₁ propagation every particle takes one velocity-Verlet sub-step with F₁ between consecutive hard collisions anywhere in the system, so the accelerated ions move on parabolic paths and their collision times are the smallest positive root of the quartic

\[ \left|\Delta q + \Delta v\,\tau + \tfrac12\,\Delta a\,\tau^2\right|^2 = \sigma^2 , \]

solved on monotone pieces by bracketing and Newton refinement (Figure 2c). F₂ is computed once per step as the full Ewald force minus the pairwise F₁; F₁ acts only between ions closer than q₂ = 1.5 σ, so in practice only a handful of ions are accelerated at any time. The implementation (--integrator cv) was checked line by line against Table 1 and against an independent re-implementation of the algorithm driven with the same particles, which agreed to 10⁻¹³ in positions and velocities over hundreds of steps.

The order test of Section 4.2 showed that the literal scheme is nevertheless first order on this system. HLL's argument needs the long-range force to vanish for the colliding pair; an ion that collides with a neutral sphere, or a +/− pair in the field of a third ion, feels a non-zero F₂ at the instant of contact, and the eq.-18 error term reappears with F₂ in place of φ_c′. The shipped default cv-phys keeps the HLL splitting and sub-steps but reflects the physical velocity v + F₂(τ₁ − h/2) at each collision and returns to the drift frame afterwards, which removes the leading error term; a drift-frame separation guard and a zero-time-event watchdog make it robust at grazing contacts. Two further variants are kept for comparison: kdk-phys (kick–drift–kick with the same collision rule) and naive (HLL eq. 9).

![Figure 2. Collision Verlet (Houndonougbo, Laird & Leimkuhler 2000) as implemented. (a) The +/− Coulomb pair potential φc = −l_B/r (l_B = 5 σ) is split into a long-range part φ₂ that follows the Hermite cubic P(r) on [q₁, q₂] = [1, 1.5] σ and is flat at P(q₁) = −3.70 kT inside q₁, and a short-range part φ₁ = φc − φ₂ that vanishes beyond q₂; the hard core r < σ is shaded. (b) The corresponding pair forces: F₂ and its slope vanish at contact, F₁ carries the full contact force −5 kT/σ and goes to zero continuously at q₂. (c) One step h: outer half-kicks with F₂, exact hard-sphere propagation in between in which every hard collision (anywhere in the system) cuts the step into velocity-Verlet sub-steps with F₁; accelerated ions follow parabolic paths and their collision times are the smallest positive root of the quartic |Δq + Δvτ + ½Δaτ²|² = σ². The shipped cv-phys variant reflects the physical velocity v + F₂(τᵢ − h/2) at each collision instead of the drift velocity.](figures/fig_cv_scheme.png)

*Figure 2. Collision Verlet (Houndonougbo, Laird & Leimkuhler 2000) as implemented. (a) The +/− Coulomb pair potential φc = −l_B/r (l_B = 5 σ) is split into a long-range part φ₂ that follows the Hermite cubic P(r) on [q₁, q₂] = [1, 1.5] σ and is flat at P(q₁) = −3.70 kT inside q₁, and a short-range part φ₁ = φc − φ₂ that vanishes beyond q₂; the hard core r < σ is shaded. (b) The corresponding pair forces: F₂ and its slope vanish at contact, F₁ carries the full contact force −5 kT/σ and goes to zero continuously at q₂. (c) One step h: outer half-kicks with F₂, exact hard-sphere propagation in between in which every hard collision (anywhere in the system) cuts the step into velocity-Verlet sub-steps with F₁; accelerated ions follow parabolic paths and their collision times are the smallest positive root of the quartic |Δq + Δvτ + ½Δaτ²|² = σ². The shipped cv-phys variant reflects the physical velocity v + F₂(τᵢ − h/2) at each collision instead of the drift velocity.*

### 3.5 Equilibration and production protocol

Both runs start from the same seed and the same simple-cubic lattice with Maxwell–Boltzmann velocities. Stage A equilibrates the neutral fluid for 5 τ with velocity rescaling to kT = 1 every 0.1 τ (about 70 collisions per sphere). The charged run then assigns +e to 20 and −e to 20 randomly chosen spheres and equilibrates for another 15 τ with rescaling; the pairing relaxation time is 4–6 τ and a 5 τ stage was shown to leave the production about 2σ above the equilibrium pair count, so 15 τ is used. Production is 18 τ of NVE dynamics with a frame every 0.025 τ (721 frames). The neutral trajectory of the charged run's stage A is bit-identical to the neutral control run.

### 3.6 Rendering

The CPK look follows Materials Studio's defaults: opaque spheres of diameter σ, light gray for neutrals and saturated red (+) and blue (−) for the ions, a light studio background, faint dark-gray box edges, a perspective camera in a three-quarter view that orbits 40° over the movie, and a small caption with the system parameters and the time in τ. Tachyon renders with 16 ambient-occlusion samples and 4× antialiasing; the camera distance is fixed once from the widest orbit azimuth so the box never pops. The x-ray variant gives neutrals 85 % transparency so that all 40 ions and their contacts are visible inside the box. Frames are encoded with libx264 (CRF 18, yuv420p, faststart). On the 4-core test machine a 720p frame takes about 1.4 s (CPK) or 1.2 s (x-ray), 17 and 15 minutes per movie; the numpy fallback needs 0.2 s per frame with four workers and produces the same scene with a slightly glossier look (Figure 10).

### 3.7 Verification process

The simulator carries 50 self-tests (two-body collisions including across periodic faces, brute-force overlap scans, Ewald and DSF finite-difference forces, the Madelung constant, the potential splitting identities, the quartic collision-time solver against brute-force time sampling, bound-pair energy scaling for all integrators, grazing-contact and livelock cases). Beyond that, every component was reviewed adversarially by independent agents that re-derived the formulas, re-implemented Ewald and HLL Table 1, ran time-reversal tests, re-computed every statistic quoted here from the raw trajectories, decoded and inspected the encoded movies, and reproduced the full pipeline from a clean copy of the project with the default, the literal-CV and the numpy-renderer settings. The numbers in this report are the audited ones.

## 4. Results

### 4.1 Hard-sphere validation

Table 2 collects the checks on the production runs. The neutral run reproduces the Carnahan–Starling compressibility factor to 0.5 % (5.233 vs 5.206; the remaining difference is within the statistical error of an 18 τ virial average) and the Enskog collision rate to 0.6 %. The kinetic energy is conserved to machine precision because no energy is exchanged between kinetic and potential forms, total momentum stays at 10⁻¹³, and the smallest centre distance among the 721 000 sphere positions written to the movie is 1.000000089 σ. The reviewers' exact time-reversal test (run forward, flip velocities, run back) returned the initial configuration to 10⁻¹¹ with the same number of collisions, which is the sharpest statement of correctness for an event-driven code. Figure 5a shows the radial distribution function against the Carnahan–Starling contact value, and Figure 6c the free-flight interpolation check.

Table 2. Hard-sphere and integrator validation of the production runs (independent re-computation by the reviewers agreed with every entry).

| check | neutral control | charged run | reference |
|---|---|---|---|
| collisional virial Z = PV/NkT | 5.233 | 5.224 | Carnahan–Starling 5.206 |
| collision rate per sphere per τ | 14.32 (+0.6 %) | 14.31 (+0.5 %) | Enskog 14.24 |
| kinetic energy in production | constant to 4×10⁻¹⁴ | T = 0.9990 ± 0.0027 | kT = 1 (NVE scatter ±0.003) |
| total energy drift (E_end − E₀)/|E₀| | 0 (no potential) | 2.3e-05 (max 2.4e-05) | — |
| min. pair distance over all 721 frames | 1.000000089 σ | 1.000000381 σ | ≥ 1 σ (no overlap, no tunnelling) |
| free-flight interpolation check | 0.491 of spheres exactly ballistic over 2 frames | 0.492 (neutrals) | exp(−2Γ_E Δt_frame) = 0.492 |
| time reversibility (reviewer test) | x(−t) returns to 1.7×10⁻¹¹ | — | exact for EDMD |
| Ewald vs independent implementation | — | ΔU = 3×10⁻¹⁴, ΔF = 2×10⁻¹⁴ | same α, k_max |

### 4.2 Integrator study

Figure 3 and Table 3 give the energy-error order test on the production system itself: after the standard equilibration, 2 τ of NVE production were run at h = 0.02, 0.01, 0.005 and 0.0025 τ for each integrator and the maximum |E(t) − E₀|/N recorded. The naive splitting is first order, as HLL found. The literal collision Verlet is roughly first order here (slope 0.73, 35× less accurate than cv-phys at h = 0.005 τ) and worse than the plain kick–drift–kick with the physical-velocity rule at every step size. Several diagnostics isolate the cause: a two-body +/− pair in a large box, where every collision is an ion–ion contact with F₂ = 0, gives slope 1.9 for the literal scheme; raising q₂ to 3 σ or 5.5 σ shrinks the error by at most 3× without changing the slope; an all-ion system (no neutrals) is still first order; and with the smooth DSF potential the literal scheme reaches only slope 1.2 while cv-phys reaches 1.9. The residual is therefore the eq.-18 error with F₂ at ion–neutral collisions and at +/− contacts in third-body fields, which HLL's one-component argument does not cover. cv-phys reaches slope 2.02 with Ewald and is the most accurate choice at every h; at h = 0.0025 the two accurate schemes reach a floor of about 10⁻⁶ kT per particle that comes from the Ewald real-space truncation at r_c = L/2 (erfc(2.8) ≈ 7.5×10⁻⁵) rather than from the integrator, whereas the literal scheme and the naive splitting are flat there because of their own first-order collision error.

![Figure 3. Energy-error order test on the production system (N = 1000, η = 0.35, 20+/20−, l_B = 5 σ): maximum |E(t) − E(0)|/N over 2 τ of NVE dynamics versus the step h for the four integrators, with the fitted log–log slopes in the legends and dashed reference lines ∝ h and ∝ h². (a) Equilibrated start (15 τ with charges on): the shipped cv-phys is second order (slope 2.02) whereas the literal HLL collision Verlet is only sub-first-order (slope 0.73) and 35× less accurate at the production step h = 0.005 τ; kdk-phys gives 1.82 and the naive splitting 1.09. (b) Same test from the un-equilibrated start with Ewald (solid; slopes 1.40 / 0.78 / 1.69 / 0.89) and with the smooth DSF Coulomb model (dashed, cv-phys 1.92, cv 1.21). The Ewald real-space truncation erfc(αr_c) ≈ 7.5×10⁻⁵ gives an energy floor of ≈ 0.5–1 × 10⁻⁶ kT/N that limits the cv-phys curve at h = 0.0025 τ (local slope 1.0 between 0.005 and 0.0025 τ; 6 × 10⁻⁷ kT/N in (a)); cv and naive are flat there because of their own first-order collision error (65× and 229× above the floor, local slopes 0.11 and −0.06), and kdk-phys is not yet affected (local slope 2.9).](figures/fig_order.png)

*Figure 3. Energy-error order test on the production system (N = 1000, η = 0.35, 20+/20−, l_B = 5 σ): maximum |E(t) − E(0)|/N over 2 τ of NVE dynamics versus the step h for the four integrators, with the fitted log–log slopes in the legends and dashed reference lines ∝ h and ∝ h². (a) Equilibrated start (15 τ with charges on): the shipped cv-phys is second order (slope 2.02) whereas the literal HLL collision Verlet is only sub-first-order (slope 0.73) and 35× less accurate at the production step h = 0.005 τ; kdk-phys gives 1.82 and the naive splitting 1.09. (b) Same test from the un-equilibrated start with Ewald (solid; slopes 1.40 / 0.78 / 1.69 / 0.89) and with the smooth DSF Coulomb model (dashed, cv-phys 1.92, cv 1.21). The Ewald real-space truncation erfc(αr_c) ≈ 7.5×10⁻⁵ gives an energy floor of ≈ 0.5–1 × 10⁻⁶ kT/N that limits the cv-phys curve at h = 0.0025 τ (local slope 1.0 between 0.005 and 0.0025 τ; 6 × 10⁻⁷ kT/N in (a)); cv and naive are flat there because of their own first-order collision error (65× and 229× above the floor, local slopes 0.11 and −0.06), and kdk-phys is not yet affected (local slope 2.9).*

Table 3. Order test (equilibrated start, production system, 2 τ): maximum |E(t)−E₀|/N in kT (end drift (E_end−E₀)/N in parentheses), from notes/order_test.csv.

| h (τ) | cv-phys | cv | kdk-phys | naive |
|---|---|---|---|---|
| 0.02 | 4.75e-05 (4.54e-05) | 3.81e-04 (-1.46e-04) | 6.74e-05 (4.72e-05) | 2.38e-03 (2.38e-03) |
| 0.01 | 6.83e-06 (6.83e-06) | 1.15e-04 (7.22e-05) | 3.43e-05 (3.43e-05) | 9.16e-04 (-7.96e-04) |
| 0.005 | 2.86e-06 (1.99e-06) | 1.00e-04 (9.88e-05) | 5.98e-06 (4.86e-06) | 3.89e-04 (2.68e-04) |
| 0.0025 | 5.97e-07 (-7.55e-08) | 7.47e-05 (-7.47e-05) | 1.82e-06 (6.26e-07) | 2.53e-04 (-1.92e-04) |
| log–log slope | 2.02 | 0.73 | 1.82 | 1.09 |

Table 4 compares the four integrators on the full 18 τ production. The thermodynamic state (pair count, potential energy) agrees between all of them within statistical scatter (seven seeds each, Welch t ≤ 1.4), so the choice affects the energy budget, not the physics seen in the movie; cv-phys costs the same as the literal scheme (about 2× the plain kick–drift–kick) and was used for the shipped movie.

Table 4. The four charged-run integrators on the full 5 + 15 + 18 τ production (seed 1).

| integrator | what it is | ΔE/|E₀| over 18 τ | max |δE|/|E₀| | sim wall |
|---|---|---|---|---|
| cv-phys (default) | HLL splitting + Verlet sub-steps; collisions reflect v + F₂(τ₁ − h/2) | +2.3e-5 | 2.4e-5 | ≈10 s |
| cv | HLL collision Verlet exactly as published | −1.3e-4 | 1.6e-4 | ≈11 s |
| kdk-phys | kick–drift–kick (HLL eq. 9) with the physical-velocity collision rule | +5.2e-5 | 5.2e-5 | ≈5 s |
| naive | kick–drift–kick exactly as HLL eq. 9 (Suh et al.) | +1.0e-3 | 1.3e-3 | ≈5 s |

### 4.3 Energy, temperature and stationarity

Figure 4 shows the energy budget of the production runs. The total energy of the charged cv-phys run rises by 2.3e-05 relative (0.034 kT in 1425 kT) over 18 τ, almost linearly, while the kinetic and potential energies exchange about ±4 kT as pairs form and break; the temperature is 0.9990 ± 0.003, the NVE scatter expected from potential-energy fluctuations. The first and second halves of the run agree in temperature, potential energy and pair count within one standard deviation of the block averages, so the movie starts from an equilibrated state. The cumulative collision count grows linearly at the Enskog rate in both runs; ions collide 13.3 times per τ, neutrals 13.6, so the charges do not heat up or slow down relative to the solvent.

![Figure 4. Energy conservation and collision statistics of the production runs (N = 1000, η = 0.35, 18 τ NVE). (a) Total-energy deviation E(t) − E(0) for the four integrators of the charged system (20+/20−, l_B = 5 σ, Δt = 0.005 τ): the naive splitting drifts by +1.48 kT, literal collision Verlet (cv) by −0.19 kT, kdk-phys by +0.075 kT and the shipped cv-phys by +0.033 kT (2.3 × 10⁻⁵ of |E₀|; inset without naive). (b) cv-phys: kinetic and potential energy fluctuate in antiphase with a standard deviation of 4.0 kT each while their sum changes by 0.034 kT at most. (c) Instantaneous temperature of the cv-phys run, ⟨T⟩ = 0.9990 ± 0.0027 kT. (d) Cumulative hard-sphere collisions in the neutral and charged runs against the Enskog rate N Γ_E/2 with Γ_E = 14.24 per particle per τ; the neutral run exceeds Enskog by 0.6 % and the charged run by 0.5 % (inset).](figures/fig_energy.png)

*Figure 4. Energy conservation and collision statistics of the production runs (N = 1000, η = 0.35, 18 τ NVE). (a) Total-energy deviation E(t) − E(0) for the four integrators of the charged system (20+/20−, l_B = 5 σ, Δt = 0.005 τ): the naive splitting drifts by +1.48 kT, literal collision Verlet (cv) by −0.19 kT, kdk-phys by +0.075 kT and the shipped cv-phys by +0.033 kT (2.3 × 10⁻⁵ of |E₀|; inset without naive). (b) cv-phys: kinetic and potential energy fluctuate in antiphase with a standard deviation of 4.0 kT each while their sum changes by 0.034 kT at most. (c) Instantaneous temperature of the cv-phys run, ⟨T⟩ = 0.9990 ± 0.0027 kT. (d) Cumulative hard-sphere collisions in the neutral and charged runs against the Enskog rate N Γ_E/2 with Γ_E = 14.24 per particle per τ; the neutral run exceeds Enskog by 0.6 % and the charged run by 0.5 % (inset).*

### 4.4 Structure

Figure 5a is the neutral–neutral radial distribution function of the control run. Its contact value extrapolated from 1.0–1.1 σ is g(σ⁺) = 3.01, against the Carnahan–Starling (1 − η/2)/(1 − η)³ = 3.00 at η = 0.35 and 3.02 implied by the collisional virial, and the first minimum and second peak sit where a hard-sphere liquid at this density has them. Figure 5b shows the ion–ion correlations of the charged run: a strong +/− contact peak (g₊₋ = 16.8 in the contact bin, about 20 extrapolated to σ⁺), like-charge pairs depleted at contact (g ≈ 0.5) and back to the bulk value by 2 σ, the + − + chain distance, and ion–neutral correlations that follow the neutral fluid beyond the first shell with a 6 % deficit in the contact bin, the sites taken by counter-ion partners. Figure 5c converts the +/− correlation into a potential of mean force relative to the hard-sphere background, w₊₋(r) = −kT ln[g₊₋/g_HS]. In the contact bin w₊₋ = −1.81 ± 0.02 kT, about −1.9 kT extrapolated to σ⁺, far above the bare Coulomb −5 kT and close to a Debye–Hückel estimate −l_B/(1 + κσ) = −2.2 kT with κσ = 1.3: the other 38 ions screen the contact attraction substantially even at this low ion fraction. This is why the ideal Bjerrum mass-action estimate (about 61 % paired at the 1.2 σ criterion) overshoots the measured 38 %.

![Figure 5. Static structure. (a) Neutral–neutral pair distribution of the hard-sphere run (η = 0.35, 0.01 σ bins, 721 frames, minimum image): the contact value extrapolated from 1.0–1.1 σ is g(σ⁺) = 3.01, against the Carnahan–Starling value 3.00 and 3.02 implied by the collisional virial Z = 5.233. (b) Ion correlations in the charged run (0.05 σ bins, all 721 frames; error bars are Poisson): g₊₋ reaches 16.8 ± 0.4 in the contact bin (≈20 extrapolated to σ⁺); same-sign pairs are depleted at contact (g ≈ 0.5), recover the bulk value by ≈2 σ (the + − + chain distance; 0.98 ± 0.04 over 1.95–2.25 σ) and show a weak second depletion (g ≈ 0.84) at 2.3–2.75 σ; ion–neutral correlations follow the neutral fluid beyond the first shell, while in the contact bin g_ion–neutral = 2.57 against 2.75 for neutral pairs, a 6.5 % deficit corresponding to the contact sites occupied by counter-ion partners (0.15 per ion in that bin). (c) Potential of mean force w₊₋ = −ln[g₊₋/g_HS] compared with the bare Coulomb pair energy −l_B/r and the linearised Debye–Hückel screened potential (κσ = 1.30): the effective attraction at contact is −1.81 ± 0.02 kT, far weaker than the bare −5 kT, because the other 38 ions screen the pair; the DH curve is shown only as a scale since l_B/σ = 5 is outside its range of validity.](figures/fig_structure.png)

*Figure 5. Static structure. (a) Neutral–neutral pair distribution of the hard-sphere run (η = 0.35, 0.01 σ bins, 721 frames, minimum image): the contact value extrapolated from 1.0–1.1 σ is g(σ⁺) = 3.01, against the Carnahan–Starling value 3.00 and 3.02 implied by the collisional virial Z = 5.233. (b) Ion correlations in the charged run (0.05 σ bins, all 721 frames; error bars are Poisson): g₊₋ reaches 16.8 ± 0.4 in the contact bin (≈20 extrapolated to σ⁺); same-sign pairs are depleted at contact (g ≈ 0.5), recover the bulk value by ≈2 σ (the + − + chain distance; 0.98 ± 0.04 over 1.95–2.25 σ) and show a weak second depletion (g ≈ 0.84) at 2.3–2.75 σ; ion–neutral correlations follow the neutral fluid beyond the first shell, while in the contact bin g_ion–neutral = 2.57 against 2.75 for neutral pairs, a 6.5 % deficit corresponding to the contact sites occupied by counter-ion partners (0.15 per ion in that bin). (c) Potential of mean force w₊₋ = −ln[g₊₋/g_HS] compared with the bare Coulomb pair energy −l_B/r and the linearised Debye–Hückel screened potential (κσ = 1.30): the effective attraction at contact is −1.81 ± 0.02 kT, far weaker than the bare −5 kT, because the other 38 ions screen the pair; the DH curve is shown only as a scale since l_B/σ = 5 is outside its range of validity.*

### 4.5 Pairing and chains

Figure 6 and Table 5 quantify what the charged movie shows. With a +/− bond defined as r < 1.2 σ, 37.7 % of the ions are paired at any instant (±3 % statistical uncertainty of the run mean, ±11 % frame-to-frame spread; 8.3 pairs per frame, fluctuating between 2 and 15), 62 % are free, 10 % sit in clusters of three or more, and the clusters are almost always linear chains (+ − + or − + −); branched clusters occur in 1 % of the frames and the largest cluster seen has six ions. A contact episode lasts 0.52 τ on average (median 0.35 τ, 90th percentile 1.18 τ, longest 2.48 τ; the survival curve in Figure 6d is close to exponential), i.e. about 0.7 s of movie, and each cation meets 2.4 different anions over the run, so the pairs are transient and exchange partners rather than forming a permanent salt. The looser classical criterion r < l_B/2 = 2.5 σ counts 86 % of the ions as associated, but random 20 + 20 subsets of the neutral run already give 58.5 % by hard-sphere packing alone at this ion density (8 % for the 1.2 σ criterion), so the contact criterion is the informative one.

![Figure 6. Bjerrum pairing in the charged production run (20+/20−, l_B = 5 σ, cv-phys, 721 frames). (a) Number of +/− contact pairs (r < 1.2 σ) per frame, mean 8.3 (range 2–15). (b) Fraction of ions in a contact pair (38 ± 11 %) and fraction with an opposite ion within l_B/2 = 2.5 σ (86 %); dashed and dotted lines are the random-ion baselines obtained from 100 random 20+20 subsets of the neutral run, 0.585 ± 0.006 (2.5 σ) and 0.080 ± 0.002 (1.2 σ; standard errors of the subset mean); the hard-core-only estimate 1 − (1 − V_shell/V)^20 = 0.567 for 2.5 σ is a consistent cross-check. (c) Fraction of ions in connected +/− clusters of size 1–6: 10 % sit in clusters of three or more, almost all linear chains. (d) Survival function of contact-episode lifetimes: median 0.35 τ, mean 0.52 τ, 90th percentile 1.18 τ, longest 2.48 τ (3.3 s of movie); the dotted curve is an exponential with the same mean.](figures/fig_pairing.png)

*Figure 6. Bjerrum pairing in the charged production run (20+/20−, l_B = 5 σ, cv-phys, 721 frames). (a) Number of +/− contact pairs (r < 1.2 σ) per frame, mean 8.3 (range 2–15). (b) Fraction of ions in a contact pair (38 ± 11 %) and fraction with an opposite ion within l_B/2 = 2.5 σ (86 %); dashed and dotted lines are the random-ion baselines obtained from 100 random 20+20 subsets of the neutral run, 0.585 ± 0.006 (2.5 σ) and 0.080 ± 0.002 (1.2 σ; standard errors of the subset mean); the hard-core-only estimate 1 − (1 − V_shell/V)^20 = 0.567 for 2.5 σ is a consistent cross-check. (c) Fraction of ions in connected +/− clusters of size 1–6: 10 % sit in clusters of three or more, almost all linear chains. (d) Survival function of contact-episode lifetimes: median 0.35 τ, mean 0.52 τ, 90th percentile 1.18 τ, longest 2.48 τ (3.3 s of movie); the dotted curve is an exponential with the same mean.*

Table 5. Ion pairing and clustering in the charged production run (721 frames; +/− bond = r < 1.2 σ). Statistical uncertainty of the pairing fraction from one 18 τ run ≈ ±3 percentage points.

| quantity | value |
|---|---|
| ions in a +/− contact pair (mean / min / max) | 37.7 % / 10 % / 65 % |
| +/− contact pairs per frame (mean / min / max) | 8.3 / 2 / 15 |
| ions with an opposite ion within l_B/2 = 2.5 σ | 85.8 % (random 20+20 subsets of the neutral run: 58.5 %; 1.2 σ criterion: 8.0 %) |
| cluster sizes 1 / 2 / 3 / 4 / 5 / 6 (fraction of ions) | 62.3 / 27.5 / 8.1 / 1.7 / 0.3 / 0.2 % |
| clusters of ≥ 3 per frame: chains / branched | 1.27 / 0.011 |
| largest cluster (mean / max) | 3.0 / 6 ions |
| contact-episode lifetime (mean / median / 90 % / max) | 0.52 / 0.35 / 1.18 / 2.48 τ  (0.69 / 0.47 / 1.57 / 3.3 s of movie) |
| distinct − partners per + ion over 18 τ | 2.4 (1–6) |
| potential of mean force at contact, w₊₋ | −1.81 ± 0.02 kT in the 1.00–1.05 σ bin, ≈ −1.9 kT at σ⁺ (bare Coulomb −5 kT; Debye–Hückel −2.2 kT, κσ = 1.3) |

### 4.6 Dynamics

Figure 7 shows the single-particle dynamics behind the motion in the movies. The mean-squared displacement of the neutrals is ballistic below the collision time (0.07 τ) and diffusive beyond about 1 τ, with the same diffusion coefficient in both runs, D = 0.123–0.125 σ²/τ from the 1–4 τ slope (1.17 × the Enskog value); the ions follow the same curve within the statistics of 40 particles (D = 0.124 σ²/τ, short-lag ratio 1.04, long-time rms displacement 3.5 σ versus 3.7 σ for neutrals over 18 τ). A sphere moves 0.038 σ per frame on average and never more than 0.14 σ, which is why the 30 fps movie reads as continuous motion. Panel (c) is the free-flight interpolation check described in Section 3.2.

![Figure 7. Single-particle dynamics (trajectories unwrapped through the periodic box). (a) Mean-squared displacement versus lag time for neutral spheres in the neutral run, neutral spheres in the charged run and the 40 ions, with t² (ballistic) and t (diffusive) guides and the Enskog prediction; the slopes over 1–4 τ give D = 0.123, 0.125 and 0.124 σ²/τ (1.17 × Enskog), i.e. ions and neutrals diffuse alike. (b) Distribution of the displacement per frame (Δt_frame = 0.025 τ): mean 0.038 σ for neutrals, 0.039 σ for ions, maximum 0.14 σ; the dashed curve is the Maxwell speed distribution times Δt_frame. (c) Free-flight check: fraction of neutral spheres whose displacement vector is identical over two consecutive frame intervals, 0.491 on average against exp(−2ΓΔt_frame) = 0.489 for the measured collision rate Γ = 14.32 τ⁻¹.](figures/fig_dynamics.png)

*Figure 7. Single-particle dynamics (trajectories unwrapped through the periodic box). (a) Mean-squared displacement versus lag time for neutral spheres in the neutral run, neutral spheres in the charged run and the 40 ions, with t² (ballistic) and t (diffusive) guides and the Enskog prediction; the slopes over 1–4 τ give D = 0.123, 0.125 and 0.124 σ²/τ (1.17 × Enskog), i.e. ions and neutrals diffuse alike. (b) Distribution of the displacement per frame (Δt_frame = 0.025 τ): mean 0.038 σ for neutrals, 0.039 σ for ions, maximum 0.14 σ; the dashed curve is the Maxwell speed distribution times Δt_frame. (c) Free-flight check: fraction of neutral spheres whose displacement vector is identical over two consecutive frame intervals, 0.491 on average against exp(−2ΓΔt_frame) = 0.489 for the measured collision rate Γ = 14.32 τ⁻¹.*

### 4.7 The movies

Figure 8 shows frames from the three movies at the start, middle and end of the 40° orbit; Figure 9 zooms into the x-ray movie on a +/− contact pair and a short chain. In the opaque CPK movie only the ions on the three camera-facing faces are visible (typically 5–13 of the 40), which is faithful to a Materials Studio view of a dense liquid but hides most of the pairing; the x-ray movie exists for that reason. What to look for: in the neutral movie, every sphere rattles in the cage of its neighbours and the surface layer slowly exchanges with the interior; in the charged movies, red and blue spheres that come within a diameter stick for a few seconds, rattle against each other, are knocked apart by gray spheres and find new partners, and occasionally three or four ions line up alternately for a second or two. Charges never move faster than neutrals and no sphere ever passes through another.

![Figure 8. Stills taken from the three movies (frames 0, 360 and 720 of 721, i.e. t = 0, 9 and 18 τ of the NVE production run; the camera orbits from azimuth 25° to 65° over the movie): (a) the neutral control (1000 hard spheres, η = 0.35), (b) the charged run in the opaque CPK style (20 red + and 20 blue − ions, l_B = 5 σ), of which only the ions near the three camera-facing faces are visible, and (c) the same charged frames in the x-ray style with the neutrals 85 % transparent, which shows all 40 ions and their contact pairs inside the box.](figures/fig_stills.png)

*Figure 8. Stills taken from the three movies (frames 0, 360 and 720 of 721, i.e. t = 0, 9 and 18 τ of the NVE production run; the camera orbits from azimuth 25° to 65° over the movie): (a) the neutral control (1000 hard spheres, η = 0.35), (b) the charged run in the opaque CPK style (20 red + and 20 blue − ions, l_B = 5 σ), of which only the ions near the three camera-facing faces are visible, and (c) the same charged frames in the x-ray style with the neutrals 85 % transparent, which shows all 40 ions and their contact pairs inside the box.*

![Figure 9. Close-ups of the charged production run in the x-ray style of the movie (neutrals 85 % transparent, ions opaque; red +, blue −), re-rendered at 3× the movie resolution for the three frames chosen from a cluster scan with the r < 1.2 σ contact criterion: (a) an isolated +/− contact pair at the front face of the box at t = 16.27 τ (nearest other ion 2.5 σ away, no other ion inside the circle in projection, no neutral in front of it), (b) a linear four-ion chain (+ − + −) at t = 9.75 τ and (c) a linear five-ion chain (− + − + −) at t = 9.22 τ, 42 σ from the camera. Dotted circles enclose the cluster; the bar gives 1 σ at the depth of the cluster. Such chains are transient: clusters of four or more ions exist in 142 of the 721 frames and the largest cluster seen has six ions.](figures/fig_pair_closeup.png)

*Figure 9. Close-ups of the charged production run in the x-ray style of the movie (neutrals 85 % transparent, ions opaque; red +, blue −), re-rendered at 3× the movie resolution for the three frames chosen from a cluster scan with the r < 1.2 σ contact criterion: (a) an isolated +/− contact pair at the front face of the box at t = 16.27 τ (nearest other ion 2.5 σ away, no other ion inside the circle in projection, no neutral in front of it), (b) a linear four-ion chain (+ − + −) at t = 9.75 τ and (c) a linear five-ion chain (− + − + −) at t = 9.22 τ, 42 σ from the camera. Dotted circles enclose the cluster; the bar gives 1 σ at the depth of the cluster. Such chains are transient: clusters of four or more ions exist in 142 of the 721 frames and the largest cluster seen has six ions.*

Table 6. Movie specifications.

| file | content | format |
|---|---|---|
| movies/hs_neutral.mp4 | 1000 HS, η = 0.35, neutral control, EDMD | H.264 yuv420p, 1280×720, 30 fps, 721 frames, 24.0 s, 12.0 MB |
| movies/hs_charges_20p20m.mp4 | same + 20+/20−, l_B = 5 σ, collision Verlet; opaque CPK | same, 12.4 MB |
| movies/hs_charges_20p20m_xray.mp4 | same run, neutrals rendered 85 % transparent | same, 15.6 MB |

![Figure 10. Rendering cost per 1280×720 frame on the 4-core Xeon box (no GPU), measured on the 1000-sphere test trajectory including overlay and PNG writing: OVITO/Tachyon cpk at final quality (16 ambient-occlusion samples, 16 Tachyon threads) takes 1.43 s per frame, i.e. ≈ 17 min for a 721-frame movie, the x-ray preset 1.27 s (≈ 15 min), the AO-free draft preset 0.26 s (≈ 3 min) and the numpy fallback renderer with 4 worker processes 0.24 s wall (≈ 3 min). The annotated vertical ticks mark the three production movies: 1028 s and 1031 s wall for render plus encode of the two cpk movies (1.43 s per frame, two coinciding ticks) and 896 s for the x-ray movie (1.24 s per frame).](figures/fig_render_cost.png)

*Figure 10. Rendering cost per 1280×720 frame on the 4-core Xeon box (no GPU), measured on the 1000-sphere test trajectory including overlay and PNG writing: OVITO/Tachyon cpk at final quality (16 ambient-occlusion samples, 16 Tachyon threads) takes 1.43 s per frame, i.e. ≈ 17 min for a 721-frame movie, the x-ray preset 1.27 s (≈ 15 min), the AO-free draft preset 0.26 s (≈ 3 min) and the numpy fallback renderer with 4 worker processes 0.24 s wall (≈ 3 min). The annotated vertical ticks mark the three production movies: 1028 s and 1031 s wall for render plus encode of the two cpk movies (1.43 s per frame, two coinciding ticks) and 896 s for the x-ray movie (1.24 s per frame).*

## 5. Discussion

What the scout shows. At l_B = 5 σ and a 4 % ion fraction the system is a screened, partially associated electrolyte rather than either a free plasma or a salt: a third of the ions are in contact pairs at any moment, but those pairs live for half a collision-free diffusion step of the solvent and the ions keep exchanging partners. The measured contact potential of mean force, −1.9 kT, is the number to remember when comparing with Bjerrum-type estimates that use the bare −l_B/σ: screening by the other ions (κσ = 1.3) and the hard-sphere packing of the solvent together cut the effective attraction by more than half. Chains of three to six alternating ions appear as fluctuations and disappear again; nothing in 18 τ suggests the onset of larger aggregates at this coupling. For a movie that shows chains and clusters rather than pairs, l_B of 8–10 σ (T* ≈ 0.1) is the knob to turn.

Collision Verlet in a hard-sphere solvent. The order test is the most useful methodological result of this exercise. HLL demonstrated second order for a one-component fluid in which every collision is between particles whose mutual long-range force has been flattened to zero at contact. In an electrolyte in a neutral hard-sphere solvent most collisions are ion–neutral, and the ion carries the long-range field of the other ions through the collision; the local error −(ΔF₂·n)(h♭ − h#)Δv_n of their eq. 18 then survives with F₂ in place of φ_c′ and accumulates as a first-order random walk. Including the solvent collisions in the inner Verlet with the full force would cure it at the price of an Ewald evaluation per collision (7000 per τ here), which HLL themselves rule out for long-range potentials. The cv-phys rule is the cheap alternative: it corrects the momentum that is reflected at the collision from the half-kicked drift value to the physical value, which is exactly the quantity whose mismatch produces the eq.-18 term, and it recovers slope 2 while keeping the splitting and the sub-step structure of the published method. For FB-DMD / RPM production work this matters: a scheme that is advertised as second order can silently be first order in a mixture, and the per-step energy budget should be checked with an h-scan on the actual system rather than inferred from the single-component test.

Visualization choices. Opaque CPK rendering of a dense liquid shows a surface, not a bulk; the first renders made that obvious. The x-ray variant (neutrals at 85 % transparency) was added so that the interior ions and their contacts are visible while the solvent is still there as context. Frames are taken at uniform times from the exact ballistic segments, which is what makes the motion smooth at 30 fps; rendering events would produce jitter. Tachyon with 16 ambient-occlusion samples removed a static grain that 6 samples left on every sphere, and setting OVITO's thread count to four times the core count halved the render time because its per-pixel stage is otherwise single-threaded.

Limitations. All statistics come from one 18 τ run with 40 ions, so the pairing fraction carries a ±3 percentage-point uncertainty and the ion dynamics a few per cent. The Ewald real-space cut-off at L/2 introduces an energy-error floor of about 10⁻⁶ kT per particle that limits order tests below h = 0.0025 τ; a larger α or a second real-space image would remove it. The DSF option is a weaker-coupled model than the Ewald default at this box size and should not be mixed with it quantitatively. The pipeline was executed and audited on a 4-core Linux cloud machine; it was made portable to macOS (clang, stock bash 3.2, no GPU) by inspection and a clean-copy run, but not executed on a Mac. The rendering is CPU ray tracing and takes about 50 minutes for the three movies at full quality.

## 6. Conclusion

The requested movies exist, are reproducible from one script on a CPU-only machine, and show physically exact hard-sphere dynamics with a second-order, energy-conserving treatment of the Coulomb interactions. The neutral control validates the event-driven engine against Carnahan–Starling and Enskog to half a per cent and against exact time reversal. The charged scout at l_B = 5 σ is a partially associated electrolyte with transient Bjerrum pairs (38 ± 3 % of ions in contact, 0.5 τ lifetimes, 2.4 partners per ion over the run), occasional short chains and no larger aggregates; screening reduces the contact attraction from −5 kT to −1.9 kT. The collision Verlet scheme of Houndonougbo, Laird and Leimkuhler, implemented and verified line by line, is first order rather than second order in a hard-sphere solvent because of ion–neutral collisions under the long-range force; reflecting the physical velocity at collisions restores second order at no extra cost and is the integrator used for the movie. All code, trajectories' metadata, analyses and movies are committed in the project folder together with the documentation needed to rerun or extend them (other couplings, box sizes, dark background, 1080p).

## Appendix

### A. Reproduction

```
cd fbdmd-hs-movies
bash run_all.sh                                   # venv, build, selftest, 2 sims, analysis, 3 renders, NOTES.md (~1 h on 4 cores)
QUALITY=draft WIDTH=640 HEIGHT=360 T_RUN=6 bash run_all.sh      # quick preview (~3 min)
INTEGRATOR=cv bash run_all.sh                     # literal HLL collision Verlet instead of cv-phys
ORDER_TEST=1 bash run_all.sh                      # also run the energy-order test (notes/order_test.{txt,csv})
RENDERER=numpy bash run_all.sh                    # dependency-free renderer
# by hand
make -C sim && sim/hsdmd --selftest
python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20 --bjerrum 5 --integrator cv-phys --dt 0.005 \
       --t-equil-neutral 5 --t-equil-charged 15 --t-run 18 --dt-frame 0.025 --seed 1
python sim/analyze.py traj/hs_charges_20p20m
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m --out movies/hs_charges_20p20m.mp4 --style cpk
python report/build_report.py                     # this report
```

### B. Success criteria

Table 7. Success criteria from the brief.

| criterion | status | evidence |
|---|---|---|
| run_all.sh produces both mp4s on a CPU-only machine | met | clean-copy runs (default, INTEGRATOR=cv, RENDERER=numpy) exit 0; bash 3.2-safe; no CUDA/Metal anywhere |
| charged movie shows ± spheres in distinct colours vs gray neutrals | met | red/blue/gray hue preserved after yuv420p; 0 red/blue pixels in the neutral movie |
| motions look physical (no tunnelling, charges do not explode) | met | min pair distance 1.0000004 σ over 721 000 positions; ion and neutral displacement statistics identical; energy conserved to 2.4e-05 |
| README has exact run commands and file paths | met | every quoted flag/default/path verified against the CLIs |
| no secrets, no Kalshi/trading code | met | grep of the project folder finds only the two self-referential statements |
| Materials Studio CPK look, 10–30 s, 24–30 fps, 720p, smooth free flight | met | Tachyon AO rendering, 24 s @ 30 fps, frames at uniform times from the exact ballistic flight |

### C. Files

```
fbdmd-hs-movies/
  run_all.sh  README.md  NOTES.md  FORMAT.md  requirements.txt  make_notes.py
  sim/hsdmd.cpp  sim/Makefile  sim/run_sim.py  sim/analyze.py  sim/order_test.py  sim/README_sim.md
  render/render_ovito.py  render/render_numpy.py  render/common.py  render/trajio.py  render/make_test_traj.py  render/README_render.md
  notes/collision_verlet_spec.md  notes/analysis_*.txt  notes/order_test.{txt,csv}  notes/render_timing.txt  notes/observations.md
  traj/hs_neutral.{json,_diag.csv}  traj/hs_charges_20p20m.{json,_diag.csv}     (xyz/npz regenerated by run_all.sh)
  movies/hs_neutral.mp4  movies/hs_charges_20p20m.mp4  movies/hs_charges_20p20m_xray.mp4
  tools/export_vmd_travis.py  tools/compare_travis.py  vmd/render_cpk.tcl  vmd/view_cpk.tcl  vmd/render_vmd.sh  travis/README_travis.md  travis/run_travis.sh
  report/report.html  report/REPORT.md  report/report.pdf  report/figures/  report/data/
```

### D. VMD and TRAVIS

The trajectories are also exported for the tools installed on the target Mac. tools/export_vmd_travis.py writes an element-labelled multi-frame XYZ, a LAMMPS dump with the periodic box in every frame and a PDB with a CRYST1 record, mapping the reduced units to 1 σ = 4 Å (Ar = neutral sphere, Na = +e, Cl = −e) so that neither program detects spurious bonds. vmd/render_cpk.tcl reproduces the CPK look (VDW spheres of radius σ/2, the same gray/red/blue, ambient occlusion and shadows, 40° orbit, faint box) with VMD's bundled Tachyon and vmd/render_vmd.sh encodes the result; vmd/view_cpk.tcl opens the same scene interactively. travis/README_travis.md gives the TRAVIS session (cubic cell 4574.78 pm, 25 fs per frame so that 1 ps = 1 τ) for radial distribution functions, the aggregation/dimer-lifetime analysis with the 480 pm = 1.2 σ contact criterion, hierarchical cluster analysis and the mean-squared displacement from the unwrapped export, and tools/compare_travis.py overlays a TRAVIS g(r) on the one computed here. Neither VMD nor TRAVIS could be run in the cloud session (their downloads are blocked there); the scripts were syntax-checked and reviewed but should be confirmed on the Mac.

### E. References

- Y. A. Houndonougbo, B. B. Laird, B. J. Leimkuhler, A molecular dynamics algorithm for mixed hard-core/continuous potentials, Mol. Phys. 98, 309–316 (2000).

- B. J. Alder, T. E. Wainwright, Phase transition for a hard sphere system, J. Chem. Phys. 27, 1208 (1957).

- N. F. Carnahan, K. E. Starling, Equation of state for nonattracting rigid spheres, J. Chem. Phys. 51, 635 (1969).

- C. J. Fennell, J. D. Gezelter, Is the Ewald summation still necessary? Pairwise alternatives to the accepted standard for long-range electrostatics, J. Chem. Phys. 124, 234104 (2006).

- D. C. Rapaport, The Art of Molecular Dynamics Simulation, 2nd ed., Cambridge University Press (2004).

- A. Stukowski, Visualization and analysis of atomistic simulation data with OVITO, Modelling Simul. Mater. Sci. Eng. 18, 015012 (2010).

- J. E. Stone, An efficient library for parallel ray tracing and animation (Tachyon), M.Sc. thesis, University of Missouri-Rolla (1998).

- W. Humphrey, A. Dalke, K. Schulten, VMD: Visual Molecular Dynamics, J. Mol. Graphics 14, 33–38 (1996).

- M. Brehm, B. Kirchner, TRAVIS: a free analyzer and visualizer for Monte Carlo and molecular dynamics trajectories, J. Chem. Inf. Model. 51, 2007–2023 (2011); M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, TRAVIS: a free analyzer for trajectories from molecular simulation, J. Chem. Phys. 152, 164105 (2020).

Generated by report/build_report.py on 2026-10-25 from the committed run metadata, analyses and figure manifests (commit 975bd1a).
