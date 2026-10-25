# fbdmd-hs-movies — Materials-Studio-style hard-sphere + point-charge movies

Short CPK-style mp4 movies of the classic "scout" system used to sanity-check FB-DMD / RPM work:
**N = 1000 hard spheres** in a periodic cubic box at liquid-like packing (η = 0.35), once **neutral**
and once with **20 (+e) and 20 (−e) monovalent charges** on a subset of the spheres (restricted
primitive model style, same diameter σ). The point is to *see* the dynamics — free flight between
collisions, Bjerrum pairing of opposite charges, transient chains — not to produce paper-grade data.

| movie | what | integrator |
|---|---|---|
| `movies/hs_neutral.mp4` | 1000 HS, no charges | event-driven hard-sphere MD (exact) |
| `movies/hs_charges_20p20m.mp4` | same box + 20+/20− (l_B = 5 σ) | EDMD hard core + collision Verlet (Houndonougbo, Laird & Leimkuhler 2000) for the Coulomb part |
| `movies/hs_charges_20p20m_xray.mp4` | same charged run, neutrals ghosted so the ions are visible inside | — |

Everything is CPU-only (no CUDA, no Metal needed). Rendering uses OVITO's Tachyon ray tracer
(`pip install ovito`); if OVITO is unavailable the scripts fall back to a pure numpy/Pillow renderer
that produces the same movie.

## Quick start

```bash
cd fbdmd-hs-movies
bash run_all.sh                       # venv + build + selftest (~7 s) + 2 sims (~15 s) + analysis (~1 min) + 3 renders (~50 min on 4 cores, ~17 min each)
# faster preview:
QUALITY=draft WIDTH=640 HEIGHT=360 T_RUN=6 bash run_all.sh
# literal HLL collision Verlet instead of the default cv-phys variant (see Physics):
INTEGRATOR=cv bash run_all.sh
# also run the integrator energy-order test (writes notes/order_test.{txt,csv}):
ORDER_TEST=1 bash run_all.sh
```
Outputs: `movies/*.mp4`, trajectories in `traj/` (ext-XYZ readable by OVITO + `.npz` cache + `.json`
metadata), per-frame diagnostics `traj/*_diag.csv`, analysis in `notes/`, and a regenerated one-page `NOTES.md`.

Requirements: python ≥ 3.10, a C++17 compiler (`g++`/`clang++`; Xcode CLT on macOS), `make`, `ffmpeg`
(`brew install ffmpeg`), bash ≥ 3.2 (`run_all.sh` uses no bash-4-only features, so the stock macOS `/bin/bash` 3.2.57
works). Python packages: see `requirements.txt` (numpy, pillow, optional ovito).
On Linux without a GPU, OVITO additionally needs a software Vulkan driver (`apt install mesa-vulkan-drivers`);
on macOS the pip wheel is self-contained. Each OVITO render prints one harmless
`VkInstanceCreateInfo ... apiVersion` line on stderr.

## Step by step (the commands run_all.sh runs, for doing it by hand)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # run_all.sh falls back to numpy+pillow if no ovito wheel exists
make -C sim                                                     # builds sim/hsdmd (single C++17 file)
sim/hsdmd --selftest                                            # 50 unit checks (collisions, Ewald, splitting, quartic solver, ...), ~7 s; SELFTEST=0 skips it in run_all.sh
# neutral control: 5 τ equilibration (velocity rescaling), 18 τ NVE production, frame every 0.025 τ
python sim/run_sim.py --name hs_neutral --N 1000 --eta 0.35 --seed 1 \
       --t-equil-neutral 5 --t-run 18 --dt-frame 0.025
# charged: identical neutral start (same seed), then 20+/20- switched on, 15 τ more equilibration, 18 τ NVE
python sim/run_sim.py --name hs_charges_20p20m --N 1000 --eta 0.35 --seed 1 \
       --t-equil-neutral 5 --nplus 20 --nminus 20 --bjerrum 5 --coulomb ewald \
       --integrator cv-phys --dt 0.005 --t-equil-charged 15 --t-run 18 --dt-frame 0.025
python sim/analyze.py traj/hs_neutral                         # overlap scan (all frames), Z vs Carnahan-Starling
python sim/analyze.py traj/hs_charges_20p20m                  # + energy drift, stationarity, pairing / cluster stats
python sim/order_test.py --t-equil-charged 15                 # optional: max|E(t)-E0|/N vs dt for the 4 integrators
# render (OVITO/Tachyon; use render/render_numpy.py with the same flags if `import ovito` fails;
#  run_all.sh additionally passes --label "..." with the system parameters, otherwise a default caption is used)
python render/render_ovito.py --traj traj/hs_neutral.npz        --name hs_neutral        --out movies/hs_neutral.mp4        --style cpk
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m --out movies/hs_charges_20p20m.mp4 --style cpk
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m_xray --out movies/hs_charges_20p20m_xray.mp4 --style xray
python make_notes.py
```

## Physics and units

* Reduced units: σ = 1, m = 1, kT = 1, time unit τ = σ√(m/kT). Box L = (Nπ/6η)^{1/3} = 11.437 σ for N = 1000, η = 0.35.
* Hard spheres: exact event-driven MD (cell lists, binary event heap, lazy invalidation). Frames are
  evaluated at uniform times t_k = k·Δt_frame from the ballistic free flight, so the movie is smooth
  (no per-collision jitter) and nothing can tunnel — `analyze.py` scans every frame for r < σ.
  Validation: collisional virial Z = 5.233 (neutral) / 5.224 (charged) vs Carnahan–Starling 5.206, collision rate
  14.32 / 14.31 per particle per τ vs Enskog 14.24, exact time reversibility (reviewer tests in `sim/README_sim.md`).
* Charges: ±e on 20 + 20 randomly chosen spheres. Coulomb energy U = l_B kT Σ q_i q_j / r_ij with
  l_B/σ = 5 (i.e. T* = 1/5 = 0.2; the contact energy of a +/− pair is −5 kT, so Bjerrum pairs form but
  are broken up by collisions with the gray solvent spheres). Long range handled by 3D Ewald summation
  for the 40 ions (α = 5.6/L, k-space |n|∞ ≤ 6); `--coulomb dsf` gives the Fennell–Gezelter damped-
  shifted-force alternative (note: with r_c = L/2 it weakens the contact energy by ~30 %).
  `--bjerrum` changes the coupling (2 → mostly free ions, 8–10 → chains/clusters).
* Integrator for the charged run — collision Verlet (CV) of Houndonougbo, Laird & Leimkuhler,
  Mol. Phys. 98, 309 (2000), see `notes/collision_verlet_spec.md`:
  the pair potential is split (their eqs. 19–25) into a short-range part φ₁ (non-zero only for r < q₂ = 1.5 σ)
  and a long-range part φ₂ that is flattened by a Hermite cubic so that its force vanishes at contact
  (q₁ = σ). Each step Δt = 0.005 τ: half-kick with F₂ → exact hard-sphere propagation in which every
  accelerated ion takes one velocity-Verlet sub-step with F₁ between consecutive collisions (quadratic
  trajectories, collision times from the quartic of their eq. 29) → half-kick with F₂ (their Table 1).
  Equilibration uses velocity rescaling every 0.1 τ; production is NVE and the energy drift is in NOTES.md.
  * **Finding.** On this system the literal HLL scheme (`INTEGRATOR=cv`) is only first order in the
    energy error (log–log slope 0.7–0.8 in max|E(t)−E₀| vs Δt): ions collide ~14 times per τ with
    force-free neutral spheres while F₂ from the other ions is non-zero, which is exactly the O(h)
    collision error of their eq. 18 (their case 3, "force vanishes at contact", only holds for ion–ion contacts).
    The shipped default `cv-phys` keeps the HLL splitting and sub-steps but reflects the *physical*
    velocity v + F₂(τ₁ − Δt/2) at each collision (then returns to the drift frame), which removes that
    term: slope 2.0 (equilibrated start, Ewald) / 1.9 (DSF), production drift 2×10⁻⁵ over 18 τ versus
    1.3×10⁻⁴ for literal `cv`. Also available: `kdk-phys` (plain kick–drift–kick with the same physical-velocity
    rule, 2× cheaper) and `naive` (their eq. 9, first order). The thermodynamic state (pairing fraction,
    potential energy) agrees between all variants within statistical scatter; all are exact on the hard cores.

## Visualization

CPK look: opaque spheres of diameter σ; neutrals light gray, + red, − blue; perspective 3/4 view with a
slow 40° orbit; Tachyon ray tracing with ambient occlusion (16 samples) and slight specular; faint box
edges; light studio background (`--bg dark` for the old Materials Studio navy look); small label with the
system parameters and the current time in τ. 1280×720, 30 fps, 721 frames (24 s) per movie, H.264 yuv420p.
On a 4-core box a frame takes ≈1.4 s (cpk) / ≈1.25 s (x-ray), i.e. ≈17 / ≈15 min per 721-frame movie; `OVITO_THREAD_COUNT` is set to 4× the core count
automatically because Tachyon otherwise leaves half the CPU idle.

## Layout

```
run_all.sh          one-shot pipeline (env knobs: QUALITY, N, ETA, BJERRUM, SEED, INTEGRATOR, T_EQ_N, T_EQ_C, T_RUN, DT, DT_FRAME,
                    WIDTH, HEIGHT, FPS, ORBIT, RENDERER, XRAY, ORDER_TEST, SELFTEST, FORCE, PYTHON)
sim/hsdmd.cpp       C++17 EDMD + collision Verlet + Ewald/DSF (single file, no deps)   sim/Makefile
sim/run_sim.py      CLI wrapper: build, run, convert to traj/<name>.{xyz,npz,json,_diag.csv}
sim/analyze.py      overlap scan, energy/temperature/stationarity, pairing + cluster statistics
sim/order_test.py   energy-error order test of the four integrators (HLL Fig. 4 analogue)
sim/README_sim.md   algorithm details and the full validation tables
render/render_ovito.py   primary renderer (OVITO Tachyon) -> frames/<name>/*.png -> movies/<name>.mp4
render/render_numpy.py   dependency-free fallback renderer, same CLI
render/make_test_traj.py synthetic trajectory for renderer tests;  render/README_render.md  timings and settings
make_notes.py       assembles NOTES.md from traj/*.json, ffprobe and machine info
FORMAT.md           trajectory/file-format contract shared by sim and render
notes/collision_verlet_spec.md   the integrator spec derived from the paper (+ §4a: why literal CV is first order here)
movies/ traj/ frames/ notes/   outputs (frames, .xyz and .npz are gitignored; movies, json and csv are kept)
```

No trading code, no secrets, no GPU stack. See `NOTES.md` for the run parameters, wall times and what the
charged trajectory shows.
