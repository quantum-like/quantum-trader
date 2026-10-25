# fbdmd-hs-movies — Materials-Studio-style hard-sphere + point-charge movies

Short CPK-style mp4 movies of the classic "scout" system used to sanity-check FB-DMD / RPM work:
**N = 1000 hard spheres** in a periodic cubic box at liquid-like packing (η = 0.35), once **neutral**
and once with **20 (+e) and 20 (−e) monovalent charges** on a subset of the spheres (restricted
primitive model style, same diameter σ). The point is to *see* the dynamics — free flight between
collisions, Bjerrum pairing of opposite charges, transient chains — not to produce paper-grade data.

| movie | what | integrator |
|---|---|---|
| `movies/hs_neutral.mp4` | 1000 HS, no charges | event-driven hard-sphere MD (exact) |
| `movies/hs_charges_20p20m.mp4` | same box + 20+/20− (l_B = 5 σ) | EDMD hard core + impulsive (collision) Verlet for Coulomb |
| `movies/hs_charges_20p20m_xray.mp4` | same charged run, neutrals ghosted so the ions are visible inside | — |

Everything is CPU-only (no CUDA, no Metal needed). Rendering uses OVITO's Tachyon ray tracer
(`pip install ovito`); if OVITO is unavailable the scripts fall back to a pure numpy/Pillow renderer
that produces the same movie.

## Quick start

```bash
cd fbdmd-hs-movies
bash run_all.sh                       # ~? min on a 4-core box: venv + build + 2 sims + 3 renders
# faster preview:
QUALITY=draft WIDTH=640 HEIGHT=360 T_RUN=6 bash run_all.sh
```
Outputs: `movies/*.mp4`, trajectories in `traj/` (ext-XYZ readable by OVITO + `.npz` cache),
per-frame diagnostics `traj/*_diag.csv`, analysis in `notes/`, and a regenerated one-page `NOTES.md`.

Requirements: python ≥ 3.10, a C++17 compiler (`g++`/`clang++`; Xcode CLT on macOS), `make`, `ffmpeg`
(`brew install ffmpeg`). Python packages: see `requirements.txt` (numpy, pillow, optional ovito).

## Step by step (what run_all.sh does)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
make -C sim                                                     # builds sim/hsdmd
# neutral control: 5 τ equilibration (velocity rescaling), 18 τ NVE production, frame every 0.025 τ
python sim/run_sim.py --name hs_neutral --N 1000 --eta 0.35 --seed 1 \
       --t-equil-neutral 5 --t-run 18 --dt-frame 0.025
# charged: identical neutral start (same seed), then 20+/20- switched on, 5 τ more equilibration, 18 τ NVE
python sim/run_sim.py --name hs_charges_20p20m --N 1000 --eta 0.35 --seed 1 \
       --t-equil-neutral 5 --nplus 20 --nminus 20 --bjerrum 5 --coulomb ewald --dt 0.005 \
       --t-equil-charged 5 --t-run 18 --dt-frame 0.025
python sim/analyze.py traj/hs_neutral.npz                     # overlap scan, Z vs Carnahan-Starling
python sim/analyze.py traj/hs_charges_20p20m.npz              # + energy drift, pairing / cluster stats
# render (OVITO/Tachyon; use render/render_numpy.py with the same flags if `import ovito` fails)
python render/render_ovito.py --traj traj/hs_neutral.npz        --name hs_neutral        --out movies/hs_neutral.mp4        --style cpk
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m --out movies/hs_charges_20p20m.mp4 --style cpk
python render/render_ovito.py --traj traj/hs_charges_20p20m.npz --name hs_charges_20p20m_xray --out movies/hs_charges_20p20m_xray.mp4 --style xray
python make_notes.py
```

## Physics and units

* Reduced units: σ = 1, m = 1, kT = 1, time unit τ = σ√(m/kT). Box L = (Nπ/6η)^{1/3} = 11.44 σ for N = 1000, η = 0.35.
* Hard spheres: exact event-driven MD (cell lists, binary event heap, lazy invalidation). Frames are
  evaluated at uniform times t_k = k·Δt_frame from the ballistic free flight, so the movie is smooth
  (no per-collision jitter) and nothing can tunnel — `analyze.py` scans every frame for r < σ.
* Charges: ±e on 20 + 20 randomly chosen spheres. Coulomb energy U = l_B kT Σ q_i q_j / r_ij with
  l_B/σ = 5 (i.e. T* = 1/5 = 0.2; contact energy of a +/− pair is −5 kT, so Bjerrum pairs form but are
  broken up by collisions with the gray solvent spheres). Long range handled by 3D Ewald summation for the
  40 ions (α = 5.6/L, k-space |n|∞ ≤ 6); `--coulomb dsf` gives the Fennell–Gezelter damped-shifted-force
  alternative. `--bjerrum` changes the coupling (2 → mostly free ions, 8–10 → chains/clusters).
* Integrator for the charged run: impulsive / collision Verlet (Houndonougbo, Laird & Leimkuhler,
  Mol. Phys. 98, 309 (2000)): kick (Δt/2, ions only) → exact EDMD drift over Δt including all hard
  collisions → kick (Δt/2). Δt = 0.005 τ. Production is NVE; the energy drift is reported in NOTES.md.
* Equilibration only: velocity rescaling every 0.1 τ to kT = 1.

## Visualization

CPK look: opaque spheres of diameter σ; neutrals light gray, + red, − blue; perspective 3/4 view with a
slow 40° orbit; soft key/fill lighting with ambient occlusion and slight specular; faint box edges;
light studio background (`--bg dark` for the old Materials Studio navy look); small label with the
system parameters and the current time in τ. 1280×720, 30 fps, 721 frames (24 s) per movie, H.264 yuv420p.

## Layout

```
run_all.sh          one-shot pipeline (env knobs: QUALITY, N, ETA, BJERRUM, SEED, T_RUN, DT_FRAME, WIDTH, HEIGHT, RENDERER, XRAY, FORCE)
sim/hsdmd.cpp       C++17 EDMD + impulsive Verlet + Ewald/DSF (single file, no deps)   sim/Makefile
sim/run_sim.py      CLI wrapper: build, run, convert to traj/<name>.{xyz,npz,json,_diag.csv}
sim/analyze.py      overlap scan, energy/temperature, pairing + cluster statistics
render/render_ovito.py   primary renderer (OVITO Tachyon) -> frames/<name>/*.png -> movies/<name>.mp4
render/render_numpy.py   dependency-free fallback renderer, same CLI
render/make_test_traj.py synthetic trajectory for renderer tests
make_notes.py       assembles NOTES.md from traj/*.json, ffprobe and machine info
FORMAT.md           trajectory/file-format contract shared by sim and render
movies/ traj/ frames/ notes/   outputs (frames and .xyz are gitignored)
```

No trading code, no secrets, no GPU stack. See `NOTES.md` for the run parameters, wall times and what the
charged trajectory shows.
