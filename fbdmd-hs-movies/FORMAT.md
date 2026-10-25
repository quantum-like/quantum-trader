# fbdmd-hs-movies — shared interface spec (sim ⇄ render)

Project root: `/home/user/quantum-trader/fbdmd-hs-movies` (self-contained; no trading code).
Python env: `.venv/` (python 3.13, numpy, pillow, ovito 3.17). Activate with `. .venv/bin/activate`.
System tools present: g++ 13 / clang, make, ffmpeg 6.1 (libx264).

## Reduced units
σ = 1 (hard-sphere diameter), m = 1, kT = 1  ⇒  time unit τ = σ sqrt(m/kT).
Charges are ±e on a subset of spheres; Coulomb energy U = l_B · kT · Σ_{i<j} q_i q_j / r_ij  (q = ±1),
so `bjerrum` = l_B/σ is the single coupling parameter (T* = σkT/(e²/4πε) = 1/l_B*).

## Canonical trajectory file: extended XYZ  `traj/<name>.xyz`
One block per frame, OVITO-readable:

```
1000
Lattice="L 0 0 0 L 0 0 0 L" Properties=species:S:1:pos:R:3:id:I:1:type:I:1:charge:R:1 Time=0.025 Frame=1 pbc="T T T"
HS 0.123456 4.567890 7.891234 1 1 0
POS 3.1 2.2 1.1 2 2 1
NEG 5.5 6.6 7.7 3 3 -1
...
```
* species ∈ {HS, POS, NEG}; type ∈ {1 (neutral), 2 (+), 3 (−)}; charge ∈ {0, +1, −1}.
* Positions wrapped into [0, L). Centres only; every sphere has radius 0.5.
* Frames are at uniform times t_k = k·dt_frame (free flight already interpolated: the simulator
  evaluates ballistic positions at exactly t_k, no event jitter).
* Particle order (id) is identical in every frame.

## Cache: `traj/<name>.npz`  (written by sim/run_sim.py, read by renderers for speed)
* `pos`   float32 (F, N, 3)   wrapped positions
* `types` int8   (N,)          1/2/3 as above
* `charge` int8  (N,)          0/+1/−1
* `times` float64 (F,)
* `L`     float64 scalar
* `meta`  JSON string (same content as `traj/<name>.json`)

## Metadata: `traj/<name>.json`
```json
{"name": "hs_charges_20p20m", "N": 1000, "eta": 0.35, "L": 11.444, "kT": 1.0, "nplus": 20, "nminus": 20,
 "bjerrum": 5.0, "coulomb": "ewald", "integrator": "EDMD + impulsive (collision) Verlet", "dt": 0.005,
 "dt_frame": 0.025, "t_equil_neutral": 5.0, "t_equil_charged": 5.0, "t_run": 18.0, "n_frames": 721,
 "seed": 1, "n_collisions": 123456, "wall_seconds": 12.3, "Z_virial": 5.2, "Z_CS": 5.206,
 "energy_drift_rel": 1e-5, "machine": "..."}
```
(field names are the contract; add more freely.)

## Diagnostics: `traj/<name>_diag.csv`
header: `time,KE,PE,Etot,T_inst,n_collisions,min_dist,n_pairs_pm_1p2` (one row per frame).

## Movies
* `movies/hs_neutral.mp4`, `movies/hs_charges_20p20m.mp4` (required), optional `movies/hs_charges_20p20m_xray.mp4`.
* H.264 (libx264), yuv420p, 30 fps, 1280×720, `-movflags +faststart`.
* Colours: neutral light gray (0.78,0.78,0.80), + red (0.85,0.15,0.15), − blue (0.15,0.35,0.90).
  Light studio background (0.97,0.97,0.98) default; faint dark-gray box edges.
* Frames go to `frames/<name>/frame_%05d.png` (gitignored).
