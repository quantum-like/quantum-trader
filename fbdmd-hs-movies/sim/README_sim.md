# sim/ — hard-sphere EDMD + RPM charges trajectory generator

Single-file C++17 simulator (`hsdmd.cpp`, standard library only, no OpenMP/GPU) plus two Python
wrappers. Output follows `../FORMAT.md` exactly (ext-XYZ, npz, json, diag csv).

## Units
σ = 1 (sphere diameter), m = 1, kT = 1, τ = σ√(m/kT). Box L = (Nπ/6η)^(1/3) (η = 0.35, N = 1000 → L = 11.4369).
Coulomb: U = l_B·kT·Σ_{i<j} q_i q_j φ(r_ij), q = ±1, `--bjerrum` = l_B/σ (default 5 ⇒ T* = 0.2).

## Algorithm
* **Hard cores: event-driven MD (EDMD).** Cell list (cell width ≥ σ, 27-cell neighbourhood), lazy
  positions x_i(t) = x_i + v_i (t − t_i), binary min-heap of events (pair collisions + cell crossings)
  with lazy invalidation via per-particle collision counters. Collision time from
  b = r·v < 0, disc = b² − v²(r² − σ²); roundoff overlaps (r² ≤ σ²(1+1e-12), approaching) collide
  immediately; grazing pairs that already separate get no impulse. Elastic equal-mass reflection,
  virial Σσ|Δp| accumulated ⇒ Z = 1 + Σσ|Δp| /(3NkT t) compared with Carnahan–Starling.
  Frames are written at exactly t_k = k·dt_frame by evaluating the ballistic positions (free-flight
  interpolation; no event jitter). Periodic wrap happens at box-face cell crossings.
* **Charges: collision (impulsive) Verlet** (Houndonougbo, Laird, Leimkuhler, Mol. Phys. 98, 309, 2000):
  kick(h/2) → exact EDMD drift of length h including all hard collisions → kick(h/2); forces act on the
  40 ions only, computed once per step (the second half kick of step n and the first of step n+1
  share the force). Kicked ions get their collision counter bumped and their crossing/collision
  events re-predicted; neutral spheres' events stay valid. dt is rounded so that dt_frame/dt is an
  integer (0.025/0.005 = 5), so every frame is a step boundary with synchronized x and v.
  * **Collision rule (documented deviation).** In a plain kick–drift–kick the hard-core reflection
    acts on the leapfrog *drift* velocity u = v_n + (h/2)a instead of the physical velocity at the
    collision time v = v_n + τ₁a. That is an O(dt) velocity error (h/2 − τ₁)(Ra − a) at every
    collision of an ion, measured here as a random-walk energy error ≈ 1e-3 relative over 4 τ for
    N = 216 that only shrank like dt¹·³. `--collision-rule phys` (default) therefore reflects the
    physical velocity: c = a(τ₁ − h/2) + ȧτ₁²/2 (ȧ from a backward difference of the previous
    step's force), v = u + c, elastic reflection, u⁺ = v⁺ − c. Pair momentum is conserved exactly, the
    drift position at the end of the step is unchanged to O(dt²), and a 1e-9 separating bias on the
    drift velocity guarantees the EDMD never overlaps. Energy error drops ~40× (see numbers below) and
    scales ≈ dt². `--collision-rule naive` gives the textbook drift reflection for comparison.
* **Coulomb** (`--coulomb`): `ewald` (default): α = 5.6/L, real space erfc(αr)/r over minimum images
  with r_c = L/2, reciprocal sum over 0 < |n|∞ ≤ 6 (1098 half-space k-vectors, per-ion per-axis
  phase cache), self term −α/√π Σq², tin-foil boundary. `dsf`: Fennell–Gezelter damped shifted force,
  α = 0.2, r_c = L/2 (energy and force vanish continuously at r_c; constant self term included).
  Both variants pass a central-finite-difference force check (`--selftest`).
* **Thermostat** only in the two equilibration stages: every 0.1 τ all velocities are rescaled to
  KE = 3/2 (N−1) kT (events rebuilt). Production is NVE. The number of thermostat blocks is
  ⌊t_equil/0.1⌋ (the remainder is simulated without a rescale), so any `t_equil_neutral` is allowed;
  the clock never runs backwards (`advance_to` aborts with exit 3 if it ever would).
* **Run plan (one process):** A) neutral equilibration `t_equil_neutral` (5 τ); B) if ions requested,
  assign ±e to randomly chosen spheres (seeded), charged equilibration `t_equil_charged` (**15 τ**);
  C) production `t_run` (18 τ), frames + diag rows. Same `--seed` ⇒ identical stage A for the neutral
  and charged runs (the RNG is used only for the initial lattice/velocities and the charge selection).
* **Charged equilibration length.** Switching the charges on in a neutral fluid starts a pairing
  relaxation with a time constant of ≈ 4–6 τ (PE −35 → −61 → −68 → −72 kT in successive 2 τ blocks,
  then plateau). `t_equil_charged` **must be ≥ 10 τ**; the default is 15 τ (≈ 3 relaxation times,
  1.6 s wall at N = 1000). With 5 τ the movie starts ≈ 10 kT above the plateau and T_mean comes out
  1.007 instead of 1.000; with 2 τ production is grossly non-stationary (see Validation).
  `analyze.py` flags both cases.
* **Initial state:** simple-cubic lattice n³ ≥ N with jitter 0.45·min(a−σ, 0.5) (abort if a ≤ σ or any
  overlap), Maxwell–Boltzmann velocities, zero total momentum, exact KE = 3/2 (N−1) kT.
* **Output order:** neutrals first, then + ions, then − ions (ids 1..N, identical in every frame).
  With this order OVITO's first-appearance numeric type ids coincide with FORMAT.md (HS=1, POS=2,
  NEG=3); renderers should nevertheless colour by species name or by the `charge`/`type` columns.
  Positions are written with 10 decimals so the hard-core check can be done at 1e-9 from the xyz.

## Build / CLI
```
make -C sim                      # c++ -O3 -std=c++17 -Wall -Wextra  (clang++ on macOS, g++ on Linux)
make -C sim CXX=clang++          # force a compiler; both compile warning-free
sim/hsdmd --selftest             # 28 unit checks, exit 0 = pass
sim/hsdmd --help

. .venv/bin/activate
python sim/run_sim.py --name hs_neutral                                  # N=1000, eta=0.35, 5+18 tau
python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20   # + 15 tau charged equil
python sim/analyze.py traj/hs_charges_20p20m                             # exit 1 overlap, 2 contract violation
```
`run_sim.py` flags (defaults): `--name` (required) `--N 1000 --eta 0.35 --nplus 0 --nminus 0
--bjerrum 5.0 --coulomb ewald|dsf --dt 0.005 --t-equil-neutral 5 --t-equil-charged 15 --t-run 18
--dt-frame 0.025 --seed 1 --outdir traj [--rebuild] [--cxx CXX] [--quiet]`. It (re)builds the binary
when `hsdmd.cpp`/`Makefile` are newer, runs it, converts the ext-XYZ to `traj/<name>.npz`, merges
the binary's summary with `platform` machine info into `traj/<name>.json` (`machine` is a string,
the full dict is `machine_info`), and prints a one-paragraph summary. The binary itself writes
`traj/<name>.xyz` and `traj/<name>_diag.csv`
(`time,KE,PE,Etot,T_inst,n_collisions,min_dist,n_pairs_pm_1p2`; min_dist via the cell list at frame
time, n_pairs = +/− pairs with r < 1.2σ). The binary also accepts `--collision-rule phys|naive`.

**Argument validation** (binary and wrapper, exit 2 with a message): N ≥ 1, nplus/nminus ≥ 0,
nplus+nminus ≤ N, 0 < η < 0.74, dt > 0, dt_frame > 0, all times ≥ 0, bjerrum ≥ 0 (plus the existing
coulomb/collision-rule/box-size checks). `--t-run` smaller than `--dt-frame` (e.g. `--t-run 0`) is
legal for debugging: one frame is written and the undefined quantities (`Z_virial`, `Z_rel_dev`,
`collision_rate_per_particle_per_tau`) are `null` in the json — the binary never writes `nan`/`inf`
(non-finite doubles become `null`). If the summary is nevertheless unparsable, `run_sim.py` prints the
offending line, keeps the file as `traj/<name>_summary_INVALID.json` and exits 1.

`analyze.py traj/<name>[.xyz|.npz]`:
* minimum pair distance over all frames (blocked O(N²) numpy, ~3 s per 81 full frames ⇒ ~28 s for
  721); exit 1 on any overlap. Reads the xyz (float64) when present; with the float32 npz the
  tolerance is relaxed to 4·L·eps32 and reported.
* FORMAT.md contract check (exit 2 on violation): type ids ∈ {1,2,3} and non-decreasing (neutrals,
  then +, then −), charge column consistent with the type column, json has the contract fields and a
  string `machine`, json N / n_frames match the trajectory, npz dtypes, uniform frame times.
* temperature and energy drift from the diag csv.
* **stationarity check** (warnings only, exit code unaffected): first-half vs second-half means of
  T, PE and the +/− pair count, linear trends dPE/dt and dT/dt, the first 2 τ of production vs the
  rest in units of the 2 τ-block std, and T_mean − 1 with its expected NVE scatter σ_PE/(1.5 N).
  `WARNING: production is not stationary` is printed when the linear PE change over the run exceeds
  10 % of |PE| **and** 2 per-frame std devs, or the T trend exceeds 1 %, or the first 2 τ differ from
  the rest by > 3σ (block). The thresholds are tuned for the production system (N = 1000, 40 ions):
  a 2 τ charged equilibration fails, ≥ 10 τ passes (3 seeds). For small test systems (N ≈ 200, 16 ions,
  t_run ≤ 6 τ) the relative PE fluctuations are large enough that the 10 %/2σ rule can trigger on
  noise (one of four such runs below) — treat warnings there as advisory.
* for charged runs per-frame pairing (fraction of ions whose nearest opposite ion is within 1.2σ),
  cluster-size histogram with 1.2σ +/− bonds (connected components), chain vs branched clusters and a
  verdict line (suffixed with the stationarity warning when present).

## Validation (this box: Intel Xeon 2.1 GHz, Linux x86_64, 4 cpus, g++ 13.3 / clang++ 18.1; hsdmd 1.1)
* `--selftest` (28 checks, both compilers): Ewald and DSF |F_num − F|/max|F| = 8.0e-10 / 3.4e-10,
  Σ F = 0 to 1e-15; Ewald reproduces the NaCl Madelung constant (1.7476) on a 64-ion supercell;
  head-on collisions (also across a periodic face) exact; 200 random two-body collisions: contact error
  1.3e-15, ΔKE 1.3e-15, |ΔP| 4e-16, no overlap; N=125 EDMD 1 τ brute-force overlap 0, KE 2e-15;
  bound +/− pair 20 τ: max|ΔE|/|E0| phys 1.7e-3 (dt=0.005) → 4.7e-4 (dt=0.0025) vs naive 9.3e-2.
* (c) `--N 216 --t-equil-neutral 2 --t-run 4` (neutral): 6091 collisions, 14.10 per particle per τ
  (Enskog 14.24), Z = 5.2041 vs Z_CS = 5.2057 (−0.03 %), KE constant to 1e-14, |P| = 1.4e-14,
  min pair distance 1.0000087 (no overlap), 0.10 s.
* (d) `--N 216 --nplus 8 --nminus 8 --t-equil-neutral 2 --t-run 6` (default 15 τ charged equil),
  ewald: energy drift (E_end−E0)/|E0| = +6.0e-5, max 6.3e-5 over 6 τ; T = 0.988 ± 0.007 (16 ions:
  the NVE offset is set by PE at the end of equilibration, expected scatter ±0.007); min distance
  1.0000008; 57 % of ions in +/− contact pairs, 22 % in clusters ≥ 3; stationarity OK; 0.90 s
  (equil-charged 0.53 s). Same with `--t-equil-charged 2`: PE −28.0 | −30.1 (halves), trend 20 % of
  |PE| = 2.5σ, T +1.8 % ⇒ `WARNING: not stationary`. dsf (`--t-run 4`): drift +2.2e-5, 55 % paired,
  warning on a 13 %/2.1σ PE trend (noise level for 16 ions over 4 τ, see above).
  dt scan (ewald, 2 τ equil, 4 τ): max|ΔE|/|E0| = 7.9e-5, 2.5e-5, 8.9e-6, 4.6e-6 for
  dt = 0.0083, 0.005, 0.0025, 0.00125 (naive rule: 1.05e-3, 1.06e-3, 2.1e-4, 3.8e-4).
* (e) **full-size defaults, N = 1000, η = 0.35.** Neutral (5 + 18 τ): 128 890 production collisions,
  14.32 per particle per τ (Enskog 14.24), Z = 5.2330 (+0.52 %), KE exact, min distance 1.00000009,
  sim 1.50 s (+0.7 s xyz→npz, analyze 27 s). Charged 20+/20−, l_B = 5, ewald (5 + 15 + 18 τ, seed 1):
  128 660 collisions, 14.30 per particle per τ, Z = 5.2184 (+0.24 %, collisional part), energy drift
  +5.2e-5 (max 5.2e-5, rms 1.5e-5, slope 2.8e-6/τ), T_mean = 1.0031 ± 0.0025, min distance
  1.0000001, 38 % of ions in +/− contact pairs (8.4 pairs per frame, 7.6 | 9.2 in the two halves),
  10 % in triplets/short chains, largest cluster 3.2 (max 6), stationarity OK (PE first 2 τ vs rest
  +0.3σ); sim 5.1 s (equil-neutral 0.13 s, equil-charged 1.6 s, production 3.4 s) + 0.7 s convert,
  analyze 28 s. Seeds 2 / 3: T_mean 0.9999 / 0.9975, PE mean −70.7 / −70.3 kT (seed 1: −70.7),
  drift 4.9e-5 / 4.7e-5, 38 % / 36 % paired.
* (f) **equilibration length** (N = 1000, 20+/20−): production-start PE −66.1 / −70.8 / −74.1 kT for
  15 τ (seeds 1–3) vs plateau ≈ −70.6 ± 3.4 (2 τ-block std) — an equilibrated start within the
  fluctuations, T_mean = 1.000 ± 0.003. 5 τ (old default): PE(0) = −61.8, first 2 τ −69.2 vs rest
  −72.4 (+2.0σ), T_mean = 1.0069, analyze passes but reports the initial relaxation. 2 τ: PE halves
  −53.1 | −68.9 (trend 51 % of |PE| = 3.2σ), T 1.006 → 1.017 ⇒ WARNING.
* (g) **robustness** (fixes in 1.1): any `--t-equil-neutral` (0.25 … 2.25 tested at N = 216/η = 0.45 and
  N = 1000) gives min_dist ≥ 1 in every frame (1.0 had overlapping first frames for values that were
  not multiples of 0.1); `--t-run 0` / `0.01` produce a valid json with `null` fields; negative or
  zero N / nplus / nminus / dt / dt_frame / times, η ∉ (0, 0.74), nplus+nminus > N all exit 2 from
  both the binary and `run_sim.py`.

## Limitations / notes
* The top-level `run_all.sh` passes its own `T_EQ_C` (default 5 there) to `--t-equil-charged`; set
  `T_EQ_C=15` (or change its default) to get the equilibrated start documented above.
* Z_virial for charged runs contains only the hard-core (collisional) virial, not the Coulomb virial.
* T_mean of a charged NVE production deviates from 1 by (PE_end-of-equil − ⟨PE⟩)/(1.5 N kT), i.e. by
  ±0.3 % at N = 1000 and ±1 % at N = 216 even when fully equilibrated; only a *trend* indicates a
  problem.
* Exit codes of the binary: 0 ok, 2 invalid arguments / box too small / cannot open output, 3 internal
  consistency error (clock would run backwards).
* Ewald real-space cutoff r_c = L/2 with α = 5.6/L truncates at erfc(2.8) ≈ 7.5e-5 (viz quality).
* `dt` is rounded so dt_frame/dt is an integer (reported as `dt` in the json; the request as `dt_requested`).
* ncell = floor(L/σ) must be ≥ 3 (N ≥ ~30 at η = 0.35); the program aborts otherwise.
