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
* **Charges: collision Verlet (CV)** of Houndonougbo, Laird & Leimkuhler, Mol. Phys. 98, 309 (2000)
  ("HLL"; derivation and the post-review amendment §4a in `../notes/collision_verlet_spec.md`), shipped as
  `--integrator cv-phys` (default; `cv` = HLL exactly, first order here, see below). The Coulomb
  pair potential φ_c = l_B q_i q_j / r is split (HLL eqs. 19–25) into a short-range φ₁ and a long-range
  φ₂ with a Hermite cubic P(r) on [q₁, q₂]: φ₂ = P(q₁) for r < q₁ (flat: zero force at contact), P(r) on
  [q₁, q₂), φ_c beyond; φ₁ = φ_c − φ₂ vanishes with its derivative at q₂. Defaults q₁ = 1.0 σ, q₂ = 1.5 σ
  (`--cv-q1/--cv-q2`). One step of length h (HLL Table 1): ions get a half kick with F₂ = F_Ewald − F₁;
  then H₁ = T + V_hs + V₁ is propagated for h: between consecutive hard collisions **anywhere in the
  system** all ions take one velocity-Verlet sub-step with F₁ (quadratic paths x + vτ + ½aτ² for the
  "accelerated set" A of ions with F₁ ≠ 0, quartic collision times, per-axis quadratic cell crossings;
  neutrals and ions with F₁ = 0 keep the exact ballistic EDMD); at every collision the accelerated ions
  are advanced to the collision time, F₁ is recomputed (ion-only O(n_ion²) loop), the two half kicks
  are merged (v ← v + ½τ(a_old + a_new)), the colliding pair is reflected with the updated velocities,
  and every particle whose v or a changed is re-predicted (27 cells); the step ends with a final
  sub-step and the second F₂ half kick at the new positions. F_Ewald (or DSF) and F₁ are evaluated once
  per step boundary. The diag PE is the full Coulomb energy, so Etot is the physical conserved quantity.
  dt is rounded so that dt_frame/dt is an integer (0.025/0.005 = 5): every frame is a step boundary
  with synchronized x and v.
  * **Quartic collision times.** For pairs with Δa ≠ 0 the smallest τ in (0, t_step_end − t] with
    |Δq + Δvτ + ½Δaτ²| crossing σ from above is found by splitting the quartic into monotone pieces at
    its critical points (roots of the cubic f′, bracketed between the roots of f″) and safeguarded
    Newton on each decreasing piece with a sign change; pairs that are separating at contact get their
    re-entry root (attraction), the EDMD "b < 0" shortcut is not used. The window ends at the step end
    because all ions are re-kicked and re-predicted there; a triangle-inequality reach bound rejects
    ~99 % of the 27-cell candidates before any solve. Pairs with Δa = 0 (neutral–neutral, ballistic
    ions) keep the exact EDMD quadratic with no window (byte-identical neutral runs).
  * **Resting-contact guard.** A pair pressed together by attraction bounces with a period 2v_n/a_n that
    diverges as v_n → 0 and whose excursion σv_n²/a_n drops below the quartic's roundoff at v_n ≲ 1e-7.
    After any processed collision of a pair with Δa ≠ 0 the normal separation speed is therefore raised
    to at least `cv_vmin` = 2e-5 (momentum conserving, ≤ 1e-10 kT per event, counted as
    `n_cv_vmin_bias`); it never fired in the runs below (0 events) but bounds the worst case.
  * **`--integrator cv-phys` (default: HLL + physical-velocity rule for F₂).** HLL's order-2 argument
    needs the long-range force to vanish for *every* colliding pair at contact (Mol. Phys. 98, p. 312:
    "H₂ constant in a neighbourhood of (q₀, p₀)"); the splitting flattens only the colliding ion pair's
    own φ_c. In this fluid that premise fails twice: (i) each ion makes ≈ 14 collisions/τ with force-free
    *neutrals* while F₂ (partner beyond q₂, third ions, periodic images) still acts on it, and (ii) for a
    +/− contact pair the third-body fields on + and − are opposite and *add* on the relative coordinate
    (they cancel only in a one-component fluid such as HLL's LJ test). Every such collision carries the
    O(h) impulse-timing error of HLL eq. 18 with F₂ in place of φ_c′, δE = (h/2 − τ₁)(ΔF₂·n)Δv_n with
    random sign, i.e. a random-walk energy error ∝ h: plain `cv` is first order here (slope 0.78, 8–65×
    worse than kdk-phys; no q₂ cures it, and an all-ion system without any neutrals still gives slope 1.5
    — see (h)). `cv-phys` applies the physical-velocity rule of kdk-phys to F₂ only: at a collision τ₁
    into the step it reflects v + F₂(τ₁ − h/2) and returns to the drift frame, keeping F₁ and the hard
    cores exact. Because the EDMD moves the spheres with the *drift* velocities, the pair's drift normal
    velocity is then forced to be separating (≥ 1e-9 for ballistic pairs, ≥ `cv_vmin` for accelerated
    ones; counted as `n_drift_fix`, physically-separating no-ops as `n_phys_separating`). Without that
    guard a physically separating pair that still approaches in the drift frame overlapped (ballistic
    pair, up to 1.2e-6) or was re-predicted at τ = 0 forever (accelerated pair: the hsdmd 1.2 livelock,
    reproduced with all-ion N = 216 at dt = 0.02 and fixed in 1.3; both cases are selftests, and a
    watchdog now exits 3 after 10⁴ consecutive events at one time instead of hanging). The guard fires
    only in that corner case (0 events in every production-system run below, 132 in the all-ion
    dt = 0.02 run) and changes the energy by ≤ w_n²/4 ≲ 1e-6 kT per event (w_n ≤ |ΔF₂| h/2). Measured
    below: second order (slope 1.92 with DSF, 2.02 with Ewald from the equilibrated start), 2–10× better
    than kdk-phys at dt ≥ 0.005 and 10–40× better than `cv`, at the same cost as `cv`.
  * **`--integrator cv` (HLL 2000 exactly)** stays selectable for the comparison: same hard cores, F₁
    sub-steps and quartics as cv-phys, only the collision rule differs (reflects the drift velocities).
  * **`--integrator kdk-phys` / `naive`** (hsdmd ≤ 1.1, kept unchanged, bit-identical results): naive =
    HLL eq. 9 (kick(h/2, full force) → exact hard-sphere flow → kick(h/2)) with the hard-core reflection
    acting on the leapfrog drift velocity — O(dt) energy error at every ion collision (HLL section 2).
    kdk-phys reflects the physical velocity v = u + a(τ₁ − h/2) + ȧτ₁²/2 (ȧ from a backward difference)
    and returns to the drift frame; pair momentum exact, 1e-9 separating bias on the drift velocity
    (`n_drift_fix`), energy error ≈ dt². `--collision-rule phys|naive` is a deprecated alias.
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
sim/hsdmd --selftest             # 50 unit checks (~7 s), exit 0 = pass
sim/hsdmd --help
make -C sim order                # energy-error order test (sim/order_test.py), ~10 s on 4 cpus;
                                 #   extra flags via ORDER_ARGS="--coulomb dsf --name ..."

. .venv/bin/activate
python sim/run_sim.py --name hs_neutral                                  # N=1000, eta=0.35, 5+18 tau
python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20   # + 15 tau charged equil, cv-phys
python sim/run_sim.py --name hs_charges_20p20m --nplus 20 --nminus 20 --integrator cv   # HLL exactly
python sim/analyze.py traj/hs_charges_20p20m                             # exit 1 overlap, 2 contract violation
python sim/order_test.py [--N 216 --nplus 8 --nminus 8 --jobs 4]        # table + traj/<name>_order.csv
```
`run_sim.py` flags (defaults): `--name` (required) `--N 1000 --eta 0.35 --nplus 0 --nminus 0
--bjerrum 5.0 --coulomb ewald|dsf --integrator cv-phys|cv|kdk-phys|naive --cv-q1 1.0 --cv-q2 1.5
--dt 0.005 --t-equil-neutral 5 --t-equil-charged 15 --t-run 18 --dt-frame 0.025 --seed 1 --outdir traj
[--rebuild] [--cxx CXX] [--quiet]`. It (re)builds the binary when `hsdmd.cpp`/`Makefile` are newer,
runs it, converts the ext-XYZ to `traj/<name>.npz`, merges the binary's summary with `platform`
machine info into `traj/<name>.json` (`machine` is a string, the full dict is `machine_info`), and
prints a one-paragraph summary. The json carries `"integrator"` as a descriptive string
(`"collision Verlet (HLL 2000), q1=1, q2=1.5"`, `"... + F2 physical-velocity collision rule, ..."`,
`"EDMD + kick-drift-kick (...)"`, `"EDMD"` for neutral runs) plus the raw fields `integrator_id`
(`cv|cv-phys|kdk-phys|naive|edmd`), `cv_q1`, `cv_q2`, `cv_vmin`, `n_verlet_substeps`,
`n_quartic_solves`, `n_cv_vmin_bias`, `n_cv_grazing`, `n_drift_fix`, `n_phys_separating` (both also
used by kdk-phys), `max_contact_err`. The binary itself writes
`traj/<name>.xyz` and `traj/<name>_diag.csv`
(`time,KE,PE,Etot,T_inst,n_collisions,min_dist,n_pairs_pm_1p2`; min_dist via the cell list at frame
time, n_pairs = +/− pairs with r < 1.2σ). `--collision-rule phys|naive` is still accepted by the
binary as a deprecated alias of `--integrator kdk-phys|naive`.

`order_test.py` runs the binary for a short NVE production (2 τ) at dt ∈ {0.02, 0.01, 0.005, 0.0025}
for `cv-phys cv kdk-phys naive` on the production system, reports max_t |E(t) − E(0)|/N, the end
drift and log–log slopes, and writes `traj/<name>_order.csv`. By default `--t-equil-charged 0`, so
every run starts from the *same* (neutral-equilibrated, unpaired) configuration; the order test is
therefore not an equilibrium run — irrelevant for the convergence order, but its collision statistics
are not those of the movie runs. `--t-equil-charged 15` runs it from the equilibrated, paired state of
the movies (then every dt has its own start, see (h)). `--dt-frame 0.02` so that every dt divides the
frame spacing.

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

## Validation (this box: Intel Xeon 2.1 GHz, Linux x86_64, 4 cpus, g++ 13.3 / clang++ 18.1; hsdmd 1.3)
* `--selftest` (50 checks, both compilers, 7 s): Ewald and DSF |F_num − F|/max|F| = 8.0e-10 / 3.4e-10,
  Σ F = 0 to 1e-15; Ewald reproduces the NaCl Madelung constant (1.7476) on a 64-ion supercell;
  head-on collisions (also across a periodic face) exact; 200 random two-body collisions: contact error
  1.3e-15, ΔKE 1.3e-15, |ΔP| 4e-16, no overlap; N=125 EDMD 1 τ brute-force overlap 0, KE 2e-15;
  bound +/− pair 20 τ (L = 6): max|ΔE|/|E0| kdk-phys 1.7e-3 (dt=0.005) → 4.7e-4 (dt=0.0025) vs naive 9.3e-2.
  * **Potential splitting:** |φ₁ + φ₂ − φ_c| = 0 on [1, 3]; φ₁′ vs central differences 2.6e-10;
    φ₂′(q₁) = 1.3e-10, φ₂″(q₁) = 2.7e-5 (numerical, h = 1e-5), P′(q₁) = 0 and φ₂ constant below q₁
    exactly; φ₁(q₂⁻), φ₁′(q₂⁻) ≤ 1.2e-8; P(q₂) = φ_c(q₂), P′(q₂) = φ_c′(q₂) to 1e-14; `compute_F1`
    vs −∂V₁/∂x by finite differences for 16 random ions: 2.4e-10 relative.
  * **Quartic solver** vs brute-force sampling at 1e-6 resolution: 2500 random (Δq, Δv, Δa) paths
    (at-contact separating with attraction, just-outside separating, generic, strong attraction, weak
    acceleration; windows 0.02–0.3 τ): 1070 collisions found, identical to brute force, 433 of them
    re-entries of pairs starting at contact; 0 missed/late contacts, 0 overlaps > 1e-9 at the
    returned τ (worst 0), 0 spurious roots.
  * **Bound +/− pair, 20 τ, dt scan** (max|ΔE|/|E0|, |E0| ≈ 3):

    | box | dt | cv (HLL) | cv-phys | kdk-phys | naive |
    |---|---|---|---|---|---|
    | L = 30 | 0.01 | 2.95e-4 | 5.00e-4 | 6.96e-3 | 8.2e-2 |
    | | 0.005 | 7.63e-5 | 9.65e-5 | 2.14e-3 | 5.7e-2 |
    | | 0.0025 | 1.98e-5 | 2.19e-5 | 4.78e-4 | 2.2e-2 |
    | | 0.00125 | 5.42e-6 | 5.12e-6 | 1.30e-4 | 2.2e-2 |
    | | slope | **1.92** | 2.20 | 1.91 | 0.64 |
    | L = 6 | 0.01 | 1.87e-3 | 3.29e-4 | 6.45e-3 | 9.8e-2 |
    | | 0.005 | 1.14e-3 | 7.69e-5 | 1.70e-3 | 3.9e-2 |
    | | 0.0025 | 4.54e-4 | 1.84e-5 | 4.66e-4 | 6.2e-2 |
    | | 0.00125 | 1.82e-4 | 4.43e-6 | 1.21e-4 | 2.0e-2 |
    | | slope | 1.12 | **2.07** | 1.91 | 0.76 |

    In the large box (image part of F₂ negligible at contact) HLL's cv is second order and 28× more
    accurate than kdk-phys at every dt. In L = 6 the periodic-image part of F₂ at contact is ≈ 2 % of
    F₁ (the splitting only flattens the bare 1/r), which leaves cv an O(dt) residual of ≈ 1–3 % of
    naive's error; cv-phys removes it (slope 2.07, 22–27× better than kdk-phys).
  * **cv-phys drift-frame guard (new in 1.3):** a + ion at rest under F₂ = +1.25 (its − partner at r = 2
    in L = 30) grazed at τ₁ = 0.9 h by a neutral whose drift approach speed (1.4e-3 … 5.9e-3) is below
    F₂(τ₁ − h/2), once ballistic and once accelerated (second − ion at 1.2 σ): 6 speeds each, 4 steps;
    min pair distance 1 + 3.6e-15 / 1 + 8.8e-11 (hsdmd 1.2: 0.9999993, i.e. overlap, resp. no
    termination), ≤ 6 sub-steps, 5 drift fixes per set. Plus the all-ion N = 216 cv-phys dt = 0.02
    0.5 τ run that hung in 1.2: exit 0 in 0.8 s.
* (c) `--N 216 --t-equil-neutral 2 --t-run 4` (neutral): 6091 collisions, 14.10 per particle per τ
  (Enskog 14.24), Z = 5.2041 vs Z_CS = 5.2057 (−0.03 %), KE constant to 1e-14, |P| = 1.4e-14,
  min pair distance 1.0000087 (no overlap), 0.10 s.
* (d) `--N 216 --nplus 8 --nminus 8 --t-equil-neutral 2 --t-run 6` (default 15 τ charged equil),
  ewald, hsdmd 1.3 default **cv-phys**: energy drift (E_end−E0)/|E0| = +6.7e-6, max 9.8e-6 over 6 τ,
  rms 3.1e-6; T = 0.990 ± 0.008 (16 ions: the NVE offset is set by PE at the end of equilibration,
  expected scatter ±0.007); min distance 1.0000022; 48 % of ions in +/− contact pairs (4.0 pairs per
  frame), 6 % in clusters ≥ 3; stationarity OK; 36 592 Verlet sub-steps, 0 vmin / drift-fix events;
  1.28 s (equil-charged 0.80 s). `--integrator cv` (HLL): drift +2.8e-5, max 3.0e-4, T 0.999, 39 %
  paired, 1.47 s. `--integrator kdk-phys` (= the hsdmd 1.1 numbers): drift +6.0e-5, max 6.3e-5,
  T = 0.988 ± 0.007, min distance 1.0000008, 57 % paired, 22 % in clusters ≥ 3, 0.96 s. (The three
  integrators diverge chaotically within a few τ, so the pairing statistics of one 6 τ run of 16 ions
  differ by ±10 % between them.) kdk-phys with `--t-equil-charged 2`: PE −28.0 | −30.1 (halves),
  trend 20 % of |PE| = 2.5σ, T +1.8 % ⇒ `WARNING: not stationary`. dsf (`--t-run 4`), cv-phys: drift
  +4.8e-6, max 4.9e-6, 45 % paired, stationarity OK (kdk-phys: drift +2.2e-5, 55 % paired, warning on
  a 13 %/2.1σ PE trend — noise level for 16 ions over 4 τ, see above).
  kdk-phys dt scan (ewald, 2 τ equil, 4 τ): max|ΔE|/|E0| = 7.9e-5, 2.5e-5, 8.9e-6, 4.6e-6 for
  dt = 0.0083, 0.005, 0.0025, 0.00125 (naive rule: 1.05e-3, 1.06e-3, 2.1e-4, 3.8e-4).
* (e) **full-size defaults, N = 1000, η = 0.35.** Neutral (5 + 18 τ): 128 890 production collisions,
  14.32 per particle per τ (Enskog 14.24), Z = 5.2330 (+0.52 %), KE exact, min distance 1.00000009,
  sim 1.50 s (+0.6 s xyz→npz, analyze 27 s); byte-identical (md5 of xyz and diag csv) to hsdmd 1.1
  and 1.2. Charged 20+/20−, l_B = 5, ewald (5 + 15 + 18 τ, seed 1), hsdmd 1.3 default **cv-phys**:
  128 814 collisions, 14.31 per particle per τ, Z = 5.2239 (+0.35 %, collisional part), energy drift
  +2.3e-5 (max 2.4e-5, rms 7.5e-6, slope 1.4e-6/τ), T_mean = 0.9990 ± 0.0027, min distance 1.0000004,
  38 % of ions in +/− contact pairs (8.3 pairs per frame, 7.9 | 8.7 in the two halves), 10 % in
  triplets/short chains, largest cluster 3.0 (max 6), stationarity OK (PE first 2 τ vs rest +0.1σ);
  sim 9.4 s (equil-neutral 0.19 s, equil-charged 3.2 s, production 6.0 s; three idle back-to-back
  runs 9.4–9.5 s) + 0.6 s convert, analyze 32 s. Seeds 2 / 3 / 4 / 5 (cv-phys): drift +2.3e-5 /
  +1.4e-5 / +2.0e-5 / +1.7e-5, T_mean 0.9935 / 0.9977 / 1.0046 / 0.9999, PE mean −77.9 / −69.4 / −74.9 /
  −71.1 kT, 48 / 34 / 45 / 39 % paired. Seed 2 ends its 15 τ equilibration in a deep pairing
  fluctuation (PE −85 kT and 12.3 pairs in the first 2 τ of production vs −73 kT / 9.0 pairs in the
  second half) and `analyze.py` prints its stationarity WARNING (PE trend 25 % of |PE| = 3.0σ,
  T −1.3 %) although the energy is conserved to 2.3e-5 and the other three seeds pass; the same seed
  with `--t-equil-charged 25` passes (drift +2.1e-5, PE −73.2 | −77.9, 44 % paired), as does seed 2
  with kdk-phys (below) — the pairing count of 40 ions simply fluctuates by ±3 pairs on a 5–10 τ
  scale and the thresholds catch the largest excursions. The same run with `--integrator kdk-phys`
  (the hsdmd 1.1 default)
  reproduces the 1.1 numbers bit for bit: 128 660 collisions, 14.30 per particle per τ, Z = 5.2184
  (+0.24 %), drift +5.2e-5 (max 5.2e-5, rms 1.5e-5, slope 2.8e-6/τ), T_mean = 1.0031 ± 0.0025, min
  distance 1.0000001, 38 % paired (8.4 pairs per frame, 7.6 | 9.2), largest cluster 3.2 (max 6),
  sim 4.9 s; its seeds 2 / 3: T_mean 0.9999 / 0.9975, PE mean −70.7 / −70.3 kT, drift 4.9e-5 / 4.7e-5,
  38 % / 36 % paired (seed 2 re-measured with 1.3: identical). The integrator comparison is in (i).
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

* (h) **order test, production system** (`make -C sim order`: N = 1000, η = 0.35, 20+/20−, l_B = 5,
  Ewald, 5 τ neutral equilibration, charges switched on, 2 τ NVE, dt_frame 0.02; max_t |E(t) − E(0)|/N
  in kT, end drift in parentheses; `traj/hs_charges_20p20m_order.csv`; all tables below re-measured
  with hsdmd 1.3 and identical to the 1.2 numbers where those existed — the drift-frame guard never
  fired in any of these 56 production-system runs):

  | dt | cv-phys (default) | cv (HLL) | kdk-phys | naive |
  |---|---|---|---|---|
  | 0.02 | 2.69e-5 (+2.5e-5) | 2.82e-4 (−6.9e-5) | 3.41e-5 (+9.0e-6) | 1.90e-3 (+1.9e-3) |
  | 0.01 | 6.98e-6 (+5.7e-6) | 1.84e-4 (+2.2e-5) | 2.24e-5 (+2.1e-5) | 1.80e-4 (−1.1e-4) |
  | 0.005 | 2.92e-6 (+2.0e-6) | 6.97e-5 (+1.9e-5) | 7.46e-6 (+7.5e-6) | 2.21e-4 (+8.4e-5) |
  | 0.0025 | 1.43e-6 (−1.1e-6) | 6.48e-5 (−1.9e-5) | 1.00e-6 (+7.1e-7) | 2.29e-4 (+1.0e-4) |
  | slope | 1.40 | 0.78 | 1.69 | 0.89 |

  naive is first order as in HLL. HLL's cv is **not** second order on this system (8× … 65× worse than
  kdk-phys, see the `cv-phys` bullet above for the mechanism). cv-phys is 10–40× better than cv and
  3–5× better than kdk-phys at dt ≥ 0.005; at dt = 0.0025 all three good integrators reach a floor of
  ≈ 1e-6 kT/N (≈ 1e-3 kT absolute) that is set by the Ewald real-space truncation at r_c = L/2
  (erfc(2.8) ≈ 7.5e-5 → a 6.5e-5 kT energy jump every time an ion pair crosses r_c), not by the
  integrators; the same floor is what limits cv-phys's slope here to 1.40.
  * `--coulomb dsf` (energy and force continuous at r_c, no floor):

    | dt | cv-phys | cv (HLL) | kdk-phys | naive |
    |---|---|---|---|---|
    | 0.02 | 2.71e-5 | 2.69e-4 | 7.29e-5 | 1.81e-3 |
    | 0.01 | 5.98e-6 | 1.39e-4 | 9.55e-6 | 2.95e-4 |
    | 0.005 | 2.52e-6 | 4.64e-5 | 5.86e-6 | 4.71e-4 |
    | 0.0025 | 4.23e-7 | 2.37e-5 | 1.78e-6 | 1.21e-4 |
    | slope | **1.92** | 1.21 | 1.68 | 1.11 |

  * `--t-equil-charged 15` (Ewald; the equilibrated, paired state of the movies, each dt with its own
    start):

    | dt | cv-phys | cv (HLL) | kdk-phys | naive |
    |---|---|---|---|---|
    | 0.02 | 4.75e-5 | 3.81e-4 | 6.74e-5 | 2.38e-3 |
    | 0.01 | 6.83e-6 | 1.15e-4 | 3.43e-5 | 9.16e-4 |
    | 0.005 | 2.87e-6 | 1.00e-4 | 5.98e-6 | 3.89e-4 |
    | 0.0025 | 5.97e-7 | 7.47e-5 | 1.82e-6 | 2.53e-4 |
    | slope | **2.02** | 0.73 | 1.82 | 1.09 |

  * **all-ion system, no neutrals** (`--N 216 --nplus 108 --nminus 108`, Ewald, same protocol; the
    cv-phys dt = 0.02 run is the one that hung in 1.2 — it now finishes in 6 s with 132 drift fixes,
    131 physically-separating no-ops, min distance 1.000000000):

    | dt | cv-phys | cv (HLL) | kdk-phys | naive |
    |---|---|---|---|---|
    | 0.02 | 6.76e-3 | 1.04e-1 | 1.17e-2 | 2.21e-1 |
    | 0.01 | 1.88e-3 | 2.12e-2 | 1.85e-3 | 3.19e-2 |
    | 0.005 | 2.89e-4 | 6.64e-3 | 1.79e-3 | 3.26e-2 |
    | 0.0025 | 1.79e-4 | 4.72e-3 | 5.71e-4 | 4.74e-3 |
    | slope | 1.84 | 1.51 | 1.31 | 1.66 |

    With zero neutrals HLL's cv is still 4–11× worse than kdk-phys and only 2–7× better than naive:
    the +/− contact pairs collide in the (additive) third-body field, mechanism (ii) above. The error
    tracks |F₂| at the collisions, not the neutral count: a larger `--cv-q2` moves more of the ion–ion
    force into F₁ and shrinks cv's error without changing its order — q₂ = 3.0: 2.21e-4, 1.04e-4,
    7.42e-5, 3.47e-5 (slope 0.85); q₂ = 5.5 ≈ L/2: 9.22e-5, 4.19e-5, 2.42e-5, 1.48e-5 (0.87). (cv-phys
    with q₂ = 3.0 sits at the Ewald floor, 6.2e-6 … 1.7e-6, for every dt.)
* (i) **full-size charged production with hsdmd 1.3** (`run_sim.py`, N = 1000, 20+/20−, l_B = 5, ewald,
  5 + 15 + 18 τ, seed 1, dt = 0.005; `analyze.py` exit 0 for all; the cv-phys and cv trajectories are
  identical to the hsdmd 1.2 ones — the 1.3 guard never fired in them):

  | integrator | drift (E_end−E₀)/|E₀| | max |δE|/|E₀| | rms | T_mean | paired | coll/part/τ | Z | wall sim |
  |---|---|---|---|---|---|---|---|---|
  | cv-phys (default) | +2.32e-5 | 2.36e-5 | 7.5e-6 | 0.9990 ± 0.0027 | 38 % (8.3 pairs) | 14.31 | 5.2239 (+0.35 %) | 9.4 s |
  | cv (HLL) | −1.30e-4 | 1.62e-4 | 7.2e-5 | 1.0025 ± 0.0024 | 38 % (8.2 pairs) | 14.34 | 5.2286 (+0.44 %) | 9.6 s |
  | kdk-phys (1.1 default) | +5.23e-5 | 5.23e-5 | 1.5e-5 | 1.0031 ± 0.0025 | 38 % (8.4 pairs) | 14.30 | 5.2184 (+0.24 %) | 4.9 s |

  cv-phys: 243 150 Verlet sub-steps, 1.0e8 quartic candidates (the json key `n_quartic_solves` counts
  `quartic_collision_time()` candidates, ≈ 99 % of which the reach pre-filter rejects before any root
  solve), 0 `vmin`, 0 drift-fix and 0 physically-separating events, max contact error 1.9e-14, min
  pair distance 1.000000381 over all 721 frames (no overlap), |P| = 2e-13, stationarity OK (PE first
  2 τ vs rest +0.1σ), 10 % of ions in clusters ≥ 3; cv: 242 892 sub-steps, contact error 2.3e-14, min
  distance 1.000000090, +0.3σ, 8 % in clusters ≥ 3, and a frame-to-frame energy scatter 28× that of
  kdk-phys (the ∝ dt random walk of (h)); kdk-phys reproduces the hsdmd 1.1 numbers bit for bit
  (128 660 collisions, drift +5.2e-5). Wall times are for the simulator alone (xyz→npz +0.6 s,
  analyze 28–32 s); the 5 + 15 + 18 τ cv-phys run is ≈ 2× kdk-phys and far below the 3-minute budget.
  **Neutral run unchanged:** `hs_neutral` from hsdmd 1.3 is byte-identical (md5 of the xyz and of the
  diag csv) to the hsdmd 1.1 and 1.2 binaries for the same seed; KE constant to 4e-14.
  **Determinism:** the full charged cv-phys run repeated with the same seed gives a byte-identical xyz.

## Limitations / notes
* **Which integrator.** The default `cv-phys` (HLL + F₂ physical-velocity rule) is the only variant
  that meets the spec's order criterion on this system and is the most accurate choice at every dt
  (production drift +2.3e-5 vs +5.2e-5 kdk-phys vs −1.3e-4 cv); kdk-phys is 2× cheaper. `cv` is HLL
  2000 exactly but first order in an ion/neutral mixture (spec §4a): 2–3× *worse* than kdk-phys and
  only ≈ 3× better than naive, with a systematic negative drift (7 seeds: −2e-6 … −1.6e-4, mean
  −9e-5), bit-exact hard cores nevertheless. A larger `--cv-q2` (2–3 σ) halves cv's error but leaves
  it O(dt). The hsdmd 1.2 cv-phys livelock/overlap at grazing collisions under F₂ is fixed (drift-frame
  guard + watchdog, see the algorithm section); the guard is a deviation from a literal infinite-precision
  EDMD in the same sense as kdk-phys's `n_drift_fix` and `cv_vmin`.
* **CV cost.** Every system-wide collision ends a Verlet sub-step: F₁ (780 ion pairs) is recomputed and
  the ≈ 10–20 accelerated ions are re-predicted (27 cells, ≈ 400 quartic candidates per sub-step, 99 %
  rejected by the reach bound). 243 k sub-steps and 1.0e8 quartic candidates for the 5 + 15 + 18 τ run
  cost 4.5 s on top of kdk-phys (9.4 s vs 4.9 s). The resting-contact guard (`cv_vmin`, 0 events in all
  runs) bounds the sub-step rate of a pressed pair to a_n/(2·2e-5) ≈ 5e5 per τ in the worst case.
* The top-level `run_all.sh` passes its own `T_EQ_C` (default 5 there) to `--t-equil-charged`; set
  `T_EQ_C=15` (or change its default) to get the equilibrated start documented above.
* Z_virial for charged runs contains only the hard-core (collisional) virial, not the Coulomb virial.
* T_mean of a charged NVE production deviates from 1 by (PE_end-of-equil − ⟨PE⟩)/(1.5 N kT), i.e. by
  ±0.3 % at N = 1000 and ±1 % at N = 216 even when fully equilibrated; only a *trend* indicates a
  problem.
* Exit codes of the binary: 0 ok, 2 invalid arguments / box too small / cannot open output, 3 internal
  consistency error (clock would run backwards, or the event-loop watchdog: > 10⁴ consecutive events at
  one and the same time — the symptom of the hsdmd 1.2 cv-phys livelock, which no longer occurs).
* Ewald real-space cutoff r_c = L/2 with α = 5.6/L truncates at erfc(2.8) ≈ 7.5e-5 (viz quality).
* `dt` is rounded so dt_frame/dt is an integer (reported as `dt` in the json; the request as `dt_requested`).
* ncell = floor(L/σ) must be ≥ 3 (N ≥ ~30 at η = 0.35); the program aborts otherwise.
