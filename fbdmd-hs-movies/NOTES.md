# NOTES — hard-sphere + point-charge trajectory movies

Generated 2026-10-25 13:03 on vm (Linux x86_64, 4 CPUs, Intel(R) Xeon(R) Processor @ 2.10GHz).

Units: σ = 1, m = 1, kT = 1, τ = σ√(m/kT). Charges ±e on a subset of spheres; Coulomb coupling given as l_B/σ.


## Runs

| run | N | η | L/σ | kT | +/− | l_B/σ | Coulomb | integrator | Δt | frames (Δt_frame) | t_run | collisions | Z (virial) / Z_CS | ΔE/E (NVE) | sim wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hs_charges_20p20m | 1000 | 0.35 | 11.437 | 1.0 | 20/20 | 5 | ewald | collision Verlet (HLL 2000) + F2 physical-velocity collision rule, q1=1, q2=1.5 | 0.005 | 721 (0.025) | 18 | 128814 | 5.224 / 5.206 | 2.3e-05 | 10 s |
| hs_neutral | 1000 | 0.35 | 11.437 | 1.0 | 0/0 | — | — | EDMD | event-driven | 721 (0.025) | 18 | 128890 | 5.233 / 5.206 | exact (KE const) | 2 s |

Equilibration (velocity rescaling to kT = 1 every 0.1 τ, then NVE production): hs_charges_20p20m: 5 τ neutral + 15 τ with charges on; hs_neutral: 5 τ neutral. Same seed ⇒ identical neutral start for both runs.

Charged run details: Coulomb = ewald (Ewald α = 5.6/L, |n|∞ ≤ 6 for 40 ions), integrator = collision Verlet (HLL 2000) + F2 physical-velocity collision rule, q1=1, q2=1.5, 243150 Verlet sub-steps, T_mean = 0.9990, max |δE|/|E| = 2.4e-05, min pair distance 1.000000381 σ (no overlap).


## Integrator energy-order test (max |E(t)−E₀|/N in kT over 2 τ, equilibrated start)

```
  dt                       cv-phys                     cv               kdk-phys                  naive
  0.02       4.749e-05 (+4.5e-05)  3.811e-04 (-1.5e-04)  6.737e-05 (+4.7e-05)  2.376e-03 (+2.4e-03)
  0.01       6.826e-06 (+6.8e-06)  1.151e-04 (+7.2e-05)  3.431e-05 (+3.4e-05)  9.157e-04 (-8.0e-04)
  0.005      2.865e-06 (+2.0e-06)  1.001e-04 (+9.9e-05)  5.979e-06 (+4.9e-06)  3.891e-04 (+2.7e-04)
  0.0025     5.973e-07 (-7.6e-08)  7.466e-05 (-7.5e-05)  1.819e-06 (+6.3e-07)  2.531e-04 (-1.9e-04)
  slope           2.02 (   +2.95)       0.73 (   +0.24)       1.82 (   +2.15)       1.09 (   +1.25)    (log-log fit of max |dE|/N, end drift in parentheses)
```
cv-phys = shipped default (HLL collision Verlet + physical-velocity rule for F₂ at collisions); cv = literal HLL; kdk-phys = kick–drift–kick with the same rule; naive = HLL eq. 9. Slopes ≈2 / ≈0.7 / ≈1.8 / ≈1.

## Movies

* `/home/user/quantum-trader/fbdmd-hs-movies/movies/hs_charges_20p20m.mp4` — h264 1280x720 @ 30/1 fps, 24.0 s, 12.4 MB
* `/home/user/quantum-trader/fbdmd-hs-movies/movies/hs_charges_20p20m_xray.mp4` — h264 1280x720 @ 30/1 fps, 24.0 s, 15.6 MB
* `/home/user/quantum-trader/fbdmd-hs-movies/movies/hs_neutral.mp4` — h264 1280x720 @ 30/1 fps, 24.0 s, 12.0 MB

## Render timing

```
hs_neutral: 1028 s wall for render+encode (ovito, final, 1280x720)   [721 frames at 1.40 s/frame; the pipeline log shows 19 s because the frames were already rendered and only re-encoded]
hs_charges_20p20m: 1031 s wall for render+encode (ovito, final, 1280x720)
hs_charges_20p20m_xray: 896 s wall for render+encode (ovito, final, 1280x720)
Simulation wall: hs_neutral 1.7 s (5+18 tau EDMD), hs_charges_20p20m 10 s (5+15+18 tau, collision Verlet dt=0.005); analysis 28 s per run; order test 60 s.
Machine: Linux x86_64 cloud VM, 4 x Intel Xeon 2.10 GHz, no GPU (Tachyon CPU ray tracing with OVITO_THREAD_COUNT=16).
```

## What the charged trajectory shows (20+/20−, l_B = 5 σ, η = 0.35)

Numbers from `notes/analysis_hs_charges_20p20m.txt` (721 frames, 18 τ, NVE, collision Verlet cv-phys):

1. **Partially paired electrolyte, not a plasma and not a salt crystal.** At any instant ≈38 % of the ions
   (min 10 %, max 65 %) sit in a +/− contact pair (r < 1.2 σ): on average 8.3 pairs per frame out of 20 possible.
   The remaining ≈62 % move as free ions between collisions with the gray solvent spheres.
2. **Pairs form and break on the movie's time scale.** The pair count fluctuates between 2 and 15 within the
   18 τ. A contact episode (r < 1.2 σ) lasts 0.5 τ on average (median 0.35 τ, 90th percentile 1.2 τ, longest 2.5 τ),
   i.e. ≈0.7 s of movie typically and up to ≈3 s: partners meet, rattle against each other in contact, get knocked
   apart by neutrals and re-pair; each + ion has 2.4 different − partners (1–6) over the 18 τ. With the looser
   Bjerrum criterion (r < l_B/2 = 2.5 σ) about three quarters of the ions count as associated at any time.
3. **Short chains, no branched clusters.** ≈10 % of the ions are in clusters of three or more at any instant
   (triplets + − + or − + −, occasionally four to six ions), almost always linear chains (max degree 2);
   branched clusters appear in 1 % of the frames. The largest cluster seen is 6 ions.
4. **Everything else is hard-sphere liquid dynamics.** Each sphere collides ≈14 times per τ (Enskog 14.24),
   so a sphere moves only ≈0.1 σ between collisions and 0.038 σ per frame on average (max 0.14 σ): the motion
   reads as dense rattling with slow diffusion. Over the 18 τ a neutral sphere travels 3.7 σ rms, an ion 3.5 σ
   (slightly slower, dragged by its partners). Nothing tunnels: the minimum pair distance over all
   721 000 sphere positions is 1.0000004 σ.
5. **Integrator sanity.** Total energy is conserved to 2.3×10⁻⁵ (relative) over 18 τ, temperature 0.999 ± 0.003,
   no trend in potential energy or pair count (stationary). The literal HLL collision Verlet gives the same
   thermodynamic state but a 6× larger energy drift (first order on this ion-in-neutral-solvent system, see README).

The opaque CPK movie (`hs_charges_20p20m.mp4`) shows the ions that happen to be in the outer ~1.5 σ of the box
(≈40 % of them); the x-ray variant (`hs_charges_20p20m_xray.mp4`) ghosts the neutrals so all 40 ions and their
pairing are visible inside the box.

