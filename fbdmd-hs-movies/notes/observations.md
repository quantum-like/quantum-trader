## What the charged trajectory shows (20+/20−, l_B = 5 σ, η = 0.35)

Numbers from `notes/analysis_hs_charges_20p20m.txt` (721 frames, 18 τ, NVE, collision Verlet cv-phys); the l_B/2
figures in item 2 were recomputed from the float64 `traj/hs_charges_20p20m.xyz` (`sim/analyze.py` now prints them as its
"looser Bjerrum criterion" line):

1. **Partially paired electrolyte, not a plasma and not a salt crystal.** At any instant 38 ± 3 % of the ions
   (min 10 %, max 65 %; ±3 % is the statistical uncertainty of one 18 τ run) sit in a +/− contact pair (r < 1.2 σ): on average 8.3 pairs per frame out of 20 possible.
   The remaining ≈62 % move as free ions between collisions with the gray solvent spheres.
2. **Pairs form and break on the movie's time scale.** The pair count fluctuates between 2 and 15 within the
   18 τ. A contact episode (r < 1.2 σ) lasts 0.5 τ on average (median 0.35 τ, 90th percentile 1.2 τ, longest 2.5 τ),
   i.e. ≈0.7 s of movie typically and up to ≈3.3 s: partners meet, rattle against each other in contact, get knocked
   apart by neutrals and re-pair; each + ion has 2.4 different − partners (1–6) over the 18 τ. With the looser
   Bjerrum criterion (r < l_B/2 = 2.5 σ) 86 % of the ions (min 75 %, max 93 %) have an opposite ion within range at
   any instant (1.5 opposite ions each on average) — but random 20+20 subsets of the *neutral* run already give
   58 ± 7 % (ideal-gas estimate 57 %): at this ion density (mean spacing 3.3 σ, κσ = 1.3) the 2.5 σ count is mostly
   hard-sphere packing, so the contact criterion is the meaningful one.
3. **Short chains, no branched clusters.** ≈10 % of the ions are in clusters of three or more at any instant
   (triplets + − + or − + −, occasionally four to six ions), almost always linear chains (max degree 2);
   branched clusters appear in 1 % of the frames. The largest cluster seen is 6 ions.
4. **Everything else is hard-sphere liquid dynamics.** Each sphere collides ≈14 times per τ (Enskog 14.24),
   so a sphere moves only ≈0.1 σ between collisions and 0.038 σ per frame on average (max 0.14 σ): the motion
   reads as dense rattling with slow diffusion. Over the 18 τ a neutral sphere travels 3.7 σ rms, an ion 3.5 σ
   (the same within the statistics of 40 ions; ions and neutrals also share the same kinetic temperature to within ~3 %). Nothing tunnels: the minimum pair distance over all
   721 000 sphere positions is 1.0000004 σ.
5. **Integrator sanity.** Total energy is conserved to 2.3×10⁻⁵ (relative) over 18 τ, temperature 0.999 ± 0.003,
   no trend in potential energy or pair count (stationary). The literal HLL collision Verlet gives the same
   thermodynamic state but a 6× larger energy drift (first order on this ion-in-neutral-solvent system, see README).

The opaque CPK movie (`hs_charges_20p20m.mp4`) shows only the ions within ~1.5 σ of the three camera-facing faces
(about a third of them, typically 5–13 of the 40 at any moment); the x-ray variant (`hs_charges_20p20m_xray.mp4`) ghosts the neutrals so all 40 ions and their
pairing are visible inside the box.
