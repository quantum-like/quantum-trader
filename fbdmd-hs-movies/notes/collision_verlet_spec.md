# Collision Verlet (CV) for hard spheres + Coulomb — implementation spec

Source: Y. A. Houndonougbo, B. B. Laird, B. J. Leimkuhler, "A molecular dynamics algorithm for mixed
hard-core/continuous potentials", Mol. Phys. 98, 309–316 (2000). Equation/table numbers below refer to it.
PDF (read-only, for reference): /root/.claude/uploads/51ecfd3f-2c9f-5b9b-ba8a-5c6a2ddea84a/90937701-Houndonougbo_Laird_Leimkuhler_2000_Collision_Verlet_MolPhys98.pdf
(pages 3–7 of the PDF = journal pages 310–314 contain everything needed; Table 1 is on journal page 314.)

## Why not the "naive" splitting
Naive splitting (eq. 9; ≈ Suh et al.): kick(h/2, full continuous force) → exact hard-sphere flow over h
(all collisions) → kick(h/2). Section 2 shows that when the continuous pair force is NON-ZERO at contact
(Coulomb: F(σ) = l_B/σ² ≠ 0) every step containing a collision introduces a LOCAL energy error O(h)
(eq. 18), so the global error is O(h) instead of O(h²) and the energy drifts (Fig. 3, dashed).
Third order is recovered only if (case 1) collisions happen exactly at step boundaries or (case 3) the
continuous force vanishes at contact. CV engineers both: it splits the potential so the part treated by
the outer kicks is flat at contact, and treats the rest with Verlet sub-steps that begin/end at collisions.

## 1. Potential splitting (eqs. 19–25)
Pair potential φ_c(q) = l_B q_i q_j / q (q = pair distance; l_B = bjerrum; q_i = ±1). Parameters σ ≤ q₁ < q₂.
Hermite cubic P(q) = A₀ + A₁q + A₂q² + A₃q³ with P(q₂) = φ_c(q₂), P′(q₂) = φ_c′(q₂), P′(q₁) = 0, P″(q₁) = 0:
    A₃ = φ_c′(q₂) / (3 (q₂ − q₁)²)          [= eq. 22 after simplification]
    A₂ = −3 q₁ A₃,  A₁ = 3 q₁² A₃,  A₀ = φ_c(q₂) − (A₁q₂ + A₂q₂² + A₃q₂³)
  equivalently  P(q)  = φ_c(q₂) + φ_c′(q₂) [ (q−q₁)³ − (q₂−q₁)³ ] / (3 (q₂−q₁)²)
                P′(q) = φ_c′(q₂) (q−q₁)² / (q₂−q₁)²
Long-range part (eq. 20):  φ₂(q) = P(q₁)  (q < q₁);  P(q)  (q₁ ≤ q < q₂);  φ_c(q)  (q ≥ q₂)
Short-range part (eq. 21): φ₁(q) = φ_c(q) − P(q₁) (q < q₁);  φ_c(q) − P(q) (q₁ ≤ q < q₂);  0 (q ≥ q₂)
Properties to assert in a self-test: φ₁ + φ₂ = φ_c everywhere; φ₂′ = 0 and φ₂″ = 0 at q₁ (and below);
φ₁(q₂) = φ₁′(q₂) = 0 (continuous force); F₁ = −φ₁′ is non-zero only for q < q₂.
Defaults: q₁ = 1.0 σ, q₂ = 1.5 σ (Fig. 4b used q₁ = 1.000, q₂ = 1.5–1.7 and showed clean 2nd order).
With Coulomb: φ_c′(q₂) = −l_B q_i q_j / q₂².

Forces: F_total = Ewald (or DSF) force on each ion (as already implemented).
        F₁,i = Σ_{j ion, r_ij<q₂} (−φ₁′(r_ij)) r̂_ij   (minimum image; only ion pairs; O(n_ion²) trivial)
        F₂,i = F_total,i − F₁,i   (this is the force of H₂ = V₂; identical to applying φ₂ pairwise for the
                                   minimum-image pair plus all the Ewald long-range contributions)

## 2. Outer splitting (eqs. 26–28, Table 1)
H₁ = T + V_hs + V₁ (hard core + short-range Coulomb),  H₂ = V₂.
One step of length h:
    p += (h/2) F₂(q)                      # outer half kick — ions only (neutrals have no force)
    propagate H₁ for time h (Section 3)   # hard-sphere collisions + Verlet sub-steps with F₁
    p += (h/2) F₂(q)                      # outer half kick with F₂ at the new positions
Compute F_total and F₁ once per step boundary and reuse for the second half-kick of step n and the
first of step n+1.

## 3. Propagation of H₁ = T + V_hs + V₁ (Table 1, eqs. 28–29)
Between consecutive hard collisions (anywhere in the system) all particles take ONE velocity-Verlet
sub-step with the short-range force F₁, of length τ equal to the time to the next collision
(τ_{n_c+1} = remaining time to the end of the step h). For sub-step k with duration τ:
    p_half = p + (τ/2) F₁(q)              # F₁ evaluated at the sub-step start positions
    q_new  = q + (τ/m) p_half             # ⇒ q(τ') = q + v τ' + ½ a τ'² with a = F₁(q)/m  (quadratic path)
    p_tilde = p_half + (τ/2) F₁(q_new)     # F₁ at the new positions
    (if this sub-step ended on a collision) p = ψ_Vhs(q_new, p_tilde): elastic collision of the pair,
       applied to p_tilde (i.e. AFTER the second half-kick), using the contact normal at q_new.
Collision time τ (eq. 29): smallest τ > 0 with |Δq + Δv τ + ½ Δa τ²|² = σ² where Δ = i − j and the
velocities used are the half-kicked ones, i.e. v = p/m + (τ/2m) F₁ ... NOTE: because p_half depends on τ,
write the path as q(τ') = q + (p/m) τ' + ½ (F₁/m) τ'²  (identical algebra) — the quartic is in τ' and does not
depend on the (unknown) sub-step length. For pairs with Δa = 0 (both neutrals, or ions with no F₁) the
quartic reduces to the usual EDMD quadratic. For Δa ≠ 0 solve the quartic
    c₄ τ⁴ + c₃ τ³ + c₂ τ² + c₁ τ + c₀ = 0,
    c₄ = ¼|Δa|², c₃ = Δv·Δa, c₂ = |Δv|² + Δq·Δa, c₁ = 2 Δq·Δv, c₀ = |Δq|² − σ²
robustly (Laguerre/companion or bracketing + Newton; the paper used Laguerre + Newton refinement). Take the
SMALLEST positive real root where the distance crosses σ from above (d/dτ of the distance² negative at the root).
Note: with attraction (opposite charges) a pair that is currently separating (Δq·Δv > 0) can still reach
contact inside the sub-step — do NOT use the EDMD "b < 0" shortcut for pairs with Δa ≠ 0.
Grazing/roundoff: if |Δq|² ≤ σ²(1+1e-12) and the pair is approaching at τ = 0, collide immediately.

Event-driven realisation (recommended; equivalent to Table 1):
  * Particles with F₁ = 0 (all neutrals and most ions) keep lazy ballistic positions; their predicted events
    remain valid across sub-steps because the sub-step kicks are no-ops for them.
  * Keep a small set A of "accelerated" particles (ions with F₁ ≠ 0, i.e. with another ion inside q₂). For
    i ∈ A store (x_i, v_i, a_i, t_i) and use the quadratic path for position, cell-crossing prediction
    (per-axis quadratic) and collision prediction (quartic against any neighbour; quadratic when Δa = 0).
  * At EVERY processed hard collision (any pair, at time t_c) and at the end of the step:
      1. advance all i ∈ A to t_c on their quadratic paths;  (p_half + drift of Table 1)
      2. recompute F₁ at the positions at t_c (ions only, O(n_ion²));
      3. v_i ← v_i + ½ τ (a_i,old + a_i,new),  a_i ← a_i,new  for i ∈ A ∪ {ions that just entered A}
         (this is the two half-kicks of Table 1 merged across the sub-step boundary);
      4. resolve the elastic collision for the colliding pair with the updated velocities;
      5. bump the event counters of every particle whose velocity or acceleration changed, reschedule their
         cell crossings and re-predict their collisions (27-cell neighbourhood).  Only A ∪ colliding pair.
    Because τ between system-wide collisions (≈ 1.4e-4 τ at N = 1000, η = 0.35) ≪ h, the inner Verlet is
    very finely resolved; the scheme remains exactly Table 1 for any N.
  * Ions leave A when F₁ becomes exactly 0 (no ion within q₂); F₁ is continuous at q₂ so entering/leaving is smooth.
  * Thermostat (equilibration only) rescales v after a completed step; never inside the H₁ propagation.

## 4. Diagnostics / validation required
  * Total energy H = KE + U_coulomb(Ewald, full) must be conserved by the production (NVE) run: report
    relative drift and the maximum instantaneous |δE| / N.
  * Order test on the production system (N = 1000, η = 0.35, 20+/20−, l_B = 5, same start): run t = 2 τ at
    h = 0.02, 0.01, 0.005, 0.0025 for BOTH --integrator cv and --integrator naive; max |E(t) − E(0)| versus h
    must have slope ≈ 2 for CV and ≈ 1 for naive (log–log). Write the table to traj/<name>_order.csv.
  * Momentum conservation, no overlaps (analyze.py), temperature ≈ 1, collision count ≈ Enskog.
  * Keep the naive splitting selectable (--integrator naive) for the comparison; CV is the default.
